// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Filter 18: resolve MCP tool declarations into concrete tool
//! definitions by calling upstream MCP servers' `tools/list`.
//!
//! Runs after `openai_tool_parse`, gated on `openai_tool_parse.has_mcp`.
//! Reads MCP entries from the buffered request body, checks
//! `previous_tools` for cached listings, calls `tools/list` via
//! the `mcp_client` module, writes `mcp_tool_map` to
//! [`ResponsesState`], and rewrites `type: "mcp"` entries in the
//! request body to `type: "function"`.
//!
//! # Function name encoding
//!
//! Each rewritten function tool is named
//! `{server_label}__{tool_name}`. The prefix is required because
//! the backend does not know about MCP servers: without it, two
//! servers exposing a tool with the same name (e.g. `search`)
//! would produce duplicate `type: "function"` entries, and the
//! proxy would have no way to dispatch tool-call responses back
//! to the correct server. Names are sanitized to match the
//! OpenAI schema (`^[a-zA-Z0-9_-]+$`, max 64 chars) and
//! truncated when necessary; because truncation is lossy,
//! [`detect_name_collisions`] runs after building the full tools
//! array to reject ambiguous results.
//!
//! [`ResponsesState`]: super::state::ResponsesState

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;

use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::debug;

use self::config::{McpToolResolveConfig, build_config};
use super::{error::responses_error_rejection, state::ResponsesState};
use crate::{
    json_body::{SerializedJson, serialize_json_body},
    mcp_client,
};

/// Maximum length for generated function names per the OpenAI
/// Responses POST schema (`^[a-zA-Z0-9_-]+$`, max 64 chars).
const MAX_FUNCTION_NAME_LEN: usize = 64;

// -----------------------------------------------------------------------------
// McpToolResolveFilter
// -----------------------------------------------------------------------------

/// Resolves MCP tool entries from the Responses API `tools` array
/// into concrete tool definitions by calling `tools/list` on each
/// upstream MCP server.
///
/// Rejects the request with HTTP 400 before any callouts if two or
/// more resolvable MCP entries share the same `server_label`
/// (including entries that differ only by credentials).
///
/// # YAML
///
/// ```yaml
/// filter: openai_mcp_tool_resolve
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_mcp_tool_resolve
/// timeout_ms: 5000
/// max_body_bytes: 67108864
/// max_tools: 128
/// ```
pub struct McpToolResolveFilter {
    /// Allow connections to loopback addresses.
    allow_loopback: bool,

    /// Connector ID to server URL mapping.
    connectors: HashMap<String, url::Url>,

    /// Maximum request body bytes for `StreamBuffer`.
    max_body_bytes: usize,

    /// Maximum number of distinct MCP servers per request.
    max_servers: usize,

    /// Maximum number of tools returned by a single MCP server.
    max_tools: usize,

    /// Per-server timeout for `tools/list` calls.
    timeout: Duration,
}

impl McpToolResolveFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpToolResolveConfig = parse_filter_config("openai_mcp_tool_resolve", config)?;
        let validated = build_config(cfg)?;
        let connectors = validated
            .connectors
            .iter()
            .map(|c| {
                let url = url::Url::parse(&c.server_url)
                    .map_err(|e| FilterError::from(format!("openai_mcp_tool_resolve: connector \"{}\": {e}", c.id)))?;
                Ok((c.id.clone(), url))
            })
            .collect::<Result<HashMap<String, url::Url>, FilterError>>()?;
        Ok(Box::new(Self {
            allow_loopback: validated.allow_loopback,
            connectors,
            max_body_bytes: validated.max_body_bytes,
            max_servers: validated.max_servers,
            max_tools: validated.max_tools,
            timeout: Duration::from_millis(validated.timeout_ms),
        }))
    }

    /// Core resolution: parse MCP entries, check cache, call
    /// `tools/list`, build `mcp_tool_map`, rewrite the request
    /// body to replace `type: "mcp"` entries with `type: "function"`,
    /// and synchronize the rewritten body into `ResponsesState`.
    async fn resolve_mcp_tools(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        original_bytes: Bytes,
    ) -> Result<FilterAction, ResolveError> {
        let mut mcp_entries = extract_mcp_entries(&original_bytes);
        if mcp_entries.is_empty() {
            return Ok(FilterAction::Continue);
        }

        resolve_connector_ids(&self.connectors, &mut mcp_entries)?;
        self.validate_entries(&mcp_entries)?;

        let previous_tools = ctx.extensions.get::<ResponsesState>().map(|s| &s.previous_tools);

        let resolution = self.resolve_all_entries(&mcp_entries, previous_tools).await?;

        if !resolution.has_resolved {
            return Ok(FilterAction::Continue);
        }

        debug!(tool_count = resolution.tool_map.len(), "mcp_tool_map built");

        let Resolution {
            per_entry,
            tool_map,
            resolved_labels,
            ..
        } = resolution;
        let Some(serialized) = rewrite_request_body(&original_bytes, per_entry, &tool_map, &resolved_labels)? else {
            return Ok(FilterAction::Continue);
        };
        check_body_size(&serialized, self.max_body_bytes)?;
        serialized.commit(body, self.name(), "tools");

        let body_for_state = body.as_ref().map_or_else(|| original_bytes.as_ref(), |b| b.as_ref());
        write_state(ctx, body_for_state, tool_map);
        Ok(FilterAction::Continue)
    }

    /// Validate MCP entries: check server count and duplicate labels.
    fn validate_entries(&self, entries: &[serde_json::Value]) -> Result<(), ResolveError> {
        let server_count = count_distinct_servers(entries);
        if server_count > self.max_servers {
            return Err(ResolveError::TooManyServers {
                count: server_count,
                max: self.max_servers,
            });
        }
        check_duplicate_labels(entries)
    }

    /// Resolve all MCP entries, building both the global dispatch
    /// map and pre-built function tools for body rewriting.
    ///
    /// Server resolutions run concurrently via
    /// [`futures::future::try_join_all`]. Non-credentialed entries
    /// sharing the same `(server_label, server_url)` are
    /// deduplicated to a single resolution task; credentialed
    /// entries always resolve independently.
    async fn resolve_all_entries(
        &self,
        entries: &[serde_json::Value],
        previous_tools: Option<&Vec<serde_json::Value>>,
    ) -> Result<Resolution, ResolveError> {
        let (entry_to_task, task_entries, task_allowed_names) = dedup_entries(entries);

        let futures: Vec<_> = task_entries
            .iter()
            .zip(&task_allowed_names)
            .map(|(entry, allowed)| async {
                let result = self.resolve_entry(entry, previous_tools, allowed.as_deref()).await;
                redact_connector_client_error(result, entry)
            })
            .collect();
        let task_results = futures::future::try_join_all(futures).await?;

        Ok(collect_resolutions(entries, &entry_to_task, &task_results))
    }

    /// Resolve tools for a single MCP entry independently.
    ///
    /// `cache_allowed_names` is the union of `allowed_tools` names
    /// across all entries sharing this resolution task; it is passed
    /// to [`find_cached_listing`] so the cache is only used when it
    /// covers every entry in the group.
    async fn resolve_entry(
        &self,
        entry: &serde_json::Value,
        previous_tools: Option<&Vec<serde_json::Value>>,
        cache_allowed_names: Option<&[String]>,
    ) -> Result<Option<Vec<serde_json::Value>>, ResolveError> {
        let Some(server_url) = resolvable_server_url(entry) else {
            return Ok(None);
        };
        let label = server_label(entry);
        let is_connector = entry.get("connector_id").is_some();
        mcp_client::validate_mcp_url(server_url, self.timeout, self.allow_loopback)
            .await
            .map_err(ResolveError::Client)?;
        if !has_entry_credentials(entry)
            && let Some(cached) =
                find_cached_listing(previous_tools, label, server_url, cache_allowed_names, is_connector)
        {
            debug!(label, tool_count = cached.len(), "reusing cached MCP tool listing");
            return Ok(Some(cached));
        }
        let tools = fetch_tools(entry, server_url, self.timeout, self.max_tools, self.allow_loopback).await?;
        Ok(Some(tools))
    }
}

#[async_trait]
impl HttpFilter for McpToolResolveFilter {
    fn name(&self) -> &'static str {
        "openai_mcp_tool_resolve"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if !has_mcp_tools(ctx) {
            return Ok(FilterAction::Continue);
        }

        let Some(bytes) = body.as_ref().cloned() else {
            return Ok(FilterAction::Continue);
        };

        let streaming = is_streaming(ctx);

        match Box::pin(self.resolve_mcp_tools(ctx, body, bytes)).await {
            Ok(action) => Ok(action),
            Err(e) => Ok(resolve_error_rejection(&e, streaming)),
        }
    }
}

// -----------------------------------------------------------------------------
// Internal Types
// -----------------------------------------------------------------------------

/// Internal error type for the resolution flow.
#[derive(Debug, thiserror::Error)]
enum ResolveError {
    /// MCP client call failed.
    #[error("{0}")]
    Client(#[from] mcp_client::McpClientError),

    /// Generated function names collide after sanitization or
    /// truncation.
    #[error("generated function name collision: \"{0}\" maps to multiple tools")]
    NameCollision(String),

    /// Failed to serialize the rewritten request body.
    #[error("failed to serialize rewritten request body: {0}")]
    Serialization(serde_json::Error),

    /// Duplicate `server_label` among resolvable MCP entries.
    #[error("duplicate server_label \"{0}\" in MCP tools array")]
    DuplicateLabel(String),

    /// Too many distinct MCP servers in one request.
    #[error("too many MCP servers: {count} exceeds limit of {max}")]
    TooManyServers {
        /// Actual count.
        count: usize,
        /// Configured maximum.
        max: usize,
    },

    /// Expanded request body exceeds `max_body_bytes`.
    #[error("expanded request body is {actual} bytes, exceeding the {limit} byte limit")]
    BodyTooLarge {
        /// Serialized body size after MCP tool expansion.
        actual: usize,
        /// Configured `max_body_bytes` limit.
        limit: usize,
    },

    /// Unknown `connector_id` was referenced.
    #[error("unknown connector_id \"{0}\"")]
    UnknownConnector(String),

    /// Both `connector_id` and `server_url` were provided.
    #[error("connector_id and server_url are mutually exclusive on entry \"{0}\"")]
    MutuallyExclusiveTarget(String),

    /// Connector ID combined with `defer_loading`.
    #[error("connector_id cannot be combined with defer_loading on entry \"{0}\"")]
    DeferredConnector(String),

    /// Invalid `connector_id` value (not a non-empty string).
    #[error("connector_id must be a non-empty string")]
    InvalidConnectorId,

    /// Connector entry missing required `server_label`.
    #[error("connector_id requires a non-empty server_label on entry \"{0}\"")]
    MissingConnectorLabel(String),

    /// Connector-backed MCP resolution failed (URL redacted).
    #[error("connector \"{connector_id}\" failed to resolve for server_label \"{label}\"")]
    ConnectorClient {
        /// The connector ID from the request.
        connector_id: String,
        /// The server label from the request.
        label: String,
    },

    /// A `tool_choice` references a resolved server with zero tools.
    #[error("tool_choice references server_label \"{0}\" which resolved to zero eligible tools")]
    EmptyResolvedToolChoice(String),
}

/// Per-entry resolution outcome.
enum EntryResolution {
    /// Entry was not resolved (deferred or no `server_url`).
    PassThrough,
    /// Entry was resolved to zero or more function tools.
    Resolved(Vec<serde_json::Value>),
}

/// Result of resolving all MCP entries.
struct Resolution {
    /// Per-entry resolution outcomes parallel to the input MCP entries.
    per_entry: Vec<EntryResolution>,
    /// Global dispatch map keyed by `(server_label, tool_name)`.
    tool_map: HashMap<(String, String), serde_json::Value>,
    /// Whether any entry was resolved (used to skip body rewrite when nothing resolved).
    has_resolved: bool,
    /// Labels of entries that were resolved (including those that produced zero tools).
    resolved_labels: HashSet<String>,
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

/// Reject the expanded body if it exceeds `max_body_bytes`.
fn check_body_size(serialized: &SerializedJson, max_body_bytes: usize) -> Result<(), ResolveError> {
    if serialized.len() > max_body_bytes {
        debug!(
            actual = serialized.len(),
            limit = max_body_bytes,
            "expanded request body exceeds configured limit"
        );
        return Err(ResolveError::BodyTooLarge {
            actual: serialized.len(),
            limit: max_body_bytes,
        });
    }
    Ok(())
}

/// Collect per-entry resolutions from task results.
fn collect_resolutions(
    entries: &[serde_json::Value],
    entry_to_task: &[Option<usize>],
    task_results: &[Option<Vec<serde_json::Value>>],
) -> Resolution {
    let mut tool_map = HashMap::new();
    let mut per_entry = Vec::with_capacity(entries.len());
    let mut has_resolved = false;
    let mut resolved_labels = HashSet::new();
    for (entry, task_idx) in entries.iter().zip(entry_to_task) {
        if let Some(resolution) = build_entry_resolution(entry, *task_idx, task_results, &mut tool_map) {
            has_resolved = true;
            resolved_labels.insert(server_label(entry).to_owned());
            per_entry.push(resolution);
        } else {
            per_entry.push(EntryResolution::PassThrough);
        }
    }
    Resolution {
        per_entry,
        tool_map,
        has_resolved,
        resolved_labels,
    }
}

/// Resolve connector IDs to server URLs for MCP tool entries.
///
/// Mutates entries in place, replacing `connector_id` with the
/// corresponding `server_url` from the configured connectors map.
fn resolve_connector_ids(
    connectors: &HashMap<String, url::Url>,
    entries: &mut [serde_json::Value],
) -> Result<(), ResolveError> {
    for entry in entries.iter_mut() {
        let Some(connector_id_value) = entry.get("connector_id") else {
            continue;
        };

        let connector_id = match connector_id_value.as_str() {
            Some(s) if !s.is_empty() && s.len() <= config::MAX_CONNECTOR_ID_LEN => s.to_owned(),
            _ => return Err(ResolveError::InvalidConnectorId),
        };

        validate_connector_entry(entry, &connector_id)?;

        let resolved_url = connectors
            .get(&connector_id)
            .ok_or(ResolveError::UnknownConnector(connector_id))?;

        if let Some(obj) = entry.as_object_mut() {
            obj.insert(
                "server_url".to_owned(),
                serde_json::Value::String(resolved_url.to_string()),
            );
        }
    }
    Ok(())
}

/// Validate that a connector entry has required fields and no conflicts.
fn validate_connector_entry(entry: &serde_json::Value, connector_id: &str) -> Result<(), ResolveError> {
    let label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty());
    if label.is_none() {
        return Err(ResolveError::MissingConnectorLabel(connector_id.to_owned()));
    }

    if entry.get("server_url").is_some() {
        return Err(ResolveError::MutuallyExclusiveTarget(connector_id.to_owned()));
    }

    if entry
        .get("defer_loading")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return Err(ResolveError::DeferredConnector(connector_id.to_owned()));
    }

    Ok(())
}

/// Build resolution for a single entry given task results.
fn build_entry_resolution(
    entry: &serde_json::Value,
    task_idx: Option<usize>,
    task_results: &[Option<Vec<serde_json::Value>>],
    tool_map: &mut HashMap<(String, String), serde_json::Value>,
) -> Option<EntryResolution> {
    let tools = task_idx.and_then(|idx| task_results.get(idx)?.clone())?;
    let allowed = extract_allowed_tools(entry);
    let filtered = apply_allowed_tools_filter(tools, &allowed);
    let label = server_label(entry);
    let function_tools: Vec<serde_json::Value> = filtered
        .iter()
        .map(|def| mcp_tool_to_function_tool(label, def))
        .collect();
    insert_tools(filtered, entry, tool_map);
    Some(EntryResolution::Resolved(function_tools))
}

/// Replace a [`ResolveError::Client`] with [`ResolveError::ConnectorClient`]
/// when the failing entry was connector-resolved, preventing internal URLs
/// from leaking to clients.
fn redact_connector_client_error(
    result: Result<Option<Vec<serde_json::Value>>, ResolveError>,
    entry: &serde_json::Value,
) -> Result<Option<Vec<serde_json::Value>>, ResolveError> {
    match result {
        Err(ResolveError::Client(client_err)) if entry.get("connector_id").is_some() => {
            debug!(error = %client_err, "connector resolution failed (redacting URL for client)");
            let connector_id = entry
                .get("connector_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_owned();
            let label = server_label(entry).to_owned();
            Err(ResolveError::ConnectorClient { connector_id, label })
        },
        other => other,
    }
}

/// Map a [`ResolveError`] to an appropriate rejection response.
fn resolve_error_rejection(err: &ResolveError, streaming: bool) -> FilterAction {
    let (status, error_type) = match err {
        ResolveError::DuplicateLabel(_)
        | ResolveError::TooManyServers { .. }
        | ResolveError::NameCollision(_)
        | ResolveError::UnknownConnector(_)
        | ResolveError::MutuallyExclusiveTarget(_)
        | ResolveError::DeferredConnector(_)
        | ResolveError::InvalidConnectorId
        | ResolveError::MissingConnectorLabel(_)
        | ResolveError::EmptyResolvedToolChoice(_) => (400, "invalid_request_error"),
        ResolveError::BodyTooLarge { .. } => (413, "invalid_request_error"),
        ResolveError::Client(_) | ResolveError::ConnectorClient { .. } => (502, "server_error"),
        ResolveError::Serialization(_) => (500, "server_error"),
    };
    let msg = err.to_string();
    debug!(error = %msg, "openai_mcp_tool_resolve rejected");
    FilterAction::Reject(responses_error_rejection(status, error_type, &msg, streaming))
}

/// Return the `server_url` if the entry should be eagerly
/// resolved: requires `server_label`, `server_url`, and
/// `defer_loading` not set to `true`.
fn resolvable_server_url(entry: &serde_json::Value) -> Option<&str> {
    if entry.get("server_label").and_then(serde_json::Value::as_str).is_none() {
        debug!("skipping MCP entry without server_label");
        return None;
    }
    let server_url = entry.get("server_url").and_then(serde_json::Value::as_str)?;
    if entry
        .get("defer_loading")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        debug!(server_url, "skipping deferred MCP entry");
        return None;
    }
    Some(server_url)
}

/// Reject requests containing more than one resolvable MCP entry
/// with the same `server_label`.
///
/// Without this check, duplicate credentialed entries bypass the
/// within-request cache (which is keyed on `(label, url)` and
/// skipped when the entry carries `authorization` or `headers`),
/// causing one `tools/list` callout per duplicate regardless of
/// the `max_servers` bound.
fn check_duplicate_labels(entries: &[serde_json::Value]) -> Result<(), ResolveError> {
    let mut seen = HashSet::new();
    for entry in entries {
        if resolvable_server_url(entry).is_none() {
            continue;
        }
        let label = server_label(entry);
        if !seen.insert(label) {
            return Err(ResolveError::DuplicateLabel(label.to_owned()));
        }
    }
    Ok(())
}

/// Return type for [`dedup_entries`]: `(entry_to_task, task_entries, task_allowed_names)`.
type DedupResult<'a> = (Vec<Option<usize>>, Vec<&'a serde_json::Value>, Vec<Option<Vec<String>>>);

/// Deduplicate MCP entries into resolution tasks. Non-credentialed
/// entries sharing the same `(label, url)` map to a single task;
/// credentialed entries always get their own task.
fn dedup_entries(entries: &[serde_json::Value]) -> DedupResult<'_> {
    let mut entry_to_task: Vec<Option<usize>> = vec![None; entries.len()];
    let mut task_entries: Vec<&serde_json::Value> = Vec::new();
    let mut task_allowed_names: Vec<Option<Vec<String>>> = Vec::new();
    let mut seen: HashMap<(&str, &str), usize> = HashMap::new();

    for (slot, entry) in entry_to_task.iter_mut().zip(entries) {
        let Some(url) = resolvable_server_url(entry) else {
            continue;
        };
        let allowed = extract_allowed_tools(entry);
        let existing = if has_entry_credentials(entry) {
            None
        } else {
            seen.get(&(server_label(entry), url)).copied()
        };
        if let Some(task_idx) = existing {
            if let Some(merged) = task_allowed_names.get_mut(task_idx) {
                merge_allowed_names(merged, &allowed.names);
            }
            *slot = Some(task_idx);
        } else {
            let task_idx = task_entries.len();
            if !has_entry_credentials(entry) {
                seen.insert((server_label(entry), url), task_idx);
            }
            task_entries.push(entry);
            task_allowed_names.push(allowed.names);
            *slot = Some(task_idx);
        }
    }

    (entry_to_task, task_entries, task_allowed_names)
}

/// Merge `new` into `existing` as a union. If either side is
/// unrestricted (`None`), the result is unrestricted.
fn merge_allowed_names(existing: &mut Option<Vec<String>>, new: &Option<Vec<String>>) {
    let Some(new_names) = new else {
        *existing = None;
        return;
    };
    let Some(existing_names) = existing.as_mut() else {
        return;
    };
    for name in new_names {
        if !existing_names.contains(name) {
            existing_names.push(name.clone());
        }
    }
}

/// Count distinct resolvable `(server_label, server_url)` pairs.
fn count_distinct_servers(entries: &[serde_json::Value]) -> usize {
    let mut seen = HashSet::new();
    for entry in entries {
        if let Some(url) = resolvable_server_url(entry) {
            seen.insert((server_label(entry), url));
        }
    }
    seen.len()
}

/// Call `tools/list` on the MCP server with per-page
/// `max_tools` enforcement.
async fn fetch_tools(
    entry: &serde_json::Value,
    server_url: &str,
    timeout: Duration,
    max_tools: usize,
    allow_loopback: bool,
) -> Result<Vec<serde_json::Value>, ResolveError> {
    debug!(server_url, "calling MCP tools/list");
    let auth = entry.get("authorization").and_then(serde_json::Value::as_str);
    mcp_client::list_tools(
        server_url,
        entry.get("headers"),
        auth,
        timeout,
        max_tools,
        allow_loopback,
    )
    .await
    .map_err(ResolveError::Client)
}

/// Rewrite the request body, replacing resolved `type: "mcp"`
/// entries with `type: "function"` entries and translating any
/// MCP `tool_choice` references.
///
/// Returns `None` only when the body cannot be rewritten at all
/// (unparseable body, non-object root, or missing `tools` array).
/// The caller is responsible for checking the serialized size against
/// `max_body_bytes` before committing.
///
/// A resolved MCP entry is always dropped from the outgoing tools
/// array, even when it produced zero permitted tools; otherwise the
/// entry's `authorization`/`headers` credentials would leak to the
/// inference backend. When every tool resolves away this yields an
/// empty `tools` array rather than the original (credentialed) body.
///
/// MCP entries that were not resolved (no `server_url` or deferred)
/// are left unchanged for upstream to handle.
fn rewrite_request_body(
    original_bytes: &[u8],
    per_entry: Vec<EntryResolution>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    resolved_labels: &HashSet<String>,
) -> Result<Option<SerializedJson>, ResolveError> {
    let Ok(mut parsed) = serde_json::from_slice::<serde_json::Value>(original_bytes) else {
        return Ok(None);
    };
    let Some(obj) = parsed.as_object_mut() else {
        return Ok(None);
    };
    let Some(serde_json::Value::Array(tools)) = obj.remove("tools") else {
        return Ok(None);
    };

    // An empty `rewritten` array here means every tool in the request
    // was a resolved MCP entry that produced zero permitted tools.
    // Commit the emptied array anyway: returning `None` would make the
    // caller forward the *original* body, leaking those entries'
    // `authorization`/`headers` credentials to the inference backend.
    let (rewritten, generated_names) = rewrite_tools_array(tools, per_entry);
    detect_name_collisions(&rewritten, &generated_names)?;

    let rewritten_count = rewritten.len();
    obj.insert("tools".to_owned(), serde_json::Value::Array(rewritten));
    rewrite_tool_choice(obj, tool_map, resolved_labels)?;

    let serialized = serialize_json_body(&parsed).map_err(|e| {
        debug!(error = %e, "failed to serialize rewritten body");
        ResolveError::Serialization(e)
    })?;
    debug!(tool_count = rewritten_count, "rewrote MCP tools to function tools");
    Ok(Some(serialized))
}

/// Rewrite a tools array, replacing resolved MCP entries with
/// pre-built function tools from `per_entry`.
fn rewrite_tools_array(
    tools: Vec<serde_json::Value>,
    per_entry: Vec<EntryResolution>,
) -> (Vec<serde_json::Value>, HashSet<String>) {
    let mut result = Vec::with_capacity(tools.len());
    let mut generated_names = HashSet::new();
    let mut entries = per_entry.into_iter();

    for tool in tools {
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("mcp") {
            result.push(tool);
            continue;
        }

        let resolution = entries.next().unwrap_or(EntryResolution::PassThrough);

        match resolution {
            EntryResolution::PassThrough => {
                result.push(tool);
            },
            EntryResolution::Resolved(function_tools) => {
                for ft in function_tools {
                    if let Some(name) = ft.get("name").and_then(serde_json::Value::as_str) {
                        generated_names.insert(name.to_owned());
                    }
                    result.push(ft);
                }
            },
        }
    }

    (result, generated_names)
}

/// Detect duplicate function names involving at least one
/// generated name.
///
/// Client-supplied duplicate functions are the backend's concern.
/// This only rejects collisions where lossy encoding produced a
/// generated name that clashes with another tool.
fn detect_name_collisions(tools: &[serde_json::Value], generated_names: &HashSet<String>) -> Result<(), ResolveError> {
    let mut seen = HashSet::new();
    for tool in tools {
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("function") {
            continue;
        }
        let Some(name) = tool.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if !seen.insert(name) && generated_names.contains(name) {
            return Err(ResolveError::NameCollision(name.to_owned()));
        }
    }
    Ok(())
}

/// Rewrite `tool_choice` when it references MCP tools.
///
/// Handles three cases:
///
/// - **Named MCP**: `{"type":"mcp","server_label":"X","name":"Y"}` → `{"type":"function","name":"X__Y"}`.
///
/// - **Server-level MCP**: `{"type":"mcp","server_label":"X"}` →
///   `{"type":"allowed_tools","mode":"required","tools":[...]}`.
///
/// - **MCP selectors in `allowed_tools`**: expands each MCP selector to its generated function equivalents.
fn rewrite_tool_choice(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    resolved_labels: &HashSet<String>,
) -> Result<(), ResolveError> {
    let Some(serde_json::Value::Object(choice_obj)) = obj.get("tool_choice").cloned() else {
        return Ok(());
    };
    let choice_type = choice_obj.get("type").and_then(serde_json::Value::as_str);

    match choice_type {
        Some("mcp") => rewrite_mcp_tool_choice(obj, &choice_obj, tool_map, resolved_labels),
        Some("allowed_tools") => rewrite_allowed_tools_choice(obj, &choice_obj, tool_map, resolved_labels),
        _ => Ok(()),
    }
}

/// Rewrite an MCP-typed `tool_choice` to its function equivalent.
///
/// Returns [`ResolveError::EmptyResolvedToolChoice`] when the
/// targeted label was resolved but produced zero eligible tools.
fn rewrite_mcp_tool_choice(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    choice_obj: &serde_json::Map<String, serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    resolved_labels: &HashSet<String>,
) -> Result<(), ResolveError> {
    let label = choice_obj
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    if let Some(name) = choice_obj.get("name").and_then(serde_json::Value::as_str) {
        if tool_map.contains_key(&(label.to_owned(), name.to_owned())) {
            let function_name = encode_function_name(label, name);
            obj.insert(
                "tool_choice".to_owned(),
                serde_json::json!({"type": "function", "name": function_name}),
            );
        } else if resolved_labels.contains(label) {
            return Err(ResolveError::EmptyResolvedToolChoice(label.to_owned()));
        }
        return Ok(());
    }

    let function_refs = collect_function_refs_for_label(label, tool_map);
    if !function_refs.is_empty() {
        obj.insert(
            "tool_choice".to_owned(),
            serde_json::json!({"type": "allowed_tools", "mode": "required", "tools": function_refs}),
        );
    } else if resolved_labels.contains(label) {
        return Err(ResolveError::EmptyResolvedToolChoice(label.to_owned()));
    }
    Ok(())
}

/// Rewrite MCP selectors inside an `allowed_tools`-typed
/// `tool_choice`.
///
/// Resolved selectors expand to their generated function refs. A
/// selector whose `server_label` was resolved locally but yields no
/// eligible tool is dropped, so a locally consumed MCP reference never
/// reaches the backend. Genuinely unresolved selectors (deferred,
/// connector-only, or unknown labels) are preserved for upstream.
///
/// If dropping locally consumed selectors leaves the list empty,
/// emitting `tools: []` would be a schema-invalid `allowed_tools`
/// choice. A `mode: "required"` choice is then rejected as unsatisfiable
/// with [`ResolveError::EmptyResolvedToolChoice`] (matching the
/// server-level MCP `tool_choice` behavior); any other mode is
/// normalized to `"none"`, because the restriction now permits no tool
/// and the model must not fall back to any other tool still supplied in
/// the request.
fn rewrite_allowed_tools_choice(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    choice_obj: &serde_json::Map<String, serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    resolved_labels: &HashSet<String>,
) -> Result<(), ResolveError> {
    let Some(tools_arr) = choice_obj.get("tools").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };

    let (new_tools, changed, dropped_label) = rebuild_allowed_tools_selectors(tools_arr, tool_map, resolved_labels);

    if !changed {
        return Ok(());
    }

    if new_tools.is_empty() {
        // Every selector was a locally resolved server that produced no
        // eligible tool, so the restricted set is now empty. An empty
        // `allowed_tools` is invalid, so resolve it by mode.
        if choice_obj.get("mode").and_then(serde_json::Value::as_str) == Some("required") {
            // `required` over an empty set is unsatisfiable: reject.
            return Err(ResolveError::EmptyResolvedToolChoice(
                dropped_label.unwrap_or_else(|| "unknown".to_owned()),
            ));
        }
        // `auto` restricted the model to this now-empty set. Normalize to
        // `"none"` rather than removing `tool_choice`: removal would let
        // the model call any other tool left in the request, which the
        // original choice deliberately excluded.
        obj.insert("tool_choice".to_owned(), serde_json::Value::String("none".to_owned()));
        return Ok(());
    }

    let mut new_choice = choice_obj.clone();
    new_choice.insert("tools".to_owned(), serde_json::Value::Array(new_tools));
    obj.insert("tool_choice".to_owned(), serde_json::Value::Object(new_choice));
    Ok(())
}

/// Rebuild an `allowed_tools` selector list: expand resolved MCP
/// selectors to their function refs, drop locally consumed
/// resolved-empty selectors, and preserve unresolved ones.
///
/// Returns the rebuilt selector list, whether it differs from the
/// input, and the first dropped `server_label` (used for error
/// reporting when the list collapses to empty).
fn rebuild_allowed_tools_selectors(
    tools_arr: &[serde_json::Value],
    tool_map: &HashMap<(String, String), serde_json::Value>,
    resolved_labels: &HashSet<String>,
) -> (Vec<serde_json::Value>, bool, Option<String>) {
    let mut new_tools = Vec::with_capacity(tools_arr.len());
    let mut changed = false;
    let mut dropped_label: Option<String> = None;

    for tool_ref in tools_arr {
        if tool_ref.get("type").and_then(serde_json::Value::as_str) != Some("mcp") {
            new_tools.push(tool_ref.clone());
            continue;
        }
        let before = new_tools.len();
        expand_mcp_selector(tool_ref, tool_map, &mut new_tools);
        if new_tools.len() > before {
            changed = true;
        } else if selector_label_resolved(tool_ref, resolved_labels) {
            // The server was resolved locally but exposes no matching
            // tool; the selector is locally consumed. Drop it so the
            // MCP reference never reaches the inference backend.
            changed = true;
            if dropped_label.is_none() {
                dropped_label = tool_ref
                    .get("server_label")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
        } else {
            // Unresolved (deferred / connector-only / unknown) selector:
            // preserve it for the backend to handle.
            new_tools.push(tool_ref.clone());
        }
    }

    (new_tools, changed, dropped_label)
}

/// Whether an MCP `tool_choice` selector targets a `server_label`
/// that was resolved locally. Such a selector is locally consumed and
/// must not survive into the backend request even when it produced
/// zero eligible tools.
fn selector_label_resolved(selector: &serde_json::Value, resolved_labels: &HashSet<String>) -> bool {
    selector
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|label| resolved_labels.contains(label))
}

/// Expand a single MCP selector into function tool references.
fn expand_mcp_selector(
    selector: &serde_json::Value,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    out: &mut Vec<serde_json::Value>,
) {
    let label = selector
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    if let Some(name) = selector.get("name").and_then(serde_json::Value::as_str) {
        if tool_map.contains_key(&(label.to_owned(), name.to_owned())) {
            out.push(serde_json::json!({"type": "function", "name": encode_function_name(label, name)}));
        }
    } else {
        out.extend(collect_function_refs_for_label(label, tool_map));
    }
}

/// Collect `{"type":"function","name":"..."}` refs for all tools
/// belonging to a given server label.
fn collect_function_refs_for_label(
    label: &str,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Vec<serde_json::Value> {
    tool_map
        .keys()
        .filter(|(l, _)| l == label)
        .map(|(l, n)| serde_json::json!({"type": "function", "name": encode_function_name(l, n)}))
        .collect()
}

/// Convert a single MCP tool definition to a Responses API
/// function tool.
///
/// The tool name is encoded as a bounded, schema-valid identifier
/// via [`encode_function_name`].
fn mcp_tool_to_function_tool(label: &str, definition: &serde_json::Value) -> serde_json::Value {
    let tool_name = definition
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let encoded_name = encode_function_name(label, tool_name);

    let description = definition.get("description").cloned();
    let parameters = definition
        .get("inputSchema")
        .or_else(|| definition.get("input_schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));

    let mut obj = serde_json::Map::new();
    obj.insert("type".to_owned(), serde_json::json!("function"));
    obj.insert("name".to_owned(), serde_json::json!(encoded_name));
    if let Some(desc) = description {
        obj.insert("description".to_owned(), desc);
    }
    obj.insert("parameters".to_owned(), parameters);

    serde_json::Value::Object(obj)
}

/// Encode `(label, tool_name)` into a bounded, schema-valid
/// function name matching `^[a-zA-Z0-9_-]+$` with max 64 chars.
///
/// The `{label}__{tool_name}` prefix is required for dispatch:
/// the backend has no concept of MCP servers, so the proxy must
/// embed the server identity in the function name to route
/// tool-call responses back to the correct upstream server.
///
/// Replaces invalid characters with `_` and truncates to fit
/// within [`MAX_FUNCTION_NAME_LEN`]. Lossy: distinct inputs can
/// produce the same output. Use [`detect_name_collisions`] after
/// building the full tools array to catch this.
pub(crate) fn encode_function_name(label: &str, tool_name: &str) -> String {
    let raw = format!("{label}__{tool_name}");
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.len() <= MAX_FUNCTION_NAME_LEN {
        sanitized
    } else {
        sanitized.chars().take(MAX_FUNCTION_NAME_LEN).collect()
    }
}

/// Write the resolved tool map and rewritten body to
/// `ResponsesState`, creating the state from the body if none
/// exists yet.
///
/// When a state already exists (e.g. from rehydration),
/// synchronizes `request_body`, `tools`, and `tool_choice` so
/// downstream filters (`openai_responses_proxy`) use the
/// rewritten body.
///
/// Skips state creation when the body carries
/// `previous_response_id` to avoid the downstream rebuild
/// path in `openai_responses_proxy` which would strip it.
fn write_state(ctx: &mut HttpFilterContext<'_>, body: &[u8], map: HashMap<(String, String), serde_json::Value>) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.mcp_tool_map = map;
        if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(body) {
            state.tools = parsed
                .get("tools")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            if let Some(tc) = parsed.get("tool_choice") {
                state.tool_choice = tc.clone();
            }
            state.request_body = parsed;
        }
    } else if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(body) {
        let mut state = ResponsesState::from_request_body(parsed);
        state.mcp_tool_map = map;
        ctx.extensions.insert(state);
    }
}

/// Check whether `openai_tool_parse` detected MCP tools.
fn has_mcp_tools(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_tool_parse.has_mcp")
        .is_some_and(|v| v == "true")
}

/// Check whether the request is streaming.
fn is_streaming(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream")
        .is_some_and(|v| v == "true")
}

/// Whether the entry carries per-entry credentials that
/// affect the `tools/list` response.
fn has_entry_credentials(entry: &serde_json::Value) -> bool {
    entry.get("authorization").and_then(serde_json::Value::as_str).is_some()
        || entry
            .get("headers")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|h| !h.is_empty())
}

/// Extract `server_label` from an MCP tool entry.
fn server_label(entry: &serde_json::Value) -> &str {
    entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
}

/// Extract MCP tool entries from the request body.
fn extract_mcp_entries(body: &[u8]) -> Vec<serde_json::Value> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return Vec::new();
    };

    let Some(tools) = value.get("tools").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };

    tools
        .iter()
        .filter(|t| t.get("type").and_then(serde_json::Value::as_str) == Some("mcp"))
        .cloned()
        .collect()
}

/// Extract `allowed_tools` from an MCP tool entry.
///
/// Handles both the string-array form (`["a", "b"]`) and the
/// `MCPToolFilter` object form (`{"tool_names": ["a"]}`).
fn extract_allowed_tools(entry: &serde_json::Value) -> AllowedTools {
    let Some(value) = entry.get("allowed_tools") else {
        return AllowedTools::unrestricted();
    };
    if let Some(arr) = value.as_array() {
        return AllowedTools {
            names: Some(extract_string_list(arr)),
            read_only: None,
        };
    }
    if let Some(obj) = value.as_object() {
        return extract_from_filter_object(obj);
    }
    AllowedTools::unrestricted()
}

/// Parse an `MCPToolFilter` object: `{tool_names?, read_only?}`.
fn extract_from_filter_object(obj: &serde_json::Map<String, serde_json::Value>) -> AllowedTools {
    let names = obj
        .get("tool_names")
        .and_then(serde_json::Value::as_array)
        .map(|arr| extract_string_list(arr));
    let read_only = obj.get("read_only").and_then(serde_json::Value::as_bool);
    AllowedTools { names, read_only }
}

/// Collect string elements from a JSON array.
fn extract_string_list(arr: &[serde_json::Value]) -> Vec<String> {
    arr.iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// Outcome of parsing `allowed_tools`.
struct AllowedTools {
    /// Optional tool-name allowlist.
    names: Option<Vec<String>>,
    /// `Some(true)` = only read-only tools,
    /// `Some(false)` = only writable tools,
    /// `None` = no read-only filter.
    read_only: Option<bool>,
}

impl AllowedTools {
    /// No filter — expose all tools.
    fn unrestricted() -> Self {
        Self {
            names: None,
            read_only: None,
        }
    }

    /// Return the name list as a slice, or `None` if
    /// unrestricted.
    fn as_names(&self) -> Option<&[String]> {
        self.names.as_deref()
    }
}

/// Check `previous_tools` for a cached listing matching
/// `server_label` and `server_url`.
///
/// When the cached entry has `server_url`, both label and
/// URL must match. When the cached entry lacks `server_url`
/// (real `mcp_list_tools` output items from the API omit
/// it), label-only matching is used.
///
/// # Safety of label-only matching
///
/// Real `mcp_list_tools` items in the API response carry
/// `server_label` and `tools` but not `server_url`.
/// Label-only matching is safe because:
///
/// 1. Tool dispatch uses the current request's `server_url`, so stale tools fail safely at call time.
/// 2. When the cached entry _does_ carry `server_url` (e.g. enriched by a future storage layer), exact URL matching
///    applies automatically.
///
/// Requires `allowed_tools` to be `Some` and verifies the
/// cache covers all named tools. Returns `None` for
/// unrestricted entries because the cached listing may be
/// a filtered subset from a previous response.
fn find_cached_listing(
    previous_tools: Option<&Vec<serde_json::Value>>,
    label: &str,
    server_url: &str,
    allowed_tools: Option<&[String]>,
    require_url_match: bool,
) -> Option<Vec<serde_json::Value>> {
    let previous = previous_tools?;
    let allowed = allowed_tools?;

    let entry = previous.iter().find(|pt| {
        let label_matches = pt.get("server_label").and_then(serde_json::Value::as_str) == Some(label);
        let url_ok = match pt.get("server_url").and_then(serde_json::Value::as_str) {
            Some(cached_url) => cached_url == server_url,
            None => !require_url_match,
        };
        label_matches && url_ok
    })?;

    let cached_tools = entry.get("tools").and_then(serde_json::Value::as_array)?;
    let all_present = allowed.iter().all(|name| {
        cached_tools
            .iter()
            .any(|t| t.get("name").and_then(serde_json::Value::as_str) == Some(name))
    });
    if !all_present {
        return None;
    }

    Some(cached_tools.clone())
}

/// Filter tools by name list and/or read-only annotation.
fn apply_allowed_tools_filter(tools: Vec<serde_json::Value>, allowed: &AllowedTools) -> Vec<serde_json::Value> {
    let names = allowed.as_names();
    let read_only = allowed.read_only;

    if names.is_none() && read_only.is_none() {
        return tools;
    }

    tools
        .into_iter()
        .filter(|t| {
            if let Some(list) = names {
                let matches_name = t
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|n| list.iter().any(|a| a == n));
                if !matches_name {
                    return false;
                }
            }
            if let Some(want_read_only) = read_only {
                return tool_read_only_hint(t) == want_read_only;
            }
            true
        })
        .collect()
}

/// Return whether an MCP tool has `annotations.readOnlyHint`
/// set to `true` (defaults to `false` when absent).
fn tool_read_only_hint(tool: &serde_json::Value) -> bool {
    tool.get("annotations")
        .and_then(|a| a.get("readOnlyHint"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Insert resolved tools into the tool map keyed by
/// `(server_label, tool_name)`, consuming the definitions.
fn insert_tools(
    tools: Vec<serde_json::Value>,
    entry: &serde_json::Value,
    tool_map: &mut HashMap<(String, String), serde_json::Value>,
) {
    let label = server_label(entry);
    let server_url = entry
        .get("server_url")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers").cloned();
    let authorization = entry.get("authorization").cloned();
    let require_approval = entry.get("require_approval").cloned();
    let connector_id = entry.get("connector_id").cloned();

    for tool in tools {
        let tool_name = tool.get("name").and_then(serde_json::Value::as_str).map(str::to_owned);
        let Some(tool_name) = tool_name else {
            continue;
        };

        let key = (label.to_owned(), tool_name);
        tool_map.insert(
            key,
            serde_json::json!({
                "server_label": label,
                "server_url": server_url,
                "headers": headers,
                "authorization": authorization,
                "require_approval": require_approval,
                "connector_id": connector_id,
                "tool_definition": tool,
            }),
        );
    }
}
