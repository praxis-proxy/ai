// SPDX-License-Identifier: MIT
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
    borrow::Cow,
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
use crate::mcp_client;

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
        Ok(Box::new(Self {
            allow_loopback: validated.allow_loopback,
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
        let mcp_entries = extract_mcp_entries(&original_bytes);
        if mcp_entries.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let server_count = count_distinct_servers(&mcp_entries);
        if server_count > self.max_servers {
            return Err(ResolveError::TooManyServers {
                count: server_count,
                max: self.max_servers,
            });
        }

        let previous_tools = ctx.extensions.get::<ResponsesState>().map(|s| &s.previous_tools);

        let resolution = self.resolve_all_entries(&mcp_entries, previous_tools).await?;

        if resolution.tool_map.is_empty() {
            return Ok(FilterAction::Continue);
        }

        debug!(tool_count = resolution.tool_map.len(), "mcp_tool_map built");

        let Resolution { per_entry, tool_map } = resolution;
        rewrite_request_body(ctx, body, &original_bytes, per_entry, &tool_map)?;

        let body_for_state = body.as_ref().map_or_else(|| original_bytes.as_ref(), |b| b.as_ref());
        write_state(ctx, body_for_state, tool_map);
        Ok(FilterAction::Continue)
    }

    /// Resolve all MCP entries, building both the global dispatch
    /// map and pre-built function tools for body rewriting.
    async fn resolve_all_entries(
        &self,
        entries: &[serde_json::Value],
        previous_tools: Option<&Vec<serde_json::Value>>,
    ) -> Result<Resolution, ResolveError> {
        let mut tool_map = HashMap::new();
        let mut per_entry = Vec::with_capacity(entries.len());
        let mut fetched: HashMap<(&str, &str), Vec<serde_json::Value>> = HashMap::new();

        for entry in entries {
            let Some(tools) = self.resolve_entry(entry, previous_tools, &mut fetched).await? else {
                per_entry.push(Vec::new());
                continue;
            };
            let allowed = extract_allowed_tools(entry);
            let filtered = apply_allowed_tools_filter(tools, &allowed);
            let label = server_label(entry);
            let function_tools: Vec<serde_json::Value> = filtered
                .iter()
                .map(|def| mcp_tool_to_function_tool(label, def))
                .collect();
            insert_tools(filtered, entry, &mut tool_map);
            per_entry.push(function_tools);
        }

        Ok(Resolution { per_entry, tool_map })
    }

    /// Resolve tools for a single entry, reusing within-request
    /// cached results for the same `(label, url)`.
    async fn resolve_entry<'a>(
        &self,
        entry: &'a serde_json::Value,
        previous_tools: Option<&Vec<serde_json::Value>>,
        fetched: &mut HashMap<(&'a str, &'a str), Vec<serde_json::Value>>,
    ) -> Result<Option<Vec<serde_json::Value>>, ResolveError> {
        let Some(server_url) = resolvable_server_url(entry) else {
            return Ok(None);
        };
        let label = server_label(entry);
        let has_credentials = has_entry_credentials(entry);
        if !has_credentials && let Some(cached) = fetched.get(&(label, server_url)) {
            return Ok(Some(cached.clone()));
        }
        mcp_client::validate_mcp_url(server_url, self.timeout, self.allow_loopback)
            .await
            .map_err(ResolveError::Client)?;
        let allowed = extract_allowed_tools(entry);
        if !has_credentials
            && let Some(cached) = find_cached_listing(previous_tools, label, server_url, allowed.as_names())
        {
            debug!(label, tool_count = cached.len(), "reusing cached MCP tool listing");
            return Ok(Some(cached));
        }
        let tools = fetch_tools(entry, server_url, self.timeout, self.max_tools, self.allow_loopback).await?;
        if !has_credentials {
            fetched.insert((label, server_url), tools.clone());
        }
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

    /// Too many distinct MCP servers in one request.
    #[error("too many MCP servers: {count} exceeds limit of {max}")]
    TooManyServers {
        /// Actual count.
        count: usize,
        /// Configured maximum.
        max: usize,
    },
}

/// Result of resolving all MCP entries.
struct Resolution {
    /// Pre-built function tools parallel to the input MCP entries.
    /// Empty vec means the entry was not resolved.
    per_entry: Vec<Vec<serde_json::Value>>,
    /// Global dispatch map keyed by `(server_label, tool_name)`.
    tool_map: HashMap<(String, String), serde_json::Value>,
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

/// Map a [`ResolveError`] to an appropriate rejection response.
fn resolve_error_rejection(err: &ResolveError, streaming: bool) -> FilterAction {
    let (status, error_type) = match err {
        ResolveError::TooManyServers { .. } | ResolveError::NameCollision(_) => (400, "invalid_request_error"),
        ResolveError::Client(_) => (502, "server_error"),
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
/// MCP entries that were not resolved (no `server_url`, deferred,
/// or `connector_id` only) are left unchanged for upstream to
/// handle.
fn rewrite_request_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    original_bytes: &[u8],
    per_entry: Vec<Vec<serde_json::Value>>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Result<(), ResolveError> {
    let Ok(mut parsed) = serde_json::from_slice::<serde_json::Value>(original_bytes) else {
        return Ok(());
    };

    let Some(obj) = parsed.as_object_mut() else {
        return Ok(());
    };

    let Some(serde_json::Value::Array(tools)) = obj.remove("tools") else {
        return Ok(());
    };

    let (rewritten, generated_names) = rewrite_tools_array(tools, per_entry);

    if rewritten.is_empty() {
        return Ok(());
    }

    detect_name_collisions(&rewritten, &generated_names)?;

    let rewritten_count = rewritten.len();
    obj.insert("tools".to_owned(), serde_json::Value::Array(rewritten));
    rewrite_tool_choice(obj, tool_map);

    let serialized = serde_json::to_vec(&parsed).map_err(|e| {
        debug!(error = %e, "failed to serialize rewritten body");
        ResolveError::Serialization(e)
    })?;

    let len = serialized.len();
    *body = Some(Bytes::from(serialized));
    ctx.extra_request_headers
        .push((Cow::Borrowed("content-length"), len.to_string()));

    debug!(tool_count = rewritten_count, "rewrote MCP tools to function tools");
    Ok(())
}

/// Rewrite a tools array, replacing resolved MCP entries with
/// pre-built function tools from `per_entry`.
fn rewrite_tools_array(
    tools: Vec<serde_json::Value>,
    per_entry: Vec<Vec<serde_json::Value>>,
) -> (Vec<serde_json::Value>, HashSet<String>) {
    let mut result = Vec::with_capacity(tools.len());
    let mut generated_names = HashSet::new();
    let mut entries = per_entry.into_iter();

    for tool in tools {
        if tool.get("type").and_then(serde_json::Value::as_str) != Some("mcp") {
            result.push(tool);
            continue;
        }

        let function_tools = entries.next().unwrap_or_default();

        if function_tools.is_empty() {
            result.push(tool);
            continue;
        }

        for ft in function_tools {
            if let Some(name) = ft.get("name").and_then(serde_json::Value::as_str) {
                generated_names.insert(name.to_owned());
            }
            result.push(ft);
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
) {
    let Some(serde_json::Value::Object(choice_obj)) = obj.get("tool_choice").cloned() else {
        return;
    };
    let choice_type = choice_obj.get("type").and_then(serde_json::Value::as_str);

    match choice_type {
        Some("mcp") => rewrite_mcp_tool_choice(obj, &choice_obj, tool_map),
        Some("allowed_tools") => rewrite_allowed_tools_choice(obj, &choice_obj, tool_map),
        _ => {},
    }
}

/// Rewrite an MCP-typed `tool_choice` to its function equivalent.
fn rewrite_mcp_tool_choice(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    choice_obj: &serde_json::Map<String, serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) {
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
        }
        return;
    }

    let function_refs = collect_function_refs_for_label(label, tool_map);
    if !function_refs.is_empty() {
        obj.insert(
            "tool_choice".to_owned(),
            serde_json::json!({"type": "allowed_tools", "mode": "required", "tools": function_refs}),
        );
    }
}

/// Rewrite MCP selectors inside an `allowed_tools`-typed
/// `tool_choice`.
fn rewrite_allowed_tools_choice(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    choice_obj: &serde_json::Map<String, serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) {
    let Some(tools_arr) = choice_obj.get("tools").and_then(serde_json::Value::as_array) else {
        return;
    };

    let mut new_tools = Vec::with_capacity(tools_arr.len());
    let mut changed = false;

    for tool_ref in tools_arr {
        if tool_ref.get("type").and_then(serde_json::Value::as_str) != Some("mcp") {
            new_tools.push(tool_ref.clone());
            continue;
        }
        let before = new_tools.len();
        expand_mcp_selector(tool_ref, tool_map, &mut new_tools);
        if new_tools.len() > before {
            changed = true;
        } else {
            new_tools.push(tool_ref.clone());
        }
    }

    if changed {
        let mut new_choice = choice_obj.clone();
        new_choice.insert("tools".to_owned(), serde_json::Value::Array(new_tools));
        obj.insert("tool_choice".to_owned(), serde_json::Value::Object(new_choice));
    }
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
fn encode_function_name(label: &str, tool_name: &str) -> String {
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
) -> Option<Vec<serde_json::Value>> {
    let previous = previous_tools?;
    let allowed = allowed_tools?;

    let entry = previous.iter().find(|pt| {
        let label_matches = pt.get("server_label").and_then(serde_json::Value::as_str) == Some(label);
        let url_ok = match pt.get("server_url").and_then(serde_json::Value::as_str) {
            Some(cached_url) => cached_url == server_url,
            None => true,
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
                "tool_definition": tool,
            }),
        );
    }
}
