// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Filter 18: resolve MCP tool declarations into concrete tool
//! definitions by calling upstream MCP servers' `tools/list`.
//!
//! Runs after `tool_parse`, gated on `tool_parse.has_mcp`.
//! Reads MCP entries from the buffered request body, checks
//! `previous_tools` for cached listings, calls `tools/list` via
//! the `mcp_client` module, and writes `mcp_tool_map` to
//! [`ResponsesState`].
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

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use tracing::debug;

use self::config::{McpToolResolveConfig, build_config};
use super::{error::responses_error_rejection, state::ResponsesState};
use crate::mcp_client;

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
/// filter: mcp_tool_resolve
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: mcp_tool_resolve
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
        let cfg: McpToolResolveConfig = parse_filter_config("mcp_tool_resolve", config)?;
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
    /// `tools/list`, build `mcp_tool_map`.
    async fn resolve_mcp_tools(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &[u8],
    ) -> Result<FilterAction, ResolveError> {
        let mcp_entries = extract_mcp_entries(body);
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

        let map = self.build_tool_map(&mcp_entries, previous_tools).await?;

        if map.is_empty() {
            return Ok(FilterAction::Continue);
        }

        debug!(tool_count = map.len(), "mcp_tool_map built");
        write_tool_map(ctx, body, map);
        Ok(FilterAction::Continue)
    }

    /// Build tool map from all MCP entries, resolving each
    /// server once and applying per-entry filters.
    async fn build_tool_map(
        &self,
        entries: &[serde_json::Value],
        previous_tools: Option<&Vec<serde_json::Value>>,
    ) -> Result<HashMap<(String, String), serde_json::Value>, ResolveError> {
        let mut tool_map = HashMap::new();
        let mut fetched: HashMap<(&str, &str), Vec<serde_json::Value>> = HashMap::new();

        for entry in entries {
            let Some(tools) = self.resolve_entry(entry, previous_tools, &mut fetched).await? else {
                continue;
            };
            let allowed = extract_allowed_tools(entry);
            insert_tools(&apply_allowed_tools_filter(tools, &allowed), entry, &mut tool_map);
        }

        Ok(tool_map)
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
        "mcp_tool_resolve"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
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

        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };

        let streaming = is_streaming(ctx);

        match Box::pin(self.resolve_mcp_tools(ctx, bytes)).await {
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

    /// Too many distinct MCP servers in one request.
    #[error("too many MCP servers: {count} exceeds limit of {max}")]
    TooManyServers {
        /// Actual count.
        count: usize,
        /// Configured maximum.
        max: usize,
    },
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

/// Map a [`ResolveError`] to an appropriate rejection response.
fn resolve_error_rejection(err: &ResolveError, streaming: bool) -> FilterAction {
    match err {
        ResolveError::TooManyServers { count, max } => {
            let msg = format!("too many MCP servers: {count} exceeds limit of {max}");
            debug!(error = %msg, "mcp_tool_resolve rejected");
            FilterAction::Reject(responses_error_rejection(400, "invalid_request_error", &msg, streaming))
        },
        ResolveError::Client(e) => {
            debug!(error = %e, "mcp_tool_resolve failed");
            FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                &err.to_string(),
                streaming,
            ))
        },
    }
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
    let mut seen = std::collections::HashSet::new();
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

/// Write the resolved tool map to `ResponsesState`, creating
/// the state from the request body if none exists yet.
///
/// Skips state creation when the body carries
/// `previous_response_id` to avoid the downstream rebuild
/// path in `responses_proxy` which would strip it.
fn write_tool_map(ctx: &mut HttpFilterContext<'_>, body: &[u8], map: HashMap<(String, String), serde_json::Value>) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.mcp_tool_map = map;
    } else if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(body) {
        if parsed
            .get("previous_response_id")
            .and_then(serde_json::Value::as_str)
            .is_some()
        {
            debug!("skipping state creation for continuation request");
            return;
        }
        let mut state = ResponsesState::from_request_body(parsed);
        state.mcp_tool_map = map;
        ctx.extensions.insert(state);
    }
}

/// Check whether `tool_parse` detected MCP tools.
fn has_mcp_tools(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("tool_parse.has_mcp").is_some_and(|v| v == "true")
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
/// `(server_label, tool_name)`.
fn insert_tools(
    tools: &[serde_json::Value],
    entry: &serde_json::Value,
    tool_map: &mut HashMap<(String, String), serde_json::Value>,
) {
    let label = server_label(entry);
    let server_url = entry
        .get("server_url")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization");
    let require_approval = entry.get("require_approval");

    for tool in tools {
        let Some(tool_name) = tool.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };

        let key = (label.to_owned(), tool_name.to_owned());
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
