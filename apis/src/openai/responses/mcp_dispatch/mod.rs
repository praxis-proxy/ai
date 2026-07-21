// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Filter 8: execute MCP tool calls against upstream MCP servers.
//!
//! Operates in two phases within the agentic loop:
//!
//! 1. **Response path** (`on_response_body`): identifies MCP tool calls in [`ResponsesState::tool_calls`], checks
//!    approval policies, and writes `openai_mcp_dispatch.action` into [`filter_metadata`] to signal re-entry.
//! 2. **Request path** (`on_request`, re-entry): executes pending MCP calls via [`mcp_client::call_tool`] and appends
//!    results to `messages`, `persisted_messages`, and `output_items`.
//!
//! # Pipeline dependencies
//!
//! - **`mcp_tool_resolve`** must run before this filter so that [`ResponsesState::mcp_tool_map`] is populated.
//! - **`openai_stream_events`** (or equivalent accumulator) must populate [`ResponsesState::tool_calls`] from the
//!   upstream response. Currently only `function_call` events are accumulated; native `mcp_call` events require either
//!   `mcp_tool_resolve` rewriting MCP tools into function tools or the accumulator adding `mcp_call` support.
//! - **`agentic_loop`** (issue #26, not yet built) must read `openai_mcp_dispatch.action` from [`filter_metadata`] and
//!   trigger pipeline re-entry so that phase 2 executes.
//!
//! Uses [`filter_metadata`] (not `filter_results`) because
//! `filter_results` are cleared after branch evaluation and no
//! branch evaluation runs in the response-body phase.
//!
//! [`ResponsesState::tool_calls`]: super::state::ResponsesState
//! [`ResponsesState::mcp_tool_map`]: super::state::ResponsesState
//! [`filter_metadata`]: HttpFilterContext::filter_metadata

pub(crate) mod approval;
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
use tracing::{debug, warn};

use self::{
    approval::{parse_approval_policy, requires_approval},
    config::{McpDispatchConfig, build_config},
};
use super::state::ResponsesState;
use crate::mcp_client;

// -----------------------------------------------------------------------------
// McpDispatchFilter
// -----------------------------------------------------------------------------

/// Executes MCP tool calls against upstream MCP servers within
/// the Responses API agentic loop.
///
/// # YAML
///
/// ```yaml
/// filter: openai_mcp_dispatch
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_mcp_dispatch
/// timeout_ms: 30000
/// max_body_bytes: 67108864
/// ```
pub struct McpDispatchFilter {
    /// Allow connections to loopback addresses.
    allow_loopback: bool,
    /// Maximum response body bytes.
    max_body_bytes: usize,
    /// Timeout for MCP tool calls.
    timeout: Duration,
}

impl McpDispatchFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpDispatchConfig = parse_filter_config("openai_mcp_dispatch", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self {
            allow_loopback: validated.allow_loopback,
            max_body_bytes: validated.max_body_bytes,
            timeout: Duration::from_millis(validated.timeout_ms),
        }))
    }

    /// Handle a tool call that requires approval.
    fn handle_approval_required(ctx: &mut HttpFilterContext<'_>, pending: &PendingApproval) -> FilterAction {
        debug!(
            tool_name = %pending.tool_name,
            server_label = %pending.server_label,
            "MCP tool call requires approval"
        );

        let approval_event = serde_json::json!({
            "type": "mcp_approval_request",
            "id": pending.call_id,
            "name": pending.tool_name,
            "server_label": pending.server_label,
            "arguments": pending.arguments,
        });

        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            warn!("ResponsesState missing when handling approval");
            return FilterAction::Continue;
        };
        state.output_items_mut().push(approval_event);

        ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "done".to_owned());

        FilterAction::Continue
    }
}

#[async_trait]
impl HttpFilter for McpDispatchFilter {
    fn name(&self) -> &'static str {
        "openai_mcp_dispatch"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        let mcp_calls = extract_mcp_tool_calls(&state.tool_calls, &state.mcp_tool_map);
        if mcp_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        debug!(count = mcp_calls.len(), "executing pending MCP tool calls");

        let parallel = state.parallel_tool_calls;
        let tool_map = std::sync::Arc::new(state.mcp_tool_map.clone());
        let results = execute_mcp_calls(&mcp_calls, &tool_map, parallel, self.timeout, self.allow_loopback).await;

        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            warn!("ResponsesState missing when appending results");
            return Ok(FilterAction::Continue);
        };
        for result in results {
            state.messages.push(result.message.clone());
            state.persisted_messages.push(result.message);
            state.output_items_mut().push(result.output_item);
        }

        let tool_map_ref = &state.mcp_tool_map;
        state.tool_calls.retain(|tc| !is_mcp_tool_call(tc, tool_map_ref));

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Release);
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        let mcp_calls = extract_mcp_tool_calls(&state.tool_calls, &state.mcp_tool_map);
        if mcp_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        if let Some(pending) = find_approval_required(&mcp_calls, &state.mcp_tool_map) {
            return Ok(Self::handle_approval_required(ctx, &pending));
        }

        ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "execute_mcp".to_owned());

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Approval Handling
// -----------------------------------------------------------------------------

/// Info about a tool call that requires approval.
struct PendingApproval {
    /// Call ID.
    call_id: String,
    /// Server label.
    server_label: String,
    /// Tool name.
    tool_name: String,
    /// Tool arguments as JSON string.
    arguments: String,
}

// -----------------------------------------------------------------------------
// MCP Tool Call Identification
// -----------------------------------------------------------------------------

/// Extract MCP tool calls from the `tool_calls` list by checking
/// `mcp_tool_map`.
fn extract_mcp_tool_calls(
    tool_calls: &[serde_json::Value],
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Vec<serde_json::Value> {
    tool_calls
        .iter()
        .filter(|tc| is_mcp_tool_call(tc, tool_map))
        .cloned()
        .collect()
}

/// Check whether a tool call is an MCP tool call.
fn is_mcp_tool_call(tool_call: &serde_json::Value, tool_map: &HashMap<(String, String), serde_json::Value>) -> bool {
    tool_call
        .get("name")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|name| find_by_tool_name(tool_map, name).is_some())
}

/// Find an entry in the tool map by tool name (second element of
/// the `(server_label, tool_name)` key). Returns the first match.
///
/// Warns when multiple servers expose the same tool name, since
/// routing is first-match and may be nondeterministic.
fn find_by_tool_name<'a>(
    tool_map: &'a HashMap<(String, String), serde_json::Value>,
    tool_name: &str,
) -> Option<&'a serde_json::Value> {
    let mut matches = tool_map.iter().filter(|((_, name), _)| name == tool_name);
    let first = matches.next()?;
    if matches.next().is_some() {
        warn!(
            tool_name,
            "multiple servers expose the same tool name; routing to first match"
        );
    }
    Some(first.1)
}

// -----------------------------------------------------------------------------
// Approval Pre-check
// -----------------------------------------------------------------------------

/// Scan all MCP tool calls for approval requirements. Returns the
/// first tool call that requires approval, or `None` if all are
/// approved.
fn find_approval_required(
    mcp_calls: &[serde_json::Value],
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Option<PendingApproval> {
    mcp_calls.iter().find_map(|tc| check_single_approval(tc, tool_map))
}

/// Extract the call ID from a tool call value.
fn extract_call_id(tc: &serde_json::Value) -> String {
    tc.get("call_id")
        .or_else(|| tc.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned()
}

/// Extract serialised arguments from a tool call value.
fn extract_arguments(tc: &serde_json::Value) -> String {
    tc.get("arguments").map(ToString::to_string).unwrap_or_default()
}

/// Check a single tool call for approval requirement.
///
/// Returns `Some` with approval details if approval is required,
/// or if the tool name is ambiguous across servers.
fn check_single_approval(
    tc: &serde_json::Value,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Option<PendingApproval> {
    let tool_name = tc.get("name").and_then(serde_json::Value::as_str)?;

    let match_count = tool_map.keys().filter(|(_, name)| name == tool_name).count();
    if match_count > 1 {
        warn!(
            tool_name,
            server_count = match_count,
            "ambiguous tool name in approval check; requiring approval"
        );
        return Some(PendingApproval {
            call_id: extract_call_id(tc),
            server_label: "unknown".to_owned(),
            tool_name: tool_name.to_owned(),
            arguments: extract_arguments(tc),
        });
    }

    let entry = find_by_tool_name(tool_map, tool_name)?;
    if !requires_approval(&parse_approval_policy(entry), tool_name) {
        return None;
    }

    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    Some(PendingApproval {
        call_id: extract_call_id(tc),
        server_label,
        tool_name: tool_name.to_owned(),
        arguments: extract_arguments(tc),
    })
}

// -----------------------------------------------------------------------------
// Execution
// -----------------------------------------------------------------------------

/// Result of executing a single MCP tool call.
#[derive(Debug)]
struct McpCallResult {
    /// Tool result message for `messages` and `persisted_messages`.
    message: serde_json::Value,
    /// Output item for `output_items`.
    output_item: serde_json::Value,
}

/// Execute MCP tool calls — concurrently when `parallel` is true,
/// sequentially otherwise.
async fn execute_mcp_calls(
    mcp_calls: &[serde_json::Value],
    tool_map: &std::sync::Arc<HashMap<(String, String), serde_json::Value>>,
    parallel: bool,
    timeout: Duration,
    allow_loopback: bool,
) -> Vec<McpCallResult> {
    if parallel {
        execute_parallel(mcp_calls, tool_map, timeout, allow_loopback).await
    } else {
        execute_sequential(mcp_calls, tool_map, timeout, allow_loopback).await
    }
}

/// Execute MCP tool calls concurrently, emitting error results
/// for any dropped or panicked tasks.
async fn execute_parallel(
    mcp_calls: &[serde_json::Value],
    tool_map: &std::sync::Arc<HashMap<(String, String), serde_json::Value>>,
    timeout: Duration,
    allow_loopback: bool,
) -> Vec<McpCallResult> {
    let handles: Vec<_> = mcp_calls
        .iter()
        .map(|tc| {
            let tc = tc.clone();
            let map = std::sync::Arc::clone(tool_map);
            tokio::spawn(async move { execute_single_call(&tc, &map, timeout, allow_loopback).await })
        })
        .collect();
    let mut results = Vec::with_capacity(handles.len());
    for (tc, handle) in mcp_calls.iter().zip(handles) {
        match handle.await {
            Ok(Some(result)) => results.push(result),
            Ok(None) => {
                warn!(tool = ?tc.get("name"), "parallel MCP call returned None, emitting error");
                results.push(error_result_for_dropped_call(
                    tc,
                    "internal error: call produced no result",
                ));
            },
            Err(e) => {
                warn!(tool = ?tc.get("name"), error = %e, "parallel MCP call task failed, emitting error");
                results.push(error_result_for_dropped_call(tc, &format!("task failed: {e}")));
            },
        }
    }
    results
}

/// Execute MCP tool calls sequentially, emitting error results
/// for any calls that produce no result.
async fn execute_sequential(
    mcp_calls: &[serde_json::Value],
    tool_map: &std::sync::Arc<HashMap<(String, String), serde_json::Value>>,
    timeout: Duration,
    allow_loopback: bool,
) -> Vec<McpCallResult> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    for tc in mcp_calls {
        if let Some(result) = execute_single_call(tc, tool_map, timeout, allow_loopback).await {
            results.push(result);
        } else {
            warn!(tool = ?tc.get("name"), "sequential MCP call returned None, emitting error");
            results.push(error_result_for_dropped_call(
                tc,
                "internal error: call produced no result",
            ));
        }
    }
    results
}

/// Resolve a tool name to its unique entry, rejecting ambiguity.
fn resolve_tool_entry<'a>(
    tool_map: &'a HashMap<(String, String), serde_json::Value>,
    tool_name: &str,
    call_id: &str,
) -> Result<&'a serde_json::Value, Option<Box<McpCallResult>>> {
    let match_count = tool_map.keys().filter(|(_, name)| name == tool_name).count();
    if match_count == 0 {
        warn!(tool_name, "tool not found in mcp_tool_map, skipping");
        return Err(None);
    }
    if match_count > 1 {
        warn!(
            tool_name,
            server_count = match_count,
            "ambiguous MCP tool name: multiple servers expose this tool"
        );
        return Err(Some(Box::new(build_error_result(
            call_id,
            "unknown",
            tool_name,
            "",
            &format!("ambiguous tool name: {match_count} servers expose '{tool_name}'"),
        ))));
    }
    find_by_tool_name(tool_map, tool_name).ok_or(None)
}

/// Parse tool call arguments, handling JSON-string encoding.
fn parse_call_arguments(
    tool_call: &serde_json::Value,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
) -> Result<(serde_json::Value, String), Box<McpCallResult>> {
    let raw = tool_call
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
    let arguments = match &raw {
        serde_json::Value::String(s) => match serde_json::from_str(s) {
            Ok(parsed) => parsed,
            Err(e) => {
                warn!(tool_name, error = %e, "malformed JSON in tool call arguments");
                return Err(Box::new(build_error_result(
                    call_id,
                    server_label,
                    tool_name,
                    s,
                    &format!("malformed tool arguments: {e}"),
                )));
            },
        },
        other => other.clone(),
    };
    let arguments_string = arguments.to_string();
    Ok((arguments, arguments_string))
}

/// Build result from a completed (successful or failed) MCP call.
#[expect(clippy::too_many_lines, reason = "match branches with structured logging")]
fn process_call_result(
    result: Result<rmcp::model::CallToolResult, mcp_client::McpClientError>,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments_string: &str,
) -> McpCallResult {
    match result {
        Ok(r) => {
            let is_error = r.is_error.unwrap_or(false);
            let non_text = r
                .content
                .iter()
                .filter(|b| !matches!(b, rmcp::model::ContentBlock::Text(_)))
                .count();
            if non_text > 0 {
                warn!(
                    tool_name,
                    call_id, non_text, "non-text content blocks discarded from MCP response"
                );
            }
            let output_text = content_blocks_to_text(&r.content);
            debug!(
                tool_name,
                call_id,
                is_error,
                content_count = r.content.len(),
                "MCP tool call completed"
            );
            build_success_result(
                call_id,
                server_label,
                tool_name,
                arguments_string,
                &output_text,
                is_error,
            )
        },
        Err(e) => {
            warn!(tool_name, call_id, error = %e, "MCP tool call failed");
            build_error_result(call_id, server_label, tool_name, arguments_string, &e.to_string())
        },
    }
}

/// Execute a single MCP tool call.
#[expect(clippy::too_many_lines, reason = "linear validation + async call")]
async fn execute_single_call(
    tool_call: &serde_json::Value,
    tool_map: &HashMap<(String, String), serde_json::Value>,
    timeout: Duration,
    allow_loopback: bool,
) -> Option<McpCallResult> {
    let tool_name = tool_call.get("name").and_then(serde_json::Value::as_str)?;
    let call_id = tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    let entry = match resolve_tool_entry(tool_map, tool_name, call_id) {
        Ok(e) => e,
        Err(opt) => return opt.map(|b| *b),
    };
    let server_url = entry.get("server_url").and_then(serde_json::Value::as_str)?;
    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization").and_then(serde_json::Value::as_str);
    let (arguments, arguments_string) = match parse_call_arguments(tool_call, call_id, server_label, tool_name) {
        Ok(r) => r,
        Err(r) => return Some(*r),
    };

    debug!(tool_name, server_label, call_id, "executing MCP tool call");

    let result = mcp_client::call_tool(
        server_url,
        headers,
        authorization,
        tool_name,
        arguments,
        timeout,
        allow_loopback,
    )
    .await;
    Some(process_call_result(
        result,
        call_id,
        server_label,
        tool_name,
        &arguments_string,
    ))
}

// -----------------------------------------------------------------------------
// Result Construction
// -----------------------------------------------------------------------------

/// Extract text from rmcp `ContentBlock` values, joining
/// multiple text blocks with newlines.
fn content_blocks_to_text(blocks: &[rmcp::model::ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            rmcp::model::ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build result structs for a successful MCP call.
#[expect(clippy::too_many_arguments, reason = "all args needed for result construction")]
#[expect(clippy::too_many_lines, reason = "success/error branches expand the json! blocks")]
fn build_success_result(
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
    output_text: &str,
    is_error: bool,
) -> McpCallResult {
    let message = serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": if is_error {
            format!("Error: {output_text}")
        } else {
            output_text.to_owned()
        },
    });

    let output_item = if is_error {
        serde_json::json!({
            "type": "mcp_call",
            "id": call_id,
            "approval_request_id": null,
            "server_label": server_label,
            "name": tool_name,
            "arguments": arguments,
            "output": output_text,
            "error": output_text,
        })
    } else {
        serde_json::json!({
            "type": "mcp_call",
            "id": call_id,
            "approval_request_id": null,
            "server_label": server_label,
            "name": tool_name,
            "arguments": arguments,
            "output": output_text,
        })
    };

    McpCallResult { message, output_item }
}

/// Build an error result for a tool call that was dropped
/// (task panic, cancellation, or missing fields).
fn error_result_for_dropped_call(tool_call: &serde_json::Value, reason: &str) -> McpCallResult {
    let call_id = tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let tool_name = tool_call
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    build_error_result(call_id, "unknown", tool_name, "", reason)
}

/// Build result structs for a failed MCP call.
fn build_error_result(
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
    error_message: &str,
) -> McpCallResult {
    let message = serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": format!("Error: {error_message}"),
    });

    let output_item = serde_json::json!({
        "type": "mcp_call",
        "id": call_id,
        "approval_request_id": null,
        "server_label": server_label,
        "name": tool_name,
        "arguments": arguments,
        "output": "",
        "error": error_message,
    });

    McpCallResult { message, output_item }
}
