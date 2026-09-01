// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Filter 8: execute MCP tool calls against upstream MCP servers.
//!
//! Operates in two phases within an
//! `iterative_request_router` inference step:
//!
//! 1. **Response path** (`on_response_body`): after `openai_agentic_loop` extracts model-produced function calls,
//!    identifies calls backed by [`ResponsesState::mcp_tool_map`], checks approval policies, and writes
//!    `openai_mcp_dispatch.action = "loop"` to filter results.
//! 2. **Request-body path** (`on_request_body`, next IRR iteration): executes pending MCP calls via
//!    [`mcp_client::call_tool`] and appends results to `messages`, `persisted_messages`, and `output_items` before
//!    `openai_responses_proxy` serializes the next inference request.
//!
//! # Pipeline dependencies
//!
//! - **`mcp_tool_resolve`** must run before this filter so that [`ResponsesState::mcp_tool_map`] is populated.
//! - **`openai_stream_events`** (or equivalent accumulator) must populate [`ResponsesState::tool_calls`] from the
//!   upstream response. Currently only `function_call` events are accumulated; native `mcp_call` events require either
//!   `mcp_tool_resolve` rewriting MCP tools into function tools or the accumulator adding `mcp_call` support.
//! - **`openai_agentic_loop`** must run after this filter in request order, so response order is `openai_agentic_loop`
//!   then `openai_mcp_dispatch`.
//! - The IRR transition must match `openai_mcp_dispatch.action = "loop"` and target the same inference step.
//!
//! Ordinary client-side function calls do not match the MCP tool map,
//! so this filter reports `done` and returns them to the client.
//!
//! [`ResponsesState::tool_calls`]: super::state::ResponsesState
//! [`ResponsesState::mcp_tool_map`]: super::state::ResponsesState
//! [`filter_results`]: HttpFilterContext::filter_results

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
use super::{openai_mcp_tool_resolve::encode_function_name, state::ResponsesState};
use crate::mcp_client;

/// Filter result key consumed by `iterative_request_router`.
const FILTER_RESULT_KEY: &str = "openai_mcp_dispatch";

/// Continue with another inference iteration after MCP execution.
const ACTION_LOOP: &str = "loop";

/// Return the current model response to the client.
const ACTION_DONE: &str = "done";

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
    fn handle_approval_required(
        ctx: &mut HttpFilterContext<'_>,
        pending: &PendingApproval,
    ) -> Result<FilterAction, FilterError> {
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
            return Ok(FilterAction::Continue);
        };
        state.accumulated_output.push(approval_event);

        ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "done".to_owned());
        set_action(ctx, ACTION_DONE)?;

        Ok(FilterAction::Continue)
    }
}

#[async_trait]
impl HttpFilter for McpDispatchFilter {
    fn name(&self) -> &'static str {
        "openai_mcp_dispatch"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
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
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

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
            state.accumulated_output.push(result.output_item);
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
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }

        if let Some(pending) = find_approval_required(&mcp_calls, &state.mcp_tool_map) {
            return Self::handle_approval_required(ctx, &pending);
        }

        ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "execute_mcp".to_owned());
        set_action(ctx, ACTION_LOOP)?;

        Ok(FilterAction::Continue)
    }
}

/// Publish the dispatch decision for IRR transition evaluation.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_RESULT_KEY)
        .or_default()
        .set("action", action)?;
    Ok(())
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

/// Check whether a tool call is an MCP tool call by matching the
/// encoded function name against the tool map.
fn is_mcp_tool_call(tool_call: &serde_json::Value, tool_map: &HashMap<(String, String), serde_json::Value>) -> bool {
    tool_call
        .get("name")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|name| find_by_encoded_name(tool_map, name).is_some())
}

/// Find an entry in the tool map by encoded function name.
///
/// Computes `encode_function_name(label, tool_name)` for each key
/// and returns the first match along with its key. Warns when
/// multiple entries produce the same encoded name, since routing
/// is first-match and may be nondeterministic.
fn find_by_encoded_name<'a>(
    tool_map: &'a HashMap<(String, String), serde_json::Value>,
    encoded_name: &str,
) -> Option<(&'a (String, String), &'a serde_json::Value)> {
    let mut matches = tool_map
        .iter()
        .filter(|((label, name), _)| encode_function_name(label, name) == encoded_name);
    let first = matches.next()?;
    if matches.next().is_some() {
        warn!(
            encoded_name,
            "multiple entries produce the same encoded tool name; routing to first match"
        );
    }
    Some((first.0, first.1))
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

/// Normalize raw tool call arguments.
///
/// Function calls carry arguments either as a JSON string
/// (e.g., `"{\"city\":\"Paris\"}"`) or directly as a JSON value.
/// Returns `(parsed_value, canonical_string)` or an error for
/// malformed JSON strings.
fn normalize_arguments(raw: &serde_json::Value) -> Result<(serde_json::Value, String), String> {
    match raw {
        serde_json::Value::String(s) => {
            let parsed = serde_json::from_str(s).map_err(|e| format!("malformed tool arguments: {e}"))?;
            Ok((parsed, s.clone()))
        },
        other => Ok((other.clone(), other.to_string())),
    }
}

/// Extract serialised arguments from a tool call value.
///
/// Uses the same string-vs-non-string convention as
/// [`normalize_arguments`] but only produces the canonical string,
/// avoiding the deep clone that full normalization performs on
/// non-string values.
fn extract_arguments(tc: &serde_json::Value) -> String {
    match tc.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Check a single tool call for approval requirement.
///
/// Returns `Some` with approval details if approval is required,
/// or if the encoded tool name is ambiguous across servers.
#[expect(clippy::too_many_lines, reason = "linear validation with clear structure")]
fn check_single_approval(
    tc: &serde_json::Value,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> Option<PendingApproval> {
    let encoded_name = tc.get("name").and_then(serde_json::Value::as_str)?;

    let match_count = tool_map
        .keys()
        .filter(|(label, name)| encode_function_name(label, name) == encoded_name)
        .count();
    if match_count > 1 {
        warn!(
            encoded_name,
            server_count = match_count,
            "ambiguous encoded tool name in approval check; requiring approval"
        );
        return Some(PendingApproval {
            call_id: extract_call_id(tc),
            server_label: "unknown".to_owned(),
            tool_name: encoded_name.to_owned(),
            arguments: extract_arguments(tc),
        });
    }

    let (key, entry) = find_by_encoded_name(tool_map, encoded_name)?;
    let original_tool_name = &key.1;
    if !requires_approval(&parse_approval_policy(entry), original_tool_name) {
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
        tool_name: original_tool_name.to_owned(),
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

/// Resolve an encoded function name to its unique entry, rejecting
/// ambiguity.
#[expect(clippy::type_complexity, reason = "key+value pair needed by callers")]
fn resolve_tool_entry<'a>(
    tool_map: &'a HashMap<(String, String), serde_json::Value>,
    encoded_name: &str,
    call_id: &str,
) -> Result<(&'a (String, String), &'a serde_json::Value), Option<Box<McpCallResult>>> {
    let match_count = tool_map
        .keys()
        .filter(|(label, name)| encode_function_name(label, name) == encoded_name)
        .count();
    if match_count == 0 {
        warn!(encoded_name, "tool not found in mcp_tool_map, skipping");
        return Err(None);
    }
    if match_count > 1 {
        warn!(
            encoded_name,
            server_count = match_count,
            "ambiguous MCP tool name: multiple entries produce this encoded name"
        );
        return Err(Some(Box::new(build_error_result(
            call_id,
            "unknown",
            encoded_name,
            "",
            &format!("ambiguous tool name: {match_count} servers expose '{encoded_name}'"),
        ))));
    }
    find_by_encoded_name(tool_map, encoded_name).ok_or(None)
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
    normalize_arguments(&raw).map_err(|e| {
        warn!(tool_name, error = %e, "malformed JSON in tool call arguments");
        Box::new(build_error_result(
            call_id,
            server_label,
            tool_name,
            raw.as_str().unwrap_or_default(),
            &e,
        ))
    })
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
    let encoded_name = tool_call.get("name").and_then(serde_json::Value::as_str)?;
    let call_id = tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    let (key, entry) = match resolve_tool_entry(tool_map, encoded_name, call_id) {
        Ok(r) => r,
        Err(opt) => return opt.map(|b| *b),
    };
    let original_tool_name = &key.1;
    let server_url = entry.get("server_url").and_then(serde_json::Value::as_str)?;
    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization").and_then(serde_json::Value::as_str);
    let (arguments, arguments_string) = match parse_call_arguments(tool_call, call_id, server_label, original_tool_name)
    {
        Ok(r) => r,
        Err(r) => return Some(*r),
    };

    debug!(
        tool_name = original_tool_name,
        server_label, call_id, "executing MCP tool call"
    );

    let result = mcp_client::call_tool(
        server_url,
        headers,
        authorization,
        original_tool_name,
        arguments,
        timeout,
        allow_loopback,
    )
    .await;
    Some(process_call_result(
        result,
        call_id,
        server_label,
        original_tool_name,
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
