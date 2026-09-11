// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Agentic loop controller for the Responses API pipeline.
//!
//! Manages the inference loop lifecycle: iteration counting,
//! tool-choice reset, exit conditions, and the loop/done signal
//! available to `iterative_request_router` step transitions.
//!
//! Does **not** classify tool calls by type or execute them —
//! MCP classification and execution are handled by
//! `openai_mcp_dispatch`, web search execution by
//! `openai_web_search_dispatch`.
//!
//! # Loop control
//!
//! Writes `filter_results` during `on_response_body` where
//! `iterative_request_router` evaluates step transitions:
//!   - `openai_agentic_loop.action = "loop"` — tool calls present, loop back
//!   - `openai_agentic_loop.action = "done"` — exit to client
//!
//! # Tool call extraction
//!
//! For non-streaming responses,
//! this filter parses the response body JSON and extracts
//! `function_call` items from the `output` array into
//! `state.tool_calls` and `web_search_call` items into
//! `state.web_search_calls`. Client `function_call` and `reasoning`
//! items are appended to `state.messages` so the model sees its own
//! calls on re-entry. Hosted `web_search_call` items are **not** valid
//! `OpenResponses` input (issue #808), so they never enter
//! `state.messages`; they remain in `state.web_search_calls` for
//! `openai_web_search_dispatch` to dispatch and bridge into backend history, and
//! reach the client only through `state.accumulated_output`.
//!
//! For streaming responses, `stream_events` populates
//! `state.tool_calls` via SSE event parsing. When the body is
//! `None` at end-of-stream (consumed by streaming filters), this
//! filter skips body parsing and checks `state.tool_calls` as-is.
//!
//! `on_request_body` handles iteration bookkeeping: clearing stale
//! tool calls and web search calls from the previous round and
//! resetting `tool_choice` to `"auto"` on re-entry. The client's
//! `parallel_tool_calls` value is preserved across every round.
//!
//! # Filter order
//!
//! For tool execution, it must appear after `openai_web_search_dispatch`
//! and `openai_mcp_dispatch` and before `openai_proxy`.
//! Response filters execute in reverse order, so the loop
//! extracts tool calls before dispatch filters classify them and
//! publish the IRR transition.
//!
//! ```yaml
//! filter: iterative_request_router
//! initial_step: inference
//! max_iterations: 11
//! steps:
//!   - name: inference
//!     filters:
//!       - filter: openai_web_search_dispatch
//!         provider: brave
//!         api_key: ${WEB_SEARCH_API_KEY}
//!       - filter: openai_mcp_dispatch
//!       - filter: openai_agentic_loop
//!         max_infer_iters: 10
//!       - filter: openai_proxy
//!       - filter: router
//!         routes:
//!           - cluster: model-backend
//!       - filter: load_balancer
//!         clusters:
//!           - name: model-backend
//!             endpoints: ["127.0.0.1:3001"]
//!     on_result:
//!       - filter: openai_mcp_dispatch
//!         key: action
//!         value: loop
//!         next: inference
//!       - filter: openai_web_search_dispatch
//!         key: action
//!         value: loop
//!         next: inference
//!       - default: true
//!         done: true
//! ```
//!
//! # State dependency
//!
//! Requires [`ResponsesState`] in request extensions. Without it
//! the filter passes through silently. State is created by
//! `openai_validate` for every Responses API create
//! request.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{CONTENT_TYPE, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, SubRequestResponseMode,
    body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde_json::{Value, json};
use tracing::{debug, trace};

use self::config::{AgenticLoopConfig, build_config};
use super::{
    error::responses_error_rejection,
    mcp_tool_resolve::McpToolIndex,
    state::{McpApprovalState, ResponsesState},
    stream_events::encode_local_completion,
    usage::merge_usage,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter results key for the loop control action.
const FILTER_RESULT_KEY: &str = "openai_agentic_loop";

/// Action value signalling a loop-back to `responses_proxy`.
const ACTION_LOOP: &str = "loop";

/// Action value signalling loop exit.
const ACTION_DONE: &str = "done";

/// Metadata key for the response status set on incomplete exits.
const META_STATUS: &str = "responses.status";

// -----------------------------------------------------------------------------
// AgenticLoopFilter
// -----------------------------------------------------------------------------

/// Agentic loop controller for the Responses API pipeline.
///
/// Manages iteration bookkeeping in `on_request_body`, extracts tool
/// calls from non-streaming response bodies, and evaluates loop
/// control in `on_response_body` (end-of-stream), writing
/// `filter_results` for `iterative_request_router` transitions.
///
/// # YAML
///
/// ```yaml
/// filter: openai_agentic_loop
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_agentic_loop
/// max_infer_iters: 10
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_apis::openai::AgenticLoopFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = AgenticLoopFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "openai_agentic_loop");
/// ```
pub struct AgenticLoopFilter {
    /// Parsed and validated configuration.
    config: AgenticLoopConfig,
}

impl AgenticLoopFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown
    /// fields or invalid values.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AgenticLoopConfig = if config.is_null() {
            AgenticLoopConfig::default()
        } else {
            parse_filter_config("openai_agentic_loop", config)?
        };
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AgenticLoopFilter {
    fn name(&self) -> &'static str {
        "openai_agentic_loop"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        // Accept up to the absolute ceiling; the pipeline's body_limits
        // decides the real raw cap for each direction.
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Fail closed on an unsafe terminal-streaming configuration *before* any
        // upstream dispatch. Within each IRR round this filter's `on_request`
        // runs after `openai_proxy` has selected the typed transport
        // and after `openai_stream_events` has published whether a logical-stream
        // finalizer is armed, so both facts are observable here.
        //
        // When the sub-request will commit a typed stream (an effective
        // `"stream": true` request, for which `openai_proxy` selects
        // streaming automatically) but no `openai_stream_events` logical-stream
        // finalizer is present, a loop-terminal error detected later in
        // `on_response_body` cannot reach the client: typed streaming has already
        // committed `response.completed`, so the error would be silently dropped
        // after a truncated success is on the wire. This is a server
        // misconfiguration, so reject with a 500 rather than forward that
        // truncated success. Buffered rounds retain full error handling and are
        // unaffected.
        if ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming {
            // `openai_stream_events` publishes this marker on every armed round
            // (it always composes its stream into one logical Responses
            // lifecycle). Consume it so a `"true"` published by another IRR step
            // cannot satisfy a later step's check; the filter re-publishes it
            // every armed round before this filter reads it.
            let stream_events_armed = ctx.get_metadata("responses.logical_stream") == Some("true");
            ctx.set_metadata("responses.logical_stream", "false");
            if !stream_events_armed {
                return Ok(FilterAction::Reject(responses_error_rejection(
                    500,
                    "server_error",
                    "openai_agentic_loop with a streaming openai_proxy sub-request requires \
                     openai_stream_events in the same step so loop-terminal errors can reach the client",
                )));
            }
        }
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

        let Some(mut state) = ctx.extensions.remove::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if state.deferred_tool_limit_completion || state.mcp_approval_state != McpApprovalState::None {
            return finish_deferred_local_response(ctx, state);
        }

        prepare_iteration(ctx, &mut state);
        trace!(iteration = state.iteration, "openai_agentic_loop on_request_body");
        ctx.extensions.insert(state);
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(mut state) = ctx.extensions.remove::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if let Some(bytes) = body.as_ref() {
            extract_tool_calls_from_body(bytes, &mut state);
        } else if !prepare_streamed_round(ctx, &mut state)? {
            ctx.extensions.insert(state);
            return Ok(FilterAction::Continue);
        }

        if is_finish_reason_length(&state) {
            return finish_incomplete_round(ctx, state, body);
        }

        if has_mixed_function_call_ownership(&state) {
            return reject_mixed_ownership_round(ctx, state);
        }

        let result = evaluate_loop_decision(ctx, &mut state, body, &self.config)?;
        ctx.extensions.insert(state);
        Ok(result)
    }
}

/// Complete a locally terminal round after every request-side dispatcher ran.
fn finish_deferred_local_response(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
) -> Result<FilterAction, FilterError> {
    state.deferred_tool_limit_completion = false;
    state.mcp_approval_state = McpApprovalState::None;
    state.tool_calls.clear();
    state.web_search_calls.clear();
    let streaming = state.request_body.get("stream").and_then(Value::as_bool) == Some(true);
    let mut body = None;
    if !streaming {
        state.finalize_response_body(&mut body);
    }
    ctx.extensions.insert(state);
    if streaming {
        body = encode_local_completion(ctx);
    }
    set_action(ctx, ACTION_DONE)?;

    let mut response = Rejection::status(200)
        .with_header(
            "content-type",
            if streaming {
                "text/event-stream"
            } else {
                "application/json"
            },
        )
        .preserving_keepalive();
    if let Some(body) = body {
        response = response.with_body(body);
    }
    Ok(FilterAction::Reject(response))
}

/// Preserve a model-owned incomplete response before validating tool ownership.
fn finish_incomplete_round(
    ctx: &mut HttpFilterContext<'_>,
    state: ResponsesState,
    body: &mut Option<Bytes>,
) -> Result<FilterAction, FilterError> {
    ctx.set_metadata(META_STATUS, "incomplete");
    state.finalize_response_body(body);
    set_action(ctx, ACTION_DONE)?;
    ctx.extensions.insert(state);
    Ok(FilterAction::Continue)
}

/// Reject a completed round that cannot safely split execution ownership.
fn reject_mixed_ownership_round(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
) -> Result<FilterAction, FilterError> {
    const MESSAGE: &str = "model response mixed server-owned and client-owned tool calls in one round";
    if state.request_body.get("stream").and_then(Value::as_bool) == Some(true) {
        end_stream_with_error(ctx, &mut state, "server_error", MESSAGE)?;
        ctx.extensions.insert(state);
        return Ok(FilterAction::Continue);
    }
    ctx.extensions.insert(state);
    Ok(FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        MESSAGE,
    )))
}

/// Collect an authoritative successful stream or terminate without dispatch.
fn prepare_streamed_round(ctx: &mut HttpFilterContext<'_>, state: &mut ResponsesState) -> Result<bool, FilterError> {
    if !super::streamed_round_is_dispatchable(ctx, state) {
        collect_streaming_output_items(state);
        state.tool_calls.clear();
        state.web_search_calls.clear();
        set_action(ctx, ACTION_DONE)?;
        return Ok(false);
    }
    collect_streaming_output_items(state);
    Ok(true)
}

/// End an already-committed stream with a local SSE error and no side effects.
fn end_stream_with_error(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    code: &'static str,
    message: &'static str,
) -> Result<(), FilterError> {
    state.tool_calls.clear();
    state.web_search_calls.clear();
    ctx.set_metadata("responses.stream_error_code", code);
    ctx.set_metadata("responses.stream_error_message", message);
    ctx.set_metadata("responses.skip_persist", "true");
    set_action(ctx, ACTION_DONE)
}

// -----------------------------------------------------------------------------
// Request-Side Bookkeeping
// -----------------------------------------------------------------------------

/// Prepare state for the current iteration: clear stale tool calls and, on
/// re-entry, reset `tool_choice` and set `Content-Type` (subrequests do not
/// inherit the original client header).
fn prepare_iteration(ctx: &mut HttpFilterContext<'_>, state: &mut ResponsesState) {
    state.tool_calls.clear();
    state.web_search_calls.clear();

    if state.iteration > 0 {
        let original = std::mem::replace(&mut state.tool_choice, json!("auto"));
        state.original_tool_choice.get_or_insert(original);
        set_request_body_field(state, "tool_choice", json!("auto"));
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
    }
}

/// Set a provider-visible request field and record whether its value changed.
fn set_request_body_field(state: &mut ResponsesState, name: &str, value: Value) {
    let Some(obj) = state.request_body.as_object_mut() else {
        return;
    };
    if obj.get(name) != Some(&value) {
        obj.insert(name.to_owned(), value);
        state.mark_request_body_for_rebuild();
    }
}

// -----------------------------------------------------------------------------
// Loop Decision
// -----------------------------------------------------------------------------

/// Decide the loop outcome: done (no tool calls or model-owned finish),
/// 508 (iteration limit), or loop (continue to tool execution).
fn evaluate_loop_decision(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    body: &mut Option<Bytes>,
    config: &AgenticLoopConfig,
) -> Result<FilterAction, FilterError> {
    if state.tool_calls.is_empty() && state.web_search_calls.is_empty() {
        trace!("no tool calls, signaling done");
        state.finalize_response_body(body);
        return set_done(ctx);
    }
    match check_exit_conditions(state, config) {
        Some(ExitReason::IterationLimit) => end_at_iteration_limit(ctx, state),
        None => {
            state.iteration += 1;
            let (tc, wsc) = (state.tool_calls.len(), state.web_search_calls.len());
            debug!(iteration = state.iteration, tc, wsc, "pending calls, signaling loop");
            state.finalize_response_body(body);
            set_action(ctx, ACTION_LOOP)?;
            Ok(FilterAction::Continue)
        },
    }
}

/// Terminate at the iteration cap using the transport that is still writable.
fn end_at_iteration_limit(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
) -> Result<FilterAction, FilterError> {
    if state.request_body.get("stream").and_then(Value::as_bool) == Some(true) {
        end_stream_with_error(ctx, state, "server_error", "agentic loop iteration limit exceeded")?;
        return Ok(FilterAction::Continue);
    }
    Ok(FilterAction::Reject(responses_error_rejection(
        508,
        "server_error",
        "agentic loop iteration limit exceeded",
    )))
}

// -----------------------------------------------------------------------------
// Body Parsing
// -----------------------------------------------------------------------------

/// Extract completed function-call items from a non-streaming response body
/// and populate `state.tool_calls` and `state.messages`.
fn extract_tool_calls_from_body(body: &Bytes, state: &mut ResponsesState) {
    let response = serde_json::from_slice::<Value>(body)
        .ok()
        .filter(is_responses_api_output);
    let Some(response) = response else {
        state.response_object = Value::Null;
        state.tool_calls.clear();
        return;
    };
    collect_output_items(&response, state);
    if let Some(usage) = response.get("usage").filter(|u| !u.is_null()) {
        merge_usage(&mut state.usage, usage);
    }
    state.response_object = response;
}

/// Return whether one model round mixed server-owned MCP or web-search calls
/// with calls that must be executed by the API client. The
/// current IRR continuation cannot execute the former without sending the
/// latter back to inference as an unresolved call, so fail before any external
/// side effect.
fn has_mixed_function_call_ownership(state: &ResponsesState) -> bool {
    let mut has_server = !state.web_search_calls.is_empty();
    let mut has_client = state
        .output_items()
        .iter()
        .any(super::state::is_client_executed_tool_call);
    if state.tool_calls.is_empty() {
        return has_server && has_client;
    }
    if state.mcp_tool_map.is_empty() {
        // With no resolved MCP tools, every function call is client-owned.
        return has_server;
    }
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    for call in &state.tool_calls {
        let is_mcp = call
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|encoded| tool_index.contains(encoded));
        has_server |= is_mcp;
        has_client |= !is_mcp;
    }
    has_server && has_client
}

/// Distribute output items from a parsed response into the accumulator and state vectors.
fn collect_output_items(response: &Value, state: &mut ResponsesState) {
    let Some(Value::Array(output)) = response.get("output") else {
        return;
    };
    state.current_round_output_start = state.accumulated_output.len();
    for item in output {
        state.accumulated_output.push(item.clone());
        match item.get("type").and_then(Value::as_str) {
            Some("function_call")
                if item
                    .get("status")
                    .is_none_or(|v| v.is_null() || v.as_str() == Some("completed")) =>
            {
                state.tool_calls.push(item.clone());
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            Some("reasoning") => {
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            Some("web_search_call") => {
                // A hosted web_search_call is not a valid OpenResponses input
                // item (issue #808), so it must not enter `messages`. The
                // openai_web_search_dispatch dispatch consumes `web_search_calls` and
                // appends a backend-valid function_call/function_call_output
                // bridge for the next inference step.
                state.web_search_calls.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            _ => {},
        }
    }
}

/// Retain the current streamed round after `openai_stream_events` has built
/// its authoritative response object and tool-call list incrementally.
fn collect_streaming_output_items(state: &mut ResponsesState) {
    let output = state.output_items().to_vec();
    state.current_round_output_start = state.accumulated_output.len();
    for item in output {
        // `output` is owned (`to_vec`), so route the owned value into its last
        // consumer by move; only the earlier consumers of a shared item clone.
        // Non-matching items go solely to `accumulated_output` and are moved.
        match item.get("type").and_then(Value::as_str) {
            Some("function_call" | "reasoning") => {
                state.messages.push(item.clone());
                state.accumulated_output.push(item.clone());
                state.persisted_messages.push(item);
            },
            Some("web_search_call") => {
                // Mirror `collect_output_items`: a hosted web_search_call is not
                // a valid OpenResponses input item (issue #808), so it must not
                // enter `messages`. The openai_web_search_dispatch dispatch consumes
                // `web_search_calls` and appends a backend-valid
                // function_call/function_call_output bridge for the next round.
                state.web_search_calls.push(item.clone());
                state.accumulated_output.push(item.clone());
                state.persisted_messages.push(item);
            },
            _ => {
                state.accumulated_output.push(item);
            },
        }
    }
}

/// Check whether a parsed response is a valid Responses API output.
///
/// Returns `false` for error bodies (`"object": "error"`) and
/// responses without the canonical `"object": "response"` marker,
/// preventing usage injection into upstream error responses.
fn is_responses_api_output(response: &Value) -> bool {
    response
        .get("object")
        .and_then(Value::as_str)
        .is_some_and(|v| v == "response")
}

// -----------------------------------------------------------------------------
// Exit Condition Checks
// -----------------------------------------------------------------------------

/// Why the loop should exit early.
enum ExitReason {
    /// The proxy's `max_infer_iters` cap was reached — a
    /// proxy-owned reason, returned as a 508 error.
    IterationLimit,
}

/// Check whether the loop should exit early.
fn check_exit_conditions(state: &ResponsesState, config: &AgenticLoopConfig) -> Option<ExitReason> {
    if state.iteration >= config.max_infer_iters {
        debug!(
            iteration = state.iteration,
            max = config.max_infer_iters,
            "config iteration limit reached"
        );
        return Some(ExitReason::IterationLimit);
    }
    None
}

/// Check whether the response finished due to length limit.
fn is_finish_reason_length(state: &ResponsesState) -> bool {
    state
        .response_object
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "incomplete")
        || state
            .response_object
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(Value::as_str)
            .is_some_and(|r| r == "max_output_tokens")
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Shorthand: set `action = "done"` and return `Continue`.
fn set_done(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
    set_action(ctx, ACTION_DONE)?;
    Ok(FilterAction::Continue)
}

/// Write the loop control action to filter results.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    let results = ctx.filter_results.entry(FILTER_RESULT_KEY).or_default();
    results.set("action", action)?;
    Ok(())
}
