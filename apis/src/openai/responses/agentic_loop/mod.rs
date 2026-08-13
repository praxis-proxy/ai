// SPDX-License-Identifier: MIT
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
//! `openai_web_search`.
//!
//! # Loop control
//!
//! Writes `filter_results` during `on_response_body` where
//! `iterative_request_router` evaluates step transitions:
//!   - `agentic_loop.action = "loop"` — tool calls present, loop back
//!   - `agentic_loop.action = "done"` — exit to client
//!
//! # Non-streaming tool call extraction
//!
//! For non-streaming responses (the only mode supported by IRR),
//! this filter parses the response body JSON and extracts
//! `function_call` items from the `output` array into
//! `state.tool_calls` and `web_search_call` items into
//! `state.web_search_calls`. It also appends these items to
//! `state.messages` so the model sees its own calls on re-entry.
//!
//! For streaming responses (future), `stream_events` populates
//! `state.tool_calls` via SSE event parsing. When the body is
//! `None` at end-of-stream (consumed by streaming filters), this
//! filter skips body parsing and checks `state.tool_calls` as-is.
//!
//! `on_request_body` handles iteration bookkeeping: clearing stale
//! tool calls and web search calls from the previous round,
//! forcing `parallel_tool_calls` to `false` (v1 supports one
//! function call per round), and resetting `tool_choice` to
//! `"auto"` on re-entry.
//!
//! # Filter order
//!
//! For tool execution, it must appear after `openai_web_search`
//! and `openai_mcp_dispatch` and before `openai_responses_proxy`.
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
//!       - filter: openai_web_search
//!         provider: brave
//!         api_key: ${WEB_SEARCH_API_KEY}
//!       - filter: openai_mcp_dispatch
//!       - filter: agentic_loop
//!         max_infer_iters: 10
//!       - filter: openai_responses_proxy
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
//!       - filter: openai_web_search
//!         key: action
//!         value: loop
//!         next: inference
//!       - default: true
//!         done: true
//! ```
//!
//! # Streaming limitation
//!
//! Streaming requests (`stream: true`) are rejected with a 400
//! error. `iterative_request_router` fully buffers all responses
//! within the loop and cannot forward incremental SSE events.
//!
//! # State dependency
//!
//! Requires [`ResponsesState`] in request extensions. Without it
//! the filter passes through silently. State is created by
//! `openai_responses_validate` for every Responses API create
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
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde_json::{Value, json};
use tracing::{debug, trace};

use self::config::{AgenticLoopConfig, build_config};
use super::{error::responses_error_rejection, state::ResponsesState, stream_events::accumulator::merge_usage};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter results key for the loop control action.
const FILTER_RESULT_KEY: &str = "agentic_loop";

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
/// filter: agentic_loop
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: agentic_loop
/// max_infer_iters: 10
/// max_body_bytes: 10485760
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_apis::openai::AgenticLoopFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = AgenticLoopFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "agentic_loop");
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
            parse_filter_config("agentic_loop", config)?
        };
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AgenticLoopFilter {
    fn name(&self) -> &'static str {
        "agentic_loop"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
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

        let Some(mut state) = ctx.extensions.remove::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if state.request_body.get("stream") == Some(&Value::Bool(true)) {
            ctx.extensions.insert(state);
            return Ok(FilterAction::Reject(responses_error_rejection(
                400,
                "invalid_request_error",
                "streaming is not supported with agentic_loop",
                false,
            )));
        }

        prepare_iteration(ctx, &mut state);
        trace!(iteration = state.iteration, "agentic_loop on_request_body");
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

        if let Some(bytes) = body.as_ref()
            && let Err(msg) = extract_tool_calls_from_body(bytes, &mut state)
        {
            ctx.extensions.insert(state);
            return Ok(FilterAction::Reject(responses_error_rejection(
                400,
                "invalid_request_error",
                msg,
                false,
            )));
        }

        let result = evaluate_loop_decision(ctx, &mut state, body, &self.config)?;
        ctx.extensions.insert(state);
        Ok(result)
    }
}

// -----------------------------------------------------------------------------
// Request-Side Bookkeeping
// -----------------------------------------------------------------------------

/// Prepare state for the current iteration: clear stale tool calls,
/// force `parallel_tool_calls=false`, and on re-entry reset
/// `tool_choice` and set `Content-Type` (subrequests do not inherit
/// the original client header).
fn prepare_iteration(ctx: &mut HttpFilterContext<'_>, state: &mut ResponsesState) {
    state.tool_calls.clear();
    state.web_search_calls.clear();
    state.parallel_tool_calls = false;
    set_request_body_field(state, "parallel_tool_calls", Value::Bool(false));

    if state.iteration > 0 {
        state.tool_choice = json!("auto");
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
        finalize_response_body(state, body);
        return set_done(ctx);
    }
    match check_exit_conditions(state, config) {
        Some(ExitReason::FinishReasonLength) => {
            ctx.set_metadata(META_STATUS, "incomplete");
            finalize_response_body(state, body);
            set_action(ctx, ACTION_DONE)?;
            Ok(FilterAction::Continue)
        },
        Some(ExitReason::IterationLimit) => Ok(FilterAction::Reject(responses_error_rejection(
            508,
            "server_error",
            "agentic loop iteration limit exceeded",
            false,
        ))),
        None => {
            state.iteration += 1;
            let (tc, wsc) = (state.tool_calls.len(), state.web_search_calls.len());
            debug!(iteration = state.iteration, tc, wsc, "pending calls, signaling loop");
            finalize_response_body(state, body);
            set_action(ctx, ACTION_LOOP)?;
            Ok(FilterAction::Continue)
        },
    }
}

// -----------------------------------------------------------------------------
// Body Parsing
// -----------------------------------------------------------------------------

/// Extract completed function-call items from a non-streaming
/// response body and populate `state.tool_calls` and
/// `state.messages`.
///
/// Returns `Err` if multiple function calls are found — v1
/// supports exactly one function call per round.
fn extract_tool_calls_from_body(body: &Bytes, state: &mut ResponsesState) -> Result<(), &'static str> {
    let response = serde_json::from_slice::<Value>(body)
        .ok()
        .filter(is_responses_api_output);
    let Some(response) = response else {
        state.response_object = Value::Null;
        state.tool_calls.clear();
        return Ok(());
    };
    collect_output_items(&response, state);
    if let Some(usage) = response.get("usage").filter(|u| !u.is_null()) {
        merge_usage(&mut state.usage, usage);
    }
    state.response_object = response;
    if state.tool_calls.len() > 1 {
        return Err("agentic_loop supports exactly one function call per round");
    }
    Ok(())
}

/// Distribute output items from a parsed response into the accumulator and state vectors.
fn collect_output_items(response: &Value, state: &mut ResponsesState) {
    let Some(Value::Array(output)) = response.get("output") else {
        return;
    };
    for item in output {
        state.accumulated_output.push(item.clone());
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") if item.get("status").and_then(Value::as_str) == Some("completed") => {
                state.tool_calls.push(item.clone());
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            Some("reasoning") => {
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            Some("web_search_call") => {
                state.web_search_calls.push(item.clone());
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            _ => {},
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
    /// The model reported `status: "incomplete"` due to output
    /// token limits — a model-owned reason, passed through as-is.
    FinishReasonLength,
    /// The proxy's `max_infer_iters` cap was reached — a
    /// proxy-owned reason, returned as a 508 error.
    IterationLimit,
}

/// Check whether the loop should exit early.
fn check_exit_conditions(state: &ResponsesState, config: &AgenticLoopConfig) -> Option<ExitReason> {
    if is_finish_reason_length(state) {
        debug!("finish_reason is length, exiting loop as incomplete");
        return Some(ExitReason::FinishReasonLength);
    }
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

/// Build the final response body from accumulated state.
///
/// Replaces `response_object["output"]` with the full
/// `accumulated_output` (all rounds), stamps accumulated usage,
/// and serializes back to body bytes.
fn finalize_response_body(state: &ResponsesState, body: &mut Option<Bytes>) {
    if !state.response_object.is_object() {
        return;
    }
    let mut response = state.response_object.clone();
    if let Some(obj) = response.as_object_mut() {
        if !state.accumulated_output.is_empty() {
            obj.insert("output".to_owned(), Value::Array(state.accumulated_output.clone()));
        }
        if !state.usage.is_null() {
            obj.insert("usage".to_owned(), state.usage.clone());
        }
    }
    if let Ok(serialized) = serde_json::to_vec(&response) {
        *body = Some(Bytes::from(serialized));
    }
}

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
