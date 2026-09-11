// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Agentic loop controller for the Responses API pipeline.
//!
//! Manages the inference loop lifecycle: iteration counting,
//! tool-choice reset, exit conditions, and the loop/done signal
//! available to `iterative_request_router` step transitions.
//!
//! Classifies each round's output by dispatch target to make the
//! loop decision, but does **not** execute any call — MCP execution
//! is handled by `openai_mcp_dispatch`, web search by
//! `openai_web_search`, and file search by
//! `openai_file_search_callout`. As the sole owner it runs before
//! every dispatcher in the response phase, so it is the central
//! authority that decides loop-vs-done; dispatchers only consume the
//! calls the owner has already routed to them.
//!
//! # Loop control
//!
//! Writes `filter_results` during `on_response_body` where
//! `iterative_request_router` evaluates step transitions:
//!   - `openai_agentic_loop.action = "loop"` — at least one *dispatchable* call present (web search, file search, or an
//!     MCP `function_call`), loop back
//!   - `openai_agentic_loop.action = "done"` — no dispatchable calls (empty, model-owned finish, or client-only
//!     `function_call`s), exit to client
//!
//! A `function_call` that resolves to no configured MCP tool is
//! client-owned: no local dispatcher can execute it, so a round
//! carrying only client calls exits as `done` and returns those calls
//! to the client, rather than looping uselessly to the iteration cap.
//!
//! # Tool call extraction
//!
//! For non-streaming responses,
//! this filter parses the response body JSON and routes each `output`
//! item to its dispatch target: `function_call` items into
//! `state.tool_calls`, `web_search_call` items into
//! `state.web_search_calls`, and each pending `file_search_call` item —
//! appended to the canonical `state.accumulated_output` — into a
//! `state.file_search_assignments` entry recording its absolute output
//! index and synthesis origin. Client `function_call` and `reasoning`
//! items are appended to `state.messages` so the model sees its own
//! calls on re-entry. Hosted `web_search_call` and `file_search_call`
//! items are **not** valid `OpenResponses` input (issue #808), so they
//! never enter `state.messages`; `web_search_call`s remain in
//! `state.web_search_calls` and `file_search_call`s are reached by index
//! through `state.file_search_assignments`, for `openai_web_search` /
//! `openai_file_search_callout` to dispatch and bridge into backend
//! history, and reach the client only through `state.accumulated_output`.
//!
//! For streaming responses, `stream_events` populates
//! `state.tool_calls` via SSE event parsing. When the body is
//! `None` at end-of-stream (consumed by streaming filters), this
//! filter skips body parsing and checks `state.tool_calls` as-is.
//!
//! `on_request_body` handles iteration bookkeeping: clearing stale
//! tool calls, web search calls, and file search assignments from the
//! previous round and resetting `tool_choice` to `"auto"` on re-entry.
//! The client's `parallel_tool_calls` value is preserved across every
//! round.
//!
//! # Filter order
//!
//! For tool execution, it must appear after `openai_web_search`,
//! `openai_mcp_dispatch`, and `openai_file_search_callout` and before
//! `openai_responses_proxy`. Response filters execute in reverse
//! order, so the owner parses and classifies each round's output
//! *before* the dispatchers run, routing every call to the vector the
//! matching dispatcher consumes. Because the owner is the sole filter
//! that publishes the IRR transition, `on_result` keys **only** on
//! `openai_agentic_loop.action`; the dispatchers publish no action of
//! their own.
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
//!       - filter: openai_file_search_callout
//!       - filter: openai_agentic_loop
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
//!       - filter: openai_agentic_loop
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
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, Rejection,
    SubRequestResponseMode, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde_json::{Value, json};
use tracing::{debug, trace};

use self::config::{AgenticLoopConfig, build_config};
use super::{
    error::responses_error_rejection,
    file_search_callout::{
        ensure_public_output_item_ids_in_response, has_file_search_tool, is_file_search_function_call,
        is_pending_file_search_call, translate_function_calls_to_file_search,
    },
    mcp_classify::{McpDisposition, classify_mcp},
    mcp_dispatch::{configured_max_calls_per_round as configured_mcp_max_calls, prepare_response_round},
    openai_mcp_tool_resolve::McpToolIndex,
    state::{
        DispatchFailure, FileSearchAssignment, McpApprovalState, ResponsesState, SynthesisKind,
        current_round_file_search_admissions,
    },
    stream_events::{encode_local_completion, encode_local_error},
    usage::merge_usage,
    web_search::configured_max_calls_per_round as configured_web_max_calls,
};
use crate::http_hop::{connection_nominates_header, is_hop_by_hop};

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
        // runs after `openai_responses_proxy` has selected the typed transport
        // and after `openai_stream_events` has published whether a logical-stream
        // finalizer is armed, so both facts are observable here.
        //
        // When the sub-request will commit a typed stream (an effective
        // `"stream": true` request, for which `openai_responses_proxy` selects
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
                    "openai_agentic_loop with a streaming openai_responses_proxy sub-request requires \
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

        // A request-phase dispatcher (e.g. `openai_file_search_callout`) that failed
        // records a shared terminal outcome instead of committing a second terminal
        // response. The sole loop owner converts it here — before preparing another
        // inference request — into a buffered JSON rejection (pre-commitment) or a
        // logical-stream SSE error (post-commitment). See issue #1046.
        if let Some(failure) = state.dispatch_failure.take() {
            return convert_dispatch_failure(ctx, state, &failure);
        }

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

        if let Err(failure) = prepare_dispatcher_round(ctx, &mut state) {
            return finish_response_failure(ctx, state, &failure);
        }

        let result = evaluate_loop_decision(ctx, &mut state, body, &self.config)?;
        ctx.extensions.insert(state);
        Ok(result)
    }
}

/// Apply dispatcher-specific response validation before the sole loop decision.
fn prepare_dispatcher_round(ctx: &HttpFilterContext<'_>, state: &mut ResponsesState) -> Result<(), DispatchFailure> {
    if let Some(max_calls) = configured_web_max_calls(ctx)
        && state.web_search_calls.len() > max_calls
    {
        return Err(DispatchFailure {
            status: 502,
            code: "server_error",
            message: "model response exceeded the configured web-search call limit".to_owned(),
        });
    }
    // The configured dispatcher publishes its cap during request processing.
    // Direct owner tests and pipelines without that optional metadata still
    // need MCP ownership/approval preparation, so only the cap becomes
    // effectively unbounded when the metadata is absent.
    prepare_response_round(state, configured_mcp_max_calls(ctx).unwrap_or(usize::MAX))?;
    Ok(())
}

/// Return a response-phase dispatcher validation failure through the owner.
fn finish_response_failure(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
    failure: &DispatchFailure,
) -> Result<FilterAction, FilterError> {
    state.tool_calls.clear();
    state.web_search_calls.clear();
    state.file_search_assignments.clear();
    if request_is_streaming(&state) {
        end_stream_with_error(ctx, &mut state, failure.code, &failure.message)?;
        ctx.extensions.insert(state);
        return Ok(FilterAction::Continue);
    }
    ctx.extensions.insert(state);
    Ok(FilterAction::Reject(responses_error_rejection(
        failure.status,
        failure.code,
        &failure.message,
    )))
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
    state.file_search_assignments.clear();
    let streaming = request_is_streaming(&state);
    let mut body = None;
    if !streaming && let Err(rejection) = state.finalize_response_body(&mut body) {
        ctx.extensions.insert(state);
        return Ok(rejection);
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

/// Convert a dispatcher's shared terminal outcome into the correct client-facing
/// terminal for the active transport.
///
/// A request-phase dispatcher cannot own a terminal response without becoming a
/// second commitment authority, so it records [`DispatchFailure`] and lets the
/// loop owner decide the wire form here (issue #1046). Buffered rounds have not
/// committed anything, so they short-circuit with the JSON error envelope.
/// Streaming rounds have already committed `text/event-stream`, so the owner
/// appends an SSE `error` event to the live logical stream, mirroring
/// [`finish_deferred_local_response`]'s request-phase terminal emission.
fn convert_dispatch_failure(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
    failure: &DispatchFailure,
) -> Result<FilterAction, FilterError> {
    // The loop terminates here; drop this round's dispatch bookkeeping.
    state.tool_calls.clear();
    state.web_search_calls.clear();
    state.file_search_assignments.clear();
    let streaming = request_is_streaming(&state);
    if streaming {
        // No `deferred_stream_done` here: a terminal SSE `error` frame is the stream
        // terminator and is never followed by a `[DONE]` sentinel (see
        // `encode_local_error`). Only skip persistence of the failed round.
        ctx.set_metadata("responses.skip_persist", "true");
    }
    ctx.extensions.insert(state);
    set_action(ctx, ACTION_DONE)?;

    if streaming {
        let body = encode_local_error(ctx, failure.code, &failure.message);
        let mut response = Rejection::status(200)
            .with_header("content-type", "text/event-stream")
            .preserving_keepalive();
        if let Some(body) = body {
            response = response.with_body(body);
        }
        return Ok(FilterAction::Reject(response));
    }
    Ok(FilterAction::Reject(responses_error_rejection(
        failure.status,
        failure.code,
        &failure.message,
    )))
}

/// Preserve a model-owned incomplete response before validating tool ownership.
fn finish_incomplete_round(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
    body: &mut Option<Bytes>,
) -> Result<FilterAction, FilterError> {
    ctx.set_metadata(META_STATUS, "incomplete");
    // A streamed `incomplete` terminal (status == "incomplete", e.g. truncated at
    // max_output_tokens) is finalized by `openai_stream_events` from
    // `accumulated_output`; the draining finalizer must not run on that path (see
    // `request_is_streaming`). Only the buffered path serializes into `body`.
    let rewrites_buffered_body = !request_is_streaming(&state) && state.response_object.is_object();
    if rewrites_buffered_body && let Err(rejection) = state.finalize_response_body(body) {
        ctx.extensions.insert(state);
        return Ok(rejection);
    }
    if rewrites_buffered_body {
        clear_rewritten_response_headers(ctx);
    }
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
    if request_is_streaming(&state) {
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
        // No dispatcher runs on a non-dispatchable (terminal) round, so drop the
        // file-search assignments and their synthesis queue: nothing will complete
        // those items, so `openai_stream_events` must not synthesize a lifecycle.
        state.file_search_assignments.clear();
        state.pending_local_tool_synthesis.clear();
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
    code: &str,
    message: &str,
) -> Result<(), FilterError> {
    state.tool_calls.clear();
    state.web_search_calls.clear();
    state.file_search_assignments.clear();
    ctx.set_metadata("responses.stream_error_code", code.to_owned());
    ctx.set_metadata("responses.stream_error_message", message.to_owned());
    ctx.set_metadata("responses.skip_persist", "true");
    set_action(ctx, ACTION_DONE)
}

// -----------------------------------------------------------------------------
// Request-Side Bookkeeping
// -----------------------------------------------------------------------------

/// Prepare state for the current iteration: clear stale tool calls and, on
/// re-entry, reset `tool_choice`, replay the client's end-to-end headers, and
/// set `Content-Type` (subrequests do not inherit the original client header).
fn prepare_iteration(ctx: &mut HttpFilterContext<'_>, state: &mut ResponsesState) {
    state.tool_calls.clear();
    state.web_search_calls.clear();

    if state.iteration > 0 {
        let original = std::mem::replace(&mut state.tool_choice, json!("auto"));
        state.original_tool_choice.get_or_insert(original);
        set_request_body_field(state, "tool_choice", json!("auto"));
        preserve_original_request_headers(ctx);
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
    }
}

/// Restore end-to-end client headers after the iterative router isolates a
/// transitioned step. The owner builds every continuation inference request
/// (#1046), so it replays the credentials and tenancy headers the next backend
/// round needs. Headers tied to the original wire representation or request
/// identity cannot be replayed after the continuation body is rewritten.
fn preserve_original_request_headers(ctx: &mut HttpFilterContext<'_>) {
    let Some(state) = ctx.extensions.get::<IterationState>() else {
        return;
    };
    if state.iteration() == 0 {
        return;
    }
    let headers = state
        .original_request
        .headers
        .iter()
        .filter(|(name, _)| {
            should_replay_original_header(name) && !connection_nominates_header(&state.original_request.headers, name)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Vec<_>>();
    ctx.request_headers_to_set.extend(headers);
}

/// Whether a header remains valid after the continuation body is rewritten.
fn should_replay_original_header(name: &http::header::HeaderName) -> bool {
    !praxis_core::reserved_headers::is_reserved(name.as_str())
        && !is_hop_by_hop(name.as_str())
        && !matches!(
            name.as_str(),
            "accept-encoding"
                | "content-encoding"
                | "content-length"
                | "content-md5"
                | "digest"
                | "expect"
                | "host"
                | "idempotency-key"
                | "signature"
                | "signature-input"
        )
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

/// Whether the client requested a streaming (`"stream": true`) response.
///
/// On the streaming path `openai_stream_events` owns the terminal SSE frame and
/// the persisted snapshot, rebuilding both from
/// [`ResponsesState::accumulated_output`] in `canonicalize_logical_response`. The
/// agentic loop must therefore NOT run the terminal
/// [`ResponsesState::finalize_response_body`] on that path: it drains
/// `accumulated_output` via `mem::take`, which would strand `stream_events` with
/// an empty snapshot and emit an empty `response.completed` frame.
fn request_is_streaming(state: &ResponsesState) -> bool {
    state.request_body.get("stream").and_then(Value::as_bool) == Some(true)
}

// -----------------------------------------------------------------------------
// Loop Decision
// -----------------------------------------------------------------------------

/// Return whether the completed round produced at least one call some local
/// dispatcher can execute this request.
///
/// `openai_agentic_loop` is the sole loop authority (§3): it emits
/// `action = loop` only when a dispatcher can act on the round's output.
/// Web-search calls and pending file-search calls are dispatchable — a
/// dedicated callout dispatcher consumes each. A completed file-search call is
/// already terminal and must not trigger another round. A `function_call` is
/// dispatchable only when it resolves to a configured MCP tool (auto or
/// approval-required, both server-owned); a client-owned `function_call`
/// resolves to no dispatcher, so a round carrying only client calls must
/// terminate as `done` rather than loop uselessly to the `max_infer_iters` cap.
fn has_dispatchable_calls(state: &ResponsesState) -> bool {
    if !state.web_search_calls.is_empty() || !state.file_search_assignments.is_empty() {
        return true;
    }
    if state.tool_calls.is_empty() || state.mcp_tool_map.is_empty() {
        return false;
    }
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    state
        .tool_calls
        .iter()
        .any(|call| classify_mcp(call, &tool_index) != McpDisposition::NotMcp)
}

/// Decide the loop outcome: done (no dispatchable tool calls or model-owned
/// finish), 508 (iteration limit), or loop (continue to tool execution).
fn evaluate_loop_decision(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    body: &mut Option<Bytes>,
    config: &AgenticLoopConfig,
) -> Result<FilterAction, FilterError> {
    if !has_dispatchable_calls(state) {
        trace!("no dispatchable tool calls, signaling done");
        // Streaming terminals are owned by `openai_stream_events`, which rebuilds
        // the final frame and stored snapshot from `accumulated_output`; running
        // the draining finalizer here would leave it empty (see
        // `request_is_streaming`). Only the buffered path finalizes into `body`.
        let rewrites_buffered_body = !request_is_streaming(state) && state.response_object.is_object();
        if rewrites_buffered_body && let Err(rejection) = state.finalize_response_body(body) {
            return Ok(rejection);
        }
        if rewrites_buffered_body {
            clear_rewritten_response_headers(ctx);
        }
        return set_done(ctx);
    }
    match check_exit_conditions(state, config) {
        Some(ExitReason::IterationLimit) => end_at_iteration_limit(ctx, state),
        None => {
            state.iteration += 1;
            let (tc, wsc, fsc) = (
                state.tool_calls.len(),
                state.web_search_calls.len(),
                state.file_search_assignments.len(),
            );
            debug!(
                iteration = state.iteration,
                tc, wsc, fsc, "pending calls, signaling loop"
            );
            // A loop-back round's response body is discarded by the router (the
            // next request body comes from `next_iteration_body`/the prior request,
            // and the loop response is retained only in the router-internal
            // `previous_response`, which no Responses filter reads). Finalizing here
            // would be dead work — and would consume `accumulated_output` mid-loop.
            // Terminal finalization happens once, on loop exit.
            set_action(ctx, ACTION_LOOP)?;
            Ok(FilterAction::Continue)
        },
    }
}

/// Remove representation metadata after replacing a buffered response body.
fn clear_rewritten_response_headers(ctx: &mut HttpFilterContext<'_>) {
    let Some(response) = &mut ctx.response_header else {
        return;
    };
    let mut changed = false;
    for name in [
        http::header::CONTENT_ENCODING,
        http::header::CONTENT_LENGTH,
        http::header::CONTENT_RANGE,
        http::header::ETAG,
        http::header::LAST_MODIFIED,
    ] {
        changed |= response.headers.remove(name).is_some();
    }
    ctx.response_headers_modified |= changed;
}

/// Terminate at the iteration cap using the transport that is still writable.
fn end_at_iteration_limit(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
) -> Result<FilterAction, FilterError> {
    if request_is_streaming(state) {
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
    let Some(mut response) = response else {
        state.response_object = Value::Null;
        state.tool_calls.clear();
        return;
    };
    // Normalize private `function_call(name=file_search)` into canonical
    // `file_search_call`, gated on an actually configured hosted file-search
    // tool. The returned round-local indices identify the normalized items so
    // their assignment records the correct synthesis origin (§ issue #1046).
    let private_indices = if has_file_search_tool(state) {
        translate_function_calls_to_file_search(&mut response)
    } else {
        Vec::new()
    };
    // Stamp stable synthetic IDs on any id-less output items before accumulation,
    // so the public response never ships an item without an id (issue #955). Runs
    // after normalization so translated file-search calls are seen as such.
    ensure_public_output_item_ids_in_response(&mut response);
    collect_output_items(&response, state, &private_indices);
    if let Some(usage) = response.get("usage").filter(|u| !u.is_null()) {
        merge_usage(&mut state.usage, usage);
    }
    state.response_object = response;
}

/// Return whether one model round mixed server-owned MCP, web-search, or pending
/// file-search calls with calls that must be executed by the API client. The
/// current IRR continuation cannot execute the former without sending the
/// latter back to inference as an unresolved call, so fail before any external
/// side effect.
fn has_mixed_function_call_ownership(state: &ResponsesState) -> bool {
    let mut has_server = !state.web_search_calls.is_empty() || !state.file_search_assignments.is_empty();
    // Scan this round's items in `accumulated_output` rather than
    // `response_object["output"]`: `collect_streaming_output_items` drains the
    // streamed round out of the response object into the accumulator before this
    // check runs, so `output_items()` is already empty on the streaming path.
    let mut has_client = state
        .accumulated_output
        .get(
            state
                .current_round_output_start
                .unwrap_or(state.accumulated_output.len())..,
        )
        .unwrap_or_default()
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
///
/// `private_indices` holds the ascending round-local indices that
/// [`translate_function_calls_to_file_search`] normalized from private
/// `function_call(name=file_search)` items this round, so each recorded
/// [`FileSearchAssignment`] carries the correct synthesis origin.
#[expect(
    clippy::too_many_lines,
    reason = "linear per-item classification of one response's output"
)]
fn collect_output_items(response: &Value, state: &mut ResponsesState, private_indices: &[usize]) {
    let Some(Value::Array(output)) = response.get("output") else {
        return;
    };
    state.current_round_output_start = Some(state.accumulated_output.len());
    let mut pending_file_search: Vec<(usize, SynthesisKind)> = Vec::new();
    for (round_index, item) in output.iter().enumerate() {
        let absolute_index = state.accumulated_output.len();
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
                // openai_web_search dispatch consumes `web_search_calls` and
                // appends a backend-valid function_call/function_call_output
                // bridge for the next inference step.
                state.web_search_calls.push(item.clone());
                state.persisted_messages.push(item.clone());
            },
            Some("file_search_call") => {
                // Like a hosted web_search_call, a file_search_call is not valid
                // OpenResponses input, so it must not enter `messages`. The sole
                // parse owner records its absolute index as a
                // `FileSearchAssignment`; openai_file_search_callout drains those
                // at request-body EOS, runs the vector-store callouts, and mutates
                // the indexed accumulator item in place.
                state.persisted_messages.push(item.clone());
                if is_pending_file_search_call(item) {
                    let synthesis = if private_indices.binary_search(&round_index).is_ok() {
                        SynthesisKind::Private
                    } else {
                        SynthesisKind::Native
                    };
                    pending_file_search.push((absolute_index, synthesis));
                }
            },
            _ => {},
        }
    }
    record_file_search_assignments(state, pending_file_search);
}

/// Record file-search assignments for the dispatcher, gated on the shared
/// built-in tool-call budget.
///
/// The parse owner records one [`FileSearchAssignment`] per pending file-search
/// call it accumulated this round. When the response-wide built-in budget
/// (`max_tool_calls`) is already exhausted, no dispatch may run, so every pending
/// call is terminalized to `incomplete` in place and no assignment is recorded —
/// leaving `has_dispatchable_calls` free to exit the loop. The dispatcher applies
/// only its independent per-continuation server cap to the recorded assignments.
fn record_file_search_assignments(state: &mut ResponsesState, pending: Vec<(usize, SynthesisKind)>) {
    if pending.is_empty() {
        return;
    }
    let candidates = pending
        .into_iter()
        .map(|(output_index, synthesis)| FileSearchAssignment {
            output_index,
            synthesis,
        })
        .collect::<Vec<_>>();
    let admissions = current_round_file_search_admissions(state, &candidates);
    for (index, assignment) in candidates.into_iter().enumerate() {
        let admitted = admissions.get(index).copied().unwrap_or(false);
        if admitted {
            state.file_search_assignments.push(assignment);
        } else {
            terminalize_file_search_item(state, assignment.output_index);
        }
    }
}

/// Mark the `file_search_call` at `index` in `accumulated_output` `incomplete`,
/// dropping any partial results, when it cannot be dispatched.
fn terminalize_file_search_item(state: &mut ResponsesState, index: usize) {
    if let Some(item) = state.accumulated_output.get_mut(index)
        && let Some(object) = item.as_object_mut()
    {
        object.insert("status".to_owned(), Value::String("incomplete".to_owned()));
        object.remove("results");
    }
}

/// Retain the current streamed round after `openai_stream_events` has built
/// its authoritative response object and tool-call list incrementally.
///
/// Mirrors [`collect_output_items`] for the streaming transport: it normalizes
/// private `function_call(name=file_search)` items into canonical
/// `file_search_call`, moves the round out of `response_object["output"]` into
/// `accumulated_output` (no per-item clone of the drained items), records a
/// [`FileSearchAssignment`] per pending file-search call, and queues each
/// reconciled call for EOS lifecycle synthesis by `openai_stream_events`.
#[expect(
    clippy::too_many_lines,
    reason = "linear per-item classification of the streamed round"
)]
fn collect_streaming_output_items(state: &mut ResponsesState) {
    // Normalize private file-search `function_call`s in the streamed response
    // object before draining it. The returned round-local indices tag the
    // synthesis origin (private openings were suppressed live and must be
    // reproduced; native openings already streamed).
    let private_indices = if has_file_search_tool(state) {
        translate_function_calls_to_file_search(&mut state.response_object)
    } else {
        Vec::new()
    };
    // `openai_stream_events` built `tool_calls` from the raw streamed round
    // before normalization, so any private `function_call(name=file_search)` is
    // still present there as a client-looking call. It is now a
    // `file_search_call` in `response_object`/`accumulated_output`, so drop it
    // from `tool_calls` to mirror the buffered path, where `collect_output_items`
    // builds `tool_calls` from the already-normalized response. Otherwise
    // `has_mixed_function_call_ownership` misreads a pure file-search round (no
    // MCP tool map, a recorded assignment) as mixed client/server ownership.
    if !private_indices.is_empty() {
        state.tool_calls.retain(|call| !is_file_search_function_call(call));
    }
    // Stamp stable synthetic IDs on any id-less streamed items before draining the
    // round into the accumulator (issue #955), mirroring the buffered path.
    ensure_public_output_item_ids_in_response(&mut state.response_object);
    let base = state.accumulated_output.len();
    state.current_round_output_start = Some(base);
    // Move the round out (leaving a valid `[]`), so each item is routed to its
    // last consumer by move; only earlier consumers of a shared item clone.
    let round = std::mem::take(state.output_items_mut());
    let mut pending_file_search: Vec<(usize, SynthesisKind)> = Vec::new();
    for (round_index, item) in round.into_iter().enumerate() {
        let absolute_index = state.accumulated_output.len();
        match item.get("type").and_then(Value::as_str) {
            Some("function_call" | "reasoning") => {
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
                state.accumulated_output.push(item);
            },
            Some("web_search_call") => {
                // Mirror `collect_output_items`: a hosted web_search_call is not
                // a valid OpenResponses input item (issue #808), so it must not
                // enter `messages`. The openai_web_search dispatch consumes
                // `web_search_calls` and appends a backend-valid
                // function_call/function_call_output bridge for the next round.
                state.web_search_calls.push(item.clone());
                state.persisted_messages.push(item.clone());
                state.accumulated_output.push(item);
            },
            Some("file_search_call") => {
                let is_private = private_indices.binary_search(&round_index).is_ok();
                let synthesis = if is_private {
                    SynthesisKind::Private
                } else {
                    SynthesisKind::Native
                };
                // Queue EOS synthesis unless a NATIVE call whose terminal
                // lifecycle the provider already streamed live — re-synthesizing
                // its tail would emit a duplicate `output_item.done` (#313 P1).
                // Private calls (opening suppressed) always synthesize.
                let observed_provider_terminal = item
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| state.provider_streamed_terminal_ids.contains(id));
                if is_private || !observed_provider_terminal {
                    state.pending_local_tool_synthesis.push((absolute_index, synthesis));
                }
                if is_pending_file_search_call(&item) {
                    pending_file_search.push((absolute_index, synthesis));
                }
                state.persisted_messages.push(item.clone());
                state.accumulated_output.push(item);
            },
            _ => {
                state.accumulated_output.push(item);
            },
        }
    }
    record_file_search_assignments(state, pending_file_search);
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
