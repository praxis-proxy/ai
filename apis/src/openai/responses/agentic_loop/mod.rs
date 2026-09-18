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
    SubRequestResponseMode, TrustedHeaderMutation, body::MAX_JSON_BODY_BYTES, parse_filter_config,
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
    openai_mcp_tool_resolve::{McpToolIndex, has_pending_deferred_discovery},
    state::{
        DispatchFailure, FileSearchAssignment, McpApprovalState, OutputAssignment, ResponsesState, SynthesisKind,
        ToolCallAssignment, WebSearchAssignment, current_round_file_search_admissions,
        tool_search_discovery_is_within_budget,
    },
    stream_events::{encode_local_completion, encode_local_error, retained_stream_payload_bytes},
    usage::{merge_usage_owned, merged_usage_json_bytes},
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
/// Also extracts `tool_search_call` items into
/// `ResponsesState.tool_search_calls` so `openai_mcp_dispatch` can
/// load deferred connectors on the next iteration without forwarding
/// those items to the inference backend.
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
/// max_retained_bytes: 67108864
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
        if let Some(action) = admit_retained_payload_budget(ctx, self.config.max_retained_bytes)? {
            return Ok(action);
        }

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

        // StreamBuffer pre-read reaches this hook before `on_request`. Admit the
        // starting state here so a deferred approval cannot execute first.
        if let Some(action) = admit_retained_payload_budget(ctx, self.config.max_retained_bytes)? {
            return Ok(action);
        }
        // Defer the sole request-side mutation until the terminal request
        // serializer, which follows every loop instance in canonical step order.
        // This lets every instance lower the shared budget before local
        // completion or dispatch can commit.
        if ctx.extensions.get::<IterationState>().is_some() {
            ctx.extensions.insert(DeferredAgenticRequestFinish);
            Ok(FilterAction::Continue)
        } else {
            // Unit-level and defensive non-IRR invocation retains the historic
            // direct behavior; production loop pipelines always carry IRR state.
            finish_request_after_dispatch(ctx)
        }
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

        // Several conditionally composed loop instances may touch one request
        // to contribute safety policy. Only the first response callback owns
        // parsing and the loop transition; otherwise the same provider round is
        // accumulated and charged once per configured instance.
        let router_iteration = ctx.extensions.get::<IterationState>().map(IterationState::iteration);
        if !claim_response_round(ctx, router_iteration) {
            return Ok(FilterAction::Continue);
        }

        process_response_body(ctx, body, &self.config)
    }
}

/// Marker consumed by the terminal request serializer after every loop instance
/// has admitted its configured retained-payload limit.
struct DeferredAgenticRequestFinish;

/// Return response state to the request extensions across the large-state box
/// allocation boundary without inflating the response hook's stack frame.
#[inline(never)]
fn restore_response_state(ctx: &mut HttpFilterContext<'_>, state: ResponsesState) {
    ctx.extensions.insert(state);
}

/// Extract one provider round before applying ownership and budget decisions.
fn process_response_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    config: &AgenticLoopConfig,
) -> Result<FilterAction, FilterError> {
    let Some(mut state) = ctx.extensions.remove::<ResponsesState>() else {
        return Ok(FilterAction::Continue);
    };

    if let Some(bytes) = body.as_ref() {
        if let Err(failure) = extract_tool_calls_from_body(bytes, &mut state) {
            return finish_response_failure(ctx, state, &failure, true);
        }
    } else {
        match prepare_streamed_round(ctx, &mut state) {
            Ok(true) => {},
            Ok(false) => {
                set_action(ctx, ACTION_DONE)?;
                restore_response_state(ctx, state);
                return Ok(FilterAction::Continue);
            },
            Err(failure) => return finish_response_failure(ctx, state, &failure, true),
        }
    }

    continue_response_round(ctx, body, state, config)
}

/// Apply ownership, dispatcher, budget, and loop decisions to one extracted round.
fn continue_response_round(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    mut state: ResponsesState,
    config: &AgenticLoopConfig,
) -> Result<FilterAction, FilterError> {
    if is_finish_reason_length(&state) {
        return finish_incomplete_round(ctx, state, body);
    }

    if has_mixed_function_call_ownership(&state) {
        return reject_mixed_ownership_round(ctx, state);
    }

    if let Err(failure) = prepare_dispatcher_round(ctx, &mut state) {
        let retained_budget_failure = state.retained_payload_failed;
        return finish_response_failure(ctx, state, &failure, retained_budget_failure);
    }

    let stream_payload_bytes = retained_stream_payload_bytes(ctx).unwrap_or(usize::MAX);
    if !state.can_replace_retained_payload(0, 0, stream_payload_bytes) {
        if request_is_streaming(&state) {
            state.discard_payload_for_budget_error();
        }
        return finish_response_failure(ctx, state, &retained_payload_failure(), true);
    }

    let result = evaluate_loop_decision(ctx, &mut state, body, config)?;
    restore_response_state(ctx, state);
    Ok(result)
}

/// Claim sole ownership of one IRR provider response while allowing every
/// configured loop instance to participate in request-side budget admission.
fn claim_response_round(ctx: &mut HttpFilterContext<'_>, iteration: Option<u32>) -> bool {
    let Some(iteration) = iteration else {
        return true;
    };
    let marker = iteration.to_string();
    if ctx.get_metadata("responses.agentic_response_processed_iteration") == Some(marker.as_str()) {
        return false;
    }
    ctx.set_metadata("responses.agentic_response_processed_iteration", marker);
    true
}

/// Return whether request-side loop preparation is waiting for the proxy.
pub(crate) fn request_finish_is_deferred(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions.get::<DeferredAgenticRequestFinish>().is_some()
}

/// Complete request-side loop preparation after all local dispatchers ran.
fn finish_request_after_dispatch(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
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
    trace!(
        iteration = state.iteration,
        "openai_agentic_loop request dispatch complete"
    );
    ctx.extensions.insert(state);
    Ok(FilterAction::Continue)
}

/// Complete the first request after MCP execution was deferred past all budget
/// admissions in the `StreamBuffer` pre-read phase.
pub(crate) fn finish_request_after_deferred_dispatch(
    ctx: &mut HttpFilterContext<'_>,
) -> Result<FilterAction, FilterError> {
    ctx.extensions.remove::<DeferredAgenticRequestFinish>();
    finish_request_after_dispatch(ctx)
}

/// Apply the request-wide retained-payload limit before any local dispatcher
/// side effect or upstream inference request.
#[expect(clippy::too_many_lines, reason = "shared initial and continuation budget admission")]
fn admit_retained_payload_budget(
    ctx: &mut HttpFilterContext<'_>,
    configured_limit: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let stream_payload_bytes = retained_stream_payload_bytes(ctx).unwrap_or(usize::MAX);
    let store_payload_bytes = super::store::retained_request_payload_bytes(ctx).unwrap_or(usize::MAX);
    let mut retained_overflow = None;
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.apply_retained_payload_limit(configured_limit);
        state.set_retained_external_payload_bytes(store_payload_bytes);
        if !state.can_replace_retained_payload(0, 0, stream_payload_bytes) {
            let initial = state.iteration == 0;
            let streaming = request_is_streaming(state);
            state.discard_payload_for_budget_error();
            retained_overflow = Some((initial, streaming));
        }
    }
    let Some((initial, streaming)) = retained_overflow else {
        return Ok(None);
    };

    ctx.set_metadata("responses.skip_persist", "true");
    set_action(ctx, ACTION_DONE)?;
    if initial {
        // The terminal 413 cannot persist or dispatch anything. Release the
        // sibling store snapshot as well as the shared response state so an
        // oversized rejected request does not remain live for the response
        // lifetime.
        super::store::discard_retained_request_payload(ctx);
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.set_retained_external_payload_bytes(0);
        }
        return Ok(Some(FilterAction::Reject(responses_error_rejection(
            413,
            "invalid_request_error",
            "request and rehydrated state exceed openai_agentic_loop.max_retained_bytes",
        ))));
    }
    if streaming {
        let body = encode_local_error(
            ctx,
            "server_error",
            "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes",
        );
        let mut rejection = Rejection::status(200)
            .with_header("content-type", "text/event-stream")
            .preserving_keepalive();
        if let Some(body) = body {
            rejection = rejection.with_body(body);
        }
        return Ok(Some(FilterAction::Reject(rejection)));
    }
    Ok(Some(FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes",
    ))))
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
    retained_budget_failure: bool,
) -> Result<FilterAction, FilterError> {
    // Every terminal response drops this round's dispatcher selections, but only
    // a retained-payload overflow may discard pending local synthesis and mark
    // the request as budget-failed. Dispatcher validation/execution failures
    // must leave already-executed tool events available to the stream finalizer.
    clear_round_dispatch_state(&mut state);
    if retained_budget_failure {
        state.fail_retained_payload_budget();
    }
    ctx.set_metadata("responses.skip_persist", "true");
    set_action(ctx, ACTION_DONE)?;
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
#[expect(
    clippy::too_many_lines,
    reason = "shared buffered and streaming local terminal construction"
)]
fn finish_deferred_local_response(
    ctx: &mut HttpFilterContext<'_>,
    mut state: ResponsesState,
) -> Result<FilterAction, FilterError> {
    state.deferred_tool_limit_completion = false;
    state.mcp_approval_state = McpApprovalState::None;
    clear_round_dispatch_state(&mut state);
    let streaming = request_is_streaming(&state);
    let mut body = None;
    if !streaming && let Err(rejection) = state.finalize_response_body(&mut body) {
        if state.retained_payload_failed {
            ctx.set_metadata("responses.skip_persist", "true");
            set_action(ctx, ACTION_DONE)?;
        }
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
    clear_round_dispatch_state(&mut state);
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
        if state.retained_payload_failed {
            ctx.set_metadata("responses.skip_persist", "true");
            set_action(ctx, ACTION_DONE)?;
        }
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
fn prepare_streamed_round(ctx: &HttpFilterContext<'_>, state: &mut ResponsesState) -> Result<bool, DispatchFailure> {
    if !super::streamed_round_is_dispatchable(ctx, state) {
        collect_streaming_output_items(state)?;
        state.tool_calls.clear();
        state.tool_search_calls.clear();
        state.web_search_calls.clear();
        // No dispatcher runs on a non-dispatchable (terminal) round, so drop the
        // file-search assignments and their synthesis queue: nothing will complete
        // those items, so `openai_stream_events` must not synthesize a lifecycle.
        state.file_search_assignments.clear();
        state.pending_local_tool_synthesis.clear();
        return Ok(false);
    }
    collect_streaming_output_items(state)?;
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
    state.tool_search_calls.clear();
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
    state.tool_search_calls.clear();
    state.web_search_calls.clear();
    state.current_round_output_start = None;

    if state.iteration > 0 {
        let original = std::mem::replace(&mut state.tool_choice, json!("auto"));
        state.original_tool_choice.get_or_insert(original);
        set_request_body_field(state, "tool_choice", json!("auto"));
        preserve_original_request_headers(ctx);
        queue_continuation_header(ctx, CONTENT_TYPE, HeaderValue::from_static("application/json"));
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
    for (name, value) in headers {
        queue_continuation_header(ctx, name, value);
    }
}

/// Queue a continuation header in both representations used by Core pre-read.
///
/// Owner projection or another provenance-aware filter may already have
/// activated the ordered mutation log. Once active, Core intentionally ignores
/// the legacy grouped queues for this pass, so replayed credentials must join
/// that log as well as remaining available to the normal request phase.
fn queue_continuation_header(ctx: &mut HttpFilterContext<'_>, name: http::HeaderName, value: HeaderValue) {
    ctx.request_headers_to_set.push((name.clone(), value.clone()));
    if !ctx.pre_read_mutations.is_empty() {
        ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(name, value));
    }
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
/// Web-search calls, pending file-search calls, and in-budget hosted
/// `tool_search_call` items that still have deferred connectors to list are
/// dispatchable. A completed file-search call is already terminal and must not
/// trigger another round. A `function_call` is
/// dispatchable only when it resolves to a configured MCP tool (auto or
/// approval-required, both server-owned); a client-owned `function_call`
/// resolves to no dispatcher, so a round carrying only client calls must
/// terminate as `done` rather than loop uselessly to the `max_infer_iters` cap.
fn has_dispatchable_calls(state: &ResponsesState) -> bool {
    if !state.web_search_calls.is_empty()
        || !state.file_search_assignments.is_empty()
        || has_pending_deferred_discovery(state)
    {
        return true;
    }
    if state.tool_calls.is_empty() || state.mcp_tool_map.is_empty() {
        return false;
    }
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    state.tool_calls.iter().any(|call| match call {
        ToolCallAssignment::Output(_) => assigned_tool_call(state, call)
            .is_some_and(|call| classify_mcp(call, &tool_index) != McpDisposition::NotMcp),
        ToolCallAssignment::Approved(invocation) => tool_index.contains(&invocation.encoded_name),
    })
}

/// Decide the loop outcome: done (no dispatchable tool calls or model-owned
/// finish), 508 (iteration limit), or loop (continue to tool execution).
#[expect(clippy::too_many_lines, reason = "transactional terminal and continuation decisions")]
fn evaluate_loop_decision(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    body: &mut Option<Bytes>,
    config: &AgenticLoopConfig,
) -> Result<FilterAction, FilterError> {
    mark_over_budget_tool_searches_incomplete(state);
    if !has_dispatchable_calls(state) {
        trace!("no dispatchable tool calls, signaling done");
        // Streaming terminals are owned by `openai_stream_events`, which rebuilds
        // the final frame and stored snapshot from `accumulated_output`; running
        // the draining finalizer here would leave it empty (see
        // `request_is_streaming`). Only the buffered path finalizes into `body`.
        let rewrites_buffered_body = !request_is_streaming(state) && state.response_object.is_object();
        if rewrites_buffered_body && let Err(rejection) = state.finalize_response_body(body) {
            if state.retained_payload_failed {
                ctx.set_metadata("responses.skip_persist", "true");
                set_action(ctx, ACTION_DONE)?;
            }
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

/// Rewrite queued hosted searches to `incomplete` when they cannot consume
/// remaining `max_tool_calls` budget, and drop them from dispatch.
#[expect(
    clippy::too_many_lines,
    reason = "updates canonical and persistence owners from one assignment set"
)]
fn mark_over_budget_tool_searches_incomplete(state: &mut ResponsesState) {
    if state.tool_search_calls.is_empty() || tool_search_discovery_is_within_budget(state) {
        return;
    }
    let queued_indices = state
        .tool_search_calls
        .iter()
        .map(|assignment| assignment.output_index)
        .collect::<Vec<_>>();
    let queued_ids: Vec<String> = queued_indices
        .iter()
        .filter_map(|index| {
            state
                .accumulated_output
                .get(*index)
                .and_then(|item| item.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    let mark_unidentified = queued_ids.is_empty();
    let mark = |item: &mut Value| {
        if item.get("type").and_then(Value::as_str) != Some("tool_search_call") {
            return;
        }
        let matches = match item.get("id").and_then(Value::as_str) {
            Some(id) => queued_ids.iter().any(|queued| queued == id),
            None => mark_unidentified,
        };
        if matches && let Some(obj) = item.as_object_mut() {
            obj.insert("status".to_owned(), json!("incomplete"));
        }
    };
    for index in queued_indices {
        if let Some(item) = state.accumulated_output.get_mut(index) {
            mark(item);
        }
    }
    for item in &mut state.persisted_messages {
        mark(item);
    }
    state.tool_search_calls.clear();
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
#[expect(
    clippy::too_many_lines,
    reason = "transactional parse, normalization, and retention admission"
)]
fn extract_tool_calls_from_body(body: &Bytes, state: &mut ResponsesState) -> Result<(), DispatchFailure> {
    // The framework owns and separately bounds the raw response body. Only the
    // parsed response and the copies retained by agentic-loop state belong in
    // this aggregate budget. The body length is an allocation-free upper bound
    // for the parsed value's compact JSON payload, so reserve that projection
    // before serde allocates it. The exact retained-owner accounting below runs
    // after normalization and ID insertion without charging the framework body.
    if !state.can_retain_payload(body.len()) {
        return Err(retained_payload_failure());
    }
    let response = serde_json::from_slice::<Value>(body)
        .ok()
        .filter(is_responses_api_output);
    let Some(mut response) = response else {
        state.response_object = Value::Null;
        state.tool_calls.clear();
        return Ok(());
    };
    let normalization_staging = output_normalization_staging_bytes(&response, has_file_search_tool(state))
        .ok_or_else(retained_payload_failure)?;
    if !body
        .len()
        .checked_add(normalization_staging)
        .is_some_and(|bytes| state.can_retain_payload(bytes))
    {
        return Err(retained_payload_failure());
    }
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
    let output = take_response_output(&mut response);
    let usage = response
        .as_object_mut()
        .and_then(|object| object.remove("usage"))
        .unwrap_or(Value::Null);
    if !buffered_response_retention_fits(state, &response, &output, &usage) {
        return Err(retained_payload_failure());
    }
    collect_output_items(output, state, &private_indices);
    if !usage.is_null() {
        merge_usage_owned(&mut state.usage, usage);
    }
    state.response_object = response;
    Ok(())
}

/// Preflight every independently owned value retained from one buffered model
/// response, including parser-local bytes that remain live through the commit.
/// The parsed response remains local until this succeeds, so an oversized round
/// commits no state mutation.
#[expect(clippy::too_many_lines, reason = "exhaustive per-owner buffered response accounting")]
fn buffered_response_retention_fits(state: &ResponsesState, response: &Value, output: &[Value], usage: &Value) -> bool {
    let Some(response_bytes) = super::state::retained_json_bytes(response) else {
        return false;
    };
    let Some(old_response_bytes) = super::state::retained_json_bytes(&state.response_object) else {
        return false;
    };
    let Some(old_usage_bytes) = super::state::retained_json_bytes(&state.usage) else {
        return false;
    };

    let Some(output_bytes) = super::state::retained_json_values_bytes(output) else {
        return false;
    };
    let Some(usage_bytes) = super::state::retained_json_bytes(usage) else {
        return false;
    };
    // The parsed response has been split into response metadata, output items,
    // and usage by move. Admit those parser-local owners before projecting the
    // merged usage tree.
    if !state.can_retain_payload(response_bytes.saturating_add(output_bytes).saturating_add(usage_bytes)) {
        return false;
    }

    let new_usage_bytes = if usage.is_null() {
        super::state::retained_json_bytes(&state.usage)
    } else {
        merged_usage_json_bytes(&state.usage, usage)
    };
    let Some(new_usage_bytes) = new_usage_bytes else {
        return false;
    };
    let mut copied_item_bytes = 0_usize;

    for item in output {
        let Some(bytes) = super::state::retained_json_bytes(item) else {
            return false;
        };
        // Every item moves into accumulated_output. Only the distinct backend
        // and persistence histories create payload-sized copies; dispatch
        // queues retain fixed-size absolute assignments.
        let copies = match item.get("type").and_then(Value::as_str) {
            Some("function_call")
                if item
                    .get("status")
                    .is_none_or(|status| status.is_null() || status.as_str() == Some("completed")) =>
            {
                3
            },
            Some("reasoning") => 3,
            Some("web_search_call" | "tool_search_call" | "file_search_call") => 2,
            _ => 1,
        };
        copied_item_bytes = copied_item_bytes.saturating_add(bytes.saturating_mul(copies));
    }
    // Items move from the parser-local output vector into the accumulator, so
    // the construction peak adds only the history copies and merged usage.
    let history_copy_bytes = copied_item_bytes.saturating_sub(output_bytes);
    let peak_added = response_bytes
        .saturating_add(output_bytes)
        .saturating_add(usage_bytes)
        .saturating_add(history_copy_bytes)
        .saturating_add(new_usage_bytes.saturating_sub(old_usage_bytes));
    let final_removed = old_response_bytes.saturating_add(old_usage_bytes);
    let final_added = response_bytes
        .saturating_add(new_usage_bytes)
        .saturating_add(copied_item_bytes);
    state.can_retain_payload(peak_added) && state.can_replace_retained_payload(final_removed, final_added, 0)
}

/// Shared server-side failure for payload growth after initial admission.
fn retained_payload_failure() -> DispatchFailure {
    DispatchFailure {
        status: 502,
        code: "server_error",
        message: "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes".to_owned(),
    }
}

/// Bound payload allocated while provider output is normalized in place.
///
/// Private file-search translation parses the arguments JSON and clones its
/// decoded queries before removing the original argument string. Two argument
/// lengths cover the parsed tree plus decoded-query owner; call-id copies and
/// fixed rewritten fields are charged separately. Synthetic public IDs are
/// bounded without formatting them first.
fn output_normalization_staging_bytes(response: &Value, translate_file_search: bool) -> Option<usize> {
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return Some(0);
    };
    output.iter().try_fold(0_usize, |used, item| {
        let mut added = 0_usize;
        if translate_file_search && is_file_search_function_call(item) {
            let arguments = item.get("arguments").and_then(Value::as_str).map_or(0, str::len);
            let call_id = item.get("call_id").and_then(Value::as_str).map_or(0, str::len);
            added = arguments
                .checked_mul(2)?
                .checked_add(call_id.checked_mul(2)?)?
                .checked_add(256)?;
        }
        if item.get("id").and_then(Value::as_str).is_none_or(str::is_empty) {
            added = added.checked_add(64)?;
        }
        used.checked_add(added)
    })
}

/// Move the current round's output array out of a parsed response while leaving
/// a valid empty canonical slot for terminal finalization.
fn take_response_output(response: &mut Value) -> Vec<Value> {
    let Some(object) = response.as_object_mut() else {
        return Vec::new();
    };
    let Some(output) = object.get_mut("output") else {
        return Vec::new();
    };
    match std::mem::replace(output, Value::Array(Vec::new())) {
        Value::Array(output) => output,
        other => {
            *output = other;
            Vec::new()
        },
    }
}

/// Borrow the canonical provider output behind one dispatch assignment.
fn assigned_tool_call<'a>(state: &'a ResponsesState, call: &'a ToolCallAssignment) -> Option<&'a Value> {
    match call {
        ToolCallAssignment::Output(assignment) => state.assigned_output(*assignment),
        ToolCallAssignment::Approved(_) => None,
    }
}

/// Return whether one model round mixed server-owned MCP, web-search, pending
/// file-search, or hosted tool-search calls with calls that must be executed by
/// the API client. The current IRR continuation cannot execute the former
/// without sending the latter back to inference as an unresolved call, so fail
/// before any external side effect.
#[expect(
    clippy::too_many_lines,
    reason = "checks every server- and client-owned call representation"
)]
fn has_mixed_function_call_ownership(state: &ResponsesState) -> bool {
    let mut has_server = !state.web_search_calls.is_empty()
        || !state.file_search_assignments.is_empty()
        || has_hosted_queued_tool_search(state);
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
        let is_mcp = match call {
            ToolCallAssignment::Output(_) => assigned_tool_call(state, call)
                .and_then(|call| call.get("name"))
                .and_then(Value::as_str)
                .is_some_and(|encoded| tool_index.contains(encoded)),
            ToolCallAssignment::Approved(invocation) => tool_index.contains(&invocation.encoded_name),
        };
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
fn collect_output_items(output: Vec<Value>, state: &mut ResponsesState, private_indices: &[usize]) {
    state.current_round_output_start = Some(state.accumulated_output.len());
    let mut pending_file_search: Vec<(usize, SynthesisKind)> = Vec::new();
    let mut web_ordinal = 0_usize;
    for (round_index, item) in output.into_iter().enumerate() {
        let absolute_index = state.accumulated_output.len();
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") if is_dispatchable_function_call(&item) => {
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
                state.tool_calls.push(ToolCallAssignment::Output(OutputAssignment {
                    output_index: absolute_index,
                }));
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
                state.web_search_calls.push(WebSearchAssignment {
                    output_index: absolute_index,
                    ordinal: web_ordinal,
                });
                web_ordinal = web_ordinal.saturating_add(1);
                state.persisted_messages.push(item.clone());
            },
            Some("tool_search_call") if is_hosted_completed_tool_search(&item) => {
                // Only a completed hosted search may trigger deferred
                // `tools/list`. Client-executed searches return to the caller
                // without listing or another inference round.
                state.tool_search_calls.push(OutputAssignment {
                    output_index: absolute_index,
                });
                state.persisted_messages.push(item.clone());
            },
            Some("tool_search_call") if is_completed_output_item(&item) => {
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
                if is_pending_file_search_call(&item) {
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
        state.accumulated_output.push(item);
    }
    record_file_search_assignments(state, pending_file_search);
    mark_provider_history(state);
}

/// The provider persists every successful round in a native conversation.
fn mark_provider_history(state: &mut ResponsesState) {
    if !state.history_rehydrated
        && state
            .conversation
            .as_ref()
            .is_some_and(|conversation| !conversation.is_null())
    {
        state.provider_history_len = state.messages.len();
    }
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
fn collect_streaming_output_items(state: &mut ResponsesState) -> Result<(), DispatchFailure> {
    // A preceding owner may already have drained this round into the canonical
    // accumulator. Preserve its fixed-index dispatch assignments on re-entry.
    if state.output_items().is_empty() && state.current_round_output_start.is_some() {
        return Ok(());
    }
    let normalization_staging = output_normalization_staging_bytes(&state.response_object, has_file_search_tool(state))
        .ok_or_else(retained_payload_failure)?;
    if !state.can_retain_payload(normalization_staging) {
        state.discard_payload_for_budget_error();
        return Err(retained_payload_failure());
    }
    // Normalize private file-search `function_call`s in the streamed response
    // object before draining it. The returned round-local indices tag the
    // synthesis origin (private openings were suppressed live and must be
    // reproduced; native openings already streamed).
    let private_indices = if has_file_search_tool(state) {
        translate_function_calls_to_file_search(&mut state.response_object)
    } else {
        Vec::new()
    };
    // Stamp stable synthetic IDs on any id-less streamed items before draining the
    // round into the accumulator (issue #955), mirroring the buffered path.
    ensure_public_output_item_ids_in_response(&mut state.response_object);
    if !streaming_collection_retention_fits(state) {
        state.discard_payload_for_budget_error();
        return Err(retained_payload_failure());
    }
    let base = state.accumulated_output.len();
    state.current_round_output_start = Some(base);
    // Move the round out (leaving a valid `[]`), so each item is routed to its
    // last consumer by move; only earlier consumers of a shared item clone.
    let round = std::mem::take(state.output_items_mut());
    state.tool_calls.clear();
    state.tool_search_calls.clear();
    state.web_search_calls.clear();
    let mut pending_file_search: Vec<(usize, SynthesisKind)> = Vec::new();
    let mut web_ordinal = 0_usize;
    for (round_index, item) in round.into_iter().enumerate() {
        let absolute_index = state.accumulated_output.len();
        match item.get("type").and_then(Value::as_str) {
            Some("function_call") => {
                state.messages.push(item.clone());
                state.persisted_messages.push(item.clone());
                if is_dispatchable_function_call(&item) {
                    state.tool_calls.push(ToolCallAssignment::Output(OutputAssignment {
                        output_index: absolute_index,
                    }));
                }
                state.accumulated_output.push(item);
            },
            Some("reasoning") => {
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
                state.web_search_calls.push(WebSearchAssignment {
                    output_index: absolute_index,
                    ordinal: web_ordinal,
                });
                web_ordinal = web_ordinal.saturating_add(1);
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
            Some("tool_search_call") if is_hosted_completed_tool_search(&item) => {
                // Mirror the buffered collector: a completed hosted
                // `tool_search_call` is not a valid OpenResponses input item, so
                // it must not enter `messages`. `openai_mcp_dispatch` consumes
                // `tool_search_calls` to list deferred connectors.
                state.tool_search_calls.push(OutputAssignment {
                    output_index: absolute_index,
                });
                state.accumulated_output.push(item.clone());
                state.persisted_messages.push(item);
            },
            Some("tool_search_call") if is_completed_output_item(&item) => {
                // Client-executed searches stay client-visible and stored, but
                // must not queue server-side connector discovery.
                state.accumulated_output.push(item.clone());
                state.persisted_messages.push(item);
            },
            _ => {
                state.accumulated_output.push(item);
            },
        }
    }
    record_file_search_assignments(state, pending_file_search);
    mark_provider_history(state);
    Ok(())
}

/// Preflight the new owners created when the canonical streamed response output
/// is drained into the cross-round accumulator and classified for dispatch.
fn streaming_collection_retention_fits(state: &ResponsesState) -> bool {
    let Some(output) = state.response_object.get("output").and_then(Value::as_array) else {
        return true;
    };
    let Some(response_bytes) = super::state::retained_json_bytes(&state.response_object) else {
        return false;
    };
    let Some(output_array_bytes) = super::bounded_json_size(output, usize::MAX).ok().flatten() else {
        return false;
    };
    let drained_response_bytes = response_bytes.saturating_sub(output_array_bytes).saturating_add(2);
    let mut added = drained_response_bytes;
    for item in output {
        let Some(bytes) = super::state::retained_json_bytes(item) else {
            return false;
        };
        let copies = match item.get("type").and_then(Value::as_str) {
            Some("function_call" | "reasoning") => 3,
            Some("web_search_call" | "tool_search_call" | "file_search_call") => 2,
            _ => 1,
        };
        added = added.saturating_add(bytes.saturating_mul(copies));
    }
    // The old response tree is replaced by the same tree with an empty output.
    state.can_replace_retained_payload(response_bytes, added, 0)
}

/// Whether a function call is complete enough to dispatch.
///
/// OpenAI omits `status` or sends `null` on some completed calls (#955), so
/// missing/null is treated as completed. Explicit non-completed statuses are not.
fn is_dispatchable_function_call(item: &Value) -> bool {
    item.get("status")
        .is_none_or(|status| status.is_null() || status.as_str() == Some("completed"))
}

/// Whether an output item is a completed tool or search call.
fn is_completed_output_item(item: &Value) -> bool {
    item.get("status").and_then(Value::as_str) == Some("completed")
}

/// Whether a completed `tool_search_call` is owned by the proxy, not the client.
fn is_hosted_completed_tool_search(item: &Value) -> bool {
    is_completed_output_item(item) && !super::state::is_client_executed_tool_call(item)
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

/// Drop this round's dispatcher queues so a terminal outcome cannot re-dispatch.
fn clear_round_dispatch_state(state: &mut ResponsesState) {
    state.tool_calls.clear();
    state.web_search_calls.clear();
    state.tool_search_calls.clear();
    state.file_search_assignments.clear();
}

/// Whether a hosted `tool_search_call` is queued for deferred connector listing.
fn has_hosted_queued_tool_search(state: &ResponsesState) -> bool {
    state
        .tool_search_calls
        .iter()
        .filter_map(|assignment| state.assigned_output(*assignment))
        .any(|item| !super::state::is_client_executed_tool_call(item))
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
