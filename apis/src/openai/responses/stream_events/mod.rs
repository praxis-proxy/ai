// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Composes the current iterative-request-router (IRR) execution into
//! one logical Responses API SSE stream.
//!
//! Parses backend SSE chunks using [`SseFrameParser`], dispatches typed
//! events to update [`ResponsesState`] in request extensions, and
//! normalizes successive IRR inference streams into one downstream
//! Responses lifecycle. A single inference round is just a one-round
//! logical stream, so the filter always normalizes. It must run inside
//! an `iterative_request_router` step; running it anywhere else is a
//! misconfiguration and fails closed at request time.
//!
//! [`SseFrameParser`]: crate::openai::sse::SseFrameParser
//! [`ResponsesState`]: super::state::ResponsesState

pub(crate) mod accumulator;
pub(super) mod client_tools;
mod config;
mod local_tools;

use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    hash::{Hash as _, Hasher as _},
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState,
    StreamTerminationCause, SubRequestResponseMode, parse_filter_config,
};
use serde::{Serialize, ser::SerializeMap as _};
use serde_json::Value;
use tracing::{debug, trace, warn};

#[cfg(test)]
use self::accumulator::accumulate_response_object;
use self::{
    accumulator::{accumulate_event, find_output_item, tool_call_key},
    config::StreamEventsConfig,
};
use crate::{
    classifier::is_responses_create,
    is_event_stream_content_type,
    openai::{
        responses::{
            error::{responses_error_rejection, responses_error_sse_payload},
            openai_client_tool_compat::{restore_snapshot, restore_snapshot_tools},
            state::{
                ClientToolRestore, EmittedItem, PayloadMeter, ResponsesState, retained_json_bytes,
                retained_json_values_bytes,
            },
        },
        sse::{SseFrame, SseFrameParser, SseParseError, SseParserConfig, responses::ResponsesEvent},
    },
};

/// Client-visible terminal message for an exhausted agentic payload budget.
pub(crate) const RETAINED_PAYLOAD_OVERFLOW_MESSAGE: &str =
    "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes";

/// A per-turn terminal event held until the agentic transition is known.
struct DeferredTerminalEvent {
    /// Canonical event type.
    event_type: String,
    /// Parsed envelope fields. A `response: null` marker replaces the moved
    /// tree, so the live stream holds only metadata on conformant terminals.
    payload: Value,
}

impl DeferredTerminalEvent {
    /// Payload bytes retained while the loop owner decides whether this round
    /// is terminal.
    fn retained_payload_bytes(&self) -> Option<usize> {
        self.event_type.len().checked_add(retained_json_bytes(&self.payload)?)
    }
}

/// Completion state observed while parsing a Responses SSE stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompletionState {
    /// No completion signal has been observed.
    Open,
    /// A terminal lifecycle event was observed.
    TerminalLifecycle,
    /// A stream-level error event was observed.
    Error,
}

/// Per-request parser and accumulation state.
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent per-round lifecycle flags (deferred [DONE], local-item flush, poison)"
)]
pub(super) struct StreamEventsState {
    /// Byte-level SSE frame parser.
    frame_parser: SseFrameParser,
    /// Number of non-sentinel events parsed so far.
    event_count: usize,
    /// Maximum allowed event count.
    max_events: usize,
    /// Maximum allowed wall-clock time.
    timeout: Duration,
    /// Timestamp of first chunk.
    started_at: Option<Instant>,
    /// Timestamp when a terminal state was first observed.
    completed_at: Option<Instant>,
    /// Stream completion state (`Open` / `TerminalLifecycle` / `Error`).
    completion_state: CompletionState,
    /// Accumulated function-call argument deltas, keyed by item id or output index.
    tool_call_args: std::collections::HashMap<String, String>,
    /// Tool-call keys whose arguments exceeded the configured byte cap.
    rejected_tool_call_args: std::collections::HashSet<String>,
    /// Cap on accumulated bytes per tool-call argument string.
    max_tool_call_argument_bytes: usize,
    /// Aggregate accumulation byte ceiling; the stream fails closed once passed.
    ///
    /// The running total it bounds lives request-wide in
    /// [`ResponsesState::stream_accumulated_bytes`], not here, so it survives the
    /// per-round re-arm.
    max_accumulated_bytes: usize,
    /// Cap on accumulated output items; the stream fails closed once passed.
    ///
    /// The count it bounds is derived request-wide from the retained output
    /// (`accumulated_output` + the current round's `output`), not tracked here.
    max_output_items: usize,
    /// Inference iteration number for lifecycle suppression and index offsets.
    iteration: u32,
    /// Output index offset contributed by preceding inference/tool rounds.
    output_index_offset: u64,
    /// Terminal event withheld until completion filters publish a transition.
    deferred_terminal: Option<DeferredTerminalEvent>,
    /// Whether a provider `[DONE]` sentinel should follow the logical terminal.
    deferred_done: bool,
    /// Whether this round already ran the in-band flush of pending local tool
    /// items. `accumulated_output` is fixed for the duration of a round (the
    /// agentic loop only rewrites it at round boundaries), so the flush is run at
    /// most once per round rather than re-serializing every local item ahead of
    /// each resumed event.
    local_items_flushed: bool,
    /// Locally-executable tool items opened this round, keyed by `item:{id}` and
    /// `index:{output_index}` → suppression mode (§4.1). Transient per-round: created
    /// in `arm()`, dropped when the state is removed at `finalize_logical_stream`.
    local_tool_items: std::collections::HashMap<String, local_tools::LocalToolMode>,
    /// #1159: lowered client-tool items tracked across their streaming lifecycle,
    /// so the plan pass can enforce lifecycle order and (Tasks 5-7) synthesize the
    /// typed restoration. Empty for native (non-lowered) traffic.
    client_tool_items: Vec<client_tools::ClientToolStreamItem>,
    /// #1159 C1: once any chunk fails, the whole logical stream is poisoned; every
    /// later chunk is dropped closed so a post-failure event (e.g. a co-batched
    /// lowered `output_item.done` whose `.added` was rolled back) cannot emit.
    stream_failed: bool,
    /// Request and history charge cached across ordinary upstream chunks.
    /// Refreshed at EOS after the agentic owner may append round output.
    shared_stable_bytes: OnceLock<usize>,
    /// Prior-round output is immutable during an inference stream. Cache its
    /// charge separately from current-round output and parser state.
    shared_prior_output_bytes: OnceLock<Option<PriorOutputCache>>,
    /// Exact charge for the current response and fallback terminal template.
    /// Output/terminal/done events change these owners
    /// during a live stream; later argument deltas leave them intact.
    /// Zero means unmeasured; the serialized `response_object` is always at
    /// least `null` (four bytes), so every valid measurement is nonzero.
    shared_current_output_bytes: AtomicUsize,
}

/// A prior-output measurement and the state shape it describes.
#[derive(Clone, Copy)]
struct PriorOutputCache {
    /// Inference round owning the completed output.
    iteration: u32,
    /// Stable-payload revision at measurement time.
    revision: u64,
    /// Number of completed items at measurement time.
    output_len: usize,
    /// Compact JSON bytes retained by those items.
    bytes: usize,
}

impl PriorOutputCache {
    /// Whether this charge still describes the shared output owner.
    fn matches(self, state: &ResponsesState) -> bool {
        self.iteration == state.iteration
            && Some(self.revision) == state.replay_stable_payload_revision
            && self.output_len == state.accumulated_output.len()
    }
}

impl StreamEventsState {
    /// The owner can append history or rewrite the terminal between EOS
    /// callbacks; recalculate every shared charge after that boundary.
    fn clear_shared_budget_caches(&mut self) {
        self.shared_stable_bytes = OnceLock::new();
        self.shared_prior_output_bytes = OnceLock::new();
        self.shared_current_output_bytes.store(0, Ordering::Relaxed);
    }

    /// Raw parser, argument, deferred-terminal, and local suppression payload
    /// retained outside [`ResponsesState`].
    fn retained_payload_bytes(&self) -> Option<usize> {
        let argument_bytes = self.tool_call_args.iter().try_fold(0_usize, |used, (key, value)| {
            used.checked_add(key.len())?.checked_add(value.len())
        })?;
        let rejected_bytes = self
            .rejected_tool_call_args
            .iter()
            .try_fold(0_usize, |used, key| used.checked_add(key.len()))?;
        let local_key_bytes = self
            .local_tool_items
            .keys()
            .try_fold(0_usize, |used, key| used.checked_add(key.len()))?;
        let client_tool_bytes = self.client_tool_items.iter().try_fold(0_usize, |used, item| {
            used.checked_add(item.key.len())?
                .checked_add(item.private_name.len())?
                .checked_add(item.item_id.as_ref().map_or(0, String::len))
        })?;
        self.frame_parser
            .retained_bytes()
            .checked_add(argument_bytes)?
            .checked_add(rejected_bytes)?
            .checked_add(local_key_bytes)?
            .checked_add(client_tool_bytes)?
            .checked_add(
                self.deferred_terminal
                    .as_ref()
                    .map_or(Some(0), DeferredTerminalEvent::retained_payload_bytes)?,
            )
    }
}

/// Publish this filter's local owner in request-shared state. The agentic-loop
/// filter has a different filter ID and cannot read our local state directly.
fn publish_stream_payload(ctx: &mut HttpFilterContext<'_>, state: &StreamEventsState) -> bool {
    let Some(bytes) = state.retained_payload_bytes() else {
        return false;
    };
    if !stream_payload_fits(ctx, state, 0) {
        return false;
    }
    let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() else {
        return true;
    };
    responses.retained_stream_parser_bytes = bytes;
    true
}

/// Composes the current IRR execution into one logical Responses stream.
///
/// Must run inside an `iterative_request_router` step. Running it
/// elsewhere is a misconfiguration and fails closed at request time.
/// Place it after `load_balancer` so IRR body hooks run with a selected
/// peer when needed. `timeout_secs` is an absolute deadline from the first
/// SSE chunk; each chunk recaps that absolute cutoff onto the live body through
/// [`praxis_filter::HttpFilterContext::cap_stream_deadline`].
///
/// # YAML
///
/// ```yaml
/// filter: openai_stream_events
/// # All fields optional:
/// # max_buffer_bytes: 10485760
/// # max_events: 100000
/// # timeout_secs: 300
/// # max_tool_call_argument_bytes: 1048576
/// # max_accumulated_bytes: 67108864
/// # max_output_items: 100000
/// ```
pub struct OpenaiStreamEventsFilter {
    /// Configuration for the SSE frame parser.
    parser_config: SseParserConfig,
    /// Cap on accumulated bytes per tool-call argument string.
    max_tool_call_argument_bytes: usize,
    /// Aggregate accumulation byte ceiling across output items and tool-call args.
    max_accumulated_bytes: usize,
    /// Cap on accumulated streaming output items.
    max_output_items: usize,
}

impl OpenaiStreamEventsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config)?))
    }

    /// Build the concrete filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    fn build(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: StreamEventsConfig = parse_filter_config("openai_stream_events", config)?;
        cfg.validate()?;
        Ok(Self {
            parser_config: cfg.to_parser_config(),
            max_tool_call_argument_bytes: cfg.max_tool_call_argument_bytes(),
            max_accumulated_bytes: cfg.max_accumulated_bytes(),
            max_output_items: cfg.max_output_items(),
        })
    }

    /// Whether per-request parser state has been installed.
    fn is_armed(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.get_filter_state::<StreamEventsState>().is_some()
    }

    /// Build fresh per-round parser state seeded with this round's `iteration`
    /// and `output_index_offset`.
    ///
    /// The accumulation budget's running totals are not reset here — they live
    /// request-wide in [`ResponsesState`] so they survive the re-arm.
    fn new_round_state(&self, iteration: u32, output_index_offset: u64) -> StreamEventsState {
        StreamEventsState {
            frame_parser: SseFrameParser::new(self.parser_config.max_buffer_bytes),
            event_count: 0,
            max_events: self.parser_config.max_events,
            timeout: self.parser_config.timeout,
            started_at: None,
            completed_at: None,
            completion_state: CompletionState::Open,
            tool_call_args: std::collections::HashMap::new(),
            rejected_tool_call_args: std::collections::HashSet::new(),
            max_tool_call_argument_bytes: self.max_tool_call_argument_bytes,
            max_accumulated_bytes: self.max_accumulated_bytes,
            max_output_items: self.max_output_items,
            iteration,
            output_index_offset,
            deferred_terminal: None,
            deferred_done: false,
            local_items_flushed: false,
            local_tool_items: std::collections::HashMap::new(),
            client_tool_items: Vec::new(),
            stream_failed: false,
            shared_stable_bytes: OnceLock::new(),
            shared_prior_output_bytes: OnceLock::new(),
            shared_current_output_bytes: AtomicUsize::new(0),
        }
    }

    /// Install fresh parser state for one inference stream.
    fn arm(&self, ctx: &mut HttpFilterContext<'_>) {
        let (iteration, output_index_offset) = ctx.extensions.get_mut::<ResponsesState>().map_or((0, 0), |state| {
            let output_index_offset = u64::try_from(state.accumulated_output.len()).unwrap_or(u64::MAX);
            // Invalidate the previous round's terminal response object before a
            // resumed round begins. Move it instead of dropping it: a request-side
            // dispatcher can terminate locally (approval/tool limit) before a new
            // upstream response exists and still needs the response metadata to
            // encode `response.completed`. Once upstream response headers arrive,
            // `on_response` drops this fallback so a later provider `error` cannot
            // persist stale success as the logical result.
            if state.response_object.is_object() {
                state.local_completion_response_template = std::mem::take(&mut state.response_object);
            } else {
                state.response_object = Value::Null;
            }
            state.mark_current_output_changed();
            (state.iteration, output_index_offset)
        });
        // Re-arming drops the preceding round's parser owner. Publish the new
        // empty owner before the loop filter's request-side admission runs.
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.retained_stream_parser_bytes = 0;
        }
        ctx.insert_filter_state(self.new_round_state(iteration, output_index_offset));
        ctx.set_metadata(super::STREAM_ERROR_FINALIZER_ARMED_KEY, "true");
        ctx.set_metadata("responses.stream_completion", "open");
        // Publish a per-round marker that `openai_agentic_loop` reads (and then
        // consumes) to confirm this typed-streaming round can surface
        // loop-terminal errors through `finalize_logical_stream`. Refreshed
        // every armed round because the agentic loop overwrites it after each
        // check.
        ctx.set_metadata("responses.logical_stream", "true");
        // #1159: advertise to openai_client_tool_compat (which runs later, in
        // on_request_body) that this logical SSE owner is present and will drive
        // streaming client-tool restoration, so the compat filter arms lowering
        // instead of failing streaming closed.
        ctx.set_metadata(CLIENT_TOOL_STREAM_RESTORATION_MARKER, "true");
    }

    /// Apply the guard [`ArmDecision`], returning an early [`FilterAction`] when
    /// the request must be rejected before any upstream dispatch.
    ///
    /// The pure [`arm_decision`] classifies the request; this applies the
    /// effects that need the context — installing parser state, stripping
    /// `Accept-Encoding`, or building the fail-closed rejection.
    fn apply_arm_decision(&self, ctx: &mut HttpFilterContext<'_>, decision: ArmDecision) -> Option<FilterAction> {
        match decision {
            ArmDecision::Ignore => None,
            ArmDecision::RejectOutsideIrr => {
                // The filter always composes the current IRR execution into one
                // logical Responses stream, so it must run inside an
                // `iterative_request_router` step. A missing `IterationState`
                // means the filter is placed outside IRR — a server
                // misconfiguration. Fail closed before any upstream dispatch
                // rather than emit an unnormalized stream that later
                // loop-terminal errors could not correct.
                warn!("openai_stream_events is not inside an iterative_request_router step");
                Some(FilterAction::Reject(responses_error_rejection(
                    500,
                    "server_error",
                    "openai_stream_events must run inside an iterative_request_router step",
                )))
            },
            ArmDecision::Arm => {
                trace!("arming stream_events for streaming Responses API request");
                self.arm(ctx);
                // The SSE frame parser consumes raw bytes, so a compressed
                // upstream body would be parsed as opaque data — suppressing the
                // stream and failing an otherwise valid request. Strip
                // `Accept-Encoding` whenever logical parsing is armed so a
                // compliant backend returns plaintext SSE.
                ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
                None
            },
        }
    }
}

/// Outcome of the request-phase IRR-placement guard.
///
/// Factored out of [`OpenaiStreamEventsFilter`]'s `on_request` so the guard's
/// fail-closed decision table — the invariant that logical composition only
/// arms inside an `iterative_request_router` step — is exhaustively unit
/// testable. The runtime signal it depends on, an [`IterationState`] in request
/// extensions, cannot be constructed outside praxis-filter (its fields are
/// private), so the end-to-end arming effect is covered by functional
/// integration tests while this pure decision is covered directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArmDecision {
    /// Not a streaming Responses create request; leave the stream untouched.
    Ignore,
    /// Streaming Responses request placed outside IRR; reject fail-closed.
    RejectOutsideIrr,
    /// Streaming Responses request inside IRR; arm logical composition.
    Arm,
}

/// Metadata marker published so `openai_client_tool_compat` (which lowers rich
/// client tools in `on_request_body`) knows this logical SSE owner is present
/// and will restore the lowered calls live in the stream (#1159). Published in
/// both the request and request-body phases so the compat guard observes it
/// regardless of which phase the IRR step executor runs first.
const CLIENT_TOOL_STREAM_RESTORATION_MARKER: &str = "responses.client_tool_stream_restoration";

/// Decide whether to arm logical composition for the current request.
///
/// Arms only for a streaming Responses create request, and only inside an IRR
/// step; the same request outside IRR fails closed rather than emit an
/// unnormalized stream that later loop-terminal errors could not correct.
const fn arm_decision(is_streaming_responses: bool, inside_irr: bool) -> ArmDecision {
    match (is_streaming_responses, inside_irr) {
        (false, _) => ArmDecision::Ignore,
        (true, true) => ArmDecision::Arm,
        (true, false) => ArmDecision::RejectOutsideIrr,
    }
}

/// Classify the current request from context signals, shared by the request
/// and request-body phases.
///
/// The `iterative_request_router` runner moves request extensions into each
/// step but builds a fresh `filter_metadata` map, so metadata set by pre-IRR
/// filters (e.g. `openai_responses_format`) is not visible here. `ResponsesState`
/// is created pre-IRR and travels through extensions, so fall back to it for
/// format and stream detection — mirroring how `responses_to_chat_completions`
/// resolves `request_is_streaming`. `IterationState` is inserted by the IRR
/// runner before the request phase of every iteration (including iteration 0),
/// so its presence is the runtime signal that the filter is placed inside an
/// IRR step.
///
/// Called from both `on_request` and `on_request_body` because the IRR step
/// executor's phase order depends on the step's aggregate request body mode: a
/// `StreamBuffer`-mode step (forced when `openai_agentic_loop` shares the step)
/// runs `on_request_body` before `on_request`, while a `Stream`-mode step runs
/// `on_request` first. The streaming client-tool-restoration marker (#1159)
/// must be published in whichever phase runs first, so both call this.
fn arm_decision_for(ctx: &HttpFilterContext<'_>) -> ArmDecision {
    let typed_streaming = ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming;
    let responses_state = ctx.extensions.get::<ResponsesState>();
    let has_responses_state = responses_state.is_some();
    let body_stream = responses_state
        .and_then(|state| state.request_body.get("stream"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let is_responses = is_responses_create(&ctx.request.method, ctx.request.uri.path())
        && (typed_streaming
            || ctx.get_metadata("openai_responses_format.format") == Some("openai_responses")
            || has_responses_state);
    let is_streaming =
        typed_streaming || ctx.get_metadata("openai_responses_format.stream") == Some("true") || body_stream;
    let inside_irr = ctx.extensions.get::<IterationState>().is_some();
    arm_decision(is_responses && is_streaming, inside_irr)
}

#[async_trait]
impl HttpFilter for OpenaiStreamEventsFilter {
    fn name(&self) -> &'static str {
        "openai_stream_events"
    }

    fn request_body_access(&self) -> BodyAccess {
        // ReadOnly (not None) so the IRR step executor invokes `on_request_body`,
        // where the streaming client-tool-restoration marker is published for the
        // `StreamBuffer`-first phase ordering (#1159). The body is never mutated;
        // the mode stays `Stream` so this never escalates the aggregate step to
        // buffering.
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn response_body_access(&self) -> BodyAccess {
        // Always ReadWrite: the filter normalizes every armed stream into one
        // logical Responses lifecycle, rewriting per-round SSE bytes.
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let decision = arm_decision_for(ctx);
        if let Some(action) = self.apply_arm_decision(ctx, decision) {
            return Ok(action);
        }

        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // #1159: When `openai_agentic_loop` shares this IRR step its
        // `StreamBuffer` request body mode makes the step executor run
        // `on_request_body` for every step filter BEFORE any `on_request`. In
        // that ordering `on_request`'s `arm()` — which publishes the
        // restoration marker — has not run yet when `openai_client_tool_compat`
        // reaches its own `on_request_body` streaming guard, so it would fail the
        // streaming request closed. Publish the marker here too: this filter
        // precedes the compat filter in step order, so its `on_request_body`
        // runs first in BOTH phase orderings and the guard always observes the
        // marker. Full arming (parser-state install) still happens in
        // `on_request`, which always runs before the upstream response, so the
        // response-phase restoration path is unaffected. The body is only read,
        // never mutated.
        if arm_decision_for(ctx) == ArmDecision::Arm {
            ctx.set_metadata(CLIENT_TOOL_STREAM_RESTORATION_MARKER, "true");
        }
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !Self::is_armed(ctx) {
            return Ok(FilterAction::Continue);
        }

        // A real upstream response now owns this round's terminal lifecycle.
        // The request-side fallback is no longer reachable and retaining it
        // could leave stale success metadata live after an upstream error.
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.local_completion_response_template = Value::Null;
            state.mark_current_output_changed();
        }
        if let Some(stream) = ctx.get_filter_state::<StreamEventsState>() {
            stream.shared_current_output_bytes.store(0, Ordering::Relaxed);
        }

        if !is_success_sse_response(ctx) {
            debug!("disarming stream_events: response is not 2xx text/event-stream");
            ctx.remove_filter_state::<StreamEventsState>();
            ctx.set_metadata(super::STREAM_ERROR_FINALIZER_ARMED_KEY, "false");
            return Ok(FilterAction::Continue);
        }

        // Defense in depth: `on_request` strips `Accept-Encoding`, but a
        // non-compliant backend may still return an encoded body. The SSE
        // parser cannot decode it, so decline to parse and let the response
        // pass through untransformed rather than corrupt an otherwise valid
        // stream into a spurious error.
        if response_is_encoded(ctx) {
            debug!("disarming stream_events: response carries Content-Encoding");
            ctx.remove_filter_state::<StreamEventsState>();
            ctx.set_metadata(super::STREAM_ERROR_FINALIZER_ARMED_KEY, "false");
            return Ok(FilterAction::Continue);
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !Self::is_armed(ctx) {
            debug!("stream_events not armed, passing through");
            return Ok(FilterAction::Continue);
        }

        // The store can discover an aggregate overflow after this filter has
        // emitted an earlier chunk. Suppress every later provider frame and
        // its finalizer so only the store's in-band error reaches the client.
        if ctx.get_metadata("responses.store_stream_budget_failed") == Some("true") {
            *body = None;
            return Ok(FilterAction::Continue);
        }

        process_chunk(ctx, body, end_of_stream);

        if end_of_stream {
            record_idle_transport_timeout(ctx);
            validate_stream_end(ctx);
            finalize_logical_stream(ctx, body);
        }

        Ok(FilterAction::Continue)
    }
}

/// Absolute deadline for `timeout_secs` from the first SSE chunk.
fn stream_deadline_at(state: &StreamEventsState) -> Option<Instant> {
    state.started_at.and_then(|started| started.checked_add(state.timeout))
}

/// Publish the absolute stream cutoff for the streaming executor to copy onto
/// the live body after this body-filter pass.
fn recap_stream_deadline(ctx: &mut HttpFilterContext<'_>, deadline: Instant) {
    ctx.cap_stream_deadline(deadline);
}

/// Whether an `Io` termination is the stream deadline, not a reset.
///
/// Praxis 0.5.4 reports a winning peer `read_timeout` as
/// [`StreamTerminationCause::Io`]. Matching every `Io` would also
/// label truncated chunks and ordinary resets as timeouts. Only an
/// open parser that has already seen a chunk and exhausted
/// `timeout_secs` is a stream timeout.
fn io_exceeded_stream_deadline(state: &StreamEventsState, now: Instant) -> bool {
    state
        .started_at
        .is_some_and(|started| now.duration_since(started) >= state.timeout)
}

/// Treat an IRR idle, deadline, or exhausted stream-budget abort as a timeout.
///
/// Those failures arrive as end-of-stream with [`StreamTerminationCause`]
/// set, not as another SSE chunk, so [`check_timeout`] never ran while
/// the backend was silent. Ordinary `Io` (truncated chunks, resets)
/// is left unhandled so IRR does not replace committed SSE with a
/// timeout error. A backend that already sent a terminal event can
/// still trip a timer while closing the HTTP body; that is not a
/// stream error.
fn record_idle_transport_timeout(ctx: &mut HttpFilterContext<'_>) {
    let Some(cause) = ctx.stream_termination().map(praxis_filter::StreamTermination::cause) else {
        return;
    };
    let timed_out = match cause {
        StreamTerminationCause::IdleTimeout | StreamTerminationCause::DeadlineExceeded => true,
        StreamTerminationCause::Io => ctx
            .get_filter_state::<StreamEventsState>()
            .is_some_and(|state| io_exceeded_stream_deadline(state, Instant::now())),
        _ => false,
    };
    if !timed_out {
        return;
    }
    publish_idle_timeout_if_incomplete(ctx);
    ctx.mark_stream_termination_handled();
}

/// Set timeout error metadata only when the SSE parser never saw a terminal event.
fn publish_idle_timeout_if_incomplete(ctx: &mut HttpFilterContext<'_>) {
    let parser_complete = ctx
        .get_filter_state::<StreamEventsState>()
        .is_some_and(|state| state.completion_state != CompletionState::Open);
    if !parser_complete && ctx.get_metadata("responses.stream_error_code").is_none() {
        ctx.set_metadata("responses.stream_error_code", "server_error");
        ctx.set_metadata(
            "responses.stream_error_message",
            "upstream Responses stream exceeded timeout",
        );
        ctx.set_metadata("responses.skip_persist", "true");
    }
}

/// Parse SSE frames, accumulating state and optionally normalizing output.
fn process_chunk(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, end_of_stream: bool) {
    let Some(bytes) = body.as_ref() else {
        return;
    };

    let Some(mut state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };

    if end_of_stream {
        // The agentic-loop callback can append streamed output to history
        // immediately before this EOS callback. Re-measure that changed
        // baseline before admitting even the final provider chunk.
        state.clear_shared_budget_caches();
    }

    // Aggregate overflow is terminal for the logical stream. Suppress every
    // later upstream frame without reparsing it so the first bounded budget
    // error remains authoritative and cannot be overwritten by an
    // event-after-terminal parse error.
    if ctx.get_metadata("responses.stream_error_message") == Some(RETAINED_PAYLOAD_OVERFLOW_MESSAGE) {
        *body = None;
        ctx.insert_filter_state(state);
        return;
    }

    let now = Instant::now();
    state.started_at.get_or_insert(now);

    let parsed = parse_and_accumulate(&mut state, ctx, bytes, now);
    // #1159 C1: any fatal chunk error poisons the whole logical stream. Mark it
    // sticky before re-inserting state so the next chunk fails closed at the top
    // of `parse_and_accumulate` — a co-batched lowered `output_item.done` whose
    // `.added` was rolled back by this chunk must never emit its raw private name.
    if parsed.is_err() {
        state.stream_failed = true;
    }
    handle_parse_result(ctx, body, &state, parsed);

    if !publish_stream_payload(ctx, &state) {
        *body = None;
        record_retained_payload_overflow(ctx, &mut state);
    }

    if let Some(deadline) = stream_deadline_at(&state) {
        recap_stream_deadline(ctx, deadline);
    }
    ctx.insert_filter_state(state);
}

/// Publish parser state and rewrite logical-stream output when needed.
fn handle_parse_result(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    state: &StreamEventsState,
    parsed: Result<Option<Bytes>, SseParseError>,
) {
    let parsed = match parsed {
        Ok(parsed) => parsed,
        Err(error) => {
            handle_parse_error(ctx, body, &error);
            return;
        },
    };
    let completion = match state.completion_state {
        CompletionState::Open => "open",
        CompletionState::TerminalLifecycle => "terminal",
        CompletionState::Error => "error",
    };
    ctx.set_metadata("responses.stream_completion", completion);
    *body = parsed;
}

/// Record a parse failure and suppress unnormalized logical-stream bytes.
fn handle_parse_error(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, error: &SseParseError) {
    warn!(%error, "SSE parse error in stream_events");
    ctx.set_metadata("responses.stream_parse_error", "true".to_owned());
    ctx.set_metadata("responses.stream_error_code", "server_error");
    // M3: surface a client-tool-restore-specific message when the failure is a
    // restore error. `reason` is client-safe by construction (never a private
    // name; see `client_tools::client_tool_restore_error`). Timeouts keep their
    // dedicated message; all other error kinds keep the generic message.
    // Diagnostic-only — control flow is unchanged.
    match error {
        SseParseError::ClientToolRestore { reason, .. } => {
            ctx.set_metadata(
                "responses.stream_error_message",
                format!("client tool restoration failed: {reason}"),
            );
        },
        SseParseError::Timeout { .. } => {
            ctx.set_metadata(
                "responses.stream_error_message",
                "upstream Responses stream exceeded timeout",
            );
        },
        _ => {
            ctx.set_metadata(
                "responses.stream_error_message",
                "upstream Responses stream could not be parsed",
            );
        },
    }
    ctx.set_metadata("responses.skip_persist", "true");
    *body = None;
}

/// Parse frames from raw bytes and accumulate events.
#[expect(
    clippy::too_many_lines,
    reason = "transactional parser, construction, and emission accounting"
)]
fn parse_and_accumulate(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    bytes: &Bytes,
    now: Instant,
) -> Result<Option<Bytes>, SseParseError> {
    // #1159 C1: a prior chunk already failed the logical stream. Fail every
    // remaining chunk closed before parsing so a later terminal or lowered event
    // cannot emit on a poisoned stream (mirrors the accumulation-budget guard
    // below). This covers ALL error kinds — parse, restore, timeout, budget — not
    // just client-tool restore, so it is not a `client_tool_restore_error`.
    if state.stream_failed {
        return Err(SseParseError::StreamPoisoned);
    }

    check_timeout(state, now)?;

    // A prior chunk or round may have tripped the aggregate accumulation budget.
    // Fail every remaining chunk closed before parsing so a later terminal event
    // cannot commit on a poisoned stream. The budget is request-wide, so this
    // stays tripped across the per-round re-arm as well as across chunks.
    if let Some(error) = accumulation_budget_exceeded(state, ctx) {
        return Err(error);
    }
    // Parsing changes only the stream-local buffers and a numeric accumulation
    // counter in ResponsesState. Reuse one shared JSON measurement for every
    // pre-commit admission in this chunk; measure again after commit mutates
    // the shared response tree.
    let shared_budget = shared_retained_budget(ctx, state);
    // The incoming chunk is framework-owned, but parsing can temporarily own
    // both the current line and the copied data/event field before the line is
    // cleared. An invalid UTF-8 event field can expand each input byte to the
    // three-byte replacement character, so four times the chunk length bounds
    // the line plus its decoded field. Existing parser scratch is already
    // included by `stream_payload_fits`; fields or a data buffer carried from
    // earlier chunks can acquire second owners on line completion.
    let carried_copy_bytes = if shared_budget.is_some() {
        state.frame_parser.previous_chunk_copy_bound(bytes)
    } else {
        Some(0)
    };
    let Some(parser_projection_bytes) = bytes
        .len()
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(carried_copy_bytes?))
    else {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    };
    if !stream_payload_fits_with_budget(state, parser_projection_bytes, shared_budget) {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    }

    let frames = state.frame_parser.parse_chunk_with_counted_event_limit(
        bytes,
        state.event_count,
        state.max_events,
        |frame| frame.data != b"[DONE]",
    )?;

    let frame_payload_bytes = retained_frame_payload_bytes(&frames);
    let event_construction_bytes =
        frames
            .iter()
            .filter(|frame| frame.data != b"[DONE]")
            .try_fold(0_usize, |used, frame| {
                // serde_json can expand compact numeric spellings: `1e15`
                // serializes as `1000000000000000.0`. Reserve that parsed
                // owner before from_frame allocates its Value tree.
                used.checked_add(frame.data.len())
                    .and_then(|used| used.checked_add(json_number_normalization_bound(&frame.data)?))
                    .and_then(|used| used.checked_add(frame.event_type.as_ref().map_or(0, String::len)))
                    // An unknown event with no SSE `event:` line also owns a
                    // copy of its JSON `type` string. It cannot exceed the
                    // complete raw data length.
                    .and_then(|used| {
                        if frame.event_type.is_none() {
                            used.checked_add(frame.data.len())
                        } else {
                            Some(used)
                        }
                    })
            });
    // `frames` stays live while `ResponsesEvent` owns newly parsed JSON values.
    // The incoming chunk is framework-owned and separately bounded, so reserve
    // only the parser products before constructing the event vector.
    let parse_staging_bytes = frame_payload_bytes
        .zip(event_construction_bytes)
        .and_then(|(frames, events)| frames.checked_add(events));
    if !parse_staging_bytes.is_some_and(|bytes| stream_payload_fits_with_budget(state, bytes, shared_budget)) {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    }

    // Parse and validate every frame before mutating shared state or emitting a
    // byte, then commit accumulation and emission only once the whole chunk
    // parses. A malformed frame aborts the chunk atomically, so no local-tool
    // milestone is recorded for bytes that never reach the client and EOS
    // recovery still re-synthesizes the executed tool items (#276 finding 3).
    let events = parse_chunk_events(state, ctx, &frames, now)?;
    let construction_bytes = frame_payload_bytes.and_then(|frame_bytes| {
        events.iter().try_fold(frame_bytes, |used, event| {
            used.checked_add(retained_event_payload_bytes(event)?)
        })
    });
    if !construction_bytes.is_some_and(|staging| stream_payload_fits_with_budget(state, staging, shared_budget)) {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    }
    let Some(logical_output_upper_bound) = logical_output_upper_bound(ctx, state.local_items_flushed, &events) else {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    };
    let projected_state_clone_bytes = projected_responses_state_clone_bytes(ctx, &events);
    let commit_staging_bytes = construction_bytes
        .and_then(|staging| staging.checked_add(projected_state_clone_bytes?))
        .and_then(|staging| staging.checked_add(logical_output_upper_bound.checked_mul(2)?));
    if !commit_staging_bytes.is_some_and(|staging| stream_payload_fits_with_budget(state, staging, shared_budget)) {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    }
    // The existing two-phase commit enforces the stream's own item and byte caps
    // before it records any client-visible lifecycle milestones.
    let staging = CommitChunkStaging {
        parsed_owner_bytes: construction_bytes.unwrap_or(usize::MAX),
        frame_payload_bytes,
        logical_output_upper_bound,
    };
    let Some(logical_output) = commit_chunk_events(state, ctx, events, &staging)? else {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    };
    let output_staging_bytes = frame_payload_bytes.and_then(|frames| frames.checked_add(logical_output.len()));
    if !output_staging_bytes.is_some_and(|staging| stream_payload_fits(ctx, state, staging)) {
        record_retained_payload_overflow(ctx, state);
        return Ok(None);
    }

    Ok((!logical_output.is_empty()).then(|| Bytes::from(logical_output)))
}

/// Count the independently owned payloads in the completed frame collection.
fn retained_frame_payload_bytes(frames: &[SseFrame]) -> Option<usize> {
    frames.iter().try_fold(0_usize, |used, frame| {
        used.checked_add(frame.event_type.as_ref().map_or(0, String::len))?
            .checked_add(frame.data.len())
    })
}

/// Bound compact JSON growth when `serde_json` normalizes exponent notation.
///
/// This build does not enable `serde_json`'s arbitrary-precision feature, so a
/// parsed number is an i64/u64/f64; 32 extra bytes per lexical number cover
/// its longest compact output. Strings and escaped quotes are skipped, and the
/// source length is already charged separately by the caller. Invalid JSON
/// may overcount but is rejected by the parser without committing any event.
fn json_number_normalization_bound(data: &[u8]) -> Option<usize> {
    let mut in_string = false;
    let mut escaped = false;
    let mut in_number = false;
    let mut numbers = 0_usize;
    for &byte in data {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
            in_number = false;
            continue;
        }
        if in_number && matches!(byte, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
            continue;
        }
        in_number = false;
        if matches!(byte, b'0'..=b'9' | b'-') {
            numbers = numbers.checked_add(1)?;
            in_number = true;
        }
    }
    numbers.checked_mul(32)
}

/// Count one parsed event's independently owned payloads.
fn retained_event_payload_bytes(event: &ResponsesEvent) -> Option<usize> {
    let payload_bytes = retained_json_bytes(event.payload())?;
    match event {
        // Unknown events retain a second owned copy of their discriminator;
        // known variants use the static match arm returned by `event_type()`.
        ResponsesEvent::Unknown { event_type, .. } => payload_bytes.checked_add(event_type.len()),
        _ => Some(payload_bytes),
    }
}

/// Count payloads that accumulation may clone into [`ResponsesState`] while
/// the parsed event vector is still live.
///
/// This is intentionally an upper bound: replacement writes may release the
/// previous state owner during commit, but charging the projected owner keeps
/// the preflight transactional at the allocation peak.
#[expect(clippy::too_many_lines, reason = "event variants require distinct clone projections")]
fn projected_responses_state_clone_bytes(ctx: &HttpFilterContext<'_>, events: &[ResponsesEvent]) -> Option<usize> {
    let state = ctx.extensions.get::<ResponsesState>();
    events.iter().try_fold(0_usize, |used, event| {
        let event_bytes = retained_event_payload_bytes(event)?;
        let additional = match event {
            ResponsesEvent::ResponseCompleted(payload)
            | ResponsesEvent::ResponseIncomplete(payload)
            | ResponsesEvent::ResponseFailed(payload) => {
                let response = payload.get("response").unwrap_or(payload);
                let response_bytes = retained_json_bytes(response)?;
                let previous_usage_bytes = state.map_or(Some(0), |state| retained_json_bytes(&state.usage))?;
                let incoming_usage_bytes = response.get("usage").map_or(Some(0), retained_json_bytes)?;
                // The conformant terminal response moves from the parsed event
                // into shared state. Usage merging can still grow the distinct
                // accumulator and clone the merged usage into that response.
                let usage_peak = previous_usage_bytes.checked_add(incoming_usage_bytes)?.checked_mul(2)?;
                if payload.get("response").is_some() {
                    usage_peak
                } else {
                    // Preserve the old bare-response fallback, which clones
                    // the whole payload so its envelope can still be emitted.
                    usage_peak.checked_add(response_bytes)?
                }
            },
            ResponsesEvent::OutputItemAdded(payload) | ResponsesEvent::OutputItemDone(payload) => {
                let item_bytes = payload.get("item").map_or(Some(0), retained_json_bytes)?;
                // Besides the canonical output item, logical-stream
                // reconciliation can retain item ids, local-tool keys, and a
                // normalized payload while the parsed event is still live.
                item_bytes.checked_add(event_bytes.checked_mul(2)?)?
            },
            ResponsesEvent::FunctionCallArgumentsDelta(payload) => {
                let key_bytes = projected_tool_call_key_bytes(payload)?;
                let delta_bytes = payload.get("delta").and_then(Value::as_str).map_or(0, str::len);
                key_bytes.checked_add(delta_bytes)?.checked_add(event_bytes)?
            },
            ResponsesEvent::FunctionCallArgumentsDone(_) => {
                // Lowered client-tool completion can snapshot this item too.
                // That copy is reserved against the live item during commit,
                // including when added earlier in the same chunk.
                event_bytes.checked_mul(3)?
            },
            // Other events can still acquire logical-stream bookkeeping keys
            // or grow their normalized payload before serialization.
            _ => event_bytes,
        };
        used.checked_add(additional)
    })
}

/// Bound the client-tool restoration plan before it clones any response
/// snapshots. Each nonterminal snapshot may acquire the echoed tools, a
/// restored output item, a serialized event, and a growing output buffer while
/// the original parsed event remains live. Native streams pay no plan charge.
#[expect(
    clippy::too_many_lines,
    reason = "checked bound covers all restoration owners together"
)]
fn projected_client_tool_restore_bytes(
    ctx: &HttpFilterContext<'_>,
    stream: &StreamEventsState,
    events: &[ResponsesEvent],
) -> Option<usize> {
    let Some(responses) = ctx.extensions.get::<ResponsesState>() else {
        return Some(0);
    };
    if responses.client_tool_lowering.is_empty() {
        return Some(0);
    }
    let max_name_bytes = responses
        .client_tool_lowering
        .values()
        .try_fold(0_usize, |largest, lowered| {
            let bytes = lowered
                .original_name
                .len()
                .checked_add(lowered.namespace.as_ref().map_or(0, String::len))?
                .checked_add(64)?;
            Some(largest.max(bytes))
        })?;
    // The plan clones its tracked item list even if this chunk has no snapshot.
    let tracked = stream.client_tool_items.iter().try_fold(0_usize, |used, item| {
        used.checked_add(item.key.len())?
            .checked_add(item.private_name.len())?
            .checked_add(item.item_id.as_ref().map_or(0, String::len))
    })?;
    let mut projected = tracked.checked_mul(2)?;
    let mut cached_echo_bytes = None;
    for (index, event) in events.iter().enumerate() {
        projected = projected.checked_add(projected_restored_item_name_bytes(
            responses,
            stream,
            events.get(..index)?,
            event,
            max_name_bytes,
        )?)?;
        if event.is_terminal() {
            continue;
        }
        let Some(snapshot) = event.payload().get("response") else {
            continue;
        };
        let echo_bytes = if let Some(bytes) = cached_echo_bytes {
            bytes
        } else {
            let bytes = responses.client_tool_echo.as_ref().map_or(Some(0), |echo| {
                retained_json_values_bytes(&echo.tools)?.checked_add(retained_json_bytes(&echo.tool_choice)?)
            })?;
            cached_echo_bytes = Some(bytes);
            bytes
        };
        let source_bytes = retained_json_bytes(snapshot)?;
        let item_count = snapshot.get("output").and_then(Value::as_array).map_or(0, Vec::len);
        let name_growth = item_count.checked_mul(max_name_bytes)?;
        // Eight source copies cover the retained plan snapshot, temporary
        // output-item conversions, and JSON normalization. Four echoed copies
        // cover tool cloning and event/wire construction before old owners die.
        let snapshot_peak = source_bytes
            .checked_mul(8)?
            .checked_add(echo_bytes.checked_mul(4)?)?
            .checked_add(name_growth.checked_mul(4)?)?
            .checked_add(256)?;
        projected = projected.checked_add(snapshot_peak)?;
    }
    Some(projected)
}

/// An individual item event has no `response` snapshot. The restoration plan
/// still owns a public name and namespace for each lowered item while the
/// retyped event and growing wire buffer can hold further encoded copies.
/// Reserve these copies before planning a co-batched set of item events.
#[expect(
    clippy::too_many_lines,
    reason = "identify tracked and co-batched item ownership without allocating keys"
)]
fn projected_restored_item_name_bytes(
    responses: &ResponsesState,
    stream: &StreamEventsState,
    prior_events: &[ResponsesEvent],
    event: &ResponsesEvent,
    max_name_bytes: usize,
) -> Option<usize> {
    let (payload, is_done) = match event {
        ResponsesEvent::OutputItemAdded(payload) => (payload, false),
        ResponsesEvent::OutputItemDone(payload) => (payload, true),
        _ => return Some(0),
    };
    let item = payload.get("item");
    if !is_done && item.and_then(|item| item.get("type")).and_then(Value::as_str) != Some("function_call") {
        return Some(0);
    }
    let name = item.and_then(|item| item.get("name")).and_then(Value::as_str);
    if is_done {
        let tracked = stream
            .client_tool_items
            .iter()
            .find(|tracked| tracked_item_key_matches(&tracked.key, payload));
        if let Some(tracked) = tracked {
            if name != Some(tracked.private_name.as_str()) {
                return max_name_bytes.checked_mul(21)?.checked_add(128);
            }
        } else if name.is_none_or(|name| !responses.client_tool_lowering.contains_key(name))
            && !prior_events.iter().any(|prior| {
                matches!(prior, ResponsesEvent::OutputItemAdded(added)
                    if item_event_keys_match(added, payload)
                        && added.get("item")
                            .and_then(|item| item.get("name"))
                            .and_then(Value::as_str)
                            .is_some_and(|name| responses.client_tool_lowering.contains_key(name)))
            })
        {
            // An unrelated native function call does not use any lowering.
            return Some(0);
        }
    }
    let Some(lowered) = name.and_then(|name| responses.client_tool_lowering.get(name)) else {
        // A done item is resolved by its previously tracked ID, even when its
        // name is absent or differs from the added item. Charge the largest
        // possible restoration when this payload cannot identify the lowering.
        return if is_done {
            max_name_bytes.checked_mul(21)?.checked_add(128)
        } else {
            Some(0)
        };
    };
    if !matches!(
        lowered.restore,
        ClientToolRestore::Namespace | ClientToolRestore::Custom | ClientToolRestore::NamespaceCustom
    ) {
        return Some(0);
    }
    let native = lowered
        .original_name
        .len()
        .checked_add(lowered.namespace.as_ref().map_or(0, String::len))?
        .checked_add(64)?;
    let encoded = retained_json_bytes(lowered.original_name.as_str())?
        .checked_add(lowered.namespace.as_deref().map_or(Some(0), retained_json_bytes)?)?;
    native
        .checked_mul(3)?
        .checked_add(encoded.checked_mul(3)?)?
        .checked_add(128)
}

/// Read the ID chosen by the restoration planner, without copying it.
fn item_event_id(payload: &Value) -> Option<&str> {
    payload
        .get("item")
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
        .or_else(|| payload.get("item_id").and_then(Value::as_str))
}

/// Compare two item events using the planner's ID-first, index-fallback key.
fn item_event_keys_match(added: &Value, done: &Value) -> bool {
    match (item_event_id(added), item_event_id(done)) {
        (Some(left), Some(right)) => left == right,
        (None, None) => added
            .get("output_index")
            .and_then(Value::as_u64)
            .is_some_and(|index| done.get("output_index").and_then(Value::as_u64) == Some(index)),
        _ => false,
    }
}

/// Compare an existing tracked key with an item event without formatting a new key.
fn tracked_item_key_matches(key: &str, payload: &Value) -> bool {
    if let Some(id) = item_event_id(payload) {
        return key.strip_prefix("item:") == Some(id);
    }
    key.strip_prefix("index:").and_then(|index| index.parse::<u64>().ok())
        == payload.get("output_index").and_then(Value::as_u64)
}

/// Bound the owned key created while accumulating a function-call argument
/// delta, without allocating the formatted key.
fn projected_tool_call_key_bytes(payload: &Value) -> Option<usize> {
    if let Some(item_id) = payload.get("item_id").and_then(Value::as_str) {
        return 5_usize.checked_add(item_id.len());
    }
    let Some(index) = payload.get("output_index").and_then(Value::as_u64) else {
        return Some(0);
    };
    let digits = if index == 0 {
        1
    } else {
        usize::try_from(index.ilog10()).ok()?.checked_add(1)?
    };
    6_usize.checked_add(digits)
}

/// Return whether shared response state, filter-local parser state, and one
/// transient construction/staging buffer fit the active aggregate budget.
fn stream_payload_fits(ctx: &HttpFilterContext<'_>, state: &StreamEventsState, staging_bytes: usize) -> bool {
    stream_payload_fits_with_budget(state, staging_bytes, shared_retained_budget(ctx, state))
}

/// Snapshot the shared retained JSON total once for pre-commit checks.
/// `None` means no aggregate budget is active; the inner `None` signals that
/// the current retained state already exceeds the limit.
#[expect(
    clippy::too_many_lines,
    reason = "stable and changing shared owners need one coherent budget snapshot"
)]
fn shared_retained_budget(ctx: &HttpFilterContext<'_>, stream: &StreamEventsState) -> Option<(usize, Option<usize>)> {
    let responses = ctx.extensions.get::<ResponsesState>()?;
    let limit = responses.retained_payload_limit()?;
    let stable = *stream.shared_stable_bytes.get_or_init(|| {
        responses
            .stream_stable_payload_bytes_bounded(limit)
            .unwrap_or(usize::MAX)
    });
    // A resumed round can carry megabytes of completed prior output. It does
    // not change between ordinary chunks, so measure it once; an in-place
    // update or round transition invalidates the cached shape.
    let prior = stream.shared_prior_output_bytes.get_or_init(|| {
        let revision = responses.replay_stable_payload_revision?;
        let mut meter = PayloadMeter::new(limit);
        meter.json_values(&responses.accumulated_output)?;
        Some(PriorOutputCache {
            iteration: responses.iteration,
            revision,
            output_len: responses.accumulated_output.len(),
            bytes: meter.used(),
        })
    });
    let prior_bytes = match (prior, responses.replay_stable_payload_revision) {
        (Some(cache), Some(_)) if cache.matches(responses) => Some(cache.bytes),
        (_, Some(_)) => {
            let mut meter = PayloadMeter::new(limit);
            meter.json_values(&responses.accumulated_output).map(|()| meter.used())
        },
        (_, None) => None,
    };
    let cached_current_output_bytes = stream.shared_current_output_bytes.load(Ordering::Relaxed);
    let current_output_bytes = (cached_current_output_bytes != 0)
        .then_some(cached_current_output_bytes)
        .or_else(|| {
            let mut meter = PayloadMeter::new(limit);
            meter.json(&responses.response_object)?;
            meter.json(&responses.local_completion_response_template)?;
            let bytes = meter.used();
            stream.shared_current_output_bytes.store(bytes, Ordering::Relaxed);
            Some(bytes)
        });
    let current = limit
        .checked_sub(stable)
        .and_then(|remaining| remaining.checked_sub(prior_bytes?))
        .and_then(|remaining| {
            responses.stream_changing_payload_bytes_bounded_with_current_output(remaining, current_output_bytes?)
        })
        .and_then(|changing| stable.checked_add(prior_bytes?)?.checked_add(changing));
    Some((limit, current))
}

/// Check a staging projection against a previously measured shared state.
fn stream_payload_fits_with_budget(
    state: &StreamEventsState,
    staging_bytes: usize,
    shared_budget: Option<(usize, Option<usize>)>,
) -> bool {
    let Some(local_bytes) = state.retained_payload_bytes() else {
        return false;
    };
    let Some(external_bytes) = local_bytes.checked_add(staging_bytes) else {
        return false;
    };
    match shared_budget {
        None => true,
        Some((limit, Some(current))) => current.checked_add(external_bytes).is_some_and(|total| total <= limit),
        Some((_, None)) => false,
    }
}

/// Return an allocation-safe upper bound for one normalized SSE event.
///
/// Normalization can replace an existing response id/output index and always
/// writes a sequence number. Count the original JSON plus the replacement
/// values and fixed envelope overhead; retaining the old values in the bound is
/// deliberate and makes this an upper bound without cloning the payload.
fn normalized_sse_event_upper_bound(ctx: &HttpFilterContext<'_>, event_type: &str, payload: &Value) -> Option<usize> {
    let mut bound = retained_json_bytes(payload)?
        .checked_add(event_type.len())?
        .checked_add(b"event: \ndata: \n\n".len())?
        .checked_add(64)?; // sequence_number key/value and object punctuation
    let Some(object) = payload.as_object() else {
        return Some(bound);
    };

    if object.contains_key("output_index") {
        bound = bound.checked_add(20)?; // max serialized u64
    }
    let logical_id = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|state| state.logical_stream_response_id.as_deref());
    if let Some(logical_id) = logical_id {
        let id_bytes = retained_json_bytes(logical_id)?;
        if object.contains_key("response_id") {
            bound = bound.checked_add(id_bytes)?;
        }
        if object
            .get("response")
            .and_then(Value::as_object)
            .is_some_and(|response| response.contains_key("id"))
        {
            bound = bound.checked_add(id_bytes)?;
        }
    }
    Some(bound)
}

/// Return a conservative bound for the lifecycle SSE events synthesized for a
/// locally executed output item. The item is serialized at most twice (added
/// and done), while up to three phase events repeat its id. The fixed allowance
/// covers SSE/object envelopes and normalized sequence/response-id fields.
fn local_item_sse_upper_bound(state: &ResponsesState, item: &Value) -> Option<usize> {
    let item_bytes = retained_json_bytes(item)?;
    let item_id_bytes = item
        .get("id")
        .and_then(Value::as_str)
        .map_or(Some(0), retained_json_bytes)?;
    let logical_id_bytes = state
        .logical_stream_response_id
        .as_deref()
        .map_or(Some(0), retained_json_bytes)?;
    item_bytes
        .checked_mul(5)?
        .checked_add(item_id_bytes.checked_mul(3)?)?
        .checked_add(logical_id_bytes.checked_mul(2)?)?
        .checked_add(5 * 320)
}

/// Bound local lifecycle output without constructing any synthesized payloads.
fn local_terminal_output_upper_bound(state: &ResponsesState) -> Option<usize> {
    let mut bound = 0_usize;
    for item in &state.accumulated_output {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        if is_local_tool_item(item) && state.locally_executed_output_items.contains(id) {
            bound = bound.checked_add(local_item_sse_upper_bound(state, item)?)?;
        }
    }
    for (index, _) in &state.pending_local_tool_synthesis {
        let item = state.accumulated_output.get(*index)?;
        bound = bound.checked_add(local_item_sse_upper_bound(state, item)?)?;
    }
    Some(bound)
}

/// Reserve the owned ready-item copies and wire buffer before file-search EOS
/// synthesis. Pending items are only re-queued, so they create no payload here.
/// The lifecycle bound covers the two item envelopes, progress events, and
/// temporary payload/Vec growth; the extra item charge covers the ready queue's
/// clone, which stays live while those events are built.
fn ready_file_search_synthesis_staging_bytes(state: &ResponsesState) -> Option<usize> {
    state
        .pending_local_tool_synthesis
        .iter()
        .try_fold(0_usize, |used, (index, _)| {
            let item = state.accumulated_output.get(*index)?;
            if !matches!(
                item.get("status").and_then(Value::as_str),
                Some("completed" | "incomplete")
            ) {
                return Some(used);
            }
            used.checked_add(retained_json_bytes(item)?)?
                .checked_add(local_item_sse_upper_bound(state, item)?)
        })
}

/// Request-side local terminal builders already reserve the synthesized wire;
/// add the ready queue's independent item clones before either builder drains it.
fn ready_file_search_clone_bytes(state: &ResponsesState) -> Option<usize> {
    state
        .pending_local_tool_synthesis
        .iter()
        .try_fold(0_usize, |used, (index, _)| {
            let item = state.accumulated_output.get(*index)?;
            if !matches!(
                item.get("status").and_then(Value::as_str),
                Some("completed" | "incomplete")
            ) {
                return Some(used);
            }
            used.checked_add(retained_json_bytes(item)?)
        })
}

/// Bound all output that `commit_chunk_events` can append for this chunk.
fn logical_output_upper_bound(
    ctx: &HttpFilterContext<'_>,
    local_items_flushed: bool,
    events: &[ResponsesEvent],
) -> Option<usize> {
    // The chunk commit cannot flush local items again after its first model
    // content event. Terminal synthesis has a separate admission check.
    let mut bound = if local_items_flushed {
        0
    } else {
        ctx.extensions
            .get::<ResponsesState>()
            .map_or(Some(0), local_terminal_output_upper_bound)?
    };
    for event in events {
        // Terminal payloads are held for finalization, not appended to this
        // chunk. Their wire bytes are reserved when the finalizer emits them.
        if event.is_terminal() {
            continue;
        }
        bound = bound.checked_add(normalized_sse_event_upper_bound(
            ctx,
            event.event_type(),
            event.payload(),
        )?)?;
    }
    Some(bound)
}

/// Reserve the transient owners created before a logical terminal is emitted.
///
/// Canonicalization moves the accumulated output into `response_object` and
/// borrows it for the wire. Reserve metadata, usage, citation, and client-tool
/// restoration copies before mutating the canonical response.
#[expect(
    clippy::too_many_lines,
    reason = "one terminal projection covers several independent owners"
)]
fn canonicalization_staging_bytes(state: &ResponsesState, existing_output_bytes: usize) -> Option<usize> {
    let output = if state.accumulated_output.is_empty() {
        state.output_items()
    } else {
        &state.accumulated_output
    };
    // Annotation staging includes the cleaned text and projected citation
    // objects. Reserve the temporary annotation owners before rewriting the
    // canonical output in place.
    let annotation_bytes = crate::openai::responses::file_search_callout::citations::annotation_staging_bytes(
        output,
        &state.citation_files,
    )
    .ok()?;
    let id_bytes = state
        .logical_stream_response_id
        .as_ref()
        .map_or(Some(0), retained_json_bytes)?;
    let previous_id_bytes = state
        .previous_response_id
        .as_ref()
        .map_or(Some(0), retained_json_bytes)?;
    let usage_bytes = if state.usage.is_null() {
        0
    } else {
        retained_json_bytes(&state.usage)?
    };
    let client_tool_bytes = if state.client_tool_lowering.is_empty() {
        0
    } else {
        // Rich client-tool restoration parses and rebuilds individual output
        // items in place. A second output-sized allowance bounds those local
        // construction owners. Namespaced calls also copy the public name and
        // namespace into every restored item; one reverse-map entry can fan
        // out across many short private calls.
        let restored_names = output.iter().try_fold(0_usize, |used, item| {
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                return Some(used);
            }
            let Some(lowered) = item
                .get("name")
                .and_then(Value::as_str)
                .and_then(|name| state.client_tool_lowering.get(name))
            else {
                return Some(used);
            };
            if !matches!(
                lowered.restore,
                ClientToolRestore::Namespace | ClientToolRestore::NamespaceCustom
            ) {
                return Some(used);
            }
            used.checked_add(retained_json_bytes(&lowered.original_name)?)?
                .checked_add(lowered.namespace.as_ref().map_or(Some(0), retained_json_bytes)?)?
                .checked_add(64)
        })?;
        retained_json_values_bytes(output)?
            .checked_mul(2)?
            .checked_add(restored_names)?
    };
    let echo_bytes = state.client_tool_echo.as_ref().map_or(Some(0), |echo| {
        // Restoration adds one tools/choice owner to the canonical response;
        // the deferred terminal only borrows that response for the wire.
        retained_json_values_bytes(&echo.tools)?
            .checked_add(retained_json_bytes(&echo.tool_choice)?)?
            .checked_add(64)
    })?;
    annotation_bytes
        .checked_mul(3)?
        .checked_add(id_bytes)?
        .checked_add(previous_id_bytes)?
        .checked_add(usage_bytes)?
        .checked_add(client_tool_bytes)?
        .checked_add(echo_bytes)?
        .checked_add(existing_output_bytes)
}

/// Abort an offending chunk after aggregate admission fails. The caller drops
/// its bytes, the logical finalizer emits exactly one bounded error event, and
/// all dispatch/persistence paths are disabled before another round can run.
fn record_retained_payload_overflow(ctx: &mut HttpFilterContext<'_>, state: &mut StreamEventsState) {
    warn!("Responses streaming state exceeded openai_agentic_loop.max_retained_bytes");
    if let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() {
        responses.discard_payload_for_budget_error();
        responses.deferred_stream_done = false;
    }
    state.tool_call_args.clear();
    state.rejected_tool_call_args.clear();
    state.frame_parser.clear();
    state.deferred_terminal = None;
    state.deferred_done = false;
    state.local_tool_items.clear();
    state.client_tool_items.clear();
    if let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() {
        responses.retained_stream_parser_bytes = 0;
    }
    state.stream_failed = true;
    state.completion_state = CompletionState::Error;
    state.completed_at.get_or_insert_with(Instant::now);
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.stream_error_message", RETAINED_PAYLOAD_OVERFLOW_MESSAGE);
    ctx.set_metadata("responses.skip_persist", "true");
    crate::openai::responses::fs_arm_stream_stop(ctx);
}

/// Phase 1: parse and validate every frame in a chunk before any mutation.
///
/// Returns the parsed non-`[DONE]` events, failing closed on the first malformed
/// frame so the caller can discard the whole chunk without having recorded any
/// local-tool milestone (#276 finding 3).
fn parse_chunk_events(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    frames: &[SseFrame],
    now: Instant,
) -> Result<Vec<ResponsesEvent>, SseParseError> {
    let mut events = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.data == b"[DONE]" {
            state.deferred_done = true;
            continue;
        }

        state.event_count += 1;
        let event = ResponsesEvent::from_frame(frame)?;
        record_completion(state, &event, now)?;
        charge_accumulation_budget(state, ctx, &event, frame)?;
        events.push(event);
    }
    Ok(events)
}

/// Charge one parsed event's wire bytes against the request-wide accumulation
/// byte budget.
///
/// Bounds every accumulator this filter grows *from the driving frame* — the
/// response output-item list, the per-tool-call argument buffers, the local-tool
/// `emitted_output_items` map (keyed by an owned `item_id`, with a
/// `streamed_phases` set of owned event-type strings), and the terminal snapshot
/// retained in `response_object` and the deferred terminal — by an aggregate byte
/// ceiling. Each of these grows by a substring of the frame that drives it, so
/// `frame.data.len()` is a conservative upper bound on the bytes each event
/// contributes to shared state, and the budget bounds total accumulated memory even
/// when every individual event stays within `max_buffer_bytes`.
///
/// The running total lives in [`ResponsesState::stream_accumulated_bytes`], so it
/// is charged once per request and survives the per-round re-arm: a multi-round
/// stream cannot reset the counter between IRR rounds and accumulate unbounded
/// state while no single round trips the cap.
///
/// `frame.data.len()` is the compact JSON wire size, a proxy that undercounts the
/// parsed `serde_json::Value` heap footprint (per-entry `String` keys and enum
/// discriminants) by a bounded constant factor; the ceiling therefore bounds
/// memory up to that factor, not to the byte. This is deliberate: charging wire
/// bytes keeps the budget aligned with what a backend can actually stream and
/// avoids rejecting a streamed response whose equivalent non-streaming body the
/// buffered path would accept.
///
/// Terminal lifecycle events (`response.completed`/`incomplete`/`failed`) are
/// charged: their frame owns a full response snapshot while the parsed response
/// moves into `response_object`, so a terminal frame that alone exceeds the
/// ceiling (yet still fits `max_buffer_bytes`) must fail closed. The deferred
/// terminal retains only envelope metadata after that move. The
/// per-frame `added`/`done`/terminal charges over-count an item that also streams
/// the paired envelopes; that is a deliberately conservative, fail-closed-earlier
/// byte bound. The distinct item-count dimension is enforced separately from the
/// retained output (see [`accumulation_count_exceeded`]).
///
/// Runs in phase 1 (parse) so a frame-bounded accumulator's growth aborts the chunk
/// atomically before [`commit_chunk_events`] mutates shared state.
fn charge_accumulation_budget(
    state: &StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    event: &ResponsesEvent,
    frame: &SseFrame,
) -> Result<(), SseParseError> {
    if !charges_accumulation_bytes(event) {
        return Ok(());
    }
    let accumulated_bytes = {
        let responses = ctx.extensions.get_or_insert_with(ResponsesState::default);
        responses.stream_accumulated_bytes = responses.stream_accumulated_bytes.checked_add(frame.data.len()).ok_or(
            SseParseError::AccumulationLimitExceeded {
                dimension: "accumulated_bytes",
                value: usize::MAX,
                limit: state.max_accumulated_bytes,
            },
        )?;
        responses.stream_accumulated_bytes
    };
    accumulation_bytes_exceeded(state, accumulated_bytes).map_or(Ok(()), Err)
}

/// Whether an event's wire bytes grow retained accumulation state and so must be
/// charged against the aggregate byte budget.
fn charges_accumulation_bytes(event: &ResponsesEvent) -> bool {
    match event {
        ResponsesEvent::OutputItemAdded(_)
        | ResponsesEvent::OutputItemDone(_)
        | ResponsesEvent::FunctionCallArgumentsDelta(_)
        | ResponsesEvent::FunctionCallArgumentsDone(_)
        | ResponsesEvent::ResponseCompleted(_)
        | ResponsesEvent::ResponseIncomplete(_)
        | ResponsesEvent::ResponseFailed(_) => true,
        // Local-tool progress events (`response.web_search_call.*`,
        // `response.mcp_call.*`, `response.mcp_list_tools.*`) grow
        // `emitted_output_items` via `record_model_output_item`; charge them so
        // that accumulator cannot be inflated by many distinct `item_id`s or
        // event-type suffixes while the byte ceiling stays at zero.
        ResponsesEvent::Unknown { event_type, .. } => is_local_tool_progress_event(event_type),
        _ => false,
    }
}

/// The sticky accumulation-budget error, if any dimension is over its cap.
///
/// Returned as a fast-path guard on later chunks — including the first chunk of a
/// re-armed round — so once either the request-wide byte total or the retained
/// item count is over its cap the stream stays failed closed.
fn accumulation_budget_exceeded(state: &StreamEventsState, ctx: &HttpFilterContext<'_>) -> Option<SseParseError> {
    let accumulated_bytes = ctx
        .extensions
        .get::<ResponsesState>()
        .map_or(0, |responses| responses.stream_accumulated_bytes);
    accumulation_bytes_exceeded(state, accumulated_bytes).or_else(|| accumulation_count_exceeded(state, ctx))
}

/// The byte-budget error, if the request-wide charged total is over its cap.
fn accumulation_bytes_exceeded(state: &StreamEventsState, accumulated_bytes: usize) -> Option<SseParseError> {
    (accumulated_bytes > state.max_accumulated_bytes).then_some(SseParseError::AccumulationLimitExceeded {
        dimension: "accumulated_bytes",
        value: accumulated_bytes,
        limit: state.max_accumulated_bytes,
    })
}

/// The item-count error, if the request-wide retained output exceeds its cap.
///
/// The count is the number of retained output items — those already drained into
/// [`ResponsesState::accumulated_output`] by preceding rounds plus the current
/// round's live `output` array — not a per-envelope tally. Because
/// `output_item.done` replaces in place the item its matching `output_item.added`
/// pushed (by output index or id), a canonical `added`/`done` pair is one retained
/// item, so deriving the count from the retained lists dedups that pair and spans
/// every round with no per-round counter to reset. The two lists never overlap
/// while a round streams: the agentic loop moves a round's `output` into
/// `accumulated_output` (via `mem::take`) only at the round boundary, so the sum
/// is monotonic across the whole request and the guard stays sticky once tripped.
fn accumulation_count_exceeded(state: &StreamEventsState, ctx: &HttpFilterContext<'_>) -> Option<SseParseError> {
    let count = ctx.extensions.get::<ResponsesState>().map_or(0, |responses| {
        responses
            .accumulated_output
            .len()
            .saturating_add(responses.output_items().len())
    });
    (count > state.max_output_items).then_some(SseParseError::AccumulationLimitExceeded {
        dimension: "output_items",
        value: count,
        limit: state.max_output_items,
    })
}

/// Transient owners needed at each phase of one chunk commit.
struct CommitChunkStaging {
    /// Parsed frames and events remain live after accumulation. The pre-commit
    /// clone projection is omitted once those clones enter shared state.
    parsed_owner_bytes: usize,
    /// Completed frame data remains live during restoration planning.
    frame_payload_bytes: Option<usize>,
    /// Maximum logical wire output built from this chunk.
    logical_output_upper_bound: usize,
}

/// Phase 2: commit accumulation and logical emission for a fully parsed chunk.
///
/// Split into two passes so both retained-state budgets are validated before any
/// delivery milestone is recorded:
///
/// - Phase 2a accumulates every event into the retained output (`output_items`, `response_object`) — the only state the
///   count derives from, and state no delivery milestone depends on. Lowered-tool completion snapshots are admitted
///   before cloning each completed item.
/// - The count guard then runs against the grown output, *before* any delivery milestone is recorded. A chunk that
///   overflows the item cap therefore fails closed without leaving a committed local-tool milestone that EOS recovery
///   would trust for bytes the client never received (review finding: the former post-commit count check let an
///   already-executed tool's milestone survive a rejected chunk, dropping that tool from the client-visible stream).
/// - Phase 2b records milestones and emits the logical bytes.
///
/// On a rejected count-overflow chunk phase 2a has already grown the retained output past a cap, so
/// the request-wide guards in [`accumulation_budget_exceeded`] stay sticky for every
/// later chunk and round. Phase 2b's only
/// fallible step, the client-tool restoration plan pass (see
/// [`restore_and_append_chunk`]), runs before any byte is appended or milestone is
/// recorded, so a malformed lowered lifecycle fails the chunk closed with nothing
/// delivered; every recorded milestone therefore still corresponds to bytes that
/// actually reach the client. Returns `None` after an aggregate payload failure;
/// otherwise returns the logical-stream bytes.
#[expect(
    clippy::too_many_lines,
    reason = "two-phase event commit and restoration admission share one atomic boundary"
)]
fn commit_chunk_events(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    mut events: Vec<ResponsesEvent>,
    staging: &CommitChunkStaging,
) -> Result<Option<Vec<u8>>, SseParseError> {
    // Phase 2a accumulates shared output and captures charged completion snapshots.
    let Some((completions, completion_bytes)) = accumulate_chunk(state, ctx, &mut events, staging.parsed_owner_bytes)?
    else {
        return Ok(None);
    };

    let result = (|| {
        // The retained item count only exists after phase 2a grows it. Enforce it here,
        // before phase 2b records any delivery milestone, so a rejected chunk leaves no
        // milestone for undelivered bytes. Any overshoot is bounded to a single chunk.
        if let Some(error) = accumulation_count_exceeded(state, ctx) {
            return Err(error);
        }

        // Native passthrough has no restoration plan. For lowered traffic,
        // remeasure event owners after accumulation because a terminal response
        // may already have moved from the event into shared state.
        let has_restoration_plan = ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|responses| !responses.client_tool_lowering.is_empty());
        if has_restoration_plan {
            let post_commit_staging = staging
                .frame_payload_bytes
                .and_then(|frame_bytes| {
                    events.iter().try_fold(frame_bytes, |used, event| {
                        used.checked_add(retained_event_payload_bytes(event)?)
                    })
                })
                .and_then(|bytes| bytes.checked_add(staging.logical_output_upper_bound.checked_mul(2)?))
                .and_then(|bytes| bytes.checked_add(projected_client_tool_restore_bytes(ctx, state, &events)?));
            if !post_commit_staging.is_some_and(|bytes| stream_payload_fits(ctx, state, bytes)) {
                return Ok(None);
            }
        }

        // Phase 2b: plan lowered client-tool restoration (fallible) then append every
        // committed event to the logical stream applying its disposition (infallible).
        let logical_output = restore_and_append_chunk(state, ctx, events, &completions)?;

        // Mirror the parser's deferred-`[DONE]` decision into shared response state,
        // but only now that the whole chunk has parsed and committed. Filter-local
        // parser state is re-armed before request-side dispatchers run on the next
        // IRR step, so the sentinel must survive in shared state as well.
        if state.deferred_done
            && let Some(response_state) = ctx.extensions.get_mut::<ResponsesState>()
        {
            response_state.deferred_stream_done = true;
        }

        Ok(Some(logical_output))
    })();
    // The restoration plan is the last owner of these snapshots. Release their
    // aggregate charge on success and on every count/restore error.
    release_client_tool_completions(ctx, completions, completion_bytes);
    result
}

/// Completion snapshots and the aggregate charge held until restoration ends.
type ChargedClientToolCompletions = (Vec<client_tools::ClientToolCompletion>, usize);

/// Parsed owners stay live for a chunk while its shared-state estimate grows.
struct CompletionAdmission {
    /// Parsed SSE frames and event values still owned during accumulation.
    parsed_owner_bytes: usize,
    /// Conservative shared charge after prior events and their snapshots.
    shared_upper_bound: Option<usize>,
}

/// Drop completion snapshots before releasing their shared budget charge.
fn release_client_tool_completions(
    ctx: &mut HttpFilterContext<'_>,
    completions: Vec<client_tools::ClientToolCompletion>,
    bytes: usize,
) {
    drop(completions);
    release_client_tool_charge(ctx, bytes);
}

/// Release one filter-local completion snapshot's shared charge.
fn release_client_tool_charge(ctx: &mut HttpFilterContext<'_>, bytes: usize) {
    if let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() {
        responses.release_external_payload_bytes(bytes);
    }
}

/// Phase 2a of the chunk commit: accumulate every event into `ResponsesState`
/// and capture lowered client-tool completion artifacts for phase 2b (#1159).
///
/// Split out of [`commit_chunk_events`] so the per-event byte charge and the
/// artifact capture stay under one owner. Fails closed the instant the request-wide
/// accumulation byte total exceeds the cap (#556); the returned completions are the
/// authoritative source the plan pass reads when synthesizing the `custom_tool_call`
/// lifecycle.
#[expect(
    clippy::too_many_lines,
    reason = "accumulation and snapshot admission share one atomic boundary"
)]
fn accumulate_chunk(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    events: &mut [ResponsesEvent],
    parsed_owner_bytes: usize,
) -> Result<Option<ChargedClientToolCompletions>, SseParseError> {
    let mut completions = Vec::new();
    let mut completion_bytes = 0_usize;
    // A chunk may contain hundreds of completed calls and interleaved item
    // events. Keep a conservative shared-state charge instead of serializing
    // the entire response tree again for every snapshot admission.
    let mut admission = CompletionAdmission {
        parsed_owner_bytes,
        shared_upper_bound: None,
    };
    for event in events {
        // Earlier completion snapshots remain live until phase 2b. A native
        // done event also clones its output item without adding a snapshot,
        // so admit each done mutation even while completion_bytes is zero.
        if (completion_bytes > 0 || matches!(event, ResponsesEvent::FunctionCallArgumentsDone(_)))
            && !pending_completion_event_fits(ctx, state, event, &mut admission)
        {
            release_client_tool_completions(ctx, completions, completion_bytes);
            return Ok(None);
        }
        // A done event moves previously buffered arguments into the canonical
        // output item. The filter-local owner shrinks, but the cached shared
        // charge must grow before the next completion snapshot is admitted.
        let transferred_arguments = if admission.shared_upper_bound.is_some() {
            match event {
                ResponsesEvent::FunctionCallArgumentsDone(payload) => tool_call_key(payload)
                    .as_ref()
                    .and_then(|key| state.tool_call_args.get(key))
                    .map_or(Some(0), retained_json_bytes),
                _ => Some(0),
            }
        } else {
            Some(0)
        };
        accumulate_event(ctx, state, event);
        if matches!(
            event,
            ResponsesEvent::ResponseCompleted(_)
                | ResponsesEvent::ResponseIncomplete(_)
                | ResponsesEvent::ResponseFailed(_)
                | ResponsesEvent::OutputItemAdded(_)
                | ResponsesEvent::OutputItemDone(_)
                | ResponsesEvent::FunctionCallArgumentsDone(_)
        ) {
            state.shared_current_output_bytes.store(0, Ordering::Relaxed);
            if let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() {
                responses.mark_current_output_changed();
            }
        }
        if let Some(upper) = admission.shared_upper_bound.as_mut() {
            // The event projection bounds new copies; the transferred string
            // bounds growth of the shared owner that held no such arguments.
            let growth = projected_responses_state_clone_bytes(ctx, std::slice::from_ref(event))
                .and_then(|bytes| bytes.checked_add(transferred_arguments?))
                .and_then(|bytes| bytes.checked_add(64));
            *upper = upper.checked_add(growth.unwrap_or(usize::MAX)).unwrap_or(usize::MAX);
        }
        match capture_client_tool_completion(state, ctx, &mut completions, event, &mut admission) {
            Ok(Some(bytes)) => {
                let Some(total) = completion_bytes.checked_add(bytes) else {
                    release_client_tool_completions(ctx, completions, completion_bytes);
                    release_client_tool_charge(ctx, bytes);
                    return Ok(None);
                };
                completion_bytes = total;
            },
            Ok(None) => {
                release_client_tool_completions(ctx, completions, completion_bytes);
                return Ok(None);
            },
            Err(error) => {
                release_client_tool_completions(ctx, completions, completion_bytes);
                return Err(error);
            },
        }
    }
    Ok(Some((completions, completion_bytes)))
}

/// Preflight an event while earlier completion snapshots are still retained.
/// Repeated done events use their matching item and argument sizes; other
/// events use the ordinary per-event clone projection to extend the running
/// charge for their distinct mutations.
fn pending_completion_event_fits(
    ctx: &HttpFilterContext<'_>,
    state: &StreamEventsState,
    event: &ResponsesEvent,
    admission: &mut CompletionAdmission,
) -> bool {
    let Some(limit) = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(ResponsesState::retained_payload_limit)
    else {
        return true;
    };
    let projection = match event {
        ResponsesEvent::FunctionCallArgumentsDone(payload) => projected_done_mutation_peak_bytes(ctx, state, payload),
        _ => projected_responses_state_clone_bytes(ctx, std::slice::from_ref(event)),
    };
    let Some(staging) = projection.and_then(|bytes| bytes.checked_add(admission.parsed_owner_bytes)) else {
        return false;
    };
    let mut current = admission.shared_upper_bound.unwrap_or(usize::MAX);
    if admission.shared_upper_bound.is_none()
        || !stream_payload_fits_with_budget(state, staging, Some((limit, Some(current))))
    {
        let Some((_, Some(exact))) = shared_retained_budget(ctx, state) else {
            return false;
        };
        current = exact;
        if !stream_payload_fits_with_budget(state, staging, Some((limit, Some(current)))) {
            return false;
        }
    }
    admission.shared_upper_bound = Some(current);
    true
}

/// Bound the new arguments in the response item plus the completed tool-call
/// clone, including the case where a tiny done event consumes a large delta
/// string held by the parser from earlier chunks.
fn projected_done_mutation_peak_bytes(
    ctx: &HttpFilterContext<'_>,
    state: &StreamEventsState,
    payload: &Value,
) -> Option<usize> {
    let item = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|responses| find_output_item(responses.output_items(), payload));
    let Some(item) = item.filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call")) else {
        return Some(0);
    };
    let key = tool_call_key(payload);
    let arguments = payload
        .get("arguments")
        .and_then(Value::as_str)
        .or_else(|| {
            key.as_ref()
                .and_then(|key| state.tool_call_args.get(key).map(String::as_str))
        })
        .unwrap_or_default();
    let arguments_bytes = retained_json_bytes(arguments)?;
    retained_json_bytes(item)?
        .checked_add(arguments_bytes.checked_mul(2)?)?
        .checked_add(key.as_ref().map_or(0, String::len))?
        .checked_add(128)
}

/// Capture one completed lowered call for the restoration plan. The snapshot
/// is charged before cloning and stays charged until phase 2b releases it.
#[expect(
    clippy::too_many_lines,
    reason = "preflight and charge both budgets before copying a lowered completion"
)]
fn capture_client_tool_completion(
    state: &StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    completions: &mut Vec<client_tools::ClientToolCompletion>,
    event: &ResponsesEvent,
    admission: &mut CompletionAdmission,
) -> Result<Option<usize>, SseParseError> {
    let ResponsesEvent::FunctionCallArgumentsDone(payload) = event else {
        return Ok(Some(0));
    };
    let Some(responses) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(Some(0));
    };
    if responses.client_tool_lowering.is_empty() {
        return Ok(Some(0));
    }
    let Some(item) = find_output_item(responses.output_items(), payload) else {
        return Ok(Some(0));
    };
    let is_lowered = item
        .get("name")
        .and_then(Value::as_str)
        .is_some_and(|name| responses.client_tool_lowering.contains_key(name));
    if !is_lowered {
        return Ok(Some(0));
    }
    let Some(key_bytes) = projected_tool_call_key_bytes(payload) else {
        return Ok(None);
    };
    if key_bytes == 0 {
        return Ok(Some(0));
    }
    let Some(snapshot_bytes) = retained_json_bytes(item).and_then(|bytes| bytes.checked_add(key_bytes)) else {
        return Ok(None);
    };
    let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(None);
    };
    responses.stream_accumulated_bytes = responses.stream_accumulated_bytes.checked_add(snapshot_bytes).ok_or(
        SseParseError::AccumulationLimitExceeded {
            dimension: "accumulated_bytes",
            value: usize::MAX,
            limit: state.max_accumulated_bytes,
        },
    )?;
    if let Some(error) = accumulation_bytes_exceeded(state, responses.stream_accumulated_bytes) {
        return Err(error);
    }
    if !admission
        .parsed_owner_bytes
        .checked_add(snapshot_bytes)
        .is_some_and(|bytes| {
            completion_snapshot_fits(ctx, state, bytes, snapshot_bytes, &mut admission.shared_upper_bound)
        })
    {
        return Ok(None);
    }
    let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(None);
    };
    if !responses.retain_external_payload_bytes(snapshot_bytes) {
        return Ok(None);
    }
    let Some(key) = tool_call_key(payload) else {
        release_client_tool_charge(ctx, snapshot_bytes);
        return Ok(None);
    };
    let Some(item) = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|responses| find_output_item(responses.output_items(), payload))
    else {
        release_client_tool_charge(ctx, snapshot_bytes);
        return Ok(None);
    };
    // Necessary clone (AGENTS.md boundary): the restoration plan retains a
    // completed lowered call while subsequent events can mutate shared output.
    // It cannot hold a borrow of `ResponsesState` across that plan (#1159).
    completions.push(client_tools::ClientToolCompletion {
        key,
        item: item.clone(),
    });
    Ok(Some(snapshot_bytes))
}

/// Admit one snapshot against the current chunk's shared-state upper bound.
/// Remeasure exact shared owners only when that conservative bound reaches the
/// ceiling; this preserves near-cap requests without a large JSON scan per
/// co-batched completion.
fn completion_snapshot_fits(
    ctx: &HttpFilterContext<'_>,
    state: &StreamEventsState,
    staging_bytes: usize,
    snapshot_bytes: usize,
    shared_upper_bound: &mut Option<usize>,
) -> bool {
    let Some(limit) = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(ResponsesState::retained_payload_limit)
    else {
        return stream_payload_fits_with_budget(state, staging_bytes, None);
    };
    let mut current = shared_upper_bound.unwrap_or(usize::MAX);
    if shared_upper_bound.is_none()
        || !stream_payload_fits_with_budget(state, staging_bytes, Some((limit, Some(current))))
    {
        let Some((_, Some(exact))) = shared_retained_budget(ctx, state) else {
            return false;
        };
        current = exact;
        if !stream_payload_fits_with_budget(state, staging_bytes, Some((limit, Some(current)))) {
            return false;
        }
    }
    *shared_upper_bound = current.checked_add(snapshot_bytes);
    shared_upper_bound.is_some()
}

/// Phase 2b of the chunk commit: plan lowered client-tool restoration, then append
/// every committed event to the logical stream applying its disposition (#1159).
///
/// The plan pass ([`client_tools::plan_client_tool_restore`]) is the only fallible
/// step: it runs after phase 2a (accumulate) and before any byte is appended, so a
/// malformed lowered lifecycle fails the chunk closed with nothing delivered. The
/// lowering map + echo live on the shared `ResponsesState` (populated by
/// `openai_client_tool_compat` earlier in this pipeline); the per-round lifecycle
/// progress lives on this `StreamEventsState`. Native passthrough (no lowering
/// armed) skips the plan pass entirely — zero overhead, no behavior change for
/// non-lowered traffic. Appending each event is infallible.
fn restore_and_append_chunk(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    events: Vec<ResponsesEvent>,
    completions: &[client_tools::ClientToolCompletion],
) -> Result<Vec<u8>, SseParseError> {
    let dispositions = match ctx.extensions.get::<ResponsesState>() {
        Some(responses) if !responses.client_tool_lowering.is_empty() => {
            let plan = client_tools::plan_client_tool_restore(
                &responses.client_tool_lowering,
                responses.client_tool_echo.as_ref(),
                &state.client_tool_items,
                &events,
                completions,
            )?;
            state.client_tool_items = plan.next_items;
            Some(plan.dispositions)
        },
        _ => None,
    };

    let mut logical_output = Vec::new();
    for (index, event) in events.into_iter().enumerate() {
        match dispositions.as_ref().and_then(|dispositions| dispositions.get(index)) {
            None | Some(client_tools::ClientToolDisposition::Passthrough) => {
                append_logical_event(state, ctx, event, &mut logical_output);
            },
            Some(client_tools::ClientToolDisposition::Suppress) => {},
            Some(disposition) => {
                apply_client_tool_disposition(state, ctx, disposition, event, &mut logical_output);
            },
        }
    }
    Ok(logical_output)
}

/// Whether an event is a response lifecycle-creation event
/// (`response.created`/`queued`/`in_progress`).
///
/// These open the logical response and must be emitted exactly once, ahead of any
/// output item: at `iteration > 0` a resumed round suppresses them (the first round
/// already sent them), and at `iteration 0` they precede a locally executed item's
/// synthesized flush.
fn is_response_lifecycle_creation(event: &ResponsesEvent) -> bool {
    matches!(
        event,
        ResponsesEvent::ResponseCreated(_) | ResponsesEvent::ResponseQueued(_) | ResponsesEvent::ResponseInProgress(_)
    )
}

/// Append one provider event to the logical stream or defer/suppress it.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: file_search suppression + deferred-delta gate + seven event type arms, each with its own payload normalization"
)]
fn append_logical_event(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    event: ResponsesEvent,
    output: &mut Vec<u8>,
) {
    // #313 §4/§6: classify locally-executable file_search items at first sight and
    // suppress their raw wire representation. Only runs on the logical stream with a
    // hosted file_search tool declared (the load-bearing configured-tool gate, P1 round-11).
    let file_search_active = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(crate::openai::responses::file_search_callout::has_file_search_tool);
    if file_search_active && event.event_type() == "response.output_item.added" {
        let payload = event.payload();
        if let Some(item) = payload.get("item") {
            use crate::openai::responses::file_search_callout::{
                is_file_search_function_call, is_pending_file_search_call,
            };
            if is_file_search_function_call(item) {
                local_tools::register_local_tool(
                    &mut state.local_tool_items,
                    payload,
                    local_tools::LocalToolMode::Suppress,
                );
            } else if is_pending_file_search_call(item) {
                local_tools::register_local_tool(
                    &mut state.local_tool_items,
                    payload,
                    local_tools::LocalToolMode::NativeHybridPending,
                );
            }
        }
    }
    // Mode-aware suppression: Suppress drops all events; NativeHybridPending drops only
    // a still-PENDING output_item.done (EOS synthesizes the completed tail), but passes
    // through terminal done (completed/failed/incomplete) and removes keys (cancels synthesis).
    if !state.local_tool_items.is_empty() {
        let keys: Vec<String> = local_tools::event_local_tool_keys(event.payload()).collect();
        if let Some(mode) = keys.iter().find_map(|k| state.local_tool_items.get(k).copied()) {
            match mode {
                local_tools::LocalToolMode::Suppress => return,
                local_tools::LocalToolMode::NativeHybridPending => {
                    if event.event_type() == "response.output_item.done" {
                        let status = event
                            .payload()
                            .get("item")
                            .and_then(|item| item.get("status"))
                            .and_then(Value::as_str);
                        if matches!(status, Some("searching" | "in_progress")) {
                            return; // still-pending done: EOS synthesizes the completed tail.
                        }
                        // Terminal done (completed/failed/incomplete): pass through and
                        // cancel EOS synthesis — the provider resolved the call. Record the
                        // item id so the file_search EOS reconcile skips re-queuing this
                        // call; a synthesized tail would duplicate this live done (#313 P1).
                        // Recorded for every terminal status, not just `completed`.
                        if let Some(id) = event
                            .payload()
                            .get("item")
                            .and_then(|item| item.get("id"))
                            .and_then(Value::as_str)
                        {
                            ctx.extensions
                                .get_or_insert_with(ResponsesState::default)
                                .provider_streamed_terminal_ids
                                .insert(id.to_owned());
                        }
                        for k in &keys {
                            state.local_tool_items.remove(k);
                        }
                    }
                    // Opening + progress fall through and pass normally.
                },
            }
        }
    }

    if event.is_terminal() {
        let event_type = event.event_type().to_owned();
        state.deferred_terminal = Some(DeferredTerminalEvent {
            event_type,
            payload: event.into_payload(),
        });
        return;
    }
    if state.iteration > 0 && is_response_lifecycle_creation(&event) {
        return;
    }

    // #276: reconcile locally executed tool items with the resumed model stream
    // (record streamed milestones, flush pending local items ahead of the first
    // resumed event, suppress a premature local-tool `done`). Returns true when
    // this event must not be forwarded.
    if commit_local_tool_milestones(state, ctx, &event, output) {
        return;
    }

    let event_type = event.event_type().to_owned();
    let mut payload = event.into_payload();
    normalize_logical_payload(ctx, &mut payload, state.output_index_offset);
    encode_sse_event(&event_type, &payload, output);
}

/// Apply a non-`Passthrough`/`Suppress` client-tool restoration disposition to a
/// committed event, then append it to the logical stream (#1159).
///
/// Infallible by contract: the fallible planning already ran in
/// [`commit_chunk_events`], so this only mutates or replaces the event payload and
/// forwards it. `RetypeInPlace` retypes a lowered `Namespace` member's
/// `function_call` item back to its original name + namespace in place (Task 4).
/// `RetypeArgumentsName` overwrites the top-level `name` on a `Namespace`
/// `function_call_arguments.done` (r2c populates it per the Responses schema, unlike
/// native backends) with the member name, in place (#1206). `EmitCustomShell`/`EmitCustomItemDone` splice the
/// fully-typed public `custom_tool_call` payload the plan pass already built onto the corresponding
/// lifecycle event (Task 5). `EmitCustomInput`
/// synthesizes the canonical `custom_tool_call_input` delta+done pair and drops the
/// backend's `function_call_arguments.done` it replaces. `EmitTypedAdded`/`EmitTypedDone`
/// construct a fresh `output_item.added`/`output_item.done` carrying the restored
/// `shell_call`/`tool_search_call` (Task 6) — `EmitTypedAdded` fires on the incoming
/// `function_call_arguments.done` and must emit under an `output_item.added` line, so it
/// cannot splice onto the incoming event; the dropped args.done is replaced by the
/// synthesized added. None of these ever leak the private `agentic_ns__{ns}__{member}` /
/// lowered `fc_` id. `RestoreSnapshot` (which the plan pass produces for the
/// non-terminal `response.created`/`queued`/`in_progress` snapshots) splices the
/// already-restored response object back onto the lifecycle event's payload before
/// forwarding it, so a lowered name never appears in an intermediate snapshot;
/// `Suppress` is dropped before dispatch and must never reach the applier.
#[expect(
    clippy::too_many_lines,
    reason = "linear match dispatch over the nine client-tool restore arms, each with a load-bearing comment"
)]
fn apply_client_tool_disposition(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    disposition: &client_tools::ClientToolDisposition,
    mut event: ResponsesEvent,
    logical_output: &mut Vec<u8>,
) {
    use client_tools::ClientToolDisposition as D;
    match disposition {
        D::RetypeInPlace {
            item_type,
            name,
            namespace,
        } => {
            retype_item_in_place(event.payload_mut(), item_type, name, namespace.as_deref());
            append_logical_event(state, ctx, event, logical_output);
        },
        // Namespace `function_call_arguments.done`: r2c populated the top-level `name`
        // with the private lowered name; overwrite it with the member name in place so
        // the `agentic_ns__{ns}__{member}` name never reaches the client. Only produced
        // for frames that already carry a `name`, so this overwrites, never injects.
        D::RetypeArgumentsName { name } => {
            if let Some(object) = event.payload_mut().as_object_mut() {
                object.insert("name".to_owned(), Value::String(name.clone()));
            }
            append_logical_event(state, ctx, event, logical_output);
        },
        // Custom/NamespaceCustom synthesized custom_tool_call output-item events.
        // Each fires on the matching incoming lifecycle event (added / done), so the
        // carried payload already has the right shape; splice it on and append.
        D::EmitCustomShell { item } | D::EmitCustomItemDone { item } => {
            *event.payload_mut() = item.clone();
            append_logical_event(state, ctx, event, logical_output);
        },
        D::EmitCustomInput {
            item_id,
            output_index,
            input,
            ..
        } => {
            // Synthesize the canonical custom_tool_call_input delta+done pair (public
            // `ctc_` id); the backend `function_call_arguments.done` it replaces is
            // dropped by not appending `event` (#1159).
            synthesize_custom_tool_input(state, ctx, (item_id, *output_index, input), logical_output);
        },
        // Shell/ToolSearch synthesized typed `output_item.added`. Fires on the incoming
        // function_call_arguments.done, so CONSTRUCT a fresh OutputItemAdded — the SSE event
        // type derives from the variant; splicing onto the args.done event would emit the
        // added body under a function_call_arguments.done line (#1159 R-T6a).
        D::EmitTypedAdded { item } => {
            append_logical_event(
                state,
                ctx,
                ResponsesEvent::OutputItemAdded(item.clone()),
                logical_output,
            );
        },
        // Shell/ToolSearch restored typed `output_item.done` (incoming is already
        // output_item.done; construct fresh for symmetry with the added path).
        D::EmitTypedDone { item } => {
            append_logical_event(state, ctx, ResponsesEvent::OutputItemDone(item.clone()), logical_output);
        },
        // Non-terminal snapshot restoration: splice the restored response object back
        // onto the lifecycle event's payload (response.created/queued/in_progress).
        // Tools/tool_choice and output items are already restored in the disposition.
        D::RestoreSnapshot { response } => {
            if let Some(object) = event.payload_mut().as_object_mut() {
                object.insert("response".to_owned(), response.clone());
            }
            append_logical_event(state, ctx, event, logical_output);
        },
        // Passthrough is forwarded by restore_and_append_chunk before dispatch;
        // reaching it here would still mean "forward", so append.
        D::Passthrough => append_logical_event(state, ctx, event, logical_output),
        // Suppress is dropped by restore_and_append_chunk before dispatch and must
        // NEVER forward; reaching the applier is a bug (debug panic; release drops).
        D::Suppress => {
            debug_assert!(
                false,
                "Suppress is dropped in restore_and_append_chunk, never dispatched to the applier"
            );
        },
    }
}

/// Retype a lowered `Namespace` member's `function_call` item in place: set
/// `type`/`name` and re-add or remove `namespace`, never leaking the private
/// `agentic_ns__{ns}__{member}` name (#1159).
fn retype_item_in_place(payload: &mut Value, item_type: &str, name: &str, namespace: Option<&str>) {
    let Some(item) = payload.get_mut("item").and_then(Value::as_object_mut) else {
        return;
    };
    item.insert("type".to_owned(), Value::String(item_type.to_owned()));
    item.insert("name".to_owned(), Value::String(name.to_owned()));
    match namespace {
        Some(namespace) => {
            item.insert("namespace".to_owned(), Value::String(namespace.to_owned()));
        },
        None => {
            item.remove("namespace");
        },
    }
}

/// Synthesize the canonical `custom_tool_call_input` delta+done pair for a restored
/// `Custom`/`NamespaceCustom` call (#1159).
///
/// `fields` is `(item_id, output_index, input)`. Both frames reference the PUBLIC
/// `ctc_` item id (never the private `fc_` id) and carry the unwrapped plain-string
/// input. The input is emitted on two distinct SSE frames, so it is necessarily
/// copied once per frame at this boundary.
fn synthesize_custom_tool_input(
    state: &StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    fields: (&str, u64, &str),
    output: &mut Vec<u8>,
) {
    let (item_id, output_index, input) = fields;
    let delta = serde_json::json!({
        "type": "response.custom_tool_call_input.delta",
        "item_id": item_id,
        "output_index": output_index,
        "delta": input,
    });
    emit_synthetic_event(state, ctx, "response.custom_tool_call_input.delta", delta, output);
    let done = serde_json::json!({
        "type": "response.custom_tool_call_input.done",
        "item_id": item_id,
        "output_index": output_index,
        "input": input,
    });
    emit_synthetic_event(state, ctx, "response.custom_tool_call_input.done", done, output);
}

/// Emit a synthesized logical SSE event with no originating provider frame (#1159).
///
/// Mirrors [`append_logical_event`]'s tail: normalizes the payload (stamping the
/// running `output_index` offset and logical sequence number) then encodes it. Used
/// by the `custom_tool_call_input` synthesis, whose delta+done frames are generated
/// wholesale rather than derived from a committed [`ResponsesEvent`].
fn emit_synthetic_event(
    state: &StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    event_type: &str,
    mut payload: Value,
    output: &mut Vec<u8>,
) {
    normalize_logical_payload(ctx, &mut payload, state.output_index_offset);
    encode_sse_event(event_type, &payload, output);
}

/// Reconcile locally executed tool items against the resumed model stream for one
/// forwarded event, returning `true` when that event must be suppressed.
fn commit_local_tool_milestones(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    event: &ResponsesEvent,
    output: &mut Vec<u8>,
) -> bool {
    // Record which client-visible milestones the model backend streamed for this
    // item. `output_item.added`/`.done` mark it announced (so a later flush does
    // not re-emit `output_item.added`); an actual `response.web_search_call.*` /
    // `response.mcp_call.*` / `response.mcp_list_tools.*` progress event marks
    // the lifecycle as already streamed in-band (so the flush does not
    // re-synthesize it). Persisted across rounds via `emitted_output_items`,
    // this is what a resumed round's flush consults.
    record_model_output_item(ctx, event);

    // #276: ahead of the first model output *content* event, stream any locally
    // generated tool items (MCP calls/approvals/listings, or web searches
    // absent from the upstream stream) that the tool-dispatch filters appended
    // to `accumulated_output` but never emitted incrementally. They must
    // precede the resumed model output and occupy their reserved output indices.
    // `accumulated_output` is fixed for the round, so the flush runs once here
    // rather than re-serializing every local item ahead of each event; the EOS
    // flush still catches items whose round produced no resumed model event.
    //
    // The flush is deferred past `response.created`/`queued`/`in_progress` rather
    // than gated on `iteration > 0`: an MCP approval resume (#1029) executes the
    // approved tool during `on_request_body`, before any inference round, so its
    // local `mcp_call` sits at `accumulated_output[0]` while `iteration` is still 0
    // and the round still forwards its own lifecycle-creation events. Gating on the
    // round number left that index-0 item unannounced ahead of the model output
    // shifted to index 1, tripping client stream accumulators. At `iteration > 0`
    // the creation events never reach here (suppressed above), so the first event
    // seen is already content and the behavior is unchanged. When no local item is
    // pending — the common first round — `flush_local_output_items` is a no-op.
    if !state.local_items_flushed && !is_response_lifecycle_creation(event) {
        flush_local_output_items(ctx, output);
        state.local_items_flushed = true;
    }

    // #276 (finding): the model may finalize a local tool item with
    // `output_item.done` in the very round that declares it, before the dispatch
    // filter has executed the tool. Passing that `done` through here is premature:
    // the tool-specific progress lifecycle and real outcome are still unknown, and
    // the resumed round would then synthesize the progress events plus a second
    // `done` — leaving the client with `added -> done -> in_progress -> ... ->
    // done`. Suppress that premature `done`; the flush that follows local
    // execution emits the single ordered `done` after the progress events. An item
    // whose lifecycle the model *did* stream in-band keeps its `done` (it is real).
    if is_premature_local_tool_done(ctx, event) {
        return true;
    }

    // A local-tool `output_item.done` that survives the premature check finalizes
    // the item for the client, so record the envelope as delivered. Tracked apart
    // from `added`/content: a resumed flush must then synthesize neither a
    // duplicate `done` nor (when unchanged) drop the finalizer the client already
    // received.
    mark_local_done_delivered(ctx, event);
    false
}

/// Record that a model-streamed `output_item.done` envelope reached the client for
/// a locally executed tool item, so a resumed round finalizes each local item with
/// exactly one `done` — neither dropping it nor duplicating it.
fn mark_local_done_delivered(ctx: &mut HttpFilterContext<'_>, event: &ResponsesEvent) {
    let ResponsesEvent::OutputItemDone(payload) = event else {
        return;
    };
    if let Some(item) = payload.get("item").filter(|item| is_local_tool_item(item))
        && let Some(id) = item.get("id").and_then(Value::as_str)
    {
        update_emitted_item(ctx, id, |emitted| emitted.done_delivered = true);
    }
}

/// Record which client-visible milestones the model backend streamed for a
/// local-tool output item, so [`flush_local_output_items`] neither duplicates
/// them nor drops the progress lifecycle the model never sends.
///
/// `output_item.added`/`output_item.done` mark the item *announced* and record
/// its latest content, but prove nothing about the tool-specific progress
/// lifecycle: a backend may stream `added` then `done` with no progress events in
/// between, or only some of them (e.g. `in_progress` then `done`). Only observing
/// an actual `response.web_search_call.*` / `response.mcp_call.*` /
/// `response.mcp_list_tools.*` event proves that specific phase reached the
/// client, so each is recorded individually by its event type. Deriving the
/// lifecycle from `done` would suppress the synthesized progress a partial
/// `added → in_progress → done` sequence still owes for its missing
/// `searching`/`completed` phases.
fn record_model_output_item(ctx: &mut HttpFilterContext<'_>, event: &ResponsesEvent) {
    match event {
        ResponsesEvent::OutputItemAdded(payload) | ResponsesEvent::OutputItemDone(payload) => {
            if let Some(item) = payload.get("item").filter(|item| is_local_tool_item(item))
                && let Some(id) = item.get("id").and_then(Value::as_str)
            {
                let digest = item_digest(item);
                update_emitted_item(ctx, id, |emitted| {
                    emitted.added = true;
                    emitted.content_digest = digest;
                });
            }
        },
        ResponsesEvent::Unknown { event_type, data } if is_local_tool_progress_event(event_type) => {
            if let Some(id) = data.get("item_id").and_then(Value::as_str) {
                let phase = event_type.clone();
                update_emitted_item(ctx, id, move |emitted| {
                    emitted.added = true;
                    emitted.streamed_phases.insert(phase);
                });
            }
        },
        _ => {},
    }
}

/// Whether this event is a model-streamed `output_item.done` for a locally
/// executed tool item whose *terminal* lifecycle phase has not streamed in-band.
///
/// Such a `done` is premature: the dispatch filter runs the tool *after* this
/// round, so the item's real progress and outcome are unknown here. Emitting it
/// now would leave the resumed round's flush to add the missing progress events
/// plus a second `done`, so the caller suppresses it and lets the flush emit the
/// single ordered `done`.
///
/// Prematurity keys on the *terminal* expected phase, not on every phase: once
/// the last phase of the lifecycle has streamed in-band the item is
/// authoritatively finished and its `done` passes through, even if the backend
/// skipped an optional earlier phase (a `web_search_call` may stream `in_progress`
/// then `completed` without `searching`). Suppressing that `done` would drop the
/// backend's real terminal event and force the flush to back-fill the skipped
/// phase *after* the outcome, out of canonical order.
fn is_premature_local_tool_done(ctx: &HttpFilterContext<'_>, event: &ResponsesEvent) -> bool {
    let ResponsesEvent::OutputItemDone(payload) = event else {
        return false;
    };
    let Some(item) = payload.get("item").filter(|item| is_local_tool_item(item)) else {
        return false;
    };
    let Some(id) = item.get("id").and_then(Value::as_str) else {
        return false;
    };
    let expected = expected_phase_events(item);
    let streamed = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|state| state.emitted_output_items.get(id))
        .map(|emitted| &emitted.streamed_phases);
    match expected.last() {
        // No tool-specific lifecycle (e.g. `mcp_approval_request`): the `done` is
        // never premature.
        None => false,
        Some(&terminal_phase) => !streamed.is_some_and(|phases| phases.contains(terminal_phase)),
    }
}

/// Whether an event type is a tool-specific progress or outcome event the model
/// backend streams in-band for a hosted `web_search_call`, `mcp_call`, or
/// `mcp_list_tools` (`response.web_search_call.*` / `response.mcp_call.*` /
/// `response.mcp_list_tools.*`). Observing one proves the progress lifecycle
/// reached the client, so the proxy must not synthesize it again.
///
/// `mcp_list_tools` is included because a *deferred* MCP entry (`defer_loading:
/// true`, or one lacking a `server_url`) is passed through unresolved by
/// `openai_mcp_tool_resolve`, so the backend performs `tools/list` itself and
/// natively streams the discovery lifecycle. Recording those phases keeps a
/// backend-executed listing's real `output_item.done` from being suppressed as
/// premature (issue #1022), exactly as native `web_search_call`/`mcp_call`
/// passthrough is already handled. A locally seeded listing streams no such
/// events, so this predicate is inert for it and its lifecycle is synthesized by
/// [`flush_local_output_items`] as before.
fn is_local_tool_progress_event(event_type: &str) -> bool {
    event_type.starts_with("response.web_search_call.")
        || event_type.starts_with("response.mcp_call.")
        || event_type.starts_with("response.mcp_list_tools.")
}

/// Whether an output item is one of the tool types whose streaming lifecycle the
/// proxy reconciles — whether locally synthesized (seeded by a tool-dispatch
/// filter) or streamed natively by the model backend.
///
/// `mcp_list_tools` is such a type. On eager resolution `openai_mcp_tool_resolve`
/// runs the MCP `tools/list` and seeds the discovery listing into
/// `accumulated_output` before any inference round (issue #1022), so the backend —
/// which then only sees the rewritten `type: "function"` tools — never streams it.
/// But a *deferred* entry (`defer_loading: true`, or one lacking a `server_url`) is
/// passed through unresolved, so the backend performs `tools/list` itself and
/// natively streams the listing. Both are recognized here; whether a given
/// lifecycle event is synthesized or forwarded is then decided by the phases
/// actually streamed in-band ([`is_local_tool_progress_event`] →
/// `streamed_phases`) and by provenance (`locally_executed_output_items`, which
/// gates [`collect_pending_local_items`]), not by this type check alone.
fn is_local_tool_item(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("mcp_call" | "mcp_approval_request" | "web_search_call" | "mcp_list_tools")
    )
}

/// Fixed-size content digest of an output item, compared across rounds to detect
/// when a previously streamed item changed and must re-emit its `output_item.done`
/// envelope.
///
/// The whole item is hashed rather than keyed on `type|status` alone so a payload
/// the model never streamed — e.g. the `action.sources` list `openai_web_search`
/// adds to a `web_search_call` after local execution — is detected as a change even
/// when the item's type and status are unchanged.
///
/// A `u64` digest rather than a retained serialized string: local tool payloads
/// already live in `accumulated_output`, and an IRR response can reach tens of MiB
/// across rounds, so keeping a second full copy per item in `EmittedItem` would be
/// payload-scale memory amplification. The digest is walked *canonically* (object
/// keys hashed in sorted order) so it depends only on content: the two sides of a
/// change comparison come from different backend serializations — an
/// `output_item.done` payload recorded in one round versus the `accumulated_output`
/// snapshot rebuilt in the next — whose object key order is not guaranteed stable
/// (`preserve_order` makes `serde_json` retain insertion order), and a mere key
/// reorder must not masquerade as a content change and trigger a spurious duplicate
/// `done`. The walk allocates no intermediate `Value`/`String`.
///
/// [`DefaultHasher`]'s algorithm is explicitly not guaranteed stable across Rust
/// releases, but that is irrelevant here: a digest is only ever compared against
/// another digest produced by the *same running binary* within one request (it is
/// never persisted, sent on the wire, or compared across processes or releases), so
/// only its determinism within a single process — which the fixed-key seed
/// guarantees — is load-bearing.
fn item_digest(item: &Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_value_canonical(item, &mut hasher);
    hasher.finish()
}

/// Feed a JSON value into `hasher` canonically: each value is type-tagged and
/// containers are length-prefixed so different shapes cannot collide, and object
/// entries are hashed in sorted key order so key order does not affect the digest.
///
/// Sorting keys on the fly (over borrowed `&str`) avoids materializing a
/// recursively key-sorted copy of the item, so no second full `Value` is allocated
/// (AGENTS.md ownership rule).
fn hash_value_canonical(value: &Value, hasher: &mut DefaultHasher) {
    // Each arm leads with a distinct type tag so different shapes cannot collide
    // (`0` vs `"0"` vs `false`); scalars fold tag and payload into one tuple hash.
    // `Number` has no stable `Hash`, so its primitive representation is hashed via
    // `hash_number_canonical` — covering integer and float without allocating.
    match value {
        Value::Null => 0_u8.hash(hasher),
        Value::Bool(boolean) => (1_u8, boolean).hash(hasher),
        Value::Number(number) => {
            2_u8.hash(hasher);
            hash_number_canonical(number, hasher);
        },
        Value::String(string) => (3_u8, string).hash(hasher),
        Value::Array(items) => {
            (4_u8, items.len()).hash(hasher);
            for item in items {
                hash_value_canonical(item, hasher);
            }
        },
        Value::Object(map) => {
            (5_u8, map.len()).hash(hasher);
            let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
            keys.sort_unstable();
            for key in keys {
                key.hash(hasher);
                // Present because the key came from the map's own key set.
                hash_value_canonical(&map[key], hasher);
            }
        },
    }
}

/// Feed a JSON number into `hasher` from its primitive representation, allocating
/// nothing.
///
/// `serde_json` is built here without `arbitrary_precision`, so every `Number` is
/// stored as exactly one of `u64`/`i64`/`f64` and hashing that primitive is exact.
/// Integers fold into a common `i128` space under one sub-tag, so the same integer
/// hashes identically whether serde stored it as `u64` or `i64`; floats hash their
/// bit pattern under a distinct sub-tag, so integer `5` and float `5.0` stay
/// distinct values — matching the earlier string-form (`"5"` vs `"5.0"`) behavior
/// without its per-value allocation. `as_f64` always succeeds for a representable
/// number, so the final branch is total.
fn hash_number_canonical(number: &serde_json::Number, hasher: &mut DefaultHasher) {
    if let Some(unsigned) = number.as_u64() {
        (0_u8, i128::from(unsigned)).hash(hasher);
    } else if let Some(signed) = number.as_i64() {
        (0_u8, i128::from(signed)).hash(hasher);
    } else if let Some(float) = number.as_f64() {
        (1_u8, float.to_bits()).hash(hasher);
    }
}

/// Merge an update into the tracked client-visible milestones for `id`, creating
/// the entry when the item has not been seen before.
///
/// Milestones accrue independently across events and rounds (an `added` here, a
/// `streamed_phases` entry there), so callers mutate only the fields they observe
/// rather than overwriting the whole record and clobbering an earlier milestone.
fn update_emitted_item(ctx: &mut HttpFilterContext<'_>, id: &str, update: impl FnOnce(&mut EmittedItem)) {
    let items = &mut ctx
        .extensions
        .get_or_insert_with(ResponsesState::default)
        .emitted_output_items;
    // The common path across rounds updates an item that already exists; look it up
    // by borrowed `&str` first so only a genuine first insert allocates an owned key,
    // rather than allocating one on every `entry()` probe.
    if let Some(emitted) = items.get_mut(id) {
        update(emitted);
        return;
    }
    update(items.entry(id.to_owned()).or_default());
}

/// Emit incremental events for locally generated tool items in
/// `accumulated_output` that have not yet reached the client or whose outcome
/// changed since they were last streamed.
///
/// Each synthesized item reuses its absolute index in `accumulated_output`, so
/// the incremental events agree with the final `response.completed` snapshot.
/// The tracked milestones make this idempotent across rounds, skip the
/// `output_item.added` and progress events the model backend already streamed,
/// and trigger outcome-only re-emission when a previously seen item changed.
fn flush_local_output_items(ctx: &mut HttpFilterContext<'_>, output: &mut Vec<u8>) {
    let pending = match ctx.extensions.get::<ResponsesState>() {
        Some(state) => collect_pending_local_items(state),
        None => return,
    };
    for pending in pending {
        let PendingItem {
            index,
            item,
            digest,
            plan,
        } = pending;
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            let id = id.to_owned();
            // This pass emits the `done` envelope iff the plan says so, so record
            // the finalizer only when it is actually delivered.
            let done_delivered = plan.emit_done;
            update_emitted_item(ctx, &id, |emitted| {
                emitted.added = true;
                // Record exactly the phases this pass delivers; the ones the model
                // already streamed in-band are tracked as they arrived, and a phase
                // the backend deliberately skipped is left unrecorded so the frontier
                // rule never back-fills it later. Borrow `plan.phases` rather than
                // cloning it: the closure runs to completion inside
                // `update_emitted_item` before `plan` is moved into
                // `synthesize_local_item` below.
                for event_type in &plan.phases {
                    emitted.streamed_phases.insert((*event_type).to_owned());
                }
                emitted.done_delivered = emitted.done_delivered || done_delivered;
                emitted.content_digest = digest;
            });
        }
        synthesize_local_item(ctx, output, index, item, plan);
    }
}

/// Collect the locally generated tool items that must be (re)synthesized: those
/// whose progress lifecycle has not yet been streamed, or whose content changed
/// since the client last saw them.
///
/// Synthesis is gated on execution provenance, not item type. A dispatch filter
/// records the ids it actually executed in
/// [`ResponsesState::locally_executed_output_items`]; a tool-typed item that only
/// reached `accumulated_output` because a failed (non-dispatchable) round copied
/// the model's placeholder there — e.g. `agentic_loop::collect_streaming_output_items`
/// after a parse error — has no provenance entry and is skipped, so the terminal
/// error flush never fabricates a lifecycle for a search that never ran.
///
/// One clone per pending item is required to escape the immutable borrow of
/// `accumulated_output` before the mutable-borrowing synthesis calls in
/// [`flush_local_output_items`]; the owned item is then moved through the
/// synthesized events without any further clone (AGENTS.md ownership rule).
///
/// [`item_digest`] re-walks each pending item's content (including its on-the-fly
/// object-key sort) here rather than reading a cached value. The cost is bounded:
/// this runs at most once per round in-band plus once at end-of-stream, over only
/// the handful of local tools a round actually executes — so caching the digest
/// (and the invalidation state a cache would need) buys nothing over recomputing it
/// against the small, round-stable item set.
fn collect_pending_local_items(state: &ResponsesState) -> Vec<PendingItem> {
    state
        .accumulated_output
        .iter()
        .enumerate()
        .filter(|(_, item)| is_local_tool_item(item))
        .filter_map(|(index, item)| {
            let id = item.get("id").and_then(Value::as_str)?;
            if !state.locally_executed_output_items.contains(id) {
                return None;
            }
            let digest = item_digest(item);
            let previous = state.emitted_output_items.get(id);
            let plan = plan_pending_item(previous, item, digest)?;
            Some(PendingItem {
                index,
                item: item.clone(),
                digest,
                plan,
            })
        })
        .collect()
}

/// Decide whether a locally generated item still owes the client any events and,
/// if so, exactly which lifecycle milestones this synthesis pass must emit.
///
/// Returns `None` when the item is fully delivered and unchanged — the client
/// already saw `output_item.added`, every phase the lifecycle owes, the finalizing
/// `output_item.done`, and this exact content. Otherwise the plan emits
/// `output_item.added` only if the item was never announced, the progress events
/// that come *after* the frontier of phases already streamed in-band, and the
/// `output_item.done` envelope when it has not yet been delivered or the item's
/// content changed since it last was.
///
/// The three milestones are tracked independently. A backend can stream every
/// phase in-band yet be cut off before the `done` envelope, so `done_delivered` —
/// not the phase set or the content digest — governs finalization: without it a
/// terminal-phase-in-band item with unchanged content would never be finalized.
/// On a genuine content change only the `done` envelope is re-emitted (it carries
/// the refreshed item, e.g. a `web_search_call` that gained `action.sources`); the
/// terminal *phase* event is not, because it carries no item data and re-emitting
/// it would be a pure duplicate.
///
/// The frontier rule is what keeps synthesis in canonical order. Synthesized
/// events are appended *after* whatever the backend already streamed in-band, so a
/// phase ordinally earlier than one already streamed can never be emitted without
/// landing out of order (e.g. `searching` after an already-streamed `completed`).
/// A backend that streams `in_progress` then `completed` (skipping the optional
/// `searching`) therefore has its skip honored rather than back-filled, and the
/// real terminal event it already sent is not duplicated.
fn plan_pending_item(previous: Option<&EmittedItem>, item: &Value, digest: u64) -> Option<EmissionPlan> {
    let expected = expected_phase_events(item);
    let content_changed = previous.is_none_or(|p| p.content_digest != digest);
    let streamed = previous.map(|p| &p.streamed_phases);
    let frontier = max_streamed_ordinal(&expected, streamed);
    let phases: Vec<&'static str> = expected
        .iter()
        .enumerate()
        .filter(|&(ordinal, _)| frontier.is_none_or(|frontier| ordinal > frontier))
        .map(|(_, &event)| event)
        .collect();
    let already_announced = previous.is_some_and(|p| p.added);
    let done_delivered = previous.is_some_and(|p| p.done_delivered);
    let emit_done = !done_delivered || content_changed;
    if already_announced && phases.is_empty() && !emit_done {
        return None;
    }
    Some(EmissionPlan {
        emit_added: !already_announced,
        phases,
        emit_done,
    })
}

/// The highest ordinal position within `expected` of a phase already streamed to
/// the client, or `None` when none have streamed.
///
/// This is the frontier past which [`plan_pending_item`] may synthesize leading
/// progress events. A phase at or before the frontier was either already streamed
/// or deliberately skipped by the backend; either way re-emitting it now would
/// place it out of canonical order behind a later phase already sent.
fn max_streamed_ordinal(expected: &[&'static str], streamed: Option<&BTreeSet<String>>) -> Option<usize> {
    let streamed = streamed?;
    expected
        .iter()
        .enumerate()
        .filter_map(|(ordinal, phase)| streamed.contains(*phase).then_some(ordinal))
        .max()
}

/// A locally generated output item awaiting synthesis, carried out of the
/// immutable `accumulated_output` borrow as a single owned clone.
struct PendingItem {
    /// Absolute output index in `accumulated_output`.
    index: usize,
    /// The owned output item, moved through the synthesized events.
    item: Value,
    /// The item's content digest, recorded before synthesis.
    digest: u64,
    /// Which lifecycle milestones this synthesis pass must emit.
    plan: EmissionPlan,
}

/// Which parts of a local item's lifecycle a single synthesis pass must emit.
struct EmissionPlan {
    /// Whether to emit `output_item.added` (item never announced yet) versus
    /// reusing the announcement the model or a prior synthesis already streamed.
    emit_added: bool,
    /// The exact tool-specific progress/outcome events this pass must synthesize,
    /// in order — the expected phases that fall after the frontier of phases
    /// already streamed in-band.
    phases: Vec<&'static str>,
    /// Whether to emit the finalizing `output_item.done` envelope: the item has no
    /// delivered `done` yet, or its content changed since the last one and the
    /// refreshed item (e.g. now carrying `action.sources`) must reach the client.
    /// Only the envelope is re-emitted on a content change — the payloadless
    /// terminal *phase* event carries no item data, so re-emitting it would be a
    /// pure duplicate.
    emit_done: bool,
}

/// Synthesize the incremental event sequence for one locally generated item.
///
/// Emits `output_item.added` (only when `plan.emit_added`), then exactly the
/// tool-specific events in `plan.phases` (the progress/outcome events still owed
/// after accounting for anything the model streamed in-band), then the finalizing
/// `output_item.done` (only when `plan.emit_done`). A partial in-band lifecycle
/// therefore gets only its missing phases; a content-change re-emission gets just
/// the refreshed `done` envelope. The owned item is moved into `added`, reclaimed
/// via `take`, then moved into `done`, so the full item is never cloned here.
fn synthesize_local_item(
    ctx: &mut HttpFilterContext<'_>,
    output: &mut Vec<u8>,
    index: usize,
    mut item: Value,
    plan: EmissionPlan,
) {
    let output_index = u64::try_from(index).unwrap_or(u64::MAX);
    let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();

    if plan.emit_added {
        let mut added = item_lifecycle_payload("response.output_item.added", output_index, item);
        normalize_logical_payload(ctx, &mut added, 0);
        encode_sse_event("response.output_item.added", &added, output);
        // Reclaim ownership of the item (leaving `null` behind) so the `done`
        // event below reuses it without a second clone.
        item = added.get_mut("item").map(Value::take).unwrap_or_default();
    }

    for event_type in plan.phases {
        let mut payload = serde_json::json!({
            "type": event_type,
            "item_id": item_id,
            "output_index": output_index,
            "sequence_number": 0,
        });
        normalize_logical_payload(ctx, &mut payload, 0);
        encode_sse_event(event_type, &payload, output);
    }

    if plan.emit_done {
        let mut done = item_lifecycle_payload("response.output_item.done", output_index, item);
        normalize_logical_payload(ctx, &mut done, 0);
        encode_sse_event("response.output_item.done", &done, output);
    }
}

/// Build an `output_item.added`/`output_item.done` payload, moving the item in.
///
/// The object is assembled with `Map::insert` rather than `json!` so the item is
/// moved rather than deep-cloned through serialization (AGENTS.md ownership
/// rule). `output_index` is already absolute, so callers normalize with a zero
/// offset; normalization only rewrites the logical response id and sequence.
fn item_lifecycle_payload(event_type: &str, output_index: u64, item: Value) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("type".to_owned(), Value::String(event_type.to_owned()));
    object.insert("response_id".to_owned(), Value::Null);
    object.insert("output_index".to_owned(), Value::from(output_index));
    object.insert("item".to_owned(), item);
    object.insert("sequence_number".to_owned(), Value::from(0));
    Value::Object(object)
}

/// The full ordered tool-specific lifecycle a local item owes the client between
/// `output_item.added` and `output_item.done`, per issue #276.
///
/// `mcp_call` progresses `in_progress` then `completed`/`failed`, selected by
/// whether the item carries a non-null `error`. `web_search_call` progresses
/// `in_progress`, `searching`, then `completed` only when it actually completed
/// (web search has no conformant `failed` event, so other outcomes surface
/// through `output_item.done` alone). `mcp_list_tools` progresses `in_progress`
/// then `completed`/`failed`, selected the same way as `mcp_call`: a locally
/// seeded listing is created only on successful discovery (issue #1022) so its
/// terminal phase is `completed`, but a *deferred* entry the backend resolves
/// natively can also fail its `tools/list`, streaming `mcp_list_tools.failed` on
/// an item carrying an `error` — matching the expected terminal phase to that
/// error keeps the backend's real `output_item.done` from being dropped as
/// premature (issue #1022). A *local* discovery failure instead takes the
/// separate `response.mcp_list_tools.failed` terminal-SSE path in
/// `openai_mcp_tool_resolve` (issue #320) and never reaches this synthesis.
/// `mcp_approval_request` has no dedicated progress events; it surfaces through
/// `output_item.added`/`output_item.done` alone.
///
/// These are distinct API lifecycle events, not one combined milestone. Callers
/// diff this list against the phases already streamed so a partial in-band
/// lifecycle still gets exactly its missing events synthesized.
fn expected_phase_events(item: &Value) -> Vec<&'static str> {
    match item.get("type").and_then(Value::as_str) {
        Some("mcp_list_tools") => {
            let outcome = if item.get("error").is_some_and(|error| !error.is_null()) {
                "response.mcp_list_tools.failed"
            } else {
                "response.mcp_list_tools.completed"
            };
            vec!["response.mcp_list_tools.in_progress", outcome]
        },
        Some("mcp_call") => {
            let outcome = if item.get("error").is_some_and(|error| !error.is_null()) {
                "response.mcp_call.failed"
            } else {
                "response.mcp_call.completed"
            };
            vec!["response.mcp_call.in_progress", outcome]
        },
        Some("web_search_call") => {
            let mut events = vec![
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
            ];
            if item.get("status").and_then(Value::as_str) == Some("completed") {
                events.push("response.web_search_call.completed");
            }
            events
        },
        _ => Vec::new(),
    }
}

/// Normalize response identity, sequence numbers, and output indices.
#[expect(
    clippy::too_many_lines,
    reason = "single-pass normalization of three related SSE fields"
)]
fn normalize_logical_payload(ctx: &mut HttpFilterContext<'_>, payload: &mut Value, output_index_offset: u64) {
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
    if state.logical_stream_response_id.is_none() {
        state.logical_stream_response_id = payload
            .get("response")
            .and_then(|response| response.get("id"))
            .or_else(|| payload.get("response_id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }
    let response_id = state.logical_stream_response_id.as_deref();
    if let Some(object) = payload.as_object_mut() {
        if let Some(index) = object.get("output_index").and_then(Value::as_u64) {
            object.insert(
                "output_index".to_owned(),
                Value::Number(serde_json::Number::from(index.saturating_add(output_index_offset))),
            );
        }
        if let Some(response_id) = response_id {
            if object.contains_key("response_id") {
                object.insert("response_id".to_owned(), Value::String(response_id.to_owned()));
            }
            if let Some(response) = object.get_mut("response").and_then(Value::as_object_mut) {
                response.insert("id".to_owned(), Value::String(response_id.to_owned()));
            }
        }
        // #276/#985: stamp every emitted logical event with the running sequence
        // number so the client always sees a contiguous `0..N` series. Conformant
        // OpenAI Responses events always carry `sequence_number`, so this is a
        // no-op on the exercised paths; inserting it when absent keeps a future or
        // non-conformant event type from passing through unstamped and silently
        // opening a gap (the counter still advances once per emitted event).
        object.insert(
            "sequence_number".to_owned(),
            Value::Number(serde_json::Number::from(state.logical_stream_sequence)),
        );
    }
    state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);
}

/// Encode one canonical single-line SSE event.
fn encode_sse_event<T: Serialize + ?Sized>(event_type: &str, payload: &T, output: &mut Vec<u8>) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    // Serialize into the output buffer so logical-stream emission does not
    // allocate an intermediate `String` via `Display`. Truncate on failure so
    // a partial JSON write cannot be followed by the SSE delimiter.
    if let Err(error) = write_json_or_rollback(output, |out| serde_json::to_writer(out, payload)) {
        debug!(%error, "logical-stream payload serialization failed");
        return;
    }
    output.extend_from_slice(b"\n\n");
}

/// Write into `output`, restoring the pre-write length if `write` fails.
fn write_json_or_rollback<E>(output: &mut Vec<u8>, write: impl FnOnce(&mut Vec<u8>) -> Result<(), E>) -> Result<(), E> {
    let start = output.len();
    match write(output) {
        Ok(()) => Ok(()),
        Err(error) => {
            output.truncate(start);
            Err(error)
        },
    }
}

/// Encode a locally completed logical stream after a request-phase dispatch.
///
/// Some dispatch lifecycles finish before another upstream response exists, so
/// the response-body finalizer cannot emit the deferred terminal event. Build
/// the same canonical terminal representation directly from shared response
/// state for IRR to append after already-emitted logical stream chunks.
#[expect(
    clippy::too_many_lines,
    reason = "canonical local terminal construction and budget fallback"
)]
pub(crate) fn encode_local_completion(ctx: &mut HttpFilterContext<'_>) -> Option<Bytes> {
    let parser_deferred_done = ctx
        .get_filter_state::<StreamEventsState>()
        .is_some_and(|state| state.deferred_done);
    if let Some(stream) = ctx.get_filter_state::<StreamEventsState>() {
        stream.shared_current_output_bytes.store(0, Ordering::Relaxed);
    }
    let state = ctx.extensions.get::<ResponsesState>()?;
    let Some(local_output_upper_bound) = local_terminal_output_upper_bound(state) else {
        return Some(encode_retained_payload_error(ctx));
    };
    let local_output_capacity = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(ResponsesState::retained_payload_limit)
        .map_or(0, |_| local_output_upper_bound);
    {
        let state = ctx.extensions.get::<ResponsesState>()?;
        if !canonicalization_staging_bytes(state, local_output_upper_bound)
            .and_then(|staging| staging.checked_add(ready_file_search_clone_bytes(state)?))
            .is_some_and(|staging| state.can_replace_retained_payload(0, 0, staging))
        {
            return Some(encode_retained_payload_error(ctx));
        }
    }
    let mut output = prepare_local_terminal_events(ctx, local_output_capacity);
    let state = ctx.extensions.get_mut::<ResponsesState>()?;
    let deferred_done = state.deferred_stream_done || parser_deferred_done;
    if !state.response_object.is_object() {
        state.response_object = std::mem::take(&mut state.local_completion_response_template);
    }
    state.mark_current_output_changed();
    if !canonicalization_staging_bytes(state, output.len())
        .is_some_and(|staging| state.can_replace_retained_payload(0, 0, staging))
    {
        drop(output);
        return Some(encode_retained_payload_error(ctx));
    }
    // A local completion synthesizes both the wire terminal (below) and the store
    // source from this same `response_object`, so they cannot diverge; restore the
    // caller id on any rehydrated turn — there is no separate upstream wire whose
    // narrower eligibility to match.
    let restore_previous_response_id = state.history_rehydrated;
    if let Err(e) = canonicalize_logical_response(state, restore_previous_response_id) {
        // #1159: a lossy terminal restore fails closed. `output` already holds the
        // drained local-tool events, so emit the error terminal INLINE rather than via
        // `encode_local_error` (which would re-drain those events).
        return Some(encode_local_restore_error(ctx, output, &e));
    }
    state.mark_current_output_changed();
    if !state.response_object.is_object() {
        return None;
    }

    let sequence_number = state.logical_stream_sequence;
    state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);

    // The canonical response remains in state while its JSON is written into
    // the outgoing SSE buffer. Reserve the exact final frame before writing:
    // incremental Vec growth can otherwise retain twice the wire length.
    let terminal_frame_bytes = retained_json_bytes(&state.response_object)
        .and_then(|bytes| {
            bytes.checked_add(b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":".len())
        })
        .and_then(|bytes| bytes.checked_add(b",\"sequence_number\":".len()))
        .and_then(|bytes| bytes.checked_add(retained_json_bytes(&sequence_number)?))
        .and_then(|bytes| bytes.checked_add(b"}\n\n".len()))
        .and_then(|bytes| {
            if deferred_done {
                bytes.checked_add(b"data: [DONE]\n\n".len())
            } else {
                Some(bytes)
            }
        });
    let wire_peak = terminal_frame_bytes
        .and_then(|frame_bytes| output.len().checked_add(frame_bytes))
        .and_then(|final_len| {
            if final_len > output.capacity() {
                final_len.checked_add(output.capacity())
            } else {
                Some(output.capacity())
            }
        });
    if !wire_peak.is_some_and(|bytes| state.can_retain_payload(bytes))
        || terminal_frame_bytes.is_none_or(|bytes| output.try_reserve_exact(bytes).is_err())
        || !state.can_retain_payload(output.capacity())
    {
        drop(output);
        return Some(encode_retained_payload_error(ctx));
    }

    // #937: deliberately do NOT set `logical_stream_terminal_emitted` here.
    // Unlike `emit_deferred_terminal`, this local completion is returned to the
    // store as a buffered `TerminalResponse` at end-of-stream (see
    // `finish_deferred_local_response`), where the store already persists before
    // the body is written. Marking the flag would make the store skip that
    // end-of-stream persist and lose the record (#937 review regression).
    output.extend_from_slice(b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":");
    serde_json::to_writer(&mut output, &state.response_object).ok()?;
    output.extend_from_slice(b",\"sequence_number\":");
    serde_json::to_writer(&mut output, &sequence_number).ok()?;
    output.extend_from_slice(b"}\n\n");
    if deferred_done {
        output.extend_from_slice(b"data: [DONE]\n\n");
    }
    if !state.can_replace_retained_payload(0, 0, output.capacity()) {
        drop(output);
        return Some(encode_retained_payload_error(ctx));
    }
    // IRR can surface the local terminal body to outer filters as a non-EOS
    // chunk. Mark it only after encoding succeeded so they persist before
    // releasing that chunk; the store retains an EOS fallback if no such chunk
    // arrives.
    state.local_stream_terminal_emitted = true;
    Some(Bytes::from(output))
}

/// Emit a fail-closed terminal `error` frame INLINE for a locally completed stream
/// whose terminal client-tool restore was lossy (#1159).
///
/// `output` already holds the drained local-tool events from
/// `prepare_local_terminal_events`, so the error frame is appended directly rather
/// than through `encode_local_error` (which would re-drain those events). Marks the
/// record un-persistable so a later GET cannot serve the private lowered shape.
fn encode_local_restore_error(ctx: &mut HttpFilterContext<'_>, mut output: Vec<u8>, error: &SseParseError) -> Bytes {
    let message = error.to_string();
    let sequence_number = ctx.extensions.get_mut::<ResponsesState>().map_or(0, |state| {
        let sequence_number = state.logical_stream_sequence;
        state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);
        sequence_number
    });
    let mut payload = responses_error_sse_payload("server_error", &message);
    if let Some(object) = payload.as_object_mut() {
        object.insert("sequence_number".to_owned(), Value::from(sequence_number));
    }
    encode_sse_event("error", &payload, &mut output);
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.stream_error_message", message);
    ctx.set_metadata("responses.skip_persist", "true");
    Bytes::from(output)
}

/// Encode a terminal `error` event for an already-committed logical stream.
///
/// The dispatch owner (`openai_agentic_loop`) calls this during request-body
/// EOS when a request-phase dispatcher recorded a
/// [`DispatchFailure`](crate::openai::responses::state::DispatchFailure) after
/// the stream was already committed. It mirrors [`encode_local_completion`]: the
/// terminal frame is built directly from shared response state so IRR can append
/// it after the logical stream chunks already emitted this round, rather than
/// through the response-body finalizer (no upstream response body exists on a
/// dispatch failure).
///
/// A terminal `error` frame is never followed by a `[DONE]` sentinel — the SSE
/// error event is itself the stream terminator, matching the response-body
/// finalizer's own error branch, which emits the error and stops.
#[expect(
    clippy::too_many_lines,
    reason = "preflights local output and the error frame before encoding"
)]
pub(crate) fn encode_local_error(ctx: &mut HttpFilterContext<'_>, code: &str, message: &str) -> Option<Bytes> {
    let budget_failed = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(|state| state.retained_payload_failed);
    let output_capacity = if budget_failed {
        0
    } else {
        let state = ctx.extensions.get::<ResponsesState>()?;
        let Some(local_output_upper_bound) = local_terminal_output_upper_bound(state) else {
            return Some(encode_retained_payload_error(ctx));
        };
        // `responses_error_sse_payload` owns a copy of the message while its
        // escaped JSON is written into the wire buffer. Eight bytes per input
        // byte cover both owners even for maximal JSON escaping.
        let error_upper_bound = message
            .len()
            .checked_add(code.len())
            .and_then(|bytes| bytes.checked_mul(8))
            .and_then(|bytes| bytes.checked_add(512));
        let projected = error_upper_bound
            .and_then(|bytes| bytes.checked_add(local_output_upper_bound))
            .and_then(|bytes| bytes.checked_add(ready_file_search_clone_bytes(state)?));
        if !projected.is_some_and(|bytes| state.can_retain_payload(bytes)) {
            return Some(encode_retained_payload_error(ctx));
        }
        state.retained_payload_limit().map_or(0, |_| local_output_upper_bound)
    };
    let mut output = if budget_failed {
        Vec::new()
    } else {
        prepare_local_terminal_events(ctx, output_capacity)
    };
    let state = ctx.extensions.get_mut::<ResponsesState>()?;
    let sequence_number = state.logical_stream_sequence;
    state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);

    let mut payload = responses_error_sse_payload(code, message);
    if let Some(object) = payload.as_object_mut() {
        object.insert("sequence_number".to_owned(), Value::from(sequence_number));
    }
    encode_sse_event("error", &payload, &mut output);
    Some(Bytes::from(output))
}

/// Emit locally executed tool lifecycles before a request-phase terminal frame.
///
/// Request-phase completion and dispatch failure have no later upstream
/// response, so the normal response-body finalizer cannot drain pending
/// synthesis or flush locally generated output items for them.
fn prepare_local_terminal_events(ctx: &mut HttpFilterContext<'_>, output_capacity: usize) -> Vec<u8> {
    let mut output = if output_capacity == 0 {
        Vec::new()
    } else {
        Vec::with_capacity(output_capacity)
    };
    local_tools::drain_local_tool_synthesis(ctx, &mut output);
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.provider_streamed_terminal_ids.clear();
    }
    flush_local_output_items(ctx, &mut output);
    output
}

/// Drop retained payload and construct the bounded terminal error used after a
/// local-output admission failure.
pub(crate) fn encode_retained_payload_error(ctx: &mut HttpFilterContext<'_>) -> Bytes {
    let state = ctx.extensions.get_mut::<ResponsesState>();
    let sequence_number = state.map_or(0, |state| {
        state.discard_payload_for_budget_error();
        let sequence_number = state.logical_stream_sequence;
        state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);
        sequence_number
    });
    #[cfg(feature = "store")]
    super::store::discard_retained_request_payload(ctx);
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.stream_error_message", RETAINED_PAYLOAD_OVERFLOW_MESSAGE);
    ctx.set_metadata("responses.skip_persist", "true");
    let mut output = Vec::new();
    let mut payload = responses_error_sse_payload("server_error", RETAINED_PAYLOAD_OVERFLOW_MESSAGE);
    if let Some(object) = payload.as_object_mut() {
        object.insert("sequence_number".to_owned(), Value::from(sequence_number));
    }
    encode_sse_event("error", &payload, &mut output);
    Bytes::from(output)
}

/// Emit the held terminal event only when the current IRR step is terminal.
#[expect(
    clippy::too_many_lines,
    reason = "deferred terminal finalization and published charge update"
)]
fn finalize_logical_stream(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(mut parser_state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };
    ctx.set_metadata(super::STREAM_ERROR_FINALIZER_ARMED_KEY, "false");

    // Preserve any non-terminal logical events `process_chunk` already emitted
    // for this final chunk, then append synthesized local-tool events and the
    // deferred terminal. A transport that reassembles the whole stream before
    // releasing it (e.g. `responses_to_chat_completions`) delivers the
    // created/delta events and deferred terminal together in the end-of-stream
    // chunk; starting from an empty buffer here would drop those earlier events.
    // Agentic collection can append history immediately before this EOS callback.
    // Refresh the shared baseline before admitting synthesis construction.
    parser_state.clear_shared_budget_caches();
    let mut output = match body.take() {
        None => Vec::new(),
        Some(bytes) => {
            // A unique Bytes can move its allocation into Vec. A shared or
            // sliced Bytes needs a second buffer, so admit both live owners
            // before copying the final callback's already-normalized wire.
            let buffer = bytes.try_into_mut();
            let copy_peak = match &buffer {
                Ok(unique) => Some(unique.len()),
                Err(shared) => shared.len().checked_mul(2),
            };
            if copy_peak.is_some_and(|bytes| stream_payload_fits(ctx, &parser_state, bytes)) {
                match buffer {
                    Ok(unique) => Vec::from(unique),
                    Err(shared) => shared.to_vec(),
                }
            } else {
                drop(buffer);
                record_retained_payload_overflow(ctx, &mut parser_state);
                Vec::new()
            }
        },
    };
    let synthesis_staging = if ctx.get_metadata("responses.stream_error_code").is_some() {
        Some(0)
    } else {
        ctx.extensions
            .get::<ResponsesState>()
            .map_or(Some(0), ready_file_search_synthesis_staging_bytes)
    };
    let preflight_staging = synthesis_staging.and_then(|bytes| {
        // Appending synthesized events can reallocate the existing EOS output
        // while its old allocation is still live.
        let existing = if bytes == 0 {
            Some(output.len())
        } else {
            output.len().checked_mul(2)
        }?;
        existing.checked_add(bytes)
    });
    if preflight_staging.is_some_and(|bytes| stream_payload_fits(ctx, &parser_state, bytes)) {
        // #1046 §4.2: drain file_search synthesis before terminal/error finalization,
        // under the precedence policy. The owner queues each reconciled call by its
        // absolute output index this round, but the request-phase dispatcher only
        // reconciles it at the NEXT re-entry's request-body EOS; drain therefore defers
        // still-pending items and synthesizes them at the finalize that follows their
        // reconciliation. A validation failure here calls fs_end_stream_with_error_ctx
        // (site (b), §7.3) so the error branch below is selected and the router does not
        // re-fire.
        local_tools::drain_local_tool_synthesis(ctx, &mut output);
    } else {
        output.clear();
        record_retained_payload_overflow(ctx, &mut parser_state);
    }
    // #313 P1 (DoS bound): this round's provider-streamed observation set is stale
    // once the round that recorded it finalizes; clear it unconditionally here — NOT
    // inside drain_local_tool_synthesis, which early-returns on an empty synthesis
    // queue (exactly the all-natives-streamed-live case) — so it cannot accumulate
    // across IRR continuation rounds and bypass the max_state_bytes ceiling. The ids
    // are stale after the round that recorded them, so clearing loses nothing.
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.provider_streamed_terminal_ids.clear();
    }
    // The agentic-loop callback runs before this finalizer and can append the
    // just-finished round to messages/persisted_messages. Their per-stream
    // baseline from the first chunk is therefore stale at EOS.
    parser_state.clear_shared_budget_caches();
    if !stream_payload_fits(ctx, &parser_state, output.len()) {
        output.clear();
        record_retained_payload_overflow(ctx, &mut parser_state);
    }
    // #1046 P1: a terminal failure recorded after the owner already published its
    // per-round continuation — our own parse/validation error (`stream_error_code`, e.g.
    // set by validate_stream_end at EOS, which runs AFTER the owner records assignments)
    // or a flat upstream `error` completion (`stream_completion == "error"`, which sets no
    // error code) — must clear a stale owner `action="loop"` to the two-key stop, or the
    // error frame is suppressed and another IRR round fires. Scoped to the owner: it is
    // the single continuation authority, and clearing it also covers the oversized
    // web_search batch case (the owner sets loop before web_search caps the batch).
    let owner_looping = ctx
        .filter_results
        .get("openai_agentic_loop")
        .and_then(|results| results.get("action"))
        == Some("loop");
    let terminal_error = ctx.get_metadata("responses.stream_error_code").is_some()
        || ctx.get_metadata("responses.stream_completion") == Some("error");
    if owner_looping && terminal_error {
        crate::openai::responses::fs_arm_stream_stop(ctx);
    }
    let continues = logical_stream_continues(ctx); // re-read AFTER drain + arm-stop: a
    // (b)-site failure or the arm-stop above flips the owner action=done.
    finalize_emit_terminal(ctx, &mut parser_state, &mut output, continues);
    if !publish_stream_payload(ctx, &parser_state) {
        output.clear();
        record_retained_payload_overflow(ctx, &mut parser_state);
        if let Some(error) = encode_local_error(ctx, "server_error", RETAINED_PAYLOAD_OVERFLOW_MESSAGE) {
            output.extend_from_slice(&error);
        }
    }
    *body = (!output.is_empty()).then(|| Bytes::from(output));
    // This upstream round is finished. IRR carries ResponsesState into the
    // next step but drops filter-local state; leaving its published charge
    // behind would make the next agentic admission see a ghost parser owner.
    // Keep only lifecycle metadata for downstream inspection of this step.
    parser_state.frame_parser.clear();
    parser_state.tool_call_args.clear();
    parser_state.rejected_tool_call_args.clear();
    parser_state.deferred_terminal = None;
    parser_state.local_tool_items.clear();
    parser_state.client_tool_items.clear();
    debug_assert_eq!(
        parser_state.retained_payload_bytes(),
        Some(0),
        "all parser payload owners must be released before the shared charge is cleared"
    );
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.retained_stream_parser_bytes = 0;
    }
    ctx.insert_filter_state(parser_state);
}

/// Emit the logical stream's terminal frame: a locally recorded `error`, otherwise
/// the held deferred terminal snapshot. A lossy #1159 client-tool restore on the
/// deferred path is routed to a fail-closed error terminal instead.
fn finalize_emit_terminal(
    ctx: &mut HttpFilterContext<'_>,
    parser_state: &mut StreamEventsState,
    output: &mut Vec<u8>,
    continues: bool,
) {
    if !continues && let Some(mut error) = logical_stream_error(ctx) {
        // #276: surface any locally executed tool items that never reached the
        // client before the stream terminates with an error, so already-executed
        // tool activity is not silently dropped by a resumed-round parse failure.
        if !flush_local_output_items_with_budget(ctx, parser_state, output) {
            error = logical_stream_error(ctx).unwrap_or(error);
        }
        normalize_logical_payload(ctx, &mut error, parser_state.output_index_offset);
        encode_sse_event("error", &error, output);
    } else if !continues
        && let Some(mut terminal) = parser_state.deferred_terminal.take()
        && let Err(e) = emit_deferred_terminal(ctx, &mut terminal, parser_state, output)
    {
        output.clear();
        if ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| state.retained_payload_failed)
        {
            if let Some(mut error) = logical_stream_error(ctx) {
                normalize_logical_payload(ctx, &mut error, parser_state.output_index_offset);
                encode_sse_event("error", &error, output);
            }
        } else {
            emit_deferred_terminal_restore_error(ctx, output, &e);
        }
    }
}

/// Emit the deferred terminal snapshot as the logical stream's final event,
/// preceded by any locally generated tool items not yet streamed to the client.
#[expect(clippy::too_many_lines, reason = "ordered logical stream terminal reconciliation")]
fn emit_deferred_terminal(
    ctx: &mut HttpFilterContext<'_>,
    terminal: &mut DeferredTerminalEvent,
    parser_state: &mut StreamEventsState,
    output: &mut Vec<u8>,
) -> Result<(), SseParseError> {
    if !flush_local_output_items_with_budget(ctx, parser_state, output) {
        return Err(SseParseError::StreamPoisoned);
    }
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
    // Match the wire-rewrite decision so the persisted store source cannot disagree
    // with the streamed frame (#1150).
    let restore_previous_response_id = state.previous_response_id_stream_restore_armed;
    let preflight = canonicalization_staging_bytes(state, output.len())
        .and_then(|staging| terminal.retained_payload_bytes()?.checked_add(staging));
    // The parser owner is already published in ResponsesState for cross-filter
    // admission. Replace that published charge with the current local size;
    // only the terminal-construction copy above is additional staging.
    if !preflight.is_some_and(|staging| {
        parser_state.retained_payload_bytes().is_some_and(|parser_bytes| {
            state.can_replace_retained_payload(state.retained_stream_parser_bytes, parser_bytes, staging)
        })
    }) {
        output.clear();
        record_retained_payload_overflow(ctx, parser_state);
        return Err(SseParseError::StreamPoisoned);
    }
    let sequence_before_terminal = state.logical_stream_sequence;
    canonicalize_logical_response(state, restore_previous_response_id)?;
    state.mark_current_output_changed();
    parser_state.shared_current_output_bytes.store(0, Ordering::Relaxed);
    normalize_logical_payload(ctx, &mut terminal.payload, parser_state.output_index_offset);
    let terminal_bytes = ctx.extensions.get::<ResponsesState>().and_then(|state| {
        retained_json_bytes(&BorrowedTerminalPayload {
            metadata: &terminal.payload,
            response: &state.response_object,
        })
    });
    let terminal_frame_bytes = terminal_bytes
        .and_then(|bytes| bytes.checked_add(b"event: ".len()))
        .and_then(|bytes| bytes.checked_add(terminal.event_type.len()))
        .and_then(|bytes| bytes.checked_add(b"\ndata: ".len()))
        .and_then(|bytes| bytes.checked_add(b"\n\n".len()))
        .and_then(|bytes| {
            if parser_state.deferred_done {
                bytes.checked_add(b"data: [DONE]\n\n".len())
            } else {
                Some(bytes)
            }
        });
    // The deferred metadata remains independently owned while the borrowed
    // response is serialized into the wire buffer. The wire and metadata
    // therefore coexist even though the response tree itself is only borrowed.
    let final_wire_bytes = terminal_frame_bytes.and_then(|bytes| output.len().checked_add(bytes));
    let staging = final_wire_bytes
        .and_then(|bytes| {
            // If the existing output buffer cannot hold the terminal frame,
            // reserve its final capacity once before writing. The old Vec
            // allocation stays live while that new buffer is allocated.
            let old_capacity = if bytes > output.capacity() {
                output.capacity()
            } else {
                0
            };
            bytes.checked_add(old_capacity)
        })
        .and_then(|bytes| terminal.retained_payload_bytes()?.checked_add(bytes));
    if !staging.is_some_and(|bytes| stream_payload_fits(ctx, parser_state, bytes))
        || terminal_frame_bytes.is_none_or(|bytes| output.try_reserve_exact(bytes).is_err())
    {
        // The failed terminal never reached the client. Reuse its sequence
        // number for the single error event emitted by the finalizer.
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.logical_stream_sequence = sequence_before_terminal;
        }
        output.clear();
        record_retained_payload_overflow(ctx, parser_state);
        return Err(SseParseError::StreamPoisoned);
    }
    // Persist before the deferred terminal chunk can be released to the client.
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Err(SseParseError::StreamPoisoned);
    };
    encode_sse_event(
        &terminal.event_type,
        &BorrowedTerminalPayload {
            metadata: &terminal.payload,
            response: &state.response_object,
        },
        output,
    );
    if parser_state.deferred_done {
        output.extend_from_slice(b"data: [DONE]\n\n");
    }
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.logical_stream_terminal_emitted = true;
    }
    Ok(())
}

/// Route a lossy deferred-terminal client-tool restore to a fail-closed error
/// terminal (#1159): emit an `error` event instead of leaking a private lowered
/// shape and mark the record un-persistable so a later GET cannot serve the leak.
fn emit_deferred_terminal_restore_error(ctx: &mut HttpFilterContext<'_>, output: &mut Vec<u8>, error: &SseParseError) {
    let message = error.to_string();
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.skip_persist", "true");
    if let Some(err_bytes) = encode_local_error(ctx, "server_error", &message) {
        output.extend_from_slice(&err_bytes);
    }
    ctx.set_metadata("responses.stream_error_message", message);
}

/// Emit pending local lifecycle events only when their complete SSE staging
/// fits the active aggregate budget.
///
/// On failure the entire current chunk is suppressed before the caller appends
/// the single bounded terminal error. The emitted-item bookkeeping performed by
/// [`flush_local_output_items`] is request payload and is cleared by
/// [`record_retained_payload_overflow`] along with the rest of the failed state.
fn flush_local_output_items_with_budget(
    ctx: &mut HttpFilterContext<'_>,
    parser_state: &mut StreamEventsState,
    output: &mut Vec<u8>,
) -> bool {
    let Some(responses) = ctx.extensions.get::<ResponsesState>() else {
        flush_local_output_items(ctx, output);
        return true;
    };
    let projected = local_terminal_output_upper_bound(responses).and_then(|bytes| output.len().checked_add(bytes));
    if !projected.is_some_and(|bytes| stream_payload_fits(ctx, parser_state, bytes)) {
        output.clear();
        record_retained_payload_overflow(ctx, parser_state);
        return false;
    }

    flush_local_output_items(ctx, output);
    if !stream_payload_fits(ctx, parser_state, output.len()) {
        output.clear();
        record_retained_payload_overflow(ctx, parser_state);
        return false;
    }
    true
}

/// Whether the agentic-loop owner requested another inference step.
///
/// After the #1046 unification the owner (`openai_agentic_loop`) is the single
/// authority that decides whether the logical stream continues: its
/// `has_dispatchable_calls` signal is a strict superset of every dispatcher's
/// per-round work (`web_search` calls, `file_search` assignments, MCP-classified
/// tool calls), so keying on the owner alone covers all three dispatchers.
fn logical_stream_continues(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.filter_results
        .get("openai_agentic_loop")
        .and_then(|results| results.get("action"))
        == Some("loop")
}

/// Return a locally generated terminal error for an already-committed stream.
fn logical_stream_error(ctx: &HttpFilterContext<'_>) -> Option<Value> {
    let code = ctx.get_metadata("responses.stream_error_code")?;
    let message = ctx.get_metadata("responses.stream_error_message")?;
    Some(responses_error_sse_payload(code, message))
}

/// Make the response-store source agree with the logical SSE terminal.
///
/// `restore_previous_response_id` decides whether to repair the id the backend
/// echoed as `null` after a rehydrated turn stripped it from the upstream request
/// (#1150). It MUST mirror whichever path owns the client-visible terminal, or a
/// later GET disagrees with what the client streamed:
/// - upstream streaming: the wire is rewritten by `openai_responses_rehydrate`, so the caller passes
///   `state.previous_response_id_stream_restore_armed` — the filter's own `eligible_previous_response_id_stream`
///   decision, which declines validator-bearing and non-200 streams the store must therefore also leave alone (issue
///   #1150 review);
/// - local completion: this function's output *is* the wire terminal, so the two cannot diverge and the caller restores
///   on `history_rehydrated` alone.
///
/// Either way the id is only ever restored, never fabricated: a non-rehydrated
/// turn keeps the real value the backend echoed.
#[expect(
    clippy::too_many_lines,
    reason = "canonical terminal construction follows ordered state transitions"
)]
fn canonicalize_logical_response(
    state: &mut ResponsesState,
    restore_previous_response_id: bool,
) -> Result<(), SseParseError> {
    let logical_id = state.logical_stream_response_id.clone();
    let restored_previous_response_id = restore_previous_response_id
        .then(|| state.previous_response_id.clone())
        .flatten();
    // Prefer the cross-round accumulator populated by dispatch/loop filters
    // (agentic pipelines). When no such filter ran — a plain one-round logical
    // stream — it stays empty, so fall back to the terminal event's own output
    // rather than clobber it with nothing. Mirrors `finalize_response_body`.
    if !state.accumulated_output.is_empty()
        && let Some(response) = state.response_object.as_object_mut()
    {
        response.insert(
            "output".to_owned(),
            Value::Array(std::mem::take(&mut state.accumulated_output)),
        );
    }
    // No dispatcher runs after terminal canonicalization. The assignments
    // index the old accumulator and must not outlive its moved output.
    state.tool_calls.clear();
    state.tool_search_calls.clear();
    state.web_search_calls.clear();
    // Rewrite file_search citation markers into typed annotations on the final
    // assistant message, mirroring the buffered `annotate_response` finalize
    // path. In streaming the dispatcher reconciled `citation_files` during a
    // prior request-phase round, but the model's citing answer only arrives in
    // the terminal round — so this is the single point where both are present.
    // No-op when no dispatcher recorded citation files. Best-effort: the logical
    // stream is already committed here, so a malformed marker degrades to
    // un-annotated text rather than aborting the terminal.
    let output = state
        .response_object
        .get_mut("output")
        .and_then(Value::as_array_mut)
        .map(Vec::as_mut_slice)
        .unwrap_or_default();
    if let Err(error) =
        crate::openai::responses::file_search_callout::citations::annotate_output_items(output, &state.citation_files)
    {
        tracing::warn!(%error, "failed to annotate logical stream response citations");
    }
    if let Some(response) = state.response_object.as_object_mut() {
        if let Some(logical_id) = logical_id {
            response.insert("id".to_owned(), Value::String(logical_id));
        }
        if let Some(prev_id) = restored_previous_response_id {
            response.insert("previous_response_id".to_owned(), Value::String(prev_id));
        }
        if !state.usage.is_null() {
            // Keep the request-wide usage accumulator available to downstream
            // consumers after the terminal response is recorded.
            response.insert("usage".to_owned(), state.usage.clone());
        }
    }
    restore_terminal_client_tools(state)
}

/// #1159: last-chance restoration of every lowered client-tool call plus
/// `tools`/`tool_choice` on the terminal response object. A lossy restore fails
/// the whole logical stream closed rather than leaking a private lowered shape.
///
/// The deferred terminal borrows this same restored response object for the
/// wire, so there is no second output tree to synchronize.
fn restore_terminal_client_tools(state: &mut ResponsesState) -> Result<(), SseParseError> {
    if !state.client_tool_lowering.is_empty() {
        // The output items were moved into `response_object` above. A lossy
        // restore must fail closed before that object is sent or persisted.
        if !state.response_object.is_object() {
            return Err(SseParseError::ClientToolRestore {
                key: "terminal".to_owned(),
                reason: "terminal response object unavailable for client-tool restore".to_owned(),
            });
        }
        restore_snapshot(&mut state.response_object, &state.client_tool_lowering).map_err(|item_type| {
            SseParseError::ClientToolRestore {
                key: "terminal".to_owned(),
                reason: format!("lossy restore of {item_type}"),
            }
        })?;
    }
    restore_snapshot_tools(&mut state.response_object, state.client_tool_echo.as_ref());
    Ok(())
}

/// Serialize the terminal envelope while borrowing the canonical response
/// owned by `ResponsesState` for persistence. The deferred metadata never owns
/// another full response tree.
struct BorrowedTerminalPayload<'a> {
    /// Deferred lifecycle metadata and event type.
    metadata: &'a Value,
    /// Canonical response owned by request state.
    response: &'a Value,
}

impl Serialize for BorrowedTerminalPayload<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Some(metadata) = self.metadata.as_object() else {
            return self.metadata.serialize(serializer);
        };
        let mut map = serializer.serialize_map(Some(metadata.len()))?;
        for (key, value) in metadata {
            if key == "response" {
                // Keep the existing map position, including JSON key order,
                // while borrowing the canonical response from shared state.
                map.serialize_entry(key, self.response)?;
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

/// Check whether the stream has exceeded its wall-clock timeout.
fn check_timeout(state: &StreamEventsState, now: Instant) -> Result<(), SseParseError> {
    let Some(started_at) = state.started_at else {
        return Ok(());
    };
    let elapsed = now.duration_since(started_at);
    if elapsed >= state.timeout {
        return Err(SseParseError::Timeout {
            elapsed,
            limit: state.timeout,
        });
    }
    Ok(())
}

/// Record whether an event signals stream completion.
fn record_completion(state: &mut StreamEventsState, event: &ResponsesEvent, now: Instant) -> Result<(), SseParseError> {
    if matches!(event, ResponsesEvent::Error(_)) {
        if state.completion_state == CompletionState::Error {
            return Err(SseParseError::EventAfterTerminal {
                event_type: event.event_type().to_owned(),
            });
        }
        mark_complete(state, CompletionState::Error, now);
        return Ok(());
    }

    if state.completion_state != CompletionState::Open {
        return Err(SseParseError::EventAfterTerminal {
            event_type: event.event_type().to_owned(),
        });
    }

    if event.is_terminal() {
        mark_complete(state, CompletionState::TerminalLifecycle, now);
    }

    Ok(())
}

/// Record the first terminal-state timestamp while allowing stronger
/// states to replace weaker ones.
fn mark_complete(state: &mut StreamEventsState, new_state: CompletionState, now: Instant) {
    state.completion_state = new_state;
    state.completed_at.get_or_insert(now);
}

/// Check that the SSE stream terminated with a terminal event.
fn validate_stream_end(ctx: &mut HttpFilterContext<'_>) {
    match stream_end_kind(ctx) {
        StreamEndKind::Complete => {},
        StreamEndKind::Incomplete { timed_out } => {
            record_incomplete_stream(ctx, timed_out);
        },
    }
    debug!("stream_events processing complete");
}

/// Classify how the current parser state ended.
fn stream_end_kind(ctx: &HttpFilterContext<'_>) -> StreamEndKind {
    let Some(state) = ctx.get_filter_state::<StreamEventsState>() else {
        return StreamEndKind::Complete;
    };
    let checked_at = state.completed_at.unwrap_or_else(Instant::now);
    match check_timeout(state, checked_at) {
        Err(error) => {
            warn!(%error, "stream did not terminate cleanly");
            StreamEndKind::Incomplete { timed_out: true }
        },
        Ok(()) if state.completion_state == CompletionState::Open => {
            warn!("stream did not terminate cleanly: missing terminal event");
            StreamEndKind::Incomplete { timed_out: false }
        },
        Ok(())
            if state
                .client_tool_items
                .iter()
                .any(|item| !matches!(item.phase, client_tools::ClientToolPhase::Done)) =>
        {
            // #1159: incomplete client-tool lifecycle (never reached Done) → fail closed.
            warn!("stream did not terminate cleanly: incomplete client-tool lifecycle");
            StreamEndKind::Incomplete { timed_out: false }
        },
        Ok(()) => StreamEndKind::Complete,
    }
}

/// Publish incomplete-stream metadata for persistence and logical-stream errors.
fn record_incomplete_stream(ctx: &mut HttpFilterContext<'_>, timed_out: bool) {
    ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
    if ctx.get_metadata("responses.stream_error_code").is_some() {
        return;
    }
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata(
        "responses.stream_error_message",
        if timed_out {
            "upstream Responses stream exceeded timeout"
        } else {
            "upstream Responses stream did not terminate cleanly"
        },
    );
    ctx.set_metadata("responses.skip_persist", "true");
}

/// How an SSE stream ended from the parser's point of view.
enum StreamEndKind {
    /// A terminal lifecycle or error event was observed in time.
    Complete,
    /// The stream ended without a clean terminal event.
    Incomplete {
        /// Whether the wall-clock budget was exceeded.
        timed_out: bool,
    },
}

/// Whether the response is a successful `text/event-stream` response.
fn is_success_sse_response(ctx: &HttpFilterContext<'_>) -> bool {
    let Some(resp) = ctx.response_header.as_ref() else {
        return true;
    };

    if !resp.status.is_success() {
        return false;
    }

    resp.headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(is_event_stream_content_type)
}

/// Whether the upstream response advertises a `Content-Encoding`.
///
/// The SSE frame parser consumes raw bytes, so a compressed body would be
/// parsed as opaque data. `on_request` strips `Accept-Encoding` to keep a
/// compliant backend from encoding; this guards the residual case of a
/// non-compliant backend that encodes anyway.
fn response_is_encoded(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_some_and(|resp| resp.headers.contains_key(http::header::CONTENT_ENCODING))
}

#[cfg(test)]
mod tests;
