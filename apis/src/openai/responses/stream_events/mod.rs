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
mod config;
mod local_tools;

use std::{
    collections::{BTreeSet, hash_map::DefaultHasher},
    hash::{Hash as _, Hasher as _},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState,
    SubRequestResponseMode, parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, trace, warn};

#[cfg(test)]
use self::accumulator::accumulate_response_object;
use self::{accumulator::accumulate_event, config::StreamEventsConfig};
use crate::{
    classifier::is_responses_create,
    is_event_stream_content_type,
    openai::{
        responses::{
            error::{responses_error_rejection, responses_error_sse_payload},
            state::{EmittedItem, ResponsesState},
        },
        sse::{SseFrame, SseFrameParser, SseParseError, SseParserConfig, responses::ResponsesEvent},
    },
};

/// A per-turn terminal event held until the agentic transition is known.
struct DeferredTerminalEvent {
    /// Canonical event type.
    event_type: String,
    /// Parsed event payload.
    payload: Value,
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
}

/// Composes the current IRR execution into one logical Responses stream.
///
/// Must run inside an `iterative_request_router` step. Running it
/// elsewhere is a misconfiguration and fails closed at request time.
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
/// ```
pub struct OpenaiStreamEventsFilter {
    /// Configuration for the SSE frame parser.
    parser_config: SseParserConfig,
    /// Cap on accumulated bytes per tool-call argument string.
    max_tool_call_argument_bytes: usize,
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
        })
    }

    /// Whether per-request parser state has been installed.
    fn is_armed(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.get_filter_state::<StreamEventsState>().is_some()
    }

    /// Install fresh parser state for one inference stream.
    fn arm(&self, ctx: &mut HttpFilterContext<'_>) {
        let (iteration, output_index_offset) = ctx.extensions.get_mut::<ResponsesState>().map_or((0, 0), |state| {
            let output_index_offset = u64::try_from(state.accumulated_output.len()).unwrap_or(u64::MAX);
            // Invalidate the previous round's terminal response object before a
            // resumed round begins. Only this round's own terminal event may
            // repopulate it; otherwise a provider `error` in the resumed round
            // would leave the prior round's completed response live and let the
            // store persist stale success as the logical result.
            state.response_object = Value::Null;
            (state.iteration, output_index_offset)
        });
        ctx.insert_filter_state(StreamEventsState {
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
            iteration,
            output_index_offset,
            deferred_terminal: None,
            deferred_done: false,
            local_items_flushed: false,
            local_tool_items: std::collections::HashMap::new(),
        });
        ctx.set_metadata("responses.stream_completion", "open");
        // Publish a per-round marker that `openai_agentic_loop` reads (and then
        // consumes) to confirm this typed-streaming round can surface
        // loop-terminal errors through `finalize_logical_stream`. Refreshed
        // every armed round because the agentic loop overwrites it after each
        // check.
        ctx.set_metadata("responses.logical_stream", "true");
        // Per-consumer capability marker (§7.1). file_search is the only
        // consumer today.
        ctx.set_metadata("responses.logical_stream.file_search", "true");
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

#[async_trait]
impl HttpFilter for OpenaiStreamEventsFilter {
    fn name(&self) -> &'static str {
        "openai_stream_events"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::None
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
        let typed_streaming = ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming;
        // The `iterative_request_router` runner moves request extensions into
        // each step but builds a fresh `filter_metadata` map, so metadata set by
        // pre-IRR filters (e.g. `openai_responses_format`) is not visible here.
        // `ResponsesState` is created pre-IRR and travels through extensions, so
        // fall back to it for format and stream detection — mirroring how
        // `responses_to_chat_completions` resolves `request_is_streaming`.
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
        // `IterationState` is inserted by the IRR runner before the request phase
        // of every iteration (including iteration 0), so its presence is the
        // runtime signal that the filter is placed inside an IRR step.
        let inside_irr = ctx.extensions.get::<IterationState>().is_some();
        let decision = arm_decision(is_responses && is_streaming, inside_irr);
        if let Some(action) = self.apply_arm_decision(ctx, decision) {
            return Ok(action);
        }

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !Self::is_armed(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !is_success_sse_response(ctx) {
            debug!("disarming stream_events: response is not 2xx text/event-stream");
            ctx.remove_filter_state::<StreamEventsState>();
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

        process_chunk(ctx, body);

        if end_of_stream {
            validate_stream_end(ctx);
            finalize_logical_stream(ctx, body);
        }

        Ok(FilterAction::Continue)
    }
}

/// Parse SSE frames, accumulating state and optionally normalizing output.
fn process_chunk(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(bytes) = body.as_ref() else {
        return;
    };

    let Some(mut state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };

    let now = Instant::now();
    state.started_at.get_or_insert(now);

    let parsed = parse_and_accumulate(&mut state, ctx, bytes, now);
    handle_parse_result(ctx, body, &state, parsed);

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
    ctx.set_metadata(
        "responses.stream_error_message",
        "upstream Responses stream could not be parsed",
    );
    ctx.set_metadata("responses.skip_persist", "true");
    *body = None;
}

/// Parse frames from raw bytes and accumulate events.
fn parse_and_accumulate(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    bytes: &Bytes,
    now: Instant,
) -> Result<Option<Bytes>, SseParseError> {
    check_timeout(state, now)?;

    let frames = state.frame_parser.parse_chunk_with_counted_event_limit(
        bytes,
        state.event_count,
        state.max_events,
        |frame| frame.data != b"[DONE]",
    )?;

    // Parse and validate every frame before mutating shared state or emitting a
    // byte, then commit accumulation and emission only once the whole chunk
    // parses. A malformed frame aborts the chunk atomically, so no local-tool
    // milestone is recorded for bytes that never reach the client and EOS
    // recovery still re-synthesizes the executed tool items (#276 finding 3).
    let events = parse_chunk_events(state, &frames, now)?;
    let logical_output = commit_chunk_events(state, ctx, events);

    Ok((!logical_output.is_empty()).then(|| Bytes::from(logical_output)))
}

/// Phase 1: parse and validate every frame in a chunk before any mutation.
///
/// Returns the parsed non-`[DONE]` events, failing closed on the first malformed
/// frame so the caller can discard the whole chunk without having recorded any
/// local-tool milestone (#276 finding 3).
fn parse_chunk_events(
    state: &mut StreamEventsState,
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
        events.push(event);
    }
    Ok(events)
}

/// Phase 2: commit accumulation and logical emission for a fully parsed chunk.
///
/// Both steps are infallible, so every recorded milestone corresponds to bytes
/// that actually reach the client. Returns the logical-stream bytes.
fn commit_chunk_events(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    events: Vec<ResponsesEvent>,
) -> Vec<u8> {
    let mut logical_output = Vec::new();
    for event in events {
        accumulate_event(ctx, state, &event);
        append_logical_event(state, ctx, event, &mut logical_output);
    }

    // Mirror the parser's deferred-`[DONE]` decision into shared response state,
    // but only now that the whole chunk has parsed and committed. Filter-local
    // parser state is re-armed before request-side dispatchers run on the next
    // IRR step, so the sentinel must survive in shared state as well.
    if state.deferred_done
        && let Some(response_state) = ctx.extensions.get_mut::<ResponsesState>()
    {
        response_state.deferred_stream_done = true;
    }

    logical_output
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
    // `response.mcp_call.*` progress event marks the lifecycle as already streamed
    // in-band (so the flush does not re-synthesize it). Persisted across rounds
    // via `emitted_output_items`, this is what a resumed round's flush consults.
    record_model_output_item(ctx, event);

    // #276: ahead of the first model output *content* event, stream any locally
    // generated tool items (MCP calls/approvals, or web searches absent from the
    // upstream stream) that the tool-dispatch filters appended to
    // `accumulated_output` but never emitted incrementally. They must precede the
    // resumed model output and occupy their reserved output indices.
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
/// an actual `response.web_search_call.*` / `response.mcp_call.*` event proves
/// that specific phase reached the client, so each is recorded individually by
/// its event type. Deriving the lifecycle from `done` would suppress the
/// synthesized progress a partial `added → in_progress → done` sequence still
/// owes for its missing `searching`/`completed` phases.
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
fn encode_sse_event(event_type: &str, payload: &Value, output: &mut Vec<u8>) {
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
pub(crate) fn encode_local_completion(ctx: &mut HttpFilterContext<'_>) -> Option<Bytes> {
    let parser_deferred_done = ctx
        .get_filter_state::<StreamEventsState>()
        .is_some_and(|state| state.deferred_done);
    let state = ctx.extensions.get_mut::<ResponsesState>()?;
    let deferred_done = state.deferred_stream_done || parser_deferred_done;
    canonicalize_logical_response(state);
    if !state.response_object.is_object() {
        return None;
    }

    let sequence_number = state.logical_stream_sequence;
    state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);

    let mut output = b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":".to_vec();
    serde_json::to_writer(&mut output, &state.response_object).ok()?;
    output.extend_from_slice(b",\"sequence_number\":");
    serde_json::to_writer(&mut output, &sequence_number).ok()?;
    output.extend_from_slice(b"}\n\n");
    if deferred_done {
        output.extend_from_slice(b"data: [DONE]\n\n");
    }
    Some(Bytes::from(output))
}

/// Emit the held terminal event only when the current IRR step is terminal.
fn finalize_logical_stream(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(mut parser_state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };

    // Preserve any non-terminal logical events `process_chunk` already emitted
    // for this final chunk, then append synthesized local-tool events and the
    // deferred terminal. A transport that reassembles the whole stream before
    // releasing it (e.g. `responses_to_chat_completions`) delivers the
    // created/delta events and deferred terminal together in the end-of-stream
    // chunk; starting from an empty buffer here would drop those earlier events.
    let mut output = body.take().map_or_else(Vec::new, |bytes| bytes.to_vec());
    // #313 §4.2: drain file_search synthesis before terminal/error finalization,
    // under the precedence policy. A validation failure here calls
    // fs_end_stream_with_error_ctx (site (b), §7.3) so the error branch below is
    // selected and the router does not re-fire.
    local_tools::drain_local_tool_synthesis(ctx, parser_state.output_index_offset, &mut output);
    // #313 P1 (DoS bound): file_search's EOS reconcile (a prior response-phase filter) has
    // already read this round's provider-streamed observation set; clear it unconditionally
    // here — NOT inside drain_local_tool_synthesis, which early-returns on an empty synthesis
    // queue (exactly the all-natives-streamed-live case) — so it cannot accumulate across IRR
    // continuation rounds and bypass the max_state_bytes ceiling. The ids are stale after the
    // round that recorded them, so clearing loses nothing.
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.provider_streamed_terminal_ids.clear();
    }
    // #313 P1: a terminal failure recorded after file_search already published its
    // per-round continuation — our own parse/validation error (`stream_error_code`, e.g.
    // set by validate_stream_end at EOS, which runs AFTER file_search's reconcile) or a
    // flat upstream `error` completion (`stream_completion == "error"`, which sets no
    // error code) — must clear a stale file_search `action="loop"` to the two-key stop,
    // or the error frame is suppressed and another IRR round fires. Scoped to file_search:
    // web_search/mcp own their own stop signalling and are left untouched.
    let file_search_looping = ctx
        .filter_results
        .get("openai_file_search_callout")
        .and_then(|results| results.get("action"))
        == Some("loop");
    let terminal_error = ctx.get_metadata("responses.stream_error_code").is_some()
        || ctx.get_metadata("responses.stream_completion") == Some("error");
    if file_search_looping && terminal_error {
        crate::openai::responses::fs_arm_stream_stop(ctx);
    }
    let continues = logical_stream_continues(ctx); // re-read AFTER drain + arm-stop: a
    // (b)-site failure or the arm-stop above flips file_search action=done.
    if !continues && let Some(mut error) = logical_stream_error(ctx) {
        // #276: surface any locally executed tool items that never reached the
        // client before the stream terminates with an error, so already-executed
        // tool activity is not silently dropped by a resumed-round parse failure.
        flush_local_output_items(ctx, &mut output);
        normalize_logical_payload(ctx, &mut error, parser_state.output_index_offset);
        encode_sse_event("error", &error, &mut output);
    } else if !continues && let Some(mut terminal) = parser_state.deferred_terminal.take() {
        emit_deferred_terminal(ctx, &mut terminal, &parser_state, &mut output);
    }
    *body = (!output.is_empty()).then(|| Bytes::from(output));
    ctx.insert_filter_state(parser_state);
}

/// Emit the deferred terminal snapshot as the logical stream's final event,
/// preceded by any locally generated tool items not yet streamed to the client.
fn emit_deferred_terminal(
    ctx: &mut HttpFilterContext<'_>,
    terminal: &mut DeferredTerminalEvent,
    parser_state: &StreamEventsState,
    output: &mut Vec<u8>,
) {
    // #276: stream any locally generated tool items that never reached the
    // client as incremental events (e.g. an MCP approval request that ends the
    // loop without a resumed round) before the terminal snapshot.
    flush_local_output_items(ctx, output);
    let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
    let (accumulated_output, usage) = canonicalize_logical_response(state);
    if let Some(response) = terminal.payload.get_mut("response").and_then(Value::as_object_mut) {
        response.insert("output".to_owned(), Value::Array(accumulated_output));
        if !usage.is_null() {
            response.insert("usage".to_owned(), usage);
        }
    }
    normalize_logical_payload(ctx, &mut terminal.payload, parser_state.output_index_offset);
    encode_sse_event(&terminal.event_type, &terminal.payload, output);
    if parser_state.deferred_done {
        output.extend_from_slice(b"data: [DONE]\n\n");
    }
}

/// Whether a dispatch filter requested another inference step.
fn logical_stream_continues(ctx: &HttpFilterContext<'_>) -> bool {
    ["openai_mcp_dispatch", "openai_web_search", "openai_file_search_callout"]
        .iter()
        .any(|filter| ctx.filter_results.get(filter).and_then(|results| results.get("action")) == Some("loop"))
}

/// Return a locally generated terminal error for an already-committed stream.
fn logical_stream_error(ctx: &HttpFilterContext<'_>) -> Option<Value> {
    let code = ctx.get_metadata("responses.stream_error_code")?;
    let message = ctx.get_metadata("responses.stream_error_message")?;
    Some(responses_error_sse_payload(code, message))
}

/// Make the response-store source agree with the logical SSE terminal.
fn canonicalize_logical_response(state: &mut ResponsesState) -> (Vec<Value>, Value) {
    let logical_id = state.logical_stream_response_id.clone();
    let usage = state.usage.clone();
    // Prefer the cross-round accumulator populated by dispatch/loop filters
    // (agentic pipelines). When no such filter ran — a plain one-round logical
    // stream — it stays empty, so fall back to the terminal event's own output
    // rather than clobber it with nothing. Mirrors `finalize_response_body`.
    let output = if state.accumulated_output.is_empty() {
        state.output_items().to_vec()
    } else {
        state.accumulated_output.clone()
    };
    if let Some(response) = state.response_object.as_object_mut() {
        if let Some(logical_id) = logical_id {
            response.insert("id".to_owned(), Value::String(logical_id));
        }
        response.insert("output".to_owned(), Value::Array(output.clone()));
        if !usage.is_null() {
            response.insert("usage".to_owned(), usage.clone());
        }
    }
    (output, usage)
}

/// Check whether the stream has exceeded its wall-clock timeout.
fn check_timeout(state: &StreamEventsState, now: Instant) -> Result<(), SseParseError> {
    let Some(started_at) = state.started_at else {
        return Ok(());
    };
    let elapsed = now.duration_since(started_at);
    if elapsed > state.timeout {
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
    let incomplete = ctx.get_filter_state::<StreamEventsState>().is_some_and(|state| {
        let checked_at = state.completed_at.unwrap_or_else(Instant::now);
        if let Err(e) = check_timeout(state, checked_at) {
            warn!(error = %e, "stream did not terminate cleanly");
            true
        } else if state.completion_state == CompletionState::Open {
            warn!("stream did not terminate cleanly: missing terminal event");
            true
        } else {
            false
        }
    });
    if incomplete {
        ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
        if ctx.get_metadata("responses.stream_error_code").is_none() {
            ctx.set_metadata("responses.stream_error_code", "server_error");
            ctx.set_metadata(
                "responses.stream_error_message",
                "upstream Responses stream did not terminate cleanly",
            );
            ctx.set_metadata("responses.skip_persist", "true");
        }
    }
    debug!("stream_events processing complete");
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
