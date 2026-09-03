// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Accumulates state from native Responses API SSE event streams.
//!
//! Parses backend SSE chunks using [`SseFrameParser`], dispatches
//! typed events to update [`ResponsesState`] in request extensions.
//! With `logical_stream: true`, successive IRR inference streams are
//! normalized into one downstream Responses lifecycle.
//!
//! [`SseFrameParser`]: crate::openai::sse::SseFrameParser
//! [`ResponsesState`]: super::state::ResponsesState

pub(crate) mod accumulator;
mod config;

use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, SubRequestResponseMode,
    parse_filter_config,
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
        responses::{error::responses_error_sse_payload, state::ResponsesState},
        sse::{SseFrameParser, SseParseError, SseParserConfig, responses::ResponsesEvent},
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
    /// Whether this parser normalizes an IRR multi-round logical stream.
    logical_stream: bool,
    /// Inference iteration number for lifecycle suppression and index offsets.
    iteration: u32,
    /// Output index offset contributed by preceding inference/tool rounds.
    output_index_offset: u64,
    /// Terminal event withheld until completion filters publish a transition.
    deferred_terminal: Option<DeferredTerminalEvent>,
    /// Whether a provider `[DONE]` sentinel should follow the logical terminal.
    deferred_done: bool,
}

/// Accumulates state from native Responses API SSE event streams.
///
/// # YAML
///
/// ```yaml
/// filter: openai_stream_events
/// # All fields optional:
/// # logical_stream: false
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
    /// Normalize successive IRR turns into one logical Responses stream.
    logical_stream: bool,
}

impl OpenaiStreamEventsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: StreamEventsConfig = parse_filter_config("openai_stream_events", config)?;
        cfg.validate()?;
        Ok(Box::new(Self {
            parser_config: cfg.to_parser_config(),
            max_tool_call_argument_bytes: cfg.max_tool_call_argument_bytes(),
            logical_stream: cfg.logical_stream,
        }))
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
            logical_stream: self.logical_stream,
            iteration,
            output_index_offset,
            deferred_terminal: None,
            deferred_done: false,
        });
        ctx.set_metadata("responses.stream_completion", "open");
        // Publish a per-round marker that `openai_agentic_loop` reads (and then
        // consumes) to confirm this typed-streaming round can surface
        // loop-terminal errors through `finalize_logical_stream`. Only meaningful
        // when `logical_stream` is enabled; refreshed every round because the
        // agentic loop overwrites it after each check.
        if self.logical_stream {
            ctx.set_metadata("responses.logical_stream", "true");
        }
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
        if self.logical_stream {
            BodyAccess::ReadWrite
        } else {
            BodyAccess::ReadOnly
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let typed_streaming = ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming;
        let is_responses = is_responses_create(&ctx.request.method, ctx.request.uri.path())
            && (typed_streaming || ctx.get_metadata("openai_responses_format.format") == Some("openai_responses"));
        let is_streaming = typed_streaming || ctx.get_metadata("openai_responses_format.stream") == Some("true");

        if is_responses && is_streaming {
            trace!("arming stream_events for streaming Responses API request");
            self.arm(ctx);
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
            handle_parse_error(ctx, body, state, &error);
            return;
        },
    };
    let completion = match state.completion_state {
        CompletionState::Open => "open",
        CompletionState::TerminalLifecycle => "terminal",
        CompletionState::Error => "error",
    };
    ctx.set_metadata("responses.stream_completion", completion);
    if state.logical_stream {
        *body = parsed;
    }
}

/// Record a parse failure and suppress unnormalized logical-stream bytes.
fn handle_parse_error(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    state: &StreamEventsState,
    error: &SseParseError,
) {
    warn!(%error, "SSE parse error in stream_events");
    ctx.set_metadata("responses.stream_parse_error", "true".to_owned());
    if state.logical_stream {
        ctx.set_metadata("responses.stream_error_code", "server_error");
        ctx.set_metadata(
            "responses.stream_error_message",
            "upstream Responses stream could not be parsed",
        );
        ctx.set_metadata("responses.skip_persist", "true");
        *body = None;
    }
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

    let mut logical_output = Vec::new();
    for frame in &frames {
        if frame.data == b"[DONE]" {
            if state.logical_stream {
                state.deferred_done = true;
            }
            continue;
        }

        state.event_count += 1;
        let event = ResponsesEvent::from_frame(frame)?;
        record_completion(state, &event, now)?;
        accumulate_event(ctx, state, &event);
        if state.logical_stream {
            append_logical_event(state, ctx, &event, &mut logical_output);
        }
    }

    Ok(state
        .logical_stream
        .then(|| Bytes::from(logical_output))
        .filter(|bytes| !bytes.is_empty()))
}

/// Append one provider event to the logical stream or defer/suppress it.
fn append_logical_event(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    event: &ResponsesEvent,
    output: &mut Vec<u8>,
) {
    if event.is_terminal() {
        state.deferred_terminal = Some(DeferredTerminalEvent {
            event_type: event.event_type().to_owned(),
            payload: event.payload().clone(),
        });
        return;
    }
    if state.iteration > 0
        && matches!(
            event,
            ResponsesEvent::ResponseCreated(_)
                | ResponsesEvent::ResponseQueued(_)
                | ResponsesEvent::ResponseInProgress(_)
        )
    {
        return;
    }

    let mut payload = event.payload().clone();
    normalize_logical_payload(ctx, &mut payload, state.output_index_offset);
    encode_sse_event(event.event_type(), &payload, output);
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
        if object.contains_key("sequence_number") {
            object.insert(
                "sequence_number".to_owned(),
                Value::Number(serde_json::Number::from(state.logical_stream_sequence)),
            );
        }
    }
    state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);
}

/// Encode one canonical single-line SSE event.
fn encode_sse_event(event_type: &str, payload: &Value, output: &mut Vec<u8>) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    output.extend_from_slice(payload.to_string().as_bytes());
    output.extend_from_slice(b"\n\n");
}

/// Emit the held terminal event only when the current IRR step is terminal.
fn finalize_logical_stream(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>) {
    let Some(mut parser_state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };
    if !parser_state.logical_stream {
        ctx.insert_filter_state(parser_state);
        return;
    }

    let continues = logical_stream_continues(ctx);
    let mut output = Vec::new();
    if !continues && let Some(mut error) = logical_stream_error(ctx) {
        normalize_logical_payload(ctx, &mut error, parser_state.output_index_offset);
        encode_sse_event("error", &error, &mut output);
    } else if !continues && let Some(mut terminal) = parser_state.deferred_terminal.take() {
        let state = ctx.extensions.get_or_insert_with(ResponsesState::default);
        let (accumulated_output, usage) = canonicalize_logical_response(state);
        if let Some(response) = terminal.payload.get_mut("response").and_then(Value::as_object_mut) {
            response.insert("output".to_owned(), Value::Array(accumulated_output));
            if !usage.is_null() {
                response.insert("usage".to_owned(), usage);
            }
        }
        normalize_logical_payload(ctx, &mut terminal.payload, parser_state.output_index_offset);
        encode_sse_event(&terminal.event_type, &terminal.payload, &mut output);
        if parser_state.deferred_done {
            output.extend_from_slice(b"data: [DONE]\n\n");
        }
    }
    *body = (!output.is_empty()).then(|| Bytes::from(output));
    ctx.insert_filter_state(parser_state);
}

/// Whether a dispatch filter requested another inference step.
fn logical_stream_continues(ctx: &HttpFilterContext<'_>) -> bool {
    ["openai_mcp_dispatch", "openai_web_search"]
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
    let accumulated_output = state.accumulated_output.clone();
    let usage = state.usage.clone();
    if let Some(response) = state.response_object.as_object_mut() {
        if let Some(logical_id) = logical_id {
            response.insert("id".to_owned(), Value::String(logical_id));
        }
        response.insert("output".to_owned(), Value::Array(accumulated_output.clone()));
        if !usage.is_null() {
            response.insert("usage".to_owned(), usage.clone());
        }
    }
    (accumulated_output, usage)
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
    let incomplete_logical_stream = ctx.get_filter_state::<StreamEventsState>().and_then(|state| {
        let checked_at = state.completed_at.unwrap_or_else(Instant::now);
        if let Err(e) = check_timeout(state, checked_at) {
            warn!(error = %e, "stream did not terminate cleanly");
            Some(state.logical_stream)
        } else if state.completion_state == CompletionState::Open {
            warn!("stream did not terminate cleanly: missing terminal event");
            Some(state.logical_stream)
        } else {
            None
        }
    });
    if let Some(logical_stream) = incomplete_logical_stream {
        ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
        if logical_stream && ctx.get_metadata("responses.stream_error_code").is_none() {
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


#[cfg(test)]
mod tests;
