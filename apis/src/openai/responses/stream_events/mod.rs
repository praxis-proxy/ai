// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Accumulates state from Responses API streaming and JSON responses.
//!
//! Parses backend SSE chunks incrementally using [`SseFrameParser`].
//! Successful non-streaming JSON responses that participate in file search
//! are collected with a bounded [`BodyMode::StreamBuffer`] and parsed at
//! end-of-stream. Both paths update [`ResponsesState`]; only citation-bearing
//! JSON bodies are rewritten.
//!
//! [`SseFrameParser`]: crate::openai::sse::SseFrameParser
//! [`ResponsesState`]: super::state::ResponsesState

pub(crate) mod accumulator;
mod config;

use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES,
    parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    accumulator::{accumulate_event, accumulate_response_object},
    config::StreamEventsConfig,
};
use crate::{
    classifier::is_responses_create,
    is_event_stream_content_type,
    openai::{
        responses::{bounded_json_size, file_search_callout::citations::annotate_response, state::ResponsesState},
        sse::{SseFrameParser, SseParseError, SseParserConfig, responses::ResponsesEvent},
    },
};

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
    /// Cap on accumulated bytes per tool-call argument string.
    max_tool_call_argument_bytes: usize,
}

/// State retained while a bounded non-streaming JSON body is collected.
struct NonStreamingResponseState {
    /// Whether releasing an unparsed body would break public response continuity.
    rewrites_response: bool,
}

/// Accumulates state from Responses API SSE and non-streaming JSON responses.
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
        let cfg: StreamEventsConfig = parse_filter_config("openai_stream_events", config)?;
        cfg.validate()?;
        Ok(Box::new(Self {
            parser_config: cfg.to_parser_config(),
            max_tool_call_argument_bytes: cfg.max_tool_call_argument_bytes(),
        }))
    }

    /// Whether per-request parser state has been installed.
    fn is_armed(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.get_filter_state::<StreamEventsState>().is_some()
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
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let is_responses = is_responses_create(&ctx.request.method, ctx.request.uri.path())
            && ctx.get_metadata("openai_responses_format.format") == Some("openai_responses");
        let is_streaming = ctx.get_metadata("openai_responses_format.stream") == Some("true");

        if is_responses && is_streaming {
            trace!("arming stream_events for streaming Responses API request");
            ctx.insert_filter_state(StreamEventsState {
                frame_parser: SseFrameParser::new(self.parser_config.max_buffer_bytes),
                event_count: 0,
                max_events: self.parser_config.max_events,
                timeout: self.parser_config.timeout,
                started_at: None,
                completed_at: None,
                completion_state: CompletionState::Open,
                tool_call_args: std::collections::HashMap::new(),
                max_tool_call_argument_bytes: self.max_tool_call_argument_bytes,
            });
        } else if is_responses && requires_file_search_accumulation(ctx) {
            force_identity_encoding(ctx);
        }

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(prepare_response_accumulation(ctx))
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.get_filter_state::<NonStreamingResponseState>().is_some() {
            if end_of_stream {
                let action = match process_non_streaming_response(ctx, body) {
                    Ok(()) => FilterAction::Continue,
                    Err(error) => {
                        warn!(%error, "failed to safely rewrite non-streaming Responses body");
                        FilterAction::Reject(super::error::responses_error_rejection(
                            502,
                            "server_error",
                            "openai_stream_events: failed to safely rewrite the file-search response",
                            false,
                        ))
                    },
                };
                ctx.remove_filter_state::<NonStreamingResponseState>();
                return Ok(action);
            }
            return Ok(FilterAction::Continue);
        }

        if !Self::is_armed(ctx) {
            debug!("stream_events not armed, passing through");
            return Ok(FilterAction::Continue);
        }

        process_chunk(ctx, body);

        if end_of_stream {
            validate_stream_end(ctx);
        }

        Ok(FilterAction::Continue)
    }
}

/// Select streaming or bounded JSON response accumulation.
fn prepare_response_accumulation(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
    if !is_responses_create(&ctx.request.method, ctx.request.uri.path())
        || ctx.get_metadata("openai_responses_format.format") != Some("openai_responses")
    {
        return FilterAction::Continue;
    }
    if ctx.get_metadata("openai_responses_format.stream") == Some("true") {
        prepare_streaming_response(ctx);
    } else {
        return prepare_json_response(ctx);
    }
    FilterAction::Continue
}

/// Disarm streaming accumulation when the upstream response is not SSE.
fn prepare_streaming_response(ctx: &mut HttpFilterContext<'_>) {
    if OpenaiStreamEventsFilter::is_armed(ctx) && !is_success_sse_response(ctx) {
        debug!("disarming stream_events: response is not 2xx text/event-stream");
        ctx.remove_filter_state::<StreamEventsState>();
    }
}

/// Arm bounded JSON accumulation for a successful non-streaming response.
#[expect(
    clippy::too_many_lines,
    reason = "response arming checks must preserve rejection order"
)]
fn prepare_json_response(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
    if !ctx
        .response_header
        .as_ref()
        .is_some_and(|response| response.status.is_success())
    {
        return FilterAction::Continue;
    }
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        warn!("not accumulating non-streaming response: ResponsesState is missing");
        return FilterAction::Continue;
    };
    if !requires_file_search_accumulation(ctx) {
        return FilterAction::Continue;
    }
    let rewrites_response = !state.citation_files.is_empty() || state.continuation_output_count != 0;
    if !is_success_json_response(ctx) {
        if rewrites_response {
            return FilterAction::Reject(super::error::responses_error_rejection(
                502,
                "server_error",
                "openai_stream_events: continuation response must be application/json",
                false,
            ));
        }
        return FilterAction::Continue;
    }
    if !has_identity_content_encoding(ctx) {
        return FilterAction::Reject(super::error::responses_error_rejection(
            502,
            "server_error",
            "openai_stream_events: file-search responses must not be content-encoded",
            false,
        ));
    }
    if !content_length_allows_continuation(ctx, state) {
        return FilterAction::Reject(super::error::responses_error_rejection(
            502,
            "server_error",
            "openai_stream_events: merged continuation response exceeds the JSON body byte limit",
            false,
        ));
    }

    ctx.set_response_body_mode(BodyMode::StreamBuffer {
        max_bytes: Some(MAX_JSON_BODY_BYTES),
    });
    if rewrites_response {
        prepare_rewritten_json_headers(ctx);
    }
    ctx.insert_filter_state(NonStreamingResponseState { rewrites_response });
    trace!("armed stream_events for bounded non-streaming JSON accumulation");
    FilterAction::Continue
}

/// Remove representation metadata invalidated by JSON body rewriting.
fn prepare_rewritten_json_headers(ctx: &mut HttpFilterContext<'_>) {
    let Some(response) = &mut ctx.response_header else {
        return;
    };
    response.headers.remove(http::header::CONTENT_LENGTH);
    response.headers.remove(http::header::CONTENT_RANGE);
    response.headers.remove(http::header::ETAG);
    response.headers.remove("content-md5");
    response.headers.remove("digest");
    response.headers.remove("content-digest");
    response.headers.remove("repr-digest");
    ctx.response_headers_modified = true;
}

/// Parse and accumulate one complete non-streaming Responses JSON body.
fn process_non_streaming_response(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
) -> Result<(), FilterError> {
    let rewrites_response = non_streaming_rewrite_required(ctx);
    let Some(bytes) = body.as_ref().filter(|bytes| !bytes.is_empty()) else {
        warn!("non-streaming Responses body was empty");
        ctx.set_metadata("responses.response_parse_error", "true");
        if rewrites_response {
            return Err("openai_stream_events: required response rewrite received an empty body".into());
        }
        return Ok(());
    };

    match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(response) => apply_non_streaming_response(ctx, body, response)?,
        Err(error) => {
            warn!(%error, "failed to parse non-streaming Responses JSON");
            ctx.set_metadata("responses.response_parse_error", "true");
            if rewrites_response {
                return Err("openai_stream_events: required response rewrite received invalid JSON".into());
            }
        },
    }
    Ok(())
}

/// Preserve the header-phase continuity decision through body collection.
fn non_streaming_rewrite_required(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_filter_state::<NonStreamingResponseState>().map_or_else(
        || {
            ctx.extensions
                .get::<ResponsesState>()
                .is_some_and(|state| !state.citation_files.is_empty() || state.continuation_output_count != 0)
        },
        |state| state.rewrites_response,
    )
}

/// Apply output continuity, citation rewriting, and state accumulation.
#[expect(
    clippy::too_many_lines,
    reason = "response rewrite and state commit are one ordered transaction"
)]
fn apply_non_streaming_response(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    mut response: serde_json::Value,
) -> Result<(), FilterError> {
    let policy_modified = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(|state| restore_original_response_policy(&mut response, state));
    if let Some(state) = ctx.extensions.get::<ResponsesState>() {
        let prior = continuation_output_prefix(state);
        if !merged_response_fits(&response, prior, MAX_JSON_BODY_BYTES) {
            return Err("openai_stream_events: merged continuation response exceeds the JSON body byte limit".into());
        }
    }
    let continuity_modified = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(|state| prepend_continuation_output(&mut response, continuation_output_prefix(state)));
    let citations_modified = if let Some(state) = ctx.extensions.get::<ResponsesState>()
        && !state.citation_files.is_empty()
    {
        annotate_response(&mut response, &state.citation_files)
            .map_err(|error| -> FilterError { format!("openai_stream_events: {error}").into() })?
    } else {
        false
    };
    let usage_modified = accumulate_response_object(ctx, response, None);

    if !policy_modified && !continuity_modified && !citations_modified && !usage_modified {
        return Ok(());
    }
    let response = ctx
        .extensions
        .get::<ResponsesState>()
        .map(|state| &state.response_object)
        .ok_or_else(|| -> FilterError { "openai_stream_events: ResponsesState disappeared".into() })?;
    let serialized = serde_json::to_vec(response)
        .map_err(|error| -> FilterError { format!("openai_stream_events: {error}").into() })?;
    if serialized.len() > MAX_JSON_BODY_BYTES {
        return Err(
            format!("openai_stream_events: rewritten response exceeds {MAX_JSON_BODY_BYTES} byte limit").into(),
        );
    }
    *body = Some(Bytes::from(serialized));
    Ok(())
}

/// Restore client-declared policy fields after an internal continuation.
fn restore_original_response_policy(response: &mut serde_json::Value, state: &ResponsesState) -> bool {
    if state.iteration == 0 {
        return false;
    }
    let Some(response) = response.as_object_mut() else {
        return false;
    };
    let mut modified = false;
    for field in ["tool_choice", "max_tool_calls"] {
        let original = if field == "tool_choice" {
            Some(&state.tool_choice)
        } else {
            state.request_body.get(field)
        };
        let Some(original) = original else {
            continue;
        };
        if response.get(field) != Some(original) {
            response.insert(field.to_owned(), original.clone());
            modified = true;
        }
    }
    modified
}

/// Prepend prior continuation output unless it is already the response prefix.
fn prepend_continuation_output(response: &mut serde_json::Value, prior: &[serde_json::Value]) -> bool {
    if prior.is_empty() {
        return false;
    }
    let Some(output) = response.get_mut("output").and_then(serde_json::Value::as_array_mut) else {
        return false;
    };
    if output.starts_with(prior) {
        return false;
    }

    let current = std::mem::take(output);
    output.reserve(prior.len().saturating_add(current.len()));
    output.extend_from_slice(prior);
    output.extend(current);
    true
}

/// Borrow the output prefix retained for the current continuation.
fn continuation_output_prefix(state: &ResponsesState) -> &[serde_json::Value] {
    let output = state.output_items();
    let prior_count = state.continuation_output_count.min(output.len());
    output.get(..prior_count).unwrap_or_default()
}

/// Reject a known oversized merge during the response-header phase.
fn content_length_allows_continuation(ctx: &HttpFilterContext<'_>, state: &ResponsesState) -> bool {
    let prior = continuation_output_prefix(state);
    if prior.is_empty() {
        return true;
    }
    let Some(content_length) = ctx
        .response_header
        .as_ref()
        .and_then(|response| response.headers.get(http::header::CONTENT_LENGTH))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return true;
    };
    let Some(prior_bytes) = bounded_json_size(prior, MAX_JSON_BODY_BYTES).ok().flatten() else {
        return false;
    };
    let added_bytes = prior_bytes.saturating_sub(1);
    content_length
        .checked_add(added_bytes)
        .is_some_and(|merged| merged <= MAX_JSON_BODY_BYTES)
}

/// Calculate the compact merged body size before cloning prior output items.
fn merged_response_fits(response: &serde_json::Value, prior: &[serde_json::Value], max_bytes: usize) -> bool {
    let Some(response_bytes) = bounded_json_size(response, max_bytes).ok().flatten() else {
        return false;
    };
    if prior.is_empty() {
        return true;
    }
    let Some(current) = response.get("output").and_then(serde_json::Value::as_array) else {
        return false;
    };
    if current.starts_with(prior) {
        return true;
    }
    let Some(prior_bytes) = bounded_json_size(prior, max_bytes).ok().flatten() else {
        return false;
    };
    let separator = usize::from(!current.is_empty());
    response_bytes
        .checked_add(prior_bytes.saturating_sub(2))
        .and_then(|bytes| bytes.checked_add(separator))
        .is_some_and(|bytes| bytes <= max_bytes)
}

/// Return whether this request needs non-streaming response state for file search.
fn requires_file_search_accumulation(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
        !state.citation_files.is_empty()
            || state.continuation_output_count != 0
            || state
                .tools
                .iter()
                .any(|tool| tool.get("type").and_then(serde_json::Value::as_str) == Some("file_search"))
    })
}

/// Force an unencoded upstream response so bounded JSON accumulation is possible.
fn force_identity_encoding(ctx: &mut HttpFilterContext<'_>) {
    ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
    ctx.request_headers_to_set.push((
        http::header::ACCEPT_ENCODING,
        http::HeaderValue::from_static("identity"),
    ));
}

/// Return whether the backend response is unencoded or explicitly identity encoded.
fn has_identity_content_encoding(ctx: &HttpFilterContext<'_>) -> bool {
    let Some(response) = ctx.response_header.as_ref() else {
        return true;
    };
    response
        .headers
        .get_all(http::header::CONTENT_ENCODING)
        .iter()
        .all(|value| {
            value.to_str().is_ok_and(|encodings| {
                encodings.split(',').all(|encoding| {
                    let encoding = encoding.trim();
                    !encoding.is_empty() && encoding.eq_ignore_ascii_case("identity")
                })
            })
        })
}

/// Parse SSE frames and accumulate state without modifying the body.
fn process_chunk(ctx: &mut HttpFilterContext<'_>, body: &Option<Bytes>) {
    let Some(bytes) = body.as_ref() else {
        return;
    };

    let Some(mut state) = ctx.remove_filter_state::<StreamEventsState>() else {
        return;
    };

    let now = Instant::now();
    state.started_at.get_or_insert(now);

    if let Err(e) = parse_and_accumulate(&mut state, ctx, bytes, now) {
        warn!(error = %e, "SSE parse error in stream_events");
        ctx.set_metadata("responses.stream_parse_error", "true".to_owned());
    }

    ctx.insert_filter_state(state);
}

/// Parse frames from raw bytes and accumulate events.
fn parse_and_accumulate(
    state: &mut StreamEventsState,
    ctx: &mut HttpFilterContext<'_>,
    bytes: &Bytes,
    now: Instant,
) -> Result<(), SseParseError> {
    check_timeout(state, now)?;

    let frames = state.frame_parser.parse_chunk_with_counted_event_limit(
        bytes,
        state.event_count,
        state.max_events,
        |frame| frame.data != b"[DONE]",
    )?;

    for frame in &frames {
        if frame.data == b"[DONE]" {
            continue;
        }

        state.event_count += 1;
        let event = ResponsesEvent::from_frame(frame)?;
        record_completion(state, &event, now)?;
        accumulate_event(ctx, state, &event);
    }

    Ok(())
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
    if let Some(state) = ctx.get_filter_state::<StreamEventsState>() {
        let checked_at = state.completed_at.unwrap_or_else(Instant::now);
        if let Err(e) = check_timeout(state, checked_at) {
            warn!(error = %e, "stream did not terminate cleanly");
            ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
        } else if state.completion_state == CompletionState::Open {
            warn!("stream did not terminate cleanly: missing terminal event");
            ctx.set_metadata("responses.stream_incomplete", "true".to_owned());
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

/// Whether the response is a successful `application/json` response.
fn is_success_json_response(ctx: &HttpFilterContext<'_>) -> bool {
    let Some(resp) = ctx.response_header.as_ref() else {
        return false;
    };

    resp.status.is_success()
        && resp
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|content_type| {
                content_type
                    .split(';')
                    .next()
                    .is_some_and(|media| media.trim().eq_ignore_ascii_case("application/json"))
            })
}

#[cfg(test)]
mod tests;
