// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-step Messages streaming and the handoff to the next MCP inference round.
//!
//! The retained-state probes are migration debt for the router lifecycle in
//! <https://github.com/praxis-proxy/praxis/issues/984>. Replace the empty
//! emissions and snapshot swaps when core exposes a retained-state check.

use bytes::Bytes;
use http::header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, ETAG};
use praxis_filter::{FilterAction, FilterError, HttpFilterContext, IterationState, Rejection, StreamTerminationCause};
use serde_json::Value;

use super::{ACTION_DONE, ACTION_LOOP, ConnectorState, McpToolIndex, response, router_can_reenter, set_action};
use crate::{
    anthropic::{
        tool_stream::{BlockPresentation, MessagesStream, StreamError},
        wire,
    },
    is_event_stream_content_type,
};

/// Parser and client output for the active filter instance. IRR creates fresh
/// filter state for each step, so only a completed MCP round crosses via extensions.
pub(super) struct StreamRound {
    /// Across-round usage, block numbering, and current Messages parser.
    pub(super) messages: MessagesStream,
    /// Completed MCP results to send before the first bytes of the next round.
    pub(super) results: Vec<u8>,
    /// Bytes observed on the Messages side of this inference round.
    received: usize,
    /// The backend returned a successful identity-encoded SSE response.
    transformable: bool,
    /// Whether HTTP 200 SSE headers have opened the logical response.
    committed: bool,
    /// Header snapshot IRR charges when the currently open round completes.
    response: Option<praxis_filter::SubResponse>,
}

/// Output returned once core cancels the upstream and runs the completion hook.
struct TerminalOutput(Bytes);

/// A validated `message_stop` already published the round decision and output.
struct RoundEnded;

/// Bounded model history and optional mixed-round terminal held by the router.
struct CompletedRound {
    /// Model-facing assistant turn for dispatch and replay.
    message: Bytes,
    /// Final delta and stop to send after results if client tools are also present.
    terminal: Option<Bytes>,
}

impl CompletedRound {
    /// Charge replay and mixed-round terminal bytes to IRR until re-entry.
    fn retain(self, ctx: &mut HttpFilterContext<'_>) -> Result<(), FilterError> {
        let iteration = ctx
            .extensions
            .get_mut::<IterationState>()
            .ok_or_else(|| FilterError::from("streamed MCP continuation requires iterative_request_router"))?;
        iteration
            .accumulator
            .insert(super::ROUND_ACCUMULATOR_KEY.to_owned(), self.message);
        if let Some(terminal) = self.terminal {
            iteration.accumulator.insert(super::TERMINAL_KEY.to_owned(), terminal);
        }
        Ok(())
    }
}

impl StreamRound {
    /// Start the first inference round.
    pub(super) fn new(limit: usize) -> Self {
        let mut messages = MessagesStream::new(limit, limit);
        messages.set_round_limit(limit);
        Self {
            messages,
            results: Vec::new(),
            received: 0,
            transformable: true,
            committed: false,
            response: None,
        }
    }

    /// Release prior round buffers and keep the logical client lifecycle.
    pub(super) fn begin_round(&mut self) {
        self.messages.begin_round();
        self.received = 0;
        self.transformable = true;
    }

    /// Deliver visible frames immediately. A completed managed round returns
    /// its model-facing message for dispatch in the next request hook.
    fn consume(
        &mut self,
        connector: &ConnectorState,
        chunk: &[u8],
        eof: bool,
        output: &mut Vec<u8>,
    ) -> Result<Option<CompletedRound>, &'static str> {
        self.received = self.received.saturating_add(chunk.len());
        if self.received > self.messages.body_limit() {
            return Err("MCP connector upstream response exceeds the configured size limit");
        }

        let index = McpToolIndex::Resolved(&connector.tool_map);
        self.messages
            .write_chunk(chunk, eof, output, |block| present_block(block, connector, &index))
            .map_err(|error| stream_error(&error))?;
        if (!eof && !self.messages.received_stop()) || self.messages.has_failed() {
            return Ok(None);
        }

        self.finish_round(connector, &index, output)
    }

    /// Reconstruct the completed round and choose re-entry or a client stop.
    fn finish_round(
        &mut self,
        connector: &ConnectorState,
        index: &McpToolIndex<'_>,
        output: &mut Vec<u8>,
    ) -> Result<Option<CompletedRound>, &'static str> {
        self.messages
            .require_complete_lifecycle()
            .map_err(|error| stream_error(&error))?;
        let previous = self
            .messages
            .reconstruct_message()
            .map_err(|error| stream_error(&error))?;
        if response::round_calls_mcp(&previous, index, &connector.admission) {
            let message = Bytes::from(
                self.messages
                    .serialize_round(&previous)
                    .map_err(|error| stream_error(&error))?,
            );
            let mixed = response::has_client_tool_use(response::content_blocks(&previous), index, &connector.admission);
            let terminal = mixed
                .then(|| self.messages.client_tool_terminal())
                .transpose()
                .map_err(|error| stream_error(&error))?
                .map(Bytes::from);
            self.begin_round();
            return Ok(Some(CompletedRound { message, terminal }));
        }
        if praxis_filter::json_size::serialized_len_capped(&previous, self.messages.body_limit()).is_err() {
            return Err("MCP connector upstream response exceeds the configured size limit");
        }
        output.extend(self.messages.terminal().map_err(|error| stream_error(&error))?);
        Ok(None)
    }
}

/// Explain a failed model round without including backend payloads.
fn stream_error(error: &StreamError) -> &'static str {
    match error {
        StreamError::Oversized | StreamError::OversizedPartialEvent => {
            "MCP connector upstream response exceeds the configured size limit"
        },
        StreamError::IncompleteStream => "MCP connector upstream Messages stream ended before completion",
        _ => "MCP connector upstream Messages stream is malformed",
    }
}

/// Finish an already-open Messages response if IRR cancels asynchronous MCP
/// dispatch. A timeout in a request hook returns a local empty 504; the router
/// still drains the chunks emitted by that hook.
pub(super) struct ReentryGuard<'ctx, 'request> {
    /// Context containing the router's output queue.
    pub(super) ctx: &'ctx mut HttpFilterContext<'request>,
    /// Only the instance dispatching a streamed round owns the terminal error.
    armed: bool,
}

impl<'ctx, 'request> ReentryGuard<'ctx, 'request> {
    /// Arm before polling the dispatch future, while its handoff is still present.
    pub(super) fn new(ctx: &'ctx mut HttpFilterContext<'request>) -> Self {
        let armed = ctx
            .extensions
            .get::<ConnectorState>()
            .is_some_and(|state| matches!(state.pending, Some(super::PendingRound::Streaming(_))));
        Self { ctx, armed }
    }

    /// The request hook returned normally and owns its result or rejection.
    pub(super) fn finish(mut self) {
        self.armed = false;
    }
}

impl Drop for ReentryGuard<'_, '_> {
    fn drop(&mut self) {
        if self.armed {
            let mut error = Vec::new();
            append_error(&mut error, "MCP tool execution was interrupted before completion");
            discard_handoff(self.ctx);
            drop(self.ctx.emit_stream_chunk(Bytes::from(error)));
        }
    }
}

/// Queue the results ahead of the next response and install its per-instance parser.
pub(super) fn resume(ctx: &mut HttpFilterContext<'_>, mut round: Box<StreamRound>, limit: usize) -> FilterAction {
    round.messages.set_round_limit(limit);
    // IRR owns the FIFO and checks the queued output together with the
    // rebuilt request against its retained-state budget.
    if ctx
        .emit_stream_chunk(Bytes::from(std::mem::take(&mut round.results)))
        .is_err()
    {
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
        }
        return rejection(
            ctx,
            Some(round),
            wire::error_rejection(
                502,
                wire::ErrorType::Api,
                "MCP connector results and request exceed the router's retained-state limit",
            ),
        );
    }
    ctx.insert_filter_state(*round);
    FilterAction::Continue
}

/// Resolve block starts against the same index used for dispatch.
fn present_block(block: &Value, connector: &ConnectorState, index: &McpToolIndex<'_>) -> BlockPresentation {
    if block["type"] != "tool_use" {
        return BlockPresentation::Forward;
    }
    let Some(name) = block["name"].as_str() else {
        return BlockPresentation::Forward;
    };
    let Some((server, tool)) = response::mcp_names(name, index, &connector.admission) else {
        return BlockPresentation::Forward;
    };
    BlockPresentation::Replace(serde_json::json!({
        "type": super::request::MCP_TOOL_USE_TYPE, "server_name": server, "name": tool,
        "id": block["id"], "input": block["input"],
    }))
}

/// Accept unencoded Messages events, including an explicit identity encoding.
fn is_messages_stream(header: &praxis_filter::Response) -> bool {
    let sse = header
        .headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(is_event_stream_content_type);
    header.status.is_success() && super::super::has_identity_encoding(&header.headers) && sse
}

/// IRR only supplies upstream headers in this hook.
pub(super) fn on_headers(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
    let committed = ctx
        .get_filter_state::<StreamRound>()
        .is_some_and(|round| round.committed);
    let Some(header) = ctx.response_header.as_deref_mut() else {
        return FilterAction::Continue;
    };
    let transformable = is_messages_stream(header);
    if transformable {
        header.headers.remove(CONTENT_LENGTH);
        header.headers.remove(CONTENT_ENCODING);
        header.headers.remove(ETAG);
    } else if committed {
        let mut error = Vec::new();
        append_error(
            &mut error,
            "MCP connector upstream returned an unusable streaming response",
        );
        return finish(ctx, error);
    } else if header.status.is_success() || !is_json_response(header) {
        return unusable_response(header.status);
    }
    let response = response_snapshot(header);
    if !stage_response(ctx, response) {
        return oversized_headers(ctx, committed);
    }
    if let Some(round) = ctx.get_filter_state_mut::<StreamRound>() {
        round.transformable = transformable;
        round.committed |= transformable;
    }
    FilterAction::Continue
}

/// Whether the first upstream HTTP error has a JSON representation.
fn is_json_response(header: &praxis_filter::Response) -> bool {
    header
        .headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json"))
}

/// The body hook cannot borrow response headers. Share their values until
/// IRR accounts for the current response at completion.
fn response_snapshot(header: &praxis_filter::Response) -> praxis_filter::SubResponse {
    praxis_filter::SubResponse {
        version: http::Version::HTTP_11,
        status: header.status.as_u16(),
        headers: header.headers.clone(),
        body: Bytes::new(),
    }
}

/// Check the current response with the router's own accounting before headers
/// commit. Restore the previous round for other filters until this one finishes.
fn stage_response(ctx: &mut HttpFilterContext<'_>, response: praxis_filter::SubResponse) -> bool {
    let mut snapshot = Some(response);
    if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
        std::mem::swap(&mut state.previous_response, &mut snapshot);
        let fits = ctx.emit_stream_chunk(Bytes::new()).is_ok();
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            std::mem::swap(&mut state.previous_response, &mut snapshot);
        }
        if !fits {
            return false;
        }
    }
    if let Some(round) = ctx.get_filter_state_mut::<StreamRound>() {
        round.response = snapshot;
    }
    true
}

/// Release connector history before rejecting a response the router cannot retain.
fn oversized_headers(ctx: &mut HttpFilterContext<'_>, committed: bool) -> FilterAction {
    const MESSAGE: &str = "MCP connector response headers exceed the router's retained-state limit";
    if committed {
        let mut error = Vec::new();
        append_error(&mut error, MESSAGE);
        return finish(ctx, error);
    }
    if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
        state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
    }
    FilterAction::Reject(wire::error_rejection(502, wire::ErrorType::Api, MESSAGE))
}

/// Produce a Messages HTTP error while the first headers can still change.
fn unusable_response(status: http::StatusCode) -> FilterAction {
    let status = if status.is_success() { 502 } else { status.as_u16() };
    FilterAction::Reject(wire::error_rejection(
        status,
        wire::ErrorType::from_status(status),
        "MCP connector inference did not return a usable Messages response",
    ))
}

/// Consume one chunk. Results precede resumed inference, including a terminal
/// failure from that inference. A first-round HTTP error keeps its upstream body.
#[expect(
    clippy::too_many_lines,
    reason = "the response hook owns transport termination, round handoff and terminal cleanup"
)]
pub(super) fn on_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    eof: bool,
    limit: usize,
) -> Result<FilterAction, FilterError> {
    if ctx.get_filter_state::<RoundEnded>().is_some() {
        *body = None;
        return finish_transport(ctx, eof);
    }
    if eof && let Some(terminal) = ctx.remove_filter_state::<TerminalOutput>() {
        ctx.mark_stream_termination_handled();
        *body = complete_output(ctx, terminal.0);
        set_action(ctx, ACTION_DONE)?;
        return Ok(FilterAction::Continue);
    }
    let Some(mut round) = ctx.remove_filter_state::<StreamRound>() else {
        return Ok(FilterAction::Continue);
    };
    if eof
        && let Some(state) = ctx.extensions.get_mut::<IterationState>()
        && round.response.is_some()
    {
        state.previous_response = round.response.take();
    }
    if !round.transformable {
        return passthrough_error(ctx, body.as_ref(), round, eof, limit);
    }
    let mut output = Vec::new();

    if let Some(cause) = ctx.stream_termination().map(praxis_filter::StreamTermination::cause) {
        let reason = if cause == StreamTerminationCause::DeadlineExceeded {
            "MCP connector inference deadline exceeded"
        } else {
            "MCP connector upstream stream terminated before completion"
        };
        if round.committed || round.messages.has_started() {
            fail(&mut round, &mut output, reason);
        } else {
            output.extend_from_slice(&wire::error_body(wire::ErrorType::Api, reason, None));
        }
        ctx.mark_stream_termination_handled();
    } else if ctx.response_body_bytes > limit as u64 {
        fail(
            &mut round,
            &mut output,
            "MCP connector upstream response exceeds the configured size limit",
        );
    } else if !round.messages.has_failed()
        && let Some(connector) = ctx.extensions.get::<ConnectorState>()
    {
        match round.consume(connector, body.as_deref().unwrap_or_default(), eof, &mut output) {
            Ok(Some(previous)) if router_can_reenter(ctx) => {
                close_protocol_round(ctx, &mut round, eof);
                *body = handoff(ctx, round, previous, output, eof)?;
                return Ok(FilterAction::Continue);
            },
            Ok(Some(_)) => fail(
                &mut round,
                &mut output,
                "MCP connector inference iteration or deadline limit reached",
            ),
            Ok(None) => {},
            Err(message) => {
                append_error(&mut output, message);
                round.messages.fail();
            },
        }
    }
    if !eof && round.messages.has_failed() {
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            state.previous_response = round.response.take();
        }
        ctx.insert_filter_state(TerminalOutput(Bytes::from(output)));
        return Err(FilterError::from("MCP connector closed a failed inference stream"));
    }
    if !eof && round.messages.received_stop() {
        close_protocol_round(ctx, &mut round, false);
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
        }
        *body = (!output.is_empty()).then(|| Bytes::from(output));
        set_action(ctx, ACTION_DONE)?;
        return Ok(FilterAction::Continue);
    }
    if !eof {
        *body = (!output.is_empty()).then(|| Bytes::from(output));
        ctx.insert_filter_state(round);
        return Ok(FilterAction::Continue);
    }
    *body = complete_output(ctx, Bytes::from(output));
    set_action(ctx, ACTION_DONE)?;
    Ok(FilterAction::Continue)
}

/// A completed round tolerates a backend that holds its body open. A later
/// filter can still reject the chunk containing the stop before delivery.
fn finish_transport(ctx: &mut HttpFilterContext<'_>, eof: bool) -> Result<FilterAction, FilterError> {
    if !eof {
        // Stop reading trailing bytes once a ready transport EOF was allowed.
        ctx.cap_stream_deadline(std::time::Instant::now());
        return Ok(FilterAction::Continue);
    }
    ctx.remove_filter_state::<RoundEnded>();
    let cause = ctx.stream_termination().map(praxis_filter::StreamTermination::cause);
    if matches!(
        cause,
        Some(StreamTerminationCause::Filter | StreamTerminationCause::ResponseTooLarge)
    ) {
        if let Some(connector) = ctx.extensions.get_mut::<ConnectorState>() {
            connector.pending = None;
        }
        discard_handoff(ctx);
        set_action(ctx, ACTION_DONE)?;
    } else {
        ctx.mark_stream_termination_handled();
    }
    Ok(FilterAction::Continue)
}

/// Release retained replay data when no further model round can use it.
fn discard_handoff(ctx: &mut HttpFilterContext<'_>) {
    if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
        state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
        state.accumulator.remove(super::ROUND_ACCUMULATOR_KEY);
        state.accumulator.remove(super::TERMINAL_KEY);
    }
}

/// IRR transitions after its transport completion hook. Once a validated
/// `message_stop` arrives, allow a ready HTTP EOF to recycle the connection
/// without waiting on an upstream that holds the terminated message open.
fn close_protocol_round(ctx: &mut HttpFilterContext<'_>, round: &mut StreamRound, eof: bool) {
    if !eof {
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            state.previous_response = round.response.take();
        }
        ctx.insert_filter_state(RoundEnded);
        ctx.cap_stream_read_timeout(std::time::Duration::ZERO);
    }
}

/// A first-round HTTP error keeps its representation. If forwarding fails
/// after headers commit, let the transport signal truncation to the client.
fn passthrough_error(
    ctx: &mut HttpFilterContext<'_>,
    body: Option<&Bytes>,
    mut round: StreamRound,
    eof: bool,
    limit: usize,
) -> Result<FilterAction, FilterError> {
    if eof {
        if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
            state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
        }
        set_action(ctx, ACTION_DONE)?;
        return Ok(FilterAction::Continue);
    }
    round.received = round.received.saturating_add(body.map_or(0, Bytes::len));
    let exceeded = ctx.response_body_bytes > limit as u64 || round.received > limit;
    ctx.insert_filter_state(round);
    if exceeded {
        return Err(FilterError::from(
            "MCP connector upstream HTTP error exceeds the configured size limit",
        ));
    }
    Ok(FilterAction::Continue)
}

/// Drop replay history and validate terminal frames with the current headers.
fn complete_output(ctx: &mut HttpFilterContext<'_>, mut output: Bytes) -> Option<Bytes> {
    let Some(state) = ctx.extensions.get_mut::<IterationState>() else {
        return (!output.is_empty()).then_some(output);
    };
    state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
    if output.is_empty() {
        return None;
    }
    if fits_completion(ctx, &mut output) {
        return Some(output);
    }
    let mut error = Vec::new();
    append_error(
        &mut error,
        "MCP connector terminal output exceeds the router's retained-state limit",
    );
    Some(Bytes::from(error))
}

/// IRR retains completion output alongside the header snapshot. Temporarily
/// charge the bytes to that snapshot, then return them through the body hook
/// so remaining response filters observe the same frames at any chunk split.
fn fits_completion(ctx: &mut HttpFilterContext<'_>, output: &mut Bytes) -> bool {
    let snapshot = |state: &mut IterationState, bytes: &mut Bytes| {
        if let Some(response) = &mut state.previous_response {
            std::mem::swap(&mut response.body, bytes);
        }
    };
    let Some(state) = ctx.extensions.get_mut::<IterationState>() else {
        return true;
    };
    snapshot(state, output);
    let fits = ctx.emit_stream_chunk(Bytes::new()).is_ok();
    if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
        snapshot(state, output);
    }
    fits
}

/// Retain the reconstructed message and flush the final visible frames as one
/// bounded router operation. On refusal, release the handoff before returning
/// a Messages error through the committed response.
fn handoff(
    ctx: &mut HttpFilterContext<'_>,
    round: StreamRound,
    previous: CompletedRound,
    output: Vec<u8>,
    eof: bool,
) -> Result<Option<Bytes>, FilterError> {
    previous.retain(ctx)?;
    let mut output = Bytes::from(output);
    let fits = if eof {
        fits_completion(ctx, &mut output)
    } else {
        // Model frames from a live body are delivered before IRR retains the round.
        ctx.emit_stream_chunk(Bytes::new()).is_ok()
    };
    if !fits {
        discard_handoff(ctx);
        let mut error = Vec::new();
        append_error(
            &mut error,
            "MCP connector round exceeds the router's retained-state limit",
        );
        set_action(ctx, ACTION_DONE)?;
        return Ok(Some(Bytes::from(error)));
    }
    if let Some(state) = ctx.extensions.get_mut::<ConnectorState>() {
        state.pending = Some(super::PendingRound::Streaming(Box::new(round)));
    }
    set_action(ctx, ACTION_LOOP)?;
    Ok((!output.is_empty()).then_some(output))
}

/// Emit a single terminal error and release the parser.
fn fail(round: &mut StreamRound, output: &mut Vec<u8>, message: &str) {
    if !round.messages.has_failed() {
        append_error(output, message);
    }
    round.messages.fail();
}

/// Encode an inference failure using the Anthropic error envelope.
fn append_error(output: &mut Vec<u8>, message: &str) {
    append_error_body(output, &wire::error_body(wire::ErrorType::Api, message, None));
}

/// Frame an already serialized, compact Anthropic error envelope.
fn append_error_body(output: &mut Vec<u8>, body: &[u8]) {
    output.extend_from_slice(b"event: error\ndata: ");
    output.extend_from_slice(body);
    output.extend_from_slice(b"\n\n");
}

/// A request-phase rejection on re-entry is drained verbatim by the IRR into
/// the already committed SSE response.
pub(super) fn rejection(
    ctx: &mut HttpFilterContext<'_>,
    round: Option<Box<StreamRound>>,
    rejection: Rejection,
) -> FilterAction {
    let Some(mut round) = round else {
        return FilterAction::Reject(rejection);
    };
    if let Some(body) = rejection.body {
        append_error_body(&mut round.results, &body);
    }
    finish(ctx, round.results)
}

/// Finish an open SSE lifecycle from the IRR request hook.
pub(super) fn finish(ctx: &mut HttpFilterContext<'_>, bytes: Vec<u8>) -> FilterAction {
    if let Some(state) = ctx.extensions.get_mut::<IterationState>() {
        state.accumulator.remove(super::REQUEST_ACCUMULATOR_KEY);
    }
    // A local rejection body is also retained as previous_response by IRR.
    // Queue the terminal exactly once and close the step with an empty body.
    if ctx.emit_stream_chunk(Bytes::from(bytes)).is_ok() {
        return FilterAction::Reject(Rejection::status(200));
    }
    let mut error = Vec::new();
    append_error(
        &mut error,
        "MCP connector terminal output exceeds the router's retained-state limit",
    );
    FilterAction::Reject(Rejection::status(502).with_body(Bytes::from(error)))
}
