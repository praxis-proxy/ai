// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Web search policy for the shared Messages stream lifecycle.


use std::borrow::Cow;

pub(super) use super::super::tool_stream::StreamError;
use super::super::tool_stream::{BlockPresentation, MessagesStream};


/// The loop transition selected after finishing one backend round.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum RoundAction {
    /// Re-enter inference: the round produced a managed `WebSearch` call.
    Loop,
    /// Return the terminal response: the round produced the final answer.
    Done,
}

/// The result of finishing one backend round.
#[derive(Debug)]
pub(super) struct FinishOutcome {
    /// Whether the loop re-enters inference or returns the terminal response.
    pub(super) action: RoundAction,
    /// Terminal `message_delta` / `message_stop` bytes, empty unless done.
    pub(super) terminal: Vec<u8>,
}

/// Web search classification and cross-round replay.
pub(super) struct LogicalStream {
    /// Shared parser, reconstructed blocks and cross-round usage.
    messages: MessagesStream,
    /// Configured limit for a managed search query.
    max_query_bytes: usize,
    /// Completed managed round awaiting IRR re-entry.
    reconstructed: Option<Vec<u8>>,
    /// Whether this round hid a model-facing web search block.
    suppressed: bool,
}

impl LogicalStream {
    /// Create a bounded web search stream.
    pub(super) fn new(max_body_bytes: usize, max_partial_event_bytes: usize, max_query_bytes: usize) -> Self {
        Self {
            messages: MessagesStream::new(max_body_bytes, max_partial_event_bytes),
            max_query_bytes,
            reconstructed: None,
            suppressed: false,
        }
    }

    /// Whether a terminal error has ended the lifecycle.
    pub(super) fn has_failed(&self) -> bool {
        self.messages.has_failed()
    }

    /// Whether the first `message_start` has reached the client.
    pub(super) fn has_started(&self) -> bool {
        self.messages.has_started()
    }

    /// Stop forwarding after an upstream or transport error.
    pub(super) fn fail(&mut self) {
        self.messages.fail();
    }

    /// Move the classified assistant turn into the next model request.
    pub(super) fn take_reconstructed(&mut self) -> Option<Vec<u8>> {
        self.reconstructed.take()
    }

    /// Reset per-round reconstruction while preserving the client lifecycle.
    pub(super) fn begin_round(&mut self) {
        self.messages.begin_round();
        self.suppressed = false;
    }

    /// Forward Messages events incrementally, hiding the managed search call.
    pub(super) fn on_chunk(&mut self, chunk: &[u8], end_of_stream: bool) -> Result<Vec<u8>, StreamError> {
        let mut output = Vec::new();
        self.messages.write_chunk(chunk, end_of_stream, &mut output, |block| {
            let suppress = block["type"] == "tool_use" && block["name"] == "WebSearch";
            self.suppressed |= suppress;
            if suppress {
                BlockPresentation::Suppress
            } else {
                BlockPresentation::Forward
            }
        })?;
        Ok(output)
    }

    /// Classify the reconstructed message and close or continue the stream.
    pub(super) fn finish_round(&mut self) -> Result<FinishOutcome, StreamError> {
        if self.has_failed() {
            return Ok(FinishOutcome {
                action: RoundAction::Done,
                terminal: Vec::new(),
            });
        }
        let message = self.messages.reconstruct_message()?;
        let bytes = self.messages.serialize_round(&message)?;
        match super::classify_response(&bytes, self.max_query_bytes) {
            super::ResponseDecision::Managed(_) => {
                self.messages.require_complete_lifecycle()?;
                self.messages.begin_round();
                self.reconstructed = Some(bytes);
                Ok(FinishOutcome {
                    action: RoundAction::Loop,
                    terminal: Vec::new(),
                })
            },
            super::ResponseDecision::Done if self.suppressed => Err(StreamError::SuppressionMismatch),
            super::ResponseDecision::Done => Ok(FinishOutcome {
                action: RoundAction::Done,
                terminal: self.messages.terminal()?,
            }),
            super::ResponseDecision::InvalidManagedCall => Err(StreamError::InvalidManagedCall),
            super::ResponseDecision::QueryTooLong => Err(StreamError::QueryTooLong {
                max_bytes: self.max_query_bytes,
            }),
        }
    }
}

/// Encode a fail-closed terminal Anthropic `error` SSE event for a stream error.
///
/// The Anthropic error envelope is carried verbatim from [`super::super::wire`]
/// so no raw upstream bytes leak into the transformed stream.
pub(super) fn error_event_bytes(error: &StreamError) -> Vec<u8> {
    let (error_type, message) = anthropic_error(error);
    let body = super::super::wire::error_body(error_type, message.as_ref(), None);
    let mut out = Vec::with_capacity(body.len() + 20);
    out.extend_from_slice(b"event: error\ndata: ");
    out.extend_from_slice(&body);
    out.extend_from_slice(b"\n\n");
    out
}

/// Map a [`StreamError`] to an Anthropic error type and client-safe message.
#[expect(
    clippy::too_many_lines,
    reason = "each stream failure has an explicit client-safe message"
)]
fn anthropic_error(error: &StreamError) -> (super::super::wire::ErrorType, Cow<'static, str>) {
    match error {
        StreamError::InvalidManagedCall => (
            super::super::wire::ErrorType::InvalidRequest,
            Cow::Borrowed("WebSearch tool use requires a non-empty id and input.query"),
        ),
        StreamError::QueryTooLong { max_bytes } => (
            super::super::wire::ErrorType::InvalidRequest,
            Cow::Owned(format!("WebSearch input.query must not exceed {max_bytes} bytes")),
        ),
        StreamError::IterationLimit => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("web search exceeded the maximum number of search iterations"),
        ),
        StreamError::DeadlineExceeded => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("web search exceeded the configured deadline"),
        ),
        StreamError::ReentryTooLarge => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("Web search request body exceeds a size limit set by the platform administrator."),
        ),
        StreamError::IncompleteStream => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("web search response stream ended before completion"),
        ),
        StreamError::UpstreamTerminated => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("web search stream terminated before completion"),
        ),
        StreamError::MalformedUtf8
        | StreamError::MalformedEvent
        | StreamError::OversizedPartialEvent
        | StreamError::Oversized
        | StreamError::SuppressionMismatch
        | StreamError::UpstreamUnprocessable => (
            super::super::wire::ErrorType::Api,
            Cow::Borrowed("web search stream could not be processed"),
        ),
    }
}


#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    const MAX_PARTIAL: usize = 64 * 1024;
    const MAX_BODY: usize = 1 << 20;
    const MAX_QUERY: usize = crate::web_search::config::MAX_SEARCH_QUERY_BYTES;

    /// Encode one native Messages SSE event as raw bytes.
    fn sse(event_type: &str, data: &Value) -> Vec<u8> {
        format!("event: {event_type}\ndata: {data}\n\n").into_bytes()
    }

    fn message_start(id: &str) -> Vec<u8> {
        sse(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": "test-model",
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 10, "output_tokens": 0}
                }
            }),
        )
    }

    fn text_output(stream: &mut LogicalStream, chunk: &[u8], end_of_stream: bool) -> String {
        String::from_utf8(stream.on_chunk(chunk, end_of_stream).unwrap()).unwrap()
    }

    /// A `message_delta` frame carrying a stop reason and output-token count.
    fn message_delta(stop_reason: &str, output_tokens: u64) -> Vec<u8> {
        sse(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": output_tokens}
            }),
        )
    }

    /// A terminal `message_delta` carrying metadata and usage counters.
    fn message_delta_with_metadata(data: &Value) -> Vec<u8> {
        sse("message_delta", data)
    }

    /// A `message_stop` frame.
    fn message_stop() -> Vec<u8> {
        sse("message_stop", &json!({"type": "message_stop"}))
    }

    /// A text content block's start/delta/stop frames at a backend `index`.
    fn text_block(index: u64, text: &str) -> Vec<u8> {
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
        );
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text}
            }),
        ));
        bytes.extend(sse(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        bytes
    }

    /// A text content block's start/delta frames with no `content_block_stop`,
    /// modeling a block left open by a truncated upstream stream.
    fn open_text_block(index: u64, text: &str) -> Vec<u8> {
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
        );
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text}
            }),
        ));
        bytes
    }

    #[test]
    fn round_zero_message_start_is_forwarded_with_stable_id() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);

        let out = text_output(&mut stream, &message_start("msg_1"), false);

        assert!(
            out.contains("event: message_start"),
            "round 0 must forward message_start"
        );
        assert!(
            out.contains("\"id\":\"msg_1\""),
            "the client-visible id comes from round 0"
        );
    }

    /// A managed `WebSearch` `tool_use` block's start/delta/stop frames.
    fn web_search_block(index: u64, id: &str, query: &str) -> Vec<u8> {
        let partial_json = json!({"query": query}).to_string();
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": id, "name": "WebSearch", "input": {}}
            }),
        );
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json}
            }),
        ));
        bytes.extend(sse(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        bytes
    }

    /// A `thinking` content block's start, `thinking_delta`, `signature_delta`,
    /// and stop frames at a backend `index`.
    fn thinking_block(index: u64, thinking: &str, signature: &str) -> Vec<u8> {
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}
            }),
        );
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": thinking}
            }),
        ));
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": signature}
            }),
        ));
        bytes.extend(sse(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        bytes
    }

    /// A client-owned (non-`WebSearch`) `tool_use` block's start/delta/stop frames.
    fn client_tool_block(index: u64, id: &str) -> Vec<u8> {
        let partial_json = json!({"path": "/tmp"}).to_string();
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": id, "name": "Bash", "input": {}}
            }),
        );
        bytes.extend(sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": partial_json}
            }),
        ));
        bytes.extend(sse(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        bytes
    }

    #[test]
    fn managed_round_exposes_reconstructed_message_for_reentry() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_ws", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        let reconstructed = stream
            .take_reconstructed()
            .expect("managed round retains reconstruction");
        let message: Value = serde_json::from_slice(&reconstructed).unwrap();
        assert_eq!(
            message["content"][0]["name"], "WebSearch",
            "the managed call is reconstructed"
        );
        assert_eq!(
            message["content"][0]["input"]["query"], "rust proxy",
            "the query is reconstructed"
        );
        assert_eq!(
            message["stop_reason"], "tool_use",
            "the managed stop reason is reconstructed"
        );
    }

    #[test]
    fn done_round_does_not_retain_reconstructed_message() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Hello"), false);
        text_output(&mut stream, &message_delta("end_turn", 7), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Done));

        assert!(
            stream.take_reconstructed().is_none(),
            "a terminal round retains nothing for re-entry"
        );
    }

    #[test]
    fn error_event_bytes_encode_fail_closed_error_event() {
        let text = String::from_utf8(error_event_bytes(&StreamError::MalformedUtf8)).unwrap();

        assert!(text.starts_with("event: error\n"), "a terminal error event is emitted");
        assert!(
            text.contains("\"type\":\"error\""),
            "the Anthropic error envelope is carried"
        );
        assert!(
            text.contains("\"type\":\"api_error\""),
            "malformed upstream maps to api_error"
        );
        assert!(text.ends_with("\n\n"), "the SSE event is framed");
    }

    #[test]
    fn error_event_bytes_map_invalid_managed_call_to_invalid_request() {
        let text = String::from_utf8(error_event_bytes(&StreamError::InvalidManagedCall)).unwrap();

        assert!(
            text.contains("\"type\":\"invalid_request_error\""),
            "a malformed managed call is a request error"
        );
        assert!(
            text.contains("non-empty id and input.query"),
            "the client-safe reason is carried"
        );
    }

    #[test]
    fn error_event_bytes_map_query_too_long_to_invalid_request_with_limit() {
        let text = String::from_utf8(error_event_bytes(&StreamError::QueryTooLong { max_bytes: 1_024 })).unwrap();

        assert!(
            text.contains("\"type\":\"invalid_request_error\""),
            "an oversized query is a request error"
        );
        assert!(
            text.contains("must not exceed 1024 bytes"),
            "the client-safe reason names the configured limit"
        );
    }

    #[test]
    fn error_event_bytes_map_reentry_too_large_to_api_error() {
        // An oversized rebuilt re-entry request on a committed streaming response
        // fails closed to one terminal SSE `error` event, mirroring the buffered
        // path's 413 reason under the api_error envelope the open stream requires.
        let text = String::from_utf8(error_event_bytes(&StreamError::ReentryTooLarge)).unwrap();

        assert!(text.starts_with("event: error\n"), "a terminal error event is emitted");
        assert!(
            text.contains("\"type\":\"api_error\""),
            "an oversized re-entry maps to api_error on the committed stream"
        );
        assert!(
            text.contains("size limit set by the platform administrator"),
            "the client-safe reason attributes the size limit to the platform administrator"
        );
        assert!(text.ends_with("\n\n"), "the SSE event is framed");
    }

    #[test]
    fn finish_round_fails_closed_on_oversized_reconstruction() {
        // A tiny body bound forces the reconstructed message over the limit.
        let mut stream = LogicalStream::new(64, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(
            &mut stream,
            &text_block(0, "a long enough answer to exceed the tiny bound"),
            false,
        );
        text_output(&mut stream, &message_delta("end_turn", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::Oversized,
            "an oversized reconstruction fails closed"
        );
    }

    #[test]
    fn finish_round_fails_closed_on_invalid_managed_call() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_ws", ""), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::InvalidManagedCall,
            "an empty query is a malformed managed call"
        );
    }

    #[test]
    fn finish_round_fails_closed_on_oversized_query() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        let long_query = "a".repeat(9 * 1024);
        text_output(&mut stream, &web_search_block(0, "toolu_ws", &long_query), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::QueryTooLong { max_bytes: MAX_QUERY },
            "an oversized query fails closed"
        );
    }

    #[test]
    fn finish_round_fails_closed_when_suppressed_search_is_not_managed() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // A managed WebSearch plus a second (client) tool_use makes the round
        // non-managed, yet the WebSearch block was already suppressed.
        text_output(&mut stream, &web_search_block(0, "toolu_ws", "rust proxy"), false);
        text_output(&mut stream, &client_tool_block(1, "toolu_cli"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(error, StreamError::SuppressionMismatch);
    }

    #[test]
    fn suppresses_managed_web_search_tool_use_block() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);

        let out = text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);

        assert!(
            out.is_empty(),
            "the managed WebSearch tool_use block is fully suppressed"
        );
    }

    #[test]
    fn output_index_is_monotonic_across_rounds() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // Round 0: a text block (output index 0) then a suppressed WebSearch call.
        text_output(&mut stream, &text_block(0, "Searching"), false);
        text_output(&mut stream, &web_search_block(1, "toolu_1", "rust proxy"), false);

        stream.begin_round();
        text_output(&mut stream, &message_start("msg_2"), false);
        let out = text_output(&mut stream, &text_block(0, "Answer"), false);

        assert!(out.contains("\"index\":1"), "round 1 text continues at output index 1");
        assert!(
            !out.contains("\"index\":0"),
            "the suppressed WebSearch must not consume index 0's successor"
        );
    }

    #[test]
    fn forwards_text_content_block_frames() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);

        let out = text_output(&mut stream, &text_block(0, "Hello"), false);

        assert!(out.contains("event: content_block_start"), "text block start forwarded");
        assert!(out.contains("event: content_block_delta"), "text delta forwarded");
        assert!(out.contains("event: content_block_stop"), "text block stop forwarded");
        assert!(out.contains("\"text\":\"Hello\""), "text delta payload forwarded");
        assert!(out.contains("\"index\":0"), "first client block keeps output index 0");
    }

    #[test]
    fn forwards_ping_keepalive() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);

        let out = text_output(&mut stream, &sse("ping", &json!({"type": "ping"})), false);

        assert!(
            out.contains("event: ping"),
            "ping keep-alives are forwarded to the client"
        );
    }

    #[test]
    fn finish_round_emits_terminal_for_done_message() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Hello"), false);
        text_output(&mut stream, &message_delta("end_turn", 7), false);
        text_output(&mut stream, &message_stop(), false);

        let outcome = stream.finish_round().unwrap();

        assert!(
            matches!(outcome.action, RoundAction::Done),
            "a plain text answer finishes the loop"
        );
        let terminal = String::from_utf8(outcome.terminal).unwrap();
        assert!(
            terminal.contains("event: message_delta"),
            "terminal message_delta emitted"
        );
        assert!(
            terminal.contains("\"stop_reason\":\"end_turn\""),
            "final stop reason carried"
        );
        assert!(
            terminal.contains("\"output_tokens\":7"),
            "aggregated output tokens carried"
        );
        assert!(
            terminal.contains("event: message_stop"),
            "terminal message_stop emitted"
        );
    }

    #[test]
    fn finish_round_loops_on_managed_web_search() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);

        let outcome = stream.finish_round().unwrap();

        assert!(
            matches!(outcome.action, RoundAction::Loop),
            "a managed WebSearch call re-enters inference"
        );
        assert!(
            outcome.terminal.is_empty(),
            "no terminal frames are emitted on a managed round"
        );
    }

    #[test]
    fn finish_round_aggregates_output_tokens_across_rounds() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        stream.begin_round();
        text_output(&mut stream, &message_start("msg_2"), false);
        text_output(&mut stream, &text_block(0, "Answer"), false);
        text_output(&mut stream, &message_delta("end_turn", 9), false);
        text_output(&mut stream, &message_stop(), false);

        let terminal = String::from_utf8(stream.finish_round().unwrap().terminal).unwrap();
        assert!(
            terminal.contains("\"output_tokens\":14"),
            "terminal aggregates 5 + 9 output tokens"
        );
    }

    #[test]
    fn finish_round_fails_closed_when_stop_reason_missing() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Hello"), false);
        // The upstream stream ended before delivering a `message_delta`, so no
        // stop reason ever arrived. The terminal must not fabricate a successful
        // `end_turn`; it must fail closed instead.
        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::IncompleteStream,
            "a truncated round fails closed rather than fabricating end_turn"
        );
    }

    #[test]
    fn finish_round_fails_closed_when_message_stop_missing() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Hello"), false);
        // The stop reason arrived, but the upstream body ended before the closing
        // `message_stop`. A clean transport EOF with an incomplete lifecycle must
        // fail closed rather than fabricate a successful terminal.
        text_output(&mut stream, &message_delta("end_turn", 7), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::IncompleteStream,
            "a round without message_stop fails closed rather than fabricating success"
        );
    }

    #[test]
    fn finish_round_fails_closed_when_content_block_left_open() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // A text block starts and streams a delta but never receives its
        // `content_block_stop`: the answer was truncated mid-block.
        text_output(&mut stream, &open_text_block(0, "Half an ans"), false);
        text_output(&mut stream, &message_delta("end_turn", 7), false);
        text_output(&mut stream, &message_stop(), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::IncompleteStream,
            "an unclosed content block fails closed rather than fabricating success"
        );
    }

    #[test]
    fn finish_round_fails_closed_when_managed_round_missing_message_stop() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_ws", "rust proxy"), false);
        // A managed round is also a complete Messages response: without the
        // closing `message_stop` it was truncated and must not re-enter on
        // partial data.
        text_output(&mut stream, &message_delta("tool_use", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(
            error,
            StreamError::IncompleteStream,
            "a truncated managed round fails closed instead of re-entering"
        );
    }

    #[test]
    fn stray_content_block_stop_for_unopened_index_fails_closed() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &open_text_block(0, "Half an ans"), false);
        // A `content_block_stop` for an index that was never started must not
        // silently cancel the genuinely-open block 0: a bare open-block counter
        // would drop to zero and let a truncated round fabricate success. Reject
        // the stray stop so the round fails closed.
        let stray_stop = sse("content_block_stop", &json!({"type": "content_block_stop", "index": 5}));

        let error = stream.on_chunk(&stray_stop, false).unwrap_err();

        assert_eq!(
            error,
            StreamError::MalformedEvent,
            "a stop for an unopened index fails closed instead of masking an open block"
        );
    }

    #[test]
    fn duplicate_content_block_stop_fails_closed() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &open_text_block(0, "first"), false);
        text_output(&mut stream, &open_text_block(1, "second"), false);
        // Close block 0 legitimately, then a duplicate stop for block 0 arrives. A
        // bare counter would decrement again and mask block 1 (still open), letting
        // a truncated round pass the completeness gate. Reject the duplicate stop.
        let stop_zero = sse("content_block_stop", &json!({"type": "content_block_stop", "index": 0}));
        text_output(&mut stream, &stop_zero, false);

        let error = stream.on_chunk(&stop_zero, false).unwrap_err();

        assert_eq!(
            error,
            StreamError::MalformedEvent,
            "a duplicate stop fails closed instead of masking another open block"
        );
    }

    #[test]
    fn finish_round_saturates_output_token_aggregate_on_overflow() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_prior"), false);
        text_output(&mut stream, &web_search_block(0, "toolu_prior", "rust"), false);
        text_output(&mut stream, &message_delta("tool_use", u64::MAX - 1), false);
        text_output(&mut stream, &message_stop(), false);
        stream.finish_round().unwrap();
        stream.begin_round();
        text_output(&mut stream, &message_start("msg_final"), false);
        text_output(&mut stream, &message_delta("end_turn", 5), false);
        text_output(&mut stream, &message_stop(), false);

        let terminal = String::from_utf8(stream.finish_round().unwrap().terminal).unwrap();

        assert!(
            terminal.contains(&format!("\"output_tokens\":{}", u64::MAX)),
            "output-token aggregation saturates instead of overflowing"
        );
    }

    #[test]
    fn on_chunk_is_silent_after_a_failure() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // A `content_block_start` without an index is malformed.
        let malformed = sse(
            "content_block_start",
            &json!({"type": "content_block_start", "content_block": {"type": "text"}}),
        );

        let error = stream.on_chunk(&malformed, false).unwrap_err();
        assert_eq!(error, StreamError::MalformedEvent, "a missing block index fails closed");
        assert!(
            stream.has_failed(),
            "the logical stream is poisoned after a malformed event"
        );

        // A subsequent valid text block yields nothing: the stream stays closed so
        // exactly one terminal error reaches the client.
        let out = stream.on_chunk(&text_block(0, "late"), false).unwrap();
        assert!(
            out.is_empty(),
            "no bytes are forwarded once the stream has failed closed"
        );
    }

    #[test]
    fn later_round_message_start_is_suppressed() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        assert!(
            !text_output(&mut stream, &message_start("msg_1"), false).is_empty(),
            "round 0 forwards message_start"
        );

        stream.begin_round();
        let out = text_output(&mut stream, &message_start("msg_2"), false);

        assert!(
            out.is_empty(),
            "only the first round's message_start reaches the client"
        );
    }

    /// A `message_start` frame whose usage carries input and cache-read tokens.
    fn message_start_with_usage(id: &str, input_tokens: u64, cache_read: u64) -> Vec<u8> {
        sse(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": "test-model",
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "cache_read_input_tokens": cache_read,
                        "output_tokens": 0
                    }
                }
            }),
        )
    }

    /// A text content block delivered as several `text_delta` fragments.
    fn multi_delta_text_block(index: u64, fragments: &[&str]) -> Vec<u8> {
        let mut bytes = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
        );
        for fragment in fragments {
            bytes.extend(sse(
                "content_block_delta",
                &json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": fragment}
                }),
            ));
        }
        bytes.extend(sse(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        ));
        bytes
    }

    /// Parse the JSON `data` payload of the first `event_type` event in `bytes`.
    fn parse_event(bytes: &[u8], event_type: &str) -> Value {
        let text = std::str::from_utf8(bytes).unwrap();
        let marker = format!("event: {event_type}\ndata: ");
        let payload = text
            .split_once(&marker)
            .and_then(|(_, rest)| rest.split_once("\n\n"))
            .map(|(data, _)| data)
            .expect("event present and framed");
        serde_json::from_str(payload).unwrap()
    }

    #[test]
    fn on_block_start_rejects_non_sequential_index() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // A backend-controlled index that skips ahead would force an unbounded
        // sparse allocation; the dense-sequential invariant fails closed instead.
        let sparse = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 5,
                "content_block": {"type": "text", "text": ""}
            }),
        );

        let error = stream.on_chunk(&sparse, false).unwrap_err();

        assert_eq!(
            error,
            StreamError::MalformedEvent,
            "a non-sequential content-block index fails closed"
        );
    }

    #[test]
    fn accumulate_delta_fails_closed_on_cumulative_text_overflow() {
        // A tiny body bound is exceeded by streamed text fragments long before
        // any reconstruction: the cumulative bound must fail closed mid-stream.
        let mut stream = LogicalStream::new(64, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        let start = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        );
        text_output(&mut stream, &start, false);
        let big = sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "a".repeat(100)}
            }),
        );

        let error = stream.on_chunk(&big, false).unwrap_err();

        assert_eq!(
            error,
            StreamError::Oversized,
            "cumulative text beyond the body bound fails closed before reconstruction"
        );
    }

    #[test]
    fn accumulate_delta_fails_closed_on_cumulative_json_overflow() {
        let mut stream = LogicalStream::new(64, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        let start = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
            }),
        );
        text_output(&mut stream, &start, false);
        let big = sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "b".repeat(100)}
            }),
        );

        let error = stream.on_chunk(&big, false).unwrap_err();

        assert_eq!(
            error,
            StreamError::Oversized,
            "cumulative tool-input fragments beyond the body bound fail closed"
        );
    }

    #[test]
    fn reconstructs_multi_delta_text_across_a_managed_round() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(
            &mut stream,
            &multi_delta_text_block(0, &["Hel", "lo, ", "world"]),
            false,
        );
        text_output(&mut stream, &web_search_block(1, "toolu_ws", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        let reconstructed = stream.take_reconstructed().unwrap();
        let message: Value = serde_json::from_slice(&reconstructed).unwrap();
        assert_eq!(
            message["content"][0]["text"], "Hello, world",
            "multi-delta text is reconstructed intact for re-entry"
        );
    }

    #[test]
    fn reconstructs_thinking_and_signature_for_reentry() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        // An extended-thinking block precedes the managed WebSearch call. Its
        // reasoning text and signature must be reconstructed so the assistant
        // turn replayed on re-entry is faithful (Anthropic verifies the
        // thinking-block signature on the follow-up request).
        text_output(
            &mut stream,
            &thinking_block(0, "Let me search the web", "sig_abc123"),
            false,
        );
        text_output(&mut stream, &web_search_block(1, "toolu_ws", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        let reconstructed = stream.take_reconstructed().unwrap();
        let message: Value = serde_json::from_slice(&reconstructed).unwrap();
        assert_eq!(
            message["content"][0]["thinking"], "Let me search the web",
            "thinking text is reconstructed for re-entry"
        );
        assert_eq!(
            message["content"][0]["signature"], "sig_abc123",
            "the thinking-block signature is reconstructed for re-entry"
        );
    }

    #[test]
    fn forwards_upstream_error_event_without_synthesizing_success() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Partial"), false);
        let error_event = sse(
            "error",
            &json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "backend overloaded"}
            }),
        );

        let forwarded = text_output(&mut stream, &error_event, false);

        assert!(
            forwarded.contains("event: error"),
            "the upstream error is forwarded to the client"
        );
        assert!(
            forwarded.contains("overloaded_error"),
            "the upstream error type is preserved verbatim"
        );
        assert!(stream.has_failed(), "an upstream error poisons the stream");

        let outcome = stream.finish_round().unwrap();
        assert!(
            matches!(outcome.action, RoundAction::Done),
            "an upstream error ends the loop"
        );
        assert!(
            outcome.terminal.is_empty(),
            "no synthetic success terminal is fabricated after an upstream error"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "focused metadata preservation regression coverage")]
    fn terminal_message_delta_preserves_metadata_and_aggregates_usage() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        // Round 0: a managed WebSearch call; usage input=10, cache_read=2.
        text_output(&mut stream, &message_start_with_usage("msg_1", 10, 2), false);
        text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);
        text_output(
            &mut stream,
            &message_delta_with_metadata(&json!({
                "type": "message_delta",
                "delta": {
                    "container": null,
                    "stop_details": null,
                    "stop_reason": "tool_use",
                    "stop_sequence": null
                },
                "usage": {
                    "cache_creation_input_tokens": 4,
                    "output_tokens": 5,
                    "output_tokens_details": {"thinking_tokens": 2},
                    "server_tool_use": {"web_search_requests": 1}
                }
            })),
            false,
        );
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        // Round 1: the final answer; usage input=8, cache_read=3.
        stream.begin_round();
        text_output(&mut stream, &message_start_with_usage("msg_2", 8, 3), false);
        text_output(&mut stream, &text_block(0, "Answer"), false);
        text_output(
            &mut stream,
            &message_delta_with_metadata(&json!({
                "type": "message_delta",
                "delta": {
                    "container": {"id": "container_1", "expires_at": "2026-10-06T00:00:00Z"},
                    "stop_details": {"type": "refusal", "category": "general_harms", "explanation": "policy"},
                    "stop_reason": "stop_sequence",
                    "stop_sequence": "STOP"
                },
                "usage": {
                    "cache_creation_input_tokens": 6,
                    "output_tokens": 9,
                    "output_tokens_details": {"thinking_tokens": 7},
                    "server_tool_use": {"web_search_requests": 2}
                }
            })),
            false,
        );
        text_output(&mut stream, &message_stop(), false);
        let terminal = stream.finish_round().unwrap().terminal;

        let event = parse_event(&terminal, "message_delta");
        assert_eq!(
            event["delta"]["container"],
            json!({"id": "container_1", "expires_at": "2026-10-06T00:00:00Z"}),
            "terminal container metadata is preserved"
        );
        assert_eq!(
            event["delta"]["stop_details"],
            json!({"type": "refusal", "category": "general_harms", "explanation": "policy"}),
            "terminal stop details are preserved"
        );
        assert_eq!(
            event["delta"]["stop_reason"], "stop_sequence",
            "final stop reason carried"
        );
        assert_eq!(
            event["delta"]["stop_sequence"], "STOP",
            "matched stop sequence is preserved"
        );
        assert_eq!(event["usage"]["output_tokens"], 14, "output tokens aggregate 5 + 9");
        assert_eq!(event["usage"]["input_tokens"], 18, "input tokens aggregate 10 + 8");
        assert_eq!(
            event["usage"]["cache_read_input_tokens"], 5,
            "cache-read tokens aggregate 2 + 3"
        );
        assert_eq!(
            event["usage"]["cache_creation_input_tokens"], 10,
            "cache-creation tokens aggregate 4 + 6"
        );
        assert_eq!(
            event["usage"]["output_tokens_details"]["thinking_tokens"], 9,
            "output-token details aggregate 2 + 7"
        );
        assert_eq!(
            event["usage"]["server_tool_use"]["web_search_requests"], 3,
            "server-tool usage aggregates 1 + 2"
        );
    }

    #[test]
    fn forwarded_ping_bytes_match_canonical_sse() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        let payload = json!({"type": "ping"});
        let out = stream.on_chunk(&sse("ping", &payload), false).unwrap();
        let expected = sse("ping", &payload);
        assert_eq!(out, expected, "forwarded ping bytes must match canonical SSE framing");
        assert!(
            out.ends_with(b"\n\n"),
            "canonical SSE delimiter is a trailing blank line"
        );
    }

    fn stream_ready_for_second_round_text() -> LogicalStream {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL, MAX_QUERY);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "first"), false);
        stream.begin_round();
        text_output(&mut stream, &message_start("msg_2"), false);
        stream
    }

    #[test]
    fn remapped_block_start_preserves_payload_except_index() {
        let mut stream = stream_ready_for_second_round_text();
        let start = sse(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
        );
        let start_out = stream.on_chunk(&start, false).unwrap();
        let start_payload = parse_event(&start_out, "content_block_start");
        assert_eq!(start_payload["index"], 1, "round 1 continues at client index 1");
        assert_eq!(start_payload["type"], "content_block_start");
        assert_eq!(start_payload["content_block"]["type"], "text");
        assert!(
            std::str::from_utf8(&start_out).unwrap().ends_with("\n\n"),
            "remapped start keeps the SSE delimiter"
        );
    }

    #[test]
    fn remapped_delta_preserves_payload_except_index() {
        let mut stream = stream_ready_for_second_round_text();
        text_output(
            &mut stream,
            &sse(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
            ),
            false,
        );
        let delta = sse(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "second"}
            }),
        );
        let delta_out = stream.on_chunk(&delta, false).unwrap();
        let delta_payload = parse_event(&delta_out, "content_block_delta");
        assert_eq!(delta_payload["index"], 1, "delta index is remapped densely");
        assert_eq!(delta_payload["type"], "content_block_delta");
        assert_eq!(delta_payload["delta"]["text"], "second");
        assert!(
            std::str::from_utf8(&delta_out).unwrap().ends_with("\n\n"),
            "remapped delta keeps the SSE delimiter"
        );
    }
}
