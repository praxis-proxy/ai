// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SSE parsing for OpenAI streaming APIs.
//!
//! - [`frame::SseFrameParser`] — byte-level SSE chunk reassembly
//! - `responses::ResponsesEvent`: typed Responses API event enum

#![cfg_attr(
    any(not(test), not(feature = "openai-responses")),
    allow(dead_code, reason = "used by filter implementations")
)]

#[cfg(feature = "openai-responses")]
mod config;
mod frame;
pub(crate) mod responses;

#[cfg(feature = "openai-responses")]
pub(crate) use config::SseParserConfig;
pub(crate) use frame::{SseFrame, SseFrameParser, SseParseError};

/// Decode a complete, already-buffered SSE body into each event's joined
/// `data` payload, in order.
///
/// Runs the crate's conforming SSE frame parser: CR, LF, and CRLF line endings, an
/// optional leading UTF-8 BOM, comments, and colonless fields. As the SSE
/// specification requires at end of stream, a final event without a
/// terminating blank line is discarded.
pub fn buffered_sse_event_data(body: &[u8]) -> Vec<Vec<u8>> {
    // The caller already bounded the body. The parser stores an `event:` value
    // lossily decoded, and each invalid byte becomes a three-byte U+FFFD, so
    // three times the body length covers any valid input.
    let mut parser = SseFrameParser::new(body.len().saturating_mul(3));
    let frames = parser.parse_chunk(body).unwrap_or_default();
    frames.into_iter().map(|frame| frame.data).collect()
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::indexing_slicing, reason = "tests")]
mod buffered_tests {
    use super::buffered_sse_event_data;

    #[test]
    fn decodes_cr_only_framing() {
        let events = buffered_sse_event_data(b"event: message\rdata: {\"a\":1}\r\r");
        assert_eq!(events, vec![b"{\"a\":1}".to_vec()], "bare CR must end lines and events");
    }

    #[test]
    fn skips_leading_bom() {
        let events = buffered_sse_event_data(b"\xEF\xBB\xBFdata: x\n\n");
        assert_eq!(
            events,
            vec![b"x".to_vec()],
            "a leading BOM must not hide the first field"
        );
    }

    #[test]
    fn discards_unterminated_final_event() {
        let events = buffered_sse_event_data(b"data: first\n\ndata: last");
        assert_eq!(
            events,
            vec![b"first".to_vec()],
            "an event without a terminating blank line must be discarded"
        );
    }

    #[test]
    fn keeps_frames_when_invalid_event_bytes_expand() {
        // SSE decodes the stream as UTF-8 with replacement (HTML §9.2.6), so
        // each 0xFF in the event name grows to three bytes once stored.
        let mut body = b"data: ok\n\nevent: ".to_vec();
        body.extend([0xFF; 256]);
        body.extend(b"\ndata: x\n\n");

        let events = buffered_sse_event_data(&body);

        assert_eq!(
            events,
            vec![b"ok".to_vec(), b"x".to_vec()],
            "replacement expansion must not drop complete frames"
        );
    }
}
