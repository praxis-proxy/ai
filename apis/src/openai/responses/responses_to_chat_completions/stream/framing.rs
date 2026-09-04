// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Framing boundary over the AI-local SSE parser.
//!
//! This is the only module in the streaming converter that names
//! [`SseFrameParser`]. It re-exposes completed frames and end-of-stream state
//! through local types so the state machine stays independent of the parser.
//! When a shared Praxis SSE codec becomes available, only this module and the
//! event encoder need to change (see
//! [issue #842](https://github.com/praxis-proxy/ai/issues/842)).

use crate::openai::sse::{SseFrameParser, SseParseError};

/// One completed SSE frame: its joined data payload.
///
/// Chat Completions streams carry data only on `data:` lines, so the `event:`
/// field is intentionally discarded at this boundary.
#[derive(Debug)]
pub(super) struct Frame {
    /// Joined `data:` payload bytes for the frame.
    pub(super) data: Vec<u8>,
}

/// Framing failures surfaced to the state machine.
#[derive(Debug)]
pub(super) enum FramingError {
    /// Buffered bytes exceeded the configured SSE buffer limit.
    BufferOverflow {
        /// Bytes buffered when the limit tripped.
        buffered_bytes: usize,
        /// Configured buffer byte limit.
        limit: usize,
    },
    /// Decoding a chunk would exceed the configured decoded-frame limit.
    FrameLimit {
        /// Frame count that tripped the limit.
        count: usize,
        /// Configured decoded-frame limit.
        limit: usize,
    },
}

/// Incremental SSE reassembly scoped to one streaming response.
pub(super) struct Framing {
    /// Underlying byte-level frame parser.
    parser: SseFrameParser,
}

impl Framing {
    /// Create a framing adapter bounded by the SSE buffer byte limit.
    pub(super) fn new(max_buffer_bytes: usize) -> Self {
        Self {
            parser: SseFrameParser::new(max_buffer_bytes),
        }
    }

    /// Feed a response body chunk, returning any completed frames.
    ///
    /// `frames_seen` is the number of frames decoded in earlier callbacks and
    /// `max_frames` the ceiling across the whole response. The limit is enforced
    /// *during* parsing so decoding stops at the cap rather than materializing
    /// every frame in an oversized callback before rejection.
    pub(super) fn push(
        &mut self,
        chunk: &[u8],
        frames_seen: usize,
        max_frames: usize,
    ) -> Result<Vec<Frame>, FramingError> {
        let frames = self
            .parser
            .parse_chunk_with_counted_event_limit(chunk, frames_seen, max_frames, |_| true)
            .map_err(map_parse_error)?;
        Ok(frames.into_iter().map(|frame| Frame { data: frame.data }).collect())
    }

    /// Return whether an unterminated frame remains buffered at end of stream.
    pub(super) fn has_incomplete_frame(&self) -> bool {
        self.parser.has_incomplete_frame()
    }
}

/// Translate a parser error into a framing error the state machine understands.
fn map_parse_error(error: SseParseError) -> FramingError {
    match error {
        SseParseError::BufferOverflow { buffered_bytes, limit } => {
            FramingError::BufferOverflow { buffered_bytes, limit }
        },
        SseParseError::EventLimitExceeded { count, limit } => FramingError::FrameLimit { count, limit },
        // `push` enforces only the buffer and decoded-frame limits; other variants
        // describe higher-level SSE semantics this adapter never invokes.
        other => {
            tracing::warn!(error = %other, "unexpected SSE parser error at framing boundary");
            FramingError::BufferOverflow {
                buffered_bytes: 0,
                limit: 0,
            }
        },
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// A frame budget large enough never to trip in the reassembly tests.
    const WIDE_FRAMES: usize = 1_000_000;

    #[test]
    fn reassembles_frame_across_chunks() {
        let mut framing = Framing::new(4096);
        assert!(framing.push(b"data: {\"a\":1}", 0, WIDE_FRAMES).unwrap().is_empty());
        let frames = framing.push(b"\n\n", 0, WIDE_FRAMES).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, b"{\"a\":1}");
    }

    #[test]
    fn reports_incomplete_frame_at_eof() {
        let mut framing = Framing::new(4096);
        framing.push(b"data: partial", 0, WIDE_FRAMES).unwrap();
        assert!(framing.has_incomplete_frame());
    }

    #[test]
    fn buffer_overflow_maps_to_framing_error() {
        let mut framing = Framing::new(8);
        let error = framing
            .push(b"data: way too long for the tiny buffer\n\n", 0, WIDE_FRAMES)
            .unwrap_err();
        assert!(matches!(error, FramingError::BufferOverflow { .. }));
    }

    #[test]
    fn push_stops_at_frame_budget() {
        // Five complete frames arrive in a single callback, but only three may be
        // processed. Decoding must stop at the cap and error rather than
        // materialize every frame first.
        let mut framing = Framing::new(4096);
        let chunk = b"data: 1\n\ndata: 2\n\ndata: 3\n\ndata: 4\n\ndata: 5\n\n";
        let error = framing.push(chunk, 0, 3).unwrap_err();
        assert!(
            matches!(error, FramingError::FrameLimit { .. }),
            "exceeding the frame budget within one callback must stop decoding and error",
        );
    }

    #[test]
    fn push_counts_prior_frames_against_budget() {
        // Frames already seen in earlier callbacks count toward the budget, so a
        // later callback that would push the total over the cap is rejected.
        let mut framing = Framing::new(4096);
        let error = framing.push(b"data: 1\n\n", 3, 3).unwrap_err();
        assert!(
            matches!(error, FramingError::FrameLimit { .. }),
            "a frame arriving at the budget must be rejected",
        );
    }
}
