// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Bounded incremental SSE scanner for A2A streaming task-route capture.
//!
//! Processes `text/event-stream` response body chunks, extracts completed
//! `data:` payloads from SSE frames, and returns them for task-route
//! extraction. Handles arbitrary chunk boundaries, multi-line `data:`
//! fields, CRLF/LF/CR line endings, and comment lines.
//!
//! The scanner never modifies response bytes. It inspects chunks and
//! yields completed payloads; the caller passes bytes through unchanged.

// -----------------------------------------------------------------------------
// SseScanState
// -----------------------------------------------------------------------------

/// Incremental SSE parser state carried across response body chunks.
///
/// Stored in `filter_metadata` via hex encoding between
/// [`on_response_body`] calls.
///
/// [`on_response_body`]: praxis_filter::HttpFilter::on_response_body
#[derive(Default)]
pub(crate) struct SseScanState {
    /// Bytes of an incomplete line from the previous chunk.
    pub line_buf: Vec<u8>,

    /// Accumulated `data:` field values for the current SSE event,
    /// joined with `\n` per the SSE specification.
    pub data_buf: Vec<u8>,

    /// Whether any `data:` field has been seen for the current event.
    /// Distinguishes "no data lines" from "data lines with empty value".
    pub has_data: bool,

    /// Whether the previous chunk ended with CR, so a leading LF
    /// in the next chunk should be consumed as part of a CRLF pair.
    pub prev_cr: bool,

    /// Total scratch bytes consumed (`line_buf` + `data_buf`).
    pub scratch_bytes: usize,

    /// Progress discarding an event that exceeded `max_scratch_bytes`,
    /// if any. See [`SkipPhase`].
    pub skip: SkipPhase,

    /// Most recently completed stream action. Distinguishes "drop then
    /// recovered usage" from "usage then dropped terminal event."
    pub tail: ScanTail,
}

/// Progress discarding an oversized event's bytes without buffering them.
///
/// An event exceeding `max_scratch_bytes` is dropped rather than
/// retained: its bytes are discarded as they arrive, and scanning
/// resumes normally once a blank line (the SSE event boundary) is
/// found — tracked here without ever buffering the discarded content.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipPhase {
    /// Not discarding; bytes are buffered normally.
    #[default]
    NotSkipping,
    /// Discarding; the line since the last newline has been empty so
    /// far. A newline seen now means a blank line — the boundary.
    LineEmptySoFar,
    /// Discarding; the line since the last newline has had at least
    /// one byte. A newline seen now just ends that (non-blank) line.
    LineHasContent,
}

impl SkipPhase {
    /// Encode this phase for `filter_metadata` persistence.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NotSkipping => "not_skipping",
            Self::LineEmptySoFar => "line_empty",
            Self::LineHasContent => "line_has_content",
        }
    }

    /// Decode a persisted metadata value. Missing or unknown strings
    /// map to [`Self::NotSkipping`] so a restart resumes buffering.
    pub(crate) fn from_metadata_str(s: Option<&str>) -> Self {
        match s {
            Some("line_empty") => Self::LineEmptySoFar,
            Some("line_has_content") => Self::LineHasContent,
            _ => Self::NotSkipping,
        }
    }
}

/// Whether the last completed SSE action dispatched a payload or dropped
/// an oversized event. Persisted across chunks so finalize can tell a
/// recovered terminal usage event from a dropped one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScanTail {
    /// Last completed action dispatched a `data:` payload.
    #[default]
    Payload,
    /// Last completed action discarded an oversized event.
    Dropped,
}

impl ScanTail {
    /// Encode this tail for `filter_metadata` persistence.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Payload => "payload",
            Self::Dropped => "dropped",
        }
    }

    /// Decode a persisted metadata value. Missing or unknown strings
    /// map to [`Self::Payload`].
    pub(crate) fn from_metadata_str(s: Option<&str>) -> Self {
        match s {
            Some("dropped") => Self::Dropped,
            _ => Self::Payload,
        }
    }
}

// -----------------------------------------------------------------------------
// SseScanResult
// -----------------------------------------------------------------------------

/// Outcome of [`scan_sse_chunk`].
pub(crate) struct SseScanResult {
    /// Completed `data:` payloads dispatched during this chunk.
    pub payloads: Vec<Vec<u8>>,

    /// Number of events discarded during this chunk for exceeding
    /// `max_scratch_bytes`. Informational only — scanning already
    /// recovered at the next event boundary, so the caller does not
    /// need to take any corrective action.
    pub dropped_events: usize,
}

// -----------------------------------------------------------------------------
// Scanning
// -----------------------------------------------------------------------------

/// Process one SSE chunk, returning completed `data:` payloads and the
/// number of oversized events discarded.
///
/// An event that would exceed `max_scratch_bytes` is dropped rather than
/// buffered: its bytes are discarded without ever being retained, and
/// scanning resumes normally at the next blank-line event boundary. This
/// bounds memory use per event without losing later events on the same
/// stream — in particular, a terminal usage/summary event arriving after
/// an oversized one is still captured.
#[expect(clippy::too_many_lines, reason = "linear byte-processing loop")]
pub(crate) fn scan_sse_chunk(state: &mut SseScanState, chunk: &[u8], max_scratch_bytes: usize) -> SseScanResult {
    let mut payloads = Vec::new();
    let mut dropped_events = 0_usize;
    let mut i = 0;

    // If previous chunk ended with CR and this starts with LF, consume it
    // as the second half of a CRLF pair (not a new line boundary).
    if state.prev_cr && chunk.first() == Some(&b'\n') {
        i = 1;
    }
    state.prev_cr = false;

    while let Some(&b) = chunk.get(i) {
        let is_newline = b == b'\n' || b == b'\r';

        match state.skip {
            SkipPhase::NotSkipping => {
                if is_newline {
                    let before = payloads.len();
                    process_line(&state.line_buf, &mut state.data_buf, &mut state.has_data, &mut payloads);
                    if payloads.len() > before {
                        state.tail = ScanTail::Payload;
                    }
                    state.line_buf.clear();
                } else {
                    state.line_buf.push(b);
                }
            },
            SkipPhase::LineEmptySoFar => {
                state.skip = if is_newline {
                    SkipPhase::NotSkipping // blank line: event boundary found
                } else {
                    SkipPhase::LineHasContent
                };
            },
            SkipPhase::LineHasContent => {
                if is_newline {
                    state.skip = SkipPhase::LineEmptySoFar;
                }
            },
        }

        // CRLF within the same chunk: skip the LF. Applies regardless of
        // skip state, since it tracks raw byte position, not buffering.
        if b == b'\r' {
            if let Some(&next) = chunk.get(i + 1) {
                if next == b'\n' {
                    i += 1;
                }
            } else {
                state.prev_cr = true;
            }
        }

        if state.skip == SkipPhase::NotSkipping {
            state.scratch_bytes = state.line_buf.len() + state.data_buf.len();
            if state.scratch_bytes > max_scratch_bytes {
                state.line_buf.clear();
                state.data_buf.clear();
                state.has_data = false;
                state.scratch_bytes = 0;
                // The line since the last newline already has content
                // unless byte `b` itself was the newline starting it.
                state.skip = if is_newline {
                    SkipPhase::LineEmptySoFar
                } else {
                    SkipPhase::LineHasContent
                };
                dropped_events += 1;
                state.tail = ScanTail::Dropped;
            }
        }

        i += 1;
    }

    SseScanResult {
        payloads,
        dropped_events,
    }
}

/// Flush any incomplete line or pending `data:` event at stream end.
///
/// Providers such as Google Gemini may omit a trailing blank line before
/// closing the connection, so the scanner must dispatch buffered state on
/// `end_of_stream` rather than waiting for another `\n\n` boundary.
pub(crate) fn flush_sse_state(state: &mut SseScanState, payloads: &mut Vec<Vec<u8>>) {
    let before = payloads.len();

    if !state.line_buf.is_empty() {
        process_line(&state.line_buf, &mut state.data_buf, &mut state.has_data, payloads);
        state.line_buf.clear();
    }

    if state.has_data {
        payloads.push(std::mem::take(&mut state.data_buf));
        state.has_data = false;
    }

    if payloads.len() > before {
        state.tail = ScanTail::Payload;
    }

    state.scratch_bytes = 0;
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Only `data:` fields are captured; other SSE fields (`event:`, `id:`,
/// `retry:`) and comment lines are intentionally ignored because A2A
/// task payloads are always in `data:`.
fn process_line(line: &[u8], data_buf: &mut Vec<u8>, has_data: &mut bool, payloads: &mut Vec<Vec<u8>>) {
    if line.is_empty() {
        if *has_data {
            payloads.push(std::mem::take(data_buf));
            *has_data = false;
        }
        return;
    }

    if line.first() == Some(&b':') {
        return;
    }

    let Some(colon_pos) = line.iter().position(|&b| b == b':') else {
        return;
    };

    if line.get(..colon_pos) != Some(b"data".as_slice()) {
        return;
    }

    // Skip one optional space after the colon per SSE spec.
    let value_start = if line.get(colon_pos + 1) == Some(&b' ') {
        colon_pos + 2
    } else {
        colon_pos + 1
    };
    let value = line.get(value_start..).unwrap_or_default();

    if *has_data {
        data_buf.push(b'\n');
    }
    *has_data = true;
    data_buf.extend_from_slice(value);
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

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
    use super::*;

    const MAX_SCRATCH: usize = 65_536;

    // -------------------------------------------------------------------------
    // Single Complete Frame
    // -------------------------------------------------------------------------

    #[test]
    fn flush_pending_event_without_trailing_blank_line() {
        let mut state = SseScanState::default();
        let chunk = b"data: {\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);
        assert!(payloads.is_empty(), "no blank line yet");

        let mut flushed = Vec::new();
        flush_sse_state(&mut state, &mut flushed);
        assert_eq!(flushed.len(), 1, "EOF flush should dispatch pending event");
        assert_eq!(
            flushed[0],
            b"{\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}"
        );
    }

    #[test]
    fn single_data_frame_yields_payload() {
        let mut state = SseScanState::default();
        let chunk = b"data: {\"id\":\"task-1\"}\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "one event should yield one payload");
        assert_eq!(payloads[0], b"{\"id\":\"task-1\"}");
    }

    #[test]
    fn multiple_frames_in_one_chunk() {
        let mut state = SseScanState::default();
        let chunk = b"data: first\n\ndata: second\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 2, "two events should yield two payloads");
        assert_eq!(payloads[0], b"first");
        assert_eq!(payloads[1], b"second");
    }

    // -------------------------------------------------------------------------
    // Chunk Splitting
    // -------------------------------------------------------------------------

    #[test]
    fn frame_split_across_two_chunks() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"data: {\"id\":", MAX_SCRATCH);
        assert!(r.payloads.is_empty(), "incomplete frame yields no payload");

        let r = scan_sse_chunk(&mut state, b"\"task-1\"}\n\n", MAX_SCRATCH);
        assert_eq!(r.payloads.len(), 1, "completed frame yields payload");
        assert_eq!(r.payloads[0], b"{\"id\":\"task-1\"}");
    }

    #[test]
    fn line_split_across_chunks() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"da", MAX_SCRATCH);
        assert!(r.payloads.is_empty(), "partial field name yields no payload");

        let r = scan_sse_chunk(&mut state, b"ta: hello\n\n", MAX_SCRATCH);
        assert_eq!(r.payloads.len(), 1, "completed line yields payload");
        assert_eq!(r.payloads[0], b"hello");
    }

    #[test]
    fn blank_line_split_across_chunks() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"data: hello\n", MAX_SCRATCH);
        assert!(r.payloads.is_empty(), "first newline is end-of-line, not dispatch");

        let r = scan_sse_chunk(&mut state, b"\n", MAX_SCRATCH);
        assert_eq!(r.payloads.len(), 1, "second newline dispatches event");
        assert_eq!(r.payloads[0], b"hello");
    }

    // -------------------------------------------------------------------------
    // Multi-line Data
    // -------------------------------------------------------------------------

    #[test]
    fn multiline_data_joined_with_newline() {
        let mut state = SseScanState::default();
        let chunk = b"data: line1\ndata: line2\ndata: line3\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "one event with multi-line data");
        assert_eq!(payloads[0], b"line1\nline2\nline3");
    }

    #[test]
    fn multiline_data_split_across_chunks() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"data: line1\n", MAX_SCRATCH);
        assert!(r.payloads.is_empty(), "not dispatched yet");

        let r = scan_sse_chunk(&mut state, b"data: line2\n\n", MAX_SCRATCH);
        assert_eq!(r.payloads.len(), 1, "dispatched on blank line");
        assert_eq!(r.payloads[0], b"line1\nline2");
    }

    // -------------------------------------------------------------------------
    // CRLF
    // -------------------------------------------------------------------------

    #[test]
    fn crlf_line_endings() {
        let mut state = SseScanState::default();
        let chunk = b"data: hello\r\n\r\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "CRLF should work as line endings");
        assert_eq!(payloads[0], b"hello");
    }

    #[test]
    fn crlf_split_across_chunks() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"data: hello\r", MAX_SCRATCH);
        assert!(r.payloads.is_empty(), "CR at end of chunk, waiting for potential LF");

        let r = scan_sse_chunk(&mut state, b"\n\r\n", MAX_SCRATCH);
        assert_eq!(r.payloads.len(), 1, "CRLF spanning chunks should dispatch");
        assert_eq!(r.payloads[0], b"hello");
    }

    #[test]
    fn bare_cr_line_ending() {
        let mut state = SseScanState::default();
        let chunk = b"data: hello\r\r";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "bare CR should be a valid line terminator");
        assert_eq!(payloads[0], b"hello");
    }

    // -------------------------------------------------------------------------
    // Comments and Unknown Fields
    // -------------------------------------------------------------------------

    #[test]
    fn comments_ignored() {
        let mut state = SseScanState::default();
        let chunk = b": this is a comment\ndata: hello\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "comment should be ignored");
        assert_eq!(payloads[0], b"hello");
    }

    #[test]
    fn unknown_fields_ignored() {
        let mut state = SseScanState::default();
        let chunk = b"event: message\nid: 42\ndata: hello\nretry: 1000\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "unknown fields should be ignored");
        assert_eq!(payloads[0], b"hello");
    }

    #[test]
    fn empty_frames_ignored() {
        let mut state = SseScanState::default();
        let chunk = b"\n\ndata: hello\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(
            payloads.len(),
            1,
            "empty frames (consecutive blank lines) should be ignored"
        );
        assert_eq!(payloads[0], b"hello");
    }

    #[test]
    fn line_without_colon_ignored() {
        let mut state = SseScanState::default();
        let chunk = b"justtext\ndata: hello\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "line without colon should be ignored per SSE spec");
        assert_eq!(payloads[0], b"hello");
    }

    // -------------------------------------------------------------------------
    // Data Without Leading Space
    // -------------------------------------------------------------------------

    #[test]
    fn data_without_space_after_colon() {
        let mut state = SseScanState::default();
        let chunk = b"data:nospace\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "data without space after colon should work");
        assert_eq!(payloads[0], b"nospace");
    }

    #[test]
    fn data_with_empty_value() {
        let mut state = SseScanState::default();
        let chunk = b"data:\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "data with empty value should yield empty payload");
        assert!(payloads[0].is_empty(), "payload should be empty");
    }

    // -------------------------------------------------------------------------
    // Scratch Overflow
    // -------------------------------------------------------------------------

    #[test]
    fn scratch_overflow_recovers_at_next_event_boundary() {
        let mut state = SseScanState::default();
        let chunk = b"data: this-line-is-too-long-to-fit\n\ndata: ok\n\n";

        let result = scan_sse_chunk(&mut state, chunk, 10);

        assert_eq!(
            result.dropped_events, 1,
            "oversized event should be dropped, not abort scanning"
        );
        assert_eq!(result.payloads.len(), 1, "scanning should resume for the next event");
        assert_eq!(result.payloads[0], b"ok");
        assert!(
            state.tail != ScanTail::Dropped,
            "a payload after the drop means the tail is recovered usage, not a drop"
        );
    }

    #[test]
    fn completed_payload_returned_before_later_overflow() {
        let mut state = SseScanState::default();
        // First event completes (short), then a second event overflows (long).
        let chunk = b"data: ok\n\ndata: this-is-way-too-long-for-the-limit\n\n";

        let result = scan_sse_chunk(&mut state, chunk, 15);

        assert_eq!(result.dropped_events, 1, "should drop the second, oversized event");
        assert_eq!(
            result.payloads.len(),
            1,
            "first completed event should still be returned"
        );
        assert_eq!(result.payloads[0], b"ok");
        assert!(
            state.tail == ScanTail::Dropped,
            "dropping the last event must leave dropped_tail set"
        );
    }

    #[test]
    fn scratch_overflow_recovers_across_chunk_boundary() {
        let mut state = SseScanState::default();

        let r1 = scan_sse_chunk(&mut state, b"data: aaaaaaaaaaaaaaaa", 10);
        assert_eq!(r1.dropped_events, 1, "overflow should be detected mid-line");
        assert!(r1.payloads.is_empty());
        assert_eq!(state.tail, ScanTail::Dropped, "mid-line overflow is a dropped tail");

        let r2 = scan_sse_chunk(&mut state, b"aaaaaaaa\n\ndata: ok\n\n", 10);
        assert_eq!(r2.dropped_events, 0, "no further drops once the boundary is found");
        assert_eq!(r2.payloads.len(), 1, "next event should be captured normally");
        assert_eq!(r2.payloads[0], b"ok");
        assert!(
            state.tail != ScanTail::Dropped,
            "dispatching a later payload clears the dropped tail"
        );
    }

    #[test]
    fn skip_phase_round_trips_through_metadata_strings() {
        assert_eq!(SkipPhase::NotSkipping.as_str(), "not_skipping");
        assert_eq!(SkipPhase::LineEmptySoFar.as_str(), "line_empty");
        assert_eq!(SkipPhase::LineHasContent.as_str(), "line_has_content");

        assert_eq!(
            SkipPhase::from_metadata_str(Some("not_skipping")),
            SkipPhase::NotSkipping
        );
        assert_eq!(
            SkipPhase::from_metadata_str(Some("line_empty")),
            SkipPhase::LineEmptySoFar
        );
        assert_eq!(
            SkipPhase::from_metadata_str(Some("line_has_content")),
            SkipPhase::LineHasContent
        );
        assert_eq!(SkipPhase::from_metadata_str(None), SkipPhase::NotSkipping);
        assert_eq!(
            SkipPhase::from_metadata_str(Some("unknown")),
            SkipPhase::NotSkipping,
            "unknown values should resume normal buffering"
        );
    }

    #[test]
    fn scratch_resets_after_dispatch() {
        let mut state = SseScanState::default();

        let r = scan_sse_chunk(&mut state, b"data: ab\n\n", 20);
        assert_eq!(r.payloads.len(), 1, "first event dispatched");

        assert_eq!(
            state.scratch_bytes, 0,
            "scratch should reset after data_buf is cleared by dispatch"
        );

        let r = scan_sse_chunk(&mut state, b"data: cd\n\n", 20);
        assert_eq!(r.payloads.len(), 1, "second event should also succeed after reset");
    }

    // -------------------------------------------------------------------------
    // No Event Name Required
    // -------------------------------------------------------------------------

    #[test]
    fn data_without_event_name_yields_payload() {
        let mut state = SseScanState::default();
        let chunk = b"data: {\"task\":{\"id\":\"t1\"}}\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "data without event: name should still yield payload");
    }

    // -------------------------------------------------------------------------
    // Mixed Content
    // -------------------------------------------------------------------------

    #[test]
    fn json_rpc_response_in_sse_data() {
        let mut state = SseScanState::default();
        let chunk = b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"task\":{\"id\":\"task-42\",\"status\":{\"state\":\"TASK_STATE_WORKING\"}}}}\n\n";

        let SseScanResult { payloads, .. } = scan_sse_chunk(&mut state, chunk, MAX_SCRATCH);

        assert_eq!(payloads.len(), 1, "JSON-RPC response in SSE data");
        let parsed: serde_json::Value = serde_json::from_slice(&payloads[0]).expect("should be valid JSON");
        assert_eq!(
            parsed["result"]["task"]["id"].as_str(),
            Some("task-42"),
            "task ID should be extractable from parsed payload"
        );
    }
}
