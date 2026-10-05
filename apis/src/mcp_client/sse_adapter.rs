// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Adapt a praxis streaming response body into an rmcp SSE stream.
//!
//! [`sse_stream_from_body`] wraps a [`StreamingResponseBody`] as the
//! `BoxStream<Result<Sse, SseError>>` rmcp's `StreamableHttpPostResponse::Sse`
//! and `get_stream` expect. Two independent byte budgets are enforced at the
//! raw byte layer, *before* SSE parsing:
//!
//! * a cumulative *operation-stream* ceiling (total bytes across the stream), and
//! * a per-event ceiling (the retained size of one SSE event), ported from rmcp 3.4.0's private `SseEventSizeLimiter`.
//!
//! Either breach records a [`TransportSignal::ResponseTooLarge`] out-of-band
//! (first signal wins) so the caller classifies it as HTTP 413, and terminates
//! the stream with a credential-safe [`SseByteStreamError`]. rmcp lacks a
//! pin-project dependency here, so the per-event accounting is folded into a
//! single [`futures::stream::try_unfold`] rather than the upstream
//! `pin_project!` wrapper.

use std::{
    mem::size_of,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures::stream::{BoxStream, StreamExt as _};
use praxis_filter::{CalloutResponseTooLarge, StreamingResponseBody};
use sse_stream::{Error as SseError, Sse, SseStream};

use super::subrequest_transport::{TransportSignal, TransportSignalState};

/// Destination for an SSE size classification.
#[derive(Clone)]
pub(super) enum SseSignalTarget {
    /// One POST response owns a fixed signal generation.
    Fixed(Arc<OnceLock<TransportSignal>>),
    /// A standalone GET stream reports only to the call active when it fails.
    Active(Arc<TransportSignalState>),
}

impl SseSignalTarget {
    /// Record one first-wins transport classification.
    fn record(&self, classification: TransportSignal) {
        match self {
            Self::Fixed(signal) => {
                signal.get_or_init(|| classification);
            },
            Self::Active(state) => state.record_active(classification),
        }
    }
}

impl From<Arc<OnceLock<TransportSignal>>> for SseSignalTarget {
    fn from(signal: Arc<OnceLock<TransportSignal>>) -> Self {
        Self::Fixed(signal)
    }
}

/// Credential-safe failure surfaced from the SSE byte adapter.
///
/// Every variant carries only sizes — never the URL, headers, or body — so it
/// is safe to bubble through rmcp's `SseError::Body`.
#[allow(clippy::allow_attributes, dead_code, reason = "wired by selector filter in task 4")]
#[derive(Debug)]
pub(super) enum SseByteStreamError {
    /// The cumulative operation-stream byte budget was exceeded.
    Ceiling {
        /// The cumulative ceiling that was exceeded.
        limit: usize,
    },
    /// One SSE event exceeded the per-event byte ceiling.
    EventTooLarge {
        /// The per-event ceiling that was exceeded.
        max_size: usize,
    },
    /// A JSON-RPC event would exceed the request's parse-peak allowance.
    JsonExpansion {
        /// Maximum raw plus parsed bytes admitted for this call.
        limit: usize,
    },
    /// The underlying subrequest body errored mid-stream.
    Upstream,
}

impl std::fmt::Display for SseByteStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ceiling { limit } => write!(f, "mcp sse stream exceeded the {limit}-byte ceiling"),
            Self::EventTooLarge { max_size } => write!(f, "mcp sse event exceeded the {max_size}-byte limit"),
            Self::JsonExpansion { limit } => write!(f, "mcp sse JSON exceeded the {limit}-byte parse limit"),
            Self::Upstream => write!(f, "mcp sse upstream body error"),
        }
    }
}

impl std::error::Error for SseByteStreamError {}

/// Line-oriented per-event byte accounting, ported verbatim from rmcp 3.4.0's
/// private `SseEventSizeLimiter` (`transport/common/client_side_sse.rs`).
///
/// Tracks the retained size of the SSE event currently being assembled (data /
/// id / event lines, excluding comment lines starting with `:`) and reports a
/// breach once it would exceed `max_size`.
#[allow(
    clippy::allow_attributes,
    dead_code,
    reason = "used by sse_stream_from_body wired in task 4"
)]
#[derive(Debug)]
struct SseEventSizeLimiter {
    /// Maximum retained size for one SSE event.
    max_size: usize,
    /// Cumulative retained size of the current event.
    retained_size: usize,
    /// Size of the current line being assembled.
    line_size: usize,
    /// Whether the current line is a comment line (starts with `:`).
    line_is_comment: bool,
    /// Whether the previous byte was a `\r`.
    previous_was_cr: bool,
}

impl SseEventSizeLimiter {
    /// Create a new limiter with the given per-event size cap.
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            retained_size: 0,
            line_size: 0,
            line_is_comment: false,
            previous_was_cr: false,
        }
    }

    /// Observe a chunk of bytes and update the retained size.
    ///
    /// Returns `Err(())` if the size limit is exceeded.
    fn observe(&mut self, chunk: &[u8]) -> Result<(), ()> {
        for &byte in chunk {
            if self.previous_was_cr {
                self.previous_was_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' => {
                    self.finish_line()?;
                    self.previous_was_cr = true;
                },
                b'\n' => self.finish_line()?,
                _ => {
                    if self.line_size == 0 {
                        self.line_is_comment = byte == b':';
                    }
                    self.line_size = self.line_size.saturating_add(1);
                    self.check_limit()?;
                },
            }
        }
        Ok(())
    }

    /// Finish the current line and reset line state.
    ///
    /// Returns `Err(())` if the size limit is exceeded.
    fn finish_line(&mut self) -> Result<(), ()> {
        if self.line_size == 0 {
            self.retained_size = 0;
        } else if !self.line_is_comment {
            // The SSE parser inserts a newline when joining multiple data fields.
            self.retained_size = self.retained_size.saturating_add(self.line_size).saturating_add(1);
        }
        self.line_size = 0;
        self.line_is_comment = false;
        self.check_limit()
    }

    /// Check if the current retained size plus pending line size exceeds the limit.
    ///
    /// Returns `Err(())` if the limit is exceeded.
    fn check_limit(&self) -> Result<(), ()> {
        if self.retained_size.saturating_add(self.line_size) > self.max_size {
            Err(())
        } else {
            Ok(())
        }
    }
}

/// Mutable state threaded through the byte-layer [`futures::stream::try_unfold`].
#[allow(
    clippy::allow_attributes,
    dead_code,
    reason = "used by sse_stream_from_body wired in task 4"
)]
struct ByteState {
    /// The praxis streaming response body being adapted.
    body: Box<dyn StreamingResponseBody>,
    /// Cumulative bytes emitted so far.
    emitted: usize,
    /// Total operation-stream byte ceiling.
    operation_cap: usize,
    /// Per-event size limiter.
    per_event: SseEventSizeLimiter,
    /// Raw bytes in the unfinished SSE event, including comments and metadata.
    raw_event: RawSseEventWindow,
    /// Conservative charge for every event the parser may queue from one chunk.
    batch_charge: Arc<AtomicUsize>,
    /// `sse-stream` retains its `VecDeque<Sse>` allocation after draining it.
    queued_capacity_charge: usize,
    /// Out-of-band signal for recording `ResponseTooLarge`.
    signal: SseSignalTarget,
}

/// Track the borrowed raw SSE window before the parser can allocate owners.
/// A blank line ends the current event; CRLF is one line ending.
#[derive(Clone, Copy, Default)]
struct RawSseEventWindow {
    /// Raw bytes since the last blank event boundary.
    bytes: usize,
    /// Whether the current line has any content.
    line_has_content: bool,
    /// Treat a CRLF pair as one line ending across chunks.
    previous_was_cr: bool,
    /// Length of the line currently held in `sse-stream`'s unfinished-line Vec.
    unfinished_line_bytes: usize,
    /// Largest unfinished line ever buffered; its cleared Vec keeps capacity.
    max_buffered_line_bytes: usize,
}

impl RawSseEventWindow {
    /// Count completed frames while advancing the borrowed raw event window.
    fn observe(&mut self, chunk: &[u8]) -> usize {
        let mut completed = 0_usize;
        let mut line_was_buffered = self.unfinished_line_bytes != 0;
        for &byte in chunk {
            self.bytes = self.bytes.saturating_add(1);
            if self.previous_was_cr {
                self.previous_was_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' | b'\n' => {
                    if line_was_buffered {
                        self.max_buffered_line_bytes = self.max_buffered_line_bytes.max(self.unfinished_line_bytes);
                    }
                    self.unfinished_line_bytes = 0;
                    line_was_buffered = false;
                    if !self.line_has_content {
                        self.bytes = 0;
                        completed = completed.saturating_add(1);
                    }
                    self.line_has_content = false;
                    self.previous_was_cr = byte == b'\r';
                },
                _ => {
                    self.line_has_content = true;
                    self.unfinished_line_bytes = self.unfinished_line_bytes.saturating_add(1);
                },
            }
        }
        if self.unfinished_line_bytes != 0 {
            self.max_buffered_line_bytes = self.max_buffered_line_bytes.max(self.unfinished_line_bytes);
        }
        completed
    }
}

/// Adapt `body` into an rmcp SSE stream, enforcing both byte budgets.
///
/// `per_event_cap` bounds the retained size of any single SSE event;
/// `operation_cap` bounds the cumulative raw bytes across the whole stream;
/// `max_sse_event_size` is an outer per-event backstop (the effective per-event
/// cap is `min(per_event_cap, max_sse_event_size)`). A breach of either budget
/// records `signal` (first wins) and terminates the stream.
#[allow(clippy::allow_attributes, dead_code, reason = "wired by selector filter in task 4")]
#[expect(
    clippy::too_many_lines,
    reason = "two-budget byte-layer adapter is inherently sequential"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "wire and parsed limits are separate SSE boundaries"
)]
pub(super) fn sse_stream_from_body(
    body: Box<dyn StreamingResponseBody>,
    per_event_cap: usize,
    operation_cap: usize,
    max_sse_event_size: usize,
    signal: SseSignalTarget,
    preparse_peak_limit: Option<usize>,
) -> BoxStream<'static, Result<Sse, SseError>> {
    // SSE metadata is allocated by the parser too. Apply the parse allowance
    // to raw event bytes before handing a chunk to SseStream.
    let effective_per_event = per_event_cap
        .min(max_sse_event_size)
        .min(preparse_peak_limit.unwrap_or(usize::MAX));
    let parse_signal = signal.clone();
    let batch_charge = Arc::new(AtomicUsize::new(0));
    let state = ByteState {
        body,
        emitted: 0,
        operation_cap,
        per_event: SseEventSizeLimiter::new(effective_per_event),
        raw_event: RawSseEventWindow::default(),
        batch_charge: Arc::clone(&batch_charge),
        queued_capacity_charge: 0,
        signal,
    };

    let byte_stream = futures::stream::try_unfold(state, move |mut st| async move {
        loop {
            match st.body.next_chunk().await {
                Ok(Some(chunk)) => {
                    if let Some(limit) = preparse_peak_limit {
                        // sse-stream may hold the raw chunk, an unfinished
                        // line, and owned fields for every event it queues.
                        // Include comments, which its retained-field limiter
                        // intentionally excludes.
                        let raw_window = st.raw_event.bytes.checked_add(chunk.len());
                        let mut next_window = st.raw_event;
                        let completed = next_window.observe(&chunk);
                        let queued_nodes = if completed == 0 {
                            Some(0)
                        } else {
                            completed
                                .checked_mul(2)
                                .map(|slots| slots.max(4))
                                .and_then(|slots| slots.checked_mul(size_of::<Sse>()))
                        };
                        // `sse-stream` drains the parsed events before reading
                        // another chunk, but its VecDeque allocation survives.
                        let queued_capacity = st.queued_capacity_charge.max(queued_nodes.unwrap_or(limit));
                        // Its unfinished-line Vec is also cleared, not freed.
                        // Vec's amortized growth can reserve up to twice the
                        // largest fragmented line. Charge the previous capacity
                        // before handing this chunk to the parser.
                        let previous_line_capacity = st.raw_event.max_buffered_line_bytes.checked_mul(2);
                        if raw_window
                            .and_then(|bytes| bytes.checked_mul(3))
                            .and_then(|bytes| bytes.checked_add(queued_capacity))
                            .and_then(|bytes| bytes.checked_add(previous_line_capacity?))
                            .is_none_or(|bytes| bytes > limit)
                        {
                            st.signal.record(TransportSignal::ResponseTooLarge { limit });
                            return Err(SseByteStreamError::JsonExpansion { limit });
                        }
                        // Keep both parser allocations charged after the raw
                        // event window resets or the queued events drain.
                        let line_capacity = next_window.max_buffered_line_bytes.checked_mul(2);
                        let charge = raw_window
                            .and_then(|bytes| bytes.checked_add(queued_capacity))
                            .and_then(|bytes| bytes.checked_add(line_capacity?))
                            .unwrap_or(limit);
                        st.batch_charge.store(charge, Ordering::Relaxed);
                        st.queued_capacity_charge = queued_capacity;
                        st.raw_event = next_window;
                    }
                    // Per-event (per-message) ceiling, before parsing.
                    if st.per_event.observe(&chunk).is_err() {
                        let limit = st.per_event.max_size;
                        st.signal.record(TransportSignal::ResponseTooLarge { limit });
                        return Err(SseByteStreamError::EventTooLarge { max_size: limit });
                    }
                    // Cumulative operation-stream ceiling.
                    st.emitted = st.emitted.saturating_add(chunk.len());
                    if st.emitted > st.operation_cap {
                        let limit = st.operation_cap;
                        st.signal.record(TransportSignal::ResponseTooLarge { limit });
                        return Err(SseByteStreamError::Ceiling { limit });
                    }
                    if chunk.is_empty() {
                        continue; // keep pulling; never yield an empty frame
                    }
                    return Ok(Some((chunk, st)));
                },
                Ok(None) => return Ok(None),
                Err(error) => {
                    if error.downcast_ref::<CalloutResponseTooLarge>().is_some() {
                        // The core backstop is deliberately wider than this
                        // adapter's operation cap. Report the binding cap.
                        let limit = st.operation_cap;
                        st.signal.record(TransportSignal::ResponseTooLarge { limit });
                        return Err(SseByteStreamError::Ceiling { limit });
                    }
                    return Err(SseByteStreamError::Upstream);
                },
            }
        }
    });

    SseStream::from_bytes_stream(byte_stream)
        .map(move |event| {
            let frame: Sse = event?;
            // The raw event and parsed id/event strings can coexist.
            let metadata_bytes = frame
                .id
                .as_ref()
                .map_or(0, String::len)
                .checked_add(frame.event.as_ref().map_or(0, String::len))
                .and_then(|bytes| bytes.checked_mul(2));
            let parse_fits = preparse_peak_limit.is_none_or(|limit| {
                metadata_bytes
                    .and_then(|bytes| bytes.checked_add(batch_charge.load(Ordering::Relaxed)))
                    .and_then(|bytes| {
                        frame.data.as_deref().map_or(Some(bytes), |data| {
                            super::subrequest_transport::json_preparse_peak_bytes(data.as_bytes())
                                .and_then(|peak| bytes.checked_add(peak))
                        })
                    })
                    .is_some_and(|bytes| bytes <= limit)
            });
            if !parse_fits {
                let limit = preparse_peak_limit.unwrap_or(0);
                parse_signal.record(TransportSignal::ResponseTooLarge { limit });
                return Err(SseError::Body(Box::new(SseByteStreamError::JsonExpansion { limit })));
            }
            Ok(frame)
        })
        .boxed()
}

/// Test double for [`StreamingResponseBody`] that yields queued chunks and
/// records whether [`cancel`](StreamingResponseBody::cancel) ran.
#[cfg(test)]
pub(super) struct FakeStreamingBody {
    chunks: std::collections::VecDeque<bytes::Bytes>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    err_after: Option<usize>,
    /// Core ceiling to return instead of a generic upstream error.
    overflow_limit: Option<usize>,
    yielded: usize,
    /// Optional probe for confirming the chunk that trips parser admission.
    yielded_count: Option<Arc<AtomicUsize>>,
}

#[cfg(test)]
impl FakeStreamingBody {
    pub(super) fn from_chunks<I>(chunks: I, cancelled: Arc<std::sync::atomic::AtomicBool>) -> Self
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        Self {
            chunks: chunks.into_iter().collect(),
            cancelled,
            err_after: None,
            overflow_limit: None,
            yielded: 0,
            yielded_count: None,
        }
    }

    fn counting_chunks<I>(chunks: I, cancelled: Arc<std::sync::atomic::AtomicBool>, count: Arc<AtomicUsize>) -> Self
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut body = Self::from_chunks(chunks, cancelled);
        body.yielded_count = Some(count);
        body
    }

    pub(super) fn erroring_after<I>(chunks: I, cancelled: Arc<std::sync::atomic::AtomicBool>) -> Self
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut body = Self::from_chunks(chunks, cancelled);
        body.err_after = Some(body.chunks.len());
        body
    }

    /// Return the core's typed byte-limit error after the queued chunks.
    pub(super) fn overflowing_after<I>(chunks: I, cancelled: Arc<std::sync::atomic::AtomicBool>, limit: usize) -> Self
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut body = Self::erroring_after(chunks, cancelled);
        body.overflow_limit = Some(limit);
        body
    }

    #[expect(dead_code, reason = "probe for future cancellation tests")]
    pub(super) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
use async_trait::async_trait;

#[cfg(test)]
#[async_trait]
impl StreamingResponseBody for FakeStreamingBody {
    async fn next_chunk(&mut self) -> Result<Option<bytes::Bytes>, praxis_filter::FilterError> {
        if let Some(n) = self.err_after
            && self.yielded >= n
        {
            if let Some(limit) = self.overflow_limit {
                return Err(Box::new(CalloutResponseTooLarge { limit }));
            }
            return Err(praxis_filter::FilterError::from("fake upstream body error".to_owned()));
        }
        match self.chunks.pop_front() {
            Some(chunk) => {
                self.yielded += 1;
                if let Some(count) = &self.yielded_count {
                    count.store(self.yielded, Ordering::Relaxed);
                }
                Ok(Some(chunk))
            },
            None => Ok(None),
        }
    }

    async fn suppress(&mut self) -> Result<(), praxis_filter::FilterError> {
        Ok(())
    }

    async fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "tests use unwrap/expect/indexing for brevity"
)]
mod tests {
    use std::sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use bytes::Bytes;
    use futures::StreamExt as _;

    use super::{FakeStreamingBody, sse_stream_from_body};
    use crate::mcp_client::subrequest_transport::TransportSignal;

    fn cancelled_flag() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[tokio::test]
    async fn forwards_events_within_budgets() {
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"data: {\"jsonrpc\":\"2.0\"}\n\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 4_096, 16 * 1024 * 1024, Arc::clone(&signal).into(), None);

        let first = stream.next().await.expect("one event").expect("ok event");
        assert_eq!(first.data.as_deref(), Some("{\"jsonrpc\":\"2.0\"}"));
        assert!(stream.next().await.is_none(), "clean EOF");
        assert!(signal.get().is_none(), "no size signal on a clean stream");
    }

    #[tokio::test]
    async fn rejects_expanding_json_before_rmcp_reads_sse_event() {
        let numbers = vec!["1e15"; 3_000].join(",");
        let event = format!(
            "data: {{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{{\"code\":-32000,\"message\":\"failure\",\"data\":[{numbers}]}}}}\n\n"
        );
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(event)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            100_000,
            100_000,
            100_000,
            Arc::clone(&signal).into(),
            Some(40_000),
        );
        assert!(
            stream.next().await.expect("one event").is_err(),
            "the expanding JSON event must be rejected before rmcp parses it"
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 40_000 })
        ));
    }

    #[tokio::test]
    async fn cumulative_ceiling_breach_errors_and_signals_413() {
        // Two 3-byte chunks exceed a 4-byte operation cap on the second chunk.
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"aaa"), Bytes::from_static(b"bbb")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 4, 16 * 1024 * 1024, Arc::clone(&signal).into(), None);

        // The underlying byte stream errors; SseStream surfaces it as an Err item.
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "cumulative breach must surface an SSE error");
        assert!(
            matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit: 4 })),
            "cumulative breach records a 413 signal at the operation cap"
        );
    }

    #[tokio::test]
    async fn per_event_ceiling_breach_errors_and_signals_413() {
        // A single 20-byte data line exceeds a per-event cap of 8.
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"data: aaaaaaaaaaaaaaaaaaaa\n\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 8, 1_000_000, 16 * 1024 * 1024, Arc::clone(&signal).into(), None);

        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "per-event breach must surface an SSE error");
        assert!(
            matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit: 8 })),
            "per-event breach records a 413 signal at the per-event cap"
        );
    }

    #[tokio::test]
    async fn max_sse_event_size_clamps_the_per_event_cap() {
        // per_event_cap 1_000_000 but max_sse_event_size 8 => effective cap is 8.
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"data: aaaaaaaaaaaaaaaaaaaa\n\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_000_000, 1_000_000, 8, Arc::clone(&signal).into(), None);
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "max_sse_event_size acts as a per-event backstop");
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 8 })
        ));
    }

    #[tokio::test]
    async fn upstream_body_error_surfaces_without_a_size_signal() {
        let body = Box::new(FakeStreamingBody::erroring_after(
            [Bytes::from_static(b"data: partial\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 1_024, 16 * 1024 * 1024, Arc::clone(&signal).into(), None);
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
            }
        }
        assert!(saw_err, "an upstream body error must surface");
        assert!(signal.get().is_none(), "a transport error is not a size breach");
    }

    #[tokio::test]
    async fn core_streaming_ceiling_error_signals_binding_cap() {
        let body = Box::new(FakeStreamingBody::overflowing_after([], cancelled_flag(), 8));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 4, 4, 4, Arc::clone(&signal).into(), None);
        assert!(stream.next().await.expect("typed body error").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 4 })
        ));
    }

    #[tokio::test]
    async fn sse_id_metadata_obeys_parse_allowance_before_parser() {
        let event = format!("id: {}\ndata: {{\"ok\":true}}\n\n", "x".repeat(16_384));
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(event)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 65_536, 65_536, 65_536, Arc::clone(&signal).into(), Some(2_048));
        assert!(stream.next().await.expect("oversized id").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_048 })
        ));
    }

    #[tokio::test]
    async fn sse_id_and_json_peak_share_parse_allowance() {
        let event = format!("id: {}\ndata: {{\"ok\":true}}\n\n", "x".repeat(900));
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(event)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 65_536, 65_536, 65_536, Arc::clone(&signal).into(), Some(1_024));
        assert!(stream.next().await.expect("combined peak").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 1_024 })
        ));
    }

    #[tokio::test]
    async fn id_only_frame_is_bounded_before_parser_allocates_it() {
        let event = format!("id: {}\n\n", "x".repeat(900));
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(event)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 65_536, 65_536, 65_536, Arc::clone(&signal).into(), Some(1_024));
        assert!(stream.next().await.expect("oversized id-only frame").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 1_024 })
        ));
    }

    #[tokio::test]
    async fn fragmented_comment_line_is_bounded_before_parser_allocates_it() {
        let body = Box::new(FakeStreamingBody::from_chunks(
            [
                Bytes::from(format!(":{}", "x".repeat(300))),
                Bytes::from("x".repeat(300)),
            ],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 65_536, 65_536, 65_536, Arc::clone(&signal).into(), Some(1_024));
        assert!(stream.next().await.expect("oversized unfinished comment").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 1_024 })
        ));
    }

    #[tokio::test]
    async fn many_small_events_in_one_chunk_share_parse_allowance() {
        // The raw 6 KiB batch fits a 32 KiB allowance after the 3x
        // parser-copy reserve; 1,000 queued Sse structs do not.
        let chunk = "id:a\n\n".repeat(1_000);
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(chunk)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 65_536, 65_536, 65_536, Arc::clone(&signal).into(), Some(32_768));
        assert!(stream.next().await.expect("oversized event batch").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 32_768 })
        ));
    }

    #[tokio::test]
    async fn prior_queue_and_cleared_comment_line_remain_charged_before_later_result() {
        // Ten batches leave 2,000 parsed ID-only events and a reusable
        // VecDeque allocation. The fragmented comment leaves a large cleared
        // unfinished-line Vec. Both allocations still exist when the result
        // chunk reaches the byte layer.
        let id_batch = Bytes::from("id:a\n\n".repeat(200));
        let mut chunks = vec![id_batch; 10];
        chunks.push(Bytes::from(format!(":{}", "x".repeat(45_000))));
        chunks.push(Bytes::from(format!("{}\n\n", "x".repeat(45_000))));
        chunks.push(Bytes::from(format!(
            "data: {{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"value\":\"{}\"}}}}\n\n",
            "y".repeat(90_000)
        )));
        let yielded = Arc::new(AtomicUsize::new(0));
        let body = Box::new(FakeStreamingBody::counting_chunks(
            chunks,
            cancelled_flag(),
            Arc::clone(&yielded),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            200_000,
            300_000,
            200_000,
            Arc::clone(&signal).into(),
            Some(400_500),
        );
        for _ in 0..2_000 {
            let event = stream
                .next()
                .await
                .expect("ID-only event")
                .expect("admitted ID-only event");
            assert_eq!(event.id.as_deref(), Some("a"));
        }
        assert!(stream.next().await.expect("later result rejected").is_err());
        assert_eq!(
            yielded.load(Ordering::Relaxed),
            13,
            "the result chunk reaches pre-parse admission"
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 400_500 })
        ));
    }

    #[tokio::test]
    async fn drained_sse_queue_capacity_is_charged_in_json_map() {
        let first = Bytes::from("id:a\n\n".repeat(2_000));
        let second = Bytes::from(format!(
            "data: {{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"value\":\"{}\"}}}}\n\n",
            "z".repeat(15_000)
        ));
        let body = Box::new(FakeStreamingBody::from_chunks([first, second], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            200_000,
            300_000,
            200_000,
            Arc::clone(&signal).into(),
            Some(400_500),
        );
        for _ in 0..2_000 {
            assert!(stream.next().await.expect("ID-only event").is_ok());
        }
        assert!(stream.next().await.expect("JSON peak rejected").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 400_500 })
        ));
    }

    #[tokio::test]
    async fn split_crlf_boundary_keeps_small_event_admitted() {
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"data: {}\r"), Bytes::from_static(b"\n\r\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 1_024, 1_024, Arc::clone(&signal).into(), Some(1_024));
        assert_eq!(
            stream
                .next()
                .await
                .expect("one event")
                .expect("admitted")
                .data
                .as_deref(),
            Some("{}")
        );
        assert!(signal.get().is_none());
    }
}
