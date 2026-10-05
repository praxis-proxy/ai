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
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use futures::stream::{BoxStream, StreamExt as _};
use praxis_filter::{CalloutResponseTooLarge, StreamingResponseBody};
use sse_stream::{Error as SseError, Sse, SseStream};

use super::subrequest_transport::{TransportSignal, TransportSignalState};

/// One request-session allowance shared by every rmcp GET and POST SSE stream.
/// Parser reservations are released with their streams; yielded frames remain
/// charged because rmcp does not tell us when its decoded queue is drained.
pub(super) struct SessionSseLedger {
    /// Session-wide allowance, absent for callers without a preparse budget.
    limit: Option<usize>,
    /// Charges shared by concurrently polled GET and POST streams.
    counters: Mutex<SessionSseCounters>,
}

/// Charges that must be updated together under the ledger lock.
#[derive(Default)]
struct SessionSseCounters {
    /// Decoded frames handed to rmcp, whose release is not observable here.
    yielded: usize,
    /// Sum of all live stream parser reservations.
    active_parser: usize,
}

impl SessionSseLedger {
    /// Create accounting for one rmcp client session.
    pub(super) fn new(limit: Option<usize>) -> Self {
        Self {
            limit,
            counters: Mutex::new(SessionSseCounters::default()),
        }
    }

    /// Report whether a budgeted session has yielded opaque rmcp messages.
    pub(super) fn has_yielded(&self) -> bool {
        self.counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .yielded
            != 0
    }

    /// Give one stream its own releasable parser reservation.
    fn stream(self: &Arc<Self>) -> Arc<StreamPermit> {
        Arc::new(StreamPermit {
            ledger: Arc::clone(self),
            parser: AtomicUsize::new(0),
        })
    }

    /// Apply the stricter of the session and operation limits.
    fn effective_limit(&self, local_limit: usize) -> usize {
        self.limit
            .map_or(local_limit, |session_limit| session_limit.min(local_limit))
    }
}

/// The two halves of one SSE adapter share this permit. No lock spans an
/// upstream await or a parser poll; each accounting transition is atomic.
struct StreamPermit {
    /// Session whose counters include this stream's parser charge.
    ledger: Arc<SessionSseLedger>,
    /// This stream's portion of the active parser charge.
    parser: AtomicUsize,
}

impl StreamPermit {
    /// Replace this stream's parser reservation under the session lock.
    fn reserve(&self, next_parser: usize, local_limit: usize) -> bool {
        let mut counters = self
            .ledger
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = self.parser.load(Ordering::Relaxed);
        let Some(active_parser) = counters
            .active_parser
            .checked_sub(current)
            .and_then(|other| other.checked_add(next_parser))
        else {
            return false;
        };
        if counters
            .yielded
            .checked_add(active_parser)
            .is_none_or(|total| total > self.ledger.effective_limit(local_limit))
        {
            return false;
        }
        counters.active_parser = active_parser;
        self.parser.store(next_parser, Ordering::Relaxed);
        drop(counters);
        true
    }

    /// Reserve the entire permitted raw chunk before awaiting the body. A
    /// budgeted session leaves half the currently free allowance available so
    /// an idle common GET does not consume every byte needed by a tool POST.
    fn reserve_pull(&self, local_limit: usize, operation_headroom: usize) -> Option<usize> {
        let mut counters = self
            .ledger
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = self.parser.load(Ordering::Relaxed);
        let total = counters.yielded.checked_add(counters.active_parser)?;
        let free = self.ledger.effective_limit(local_limit).checked_sub(total)?;
        let slice = if self.ledger.limit.is_some() {
            free.div_ceil(2)
        } else {
            free
        };
        let cap = operation_headroom.min(slice);
        let next = current.checked_add(cap)?;
        counters.active_parser = counters.active_parser.checked_add(cap)?;
        self.parser.store(next, Ordering::Relaxed);
        drop(counters);
        Some(cap)
    }

    /// Atomically move the parsed frame into the opaque rmcp-owned queue while
    /// replacing this stream's peak parser reservation with its steady charge.
    fn yield_frame(&self, frame: usize, next_parser: usize, local_limit: usize) -> bool {
        let mut counters = self
            .ledger
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = self.parser.load(Ordering::Relaxed);
        let Some(active_parser) = counters
            .active_parser
            .checked_sub(current)
            .and_then(|other| other.checked_add(next_parser))
        else {
            return false;
        };
        let Some(yielded) = counters.yielded.checked_add(frame) else {
            return false;
        };
        if yielded
            .checked_add(active_parser)
            .is_none_or(|total| total > self.ledger.effective_limit(local_limit))
        {
            return false;
        }
        counters.yielded = yielded;
        counters.active_parser = active_parser;
        self.parser.store(next_parser, Ordering::Relaxed);
        drop(counters);
        true
    }
}

impl Drop for StreamPermit {
    fn drop(&mut self) {
        let mut counters = self
            .ledger
            .counters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        counters.active_parser = counters
            .active_parser
            .saturating_sub(self.parser.load(Ordering::Relaxed));
    }
}

/// Destination for an SSE size classification.
#[derive(Clone)]
pub(super) enum SseSignalTarget {
    /// One POST response owns a fixed signal generation.
    Fixed {
        /// First-wins signal for the POST generation.
        signal: Arc<OnceLock<TransportSignal>>,
        /// Shared accounting for every stream in this rmcp session.
        ledger: Arc<SessionSseLedger>,
    },
    /// A standalone GET stream reports only to the call active when it fails.
    Active(Arc<TransportSignalState>),
}

impl SseSignalTarget {
    /// Bind a POST to the shared charge held by its rmcp client session.
    pub(super) fn fixed_with_ledger(signal: Arc<OnceLock<TransportSignal>>, ledger: Arc<SessionSseLedger>) -> Self {
        Self::Fixed { signal, ledger }
    }

    /// Keep decoded messages charged when rmcp resumes through a new stream.
    fn ledger(&self) -> Arc<SessionSseLedger> {
        match self {
            Self::Fixed { ledger, .. } => Arc::clone(ledger),
            Self::Active(state) => state.sse_ledger(),
        }
    }

    /// Record one first-wins transport classification.
    fn record(&self, classification: TransportSignal) {
        match self {
            Self::Fixed { signal, .. } => {
                signal.get_or_init(|| classification);
            },
            Self::Active(state) => state.record_active(classification),
        }
    }
}

impl From<Arc<OnceLock<TransportSignal>>> for SseSignalTarget {
    fn from(signal: Arc<OnceLock<TransportSignal>>) -> Self {
        Self::fixed_with_ledger(signal, Arc::new(SessionSseLedger::new(None)))
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
    /// Upper bound for data Strings still queued inside `sse-stream`.
    queued_data_charge: Arc<AtomicUsize>,
    /// Session-wide reservation for this parser and its rmcp-owned messages.
    permit: Arc<StreamPermit>,
    /// `sse-stream` retains its `VecDeque<Sse>` allocation after draining it.
    queued_capacity_charge: usize,
    /// Whether the last tightened chunk cap came from parse headroom.
    parse_chunk_cap: bool,
    /// Out-of-band signal for recording `ResponseTooLarge`.
    signal: SseSignalTarget,
}

/// Track the borrowed raw SSE window before the parser can allocate owners.
/// A blank line ends the current event; CRLF is one line ending.
#[derive(Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "SSE scanning tracks independent line, CRLF, and data-field facts"
)]
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
    /// First five bytes of the current line, including fragments across chunks.
    line_prefix: [u8; 5],
    /// Number of prefix bytes captured for the current line.
    line_prefix_len: u8,
    /// Whether the current event contains a `data:` field.
    has_data: bool,
}

impl RawSseEventWindow {
    /// Count completed frames while advancing the borrowed raw event window.
    #[expect(clippy::too_many_lines, reason = "scans SSE line and frame boundaries in one pass")]
    fn observe(&mut self, chunk: &[u8]) -> (usize, bool) {
        let mut completed = 0_usize;
        let mut saw_data = self.has_data;
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
                        self.has_data = false;
                    }
                    self.line_has_content = false;
                    self.line_prefix_len = 0;
                    self.previous_was_cr = byte == b'\r';
                },
                _ => {
                    if let Some(slot) = self.line_prefix.get_mut(usize::from(self.line_prefix_len)) {
                        *slot = byte;
                        self.line_prefix_len += 1;
                        if self.line_prefix_len == 5 && self.line_prefix == *b"data:" {
                            self.has_data = true;
                            saw_data = true;
                        }
                    }
                    self.line_has_content = true;
                    self.unfinished_line_bytes = self.unfinished_line_bytes.saturating_add(1);
                },
            }
        }
        if self.unfinished_line_bytes != 0 {
            self.max_buffered_line_bytes = self.max_buffered_line_bytes.max(self.unfinished_line_bytes);
        }
        (completed, saw_data)
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
    let queued_data_charge = Arc::new(AtomicUsize::new(0));
    let permit = signal.ledger().stream();
    let state = ByteState {
        body,
        emitted: 0,
        operation_cap,
        per_event: SseEventSizeLimiter::new(effective_per_event),
        raw_event: RawSseEventWindow::default(),
        batch_charge: Arc::clone(&batch_charge),
        queued_data_charge: Arc::clone(&queued_data_charge),
        permit: Arc::clone(&permit),
        queued_capacity_charge: 0,
        parse_chunk_cap: false,
        signal,
    };

    let byte_stream = futures::stream::try_unfold(state, move |mut st| async move {
        loop {
            if let Some(limit) = preparse_peak_limit {
                let shared_limit = st.permit.ledger.effective_limit(limit);
                // `sse-stream` drains its parsed queue before asking for the
                // next raw chunk. Only the unfinished current event can still
                // own a data String at this boundary.
                let pending_data = if st.raw_event.has_data {
                    st.raw_event.bytes.checked_mul(2).unwrap_or(limit)
                } else {
                    0
                };
                st.queued_data_charge.store(pending_data, Ordering::Relaxed);
                // The previous parser batch and its queue capacity remain
                // live after the last yielded event. Refresh this stream's
                // steady charge before reserving an awaited raw chunk.
                let steady = st.batch_charge.load(Ordering::Relaxed).checked_add(pending_data);
                if !steady.is_some_and(|bytes| st.permit.reserve(bytes, limit)) {
                    st.signal
                        .record(TransportSignal::ResponseTooLarge { limit: shared_limit });
                    st.body.cancel().await;
                    return Err(SseByteStreamError::JsonExpansion { limit: shared_limit });
                }
                // Keep the existing local raw-window bound as well as the
                // shared session allowance. The permit reserves the selected
                // chunk maximum before the asynchronous body pull.
                let line_capacity = st.raw_event.max_buffered_line_bytes.checked_mul(2);
                let parse_headroom = st
                    .raw_event
                    .bytes
                    .checked_mul(2)
                    .and_then(|bytes| bytes.checked_add(st.queued_capacity_charge))
                    .and_then(|bytes| bytes.checked_add(line_capacity?))
                    .map_or(0, |retained| limit.saturating_sub(retained));
                let operation_headroom = st.operation_cap.saturating_sub(st.emitted);
                let Some(chunk_cap) = st.permit.reserve_pull(limit, parse_headroom.min(operation_headroom)) else {
                    st.signal
                        .record(TransportSignal::ResponseTooLarge { limit: shared_limit });
                    st.body.cancel().await;
                    return Err(SseByteStreamError::JsonExpansion { limit: shared_limit });
                };
                st.parse_chunk_cap = chunk_cap < operation_headroom;
                if !st.body.try_cap_chunk_bytes(chunk_cap) {
                    st.signal
                        .record(TransportSignal::ResponseTooLarge { limit: shared_limit });
                    st.body.cancel().await;
                    return Err(SseByteStreamError::JsonExpansion { limit: shared_limit });
                }
            }
            match st.body.next_chunk().await {
                Ok(Some(chunk)) => {
                    if let Some(limit) = preparse_peak_limit {
                        // sse-stream may hold the raw chunk, an unfinished
                        // line, and owned fields for every event it queues.
                        // Include comments, which its retained-field limiter
                        // intentionally excludes.
                        let raw_window = st.raw_event.bytes.checked_add(chunk.len());
                        let mut next_window = st.raw_event;
                        let (completed, saw_data) = next_window.observe(&chunk);
                        let queued_nodes = if completed == 0 {
                            Some(0)
                        } else {
                            completed
                                .checked_mul(2)
                                .map(|slots| slots.max(4))
                                .and_then(|slots| slots.checked_mul(size_of::<Sse>()))
                        };
                        // A growing VecDeque can hold its old and new backing
                        // allocations during realloc. Its old capacity is at
                        // most half the new bound for this chunk.
                        let queued_growth_peak = if completed == 0 {
                            Some(0)
                        } else {
                            completed
                                .checked_mul(3)
                                .map(|slots| slots.max(4))
                                .and_then(|slots| slots.checked_mul(size_of::<Sse>()))
                        };
                        // `sse-stream` drains the parsed events before reading
                        // another chunk, but its VecDeque allocation survives.
                        let queued_capacity = st.queued_capacity_charge.max(queued_nodes.unwrap_or(limit));
                        let queued_preparse_peak = st.queued_capacity_charge.max(queued_growth_peak.unwrap_or(limit));
                        // After parsing this chunk, more than one completed
                        // event (or one completed plus an unfinished data
                        // event) can leave additional owned data Strings in
                        // the parser while the first event is deserialized.
                        // Every data byte comes from this raw window; double
                        // it for String capacity, then remove each yielded
                        // frame's own data from the charge in the map stage.
                        let queued_data = if saw_data && (completed > 1 || (completed > 0 && next_window.has_data)) {
                            raw_window.and_then(|bytes| bytes.checked_mul(2)).unwrap_or(limit)
                        } else if next_window.has_data {
                            next_window.bytes.checked_mul(2).unwrap_or(limit)
                        } else {
                            0
                        };
                        // Its unfinished-line Vec is also cleared, not freed.
                        // A `data:` field can grow the parser's String while
                        // that Vec is live, so reserve both old/new String
                        // allocations plus the projected line capacity. A
                        // comment-only batch cannot grow the data String and
                        // keeps the smaller existing three-copy admission.
                        let line_capacity = next_window.max_buffered_line_bytes.checked_mul(2);
                        let previous_line_capacity = st.raw_event.max_buffered_line_bytes.checked_mul(2);
                        let parse_line_capacity = if saw_data {
                            line_capacity
                        } else {
                            previous_line_capacity
                        };
                        let parser_peak = raw_window
                            .and_then(|bytes| bytes.checked_mul(if saw_data { 4 } else { 3 }))
                            .and_then(|bytes| bytes.checked_add(queued_preparse_peak))
                            .and_then(|bytes| bytes.checked_add(parse_line_capacity?));
                        if !parser_peak.is_some_and(|bytes| st.permit.reserve(bytes, limit)) {
                            let shared_limit = st.permit.ledger.effective_limit(limit);
                            st.signal
                                .record(TransportSignal::ResponseTooLarge { limit: shared_limit });
                            return Err(SseByteStreamError::JsonExpansion { limit: shared_limit });
                        }
                        // Keep both parser allocations charged after the raw
                        // event window resets or the queued events drain.
                        let charge = raw_window
                            .and_then(|bytes| bytes.checked_add(queued_capacity))
                            .and_then(|bytes| bytes.checked_add(line_capacity?))
                            .unwrap_or(limit);
                        st.batch_charge.store(charge, Ordering::Relaxed);
                        st.queued_data_charge.store(queued_data, Ordering::Relaxed);
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
                        // An armed chunk cap reports the binding limit whose
                        // remaining headroom was exhausted. Without a parse
                        // allowance, this is the cumulative core backstop.
                        let limit = if st.parse_chunk_cap {
                            preparse_peak_limit
                                .map_or(st.operation_cap, |limit| st.permit.ledger.effective_limit(limit))
                        } else {
                            st.operation_cap
                        };
                        st.signal.record(TransportSignal::ResponseTooLarge { limit });
                        return Err(if st.parse_chunk_cap {
                            SseByteStreamError::JsonExpansion { limit }
                        } else {
                            SseByteStreamError::Ceiling { limit }
                        });
                    }
                    return Err(SseByteStreamError::Upstream);
                },
            }
        }
    });

    SseStream::from_bytes_stream(byte_stream)
        .map(move |event| {
            let frame: Sse = event?;
            if let Some(limit) = preparse_peak_limit {
                // rmcp may queue multiple decoded notifications after they
                // leave this stream. Their release is unobservable here, so
                // keep a conservative charge for every yielded frame until
                // this rmcp session ends. Include the current parser batch while
                // rmcp materializes the next message.
                let frame_charge = frame
                    .id
                    .as_ref()
                    .map_or(0, String::len)
                    .checked_add(frame.event.as_ref().map_or(0, String::len))
                    .and_then(|bytes| bytes.checked_mul(2))
                    .and_then(|bytes| {
                        frame.data.as_deref().map_or(Some(bytes), |data| {
                            super::subrequest_transport::json_preparse_peak_bytes(data.as_bytes())
                                .and_then(|peak| bytes.checked_add(peak))
                        })
                    });
                let queued_after_current = frame
                    .data
                    .as_ref()
                    .map_or(Some(0), |data| data.len().checked_mul(2))
                    .map(|current| queued_data_charge.load(Ordering::Relaxed).saturating_sub(current));
                let next_parser =
                    queued_after_current.and_then(|bytes| batch_charge.load(Ordering::Relaxed).checked_add(bytes));
                if !frame_charge
                    .zip(next_parser)
                    .is_some_and(|(frame, parser)| permit.yield_frame(frame, parser, limit))
                {
                    let shared_limit = permit.ledger.effective_limit(limit);
                    parse_signal.record(TransportSignal::ResponseTooLarge { limit: shared_limit });
                    return Err(SseError::Body(Box::new(SseByteStreamError::JsonExpansion {
                        limit: shared_limit,
                    })));
                }
                if let Some(bytes) = queued_after_current {
                    queued_data_charge.store(bytes, Ordering::Relaxed);
                }
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
    /// Emulate the core body's per-chunk admission for budgeted SSE tests.
    max_chunk_bytes: Option<usize>,
    /// Exercise fail-closed handling for other streaming body implementations.
    supports_chunk_cap: bool,
    /// Synchronize two independent adapter pulls in a concurrency regression.
    pull_barrier: Option<Arc<tokio::sync::Barrier>>,
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
            max_chunk_bytes: None,
            supports_chunk_cap: true,
            pull_barrier: None,
        }
    }

    pub(super) fn counting_chunks<I>(
        chunks: I,
        cancelled: Arc<std::sync::atomic::AtomicBool>,
        count: Arc<AtomicUsize>,
    ) -> Self
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

    /// Model a body that cannot enforce per-chunk admission.
    fn without_chunk_cap(mut self) -> Self {
        self.supports_chunk_cap = false;
        self
    }

    fn with_pull_barrier(mut self, barrier: Arc<tokio::sync::Barrier>) -> Self {
        self.pull_barrier = Some(barrier);
        self
    }
}

#[cfg(test)]
use async_trait::async_trait;

#[cfg(test)]
#[async_trait]
impl StreamingResponseBody for FakeStreamingBody {
    fn try_cap_chunk_bytes(&mut self, limit: usize) -> bool {
        if !self.supports_chunk_cap {
            return false;
        }
        self.max_chunk_bytes = Some(self.max_chunk_bytes.map_or(limit, |current| current.min(limit)));
        true
    }

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
                if let Some(limit) = self.max_chunk_bytes
                    && chunk.len() > limit
                {
                    self.chunks.clear();
                    return Err(Box::new(CalloutResponseTooLarge { limit }));
                }
                if let Some(barrier) = &self.pull_barrier {
                    barrier.wait().await;
                }
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

    use super::{FakeStreamingBody, SessionSseLedger, SseSignalTarget, sse_stream_from_body};
    use crate::mcp_client::subrequest_transport::{TransportSignal, TransportSignalState};

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
    async fn yielded_notifications_share_one_parse_allowance() {
        let json = format!(
            "{{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{{\"level\":\"info\",\"data\":\"{}\"}}}}",
            "x".repeat(100_000)
        );
        let chunk = Bytes::from(format!("data: {json}\n\n"));
        let body = Box::new(FakeStreamingBody::from_chunks(vec![chunk; 5], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            1_000_000,
            6_000_000,
            1_000_000,
            Arc::clone(&signal).into(),
            Some(430_000),
        );
        assert!(stream.next().await.expect("first notification").is_ok());
        assert!(
            stream.next().await.expect("later notification rejected").is_err(),
            "rmcp may still retain the first decoded notification in its worker queue"
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 430_000 })
        ));
    }

    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "checks batch admission and typed failure signal")]
    async fn queued_data_strings_share_the_first_frame_parse_allowance() {
        let first = format!(
            r#"{{"jsonrpc":"2.0","id":1,"result":{{"content":[{{"type":"text","text":"{}\n"}}]"#,
            "x".repeat(400_000)
        );
        let second = format!(
            r#"{{"jsonrpc":"2.0","id":2,"result":{{"content":[{{"type":"text","text":"{}"}}]"#,
            "y".repeat(600_000)
        );
        let raw = format!("data: {first}\ndata: }}}}\n\ndata: {second}\ndata: }}}}\n\n");
        // The 16x escaped-message reserve and batch copies fit below this
        // allowance. The extra queued data String must make it fail.
        let limit = 8_000_000;
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(raw)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            10_000_000,
            10_000_000,
            10_000_000,
            Arc::clone(&signal).into(),
            Some(limit),
        );
        assert!(
            stream
                .next()
                .await
                .expect("first frame rejected before JSON parse")
                .is_err(),
            "the second queued data String must remain charged while the first frame is deserialized"
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: recorded }) if *recorded == limit
        ));
    }

    #[tokio::test]
    async fn resumed_streams_share_queued_message_charge() {
        let event = Bytes::from(format!(
            "data: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{\"message\":\"{}\"}}}}\n\n",
            "x".repeat(100_000)
        ));
        let limit = 500_000;
        let state = Arc::new(TransportSignalState::new(Some(limit)));
        let signal = state.current();
        let first_body = Box::new(FakeStreamingBody::from_chunks([event.clone()], cancelled_flag()));
        let first_target = SseSignalTarget::fixed_with_ledger(Arc::clone(&signal), state.sse_ledger());
        let mut first = sse_stream_from_body(first_body, 500_000, 500_000, 500_000, first_target, Some(limit));
        assert!(first.next().await.expect("first message").is_ok());
        assert!(state.has_budgeted_sse_yields());
        drop(first);

        let resumed_body = Box::new(FakeStreamingBody::from_chunks([event], cancelled_flag()));
        let resumed_target = SseSignalTarget::Active(state);
        let mut resumed = sse_stream_from_body(resumed_body, 500_000, 500_000, 500_000, resumed_target, Some(limit));
        assert!(
            resumed
                .next()
                .await
                .expect("resumed message exceeds shared allowance")
                .is_err()
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 500_000 })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[expect(clippy::too_many_lines, reason = "sets up two independently polled SSE streams")]
    async fn concurrent_get_and_post_share_parser_and_yield_allowance() {
        let limit = 600_000;
        let state = Arc::new(TransportSignalState::new(Some(limit)));
        let signal = state.current();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let event = Bytes::from(format!(
            "data: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{{\"message\":\"{}\"}}}}\n\n",
            "x".repeat(100_000)
        ));
        let post_body = Box::new(
            FakeStreamingBody::from_chunks([event.clone()], cancelled_flag()).with_pull_barrier(Arc::clone(&barrier)),
        );
        let get_body = Box::new(FakeStreamingBody::from_chunks([event], cancelled_flag()).with_pull_barrier(barrier));
        let mut post = sse_stream_from_body(
            post_body,
            limit,
            limit,
            limit,
            SseSignalTarget::fixed_with_ledger(Arc::clone(&signal), state.sse_ledger()),
            Some(limit),
        );
        let mut get = sse_stream_from_body(
            get_body,
            limit,
            limit,
            limit,
            SseSignalTarget::Active(state),
            Some(limit),
        );

        let (post_result, get_result) = tokio::join!(post.next(), get.next());
        let admitted = [post_result, get_result]
            .into_iter()
            .filter(|result| result.as_ref().is_some_and(Result::is_ok))
            .count();
        assert_eq!(admitted, 1, "only one large frame fits alongside both live parsers");
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 600_000 })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_small_get_and_post_both_fit() {
        let limit = 600_000;
        let state = Arc::new(TransportSignalState::new(Some(limit)));
        let signal = state.current();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let event =
            Bytes::from_static(b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n");
        let post_body = Box::new(
            FakeStreamingBody::from_chunks([event.clone()], cancelled_flag()).with_pull_barrier(Arc::clone(&barrier)),
        );
        let get_body = Box::new(FakeStreamingBody::from_chunks([event], cancelled_flag()).with_pull_barrier(barrier));
        let mut post = sse_stream_from_body(
            post_body,
            limit,
            limit,
            limit,
            SseSignalTarget::fixed_with_ledger(Arc::clone(&signal), state.sse_ledger()),
            Some(limit),
        );
        let mut get = sse_stream_from_body(
            get_body,
            limit,
            limit,
            limit,
            SseSignalTarget::Active(state),
            Some(limit),
        );

        let (post_result, get_result) = tokio::join!(post.next(), get.next());
        assert!(post_result.expect("POST frame").is_ok());
        assert!(get_result.expect("GET frame").is_ok());
        assert!(signal.get().is_none());
    }

    #[test]
    fn concurrent_yield_reservations_cannot_lose_a_charge() {
        let ledger = Arc::new(SessionSseLedger::new(Some(100)));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let first = ledger.stream();
        let second = ledger.stream();
        let (first_admitted, second_admitted) = std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let first = scope.spawn(move || {
                first_barrier.wait();
                first.yield_frame(60, 0, 100)
            });
            let second = scope.spawn(move || {
                barrier.wait();
                second.yield_frame(60, 0, 100)
            });
            (first.join().unwrap(), second.join().unwrap())
        });
        assert_ne!(first_admitted, second_admitted, "only one sixty-byte frame fits");
        assert!(ledger.has_yielded());
    }

    #[tokio::test]
    async fn small_two_event_batch_remains_admitted() {
        let chunk = Bytes::from_static(
            b"data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n\
              data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n",
        );
        let body = Box::new(FakeStreamingBody::from_chunks([chunk], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 20_000, 20_000, 20_000, Arc::clone(&signal).into(), Some(20_000));
        assert!(stream.next().await.expect("first small event").is_ok());
        assert!(stream.next().await.expect("second small event").is_ok());
        assert!(stream.next().await.is_none(), "batch ends cleanly");
        assert!(signal.get().is_none(), "small batches should not exhaust the allowance");
    }

    #[tokio::test]
    async fn rejects_multiline_data_string_growth_before_parser_allocation() {
        let raw = format!("id:a\n\ndata: {}\ndata: y\n\n", "x".repeat(100_000));
        let limit = raw.len() * 3 + 1_000;
        assert!(
            raw.len() * 4 > limit,
            "the old three-copy bound would admit this multiline growth"
        );
        let body = Box::new(FakeStreamingBody::from_chunks([Bytes::from(raw)], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(
            body,
            1_000_000,
            1_000_000,
            1_000_000,
            Arc::clone(&signal).into(),
            Some(limit),
        );
        assert!(stream.next().await.expect("multiline growth rejected").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: recorded }) if *recorded == limit
        ));
    }

    #[tokio::test]
    async fn split_data_prefix_still_reserves_multiline_growth() {
        let continuation = format!("ta: {}\ndata: y\n\n", "x".repeat(1_000));
        let raw_len = 2 + continuation.len();
        let limit = raw_len * 3 + 100;
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"da"), Bytes::from(continuation)],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 10_000, 10_000, 10_000, Arc::clone(&signal).into(), Some(limit));
        assert!(stream.next().await.expect("split data prefix rejected").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: recorded }) if *recorded == limit
        ));
    }

    #[tokio::test]
    async fn budgeted_sse_requires_chunk_admission_before_first_pull() {
        let yielded = Arc::new(AtomicUsize::new(0));
        let cancelled = cancelled_flag();
        let body = Box::new(
            FakeStreamingBody::counting_chunks(
                [Bytes::from_static(b"data: {}\n\n")],
                Arc::clone(&cancelled),
                Arc::clone(&yielded),
            )
            .without_chunk_cap(),
        );
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 1_024, 1_024, Arc::clone(&signal).into(), Some(2_048));
        assert!(stream.next().await.expect("unsupported body rejected").is_err());
        assert_eq!(yielded.load(Ordering::Relaxed), 0, "no chunk was pulled");
        assert!(cancelled.load(Ordering::SeqCst), "the body was cancelled");
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_048 })
        ));
    }

    #[tokio::test]
    async fn budgeted_sse_withholds_oversized_first_chunk() {
        let yielded = Arc::new(AtomicUsize::new(0));
        let body = Box::new(FakeStreamingBody::counting_chunks(
            [Bytes::from("x".repeat(3_000))],
            cancelled_flag(),
            Arc::clone(&yielded),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 10_000, 10_000, 10_000, Arc::clone(&signal).into(), Some(2_048));
        assert!(stream.next().await.expect("oversized chunk withheld").is_err());
        assert_eq!(yielded.load(Ordering::Relaxed), 0, "chunk did not reach the adapter");
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 2_048 })
        ));
    }

    #[tokio::test]
    async fn prior_parser_owners_tighten_chunk_cap_before_next_pull() {
        let yielded = Arc::new(AtomicUsize::new(0));
        let body = Box::new(FakeStreamingBody::counting_chunks(
            [
                Bytes::from(format!("data: {}", "x".repeat(1_000))),
                Bytes::from("y".repeat(3_000)),
            ],
            cancelled_flag(),
            Arc::clone(&yielded),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 10_000, 10_000, 10_000, Arc::clone(&signal).into(), Some(6_500));
        assert!(
            stream
                .next()
                .await
                .expect("prior retained line rejects next chunk")
                .is_err()
        );
        assert_eq!(
            yielded.load(Ordering::Relaxed),
            1,
            "second chunk was withheld by core cap"
        );
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 6_500 })
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
    #[expect(
        clippy::too_many_lines,
        reason = "checks queue, line, and later result in one stream"
    )]
    async fn prior_queue_and_cleared_comment_line_remain_charged_before_later_result() {
        // Ten batches process 2,000 ID-only events and leave a reusable
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
            "z".repeat(80_000)
        ));
        let cap = 600_000;
        assert!(
            first.len() * 3 + 2_000 * 3 * size_of::<super::Sse>() <= cap,
            "the ID batch fits the queue growth peak"
        );
        assert!(
            second.len() * 3 + 2_000 * 2 * size_of::<super::Sse>() <= cap,
            "the result chunk fits preparse before JSON mapping"
        );
        let body = Box::new(FakeStreamingBody::from_chunks([first, second], cancelled_flag()));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 200_000, 300_000, 200_000, Arc::clone(&signal).into(), Some(cap));
        for _ in 0..2_000 {
            assert!(stream.next().await.expect("ID-only event").is_ok());
        }
        assert!(stream.next().await.expect("JSON peak rejected").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 600_000 })
        ));
    }

    #[tokio::test]
    async fn queue_growth_peak_is_rejected_before_sse_parser_allocates() {
        // The 1,025th event can grow VecDeque from 1,024 to 2,048 slots.
        // Both backing allocations may be live during that growth.
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from("id:a\n\n".repeat(1_025))],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 10_000, 10_000, 10_000, Arc::clone(&signal).into(), Some(220_000));
        assert!(stream.next().await.expect("queue growth rejected").is_err());
        assert!(matches!(
            signal.get(),
            Some(TransportSignal::ResponseTooLarge { limit: 220_000 })
        ));
    }

    #[tokio::test]
    async fn split_crlf_boundary_keeps_small_event_admitted() {
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"data: {}\r"), Bytes::from_static(b"\n\r\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 1_024, 1_024, Arc::clone(&signal).into(), Some(2_048));
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
