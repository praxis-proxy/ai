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
//! * a per-event ceiling (the retained size of one SSE event), ported from
//!   rmcp 3.4.0's private `SseEventSizeLimiter`.
//!
//! Either breach records a [`TransportSignal::ResponseTooLarge`] out-of-band
//! (first signal wins) so the caller classifies it as HTTP 413, and terminates
//! the stream with a credential-safe [`SseByteStreamError`]. rmcp lacks a
//! pin-project dependency here, so the per-event accounting is folded into a
//! single [`futures::stream::try_unfold`] rather than the upstream
//! `pin_project!` wrapper.

use std::sync::{Arc, OnceLock};

use futures::stream::{BoxStream, StreamExt as _};
use praxis_filter::StreamingResponseBody;
use sse_stream::{Error as SseError, Sse, SseStream};

use super::subrequest_transport::TransportSignal;

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
    /// The underlying subrequest body errored mid-stream.
    Upstream,
}

impl std::fmt::Display for SseByteStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ceiling { limit } => write!(f, "mcp sse stream exceeded the {limit}-byte ceiling"),
            Self::EventTooLarge { max_size } => write!(f, "mcp sse event exceeded the {max_size}-byte limit"),
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
#[allow(clippy::allow_attributes, dead_code, reason = "used by sse_stream_from_body wired in task 4")]
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
#[allow(clippy::allow_attributes, dead_code, reason = "used by sse_stream_from_body wired in task 4")]
struct ByteState {
    /// The praxis streaming response body being adapted.
    body: Box<dyn StreamingResponseBody>,
    /// Cumulative bytes emitted so far.
    emitted: usize,
    /// Total operation-stream byte ceiling.
    operation_cap: usize,
    /// Per-event size limiter.
    per_event: SseEventSizeLimiter,
    /// Out-of-band signal for recording `ResponseTooLarge`.
    signal: Arc<OnceLock<TransportSignal>>,
}

/// Adapt `body` into an rmcp SSE stream, enforcing both byte budgets.
///
/// `per_event_cap` bounds the retained size of any single SSE event;
/// `operation_cap` bounds the cumulative raw bytes across the whole stream;
/// `max_sse_event_size` is an outer per-event backstop (the effective per-event
/// cap is `min(per_event_cap, max_sse_event_size)`). A breach of either budget
/// records `signal` (first wins) and terminates the stream.
#[allow(clippy::allow_attributes, dead_code, reason = "wired by selector filter in task 4")]
#[expect(clippy::too_many_lines, reason = "two-budget byte-layer adapter is inherently sequential")]
pub(super) fn sse_stream_from_body(
    body: Box<dyn StreamingResponseBody>,
    per_event_cap: usize,
    operation_cap: usize,
    max_sse_event_size: usize,
    signal: Arc<OnceLock<TransportSignal>>,
) -> BoxStream<'static, Result<Sse, SseError>> {
    let effective_per_event = per_event_cap.min(max_sse_event_size);
    let state = ByteState {
        body,
        emitted: 0,
        operation_cap,
        per_event: SseEventSizeLimiter::new(effective_per_event),
        signal,
    };

    let byte_stream = futures::stream::try_unfold(state, |mut st| async move {
        loop {
            match st.body.next_chunk().await {
                Ok(Some(chunk)) => {
                    // Per-event (per-message) ceiling, before parsing.
                    if st.per_event.observe(&chunk).is_err() {
                        let limit = st.per_event.max_size;
                        st.signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                        return Err(SseByteStreamError::EventTooLarge { max_size: limit });
                    }
                    // Cumulative operation-stream ceiling.
                    st.emitted = st.emitted.saturating_add(chunk.len());
                    if st.emitted > st.operation_cap {
                        let limit = st.operation_cap;
                        st.signal.get_or_init(|| TransportSignal::ResponseTooLarge { limit });
                        return Err(SseByteStreamError::Ceiling { limit });
                    }
                    if chunk.is_empty() {
                        continue; // keep pulling; never yield an empty frame
                    }
                    return Ok(Some((chunk, st)));
                },
                Ok(None) => return Ok(None),
                Err(_error) => return Err(SseByteStreamError::Upstream),
            }
        }
    });

    SseStream::from_bytes_stream(byte_stream).boxed()
}

/// Test double for [`StreamingResponseBody`] that yields queued chunks and
/// records whether [`cancel`](StreamingResponseBody::cancel) ran.
#[cfg(test)]
pub(super) struct FakeStreamingBody {
    chunks: std::collections::VecDeque<bytes::Bytes>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    err_after: Option<usize>,
    yielded: usize,
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
            yielded: 0,
        }
    }

    pub(super) fn erroring_after<I>(chunks: I, cancelled: Arc<std::sync::atomic::AtomicBool>) -> Self
    where
        I: IntoIterator<Item = bytes::Bytes>,
    {
        let mut body = Self::from_chunks(chunks, cancelled);
        body.err_after = Some(body.chunks.len());
        body
    }

    #[expect(dead_code, reason = "probe for future cancellation tests")]
    pub(super) fn cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
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
            return Err(praxis_filter::FilterError::from("fake upstream body error".to_owned()));
        }
        match self.chunks.pop_front() {
            Some(chunk) => {
                self.yielded += 1;
                Ok(Some(chunk))
            },
            None => Ok(None),
        }
    }

    async fn suppress(&mut self) -> Result<(), praxis_filter::FilterError> {
        Ok(())
    }

    async fn cancel(&mut self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests use unwrap/expect/indexing for brevity")]
mod tests {
    use std::sync::{
        Arc, OnceLock,
        atomic::AtomicBool,
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
        let mut stream = sse_stream_from_body(body, 1_024, 4_096, 16 * 1024 * 1024, Arc::clone(&signal));

        let first = stream.next().await.expect("one event").expect("ok event");
        assert_eq!(first.data.as_deref(), Some("{\"jsonrpc\":\"2.0\"}"));
        assert!(stream.next().await.is_none(), "clean EOF");
        assert!(signal.get().is_none(), "no size signal on a clean stream");
    }

    #[tokio::test]
    async fn cumulative_ceiling_breach_errors_and_signals_413() {
        // Two 3-byte chunks exceed a 4-byte operation cap on the second chunk.
        let body = Box::new(FakeStreamingBody::from_chunks(
            [Bytes::from_static(b"aaa"), Bytes::from_static(b"bbb")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 4, 16 * 1024 * 1024, Arc::clone(&signal));

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
        let mut stream = sse_stream_from_body(body, 8, 1_000_000, 16 * 1024 * 1024, Arc::clone(&signal));

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
        let mut stream = sse_stream_from_body(body, 1_000_000, 1_000_000, 8, Arc::clone(&signal));
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "max_sse_event_size acts as a per-event backstop");
        assert!(matches!(signal.get(), Some(TransportSignal::ResponseTooLarge { limit: 8 })));
    }

    #[tokio::test]
    async fn upstream_body_error_surfaces_without_a_size_signal() {
        let body = Box::new(FakeStreamingBody::erroring_after(
            [Bytes::from_static(b"data: partial\n")],
            cancelled_flag(),
        ));
        let signal = Arc::new(OnceLock::new());
        let mut stream = sse_stream_from_body(body, 1_024, 1_024, 16 * 1024 * 1024, Arc::clone(&signal));
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_err = true;
            }
        }
        assert!(saw_err, "an upstream body error must surface");
        assert!(signal.get().is_none(), "a transport error is not a size breach");
    }
}
