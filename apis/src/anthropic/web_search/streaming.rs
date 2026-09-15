// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Terminal Anthropic Messages SSE streaming for the web-search loop.
//!
//! The buffered loop accumulates each backend round, classifies it, and either
//! loops (server-owned `WebSearch`) or returns the final message. Terminal
//! streaming keeps that control flow but delivers the client-visible response
//! incrementally: text content blocks are forwarded as they arrive across IRR
//! rounds, the managed `WebSearch` tool-use block is suppressed, and a single
//! coherent Anthropic Messages SSE lifecycle is presented to the client.
//!
//! One logical stream spans every IRR round:
//!
//! * `message_start` is forwarded once, from the first round, with a stable id.
//! * `content_block_*` frames for text (and client-owned tools) are forwarded with a running output-block-index offset
//!   so indices stay monotonic across rounds.
//! * The managed `WebSearch` `tool_use` block is suppressed entirely.
//! * Per-round `message_delta` / `message_stop` are deferred; a single terminal `message_delta` (aggregated
//!   `output_tokens`) and `message_stop` are emitted only when the loop finishes on a non-managed round.
//!
//! Every failure mode is fail-closed: a terminal `error` event is emitted and no
//! raw upstream bytes leak into the transformed stream.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use serde_json::{Value, json};

/// A parsed SSE event: its type and decoded JSON `data` payload.
#[derive(Debug)]
pub(super) struct SseEvent {
    /// Event type, taken from the payload's `type` field.
    pub(super) event_type: String,
    /// Decoded JSON `data` payload.
    pub(super) data: Value,
}

/// Fail-closed streaming error, mapped to a terminal Anthropic `error` event.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum StreamError {
    /// Upstream bytes were not valid UTF-8.
    MalformedUtf8,
    /// An SSE event — a retained partial or a single complete frame — exceeded
    /// the configured bound.
    OversizedPartialEvent,
    /// An SSE event was missing a required field (for example a block index).
    MalformedEvent,
    /// A managed `WebSearch` call was malformed (empty id or query).
    InvalidManagedCall,
    /// A managed `WebSearch` query exceeded the configured size bound.
    QueryTooLong,
    /// A `WebSearch` block was suppressed but the round did not classify as a
    /// managed call, so forwarding it would silently drop a tool call.
    SuppressionMismatch,
    /// The reconstructed round message exceeded the configured body-size bound.
    Oversized,
    /// Re-entering inference would exceed the router's iteration ceiling; the
    /// loop is terminated with a coherent error rather than an abrupt EOF.
    IterationLimit,
    /// Re-entering inference would exceed the router's deadline; opening the next
    /// round would fail, so the loop is terminated with a coherent error rather
    /// than an abrupt EOF.
    DeadlineExceeded,
    /// A later round returned a non-2xx status or a content-encoded body after
    /// the client-visible SSE lifecycle had already started, so it cannot be
    /// transformed and its raw bytes must not corrupt the open stream.
    UpstreamUnprocessable,
    /// The upstream stream ended cleanly before the round reported a stop reason
    /// (a truncated `message_delta`), so a successful terminal cannot be
    /// fabricated in its place.
    IncompleteStream,
    /// The upstream streaming source terminated abnormally mid-round — a
    /// transport fault, circuit trip, admission or idle timeout, or byte-ceiling
    /// breach reported by Praxis as a stream termination — so the round cannot
    /// complete and the loop fails closed rather than ending in an abrupt EOF.
    UpstreamTerminated,
}

/// Incremental native-Messages SSE parser for one backend round.
///
/// Owns its cross-chunk continuation buffers so it is self-contained and unit
/// testable without a filter context. A new parser is used for each round.
#[derive(Default)]
pub(super) struct SseParser {
    /// Trailing bytes of an incomplete UTF-8 sequence carried between chunks.
    utf8_tail: Vec<u8>,
    /// Trailing text of an incomplete SSE event block carried between chunks.
    ///
    /// Line endings are normalized to `\n` as bytes are appended, so this buffer
    /// only ever contains `\n`; a complete event ends at a `\n\n` boundary.
    line_buffer: String,
    /// A lone trailing `\r` withheld from `line_buffer` because it may be the
    /// first half of a `\r\n` split across chunks. Resolved on the next push (or
    /// at end-of-stream) into a single `\n`.
    pending_cr: bool,
}

impl SseParser {
    /// Feed one response chunk, returning every complete SSE event it yields.
    ///
    /// Partial UTF-8 sequences and partial event blocks are buffered until the
    /// next chunk completes them. `max_partial_event_bytes` bounds the retained
    /// partial-event buffer.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::MalformedUtf8`] when the combined bytes contain an
    /// invalid (not merely incomplete) UTF-8 sequence, or when a trailing
    /// incomplete sequence survives end-of-stream. Returns
    /// [`StreamError::OversizedPartialEvent`] when the retained partial event
    /// exceeds `max_partial_event_bytes`.
    pub(super) fn push(
        &mut self,
        chunk: &[u8],
        end_of_stream: bool,
        max_partial_event_bytes: usize,
    ) -> Result<Vec<SseEvent>, StreamError> {
        let text = self.decode_utf8(chunk, end_of_stream)?;

        // Append the newly decoded text onto the retained partial in place,
        // normalizing line endings as it goes, instead of rebuilding the whole
        // buffer each chunk (which was quadratic across a long fragmented event).
        let prev_len = self.line_buffer.len();
        self.normalize_into_buffer(&text, end_of_stream);

        // A `\n\n` terminator can only straddle the old/new boundary when the
        // last retained byte was already a `\n`; every other complete event lies
        // entirely within the appended region. Resume the scan there, never from
        // the start, so total work stays linear in the buffer length.
        let starts_on_prior_newline = prev_len > 0 && self.line_buffer.as_bytes().get(prev_len - 1) == Some(&b'\n');
        let mut from = prev_len - usize::from(starts_on_prior_newline);

        let mut events = Vec::new();
        let mut consumed = 0;
        while let Some(offset) = self.line_buffer.get(from..).and_then(|tail| tail.find("\n\n")) {
            let end = from + offset;
            let Some(block) = self.line_buffer.get(consumed..end) else {
                break;
            };
            // Bound each complete event by the same cap as the retained partial,
            // so a single huge frame cannot be materialized in full and bypass the
            // body-size bound.
            if block.len() > max_partial_event_bytes {
                return Err(StreamError::OversizedPartialEvent);
            }
            if let Some(event) = parse_event_block(block)? {
                events.push(event);
            }
            consumed = end + 2;
            from = consumed;
        }

        // The retained partial (a deferred `\r` lives in `pending_cr`, not here)
        // is bounded too.
        if self.line_buffer.len() - consumed > max_partial_event_bytes {
            return Err(StreamError::OversizedPartialEvent);
        }
        self.line_buffer.drain(..consumed);
        Ok(events)
    }

    /// Append `text` to `line_buffer`, normalizing `\r\n` and lone `\r` to `\n`.
    ///
    /// A trailing `\r` is withheld in `pending_cr` (unless `end_of_stream`) so a
    /// `\r\n` split across chunks collapses to one `\n` rather than two.
    fn normalize_into_buffer(&mut self, text: &str, end_of_stream: bool) {
        if !self.pending_cr && !text.contains('\r') {
            // Hot path: no carriage returns to fold, so append in a single copy.
            self.line_buffer.push_str(text);
            return;
        }
        for ch in text.chars() {
            if self.pending_cr {
                self.pending_cr = false;
                self.line_buffer.push('\n');
                if ch == '\n' {
                    // `\r\n` already folded to a single `\n`; drop this `\n`.
                    continue;
                }
            }
            if ch == '\r' {
                self.pending_cr = true;
            } else {
                self.line_buffer.push(ch);
            }
        }
        if end_of_stream && self.pending_cr {
            // No further chunk can complete a `\r\n`; emit the lone `\r` as `\n`.
            self.pending_cr = false;
            self.line_buffer.push('\n');
        }
    }

    /// Combine any retained UTF-8 tail with `chunk` and return the valid prefix.
    ///
    /// A trailing incomplete sequence (fewer than four bytes) is retained for
    /// the next chunk; a genuinely invalid sequence, or an incomplete sequence
    /// at end-of-stream, fails closed.
    ///
    /// When no tail is retained (the common case) the valid prefix borrows
    /// `chunk` directly, so a well-formed chunk is validated once and reaches
    /// `line_buffer` in a single copy with no intermediate heap allocation.
    /// Only joining a retained tail with `chunk` — the rare split-code-point
    /// case — needs one owned buffer, which is handed back by value (moved, not
    /// copied) so no further copy follows.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::MalformedUtf8`] on genuinely invalid UTF-8, on an
    /// incomplete sequence at end-of-stream, or on a retained tail longer than a
    /// single truncated code point.
    fn decode_utf8<'chunk>(
        &mut self,
        chunk: &'chunk [u8],
        end_of_stream: bool,
    ) -> Result<Cow<'chunk, str>, StreamError> {
        if self.utf8_tail.is_empty() {
            // Hot path: with no retained tail the valid prefix borrows `chunk`,
            // so there is no intermediate `String` to allocate and drop.
            return match std::str::from_utf8(chunk) {
                Ok(text) => Ok(Cow::Borrowed(text)),
                Err(error) if error.error_len().is_none() => {
                    let valid = self.retain_incomplete_tail(chunk, error.valid_up_to(), end_of_stream)?;
                    Ok(Cow::Borrowed(
                        std::str::from_utf8(chunk.get(..valid).ok_or(StreamError::MalformedUtf8)?)
                            .map_err(|_prefix_not_utf8| StreamError::MalformedUtf8)?,
                    ))
                },
                Err(_) => Err(StreamError::MalformedUtf8),
            };
        }
        // A code point split across the previous chunk boundary: joining the
        // retained tail with `chunk` needs one owned buffer, returned by value
        // so the decoded text is never copied a second time.
        let mut pending = std::mem::take(&mut self.utf8_tail);
        pending.extend_from_slice(chunk);
        let valid = match std::str::from_utf8(&pending) {
            Ok(_) => pending.len(),
            Err(error) if error.error_len().is_none() => {
                self.retain_incomplete_tail(&pending, error.valid_up_to(), end_of_stream)?
            },
            Err(_) => return Err(StreamError::MalformedUtf8),
        };
        pending.truncate(valid);
        Ok(Cow::Owned(
            String::from_utf8(pending).map_err(|_combined_not_utf8| StreamError::MalformedUtf8)?,
        ))
    }

    /// Retain the trailing incomplete UTF-8 sequence of `bytes` (the run after
    /// `valid`) for the next chunk and return `valid`.
    ///
    /// # Errors
    ///
    /// Fails closed at end-of-stream (no later chunk can complete the sequence)
    /// and when the retained tail is longer than a single truncated code point
    /// (at most three bytes), which is not a mere split.
    fn retain_incomplete_tail(
        &mut self,
        bytes: &[u8],
        valid: usize,
        end_of_stream: bool,
    ) -> Result<usize, StreamError> {
        if end_of_stream {
            return Err(StreamError::MalformedUtf8);
        }
        let tail = bytes.get(valid..).ok_or(StreamError::MalformedUtf8)?;
        if tail.len() > 3 {
            return Err(StreamError::MalformedUtf8);
        }
        self.utf8_tail.extend_from_slice(tail);
        Ok(valid)
    }
}

/// Parse one SSE event block into an [`SseEvent`], reading the JSON `data`.
///
/// Returns `Ok(None)` for a block that carries no `data` field (for example a
/// bare comment or keep-alive) or whose sole `data` payload is the `[DONE]`
/// sentinel, matching the SSE specification's handling of `data`, `data:value`,
/// and `data: value` field forms.
///
/// # Errors
///
/// Returns [`StreamError::MalformedEvent`] when a `data` payload is present but
/// is not a JSON object carrying a string `type`. Such a block is malformed
/// framing: silently dropping it would truncate the client-visible stream and
/// could turn a real failure into a successful-but-incomplete response, so the
/// caller fails closed instead.
fn parse_event_block(block: &str) -> Result<Option<SseEvent>, StreamError> {
    let Some(data) = collect_data_field(block) else {
        return Ok(None);
    };
    // The OpenAI-style `[DONE]` sentinel is not Anthropic JSON; skip it rather
    // than fail so a backend that emits it does not poison the stream.
    if data == "[DONE]" {
        return Ok(None);
    }
    let data: Value = serde_json::from_str(&data).map_err(|_invalid_json| StreamError::MalformedEvent)?;
    let event_type = data
        .get("type")
        .and_then(Value::as_str)
        .ok_or(StreamError::MalformedEvent)?
        .to_owned();
    Ok(Some(SseEvent { event_type, data }))
}

/// Concatenate an SSE block's `data` field lines, or `None` if it carries none.
///
/// Joins multiple `data:` lines with a newline per the SSE specification and
/// accepts the `data`, `data:value`, and `data: value` field forms.
fn collect_data_field(block: &str) -> Option<String> {
    let mut data: Option<String> = None;
    for line in block.lines() {
        let field = if line == "data" {
            Some("")
        } else {
            line.strip_prefix("data:")
                .map(|value| value.strip_prefix(' ').unwrap_or(value))
        };
        if let Some(field) = field {
            match &mut data {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(field);
                },
                None => data = Some(field.to_owned()),
            }
        }
    }
    data
}

// -----------------------------------------------------------------------------
// Cross-round logical stream
// -----------------------------------------------------------------------------

/// One round's incremental reconstruction and forwarding bookkeeping.
#[derive(Default)]
struct RoundState {
    /// Base message object captured from this round's `message_start`.
    message: Option<Value>,
    /// Reconstructed content blocks, dense by backend block index.
    content: Vec<Value>,
    /// Accumulated text per backend block index, materialized once at reconstruct.
    partial_text: HashMap<usize, String>,
    /// Accumulated tool `input_json` text per backend block index.
    partial_json: HashMap<usize, String>,
    /// Accumulated extended-thinking text per backend block index.
    partial_thinking: HashMap<usize, String>,
    /// Accumulated thinking-block signature per backend block index.
    partial_signature: HashMap<usize, String>,
    /// Backend block index -> client output index for forwarded blocks.
    forwarded: HashMap<usize, u64>,
    /// Backend index of a suppressed managed `WebSearch` block, if any.
    suppressed_web_search: Option<usize>,
    /// Stop reason captured from this round's `message_delta`.
    stop_reason: Option<String>,
    /// Output tokens captured from this round's `message_delta`.
    round_output_tokens: u64,
    /// Input tokens captured from this round's `message_start` / `message_delta`.
    round_input_tokens: Option<u64>,
    /// Cache-read input tokens captured from this round's usage.
    round_cache_read: Option<u64>,
    /// Cumulative bytes of streamed text/tool-input fragments this round.
    ///
    /// Bounds attacker-controlled fragment growth before the round message is
    /// reconstructed, independent of the final serialized-size check.
    accumulated_bytes: usize,
    /// Whether this round's closing `message_stop` frame arrived.
    ///
    /// A clean transport EOF can still leave the Messages lifecycle incomplete;
    /// without `message_stop` the round was truncated and must fail closed.
    message_stopped: bool,
    /// Backend indices of content blocks started but not yet closed.
    ///
    /// Tracked by identity, not a bare count: a stray or duplicate
    /// `content_block_stop` must not silently cancel a different genuinely-open
    /// block. A non-empty set at round end means a block was left open by a
    /// truncated stream, so the round must fail closed rather than fabricate
    /// success.
    open_blocks: HashSet<usize>,
}

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

/// A coherent Anthropic Messages SSE lifecycle assembled across IRR rounds.
pub(super) struct LogicalStream {
    /// Per-round native-Messages SSE parser.
    parser: SseParser,
    /// Maximum reconstructed round-message size before failing closed.
    max_body_bytes: usize,
    /// Maximum retained partial-event bytes for the SSE parser.
    max_partial_event_bytes: usize,
    /// Whether the client-visible `message_start` has been emitted.
    message_started: bool,
    /// Next client-visible content-block index to assign (monotonic across rounds).
    next_output_index: u64,
    /// Aggregated output tokens across all rounds, for the terminal `message_delta`.
    output_tokens_total: u64,
    /// Aggregated input tokens across rounds, `None` until any round reports usage.
    input_tokens_total: Option<u64>,
    /// Aggregated cache-read input tokens across rounds, `None` until reported.
    cache_read_total: Option<u64>,
    /// Reconstructed managed-round message bytes retained for IRR re-entry.
    ///
    /// A streamed round leaves the IRR buffered response body empty, so re-entry
    /// recovers the pending search from this reconstruction instead.
    reconstructed: Option<Vec<u8>>,
    /// Whether the stream has failed closed; once set, no more bytes are forwarded.
    failed: bool,
    /// Current round reconstruction and forwarding state.
    round: RoundState,
}

impl LogicalStream {
    /// Create a logical stream bounded by the loop's body and partial-event caps.
    pub(super) fn new(max_body_bytes: usize, max_partial_event_bytes: usize) -> Self {
        Self {
            parser: SseParser::default(),
            max_body_bytes,
            max_partial_event_bytes,
            message_started: false,
            next_output_index: 0,
            output_tokens_total: 0,
            input_tokens_total: None,
            cache_read_total: None,
            reconstructed: None,
            failed: false,
            round: RoundState::default(),
        }
    }

    /// Whether the stream has failed closed and forwards nothing further.
    pub(super) fn has_failed(&self) -> bool {
        self.failed
    }

    /// Whether the client-visible `message_start` has already been forwarded.
    ///
    /// Once true the client is mid-stream on a committed 200 SSE lifecycle, so a
    /// later untransformable round must fail closed rather than pass raw bytes
    /// through and corrupt the open stream.
    pub(super) fn has_started(&self) -> bool {
        self.message_started
    }

    /// Poison the stream so subsequent chunks forward nothing further.
    ///
    /// The caller emits the single terminal `error` event; this only records that
    /// the stream has failed closed.
    pub(super) fn fail(&mut self) {
        self.failed = true;
    }

    /// Take the reconstructed managed-round message bytes retained for re-entry.
    pub(super) fn take_reconstructed(&mut self) -> Option<Vec<u8>> {
        self.reconstructed.take()
    }

    /// Reset per-round state before a resumed inference round.
    pub(super) fn begin_round(&mut self) {
        self.parser = SseParser::default();
        self.round = RoundState::default();
    }

    /// Process one response chunk, returning client-visible forwarded bytes.
    ///
    /// Once the stream has failed closed, further chunks are dropped so exactly
    /// one terminal `error` event reaches the client.
    ///
    /// # Errors
    ///
    /// Returns a [`StreamError`] on malformed upstream framing or a suppression
    /// invariant violation, so the caller can fail closed. The stream is then
    /// poisoned: subsequent calls yield nothing.
    pub(super) fn on_chunk(&mut self, chunk: &[u8], end_of_stream: bool) -> Result<Vec<u8>, StreamError> {
        if self.failed {
            return Ok(Vec::new());
        }
        self.process_chunk(chunk, end_of_stream)
            .inspect_err(|_| self.failed = true)
    }

    /// Parse and forward one chunk without the fail-closed guard.
    fn process_chunk(&mut self, chunk: &[u8], end_of_stream: bool) -> Result<Vec<u8>, StreamError> {
        let events = self.parser.push(chunk, end_of_stream, self.max_partial_event_bytes)?;
        let mut output = Vec::new();
        for event in events {
            // A forwarded upstream `error` poisons the stream; stop processing so
            // no further frames follow the terminal error.
            if self.failed {
                break;
            }
            self.process_event(&event, &mut output)?;
        }
        Ok(output)
    }

    /// Reconstruct and forward one parsed SSE event.
    fn process_event(&mut self, event: &SseEvent, output: &mut Vec<u8>) -> Result<(), StreamError> {
        match event.event_type.as_str() {
            "message_start" => {
                self.on_message_start(event, output);
                Ok(())
            },
            "content_block_start" => self.on_block_start(event, output),
            "content_block_delta" => self.on_block_delta(event, output),
            "content_block_stop" => self.on_block_stop(event, output),
            "message_delta" => {
                self.on_message_delta(event);
                Ok(())
            },
            // Keep-alives are forwarded so the client connection stays live while
            // intermediate rounds run.
            "ping" => {
                emit_event(output, "ping", &event.data);
                Ok(())
            },
            // An upstream `error` after HTTP 200 is a real backend failure: forward
            // it verbatim and poison the stream so `finish_round` never fabricates a
            // synthetic success terminal in its place.
            "error" => {
                emit_event(output, "error", &event.data);
                self.failed = true;
                Ok(())
            },
            // `message_stop` closes the round's Messages lifecycle. It is not
            // forwarded per round (the terminal lifecycle is emitted once by
            // `finish_round`), but it is recorded so a round truncated before it
            // arrives fails closed instead of fabricating a successful terminal.
            "message_stop" => {
                self.round.message_stopped = true;
                Ok(())
            },
            // Every other unmodeled event is deferred without forwarding.
            _ => Ok(()),
        }
    }

    /// Capture the base message and forward the client-visible `message_start`.
    fn on_message_start(&mut self, event: &SseEvent, output: &mut Vec<u8>) {
        if let Some(message) = event.data.get("message") {
            if let Some(usage) = message.get("usage") {
                self.capture_round_usage(usage);
            }
            self.round.message = Some(message.clone());
        }
        if !self.message_started {
            self.message_started = true;
            emit_event(output, "message_start", &event.data);
        }
    }

    /// Capture per-round input and cache-read token counts from a usage object.
    ///
    /// The final `message_delta` may repeat these fields; the last value seen
    /// wins for the round, and rounds are summed in [`Self::finish_round`].
    fn capture_round_usage(&mut self, usage: &Value) {
        if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
            self.round.round_input_tokens = Some(input);
        }
        if let Some(cache_read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
            self.round.round_cache_read = Some(cache_read);
        }
    }

    /// Assign a client output index to a content block and forward its start.
    ///
    /// The managed `WebSearch` `tool_use` block is suppressed entirely: it is
    /// neither forwarded nor assigned a client output index, so its subsequent
    /// delta and stop frames are dropped as well.
    fn on_block_start(&mut self, event: &SseEvent, output: &mut Vec<u8>) -> Result<(), StreamError> {
        let index = block_index(&event.data)?;
        // Content blocks are dense and sequential in the Anthropic wire format. A
        // backend-controlled index that skips ahead would force an unbounded
        // sparse allocation, so a non-sequential index fails closed instead.
        if index != self.round.content.len() {
            return Err(StreamError::MalformedEvent);
        }
        // Reconstruct every block, including the suppressed managed call, so the
        // finished message can be classified exactly as the buffered loop would.
        let block = event.data.get("content_block").cloned().unwrap_or(Value::Null);
        self.round.content.push(block);
        // Track every started block by index (suppressed calls included) so a
        // block left open by a truncated stream is detected at round end. The
        // index is fresh: it was validated equal to the pre-push content length.
        self.round.open_blocks.insert(index);
        if is_web_search_tool_use(&event.data) {
            self.round.suppressed_web_search = Some(index);
            return Ok(());
        }
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        self.round.forwarded.insert(index, output_index);
        emit_remapped(output, "content_block_start", &event.data, output_index);
        Ok(())
    }

    /// Reconstruct and, for forwarded blocks, forward a content-block delta.
    fn on_block_delta(&mut self, event: &SseEvent, output: &mut Vec<u8>) -> Result<(), StreamError> {
        let index = block_index(&event.data)?;
        self.accumulate_delta(index, &event.data)?;
        if let Some(&output_index) = self.round.forwarded.get(&index) {
            emit_remapped(output, "content_block_delta", &event.data, output_index);
        }
        Ok(())
    }

    /// Finalize text and tool input and, for forwarded blocks, forward a stop.
    fn on_block_stop(&mut self, event: &SseEvent, output: &mut Vec<u8>) -> Result<(), StreamError> {
        let index = block_index(&event.data)?;
        // A stop must close a currently-open block. A stray (never-started) or
        // duplicate stop is rejected so it cannot silently cancel a different
        // genuinely-open block and mask a truncated round from the completeness
        // gate. Reject before any materialize/forward side effects.
        if !self.round.open_blocks.remove(&index) {
            return Err(StreamError::MalformedEvent);
        }
        materialize_string_field(&mut self.round.content, index, "text", &mut self.round.partial_text);
        materialize_string_field(
            &mut self.round.content,
            index,
            "thinking",
            &mut self.round.partial_thinking,
        );
        materialize_string_field(
            &mut self.round.content,
            index,
            "signature",
            &mut self.round.partial_signature,
        );
        if let Some(fragment) = self.round.partial_json.remove(&index)
            && let Ok(input) = serde_json::from_str::<Value>(&fragment)
            && let Some(block) = self.round.content.get_mut(index)
        {
            block["input"] = input;
        }
        if let Some(&output_index) = self.round.forwarded.get(&index) {
            emit_remapped(output, "content_block_stop", &event.data, output_index);
        }
        Ok(())
    }

    /// Capture the round's stop reason and output-token count; do not forward.
    fn on_message_delta(&mut self, event: &SseEvent) {
        if let Some(stop_reason) = event
            .data
            .get("delta")
            .and_then(|delta| delta.get("stop_reason"))
            .and_then(Value::as_str)
        {
            self.round.stop_reason = Some(stop_reason.to_owned());
        }
        if let Some(usage) = event.data.get("usage") {
            if let Some(tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
                self.round.round_output_tokens = tokens;
            }
            self.capture_round_usage(usage);
        }
    }

    /// Accumulate one content-block delta into a per-index buffer.
    ///
    /// Text and tool-input fragments are appended to mutable buffers and
    /// materialized once at block stop, so reconstruction stays linear rather
    /// than recopying the accumulated content on every delta.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::Oversized`] once the cumulative fragment bytes for
    /// the round exceed the configured body-size bound, failing closed before the
    /// round message is reconstructed.
    fn accumulate_delta(&mut self, index: usize, data: &Value) -> Result<(), StreamError> {
        let Some(delta) = data.get("delta") else {
            return Ok(());
        };
        // Each streamed delta carries its fragment under a type-specific field.
        // Text and tool-input reconstruct the client-visible content; thinking
        // and signature reconstruct the extended-thinking block so the assistant
        // turn replayed on re-entry keeps its verifiable signature.
        let Some(delta_type) = delta.get("type").and_then(Value::as_str) else {
            return Ok(());
        };
        let field = match delta_type {
            "text_delta" => "text",
            "input_json_delta" => "partial_json",
            "thinking_delta" => "thinking",
            "signature_delta" => "signature",
            _ => return Ok(()),
        };
        let Some(fragment) = delta.get(field).and_then(Value::as_str) else {
            return Ok(());
        };
        // Charge before taking the target borrow so the running budget stays a
        // single mutable borrow of `self`.
        self.charge_fragment_bytes(fragment.len())?;
        let bucket = match delta_type {
            "text_delta" => &mut self.round.partial_text,
            "input_json_delta" => &mut self.round.partial_json,
            "thinking_delta" => &mut self.round.partial_thinking,
            "signature_delta" => &mut self.round.partial_signature,
            _ => return Ok(()),
        };
        bucket.entry(index).or_default().push_str(fragment);
        Ok(())
    }

    /// Charge `bytes` against the round's cumulative fragment budget.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::Oversized`] when the running total exceeds the
    /// configured body-size bound.
    fn charge_fragment_bytes(&mut self, bytes: usize) -> Result<(), StreamError> {
        self.round.accumulated_bytes = self.round.accumulated_bytes.saturating_add(bytes);
        if self.round.accumulated_bytes > self.max_body_bytes {
            return Err(StreamError::Oversized);
        }
        Ok(())
    }

    /// Reconstruct this round's Messages JSON from the accumulated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::MalformedEvent`] when the captured `message_start`
    /// object is missing or serialization fails, or [`StreamError::Oversized`]
    /// when the reconstructed message exceeds the configured body-size bound.
    fn reconstruct_round_message(&mut self) -> Result<Vec<u8>, StreamError> {
        // Materialize any fragments whose `content_block_stop` never arrived
        // (a truncated block) so the reconstruction is faithful for re-entry.
        self.flush_pending_fragments();
        let mut message = self.round.message.take().ok_or(StreamError::MalformedEvent)?;
        let object = message.as_object_mut().ok_or(StreamError::MalformedEvent)?;
        object.insert(
            "content".to_owned(),
            Value::Array(std::mem::take(&mut self.round.content)),
        );
        if let Some(stop_reason) = &self.round.stop_reason {
            object.insert("stop_reason".to_owned(), Value::from(stop_reason.as_str()));
        }
        match object.get_mut("usage").and_then(Value::as_object_mut) {
            Some(usage) => {
                usage.insert("output_tokens".to_owned(), Value::from(self.round.round_output_tokens));
            },
            None => {
                object.insert(
                    "usage".to_owned(),
                    json!({ "output_tokens": self.round.round_output_tokens }),
                );
            },
        }
        let bytes = serde_json::to_vec(&message).map_err(|_serialization_failed| StreamError::MalformedEvent)?;
        if bytes.len() > self.max_body_bytes {
            return Err(StreamError::Oversized);
        }
        Ok(bytes)
    }

    /// Materialize any text/thinking/tool-input fragments still pending at reconstruct.
    fn flush_pending_fragments(&mut self) {
        flush_string_fragments(&mut self.round.content, "text", &mut self.round.partial_text);
        flush_string_fragments(&mut self.round.content, "thinking", &mut self.round.partial_thinking);
        flush_string_fragments(&mut self.round.content, "signature", &mut self.round.partial_signature);
        for (index, fragment) in std::mem::take(&mut self.round.partial_json) {
            if let Ok(input) = serde_json::from_str::<Value>(&fragment)
                && let Some(block) = self.round.content.get_mut(index)
            {
                block["input"] = input;
            }
        }
    }

    /// Reconstruct the round's message, classify it, and select the loop action.
    ///
    /// A managed `WebSearch` call yields [`RoundAction::Loop`] with no terminal
    /// bytes; a final answer yields [`RoundAction::Done`] with the terminal
    /// `message_delta` / `message_stop` lifecycle.
    ///
    /// # Errors
    ///
    /// Returns a [`StreamError`] when the reconstructed message is malformed or
    /// carries an invalid managed call, so the caller can fail closed.
    pub(super) fn finish_round(&mut self) -> Result<FinishOutcome, StreamError> {
        // A forwarded upstream `error` already reached the client: end the loop
        // without reconstructing or fabricating a synthetic success terminal.
        if self.failed {
            return Ok(FinishOutcome {
                action: RoundAction::Done,
                terminal: Vec::new(),
            });
        }
        let bytes = self.reconstruct_round_message()?;
        self.output_tokens_total = self.output_tokens_total.saturating_add(self.round.round_output_tokens);
        self.aggregate_round_usage();
        match super::classify_response(&bytes) {
            super::ResponseDecision::Managed(_) => self.finish_managed_round(bytes),
            // A suppressed WebSearch block on a non-managed round would silently
            // drop a tool call from the client-visible stream: fail closed.
            super::ResponseDecision::Done if self.round.suppressed_web_search.is_some() => {
                Err(StreamError::SuppressionMismatch)
            },
            super::ResponseDecision::Done => self.finish_terminal_round(),
            super::ResponseDecision::InvalidManagedCall => Err(StreamError::InvalidManagedCall),
            super::ResponseDecision::QueryTooLong => Err(StreamError::QueryTooLong),
        }
    }

    /// Complete a managed round: verify the full lifecycle arrived, then retain
    /// the reconstruction so re-entry can recover the pending search (the
    /// streamed round leaves the IRR response body empty). A managed round is a
    /// complete Messages response too, so re-entering on a truncated one would
    /// replay a partial assistant turn.
    fn finish_managed_round(&mut self, bytes: Vec<u8>) -> Result<FinishOutcome, StreamError> {
        self.require_complete_lifecycle()?;
        self.reconstructed = Some(bytes);
        Ok(FinishOutcome {
            action: RoundAction::Loop,
            terminal: Vec::new(),
        })
    }

    /// Complete the terminal round: verify the full lifecycle arrived (stop
    /// reason, every block closed, and `message_stop`), then emit the aggregated
    /// terminal `message_delta` / `message_stop`. A premature EOF that skips any
    /// of these must not fabricate a successful `end_turn` and misreport a
    /// truncated round as complete.
    fn finish_terminal_round(&self) -> Result<FinishOutcome, StreamError> {
        self.require_complete_lifecycle()?;
        let stop_reason = self.round.stop_reason.clone().ok_or(StreamError::IncompleteStream)?;
        Ok(FinishOutcome {
            action: RoundAction::Done,
            terminal: self.build_terminal(&stop_reason),
        })
    }

    /// Fail closed unless this round delivered a complete Messages lifecycle.
    ///
    /// A clean round carries a stop reason (from `message_delta`), closes every
    /// content block with `content_block_stop`, and ends with `message_stop`. A
    /// clean transport EOF can still leave any of these missing; such a round is
    /// truncated and must not be reconstructed into a successful terminal or
    /// re-entered as if complete.
    fn require_complete_lifecycle(&self) -> Result<(), StreamError> {
        if self.round.stop_reason.is_some() && self.round.message_stopped && self.round.open_blocks.is_empty() {
            Ok(())
        } else {
            Err(StreamError::IncompleteStream)
        }
    }

    /// Fold this round's input and cache-read tokens into the running totals.
    fn aggregate_round_usage(&mut self) {
        if let Some(input) = self.round.round_input_tokens {
            self.input_tokens_total = Some(self.input_tokens_total.unwrap_or(0).saturating_add(input));
        }
        if let Some(cache_read) = self.round.round_cache_read {
            self.cache_read_total = Some(self.cache_read_total.unwrap_or(0).saturating_add(cache_read));
        }
    }

    /// Build the terminal `message_delta` and `message_stop` frames.
    ///
    /// The delta and its usage are serialized through the shared Anthropic wire
    /// types so the schema (container, stop details, and every usage field) is
    /// identical to the non-streaming translator, and usage is aggregated across
    /// every IRR round rather than reporting only the final round's output.
    fn build_terminal(&self, stop_reason: &str) -> Vec<u8> {
        let usage = super::super::wire::MessageDeltaUsage::new(
            self.output_tokens_total,
            self.input_tokens_total,
            self.cache_read_total,
        );
        let mut out = Vec::new();
        emit_event(
            &mut out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "container": null,
                    "stop_details": null,
                    "stop_reason": stop_reason,
                    "stop_sequence": null
                },
                "usage": usage
            }),
        );
        emit_event(&mut out, "message_stop", &json!({"type": "message_stop"}));
        out
    }
}

/// Whether a `content_block_start` payload begins a managed `WebSearch` call.
fn is_web_search_tool_use(data: &Value) -> bool {
    let block = &data["content_block"];
    block.get("type").and_then(Value::as_str) == Some("tool_use")
        && block.get("name").and_then(Value::as_str) == Some("WebSearch")
}

/// Materialize one index's pending string fragment into a content-block field.
///
/// `content` and `buffer` are passed as disjoint borrows so the caller keeps a
/// single mutable borrow of the surrounding round state.
fn materialize_string_field(content: &mut [Value], index: usize, field: &str, buffer: &mut HashMap<usize, String>) {
    if let Some(value) = buffer.remove(&index)
        && let Some(block) = content.get_mut(index)
    {
        block[field] = Value::from(value);
    }
}

/// Materialize every remaining pending string fragment into its content block.
fn flush_string_fragments(content: &mut [Value], field: &str, buffer: &mut HashMap<usize, String>) {
    for (index, value) in std::mem::take(buffer) {
        if let Some(block) = content.get_mut(index) {
            block[field] = Value::from(value);
        }
    }
}

/// Read the backend content-block `index` from an SSE event payload.
fn block_index(data: &Value) -> Result<usize, StreamError> {
    data.get("index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or(StreamError::MalformedEvent)
}

/// Emit an SSE event with its content-block `index` remapped to the client index.
fn emit_remapped(output: &mut Vec<u8>, event_type: &str, data: &Value, output_index: u64) {
    let mut data = data.clone();
    if let Some(object) = data.as_object_mut() {
        object.insert("index".to_owned(), Value::from(output_index));
    }
    emit_event(output, event_type, &data);
}

/// Encode a fail-closed terminal Anthropic `error` SSE event for a stream error.
///
/// The Anthropic error envelope is carried verbatim from [`super::super::wire`]
/// so no raw upstream bytes leak into the transformed stream.
pub(super) fn error_event_bytes(error: &StreamError) -> Vec<u8> {
    let (error_type, message) = anthropic_error(error);
    let body = super::super::wire::error_body(error_type, message, None);
    let mut out = Vec::with_capacity(body.len() + 20);
    out.extend_from_slice(b"event: error\ndata: ");
    out.extend_from_slice(&body);
    out.extend_from_slice(b"\n\n");
    out
}

/// Map a [`StreamError`] to an Anthropic error type and client-safe message.
fn anthropic_error(error: &StreamError) -> (&'static str, &'static str) {
    match error {
        StreamError::InvalidManagedCall => (
            "invalid_request_error",
            "WebSearch tool use requires a non-empty id and input.query",
        ),
        StreamError::QueryTooLong => (
            "invalid_request_error",
            "WebSearch input.query must not exceed 8192 bytes",
        ),
        StreamError::IterationLimit => (
            "api_error",
            "web search exceeded the maximum number of search iterations",
        ),
        StreamError::DeadlineExceeded => ("api_error", "web search exceeded the configured deadline"),
        StreamError::IncompleteStream => ("api_error", "web search response stream ended before completion"),
        StreamError::UpstreamTerminated => ("api_error", "web search stream terminated before completion"),
        StreamError::MalformedUtf8
        | StreamError::MalformedEvent
        | StreamError::OversizedPartialEvent
        | StreamError::Oversized
        | StreamError::SuppressionMismatch
        | StreamError::UpstreamUnprocessable => ("api_error", "web search stream could not be processed"),
    }
}

/// Encode one canonical single-line Anthropic SSE event.
fn emit_event(output: &mut Vec<u8>, event_type: &str, data: &Value) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    output.extend_from_slice(data.to_string().as_bytes());
    output.extend_from_slice(b"\n\n");
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
    use serde_json::json;

    use super::*;

    const MAX_PARTIAL: usize = 64 * 1024;
    const MAX_BODY: usize = 1 << 20;

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
    fn parses_one_complete_event() {
        let mut parser = SseParser::default();
        let chunk = b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n";

        let events = parser.push(chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[0].data["type"], "message_start");
    }

    #[test]
    fn buffers_partial_event_across_chunks() {
        let mut parser = SseParser::default();

        let first = parser
            .push(
                b"event: content_block_delta\ndata: {\"type\":\"conten",
                false,
                MAX_PARTIAL,
            )
            .unwrap();
        assert!(first.is_empty(), "an incomplete event must yield nothing yet");

        let second = parser
            .push(b"t_block_delta\",\"index\":0}\n\n", false, MAX_PARTIAL)
            .unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].event_type, "content_block_delta");
        assert_eq!(second[0].data["index"], 0);
    }

    #[test]
    fn buffers_incomplete_utf8_across_chunks() {
        let mut parser = SseParser::default();
        // "é" is 0xC3 0xA9; split the two bytes across chunks inside the JSON.
        let mut first_chunk = b"data: {\"type\":\"x\",\"t\":\"".to_vec();
        first_chunk.push(0xC3);

        let first = parser.push(&first_chunk, false, MAX_PARTIAL).unwrap();
        assert!(first.is_empty());

        let mut second_chunk = vec![0xA9];
        second_chunk.extend_from_slice(b"\"}\n\n");
        let second = parser.push(&second_chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].data["t"], "é");
    }

    #[test]
    fn buffers_four_byte_utf8_split_one_byte_per_chunk() {
        // A 4-byte code point (😀 = 0xF0 0x9F 0x98 0x80) delivered one byte per
        // chunk exercises the retained-tail path re-stashing a *still-incomplete*
        // sequence across successive chunks, not just a single 2-byte split: the
        // first byte stashes with an empty tail, the second and third re-stash a
        // non-empty tail that is not yet a full code point, and the fourth
        // completes it.
        let mut parser = SseParser::default();
        let prefix = b"data: {\"type\":\"x\",\"t\":\"";
        let suffix = b"\"}\n\n";

        assert!(parser.push(prefix, false, MAX_PARTIAL).unwrap().is_empty());
        for byte in "😀".as_bytes() {
            assert!(
                parser.push(&[*byte], false, MAX_PARTIAL).unwrap().is_empty(),
                "an incomplete code point yields no event until its final byte arrives",
            );
        }
        let events = parser.push(suffix, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data["t"], "😀");
    }

    #[test]
    fn rejects_invalid_utf8() {
        let mut parser = SseParser::default();
        // 0xFF is never valid in UTF-8 (a genuine error, not a truncation).
        let error = parser.push(&[0xFF, 0xFE], false, MAX_PARTIAL).unwrap_err();
        assert_eq!(error, StreamError::MalformedUtf8);
    }

    #[test]
    fn rejects_incomplete_utf8_at_end_of_stream() {
        let mut parser = SseParser::default();
        let error = parser.push(&[0xC3], true, MAX_PARTIAL).unwrap_err();
        assert_eq!(error, StreamError::MalformedUtf8);
    }

    #[test]
    fn parses_crlf_line_endings() {
        let mut parser = SseParser::default();
        let chunk = b"event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n";

        let events = parser.push(chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message_stop");
    }

    #[test]
    fn parses_crlf_terminator_split_across_chunks() {
        let mut parser = SseParser::default();
        // The event terminator "\r\n\r\n" is split so the first chunk ends on a
        // lone "\r": the parser must reassemble it across the chunk boundary
        // without emitting or dropping the event.
        let first = parser
            .push(
                b"event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r",
                false,
                MAX_PARTIAL,
            )
            .unwrap();
        assert!(first.is_empty(), "the event is incomplete until its terminator arrives");

        let second = parser.push(b"\n", false, MAX_PARTIAL).unwrap();

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].event_type, "message_stop");
    }

    #[test]
    fn parses_text_split_across_many_chunks() {
        let mut parser = SseParser::default();
        // A single event delivered one byte at a time must reassemble exactly,
        // exercising the incremental append path used for the O(n) buffering.
        let event = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0}\n\n";
        let mut events = Vec::new();
        for (offset, byte) in event.iter().enumerate() {
            let end_of_stream = offset + 1 == event.len();
            events.extend(parser.push(&[*byte], end_of_stream, MAX_PARTIAL).unwrap());
        }

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "content_block_delta");
        assert_eq!(events[0].data["index"], 0);
    }

    #[test]
    fn rejects_oversized_partial_event() {
        let mut parser = SseParser::default();
        let chunk = b"data: {\"type\":\"content_block_delta\"";

        let error = parser.push(chunk, false, 8).unwrap_err();

        assert_eq!(error, StreamError::OversizedPartialEvent);
    }

    #[test]
    fn rejects_oversized_complete_event() {
        let mut parser = SseParser::default();
        // A single *complete* event whose size exceeds the cap must fail closed,
        // not just the retained partial: a huge complete frame would otherwise be
        // materialized in full, bypassing the body-size bound.
        let big = format!("data: {{\"type\":\"x\",\"t\":\"{}\"}}\n\n", "a".repeat(100));

        let error = parser.push(big.as_bytes(), false, 8).unwrap_err();

        assert_eq!(error, StreamError::OversizedPartialEvent);
    }

    #[test]
    fn ignores_blocks_without_json_data() {
        let mut parser = SseParser::default();
        // A comment-only block and the OpenAI-style [DONE] sentinel carry no
        // Anthropic JSON `data`; both are dropped rather than surfaced.
        let events = parser
            .push(b": keep-alive\n\ndata: [DONE]\n\n", false, MAX_PARTIAL)
            .unwrap();

        assert!(events.is_empty());
    }

    #[test]
    fn fails_closed_on_invalid_json_data() {
        let mut parser = SseParser::default();
        // A complete event block whose `data` is present but not valid JSON is
        // malformed framing. Silently dropping it would truncate the
        // client-visible stream, so it must fail closed instead.
        let error = parser
            .push(b"data: {oops not json}\n\n", false, MAX_PARTIAL)
            .unwrap_err();

        assert_eq!(error, StreamError::MalformedEvent);
    }

    #[test]
    fn fails_closed_on_data_without_type() {
        let mut parser = SseParser::default();
        // Valid JSON with no string `type` cannot be reconstructed as a Messages
        // event; dropping it silently would truncate the stream, so fail closed.
        let error = parser.push(b"data: {\"index\":0}\n\n", false, MAX_PARTIAL).unwrap_err();

        assert_eq!(error, StreamError::MalformedEvent);
    }

    #[test]
    fn parses_two_events_in_one_chunk() {
        let mut parser = SseParser::default();
        let chunk = b"data: {\"type\":\"ping\"}\n\ndata: {\"type\":\"message_stop\"}\n\n";

        let events = parser.push(chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "ping");
        assert_eq!(events[1].event_type, "message_stop");
    }

    #[test]
    fn round_zero_message_start_is_forwarded_with_stable_id() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);

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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
    fn finish_round_fails_closed_on_oversized_reconstruction() {
        // A tiny body bound forces the reconstructed message over the limit.
        let mut stream = LogicalStream::new(64, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
        text_output(&mut stream, &message_start("msg_1"), false);
        let long_query = "a".repeat(9 * 1024);
        text_output(&mut stream, &web_search_block(0, "toolu_ws", &long_query), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);

        let error = stream.finish_round().unwrap_err();

        assert_eq!(error, StreamError::QueryTooLong, "an oversized query fails closed");
    }

    #[test]
    fn finish_round_fails_closed_when_suppressed_search_is_not_managed() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
        text_output(&mut stream, &message_start("msg_1"), false);

        let out = text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);

        assert!(
            out.is_empty(),
            "the managed WebSearch tool_use block is fully suppressed"
        );
    }

    #[test]
    fn output_index_is_monotonic_across_rounds() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
        text_output(&mut stream, &message_start("msg_1"), false);

        let out = text_output(&mut stream, &sse("ping", &json!({"type": "ping"})), false);

        assert!(
            out.contains("event: ping"),
            "ping keep-alives are forwarded to the client"
        );
    }

    #[test]
    fn finish_round_emits_terminal_for_done_message() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
        text_output(&mut stream, &message_start("msg_1"), false);
        text_output(&mut stream, &text_block(0, "Hi"), false);
        text_output(&mut stream, &message_delta("end_turn", 5), false);
        text_output(&mut stream, &message_stop(), false);
        // Pre-load the aggregate one below the ceiling so this round's output
        // tokens would overflow a plain add.
        stream.output_tokens_total = u64::MAX - 1;

        let terminal = String::from_utf8(stream.finish_round().unwrap().terminal).unwrap();

        assert!(
            terminal.contains(&format!("\"output_tokens\":{}", u64::MAX)),
            "output-token aggregation saturates instead of overflowing"
        );
    }

    #[test]
    fn on_chunk_is_silent_after_a_failure() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(64, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(64, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
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
    fn terminal_message_delta_matches_schema_and_aggregates_usage() {
        let mut stream = LogicalStream::new(MAX_BODY, MAX_PARTIAL);
        // Round 0: a managed WebSearch call; usage input=10, cache_read=2.
        text_output(&mut stream, &message_start_with_usage("msg_1", 10, 2), false);
        text_output(&mut stream, &web_search_block(0, "toolu_1", "rust proxy"), false);
        text_output(&mut stream, &message_delta("tool_use", 5), false);
        text_output(&mut stream, &message_stop(), false);
        assert!(matches!(stream.finish_round().unwrap().action, RoundAction::Loop));

        // Round 1: the final answer; usage input=8, cache_read=3.
        stream.begin_round();
        text_output(&mut stream, &message_start_with_usage("msg_2", 8, 3), false);
        text_output(&mut stream, &text_block(0, "Answer"), false);
        text_output(&mut stream, &message_delta("end_turn", 9), false);
        text_output(&mut stream, &message_stop(), false);
        let terminal = stream.finish_round().unwrap().terminal;

        let event = parse_event(&terminal, "message_delta");
        assert_eq!(
            event["delta"]["container"],
            Value::Null,
            "schema-complete delta.container"
        );
        assert_eq!(
            event["delta"]["stop_details"],
            Value::Null,
            "schema-complete delta.stop_details"
        );
        assert_eq!(event["delta"]["stop_reason"], "end_turn", "final stop reason carried");
        assert_eq!(event["usage"]["output_tokens"], 14, "output tokens aggregate 5 + 9");
        assert_eq!(event["usage"]["input_tokens"], 18, "input tokens aggregate 10 + 8");
        assert_eq!(
            event["usage"]["cache_read_input_tokens"], 5,
            "cache-read tokens aggregate 2 + 3"
        );
    }
}
