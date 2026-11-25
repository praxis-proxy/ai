// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Incremental Messages framing, round reconstruction and client lifecycle for
//! proxy-executed Anthropic tools. Each adapter decides how to present tool
//! blocks and whether to continue inference after a complete round.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use serde_json::{Value, json};

/// A parsed SSE event: the decoded JSON `data` payload.
#[derive(Debug)]
pub(super) struct SseEvent {
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
    /// A managed `WebSearch` query exceeded the configured size bound, carried
    /// so the client-facing error names the operator's actual limit.
    QueryTooLong {
        /// Configured per-query byte bound the query exceeded.
        max_bytes: usize,
    },
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
    /// Re-entering inference would send a rebuilt request larger than the
    /// configured body-size bound — the appended search result pushed it over the
    /// limit. On a committed streaming response the buffered path's request-phase
    /// 413 cannot replace the open SSE lifecycle, so the loop terminates with one
    /// coherent error event instead of forwarding an oversized round.
    ReentryTooLarge,
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
    /// The preceding CR already ended a line. Consume an optional following LF.
    after_cr: bool,
    /// Whether the first decoded character has been examined for a BOM.
    started: bool,
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
        let text = if !self.started && !text.is_empty() {
            self.started = true;
            text.strip_prefix('\u{feff}').unwrap_or(&text)
        } else {
            &text
        };

        // Append the newly decoded text onto the retained partial in place,
        // normalizing line endings as it goes, instead of rebuilding the whole
        // buffer each chunk (which was quadratic across a long fragmented event).
        let prev_len = self.line_buffer.len();
        self.normalize_into_buffer(text);

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

        // Bound the retained partial as well as complete events.
        if self.line_buffer.len()
            - consumed
            - usize::from(
                self.line_buffer
                    .get(consumed..)
                    .is_some_and(|tail| tail.ends_with('\n')),
            )
            > max_partial_event_bytes
        {
            return Err(StreamError::OversizedPartialEvent);
        }
        self.line_buffer.drain(..consumed);
        Ok(events)
    }

    /// Append `text` to `line_buffer`, normalizing `\r\n` and lone `\r` to `\n`.
    ///
    /// A CR terminates a line immediately, including at the end of a chunk.
    fn normalize_into_buffer(&mut self, text: &str) {
        if !self.after_cr && !text.contains('\r') {
            self.line_buffer.push_str(text);
            return;
        }
        for ch in text.chars() {
            let skip_lf = self.after_cr && ch == '\n';
            self.after_cr = ch == '\r';
            if !skip_lf {
                self.line_buffer.push(if self.after_cr { '\n' } else { ch });
            }
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
/// bare comment or keep-alive). Accepts `data`, `data:value`, and `data: value`
/// field forms.
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
    let data: Value = serde_json::from_str(&data).map_err(|_invalid_json| StreamError::MalformedEvent)?;
    if data.get("type").and_then(Value::as_str).is_none() {
        return Err(StreamError::MalformedEvent);
    }
    Ok(Some(SseEvent { data }))
}

/// Concatenate an SSE block's `data` field lines, or `None` if it carries none.
///
/// Joins multiple `data:` lines with a newline per the SSE specification and
/// accepts the `data`, `data:value`, and `data: value` field forms.
fn collect_data_field(block: &str) -> Option<Cow<'_, str>> {
    let mut data: Option<Cow<'_, str>> = None;
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
                    let existing = existing.to_mut();
                    existing.push('\n');
                    existing.push_str(field);
                },
                None => data = Some(Cow::Borrowed(field)),
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
    /// Accumulated tool `input_json` text per backend block index.
    partial_json: HashMap<usize, String>,
    /// Backend block index -> client output index for forwarded blocks.
    forwarded: HashMap<usize, u64>,
    /// Stop reason captured from this round's `message_delta`.
    stop_reason: Option<String>,
    /// Final provider delta, including beta metadata that the connector does not synthesize.
    terminal_event: Value,
    /// Matched stop sequence captured from this round's `message_delta`.
    stop_sequence: Option<String>,
    /// Container metadata captured from this round's `message_delta`.
    container: Option<Value>,
    /// Stop details captured from this round's `message_delta`.
    stop_details: Option<Value>,
    /// Output tokens captured from this round's `message_delta`.
    round_output_tokens: u64,
    /// Input tokens captured from this round's `message_start` / `message_delta`.
    round_input_tokens: Option<u64>,
    /// Cache-read input tokens captured from this round's usage.
    round_cache_read: Option<u64>,
    /// Cache-creation input tokens captured from this round's usage.
    round_cache_creation: Option<u64>,
    /// Output-token details captured from this round's usage.
    round_output_tokens_details: Option<Value>,
    /// Server-tool usage captured from this round's usage.
    round_server_tool_use: Option<Value>,
    /// Cumulative bytes of streamed text/tool-input fragments this round.
    ///
    /// Bounds attacker-controlled fragment growth before the round message is
    /// reconstructed, independent of the final serialized-size check.
    accumulated_bytes: usize,
    /// A tool block had incomplete JSON. Only a truncated `max_tokens` round
    /// may return such a fragment, and it must never dispatch a tool.
    invalid_input: bool,
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

/// How a managed tool presents a model-facing block to the client.
pub(super) enum BlockPresentation {
    /// Preserve the model's block and stream its deltas.
    Forward,
    /// Retain the block for model replay only.
    Suppress,
    /// Emit a protocol-specific replacement and keep the original for replay.
    #[cfg(feature = "anthropic-mcp-connector")]
    Replace(Value),
}

/// A coherent Anthropic Messages SSE lifecycle assembled across IRR rounds.
pub(super) struct MessagesStream {
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
    /// Aggregated cache-creation input tokens across rounds.
    cache_creation_total: Option<u64>,
    /// Aggregated output-token details across rounds.
    output_tokens_details_total: Option<Value>,
    /// Aggregated server-tool usage across rounds.
    server_tool_use_total: Option<Value>,
    /// Whether the stream has failed closed; once set, no more bytes are forwarded.
    failed: bool,
    /// Validate the message when the backend emits its stop event.
    validate_stop: bool,
    /// Current round reconstruction and forwarding state.
    round: RoundState,
}

impl MessagesStream {
    /// Start one client lifecycle with bounded model rounds and SSE events.
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
            cache_creation_total: None,
            output_tokens_details_total: None,
            server_tool_use_total: None,
            failed: false,
            validate_stop: false,
            round: RoundState::default(),
        }
    }

    /// Bound for model-facing bytes retained in a single inference round.
    #[cfg(feature = "anthropic-mcp-connector")]
    pub(super) fn body_limit(&self) -> usize {
        self.max_body_bytes
    }

    /// Whether the stream has failed closed and forwards nothing further.
    pub(super) fn has_failed(&self) -> bool {
        self.failed
    }

    /// Whether the backend sent its protocol terminal in the current round.
    #[cfg(feature = "anthropic-mcp-connector")]
    pub(super) fn received_stop(&self) -> bool {
        self.round.message_stopped
    }

    /// Whether the client-visible `message_start` has already been forwarded.
    ///
    /// Once true the client is mid-stream on a committed 200 SSE lifecycle, so a
    /// later untransformable round must fail closed rather than pass raw bytes
    /// through and corrupt the open stream.
    pub(super) fn has_started(&self) -> bool {
        self.message_started
    }

    /// Release retained round data and suppress further output.
    pub(super) fn fail(&mut self) {
        self.failed = true;
        self.begin_round();
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
    pub(super) fn write_chunk(
        &mut self,
        chunk: &[u8],
        end_of_stream: bool,
        output: &mut Vec<u8>,
        mut present: impl FnMut(&Value) -> BlockPresentation,
    ) -> Result<(), StreamError> {
        if self.failed {
            return Ok(());
        }
        let events = self
            .parser
            .push(chunk, end_of_stream, self.max_partial_event_bytes)
            .inspect_err(|_| self.fail())?;
        for event in events {
            self.process_event(event, output, &mut present)
                .inspect_err(|_| self.fail())?;
            if self.failed || (self.validate_stop && self.round.message_stopped) {
                break;
            }
        }
        Ok(())
    }

    /// Reconstruct and forward one parsed SSE event.
    fn process_event(
        &mut self,
        event: SseEvent,
        output: &mut Vec<u8>,
        present: &mut impl FnMut(&Value) -> BlockPresentation,
    ) -> Result<(), StreamError> {
        match event_kind(&event.data) {
            EventKind::MessageStart => self.on_message_start(event.data, output),
            EventKind::ContentBlockStart => self.on_block_start(event.data, output, present),
            EventKind::ContentBlockDelta => self.on_block_delta(event.data, output),
            EventKind::ContentBlockStop => self.on_block_stop(event.data, output),
            EventKind::MessageDelta => self.on_message_delta(event.data),
            // Keep-alives are forwarded so the client connection stays live while
            // intermediate rounds run.
            EventKind::Ping => {
                emit_event(output, "ping", &event.data);
                Ok(())
            },
            // An upstream `error` after HTTP 200 is a real backend failure: forward
            // it verbatim and poison the stream so `finish_round` never fabricates a
            // synthetic success terminal in its place.
            EventKind::Error => {
                emit_event(output, "error", &event.data);
                self.fail();
                Ok(())
            },
            // `message_stop` closes the round's Messages lifecycle. It is not
            // forwarded per round (the terminal lifecycle is emitted once by
            // `finish_round`), but it is recorded so a round truncated before it
            // arrives fails closed instead of fabricating a successful terminal.
            EventKind::MessageStop => {
                self.round.message_stopped = true;
                if self.validate_stop {
                    self.require_complete_lifecycle()?;
                }
                Ok(())
            },
            // Every other unmodeled event is deferred without forwarding.
            EventKind::Other => Ok(()),
        }
    }

    /// Capture the base message and forward the client-visible `message_start`.
    fn on_message_start(&mut self, mut data: Value, output: &mut Vec<u8>) -> Result<(), StreamError> {
        if self.round.message.is_some() || !data.get("message").is_some_and(Value::is_object) {
            return Err(StreamError::MalformedEvent);
        }
        if !self.message_started {
            self.message_started = true;
            emit_event(output, "message_start", &data);
        }
        let mut message = data.get_mut("message").ok_or(StreamError::MalformedEvent)?.take();
        if let Some(usage) = message.get_mut("usage") {
            self.capture_round_usage(usage);
        }
        self.round.message = Some(message);
        Ok(())
    }

    /// Capture per-round input and cache-read token counts from a usage object.
    ///
    /// The final `message_delta` may repeat these fields; the last value seen
    /// wins for the round, and rounds are summed in [`Self::reconstruct_message`].
    fn capture_round_usage(&mut self, usage: &mut Value) {
        if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
            self.round.round_input_tokens = Some(input);
        }
        if let Some(cache_read) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
            self.round.round_cache_read = Some(cache_read);
        }
        if let Some(cache_creation) = usage.get("cache_creation_input_tokens").and_then(Value::as_u64) {
            self.round.round_cache_creation = Some(cache_creation);
        }
        if let Some(output_tokens_details) = usage.get_mut("output_tokens_details").filter(|value| !value.is_null()) {
            self.round.round_output_tokens_details = Some(output_tokens_details.take());
        }
        if let Some(server_tool_use) = usage.get_mut("server_tool_use").filter(|value| !value.is_null()) {
            self.round.round_server_tool_use = Some(server_tool_use.take());
        }
    }

    /// Assign a client output index to a content block and forward its start.
    ///
    /// Model replay retains the original block even when the adapter suppresses
    /// it or presents it under a different client-visible type.
    fn on_block_start(
        &mut self,
        mut data: Value,
        output: &mut Vec<u8>,
        present: &mut impl FnMut(&Value) -> BlockPresentation,
    ) -> Result<(), StreamError> {
        let index = block_index(&data)?;
        // Content blocks are dense and sequential in the Anthropic wire format. A
        // backend-controlled index that skips ahead would force an unbounded
        // sparse allocation, so a non-sequential index fails closed instead.
        if index != self.round.content.len() {
            return Err(StreamError::MalformedEvent);
        }
        let block = data.get_mut("content_block").ok_or(StreamError::MalformedEvent)?;
        if !block.is_object() || self.round.message.is_none() {
            return Err(StreamError::MalformedEvent);
        }
        self.round.open_blocks.insert(index);
        let original = match present(block) {
            BlockPresentation::Forward => None,
            BlockPresentation::Suppress => {
                self.round.content.push(block.take());
                return Ok(());
            },
            #[cfg(feature = "anthropic-mcp-connector")]
            BlockPresentation::Replace(client) => Some(std::mem::replace(block, client)),
        };
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        self.round.forwarded.insert(index, output_index);
        remap_index(&mut data, output_index);
        emit_event(output, "content_block_start", &data);
        let original = match original {
            Some(original) => original,
            None => data.get_mut("content_block").ok_or(StreamError::MalformedEvent)?.take(),
        };
        self.round.content.push(original);
        Ok(())
    }

    /// Reconstruct and, for forwarded blocks, forward a content-block delta.
    fn on_block_delta(&mut self, mut data: Value, output: &mut Vec<u8>) -> Result<(), StreamError> {
        let index = block_index(&data)?;
        if !self.round.open_blocks.contains(&index) {
            return Err(StreamError::MalformedEvent);
        }
        let start = output.len();
        if let Some(output_index) = self.round.forwarded.get(&index).copied() {
            remap_index(&mut data, output_index);
            emit_event(output, "content_block_delta", &data);
        }
        // Serialize first so opaque delta fields can move directly into history.
        // An invalid delta must never reach the client.
        self.accumulate_delta(index, &mut data)
            .inspect_err(|_| output.truncate(start))
    }

    /// Finalize text and tool input and, for forwarded blocks, forward a stop.
    fn on_block_stop(&mut self, data: Value, output: &mut Vec<u8>) -> Result<(), StreamError> {
        let index = block_index(&data)?;
        // A stop must close a currently-open block. A stray (never-started) or
        // duplicate stop is rejected so it cannot silently cancel a different
        // genuinely-open block and mask a truncated round from the completeness
        // gate. Reject before any materialize/forward side effects.
        if !self.round.open_blocks.remove(&index) {
            return Err(StreamError::MalformedEvent);
        }
        if let Some(fragment) = self.round.partial_json.remove(&index) {
            match serde_json::from_str::<Value>(&fragment) {
                Ok(input @ Value::Object(_)) => {
                    let block = self.round.content.get_mut(index).ok_or(StreamError::MalformedEvent)?;
                    block["input"] = input;
                },
                _ => self.round.invalid_input = true,
            }
        }
        if let Some(output_index) = self.round.forwarded.get(&index).copied() {
            emit_remapped(output, "content_block_stop", data, output_index);
        }
        Ok(())
    }

    /// Capture the round's stop reason and output-token count; do not forward.
    fn on_message_delta(&mut self, mut data: Value) -> Result<(), StreamError> {
        if ["delta", "usage"]
            .iter()
            .any(|key| data.get(key).is_some_and(|value| !value.is_object()))
        {
            return Err(StreamError::MalformedEvent);
        }
        if let Some(delta) = data.get_mut("delta") {
            if let Some(stop_reason) = delta.get("stop_reason").and_then(Value::as_str) {
                self.round.stop_reason = Some(stop_reason.to_owned());
            }
            if let Some(stop_sequence) = delta.get("stop_sequence") {
                self.round.stop_sequence = stop_sequence.as_str().map(ToOwned::to_owned);
            }
            if let Some(container) = delta.get_mut("container").filter(|value| !value.is_null()) {
                self.round.container = Some(container.take());
            }
            if let Some(stop_details) = delta.get_mut("stop_details").filter(|value| !value.is_null()) {
                self.round.stop_details = Some(stop_details.take());
            }
        }
        if let Some(usage) = data.get_mut("usage") {
            if let Some(tokens) = usage.get("output_tokens").and_then(Value::as_u64) {
                self.round.round_output_tokens = tokens;
            }
            self.capture_round_usage(usage);
        }
        merge_delta_metadata(&mut self.round.terminal_event, data);
        Ok(())
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
    fn accumulate_delta(&mut self, index: usize, data: &mut Value) -> Result<(), StreamError> {
        let Some(delta) = data.get_mut("delta") else {
            return Ok(());
        };
        // Each streamed delta carries its fragment under a type-specific field.
        // Text and tool-input reconstruct the client-visible content; thinking
        // and signature reconstruct the extended-thinking block so the assistant
        // turn replayed on re-entry keeps its verifiable signature.
        let field = match delta.get("type").and_then(Value::as_str) {
            Some("compaction_delta") => return self.apply_compaction(index, delta),
            Some("citations_delta") => return self.accumulate_citation(index, delta),
            Some("text_delta") => "text",
            Some("input_json_delta") => "partial_json",
            Some("thinking_delta") => "thinking",
            Some("signature_delta") => "signature",
            _ => return Ok(()),
        };
        let fragment = delta
            .get(field)
            .and_then(Value::as_str)
            .ok_or(StreamError::MalformedEvent)?;
        self.charge_fragment_bytes(fragment.len())?;
        if field == "partial_json" {
            if !fragment.is_empty() {
                self.round.partial_json.entry(index).or_default().push_str(fragment);
            }
        } else {
            let block = self.round.content.get_mut(index).ok_or(StreamError::MalformedEvent)?;
            if block[field].is_null() {
                block[field] = Value::String(String::new());
            }
            let Value::String(text) = &mut block[field] else {
                return Err(StreamError::MalformedEvent);
            };
            text.push_str(fragment);
        }
        Ok(())
    }

    /// Preserve a streamed citation in the assistant turn replayed to the model.
    fn accumulate_citation(&mut self, index: usize, delta: &mut Value) -> Result<(), StreamError> {
        let citation = delta.get_mut("citation").ok_or(StreamError::MalformedEvent)?;
        self.charge_fragment_bytes(
            crate::json_body::serialized_len(citation).map_err(|_error| StreamError::MalformedEvent)?,
        )?;
        let block = self.round.content.get_mut(index).ok_or(StreamError::MalformedEvent)?;
        if block["citations"].is_null() {
            block["citations"] = Value::Array(Vec::new());
        }
        block["citations"]
            .as_array_mut()
            .ok_or(StreamError::MalformedEvent)?
            .push(citation.take());
        Ok(())
    }

    /// Compaction deltas replace the summary and its opaque replay metadata.
    fn apply_compaction(&mut self, index: usize, delta: &mut Value) -> Result<(), StreamError> {
        for key in ["content", "encrypted_content"] {
            if let Some(value) = delta.get_mut(key) {
                if !value.is_null() {
                    self.charge_fragment_bytes(value.as_str().ok_or(StreamError::MalformedEvent)?.len())?;
                }
                let block = self.round.content.get_mut(index).ok_or(StreamError::MalformedEvent)?;
                block[key] = value.take();
            }
        }
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
    /// object is missing.
    pub(super) fn reconstruct_message(&mut self) -> Result<Value, StreamError> {
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
        self.output_tokens_total = self.output_tokens_total.saturating_add(self.round.round_output_tokens);
        self.aggregate_round_usage();
        Ok(message)
    }

    /// Serialize a reconstructed turn once, stopping at the replay byte limit.
    pub(super) fn serialize_round(&self, message: &Value) -> Result<Vec<u8>, StreamError> {
        praxis_filter::json_size::to_vec_capped(message, self.max_body_bytes).map_err(|error| match error {
            praxis_filter::json_size::JsonSizeError::Serialization(_) => StreamError::MalformedEvent,
            _ => StreamError::Oversized,
        })
    }

    /// Emit the single terminal pair after the final round.
    pub(super) fn terminal(&mut self) -> Result<Vec<u8>, StreamError> {
        self.require_complete_lifecycle()?;
        self.build_terminal()
    }

    /// Apply the receiving step's bound without resetting the logical message.
    #[cfg(feature = "anthropic-mcp-connector")]
    pub(super) fn set_round_limit(&mut self, limit: usize) {
        self.validate_stop = true;
        self.max_body_bytes = limit;
        self.max_partial_event_bytes = limit;
    }

    /// Emit a terminal pair that signals a pending client tool.
    #[cfg(feature = "anthropic-mcp-connector")]
    pub(super) fn client_tool_terminal(&mut self) -> Result<Vec<u8>, StreamError> {
        self.require_complete_lifecycle()?;
        self.round.stop_reason = Some("tool_use".to_owned());
        self.build_terminal()
    }

    /// Append a proxy-produced block to the same index sequence as model blocks.
    #[cfg(feature = "anthropic-mcp-connector")]
    pub(super) fn emit_block(&mut self, block: Value, output: &mut Vec<u8>) {
        let index = self.next_output_index;
        self.next_output_index += 1;
        let event = Value::Object(serde_json::Map::from_iter([
            ("type".to_owned(), Value::from("content_block_start")),
            ("index".to_owned(), Value::from(index)),
            ("content_block".to_owned(), block),
        ]));
        emit_event(output, "content_block_start", &event);
        emit_event(
            output,
            "content_block_stop",
            &json!({"type":"content_block_stop","index":index}),
        );
    }

    /// Fail closed unless this round delivered a complete Messages lifecycle.
    ///
    /// A clean round carries a stop reason (from `message_delta`), closes every
    /// content block with `content_block_stop`, and ends with `message_stop`. A
    /// clean transport EOF can still leave any of these missing; such a round is
    /// truncated and must not be reconstructed into a successful terminal or
    /// re-entered as if complete.
    pub(super) fn require_complete_lifecycle(&self) -> Result<(), StreamError> {
        if self.round.invalid_input && self.round.stop_reason.as_deref() != Some("max_tokens") {
            return Err(StreamError::MalformedEvent);
        }
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
        if let Some(cache_creation) = self.round.round_cache_creation {
            self.cache_creation_total = Some(self.cache_creation_total.unwrap_or(0).saturating_add(cache_creation));
        }
        aggregate_usage_value(
            &mut self.output_tokens_details_total,
            self.round.round_output_tokens_details.take(),
        );
        aggregate_usage_value(&mut self.server_tool_use_total, self.round.round_server_tool_use.take());
    }

    /// Build the terminal `message_delta` and `message_stop` frames.
    ///
    /// Keep the final backend event intact apart from the fields composed across
    /// the logical message. Beta metadata belongs to its provider and passes through.
    fn build_terminal(&mut self) -> Result<Vec<u8>, StreamError> {
        let mut event = self.round.terminal_event.take();
        let fields = event.as_object_mut().ok_or(StreamError::MalformedEvent)?;
        let delta = fields.entry("delta").or_insert(Value::Null);
        delta["container"] = self.round.container.take().unwrap_or(Value::Null);
        delta["stop_details"] = self.round.stop_details.take().unwrap_or(Value::Null);
        delta["stop_reason"] = Value::from(self.round.stop_reason.take());
        delta["stop_sequence"] = Value::from(self.round.stop_sequence.take());
        let usage = fields.entry("usage").or_insert(Value::Null);
        usage["cache_creation_input_tokens"] = Value::from(self.cache_creation_total);
        usage["cache_read_input_tokens"] = Value::from(self.cache_read_total);
        usage["input_tokens"] = Value::from(self.input_tokens_total);
        usage["output_tokens"] = Value::from(self.output_tokens_total);
        usage["output_tokens_details"] = self.output_tokens_details_total.take().unwrap_or(Value::Null);
        usage["server_tool_use"] = self.server_tool_use_total.take().unwrap_or(Value::Null);
        let mut out = Vec::new();
        emit_event(&mut out, "message_delta", &event);
        emit_event(&mut out, "message_stop", &json!({"type": "message_stop"}));
        Ok(out)
    }
}

/// Retain optional terminal fields across cumulative deltas in one round.
fn merge_delta_metadata(previous: &mut Value, incoming: Value) {
    let Value::Object(incoming) = incoming else {
        return;
    };
    let Value::Object(previous) = previous else {
        *previous = Value::Object(incoming);
        return;
    };
    for (key, value) in incoming {
        if matches!(key.as_str(), "delta" | "usage")
            && let Some(Value::Object(fields)) = previous.get_mut(&key)
            && let Value::Object(update) = value
        {
            fields.extend(update.into_iter().filter(|(_, value)| !value.is_null()));
        } else if !value.is_null() {
            previous.insert(key, value);
        }
    }
}

/// Aggregate a usage object whose fields are cumulative counters.
fn aggregate_usage_value(total: &mut Option<Value>, incoming: Option<Value>) {
    let Some(incoming) = incoming else {
        return;
    };
    if incoming.is_null() {
        return;
    }
    let Value::Object(incoming_object) = incoming else {
        *total = Some(incoming);
        return;
    };
    let Some(Value::Object(total_object)) = total.as_mut() else {
        *total = Some(Value::Object(incoming_object));
        return;
    };
    for (key, value) in incoming_object {
        let Some(incoming_number) = value.as_u64() else {
            total_object.insert(key, value);
            continue;
        };
        let current_number = total_object.get(&key).and_then(Value::as_u64).unwrap_or(0);
        total_object.insert(key, Value::from(current_number.saturating_add(incoming_number)));
    }
}

/// Read the backend content-block `index` from an SSE event payload.
fn block_index(data: &Value) -> Result<usize, StreamError> {
    data.get("index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or(StreamError::MalformedEvent)
}

/// Payload discriminator for Anthropic SSE events.
#[derive(Clone, Copy)]
enum EventKind {
    /// `message_start`.
    MessageStart,
    /// `content_block_start`.
    ContentBlockStart,
    /// `content_block_delta`.
    ContentBlockDelta,
    /// `content_block_stop`.
    ContentBlockStop,
    /// `message_delta`.
    MessageDelta,
    /// `ping` keepalive.
    Ping,
    /// Upstream `error`.
    Error,
    /// `message_stop`.
    MessageStop,
    /// Any other event type.
    Other,
}

/// Classify a payload by its JSON `type` field.
fn event_kind(data: &Value) -> EventKind {
    match data.get("type").and_then(Value::as_str) {
        Some("message_start") => EventKind::MessageStart,
        Some("content_block_start") => EventKind::ContentBlockStart,
        Some("content_block_delta") => EventKind::ContentBlockDelta,
        Some("content_block_stop") => EventKind::ContentBlockStop,
        Some("message_delta") => EventKind::MessageDelta,
        Some("ping") => EventKind::Ping,
        Some("error") => EventKind::Error,
        Some("message_stop") => EventKind::MessageStop,
        _ => EventKind::Other,
    }
}

/// Emit an SSE event with its content-block `index` remapped to the client index.
fn emit_remapped(output: &mut Vec<u8>, event_type: &str, mut data: Value, output_index: u64) {
    remap_index(&mut data, output_index);
    emit_event(output, event_type, &data);
}

/// Replace the payload `index` with the client-visible output index.
fn remap_index(data: &mut Value, output_index: u64) {
    let Some(object) = data.as_object_mut() else {
        return;
    };
    match object.get_mut("index") {
        Some(index) => *index = Value::from(output_index),
        None => {
            object.insert("index".to_owned(), Value::from(output_index));
        },
    }
}

/// Encode one canonical single-line Anthropic SSE event.
fn emit_event(output: &mut Vec<u8>, event_type: &str, data: &Value) {
    emit_framed_event(output, event_type, |out| serde_json::to_writer(out, data));
}

/// Write `event:` / `data:` framing, rolling back on payload failure.
fn emit_framed_event<E>(
    output: &mut Vec<u8>,
    event_type: &str,
    write_payload: impl FnOnce(&mut Vec<u8>) -> Result<(), E>,
) {
    let start = output.len();
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    if write_json_or_rollback(output, write_payload).is_err() {
        output.truncate(start);
        return;
    }
    output.extend_from_slice(b"\n\n");
}

/// Write into `output`, restoring the pre-write length on failure.
fn write_json_or_rollback<E>(output: &mut Vec<u8>, write: impl FnOnce(&mut Vec<u8>) -> Result<(), E>) -> Result<(), E> {
    let json_start = output.len();
    match write(output) {
        Ok(()) => Ok(()),
        Err(error) => {
            output.truncate(json_start);
            Err(error)
        },
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "parser test assertions"
)]
mod tests {
    use super::*;

    const MAX_PARTIAL: usize = 64 * 1024;

    #[test]
    fn parses_one_complete_event() {
        let mut parser = SseParser::default();
        let chunk = b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n";

        let events = parser.push(chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 1);
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
        assert_eq!(second[0].data["type"], "content_block_delta");
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
        assert_eq!(events[0].data["type"], "message_stop");
    }

    #[test]
    fn parses_crlf_terminator_split_across_chunks() {
        let mut parser = SseParser::default();
        // The last CR completes the event even if the optional LF is delayed.
        let first = parser
            .push(
                b"event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r",
                false,
                MAX_PARTIAL,
            )
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].data["type"], "message_stop");

        let second = parser.push(b"\n", false, MAX_PARTIAL).unwrap();

        assert!(second.is_empty(), "the optional LF does not emit a second event");
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
        assert_eq!(events[0].data["type"], "content_block_delta");
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
        let events = parser.push(b": keep-alive\n\n", false, MAX_PARTIAL).unwrap();
        assert!(events.is_empty(), "SSE comments carry no event data");
    }

    #[test]
    fn rejects_chat_done_sentinel() {
        let error = SseParser::default()
            .push(b"data: [DONE]\n\n", false, MAX_PARTIAL)
            .unwrap_err();
        assert_eq!(error, StreamError::MalformedEvent, "Messages data must be a JSON event");
    }

    #[test]
    fn accepts_split_bom_and_exact_size_event() {
        let data = br#"data: {"type":"ping"}"#;
        let mut parser = SseParser::default();
        assert!(
            parser.push(&[0xEF], false, data.len()).unwrap().is_empty(),
            "partial BOM emits nothing"
        );
        assert!(
            parser.push(&[0xBB, 0xBF], false, data.len()).unwrap().is_empty(),
            "BOM carries no event data"
        );
        assert!(
            parser.push(data, false, data.len()).unwrap().is_empty(),
            "event needs a delimiter"
        );
        assert!(
            parser.push(b"\n", false, data.len()).unwrap().is_empty(),
            "first delimiter byte fits"
        );
        assert_eq!(
            parser.push(b"\n", false, data.len()).unwrap().len(),
            1,
            "complete event fits the same limit"
        );
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
        assert_eq!(events[0].data["type"], "ping");
        assert_eq!(events[1].data["type"], "message_stop");
    }

    #[test]
    fn collect_data_field_single_line_borrows_input() {
        let block = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0}";
        let collected = collect_data_field(block).expect("data line present");
        assert!(
            matches!(collected, Cow::Borrowed(_)),
            "a single data line must borrow the input block"
        );
        let expected = "{\"type\":\"content_block_delta\",\"index\":0}";
        assert_eq!(&*collected, expected);
        let data_line = block.split_once("data: ").expect("data prefix").1;
        assert!(
            std::ptr::eq(collected.as_ptr(), data_line.as_ptr()),
            "borrowed data must retain the caller's buffer"
        );
    }

    #[test]
    fn collect_data_field_borrows_data_value_without_space() {
        let block = "data:{\"type\":\"ping\"}";
        let collected = collect_data_field(block).expect("data line present");
        assert!(
            matches!(collected, Cow::Borrowed(_)),
            "the data:value form must borrow the input block"
        );
        assert_eq!(&*collected, "{\"type\":\"ping\"}");
        let data_line = block.strip_prefix("data:").expect("data prefix");
        assert!(
            std::ptr::eq(collected.as_ptr(), data_line.as_ptr()),
            "borrowed data:value payload must retain the caller's buffer"
        );
    }

    #[test]
    fn collect_data_field_joins_multiline_data() {
        let block = "data: {\"type\":\"x\",\ndata: \"i\":1}";
        let collected = collect_data_field(block).expect("data lines present");
        assert!(
            matches!(collected, Cow::Owned(_)),
            "multiple data lines must allocate a joined payload"
        );
        assert_eq!(&*collected, "{\"type\":\"x\",\n\"i\":1}");
    }

    #[test]
    fn parses_multiline_data_payload() {
        let mut parser = SseParser::default();
        let chunk = b"data: {\"type\":\"ping\",\ndata: \"k\":1}\n\n";

        let events = parser.push(chunk, false, MAX_PARTIAL).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data["type"], "ping");
        assert_eq!(events[0].data["k"], 1);
    }

    #[test]
    fn fails_closed_on_malformed_multiline_json() {
        let mut parser = SseParser::default();
        let error = parser
            .push(b"data: {\"type\":\"ping\",\ndata: not-json}\n\n", false, MAX_PARTIAL)
            .unwrap_err();
        assert_eq!(
            error,
            StreamError::MalformedEvent,
            "joined multiline data that is not JSON fails closed"
        );
    }

    #[test]
    fn emit_event_matches_legacy_to_string_format_bytes() {
        let data = json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "hello"}
        });
        let mut output = Vec::new();
        emit_event(&mut output, "content_block_delta", &data);
        let data_str = serde_json::to_string(&data).unwrap_or_default();
        let expected = format!("event: content_block_delta\ndata: {data_str}\n\n");
        assert_eq!(
            output,
            expected.as_bytes(),
            "direct Vec writes must match to_string SSE bytes"
        );
    }

    #[test]
    fn write_json_or_rollback_discards_partial_bytes_on_failure() {
        let mut output = b"event: content_block_delta\ndata: ".to_vec();
        let prefix = output.clone();
        let result = write_json_or_rollback(&mut output, |out| {
            out.extend_from_slice(br#"{"type":"content_block_delta""#);
            Err(std::io::Error::other("injected write failure"))
        });
        assert!(
            result.is_err(),
            "injected failure must surface so emit_event can roll back"
        );
        assert_eq!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::Other,
            "rollback must preserve the original write error"
        );
        assert_eq!(
            output, prefix,
            "a failed payload write must not leave partial JSON in the SSE buffer"
        );
    }

    #[test]
    fn emit_event_rolls_back_entire_event_on_serialization_failure() {
        let mut output = b"event: ping\ndata: {\"type\":\"ping\"}\n\n".to_vec();
        let prefix = output.clone();
        emit_framed_event(&mut output, "content_block_delta", |out| {
            out.extend_from_slice(br#"{"type":"content_block_delta""#);
            Err(std::io::Error::other("injected write failure"))
        });
        assert_eq!(
            output, prefix,
            "serialization failure must roll back the event prefix and skip the delimiter"
        );
    }

    fn emit_event_legacy(output: &mut Vec<u8>, event_type: &str, data: &Value) {
        output.extend_from_slice(b"event: ");
        output.extend_from_slice(event_type.as_bytes());
        output.extend_from_slice(b"\ndata: ");
        output.extend_from_slice(data.to_string().as_bytes());
        output.extend_from_slice(b"\n\n");
    }

    fn collect_data_field_legacy(block: &str) -> Option<String> {
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

    fn emit_remapped_legacy(output: &mut Vec<u8>, event_type: &str, data: &Value, output_index: u64) {
        let mut data = data.clone();
        if let Some(object) = data.as_object_mut() {
            object.insert("index".to_owned(), Value::from(output_index));
        }
        emit_event_legacy(output, event_type, &data);
    }

    fn representative_delta(text: &str, index: u64) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text}
        })
    }

    #[test]
    fn emit_event_allocates_less_than_legacy_to_string() {
        let payload = representative_delta(&"x".repeat(4096), 0);
        let capacity = serde_json::to_vec(&payload).unwrap().len() + 64;
        let mut via_writer = Vec::with_capacity(capacity);
        let mut via_string = Vec::with_capacity(capacity);
        let writer = allocation_counter::measure(|| {
            emit_event(&mut via_writer, "content_block_delta", &payload);
        });
        let string = allocation_counter::measure(|| {
            emit_event_legacy(&mut via_string, "content_block_delta", &payload);
        });
        assert_eq!(
            via_writer, via_string,
            "content_block_delta SSE bytes must stay equivalent while measuring allocations"
        );
        assert!(
            writer.bytes_total < string.bytes_total,
            "to_writer must allocate fewer bytes than to_string: writer={} string={}",
            writer.bytes_total,
            string.bytes_total
        );
    }

    #[test]
    fn collect_data_field_single_line_allocates_less_than_legacy_to_owned() {
        let block = concat!(
            "data: {\"type\":\"content_block_delta\",\"index\":0,",
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}",
        );
        let optimized = allocation_counter::measure(|| {
            std::hint::black_box(collect_data_field(block));
        });
        let legacy = allocation_counter::measure(|| {
            std::hint::black_box(collect_data_field_legacy(block));
        });
        assert_eq!(
            collect_data_field(block).as_deref(),
            collect_data_field_legacy(block).as_deref(),
            "borrowed and owned collection must yield the same payload"
        );
        assert!(
            optimized.bytes_total < legacy.bytes_total,
            "single-line collect must allocate fewer bytes than always-to_owned: optimized={} legacy={}",
            optimized.bytes_total,
            legacy.bytes_total
        );
    }

    #[test]
    fn remap_index_avoids_deep_clone_of_delta_payload() {
        let payload = representative_delta(&"x".repeat(8192), 0);

        let clone_source = payload.clone();
        let cloned = allocation_counter::measure(|| {
            let mut data = clone_source.clone();
            if let Some(object) = data.as_object_mut() {
                object.insert("index".to_owned(), Value::from(1_u64));
            }
            std::hint::black_box(data);
        });

        let mut owned = payload.clone();
        let mutated = allocation_counter::measure(|| {
            remap_index(&mut owned, 1);
            std::hint::black_box(&owned);
        });

        assert_eq!(owned["index"], 1, "in-place remap must rewrite the client index");
        assert_eq!(owned["delta"]["text"], payload["delta"]["text"]);
        assert!(
            cloned.bytes_total >= 8192,
            "deep-cloning a delta payload must copy the text, allocated {}",
            cloned.bytes_total
        );
        assert!(
            mutated.bytes_total < cloned.bytes_total,
            "in-place remap must allocate fewer bytes than cloning: mutate={} clone={}",
            mutated.bytes_total,
            cloned.bytes_total
        );
    }

    #[test]
    fn remapped_delta_emit_allocates_less_than_legacy_clone() {
        let payload = representative_delta(&"x".repeat(4096), 0);
        let capacity = serde_json::to_vec(&payload).unwrap().len() + 64;
        let mut optimized_out = Vec::with_capacity(capacity);
        let mut legacy_out = Vec::with_capacity(capacity);

        let owned = payload.clone();
        let optimized = allocation_counter::measure(|| {
            emit_remapped(&mut optimized_out, "content_block_delta", owned, 1);
        });
        let legacy = allocation_counter::measure(|| {
            emit_remapped_legacy(&mut legacy_out, "content_block_delta", &payload, 1);
        });
        assert_eq!(
            optimized_out, legacy_out,
            "in-place remap must emit the same SSE bytes as the clone baseline"
        );
        assert!(
            optimized.bytes_total < legacy.bytes_total,
            "owned remap+to_writer must allocate fewer bytes than clone+to_string: optimized={} legacy={}",
            optimized.bytes_total,
            legacy.bytes_total
        );
    }
}
