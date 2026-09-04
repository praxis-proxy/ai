// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Incremental Chat Completions SSE to Responses SSE conversion.
//!
//! [`StreamConverter`] is a pure, request-scoped state machine. It consumes
//! Chat Completions streaming chunks, emits the corresponding Responses
//! streaming events with a monotonic `sequence_number`, and builds the terminal
//! resource by reusing the finite translation builders so the streamed terminal
//! matches the non-streaming translation exactly.
//!
//! The converter never buffers a full response: each [`push`](StreamConverter::push)
//! returns only the bytes for events completed by that chunk. Framing is
//! isolated in the `framing` module so a future shared SSE codec can replace
//! it without touching this state machine.

mod chat;
mod events;
mod framing;

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
mod tests;

use praxis_filter::FilterError;
use serde_json::{Map, Value, json};
use tracing::warn;

use self::{
    chat::{ChatChoice, ChatChunk, ChatToolCallFragment},
    events::StreamEvent,
    framing::{Framing, FramingError},
};
use crate::openai::translation::chat_completions::{
    ResponseContext, chat_response_to_response_resource, function_call_output_item_from_parts,
    in_progress_response_resource, message_output_item, output_text_item, refusal_item,
};

/// Resource limits governing one streaming translation.
#[derive(Debug, Clone, Copy)]
pub(super) struct StreamLimits {
    /// Maximum bytes buffered by the SSE frame parser.
    pub(super) max_sse_buffer_bytes: usize,
    /// Maximum total number of Responses SSE events emitted, including the
    /// terminal event (see [`emit_event`]).
    pub(super) max_stream_events: usize,
    /// Maximum accumulated argument bytes for a single tool call.
    pub(super) max_tool_call_argument_bytes: usize,
    /// Maximum number of distinct tool calls in one response.
    pub(super) max_tool_calls: usize,
    /// Wall-clock streaming timeout in seconds; `0` disables the guard.
    pub(super) stream_timeout_secs: u64,
    /// Ceiling for total accumulated semantic bytes (text, refusal, arguments).
    pub(super) max_body_bytes: usize,
    /// Maximum number of decoded SSE frames processed in one response.
    pub(super) max_stream_frames: usize,
    /// Maximum complete encoded size, in bytes, of a single emitted Responses
    /// SSE frame (see [`emit_event`]). Kept strictly below the downstream
    /// accumulator's SSE buffer so every emitted frame fits the buffer it is
    /// reassembled in; an oversized frame fails the stream closed instead of
    /// being emitted and later dropped by the accumulator.
    pub(super) max_emitted_sse_frame_bytes: usize,
}

/// Per-callback inputs needed to build resource snapshots.
pub(super) struct SnapshotInputs<'a> {
    /// Original canonical Responses request body.
    pub(super) request_body: &'a Value,
    /// Current wall-clock time in seconds.
    pub(super) now: u64,
}

/// Lifecycle phase of the converter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Streaming; no terminal finish reason observed yet.
    Open,
    /// A finish reason was observed and open items were closed.
    ProviderDone,
    /// A terminal `response.completed`/`response.incomplete` was emitted.
    EmittedTerminal,
    /// Translation failed; a `response.failed` was emitted and no further
    /// provider input is processed.
    Failed,
}

/// Monotonic event sequencing state.
///
/// `Copy` so a terminal closeout can snapshot it before emitting several frames
/// and restore it verbatim if any frame fails, rolling the sequence counter back
/// in lockstep with the output buffer (see [`StreamConverter::emit_terminal`]).
#[derive(Debug, Clone, Copy)]
struct EmitState {
    /// Next `sequence_number` to assign.
    sequence_number: u64,
    /// Count of events emitted so far.
    events_emitted: usize,
}

/// Errors produced while translating a stream.
#[derive(Debug)]
enum ConvertError {
    /// Serializing a Responses event failed; surfaces as a [`FilterError`].
    Serialize(serde_json::Error),
    /// A Chat chunk was not valid JSON.
    MalformedJson,
    /// The upstream response contained more than one choice.
    MultipleChoices,
    /// The single choice used an index other than zero.
    InvalidChoiceIndex,
    /// Chunk `id` or `model` changed mid-stream.
    InconsistentMetadata,
    /// The first Chat chunk was missing a required identity field (`id`,
    /// `model`, or `object`).
    MissingChunkMetadata,
    /// The chunk `object` was not `chat.completion.chunk`.
    UnexpectedObject,
    /// Choice data arrived after a terminal finish reason.
    DataAfterFinish,
    /// A provider frame arrived after a successful terminal streaming event
    /// (`response.completed`/`response.incomplete`) was emitted.
    DataAfterTerminal,
    /// A tool-call name fragment arrived after arguments began.
    NameAfterArguments,
    /// A tool-call id fragment arrived after the item was announced.
    IdAfterArguments,
    /// Tool-call arguments arrived before its id and name were known.
    ToolCallMissingIdentity,
    /// The SSE buffer limit was exceeded.
    BufferOverflow,
    /// The emitted-event limit was exceeded.
    EventLimit,
    /// A single encoded Responses SSE frame exceeded the emitted-frame ceiling.
    FrameSizeLimit,
    /// The decoded-frame limit was exceeded.
    FrameLimit,
    /// The accumulated semantic byte ceiling was exceeded.
    ByteLimit,
    /// A single tool call exceeded the argument byte limit.
    ToolArgumentLimit,
    /// The tool-call count limit was exceeded.
    ToolCountLimit,
    /// A tool-call fragment arrived without an `index` to correlate it.
    ToolCallMissingIndex,
    /// A delta carried the deprecated singular `function_call` field, which the
    /// converter does not translate.
    LegacyFunctionCall,
    /// The stream exceeded its wall-clock timeout.
    Timeout,
    /// The stream ended without a supported finish reason.
    MissingFinishReason,
    /// A choice carried a finish reason the converter does not recognize.
    UnknownFinishReason,
    /// The stream ended with an unterminated SSE frame buffered.
    IncompleteFrame,
}

impl From<FramingError> for ConvertError {
    fn from(error: FramingError) -> Self {
        match error {
            FramingError::BufferOverflow { buffered_bytes, limit } => {
                warn!(
                    buffered_bytes,
                    limit, "SSE buffer limit exceeded during streaming translation"
                );
                Self::BufferOverflow
            },
            FramingError::FrameLimit { count, limit } => {
                warn!(
                    count,
                    limit, "decoded-frame limit exceeded during streaming translation"
                );
                Self::FrameLimit
            },
        }
    }
}

/// Per-message streaming state for the assistant output item.
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag tracks a distinct, independent lifecycle stage of the assistant message item (item added, closed, text/refusal part opened)"
)]
struct MessageState {
    /// Responses output index for this message item.
    output_index: usize,
    /// Stable message item id (`msg_{response_id}`).
    item_id: String,
    /// Whether `response.output_item.added` was emitted.
    item_added: bool,
    /// Whether the item's `done` events were emitted.
    closed: bool,
    /// Next content-part index to allocate.
    next_content_index: usize,
    /// Whether the text content part was opened.
    text_part_open: bool,
    /// Content index assigned to the text part.
    text_content_index: Option<usize>,
    /// Accumulated assistant text.
    text: String,
    /// Accumulated text token logprobs.
    logprobs: Vec<Value>,
    /// Whether the refusal content part was opened.
    refusal_part_open: bool,
    /// Content index assigned to the refusal part.
    refusal_content_index: Option<usize>,
    /// Accumulated refusal text.
    refusal: String,
}

impl MessageState {
    /// Create empty message state at the given output index.
    fn new(output_index: usize, item_id: String) -> Self {
        Self {
            output_index,
            item_id,
            item_added: false,
            closed: false,
            next_content_index: 0,
            text_part_open: false,
            text_content_index: None,
            text: String::new(),
            logprobs: Vec::new(),
            refusal_part_open: false,
            refusal_content_index: None,
            refusal: String::new(),
        }
    }

    /// Allocate the next content-part index.
    fn alloc_content_index(&mut self) -> usize {
        let index = self.next_content_index;
        self.next_content_index += 1;
        index
    }
}

/// Per-tool-call streaming state keyed by the Chat tool-call index.
#[derive(Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag tracks a distinct, independent lifecycle stage of a tool-call item (arguments started, item added, closed)"
)]
struct ToolCallState {
    /// Chat Completions tool-call index (stable across fragments).
    chat_index: u64,
    /// Responses output index, assigned when the item's `output_item.added`
    /// event is emitted (at the first argument fragment, or the close late-add),
    /// so it stays dense with no holes.
    output_index: Option<usize>,
    /// Stable function-call item id (`fc_{call_id}`).
    item_id: Option<String>,
    /// Accumulated call id.
    call_id: String,
    /// Accumulated function name.
    name: String,
    /// Accumulated function arguments.
    arguments: String,
    /// Whether argument fragments have begun.
    args_started: bool,
    /// Whether `response.output_item.added` was emitted.
    item_added: bool,
    /// Whether the item's `done` events were emitted.
    closed: bool,
}

impl ToolCallState {
    /// Create empty state for a Chat tool-call index.
    fn new(chat_index: u64) -> Self {
        Self {
            chat_index,
            output_index: None,
            item_id: None,
            call_id: String::new(),
            name: String::new(),
            arguments: String::new(),
            args_started: false,
            item_added: false,
            closed: false,
        }
    }

    /// Whether the call has both an id and a name and can form an item.
    fn has_identity(&self) -> bool {
        !self.call_id.is_empty() && !self.name.is_empty()
    }
}

/// Incremental Chat Completions to Responses SSE converter.
pub(super) struct StreamConverter {
    /// Stable Responses resource id.
    response_id: String,
    /// Responses resource creation timestamp.
    created_at: u64,
    /// Configured resource limits.
    limits: StreamLimits,
    /// SSE frame reassembly.
    framing: Framing,
    /// Current lifecycle phase.
    phase: Phase,
    /// Event sequencing state.
    emit: EmitState,
    /// Whether the opening lifecycle events were emitted.
    lifecycle_started: bool,
    /// Whether the first data chunk's required identity fields were validated.
    initial_metadata_checked: bool,
    /// Wall-clock start time in seconds, set on first push.
    start_time: Option<u64>,
    /// Stable Chat completion id, when provided.
    chat_id: Option<String>,
    /// Stable Chat model, when provided.
    model: Option<String>,
    /// Provider service tier, when provided.
    service_tier: Option<Value>,
    /// Observed finish reason.
    finish_reason: Option<String>,
    /// Accumulated token usage.
    usage: Option<Value>,
    /// Next Responses output index to allocate.
    next_output_index: usize,
    /// Assistant message state, if any content appeared.
    message: Option<MessageState>,
    /// Tool-call states in first-appearance order.
    tool_calls: Vec<ToolCallState>,
    /// Total accumulated semantic bytes.
    accumulated_bytes: usize,
    /// Count of decoded SSE frames processed.
    frames_processed: usize,
}

impl StreamConverter {
    /// Create a converter for a streaming response.
    pub(super) fn new(response_id: String, created_at: u64, limits: StreamLimits) -> Self {
        Self {
            framing: Framing::new(limits.max_sse_buffer_bytes),
            response_id,
            created_at,
            limits,
            phase: Phase::Open,
            emit: EmitState {
                sequence_number: 0,
                events_emitted: 0,
            },
            lifecycle_started: false,
            initial_metadata_checked: false,
            start_time: None,
            chat_id: None,
            model: None,
            service_tier: None,
            finish_reason: None,
            usage: None,
            next_output_index: 0,
            message: None,
            tool_calls: Vec::new(),
            accumulated_bytes: 0,
            frames_processed: 0,
        }
    }

    /// Feed one response body chunk, returning any completed Responses SSE bytes.
    ///
    /// Returns `None` when the chunk produced no complete event. Recoverable
    /// translation failures emit a `response.failed` terminal instead of
    /// propagating; only internal serialization failures return [`FilterError`].
    pub(super) fn push(&mut self, chunk: &[u8], inputs: &SnapshotInputs<'_>) -> Result<Option<Vec<u8>>, FilterError> {
        let mut out = Vec::new();
        if let Err(error) = self.try_push(chunk, inputs, &mut out) {
            self.handle_failure(&error, inputs, &mut out)?;
        }
        Ok((!out.is_empty()).then_some(out))
    }

    /// Finalize the stream at end of upstream body.
    pub(super) fn finish(&mut self, inputs: &SnapshotInputs<'_>) -> Result<Option<Vec<u8>>, FilterError> {
        let mut out = Vec::new();
        if let Err(error) = self.try_finish(inputs, &mut out) {
            self.handle_failure(&error, inputs, &mut out)?;
        }
        Ok((!out.is_empty()).then_some(out))
    }

    /// Translate one chunk, appending events to `out`.
    fn try_push(&mut self, chunk: &[u8], inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        match self.phase {
            Phase::Failed => {
                // We already emitted `response.failed` by our own fail-closed
                // decision. The upstream is unaware and keeps sending the frames it
                // always intended to; that is expected continuation, not a protocol
                // violation, so drop it rather than fail the transport.
                if !chunk.is_empty() {
                    warn!("dropping provider bytes received after a failed streaming terminal");
                }
                return Ok(());
            },
            Phase::EmittedTerminal => return self.push_after_terminal(chunk),
            Phase::Open | Phase::ProviderDone => {},
        }
        self.check_timeout(inputs.now)?;
        // Emit the opening lifecycle events on the first non-empty callback, even
        // when it carries only part of the first SSE frame; otherwise a
        // fragmented first frame would delay `response.created` until the frame
        // completes.
        if !chunk.is_empty() {
            self.ensure_lifecycle(inputs, out)?;
        }
        // The frame cap is enforced during decoding (see `Framing::push`), so an
        // oversized callback stops at the ceiling instead of materializing every
        // frame first. Count the frames this callback actually decoded so the cap
        // spans the whole response.
        let frames = self
            .framing
            .push(chunk, self.frames_processed, self.limits.max_stream_frames)?;
        self.frames_processed = self.frames_processed.saturating_add(frames.len());
        for frame in &frames {
            if matches!(self.phase, Phase::EmittedTerminal) {
                // A successful terminal was emitted by an earlier frame in this same
                // callback; any further decoded frame is post-terminal provider data
                // and must fail closed, not be silently dropped. `Phase::Failed` is
                // unreachable here — fail-closed errors propagate out via `?` and are
                // only turned into `Phase::Failed` by `handle_failure` after the loop.
                return Err(ConvertError::DataAfterTerminal);
            }
            self.process_frame(frame, inputs, out)?;
        }
        Ok(())
    }

    /// Handle a callback that arrives after a successful terminal was emitted.
    ///
    /// Benign trailing callbacks (empty drains, comments, keepalives, partial
    /// frames) decode to no frame and are tolerated, but a fully decoded provider
    /// frame is post-terminal data — response splitting or a misbehaving upstream
    /// — so fail closed rather than silently drop it. A post-terminal flood that
    /// trips the frame cap mid-decode is equally post-terminal data, so map it to
    /// the same failure.
    fn push_after_terminal(&mut self, chunk: &[u8]) -> Result<(), ConvertError> {
        match self
            .framing
            .push(chunk, self.frames_processed, self.limits.max_stream_frames)
        {
            Ok(frames) if frames.is_empty() => Ok(()),
            Ok(_) | Err(FramingError::FrameLimit { .. }) => Err(ConvertError::DataAfterTerminal),
            Err(other) => Err(other.into()),
        }
    }

    /// Emit the terminal event at clean end of stream.
    fn try_finish(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        match self.phase {
            // Our own fail-closed terminal was already emitted; the upstream is
            // unaware and its trailing bytes are expected continuation, so any
            // buffered partial frame at EOF is not a protocol violation.
            Phase::Failed => return Ok(()),
            // A successful terminal was emitted. A partial frame still buffered at
            // EOF is post-terminal provider data — response splitting or a
            // truncated continuation — so fail closed rather than accept it.
            Phase::EmittedTerminal => {
                return if self.framing.has_incomplete_frame() {
                    Err(ConvertError::DataAfterTerminal)
                } else {
                    Ok(())
                };
            },
            Phase::Open | Phase::ProviderDone => {},
        }
        // The EOF callback must honor the wall-clock limit too; otherwise a
        // provider-done stream whose terminal is deferred to `finish` could
        // emit a successful terminal past the timeout window.
        self.check_timeout(inputs.now)?;
        if self.framing.has_incomplete_frame() {
            return Err(ConvertError::IncompleteFrame);
        }
        match self.phase {
            Phase::ProviderDone => self.emit_terminal(inputs, out),
            _ => Err(ConvertError::MissingFinishReason),
        }
    }

    /// Enforce the wall-clock streaming timeout.
    ///
    /// This is checked on each body callback, so it bounds the elapsed time
    /// across active chunks but cannot fire while the upstream is idle and no
    /// callback arrives. Idle/stalled connections are the proxy's concern via
    /// its upstream read timeout (see the Pingora boundary); this guard bounds a
    /// slow-but-active stream.
    fn check_timeout(&mut self, now: u64) -> Result<(), ConvertError> {
        let start = *self.start_time.get_or_insert(now);
        if self.limits.stream_timeout_secs > 0 && now.saturating_sub(start) > self.limits.stream_timeout_secs {
            return Err(ConvertError::Timeout);
        }
        Ok(())
    }

    /// Process one reassembled SSE frame.
    fn process_frame(
        &mut self,
        frame: &framing::Frame,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        if is_done_sentinel(frame) {
            return self.handle_done(inputs, out);
        }
        self.ensure_lifecycle(inputs, out)?;
        let chunk: ChatChunk<'_> = serde_json::from_slice(&frame.data).map_err(|error| {
            tracing::trace!(%error, "malformed chat completion chunk");
            ConvertError::MalformedJson
        })?;
        Self::validate_chunk(&chunk)?;
        self.require_initial_metadata(&chunk)?;
        self.capture_metadata(&chunk)?;
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
        match chunk.choices.as_slice() {
            [] => Ok(()),
            [choice] => {
                if matches!(self.phase, Phase::ProviderDone) {
                    return Err(ConvertError::DataAfterFinish);
                }
                if choice.index != 0 {
                    return Err(ConvertError::InvalidChoiceIndex);
                }
                self.process_choice(choice, inputs, out)
            },
            _ => Err(ConvertError::MultipleChoices),
        }
    }

    /// Consume a `[DONE]` sentinel, emitting the terminal event when ready.
    fn handle_done(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        match self.phase {
            Phase::ProviderDone => self.emit_terminal(inputs, out),
            Phase::Open => Err(ConvertError::MissingFinishReason),
            Phase::EmittedTerminal | Phase::Failed => Ok(()),
        }
    }

    /// Require the first data chunk to carry the Chat schema's mandatory
    /// identity fields (`id`, `model`, `object`).
    ///
    /// Providers send these on every `chat.completion.chunk`; a first chunk
    /// missing any of them signals a malformed or non-Chat stream, so fail closed
    /// rather than translate it into a successful Responses stream with a null
    /// model. Only the first parsed chunk is checked — comment/keepalive frames
    /// never dispatch (they carry no `data:`), and later chunks may legitimately
    /// omit repeated identity fields, whose consistency `capture_metadata` still
    /// enforces.
    fn require_initial_metadata(&mut self, chunk: &ChatChunk<'_>) -> Result<(), ConvertError> {
        if self.initial_metadata_checked {
            return Ok(());
        }
        self.initial_metadata_checked = true;
        let has_id = chunk.id.as_deref().is_some_and(|id| !id.is_empty());
        let has_model = chunk.model.as_deref().is_some_and(|model| !model.is_empty());
        let has_object = chunk.object.as_deref() == Some("chat.completion.chunk");
        if has_id && has_model && has_object {
            Ok(())
        } else {
            Err(ConvertError::MissingChunkMetadata)
        }
    }

    /// Validate stable per-chunk invariants.
    fn validate_chunk(chunk: &ChatChunk<'_>) -> Result<(), ConvertError> {
        if let Some(object) = chunk.object.as_deref()
            && object != "chat.completion.chunk"
        {
            return Err(ConvertError::UnexpectedObject);
        }
        Ok(())
    }

    /// Capture and enforce consistency of chunk id, model, and service tier.
    fn capture_metadata(&mut self, chunk: &ChatChunk<'_>) -> Result<(), ConvertError> {
        if let Some(id) = chunk.id.as_deref() {
            match &self.chat_id {
                Some(existing) if existing != id => return Err(ConvertError::InconsistentMetadata),
                Some(_) => {},
                None => self.chat_id = Some(id.to_owned()),
            }
        }
        if let Some(model) = chunk.model.as_deref() {
            match &self.model {
                Some(existing) if existing != model => return Err(ConvertError::InconsistentMetadata),
                Some(_) => {},
                None => self.model = Some(model.to_owned()),
            }
        }
        if let Some(service_tier) = chunk.service_tier.as_deref() {
            match &self.service_tier {
                Some(Value::String(existing)) if existing != service_tier => {
                    return Err(ConvertError::InconsistentMetadata);
                },
                Some(_) => {},
                None => self.service_tier = Some(Value::String(service_tier.to_owned())),
            }
        }
        Ok(())
    }

    /// Translate one choice's delta and finish reason.
    fn process_choice(
        &mut self,
        choice: &ChatChoice<'_>,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        if let Some(delta) = &choice.delta {
            if delta.function_call.is_some() {
                // Legacy singular function calling is not translated; failing
                // closed avoids silently completing the stream with empty output.
                return Err(ConvertError::LegacyFunctionCall);
            }
            if let Some(content) = delta.content.as_deref()
                && !content.is_empty()
            {
                self.process_text_delta(content, choice.logprobs.as_ref(), inputs, out)?;
            }
            if let Some(refusal) = delta.refusal.as_deref()
                && !refusal.is_empty()
            {
                self.process_refusal_delta(refusal, inputs, out)?;
            }
            for fragment in &delta.tool_calls {
                self.process_tool_call_fragment(fragment, inputs, out)?;
            }
        }
        if let Some(finish) = choice.finish_reason.as_deref() {
            self.process_finish_reason(finish)?;
        }
        Ok(())
    }

    /// Record a terminal finish reason and enter the provider-done phase.
    ///
    /// Open output items are *not* closed here. Their `response.output_item.done`
    /// events are deferred to [`emit_terminal`](Self::emit_terminal) so an item is
    /// committed to the client only once the stream is known to close cleanly. A
    /// recoverable failure in the `ProviderDone` window (a post-finish timeout,
    /// malformed or inconsistent trailing chunk, data after finish, or an
    /// incomplete frame at EOF) therefore fails closed with a coherent empty
    /// `response.failed` — never a `response.failed` that follows an already-sent
    /// `response.output_item.done`.
    fn process_finish_reason(&mut self, finish: &str) -> Result<(), ConvertError> {
        if finish == "function_call" {
            // A `function_call` finish signals legacy function calling even when
            // no `delta.function_call` payload arrived; the converter does not
            // translate it, so fail closed rather than complete with empty output.
            return Err(ConvertError::LegacyFunctionCall);
        }
        if !is_recognized_finish_reason(finish) {
            return Err(ConvertError::UnknownFinishReason);
        }
        if self.finish_reason.is_none() {
            self.finish_reason = Some(finish.to_owned());
        }
        self.phase = Phase::ProviderDone;
        Ok(())
    }

    /// Emit `response.created` and `response.in_progress` once.
    fn ensure_lifecycle(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        if self.lifecycle_started {
            return Ok(());
        }
        self.lifecycle_started = true;
        let context = self.response_context(inputs, None);
        let resource = in_progress_response_resource(&context);
        if serialized_json_len(&resource)? > self.limits.max_body_bytes {
            return Err(ConvertError::ByteLimit);
        }
        self.commit_lifecycle(&resource, out)
    }

    /// Emit the lifecycle pair atomically against the per-frame ceiling.
    fn commit_lifecycle(&mut self, resource: &Value, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        // Commit the lifecycle pair atomically. The in-progress event's longer
        // framing can exceed a tight per-frame ceiling after the created event
        // fits; roll both back so failure handling never emits `response.failed`
        // after a partial lifecycle.
        let out_checkpoint = out.len();
        let emit_checkpoint = self.emit;
        let result = emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::response_created(resource),
            out,
        )
        .and_then(|()| {
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::response_in_progress(resource),
                out,
            )
        });
        if let Err(error) = result {
            out.truncate(out_checkpoint);
            self.emit = emit_checkpoint;
            return Err(error);
        }
        Ok(())
    }

    /// Translate one assistant text delta.
    #[expect(
        clippy::expect_used,
        reason = "message item and text content part were just ensured above; their absence is an unreachable state-machine invariant"
    )]
    fn process_text_delta(
        &mut self,
        delta: &str,
        logprobs: Option<&Value>,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        self.charge_bytes(delta.len())?;
        self.ensure_message_item(inputs, out)?;
        self.ensure_text_part(out)?;
        let delta_logprobs = chat::logprobs_content(logprobs);
        // Retained logprobs are cloned into message state and re-serialized in
        // the terminal snapshot, so their bytes must count against the aggregate
        // ceiling; otherwise a stream of tiny text deltas carrying large
        // logprobs could accumulate unbounded retained data past max_body_bytes.
        let logprob_cost = logprobs_byte_cost(&delta_logprobs);
        if logprob_cost > 0 {
            self.charge_bytes(logprob_cost)?;
        }
        if let (Some(message), Value::Array(items)) = (self.message.as_mut(), &delta_logprobs) {
            message.text.push_str(delta);
            message.logprobs.extend(items.iter().cloned());
        }
        let message = self.message.as_ref().expect("message ensured");
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::output_text_delta(
                &message.item_id,
                message.output_index,
                message.text_content_index.expect("text part ensured"),
                delta,
                &delta_logprobs,
            ),
            out,
        )
    }

    /// Translate one assistant refusal delta.
    #[expect(
        clippy::expect_used,
        reason = "message item and refusal content part were just ensured above; their absence is an unreachable state-machine invariant"
    )]
    fn process_refusal_delta(
        &mut self,
        delta: &str,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        self.charge_bytes(delta.len())?;
        self.ensure_message_item(inputs, out)?;
        self.ensure_refusal_part(out)?;
        if let Some(message) = self.message.as_mut() {
            message.refusal.push_str(delta);
        }
        let message = self.message.as_ref().expect("message ensured");
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::refusal_delta(
                &message.item_id,
                message.output_index,
                message.refusal_content_index.expect("refusal part ensured"),
                delta,
            ),
            out,
        )
    }

    /// Ensure the assistant message item has been announced.
    fn ensure_message_item(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        if self.message.is_some() {
            return Ok(());
        }
        let output_index = self.alloc_output_index();
        // Matches the finite translation `msg_{response_id}` item-id convention.
        let item_id = format!("msg_{}", self.response_id);
        self.message = Some(MessageState::new(output_index, item_id));
        let context = self.response_context(inputs, None);
        let item = message_output_item(&context, "in_progress", &[]);
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::output_item_added(output_index, &item),
            out,
        )?;
        if let Some(message) = self.message.as_mut() {
            message.item_added = true;
        }
        Ok(())
    }

    /// Ensure the text content part has been announced.
    #[expect(
        clippy::expect_used,
        reason = "callers only reach this after ensure_message_item, so the message item is present; its absence is an unreachable state-machine invariant"
    )]
    fn ensure_text_part(&mut self, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        if self.message.as_ref().is_some_and(|message| message.text_part_open) {
            return Ok(());
        }
        let message = self.message.as_mut().expect("message ensured");
        let content_index = message.alloc_content_index();
        message.text_content_index = Some(content_index);
        message.text_part_open = true;
        let message = self.message.as_ref().expect("message ensured");
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::content_part_added(
                &message.item_id,
                message.output_index,
                content_index,
                &output_text_item("", &[]),
            ),
            out,
        )
    }

    /// Ensure the refusal content part has been announced.
    #[expect(
        clippy::expect_used,
        reason = "callers only reach this after ensure_message_item, so the message item is present; its absence is an unreachable state-machine invariant"
    )]
    fn ensure_refusal_part(&mut self, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        if self.message.as_ref().is_some_and(|message| message.refusal_part_open) {
            return Ok(());
        }
        let message = self.message.as_mut().expect("message ensured");
        let content_index = message.alloc_content_index();
        message.refusal_content_index = Some(content_index);
        message.refusal_part_open = true;
        let message = self.message.as_ref().expect("message ensured");
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::content_part_added(
                &message.item_id,
                message.output_index,
                content_index,
                &json!({"type": "refusal", "refusal": ""}),
            ),
            out,
        )
    }

    /// Translate one tool-call fragment.
    #[expect(
        clippy::too_many_lines,
        reason = "single cohesive tool-call fragment state transition (id, name, arguments) that reads clearer as one function"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "position comes from tool_call_position(), which returns an existing index or pushes and returns len()-1, so it is always in bounds"
    )]
    #[expect(
        clippy::expect_used,
        reason = "begin_tool_arguments set item_id and output_index for this position before the delta is emitted; their absence is an unreachable invariant"
    )]
    fn process_tool_call_fragment(
        &mut self,
        fragment: &ChatToolCallFragment<'_>,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        let chat_index = fragment.index.ok_or(ConvertError::ToolCallMissingIndex)?;
        let position = self.tool_call_position(chat_index)?;
        if let Some(id) = fragment.id.as_deref()
            && !id.is_empty()
        {
            // The item id is frozen as `fc_{call_id}` when arguments begin and the
            // item is announced. A later id fragment would change the terminal id
            // after streaming events already announced another, so reject it.
            if self.tool_calls[position].args_started {
                return Err(ConvertError::IdAfterArguments);
            }
            self.charge_bytes(id.len())?;
            self.tool_calls[position].call_id.push_str(id);
        }
        let Some(function) = &fragment.function else {
            return Ok(());
        };
        if let Some(name) = function.name.as_deref()
            && !name.is_empty()
        {
            if self.tool_calls[position].args_started {
                return Err(ConvertError::NameAfterArguments);
            }
            self.charge_bytes(name.len())?;
            self.tool_calls[position].name.push_str(name);
        }
        if let Some(arguments) = function.arguments.as_deref()
            && !arguments.is_empty()
        {
            self.begin_tool_arguments(position, inputs, out)?;
            self.charge_tool_arguments(position, arguments.len())?;
            self.tool_calls[position].arguments.push_str(arguments);
            let call = &self.tool_calls[position];
            let item_id = call.item_id.clone().expect("item id set");
            let output_index = call.output_index.expect("output index set");
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::function_call_arguments_delta(&item_id, output_index, arguments),
                out,
            )?;
        }
        Ok(())
    }

    /// Find or create tool-call state for a Chat tool-call index.
    ///
    /// Creating state does **not** claim a terminal output index. The index is
    /// allocated later, at the moment the call's `response.output_item.added`
    /// event is emitted (see [`Self::begin_tool_arguments`] and the late-add path
    /// in [`Self::close_tool_calls`]). Canonical `OpenResponses` defines
    /// `output_index` as the index of the item actually added, so allocating at
    /// emit keeps the streamed indices dense: a call the provider starts but never
    /// fully identifies (id-only or name-only) is never announced, so it never
    /// consumes an index and can never leave a hole. The trade-off is that an
    /// id-only fragment that is only named later is ordered by when its item is
    /// emitted, not by when it first appeared; for realistic providers, which send
    /// id and name together in the first fragment, emit order equals appearance
    /// order.
    fn tool_call_position(&mut self, chat_index: u64) -> Result<usize, ConvertError> {
        if let Some(position) = self.tool_calls.iter().position(|call| call.chat_index == chat_index) {
            return Ok(position);
        }
        if self.tool_calls.len() >= self.limits.max_tool_calls {
            return Err(ConvertError::ToolCountLimit);
        }
        self.tool_calls.push(ToolCallState::new(chat_index));
        Ok(self.tool_calls.len() - 1)
    }

    /// Announce a tool-call item at the first argument fragment, allocating its
    /// terminal output index at the moment `response.output_item.added` is
    /// emitted.
    #[expect(
        clippy::indexing_slicing,
        reason = "position comes from tool_call_position(), which returns an existing index or pushes and returns len()-1, so it is always in bounds"
    )]
    fn begin_tool_arguments(
        &mut self,
        position: usize,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        if self.tool_calls[position].args_started {
            return Ok(());
        }
        if !self.tool_calls[position].has_identity() {
            return Err(ConvertError::ToolCallMissingIdentity);
        }
        // Text and refusal are ordered before tool calls; opening a message
        // that later received tool calls keeps output indexes stable.
        let _ = inputs;
        self.tool_calls[position].args_started = true;
        // Claim the dense output index here, at the emit point, not when the call
        // first appeared: an id-only call that is never named never reaches this
        // point and so never consumes an index.
        let output_index = self.alloc_output_index();
        self.tool_calls[position].output_index = Some(output_index);
        let call_id = self.tool_calls[position].call_id.clone();
        let item_id = format!("fc_{call_id}");
        self.tool_calls[position].item_id = Some(item_id);
        let name = self.tool_calls[position].name.clone();
        let item = function_call_output_item_from_parts(&call_id, &name, "", "in_progress");
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::output_item_added(output_index, &item),
            out,
        )?;
        self.tool_calls[position].item_added = true;
        Ok(())
    }

    /// Close all open output items as the terminal is emitted.
    ///
    /// Called from [`emit_terminal`](Self::emit_terminal) — not when the finish
    /// reason arrives — so an item's `response.output_item.done` is committed to
    /// the client atomically with the terminal event.
    fn close_open_items(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        // Validate every started tool call's identity before emitting any done
        // events. `close_message` below streams the message's
        // `response.output_item.done`; if a later tool call is missing its id or
        // name, `close_tool_calls` would only discover that afterwards, leaving a
        // completed message item that the failed terminal then omits from its
        // (empty) output. Failing up front keeps the stream coherent: nothing is
        // completed once the response is destined to fail.
        self.validate_tool_call_identities()?;
        // Preflight the whole closeout against the event budget before emitting
        // any of it. `close_message` and `close_tool_calls` emit several capped
        // events; if the budget were exhausted midway, an already-completed item
        // (for example the message's `output_item.done`) would reach the client
        // while the failed terminal reports empty output. Failing up front keeps
        // closeout atomic: either every closeout event *and* the terminal fit, or
        // none is emitted. The trailing `+ 1` reserves the terminal slot, matching
        // `emit_event`'s reservation.
        let needed = self.closeout_event_budget();
        if self.emit.events_emitted.saturating_add(needed).saturating_add(1) > self.limits.max_stream_events {
            return Err(ConvertError::EventLimit);
        }
        self.close_message(inputs, out)?;
        self.close_tool_calls(out)
    }

    /// Count the capped events the closeout will emit.
    ///
    /// Mirrors exactly what [`close_message`](Self::close_message) and
    /// [`close_tool_calls`](Self::close_tool_calls) emit so the preflight in
    /// [`close_open_items`](Self::close_open_items) neither over- nor
    /// under-reserves the event budget. Keep the arithmetic here in lockstep with
    /// those two functions.
    fn closeout_event_budget(&self) -> usize {
        let mut needed = 0;
        // Message close: `output_text`/`refusal` done + their `content_part.done`,
        // then one `output_item.done`. Skipped when the message was never added or
        // is already closed, matching `close_message`'s early return.
        if let Some(message) = self.message.as_ref()
            && message.item_added
            && !message.closed
        {
            if message.text_part_open {
                needed += 2;
            }
            if message.refusal_part_open {
                needed += 2;
            }
            needed += 1;
        }
        // Tool close: an optional late-add `output_item.added` for a call whose
        // arguments never began, then `function_call_arguments.done` +
        // `output_item.done`. Already-closed calls are skipped, matching
        // `close_tool_calls`.
        for call in &self.tool_calls {
            if call.closed {
                continue;
            }
            if !call.item_added {
                needed += 1;
            }
            needed += 2;
        }
        needed
    }

    /// Ensure every started tool call carries both an id and a name.
    ///
    /// A call the provider began but never fully identified is incomplete;
    /// dropping it would silently turn an intended call into an empty success, so
    /// fail closed before any output item is completed.
    fn validate_tool_call_identities(&self) -> Result<(), ConvertError> {
        if self.tool_calls.iter().any(|call| !call.has_identity()) {
            return Err(ConvertError::ToolCallMissingIdentity);
        }
        Ok(())
    }

    /// Close the assistant message item.
    #[expect(
        clippy::too_many_lines,
        reason = "single cohesive message-close sequence (text done, refusal done, item done) that reads clearer as one function"
    )]
    #[expect(
        clippy::expect_used,
        reason = "the guard above returns early unless the message item was added, so message is present and its open parts have their content indices set; their absence is an unreachable invariant"
    )]
    fn close_message(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        if self
            .message
            .as_ref()
            .is_none_or(|message| !message.item_added || message.closed)
        {
            return Ok(());
        }
        let status = self.terminal_item_status();
        if self.message.as_ref().expect("present").text_part_open {
            let message = self.message.as_ref().expect("present");
            let logprobs = Value::Array(message.logprobs.clone());
            let content_index = message.text_content_index.expect("text index");
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::output_text_done(
                    &message.item_id,
                    message.output_index,
                    content_index,
                    &message.text,
                    &logprobs,
                ),
                out,
            )?;
            let message = self.message.as_ref().expect("present");
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::content_part_done(
                    &message.item_id,
                    message.output_index,
                    content_index,
                    &output_text_item(&message.text, &message.logprobs),
                ),
                out,
            )?;
        }
        if self.message.as_ref().expect("present").refusal_part_open {
            let message = self.message.as_ref().expect("present");
            let content_index = message.refusal_content_index.expect("refusal index");
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::refusal_done(&message.item_id, message.output_index, content_index, &message.refusal),
                out,
            )?;
            let message = self.message.as_ref().expect("present");
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::content_part_done(
                    &message.item_id,
                    message.output_index,
                    content_index,
                    &refusal_item(&message.refusal),
                ),
                out,
            )?;
        }
        let content_items = self.build_message_content_items();
        let context = self.response_context(inputs, None);
        let output_index = self.message.as_ref().expect("present").output_index;
        emit_event(
            &mut self.emit,
            &self.limits,
            true,
            events::output_item_done(output_index, &message_output_item(&context, status, &content_items)),
            out,
        )?;
        self.message.as_mut().expect("present").closed = true;
        Ok(())
    }

    /// Close all tool-call items, adding name-only calls that never saw arguments.
    #[expect(
        clippy::too_many_lines,
        reason = "single cohesive loop that closes every tool-call item (late add, arguments done, item done) and reads clearer as one function"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "position iterates 0..self.tool_calls.len(), so every index is in bounds"
    )]
    #[expect(
        clippy::expect_used,
        reason = "each closed call had its item_id and output_index set when its item was added — either during streaming or by the late-add branch below; their absence is an unreachable invariant"
    )]
    fn close_tool_calls(&mut self, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        let status = self.terminal_item_status();
        for position in 0..self.tool_calls.len() {
            if !self.tool_calls[position].has_identity() {
                // Defense in depth: `close_open_items` already validated every
                // identity before any item was completed, so this branch is
                // unreachable in practice. Retained so a future direct caller
                // still fails closed rather than emitting an empty-identity item.
                return Err(ConvertError::ToolCallMissingIdentity);
            }
            if self.tool_calls[position].closed {
                continue;
            }
            if !self.tool_calls[position].item_added {
                // An identified call whose arguments never began was never
                // announced during streaming; claim its dense output index here,
                // at this emit point, so it lands after every item already added.
                let output_index = self.alloc_output_index();
                self.tool_calls[position].output_index = Some(output_index);
                let call_id = self.tool_calls[position].call_id.clone();
                self.tool_calls[position].item_id = Some(format!("fc_{call_id}"));
                self.tool_calls[position].item_added = true;
                let name = self.tool_calls[position].name.clone();
                emit_event(
                    &mut self.emit,
                    &self.limits,
                    true,
                    events::output_item_added(
                        output_index,
                        &function_call_output_item_from_parts(&call_id, &name, "", "in_progress"),
                    ),
                    out,
                )?;
            }
            let call = &self.tool_calls[position];
            let item_id = call.item_id.clone().expect("item id set");
            let output_index = call.output_index.expect("output index set");
            let call_id = call.call_id.clone();
            let name = call.name.clone();
            let arguments = call.arguments.clone();
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::function_call_arguments_done(&item_id, output_index, &name, &arguments),
                out,
            )?;
            emit_event(
                &mut self.emit,
                &self.limits,
                true,
                events::output_item_done(
                    output_index,
                    &function_call_output_item_from_parts(&call_id, &name, &arguments, status),
                ),
                out,
            )?;
            self.tool_calls[position].closed = true;
        }
        Ok(())
    }

    /// Emit the terminal `response.completed` or `response.incomplete` event.
    fn emit_terminal(&mut self, inputs: &SnapshotInputs<'_>, out: &mut Vec<u8>) -> Result<(), ConvertError> {
        let finish = self.finish_reason.clone().unwrap_or_default();
        let context = self.response_context(inputs, Some(inputs.now));
        let synthetic = self.synthetic_completion(&finish);
        let mut resource = chat_response_to_response_resource(&synthetic, &context)
            .unwrap_or_else(|_| in_progress_response_resource(&context));
        self.order_output_by_stream_index(&mut resource);
        self.order_message_content_by_stream_index(&mut resource);
        // The finite path rejects a serialized response exceeding `max_body_bytes`
        // before sending it; bound the streamed terminal resource the same way so
        // both paths enforce an identical response-size ceiling. This runs *before*
        // `close_open_items` below so a terminal that trips the ceiling fails
        // closed with no `response.output_item.done` already on the wire; the
        // resource is built from accumulated state, not the closeout events, so it
        // is identical whether checked before or after the close.
        if serialized_json_len(&resource)? > self.limits.max_body_bytes {
            return Err(ConvertError::ByteLimit);
        }
        // Commit the open output items now, atomically with the terminal: their
        // `response.output_item.done` events are emitted only once the stream is
        // known to close cleanly (finish reason recorded, byte ceiling honored, and
        // the closeout event budget preflighted). A failure before this point stays
        // in `ProviderDone` with no committed items, so `emit_failed` produces a
        // coherent empty `response.failed` rather than one contradicting a
        // completed item.
        //
        // The closeout emits several frames (each item's `output_item.done`) before
        // the terminal, and any of them — or the terminal itself — can still trip
        // the per-frame size ceiling. Snapshot both the output buffer and the
        // sequence counters and restore them on failure so a mid-closeout
        // `FrameSizeLimit` rolls the whole closeout back: no `output_item.done`
        // reaches the client ahead of the `response.failed` that `handle_failure`
        // then emits. The `close_open_items` side effects on message/tool-call
        // state are not rolled back, but the converter transitions to `Failed`
        // immediately afterward and never emits from that state again.
        let out_checkpoint = out.len();
        let emit_checkpoint = self.emit;
        if let Err(error) = self.commit_terminal(&finish, &resource, inputs, out) {
            out.truncate(out_checkpoint);
            self.emit = emit_checkpoint;
            return Err(error);
        }
        self.phase = Phase::EmittedTerminal;
        Ok(())
    }

    /// Close the open output items and emit the terminal event.
    ///
    /// Split from [`emit_terminal`](Self::emit_terminal) so the caller can snapshot
    /// and roll back the output buffer and sequence counters around the whole
    /// closeout: this either emits every `output_item.done` *and* the terminal, or
    /// (on any error, including a per-frame size trip) leaves nothing on the wire.
    fn commit_terminal(
        &mut self,
        finish: &str,
        resource: &Value,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        self.close_open_items(inputs, out)?;
        let event = match finish {
            "length" | "content_filter" => events::response_incomplete(resource),
            _ => events::response_completed(resource),
        };
        emit_event(&mut self.emit, &self.limits, false, event, out)
    }

    /// Emit a `response.failed` terminal with a partial snapshot.
    #[expect(
        clippy::too_many_lines,
        reason = "one cohesive failure path: tolerate an oversized lifecycle, build the partial snapshot, and emit the terminal with byte-ceiling and per-frame-ceiling fallbacks to the minimal resource"
    )]
    fn emit_failed(
        &mut self,
        inputs: &SnapshotInputs<'_>,
        message: &str,
        out: &mut Vec<u8>,
    ) -> Result<(), ConvertError> {
        // Emit the in-progress lifecycle first, but tolerate its own frame-size
        // trip: the `response.created`/`response.in_progress` snapshot echoes
        // request-controlled fields (instructions, input, ...) and can itself
        // exceed the per-frame ceiling. `ensure_lifecycle` then emits nothing and
        // marks the lifecycle started, so a coherent (minimal) `response.failed`
        // can still terminate the stream below.
        match self.ensure_lifecycle(inputs, out) {
            Ok(()) | Err(ConvertError::FrameSizeLimit) => {},
            Err(other) => return Err(other),
        }
        // A failed response has no completion moment, so no completed_at timestamp.
        let context = self.response_context(inputs, None);
        // The synthetic completion only supplies the request-echoed envelope
        // (id, model, usage); mark_failed drops its output because the failed
        // terminal never streamed completion events for the accumulated items.
        let synthetic = self.synthetic_completion("stop");
        let mut resource = chat_response_to_response_resource(&synthetic, &context)
            .unwrap_or_else(|_| in_progress_response_resource(&context));
        mark_failed(&mut resource, message);
        // The partial snapshot can be arbitrarily large — this failure may itself
        // be the byte-limit trip, and the snapshot echoes request-controlled
        // fields (input, instructions, metadata, tools, ...) that are themselves
        // unbounded. If it would exceed the response-size ceiling, fall back to a
        // constant-bounded failed resource that echoes no request data, so the
        // terminal never re-serializes the oversized content the ceiling bounds.
        if serialized_json_len(&resource)? > self.limits.max_body_bytes {
            resource = minimal_failed_resource(&context, message);
        }
        // Emit the failed terminal. Even after the byte-ceiling fallback the frame
        // can exceed the (tighter) per-frame ceiling — the response-size ceiling
        // may be configured above it. `emit_event` rolls the oversized frame back
        // off `out`; retry once with the constant-bounded minimal resource, which
        // is guaranteed to fit, so the stream always terminates with a
        // schema-complete `response.failed`.
        match emit_event(
            &mut self.emit,
            &self.limits,
            false,
            events::response_failed(&resource),
            out,
        ) {
            Ok(()) => {},
            Err(ConvertError::FrameSizeLimit) => {
                let minimal = minimal_failed_resource(&context, message);
                emit_event(
                    &mut self.emit,
                    &self.limits,
                    false,
                    events::response_failed(&minimal),
                    out,
                )?;
            },
            Err(other) => return Err(other),
        }
        self.phase = Phase::Failed;
        Ok(())
    }

    /// Handle a translation failure by emitting `response.failed` when possible.
    fn handle_failure(
        &mut self,
        error: &ConvertError,
        inputs: &SnapshotInputs<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), FilterError> {
        if let ConvertError::Serialize(serialize_error) = error {
            return Err(serialize_filter_error(serialize_error));
        }
        if let ConvertError::DataAfterTerminal = error {
            // The terminal event is already on the wire, so a second terminal
            // cannot be emitted; continuing would forward split or injected
            // upstream data. Fail the transport so the proxy tears the stream down.
            warn!("provider sent data after a terminal streaming event; failing the stream");
            return Err("responses_to_chat_completions: upstream sent data after a terminal streaming event".into());
        }
        if matches!(self.phase, Phase::EmittedTerminal | Phase::Failed) {
            warn!(error = ?error, "streaming translation error after a terminal event; dropping");
            return Ok(());
        }
        match self.emit_failed(inputs, failure_message(error), out) {
            Ok(()) => Ok(()),
            Err(ConvertError::Serialize(serialize_error)) => Err(serialize_filter_error(&serialize_error)),
            Err(other) => {
                warn!(error = ?other, "failed to emit response.failed terminal event");
                Ok(())
            },
        }
    }

    /// Build a synthetic finite Chat completion from accumulated stream state.
    #[expect(
        clippy::too_many_lines,
        reason = "assembles one synthetic chat.completion object (message, choice, completion envelope) that reads clearer built inline"
    )]
    fn synthetic_completion(&self, finish_reason: &str) -> Value {
        let mut message = Map::new();
        message.insert("role".to_owned(), Value::String("assistant".to_owned()));
        let text = self.message.as_ref().map_or("", |state| state.text.as_str());
        message.insert(
            "content".to_owned(),
            if text.is_empty() {
                Value::Null
            } else {
                Value::String(text.to_owned())
            },
        );
        if let Some(state) = &self.message
            && !state.refusal.is_empty()
        {
            message.insert("refusal".to_owned(), Value::String(state.refusal.clone()));
        }
        let tool_calls = self.synthetic_tool_calls();
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        }

        let mut choice = Map::new();
        choice.insert("index".to_owned(), json!(0));
        choice.insert("message".to_owned(), Value::Object(message));
        choice.insert("finish_reason".to_owned(), Value::String(finish_reason.to_owned()));
        if let Some(state) = &self.message
            && !state.logprobs.is_empty()
        {
            choice.insert("logprobs".to_owned(), json!({"content": state.logprobs.clone()}));
        }

        let mut completion = Map::new();
        if let Some(id) = &self.chat_id {
            completion.insert("id".to_owned(), Value::String(id.clone()));
        }
        completion.insert("object".to_owned(), Value::String("chat.completion".to_owned()));
        if let Some(model) = &self.model {
            completion.insert("model".to_owned(), Value::String(model.clone()));
        }
        if let Some(service_tier) = &self.service_tier {
            completion.insert("service_tier".to_owned(), service_tier.clone());
        }
        completion.insert("choices".to_owned(), json!([Value::Object(choice)]));
        if let Some(usage) = &self.usage {
            completion.insert("usage".to_owned(), usage.clone());
        }
        Value::Object(completion)
    }

    /// Build synthetic tool-call objects ordered by streamed output index.
    fn synthetic_tool_calls(&self) -> Vec<Value> {
        let mut calls: Vec<&ToolCallState> = self.tool_calls.iter().filter(|call| call.has_identity()).collect();
        calls.sort_by_key(|call| call.output_index.unwrap_or(usize::MAX));
        calls
            .into_iter()
            .map(|call| {
                json!({
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })
            })
            .collect()
    }

    /// Reorder the terminal `output` array so each item sits at the position its
    /// streamed `output_index` announced. The finite builder always emits the
    /// assistant message before tool calls, but a tool call that streamed before
    /// any text claimed a lower output index; honoring the emitted order keeps
    /// incremental events and the terminal snapshot consistent.
    fn order_output_by_stream_index(&self, resource: &mut Value) {
        let Some(output) = resource.get_mut("output").and_then(Value::as_array_mut) else {
            return;
        };
        output.sort_by_key(|item| self.streamed_output_index(item));
    }

    /// Reorder the assistant message item's content parts to match the order in
    /// which they streamed. The finite builder always lays out text before
    /// refusal, but each content part was announced at the `content_index` of its
    /// arrival, so a refusal that streamed before text claimed the lower index.
    /// Honoring that keeps the terminal snapshot consistent with the streamed
    /// `content_part` events.
    fn order_message_content_by_stream_index(&self, resource: &mut Value) {
        let Some(message) = self.message.as_ref() else {
            return;
        };
        let Some(item) = resource
            .get_mut("output")
            .and_then(Value::as_array_mut)
            .and_then(|output| {
                output
                    .iter_mut()
                    .find(|item| item.get("id").and_then(Value::as_str) == Some(message.item_id.as_str()))
            })
        else {
            return;
        };
        let Some(content) = item.get_mut("content").and_then(Value::as_array_mut) else {
            return;
        };
        content.sort_by_key(|part| match part.get("type").and_then(Value::as_str) {
            Some("refusal") => message.refusal_content_index.unwrap_or(usize::MAX),
            Some("output_text") => message.text_content_index.unwrap_or(usize::MAX),
            _ => usize::MAX,
        });
    }

    /// Look up the streamed output index for a terminal output item by its id.
    /// Items without a recorded stream index sort last, preserving builder order.
    fn streamed_output_index(&self, item: &Value) -> usize {
        let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
        if let Some(message) = self.message.as_ref()
            && message.item_id == id
        {
            return message.output_index;
        }
        self.tool_calls
            .iter()
            .find(|call| call.item_id.as_deref() == Some(id))
            .and_then(|call| call.output_index)
            .unwrap_or(usize::MAX)
    }

    /// Build the message content items in content-index order.
    fn build_message_content_items(&self) -> Vec<Value> {
        let Some(message) = self.message.as_ref() else {
            return Vec::new();
        };
        let mut parts: Vec<(usize, Value)> = Vec::new();
        if let Some(index) = message.text_content_index {
            parts.push((index, output_text_item(&message.text, &message.logprobs)));
        }
        if let Some(index) = message.refusal_content_index {
            parts.push((index, refusal_item(&message.refusal)));
        }
        parts.sort_by_key(|(index, _)| *index);
        parts.into_iter().map(|(_, part)| part).collect()
    }

    /// Build a response context borrowing the request body.
    fn response_context<'a>(&self, inputs: &SnapshotInputs<'a>, completed_at: Option<u64>) -> ResponseContext<'a> {
        let context =
            ResponseContext::from_responses_request(inputs.request_body, self.response_id.clone(), self.created_at);
        match completed_at {
            Some(timestamp) => context.with_completed_at(timestamp),
            None => context,
        }
    }

    /// Map the observed finish reason to a terminal item status.
    fn terminal_item_status(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("length" | "content_filter") => "incomplete",
            _ => "completed",
        }
    }

    /// Allocate the next Responses output index.
    fn alloc_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    /// Charge accumulated semantic bytes against the body ceiling.
    fn charge_bytes(&mut self, bytes: usize) -> Result<(), ConvertError> {
        self.accumulated_bytes = self.accumulated_bytes.saturating_add(bytes);
        if self.accumulated_bytes > self.limits.max_body_bytes {
            return Err(ConvertError::ByteLimit);
        }
        Ok(())
    }

    /// Charge tool-call argument bytes against per-call and global limits.
    #[expect(
        clippy::indexing_slicing,
        reason = "position comes from tool_call_position(), which returns an existing index or pushes and returns len()-1, so it is always in bounds"
    )]
    fn charge_tool_arguments(&mut self, position: usize, bytes: usize) -> Result<(), ConvertError> {
        let projected = self.tool_calls[position].arguments.len().saturating_add(bytes);
        if projected > self.limits.max_tool_call_argument_bytes {
            return Err(ConvertError::ToolArgumentLimit);
        }
        self.charge_bytes(bytes)
    }
}

/// Encode one event, injecting the sequence number and enforcing the event and
/// per-frame size caps.
///
/// `limits.max_stream_events` bounds the *total* number of Responses SSE events
/// emitted for one response, including its single terminal event. The terminal is
/// emitted with `capped == false` so a stream at the ceiling still closes
/// cleanly; to keep the running total within the cap, capped (non-terminal)
/// events must therefore leave one slot free for that terminal. Downstream
/// accumulators (`openai_stream_events`) count every emitted event identically,
/// so this bound keeps a client-visible stream within their budget and never
/// silently skips persistence.
///
/// `limits.max_emitted_sse_frame_bytes` bounds the *complete* encoded frame. The
/// frame is encoded into `out` first, then measured; if it exceeds the ceiling it
/// is truncated back off `out` and [`ConvertError::FrameSizeLimit`] is returned,
/// so nothing partial reaches the client and the sequence counters are left
/// untouched. Measuring the fully encoded frame (rather than approximating from
/// the payload) makes the check exact and keeps every emitted frame within the
/// downstream accumulator's reassembly buffer. A serialization failure is rolled
/// back the same way before surfacing as [`ConvertError::Serialize`].
fn emit_event(
    emit: &mut EmitState,
    limits: &StreamLimits,
    capped: bool,
    event: StreamEvent,
    out: &mut Vec<u8>,
) -> Result<(), ConvertError> {
    if capped && emit.events_emitted.saturating_add(1) >= limits.max_stream_events {
        return Err(ConvertError::EventLimit);
    }
    let start = out.len();
    if let Err(error) = events::encode(event, emit.sequence_number, out) {
        // Drop any partially written frame so a failed encode leaves `out`
        // byte-for-byte as it was before this call.
        out.truncate(start);
        return Err(ConvertError::Serialize(error));
    }
    if out.len().saturating_sub(start) > limits.max_emitted_sse_frame_bytes {
        // The complete encoded frame exceeds the emitted-frame ceiling. Roll it
        // off `out` so no partial frame reaches the client, and fail closed
        // without consuming a sequence number.
        out.truncate(start);
        return Err(ConvertError::FrameSizeLimit);
    }
    emit.sequence_number += 1;
    emit.events_emitted += 1;
    Ok(())
}

/// Whether a frame is the Chat Completions `[DONE]` sentinel.
fn is_done_sentinel(frame: &framing::Frame) -> bool {
    frame.data.as_slice().trim_ascii() == b"[DONE]"
}

/// Serialized byte cost of retained logprobs, used for aggregate accounting.
///
/// Returns the JSON-serialized length of the logprobs `content` array so its
/// retained clone counts against the body ceiling; empty or non-array logprobs
/// cost nothing.
fn logprobs_byte_cost(logprobs: &Value) -> usize {
    match logprobs {
        Value::Array(items) if !items.is_empty() => serde_json::to_vec(logprobs).map_or(0, |bytes| bytes.len()),
        _ => 0,
    }
}

/// Build a filter error for an internal serialization failure.
fn serialize_filter_error(error: &serde_json::Error) -> FilterError {
    format!("responses_to_chat_completions: {error}").into()
}

/// Build a constant-bounded `response.failed` resource.
///
/// Used only when the full failure snapshot would exceed `max_body_bytes`. It is
/// assembled directly from constants and nulls — never from the request body — so
/// every schema-required `Response` field is present while its serialized size
/// cannot grow with the client's request. Only the proxy-generated, bounded
/// `id`/`created_at` and the client-safe constant failure `message` vary; all
/// request-controlled fields (`model`, `previous_response_id`, `instructions`,
/// `input`, `metadata`, `tools`, `text`, `tool_choice`, `prompt_cache_key`,
/// `safety_identifier`, ...) are neutralized to fixed constants regardless of
/// what the request carried.
#[expect(
    clippy::too_many_lines,
    reason = "one constant Response literal; every schema field is spelled out so the bounded fallback stays a complete, valid resource"
)]
fn minimal_failed_resource(context: &ResponseContext<'_>, message: &str) -> Value {
    json!({
        "id": context.response_id,
        "object": "response",
        "created_at": context.created_at,
        "completed_at": Value::Null,
        "status": "failed",
        "error": {"code": "server_error", "message": message},
        "incomplete_details": Value::Null,
        "instructions": Value::Null,
        "max_output_tokens": Value::Null,
        "max_tool_calls": Value::Null,
        "model": "",
        "input": Value::Null,
        "output": Value::Array(Vec::new()),
        "parallel_tool_calls": true,
        "previous_response_id": Value::Null,
        "reasoning": Value::Null,
        "store": false,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": Value::Array(Vec::new()),
        "top_p": 1.0,
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": Value::Null,
        "metadata": json!({}),
        "background": false,
        "service_tier": "default",
        "prompt_cache_key": Value::Null,
        "safety_identifier": Value::Null,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    })
}

/// Stamp a Responses resource with the `failed` status, error object, a cleared
/// `incomplete_details`, a null `completed_at`, and empty `output`.
///
/// A failed terminal aborts mid-stream: the accumulated message and tool-call
/// items never received their `output_item.done` completion events, so reporting
/// them in the snapshot — as completed or otherwise — would claim output the
/// stream never finished. The error object carries the failure instead. Clearing
/// `completed_at` matches the schema, where only a completed response has one.
fn mark_failed(resource: &mut Value, message: &str) {
    if let Some(object) = resource.as_object_mut() {
        object.insert("status".to_owned(), Value::String("failed".to_owned()));
        object.insert("error".to_owned(), json!({"code": "server_error", "message": message}));
        object.insert("incomplete_details".to_owned(), Value::Null);
        object.insert("completed_at".to_owned(), Value::Null);
        object.insert("output".to_owned(), Value::Array(Vec::new()));
    }
}

/// Measure the serialized JSON byte length of `value` without allocating the
/// full serialized buffer.
///
/// The terminal resource is measured for parity with the finite response-size
/// ceiling; counting into a sink avoids buffering a second full copy alongside
/// the one `emit_event` serializes.
fn serialized_json_len(value: &Value) -> Result<usize, ConvertError> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(buf.len());
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value).map_err(ConvertError::Serialize)?;
    Ok(counter.0)
}

/// Return whether `reason` is a finish reason the converter can translate.
///
/// The finite translation only maps the documented Chat Completions finish
/// reasons; an unrecognized value would be silently mapped to a completed
/// terminal, so streaming rejects it to stay consistent with the finite path.
/// The legacy `function_call` reason is handled separately (fail closed) because
/// this converter does not translate legacy function calling.
fn is_recognized_finish_reason(reason: &str) -> bool {
    matches!(reason, "stop" | "length" | "tool_calls" | "content_filter")
}

/// Map a conversion error to a client-safe failure message that never leaks
/// upstream bytes.
fn failure_message(error: &ConvertError) -> &'static str {
    match error {
        ConvertError::MalformedJson => "upstream returned a malformed streaming chunk",
        ConvertError::MultipleChoices | ConvertError::InvalidChoiceIndex => {
            "upstream returned an unsupported multi-choice stream"
        },
        ConvertError::InconsistentMetadata => "upstream stream metadata changed mid-response",
        ConvertError::MissingChunkMetadata => "upstream stream omitted required chunk metadata",
        ConvertError::UnexpectedObject => "upstream returned an unexpected stream object",
        ConvertError::DataAfterFinish => "upstream sent data after the finish reason",
        ConvertError::DataAfterTerminal => "upstream sent data after a terminal streaming event",
        ConvertError::NameAfterArguments
        | ConvertError::IdAfterArguments
        | ConvertError::ToolCallMissingIdentity
        | ConvertError::ToolCallMissingIndex => "upstream sent an invalid tool-call stream",
        ConvertError::LegacyFunctionCall => "upstream used unsupported legacy function calling",
        ConvertError::BufferOverflow => "upstream stream exceeded the SSE buffer limit",
        ConvertError::EventLimit => "upstream stream exceeded the event limit",
        ConvertError::FrameSizeLimit => "upstream stream exceeded the per-event size limit",
        ConvertError::FrameLimit => "upstream stream exceeded the frame limit",
        ConvertError::ByteLimit => "upstream stream exceeded the response size limit",
        ConvertError::ToolArgumentLimit => "upstream tool-call arguments exceeded the size limit",
        ConvertError::ToolCountLimit => "upstream stream exceeded the tool-call limit",
        ConvertError::Timeout => "upstream stream exceeded the time limit",
        ConvertError::MissingFinishReason => "upstream stream ended without a finish reason",
        ConvertError::UnknownFinishReason => "upstream stream used an unrecognized finish reason",
        ConvertError::IncompleteFrame => "upstream stream ended with an incomplete frame",
        ConvertError::Serialize(_) => "internal serialization error",
    }
}
