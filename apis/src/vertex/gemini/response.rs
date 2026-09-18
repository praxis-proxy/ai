// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI Gemini response to OpenAI Chat Completions translation.
//!
//! Converts `generateContent` responses: `candidates` → `choices`,
//! `usageMetadata` → `usage`, `finishReason` mapping, and
//! `functionCall` parts → `tool_calls`.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
};

use serde_json::{Map, Value, json};
use tracing::warn;

// -----------------------------------------------------------------------------
// Tool-call Slots
// -----------------------------------------------------------------------------

/// Disambiguator mixed into minted OpenAI `tool_calls[].id` values.
///
/// Not used as the id itself: a process-global counter restarts per
/// replica and would collide across pods. Combined with pid + time it
/// only breaks ties within one process.
static TOOL_CALL_ID_SEQ: AtomicU64 = AtomicU64::new(1);

/// Per-stream translation state. Lives in filter `StreamState` so tool-call
/// slots and the OpenAI completion `id` survive across SSE frames.
#[derive(Debug)]
pub(crate) struct StreamTranslateState {
    /// Unix timestamp fixed at stream start.
    pub(crate) created: u64,
    /// OpenAI chunk `id`, locked on the first frame and never swapped.
    completion_id: Option<String>,
    /// Tool-call slots per candidate (`index` is the inner vec position).
    slots_by_candidate: BTreeMap<u64, Vec<ToolCallSlot>>,
    /// Candidate indices that have already emitted at least one delta.
    started_candidates: BTreeSet<u64>,
    /// Client asked for `stream_options.include_usage`.
    include_usage: bool,
    /// Latest Vertex `usageMetadata`, converted to OpenAI `usage`.
    ///
    /// Vertex sends cumulative counts on the last frame (often together
    /// with `finishReason`). OpenAI wants them on a *separate* trailing
    /// chunk with `choices: []`, so this is stashed rather than emitted
    /// inline.
    usage: Option<Value>,
}

/// One OpenAI `tool_calls[]` entry assembled across Gemini SSE frames.
#[derive(Debug)]
struct ToolCallSlot {
    /// Id sent to the Chat Completions client (`functionCall.id` or minted).
    openai_id: String,
    /// Vertex `functionCall.id` used to match later SSE frames of this call.
    google_id: Option<String>,
    /// Function name, filled when Gemini first sends it.
    name: String,
    /// `true` after the first delta that carried `id` / `type`.
    started: bool,
    /// `true` after `function.name` has been sent on a delta.
    name_emitted: bool,
    /// `Some` after `extra_content.google.thought_signature` was sent.
    thought_sent: Option<()>,
    /// Serialized Gemini `args` already sent as OpenAI `arguments`.
    ///
    /// OpenAI clients concatenate argument strings; Gemini often sends a
    /// complete JSON object per frame. Emit arguments once.
    emitted_args: Option<String>,
}

impl StreamTranslateState {
    /// Create state for a new SSE response.
    pub(crate) fn new() -> Self {
        Self {
            created: created_timestamp(),
            completion_id: None,
            slots_by_candidate: BTreeMap::new(),
            started_candidates: BTreeSet::new(),
            include_usage: false,
            usage: None,
        }
    }

    /// Enable the OpenAI trailing usage chunk for this stream.
    pub(crate) fn set_include_usage(&mut self, include_usage: bool) {
        self.include_usage = include_usage;
    }

    /// Record Vertex `usageMetadata` from this frame. Later frames
    /// overwrite earlier ones; Vertex counts are cumulative.
    ///
    /// No-op when `include_usage` is `false` — the allocation is
    /// skipped entirely since the stash will never be read.
    fn capture_usage(&mut self, obj: Option<&Map<String, Value>>) {
        if !self.include_usage {
            return;
        }
        if let Some(usage) = obj.and_then(|o| o.get("usageMetadata")) {
            self.usage = Some(convert_usage(usage));
        }
    }

    /// Set the OpenAI completion id once: Vertex `responseId` when the first
    /// frame has one, otherwise the shared fallback. Later frames cannot
    /// change it (a late `responseId` must not rewrite earlier chunk ids).
    fn ensure_completion_id(&mut self, response_id: Option<&str>) {
        if self.completion_id.is_none() {
            self.completion_id = Some(
                response_id
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map_or_else(|| super::FALLBACK_RESPONSE_ID.to_owned(), str::to_owned),
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Public Entry Point
// -----------------------------------------------------------------------------

/// Transform a Gemini `generateContent` response body into an OpenAI
/// Chat Completions response.
///
/// `model` is the model name extracted from the original request
/// (Gemini responses carry `modelVersion` which may differ from what
/// the client sent).
///
/// # Errors
///
/// Returns a human-readable error string when the body is not valid JSON.
pub(crate) fn transform_response(body: &[u8], model: &str) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let obj = value.as_object();

    let id = obj
        .and_then(|o| o.get("responseId"))
        .and_then(Value::as_str)
        .unwrap_or(super::FALLBACK_RESPONSE_ID);

    let candidates = obj.and_then(|o| o.get("candidates")).and_then(Value::as_array);

    if candidates.is_none() {
        reject_upstream_error_frame(obj)?;
    }

    let choices = candidates.map(|c| convert_candidates(c)).unwrap_or_default();

    let usage = obj
        .and_then(|o| o.get("usageMetadata"))
        .map_or_else(|| json!({}), convert_usage);

    let response = json!({
        "id": id,
        "object": "chat.completion",
        "created": created_timestamp(),
        "model": model,
        "choices": choices,
        "usage": usage,
    });

    serde_json::to_vec(&response).map_err(|e| format!("serialization failed: {e}"))
}

// -----------------------------------------------------------------------------
// Candidates → Choices
// -----------------------------------------------------------------------------

/// Convert Gemini `candidates` array to OpenAI `choices`.
fn convert_candidates(candidates: &[Value]) -> Vec<Value> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| convert_candidate(c, i))
        .collect()
}

/// Convert a single Gemini candidate to an OpenAI choice.
fn convert_candidate(candidate: &Value, default_index: usize) -> Value {
    let index = candidate
        .get("index")
        .and_then(Value::as_u64)
        .unwrap_or(default_index as u64);

    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    let (content, tool_calls) = extract_content_and_tool_calls(parts);
    let has_tool_calls = !tool_calls.is_empty();

    let finish_reason = convert_finish_reason(candidate.get("finishReason").and_then(Value::as_str), has_tool_calls);

    let mut message = Map::new();
    message.insert("role".to_owned(), Value::String("assistant".to_owned()));
    message.insert("content".to_owned(), content);

    if has_tool_calls {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }

    let mut choice = json!({
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
    });

    if let Some(logprobs) = candidate.get("logprobsResult").and_then(convert_logprobs_result)
        && let Some(obj) = choice.as_object_mut()
    {
        obj.insert("logprobs".to_owned(), logprobs);
    }

    choice
}

// -----------------------------------------------------------------------------
// Parts → content + tool_calls
// -----------------------------------------------------------------------------

/// Extract text content and tool calls from Gemini response parts.
///
/// - `text` parts → concatenated into a single string (or `null` if none)
/// - `functionCall` parts → converted to OpenAI `tool_calls` entries
/// - `thoughtSignature` on a function-call part is copied onto the tool call as
///   `extra_content.google.thought_signature` so the client can send it back on the next turn (Gemini 3 rejects
///   continuations without it)
///
/// Each `functionCall` becomes its own OpenAI tool call. Prefer Vertex's
/// `functionCall.id` when present; otherwise mint a unique id. Do not derive
/// the id from `responseId` or the function name — those collide across rounds
/// or parallel same-name calls.
fn extract_content_and_tool_calls(parts: Option<&Vec<Value>>) -> (Value, Vec<Value>) {
    let Some(parts) = parts else {
        return (Value::Null, Vec::new());
    };

    let mut text_segments = Vec::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            text_segments.push(text);
        }

        if let Some(fc) = part.get("functionCall").and_then(Value::as_object) {
            tool_calls.push(convert_function_call_to_tool_call(part, fc));
        }
    }

    let content = if text_segments.is_empty() {
        Value::Null
    } else {
        Value::String(text_segments.join(""))
    };

    (content, tool_calls)
}

/// Convert a Gemini `functionCall` part to an OpenAI `tool_calls` entry.
///
/// `thoughtSignature` lives on the Gemini `Part` (sibling of
/// `functionCall`), not inside the call object. Chat Completions has no
/// native field for it; Google's OpenAI-compatible shim stashes the
/// value at `extra_content.google.thought_signature`.
fn convert_function_call_to_tool_call(part: &Value, fc: &Map<String, Value>) -> Value {
    let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
    let args = serialize_function_args(fc);

    let mut call = json!({
        "id": openai_tool_call_id(fc),
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        }
    });

    if let Some(sig) = thought_signature(part, fc)
        && let Some(obj) = call.as_object_mut()
    {
        obj.insert(
            "extra_content".to_owned(),
            json!({ "google": { "thought_signature": sig } }),
        );
    }

    call
}

/// Gemini REST puts `thoughtSignature` on the Part; some payloads nest
/// it inside `functionCall`. Either form is copied through unchanged.
fn thought_signature(part: &Value, fc: &Map<String, Value>) -> Option<Value> {
    part.get("thoughtSignature")
        .or_else(|| fc.get("thoughtSignature"))
        .filter(|v| !v.is_null())
        .cloned()
}

/// OpenAI `tool_calls[].id`: Vertex's `functionCall.id` when it is a
/// non-empty string, otherwise a minted unique value.
fn openai_tool_call_id(fc: &Map<String, Value>) -> String {
    google_function_call_id(fc).map_or_else(mint_tool_call_id, ToOwned::to_owned)
}

/// Non-empty Vertex `functionCall.id`, if present.
fn google_function_call_id(fc: &Map<String, Value>) -> Option<&str> {
    fc.get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Serialize Gemini `functionCall.args` to the OpenAI arguments string.
fn serialize_function_args(fc: &Map<String, Value>) -> String {
    fc.get("args").map_or_else(
        || "{}".to_owned(),
        |a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_owned()),
    )
}

/// Mint an OpenAI-shaped tool-call id that is unique across rounds and
/// replicas without embedding `responseId` (that field can arrive on a
/// later SSE frame and must not rewrite ids already streamed).
fn mint_tool_call_id() -> String {
    let seq = TOOL_CALL_ID_SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("call_{:x}{nanos:x}{seq:x}", std::process::id())
}

// -----------------------------------------------------------------------------
// Finish Reason
// -----------------------------------------------------------------------------

/// Map Gemini `finishReason` to OpenAI `finish_reason`.
///
/// When the response contains `functionCall` parts, the finish reason
/// is overridden to `"tool_calls"` regardless of the Gemini value
/// (Gemini uses `STOP` for both text and function call completions).
///
/// Gemini `FinishReason` enum (full list, as of Vertex AI REST v1):
/// `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`,
/// `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`,
/// `IMAGE_SAFETY`, `MODEL_ARMOR`, `FINISH_REASON_UNSPECIFIED`.
///
/// `MODEL_ARMOR` indicates the response was blocked by [Model Armor],
/// Google's enterprise content-safety layer. It must map to
/// `"content_filter"` — treating it as `"stop"` misreports a blocked
/// response as a normal completion.
///
/// [Model Armor]: https://cloud.google.com/model-armor/docs/
fn convert_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_calls";
    }

    match reason {
        Some("MAX_TOKENS") => "length",
        Some(
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" | "LANGUAGE"
            | "OTHER" | "MODEL_ARMOR",
        ) => "content_filter",
        // STOP, MALFORMED_FUNCTION_CALL, FINISH_REASON_UNSPECIFIED,
        // unknown, or absent all map to "stop".
        _ => "stop",
    }
}

// -----------------------------------------------------------------------------
// Usage
// -----------------------------------------------------------------------------

/// Convert Gemini `usageMetadata` to OpenAI `usage`.
fn convert_usage(usage: &Value) -> Value {
    let prompt = usage.get("promptTokenCount").and_then(Value::as_u64).unwrap_or(0);
    let completion = usage.get("candidatesTokenCount").and_then(Value::as_u64).unwrap_or(0);
    let total = usage
        .get("totalTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or(prompt + completion);

    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    })
}

// -----------------------------------------------------------------------------
// Logprobs
// -----------------------------------------------------------------------------

/// Convert Gemini `logprobsResult` to OpenAI `logprobs`.
///
/// Gemini returns two parallel arrays: `chosenCandidates` (the selected
/// token at each decoding step) and `topCandidates` (ranked alternatives
/// at each step). OpenAI merges these into a single `content` array
/// where each entry carries the chosen token plus a nested `top_logprobs`
/// with alternatives.
///
/// Returns `None` when there is nothing to convert.
fn convert_logprobs_result(logprobs_result: &Value) -> Option<Value> {
    let chosen = logprobs_result.get("chosenCandidates").and_then(Value::as_array)?;
    if chosen.is_empty() {
        return None;
    }

    let top = logprobs_result.get("topCandidates").and_then(Value::as_array);

    let content: Vec<Value> = chosen
        .iter()
        .enumerate()
        .map(|(i, chosen_token)| {
            let token = chosen_token.get("token").and_then(Value::as_str).unwrap_or("");
            let logprob = chosen_token
                .get("logProbability")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);

            let top_logprobs: Vec<Value> = top
                .and_then(|t| t.get(i))
                .and_then(|tc| tc.get("candidates"))
                .and_then(Value::as_array)
                .map(|candidates| candidates.iter().map(convert_logprob_entry).collect())
                .unwrap_or_default();

            json!({
                "token": token,
                "logprob": logprob,
                "bytes": token_utf8_bytes(token),
                "top_logprobs": top_logprobs,
            })
        })
        .collect();

    Some(json!({ "content": content }))
}

/// Convert a single Gemini token candidate to an OpenAI `top_logprobs` entry.
fn convert_logprob_entry(entry: &Value) -> Value {
    let token = entry.get("token").and_then(Value::as_str).unwrap_or("");
    json!({
        "token": token,
        "logprob": entry.get("logProbability").and_then(Value::as_f64).unwrap_or(0.0),
        "bytes": token_utf8_bytes(token),
    })
}

/// OpenAI `bytes` field: UTF-8 byte representation of the token.
fn token_utf8_bytes(token: &str) -> Value {
    Value::Array(token.bytes().map(|b| Value::Number(b.into())).collect())
}

// -----------------------------------------------------------------------------
// Streaming Chunk Translation
// -----------------------------------------------------------------------------

/// Transform a single Gemini SSE chunk into an OpenAI
/// `chat.completion.chunk` JSON string.
///
/// Gemini `streamGenerateContent` emits SSE `data:` lines where each
/// payload is a partial `GenerateContentResponse`. This converts one
/// such payload into the OpenAI streaming shape (`delta` instead of
/// `message`, `object: "chat.completion.chunk"`).
///
/// `state` holds the sticky completion id and tool-call slots so a
/// `functionCall` that arrives across frames keeps the same OpenAI `id`
/// and `index`. First-delta state is tracked per candidate so a candidate
/// that first appears in a later frame still includes `delta.role`.
///
/// Returns `Ok(None)` for frames with no `candidates` that are not
/// errors (usage-only / empty JSON). Vertex often sends `usageMetadata`
/// on such a frame, or on the last candidate frame; it is stashed for
/// [`take_stream_usage_chunk`].
pub(crate) fn transform_stream_chunk(
    data: &[u8],
    model: &str,
    state: &mut StreamTranslateState,
) -> Result<Option<Vec<u8>>, String> {
    let value: Value = serde_json::from_slice(data).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = value.as_object();

    state.ensure_completion_id(obj.and_then(|o| o.get("responseId")).and_then(Value::as_str));
    state.capture_usage(obj);

    let candidates = obj.and_then(|o| o.get("candidates")).and_then(Value::as_array);

    // Detect upstream errors and content blocks that arrive as valid JSON
    // but carry no candidates. Without this check these frames silently
    // produce empty deltas that look like successful completions.
    if candidates.is_none_or(Vec::is_empty) {
        reject_upstream_error_frame(obj)?;
        return Ok(None);
    }

    let choices = candidates
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(position, candidate)| build_stream_choice(candidate, position as u64, state))
        .collect::<Vec<_>>();

    let mut chunk = json!({
        "id": state.completion_id.as_deref().unwrap_or(super::FALLBACK_RESPONSE_ID),
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": model,
        "choices": choices,
    });
    // OpenAI: when include_usage is set, every content chunk carries
    // `usage: null`; the populated object is a later, separate chunk.
    if state.include_usage
        && let Some(obj) = chunk.as_object_mut()
    {
        obj.insert("usage".to_owned(), Value::Null);
    }

    serde_json::to_vec(&chunk)
        .map(Some)
        .map_err(|e| format!("serialization failed: {e}"))
}

/// Build the trailing OpenAI usage chunk (`choices: []`) if the client
/// asked for `include_usage` and Vertex sent `usageMetadata`.
///
/// Call once at `end_of_stream`, immediately before `[DONE]`. Do not
/// call after a stream error — OpenAI omits the usage chunk when the
/// stream is interrupted.
pub(crate) fn take_stream_usage_chunk(state: &mut StreamTranslateState, model: &str) -> Option<Vec<u8>> {
    if !state.include_usage {
        return None;
    }
    let usage = state.usage.take()?;
    let chunk = json!({
        "id": state.completion_id.as_deref().unwrap_or(super::FALLBACK_RESPONSE_ID),
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": model,
        "choices": [],
        "usage": usage,
    });
    serde_json::to_vec(&chunk)
        .map_err(|e| warn!(error = %e, "failed to serialize streaming usage chunk"))
        .ok()
}

/// Build a single OpenAI streaming choice from a Gemini candidate.
fn build_stream_choice(candidate: &Value, default_index: u64, state: &mut StreamTranslateState) -> Value {
    let candidate_index = candidate.get("index").and_then(Value::as_u64).unwrap_or(default_index);
    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    let finish_reason = candidate.get("finishReason").and_then(Value::as_str);
    let is_first_delta = state.started_candidates.insert(candidate_index);
    let slots = state.slots_by_candidate.entry(candidate_index).or_default();
    let (content, tool_calls) = extract_stream_content_and_tool_calls(parts, slots);
    let has_tool_calls = !tool_calls.is_empty() || !slots.is_empty();
    let delta = build_stream_delta(is_first_delta, content, tool_calls);
    let finish = finish_reason.map_or(Value::Null, |_| {
        Value::String(convert_finish_reason(finish_reason, has_tool_calls).to_owned())
    });

    let logprobs = candidate
        .get("logprobsResult")
        .and_then(convert_logprobs_result)
        .unwrap_or(Value::Null);

    json!({ "index": candidate_index, "delta": delta, "logprobs": logprobs, "finish_reason": finish })
}

/// Reject SSE frames that carry an upstream error or prompt-level content block.
///
/// Vertex can send these on an already-established `text/event-stream`
/// connection (HTTP 200). Without this check the frame silently becomes
/// an empty `chat.completion.chunk` delta followed by `[DONE]`, hiding
/// the failure from the client.
///
/// Only frames **without** `candidates` are rejected. A `finishReason: SAFETY`
/// on an actual candidate is translated normally (`content_filter`).
/// Usage-only frames (`usageMetadata` without `candidates` or `error`) are
/// legitimate and pass through.
fn reject_upstream_error_frame(obj: Option<&Map<String, Value>>) -> Result<(), String> {
    let Some(obj) = obj else {
        return Ok(());
    };

    // Vertex error: {"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}
    if let Some(error) = obj.get("error") {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("upstream returned an error frame");
        return Err(format!("Vertex error in SSE frame: {message}"));
    }

    // Prompt blocked: {"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"..."}}
    // Sent only in the first stream chunk when no candidates are generated.
    if let Some(feedback) = obj.get("promptFeedback")
        && feedback.get("blockReason").and_then(Value::as_str).is_some()
    {
        let message = feedback
            .get("blockReasonMessage")
            .and_then(Value::as_str)
            .unwrap_or("prompt blocked by content filter");
        return Err(format!("prompt blocked: {message}"));
    }

    Ok(())
}

/// Extract text and streaming tool-call deltas from one Gemini frame.
///
/// Tool-call `index` is the slot position and is set on every delta —
/// this repo's Chat Completions stream consumer fails closed when
/// `index` is missing and would otherwise merge parallel calls.
fn extract_stream_content_and_tool_calls(
    parts: Option<&Vec<Value>>,
    slots: &mut Vec<ToolCallSlot>,
) -> (Value, Vec<Value>) {
    let Some(parts) = parts else {
        return (Value::Null, Vec::new());
    };

    let mut text_segments = Vec::new();
    let mut tool_calls = Vec::new();

    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            text_segments.push(text);
        }

        if let Some(fc) = part.get("functionCall").and_then(Value::as_object)
            && let Some(delta) = stream_tool_call_delta(part, fc, slots)
        {
            tool_calls.push(delta);
        }
    }

    let content = if text_segments.is_empty() {
        Value::Null
    } else {
        Value::String(text_segments.join(""))
    };

    (content, tool_calls)
}

/// Resolve a Gemini `functionCall` onto a stream slot and emit one OpenAI
/// `tool_calls` delta.
///
/// Matching order:
/// 1. Existing slot with the same Vertex `functionCall.id`
/// 2. Nameless / args-only part → the most recently opened slot
/// 3. Otherwise a new slot (Google id if present, else minted)
fn stream_tool_call_delta(part: &Value, fc: &Map<String, Value>, slots: &mut Vec<ToolCallSlot>) -> Option<Value> {
    let index = resolve_stream_slot(fc, slots);
    #[expect(
        clippy::indexing_slicing,
        reason = "resolve_stream_slot returns an existing index or the slot just pushed"
    )]
    let slot = &mut slots[index];
    let first = !slot.started;
    slot.started = true;
    apply_function_call_to_slot(slot, fc);
    let function = take_stream_function_delta(slot, fc);
    let thought = take_stream_thought_signature(slot, part, fc);
    if !first && function.is_empty() && thought.is_none() {
        return None;
    }
    Some(assemble_stream_tool_call(index, slot, first, function, thought))
}

/// Build one OpenAI streaming `tool_calls[]` object from slot fields.
fn assemble_stream_tool_call(
    index: usize,
    slot: &ToolCallSlot,
    first: bool,
    function: Map<String, Value>,
    thought: Option<Value>,
) -> Value {
    let mut call = Map::new();
    call.insert("index".to_owned(), json!(index));
    if first {
        attach_first_stream_tool_call_fields(&mut call, slot);
    }
    if let Some(sig) = thought {
        call.insert(
            "extra_content".to_owned(),
            json!({ "google": { "thought_signature": sig } }),
        );
    }
    if !function.is_empty() {
        call.insert("function".to_owned(), Value::Object(function));
    }
    Value::Object(call)
}

/// Copy name from this Gemini part onto the slot when present.
fn apply_function_call_to_slot(slot: &mut ToolCallSlot, fc: &Map<String, Value>) {
    if let Some(name) = fc.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()) {
        name.clone_into(&mut slot.name);
    }
}

/// Fields for `delta.tool_calls[].function` that have not been sent yet.
fn take_stream_function_delta(slot: &mut ToolCallSlot, fc: &Map<String, Value>) -> Map<String, Value> {
    let mut function = Map::new();
    if !slot.name_emitted && !slot.name.is_empty() {
        function.insert("name".to_owned(), Value::String(slot.name.clone()));
        slot.name_emitted = true;
    }
    if slot.emitted_args.is_none() && fc.get("args").is_some() {
        let args = serialize_function_args(fc);
        function.insert("arguments".to_owned(), Value::String(args.clone()));
        slot.emitted_args = Some(args);
    }
    function
}

/// Copy `thoughtSignature` the first time it appears on this slot, including
/// a later SSE frame of the same `functionCall.id`.
fn take_stream_thought_signature(slot: &mut ToolCallSlot, part: &Value, fc: &Map<String, Value>) -> Option<Value> {
    if slot.thought_sent.is_some() {
        return None;
    }
    let sig = thought_signature(part, fc)?;
    slot.thought_sent = Some(());
    Some(sig)
}

/// `id` and `type` belong only on the first delta of a slot.
fn attach_first_stream_tool_call_fields(call: &mut Map<String, Value>, slot: &ToolCallSlot) {
    call.insert("id".to_owned(), Value::String(slot.openai_id.clone()));
    call.insert("type".to_owned(), Value::String("function".to_owned()));
}

/// Find or create the slot for this `functionCall`.
fn resolve_stream_slot(fc: &Map<String, Value>, slots: &mut Vec<ToolCallSlot>) -> usize {
    if let Some(gid) = google_function_call_id(fc)
        && let Some(index) = slots.iter().position(|s| s.google_id.as_deref() == Some(gid))
    {
        return index;
    }

    let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
    if name.is_empty() && google_function_call_id(fc).is_none() && !slots.is_empty() {
        return slots.len() - 1;
    }

    let google_id = google_function_call_id(fc).map(ToOwned::to_owned);
    let openai_id = google_id.clone().unwrap_or_else(mint_tool_call_id);
    slots.push(ToolCallSlot {
        openai_id,
        google_id,
        name: name.to_owned(),
        started: false,
        name_emitted: false,
        thought_sent: None,
        emitted_args: None,
    });
    slots.len() - 1
}

/// Build the `delta` object for a streaming chunk.
fn build_stream_delta(is_first: bool, content: Value, tool_calls: Vec<Value>) -> Map<String, Value> {
    let mut delta = Map::new();
    if is_first {
        delta.insert("role".to_owned(), Value::String("assistant".to_owned()));
    }
    if !content.is_null() {
        delta.insert("content".to_owned(), content);
    }
    if !tool_calls.is_empty() {
        delta.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    delta
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Current Unix timestamp (seconds since epoch).
pub(crate) fn created_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    fn new_stream_state(created: u64) -> StreamTranslateState {
        let mut state = StreamTranslateState::new();
        state.created = created;
        state
    }

    fn translate_stream(state: &mut StreamTranslateState, data: &[u8]) -> Value {
        let output = transform_stream_chunk(data, "gemini-1.5-pro", state).unwrap().unwrap();
        serde_json::from_slice(&output).unwrap()
    }

    #[test]
    fn basic_text_response() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7}
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["model"], "gemini-1.5-pro");
        assert_eq!(parsed["choices"][0]["message"]["role"], "assistant");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["choices"][0]["index"], 0);
        assert_eq!(parsed["usage"]["prompt_tokens"], 5);
        assert_eq!(parsed["usage"]["completion_tokens"], 2);
        assert_eq!(parsed["usage"]["total_tokens"], 7);
    }

    #[test]
    fn response_id_used() {
        let body = br#"{
            "responseId": "resp_abc123",
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["id"], "resp_abc123");
    }

    #[test]
    fn missing_response_id_uses_fallback() {
        let body = br#"{"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-vertex");
    }

    #[test]
    fn function_call_response() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(
            parsed["choices"][0]["finish_reason"], "tool_calls",
            "functionCall should override finish_reason to tool_calls"
        );
        assert!(parsed["choices"][0]["message"]["content"].is_null());

        let tc = &parsed["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        assert!(tc.get("index").is_none(), "non-stream tool_calls have no index");
        let id = tc["id"].as_str().unwrap();
        assert!(id.starts_with("call_"), "minted ids should look OpenAI-shaped: {id}");
        assert_ne!(id, "get_weather", "function name must not be used as tool_call id");
        assert!(!id.contains("sig"), "thought signatures must not be stuffed into id");

        let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "NYC");
    }

    #[test]
    fn function_call_prefers_google_id() {
        let body = br#"{
            "responseId": "resp-aaa",
            "candidates": [{
                "content": {"parts": [{"functionCall": {
                    "id": "adk-call-123",
                    "name": "get_weather",
                    "args": {}
                }}]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let id = parsed["choices"][0]["message"]["tool_calls"][0]["id"].as_str().unwrap();
        assert_eq!(id, "adk-call-123");
        assert!(
            !id.contains("resp-aaa"),
            "OpenAI tool_call id must not embed responseId"
        );
    }

    #[test]
    fn function_call_blank_google_id_is_minted() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{"functionCall": {"id": "  ", "name": "get_weather", "args": {}}}]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let id = parsed["choices"][0]["message"]["tool_calls"][0]["id"].as_str().unwrap();
        assert_ne!(id.trim(), "");
        assert_ne!(id, "  ");
        assert!(id.starts_with("call_"));
    }

    #[test]
    fn function_call_ids_differ_across_rounds() {
        let round = |response_id: &str, name: &str| {
            let body = json!({
                "responseId": response_id,
                "candidates": [{
                    "content": {
                        "parts": [{ "functionCall": { "name": name, "args": {} } }]
                    }
                }]
            });
            let output = transform_response(&serde_json::to_vec(&body).unwrap(), "gemini-1.5-pro").unwrap();
            let parsed: Value = serde_json::from_slice(&output).unwrap();
            parsed["choices"][0]["message"]["tool_calls"][0]["id"]
                .as_str()
                .unwrap()
                .to_owned()
        };

        let first = round("resp-round1", "get_weather");
        let second = round("resp-round2", "get_calendar");
        assert_ne!(first, second, "tool-call ids must not restart at call_vertex_0");
        assert!(!first.contains("resp-round1"));
        assert!(!second.contains("resp-round2"));
    }

    #[test]
    fn function_call_ids_differ_when_response_id_is_absent() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {}}}]},
                "finishReason": "STOP"
            }]
        }"#;
        let first = transform_response(body, "gemini-1.5-pro").unwrap();
        let second = transform_response(body, "gemini-1.5-pro").unwrap();
        let first_id = serde_json::from_slice::<Value>(&first).unwrap()["choices"][0]["message"]["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .to_owned();
        let second_id =
            serde_json::from_slice::<Value>(&second).unwrap()["choices"][0]["message"]["tool_calls"][0]["id"]
                .as_str()
                .unwrap()
                .to_owned();
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn parallel_function_calls_get_distinct_ids() {
        let body = br#"{
            "responseId": "resp-p",
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "get_weather", "args": {}}},
                    {"functionCall": {"name": "get_calendar", "args": {}}}
                ]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let calls = parsed["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[1]["function"]["name"], "get_calendar");
    }

    #[test]
    fn parallel_same_name_calls_get_distinct_ids() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}},
                    {"functionCall": {"name": "get_weather", "args": {"city": "SFO"}}}
                ]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let calls = parsed["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_ne!(calls[0]["id"], calls[1]["id"]);
    }

    #[test]
    fn function_call_id_not_thought_signature() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"id": "google-fc-1", "name": "get_weather", "args": {}},
                    "thoughtSignature": "sig-abc"
                }]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let tc = &parsed["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "google-fc-1");
        assert_eq!(tc["extra_content"]["google"]["thought_signature"], "sig-abc");
        assert!(!tc["id"].as_str().unwrap().contains("sig-abc"));
    }

    #[test]
    fn nested_thought_signature_inside_function_call() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {},
                        "thoughtSignature": "nested-sig"
                    }
                }]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let tc = &parsed["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["extra_content"]["google"]["thought_signature"], "nested-sig");
        assert!(!tc["id"].as_str().unwrap().contains("nested-sig"));
    }

    #[test]
    fn function_call_preserves_thought_signature() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "get_weather", "args": {"city": "NYC"}},
                    "thoughtSignature": "sig-abc"
                }]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let tc = &parsed["choices"][0]["message"]["tool_calls"][0];

        assert_eq!(tc["extra_content"]["google"]["thought_signature"], "sig-abc");
        assert_eq!(tc["function"]["name"], "get_weather");
    }

    #[test]
    fn text_and_function_call_combined() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [
                    {"text": "Let me check."},
                    {"functionCall": {"name": "search", "args": {"q": "rust"}}}
                ]},
                "finishReason": "STOP"
            }]
        }"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["choices"][0]["message"]["content"], "Let me check.");
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(parsed["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn max_tokens_finish_reason() {
        let body = br#"{"candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": "MAX_TOKENS"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn safety_finish_reason() {
        let body = br#"{"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["choices"][0]["finish_reason"], "content_filter");
    }

    #[test]
    fn language_and_other_finish_reasons_are_content_filter() {
        for reason in ["LANGUAGE", "OTHER"] {
            let body = format!(r#"{{"candidates": [{{"content": {{"parts": []}}, "finishReason": "{reason}"}}]}}"#);
            let output = transform_response(body.as_bytes(), "gemini-1.5-pro").unwrap();
            let parsed: Value = serde_json::from_slice(&output).unwrap();

            assert_eq!(
                parsed["choices"][0]["finish_reason"], "content_filter",
                "{reason} should map to content_filter"
            );
        }
    }

    #[test]
    fn model_armor_finish_reason_maps_to_content_filter() {
        // MODEL_ARMOR means the response was blocked by Google's Model Armor
        // enterprise safety layer. It must NOT silently map to "stop" — that
        // would misreport a blocked response as a normal completion.
        let body = br#"{"candidates": [{"content": {"parts": []}, "finishReason": "MODEL_ARMOR"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(
            parsed["choices"][0]["finish_reason"], "content_filter",
            "MODEL_ARMOR must map to content_filter, not stop"
        );
    }

    #[test]
    fn malformed_function_call_maps_to_stop() {
        let body = br#"{"candidates": [{"content": {"parts": []}, "finishReason": "MALFORMED_FUNCTION_CALL"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn missing_usage_returns_empty_object() {
        let body = br#"{"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed["usage"].is_object());
    }

    #[test]
    fn logprobs_translated_from_gemini_format() {
        let body = br#"{
            "candidates": [{
                "content": {"parts": [{"text": "Hi"}]},
                "finishReason": "STOP",
                "logprobsResult": {
                    "chosenCandidates": [
                        {"token": "Hi", "tokenId": 1, "logProbability": -0.05}
                    ],
                    "topCandidates": [
                        {"candidates": [
                            {"token": "Hi", "tokenId": 1, "logProbability": -0.05},
                            {"token": "Hello", "tokenId": 2, "logProbability": -1.2}
                        ]}
                    ]
                }
            }]
        }"#;
        let output = transform_response(body, "gemini-2.0-flash").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();
        let lp = &parsed["choices"][0]["logprobs"]["content"][0];

        assert_eq!(lp["token"], "Hi");
        assert_eq!(lp["logprob"], -0.05);
        assert_eq!(lp["bytes"], json!([72, 105]));
        assert_eq!(lp["top_logprobs"].as_array().unwrap().len(), 2);
        assert_eq!(lp["top_logprobs"][1]["token"], "Hello");
    }

    #[test]
    fn logprobs_absent_when_not_requested() {
        let body = br#"{"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed["choices"][0].get("logprobs").is_none());
    }

    #[test]
    fn empty_candidates_returns_empty_choices() {
        let body = br#"{"candidates": []}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed["choices"].as_array().unwrap().is_empty());
    }

    #[test]
    fn no_content_parts_returns_null_content() {
        let body = br#"{"candidates": [{"finishReason": "SAFETY"}]}"#;
        let output = transform_response(body, "gemini-1.5-pro").unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn invalid_json_fails() {
        let err = transform_response(b"not json", "gemini-1.5-pro").unwrap_err();
        assert!(err.contains("invalid JSON"));
    }

    #[test]
    fn prompt_blocked_without_candidates_returns_error() {
        let body = br#"{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"prompt violates policy"}}"#;
        let err = transform_response(body, "gemini-2.0-flash").unwrap_err();
        assert!(err.contains("prompt"), "{err}");
    }

    #[test]
    fn vertex_error_in_200_returns_error() {
        let body = br#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#;
        let err = transform_response(body, "gemini-2.0-flash").unwrap_err();
        assert!(err.contains("Quota exceeded"), "{err}");
    }

    // -------------------------------------------------------------------------
    // Streaming chunks
    // -------------------------------------------------------------------------

    #[test]
    fn stream_chunk_first_includes_role() {
        let mut state = new_stream_state(1_700_000_000);
        let parsed = translate_stream(&mut state, br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#);

        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["created"], 1_700_000_000);
        assert_eq!(parsed["id"], "chatcmpl-vertex");
        assert_eq!(parsed["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(parsed["choices"][0]["delta"]["content"], "Hi");
        assert!(parsed["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn stream_chunk_subsequent_omits_role() {
        let mut state = new_stream_state(1_700_000_000);
        drop(translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#,
        ));
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":" world"}]}}]}"#,
        );

        assert!(parsed["choices"][0]["delta"].get("role").is_none());
        assert_eq!(parsed["choices"][0]["delta"]["content"], " world");
    }

    #[test]
    fn stream_chunk_finish_reason() {
        let mut state = new_stream_state(1_700_000_000);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}"#,
        );

        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn stream_chunk_logprobs_translated() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]},"logprobsResult":{"chosenCandidates":[{"token":"Hi","logProbability":-0.1}],"topCandidates":[{"candidates":[{"token":"Hi","logProbability":-0.1}]}]}}]}"#,
        );

        let lp = &parsed["choices"][0]["logprobs"]["content"][0];
        assert_eq!(lp["token"], "Hi");
        assert_eq!(lp["logprob"], -0.1);
        assert!(lp["bytes"].is_array());
    }

    #[test]
    fn stream_chunk_logprobs_null_when_absent() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(&mut state, br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#);

        assert!(parsed["choices"][0]["logprobs"].is_null());
    }

    #[test]
    fn stream_function_call_sets_index_and_does_not_embed_response_id() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"responseId":"resp-s","candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{}}}]}}]}"#,
        );

        assert_eq!(parsed["id"], "resp-s");
        let tc = &parsed["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "search");
        let id = tc["id"].as_str().unwrap();
        assert_ne!(id, "search");
        assert!(!id.contains("resp-s"));
        assert_eq!(tc["function"]["arguments"], "{}");
    }

    #[test]
    fn stream_function_call_prefers_google_id() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"responseId":"resp-s","candidates":[{"content":{"parts":[{"functionCall":{"id":"vertex-fc-9","name":"search","args":{"q":"rust"}}}]}}]}"#,
        );
        let tc = &parsed["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "vertex-fc-9");
        assert_eq!(tc["index"], 0);
    }

    #[test]
    fn stream_parallel_function_calls_in_one_frame_get_distinct_indices() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[
                {"functionCall":{"name":"get_weather","args":{}}},
                {"functionCall":{"name":"get_calendar","args":{}}}
            ]}}]}"#,
        );
        let calls = parsed["choices"][0]["delta"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[1]["index"], 1);
        assert_ne!(calls[0]["id"], calls[1]["id"]);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[1]["function"]["name"], "get_calendar");
    }

    #[test]
    fn stream_two_frames_keep_stable_ids_and_indices() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"responseId":"resp-s","candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"NYC"}}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"responseId":"resp-s","candidates":[{"content":{"parts":[{"functionCall":{"name":"get_calendar","args":{}}}]}}]}"#,
        );

        let weather = &first["choices"][0]["delta"]["tool_calls"][0];
        let calendar = &second["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(weather["index"], 0);
        assert_eq!(calendar["index"], 1);
        assert_ne!(weather["id"], calendar["id"]);
        assert_eq!(weather["function"]["name"], "get_weather");
        assert_eq!(calendar["function"]["name"], "get_calendar");
        assert_eq!(first["id"], second["id"]);
        assert_eq!(first["id"], "resp-s");
    }

    #[test]
    fn stream_same_google_id_across_frames_reuses_slot() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-1","name":"search"}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-1","args":{"q":"rust"}}}]}}]}"#,
        );

        let start = &first["choices"][0]["delta"]["tool_calls"][0];
        let cont = &second["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(start["id"], "fc-1");
        assert_eq!(start["index"], 0);
        assert_eq!(start["function"]["name"], "search");
        assert!(start["function"].get("arguments").is_none());
        assert_eq!(cont["index"], 0);
        assert!(cont.get("id").is_none(), "continuation omits id");
        assert_eq!(cont["function"]["arguments"], r#"{"q":"rust"}"#);
    }

    #[test]
    fn stream_args_only_frame_attaches_to_last_slot() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search"}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"args":{"q":"praxis"}}}]}}]}"#,
        );

        let start = &first["choices"][0]["delta"]["tool_calls"][0];
        let cont = &second["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(start["index"], 0);
        assert_eq!(cont["index"], 0);
        assert!(cont.get("id").is_none(), "args-only continuation omits id");
        assert_eq!(cont["function"]["arguments"], r#"{"q":"praxis"}"#);
    }

    #[test]
    fn stream_duplicate_complete_call_does_not_emit_second_delta() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-dup","name":"search","args":{"q":"a"}}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-dup","name":"search","args":{"q":"a"}}}]}}]}"#,
        );

        assert_eq!(first["choices"][0]["delta"]["tool_calls"][0]["id"], "fc-dup");
        assert!(
            second["choices"][0]["delta"].get("tool_calls").is_none(),
            "identical follow-up must not concatenate a second JSON args blob"
        );
    }

    #[test]
    fn stream_completion_id_is_sticky_when_response_id_arrives_late() {
        let mut state = new_stream_state(1);
        let first = translate_stream(&mut state, br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#);
        let second = translate_stream(
            &mut state,
            br#"{"responseId":"resp-late","candidates":[{"content":{"parts":[{"text":"!"}]}}]}"#,
        );

        assert_eq!(first["id"], "chatcmpl-vertex");
        assert_eq!(second["id"], "chatcmpl-vertex");
        assert_ne!(second["id"], "resp-late");
    }

    #[test]
    fn stream_finish_after_tool_call_maps_to_tool_calls() {
        let mut state = new_stream_state(1);
        drop(translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{}}}]}}]}"#,
        ));
        let done = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#,
        );

        assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn stream_thought_signature_on_first_delta_only() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{
                "functionCall":{"id":"fc-sig","name":"search"},
                "thoughtSignature":"sig-abc"
            }]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-sig","args":{"q":"x"}}}]}}]}"#,
        );

        let start = &first["choices"][0]["delta"]["tool_calls"][0];
        let cont = &second["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(start["extra_content"]["google"]["thought_signature"], "sig-abc");
        assert!(!start["id"].as_str().unwrap().contains("sig-abc"));
        assert!(cont.get("extra_content").is_none());
    }

    #[test]
    fn stream_thought_signature_on_later_frame() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-sig","name":"search"}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{
                "functionCall":{"id":"fc-sig","args":{"q":"x"}},
                "thoughtSignature":"sig-late"
            }]}}]}"#,
        );

        let start = &first["choices"][0]["delta"]["tool_calls"][0];
        let cont = &second["choices"][0]["delta"]["tool_calls"][0];
        assert!(start.get("extra_content").is_none());
        assert_eq!(cont["index"], 0);
        assert_eq!(cont["extra_content"]["google"]["thought_signature"], "sig-late");
        assert_eq!(cont["function"]["arguments"], r#"{"q":"x"}"#);
        assert!(cont.get("id").is_none());
    }

    #[test]
    fn stream_thought_signature_alone_on_later_frame() {
        let mut state = new_stream_state(1);
        drop(translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"functionCall":{"id":"fc-sig","name":"search","args":{}}}]}}]}"#,
        ));
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{
                "functionCall":{"id":"fc-sig"},
                "thoughtSignature":"sig-late"
            }]}}]}"#,
        );

        let cont = &second["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(cont["index"], 0);
        assert_eq!(cont["extra_content"]["google"]["thought_signature"], "sig-late");
        assert!(cont.get("function").is_none());
        assert!(cont.get("id").is_none());
    }

    // -------------------------------------------------------------------------
    // Upstream error and content block detection
    // -------------------------------------------------------------------------

    #[test]
    fn stream_vertex_error_frame_rejected() {
        let mut state = new_stream_state(1);
        let err = transform_stream_chunk(
            br#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#,
            "gemini-2.0-flash",
            &mut state,
        )
        .unwrap_err();
        assert!(err.contains("Quota exceeded"), "error message should propagate: {err}");
    }

    #[test]
    fn stream_vertex_error_frame_missing_message() {
        let mut state = new_stream_state(1);
        let err = transform_stream_chunk(
            br#"{"error":{"code":500,"status":"INTERNAL"}}"#,
            "gemini-2.0-flash",
            &mut state,
        )
        .unwrap_err();
        assert!(err.contains("upstream returned an error frame"), "{err}");
    }

    #[test]
    fn stream_prompt_blocked_frame_rejected() {
        let mut state = new_stream_state(1);
        let err = transform_stream_chunk(
            br#"{"promptFeedback":{"blockReason":"SAFETY","blockReasonMessage":"prompt violates policy"}}"#,
            "gemini-2.0-flash",
            &mut state,
        )
        .unwrap_err();
        assert!(
            err.contains("prompt violates policy"),
            "block message should propagate: {err}"
        );
    }

    #[test]
    fn stream_prompt_blocked_without_message() {
        let mut state = new_stream_state(1);
        let err = transform_stream_chunk(
            br#"{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}"#,
            "gemini-2.0-flash",
            &mut state,
        )
        .unwrap_err();
        assert!(err.contains("prompt blocked"), "{err}");
    }

    #[test]
    fn stream_include_usage_defers_usage_to_trailing_chunk() {
        let mut state = new_stream_state(1);
        state.set_include_usage(true);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}"#,
        );

        assert!(parsed["usage"].is_null(), "content chunks carry usage: null");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");

        let usage_bytes = take_stream_usage_chunk(&mut state, "gemini-1.5-pro").unwrap();
        let usage: Value = serde_json::from_slice(&usage_bytes).unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 3);
        assert_eq!(usage["usage"]["completion_tokens"], 1);
        assert_eq!(usage["usage"]["total_tokens"], 4);
        assert_eq!(usage["id"], parsed["id"]);
    }

    #[test]
    fn stream_preserves_all_candidates_and_their_indices() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"index":0,"content":{"parts":[{"text":"first"}]}},{"index":1,"content":{"parts":[{"text":"second"}]},"finishReason":"STOP"}]}"#,
        );

        assert_eq!(parsed["choices"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["choices"][0]["index"], 0);
        assert_eq!(parsed["choices"][0]["delta"]["content"], "first");
        assert_eq!(parsed["choices"][1]["index"], 1);
        assert_eq!(parsed["choices"][1]["delta"]["content"], "second");
        assert_eq!(parsed["choices"][1]["finish_reason"], "stop");
    }

    #[test]
    fn stream_tracks_first_delta_and_tool_slots_per_candidate() {
        let mut state = new_stream_state(1);
        let first = translate_stream(
            &mut state,
            br#"{"candidates":[{"index":0,"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"first"}}}]}}]}"#,
        );
        let second = translate_stream(
            &mut state,
            br#"{"candidates":[{"index":0,"content":{"parts":[{"text":"continued"}]}},{"index":1,"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"second"}}}]}}]}"#,
        );

        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert!(second["choices"][0]["delta"].get("role").is_none());
        assert_eq!(second["choices"][1]["delta"]["role"], "assistant");
        assert_eq!(first["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(second["choices"][1]["delta"]["tool_calls"][0]["index"], 0);
        assert_ne!(
            first["choices"][0]["delta"]["tool_calls"][0]["id"], second["choices"][1]["delta"]["tool_calls"][0]["id"],
            "each candidate must maintain independent tool-call identity"
        );
    }

    /// When the client requests `include_usage` but Vertex never sends
    /// `usageMetadata`, content chunks still carry `usage: null` and no
    /// trailing usage chunk is produced.  The `[DONE]` sentinel follows
    /// immediately.  This is intentional: the proxy cannot fabricate
    /// counts it did not receive.
    #[test]
    fn stream_include_usage_without_metadata_emits_no_trailing_chunk() {
        let mut state = new_stream_state(1);
        state.set_include_usage(true);
        // Frame has candidates but NO usageMetadata.
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"STOP"}]}"#,
        );

        assert!(parsed["usage"].is_null(), "content chunk still carries usage: null");
        assert!(
            take_stream_usage_chunk(&mut state, "gemini-2.0-flash").is_none(),
            "no usageMetadata received → no trailing usage chunk"
        );
    }

    #[test]
    fn stream_without_include_usage_omits_usage_field() {
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}"#,
        );

        assert!(
            parsed.get("usage").is_none(),
            "OpenAI omits usage unless include_usage is set"
        );
        assert!(take_stream_usage_chunk(&mut state, "gemini-1.5-pro").is_none());
    }

    #[test]
    fn stream_usage_only_frame_is_not_an_error() {
        let mut state = new_stream_state(1);
        let result = transform_stream_chunk(
            br#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
            "gemini-2.0-flash",
            &mut state,
        );
        assert!(
            matches!(result, Ok(None)),
            "usage-only frame should be skipped, not error: {result:?}"
        );
    }

    #[test]
    fn stream_usage_only_frame_feeds_trailing_chunk() {
        let mut state = new_stream_state(1);
        state.set_include_usage(true);
        assert!(
            transform_stream_chunk(
                br#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
                "gemini-2.0-flash",
                &mut state,
            )
            .unwrap()
            .is_none()
        );

        let usage: Value =
            serde_json::from_slice(&take_stream_usage_chunk(&mut state, "gemini-2.0-flash").unwrap()).unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["total_tokens"], 15);
    }

    #[test]
    fn stream_empty_json_is_not_an_error() {
        let mut state = new_stream_state(1);
        let result = transform_stream_chunk(b"{}", "gemini-2.0-flash", &mut state);
        assert!(
            matches!(result, Ok(None)),
            "empty JSON should be skipped, not error: {result:?}"
        );
    }

    #[test]
    fn stream_prompt_feedback_without_block_reason_is_not_an_error() {
        // promptFeedback can carry safetyRatings without blockReason.
        let mut state = new_stream_state(1);
        let result = transform_stream_chunk(
            br#"{"promptFeedback":{"safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"NEGLIGIBLE"}]}}"#,
            "gemini-2.0-flash",
            &mut state,
        );
        assert!(
            result.is_ok(),
            "promptFeedback without blockReason should pass: {result:?}"
        );
    }

    #[test]
    fn stream_candidates_with_safety_finish_reason_translates_normally() {
        // finishReason: SAFETY on an actual candidate is not an error frame;
        // it should translate to content_filter.
        let mut state = new_stream_state(1);
        let parsed = translate_stream(
            &mut state,
            br#"{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}]}"#,
        );
        assert_eq!(parsed["choices"][0]["finish_reason"], "content_filter");
    }
}
