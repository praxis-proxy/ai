// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Response translation: AWS Bedrock Converse → OpenAI Chat Completions.
//!
//! Handles two distinct paths:
//!
//! ## Non-streaming (`transform_response`)
//!
//! Translates the full `POST /model/{id}/converse` JSON body:
//!
//! ```text
//! Bedrock                            OpenAI Chat Completions
//! ──────────────────────────────     ────────────────────────────────────────
//! output.message.content[].text  →  choices[0].message.content
//! output.message.content[].toolUse → choices[0].message.tool_calls[]
//! stopReason                     →  choices[0].finish_reason
//! usage.inputTokens              →  usage.prompt_tokens
//! usage.outputTokens             →  usage.completion_tokens
//! usage.totalTokens              →  usage.total_tokens
//! ```
//!
//! ## Streaming (`transform_stream_event`)
//!
//! Translates individual [`EventStreamMessage`]s from
//! `POST /model/{id}/converse-stream`.  Each call returns the JSON bytes
//! for one SSE `data:` frame (without the `data: ` prefix or `\n\n`
//! terminator — the filter in `mod.rs` wraps those).
//!
//! The event sequence Bedrock emits:
//!
//! ```text
//! messageStart       → delta with role:"assistant"
//! contentBlockStart  → ignored (signals start of a block, no text yet)
//! contentBlockDelta  → delta with content text  OR  partial tool arguments
//! contentBlockStop   → ignored (block is complete)
//! messageStop        → delta with finish_reason
//! metadata           → usage chunk
//! ```

use serde::Serialize;
use serde_json::Value;
use tracing::debug;

use crate::bedrock::eventstream::EventStreamMessage;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Synthetic response ID used when Bedrock returns no identifier.
pub(crate) const FALLBACK_ID: &str = "chatcmpl-bedrock";

// -----------------------------------------------------------------------------
// stop-reason mapping
// -----------------------------------------------------------------------------

/// Map a Bedrock `stopReason` value to an OpenAI `finish_reason` string.
///
/// | Bedrock `stopReason`              | OpenAI `finish_reason` |
/// |-----------------------------------|------------------------|
/// | `end_turn`                        | `stop`                 |
/// | `tool_use`                        | `tool_calls`           |
/// | `max_tokens`                      | `length`               |
/// | `stop_sequence`                   | `stop`                 |
/// | `content_filtered`                | `content_filter`       |
/// | `guardrail_intervened`            | `content_filter`       |
/// | `model_context_window_exceeded`   | `length`               |
/// | anything else                     | `stop`                 |
fn map_stop_reason(reason: &str) -> &'static str {
    match reason {
        "tool_use" => "tool_calls",
        "max_tokens" | "model_context_window_exceeded" => "length",
        "content_filtered" | "guardrail_intervened" => "content_filter",
        _ => "stop",
    }
}

// -----------------------------------------------------------------------------
// Non-streaming response translation
// -----------------------------------------------------------------------------

/// Translate a complete Bedrock Converse JSON response body into an OpenAI
/// Chat Completions response.
///
/// Returns the JSON bytes of the translated response, or an error message.
#[expect(
    clippy::too_many_lines,
    reason = "linear translation of several Bedrock fields; extracting further helpers would obscure the mapping"
)]
pub(crate) fn transform_response(body: &[u8], model: &str) -> Result<Vec<u8>, String> {
    let root: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = root.as_object().ok_or("response body is not a JSON object")?;

    // --- Message content ---
    let message = obj
        .get("output")
        .and_then(|o| o.get("message"))
        .ok_or("missing required field `output.message`")?;

    let (content, tool_calls) = extract_message_content(message)?;

    // --- Finish reason ---
    let finish_reason = obj
        .get("stopReason")
        .and_then(Value::as_str)
        .map_or("stop", map_stop_reason);

    // --- Usage ---
    let usage = extract_usage(obj.get("usage"));

    // --- Build choices[0].message ---
    let mut message_obj = serde_json::json!({
        "role": "assistant",
        "content": content,
        "refusal": null
    });
    if !tool_calls.is_empty()
        && let Some(obj) = message_obj.as_object_mut()
    {
        obj.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }

    let response = serde_json::json!({
        "id": FALLBACK_ID,
        "object": "chat.completion",
        "created": created_timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message_obj,
            "finish_reason": finish_reason
        }],
        "usage": usage
    });

    serde_json::to_vec(&response).map_err(|e| format!("serialization failed: {e}"))
}

/// Extract the text `content` string and `tool_calls` array from a Bedrock
/// `output.message` object.
///
/// A single Bedrock message may contain multiple content blocks of mixed
/// types; text blocks are concatenated and toolUse blocks become
/// individual `tool_call` objects.
fn extract_message_content(message: &Value) -> Result<(Value, Vec<Value>), String> {
    let content_arr = message
        .get("content")
        .and_then(Value::as_array)
        .ok_or("missing required field `output.message.content`")?;

    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in content_arr {
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            text_parts.push(text);
        } else if let Some(tool_use) = block.get("toolUse") {
            tool_calls.push(translate_tool_use_block(tool_use)?);
        } else {
            return Err("unsupported Bedrock response content block".to_owned());
        }
    }

    let content = if text_parts.is_empty() {
        Value::Null
    } else {
        Value::String(text_parts.join(""))
    };

    Ok((content, tool_calls))
}

/// Convert a Bedrock `toolUse` content block to an OpenAI `tool_calls` entry.
fn translate_tool_use_block(tool_use: &Value) -> Result<Value, String> {
    let id = tool_use
        .get("toolUseId")
        .and_then(Value::as_str)
        .ok_or("Bedrock toolUse block is missing `toolUseId`")?;
    let name = tool_use
        .get("name")
        .and_then(Value::as_str)
        .ok_or("Bedrock toolUse block is missing `name`")?;
    let args = tool_use.get("input").map_or_else(
        || "{}".to_owned(),
        |v| serde_json::to_string(v).unwrap_or_else(|_| "{}".to_owned()),
    );

    Ok(serde_json::json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": args
        }
    }))
}

/// Build the OpenAI `usage` object from Bedrock `usage` fields.
fn extract_usage(usage: Option<&Value>) -> Value {
    let prompt = usage
        .and_then(|u| u.get("inputTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion = usage
        .and_then(|u| u.get("outputTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .and_then(|u| u.get("totalTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(prompt + completion);

    serde_json::json!({
        "prompt_tokens":     prompt,
        "completion_tokens": completion,
        "total_tokens":      total
    })
}

/// Current Unix timestamp in seconds (falls back to 0 on error).
fn created_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

// -----------------------------------------------------------------------------
// Streaming event translation
// -----------------------------------------------------------------------------

/// Per-event streaming state passed through `on_response_body` calls.
///
/// Stored in `HttpFilterContext` filter state so it persists across chunks.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    /// Index of the next tool call being streamed.
    pub tool_call_index: u32,
    /// Whether the first `messageStart` frame has been processed.
    pub role_emitted: bool,
    /// Accumulated finish reason from `messageStop`.
    pub finish_reason: Option<String>,
}

/// Translate a single decoded [`EventStreamMessage`] into OpenAI SSE
/// `data:` payload bytes.
///
/// Returns `None` when the event type produces no client-visible output
/// (e.g. `contentBlockStart`, `contentBlockStop`).
///
/// The caller (`mod.rs`) wraps the returned bytes with `data: ` + `\n\n`.
pub(crate) fn transform_stream_event(
    msg: &EventStreamMessage,
    model: &str,
    id: &str,
    state: &mut StreamState,
) -> Option<Vec<u8>> {
    if msg.is_exception() {
        return translate_exception_event(msg, model, id);
    }

    let event_type = msg.event_type()?;

    match event_type {
        "messageStart" => translate_message_start(model, id, state),
        "contentBlockStart" => translate_content_block_start(msg, model, id, state),
        "contentBlockDelta" => translate_content_block_delta(msg, model, id, state),
        "contentBlockStop" => None, // no client output for block-stop
        "messageStop" => translate_message_stop(msg, model, id, state),
        "metadata" => translate_metadata_event(msg, model, id),
        other => {
            debug!(event_type = other, "ignoring unknown Bedrock event type");
            None
        },
    }
}

// -----------------------------------------------------------------------------
// Per-event translators
// -----------------------------------------------------------------------------

/// `messageStart` → emit role delta on the first event.
fn translate_message_start(model: &str, id: &str, state: &mut StreamState) -> Option<Vec<u8>> {
    state.role_emitted = true;
    chunk_bytes(
        id,
        model,
        serde_json::json!({
            "role": "assistant",
            "content": ""
        }),
        None::<&str>,
        None,
    )
}

/// `contentBlockStart` → emit `tool_call` header if this is a `toolUse` block.
fn translate_content_block_start(
    msg: &EventStreamMessage,
    model: &str,
    id: &str,
    state: &mut StreamState,
) -> Option<Vec<u8>> {
    let payload: Value = serde_json::from_slice(&msg.payload).ok()?;
    let tool_use = payload.get("start")?.get("toolUse")?;

    let tool_use_id = tool_use.get("toolUseId").and_then(Value::as_str)?;
    let name = tool_use.get("name").and_then(Value::as_str)?;
    let index = state.tool_call_index;
    state.tool_call_index += 1;

    chunk_bytes(
        id,
        model,
        serde_json::json!({}),
        None::<&str>,
        Some(vec![serde_json::json!({
            "index":    index,
            "id":       tool_use_id,
            "type":     "function",
            "function": {"name": name, "arguments": ""}
        })]),
    )
}

/// `contentBlockDelta` → emit text delta or accumulated tool-call arguments.
fn translate_content_block_delta(
    msg: &EventStreamMessage,
    model: &str,
    id: &str,
    state: &StreamState,
) -> Option<Vec<u8>> {
    let payload: Value = serde_json::from_slice(&msg.payload).ok()?;
    let delta = payload.get("delta")?;

    if let Some(text) = delta.get("text").and_then(Value::as_str) {
        // Text delta.
        return chunk_bytes(id, model, serde_json::json!({"content": text}), None::<&str>, None);
    }

    if let Some(tool_input) = delta.get("toolUse") {
        // Partial tool-call arguments string.
        let partial_args = tool_input.get("input").and_then(Value::as_str).unwrap_or("");

        // The index of the in-progress tool call is one before the counter
        // (it was incremented in contentBlockStart).
        let index = state.tool_call_index.saturating_sub(1);

        return chunk_bytes(
            id,
            model,
            serde_json::json!({}),
            None::<&str>,
            Some(vec![serde_json::json!({
                "index": index,
                "function": {"arguments": partial_args}
            })]),
        );
    }

    None
}

/// `messageStop` → emit `finish_reason` chunk.
fn translate_message_stop(msg: &EventStreamMessage, model: &str, id: &str, state: &mut StreamState) -> Option<Vec<u8>> {
    let finish_reason = if let Ok(payload) = serde_json::from_slice::<Value>(&msg.payload) {
        payload
            .get("stopReason")
            .and_then(Value::as_str)
            .map_or("stop", map_stop_reason)
    } else {
        "stop"
    };

    state.finish_reason = Some(finish_reason.to_owned());

    chunk_bytes(id, model, serde_json::json!({}), Some(finish_reason), None)
}

/// `metadata` → emit usage chunk (may be sent as an additional SSE frame).
fn translate_metadata_event(msg: &EventStreamMessage, model: &str, id: &str) -> Option<Vec<u8>> {
    let payload: Value = serde_json::from_slice(&msg.payload).ok()?;
    let usage = extract_usage(payload.get("usage"));

    let chunk = serde_json::json!({
        "id":     id,
        "object": "chat.completion.chunk",
        "model":  model,
        "choices": [],
        "usage":  usage
    });

    serde_json::to_vec(&chunk).ok()
}

/// Exception frame → emit an error-shaped SSE chunk so clients see the error.
fn translate_exception_event(msg: &EventStreamMessage, model: &str, id: &str) -> Option<Vec<u8>> {
    let error_message = serde_json::from_slice::<Value>(&msg.payload)
        .ok()
        .and_then(|v| v.get("message").and_then(Value::as_str).map(ToOwned::to_owned))
        .unwrap_or_else(|| "upstream exception".to_owned());

    let exception_type = msg.header_str(":exception-type").unwrap_or("unknownException");

    debug!(exception_type, error_message, "Bedrock stream exception");

    // Emit an error SSE frame in OpenAI error envelope so SDK clients
    // surface a structured error rather than a silent stream end.
    let chunk = serde_json::json!({
        "id":     id,
        "object": "chat.completion.chunk",
        "model":  model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "error": {
            "message": error_message,
            "type":    "server_error",
            "code":    exception_type
        }
    });

    serde_json::to_vec(&chunk).ok()
}

// -----------------------------------------------------------------------------
// Chunk builder helper
// -----------------------------------------------------------------------------

/// Build a `chat.completion.chunk` JSON object and serialize it.
///
/// * `delta` — the `choices[0].delta` object
/// * `finish_reason` — `Some(reason)` for the final chunk, `None` otherwise
/// * `tool_calls` — optional list of tool-call delta objects
fn chunk_bytes(
    id: &str,
    model: &str,
    delta: Value,
    finish_reason: Option<impl Serialize>,
    tool_calls: Option<Vec<Value>>,
) -> Option<Vec<u8>> {
    let finish_reason_value = finish_reason.map_or(Value::Null, |r| serde_json::to_value(r).unwrap_or(Value::Null));

    let mut delta_obj = delta;
    if let Some(tcs) = tool_calls
        && let Some(obj) = delta_obj.as_object_mut()
    {
        obj.insert("tool_calls".to_owned(), Value::Array(tcs));
    }

    let chunk = serde_json::json!({
        "id":     id,
        "object": "chat.completion.chunk",
        "model":  model,
        "choices": [{
            "index":         0,
            "delta":         delta_obj,
            "finish_reason": finish_reason_value
        }]
    });

    serde_json::to_vec(&chunk).ok()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;
    use crate::bedrock::eventstream::build_frame;

    const MODEL: &str = "anthropic.claude-3-sonnet-20240229-v1:0";

    fn decoded_msg(event_type: &str, payload: &[u8]) -> EventStreamMessage {
        use crate::bedrock::eventstream::EventStreamDecoder;
        let raw = build_frame(&[(":message-type", "event"), (":event-type", event_type)], payload);
        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        dec.decode().unwrap().unwrap()
    }

    fn decoded_exception(exception_type: &str, payload: &[u8]) -> EventStreamMessage {
        use crate::bedrock::eventstream::EventStreamDecoder;
        let raw = build_frame(
            &[(":message-type", "exception"), (":exception-type", exception_type)],
            payload,
        );
        let mut dec = EventStreamDecoder::new();
        dec.push(&raw);
        dec.decode().unwrap().unwrap()
    }

    // ── Non-streaming response ─────────────────────────────────────────────

    #[test]
    fn transform_response_basic_text() {
        let bedrock = serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Paris is the capital of France."}]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 20, "outputTokens": 8, "totalTokens": 28}
        });

        let body = serde_json::to_vec(&bedrock).unwrap();
        let result: Value = serde_json::from_slice(&transform_response(&body, MODEL).unwrap()).unwrap();

        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["model"], MODEL);
        assert_eq!(result["choices"][0]["message"]["role"], "assistant");
        assert_eq!(
            result["choices"][0]["message"]["content"],
            "Paris is the capital of France."
        );
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        assert_eq!(result["usage"]["prompt_tokens"], 20);
        assert_eq!(result["usage"]["completion_tokens"], 8);
        assert_eq!(result["usage"]["total_tokens"], 28);
    }

    #[test]
    fn transform_response_tool_call() {
        let bedrock = serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{
                        "toolUse": {
                            "toolUseId": "call_abc",
                            "name": "get_weather",
                            "input": {"city": "Paris"}
                        }
                    }]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
        });

        let body = serde_json::to_vec(&bedrock).unwrap();
        let result: Value = serde_json::from_slice(&transform_response(&body, MODEL).unwrap()).unwrap();

        assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
        let tc = &result["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_abc");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "get_weather");
        // arguments should be a JSON string
        let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn transform_response_text_and_tool_call() {
        let bedrock = serde_json::json!({
            "output": {
                "message": {
                    "content": [
                        {"text": "Let me check."},
                        {"toolUse": {"toolUseId": "tc1", "name": "lookup", "input": {}}}
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 5, "outputTokens": 5, "totalTokens": 10}
        });

        let body = serde_json::to_vec(&bedrock).unwrap();
        let result: Value = serde_json::from_slice(&transform_response(&body, MODEL).unwrap()).unwrap();

        assert_eq!(result["choices"][0]["message"]["content"], "Let me check.");
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn transform_response_stop_reasons_mapped() {
        for (bedrock_reason, expected) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("content_filtered", "content_filter"),
            ("guardrail_intervened", "content_filter"),
            ("model_context_window_exceeded", "length"),
            ("unknown_future_reason", "stop"),
        ] {
            let bedrock = serde_json::json!({
                "output": {"message": {"content": [{"text": "hi"}]}},
                "stopReason": bedrock_reason,
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
            });
            let body = serde_json::to_vec(&bedrock).unwrap();
            let result: Value = serde_json::from_slice(&transform_response(&body, MODEL).unwrap()).unwrap();
            assert_eq!(
                result["choices"][0]["finish_reason"], expected,
                "stopReason={bedrock_reason}"
            );
        }
    }

    #[test]
    fn transform_response_missing_output_is_rejected() {
        let bedrock = serde_json::json!({
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        });
        let body = serde_json::to_vec(&bedrock).unwrap();
        let error = transform_response(&body, MODEL).unwrap_err();
        assert!(error.contains("output.message"));
    }

    #[test]
    fn transform_response_invalid_json_errors() {
        assert!(transform_response(b"not json", MODEL).is_err());
    }

    // ── Streaming: messageStart ────────────────────────────────────────────

    #[test]
    fn stream_message_start_emits_role_delta() {
        let msg = decoded_msg("messageStart", br#"{"role":"assistant"}"#);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert!(state.role_emitted);
    }

    // ── Streaming: contentBlockDelta (text) ───────────────────────────────

    #[test]
    fn stream_text_delta_emits_content() {
        let payload = br#"{"contentBlockIndex":0,"delta":{"text":"Hello!"}}"#;
        let msg = decoded_msg("contentBlockDelta", payload);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["choices"][0]["delta"]["content"], "Hello!");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }

    // ── Streaming: contentBlockStart (toolUse) ────────────────────────────

    #[test]
    fn stream_tool_block_start_emits_tool_call_header() {
        let payload = br#"{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"tc1","name":"lookup"}}}"#;
        let msg = decoded_msg("contentBlockStart", payload);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        let tc = &chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "tc1");
        assert_eq!(tc["function"]["name"], "lookup");
        assert_eq!(state.tool_call_index, 1);
    }

    // ── Streaming: contentBlockDelta (toolUse arguments) ──────────────────

    #[test]
    fn stream_tool_delta_emits_partial_arguments() {
        let payload = br#"{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"city\":"}}}"#;
        let msg = decoded_msg("contentBlockDelta", payload);
        let mut state = StreamState {
            tool_call_index: 1,
            ..Default::default()
        };
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        let tc = &chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0); // saturating_sub(1)
        assert_eq!(tc["function"]["arguments"], "{\"city\":");
    }

    // ── Streaming: contentBlockStop ───────────────────────────────────────

    #[test]
    fn stream_content_block_stop_produces_no_output() {
        let msg = decoded_msg("contentBlockStop", br#"{"contentBlockIndex":0}"#);
        let mut state = StreamState::default();
        let result = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state);
        assert!(result.is_none());
    }

    // ── Streaming: messageStop ────────────────────────────────────────────

    #[test]
    fn stream_message_stop_emits_finish_reason() {
        let payload = br#"{"stopReason":"end_turn","additionalModelResponseFields":null}"#;
        let msg = decoded_msg("messageStop", payload);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(state.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn stream_message_stop_tool_use_finish_reason() {
        let msg = decoded_msg("messageStop", br#"{"stopReason":"tool_use"}"#);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["choices"][0]["finish_reason"], "tool_calls");
    }

    // ── Streaming: metadata ───────────────────────────────────────────────

    #[test]
    fn stream_metadata_emits_usage_chunk() {
        let payload = br#"{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15},"metrics":{"latencyMs":200}}"#;
        let msg = decoded_msg("metadata", payload);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["usage"]["prompt_tokens"], 10);
        assert_eq!(chunk["usage"]["completion_tokens"], 5);
        assert_eq!(chunk["usage"]["total_tokens"], 15);
        assert!(chunk["choices"].as_array().unwrap().is_empty());
    }

    // ── Streaming: exception frame ────────────────────────────────────────

    #[test]
    fn stream_exception_emits_error_chunk() {
        let msg = decoded_exception("throttlingException", br#"{"message":"Too many requests"}"#);
        let mut state = StreamState::default();
        let bytes = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state).unwrap();
        let chunk: Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(chunk["error"]["message"], "Too many requests");
        assert_eq!(chunk["error"]["code"], "throttlingException");
    }

    // ── Streaming: unknown event type ─────────────────────────────────────

    #[test]
    fn stream_unknown_event_produces_no_output() {
        let msg = decoded_msg("someFutureEvent", b"{}");
        let mut state = StreamState::default();
        let result = transform_stream_event(&msg, MODEL, FALLBACK_ID, &mut state);
        assert!(result.is_none());
    }

    // ── map_stop_reason exhaustive ────────────────────────────────────────

    #[test]
    fn stop_reason_mapping_complete() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("model_context_window_exceeded"), "length");
        assert_eq!(map_stop_reason("content_filtered"), "content_filter");
        assert_eq!(map_stop_reason("guardrail_intervened"), "content_filter");
        assert_eq!(map_stop_reason("malformed_tool_use"), "stop");
        assert_eq!(map_stop_reason(""), "stop");
    }
}
