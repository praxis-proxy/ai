// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI Gemini response to OpenAI Chat Completions translation.
//!
//! Converts `generateContent` responses: `candidates` → `choices`,
//! `usageMetadata` → `usage`, `finishReason` mapping, and
//! `functionCall` parts → `tool_calls`.

use serde_json::{Map, Value, json};

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

    let choices = obj
        .and_then(|o| o.get("candidates"))
        .and_then(Value::as_array)
        .map(|candidates| convert_candidates(candidates))
        .unwrap_or_default();

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

    json!({
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
    })
}

// -----------------------------------------------------------------------------
// Parts → content + tool_calls
// -----------------------------------------------------------------------------

/// Extract text content and tool calls from Gemini response parts.
///
/// - `text` parts → concatenated into a single string (or `null` if none)
/// - `functionCall` parts → converted to OpenAI `tool_calls` entries
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
            tool_calls.push(convert_function_call_to_tool_call(fc, tool_calls.len()));
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
fn convert_function_call_to_tool_call(fc: &Map<String, Value>, index: usize) -> Value {
    let name = fc.get("name").and_then(Value::as_str).unwrap_or("");

    let args = fc.get("args").map_or_else(
        || "{}".to_owned(),
        |a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_owned()),
    );

    json!({
        "id": format!("call_vertex_{index}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        }
    })
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
/// Gemini `FinishReason` enum (full list):
/// `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `LANGUAGE`, `OTHER`,
/// `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `MALFORMED_FUNCTION_CALL`,
/// `IMAGE_SAFETY`, `FINISH_REASON_UNSPECIFIED`.
fn convert_finish_reason(reason: Option<&str>, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_calls";
    }

    match reason {
        Some("MAX_TOKENS") => "length",
        Some(
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" | "LANGUAGE"
            | "OTHER",
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
/// `is_first` controls whether `delta` includes `role: "assistant"`
/// (only the first chunk should).
pub(crate) fn transform_stream_chunk(
    data: &[u8],
    model: &str,
    id: &str,
    is_first: bool,
    created: u64,
) -> Result<Vec<u8>, String> {
    let value: Value = serde_json::from_slice(data).map_err(|e| format!("invalid JSON: {e}"))?;
    let obj = value.as_object();

    let candidate = obj
        .and_then(|o| o.get("candidates"))
        .and_then(Value::as_array)
        .and_then(|c| c.first());

    let parts = candidate
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(Value::as_array);

    let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(Value::as_str);
    let (content, tool_calls) = extract_content_and_tool_calls(parts);
    let has_tool_calls = !tool_calls.is_empty();
    let delta = build_stream_delta(is_first, content, tool_calls);
    let finish = finish_reason.map_or(Value::Null, |_| {
        Value::String(convert_finish_reason(finish_reason, has_tool_calls).to_owned())
    });

    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });

    serde_json::to_vec(&chunk).map_err(|e| format!("serialization failed: {e}"))
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
        assert!(tc["id"].as_str().unwrap().starts_with("call_vertex_"));

        let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "NYC");
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

    // -------------------------------------------------------------------------
    // Streaming chunks
    // -------------------------------------------------------------------------

    #[test]
    fn stream_chunk_first_includes_role() {
        let data = br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#;
        let output = transform_stream_chunk(data, "gemini-1.5-pro", "id-1", true, 1_700_000_000).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["created"], 1_700_000_000);
        assert_eq!(parsed["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(parsed["choices"][0]["delta"]["content"], "Hi");
        assert!(parsed["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn stream_chunk_subsequent_omits_role() {
        let data = br#"{"candidates":[{"content":{"parts":[{"text":" world"}]}}]}"#;
        let output = transform_stream_chunk(data, "gemini-1.5-pro", "id-1", false, 1_700_000_000).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert!(parsed["choices"][0]["delta"].get("role").is_none());
        assert_eq!(parsed["choices"][0]["delta"]["content"], " world");
    }

    #[test]
    fn stream_chunk_finish_reason() {
        let data = br#"{"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}"#;
        let output = transform_stream_chunk(data, "gemini-1.5-pro", "id-1", false, 1_700_000_000).unwrap();
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }
}
