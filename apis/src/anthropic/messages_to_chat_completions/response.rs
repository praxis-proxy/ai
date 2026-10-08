// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Chat Completions-compatible response to Anthropic Messages transformation.

use http::StatusCode;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::anthropic::wire::{self, ContentBlock, MessageResponse, MessageUsage};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default response type.
const RESPONSE_TYPE: &str = "message";

/// Default response role.
const RESPONSE_ROLE: &str = "assistant";

/// Minimal upstream error fields needed for Anthropic normalization.
#[derive(Deserialize)]
struct UpstreamError {
    /// Nested error details, when present.
    error: Option<Value>,
    /// Top-level error message, when present.
    message: Option<Value>,
    /// Upstream request identifier, when present.
    request_id: Option<Value>,
}

// -----------------------------------------------------------------------------
// Response Transformation
// -----------------------------------------------------------------------------

/// Result of a response transformation.
pub(crate) struct TransformResult {
    /// Transformed response body bytes.
    pub body: Vec<u8>,
    /// Original Chat Completions `finish_reason` (preserved for metadata).
    pub original_finish_reason: String,
}

/// Transform a Chat Completions-compatible response body into Anthropic
/// Messages format.
pub(crate) fn transform_response(
    body: &[u8],
    request_model: &str,
    stop_sequences: &[String],
) -> Result<TransformResult, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let Some(obj) = value.as_object() else {
        return Err("response body is not a JSON object".to_owned());
    };
    validate_translatable_response(obj)?;

    let id = match obj.get("id").and_then(Value::as_str) {
        Some(id) => format!("msg_{id}"),
        None => format!("msg_{}", timestamp_hex_id()),
    };

    let model = obj.get("model").and_then(Value::as_str).unwrap_or(request_model);

    let (stop_reason, original_finish_reason, stop_sequence) = map_finish_reason(obj, stop_sequences);
    let response = MessageResponse {
        content: build_content_blocks(obj)?,
        container: None,
        id,
        model,
        role: RESPONSE_ROLE,
        stop_details: None,
        stop_reason,
        stop_sequence,
        r#type: RESPONSE_TYPE,
        usage: build_usage(obj),
    };

    let body = serde_json::to_vec(&response).map_err(|e| format!("serialization failed: {e}"))?;
    Ok(TransformResult {
        body,
        original_finish_reason,
    })
}

/// Refuse a successful Chat response whose output would be silently lost.
#[expect(clippy::too_many_lines, reason = "sequential response shape validation")]
fn validate_translatable_response(obj: &Map<String, Value>) -> Result<(), String> {
    let choices = obj
        .get("choices")
        .and_then(Value::as_array)
        .ok_or("Chat response requires `choices` array")?;
    let [choice] = choices.as_slice() else {
        return Err("Chat response must contain exactly one choice".to_owned());
    };
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or("Chat choice requires a `message` object")?;
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err("Chat choice requires an assistant message".to_owned());
    }
    if !matches!(message.get("content"), None | Some(Value::Null | Value::String(_))) {
        return Err("Chat assistant `content` cannot be translated to Anthropic Messages".to_owned());
    }
    if message
        .get("tool_calls")
        .is_some_and(|value| !value.is_null() && !value.is_array())
    {
        return Err("Chat assistant `tool_calls` must be an array".to_owned());
    }
    if message.get("refusal").is_some_and(|value| !value.is_null())
        || message.get("audio").is_some_and(|value| !value.is_null())
        || message.get("function_call").is_some_and(|value| !value.is_null())
        || message
            .get("annotations")
            .is_some_and(|value| !value.is_null() && value.as_array().is_none_or(|annotations| !annotations.is_empty()))
        || choice.get("logprobs").is_some_and(|value| !value.is_null())
    {
        return Err("Chat response field cannot be translated to Anthropic Messages".to_owned());
    }
    match choice.get("finish_reason").and_then(Value::as_str) {
        Some("stop" | "length" | "tool_calls") => {},
        _ => return Err("Chat finish_reason cannot be translated to Anthropic Messages".to_owned()),
    }
    Ok(())
}

/// Transform an upstream 4xx or 5xx response into Anthropic error format.
pub(crate) fn transform_error_response(body: &[u8], status: StatusCode, header_request_id: Option<&str>) -> Vec<u8> {
    let parsed = serde_json::from_slice::<UpstreamError>(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|value| value.error.as_ref())
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| parsed.as_ref()?.message.as_ref()?.as_str())
        .unwrap_or("upstream request failed");
    let upstream_error_type = parsed
        .as_ref()
        .and_then(|value| value.error.as_ref())
        .and_then(|error| error.get("type"))
        .and_then(Value::as_str)
        .and_then(wire::ErrorType::parse);
    let request_id = parsed
        .as_ref()
        .and_then(|value| value.request_id.as_ref())
        .and_then(Value::as_str)
        .or(header_request_id);
    let error_type = upstream_error_type.unwrap_or_else(|| wire::ErrorType::from_status(status.as_u16()));

    wire::error_body(error_type, message, request_id)
}

// -----------------------------------------------------------------------------
// Content Block Building
// -----------------------------------------------------------------------------

/// Extract content blocks from the first choice.
fn build_content_blocks<'a>(obj: &'a Map<String, Value>) -> Result<Vec<ContentBlock<'a>>, String> {
    let mut blocks = Vec::new();

    let choice = obj.get("choices").and_then(Value::as_array).and_then(|c| c.first());

    let Some(choice) = choice else {
        return Ok(blocks);
    };

    let message = choice.get("message");
    extract_text_block(message, &mut blocks);
    extract_tool_call_blocks(message, &mut blocks)?;

    Ok(blocks)
}

/// Extract a text content block from the message if present.
fn extract_text_block<'a>(message: Option<&'a Value>, blocks: &mut Vec<ContentBlock<'a>>) {
    if let Some(content) = message.and_then(|m| m.get("content")).and_then(Value::as_str)
        && !content.is_empty()
    {
        blocks.push(ContentBlock::text(content));
    }
}

/// Extract tool call blocks from the message.
fn extract_tool_call_blocks<'a>(message: Option<&'a Value>, blocks: &mut Vec<ContentBlock<'a>>) -> Result<(), String> {
    let Some(Value::Array(tool_calls)) = message.and_then(|m| m.get("tool_calls")) else {
        return Ok(());
    };

    for tc in tool_calls {
        let id = tc
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "tool call missing required non-empty `id`".to_owned())?;
        if !wire::is_valid_tool_use_id(id) {
            return Err("tool call `id` must match ^[a-zA-Z0-9_-]+$".to_owned());
        }
        let name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "tool call missing required non-empty function `name`".to_owned())?;
        let args_str = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .ok_or_else(|| "tool call arguments must be a JSON-encoded object string".to_owned())?;
        let input = serde_json::from_str::<Map<String, Value>>(args_str)
            .map_err(|error| format!("invalid tool call arguments: {error}"))?;

        blocks.push(ContentBlock::tool_use(id, input, name));
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Finish Reason Mapping
// -----------------------------------------------------------------------------

/// Map Chat Completions `finish_reason` to Anthropic `stop_reason`.
///
/// Returns `(anthropic_stop_reason, original_finish_reason, matched_stop_sequence)`.
/// The `content_filter` to `end_turn` mapping is lossy; the
/// original is preserved so callers can store it in metadata.
///
/// `finish_reason: stop` covers both a natural stop and a stop sequence.
/// vLLM disambiguates through a choice-level `stop_reason` holding the
/// matched stop string; only a client-provided sequence is reported back,
/// since a server-side stop string or integer stop token id is not one.
fn map_finish_reason<'a>(obj: &'a Map<String, Value>, stop_sequences: &[String]) -> (String, String, Option<&'a str>) {
    let choice = obj.get("choices").and_then(Value::as_array).and_then(|c| c.first());
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let stop_sequence = choice
        .and_then(|c| c.get("stop_reason"))
        .and_then(Value::as_str)
        .filter(|matched| stop_sequences.iter().any(|sequence| sequence == matched));

    let mapped = match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "stop" if stop_sequence.is_some() => "stop_sequence",
        _ => "end_turn",
    };

    (mapped.to_owned(), finish_reason.to_owned(), stop_sequence)
}

// -----------------------------------------------------------------------------
// Usage Mapping
// -----------------------------------------------------------------------------

/// Build Anthropic usage object from Chat Completions usage.
///
/// Anthropic's `input_tokens` excludes cached tokens (they are reported
/// separately via `cache_read_input_tokens`), whereas OpenAI's
/// `prompt_tokens` includes them. The cached count must be subtracted
/// here so downstream Anthropic-format consumers that sum
/// `input_tokens + cache_read_input_tokens` don't double-count.
fn build_usage(obj: &Map<String, Value>) -> MessageUsage {
    let usage = obj.get("usage");

    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let cache_read = usage
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64);

    let input_tokens = match cache_read {
        Some(cached) => prompt_tokens.saturating_sub(cached),
        None => prompt_tokens,
    };

    MessageUsage::new(input_tokens, output_tokens, cache_read)
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Generate a timestamp-based hex identifier for response IDs.
fn timestamp_hex_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    format!("{nanos:024x}")
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use http::StatusCode;
    use serde_json::json;

    use super::*;

    #[test]
    fn semantic_chat_response_losses_are_rejected() {
        for (message, finish_reason) in [
            (
                json!({"role": "assistant", "content": [{"type": "text", "text": "hello"}]}),
                "stop",
            ),
            (json!({"role": "assistant", "content": null, "refusal": "no"}), "stop"),
            (json!({"role": "assistant", "content": "partial"}), "content_filter"),
        ] {
            let body = json!({"choices": [{"message": message, "finish_reason": finish_reason}]});
            let error = transform_response(body.to_string().as_bytes(), "m", &[]).err().unwrap();
            assert!(!error.is_empty());
        }
    }

    fn assert_null_fields(value: &Value, fields: &[&str]) {
        for field in fields {
            assert!(value.get(*field).is_some(), "expected {field} to be present");
            assert!(value[*field].is_null(), "expected {field} to be null");
        }
    }

    #[test]
    fn compatible_upstream_error_is_preserved() {
        let body = br#"{"error":{"type":"rate_limit_error","message":"slow down"},"request_id":"req_body"}"#;
        let output = transform_error_response(body, StatusCode::TOO_MANY_REQUESTS, Some("req_header"));
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert_eq!(parsed["error"]["message"], "slow down");
        assert_eq!(parsed["request_id"], "req_body");
    }

    #[test]
    fn future_anthropic_error_type_uses_status_fallback() {
        let body = br#"{"type":"error","error":{"type":"future_error","message":"new failure"}}"#;
        let output = transform_error_response(body, StatusCode::INTERNAL_SERVER_ERROR, None);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "new failure");
    }

    #[test]
    fn incompatible_error_type_uses_status_mapping_and_header_request_id() {
        let output = transform_error_response(
            br#"{"error":{"type":"server_error","message":"failed"}}"#,
            StatusCode::SERVICE_UNAVAILABLE,
            Some("req_header"),
        );
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "failed");
        assert_eq!(parsed["request_id"], "req_header");
    }

    #[test]
    fn top_level_error_message_is_preserved() {
        let output = transform_error_response(
            br#"{"message":"backend rejected the request"}"#,
            StatusCode::BAD_REQUEST,
            None,
        );
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "backend rejected the request");
        assert!(parsed["request_id"].is_null());
    }

    #[test]
    fn irrelevant_error_fields_are_ignored() {
        let body =
            br#"{"message":"backend rejected the request","irrelevant":[{"nested":"value"},{"nested":"value"}]}"#;
        let output = transform_error_response(body, StatusCode::BAD_REQUEST, None);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "backend rejected the request");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn malformed_optional_error_fields_do_not_discard_message() {
        let body = br#"{"error":{"type":"rate_limit_error","message":"slow down"},"request_id":123}"#;
        let output = transform_error_response(body, StatusCode::TOO_MANY_REQUESTS, None);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "slow down");
        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert!(parsed["request_id"].is_null());
    }

    #[test]
    fn unstructured_errors_do_not_reflect_unknown_text() {
        for body in [
            b"".as_slice(),
            b"[]".as_slice(),
            b"<html>secret backend diagnostic</html>".as_slice(),
        ] {
            let output = transform_error_response(body, StatusCode::BAD_GATEWAY, None);
            let parsed: Value = serde_json::from_slice(&output).unwrap();

            assert_eq!(parsed["error"]["type"], "api_error");
            assert_eq!(parsed["error"]["message"], "upstream request failed");
            assert!(parsed["request_id"].is_null());
        }
    }

    #[test]
    fn error_statuses_map_to_anthropic_types() {
        for (status, expected) in [
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::UNAUTHORIZED, "authentication_error"),
            (StatusCode::PAYMENT_REQUIRED, "billing_error"),
            (StatusCode::FORBIDDEN, "permission_error"),
            (StatusCode::NOT_FOUND, "not_found_error"),
            (StatusCode::CONFLICT, "invalid_request_error"),
            (StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
            (StatusCode::from_u16(529).unwrap(), "overloaded_error"),
            (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
        ] {
            let output = transform_error_response(b"", status, None);
            let parsed: Value = serde_json::from_slice(&output).unwrap();

            assert_eq!(parsed["error"]["type"], expected, "status {status}");
        }
    }

    #[test]
    fn basic_text_response() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let result = tr.body;
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["type"], "message", "type should be message");
        assert_eq!(parsed["role"], "assistant", "role should be assistant");
        assert_eq!(parsed["content"][0]["type"], "text", "content block type");
        assert_eq!(parsed["content"][0]["text"], "Hello!", "content text");
        assert!(
            parsed["content"][0].get("citations").is_some(),
            "text content should include citations"
        );
        assert!(parsed["content"][0]["citations"].is_null(), "citations should be null");
        assert_eq!(parsed["stop_reason"], "end_turn", "stop → end_turn");
        assert_null_fields(&parsed, &["container", "stop_details", "stop_sequence"]);
        assert_eq!(parsed["usage"]["input_tokens"], 10, "input tokens");
        assert_eq!(parsed["usage"]["output_tokens"], 5, "output tokens");
        assert_null_fields(
            &parsed["usage"],
            &[
                "cache_creation",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "inference_geo",
                "output_tokens_details",
                "server_tool_use",
                "service_tier",
            ],
        );
    }

    #[test]
    fn tool_calls_response() {
        let body = br#"{"id":"chatcmpl-2","model":"gpt-4","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":15}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let result = tr.body;
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop_reason"], "tool_use", "tool_calls → tool_use");
        assert_eq!(parsed["content"][0]["type"], "tool_use", "tool_use block");
        assert_eq!(parsed["content"][0]["name"], "get_weather", "tool name");
        assert_eq!(parsed["content"][0]["input"]["city"], "NYC", "parsed input");
        assert_eq!(
            parsed["content"][0]["caller"]["type"], "direct",
            "tool_use caller should identify a direct invocation"
        );
    }

    #[test]
    fn matched_stop_sequence_is_reported() {
        let body = br#"{"id":"chatcmpl-7","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Count: 1"},"finish_reason":"stop","stop_reason":","}],"usage":{"prompt_tokens":20,"completion_tokens":4}}"#;
        let tr = transform_response(body, "gpt-4", &[",".to_owned()]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert_eq!(parsed["stop_reason"], "stop_sequence", "matched stop → stop_sequence");
        assert_eq!(parsed["stop_sequence"], ",", "matched value is reported");
        assert_eq!(tr.original_finish_reason, "stop", "original finish reason preserved");
    }

    #[test]
    fn stop_reason_outside_client_sequences_stays_end_turn() {
        for stop_reason in [r#""</s>""#, "128009"] {
            let body = format!(
                r#"{{"id":"chatcmpl-8","model":"gpt-4","choices":[{{"message":{{"role":"assistant","content":"Hi"}},"finish_reason":"stop","stop_reason":{stop_reason}}}],"usage":{{"prompt_tokens":1,"completion_tokens":1}}}}"#
            );
            let tr = transform_response(body.as_bytes(), "gpt-4", &[",".to_owned()]).unwrap();
            let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

            assert_eq!(parsed["stop_reason"], "end_turn", "stop_reason {stop_reason}");
            assert!(parsed["stop_sequence"].is_null(), "stop_reason {stop_reason}");
        }
    }

    #[test]
    fn stop_without_backend_signal_stays_end_turn() {
        let body = br#"{"id":"chatcmpl-9","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        let tr = transform_response(body, "gpt-4", &[",".to_owned()]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert_eq!(parsed["stop_reason"], "end_turn", "no signal → end_turn");
        assert!(parsed["stop_sequence"].is_null(), "no signal → null stop_sequence");
    }

    #[test]
    fn length_finish_reason() {
        let body = br#"{"id":"chatcmpl-3","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"truncated..."},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":100}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let result = tr.body;
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop_reason"], "max_tokens", "length → max_tokens");
    }

    #[test]
    fn cached_tokens_in_usage() {
        let body = br#"{"id":"chatcmpl-4","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let result = tr.body;
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["usage"]["cache_read_input_tokens"], 80, "cached tokens mapped");
        assert_eq!(
            parsed["usage"]["input_tokens"], 20,
            "input_tokens should exclude cached tokens (100 prompt - 80 cached)"
        );
        assert_null_fields(
            &parsed["usage"],
            &[
                "cache_creation",
                "cache_creation_input_tokens",
                "inference_geo",
                "output_tokens_details",
                "server_tool_use",
                "service_tier",
            ],
        );
    }

    #[test]
    fn cached_tokens_not_double_counted_when_summed() {
        // OpenAI's prompt_tokens (100) includes the 80 cached tokens. Anthropic's
        // contract has input_tokens exclude cache, so a downstream consumer that
        // sums input_tokens + cache_read_input_tokens must recover the original
        // prompt_tokens total, not double-count the cached portion.
        let body = br#"{"id":"chatcmpl-5","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        let input_tokens = parsed["usage"]["input_tokens"].as_u64().unwrap();
        let cache_read = parsed["usage"]["cache_read_input_tokens"].as_u64().unwrap();
        assert_eq!(
            input_tokens + cache_read,
            100,
            "input_tokens + cache_read_input_tokens should equal original prompt_tokens"
        );
    }

    #[test]
    fn no_cached_tokens_leaves_input_tokens_unchanged() {
        let body = br#"{"id":"chatcmpl-6","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":5}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert_eq!(
            parsed["usage"]["input_tokens"], 42,
            "input_tokens should be unchanged when no cache info is present"
        );
        assert_null_fields(&parsed["usage"], &["cache_read_input_tokens"]);
    }

    #[test]
    fn transform_response_non_json_body() {
        let result = transform_response(b"not json at all", "gpt-4", &[]);
        let err = result.err().unwrap();
        assert!(err.contains("invalid JSON"), "error should mention invalid JSON: {err}");
    }

    #[test]
    fn transform_response_json_array_body() {
        let result = transform_response(b"[1,2,3]", "gpt-4", &[]);
        let err = result.err().unwrap();
        assert!(
            err.contains("not a JSON object"),
            "error should mention not a JSON object: {err}"
        );
    }

    #[test]
    fn missing_id_generates_msg_prefixed_id() {
        let body = br#"{"model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        let id = parsed["id"].as_str().unwrap();
        assert!(
            id.starts_with("msg_"),
            "generated ID should start with msg_ but got: {id}"
        );
    }

    #[test]
    fn empty_choices_are_rejected() {
        let body =
            br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();
        assert!(error.contains("exactly one choice"), "{error}");
    }

    #[test]
    fn empty_string_content_produces_no_text_block() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":0}}"#;
        let tr = transform_response(body, "gpt-4", &[]).unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert!(
            parsed["content"].as_array().unwrap().is_empty(),
            "empty content string should not produce a text block"
        );
    }

    #[test]
    fn invalid_tool_call_arguments_fail_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"not{json"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("invalid tool call arguments"),
            "malformed arguments should fail response transformation: {error}"
        );
    }

    #[test]
    fn non_object_tool_call_arguments_fail_transformation() {
        for arguments in ["[]", "null", "\"text\""] {
            let body = json!({
                "id": "chatcmpl-1",
                "model": "gpt-4",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": arguments
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            });
            let encoded = serde_json::to_vec(&body).unwrap();
            let error = transform_response(&encoded, "gpt-4", &[]).err().unwrap();

            assert!(
                error.contains("invalid tool call arguments"),
                "non-object arguments {arguments} should fail response transformation: {error}"
            );
        }
    }

    #[test]
    fn missing_tool_call_id_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("non-empty `id`"),
            "a missing tool call id should fail response transformation: {error}"
        );
    }

    #[test]
    fn empty_tool_call_id_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"","type":"function","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("non-empty `id`"),
            "an empty tool call id should fail response transformation: {error}"
        );
    }

    #[test]
    fn missing_tool_call_function_name_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("non-empty function `name`"),
            "a missing function name should fail response transformation: {error}"
        );
    }

    #[test]
    fn empty_tool_call_function_name_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("non-empty function `name`"),
            "an empty function name should fail response transformation: {error}"
        );
    }

    #[test]
    fn invalid_tool_call_id_format_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call.bad","type":"function","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("must match ^[a-zA-Z0-9_-]+$"),
            "a syntactically invalid id should fail response transformation: {error}"
        );
    }

    #[test]
    fn missing_tool_call_arguments_field_fails_transformation() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_time"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let error = transform_response(body, "gpt-4", &[]).err().unwrap();

        assert!(
            error.contains("tool call arguments must be a JSON-encoded object string"),
            "an absent arguments field should fail response transformation: {error}"
        );
    }
}
