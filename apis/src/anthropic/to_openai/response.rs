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

/// Anthropic error types that may be preserved from an upstream response.
const ANTHROPIC_ERROR_TYPES: &[&str] = &[
    "invalid_request_error",
    "authentication_error",
    "billing_error",
    "permission_error",
    "not_found_error",
    "conflict_error",
    "request_too_large",
    "rate_limit_error",
    "timeout_error",
    "api_error",
    "overloaded_error",
];

/// Minimal upstream error fields needed for Anthropic normalization.
#[derive(Deserialize)]
struct UpstreamError {
    /// Nested error details, when present.
    error: Option<Value>,
    /// Top-level error message, when present.
    message: Option<Value>,
    /// Upstream request identifier, when present.
    request_id: Option<Value>,
    /// Top-level response discriminator, when present.
    #[serde(rename = "type")]
    r#type: Option<Value>,
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
pub(crate) fn transform_response(body: &[u8], request_model: &str) -> Result<TransformResult, String> {
    let value: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;

    let Some(obj) = value.as_object() else {
        return Err("response body is not a JSON object".to_owned());
    };

    let id = match obj.get("id").and_then(Value::as_str) {
        Some(id) => format!("msg_{id}"),
        None => format!("msg_{}", timestamp_hex_id()),
    };

    let model = obj.get("model").and_then(Value::as_str).unwrap_or(request_model);

    let (stop_reason, original_finish_reason) = map_finish_reason(obj);
    let response = MessageResponse {
        content: build_content_blocks(obj),
        container: None,
        id,
        model,
        role: RESPONSE_ROLE,
        stop_details: None,
        stop_reason,
        stop_sequence: None,
        r#type: RESPONSE_TYPE,
        usage: build_usage(obj),
    };

    let body = serde_json::to_vec(&response).map_err(|e| format!("serialization failed: {e}"))?;
    Ok(TransformResult {
        body,
        original_finish_reason,
    })
}

/// Transform an upstream 4xx or 5xx response into Anthropic error format.
pub(crate) fn transform_error_response(body: &[u8], status: StatusCode, header_request_id: Option<&str>) -> Vec<u8> {
    let parsed = serde_json::from_slice::<UpstreamError>(body).ok();
    let is_anthropic_error = parsed
        .as_ref()
        .and_then(|value| value.r#type.as_ref())
        .and_then(Value::as_str)
        .is_some_and(|value| value == "error");
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
        .filter(|error_type| is_anthropic_error || ANTHROPIC_ERROR_TYPES.contains(error_type));
    let request_id = parsed
        .as_ref()
        .and_then(|value| value.request_id.as_ref())
        .and_then(Value::as_str)
        .or(header_request_id);
    let error_type = upstream_error_type.unwrap_or_else(|| error_type_for_status(status));

    wire::error_body(error_type, message, request_id)
}

/// Map an HTTP error status to its Anthropic error type.
fn error_type_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "authentication_error",
        402 => "billing_error",
        403 => "permission_error",
        404 => "not_found_error",
        409 => "conflict_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        504 => "timeout_error",
        529 => "overloaded_error",
        500..=599 => "api_error",
        _ => "invalid_request_error",
    }
}

// -----------------------------------------------------------------------------
// Content Block Building
// -----------------------------------------------------------------------------

/// Extract content blocks from the first choice.
fn build_content_blocks<'a>(obj: &'a Map<String, Value>) -> Vec<ContentBlock<'a>> {
    let mut blocks = Vec::new();

    let choice = obj.get("choices").and_then(Value::as_array).and_then(|c| c.first());

    let Some(choice) = choice else {
        return blocks;
    };

    let message = choice.get("message");
    extract_text_block(message, &mut blocks);
    extract_tool_call_blocks(message, &mut blocks);

    blocks
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
fn extract_tool_call_blocks<'a>(message: Option<&'a Value>, blocks: &mut Vec<ContentBlock<'a>>) {
    let Some(Value::Array(tool_calls)) = message.and_then(|m| m.get("tool_calls")) else {
        return;
    };

    for tc in tool_calls {
        let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
        let name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let args_str = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let input = serde_json::from_str::<Map<String, Value>>(args_str).unwrap_or_default();

        blocks.push(ContentBlock::tool_use(id, input, name));
    }
}

// -----------------------------------------------------------------------------
// Finish Reason Mapping
// -----------------------------------------------------------------------------

/// Map Chat Completions `finish_reason` to Anthropic `stop_reason`.
///
/// Returns `(anthropic_stop_reason, original_finish_reason)`.
/// The `content_filter` to `end_turn` mapping is lossy; the
/// original is preserved so callers can store it in metadata.
fn map_finish_reason(obj: &Map<String, Value>) -> (String, String) {
    let finish_reason = obj
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("stop");

    let mapped = match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    };

    (mapped.to_owned(), finish_reason.to_owned())
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

    fn assert_absent_fields(value: &Value, fields: &[&str]) {
        for field in fields {
            assert!(value.get(*field).is_none(), "expected {field} to be absent");
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
    fn future_anthropic_error_type_is_preserved() {
        let body = br#"{"type":"error","error":{"type":"future_error","message":"new failure"}}"#;
        let output = transform_error_response(body, StatusCode::INTERNAL_SERVER_ERROR, None);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "future_error");
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
            (StatusCode::CONFLICT, "conflict_error"),
            (StatusCode::PAYLOAD_TOO_LARGE, "request_too_large"),
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
        let tr = transform_response(body, "gpt-4").unwrap();
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
        assert_absent_fields(&parsed["usage"], &["output_tokens_details"]);
        assert_null_fields(
            &parsed["usage"],
            &[
                "cache_creation",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "inference_geo",
                "server_tool_use",
                "service_tier",
            ],
        );
    }

    #[test]
    fn tool_calls_response() {
        let body = br#"{"id":"chatcmpl-2","model":"gpt-4","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":15}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
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
    fn length_finish_reason() {
        let body = br#"{"id":"chatcmpl-3","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"truncated..."},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":100}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
        let result = tr.body;
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["stop_reason"], "max_tokens", "length → max_tokens");
    }

    #[test]
    fn cached_tokens_in_usage() {
        let body = br#"{"id":"chatcmpl-4","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
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
                "server_tool_use",
                "service_tier",
            ],
        );
        assert_absent_fields(&parsed["usage"], &["output_tokens_details"]);
    }

    #[test]
    fn cached_tokens_not_double_counted_when_summed() {
        // OpenAI's prompt_tokens (100) includes the 80 cached tokens. Anthropic's
        // contract has input_tokens exclude cache, so a downstream consumer that
        // sums input_tokens + cache_read_input_tokens must recover the original
        // prompt_tokens total, not double-count the cached portion.
        let body = br#"{"id":"chatcmpl-5","model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":80}}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
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
        let tr = transform_response(body, "gpt-4").unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert_eq!(
            parsed["usage"]["input_tokens"], 42,
            "input_tokens should be unchanged when no cache info is present"
        );
        assert_null_fields(&parsed["usage"], &["cache_read_input_tokens"]);
    }

    #[test]
    fn transform_response_non_json_body() {
        let result = transform_response(b"not json at all", "gpt-4");
        let err = result.err().unwrap();
        assert!(err.contains("invalid JSON"), "error should mention invalid JSON: {err}");
    }

    #[test]
    fn transform_response_json_array_body() {
        let result = transform_response(b"[1,2,3]", "gpt-4");
        let err = result.err().unwrap();
        assert!(
            err.contains("not a JSON object"),
            "error should mention not a JSON object: {err}"
        );
    }

    #[test]
    fn missing_id_generates_msg_prefixed_id() {
        let body = br#"{"model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        let id = parsed["id"].as_str().unwrap();
        assert!(
            id.starts_with("msg_"),
            "generated ID should start with msg_ but got: {id}"
        );
    }

    #[test]
    fn empty_choices_produces_empty_content() {
        let body =
            br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert!(
            parsed["content"].as_array().unwrap().is_empty(),
            "empty choices should produce empty content"
        );
    }

    #[test]
    fn empty_string_content_produces_no_text_block() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":0}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert!(
            parsed["content"].as_array().unwrap().is_empty(),
            "empty content string should not produce a text block"
        );
    }

    #[test]
    fn invalid_tool_call_arguments_fallback_to_empty_object() {
        let body = br#"{"id":"chatcmpl-1","model":"gpt-4","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"not{json"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let tr = transform_response(body, "gpt-4").unwrap();
        let parsed: Value = serde_json::from_slice(&tr.body).unwrap();

        assert_eq!(parsed["content"][0]["type"], "tool_use");
        assert_eq!(
            parsed["content"][0]["input"],
            json!({}),
            "invalid JSON arguments should fallback to empty object"
        );
    }

    #[test]
    fn non_object_tool_call_arguments_fallback_to_empty_object() {
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
            let transformed = transform_response(&encoded, "gpt-4").unwrap();
            let parsed: Value = serde_json::from_slice(&transformed.body).unwrap();

            assert_eq!(
                parsed["content"][0]["input"],
                json!({}),
                "{arguments} should not produce a non-object tool input"
            );
        }
    }
}
