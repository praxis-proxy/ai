// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared Anthropic wire response types.

use bytes::Bytes;
use praxis_filter::Rejection;
use serde::Serialize;
use serde_json::Value;

/// Fallback error body used only if Serde serialization fails.
const ERROR_SERIALIZATION_FALLBACK: &[u8] = br#"{"type":"error","error":{"type":"api_error","message":"failed to serialize error response"},"request_id":null}"#;

/// Complete Anthropic Messages response.
#[derive(Serialize)]
pub(crate) struct MessageResponse<'a> {
    /// Message identifier.
    pub id: String,
    /// Response discriminator.
    pub r#type: &'static str,
    /// Message author role.
    pub role: &'static str,
    /// Model that generated the response.
    pub model: &'a str,
    /// Ordered response content blocks.
    pub content: Vec<Value>,
    /// Reason generation stopped.
    pub stop_reason: String,
    /// Matched stop sequence, when available.
    pub stop_sequence: Option<&'static str>,
    /// Extended stop information, when available.
    pub stop_details: Option<Value>,
    /// Container information, when available.
    pub container: Option<Value>,
    /// Token and service usage.
    pub usage: MessageUsage,
}

/// Complete Anthropic Messages usage object.
#[derive(Serialize)]
pub(crate) struct MessageUsage {
    /// Cache creation details, when available.
    pub cache_creation: Option<Value>,
    /// Tokens used to create cache entries.
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from cache.
    pub cache_read_input_tokens: Option<u64>,
    /// Inference geography, when reported.
    pub inference_geo: Option<String>,
    /// Non-cached input tokens.
    pub input_tokens: u64,
    /// Generated output tokens.
    pub output_tokens: u64,
    /// Extended output token details, when reported.
    pub output_tokens_details: Option<Value>,
    /// Server tool usage, when reported.
    pub server_tool_use: Option<Value>,
    /// Service tier, when reported.
    pub service_tier: Option<String>,
}

impl MessageUsage {
    /// Create usage from token values available in Chat Completions.
    pub(crate) fn new(input_tokens: u64, output_tokens: u64, cache_read_input_tokens: Option<u64>) -> Self {
        Self {
            cache_creation: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens,
            inference_geo: None,
            input_tokens,
            output_tokens,
            output_tokens_details: None,
            server_tool_use: None,
            service_tier: None,
        }
    }
}

/// Anthropic terminal streaming delta usage object.
#[derive(Serialize)]
pub(crate) struct MessageDeltaUsage {
    /// Tokens used to create cache entries.
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from cache.
    pub cache_read_input_tokens: Option<u64>,
    /// Input tokens, when reported in the terminal delta.
    pub input_tokens: Option<u64>,
    /// Cumulative generated output tokens.
    pub output_tokens: u64,
    /// Extended output token details, when reported.
    pub output_tokens_details: Option<Value>,
    /// Server tool usage, when reported.
    pub server_tool_use: Option<Value>,
}

impl MessageDeltaUsage {
    /// Create terminal delta usage from the cumulative output token count.
    pub(crate) fn new(output_tokens: u64) -> Self {
        Self {
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            input_tokens: None,
            output_tokens,
            output_tokens_details: None,
            server_tool_use: None,
        }
    }
}

/// Anthropic error response envelope.
#[derive(Serialize)]
struct ErrorResponse<'a> {
    /// Top-level response discriminator.
    r#type: &'static str,
    /// Structured error details.
    error: ErrorDetail<'a>,
    /// Anthropic request identifier, when supplied upstream.
    request_id: Option<&'a str>,
}

/// Anthropic structured error details.
#[derive(Serialize)]
struct ErrorDetail<'a> {
    /// Anthropic error category.
    r#type: &'a str,
    /// Human-readable diagnostic.
    message: &'a str,
}

/// Serialize a schema-complete Anthropic error response.
pub(crate) fn error_body(error_type: &str, message: &str, request_id: Option<&str>) -> Vec<u8> {
    serde_json::to_vec(&ErrorResponse {
        r#type: "error",
        error: ErrorDetail {
            r#type: error_type,
            message,
        },
        request_id,
    })
    .unwrap_or_else(|_| ERROR_SERIALIZATION_FALLBACK.to_vec())
}

/// Build a schema-complete Anthropic invalid-request rejection.
pub(crate) fn invalid_request_rejection(message: &str) -> Rejection {
    Rejection::status(400)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(error_body("invalid_request_error", message, None)))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn error_body_is_schema_complete_and_json_safe() {
        let body = error_body("invalid_request_error", "bad \"model\"\nvalue", None);
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "bad \"model\"\nvalue");
        assert!(parsed.get("request_id").is_some());
        assert!(parsed["request_id"].is_null());
    }

    #[test]
    fn error_body_preserves_request_id() {
        let body = error_body("rate_limit_error", "rate limited", Some("req_01"));
        let parsed: Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["request_id"], "req_01");
    }

    #[test]
    fn invalid_request_rejection_uses_error_envelope() {
        let rejection = invalid_request_rejection("bad request");
        let parsed: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();

        assert_eq!(rejection.status, 400);
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "bad request");
        assert!(parsed["request_id"].is_null());
    }
}
