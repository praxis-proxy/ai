// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic error response formatter for Praxis fatal proxy failures.
//!
//! Implements [`ErrorResponseFormatter`] to produce the schema-valid
//! Anthropic error envelope that Anthropic SDKs expect. Installed as
//! a request extension after positive Anthropic classification.
//!
//! # Note on response headers
//!
//! `FormattedErrorResponse` provides `body` and `content_type`. When Praxis
//! supports custom response headers on formatted error responses, the matching
//! `request-id` response header can be emitted alongside the JSON body.
//! Currently, the request ID is included in the JSON body.

use praxis_filter::{ErrorResponseContext, ErrorResponseFormatter, FormattedErrorResponse};

use super::wire;

/// Formats Praxis fatal proxy failures as Anthropic error JSON.
///
/// Produces `{"type":"error","error":{"type":"…","message":"…"},"request_id":"…"}`.
///
/// The request ID is captured at construction time — either from the
/// incoming `x-request-id` header or generated when the formatter is
/// installed.
pub(crate) struct AnthropicErrorFormatter {
    /// Captured request identifier for the JSON body.
    request_id: String,
}

impl AnthropicErrorFormatter {
    /// Create a formatter with a captured request identifier.
    pub(crate) fn new(request_id: String) -> Self {
        Self { request_id }
    }

    /// Create a formatter, deriving the request identifier from the request head.
    ///
    /// Shared by every install site so a failure carries the same identifier
    /// whether the formatter was installed from the classified request head or
    /// from body-format classification.
    pub(crate) fn from_request_headers(headers: &http::HeaderMap) -> Self {
        let request_id = headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map_or_else(generate_request_id, ToOwned::to_owned);
        Self::new(request_id)
    }
}

/// Generate a request identifier when the client did not send one.
///
/// Uses a timestamp-based hex string prefixed with `req_` to match
/// the Anthropic convention without adding a UUID dependency.
fn generate_request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("req_{nanos:032x}")
}

impl ErrorResponseFormatter for AnthropicErrorFormatter {
    fn format(&self, context: &ErrorResponseContext<'_>) -> FormattedErrorResponse {
        let error_type = wire::ErrorType::from_status(context.status);
        let body = wire::error_body(error_type, context.message, Some(&self.request_id));

        FormattedErrorResponse::new(body, http::HeaderValue::from_static("application/json"))
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn connection_refusal_produces_valid_anthropic_json() {
        let formatter = AnthropicErrorFormatter::new("req_test_001".to_owned());
        let ctx = ErrorResponseContext::new("upstream_connect_error", "Connection refused", 502);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "api_error");
        assert_eq!(parsed["error"]["message"], "Connection refused");
        assert_eq!(parsed["request_id"], "req_test_001");
    }

    #[test]
    fn timeout_produces_timeout_error_type() {
        let formatter = AnthropicErrorFormatter::new("req_timeout".to_owned());
        let ctx = ErrorResponseContext::new("upstream_connect_timeout", "Connection timed out", 504);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["type"], "timeout_error");
        assert_eq!(parsed["error"]["message"], "Connection timed out");
    }

    #[test]
    fn overloaded_status_produces_overloaded_error_type() {
        let formatter = AnthropicErrorFormatter::new("req_529".to_owned());
        let ctx = ErrorResponseContext::new("upstream_overloaded", "Service overloaded", 529);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["error"]["type"], "overloaded_error");
    }

    #[test]
    fn generic_fivex_produces_api_error_type() {
        for status in [500, 502, 503] {
            let formatter = AnthropicErrorFormatter::new("req_5xx".to_owned());
            let ctx = ErrorResponseContext::new("some_code", "some message", status);
            let response = formatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                parsed["error"]["type"], "api_error",
                "status {status} should map to api_error"
            );
        }
    }

    #[test]
    fn fourx_status_maps_to_anthropic_error_types() {
        let cases = [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (402, "billing_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (409, "invalid_request_error"),
            (413, "invalid_request_error"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (418, "invalid_request_error"),
        ];

        for (status, expected_type) in cases {
            let formatter = AnthropicErrorFormatter::new("req_4xx".to_owned());
            let ctx = ErrorResponseContext::new("test_code", "test message", status);
            let response = formatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                parsed["error"]["type"], expected_type,
                "status {status} should map to error type '{expected_type}'"
            );
        }
    }

    #[test]
    fn content_type_is_application_json() {
        let formatter = AnthropicErrorFormatter::new("req_ct".to_owned());
        let ctx = ErrorResponseContext::new("upstream_connect_error", "Connection refused", 502);
        let response = formatter.format(&ctx);

        assert_eq!(
            response.content_type,
            http::HeaderValue::from_static("application/json")
        );
    }

    #[test]
    fn top_level_type_is_always_error() {
        for status in [400, 401, 403, 404, 413, 429, 500, 502, 504, 529] {
            let formatter = AnthropicErrorFormatter::new("req_type".to_owned());
            let ctx = ErrorResponseContext::new("test", "test", status);
            let response = formatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(parsed["type"], "error", "top-level type must be 'error' for {status}");
        }
    }

    #[test]
    fn nested_error_type_is_always_in_allowed_vocabulary() {
        let allowed = [
            "invalid_request_error",
            "authentication_error",
            "billing_error",
            "permission_error",
            "not_found_error",
            "rate_limit_error",
            "timeout_error",
            "api_error",
            "overloaded_error",
        ];

        for status in [400, 401, 402, 403, 404, 409, 413, 422, 429, 500, 502, 503, 504, 529] {
            let formatter = AnthropicErrorFormatter::new("req_vocab".to_owned());
            let ctx = ErrorResponseContext::new("test_code", "test message", status);
            let response = formatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            let error_type = parsed["error"]["type"].as_str().unwrap();
            assert!(
                allowed.contains(&error_type),
                "error type '{error_type}' for status {status} must be in allowed vocabulary"
            );
        }
    }

    #[test]
    fn request_id_present_in_body() {
        let formatter = AnthropicErrorFormatter::new("req_abc123".to_owned());
        let ctx = ErrorResponseContext::new("upstream_error", "failed", 502);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["request_id"], "req_abc123");
    }

    #[test]
    fn json_escaping_handles_special_characters() {
        let formatter = AnthropicErrorFormatter::new("req_escape".to_owned());
        let ctx = ErrorResponseContext::new("server_error", "line1\nline2\"quoted\"\tand\\backslash", 500);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str().unwrap(),
            "line1\nline2\"quoted\"\tand\\backslash"
        );
    }

    #[test]
    fn json_escaping_handles_unicode() {
        let formatter = AnthropicErrorFormatter::new("req_unicode".to_owned());
        let ctx = ErrorResponseContext::new("server_error", "Connection to サーバー failed 🔥 (مرحبا)", 500);
        let response = formatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str().unwrap(),
            "Connection to サーバー failed 🔥 (مرحبا)"
        );

        assert!(std::str::from_utf8(&response.body).is_ok());
    }
}
