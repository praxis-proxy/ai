// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! OpenAI error response formatter for Praxis fatal proxy failures.
//!
//! Implements [`ErrorResponseFormatter`] to produce the standard
//! `{"error": {...}}` envelope that OpenAI SDKs expect. Installed
//! as a request extension after positive OpenAI classification so
//! that Praxis calls it from `fail_to_proxy` instead of emitting
//! RFC 9457 Problem Details.

use bytes::Bytes;
use http::HeaderValue;
use praxis_filter::{ErrorResponseContext, ErrorResponseFormatter, FormattedErrorResponse};

/// Formats Praxis fatal proxy failures as OpenAI error JSON.
///
/// Produces `{"error":{"message":"…","type":"…","param":null,"code":"…"}}`.
///
/// Mapping:
/// - `error.message` ← `context.message`
/// - `error.code` ← `context.code` (Praxis machine-readable code)
/// - `error.type` ← standardized OpenAI error type based on HTTP status code
/// - `error.param` ← always `null`
pub(crate) struct OpenAiErrorFormatter;

/// Maps an HTTP status code to a standardized OpenAI error type string.
fn map_error_type(status: u16) -> &'static str {
    match status {
        500..=599 => "server_error",
        429 => "rate_limit_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        400 | 422 => "invalid_request_error",
        _ => "api_error",
    }
}

impl ErrorResponseFormatter for OpenAiErrorFormatter {
    fn format(&self, context: &ErrorResponseContext<'_>) -> FormattedErrorResponse {
        let error_type = map_error_type(context.status);

        let body = serde_json::json!({
            "error": {
                "message": context.message,
                "type": error_type,
                "param": null,
                "code": context.code,
            },
        });

        FormattedErrorResponse::new(
            Bytes::from(body.to_string()),
            HeaderValue::from_static("application/json"),
        )
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
    fn connection_refusal_produces_valid_openai_json() {
        let ctx = ErrorResponseContext::new("upstream_connect_refused", "Connection refused", 502);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["error"]["message"], "Connection refused");
        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["code"], "upstream_connect_refused");
        assert!(parsed["error"]["param"].is_null());
    }

    #[test]
    fn timeout_produces_valid_openai_json() {
        let ctx = ErrorResponseContext::new("upstream_connect_timeout", "Connection timed out", 504);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["error"]["message"], "Connection timed out");
        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["code"], "upstream_connect_timeout");
        assert!(parsed["error"]["param"].is_null());
    }

    #[test]
    fn content_type_is_application_json() {
        let ctx = ErrorResponseContext::new("upstream_connect_error", "Connection refused", 502);
        let response = OpenAiErrorFormatter.format(&ctx);

        assert_eq!(response.content_type, HeaderValue::from_static("application/json"));
    }

    #[test]
    fn json_escaping_handles_special_characters() {
        let ctx = ErrorResponseContext::new("server_error", "line1\nline2\"quoted\"\tand\\backslash", 500);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str().unwrap(),
            "line1\nline2\"quoted\"\tand\\backslash"
        );
    }

    #[test]
    fn json_escaping_handles_unicode() {
        let ctx = ErrorResponseContext::new("server_error", "Connection to サーバー failed 🔥 (مرحبا)", 500);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str().unwrap(),
            "Connection to サーバー failed 🔥 (مرحبا)"
        );

        assert!(std::str::from_utf8(&response.body).is_ok());
    }

    #[test]
    fn fivex_status_uses_server_error_type() {
        for status in [500, 502, 503, 504] {
            let ctx = ErrorResponseContext::new("some_code", "some message", status);
            let response = OpenAiErrorFormatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                parsed["error"]["type"], "server_error",
                "status {status} should use server_error type"
            );
            assert_eq!(
                parsed["error"]["code"], "some_code",
                "status {status} should preserve the Praxis code"
            );
        }
    }

    #[test]
    fn fourx_status_maps_to_openai_error_types() {
        let cases = [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (418, "api_error"),
        ];

        for (status, expected_type) in cases {
            let ctx = ErrorResponseContext::new("custom_code", "test error", status);
            let response = OpenAiErrorFormatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(
                parsed["error"]["type"], expected_type,
                "status {status} should map to error type '{expected_type}'"
            );
            assert_eq!(
                parsed["error"]["code"], "custom_code",
                "status {status} should preserve the Praxis code"
            );
        }
    }

    #[test]
    fn param_is_always_null() {
        for status in [400, 429, 500, 502, 504] {
            let ctx = ErrorResponseContext::new("test_code", "test message", status);
            let response = OpenAiErrorFormatter.format(&ctx);

            let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
            assert!(
                parsed["error"]["param"].is_null(),
                "param should be null for status {status}"
            );
        }
    }

    #[test]
    fn output_is_valid_json() {
        let ctx = ErrorResponseContext::new("upstream_connect_error", "failed", 502);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&response.body);
        assert!(parsed.is_ok(), "output must be valid JSON");

        let parsed = parsed.unwrap();
        assert!(parsed.get("error").is_some(), "must have top-level error key");
        assert!(parsed["error"].get("message").is_some());
        assert!(parsed["error"].get("type").is_some());
        assert!(parsed["error"].get("param").is_some());
        assert!(parsed["error"].get("code").is_some());
    }
}
