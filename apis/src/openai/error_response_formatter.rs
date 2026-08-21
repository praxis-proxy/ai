// SPDX-License-Identifier: MIT
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
/// - `error.type` ← `"server_error"` for 5xx, `context.code` otherwise
/// - `error.param` ← always `null`
pub(crate) struct OpenAiErrorFormatter;

impl ErrorResponseFormatter for OpenAiErrorFormatter {
    fn format(&self, context: &ErrorResponseContext<'_>) -> FormattedErrorResponse {
        let error_type = if context.status >= 500 {
            "server_error"
        } else {
            context.code
        };

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
        let ctx = ErrorResponseContext::new("upstream_connect_error", "Connection refused", 502);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["error"]["message"], "Connection refused");
        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["code"], "upstream_connect_error");
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
    fn fourx_status_uses_code_as_type() {
        let ctx = ErrorResponseContext::new("rate_limit_exceeded", "Too many requests", 429);
        let response = OpenAiErrorFormatter.format(&ctx);

        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(parsed["error"]["type"], "rate_limit_exceeded");
        assert_eq!(parsed["error"]["code"], "rate_limit_exceeded");
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
