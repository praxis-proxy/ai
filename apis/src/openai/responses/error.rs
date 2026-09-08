// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses API error formatting.
//!
//! Builds OpenAI-compatible error responses. Locally generated rejections
//! are pre-commitment short-circuits and always use the HTTP API
//! `{"error":{...}}` envelope, matching OpenAI's behavior of returning a
//! JSON error with a non-2xx status even when the request set `stream: true`.
//! The Responses API SSE `error` event shape is used only for errors on an
//! already-committed `text/event-stream` (see the `stream_events` filter).

use bytes::Bytes;
use praxis_filter::Rejection;

/// Build a non-streaming OpenAI API error JSON body.
///
/// Produces `{"error":{"message":"<msg>","type":"<code>","param":null,"code":"<code>"}}`.
pub(crate) fn responses_error_body(code: &str, message: &str) -> Bytes {
    responses_error_body_with_code(code, code, message)
}

/// Build a non-streaming OpenAI API error JSON body with distinct type and code.
fn responses_error_body_with_code(error_type: &str, code: &str, message: &str) -> Bytes {
    Bytes::from(
        serde_json::json!({
            "error": {
                "message": message,
                "type": error_type,
                "param": null,
                "code": code,
            },
        })
        .to_string(),
    )
}

/// Build the SSE `error` event frame at a logical-stream sequence number.
///
/// Produces `event: error\ndata: <ResponseErrorEvent json>\n\n`. Used by
/// dispatch filters that finalize an already-committed stream directly, without
/// a downstream `stream_events` finalizer.
pub(crate) fn responses_error_sse_body_at_sequence(code: &str, message: &str, sequence_number: u64) -> Bytes {
    let json = responses_error_sse_payload_at_sequence(code, message, sequence_number);
    Bytes::from(format!("event: error\ndata: {json}\n\n"))
}

/// Build the JSON payload for a Responses API SSE `error` event.
///
/// Matches the pinned OpenAI `ResponseErrorEvent` schema: `type` (always
/// `"error"`), `code`, `message`, `param`, and `sequence_number` are all
/// top-level fields — there is no nested `error` object. The caller-selected
/// machine-readable `code` is preserved as the top-level `code`.
///
/// Used only for errors on an already-committed streaming response (the
/// `stream_events` logical-stream terminal error, or a dispatch filter that
/// finalizes the stream directly). Pre-commitment rejections use the JSON
/// envelope via [`responses_error_rejection`] instead.
pub(crate) fn responses_error_sse_payload(code: &str, message: &str) -> serde_json::Value {
    responses_error_sse_payload_at_sequence(code, message, 0)
}

/// Build the JSON payload for an SSE error at a logical-stream sequence.
fn responses_error_sse_payload_at_sequence(code: &str, message: &str, sequence_number: u64) -> serde_json::Value {
    serde_json::json!({
        "type": "error",
        "sequence_number": sequence_number,
        "code": code,
        "message": message,
        "param": null,
    })
}

/// Build a [`Rejection`] carrying the OpenAI HTTP error envelope.
///
/// A [`Rejection`] is always a pre-commitment short-circuit: it replaces the
/// entire HTTP response before any streaming body is committed downstream, so
/// no `text/event-stream` has been established. OpenAI returns errors detected
/// before a stream starts as an ordinary JSON `{"error":{...}}` body with a
/// non-2xx status, even when the request set `stream: true`; SSE `error`
/// events are reserved for failures on an already-committed stream and are
/// emitted by the `stream_events` filter, not here. Every rejection therefore
/// uses `application/json`.
pub(crate) fn responses_error_rejection(status: u16, code: &str, message: &str) -> Rejection {
    responses_error_rejection_with_code(status, code, code, message)
}

/// Build a [`Rejection`] with distinct OpenAI error type and code values.
pub(crate) fn responses_error_rejection_with_code(
    status: u16,
    error_type: &str,
    code: &str,
    message: &str,
) -> Rejection {
    Rejection::status(status)
        .with_header("content-type", "application/json")
        .with_body(responses_error_body_with_code(error_type, code, message))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn error_body_has_correct_shape() {
        let body = responses_error_body("invalid_request_error", "bad input");
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "error type should match"
        );
        assert_eq!(
            parsed["error"]["code"], "invalid_request_error",
            "error code should match"
        );
        assert_eq!(parsed["error"]["message"], "bad input", "message field should match");
        assert!(parsed["error"]["param"].is_null(), "param should be null");
    }

    #[test]
    fn rejection_supports_distinct_error_type_and_code() {
        let rejection = responses_error_rejection_with_code(
            400,
            "invalid_request_error",
            "mutually_exclusive_parameters",
            "bad input",
        );
        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();

        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "error type should match"
        );
        assert_eq!(
            body["error"]["code"], "mutually_exclusive_parameters",
            "error code should match"
        );
        assert!(body["error"]["param"].is_null(), "param should be null");
    }

    #[test]
    fn error_body_escapes_special_characters() {
        let body = responses_error_body("server_error", "line1\nline2\"quoted\"");
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str(),
            Some("line1\nline2\"quoted\""),
            "special characters should survive JSON round-trip"
        );
    }

    #[test]
    fn sse_payload_matches_pinned_response_error_event_schema() {
        let payload = responses_error_sse_payload("rate_limit_exceeded", "slow down");

        assert_eq!(payload["type"], "error", "event type is always \"error\"");
        assert_eq!(payload["sequence_number"], 0, "sequence_number is a top-level field");
        assert_eq!(
            payload["code"], "rate_limit_exceeded",
            "the caller-selected machine-readable code is preserved at the top level"
        );
        assert_eq!(payload["message"], "slow down", "message is a top-level field");
        assert!(payload["param"].is_null(), "param is a top-level field");
        assert!(
            payload.get("error").is_none(),
            "an SSE error event must not nest fields under an \"error\" object"
        );

        let keys: std::collections::BTreeSet<&str> = payload.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["code", "message", "param", "sequence_number", "type"]
                .into_iter()
                .collect(),
            "payload must contain only the schema-defined top-level fields"
        );
    }

    #[test]
    fn sse_body_contains_valid_json() {
        let body = responses_error_sse_body_at_sequence("invalid_request_error", "missing field", 0);
        let text = std::str::from_utf8(&body).unwrap();

        let data_line = text
            .lines()
            .find(|l| l.starts_with("data: "))
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(data_line).unwrap();

        assert_eq!(parsed["type"], "error", "SSE event type should be error");
        assert_eq!(parsed["sequence_number"], 0, "SSE error should include sequence number");
        assert_eq!(
            parsed["code"], "invalid_request_error",
            "SSE error code should be a top-level field"
        );
        assert_eq!(
            parsed["message"], "missing field",
            "SSE error message should be a top-level field"
        );
        assert!(parsed["param"].is_null(), "SSE error param should be a top-level null");
        assert!(
            parsed.get("error").is_none(),
            "an SSE error event must not nest fields under an \"error\" object"
        );
    }

    #[test]
    fn sse_body_preserves_logical_sequence() {
        let body = responses_error_sse_body_at_sequence("server_error", "oops", 7);
        let text = std::str::from_utf8(&body).unwrap();
        let data = text.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
        let payload: serde_json::Value = serde_json::from_str(data).unwrap();

        assert_eq!(payload["sequence_number"], 7);
    }

    #[test]
    fn rejection_uses_json_content_type() {
        let r = responses_error_rejection(400, "invalid_request_error", "bad");

        assert_eq!(r.status, 400, "status should be preserved");
        let ct = r.headers.iter().find(|(k, _)| k == "content-type");
        assert_eq!(
            ct.map(|(_, v)| v.as_str()),
            Some("application/json"),
            "rejections use application/json"
        );
    }

    #[test]
    fn rejection_uses_json_content_type_for_server_errors() {
        let r = responses_error_rejection(500, "server_error", "fail");

        assert_eq!(r.status, 500, "status should be preserved");
        let ct = r.headers.iter().find(|(k, _)| k == "content-type");
        assert_eq!(
            ct.map(|(_, v)| v.as_str()),
            Some("application/json"),
            "rejections use application/json regardless of status"
        );
    }

    #[test]
    fn rejection_body_is_json_envelope() {
        let r = responses_error_rejection(404, "invalid_request_error", "not found");
        let body = r.body.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(
            !text.starts_with("event: "),
            "a rejection body must not be an SSE event"
        );
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "rejection error type should match"
        );
        assert_eq!(
            parsed["error"]["code"], "invalid_request_error",
            "rejection error code should match"
        );
        assert_eq!(
            parsed["error"]["message"], "not found",
            "rejection error message should match"
        );
        assert!(
            parsed["error"]["param"].is_null(),
            "rejection error param should be null"
        );
    }
}
