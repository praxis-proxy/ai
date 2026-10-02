// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bedrock error normalization to the OpenAI Chat Completions error envelope.
//!
//! Bedrock returns errors as a simple JSON object:
//!
//! ```json
//! {"message": "The model is not ready to serve inference requests."}
//! ```
//!
//! This module translates that into the Chat Completions error shape:
//!
//! ```json
//! {
//!   "error": {
//!     "message": "The model is not ready to serve inference requests.",
//!     "type": "server_error",
//!     "code": "model_not_ready"
//!   }
//! }
//! ```
//!
//! The mapping relies on the HTTP status code because Bedrock error bodies
//! carry only a `message` field — there is no structured `type` or `code`.

use http::StatusCode;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Bedrock Error Shape
// -----------------------------------------------------------------------------

/// Bedrock error response body.
///
/// Bedrock returns a flat `{"message": "..."}` for all error types.
/// The specific exception type (e.g. `ValidationException`,
/// `ThrottlingException`) is signalled by the HTTP status code, not
/// by a field in the body.
#[derive(Deserialize)]
struct BedrockError {
    /// Human-readable diagnostic from the upstream service.
    #[serde(default)]
    message: String,
}

// -----------------------------------------------------------------------------
// OpenAI Error Envelope
// -----------------------------------------------------------------------------

/// Build a complete OpenAI Chat Completions error JSON body.
///
/// # Arguments
///
/// * `message` — human-readable error message
/// * `error_type` — OpenAI error category (e.g. `"invalid_request_error"`)
/// * `code` — machine-readable error code (e.g. `"model_not_found"`)
///
/// Returns the JSON bytes. Falls back to a minimal hard-coded body if
/// serialization unexpectedly fails.
pub(crate) fn build_openai_error_body(message: &str, error_type: &str, code: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    }))
    .unwrap_or_else(|_| {
        br#"{"error":{"message":"internal error","type":"server_error","code":"internal_error"}}"#.to_vec()
    })
}

// -----------------------------------------------------------------------------
// Error Normalization
// -----------------------------------------------------------------------------

/// Normalize a Bedrock error response into the OpenAI Chat Completions
/// error envelope.
///
/// Extracts the `message` from the Bedrock JSON body (or falls back to
/// a generic message) and maps the HTTP status code to the appropriate
/// OpenAI `type` and `code`.
pub(crate) fn normalize_error_response(body: &[u8], status: StatusCode) -> Vec<u8> {
    let message = serde_json::from_slice::<BedrockError>(body)
        .map(|e| e.message)
        .unwrap_or_default();

    let message = if message.is_empty() {
        format!("Bedrock returned HTTP {status}")
    } else {
        message
    };

    let (error_type, code) = openai_mapping_for_http_status(status);
    build_openai_error_body(&message, error_type, code)
}

/// Map an HTTP status code to the OpenAI `(type, code)` pair.
///
/// Bedrock uses standard HTTP codes for its exception types:
///
/// | HTTP | Bedrock Exception            | OpenAI type              | OpenAI code          |
/// |------|------------------------------|--------------------------|----------------------|
/// | 400  | `ValidationException`        | `invalid_request_error`  | `invalid_request`    |
/// | 403  | `AccessDeniedException`      | `invalid_request_error`  | `access_denied`      |
/// | 404  | `ResourceNotFoundException`  | `invalid_request_error`  | `model_not_found`    |
/// | 408  | `ModelTimeoutException`      | `server_error`           | `timeout`            |
/// | 424  | `ModelErrorException`        | `server_error`           | `model_error`        |
/// | 429  | `ThrottlingException`        | `rate_limit_error`       | `rate_limit_exceeded`|
/// | 500  | `InternalServerException`    | `server_error`           | `internal_error`     |
/// | 503  | `ServiceUnavailableException`| `server_error`           | `service_unavailable`|
fn openai_mapping_for_http_status(status: StatusCode) -> (&'static str, &'static str) {
    match status.as_u16() {
        400 => ("invalid_request_error", "invalid_request"),
        403 => ("invalid_request_error", "access_denied"),
        404 => ("invalid_request_error", "model_not_found"),
        408 => ("server_error", "timeout"),
        424 => ("server_error", "model_error"),
        429 => ("rate_limit_error", "rate_limit_exceeded"),
        503 => ("server_error", "service_unavailable"),
        // All other 4xx → invalid_request_error, all other 5xx → server_error.
        s if (400..500).contains(&s) => ("invalid_request_error", "invalid_request"),
        _ => ("server_error", "internal_error"),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    // --- build_openai_error_body ---

    #[test]
    fn error_body_produces_valid_openai_envelope() {
        let body = build_openai_error_body("bad model", "invalid_request_error", "model_not_found");
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            parsed["error"]["message"], "bad model",
            "error envelope must carry the provided message"
        );
        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "error envelope must carry the provided type"
        );
        assert_eq!(
            parsed["error"]["code"], "model_not_found",
            "error envelope must carry the provided code"
        );
    }

    #[test]
    fn error_body_escapes_special_characters() {
        let body = build_openai_error_body("bad \"model\"\nnewline", "invalid_request_error", "invalid_request");
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            parsed["error"]["message"], "bad \"model\"\nnewline",
            "special characters in the message must round-trip through JSON encoding"
        );
    }

    // --- normalize_error_response ---

    #[test]
    fn normalize_validation_error() {
        let body = br#"{"message":"The input fails to satisfy the constraints"}"#;
        let result = normalize_error_response(body, StatusCode::BAD_REQUEST);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["message"], "The input fails to satisfy the constraints",
            "Bedrock message must be preserved in the normalized envelope"
        );
        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "HTTP 400 must map to invalid_request_error"
        );
        assert_eq!(
            parsed["error"]["code"], "invalid_request",
            "HTTP 400 must map to the invalid_request code"
        );
    }

    #[test]
    fn normalize_throttling_error() {
        let body = br#"{"message":"Too many requests"}"#;
        let result = normalize_error_response(body, StatusCode::TOO_MANY_REQUESTS);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "rate_limit_error",
            "HTTP 429 must map to rate_limit_error"
        );
        assert_eq!(
            parsed["error"]["code"], "rate_limit_exceeded",
            "HTTP 429 must map to the rate_limit_exceeded code"
        );
    }

    #[test]
    fn normalize_access_denied() {
        let body = br#"{"message":"Access denied for model"}"#;
        let result = normalize_error_response(body, StatusCode::FORBIDDEN);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "HTTP 403 must map to invalid_request_error"
        );
        assert_eq!(
            parsed["error"]["code"], "access_denied",
            "HTTP 403 must map to the access_denied code"
        );
    }

    #[test]
    fn normalize_model_not_found() {
        let body = br#"{"message":"Could not resolve the foundation model"}"#;
        let result = normalize_error_response(body, StatusCode::NOT_FOUND);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "HTTP 404 must map to invalid_request_error"
        );
        assert_eq!(
            parsed["error"]["code"], "model_not_found",
            "HTTP 404 must map to the model_not_found code"
        );
    }

    #[test]
    fn normalize_model_error_424() {
        let body = br#"{"message":"Model processing failed"}"#;
        let result = normalize_error_response(body, StatusCode::from_u16(424).unwrap());
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "server_error",
            "HTTP 424 must map to server_error"
        );
        assert_eq!(
            parsed["error"]["code"], "model_error",
            "HTTP 424 must map to the model_error code"
        );
    }

    #[test]
    fn normalize_timeout() {
        let body = br#"{"message":"Request timed out"}"#;
        let result = normalize_error_response(body, StatusCode::REQUEST_TIMEOUT);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "server_error",
            "HTTP 408 must map to server_error"
        );
        assert_eq!(
            parsed["error"]["code"], "timeout",
            "HTTP 408 must map to the timeout code"
        );
    }

    #[test]
    fn normalize_service_unavailable() {
        let body = br#"{"message":"Service unavailable"}"#;
        let result = normalize_error_response(body, StatusCode::SERVICE_UNAVAILABLE);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "server_error",
            "HTTP 503 must map to server_error"
        );
        assert_eq!(
            parsed["error"]["code"], "service_unavailable",
            "HTTP 503 must map to the service_unavailable code"
        );
    }

    #[test]
    fn normalize_internal_server_error() {
        let body = br#"{"message":"Internal failure"}"#;
        let result = normalize_error_response(body, StatusCode::INTERNAL_SERVER_ERROR);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["type"], "server_error",
            "HTTP 500 must map to server_error"
        );
        assert_eq!(
            parsed["error"]["code"], "internal_error",
            "HTTP 500 must map to the internal_error code"
        );
    }

    #[test]
    fn normalize_unparseable_body_uses_status_fallback() {
        let body = b"not json at all";
        let result = normalize_error_response(body, StatusCode::BAD_REQUEST);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["message"], "Bedrock returned HTTP 400 Bad Request",
            "an unparseable body must fall back to a status-derived message"
        );
        assert_eq!(
            parsed["error"]["type"], "invalid_request_error",
            "status mapping must still apply when the body is unparseable"
        );
    }

    #[test]
    fn normalize_empty_message_uses_status_fallback() {
        let body = br#"{"message":""}"#;
        let result = normalize_error_response(body, StatusCode::FORBIDDEN);
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed["error"]["message"], "Bedrock returned HTTP 403 Forbidden",
            "an empty message must fall back to a status-derived message"
        );
        assert_eq!(
            parsed["error"]["code"], "access_denied",
            "status mapping must still apply when the message is empty"
        );
    }

    // --- openai_mapping_for_http_status ---

    #[test]
    fn unknown_4xx_maps_to_invalid_request() {
        let (t, c) = openai_mapping_for_http_status(StatusCode::GONE);
        assert_eq!(
            t, "invalid_request_error",
            "an unmapped 4xx must default to invalid_request_error"
        );
        assert_eq!(
            c, "invalid_request",
            "an unmapped 4xx must default to the invalid_request code"
        );
    }

    #[test]
    fn unknown_5xx_maps_to_server_error() {
        let (t, c) = openai_mapping_for_http_status(StatusCode::BAD_GATEWAY);
        assert_eq!(t, "server_error", "an unmapped 5xx must default to server_error");
        assert_eq!(
            c, "internal_error",
            "an unmapped 5xx must default to the internal_error code"
        );
    }
}
