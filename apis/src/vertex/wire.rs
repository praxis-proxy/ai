// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared Vertex AI wire-format utilities.
//!
//! Normalizes Vertex AI error responses into the OpenAI Chat Completions
//! error envelope so that clients see a consistent shape regardless of
//! whether the upstream is native OpenAI or a Vertex AI deployment.

use http::StatusCode;
use serde::Deserialize;
use serde_json::Value;

// -----------------------------------------------------------------------------
// Vertex Error Deserialization
// -----------------------------------------------------------------------------

/// Minimal upstream Vertex AI error structure used for normalization.
///
/// Vertex returns errors in the Google Cloud REST format:
///
/// ```json
/// {
///   "error": {
///     "code": 400,
///     "message": "Request contains an invalid argument.",
///     "status": "INVALID_ARGUMENT"
///   }
/// }
/// ```
///
/// Only the fields needed for normalization are deserialized; unknown
/// fields are silently ignored.
#[derive(Deserialize)]
struct VertexError {
    /// Nested error object, when present.
    error: Option<VertexErrorDetail>,
}

/// Inner `error` object from Vertex AI responses.
#[derive(Deserialize)]
struct VertexErrorDetail {
    /// Human-readable error description.
    message: Option<String>,
    /// Google Cloud gRPC status string (e.g. `INVALID_ARGUMENT`).
    status: Option<String>,
}

// -----------------------------------------------------------------------------
// Error Normalization
// -----------------------------------------------------------------------------

/// Normalize a Vertex AI error body into the OpenAI Chat Completions
/// error envelope.
///
/// Extracts the message and gRPC status from the Vertex error JSON,
/// maps them to the OpenAI `type` and `code` fields, and serializes
/// a new body. Falls back to the HTTP status code when the Vertex
/// body is absent or unparseable.
///
/// # Output format
///
/// ```json
/// {
///   "error": {
///     "message": "...",
///     "type": "invalid_request_error",
///     "param": null,
///     "code": "invalid_argument"
///   }
/// }
/// ```
pub(crate) fn normalize_error_response(body: &[u8], status: StatusCode) -> Vec<u8> {
    let parsed = serde_json::from_slice::<VertexError>(body).ok();

    let message = parsed
        .as_ref()
        .and_then(|e| e.error.as_ref())
        .and_then(|d| d.message.as_deref())
        .unwrap_or("upstream request failed");

    let vertex_status = parsed
        .as_ref()
        .and_then(|e| e.error.as_ref())
        .and_then(|d| d.status.as_deref());

    let (error_type, code) = vertex_status.map_or_else(
        || openai_mapping_for_http_status(status),
        openai_mapping_for_vertex_status,
    );

    build_openai_error_body(message, error_type, code)
}

// -----------------------------------------------------------------------------
// Status Mapping
// -----------------------------------------------------------------------------

/// Map a Vertex AI gRPC status string to OpenAI `(error.type, error.code)`.
///
/// `type` is the broad category (`invalid_request_error`, `server_error`,
/// …) and `code` provides finer granularity derived from the gRPC status.
///
/// Uses the [canonical gRPC status codes] that Vertex AI returns.
///
/// [canonical gRPC status codes]: https://grpc.github.io/grpc/core/md_doc_statuscodes.html
fn openai_mapping_for_vertex_status(status: &str) -> (&'static str, &'static str) {
    match status {
        "INVALID_ARGUMENT" => ("invalid_request_error", "invalid_argument"),
        "FAILED_PRECONDITION" => ("invalid_request_error", "failed_precondition"),
        "OUT_OF_RANGE" => ("invalid_request_error", "out_of_range"),
        "UNAUTHENTICATED" => ("authentication_error", "unauthenticated"),
        "PERMISSION_DENIED" => ("permission_error", "permission_denied"),
        "NOT_FOUND" => ("not_found_error", "not_found"),
        "RESOURCE_EXHAUSTED" => ("rate_limit_error", "resource_exhausted"),
        "DEADLINE_EXCEEDED" => ("server_error", "deadline_exceeded"),
        "CANCELLED" => ("server_error", "cancelled"),
        "ABORTED" => ("server_error", "aborted"),
        "UNAVAILABLE" => ("server_error", "unavailable"),
        "INTERNAL" => ("server_error", "internal"),
        "DATA_LOSS" => ("server_error", "data_loss"),
        _ => ("server_error", "unknown"),
    }
}

/// Fallback: map an HTTP status code to OpenAI `(error.type, error.code)`
/// when no Vertex gRPC status string is available.
///
/// Both fields carry the same value because without the Vertex status
/// there is no extra granularity to convey.
fn openai_mapping_for_http_status(status: StatusCode) -> (&'static str, &'static str) {
    let value = match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        _ => "server_error",
    };
    (value, value)
}

// -----------------------------------------------------------------------------
// OpenAI Error Body Builder
// -----------------------------------------------------------------------------

/// Serialize an OpenAI Chat Completions error envelope.
///
/// Produces valid JSON even when `message` contains special characters.
/// Falls back to a hardcoded JSON string if serialization fails.
pub(crate) fn build_openai_error_body(message: &str, error_type: &str, code: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": Value::Null,
            "code": code,
        }
    }))
    .unwrap_or_else(|_| {
        br#"{"error":{"message":"upstream request failed","type":"server_error","param":null,"code":"server_error"}}"#
            .to_vec()
    })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // normalize_error_response
    // -------------------------------------------------------------------------

    #[test]
    fn vertex_error_with_status_maps_to_openai_type_and_code() {
        let body = br#"{"error":{"code":400,"message":"Invalid value","status":"INVALID_ARGUMENT"}}"#;
        let output = normalize_error_response(body, StatusCode::BAD_REQUEST);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "Invalid value");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["code"], "invalid_argument");
        assert!(parsed["error"]["param"].is_null());
    }

    #[test]
    fn vertex_permission_denied_maps_correctly() {
        let body =
            br#"{"error":{"code":403,"message":"Caller does not have permission","status":"PERMISSION_DENIED"}}"#;
        let output = normalize_error_response(body, StatusCode::FORBIDDEN);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "permission_error");
        assert_eq!(parsed["error"]["code"], "permission_denied");
    }

    #[test]
    fn vertex_resource_exhausted_maps_to_rate_limit() {
        let body = br#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#;
        let output = normalize_error_response(body, StatusCode::TOO_MANY_REQUESTS);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert_eq!(parsed["error"]["code"], "resource_exhausted");
    }

    #[test]
    fn vertex_not_found_maps_correctly() {
        let body = br#"{"error":{"code":404,"message":"Model not found","status":"NOT_FOUND"}}"#;
        let output = normalize_error_response(body, StatusCode::NOT_FOUND);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "not_found_error");
        assert_eq!(parsed["error"]["code"], "not_found");
    }

    #[test]
    fn vertex_unauthenticated_maps_to_auth_error() {
        let body =
            br#"{"error":{"code":401,"message":"Request had invalid authentication","status":"UNAUTHENTICATED"}}"#;
        let output = normalize_error_response(body, StatusCode::UNAUTHORIZED);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "authentication_error");
        assert_eq!(parsed["error"]["code"], "unauthenticated");
    }

    #[test]
    fn vertex_internal_maps_to_server_error() {
        let body = br#"{"error":{"code":500,"message":"Internal error","status":"INTERNAL"}}"#;
        let output = normalize_error_response(body, StatusCode::INTERNAL_SERVER_ERROR);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["code"], "internal");
    }

    #[test]
    fn unknown_vertex_status_falls_back_to_server_error() {
        let body = br#"{"error":{"code":500,"message":"Something new","status":"FUTURE_STATUS"}}"#;
        let output = normalize_error_response(body, StatusCode::INTERNAL_SERVER_ERROR);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["code"], "unknown");
    }

    // -------------------------------------------------------------------------
    // Fallback to HTTP status (no Vertex status field)
    // -------------------------------------------------------------------------

    #[test]
    fn missing_vertex_status_falls_back_to_http_status() {
        let body = br#"{"error":{"code":400,"message":"Bad request"}}"#;
        let output = normalize_error_response(body, StatusCode::BAD_REQUEST);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["code"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "Bad request");
    }

    #[test]
    fn http_429_without_vertex_status_maps_to_rate_limit() {
        let body = br#"{"error":{"message":"Too many requests"}}"#;
        let output = normalize_error_response(body, StatusCode::TOO_MANY_REQUESTS);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["type"], "rate_limit_error");
        assert_eq!(parsed["error"]["code"], "rate_limit_error");
    }

    // -------------------------------------------------------------------------
    // Unparseable and missing bodies
    // -------------------------------------------------------------------------

    #[test]
    fn empty_body_uses_fallback_message() {
        let output = normalize_error_response(b"", StatusCode::BAD_GATEWAY);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "upstream request failed");
        assert_eq!(parsed["error"]["type"], "server_error");
    }

    #[test]
    fn html_body_uses_fallback_message() {
        let output = normalize_error_response(b"<html>service unavailable</html>", StatusCode::SERVICE_UNAVAILABLE);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "upstream request failed");
        assert_eq!(parsed["error"]["type"], "server_error");
    }

    #[test]
    fn json_array_body_uses_fallback_message() {
        let output = normalize_error_response(b"[1,2,3]", StatusCode::INTERNAL_SERVER_ERROR);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "upstream request failed");
    }

    // -------------------------------------------------------------------------
    // Message preservation
    // -------------------------------------------------------------------------

    #[test]
    fn special_characters_in_message_are_json_safe() {
        let body = br#"{"error":{"code":400,"message":"bad \"model\"\nvalue","status":"INVALID_ARGUMENT"}}"#;
        let output = normalize_error_response(body, StatusCode::BAD_REQUEST);
        let parsed: Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(parsed["error"]["message"], "bad \"model\"\nvalue");
    }

    // -------------------------------------------------------------------------
    // Status mapping coverage
    // -------------------------------------------------------------------------

    #[test]
    fn all_vertex_statuses_have_defined_mappings() {
        let statuses = [
            ("INVALID_ARGUMENT", "invalid_request_error", "invalid_argument"),
            ("FAILED_PRECONDITION", "invalid_request_error", "failed_precondition"),
            ("OUT_OF_RANGE", "invalid_request_error", "out_of_range"),
            ("UNAUTHENTICATED", "authentication_error", "unauthenticated"),
            ("PERMISSION_DENIED", "permission_error", "permission_denied"),
            ("NOT_FOUND", "not_found_error", "not_found"),
            ("RESOURCE_EXHAUSTED", "rate_limit_error", "resource_exhausted"),
            ("DEADLINE_EXCEEDED", "server_error", "deadline_exceeded"),
            ("CANCELLED", "server_error", "cancelled"),
            ("ABORTED", "server_error", "aborted"),
            ("UNAVAILABLE", "server_error", "unavailable"),
            ("INTERNAL", "server_error", "internal"),
            ("DATA_LOSS", "server_error", "data_loss"),
        ];

        for (vertex, expected_type, expected_code) in statuses {
            let (actual_type, actual_code) = openai_mapping_for_vertex_status(vertex);
            assert_eq!(actual_type, expected_type, "type mapping for {vertex}");
            assert_eq!(actual_code, expected_code, "code mapping for {vertex}");
        }
    }

    #[test]
    fn all_http_fallback_statuses_have_defined_mappings() {
        let statuses = [
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::UNAUTHORIZED, "authentication_error"),
            (StatusCode::FORBIDDEN, "permission_error"),
            (StatusCode::NOT_FOUND, "not_found_error"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        ];

        for (status, expected) in statuses {
            let (actual_type, actual_code) = openai_mapping_for_http_status(status);
            assert_eq!(actual_type, expected, "HTTP fallback type for {status}");
            assert_eq!(
                actual_code, expected,
                "HTTP fallback code should equal type for {status}"
            );
        }
    }
}
