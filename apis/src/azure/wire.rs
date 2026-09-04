// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared Azure OpenAI wire types.
//!
//! Azure OpenAI uses the Chat Completions error envelope, so these
//! helpers normalize Azure-specific differences (e.g. `type: null`)
//! back to the standard Chat Completions error shape.

use serde_json::Value;

/// Normalize an Azure OpenAI error response into Chat Completions form.
///
/// Azure errors already use the `{"error":{...}}` envelope but may include
/// `innererror` and use `null` for the `type` field. This normalizes the
/// `type` from the HTTP status when the upstream omits it.
///
/// Returns `Some(normalized_bytes)` when the response was modified, or
/// `None` when no changes were necessary.
pub(crate) fn normalize_error_response(body: &[u8], status: http::StatusCode) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let error = value.get_mut("error")?.as_object_mut()?;

    let mut modified = false;

    let needs_type_fix = error
        .get("type")
        .is_some_and(|v| v.is_null() || v.as_str().is_some_and(str::is_empty));

    if needs_type_fix {
        error.insert(
            "type".to_owned(),
            Value::String(error_type_for_status(status).to_owned()),
        );
        modified = true;
    }

    if error.remove("innererror").is_some() {
        modified = true;
    }

    if !modified {
        return None;
    }

    serde_json::to_vec(&value).ok()
}

/// Map an HTTP error status to an OpenAI error type.
fn error_type_for_status(status: http::StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "invalid_api_key",
        403 => "permission_error",
        429 => "rate_limit_error",
        500..=599 => "server_error",
        _ => "invalid_request_error",
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn normalize_fills_null_type_from_status() {
        let azure_error = br#"{"error":{"message":"deployment not found","type":null,"code":"DeploymentNotFound","innererror":{"code":"DeploymentNotFound"}}}"#;
        let result = normalize_error_response(azure_error, http::StatusCode::NOT_FOUND).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["message"], "deployment not found");
        assert_eq!(parsed["error"]["code"], "DeploymentNotFound");
        assert!(
            parsed["error"].get("innererror").is_none(),
            "innererror should be stripped"
        );
    }

    #[test]
    fn normalize_strips_innererror_even_when_type_present() {
        let error =
            br#"{"error":{"message":"bad","type":"invalid_request_error","code":"bad","innererror":{"code":"bad"}}}"#;
        let result = normalize_error_response(error, http::StatusCode::BAD_REQUEST).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert!(
            parsed["error"].get("innererror").is_none(),
            "innererror should be stripped"
        );
    }

    #[test]
    fn normalize_returns_none_when_type_present() {
        let openai_error = br#"{"error":{"message":"bad","type":"invalid_request_error","code":"bad_request"}}"#;
        let result = normalize_error_response(openai_error, http::StatusCode::BAD_REQUEST);

        assert!(result.is_none(), "already-valid errors should not be rewritten");
    }

    #[test]
    fn normalize_returns_none_for_non_json() {
        let result = normalize_error_response(b"not json", http::StatusCode::BAD_REQUEST);

        assert!(result.is_none());
    }

    #[test]
    fn normalize_fills_empty_string_type() {
        let error = br#"{"error":{"message":"bad","type":"","code":"bad_request"}}"#;
        let result = normalize_error_response(error, http::StatusCode::BAD_REQUEST).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn error_type_mapping_covers_key_statuses() {
        assert_eq!(error_type_for_status(http::StatusCode::UNAUTHORIZED), "invalid_api_key");
        assert_eq!(
            error_type_for_status(http::StatusCode::TOO_MANY_REQUESTS),
            "rate_limit_error"
        );
        assert_eq!(
            error_type_for_status(http::StatusCode::INTERNAL_SERVER_ERROR),
            "server_error"
        );
    }
}
