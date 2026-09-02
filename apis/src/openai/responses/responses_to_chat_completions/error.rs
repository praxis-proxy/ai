// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use http::StatusCode;
use serde_json::Value;

/// Provider error codes that can be represented directly by the Responses API.
const VALID_RESPONSE_ERROR_CODES: &[&str] = &[
    "server_error",
    "rate_limit_exceeded",
    "invalid_prompt",
    "vector_store_timeout",
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_content_policy_violation",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
];

/// Safe fallback used when the provider body has no public message.
const GENERIC_PROVIDER_ERROR_MESSAGE: &str = "upstream provider returned an error";

/// Responses-compatible provider error fields.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct NormalizedProviderError {
    /// Responses API error code.
    pub code: String,
    /// Public error message returned to the client.
    pub message: String,
}

/// Normalize a finite provider error without reflecting unknown payload fields.
pub(super) fn normalize_provider_error(status: StatusCode, body: &[u8]) -> NormalizedProviderError {
    let parsed = serde_json::from_slice::<Value>(body).ok();
    let error = parsed
        .as_ref()
        .map(|parsed| parsed.get("error").filter(|error| error.is_object()).unwrap_or(parsed));
    let (provider_code, message) = error.map_or((None, None), |error| {
        (
            error.get("code").and_then(Value::as_str),
            error.get("message").and_then(Value::as_str),
        )
    });

    NormalizedProviderError {
        code: normalize_code(status, provider_code),
        message: message.unwrap_or(GENERIC_PROVIDER_ERROR_MESSAGE).to_owned(),
    }
}

/// Map a provider code and HTTP status to a Responses-compatible code.
fn normalize_code(status: StatusCode, provider_code: Option<&str>) -> String {
    if provider_code == Some("invalid_base64") {
        return "invalid_base64_image".to_owned();
    }
    if let Some(code) = provider_code.filter(|code| VALID_RESPONSE_ERROR_CODES.contains(code)) {
        return code.to_owned();
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        return "rate_limit_exceeded".to_owned();
    }
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
        return "server_error".to_owned();
    }
    if status.is_client_error() {
        "invalid_prompt".to_owned()
    } else {
        "server_error".to_owned()
    }
}
