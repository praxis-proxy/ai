// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Responses format classifier filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::OnInvalidBehavior};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Behavior Enums
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// ResponsesFormatHeaders
// -----------------------------------------------------------------------------

/// Configurable header names for promoted classification facts.
///
/// Transport, credential, API-key, and other internal `x-praxis-*` names
/// are rejected. Each field may use its dedicated default or a custom
/// non-`x-praxis-*` header.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponsesFormatHeaders {
    /// Header name for the detected format (e.g. `openai_responses`, `openai_chat_completions`).
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-format` remains allowed.
    #[serde(default = "default_format_header")]
    pub format: Option<String>,

    /// Header name for the extracted model value.
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-model` remains allowed. Must not overwrite other
    /// classification facts such as `x-praxis-ai-format`.
    #[serde(default = "default_model_header")]
    pub model: Option<String>,

    /// Header name for the extracted stream flag.
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-stream` remains allowed.
    #[serde(default = "default_stream_header")]
    pub stream: Option<String>,

    /// Header name for the computed mode (`stateless` or `stateful`).
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-responses-mode` remains allowed.
    #[serde(default = "default_mode_header")]
    pub mode: Option<String>,
}

impl Default for ResponsesFormatHeaders {
    fn default() -> Self {
        Self {
            format: default_format_header(),
            model: default_model_header(),
            stream: default_stream_header(),
            mode: default_mode_header(),
        }
    }
}

/// Default format header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_format_header() -> Option<String> {
    Some("x-praxis-ai-format".to_owned())
}

/// Default model header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_model_header() -> Option<String> {
    Some("x-praxis-ai-model".to_owned())
}

/// Default stream header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_stream_header() -> Option<String> {
    Some("x-praxis-ai-stream".to_owned())
}

/// Default mode header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_mode_header() -> Option<String> {
    Some("x-praxis-responses-mode".to_owned())
}

// -----------------------------------------------------------------------------
// ResponsesFormatConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`ResponsesFormatFilter`].
///
/// [`ResponsesFormatFilter`]: super::ResponsesFormatFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponsesFormatConfig {
    /// Behavior when the body cannot be classified.
    #[serde(default = "OnInvalidBehavior::default_continue")]
    pub on_invalid: OnInvalidBehavior,

    /// Header names for promoted classification facts.
    ///
    /// Must not be hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` names. Dedicated defaults remain allowed.
    #[serde(default)]
    pub headers: ResponsesFormatHeaders,
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: ResponsesFormatConfig) -> Result<ResponsesFormatConfig, FilterError> {
    validate_responses_format_headers(&cfg.headers)?;
    Ok(cfg)
}

/// Validate dedicated names and reject collisions across header fields.
fn validate_responses_format_headers(headers: &ResponsesFormatHeaders) -> Result<(), FilterError> {
    for (field, name, dedicated) in [
        ("format", headers.format.as_deref(), "x-praxis-ai-format"),
        ("model", headers.model.as_deref(), "x-praxis-ai-model"),
        ("stream", headers.stream.as_deref(), "x-praxis-ai-stream"),
        ("mode", headers.mode.as_deref(), "x-praxis-responses-mode"),
    ] {
        crate::promotion::validate_dedicated_promotion_header("openai_responses_format", field, name, &[dedicated])?;
    }
    crate::promotion::reject_duplicate_promotion_fields(
        "openai_responses_format",
        &[
            ("format", headers.format.as_deref()),
            ("model", headers.model.as_deref()),
            ("stream", headers.stream.as_deref()),
            ("mode", headers.mode.as_deref()),
        ],
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -- Serde defaults -------------------------------------------------------

    #[test]
    fn serde_defaults_responses_format_config() {
        let cfg: ResponsesFormatConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(cfg.on_invalid, OnInvalidBehavior::Continue);
    }

    #[test]
    fn responses_format_headers_defaults() {
        let h = ResponsesFormatHeaders::default();
        assert_eq!(h.format.as_deref(), Some("x-praxis-ai-format"));
        assert_eq!(h.model.as_deref(), Some("x-praxis-ai-model"));
        assert_eq!(h.stream.as_deref(), Some("x-praxis-ai-stream"));
        assert_eq!(h.mode.as_deref(), Some("x-praxis-responses-mode"));
    }

    // -- deny_unknown_fields --------------------------------------------------

    #[test]
    fn deny_unknown_fields_responses_format_config() {
        let res = serde_yaml::from_str::<ResponsesFormatConfig>(
            r#"
bogus: true
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn deny_unknown_fields_responses_format_headers() {
        let res = serde_yaml::from_str::<ResponsesFormatHeaders>(
            r#"
format: x-test
extra: true
"#,
        );
        assert!(res.is_err());
    }

    // -- build_config ---------------------------------------------------------

    #[test]
    fn build_config_minimal_ok() {
        let cfg: ResponsesFormatConfig = serde_yaml::from_str("{}").unwrap();
        assert!(build_config(cfg).is_ok());
    }

    #[test]
    fn build_config_invalid_header_name_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: Some("not a valid header!".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("not a valid HTTP header name"),
            "expected invalid header error, got: {err}"
        );
    }

    #[test]
    fn build_config_valid_custom_headers_ok() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: Some("x-custom-format".into()),
                model: Some("x-custom-model".into()),
                stream: Some("x-custom-stream".into()),
                mode: Some("x-custom-mode".into()),
            },
        };
        assert!(build_config(cfg).is_ok());
    }

    #[test]
    fn build_config_authorization_header_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: default_format_header(),
                model: Some("authorization".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("authorization"),
            "authorization promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_api_key_header_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: default_format_header(),
                model: Some("x-api-key".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-api-key"),
            "x-api-key promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_unrelated_internal_header_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: Some("x-praxis-route".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-route"),
            "unrelated x-praxis-* promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_model_header_rejects_format_routing_fact() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: default_format_header(),
                model: Some("x-praxis-ai-format".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-format"),
            "client-derived model must not overwrite format routing: {err}"
        );
    }

    #[test]
    fn build_config_format_header_rejects_model_rewrite_fact() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: Some("x-praxis-ai-effective-model".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-effective-model"),
            "format fact must not overwrite model-rewrite routing: {err}"
        );
    }

    #[test]
    fn build_config_accepts_dedicated_defaults() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders::default(),
        };
        assert!(
            build_config(cfg).is_ok(),
            "dedicated classification defaults should remain allowed"
        );
    }

    #[test]
    fn build_config_rejects_duplicate_promotion_headers() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesFormatHeaders {
                format: Some("x-foo".into()),
                model: Some("X-Foo".into()),
                stream: Some("x-praxis-ai-stream".into()),
                mode: Some("x-praxis-responses-mode".into()),
            },
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("same header name"),
            "duplicate format and model headers should be rejected: {err}"
        );
    }

    // -- null header disables promotion ---------------------------------------

    #[test]
    fn null_header_disables_promotion() {
        let cfg: ResponsesFormatConfig = serde_yaml::from_str(
            r#"
headers:
  format: null
  model: null
  stream: null
  mode: null
"#,
        )
        .unwrap();

        assert!(cfg.headers.format.is_none());
        assert!(cfg.headers.model.is_none());
        assert!(cfg.headers.stream.is_none());
        assert!(cfg.headers.mode.is_none());
        assert!(build_config(cfg).is_ok());
    }
}
