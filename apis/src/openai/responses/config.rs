// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Responses format classifier filter.

use praxis_filter::{
    FilterError,
    body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::{
        OnInvalidBehavior,
        config_validation::{validate_header_name, validate_max_body_bytes},
    },
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Behavior Enums
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// ResponsesFormatHeaders
// -----------------------------------------------------------------------------

/// Configurable header names for promoted classification facts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponsesFormatHeaders {
    /// Header name for the detected format (e.g. `openai_responses`, `openai_chat_completions`).
    #[serde(default = "default_format_header")]
    pub format: Option<String>,

    /// Header name for the extracted model value.
    #[serde(default = "default_model_header")]
    pub model: Option<String>,

    /// Header name for the extracted stream flag.
    #[serde(default = "default_stream_header")]
    pub stream: Option<String>,

    /// Header name for the computed mode (`stateless` or `stateful`).
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

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Header names for promoted classification facts.
    #[serde(default)]
    pub headers: ResponsesFormatHeaders,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: ResponsesFormatConfig) -> Result<ResponsesFormatConfig, FilterError> {
    validate_max_body_bytes("openai_responses_format", cfg.max_body_bytes)?;

    validate_header_name("openai_responses_format", "format", cfg.headers.format.as_deref())?;
    validate_header_name("openai_responses_format", "model", cfg.headers.model.as_deref())?;
    validate_header_name("openai_responses_format", "stream", cfg.headers.stream.as_deref())?;
    validate_header_name("openai_responses_format", "mode", cfg.headers.mode.as_deref())?;

    Ok(cfg)
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

        assert_eq!(cfg.max_body_bytes, DEFAULT_JSON_BODY_MAX_BYTES);
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
    fn build_config_zero_max_body_bytes_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            max_body_bytes: 0,
            headers: ResponsesFormatHeaders::default(),
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "expected 'must be greater than 0' error, got: {err}"
        );
    }

    #[test]
    fn build_config_invalid_header_name_rejected() {
        let cfg = ResponsesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
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
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            headers: ResponsesFormatHeaders {
                format: Some("x-custom-format".into()),
                model: Some("x-custom-model".into()),
                stream: Some("x-custom-stream".into()),
                mode: Some("x-custom-mode".into()),
            },
        };
        assert!(build_config(cfg).is_ok());
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
