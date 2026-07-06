// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Anthropic Messages format classifier filter.

use praxis_filter::{
    FilterError,
    builtins::http::payload_processing::{
        OnInvalidBehavior,
        config_validation::{validate_header_name, validate_max_body_bytes},
    },
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum request body size for `StreamBuffer` mode (1 MiB).
///
/// Smaller than the OpenAI Responses default (10 MiB) because Anthropic
/// Messages API payloads are typically text-only and do not carry inline
/// file data URLs.  Operators needing larger payloads can override via
/// `max_body_bytes` in config.
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// Behavior Enums
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// AnthropicMessagesFormatHeaders
// -----------------------------------------------------------------------------

/// Configurable header names for promoted classification facts.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicMessagesFormatHeaders {
    /// Header name for the detected format.
    #[serde(default = "default_format_header")]
    pub format: Option<String>,

    /// Header name for the extracted model value.
    #[serde(default = "default_model_header")]
    pub model: Option<String>,

    /// Header name for the extracted stream flag.
    #[serde(default = "default_stream_header")]
    pub stream: Option<String>,
}

impl Default for AnthropicMessagesFormatHeaders {
    fn default() -> Self {
        Self {
            format: default_format_header(),
            model: default_model_header(),
            stream: default_stream_header(),
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

// -----------------------------------------------------------------------------
// AnthropicMessagesFormatConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`AnthropicMessagesFormatFilter`].
///
/// [`AnthropicMessagesFormatFilter`]: super::AnthropicMessagesFormatFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicMessagesFormatConfig {
    /// Behavior when the body cannot be classified.
    #[serde(default = "OnInvalidBehavior::default_continue")]
    pub on_invalid: OnInvalidBehavior,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Header names for promoted classification facts.
    #[serde(default)]
    pub headers: AnthropicMessagesFormatHeaders,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: AnthropicMessagesFormatConfig) -> Result<AnthropicMessagesFormatConfig, FilterError> {
    validate_max_body_bytes("anthropic_messages_format", cfg.max_body_bytes)?;

    validate_header_name("anthropic_messages_format", "format", cfg.headers.format.as_deref())?;
    validate_header_name("anthropic_messages_format", "model", cfg.headers.model.as_deref())?;
    validate_header_name("anthropic_messages_format", "stream", cfg.headers.stream.as_deref())?;

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
    fn serde_defaults_anthropic_messages_format_config() {
        let cfg: AnthropicMessagesFormatConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(cfg.max_body_bytes, 1_048_576, "default should be 1 MiB");
        assert_eq!(cfg.on_invalid, OnInvalidBehavior::Continue);
    }

    #[test]
    fn default_max_body_bytes_is_1_mib() {
        assert_eq!(DEFAULT_MAX_BODY_BYTES, 1_048_576);
    }

    #[test]
    fn anthropic_messages_format_headers_defaults() {
        let h = AnthropicMessagesFormatHeaders::default();
        assert_eq!(h.format.as_deref(), Some("x-praxis-ai-format"));
        assert_eq!(h.model.as_deref(), Some("x-praxis-ai-model"));
        assert_eq!(h.stream.as_deref(), Some("x-praxis-ai-stream"));
    }

    // -- deny_unknown_fields --------------------------------------------------

    #[test]
    fn deny_unknown_fields_anthropic_messages_format_config() {
        let res = serde_yaml::from_str::<AnthropicMessagesFormatConfig>(
            r#"
bogus: true
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn deny_unknown_fields_anthropic_messages_format_headers() {
        let res = serde_yaml::from_str::<AnthropicMessagesFormatHeaders>(
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
        let cfg: AnthropicMessagesFormatConfig = serde_yaml::from_str("{}").unwrap();
        assert!(build_config(cfg).is_ok());
    }

    #[test]
    fn build_config_zero_max_body_bytes_rejected() {
        let cfg = AnthropicMessagesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            max_body_bytes: 0,
            headers: AnthropicMessagesFormatHeaders::default(),
        };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "expected 'must be greater than 0' error, got: {err}"
        );
    }

    #[test]
    fn build_config_invalid_header_name_rejected() {
        let cfg = AnthropicMessagesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            headers: AnthropicMessagesFormatHeaders {
                format: Some("not a valid header!".into()),
                model: default_model_header(),
                stream: default_stream_header(),
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
        let cfg = AnthropicMessagesFormatConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            headers: AnthropicMessagesFormatHeaders {
                format: Some("x-custom-format".into()),
                model: Some("x-custom-model".into()),
                stream: Some("x-custom-stream".into()),
            },
        };
        assert!(build_config(cfg).is_ok());
    }

    // -- null header disables promotion ---------------------------------------

    #[test]
    fn null_header_disables_promotion() {
        let cfg: AnthropicMessagesFormatConfig = serde_yaml::from_str(
            r#"
headers:
  format: null
  model: null
  stream: null
"#,
        )
        .unwrap();

        assert!(cfg.headers.format.is_none());
        assert!(cfg.headers.model.is_none());
        assert!(cfg.headers.stream.is_none());
        assert!(build_config(cfg).is_ok());
    }
}
