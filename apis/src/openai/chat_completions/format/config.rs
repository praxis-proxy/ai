// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Chat Completions format classifier filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter name, shared by config validation messages.
pub(super) const FILTER_NAME: &str = "openai_chat_completions_format";

/// Default maximum request body size for `StreamBuffer` mode (1 MiB).
///
/// Chat Completions payloads are text-like and do not carry inline file data
/// URLs, so the default matches the Anthropic Messages classifier rather than
/// the larger OpenAI Responses budget. Operators needing larger payloads can
/// override `max_body_bytes`.
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// ChatCompletionsFormatConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`OpenaiChatCompletionsFormatFilter`].
///
/// The filter is a transparent fact producer: it reads the model from the
/// buffered body and never rejects a request, so it carries no `on_invalid`
/// behavior and no promotion headers — only the buffer bound.
///
/// [`OpenaiChatCompletionsFormatFilter`]: super::OpenaiChatCompletionsFormatFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatCompletionsFormatConfig {
    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: ChatCompletionsFormatConfig) -> Result<ChatCompletionsFormatConfig, FilterError> {
    validate_max_body_bytes(FILTER_NAME, cfg.max_body_bytes)?;
    Ok(cfg)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn serde_defaults_chat_completions_format_config() {
        let cfg: ChatCompletionsFormatConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(cfg.max_body_bytes, 1_048_576, "default should be 1 MiB");
    }

    #[test]
    fn default_max_body_bytes_is_1_mib() {
        assert_eq!(DEFAULT_MAX_BODY_BYTES, 1_048_576);
    }

    #[test]
    fn deny_unknown_fields_chat_completions_format_config() {
        let res = serde_yaml::from_str::<ChatCompletionsFormatConfig>("bogus: true");
        assert!(res.is_err(), "unknown fields should be rejected");
    }

    #[test]
    fn build_config_minimal_ok() {
        let cfg: ChatCompletionsFormatConfig = serde_yaml::from_str("{}").unwrap();
        assert!(build_config(cfg).is_ok());
    }

    #[test]
    fn build_config_zero_max_body_bytes_rejected() {
        let cfg = ChatCompletionsFormatConfig { max_body_bytes: 0 };
        let err = build_config(cfg).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "expected 'must be greater than 0' error, got: {err}"
        );
    }

    #[test]
    fn build_config_custom_max_body_bytes_ok() {
        let cfg: ChatCompletionsFormatConfig = serde_yaml::from_str("max_body_bytes: 2048").unwrap();
        assert_eq!(cfg.max_body_bytes, 2048);
        assert!(build_config(cfg).is_ok());
    }
}
