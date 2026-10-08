// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Chat Completions to Bedrock Converse translation filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter name used in YAML configs and the filter registry.
pub(super) const FILTER_NAME: &str = "openai_chat_completions_to_bedrock_converse";

/// Default maximum request/response body size (4 MiB).
///
/// Bedrock Converse has a 3.75 MiB payload limit; 4 MiB gives headroom
/// for the JSON envelope expansion that happens during translation.
const DEFAULT_MAX_BODY_BYTES: usize = 4 * 1_048_576; // 4 MiB

// -----------------------------------------------------------------------------
// BedrockConverseConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`OpenaiChatCompletionsToBedrockConverseFilter`].
///
/// # Required fields
///
/// None — the `model` is extracted from the request body at runtime.
///
/// # Optional fields
///
/// - `max_body_bytes`: maximum body size for `StreamBuffer` mode (default 4 MiB).
///
/// [`OpenaiChatCompletionsToBedrockConverseFilter`]: super::OpenaiChatCompletionsToBedrockConverseFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BedrockConverseConfig {
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

/// Validate the parsed configuration and return the sanitized config.
pub(crate) fn build_config(cfg: BedrockConverseConfig) -> Result<BedrockConverseConfig, FilterError> {
    validate_max_body_bytes(FILTER_NAME, cfg.max_body_bytes)?;

    Ok(cfg)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use praxis_filter::parse_filter_config;

    use super::*;

    #[test]
    fn defaults_applied_when_no_fields() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        let cfg: BedrockConverseConfig = parse_filter_config(FILTER_NAME, &yaml).unwrap();
        let validated = build_config(cfg).unwrap();

        assert_eq!(validated.max_body_bytes, 4 * 1_048_576);
    }

    #[test]
    fn custom_body_bytes() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 2097152").unwrap();
        let cfg: BedrockConverseConfig = parse_filter_config(FILTER_NAME, &yaml).unwrap();
        let validated = build_config(cfg).unwrap();

        assert_eq!(validated.max_body_bytes, 2_097_152);
    }

    #[test]
    fn unknown_fields_rejected() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("bogus: true").unwrap();
        let result = parse_filter_config::<BedrockConverseConfig>(FILTER_NAME, &yaml);

        assert!(result.is_err(), "unknown fields must be rejected");
    }
}
