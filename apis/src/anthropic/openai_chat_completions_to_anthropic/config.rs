// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the legacy OpenAI-client to Anthropic-provider bridge.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Maximum accepted request or response body limit.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Default maximum translated Chat Completions request size.
const DEFAULT_REQUEST_BODY_BYTES: usize = 2 * 1024 * 1024;
/// Default maximum buffered Anthropic JSON response size.
const DEFAULT_RESPONSE_BODY_BYTES: usize = 1024 * 1024;

/// Filter configuration.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OpenaiChatCompletionsToAnthropicConfig {
    /// Permission from trusted application configuration. Anthropic requests
    /// fail closed when this is false (the default).
    #[serde(default)]
    pub(crate) api_translation_enabled: bool,
    /// Compatibility setting for the former shared request/response limit.
    /// A direction-specific limit takes precedence when set.
    pub(crate) max_body_bytes: Option<usize>,
    /// Maximum complete translated Chat Completions request body.
    pub(crate) max_request_body_bytes: Option<usize>,
    /// Maximum complete Anthropic JSON response body retained for conversion.
    pub(crate) max_response_body_bytes: Option<usize>,
}

impl OpenaiChatCompletionsToAnthropicConfig {
    /// Effective complete Chat Completions request limit.
    pub(crate) fn request_limit(&self) -> usize {
        self.max_request_body_bytes
            .or(self.max_body_bytes)
            .unwrap_or(DEFAULT_REQUEST_BODY_BYTES)
    }

    /// Effective complete Anthropic JSON response limit.
    pub(crate) fn response_limit(&self) -> usize {
        self.max_response_body_bytes
            .or(self.max_body_bytes)
            .unwrap_or(DEFAULT_RESPONSE_BODY_BYTES)
    }

    /// Reject zero or unreasonably large per-direction body limits.
    pub(crate) fn validate(self) -> Result<Self, FilterError> {
        for (name, limit) in [
            ("max_body_bytes", self.max_body_bytes),
            ("max_request_body_bytes", self.max_request_body_bytes),
            ("max_response_body_bytes", self.max_response_body_bytes),
        ] {
            if limit.is_some_and(|value| !(1..=MAX_BODY_BYTES).contains(&value)) {
                return Err(format!("openai_chat_completions_to_anthropic.{name} must be 1-{MAX_BODY_BYTES}").into());
            }
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::OpenaiChatCompletionsToAnthropicConfig;

    #[test]
    fn defaults_request_and_response_limits_independently() {
        let validation = OpenaiChatCompletionsToAnthropicConfig::default().validate();
        assert!(validation.is_ok(), "default limits must validate: {validation:?}");
        let config = validation.unwrap_or_default();
        assert!(!config.api_translation_enabled, "translation must default to disabled");
        assert_eq!(config.request_limit(), 2 * 1024 * 1024);
        assert_eq!(config.response_limit(), 1024 * 1024);
    }

    /// Deserialize a known-valid test configuration without panicking on parse errors.
    fn parse_config(yaml: &str) -> OpenaiChatCompletionsToAnthropicConfig {
        let parsed = serde_yaml::from_str(yaml);
        assert!(parsed.is_ok(), "test configuration must parse: {parsed:?}");
        parsed.unwrap_or_default()
    }

    #[test]
    fn legacy_shared_limit_applies_to_both_unless_overridden() {
        let legacy = parse_config("max_body_bytes: 4096");
        let validation = legacy.validate();
        assert!(validation.is_ok(), "legacy limit must validate: {validation:?}");
        let legacy = validation.unwrap_or_default();
        assert_eq!(legacy.request_limit(), 4096);
        assert_eq!(legacy.response_limit(), 4096);

        let split = parse_config("max_body_bytes: 4096\nmax_request_body_bytes: 8192\nmax_response_body_bytes: 2048\n");
        let validation = split.validate();
        assert!(validation.is_ok(), "split limits must validate: {validation:?}");
        let split = validation.unwrap_or_default();
        assert_eq!(split.request_limit(), 8192);
        assert_eq!(split.response_limit(), 2048);
    }

    #[test]
    fn invalid_independent_limit_is_rejected() {
        let config = parse_config("max_request_body_bytes: 0\n");
        let validation = config.validate();
        assert!(validation.is_err(), "zero request limit must fail validation");
        let error = validation.err().map(|value| value.to_string()).unwrap_or_default();
        assert!(error.contains("max_request_body_bytes"));
    }
}
