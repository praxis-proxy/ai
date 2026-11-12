// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the legacy OpenAI-client to Anthropic-provider bridge.

use praxis_filter::FilterError;
use serde::Deserialize;

use crate::anthropic::{API_VERSION_2023_06_01, API_VERSION_2026_10_21};

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
    /// Operator-selected Anthropic contract: `2023-06-01` (default) or
    /// `2026-10-21`. Independent of caller headers; sampling validation
    /// remains the responsibility of the backend.
    #[serde(default)]
    pub(crate) api_version: AnthropicApiVersion,
    /// Compatibility setting for the former shared request/response limit.
    /// A direction-specific limit takes precedence when set.
    pub(crate) max_body_bytes: Option<usize>,
    /// Maximum complete translated Chat Completions request body.
    pub(crate) max_request_body_bytes: Option<usize>,
    /// Maximum complete Anthropic JSON response body retained for conversion.
    pub(crate) max_response_body_bytes: Option<usize>,
}

/// Supported contracts for requests translated to Anthropic Messages.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(try_from = "String")]
pub(crate) enum AnthropicApiVersion {
    /// Anthropic Messages API version 2023-06-01.
    #[default]
    V20230601,
    /// Anthropic Messages API version 2026-10-21.
    V20261021,
}

impl AnthropicApiVersion {
    /// Version sent to the selected Messages backend.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::V20230601 => API_VERSION_2023_06_01,
            Self::V20261021 => API_VERSION_2026_10_21,
        }
    }
}

impl TryFrom<String> for AnthropicApiVersion {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            API_VERSION_2023_06_01 => Ok(Self::V20230601),
            API_VERSION_2026_10_21 => Ok(Self::V20261021),
            _ => Err(format!(
                "unsupported Anthropic api_version {value:?}; expected {API_VERSION_2023_06_01} or {API_VERSION_2026_10_21}"
            )),
        }
    }
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
    use super::{AnthropicApiVersion, OpenaiChatCompletionsToAnthropicConfig};

    #[test]
    fn version_selection_is_explicit_and_rejects_unknown_contracts() {
        assert_eq!(
            OpenaiChatCompletionsToAnthropicConfig::default().api_version,
            AnthropicApiVersion::V20230601,
            "existing routes must retain the legacy contract"
        );
        for version in ["2023-06-01", "2026-10-21"] {
            let config = parse_config(&format!("api_version: '{version}'"));
            assert_eq!(config.api_version.as_str(), version, "configured version must be used");
        }
        let unknown = serde_yaml::from_str::<OpenaiChatCompletionsToAnthropicConfig>("api_version: '2099-01-01'");
        assert!(unknown.is_err(), "unknown contracts must fail configuration parsing");
    }

    #[test]
    fn defaults_request_and_response_limits_independently() {
        let validation = OpenaiChatCompletionsToAnthropicConfig::default().validate();
        assert!(validation.is_ok(), "default limits must validate: {validation:?}");
        let config = validation.unwrap_or_default();
        assert!(!config.api_translation_enabled, "translation must default to disabled");
        assert_eq!(
            config.request_limit(),
            2 * 1024 * 1024,
            "default request allowance must be 2 MiB"
        );
        assert_eq!(
            config.response_limit(),
            1024 * 1024,
            "default response allowance must remain 1 MiB"
        );
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
        assert_eq!(legacy.request_limit(), 4096, "legacy shared limit must govern requests");
        assert_eq!(
            legacy.response_limit(),
            4096,
            "legacy shared limit must govern responses"
        );

        let split = parse_config("max_body_bytes: 4096\nmax_request_body_bytes: 8192\nmax_response_body_bytes: 2048\n");
        let validation = split.validate();
        assert!(validation.is_ok(), "split limits must validate: {validation:?}");
        let split = validation.unwrap_or_default();
        assert_eq!(
            split.request_limit(),
            8192,
            "request-specific limit must override the shared limit"
        );
        assert_eq!(
            split.response_limit(),
            2048,
            "response-specific limit must override the shared limit"
        );
    }

    #[test]
    fn invalid_independent_limit_is_rejected() {
        let config = parse_config("max_request_body_bytes: 0\n");
        let validation = config.validate();
        assert!(validation.is_err(), "zero request limit must fail validation");
        let error = validation.err().map(|value| value.to_string()).unwrap_or_default();
        assert!(
            error.contains("max_request_body_bytes"),
            "validation must identify the invalid request limit: {error}"
        );
    }
}
