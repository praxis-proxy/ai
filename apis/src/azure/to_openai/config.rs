// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Azure OpenAI translation filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

/// Default maximum request body size (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

/// Default Azure OpenAI API version.
const DEFAULT_API_VERSION: &str = "2024-10-21";

/// YAML configuration for the [`ChatCompletionsToAzureaiChatCompletionsFilter`].
///
/// # YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_azureai_chat_completions
/// api_version: "2024-10-21"
/// ```
///
/// [`ChatCompletionsToAzureaiChatCompletionsFilter`]: super::ChatCompletionsToAzureaiChatCompletionsFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChatCompletionsToAzureaiChatCompletionsConfig {
    /// Azure API version query parameter injected on every upstream request.
    #[serde(default = "default_api_version")]
    pub api_version: String,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

/// Serde default for [`ChatCompletionsToAzureaiChatCompletionsConfig::api_version`].
fn default_api_version() -> String {
    DEFAULT_API_VERSION.to_owned()
}

/// Serde default for [`ChatCompletionsToAzureaiChatCompletionsConfig::max_body_bytes`].
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Validate the parsed configuration.
pub(crate) fn build_config(
    cfg: ChatCompletionsToAzureaiChatCompletionsConfig,
) -> Result<ChatCompletionsToAzureaiChatCompletionsConfig, FilterError> {
    validate_max_body_bytes(
        "openai_chat_completions_to_azureai_chat_completions",
        cfg.max_body_bytes,
    )?;
    if cfg.api_version.is_empty() {
        return Err(FilterError::from(
            "openai_chat_completions_to_azureai_chat_completions: api_version must not be empty",
        ));
    }
    Ok(cfg)
}
