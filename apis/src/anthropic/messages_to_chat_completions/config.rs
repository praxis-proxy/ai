// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Anthropic-to-Chat-Completions transformation filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum request body size (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// AnthropicMessagesToChatCompletionsConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`AnthropicMessagesToChatCompletionsFilter`].
///
/// [`AnthropicMessagesToChatCompletionsFilter`]: super::AnthropicMessagesToChatCompletionsFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AnthropicMessagesToChatCompletionsConfig {
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
pub(crate) fn build_config(
    cfg: AnthropicMessagesToChatCompletionsConfig,
) -> Result<AnthropicMessagesToChatCompletionsConfig, FilterError> {
    validate_max_body_bytes("anthropic_messages_to_chat_completions", cfg.max_body_bytes)?;
    Ok(cfg)
}
