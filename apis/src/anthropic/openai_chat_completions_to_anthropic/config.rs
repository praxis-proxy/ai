// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the legacy OpenAI-client to Anthropic-provider bridge.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Maximum accepted request or response body limit.
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Filter configuration as written in `ExtProc` YAML.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OpenaiChatCompletionsToAnthropicConfig {
    /// Permission projected from the authoritative `MaaS` tenant setting.
    /// Anthropic candidates fail closed when this is false.
    pub(crate) api_translation_enabled: bool,
    /// Maximum complete JSON body retained for one request or response.
    pub(crate) max_body_bytes: usize,
}

impl Default for OpenaiChatCompletionsToAnthropicConfig {
    fn default() -> Self {
        Self {
            api_translation_enabled: false,
            max_body_bytes: 1024 * 1024,
        }
    }
}

impl OpenaiChatCompletionsToAnthropicConfig {
    /// Reject zero or unreasonably large per-request body limits.
    pub(crate) fn validate(self) -> Result<Self, FilterError> {
        if !(1..=MAX_BODY_BYTES).contains(&self.max_body_bytes) {
            return Err(
                format!("openai_chat_completions_to_anthropic.max_body_bytes must be 1-{MAX_BODY_BYTES}").into(),
            );
        }
        Ok(self)
    }
}
