// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for incremental Anthropic-to-OpenAI SSE conversion.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Streaming filter configuration.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OpenaiChatCompletionsToAnthropicStreamConfig {
    /// Maximum incomplete event bytes retained across response chunks.
    pub(crate) max_partial_event_bytes: usize,
    /// Maximum Anthropic tool blocks retained for one response.
    pub(crate) max_tool_blocks: usize,
}

impl Default for OpenaiChatCompletionsToAnthropicStreamConfig {
    fn default() -> Self {
        Self {
            max_partial_event_bytes: 1024 * 1024,
            max_tool_blocks: 256,
        }
    }
}

impl OpenaiChatCompletionsToAnthropicStreamConfig {
    /// Validate finite parser and tool-state limits.
    pub(crate) fn validate(self) -> Result<Self, FilterError> {
        if !(1..=16 * 1024 * 1024).contains(&self.max_partial_event_bytes) {
            return Err(
                "openai_chat_completions_to_anthropic_stream.max_partial_event_bytes must be 1-16777216".into(),
            );
        }
        if !(1..=10000).contains(&self.max_tool_blocks) {
            return Err("openai_chat_completions_to_anthropic_stream.max_tool_blocks must be 1-10000".into());
        }
        Ok(self)
    }
}
