// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for Responses-to-Chat translation.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use crate::openai::responses::body_limits::validate_size_limit;

/// Bounded body configuration for the translation filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesToChatCompletionsConfig {
    /// Maximum size in bytes of the request or finite response body this
    /// filter *produces* when translating between Responses and Chat
    /// Completions wire formats.
    ///
    /// Raw transport body size is governed by the pipeline's `body_limits`,
    /// not this field. This bounds only the translated body, which can grow
    /// larger than the raw input.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,
}

impl Default for ResponsesToChatCompletionsConfig {
    fn default() -> Self {
        Self {
            max_rewritten_body_bytes: MAX_JSON_BODY_BYTES,
        }
    }
}

/// Serde default for `max_rewritten_body_bytes`.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Validate the parsed filter configuration.
pub(super) fn build_config(
    config: ResponsesToChatCompletionsConfig,
) -> Result<ResponsesToChatCompletionsConfig, FilterError> {
    validate_size_limit(
        "responses_to_chat_completions",
        "max_rewritten_body_bytes",
        config.max_rewritten_body_bytes,
    )?;
    Ok(config)
}
