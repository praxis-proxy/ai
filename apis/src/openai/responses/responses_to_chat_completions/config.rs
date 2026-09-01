// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for Responses-to-Chat translation.

use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

/// Bounded body configuration for the translation filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesToChatCompletionsConfig {
    /// Maximum assembled request or finite response body size.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

impl Default for ResponsesToChatCompletionsConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: MAX_JSON_BODY_BYTES,
        }
    }
}

/// Return the repository-wide JSON body ceiling used by default.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Validate the parsed filter configuration.
pub(super) fn build_config(
    config: ResponsesToChatCompletionsConfig,
) -> Result<ResponsesToChatCompletionsConfig, FilterError> {
    validate_max_body_bytes("responses_to_chat_completions", config.max_body_bytes)?;
    Ok(config)
}
