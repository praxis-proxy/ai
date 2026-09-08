// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for Responses-to-Chat translation.

use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

use crate::openai::translation::reasoning::{ReasoningDialect, ReasoningOptions};

/// Bounded body configuration for the translation filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesToChatCompletionsConfig {
    /// Maximum assembled request or finite response body size.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Backend-specific dialect behavior.
    #[serde(default)]
    pub reasoning: ReasoningOptions,
}

impl Default for ResponsesToChatCompletionsConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: MAX_JSON_BODY_BYTES,
            reasoning: ReasoningOptions::default(),
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
    if config.reasoning.dialect != ReasoningDialect::None {
        validate_max_reasoning_bytes(config.reasoning.max_reasoning_bytes, config.max_body_bytes)?;
    }
    Ok(config)
}

/// Validate reasoning byte limit for a dialect that extracts raw reasoning.
/// It must be non-zero and within the max body bytes ceiling.
fn validate_max_reasoning_bytes(value: usize, max_body_bytes: usize) -> Result<(), FilterError> {
    if value == 0 {
        return Err("responses_to_chat_completions: reasoning.max_reasoning_bytes must be greater than 0".into());
    }
    if value > max_body_bytes {
        return Err(format!(
            "responses_to_chat_completions: reasoning.max_reasoning_bytes ({value}) must not exceed max_body_bytes ({max_body_bytes})"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn reasoning_bytes_zero_rejected() {
        let err = validate_max_reasoning_bytes(0, 1024).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "zero should be rejected, got: {err}"
        );
    }

    #[test]
    fn reasoning_bytes_over_body_limit_rejected() {
        let err = validate_max_reasoning_bytes(2048, 1024).unwrap_err();
        assert!(
            err.to_string().contains("must not exceed max_body_bytes"),
            "value above the body limit should be rejected, got: {err}"
        );
    }

    #[test]
    fn reasoning_bytes_within_limit_accepted() {
        validate_max_reasoning_bytes(512, 1024).expect("value below the body limit should be accepted");
    }

    #[test]
    fn reasoning_bytes_equal_to_body_limit_accepted() {
        validate_max_reasoning_bytes(1024, 1024).expect("value equal to the body limit should be accepted");
    }
}
