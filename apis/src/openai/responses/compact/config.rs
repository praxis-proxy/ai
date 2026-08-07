// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_responses_compact` filter.

use praxis_filter::FilterError;
use serde::Deserialize;

use crate::openai::responses::config_validation::{self, CalloutSettings, FailureMode};

/// Default callout timeout (30 seconds — summarization can be slow).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default HTTP status when the summarization callout fails in closed mode.
const DEFAULT_STATUS_ON_ERROR: u16 = 502;

// -----------------------------------------------------------------------------
// CompactFilterConfig (YAML deserialization)
// -----------------------------------------------------------------------------

/// Raw YAML config, deserialized then validated.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompactFilterConfig {
    /// URL of the inference backend for summarization calls.
    /// E.g., `"http://localhost:11434/v1/chat/completions"`
    pub inference_url: String,

    /// Default model for summarization when not overridden
    /// in the request's `context_management`.
    #[serde(default = "default_model")]
    pub default_model: String,

    /// Tiktoken encoding name for local token estimation of the
    /// conversation text.
    #[serde(default = "default_tiktoken_encoding")]
    pub tiktoken_encoding: String,

    /// Callout timeout in milliseconds.
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Failure mode for the inference callout.
    #[serde(default)]
    pub callout_failure_mode: Option<FailureMode>,

    /// HTTP status code to return when rejecting on error.
    #[serde(default)]
    pub status_on_error: Option<u16>,
}

/// Default summarization model when not overridden per-request.
fn default_model() -> String {
    "gpt-4o-mini".to_owned()
}

/// Default tiktoken encoding for local token estimation.
fn default_tiktoken_encoding() -> String {
    "cl100k_base".to_owned()
}

// -----------------------------------------------------------------------------
// ValidatedConfig (post-validation)
// -----------------------------------------------------------------------------

/// Validated configuration with defaults applied.
#[derive(Debug)]
pub(super) struct ValidatedConfig {
    /// URL of the inference backend for summarization calls.
    pub inference_url: String,

    /// Default model for summarization.
    pub default_model: String,

    /// Tiktoken encoding name.
    pub tiktoken_encoding: String,

    /// Shared callout settings (timeout, failure mode, status).
    pub callout: CalloutSettings,
}

/// Supported tiktoken encoding names.
const SUPPORTED_ENCODINGS: &[&str] = &["cl100k_base", "o200k_base"];

/// Validate raw config and apply defaults.
///
/// # Errors
///
/// Returns [`FilterError`] if `inference_url` is empty,
/// `tiktoken_encoding` is not a supported encoding name,
/// `timeout_ms` is zero, or `status_on_error` is out of range.
pub(super) fn build_config(raw: &CompactFilterConfig) -> Result<ValidatedConfig, FilterError> {
    if raw.inference_url.is_empty() {
        return Err(FilterError::from("openai_responses_compact: inference_url is empty"));
    }

    if !SUPPORTED_ENCODINGS.contains(&raw.tiktoken_encoding.as_str()) {
        return Err(FilterError::from(format!(
            "openai_responses_compact: unsupported tiktoken_encoding {:?}; supported: {}",
            raw.tiktoken_encoding,
            SUPPORTED_ENCODINGS.join(", ")
        )));
    }

    let timeout_ms =
        config_validation::validate_timeout_ms("openai_responses_compact", raw.timeout_ms, DEFAULT_TIMEOUT_MS)?;

    let status_on_error = config_validation::validate_status_on_error(
        "openai_responses_compact",
        raw.status_on_error,
        DEFAULT_STATUS_ON_ERROR,
    )?;

    Ok(ValidatedConfig {
        inference_url: raw.inference_url.clone(),
        default_model: raw.default_model.clone(),
        tiktoken_encoding: raw.tiktoken_encoding.clone(),
        callout: CalloutSettings {
            timeout_ms,
            failure_mode: raw.callout_failure_mode.unwrap_or(FailureMode::Closed),
            status_on_error,
        },
    })
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod yaml_tests {
    use super::*;

    #[test]
    fn callout_failure_mode_open_deserializes_from_yaml() {
        let cfg: CompactFilterConfig =
            serde_yaml::from_str("inference_url: http://localhost/v1/chat/completions\ncallout_failure_mode: open")
                .expect("should deserialize");
        assert_eq!(cfg.callout_failure_mode, Some(FailureMode::Open));
    }

    #[test]
    fn callout_failure_mode_closed_deserializes_from_yaml() {
        let cfg: CompactFilterConfig =
            serde_yaml::from_str("inference_url: http://localhost/v1/chat/completions\ncallout_failure_mode: closed")
                .expect("should deserialize");
        assert_eq!(cfg.callout_failure_mode, Some(FailureMode::Closed));
    }

    #[test]
    fn callout_failure_mode_absent_defaults_to_none() {
        let cfg: CompactFilterConfig =
            serde_yaml::from_str("inference_url: http://localhost/v1/chat/completions").expect("should deserialize");
        assert_eq!(cfg.callout_failure_mode, None);
    }
}
