// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_responses_compact` filter.

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default callout timeout (30 seconds — summarization can be slow).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default HTTP status when the summarization callout fails in closed mode.
const DEFAULT_STATUS_ON_ERROR: u16 = 502;

// -----------------------------------------------------------------------------
// FailureMode
// -----------------------------------------------------------------------------

/// What happens when the summarization callout fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FailureMode {
    /// Reject the request on summarization failure (default).
    Closed,
    /// Continue with full (uncompacted) history on failure.
    Open,
}

// -----------------------------------------------------------------------------
// CompactFilterConfig (YAML deserialization)
// -----------------------------------------------------------------------------

#[expect(
    dead_code,
    reason = "scaffolding — fields used once build_config is implemented"
)]
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

    /// Tiktoken encoding name for local token estimation.
    /// Used as fallback when `previous_usage` is unavailable.
    #[serde(default = "default_tiktoken_encoding")]
    pub tiktoken_encoding: String,

    /// Callout timeout in milliseconds.
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Failure mode for the inference callout.
    #[serde(default)]
    pub failure_mode: Option<FailureMode>,

    /// HTTP status code to return when rejecting on error.
    #[serde(default)]
    pub status_on_error: Option<u16>,
}

fn default_model() -> String {
    "gpt-4o-mini".to_owned()
}

fn default_tiktoken_encoding() -> String {
    "cl100k_base".to_owned()
}

// -----------------------------------------------------------------------------
// ValidatedConfig (post-validation)
// -----------------------------------------------------------------------------

#[expect(
    dead_code,
    reason = "scaffolding — fields used once from_config and on_request_body are implemented"
)]
/// Validated configuration with defaults applied.
pub(super) struct ValidatedConfig {
    /// URL of the inference backend for summarization calls.
    pub inference_url: String,

    /// Default model for summarization.
    pub default_model: String,

    /// Tiktoken encoding name.
    pub tiktoken_encoding: String,

    /// Callout timeout in milliseconds.
    pub timeout_ms: u64,

    /// Failure mode for the inference callout.
    pub failure_mode: FailureMode,

    /// HTTP status on error.
    pub status_on_error: u16,
}

/// Validate raw config and apply defaults.
///
/// # Errors
///
/// Returns [`FilterError`] if `inference_url` is empty,
/// `timeout_ms` is zero, or `status_on_error` is out of range.
///
/// Follow the pattern in `web_search/config.rs:build_config()`:
///
/// - Validate `inference_url` is not empty
/// - Validate `timeout_ms` (default [`DEFAULT_TIMEOUT_MS`], reject 0)
/// - Validate `status_on_error` (default [`DEFAULT_STATUS_ON_ERROR`],
///   must be 100..=599)
/// - Validate `tiktoken_encoding` is a known encoding name
///   (use `tiktoken_rs` to check)
/// - Return `ValidatedConfig` with defaults applied
#[expect(
    clippy::todo,
    reason = "scaffolding — implement config validation"
)]
pub(super) fn build_config(raw: &CompactFilterConfig) -> Result<ValidatedConfig, FilterError> {
    let _ = (raw, DEFAULT_TIMEOUT_MS, DEFAULT_STATUS_ON_ERROR);
    todo!("implement config validation")
}
