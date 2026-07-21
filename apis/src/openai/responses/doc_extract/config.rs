// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the `openai_doc_extract` filter.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

/// Default maximum number of `input_file` parts extracted per request.
const DEFAULT_MAX_FILE_REFERENCES: usize = 32;

/// Maximum configurable `input_file` count per request.
const MAX_CONFIGURABLE_FILE_REFERENCES: usize = 128;

/// Default per-file decoded content limit (10 MiB), matching the
/// OpenAI `InputTextContentParam.text` schema maximum.
const DEFAULT_MAX_CONTENT_BYTES: usize = 10_485_760;

/// Schema ceiling for `input_text.text` (10 MiB).
const MAX_INPUT_TEXT_BYTES: usize = 10_485_760;

/// Default total extracted text limit across all files (64 MiB).
const DEFAULT_MAX_TOTAL_TEXT_BYTES: usize = 67_108_864;

/// Behavior when a file's MIME type is not text-safe.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnUnsupported {
    /// Leave the `input_file` part unchanged and continue.
    #[default]
    Continue,

    /// Return an error response to the client.
    Reject,
}

/// YAML configuration for the [`DocExtractFilter`].
///
/// [`DocExtractFilter`]: super::DocExtractFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DocExtractConfig {
    /// Acknowledge that `StreamBuffer` body processing runs before
    /// header-phase security filters.  Must be `true`.
    #[serde(default)]
    pub allow_pre_security_callout: bool,

    /// Maximum request body bytes for `StreamBuffer` (default 64 MiB).
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Maximum decoded content bytes per file (default 10 MiB).
    #[serde(default = "default_max_content_bytes")]
    pub max_content_bytes: usize,

    /// Maximum `input_file` parts processed per request (default 32).
    #[serde(default = "default_max_file_references")]
    pub max_file_references: usize,

    /// Maximum total extracted text bytes across all files in one
    /// request (default 64 MiB).
    #[serde(default = "default_max_total_text_bytes")]
    pub max_total_text_bytes: usize,

    /// Behavior when a file's MIME type is not text-safe.
    #[serde(default)]
    pub on_unsupported: OnUnsupported,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Default max content bytes per file.
fn default_max_content_bytes() -> usize {
    DEFAULT_MAX_CONTENT_BYTES
}

/// Default maximum file reference count.
fn default_max_file_references() -> usize {
    DEFAULT_MAX_FILE_REFERENCES
}

/// Default total extracted text byte limit.
fn default_max_total_text_bytes() -> usize {
    DEFAULT_MAX_TOTAL_TEXT_BYTES
}

/// Validate and normalize a deserialized [`DocExtractConfig`].
pub(crate) fn validate_config(cfg: DocExtractConfig) -> Result<DocExtractConfig, FilterError> {
    validate_pre_security_callout(&cfg)?;
    validate_limits(&cfg)?;
    Ok(cfg)
}

/// Require explicit acknowledgement of the pre-read security boundary.
fn validate_pre_security_callout(cfg: &DocExtractConfig) -> Result<(), FilterError> {
    if !cfg.allow_pre_security_callout {
        return Err(
            "openai_doc_extract: 'allow_pre_security_callout' must be true because StreamBuffer body processing runs before header-phase security filters; place authentication and authorization in an outer trust boundary"
                .into(),
        );
    }
    Ok(())
}

/// Validate numeric limits applied while buffering and extracting.
fn validate_limits(cfg: &DocExtractConfig) -> Result<(), FilterError> {
    if cfg.max_body_bytes == 0 {
        return Err("openai_doc_extract: 'max_body_bytes' must be greater than 0".into());
    }
    if cfg.max_body_bytes > MAX_JSON_BODY_BYTES {
        return Err(format!(
            "openai_doc_extract: 'max_body_bytes' ({}) exceeds maximum ({MAX_JSON_BODY_BYTES})",
            cfg.max_body_bytes
        )
        .into());
    }

    if cfg.max_content_bytes == 0 {
        return Err("openai_doc_extract: 'max_content_bytes' must be greater than 0".into());
    }
    if cfg.max_content_bytes > MAX_INPUT_TEXT_BYTES {
        return Err(format!(
            "openai_doc_extract: 'max_content_bytes' ({}) exceeds input_text schema maximum ({MAX_INPUT_TEXT_BYTES})",
            cfg.max_content_bytes
        )
        .into());
    }

    validate_extraction_limits(cfg)
}

/// Validate file-count and total-text limits.
fn validate_extraction_limits(cfg: &DocExtractConfig) -> Result<(), FilterError> {
    if cfg.max_file_references == 0 {
        return Err("openai_doc_extract: 'max_file_references' must be greater than 0".into());
    }
    if cfg.max_file_references > MAX_CONFIGURABLE_FILE_REFERENCES {
        return Err(format!(
            "openai_doc_extract: 'max_file_references' ({}) exceeds maximum ({MAX_CONFIGURABLE_FILE_REFERENCES})",
            cfg.max_file_references
        )
        .into());
    }

    if cfg.max_total_text_bytes == 0 {
        return Err("openai_doc_extract: 'max_total_text_bytes' must be greater than 0".into());
    }
    if cfg.max_total_text_bytes > MAX_JSON_BODY_BYTES {
        return Err(format!(
            "openai_doc_extract: 'max_total_text_bytes' ({}) exceeds maximum ({MAX_JSON_BODY_BYTES})",
            cfg.max_total_text_bytes
        )
        .into());
    }

    Ok(())
}

/// Return whether a MIME type is text-safe and can be decoded to UTF-8
/// for use as `input_text`.
pub(crate) fn is_text_safe_mime(mime: &str) -> bool {
    let lower = mime.to_ascii_lowercase();
    let base = lower.split(';').next().unwrap_or("").trim();
    base.starts_with("text/") || base == "application/json" || base == "application/xml"
}
