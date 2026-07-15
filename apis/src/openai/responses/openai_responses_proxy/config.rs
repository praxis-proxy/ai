// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Responses proxy filter.

use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// ResponsesProxyConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the Responses proxy filter.
///
/// ```yaml
/// filter: openai_responses_proxy
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesProxyConfig {
    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

impl Default for ResponsesProxyConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: MAX_JSON_BODY_BYTES,
        }
    }
}

/// Serde default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(super) fn build_config(cfg: ResponsesProxyConfig) -> Result<ResponsesProxyConfig, FilterError> {
    validate_max_body_bytes("openai_responses_proxy", cfg.max_body_bytes)?;
    Ok(cfg)
}
