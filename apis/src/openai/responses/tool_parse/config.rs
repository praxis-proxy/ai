// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the tool parse filter.

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// ToolParseConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`ToolParseFilter`].
///
/// [`ToolParseFilter`]: super::ToolParseFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolParseConfig {
    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: ToolParseConfig) -> Result<ToolParseConfig, FilterError> {
    validate_max_body_bytes("tool_parse", cfg.max_body_bytes)?;
    Ok(cfg)
}
