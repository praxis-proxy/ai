// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_dispatch` filter.

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

/// Default timeout for MCP `tools/call` calls (30 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// YAML configuration for the `openai_mcp_dispatch` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpDispatchConfig {
    /// Allow connections to loopback addresses (default: false).
    #[serde(default)]
    pub allow_loopback: bool,

    /// Maximum response body bytes for `StreamBuffer`.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Per-call timeout in milliseconds for `tools/call` calls.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

/// Default value for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Default value for `timeout_ms`.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: McpDispatchConfig) -> Result<McpDispatchConfig, FilterError> {
    validate_max_body_bytes("openai_mcp_dispatch", cfg.max_body_bytes)?;
    if cfg.timeout_ms == 0 {
        return Err("openai_mcp_dispatch: timeout_ms must be greater than 0".into());
    }
    Ok(cfg)
}
