// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_tool_resolve` filter.

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default timeout for MCP `tools/list` calls (5 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Default maximum number of MCP servers per request.
const DEFAULT_MAX_SERVERS: usize = 10;

/// Default maximum number of tools returned by a single MCP server.
const DEFAULT_MAX_TOOLS: usize = 128;

// -----------------------------------------------------------------------------
// McpToolResolveConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the `openai_mcp_tool_resolve` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpToolResolveConfig {
    /// Maximum request body bytes for `StreamBuffer`.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Per-server timeout in milliseconds for `tools/list` calls.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Maximum number of distinct MCP servers per request.
    #[serde(default = "default_max_servers")]
    pub max_servers: usize,

    /// Maximum number of tools returned by a single MCP server.
    #[serde(default = "default_max_tools")]
    pub max_tools: usize,

    /// Allow connections to loopback addresses (`127.0.0.0/8`,
    /// `::1`, `localhost`). Disabled by default for SSRF
    /// protection; enable for development environments where MCP
    /// servers run locally.
    #[serde(default)]
    pub allow_loopback: bool,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

/// Default timeout in milliseconds.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Default max MCP servers.
fn default_max_servers() -> usize {
    DEFAULT_MAX_SERVERS
}

/// Default max tools per server.
fn default_max_tools() -> usize {
    DEFAULT_MAX_TOOLS
}

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: McpToolResolveConfig) -> Result<McpToolResolveConfig, FilterError> {
    validate_max_body_bytes("openai_mcp_tool_resolve", cfg.max_body_bytes)?;
    if cfg.timeout_ms == 0 {
        return Err("openai_mcp_tool_resolve: timeout_ms must be greater than 0".into());
    }
    if cfg.max_servers == 0 {
        return Err("openai_mcp_tool_resolve: max_servers must be greater than 0".into());
    }
    if cfg.max_tools == 0 {
        return Err("openai_mcp_tool_resolve: max_tools must be greater than 0".into());
    }
    Ok(cfg)
}
