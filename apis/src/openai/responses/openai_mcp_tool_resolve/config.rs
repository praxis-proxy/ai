// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_tool_resolve` filter.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use crate::openai::responses::body_limits::validate_size_limit;

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
    /// Maximum size in bytes of the request body this filter *produces*
    /// after expanding `mcp` tool entries into `function` entries.
    ///
    /// Raw request body size is governed by the pipeline's `body_limits`,
    /// not this field. This bounds only the post-expansion body, which can
    /// grow larger than the raw input.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,

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

/// Default max rewritten body bytes (64 MiB): a post-expansion backstop.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
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
    validate_size_limit(
        "openai_mcp_tool_resolve",
        "max_rewritten_body_bytes",
        cfg.max_rewritten_body_bytes,
    )?;
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
