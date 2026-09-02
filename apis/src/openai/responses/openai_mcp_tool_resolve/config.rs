// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_tool_resolve` filter.

use std::collections::HashSet;

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;
use url::Url;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default timeout for MCP `tools/list` calls (5 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Default maximum number of MCP servers per request.
const DEFAULT_MAX_SERVERS: usize = 10;

/// Default maximum number of tools returned by a single MCP server.
const DEFAULT_MAX_TOOLS: usize = 128;

/// Maximum number of connectors allowed in filter configuration.
const MAX_CONNECTORS: usize = 64;
/// Maximum length in bytes for a connector ID.
pub(super) const MAX_CONNECTOR_ID_LEN: usize = 128;
/// Maximum length in bytes for a connector server URL.
const MAX_CONNECTOR_URL_LEN: usize = 2048;

// -----------------------------------------------------------------------------
// ConnectorConfig
// -----------------------------------------------------------------------------

/// Configuration for a named MCP connector.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConnectorConfig {
    /// Connector identifier referenced in requests.
    pub id: String,
    /// MCP server URL for this connector.
    pub server_url: String,
}

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

    /// Named connectors mapping connector IDs to server URLs.
    #[serde(default)]
    pub connectors: Vec<ConnectorConfig>,
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
    validate_connectors(&cfg.connectors)?;
    Ok(cfg)
}

/// Validate connector configuration entries.
fn validate_connectors(connectors: &[ConnectorConfig]) -> Result<(), FilterError> {
    if connectors.len() > MAX_CONNECTORS {
        return Err(format!(
            "openai_mcp_tool_resolve: too many connectors: {} exceeds limit of {MAX_CONNECTORS}",
            connectors.len()
        )
        .into());
    }
    let mut seen_ids = HashSet::new();
    for connector in connectors {
        validate_single_connector(connector)?;
        if !seen_ids.insert(&connector.id) {
            return Err(format!("openai_mcp_tool_resolve: duplicate connector id \"{}\"", connector.id).into());
        }
    }
    Ok(())
}

/// Validate a single connector's ID and URL.
fn validate_single_connector(connector: &ConnectorConfig) -> Result<(), FilterError> {
    validate_connector_id(&connector.id)?;
    validate_connector_server_url(&connector.server_url, &connector.id)
}

/// Validate connector ID is non-empty and within length limit.
fn validate_connector_id(id: &str) -> Result<(), FilterError> {
    if id.is_empty() {
        return Err("openai_mcp_tool_resolve: connector id must not be empty".into());
    }
    if id.len() > MAX_CONNECTOR_ID_LEN {
        return Err(
            format!("openai_mcp_tool_resolve: connector id \"{id}\" exceeds {MAX_CONNECTOR_ID_LEN} bytes").into(),
        );
    }
    Ok(())
}

/// Validate connector server URL string and parse it.
fn validate_connector_server_url(server_url: &str, id: &str) -> Result<(), FilterError> {
    if server_url.is_empty() {
        return Err(format!("openai_mcp_tool_resolve: connector \"{id}\" has empty server_url").into());
    }
    if server_url.len() > MAX_CONNECTOR_URL_LEN {
        return Err(format!(
            "openai_mcp_tool_resolve: connector \"{id}\" server_url exceeds {MAX_CONNECTOR_URL_LEN} bytes"
        )
        .into());
    }
    let url = Url::parse(server_url).map_err(|e| {
        FilterError::from(format!(
            "openai_mcp_tool_resolve: connector \"{id}\" has invalid server_url: {e}"
        ))
    })?;
    validate_connector_url(&url, id)
}

/// Validate a connector URL's scheme, host, and credentials.
fn validate_connector_url(url: &Url, connector_id: &str) -> Result<(), FilterError> {
    match url.scheme() {
        "http" | "https" => {},
        scheme => {
            return Err(format!(
                "openai_mcp_tool_resolve: connector \"{connector_id}\" server_url must use http or https, got \"{scheme}\""
            )
            .into());
        },
    }
    if url.host().is_none() {
        return Err(format!("openai_mcp_tool_resolve: connector \"{connector_id}\" server_url has no host").into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(format!(
            "openai_mcp_tool_resolve: connector \"{connector_id}\" server_url must not contain credentials"
        )
        .into());
    }
    Ok(())
}
