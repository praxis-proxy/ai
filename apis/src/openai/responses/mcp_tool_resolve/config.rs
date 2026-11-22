// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_tool_resolve` filter.

use praxis_core::config::ChainRef;
use praxis_filter::FilterError;
use serde::Deserialize;

use crate::{
    json_body::MAX_JSON_BODY_BYTES,
    mcp_core::connector::{ConnectorConfig, ToolAdmissionConfig, validate_connectors},
    openai::responses::body_limits::validate_size_limit,
};

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
    /// Optional per-user bearer slot from `callout_credentials`. Must match the
    /// corresponding `openai_mcp_dispatch` setting.
    #[serde(default)]
    pub user_credential: Option<String>,

    /// Optional opaque assertion slot from `callout_credentials.assertions`. Must match
    /// the corresponding `openai_mcp_dispatch` setting.
    #[serde(default)]
    pub authorization_assertion: Option<String>,

    /// Require MCP tool entries to reference an operator-configured connector.
    ///
    /// When false, client-selected direct `server_url` entries remain allowed for
    /// deployments that intentionally mix public direct servers with scoped
    /// connector traffic.
    #[serde(default)]
    pub require_connector: bool,

    /// Trusted request headers forwarded to connector-backed MCP `initialize`
    /// and `tools/list` requests.
    /// No request headers are forwarded by default. Credential headers such as
    /// `authorization` are rejected because MCP destinations are client-selected;
    /// use the MCP tool entry's dedicated `authorization` field instead. Direct,
    /// client-selected `server_url` targets never receive ambient request headers.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Maximum size in bytes of the request body this filter *produces*
    /// after expanding `mcp` tool entries into `function` entries.
    ///
    /// Raw request body size is governed by the pipeline's `body_limits`,
    /// not this field. This bounds only the post-expansion body, which can
    /// grow larger than the raw input.
    #[serde(default = "default_max_rewritten_body_bytes")]
    pub max_rewritten_body_bytes: usize,

    /// Per-server timeout in milliseconds for `tools/list` calls. Inside an
    /// iterative request router, initialize and listing exchanges are capped by
    /// the router's remaining deadline.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Maximum number of distinct MCP servers per request.
    #[serde(default = "default_max_servers")]
    pub max_servers: usize,

    /// Maximum number of tools returned by a single MCP server.
    #[serde(default = "default_max_tools")]
    pub max_tools: usize,

    /// Outbound filter chain the MCP `tools/list` callout runs through.
    ///
    /// The chain is bound at build time and carries only operator-configured
    /// cross-cutting filters, which observe and can act on the outbound MCP
    /// request. The SSRF-validated dial target is staged by the transport, so no
    /// upstream-selecting filter is prepended. Both an inline chain and a named
    /// reference (resolved against the top-level `filter_chains`) are accepted,
    /// because this filter binds at top level. When omitted, the callout runs
    /// through an empty chain and dials the staged target directly.
    ///
    /// Whether loopback/private MCP destinations are permitted is governed by
    /// the operator's global insecure posture (which pipeline finalization
    /// applies to this bound chain), not a per-filter flag.
    #[serde(default)]
    pub outbound_chain: Option<ChainRef>,

    /// Named connectors mapping connector IDs to server URLs.
    #[serde(default)]
    pub connectors: Vec<ConnectorConfig>,

    /// Operator metadata-admission policy. Defaults to advisory (unchanged
    /// behavior), so existing non-protected deployments are not affected.
    #[serde(default)]
    pub tool_admission: ToolAdmissionConfig,
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
pub(crate) fn build_config(mut cfg: McpToolResolveConfig) -> Result<McpToolResolveConfig, FilterError> {
    crate::openai::api_client::validate_forward_headers("openai_mcp_tool_resolve", &mut cfg.forward_headers)?;
    reject_mcp_sensitive_forward_headers(&cfg.forward_headers)?;
    validate_context_slot("user_credential", cfg.user_credential.as_deref())?;
    validate_context_slot("authorization_assertion", cfg.authorization_assertion.as_deref())?;
    if !cfg.require_connector && (cfg.user_credential.is_some() || cfg.authorization_assertion.is_some()) {
        tracing::warn!(
            user_credential = ?cfg.user_credential,
            authorization_assertion = ?cfg.authorization_assertion,
            "openai_mcp_tool_resolve: per-user MCP authorization is configured without require_connector; direct server_url entries remain allowed"
        );
    }
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
    validate_connectors("openai_mcp_tool_resolve", &cfg.connectors)?;
    Ok(cfg)
}

/// Validate one optional request-scoped context slot name.
fn validate_context_slot(field: &str, slot: Option<&str>) -> Result<(), FilterError> {
    if let Some(slot) = slot
        && (slot.is_empty() || slot.len() > 128)
    {
        return Err(format!("openai_mcp_tool_resolve: {field} must be 1..=128 bytes").into());
    }
    Ok(())
}

/// Reject ambient credentials and protocol-controlled fields at MCP's
/// client-selected destination boundary.
fn reject_mcp_sensitive_forward_headers(headers: &[String]) -> Result<(), FilterError> {
    for configured in headers {
        let name = http::HeaderName::from_bytes(configured.as_bytes()).map_err(|error| -> FilterError {
            format!("openai_mcp_tool_resolve: invalid forward header: {error}").into()
        })?;
        if crate::mcp_client::is_blocked_mcp_header(&name) {
            return Err(format!(
                "openai_mcp_tool_resolve: 'forward_headers' must not include credential or MCP-controlled header '{name}'"
            )
            .into());
        }
    }
    Ok(())
}
