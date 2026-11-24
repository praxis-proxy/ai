// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `anthropic_mcp_connector` filter.

use std::{collections::HashMap, time::Duration};

use praxis_core::config::ChainRef;
use praxis_filter::FilterError;
use serde::Deserialize;

use crate::{
    json_body::MAX_JSON_BODY_BYTES,
    mcp_core::{
        admission::{ConnectorToolPolicy, McpAdmissionPolicy},
        connector::{ConnectorConfig, ToolAdmissionConfig, validate_connectors},
        execute::MIN_RETAINED_RESULT_BYTES,
    },
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Registry name and filter-results namespace.
pub(super) const FILTER_NAME: &str = "anthropic_mcp_connector";

/// Default end-to-end timeout for one MCP `tools/list` or `tools/call`.
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default maximum number of MCP servers one request may declare.
const DEFAULT_MAX_SERVERS: usize = 10;

/// Default maximum number of tools accepted from one MCP server.
const DEFAULT_MAX_TOOLS: usize = 128;

/// Default maximum number of MCP calls running at once.
const DEFAULT_MAX_PARALLEL_CALLS: usize = 8;

/// Upper bound for `max_parallel_calls`.
const MAX_PARALLEL_CALLS: usize = 64;

/// Default maximum bytes retained for one MCP result.
const DEFAULT_MAX_RESULT_BYTES: usize = 1_048_576;

/// Upper bound for `max_result_bytes`.
const MAX_RESULT_BYTES: usize = 16_777_216;

/// Default maximum bytes retained for all results of one model round.
const DEFAULT_MAX_TOTAL_RESULT_BYTES: usize = 8_388_608;

/// Upper bound for `max_total_result_bytes`.
const MAX_TOTAL_RESULT_BYTES: usize = 67_108_864;

/// Longest accepted per-user context slot id.
const MAX_SLOT_ID_LEN: usize = 128;

// -----------------------------------------------------------------------------
// Raw Configuration
// -----------------------------------------------------------------------------

/// Raw YAML configuration for `anthropic_mcp_connector`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AnthropicMcpConnectorConfig {
    /// Operator-configured MCP servers. A client `mcp_servers` entry whose
    /// `url` equals a connector's `server_url` is executed through that
    /// connector, with operator identity and tool policy.
    #[serde(default)]
    pub connectors: Vec<ConnectorConfig>,

    /// Whether client `mcp_servers` URLs that match no connector may be dialed
    /// directly. Off by default: unmatched servers are rejected with a 400.
    #[serde(default)]
    pub allow_client_servers: bool,

    /// Operator metadata admission for MCP tools.
    #[serde(default)]
    pub tool_admission: ToolAdmissionConfig,

    /// Trusted request headers forwarded to configured connectors only.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Callout-credential slot carrying the per-user bearer for connectors.
    #[serde(default)]
    pub user_credential: Option<String>,

    /// Callout-credential slot carrying an opaque authorization assertion.
    #[serde(default)]
    pub authorization_assertion: Option<String>,

    /// Outbound filter chain every MCP callout runs through.
    #[serde(default)]
    pub outbound_chain: Option<ChainRef>,

    /// Timeout for one MCP `tools/list` or `tools/call`, in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Maximum number of MCP servers one request may declare.
    #[serde(default = "default_max_servers")]
    pub max_servers: usize,

    /// Maximum number of tools accepted from one MCP server.
    #[serde(default = "default_max_tools")]
    pub max_tools: usize,

    /// Maximum request and response body size buffered by the loop.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Maximum number of MCP calls running at once.
    #[serde(default = "default_max_parallel_calls")]
    pub max_parallel_calls: usize,

    /// Maximum bytes retained for one MCP result.
    #[serde(default = "default_max_result_bytes")]
    pub max_result_bytes: usize,

    /// Maximum bytes retained for all results of one model round.
    #[serde(default = "default_max_total_result_bytes")]
    pub max_total_result_bytes: usize,
}

/// Serde default for `timeout_ms`.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Serde default for `max_servers`.
fn default_max_servers() -> usize {
    DEFAULT_MAX_SERVERS
}

/// Serde default for `max_tools`.
fn default_max_tools() -> usize {
    DEFAULT_MAX_TOOLS
}

/// Serde default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Serde default for `max_parallel_calls`.
fn default_max_parallel_calls() -> usize {
    DEFAULT_MAX_PARALLEL_CALLS
}

/// Serde default for `max_result_bytes`.
fn default_max_result_bytes() -> usize {
    DEFAULT_MAX_RESULT_BYTES
}

/// Serde default for `max_total_result_bytes`.
fn default_max_total_result_bytes() -> usize {
    DEFAULT_MAX_TOTAL_RESULT_BYTES
}

// -----------------------------------------------------------------------------
// Validated Configuration
// -----------------------------------------------------------------------------

/// One operator connector, keyed by its normalized URL.
#[derive(Debug)]
pub(super) struct Connector {
    /// Connector id recorded in the tool map.
    pub id: String,
    /// Configured server URL, dialed for every call.
    pub server_url: String,
}

/// Validated settings the filter runs with.
#[derive(Debug)]
pub(super) struct ValidatedConfig {
    /// Connectors keyed by normalized `server_url`.
    pub connectors: HashMap<String, Connector>,
    /// Per-connector tool policies keyed by connector id.
    pub connector_tool_policies: HashMap<String, ConnectorToolPolicy>,
    /// Whether unmatched client servers may be dialed directly.
    pub allow_client_servers: bool,
    /// Operator admission policy.
    pub admission_policy: McpAdmissionPolicy,
    /// Trusted headers forwarded to connectors.
    pub forward_headers: Vec<http::HeaderName>,
    /// Per-user bearer slot.
    pub user_credential: Option<String>,
    /// Authorization-assertion slot.
    pub authorization_assertion: Option<String>,
    /// Per-call MCP timeout.
    pub timeout: Duration,
    /// Maximum declared servers.
    pub max_servers: usize,
    /// Maximum tools per server.
    pub max_tools: usize,
    /// Body ceiling for requests and responses.
    pub max_body_bytes: usize,
    /// Concurrent call ceiling.
    pub max_parallel_calls: usize,
    /// Per-result retained ceiling.
    pub max_result_bytes: usize,
    /// Per-round retained ceiling.
    pub max_total_result_bytes: usize,
}

/// Validate the raw configuration.
///
/// # Errors
///
/// Returns [`FilterError`] for an invalid limit, slot, forwarded header, or
/// connector, or for two connectors sharing one URL.
#[expect(clippy::too_many_lines, reason = "linear validation and connector indexing")]
pub(super) fn build_config(cfg: AnthropicMcpConnectorConfig) -> Result<ValidatedConfig, FilterError> {
    validate_connectors(FILTER_NAME, &cfg.connectors)?;
    validate_slot("user_credential", cfg.user_credential.as_deref())?;
    validate_slot("authorization_assertion", cfg.authorization_assertion.as_deref())?;
    let forward_headers = parse_forward_headers(&cfg.forward_headers)?;
    validate_limits(&cfg)?;

    let mut connectors = HashMap::with_capacity(cfg.connectors.len());
    let mut connector_tool_policies = HashMap::with_capacity(cfg.connectors.len());
    for connector in cfg.connectors {
        let key = normalize_server_url(&connector.server_url).ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: connector \"{}\" has an invalid server_url",
                connector.id
            ))
        })?;
        let tool_policy = connector.tool_policy.snapshot();
        connector_tool_policies.insert(connector.id.clone(), tool_policy);
        let previous = connectors.insert(
            key,
            Connector {
                id: connector.id,
                server_url: connector.server_url,
            },
        );
        if let Some(previous) = previous {
            return Err(format!(
                "{FILTER_NAME}: connector \"{}\" shares its server_url with another connector",
                previous.id
            )
            .into());
        }
    }

    Ok(ValidatedConfig {
        connectors,
        connector_tool_policies,
        allow_client_servers: cfg.allow_client_servers,
        admission_policy: cfg.tool_admission.snapshot(),
        forward_headers,
        user_credential: cfg.user_credential,
        authorization_assertion: cfg.authorization_assertion,
        timeout: Duration::from_millis(cfg.timeout_ms),
        max_servers: cfg.max_servers,
        max_tools: cfg.max_tools,
        max_body_bytes: cfg.max_body_bytes,
        max_parallel_calls: cfg.max_parallel_calls,
        max_result_bytes: cfg.max_result_bytes,
        max_total_result_bytes: cfg.max_total_result_bytes,
    })
}

/// Normalize a server URL for connector matching.
///
/// Parsing lowercases the scheme and host and adds the root path, so
/// `https://MCP.example.com` and `https://mcp.example.com/` match.
pub(super) fn normalize_server_url(raw: &str) -> Option<String> {
    url::Url::parse(raw).ok().map(String::from)
}

/// Validate numeric limits.
fn validate_limits(cfg: &AnthropicMcpConnectorConfig) -> Result<(), FilterError> {
    if cfg.timeout_ms == 0 {
        return Err(format!("{FILTER_NAME}: timeout_ms must be greater than 0").into());
    }
    if cfg.max_servers == 0 {
        return Err(format!("{FILTER_NAME}: max_servers must be greater than 0").into());
    }
    if cfg.max_tools == 0 {
        return Err(format!("{FILTER_NAME}: max_tools must be greater than 0").into());
    }
    if cfg.max_body_bytes == 0 || cfg.max_body_bytes > MAX_JSON_BODY_BYTES {
        return Err(format!("{FILTER_NAME}: max_body_bytes must be between 1 and {MAX_JSON_BODY_BYTES}").into());
    }
    if cfg.max_parallel_calls == 0 || cfg.max_parallel_calls > MAX_PARALLEL_CALLS {
        return Err(format!("{FILTER_NAME}: max_parallel_calls must be between 1 and {MAX_PARALLEL_CALLS}").into());
    }
    if cfg.max_result_bytes < MIN_RETAINED_RESULT_BYTES || cfg.max_result_bytes > MAX_RESULT_BYTES {
        return Err(format!(
            "{FILTER_NAME}: max_result_bytes must be between {MIN_RETAINED_RESULT_BYTES} and {MAX_RESULT_BYTES}"
        )
        .into());
    }
    if cfg.max_total_result_bytes < cfg.max_result_bytes || cfg.max_total_result_bytes > MAX_TOTAL_RESULT_BYTES {
        return Err(format!(
            "{FILTER_NAME}: max_total_result_bytes must be between max_result_bytes and {MAX_TOTAL_RESULT_BYTES}"
        )
        .into());
    }
    Ok(())
}

/// Validate a per-user context slot id.
fn validate_slot(field: &str, slot: Option<&str>) -> Result<(), FilterError> {
    if let Some(slot) = slot
        && (slot.is_empty() || slot.len() > MAX_SLOT_ID_LEN)
    {
        return Err(format!("{FILTER_NAME}: {field} must be 1..={MAX_SLOT_ID_LEN} bytes").into());
    }
    Ok(())
}

/// Parse forwarded header names, rejecting credential and MCP-owned headers.
fn parse_forward_headers(names: &[String]) -> Result<Vec<http::HeaderName>, FilterError> {
    names
        .iter()
        .map(|configured| {
            let name = http::HeaderName::from_bytes(configured.as_bytes())
                .map_err(|error| FilterError::from(format!("{FILTER_NAME}: invalid forward header: {error}")))?;
            if crate::mcp_client::is_blocked_mcp_header(&name) {
                return Err(FilterError::from(format!(
                    "{FILTER_NAME}: 'forward_headers' must not include credential or MCP-controlled header '{name}'"
                )));
            }
            Ok(name)
        })
        .collect()
}
