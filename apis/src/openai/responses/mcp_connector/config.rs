// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_connector` filter.

use praxis_core::config::ChainRef;
use praxis_filter::FilterError;
use serde::Deserialize;

use crate::{
    json_body::MAX_JSON_BODY_BYTES,
    mcp_core::{
        classify::ApprovalFloor,
        connector::{ConnectorConfig, ToolAdmissionConfig, validate_connectors},
        execute::MIN_RETAINED_RESULT_BYTES,
    },
    openai::responses::body_limits::validate_size_limit,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default timeout for MCP `tools/call` calls (30 seconds).
const DEFAULT_CALL_TIMEOUT_MS: u64 = 30_000;

/// Default hard cap on MCP calls accepted from one model round.
const DEFAULT_MAX_CALLS_PER_ROUND: usize = 32;

/// Default maximum number of in-flight MCP calls.
const DEFAULT_MAX_PARALLEL_CALLS: usize = 8;

/// Default maximum retained bytes for one MCP result.
const DEFAULT_MAX_RESULT_BYTES: usize = 1_048_576;

/// Default maximum retained bytes for one MCP result batch.
const DEFAULT_MAX_TOTAL_RESULT_BYTES: usize = 8_388_608;

/// Absolute maximum MCP calls accepted from one model round.
const MAX_CALLS_PER_ROUND: usize = 1_024;

/// Absolute maximum concurrent MCP calls.
const MAX_PARALLEL_CALLS: usize = 64;

/// Absolute maximum retained bytes for one MCP result.
const MAX_RESULT_BYTES: usize = 16_777_216;

/// Absolute maximum retained bytes for one MCP result batch.
const MAX_TOTAL_RESULT_BYTES: usize = 67_108_864;

/// Maximum UTF-8 bytes accepted for one MCP correlation `call_id`.
///
/// This is the raw-length candidate bound. Approval records persist the call id
/// verbatim and retained results echo it as `approval_request_id`, so the exact
/// serialized fallback — including JSON escaping — must also fit
/// [`MIN_RETAINED_RESULT_BYTES`]. Ids that fail either check are rejected
/// before any approval is persisted; they are never truncated, because
/// truncation would break correlation.
#[cfg(any(feature = "openai-agentic-loop", all(test, feature = "store-sqlite")))]
pub(in crate::openai::responses) const MAX_MCP_CALL_ID_BYTES: usize = 128;

/// Default timeout for MCP `tools/list` calls (5 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Default maximum number of MCP servers per request.
const DEFAULT_MAX_SERVERS: usize = 10;

/// Default maximum number of tools returned by a single MCP server.
const DEFAULT_MAX_TOOLS: usize = 128;


// -----------------------------------------------------------------------------
// McpConnectorConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the `openai_mcp_connector` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpConnectorConfig {
    /// Optional per-user bearer slot from `callout_credentials`.
    #[serde(default)]
    pub user_credential: Option<String>,

    /// Optional opaque assertion slot from `callout_credentials.assertions`.
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
    /// and tool requests.
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

    /// Per-server timeout in milliseconds for `tools/list` calls (default: 5000). Inside an
    /// iterative request router, initialize and listing exchanges are capped by
    /// the router's remaining deadline.
    #[serde(default = "default_discovery_timeout_ms")]
    pub discovery_timeout_ms: u64,

    /// Maximum number of distinct MCP servers per request.
    #[serde(default = "default_max_servers")]
    pub max_servers: usize,

    /// Maximum number of tools returned by a single MCP server.
    #[serde(default = "default_max_tools")]
    pub max_tools: usize,

    /// Outbound filter chain shared by eager discovery, deferred discovery,
    /// and execution. Accepts an inline chain or a reference to a top-level
    /// `filter_chains` entry. A named chain must start with an unconditional
    /// `openai_mcp_streaming_selector`. The connector supplies the selector for
    /// inline or omitted chains.
    /// The transport stages and validates the destination before the exchange.
    /// Private upstreams are governed by the global insecure options.
    #[serde(default)]
    pub outbound_chain: Option<ChainRef>,

    /// Named connectors mapping connector IDs to server URLs.
    #[serde(default)]
    pub connectors: Vec<ConnectorConfig>,

    /// Operator metadata-admission policy. Defaults to advisory (unchanged
    /// behavior), so existing non-protected deployments are not affected.
    #[serde(default)]
    pub tool_admission: ToolAdmissionConfig,

    /// Operator approval floor applied to every MCP call in this pipeline,
    /// combined monotonically with each tool's client `require_approval`:
    /// `approval_required = operator_floor || client_requires_approval`. Set
    /// `always` to require approval for every MCP call regardless of the client's
    /// request — a client `"never"` or a server-supplied `readOnlyHint` can never
    /// suppress it. Pending approvals are correlated through `previous_response_id`,
    /// so the pipeline needs `openai_responses_store` and `openai_responses_rehydrate`;
    /// without persistence the request fails closed (`store: false` → 400, no store → 500).
    /// Omit to preserve the client-driven behavior (the default).
    #[serde(default)]
    pub minimum_approval: Option<MinimumApproval>,

    /// Timeout in milliseconds for each `tools/call` exchange (default: 30000). Inside an
    /// iterative request router, every MCP exchange is capped by the router's
    /// remaining deadline.
    #[serde(default = "default_call_timeout_ms")]
    pub call_timeout_ms: u64,

    /// Hard cap on MCP calls processed from one model response, and the maximum
    /// number of `mcp_approval_response` items accepted from a single resume
    /// request — a resume batch cannot exceed what one round could have emitted
    /// (1..=1024; default: 32).
    #[serde(default = "default_max_calls_per_round")]
    pub max_calls_per_round: usize,

    /// Maximum concurrent MCP calls when `parallel_tool_calls` is enabled (1..=64; default: 8).
    #[serde(default = "default_max_parallel_calls")]
    pub max_parallel_calls: usize,

    /// Maximum retained bytes for one MCP result (minimum: 1 `KiB`; default: 1 `MiB`).
    #[serde(default = "default_max_result_bytes")]
    pub max_result_bytes: usize,

    /// Maximum retained bytes across one MCP result batch (default: 8 MiB).
    #[serde(default = "default_max_total_result_bytes")]
    pub max_total_result_bytes: usize,
}

/// Operator approval-floor setting for `openai_mcp_connector`.
///
/// The YAML surface for the operator floor combined with each tool's client
/// `require_approval` policy. Only `always` is meaningful: a `never` floor would
/// be a no-op (the floor can only *raise* the requirement, never weaken it), so
/// it is deliberately not a variant and is rejected by `deny_unknown_fields`.
/// Maps to [`ApprovalFloor`] via [`MinimumApproval::as_floor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MinimumApproval {
    /// Require approval for every MCP call, regardless of client settings.
    Always,
}

impl MinimumApproval {
    /// Project the configured floor onto the shared classification [`ApprovalFloor`].
    pub(crate) const fn as_floor(self) -> ApprovalFloor {
        match self {
            Self::Always => ApprovalFloor::Always,
        }
    }
}

/// Default value for `timeout_ms`.
fn default_call_timeout_ms() -> u64 {
    DEFAULT_CALL_TIMEOUT_MS
}

/// Default value for `max_calls_per_round`.
fn default_max_calls_per_round() -> usize {
    DEFAULT_MAX_CALLS_PER_ROUND
}

/// Default value for `max_parallel_calls`.
fn default_max_parallel_calls() -> usize {
    DEFAULT_MAX_PARALLEL_CALLS
}

/// Default value for `max_result_bytes`.
fn default_max_result_bytes() -> usize {
    DEFAULT_MAX_RESULT_BYTES
}

/// Default value for `max_total_result_bytes`.
fn default_max_total_result_bytes() -> usize {
    DEFAULT_MAX_TOTAL_RESULT_BYTES
}

/// Default max rewritten body bytes (64 MiB): a post-expansion backstop.
fn default_max_rewritten_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Default timeout in milliseconds.
fn default_discovery_timeout_ms() -> u64 {
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
#[expect(
    clippy::too_many_lines,
    reason = "validate connector settings and both sets of bounds together"
)]
pub(crate) fn build_config(mut cfg: McpConnectorConfig) -> Result<McpConnectorConfig, FilterError> {
    crate::openai::api_client::validate_forward_headers("openai_mcp_connector", &mut cfg.forward_headers)?;
    reject_mcp_sensitive_forward_headers(&cfg.forward_headers)?;
    validate_context_slot("user_credential", cfg.user_credential.as_deref())?;
    validate_context_slot("authorization_assertion", cfg.authorization_assertion.as_deref())?;
    if !cfg.require_connector && (cfg.user_credential.is_some() || cfg.authorization_assertion.is_some()) {
        tracing::warn!(
            user_credential = ?cfg.user_credential,
            authorization_assertion = ?cfg.authorization_assertion,
            "openai_mcp_connector: per-user MCP authorization is configured without require_connector; direct server_url entries remain allowed"
        );
    }
    validate_size_limit(
        "openai_mcp_connector",
        "max_rewritten_body_bytes",
        cfg.max_rewritten_body_bytes,
    )?;
    if cfg.discovery_timeout_ms == 0 {
        return Err("openai_mcp_connector: discovery_timeout_ms must be greater than 0".into());
    }
    if cfg.max_servers == 0 {
        return Err("openai_mcp_connector: max_servers must be greater than 0".into());
    }
    if cfg.max_tools == 0 {
        return Err("openai_mcp_connector: max_tools must be greater than 0".into());
    }
    validate_connectors("openai_mcp_connector", &cfg.connectors)?;
    if cfg.call_timeout_ms == 0 {
        return Err("openai_mcp_connector: call_timeout_ms must be greater than 0".into());
    }
    if cfg.max_calls_per_round == 0 {
        return Err("openai_mcp_connector: max_calls_per_round must be greater than 0".into());
    }
    if cfg.max_calls_per_round > MAX_CALLS_PER_ROUND {
        return Err(format!("openai_mcp_connector: max_calls_per_round must not exceed {MAX_CALLS_PER_ROUND}").into());
    }
    if cfg.max_parallel_calls == 0 {
        return Err("openai_mcp_connector: max_parallel_calls must be greater than 0".into());
    }
    if cfg.max_parallel_calls > MAX_PARALLEL_CALLS {
        return Err(format!("openai_mcp_connector: max_parallel_calls must not exceed {MAX_PARALLEL_CALLS}").into());
    }
    if cfg.max_result_bytes == 0 {
        return Err("openai_mcp_connector: max_result_bytes must be greater than 0".into());
    }
    if cfg.max_result_bytes < MIN_RETAINED_RESULT_BYTES {
        return Err(
            format!("openai_mcp_connector: max_result_bytes must be at least {MIN_RETAINED_RESULT_BYTES}").into(),
        );
    }
    if cfg.max_result_bytes > MAX_RESULT_BYTES {
        return Err(format!("openai_mcp_connector: max_result_bytes must not exceed {MAX_RESULT_BYTES}").into());
    }
    if cfg.max_total_result_bytes < cfg.max_result_bytes {
        return Err("openai_mcp_connector: max_total_result_bytes must be at least max_result_bytes".into());
    }
    let minimum_batch_reservation = cfg
        .max_calls_per_round
        .checked_mul(MIN_RETAINED_RESULT_BYTES)
        .ok_or_else(|| -> FilterError { "openai_mcp_connector: result-byte reservation overflow".into() })?;
    if cfg.max_total_result_bytes < minimum_batch_reservation {
        return Err(format!(
            "openai_mcp_connector: max_total_result_bytes must reserve at least {MIN_RETAINED_RESULT_BYTES} bytes per max_calls_per_round"
        )
        .into());
    }
    if cfg.max_total_result_bytes > MAX_TOTAL_RESULT_BYTES {
        return Err(
            format!("openai_mcp_connector: max_total_result_bytes must not exceed {MAX_TOTAL_RESULT_BYTES}").into(),
        );
    }
    Ok(cfg)
}

/// Validate one optional request-scoped context slot name.
fn validate_context_slot(field: &str, slot: Option<&str>) -> Result<(), FilterError> {
    if let Some(slot) = slot
        && (slot.is_empty() || slot.len() > 128)
    {
        return Err(format!("openai_mcp_connector: {field} must be 1..=128 bytes").into());
    }
    Ok(())
}

/// Reject ambient credentials and protocol-controlled fields at MCP's
/// client-selected destination boundary.
fn reject_mcp_sensitive_forward_headers(headers: &[String]) -> Result<(), FilterError> {
    for configured in headers {
        let name = http::HeaderName::from_bytes(configured.as_bytes()).map_err(|error| -> FilterError {
            format!("openai_mcp_connector: invalid forward header: {error}").into()
        })?;
        if crate::mcp_client::is_blocked_mcp_header(&name) {
            return Err(format!(
                "openai_mcp_connector: 'forward_headers' must not include credential or MCP-controlled header '{name}'"
            )
            .into());
        }
    }
    Ok(())
}
