// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_dispatch` filter.

use praxis_core::config::ChainRef;
use praxis_filter::FilterError;
use serde::Deserialize;

/// Default timeout for MCP `tools/call` calls (30 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

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

/// Minimum retained reservation needed for one bounded tool-error result.
pub(super) const MIN_RETAINED_RESULT_BYTES: usize = 1_024;

/// YAML configuration for the `openai_mcp_dispatch` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct McpDispatchConfig {
    /// Inline outbound filter chain the MCP `tools/call` callout runs through.
    ///
    /// `openai_mcp_dispatch` runs inside an `iterative_request_router` step. praxis
    /// core builds each IRR step with a live chain-binding context, so an inline
    /// chain (`{ name, filters }`) is bound at step-build time. A named reference is
    /// rejected: IRR supplies each step an empty top-level named-chain map, so a
    /// `Named` reference can never resolve inside a step (see
    /// [`require_inline_outbound_chain`]). The chain carries only
    /// operator-configured cross-cutting filters — the SSRF-validated dial target is
    /// staged by the transport, so no upstream-selecting filter is prepended. When
    /// omitted, the callout runs through an empty chain and dials the staged target
    /// directly. Whether loopback/private MCP destinations are permitted is governed
    /// by the operator's global insecure posture, not a per-filter flag.
    #[serde(default)]
    pub outbound_chain: Option<ChainRef>,

    /// Trusted request headers forwarded to connector-backed MCP `tools/call` requests.
    /// No request headers are forwarded by default. Credential headers such as
    /// `authorization` are rejected because MCP destinations are client-selected;
    /// use the MCP tool entry's dedicated `authorization` field instead. Direct,
    /// client-selected `server_url` targets never receive ambient request headers.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Per-call timeout in milliseconds for `tools/call` calls.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Hard cap on MCP calls processed from one model response (1..=1024; default: 32).
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

/// Default value for `timeout_ms`.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
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

/// Validate the parsed configuration.
#[expect(
    clippy::too_many_lines,
    reason = "related call, concurrency, and retained-byte invariants are validated together"
)]
pub(crate) fn build_config(mut cfg: McpDispatchConfig) -> Result<McpDispatchConfig, FilterError> {
    crate::openai::api_client::validate_forward_headers("openai_mcp_dispatch", &mut cfg.forward_headers)?;
    reject_mcp_sensitive_forward_headers("openai_mcp_dispatch", &cfg.forward_headers)?;
    if cfg.timeout_ms == 0 {
        return Err("openai_mcp_dispatch: timeout_ms must be greater than 0".into());
    }
    if cfg.max_calls_per_round == 0 {
        return Err("openai_mcp_dispatch: max_calls_per_round must be greater than 0".into());
    }
    if cfg.max_calls_per_round > MAX_CALLS_PER_ROUND {
        return Err(format!("openai_mcp_dispatch: max_calls_per_round must not exceed {MAX_CALLS_PER_ROUND}").into());
    }
    if cfg.max_parallel_calls == 0 {
        return Err("openai_mcp_dispatch: max_parallel_calls must be greater than 0".into());
    }
    if cfg.max_parallel_calls > MAX_PARALLEL_CALLS {
        return Err(format!("openai_mcp_dispatch: max_parallel_calls must not exceed {MAX_PARALLEL_CALLS}").into());
    }
    if cfg.max_result_bytes == 0 {
        return Err("openai_mcp_dispatch: max_result_bytes must be greater than 0".into());
    }
    if cfg.max_result_bytes < MIN_RETAINED_RESULT_BYTES {
        return Err(
            format!("openai_mcp_dispatch: max_result_bytes must be at least {MIN_RETAINED_RESULT_BYTES}").into(),
        );
    }
    if cfg.max_result_bytes > MAX_RESULT_BYTES {
        return Err(format!("openai_mcp_dispatch: max_result_bytes must not exceed {MAX_RESULT_BYTES}").into());
    }
    if cfg.max_total_result_bytes < cfg.max_result_bytes {
        return Err("openai_mcp_dispatch: max_total_result_bytes must be at least max_result_bytes".into());
    }
    let minimum_batch_reservation = cfg
        .max_calls_per_round
        .checked_mul(MIN_RETAINED_RESULT_BYTES)
        .ok_or_else(|| -> FilterError { "openai_mcp_dispatch: result-byte reservation overflow".into() })?;
    if cfg.max_total_result_bytes < minimum_batch_reservation {
        return Err(format!(
            "openai_mcp_dispatch: max_total_result_bytes must reserve at least {MIN_RETAINED_RESULT_BYTES} bytes per max_calls_per_round"
        )
        .into());
    }
    if cfg.max_total_result_bytes > MAX_TOTAL_RESULT_BYTES {
        return Err(
            format!("openai_mcp_dispatch: max_total_result_bytes must not exceed {MAX_TOTAL_RESULT_BYTES}").into(),
        );
    }
    Ok(cfg)
}

/// Reject a `Named` outbound-chain reference, requiring an inline chain.
///
/// `openai_mcp_dispatch` runs nested inside an `iterative_request_router` step,
/// and IRR builds each step's pipeline with an empty top-level named-chain map.
/// A `Named` reference (`outbound_chain: my-chain`) therefore can never resolve
/// inside a step and would fail pipeline construction with a confusing "unknown
/// chain" error. Require the chain inline instead
/// (`outbound_chain: { name: ..., filters: [...] }`), which embeds its filters
/// directly and needs no lookup. Propagating outer named chains into IRR steps
/// is a praxis-core follow-up.
///
/// # Errors
///
/// Returns [`FilterError`] when `outbound_chain` is a [`ChainRef::Named`].
pub(crate) fn require_inline_outbound_chain(outbound_chain: Option<&ChainRef>) -> Result<(), FilterError> {
    if let Some(ChainRef::Named(name)) = outbound_chain {
        return Err(format!(
            "openai_mcp_dispatch: outbound_chain must be defined inline \
             ({{ name, filters }}); a named reference ('{name}') cannot resolve \
             inside the iterative_request_router step this filter runs in"
        )
        .into());
    }
    Ok(())
}

/// Reject ambient credentials and protocol-controlled fields at MCP's
/// client-selected destination boundary.
fn reject_mcp_sensitive_forward_headers(filter: &str, headers: &[String]) -> Result<(), FilterError> {
    for configured in headers {
        let name = http::HeaderName::from_bytes(configured.as_bytes())
            .map_err(|error| -> FilterError { format!("{filter}: invalid forward header: {error}").into() })?;
        if crate::mcp_client::is_blocked_mcp_header(&name) {
            return Err(format!(
                "{filter}: 'forward_headers' must not include credential or MCP-controlled header '{name}'"
            )
            .into());
        }
    }
    Ok(())
}
