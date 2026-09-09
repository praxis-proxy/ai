// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the `openai_mcp_dispatch` filter.

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
    /// Allow connections to loopback addresses (default: false).
    #[serde(default)]
    pub allow_loopback: bool,

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
pub(crate) fn build_config(cfg: McpDispatchConfig) -> Result<McpDispatchConfig, FilterError> {
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
