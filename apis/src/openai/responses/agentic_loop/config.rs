// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the agentic loop filter.

use praxis_filter::FilterError;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------

/// Default maximum inference iterations in the agentic loop.
///
/// Not part of the OpenAI spec — this is a Praxis-only safety cap
/// on how many inference round-trips the agentic loop can perform.
const DEFAULT_MAX_INFER_ITERS: u32 = 10;

/// Default maximum response body size for `StreamBuffer` mode (10 MiB).
pub(super) const DEFAULT_MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Serde default for `max_infer_iters`.
fn default_max_infer_iters() -> u32 {
    DEFAULT_MAX_INFER_ITERS
}

/// Serde default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// AgenticLoopConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the agentic loop filter.
///
/// ```yaml
/// filter: agentic_loop
/// max_infer_iters: 10
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AgenticLoopConfig {
    /// Maximum number of inference loop iterations (Praxis-only,
    /// not part of the OpenAI API spec). When the iteration counter
    /// reaches this limit, the loop returns a 508 Loop Detected error.
    #[serde(default = "default_max_infer_iters")]
    pub max_infer_iters: u32,

    /// Maximum response body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

impl Default for AgenticLoopConfig {
    fn default() -> Self {
        Self {
            max_infer_iters: DEFAULT_MAX_INFER_ITERS,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(super) fn build_config(cfg: AgenticLoopConfig) -> Result<AgenticLoopConfig, FilterError> {
    if cfg.max_infer_iters == 0 {
        return Err("agentic_loop: max_infer_iters must be > 0".into());
    }
    if cfg.max_body_bytes == 0 {
        return Err("agentic_loop: max_body_bytes must be > 0".into());
    }
    Ok(cfg)
}
