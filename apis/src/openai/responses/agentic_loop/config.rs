// SPDX-License-Identifier: Apache-2.0
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

/// Serde default for `max_infer_iters`.
fn default_max_infer_iters() -> u32 {
    DEFAULT_MAX_INFER_ITERS
}

// -----------------------------------------------------------------------------
// AgenticLoopConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the agentic loop filter.
///
/// ```yaml
/// filter: openai_agentic_loop
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
}

impl Default for AgenticLoopConfig {
    fn default() -> Self {
        Self {
            max_infer_iters: DEFAULT_MAX_INFER_ITERS,
        }
    }
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(super) fn build_config(cfg: AgenticLoopConfig) -> Result<AgenticLoopConfig, FilterError> {
    if cfg.max_infer_iters == 0 {
        return Err("openai_agentic_loop: max_infer_iters must be > 0".into());
    }
    Ok(cfg)
}
