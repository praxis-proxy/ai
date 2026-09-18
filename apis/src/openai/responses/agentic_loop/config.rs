// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the agentic loop filter.

use praxis_core::config::MAX_ITERATIONS_CEILING;
use praxis_filter::FilterError;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Defaults
// -----------------------------------------------------------------------------

/// Default maximum inference iterations in the agentic loop.
///
/// Not part of the OpenAI spec — this is a Praxis-only safety cap
/// on how many inference round-trips the agentic loop can perform.
pub(super) const DEFAULT_MAX_INFER_ITERS: u32 = 10;

/// Default aggregate payload retained by one Responses agentic request: 64 `MiB`.
pub(super) const DEFAULT_MAX_RETAINED_BYTES: usize = 67_108_864;

/// Smallest useful aggregate retained-payload budget: 4 `KiB`.
pub(super) const MIN_MAX_RETAINED_BYTES: usize = 4_096;

/// Non-disableable aggregate retained-payload ceiling: 256 `MiB`.
pub(super) const MAX_MAX_RETAINED_BYTES: usize = 268_435_456;

/// Serde default for `max_infer_iters`.
fn default_max_infer_iters() -> u32 {
    DEFAULT_MAX_INFER_ITERS
}

/// Serde default for `max_retained_bytes`.
fn default_max_retained_bytes() -> usize {
    DEFAULT_MAX_RETAINED_BYTES
}

// -----------------------------------------------------------------------------
// AgenticLoopConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the agentic loop filter.
///
/// ```yaml
/// filter: openai_agentic_loop
/// max_infer_iters: 10
/// max_retained_bytes: 67108864
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AgenticLoopConfig {
    /// Maximum number of inference loop iterations (Praxis-only,
    /// not part of the OpenAI API spec). When the iteration counter
    /// reaches this limit, the loop returns a 508 Loop Detected error. Must be
    /// between 1 and [`MAX_ITERATIONS_CEILING`] (currently 100); defaults to 10.
    #[serde(default = "default_max_infer_iters")]
    pub max_infer_iters: u32,

    /// Maximum aggregate payload bytes retained by the Responses agentic
    /// execution. Counts compact JSON for each independently owned value and raw
    /// bytes for owned strings and streaming buffers. When several loop filters
    /// touch one request, the smallest configured value wins. Valid from 4 `KiB`
    /// through the non-disableable 256 `MiB` ceiling; defaults to 64 `MiB`.
    #[serde(default = "default_max_retained_bytes")]
    pub max_retained_bytes: usize,
}

impl Default for AgenticLoopConfig {
    fn default() -> Self {
        Self {
            max_infer_iters: DEFAULT_MAX_INFER_ITERS,
            max_retained_bytes: DEFAULT_MAX_RETAINED_BYTES,
        }
    }
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(super) fn build_config(cfg: AgenticLoopConfig) -> Result<AgenticLoopConfig, FilterError> {
    if !(1..=MAX_ITERATIONS_CEILING).contains(&cfg.max_infer_iters) {
        return Err(format!(
            "openai_agentic_loop: max_infer_iters must be in 1..={MAX_ITERATIONS_CEILING}, got {}",
            cfg.max_infer_iters
        )
        .into());
    }
    if !(MIN_MAX_RETAINED_BYTES..=MAX_MAX_RETAINED_BYTES).contains(&cfg.max_retained_bytes) {
        return Err(format!(
            "openai_agentic_loop: max_retained_bytes must be in {MIN_MAX_RETAINED_BYTES}..={MAX_MAX_RETAINED_BYTES}, got {}",
            cfg.max_retained_bytes
        )
        .into());
    }
    Ok(cfg)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "configuration tests")]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> AgenticLoopConfig {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn retained_byte_default_and_bounds() {
        assert_eq!(parse("{}").max_retained_bytes, DEFAULT_MAX_RETAINED_BYTES);
        assert!(build_config(parse("max_retained_bytes: 4095")).is_err());
        assert!(build_config(parse("max_retained_bytes: 4096")).is_ok());
        assert!(build_config(parse("max_retained_bytes: 268435456")).is_ok());
        assert!(build_config(parse("max_retained_bytes: 268435457")).is_err());
    }

    #[test]
    fn inference_iteration_bounds_match_core_ceiling() {
        for (value, accepted) in [(0, false), (1, true), (100, true), (101, false)] {
            let config = parse(&format!("max_infer_iters: {value}"));
            assert_eq!(build_config(config).is_ok(), accepted, "max_infer_iters={value}");
        }
    }
}
