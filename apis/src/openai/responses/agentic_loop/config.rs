// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the agentic loop filter.

use praxis_core::config::MAX_ITERATIONS_CEILING;
use praxis_filter::FilterError;
use serde::{Deserialize, Deserializer};

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
fn default_max_retained_bytes() -> RetainedBytes {
    RetainedBytes::default()
}

/// A retained-payload budget constrained to the supported range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RetainedBytes(usize);

impl RetainedBytes {
    /// Return the validated byte budget.
    pub(super) const fn get(self) -> usize {
        self.0
    }
}

impl Default for RetainedBytes {
    fn default() -> Self {
        Self(DEFAULT_MAX_RETAINED_BYTES)
    }
}

impl TryFrom<usize> for RetainedBytes {
    type Error = String;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if !(MIN_MAX_RETAINED_BYTES..=MAX_MAX_RETAINED_BYTES).contains(&value) {
            return Err(format!(
                "openai_agentic_loop: max_retained_bytes must be in {MIN_MAX_RETAINED_BYTES}..={MAX_MAX_RETAINED_BYTES}, got {value}"
            ));
        }
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for RetainedBytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = usize::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
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
    /// between 1 and 100 (the core IRR iteration ceiling); defaults to 10.
    #[serde(default = "default_max_infer_iters")]
    pub max_infer_iters: u32,

    /// Conservative request-wide retained-payload ceiling. Valid from 4 `KiB`
    /// through 256 `MiB`, defaults to a non-disableable 64 `MiB`, and uses the
    /// smallest limit when several loop instances are reachable. It covers
    /// typed input, history restore, compaction, file and document expansion,
    /// Store persistence, buffered and streaming output, Responses-to-Chat
    /// translation, MCP dispatch, and hosted tools across inference rounds.
    /// Initial payloads are conservatively charged at 32 times raw bytes plus
    /// a per-node reserve; cumulative buffered provider output is charged at
    /// 64 times its wire bytes plus a larger per-node reserve. Store
    /// persistence doubles those conservative charges for its independent
    /// request input snapshot and response persistence projection. Finite Chat
    /// translation reserves its additional request and response JSON owners
    /// before allocating them. Streaming reserves raw chunks, parsed frames,
    /// terminal output, and separately retained tool results. Store history
    /// reads are capped before decoding;
    /// the decoded record, replay, and replacement state are charged at 512
    /// times the stored record's serialized bytes. Conversation append-back
    /// uses a transactional bounded cache rebuild. The listener also clamps
    /// its raw request body limit to at most one
    /// thirty-second of this ceiling and its buffered IRR response limit to
    /// at most one eighth. These reserves can
    /// reject a request well below the configured ceiling. Budgeted MCP calls
    /// use fresh sessions so each call applies its current transport limit.
    /// Initial budget overflow returns HTTP 413; buffered provider overflow
    /// returns HTTP 502; a committed stream emits one SSE error on overflow.
    #[serde(default = "default_max_retained_bytes")]
    pub max_retained_bytes: RetainedBytes,
}

impl Default for AgenticLoopConfig {
    fn default() -> Self {
        Self {
            max_infer_iters: DEFAULT_MAX_INFER_ITERS,
            max_retained_bytes: RetainedBytes::default(),
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
        assert_eq!(parse("{}").max_retained_bytes.get(), DEFAULT_MAX_RETAINED_BYTES);
        assert!(serde_yaml::from_str::<AgenticLoopConfig>("max_retained_bytes: 4095").is_err());
        assert!(serde_yaml::from_str::<AgenticLoopConfig>("max_retained_bytes: 4096").is_ok());
        assert!(serde_yaml::from_str::<AgenticLoopConfig>("max_retained_bytes: 268435456").is_ok());
        assert!(serde_yaml::from_str::<AgenticLoopConfig>("max_retained_bytes: 268435457").is_err());

        let err = RetainedBytes::try_from(4095).unwrap_err();
        assert_eq!(
            err,
            "openai_agentic_loop: max_retained_bytes must be in 4096..=268435456, got 4095"
        );
    }

    #[test]
    fn inference_iteration_bounds_match_core_ceiling() {
        for (value, accepted) in [(0, false), (1, true), (100, true), (101, false)] {
            let config = parse(&format!("max_infer_iters: {value}"));
            assert_eq!(build_config(config).is_ok(), accepted, "max_infer_iters={value}");
        }
    }
}
