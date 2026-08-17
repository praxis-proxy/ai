// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the `token_rate_limit` filter.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// TokenRateLimitConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `token_rate_limit` filter.
///
/// This covers only the uncontested MVP core of `ai#658`'s proposal: a
/// single global bucket (M1's flat `default` rule; per-header/composite
/// bucket keys from M5 are explicitly deferred — that section is still
/// marked TBD in the proposal itself), reservation-based admission (M2)
/// against a fixed per-request estimate (a placeholder for M3's
/// configurable estimation strategies, which are a separate open design
/// surface), and standard 429/headers (M6).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenRateLimitConfig {
    /// Tokens replenished per second.
    pub rate: f64,

    /// Maximum bucket capacity, in tokens.
    pub burst: f64,

    /// Fixed token cost reserved at admission time, before actual usage
    /// is known.
    ///
    /// MVP placeholder for M3 (configurable estimation strategies).
    /// Real deployments will want this derived from request metadata
    /// (e.g. `max_tokens`) rather than a single fixed constant — that's
    /// out of scope for this walking skeleton.
    pub estimate_tokens: f64,
}
