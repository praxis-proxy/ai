// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the `token_rate_limit` filter.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// TokenRateLimitConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `token_rate_limit` filter.
///
/// This covers the uncontested MVP core of `ai#658`'s proposal (a single
/// global bucket, reservation-based admission (M2) against a fixed
/// per-request estimate, and standard 429/headers (M6)) plus one narrow,
/// undisputed slice of M5: keying buckets by a single request header,
/// exactly as specified in `ai#129` (`bucket_key_header`, one bucket per
/// unique header value, fall back to the global bucket when absent).
/// Composite/CEL-expression keys and per-model keys from `ai#123`/`#658`'s
/// fuller M5 are still out of scope here.
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

    /// Header whose value keys an independent bucket, per `ai#129`.
    ///
    /// When set, each unique header value gets its own bucket sized by
    /// `rate`/`burst`/`estimate_tokens`; requests missing the header fall
    /// back to a single shared global bucket. When unset (the default),
    /// every request shares one global bucket, same as today.
    #[serde(default)]
    pub bucket_key_header: Option<String>,
}
