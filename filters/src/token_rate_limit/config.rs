// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the `token_rate_limit` filter.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// TokenRateLimitConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `token_rate_limit` filter.
///
/// `window`/`capacity` match `ai#658`'s own field names ("Windows are
/// sliding: a `window: 1h` budget tracks usage in the most recent 60
/// minutes from the current instant") -- a single sliding-window budget,
/// reservation-based admission (M2) against a fixed per-request estimate,
/// and standard 429/headers (M6), plus one narrow, undisputed slice of M5:
/// keying budgets by a single request header, exactly as specified in
/// `ai#129` (`bucket_key_header`, one budget per unique header value, fall
/// back to a shared budget when absent). Composite/CEL-expression keys,
/// per-model keys (`ai#123`), and multiple budgets per rule are still out
/// of scope here.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenRateLimitConfig {
    /// Sliding window duration (e.g. `"1h"`, `"60s"`).
    pub window: String,

    /// Maximum tokens admitted within `window`.
    pub capacity: u64,

    /// Fixed token cost reserved at admission time, before actual usage
    /// is known.
    ///
    /// MVP placeholder for M3 (configurable estimation strategies).
    /// Real deployments will want this derived from request metadata
    /// (e.g. `max_tokens`) rather than a single fixed constant -- that's
    /// out of scope for this walking skeleton.
    pub estimate_tokens: u64,

    /// Header whose value keys an independent budget, per `ai#129`.
    ///
    /// When set, each unique header value gets its own budget; requests
    /// missing the header fall back to one shared budget. When unset (the
    /// default), every request shares one budget, same as today.
    #[serde(default)]
    pub bucket_key_header: Option<String>,

    /// How long an admitted-but-never-reconciled reservation (lost
    /// request: timeout, connection reset, upstream crash) is held before
    /// being conservatively charged at its estimate.
    ///
    /// Answers `ai#658`'s own still-open "lost request handling" question
    /// for this MVP. Defaults to [`DEFAULT_RESERVATION_TIMEOUT`] when unset.
    #[serde(default)]
    pub reservation_timeout: Option<String>,

    /// Where sliding-window state lives: in-process (default, one budget
    /// per gateway instance) or a shared Valkey backend (one budget
    /// shared across every gateway instance/replica).
    #[serde(default)]
    pub backend: BackendConfig,
}

/// Default reservation timeout when `reservation_timeout` is unset.
pub(super) const DEFAULT_RESERVATION_TIMEOUT: &str = "30s";

/// Backend selection and connection details.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BackendConfig {
    /// Which backend implementation to use.
    #[serde(default)]
    pub kind: BackendKind,

    /// Backend connection URL. Supports one `${ENV_VAR}` reference, so
    /// credentials/hostnames don't need to be committed to config.
    /// Required when `kind: valkey`, ignored otherwise.
    #[serde(default)]
    pub url: Option<String>,

    /// Key namespace prefix, so multiple filter rules or deployments can
    /// share one Valkey instance without colliding. Ignored for
    /// `kind: memory`. Defaults to `"praxis:token_rate_limit"` when unset.
    #[serde(default)]
    pub namespace: Option<String>,
}

/// Which state backend a `token_rate_limit` rule uses.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum BackendKind {
    /// In-process state: fast, no extra infrastructure, but not shared
    /// across gateway instances/replicas.
    #[default]
    Memory,

    /// Valkey-backed shared state: one budget shared across every gateway
    /// instance pointed at the same `namespace`.
    Valkey,
}
