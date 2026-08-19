// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration for the `token_rate_limit` filter.

use std::collections::BTreeMap;

use serde::Deserialize;

// -----------------------------------------------------------------------------
// TokenRateLimitConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `token_rate_limit` filter: an ordered
/// list of `rules`, each binding an optional match condition to an
/// algorithm choice (`sliding_window` or `token_bucket`) and that rule's
/// own budget.
///
/// Mirrors the `rules:`/`match:` shape from `ai#658`'s evolved design doc
/// (`docs/proposals/00121_token-rate-limiting.md`), scoped to this MVP's
/// static header-value matchers and per-rule algorithm choice. CEL
/// matchers, soft-limit tiers, weighted per-type accounting, and
/// configurable estimation strategies are still out of scope (see the
/// module doc comment) -- upstream itself defers the first two; the
/// latter two are deferred to a separate follow-up by design, not by
/// upstream mandate.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenRateLimitConfig {
    /// Evaluated in order; the first rule whose `match` is satisfied (or
    /// which has no `match` at all) applies to a given request. A
    /// request satisfying no rule's `match` is not rate limited by this
    /// filter instance -- add a trailing rule with no `match` to enforce
    /// a catch-all budget instead.
    pub rules: Vec<RuleConfig>,
}

/// One `rules:` entry: an optional match condition, an algorithm choice
/// with that algorithm's own parameters, and this rule's own
/// estimation/keying/backend configuration.
// `deny_unknown_fields` is deliberately omitted here: serde's flatten
// mechanism (`algorithm` below) is fundamentally incompatible with
// `deny_unknown_fields` on the containing struct -- the flattened
// enum's own fields get misreported as "unknown" because flatten
// collects remaining fields into an intermediate map before the tagged
// enum ever gets a chance to claim them. `AlgorithmConfig` itself still
// enforces `deny_unknown_fields` per-variant, so a genuinely unknown
// field (e.g. a typo, or an old flat-schema field like `window` on a
// `token_bucket` rule) is still rejected -- just attributed to the
// flattened enum's own error path instead of this struct's.
#[derive(Debug, Deserialize)]
pub(super) struct RuleConfig {
    /// Human-readable rule identifier, folded into Valkey key
    /// namespacing so distinct rules sharing one backend never collide.
    pub name: String,

    /// Static header-value match condition. Every listed header must be
    /// present on the request with an exact value match (`ANDed`) for
    /// this rule to apply. Omit entirely for a catch-all rule.
    #[serde(default)]
    pub r#match: Option<MatchConfig>,

    /// Which admission algorithm this rule enforces, and that
    /// algorithm's own parameters.
    #[serde(flatten)]
    pub algorithm: AlgorithmConfig,

    /// Fixed token cost reserved at admission time, before actual usage
    /// is known.
    ///
    /// MVP placeholder for M3 (configurable estimation strategies).
    /// Real deployments will want this derived from request metadata
    /// (e.g. `max_tokens`) rather than a single fixed constant -- that's
    /// out of scope for this walking skeleton.
    pub estimate_tokens: u64,

    /// Header whose value keys an independent budget within this rule,
    /// per `ai#129`.
    ///
    /// When set, each unique header value gets its own budget; requests
    /// missing the header fall back to one shared budget. When unset
    /// (the default), every request matching this rule shares one
    /// budget.
    #[serde(default)]
    pub bucket_key_header: Option<String>,

    /// How long an admitted-but-never-reconciled reservation (lost
    /// request: timeout, connection reset, upstream crash) is held
    /// before being conservatively charged at its estimate.
    ///
    /// Answers `ai#658`'s own still-open "lost request handling"
    /// question for this MVP. Defaults to [`DEFAULT_RESERVATION_TIMEOUT`]
    /// when unset.
    #[serde(default)]
    pub reservation_timeout: Option<String>,

    /// Where this rule's admission state lives: in-process (default,
    /// one budget per gateway instance) or a shared Valkey backend (one
    /// budget shared across every gateway instance/replica).
    #[serde(default)]
    pub backend: BackendConfig,
}

/// Static header-value match condition for a [`RuleConfig`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MatchConfig {
    /// Every header must be present on the request with this exact
    /// value for the rule to match (`ANDed` across all entries).
    pub headers: BTreeMap<String, String>,
}

/// Per-rule algorithm choice and its own parameters.
///
/// Placed at the rule level (not per-budget), matching the maintainer's
/// own comparison on `ai#789`/`praxis#551` to `praxis#548`/`#856`'s
/// "per-rule" `shadow`/enforcement-action knobs.
#[derive(Debug, Deserialize)]
#[serde(tag = "algorithm", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum AlgorithmConfig {
    /// Exact sliding-window admission (see [`super::ledger`]): tracks
    /// usage over a continuous trailing `window`.
    SlidingWindow {
        /// Sliding window duration (e.g. `"1h"`, `"60s"`).
        window: String,
        /// Maximum tokens admitted within `window`.
        capacity: u64,
    },
    /// Token-bucket admission: `capacity` tokens available at once,
    /// continuously refilled at `refill_rate` tokens/second.
    TokenBucket {
        /// Maximum tokens held at once (the bucket's ceiling).
        capacity: u64,
        /// Tokens refilled per second, up to `capacity`.
        refill_rate: f64,
    },
}

impl AlgorithmConfig {
    /// This algorithm's configured capacity, regardless of variant.
    pub(super) fn capacity(&self) -> u64 {
        match self {
            Self::SlidingWindow { capacity, .. } | Self::TokenBucket { capacity, .. } => *capacity,
        }
    }
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

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::match_wildcard_for_single_variants,
    reason = "tests intentionally fail fast on impossible fixture states"
)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Result<TokenRateLimitConfig, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    #[test]
    fn parses_a_single_sliding_window_rule_with_no_match() {
        let cfg = parse(
            "rules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 1000\n    \
             estimate_tokens: 50\n",
        )
        .unwrap();
        assert_eq!(cfg.rules.len(), 1);
        let rule = &cfg.rules[0];
        assert_eq!(rule.name, "default");
        assert!(rule.r#match.is_none(), "a rule without match: is a catch-all");
        assert!(matches!(
            rule.algorithm,
            AlgorithmConfig::SlidingWindow { capacity: 1000, .. }
        ));
        assert_eq!(rule.estimate_tokens, 50);
    }

    #[test]
    fn parses_a_token_bucket_rule() {
        let cfg = parse(
            "rules:\n  - name: bucket-rule\n    algorithm: token_bucket\n    capacity: 200\n    refill_rate: 10.5\n    \
             estimate_tokens: 20\n",
        )
        .unwrap();
        match &cfg.rules[0].algorithm {
            AlgorithmConfig::TokenBucket { capacity, refill_rate } => {
                assert_eq!(*capacity, 200);
                assert!((*refill_rate - 10.5).abs() < f64::EPSILON);
            },
            other => panic!("expected token_bucket, got {other:?}"),
        }
    }

    #[test]
    fn parses_multiple_rules_with_mixed_algorithms_and_header_match() {
        // The customer-facing scenario this feature exists for: two apps,
        // each with their own algorithm and budget, disambiguated by a
        // shared header (e.g. x-app-id).
        let cfg = parse(
            "rules:\n\
             \x20 - name: team-alpha\n\
             \x20   match:\n\
             \x20     headers:\n\
             \x20       x-app-id: alpha\n\
             \x20   algorithm: sliding_window\n\
             \x20   window: 1h\n\
             \x20   capacity: 1000\n\
             \x20   estimate_tokens: 50\n\
             \x20 - name: team-beta\n\
             \x20   match:\n\
             \x20     headers:\n\
             \x20       x-app-id: beta\n\
             \x20   algorithm: token_bucket\n\
             \x20   capacity: 500\n\
             \x20   refill_rate: 5\n\
             \x20   estimate_tokens: 20\n",
        )
        .unwrap();
        assert_eq!(cfg.rules.len(), 2);
        assert_eq!(match_header(&cfg, 0, "x-app-id"), "alpha");
        assert!(matches!(cfg.rules[0].algorithm, AlgorithmConfig::SlidingWindow { .. }));
        assert_eq!(match_header(&cfg, 1, "x-app-id"), "beta");
        assert!(matches!(cfg.rules[1].algorithm, AlgorithmConfig::TokenBucket { .. }));
    }

    /// Fetch a header-match value off `cfg.rules[idx]` for assertions.
    fn match_header<'a>(cfg: &'a TokenRateLimitConfig, idx: usize, header: &str) -> &'a str {
        cfg.rules[idx].r#match.as_ref().unwrap().headers.get(header).unwrap()
    }

    #[test]
    fn rejects_an_empty_rules_list_shape_is_still_valid_yaml_but_filter_construction_validates_non_empty() {
        // Config-level parsing accepts an empty list (YAML shape is
        // valid); business-rule validation that at least one rule is
        // required belongs to filter construction (`from_config`), not
        // deserialization -- covered in `tests.rs`.
        let cfg = parse("rules: []\n").unwrap();
        assert!(cfg.rules.is_empty());
    }

    #[test]
    fn rejects_an_unknown_top_level_field() {
        assert!(
            parse("window: 1h\ncapacity: 100\nestimate_tokens: 5\n").is_err(),
            "the old flat (pre-rules) shape must be rejected, not silently ignored"
        );
    }

    #[test]
    fn rejects_a_rule_missing_its_algorithm_tag() {
        let err = parse("rules:\n  - name: bad\n    window: 1h\n    capacity: 100\n    estimate_tokens: 5\n")
            .expect_err("should error");
        assert!(err.to_string().contains("algorithm"), "got: {err}");
    }

    #[test]
    fn rejects_a_sliding_window_rule_missing_window() {
        let err = parse(
            "rules:\n  - name: bad\n    algorithm: sliding_window\n    capacity: 100\n    \
             estimate_tokens: 5\n",
        )
        .expect_err("should error");
        assert!(err.to_string().contains("window"), "got: {err}");
    }

    #[test]
    fn rejects_a_token_bucket_rule_missing_refill_rate() {
        let err =
            parse("rules:\n  - name: bad\n    algorithm: token_bucket\n    capacity: 100\n    estimate_tokens: 5\n")
                .expect_err("should error");
        assert!(err.to_string().contains("refill_rate"), "got: {err}");
    }

    #[test]
    fn rejects_mixing_sliding_window_and_token_bucket_fields_on_one_rule() {
        assert!(
            parse(
                "rules:\n  - name: bad\n    algorithm: sliding_window\n    window: 1h\n    capacity: 100\n    \
                 refill_rate: 5\n    estimate_tokens: 5\n"
            )
            .is_err(),
            "refill_rate is not a sliding_window field, deny_unknown_fields should reject it"
        );
    }

    #[test]
    fn algorithm_config_capacity_reads_either_variant() {
        assert_eq!(
            AlgorithmConfig::SlidingWindow {
                window: "1h".into(),
                capacity: 42
            }
            .capacity(),
            42
        );
        assert_eq!(
            AlgorithmConfig::TokenBucket {
                capacity: 7,
                refill_rate: 1.0
            }
            .capacity(),
            7
        );
    }
}
