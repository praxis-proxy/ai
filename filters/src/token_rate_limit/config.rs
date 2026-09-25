// SPDX-License-Identifier: Apache-2.0
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
/// Experimental: requires the `token-rate-limit-filter` cargo feature,
/// which is off by default and activates the `experimental` marker.
/// This filter delivers the agreed M1/M2/M6 milestone scope, but its
/// parent proposal is not yet `accepted` and open questions remain
/// (HA/clustered-Valkey failure modes, and the relationship to
/// Kuadrant's `TokenRateLimitPolicy` -- see `ai#127`). The
/// configuration surface may change between releases.
///
/// Mirrors the `rules:`/`match:` shape from the `00121_token-rate-limiting`
/// proposal in `praxis-proxy/enhancements`, scoped to this milestone's
/// static header-value matchers, per-rule algorithm choice, configurable
/// estimation strategies (M3, see [`EstimationConfig`]), and M4
/// token-type weights (`default_weights` / per-rule `weights`). CEL
/// matchers and soft-limit tiers are still out of scope (see the module
/// doc comment) -- upstream itself defers those.
///
/// Assumes request identity has already been resolved upstream (this
/// filter doesn't authenticate callers) -- a catch-all rule (no
/// `match:`) reserves quota for every request that reaches it,
/// including probes and health checks. Scope rules with explicit
/// `match:` conditions, or place an identity/auth filter earlier in
/// the pipeline. Tracked as follow-on integration work in `grid#101`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenRateLimitConfig {
    /// Evaluated in order; the first rule whose `match` is satisfied (or
    /// which has no `match` at all) applies to a given request. A
    /// request satisfying no rule's `match` is not rate limited by this
    /// filter instance -- add a trailing rule with no `match` to enforce
    /// a catch-all budget instead.
    pub rules: Vec<RuleConfig>,

    /// How this filter partitions each matched rule's token budget.
    ///
    /// Accepts a scalar (`global`, `authenticated_subject`, `ip`,
    /// `model`), a list of dimensions (composite keys), a single
    /// dimension mapping (`header: x-tenant-id`), or a full spec with
    /// `dimensions` and `missing`. Defaults to one shared global bucket.
    #[serde(default)]
    pub key: KeySpec,

    /// Soft cap on distinct in-memory / Valkey budget keys retained at
    /// once, per rule. Bounds cardinality from per-header, per-IP, and
    /// composite keying. Defaults to 100000. A new distinct key past this
    /// cap is denied (429) rather than growing without bound; idle keys
    /// are reaped by the existing ledger cleanup path.
    #[serde(default)]
    pub max_keys: Option<usize>,

    /// Where every rule's admission state lives: in-process (default,
    /// one budget per gateway instance) or a shared Valkey backend (one
    /// budget shared across every gateway instance/replica). One
    /// backend for the whole filter, not per rule -- rules already
    /// share Valkey key-space isolation via `namespace`/rule-name
    /// hashing, so per-rule backend selection bought no isolation
    /// benefit, only a separate Valkey connection per rule pointed at
    /// the same URL. Revisit if a real deployment ever needs to mix
    /// in-process and Valkey rules in one filter instance.
    #[serde(default)]
    pub backend: BackendConfig,

    /// Filter-wide default per-type weights applied at reconciliation
    /// (proposal M4). Omitted types default to `1.0`. Rules may overlay
    /// individual types via [`RuleConfig::weights`]. Admission still
    /// reserves the estimation/`reserved_tokens` cost unweighted.
    #[serde(default)]
    pub default_weights: super::weights::TokenTypeWeightsConfig,
}

/// Policy applied when a key dimension cannot be resolved from the request.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MissingKeyPolicy {
    /// Fail closed: reject the request before provider contact.
    #[default]
    Reject,
    /// Drop the unresolved dimension. If nothing remains, use the global
    /// bucket (`__fallback__`) -- the #129 "header absent" behaviour.
    Fallback,
}

/// One dimension of a token-budget key (proposal M5 / ai#123).
///
/// Composite keys are an ordered list of these. A lone `Global` dimension
/// preserves the historical single-bucket-per-rule behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KeyDimension {
    /// Shared bucket for every request matching the rule.
    Global,
    /// Verified [`praxis_filter::AuthenticatedIdentity`] subject.
    AuthenticatedSubject,
    /// Downstream TCP peer address, or an optional trusted forwarding header.
    Ip {
        /// When set (e.g. `x-forwarded-for`), take the left-most IP from
        /// that header instead of `client_addr`. Only safe behind a
        /// trusted proxy that overwrites the header.
        header: Option<String>,
    },
    /// Model identity from the JSON body `model` field, then `header`
    /// (default `x-model`).
    Model {
        /// Header consulted when the body has no `model` field.
        header: Option<String>,
    },
    /// Arbitrary request header value. As trusted as whoever set the
    /// header -- pair with an upstream auth filter, do not key on a
    /// caller-controlled identity header.
    Header {
        /// Header name (case-insensitive HTTP name).
        name: String,
        /// Overrides the spec-level [`KeySpec::missing`] for this header.
        missing: Option<MissingKeyPolicy>,
    },
}

/// Filter-level budget key configuration.
///
/// YAML shapes (all equivalent for a single subject key):
///
/// ```yaml
/// key: authenticated_subject
/// key:
///   - authenticated_subject
/// key:
///   missing: reject
///   dimensions:
///     - type: authenticated_subject
/// ```
///
/// Composite example (subject + model + tenant header):
///
/// ```yaml
/// key:
///   - authenticated_subject
///   - model
///   - header: x-tenant-id
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KeySpec {
    /// Ordered dimensions joined into one opaque backend key.
    pub dimensions: Vec<KeyDimension>,
    /// Default missing-dimension policy. Per-header `missing:` overrides
    /// this for that header only.
    pub missing: MissingKeyPolicy,
}

impl Default for KeySpec {
    fn default() -> Self {
        Self {
            dimensions: vec![KeyDimension::Global],
            missing: MissingKeyPolicy::Reject,
        }
    }
}

impl<'de> Deserialize<'de> for KeySpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = KeySpecDe::deserialize(deserializer)?;
        wire.into_spec().map_err(serde::de::Error::custom)
    }
}

/// Wire form for [`KeySpec`]: scalar, list, single dimension map, or full spec.
///
/// Mapping variants that carry a sibling `missing:` are listed *before*
/// [`KeySpecDe::Dimension`] so `{ header: x-tenant-id, missing: fallback }`
/// is not swallowed by a header-only shortcut that would ignore `missing`.
/// Each of those mappings uses `deny_unknown_fields` so
/// `{ type: ip, header: x-forwarded-for }` still falls through to the
/// tagged dimension parser instead of being misread as a header key.
#[derive(Deserialize)]
#[serde(untagged)]
enum KeySpecDe {
    /// `key: authenticated_subject`
    Scalar(String),
    /// `key: [authenticated_subject, model]`
    List(Vec<KeyDimensionDe>),
    /// `key: { missing, dimensions }`
    Spec(KeySpecMapping),
    /// `key: { header: x-tenant-id }` or `{ header: x-tenant-id, missing: fallback }`
    HeaderKey(HeaderKeyMapping),
    /// `key: { model: {} }` or `{ model: { header: x-model }, missing: fallback }`
    ModelKey(ModelKeyMapping),
    /// `key: { ip: {} }` or `{ ip: { header: x-forwarded-for }, missing: fallback }`
    IpKey(IpKeyMapping),
    /// `key: { header: x-tenant-id }` (no sibling fields) or `{ type: ip, header: ... }`
    Dimension(KeyDimensionDe),
}

/// Single-header spec that may set the spec-level missing policy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderKeyMapping {
    /// Header name or name+missing spec.
    header: HeaderRef,
    /// Spec-level missing policy (per-header `missing` still wins).
    #[serde(default)]
    missing: MissingKeyPolicy,
}

/// Single-model spec that may set the spec-level missing policy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelKeyMapping {
    /// Optional model header override.
    model: ModelRef,
    /// Spec-level missing policy.
    #[serde(default)]
    missing: MissingKeyPolicy,
}

/// Single-IP spec that may set the spec-level missing policy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IpKeyMapping {
    /// Optional forwarding-header override.
    ip: IpRef,
    /// Spec-level missing policy.
    #[serde(default)]
    missing: MissingKeyPolicy,
}

/// Mapping form of [`KeySpec`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeySpecMapping {
    /// Dimension list; empty is rejected at compile time.
    #[serde(default)]
    dimensions: Vec<KeyDimensionDe>,
    /// Default missing-dimension policy.
    #[serde(default)]
    missing: MissingKeyPolicy,
}

impl KeySpecDe {
    /// Convert the wire form into a [`KeySpec`].
    fn into_spec(self) -> Result<KeySpec, String> {
        match self {
            Self::Scalar(name) => Ok(named_spec(&name)?),
            Self::List(items) => Ok(KeySpec {
                dimensions: decode_dimensions(items)?,
                missing: MissingKeyPolicy::Reject,
            }),
            Self::Spec(spec) => Ok(KeySpec {
                dimensions: decode_dimensions(spec.dimensions)?,
                missing: spec.missing,
            }),
            Self::HeaderKey(map) => Ok(KeySpec {
                dimensions: vec![header_dimension(map.header)],
                missing: map.missing,
            }),
            Self::ModelKey(map) => Ok(KeySpec {
                dimensions: vec![KeyDimension::Model {
                    header: map.model.header,
                }],
                missing: map.missing,
            }),
            Self::IpKey(map) => Ok(KeySpec {
                dimensions: vec![KeyDimension::Ip { header: map.ip.header }],
                missing: map.missing,
            }),
            Self::Dimension(item) => Ok(KeySpec {
                dimensions: vec![item.into_dimension()?],
                missing: MissingKeyPolicy::Reject,
            }),
        }
    }
}

/// One-dimension spec from a scalar name.
fn named_spec(name: &str) -> Result<KeySpec, String> {
    Ok(KeySpec {
        dimensions: vec![dimension_from_name(name)?],
        missing: MissingKeyPolicy::Reject,
    })
}

/// Decode a list of wire dimensions.
fn decode_dimensions(items: Vec<KeyDimensionDe>) -> Result<Vec<KeyDimension>, String> {
    items.into_iter().map(KeyDimensionDe::into_dimension).collect()
}

/// Wire form for one [`KeyDimension`].
#[derive(Deserialize)]
#[serde(untagged)]
enum KeyDimensionDe {
    /// Scalar name (`global`, `authenticated_subject`, `ip`, `model`).
    Name(String),
    /// Internally tagged `{ type: ..., ... }`.
    Tagged(TaggedDimension),
    /// `{ header: x-tenant-id }` or `{ header: { name, missing } }`.
    HeaderShortcut {
        /// Header name or name+missing spec.
        header: HeaderRef,
    },
    /// `{ model: {} }` or `{ model: { header } }`.
    ModelShortcut {
        /// Optional model header override.
        model: ModelRef,
    },
    /// `{ ip: {} }` or `{ ip: { header } }`.
    IpShortcut {
        /// Optional forwarding-header override.
        ip: IpRef,
    },
}

/// Header shortcut value: a name, or a name plus missing policy.
#[derive(Deserialize)]
#[serde(untagged)]
enum HeaderRef {
    /// `header: x-tenant-id`
    Name(String),
    /// `header: { name: x-tenant-id, missing: fallback }`
    Spec {
        /// Header to read.
        name: String,
        /// Optional per-header missing policy.
        #[serde(default)]
        missing: Option<MissingKeyPolicy>,
    },
}

/// `{ model: {} }` or `{ model: { header: x-model } }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRef {
    /// Header consulted when the body has no `model` field.
    #[serde(default)]
    header: Option<String>,
}

/// `{ ip: {} }` or `{ ip: { header: x-forwarded-for } }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IpRef {
    /// Trusted forwarding header, when set.
    #[serde(default)]
    header: Option<String>,
}

/// Internally tagged dimension (`type: ...`).
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum TaggedDimension {
    /// Shared bucket for every matching request.
    Global,
    /// Verified authenticated subject.
    AuthenticatedSubject,
    /// Client IP.
    Ip {
        /// Optional forwarding header.
        #[serde(default)]
        header: Option<String>,
    },
    /// Model identity.
    Model {
        /// Optional model header override.
        #[serde(default)]
        header: Option<String>,
    },
    /// Named request header.
    Header {
        /// Header to read.
        name: String,
        /// Optional per-header missing policy.
        #[serde(default)]
        missing: Option<MissingKeyPolicy>,
    },
}

impl KeyDimensionDe {
    /// Convert the wire form into a [`KeyDimension`].
    fn into_dimension(self) -> Result<KeyDimension, String> {
        match self {
            Self::Name(name) => dimension_from_name(&name),
            Self::HeaderShortcut { header } => Ok(header_dimension(header)),
            Self::ModelShortcut { model } => Ok(KeyDimension::Model { header: model.header }),
            Self::IpShortcut { ip } => Ok(KeyDimension::Ip { header: ip.header }),
            Self::Tagged(tagged) => Ok(match tagged {
                TaggedDimension::Global => KeyDimension::Global,
                TaggedDimension::AuthenticatedSubject => KeyDimension::AuthenticatedSubject,
                TaggedDimension::Ip { header } => KeyDimension::Ip { header },
                TaggedDimension::Model { header } => KeyDimension::Model { header },
                TaggedDimension::Header { name, missing } => KeyDimension::Header { name, missing },
            }),
        }
    }
}

/// Convert a header shortcut into a [`KeyDimension`].
fn header_dimension(header: HeaderRef) -> KeyDimension {
    match header {
        HeaderRef::Name(name) => KeyDimension::Header { name, missing: None },
        HeaderRef::Spec { name, missing } => KeyDimension::Header { name, missing },
    }
}

/// Parse a scalar dimension name.
fn dimension_from_name(name: &str) -> Result<KeyDimension, String> {
    match name {
        "global" => Ok(KeyDimension::Global),
        "authenticated_subject" => Ok(KeyDimension::AuthenticatedSubject),
        "ip" => Ok(KeyDimension::Ip { header: None }),
        "model" => Ok(KeyDimension::Model { header: None }),
        other => Err(format!(
            "unknown key dimension '{other}': expected global, authenticated_subject, ip, model, or a header mapping"
        )),
    }
}

/// One `rules:` entry: an optional match condition, an algorithm choice
/// with that algorithm's own parameters, and this rule's own
/// estimation/keying configuration. Backend selection is shared across
/// every rule, see [`TokenRateLimitConfig::backend`].
// `deny_unknown_fields` is deliberately omitted here: serde's flatten
// mechanism (`algorithm` below) is fundamentally incompatible with
// `deny_unknown_fields` on the containing struct -- the flattened
// enum's own fields get misreported as "unknown" because flatten
// collects remaining fields into an intermediate map before the tagged
// enum ever gets a chance to claim them. `RuleAlgorithm` itself still
// enforces `deny_unknown_fields` per-variant, so a genuinely unknown
// field (e.g. a typo, or an old flat-schema field like `window` on a
// `token_bucket` rule) is still rejected -- just attributed to the
// flattened enum's own error path instead of this struct's.
#[derive(Debug, Deserialize)]
pub(super) struct RuleConfig {
    /// Human-readable rule identifier, folded into Valkey key
    /// namespacing so distinct rules sharing one backend never collide.
    ///
    /// Renaming a live `valkey`-backed rule is therefore not a
    /// no-op for operators: it changes the Valkey key hash, so the old
    /// name's tracked budget is orphaned (left to expire on its own TTL)
    /// and the new name starts with a fresh budget. There's no
    /// migration/rename path today -- routine config hygiene (e.g.
    /// renaming `"gold"` to `"gold-tier"`) silently resets that rule's
    /// state.
    pub name: String,

    /// Static header-value match condition. Every listed header must be
    /// present on the request with an exact value match (`ANDed`) for
    /// this rule to apply. Omit entirely for a catch-all rule.
    #[serde(default)]
    pub r#match: Option<MatchConfig>,

    /// Which admission algorithm this rule enforces, and that
    /// algorithm's own parameters.
    #[serde(flatten)]
    pub algorithm: RuleAlgorithm,

    /// Fixed token cost reserved at admission time, before actual usage
    /// is known.
    ///
    /// Legacy field, retained for backward compatibility: a bare
    /// `reserved_tokens: N` is equivalent to
    /// `estimation: { strategy: fixed, fallback_estimate: N }`.
    /// Mutually exclusive with [`estimation`](Self::estimation) --
    /// specifying both on the same rule is a config error.
    #[serde(default)]
    pub reserved_tokens: Option<u64>,

    /// Configurable estimation strategy for computing the token cost
    /// reserved at admission time. Replaces the legacy `reserved_tokens`
    /// field with request-metadata-aware strategies.
    ///
    /// Mutually exclusive with [`reserved_tokens`](Self::reserved_tokens) --
    /// specifying both on the same rule is a config error. Omitting both
    /// is also an error.
    #[serde(default)]
    pub estimation: Option<EstimationConfig>,

    /// How long an admitted-but-never-reconciled reservation (lost
    /// request: timeout, connection reset, upstream crash) is tracked as
    /// active before that already-reserved-at-admission charge against
    /// its estimate becomes irreversibly locked in (sliding-window:
    /// folded into the settled total so it survives the window's normal
    /// aging-out; token-bucket: the tokens were already decremented at
    /// reserve time regardless, this only bounds how long the
    /// reservation is tracked as pending). This does **not** defer when
    /// the charge first applies -- it applies immediately at admission,
    /// same as any other reservation.
    ///
    /// Answers the proposal's still-open "lost request handling"
    /// question for this milestone. Defaults to [`DEFAULT_RESERVATION_TIMEOUT`]
    /// when unset.
    #[serde(default)]
    pub reservation_timeout: Option<String>,

    /// Optional per-rule overlay on [`TokenRateLimitConfig::default_weights`].
    /// Omitted types inherit the filter defaults (then `1.0`).
    #[serde(default)]
    pub weights: super::weights::TokenTypeWeightsConfig,
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
pub(super) enum RuleAlgorithm {
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

impl RuleAlgorithm {
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

/// Configurable estimation strategy for computing the token cost
/// reserved at admission time, per rule.
///
/// Replaces the fixed `reserved_tokens` field with request-metadata-aware
/// strategies. The operator picks one strategy per rule; all budgets on
/// that rule share the same cost model.
///
/// Experimental: the `strategy` tag is intentionally extensible —
/// future variants (e.g. `cel`) can be added without changing
/// existing configurations.
#[derive(Debug, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub(super) enum EstimationStrategy {
    /// Constant per request (equivalent to the legacy `reserved_tokens`).
    Fixed,
    /// Extracted from the request body's `max_tokens` field.
    MaxTokens,
    /// Content-Length-based input estimate plus `max_tokens`.
    InputPlusMaxTokens,
    /// `max_tokens` scaled by a per-model multiplier.
    ModelScaled,
}

/// Full estimation configuration block, combining a strategy tag with
/// shared tuning knobs.
#[derive(Debug, Deserialize)]
pub(super) struct EstimationConfig {
    /// Which strategy to use for this rule's cost estimation.
    #[serde(flatten)]
    pub strategy: EstimationStrategy,

    /// Safety-margin multiplier applied to the computed estimate.
    /// Defaults to 1.0 (no margin). Must be positive and finite.
    #[serde(default)]
    pub multiplier: Option<f64>,

    /// Token count to use when `max_tokens` is absent from the request.
    /// Required for `fixed`; optional for body-dependent strategies
    /// (if unset and the strategy can't extract a value, the request is
    /// admitted without a reservation).
    #[serde(default)]
    pub fallback_estimate: Option<u64>,

    /// Per-model multiplier map for `model_scaled` strategy.
    #[serde(default)]
    pub model_multipliers: Option<BTreeMap<String, f64>>,

    /// Default multiplier for models not listed in `model_multipliers`.
    #[serde(default)]
    pub default_multiplier: Option<f64>,

    /// Approximate bytes-per-token ratio for `input_plus_max_tokens`.
    /// Defaults to 4.0.
    #[serde(default)]
    pub bytes_per_token: Option<f64>,
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::match_wildcard_for_single_variants,
    clippy::too_many_lines,
    reason = "tests intentionally fail fast on impossible fixture states"
)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Result<TokenRateLimitConfig, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// One catch-all sliding-window rule, so tests can focus on `key:`.
    fn key_yaml(key: &str) -> String {
        format!(
            "{key}\nrules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 1000\n    \
             reserved_tokens: 50\n"
        )
    }

    #[test]
    fn parses_a_single_sliding_window_rule_with_no_match() {
        let cfg = parse(
            "rules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 1000\n    \
             reserved_tokens: 50\n",
        )
        .unwrap();
        assert_eq!(cfg.rules.len(), 1);
        let rule = &cfg.rules[0];
        assert_eq!(rule.name, "default");
        assert!(rule.r#match.is_none(), "a rule without match: is a catch-all");
        assert!(matches!(
            rule.algorithm,
            RuleAlgorithm::SlidingWindow { capacity: 1000, .. }
        ));
        assert_eq!(rule.reserved_tokens, Some(50));
        assert_eq!(cfg.key, KeySpec::default());
    }

    #[test]
    fn parses_authenticated_subject_key_source() {
        let cfg = parse(&key_yaml("key: authenticated_subject")).unwrap();
        assert_eq!(cfg.key.dimensions, vec![KeyDimension::AuthenticatedSubject]);
        assert_eq!(cfg.key.missing, MissingKeyPolicy::Reject);
    }

    #[test]
    fn parses_a_composite_subject_model_and_header_list() {
        let cfg = parse(&key_yaml(
            "key:\n  - authenticated_subject\n  - model\n  - header: x-tenant-id",
        ))
        .unwrap();
        assert_eq!(
            cfg.key.dimensions,
            vec![
                KeyDimension::AuthenticatedSubject,
                KeyDimension::Model { header: None },
                KeyDimension::Header {
                    name: "x-tenant-id".into(),
                    missing: None,
                },
            ]
        );
    }

    #[test]
    fn parses_tagged_dimensions_with_per_header_missing_override() {
        let cfg = parse(&key_yaml(
            "key:\n  missing: fallback\n  dimensions:\n    - type: header\n      name: x-api-key\n      missing: reject\n    - type: ip\n      header: x-forwarded-for",
        ))
        .unwrap();
        assert_eq!(cfg.key.missing, MissingKeyPolicy::Fallback);
        assert_eq!(
            cfg.key.dimensions,
            vec![
                KeyDimension::Header {
                    name: "x-api-key".into(),
                    missing: Some(MissingKeyPolicy::Reject),
                },
                KeyDimension::Ip {
                    header: Some("x-forwarded-for".into()),
                },
            ]
        );
    }

    #[test]
    fn parses_header_mapping_with_spec_level_missing_fallback() {
        let cfg = parse(&key_yaml("key:\n  header: x-tenant-id\n  missing: fallback")).unwrap();
        assert_eq!(cfg.key.missing, MissingKeyPolicy::Fallback);
        assert_eq!(
            cfg.key.dimensions,
            vec![KeyDimension::Header {
                name: "x-tenant-id".into(),
                missing: None,
            }]
        );
    }

    #[test]
    fn parses_tagged_ip_with_forwarding_header() {
        let cfg = parse(&key_yaml("key:\n  type: ip\n  header: x-forwarded-for")).unwrap();
        assert_eq!(
            cfg.key.dimensions,
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
            }]
        );
        assert_eq!(cfg.key.missing, MissingKeyPolicy::Reject);
    }

    #[test]
    fn rejects_unknown_key_source() {
        let result = parse(
            "key: request_header\nrules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 1000\n    reserved_tokens: 50\n",
        );

        assert!(result.is_err());
    }

    #[test]
    fn parses_a_token_bucket_rule() {
        let cfg = parse(
            "rules:\n  - name: bucket-rule\n    algorithm: token_bucket\n    capacity: 200\n    refill_rate: 10.5\n    \
             reserved_tokens: 20\n",
        )
        .unwrap();
        match &cfg.rules[0].algorithm {
            RuleAlgorithm::TokenBucket { capacity, refill_rate } => {
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
             \x20   reserved_tokens: 50\n\
             \x20 - name: team-beta\n\
             \x20   match:\n\
             \x20     headers:\n\
             \x20       x-app-id: beta\n\
             \x20   algorithm: token_bucket\n\
             \x20   capacity: 500\n\
             \x20   refill_rate: 5\n\
             \x20   reserved_tokens: 20\n",
        )
        .unwrap();
        assert_eq!(cfg.rules.len(), 2);
        assert_eq!(match_header(&cfg, 0, "x-app-id"), "alpha");
        assert!(matches!(cfg.rules[0].algorithm, RuleAlgorithm::SlidingWindow { .. }));
        assert_eq!(match_header(&cfg, 1, "x-app-id"), "beta");
        assert!(matches!(cfg.rules[1].algorithm, RuleAlgorithm::TokenBucket { .. }));
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
            parse("window: 1h\ncapacity: 100\nreserved_tokens: 5\n").is_err(),
            "the old flat (pre-rules) shape must be rejected, not silently ignored"
        );
    }

    #[test]
    fn rejects_a_rule_missing_its_algorithm_tag() {
        let err = parse("rules:\n  - name: bad\n    window: 1h\n    capacity: 100\n    reserved_tokens: 5\n")
            .expect_err("should error");
        assert!(err.to_string().contains("algorithm"), "got: {err}");
    }

    #[test]
    fn rejects_a_sliding_window_rule_missing_window() {
        let err = parse(
            "rules:\n  - name: bad\n    algorithm: sliding_window\n    capacity: 100\n    \
             reserved_tokens: 5\n",
        )
        .expect_err("should error");
        assert!(err.to_string().contains("window"), "got: {err}");
    }

    #[test]
    fn rejects_a_token_bucket_rule_missing_refill_rate() {
        let err =
            parse("rules:\n  - name: bad\n    algorithm: token_bucket\n    capacity: 100\n    reserved_tokens: 5\n")
                .expect_err("should error");
        assert!(err.to_string().contains("refill_rate"), "got: {err}");
    }

    #[test]
    fn rejects_mixing_sliding_window_and_token_bucket_fields_on_one_rule() {
        assert!(
            parse(
                "rules:\n  - name: bad\n    algorithm: sliding_window\n    window: 1h\n    capacity: 100\n    \
                 refill_rate: 5\n    reserved_tokens: 5\n"
            )
            .is_err(),
            "refill_rate is not a sliding_window field, deny_unknown_fields should reject it"
        );
    }

    #[test]
    fn algorithm_config_capacity_reads_either_variant() {
        assert_eq!(
            RuleAlgorithm::SlidingWindow {
                window: "1h".into(),
                capacity: 42
            }
            .capacity(),
            42
        );
        assert_eq!(
            RuleAlgorithm::TokenBucket {
                capacity: 7,
                refill_rate: 1.0
            }
            .capacity(),
            7
        );
    }

    #[test]
    fn parses_filter_wide_and_per_rule_weights() {
        let cfg = parse(
            "default_weights:\n\
             \x20 cached_input: 0.1\n\
             \x20 reasoning: 0.9\n\
             rules:\n\
             \x20 - name: team-alpha\n\
             \x20   algorithm: sliding_window\n\
             \x20   window: 1h\n\
             \x20   capacity: 1000\n\
             \x20   reserved_tokens: 50\n\
             \x20   weights:\n\
             \x20     cached_input: 0.05\n",
        )
        .unwrap();
        assert_eq!(cfg.default_weights.cached_input, Some(0.1));
        assert_eq!(cfg.default_weights.reasoning, Some(0.9));
        assert!(cfg.default_weights.input.is_none());
        assert_eq!(cfg.rules[0].weights.cached_input, Some(0.05));
        assert!(cfg.rules[0].weights.reasoning.is_none());
    }

    #[test]
    fn rejects_an_unknown_weight_type_name() {
        let err = parse(
            "default_weights:\n  cached: 0.1\nrules:\n  - name: default\n    algorithm: sliding_window\n    \
             window: 1h\n    capacity: 100\n    reserved_tokens: 5\n",
        )
        .expect_err("typo'd type name must fail");
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }

    #[test]
    fn omits_weights_when_unset_so_pre_m4_configs_still_parse() {
        let cfg = parse(
            "rules:\n  - name: default\n    algorithm: sliding_window\n    window: 1h\n    capacity: 1000\n    \
             reserved_tokens: 50\n",
        )
        .unwrap();
        assert_eq!(
            cfg.default_weights,
            super::super::weights::TokenTypeWeightsConfig::default()
        );
        assert_eq!(
            cfg.rules[0].weights,
            super::super::weights::TokenTypeWeightsConfig::default()
        );
    }
}
