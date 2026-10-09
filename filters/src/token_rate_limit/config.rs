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
/// This filter delivers the agreed M1/M2/M6/M7 milestone scope, but its
/// parent proposal is not yet `accepted` and open questions remain
/// (HA/clustered-Valkey failure modes, and the relationship to
/// Kuadrant's `TokenRateLimitPolicy` -- see `ai#127`). The
/// configuration surface may change between releases.
///
/// Mirrors the `rules:`/`match:` shape from the `00121_token-rate-limiting`
/// proposal in `praxis-proxy/enhancements`, scoped to this milestone's
/// static header-value matchers, per-rule algorithm choice, configurable
/// estimation strategies (M3, see [`EstimationConfig`]), M4
/// token-type weights (`default_weights` / per-rule `weights`), graduated
/// soft-limit tiers (S1), and per-rule soft over-quota enforcement
/// (`ai#1241`). CEL matchers remain deferred (see the module doc comment).
///
/// Assumes request identity has already been resolved upstream (this
/// filter doesn't authenticate callers) -- a catch-all rule (no
/// `match:`) reserves quota for every request that reaches it,
/// including probes and health checks. Scope rules with explicit
/// `match:` conditions, or place an identity/auth filter earlier in
/// the pipeline. Tracked as follow-on integration work in `grid#101`.
///
/// Observability is group-level by rule, never by user. Metrics carry only
/// bounded `rule`, `algorithm`, `backend`, `error`, `result`, and `capacity` labels;
/// accounting logs and optional OpenTelemetry spans likewise omit raw
/// subject and bucket-key values. The Prometheus contract is:
///
/// - `praxis_trl_requests_total{rule,result}` (`admitted`, `denied`, or `soft_over_quota`): budget decisions only. Soft
///   over-quota forwards are **not** reserved or reconciled (meter-only path does not debit the window). Requests
///   rejected before a decision are counted by `praxis_trl_unauthenticated_total` (401, no trusted subject) and
///   `praxis_trl_backend_errors_total` (503, fail closed) instead.
///
/// Accounting log field `outcome` on hard denials is one of: `budget_exhausted` (window/bucket capacity),
/// `key_capacity` (per-rule distinct-key cap), `invalid_key`, or `reservation_capacity`. Soft over-quota forwards
/// also use `budget_exhausted`. Admitted reservations use `reserved`.
///
/// - `praxis_trl_unauthenticated_total{rule}`
///
/// - `praxis_trl_tokens_reserved_total{rule}`
///
/// - `praxis_trl_tokens_reconciled_total{rule}`
///
/// - `praxis_trl_tokens_refunded_total{rule}`
///
/// - `praxis_trl_tokens_overage_total{rule}`
///
/// - `praxis_trl_reservations_total{rule,result}` (`reconciled` or `orphaned`)
///
/// - `praxis_trl_soft_tier_activations_total{rule,capacity}`
///
/// - `praxis_trl_backend_errors_total{rule,backend,error}`: failed reservations (the 503 path), failed reconciliation
///   enqueues, and reconciliations abandoned after their retries. `error` is one of `unavailable`, `invalid_response`,
///   or `configuration_mismatch`.
///
/// - `praxis_trl_backend_reconciliation_total{rule,backend,result}`: reconciliations completed by a Valkey worker.
///
/// - `praxis_trl_budget_remaining{rule,algorithm}`
///
/// - `praxis_trl_reservations_active{rule}`
///
/// - `praxis_trl_active_keys{rule}`
///
/// Every previous `praxis_ai_token_rate_limit_*` name has moved to this
/// prefix; no compatibility aliases are emitted.
///
/// `budget_remaining` is the remaining budget for the key of the most recent
/// admission decision on this replica (admitted or denied), and
/// `active_keys` is how many keys the backend currently retains. Both are
/// snapshots taken as decisions happen, not continuously refreshed values.
/// Like all Prometheus gauges they are f64 and saturate at the largest
/// exactly representable integer (2^53 - 1).
///
/// Gauge scope depends on the backend. With the `memory` backend every
/// gauge describes this process only, so aggregate replicas with `sum`.
/// With the `valkey` backend, `reservations_active` is scoped to the
/// namespace and algorithm, so summing it over rules double-counts;
/// aggregate with `max` across replicas and rules. `active_keys` is scoped
/// per rule, so aggregate with `max` across replicas for each rule.
/// Each replica exports the value it last observed from the shared store.
/// `budget_remaining` stays per replica and per
/// last decision on either backend: it describes whichever key that
/// replica decided last, so `max` or `sum` across replicas says little
/// beyond "some key had this much left". A replica that stops seeing
/// traffic for a rule keeps exporting its last observation until it does.
///
/// The `valkey` backend requires Valkey or Redis 7.0+ (`PEXPIRE NX`/`GT`
/// is used). The `valkey` backend keeps
/// sliding-window usage in 60 fixed sub-windows per window (one per
/// second for windows under a minute); usage leaves the window up to one
/// sub-window late, never early. Changing a window's length changes its
/// sub-window width and so starts that window's usage from zero. On the
/// sliding window, concurrent admissions on one key are not serialised,
/// so they can overshoot the budget by their combined estimates for one
/// round trip. Usage written is never lost.
///
/// The `valkey` token bucket, by contrast, serialises admissions per key
/// through an optimistic transaction: one key admits at most about one
/// request per two Valkey round trips across the whole fleet, and
/// contention shows up first as added latency, up to the 500 ms Valkey
/// timeout, then as 503s. Use a non-`global` `key` for high-throughput
/// token-bucket rules so the load spreads over many buckets.
///
/// During a rolling upgrade from the earlier scripted `valkey` backend,
/// replicas on the old and new versions keep separate state, so for one
/// window (and until old token buckets have drained) combined admissions
/// can reach about twice the budget. All `valkey` timestamps come from
/// the proxy replicas' clocks, not Valkey's: skew between replicas can
/// under-count usage at window edges by up to the skew, and a replica
/// whose clock runs fast trims other replicas' live reservations and keys
/// from the caps early.
///
/// Admissions, denials, reconciliations, and backend failures also emit
/// structured records on the `praxis_ai::token_rate_limit::accounting`
/// tracing target: `INFO` for admissions and settlements, `WARN` for
/// failures. They are on by default at `INFO`, so every admitted or denied
/// request produces one line in the operational log stream; keep only
/// failures with `runtime.log_overrides:
/// {"praxis_ai::token_rate_limit::accounting": "warn"}`, and separate them
/// from other operational logs by filtering on the `target` field. The
/// records contain bounded policy and token-count fields only. They are
/// best-effort operational audit records, not a durable billing source.
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
    ///
    /// Composite dimension *order does not matter*: compiled dimensions
    /// are sorted into a canonical order so reordering a list cannot
    /// silently reset live budgets.
    ///
    /// `ip` and `model` are as caller-controlled as `header` when they
    /// come from a forwarding header or a client-supplied model string.
    /// Pair them with `max_keys` so one client cannot fill the table.
    #[serde(default)]
    pub key: KeySpec,

    /// Soft cap on distinct budget keys retained at once, **per rule**.
    ///
    /// Bounds cardinality from per-header, per-IP, and composite
    /// keying. Defaults to [`super::MAX_KEYS`]. A new distinct key past
    /// this cap is denied (429, accounting outcome `key_capacity`)
    /// rather than growing without bound.
    ///
    /// In-process ledgers enforce the cap per rule. Valkey enforces it
    /// against the per-rule retained-key set (`{namespace}:v2:keys:{rule_hash}`,
    /// or the token-bucket equivalent).
    /// Idle in-process keys are reaped by ledger cleanup, which walks a
    /// bounded number of entries per request (including busy ones) so a
    /// single in-window key cannot pin the table at this cap.
    #[serde(default = "default_max_keys")]
    pub max_keys: usize,

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

/// Serde default for [`TokenRateLimitConfig::max_keys`].
fn default_max_keys() -> usize {
    super::MAX_KEYS
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
    /// Downstream TCP peer address, or an optional forwarding header.
    ///
    /// When `header` is set, the right-most hop after `trusted_hops`
    /// trusted proxies is used (appending LBs put client-controlled
    /// values on the left). A missing header falls back to the TCP peer
    /// so health checks and in-cluster clients are not `missing_ip`.
    /// A present but unusable header is fail-closed.
    ///
    /// The selected address is hashed, so distinct IPs (or models, via
    /// [`Self::Model`]) can fill `max_keys` the same way a caller-controlled
    /// header can. Pair high-cardinality dimensions with a tighter cap.
    Ip {
        /// Forwarding header (e.g. `x-forwarded-for`). Blank names are
        /// rejected at compile time.
        header: Option<String>,
        /// Trusted proxy hops to skip from the right of the header.
        /// `0` (default) selects the right-most hop.
        trusted_hops: u32,
        /// Optional IPv6 prefix length (1–128). When set, IPv6 addresses
        /// are masked to this prefix before hashing so a client holding
        /// a `/64` cannot rotate host bits for a fresh budget. IPv4 is
        /// unchanged. Unset keeps the historical `/128` (full address).
        ipv6_prefix: Option<u8>,
    },
    /// Model identity from `header` (default `x-model`), then the JSON
    /// body `model` field when the request is already buffered for
    /// estimation. Preferring the header keeps `key: model` from forcing
    /// `StreamBuffer` on every request.
    ///
    /// Model strings are caller-controlled; bound cardinality with
    /// `max_keys`.
    Model {
        /// Header consulted before the body `model` field.
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
/// Other dimension shapes:
///
/// ```yaml
/// key: ip
/// key:
///   ip:
///     header: x-forwarded-for
///     trusted_hops: 1          # skip the last appending proxy
///     ipv6_prefix: 64          # optional; IPv4 unchanged
/// key:
///   model:
///     header: x-model          # omit to keep the default
/// key:
///   header: x-tenant-id
///   missing: fallback
/// key:
///   - type: header
///     name: x-api-key
///     missing: reject
/// ```
///
/// Composite example (subject + model + tenant header). List order is
/// canonicalized at compile time; reordering does not reset budgets:
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
        Ok(wire.into_spec())
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
    Scalar(NamedDimension),
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
    fn into_spec(self) -> KeySpec {
        match self {
            Self::Scalar(name) => named_spec(name),
            Self::List(items) => KeySpec {
                dimensions: decode_dimensions(items),
                missing: MissingKeyPolicy::Reject,
            },
            Self::Spec(spec) => KeySpec {
                dimensions: decode_dimensions(spec.dimensions),
                missing: spec.missing,
            },
            Self::HeaderKey(map) => KeySpec {
                dimensions: vec![header_dimension(map.header)],
                missing: map.missing,
            },
            Self::ModelKey(map) => KeySpec {
                dimensions: vec![KeyDimension::Model {
                    header: map.model.header,
                }],
                missing: map.missing,
            },
            Self::IpKey(map) => KeySpec {
                dimensions: vec![ip_dimension(map.ip)],
                missing: map.missing,
            },
            Self::Dimension(item) => KeySpec {
                dimensions: vec![item.into_dimension()],
                missing: MissingKeyPolicy::Reject,
            },
        }
    }
}

/// Closed set of scalar dimension names. Enumerated so unknown strings
/// fail at parse time (see `docs/developing/type-design.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum NamedDimension {
    /// Shared bucket for every matching request.
    Global,
    /// Verified authenticated subject.
    AuthenticatedSubject,
    /// Client IP (peer address; no forwarding header).
    Ip,
    /// Model identity from the default `x-model` header, then the body.
    Model,
}

impl NamedDimension {
    /// Convert a scalar name into its dimension (no header overrides).
    fn into_dimension(self) -> KeyDimension {
        match self {
            Self::Global => KeyDimension::Global,
            Self::AuthenticatedSubject => KeyDimension::AuthenticatedSubject,
            Self::Ip => ip_dimension(IpRef::default()),
            Self::Model => KeyDimension::Model { header: None },
        }
    }
}

/// One-dimension spec from a scalar name.
fn named_spec(name: NamedDimension) -> KeySpec {
    KeySpec {
        dimensions: vec![name.into_dimension()],
        missing: MissingKeyPolicy::Reject,
    }
}

/// Decode a list of wire dimensions.
fn decode_dimensions(items: Vec<KeyDimensionDe>) -> Vec<KeyDimension> {
    items.into_iter().map(KeyDimensionDe::into_dimension).collect()
}

/// Wire form for one [`KeyDimension`].
///
/// Struct variants use `deny_unknown_fields` so a list item like
/// `{ ip: {}, header: x-forwarded-for }` cannot silently parse as a
/// header dimension, and `{ header: x-tenant-id, missing: fallback }`
/// cannot drop `missing`.
#[derive(Deserialize)]
#[serde(untagged)]
enum KeyDimensionDe {
    /// Scalar name (`global`, `authenticated_subject`, `ip`, `model`).
    Name(NamedDimension),
    /// Internally tagged `{ type: ..., ... }`.
    Tagged(TaggedDimension),
    /// `{ header: x-tenant-id }` or `{ header: x-tenant-id, missing: fallback }`.
    HeaderShortcut(HeaderShortcutMapping),
    /// `{ model: {} }` or `{ model: { header } }`.
    ModelShortcut(ModelShortcutMapping),
    /// `{ ip: {} }` or `{ ip: { header, trusted_hops, ipv6_prefix } }`.
    IpShortcut(IpShortcutMapping),
}

/// List-item `{ header: NAME, missing?: ... }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderShortcutMapping {
    /// Header name or name+missing spec.
    header: HeaderRef,
    /// Optional per-header missing policy (sibling of `header:`).
    #[serde(default)]
    missing: Option<MissingKeyPolicy>,
}

/// List-item `{ model: { header?: ... } }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelShortcutMapping {
    /// Optional model header override.
    model: ModelRef,
}

/// List-item `{ ip: { header?, trusted_hops?, ipv6_prefix? } }`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IpShortcutMapping {
    /// Optional forwarding-header override.
    ip: IpRef,
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

/// `{ ip: {} }` or `{ ip: { header: x-forwarded-for, trusted_hops: 1 } }`.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct IpRef {
    /// Forwarding header, when set.
    #[serde(default)]
    header: Option<String>,
    /// Trusted hops skipped from the right. Default `0` (right-most).
    #[serde(default)]
    trusted_hops: u32,
    /// Optional IPv6 prefix mask applied before hashing.
    #[serde(default)]
    ipv6_prefix: Option<u8>,
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
        /// Trusted hops skipped from the right.
        #[serde(default)]
        trusted_hops: u32,
        /// Optional IPv6 prefix mask.
        #[serde(default)]
        ipv6_prefix: Option<u8>,
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
    fn into_dimension(self) -> KeyDimension {
        match self {
            Self::Name(name) => name.into_dimension(),
            Self::HeaderShortcut(HeaderShortcutMapping { header, missing }) => {
                header_dimension_with_missing(header, missing)
            },
            Self::ModelShortcut(ModelShortcutMapping { model }) => KeyDimension::Model { header: model.header },
            Self::IpShortcut(IpShortcutMapping { ip }) => ip_dimension(ip),
            Self::Tagged(tagged) => match tagged {
                TaggedDimension::Global => KeyDimension::Global,
                TaggedDimension::AuthenticatedSubject => KeyDimension::AuthenticatedSubject,
                TaggedDimension::Ip {
                    header,
                    trusted_hops,
                    ipv6_prefix,
                } => ip_dimension(IpRef {
                    header,
                    trusted_hops,
                    ipv6_prefix,
                }),
                TaggedDimension::Model { header } => KeyDimension::Model { header },
                TaggedDimension::Header { name, missing } => KeyDimension::Header { name, missing },
            },
        }
    }
}

/// Convert a header shortcut into a [`KeyDimension`].
fn header_dimension(header: HeaderRef) -> KeyDimension {
    header_dimension_with_missing(header, None)
}

/// Convert a header shortcut, applying an optional sibling `missing:`.
fn header_dimension_with_missing(header: HeaderRef, sibling_missing: Option<MissingKeyPolicy>) -> KeyDimension {
    match header {
        HeaderRef::Name(name) => KeyDimension::Header {
            name,
            missing: sibling_missing,
        },
        HeaderRef::Spec { name, missing } => KeyDimension::Header {
            name,
            missing: missing.or(sibling_missing),
        },
    }
}

/// Convert an IP mapping into a [`KeyDimension`].
fn ip_dimension(ip: IpRef) -> KeyDimension {
    KeyDimension::Ip {
        header: ip.header,
        trusted_hops: ip.trusted_hops,
        ipv6_prefix: ip.ipv6_prefix,
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

    /// Graduated enforcement tiers (proposal S1). Each tier defines a
    /// usage threshold and an action (`inject` or `deny`). When the
    /// backend admits a request, every tier whose `capacity` is at or
    /// below the current usage level fires:
    ///
    /// - `inject`: the request continues and the tier's `headers` are set on the upstream request.
    /// - `deny`: hard-reject with 429 (same as M6; must be the last tier).
    ///
    /// Tiers must have strictly ascending `capacity` values. At most one
    /// `deny` tier is allowed, and it must be the last. Its `capacity`
    /// must equal the algorithm's own `capacity`.
    ///
    /// When omitted, the rule behaves as before: a single hard deny at
    /// the algorithm's `capacity`.
    #[serde(default)]
    pub tiers: Option<Vec<TierConfig>>,

    /// What happens when the algorithm denies a reservation because the
    /// token budget is exhausted (`ai#1241`). Defaults to [`EnforcementMode::Hard`]
    /// (429). Soft forwards with [`over_quota`](Self::over_quota) annotation.
    #[serde(default)]
    pub enforcement: EnforcementMode,

    /// Request-header annotation applied when [`enforcement`](Self::enforcement)
    /// is [`EnforcementMode::Soft`] and the algorithm denies the reservation
    /// for budget exhaustion. Required for `soft` (at least one static header
    /// and/or `include_remaining` / `include_used`). Rejected for `hard`.
    #[serde(default)]
    pub over_quota: Option<OverQuotaConfig>,
}

/// Per-rule action when the admission algorithm denies a reservation.
///
/// Distinct from graduated S1 `tiers` (which annotate admitted traffic
/// below capacity): this chooses hard 429 vs soft annotate when the
/// request is *over* the algorithm's token budget.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum EnforcementMode {
    /// Reject with 429 and token-denominated rate-limit response headers.
    #[default]
    Hard,
    /// Forward the request and annotate it for downstream handling (`ai#1241`).
    Soft,
}

/// Annotation surface for soft over-quota forwarding.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OverQuotaConfig {
    /// Static headers set on the upstream request when over quota.
    /// Required to be non-empty unless `include_remaining` or
    /// `include_used` is true.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,

    /// When true, also set the remaining-quota header from the backend
    /// snapshot at denial time (tokens still available in the window /
    /// bucket — often `0` when the estimate no longer fits).
    #[serde(default)]
    pub include_remaining: bool,

    /// When true, also set a used-quota header as `limit - remaining`
    /// from the same denial-time backend snapshot.
    #[serde(default)]
    pub include_used: bool,

    /// Header name for remaining tokens when `include_remaining` is true.
    /// Defaults to `X-RateLimit-Remaining-Tokens`.
    #[serde(default)]
    pub remaining_header: Option<String>,

    /// Header name for used tokens when `include_used` is true.
    /// Defaults to `X-Token-Quota-Used`.
    #[serde(default)]
    pub used_header: Option<String>,
}

/// One graduated enforcement tier (proposal S1).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TierConfig {
    /// Usage threshold at which this tier activates.
    pub capacity: u64,
    /// What happens when usage crosses this tier's threshold.
    pub action: ActionConfig,
}

/// Action to take when a tier's usage threshold is crossed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ActionConfig {
    /// Whether to continue with injected headers or hard-reject.
    #[serde(rename = "type")]
    pub action_type: ActionType,
    /// Headers to inject on the upstream request (required for
    /// `inject`, ignored for `deny`).
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

/// The type of enforcement a tier performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ActionType {
    /// Continue the request and inject the configured headers.
    Inject,
    /// Hard-reject with 429 (the existing M6 behavior).
    Deny,
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
    /// share one Valkey instance without colliding. The namespace is
    /// configured once for the whole filter, not per rule, so changing it
    /// starts a fresh accounting generation and fresh budgets for every rule
    /// in that filter. Ignored for `kind: memory`. Defaults to
    /// `"praxis:token_rate_limit"` when unset.
    ///
    /// Valkey permanently records a schema-versioned fingerprint for each
    /// namespace/rule/algorithm identity. Replicas with a different window,
    /// capacity, refill rate, reservation timeout, state bound, or compiled
    /// budget-key policy fail closed with 503 before mutating shared state.
    /// The compatibility markers use
    /// `{namespace}:v2:sw:rule:{hash}:accounting-config` for sliding windows
    /// and `{namespace}:v2:tb:rule:{hash}:accounting-config` for token
    /// buckets. The fingerprint also includes the compiled estimation
    /// strategy and effective token-type weights, but never request-derived
    /// values such as one request's `max_tokens`. On first use, the backend
    /// scans the namespace for unmarked `v2` keys before claiming a persistent
    /// generation marker; the scan is bounded and fails closed rather than
    /// adopting a partial result. Pre-marker state fails closed and leaves a
    /// persistent `{namespace}:v2:accounting-generation` tombstone that
    /// requires a new namespace generation rather than being silently
    /// adopted. This protects against quota state surviving an expiring
    /// per-rule retained-key index. The first bootstrap must run with
    /// pre-marker writers quiesced because those writers do not participate in
    /// the scan/claim protocol.
    ///
    /// To make an intentional semantic change, quiesce the old generation,
    /// cut every writer over to a new namespace generation, and only then
    /// retire the complete old namespace. Do not delete only a compatibility
    /// marker: doing so can bind a new configuration to incompatible residual
    /// quota state. Avoid serving traffic from both generations during the
    /// cutover because their budgets are independent. Changing configuration
    /// in place is deliberately rejected.
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

    fn ip_dim(header: Option<&str>) -> KeyDimension {
        KeyDimension::Ip {
            header: header.map(str::to_owned),
            trusted_hops: 0,
            ipv6_prefix: None,
        }
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
                ip_dim(Some("x-forwarded-for")),
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
        assert_eq!(cfg.key.dimensions, vec![ip_dim(Some("x-forwarded-for"))]);
        assert_eq!(cfg.key.missing, MissingKeyPolicy::Reject);
    }

    #[test]
    fn list_item_header_honours_sibling_missing() {
        let cfg = parse(&key_yaml("key:\n  - header: x-tenant-id\n    missing: fallback")).unwrap();
        assert_eq!(
            cfg.key.dimensions,
            vec![KeyDimension::Header {
                name: "x-tenant-id".into(),
                missing: Some(MissingKeyPolicy::Fallback),
            }]
        );
    }

    #[test]
    fn rejects_shortcut_list_item_with_extra_keys() {
        let err = parse(&key_yaml("key:\n  - ip: {}\n    header: x-forwarded-for")).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown field") || msg.contains("did not match"),
            "got: {msg}"
        );
    }

    #[test]
    fn parses_ip_trusted_hops_and_ipv6_prefix() {
        let cfg = parse(&key_yaml(
            "key:\n  ip:\n    header: x-forwarded-for\n    trusted_hops: 1\n    ipv6_prefix: 64",
        ))
        .unwrap();
        assert_eq!(
            cfg.key.dimensions,
            vec![KeyDimension::Ip {
                header: Some("x-forwarded-for".into()),
                trusted_hops: 1,
                ipv6_prefix: Some(64),
            }]
        );
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
