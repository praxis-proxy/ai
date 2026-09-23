// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Token-denominated rate limiting filter.
//!
//! **Experimental.** Requires the `token-rate-limit-filter` cargo
//! feature, which is off by default and activates the `experimental`
//! marker. This filter delivers the epic's agreed M1/M2/M6/M7 scope (see
//! below), but its parent proposal (`00121_token-rate-limiting.md`) is
//! not yet `accepted`, and open questions remain: HA/clustered-Valkey
//! failure modes, and this filter's relationship to Kuadrant's
//! `TokenRateLimitPolicy` (a separate, already-shipped mechanism for
//! the same problem -- see `ai#127`). The configuration surface may
//! change between releases. Anything beyond the agreed M1/M2/M6/M7 scope
//! belongs in `praxis-proxy/experimental` first, not here -- see
//! `grid#101`.
//!
//! Implements the agreed M1/M2/M6 core of the token rate limiting
//! proposal (`00121_token-rate-limiting.md` in `praxis-proxy/enhancements`,
//! tracked by epic `ai#121`): an ordered list of `rules`, each an optional
//! static header-value match condition ("static matchers")
//! bound to an independently-chosen admission algorithm and its own
//! token budget, reservation-based admission reconciled against actual
//! provider-reported usage, and standard 429 responses with
//! token-denominated rate limit headers.
//!
//! Two algorithms are supported per rule, chosen via `algorithm:`:
//!
//! - `sliding_window` (see [`ledger`]): exact sliding-window admission, adapted from nerdalert's
//!   `poc/distributed-token-rate-limit-demo` spike branch. Tracks usage over a continuous trailing `window`.
//! - `token_bucket` (see [`token_bucket_ledger`]): continuous refill up to `capacity` at `refill_rate` tokens/second,
//!   reusing the refill formula from Praxis's own lock-free `traffic_management::token_bucket`, extended with the
//!   reserve/reconcile split this filter needs.
//!
//! Both sit behind the same pluggable [`backend`] trait, so either
//! algorithm runs in-process (default) or against a shared Valkey
//! backend (`backend: {kind: valkey}`) for state shared across gateway
//! instances/replicas -- see `praxis-proxy/grid#83` for the fuller
//! Valkey-backend spec this milestone is a narrower slice of.
//!
//! Per-rule algorithm choice, rather than one fixed algorithm for the
//! whole filter, mirrors the `GuardrailsFilter`'s own `rules: Vec<RuleConfig>`
//! architecture and answers the maintainer's own framing on
//! `ai#789`/`praxis#551` ("this looks like a per-rule choice, similar to
//! `shadow`/enforcement-action knobs elsewhere").
//!
//! Deliberately deferred, pending the proposal's own open design questions:
//!
//! - **CEL-expression matchers**: only static, exact header-value equality matching is implemented (overlapping
//!   `praxis#189`/`#232`).
//! - **Composite/multi-dimension keys, per-model keys**: flagged as TBD under the proposal's own M5 goal (see
//!   `ai#123`/`ai#232`). This filter supports either one global bucket per rule or one bucket per trusted
//!   [`AuthenticatedIdentity`] subject. Arbitrary header-derived and compound keys remain deferred.
//! - **Configurable estimation (M3)**: implemented -- per-rule `estimation:` block with pluggable strategies (`fixed`,
//!   `max_tokens`, `input_plus_max_tokens`, `model_scaled`). See [`config::EstimationConfig`].
//! - **Multiple budgets per rule, soft-limit tiers**: the proposal allows several `token_budgets` (e.g. hourly + daily)
//!   and graduated tiers per rule; this milestone supports one budget per rule with graduated soft-limit tiers
//!   (`inject` action) and a hard deny at capacity (`deny` action) — see [`config::TierConfig`] and proposal S1.
//! - **Observability (M7)**: implemented by `ai#883`: bounded Prometheus metrics, privacy-safe accounting records, and
//!   an optional request-scoped decision span when the `opentelemetry` feature is enabled.
//! - **Billing-grade metering (S3)**: still out of scope. Accounting records are best-effort operational audit data,
//!   not a durable billing ledger.
//! - **Trust boundary, non-inference traffic scoping**: this filter assumes request identity has already been resolved
//!   upstream (the proposal's own Non-Goals) and does not itself authenticate callers or exempt probes/health
//!   checks/malformed requests from a catch-all rule's reservation. Scope rules with explicit `match:` conditions, or
//!   place an identity/auth filter earlier in the pipeline. Tracked as follow-on integration work in `grid#101`.
//!
//! `X-RateLimit-*` headers are emitted on 429 rejection only, matching
//! the validated pattern on the source spike branch: computing
//! "remaining" for every successful response would need an extra read
//! on the Valkey path (doubling backend round-trips per request), so
//! both backends behave identically here rather than diverging by
//! backend.
//!
//! Token-type-aware accounting (M4) applies configurable per-type weights at
//! reconciliation (see [`weights`]). Cache read/write are partitioned out of
//! `token.input`; reasoning is nested in `token.output` on OpenAI/Anthropic and
//! additive on Google. Any remainder of `token.total` not covered by the typed
//! parents (Bedrock Converse hiding cache in `total`) is charged at the input
//! weight. Typed parents missing falls back to `token.total` at
//! weight 1.0; if that is missing too, the reservation estimate stands.
//! Admission still reserves the estimation/`reserved_tokens` cost unweighted.
//!
//! Depends on `token_count` running earlier in the response phase to
//! populate `token.*` in `filter_metadata`; if that metadata is
//! absent when the response completes, the reservation is left as
//! final rather than guessed at.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

mod backend;
mod config;
mod ledger;
mod token_bucket_ledger;
mod weights;

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
    time::Instant,
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::header::HeaderName;
use metrics::{counter, gauge};
use praxis_ai_apis::hash::Sha256;
use praxis_filter::{
    AuthenticatedIdentity, BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    TrustedHeaderMutation, parse_filter_config,
};

use self::{
    backend::{
        BackendError, BackendReserve, BackendSettlement, CleanupReport, InMemoryTokenBucketBackend,
        InMemoryTokenRateLimitBackend, ReconcileRequest, ReserveRequest, TokenRateLimitStateBackend,
        ValkeyBackendConfig, ValkeyEval, ValkeyTokenBucketBackend, ValkeyTokenBucketConfig,
        ValkeyTokenRateLimitBackend,
    },
    config::{
        ActionType, BackendConfig, BackendKind, DEFAULT_RESERVATION_TIMEOUT, EstimationConfig, EstimationStrategy,
        KeySource, MatchConfig, RuleAlgorithm, RuleConfig, TierConfig, TokenRateLimitConfig,
    },
    ledger::{Budget, Ledger, LedgerConfig},
    token_bucket_ledger::{TokenBucketConfig, TokenBucketLedger},
    weights::{TokenWeights, UsageCounts, parse_u64_meta, weighted_cost},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key stashing this request's reservation ID, read back during
/// response-phase reconciliation.
const META_RESERVATION_ID: &str = "token_rate_limit.reservation_id";

/// Metadata key stashing the resolved bucket key, read back in
/// reconciliation so it operates on the same budget `on_request` reserved
/// from.
const META_BUCKET_KEY: &str = "token_rate_limit.bucket_key";

/// Metadata key stashing the index of the [`CompiledRule`] that admitted
/// this request, so reconciliation settles against the same rule's
/// backend/estimate even when other rules exist.
const META_RULE_INDEX: &str = "token_rate_limit.rule_index";

/// Metadata key stashing this request's computed estimate, so
/// reconciliation can use the actual per-request estimate (not just
/// the compiled default) for its settlement math.
const META_ESTIMATE: &str = "token_rate_limit.estimate";

/// The budget key used by the backward-compatible global key mode.
const FALLBACK_KEY: &str = "__fallback__";

/// Bound on distinct budget keys retained at once, per rule.
///
/// Authenticated-subject keying can create one entry per verified subject.
/// This mirrors the soft cap `rate_limit` uses for per-IP entries.
const MAX_KEYS: usize = 100_000;

/// Bound on a single budget key's length.
const MAX_KEY_LENGTH: usize = 256;

/// Bound on reservations awaiting reconciliation across all keys, per rule.
const MAX_ACTIVE_RESERVATIONS: usize = 200_000;

/// Largest exact integer representable by Prometheus's f64 gauge values and
/// by Valkey's Lua numeric type. Aggregate remaining-budget gauges saturate
/// here instead of wrapping or reporting backend-dependent precision loss.
const MAX_REPORTED_REMAINING: u64 = 9_007_199_254_740_991;

/// Maximum request body bytes buffered for body-dependent estimation
/// strategies. 2 MiB -- generous enough for the largest realistic
/// inference request body, small enough not to materially affect
/// memory pressure.
const MAX_ESTIMATION_BODY_BYTES: usize = 2_097_152;

/// Rate limit header: configured token budget.
///
/// Uses the `-Tokens` suffix per `ai#124`'s spec, distinct from the
/// existing `rate_limit` filter's unsuffixed `X-RateLimit-Limit`, to
/// avoid a header collision when both filters run in the same
/// pipeline.
const HEADER_RATELIMIT_LIMIT_TOKENS: &str = "X-RateLimit-Limit-Tokens";

/// Rate limit header: remaining tokens (always `0` -- only sent on 429).
const HEADER_RATELIMIT_REMAINING_TOKENS: &str = "X-RateLimit-Remaining-Tokens";

/// Rate limit header: seconds until another admission attempt may succeed.
const HEADER_RATELIMIT_RESET: &str = "X-RateLimit-Reset-Tokens";

/// Resolved, ready-to-use form of the filter-level `backend:` config:
/// either every rule uses in-process state, or every rule shares one
/// already-open Valkey connection (see [`ValkeyEval`]'s doc comment for
/// why this is built once and `Clone`d, not once per rule).
enum BackendResource {
    /// Every rule gets its own in-process ledger (the default).
    Memory,
    /// Every rule shares this one Valkey connection, differentiated by
    /// `namespace`/rule-name key hashing. Boxed: `ValkeyEval` embeds a
    /// `redis::Client`/`ConnectionInfo`, large enough that an unboxed
    /// field here would size the whole enum (including the zero-data
    /// `Memory` variant) up to match it.
    Valkey {
        /// Filter-shared connection, `Clone`d into each Valkey-backed
        /// rule's own backend.
        valkey: Box<ValkeyEval>,
        /// Key namespace prefix, see [`BackendConfig::namespace`].
        namespace: String,
    },
}

/// Resolve the filter's `backend:` block into a [`BackendResource`],
/// opening the Valkey connection once up front if configured.
///
/// # Errors
///
/// Returns [`FilterError`] if `backend.kind: valkey` is set without a
/// `url`, or the URL fails to parse/expand.
fn build_backend_resource(backend: &BackendConfig) -> Result<BackendResource, FilterError> {
    match backend.kind {
        BackendKind::Memory => Ok(BackendResource::Memory),
        BackendKind::Valkey => {
            let url = backend
                .url
                .as_deref()
                .ok_or("token_rate_limit: backend.url is required for backend.kind: valkey")?;
            let url = expand_backend_url(url)?;
            let namespace = backend
                .namespace
                .clone()
                .unwrap_or_else(|| "praxis:token_rate_limit".to_owned());
            let valkey = Box::new(ValkeyEval::new(url)?);
            Ok(BackendResource::Valkey { valkey, namespace })
        },
    }
}

/// Build the configured state backend for a `sliding_window` rule
/// (in-process ledger, or a handle onto the filter's shared Valkey
/// connection).
///
/// # Errors
///
/// Returns [`FilterError`] if the ledger config is invalid.
fn build_sliding_window_backend(
    backend: &BackendResource,
    rule_name: &str,
    budgets: Vec<Budget>,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match backend {
        BackendResource::Memory => {
            let ledger = Ledger::new(LedgerConfig {
                budgets,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_key_length: MAX_KEY_LENGTH,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })
            .map_err(|error| format!("token_rate_limit: rule '{rule_name}': {error}"))?;
            Ok(Arc::new(InMemoryTokenRateLimitBackend::new(ledger)))
        },
        BackendResource::Valkey { valkey, namespace } => {
            Ok(Arc::new(ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
                valkey: (**valkey).clone(),
                namespace: namespace.clone(),
                rule: rule_name.to_owned(),
                budgets,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })))
        },
    }
}

/// Build the configured state backend for a `token_bucket` rule
/// (in-process ledger, or a handle onto the filter's shared Valkey
/// connection).
///
/// # Errors
///
/// Returns [`FilterError`] if the ledger config is invalid.
fn build_token_bucket_backend(
    backend: &BackendResource,
    rule_name: &str,
    capacity: u64,
    refill_rate: f64,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match backend {
        BackendResource::Memory => {
            let ledger = TokenBucketLedger::new(TokenBucketConfig {
                capacity,
                refill_rate,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_key_length: MAX_KEY_LENGTH,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })
            .map_err(|error| format!("token_rate_limit: rule '{rule_name}': {error}"))?;
            Ok(Arc::new(InMemoryTokenBucketBackend::new(ledger)))
        },
        BackendResource::Valkey { valkey, namespace } => {
            Ok(Arc::new(ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                valkey: (**valkey).clone(),
                namespace: namespace.clone(),
                rule: rule_name.to_owned(),
                capacity,
                refill_rate,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })?))
        },
    }
}

// -----------------------------------------------------------------------------
// Estimation
// -----------------------------------------------------------------------------

/// Minimal serde struct for probing request-body fields needed by
/// body-dependent estimation strategies, without deserializing the
/// entire request payload.
#[derive(serde::Deserialize)]
struct BodyProbe {
    /// OpenAI-style `max_tokens` field (preferred over `max_completion_tokens`).
    max_tokens: Option<u64>,
    /// OpenAI-style `max_completion_tokens` fallback field.
    max_completion_tokens: Option<u64>,
    /// Model identifier, used by `model_scaled` strategy.
    model: Option<String>,
}

/// Compiled, ready-to-use estimation strategy for one rule.
///
/// Built once at filter construction from the deserialized
/// [`EstimationConfig`]; all per-request math runs against this
/// pre-validated form.
enum CompiledEstimation {
    /// Constant per request (the legacy `reserved_tokens` path, plus
    /// the `estimation: { strategy: fixed }` equivalent).
    Fixed {
        /// Pre-computed token estimate (already incorporates any multiplier).
        estimate: u64,
    },

    /// From request body `max_tokens` (or `max_completion_tokens`).
    MaxTokens {
        /// Fallback when body has no `max_tokens`/`max_completion_tokens`.
        fallback_estimate: Option<u64>,
        /// Safety-margin multiplier applied to the extracted value.
        multiplier: f64,
    },

    /// Content-Length-based input estimate plus `max_tokens` from body.
    InputPlusMaxTokens {
        /// Fallback for the output portion when `max_tokens` is absent.
        fallback_estimate: Option<u64>,
        /// Safety-margin multiplier applied to the total.
        multiplier: f64,
        /// Approximate bytes-per-token ratio for input estimation.
        bytes_per_token: f64,
    },

    /// `max_tokens` scaled by a per-model multiplier.
    ModelScaled {
        /// Fallback when body has no `max_tokens`/`max_completion_tokens`.
        fallback_estimate: Option<u64>,
        /// Per-model multiplier overrides.
        model_multipliers: BTreeMap<String, f64>,
        /// Multiplier for models not in `model_multipliers`.
        default_multiplier: f64,
    },
}

impl CompiledEstimation {
    /// Whether this strategy requires access to the request body.
    fn needs_body(&self) -> bool {
        !matches!(self, Self::Fixed { .. })
    }

    /// Compute the token estimate for a single request.
    ///
    /// Returns `None` when the strategy cannot extract the needed
    /// value (e.g. `max_tokens` absent from the body) and no
    /// `fallback_estimate` is configured -- the caller should admit
    /// without a reservation in that case.
    fn estimate(&self, headers: &http::HeaderMap, body_probe: Option<&BodyProbe>) -> Option<u64> {
        match self {
            Self::Fixed { estimate } => Some(*estimate),
            Self::MaxTokens {
                fallback_estimate,
                multiplier,
            } => {
                let raw = extract_max_tokens(body_probe, *fallback_estimate)?;
                Some(apply_multiplier(raw, *multiplier))
            },
            Self::InputPlusMaxTokens {
                fallback_estimate,
                multiplier,
                bytes_per_token,
            } => {
                let input = content_length_tokens(headers, *bytes_per_token);
                let output = extract_max_tokens(body_probe, *fallback_estimate)?;
                Some(apply_multiplier(input.saturating_add(output), *multiplier))
            },
            Self::ModelScaled {
                fallback_estimate,
                model_multipliers,
                default_multiplier,
            } => {
                let raw = extract_max_tokens(body_probe, *fallback_estimate)?;
                let model_mult = resolve_model_multiplier(body_probe, headers, model_multipliers, *default_multiplier);
                Some(apply_multiplier(raw, model_mult))
            },
        }
    }
}

/// Extract `max_tokens` (preferred) or `max_completion_tokens` from a
/// body probe, falling back to `fallback` if neither is present.
fn extract_max_tokens(body_probe: Option<&BodyProbe>, fallback: Option<u64>) -> Option<u64> {
    body_probe
        .and_then(|bp| bp.max_tokens.or(bp.max_completion_tokens))
        .or(fallback)
}

/// Apply a safety-margin `multiplier` to a raw token count, rounding up.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "token estimates are bounded by config-validated capacity (u64-safe)"
)]
fn apply_multiplier(raw: u64, multiplier: f64) -> u64 {
    ((raw as f64) * multiplier).ceil() as u64
}

/// Estimate input tokens from `Content-Length` and `bytes_per_token`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "token estimates are bounded by config-validated capacity (u64-safe)"
)]
fn content_length_tokens(headers: &http::HeaderMap, bytes_per_token: f64) -> u64 {
    let content_length = headers
        .get(http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    (content_length as f64 / bytes_per_token).ceil() as u64
}

/// Resolve the model multiplier: body `model` field preferred, then
/// `x-model` header, then `default_multiplier`.
fn resolve_model_multiplier(
    body_probe: Option<&BodyProbe>,
    headers: &http::HeaderMap,
    model_multipliers: &BTreeMap<String, f64>,
    default_multiplier: f64,
) -> f64 {
    let model_name = body_probe
        .and_then(|bp| bp.model.as_deref())
        .or_else(|| headers.get("x-model").and_then(|v| v.to_str().ok()));
    model_name
        .and_then(|name| model_multipliers.get(name))
        .copied()
        .unwrap_or(default_multiplier)
}

// -----------------------------------------------------------------------------
// CompiledRule
// -----------------------------------------------------------------------------

/// One `rules:` entry, fully resolved: its match condition (if any),
/// backend, estimation, and graduated enforcement tiers (S1).
struct CompiledRule {
    /// Human-readable identifier, used in metrics labels and error
    /// messages, and folded into Valkey key namespacing.
    name: String,

    /// Static header-value match condition. `None` matches every
    /// request unconditionally (a catch-all rule).
    matcher: Option<HeaderMatchers>,

    /// This rule's own admission state: in-process, or shared via
    /// Valkey; sliding-window or token-bucket.
    backend: Arc<dyn TokenRateLimitStateBackend>,

    /// Compiled estimation strategy for this rule.
    estimation: CompiledEstimation,

    /// Resolved per-type weights for this rule (filter defaults overlaid
    /// with the rule's `weights:`). Applied only at reconciliation.
    weights: TokenWeights,

    /// Graduated enforcement tiers (proposal S1), sorted by ascending
    /// capacity. The last tier may be a `deny` tier (matching the
    /// algorithm's capacity). When empty, the rule has a single implicit
    /// deny at the algorithm's capacity (the pre-S1 default).
    tiers: Vec<CompiledTier>,

    /// Unique header names used by any `inject` tier in this rule.
    /// Stripped from inbound requests on admission so that clients
    /// cannot spoof tier signals.
    inject_header_names: Vec<HeaderName>,
}

/// One graduated enforcement tier, resolved at config time.
struct CompiledTier {
    /// Usage threshold at which this tier activates.
    capacity: u64,
    /// What happens when usage crosses this threshold.
    action: CompiledAction,
}

/// Pre-validated action for one tier.
enum CompiledAction {
    /// Continue the request with these headers injected upstream.
    Inject {
        /// Header name-value pairs to set on the upstream request.
        headers: Vec<(HeaderName, http::HeaderValue)>,
    },
    /// Hard-reject with 429 (identical to the existing M6 behavior).
    Deny,
}

impl CompiledRule {
    /// Whether `headers` satisfies this rule's match condition (or the
    /// rule is a catch-all with no condition at all).
    fn matches(&self, headers: &http::HeaderMap) -> bool {
        self.matcher.as_deref().is_none_or(|conditions| {
            conditions
                .iter()
                .all(|(name, value)| headers.get(name).and_then(|v| v.to_str().ok()) == Some(value.as_str()))
        })
    }
}

/// Validate a rule's `capacity`/`reserved_tokens` bounds and resolve its
/// `reservation_timeout` string to milliseconds.
///
/// # Errors
///
/// Returns [`FilterError`] if `capacity` is zero, exceeds the Lua
/// `f64` safe-integer bound, `reserved_tokens` is zero or exceeds
/// `capacity`, or `reservation_timeout` isn't a valid duration.
fn validate_rule_bounds(rule: &RuleConfig, capacity: u64) -> Result<u64, FilterError> {
    validate_capacity_safe_integer_bound(&rule.name, capacity)?;
    if let Some(reserved) = rule.reserved_tokens {
        if reserved == 0 {
            return Err(format!(
                "token_rate_limit: rule '{}': reserved_tokens must be greater than 0",
                rule.name
            )
            .into());
        }
        if reserved > capacity {
            return Err(format!(
                "token_rate_limit: rule '{}': reserved_tokens must not exceed capacity",
                rule.name
            )
            .into());
        }
    }
    parse_duration_ms(
        rule.reservation_timeout
            .as_deref()
            .unwrap_or(DEFAULT_RESERVATION_TIMEOUT),
    )
}

/// Reject a zero `capacity`, or one beyond the Lua `f64` safe-integer
/// bound.
///
/// `token_bucket_ledger` re-checks this same bound on its own
/// construction path (see `MAX_F64_SAFE_INTEGER`'s doc comment), but
/// `ledger` (`sliding_window`) has no such gate downstream -- checking
/// it here, before either algorithm's backend is built, closes that
/// gap for both.
fn validate_capacity_safe_integer_bound(rule_name: &str, capacity: u64) -> Result<(), FilterError> {
    if capacity == 0 {
        return Err(format!("token_rate_limit: rule '{rule_name}': capacity must be greater than 0").into());
    }
    if capacity > token_bucket_ledger::MAX_F64_SAFE_INTEGER {
        return Err(format!(
            "token_rate_limit: rule '{rule_name}': capacity must not exceed {} (2^53)",
            token_bucket_ledger::MAX_F64_SAFE_INTEGER
        )
        .into());
    }
    Ok(())
}

/// A compiled `match: {headers: ...}` condition: an ordered list of
/// header-name/expected-value pairs that must all match (see
/// [`CompiledRule::matches`]).
type HeaderMatchers = Vec<(HeaderName, String)>;

/// Parse a rule's optional `match: {headers: ...}` block into concrete
/// [`HeaderName`]s, validating each header name along the way.
///
/// # Errors
///
/// Returns [`FilterError`] if any header name is invalid.
fn compile_matcher(rule_name: &str, r#match: Option<MatchConfig>) -> Result<Option<HeaderMatchers>, FilterError> {
    r#match
        .map(|m| {
            m.headers
                .into_iter()
                .map(|(name, value)| {
                    HeaderName::try_from(name.as_str())
                        .map(|header_name| (header_name, value))
                        .map_err(|error| {
                            FilterError::from(format!(
                                "token_rate_limit: rule '{rule_name}': invalid match header '{name}': {error}"
                            ))
                        })
                })
                .collect::<Result<Vec<_>, FilterError>>()
        })
        .transpose()
}

/// Build this rule's state backend per its chosen `algorithm:` variant.
///
/// # Errors
///
/// See [`build_sliding_window_backend`]/[`build_token_bucket_backend`].
fn build_rule_backend(
    algorithm: &RuleAlgorithm,
    backend: &BackendResource,
    rule_name: &str,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match algorithm {
        RuleAlgorithm::SlidingWindow { window, capacity } => {
            let window_ms = parse_duration_ms(window)?;
            let budgets = vec![Budget {
                window_ms,
                capacity: *capacity,
            }];
            build_sliding_window_backend(backend, rule_name, budgets, reservation_timeout_ms)
        },
        RuleAlgorithm::TokenBucket { capacity, refill_rate } => {
            build_token_bucket_backend(backend, rule_name, *capacity, *refill_rate, reservation_timeout_ms)
        },
    }
}

/// Validate and compile the estimation configuration for one rule,
/// enforcing mutual exclusion between `reserved_tokens` and `estimation`.
fn compile_estimation(
    rule_name: &str,
    reserved_tokens: Option<u64>,
    estimation: Option<EstimationConfig>,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    if reserved_tokens.is_some() && estimation.is_some() {
        return Err(format!(
            "token_rate_limit: rule '{rule_name}': cannot specify both reserved_tokens and estimation"
        )
        .into());
    }
    if let Some(est) = estimation {
        compile_estimation_config(rule_name, est, capacity)
    } else if let Some(reserved) = reserved_tokens {
        Ok(CompiledEstimation::Fixed { estimate: reserved })
    } else {
        Err(format!("token_rate_limit: rule '{rule_name}': must have either reserved_tokens or estimation").into())
    }
}

/// Compile a deserialized [`EstimationConfig`] into a [`CompiledEstimation`],
/// validating all strategy-specific invariants.
fn compile_estimation_config(
    rule_name: &str,
    est: EstimationConfig,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    let multiplier = est.multiplier.unwrap_or(1.0);
    validate_positive_finite(rule_name, "estimation.multiplier", multiplier)?;
    match est.strategy {
        EstimationStrategy::Fixed => compile_fixed_estimation(rule_name, &est, multiplier, capacity),
        EstimationStrategy::MaxTokens => compile_max_tokens_estimation(rule_name, &est, multiplier, capacity),
        EstimationStrategy::InputPlusMaxTokens => compile_input_plus_estimation(rule_name, &est, multiplier, capacity),
        EstimationStrategy::ModelScaled => compile_model_scaled_estimation(rule_name, est, multiplier, capacity),
    }
}

/// Compile a `fixed` estimation strategy.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "token estimates are bounded by config-validated capacity (u64-safe)"
)]
fn compile_fixed_estimation(
    rule_name: &str,
    est: &EstimationConfig,
    multiplier: f64,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    reject_unused_estimation_fields(
        rule_name,
        "fixed",
        &[
            ("model_multipliers", est.model_multipliers.is_some()),
            ("default_multiplier", est.default_multiplier.is_some()),
            ("bytes_per_token", est.bytes_per_token.is_some()),
        ],
    )?;
    let estimate = est.fallback_estimate.ok_or_else(|| {
        FilterError::from(format!(
            "token_rate_limit: rule '{rule_name}': fixed strategy requires fallback_estimate"
        ))
    })?;
    let effective = ((estimate as f64) * multiplier).ceil() as u64;
    if effective == 0 {
        return Err(
            format!("token_rate_limit: rule '{rule_name}': effective fixed estimate must be greater than 0").into(),
        );
    }
    if effective > capacity {
        return Err(
            format!("token_rate_limit: rule '{rule_name}': effective fixed estimate must not exceed capacity").into(),
        );
    }
    Ok(CompiledEstimation::Fixed { estimate: effective })
}

/// Compile a `max_tokens` estimation strategy.
fn compile_max_tokens_estimation(
    rule_name: &str,
    est: &EstimationConfig,
    multiplier: f64,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    reject_unused_estimation_fields(
        rule_name,
        "max_tokens",
        &[
            ("model_multipliers", est.model_multipliers.is_some()),
            ("default_multiplier", est.default_multiplier.is_some()),
            ("bytes_per_token", est.bytes_per_token.is_some()),
        ],
    )?;
    validate_fallback_within_capacity(rule_name, est.fallback_estimate, capacity)?;
    Ok(CompiledEstimation::MaxTokens {
        fallback_estimate: est.fallback_estimate,
        multiplier,
    })
}

/// Compile an `input_plus_max_tokens` estimation strategy.
fn compile_input_plus_estimation(
    rule_name: &str,
    est: &EstimationConfig,
    multiplier: f64,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    reject_unused_estimation_fields(
        rule_name,
        "input_plus_max_tokens",
        &[
            ("model_multipliers", est.model_multipliers.is_some()),
            ("default_multiplier", est.default_multiplier.is_some()),
        ],
    )?;
    let bytes_per_token = est.bytes_per_token.unwrap_or(4.0);
    validate_positive_finite(rule_name, "estimation.bytes_per_token", bytes_per_token)?;
    validate_fallback_within_capacity(rule_name, est.fallback_estimate, capacity)?;
    Ok(CompiledEstimation::InputPlusMaxTokens {
        fallback_estimate: est.fallback_estimate,
        multiplier,
        bytes_per_token,
    })
}

/// Compile a `model_scaled` estimation strategy.
fn compile_model_scaled_estimation(
    rule_name: &str,
    est: EstimationConfig,
    multiplier: f64,
    capacity: u64,
) -> Result<CompiledEstimation, FilterError> {
    reject_unused_estimation_fields(
        rule_name,
        "model_scaled",
        &[("bytes_per_token", est.bytes_per_token.is_some())],
    )?;
    let default_multiplier = est.default_multiplier.unwrap_or(1.0) * multiplier;
    validate_positive_finite(rule_name, "estimation.default_multiplier", default_multiplier)?;
    let model_multipliers: BTreeMap<String, f64> = est
        .model_multipliers
        .unwrap_or_default()
        .into_iter()
        .map(|(model, mult)| (model, mult * multiplier))
        .collect();
    for (model, &mult) in &model_multipliers {
        if !mult.is_finite() || mult <= 0.0 {
            return Err(format!(
                "token_rate_limit: rule '{rule_name}': model_multipliers['{model}'] must be finite and positive"
            )
            .into());
        }
    }
    validate_fallback_within_capacity(rule_name, est.fallback_estimate, capacity)?;
    Ok(CompiledEstimation::ModelScaled {
        fallback_estimate: est.fallback_estimate,
        model_multipliers,
        default_multiplier,
    })
}

/// Reject strategy-irrelevant `estimation:` fields so a leftover knob from
/// a previous strategy is a config error instead of a silent no-op.
fn reject_unused_estimation_fields(
    rule_name: &str,
    strategy: &str,
    unused: &[(&str, bool)],
) -> Result<(), FilterError> {
    let names: Vec<&str> = unused
        .iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| *name)
        .collect();
    if names.is_empty() {
        return Ok(());
    }
    Err(format!(
        "token_rate_limit: rule '{rule_name}': {strategy} strategy does not accept {}",
        names.join(", ")
    )
    .into())
}

/// Validate that a named f64 parameter is finite and positive.
fn validate_positive_finite(rule_name: &str, param: &str, value: f64) -> Result<(), FilterError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("token_rate_limit: rule '{rule_name}': {param} must be finite and positive").into());
    }
    Ok(())
}

/// Validate that a `fallback_estimate`, if set, doesn't exceed `capacity`.
fn validate_fallback_within_capacity(rule_name: &str, fallback: Option<u64>, capacity: u64) -> Result<(), FilterError> {
    if let Some(fb) = fallback
        && fb > capacity
    {
        return Err(format!("token_rate_limit: rule '{rule_name}': fallback_estimate must not exceed capacity").into());
    }
    Ok(())
}

/// Validate and compile the optional graduated tier list for one rule.
///
/// When `tiers` is `None`, returns an empty `Vec` — the pre-S1 default
/// behavior (single implicit deny at the algorithm's `capacity`).
///
/// # Errors
///
/// Returns [`FilterError`] if capacities are not strictly ascending, the
/// deny tier is not the last, its capacity doesn't match the algorithm's
/// capacity, an inject tier has no headers, or a header name is invalid.
fn compile_tiers(
    rule_name: &str,
    tiers: Option<Vec<TierConfig>>,
    algorithm_capacity: u64,
) -> Result<Vec<CompiledTier>, FilterError> {
    let Some(tiers) = tiers else {
        return Ok(Vec::new());
    };
    if tiers.is_empty() {
        return Err(format!("token_rate_limit: rule '{rule_name}': tiers must not be empty when specified").into());
    }

    let mut compiled = Vec::with_capacity(tiers.len());
    let mut prev_capacity = 0_u64;
    let mut saw_deny = false;

    for (index, tier) in tiers.into_iter().enumerate() {
        validate_tier_ordering(rule_name, index, tier.capacity, prev_capacity, saw_deny)?;
        prev_capacity = tier.capacity;
        let action = compile_tier_action(rule_name, index, &tier, algorithm_capacity, &mut saw_deny)?;
        compiled.push(CompiledTier {
            capacity: tier.capacity,
            action,
        });
    }
    Ok(compiled)
}

/// Collect the de-duplicated set of header names across all `inject`
/// tiers. These are stripped from the inbound client request so that
/// downstream schedulers never see a spoofed tier header.
fn collect_inject_header_names(tiers: &[CompiledTier]) -> Vec<HeaderName> {
    let mut names = Vec::new();
    for tier in tiers {
        if let CompiledAction::Inject { headers } = &tier.action {
            for (name, _) in headers {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }
    }
    names
}

/// Check tier ordering invariants: strictly ascending, no tier after deny.
fn validate_tier_ordering(
    rule_name: &str,
    index: usize,
    capacity: u64,
    prev_capacity: u64,
    saw_deny: bool,
) -> Result<(), FilterError> {
    if saw_deny {
        return Err(format!(
            "token_rate_limit: rule '{rule_name}': deny tier must be the last tier \
             (found tier at index {index} after deny)"
        )
        .into());
    }
    if capacity == 0 {
        return Err(
            format!("token_rate_limit: rule '{rule_name}': tier at index {index} must have capacity > 0").into(),
        );
    }
    if index > 0 && capacity <= prev_capacity {
        return Err(format!(
            "token_rate_limit: rule '{rule_name}': tier capacities must be strictly ascending \
             (tier at index {index} has capacity {capacity} <= previous {prev_capacity})"
        )
        .into());
    }
    Ok(())
}

/// Compile a single tier's action, validating headers and deny placement.
#[expect(
    clippy::too_many_lines,
    reason = "validation for both inject (header parsing) and deny (capacity match) is a natural unit"
)]
fn compile_tier_action(
    rule_name: &str,
    index: usize,
    tier: &TierConfig,
    algorithm_capacity: u64,
    saw_deny: &mut bool,
) -> Result<CompiledAction, FilterError> {
    match tier.action.action_type {
        ActionType::Inject => {
            if tier.capacity > algorithm_capacity {
                return Err(format!(
                    "token_rate_limit: rule '{rule_name}': inject tier at index {index} \
                     has capacity ({}) above the algorithm's capacity ({algorithm_capacity}); \
                     it would never fire",
                    tier.capacity
                )
                .into());
            }
            if tier.action.headers.is_empty() {
                return Err(format!(
                    "token_rate_limit: rule '{rule_name}': inject tier at index {index} must have at least one header"
                )
                .into());
            }
            let headers = tier
                .action
                .headers
                .iter()
                .map(|(name, value)| {
                    let header_name = HeaderName::try_from(name.as_str()).map_err(|error| {
                        FilterError::from(format!(
                            "token_rate_limit: rule '{rule_name}': invalid inject header '{name}': {error}"
                        ))
                    })?;
                    let lower = header_name.as_str();
                    if lower == http::header::HOST.as_str()
                        || lower == http::header::CONTENT_LENGTH.as_str()
                        || lower == http::header::TRANSFER_ENCODING.as_str()
                        || praxis_core::reserved_headers::HOP_BY_HOP_HEADERS.contains(&lower)
                        || praxis_core::reserved_headers::is_reserved(lower)
                    {
                        return Err(FilterError::from(format!(
                            "token_rate_limit: rule '{rule_name}': inject tier at index {index} \
                             uses reserved/hop-by-hop header '{lower}'"
                        )));
                    }
                    let header_value = http::HeaderValue::from_str(value).map_err(|error| {
                        FilterError::from(format!(
                            "token_rate_limit: rule '{rule_name}': invalid inject header value for '{name}': {error}"
                        ))
                    })?;
                    Ok((header_name, header_value))
                })
                .collect::<Result<Vec<_>, FilterError>>()?;

            // Reject duplicate header names after case normalization.
            // `BTreeMap<String, String>` keys are the raw YAML spelling,
            // so `X-Token-Tier` and `x-token-tier` both pass syntax
            // validation but map to the same `HeaderName`.
            {
                let mut seen = HashSet::with_capacity(headers.len());
                for (name, _) in &headers {
                    if !seen.insert(name.clone()) {
                        return Err(format!(
                            "token_rate_limit: rule '{rule_name}': inject tier at index {index} \
                             has duplicate header '{name}' (after case normalization)"
                        )
                        .into());
                    }
                }
            }

            Ok(CompiledAction::Inject { headers })
        },
        ActionType::Deny => {
            if tier.capacity != algorithm_capacity {
                return Err(format!(
                    "token_rate_limit: rule '{rule_name}': deny tier capacity ({}) \
                     must equal the algorithm's capacity ({algorithm_capacity})",
                    tier.capacity
                )
                .into());
            }
            *saw_deny = true;
            Ok(CompiledAction::Deny)
        },
    }
}

/// Compile one YAML `rules:` entry into a [`CompiledRule`], validating
/// and constructing its backend.
///
/// # Errors
///
/// Returns [`FilterError`] if `capacity` is zero, `reserved_tokens` is
/// zero or exceeds `capacity`, `window`/`reservation_timeout` aren't
/// valid durations, a `match` header name is invalid, a weight is
/// negative or non-finite, or the rule's backend fails to construct
/// (see [`build_rule_backend`]).
fn compile_rule(
    rule: RuleConfig,
    backend: &BackendResource,
    filter_defaults: TokenWeights,
) -> Result<CompiledRule, FilterError> {
    let capacity = rule.algorithm.capacity();
    let reservation_timeout_ms = validate_rule_bounds(&rule, capacity)?;
    let estimation = compile_estimation(&rule.name, rule.reserved_tokens, rule.estimation, capacity)?;
    let backend = build_rule_backend(&rule.algorithm, backend, &rule.name, reservation_timeout_ms)?;
    let matcher = compile_matcher(&rule.name, rule.r#match)?;
    let loc = format!("rule '{}'", rule.name);
    let weights = filter_defaults.overlay(&rule.weights, &loc)?;
    let tiers = compile_tiers(&rule.name, rule.tiers, capacity)?;
    let inject_header_names = collect_inject_header_names(&tiers);

    Ok(CompiledRule {
        name: rule.name,
        matcher,
        backend,
        estimation,
        weights,
        tiers,
        inject_header_names,
    })
}

// -----------------------------------------------------------------------------
// TokenRateLimitFilter
// -----------------------------------------------------------------------------

/// Token-denominated rate limiter: reserves an estimated cost at
/// admission, reconciles against actual usage after the response
/// completes. Evaluates an ordered list of rules, each with its own
/// optional match condition, algorithm choice, and budget.
///
/// # YAML configuration
///
/// ```yaml
/// filter: token_rate_limit
/// backend:                           # optional: defaults to in-process state, shared by every rule
///   kind: valkey                      # memory (default) | valkey
///   url: "${TOKEN_RATE_LIMIT_VALKEY_URL}"
///   namespace: praxis:token_rate_limit
/// default_weights:                   # optional: omitted types default to 1.0
///   input: 1.0
///   output: 1.0
///   cached_input: 0.1                # prompt-cache hits (token.cache_read)
///   cache_write: 1.25                # prompt-cache writes (token.cache_write)
///   reasoning: 0.9                   # thinking tokens (token.reasoning)
/// rules:
///   - name: team-alpha                 # human-readable, unique per filter instance
///     match:                           # optional: omit for a catch-all rule
///       headers:
///         x-app-id: alpha
///     algorithm: sliding_window        # sliding_window | token_bucket
///     window: 1h                       # sliding_window only: window duration
///     capacity: 100000                 # max tokens admitted (sliding_window) or held (token_bucket)
///     estimation:                      # M3: configurable estimation strategy
///       strategy: max_tokens           # fixed | max_tokens | input_plus_max_tokens | model_scaled
///       multiplier: 1.2                # optional safety margin (default: 1.0)
///       fallback_estimate: 500         # used when max_tokens absent from request
///     weights:                         # optional per-rule overlay on default_weights
///       cached_input: 0.05
///   - name: team-beta
///     match:
///       headers:
///         x-app-id: beta
///     algorithm: token_bucket
///     capacity: 50000
///     refill_rate: 50                  # token_bucket only: tokens refilled per second
///     reserved_tokens: 200             # legacy: equivalent to estimation: { strategy: fixed, fallback_estimate: 200 }
/// ```
///
/// Rules are evaluated in order; the first whose `match` is satisfied
/// (or which has no `match` at all) applies. A request satisfying no
/// rule's `match` is **not** rate limited by this filter instance --
/// add a trailing rule with no `match` for a catch-all budget instead.
pub struct TokenRateLimitFilter {
    /// Compiled rules, evaluated in configured order.
    rules: Vec<CompiledRule>,

    /// Whether any rule uses a body-dependent estimation strategy.
    /// When `true`, `on_request` defers reservation to `on_request_body`
    /// and the pipeline buffers the request body for inspection.
    needs_body: bool,

    /// Trusted request identity used to partition every rule's budget.
    key_source: KeySource,

    /// Monotonic clock reference; all timestamps are offsets from this.
    epoch: Instant,
}

impl TokenRateLimitFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid, `rules` is
    /// empty, two rules share a `name`, or any individual rule fails to
    /// compile (see `compile_rule`).
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenRateLimitConfig = parse_filter_config("token_rate_limit", config)?;
        if cfg.rules.is_empty() {
            return Err("token_rate_limit: at least one rule is required".into());
        }
        let mut seen_names = HashSet::with_capacity(cfg.rules.len());
        for rule in &cfg.rules {
            if !seen_names.insert(rule.name.clone()) {
                return Err(format!("token_rate_limit: duplicate rule name '{}'", rule.name).into());
            }
        }

        let backend = build_backend_resource(&cfg.backend)?;
        let filter_defaults = TokenWeights::UNITY.overlay(&cfg.default_weights, "default_weights")?;
        let rules = cfg
            .rules
            .into_iter()
            .map(|rule| compile_rule(rule, &backend, filter_defaults))
            .collect::<Result<Vec<_>, _>>()?;

        let needs_body = rules.iter().any(|r| r.estimation.needs_body());

        Ok(Box::new(Self {
            rules,
            needs_body,
            key_source: cfg.key,
            epoch: Instant::now(),
        }))
    }

    /// Milliseconds elapsed since this filter's epoch.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "millis fit u64 for any realistic process uptime"
    )]
    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
    }

    /// The first rule (in configured order) whose `match` is satisfied
    /// by `headers`, alongside its index for reconciliation.
    fn matching_rule(&self, headers: &http::HeaderMap) -> Option<(usize, &CompiledRule)> {
        self.rules.iter().enumerate().find(|(_, rule)| rule.matches(headers))
    }

    /// Resolve an opaque backend bucket key from trusted request state.
    fn resolve_key(&self, ctx: &HttpFilterContext<'_>) -> Option<String> {
        match self.key_source {
            KeySource::Global => Some(FALLBACK_KEY.to_owned()),
            KeySource::AuthenticatedSubject => ctx
                .extensions
                .get::<AuthenticatedIdentity>()
                .map(AuthenticatedIdentity::subject_id)
                .map(subject_bucket_key),
        }
    }

    /// Resolve the trusted quota key and record a fail-closed identity miss.
    fn resolve_key_or_record_rejection(
        &self,
        ctx: &HttpFilterContext<'_>,
        rule: &CompiledRule,
        estimate: u64,
    ) -> Option<String> {
        let key = self.resolve_key(ctx);
        if key.is_none() {
            tracing::info!(
                rule = rule.name,
                "token_rate_limit: rejecting request (401), no authenticated subject"
            );
            record_accounting_admission(rule, "denied", estimate, "missing_authenticated_subject");
        }
        key
    }

    /// Reclaim idle/orphaned in-process state for one rule and publish
    /// its gauges.
    ///
    /// No-ops for a Valkey backend (`cleanup()` returns `None`): expiry
    /// there is handled by the Lua reserve script itself.
    fn cleanup_and_record_state(rule: &CompiledRule, now_ms: u64) {
        let Some(report) = rule.backend.cleanup(now_ms, 1) else {
            return;
        };
        record_cleanup_metrics(&rule.name, report);
    }

    /// Record metrics/metadata for an admitted reservation.
    fn record_admission(
        ctx: &mut HttpFilterContext<'_>,
        rule_index: usize,
        rule: &CompiledRule,
        admitted: AdmittedReservation,
    ) {
        record_request_metric(&rule.name, "admitted");
        record_reserved_metric(&rule.name, admitted.estimate);
        record_state_metrics(&rule.name, rule.backend.as_ref());
        record_accounting_admission(rule, "admitted", admitted.estimate, "reserved");
        ctx.set_metadata(META_RESERVATION_ID, admitted.reservation_id.to_string());
        ctx.set_metadata(META_BUCKET_KEY, admitted.key);
        ctx.set_metadata(META_RULE_INDEX, rule_index.to_string());
    }

    /// Build the 429 rejection for a denied reservation, including the
    /// token-denominated rate limit headers.
    fn denied_action(rule: &CompiledRule, estimate: u64, retry_after_ms: u64) -> FilterAction {
        record_request_metric(&rule.name, "denied");
        record_state_metrics(&rule.name, rule.backend.as_ref());
        record_accounting_admission(rule, "denied", estimate, "budget_exhausted");
        let retry_secs = retry_after_ms.saturating_add(999) / 1000;
        let retry_secs = retry_secs.max(1);
        FilterAction::Reject(
            Rejection::status(429)
                .with_header("Retry-After", retry_secs.to_string())
                .with_header(HEADER_RATELIMIT_LIMIT_TOKENS, rule.backend.limit().to_string())
                .with_header(HEADER_RATELIMIT_REMAINING_TOKENS, "0")
                .with_header(HEADER_RATELIMIT_RESET, retry_secs.to_string()),
        )
    }

    /// Turn a completed `reserve()` call into the `on_request` result:
    /// record admission metadata/metrics, evaluate graduated tiers and
    /// inject headers (S1), build the 429 rejection, or fail closed
    /// (503) on a backend error.
    #[expect(
        clippy::too_many_lines,
        reason = "the three outcome arms (admit+tiers / deny / error) are a natural unit"
    )]
    fn handle_reserve_outcome(
        ctx: &mut HttpFilterContext<'_>,
        rule_index: usize,
        rule: &CompiledRule,
        pending: PendingReservation,
        outcome: Result<BackendReserve, BackendError>,
    ) -> FilterAction {
        match outcome {
            Ok(BackendReserve::Admitted {
                reservation_id,
                estimate,
                usage_after,
            }) => {
                let admitted = AdmittedReservation {
                    key: pending.key,
                    reservation_id,
                    estimate,
                };
                Self::record_admission(ctx, rule_index, rule, admitted);
                ctx.set_metadata(META_ESTIMATE, pending.request_estimate.to_string());
                Self::evaluate_tiers(ctx, rule, usage_after);
                record_admission_span(ctx, rule, pending.request_estimate, "admitted");
                FilterAction::Continue
            },
            Ok(BackendReserve::Denied { retry_after_ms }) => {
                tracing::info!(
                    estimate = pending.request_estimate,
                    key = pending.key,
                    rule = rule.name,
                    "token_rate_limit: rejecting request (429)"
                );
                record_admission_span(ctx, rule, pending.request_estimate, "denied");
                Self::denied_action(rule, pending.request_estimate, retry_after_ms)
            },
            Err(error) => {
                record_backend_error_metric(&rule.name, rule.backend.backend_name());
                record_accounting_failure(rule, "reserve", &error);
                record_admission_span(ctx, rule, pending.request_estimate, "denied");
                tracing::error!(%error, rule = rule.name, "token_rate_limit: admission backend failed, failing closed");
                FilterAction::Reject(Rejection::status(503))
            },
        }
    }

    /// Evaluate graduated soft-limit tiers (proposal S1) and inject
    /// headers for every breached `inject` tier.
    ///
    /// Tiers are sorted by ascending `capacity` at compile time. We walk
    /// from lowest to highest, injecting headers for every tier whose
    /// threshold is at or below the current `usage_after`. The last tier
    /// pushed for a given header name wins (Rust's `request_headers_to_set`
    /// semantics), so a higher tier's header value naturally overrides a
    /// lower one for the same name — exactly the behavior the proposal
    /// specifies for overlapping header names across tiers.
    ///
    /// ## Client header stripping
    ///
    /// All configured inject header names are removed from the inbound
    /// request *before* any tier values are set, preventing a client
    /// below threshold from spoofing tier signals.
    ///
    /// ## Body-phase ordered mutations
    ///
    /// When the filter runs from `on_request_body` (body-dependent
    /// estimation strategies) and an earlier filter already activated
    /// the ordered `pre_read_mutations` log, injected headers are also
    /// pushed there so Core's pre-read replay applies them. This
    /// mirrors `state_owner_headers::queue_projection` and
    /// `agentic_loop::queue_continuation_header`.
    fn evaluate_tiers(ctx: &mut HttpFilterContext<'_>, rule: &CompiledRule, usage_after: u64) {
        if rule.tiers.is_empty() {
            return;
        }

        let ordered = !ctx.pre_read_mutations.is_empty();

        // Strip all configured inject header names from the inbound
        // request so a client cannot spoof tier signals.
        for name in &rule.inject_header_names {
            ctx.request_headers_to_remove.push(name.clone());
            if ordered {
                ctx.pre_read_mutations.push(TrustedHeaderMutation::Remove(name.clone()));
            }
        }

        for tier in &rule.tiers {
            if usage_after < tier.capacity {
                break;
            }
            if let CompiledAction::Inject { headers } = &tier.action {
                counter!(
                    "praxis_ai_token_rate_limit_soft_tier_activations_total",
                    "rule" => rule.name.clone(),
                    "capacity" => tier.capacity.to_string(),
                )
                .increment(1);
                Self::queue_tier_headers(ctx, headers, ordered);
            }
        }
    }

    /// Push inject-tier headers into the grouped queue and, when the
    /// ordered pre-read mutation log is active, into that log as well.
    fn queue_tier_headers(ctx: &mut HttpFilterContext<'_>, headers: &[(HeaderName, http::HeaderValue)], ordered: bool) {
        for (name, value) in headers {
            ctx.request_headers_to_set.push((name.clone(), value.clone()));
            if ordered {
                ctx.pre_read_mutations
                    .push(TrustedHeaderMutation::Set(name.clone(), value.clone()));
            }
        }
    }

    /// Look up the reservation/key/rule metadata `on_request` stashed for
    /// this exchange, if all three are present and the rule index still
    /// resolves -- the shared precondition for [`Self::reconcile`].
    fn reconciliation_context(&self, ctx: &HttpFilterContext<'_>) -> Option<(ReconcileRequest, &CompiledRule)> {
        let reservation_id = parse_u64_meta(ctx, META_RESERVATION_ID)?;
        let key = ctx.get_metadata(META_BUCKET_KEY).map(str::to_owned)?;
        let rule = ctx
            .get_metadata(META_RULE_INDEX)
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|index| self.rules.get(index))?;
        let actual = weighted_cost(UsageCounts::from_context(ctx), rule.weights);
        if actual.is_none() {
            tracing::trace!("token_rate_limit: no usable token usage metadata at end of stream, charging at estimate");
        }
        let Some(estimate) = parse_u64_meta(ctx, META_ESTIMATE) else {
            tracing::warn!("token_rate_limit: META_ESTIMATE missing at reconciliation, skipping");
            return None;
        };
        let request = ReconcileRequest {
            key,
            reservation_id,
            actual,
            estimate,
            now_ms: self.now_ms(),
        };
        Some((request, rule))
    }

    /// Reconcile a prior reservation against actual usage, if known.
    ///
    /// No-ops if the reservation, bucket key, or originating rule index
    /// metadata is absent -- reservation stands as final rather than
    /// guessing.
    ///
    /// In-process state reconciles synchronously and immediately (cheap,
    /// no I/O). A Valkey backend instead enqueues the reconciliation onto
    /// a background worker (see `backend::ValkeyTokenRateLimitBackend`/
    /// `backend::ValkeyTokenBucketBackend`) so the response is never held
    /// up on a network round-trip that has no bearing on whether *this*
    /// request was admitted.
    fn reconcile(&self, ctx: &HttpFilterContext<'_>) {
        let Some((request, rule)) = self.reconciliation_context(ctx) else {
            return;
        };
        if let Some(actual) = request.actual {
            record_actual_cost(ctx, actual);
        }

        if let Some(settlement) = rule.backend.reconcile_sync(&request) {
            record_settlement_metrics(&rule.name, &settlement);
            record_state_metrics(&rule.name, rule.backend.as_ref());
            record_accounting_settlement(&rule.name, rule.backend.as_ref(), &settlement);
            tracing::debug!(
                ?settlement,
                rule = rule.name,
                "token_rate_limit: reconciled reservation against actual usage"
            );
            return;
        }
        if let Err(error) = rule.backend.enqueue_reconcile(request) {
            record_backend_error_metric(&rule.name, rule.backend.backend_name());
            record_accounting_failure(rule, "enqueue_reconcile", &error);
            tracing::error!(%error, rule = rule.name, "token_rate_limit: failed to enqueue reconciliation");
        }
    }
}

/// Bundle for the budget key and computed estimate awaiting a backend
/// `reserve()` call, keeping [`TokenRateLimitFilter::handle_reserve_outcome`]
/// within clippy's argument-count budget.
struct PendingReservation {
    /// The budget key this reservation will use.
    key: String,
    /// The computed token estimate for this request.
    request_estimate: u64,
}

/// The bucket key, reservation ID, and estimate an admitted
/// [`BackendReserve::Admitted`] carries, bundled so
/// [`TokenRateLimitFilter::record_admission`] stays within clippy's
/// argument-count budget.
struct AdmittedReservation {
    /// The budget key this reservation was admitted under (see
    /// [`FALLBACK_KEY`] in global mode, or an opaque subject hash.
    key: String,
    /// The backend-issued reservation ID, stashed for later reconciliation.
    reservation_id: u64,
    /// Tokens reserved at admission (the rule's computed estimate, echoed
    /// back by the backend).
    estimate: u64,
}

/// Count an admission outcome using the issue #883 bounded label contract.
fn record_request_metric(rule_name: &str, result: &'static str) {
    counter!(
        "praxis_trl_requests_total",
        "result" => result,
        "rule" => rule_name.to_owned(),
    )
    .increment(1);
}

/// Count tokens reserved at admission.
fn record_reserved_metric(rule_name: &str, estimate: u64) {
    counter!("praxis_trl_tokens_reserved_total", "rule" => rule_name.to_owned()).increment(estimate);
}

/// Count one bounded backend failure.
fn record_backend_error_metric(rule_name: &str, backend: &'static str) {
    counter!(
        "praxis_trl_backend_errors_total",
        "backend" => backend,
        "rule" => rule_name.to_owned(),
    )
    .increment(1);
}

/// Emit gauges/counters for one rule's cleanup pass.
fn record_cleanup_metrics(rule_name: &str, report: CleanupReport) {
    if report.orphaned > 0 {
        counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "orphaned", "rule" => rule_name.to_owned())
            .increment(report.orphaned as u64);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "metrics gauges use f64, bounded by config caps in practice"
    )]
    {
        gauge!("praxis_trl_reservations_active", "rule" => rule_name.to_owned()).set(report.active_reservations as f64);
        gauge!("praxis_ai_token_rate_limit_active_keys", "rule" => rule_name.to_owned()).set(report.active_keys as f64);
    }
}

/// Emit counters for a completed synchronous (in-process) reconciliation.
fn record_settlement_metrics(rule_name: &str, settlement: &BackendSettlement) {
    if let BackendSettlement::Applied {
        actual,
        refund,
        overage,
    } = *settlement
    {
        counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "reconciled", "rule" => rule_name.to_owned())
            .increment(1);
        counter!("praxis_trl_tokens_reconciled_total", "rule" => rule_name.to_owned()).increment(actual);
        counter!("praxis_trl_tokens_refunded_total", "rule" => rule_name.to_owned()).increment(refund);
        counter!("praxis_trl_tokens_overage_total", "rule" => rule_name.to_owned()).increment(overage);
    }
}

/// Publish the latest rule-level gauges without backend I/O.
fn record_state_metrics(rule_name: &str, backend: &dyn TokenRateLimitStateBackend) {
    let snapshot = backend.snapshot();
    #[expect(
        clippy::cast_precision_loss,
        reason = "metrics gauges use f64; configured capacities and reservation counts are bounded"
    )]
    {
        gauge!(
            "praxis_trl_budget_remaining",
            "algorithm" => backend.algorithm_name(),
            "rule" => rule_name.to_owned(),
        )
        .set(snapshot.budget_remaining as f64);
        gauge!("praxis_trl_reservations_active", "rule" => rule_name.to_owned())
            .set(snapshot.active_reservations as f64);
        gauge!("praxis_ai_token_rate_limit_active_keys", "rule" => rule_name.to_owned())
            .set(snapshot.active_keys as f64);
    }
}

/// Emit one bounded admission accounting record on its dedicated target.
fn record_accounting_admission(rule: &CompiledRule, result: &'static str, estimate: u64, outcome: &'static str) {
    tracing::info!(
        target: "praxis_ai::token_rate_limit::accounting",
        phase = "admission",
        rule = rule.name,
        algorithm = rule.backend.algorithm_name(),
        backend = rule.backend.backend_name(),
        result,
        estimate,
        outcome,
        "token rate limit accounting"
    );
}

/// Emit one bounded synchronous-settlement accounting record.
fn record_accounting_settlement(
    rule_name: &str,
    backend: &dyn TokenRateLimitStateBackend,
    settlement: &BackendSettlement,
) {
    if let BackendSettlement::Applied {
        actual,
        refund,
        overage,
    } = *settlement
    {
        tracing::info!(
            target: "praxis_ai::token_rate_limit::accounting",
            phase = "reconciliation",
            rule = rule_name,
            algorithm = backend.algorithm_name(),
            backend = backend.backend_name(),
            result = "applied",
            actual,
            refund,
            overage,
            "token rate limit accounting"
        );
    }
}

/// Emit a bounded accounting failure without request or user identifiers.
fn record_accounting_failure(rule: &CompiledRule, operation: &'static str, error: &BackendError) {
    tracing::warn!(
        target: "praxis_ai::token_rate_limit::accounting",
        phase = operation,
        rule = rule.name,
        algorithm = rule.backend.algorithm_name(),
        backend = rule.backend.backend_name(),
        result = "failed",
        error = %error,
        "token rate limit accounting"
    );
}

/// Attach the bounded admission decision to this request's trace.
#[cfg(feature = "opentelemetry")]
fn record_admission_span(ctx: &mut HttpFilterContext<'_>, rule: &CompiledRule, estimate: u64, decision: &'static str) {
    ctx.extensions.insert(crate::opentelemetry::token_rate_limit_span(
        &rule.name,
        rule.backend.algorithm_name(),
        estimate,
        decision,
    ));
}

/// No-op when semantic tracing is not compiled in.
#[cfg(not(feature = "opentelemetry"))]
fn record_admission_span(
    _ctx: &mut HttpFilterContext<'_>,
    _rule: &CompiledRule,
    _estimate: u64,
    _decision: &'static str,
) {
}

/// Complete the request span with actual cost when provider usage exists.
#[cfg(feature = "opentelemetry")]
fn record_actual_cost(ctx: &HttpFilterContext<'_>, actual: u64) {
    if let Some(span) = ctx.extensions.get::<crate::opentelemetry::TokenRateLimitSpan>() {
        span.record_actual(actual);
    }
}

/// No-op when semantic tracing is not compiled in.
#[cfg(not(feature = "opentelemetry"))]
fn record_actual_cost(_ctx: &HttpFilterContext<'_>, _actual: u64) {}

#[async_trait]
impl HttpFilter for TokenRateLimitFilter {
    fn name(&self) -> &'static str {
        "token_rate_limit"
    }

    fn request_body_access(&self) -> BodyAccess {
        if self.needs_body {
            BodyAccess::ReadOnly
        } else {
            BodyAccess::None
        }
    }

    fn request_body_mode(&self) -> BodyMode {
        if self.needs_body {
            BodyMode::StreamBuffer {
                max_bytes: Some(MAX_ESTIMATION_BODY_BYTES),
            }
        } else {
            BodyMode::Stream
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if self.needs_body {
            return Ok(FilterAction::Continue);
        }
        let now_ms = self.now_ms();
        let Some((rule_index, rule)) = self.matching_rule(&ctx.request.headers) else {
            return Ok(FilterAction::Continue);
        };
        Self::cleanup_and_record_state(rule, now_ms);
        let estimate = rule.estimation.estimate(&ctx.request.headers, None);
        let Some(estimate) = estimate else {
            return Ok(FilterAction::Continue);
        };
        let Some(key) = self.resolve_key_or_record_rejection(ctx, rule, estimate) else {
            record_admission_span(ctx, rule, estimate, "denied");
            return Ok(FilterAction::Reject(Rejection::status(401)));
        };
        let outcome = rule
            .backend
            .reserve(ReserveRequest {
                key: key.clone(),
                estimate,
                now_ms,
            })
            .await;
        let pending = PendingReservation {
            key,
            request_estimate: estimate,
        };
        Ok(Self::handle_reserve_outcome(ctx, rule_index, rule, pending, outcome))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream || !self.needs_body {
            return Ok(FilterAction::Continue);
        }
        let now_ms = self.now_ms();
        let Some((rule_index, rule)) = self.matching_rule(&ctx.request.headers) else {
            return Ok(FilterAction::Continue);
        };
        Self::cleanup_and_record_state(rule, now_ms);

        let body_probe = parse_body_probe(body);
        let estimate = rule.estimation.estimate(&ctx.request.headers, body_probe.as_ref());

        let Some(estimate) = estimate else {
            return Ok(FilterAction::Continue);
        };

        let Some(key) = self.resolve_key_or_record_rejection(ctx, rule, estimate) else {
            record_admission_span(ctx, rule, estimate, "denied");
            return Ok(FilterAction::Reject(Rejection::status(401)));
        };
        let outcome = rule
            .backend
            .reserve(ReserveRequest {
                key: key.clone(),
                estimate,
                now_ms,
            })
            .await;
        let pending = PendingReservation {
            key,
            request_estimate: estimate,
        };
        Ok(Self::handle_reserve_outcome(ctx, rule_index, rule, pending, outcome))
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            self.reconcile(ctx);
            ctx.filter_metadata.remove(META_RESERVATION_ID);
            ctx.filter_metadata.remove(META_BUCKET_KEY);
            ctx.filter_metadata.remove(META_RULE_INDEX);
            ctx.filter_metadata.remove(META_ESTIMATE);
            #[cfg(feature = "opentelemetry")]
            ctx.extensions.remove::<crate::opentelemetry::TokenRateLimitSpan>();
        }
        Ok(FilterAction::Continue)
    }
}

/// Convert a verified subject into a fixed-size, non-identifying backend key.
fn subject_bucket_key(subject: &str) -> String {
    let digest = Sha256::digest(subject.as_bytes());
    format!("subject:v1:{}", URL_SAFE_NO_PAD.encode(digest))
}

/// Parse the bounded request body fields used by estimation strategies.
fn parse_body_probe(body: &Option<Bytes>) -> Option<BodyProbe> {
    body.as_ref().and_then(|raw| serde_json::from_slice(raw).ok())
}

/// Expand one `${ENV_VAR}` reference in a backend URL, if present.
///
/// Takes an explicit lookup function (rather than calling
/// [`std::env::var`] directly) so tests can exercise both branches
/// without mutating real process-global environment state.
fn expand_backend_url_with(
    url: &str,
    lookup: impl Fn(&str) -> Result<String, std::env::VarError>,
) -> Result<String, FilterError> {
    let Some(start) = url.find("${") else {
        return Ok(url.to_owned());
    };
    let Some(name) = url.strip_prefix("${").and_then(|value| value.strip_suffix('}')) else {
        return Err("token_rate_limit: backend.url supports one complete ${ENV_VAR} reference".into());
    };
    if start != 0 || name.contains("${") {
        return Err("token_rate_limit: backend.url supports one complete ${ENV_VAR} reference".into());
    }
    if name.is_empty()
        || !name
            .bytes()
            .enumerate()
            .all(|(index, byte)| byte == b'_' || byte.is_ascii_uppercase() || (index > 0 && byte.is_ascii_digit()))
    {
        return Err("token_rate_limit: backend.url contains an invalid environment variable reference".into());
    }
    lookup(name).map_err(|_error| "token_rate_limit: backend.url environment variable is not set".into())
}

/// Expand one `${ENV_VAR}` reference in a backend URL against the real
/// process environment.
fn expand_backend_url(url: &str) -> Result<String, FilterError> {
    expand_backend_url_with(url, |name| std::env::var(name))
}

/// Parse a simple `<number><unit>` duration (`ms`, `s`, `m`, `h`) into
/// milliseconds, as used by `window` and `reservation_timeout`.
fn parse_duration_ms(value: &str) -> Result<u64, FilterError> {
    let value = value.trim();
    let (number, multiplier) = if let Some(value) = value.strip_suffix("ms") {
        (value, 1_u64)
    } else if let Some(value) = value.strip_suffix('s') {
        (value, 1_000_u64)
    } else if let Some(value) = value.strip_suffix('m') {
        (value, 60_000_u64)
    } else if let Some(value) = value.strip_suffix('h') {
        (value, 3_600_000_u64)
    } else {
        return Err(format!("token_rate_limit: invalid duration '{value}'").into());
    };
    let amount = number
        .parse::<u64>()
        .map_err(|error| format!("token_rate_limit: invalid duration '{value}': {error}"))?;
    let millis = amount
        .checked_mul(multiplier)
        .ok_or("token_rate_limit: duration overflow")?;
    if millis == 0 {
        return Err("token_rate_limit: duration must be positive".into());
    }
    Ok(millis)
}

impl std::fmt::Debug for TokenRateLimitFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenRateLimitFilter")
            .field("rules", &self.rules.iter().map(|rule| &rule.name).collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

/// White-box tests that construct a [`CompiledRule`] directly with a
/// purpose-built [`backend::TokenRateLimitStateBackend`], for behavior
/// unreachable through [`TokenRateLimitFilter::from_config`] with any
/// real backend. Kept separate from `tests.rs`, which deliberately only
/// drives the filter through its public `HttpFilter` surface.
#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod backend_injection_tests {
    use praxis_filter::HttpFilter as _;

    use super::{
        CompiledEstimation, CompiledRule, TokenRateLimitFilter,
        backend::{
            BackendError, BackendReserve, BackendSettlement, BackendSnapshot, ReconcileRequest, ReserveRequest,
            TokenRateLimitStateBackend,
        },
        record_backend_error_metric, record_request_metric, record_reserved_metric, record_settlement_metrics,
        record_state_metrics,
    };

    /// A backend that admits every reservation but always fails to
    /// enqueue its reconciliation -- the one way
    /// [`TokenRateLimitFilter::reconcile`]'s enqueue-failure log line is
    /// reachable in production (a real Valkey-backed rule's
    /// background-worker channel saturated or its receiver gone), and
    /// impractical to drive there for real without either exhausting a
    /// live worker's 1024-deep channel or tearing down its receiver
    /// mid-test.
    struct EnqueueAlwaysFailsBackend;

    #[async_trait::async_trait]
    impl TokenRateLimitStateBackend for EnqueueAlwaysFailsBackend {
        async fn reserve(&self, _request: ReserveRequest) -> Result<BackendReserve, BackendError> {
            Ok(BackendReserve::Admitted {
                reservation_id: 1,
                estimate: 1,
                usage_after: 1,
            })
        }

        async fn reconcile(&self, _request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
            panic!("not exercised by this test")
        }

        fn enqueue_reconcile(&self, _request: ReconcileRequest) -> Result<(), BackendError> {
            Err(BackendError::Unavailable("enqueue failed (test)".into()))
        }

        fn limit(&self) -> u64 {
            1
        }

        fn snapshot(&self) -> BackendSnapshot {
            BackendSnapshot {
                budget_remaining: 73,
                active_reservations: 2,
                active_keys: 3,
            }
        }

        fn backend_name(&self) -> &'static str {
            "test"
        }

        fn algorithm_name(&self) -> &'static str {
            "test"
        }
    }

    /// [`TokenRateLimitFilter::reconcile`] falls back to
    /// `enqueue_reconcile` when `reconcile_sync` returns `None` (the
    /// default, unimplemented by [`EnqueueAlwaysFailsBackend`]). When
    /// that enqueue itself fails, the error must be logged and
    /// swallowed, not propagated -- a reconciliation failure is never
    /// the inbound request's fault, so it must not affect the response
    /// already on its way out.
    #[tokio::test]
    async fn reconcile_logs_rather_than_propagates_an_enqueue_reconcile_failure() {
        let filter = TokenRateLimitFilter {
            rules: vec![CompiledRule {
                name: "default".to_owned(),
                matcher: None,
                backend: std::sync::Arc::new(EnqueueAlwaysFailsBackend),
                estimation: CompiledEstimation::Fixed { estimate: 1 },
                weights: super::TokenWeights::UNITY,
                tiers: Vec::new(),
                inject_header_names: Vec::new(),
            }],
            needs_body: false,
            key_source: super::KeySource::Global,
            epoch: std::time::Instant::now(),
        };

        let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        drop(filter.on_request(&mut ctx).await.unwrap());

        #[cfg(feature = "opentelemetry")]
        assert!(
            ctx.extensions
                .get::<crate::opentelemetry::TokenRateLimitSpan>()
                .is_some()
        );

        let mut body = None;
        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        #[cfg(feature = "opentelemetry")]
        assert!(
            ctx.extensions
                .get::<crate::opentelemetry::TokenRateLimitSpan>()
                .is_none()
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one contract test verifies all eight issue-defined metric families together"
    )]
    fn prometheus_contract_emits_all_issue_883_metric_families_with_bounded_labels() {
        let rule = CompiledRule {
            name: "engineering".to_owned(),
            matcher: None,
            backend: std::sync::Arc::new(EnqueueAlwaysFailsBackend),
            estimation: CompiledEstimation::Fixed { estimate: 1 },
            weights: super::TokenWeights::UNITY,
            tiers: Vec::new(),
            inject_header_names: Vec::new(),
        };
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            record_request_metric(&rule.name, "admitted");
            record_request_metric(&rule.name, "denied");
            record_reserved_metric(&rule.name, 50);
            record_settlement_metrics(
                &rule.name,
                &BackendSettlement::Applied {
                    actual: 40,
                    refund: 10,
                    overage: 0,
                },
            );
            record_state_metrics(&rule.name, rule.backend.as_ref());
            record_backend_error_metric(&rule.name, rule.backend.backend_name());
        });

        let snapshot = snapshotter.snapshot().into_vec();
        let labels_for = |name: &str| {
            snapshot
                .iter()
                .filter(|(key, ..)| key.key().name() == name)
                .map(|(key, ..)| {
                    let mut labels = key
                        .key()
                        .labels()
                        .map(|label| label.key().to_owned())
                        .collect::<Vec<_>>();
                    labels.sort();
                    labels
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(
            labels_for("praxis_trl_requests_total"),
            vec![vec!["result".to_owned(), "rule".to_owned()]; 2]
        );
        for name in [
            "praxis_ai_token_rate_limit_reservations_total",
            "praxis_trl_tokens_reserved_total",
            "praxis_trl_tokens_reconciled_total",
            "praxis_trl_tokens_refunded_total",
            "praxis_trl_tokens_overage_total",
            "praxis_trl_reservations_active",
        ] {
            assert_eq!(
                labels_for(name),
                if name == "praxis_ai_token_rate_limit_reservations_total" {
                    vec![vec!["result".to_owned(), "rule".to_owned()]]
                } else {
                    vec![vec!["rule".to_owned()]]
                },
                "wrong labels for {name}"
            );
        }
        assert_eq!(
            labels_for("praxis_trl_budget_remaining"),
            vec![vec!["algorithm".to_owned(), "rule".to_owned()]]
        );
        assert_eq!(
            labels_for("praxis_trl_backend_errors_total"),
            vec![vec!["backend".to_owned(), "rule".to_owned()]]
        );
    }
}
