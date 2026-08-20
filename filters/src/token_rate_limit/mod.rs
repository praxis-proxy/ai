// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token-denominated rate limiting filter.
//!
//! Implements the uncontested MVP core of `ai#658`'s token rate limiting
//! proposal (`docs/proposals/00121_token-rate-limiting.md`, tracked by
//! epic `ai#121`): an ordered list of `rules`, each an optional static
//! header-value match condition (`ai#658`'s "static matchers (MVP)")
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
//! Valkey-backend spec this MVP is a narrower slice of.
//!
//! Per-rule algorithm choice, rather than one fixed algorithm for the
//! whole filter, mirrors the `GuardrailsFilter`'s own `rules: Vec<RuleConfig>`
//! architecture and answers the maintainer's own framing on
//! `ai#789`/`praxis#551` ("this looks like a per-rule choice, similar to
//! `shadow`/enforcement-action knobs elsewhere").
//!
//! Deliberately deferred, pending open design threads on `ai#658`:
//!
//! - **CEL-expression matchers**: only static, exact header-value equality matching is implemented (overlapping
//!   `praxis#189`/`#232`).
//! - **Composite/multi-dimension keys, per-model keys (`ai#123`)**: `ai#129`'s single-header-value keying (one budget
//!   applied uniformly per key, fallback to global) is implemented per rule; richer composite keys are not.
//! - **Configurable estimation (M3)**: `estimate_tokens` is a fixed constant per rule, not derived from request
//!   metadata (e.g. `max_tokens`).
//! - **Token-type-aware accounting (M4)**: reconciles against `token.total` only; per-type (input/output/cached)
//!   weighting is not modeled yet.
//! - **Multiple budgets per rule, soft-limit tiers**: `ai#658`'s proposal allows several `token_budgets` (e.g. hourly +
//!   daily) and graduated tiers per rule; this MVP admits exactly one budget per rule with a hard deny at capacity.
//! - **Observability (M7/M8) and metering (S3)**: out of scope here -- both are recommended to split into their own
//!   proposals on the `ai#658` review thread.
//!
//! `X-RateLimit-*` headers are emitted on 429 rejection only, matching
//! the validated pattern on the source spike branch: computing
//! "remaining" for every successful response would need an extra read
//! on the Valkey path (doubling backend round-trips per request), so
//! both backends behave identically here rather than diverging by
//! backend.
//!
//! Depends on `token_count` running earlier in the response phase to
//! populate `token.total` in `filter_metadata`; if that metadata is
//! absent when the response completes, the reservation is left as
//! final rather than guessed at.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests;

mod backend;
mod config;
mod ledger;
mod token_bucket_ledger;

use std::{collections::HashSet, sync::Arc, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
use http::header::HeaderName;
use metrics::{counter, gauge};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};

use self::{
    backend::{
        BackendError, BackendReserve, BackendSettlement, CleanupReport, InMemoryTokenBucketBackend,
        InMemoryTokenRateLimitBackend, ReconcileRequest, ReserveRequest, TokenRateLimitStateBackend,
        ValkeyBackendConfig, ValkeyTokenBucketBackend, ValkeyTokenBucketConfig, ValkeyTokenRateLimitBackend,
    },
    config::{
        AlgorithmConfig, BackendConfig, BackendKind, DEFAULT_RESERVATION_TIMEOUT, MatchConfig, RuleConfig,
        TokenRateLimitConfig,
    },
    ledger::{Budget, Ledger, LedgerConfig},
    token_bucket_ledger::{TokenBucketConfig, TokenBucketLedger},
};
use crate::token_usage::META_TOKEN_TOTAL;

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

/// Sentinel key for requests with no `bucket_key_header` configured, or
/// where the configured header was absent, not valid UTF-8, empty, or
/// longer than [`MAX_KEY_LENGTH`] on this request -- all of these cases
/// share one budget, sized the same as every per-key budget.
const FALLBACK_KEY: &str = "__fallback__";

/// Bound on distinct `bucket_key_header` values retained at once, per rule.
///
/// App/tenant cardinality is expected to be far lower in practice; this
/// mirrors the soft cap `rate_limit` uses for per-IP entries.
const MAX_KEYS: usize = 100_000;

/// Bound on a single `bucket_key_header` value's length.
const MAX_KEY_LENGTH: usize = 256;

/// Bound on reservations awaiting reconciliation across all keys, per rule.
const MAX_ACTIVE_RESERVATIONS: usize = 200_000;

/// Rate limit header: configured token budget.
///
/// Uses the `-Tokens` suffix per `ai#124`'s spec, distinct from the
/// existing `rate_limit` filter's unsuffixed `X-RateLimit-Limit`, to
/// avoid a header collision when both filters run in the same
/// pipeline (flagged on the `ai#658` review thread).
const HEADER_RATELIMIT_LIMIT_TOKENS: &str = "X-RateLimit-Limit-Tokens";

/// Rate limit header: remaining tokens (always `0` -- only sent on 429).
const HEADER_RATELIMIT_REMAINING_TOKENS: &str = "X-RateLimit-Remaining-Tokens";

/// Rate limit header: seconds until another admission attempt may succeed.
const HEADER_RATELIMIT_RESET: &str = "X-RateLimit-Reset-Tokens";

/// Resolve a `backend: {kind: valkey}` block's `url`/`namespace` down to
/// concrete values, shared by both algorithms' Valkey backend builders.
///
/// # Errors
///
/// Returns [`FilterError`] if `url` is missing or fails to parse/expand.
fn resolve_valkey_backend(backend: &BackendConfig, rule_name: &str) -> Result<(String, String), FilterError> {
    let url = backend.url.as_deref().ok_or_else(|| {
        format!("token_rate_limit: rule '{rule_name}': backend.url is required for backend.kind: valkey")
    })?;
    let url = expand_backend_url(url)?;
    let namespace = backend
        .namespace
        .clone()
        .unwrap_or_else(|| "praxis:token_rate_limit".to_owned());
    Ok((url, namespace))
}

/// Build the configured state backend for a `sliding_window` rule
/// (in-process ledger or Valkey).
///
/// # Errors
///
/// Returns [`FilterError`] if the ledger config is invalid, a `valkey`
/// backend is configured without a `url`, or the URL fails to parse/expand.
fn build_sliding_window_backend(
    backend: &BackendConfig,
    rule_name: &str,
    budgets: Vec<Budget>,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match backend.kind {
        BackendKind::Memory => {
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
        BackendKind::Valkey => {
            let (url, namespace) = resolve_valkey_backend(backend, rule_name)?;
            Ok(Arc::new(ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
                url,
                namespace,
                rule: rule_name.to_owned(),
                budgets,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })?))
        },
    }
}

/// Build the configured state backend for a `token_bucket` rule
/// (in-process ledger or Valkey).
///
/// # Errors
///
/// Returns [`FilterError`] if the ledger config is invalid, a `valkey`
/// backend is configured without a `url`, or the URL fails to parse/expand.
fn build_token_bucket_backend(
    backend: &BackendConfig,
    rule_name: &str,
    capacity: u64,
    refill_rate: f64,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match backend.kind {
        BackendKind::Memory => {
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
        BackendKind::Valkey => {
            let (url, namespace) = resolve_valkey_backend(backend, rule_name)?;
            Ok(Arc::new(ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                url,
                namespace,
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
// CompiledRule
// -----------------------------------------------------------------------------

/// One `rules:` entry, fully resolved: its match condition (if any),
/// backend, and per-rule keying/estimation.
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

    /// Header whose value keys an independent budget within this rule,
    /// per `ai#129`.
    bucket_key_header: Option<HeaderName>,

    /// Fixed token cost reserved at admission (M3 placeholder).
    estimate_tokens: u64,
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

    /// Resolve which budget key a request should use within this rule,
    /// given its configured key header (if any).
    ///
    /// Returns [`FALLBACK_KEY`] when keying isn't configured at all, or
    /// when it is configured but this particular request's header is
    /// absent, not valid UTF-8, empty, or exceeds [`MAX_KEY_LENGTH`] --
    /// all of these route to one shared budget.
    fn resolve_key(&self, headers: &http::HeaderMap) -> String {
        self.bucket_key_header
            .as_ref()
            .and_then(|name| headers.get(name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty() && value.len() <= MAX_KEY_LENGTH)
            .map_or_else(|| FALLBACK_KEY.to_owned(), str::to_owned)
    }
}

/// Validate a rule's `capacity`/`estimate_tokens` bounds and resolve its
/// `reservation_timeout` string to milliseconds.
///
/// # Errors
///
/// Returns [`FilterError`] if `capacity` is zero, `estimate_tokens` is
/// zero or exceeds `capacity`, or `reservation_timeout` isn't a valid
/// duration.
fn validate_rule_bounds(rule: &RuleConfig, capacity: u64) -> Result<u64, FilterError> {
    if capacity == 0 {
        return Err(format!(
            "token_rate_limit: rule '{}': capacity must be greater than 0",
            rule.name
        )
        .into());
    }
    if rule.estimate_tokens == 0 {
        return Err(format!(
            "token_rate_limit: rule '{}': estimate_tokens must be greater than 0",
            rule.name
        )
        .into());
    }
    if rule.estimate_tokens > capacity {
        return Err(format!(
            "token_rate_limit: rule '{}': estimate_tokens must not exceed capacity",
            rule.name
        )
        .into());
    }
    parse_duration_ms(
        rule.reservation_timeout
            .as_deref()
            .unwrap_or(DEFAULT_RESERVATION_TIMEOUT),
    )
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
    algorithm: &AlgorithmConfig,
    backend: &BackendConfig,
    rule_name: &str,
    reservation_timeout_ms: u64,
) -> Result<Arc<dyn TokenRateLimitStateBackend>, FilterError> {
    match algorithm {
        AlgorithmConfig::SlidingWindow { window, capacity } => {
            let window_ms = parse_duration_ms(window)?;
            let budgets = vec![Budget {
                window_ms,
                capacity: *capacity,
            }];
            build_sliding_window_backend(backend, rule_name, budgets, reservation_timeout_ms)
        },
        AlgorithmConfig::TokenBucket { capacity, refill_rate } => {
            build_token_bucket_backend(backend, rule_name, *capacity, *refill_rate, reservation_timeout_ms)
        },
    }
}

/// Compile one YAML `rules:` entry into a [`CompiledRule`], validating
/// and constructing its backend.
///
/// # Errors
///
/// Returns [`FilterError`] if `capacity` is zero, `estimate_tokens` is
/// zero or exceeds `capacity`, `window`/`reservation_timeout` aren't
/// valid durations, a `match` header name is invalid, or the rule's
/// backend fails to construct (see [`build_rule_backend`]).
fn compile_rule(rule: RuleConfig) -> Result<CompiledRule, FilterError> {
    let capacity = rule.algorithm.capacity();
    let reservation_timeout_ms = validate_rule_bounds(&rule, capacity)?;
    let backend = build_rule_backend(&rule.algorithm, &rule.backend, &rule.name, reservation_timeout_ms)?;
    let matcher = compile_matcher(&rule.name, rule.r#match)?;

    let bucket_key_header = rule
        .bucket_key_header
        .map(|name| {
            HeaderName::try_from(name.as_str()).map_err(|error| {
                FilterError::from(format!(
                    "token_rate_limit: rule '{}': invalid bucket_key_header: {error}",
                    rule.name
                ))
            })
        })
        .transpose()?;

    Ok(CompiledRule {
        name: rule.name,
        matcher,
        backend,
        bucket_key_header,
        estimate_tokens: rule.estimate_tokens,
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
/// rules:
///   - name: team-alpha                 # human-readable, unique per filter instance
///     match:                           # optional: omit for a catch-all rule
///       headers:
///         x-app-id: alpha
///     algorithm: sliding_window        # sliding_window | token_bucket
///     window: 1h                       # sliding_window only: window duration
///     capacity: 100000                 # max tokens admitted (sliding_window) or held (token_bucket)
///     estimate_tokens: 500             # fixed cost reserved per request at admission
///     bucket_key_header: x-team-id     # optional (ai#129): one budget per header value within this rule
///     backend:                         # optional: defaults to in-process state
///       kind: valkey                    # memory (default) | valkey
///       url: "${TOKEN_RATE_LIMIT_VALKEY_URL}"
///       namespace: praxis:token_rate_limit
///   - name: team-beta
///     match:
///       headers:
///         x-app-id: beta
///     algorithm: token_bucket
///     capacity: 50000
///     refill_rate: 50                  # token_bucket only: tokens refilled per second
///     estimate_tokens: 200
/// ```
///
/// Rules are evaluated in order; the first whose `match` is satisfied
/// (or which has no `match` at all) applies. A request satisfying no
/// rule's `match` is **not** rate limited by this filter instance --
/// add a trailing rule with no `match` for a catch-all budget instead.
pub struct TokenRateLimitFilter {
    /// Compiled rules, evaluated in configured order.
    rules: Vec<CompiledRule>,

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

        let rules = cfg.rules.into_iter().map(compile_rule).collect::<Result<Vec<_>, _>>()?;

        Ok(Box::new(Self {
            rules,
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
        counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "admitted", "rule" => rule.name.clone())
            .increment(1);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "estimated", "rule" => rule.name.clone())
            .increment(admitted.estimate);
        ctx.set_metadata(META_RESERVATION_ID, admitted.reservation_id.to_string());
        ctx.set_metadata(META_BUCKET_KEY, admitted.key);
        ctx.set_metadata(META_RULE_INDEX, rule_index.to_string());
    }

    /// Build the 429 rejection for a denied reservation, including the
    /// token-denominated rate limit headers.
    fn denied_action(rule: &CompiledRule, retry_after_ms: u64) -> FilterAction {
        counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "rule" => rule.name.clone())
            .increment(1);
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
    /// record admission metadata/metrics, build the 429 rejection, or
    /// fail closed (503) on a backend error.
    fn handle_reserve_outcome(
        ctx: &mut HttpFilterContext<'_>,
        rule_index: usize,
        rule: &CompiledRule,
        key: String,
        outcome: Result<BackendReserve, BackendError>,
    ) -> FilterAction {
        match outcome {
            Ok(BackendReserve::Admitted {
                reservation_id,
                estimate,
            }) => {
                let admitted = AdmittedReservation {
                    key,
                    reservation_id,
                    estimate,
                };
                Self::record_admission(ctx, rule_index, rule, admitted);
                FilterAction::Continue
            },
            Ok(BackendReserve::Denied { retry_after_ms }) => {
                tracing::info!(
                    estimate = rule.estimate_tokens,
                    key,
                    rule = rule.name,
                    "token_rate_limit: rejecting request (429)"
                );
                Self::denied_action(rule, retry_after_ms)
            },
            Err(error) => {
                counter!("praxis_ai_token_rate_limit_backend_errors_total", "operation" => "reserve", "rule" => rule.name.clone())
                    .increment(1);
                tracing::error!(%error, rule = rule.name, "token_rate_limit: admission backend failed, failing closed");
                FilterAction::Reject(Rejection::status(503))
            },
        }
    }

    /// Look up the reservation/key/rule metadata `on_request` stashed for
    /// this exchange, if all three are present and the rule index still
    /// resolves -- the shared precondition for [`Self::reconcile`].
    fn reconciliation_context(&self, ctx: &HttpFilterContext<'_>) -> Option<(ReconcileRequest, &CompiledRule)> {
        let reservation_id = ctx
            .get_metadata(META_RESERVATION_ID)
            .and_then(|v| v.parse::<u64>().ok())?;
        let key = ctx.get_metadata(META_BUCKET_KEY).map(str::to_owned)?;
        let rule = ctx
            .get_metadata(META_RULE_INDEX)
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|index| self.rules.get(index))?;
        let actual = ctx.get_metadata(META_TOKEN_TOTAL).and_then(|v| v.parse::<u64>().ok());
        if actual.is_none() {
            tracing::trace!("token_rate_limit: no token.total metadata at end of stream, charging at estimate");
        }
        let request = ReconcileRequest {
            key,
            reservation_id,
            actual,
            estimate: rule.estimate_tokens,
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

        if let Some(settlement) = rule.backend.reconcile_sync(&request) {
            record_settlement_metrics(&rule.name, &settlement);
            tracing::debug!(
                ?settlement,
                rule = rule.name,
                "token_rate_limit: reconciled reservation against actual usage"
            );
            return;
        }
        if let Err(error) = rule.backend.enqueue_reconcile(request) {
            tracing::error!(%error, rule = rule.name, "token_rate_limit: failed to enqueue reconciliation");
        }
    }
}

/// The bucket key, reservation ID, and estimate an admitted
/// [`BackendReserve::Admitted`] carries, bundled so
/// [`TokenRateLimitFilter::record_admission`] stays within clippy's
/// argument-count budget.
struct AdmittedReservation {
    /// The resolved per-request budget key (see [`CompiledRule::resolve_key`]).
    key: String,
    /// The backend-issued reservation ID, stashed for later reconciliation.
    reservation_id: u64,
    /// Tokens reserved at admission (the rule's `estimate_tokens`, echoed
    /// back by the backend).
    estimate: u64,
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
        gauge!("praxis_ai_token_rate_limit_active_reservations", "rule" => rule_name.to_owned())
            .set(report.active_reservations as f64);
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
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "actual", "rule" => rule_name.to_owned())
            .increment(actual);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "refunded", "rule" => rule_name.to_owned())
            .increment(refund);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "overage", "rule" => rule_name.to_owned())
            .increment(overage);
    }
}

#[async_trait]
impl HttpFilter for TokenRateLimitFilter {
    fn name(&self) -> &'static str {
        "token_rate_limit"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let now_ms = self.now_ms();
        let Some((rule_index, rule)) = self.matching_rule(&ctx.request.headers) else {
            // No configured rule applies to this request -- not rate
            // limited by this filter instance. Operators wanting a
            // catch-all budget add a trailing rule with no `match`.
            return Ok(FilterAction::Continue);
        };
        Self::cleanup_and_record_state(rule, now_ms);
        let key = rule.resolve_key(&ctx.request.headers);
        let outcome = rule
            .backend
            .reserve(ReserveRequest {
                key: key.clone(),
                estimate: rule.estimate_tokens,
                now_ms,
            })
            .await;
        Ok(Self::handle_reserve_outcome(ctx, rule_index, rule, key, outcome))
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
        }
        Ok(FilterAction::Continue)
    }
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
