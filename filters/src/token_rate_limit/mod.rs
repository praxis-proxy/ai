// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token-denominated rate limiting filter.
//!
//! Implements the uncontested MVP core of `ai#658`'s token rate limiting
//! proposal (`docs/proposals/00121_token-rate-limiting.md`, tracked by
//! epic `ai#121`): a single sliding-window token budget (global, or one
//! per header value per `ai#129`), reservation-based admission reconciled
//! against actual provider-reported usage, and standard 429 responses
//! with token-denominated rate limit headers.
//!
//! State is exact sliding-window admission (see [`ledger`]), adapted from
//! nerdalert's `poc/distributed-token-rate-limit-demo` spike branch, behind
//! a pluggable [`backend`] so the same filter logic runs in-process
//! (default) or against a shared Valkey backend (`backend: {kind: valkey}`)
//! for state shared across gateway instances/replicas.
//!
//! Deliberately deferred, pending open design threads on `ai#658`:
//!
//! - **Fuller bucket keys (rest of M5)**: `ai#129`'s single-header-value keying (one budget applied uniformly per key,
//!   fallback to global) is implemented; composite/multi-dimension keys, per-model keys (`ai#123`), and CEL-expression
//!   keys (overlapping `praxis#189`/`#232`) are not.
//! - **Configurable estimation (M3)**: `estimate_tokens` is a fixed constant per rule for now, not derived from request
//!   metadata (e.g. `max_tokens`).
//! - **Token-type-aware accounting (M4)**: reconciles against `token.total` only; per-type (input/output/cached)
//!   weighting is not modeled yet.
//! - **Multiple budgets per rule**: `ai#658`'s proposal allows several `token_budgets` (e.g. hourly + daily) per rule;
//!   this MVP admits exactly one `window`/`capacity` pair.
//! - **Observability (M7/M8) and metering (S3)**: out of scope here — both are recommended to split into their own
//!   proposals on the `ai#658` review thread.
//!
//! `X-RateLimit-*` headers are emitted on 429 rejection only, matching the
//! validated pattern on the source spike branch: computing "remaining"
//! for every successful response would need an extra read on the Valkey
//! path (doubling backend round-trips per request), so both backends
//! behave identically here rather than diverging by backend.
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

use std::{sync::Arc, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
use http::header::HeaderName;
use metrics::{counter, gauge};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};

use self::{
    backend::{
        BackendReserve, InMemoryTokenRateLimitBackend, ReconcileRequest, ReserveRequest, TokenRateLimitStateBackend,
        ValkeyBackendConfig, ValkeyTokenRateLimitBackend,
    },
    config::{BackendConfig, BackendKind, DEFAULT_RESERVATION_TIMEOUT, TokenRateLimitConfig},
    ledger::{Budget, Ledger, LedgerConfig},
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

/// Sentinel key for requests with no `bucket_key_header` configured, or
/// where the configured header was absent/not valid UTF-8 on this
/// request -- both cases share one budget, sized the same as every
/// per-key budget.
const FALLBACK_KEY: &str = "__fallback__";

/// Bound on distinct `bucket_key_header` values retained at once.
///
/// App/tenant cardinality is expected to be far lower in practice; this
/// mirrors the soft cap `rate_limit` uses for per-IP entries.
const MAX_KEYS: usize = 100_000;

/// Bound on a single `bucket_key_header` value's length.
const MAX_KEY_LENGTH: usize = 256;

/// Bound on reservations awaiting reconciliation across all keys.
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

/// Build the configured state backend (in-process ledger or Valkey).
///
/// # Errors
///
/// Returns [`FilterError`] if the ledger config is invalid, a `valkey`
/// backend is configured without a `url`, or the URL fails to parse/expand.
fn build_backend(
    backend: BackendConfig,
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
            .map_err(|error| format!("token_rate_limit: {error}"))?;
            Ok(Arc::new(InMemoryTokenRateLimitBackend::new(ledger)))
        },
        BackendKind::Valkey => {
            let url = backend
                .url
                .as_deref()
                .ok_or("token_rate_limit: backend.url is required for backend.kind: valkey")?;
            let url = expand_backend_url(url)?;
            let namespace = backend
                .namespace
                .unwrap_or_else(|| "praxis:token_rate_limit".to_owned());
            Ok(Arc::new(ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
                url,
                namespace,
                rule: "default".to_owned(),
                budgets,
                reservation_timeout_ms,
                max_keys: MAX_KEYS,
                max_active_reservations: MAX_ACTIVE_RESERVATIONS,
            })?))
        },
    }
}

// -----------------------------------------------------------------------------
// TokenRateLimitFilter
// -----------------------------------------------------------------------------

/// Token-denominated rate limiter: reserves an estimated cost at
/// admission, reconciles against actual usage after the response
/// completes.
///
/// # YAML configuration
///
/// ```yaml
/// filter: token_rate_limit
/// window: 1h                    # sliding window duration
/// capacity: 100000               # max tokens admitted within `window`
/// estimate_tokens: 500          # fixed cost reserved per request at admission
/// bucket_key_header: x-app-id   # optional (ai#129): one budget per header value, else one shared budget
/// backend:                      # optional: defaults to in-process state
///   kind: valkey                 # memory (default) | valkey
///   url: "${TOKEN_RATE_LIMIT_VALKEY_URL}"
///   namespace: praxis:token_rate_limit
/// ```
pub struct TokenRateLimitFilter {
    /// Sliding-window state: in-process, or shared via Valkey.
    backend: Arc<dyn TokenRateLimitStateBackend>,

    /// Header whose value keys an independent budget, per `ai#129`.
    bucket_key_header: Option<HeaderName>,

    /// Fixed token cost reserved at admission (M3 placeholder).
    estimate_tokens: u64,

    /// Monotonic clock reference; all timestamps are offsets from this.
    epoch: Instant,
}

impl TokenRateLimitFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid, `capacity`
    /// is zero, `estimate_tokens` is zero or exceeds `capacity` (which
    /// would make every request unadmittable), `window`/
    /// `reservation_timeout` aren't valid durations, or a `valkey`
    /// backend is configured without a `url`.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenRateLimitConfig = parse_filter_config("token_rate_limit", config)?;

        if cfg.capacity == 0 {
            return Err("token_rate_limit: capacity must be greater than 0".into());
        }
        if cfg.estimate_tokens == 0 {
            return Err("token_rate_limit: estimate_tokens must be greater than 0".into());
        }
        if cfg.estimate_tokens > cfg.capacity {
            return Err("token_rate_limit: estimate_tokens must not exceed capacity".into());
        }

        let window_ms = parse_duration_ms(&cfg.window)?;
        let reservation_timeout_ms = parse_duration_ms(
            cfg.reservation_timeout
                .as_deref()
                .unwrap_or(DEFAULT_RESERVATION_TIMEOUT),
        )?;
        let budgets = vec![Budget {
            window_ms,
            capacity: cfg.capacity,
        }];
        let backend = build_backend(cfg.backend, budgets, reservation_timeout_ms)?;

        let bucket_key_header = cfg
            .bucket_key_header
            .map(|name| {
                HeaderName::try_from(name.as_str())
                    .map_err(|error| format!("token_rate_limit: invalid bucket_key_header: {error}"))
            })
            .transpose()?;

        Ok(Box::new(Self {
            backend,
            bucket_key_header,
            estimate_tokens: cfg.estimate_tokens,
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

    /// Resolve which budget key a request should use, given the configured
    /// key header (if any) and the request's header map.
    ///
    /// Returns [`FALLBACK_KEY`] when keying isn't configured at all, or
    /// when it is configured but this particular request's header is
    /// absent/not valid UTF-8 -- both cases route to one shared budget.
    fn resolve_key(&self, headers: &http::HeaderMap) -> String {
        self.bucket_key_header
            .as_ref()
            .and_then(|name| headers.get(name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty() && value.len() <= MAX_KEY_LENGTH)
            .map_or_else(|| FALLBACK_KEY.to_owned(), str::to_owned)
    }

    /// Reclaim idle/orphaned in-process ledger state and publish gauges.
    ///
    /// No-ops for a Valkey backend (`local_state()` returns `None`):
    /// expiry there is handled by the Lua reserve script itself.
    fn cleanup_and_record_state(&self, now_ms: u64) {
        let Some((ledger, _)) = self.backend.local_state() else {
            return;
        };
        let orphaned = ledger.cleanup(now_ms, 1);
        if orphaned > 0 {
            counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "orphaned")
                .increment(orphaned as u64);
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "metrics gauges use f64, bounded by config caps in practice"
        )]
        {
            gauge!("praxis_ai_token_rate_limit_active_reservations").set(ledger.active_count() as f64);
            gauge!("praxis_ai_token_rate_limit_active_keys").set(ledger.key_count() as f64);
        }
    }

    /// Record metrics/metadata for an admitted reservation.
    fn record_admission(&self, ctx: &mut HttpFilterContext<'_>, key: String, reservation_id: u64, estimate: u64) {
        counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "admitted").increment(1);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "estimated").increment(estimate);
        ctx.set_metadata(META_RESERVATION_ID, reservation_id.to_string());
        ctx.set_metadata(META_BUCKET_KEY, key);
        self.cleanup_and_record_state(self.now_ms());
    }

    /// Build the 429 rejection for a denied reservation, including the
    /// token-denominated rate limit headers.
    fn denied_action(&self, retry_after_ms: u64) -> FilterAction {
        counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied").increment(1);
        let retry_secs = retry_after_ms.saturating_add(999) / 1000;
        let retry_secs = retry_secs.max(1);
        FilterAction::Reject(
            Rejection::status(429)
                .with_header("Retry-After", retry_secs.to_string())
                .with_header(HEADER_RATELIMIT_LIMIT_TOKENS, self.backend.limit().to_string())
                .with_header(HEADER_RATELIMIT_REMAINING_TOKENS, "0")
                .with_header(HEADER_RATELIMIT_RESET, retry_secs.to_string()),
        )
    }

    /// Reconcile a prior reservation against actual usage, if known.
    ///
    /// No-ops if either the reservation or `token.total` metadata is
    /// absent -- reservation stands as final rather than guessing.
    ///
    /// In-process state reconciles synchronously and immediately (cheap,
    /// no I/O). A Valkey backend instead enqueues the reconciliation onto
    /// a background worker (see `backend::ValkeyTokenRateLimitBackend`) so
    /// the response is never held up on a network round-trip that has no
    /// bearing on whether *this* request was admitted.
    fn reconcile(&self, ctx: &HttpFilterContext<'_>) {
        let Some(reservation_id) = ctx
            .get_metadata(META_RESERVATION_ID)
            .and_then(|v| v.parse::<u64>().ok())
        else {
            return;
        };
        let Some(key) = ctx.get_metadata(META_BUCKET_KEY).map(str::to_owned) else {
            return;
        };
        let actual = ctx.get_metadata(META_TOKEN_TOTAL).and_then(|v| v.parse::<u64>().ok());
        if actual.is_none() {
            tracing::trace!("token_rate_limit: no token.total metadata at end of stream, charging at estimate");
        }
        let now_ms = self.now_ms();

        if let Some((ledger, _)) = self.backend.local_state() {
            let settlement = ledger.reconcile(reservation_id, actual, now_ms);
            record_settlement_metrics(&settlement);
            tracing::debug!(
                ?settlement,
                "token_rate_limit: reconciled reservation against actual usage"
            );
            return;
        }
        if let Err(error) = self.backend.enqueue_reconcile(ReconcileRequest {
            key,
            reservation_id,
            actual,
            estimate: self.estimate_tokens,
            now_ms,
        }) {
            tracing::error!(%error, "token_rate_limit: failed to enqueue reconciliation");
        }
    }
}

/// Emit counters for a completed synchronous (in-process) reconciliation.
fn record_settlement_metrics(settlement: &ledger::Settlement) {
    if let ledger::Settlement::Applied {
        actual,
        refund,
        overage,
    } = *settlement
    {
        counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "reconciled").increment(1);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "actual").increment(actual);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "refunded").increment(refund);
        counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "overage").increment(overage);
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
        self.cleanup_and_record_state(now_ms);
        let key = self.resolve_key(&ctx.request.headers);

        match self
            .backend
            .reserve(ReserveRequest {
                key: key.clone(),
                estimate: self.estimate_tokens,
                now_ms,
            })
            .await
        {
            Ok(BackendReserve::Admitted {
                reservation_id,
                estimate,
            }) => {
                self.record_admission(ctx, key, reservation_id, estimate);
                Ok(FilterAction::Continue)
            },
            Ok(BackendReserve::Denied { retry_after_ms }) => {
                tracing::info!(
                    estimate = self.estimate_tokens,
                    key,
                    "token_rate_limit: rejecting request (429)"
                );
                Ok(self.denied_action(retry_after_ms))
            },
            Err(error) => {
                counter!("praxis_ai_token_rate_limit_backend_errors_total", "operation" => "reserve").increment(1);
                tracing::error!(%error, "token_rate_limit: admission backend failed, failing closed");
                Ok(FilterAction::Reject(Rejection::status(503)))
            },
        }
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
            .field("bucket_key_header", &self.bucket_key_header)
            .field("estimate_tokens", &self.estimate_tokens)
            .finish_non_exhaustive()
    }
}
