// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey sliding-window state on sub-window counters.
//!
//! Usage lives in one integer counter per sub-window; admission sums the
//! counters covering the window with one `MGET`, then charges the current
//! sub-window with one `INCRBY`. Each reservation lives in its own key
//! that carries its own `PX reservation_timeout_ms` expiry; reconcile
//! watches it before reading, then atomically applies the delta and deletes
//! it -- a nil reply means the reservation was already settled by another
//! reconcile,
//! or was abandoned and has since expired unclaimed, and either way
//! reconcile is a no-op rather than a stale or duplicate charge. Caps are
//! deadline-scored zsets trimmed with one range delete; those two zsets
//! include a namespace-wide active set and a per-rule key set, so their TTL
//! is only ever
//! extended (`PEXPIRE ... NX` then `PEXPIRE ... GT`) rather than
//! overwritten, so a short-window rule's reserve can never shorten a
//! longer-window rule's still-live entries in the same namespace. No
//! sweeps, no scripts.
//!
//! Key layout (`rk` = [`key_hash`] of namespace, rule, key; `w` = a
//! budget's sub-window width in milliseconds):
//!
//! | Key | Type | TTL |
//! |-----|------|-----|
//! | `{ns}:v2:{rk}:u{w}:{index}` | integer usage of one sub-window | longest `window + timeout + w` of the budgets with width `w` |
//! | `{ns}:v2:{rk}:r:{id}` | string `"estimate\|admitted_at_ms"` | `PX` timeout |
//! | `{ns}:v2:seq` | integer | none (one key per namespace) |
//! | `{ns}:v2:active` | zset `"{rk}\|{id}"` scored by deadline | extended only (`NX`, then `GT`) |
//! | `{ns}:v2:keys:{rule_hash}` | zset `rk` scored by expiry | extended only (`NX`, then `GT`) |
//!
//! Counters are keyed by sub-window width, not by budget position, so
//! adding, removing, or reordering budgets on a reload (or replicas on
//! different configs mid-rollout) still read and write the same counters;
//! budgets that share a width share, and charge once, one set of counters.
//! Changing a window's length changes its width, and so starts that
//! window's usage from zero.

use std::sync::Arc;

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;

use super::{
    super::{
        AccountingPolicy,
        backend::{
            BackendError, BackendReserve, BackendSettlement, BackendSnapshot, ReconcileRequest, ReconcileWorker,
            ReserveRequest, TokenRateLimitStateBackend,
        },
        ledger::{Budget, DenialReason},
    },
    RuleTelemetry, ValkeyConnection, accounting_config_key, amount,
    connection::{AbortRetry, command_error, unwatch},
    count, ensure_accounting_config, extend_shared_ttl, key_hash, parse_reservation, sliding_window_config_fingerprint,
    window::{BucketRange, bucket_index, bucket_ms, retry_after_ms},
};

/// Construction parameters for [`ValkeySlidingWindowBackend`].
pub(in crate::token_rate_limit) struct ValkeySlidingWindowConfig {
    /// Filter-wide connection, cloned per rule.
    pub(in crate::token_rate_limit) valkey: ValkeyConnection,
    /// Key namespace shared by every rule on this filter.
    pub(in crate::token_rate_limit) namespace: String,
    /// Rule name, hashed into every key.
    pub(in crate::token_rate_limit) rule: String,
    /// Budgets enforced together.
    pub(in crate::token_rate_limit) budgets: Vec<Budget>,
    /// After this long an unsettled reservation stops counting as active.
    pub(in crate::token_rate_limit) reservation_timeout_ms: u64,
    /// Maximum retained keys for this rule.
    pub(in crate::token_rate_limit) max_keys: usize,
    /// Maximum unsettled reservations per namespace and algorithm.
    pub(in crate::token_rate_limit) max_active_reservations: usize,
    /// Stable policy inputs used for reservations and reconciliation.
    pub(in crate::token_rate_limit) accounting: AccountingPolicy,
}

/// Sliding-window admission state shared across replicas.
pub(in crate::token_rate_limit) struct ValkeySlidingWindowBackend {
    /// Shared connection.
    valkey: ValkeyConnection,
    /// See [`ValkeySlidingWindowConfig::namespace`].
    namespace: String,
    /// See [`ValkeySlidingWindowConfig::rule`].
    rule: String,
    /// See [`ValkeySlidingWindowConfig::budgets`].
    budgets: Vec<Budget>,
    /// See [`ValkeySlidingWindowConfig::reservation_timeout_ms`].
    reservation_timeout_ms: u64,
    /// See [`ValkeySlidingWindowConfig::max_keys`].
    max_keys: usize,
    /// See [`ValkeySlidingWindowConfig::max_active_reservations`].
    max_active_reservations: usize,
    /// Smallest capacity, for rate-limit headers.
    limit: u64,
    /// Distinct sub-window widths across `budgets`, each written once.
    widths: Vec<CounterWidth>,
    /// Persistent marker preventing replicas with incompatible accounting
    /// semantics from sharing this rule's state.
    config_fingerprint: String,
    /// Background reconciliation queue.
    worker: ReconcileWorker,
    /// Last observed state for gauges, shared with the worker clone.
    telemetry: Arc<RuleTelemetry>,
}

/// One set of sub-window counters, shared by every budget of its width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CounterWidth {
    /// Sub-window width in milliseconds; part of the counter key.
    bucket_ms: u64,
    /// Counter TTL: the longest `window + timeout + bucket_ms` among the
    /// budgets of this width, so the longest window still sees its usage.
    ttl_ms: u64,
}

/// Everything admission needs to read, fetched in one pipeline.
struct WindowReads {
    /// Per budget, the counter values oldest first (`None` = absent).
    buckets: Vec<Vec<Option<i64>>>,
    /// Namespace-wide unsettled reservations after trimming.
    active: usize,
    /// This rule's retained keys after trimming.
    keys: usize,
    /// Whether this key is already retained.
    key_known: bool,
}

impl ValkeySlidingWindowBackend {
    /// Build a rule's backend over the filter's shared connection.
    pub(in crate::token_rate_limit) fn new(config: ValkeySlidingWindowConfig) -> Self {
        let limit = config.budgets.iter().map(|budget| budget.capacity).min().unwrap_or(0);
        let widths = counter_widths(&config.budgets, config.reservation_timeout_ms);
        let config_fingerprint = sliding_window_config_fingerprint(
            &config.budgets,
            config.reservation_timeout_ms,
            config.max_keys,
            config.max_active_reservations,
            &config.accounting,
        );
        Self {
            valkey: config.valkey,
            namespace: config.namespace,
            rule: config.rule,
            budgets: config.budgets,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            limit,
            widths,
            config_fingerprint,
            worker: ReconcileWorker::new(),
            telemetry: Arc::new(RuleTelemetry::default()),
        }
    }

    /// Clone this backend's connection/config, but with a detached
    /// [`ReconcileWorker`] -- used only to hand the background worker its
    /// own handle to `reserve`/`reconcile` (see [`ReconcileWorker::detached`]).
    fn clone_without_sender(&self) -> Self {
        Self {
            valkey: self.valkey.clone(),
            namespace: self.namespace.clone(),
            rule: self.rule.clone(),
            budgets: self.budgets.clone(),
            reservation_timeout_ms: self.reservation_timeout_ms,
            max_keys: self.max_keys,
            max_active_reservations: self.max_active_reservations,
            limit: self.limit,
            widths: self.widths.clone(),
            config_fingerprint: self.config_fingerprint.clone(),
            worker: ReconcileWorker::detached(),
            telemetry: Arc::clone(&self.telemetry),
        }
    }

    /// Lazily spawn the background reconciliation worker on the calling
    /// Tokio runtime, at most once per backend instance.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] if called outside a Tokio
    /// runtime context.
    fn start_worker(&self) -> Result<(), BackendError> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_error| BackendError::Unavailable("Valkey reconciliation requires a Tokio runtime".into()))?;
        self.worker.start(&runtime, || self.clone_without_sender());
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Key layout
    // -------------------------------------------------------------------------

    /// Opaque per-key identifier; no user value reaches the keyspace.
    fn key_id(&self, key: &str) -> String {
        key_hash(&[self.namespace.as_bytes(), self.rule.as_bytes(), key.as_bytes()])
    }

    /// Key prefix shared by every physical key derived from `id`.
    fn key_prefix(&self, id: &str) -> String {
        format!("{}:v2:{id}", self.namespace)
    }

    /// Counter for the sub-window of a `window_ms` window that contains
    /// `at_ms`.
    ///
    /// Test-only: production code always already holds a computed
    /// `prefix` (see [`Self::usage_key_at`]) rather than re-deriving it
    /// from a raw `key` on every call.
    #[cfg(test)]
    fn usage_key(&self, key: &str, window_ms: u64, at_ms: u64) -> String {
        let width = bucket_ms(window_ms);
        Self::usage_key_at(&self.key_prefix(&self.key_id(key)), width, bucket_index(at_ms, width))
    }

    /// Counter for sub-window `index` of width `bucket_ms`, given an
    /// already-computed `prefix`.
    fn usage_key_at(prefix: &str, bucket_ms: u64, index: u64) -> String {
        format!("{prefix}:u{bucket_ms}:{index}")
    }

    /// Per-reservation key holding `"{estimate}|{admitted_at_ms}"`. Its
    /// own `PX reservation_timeout_ms` expiry (set at reserve time) is
    /// what makes an abandoned reservation (worker queue full, retries
    /// exhausted, a reply lost after the write already applied) stop
    /// counting as active on its own, without a sweep.
    fn reservation_key(prefix: &str, id: u64) -> String {
        format!("{prefix}:r:{id}")
    }

    /// Namespace-wide reservation-id sequence.
    fn seq_key(&self) -> String {
        format!("{}:v2:seq", self.namespace)
    }

    /// Namespace-wide unsettled-reservation deadline zset.
    fn active_index_key(&self) -> String {
        format!("{}:v2:active", self.namespace)
    }

    /// Per-rule retained-key deadline zset.
    fn keys_key(&self) -> String {
        format!("{}:v2:keys:{}", self.namespace, key_hash(&[self.rule.as_bytes()]))
    }

    /// TTL for the bookkeeping keys. It covers every usage-counter TTL, so a
    /// retained-key index cannot expire while a counter still carries quota.
    fn state_ttl_ms(&self) -> u64 {
        self.widths.iter().map(|width| width.ttl_ms).max().unwrap_or(1_000)
    }

    /// Persistent marker for this rule's sliding-window accounting semantics.
    fn accounting_config_key(&self) -> String {
        accounting_config_key(&self.namespace, "sw", &self.rule)
    }

    /// Initialize or validate the marker before reading or mutating quota
    /// state. This is deliberately a separate pipeline so a mismatched
    /// writer does not even trim shared indexes.
    async fn ensure_accounting_config(&self) -> Result<(), BackendError> {
        let state_index = self.keys_key();
        ensure_accounting_config(
            &self.valkey,
            &self.namespace,
            &self.accounting_config_key(),
            &self.config_fingerprint,
            &state_index,
        )
        .await
    }

    // -------------------------------------------------------------------------
    // Reserve
    // -------------------------------------------------------------------------

    /// Build the read pipeline for [`Self::read_window`]: one `MGET` per
    /// budget covering its window, plus the active and per-rule key caps.
    fn build_read_pipeline(&self, prefix: &str, id: &str, now_ms: u64) -> redis::Pipeline {
        let mut pipe = redis::pipe();
        for budget in &self.budgets {
            let width = bucket_ms(budget.window_ms);
            let mut mget = redis::cmd("MGET");
            for bucket in BucketRange::covering(now_ms, budget.window_ms).indexes() {
                mget.arg(Self::usage_key_at(prefix, width, bucket));
            }
            pipe.add_command(mget);
        }
        pipe.cmd("ZREMRANGEBYSCORE")
            .arg(self.active_index_key())
            .arg("-inf")
            .arg(now_ms)
            .ignore();
        pipe.cmd("ZCARD").arg(self.active_index_key());
        pipe.cmd("ZREMRANGEBYSCORE")
            .arg(self.keys_key())
            .arg("-inf")
            .arg(now_ms)
            .ignore();
        pipe.cmd("ZCARD").arg(self.keys_key());
        pipe.cmd("ZSCORE").arg(self.keys_key()).arg(id);
        pipe
    }

    /// Parse [`Self::build_read_pipeline`]'s flattened reply into [`WindowReads`].
    fn parse_read_reply(&self, reply: Vec<redis::Value>) -> Result<WindowReads, BackendError> {
        let mut values = reply.into_iter();
        let mut buckets = Vec::with_capacity(self.budgets.len());
        for _budget in &self.budgets {
            let value = values.next().ok_or(BackendError::InvalidResponse)?;
            buckets.push(
                redis::from_redis_value::<Vec<Option<i64>>>(value).map_err(|_error| BackendError::InvalidResponse)?,
            );
        }
        let active = count(next_i64(&mut values)?)?;
        let keys = count(next_i64(&mut values)?)?;
        let key_known = !matches!(values.next(), Some(redis::Value::Nil) | None);
        Ok(WindowReads {
            buckets,
            active,
            keys,
            key_known,
        })
    }

    /// Fetch everything admission needs for `id` at `now_ms` in one pipeline.
    async fn read_window(&self, id: &str, now_ms: u64) -> Result<WindowReads, BackendError> {
        let prefix = self.key_prefix(id);
        let pipe = self.build_read_pipeline(&prefix, id, now_ms);
        let reply: Vec<redis::Value> = self.valkey.pipeline(&pipe).await?;
        self.parse_read_reply(reply)
    }

    /// One budget's contribution to [`Self::decide`]: its remaining
    /// balance, usage after admitting `estimate`, and -- if admitting
    /// would exceed capacity -- the retry hint for its own window. A
    /// negative counter (a refund that recreated an expired one) counts as
    /// zero, so it never offsets usage in another sub-window.
    fn evaluate_budget(counters: &[Option<i64>], budget: &Budget, estimate: u64, now_ms: u64) -> BudgetUsage {
        let usage = u64::try_from(
            counters
                .iter()
                .flatten()
                .map(|value| (*value).max(0))
                .fold(0_i64, i64::saturating_add),
        )
        .unwrap_or(u64::MAX);
        let remaining = budget.capacity.saturating_sub(usage);
        let after = usage.saturating_add(estimate);
        if after <= budget.capacity {
            return BudgetUsage {
                remaining,
                after,
                retry_after_ms: 0,
            };
        }
        let retry_after_ms = Self::retry_when_enough_usage_expires(counters, budget, now_ms, after);
        BudgetUsage {
            remaining,
            after,
            retry_after_ms,
        }
    }

    /// The first expiry that frees enough counted usage for this estimate.
    fn retry_when_enough_usage_expires(counters: &[Option<i64>], budget: &Budget, now_ms: u64, after: u64) -> u64 {
        let range = BucketRange::covering(now_ms, budget.window_ms);
        let needed = after.saturating_sub(budget.capacity);
        let mut expiring = 0_u64;
        let mut freeing = range.first;
        for (index, value) in range.indexes().zip(counters) {
            expiring = expiring.saturating_add(u64::try_from(value.unwrap_or(0).max(0)).unwrap_or(u64::MAX));
            freeing = index;
            if expiring >= needed {
                break;
            }
        }
        retry_after_ms(now_ms, budget.window_ms, freeing)
    }

    /// The admission decision for `estimate` tokens given the current window
    /// `reads`. Two concurrent requests that both read before either charges
    /// may both be admitted (overshoot bounded by their combined estimates).
    ///
    /// `max_keys` (per rule) and `max_active_reservations` (per namespace)
    /// are enforced without a distributed lock: two requests on any
    /// replica that race past the cap
    /// before either increments the shared counter may both be admitted. The
    /// overshoot is bounded by the number of concurrent requests fleet-wide at
    /// the instant the cap is crossed.
    fn decide(&self, reads: &WindowReads, estimate: u64, now_ms: u64) -> Decision {
        if !reads.key_known && reads.keys >= self.max_keys {
            return Decision::Denied {
                retry_after_ms: 0,
                remaining: 0,
                reason: DenialReason::KeyCapacity,
            };
        }
        let mut remaining = self.limit;
        let mut max_usage = 0_u64;
        let mut retry_after = 0_u64;
        for (index, budget) in self.budgets.iter().enumerate() {
            let counters = reads.buckets.get(index).map(Vec::as_slice).unwrap_or_default();
            let usage = Self::evaluate_budget(counters, budget, estimate, now_ms);
            remaining = remaining.min(usage.remaining);
            max_usage = max_usage.max(usage.after);
            retry_after = retry_after.max(usage.retry_after_ms);
        }
        if let Some((retry_after_ms, reason)) = self.denial(reads, retry_after) {
            return Decision::Denied {
                retry_after_ms,
                remaining,
                reason,
            };
        }
        Decision::Admit {
            max_usage,
            remaining: remaining.saturating_sub(estimate),
        }
    }

    /// Whether `reads` denies an already-retained key admission -- the
    /// active-reservation cap, or `retry_after` (`0` when no budget is
    /// exhausted) -- and if so the retry hint and reason to report. The
    /// `max_keys` denial is decided earlier by [`Self::decide`].
    fn denial(&self, reads: &WindowReads, retry_after: u64) -> Option<(u64, DenialReason)> {
        if reads.active >= self.max_active_reservations {
            return Some((self.reservation_timeout_ms, DenialReason::ReservationCapacity));
        }
        (retry_after > 0).then_some((retry_after, DenialReason::WindowCapacity))
    }

    /// Record the reservation itself in its own key and the two
    /// deadline zsets, extending (never shortening) their
    /// shared TTL.
    fn record_active_reservation(&self, pipe: &mut redis::Pipeline, key_id: &str, id: u64, request: &ReserveRequest) {
        let reservation = Self::reservation_key(&self.key_prefix(key_id), id);
        pipe.cmd("SET")
            .arg(&reservation)
            .arg(format!("{}|{}", request.estimate, request.now_ms))
            .arg("PX")
            .arg(self.reservation_timeout_ms)
            .ignore();
        let active_index = self.active_index_key();
        pipe.cmd("ZADD")
            .arg(&active_index)
            .arg(request.now_ms.saturating_add(self.reservation_timeout_ms))
            .arg(format!("{key_id}|{id}"))
            .ignore();
        extend_shared_ttl(pipe, &active_index, self.state_ttl_ms());
        self.retain_key(pipe, key_id, request.now_ms);
    }

    /// Record a key's latest state deadline without shortening an existing
    /// deadline, and keep the shared index alive for the same state lifetime.
    fn retain_key(&self, pipe: &mut redis::Pipeline, key_id: &str, now_ms: u64) {
        let keys = self.keys_key();
        pipe.cmd("ZADD")
            .arg(&keys)
            .arg("GT")
            .arg(now_ms.saturating_add(self.state_ttl_ms()))
            .arg(key_id)
            .ignore();
        extend_shared_ttl(pipe, &keys, self.state_ttl_ms());
    }

    /// Charge the estimate and record the reservation atomically.
    /// Called only on the admit path; two requests that race between the read
    /// and this write may both be admitted (overshoot bounded by concurrency).
    async fn admit(&self, key_id: &str, request: &ReserveRequest) -> Result<u64, BackendError> {
        let prefix = self.key_prefix(key_id);
        let mut seq = redis::pipe();
        seq.cmd("INCR").arg(self.seq_key());
        let (id,): (i64,) = self.valkey.pipeline(&seq).await?;
        let id = amount(id)?;
        let delta = i64::try_from(request.estimate).unwrap_or(i64::MAX);
        let mut pipe = redis::pipe();
        pipe.atomic();
        self.add_usage_delta(&mut pipe, &prefix, request.now_ms, delta);
        self.record_active_reservation(&mut pipe, key_id, id, request);
        let () = self.valkey.pipeline(&pipe).await?;
        Ok(id)
    }

    // -------------------------------------------------------------------------
    // Reconcile
    // -------------------------------------------------------------------------

    /// Apply `delta` once per distinct sub-window width to the sub-window
    /// that held the reservation admitted at `admitted_at_ms`, always
    /// re-arming the TTL so a counter recreated after expiry does not live
    /// forever.
    fn add_usage_delta(&self, pipe: &mut redis::Pipeline, prefix: &str, admitted_at_ms: u64, delta: i64) {
        for width in &self.widths {
            let key = Self::usage_key_at(prefix, width.bucket_ms, bucket_index(admitted_at_ms, width.bucket_ms));
            pipe.cmd("INCRBY").arg(&key).arg(delta).ignore();
            extend_shared_ttl(pipe, &key, width.ttl_ms);
        }
    }

    /// Read the reservation under `WATCH`. The same connection executes
    /// the settlement so an expired or concurrently settled reservation
    /// aborts `EXEC` instead of applying a stale delta.
    async fn read_reservation(
        &self,
        connection: &mut MultiplexedConnection,
        reservation: &str,
    ) -> Result<Option<String>, BackendError> {
        let mut read = redis::pipe();
        read.cmd("WATCH").arg(reservation).ignore();
        read.cmd("GET").arg(reservation);
        let (value,): (Option<String>,) = read
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        Ok(value)
    }

    /// Apply the delta and delete the watched reservation in one transaction.
    /// A lost `EXEC` reply can safely be retried: either all writes happened
    /// and the next read finds no reservation, or none happened and it can
    /// still be settled.
    #[expect(
        clippy::too_many_arguments,
        reason = "transaction fields are kept explicit at the Valkey write boundary"
    )]
    fn settlement_pipeline(
        &self,
        key_id: &str,
        prefix: &str,
        reservation: &str,
        active_member: &str,
        admitted_at_ms: u64,
        now_ms: u64,
        delta: i64,
    ) -> redis::Pipeline {
        let mut pipe = redis::pipe();
        pipe.atomic();
        self.add_usage_delta(&mut pipe, prefix, admitted_at_ms, delta);
        self.retain_key(&mut pipe, key_id, now_ms);
        pipe.cmd("DEL").arg(reservation).ignore();
        pipe.cmd("ZREM")
            .arg(self.active_index_key())
            .arg(active_member)
            .ignore();
        pipe
    }

    /// One settlement attempt; `None` means another writer changed the
    /// reservation before `EXEC`, so the caller retries within its deadline.
    #[expect(
        clippy::too_many_lines,
        reason = "one settlement attempt keeps the watched read and atomic write together"
    )]
    async fn reconcile_attempt(
        &self,
        connection: &mut MultiplexedConnection,
        request: &ReconcileRequest,
    ) -> Result<Option<BackendSettlement>, BackendError> {
        let id = self.key_id(&request.key);
        let prefix = self.key_prefix(&id);
        let reservation = Self::reservation_key(&prefix, request.reservation_id);
        let Some(value) = self.read_reservation(connection, &reservation).await? else {
            unwatch(connection).await?;
            return Ok(Some(BackendSettlement::Noop));
        };
        let (estimate, admitted_at_ms) = parse_reservation(&value)?;
        let actual = request.actual.unwrap_or(estimate);
        let delta = i64::try_from(actual)
            .unwrap_or(i64::MAX)
            .saturating_sub(i64::try_from(estimate).unwrap_or(i64::MAX));
        let active_member = format!("{id}|{}", request.reservation_id);
        let pipe = self.settlement_pipeline(
            &id,
            &prefix,
            &reservation,
            &active_member,
            admitted_at_ms,
            request.now_ms,
            delta,
        );
        let executed: Option<()> = pipe
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        Ok(executed.map(|()| BackendSettlement::Applied {
            actual,
            refund: estimate.saturating_sub(actual),
            overage: actual.saturating_sub(estimate),
        }))
    }
}

/// The distinct sub-window widths of `budgets`, each with the longest
/// counter TTL among the budgets that share it.
fn counter_widths(budgets: &[Budget], reservation_timeout_ms: u64) -> Vec<CounterWidth> {
    let mut widths: Vec<CounterWidth> = Vec::with_capacity(budgets.len());
    for budget in budgets {
        let width = bucket_ms(budget.window_ms);
        let ttl_ms = budget
            .window_ms
            .saturating_add(reservation_timeout_ms)
            .saturating_add(width);
        match widths.iter_mut().find(|known| known.bucket_ms == width) {
            Some(known) => known.ttl_ms = known.ttl_ms.max(ttl_ms),
            None => widths.push(CounterWidth {
                bucket_ms: width,
                ttl_ms,
            }),
        }
    }
    widths
}

/// One budget's contribution to a [`Decision`], computed by
/// [`ValkeySlidingWindowBackend::evaluate_budget`].
struct BudgetUsage {
    /// This budget's own remaining balance.
    remaining: u64,
    /// Usage in this budget's window after admitting the estimate.
    after: u64,
    /// Retry hint if admitting would exceed this budget's capacity, else `0`.
    retry_after_ms: u64,
}

/// Outcome of [`ValkeySlidingWindowBackend::decide`].
enum Decision {
    /// Admit; `max_usage` feeds soft tiers, `remaining` the gauge.
    Admit {
        /// Total usage across the window immediately after admission.
        max_usage: u64,
        /// Remaining budget for the gauge.
        remaining: u64,
    },
    /// Deny with the given retry hint.
    Denied {
        /// Conservative delay before another admission attempt.
        retry_after_ms: u64,
        /// Remaining budget for the gauge.
        remaining: u64,
        /// Distinguishes budget exhaustion from the `max_keys` cap.
        reason: DenialReason,
    },
}

/// Read one `i64` reply out of a flattened pipeline reply iterator.
fn next_i64(values: &mut impl Iterator<Item = redis::Value>) -> Result<i64, BackendError> {
    let value = values.next().ok_or(BackendError::InvalidResponse)?;
    redis::from_redis_value::<i64>(value).map_err(|_error| BackendError::InvalidResponse)
}

#[async_trait]
impl TokenRateLimitStateBackend for ValkeySlidingWindowBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        Box::pin(self.ensure_accounting_config()).await?;
        let id = self.key_id(&request.key);
        let reads = Box::pin(self.read_window(&id, request.now_ms)).await?;
        let (keys_after, active_after) = (reads.keys, reads.active);
        match self.decide(&reads, request.estimate, request.now_ms) {
            Decision::Denied {
                retry_after_ms,
                remaining,
                reason,
            } => {
                self.telemetry.record(remaining, active_after, keys_after);
                Ok(BackendReserve::Denied {
                    retry_after_ms,
                    remaining,
                    reason,
                })
            },
            Decision::Admit { max_usage, remaining } => {
                let reservation_id = Box::pin(self.admit(&id, &request)).await?;
                let keys_after = keys_after.saturating_add(usize::from(!reads.key_known));
                self.telemetry
                    .record(remaining, active_after.saturating_add(1), keys_after);
                Ok(BackendReserve::Admitted {
                    reservation_id,
                    estimate: request.estimate,
                    usage_after: max_usage,
                    remaining,
                })
            },
        }
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "transaction.finish() consumes the connection after the borrowed attempt"
    )]
    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        Box::pin(self.ensure_accounting_config()).await?;
        let mut retry = AbortRetry::start();
        loop {
            let mut transaction = self.valkey.transaction().await?;
            let outcome = self.reconcile_attempt(transaction.inner(), &request).await?;
            transaction.finish().await;
            if let Some(settlement) = outcome {
                return Ok(settlement);
            }
            retry.pause().await?;
        }
    }

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        self.start_worker()?;
        self.worker.enqueue(request)
    }

    fn limit(&self) -> u64 {
        self.limit
    }

    fn snapshot(&self) -> BackendSnapshot {
        self.telemetry.snapshot()
    }

    fn backend_name(&self) -> &'static str {
        "valkey"
    }

    fn algorithm_name(&self) -> &'static str {
        "sliding_window"
    }

    fn rule_name(&self) -> &str {
        &self.rule
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::{
        sync::OnceLock,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        super::{
            super::backend::{
                BackendReserve, BackendSettlement, ReconcileRequest, ReserveRequest, TokenRateLimitStateBackend as _,
            },
            ValkeyConnection,
        },
        BucketRange, CounterWidth, DenialReason, ValkeySlidingWindowBackend, ValkeySlidingWindowConfig, counter_widths,
        retry_after_ms,
    };
    use crate::token_rate_limit::{AccountingPolicy, CompiledEstimation, ledger::Budget, weights::TokenWeights};

    fn backend(
        namespace: &str,
        capacity: u64,
        window_ms: u64,
        reservation_timeout_ms: u64,
    ) -> Option<ValkeySlidingWindowBackend> {
        backend_with(
            namespace,
            "rule",
            vec![Budget { window_ms, capacity }],
            reservation_timeout_ms,
        )
    }

    /// One value per test process: keys from an earlier run with a reused
    /// pid may still be live in a long-lived Valkey.
    fn run_id() -> u128 {
        static RUN_ID: OnceLock<u128> = OnceLock::new();
        *RUN_ID.get_or_init(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_millis())
        })
    }

    fn backend_with(
        namespace: &str,
        rule: &str,
        budgets: Vec<Budget>,
        reservation_timeout_ms: u64,
    ) -> Option<ValkeySlidingWindowBackend> {
        let url = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL").ok()?;
        Some(ValkeySlidingWindowBackend::new(ValkeySlidingWindowConfig {
            valkey: ValkeyConnection::new(url).unwrap(),
            namespace: format!("praxis:test:{namespace}:{}:{}", std::process::id(), run_id()),
            rule: rule.to_owned(),
            budgets,
            reservation_timeout_ms,
            max_keys: 2,
            max_active_reservations: 2,
            accounting: AccountingPolicy {
                estimation: CompiledEstimation::Fixed { estimate: 1 },
                weights: TokenWeights::UNITY,
                key_fingerprint: "test-key-policy".to_owned(),
            },
        }))
    }

    async fn pttl(backend: &ValkeySlidingWindowBackend, key: &str) -> i64 {
        let mut pipe = redis::pipe();
        pipe.cmd("PTTL").arg(key);
        let (ttl,): (i64,) = backend.valkey.pipeline(&pipe).await.unwrap();
        ttl
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keeps the no-local-state backend contract assertions together"
    )]
    #[test]
    fn valkey_sliding_window_backend_has_no_local_state_to_reconcile_or_clean_up_synchronously() {
        let backend = ValkeySlidingWindowBackend::new(ValkeySlidingWindowConfig {
            valkey: ValkeyConnection::new("redis://127.0.0.1:1".into()).unwrap(),
            namespace: "ns".into(),
            rule: "default".into(),
            budgets: vec![Budget {
                window_ms: 1_000,
                capacity: 10,
            }],
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
            accounting: AccountingPolicy {
                estimation: CompiledEstimation::Fixed { estimate: 1 },
                weights: TokenWeights::UNITY,
                key_fingerprint: "test-key-policy".to_owned(),
            },
        });
        assert!(
            backend.cleanup(0, 8).is_none(),
            "Valkey-backed state has no local ledger to clean up in-process"
        );
        assert!(
            backend
                .reconcile_sync(&ReconcileRequest {
                    key: "a".into(),
                    reservation_id: 1,
                    actual: Some(1),
                    now_ms: 0,
                })
                .is_none(),
            "Valkey-backed reconciliation must go through enqueue_reconcile, not reconcile_sync"
        );
        drop(backend);
    }

    fn one_budget(window_ms: u64) -> Vec<Budget> {
        vec![Budget {
            window_ms,
            capacity: 100,
        }]
    }

    async fn counter_value(backend: &ValkeySlidingWindowBackend, key: &str) -> i64 {
        let mut pipe = redis::pipe();
        pipe.cmd("GET").arg(key);
        let (value,): (i64,) = backend.valkey.pipeline(&pipe).await.unwrap();
        value
    }

    fn one_width_budgets() -> Vec<Budget> {
        vec![
            Budget {
                window_ms: 60_000,
                capacity: 100,
            },
            Budget {
                window_ms: 30_000,
                capacity: 15,
            },
        ]
    }

    #[test]
    fn budgets_with_one_sub_window_width_get_one_counter_with_the_longest_ttl() {
        assert_eq!(
            counter_widths(&one_width_budgets(), 5_000),
            vec![CounterWidth {
                bucket_ms: 1_000,
                ttl_ms: 60_000 + 5_000 + 1_000,
            }],
            "a 60 s and a 30 s window share one-second counters that outlive the 60 s window"
        );
    }

    #[test]
    fn retry_waits_until_enough_sub_windows_expire_for_the_estimate() {
        let now = 1_000_000;
        let budget = Budget {
            window_ms: 60_000,
            capacity: 100,
        };
        let range = BucketRange::covering(now, budget.window_ms);
        let mut counters = vec![None; range.count];
        *counters.get_mut(0).unwrap() = Some(20);
        *counters.get_mut(1).unwrap() = Some(50);
        let usage = ValkeySlidingWindowBackend::evaluate_budget(&counters, &budget, 80, now);
        assert_eq!(usage.remaining, 30);
        assert_eq!(
            usage.retry_after_ms,
            retry_after_ms(now, budget.window_ms, range.first + 1)
        );
        assert!(
            usage.retry_after_ms > retry_after_ms(now, budget.window_ms, range.first),
            "the oldest 20 tokens alone do not make room for an 80-token estimate"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "one live test covers both rules and the cap within the second rule"
    )]
    async fn live_valkey_key_cap_is_independent_for_each_rule() {
        let (Some(mut first), Some(mut second)) = (
            backend_with("sw-rule-cap", "first", one_budget(60_000), 5_000),
            backend_with("sw-rule-cap", "second", one_budget(60_000), 5_000),
        ) else {
            return;
        };
        first.max_keys = 1;
        second.max_keys = 1;
        first.max_active_reservations = 8;
        second.max_active_reservations = 8;
        let now = 1_000_000;
        assert_ne!(
            first.keys_key(),
            second.keys_key(),
            "the rules need distinct retained-key indexes"
        );
        assert!(matches!(
            first.reserve(reserve("alice", 1, now)).await.unwrap(),
            BackendReserve::Admitted { .. }
        ));
        assert!(matches!(
            second.reserve(reserve("bob", 1, now)).await.unwrap(),
            BackendReserve::Admitted { .. }
        ));
        assert!(matches!(
            second.reserve(reserve("carol", 1, now)).await.unwrap(),
            BackendReserve::Denied {
                reason: DenialReason::KeyCapacity,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn live_valkey_budgets_with_one_sub_window_width_share_one_counter() {
        let Some(backend) = backend_with("sw-shared-width", "rule", one_width_budgets(), 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 10, now)).await.unwrap()
        else {
            panic!("10 fits both budgets");
        };
        assert_eq!(remaining, 5, "the tighter 30 s budget of 15 has 5 left");
        let counter = backend.usage_key("alice", 30_000, now);
        assert_eq!(
            counter,
            backend.usage_key("alice", 60_000, now),
            "both budgets name the same counter"
        );
        assert_eq!(
            counter_value(&backend, &counter).await,
            10,
            "the shared counter is charged once, not once per budget"
        );
        assert!(
            matches!(
                backend.reserve(reserve("alice", 10, now)).await.unwrap(),
                BackendReserve::Denied { .. }
            ),
            "the 30 s budget sees the usage in the shared counter"
        );
    }

    #[tokio::test]
    async fn live_valkey_short_ttl_rule_never_shortens_the_shared_zsets_ttl() {
        let (Some(long), Some(short)) = (
            backend_with("sw-shared-ttl", "long", one_budget(3_600_000), 5_000),
            backend_with("sw-shared-ttl", "short", one_budget(1_000), 1_000),
        ) else {
            return;
        };
        let now = 1_000_000;
        let _ = long.reserve(reserve("alice", 1, now)).await.unwrap();
        let shared = [long.active_index_key(), long.keys_key()];
        let mut before = Vec::with_capacity(shared.len());
        for key in &shared {
            before.push(pttl(&long, key).await);
        }
        let _ = short.reserve(reserve("alice", 1, now)).await.unwrap();
        for (key, before) in shared.iter().zip(before) {
            let after = pttl(&long, key).await;
            assert!(
                after > 3_600_000,
                "{key} must keep the long rule's TTL after the short rule reserved, got {after}"
            );
            assert!(
                before - after < 1_000,
                "{key}'s TTL must not be shortened by the short rule: {before} before, {after} after"
            );
        }
    }

    #[tokio::test]
    async fn live_valkey_every_written_key_expires_except_the_sequence() {
        let Some(backend) = backend("sw-ttl", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 1, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let prefix = backend.key_prefix(&backend.key_id("alice"));
        for key in [
            backend.usage_key("alice", 60_000, now),
            ValkeySlidingWindowBackend::reservation_key(&prefix, reservation_id),
            backend.active_index_key(),
            backend.keys_key(),
        ] {
            let ttl = pttl(&backend, &key).await;
            assert!(ttl > 0, "{key} must carry a TTL, got {ttl}");
        }
        assert_eq!(
            pttl(&backend, &backend.seq_key()).await,
            -1,
            "the id sequence deliberately has no TTL"
        );
    }

    fn reserve(key: &str, estimate: u64, now_ms: u64) -> ReserveRequest {
        ReserveRequest {
            key: key.to_owned(),
            estimate,
            now_ms,
        }
    }

    fn reconcile(key: &str, id: u64, actual: u64, now_ms: u64) -> ReconcileRequest {
        ReconcileRequest {
            key: key.to_owned(),
            reservation_id: id,
            actual: Some(actual),
            now_ms,
        }
    }

    #[tokio::test]
    async fn live_valkey_admits_until_the_window_is_full_and_reports_remaining() {
        let Some(backend) = backend("sw-fill", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted {
            remaining, usage_after, ..
        } = backend.reserve(reserve("alice", 60, now)).await.unwrap()
        else {
            panic!("60 of 100 fits");
        };
        assert_eq!(remaining, 40, "100 minus 60");
        assert_eq!(usage_after, 60, "usage after the first reservation");
        let BackendReserve::Denied {
            retry_after_ms,
            remaining,
            ..
        } = backend.reserve(reserve("alice", 60, now)).await.unwrap()
        else {
            panic!("another 60 exceeds the 40 left");
        };
        assert_eq!(remaining, 40, "the denial reports the checked balance");
        assert!(
            retry_after_ms > 0 && retry_after_ms <= 61_000,
            "retry waits for the oldest bucket to age out, got {retry_after_ms}"
        );
        let snapshot = backend.snapshot();
        assert_eq!(snapshot.budget_remaining, 40, "the gauge follows the last decision");
        assert_eq!(snapshot.active_keys, 1, "one retained key");
        assert_eq!(snapshot.active_reservations, 1, "one pending reservation");
    }

    #[tokio::test]
    async fn live_valkey_usage_ages_out_after_the_window() {
        let Some(backend) = backend("sw-age", 100, 60_000, 5_000) else {
            return;
        };
        let _ = backend.reserve(reserve("alice", 100, 1_000_000)).await.unwrap();
        assert!(
            matches!(
                backend.reserve(reserve("alice", 1, 1_000_000)).await.unwrap(),
                BackendReserve::Denied { .. }
            ),
            "the window is full"
        );
        let later = 1_000_000 + 60_000 + 1_000;
        assert!(
            matches!(
                backend.reserve(reserve("alice", 1, later)).await.unwrap(),
                BackendReserve::Admitted { .. }
            ),
            "one bucket past the window the usage has left"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the reconciliation test verifies exactly-once settlement and index lifetime together"
    )]
    async fn live_valkey_reconcile_refunds_the_unused_estimate_exactly_once() {
        let Some(backend) = backend("sw-reconcile", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 60, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let mut shorten = redis::pipe();
        shorten.cmd("PEXPIRE").arg(backend.keys_key()).arg(50).ignore();
        let () = backend.valkey.pipeline(&shorten).await.unwrap();
        let applied = BackendSettlement::Applied {
            actual: 40,
            refund: 20,
            overage: 0,
        };
        let settled = backend
            .reconcile(reconcile("alice", reservation_id, 40, now + 10))
            .await
            .unwrap();
        assert_eq!(settled, applied, "20 tokens come back");
        let keys_ttl = pttl(&backend, &backend.keys_key()).await;
        let expected_min = i64::try_from(backend.state_ttl_ms())
            .unwrap_or(i64::MAX)
            .saturating_sub(1_000);
        assert!(
            keys_ttl > expected_min,
            "settlement must re-arm the retained-key index with live quota state: {keys_ttl} <= {expected_min}"
        );
        let again = backend
            .reconcile(reconcile("alice", reservation_id, 40, now + 20))
            .await
            .unwrap();
        assert_eq!(again, BackendSettlement::Noop, "a repeated reconcile is a no-op");
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 60, now + 30)).await.unwrap()
        else {
            panic!("40 used, 60 fits exactly");
        };
        assert_eq!(remaining, 0, "100 - 40 - 60");
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the fault injection and retry assertions form one transaction scenario"
    )]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the transaction is consumed by finish after the aborted EXEC"
    )]
    async fn live_valkey_aborted_settlement_keeps_reservation_and_usage_for_retry() {
        let Some(backend) = backend("sw-aborted-settle", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 10, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let id = backend.key_id("alice");
        let prefix = backend.key_prefix(&id);
        let reservation = ValkeySlidingWindowBackend::reservation_key(&prefix, reservation_id);
        let mut transaction = backend.valkey.transaction().await.unwrap();
        let value = backend
            .read_reservation(transaction.inner(), &reservation)
            .await
            .unwrap()
            .unwrap();
        let mut concurrent = redis::pipe();
        concurrent
            .cmd("SET")
            .arg(&reservation)
            .arg(&value)
            .arg("PX")
            .arg(5_000)
            .ignore();
        let () = backend.valkey.pipeline(&concurrent).await.unwrap();
        let member = format!("{id}|{reservation_id}");
        let pipe = backend.settlement_pipeline(&id, &prefix, &reservation, &member, now, now, 40);
        let executed: Option<()> = pipe.query_async(transaction.inner()).await.unwrap();
        assert!(executed.is_none(), "a changed reservation aborts the whole settlement");
        transaction.finish().await;
        assert_eq!(
            counter_value(&backend, &backend.usage_key("alice", 60_000, now)).await,
            10
        );
        assert_eq!(
            backend
                .reconcile(reconcile("alice", reservation_id, 50, now + 1))
                .await
                .unwrap(),
            BackendSettlement::Applied {
                actual: 50,
                refund: 0,
                overage: 40
            }
        );
        assert_eq!(
            counter_value(&backend, &backend.usage_key("alice", 60_000, now)).await,
            50
        );
    }

    #[tokio::test]
    async fn live_valkey_duplicate_reconcile_applies_the_delta_once() {
        let Some(backend) = backend("sw-dup", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let (first, second) = tokio::join!(
            backend.reconcile(reconcile("alice", reservation_id, 10, now + 1)),
            backend.reconcile(reconcile("alice", reservation_id, 10, now + 1)),
        );
        let applied = [first.unwrap(), second.unwrap()]
            .iter()
            .filter(|settlement| matches!(settlement, BackendSettlement::Applied { .. }))
            .count();
        assert_eq!(applied, 1, "exactly one of two concurrent reconciles settles");
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now + 2)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(
            remaining, 89,
            "100 - 10 actual - 1 just reserved; the refund was applied once"
        );
    }

    #[tokio::test]
    async fn live_valkey_reconcile_after_the_reservation_timeout_is_a_noop() {
        let Some(backend) = backend("sw-timeout", 100, 60_000, 50) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let settlement = backend
            .reconcile(reconcile("alice", reservation_id, 10, now + 1))
            .await
            .unwrap();
        assert_eq!(
            settlement,
            BackendSettlement::Noop,
            "a reconcile past reservation_timeout_ms must be a no-op, matching the in-memory ledger"
        );
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now + 1)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(
            remaining, 49,
            "100 - 50 (charged at reserve, never refunded by the abandoned reconcile) - 1 just reserved"
        );
    }

    #[tokio::test]
    async fn reconcile_after_bucket_expiry_leaves_a_ttl_and_never_a_negative_sum() {
        let Some(backend) = backend("sw-expired", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let bucket = backend.usage_key("alice", 60_000, now);
        let mut pipe = redis::pipe();
        pipe.cmd("DEL").arg(&bucket).ignore();
        let () = backend.valkey.pipeline(&pipe).await.unwrap();
        let _ = backend
            .reconcile(reconcile("alice", reservation_id, 10, now + 1))
            .await
            .unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("GET").arg(&bucket);
        pipe.cmd("PTTL").arg(&bucket);
        let (value, ttl): (i64, i64) = backend.valkey.pipeline(&pipe).await.unwrap();
        assert_eq!(value, -40, "the counter was recreated by the delta");
        assert!(ttl > 0, "a recreated counter must carry a TTL, got {ttl}");
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now + 2)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(
            remaining, 99,
            "a negative counter counts as zero usage before subtracting the estimate"
        );
    }

    #[tokio::test]
    async fn live_valkey_concurrent_reserves_on_one_key_may_overshoot_but_never_lose_usage() {
        let Some(backend) = backend("sw-race", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let (first, second) = tokio::join!(
            backend.reserve(reserve("alice", 60, now)),
            backend.reserve(reserve("alice", 60, now)),
        );
        let admitted = [first.unwrap(), second.unwrap()]
            .iter()
            .filter(|reply| matches!(reply, BackendReserve::Admitted { .. }))
            .count();
        assert!(admitted >= 1, "at least one of the racing reservations is admitted");
        let expected_remaining = if admitted == 2 { 0 } else { 40 };
        let reply = backend.reserve(reserve("alice", 1, now)).await.unwrap();
        assert_eq!(
            reply
                .remaining()
                .saturating_add(if matches!(reply, BackendReserve::Admitted { .. }) {
                    1
                } else {
                    0
                }),
            expected_remaining,
            "every admitted estimate was charged, overshoot included"
        );
    }

    #[tokio::test]
    async fn live_valkey_caps_deny_new_keys_and_excess_reservations() {
        let Some(backend) = backend("sw-caps", 1_000, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let _ = backend.reserve(reserve("alice", 1, now)).await.unwrap();
        let _ = backend.reserve(reserve("bob", 1, now)).await.unwrap();
        assert!(
            matches!(
                backend.reserve(reserve("carol", 1, now)).await.unwrap(),
                BackendReserve::Denied { remaining: 0, .. }
            ),
            "max_keys is 2, and a key refused by it reports no remaining budget"
        );
        assert!(
            matches!(
                backend.reserve(reserve("alice", 1, now)).await.unwrap(),
                BackendReserve::Denied { .. }
            ),
            "max_active_reservations is 2"
        );
        let past_timeout = now + 5_000 + 1;
        assert!(
            matches!(
                backend.reserve(reserve("alice", 1, past_timeout)).await.unwrap(),
                BackendReserve::Admitted { .. }
            ),
            "expired reservations drop out of the cap"
        );
    }
}
