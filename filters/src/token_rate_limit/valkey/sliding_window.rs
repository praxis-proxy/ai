// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey sliding-window state on sub-window counters.
//!
//! Usage lives in one integer counter per sub-window; admission sums the
//! counters covering the window with one `MGET`, then charges the current
//! sub-window with one `INCRBY`. Each reservation lives in its own key
//! that carries its own `PX reservation_timeout_ms` expiry; reconcile
//! claims it with one `GETDEL`, the exactly-once settlement guard -- a nil
//! reply means the reservation was already settled by another reconcile,
//! or was abandoned and has since expired unclaimed, and either way
//! reconcile is a no-op rather than a stale or duplicate charge. Caps are
//! deadline-scored zsets trimmed with one range delete; those two zsets
//! are shared by every rule in a namespace, so their own TTL is only ever
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
//! | `{ns}:v2:keys` | zset `rk` scored by expiry | extended only (`NX`, then `GT`) |
//!
//! Counters are keyed by sub-window width, not by budget position, so
//! adding, removing, or reordering budgets on a reload (or replicas on
//! different configs mid-rollout) still read and write the same counters;
//! budgets that share a width share, and charge once, one set of counters.
//! Changing a window's length changes its width, and so starts that
//! window's usage from zero.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    super::{
        backend::{
            BackendError, BackendReserve, BackendSettlement, BackendSnapshot, ReconcileRequest, ReconcileWorker,
            ReserveRequest, TokenRateLimitStateBackend,
        },
        ledger::{Budget, DenialReason},
    },
    RuleTelemetry, ValkeyConnection, amount, count, extend_shared_ttl, key_hash, parse_reservation,
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
    /// Maximum retained keys per namespace and algorithm.
    pub(in crate::token_rate_limit) max_keys: usize,
    /// Maximum unsettled reservations per namespace and algorithm.
    pub(in crate::token_rate_limit) max_active_reservations: usize,
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
    /// Longest window, for TTLs.
    max_window_ms: u64,
    /// Distinct sub-window widths across `budgets`, each written once.
    widths: Vec<CounterWidth>,
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
    /// Namespace-wide retained keys after trimming.
    keys: usize,
    /// Whether this key is already retained.
    key_known: bool,
}

impl ValkeySlidingWindowBackend {
    /// Build a rule's backend over the filter's shared connection.
    pub(in crate::token_rate_limit) fn new(config: ValkeySlidingWindowConfig) -> Self {
        let limit = config.budgets.iter().map(|budget| budget.capacity).min().unwrap_or(0);
        let max_window_ms = config.budgets.iter().map(|budget| budget.window_ms).max().unwrap_or(0);
        let widths = counter_widths(&config.budgets, config.reservation_timeout_ms);
        Self {
            valkey: config.valkey,
            namespace: config.namespace,
            rule: config.rule,
            budgets: config.budgets,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            limit,
            max_window_ms,
            widths,
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
            max_window_ms: self.max_window_ms,
            widths: self.widths.clone(),
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

    /// Namespace-wide retained-key deadline zset.
    fn keys_key(&self) -> String {
        format!("{}:v2:keys", self.namespace)
    }

    /// TTL for namespace-wide bookkeeping keys.
    fn state_ttl_ms(&self) -> u64 {
        self.max_window_ms
            .saturating_add(self.reservation_timeout_ms)
            .max(1_000)
    }

    // -------------------------------------------------------------------------
    // Reserve
    // -------------------------------------------------------------------------

    /// Build the read pipeline for [`Self::read_window`]: one `MGET` per
    /// budget covering its window, plus the namespace-wide trims/counts.
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
        let range = BucketRange::covering(now_ms, budget.window_ms);
        let oldest = range
            .indexes()
            .zip(counters)
            .find(|(_, value)| value.is_some_and(|v| v > 0))
            .map_or(range.first, |(index, _)| index);
        BudgetUsage {
            remaining,
            after,
            retry_after_ms: retry_after_ms(now_ms, budget.window_ms, oldest),
        }
    }

    /// The admission decision given `reads` after a pre-charge has been
    /// applied (counters already include the estimate; pass `estimate = 0`).
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
    /// namespace-wide deadline zsets, extending (never shortening) their
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
        let keys = self.keys_key();
        pipe.cmd("ZADD")
            .arg(&keys)
            .arg(request.now_ms.saturating_add(self.state_ttl_ms()))
            .arg(key_id)
            .ignore();
        extend_shared_ttl(pipe, &keys, self.state_ttl_ms());
    }

    /// Charge the estimate into the current sub-window counters before
    /// reading; paired with [`Self::undo_charge`] on denial.
    async fn pre_charge(&self, prefix: &str, now_ms: u64, estimate: u64) -> Result<(), BackendError> {
        let mut pipe = redis::pipe();
        let delta = i64::try_from(estimate).unwrap_or(i64::MAX);
        self.add_usage_delta(&mut pipe, prefix, now_ms, delta);
        let () = self.valkey.pipeline(&pipe).await?;
        Ok(())
    }

    /// Undo a pre-charge that was denied.
    async fn undo_charge(&self, prefix: &str, now_ms: u64, estimate: u64) -> Result<(), BackendError> {
        let mut pipe = redis::pipe();
        let delta = -(i64::try_from(estimate).unwrap_or(i64::MAX));
        self.add_usage_delta(&mut pipe, prefix, now_ms, delta);
        let () = self.valkey.pipeline(&pipe).await?;
        Ok(())
    }

    /// Record an admitted reservation in its own key and the two
    /// namespace-wide deadline zsets, without re-charging the counters
    /// (the pre-charge already did that).
    async fn record_reservation(&self, key_id: &str, request: &ReserveRequest) -> Result<u64, BackendError> {
        let mut seq = redis::pipe();
        seq.cmd("INCR").arg(self.seq_key());
        let (id,): (i64,) = self.valkey.pipeline(&seq).await?;
        let id = amount(id)?;
        let mut pipe = redis::pipe();
        pipe.atomic();
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
            pipe.cmd("PEXPIRE").arg(&key).arg(width.ttl_ms).arg("GT").ignore();
        }
    }

    /// Apply the settled usage delta and drop the reservation from the
    /// namespace-wide active-index zset, in one pipeline -- the second
    /// (and last) round trip of [`Self::reconcile`]. `GETDEL` having
    /// returned the reservation is already the exactly-once guard, so
    /// this has no conditional revert. A settle write that fails after
    /// `GETDEL` claimed the reservation is not applied again: the worker's
    /// retry finds the reservation gone and records a `Noop`. The delta is
    /// then lost either way round -- a lost refund keeps the reservation
    /// charged at its estimate (conservative), a lost overage under-charges
    /// the window by that overage -- and the active-index member stays
    /// until its deadline trim.
    async fn settle(
        &self,
        prefix: &str,
        active_member: &str,
        admitted_at_ms: u64,
        delta: i64,
    ) -> Result<(), BackendError> {
        let mut pipe = redis::pipe();
        pipe.atomic();
        self.add_usage_delta(&mut pipe, prefix, admitted_at_ms, delta);
        pipe.cmd("ZREM")
            .arg(self.active_index_key())
            .arg(active_member)
            .ignore();
        let () = self.valkey.pipeline(&pipe).await?;
        Ok(())
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
        let id = self.key_id(&request.key);
        let prefix = self.key_prefix(&id);
        // Charge first so concurrent requests see each other's estimates
        // during the read; undo on denial.
        self.pre_charge(&prefix, request.now_ms, request.estimate).await?;
        let reads = self.read_window(&id, request.now_ms).await?;
        let (keys_after, active_after) = (reads.keys, reads.active);
        // Pass estimate=0: counters already include our charge.
        match self.decide(&reads, 0, request.now_ms) {
            Decision::Denied {
                retry_after_ms,
                remaining,
                reason,
            } => {
                // Best-effort undo; a failed undo leaves a conservative
                // over-count that expires with the sub-window TTL.
                let _ = self.undo_charge(&prefix, request.now_ms, request.estimate).await;
                self.telemetry.record(remaining, active_after, keys_after);
                Ok(BackendReserve::Denied {
                    retry_after_ms,
                    remaining,
                    reason,
                })
            },
            Decision::Admit { max_usage, remaining } => {
                let reservation_id = self.record_reservation(&id, &request).await?;
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

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        let id = self.key_id(&request.key);
        let prefix = self.key_prefix(&id);
        let mut claim = redis::pipe();
        claim
            .cmd("GETDEL")
            .arg(Self::reservation_key(&prefix, request.reservation_id));
        let (value,): (Option<String>,) = self.valkey.pipeline(&claim).await?;
        let Some(value) = value else {
            return Ok(BackendSettlement::Noop);
        };
        let (estimate, admitted_at_ms) = parse_reservation(&value)?;
        let actual = request.actual.unwrap_or(estimate);
        let delta = i64::try_from(actual)
            .unwrap_or(i64::MAX)
            .saturating_sub(i64::try_from(estimate).unwrap_or(i64::MAX));
        let active_member = format!("{id}|{}", request.reservation_id);
        self.settle(&prefix, &active_member, admitted_at_ms, delta).await?;
        Ok(BackendSettlement::Applied {
            actual,
            refund: estimate.saturating_sub(actual),
            overage: actual.saturating_sub(estimate),
        })
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
        CounterWidth, ValkeySlidingWindowBackend, ValkeySlidingWindowConfig, counter_widths,
    };
    use crate::token_rate_limit::ledger::Budget;

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
        }))
    }

    async fn pttl(backend: &ValkeySlidingWindowBackend, key: &str) -> i64 {
        let mut pipe = redis::pipe();
        pipe.cmd("PTTL").arg(key);
        let (ttl,): (i64,) = backend.valkey.pipeline(&pipe).await.unwrap();
        ttl
    }

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
    async fn live_valkey_reconcile_refunds_the_unused_estimate_exactly_once() {
        let Some(backend) = backend("sw-reconcile", 100, 60_000, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 60, now)).await.unwrap()
        else {
            panic!("admitted");
        };
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
