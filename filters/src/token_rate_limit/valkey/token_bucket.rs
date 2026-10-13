// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey token-bucket state on one optimistic transaction per operation.
//!
//! Refilling a bucket is a read-modify-write, so reserve and reconcile
//! `WATCH` the bucket hash in the same pipeline that reads it, compute the
//! refill client-side, and write under `MULTI`/`EXEC`: two round trips per
//! attempt. An aborted `EXEC` (another writer touched the bucket) is
//! retried after a short jittered backoff until [`super::connection::VALKEY_TIMEOUT`] has
//! passed since the operation started, then the backend fails closed. One
//! key therefore admits at most about one request per two round trips
//! across the fleet, and contention costs latency before it costs a 503.
//!
//! Key layout (`rk` = [`key_hash`] of namespace, `token_bucket`, rule, key):
//!
//! | Key | Type | TTL |
//! |-----|------|-----|
//! | `{ns}:v2:tb:{rk}` | hash `tokens`, `last_refill_ms` | `ceil(capacity / refill_rate)` s + timeout |
//! | `{ns}:v2:tb:{rk}:r:{id}` | string `"estimate\|admitted_at_ms"` | `PX` timeout |
//! | `{ns}:v2:tb:seq` | integer | none (one key per namespace) |
//! | `{ns}:v2:tb:active` | zset `"{rk}\|{id}"` scored by deadline | extended only (`NX`, then `GT`) |
//! | `{ns}:v2:tb:keys:{rule_hash}` | zset `rk` scored by expiry | extended only (`NX`, then `GT`) |
//!
//! Reserve, on a checked-out connection:
//!
//! ```text
//! WATCH {bucket} ; HMGET {bucket} tokens last_refill_ms
//!   ZREMRANGEBYSCORE active -inf now ; ZCARD active
//!   ZREMRANGEBYSCORE keys -inf now ; ZCARD keys ; ZSCORE keys rk
//!   INCR seq                    (an id wasted on a denial or an abort is harmless)
//!   -> refill and decide client-side; on deny: UNWATCH, return Denied
//! MULTI
//!   HSET {bucket} tokens last_refill_ms ; PEXPIRE {bucket} NX ; PEXPIRE {bucket} GT
//!   SET {bucket}:r:{id} "estimate|now" PX timeout
//!   ZADD active deadline "rk|id" ; PEXPIRE active NX ; PEXPIRE active GT
//!   ZADD keys GT expiry rk ; PEXPIRE keys NX ; PEXPIRE keys GT
//! EXEC                          (nil: back off and retry)
//! ```
//!
//! Reservation ids only need to be unique, not dense, so drawing one in
//! the read pipeline and then denying or aborting just skips it.
//!
//! Reconcile:
//!
//! ```text
//! WATCH {bucket} {bucket}:r:{id} ; GET {bucket}:r:{id} ; HMGET {bucket} tokens last_refill_ms
//!   -> nil: UNWATCH, return Noop (already settled, or abandoned and expired)
//! MULTI
//!   HSET {bucket} tokens last_refill_ms ; PEXPIRE {bucket} NX ; PEXPIRE {bucket} GT
//!   DEL {bucket}:r:{id} ; ZREM active "rk|id"
//! EXEC                          (nil: back off and retry)
//! ```
//!
//! Settlement is exactly once because of the `WATCH`: two reconciles of
//! one id both read the reservation, the first `EXEC` rewrites the bucket,
//! the second aborts, retries, and reads nil. Watching the reservation key
//! too means one that expires between the `GET` and the `EXEC` aborts the
//! transaction, so an abandoned reservation is never refunded.

use std::{future::Future, sync::Arc};

use async_trait::async_trait;
use redis::aio::MultiplexedConnection;

use super::{
    super::{
        AccountingPolicy,
        backend::{
            BackendError, BackendReserve, BackendSettlement, BackendSnapshot, ReconcileRequest, ReconcileWorker,
            ReserveRequest, TokenRateLimitStateBackend,
        },
        ledger::DenialReason,
        token_bucket_ledger,
    },
    RuleTelemetry, ValkeyConnection, accounting_config_key, amount,
    connection::{AbortRetry, command_error, unwatch},
    count, ensure_accounting_config, extend_shared_ttl, key_hash, parse_reservation, token_bucket_config_fingerprint,
};

/// Construction parameters for [`ValkeyTokenBucketBackend`].
pub(in crate::token_rate_limit) struct ValkeyTokenBucketConfig {
    /// Filter-wide connection, cloned per rule.
    pub(in crate::token_rate_limit) valkey: ValkeyConnection,
    /// Key namespace shared by every rule on this filter.
    pub(in crate::token_rate_limit) namespace: String,
    /// Rule name, hashed into every key.
    pub(in crate::token_rate_limit) rule: String,
    /// Maximum tokens held at once.
    pub(in crate::token_rate_limit) capacity: u64,
    /// Tokens refilled per second, up to `capacity`.
    pub(in crate::token_rate_limit) refill_rate: f64,
    /// After this long an unsettled reservation stops counting as active
    /// (it stays charged at its estimate).
    pub(in crate::token_rate_limit) reservation_timeout_ms: u64,
    /// Maximum retained keys for this rule.
    pub(in crate::token_rate_limit) max_keys: usize,
    /// Maximum unsettled reservations per namespace and algorithm.
    pub(in crate::token_rate_limit) max_active_reservations: usize,
    /// Stable policy inputs used for reservations and reconciliation.
    pub(in crate::token_rate_limit) accounting: AccountingPolicy,
}

/// Token-bucket admission state shared across replicas.
pub(in crate::token_rate_limit) struct ValkeyTokenBucketBackend {
    /// Shared connection and transaction pool.
    valkey: ValkeyConnection,
    /// See [`ValkeyTokenBucketConfig::namespace`].
    namespace: String,
    /// See [`ValkeyTokenBucketConfig::rule`].
    rule: String,
    /// See [`ValkeyTokenBucketConfig::capacity`].
    capacity: u64,
    /// See [`ValkeyTokenBucketConfig::refill_rate`].
    refill_rate: f64,
    /// See [`ValkeyTokenBucketConfig::reservation_timeout_ms`].
    reservation_timeout_ms: u64,
    /// See [`ValkeyTokenBucketConfig::max_keys`].
    max_keys: usize,
    /// See [`ValkeyTokenBucketConfig::max_active_reservations`].
    max_active_reservations: usize,
    /// Persistent marker preventing replicas with incompatible accounting
    /// semantics from sharing this rule's state.
    config_fingerprint: String,
    /// Background reconciliation queue.
    worker: ReconcileWorker,
    /// Last observed state for gauges, shared with the worker clone.
    telemetry: Arc<RuleTelemetry>,
}

/// `HMGET {bucket} tokens last_refill_ms`.
type BucketFields = (Option<f64>, Option<u64>);

/// Reserve's read pipeline: the bucket, `ZCARD active`, `ZCARD keys`,
/// this key's `ZSCORE` in the keys zset, and the drawn reservation id.
type ReserveReply = (BucketFields, i64, i64, Option<f64>, i64);

/// Reconcile's read pipeline: the reservation value, then the bucket.
type ReconcileReply = (Option<String>, BucketFields);

/// One key's bucket in one attempt: its names, the hash as `WATCH` read it
/// (`None` fields = no bucket yet), and the operation's clock.
struct Bucket {
    /// Opaque key identifier (`rk`).
    id: String,
    /// The bucket hash's key.
    key: String,
    /// Balance at `last_refill_ms`.
    tokens: Option<f64>,
    /// When `tokens` was last refilled.
    last_refill_ms: Option<u64>,
    /// The operation's current time.
    now_ms: u64,
}

/// Everything one reserve attempt reads under `WATCH`, in one pipeline.
struct BucketReads {
    /// The key's bucket.
    bucket: Bucket,
    /// Namespace-wide unsettled reservations after trimming.
    active: usize,
    /// This rule's retained keys after trimming.
    keys: usize,
    /// Whether this key is already retained.
    key_known: bool,
    /// Id for this attempt's reservation, drawn from `seq` in the same
    /// pipeline; unused when the attempt denies or aborts.
    reservation_id: u64,
}

impl ValkeyTokenBucketBackend {
    /// Build a rule's backend over the filter's shared connection.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] for a capacity or refill rate
    /// outside [`token_bucket_ledger::validate_capacity_and_refill_rate`]'s
    /// bounds.
    pub(in crate::token_rate_limit) fn new(config: ValkeyTokenBucketConfig) -> Result<Self, BackendError> {
        token_bucket_ledger::validate_capacity_and_refill_rate(config.capacity, config.refill_rate)
            .map_err(BackendError::Unavailable)?;
        let config_fingerprint = token_bucket_config_fingerprint(
            config.capacity,
            config.refill_rate,
            config.reservation_timeout_ms,
            config.max_keys,
            config.max_active_reservations,
            &config.accounting,
        );
        Ok(Self {
            valkey: config.valkey,
            namespace: config.namespace,
            rule: config.rule,
            capacity: config.capacity,
            refill_rate: config.refill_rate,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            config_fingerprint,
            worker: ReconcileWorker::new(),
            telemetry: Arc::new(RuleTelemetry::default()),
        })
    }

    /// Clone this backend's connection/config, but with a detached
    /// [`ReconcileWorker`] -- used only to hand the background worker its
    /// own handle to `reserve`/`reconcile` (see [`ReconcileWorker::detached`]).
    fn clone_without_sender(&self) -> Self {
        Self {
            valkey: self.valkey.clone(),
            namespace: self.namespace.clone(),
            rule: self.rule.clone(),
            capacity: self.capacity,
            refill_rate: self.refill_rate,
            reservation_timeout_ms: self.reservation_timeout_ms,
            max_keys: self.max_keys,
            max_active_reservations: self.max_active_reservations,
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

    /// Opaque per-key identifier; no user value reaches the keyspace. The
    /// `token_bucket` part keeps it apart from a sliding-window rule of the
    /// same name.
    fn key_id(&self, key: &str) -> String {
        key_hash(&[
            self.namespace.as_bytes(),
            b"token_bucket",
            self.rule.as_bytes(),
            key.as_bytes(),
        ])
    }

    /// The bucket hash for `id`; also the prefix of its reservation keys.
    fn bucket_key(&self, id: &str) -> String {
        format!("{}:v2:tb:{id}", self.namespace)
    }

    /// Per-reservation key holding `"{estimate}|{admitted_at_ms}"`; its own
    /// `PX reservation_timeout_ms` expiry retires an abandoned reservation
    /// without a sweep.
    fn reservation_key(bucket: &str, id: u64) -> String {
        format!("{bucket}:r:{id}")
    }

    /// Namespace-wide reservation-id sequence. It has no TTL: it is one key
    /// per namespace, and restarting it could reissue a live id.
    fn seq_key(&self) -> String {
        format!("{}:v2:tb:seq", self.namespace)
    }

    /// Namespace-wide unsettled-reservation deadline zset.
    fn active_index_key(&self) -> String {
        format!("{}:v2:tb:active", self.namespace)
    }

    /// Per-rule retained-key expiry zset.
    fn keys_key(&self) -> String {
        format!("{}:v2:tb:keys:{}", self.namespace, key_hash(&[self.rule.as_bytes()]))
    }

    /// Time for an empty bucket to refill, plus the reservation timeout:
    /// after that an untouched bucket is indistinguishable from a new one.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "capacity / refill_rate is validated positive and at most MAX_CAPACITY_REFILL_RATE_RATIO_SECS"
    )]
    fn state_ttl_ms(&self) -> u64 {
        let fill_ms = (self.capacity as f64 / self.refill_rate * 1000.0).ceil() as u64;
        fill_ms.saturating_add(self.reservation_timeout_ms)
    }

    /// Persistent marker for this rule's token-bucket accounting semantics.
    fn accounting_config_key(&self) -> String {
        accounting_config_key(&self.namespace, "tb", &self.rule)
    }

    /// Initialize or validate the marker before reading or mutating quota
    /// state. This is deliberately a separate pipeline so a mismatched
    /// writer cannot reinterpret the bucket before failing closed.
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
    // Bucket arithmetic
    // -------------------------------------------------------------------------

    /// A [`Bucket`] for key identifier `id` at `now_ms` from its `HMGET` fields.
    fn bucket(&self, id: String, fields: BucketFields, now_ms: u64) -> Bucket {
        Bucket {
            key: self.bucket_key(&id),
            id,
            tokens: fields.0,
            last_refill_ms: fields.1,
            now_ms,
        }
    }

    /// The balance of `bucket` refilled up to its `now_ms`, or a full
    /// bucket if there is none.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidResponse`] for a negative or NaN
    /// stored balance. One above capacity (left by a larger capacity before
    /// a config reload, or by another replica mid-rollout) is clamped.
    #[expect(
        clippy::cast_precision_loss,
        reason = "capacity is bounded by MAX_F64_SAFE_INTEGER; elapsed time only saturates the min"
    )]
    fn refill(&self, bucket: &Bucket) -> Result<f64, BackendError> {
        let capacity = self.capacity as f64;
        let (Some(tokens), Some(last_refill_ms)) = (bucket.tokens, bucket.last_refill_ms) else {
            return Ok(capacity);
        };
        if tokens.is_nan() || tokens < 0.0 {
            return Err(BackendError::InvalidResponse);
        }
        let elapsed_secs = bucket.now_ms.saturating_sub(last_refill_ms) as f64 / 1000.0;
        Ok((tokens + elapsed_secs * self.refill_rate).min(capacity))
    }

    /// Whole tokens in a balance.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "token balances are finite, non-negative, and bounded by validated capacity"
    )]
    fn whole(tokens: f64) -> u64 {
        tokens.max(0.0).floor() as u64
    }

    /// Queue the bucket write: the new balance, the refill time (never
    /// moved backwards by a replica with a lagging clock), and its TTL.
    /// Because it never moves backwards, a `last_refill_ms` ahead of the
    /// writing replica's clock freezes refill until that clock catches up:
    /// for clock skew, as long as the skew. Every write re-arms the TTL, so
    /// a corrupt far-future value freezes refill for as long as the key
    /// stays busy, not just until the TTL. Either way it is conservative:
    /// it never overcredits.
    fn write_bucket(&self, pipe: &mut redis::Pipeline, bucket: &Bucket, tokens: f64) {
        let last_refill_ms = bucket
            .last_refill_ms
            .map_or(bucket.now_ms, |last| last.max(bucket.now_ms));
        pipe.cmd("HSET")
            .arg(&bucket.key)
            .arg("tokens")
            .arg(tokens)
            .arg("last_refill_ms")
            .arg(last_refill_ms)
            .ignore();
        extend_shared_ttl(pipe, &bucket.key, self.state_ttl_ms());
    }

    // -------------------------------------------------------------------------
    // Reserve
    // -------------------------------------------------------------------------

    /// [`TokenRateLimitStateBackend::reserve`], running `between` on every
    /// attempt after the reads and before the `MULTI`. Production passes a
    /// ready no-op; tests use it to write the watched bucket.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] on Valkey errors and when
    /// aborted transactions outlast [`super::connection::VALKEY_TIMEOUT`].
    #[expect(
        clippy::significant_drop_tightening,
        reason = "transaction.finish() consumes the connection; the lint misidentifies the borrow across .await as a retained drop"
    )]
    async fn reserve_with<H, F>(&self, request: ReserveRequest, mut between: H) -> Result<BackendReserve, BackendError>
    where
        H: FnMut(u32) -> F + Send,
        F: Future<Output = ()> + Send,
    {
        Box::pin(self.ensure_accounting_config()).await?;
        let mut retry = AbortRetry::start();
        let mut attempt = 0_u32;
        loop {
            attempt = attempt.saturating_add(1);
            let mut transaction = self.valkey.transaction().await?;
            let outcome = self
                .reserve_attempt(transaction.inner(), &request, attempt, &mut between)
                .await?;
            transaction.finish().await;
            if let Some(outcome) = outcome {
                return Ok(outcome);
            }
            retry.pause().await?;
        }
    }

    /// One optimistic reserve; `None` when `EXEC` aborted.
    async fn reserve_attempt<H, F>(
        &self,
        connection: &mut MultiplexedConnection,
        request: &ReserveRequest,
        attempt: u32,
        between: &mut H,
    ) -> Result<Option<BackendReserve>, BackendError>
    where
        H: FnMut(u32) -> F + Send,
        F: Future<Output = ()> + Send,
    {
        let reads = self.read_bucket(connection, request).await?;
        let tokens = self.refill(&reads.bucket)?;
        if let Some(denied) = self.deny_reason(tokens, &reads, request.estimate) {
            unwatch(connection).await?;
            self.telemetry.record(denied.remaining(), reads.active, reads.keys);
            return Ok(Some(denied));
        }
        between(attempt).await;
        #[expect(clippy::cast_precision_loss, reason = "the estimate fits in tokens <= 2^53")]
        let tokens_after = tokens - request.estimate as f64;
        let pipe = self.reservation_pipeline(&reads.bucket, reads.reservation_id, tokens_after, request.estimate);
        let executed: Option<()> = pipe
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        Ok(executed.map(|()| self.admitted(&reads, tokens_after, request.estimate)))
    }

    /// `WATCH` the key's bucket and, in the same pipeline, read it and the
    /// namespace caps (trimming expired entries) and draw a reservation id.
    async fn read_bucket(
        &self,
        connection: &mut MultiplexedConnection,
        request: &ReserveRequest,
    ) -> Result<BucketReads, BackendError> {
        let id = self.key_id(&request.key);
        let bucket_key = self.bucket_key(&id);
        let mut pipe = redis::pipe();
        pipe.cmd("WATCH").arg(&bucket_key).ignore();
        pipe.cmd("HMGET").arg(&bucket_key).arg("tokens").arg("last_refill_ms");
        for index in [self.active_index_key(), self.keys_key()] {
            pipe.cmd("ZREMRANGEBYSCORE")
                .arg(&index)
                .arg("-inf")
                .arg(request.now_ms)
                .ignore();
            pipe.cmd("ZCARD").arg(&index);
        }
        pipe.cmd("ZSCORE").arg(self.keys_key()).arg(&id);
        pipe.cmd("INCR").arg(self.seq_key());
        let reply: redis::Value = pipe
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        let (fields, active, keys, score, reservation_id): ReserveReply =
            redis::from_redis_value(reply).map_err(|_error| BackendError::InvalidResponse)?;
        Ok(BucketReads {
            bucket: self.bucket(id, fields, request.now_ms),
            active: count(active)?,
            keys: count(keys)?,
            key_known: score.is_some(),
            reservation_id: amount(reservation_id)?,
        })
    }

    /// The denial for `estimate` against `tokens` and the namespace caps,
    /// or `None` to admit. A key refused by `max_keys` has no budget here,
    /// so it reports `0` remaining, as the in-memory ledger does.
    ///
    /// `max_keys` (per rule) and `max_active_reservations` (per namespace)
    /// are enforced without a
    /// distributed lock: two requests on any replica that race past the cap
    /// before either increments the shared counter may both be admitted. The
    /// overshoot is bounded by the number of concurrent requests fleet-wide at
    /// the instant the cap is crossed.
    fn deny_reason(&self, tokens: f64, reads: &BucketReads, estimate: u64) -> Option<BackendReserve> {
        #[expect(clippy::cast_precision_loss, reason = "compared against a balance <= 2^53")]
        let deficit = estimate as f64 - tokens;
        if !reads.key_known && reads.keys >= self.max_keys {
            return Some(BackendReserve::Denied {
                retry_after_ms: 0,
                remaining: 0,
                reason: DenialReason::KeyCapacity,
            });
        }
        if reads.active >= self.max_active_reservations {
            return Some(BackendReserve::Denied {
                retry_after_ms: self.reservation_timeout_ms,
                remaining: Self::whole(tokens),
                reason: DenialReason::ReservationCapacity,
            });
        }
        if deficit > 0.0 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a positive delay; `as` saturates for absurd estimates"
            )]
            let retry_after_ms = (deficit / self.refill_rate * 1000.0).ceil().max(1.0) as u64;
            return Some(BackendReserve::Denied {
                retry_after_ms,
                remaining: Self::whole(tokens),
                reason: DenialReason::WindowCapacity,
            });
        }
        None
    }

    /// The `MULTI` block admitting reservation `reservation_id` of `estimate`
    /// tokens against `bucket`.
    fn reservation_pipeline(
        &self,
        bucket: &Bucket,
        reservation_id: u64,
        tokens_after: f64,
        estimate: u64,
    ) -> redis::Pipeline {
        let ttl_ms = self.state_ttl_ms();
        let mut pipe = redis::pipe();
        pipe.atomic();
        self.write_bucket(&mut pipe, bucket, tokens_after);
        pipe.cmd("SET")
            .arg(Self::reservation_key(&bucket.key, reservation_id))
            .arg(format!("{estimate}|{}", bucket.now_ms))
            .arg("PX")
            .arg(self.reservation_timeout_ms)
            .ignore();
        let active_index = self.active_index_key();
        pipe.cmd("ZADD")
            .arg(&active_index)
            .arg(bucket.now_ms.saturating_add(self.reservation_timeout_ms))
            .arg(format!("{}|{reservation_id}", bucket.id))
            .ignore();
        extend_shared_ttl(&mut pipe, &active_index, ttl_ms);
        self.retain_key(&mut pipe, &bucket.id, bucket.now_ms);
        pipe
    }

    /// Record a bucket's latest state deadline without shortening an existing
    /// deadline, and keep the retained-key index alive for that state.
    fn retain_key(&self, pipe: &mut redis::Pipeline, key_id: &str, now_ms: u64) {
        let keys = self.keys_key();
        let ttl_ms = self.state_ttl_ms();
        pipe.cmd("ZADD")
            .arg(&keys)
            .arg("GT")
            .arg(now_ms.saturating_add(ttl_ms))
            .arg(key_id)
            .ignore();
        extend_shared_ttl(pipe, &keys, ttl_ms);
    }

    /// Record and report a committed admission.
    #[expect(clippy::cast_precision_loss, reason = "capacity is bounded by MAX_F64_SAFE_INTEGER")]
    fn admitted(&self, reads: &BucketReads, tokens_after: f64, estimate: u64) -> BackendReserve {
        let remaining = Self::whole(tokens_after);
        let keys = reads.keys.saturating_add(usize::from(!reads.key_known));
        self.telemetry.record(remaining, reads.active.saturating_add(1), keys);
        BackendReserve::Admitted {
            reservation_id: reads.reservation_id,
            estimate,
            usage_after: Self::whole(self.capacity as f64 - tokens_after),
            remaining,
        }
    }

    // -------------------------------------------------------------------------
    // Reconcile
    // -------------------------------------------------------------------------

    /// One optimistic reconcile; `None` when `EXEC` aborted.
    async fn reconcile_attempt(
        &self,
        connection: &mut MultiplexedConnection,
        request: &ReconcileRequest,
    ) -> Result<Option<BackendSettlement>, BackendError> {
        let (value, bucket) = self.read_reservation(connection, request).await?;
        let Some(value) = value else {
            unwatch(connection).await?;
            return Ok(Some(BackendSettlement::Noop));
        };
        let (estimate, _admitted_at_ms) = parse_reservation(&value)?;
        let actual = request.actual.unwrap_or(estimate);
        let (refund, overage) = (estimate.saturating_sub(actual), actual.saturating_sub(estimate));
        let tokens = self.settled(self.refill(&bucket)?, refund, overage);
        let pipe = self.settlement_pipeline(&bucket, request.reservation_id, tokens);
        let executed: Option<()> = pipe
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        Ok(executed.map(|()| BackendSettlement::Applied {
            actual,
            refund,
            overage,
        }))
    }

    /// `WATCH` the key's bucket and the reservation and, in the same
    /// pipeline, read them both.
    async fn read_reservation(
        &self,
        connection: &mut MultiplexedConnection,
        request: &ReconcileRequest,
    ) -> Result<(Option<String>, Bucket), BackendError> {
        let id = self.key_id(&request.key);
        let bucket_key = self.bucket_key(&id);
        let reservation_key = Self::reservation_key(&bucket_key, request.reservation_id);
        let mut read = redis::pipe();
        read.cmd("WATCH").arg(&bucket_key).arg(&reservation_key).ignore();
        read.cmd("GET").arg(&reservation_key);
        read.cmd("HMGET").arg(&bucket_key).arg("tokens").arg("last_refill_ms");
        let reply: redis::Value = read
            .query_async(connection)
            .await
            .map_err(|error| command_error(&error))?;
        let (value, fields): ReconcileReply =
            redis::from_redis_value(reply).map_err(|_error| BackendError::InvalidResponse)?;
        Ok((value, self.bucket(id, fields, request.now_ms)))
    }

    /// The `MULTI` block settling `reservation_id` to a balance of `tokens`
    /// and retiring the reservation. The bucket and retained-key index are
    /// re-armed together so the index remains evidence of live quota state.
    fn settlement_pipeline(&self, bucket: &Bucket, reservation_id: u64, tokens: f64) -> redis::Pipeline {
        let mut pipe = redis::pipe();
        pipe.atomic();
        self.write_bucket(&mut pipe, bucket, tokens);
        self.retain_key(&mut pipe, &bucket.id, bucket.now_ms);
        pipe.cmd("DEL")
            .arg(Self::reservation_key(&bucket.key, reservation_id))
            .ignore();
        pipe.cmd("ZREM")
            .arg(self.active_index_key())
            .arg(format!("{}|{reservation_id}", bucket.id))
            .ignore();
        pipe
    }

    /// `tokens` after crediting `refund` (capped at capacity) or debiting
    /// `overage` (floored at zero), as the in-memory ledger does.
    #[expect(clippy::cast_precision_loss, reason = "capacity and deltas are bounded by 2^53")]
    fn settled(&self, tokens: f64, refund: u64, overage: u64) -> f64 {
        (tokens + refund as f64 - overage as f64).clamp(0.0, self.capacity as f64)
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for ValkeyTokenBucketBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        self.reserve_with(request, |_attempt| std::future::ready(())).await
    }

    #[expect(
        clippy::significant_drop_tightening,
        reason = "transaction.finish() consumes the connection; the lint misidentifies the borrow across .await as a retained drop"
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
        self.capacity
    }

    fn snapshot(&self) -> BackendSnapshot {
        self.telemetry.snapshot()
    }

    fn backend_name(&self) -> &'static str {
        "valkey"
    }

    fn algorithm_name(&self) -> &'static str {
        "token_bucket"
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
        sync::{Arc, OnceLock},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use super::{
        super::{
            super::{
                backend::{
                    BackendError, BackendReserve, BackendSettlement, ReconcileRequest, ReserveRequest,
                    TokenRateLimitStateBackend as _,
                },
                ledger::{Budget, DenialReason},
                token_bucket_ledger::{MAX_CAPACITY_REFILL_RATE_RATIO_SECS, MAX_F64_SAFE_INTEGER},
            },
            ValkeyConnection, ValkeySlidingWindowBackend, ValkeySlidingWindowConfig,
            connection::VALKEY_TIMEOUT,
        },
        ValkeyTokenBucketBackend, ValkeyTokenBucketConfig,
    };
    use crate::token_rate_limit::{AccountingPolicy, CompiledEstimation, weights::TokenWeights};

    fn unreachable_config() -> ValkeyTokenBucketConfig {
        ValkeyTokenBucketConfig {
            valkey: ValkeyConnection::new("redis://127.0.0.1:1".into()).unwrap(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
            accounting: AccountingPolicy {
                estimation: CompiledEstimation::Fixed { estimate: 1 },
                weights: TokenWeights::UNITY,
                key_fingerprint: "test-key-policy".to_owned(),
            },
        }
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

    fn namespace(name: &str) -> String {
        format!("praxis:test:{name}:{}:{}", std::process::id(), run_id())
    }

    fn backend(
        name: &str,
        capacity: u64,
        refill_rate: f64,
        reservation_timeout_ms: u64,
    ) -> Option<ValkeyTokenBucketBackend> {
        let url = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL").ok()?;
        Some(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                valkey: ValkeyConnection::new(url).unwrap(),
                namespace: namespace(name),
                rule: "rule".to_owned(),
                capacity,
                refill_rate,
                reservation_timeout_ms,
                max_keys: 2,
                max_active_reservations: 2,
                accounting: AccountingPolicy {
                    estimation: CompiledEstimation::Fixed { estimate: 1 },
                    weights: TokenWeights::UNITY,
                    key_fingerprint: "test-key-policy".to_owned(),
                },
            })
            .unwrap(),
        )
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

    #[test]
    fn valkey_token_bucket_backend_rejects_zero_capacity_or_refill_rate() {
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                capacity: 0,
                ..unreachable_config()
            })
            .is_err(),
            "a zero capacity must be rejected"
        );
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                refill_rate: 0.0,
                ..unreachable_config()
            })
            .is_err(),
            "a zero refill_rate must be rejected"
        );
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_non_finite_refill_rate() {
        for bad_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                    refill_rate: bad_rate,
                    ..unreachable_config()
                })
                .is_err(),
                "refill_rate {bad_rate} must be rejected as non-finite/non-positive by the shared validator"
            );
        }
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_capacity_exceeding_f64_safe_integer() {
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                capacity: MAX_F64_SAFE_INTEGER + 1,
                ..unreachable_config()
            })
            .is_err(),
            "capacity above 2^53 must be rejected before precision is silently lost"
        );
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_a_refill_rate_ratio_exceeding_the_pexpire_ttl_bound() {
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                capacity: 10,
                refill_rate: 10.0 / (MAX_CAPACITY_REFILL_RATE_RATIO_SECS * 2.0),
                ..unreachable_config()
            })
            .is_err(),
            "a capacity/refill_rate ratio beyond the PEXPIRE TTL bound must be rejected"
        );
    }

    #[test]
    fn valkey_token_bucket_backend_has_no_local_state_to_reconcile_or_clean_up_synchronously() {
        let backend = ValkeyTokenBucketBackend::new(unreachable_config()).unwrap();
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

    #[tokio::test]
    async fn live_valkey_bucket_admits_refills_and_denies() {
        let Some(backend) = backend("tb-basic", 100, 10.0, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted {
            remaining, usage_after, ..
        } = backend.reserve(reserve("alice", 80, now)).await.unwrap()
        else {
            panic!("80 of 100 tokens are available");
        };
        assert_eq!(remaining, 20, "20 tokens left");
        assert_eq!(usage_after, 80, "80 consumed");
        let BackendReserve::Denied {
            retry_after_ms,
            remaining,
            ..
        } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("50 exceeds 20");
        };
        assert_eq!(remaining, 20, "the denial reports the balance it saw");
        assert_eq!(retry_after_ms, 3_000, "30 missing tokens at 10 per second");
        let BackendReserve::Admitted { remaining, .. } =
            backend.reserve(reserve("alice", 50, now + 3_000)).await.unwrap()
        else {
            panic!("three seconds later 30 tokens refilled");
        };
        assert_eq!(remaining, 0, "20 + 30 refilled - 50");
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the reconciliation test verifies exactly-once settlement and index lifetime together"
    )]
    async fn live_valkey_bucket_reconcile_refunds_once_and_repeats_are_noops() {
        let Some(backend) = backend("tb-reconcile", 100, 1.0, 5_000) else {
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
        let settle = || backend.reconcile(reconcile("alice", reservation_id, 40, now));
        let applied = BackendSettlement::Applied {
            actual: 40,
            refund: 20,
            overage: 0,
        };
        assert_eq!(settle().await.unwrap(), applied, "the unused 20 tokens are refunded");
        let mut ttl_pipe = redis::pipe();
        ttl_pipe.cmd("PTTL").arg(backend.keys_key());
        let (keys_ttl,): (i64,) = backend.valkey.pipeline(&ttl_pipe).await.unwrap();
        let expected_min = i64::try_from(backend.state_ttl_ms())
            .unwrap_or(i64::MAX)
            .saturating_sub(1_000);
        assert!(
            keys_ttl > expected_min,
            "settlement must re-arm the retained-key index with live quota state: {keys_ttl} <= {expected_min}"
        );
        assert_eq!(
            settle().await.unwrap(),
            BackendSettlement::Noop,
            "the reservation is gone"
        );
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(remaining, 59, "100 - 60 + 20 refund - 1");
    }

    #[tokio::test]
    async fn live_valkey_bucket_concurrent_duplicate_reconciles_apply_once() {
        let Some(backend) = backend("tb-dup", 100, 1.0, 5_000) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        let (first, second) = tokio::join!(
            backend.reconcile(reconcile("alice", reservation_id, 10, now)),
            backend.reconcile(reconcile("alice", reservation_id, 10, now)),
        );
        let applied = [first.unwrap(), second.unwrap()]
            .iter()
            .filter(|settlement| matches!(settlement, BackendSettlement::Applied { .. }))
            .count();
        assert_eq!(applied, 1, "exactly one of two concurrent reconciles settles");
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(remaining, 89, "100 - 10 actual - 1; the refund was applied once");
    }

    #[tokio::test]
    async fn live_valkey_bucket_reconcile_after_the_reservation_timeout_is_a_noop() {
        let Some(backend) = backend("tb-timeout", 100, 1.0, 50) else {
            return;
        };
        let now = 1_000_000;
        let BackendReserve::Admitted { reservation_id, .. } = backend.reserve(reserve("alice", 50, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            backend
                .reconcile(reconcile("alice", reservation_id, 10, now))
                .await
                .unwrap(),
            BackendSettlement::Noop,
            "an abandoned reservation expires on its own and is never refunded"
        );
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 1, now)).await.unwrap()
        else {
            panic!("admitted");
        };
        assert_eq!(remaining, 49, "100 - 50 left charged - 1");
    }

    #[tokio::test]
    async fn live_valkey_bucket_every_written_key_expires_except_the_sequence() {
        let Some(backend) = backend("tb-ttl", 100, 1.0, 5_000) else {
            return;
        };
        let BackendReserve::Admitted { reservation_id, .. } =
            backend.reserve(reserve("alice", 1, 1_000_000)).await.unwrap()
        else {
            panic!("admitted");
        };
        let bucket = backend.bucket_key(&backend.key_id("alice"));
        for key in [
            bucket.clone(),
            ValkeyTokenBucketBackend::reservation_key(&bucket, reservation_id),
            backend.active_index_key(),
            backend.keys_key(),
        ] {
            let mut pipe = redis::pipe();
            pipe.cmd("PTTL").arg(&key);
            let (ttl,): (i64,) = backend.valkey.pipeline(&pipe).await.unwrap();
            assert!(ttl > 0, "{key} must carry a TTL, got {ttl}");
        }
        let mut pipe = redis::pipe();
        pipe.cmd("PTTL").arg(backend.seq_key());
        let (ttl,): (i64,) = backend.valkey.pipeline(&pipe).await.unwrap();
        assert_eq!(ttl, -1, "the id sequence deliberately has no TTL");
    }

    #[tokio::test]
    async fn live_valkey_bucket_caps_apply_across_keys() {
        let Some(backend) = backend("tb-caps", 1_000, 1.0, 5_000) else {
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
        assert_eq!(backend.snapshot().active_keys, 2, "two retained keys");
        assert_eq!(backend.snapshot().active_reservations, 2, "two pending reservations");
        drop(backend);
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "one live test covers both rules and the cap within the second rule"
    )]
    async fn live_valkey_bucket_key_cap_is_independent_for_each_rule() {
        let (Some(mut first), Some(mut second)) = (
            backend("tb-rule-cap", 100, 1.0, 5_000),
            backend("tb-rule-cap", 100, 1.0, 5_000),
        ) else {
            return;
        };
        first.rule = "first".into();
        second.rule = "second".into();
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

    #[expect(
        clippy::too_many_lines,
        reason = "keeps the cross-algorithm namespace-isolation journey together"
    )]
    #[tokio::test]
    async fn live_valkey_bucket_and_sliding_window_on_one_namespace_do_not_share_state() {
        let Some(bucket) = backend("tb-isolation", 100, 1.0, 5_000) else {
            return;
        };
        let window = ValkeySlidingWindowBackend::new(ValkeySlidingWindowConfig {
            valkey: bucket.valkey.clone(),
            namespace: namespace("tb-isolation"),
            rule: "rule".to_owned(),
            budgets: vec![Budget {
                window_ms: 60_000,
                capacity: 100,
            }],
            reservation_timeout_ms: 5_000,
            max_keys: 2,
            max_active_reservations: 2,
            accounting: AccountingPolicy {
                estimation: CompiledEstimation::Fixed { estimate: 1 },
                weights: TokenWeights::UNITY,
                key_fingerprint: "test-key-policy".to_owned(),
            },
        });
        let now = 1_000_000;
        for (algorithm, outcome) in [
            ("token_bucket", bucket.reserve(reserve("alice", 60, now)).await.unwrap()),
            (
                "sliding_window",
                window.reserve(reserve("alice", 60, now)).await.unwrap(),
            ),
        ] {
            let BackendReserve::Admitted { remaining, .. } = outcome else {
                panic!("{algorithm} must admit against its own untouched budget");
            };
            assert_eq!(remaining, 40, "{algorithm} saw its own full budget of 100");
        }
        drop(window);
        drop(bucket);
    }

    #[tokio::test]
    async fn live_valkey_bucket_clamps_a_stored_balance_above_the_current_capacity() {
        let Some(backend) = backend("tb-shrunk", 100, 1.0, 5_000) else {
            return;
        };
        let now = 1_000_000;
        backend.ensure_accounting_config().await.unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("HSET")
            .arg(backend.bucket_key(&backend.key_id("alice")))
            .arg("tokens")
            .arg(500)
            .arg("last_refill_ms")
            .arg(now)
            .ignore();
        let () = backend.valkey.pipeline(&pipe).await.unwrap();
        let BackendReserve::Admitted { remaining, .. } = backend.reserve(reserve("alice", 10, now)).await.unwrap()
        else {
            panic!("a balance left by a larger capacity must still admit");
        };
        assert_eq!(
            remaining, 90,
            "the stored 500 is clamped to the capacity of 100 before the 10 is taken"
        );
    }

    #[tokio::test]
    async fn live_valkey_bucket_denial_of_a_new_key_writes_no_state() {
        let Some(backend) = backend("tb-denied", 100, 1.0, 5_000) else {
            return;
        };
        assert!(
            matches!(
                backend.reserve(reserve("alice", 101, 1_000_000)).await.unwrap(),
                BackendReserve::Denied { .. }
            ),
            "101 exceeds the capacity of 100"
        );
        let id = backend.key_id("alice");
        let bucket = backend.bucket_key(&id);
        let mut seq = redis::pipe();
        seq.cmd("GET").arg(backend.seq_key());
        let (drawn,): (u64,) = backend.valkey.pipeline(&seq).await.unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("EXISTS").arg(&bucket);
        pipe.cmd("EXISTS")
            .arg(ValkeyTokenBucketBackend::reservation_key(&bucket, drawn));
        pipe.cmd("ZSCORE").arg(backend.keys_key()).arg(&id);
        let (exists, reservation, score): (i64, i64, Option<f64>) = backend.valkey.pipeline(&pipe).await.unwrap();
        assert_eq!(exists, 0, "a denial must not create the bucket hash");
        assert_eq!(
            reservation, 0,
            "a denial must not create the reservation key for the id it drew"
        );
        assert!(score.is_none(), "a denial must not retain the key in the keys zset");
    }

    type ReserveOutcome = (u64, Result<BackendReserve, BackendError>);

    fn roomy_backend(url: String, name: &str, capacity: u64) -> Arc<ValkeyTokenBucketBackend> {
        Arc::new(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                valkey: ValkeyConnection::new(url).unwrap(),
                namespace: namespace(name),
                rule: "rule".to_owned(),
                capacity,
                refill_rate: 1.0,
                reservation_timeout_ms: 60_000,
                max_keys: 1_000,
                max_active_reservations: 1_000,
                accounting: AccountingPolicy {
                    estimation: CompiledEstimation::Fixed { estimate: 1 },
                    weights: TokenWeights::UNITY,
                    key_fingerprint: "test-key-policy".to_owned(),
                },
            })
            .unwrap(),
        )
    }

    async fn reserve_concurrently(
        backend: &Arc<ValkeyTokenBucketBackend>,
        clients: u64,
        per_client: usize,
        now_ms: u64,
    ) -> Vec<ReserveOutcome> {
        let tasks: Vec<_> = (1..=clients)
            .map(|estimate| {
                let backend = Arc::clone(backend);
                tokio::spawn(async move {
                    let mut outcomes = Vec::with_capacity(per_client);
                    for _ in 0..per_client {
                        outcomes.push((estimate, backend.reserve(reserve("shared", estimate, now_ms)).await));
                    }
                    outcomes
                })
            })
            .collect();
        let mut outcomes = Vec::new();
        for task in tasks {
            outcomes.extend(task.await.expect("reserve task must not panic"));
        }
        outcomes
    }

    async fn balance(backend: &ValkeyTokenBucketBackend, key: &str) -> f64 {
        let mut pipe = redis::pipe();
        pipe.cmd("HGET")
            .arg(backend.bucket_key(&backend.key_id(key)))
            .arg("tokens");
        let (balance,): (f64,) = backend.valkey.pipeline(&pipe).await.unwrap();
        balance
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn live_valkey_bucket_concurrent_reserves_on_one_key_never_fail_closed() {
        let Ok(url) = std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL") else {
            return;
        };
        let capacity = 100_000;
        let backend = roomy_backend(url, "tb-concurrent", capacity);
        let started = std::time::Instant::now();
        let outcomes = reserve_concurrently(&backend, 16, 10, 1_000_000).await;
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "160 reserves must finish in bounded time, took {:?}",
            started.elapsed()
        );
        let failures = outcomes.iter().filter(|(_, outcome)| outcome.is_err()).count();
        assert_eq!(
            failures, 0,
            "no reserve may fail closed under contention with ample capacity, {failures} of 160 did"
        );
        let admitted: u64 = outcomes
            .iter()
            .filter(|(_, outcome)| matches!(outcome, Ok(BackendReserve::Admitted { .. })))
            .map(|(estimate, _)| estimate)
            .sum();
        let balance = balance(&backend, "shared").await;
        #[expect(clippy::cast_precision_loss, reason = "test values are far below 2^53")]
        let expected = (capacity - admitted) as f64;
        assert!(
            (balance - expected).abs() < f64::EPSILON,
            "the balance must equal capacity minus every admitted estimate (no lost update): {balance} != {expected}"
        );
        drop(backend);
    }

    #[tokio::test]
    async fn token_bucket_fails_closed_when_aborts_outlast_the_deadline() {
        let Some(backend) = backend("tb-contended", 100, 1.0, 5_000) else {
            return;
        };
        let bucket = backend.bucket_key(&backend.key_id("alice"));
        let interfering = backend.valkey.clone();
        let interfere = |attempt: u32| {
            let bucket = bucket.clone();
            let interfering = interfering.clone();
            async move {
                let mut pipe = redis::pipe();
                pipe.cmd("HSET").arg(&bucket).arg("poke").arg(attempt).ignore();
                pipe.cmd("PEXPIRE").arg(&bucket).arg(60_000).ignore();
                let () = interfering.pipeline(&pipe).await.unwrap();
            }
        };
        let started = std::time::Instant::now();
        let outcome = backend.reserve_with(reserve("alice", 1, 1_000_000), interfere).await;
        let elapsed = started.elapsed();
        assert!(
            matches!(&outcome, Err(BackendError::Unavailable(message)) if message.contains("contended")),
            "when every EXEC aborts the backend fails closed, got {outcome:?}"
        );
        assert!(
            elapsed >= VALKEY_TIMEOUT,
            "aborts are retried until the {VALKEY_TIMEOUT:?} deadline, gave up after {elapsed:?}"
        );
        assert!(
            elapsed < VALKEY_TIMEOUT * 3,
            "the retry stops near the {VALKEY_TIMEOUT:?} deadline, took {elapsed:?}"
        );
    }
}
