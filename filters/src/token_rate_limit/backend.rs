// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pluggable token-rate-limit state backends.
//!
//! Adapted, unmodified in logic, from the `token_rate_limit::backend` module
//! on nerdalert's `poc/distributed-token-rate-limit-demo` spike branch
//! (<https://github.com/nerdalert/ai/tree/poc/distributed-token-rate-limit-demo>).
//! `reserve`/`reconcile` are key-agnostic (`ReserveRequest`/`ReconcileRequest`
//! carry a plain `String` key). The filter resolves M5 dimensions into an
//! opaque key before calling these backends.

use std::{
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use metrics::counter;
use tokio::sync::mpsc;

use super::{
    ledger::{Decision, DenialReason, Ledger, Settlement},
    token_bucket_ledger::{self, TokenBucketLedger},
};

/// Request to admit an estimated token cost against a key's budget.
#[derive(Debug, Clone)]
pub(super) struct ReserveRequest {
    /// Opaque budget key resolved from the filter's key spec.
    pub(super) key: String,
    /// Estimated token cost to reserve if admitted.
    pub(super) estimate: u64,
    /// Caller's current time, in milliseconds.
    pub(super) now_ms: u64,
}

/// Request to settle a prior reservation against actual usage.
#[derive(Debug, Clone)]
pub(super) struct ReconcileRequest {
    /// Same key the original [`ReserveRequest`] used.
    pub(super) key: String,
    /// Reservation ID returned by [`BackendReserve::Admitted`].
    pub(super) reservation_id: u64,
    /// Actual token usage, if known; `None` charges at the reservation's
    /// own estimate.
    pub(super) actual: Option<u64>,
    /// Caller's current time, in milliseconds.
    pub(super) now_ms: u64,
}

/// Result of a [`TokenRateLimitStateBackend::reserve`] call.
#[derive(Debug, Clone)]
pub(super) enum BackendReserve {
    /// Request may proceed with this reservation.
    Admitted {
        /// Opaque ID used for later reconciliation.
        reservation_id: u64,
        /// Estimate actually reserved.
        estimate: u64,
        /// Total committed usage in the current window (or tokens
        /// consumed from the bucket) *after* this reservation was
        /// placed. Used by the filter to evaluate graduated soft-limit
        /// tiers (proposal S1) — tiers whose capacity threshold is at
        /// or below this value fire their `inject` action.
        usage_after: u64,
        /// Remaining budget for the decided key after this reservation.
        remaining: u64,
    },
    /// Request must be rejected before routing.
    Denied {
        /// Conservative delay before another admission attempt.
        retry_after_ms: u64,
        /// Distinguishes budget exhaustion from the `max_keys` cap.
        reason: DenialReason,
        /// Remaining budget for the decided key at the time of denial.
        remaining: u64,
    },
}

impl BackendReserve {
    /// Remaining budget for the decided key, whichever way it went.
    pub(super) fn remaining(&self) -> u64 {
        match *self {
            Self::Admitted { remaining, .. } | Self::Denied { remaining, .. } => remaining,
        }
    }
}

/// Result of a [`TokenRateLimitStateBackend::reconcile`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackendSettlement {
    /// Actual usage was applied exactly once.
    Applied {
        /// Actual tokens charged.
        actual: u64,
        /// Estimate returned to the budget.
        refund: u64,
        /// Usage above the estimate.
        overage: u64,
    },
    /// The reservation was already reconciled or conservatively expired.
    Noop,
}

/// Latest bounded state published by one rule backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct BackendSnapshot {
    /// Remaining budget for the key of the most recent admission decision
    /// on this replica, on every backend.
    pub(super) budget_remaining: u64,
    /// Reservations still awaiting reconciliation: this rule's in memory;
    /// on Valkey, every rule's of the same algorithm in the namespace.
    pub(super) active_reservations: usize,
    /// Distinct budget keys currently retained: this rule's in memory; on
    /// Valkey, every rule's of the same algorithm in the namespace.
    pub(super) active_keys: usize,
}

/// Failure modes shared by every [`TokenRateLimitStateBackend`] impl.
#[derive(Debug, thiserror::Error)]
pub(super) enum BackendError {
    /// The backend could not be reached or timed out.
    #[error("shared quota backend unavailable: {0}")]
    Unavailable(String),
    /// The backend responded, but not in the expected shape.
    #[error("shared quota backend returned an invalid response")]
    InvalidResponse,
}

/// Where sliding-window admission state lives: in-process or shared.
#[async_trait]
pub(super) trait TokenRateLimitStateBackend: Send + Sync {
    /// Attempt to admit `request.estimate` against `request.key`'s budget.
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError>;

    /// Settle a prior reservation against actual usage, awaiting
    /// completion. Backends that reconcile out-of-band (e.g. Valkey via
    /// [`Self::enqueue_reconcile`]) still implement this for their own
    /// background worker to call.
    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError>;

    /// Settle a prior reservation without blocking the caller.
    ///
    /// For in-process state this may just reconcile synchronously (cheap,
    /// no I/O); for a networked backend this enqueues the work onto a
    /// background worker instead, so the response is never held up on a
    /// reconciliation round-trip.
    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError>;

    /// The smallest configured budget capacity, for rate-limit headers.
    fn limit(&self) -> u64;

    /// Latest rule-level state available without backend I/O.
    fn snapshot(&self) -> BackendSnapshot;

    /// Stable backend label used by bounded telemetry.
    fn backend_name(&self) -> &'static str;

    /// Stable algorithm label used by bounded telemetry.
    fn algorithm_name(&self) -> &'static str;

    /// Configured rule name. Required by background reconciliation telemetry.
    fn rule_name(&self) -> &str {
        ""
    }

    /// Attempt an in-process, synchronous settlement (no I/O, no async
    /// dispatch) for a prior reservation.
    ///
    /// Returns `None` for backends whose state isn't local (e.g. a
    /// networked Valkey backend) -- callers should fall back to
    /// [`Self::enqueue_reconcile`] in that case. Every in-process backend
    /// (regardless of algorithm) implements this itself rather than
    /// exposing its concrete state type, so the filter never needs to
    /// know which algorithm produced it.
    fn reconcile_sync(&self, _request: &ReconcileRequest) -> Option<BackendSettlement> {
        None
    }

    /// Reclaim idle/orphaned in-process state and report current gauges.
    ///
    /// Returns `None` for backends with no local state to reap (e.g.
    /// Valkey, where TTLs and deadline-scored sets expire state) --
    /// callers should skip gauge reporting entirely in that case rather
    /// than reporting misleading zeros.
    fn cleanup(&self, _now_ms: u64, _max_keys_to_scan: usize) -> Option<CleanupReport> {
        None
    }
}

/// In-process state snapshot after a [`TokenRateLimitStateBackend::cleanup`]
/// pass, backend-agnostic so the filter can report gauges without knowing
/// which algorithm produced them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CleanupReport {
    /// Reservations reaped this pass because they exceeded
    /// `reservation_timeout` without being reconciled.
    pub(super) orphaned: usize,
    /// Reservations still awaiting reconciliation.
    pub(super) active_reservations: usize,
    /// Distinct budget keys currently retained.
    pub(super) active_keys: usize,
}

/// In-process sliding-window state: one gateway instance, one budget.
pub(super) struct InMemoryTokenRateLimitBackend {
    /// The underlying exact sliding-window ledger.
    ledger: Arc<Ledger>,
    /// Remaining budget for the key of the most recent admission decision.
    last_remaining: AtomicU64,
}

impl InMemoryTokenRateLimitBackend {
    /// Wrap an already-constructed [`Ledger`] as a backend.
    pub(super) fn new(ledger: Ledger) -> Self {
        Self {
            ledger: Arc::new(ledger),
            last_remaining: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for InMemoryTokenRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        let reply = match self.ledger.reserve(&request.key, request.estimate, request.now_ms) {
            Decision::Admitted(reservation) => BackendReserve::Admitted {
                reservation_id: reservation.id,
                estimate: reservation.estimate,
                usage_after: reservation.usage_after,
                remaining: reservation.remaining,
            },
            Decision::Denied {
                retry_after_ms,
                reason,
                remaining,
            } => BackendReserve::Denied {
                retry_after_ms,
                reason,
                remaining,
            },
        };
        self.last_remaining.store(reply.remaining(), Ordering::Relaxed);
        Ok(reply)
    }

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        Ok(
            match self
                .ledger
                .reconcile(request.reservation_id, request.actual, request.now_ms)
            {
                Settlement::Applied {
                    actual,
                    refund,
                    overage,
                } => BackendSettlement::Applied {
                    actual,
                    refund,
                    overage,
                },
                Settlement::Noop => BackendSettlement::Noop,
            },
        )
    }

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        let _ = self
            .ledger
            .reconcile(request.reservation_id, request.actual, request.now_ms);
        Ok(())
    }

    fn limit(&self) -> u64 {
        self.ledger.limit()
    }

    fn snapshot(&self) -> BackendSnapshot {
        BackendSnapshot {
            budget_remaining: self
                .last_remaining
                .load(Ordering::Relaxed)
                .min(super::MAX_REPORTED_REMAINING),
            active_reservations: self.ledger.active_count(),
            active_keys: self.ledger.key_count(),
        }
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }

    fn algorithm_name(&self) -> &'static str {
        "sliding_window"
    }

    fn reconcile_sync(&self, request: &ReconcileRequest) -> Option<BackendSettlement> {
        Some(
            match self
                .ledger
                .reconcile(request.reservation_id, request.actual, request.now_ms)
            {
                Settlement::Applied {
                    actual,
                    refund,
                    overage,
                } => BackendSettlement::Applied {
                    actual,
                    refund,
                    overage,
                },
                Settlement::Noop => BackendSettlement::Noop,
            },
        )
    }

    fn cleanup(&self, now_ms: u64, max_keys_to_scan: usize) -> Option<CleanupReport> {
        Some(CleanupReport {
            orphaned: self.ledger.cleanup(now_ms, max_keys_to_scan),
            active_reservations: self.ledger.active_count(),
            active_keys: self.ledger.key_count(),
        })
    }
}

/// In-process token-bucket state: one gateway instance, one budget,
/// continuously refilled rather than admitted against a trailing window.
pub(super) struct InMemoryTokenBucketBackend {
    /// The underlying exact token-bucket ledger.
    ledger: Arc<TokenBucketLedger>,
    /// Remaining budget for the key of the most recent admission decision.
    last_remaining: AtomicU64,
}

impl InMemoryTokenBucketBackend {
    /// Wrap an already-constructed [`TokenBucketLedger`] as a backend.
    pub(super) fn new(ledger: TokenBucketLedger) -> Self {
        Self {
            ledger: Arc::new(ledger),
            last_remaining: AtomicU64::new(0),
        }
    }

    /// Shared reconcile path for `reconcile`/`enqueue_reconcile`/`reconcile_sync`.
    fn reconcile_ledger(&self, request: &ReconcileRequest) -> BackendSettlement {
        match self
            .ledger
            .reconcile(request.reservation_id, request.actual, request.now_ms)
        {
            token_bucket_ledger::Settlement::Applied {
                actual,
                refund,
                overage,
            } => BackendSettlement::Applied {
                actual,
                refund,
                overage,
            },
            token_bucket_ledger::Settlement::Noop => BackendSettlement::Noop,
        }
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for InMemoryTokenBucketBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        let reply = match self.ledger.reserve(&request.key, request.estimate, request.now_ms) {
            token_bucket_ledger::Decision::Admitted(reservation) => BackendReserve::Admitted {
                reservation_id: reservation.id,
                estimate: reservation.estimate,
                usage_after: reservation.usage_after,
                remaining: reservation.remaining,
            },
            token_bucket_ledger::Decision::Denied {
                retry_after_ms,
                reason,
                remaining,
            } => BackendReserve::Denied {
                retry_after_ms,
                reason,
                remaining,
            },
        };
        self.last_remaining.store(reply.remaining(), Ordering::Relaxed);
        Ok(reply)
    }

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        Ok(self.reconcile_ledger(&request))
    }

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        let _ = self.reconcile_ledger(&request);
        Ok(())
    }

    fn limit(&self) -> u64 {
        self.ledger.limit()
    }

    fn snapshot(&self) -> BackendSnapshot {
        BackendSnapshot {
            budget_remaining: self
                .last_remaining
                .load(Ordering::Relaxed)
                .min(super::MAX_REPORTED_REMAINING),
            active_reservations: self.ledger.active_count(),
            active_keys: self.ledger.key_count(),
        }
    }

    fn backend_name(&self) -> &'static str {
        "memory"
    }

    fn algorithm_name(&self) -> &'static str {
        "token_bucket"
    }

    fn reconcile_sync(&self, request: &ReconcileRequest) -> Option<BackendSettlement> {
        Some(self.reconcile_ledger(request))
    }

    fn cleanup(&self, now_ms: u64, max_keys_to_scan: usize) -> Option<CleanupReport> {
        Some(CleanupReport {
            orphaned: self.ledger.cleanup(now_ms, max_keys_to_scan),
            active_reservations: self.ledger.active_count(),
            active_keys: self.ledger.key_count(),
        })
    }
}

/// Drain `receiver`, reconciling each request against `worker`'s backend
/// with bounded retries, off the request/response path entirely.
///
/// Generic over any [`TokenRateLimitStateBackend`] (sliding-window,
/// token-bucket, or any future Valkey-backed algorithm) -- the retry/
/// audit behavior is identical regardless of which algorithm
/// `worker.reconcile` ultimately runs.
///
/// A dropped/failed reconciliation after retries is intentionally *not*
/// escalated back to the request that triggered it (that response has
/// already been sent) -- it's counted and logged so operators can audit
/// it, and the reservation still expires and gets conservatively charged
/// via `reservation_timeout` regardless.
pub(super) async fn run_reconcile_worker<B>(worker: B, mut receiver: mpsc::Receiver<ReconcileRequest>)
where
    B: TokenRateLimitStateBackend + 'static,
{
    while let Some(request) = receiver.recv().await {
        let mut attempts = 0;
        loop {
            match worker.reconcile(request.clone()).await {
                Ok(settlement) => {
                    record_completed_reconciliation(&worker, &settlement);
                    break;
                },
                Err(error) if attempts < 2 => {
                    attempts += 1;
                    tracing::warn!(attempts, %error, "token-rate-limit reconciliation retry");
                    tokio::time::sleep(Duration::from_millis(25 * attempts)).await;
                },
                Err(error) => {
                    record_abandoned_reconciliation(&worker, &error);
                    break;
                },
            }
        }
    }
}

/// Publish one completed Valkey reconciliation through the same metrics and
/// accounting helpers as the synchronous in-memory path.
pub(super) fn record_completed_reconciliation(
    backend: &impl TokenRateLimitStateBackend,
    settlement: &BackendSettlement,
) {
    counter!(
        "praxis_trl_backend_reconciliation_total",
        "backend" => backend.backend_name(),
        "result" => "completed",
        "rule" => backend.rule_name().to_owned(),
    )
    .increment(1);
    super::record_settlement_metrics(backend.rule_name(), settlement);
    super::record_accounting_settlement(backend.rule_name(), backend, settlement);
    super::record_state_metrics(backend.rule_name(), backend);
}

/// Count and log one reconciliation given up after its retries; the
/// reservation still expires and is charged at its estimate.
pub(super) fn record_abandoned_reconciliation(backend: &impl TokenRateLimitStateBackend, error: &BackendError) {
    super::record_backend_error_metric(backend.rule_name(), backend.backend_name());
    tracing::warn!(
        target: "praxis_ai::token_rate_limit::accounting",
        phase = "reconciliation",
        rule = backend.rule_name(),
        algorithm = backend.algorithm_name(),
        backend = backend.backend_name(),
        result = "failed",
        error = %error,
        "token rate limit accounting"
    );
    tracing::error!(%error, "token-rate-limit reconciliation abandoned after retries");
}

/// Shared background-reconciliation scaffolding for every Valkey-backed
/// algorithm: a queue plus a spawn-at-most-once guard for
/// [`run_reconcile_worker`].
pub(super) struct ReconcileWorker {
    /// Sending half of the reconciliation queue; cloned into the worker.
    tx: mpsc::Sender<ReconcileRequest>,
    /// Receiving half, taken exactly once by [`Self::start`].
    rx: Mutex<Option<mpsc::Receiver<ReconcileRequest>>>,
    /// Ensures the background worker is spawned at most once.
    started: OnceLock<()>,
}

impl ReconcileWorker {
    /// A live worker: holds a real receiver, ready for [`Self::start`].
    pub(super) fn new() -> Self {
        let (tx, rx) = mpsc::channel(1024);
        Self {
            tx,
            rx: Mutex::new(Some(rx)),
            started: OnceLock::new(),
        }
    }

    /// A throwaway (never-sent-to, never-started) worker -- used only
    /// when cloning a backend to hand the *real* background worker its
    /// own handle to `reserve`/`reconcile`, without that clone holding
    /// the real sender (which would keep the channel open forever) or
    /// being able to spawn a second worker.
    pub(super) fn detached() -> Self {
        let (tx, _rx) = mpsc::channel(1);
        Self {
            tx,
            rx: Mutex::new(None),
            started: OnceLock::new(),
        }
    }

    /// Enqueue a reconciliation request for the background worker.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] if the queue is full or the
    /// worker has stopped.
    pub(super) fn enqueue(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        self.tx
            .try_send(request)
            .map_err(|error| BackendError::Unavailable(format!("reconciliation queue is full or stopped: {error}")))
    }

    /// Lazily spawn [`run_reconcile_worker`] on `runtime`, at most once.
    /// `make_worker` builds the backend clone the worker itself will
    /// call `reconcile` against (see [`Self::detached`]).
    pub(super) fn start<B>(&self, runtime: &tokio::runtime::Handle, make_worker: impl FnOnce() -> B)
    where
        B: TokenRateLimitStateBackend + 'static,
    {
        self.started.get_or_init(|| {
            let Some(receiver) = self.rx.lock().ok().and_then(|mut guard| guard.take()) else {
                return;
            };
            runtime.spawn(run_reconcile_worker(make_worker(), receiver));
        });
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::token_rate_limit::ledger::{Budget, LedgerConfig};

    fn memory_backend(capacity: u64) -> InMemoryTokenRateLimitBackend {
        let ledger = Ledger::new(LedgerConfig {
            budgets: vec![Budget {
                window_ms: 60_000,
                capacity,
            }],
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_key_length: 64,
            max_active_reservations: 8,
        })
        .unwrap();
        InMemoryTokenRateLimitBackend::new(ledger)
    }

    #[tokio::test]
    async fn reconcile_sync_settles_in_process_state_without_a_network_round_trip() {
        let backend = memory_backend(100);
        let admitted = backend
            .reserve(ReserveRequest {
                key: "a".into(),
                estimate: 40,
                now_ms: 0,
            })
            .await
            .unwrap();
        let BackendReserve::Admitted { reservation_id, .. } = admitted else {
            panic!("expected admission")
        };

        let settlement = backend.reconcile_sync(&ReconcileRequest {
            key: "a".into(),
            reservation_id,
            actual: Some(10),
            now_ms: 0,
        });
        assert_eq!(
            settlement,
            Some(BackendSettlement::Applied {
                actual: 10,
                refund: 30,
                overage: 0
            }),
            "in-process backend must resolve reconcile_sync synchronously, without a Valkey-style enqueued worker"
        );
    }

    #[tokio::test]
    async fn cleanup_reports_active_reservations_and_keys_for_in_process_state() {
        let backend = memory_backend(100);
        backend
            .reserve(ReserveRequest {
                key: "a".into(),
                estimate: 10,
                now_ms: 0,
            })
            .await
            .unwrap();

        let report = backend
            .cleanup(0, 8)
            .expect("in-process backend must report cleanup state for gauges");
        assert_eq!(
            report.active_reservations, 1,
            "one un-reconciled reservation should be counted"
        );
        assert_eq!(report.active_keys, 1, "one distinct key should be tracked");
        assert_eq!(report.orphaned, 0, "nothing has timed out yet");
    }

    /// `reconcile`/`enqueue_reconcile` are part of the shared
    /// [`TokenRateLimitStateBackend`] trait contract -- callers reach an
    /// in-process backend exclusively through `reconcile_sync` today (see
    /// `TokenRateLimitFilter::reconcile`'s doc comment), but the trait
    /// methods themselves must still behave correctly for any future or
    /// generic (`Arc<dyn TokenRateLimitStateBackend>`) caller that goes
    /// through them instead.
    /// Reserve `estimate` against `backend`, returning the resulting
    /// reservation ID (panics if denied -- every caller below reserves
    /// well within its backend's configured capacity).
    async fn reserve_or_panic(backend: &impl TokenRateLimitStateBackend, estimate: u64) -> u64 {
        let admitted = backend
            .reserve(ReserveRequest {
                key: "a".into(),
                estimate,
                now_ms: 0,
            })
            .await
            .unwrap();
        let BackendReserve::Admitted { reservation_id, .. } = admitted else {
            panic!("expected admission")
        };
        reservation_id
    }

    /// The `reconcile` half of
    /// `assert_trait_reconcile_methods_apply_directly`, split out to keep
    /// both under clippy's function-length budget.
    async fn assert_trait_reconcile_applies_directly(backend: &impl TokenRateLimitStateBackend) {
        let reservation_id = reserve_or_panic(backend, 50).await;
        let settlement = backend
            .reconcile(ReconcileRequest {
                key: "a".into(),
                reservation_id,
                actual: Some(10),
                now_ms: 0,
            })
            .await
            .unwrap();
        assert_eq!(
            settlement,
            BackendSettlement::Applied {
                actual: 10,
                refund: 40,
                overage: 0
            }
        );
    }

    /// The `enqueue_reconcile` half -- see
    /// `assert_trait_reconcile_applies_directly`. The in-process
    /// implementation applies it inline rather than truly deferring it,
    /// but must still succeed and take effect.
    async fn assert_trait_enqueue_reconcile_applies_directly(backend: &impl TokenRateLimitStateBackend) {
        let reservation_id = reserve_or_panic(backend, 40).await;
        backend
            .enqueue_reconcile(ReconcileRequest {
                key: "a".into(),
                reservation_id,
                actual: Some(5),
                now_ms: 0,
            })
            .unwrap();
        let request = ReserveRequest {
            key: "a".into(),
            estimate: 35,
            now_ms: 0,
        };
        assert!(
            matches!(backend.reserve(request).await.unwrap(), BackendReserve::Admitted { .. }),
            "enqueue_reconcile must have released the 35 unused tokens from the second reservation"
        );
    }

    #[tokio::test]
    async fn in_memory_sliding_window_backend_trait_reconcile_methods_apply_directly() {
        let backend = memory_backend(100);
        assert_trait_reconcile_applies_directly(&backend).await;
        assert_trait_enqueue_reconcile_applies_directly(&backend).await;
    }

    /// The token-bucket analog of
    /// `in_memory_sliding_window_backend_trait_reconcile_methods_apply_directly`.
    #[tokio::test]
    async fn in_memory_token_bucket_backend_trait_reconcile_methods_apply_directly() {
        let backend = bucket_backend(100, 1.0);
        assert_trait_reconcile_applies_directly(&backend).await;
        assert_trait_enqueue_reconcile_applies_directly(&backend).await;
    }

    /// Reconciling an unknown/already-settled reservation ID must be a
    /// silent no-op (idempotent double-reconciliation), never a panic or
    /// a double-credit -- for both algorithms' in-process backends,
    /// through both the async `reconcile` trait method and the
    /// synchronous `reconcile_sync` fast path.
    #[tokio::test]
    async fn in_memory_backends_reconcile_is_noop_for_an_unknown_reservation_id() {
        let unknown_reservation = ReconcileRequest {
            key: "a".into(),
            reservation_id: 999_999,
            actual: Some(1),
            now_ms: 0,
        };

        let sliding = memory_backend(100);
        assert_eq!(
            sliding.reconcile(unknown_reservation.clone()).await.unwrap(),
            BackendSettlement::Noop
        );
        assert_eq!(
            sliding.reconcile_sync(&unknown_reservation),
            Some(BackendSettlement::Noop)
        );

        let bucket = bucket_backend(100, 1.0);
        assert_eq!(
            bucket.reconcile(unknown_reservation.clone()).await.unwrap(),
            BackendSettlement::Noop
        );
        assert_eq!(
            bucket.reconcile_sync(&unknown_reservation),
            Some(BackendSettlement::Noop)
        );
    }

    /// [`ReconcileWorker::start`] on a [`ReconcileWorker::detached`]
    /// worker must be a no-op: there's no receiver to hand a spawned
    /// [`run_reconcile_worker`], so it must return without ever calling
    /// `make_worker` (a real caller passes a closure that builds a live
    /// backend clone there -- doing that unnecessarily would be wasted
    /// work at best and a logic error at worst).
    #[test]
    fn reconcile_worker_start_on_a_detached_worker_never_spawns() {
        let worker = ReconcileWorker::detached();
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        worker.start(runtime.handle(), || -> AlwaysFailsReconcile {
            panic!("a detached ReconcileWorker has no receiver to hand a spawned worker -- make_worker must not run")
        });
    }

    // -------------------------------------------------------------------------
    // InMemoryTokenBucketBackend (trait-contract level -- exhaustive
    // business-scenario coverage for refill/refund/overage/DoS bounds
    // lives in `token_bucket_ledger::tests`; these confirm the backend
    // wrapper faithfully exposes that ledger through the shared trait).
    // -------------------------------------------------------------------------

    fn bucket_backend(capacity: u64, refill_rate: f64) -> InMemoryTokenBucketBackend {
        let ledger = TokenBucketLedger::new(token_bucket_ledger::TokenBucketConfig {
            capacity,
            refill_rate,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_key_length: 64,
            max_active_reservations: 8,
        })
        .unwrap();
        InMemoryTokenBucketBackend::new(ledger)
    }

    #[tokio::test]
    async fn token_bucket_backend_admits_within_capacity_and_denies_over_it() {
        let backend = bucket_backend(10, 1.0);
        assert!(matches!(
            backend
                .reserve(ReserveRequest {
                    key: "a".into(),
                    estimate: 10,
                    now_ms: 0
                })
                .await
                .unwrap(),
            BackendReserve::Admitted { .. }
        ));
        assert!(matches!(
            backend
                .reserve(ReserveRequest {
                    key: "a".into(),
                    estimate: 1,
                    now_ms: 0
                })
                .await
                .unwrap(),
            BackendReserve::Denied { .. }
        ));
    }

    #[tokio::test]
    async fn token_bucket_backend_limit_reports_configured_capacity() {
        let backend = bucket_backend(250, 5.0);
        assert_eq!(backend.limit(), 250);
    }

    #[tokio::test]
    async fn token_bucket_backend_reconcile_sync_credits_back_unused_estimate() {
        let backend = bucket_backend(100, 1.0);
        let admitted = backend
            .reserve(ReserveRequest {
                key: "a".into(),
                estimate: 50,
                now_ms: 0,
            })
            .await
            .unwrap();
        let BackendReserve::Admitted { reservation_id, .. } = admitted else {
            panic!("expected admission")
        };
        let settlement = backend.reconcile_sync(&ReconcileRequest {
            key: "a".into(),
            reservation_id,
            actual: Some(10),
            now_ms: 0,
        });
        assert_eq!(
            settlement,
            Some(BackendSettlement::Applied {
                actual: 10,
                refund: 40,
                overage: 0
            })
        );
    }

    #[tokio::test]
    async fn token_bucket_backend_cleanup_reports_active_reservations_and_keys() {
        let backend = bucket_backend(100, 1.0);
        backend
            .reserve(ReserveRequest {
                key: "a".into(),
                estimate: 10,
                now_ms: 0,
            })
            .await
            .unwrap();
        let report = backend
            .cleanup(0, 8)
            .expect("in-process token bucket backend must report cleanup state for gauges");
        assert_eq!(report.active_reservations, 1);
        assert_eq!(report.active_keys, 1);
    }

    #[test]
    fn worker_enqueue_fails_once_its_receiver_is_gone() {
        let worker = ReconcileWorker::detached();
        let request = ReconcileRequest {
            key: "a".into(),
            reservation_id: 1,
            actual: Some(1),
            now_ms: 0,
        };
        assert!(worker.enqueue(request).is_err());
    }

    /// A backend whose `reconcile` always fails, to drive
    /// [`run_reconcile_worker`]'s bounded-retry-then-abandon path.
    struct AlwaysFailsReconcile {
        attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TokenRateLimitStateBackend for AlwaysFailsReconcile {
        async fn reserve(&self, _request: ReserveRequest) -> Result<BackendReserve, BackendError> {
            panic!("not exercised by this test")
        }

        async fn reconcile(&self, _request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(BackendError::Unavailable("simulated failure".into()))
        }

        fn enqueue_reconcile(&self, _request: ReconcileRequest) -> Result<(), BackendError> {
            panic!("not exercised by this test")
        }

        fn limit(&self) -> u64 {
            0
        }

        fn snapshot(&self) -> BackendSnapshot {
            BackendSnapshot::default()
        }

        fn backend_name(&self) -> &'static str {
            "test"
        }

        fn algorithm_name(&self) -> &'static str {
            "test"
        }
    }

    #[tokio::test]
    async fn reconcile_worker_retries_then_abandons_a_persistently_failing_reconcile() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(run_reconcile_worker(
            AlwaysFailsReconcile {
                attempts: Arc::clone(&attempts),
            },
            rx,
        ));
        tx.send(ReconcileRequest {
            key: "a".into(),
            reservation_id: 1,
            actual: Some(1),
            now_ms: 0,
        })
        .await
        .unwrap();
        drop(tx);

        // 1 initial attempt + 2 retries (25ms, 50ms backoff) before abandoning.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            attempts.load(Ordering::SeqCst),
            3,
            "must retry exactly twice, then abandon"
        );
    }
}
