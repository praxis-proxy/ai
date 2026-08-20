// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Pluggable token-rate-limit state backends.
//!
//! Adapted, unmodified in logic, from the `token_rate_limit::backend` module
//! on nerdalert's `poc/distributed-token-rate-limit-demo` spike branch
//! (<https://github.com/nerdalert/ai/tree/poc/distributed-token-rate-limit-demo>).
//! `reserve`/`reconcile` are key-agnostic (`ReserveRequest`/`ReconcileRequest`
//! carry a plain `String` key); this filter supplies the key resolved from
//! its own `bucket_key_header` config instead of the source branch's
//! principal+model composite key, so no logic here was changed to adopt it.

use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use async_trait::async_trait;
use metrics::counter;
use redis::aio::MultiplexedConnection;
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc;

use super::{
    ledger::{Budget, Decision, Ledger, Settlement},
    token_bucket_ledger::{self, TokenBucketLedger},
};

/// Bound on every Valkey network operation (connect or `EVAL`), so an
/// unreachable-but-not-yet-timed-out-at-the-OS-level backend still fails
/// closed quickly rather than hanging the request indefinitely.
const VALKEY_TIMEOUT: Duration = Duration::from_millis(500);

/// Request to admit an estimated token cost against a key's budget.
#[derive(Debug, Clone)]
pub(super) struct ReserveRequest {
    /// Opaque budget key (e.g. a resolved `bucket_key_header` value).
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
    /// Actual token usage, if known; `None` charges at `estimate`.
    pub(super) actual: Option<u64>,
    /// The original reservation's estimate (for backends, like Valkey,
    /// that reconcile out-of-band and need it for a default charge).
    pub(super) estimate: u64,
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
    },
    /// Request must be rejected before routing.
    Denied {
        /// Conservative delay before another admission attempt.
        retry_after_ms: u64,
    },
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
    /// Valkey, where expiry is handled by the Lua reserve script
    /// itself) -- callers should skip gauge reporting entirely in that
    /// case rather than reporting misleading zeros.
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
}

impl InMemoryTokenRateLimitBackend {
    /// Wrap an already-constructed [`Ledger`] as a backend.
    pub(super) fn new(ledger: Ledger) -> Self {
        Self {
            ledger: Arc::new(ledger),
        }
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for InMemoryTokenRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        Ok(
            match self.ledger.reserve(&request.key, request.estimate, request.now_ms) {
                Decision::Admitted(reservation) => BackendReserve::Admitted {
                    reservation_id: reservation.id,
                    estimate: reservation.estimate,
                },
                Decision::Denied { retry_after_ms, .. } => BackendReserve::Denied { retry_after_ms },
            },
        )
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
}

impl InMemoryTokenBucketBackend {
    /// Wrap an already-constructed [`TokenBucketLedger`] as a backend.
    pub(super) fn new(ledger: TokenBucketLedger) -> Self {
        Self {
            ledger: Arc::new(ledger),
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
        Ok(
            match self.ledger.reserve(&request.key, request.estimate, request.now_ms) {
                token_bucket_ledger::Decision::Admitted(reservation) => BackendReserve::Admitted {
                    reservation_id: reservation.id,
                    estimate: reservation.estimate,
                },
                token_bucket_ledger::Decision::Denied { retry_after_ms } => BackendReserve::Denied { retry_after_ms },
            },
        )
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

/// Atomically admit a reservation against every configured budget for one
/// key, or deny it -- the Valkey/Lua analog of [`Ledger::reserve`].
///
/// `KEYS`: `[1]` physical key, `[2]` settled zset, `[3]` active hash,
/// `[4]` namespace keys zset, `[5]` namespace active-count string,
/// `[6]` namespace reservation-id sequence, `[7]` namespace active-index
/// zset (global reservation-expiry tracking). `ARGV`: reservation
/// timeout (ms), max keys, max active reservations, estimate, budget
/// count, then `(window_ms, capacity)` pairs. Returns
/// `[1, id, estimate]` on admission or `[0, retry_after_ms]` on denial.
const RESERVE_SCRIPT: &str = "
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local timeout_ms = tonumber(ARGV[1])
local max_keys = tonumber(ARGV[2])
local max_active = tonumber(ARGV[3])
local estimate = tonumber(ARGV[4])
local budget_count = tonumber(ARGV[5])
local settled = KEYS[2]
local active = KEYS[3]

local active_total = tonumber(redis.call('GET', KEYS[5]) or '0')
local expired_global = redis.call('ZRANGE', KEYS[7], '-inf', now_ms, 'BYSCORE')
for i = 1, #expired_global do
  local member = expired_global[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    local value = redis.call('HGET', active_key, reservation)
    if value then
      local value_split = string.find(value, '|')
      local amount = tonumber(string.sub(value, 1, value_split - 1))
      local reserved_at = tonumber(string.sub(value, value_split + 1))
      redis.call('ZADD', physical .. ':settled', reserved_at, 'expired:' .. reservation .. ':' .. amount)
      redis.call('HDEL', active_key, reservation)
      active_total = math.max(0, active_total - 1)
    end
  end
  redis.call('ZREM', KEYS[7], member)
end
redis.call('SET', KEYS[5], active_total)

local max_window = 0
for i = 1, budget_count do
  local window = tonumber(ARGV[5 + (i * 2) - 1])
  if window > max_window then max_window = window end
  redis.call('ZREMRANGEBYSCORE', settled, '-inf', now_ms - window)
end

local expired = {}
local active_values = redis.call('HGETALL', active)
for i = 1, #active_values, 2 do
  local id = active_values[i]
  local value = active_values[i + 1]
  local sep = string.find(value, '|')
  local reserved_at = tonumber(string.sub(value, sep + 1))
  if now_ms - reserved_at >= timeout_ms then
    local amount = tonumber(string.sub(value, 1, sep - 1))
    redis.call('ZADD', settled, reserved_at, 'expired:' .. id .. ':' .. amount)
    redis.call('HDEL', active, id)
    active_total = math.max(0, active_total - 1)
  end
end
redis.call('SET', KEYS[5], active_total)

redis.call('ZREMRANGEBYSCORE', KEYS[4], '-inf', now_ms)
local key_exists = redis.call('ZSCORE', KEYS[4], KEYS[1]) ~= false
if not key_exists and redis.call('ZCARD', KEYS[4]) >= max_keys then
  return {0, max_window}
end
if active_total >= max_active then
  return {0, max_window}
end

for i = 1, budget_count do
  local window = tonumber(ARGV[5 + (i * 2) - 1])
  local capacity = tonumber(ARGV[5 + (i * 2)])
  local settled_sum = 0
  local entries = redis.call('ZRANGE', settled, now_ms - window, '+inf', 'BYSCORE', 'WITHSCORES')
  for j = 1, #entries, 2 do
    local member = entries[j]
    local amount = string.match(member, ':(%d+)$')
    if amount then settled_sum = settled_sum + tonumber(amount) end
  end
  local active_values = redis.call('HGETALL', active)
  local active_sum = 0
  for j = 1, #active_values, 2 do
    local sep = string.find(active_values[j + 1], '|')
    active_sum = active_sum + tonumber(string.sub(active_values[j + 1], 1, sep - 1))
  end
  if settled_sum + active_sum + estimate > capacity then
    return {0, max_window}
  end
end

local id = redis.call('INCR', KEYS[6])
redis.call('HSET', active, id, estimate .. '|' .. now_ms)
redis.call('INCR', KEYS[5])
redis.call('ZADD', KEYS[7], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
local ttl = math.max(max_window + timeout_ms, 1000)
redis.call('ZADD', KEYS[4], now_ms + ttl, KEYS[1])
redis.call('PEXPIRE', settled, ttl)
redis.call('PEXPIRE', active, ttl)
redis.call('PEXPIRE', KEYS[1], ttl)
return {1, id, estimate}
";

/// Atomically settle a prior reservation against actual usage -- the
/// Valkey/Lua analog of [`Ledger::reconcile`].
///
/// `KEYS`: same layout as [`RESERVE_SCRIPT`]. `ARGV`: `[1]` reservation
/// ID, `[2]` actual usage. Returns `[0]` if the reservation was already
/// reconciled/expired (no-op), or `[1, actual, refund, overage]`.
const RECONCILE_SCRIPT: &str = "
local value = redis.call('HGET', KEYS[3], ARGV[1])
if not value then return {0} end
local sep = string.find(value, '|')
local estimate = tonumber(string.sub(value, 1, sep - 1))
local actual = tonumber(ARGV[2])
redis.call('HDEL', KEYS[3], ARGV[1])
local active_total = math.max(0, tonumber(redis.call('GET', KEYS[5]) or '0') - 1)
redis.call('SET', KEYS[5], active_total)
redis.call('ZREM', KEYS[7], KEYS[1] .. '|' .. ARGV[1])
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
redis.call('ZADD', KEYS[2], now_ms, 'settled:' .. ARGV[1] .. ':' .. actual)
return {1, actual, math.max(0, estimate - actual), math.max(0, actual - estimate)}
";

/// Drain `receiver`, reconciling each request against `worker`'s backend
/// with bounded retries, off the request/response path entirely.
///
/// Generic over any [`TokenRateLimitStateBackend`] (sliding-window,
/// token-bucket, or any future Valkey-backed algorithm) -- the retry/
/// audit behavior is identical regardless of which algorithm's Lua
/// script `worker.reconcile` ultimately calls.
///
/// A dropped/failed reconciliation after retries is intentionally *not*
/// escalated back to the request that triggered it (that response has
/// already been sent) -- it's counted and logged so operators can audit
/// it, and the reservation still expires and gets conservatively charged
/// via `reservation_timeout` regardless.
async fn run_reconcile_worker<B>(worker: B, mut receiver: mpsc::Receiver<ReconcileRequest>)
where
    B: TokenRateLimitStateBackend + 'static,
{
    while let Some(request) = receiver.recv().await {
        let mut attempts = 0;
        loop {
            match worker.reconcile(request.clone()).await {
                Ok(_) => {
                    counter!("praxis_ai_token_rate_limit_backend_reconciliation_total", "backend" => "valkey", "result" => "completed")
                        .increment(1);
                    break;
                },
                Err(error) if attempts < 2 => {
                    attempts += 1;
                    tracing::warn!(attempts, %error, "token-rate-limit reconciliation retry");
                    tokio::time::sleep(Duration::from_millis(25 * attempts)).await;
                },
                Err(error) => {
                    counter!("praxis_ai_token_rate_limit_backend_errors_total", "backend" => "valkey", "operation" => "reconcile")
                        .increment(1);
                    tracing::error!(%error, "token-rate-limit reconciliation abandoned after retries");
                    break;
                },
            }
        }
    }
}

/// Shared Valkey connection handling for every Valkey-backed algorithm:
/// opening a fresh multiplexed connection and running one `EVAL`, both
/// bounded by [`VALKEY_TIMEOUT`] so an unreachable/wedged backend fails
/// closed quickly instead of hanging the request.
#[derive(Clone)]
struct ValkeyEval {
    /// Lazy Valkey/Redis client; connections are opened per call.
    client: redis::Client,
}

impl ValkeyEval {
    /// Open a (lazy, not-yet-connected) Valkey client.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] if `url` isn't a well-formed
    /// Valkey/Redis connection URL.
    fn new(url: String) -> Result<Self, BackendError> {
        let client = redis::Client::open(url).map_err(|e| BackendError::Unavailable(e.to_string()))?;
        Ok(Self { client })
    }

    /// Open a fresh multiplexed connection, bounded by [`VALKEY_TIMEOUT`]
    /// so an unreachable Valkey fails the request quickly (fail closed)
    /// rather than hanging it indefinitely.
    async fn connection(&self) -> Result<MultiplexedConnection, BackendError> {
        tokio::time::timeout(VALKEY_TIMEOUT, self.client.get_multiplexed_async_connection())
            .await
            .map_err(|_error| BackendError::Unavailable("Valkey connection timed out".into()))?
            .map_err(|e| BackendError::Unavailable(e.to_string()))
    }

    /// Run one `EVAL script KEYS... ARGV...` command against a fresh
    /// connection. Opening that connection (via [`Self::connection`]) and
    /// running the command each get their own independent
    /// [`VALKEY_TIMEOUT`] window (fail closed rather than hang if Valkey
    /// is reachable but wedged), so this call's worst-case latency is up
    /// to 2x [`VALKEY_TIMEOUT`], not [`VALKEY_TIMEOUT`] itself -- relevant
    /// since `reserve`/`reconcile` sit synchronously on the request path.
    async fn eval<const N: usize>(
        &self,
        script: &str,
        keys: &[String; N],
        args: &[String],
    ) -> Result<Vec<i64>, BackendError> {
        let mut command = redis::cmd("EVAL");
        command.arg(script).arg(keys.len());
        for key in keys {
            command.arg(key);
        }
        for arg in args {
            command.arg(arg);
        }
        let mut connection = self.connection().await?;
        tokio::time::timeout(VALKEY_TIMEOUT, command.query_async(&mut connection))
            .await
            .map_err(|_error| BackendError::Unavailable("Valkey command timed out".into()))?
            .map_err(|e| BackendError::Unavailable(e.to_string()))
    }
}

/// Shared background-reconciliation scaffolding for every Valkey-backed
/// algorithm: a queue plus a spawn-at-most-once guard for
/// [`run_reconcile_worker`].
struct ReconcileWorker {
    /// Sending half of the reconciliation queue; cloned into the worker.
    tx: mpsc::Sender<ReconcileRequest>,
    /// Receiving half, taken exactly once by [`Self::start`].
    rx: Mutex<Option<mpsc::Receiver<ReconcileRequest>>>,
    /// Ensures the background worker is spawned at most once.
    started: OnceLock<()>,
}

impl ReconcileWorker {
    /// A live worker: holds a real receiver, ready for [`Self::start`].
    fn new() -> Self {
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
    fn detached() -> Self {
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
    fn enqueue(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        self.tx
            .try_send(request)
            .map_err(|error| BackendError::Unavailable(format!("reconciliation queue is full or stopped: {error}")))
    }

    /// Lazily spawn [`run_reconcile_worker`] on `runtime`, at most once.
    /// `make_worker` builds the backend clone the worker itself will
    /// call `reconcile` against (see [`Self::detached`]).
    fn start<B>(&self, runtime: &tokio::runtime::Handle, make_worker: impl FnOnce() -> B)
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

/// Valkey/Redis-backed sliding-window state, shared across every gateway
/// instance/replica pointed at the same `url`/`namespace`.
///
/// Admission (`reserve`) is synchronous with the request (an EVAL round-
/// trip); reconciliation (`enqueue_reconcile`) is deferred to a
/// background worker so it never adds latency to the response path.
pub(super) struct ValkeyTokenRateLimitBackend {
    /// Shared connection/EVAL handling, see [`ValkeyEval`].
    valkey: ValkeyEval,
    /// Key namespace prefix, see [`ValkeyBackendConfig::namespace`].
    namespace: String,
    /// Rule identifier, see [`ValkeyBackendConfig::rule`].
    rule: String,
    /// Sliding-window budgets enforced atomically per key.
    budgets: Vec<Budget>,
    /// See [`ValkeyBackendConfig::reservation_timeout_ms`].
    reservation_timeout_ms: u64,
    /// See [`ValkeyBackendConfig::max_keys`].
    max_keys: usize,
    /// See [`ValkeyBackendConfig::max_active_reservations`].
    max_active_reservations: usize,
    /// Smallest configured budget capacity, for rate-limit headers.
    limit: u64,
    /// Shared background-reconciliation scaffolding, see [`ReconcileWorker`].
    worker: ReconcileWorker,
}

/// Construction parameters for [`ValkeyTokenRateLimitBackend`].
pub(super) struct ValkeyBackendConfig {
    /// Valkey/Redis connection URL, already `${ENV_VAR}`-expanded.
    pub(super) url: String,
    /// Key namespace prefix, isolating this rule's state from any other
    /// rule/deployment sharing the same Valkey instance.
    pub(super) namespace: String,
    /// Rule identifier, folded into the per-key hash alongside `namespace`.
    pub(super) rule: String,
    /// Sliding-window budgets enforced atomically per key.
    pub(super) budgets: Vec<Budget>,
    /// Time after which an ambiguous (never-reconciled) reservation is
    /// charged at its estimate, mirroring the in-memory ledger's own
    /// field of the same name.
    pub(super) reservation_timeout_ms: u64,
    /// Maximum distinct keys retained per namespace.
    pub(super) max_keys: usize,
    /// Maximum reservations awaiting reconciliation across all keys in
    /// this namespace.
    pub(super) max_active_reservations: usize,
}

impl ValkeyTokenRateLimitBackend {
    /// Open a (lazy, not-yet-connected) Valkey client for this backend.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] if `config.url` isn't a
    /// well-formed Valkey/Redis connection URL.
    pub(super) fn new(config: ValkeyBackendConfig) -> Result<Self, BackendError> {
        let valkey = ValkeyEval::new(config.url)?;
        let limit = config.budgets.iter().map(|budget| budget.capacity).min().unwrap_or(0);
        Ok(Self {
            valkey,
            namespace: config.namespace,
            rule: config.rule,
            budgets: config.budgets,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            limit,
            worker: ReconcileWorker::new(),
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
            budgets: self.budgets.clone(),
            reservation_timeout_ms: self.reservation_timeout_ms,
            max_keys: self.max_keys,
            max_active_reservations: self.max_active_reservations,
            limit: self.limit,
            worker: ReconcileWorker::detached(),
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

    /// Deterministic per-key Valkey key names for this rule/namespace.
    fn key_parts(&self, key: &str) -> [String; 7] {
        let mut digest = Sha256::new();
        digest.update(self.namespace.as_bytes());
        digest.update([0]);
        digest.update(self.rule.as_bytes());
        digest.update([0]);
        digest.update(key.as_bytes());
        let hash = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let prefix = format!("{}:v1:{}", self.namespace, hash);
        [
            prefix.clone(),
            format!("{prefix}:settled"),
            format!("{prefix}:active"),
            format!("{}:keys", self.namespace),
            format!("{}:active-count", self.namespace),
            format!("{}:reservation-seq", self.namespace),
            format!("{}:active-index", self.namespace),
        ]
    }
}

/// Arguments for [`RESERVE_SCRIPT`]: timeout/bounds, then one
/// `(window_ms, capacity)` pair per configured budget.
fn reserve_args(
    reservation_timeout_ms: u64,
    max_keys: usize,
    max_active_reservations: usize,
    request: &ReserveRequest,
    budgets: &[Budget],
) -> Vec<String> {
    let mut args = vec![
        reservation_timeout_ms.to_string(),
        max_keys.to_string(),
        max_active_reservations.to_string(),
        request.estimate.to_string(),
        budgets.len().to_string(),
    ];
    for budget in budgets {
        args.push(budget.window_ms.to_string());
        args.push(budget.capacity.to_string());
    }
    args
}

#[async_trait]
impl TokenRateLimitStateBackend for ValkeyTokenRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        let keys = self.key_parts(&request.key);
        let args = reserve_args(
            self.reservation_timeout_ms,
            self.max_keys,
            self.max_active_reservations,
            &request,
            &self.budgets,
        );
        let response = self.valkey.eval(RESERVE_SCRIPT, &keys, &args).await?;
        match response.as_slice() {
            [1, id, estimate] => Ok(BackendReserve::Admitted {
                reservation_id: u64::try_from(*id).map_err(|_error| BackendError::InvalidResponse)?,
                estimate: u64::try_from(*estimate).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            [0, retry_after] => Ok(BackendReserve::Denied {
                retry_after_ms: u64::try_from(*retry_after).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            _ => Err(BackendError::InvalidResponse),
        }
    }

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        let keys = self.key_parts(&request.key);
        let actual = request.actual.unwrap_or(request.estimate);
        let args = [request.reservation_id.to_string(), actual.to_string()];
        let response = self.valkey.eval(RECONCILE_SCRIPT, &keys, &args).await?;
        match response.as_slice() {
            [0] => Ok(BackendSettlement::Noop),
            [1, actual, refund, overage] => Ok(BackendSettlement::Applied {
                actual: u64::try_from(*actual).map_err(|_error| BackendError::InvalidResponse)?,
                refund: u64::try_from(*refund).map_err(|_error| BackendError::InvalidResponse)?,
                overage: u64::try_from(*overage).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            _ => Err(BackendError::InvalidResponse),
        }
    }

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        self.start_worker()?;
        self.worker.enqueue(request)
    }

    fn limit(&self) -> u64 {
        self.limit
    }
}

// -----------------------------------------------------------------------------
// Valkey-backed token bucket
// -----------------------------------------------------------------------------

/// Atomically admit a reservation against one key's token bucket, or deny
/// it -- the Valkey/Lua analog of [`TokenBucketLedger::reserve`].
///
/// `KEYS`: `[1]` physical state hash (`tokens`, `last_refill_ms`), `[2]`
/// active hash, `[3]` namespace keys zset, `[4]` namespace active-count
/// string, `[5]` namespace reservation-id sequence, `[6]` namespace
/// active-index zset. Deliberately namespaced with a `:tb:` segment
/// distinct from [`RESERVE_SCRIPT`]'s sliding-window keys (see
/// [`ValkeyTokenBucketBackend::key_parts`]) so a `token_bucket` rule and
/// a `sliding_window` rule can safely share one `namespace:` without
/// either algorithm's bookkeeping corrupting the other's. `ARGV`: `[1]`
/// capacity, `[2]` `refill_rate` (tokens/sec), `[3]` reservation timeout
/// (ms), `[4]` max keys, `[5]` max active reservations, `[6]` estimate.
/// Returns `[1, id, estimate]` on admission or `[0, retry_after_ms]` on
/// denial.
pub(super) const TOKEN_BUCKET_RESERVE_SCRIPT: &str = "
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local timeout_ms = tonumber(ARGV[3])
local max_keys = tonumber(ARGV[4])
local max_active = tonumber(ARGV[5])
local estimate = tonumber(ARGV[6])

local active_total = tonumber(redis.call('GET', KEYS[4]) or '0')

local expired_global = redis.call('ZRANGE', KEYS[6], '-inf', now_ms, 'BYSCORE')
for i = 1, #expired_global do
  local member = expired_global[i]
  local split = string.find(member, '|')
  if split then
    local physical = string.sub(member, 1, split - 1)
    local reservation = string.sub(member, split + 1)
    local active_key = physical .. ':active'
    local value = redis.call('HGET', active_key, reservation)
    if value then
      redis.call('HDEL', active_key, reservation)
      active_total = math.max(0, active_total - 1)
      -- Tokens for an abandoned reservation stay charged (already
      -- decremented at reserve time under the immediate-decrement
      -- design): no credit-back happens on expiry, only on reconcile.
    end
  end
  redis.call('ZREM', KEYS[6], member)
end
redis.call('SET', KEYS[4], active_total)

local state = redis.call('HMGET', KEYS[1], 'tokens', 'last_refill_ms')
local tokens = tonumber(state[1])
local last_refill_ms = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  last_refill_ms = now_ms
end
local elapsed_ms = math.max(0, now_ms - last_refill_ms)
tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local ttl = math.max(math.ceil((capacity / refill_rate) * 1000) + timeout_ms, 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now_ms)
local key_exists = redis.call('EXISTS', KEYS[1]) == 1
if not key_exists and redis.call('ZCARD', KEYS[3]) >= max_keys then
  redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
  return {0, 1}
end
if active_total >= max_active then
  redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
  return {0, 1}
end
if tokens < estimate then
  local deficit = estimate - tokens
  local retry_after_ms = math.max(1, math.ceil((deficit / refill_rate) * 1000))
  redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
  redis.call('PEXPIRE', KEYS[1], ttl)
  return {0, retry_after_ms}
end

tokens = tokens - estimate
local id = redis.call('INCR', KEYS[5])
redis.call('HSET', KEYS[2], id, estimate .. '|' .. now_ms)
redis.call('INCR', KEYS[4])
redis.call('ZADD', KEYS[6], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
redis.call('ZADD', KEYS[3], now_ms + ttl, KEYS[1])
redis.call('PEXPIRE', KEYS[1], ttl)
redis.call('PEXPIRE', KEYS[2], ttl)
return {1, id, estimate}
";

/// Atomically settle a prior token-bucket reservation against actual
/// usage -- the Valkey/Lua analog of [`TokenBucketLedger::reconcile`].
///
/// `KEYS`: same layout as [`TOKEN_BUCKET_RESERVE_SCRIPT`]. `ARGV`: `[1]`
/// reservation ID, `[2]` actual usage, `[3]` capacity, `[4]` `refill_rate`.
/// Returns `[0]` if the reservation was already reconciled/expired
/// (no-op), or `[1, actual, refund, overage]`.
const TOKEN_BUCKET_RECONCILE_SCRIPT: &str = "
local value = redis.call('HGET', KEYS[2], ARGV[1])
if not value then return {0} end
local sep = string.find(value, '|')
local estimate = tonumber(string.sub(value, 1, sep - 1))
local actual = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local refill_rate = tonumber(ARGV[4])
redis.call('HDEL', KEYS[2], ARGV[1])
local active_total = math.max(0, tonumber(redis.call('GET', KEYS[4]) or '0') - 1)
redis.call('SET', KEYS[4], active_total)
redis.call('ZREM', KEYS[6], KEYS[1] .. '|' .. ARGV[1])

local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local state = redis.call('HMGET', KEYS[1], 'tokens', 'last_refill_ms')
local tokens = tonumber(state[1])
local last_refill_ms = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  last_refill_ms = now_ms
end
local elapsed_ms = math.max(0, now_ms - last_refill_ms)
tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local refund = math.max(0, estimate - actual)
local overage = math.max(0, actual - estimate)
if refund > 0 then
  tokens = math.min(capacity, tokens + refund)
elseif overage > 0 then
  tokens = math.max(0, tokens - overage)
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last_refill_ms', now_ms)
return {1, actual, refund, overage}
";

/// Valkey/Redis-backed token-bucket state, shared across every gateway
/// instance/replica pointed at the same `url`/`namespace`.
pub(super) struct ValkeyTokenBucketBackend {
    /// Shared connection/EVAL handling, see [`ValkeyEval`].
    valkey: ValkeyEval,
    /// Key namespace prefix, see [`ValkeyBackendConfig::namespace`].
    namespace: String,
    /// Rule identifier, see [`ValkeyBackendConfig::rule`].
    rule: String,
    /// Maximum tokens held at once.
    capacity: u64,
    /// Tokens refilled per second, up to `capacity`.
    refill_rate: f64,
    /// See [`ValkeyBackendConfig::reservation_timeout_ms`].
    reservation_timeout_ms: u64,
    /// See [`ValkeyBackendConfig::max_keys`].
    max_keys: usize,
    /// See [`ValkeyBackendConfig::max_active_reservations`].
    max_active_reservations: usize,
    /// Shared background-reconciliation scaffolding, see [`ReconcileWorker`].
    worker: ReconcileWorker,
}

/// Construction parameters for [`ValkeyTokenBucketBackend`].
pub(super) struct ValkeyTokenBucketConfig {
    /// Valkey/Redis connection URL, already `${ENV_VAR}`-expanded.
    pub(super) url: String,
    /// Key namespace prefix, isolating this rule's state from any other
    /// rule/deployment sharing the same Valkey instance.
    pub(super) namespace: String,
    /// Rule identifier, folded into the per-key hash alongside `namespace`.
    pub(super) rule: String,
    /// Maximum tokens held at once.
    pub(super) capacity: u64,
    /// Tokens refilled per second, up to `capacity`.
    pub(super) refill_rate: f64,
    /// Time after which an ambiguous (never-reconciled) reservation
    /// stops being tracked as active (it's already charged).
    pub(super) reservation_timeout_ms: u64,
    /// Maximum distinct keys retained per namespace/algorithm.
    pub(super) max_keys: usize,
    /// Maximum reservations awaiting reconciliation across all keys in
    /// this namespace/algorithm.
    pub(super) max_active_reservations: usize,
}

impl ValkeyTokenBucketBackend {
    /// Open a (lazy, not-yet-connected) Valkey client for this backend.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] if `config.url` isn't a
    /// well-formed Valkey/Redis connection URL, if `capacity`/
    /// `refill_rate` aren't positive and finite, or if `capacity` or
    /// `capacity / refill_rate` exceeds the bounds documented on
    /// [`token_bucket_ledger::MAX_F64_SAFE_INTEGER`]/
    /// [`token_bucket_ledger::MAX_CAPACITY_REFILL_RATE_RATIO_SECS`].
    pub(super) fn new(config: ValkeyTokenBucketConfig) -> Result<Self, BackendError> {
        token_bucket_ledger::validate_capacity_and_refill_rate(config.capacity, config.refill_rate)
            .map_err(BackendError::Unavailable)?;
        let valkey = ValkeyEval::new(config.url)?;
        Ok(Self {
            valkey,
            namespace: config.namespace,
            rule: config.rule,
            capacity: config.capacity,
            refill_rate: config.refill_rate,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            worker: ReconcileWorker::new(),
        })
    }

    /// See [`ValkeyTokenRateLimitBackend::clone_without_sender`].
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
            worker: ReconcileWorker::detached(),
        }
    }

    /// See [`ValkeyTokenRateLimitBackend::start_worker`].
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

    /// Deterministic per-key Valkey key names for this rule/namespace,
    /// under a `:tb:` segment distinct from the sliding-window backend's
    /// [`ValkeyTokenRateLimitBackend::key_parts`] -- see
    /// [`TOKEN_BUCKET_RESERVE_SCRIPT`]'s doc comment for why the two
    /// algorithms must never share bookkeeping keys.
    fn key_parts(&self, key: &str) -> [String; 6] {
        let mut digest = Sha256::new();
        digest.update(self.namespace.as_bytes());
        digest.update([0]);
        digest.update(b"token_bucket");
        digest.update([0]);
        digest.update(self.rule.as_bytes());
        digest.update([0]);
        digest.update(key.as_bytes());
        let hash = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let prefix = format!("{}:v1:tb:{}", self.namespace, hash);
        [
            prefix.clone(),
            format!("{prefix}:active"),
            format!("{}:tb:keys", self.namespace),
            format!("{}:tb:active-count", self.namespace),
            format!("{}:tb:reservation-seq", self.namespace),
            format!("{}:tb:active-index", self.namespace),
        ]
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for ValkeyTokenBucketBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        let keys = self.key_parts(&request.key);
        let args = [
            self.capacity.to_string(),
            self.refill_rate.to_string(),
            self.reservation_timeout_ms.to_string(),
            self.max_keys.to_string(),
            self.max_active_reservations.to_string(),
            request.estimate.to_string(),
        ];
        let response = self.valkey.eval(TOKEN_BUCKET_RESERVE_SCRIPT, &keys, &args).await?;
        match response.as_slice() {
            [1, id, estimate] => Ok(BackendReserve::Admitted {
                reservation_id: u64::try_from(*id).map_err(|_error| BackendError::InvalidResponse)?,
                estimate: u64::try_from(*estimate).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            [0, retry_after] => Ok(BackendReserve::Denied {
                retry_after_ms: u64::try_from(*retry_after).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            _ => Err(BackendError::InvalidResponse),
        }
    }

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError> {
        let keys = self.key_parts(&request.key);
        let actual = request.actual.unwrap_or(request.estimate);
        let args = [
            request.reservation_id.to_string(),
            actual.to_string(),
            self.capacity.to_string(),
            self.refill_rate.to_string(),
        ];
        let response = self.valkey.eval(TOKEN_BUCKET_RECONCILE_SCRIPT, &keys, &args).await?;
        match response.as_slice() {
            [0] => Ok(BackendSettlement::Noop),
            [1, actual, refund, overage] => Ok(BackendSettlement::Applied {
                actual: u64::try_from(*actual).map_err(|_error| BackendError::InvalidResponse)?,
                refund: u64::try_from(*refund).map_err(|_error| BackendError::InvalidResponse)?,
                overage: u64::try_from(*overage).map_err(|_error| BackendError::InvalidResponse)?,
            }),
            _ => Err(BackendError::InvalidResponse),
        }
    }

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError> {
        self.start_worker()?;
        self.worker.enqueue(request)
    }

    fn limit(&self) -> u64 {
        self.capacity
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::token_rate_limit::ledger::LedgerConfig;

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
            estimate: 40,
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

    #[test]
    fn valkey_backend_has_no_local_state_to_reconcile_or_clean_up_synchronously() {
        // Business behavior under test: a networked backend must never
        // silently answer a synchronous, no-I/O query -- the filter relies
        // on `None` here to route reconciliation through the background
        // worker (`enqueue_reconcile`) instead, and to skip gauge
        // reporting rather than publish misleading zeros. No live Valkey
        // is required: both methods short-circuit before any I/O.
        let backend = ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            budgets: vec![Budget {
                window_ms: 1_000,
                capacity: 10,
            }],
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        })
        .unwrap();

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
                    estimate: 1,
                    now_ms: 0,
                })
                .is_none(),
            "Valkey-backed reconciliation must go through enqueue_reconcile, not reconcile_sync"
        );
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
            estimate: 50,
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

    // -------------------------------------------------------------------------
    // ValkeyTokenBucketBackend construction-time validation and the
    // same no-I/O trait-contract checks as the sliding-window backend.
    // -------------------------------------------------------------------------

    #[test]
    fn valkey_token_bucket_backend_rejects_zero_capacity_or_refill_rate() {
        let base = || ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        };
        assert!(ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig { capacity: 0, ..base() }).is_err());
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                refill_rate: 0.0,
                ..base()
            })
            .is_err()
        );
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_non_finite_refill_rate() {
        // Proves the shared validator (see non_finite_refill_rate_is_rejected)
        // is actually wired into this backend's constructor too.
        let base = || ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        };
        for bad_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                    refill_rate: bad_rate,
                    ..base()
                })
                .is_err(),
                "refill_rate {bad_rate} must be rejected as non-finite/non-positive"
            );
        }
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_capacity_exceeding_f64_safe_integer() {
        // See MAX_F64_SAFE_INTEGER's doc comment for why this bound exists;
        // proven here for the Valkey backend's own constructor.
        let base = || ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        };
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                capacity: token_bucket_ledger::MAX_F64_SAFE_INTEGER + 1,
                ..base()
            })
            .is_err(),
            "capacity above 2^53 must be rejected before precision is silently lost"
        );
    }

    #[test]
    fn valkey_token_bucket_backend_rejects_a_refill_rate_ratio_exceeding_the_pexpire_ttl_bound() {
        // See MAX_CAPACITY_REFILL_RATE_RATIO_SECS's doc comment for why
        // this bound exists; proven here for the Valkey backend's own
        // constructor.
        let base = || ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        };
        assert!(
            ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
                capacity: 10,
                refill_rate: 10.0 / (token_bucket_ledger::MAX_CAPACITY_REFILL_RATE_RATIO_SECS * 2.0),
                ..base()
            })
            .is_err(),
            "a capacity/refill_rate ratio beyond the PEXPIRE TTL bound must be rejected"
        );
    }

    #[test]
    fn valkey_token_bucket_backend_has_no_local_state_to_reconcile_or_clean_up_synchronously() {
        let backend = ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "ns".into(),
            rule: "default".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        })
        .unwrap();
        assert!(backend.cleanup(0, 8).is_none());
        assert!(
            backend
                .reconcile_sync(&ReconcileRequest {
                    key: "a".into(),
                    reservation_id: 1,
                    actual: Some(1),
                    estimate: 1,
                    now_ms: 0,
                })
                .is_none()
        );
    }

    /// A [`ValkeyTokenBucketBackend`] and a [`ValkeyTokenRateLimitBackend`]
    /// sharing the same `namespace`/`rule` -- the plausible, even likely,
    /// operator config that
    /// [`valkey_token_bucket_and_sliding_window_key_parts_never_collide_even_in_the_same_namespace`] exercises.
    fn same_namespace_backends() -> (ValkeyTokenBucketBackend, ValkeyTokenRateLimitBackend) {
        let bucket = ValkeyTokenBucketBackend::new(ValkeyTokenBucketConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "shared".into(),
            rule: "same-rule-name".into(),
            capacity: 10,
            refill_rate: 1.0,
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        })
        .unwrap();
        let sliding = ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
            url: "redis://127.0.0.1:1".into(),
            namespace: "shared".into(),
            rule: "same-rule-name".into(),
            budgets: vec![Budget {
                window_ms: 1_000,
                capacity: 10,
            }],
            reservation_timeout_ms: 1_000,
            max_keys: 8,
            max_active_reservations: 8,
        })
        .unwrap();
        (bucket, sliding)
    }

    #[test]
    fn valkey_token_bucket_and_sliding_window_key_parts_never_collide_even_in_the_same_namespace() {
        // Two rules sharing one `namespace:` must never let one
        // algorithm's Lua script reap or mutate the other's
        // physical/bookkeeping keys.
        let (bucket, sliding) = same_namespace_backends();
        let bucket_keys = bucket.key_parts("same-key");
        let sliding_keys = sliding.key_parts("same-key");
        for bucket_key in &bucket_keys {
            assert!(
                !sliding_keys.contains(bucket_key),
                "token_bucket key {bucket_key} collided with a sliding_window key"
            );
        }
    }

    #[test]
    fn worker_enqueue_fails_once_its_receiver_is_gone() {
        let worker = ReconcileWorker::detached();
        let request = ReconcileRequest {
            key: "a".into(),
            reservation_id: 1,
            actual: Some(1),
            estimate: 1,
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
            estimate: 1,
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
