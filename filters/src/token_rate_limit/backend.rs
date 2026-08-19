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

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use metrics::counter;
use redis::aio::MultiplexedConnection;
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc;

use super::ledger::{Budget, Decision, Ledger, Settlement};

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

    /// In-process ledger handle, if this backend has one.
    ///
    /// Lets the filter reconcile synchronously (no I/O, no async
    /// dispatch) when state is local, without every backend needing to
    /// implement its own synchronous reconciliation path. Returns `None`
    /// for networked backends.
    fn local_state(&self) -> Option<(&Ledger, u64)> {
        None
    }
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

    fn local_state(&self) -> Option<(&Ledger, u64)> {
        Some((&self.ledger, 0))
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
/// A dropped/failed reconciliation after retries is intentionally *not*
/// escalated back to the request that triggered it (that response has
/// already been sent) -- it's counted and logged so operators can audit
/// it, and the reservation still expires and gets conservatively charged
/// via `reservation_timeout` regardless.
async fn run_reconcile_worker(worker: ValkeyTokenRateLimitBackend, mut receiver: mpsc::Receiver<ReconcileRequest>) {
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
                    tokio::time::sleep(std::time::Duration::from_millis(25 * attempts)).await;
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

/// Valkey/Redis-backed sliding-window state, shared across every gateway
/// instance/replica pointed at the same `url`/`namespace`.
///
/// Admission (`reserve`) is synchronous with the request (an EVAL round-
/// trip); reconciliation (`enqueue_reconcile`) is deferred to a
/// background worker so it never adds latency to the response path.
pub(super) struct ValkeyTokenRateLimitBackend {
    /// Lazy Valkey/Redis client; connections are opened per call.
    client: redis::Client,
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
    /// Sending half of the reconciliation queue; cloned into the worker.
    reconcile_tx: mpsc::Sender<ReconcileRequest>,
    /// Receiving half, taken exactly once by [`Self::start_worker`].
    reconcile_rx: Mutex<Option<mpsc::Receiver<ReconcileRequest>>>,
    /// Ensures the background worker is spawned at most once.
    worker_started: OnceLock<()>,
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
        let client = redis::Client::open(config.url).map_err(|e| BackendError::Unavailable(e.to_string()))?;
        let limit = config.budgets.iter().map(|budget| budget.capacity).min().unwrap_or(0);
        let (reconcile_tx, reconcile_rx) = mpsc::channel(1024);
        let worker_backend = Self {
            client,
            namespace: config.namespace,
            rule: config.rule,
            budgets: config.budgets,
            reservation_timeout_ms: config.reservation_timeout_ms,
            max_keys: config.max_keys,
            max_active_reservations: config.max_active_reservations,
            limit,
            reconcile_tx: reconcile_tx.clone(),
            reconcile_rx: Mutex::new(Some(reconcile_rx)),
            worker_started: OnceLock::new(),
        };
        Ok(worker_backend)
    }

    /// Clone this backend's connection/config, but with a throwaway
    /// (never-sent-to) reconciliation channel -- used only to hand the
    /// background worker its own handle to `reserve`/`reconcile` without
    /// it holding the real sender (which would keep the channel open
    /// forever).
    fn clone_without_sender(&self) -> Self {
        let (tx, _rx) = mpsc::channel(1);
        Self {
            client: self.client.clone(),
            namespace: self.namespace.clone(),
            rule: self.rule.clone(),
            budgets: self.budgets.clone(),
            reservation_timeout_ms: self.reservation_timeout_ms,
            max_keys: self.max_keys,
            max_active_reservations: self.max_active_reservations,
            limit: self.limit,
            reconcile_tx: tx,
            reconcile_rx: Mutex::new(None),
            worker_started: OnceLock::new(),
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
        self.worker_started.get_or_init(|| {
            let Some(receiver) = self.reconcile_rx.lock().ok().and_then(|mut guard| guard.take()) else {
                return;
            };
            let worker = self.clone_without_sender();
            runtime.spawn(run_reconcile_worker(worker, receiver));
        });
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

    /// Open a fresh multiplexed connection, bounded by a short timeout so
    /// an unreachable Valkey fails the request quickly (fail closed)
    /// rather than hanging it indefinitely.
    async fn connection(&self) -> Result<MultiplexedConnection, BackendError> {
        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            self.client.get_multiplexed_async_connection(),
        )
        .await
        .map_err(|_error| BackendError::Unavailable("Valkey connection timed out".into()))?
        .map_err(|e| BackendError::Unavailable(e.to_string()))
    }

    /// Run one `EVAL script KEYS... ARGV...` command against a fresh
    /// connection, bounded by a short timeout (fail closed rather than
    /// hang if Valkey is reachable but wedged).
    async fn eval(&self, script: &str, keys: &[String; 7], args: &[String]) -> Result<Vec<i64>, BackendError> {
        let mut command = redis::cmd("EVAL");
        command.arg(script).arg(keys.len());
        for key in keys {
            command.arg(key);
        }
        for arg in args {
            command.arg(arg);
        }
        let mut connection = self.connection().await?;
        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            command.query_async(&mut connection),
        )
        .await
        .map_err(|_error| BackendError::Unavailable("Valkey command timed out".into()))?
        .map_err(|e| BackendError::Unavailable(e.to_string()))
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
        let response = self.eval(RESERVE_SCRIPT, &keys, &args).await?;
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
        let response = self.eval(RECONCILE_SCRIPT, &keys, &args).await?;
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
        self.reconcile_tx
            .try_send(request)
            .map_err(|error| BackendError::Unavailable(format!("reconciliation queue is full or stopped: {error}")))
    }

    fn limit(&self) -> u64 {
        self.limit
    }
}
