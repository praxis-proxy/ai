//! Pluggable token-rate-limit state backends.

use std::{
    sync::{Arc, LazyLock, Mutex, OnceLock},
    time::Duration,
};

use async_trait::async_trait;
use redis::aio::ConnectionManager;
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc;

use super::ledger::{Budget, Decision, Ledger, Settlement};

#[derive(Debug, Clone)]
pub(crate) struct ReserveRequest {
    pub(crate) key: String,
    pub(crate) estimate: u64,
    pub(crate) now_ms: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct ReconcileRequest {
    pub(crate) key: String,
    pub(crate) reservation_id: u64,
    pub(crate) actual: Option<u64>,
    pub(crate) estimate: u64,
    pub(crate) now_ms: u64,
}

#[derive(Debug, Clone)]
pub(crate) enum BackendReserve {
    Admitted { reservation_id: u64, estimate: u64 },
    Denied { retry_after_ms: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendSettlement {
    Applied { actual: u64, refund: u64, overage: u64 },
    Noop,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum BackendError {
    #[error("shared quota backend unavailable: {0}")]
    Unavailable(String),
    #[error("shared quota backend returned an invalid response")]
    InvalidResponse,
}

#[async_trait]
pub(crate) trait TokenRateLimitStateBackend: Send + Sync {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError>;

    async fn reconcile(&self, request: ReconcileRequest) -> Result<BackendSettlement, BackendError>;

    fn enqueue_reconcile(&self, request: ReconcileRequest) -> Result<(), BackendError>;

    fn limit(&self) -> u64;

    fn local_state(&self) -> Option<(&Ledger, u64)> {
        None
    }
}

pub(crate) struct InMemoryTokenRateLimitBackend {
    ledger: Arc<Ledger>,
}

impl InMemoryTokenRateLimitBackend {
    pub(crate) fn new(ledger: Ledger) -> Self {
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
local settled_index = KEYS[8]
local active_tokens_key = KEYS[9]
local bucket_ms = tonumber(ARGV[6 + (budget_count * 2)])
local max_buckets = tonumber(ARGV[7 + (budget_count * 2)])

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
      local bucket = math.floor(reserved_at / bucket_ms) * bucket_ms
      redis.call('HINCRBY', physical .. ':settled', bucket, amount)
      redis.call('ZADD', physical .. ':settled-index', bucket, bucket)
      redis.call('HDEL', active_key, reservation)
      redis.call('DECRBY', physical .. ':active-tokens', amount)
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
end
local expired_buckets = redis.call('ZRANGEBYSCORE', settled_index, '-inf', now_ms - max_window)
for i = 1, #expired_buckets do
  redis.call('HDEL', settled, expired_buckets[i])
  redis.call('ZREM', settled_index, expired_buckets[i])
end
if redis.call('ZCARD', settled_index) > max_buckets then
  return {0, max_window}
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
  local buckets = redis.call('ZRANGEBYSCORE', settled_index, now_ms - window, '+inf')
  for j = 1, #buckets do
    settled_sum = settled_sum + tonumber(redis.call('HGET', settled, buckets[j]) or '0')
  end
  local active_sum = tonumber(redis.call('GET', active_tokens_key) or '0')
  if settled_sum + active_sum + estimate > capacity then
    return {0, max_window}
  end
end

local id = redis.call('INCR', KEYS[6])
redis.call('HSET', active, id, estimate .. '|' .. now_ms)
redis.call('INCR', KEYS[5])
redis.call('INCRBY', active_tokens_key, estimate)
redis.call('ZADD', KEYS[7], now_ms + timeout_ms, KEYS[1] .. '|' .. id)
local ttl = math.max(max_window + timeout_ms, 1000)
redis.call('ZADD', KEYS[4], now_ms + ttl, KEYS[1])
redis.call('PEXPIRE', settled, ttl)
redis.call('PEXPIRE', active, ttl)
redis.call('PEXPIRE', settled_index, ttl)
redis.call('PEXPIRE', active_tokens_key, ttl)
redis.call('PEXPIRE', KEYS[1], ttl)
return {1, id, estimate}
";

const RECONCILE_SCRIPT: &str = "
local value = redis.call('HGET', KEYS[3], ARGV[1])
if not value then return {0} end
local sep = string.find(value, '|')
local estimate = tonumber(string.sub(value, 1, sep - 1))
local reserved_at = tonumber(string.sub(value, sep + 1))
local actual = tonumber(ARGV[2])
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('DECRBY', KEYS[9], estimate)
local active_total = math.max(0, tonumber(redis.call('GET', KEYS[5]) or '0') - 1)
redis.call('SET', KEYS[5], active_total)
redis.call('ZREM', KEYS[7], KEYS[1] .. '|' .. ARGV[1])
local bucket_ms = tonumber(ARGV[3])
local bucket = math.floor(reserved_at / bucket_ms) * bucket_ms
redis.call('HINCRBY', KEYS[2], bucket, actual)
redis.call('ZADD', KEYS[8], bucket, bucket)
return {1, actual, math.max(0, estimate - actual), math.max(0, actual - estimate)}
";

static RESERVE_LUA: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(RESERVE_SCRIPT));
static RECONCILE_LUA: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(RECONCILE_SCRIPT));

pub(crate) struct ValkeyTokenRateLimitBackend {
    client: redis::Client,
    connection_manager: tokio::sync::OnceCell<ConnectionManager>,
    namespace: String,
    rule: String,
    budgets: Vec<Budget>,
    reservation_timeout_ms: u64,
    max_keys: usize,
    max_active_reservations: usize,
    limit: u64,
    reconcile_tx: mpsc::Sender<ReconcileRequest>,
    reconcile_rx: Mutex<Option<mpsc::Receiver<ReconcileRequest>>>,
    worker_started: OnceLock<()>,
}

pub(crate) struct ValkeyBackendConfig {
    pub(crate) url: String,
    pub(crate) namespace: String,
    pub(crate) rule: String,
    pub(crate) budgets: Vec<Budget>,
    pub(crate) reservation_timeout_ms: u64,
    pub(crate) max_keys: usize,
    pub(crate) max_active_reservations: usize,
}

impl ValkeyTokenRateLimitBackend {
    pub(crate) fn new(config: ValkeyBackendConfig) -> Result<Self, BackendError> {
        let client = redis::Client::open(config.url).map_err(|e| BackendError::Unavailable(e.to_string()))?;
        let limit = config.budgets.iter().map(|budget| budget.capacity).min().unwrap_or(0);
        let (reconcile_tx, reconcile_rx) = mpsc::channel(1024);
        let worker_backend = Self {
            client,
            connection_manager: tokio::sync::OnceCell::const_new(),
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

    fn clone_without_sender(&self) -> Self {
        let (tx, _rx) = mpsc::channel(1);
        Self {
            client: self.client.clone(),
            connection_manager: tokio::sync::OnceCell::const_new(),
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

    fn start_worker(&self) -> Result<(), BackendError> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_error| BackendError::Unavailable("Valkey reconciliation requires a Tokio runtime".into()))?;
        self.worker_started.get_or_init(|| {
            let Some(mut receiver) = self.reconcile_rx.lock().ok().and_then(|mut guard| guard.take()) else {
                return;
            };
            let worker = self.clone_without_sender();
            runtime.spawn(async move {
                while let Some(request) = receiver.recv().await {
                    let mut attempts = 0;
                    loop {
                        match worker.reconcile(request.clone()).await {
                            Ok(_) => {
                                metrics::counter!("praxis_ai_token_rate_limit_backend_reconciliation_total", "backend" => "valkey", "result" => "completed").increment(1);
                                break;
                            },
                            Err(error) if attempts < 2 => {
                                attempts += 1;
                                tracing::warn!(attempts, %error, "token-rate-limit reconciliation retry");
                                tokio::time::sleep(Duration::from_millis(25 * attempts)).await;
                            },
                            Err(error) => {
                                metrics::counter!("praxis_ai_token_rate_limit_backend_errors_total", "backend" => "valkey", "operation" => "reconcile").increment(1);
                                tracing::error!(%error, "token-rate-limit reconciliation abandoned after retries");
                                break;
                            },
                        }
                    }
                }
            });
        });
        Ok(())
    }

    fn key_parts(&self, key: &str) -> [String; 9] {
        let mut rule_digest = Sha256::new();
        rule_digest.update(self.namespace.as_bytes());
        rule_digest.update([0]);
        rule_digest.update(self.rule.as_bytes());
        let rule_hash = rule_digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let rule_prefix = format!("{}:v1:rule:{}", self.namespace, rule_hash);
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
        let prefix = format!("{rule_prefix}:key:{hash}");
        [
            prefix.clone(),
            format!("{prefix}:settled"),
            format!("{prefix}:active"),
            format!("{rule_prefix}:keys"),
            format!("{rule_prefix}:active-count"),
            format!("{rule_prefix}:reservation-seq"),
            format!("{rule_prefix}:active-index"),
            format!("{rule_prefix}:settled-index"),
            format!("{rule_prefix}:active-tokens"),
        ]
    }

    async fn connection(&self) -> Result<ConnectionManager, BackendError> {
        let manager = self
            .connection_manager
            .get_or_try_init(|| async {
                tokio::time::timeout(Duration::from_millis(500), self.client.get_connection_manager())
                    .await
                    .map_err(|_error| BackendError::Unavailable("Valkey connection timed out".into()))?
                    .map_err(|e| BackendError::Unavailable(e.to_string()))
            })
            .await?;
        Ok(manager.clone())
    }
}

#[async_trait]
impl TokenRateLimitStateBackend for ValkeyTokenRateLimitBackend {
    async fn reserve(&self, request: ReserveRequest) -> Result<BackendReserve, BackendError> {
        let keys = self.key_parts(&request.key);
        let mut args: Vec<String> = vec![
            self.reservation_timeout_ms.to_string(),
            self.max_keys.to_string(),
            self.max_active_reservations.to_string(),
            request.estimate.to_string(),
            self.budgets.len().to_string(),
        ];
        for budget in &self.budgets {
            args.push(budget.window_ms.to_string());
            args.push(budget.capacity.to_string());
        }
        args.push("1000".into());
        args.push("4096".into());
        let mut invocation = RESERVE_LUA.key(&keys[0]);
        for key in keys.iter().skip(1) {
            invocation.key(key);
        }
        for arg in args {
            invocation.arg(arg);
        }
        let mut connection = self.connection().await?;
        let response: Vec<i64> =
            tokio::time::timeout(Duration::from_millis(500), invocation.invoke_async(&mut connection))
                .await
                .map_err(|_error| BackendError::Unavailable("Valkey reservation timed out".into()))?
                .map_err(|e| BackendError::Unavailable(e.to_string()))?;
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
        let mut invocation = RECONCILE_LUA.key(&keys[0]);
        for key in keys.iter().skip(1) {
            invocation.key(key);
        }
        invocation.arg(request.reservation_id).arg(actual).arg(1000_u64);
        let mut connection = self.connection().await?;
        let response: Vec<i64> =
            tokio::time::timeout(Duration::from_millis(500), invocation.invoke_async(&mut connection))
                .await
                .map_err(|_error| BackendError::Unavailable("Valkey reconciliation timed out".into()))?
                .map_err(|e| BackendError::Unavailable(e.to_string()))?;
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

#[cfg(test)]
mod tests {
    use super::{super::ledger::Budget, *};

    #[test]
    #[expect(clippy::unwrap_used, reason = "the test supplies a valid local Redis URL")]
    fn key_parts_are_scoped_and_hash_quota_keys() {
        let backend = ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
            url: "redis://127.0.0.1:6379".into(),
            namespace: "praxis-test".into(),
            rule: "default".into(),
            budgets: vec![
                Budget {
                    window_ms: 60_000,
                    capacity: 100,
                },
                Budget {
                    window_ms: 3_600_000,
                    capacity: 1_000,
                },
            ],
            reservation_timeout_ms: 120_000,
            max_keys: 100,
            max_active_reservations: 100,
        })
        .unwrap();
        let parts = backend.key_parts("alice/model-a");

        assert_eq!(parts.len(), 9);
        assert_eq!(backend.limit(), 100);
        assert!(parts.iter().all(|part| part.starts_with("praxis-test:")));
        assert!(parts.iter().all(|part| !part.contains("alice/model-a")));
    }
}
