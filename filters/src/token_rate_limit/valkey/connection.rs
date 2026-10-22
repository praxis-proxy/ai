// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey connections for the token rate limit backends: one shared
//! multiplexed connection for pipelines, and a small pool of dedicated
//! connections for `WATCH` transactions, which are per-connection state.

use std::{
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use metrics::counter;
use praxis_filter::FilterError;
use rand::RngExt as _;
use redis::{
    aio::MultiplexedConnection,
    sentinel::{SentinelClientBuilder, SentinelServerType},
};

use super::{
    super::{
        backend::BackendError,
        config::{BackendConfig, BackendTopology},
        expand_backend_url_with,
    },
    primary_cache::{DiscoveryStart, Lookup, PrimaryCache, PublishResult},
};

/// Default bound on connecting and on every command's response. The configured
/// operation timeout is also the overall deadline for transaction retries.
pub(in crate::token_rate_limit) const VALKEY_TIMEOUT: Duration = Duration::from_millis(500);
/// Default overall bound for native Sentinel traversal and convergence.
pub(in crate::token_rate_limit) const SENTINEL_TIMEOUT: Duration = Duration::from_secs(3);
/// Delay between bounded native Sentinel convergence attempts.
const SENTINEL_RETRY_INTERVAL: Duration = Duration::from_millis(50);
/// Maximum time spent on one plain redis-rs Sentinel endpoint attempt.
///
/// The public `SentinelClient` API does not accept an async connection config
/// for its internal Sentinel probes. Building one client per endpoint lets the
/// caller apply this outer bound without depending on redis-rs internals.
const SENTINEL_ENDPOINT_TIMEOUT: Duration = VALKEY_TIMEOUT;

/// Prometheus counter for bounded Sentinel discovery attempts.
const METRIC_SENTINEL_DISCOVERY: &str = "praxis_trl_sentinel_discovery_total";
/// Prometheus counter for generation-fenced Sentinel recovery outcomes.
const METRIC_SENTINEL_FAILOVER: &str = "praxis_trl_sentinel_failover_total";

/// Finite bounds for backend connection work.
#[derive(Debug, Clone, Copy)]
pub(in crate::token_rate_limit) struct ValkeyTimeouts {
    /// Data-node connection establishment.
    pub(in crate::token_rate_limit) connect: Duration,
    /// One response and transaction-pool wait.
    pub(in crate::token_rate_limit) operation: Duration,
    /// Complete Sentinel discovery/convergence cycle.
    pub(in crate::token_rate_limit) discovery: Duration,
}

impl Default for ValkeyTimeouts {
    fn default() -> Self {
        Self {
            connect: VALKEY_TIMEOUT,
            operation: VALKEY_TIMEOUT,
            discovery: SENTINEL_TIMEOUT,
        }
    }
}

/// Bound one plain Sentinel endpoint attempt without exceeding the enclosing
/// discovery deadline or the configured connect/operation limits.
fn sentinel_endpoint_timeout(timeouts: ValkeyTimeouts, remaining: Duration) -> Duration {
    timeouts
        .connect
        .min(timeouts.operation)
        .min(SENTINEL_ENDPOINT_TIMEOUT)
        .min(remaining)
}

/// Most idle transaction connections kept for reuse; [`TransactionConnection::finish`]
/// drops any beyond this. The pool is kept small on purpose: a pooled
/// connection that died (a Valkey restart, an idle timeout) is only found
/// out by the request that checks it out next, which fails, so every idle
/// connection is one potential failed request after an outage.
const MAX_IDLE_TRANSACTION_CONNECTIONS: usize = 8;

/// Maximum checked-out transaction connections. At most
/// [`MAX_IDLE_TRANSACTION_CONNECTIONS`] more can be idle. A permit is
/// released when a connection is returned so waiters can reuse it.
const MAX_TRANSACTION_CONNECTIONS: usize = 64;

/// Backoff range after an aborted optimistic transaction.
const FIRST_ABORT_BACKOFF: Duration = Duration::from_millis(1);
/// Longest pause between retries, short against the overall timeout.
const MAX_ABORT_BACKOFF: Duration = Duration::from_millis(16);

/// Filter-wide Valkey access, cloned into every Valkey-backed rule.
///
/// `pub(in crate::token_rate_limit)`, not `pub(super)`: the Valkey
/// backends that hold this are siblings of this module, not descendants
/// of it, matching the same nested-module pattern as
/// `crate::guardrails::providers::nemo`'s `pub(in crate::guardrails)`.
#[derive(Clone)]
pub(in crate::token_rate_limit) struct ValkeyConnection {
    /// Lazy client used to (re-)establish connections.
    source: Arc<ConnectionSource>,
    /// Cached multiplexed connection for pipelines; cleared on any error so
    /// the next call reconnects instead of reusing a wedged socket.
    connection: Arc<Mutex<Option<CachedConnection>>>,
    /// Single-flight shared-socket establishment; cache hits never acquire it.
    shared_open: Arc<tokio::sync::Mutex<()>>,
    /// Dedicated connections for `WATCH` transactions.
    transactions: Arc<TransactionConnections>,
    /// Generation of the cached shared connection.
    generation: Arc<AtomicU64>,
    /// Generation of the discovered Sentinel topology.
    topology_generation: Arc<AtomicU64>,
    /// Synchronizes generation advancement with cache publication and checkout.
    generation_lock: Arc<RwLock<()>>,
    /// Cached Sentinel-discovered primary client, invalidated with topology generation.
    sentinel_primary: Arc<tokio::sync::Mutex<PrimaryCache<redis::Client>>>,
    /// Bounds for connection and operation work.
    timeouts: ValkeyTimeouts,
    /// Bounded operator-facing selector label (`redis` or `valkey`).
    backend_name: &'static str,
}

/// Lazy source for standalone or Sentinel-discovered writable-primary data.
#[expect(
    clippy::large_enum_variant,
    reason = "the redis-rs standalone client is intentionally kept behind this source boundary"
)]
enum ConnectionSource {
    /// Fixed standalone Redis/Valkey client.
    Standalone(redis::Client),
    /// Plain Sentinel endpoints and credentials used to build one public
    /// redis-rs Sentinel client per endpoint attempt.
    Sentinel {
        /// Immutable Sentinel endpoint and credential settings.
        config: Arc<PlainSentinelConfig>,
    },
    /// TLS Sentinel discovery using redis-rs connections directly. The
    /// redis-rs high-level async Sentinel helper does not propagate its
    /// TLS/response configuration through the internal Sentinel reconnect
    /// path, so the backend keeps the same native client transport and issues
    /// the read-only Sentinel query itself.
    SentinelTls {
        /// Immutable Sentinel endpoints and credential settings.
        config: Arc<TlsSentinelConfig>,
    },
}

/// Native redis-rs inputs for bounded plain Sentinel discovery.
struct PlainSentinelConfig {
    /// Ordered plain Sentinel endpoint addresses.
    endpoints: Vec<redis::ConnectionAddr>,
    /// Monitored primary service name.
    service_name: String,
    /// Credentials used by Sentinel probes.
    sentinel_username: Option<String>,
    /// Credentials used by Sentinel probes.
    sentinel_password: Option<String>,
    /// Credentials used by discovered data nodes.
    redis_username: Option<String>,
    /// Credentials used by discovered data nodes.
    redis_password: Option<String>,
}

/// Native redis-rs inputs for the TLS Sentinel discovery fallback.
struct TlsSentinelConfig {
    /// Ordered Sentinel connection information.
    endpoints: Vec<redis::ConnectionInfo>,
    /// Monitored primary service name.
    service_name: String,
    /// Credentials used by discovered data nodes.
    data_settings: redis::RedisConnectionInfo,
    /// Optional custom trust material used by deterministic TLS tests.
    tls_certificates: Option<redis::TlsCertificates>,
}

/// Format a Sentinel-reported primary address as a Redis URL, preserving IPv6
/// literal syntax for redis-rs URL parsing.
fn primary_url(host: &str, port: u16) -> String {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_owned()
    };
    format!("redis://{host}:{port}")
}

impl ValkeyConnection {
    /// Open a lazy client for `url`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when `url` is not a Valkey URL.
    #[cfg(test)]
    pub(in crate::token_rate_limit) fn new(url: String) -> Result<Self, BackendError> {
        Self::standalone_with_timeouts_and_name(url, ValkeyTimeouts::default(), "shared_kv")
    }

    /// Open a lazy standalone client with a selector label for telemetry.
    pub(in crate::token_rate_limit) fn standalone_with_timeouts_and_name(
        url: String,
        timeouts: ValkeyTimeouts,
        backend_name: &'static str,
    ) -> Result<Self, BackendError> {
        let client = redis::Client::open(url).map_err(|error| BackendError::Unavailable(error.to_string()))?;
        Ok(Self {
            source: Arc::new(ConnectionSource::Standalone(client)),
            connection: Arc::new(Mutex::new(None)),
            shared_open: Arc::new(tokio::sync::Mutex::new(())),
            transactions: Arc::new(TransactionConnections::default()),
            generation: Arc::new(AtomicU64::new(0)),
            topology_generation: Arc::new(AtomicU64::new(0)),
            generation_lock: Arc::new(RwLock::new(())),
            sentinel_primary: Arc::new(tokio::sync::Mutex::new(PrimaryCache::new())),
            timeouts,
            backend_name,
        })
    }

    /// Build the shared backend transport from filter configuration. All
    /// Redis/Valkey topology, credential, TLS, and timeout decisions stay in
    /// this backend module; admission code receives only the backend trait.
    pub(in crate::token_rate_limit) fn from_backend_config(
        backend: &BackendConfig,
        backend_name: &'static str,
    ) -> Result<Self, FilterError> {
        Self::from_backend_config_with(backend, backend_name, |name| std::env::var(name))
    }

    /// Build the shared backend with an injected environment lookup.
    ///
    /// Production uses [`std::env::var`]. Keeping the lookup injectable makes
    /// endpoint-scheme decisions deterministic in tests without mutating the
    /// process-global environment.
    #[expect(clippy::too_many_lines, reason = "topology construction is one backend boundary")]
    pub(in crate::token_rate_limit) fn from_backend_config_with(
        backend: &BackendConfig,
        backend_name: &'static str,
        lookup: impl Fn(&str) -> Result<String, std::env::VarError> + Copy,
    ) -> Result<Self, FilterError> {
        let timeouts = ValkeyTimeouts {
            connect: backend.connect_timeout,
            operation: backend.operation_timeout,
            discovery: match backend.topology.as_ref() {
                Some(BackendTopology::Sentinel(sentinel)) => sentinel.discovery_timeout,
                _ => SENTINEL_TIMEOUT,
            },
        };
        let connection = match backend.topology.as_ref() {
            Some(BackendTopology::Standalone(url)) => {
                Self::standalone_with_timeouts_and_name(expand_backend_url_with(url, lookup)?, timeouts, backend_name)?
            },
            Some(BackendTopology::Sentinel(sentinel)) => {
                let endpoints = sentinel
                    .endpoints
                    .iter()
                    .map(|endpoint| expand_backend_url_with(endpoint, lookup))
                    .collect::<Result<Vec<_>, _>>()?;
                let expand_credential = |value: &Option<String>| {
                    value
                        .as_deref()
                        .map(|value| expand_backend_url_with(value, lookup))
                        .transpose()
                };
                let tls = endpoints
                    .first()
                    .is_some_and(|endpoint| endpoint.starts_with("rediss://") || endpoint.starts_with("valkeys://"));
                Self::sentinel(
                    endpoints,
                    sentinel.service_name.clone(),
                    expand_credential(&sentinel.sentinel_username)?,
                    expand_credential(&sentinel.sentinel_password)?,
                    expand_credential(&sentinel.redis_username)?,
                    expand_credential(&sentinel.redis_password)?,
                    tls,
                    timeouts,
                    backend_name,
                )?
            },
            None => {
                return Err(
                    "token_rate_limit: backend.url or backend.sentinel is required for backend.kind: redis/valkey"
                        .into(),
                );
            },
        };
        Ok(connection)
    }

    /// Build a lazy native Sentinel source; discovery is deferred until use.
    #[expect(clippy::too_many_arguments, reason = "independent topology credentials")]
    #[expect(
        clippy::too_many_lines,
        reason = "credentialed native client construction is one validation boundary"
    )]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "topology construction owns immutable endpoint configuration"
    )]
    pub(in crate::token_rate_limit) fn sentinel(
        endpoints: Vec<String>,
        service_name: String,
        sentinel_username: Option<String>,
        sentinel_password: Option<String>,
        redis_username: Option<String>,
        redis_password: Option<String>,
        tls: bool,
        timeouts: ValkeyTimeouts,
        backend_name: &'static str,
    ) -> Result<Self, BackendError> {
        if endpoints.is_empty() || endpoints.len() > 16 || service_name.trim().is_empty() || service_name.len() > 128 {
            return Err(BackendError::Unavailable(
                "Sentinel requires 1..=16 endpoints and a 1..=128 byte service_name".into(),
            ));
        }
        let addresses = endpoints
            .iter()
            .cloned()
            .map(|endpoint| {
                let info = redis::IntoConnectionInfo::into_connection_info(endpoint)
                    .map_err(|error| BackendError::Unavailable(format!("Sentinel endpoint: {error}")))?;
                let endpoint_uses_tls = matches!(info.addr(), redis::ConnectionAddr::TcpTls { .. });
                if endpoint_uses_tls != tls {
                    return Err(BackendError::Unavailable(
                        "Sentinel endpoint scheme must use TLS (rediss:// or valkeys://) when tls is true and a plain redis:// or valkey:// scheme otherwise".into(),
                    ));
                }
                Ok(info.addr().clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source = if tls {
            let sentinel_settings = redis::RedisConnectionInfo::default();
            let sentinel_settings = if let Some(value) = &sentinel_username {
                sentinel_settings.set_username(value)
            } else {
                sentinel_settings
            };
            let sentinel_settings = if let Some(value) = &sentinel_password {
                sentinel_settings.set_password(value)
            } else {
                sentinel_settings
            };
            let endpoints = endpoints
                .iter()
                .cloned()
                .map(|endpoint| {
                    redis::IntoConnectionInfo::into_connection_info(endpoint)
                        .map(|info| {
                            let addr = match info.addr() {
                                redis::ConnectionAddr::TcpTls { host, port, .. } => redis::ConnectionAddr::TcpTls {
                                    host: host.clone(),
                                    port: *port,
                                    insecure: false,
                                    tls_params: None,
                                },
                                other => other.clone(),
                            };
                            info.set_addr(addr).set_redis_settings(sentinel_settings.clone())
                        })
                        .map_err(|error| BackendError::Unavailable(format!("Sentinel endpoint: {error}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let data_settings = redis::RedisConnectionInfo::default();
            let data_settings = if let Some(value) = &redis_username {
                data_settings.set_username(value)
            } else {
                data_settings
            };
            let data_settings = if let Some(value) = &redis_password {
                data_settings.set_password(value)
            } else {
                data_settings
            };
            ConnectionSource::SentinelTls {
                config: Arc::new(TlsSentinelConfig {
                    endpoints,
                    service_name,
                    data_settings,
                    tls_certificates: None,
                }),
            }
        } else {
            ConnectionSource::Sentinel {
                config: Arc::new(PlainSentinelConfig {
                    endpoints: addresses,
                    service_name,
                    sentinel_username,
                    sentinel_password,
                    redis_username,
                    redis_password,
                }),
            }
        };
        Ok(Self {
            source: Arc::new(source),
            connection: Arc::new(Mutex::new(None)),
            shared_open: Arc::new(tokio::sync::Mutex::new(())),
            transactions: Arc::new(TransactionConnections::default()),
            generation: Arc::new(AtomicU64::new(0)),
            topology_generation: Arc::new(AtomicU64::new(0)),
            generation_lock: Arc::new(RwLock::new(())),
            sentinel_primary: Arc::new(tokio::sync::Mutex::new(PrimaryCache::new())),
            timeouts,
            backend_name,
        })
    }

    #[cfg(test)]
    /// Install test-only TLS trust material without changing production config.
    fn install_test_tls_certificates(&mut self, certificates: redis::TlsCertificates) -> Result<(), BackendError> {
        let Some(source) = Arc::get_mut(&mut self.source) else {
            return Err(BackendError::Unavailable("test source is not uniquely owned".into()));
        };
        let ConnectionSource::SentinelTls { config } = source else {
            return Err(BackendError::Unavailable("test source is not TLS Sentinel".into()));
        };
        let Some(config) = Arc::get_mut(config) else {
            return Err(BackendError::Unavailable(
                "test TLS config is not uniquely owned".into(),
            ));
        };
        config.tls_certificates = Some(certificates);
        Ok(())
    }

    /// Stable selector label for bounded telemetry.
    pub(in crate::token_rate_limit) const fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Start one bounded backend operation deadline.
    pub(in crate::token_rate_limit) fn operation_deadline(&self) -> Instant {
        Instant::now() + self.timeouts.operation
    }

    /// Run `pipe` on the shared connection, dropping that connection on
    /// any failure so a wedged one is not reused.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] before dispatch, or
    /// [`BackendError::Unconfirmed`] when a dispatched mutation's reply fails.
    pub(in crate::token_rate_limit) async fn pipeline<T: redis::FromRedisValue>(
        &self,
        pipe: &redis::Pipeline,
    ) -> Result<T, BackendError> {
        self.pipeline_with_error(pipe, command_error).await
    }

    /// Run a pipeline containing only reads. A failed reply is retryable because
    /// no accounting mutation was dispatched; never use this for writes.
    pub(super) async fn read_pipeline<T: redis::FromRedisValue>(
        &self,
        pipe: &redis::Pipeline,
    ) -> Result<T, BackendError> {
        self.pipeline_with_error(pipe, |error| map_valkey_error("read", error))
            .await
    }

    /// Share connection fencing without conflating read and mutation failures.
    async fn pipeline_with_error<T: redis::FromRedisValue>(
        &self,
        pipe: &redis::Pipeline,
        classify: impl FnOnce(&redis::RedisError) -> BackendError,
    ) -> Result<T, BackendError> {
        let (mut connection, generation) = self.shared().await?;
        match pipe.query_async::<T>(&mut connection).await {
            Ok(value) => Ok(value),
            Err(error) => {
                self.invalidate(generation, advances_topology_generation(&error)).await;
                Err(classify(&error))
            },
        }
    }

    /// Map a command error from a dedicated transaction while preserving
    /// topology-changing replies for the next Sentinel checkout.
    pub(super) fn transaction_command_error(&self, error: &redis::RedisError) -> BackendError {
        self.fence_transaction_error(error);
        command_error(error)
    }

    /// A WATCH/read phase has not dispatched an accounting mutation. Its socket
    /// is discarded by the caller, so retrying the read on a fresh one is safe.
    pub(super) fn transaction_read_error(&self, error: &redis::RedisError) -> BackendError {
        self.fence_transaction_error(error);
        map_valkey_error("transaction read", error)
    }

    /// Preserve topology fencing for both read and mutation transaction phases.
    fn fence_transaction_error(&self, error: &redis::RedisError) {
        if advances_topology_generation(error) {
            self.topology_generation.fetch_add(1, Ordering::AcqRel);
            self.fence_shared_connection();
        }
    }

    /// Fence shared and pooled sockets after a topology-changing transaction
    /// failure. The next shared operation must not reuse the old primary.
    fn fence_shared_connection(&self) {
        let _generation_guard = self
            .generation_lock
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut cached = self
            .connection
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cached = None;
        drop(cached);
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.transactions.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Invalidate the shared connection after a failed command. Ordinary
    /// server replies clear the data socket but retain the discovered primary;
    /// only connection-level or topology-changing failures force discovery in
    /// the next topology generation. No failed mutation is replayed here.
    async fn invalidate(&self, generation: u64, advance_topology: bool) {
        if advance_topology {
            self.topology_generation.fetch_add(1, Ordering::AcqRel);
        }
        let should_invalidate = {
            let _generation_guard = self
                .generation_lock
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut cached = self
                .connection
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cached.as_ref().is_some_and(|entry| entry.generation == generation) {
                *cached = None;
                drop(cached);
                self.generation
                    .fetch_max(generation.saturating_add(1), Ordering::AcqRel);
                self.transactions.generation.fetch_add(1, Ordering::AcqRel);
                true
            } else {
                false
            }
        };
        if should_invalidate
            && let Ok(mut idle) = tokio::time::timeout(self.timeouts.operation, self.transactions.idle.lock()).await
        {
            idle.clear();
        }
    }

    /// Check out a dedicated connection for one `WATCH` transaction.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when a new connection cannot
    /// be opened, or when [`MAX_TRANSACTION_CONNECTIONS`] are all in use
    /// and none becomes available within [`VALKEY_TIMEOUT`].
    #[expect(
        clippy::too_many_lines,
        reason = "generation-fenced transaction checkout keeps pool invariants together"
    )]
    pub(super) async fn transaction(&self) -> Result<TransactionConnection, BackendError> {
        let permit = tokio::time::timeout(
            self.timeouts.operation,
            Arc::clone(&self.transactions.semaphore).acquire_owned(),
        )
        .await
        .map_err(|_elapsed| BackendError::Unavailable("transaction pool exhausted: timeout".into()))?
        .map_err(|_closed| BackendError::Unavailable("transaction pool closed".into()))?;
        let (connection, generation, shared_generation_at_checkout) = loop {
            let mut idle = self.transactions.idle.lock().await;
            // Read the generation while holding the same lock that invalidation
            // uses to clear idle connections. This prevents a checkout from
            // accepting a socket invalidated between the old load and pop.
            let generation = self.transactions.generation.load(Ordering::Acquire);
            let pooled = idle.pop();
            drop(idle);
            match pooled {
                Some(PooledConnection {
                    connection,
                    generation: pooled_generation,
                }) if pooled_generation == generation => {
                    let _generation_guard = self
                        .generation_lock
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    break (connection, generation, self.generation.load(Ordering::Acquire));
                },
                Some(PooledConnection { .. }) => {},
                None => {
                    let opening_shared_generation = self.generation.load(Ordering::Acquire);
                    let connection = self.open().await?;
                    let _generation_guard = self
                        .generation_lock
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if self.generation.load(Ordering::Acquire) != opening_shared_generation {
                        drop(connection);
                        continue;
                    }
                    break (connection, generation, opening_shared_generation);
                },
            }
        };
        Ok(TransactionConnection {
            connection: Some(connection),
            pool: Arc::clone(&self.transactions),
            permit: Some(permit),
            generation,
            shared_generation: Arc::clone(&self.generation),
            shared_generation_at_checkout,
            generation_lock: Arc::clone(&self.generation_lock),
        })
    }

    /// Return a shared handle only while its cache generation is current.
    fn cached_shared(&self) -> Option<(MultiplexedConnection, u64)> {
        let cached = self
            .connection
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let generation = self.generation.load(Ordering::Acquire);
        cached
            .as_ref()
            .filter(|entry| entry.generation == generation)
            .map(|entry| (entry.connection.clone(), entry.generation))
    }

    /// Return the cached shared connection, opening (and caching) a fresh
    /// one on first use or after a prior failure cleared it.
    #[expect(
        clippy::too_many_lines,
        reason = "bounded open, generation validation, and cache publication stay together"
    )]
    async fn shared(&self) -> Result<(MultiplexedConnection, u64), BackendError> {
        let lock_timeout = match self.source.as_ref() {
            ConnectionSource::Standalone(_) => self.timeouts.connect,
            ConnectionSource::Sentinel { .. } | ConnectionSource::SentinelTls { .. } => self.timeouts.discovery,
        };
        let deadline = Instant::now() + lock_timeout;
        if let Some(cached) = self.cached_shared() {
            return Ok(cached);
        }
        // Hold only the asynchronous cold-open gate across connection work.
        // Waiting and opening share one deadline, including after failover.
        let _opening_guard = tokio::time::timeout(
            deadline.saturating_duration_since(Instant::now()),
            self.shared_open.lock(),
        )
        .await
        .map_err(|_elapsed| BackendError::Unavailable("shared connection open timeout".into()))?;
        loop {
            if let Some(cached) = self.cached_shared() {
                return Ok(cached);
            }
            let opening_generation = self.generation.load(Ordering::Acquire);
            let connection = self.open_until(deadline).await?;
            if self.generation.load(Ordering::Acquire) == opening_generation {
                let generation_guard = self
                    .generation_lock
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut cached = self
                    .connection
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let current_generation = self.generation.load(Ordering::Acquire);
                if let Some(entry) = cached.as_ref()
                    && entry.generation == current_generation
                {
                    drop(connection);
                    drop(generation_guard);
                    return Ok((entry.connection.clone(), entry.generation));
                }
                if current_generation == opening_generation {
                    *cached = Some(CachedConnection {
                        connection: connection.clone(),
                        generation: opening_generation,
                    });
                    drop(cached);
                    drop(generation_guard);
                    return Ok((connection, opening_generation));
                }
                drop(cached);
                drop(generation_guard);
                drop(connection);
            } else {
                drop(connection);
            }
            if deadline.saturating_duration_since(Instant::now()).is_zero() {
                return Err(BackendError::Unavailable(
                    "backend connection generation changed during open".into(),
                ));
            }
        }
    }

    /// Open a fresh connection, bounding both the connect attempt and
    /// every command's response to the configured finite limits.
    async fn open(&self) -> Result<MultiplexedConnection, BackendError> {
        let budget = match self.source.as_ref() {
            ConnectionSource::Standalone(_) => self.timeouts.connect,
            ConnectionSource::Sentinel { .. } | ConnectionSource::SentinelTls { .. } => self.timeouts.discovery,
        };
        self.open_until(Instant::now() + budget).await
    }

    /// Open a connection without exceeding an already-started lifecycle
    /// deadline. Shared-cache lock wait and native discovery use this same
    /// deadline, so coalescing cannot double the configured bound.
    async fn open_until(&self, deadline: Instant) -> Result<MultiplexedConnection, BackendError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(BackendError::Unavailable("backend connection deadline elapsed".into()));
        }
        let config = redis::AsyncConnectionConfig::new()
            .set_connection_timeout(Some(self.timeouts.connect.min(remaining)))
            .set_response_timeout(Some(self.timeouts.operation));
        match self.source.as_ref() {
            ConnectionSource::Standalone(client) => tokio::time::timeout(
                remaining,
                Box::pin(client.get_multiplexed_async_connection_with_config(&config)),
            )
            .await
            .map_err(|_elapsed| BackendError::Unavailable("backend connection deadline elapsed".into()))?
            .map_err(|error| map_valkey_error("connection", &error)),
            ConnectionSource::Sentinel { config: sentinel } => {
                // Discovery is a cold path. Keep its transport state off the
                // stack without allocating on healthy shared/pool cache hits.
                Box::pin(self.open_sentinel_until(sentinel, deadline, &config)).await
            },
            ConnectionSource::SentinelTls { config: sentinel } => {
                Box::pin(self.open_tls_sentinel(sentinel, deadline, &config)).await
            },
        }
    }

    /// Open a transaction connection through the cached or newly discovered
    /// Sentinel primary with bounded, coalesced discovery.
    #[expect(
        clippy::too_many_lines,
        reason = "bounded discovery and primary retry state stay in one lifecycle"
    )]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the discovery action owns a cancellation-resetting cache lease across the match"
    )]
    async fn open_sentinel_until(
        &self,
        sentinel: &PlainSentinelConfig,
        deadline: Instant,
        connection_config: &redis::AsyncConnectionConfig,
    ) -> Result<MultiplexedConnection, BackendError> {
        let mut failover_pending = false;
        loop {
            let generation = self.topology_generation.load(Ordering::Acquire);
            let action = {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let cache = match tokio::time::timeout(remaining, Arc::clone(&self.sentinel_primary).lock_owned()).await
                {
                    Ok(cache) => cache,
                    Err(_elapsed) => {
                        record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                        return Err(BackendError::Unavailable("Sentinel primary cache lock timeout".into()));
                    },
                };
                primary_action(cache, generation)
            };
            match action {
                PrimaryAction::Hit(client) => {
                    match Box::pin(self.open_primary_until(&client, deadline, connection_config)).await {
                        Ok(connection) if self.topology_generation.load(Ordering::Acquire) == generation => {
                            if failover_pending {
                                record_sentinel_failover(self.backend_name, "recovered");
                            }
                            return Ok(connection);
                        },
                        Ok(_) | Err(_) => {
                            failover_pending = true;
                            if let Err(error) = self.invalidate_primary_until(generation, deadline).await {
                                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                                return Err(error);
                            }
                        },
                    }
                },
                PrimaryAction::Wait => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                        return Err(BackendError::Unavailable("Sentinel discovery timeout".into()));
                    }
                    tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(remaining)).await;
                },
                PrimaryAction::Discover(lease) => {
                    failover_pending |= generation != 0;
                    let phase = sentinel_discovery_phase(generation, failover_pending);
                    let discovery = Box::pin(self.discover_plain_primary(sentinel, deadline)).await;
                    let client = match discovery {
                        Ok(client) => client,
                        Err(error) => {
                            record_sentinel_discovery(self.backend_name, phase, "failure");
                            drop(lease);
                            let remaining = deadline.saturating_duration_since(Instant::now());
                            if remaining.is_zero() {
                                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                                return Err(error);
                            }
                            tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(remaining)).await;
                            continue;
                        },
                    };
                    if lease.publish(client.clone()) != PublishResult::Accepted {
                        record_sentinel_discovery(self.backend_name, phase, "stale");
                        continue;
                    }
                    match Box::pin(self.open_primary_until(&client, deadline, connection_config)).await {
                        Ok(connection) if self.topology_generation.load(Ordering::Acquire) == generation => {
                            record_sentinel_discovery(self.backend_name, phase, "success");
                            if failover_pending {
                                record_sentinel_failover(self.backend_name, "recovered");
                            }
                            return Ok(connection);
                        },
                        Ok(_) | Err(_) => {
                            failover_pending = true;
                            let result = if self.topology_generation.load(Ordering::Acquire) == generation {
                                "failure"
                            } else {
                                "stale"
                            };
                            record_sentinel_discovery(self.backend_name, phase, result);
                            if let Err(error) = self.invalidate_primary_until(generation, deadline).await {
                                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                                return Err(error);
                            }
                        },
                    }
                },
            }
            if deadline.saturating_duration_since(Instant::now()).is_zero() {
                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                return Err(BackendError::Unavailable("Sentinel discovery timeout".into()));
            }
        }
    }

    /// Discover a plain primary with a separate bound for each configured
    /// Sentinel endpoint. The public redis-rs helper accepts no per-call
    /// async connection config, so a one-endpoint client keeps a stalled
    /// endpoint from consuming the entire convergence deadline.
    #[expect(
        clippy::too_many_lines,
        reason = "one endpoint owns client construction, bounded probing, and error classification"
    )]
    async fn discover_plain_primary(
        &self,
        sentinel: &PlainSentinelConfig,
        deadline: Instant,
    ) -> Result<redis::Client, BackendError> {
        let mut last_error = None;
        for endpoint in &sentinel.endpoints {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let timeout = sentinel_endpoint_timeout(self.timeouts, remaining);
            let mut builder = SentinelClientBuilder::new(
                std::iter::once(endpoint.clone()),
                sentinel.service_name.as_str(),
                SentinelServerType::Master,
            )
            .map_err(|error| BackendError::Unavailable(format!("Sentinel configuration: {error}")))?;
            if let Some(value) = &sentinel.sentinel_username {
                builder = builder.set_client_to_sentinel_username(value);
            }
            if let Some(value) = &sentinel.sentinel_password {
                builder = builder.set_client_to_sentinel_password(value);
            }
            if let Some(value) = &sentinel.redis_username {
                builder = builder.set_client_to_redis_username(value);
            }
            if let Some(value) = &sentinel.redis_password {
                builder = builder.set_client_to_redis_password(value);
            }
            let mut client = builder
                .build()
                .map_err(|error| BackendError::Unavailable(format!("Sentinel configuration: {error}")))?;
            // redis-rs composes Sentinel and primary probes in this large
            // future; box only this discovery attempt, not request payloads.
            match tokio::time::timeout(timeout, Box::pin(client.async_get_client())).await {
                Ok(Ok(client)) => return Ok(client),
                Ok(Err(error)) => last_error = Some(map_valkey_error("Sentinel discovery", &error)),
                Err(_elapsed) => {
                    last_error = Some(BackendError::Unavailable("Sentinel endpoint timeout".into()));
                },
            }
        }
        Err(last_error.unwrap_or_else(|| BackendError::Unavailable("Sentinel discovery timeout".into())))
    }

    /// Open a connection to a discovered primary without another Sentinel query.
    async fn open_primary_until(
        &self,
        client: &redis::Client,
        deadline: Instant,
        connection_config: &redis::AsyncConnectionConfig,
    ) -> Result<MultiplexedConnection, BackendError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(BackendError::Unavailable("backend connection deadline elapsed".into()));
        }
        tokio::time::timeout(
            remaining,
            client.get_multiplexed_async_connection_with_config(connection_config),
        )
        .await
        .map_err(|_elapsed| BackendError::Unavailable("backend connection deadline elapsed".into()))?
        .map_err(|error| map_valkey_error("Sentinel primary", &error))
    }

    /// Invalidate a discovered primary without extending the enclosing
    /// connection deadline while waiting for the cache mutex.
    async fn invalidate_primary_until(&self, topology_generation: u64, deadline: Instant) -> Result<(), BackendError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(BackendError::Unavailable("Sentinel primary cache lock timeout".into()));
        }
        let mut cache = tokio::time::timeout(remaining, self.sentinel_primary.lock())
            .await
            .map_err(|_elapsed| BackendError::Unavailable("Sentinel primary cache lock timeout".into()))?;
        cache.invalidate(topology_generation);
        drop(cache);
        Ok(())
    }

    /// Discover a TLS primary with native redis-rs connections. This avoids
    /// redis-rs' internal async Sentinel reconnect path, which does not pass
    /// the caller's TLS configuration through its Sentinel probe.
    /// Open a TLS connection through a cached or newly discovered primary.
    #[expect(
        clippy::too_many_lines,
        reason = "bounded TLS discovery owns cache recovery and connection publication"
    )]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the discovery action owns a cancellation-resetting cache lease across the match"
    )]
    async fn open_tls_sentinel(
        &self,
        sentinel: &TlsSentinelConfig,
        deadline: Instant,
        connection_config: &redis::AsyncConnectionConfig,
    ) -> Result<MultiplexedConnection, BackendError> {
        let mut failover_pending = false;
        loop {
            let generation = self.topology_generation.load(Ordering::Acquire);
            let action = {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let cache = match tokio::time::timeout(remaining, Arc::clone(&self.sentinel_primary).lock_owned()).await
                {
                    Ok(cache) => cache,
                    Err(_elapsed) => {
                        record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                        return Err(BackendError::Unavailable("Sentinel primary cache lock timeout".into()));
                    },
                };
                primary_action(cache, generation)
            };
            match action {
                PrimaryAction::Hit(client) => {
                    match Box::pin(self.open_primary_until(&client, deadline, connection_config)).await {
                        Ok(connection) if self.topology_generation.load(Ordering::Acquire) == generation => {
                            if failover_pending {
                                record_sentinel_failover(self.backend_name, "recovered");
                            }
                            return Ok(connection);
                        },
                        Ok(_) | Err(_) => {
                            failover_pending = true;
                            if let Err(error) = self.invalidate_primary_until(generation, deadline).await {
                                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                                return Err(error);
                            }
                        },
                    }
                },
                PrimaryAction::Wait => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                        return Err(BackendError::Unavailable("Sentinel TLS discovery timeout".into()));
                    }
                    tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(remaining)).await;
                },
                PrimaryAction::Discover(lease) => {
                    failover_pending |= generation != 0;
                    let phase = sentinel_discovery_phase(generation, failover_pending);
                    let discovery = Box::pin(self.discover_tls_primary(sentinel, deadline, connection_config)).await;
                    let (client, connection) = match discovery {
                        Ok(value) => value,
                        Err(error) => {
                            record_sentinel_discovery(self.backend_name, phase, "failure");
                            drop(lease);
                            let remaining = deadline.saturating_duration_since(Instant::now());
                            if remaining.is_zero() {
                                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                                return Err(error);
                            }
                            tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(remaining)).await;
                            continue;
                        },
                    };
                    if self.topology_generation.load(Ordering::Acquire) != generation
                        || lease.publish(client) != PublishResult::Accepted
                    {
                        record_sentinel_discovery(self.backend_name, phase, "stale");
                        drop(connection);
                        continue;
                    }
                    record_sentinel_discovery(self.backend_name, phase, "success");
                    if failover_pending {
                        record_sentinel_failover(self.backend_name, "recovered");
                    }
                    return Ok(connection);
                },
            }
            if deadline.saturating_duration_since(Instant::now()).is_zero() {
                record_failed_sentinel_failover(self.backend_name, &mut failover_pending);
                return Err(BackendError::Unavailable("Sentinel TLS discovery timeout".into()));
            }
        }
    }

    /// Discover and verify a TLS primary through the configured Sentinels.
    #[expect(
        clippy::too_many_lines,
        reason = "bounded TLS discovery owns endpoint traversal, role verification, and retry timing"
    )]
    async fn discover_tls_primary(
        &self,
        sentinel: &TlsSentinelConfig,
        deadline: Instant,
        connection_config: &redis::AsyncConnectionConfig,
    ) -> Result<(redis::Client, MultiplexedConnection), BackendError> {
        loop {
            for (index, endpoint) in sentinel.endpoints.iter().enumerate() {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(BackendError::Unavailable("Sentinel TLS discovery timeout".into()));
                }
                // TLS discovery includes two verified connections and ROLE.
                // Share the enclosing budget fairly instead of applying the
                // plain helper's 500ms cap to this multi-hop exchange.
                let endpoints_left = u32::try_from(sentinel.endpoints.len() - index).unwrap_or(u32::MAX);
                let endpoint_deadline = Instant::now() + remaining / endpoints_left;
                let client = tls_client(
                    endpoint.clone(),
                    sentinel.tls_certificates.as_ref(),
                    "Sentinel TLS configuration",
                )?;
                let mut sentinel_connection = match tokio::time::timeout(
                    endpoint_deadline.saturating_duration_since(Instant::now()),
                    Box::pin(client.get_multiplexed_async_connection_with_config(connection_config)),
                )
                .await
                {
                    Ok(Ok(connection)) => connection,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                let address: (String, u16) = match tokio::time::timeout(
                    endpoint_deadline.saturating_duration_since(Instant::now()),
                    redis::cmd("SENTINEL")
                        .arg("get-master-addr-by-name")
                        .arg(&sentinel.service_name)
                        .query_async(&mut sentinel_connection),
                )
                .await
                {
                    Ok(Ok(address)) => address,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                let primary_host = address.0.trim_start_matches('[').trim_end_matches(']').to_owned();
                let base = match redis::IntoConnectionInfo::into_connection_info(primary_url(&primary_host, address.1))
                {
                    Ok(info) => info,
                    Err(_error) => continue,
                };
                let data_info = base
                    .set_addr(redis::ConnectionAddr::TcpTls {
                        host: primary_host,
                        port: address.1,
                        insecure: false,
                        tls_params: None,
                    })
                    .set_redis_settings(sentinel.data_settings.clone());
                let data_client = tls_client(
                    data_info,
                    sentinel.tls_certificates.as_ref(),
                    "TLS primary configuration",
                )?;
                let mut data_connection = match tokio::time::timeout(
                    endpoint_deadline.saturating_duration_since(Instant::now()),
                    Box::pin(data_client.get_multiplexed_async_connection_with_config(connection_config)),
                )
                .await
                {
                    Ok(Ok(connection)) => connection,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                let role: redis::Role = match tokio::time::timeout(
                    endpoint_deadline.saturating_duration_since(Instant::now()),
                    redis::cmd("ROLE").query_async(&mut data_connection),
                )
                .await
                {
                    Ok(Ok(role)) => role,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                if matches!(role, redis::Role::Primary { .. }) {
                    return Ok((data_client, data_connection));
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(BackendError::Unavailable(
                    "Sentinel TLS primary convergence timeout".into(),
                ));
            }
            tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(remaining)).await;
        }
    }
}

/// Whether a failed shared command can indicate that the discovered primary
/// is no longer usable. Ordinary server replies are handled by replacing the
/// data connection while retaining the cached topology.
fn advances_topology_generation(error: &redis::RedisError) -> bool {
    error.is_io_error()
        || error.is_connection_dropped()
        || error.is_timeout()
        || matches!(error.code(), Some("READONLY" | "MASTERDOWN"))
}

/// Classify discovery relative to the initial topology generation.
fn sentinel_discovery_phase(generation: u64, recovery_pending: bool) -> &'static str {
    if generation != 0 || recovery_pending {
        "rediscovery"
    } else {
        "initial"
    }
}

/// Record one bounded Sentinel discovery attempt.
fn record_sentinel_discovery(_backend: &'static str, phase: &'static str, result: &'static str) {
    counter!(
        METRIC_SENTINEL_DISCOVERY,
        "phase" => phase,
        "result" => result,
    )
    .increment(1);
}

/// Record one generation-fenced Sentinel recovery outcome. `recovered` means
/// a writable connection was returned after invalidation; it does not prove a
/// Sentinel promotion occurred.
fn record_sentinel_failover(_backend: &'static str, outcome: &'static str) {
    counter!(METRIC_SENTINEL_FAILOVER, "outcome" => outcome).increment(1);
}

/// Close one pending failover attempt with a bounded failure sample.
fn record_failed_sentinel_failover(backend: &'static str, pending: &mut bool) {
    if *pending {
        record_sentinel_failover(backend, "failed");
        *pending = false;
    }
}

/// Next action for one generation-scoped primary lookup.
#[expect(
    clippy::large_enum_variant,
    reason = "the cached redis-rs client remains directly owned for the hot cache-hit path"
)]
enum PrimaryAction {
    /// Reuse the cached primary client.
    Hit(redis::Client),
    /// Wait for another task's discovery attempt.
    Wait,
    /// Own a cancellation-safe discovery lease.
    Discover(DiscoveryLease),
}

/// Owned primary-cache mutex guard held by a discovery lease.
type PrimaryCacheGuard = tokio::sync::OwnedMutexGuard<PrimaryCache<redis::Client>>;

/// Owns the `Discovering` cache state while a Sentinel query is in flight.
/// Its synchronous drop path clears that state even when the surrounding
/// async task is cancelled, allowing a later request to retry discovery.
struct DiscoveryLease {
    /// Cache state that must be reset if discovery is cancelled.
    cache: Option<PrimaryCacheGuard>,
    /// Generation marked as discovering.
    generation: u64,
}

impl DiscoveryLease {
    /// Publish the discovered client and release the cache lock.
    fn publish(mut self, primary: redis::Client) -> PublishResult {
        let result = self.cache.as_mut().map_or(PublishResult::Discarded, |cache| {
            cache.publish(self.generation, primary)
        });
        if result == PublishResult::Discarded
            && let Some(cache) = self.cache.as_mut()
        {
            cache.invalidate(self.generation);
        }
        self.cache.take();
        result
    }
}

impl Drop for DiscoveryLease {
    fn drop(&mut self) {
        if let Some(cache) = self.cache.as_mut() {
            cache.invalidate(self.generation);
        }
    }
}

/// Select a cache hit, coalesced wait, or owned discovery attempt.
fn primary_action(mut cache: PrimaryCacheGuard, generation: u64) -> PrimaryAction {
    match cache.lookup(generation) {
        Lookup::Hit(client) => PrimaryAction::Hit(client.clone()),
        Lookup::InProgress => PrimaryAction::Wait,
        Lookup::Miss => match cache.begin_discovery(generation) {
            DiscoveryStart::Started => PrimaryAction::Discover(DiscoveryLease {
                cache: Some(cache),
                generation,
            }),
            DiscoveryStart::InProgress => PrimaryAction::Wait,
        },
    }
}

/// Cached shared connection with a conditional invalidation identity.
struct CachedConnection {
    /// Multiplexed connection used by plain-command pipelines.
    connection: MultiplexedConnection,
    /// Generation installed with this connection.
    generation: u64,
}

/// An idle connection. Only checked-out connections hold semaphore permits.
struct PooledConnection {
    /// The idle connection ready for reuse.
    connection: MultiplexedConnection,
    /// Topology generation at which this socket was returned.
    generation: u64,
}

/// Idle dedicated connections, reused across transactions.
struct TransactionConnections {
    /// Connections returned by [`TransactionConnection::finish`].
    idle: tokio::sync::Mutex<Vec<PooledConnection>>,
    /// Bounds the total connections alive at once (idle + checked out).
    semaphore: Arc<tokio::sync::Semaphore>,
    /// Incremented when a checked-out transaction is dropped after an error.
    generation: AtomicU64,
}

impl Default for TransactionConnections {
    fn default() -> Self {
        Self {
            idle: tokio::sync::Mutex::new(Vec::new()),
            semaphore: Arc::new(tokio::sync::Semaphore::new(MAX_TRANSACTION_CONNECTIONS)),
            generation: AtomicU64::new(0),
        }
    }
}

/// One checked-out connection. Call [`Self::finish`] after a completed
/// `EXEC` (or `UNWATCH`); dropping it instead discards the connection, so a
/// connection with a dangling `WATCH` or a mid-flight error never comes back.
pub(super) struct TransactionConnection {
    /// The connection while checked out.
    connection: Option<MultiplexedConnection>,
    /// Pool to return it to.
    pool: Arc<TransactionConnections>,
    /// Held only while checked out; released when the connection is returned.
    permit: Option<tokio::sync::OwnedSemaphorePermit>,
    /// Generation observed at checkout.
    generation: u64,
    /// Shared data-connection generation, fenced on shared connection failure.
    shared_generation: Arc<AtomicU64>,
    /// Shared generation observed when this transaction was checked out.
    shared_generation_at_checkout: u64,
    /// Synchronizes generation advancement with transaction checkout.
    generation_lock: Arc<RwLock<()>>,
}

/// Whether a checked-out transaction still belongs to both current pool
/// generations before its first command. A later generation change only
/// prevents socket reuse; a received command result remains confirmed.
fn transaction_generation_is_current(
    transaction_generation: u64,
    pool_generation: u64,
    shared_generation_at_checkout: u64,
    shared_generation: u64,
) -> bool {
    transaction_generation == pool_generation && shared_generation_at_checkout == shared_generation
}

impl TransactionConnection {
    /// The underlying connection.
    pub(super) fn inner(&mut self) -> &mut MultiplexedConnection {
        match self.connection.as_mut() {
            Some(connection) => connection,
            None => unreachable!("checked-out transaction connection is present"),
        }
    }

    /// Reject a transaction whose checkout epoch was invalidated before its
    /// first command. `finish` repeats this check after the mutation attempt.
    pub(super) fn is_current(&self) -> bool {
        transaction_generation_is_current(
            self.generation,
            self.pool.generation.load(Ordering::Acquire),
            self.shared_generation_at_checkout,
            self.shared_generation.load(Ordering::Acquire),
        )
    }

    /// Return the connection to the pool for reuse, or drop both when
    /// [`MAX_IDLE_TRANSACTION_CONNECTIONS`] are already idle. Cleanup is
    /// bounded by the caller's remaining operation deadline; a completed
    /// command result is not changed by pool invalidation or cleanup timeout.
    pub(super) async fn finish(mut self, remaining: Duration) {
        let Some(connection) = self.connection.take() else {
            unreachable!("checked-out transaction connection is present");
        };
        let pool = Arc::clone(&self.pool);
        let generation = self.generation;
        if let Ok(mut idle) = tokio::time::timeout(remaining, pool.idle.lock()).await
            && self.is_current()
            && idle.len() < MAX_IDLE_TRANSACTION_CONNECTIONS
        {
            idle.push(PooledConnection { connection, generation });
            drop(idle);
            drop(self.permit.take());
            drop(self);
            return;
        }
        drop(self.permit.take());
        drop(self);
    }

    /// Close a known-stale or otherwise cleanly abandoned socket without
    /// advancing the pool generation for unrelated checked-out connections.
    pub(super) fn discard(mut self) {
        self.connection.take();
        drop(self.permit.take());
    }
}

impl Drop for TransactionConnection {
    fn drop(&mut self) {
        if self.connection.is_some() {
            let _generation_guard = self
                .generation_lock
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.pool.generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// Clear a watch before returning a connection without executing a transaction.
/// `UNWATCH` changes only connection state, so failed pre-mutation cleanup is
/// retryable. Callers discard the socket rather than pooling its watch state.
pub(super) async fn unwatch(connection: &mut MultiplexedConnection) -> Result<(), BackendError> {
    redis::cmd("UNWATCH")
        .exec_async(connection)
        .await
        .map_err(|error| map_valkey_error("UNWATCH", &error))
}

/// Retry aborted optimistic transactions within the Valkey deadline.
pub(super) struct AbortRetry {
    /// Monotonic deadline for the retry window.
    deadline: Instant,
    /// Upper bound of the next jittered pause.
    backoff: Duration,
}

impl AbortRetry {
    /// Start the bounded retry window.
    pub(super) fn start(deadline: Instant) -> Self {
        Self {
            deadline,
            backoff: FIRST_ABORT_BACKOFF,
        }
    }

    /// Remaining time in the enclosing operation.
    pub(super) fn remaining(&self) -> Result<Duration, BackendError> {
        self.deadline
            .checked_duration_since(Instant::now())
            .filter(|left| !left.is_zero())
            .ok_or_else(|| BackendError::Unavailable("Valkey operation deadline exceeded".into()))
    }

    /// Pause before another attempt, failing closed once the deadline passes.
    pub(super) async fn pause(&mut self) -> Result<(), BackendError> {
        let left = self.remaining()?;
        let upper = u64::try_from(self.backoff.as_micros()).unwrap_or(u64::MAX);
        let jitter = Duration::from_micros(rand::rng().random_range(upper / 2..=upper));
        tokio::time::sleep(jitter.min(left)).await;
        self.backoff = self.backoff.saturating_mul(2).min(MAX_ABORT_BACKOFF);
        Ok(())
    }
}

/// [`map_valkey_error`] for a failed command (as opposed to connecting).
pub(super) fn command_error(error: &redis::RedisError) -> BackendError {
    BackendError::Unconfirmed(format!("Redis/Valkey command: {error}"))
}

/// Tag a client error with the phase it came from; the client's own text
/// ("timed out") does not say whether connecting or a command failed.
fn map_valkey_error(phase: &'static str, error: &redis::RedisError) -> BackendError {
    BackendError::Unavailable(format!("Redis/Valkey {phase}: {error}"))
}

/// Build one secure TLS client, optionally using test-provided trust material.
fn tls_client(
    connection_info: redis::ConnectionInfo,
    certificates: Option<&redis::TlsCertificates>,
    phase: &'static str,
) -> Result<redis::Client, BackendError> {
    let client = match certificates {
        Some(certificates) => redis::Client::build_with_tls(connection_info, certificates.clone()),
        None => redis::Client::open(connection_info),
    };
    client.map_err(|error| map_valkey_error(phase, &error))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
pub(super) mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };

    use super::{
        BackendError, MAX_IDLE_TRANSACTION_CONNECTIONS, VALKEY_TIMEOUT, ValkeyConnection, ValkeyTimeouts,
        command_error, map_valkey_error,
    };
    use crate::token_rate_limit::config::{
        BackendConfig, BackendKind, BackendOnFailure, BackendTopology, SentinelConfig,
    };

    fn valkey_url() -> Option<String> {
        std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL").ok()
    }

    #[derive(Clone, Copy)]
    enum TlsServerRole {
        Sentinel { primary_port: u16 },
        SentinelBadAddress,
        SentinelError,
        DropReads,
        Primary,
        PrimaryError,
        PrimaryReplica,
    }

    struct TlsRespServer {
        port: u16,
        accepted: Arc<AtomicUsize>,
        shutdown: Option<tokio::sync::oneshot::Sender<()>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for TlsRespServer {
        fn drop(&mut self) {
            let _ = self.shutdown.take().map(|sender| sender.send(()));
            self.task.abort();
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "TLS test material builds a CA and server certificate chain"
    )]
    fn tls_test_material() -> (redis::TlsCertificates, Arc<rustls::ServerConfig>) {
        use rcgen::{BasicConstraints, CertificateParams, IsCa, Issuer, KeyPair, SanType};
        use rustls_pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject as _};

        praxis_tls::provider::install();
        let ca_key = KeyPair::generate().expect("generate test CA key");
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("test CA params");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).expect("self-sign test CA");
        let issuer = Issuer::from_params(&ca_params, &ca_key);

        let server_key = KeyPair::generate().expect("generate test server key");
        let mut server_params = CertificateParams::new(vec!["localhost".to_owned()]).expect("test server params");
        server_params
            .subject_alt_names
            .push(SanType::IpAddress(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)));
        let server_cert = server_params
            .signed_by(&server_key, &issuer)
            .expect("sign test server certificate");
        let cert_pem = server_cert.pem();
        let key_pem = server_key.serialize_pem();
        let server_certs = CertificateDer::pem_slice_iter(cert_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("parse test server certificate");
        let server_key = PrivateKeyDer::from_pem_slice(key_pem.as_bytes()).expect("parse test server key");
        let server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(server_certs, server_key)
            .expect("build test TLS server config");
        let certificates = redis::TlsCertificates {
            client_tls: None,
            root_cert: Some(ca_cert.pem().into_bytes()),
        };
        (certificates, Arc::new(server_config))
    }

    async fn read_crlf_line<R: tokio::io::AsyncRead + Unpin>(reader: &mut R) -> Option<Vec<u8>> {
        let mut line = Vec::new();
        let mut byte = [0_u8; 1];
        loop {
            tokio::io::AsyncReadExt::read_exact(reader, &mut byte).await.ok()?;
            line.push(byte[0]);
            if line.ends_with(b"\r\n") {
                line.truncate(line.len().saturating_sub(2));
                return Some(line);
            }
            if line.len() > 4096 {
                return None;
            }
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "RESP test parser handles each bounded request field"
    )]
    /// Read one RESP request from a controlled loopback test client.
    pub(in crate::token_rate_limit::valkey) async fn read_resp_command<R: tokio::io::AsyncRead + Unpin>(
        reader: &mut R,
    ) -> Option<Vec<Vec<u8>>> {
        let mut prefix = [0_u8; 1];
        tokio::io::AsyncReadExt::read_exact(reader, &mut prefix).await.ok()?;
        if prefix[0] != b'*' {
            return None;
        }
        let count = String::from_utf8(read_crlf_line(reader).await?)
            .ok()?
            .parse::<usize>()
            .ok()?;
        let mut command = Vec::with_capacity(count);
        for _ in 0..count {
            tokio::io::AsyncReadExt::read_exact(reader, &mut prefix).await.ok()?;
            if prefix[0] != b'$' {
                return None;
            }
            let length = String::from_utf8(read_crlf_line(reader).await?)
                .ok()?
                .parse::<usize>()
                .ok()?;
            let mut value = vec![0_u8; length];
            tokio::io::AsyncReadExt::read_exact(reader, &mut value).await.ok()?;
            let mut crlf = [0_u8; 2];
            tokio::io::AsyncReadExt::read_exact(reader, &mut crlf).await.ok()?;
            if crlf != *b"\r\n" {
                return None;
            }
            command.push(value);
        }
        Some(command)
    }

    fn resp_bulk(value: &str) -> Vec<u8> {
        format!("${}\r\n{}\r\n", value.len(), value).into_bytes()
    }

    fn resp_sentinel_masters(primary_port: u16) -> Vec<u8> {
        let fields = [
            ("name", "mymaster"),
            ("ip", "127.0.0.1"),
            ("port", &primary_port.to_string()),
            ("flags", "master"),
        ];
        let mut response = b"*1\r\n*8\r\n".to_vec();
        for (key, value) in fields {
            response.extend(resp_bulk(key));
            response.extend(resp_bulk(value));
        }
        response
    }

    fn resp_for(command: &[Vec<u8>], role: TlsServerRole) -> Vec<u8> {
        let name = command
            .first()
            .map(|value| String::from_utf8_lossy(value).to_ascii_uppercase())
            .unwrap_or_default();
        let subcommand = command
            .get(1)
            .map(|value| String::from_utf8_lossy(value).to_ascii_uppercase())
            .unwrap_or_default();
        match (role, name.as_str()) {
            (TlsServerRole::Sentinel { primary_port }, "SENTINEL") if subcommand == "MASTERS" => {
                resp_sentinel_masters(primary_port)
            },
            (TlsServerRole::Sentinel { primary_port }, "SENTINEL") => {
                let mut response = b"*2\r\n".to_vec();
                response.extend(resp_bulk("127.0.0.1"));
                response.extend(resp_bulk(&primary_port.to_string()));
                response
            },
            (TlsServerRole::Sentinel { .. }, "ROLE") => b"*2\r\n$8\r\nsentinel\r\n*0\r\n".to_vec(),
            (TlsServerRole::SentinelBadAddress, "SENTINEL") => b"*2\r\n$1\r\n[\r\n$4\r\n6379\r\n".to_vec(),
            (TlsServerRole::SentinelError, "SENTINEL") => b"-ERR sentinel unavailable\r\n".to_vec(),
            (TlsServerRole::Primary, "ROLE") => b"*3\r\n$6\r\nmaster\r\n:0\r\n*0\r\n".to_vec(),
            (TlsServerRole::PrimaryError, "ROLE") => b"-ERR role unavailable\r\n".to_vec(),
            (TlsServerRole::PrimaryReplica, "ROLE") => {
                b"*5\r\n$5\r\nslave\r\n$9\r\n127.0.0.1\r\n:6379\r\n$9\r\nconnected\r\n:0\r\n".to_vec()
            },
            (TlsServerRole::DropReads, "GET" | "SCAN") => Vec::new(),
            (_, "PING") => b"+PONG\r\n".to_vec(),
            _ => b"+OK\r\n".to_vec(),
        }
    }

    async fn serve_tls_resp_connection(
        mut stream: tokio_rustls::server::TlsStream<tokio::net::TcpStream>,
        role: TlsServerRole,
    ) {
        while let Some(command) = read_resp_command(&mut stream).await {
            let response = resp_for(&command, role);
            if tokio::io::AsyncWriteExt::write_all(&mut stream, &response)
                .await
                .is_err()
            {
                return;
            }
            drop(tokio::io::AsyncWriteExt::flush(&mut stream).await);
        }
    }

    async fn serve_plain_resp_connection(mut stream: tokio::net::TcpStream, role: TlsServerRole) {
        while let Some(command) = read_resp_command(&mut stream).await {
            let response = resp_for(&command, role);
            if response.is_empty() {
                return;
            }
            if tokio::io::AsyncWriteExt::write_all(&mut stream, &response)
                .await
                .is_err()
            {
                return;
            }
            drop(tokio::io::AsyncWriteExt::flush(&mut stream).await);
        }
    }

    fn start_tls_resp_server(server_config: Arc<rustls::ServerConfig>, role: TlsServerRole) -> TlsRespServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind TLS test server");
        let port = listener.local_addr().expect("TLS test server address").port();
        listener.set_nonblocking(true).expect("set TLS listener nonblocking");
        let listener = tokio::net::TcpListener::from_std(listener).expect("adopt TLS listener");
        let acceptor = tokio_rustls::TlsAcceptor::from(server_config);
        let (shutdown, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_for_task = Arc::clone(&accepted);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _peer)) = accepted else { break };
                        accepted_for_task.fetch_add(1, Ordering::Relaxed);
                        let Ok(stream) = acceptor.accept(stream).await else { continue };
                        tokio::spawn(serve_tls_resp_connection(stream, role));
                    },
                }
            }
        });
        TlsRespServer {
            port,
            accepted,
            shutdown: Some(shutdown),
            task,
        }
    }

    fn start_plain_resp_server(role: TlsServerRole) -> TlsRespServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind plain test server");
        let port = listener.local_addr().expect("plain test server address").port();
        listener.set_nonblocking(true).expect("set plain listener nonblocking");
        let listener = tokio::net::TcpListener::from_std(listener).expect("adopt plain listener");
        let (shutdown, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_for_task = Arc::clone(&accepted);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _peer)) = accepted else { break };
                        accepted_for_task.fetch_add(1, Ordering::Relaxed);
                        tokio::spawn(serve_plain_resp_connection(stream, role));
                    },
                }
            }
        });
        TlsRespServer {
            port,
            accepted,
            shutdown: Some(shutdown),
            task,
        }
    }

    fn start_hanging_plain_server() -> TlsRespServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind hanging test server");
        let port = listener.local_addr().expect("hanging test server address").port();
        listener
            .set_nonblocking(true)
            .expect("set hanging listener nonblocking");
        let listener = tokio::net::TcpListener::from_std(listener).expect("adopt hanging listener");
        let (shutdown, mut shutdown_rx) = tokio::sync::oneshot::channel();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_for_task = Arc::clone(&accepted);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _peer)) = accepted else { break };
                        accepted_for_task.fetch_add(1, Ordering::Relaxed);
                        tokio::spawn(async move {
                            let _stream = stream;
                            std::future::pending::<()>().await;
                        });
                    },
                }
            }
        });
        TlsRespServer {
            port,
            accepted,
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn shared_read_burst(connection: &ValkeyConnection) {
        let barrier = Arc::new(tokio::sync::Barrier::new(128));
        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..128 {
            let connection = connection.clone();
            let barrier = Arc::clone(&barrier);
            tasks.spawn(async move {
                barrier.wait().await;
                let mut pipe = redis::pipe();
                pipe.cmd("PING");
                let reply: (String,) = connection
                    .read_pipeline(&pipe)
                    .await
                    .expect("shared burst must complete");
                assert_eq!(reply.0, "PONG", "every coalesced reader must receive its reply");
            });
        }
        while let Some(task) = tasks.join_next().await {
            task.expect("every shared reader task must succeed");
        }
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "one concurrent fixture covers standalone and Sentinel cache epochs"
    )]
    async fn concurrent_shared_cold_opens_are_single_flight() {
        for use_sentinel in [false, true] {
            let primary = start_plain_resp_server(TlsServerRole::Primary);
            let sentinel = start_plain_resp_server(TlsServerRole::Sentinel {
                primary_port: primary.port,
            });
            let connection = if use_sentinel {
                ValkeyConnection::sentinel(
                    vec![format!("redis://127.0.0.1:{}", sentinel.port)],
                    "mymaster".into(),
                    None,
                    None,
                    None,
                    None,
                    false,
                    ValkeyTimeouts::default(),
                    "shared_kv",
                )
                .unwrap()
            } else {
                ValkeyConnection::new(format!("redis://127.0.0.1:{}", primary.port)).unwrap()
            };
            for epoch in 0..2 {
                shared_read_burst(&connection).await;
                let per_epoch = if use_sentinel { 2 } else { 1 };
                assert_eq!(
                    primary.accepted.load(Ordering::Relaxed),
                    (epoch + 1) * per_epoch,
                    "a shared epoch needs one data socket plus at most one native Sentinel ROLE probe"
                );
                assert_eq!(
                    sentinel.accepted.load(Ordering::Relaxed),
                    if use_sentinel { epoch + 1 } else { 0 },
                    "Sentinel discovery must also stay coalesced across the burst"
                );
                let generation = connection.generation.load(Ordering::Acquire);
                connection.invalidate(generation, true).await;
            }
        }
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "one owned gate verifies cold wait bounds and warm cache bypass"
    )]
    async fn shared_open_wait_is_bounded_and_cache_hits_bypass_it() {
        let primary = start_plain_resp_server(TlsServerRole::Primary);
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            format!("redis://127.0.0.1:{}", primary.port),
            ValkeyTimeouts {
                connect: Duration::from_millis(100),
                ..ValkeyTimeouts::default()
            },
            "shared_kv",
        )
        .unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let guard = connection.shared_open.lock().await;
        let result = tokio::time::timeout(Duration::from_secs(1), connection.read_pipeline::<(String,)>(&pipe)).await;
        assert!(
            matches!(result, Ok(Err(BackendError::Unavailable(message))) if message.contains("shared connection open timeout")),
            "waiting for another opener must honor the existing connection deadline"
        );
        assert_eq!(
            primary.accepted.load(Ordering::Relaxed),
            0,
            "a blocked opener must not create a data socket"
        );
        drop(guard);
        let reply: (String,) = connection
            .read_pipeline(&pipe)
            .await
            .expect("released gate must permit recovery");
        assert_eq!(reply.0, "PONG", "a later opener must recover after lock timeout");
        let _guard = connection.shared_open.lock().await;
        tokio::time::timeout(Duration::from_secs(1), shared_read_burst(&connection))
            .await
            .expect("warm cache hits must not wait for the cold-open gate");
        assert_eq!(
            primary.accepted.load(Ordering::Relaxed),
            1,
            "warm readers reuse the existing shared socket"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "read and mutation error policies share one transport fault fixture"
    )]
    async fn accounting_reads_are_retryable_but_mutation_pipelines_remain_unconfirmed() {
        let server = start_plain_resp_server(TlsServerRole::DropReads);
        let connection = ValkeyConnection::new(format!("redis://127.0.0.1:{}", server.port)).unwrap();
        let mut read = redis::pipe();
        read.cmd("GET").arg("generation");
        let result: Result<(Option<String>,), _> = connection.read_pipeline(&read).await;
        assert!(
            matches!(result, Err(BackendError::Unavailable(_))),
            "a lost read reply must remain retryable before any accounting mutation"
        );
        assert!(
            connection.connection.lock().unwrap().is_none(),
            "failed read socket must be fenced"
        );
        let result: Result<(Option<String>,), _> = connection.pipeline(&read).await;
        assert!(
            matches!(result, Err(BackendError::Unconfirmed(_))),
            "the mutation-capable API must conservatively retain ambiguous outcomes"
        );

        for result in [
            super::super::read_accounting_marker(&connection, "namespace", "marker")
                .await
                .map(|_| ()),
            super::super::ensure_accounting_generation(&connection, "namespace").await,
            super::super::scan_namespace_state(&connection, "namespace")
                .await
                .map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(BackendError::Unavailable(_))),
                "marker, generation, and namespace scans must all use the read-only error policy"
            );
        }
    }

    #[tokio::test]
    async fn unavailable_sentinel_fails_closed_without_caching_a_connection() {
        let connection = ValkeyConnection::sentinel(
            vec!["redis://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts {
                connect: Duration::from_millis(20),
                operation: Duration::from_millis(20),
                discovery: Duration::from_millis(20),
            },
            "valkey",
        )
        .unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("GET").arg("praxis:test:sentinel-unavailable");
        let result: Result<(Option<String>,), BackendError> = connection.pipeline(&pipe).await;
        assert!(result.is_err(), "unavailable Sentinel must fail closed");
        assert!(connection.connection.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn invalidating_without_a_matching_shared_generation_is_a_noop() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .unwrap();
        connection.invalidate(7, false).await;
        assert_eq!(connection.generation.load(Ordering::Acquire), 0);
        assert_eq!(connection.transactions.generation.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn topology_generation_is_independent_from_shared_connection_generation() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .unwrap();
        connection.invalidate(0, true).await;
        assert_eq!(connection.generation.load(Ordering::Acquire), 0);
        assert_eq!(connection.topology_generation.load(Ordering::Acquire), 1);
    }

    #[test]
    fn only_connection_and_topology_errors_advance_discovery_generation() {
        let ordinary = redis::RedisError::from((redis::ServerErrorKind::ResponseError.into(), "WRONGTYPE"));
        let readonly = redis::RedisError::from((redis::ServerErrorKind::ReadOnly.into(), "READONLY"));
        let dropped = redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
        assert!(!super::advances_topology_generation(&ordinary));
        assert!(super::advances_topology_generation(&readonly));
        assert!(super::advances_topology_generation(&dropped));
    }

    #[test]
    fn transaction_read_failures_are_retryable_and_still_fence_topology() {
        let connection = ValkeyConnection::new("redis://127.0.0.1:1".into()).unwrap();
        let dropped = redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
        assert!(
            matches!(
                connection.transaction_read_error(&dropped),
                BackendError::Unavailable(_)
            ),
            "an undispatched settlement remains safe to retry after a read disconnect"
        );
        assert_eq!(
            connection.topology_generation.load(Ordering::Acquire),
            1,
            "read disconnect must fence topology"
        );
        assert_eq!(
            connection.generation.load(Ordering::Acquire),
            1,
            "read disconnect must fence the shared socket"
        );
    }

    #[test]
    fn topology_transaction_errors_fence_shared_connections() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .unwrap();
        let readonly = redis::RedisError::from((redis::ServerErrorKind::ReadOnly.into(), "READONLY"));
        let error = connection.transaction_command_error(&readonly);
        assert!(matches!(error, BackendError::Unconfirmed(_)));
        assert_eq!(connection.generation.load(Ordering::Acquire), 1);
        assert_eq!(connection.topology_generation.load(Ordering::Acquire), 1);
        assert!(connection.connection.lock().unwrap().is_none());
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the failed checkout result is inspected before the test connection is dropped"
    )]
    async fn closed_transaction_pool_fails_before_opening_a_socket() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .unwrap();
        connection.transactions.semaphore.close();
        let result = connection.transaction().await;
        assert!(matches!(result, Err(BackendError::Unavailable(message)) if message.contains("pool closed")));
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the bounded pool timeout result is inspected before the test connection is dropped"
    )]
    async fn exhausted_transaction_pool_returns_within_operation_deadline() {
        let mut connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts {
                connect: Duration::from_millis(5),
                operation: Duration::from_millis(5),
                discovery: Duration::from_millis(5),
            },
            "valkey",
        )
        .unwrap();
        connection.transactions = Arc::new(super::TransactionConnections {
            idle: tokio::sync::Mutex::new(Vec::new()),
            semaphore: Arc::new(tokio::sync::Semaphore::new(0)),
            generation: std::sync::atomic::AtomicU64::new(0),
        });
        let started = Instant::now();
        let result = connection.transaction().await;
        assert!(matches!(result, Err(BackendError::Unavailable(message)) if message.contains("pool exhausted")));
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn expired_connection_deadlines_fail_before_transport_work() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .unwrap();
        let config = redis::AsyncConnectionConfig::new();
        let open = connection.open_until(Instant::now() - Duration::from_millis(1)).await;
        assert!(matches!(open, Err(BackendError::Unavailable(message)) if message.contains("deadline elapsed")));
        let client = redis::Client::open("redis://127.0.0.1:1").unwrap();
        let primary = connection
            .open_primary_until(&client, Instant::now() - Duration::from_millis(1), &config)
            .await;
        assert!(matches!(primary, Err(BackendError::Unavailable(message)) if message.contains("deadline elapsed")));
        let invalidation = connection
            .invalidate_primary_until(0, Instant::now() - Duration::from_millis(1))
            .await;
        assert!(
            matches!(invalidation, Err(BackendError::Unavailable(message)) if message.contains("cache lock timeout"))
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "backend construction coverage keeps topology, credential, and validation cases together"
    )]
    fn backend_construction_covers_wrapper_credentials_and_validation_errors() {
        let standalone = BackendConfig {
            kind: BackendKind::Valkey,
            topology: Some(BackendTopology::Standalone("redis://127.0.0.1:6379".into())),
            ..BackendConfig::default()
        };
        let connection = ValkeyConnection::from_backend_config(&standalone, "shared_kv")
            .expect("standalone backend construction should use the process lookup wrapper");
        assert_eq!(connection.backend_name(), "shared_kv");

        let redis = BackendConfig {
            kind: BackendKind::Redis,
            topology: Some(BackendTopology::Standalone("redis://127.0.0.1:6379".into())),
            ..BackendConfig::default()
        };
        let redis_connection = ValkeyConnection::from_backend_config(&redis, "shared_kv")
            .expect("Redis backend construction should use the shared KV telemetry family");
        assert_eq!(redis_connection.backend_name(), "shared_kv");

        let missing_topology = ValkeyConnection::from_backend_config(&BackendConfig::default(), "valkey");
        assert!(missing_topology.is_err(), "network backends require a topology");

        let credentialed = ValkeyConnection::sentinel(
            vec!["rediss://127.0.0.1:26379".into()],
            "mymaster".into(),
            Some("sentinel-user".into()),
            Some("sentinel-password".into()),
            Some("data-user".into()),
            Some("data-password".into()),
            true,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("credentialed TLS Sentinel construction should succeed");
        assert!(matches!(
            credentialed.source.as_ref(),
            super::ConnectionSource::SentinelTls { .. }
        ));

        assert!(
            ValkeyConnection::sentinel(
                Vec::new(),
                "mymaster".into(),
                None,
                None,
                None,
                None,
                false,
                ValkeyTimeouts::default(),
                "valkey",
            )
            .is_err()
        );
        assert!(
            ValkeyConnection::sentinel(
                vec!["redis://127.0.0.1:26379".into()],
                String::new(),
                None,
                None,
                None,
                None,
                false,
                ValkeyTimeouts::default(),
                "valkey",
            )
            .is_err()
        );
        assert!(
            ValkeyConnection::sentinel(
                vec!["redis://".into()],
                "mymaster".into(),
                None,
                None,
                None,
                None,
                false,
                ValkeyTimeouts::default(),
                "valkey",
            )
            .is_err()
        );
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "primary_action consumes the owned guard and returns a wait without retaining it"
    )]
    async fn primary_actions_cover_wait_and_discarded_discovery_results() {
        let cache = Arc::new(tokio::sync::Mutex::new(super::PrimaryCache::<redis::Client>::new()));
        {
            let mut state = cache.lock().await;
            assert_eq!(state.begin_discovery(0), super::DiscoveryStart::Started);
        }
        let cache_for_action = Arc::clone(&cache);
        let action_guard = cache_for_action.lock_owned().await;
        let action = super::primary_action(action_guard, 0);
        assert!(matches!(action, super::PrimaryAction::Wait));

        let mut lease = super::DiscoveryLease {
            cache: Some(cache.lock_owned().await),
            generation: 0,
        };
        lease.cache.as_mut().expect("test lease owns the cache").invalidate(0);
        let primary = redis::Client::open("redis://127.0.0.1:1").expect("test client should parse");
        assert_eq!(lease.publish(primary), super::PublishResult::Discarded);
    }

    #[test]
    fn failed_failover_record_is_emitted_once() {
        let mut pending = true;
        super::record_failed_sentinel_failover("valkey", &mut pending);
        assert!(!pending);
        super::record_failed_sentinel_failover("valkey", &mut pending);
        assert!(!pending);
    }

    #[test]
    fn tls_sentinel_accepts_the_valkey_tls_scheme() {
        let connection = ValkeyConnection::sentinel(
            vec!["valkeys://127.0.0.1:26379".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts::default(),
            "valkey",
        );
        assert!(connection.is_ok(), "valkeys:// is a valid TLS Sentinel endpoint");
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "secure TLS discovery test covers Sentinel, primary ROLE, and command traffic"
    )]
    async fn secure_tls_sentinel_discovers_and_uses_a_verified_primary() {
        let (certificates, server_config) = tls_test_material();
        let primary = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::Primary);
        let sentinel = start_tls_resp_server(
            server_config,
            TlsServerRole::Sentinel {
                primary_port: primary.port,
            },
        );
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_secs(5),
            },
            "valkey",
        )
        .expect("secure TLS Sentinel configuration should build");
        connection
            .install_test_tls_certificates(certificates)
            .expect("install test trust material");

        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert_eq!(result.expect("secure TLS Sentinel request should succeed").0, "PONG");

        *connection.connection.lock().unwrap() = None;
        let cached: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert_eq!(cached.expect("cached TLS primary request should succeed").0, "PONG");

        {
            let mut cache = connection.sentinel_primary.lock().await;
            assert!(
                cache.invalidate(0),
                "the discovered primary should belong to generation zero"
            );
            assert_eq!(cache.begin_discovery(0), super::DiscoveryStart::Started);
            cache.publish(
                0,
                redis::Client::open("redis://127.0.0.1:1").expect("test client should parse"),
            );
        }
        *connection.connection.lock().unwrap() = None;
        let recovered: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert_eq!(
            recovered
                .expect("TLS discovery should recover after a cached primary failure")
                .0,
            "PONG"
        );
    }

    #[tokio::test]
    async fn secure_tls_sentinel_fails_closed_when_endpoint_is_unavailable() {
        let connection = ValkeyConnection::sentinel(
            vec!["rediss://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_millis(20),
                operation: Duration::from_millis(20),
                discovery: Duration::from_millis(30),
            },
            "valkey",
        )
        .expect("secure TLS Sentinel configuration should build");
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert!(result.is_err(), "unavailable secure TLS Sentinel must fail closed");
        assert!(connection.connection.lock().unwrap().is_none());
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "plain Sentinel coverage exercises credentialed discovery, cache hits, and transaction checkout"
    )]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the transaction is intentionally consumed by finish after protocol cleanup"
    )]
    async fn plain_sentinel_discovers_a_primary_and_reuses_it() {
        let primary = start_plain_resp_server(TlsServerRole::Primary);
        let sentinel = start_plain_resp_server(TlsServerRole::Sentinel {
            primary_port: primary.port,
        });
        let connection = ValkeyConnection::sentinel(
            vec![format!("redis://127.0.0.1:{}", sentinel.port)],
            "mymaster".into(),
            Some("sentinel-user".into()),
            Some("sentinel-password".into()),
            Some("data-user".into()),
            Some("data-password".into()),
            false,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_secs(1),
            },
            "valkey",
        )
        .expect("plain Sentinel configuration should build");
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert_eq!(result.expect("plain Sentinel request should succeed").0, "PONG");

        drop(sentinel);
        *connection.connection.lock().unwrap() = None;
        let cached: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert_eq!(cached.expect("cached plain primary request should succeed").0, "PONG");

        let mut transaction = connection
            .transaction()
            .await
            .expect("plain Sentinel transaction checkout should succeed");
        redis::cmd("WATCH")
            .arg("praxis:test:plain-sentinel")
            .exec_async(transaction.inner())
            .await
            .expect("plain Sentinel transaction should accept WATCH");
        super::unwatch(transaction.inner())
            .await
            .expect("plain Sentinel transaction should accept UNWATCH");
        transaction.finish(VALKEY_TIMEOUT).await;
        let second = connection
            .transaction()
            .await
            .expect("plain Sentinel transaction should reuse its pooled socket");
        drop(second);
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "silent and healthy endpoints share one bounded discovery fixture"
    )]
    async fn plain_sentinel_skips_a_connected_but_silent_first_endpoint() {
        let hanging = start_hanging_plain_server();
        let primary = start_plain_resp_server(TlsServerRole::Primary);
        let sentinel = start_plain_resp_server(TlsServerRole::Sentinel {
            primary_port: primary.port,
        });
        let connection = ValkeyConnection::sentinel(
            vec![
                format!("redis://127.0.0.1:{}", hanging.port),
                format!("redis://127.0.0.1:{}", sentinel.port),
            ],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts {
                connect: Duration::from_millis(50),
                operation: Duration::from_millis(50),
                discovery: Duration::from_secs(1),
            },
            "shared_kv",
        )
        .unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: (String,) = connection
            .read_pipeline(&pipe)
            .await
            .expect("healthy later Sentinel must be tried");
        assert_eq!(
            result.0, "PONG",
            "a silent first endpoint must not consume the complete deadline"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "TLS fallback test owns trust material and two independent Sentinels"
    )]
    async fn tls_sentinel_skips_stalled_and_malformed_endpoints() {
        let (certificates, server_config) = tls_test_material();
        let hanging = start_hanging_plain_server();
        let primary = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::Primary);
        let malformed = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::SentinelBadAddress);
        let sentinel = start_tls_resp_server(
            server_config,
            TlsServerRole::Sentinel {
                primary_port: primary.port,
            },
        );
        let mut connection = ValkeyConnection::sentinel(
            vec![
                format!("rediss://localhost:{}", hanging.port),
                format!("rediss://localhost:{}", malformed.port),
                format!("rediss://localhost:{}", sentinel.port),
            ],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_secs(5),
            },
            "shared_kv",
        )
        .unwrap();
        connection.install_test_tls_certificates(certificates).unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: (String,) = connection
            .read_pipeline(&pipe)
            .await
            .expect("valid later TLS Sentinel must be tried");
        assert_eq!(
            result.0, "PONG",
            "a malformed first address must not starve healthy Sentinels"
        );
    }

    #[tokio::test]
    async fn plain_sentinel_endpoint_timeout_covers_a_connected_but_silent_sentinel() {
        let hanging = start_hanging_plain_server();
        let connection = ValkeyConnection::sentinel(
            vec![format!("redis://127.0.0.1:{}", hanging.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts {
                connect: Duration::from_millis(5),
                operation: Duration::from_millis(5),
                discovery: Duration::from_millis(50),
            },
            "valkey",
        )
        .expect("silent Sentinel configuration should build");
        let config = match connection.source.as_ref() {
            super::ConnectionSource::Sentinel { config } => Arc::clone(config),
            _ => panic!("expected plain Sentinel source"),
        };
        let result = connection
            .discover_plain_primary(&config, Instant::now() + Duration::from_millis(50))
            .await;
        assert!(matches!(result, Err(BackendError::Unavailable(message)) if message.contains("endpoint timeout")));
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "TLS discovery failure coverage exercises malformed address, command, data, and role responses"
    )]
    async fn secure_tls_sentinel_rejects_malformed_and_non_primary_responses() {
        let (certificates, server_config) = tls_test_material();
        let sentinel = start_tls_resp_server(server_config, TlsServerRole::SentinelBadAddress);
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_millis(30),
            },
            "valkey",
        )
        .expect("malformed TLS Sentinel configuration should build");
        connection
            .install_test_tls_certificates(certificates)
            .expect("install test trust material");
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        assert!(connection.pipeline::<(String,)>(&pipe).await.is_err());

        let (certificates, server_config) = tls_test_material();
        let sentinel = start_tls_resp_server(server_config, TlsServerRole::SentinelError);
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_millis(30),
            },
            "valkey",
        )
        .expect("TLS Sentinel error configuration should build");
        connection
            .install_test_tls_certificates(certificates)
            .expect("install test trust material");
        assert!(connection.pipeline::<(String,)>(&pipe).await.is_err());

        let (certificates, server_config) = tls_test_material();
        let primary = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::PrimaryReplica);
        let sentinel = start_tls_resp_server(
            server_config,
            TlsServerRole::Sentinel {
                primary_port: primary.port,
            },
        );
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_millis(30),
            },
            "valkey",
        )
        .expect("replica TLS primary configuration should build");
        connection
            .install_test_tls_certificates(certificates)
            .expect("install test trust material");
        assert!(connection.pipeline::<(String,)>(&pipe).await.is_err());

        let (certificates, server_config) = tls_test_material();
        let primary = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::PrimaryError);
        let sentinel = start_tls_resp_server(
            server_config,
            TlsServerRole::Sentinel {
                primary_port: primary.port,
            },
        );
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_millis(30),
            },
            "valkey",
        )
        .expect("role-error TLS primary configuration should build");
        connection
            .install_test_tls_certificates(certificates)
            .expect("install test trust material");
        assert!(connection.pipeline::<(String,)>(&pipe).await.is_err());
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "certificate rejection test owns two independent TLS authorities and server guards"
    )]
    async fn secure_tls_sentinel_rejects_an_untrusted_root() {
        let (_trusted_by_server, server_config) = tls_test_material();
        let (wrong_root, _) = tls_test_material();
        let primary = start_tls_resp_server(Arc::clone(&server_config), TlsServerRole::Primary);
        let sentinel = start_tls_resp_server(
            server_config,
            TlsServerRole::Sentinel {
                primary_port: primary.port,
            },
        );
        let mut connection = ValkeyConnection::sentinel(
            vec![format!("rediss://localhost:{}", sentinel.port)],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts {
                connect: Duration::from_millis(100),
                operation: Duration::from_millis(100),
                discovery: Duration::from_millis(200),
            },
            "valkey",
        )
        .expect("secure TLS Sentinel configuration should build");
        connection
            .install_test_tls_certificates(wrong_root)
            .expect("install wrong test trust material");
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert!(result.is_err(), "an untrusted root must fail secure TLS discovery");
        assert!(connection.connection.lock().unwrap().is_none());
    }

    #[test]
    fn tls_client_builds_with_default_and_custom_trust_material() {
        let info = redis::IntoConnectionInfo::into_connection_info("rediss://localhost:1")
            .expect("TLS test connection info should parse");
        assert!(super::tls_client(info.clone(), None, "TLS test").is_ok());
        let (certificates, _server_config) = tls_test_material();
        assert!(super::tls_client(info, Some(&certificates), "TLS test").is_ok());
    }

    #[tokio::test]
    async fn primary_open_errors_and_cache_invalidation_success_are_bounded() {
        let connection = ValkeyConnection::standalone_with_timeouts_and_name(
            "redis://127.0.0.1:1".into(),
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("test connection should build");
        let config = redis::AsyncConnectionConfig::new();
        let client = redis::Client::open("redis://127.0.0.1:1").expect("test client should parse");
        let open = connection
            .open_primary_until(&client, Instant::now() + Duration::from_millis(20), &config)
            .await;
        assert!(open.is_err(), "closed primary must fail without hanging");
        let invalidation = connection
            .invalidate_primary_until(0, Instant::now() + Duration::from_millis(20))
            .await;
        assert!(invalidation.is_ok(), "uncontended cache invalidation should succeed");
    }

    #[tokio::test]
    async fn abort_retry_pause_stays_inside_the_deadline() {
        let mut retry = super::AbortRetry::start(Instant::now() + Duration::from_millis(50));
        retry.pause().await.expect("a live deadline should permit one pause");
        assert!(retry.remaining().is_ok());
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "plain and TLS cache-lock tests verify the same bounded timeout contract"
    )]
    async fn sentinel_cache_lock_timeout_is_bounded_for_plain_and_tls_sources() {
        let plain = ValkeyConnection::sentinel(
            vec!["redis://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("plain Sentinel source should build");
        let plain_config = match plain.source.as_ref() {
            super::ConnectionSource::Sentinel { config } => Arc::clone(config),
            _ => panic!("expected plain Sentinel source"),
        };
        let plain_guard = plain.sentinel_primary.lock().await;
        let plain_result = plain
            .open_sentinel_until(
                &plain_config,
                Instant::now() + Duration::from_millis(5),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(
            matches!(plain_result, Err(BackendError::Unavailable(message)) if message.contains("cache lock timeout"))
        );
        drop(plain_guard);

        let tls = ValkeyConnection::sentinel(
            vec!["rediss://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("TLS Sentinel source should build");
        let tls_config = match tls.source.as_ref() {
            super::ConnectionSource::SentinelTls { config } => Arc::clone(config),
            _ => panic!("expected TLS Sentinel source"),
        };
        let _tls_guard = tls.sentinel_primary.lock().await;
        let tls_result = tls
            .open_tls_sentinel(
                &tls_config,
                Instant::now() + Duration::from_millis(5),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(
            matches!(tls_result, Err(BackendError::Unavailable(message)) if message.contains("cache lock timeout"))
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "plain and TLS cached-primary failures share one bounded recovery contract"
    )]
    async fn sentinel_wait_and_cached_primary_failures_end_with_bounded_errors() {
        let plain = ValkeyConnection::sentinel(
            vec!["redis://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("plain Sentinel source should build");
        let plain_config = match plain.source.as_ref() {
            super::ConnectionSource::Sentinel { config } => Arc::clone(config),
            _ => panic!("expected plain Sentinel source"),
        };
        plain.sentinel_primary.lock().await.begin_discovery(0);
        let plain_wait = plain
            .open_sentinel_until(
                &plain_config,
                Instant::now() + Duration::from_millis(5),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(matches!(plain_wait, Err(BackendError::Unavailable(message)) if message.contains("discovery timeout")));

        let mut plain_cache = plain.sentinel_primary.lock().await;
        assert!(plain_cache.invalidate(0));
        plain_cache.begin_discovery(0);
        plain_cache.publish(
            0,
            redis::Client::open("redis://127.0.0.1:1").expect("test client should parse"),
        );
        drop(plain_cache);
        let plain_failure = plain
            .open_sentinel_until(
                &plain_config,
                Instant::now() + Duration::from_millis(30),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(plain_failure.is_err());

        let tls = ValkeyConnection::sentinel(
            vec!["rediss://127.0.0.1:1".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            true,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("TLS Sentinel source should build");
        let tls_config = match tls.source.as_ref() {
            super::ConnectionSource::SentinelTls { config } => Arc::clone(config),
            _ => panic!("expected TLS Sentinel source"),
        };
        tls.sentinel_primary.lock().await.begin_discovery(0);
        let tls_wait = tls
            .open_tls_sentinel(
                &tls_config,
                Instant::now() + Duration::from_millis(5),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(matches!(tls_wait, Err(BackendError::Unavailable(message)) if message.contains("discovery timeout")));

        let mut tls_cache = tls.sentinel_primary.lock().await;
        assert!(tls_cache.invalidate(0));
        tls_cache.begin_discovery(0);
        tls_cache.publish(
            0,
            redis::Client::open("redis://127.0.0.1:1").expect("test client should parse"),
        );
        drop(tls_cache);
        let tls_failure = tls
            .open_tls_sentinel(
                &tls_config,
                Instant::now() + Duration::from_millis(30),
                &redis::AsyncConnectionConfig::new(),
            )
            .await;
        assert!(tls_failure.is_err());
    }

    #[test]
    fn expanded_tls_sentinel_endpoint_selects_the_tls_source() {
        let backend = BackendConfig {
            kind: BackendKind::Valkey,
            topology: Some(BackendTopology::Sentinel(Box::new(SentinelConfig {
                endpoints: vec!["${TOKEN_RATE_LIMIT_SENTINEL_ENDPOINT}".into()],
                service_name: "mymaster".into(),
                sentinel_username: None,
                sentinel_password: None,
                redis_username: None,
                redis_password: None,
                discovery_timeout: Duration::from_secs(2),
            }))),
            namespace: None,
            on_failure: BackendOnFailure::Closed,
            connect_timeout: VALKEY_TIMEOUT,
            operation_timeout: VALKEY_TIMEOUT,
        };
        let connection = ValkeyConnection::from_backend_config_with(&backend, "valkey", |name| {
            assert_eq!(name, "TOKEN_RATE_LIMIT_SENTINEL_ENDPOINT");
            Ok("rediss://127.0.0.1:26379".into())
        })
        .expect("expanded TLS Sentinel endpoint should configure");
        assert!(matches!(
            connection.source.as_ref(),
            super::ConnectionSource::SentinelTls { .. }
        ));
    }

    #[test]
    fn expanded_mixed_sentinel_schemes_fail_closed() {
        let backend = BackendConfig {
            kind: BackendKind::Valkey,
            topology: Some(BackendTopology::Sentinel(Box::new(SentinelConfig {
                endpoints: vec!["${SENTINEL_TLS_ENDPOINT}".into(), "${SENTINEL_PLAIN_ENDPOINT}".into()],
                service_name: "mymaster".into(),
                sentinel_username: None,
                sentinel_password: None,
                redis_username: None,
                redis_password: None,
                discovery_timeout: Duration::from_secs(2),
            }))),
            namespace: None,
            on_failure: BackendOnFailure::Closed,
            connect_timeout: VALKEY_TIMEOUT,
            operation_timeout: VALKEY_TIMEOUT,
        };
        let result = ValkeyConnection::from_backend_config_with(&backend, "valkey", |name| match name {
            "SENTINEL_TLS_ENDPOINT" => Ok("rediss://127.0.0.1:26379".into()),
            "SENTINEL_PLAIN_ENDPOINT" => Ok("redis://127.0.0.1:26380".into()),
            _ => panic!("unexpected environment variable: {name}"),
        });
        let Err(error) = result else {
            panic!("mixed expanded Sentinel schemes must fail")
        };
        assert!(error.to_string().contains("scheme must use TLS"), "got: {error}");
    }

    #[test]
    fn plain_sentinel_endpoint_timeout_is_independently_bounded() {
        let timeouts = ValkeyTimeouts {
            connect: Duration::from_secs(5),
            operation: Duration::from_secs(5),
            discovery: Duration::from_secs(5),
        };
        assert_eq!(
            super::sentinel_endpoint_timeout(timeouts, Duration::from_secs(10)),
            VALKEY_TIMEOUT
        );
        assert_eq!(
            super::sentinel_endpoint_timeout(
                ValkeyTimeouts {
                    connect: Duration::from_millis(20),
                    operation: Duration::from_secs(5),
                    discovery: Duration::from_secs(5),
                },
                Duration::from_secs(10),
            ),
            Duration::from_millis(20)
        );
        assert_eq!(
            super::sentinel_endpoint_timeout(timeouts, Duration::from_millis(30)),
            Duration::from_millis(30)
        );
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the test intentionally holds the discovery lease until task cancellation"
    )]
    async fn cancelling_a_discovery_lease_resets_the_primary_cache() {
        let cache = Arc::new(tokio::sync::Mutex::new(super::PrimaryCache::<redis::Client>::new()));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_cache = Arc::clone(&cache);
        let task = tokio::spawn(async move {
            let action = super::primary_action(task_cache.lock_owned().await, 0);
            assert!(matches!(action, super::PrimaryAction::Discover(_)));
            started_tx.send(()).expect("test receiver should still be present");
            std::future::pending::<()>().await;
        });
        started_rx.await.expect("discovery should start");
        task.abort();
        drop(task.await);

        let cache = cache.lock().await;
        assert!(matches!(cache.lookup(0), super::Lookup::Miss));
        drop(cache);
    }

    #[tokio::test]
    async fn primary_cache_invalidation_respects_the_remaining_deadline() {
        let connection = ValkeyConnection::sentinel(
            vec!["redis://127.0.0.1:26379".into()],
            "mymaster".into(),
            None,
            None,
            None,
            None,
            false,
            ValkeyTimeouts::default(),
            "valkey",
        )
        .expect("test Sentinel configuration should build");
        let _cache = connection.sentinel_primary.lock().await;
        let result = connection
            .invalidate_primary_until(0, Instant::now() + Duration::from_millis(5))
            .await;
        assert!(matches!(
            result,
            Err(BackendError::Unavailable(message)) if message.contains("cache lock timeout")
        ));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one contract test verifies both Sentinel metric families and their labels"
    )]
    fn sentinel_telemetry_uses_only_bounded_labels() {
        assert_eq!(super::sentinel_discovery_phase(0, false), "initial");
        assert_eq!(super::sentinel_discovery_phase(0, true), "rediscovery");
        assert_eq!(super::sentinel_discovery_phase(1, false), "rediscovery");

        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            super::record_sentinel_discovery("valkey", "rediscovery", "success");
            super::record_sentinel_failover("valkey", "recovered");
        });
        let snapshot = snapshotter.snapshot().into_vec();
        let labels_for = |name: &str| {
            snapshot
                .iter()
                .find(|(key, ..)| key.key().name() == name)
                .map(|(key, ..)| {
                    let mut labels = key
                        .key()
                        .labels()
                        .map(|label| label.key().to_owned())
                        .collect::<Vec<_>>();
                    labels.sort();
                    labels
                })
        };
        assert_eq!(
            labels_for(super::METRIC_SENTINEL_DISCOVERY),
            Some(vec!["phase".into(), "result".into()])
        );
        assert_eq!(
            labels_for(super::METRIC_SENTINEL_FAILOVER),
            Some(vec!["outcome".into()])
        );
    }

    #[test]
    fn sentinel_primary_url_brackets_ipv6_literals() {
        assert_eq!(super::primary_url("::1", 6379), "redis://[::1]:6379");
        assert_eq!(super::primary_url("[::1]", 6379), "redis://[::1]:6379");
        assert_eq!(
            super::primary_url("primary.example", 6379),
            "redis://primary.example:6379"
        );
    }

    #[test]
    fn transaction_generation_fence_rejects_pool_or_shared_invalidation() {
        assert!(super::transaction_generation_is_current(4, 4, 7, 7));
        assert!(!super::transaction_generation_is_current(4, 5, 7, 7));
        assert!(!super::transaction_generation_is_current(4, 4, 7, 8));
    }

    #[test]
    fn abort_retry_rejects_an_expired_operation_deadline() {
        let retry = super::AbortRetry::start(Instant::now() - Duration::from_secs(1));
        assert!(retry.remaining().is_err(), "expired operation must not retry");
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "live fixture test covers discovery, bootstrap, and command admission"
    )]
    async fn live_sentinel_pipeline_uses_the_discovered_primary() {
        let Ok(endpoint_env) = std::env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS") else {
            return;
        };
        let endpoints = endpoint_env.split(',').map(str::to_owned).collect::<Vec<_>>();
        let nonempty_env = |name| std::env::var(name).ok().filter(|value| !value.is_empty());
        let backend_name = std::env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".into());
        let sentinel_password = nonempty_env("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD");
        let data_password = nonempty_env("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD");
        let connection = ValkeyConnection::sentinel(
            endpoints,
            "mymaster".into(),
            None,
            sentinel_password,
            None,
            data_password,
            false,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_secs(1),
                discovery: Duration::from_secs(5),
            },
            if backend_name == "redis" { "redis" } else { "valkey" },
        )
        .expect("fixture Sentinel configuration should build");
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let result: Result<(String,), BackendError> = connection.pipeline(&pipe).await;
        assert!(result.is_ok(), "native Sentinel discovery failed: {result:?}");
        if std::env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_HEALTH_TEST").as_deref() == Ok("1") {
            return;
        }
        let bootstrap = Box::pin(super::super::ensure_accounting_config(
            &connection,
            "praxis:test:sentinel:live",
            "praxis:test:sentinel:live:accounting-config",
            "v3:live-sentinel-test",
            "praxis:test:sentinel:live:keys",
        ))
        .await;
        assert!(bootstrap.is_ok(), "shared-backend bootstrap failed: {bootstrap:?}");
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "live fixture covers an ambiguous mutation through recovery"
    )]
    async fn live_sentinel_ambiguous_mutation_is_not_replayed() {
        let Ok(endpoint_env) = std::env::var("TOKEN_RATE_LIMIT_SENTINEL_ENDPOINTS") else {
            return;
        };
        if std::env::var("TOKEN_RATE_LIMIT_SENTINEL_REPLICATION_HEALTH_TEST").as_deref() == Ok("1") {
            // The health lane intentionally configures NOREPLICAS before the
            // child tests; it has its own closed/open admission assertion.
            return;
        }
        let endpoints = endpoint_env.split(',').map(str::to_owned).collect::<Vec<_>>();
        let nonempty_env = |name| std::env::var(name).ok().filter(|value| !value.is_empty());
        let backend_name = std::env::var("TOKEN_RATE_LIMIT_SENTINEL_PRODUCT").unwrap_or_else(|_| "valkey".into());
        let connection = ValkeyConnection::sentinel(
            endpoints,
            "mymaster".into(),
            None,
            nonempty_env("TOKEN_RATE_LIMIT_SENTINEL_PASSWORD"),
            None,
            nonempty_env("TOKEN_RATE_LIMIT_SENTINEL_DATA_PASSWORD"),
            false,
            ValkeyTimeouts {
                connect: Duration::from_secs(1),
                operation: Duration::from_millis(50),
                discovery: Duration::from_secs(5),
            },
            if backend_name == "redis" { "redis" } else { "valkey" },
        )
        .expect("fixture Sentinel configuration should build");
        let key = format!("praxis:test:sentinel:ambiguous:{}", std::process::id());
        let mut clear = redis::pipe();
        clear.cmd("DEL").arg(&key).ignore();
        let _: () = connection
            .pipeline(&clear)
            .await
            .expect("Sentinel primary should accept the setup mutation");

        // INCR is applied before DEBUG SLEEP blocks the server. The client
        // times out waiting for the pipeline reply, so the result is
        // intentionally unconfirmed and must never be replayed.
        let mut mutation = redis::pipe();
        mutation.cmd("INCR").arg(&key);
        mutation.cmd("DEBUG").arg("SLEEP").arg(0.2);
        let result: Result<Vec<redis::Value>, BackendError> = connection.pipeline(&mutation).await;
        assert!(
            matches!(result, Err(BackendError::Unconfirmed(_))),
            "a Sentinel-dispatched mutation with a lost reply must be unconfirmed: {result:?}"
        );

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut read = redis::pipe();
            read.cmd("GET").arg(&key);
            match connection.pipeline::<(Option<i64>,)>(&read).await {
                Ok((value,)) => {
                    assert_eq!(value, Some(1), "the ambiguous Sentinel mutation must apply once");
                    break;
                },
                Err(error) if Instant::now() < deadline => {
                    assert!(
                        matches!(error, BackendError::Unavailable(_) | BackendError::Unconfirmed(_)),
                        "only transient post-timeout errors are expected: {error:?}"
                    );
                    tokio::time::sleep(Duration::from_millis(100)).await;
                },
                Err(error) => panic!("Sentinel primary did not recover after the ambiguous mutation: {error:?}"),
            }
        }
    }

    #[tokio::test]
    async fn live_valkey_pipeline_runs_on_one_shared_connection() {
        let Some(url) = valkey_url() else {
            return;
        };
        let connection = ValkeyConnection::new(url).unwrap();
        let mut pipe = redis::pipe();
        pipe.cmd("SET").arg("praxis:test:conn").arg(1).ignore();
        pipe.cmd("INCRBY").arg("praxis:test:conn").arg(2);
        pipe.cmd("DEL").arg("praxis:test:conn").ignore();
        let (value,): (i64,) = connection.pipeline(&pipe).await.unwrap();
        assert_eq!(value, 3, "the pipeline ran in order on the shared connection");
        assert!(
            connection.connection.lock().unwrap().is_some(),
            "a successful pipeline keeps the cached connection"
        );
        drop(connection);
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "first and second are consumed by finish()/drop(); lint fires spuriously on transient borrows across await"
    )]
    async fn live_valkey_transaction_connections_are_reused_after_finish() {
        let Some(url) = valkey_url() else {
            return;
        };
        let connection = ValkeyConnection::new(url).unwrap();
        let first = connection.transaction().await.unwrap();
        first.finish(VALKEY_TIMEOUT).await;
        let second = connection.transaction().await.unwrap();
        assert_eq!(
            connection.transactions.idle.lock().await.len(),
            0,
            "the returned connection was handed out again instead of opening another"
        );
        drop(second);
        drop(connection);
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the checked-out connection and spawned waiter are consumed by finish and join"
    )]
    async fn live_valkey_waiter_reuses_connection_returned_at_pool_limit() {
        let Some(url) = valkey_url() else {
            return;
        };
        let mut connection = ValkeyConnection::new(url).unwrap();
        connection.transactions = Arc::new(super::TransactionConnections {
            idle: tokio::sync::Mutex::new(Vec::new()),
            semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
            generation: std::sync::atomic::AtomicU64::new(0),
        });
        let first = connection.transaction().await.unwrap();
        let waiting = connection.clone();
        let waiter = tokio::spawn(async move { waiting.transaction().await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "the sole checkout slot is still held");
        first.finish(VALKEY_TIMEOUT).await;
        let second = tokio::time::timeout(VALKEY_TIMEOUT, waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            connection.transactions.idle.lock().await.is_empty(),
            "the waiter checked out the returned connection"
        );
        drop(second);
    }

    #[tokio::test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "checked_out is consumed by the finish loop; lint fires spuriously on the Vec's element drops across await"
    )]
    async fn live_valkey_finishing_more_than_the_idle_cap_keeps_exactly_the_cap() {
        let Some(url) = valkey_url() else {
            return;
        };
        let connection = ValkeyConnection::new(url).unwrap();
        let mut checked_out = Vec::new();
        for _ in 0..MAX_IDLE_TRANSACTION_CONNECTIONS + 3 {
            checked_out.push(connection.transaction().await.unwrap());
        }
        for transaction in checked_out {
            transaction.finish(VALKEY_TIMEOUT).await;
        }
        assert_eq!(
            connection.transactions.idle.lock().await.len(),
            MAX_IDLE_TRANSACTION_CONNECTIONS,
            "connections finished beyond the idle cap are dropped, not pooled"
        );
        // Drop the connection (and its idle pool) before the test ends.
        drop(connection);
    }

    #[tokio::test]
    async fn a_connection_that_failed_is_dropped_instead_of_returned() {
        let Some(url) = valkey_url() else {
            return;
        };
        let connection = ValkeyConnection::new(url).unwrap();
        let mut checked_out = connection.transaction().await.unwrap();
        redis::cmd("WATCH")
            .arg("praxis:test:dirty")
            .exec_async(checked_out.inner())
            .await
            .unwrap();
        drop(checked_out);
        assert!(
            connection.transactions.idle.lock().await.is_empty(),
            "a dropped (unfinished) transaction connection is never pooled"
        );
    }

    #[tokio::test]
    async fn pipeline_times_out_and_invalidates_the_connection_when_wedged() {
        let Some(url) = valkey_url() else {
            return;
        };
        let (connection, wedged) = spawn_wedgeable_proxy(&url).await;
        let mut pipe = redis::pipe();
        pipe.cmd("PING");
        let (_pong,): (String,) = connection.pipeline(&pipe).await.unwrap();
        wedged.store(true, Ordering::Relaxed);
        let started = Instant::now();
        let outcome: Result<(String,), _> = connection.pipeline(&pipe).await;
        assert!(outcome.is_err(), "a wedged server must fail the pipeline");
        assert!(
            started.elapsed() < VALKEY_TIMEOUT * 3,
            "the failure must arrive near the configured timeout"
        );
        assert!(
            connection.connection.lock().unwrap().is_none(),
            "a failed pipeline drops the cached connection"
        );
        drop(connection);
    }

    #[tokio::test]
    async fn live_ambiguous_pipeline_is_not_replayed_after_the_reply_is_lost() {
        let Some(url) = valkey_url() else {
            return;
        };
        let (connection, wedged) = spawn_wedgeable_proxy(&url).await;
        let key = format!("praxis:test:ambiguous:{}", std::process::id());
        let mut clear = redis::pipe();
        clear.cmd("DEL").arg(&key).ignore();
        let _: () = connection.pipeline(&clear).await.unwrap();

        wedged.store(true, Ordering::SeqCst);
        let mut mutation = redis::pipe();
        mutation.cmd("INCR").arg(&key);
        let result: Result<(i64,), BackendError> = connection.pipeline(&mutation).await;
        assert!(
            matches!(result, Err(BackendError::Unconfirmed(_))),
            "lost command reply must be unconfirmed"
        );

        wedged.store(false, Ordering::SeqCst);
        let mut read = redis::pipe();
        read.cmd("GET").arg(&key);
        let (value,): (Option<i64>,) = connection.pipeline(&read).await.unwrap();
        assert_eq!(
            value,
            Some(1),
            "the ambiguous mutation must have been applied once, not replayed"
        );
        drop(connection);
    }

    #[test]
    fn map_valkey_error_tags_the_message_with_which_phase_failed() {
        let timed_out = redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::TimedOut));

        let BackendError::Unavailable(message) = map_valkey_error("connection", &timed_out) else {
            panic!("map_valkey_error must always return BackendError::Unavailable")
        };
        assert_eq!(
            message, "Redis/Valkey connection: timed out",
            "a connect failure must say it came from connecting"
        );

        let BackendError::Unavailable(message) = map_valkey_error("command", &timed_out) else {
            panic!("map_valkey_error must always return BackendError::Unavailable")
        };
        assert_eq!(
            message, "Redis/Valkey command: timed out",
            "a command failure must say it came from a command"
        );
    }

    #[test]
    fn dispatched_command_errors_are_marked_unconfirmed_for_no_replay() {
        let timed_out = redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::TimedOut));
        let BackendError::Unconfirmed(message) = command_error(&timed_out) else {
            panic!("a dispatched command error must not be classified as a safe-to-replay outage")
        };
        assert!(message.contains("Redis/Valkey command: timed out"));
    }

    /// A one-shot TCP proxy in front of `upstream`'s `host:port`, for
    /// fault-injecting a Valkey that accepts a command and then hangs.
    /// Real Valkey/Redis has no config knob for this; `DEBUG SLEEP`
    /// comes closest but additionally requires `enable-debug-command`
    /// server-side, so this proxies the real, unmodified Valkey under
    /// test instead.
    ///
    /// Returns the proxy's local address and a flag that, once set,
    /// makes every open connection stop relaying upstream replies back
    /// to the client -- the bytes are still read off the wire (so the
    /// upstream Valkey itself never blocks or errors), just dropped.
    #[expect(
        clippy::too_many_lines,
        reason = "fault-proxy startup preserves authenticated connection settings"
    )]
    async fn spawn_wedgeable_proxy(upstream: &str) -> (ValkeyConnection, Arc<AtomicBool>) {
        let info = redis::IntoConnectionInfo::into_connection_info(upstream).expect("parse test backend URL");
        let upstream = match info.addr() {
            redis::ConnectionAddr::Tcp(host, port) => super::primary_url(host, *port)
                .strip_prefix("redis://")
                .expect("generated plain endpoint")
                .to_owned(),
            _ => panic!("fault relay requires a plain TCP backend"),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = listener.local_addr().unwrap();
        let wedged = Arc::new(AtomicBool::new(false));

        let wedged_for_task = Arc::clone(&wedged);
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                tokio::spawn(relay_one_connection(
                    client,
                    upstream.clone(),
                    Arc::clone(&wedged_for_task),
                ));
            }
        });

        let client = redis::Client::open(info.set_addr(redis::ConnectionAddr::Tcp(
            proxy_addr.ip().to_string(),
            proxy_addr.port(),
        )))
        .expect("build authenticated fault-proxy client");
        let mut connection = ValkeyConnection::new(format!("redis://{proxy_addr}")).unwrap();
        connection.source = Arc::new(super::ConnectionSource::Standalone(client));
        (connection, wedged)
    }

    /// One [`spawn_wedgeable_proxy`] connection's relay loop: client
    /// bytes always flow through to `upstream` unmodified; `upstream`'s
    /// replies flow back to the client unless/until `wedged` is set, at
    /// which point they're read off the wire (so `upstream` never
    /// blocks) but silently dropped instead of relayed.
    async fn relay_one_connection(mut client: tokio::net::TcpStream, upstream: String, wedged: Arc<AtomicBool>) {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let Ok(mut server) = tokio::net::TcpStream::connect(&upstream).await else {
            return;
        };
        let (mut client_read, mut client_write) = client.split();
        let (mut server_read, mut server_write) = server.split();
        tokio::join!(
            async {
                drop(tokio::io::copy(&mut client_read, &mut server_write).await);
            },
            async {
                let mut buffer = [0_u8; 4096];
                loop {
                    let Ok(read @ 1..) = server_read.read(&mut buffer).await else {
                        return;
                    };
                    if wedged.load(Ordering::SeqCst) {
                        continue; // Accepted off the wire, never relayed: a silent hang.
                    }
                    let Some(bytes) = buffer.get(..read) else {
                        return;
                    };
                    if client_write.write_all(bytes).await.is_err() {
                        return;
                    }
                }
            }
        );
    }
}
