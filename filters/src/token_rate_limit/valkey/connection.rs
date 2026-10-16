// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey connections for the token rate limit backends: one shared
//! multiplexed connection for pipelines, and a small pool of dedicated
//! connections for `WATCH` transactions, which are per-connection state.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use praxis_filter::FilterError;
use rand::RngExt as _;
use redis::{
    aio::MultiplexedConnection,
    sentinel::{SentinelClient, SentinelClientBuilder, SentinelServerType},
};

use super::super::{
    backend::BackendError, config::BackendConfig, expand_backend_url, parse_backend_timeout,
    validate_insecure_tls_opt_in,
};

/// Bound on connecting and on every command's response; the token bucket
/// also uses it as the deadline for retrying aborted transactions.
pub(in crate::token_rate_limit) const VALKEY_TIMEOUT: Duration = Duration::from_millis(500);
/// Default overall bound for native Sentinel traversal and convergence.
pub(in crate::token_rate_limit) const SENTINEL_TIMEOUT: Duration = Duration::from_secs(3);
/// Delay between bounded native Sentinel convergence attempts.
const SENTINEL_RETRY_INTERVAL: Duration = Duration::from_millis(50);

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
    connection: Arc<tokio::sync::Mutex<Option<CachedConnection>>>,
    /// Dedicated connections for `WATCH` transactions.
    transactions: Arc<TransactionConnections>,
    /// Generation of the cached shared connection.
    generation: Arc<AtomicU64>,
    /// Bounds for connection and operation work.
    timeouts: ValkeyTimeouts,
    /// Bounded operator-facing selector label (`redis` or `valkey`).
    backend_name: &'static str,
}

/// Lazy source for standalone or Sentinel-discovered writable-primary data.
enum ConnectionSource {
    /// Fixed standalone Redis/Valkey client.
    Standalone(redis::Client),
    /// Native Sentinel client plus its bounded discovery deadline.
    Sentinel {
        /// Serialized access to the native discovery cache.
        client: tokio::sync::Mutex<SentinelClient>,
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

/// Native redis-rs inputs for the TLS Sentinel discovery fallback.
struct TlsSentinelConfig {
    /// Ordered Sentinel connection information.
    endpoints: Vec<redis::ConnectionInfo>,
    /// Monitored primary service name.
    service_name: String,
    /// Credentials used by discovered data nodes.
    data_settings: redis::RedisConnectionInfo,
    /// Whether certificate verification is disabled for this fixture/profile.
    tls_insecure: bool,
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
        Self::standalone_with_timeouts_and_name(url, ValkeyTimeouts::default(), "valkey")
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
            connection: Arc::new(tokio::sync::Mutex::new(None)),
            transactions: Arc::new(TransactionConnections::default()),
            generation: Arc::new(AtomicU64::new(0)),
            timeouts,
            backend_name,
        })
    }

    /// Build the shared backend transport from filter configuration. All
    /// Redis/Valkey topology, credential, TLS, and timeout decisions stay in
    /// this backend module; admission code receives only the backend trait.
    #[expect(clippy::too_many_lines, reason = "topology construction is one backend boundary")]
    pub(in crate::token_rate_limit) fn from_backend_config(
        backend: &BackendConfig,
        backend_name: &'static str,
    ) -> Result<Self, FilterError> {
        if backend.url.is_some() && backend.sentinel.is_some() {
            return Err("token_rate_limit: backend.url and backend.sentinel are mutually exclusive".into());
        }
        let url = backend.url.as_deref().map(expand_backend_url).transpose()?;
        let discovery_timeout = backend
            .sentinel
            .as_ref()
            .and_then(|sentinel| sentinel.discovery_timeout.as_deref());
        let timeouts = ValkeyTimeouts {
            connect: parse_backend_timeout(backend.connect_timeout.as_deref(), "connect_timeout", VALKEY_TIMEOUT)?,
            operation: parse_backend_timeout(
                backend.operation_timeout.as_deref(),
                "operation_timeout",
                VALKEY_TIMEOUT,
            )?,
            discovery: parse_backend_timeout(discovery_timeout, "discovery_timeout", SENTINEL_TIMEOUT)?,
        };
        let connection = match (&url, &backend.sentinel) {
            (Some(url), None) => Self::standalone_with_timeouts_and_name(url.clone(), timeouts, backend_name)?,
            (None, Some(sentinel)) => {
                let endpoints = sentinel
                    .endpoints
                    .iter()
                    .map(|endpoint| expand_backend_url(endpoint))
                    .collect::<Result<Vec<_>, _>>()?;
                let expand_credential = |value: &Option<String>| value.as_deref().map(expand_backend_url).transpose();
                let connection = Self::sentinel(
                    endpoints,
                    sentinel.service_name.clone(),
                    expand_credential(&sentinel.sentinel_username)?,
                    expand_credential(&sentinel.sentinel_password)?,
                    expand_credential(&sentinel.redis_username)?,
                    expand_credential(&sentinel.redis_password)?,
                    sentinel.tls,
                    sentinel.tls_insecure,
                    timeouts,
                    backend_name,
                )?;
                validate_insecure_tls_opt_in(sentinel.tls_insecure)?;
                connection
            },
            (None, None) => {
                return Err(
                    "token_rate_limit: backend.url or backend.sentinel is required for backend.kind: redis/valkey"
                        .into(),
                );
            },
            (Some(_), Some(_)) => unreachable!("mutually exclusive backend topology was checked above"),
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
        tls_insecure: bool,
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
        let sentinel_username_for_source = sentinel_username.clone();
        let sentinel_password_for_source = sentinel_password.clone();
        let redis_username_for_source = redis_username.clone();
        let redis_password_for_source = redis_password.clone();
        let mut builder =
            SentinelClientBuilder::new(addresses.clone(), service_name.clone(), SentinelServerType::Master)
                .map_err(|error| BackendError::Unavailable(format!("Sentinel configuration: {error}")))?;
        if let Some(value) = sentinel_username {
            builder = builder.set_client_to_sentinel_username(value);
        }
        if let Some(value) = sentinel_password {
            builder = builder.set_client_to_sentinel_password(value);
        }
        if let Some(value) = redis_username {
            builder = builder.set_client_to_redis_username(value);
        }
        if let Some(value) = redis_password {
            builder = builder.set_client_to_redis_password(value);
        }
        if tls {
            let mode = if tls_insecure {
                redis::TlsMode::Insecure
            } else {
                redis::TlsMode::Secure
            };
            builder = builder
                .set_client_to_sentinel_tls_mode(mode)
                .set_client_to_redis_tls_mode(mode);
        } else if tls_insecure {
            return Err(BackendError::Unavailable("tls_insecure requires tls: true".into()));
        }
        let client = builder
            .build()
            .map_err(|error| BackendError::Unavailable(format!("Sentinel configuration: {error}")))?;
        let source = if tls {
            let sentinel_settings = redis::RedisConnectionInfo::default();
            let sentinel_settings = if let Some(value) = &sentinel_username_for_source {
                sentinel_settings.set_username(value)
            } else {
                sentinel_settings
            };
            let sentinel_settings = if let Some(value) = &sentinel_password_for_source {
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
                                    insecure: tls_insecure,
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
            let data_settings = if let Some(value) = &redis_username_for_source {
                data_settings.set_username(value)
            } else {
                data_settings
            };
            let data_settings = if let Some(value) = &redis_password_for_source {
                data_settings.set_password(value)
            } else {
                data_settings
            };
            ConnectionSource::SentinelTls {
                config: Arc::new(TlsSentinelConfig {
                    endpoints,
                    service_name,
                    data_settings,
                    tls_insecure,
                }),
            }
        } else {
            ConnectionSource::Sentinel {
                client: tokio::sync::Mutex::new(client),
            }
        };
        Ok(Self {
            source: Arc::new(source),
            connection: Arc::new(tokio::sync::Mutex::new(None)),
            transactions: Arc::new(TransactionConnections::default()),
            generation: Arc::new(AtomicU64::new(0)),
            timeouts,
            backend_name,
        })
    }

    /// Stable selector label for bounded telemetry.
    pub(in crate::token_rate_limit) const fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    /// Run `pipe` on the shared connection, dropping that connection on
    /// any failure so a wedged one is not reused.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] on connection or command errors,
    /// including the [`VALKEY_TIMEOUT`] enforced by the client.
    pub(in crate::token_rate_limit) async fn pipeline<T: redis::FromRedisValue>(
        &self,
        pipe: &redis::Pipeline,
    ) -> Result<T, BackendError> {
        let (mut connection, generation) = self.shared().await?;
        match pipe.query_async::<T>(&mut connection).await {
            Ok(value) => Ok(value),
            Err(error) => {
                self.invalidate(generation).await;
                Err(command_error(&error))
            },
        }
    }

    /// Invalidate every cached socket after a connection-level failure. The
    /// next operation performs fresh native discovery; no failed mutation is
    /// replayed here.
    async fn invalidate(&self, generation: u64) {
        let mut cached = self.connection.lock().await;
        if cached.as_ref().is_some_and(|entry| entry.generation == generation) {
            *cached = None;
            drop(cached);
            self.generation
                .fetch_max(generation.saturating_add(1), Ordering::AcqRel);
            self.transactions.generation.fetch_add(1, Ordering::AcqRel);
            self.transactions.idle.lock().await.clear();
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
        let (connection, generation) = loop {
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
                }) if pooled_generation == generation => break (connection, generation),
                Some(PooledConnection { .. }) => {},
                None => break (self.open().await?, generation),
            }
        };
        Ok(TransactionConnection {
            connection: Some(connection),
            pool: Arc::clone(&self.transactions),
            permit: Some(permit),
            generation,
            shared_generation: Arc::clone(&self.generation),
            shared_generation_at_checkout: self.generation.load(Ordering::Acquire),
        })
    }

    /// Return the cached shared connection, opening (and caching) a fresh
    /// one on first use or after a prior failure cleared it.
    async fn shared(&self) -> Result<(MultiplexedConnection, u64), BackendError> {
        let lock_timeout = match self.source.as_ref() {
            ConnectionSource::Standalone(_) => self.timeouts.connect,
            ConnectionSource::Sentinel { .. } | ConnectionSource::SentinelTls { .. } => self.timeouts.discovery,
        };
        let deadline = Instant::now() + lock_timeout;
        let mut cached = tokio::time::timeout(lock_timeout, self.connection.lock())
            .await
            .map_err(|_elapsed| BackendError::Unavailable("shared connection lock timeout".into()))?;
        let current_generation = self.generation.load(Ordering::Acquire);
        if let Some(entry) = cached.as_ref() {
            if entry.generation == current_generation {
                return Ok((entry.connection.clone(), entry.generation));
            }
            *cached = None;
        }
        let connection = self.open_until(deadline).await?;
        let generation = self.generation.load(Ordering::Acquire);
        *cached = Some(CachedConnection {
            connection: connection.clone(),
            generation,
        });
        drop(cached);
        Ok((connection, generation))
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
    #[expect(
        clippy::allow_attributes,
        reason = "this stack-frame allowance must work across supported Clippy versions"
    )]
    #[allow(
        clippy::large_stack_frames,
        reason = "bounded connection/discovery state machine owns its timeout state"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "deadline-aware standalone and Sentinel setup share one lifecycle"
    )]
    async fn open_until(&self, deadline: Instant) -> Result<MultiplexedConnection, BackendError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(BackendError::Unavailable("backend connection deadline elapsed".into()));
        }
        let config = redis::AsyncConnectionConfig::new()
            .set_connection_timeout(Some(self.timeouts.connect.min(remaining)))
            .set_response_timeout(Some(self.timeouts.operation));
        match self.source.as_ref() {
            ConnectionSource::Standalone(client) => {
                tokio::time::timeout(remaining, client.get_multiplexed_async_connection_with_config(&config))
                    .await
                    .map_err(|_elapsed| BackendError::Unavailable("backend connection deadline elapsed".into()))?
                    .map_err(|error| map_valkey_error("connection", &error))
            },
            ConnectionSource::Sentinel { client } => {
                let mut client = tokio::time::timeout(remaining, client.lock())
                    .await
                    .map_err(|_elapsed| BackendError::Unavailable("Sentinel discovery lock timeout".into()))?;
                let mut last_error = None;
                loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match tokio::time::timeout(remaining, client.get_async_connection_with_config(&config)).await {
                        Ok(Ok(connection)) => return Ok(connection),
                        Ok(Err(error)) => last_error = Some(error),
                        Err(_elapsed) => break,
                    }
                    tokio::time::sleep(SENTINEL_RETRY_INTERVAL.min(deadline.saturating_duration_since(Instant::now())))
                        .await;
                }
                drop(client);
                Err(last_error.map_or_else(
                    || BackendError::Unavailable("Sentinel discovery timeout".into()),
                    |error| map_valkey_error("Sentinel discovery", &error),
                ))
            },
            ConnectionSource::SentinelTls { config: sentinel } => {
                self.open_tls_sentinel(sentinel, deadline, &config).await
            },
        }
    }

    /// Discover a TLS primary with native redis-rs connections. This avoids
    /// redis-rs' internal async Sentinel reconnect path, which does not pass
    /// the caller's TLS configuration through its Sentinel probe.
    #[expect(
        clippy::allow_attributes,
        reason = "this stack-frame allowance must work across supported Clippy versions"
    )]
    #[allow(
        clippy::large_stack_frames,
        reason = "bounded TLS discovery owns endpoint traversal and role verification"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "bounded TLS discovery owns endpoint traversal, role verification, and retry timing"
    )]
    async fn open_tls_sentinel(
        &self,
        sentinel: &TlsSentinelConfig,
        deadline: Instant,
        connection_config: &redis::AsyncConnectionConfig,
    ) -> Result<MultiplexedConnection, BackendError> {
        loop {
            for endpoint in &sentinel.endpoints {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(BackendError::Unavailable("Sentinel TLS discovery timeout".into()));
                }
                let client = match redis::Client::open(endpoint.clone()) {
                    Ok(client) => client,
                    Err(error) => return Err(map_valkey_error("Sentinel TLS configuration", &error)),
                };
                let mut sentinel_connection = match tokio::time::timeout(
                    remaining,
                    client.get_multiplexed_async_connection_with_config(connection_config),
                )
                .await
                {
                    Ok(Ok(connection)) => connection,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => break,
                };
                let address: (String, u16) = match tokio::time::timeout(
                    deadline.saturating_duration_since(Instant::now()),
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
                    Err(error) => return Err(map_valkey_error("Sentinel primary address", &error)),
                };
                let data_info = base
                    .set_addr(redis::ConnectionAddr::TcpTls {
                        host: primary_host,
                        port: address.1,
                        insecure: sentinel.tls_insecure,
                        tls_params: None,
                    })
                    .set_redis_settings(sentinel.data_settings.clone());
                let data_client = redis::Client::open(data_info)
                    .map_err(|error| map_valkey_error("TLS primary configuration", &error))?;
                let mut data_connection = match tokio::time::timeout(
                    deadline.saturating_duration_since(Instant::now()),
                    data_client.get_multiplexed_async_connection_with_config(connection_config),
                )
                .await
                {
                    Ok(Ok(connection)) => connection,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                let role: redis::Role = match tokio::time::timeout(
                    deadline.saturating_duration_since(Instant::now()),
                    redis::cmd("ROLE").query_async(&mut data_connection),
                )
                .await
                {
                    Ok(Ok(role)) => role,
                    Ok(Err(_error)) => continue,
                    Err(_elapsed) => continue,
                };
                if matches!(role, redis::Role::Primary { .. }) {
                    return Ok(data_connection);
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
    /// Shared cache generation, fenced on transaction failure.
    shared_generation: Arc<AtomicU64>,
    /// Shared generation observed when this transaction was checked out.
    shared_generation_at_checkout: u64,
}

impl TransactionConnection {
    /// The underlying connection.
    pub(super) fn inner(&mut self) -> &mut MultiplexedConnection {
        match self.connection.as_mut() {
            Some(connection) => connection,
            None => unreachable!("checked-out transaction connection is present"),
        }
    }

    /// Return the connection to the pool for reuse, or drop both when
    /// [`MAX_IDLE_TRANSACTION_CONNECTIONS`] are already idle.
    pub(super) async fn finish(mut self) {
        let Some(connection) = self.connection.take() else {
            unreachable!("checked-out transaction connection is present");
        };
        let pool = Arc::clone(&self.pool);
        let generation = self.generation;
        let mut idle = pool.idle.lock().await;
        if generation == pool.generation.load(Ordering::Acquire) && idle.len() < MAX_IDLE_TRANSACTION_CONNECTIONS {
            idle.push(PooledConnection { connection, generation });
        }
        drop(idle);
        drop(self.permit.take());
        drop(self);
    }
}

impl Drop for TransactionConnection {
    fn drop(&mut self) {
        if self.connection.is_some() {
            self.pool.generation.fetch_add(1, Ordering::AcqRel);
            self.shared_generation
                .fetch_max(self.shared_generation_at_checkout.saturating_add(1), Ordering::AcqRel);
        }
    }
}

/// Clear a watch before returning a connection without executing a transaction.
pub(super) async fn unwatch(connection: &mut MultiplexedConnection) -> Result<(), BackendError> {
    redis::cmd("UNWATCH")
        .exec_async(connection)
        .await
        .map_err(|error| command_error(&error))
}

/// Retry aborted optimistic transactions within the Valkey deadline.
pub(super) struct AbortRetry {
    /// Monotonic start time for the retry deadline.
    started: Instant,
    /// Upper bound of the next jittered pause.
    backoff: Duration,
}

impl AbortRetry {
    /// Start the bounded retry window.
    pub(super) fn start() -> Self {
        Self {
            started: Instant::now(),
            backoff: FIRST_ABORT_BACKOFF,
        }
    }

    /// Pause before another attempt, failing closed once the deadline passes.
    pub(super) async fn pause(&mut self) -> Result<(), BackendError> {
        let left = VALKEY_TIMEOUT
            .checked_sub(self.started.elapsed())
            .filter(|left| !left.is_zero())
            .ok_or_else(|| BackendError::Unavailable("Valkey transaction contended".into()))?;
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

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::{Duration, Instant},
    };

    use super::{
        BackendError, MAX_IDLE_TRANSACTION_CONNECTIONS, VALKEY_TIMEOUT, ValkeyConnection, ValkeyTimeouts,
        command_error, map_valkey_error,
    };

    fn valkey_url() -> Option<String> {
        std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL").ok()
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
        assert!(connection.connection.lock().await.is_none());
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
            true,
            ValkeyTimeouts::default(),
            "valkey",
        );
        assert!(connection.is_ok(), "valkeys:// is a valid TLS Sentinel endpoint");
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
        let bootstrap = super::super::ensure_accounting_config(
            &connection,
            "praxis:test:sentinel:live",
            "praxis:test:sentinel:live:accounting-config",
            "v3:live-sentinel-test",
            "praxis:test:sentinel:live:keys",
        )
        .await;
        assert!(bootstrap.is_ok(), "shared-backend bootstrap failed: {bootstrap:?}");
    }

    #[tokio::test]
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
        mutation.cmd("DEBUG").arg("SLEEP").arg(1);
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
            connection.connection.lock().await.is_some(),
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
        first.finish().await;
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
        first.finish().await;
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
            transaction.finish().await;
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
        let (proxy_addr, wedged) = spawn_wedgeable_proxy(&url).await;
        let connection = ValkeyConnection::new(format!("redis://{proxy_addr}")).unwrap();
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
            connection.connection.lock().await.is_none(),
            "a failed pipeline drops the cached connection"
        );
        drop(connection);
    }

    #[tokio::test]
    async fn live_ambiguous_pipeline_is_not_replayed_after_the_reply_is_lost() {
        let Some(url) = valkey_url() else {
            return;
        };
        let (proxy_addr, wedged) = spawn_wedgeable_proxy(&url).await;
        let connection = ValkeyConnection::new(format!("redis://{proxy_addr}")).unwrap();
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
    async fn spawn_wedgeable_proxy(upstream: &str) -> (std::net::SocketAddr, Arc<AtomicBool>) {
        let upstream = upstream
            .strip_prefix("redis://")
            .expect("test fixture: TOKEN_RATE_LIMIT_VALKEY_URL must be a bare redis://host:port URL")
            .to_owned();
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

        (proxy_addr, wedged)
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
