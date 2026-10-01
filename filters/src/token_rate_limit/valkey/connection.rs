// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Valkey connections for the token rate limit backends: one shared
//! multiplexed connection for pipelines, and a small pool of dedicated
//! connections for `WATCH` transactions, which are per-connection state.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use rand::RngExt as _;
use redis::aio::MultiplexedConnection;

use super::super::backend::BackendError;

/// Bound on connecting and on every command's response; the token bucket
/// also uses it as the deadline for retrying aborted transactions.
pub(super) const VALKEY_TIMEOUT: Duration = Duration::from_millis(500);

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
    client: redis::Client,
    /// Cached multiplexed connection for pipelines; cleared on any error so
    /// the next call reconnects instead of reusing a wedged socket.
    connection: Arc<tokio::sync::Mutex<Option<MultiplexedConnection>>>,
    /// Dedicated connections for `WATCH` transactions.
    transactions: Arc<TransactionConnections>,
}

impl ValkeyConnection {
    /// Open a lazy client for `url`.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when `url` is not a Valkey URL.
    pub(in crate::token_rate_limit) fn new(url: String) -> Result<Self, BackendError> {
        let client = redis::Client::open(url).map_err(|error| BackendError::Unavailable(error.to_string()))?;
        Ok(Self {
            client,
            connection: Arc::new(tokio::sync::Mutex::new(None)),
            transactions: Arc::new(TransactionConnections::default()),
        })
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
        let mut connection = self.shared().await?;
        match pipe.query_async::<T>(&mut connection).await {
            Ok(value) => Ok(value),
            Err(error) => {
                *self.connection.lock().await = None;
                Err(command_error(&error))
            },
        }
    }

    /// Check out a dedicated connection for one `WATCH` transaction.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when a new connection cannot
    /// be opened, or when [`MAX_TRANSACTION_CONNECTIONS`] are all in use
    /// and none becomes available within [`VALKEY_TIMEOUT`].
    pub(super) async fn transaction(&self) -> Result<TransactionConnection, BackendError> {
        let permit = tokio::time::timeout(VALKEY_TIMEOUT, Arc::clone(&self.transactions.semaphore).acquire_owned())
            .await
            .map_err(|_elapsed| BackendError::Unavailable("transaction pool exhausted: timeout".into()))?
            .map_err(|_closed| BackendError::Unavailable("transaction pool closed".into()))?;
        let idle = self.transactions.idle.lock().await.pop();
        let connection = if let Some(PooledConnection { connection }) = idle {
            connection
        } else {
            self.open().await?
        };
        Ok(TransactionConnection {
            connection,
            pool: Arc::clone(&self.transactions),
            permit,
        })
    }

    /// Return the cached shared connection, opening (and caching) a fresh
    /// one on first use or after a prior failure cleared it.
    async fn shared(&self) -> Result<MultiplexedConnection, BackendError> {
        let mut cached = self.connection.lock().await;
        if let Some(connection) = cached.as_ref() {
            return Ok(connection.clone());
        }
        let connection = self.open().await?;
        *cached = Some(connection.clone());
        drop(cached);
        Ok(connection)
    }

    /// Open a fresh connection, bounding both the connect attempt and
    /// every command's response to [`VALKEY_TIMEOUT`].
    async fn open(&self) -> Result<MultiplexedConnection, BackendError> {
        let config = redis::AsyncConnectionConfig::new()
            .set_connection_timeout(Some(VALKEY_TIMEOUT))
            .set_response_timeout(Some(VALKEY_TIMEOUT));
        self.client
            .get_multiplexed_async_connection_with_config(&config)
            .await
            .map_err(|error| map_valkey_error("connection", &error))
    }
}

/// An idle connection. Only checked-out connections hold semaphore permits.
struct PooledConnection {
    /// The idle connection ready for reuse.
    connection: MultiplexedConnection,
}

/// Idle dedicated connections, reused across transactions.
struct TransactionConnections {
    /// Connections returned by [`TransactionConnection::finish`].
    idle: tokio::sync::Mutex<Vec<PooledConnection>>,
    /// Bounds the total connections alive at once (idle + checked out).
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl Default for TransactionConnections {
    fn default() -> Self {
        Self {
            idle: tokio::sync::Mutex::new(Vec::new()),
            semaphore: Arc::new(tokio::sync::Semaphore::new(MAX_TRANSACTION_CONNECTIONS)),
        }
    }
}

/// One checked-out connection. Call [`Self::finish`] after a completed
/// `EXEC` (or `UNWATCH`); dropping it instead discards the connection, so a
/// connection with a dangling `WATCH` or a mid-flight error never comes back.
pub(super) struct TransactionConnection {
    /// The connection while checked out.
    connection: MultiplexedConnection,
    /// Pool to return it to.
    pool: Arc<TransactionConnections>,
    /// Held only while checked out; released when the connection is returned.
    permit: tokio::sync::OwnedSemaphorePermit,
}

impl TransactionConnection {
    /// The underlying connection.
    pub(super) fn inner(&mut self) -> &mut MultiplexedConnection {
        &mut self.connection
    }

    /// Return the connection to the pool for reuse, or drop both when
    /// [`MAX_IDLE_TRANSACTION_CONNECTIONS`] are already idle.
    pub(super) async fn finish(self) {
        let Self {
            connection,
            pool,
            permit,
        } = self;
        let mut idle = pool.idle.lock().await;
        if idle.len() < MAX_IDLE_TRANSACTION_CONNECTIONS {
            idle.push(PooledConnection { connection });
        }
        drop(idle);
        drop(permit);
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
    map_valkey_error("command", error)
}

/// Tag a client error with the phase it came from; the client's own text
/// ("timed out") does not say whether connecting or a command failed.
fn map_valkey_error(phase: &'static str, error: &redis::RedisError) -> BackendError {
    BackendError::Unavailable(format!("Valkey {phase}: {error}"))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    use super::{BackendError, MAX_IDLE_TRANSACTION_CONNECTIONS, VALKEY_TIMEOUT, ValkeyConnection, map_valkey_error};

    fn valkey_url() -> Option<String> {
        std::env::var("TOKEN_RATE_LIMIT_VALKEY_URL").ok()
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
        });
        let first = connection.transaction().await.unwrap();
        let waiting = connection.clone();
        let waiter = tokio::spawn(async move { waiting.transaction().await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
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
        let started = std::time::Instant::now();
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

    #[test]
    fn map_valkey_error_tags_the_message_with_which_phase_failed() {
        let timed_out = redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::TimedOut));

        let BackendError::Unavailable(message) = map_valkey_error("connection", &timed_out) else {
            panic!("map_valkey_error must always return BackendError::Unavailable")
        };
        assert_eq!(
            message, "Valkey connection: timed out",
            "a connect failure must say it came from connecting"
        );

        let BackendError::Unavailable(message) = map_valkey_error("command", &timed_out) else {
            panic!("map_valkey_error must always return BackendError::Unavailable")
        };
        assert_eq!(
            message, "Valkey command: timed out",
            "a command failure must say it came from a command"
        );
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
