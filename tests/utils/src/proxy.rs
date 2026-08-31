// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Proxy startup and configuration test utilities for integration tests.

use std::{
    collections::HashMap,
    fmt,
    path::PathBuf,
    sync::{Arc, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use pingora_core::server::{RunArgs, ShutdownSignal, ShutdownSignalWatch};
use praxis_core::{
    config::{Config, Listener},
    server::RuntimeOptions,
};
use praxis_filter::{FilterFactory, FilterPipeline, FilterRegistry, HttpFilter};
use praxis_protocol::http::load_http_handler;
use thiserror::Error;
use tokio::sync::Notify;

// -----------------------------------------------------------------------------
// Shared Test Client
// -----------------------------------------------------------------------------

/// Default sub-request client for registry-only tests without a config.
fn test_subrequest_client() -> praxis_core::subrequest::SubRequestClient {
    praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None))
}

/// Shared client honoring runtime connector settings, including the circuit breaker.
fn configured_subrequest_client(config: &Config) -> praxis_core::subrequest::SubRequestClient {
    praxis_ai::create_subrequest_client(config)
}

// -----------------------------------------------------------------------------
// Pipeline Building
// -----------------------------------------------------------------------------

/// Resolve a listener's filter chains into a [`FilterPipeline`].
///
/// Collects all [`FilterEntry`] items from the named chains
/// referenced by the listener, then builds the pipeline via
/// the provided registry.
///
/// [`FilterPipeline`]: praxis_filter::FilterPipeline
/// [`FilterEntry`]: praxis_core::config::FilterEntry
fn resolve_listener_pipeline(
    config: &Config,
    listener: &Listener,
    registry: &FilterRegistry,
    client: &praxis_core::subrequest::SubRequestClient,
) -> Arc<FilterPipeline> {
    let chains: HashMap<&str, &[_]> = config
        .filter_chains
        .iter()
        .map(|c| (c.name.as_str(), c.filters.as_slice()))
        .collect();

    let mut entries = Vec::new();
    for chain_name in &listener.filter_chains {
        let filters = chains
            .get(chain_name.as_str())
            .unwrap_or_else(|| panic!("unknown filter chain: {chain_name}"));
        entries.extend_from_slice(filters);
    }

    let mut pipeline = FilterPipeline::build_with_chains(&mut entries, registry, &chains).unwrap();
    pipeline
        .apply_body_limits(
            config.body_limits.max_request_bytes,
            config.body_limits.max_response_bytes,
            config.insecure_options.allow_unbounded_body,
        )
        .unwrap();
    pipeline.set_subrequest_client(client.clone());
    pipeline.add_pipeline_extension(Box::new(praxis_ai_apis::store::ResponseStoreRegistry::new()));
    Arc::new(pipeline)
}

/// Build the filter pipeline from the config using the
/// full AI registry (uses first listener). Resolves branch
/// chains via [`build_with_chains`].
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
///
/// [`build_with_chains`]: FilterPipeline::build_with_chains
pub fn build_pipeline(config: &Config) -> FilterPipeline {
    let client = configured_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&client);
    let listener = config
        .listeners
        .first()
        .expect("config must have at least one listener");

    Arc::try_unwrap(resolve_listener_pipeline(config, listener, &registry, &client))
        .unwrap_or_else(|_| panic!("pipeline Arc should have single owner"))
}

// -----------------------------------------------------------------------------
// Proxy Guard
// -----------------------------------------------------------------------------

/// Signals a Pingora server to shut down when notified.
struct NotifyShutdownWatch {
    /// Fires when the corresponding [`ProxyGuard`] is dropped.
    notify: Arc<Notify>,
}

#[async_trait::async_trait]
impl ShutdownSignalWatch for NotifyShutdownWatch {
    async fn recv(&self) -> ShutdownSignal {
        self.notify.notified().await;
        ShutdownSignal::FastShutdown
    }
}

/// Maximum time to wait for the proxy server thread to join on
/// [`ProxyGuard`] shutdown before giving up.
///
/// [`ProxyGuard`]: ProxyGuard
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Interval between checks that the proxy server thread has exited.
const JOIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Failure to stop and join a proxy server thread within its
/// bounded shutdown deadline.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProxyShutdownError {
    /// The producer did not finish before the shutdown deadline.
    #[error("proxy shutdown timed out")]
    Timeout,
    /// The producer exited without sending its normal completion signal.
    #[error("proxy shutdown completion channel disconnected")]
    CompletionDisconnected,
    /// The producer thread panicked.
    #[error("proxy server thread panicked during shutdown")]
    ThreadPanicked,
}

/// RAII guard that shuts down a Pingora proxy server when
/// dropped. Returned by [`start_proxy_with_registry`] and
/// related helpers so that test threads do not leak.
pub struct ProxyGuard {
    /// The address the proxy is listening on.
    addr: String,
    /// Handle to the spawned server thread, joined on drop.
    handle: Option<JoinHandle<()>>,
    /// Fires the shutdown signal on drop.
    notify: Arc<Notify>,
    /// Reports that the server returned normally from its producer thread.
    completion: mpsc::Receiver<()>,
    /// Whether the normal completion signal has already been received.
    completion_observed: bool,
    /// Whether the completion sender disconnected without sending.
    completion_disconnected: bool,
    /// Maximum duration for one shutdown attempt.
    join_timeout: Duration,
}

impl ProxyGuard {
    /// The proxy's listen address (e.g. `"127.0.0.1:12345"`).
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Signal the proxy to stop and wait for its producer thread
    /// to exit within a bounded deadline.
    ///
    /// A timeout retains the thread handle so a later call or the
    /// guard's drop fallback can make another bounded join attempt.
    ///
    /// # Errors
    ///
    /// Returns an error if the producer does not finish before the
    /// deadline, disconnects its completion channel, or panics.
    pub fn shutdown(&mut self) -> Result<(), ProxyShutdownError> {
        let deadline = Instant::now() + self.join_timeout;
        self.notify.notify_one();

        if self.handle.is_none() {
            return Ok(());
        }

        self.observe_completion(deadline)?;
        self.join_producer(deadline)
    }

    /// Waits for the normal producer completion signal once.
    fn observe_completion(&mut self, deadline: Instant) -> Result<(), ProxyShutdownError> {
        if !self.completion_observed && !self.completion_disconnected {
            match self.completion.recv_timeout(remaining_until(deadline)) {
                Ok(()) => self.completion_observed = true,
                Err(mpsc::RecvTimeoutError::Timeout) => return Err(ProxyShutdownError::Timeout),
                Err(mpsc::RecvTimeoutError::Disconnected) => self.completion_disconnected = true,
            }
        }
        Ok(())
    }

    /// Polls for thread completion within the shared deadline, then joins it.
    fn join_producer(&mut self, deadline: Instant) -> Result<(), ProxyShutdownError> {
        while self.handle.as_ref().is_some_and(|handle| !handle.is_finished()) {
            let remaining = remaining_until(deadline);
            if remaining.is_zero() {
                return Err(if self.completion_disconnected {
                    ProxyShutdownError::CompletionDisconnected
                } else {
                    ProxyShutdownError::Timeout
                });
            }
            std::thread::sleep(JOIN_POLL_INTERVAL.min(remaining));
        }

        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        if handle.join().is_err() {
            return Err(ProxyShutdownError::ThreadPanicked);
        }
        if self.completion_disconnected {
            return Err(ProxyShutdownError::CompletionDisconnected);
        }
        Ok(())
    }
}

/// Return the time remaining before `deadline` without underflow.
fn remaining_until(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// Builds a producer that cannot complete until the test releases it.
#[cfg(test)]
pub(crate) fn blocked_proxy_guard_for_test(join_timeout: Duration) -> (ProxyGuard, mpsc::Sender<()>) {
    let (release_tx, release_rx) = mpsc::channel();
    let (completion_tx, completion) = mpsc::sync_channel(1);
    let handle = std::thread::spawn(move || {
        release_rx.recv().expect("test should release blocked producer");
        let _sent = completion_tx.send(());
    });
    let guard = ProxyGuard {
        addr: "127.0.0.1:0".to_owned(),
        handle: Some(handle),
        notify: Arc::new(Notify::new()),
        completion,
        completion_observed: false,
        completion_disconnected: false,
        join_timeout,
    };
    (guard, release_tx)
}

impl fmt::Display for ProxyGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.addr)
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            tracing::warn!(
                addr = %self.addr,
                error = %error,
                timeout_secs = self.join_timeout.as_secs_f64(),
                "server thread did not shut down cleanly",
            );
        }
    }
}

/// Build a Pingora [`Server`] configured with all listeners
/// and the optional admin endpoint.
///
/// [`Server`]: pingora_core::server::Server
fn build_pingora_server(
    config: &Config,
    registry: &FilterRegistry,
    client: &praxis_core::subrequest::SubRequestClient,
) -> pingora_core::server::Server {
    let mut server = praxis_core::server::build_http_server(config.shutdown_timeout_secs, &RuntimeOptions::default());

    let mut cert_shutdowns = Vec::new();
    for listener in &config.listeners {
        let pipeline = Arc::new(ArcSwap::from(resolve_listener_pipeline(
            config, listener, registry, client,
        )));
        load_http_handler(&mut server, listener, pipeline, &mut cert_shutdowns).unwrap();
    }
    drop(cert_shutdowns);

    if let Some(admin_addr) = &config.admin.address {
        praxis_protocol::http::pingora::health::add_health_endpoint_to_pingora_server(
            &mut server,
            admin_addr,
            None,
            config.admin.verbose,
        );
    }

    server
}

/// Build a [`ProxyGuard`] by spawning a Pingora server that
/// shuts down when the guard is dropped.
fn spawn_proxy_server(
    config: &Config,
    registry: &FilterRegistry,
    client: &praxis_core::subrequest::SubRequestClient,
) -> ProxyGuard {
    let addr = config
        .listeners
        .first()
        .expect("config must have at least one listener")
        .address
        .clone();
    let server = build_pingora_server(config, registry, client);

    let notify = Arc::new(Notify::new());
    let watch_notify = Arc::clone(&notify);
    let (completion_tx, completion) = mpsc::sync_channel(1);

    let handle = std::thread::spawn(move || {
        server.run(RunArgs {
            shutdown_signal: Box::new(NotifyShutdownWatch { notify: watch_notify }),
        });
        let _sent = completion_tx.send(());
    });

    ProxyGuard {
        addr,
        handle: Some(handle),
        notify,
        completion,
        completion_observed: false,
        completion_disconnected: false,
        join_timeout: JOIN_TIMEOUT,
    }
}

// -----------------------------------------------------------------------------
// Proxy Startup
// -----------------------------------------------------------------------------

/// Start the proxy server in a background thread.
///
/// Returns a [`ProxyGuard`] that shuts down the server when
/// dropped. Use [`ProxyGuard::addr()`] to obtain the listen
/// address.
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
pub fn start_proxy(config: &Config) -> ProxyGuard {
    let client = configured_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&client);
    let guard = spawn_proxy_server(config, &registry, &client);
    crate::net::wait::wait_for_http(&guard.addr);
    guard
}

/// Start the proxy server without issuing an HTTP readiness request.
///
/// Returns a [`ProxyGuard`] immediately after spawning the server thread. The
/// caller must perform a readiness check appropriate for its trust boundary.
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
pub fn start_proxy_no_wait(config: &Config) -> ProxyGuard {
    let client = configured_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&client);
    spawn_proxy_server(config, &registry, &client)
}

/// Start the proxy with a custom filter registry.
///
/// Returns a [`ProxyGuard`] that shuts down the server when
/// dropped.
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
pub fn start_proxy_with_registry(config: &Config, registry: &FilterRegistry) -> ProxyGuard {
    let client = configured_subrequest_client(config);
    let guard = spawn_proxy_server(config, registry, &client);
    crate::net::wait::wait_for_http(&guard.addr);
    guard
}

/// Start a full proxy server (HTTP + TCP protocols) in a background thread.
pub fn start_full_proxy(config: Config) {
    std::thread::spawn(move || {
        praxis_ai::run_server(config, None);
    });
}

// -----------------------------------------------------------------------------
// Reloadable Proxy Guard
// -----------------------------------------------------------------------------

/// RAII guard for a proxy server with hot reload enabled.
///
/// Holds the temp config file so the watcher can detect
/// changes. The server thread runs until the process exits
/// (no clean shutdown; tests rely on process teardown).
pub struct ReloadableProxyGuard {
    /// The address the proxy is listening on.
    addr: String,

    /// Path to the config file (for mutation by tests).
    config_path: PathBuf,

    /// Keeps the temp file alive for the server's lifetime.
    _temp_file: tempfile::NamedTempFile,
}

impl ReloadableProxyGuard {
    /// The proxy's listen address.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Path to the config file for in-test mutation.
    pub fn config_path(&self) -> &std::path::Path {
        &self.config_path
    }

    /// Rewrite the config file with new YAML content.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be written.
    pub fn write_config(&self, yaml: &str) {
        std::fs::write(&self.config_path, yaml).expect("failed to write config file");
    }

    /// Rewrite config and wait for the debounce window.
    pub fn reload(&self, yaml: &str) {
        self.write_config(yaml);
        std::thread::sleep(RELOAD_SETTLE);
    }
}

/// Time to wait after writing a config file for the watcher
/// to debounce (500ms) and apply the reload.
const RELOAD_SETTLE: Duration = Duration::from_millis(1500);

/// Start a proxy with hot reload enabled by writing config
/// to a temp file and passing the path to the server.
///
/// Returns a guard with the listen address and config path.
/// Use [`ReloadableProxyGuard::reload`] to mutate the config
/// and wait for the change to take effect.
///
/// # Panics
///
/// Panics if the config cannot be parsed or the server fails
/// to start.
///
/// [`ReloadableProxyGuard::reload`]: ReloadableProxyGuard::reload
pub fn start_reloadable_proxy(yaml: &str) -> ReloadableProxyGuard {
    let config = Config::from_yaml(yaml).expect("test config should parse");
    let addr = config
        .listeners
        .first()
        .expect("config must have at least one listener")
        .address
        .clone();

    let mut temp_file = tempfile::NamedTempFile::new().expect("failed to create temp config file");
    std::io::Write::write_all(&mut temp_file, yaml.as_bytes()).expect("failed to write temp config");
    let config_path = temp_file.path().to_path_buf();

    let path_for_server = config_path.clone();
    std::thread::spawn(move || {
        praxis_ai::run_server(config, Some(path_for_server));
    });

    crate::net::wait::wait_for_http(&addr);

    ReloadableProxyGuard {
        addr,
        config_path,
        _temp_file: temp_file,
    }
}

/// Start an HTTP proxy with a TLS listener, waiting for HTTPS readiness before returning.
///
/// Uses the same server construction as [`start_proxy`] but
/// waits for TLS readiness instead of plain HTTP readiness.
///
/// Returns a [`ProxyGuard`] that shuts down the server when
/// dropped.
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
pub fn start_tls_proxy(config: &Config, client_config: &Arc<rustls::ClientConfig>) -> ProxyGuard {
    let client = configured_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&client);
    let guard = spawn_proxy_server(config, &registry, &client);
    crate::net::tls::wait_for_https(&guard.addr, client_config);
    guard
}

/// Start an HTTP proxy with a TLS listener without waiting for readiness.
///
/// Returns a [`ProxyGuard`] that shuts down the server when
/// dropped. The caller must wait for the proxy to become ready
/// using an appropriate readiness check.
///
/// # Panics
///
/// Panics if `config.listeners` is empty.
pub fn start_tls_proxy_no_wait(config: &Config) -> ProxyGuard {
    let client = configured_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&client);
    spawn_proxy_server(config, &registry, &client)
}

// -----------------------------------------------------------------------------
// YAML Config Test Utilities
// -----------------------------------------------------------------------------

/// Filter chain YAML: one listener, catch-all route, one backend.
pub fn simple_proxy_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// Filter chain YAML: one listener, a custom filter first,
/// then router + `load_balancer`.
pub fn custom_filter_yaml(proxy_port: u16, backend_port: u16, filter_name: &str) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: {filter_name}
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

// -----------------------------------------------------------------------------
// Registry Test Utilities
// -----------------------------------------------------------------------------

/// Build a [`FilterRegistry`] with the full AI registry plus
/// one custom test filter.
///
/// # Panics
///
/// Panics if the filter name conflicts with a builtin.
///
/// [`FilterRegistry`]: praxis_filter::FilterRegistry
pub fn registry_with(name: &str, make: fn() -> Box<dyn HttpFilter>) -> FilterRegistry {
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(name, FilterFactory::Http(Arc::new(move |_| Ok(make()))))
        .expect("duplicate filter name in test registry");
    registry
}

#[cfg(test)]
mod tests {
    use std::{
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread::JoinHandle,
        time::Duration,
    };

    use praxis_core::config::Config;
    use tokio::sync::Notify;

    use super::{ProxyGuard, ProxyShutdownError, simple_proxy_yaml, start_proxy};
    use crate::free_port_guard;

    #[test]
    fn explicit_shutdown_joins_real_proxy_and_releases_listener() {
        let proxy_port = free_port_guard().release();
        let backend_port = free_port_guard().release();
        let config =
            Config::from_yaml(&simple_proxy_yaml(proxy_port, backend_port)).expect("test proxy config should parse");
        let mut proxy = start_proxy(&config);

        proxy
            .shutdown()
            .expect("explicit shutdown should join the proxy thread");

        let rebound = TcpListener::bind(("127.0.0.1", proxy_port))
            .expect("joined proxy must release its listener before success");
        assert_eq!(rebound.local_addr().unwrap().port(), proxy_port);
        proxy
            .shutdown()
            .expect("explicit shutdown should be idempotent after join");
    }

    #[test]
    fn drop_fallback_joins_real_proxy_and_releases_listener() {
        let proxy_port = free_port_guard().release();
        let backend_port = free_port_guard().release();
        let config =
            Config::from_yaml(&simple_proxy_yaml(proxy_port, backend_port)).expect("test proxy config should parse");
        let proxy = start_proxy(&config);

        drop(proxy);

        let rebound = TcpListener::bind(("127.0.0.1", proxy_port))
            .expect("drop fallback must release its listener before returning");
        assert_eq!(rebound.local_addr().unwrap().port(), proxy_port);
    }

    #[test]
    fn explicit_shutdown_timeout_never_reports_an_unjoined_thread_as_success() {
        let exited = Arc::new(AtomicBool::new(false));
        let thread_exited = Arc::clone(&exited);
        let (release_tx, release_rx) = mpsc::channel();
        let (completion_tx, completion_rx) = mpsc::sync_channel(1);
        let handle = std::thread::spawn(move || {
            release_rx.recv().expect("test should release blocked producer");
            thread_exited.store(true, Ordering::Release);
            let _sent = completion_tx.send(());
        });
        let mut proxy = test_proxy_guard(handle, completion_rx, Duration::from_millis(20));

        let error = proxy.shutdown().expect_err("blocked producer must time out");

        assert_eq!(error, ProxyShutdownError::Timeout);
        assert!(
            !exited.load(Ordering::Acquire),
            "timeout must not claim the thread joined"
        );
        release_tx.send(()).unwrap();
        proxy
            .shutdown()
            .expect("a later bounded retry should join after release");
        assert!(exited.load(Ordering::Acquire));
    }

    #[test]
    fn explicit_shutdown_reports_completion_disconnect() {
        let (completion_tx, completion_rx) = mpsc::sync_channel(1);
        drop(completion_tx);
        let handle = std::thread::spawn(|| {});
        let mut proxy = test_proxy_guard(handle, completion_rx, Duration::from_secs(1));

        let error = proxy
            .shutdown()
            .expect_err("missing completion signal must be an error");

        assert_eq!(error, ProxyShutdownError::CompletionDisconnected);
    }

    #[test]
    fn explicit_shutdown_reports_producer_panic() {
        let (completion_tx, completion_rx) = mpsc::sync_channel(1);
        let handle = std::thread::spawn(move || {
            let _completion_tx = completion_tx;
            panic!("injected proxy thread panic");
        });
        let mut proxy = test_proxy_guard(handle, completion_rx, Duration::from_secs(1));

        let error = proxy.shutdown().expect_err("producer panic must be an error");

        assert_eq!(error, ProxyShutdownError::ThreadPanicked);
    }

    fn test_proxy_guard(handle: JoinHandle<()>, completion: mpsc::Receiver<()>, join_timeout: Duration) -> ProxyGuard {
        ProxyGuard {
            addr: "127.0.0.1:0".to_owned(),
            handle: Some(handle),
            notify: Arc::new(Notify::new()),
            completion,
            completion_observed: false,
            completion_disconnected: false,
            join_timeout,
        }
    }
}
