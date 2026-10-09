// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The #1288 workload: metered, streaming, keep-alive clients.
//!
//! Every request makes two metering callouts (a balance check before it is
//! proxied and a usage report after its response), so these tests run the
//! real `praxis-ai` binary against a counting mock metering service and check
//! the child's own descriptor table.

use std::{
    io::{BufRead as _, BufReader, Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use praxis_test_utils::{
    Backend, PraxisProcess, collect_responses, concurrent_gets, free_port, http_get, idle_keepalive_connections,
    is_closed_by_peer, open_requests, start_keepalive_backend,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Slack for descriptors the runtime opens and closes on its own.
const MARGIN: usize = 32;

/// Interval between descriptor samples while load runs.
const SAMPLE_EVERY: Duration = Duration::from_millis(20);

/// Log text of `EMFILE` under glibc and musl.
const EMFILE_TEXTS: [&str; 2] = ["Too many open files", "No file descriptors available"];

/// Balance check answer granting access.
const BALANCE: &str = r#"{"hasAccess": true, "balance": 9000.0}"#;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn issue_1288_metered_streaming_clients_fit_the_container_limit() {
    let _serial = crate::serial();
    let metering = MeteringService::start();
    let backend = Backend::chunked(vec!["data: token\n\n".to_owned(); 5])
        .chunk_delay(Duration::from_millis(20))
        .start_with_shutdown();
    let setup = Setup {
        runtime_lines: "max_open_files: 1024",
        listener_lines: "downstream_keepalive_timeout_ms: 5000",
        cluster_lines: "idle_timeout_ms: 5000",
        ..Setup::new(backend.port(), metering.url())
    };
    let mut proxy = Proxy::start(&setup, &metering);

    let (peak, report) = proxy
        .process
        .peak_open_fds_during(SAMPLE_EVERY, || concurrent_gets(&proxy.addr, "/", 128, 25, true));

    assert!(
        report.only(&[200]) && report.total() == 3_200,
        "128 metered streaming clients fit a 1024 limit without a single failure or shed: {report:?}"
    );
    assert_eq!(proxy.overload_rejects(), 0, "nothing may be shed within the budget");
    assert!(
        peak <= proxy.baseline + 4 * 128 + MARGIN,
        "each client costs a client, an upstream, and at most two metering descriptors: peak {peak}, baseline {}",
        proxy.baseline
    );
    assert_eq!(
        metering.balance_checks() - proxy.warm_up_calls,
        3_200,
        "every request must pass a balance check"
    );
    assert!(
        wait_until(Duration::from_secs(10), || metering.usage_reports()
            - proxy.warm_up_calls
            == 3_200),
        "every response must be reported, got {}",
        metering.usage_reports() - proxy.warm_up_calls
    );
    let logs = proxy.shut_down();
    assert_no_emfile(&logs);
    assert_metering_never_failed(&logs);
}

#[test]
fn issue_1288_metered_requests_are_shed_not_failed_at_the_limit() {
    let _serial = crate::serial();
    let metering = MeteringService::start();
    let backend = start_keepalive_backend("slow", Duration::from_millis(500));
    let setup = Setup {
        runtime_lines: "max_open_files: 256",
        fail_open: false,
        ..Setup::new(backend.port(), metering.url())
    };
    let mut proxy = Proxy::start(&setup, &metering);

    let report = collect_responses(open_requests(&proxy.addr, "/", 150));

    assert!(
        report.only(&[200, 503]),
        "past the limit requests must be shed with a retryable 503, never fail: {report:?}"
    );
    assert!(
        report.count(503) > 0,
        "150 metered in-flight requests must exceed 256 descriptors: {report:?}"
    );
    assert!(
        report.count(200) > 0,
        "requests within the limit must succeed: {report:?}"
    );
    assert!(proxy.overload_rejects() > 0, "sheds must be counted");
    assert_eq!(
        metering.balance_checks() - proxy.warm_up_calls,
        report.count(200),
        "admitted requests are metered and shed ones never reach the metering service: {report:?}"
    );
    assert!(
        proxy.eventually_serves(Duration::from_secs(5)),
        "the proxy must serve again once the burst drains"
    );
    let logs = proxy.shut_down();
    assert_no_emfile(&logs);
    assert_metering_never_failed(&logs);
}

#[test]
fn metering_callouts_share_the_dns_cache() {
    let _serial = crate::serial();
    let backend = start_keepalive_backend("ok", Duration::ZERO);
    let setup = Setup {
        fail_open: true,
        ..Setup::new(backend.port(), "http://praxis-ai-fd-metering.invalid:9090".to_owned())
    };
    let (port, admin_port) = (free_port(), free_port());
    let addr = format!("127.0.0.1:{port}");
    let mut process = PraxisProcess::spawn(&config(port, admin_port, &setup), &addr);

    // A lookup that fails through the shared cache is remembered, so a later
    // callout reports "recently failed"; resolving per call never does. Keep
    // sending requests until both callouts have said so, since a resolver slow
    // enough to outlast the first callouts' deadline delays the cached failure.
    for index in 0..5 {
        let report = concurrent_gets(&addr, "/", 1, 1, false);
        assert!(report.only(&[200]), "request {index} must fail open: {report:?}");
        let reported = wait_until(Duration::from_secs(10), || {
            process.logs().matches("metering usage report failed").count() > index
        });
        assert!(
            reported,
            "request {index} must attempt its usage report:\n{}",
            process.logs()
        );
        let logs = process.logs();
        if reused_cached_failure(&logs, "metering usage report failed")
            && reused_cached_failure(&logs, "balance check unreachable")
        {
            break;
        }
    }
    let status = process.terminate();
    let logs = process.logs();

    assert!(
        status.success(),
        "graceful shutdown should exit zero ({status}):\n{logs}"
    );
    assert!(
        reused_cached_failure(&logs, "metering usage report failed"),
        "usage reports must reuse the cached lookup:\n{logs}"
    );
    assert!(
        reused_cached_failure(&logs, "balance check unreachable"),
        "balance checks must reuse the cached lookup:\n{logs}"
    );
}

#[test]
fn idle_metered_clients_release_their_descriptors() {
    let _serial = crate::serial();
    let metering = MeteringService::start();
    let backend = start_keepalive_backend("ok", Duration::ZERO);
    let setup = Setup {
        listener_lines: "downstream_keepalive_timeout_ms: 1000",
        cluster_lines: "idle_timeout_ms: 500",
        ..Setup::new(backend.port(), metering.url())
    };
    let proxy = Proxy::start(&setup, &metering);

    let idle = idle_keepalive_connections(&proxy.addr, "/", 100);
    let held = proxy.process.open_fds();
    let all_closed = wait_until(Duration::from_secs(8), || idle.iter().all(is_closed_by_peer));
    let settled = proxy
        .process
        .wait_open_fds_at_most(proxy.baseline + 8, Duration::from_secs(5));

    assert!(
        held >= proxy.baseline + 95,
        "100 idle keep-alive clients hold their descriptors at first: {held}, baseline {}",
        proxy.baseline
    );
    assert!(all_closed, "the proxy must close each idle client connection");
    assert!(
        settled <= proxy.baseline + 8,
        "idle timeouts must hand every descriptor back: {settled}, baseline {}",
        proxy.baseline
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Proxy configuration under test.
struct Setup<'cfg> {
    /// Upstream port on loopback.
    upstream_port: u16,

    /// Metering service base URL.
    metering_url: String,

    /// Metering `fail_open`.
    fail_open: bool,

    /// Extra lines under the cluster.
    cluster_lines: &'cfg str,

    /// Extra lines under the listener.
    listener_lines: &'cfg str,

    /// Extra lines under `runtime:`.
    runtime_lines: &'cfg str,
}

impl Setup<'_> {
    /// Defaults: one worker thread, metering failing closed, nothing extra.
    fn new(upstream_port: u16, metering_url: String) -> Self {
        Self {
            upstream_port,
            metering_url,
            fail_open: false,
            cluster_lines: "",
            listener_lines: "",
            runtime_lines: "",
        }
    }
}

/// A running proxy and its descriptor baseline.
struct Proxy {
    /// Proxy listener address.
    addr: String,

    /// Admin address.
    admin: String,

    /// Open descriptors after startup and one proxied request.
    baseline: usize,

    /// Metering callouts of each kind made by the warm-up request.
    warm_up_calls: usize,

    /// The child process.
    process: PraxisProcess,
}

impl Proxy {
    /// Start the proxy for `setup`, send one metered request, and record the
    /// baseline.
    fn start(setup: &Setup<'_>, metering: &MeteringService) -> Self {
        let (port, admin_port) = (free_port(), free_port());
        let addr = format!("127.0.0.1:{port}");
        let process = PraxisProcess::spawn(&config(port, admin_port, setup), &addr);
        let warm = concurrent_gets(&addr, "/", 1, 1, false);
        assert!(warm.only(&[200]), "warm-up request: {warm:?}\n{}", process.logs());
        assert!(
            wait_until(Duration::from_secs(5), || metering.usage_reports() == 1),
            "the warm-up request must be reported:\n{}",
            process.logs()
        );
        assert_eq!(metering.balance_checks(), 1, "the warm-up request must be metered");
        std::thread::sleep(Duration::from_millis(300));
        let baseline = process.open_fds();
        Self {
            addr,
            admin: format!("127.0.0.1:{admin_port}"),
            baseline,
            warm_up_calls: 1,
            process,
        }
    }

    /// Poll `GET /` until it returns 200.
    fn eventually_serves(&self, timeout: Duration) -> bool {
        wait_until(timeout, || concurrent_gets(&self.addr, "/", 1, 1, false).only(&[200]))
    }

    /// `praxis_overload_rejects_total{reason="file_descriptors"}`.
    fn overload_rejects(&self) -> u64 {
        let (_, body) = http_get(&self.admin, "/metrics", None);
        body.lines()
            .find_map(|line| line.strip_prefix("praxis_overload_rejects_total{reason=\"file_descriptors\"} "))
            .and_then(|value| value.trim().parse::<f64>().ok())
            .map_or(0, |value| value as u64)
    }

    /// Gracefully stop the proxy, assert a clean exit, and return its logs.
    fn shut_down(&mut self) -> String {
        let status = self.process.terminate();
        let logs = self.process.logs();
        assert!(
            status.success(),
            "graceful shutdown should exit zero ({status}):\n{logs}"
        );
        logs
    }
}

/// Metered proxy config for `setup` on `port`, metrics on `admin`.
fn config(port: u16, admin: u16, setup: &Setup<'_>) -> String {
    format!(
        r#"
shutdown_timeout_secs: 1
admin:
  metrics_address: "127.0.0.1:{admin}"
runtime:
  threads: 1
  {runtime}
listeners:
  - name: web
    address: "127.0.0.1:{port}"
    filter_chains: [main]
    {listener}
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: external_metering
        metering_url: "{metering_url}"
        allow_private_endpoint: true
        timeout_seconds: 5
        feature_key: "inference-tokens"
        source: "ai-gateway"
        fail_open: {fail_open}
        default_username: "tenant"
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{upstream}"
            {cluster}
insecure_options:
  allow_private_endpoints: true
  allow_private_upstreams: true
"#,
        runtime = setup.runtime_lines,
        listener = setup.listener_lines,
        metering_url = setup.metering_url,
        fail_open = setup.fail_open,
        upstream = setup.upstream_port,
        cluster = setup.cluster_lines,
    )
}

/// A keep-alive mock metering service that grants every balance check and
/// accepts every usage report, counting both.
struct MeteringService {
    /// Port on loopback.
    port: u16,

    /// Balance checks answered.
    balance_checks: Arc<AtomicUsize>,

    /// Usage reports accepted.
    usage_reports: Arc<AtomicUsize>,
}

impl MeteringService {
    /// Listen on an ephemeral loopback port, on IPv6 too where available so
    /// `localhost` never costs a refused connect.
    fn start() -> Self {
        let v4 = TcpListener::bind("127.0.0.1:0").expect("bind metering service");
        let port = v4.local_addr().expect("metering service address").port();
        let service = Self {
            port,
            balance_checks: Arc::new(AtomicUsize::new(0)),
            usage_reports: Arc::new(AtomicUsize::new(0)),
        };
        let listeners = std::iter::once(v4).chain(TcpListener::bind(("::1", port)).ok());
        for listener in listeners {
            let counters = (Arc::clone(&service.balance_checks), Arc::clone(&service.usage_reports));
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let counters = (Arc::clone(&counters.0), Arc::clone(&counters.1));
                    std::thread::spawn(move || serve_metering(stream, &counters.0, &counters.1));
                }
            });
        }
        service
    }

    /// Base URL by hostname, so callouts resolve it.
    fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }

    /// Balance checks answered so far.
    fn balance_checks(&self) -> usize {
        self.balance_checks.load(Ordering::SeqCst)
    }

    /// Usage reports accepted so far.
    fn usage_reports(&self) -> usize {
        self.usage_reports.load(Ordering::SeqCst)
    }
}

/// Answer metering requests on one connection until the client closes it.
fn serve_metering(stream: TcpStream, balance_checks: &AtomicUsize, usage_reports: &AtomicUsize) {
    let mut reader = BufReader::new(stream);
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
            return;
        }
        let mut content_length = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0_u8; content_length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        let response = if request_line.starts_with("GET /api/v1/customers/") {
            balance_checks.fetch_add(1, Ordering::SeqCst);
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{BALANCE}",
                BALANCE.len()
            )
        } else if request_line.starts_with("POST /api/v1/events ") {
            usage_reports.fetch_add(1, Ordering::SeqCst);
            "HTTP/1.1 204 No Content\r\n\r\n".to_owned()
        } else {
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_owned()
        };
        if reader.get_mut().write_all(response.as_bytes()).is_err() {
            return;
        }
    }
}

/// Poll `done` every 50 ms until it holds or `timeout` passes.
fn wait_until<F: Fn() -> bool>(timeout: Duration, done: F) -> bool {
    let deadline = Instant::now() + timeout;
    while !done() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    true
}

/// Whether a `callout` log line reports a failure served from the shared DNS
/// cache.
fn reused_cached_failure(logs: &str, callout: &str) -> bool {
    logs.lines()
        .any(|line| line.contains(callout) && line.contains("recently failed"))
}

/// Assert no log line reports running out of descriptors.
fn assert_no_emfile(logs: &str) {
    for text in EMFILE_TEXTS {
        assert!(
            !logs.contains(text),
            "the proxy must never hit EMFILE ({text}):\n{logs}"
        );
    }
}

/// Assert every metering callout completed.
fn assert_metering_never_failed(logs: &str) {
    for text in ["balance check unreachable", "metering usage report failed"] {
        assert!(!logs.contains(text), "no metering callout may fail ({text}):\n{logs}");
    }
}
