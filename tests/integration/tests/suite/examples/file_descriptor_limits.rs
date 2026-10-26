// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! File descriptor limits example tests.
//!
//! The open file limit belongs to the whole process, so these run the real
//! `praxis-ai` binary against the example config. Timeouts and limits are
//! shortened where the example's production values would take minutes or
//! thousands of connections to observe.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use praxis_test_utils::{
    PraxisProcess, RoutedBackend, allow_loopback_endpoints, collect_responses, concurrent_gets, example_config_path,
    free_port, http_get, idle_keepalive_connections, is_closed_by_peer, open_requests, own_open_file_limits,
    patch_yaml, read_raw_responses, start_keepalive_backend,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// The example under test.
const EXAMPLE: &str = "file-descriptor-limits.yaml";

/// Balance check answer granting access.
const BALANCE: &str = r#"{"hasAccess": true, "balance": 9000.0}"#;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn file_descriptor_limits_example_pins_the_limit_and_meters() {
    let backend = start_keepalive_backend("ok", Duration::ZERO);
    let metering = granting_metering_service();
    let (port, admin) = (free_port(), free_port());
    let mut proxy = PraxisProcess::spawn(&example_yaml(port, admin, backend.port(), metering, &[]), &addr(port));
    let (_, hard) = own_open_file_limits();
    let pinned = hard.min(65_536);

    assert_eq!(
        proxy.open_file_limits(),
        (pinned, hard),
        "max_open_files: 65536 pins the soft limit, clamped to the hard limit"
    );
    assert!(
        concurrent_gets(&addr(port), "/", 1, 1, false).only(&[200]),
        "the example proxies once the failing-closed balance check grants access:\n{}",
        proxy.logs()
    );
    assert!(
        wait_for_scrape(admin, &format!("praxis_process_max_fds {pinned}")),
        "the limit is exported as a gauge"
    );
    let logs = shut_down(&mut proxy);
    assert!(
        hard < 65_536 || !logs.contains("open file limit is low"),
        "20000 connections fit a 65536 descriptor limit without a warning:\n{logs}"
    );
}

#[test]
fn file_descriptor_limits_example_closes_idle_connections() {
    let backend = start_keepalive_backend("ok", Duration::ZERO);
    let metering = granting_metering_service();
    let (port, admin) = (free_port(), free_port());
    let yaml = example_yaml(
        port,
        admin,
        backend.port(),
        metering,
        &[
            (
                "downstream_keepalive_timeout_ms: 60000",
                "downstream_keepalive_timeout_ms: 1000",
            ),
            ("idle_timeout_ms: 30000", "idle_timeout_ms: 500"),
        ],
    );
    let proxy = PraxisProcess::spawn(&yaml, &addr(port));
    assert!(
        concurrent_gets(&addr(port), "/", 1, 1, false).only(&[200]),
        "warm-up request"
    );
    std::thread::sleep(Duration::from_millis(300));
    let before = proxy.open_fds();

    let idle = idle_keepalive_connections(&addr(port), "/", 20);
    let held = proxy.open_fds();
    let all_closed = wait_until(Duration::from_secs(5), || idle.iter().all(is_closed_by_peer));
    let settled = proxy.wait_open_fds_at_most(before + 4, Duration::from_secs(5));

    assert!(
        held >= before + 20,
        "20 idle clients hold descriptors: {held}, before {before}"
    );
    assert!(all_closed, "the keep-alive timeout closes every idle client");
    assert!(
        settled <= before + 4,
        "idle clients and pooled upstreams are released: {settled}, before {before}"
    );
}

#[test]
fn file_descriptor_limits_example_sheds_near_the_limit() {
    let backend = start_keepalive_backend("ok", Duration::from_secs(1));
    let metering = granting_metering_service();
    let (port, admin) = (free_port(), free_port());
    let yaml = example_yaml(
        port,
        admin,
        backend.port(),
        metering,
        &[("max_open_files: 65536", "max_open_files: 256")],
    );
    let proxy = PraxisProcess::spawn(&yaml, &addr(port));

    let responses = read_raw_responses(open_requests(&addr(port), "/", 150));
    let status_of = |raw: &str| raw.split_whitespace().nth(1).map(str::to_owned);
    let ok = responses
        .iter()
        .filter(|(raw, _)| status_of(raw).as_deref() == Some("200"))
        .count();
    let shed: Vec<&String> = responses
        .iter()
        .map(|(raw, _)| raw)
        .filter(|raw| status_of(raw).as_deref() == Some("503"))
        .collect();

    assert_eq!(ok + shed.len(), 150, "every request gets a 200 or a 503: {responses:?}");
    assert!(ok > 0, "requests within the limit are served");
    assert!(
        !shed.is_empty(),
        "150 metered in-flight requests exceed a 256 descriptor limit"
    );
    assert!(
        shed.iter()
            .all(|raw| raw.to_ascii_lowercase().contains("retry-after: 1")),
        "every 503 is a shed that tells the client when to retry, never a failed balance check:\n{}",
        proxy.logs()
    );
    let (_, metrics) = http_get(&addr(admin), "/metrics", None);
    assert!(
        metrics.contains("praxis_overload_rejects_total{reason=\"file_descriptors\"}"),
        "sheds are counted:\n{metrics}"
    );
}

#[test]
fn file_descriptor_limits_example_caps_concurrent_requests() {
    let backend = start_keepalive_backend("ok", Duration::from_millis(500));
    let metering = granting_metering_service();
    let (port, admin) = (free_port(), free_port());
    let yaml = example_yaml(
        port,
        admin,
        backend.port(),
        metering,
        &[("max_connections: 20000", "max_connections: 2")],
    );
    let _proxy = PraxisProcess::spawn(&yaml, &addr(port));

    let report = collect_responses(open_requests(&addr(port), "/", 8));

    assert!(report.only(&[200, 503]), "requests past the limit get 503: {report:?}");
    assert!(
        report.count(200) >= 2 && report.count(503) > 0,
        "two of eight concurrent requests fit a limit of 2: {report:?}"
    );
    let (_, metrics) = http_get(&addr(admin), "/metrics", None);
    assert!(
        metrics.contains("praxis_overload_rejects_total{reason=\"global_connections\"}"),
        "connection limit rejects are counted:\n{metrics}"
    );
}

#[test]
fn file_descriptor_limits_example_caps_concurrent_callouts() {
    let backend = start_keepalive_backend("ok", Duration::ZERO);
    let metering = start_keepalive_backend(BALANCE, Duration::from_millis(300));
    let (port, admin) = (free_port(), free_port());
    let yaml = example_yaml(
        port,
        admin,
        backend.port(),
        metering.port(),
        &[("subrequest_max_connections: 8192", "subrequest_max_connections: 1")],
    );
    let _proxy = PraxisProcess::spawn(&yaml, &addr(port));

    let started = Instant::now();
    let report = collect_responses(open_requests(&addr(port), "/", 4));
    let elapsed = started.elapsed();

    assert!(
        report.only(&[200]) && report.total() == 4,
        "callouts past the cap wait for a slot within their timeout: {report:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(1_100),
        "one callout slot runs four 300 ms balance checks one after another, took {elapsed:?}"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// A metering service that grants every balance check and accepts every
/// usage report.
fn granting_metering_service() -> u16 {
    RoutedBackend::new()
        .route("/api/v1/customers", 200, BALANCE)
        .route("/api/v1/events", 204, "")
        .start()
}

/// The example config on test ports with `replacements` applied and a short
/// shutdown grace period, so stopping it does not wait out the default.
///
/// The metering URL keeps its `localhost` hostname so callouts resolve it.
fn example_yaml(port: u16, admin: u16, backend: u16, metering: u16, replacements: &[(&str, &str)]) -> String {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read example");
    let ports = HashMap::from([("127.0.0.1:9901", admin), ("127.0.0.1:3000", backend)]);
    let mut patched = allow_loopback_endpoints(&patch_yaml(&yaml, port, &ports));
    let metering_host = "localhost:9090";
    assert!(
        patched.contains(metering_host),
        "the example must meter via {metering_host}"
    );
    patched = patched.replacen(metering_host, &format!("localhost:{metering}"), 1);
    for (from, to) in replacements {
        assert!(patched.contains(from), "the example must contain {from:?}");
        patched = patched.replacen(from, to, 1);
    }
    format!("shutdown_timeout_secs: 1\n{patched}")
}

/// Loopback address for `port`.
fn addr(port: u16) -> String {
    format!("127.0.0.1:{port}")
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

/// Scrape `/metrics` on `admin` until it contains `needle`.
fn wait_for_scrape(admin: u16, needle: &str) -> bool {
    wait_until(Duration::from_secs(5), || {
        http_get(&addr(admin), "/metrics", None).1.contains(needle)
    })
}

/// Gracefully stop `proxy`, assert a clean exit, and return its logs.
fn shut_down(proxy: &mut PraxisProcess) -> String {
    let status = proxy.terminate();
    let logs = proxy.logs();
    assert!(
        status.success(),
        "graceful shutdown should exit zero ({status}):\n{logs}"
    );
    logs
}
