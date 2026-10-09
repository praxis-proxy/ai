// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The production binary must register admin and metrics listeners independently.

use praxis_test_utils::{PraxisProcess, free_port, http_get};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// The built-in configuration must retain the container health endpoint.
const DEFAULT_CONFIG: &str = include_str!("../../../../server/src/default.yaml");

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn default_configuration_keeps_the_container_health_probe() {
    let metrics = address();
    let config = DEFAULT_CONFIG
        .replace("127.0.0.1:8080", &address())
        .replace("127.0.0.1:9901", &metrics);
    let proxy = PraxisProcess::spawn(&config, &metrics);

    assert_eq!(http_get(&metrics, "/healthy", None).0, 200, "{}", proxy.logs());
}

#[test]
fn admin_only_listener_keeps_the_kv_api() {
    let admin = address();
    let proxy = PraxisProcess::spawn(&config(Some(&admin), None), &admin);

    assert_kv_api(&admin, &proxy);
    assert_eq!(http_get(&admin, "/healthy", None).0, 404);
}

#[test]
fn metrics_only_listener_keeps_health_without_admin_routes() {
    let metrics = address();
    let proxy = PraxisProcess::spawn(&config(None, Some(&metrics)), &metrics);

    assert_eq!(http_get(&metrics, "/healthy", None).0, 200, "{}", proxy.logs());
    assert_eq!(http_get(&metrics, "/metrics", None).0, 200, "{}", proxy.logs());
    assert_eq!(
        http_get(&metrics, "/api/kv/missing", None).1,
        r#"{"error":"not found"}"#
    );
}

#[test]
fn separate_listeners_keep_admin_and_metrics_routes_separate() {
    let admin = address();
    let metrics = address();
    let proxy = PraxisProcess::spawn(&config(Some(&admin), Some(&metrics)), &admin);

    assert_kv_api(&admin, &proxy);
    assert_eq!(http_get(&metrics, "/healthy", None).0, 200, "{}", proxy.logs());
    assert_eq!(http_get(&admin, "/healthy", None).0, 404);
    assert_eq!(
        http_get(&metrics, "/api/kv/missing", None).1,
        r#"{"error":"not found"}"#
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// A missing store distinguishes the registered KV API from an absent route.
fn assert_kv_api(admin: &str, proxy: &PraxisProcess) {
    let (status, body) = http_get(admin, "/api/kv/missing", None);
    assert_eq!(status, 404, "{}", proxy.logs());
    assert_eq!(body, r#"{"error":"store not found"}"#, "{}", proxy.logs());
}

/// Choose a loopback address for one listener.
fn address() -> String {
    format!("127.0.0.1:{}", free_port())
}

/// Isolate the binary's listener registration from upstream networking.
fn config(admin: Option<&str>, metrics: Option<&str>) -> String {
    let admin_line = admin.map_or_else(String::new, |addr| format!("  address: {addr}\n"));
    let metrics_line = metrics.map_or_else(String::new, |addr| format!("  metrics_address: {addr}\n"));
    let listener = address();
    format!(
        r#"
admin:
{admin_line}{metrics_line}
listeners:
  - name: web
    address: "{listener}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
        body: ok
"#
    )
}
