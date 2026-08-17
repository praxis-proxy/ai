// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the external metering example configuration.

use std::collections::HashMap;

use praxis_test_utils::{RoutedBackend, free_port, http_send, parse_body, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn external_metering_config_parses() {
    let metering_port = free_port();
    let config = super::load_example_config(
        "external-metering.yaml",
        29800,
        HashMap::from([("127.0.0.1:3000", 29801_u16), ("127.0.0.1:9090", metering_port)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

// -----------------------------------------------------------------------------
// Balance Check — Access Granted
// -----------------------------------------------------------------------------

#[test]
fn external_metering_allows_request_when_balance_available() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let balance_body = r#"{"hasAccess": true, "balance": 9000.0}"#;
    let metering_port = RoutedBackend::new()
        .route("/api/v1/customers", 200, balance_body)
        .route("/api/v1/events", 204, "")
        .start();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy request when balance available");
}

// -----------------------------------------------------------------------------
// Fail-Closed — Metering Unavailable
// -----------------------------------------------------------------------------

#[test]
fn external_metering_rejects_when_fail_closed_and_metering_down() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // Point metering_url at a port with nothing listening
    let metering_port = free_port();

    let path = praxis_test_utils::example_config_path("external-metering.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap();
    let patched = praxis_test_utils::patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    )
    .replace("fail_open: true", "fail_open: false");
    let config = praxis_core::config::Config::from_yaml(&patched).unwrap();

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        503,
        "should reject with 503 when metering is unavailable and fail_open=false"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("metering system unavailable"),
        "rejection body should mention metering unavailable: {body}"
    );
}

// -----------------------------------------------------------------------------
// Identity Headers Stripped
// -----------------------------------------------------------------------------

#[test]
fn external_metering_strips_tenant_and_auth_headers() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let balance_body = r#"{"hasAccess": true, "balance": 9000.0}"#;
    let metering_port = RoutedBackend::new()
        .route("/api/v1/customers", 200, balance_body)
        .route("/api/v1/events", 204, "")
        .start();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Authorization: Bearer sk-client-secret\r\n\
         x-api-key: client-key\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy successfully");
    let body = parse_body(&raw);
    assert!(
        !body.contains("x-tenant-username"),
        "tenant header should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("sk-client-secret"),
        "authorization should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("client-key"),
        "x-api-key should be stripped from upstream: {body}"
    );
}

// -----------------------------------------------------------------------------
// No Identity — Metering Skipped
// -----------------------------------------------------------------------------

#[test]
fn external_metering_skips_when_no_identity() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    // No metering mock needed — filter skips entirely without identity
    let metering_port = free_port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:9090", metering_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "should proxy without metering when no identity headers"
    );
}
