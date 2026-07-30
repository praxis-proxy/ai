// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the external metering example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_body, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn external_metering_config_parses() {
    let config = super::load_example_config(
        "external-metering.yaml",
        29800,
        HashMap::from([("127.0.0.1:3000", 29801_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

// -----------------------------------------------------------------------------
// Header Removal
// -----------------------------------------------------------------------------

#[test]
fn external_metering_strips_tenant_and_credential_headers() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-tenant-group: engineering\r\n\
         Authorization: Bearer redacted-client-token\r\n\
         x-api-key: redacted-client-key\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy successfully");
    let body = parse_body(&raw);
    assert!(
        !body.contains("x-tenant-username"),
        "tenant header should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("x-tenant-group"),
        "tenant header should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("redacted-client-token"),
        "authorization should be stripped from upstream: {body}"
    );
    assert!(
        !body.contains("redacted-client-key"),
        "x-api-key should be stripped from upstream: {body}"
    );
}

#[test]
fn external_metering_forwards_unrelated_headers() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-tenant-username: alice\r\n\
         x-request-trace: keep-me\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should proxy successfully");
    assert!(
        parse_body(&raw).contains("x-request-trace"),
        "unrelated headers should reach the upstream"
    );
}

#[test]
fn external_metering_proxies_requests_without_tenant_headers() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "external-metering.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
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
        "should proxy when no tenant headers are present"
    );
}
