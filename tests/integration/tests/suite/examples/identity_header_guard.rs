// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the identity header guard example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_body, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn identity_header_guard_config_parses() {
    let config = super::load_example_config(
        "identity-header-guard.yaml",
        29920,
        HashMap::from([("127.0.0.1:3000", 29921_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn identity_header_guard_strips_identity_headers() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "identity-header-guard.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         x-tenant-username: yossi\r\n\
         x-tenant-group: ai-eng\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw).to_lowercase();
    assert!(
        !body.contains("x-tenant-username"),
        "upstream should NOT receive x-tenant-username: {body}"
    );
    assert!(
        !body.contains("x-tenant-group"),
        "upstream should NOT receive x-tenant-group: {body}"
    );
}

#[test]
fn identity_header_guard_passes_non_matching_headers() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "identity-header-guard.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         x-custom-header: should-pass\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw).to_lowercase();
    assert!(
        body.contains("x-custom-header"),
        "upstream should receive non-matching headers: {body}"
    );
}
