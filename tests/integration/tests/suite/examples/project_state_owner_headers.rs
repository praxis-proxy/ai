// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the trusted state-owner header projection example.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_body, parse_status, start_header_echo_backend};

const EXAMPLE: &str = "project-state-owner-headers.yaml";

#[test]
fn project_state_owner_headers_config_parses() {
    let config = super::load_example_config(EXAMPLE, 29930, HashMap::from([("127.0.0.1:3000", 29931_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have one listener");
    assert_eq!(&*config.listeners[0].name, "gateway");
}

#[test]
fn project_state_owner_headers_replaces_ingress_assertions_with_ogx_contract() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();
    let config = super::load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
    );
    let proxy = praxis_test_utils::start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "POST /v1/responses HTTP/1.1\r\n\
         Host: localhost\r\n\
         x-auth-tenant: tenant-a\r\n\
         x-auth-user: alice\r\n\
         x-tenant-id: spoofed-tenant\r\n\
         x-user-id: spoofed-user\r\n\
         Content-Length: 0\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200);
    let body = parse_body(&raw).to_ascii_lowercase();
    assert!(
        body.contains("x-tenant-id: tenant-a"),
        "projected tenant missing: {body}"
    );
    assert!(body.contains("x-user-id: alice"), "projected subject missing: {body}");
    assert!(
        !body.contains("spoofed-tenant"),
        "spoofed tenant reached upstream: {body}"
    );
    assert!(
        !body.contains("spoofed-user"),
        "spoofed subject reached upstream: {body}"
    );
    assert!(
        !body.contains("x-auth-tenant"),
        "ingress tenant assertion leaked: {body}"
    );
    assert!(
        !body.contains("x-auth-user"),
        "ingress subject assertion leaked: {body}"
    );
}

#[test]
fn project_state_owner_headers_fails_closed_when_identity_is_missing() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();
    let config = super::load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_guard.port())]),
    );
    let proxy = praxis_test_utils::start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 401);
    assert!(parse_body(&raw).contains("missing_state_owner"));
}
