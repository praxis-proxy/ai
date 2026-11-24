// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Tests for the credential injection example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, parse_body, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn credential_injection_config_parses() {
    let config = super::load_example_config(
        "credential-injection.yaml",
        29900,
        HashMap::from([("127.0.0.1:3000", 29901_u16), ("127.0.0.1:3001", 29902_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn credential_injection_injects_bearer_token() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "credential-injection.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:3001", backend_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "GET /v1/chat HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    assert!(
        body.contains("Bearer sk-example-openai-key"),
        "upstream should receive Bearer token in Authorization header: {body}"
    );
}

#[test]
fn credential_injection_strips_client_credential() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "credential-injection.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:3001", backend_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "GET /v1/chat HTTP/1.1\r\n\
         Host: localhost\r\n\
         Authorization: client-spoofed-token\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    assert!(
        !body.contains("client-spoofed-token"),
        "client credential should be stripped: {body}"
    );
    assert!(
        body.contains("Bearer sk-example-openai-key"),
        "server credential should be injected: {body}"
    );
}

#[test]
fn credential_injection_internal_cluster() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "credential-injection.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:3001", backend_port)]),
    );

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "GET /internal/api HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "should return 200");
    let body = parse_body(&raw);
    assert!(
        body.contains("internal-secret"),
        "upstream should receive x-api-key for internal cluster: {body}"
    );
}

fn tenant_config(
    tenant_a_port: u16,
    tenant_b_port: u16,
    backend_port: u16,
    path: &str,
    header: &str,
    prefix: &str,
) -> praxis_core::config::Config {
    let chains: String = ["alpha", "beta"]
        .into_iter()
        .map(|tenant| {
            format!(
                r#"
  - name: {tenant}
    filters:
      - filter: basic_auth
        strip_authorization: true
        credentials:
          - username: {tenant}
            password: fixture
      - filter: router
        routes:
          - path: {path}
            cluster: {tenant}
      - filter: load_balancer
        clusters:
          - name: {tenant}
            endpoints: ["127.0.0.1:{backend_port}"]
      - filter: credential_injection
        clusters:
          - name: {tenant}
            header: {header}
            value: upstream-{tenant}
            header_prefix: "{prefix}"
"#
            )
        })
        .collect();

    praxis_core::config::Config::from_yaml(&format!(
        r#"
listeners:
  - name: alpha
    address: "127.0.0.1:{tenant_a_port}"
    filter_chains: [alpha]
  - name: beta
    address: "127.0.0.1:{tenant_b_port}"
    filter_chains: [beta]
filter_chains:
{chains}
insecure_options:
  allow_private_endpoints: true
"#
    ))
    .expect("tenant fixture should parse")
}

#[test]
fn chat_completions_requires_tenant_credentials_before_injecting_upstream_key() {
    assert_tenant_credentials("/v1/chat/completions", "Authorization", "Bearer ");
}

#[test]
fn messages_requires_tenant_credentials_before_injecting_upstream_key() {
    assert_tenant_credentials("/v1/messages", "x-api-key", "");
}

fn assert_tenant_credentials(path: &str, header: &str, prefix: &str) {
    const ALPHA: &str = "Authorization: Basic YWxwaGE6Zml4dHVyZQ==\r\n";
    const BETA: &str = "Authorization: Basic YmV0YTpmaXh0dXJl\r\n";
    let backend = start_header_echo_backend();
    let tenant_a_port = free_port();
    let tenant_b_port = free_port();
    let config = tenant_config(tenant_a_port, tenant_b_port, backend.port(), path, header, prefix);
    let _proxy = praxis_test_utils::start_proxy(&config);
    let send = |port, authorization| {
        http_send(
            &format!("127.0.0.1:{port}"),
            &format!(
                "POST {path} HTTP/1.1\r\nHost: localhost\r\n\
                 {authorization}x-api-key: client-supplied-test-key\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\
                 Connection: close\r\n\r\n{{}}"
            ),
        )
    };

    for (port, allowed, other, key) in [
        (tenant_a_port, ALPHA, BETA, "upstream-alpha"),
        (tenant_b_port, BETA, ALPHA, "upstream-beta"),
    ] {
        for rejected in ["", "Authorization: Basic aW52YWxpZA==\r\n", other] {
            assert_eq!(parse_status(&send(port, rejected)), 401);
        }
        let response = send(port, allowed);
        assert_eq!(parse_status(&response), 200);
        let headers = parse_body(&response);
        let injected = headers
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter_map(|(name, value)| name.eq_ignore_ascii_case(header).then_some(value.trim()))
            .collect::<Vec<_>>();
        assert_eq!(injected, [format!("{prefix}{key}")]);
        assert!(!headers.contains("Basic "));
    }
}
