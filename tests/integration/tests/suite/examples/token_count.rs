// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the token_count filter example configuration.
//!
//! The filter writes to `filter_metadata` which is not observable from
//! an HTTP response, so most cases only verify the proxy starts and
//! proxies traffic correctly. Token extraction correctness is covered
//! by unit tests in `praxis-ai-filters`.
//!
//! The Bedrock `InvokeModel` header path is the exception: it is
//! observable end-to-end by chaining `token_count` with
//! `token_usage_headers`, which surfaces the extracted counts as
//! response headers.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, free_port, http_send, parse_header, parse_status, start_backend_with_shutdown, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn token_count_proxies_response() {
    let backend_port_guard = start_backend_with_shutdown("ok");
    let backend_port = backend_port_guard.port();
    let proxy_port = free_port();
    let config = super::load_example_config(
        "token-counting.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
}

#[test]
fn token_count_bedrock_invoke_model_extracts_header_counts() {
    let backend_port_guard = Backend::fixed("ok")
        .header("x-amzn-bedrock-input-token-count", "25")
        .header("x-amzn-bedrock-output-token-count", "50")
        .start_with_shutdown();
    let backend_port = backend_port_guard.port();
    let proxy_port = free_port();
    let config = Config::from_yaml(&make_bedrock_invoke_model_yaml(proxy_port, backend_port)).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    assert_eq!(
        parse_header(&raw, "praxis-token-input"),
        Some("25".to_owned()),
        "input token count should be extracted from Bedrock InvokeModel headers"
    );
    assert_eq!(
        parse_header(&raw, "praxis-token-output"),
        Some("50".to_owned()),
        "output token count should be extracted from Bedrock InvokeModel headers"
    );
    assert_eq!(
        parse_header(&raw, "praxis-token-total"),
        Some("75".to_owned()),
        "total token count should be computed from input + output"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Builds a config chaining `token_count` (Bedrock `InvokeModel`) with
/// `token_usage_headers`, so the extracted counts become observable as
/// response headers.
///
/// Response hooks run in *reverse* declared order (last filter's
/// `on_response` fires first), so `token_usage_headers` is declared
/// before `token_count` here to ensure `token_count` populates
/// `filter_metadata` before `token_usage_headers` reads it.
fn make_bedrock_invoke_model_yaml(proxy_port: u16, backend_port: u16) -> String {
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
            cluster: backend
      - filter: token_usage_headers
      - filter: token_count
        provider: bedrock_invoke_model
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}
