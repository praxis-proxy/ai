// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pipeline integration tests for provider-native error response formatters.
//!
//! Proves that the OpenAI and Anthropic error formatters are invoked by
//! Praxis when the upstream is unavailable (connection refused). Each test
//! sends a classified request through a real proxy pipeline pointing to
//! a non-listening port and asserts that the synthesized error response
//! uses the provider-native JSON shape instead of RFC 9457 Problem Details.

use praxis_core::config::Config;
use praxis_test_utils::{free_port, http_send, json_post, parse_body, parse_status, start_proxy};

// -----------------------------------------------------------------------------
// OpenAI Tests
// -----------------------------------------------------------------------------

/// An OpenAI Responses request to an unavailable upstream receives an
/// OpenAI-shaped error with `{"error": {...}}`.
#[test]
fn openai_responses_connection_refused_returns_openai_error() {
    let dead_port = free_port(); // port allocated but nothing listens on it
    let proxy_port = free_port();
    let config = Config::from_yaml(&openai_yaml(proxy_port, dead_port)).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    assert!(
        status >= 500,
        "connection refused should produce a 5xx status, got {status}"
    );

    let parsed: serde_json::Value = serde_json::from_str(&response_body)
        .unwrap_or_else(|e| panic!("response should be valid JSON: {e}\nbody: {response_body}"));

    assert!(
        parsed.get("error").is_some(),
        "OpenAI response must have top-level 'error' key, got: {parsed}"
    );
    assert_eq!(
        parsed["error"]["type"], "server_error",
        "5xx should use server_error type"
    );
    assert!(parsed["error"]["param"].is_null(), "param must be null");
    assert!(parsed["error"]["code"].as_str().is_some(), "code must be a string");
    assert!(
        parsed["error"]["message"].as_str().is_some(),
        "message must be a string"
    );

    // Must NOT be RFC 9457 Problem Details
    assert!(
        parsed.get("type").is_none() || parsed["type"] != "about:blank",
        "response must not be RFC 9457 Problem Details"
    );
}

/// An OpenAI Chat Completions request to an unavailable upstream also
/// receives an OpenAI-shaped error.
#[test]
fn openai_chat_completions_connection_refused_returns_openai_error() {
    let dead_port = free_port();
    let proxy_port = free_port();
    let config = Config::from_yaml(&openai_yaml(proxy_port, dead_port)).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    assert!(status >= 500, "should be 5xx, got {status}");

    let parsed: serde_json::Value = serde_json::from_str(&response_body)
        .unwrap_or_else(|e| panic!("response should be valid JSON: {e}\nbody: {response_body}"));

    assert!(
        parsed.get("error").is_some(),
        "Chat Completions response must have 'error' key, got: {parsed}"
    );
    assert_eq!(parsed["error"]["type"], "server_error");
}

// -----------------------------------------------------------------------------
// Anthropic Tests
// -----------------------------------------------------------------------------

/// An Anthropic Messages request to an unavailable upstream receives an
/// Anthropic-shaped error with `{"type":"error","error":{...}}`.
#[test]
fn anthropic_messages_connection_refused_returns_anthropic_error() {
    let dead_port = free_port();
    let proxy_port = free_port();
    let config = Config::from_yaml(&anthropic_yaml(proxy_port, dead_port)).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"claude-opus-4-8","max_tokens":1024,"system":"Be helpful.","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &anthropic_post("/v1/messages", body));
    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    assert!(status >= 500, "should be 5xx, got {status}");

    let parsed: serde_json::Value = serde_json::from_str(&response_body)
        .unwrap_or_else(|e| panic!("response should be valid JSON: {e}\nbody: {response_body}"));

    assert_eq!(
        parsed["type"], "error",
        "Anthropic response must have top-level type 'error', got: {parsed}"
    );
    assert!(parsed.get("error").is_some(), "must have nested 'error' object");

    let error_type = parsed["error"]["type"].as_str().unwrap();
    let allowed = [
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "conflict_error",
        "request_too_large",
        "rate_limit_error",
        "timeout_error",
        "api_error",
        "overloaded_error",
    ];
    assert!(
        allowed.contains(&error_type),
        "error type '{error_type}' must be in Anthropic's allowed vocabulary"
    );

    assert!(
        parsed["error"]["message"].as_str().is_some(),
        "error message must be a string"
    );

    // request_id must be present (string or null)
    assert!(
        parsed.get("request_id").is_some(),
        "Anthropic response must include request_id"
    );
}

// -----------------------------------------------------------------------------
// Fallback Test
// -----------------------------------------------------------------------------

/// An unclassified request to an unavailable upstream receives the
/// generic Praxis fallback (RFC 9457 Problem Details), NOT a provider
/// envelope.
#[test]
fn unclassified_connection_refused_returns_generic_fallback() {
    let dead_port = free_port();
    let proxy_port = free_port();
    let config = Config::from_yaml(&openai_yaml(proxy_port, dead_port)).unwrap();
    let proxy = start_proxy(&config);

    // Send a body that classifies as UnknownJson (no input, no messages)
    let body = r#"{"prompt":"hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/inference", body));
    let status = parse_status(&raw);
    let response_body = parse_body(&raw);

    assert!(status >= 500, "should be 5xx, got {status}");

    let parsed: serde_json::Value =
        serde_json::from_str(&response_body).expect("proxy error should be valid JSON (RFC 9457)");
    assert!(
        parsed.get("error").and_then(|e| e.get("type")).is_none(),
        "unclassified request should not receive OpenAI formatted error envelope"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// YAML config with the OpenAI responses format filter.
fn openai_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [classify]

filter_chains:
  - name: classify
    filters:
      - filter: openai_responses_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: dead_upstream
      - filter: load_balancer
        clusters:
          - name: dead_upstream
            endpoints:
              - "127.0.0.1:{backend_port}"

insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// YAML config with the Anthropic messages format filter.
fn anthropic_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: test
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [classify]

filter_chains:
  - name: classify
    filters:
      - filter: anthropic_messages_format
        on_invalid: continue
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: dead_upstream
      - filter: load_balancer
        clusters:
          - name: dead_upstream
            endpoints:
              - "127.0.0.1:{backend_port}"

insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// Build a POST request with Anthropic headers.
fn anthropic_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         anthropic-version: 2023-06-01\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    )
}
