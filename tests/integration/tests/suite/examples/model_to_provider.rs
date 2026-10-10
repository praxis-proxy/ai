// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for `model_to_provider` route selection across two backends.

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send, json_post,
    json_post_with_header, parse_body, parse_status, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn model_to_provider_routes_mapped_model_to_provider_backend() {
    let vertex = capturing_backend("vertex-response");
    let fallback = capturing_backend("fallback-response");
    let proxy = start_proxy(&config(vertex.port(), fallback.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", r#"{"model":"claude-sonnet-4-5","messages":[]}"#),
    );

    assert_eq!(parse_status(&raw), 200, "mapped model should be proxied");
    assert_eq!(
        parse_body(&raw),
        "vertex-response",
        "the provider selector should route the mapped model to the vertex cluster"
    );
    let forwarded = only_request(&vertex);
    let body: serde_json::Value = serde_json::from_str(&forwarded.body).expect("forwarded body should be JSON");
    assert_eq!(
        body["model"], "vertex/claude-sonnet-4-5",
        "the provider should receive the target model"
    );
    assert!(
        !forwarded.headers.to_ascii_lowercase().contains("x-praxis-ai-provider"),
        "the internal provider selector must not reach the upstream: {}",
        forwarded.headers
    );
    assert!(
        fallback.requests().is_empty(),
        "the fallback backend must not see a mapped model"
    );
}

#[test]
fn model_to_provider_sends_unmapped_model_to_fallback_unchanged() {
    let vertex = capturing_backend("vertex-response");
    let fallback = capturing_backend("fallback-response");
    let proxy = start_proxy(&config(vertex.port(), fallback.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", r#"{"model":"unknown-model","messages":[]}"#),
    );

    assert_eq!(parse_status(&raw), 200, "unmapped model should be proxied");
    assert_eq!(
        parse_body(&raw),
        "fallback-response",
        "an unmapped model should fall through to the default route"
    );
    let body: serde_json::Value =
        serde_json::from_str(&only_request(&fallback).body).expect("forwarded body should be JSON");
    assert_eq!(
        body["model"], "unknown-model",
        "an unmapped model must not be rewritten"
    );
    assert!(
        vertex.requests().is_empty(),
        "the vertex backend must not see an unmapped model"
    );
}

#[test]
fn model_to_provider_ignores_mapped_model_on_unlisted_path() {
    let vertex = capturing_backend("vertex-response");
    let fallback = capturing_backend("fallback-response");
    let proxy = start_proxy(&config(vertex.port(), fallback.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/chat/completions", r#"{"model":"claude-sonnet-4-5","messages":[]}"#),
    );

    assert_eq!(parse_status(&raw), 200, "request on an unlisted path should be proxied");
    assert_eq!(
        parse_body(&raw),
        "fallback-response",
        "a mapping applies only on its configured paths"
    );
    let body: serde_json::Value =
        serde_json::from_str(&only_request(&fallback).body).expect("forwarded body should be JSON");
    assert_eq!(
        body["model"], "claude-sonnet-4-5",
        "the model must not be rewritten on an unlisted path"
    );
}

#[test]
fn model_to_provider_rejects_client_supplied_provider_selector() {
    let vertex = capturing_backend("vertex-response");
    let fallback = capturing_backend("fallback-response");
    let proxy = start_proxy(&config(vertex.port(), fallback.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/messages",
            r#"{"model":"unknown-model","messages":[]}"#,
            "x-praxis-ai-provider: vertex",
        ),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "a client must not be able to pick the provider route directly"
    );
    assert!(
        vertex.requests().is_empty() && fallback.requests().is_empty(),
        "a rejected request must not reach any backend"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Start a backend that answers one request with `response` and records it.
fn capturing_backend(response: &str) -> StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown()
}

/// Return the single request a backend received.
fn only_request(backend: &StatefulCapturingGuard) -> CapturedRequest {
    let mut requests = backend.requests();
    assert_eq!(requests.len(), 1, "backend should receive exactly one request");
    requests.remove(0)
}

/// Build a two-cluster config: the router sends the `vertex` provider
/// selector to one backend and everything else to a fallback.
fn config(vertex_port: u16, fallback_port: u16) -> Config {
    let proxy_port = free_port();
    let yaml = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: model_to_provider
        models:
          - model: claude-sonnet-4-5
            provider: vertex
            target_model: vertex/claude-sonnet-4-5
            paths: [/v1/messages]
      - filter: router
        routes:
          - path_prefix: "/"
            headers:
              x-praxis-ai-provider: vertex
            cluster: vertex
          - path_prefix: "/"
            cluster: fallback
      - filter: load_balancer
        clusters:
          - name: vertex
            endpoints:
              - "127.0.0.1:{vertex_port}"
          - name: fallback
            endpoints:
              - "127.0.0.1:{fallback_port}"
insecure_options:
  allow_private_endpoints: true
"#
    );
    Config::from_yaml(&yaml).expect("model_to_provider test config should parse")
}
