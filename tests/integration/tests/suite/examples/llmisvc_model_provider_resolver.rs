// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the LLMISvc model-provider resolver example configuration.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, start_echo_backend, start_header_echo_backend,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn llmisvc_model_provider_resolver_config_parses() {
    let config = super::load_example_config(
        "llmisvc-model-provider-resolver.yaml",
        29920,
        HashMap::from([("127.0.0.1:3000", 29921_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(
        &*config.listeners[0].name, "llmisvc-gateway",
        "listener name should be llmisvc-gateway"
    );
}

#[test]
fn llmisvc_rewrites_publisher_id_body_model() {
    let backend_guard = start_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmisvc-model-provider-resolver.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"publishers/rhoai/models/granite-3.1-8b","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "rewrite should return 200");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("backend should echo valid JSON");
    assert_eq!(
        parsed["model"].as_str(),
        Some("granite-3.1-8b"),
        "upstream body model should be the short name"
    );
    assert_eq!(
        parsed["messages"][0]["content"].as_str(),
        Some("hi"),
        "other body fields should be preserved"
    );
}

#[test]
fn llmisvc_preserves_routing_header_publisher_id() {
    let backend_guard = start_header_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmisvc-model-provider-resolver.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"publishers/rhoai/models/granite-3.1-8b","messages":[]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "header echo should return 200");
    let headers = parse_body(&raw);
    assert!(
        headers
            .lines()
            .any(|line| line.eq_ignore_ascii_case("x-model: publishers/rhoai/models/granite-3.1-8b")),
        "X-Model routing header must remain the publisher ID, got:\n{headers}"
    );
}

#[test]
fn llmisvc_passes_non_publisher_model_through() {
    let backend_guard = start_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "llmisvc-model-provider-resolver.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"mistral-large-latest","messages":[]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "passthrough should return 200");
    let parsed: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("backend should echo valid JSON");
    assert_eq!(
        parsed["model"].as_str(),
        Some("mistral-large-latest"),
        "non-publisher model must not be rewritten"
    );
}
