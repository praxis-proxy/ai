// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for `model_to_provider` route selection across two backends.

use std::{collections::HashMap, sync::Arc};

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send, json_post,
    load_example_config, parse_body, parse_status, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn model_to_provider_routes_without_optional_provider_metadata() {
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
        "the mapping should select its existing vertex cluster without provider tags"
    );
    let forwarded = only_request(&vertex);
    let body: serde_json::Value = serde_json::from_str(&forwarded.body).expect("forwarded body should be JSON");
    assert_eq!(
        body["model"], "claude-sonnet-4-5@20250929",
        "the provider should receive the target model"
    );
    assert!(
        fallback.requests().is_empty(),
        "the fallback backend must not see a mapped model"
    );
}

#[test]
fn model_to_provider_routes_count_tokens_to_provider_backend() {
    let vertex = capturing_backend("vertex-response");
    let fallback = capturing_backend("fallback-response");
    let proxy = start_proxy(&config(vertex.port(), fallback.port()));
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/messages/count_tokens",
            r#"{"model":"claude-sonnet-4-5","messages":[]}"#,
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "the second configured mapping path must be proxied"
    );
    let forwarded = only_request(&vertex);
    let body: serde_json::Value =
        serde_json::from_str(&forwarded.body).expect("forwarded count-token body must be JSON");
    assert_eq!(
        body["model"], "claude-sonnet-4-5@20250929",
        "count-token requests must use the provider model"
    );
    assert!(
        fallback.requests().is_empty(),
        "the fallback must not receive the second mapped path"
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
fn model_to_provider_rejects_unknown_mapped_and_fallback_clusters() {
    let example = include_str!("../../../../../examples/configs/model-to-provider.yaml");
    for yaml in [
        example.replace("upstream: vertex", "upstream: missing-mapped"),
        example.replace("fallback_upstream: fallback", "fallback_upstream: missing-fallback"),
    ] {
        let config = Config::from_yaml(&yaml).unwrap();
        let health = Arc::new(HashMap::new());
        let kv_stores = praxis_core::kv::KvStoreRegistry::new();
        let client = praxis_test_utils::test_subrequest_client();
        let registry = praxis_ai::build_full_registry(&client);
        let inflight_registry = praxis_ai::InFlightRegistry::new();
        let error = praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &inflight_registry, &client)
            .err()
            .expect("normal pipeline validation must reject unknown mapping targets");
        assert!(
            error.to_string().contains("which is not defined"),
            "unexpected validation error: {error}"
        );
    }
}

#[test]
fn model_to_provider_leaves_endpoint_selection_to_load_balancer() {
    let first = capturing_backend("first");
    let second = capturing_backend("second");
    let fallback = capturing_backend("fallback");
    let yaml = example_yaml(first.port(), fallback.port()).replace(
        &format!("              - \"127.0.0.1:{}\"", first.port()),
        &format!(
            "              - \"127.0.0.1:{}\"\n              - \"127.0.0.1:{}\"",
            first.port(),
            second.port()
        ),
    );
    let proxy = start_proxy(&Config::from_yaml(&yaml).unwrap());
    for _request in 0..2 {
        let raw = http_send(
            proxy.addr(),
            &json_post("/v1/messages", r#"{"model":"claude-sonnet-4-5"}"#),
        );
        assert_eq!(
            parse_status(&raw),
            200,
            "each endpoint should accept the mapped request"
        );
    }
    assert_eq!(
        first.requests().len(),
        1,
        "round robin must select the first endpoint once"
    );
    assert_eq!(
        second.requests().len(),
        1,
        "round robin must select the second endpoint once"
    );
    assert!(
        fallback.requests().is_empty(),
        "mapped traffic must remain in its selected cluster"
    );
}

#[test]
fn model_to_provider_preserves_target_model_on_cluster_retry() {
    let vertex = StatefulCapturingBackend::new(vec![(503, "retry".to_owned()), (200, "recovered".to_owned())])
        .start_with_shutdown();
    let fallback = capturing_backend("fallback");
    let yaml = example_yaml(vertex.port(), fallback.port()).replace(
        "          - name: vertex",
        "          - name: vertex\n            retry_policy:\n              max_retries: 1\n              retriable_status_codes: [503]\n              allow_non_idempotent: true",
    );
    let proxy = start_proxy(&Config::from_yaml(&yaml).unwrap());
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", r#"{"model":"claude-sonnet-4-5","messages":[]}"#),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "the cluster retry policy should recover the request"
    );
    assert_eq!(
        parse_body(&raw),
        "recovered",
        "the successful retry should reach the client"
    );
    let requests = vertex.requests();
    assert_eq!(
        requests.len(),
        2,
        "the load balancer's cluster policy permits exactly one retry"
    );
    for request in requests {
        let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
        assert_eq!(
            body["model"], "claude-sonnet-4-5@20250929",
            "each attempt must retain the exact target"
        );
    }
    assert!(
        fallback.requests().is_empty(),
        "endpoint retries must not become model fallback routing"
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

/// Load the actual example, assigning each cluster its own capturing backend.
fn config(vertex_port: u16, fallback_port: u16) -> Config {
    load_example_config(
        "model-to-provider.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:9001", vertex_port), ("127.0.0.1:9002", fallback_port)]),
    )
}

/// Bind the actual example's listener and upstream endpoints to test ports.
fn example_yaml(vertex_port: u16, fallback_port: u16) -> String {
    include_str!("../../../../../examples/configs/model-to-provider.yaml")
        .replace("127.0.0.1:8080", &format!("127.0.0.1:{}", free_port()))
        .replace("127.0.0.1:9001", &format!("127.0.0.1:{vertex_port}"))
        .replace("127.0.0.1:9002", &format!("127.0.0.1:{fallback_port}"))
}
