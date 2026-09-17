// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Management-path skip-list tests for `intelligent_route`.
//!
//! Covers item 3 of the contract tracked in praxis-proxy/ai#1039: management
//! and discovery endpoints (`/v1/models`, `/v1/subscriptions`, `/v1/api-keys`,
//! health) bypass model resolution entirely and are routed to the configured
//! `management_cluster`, even when they carry a body `model` field. Inference
//! paths are still resolved by model and fail closed on an unknown model.
//!
//! These drive the real proxy end-to-end against capturing mock backends so
//! the behavior is observed at the wire, not asserted from filter internals.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, load_example_config, parse_status, start_backend_with_shutdown, start_proxy,
    start_stateful_backend,
};

const EXAMPLE: &str = "intelligent-route-management-skip.yaml";

/// Endpoints as written in `intelligent-route-management-skip.yaml`.
const INFERENCE_ENDPOINT: &str = "127.0.0.1:8001";
const MANAGEMENT_ENDPOINT: &str = "127.0.0.1:8003";

/// Build a raw HTTP/1.1 request with an optional JSON body.
fn raw_request(method: &str, path: &str, body: &str) -> String {
    if body.is_empty() {
        format!(
            "{method} {path} HTTP/1.1\r\n\
             Host: localhost\r\n\
             Connection: close\r\n\r\n"
        )
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {body}",
            body.len()
        )
    }
}

/// A management POST that happens to carry a `model` field must be served by
/// the management backend, not failed closed or routed to inference.
#[test]
fn management_path_with_body_model_is_served_not_failed_closed() {
    let management = start_backend_with_shutdown("management-backend");
    // Capturing backend so we can assert inference is never contacted.
    let inference = start_stateful_backend(vec![(200, "inference-backend".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            (INFERENCE_ENDPOINT, inference.port()),
            (MANAGEMENT_ENDPOINT, management.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4o-not-configured","name":"my-key"}"#;
    let raw = http_send(proxy.addr(), &raw_request("POST", "/v1/api-keys", body));

    assert_eq!(parse_status(&raw), 200, "management path must be served: {raw}");
    assert!(
        raw.contains("management-backend"),
        "management path must reach the management backend: {raw}"
    );
    assert!(
        inference.requests().is_empty(),
        "a management path must never reach inference: {:?}",
        inference.requests()
    );
}

/// A discovery GET (no body) is skipped and path-routed to management.
#[test]
fn discovery_path_reaches_management_backend() {
    let management = start_backend_with_shutdown("management-backend");
    let inference = start_stateful_backend(vec![(200, "inference-backend".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            (INFERENCE_ENDPOINT, inference.port()),
            (MANAGEMENT_ENDPOINT, management.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &raw_request("GET", "/v1/models", ""));

    assert_eq!(parse_status(&raw), 200, "discovery path must be served: {raw}");
    assert!(
        raw.contains("management-backend"),
        "discovery path must reach the management backend: {raw}"
    );
    assert!(
        inference.requests().is_empty(),
        "a discovery path must never reach inference: {:?}",
        inference.requests()
    );
}

/// The skip is path-scoped: an unknown model on the inference path still fails
/// closed without contacting any upstream.
#[test]
fn unknown_model_on_inference_path_still_fails_closed() {
    let inference = start_stateful_backend(vec![(200, "inference-backend".to_owned())]);
    let management = start_stateful_backend(vec![(200, "management-backend".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            (INFERENCE_ENDPOINT, inference.port()),
            (MANAGEMENT_ENDPOINT, management.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4o-not-configured","messages":[]}"#;
    let raw = http_send(proxy.addr(), &raw_request("POST", "/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 404, "unknown model must fail closed: {raw}");
    assert!(
        inference.requests().is_empty(),
        "inference must not be contacted for an unknown model: {:?}",
        inference.requests()
    );
    assert!(
        management.requests().is_empty(),
        "management must not be contacted for an inference request: {:?}",
        management.requests()
    );
}

/// Control: a known model on the inference path routes to the inference
/// backend and never touches management.
#[test]
fn known_model_on_inference_path_routes_to_inference_backend() {
    let inference = start_backend_with_shutdown("inference-backend");
    let management = start_stateful_backend(vec![(200, "management-backend".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            (INFERENCE_ENDPOINT, inference.port()),
            (MANAGEMENT_ENDPOINT, management.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"granite-3.3-8b","messages":[]}"#;
    let raw = http_send(proxy.addr(), &raw_request("POST", "/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "known model must be routed: {raw}");
    assert!(
        raw.contains("inference-backend"),
        "known model must reach the inference backend: {raw}"
    );
    assert!(
        management.requests().is_empty(),
        "management must not be contacted for an inference request: {:?}",
        management.requests()
    );
}
