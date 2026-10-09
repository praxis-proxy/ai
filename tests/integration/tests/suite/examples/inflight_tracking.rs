// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `inflight-tracking` example configuration.
//!
//! The per-model counts live in an in-memory `InFlightRegistry` with no HTTP
//! surface, so they cannot be asserted through the proxy (that is the job of the
//! unit tests in `praxis_ai_filters::inflight`). These tests instead pin the
//! filter's observable contract: it reads the request body in `Stream` mode and
//! must forward it upstream byte-for-byte, whether or not the body carries a
//! trackable `model`.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, free_port, http_send, json_post, load_example_config, parse_status, start_proxy,
};

const BACKEND_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[]}"#;

/// Proxy `req_body` through the example config to a request-capturing backend
/// and return the body the backend actually received.
fn forward(req_body: &str) -> String {
    let backend = StatefulCapturingBackend::new(vec![(200, BACKEND_RESPONSE.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(
        "inflight-tracking.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", req_body));
    assert_eq!(parse_status(&raw), 200, "request should proxy to the backend");

    backend
        .requests()
        .into_iter()
        .find(|r| r.uri.starts_with("/v1/chat/completions"))
        .expect("backend should receive the forwarded request")
        .body
}

#[test]
fn example_config_inflight_tracking_forwards_tracked_request_unchanged() {
    let req_body = r#"{"model":"gpt-4","max_tokens":256,"messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(
        forward(req_body),
        req_body,
        "inflight_tracker reads the body in Stream mode and must forward it unchanged"
    );
}

#[test]
fn example_config_inflight_tracking_forwards_untracked_request_unchanged() {
    // No top-level model: the request is attributed to default_model, but the
    // filter must still forward the body untouched.
    let req_body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(
        forward(req_body),
        req_body,
        "an untracked request must also forward unchanged"
    );
}
