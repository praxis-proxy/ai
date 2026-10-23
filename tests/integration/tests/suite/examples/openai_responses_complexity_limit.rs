// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the structural-complexity-limit example config (#1618).
//!
//! Proves `openai_responses_request.max_input_nodes` rejects a managed create
//! whose parsed body exceeds the structural-node cap — even when its serialized
//! size is small — before any state is built, while a request within the cap is
//! forwarded to the backend unchanged.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_capturing_backend, start_proxy,
};

const CONFIG: &str = "openai/responses/structural-complexity-limit.yaml";

#[test]
fn request_within_node_limit_is_forwarded() {
    let inference = start_capturing_backend("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        CONFIG,
        proxy_port,
        HashMap::from([("127.0.0.1:8000", inference.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "a structurally small request should be forwarded"
    );
    assert_eq!(
        parse_body(&raw),
        "inference",
        "a request within the node limit should reach the inference backend"
    );
    // Assert on the request the backend actually received: the input must survive
    // forwarding even though #1618 moves it out of the parsed parameters.
    let forwarded: serde_json::Value =
        serde_json::from_str(&inference.body()).expect("backend received a JSON request body");
    assert_eq!(
        forwarded["input"], "Hello, world!",
        "the forwarded request must preserve the client input"
    );
}

#[test]
fn structurally_complex_request_is_rejected() {
    let inference = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        CONFIG,
        proxy_port,
        HashMap::from([("127.0.0.1:8000", inference.port())]),
    );
    let proxy = start_proxy(&config);

    // 15 tiny input items blow past the 20-node cap while the body stays small,
    // demonstrating that structural complexity — not byte size — trips the guard.
    let items: Vec<String> = (0..15)
        .map(|i| format!(r#"{{"role":"user","content":"m{i}"}}"#))
        .collect();
    let body = format!(r#"{{"model":"gpt-4.1","input":[{}]}}"#, items.join(","));
    assert!(
        body.len() < 1024,
        "fixture stays byte-small so the structural cap, not a byte cap, is what rejects"
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));

    assert_eq!(
        parse_status(&raw),
        413,
        "a structurally complex managed create should be rejected before state is built"
    );
}
