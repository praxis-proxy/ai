// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `time_to_first_token` filter.
//!
//! Verifies that the proxy passes SSE and non-streaming responses through
//! unchanged when the `time_to_first_token` filter is in the pipeline. The TTFT metric
//! itself is written to the Prometheus registry (not observable via HTTP
//! response), so these tests focus on transparency.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_proxy,
};

const SSE_BODY: &str = concat!(
    "data: {\"id\":\"resp_1\",\"object\":\"response\",\"output\":[]}\n\n",
    "data: {\"id\":\"resp_1\",\"object\":\"response\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}]}\n\n",
    "data: [DONE]\n\n",
);

const JSON_BODY: &str = r#"{"id":"resp_1","object":"response","output":[]}"#;

// -----------------------------------------------------------------------------
// Example config smoke tests
// -----------------------------------------------------------------------------

#[test]
fn example_config_time_to_first_token_sse_passthrough() {
    let backend = Backend::fixed(SSE_BODY)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "time-to-first-token.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4o","input":"hi","stream":true}"#),
    );
    assert_eq!(parse_status(&raw), 200, "example config smoke test should return 200");
    assert_eq!(parse_body(&raw), SSE_BODY, "SSE body should pass through unchanged");
}

#[test]
fn example_config_time_to_first_token_json_passthrough() {
    let backend = Backend::fixed(JSON_BODY)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "time-to-first-token.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4o","input":"hi"}"#),
    );
    assert_eq!(parse_status(&raw), 200, "non-streaming request should return 200");
    assert_eq!(parse_body(&raw), JSON_BODY, "JSON body should pass through unchanged");
}
