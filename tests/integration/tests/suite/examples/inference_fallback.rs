// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Conformance tests for inference failover with protocol translation.
//!
//! Validates that the iterative request router composes correctly with
//! Responses-to-Chat Completions translation and credential injection
//! across a failover boundary.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, free_port, http_send, json_post, parse_body, parse_status, start_proxy,
};

use super::load_example_config;

const EXAMPLE: &str = "inference/fallback-with-translation.yaml";

fn chat_completions_response() -> String {
    serde_json::json!({
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from fallback."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    })
    .to_string()
}

fn responses_request() -> String {
    serde_json::json!({
        "model": "gpt-4.1-mini",
        "input": "Hello",
        "stream": false,
        "store": false
    })
    .to_string()
}

#[test]
fn fallback_on_primary_503() {
    let primary = StatefulCapturingBackend::new(vec![(503, r#"{"error":"service unavailable"}"#.to_owned())])
        .start_with_shutdown();
    let fallback = StatefulCapturingBackend::new(vec![(200, chat_completions_response())]).start_with_shutdown();

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", primary.port()), ("127.0.0.1:3002", fallback.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &responses_request()));

    let primary_requests = primary.requests();
    assert_eq!(primary_requests.len(), 1, "primary should receive exactly one request");
    let primary_req = &primary_requests[0];
    assert_eq!(
        primary_req.uri, "/v1/chat/completions",
        "primary path should be rewritten"
    );
    let primary_body: serde_json::Value =
        serde_json::from_str(&primary_req.body).expect("primary request body should be JSON");
    assert!(
        primary_body["messages"].is_array(),
        "primary should receive translated messages array"
    );
    assert!(
        primary_req.headers.contains("Bearer primary-key"),
        "primary should receive primary credentials"
    );

    let fallback_requests = fallback.requests();
    assert_eq!(
        fallback_requests.len(),
        1,
        "fallback should receive exactly one request"
    );
    let fallback_req = &fallback_requests[0];
    assert_eq!(
        fallback_req.uri, "/v1/chat/completions",
        "fallback path should be rewritten"
    );
    let fallback_body: serde_json::Value =
        serde_json::from_str(&fallback_req.body).expect("fallback request body should be JSON");
    assert!(
        fallback_body["messages"].is_array(),
        "fallback should receive translated messages array"
    );
    assert!(
        fallback_req.headers.contains("Bearer fallback-key"),
        "fallback should receive fallback credentials"
    );

    let status = parse_status(&raw);
    assert_eq!(status, 200, "client should receive 200 after fallback succeeds");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(
        response["object"], "response",
        "response should be a Responses API resource"
    );
    assert_eq!(
        response["output"][0]["content"][0]["text"], "Hello from fallback.",
        "response text should match fallback backend"
    );
    assert_eq!(
        response["usage"]["input_tokens"], 5,
        "input token count should match fallback backend"
    );
    assert_eq!(
        response["usage"]["output_tokens"], 4,
        "output token count should match fallback backend"
    );
}

#[test]
fn primary_succeeds_no_fallback() {
    let primary_response = serde_json::json!({
        "id": "chatcmpl_primary",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from primary."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    })
    .to_string();

    let primary = StatefulCapturingBackend::new(vec![(200, primary_response)]).start_with_shutdown();
    let fallback = StatefulCapturingBackend::new(vec![(200, chat_completions_response())]).start_with_shutdown();

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", primary.port()), ("127.0.0.1:3002", fallback.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &responses_request()));

    let primary_requests = primary.requests();
    assert_eq!(primary_requests.len(), 1, "primary should receive exactly one request");

    let fallback_requests = fallback.requests();
    assert_eq!(fallback_requests.len(), 0, "fallback should receive zero requests");

    let status = parse_status(&raw);
    assert_eq!(status, 200, "client should receive 200 from primary");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(
        response["object"], "response",
        "response should be a Responses API resource"
    );
    assert_eq!(
        response["output"][0]["content"][0]["text"], "Hello from primary.",
        "response text should match primary backend"
    );
    assert_eq!(
        response["usage"]["input_tokens"], 3,
        "input token count should match primary backend"
    );
    assert_eq!(
        response["usage"]["output_tokens"], 2,
        "output token count should match primary backend"
    );
}

#[test]
fn both_backends_fail_returns_last_error() {
    let primary = StatefulCapturingBackend::new(vec![(503, r#"{"error":"primary unavailable"}"#.to_owned())])
        .start_with_shutdown();
    let fallback = StatefulCapturingBackend::new(vec![(503, r#"{"error":"fallback unavailable"}"#.to_owned())])
        .start_with_shutdown();

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", primary.port()), ("127.0.0.1:3002", fallback.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &responses_request()));

    let primary_requests = primary.requests();
    assert_eq!(primary_requests.len(), 1, "primary should receive exactly one request");

    let fallback_requests = fallback.requests();
    assert_eq!(
        fallback_requests.len(),
        1,
        "fallback should receive exactly one request"
    );

    let status = parse_status(&raw);
    assert_eq!(status, 503, "client should receive 503 when both backends fail");
}
