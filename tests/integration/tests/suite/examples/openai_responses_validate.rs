// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the request-validate example config.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, free_port, http_send, json_post, load_example_config, parse_body, parse_header, parse_status,
    start_backend_with_shutdown, start_echo_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn openai_responses_validate_example_forwards_valid_responses_request() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello, world!"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "valid responses request should be forwarded");
    assert_eq!(parse_body(&raw), "ok", "request should reach the backend");
}

#[test]
fn openai_responses_validate_example_forwards_streaming_background_unchanged() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{ "model":"gpt-4.1", "input":"test", "stream":true, "background":true, "provider_extension":{"keep":"verbatim"} }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming background request should reach backend"
    );
    assert_eq!(parse_body(&raw), body, "request body should be byte-for-byte unchanged");
}

#[test]
fn openai_responses_validate_example_forwards_store_false_background_unchanged() {
    let backend_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{ "model":"gpt-4.1", "input":"test", "background":true, "store":false, "unknown_field":[1,2,3] }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "store=false background request should reach backend"
    );
    assert_eq!(parse_body(&raw), body, "request body should be byte-for-byte unchanged");
}

#[test]
fn openai_responses_validate_example_accepts_minimal_request() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", r#"{"input":"Hello"}"#));

    assert_eq!(
        parse_status(&raw),
        200,
        "minimal request (input only) should be accepted"
    );
}

// -----------------------------------------------------------------------------
// Backend Error Formatting
// -----------------------------------------------------------------------------

#[test]
fn streaming_backend_error_returns_sse_events() {
    let backend_error = r#"{"error":{"message":"The model does not exist.","type":"NotFoundError","code":404}}"#;
    let backend_guard = Backend::status(404, backend_error)
        .header("content-type", "application/json")
        .header("content-encoding", "gzip")
        .header("content-range", "bytes 0-99/100")
        .header("etag", r#""upstream""#)
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = Config::from_yaml(&validate_yaml(proxy_port, backend_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"test","stream":true}"#),
    );

    assert_eq!(parse_status(&raw), 200, "streaming error should return 200");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "streaming error should have SSE content type"
    );
    assert_eq!(
        parse_header(&raw, "content-encoding"),
        None,
        "rewritten streaming errors should not keep upstream content-encoding"
    );
    assert_eq!(
        parse_header(&raw, "content-range"),
        None,
        "rewritten streaming errors should not keep upstream content-range"
    );
    assert_eq!(
        parse_header(&raw, "etag"),
        None,
        "rewritten streaming errors should not keep upstream etag"
    );

    let body = parse_body(&raw);
    let events: Vec<&str> = body.split("\n\n").filter(|s| !s.is_empty()).collect();
    assert_eq!(events.len(), 3, "should have 3 SSE events: {body}");

    let (created_name, created) = parse_sse_event(events[0]);
    assert_eq!(created_name, "response.created");
    assert_eq!(created["type"], "response.created");
    assert_eq!(created["response"]["status"], "in_progress");
    assert!(created["response"]["completed_at"].is_null());
    assert!(created["response"]["error"].is_null());

    let (in_progress_name, in_progress) = parse_sse_event(events[1]);
    assert_eq!(in_progress_name, "response.in_progress");
    assert_eq!(in_progress["type"], "response.in_progress");

    let (error_name, error) = parse_sse_event(events[2]);
    assert_eq!(error_name, "error");
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "NotFoundError");
    assert_eq!(error["error"]["code"], "404");
    assert_eq!(error["error"]["message"], "The model does not exist.");
}

#[test]
fn non_streaming_backend_error_returns_json() {
    let backend_error = r#"{"error":{"message":"The model does not exist.","type":"NotFoundError","code":404}}"#;
    let backend_guard = Backend::status(404, backend_error)
        .header("content-type", "application/json")
        .header("content-encoding", "gzip")
        .header("content-range", "bytes 0-99/100")
        .header("etag", r#""upstream""#)
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = Config::from_yaml(&validate_yaml(proxy_port, backend_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"test"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        404,
        "non-streaming error should keep original status"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "non-streaming error should have JSON content type"
    );
    assert_eq!(
        parse_header(&raw, "content-encoding"),
        None,
        "rewritten JSON errors should not keep upstream content-encoding"
    );
    assert_eq!(
        parse_header(&raw, "content-range"),
        None,
        "rewritten JSON errors should not keep upstream content-range"
    );
    assert_eq!(
        parse_header(&raw, "etag"),
        None,
        "rewritten JSON errors should not keep upstream etag"
    );

    let parsed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(parsed["error"]["type"], "NotFoundError");
    assert_eq!(parsed["error"]["code"], "404");
    assert_eq!(parsed["error"]["message"], "The model does not exist.");
    assert!(parsed["error"]["param"].is_null());
}

#[test]
fn successful_response_passes_through_unchanged() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = Config::from_yaml(&validate_yaml(proxy_port, backend_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"test"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "success should pass through");
    assert_eq!(parse_body(&raw), "ok", "body should be unchanged");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn validate_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_format
      - filter: openai_responses_validate
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"
      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}

fn parse_sse_event(frame: &str) -> (&str, serde_json::Value) {
    let mut lines = frame.lines();
    let event_type = lines
        .next()
        .and_then(|line| line.strip_prefix("event: "))
        .expect("SSE frame should start with event line");
    let data = lines
        .next()
        .and_then(|line| line.strip_prefix("data: "))
        .expect("SSE frame should contain data line");
    (event_type, serde_json::from_str(data).expect("SSE data should be JSON"))
}
