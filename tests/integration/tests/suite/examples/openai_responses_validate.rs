// SPDX-License-Identifier: Apache-2.0
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
// Transparent Backend Error Forwarding
// -----------------------------------------------------------------------------

#[test]
fn non_streaming_backend_error_forwarded_transparently() {
    let backend_error =
        r#"{"error":{"message":"The model does not exist.","type":"NotFoundError","code":404,"param":"model"}}"#;
    let backend_guard = Backend::status(404, backend_error)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = Config::from_yaml(&validate_yaml(proxy_port, backend_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"test"}"#),
    );

    assert_eq!(parse_status(&raw), 404, "HTTP status should be forwarded unchanged");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "content-type should be forwarded unchanged"
    );

    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["type"], "NotFoundError");
    assert_eq!(parsed["error"]["code"], 404, "numeric code should be preserved");
    assert_eq!(parsed["error"]["param"], "model", "param field should be preserved");
    assert_eq!(parsed["error"]["message"], "The model does not exist.");
}

#[test]
fn streaming_backend_error_forwarded_transparently() {
    let backend_error =
        r#"{"error":{"message":"The model does not exist.","type":"NotFoundError","code":404,"param":"model"}}"#;
    let backend_guard = Backend::status(404, backend_error)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = Config::from_yaml(&validate_yaml(proxy_port, backend_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"test","stream":true}"#),
    );

    assert_eq!(parse_status(&raw), 404, "HTTP status should be forwarded unchanged");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "content-type should be forwarded unchanged from backend"
    );

    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["error"]["type"], "NotFoundError");
    assert_eq!(parsed["error"]["code"], 404, "numeric code should be preserved");
    assert_eq!(parsed["error"]["param"], "model", "param field should be preserved");
    assert_eq!(parsed["error"]["message"], "The model does not exist.");
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
insecure_options:
  allow_private_endpoints: true
"#
    )
}
