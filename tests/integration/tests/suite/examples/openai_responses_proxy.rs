// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses proxy example config.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_capturing_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn openai_responses_proxy_example_forwards_to_backend() {
    let backend_guard = start_backend_with_shutdown("inference-ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/responses-proxy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello, world!","stream":false}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxied request should return 200");
    assert_eq!(
        parse_body(&raw),
        "inference-ok",
        "response body should be relayed from inference backend"
    );
}

#[test]
fn openai_responses_proxy_example_preserves_json_response() {
    let json_response = r#"{"id":"resp_abc","object":"response","output":[{"type":"message","content":[{"type":"output_text","text":"Hi!"}]}]}"#;
    let backend_guard = start_backend_with_shutdown(json_response);
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/responses-proxy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello","stream":false}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "JSON response should return 200");
    assert_eq!(
        parse_body(&raw),
        json_response,
        "JSON response body should be preserved exactly"
    );
}

#[test]
fn openai_responses_proxy_example_preserves_native_conversation() {
    let backend_guard = start_capturing_backend("inference-ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/responses-proxy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello","conversation":{"id":"conv_native"}}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxied request should return 200");
    assert_eq!(
        backend_guard.body(),
        body,
        "conversation should reach the native backend unchanged"
    );
}

#[test]
fn openai_responses_proxy_example_rejects_prompt_templates_for_generic_backend() {
    let backend_guard = start_backend_with_shutdown("must-not-be-contacted");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/responses-proxy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","prompt":{"id":"pmpt_123"}}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error response should be JSON");

    assert_eq!(parse_status(&raw), 400, "generic backend prompt must return HTTP 400");
    assert_eq!(
        response["error"]["type"], "invalid_request_error",
        "generic backend prompt must use the invalid-request error type"
    );
    assert_eq!(
        response["error"]["message"],
        "prompt templates are supported only when the selected upstream declares application_protocol: openai_responses and application_provider: openai",
        "generic backend prompt must explain the protocol and provider declaration requirement"
    );
}

#[test]
fn openai_responses_declaration_allows_prompt_templates_for_local_endpoint() {
    let backend_guard = start_capturing_backend("inference-ok");
    let proxy_port = free_port();
    let config = format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/v1/responses"
            cluster: "openai"
      - filter: load_balancer
        clusters:
          - name: "openai"
            http:
              application_protocol: "openai_responses"
              application_provider: "openai"
            endpoints:
              - "127.0.0.1:{}"
      - filter: openai_responses_proxy
insecure_options:
  allow_private_endpoints: true
"#,
        backend_guard.port()
    );
    let config = praxis_core::config::Config::from_yaml(&config).expect("provider test config should parse");
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","prompt":{"id":"pmpt_123"}}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "declared OpenAI Responses backend should receive prompt"
    );
    assert_eq!(
        backend_guard.body(),
        body,
        "application declarations must allow byte-exact prompt forwarding regardless of endpoint hostname"
    );
}

#[test]
fn openai_responses_proxy_example_forwards_subresource_paths() {
    let backend_guard = start_backend_with_shutdown("subresource-ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/responses-proxy.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "GET /v1/responses/resp_abc/input_items HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "subresource request should return 200");
    assert_eq!(
        parse_body(&raw),
        "subresource-ok",
        "subresource path should be proxied to inference backend"
    );
}
