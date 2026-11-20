// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use praxis_core::config::Config;
use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, start_backend_with_shutdown, start_echo_backend,
    start_header_echo_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Model Alias Rewrite
// -----------------------------------------------------------------------------

#[test]
fn responses_model_alias_reaches_upstream() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "model should be rewritten to alias target"
    );
}

#[test]
fn responses_wildcard_model_alias_reaches_upstream() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-2026-06-24","input":"Hello, wildcard!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "wildcard model alias should be rewritten to alias target"
    );
}

// -----------------------------------------------------------------------------
// Default Model Injection
// -----------------------------------------------------------------------------

#[test]
fn responses_default_model_reaches_upstream() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"input":"Hello, no model specified"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "default model should be injected"
    );
}

// -----------------------------------------------------------------------------
// Unknown Model Passthrough
// -----------------------------------------------------------------------------

#[test]
fn responses_unknown_model_reaches_upstream_unchanged() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"some-unknown-model","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("some-unknown-model"),
        "unknown model should pass through unchanged"
    );
}

// -----------------------------------------------------------------------------
// Tool Preservation
// -----------------------------------------------------------------------------

#[test]
fn responses_tool_request_preserves_tools() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","input":"test","tools":[{"type":"function","name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "model should be rewritten"
    );
    assert!(echoed["tools"].is_array(), "tools array should be preserved");
    assert_eq!(
        echoed["tools"][0]["name"].as_str(),
        Some("read_file"),
        "tool name should be preserved"
    );
    assert!(
        echoed["tools"][0]["parameters"].is_object(),
        "tool parameters should be preserved"
    );
}

// -----------------------------------------------------------------------------
// Function Call Output Preservation
// -----------------------------------------------------------------------------

#[test]
fn responses_function_call_output_preserved() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","input":[{"type":"function_call_output","call_id":"call_abc123","output":"{\"result\":42}"},{"role":"user","content":"What happened?"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(echoed["model"].as_str(), Some("llama-3.3-70b"), "model rewritten");
    let items = echoed["input"].as_array().unwrap();
    assert_eq!(items.len(), 2, "input items should be preserved");
    assert_eq!(
        items[0]["type"].as_str(),
        Some("function_call_output"),
        "function_call_output type preserved"
    );
    assert_eq!(items[0]["call_id"].as_str(), Some("call_abc123"), "call_id preserved");
    assert_eq!(
        items[0]["output"].as_str(),
        Some("{\"result\":42}"),
        "output field preserved"
    );
}

// -----------------------------------------------------------------------------
// Chat Completions Traffic
// -----------------------------------------------------------------------------

#[test]
fn chat_completions_model_alias_reaches_upstream() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "chat completions model should be rewritten to the alias target"
    );
    assert_eq!(
        echoed["messages"][0]["content"].as_str(),
        Some("Hi"),
        "chat completions messages should be preserved"
    );
}

#[test]
fn chat_completions_default_model_reaches_upstream() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "default model should be injected for chat completions"
    );
}

#[test]
fn chat_completions_unknown_model_reaches_upstream_unchanged() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions should return 200");
    assert_eq!(
        parse_body(&raw),
        body,
        "unaliased chat completions body should pass through unchanged"
    );
}

// -----------------------------------------------------------------------------
// Non-Create Traffic
// -----------------------------------------------------------------------------

#[test]
fn non_create_body_passes_unchanged() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","input":"Hi"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/embeddings", body));

    assert_eq!(parse_status(&raw), 200, "embeddings should return 200");
    assert_eq!(
        parse_body(&raw),
        body,
        "non-create traffic should pass through unchanged"
    );
}

// -----------------------------------------------------------------------------
// Content-Length
// -----------------------------------------------------------------------------

#[test]
fn content_length_matches_mutated_body() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-latest","input":"test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed_body = parse_body(&raw);
    let echoed_len = echoed_body.len();

    drop(proxy);
    drop(echo_guard);

    let header_guard = start_header_echo_backend();
    let proxy_port2 = free_port();
    let config2 = Config::from_yaml(&header_echo_yaml(proxy_port2, header_guard.port())).unwrap();
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(proxy2.addr(), &json_post("/v1/responses", body));
    assert_eq!(parse_status(&raw2), 200, "header echo should return 200");
    let headers_text = parse_body(&raw2);

    let cl_lines: Vec<&str> = headers_text
        .lines()
        .filter(|l| l.to_lowercase().starts_with("content-length:"))
        .collect();
    assert_eq!(
        cl_lines.len(),
        1,
        "backend should receive exactly one Content-Length header, got {cl_lines:?}"
    );

    let cl_value: usize = cl_lines[0]
        .split(':')
        .nth(1)
        .unwrap()
        .trim()
        .parse()
        .expect("content-length should be a number");
    assert_eq!(
        cl_value, echoed_len,
        "upstream content-length ({cl_value}) should match rewritten body length ({echoed_len})"
    );
}

// -----------------------------------------------------------------------------
// Effective Model Header Routing
// -----------------------------------------------------------------------------

#[test]
fn effective_model_header_routes_to_expected_backend() {
    let llama_guard = start_backend_with_shutdown("llama-backend");
    let qwen_guard = start_backend_with_shutdown("qwen-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = Config::from_yaml(&effective_model_routing_yaml(
        proxy_port,
        llama_guard.port(),
        qwen_guard.port(),
        default_guard.port(),
    ))
    .unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"codex-mini-2026-06-24","input":"route to llama"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(parse_status(&raw), 200, "llama route should return 200");
    assert_eq!(
        parse_body(&raw),
        "llama-backend",
        "wildcard codex model should route to llama-backend via effective model header"
    );

    let body = r#"{"model":"gpt-4.1-mini","input":"route to qwen"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(parse_status(&raw), 200, "qwen route should return 200");
    assert_eq!(
        parse_body(&raw),
        "qwen-backend",
        "gpt-4.1-mini should route to qwen-backend via effective model header"
    );

    let body = r#"{"model":"unknown-model","input":"route to default"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(parse_status(&raw), 200, "default route should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "unknown model should route to default-backend"
    );
}

// -----------------------------------------------------------------------------
// Null Model Default Injection
// -----------------------------------------------------------------------------

#[test]
fn responses_null_model_receives_default() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":null,"input":"null model test"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "proxy should return 200");
    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        echoed["model"].as_str(),
        Some("llama-3.3-70b"),
        "null model should receive configured default"
    );
}

// -----------------------------------------------------------------------------
// Malformed JSON Rejection
// -----------------------------------------------------------------------------

#[test]
fn malformed_json_rejected_in_reject_mode() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&reject_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", "not valid json {{{"));

    assert_eq!(parse_status(&raw), 400, "malformed JSON should return 400");
    let error_body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        error_body["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "rejection should have structured error type"
    );
    assert_eq!(
        error_body["error"]["message"].as_str(),
        Some("invalid JSON body"),
        "rejection should have descriptive message"
    );
}

#[test]
fn malformed_json_continues_in_continue_mode() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", "not valid json {{{"));

    assert_eq!(
        parse_status(&raw),
        200,
        "malformed JSON in continue mode should pass through"
    );
}

// -----------------------------------------------------------------------------
// Repeated Model Rejection
// -----------------------------------------------------------------------------

#[test]
fn repeated_model_rejected_in_continue_mode_on_both_endpoints() {
    let echo_guard = start_echo_backend();
    let proxy_port = free_port();

    let config = Config::from_yaml(&rewrite_without_request_classifier_yaml(proxy_port, echo_guard.port())).unwrap();
    let proxy = start_proxy(&config);

    for (path, body) in [
        (
            "/v1/responses",
            r#"{"model":"codex-mini-latest","input":"hi","model":"gpt-4.1-mini"}"#,
        ),
        (
            "/v1/chat/completions",
            r#"{"model":"codex-mini-latest","model":"codex-mini-latest","messages":[]}"#,
        ),
    ] {
        let raw = http_send(proxy.addr(), &json_post(path, body));

        assert_eq!(parse_status(&raw), 400, "{path}: a repeated model must be refused");
        let error_body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
        assert_eq!(
            error_body["error"]["type"].as_str(),
            Some("invalid_request_error"),
            "{path}: rejection should have structured error type"
        );
    }
}

#[test]
fn repeated_model_never_reaches_an_effective_model_route() {
    let llama_guard = start_backend_with_shutdown("llama-backend");
    let qwen_guard = start_backend_with_shutdown("qwen-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = Config::from_yaml(&effective_model_routing_without_request_classifier_yaml(
        proxy_port,
        llama_guard.port(),
        qwen_guard.port(),
        default_guard.port(),
    ))
    .unwrap();
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1-mini","input":"ambiguous","model":"codex-mini-2026-06-24"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(
        parse_status(&raw),
        400,
        "a repeated model must be refused before routing"
    );
    let response = parse_body(&raw);
    for backend in ["llama-backend", "qwen-backend", "default-backend"] {
        assert_ne!(response, backend, "{backend} must not answer an ambiguous model");
    }

    // Control: the same route configuration still serves an unambiguous model.
    let body = r#"{"model":"gpt-4.1-mini","input":"control"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(
        parse_status(&raw),
        200,
        "an unambiguous model must still reach its route"
    );
    assert_eq!(
        parse_body(&raw),
        "qwen-backend",
        "the control request must route through the effective model header"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// YAML config with model rewrite + `on_invalid: reject` for malformed body testing.
fn reject_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ai_operation
      - filter: openai_responses_request
        initialize_state: false
      - filter: openai_responses_model_rewrite
        default_model: "llama-3.3-70b"
        model_aliases:
          codex-mini-latest: "llama-3.3-70b"
        on_invalid: reject
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

/// YAML config with model rewrite + echo backend.
fn rewrite_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ai_operation
      - filter: openai_responses_request
        initialize_state: false
      - filter: openai_responses_model_rewrite
        default_model: "llama-3.3-70b"
        model_aliases:
          codex-mini-latest: "llama-3.3-70b"
          "codex-*": "llama-3.3-70b"
          gpt-4.1-mini: "qwen-2.5-72b"
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

/// Rewrite-only config: no earlier `openai_responses_request` classifier.
///
/// This is the standard-build shape: `openai_responses_request` is experimental,
/// so model rewrite is the first filter to read the body model on both endpoints
/// and owns the duplicate-model check there.
fn rewrite_without_request_classifier_yaml(proxy_port: u16, backend_port: u16) -> String {
    without_request_classifier(&rewrite_yaml(proxy_port, backend_port))
}

/// Remove the `openai_responses_request` classifier from a test config.
///
/// Panics if the classifier entry is absent, so a drifted config cannot
/// silently leave the classifier answering in model rewrite's place.
fn without_request_classifier(yaml: &str) -> String {
    const CLASSIFIER: &str = "      - filter: openai_responses_request\n        initialize_state: false\n";
    assert_eq!(
        yaml.matches(CLASSIFIER).count(),
        1,
        "test config must contain exactly one request classifier entry to remove"
    );
    yaml.replace(CLASSIFIER, "")
}

/// YAML config with header-echo backend for content-length verification.
fn header_echo_yaml(proxy_port: u16, backend_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ai_operation
      - filter: openai_responses_request
        initialize_state: false
      - filter: openai_responses_model_rewrite
        default_model: "llama-3.3-70b"
        model_aliases:
          codex-mini-latest: "llama-3.3-70b"
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

/// YAML config that routes by effective model header to different backends.
fn effective_model_routing_yaml(proxy_port: u16, llama_port: u16, qwen_port: u16, default_port: u16) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ai_operation
      - filter: openai_responses_request
        initialize_state: false
      - filter: openai_responses_model_rewrite
        model_aliases:
          codex-mini-latest: "llama-3.3-70b"
          "codex-*": "llama-3.3-70b"
          gpt-4.1-mini: "qwen-2.5-72b"
      - filter: router
        routes:
          - path: "/v1/responses"
            headers:
              x-praxis-ai-effective-model: "llama-3.3-70b"
            cluster: "llama"
          - path: "/v1/responses"
            headers:
              x-praxis-ai-effective-model: "qwen-2.5-72b"
            cluster: "qwen"
          - path_prefix: "/"
            cluster: "default"
      - filter: load_balancer
        clusters:
          - name: "llama"
            endpoints:
              - "127.0.0.1:{llama_port}"
          - name: "qwen"
            endpoints:
              - "127.0.0.1:{qwen_port}"
          - name: "default"
            endpoints:
              - "127.0.0.1:{default_port}"
insecure_options:
  allow_private_endpoints: true
"#
    )
}

/// Effective-model routing config with no earlier request classifier.
fn effective_model_routing_without_request_classifier_yaml(
    proxy_port: u16,
    llama_port: u16,
    qwen_port: u16,
    default_port: u16,
) -> String {
    without_request_classifier(&effective_model_routing_yaml(
        proxy_port,
        llama_port,
        qwen_port,
        default_port,
    ))
}
