// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Anthropic Messages example configs.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, Recording, free_port, http_send, json_post, parse_body, parse_status, start_backend_with_shutdown,
    start_capturing_backend, start_header_echo_backend, start_proxy,
};

use super::load_example_config;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn anthropic_validate_forwards_valid_request() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", body));

    assert_eq!(parse_status(&raw), 200, "valid request should be forwarded");
    assert_eq!(parse_body(&raw), "ok", "request should reach the backend");
}

#[test]
fn anthropic_validate_forwards_backend_owned_semantics() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"claude-opus-4-8","messages":[{"role":"user","content":"Hello"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "backend-owned Anthropic semantics should be forwarded"
    );
    assert_eq!(parse_body(&raw), "ok", "request should reach the backend");
}

#[test]
fn anthropic_validate_rejects_malformed_json() {
    let backend_guard = start_backend_with_shutdown("ok");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/request-validate.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", "not json {{{"));
    let parsed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error body should be JSON");

    assert_eq!(parse_status(&raw), 400, "malformed JSON should be rejected");
    assert_eq!(
        parsed["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "error type should be invalid_request_error"
    );
}

#[test]
fn anthropic_messages_protocol_injects_default_version() {
    let backend_guard = start_header_echo_backend();
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-protocol.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", body));
    let echoed = parse_body(&raw).to_lowercase();

    assert_eq!(parse_status(&raw), 200, "protocol request should return 200");
    assert!(
        echoed.contains("anthropic-version: 2023-06-01"),
        "backend should receive injected anthropic-version header: {echoed}"
    );
}

#[test]
fn anthropic_to_openai_transforms_response_body() {
    let recording = Recording::load("anthropic/messages/to_openai_non_streaming.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend_guard = Backend::fixed(&response_body)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request_body));
    let transformed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response body should be JSON");

    assert_eq!(parse_status(&raw), 200, "transformation should return 200");
    assert_eq!(transformed["type"], "message", "response should be Anthropic message");
    assert_eq!(
        transformed["content"][0]["text"], "Hello from a Chat Completions backend.",
        "Chat Completions response text should map to Anthropic content"
    );
    assert_eq!(
        transformed["usage"]["input_tokens"], 11,
        "prompt tokens should map to input tokens"
    );
}

#[test]
fn anthropic_to_openai_transforms_tool_cycle_request_body() {
    let recording = Recording::load("anthropic/messages/tool_result.json");
    let request_body = recording.request_body();
    let chat_response = serde_json::json!({
        "id": "chatcmpl_tool_cycle",
        "object": "chat.completion",
        "model": recording.request["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "15 times 7 equals 105."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request_body));
    let transformed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response body should be JSON");
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    let messages = forwarded["messages"]
        .as_array()
        .expect("backend request should include Chat Completions messages");

    assert_eq!(parse_status(&raw), 200, "tool-cycle translation should return 200");
    assert_eq!(
        forwarded["tools"][0]["function"]["name"], "calculator",
        "Anthropic tool definitions should become Chat Completions function tools"
    );
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["name"], "calculator",
        "assistant tool_use should become an OpenAI tool call"
    );
    assert_eq!(
        messages[1]["tool_calls"][0]["function"]["arguments"], r#"{"expression":"15 * 7"}"#,
        "tool_use input should become JSON string arguments"
    );
    assert_eq!(messages[2]["role"], "tool", "tool_result should become a tool message");
    assert_eq!(
        messages[2]["tool_call_id"], "toolu_test01",
        "tool_result should preserve the tool call id"
    );
    assert_eq!(
        messages[2]["content"], "105",
        "tool_result content should be visible to the backend"
    );
    assert_eq!(
        transformed["content"][0]["text"], "15 times 7 equals 105.",
        "Chat Completions response should still map back to Anthropic"
    );

    drop(proxy);
}

#[test]
fn anthropic_to_openai_transforms_streaming_response_body() {
    let recording = Recording::load("anthropic/messages/to_openai_streaming.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend_guard = Backend::fixed(&response_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request_body));
    let transformed = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "stream transformation should return 200");
    assert!(
        transformed.contains("event: message_start"),
        "stream should include Anthropic message_start"
    );
    assert!(
        transformed.contains("text_delta") && transformed.contains("Hello"),
        "Chat Completions delta should map to Anthropic text_delta"
    );
    assert!(
        transformed.contains("event: message_stop"),
        "stream should include Anthropic message_stop"
    );
}

#[test]
fn unified_gateway_routes_anthropic_to_correct_backend() {
    let anthropic_guard = start_backend_with_shutdown("anthropic-backend");
    let openai_guard = start_backend_with_shutdown("openai-backend");
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/unified-gateway.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", anthropic_guard.port()),
            ("127.0.0.1:3002", openai_guard.port()),
            ("127.0.0.1:3003", responses_guard.port()),
            ("127.0.0.1:3004", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let anthropic_body = r#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", anthropic_body));
    assert_eq!(parse_status(&raw), 200, "anthropic request should return 200");
    assert_eq!(
        parse_body(&raw),
        "anthropic-backend",
        "anthropic should route to anthropic-backend"
    );
}

#[test]
fn unified_gateway_routes_openai_to_correct_backend() {
    let anthropic_guard = start_backend_with_shutdown("anthropic-backend");
    let openai_guard = start_backend_with_shutdown("openai-backend");
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/unified-gateway.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", anthropic_guard.port()),
            ("127.0.0.1:3002", openai_guard.port()),
            ("127.0.0.1:3003", responses_guard.port()),
            ("127.0.0.1:3004", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let openai_body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", openai_body));
    assert_eq!(parse_status(&raw), 200, "openai request should return 200");
    assert_eq!(
        parse_body(&raw),
        "openai-backend",
        "openai should route to openai-backend"
    );
}

#[test]
fn unified_gateway_routes_responses_to_correct_backend() {
    let anthropic_guard = start_backend_with_shutdown("anthropic-backend");
    let openai_guard = start_backend_with_shutdown("openai-backend");
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/unified-gateway.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", anthropic_guard.port()),
            ("127.0.0.1:3002", openai_guard.port()),
            ("127.0.0.1:3003", responses_guard.port()),
            ("127.0.0.1:3004", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let responses_body = r#"{"model":"gpt-4","input":"What is 2+2?"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", responses_body));
    assert_eq!(parse_status(&raw), 200, "responses request should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "responses should route to responses-backend"
    );
}

#[test]
fn unified_gateway_routes_unknown_to_default_backend() {
    let anthropic_guard = start_backend_with_shutdown("anthropic-backend");
    let openai_guard = start_backend_with_shutdown("openai-backend");
    let responses_guard = start_backend_with_shutdown("responses-backend");
    let default_guard = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/unified-gateway.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", anthropic_guard.port()),
            ("127.0.0.1:3002", openai_guard.port()),
            ("127.0.0.1:3003", responses_guard.port()),
            ("127.0.0.1:3004", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/unknown", r#"{"foo":"bar"}"#));
    assert_eq!(parse_status(&raw), 200, "unknown path should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "unknown path should route to default-backend"
    );
}
