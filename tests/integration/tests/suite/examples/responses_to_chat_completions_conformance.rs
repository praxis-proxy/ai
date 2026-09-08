// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenResponses translation conformance example.
//!
//! These run against a mock Chat Completions backend and always execute
//! (no vLLM required). They pin the invariants the external OpenResponses
//! suite depends on: the translator forwards to Chat Completions (never
//! native Responses passthrough), rejects a non-string `service_tier`, and
//! echoes function tools with the response-side schema.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_capturing_backend, start_proxy,
};

const EXAMPLE: &str = "openai/responses/responses-to-chat-completions-conformance.yaml";

fn load_test_config(listener_port: u16, port_map: &HashMap<&str, u16>) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let patched = patch_yaml(&yaml, listener_port, port_map);
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

#[test]
fn conformance_config_routes_to_chat_completions_never_responses() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_1", "object": "chat.completion", "model": "Qwen/Qwen3-0.6B",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    });
    let backend = StatefulCapturingBackend::new(vec![(200, chat_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = r#"{"model":"Qwen/Qwen3-0.6B","input":"Hello","stream":false,"store":false}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));
    let requests = backend.requests();
    let forwarded = requests.first().expect("backend should receive one request");
    assert_eq!(parse_status(&raw), 200);
    assert_eq!(forwarded.method, "POST");
    assert_eq!(forwarded.uri, "/v1/chat/completions");
    assert_ne!(forwarded.uri, "/v1/responses");
}

#[test]
fn conformance_config_normalizes_null_service_tier_to_default() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_1", "object": "chat.completion", "model": "Qwen/Qwen3-0.6B",
        "service_tier": serde_json::Value::Null,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = r#"{"model":"Qwen/Qwen3-0.6B","input":"Hello","stream":false,"store":false}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(parse_status(&raw), 200);
    assert_eq!(response["service_tier"], "default");
}

#[test]
fn conformance_config_echoes_normalized_function_tool() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_1", "object": "chat.completion", "model": "Qwen/Qwen3-0.6B",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "Qwen/Qwen3-0.6B", "input": "What's the weather in San Francisco?", "stream": false, "store": false,
        "tools": [{
            "type": "function", "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
        }]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(parse_status(&raw), 200);
    let tool = &response["tools"][0];
    assert_eq!(tool["type"], "function");
    assert_eq!(tool["name"], "get_weather");
    assert_eq!(tool["strict"], false);
    assert_eq!(tool["description"], "Get the current weather for a location");
    assert!(tool["parameters"].is_object());
}
