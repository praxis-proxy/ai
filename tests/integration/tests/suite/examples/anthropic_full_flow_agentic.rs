// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Claude Code coverage for the unified agentic gateway example.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "agentic/full-flow-agentic.yaml";

fn load_config(proxy_port: u16, backend_port: u16) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new("anthropic_unified_full_flow");
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read unified example");
    let yaml = yaml
        .replace("sqlite://responses.db?mode=rwc", db.url())
        .replace("${WEB_SEARCH_API_KEY}", "test-key");
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", backend_port)]));
    let config = praxis_core::config::Config::from_yaml(&patched).expect("parse unified example");
    (config, db)
}

fn backend_with_response(body: &Value) -> praxis_test_utils::StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![
        (200, body.to_string()),
        (200, body.to_string()),
        (200, body.to_string()),
    ])
    .start_with_shutdown()
}

fn post_request(backend_model: &str, content: &Value) -> Value {
    json!({
        "model": backend_model,
        "max_tokens": 256,
        "system": "You are Claude Code.",
        "messages": [{"role": "user", "content": content}],
        "tools": [{
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }]
    })
}

fn authenticated_json_post(path: &str, body: &str) -> String {
    json_post(path, body).replacen(
        "\r\n\r\n",
        "\r\nx-auth-tenant: tenant-a\r\nx-auth-user: user-a\r\n\r\n",
        1,
    )
}

#[test]
fn unified_anthropic_path_has_no_server_owned_agentic_loop() {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read unified example");

    assert!(!yaml.contains("anthropic_web_search"));
    assert!(!yaml.contains("- filter: anthropic_messages_protocol"));
    assert!(yaml.contains("name: direct-anthropic"));
    assert!(yaml.contains("name: translated-anthropic"));
    assert!(yaml.contains("application_provider: anthropic_compat"));
}

#[test]
fn native_anthropic_preserves_claude_code_request_and_response() {
    let provider_response = json!({
        "id": "msg_native",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{
            "type": "tool_use",
            "id": "toolu_read_1",
            "name": "Read",
            "input": {"path": "src/main.rs"}
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    });
    let backend = backend_with_response(&provider_response);
    let proxy_port = free_port();
    let (config, _db) = load_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let request = post_request(
        "claude-sonnet-4-5",
        &json!([{"type": "text", "text": "Read src/main.rs"}]),
    );

    let raw = http_send(
        proxy.addr(),
        &authenticated_json_post("/v1/messages", &request.to_string()).replacen(
            "\r\n\r\n",
            "\r\nanthropic-version: 2023-06-01\r\nx-api-key: provider-key\r\n\r\n",
            1,
        ),
    );

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(
        serde_json::from_str::<Value>(&parse_body(&raw)).unwrap(),
        provider_response
    );
    let forwarded = backend
        .requests()
        .into_iter()
        .find(|request| request.method == "POST")
        .expect("native provider should receive a POST");
    assert_eq!(forwarded.uri, "/v1/messages");
    assert_eq!(serde_json::from_str::<Value>(&forwarded.body).unwrap(), request);
    assert!(
        forwarded
            .headers
            .to_ascii_lowercase()
            .contains("x-api-key: provider-key")
    );
    assert!(!forwarded.headers.to_ascii_lowercase().contains("x-tenant-id"));
    assert!(!forwarded.headers.to_ascii_lowercase().contains("x-user-id"));
}

#[test]
fn translated_anthropic_preserves_client_tool_cycle() {
    let chat_response = json!({
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "vllm-chat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "toolu_read_2",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{\"path\":\"Cargo.toml\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let backend = backend_with_response(&chat_response);
    let proxy_port = free_port();
    let (config, _db) = load_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let request = post_request(
        "vllm-chat",
        &json!([{
            "type": "tool_result",
            "tool_use_id": "toolu_previous",
            "content": "previous tool output"
        }]),
    );

    let raw = http_send(
        proxy.addr(),
        &authenticated_json_post("/v1/messages", &request.to_string()),
    );

    assert_eq!(parse_status(&raw), 200);
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("Anthropic response JSON");
    assert_eq!(response["type"], "message");
    assert_eq!(response["content"][0]["type"], "tool_use");
    assert_eq!(response["content"][0]["name"], "Read");

    let forwarded = backend
        .requests()
        .into_iter()
        .find(|request| request.method == "POST")
        .expect("Chat provider should receive a POST");
    assert_eq!(forwarded.uri, "/v1/chat/completions");
    let body: Value = serde_json::from_str(&forwarded.body).expect("Chat request JSON");
    assert!(body["messages"].as_array().is_some_and(|messages| {
        messages
            .iter()
            .any(|message| message["role"] == "tool" && message["tool_call_id"] == "toolu_previous")
    }));
}

#[test]
fn native_count_tokens_passes_through() {
    let provider_response = json!({"input_tokens": 42});
    let backend = backend_with_response(&provider_response);
    let proxy_port = free_port();
    let (config, _db) = load_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "count this"}]
    });

    let raw = http_send(
        proxy.addr(),
        &authenticated_json_post("/v1/messages/count_tokens", &request.to_string()),
    );

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(
        serde_json::from_str::<Value>(&parse_body(&raw)).unwrap(),
        provider_response
    );
    let forwarded = backend
        .requests()
        .into_iter()
        .find(|request| request.method == "POST")
        .expect("native provider should receive count_tokens");
    assert_eq!(forwarded.uri, "/v1/messages/count_tokens");
    assert_eq!(serde_json::from_str::<Value>(&forwarded.body).unwrap(), request);
}

#[test]
fn translated_count_tokens_returns_explicit_not_found() {
    let backend = backend_with_response(&json!({"unexpected": true}));
    let proxy_port = free_port();
    let (config, _db) = load_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "vllm-chat",
        "messages": [{"role": "user", "content": "count this"}]
    });

    let raw = http_send(
        proxy.addr(),
        &authenticated_json_post("/v1/messages/count_tokens", &request.to_string()),
    );

    assert_eq!(parse_status(&raw), 404);
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("Anthropic error JSON");
    assert_eq!(response["error"]["type"], "not_found_error");
    assert!(backend.requests().into_iter().all(|request| request.method != "POST"));
}
