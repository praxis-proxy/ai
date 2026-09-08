// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses-to-Chat Completions reasoning example config.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

const EXAMPLE: &str = "openai/responses/responses-to-chat-completions-reasoning.yaml";

fn load_test_config(
    test_name: &str,
    listener_port: u16,
    port_map: &HashMap<&str, u16>,
) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new(test_name);
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        listener_port,
        port_map,
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    (config, db)
}

#[test]
fn reasoning_dialect_promotes_raw_reasoning_to_a_reasoning_item() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "4",
                "reasoning": "2 plus 2 is 4.",
                "reasoning_content": null
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    });
    let backend = StatefulCapturingBackend::new(vec![(200, chat_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "reasoning_promotes_item",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": "What is 2+2? Reply with just the number.",
        "reasoning": {"effort": "medium"},
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");

    assert_eq!(parse_status(&raw), 200);

    let reasoning = &response["output"][0];
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
    assert_eq!(reasoning["content"][0]["text"], "2 plus 2 is 4.");
    assert_eq!(
        reasoning["summary"],
        serde_json::json!([]),
        "raw reasoning must never leak into the summary array"
    );
    assert_eq!(response["output"][1]["type"], "message");
    assert_eq!(response["output"][1]["content"][0]["text"], "4");
}

#[test]
fn reasoning_summary_request_is_rejected_before_forwarding() {
    let backend = StatefulCapturingBackend::new(vec![(
        200,
        serde_json::json!({
            "id": "chatcmpl_unused",
            "object": "chat.completion",
            "model": "deepseek-r1",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
        .to_string(),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "reasoning_summary_rejected",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": "What is 2+2?",
        "reasoning": {"summary": "auto"},
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error response should be JSON");

    assert_eq!(parse_status(&raw), 400);
    assert_eq!(response["error"]["code"], "invalid_request_error");
    assert!(
        backend.requests().is_empty(),
        "a summary request must not reach a dialect without a safe-summary contract"
    );
}
