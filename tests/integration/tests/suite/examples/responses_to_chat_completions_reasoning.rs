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
fn reasoning_only_completion_survives_stored_continuation() {
    // A completed choice whose only output is raw reasoning (content null) must
    // translate into a reasoning output item, not be rejected as empty.
    let chat_response = serde_json::json!({
        "id": "chatcmpl_reasoning_only",
        "object": "chat.completion",
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning": "The user only wants me to think."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
    });
    let backend =
        StatefulCapturingBackend::new(vec![(200, chat_response.to_string()), (200, chat_response.to_string())])
            .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "reasoning_only_completion",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": "Think about 2+2 but do not answer.",
        "reasoning": {"effort": "medium"},
        "stream": false,
        "store": true
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(response["status"], "completed");
    let output = response["output"].as_array().expect("output should be an array");
    assert_eq!(
        output.len(),
        1,
        "reasoning-only completion should yield exactly one item"
    );
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["text"], "The user only wants me to think.");
    assert_eq!(
        output[0]["summary"],
        serde_json::json!([]),
        "raw reasoning must never leak into the summary array"
    );
    let continuation = serde_json::json!({
        "model": "deepseek-r1",
        "previous_response_id": response["id"],
        "input": "Now answer.",
        "store": false,
        "stream": false
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &continuation.to_string()));
    assert_eq!(
        parse_status(&raw),
        200,
        "stored reasoning-only continuation should succeed"
    );
    let captured = backend.requests();
    assert_eq!(captured.len(), 2);
    let forwarded: serde_json::Value = serde_json::from_str(&captured[1].body).unwrap();
    assert_eq!(
        forwarded["messages"][1],
        serde_json::json!({
            "role": "assistant", "content": "<think>The user only wants me to think.</think>"
        })
    );
    assert_eq!(
        forwarded["messages"][2],
        serde_json::json!({"role": "user", "content": "Now answer."})
    );
}

#[test]
fn rehydrated_reasoning_item_is_replayed_inline_into_the_assistant_turn() {
    // On a continuation the client echoes a prior reasoning item ahead of the
    // assistant turn it produced.
    let chat_response = serde_json::json!({
        "id": "chatcmpl_replay",
        "object": "chat.completion",
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "42", "reasoning": "It is the answer."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    });
    let backend = StatefulCapturingBackend::new(vec![(200, chat_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "reasoning_replay_inline",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": [
            {"role": "user", "content": "Pick a number and remember it."},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "I picked 42."}]},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "What number did you pick?"}
        ],
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    assert_eq!(parse_status(&raw), 200);

    let captured = backend.requests();
    let forwarded: serde_json::Value =
        serde_json::from_str(&captured[0].body).expect("forwarded chat request should be JSON");
    let messages = forwarded["messages"].as_array().expect("messages should be an array");
    let assistant = messages
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("the assistant turn should be forwarded");
    assert_eq!(
        assistant["content"], "<think>I picked 42.</think>Done.",
        "rehydrated reasoning must be replayed inline in the assistant turn"
    );
    // The raw chain-of-thought must never surface as a separate top-level field.
    assert!(
        assistant.get("reasoning").is_none(),
        "replayed reasoning must not be forwarded as a separate reasoning field"
    );
}

#[test]
fn non_object_reasoning_block_is_rejected_before_forwarding() {
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
        "non_object_reasoning_rejected",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": "What is 2+2?",
        "reasoning": true,
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error response should be JSON");

    assert_eq!(parse_status(&raw), 400);
    assert_eq!(response["error"]["code"], "invalid_request_error");
    assert!(
        backend.requests().is_empty(),
        "a malformed reasoning block must not reach the backend"
    );
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
