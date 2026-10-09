// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses-to-Chat Completions reasoning example config.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

const EXAMPLE: &str = "openai/responses/responses-to-chat-completions-reasoning.yaml";
const EXAMPLE_AGENTIC: &str = "openai/responses/responses-to-chat-completions-reasoning-agentic.yaml";

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

/// Load the agentic variant of the reasoning example, which runs
/// `openai_agentic_loop` ahead of the translator so the output collector records
/// each round into the stored history.
fn load_agentic_config(
    test_name: &str,
    listener_port: u16,
    port_map: &HashMap<&str, u16>,
) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new(test_name);
    let yaml =
        std::fs::read_to_string(example_config_path(EXAMPLE_AGENTIC)).expect("agentic example config should exist");
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
            "role": "assistant", "content": null, "reasoning": "The user only wants me to think."
        })
    );
    assert_eq!(
        forwarded["messages"][2],
        serde_json::json!({"role": "user", "content": "Now answer."})
    );
}

#[test]
fn agentic_stored_reasoning_is_replayed_once() {
    // With openai_agentic_loop, the round collector already records the reasoning
    // into persisted history, so storage assembly must not append it a second time.
    // The continuation must forward the assistant answer with its reasoning exactly
    // once (a duplicate would exceed the per-item byte limit).
    let chat_response = serde_json::json!({
        "id": "chatcmpl_agentic_reasoning",
        "object": "chat.completion",
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "4", "reasoning": "Two plus two is four."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16}
    });
    let backend =
        StatefulCapturingBackend::new(vec![(200, chat_response.to_string()), (200, chat_response.to_string())])
            .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_agentic_config(
        "agentic_stored_reasoning",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "deepseek-r1",
        "input": "What is 2+2?",
        "reasoning": {"effort": "medium"},
        "stream": false,
        "store": true
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));
    assert_eq!(parse_status(&raw), 200);
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(response["status"], "completed");
    // The client-facing output keeps its announced order: reasoning then message.
    let types: Vec<&str> = response["output"]
        .as_array()
        .expect("output should be an array")
        .iter()
        .map(|item| item["type"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(types, ["reasoning", "message"]);

    let continuation = serde_json::json!({
        "model": "deepseek-r1",
        "previous_response_id": response["id"],
        "input": "Now what is that plus 10?",
        "store": false,
        "stream": false
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &continuation.to_string()));
    assert_eq!(parse_status(&raw), 200, "stored agentic continuation should succeed");

    let captured = backend.requests();
    assert_eq!(captured.len(), 2);
    let forwarded: serde_json::Value = serde_json::from_str(&captured[1].body).unwrap();
    // The reasoning is replayed exactly once, attached to its assistant answer.
    assert_eq!(
        forwarded["messages"][1],
        serde_json::json!({"role": "assistant", "content": "4", "reasoning": "Two plus two is four."})
    );
    assert_eq!(
        forwarded["messages"][2],
        serde_json::json!({"role": "user", "content": "Now what is that plus 10?"})
    );
    let assistant_turns = forwarded["messages"]
        .as_array()
        .expect("forwarded messages should be an array")
        .iter()
        .filter(|message| message["role"] == "assistant")
        .count();
    assert_eq!(
        assistant_turns, 1,
        "exactly one assistant turn; reasoning must not be duplicated"
    );
}

#[test]
fn rehydrated_reasoning_item_is_replayed_into_the_assistant_turn() {
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
        "reasoning_replay_field",
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
    assert_eq!(assistant["content"], "Done.", "ordinary content must be preserved");
    assert_eq!(
        assistant["reasoning"], "I picked 42.",
        "rehydrated reasoning must use the vLLM assistant reasoning field"
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

#[test]
fn streamed_reasoning_is_stored_and_replayed_with_the_assistant_answer() {
    let mut sse = String::new();
    for delta in [
        serde_json::json!({"reasoning": "I picked "}),
        serde_json::json!({"reasoning_content": "42."}),
        serde_json::json!({"content": "Done."}),
    ] {
        sse.push_str(&format!(
            "data: {}\n\n",
            serde_json::json!({
                "id":"chatcmpl_stream", "object":"chat.completion.chunk", "model":"deepseek-r1",
                "choices":[{"index":0, "delta":delta}]
            })
        ));
    }
    sse.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
    let backend = StatefulCapturingBackend::new(vec![
        (200, sse),
        (
            200,
            serde_json::json!({"id":"chatcmpl_followup", "object":"chat.completion", "model":"deepseek-r1",
            "choices":[{"index":0, "message":{"role":"assistant","content":"42"},"finish_reason":"stop"}]})
            .to_string(),
        ),
    ])
    .start_with_shutdown();
    let (config, _db) = load_test_config(
        "stream_reasoning_replay",
        free_port(),
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"deepseek-r1","input":"Pick a number.","stream":true,"store":true}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200);
    let body = parse_body(&raw);
    let events: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("translated event should be JSON"))
        .collect();
    let terminal = &events.last().expect("stream has a terminal")["response"];
    assert_eq!(terminal["status"], "completed");
    assert_eq!(terminal["output"][0]["content"][0]["text"], "I picked 42.");
    assert_eq!(terminal["output"][0]["summary"], serde_json::json!([]));
    assert_eq!(terminal["output"][1]["content"][0]["text"], "Done.");
    let reasoning_deltas: String = events
        .iter()
        .filter(|event| event["type"] == "response.reasoning_text.delta")
        .map(|event| event["delta"].as_str().unwrap())
        .collect();
    assert_eq!(reasoning_deltas, "I picked 42.");
    let continuation = serde_json::json!({"model":"deepseek-r1","previous_response_id":terminal["id"],
        "input":"Which number?","stream":false,"store":false});
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &continuation.to_string()));
    assert_eq!(parse_status(&raw), 200);
    let captured = backend.requests();
    assert_eq!(captured.len(), 2);
    let forwarded: serde_json::Value = serde_json::from_str(&captured[1].body).unwrap();
    assert_eq!(
        forwarded["messages"][1],
        serde_json::json!({"role":"assistant","content":"Done.","reasoning":"I picked 42."})
    );
}

#[test]
fn streamed_late_reasoning_is_stored_and_replayed_with_the_assistant_answer() {
    assert_late_reasoning_continuation(false);
}

#[test]
fn streamed_late_reasoning_followed_by_tool_call_replays_with_the_assistant_answer() {
    assert_late_reasoning_continuation(true);
}

fn assert_late_reasoning_continuation(with_tool_call: bool) {
    // vLLM can stream the assistant answer *before* any reasoning delta. The
    // message then claims output_index 0 and reasoning claims 1; the client-facing
    // terminal must preserve those announced positions ([message, reasoning]).
    // On a stored continuation the reasoning must still reattach to the assistant
    // turn it justifies rather than detaching into a standalone reasoning message.
    let mut sse = String::new();
    let mut deltas = vec![
        serde_json::json!({"content": "Done."}),
        serde_json::json!({"reasoning": "I picked "}),
        serde_json::json!({"reasoning_content": "42."}),
    ];
    if with_tool_call {
        deltas.push(
            serde_json::json!({"tool_calls": [{"index": 0, "id": "call_lookup", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}}]}),
        );
    }
    for delta in deltas {
        sse.push_str(&format!(
            "data: {}\n\n",
            serde_json::json!({
                "id":"chatcmpl_stream_late", "object":"chat.completion.chunk", "model":"deepseek-r1",
                "choices":[{"index":0, "delta":delta}]
            })
        ));
    }
    sse.push_str("data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
    let backend = StatefulCapturingBackend::new(vec![
        (200, sse),
        (
            200,
            serde_json::json!({"id":"chatcmpl_followup", "object":"chat.completion", "model":"deepseek-r1",
            "choices":[{"index":0, "message":{"role":"assistant","content":"42"},"finish_reason":"stop"}]})
            .to_string(),
        ),
    ])
    .start_with_shutdown();
    let (config, _db) = load_test_config(
        "stream_late_reasoning_replay",
        free_port(),
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"deepseek-r1","input":"Pick a number.","stream":true,"store":true}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200);
    let body = parse_body(&raw);
    let events: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("translated event should be JSON"))
        .collect();
    let terminal = &events.last().expect("stream has a terminal")["response"];
    assert_eq!(terminal["status"], "completed");
    // The client terminal preserves the announced positions: the message streamed
    // first (index 0) and reasoning arrived later (index 1).
    assert_eq!(terminal["output"][0]["type"], "message");
    assert_eq!(terminal["output"][0]["content"][0]["text"], "Done.");
    assert_eq!(terminal["output"][1]["type"], "reasoning");
    assert_eq!(terminal["output"][1]["content"][0]["text"], "I picked 42.");
    if with_tool_call {
        assert_eq!(terminal["output"][2]["type"], "function_call");
    }
    let input = if with_tool_call {
        serde_json::json!([
            {"type": "function_call_output", "call_id": "call_lookup", "output": "42"},
            {"role": "user", "content": "Which number?"}
        ])
    } else {
        serde_json::json!("Which number?")
    };
    let continuation = serde_json::json!({"model":"deepseek-r1","previous_response_id":terminal["id"],
        "input": input,"stream":false,"store":false});
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &continuation.to_string()));
    assert_eq!(parse_status(&raw), 200);
    let captured = backend.requests();
    assert_eq!(captured.len(), 2);
    let forwarded: serde_json::Value = serde_json::from_str(&captured[1].body).unwrap();
    // The reasoning reattaches to its assistant turn; it is not replayed as a
    // detached standalone reasoning message.
    assert_eq!(
        forwarded["messages"][1],
        serde_json::json!({"role":"assistant","content":"Done.","reasoning":"I picked 42."})
    );
    if with_tool_call {
        assert_eq!(forwarded["messages"][2]["tool_calls"][0]["id"], "call_lookup");
        assert!(forwarded["messages"][2].get("reasoning").is_none());
        assert_eq!(forwarded["messages"][3]["role"], "tool");
    }
    assert_eq!(
        forwarded["messages"].as_array().unwrap().last().unwrap(),
        &serde_json::json!({"role":"user","content":"Which number?"})
    );
}
