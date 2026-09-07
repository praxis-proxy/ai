// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for hosted file search through a Chat Completions backend.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_body, parse_status,
    patch_yaml, start_capturing_backend, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "openai/responses/file-search-chat-completions.yaml";

fn load_test_config(listener_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let patched = patch_yaml(&yaml, listener_port, port_map);
    Config::from_yaml(&patched).expect("patched config should parse")
}

#[test]
fn file_search_chat_example_runs_model_search_model_round_trip() {
    let first_model_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "file_search",
                        "arguments": "{\"query\":\"What were the Q4 results?\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });
    let final_model_response = json!({
        "id": "chatcmpl_final",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Q4 revenue was $42 million <|file-q4|>"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, first_model_response.to_string()),
        (200, final_model_response.to_string()),
    ])
    .start_with_shutdown();
    let search = start_capturing_backend(
        &json!({
            "data": [{
                "file_id": "file-q4",
                "filename": "q4-results.txt",
                "score": 0.99,
                "content": [{"type": "text", "text": "Q4 revenue was $42 million."}],
                "attributes": null
            }]
        })
        .to_string(),
    );
    let proxy_port = free_port();
    let config = load_test_config(
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_proxy(&config);
    let file_search_tool = json!({
        "type": "file_search",
        "vector_store_ids": ["vs_q4"],
        "max_num_results": 5,
        "ranking_options": {"ranker": "auto", "score_threshold": 0.2}
    });
    let request = json!({
        "model": "chat-only-model",
        "input": "What do the uploaded documents say about Q4 results?",
        "include": ["file_search_call.results"],
        "tools": [file_search_tool.clone()],
        "tool_choice": {"type": "file_search"},
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(parse_status(&raw), 200, "round trip failed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("final response should be JSON");
    assert_eq!(response["object"], "response");
    assert_eq!(response["tools"], json!([file_search_tool]));
    assert_eq!(response["tool_choice"], json!({"type": "file_search"}));
    assert_eq!(response["output"][0]["type"], "file_search_call");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][0]["results"][0]["file_id"], "file-q4");
    assert_eq!(response["output"][1]["type"], "message");
    assert_eq!(
        response["output"][1]["content"][0]["text"],
        "Q4 revenue was $42 million"
    );
    assert_eq!(
        response["output"][1]["content"][0]["annotations"][0]["file_id"],
        "file-q4"
    );
    assert_eq!(
        response["usage"],
        json!({
            "input_tokens": 30,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 42
        })
    );

    let model_requests = model.requests();
    assert_eq!(model_requests.len(), 2, "file search should drive two model calls");
    for captured in &model_requests {
        assert_eq!(captured.uri, "/v1/chat/completions");
    }
    let first_forwarded: Value =
        serde_json::from_str(&model_requests[0].body).expect("first model request should be JSON");
    assert_eq!(first_forwarded["tools"][0]["type"], "function");
    assert_eq!(first_forwarded["tools"][0]["function"]["name"], "file_search");
    assert_eq!(
        first_forwarded["tools"][0]["function"]["parameters"]["required"],
        json!(["query"])
    );
    assert_eq!(
        first_forwarded["tools"][0]["function"]["parameters"]["properties"]["query"]["minLength"],
        1
    );
    assert_eq!(
        first_forwarded["tool_choice"],
        json!({"type": "function", "function": {"name": "file_search"}})
    );
    assert!(
        first_forwarded["tools"][0].get("vector_store_ids").is_none(),
        "hosted configuration must remain private to ResponsesState"
    );

    let second_forwarded: Value =
        serde_json::from_str(&model_requests[1].body).expect("second model request should be JSON");
    assert_eq!(second_forwarded["tool_choice"], "auto");
    assert!(second_forwarded["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["tool_calls"][0]["function"]["name"] == "file_search"
                && message["tool_calls"][0]["function"]["arguments"] == "{\"query\":\"What were the Q4 results?\"}"
        })
    }));
    assert!(second_forwarded["messages"].as_array().is_some_and(|messages| {
        messages
            .iter()
            .any(|message| message["role"] == "tool" && message["tool_call_id"].as_str().is_some())
    }));

    let search_request: Value = serde_json::from_str(&search.body()).expect("search request should be JSON");
    assert_eq!(search_request["query"], "What were the Q4 results?");
    assert_eq!(search_request["max_num_results"], 5);
    assert_eq!(search_request["rewrite_query"], false);
}

#[test]
fn file_search_chat_example_rejects_function_name_collision_before_forwarding() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"id":"unexpected"}"#.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "search",
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_q4"]},
            {"type": "function", "name": "file_search", "parameters": {"type": "object"}}
        ]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(
        parse_status(&raw),
        400,
        "collision should fail before forwarding: {raw}"
    );
    assert!(parse_body(&raw).contains("conflicts with the synthesized file_search function"));
    assert!(
        backend.requests().is_empty(),
        "collision must not reach the model backend"
    );
}

#[test]
fn file_search_chat_example_rejects_malformed_configuration_before_forwarding() {
    let backend = StatefulCapturingBackend::new(vec![(200, r#"{"id":"unexpected"}"#.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "search",
        "tools": [{"type": "file_search", "vector_store_ids": []}]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(
        parse_status(&raw),
        400,
        "malformed tool should fail before forwarding: {raw}"
    );
    assert!(parse_body(&raw).contains("vector_store_ids must be a non-empty array"));
    assert!(
        backend.requests().is_empty(),
        "malformed configuration must not reach the model backend"
    );
}
