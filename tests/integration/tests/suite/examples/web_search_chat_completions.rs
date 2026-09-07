// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Responses web search through a Chat Completions backend.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "openai/responses/web-search-chat-completions.yaml";

fn load_test_config(listener_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let yaml = patch_yaml(&yaml, listener_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!(
            "api_key: test-key\n                base_url: http://127.0.0.1:{search_port}\n                allow_private_base_url: true"
        ),
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("patched config should parse")
}

#[test]
fn web_search_chat_completions_example_builds() {
    let config = load_test_config(free_port(), 19_301, 19_302);
    let _pipeline = build_pipeline(&config);
}

#[test]
fn web_search_chat_completions_completes_model_search_model_round_trip() {
    let first_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"Praxis Proxy latest release\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    });
    let second_response = json!({
        "id": "chatcmpl_answer",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Praxis Proxy has a current release."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, first_response.to_string()),
        (200, second_response.to_string()),
    ])
    .start_with_shutdown();
    let search = StatefulCapturingBackend::new(vec![(
        200,
        json!({
            "web": {"results": [{
                "title": "Praxis Proxy releases",
                "url": "https://github.com/praxis-proxy/praxis/releases",
                "description": "Current Praxis Proxy releases."
            }]}
        })
        .to_string(),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "tools": [{
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "FR"}
        }],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(parse_status(&raw), 200, "round trip should succeed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(response["output"].as_array().map(Vec::len), Some(2));
    assert_eq!(response["output"][0]["type"], "web_search_call");
    assert_eq!(response["output"][0]["id"], "call_search_1");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][0]["action"]["query"], "Praxis Proxy latest release");
    assert_eq!(
        response["output"][0]["action"]["sources"][0]["url"],
        "https://github.com/praxis-proxy/praxis/releases"
    );
    assert_eq!(
        response["output"][1]["content"][0]["text"],
        "Praxis Proxy has a current release."
    );
    assert_eq!(response["tools"], request["tools"]);
    assert_eq!(response["tool_choice"], request["tool_choice"]);
    assert_eq!(response["usage"]["input_tokens"], 36);
    assert_eq!(response["usage"]["output_tokens"], 13);
    assert_eq!(response["usage"]["total_tokens"], 49);

    let model_requests = model.requests();
    assert_eq!(model_requests.len(), 2, "web search should drive two model calls");
    assert!(
        model_requests
            .iter()
            .all(|captured| captured.uri == "/v1/chat/completions")
    );
    let first_forwarded: Value = serde_json::from_str(&model_requests[0].body).expect("first request should be JSON");
    assert_eq!(first_forwarded["tools"][0]["type"], "function");
    assert_eq!(first_forwarded["tools"][0]["function"]["name"], "web_search");
    assert_eq!(
        first_forwarded["tools"][0]["function"]["parameters"]["required"],
        json!(["query"])
    );
    assert_eq!(
        first_forwarded["tool_choice"],
        json!({"type": "function", "function": {"name": "web_search"}})
    );
    assert!(first_forwarded["tools"][0].get("search_context_size").is_none());
    assert!(first_forwarded["tools"][0].get("user_location").is_none());

    let second_forwarded: Value = serde_json::from_str(&model_requests[1].body).expect("second request should be JSON");
    assert_eq!(second_forwarded["tool_choice"], "auto");
    let messages = second_forwarded["messages"]
        .as_array()
        .expect("messages should be an array");
    assert!(messages.iter().any(|message| {
        message["tool_calls"][0]["function"]["name"] == "web_search"
            && message["tool_calls"][0]["function"]["arguments"] == "{\"query\":\"Praxis Proxy latest release\"}"
    }));
    assert!(messages.iter().any(|message| {
        message["role"] == "tool"
            && message["content"]
                .as_str()
                .is_some_and(|content| content.contains("Praxis Proxy releases"))
    }));

    let search_requests = search.requests();
    assert_eq!(search_requests.len(), 1);
    assert!(search_requests[0].uri.contains("q=Praxis%20Proxy%20latest%20release"));
    assert!(search_requests[0].uri.contains("count=10"));
}

#[test]
fn web_search_chat_completions_rejects_function_name_collision_before_forwarding() {
    let model = StatefulCapturingBackend::new(vec![(200, "unexpected".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), free_port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "search",
        "tools": [
            {"type": "web_search"},
            {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
        ]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(
        parse_status(&raw),
        400,
        "collision should fail before forwarding: {raw}"
    );
    assert!(parse_body(&raw).contains("conflicts with the synthesized web_search function"));
    assert!(
        model.requests().is_empty(),
        "collision must not reach the model backend"
    );
}
