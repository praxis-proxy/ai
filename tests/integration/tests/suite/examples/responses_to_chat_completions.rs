// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses-to-Chat Completions example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, free_port, http_send, json_post, parse_body, parse_status,
    start_capturing_backend, start_proxy,
};

use super::load_example_config;

const EXAMPLE: &str = "openai/responses/responses-to-chat-completions.yaml";

#[test]
fn responses_to_chat_completions_translates_request_and_response() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Chat."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });
    let backend = StatefulCapturingBackend::new(vec![(200, chat_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1-mini",
        "instructions": "Be concise.",
        "input": "Hello",
        "stream": false,
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses/", &request.to_string()));
    let requests = backend.requests();
    let forwarded_request = requests.first().expect("backend should receive one request");
    let forwarded: serde_json::Value =
        serde_json::from_str(&forwarded_request.body).expect("backend request should be JSON");
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(forwarded_request.method, "POST");
    assert_eq!(forwarded_request.uri, "/v1/chat/completions");
    assert_eq!(forwarded["model"], "gpt-4.1-mini");
    assert_eq!(
        forwarded["messages"][0],
        serde_json::json!({"role": "system", "content": "Be concise."})
    );
    assert_eq!(forwarded["messages"][1]["role"], "user");
    assert_eq!(forwarded["messages"][1]["content"], "Hello");
    assert_eq!(forwarded["stream"], false);
    assert!(response["id"].as_str().is_some_and(|id| id.starts_with("resp_")));
    assert_eq!(response["object"], "response");
    assert_eq!(response["output"][0]["content"][0]["text"], "Hello from Chat.");
    assert_eq!(response["usage"]["input_tokens"], 4);
    assert_eq!(response["usage"]["output_tokens"], 3);
}

#[test]
fn responses_to_chat_completions_normalizes_finite_provider_error() {
    let backend = Backend::status(429, r#"{"error":{"code":"rate_limit_exceeded","message":"slow down"}}"#)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1-mini","input":"Hello","stream":false,"store":false}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error response should be JSON");

    assert_eq!(parse_status(&raw), 429);
    assert_eq!(response["error"]["code"], "rate_limit_exceeded");
    assert_eq!(response["error"]["type"], "rate_limit_exceeded");
    assert_eq!(response["error"]["message"], "slow down");
}

#[test]
fn responses_to_chat_completions_leaves_sse_for_stream_converter() {
    let sse = "data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";
    let backend = Backend::fixed(sse)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1-mini","input":"Hello","stream":true,"store":false}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(parse_body(&raw), sse);
}

#[test]
fn responses_to_chat_completions_translates_streaming_request_body() {
    let chat_response = serde_json::json!({
        "id": "chatcmpl_stream_fallback",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "finite fallback"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1-mini","input":"Hello","stream":true,"store":false}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));
    let forwarded: serde_json::Value = serde_json::from_str(&backend.body()).expect("backend request should be JSON");

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(forwarded["stream"], true);
    assert!(forwarded["messages"].is_array());
    assert_eq!(forwarded["messages"][0]["content"], "Hello");
    assert!(forwarded.get("input").is_none());
}
