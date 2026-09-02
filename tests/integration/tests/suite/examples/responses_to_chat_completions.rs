// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Responses-to-Chat Completions example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_get, http_send, json_post,
    parse_body, parse_status, patch_yaml, start_capturing_backend, start_proxy,
};

const EXAMPLE: &str = "openai/responses/responses-to-chat-completions.yaml";

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
    let (config, _db) = load_test_config(
        "translate_request_response",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
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
    let (config, _db) = load_test_config(
        "finite_provider_error",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
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
    let (config, _db) = load_test_config(
        "sse_passthrough",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
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
    let (config, _db) = load_test_config(
        "streaming_request",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let request = r#"{"model":"gpt-4.1-mini","input":"Hello","stream":true,"store":false}"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", request));
    let forwarded: serde_json::Value = serde_json::from_str(&backend.body()).expect("backend request should be JSON");

    assert_eq!(parse_status(&raw), 200);
    assert_eq!(forwarded["stream"], true);
    assert!(forwarded["messages"].is_array());
    assert_eq!(forwarded["messages"][0]["content"], "Hello");
    assert!(
        forwarded.get("input").is_none(),
        "Responses-only field `input` must not reach the Chat Completions backend"
    );
}

#[test]
fn responses_to_chat_completions_rehydrates_finite_continuation() {
    let first_chat_response = serde_json::json!({
        "id": "chatcmpl_first",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "First answer"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    });
    let second_chat_response = serde_json::json!({
        "id": "chatcmpl_second",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Second answer"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
    });
    let backend = StatefulCapturingBackend::new(vec![
        (200, first_chat_response.to_string()),
        (200, second_chat_response.to_string()),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "finite_continuation",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let first_raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1-mini","input":"First question","stream":false,"store":true}"#,
        ),
    );
    assert_eq!(parse_status(&first_raw), 200);
    let first_response: serde_json::Value =
        serde_json::from_str(&parse_body(&first_raw)).expect("first response should be JSON");
    let first_response_id = first_response["id"]
        .as_str()
        .expect("first response should have an id")
        .to_owned();

    let second_request = serde_json::json!({
        "model": "gpt-4.1-mini",
        "input": "Follow up",
        "previous_response_id": first_response_id,
        "stream": false,
        "store": false
    });
    let second_raw = http_send(proxy.addr(), &json_post("/v1/responses", &second_request.to_string()));
    assert_eq!(parse_status(&second_raw), 200);
    let second_response: serde_json::Value =
        serde_json::from_str(&parse_body(&second_raw)).expect("second response should be JSON");
    assert_eq!(second_response["previous_response_id"], first_response_id);

    let requests = backend.requests();
    assert_eq!(requests.len(), 2, "both turns should reach the Chat backend");
    let forwarded: serde_json::Value =
        serde_json::from_str(&requests[1].body).expect("second backend request should be JSON");
    assert_eq!(requests[1].uri, "/v1/chat/completions");
    assert_eq!(
        forwarded["messages"],
        serde_json::json!([
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow up"}
        ])
    );
    assert!(
        forwarded.get("input").is_none(),
        "Responses-only field `input` must not reach the Chat Completions backend"
    );
    assert!(
        forwarded.get("previous_response_id").is_none(),
        "Responses-only field `previous_response_id` must not reach the Chat Completions backend"
    );

    let (first_get_status, _) = http_get(proxy.addr(), &format!("/v1/responses/{first_response_id}"), None);
    assert_eq!(first_get_status, 200, "store=true response should remain retrievable");
    let second_response_id = second_response["id"]
        .as_str()
        .expect("second response should have an id");
    let (second_get_status, _) = http_get(proxy.addr(), &format!("/v1/responses/{second_response_id}"), None);
    assert_eq!(second_get_status, 404, "store=false response should not be persisted");
}

#[test]
fn responses_to_chat_completions_rejects_unknown_previous_response() {
    let backend = StatefulCapturingBackend::new(vec![(
        200,
        serde_json::json!({
            "id": "chatcmpl_unused",
            "object": "chat.completion",
            "model": "gpt-4.1-mini",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
        .to_string(),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config(
        "unknown_previous_response",
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1-mini","input":"Follow up","previous_response_id":"resp_missing","stream":false,"store":false}"#,
        ),
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error response should be JSON");

    assert_eq!(parse_status(&raw), 400);
    assert_eq!(response["error"]["type"], "invalid_request_error");
    assert!(
        backend.requests().is_empty(),
        "unknown predecessor must not reach backend"
    );
}
