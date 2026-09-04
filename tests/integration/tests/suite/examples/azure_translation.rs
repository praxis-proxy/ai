// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Azure OpenAI translation example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, free_port, http_send, json_post, parse_body, parse_status, start_capturing_backend, start_proxy,
};

use super::load_example_config;

fn azure_proxy(backend_port: u16) -> praxis_test_utils::ProxyGuard {
    let proxy_port = free_port();
    let config = load_example_config(
        "azure/chat-completions-to-openai.yaml",
        proxy_port,
        HashMap::from([("my-resource.openai.azure.com:443", backend_port)]),
    );
    start_proxy(&config)
}

#[test]
fn azure_translation_forwards_request_with_api_version() {
    let backend = start_capturing_backend(
        &serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        })
        .to_string(),
    );
    let proxy = azure_proxy(backend.port());

    let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"What is the capital of France?"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200);
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(response["choices"][0]["message"]["content"], "Paris");

    let forwarded: serde_json::Value = serde_json::from_str(&backend.body()).expect("captured body should be JSON");
    assert!(
        forwarded.get("model").is_none(),
        "model field should be stripped from Azure request"
    );
}

#[test]
fn azure_translation_strips_content_filter_fields() {
    let azure_response = serde_json::json!({
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "choices": [{
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
            "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}}
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}]
    });
    let backend = Backend::fixed(&azure_response.to_string())
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy = azure_proxy(backend.port());

    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");

    assert_eq!(parse_status(&raw), 200);
    assert!(
        response.get("prompt_filter_results").is_none(),
        "prompt_filter_results should be stripped"
    );
    assert!(
        response["choices"][0].get("content_filter_results").is_none(),
        "content_filter_results should be stripped"
    );
    assert_eq!(response["choices"][0]["message"]["content"], "Hello");
}

#[test]
fn azure_translation_normalizes_error_with_null_type() {
    let azure_error = serde_json::json!({
        "error": {
            "message": "The API deployment for this resource does not exist.",
            "type": null,
            "code": "DeploymentNotFound",
            "innererror": {"code": "DeploymentNotFound"}
        }
    });
    let backend = Backend::status(404, &azure_error.to_string())
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy = azure_proxy(backend.port());

    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let body_str = parse_body(&raw);

    assert_eq!(parse_status(&raw), 404);
    let response: serde_json::Value =
        serde_json::from_str(&body_str).unwrap_or_else(|e| panic!("error should be JSON: {e}\nbody: {body_str}"));
    assert_eq!(
        response["error"]["type"], "invalid_request_error",
        "null type should be filled from status"
    );
    assert_eq!(
        response["error"]["message"],
        "The API deployment for this resource does not exist."
    );
    assert!(
        response["error"].get("innererror").is_none(),
        "innererror should be stripped from normalized errors"
    );
}

#[test]
fn azure_translation_passes_through_valid_error() {
    let openai_error = serde_json::json!({
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    });
    let backend = Backend::status(429, &openai_error.to_string())
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy = azure_proxy(backend.port());

    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error should be JSON");

    assert_eq!(parse_status(&raw), 429);
    assert_eq!(
        response["error"]["type"], "rate_limit_error",
        "already-valid error type should pass through unchanged"
    );
}

#[test]
fn azure_translation_strips_content_filter_from_sse_events() {
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-abc\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"content_filter_results\":{}}]}\n\n",
        "data: {\"id\":\"chatcmpl-abc\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"content_filter_results\":{\"hate\":{\"filtered\":false}}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let backend = Backend::fixed(sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy = azure_proxy(backend.port());

    let body = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let response_body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200);
    assert!(
        !response_body.contains("content_filter_results"),
        "content_filter_results should be stripped from SSE events"
    );
    assert!(
        response_body.contains("\"content\":\"Hi\""),
        "content should be preserved"
    );
    assert!(response_body.contains("[DONE]"), "DONE sentinel should be preserved");
}
