// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Vertex AI Gemini example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, Recording, free_port, http_send, json_post, parse_body, parse_status, start_capturing_backend, start_proxy,
};

use super::load_example_config;

// -----------------------------------------------------------------------------
// Non-streaming
// -----------------------------------------------------------------------------

#[test]
fn vertex_gemini_non_streaming_translates_response() {
    let recording = Recording::load("vertex/gemini/basic_non_streaming.json");
    let response_body = recording.response_body();
    let backend = start_capturing_backend(&response_body);
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/chat/completions", &recording.request_body()),
    );
    let status = parse_status(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be valid JSON");

    assert_eq!(status, 200, "non-streaming translation should return 200");
    assert_eq!(parsed["object"], "chat.completion");
    assert_eq!(parsed["model"], "gemini-2.0-flash");
    assert_eq!(
        parsed["choices"][0]["message"]["content"], "4",
        "Gemini text part should become OpenAI message content"
    );
    assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    assert_eq!(parsed["usage"]["prompt_tokens"], 12);
    assert_eq!(parsed["usage"]["completion_tokens"], 1);
    assert_eq!(parsed["usage"]["total_tokens"], 13);

    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    assert!(
        forwarded.get("contents").is_some(),
        "request should be translated to Gemini format with 'contents'"
    );
    assert!(
        forwarded.get("messages").is_none(),
        "OpenAI 'messages' should not be forwarded to Vertex"
    );

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Streaming
// -----------------------------------------------------------------------------

#[test]
fn vertex_gemini_streaming_translates_sse_frames() {
    let recording = Recording::load("vertex/gemini/basic_streaming.json");
    let response_body = recording.response_body();
    let backend = Backend::fixed(&response_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/chat/completions", &recording.request_body()),
    );
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streaming translation should return 200");
    assert!(
        body.contains("chat.completion.chunk"),
        "stream should contain OpenAI chunk objects"
    );
    assert!(
        body.contains("Hello"),
        "translated stream should preserve the content text"
    );
    assert!(body.contains("[DONE]"), "stream should end with [DONE]");
}

// -----------------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------------

#[test]
fn vertex_gemini_rejects_empty_body() {
    let backend = Backend::fixed("unused").start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", ""));
    let parsed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error should be valid JSON");

    assert_eq!(parse_status(&raw), 400, "empty body should be rejected with 400");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
}

#[test]
fn vertex_gemini_rejects_missing_model() {
    let backend = Backend::fixed("unused").start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", body));
    let parsed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("error should be valid JSON");

    assert_eq!(parse_status(&raw), 400, "missing model should be rejected with 400");
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
}
