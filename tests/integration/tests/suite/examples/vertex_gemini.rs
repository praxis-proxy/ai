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
    assert!(
        parsed["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-vertex-")),
        "a response without Vertex responseId must receive a generated completion id"
    );
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

#[test]
fn vertex_gemini_streaming_rejects_candidate_index_exceeding_requested_n() {
    let sse_body = "data: {\"candidates\":[{\"index\":0,\"content\":{\"parts\":[{\"text\":\"first\"}]}}]}\n\n\
                    data: {\"candidates\":[{\"index\":1,\"content\":{\"parts\":[{\"text\":\"unexpected\"}]}}]}\n\n";
    let backend = Backend::fixed(sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let request = r#"{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"Hi"}],"stream":true,"n":1}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", request));
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streaming response start status should be 200");
    assert!(
        body.contains("first"),
        "translated stream should contain first choice delta"
    );
    assert!(
        !body.contains("unexpected"),
        "out-of-bounds candidate frame must not be translated"
    );
    assert!(
        !body.contains("[DONE]"),
        "out-of-bounds candidate must prevent [DONE] sentinel"
    );
    assert!(
        body.contains("server_error"),
        "rejected stream must end with terminal error frame"
    );
    assert!(
        body.contains("upstream SSE stream ended with errors or was truncated"),
        "rejected stream error message must indicate terminal stream error"
    );

    drop(proxy);
}

#[test]
#[expect(clippy::too_many_lines, reason = "tests streaming tool call slot overflow")]
fn vertex_gemini_streaming_rejects_excess_tool_call_slots() {
    let mut sse_lines = Vec::new();
    for i in 0..129 {
        let frame_json = serde_json::json!({
            "candidates": [{
                "index": 0,
                "content": {
                    "parts": [{
                        "functionCall": {
                            "id": format!("id-{i}"),
                            "name": "f",
                            "args": {}
                        }
                    }]
                }
            }]
        });
        sse_lines.push(format!("data: {frame_json}\n\n"));
    }
    sse_lines.push(
        "data: {\"candidates\":[{\"index\":0,\"finishReason\":\"STOP\",\"content\":{\"parts\":[]}}]}\n\n".to_owned(),
    );
    let sse_body = sse_lines.join("");

    let backend = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "vertex/chat-completions-to-gemini.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let request = r#"{"model":"gemini-2.0-flash","messages":[{"role":"user","content":"Hi"}],"stream":true,"n":1}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", request));
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streaming response start status should be 200");
    assert!(body.contains("id-0"), "stream must contain first tool call slot delta");
    assert!(
        body.contains("id-127"),
        "stream must contain 128th tool call slot delta"
    );
    assert!(
        !body.contains("id-128"),
        "129th tool call slot must be rejected before translation"
    );
    assert!(
        !body.contains("[DONE]"),
        "tool slot overflow must prevent [DONE] sentinel"
    );
    assert!(
        body.contains("server_error"),
        "tool slot overflow must end with terminal error frame"
    );
    assert!(
        body.contains("upstream SSE stream ended with errors or was truncated"),
        "tool slot overflow terminal error must contain standard error message"
    );

    drop(proxy);
}
