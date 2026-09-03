// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for response-phase guardrails using the
//! `nemo-guardrails-response.yaml` example config.

use std::collections::HashMap;

use praxis_test_utils::{Backend, BackendGuard, free_port, http_post, start_backend_with_shutdown, start_proxy};

use super::load_example_config;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Build a Chat Completion JSON response body with the given assistant content.
///
/// The body is padded to be large enough that error JSON replacements
/// (via `fit_to_committed_length`) are not truncated during tests.
fn chat_completion_body(content: &str) -> String {
    serde_json::json!({
        "id": "chatcmpl-test-0000000000000000000000000000000000000000",
        "object": "chat.completion",
        "created": 1_700_000_000_i64,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    })
    .to_string()
}

/// Start a mock NeMo server returning the given JSON body.
fn nemo_mock(body: &'static str) -> BackendGuard {
    Backend::status(200, body)
        .header("Content-Type", "application/json")
        .start_with_shutdown()
}

/// Start a mock upstream that returns a Chat Completion response.
fn chat_backend(content: &str) -> BackendGuard {
    let body = chat_completion_body(content);
    Backend::status(200, &body)
        .header("Content-Type", "application/json")
        .start_with_shutdown()
}

fn load_response_config(proxy_port: u16, backend_port: u16, nemo_port: u16) -> praxis_core::config::Config {
    load_example_config(
        "nemo-guardrails-response.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:3001", nemo_port)]),
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn response_guardrails_config_parses_correctly() {
    let config = load_response_config(free_port(), 29990, 29991);
    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
}

/// NeMo returns `"success"` for the upstream response - the original Chat
/// Completion body is forwarded to the client unchanged.
#[test]
fn response_guardrails_pass_forwards_upstream_body() {
    let backend = chat_backend("Hello! I'm doing well.");
    let nemo = nemo_mock(r#"{"status":"success","rails_status":{"check output rail":{"status":"success"}}}"#);
    let proxy_port = free_port();
    let config = load_response_config(proxy_port, backend.port(), nemo.port());
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Hello"}]}"#,
    );

    assert_eq!(status, 200, "NeMo 'success' should forward upstream response");
    let json: serde_json::Value = serde_json::from_str(&body).expect("response should be JSON");
    assert_eq!(
        json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str()),
        Some("Hello! I'm doing well."),
        "upstream Chat Completion body should reach the client unchanged"
    );
}

/// NeMo returns `"blocked"` for the upstream response - the body is replaced
/// with a JSON error payload (status remains 200 because headers are
/// already committed).
#[test]
fn response_guardrails_block_replaces_body() {
    let backend = chat_backend("toxic content that should be blocked");
    let nemo = nemo_mock(r#"{"status":"blocked","rails_status":{"toxicity":{"status":"blocked"}}}"#);
    let proxy_port = free_port();
    let config = load_response_config(proxy_port, backend.port(), nemo.port());
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"Hello"}]}"#,
    );

    assert_eq!(
        status, 200,
        "response-phase block keeps 200 (headers already committed)"
    );
    let trimmed = body.trim();
    let json: serde_json::Value = serde_json::from_str(trimmed).expect("replaced body should be valid JSON");
    let error = json.get("error").expect("body should have 'error' key");
    assert_eq!(error.get("code").and_then(|v| v.as_str()), Some("content_blocked"),);
    assert!(
        error
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("toxicity"),
        "error message should include the triggered rail name"
    );
}

/// NeMo returns `"error"` for the upstream response - the body is replaced
/// with an error JSON payload (not a 500).
#[test]
fn response_guardrails_error_replaces_body() {
    let backend = chat_backend("hello");
    let nemo = nemo_mock(r#"{"status":"error","rails_status":{},"guardrails_data":{"error":"Config load failed."}}"#);
    let proxy_port = free_port();
    let config = load_response_config(proxy_port, backend.port(), nemo.port());
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(
        status, 200,
        "response-phase error keeps 200 (headers already committed)"
    );
    let trimmed = body.trim();
    let json: serde_json::Value = serde_json::from_str(trimmed).expect("error body should be valid JSON");
    assert!(json.get("error").is_some(), "body should contain error payload");
}

/// NeMo is unreachable - the response body is replaced with an error payload.
#[test]
fn response_guardrails_provider_down_replaces_body() {
    let backend = chat_backend("hello world, this is a safe response from the upstream");
    let dead_port = free_port();
    let proxy_port = free_port();
    let config = load_response_config(proxy_port, backend.port(), dead_port);
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(
        status, 200,
        "response-phase provider failure keeps 200 (headers already committed)"
    );
    assert!(
        body.contains(r#""error""#),
        "body should be replaced with an error payload; got: {body}"
    );
}

/// Upstream returns a non-Chat-Completion body (e.g. plain text) - guardrails
/// cannot parse it, body is replaced with an error payload.
#[test]
fn response_guardrails_non_chat_body_replaces_body() {
    let long_text = "x".repeat(512);
    let backend = start_backend_with_shutdown(&long_text);
    let nemo = nemo_mock(r#"{"status":"success"}"#);
    let proxy_port = free_port();
    let config = load_response_config(proxy_port, backend.port(), nemo.port());
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"test","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(
        status, 200,
        "response-phase parse failure keeps 200 (headers already committed)"
    );
    assert!(
        body.contains(r#""error""#),
        "body should be replaced with an error payload; got: {body}"
    );
}
