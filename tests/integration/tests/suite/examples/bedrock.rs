// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the Bedrock Converse translation example.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, parse_body, parse_status, start_capturing_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn bedrock_converse_translates_capital_of_france_end_to_end() {
    let bedrock_response = serde_json::json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Paris is the capital of France."}]
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19}
    });
    let backend = start_capturing_backend(&bedrock_response.to_string());
    let proxy_port = free_port();

    let path = praxis_test_utils::example_config_path("bedrock/chat-completions-to-converse.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched =
        praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend.port())]))
            .replace(
                "access_key_env_var: AWS_ACCESS_KEY_ID",
                "access_key_env_var: CARGO_PKG_NAME",
            )
            .replace(
                "secret_key_env_var: AWS_SECRET_ACCESS_KEY",
                "secret_key_env_var: CARGO_MANIFEST_DIR",
            );
    let config = praxis_core::config::Config::from_yaml(&patched)
        .unwrap_or_else(|e| panic!("parse bedrock/chat-completions-to-converse.yaml: {e}"));
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "anthropic.claude-3-haiku-20240307-v1:0",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    });
    let raw = http_send(proxy.addr(), &json_post("/v1/chat/completions", &request.to_string()));
    let response: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("translated response should be JSON");
    let upstream: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("translated upstream request should be JSON");

    assert_eq!(parse_status(&raw), 200, "translation should preserve success status");
    assert_eq!(
        response["choices"][0]["message"]["content"], "Paris is the capital of France.",
        "caller should receive OpenAI Chat Completions content"
    );
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["total_tokens"], 19);
    assert_eq!(
        upstream["messages"][0]["content"][0]["text"], "What is the capital of France?",
        "Bedrock should receive the translated Converse content block"
    );
    assert!(
        upstream.get("model").is_none(),
        "Bedrock Converse carries the model in the request path, not the body"
    );
}
