// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tests for the AI guardrails example configuration.

use std::collections::HashMap;

use praxis_test_utils::{free_port, http_send, json_post, parse_body, parse_status, start_echo_backend, start_proxy};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn ai_guardrails_config_parses() {
    let config = super::load_example_config(
        "ai-guardrails.yaml",
        29920,
        HashMap::from([("127.0.0.1:3000", 29921_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn ai_guardrails_passes_chat_request_through() {
    let backend_guard = start_echo_backend();
    let backend_port = backend_guard.port();
    let proxy_port = free_port();

    let config = super::load_example_config(
        "ai-guardrails.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    );

    let proxy = start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "request should reach backend");
    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("backend should echo valid JSON");
    assert_eq!(parsed["model"], "gpt-4o", "body should pass through unchanged");
}
