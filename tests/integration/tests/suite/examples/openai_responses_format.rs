// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenAI Responses format-routing example config.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Running format-routing example with the three documented backends.
struct Harness {
    /// Responses cluster.
    _responses: praxis_test_utils::BackendGuard,
    /// Chat Completions cluster.
    _chat: praxis_test_utils::BackendGuard,
    /// Default cluster.
    _default: praxis_test_utils::BackendGuard,
    /// The running proxy.
    proxy: praxis_test_utils::ProxyGuard,
}

/// Start the example config against all three backends.
fn start() -> Harness {
    let responses = start_backend_with_shutdown("responses-backend");
    let chat = start_backend_with_shutdown("chat-backend");
    let default = start_backend_with_shutdown("default-backend");
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/responses/format-routing.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", responses.port()),
            ("127.0.0.1:3002", chat.port()),
            ("127.0.0.1:3003", default.port()),
        ]),
    );
    let proxy = start_proxy(&config);
    Harness {
        _responses: responses,
        _chat: chat,
        _default: default,
        proxy,
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn openai_responses_format_routing_example_routes_responses_input() {
    let h = start();

    let body = r#"{"model":"gpt-4.1-mini","input":"Hello, world!"}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "responses request should return 200");
    assert_eq!(
        parse_body(&raw),
        "responses-backend",
        "responses input should route to responses-backend cluster"
    );
}

#[test]
fn openai_responses_format_routing_example_routes_chat_completions() {
    let h = start();

    let body = r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/v1/chat/completions", body));

    assert_eq!(parse_status(&raw), 200, "chat completions should return 200");
    assert_eq!(
        parse_body(&raw),
        "chat-backend",
        "chat completions should route to chat-backend cluster"
    );
}

#[test]
fn openai_responses_format_routing_example_chat_identity_ignores_the_body() {
    let h = start();

    for (name, request) in [
        (
            "malformed JSON",
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
             Content-Type: application/json\r\nContent-Length: 8\r\nConnection: close\r\n\r\n\
             not json",
        ),
        (
            "empty body",
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\
             Content-Length: 0\r\nConnection: close\r\n\r\n",
        ),
        (
            "list has no body",
            "GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        ),
    ] {
        let raw = http_send(h.proxy.addr(), request);
        assert_eq!(parse_status(&raw), 200, "{name} should return 200");
        assert_eq!(
            parse_body(&raw),
            "chat-backend",
            "{name} must still route to chat-backend without reading the body"
        );
    }
}

#[test]
fn openai_responses_format_routing_example_unsupported_chat_method_falls_to_default() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "PUT /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "unsupported method should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "PUT /v1/chat/completions must not be classified as Chat Completions"
    );
}

#[test]
fn openai_responses_format_routing_example_unknown_falls_to_default() {
    let h = start();

    let body = r#"{"prompt":"hello"}"#;
    let raw = http_send(h.proxy.addr(), &json_post("/other/path", body));

    assert_eq!(parse_status(&raw), 200, "unknown path should return 200");
    assert_eq!(
        parse_body(&raw),
        "default-backend",
        "unrecognized path should fall to default cluster"
    );
}
