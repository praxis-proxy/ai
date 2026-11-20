// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the prompt-length routing example configuration.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_post, http_send, load_example_config, parse_status, start_backend_with_shutdown, start_proxy,
};

/// Example configuration exercised by these tests.
const EXAMPLE: &str = "prompt-length-routing.yaml";

/// Build a raw JSON POST with a client-supplied prompt route header.
fn prompt_post_with_route(body: &str, route: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         X-Prompt-Route: {route}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

#[test]
fn large_prompt_routes_to_large_context_cluster() {
    let large_guard = start_backend_with_shutdown("large-context");
    let default_guard = start_backend_with_shutdown("default");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", large_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"small-model","messages":[{"role":"user","content":"Explain this very long prompt in enough detail to overflow the configured input limit."}]}"#,
    );

    assert_eq!(status, 200, "large prompt should be proxied");
    assert_eq!(body, "large-context", "large prompt should use the overflow cluster");
}

#[test]
fn small_prompt_stays_on_default_cluster() {
    let large_guard = start_backend_with_shutdown("large-context");
    let default_guard = start_backend_with_shutdown("default");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", large_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);

    let (status, body) = http_post(
        proxy.addr(),
        "/v1/chat/completions",
        r#"{"model":"small-model","messages":[{"role":"user","content":"hello"}]}"#,
    );

    assert_eq!(status, 200, "small prompt should be proxied");
    assert_eq!(body, "default", "small prompt should not escalate");
}

#[test]
fn spoofed_overflow_header_cannot_route_a_small_prompt() {
    let large_guard = start_backend_with_shutdown("large-context");
    let default_guard = start_backend_with_shutdown("default");
    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", large_guard.port()),
            ("127.0.0.1:3002", default_guard.port()),
        ]),
    );
    let proxy = start_proxy(&config);
    let request = prompt_post_with_route(
        r#"{"model":"small-model","messages":[{"role":"user","content":"hello"}]}"#,
        "large-context",
    );

    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "small prompt should be proxied: {raw}");
    assert!(
        raw.contains("default"),
        "spoofed header must not select overflow: {raw}"
    );
    assert!(
        !raw.contains("large-context"),
        "client-supplied route must be removed before routing: {raw}"
    );
}
