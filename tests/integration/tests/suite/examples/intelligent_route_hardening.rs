// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Hardening tests for `intelligent_route` (static config).
//!
//! Covers the contract tracked in praxis-proxy/ai#1039:
//!   1. fail-closed on unknown model (deterministic 4xx, no upstream call), and
//!   2. anti-spoofing on the client-supplied routing model header.
//!
//! (Item 3, the management-path skip list, is tracked separately.)
//!
//! These drive the real proxy end-to-end against capturing mock backends so
//! the behavior is observed at the wire, not asserted from filter internals.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, load_example_config, parse_status, start_backend_with_shutdown, start_proxy,
    start_stateful_backend,
};

const INFERENCE_EXAMPLE: &str = "intelligent-route-inference.yaml";

/// Endpoints as written in `intelligent-route-inference.yaml`.
const GRANITE_ENDPOINT: &str = "127.0.0.1:8001";
const LLAMA_ENDPOINT: &str = "127.0.0.1:8002";

/// Build a raw chat-completions POST carrying `body` plus arbitrary extra
/// header lines (each already `Name: value`, no CRLF).
fn chat_post_with_headers(body: &str, extra_headers: &[&str]) -> String {
    let mut headers = String::new();
    for line in extra_headers {
        headers.push_str(line);
        headers.push_str("\r\n");
    }
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         {headers}\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n\
         {body}",
        body.len()
    )
}

/// A client whose body selects `llama-3.2-8b` (→ llama-remote) but who spoofs
/// `X-Model: granite-3.3-8b` (→ granite-local) must be routed by the body-derived
/// model, never by the raw client header.
#[test]
fn spoofed_model_header_does_not_override_body_derived_route() {
    let granite = start_backend_with_shutdown("granite-backend");
    let llama = start_backend_with_shutdown("llama-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        INFERENCE_EXAMPLE,
        proxy_port,
        HashMap::from([(GRANITE_ENDPOINT, granite.port()), (LLAMA_ENDPOINT, llama.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"llama-3.2-8b","messages":[]}"#;
    let raw = http_send(
        proxy.addr(),
        &chat_post_with_headers(body, &["X-Model: granite-3.3-8b"]),
    );

    assert_eq!(parse_status(&raw), 200, "request should be routed: {raw}");
    assert!(
        raw.contains("llama-backend"),
        "body-derived model must win over the spoofed X-Model header; got: {raw}"
    );
    assert!(
        !raw.contains("granite-backend"),
        "spoofed X-Model header must not steer routing: {raw}"
    );
}

/// Fail-closed: a well-formed request whose model matches no configured
/// candidate must be rejected deterministically (404) without contacting any
/// upstream. An unknown model must never fall through to a default backend.
#[test]
fn unknown_model_fails_closed_without_upstream_call() {
    // Capturing backends so we can assert neither upstream was contacted.
    let granite = start_stateful_backend(vec![(200, "granite-backend".to_owned())]);
    let llama = start_stateful_backend(vec![(200, "llama-backend".to_owned())]);
    let proxy_port = free_port();

    let config = load_example_config(
        INFERENCE_EXAMPLE,
        proxy_port,
        HashMap::from([(GRANITE_ENDPOINT, granite.port()), (LLAMA_ENDPOINT, llama.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4o-not-configured","messages":[]}"#;
    let raw = http_send(proxy.addr(), &chat_post_with_headers(body, &[]));

    assert_eq!(
        parse_status(&raw),
        404,
        "unknown model must fail closed with 404: {raw}"
    );
    assert!(
        granite.requests().is_empty(),
        "granite upstream must not be contacted for an unknown model: {:?}",
        granite.requests()
    );
    assert!(
        llama.requests().is_empty(),
        "llama upstream must not be contacted for an unknown model: {:?}",
        llama.requests()
    );
}

/// Control: with no spoofed header, the body model routes as expected.
#[test]
fn body_model_routes_to_matching_cluster() {
    let granite = start_backend_with_shutdown("granite-backend");
    let llama = start_backend_with_shutdown("llama-backend");
    let proxy_port = free_port();

    let config = load_example_config(
        INFERENCE_EXAMPLE,
        proxy_port,
        HashMap::from([(GRANITE_ENDPOINT, granite.port()), (LLAMA_ENDPOINT, llama.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"granite-3.3-8b","messages":[]}"#;
    let raw = http_send(proxy.addr(), &chat_post_with_headers(body, &[]));

    assert_eq!(parse_status(&raw), 200, "request should be routed: {raw}");
    assert!(
        raw.contains("granite-backend"),
        "body model must route to granite: {raw}"
    );
}
