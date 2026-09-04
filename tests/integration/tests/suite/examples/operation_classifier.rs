// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the operation-classifier example config.
//!
//! `x-praxis-ai-*` uses a reserved prefix, so the protocol layer strips those
//! headers at ingress and before forwarding. These tests assert the boundary
//! properties that are observable end to end: classified traffic is forwarded
//! unharmed, the classifier's own headers stay proxy-internal, and a client
//! supplying a reserved header is rejected at ingress. Publication of the
//! match, metadata, results, and headers is covered by the filter's unit
//! tests.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_header_echo_backend,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Start the example config against a header-echoing backend.
fn start() -> (praxis_test_utils::BackendGuard, praxis_test_utils::ProxyGuard) {
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/operation-classifier.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend.port())]),
    );
    let proxy = start_proxy(&config);
    (backend, proxy)
}

/// Echoed request headers, lowercased for case-insensitive assertions.
fn echoed(raw: &str) -> String {
    parse_body(raw).to_lowercase()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn classified_responses_request_is_forwarded_unharmed() {
    let (_backend, proxy) = start();

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"hi"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "createResponse should be forwarded");
    assert!(
        echoed(&raw).contains("content-type: application/json"),
        "the original request should reach upstream intact"
    );
}

#[test]
fn classified_conversations_request_is_forwarded_unharmed() {
    let (_backend, proxy) = start();

    let raw = http_send(
        proxy.addr(),
        "GET /v1/conversations/conv_123 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "getConversation should be forwarded");
}

#[test]
fn unclassified_request_is_still_forwarded() {
    let (_backend, proxy) = start();

    for request in [
        "PUT /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        "GET /v1/unknown HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    ] {
        let raw = http_send(proxy.addr(), request);
        assert_eq!(
            parse_status(&raw),
            200,
            "an unmatched request is a routing policy decision, not a rejection"
        );
    }
}

#[test]
fn classifier_headers_never_reach_upstream() {
    let (_backend, proxy) = start();

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"hi"}"#),
    );

    let body = echoed(&raw);
    assert!(
        !body.contains("x-praxis-ai-family"),
        "reserved routing headers are proxy-internal, got: {body}"
    );
    assert!(
        !body.contains("x-praxis-ai-operation"),
        "reserved routing headers are proxy-internal, got: {body}"
    );
}

#[test]
fn client_supplied_classifier_headers_are_rejected_at_ingress() {
    let (_backend, proxy) = start();

    let raw = http_send(
        proxy.addr(),
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n\
         x-praxis-ai-family: files\r\n\
         x-praxis-ai-operation: createFile\r\n\
         Content-Type: application/json\r\nContent-Length: 32\r\nConnection: close\r\n\r\n\
         {\"model\":\"gpt-4.1\",\"input\":\"hi\"}",
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "reserved headers are proxy-owned, so a client supplying one is rejected"
    );
    let body = echoed(&raw);
    assert!(
        !body.contains("createfile"),
        "a forged operation must not cross the proxy, got: {body}"
    );
}

#[test]
fn client_supplied_headers_are_rejected_on_an_unclassified_path_too() {
    let (_backend, proxy) = start();

    let raw = http_send(
        proxy.addr(),
        "GET /v1/unknown HTTP/1.1\r\nHost: localhost\r\n\
         x-praxis-ai-family: responses\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "the reserved-header boundary does not depend on classification"
    );
    assert!(
        !echoed(&raw).contains("x-praxis-ai-family"),
        "a forged family must not cross the proxy on an unclassified path"
    );
}
