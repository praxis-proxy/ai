// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the operation-classifier example config.
//!
//! The example branches on the classifier's published filter results, so the
//! selected backend is the observable proof that a request was classified:
//! Conversations traffic reaches its own backend, everything else falls
//! through to the default one. These tests assert that routing, plus the
//! boundary properties around the proxy-owned headers.
//!
//! `x-praxis-ai-*` uses a reserved prefix, so the protocol layer strips those
//! headers at ingress and before forwarding, and rejects a client that supplies
//! one. Publication of the typed match, metadata, and results is covered by the
//! filter's unit tests.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_capturing_backend,
    start_header_echo_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Body returned by the Conversations backend, proving it was selected.
const CONVERSATIONS_MARKER: &str = "{\"selected\":\"conversations-backend\"}";

/// Started example: a header-echoing default backend and a marker-returning
/// Conversations backend.
struct Harness {
    /// Default backend, which echoes the request headers it received.
    _responses: praxis_test_utils::BackendGuard,
    /// Conversations backend, which returns [`CONVERSATIONS_MARKER`].
    _conversations: praxis_test_utils::CapturingBackendGuard,
    /// The running proxy.
    proxy: praxis_test_utils::ProxyGuard,
}

/// Start the example config against both backends.
fn start() -> Harness {
    let responses = start_header_echo_backend();
    let conversations = start_capturing_backend(CONVERSATIONS_MARKER);
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/operation-classifier.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", responses.port()),
            ("127.0.0.1:3002", conversations.port()),
        ]),
    );
    let proxy = start_proxy(&config);
    Harness {
        _responses: responses,
        _conversations: conversations,
        proxy,
    }
}

/// Echoed request headers, lowercased for case-insensitive assertions.
fn echoed(raw: &str) -> String {
    parse_body(raw).to_lowercase()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn a_classified_conversations_operation_selects_the_conversations_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "GET /v1/conversations/conv_123 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "getConversation should be forwarded");
    assert_eq!(
        parse_body(&raw),
        CONVERSATIONS_MARKER,
        "openai_conversations must branch to the Conversations backend"
    );
}

#[test]
fn a_classified_responses_operation_selects_the_default_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"hi"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "createResponse should be forwarded");
    assert_ne!(
        parse_body(&raw),
        CONVERSATIONS_MARKER,
        "openai_responses must not branch to the Conversations backend"
    );
    assert!(
        echoed(&raw).contains("content-type: application/json"),
        "the original request should reach upstream intact"
    );
}

#[test]
fn the_branch_follows_the_classification_not_the_path_prefix() {
    let h = start();

    // Same /v1/conversations prefix, but PUT classifies as nothing, so it must
    // fall through to the default backend rather than follow the prefix.
    let raw = http_send(
        h.proxy.addr(),
        "PUT /v1/conversations/conv_123 HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "an unmatched request is still forwarded");
    assert_ne!(
        parse_body(&raw),
        CONVERSATIONS_MARKER,
        "an unclassified request must not reach the Conversations backend on path prefix alone"
    );
}

#[test]
fn unclassified_requests_are_still_forwarded() {
    let h = start();

    for request in [
        "PUT /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        "GET /v1/unknown HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    ] {
        let raw = http_send(h.proxy.addr(), request);
        assert_eq!(
            parse_status(&raw),
            200,
            "an unmatched request is a routing policy decision, not a rejection"
        );
    }
}

#[test]
fn classifier_headers_never_reach_upstream() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"hi"}"#),
    );

    let body = echoed(&raw);
    assert!(
        !body.contains("x-praxis-ai-application-protocol"),
        "reserved routing headers are proxy-internal, got: {body}"
    );
    assert!(
        !body.contains("x-praxis-ai-operation"),
        "reserved routing headers are proxy-internal, got: {body}"
    );
}

#[test]
fn client_supplied_classifier_headers_are_rejected_at_ingress() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n\
         x-praxis-ai-application-protocol: openai_files\r\n\
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
fn a_forged_protocol_header_cannot_steer_the_branch() {
    let h = start();

    // Claim openai_conversations on a Responses path. The header is rejected at
    // ingress, so it can never reach the branch condition.
    let raw = http_send(
        h.proxy.addr(),
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\n\
         x-praxis-ai-application-protocol: openai_conversations\r\n\
         Content-Type: application/json\r\nContent-Length: 32\r\nConnection: close\r\n\r\n\
         {\"model\":\"gpt-4.1\",\"input\":\"hi\"}",
    );

    assert_eq!(parse_status(&raw), 400, "a client-supplied reserved header is rejected");
    assert_ne!(
        parse_body(&raw),
        CONVERSATIONS_MARKER,
        "a forged protocol must not select the Conversations backend"
    );
}

#[test]
fn client_supplied_headers_are_rejected_on_an_unclassified_path_too() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "GET /v1/unknown HTTP/1.1\r\nHost: localhost\r\n\
         x-praxis-ai-application-protocol: openai_responses\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "the reserved-header boundary does not depend on classification"
    );
    assert!(
        !echoed(&raw).contains("x-praxis-ai-application-protocol"),
        "a forged protocol must not cross the proxy on an unclassified path"
    );
}
