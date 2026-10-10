// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the operation-classifier example config.
//!
//! The example branches on the classifier's published filter results, so the
//! selected backend is the observable proof that a request was classified:
//! Conversations and Chat Completions each reach their own backend, and
//! everything else falls through to the default one. These tests assert that
//! routing, plus the boundary properties around the proxy-owned headers.
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

/// Body returned by the Chat Completions backend, proving it was selected.
const CHAT_MARKER: &str = "{\"selected\":\"chat-backend\"}";

/// Body returned by the Files backend, proving it was selected.
const FILES_MARKER: &str = "{\"selected\":\"files-backend\"}";

/// Body returned by the Vector Stores backend, proving it was selected.
const VECTOR_STORES_MARKER: &str = "{\"selected\":\"vector-stores-backend\"}";

/// Body returned by the Anthropic Messages backend, proving it was selected.
const ANTHROPIC_MARKER: &str = "{\"selected\":\"anthropic-backend\"}";

/// Started example: a header-echoing default backend and marker-returning
/// family backends.
struct Harness {
    /// Default backend, which echoes the request headers it received.
    _responses: praxis_test_utils::BackendGuard,
    /// Conversations backend, which returns [`CONVERSATIONS_MARKER`].
    _conversations: praxis_test_utils::CapturingBackendGuard,
    /// Chat Completions backend, which returns [`CHAT_MARKER`].
    _chat: praxis_test_utils::CapturingBackendGuard,
    /// Files backend, which returns [`FILES_MARKER`].
    _files: praxis_test_utils::CapturingBackendGuard,
    /// Vector Stores backend, which returns [`VECTOR_STORES_MARKER`].
    _vector_stores: praxis_test_utils::CapturingBackendGuard,
    /// Anthropic Messages backend, which returns [`ANTHROPIC_MARKER`].
    _anthropic: praxis_test_utils::CapturingBackendGuard,
    /// The running proxy.
    proxy: praxis_test_utils::ProxyGuard,
}

/// Start the example config against all backends.
fn start() -> Harness {
    let responses = start_header_echo_backend();
    let conversations = start_capturing_backend(CONVERSATIONS_MARKER);
    let chat = start_capturing_backend(CHAT_MARKER);
    let files = start_capturing_backend(FILES_MARKER);
    let vector_stores = start_capturing_backend(VECTOR_STORES_MARKER);
    let anthropic = start_capturing_backend(ANTHROPIC_MARKER);
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/operation-classifier.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", responses.port()),
            ("127.0.0.1:3002", conversations.port()),
            ("127.0.0.1:3003", chat.port()),
            ("127.0.0.1:3004", files.port()),
            ("127.0.0.1:3005", vector_stores.port()),
            ("127.0.0.1:3006", anthropic.port()),
        ]),
    );
    let proxy = start_proxy(&config);
    Harness {
        _responses: responses,
        _conversations: conversations,
        _chat: chat,
        _files: files,
        _vector_stores: vector_stores,
        _anthropic: anthropic,
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

/// A multipart upload reaches the Files backend. The body is
/// `multipart/form-data`, so only head classification can route it.
#[test]
fn a_classified_files_operation_selects_the_files_backend() {
    let h = start();

    let body = "--X\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n--X--\r\n";
    let raw = http_send(
        h.proxy.addr(),
        &format!(
            "POST /v1/files HTTP/1.1\r\nHost: localhost\r\n\
             Content-Type: multipart/form-data; boundary=X\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    );

    assert_eq!(parse_status(&raw), 200, "createFile should be forwarded");
    assert_eq!(
        parse_body(&raw),
        FILES_MARKER,
        "openai_files must branch to the Files backend"
    );
}

/// A nested Vector Stores path reaches its own backend. `{file_id}` here names
/// a vector-store file rather than a Files upload, so the two families must
/// not collide.
#[test]
fn a_classified_vector_stores_operation_selects_the_vector_stores_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "GET /v1/vector_stores/vs_1/files/file_1 HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "getVectorStoreFile should be forwarded");
    assert_eq!(
        parse_body(&raw),
        VECTOR_STORES_MARKER,
        "openai_vector_stores must branch to the Vector Stores backend"
    );
}

/// An unknown subresource under a family path does not enter that family's
/// route. A path prefix would have admitted it.
#[test]
fn an_unknown_family_subresource_falls_through() {
    let h = start();

    for path in [
        "/v1/vector_stores/vs_1/unknown",
        "/v1/files/file_1/unknown",
        "/v1/vector_stores_extra",
    ] {
        let raw = http_send(
            h.proxy.addr(),
            &format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
        );

        let body = parse_body(&raw);
        assert_ne!(body, FILES_MARKER, "{path} must not reach the Files backend");
        assert_ne!(
            body, VECTOR_STORES_MARKER,
            "{path} must not reach the Vector Stores backend"
        );
    }
}

/// An unsupported method on a family path does not enter that family's route.
#[test]
fn an_unsupported_family_method_falls_through() {
    let h = start();

    for (method, path) in [("PUT", "/v1/files"), ("DELETE", "/v1/vector_stores")] {
        let raw = http_send(
            h.proxy.addr(),
            &format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
        );

        let body = parse_body(&raw);
        assert_ne!(body, FILES_MARKER, "{method} {path} must not reach the Files backend");
        assert_ne!(
            body, VECTOR_STORES_MARKER,
            "{method} {path} must not reach the Vector Stores backend"
        );
    }
}

/// A Chat Completions client keeps OpenAI-shaped errors when the proxy itself
/// fails, without any Responses filter in the chain.
///
/// This chain runs only the operation classifier — no `openai_responses_request`
/// — so it is the case that proves the error formatter follows head
/// classification rather than body classification. Before that, a chain without
/// a Responses filter returned RFC 9457 problem details to an OpenAI client.
#[test]
fn a_chat_completions_failure_keeps_the_openai_error_shape() {
    // Point the chat backend at a port nothing is listening on, so the failure
    // is generated by the proxy rather than returned by a backend.
    let responses = start_header_echo_backend();
    let conversations = start_capturing_backend(CONVERSATIONS_MARKER);
    let dead_port = free_port();
    let proxy_port = free_port();
    let config = load_example_config(
        "openai/operation-classifier.yaml",
        proxy_port,
        HashMap::from([
            ("127.0.0.1:3001", responses.port()),
            ("127.0.0.1:3002", conversations.port()),
            ("127.0.0.1:3003", dead_port),
        ]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    let status = parse_status(&raw);
    assert!(
        (500..600).contains(&status),
        "an unreachable backend should fail the request, got {status}"
    );

    let body = parse_body(&raw);
    let parsed: serde_json::Value =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("error body should be JSON: {e}\n{body}"));

    // Assert the shape an OpenAI client actually parses, not merely that some
    // `error` key exists: `{"error": null}` would satisfy a presence check.
    let error = parsed
        .get("error")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("error must be an object, got: {body}"));
    assert!(
        error.get("message").and_then(serde_json::Value::as_str).is_some(),
        "error.message must be a string, got: {body}"
    );
    assert_eq!(
        error.get("type").and_then(serde_json::Value::as_str),
        Some("server_error"),
        "error.type should classify an upstream failure, got: {body}"
    );
    assert_eq!(
        error.get("code").and_then(serde_json::Value::as_str),
        Some("upstream_connect_refused"),
        "error.code should name the upstream failure, got: {body}"
    );
    assert!(
        !body.contains("problem+json") && !body.contains("about:blank"),
        "must not fall back to RFC 9457 problem details, got: {body}"
    );
}

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
fn a_classified_chat_completions_operation_selects_the_chat_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        &json_post(
            "/v1/chat/completions",
            r#"{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "createChatCompletion should be forwarded");
    assert_eq!(
        parse_body(&raw),
        CHAT_MARKER,
        "openai_chat_completions must branch to the Chat Completions backend"
    );
}

#[test]
fn chat_completions_identity_does_not_depend_on_the_body() {
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
        assert_eq!(parse_status(&raw), 200, "{name} should be forwarded");
        assert_eq!(
            parse_body(&raw),
            CHAT_MARKER,
            "{name} must still select the Chat Completions backend"
        );
    }
}

#[test]
fn a_classified_responses_operation_selects_the_default_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"hi"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "createResponse should be forwarded");
    let body = parse_body(&raw);
    assert_ne!(
        body, CONVERSATIONS_MARKER,
        "openai_responses must not branch to the Conversations backend"
    );
    assert_ne!(
        body, CHAT_MARKER,
        "openai_responses must not branch to the Chat Completions backend"
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
fn unsupported_chat_completions_methods_do_not_select_the_chat_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        "PUT /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );

    assert_eq!(parse_status(&raw), 200, "an unmatched request is still forwarded");
    assert_ne!(
        parse_body(&raw),
        CHAT_MARKER,
        "PUT /v1/chat/completions must not receive Chat Completions routing"
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

/// `POST /v1/messages` reaches the Anthropic Messages backend. The classifier
/// recognizes it from the head alone, so this proves the branch is wired.
#[test]
fn a_classified_anthropic_messages_operation_selects_the_anthropic_backend() {
    let h = start();

    let raw = http_send(
        h.proxy.addr(),
        &json_post(
            "/v1/messages",
            r#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "messages_post should be forwarded");
    assert_eq!(
        parse_body(&raw),
        ANTHROPIC_MARKER,
        "anthropic_messages must branch to the Anthropic backend"
    );
}

/// The body cannot change which operation `POST /v1/messages` resolves to.
/// A malformed, empty, or Chat Completions-shaped body on the Anthropic path
/// still selects the Anthropic backend.
#[test]
fn anthropic_messages_identity_does_not_depend_on_the_body() {
    let h = start();

    let chat_body = r#"{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}"#;
    let chat_on_anthropic = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{chat_body}",
        chat_body.len()
    );

    let malformed = "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\n\
         Content-Type: application/json\r\nContent-Length: 8\r\nConnection: close\r\n\r\n\
         not json";
    let empty = "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\n\
         Content-Length: 0\r\nConnection: close\r\n\r\n";

    for (name, request) in [
        ("malformed JSON", malformed),
        ("empty body", empty),
        ("chat completions body on an anthropic path", chat_on_anthropic.as_str()),
    ] {
        let raw = http_send(h.proxy.addr(), request);
        assert_eq!(parse_status(&raw), 200, "{name} should be forwarded");
        assert_eq!(
            parse_body(&raw),
            ANTHROPIC_MARKER,
            "{name} must still select the Anthropic backend"
        );
    }
}
