// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the Vertex AI dialect translation example configuration.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, StatefulCapturingBackend, example_config_path, free_port, http_send, parse_body, parse_status, patch_yaml,
    start_capturing_backend, start_header_echo_backend, start_uri_echo_backend,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

fn anthropic_request(body: &str) -> String {
    format!(
        concat!(
            "POST /v1/messages HTTP/1.1\r\n",
            "Host: localhost\r\n",
            "Content-Type: application/json\r\n",
            "x-api-key: client-key\r\n",
            "Accept-Encoding: gzip, br\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n\r\n{}",
        ),
        body.len(),
        body
    )
}

fn patched_example(proxy_port: u16, backend_port: u16) -> Config {
    super::load_example_config(
        "vertex/messages-to-anthropic.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port), ("127.0.0.1:3001", backend_port)]),
    )
}

#[test]
fn vertex_anthropic_config_parses() {
    let config = super::load_example_config(
        "vertex/messages-to-anthropic.yaml",
        29930,
        HashMap::from([("127.0.0.1:3000", 29931_u16), ("127.0.0.1:3001", 29932_u16)]),
    );

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn messages_request_is_stripped_upstream_and_response_model_restored() {
    // Vertex rejects a body model field, and answers with its snapshot
    // model id; the filter must strip one and restore the other.
    let vertex_response = r#"{"type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":5,"output_tokens":2}}"#;
    let backend = start_capturing_backend(vertex_response);
    let proxy_port = free_port();
    let proxy = praxis_test_utils::start_proxy(&patched_example(proxy_port, backend.port()));

    let raw = http_send(
        proxy.addr(),
        &anthropic_request(r#"{"model":"public-claude","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#),
    );
    assert_eq!(parse_status(&raw), 200);

    let forwarded = backend.body();
    assert!(
        !forwarded.contains(r#""model""#),
        "Vertex rejects body model; it must be stripped, got: {forwarded}"
    );
    assert!(
        forwarded.contains(r#""anthropic_version":"vertex-2023-10-16""#),
        "anthropic_version must be injected, got: {forwarded}"
    );
    assert!(
        forwarded.contains(r#""max_tokens":8"#),
        "other body fields must pass through, got: {forwarded}"
    );

    let to_client = parse_body(&raw);
    assert!(
        to_client.contains(r#""model":"public-claude""#),
        "the user-facing model id must come back, got: {to_client}"
    );
    assert!(
        !to_client.contains("20250929"),
        "the Vertex snapshot id must not leak to the client, got: {to_client}"
    );
    assert!(
        to_client.contains(r#""input_tokens":5"#),
        "metering payload must survive the rewrite, got: {to_client}"
    );
}

#[test]
fn stream_true_sends_stream_rawpredict_upstream() {
    let backend = start_uri_echo_backend();
    let proxy_port = free_port();
    let proxy = praxis_test_utils::start_proxy(&patched_example(proxy_port, backend.port()));

    let request = anthropic_request(
        r#"{"model":"public-claude","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
    );
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200);

    let echoed = parse_body(&raw);
    assert!(
        echoed.starts_with("/v1/projects/my-gcp-project/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929:streamRawPredict"),
        "model must move into the URL with the stream verb, got: {echoed}"
    );
}

#[test]
fn handled_request_reaches_vertex_without_client_key_or_encoding() {
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let proxy = praxis_test_utils::start_proxy(&patched_example(proxy_port, backend.port()));

    let raw = http_send(
        proxy.addr(),
        &anthropic_request(r#"{"model":"public-claude","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#),
    );
    assert_eq!(parse_status(&raw), 200, "header echo should succeed: {raw}");

    let upstream = parse_body(&raw).to_ascii_lowercase();
    assert!(
        !upstream.contains("x-api-key"),
        "the client's Anthropic key must not reach Vertex, got: {upstream}"
    );
    assert!(
        !upstream.contains("accept-encoding"),
        "Vertex must be asked for an uncompressed response, got: {upstream}"
    );
    assert!(
        upstream.contains("authorization: bearer ya29.example-static-token"),
        "Vertex auth must come from the credential filter, got: {upstream}"
    );
}

#[test]
fn count_tokens_preserves_the_count_response_without_injecting_a_model() {
    let backend = start_capturing_backend(r#"{"input_tokens":5}"#);
    let proxy = praxis_test_utils::start_proxy(&patched_example(free_port(), backend.port()));
    let request = anthropic_request(r#"{"model":"public-claude","messages":[{"role":"user","content":"hi"}]}"#)
        .replace("POST /v1/messages HTTP", "POST /v1/messages/count_tokens HTTP");
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200, "token counting must reach the provider");
    let forwarded: serde_json::Value = serde_json::from_str(&backend.body()).unwrap();
    assert_eq!(
        forwarded["model"], "claude-sonnet-4-5@20250929",
        "counting retains the publisher model"
    );
    assert!(
        forwarded.get("anthropic_version").is_none(),
        "counting does not accept anthropic_version"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(
        response["input_tokens"], 5,
        "the count response must survive translation"
    );
    assert!(
        response.get("model").is_none(),
        "count responses must not acquire a model"
    );
}

#[test]
fn mapped_targets_select_exact_publisher_snapshots() {
    let backend = start_uri_echo_backend();
    let proxy = praxis_test_utils::start_proxy(&patched_example(free_port(), backend.port()));
    for (model, target) in [
        ("public-claude", "claude-sonnet-4-5@20250929:rawPredict"),
        ("public-claude-snapshot", "claude-sonnet-4-5@20250929:rawPredict"),
    ] {
        let body = format!(r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#);
        let raw = http_send(proxy.addr(), &anthropic_request(&body));
        assert_eq!(parse_status(&raw), 200, "{model} must reach its configured publisher");
        assert!(
            parse_body(&raw).ends_with(target),
            "the exact mapped target must reach Vertex: {raw}"
        );
    }
}

#[test]
fn count_tokens_keeps_the_mapped_snapshot_in_the_body() {
    let backend = start_capturing_backend(r#"{"input_tokens":5}"#);
    let proxy = praxis_test_utils::start_proxy(&patched_example(free_port(), backend.port()));
    let request = anthropic_request(r#"{"model":"public-claude-snapshot","messages":[]}"#)
        .replace("POST /v1/messages HTTP", "POST /v1/messages/count_tokens HTTP");
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200);
    let body: serde_json::Value = serde_json::from_str(&backend.body()).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4-5@20250929");
    assert!(body.get("anthropic_version").is_none());
}

#[test]
fn unknown_models_and_malformed_bodies_reach_fallback_unchanged() {
    let vertex =
        StatefulCapturingBackend::new(vec![(200, "unexpected Vertex response".to_owned())]).start_with_shutdown();
    let fallback = start_capturing_backend("fallback response");
    let config = super::load_example_config(
        "vertex/messages-to-anthropic.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3000", vertex.port()), ("127.0.0.1:3001", fallback.port())]),
    );
    let proxy = praxis_test_utils::start_proxy(&config);
    for body in [r#"{"model":"vertex/claude-sonnet-4-5","messages":[]}"#, "not json"] {
        let raw = http_send(proxy.addr(), &anthropic_request(body));
        assert_eq!(parse_status(&raw), 200);
        assert_eq!(parse_body(&raw), "fallback response");
        assert_eq!(fallback.body(), body);
        assert!(vertex.requests().is_empty(), "fallback traffic must not reach Vertex");
    }
}

#[test]
fn split_sse_restores_the_mapped_public_alias() {
    let first = "id: split-event\nretry: 1000\nevent: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-son";
    let second =
        "net-4-5-20250929\",\"id\":\"msg_split\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let backend = Backend::chunked(vec![first.to_owned(), second.to_owned()])
        .header("Content-Type", "text/event-stream")
        .stall_after_first_chunk(std::time::Duration::from_millis(30))
        .start_with_shutdown();
    let proxy = praxis_test_utils::start_proxy(&patched_example(free_port(), backend.port()));
    let raw = http_send(
        proxy.addr(),
        &anthropic_request(r#"{"model":"public-claude-snapshot","stream":true,"max_tokens":8,"messages":[]}"#),
    );
    assert_eq!(parse_status(&raw), 200);
    let body = parse_body(&raw);
    assert!(
        body.contains(r#""model":"public-claude-snapshot""#),
        "alias must survive a split frame: {body}"
    );
    assert!(body.contains("id: split-event") && body.contains("retry: 1000"));
    assert!(body.contains("event: message_stop"));
    assert!(!body.contains("20250929"));
}

/// Preserve repeated beta flags per [RFC 9110 Section 5.3].
///
/// [RFC 9110 Section 5.3]: https://datatracker.ietf.org/doc/html/rfc9110#section-5.3
#[test]
fn beta_allowlist_keeps_allowed_flags_from_later_header_lines() {
    let backend = start_header_echo_backend();
    let source = std::fs::read_to_string(example_config_path("vertex/messages-to-anthropic.yaml"))
        .expect("the Vertex example must load")
        .replace("# beta_allowlist:", "beta_allowlist:");
    let source = patch_yaml(
        &source,
        free_port(),
        &HashMap::from([("127.0.0.1:3000", backend.port())]),
    );
    let config = Config::from_yaml(&source).expect("the example with its optional allowlist must parse");
    let proxy = praxis_test_utils::start_proxy(&config);
    let request = anthropic_request(r#"{"model":"public-claude","max_tokens":8,"messages":[]}"#).replace(
        "x-api-key: client-key\r\n",
        "anthropic-beta: unknown\r\nanthropic-beta: context-1m-2025-08-07\r\n",
    );
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(
        parse_status(&raw),
        200,
        "a repeated beta header must reach the configured provider"
    );
    let upstream = parse_body(&raw).to_ascii_lowercase();
    assert!(
        upstream.contains("anthropic-beta: context-1m-2025-08-07"),
        "allowed flags from later lines must reach Vertex: {upstream}"
    );
    assert!(
        !upstream.contains("unknown"),
        "disallowed flags from any header instance must be stripped: {upstream}"
    );
}
