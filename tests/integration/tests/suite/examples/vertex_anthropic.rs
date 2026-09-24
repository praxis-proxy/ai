// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the Vertex AI dialect translation example configuration.

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, parse_body, parse_status, start_capturing_backend, start_uri_echo_backend,
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
            "Content-Length: {}\r\n",
            "Connection: close\r\n\r\n{}",
        ),
        body.len(),
        body
    )
}

fn patched_example(proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    super::load_example_config(
        "vertex-anthropic.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3000", backend_port)]),
    )
}

#[test]
fn vertex_anthropic_config_parses() {
    let config = super::load_example_config(
        "vertex-anthropic.yaml",
        29930,
        HashMap::from([("127.0.0.1:3000", 29931_u16)]),
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
        &anthropic_request(
            r#"{"model":"vertex/claude-sonnet-4-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
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
        to_client.contains(r#""model":"vertex/claude-sonnet-4-5""#),
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
        r#"{"model":"vertex/claude-sonnet-4-5","stream":true,"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
    );
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200);

    let echoed = parse_body(&raw).to_owned();
    assert!(
        echoed.starts_with("/v1/projects/my-gcp-project/locations/global/publishers/anthropic/models/claude-sonnet-4-5:streamRawPredict"),
        "model must move into the URL with the stream verb, got: {echoed}"
    );
}
