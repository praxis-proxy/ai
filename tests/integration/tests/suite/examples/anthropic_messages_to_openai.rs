// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the Messages-to-Chat-Completions example config.
//!
//! `messages-to-openai.yaml` is the unauthenticated translation chain: no
//! gateway auth, no credential injection, and no `headers` filter. That makes it
//! the sharpest test of the translator's own credential boundary, because
//! nothing else in the chain removes a client header. Both SDK credential forms
//! must stop at the translator:
//!
//! - the legacy `x-api-key` header, which the translator already stripped; and
//! - the `Authorization: Bearer` header Anthropic SDK v2 sends, which used to reach the Chat Completions backend as if
//!   it were that backend's credential.
//!
//! Assertions read the headers the controlled backend actually received, so a
//! regression that re-forwards either form fails here rather than in a live
//! deployment.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, StatefulCapturingGuard, free_port, http_send, parse_body, parse_status,
    patch_yaml, start_proxy,
};
use serde_json::Value;

use super::load_example_config;

const CONFIG: &str = "anthropic/messages-to-openai.yaml";

/// The secret a caller presents in whichever credential header its SDK uses.
///
/// Distinct from any value the example config could inject, so finding it in a
/// captured request proves the client credential crossed the boundary.
const CLIENT_SECRET: &str = "client-anthropic-secret";

/// A canned Chat Completions response the fake backend returns; Praxis
/// translates it back into an Anthropic message on the way out.
const CHAT_COMPLETION_RESPONSE: &str = concat!(
    r#"{"id":"chatcmpl-a","object":"chat.completion","created":1700000000,"#,
    r#""model":"claude-opus-4-8","choices":[{"index":0,"message":{"role":"assistant","#,
    r#""content":"Hello from a Chat Completions backend."},"finish_reason":"stop"}],"#,
    r#""usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}"#,
);

/// Build the example config with the listener and backend ports patched.
fn translate_config(proxy_port: u16, backend_port: u16) -> Config {
    let path = praxis_test_utils::example_config_path(CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", backend_port)]));
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"))
}

/// Start a backend that records every request and answers with a canned
/// Chat Completion. The extra copies cover Pingora's health-check probe.
fn capturing_chat_backend() -> StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
    ])
    .start_with_shutdown()
}

/// Send one Anthropic Messages request carrying exactly the caller headers the
/// caller supplied. Header names are spelled the way each SDK sends them.
fn post_message(proxy: &praxis_test_utils::ProxyGuard, caller_headers: &str) -> String {
    let body = r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
    http_send(
        proxy.addr(),
        &format!(
            "POST /v1/messages HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             {caller_headers}\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {body}",
            body.len()
        ),
    )
}

/// Lowercased header names captured from the forwarded request.
fn captured_header_names(request: &CapturedRequest) -> Vec<String> {
    request
        .headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(name, _value)| name.trim().to_ascii_lowercase())
        .collect()
}

/// The one POST the backend received, failing loudly when translation never
/// reached it.
fn forwarded_post(backend: &StatefulCapturingGuard) -> CapturedRequest {
    backend
        .requests()
        .into_iter()
        .find(|request| request.method == "POST")
        .expect("the translated request must reach the Chat Completions backend")
}

#[test]
fn openai_config_parses() {
    let config = load_example_config(CONFIG, 29960, HashMap::from([("127.0.0.1:8000", 29961_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(
        config.filter_chains[0].name, "anthropic-transform",
        "chain name should be anthropic-transform"
    );
}

#[test]
fn legacy_x_api_key_never_reaches_the_chat_backend() {
    let backend = capturing_chat_backend();
    let proxy_port = free_port();
    let config = translate_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let raw = post_message(&proxy, &format!("x-api-key: {CLIENT_SECRET}\r\n"));

    assert_eq!(parse_status(&raw), 200, "translated request should return 200: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["type"], "message",
        "the Chat Completion must still translate back to an Anthropic message"
    );

    let forwarded = forwarded_post(&backend);
    assert_eq!(
        forwarded.uri, "/v1/chat/completions",
        "the Anthropic path must be rewritten for the backend"
    );
    assert!(
        !forwarded.headers.contains(CLIENT_SECRET),
        "the legacy x-api-key credential must not reach the backend: {}",
        forwarded.headers
    );
    let names = captured_header_names(&forwarded);
    assert!(
        !names.contains(&"x-api-key".to_owned()),
        "the backend must not receive an x-api-key header at all: {names:?}"
    );

    drop(proxy);
}

#[test]
fn sdk_v2_bearer_credential_never_reaches_the_chat_backend() {
    let backend = capturing_chat_backend();
    let proxy_port = free_port();
    let config = translate_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    // Anthropic SDK v2 sends its API key as a bearer token instead of x-api-key.
    let raw = post_message(&proxy, &format!("Authorization: Bearer {CLIENT_SECRET}\r\n"));

    assert_eq!(parse_status(&raw), 200, "translated request should return 200: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["type"], "message",
        "scrubbing the caller credential must not disturb response translation"
    );

    let forwarded = forwarded_post(&backend);
    assert!(
        !forwarded.headers.contains(CLIENT_SECRET),
        "the SDK v2 bearer credential must not reach the backend: {}",
        forwarded.headers
    );
    let names = captured_header_names(&forwarded);
    assert!(
        !names.contains(&"authorization".to_owned()),
        "the backend must not receive an authorization header at all: {names:?}"
    );

    drop(proxy);
}

#[test]
fn both_sdk_credential_forms_stop_at_the_translator_on_one_chain() {
    // A single chain must scrub whichever form arrives, including a caller that
    // sends both at once (some SDK wrappers add one while the key sets the other).
    let backend = capturing_chat_backend();
    let proxy_port = free_port();
    let config = translate_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let raw = post_message(
        &proxy,
        &format!("x-api-key: {CLIENT_SECRET}\r\nAuthorization: Bearer {CLIENT_SECRET}\r\n"),
    );

    assert_eq!(parse_status(&raw), 200, "translated request should return 200: {raw}");

    let forwarded = forwarded_post(&backend);
    assert!(
        !forwarded.headers.contains(CLIENT_SECRET),
        "no client credential form may reach the backend: {}",
        forwarded.headers
    );
    let names = captured_header_names(&forwarded);
    for forbidden in ["authorization", "x-api-key"] {
        assert!(
            !names.contains(&forbidden.to_owned()),
            "the backend must not receive a {forbidden} header: {names:?}"
        );
    }

    drop(proxy);
}
