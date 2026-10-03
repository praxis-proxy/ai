// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the transformed-vLLM Anthropic Messages example config.
//!
//! `messages-to-openai-vllm.yaml` TRANSLATES native Anthropic Messages traffic
//! into OpenAI Chat Completions for a vLLM backend that serves only
//! `/v1/chat/completions`, with the same three-boundary credential isolation as
//! the native passthrough config. These tests assert:
//!
//! 1. The chain contains the Anthropic->Chat-Completions translation filters and the `path_rewrite` mapping — the
//!    opposite of the native passthrough chain.
//! 2. `POST /v1/messages` is rewritten to `/v1/chat/completions`; the request body is translated (Anthropic `system`
//!    hoisted into a Chat Completions system message) and the OpenAI response is translated back into an Anthropic
//!    message.
//! 3. Client credentials are stripped and the backend's own Bearer token is injected.
//! 4. Native token counting remains untouched when the backend also serves that endpoint.
//!
//! Like the native tests, `credential_injection` and `basic_auth` resolve their
//! secrets at pipeline-build time. `std::env::set_var` is `unsafe` (and
//! `unsafe_code` is denied workspace-wide), so `VLLM_API_KEY` is repointed to
//! `CARGO_PKG_NAME` — always set by Cargo for a test binary — and the gateway
//! password is inlined instead of mutating the environment.

use std::{collections::HashMap, time::Duration};

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, StatefulCapturingBackend, StatefulCapturingGuard, basic_auth_header, free_port, http_send,
    json_post_with_header, parse_body, parse_status, start_proxy,
};
use serde_json::{Value, json};

use super::load_example_config;

const CONFIG: &str = "anthropic/messages-to-openai-vllm.yaml";

/// The gateway `basic_auth` username the example config configures.
const GATEWAY_USER: &str = "gateway";

/// A canned OpenAI Chat Completions response the fake backend returns; Praxis
/// translates it back into an Anthropic message on the way out.
const CHAT_COMPLETION_RESPONSE: &str = concat!(
    r#"{"id":"chatcmpl-vllm","object":"chat.completion","created":1700000000,"#,
    r#""model":"claude-opus-4-8","choices":[{"index":0,"message":{"role":"assistant","#,
    r#""content":"Hello from a vLLM Chat Completions backend."},"finish_reason":"stop"}],"#,
    r#""usage":{"prompt_tokens":11,"completion_tokens":9,"total_tokens":20}}"#,
);

/// The gateway password these tests inject in place of `GATEWAY_AUTH_PASSWORD`.
///
/// Drawn once per test process from the OS RNG rather than a source literal, so
/// it is a throwaway secret scoped to this run; the config and the caller read
/// the same value so they agree within a run.
fn gateway_password() -> &'static str {
    static PASSWORD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PASSWORD.get_or_init(|| {
        rand::random::<[u8; 16]>()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    })
}

/// Build the transformed-vLLM config with ports patched, `VLLM_API_KEY`
/// repointed to a Cargo-provided variable, and the gateway password inlined so
/// pipeline build succeeds without mutating the environment.
fn transform_vllm_config(proxy_port: u16, backend_port: u16) -> Config {
    let path = praxis_test_utils::example_config_path(CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", backend_port)]));
    let patched = patched.replace("env_var: VLLM_API_KEY", "env_var: CARGO_PKG_NAME");
    let patched = patched.replace(
        "env_var: GATEWAY_AUTH_PASSWORD",
        &format!("password: {}", gateway_password()),
    );
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"))
}

/// The `Authorization` header line a trusted caller presents to the gateway.
fn gateway_auth_line() -> String {
    format!("Authorization: {}", basic_auth_header(GATEWAY_USER, gateway_password()))
}

/// Start a capturing backend that returns the canned Chat Completions response.
///
/// The extra 200s cover Pingora's health-check probe without exhausting the
/// scripted responses before the real POST arrives.
fn start_chat_backend() -> StatefulCapturingGuard {
    StatefulCapturingBackend::new(vec![
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
        (200, CHAT_COMPLETION_RESPONSE.to_owned()),
    ])
    .start_with_shutdown()
}

// -----------------------------------------------------------------------------
// Chain shape
// -----------------------------------------------------------------------------

#[test]
fn transform_vllm_config_parses() {
    let config = load_example_config(CONFIG, 29940, HashMap::from([("127.0.0.1:8000", 29941_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(
        &*config.listeners[0].name, "anthropic-gateway",
        "listener name should be anthropic-gateway"
    );
    assert_eq!(config.filter_chains.len(), 1, "should have 1 filter chain");
    assert_eq!(
        config.filter_chains[0].name, "anthropic-transform-vllm",
        "chain name should be anthropic-transform-vllm"
    );
}

#[test]
fn transform_vllm_chain_translates_to_chat_completions() {
    let config = load_example_config(CONFIG, 29942, HashMap::from([("127.0.0.1:8000", 29943_u16)]));
    let chain = &config.filter_chains[0];
    let types: Vec<&str> = chain.filters.iter().map(|f| f.filter_type.as_str()).collect();

    assert_eq!(
        types,
        [
            "basic_auth",
            "anthropic_messages_format",
            "anthropic_validate",
            "anthropic_messages_to_chat_completions",
            "anthropic_messages_to_chat_completions_stream",
            "headers",
            "path_rewrite",
            "router",
            "credential_injection",
            "load_balancer",
        ],
        "transformed chain must translate to Chat Completions, in order"
    );

    // Translation and path rewriting are the whole point of this config: unlike
    // the native passthrough chain, these filters MUST be present.
    for required in [
        "anthropic_messages_to_chat_completions",
        "anthropic_messages_to_chat_completions_stream",
        "path_rewrite",
    ] {
        assert!(
            types.contains(&required),
            "transformed chain must contain the {required} filter: {types:?}"
        );
    }
}

#[test]
fn transform_vllm_validate_is_scoped_to_messages() {
    // `anthropic_validate` rejects bodyless requests, so it must be gated to
    // `/v1/messages` — otherwise a bodyless probe would be rejected with 400.
    let config = load_example_config(CONFIG, 29944, HashMap::from([("127.0.0.1:8000", 29945_u16)]));
    let validate = config.filter_chains[0]
        .filters
        .iter()
        .find(|f| f.filter_type == "anthropic_validate")
        .expect("chain should contain anthropic_validate");

    assert!(
        !validate.conditions.is_empty(),
        "anthropic_validate must be gated by a path condition, not run unconditionally"
    );
}

// -----------------------------------------------------------------------------
// Translation (path, request body, response body)
// -----------------------------------------------------------------------------

#[test]
fn transform_vllm_rewrites_message_path_and_translates_response() {
    let backend = start_chat_backend();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "translated request should return 200");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["type"], "message",
        "the OpenAI Chat Completion must be translated back to an Anthropic message"
    );
    assert_eq!(
        response["content"][0]["text"], "Hello from a vLLM Chat Completions backend.",
        "the translated Anthropic message must carry the backend's assistant text"
    );

    // The backend must have seen the OpenAI endpoint, not the Anthropic one.
    let requests = backend.requests();
    let forwarded = requests
        .iter()
        .find(|r| r.method == "POST")
        .expect("backend should receive a POST request");
    assert_eq!(
        forwarded.uri, "/v1/chat/completions",
        "POST /v1/messages must be rewritten to /v1/chat/completions, got: {}",
        forwarded.uri
    );

    drop(proxy);
}

#[test]
fn transform_vllm_preserves_native_count_tokens_when_backend_supports_it() {
    let count_response = r#"{"input_tokens":4}"#;
    let backend = StatefulCapturingBackend::new(vec![
        (200, count_response.to_owned()),
        (200, count_response.to_owned()),
        (200, count_response.to_owned()),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "Count these tokens"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages/count_tokens", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "native count-tokens response should pass through"
    );
    assert_eq!(parse_body(&raw), count_response);
    let requests = backend.requests();
    let forwarded = requests
        .iter()
        .find(|r| r.method == "POST")
        .expect("backend should receive a POST request");
    assert_eq!(forwarded.uri, "/v1/messages/count_tokens");
    let body: Value = serde_json::from_str(&forwarded.body).expect("forwarded body should be JSON");
    assert_eq!(body, request, "count-tokens body must not be translated");

    drop(proxy);
}

#[test]
fn transform_vllm_translates_request_body_to_chat_completions() {
    let backend = start_chat_backend();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    // `system` and `max_tokens` are exactly the Anthropic fields translation
    // rewrites: `system` is hoisted into a Chat Completions system message.
    let request = serde_json::json!({
        "model": "claude-opus-4-8",
        "max_tokens": 128,
        "system": "You are a coding assistant.",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "translated request should return 200");

    let requests = backend.requests();
    let forwarded = requests
        .iter()
        .find(|r| r.method == "POST")
        .expect("backend should receive a POST request");
    let body: Value = serde_json::from_str(&forwarded.body).expect("forwarded body should be JSON");

    assert!(
        body.get("system").is_none(),
        "the Anthropic top-level `system` must not survive translation: {body}"
    );
    let messages = body["messages"]
        .as_array()
        .expect("translated body should carry a Chat Completions `messages` array");
    assert!(
        messages
            .iter()
            .any(|m| m["role"] == "system" && m["content"] == "You are a coding assistant."),
        "the Anthropic `system` must be hoisted into a Chat Completions system message: {messages:?}"
    );

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Credential isolation
// -----------------------------------------------------------------------------

#[test]
fn transform_vllm_strips_client_credentials_and_injects_backend_bearer() {
    let injected = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    let backend = start_chat_backend();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    // The client presents two distinct credentials: the native Anthropic
    // `x-api-key` and the gateway `Authorization: Basic ...`. Neither may reach
    // the backend; only the injected server-owned Bearer token may.
    let gateway = basic_auth_header(GATEWAY_USER, gateway_password());
    let gateway_secret = gateway.trim_start_matches("Basic ").to_owned();
    let body = r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &format!(
            "POST /v1/messages HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             x-api-key: client-anthropic-secret\r\n\
             Authorization: {gateway}\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n\
             {body}",
            body.len()
        ),
    );

    assert_eq!(parse_status(&raw), 200, "credential-injected request should return 200");

    let requests = backend.requests();
    let forwarded = requests
        .iter()
        .find(|r| r.method == "POST")
        .expect("backend should receive a POST request");
    let headers = &forwarded.headers;
    assert!(
        !headers.contains("client-anthropic-secret"),
        "client x-api-key must be stripped before reaching the backend: {headers}"
    );
    assert!(
        !headers.contains(&gateway_secret),
        "the gateway Basic credential must be stripped before reaching the backend: {headers}"
    );
    assert!(
        headers.contains(&format!("Bearer {injected}")),
        "backend must receive the injected server-owned Bearer token: {headers}"
    );

    drop(proxy);
}

#[test]
fn transform_vllm_rejects_unauthenticated_gateway_request() {
    // `basic_auth` runs first and gates every path: a caller with no gateway
    // credential is rejected before any translation, routing, or injection.
    let backend = start_chat_backend();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let body = r#"{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", body, "X-Unused: 1"),
    );

    assert_eq!(
        parse_status(&raw),
        401,
        "an unauthenticated caller must be rejected by the gateway"
    );

    drop(proxy);
}

#[test]
fn streaming_non2xx_response_yields_single_anthropic_error() {
    // A streaming request whose upstream returns a non-2xx JSON error must be
    // normalized to exactly one Anthropic error object. Two 429s cover the
    // optional Pingora health probe consuming the first response; the streaming
    // POST still receives a 429 either way.
    let backend = StatefulCapturingBackend::new(vec![
        (
            429,
            json!({"error": {"message": "rate limited", "type": "rate_limit_error", "code": "rate_limit_exceeded"}})
                .to_string(),
        ),
        (
            429,
            json!({"error": {"message": "rate limited", "type": "rate_limit_error", "code": "rate_limit_exceeded"}})
                .to_string(),
        ),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let request =
        r#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", request, &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        429,
        "the upstream error status passes through: {raw}"
    );
    let body = parse_body(&raw);
    // `from_str` rejects trailing data: a clean parse proves there is no second,
    // concatenated JSON object appended after the translated error.
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["type"], "error",
        "the client receives an Anthropic error envelope: {body}"
    );
    assert_eq!(
        response["error"]["type"], "rate_limit_error",
        "the upstream 429 maps to an Anthropic rate_limit_error: {body}"
    );
    assert_eq!(
        response["error"]["message"], "rate limited",
        "the real upstream message survives translation: {body}"
    );
    drop(proxy);
}

#[test]
fn streaming_non2xx_split_across_chunks_yields_single_anthropic_error() {
    // The split-delivery regression: a standalone `stream:true` request whose
    // upstream returns a JSON error transfer-encoded across multiple chunks. The
    // translator ratchets the response body into `StreamBuffer` mode, so praxis
    // presents each raw chunk mid-stream AND re-presents the frozen full body at
    // end of stream. The filter must transform that buffered body exactly once;
    // accumulating the chunks itself would append the body twice and degrade to a
    // generic "upstream request failed", losing the real upstream message.
    //
    // `stall_after_first_chunk` forces the proxy to read the first chunk as a
    // separate frame, guaranteeing the split mid-stream delivery this exercises.
    let error =
        json!({"error": {"message": "rate limited: retry in 60 seconds", "type": "rate_limit_error"}}).to_string();
    let (head, tail) = error.split_at(30);
    let backend = Backend::chunked(vec![head.to_owned(), tail.to_owned()])
        .status(429)
        .header("content-type", "application/json")
        .stall_after_first_chunk(Duration::from_millis(50))
        .start_with_shutdown();
    let proxy_port = free_port();
    let config = transform_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let request =
        r#"{"model":"claude-opus-4-8","max_tokens":64,"messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", request, &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        429,
        "the upstream error status passes through: {raw}"
    );
    let body = parse_body(&raw);
    // `from_str` rejects trailing data: a clean parse proves the split chunks were
    // not accumulated on top of the framework's buffered body (a doubled body is
    // invalid JSON).
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["type"], "error",
        "the client receives an Anthropic error envelope: {body}"
    );
    assert_eq!(
        response["error"]["type"], "rate_limit_error",
        "the upstream 429 maps to an Anthropic rate_limit_error: {body}"
    );
    assert_eq!(
        response["error"]["message"], "rate limited: retry in 60 seconds",
        "the real upstream message survives split delivery, not a doubled-body fallback: {body}"
    );
    drop(proxy);
}
