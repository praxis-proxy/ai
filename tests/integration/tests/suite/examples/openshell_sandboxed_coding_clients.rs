// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the OpenShell sandboxed-coding-clients example config.
//!
//! `openshell/sandboxed-coding-clients.yaml` is the host-side gateway that
//! coding harnesses inside NVIDIA OpenShell sandboxes reach through
//! `host.openshell.internal`. These tests assert the properties the sandbox
//! guide depends on:
//!
//! 1. Both listeners bind a non-loopback address, because OpenShell never authorizes sandbox egress to loopback.
//! 2. Both listeners require the shared gateway Basic credential.
//! 3. Claude Code, OpenCode/pi, and Codex bodies reach vLLM natively, with the gateway credential stripped and the
//!    server-owned vLLM Bearer injected, including inside the Codex chain's iterative-router step.
//!
//! As in `anthropic_messages_native_vllm.rs`, the gateway password is inlined
//! and `VLLM_API_KEY` is repointed to `CARGO_PKG_NAME`, because both are
//! resolved at pipeline build time and `std::env::set_var` is `unsafe`.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, basic_auth_header, build_pipeline, example_config_path, free_port, http_send,
    json_post_with_header, parse_body, parse_status, start_capturing_backend, start_header_echo_backend, start_proxy,
    wait_for_tcp,
};

const CONFIG: &str = "openshell/sandboxed-coding-clients.yaml";

/// The gateway `basic_auth` username the example config configures.
const GATEWAY_USER: &str = "gateway";

/// The Codex listener address in the unpatched example.
const CODEX_LISTENER: &str = "0.0.0.0:8081";

/// The gateway password these tests inject in place of `GATEWAY_AUTH_PASSWORD`.
///
/// Drawn once per test process from the OS RNG so it is a throwaway secret
/// scoped to this run.
fn gateway_password() -> &'static str {
    static PASSWORD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PASSWORD.get_or_init(|| {
        rand::random::<[u8; 16]>()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    })
}

/// The unpatched example YAML.
fn example_yaml() -> String {
    let path = example_config_path(CONFIG);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Build the example with both listeners and the vLLM backend repointed, the
/// response store moved to `db_url`, and both secrets resolvable without
/// mutating the environment.
fn sandboxed_config(messages_port: u16, codex_port: u16, backend_port: u16, db_url: &str) -> Config {
    let yaml = example_yaml().replace("sqlite://responses.db?mode=rwc", db_url);
    let patched = praxis_test_utils::patch_yaml(
        &yaml,
        messages_port,
        &HashMap::from([(CODEX_LISTENER, codex_port), ("127.0.0.1:8000", backend_port)]),
    );
    let patched = patched.replace("env_var: VLLM_API_KEY", "env_var: CARGO_PKG_NAME");
    let patched = patched.replace(
        "env_var: GATEWAY_AUTH_PASSWORD",
        &format!("password: {}", gateway_password()),
    );
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"))
}

/// The `Authorization` header line a sandboxed harness presents once the
/// OpenShell proxy has substituted the real password.
fn gateway_auth_line() -> String {
    format!("Authorization: {}", basic_auth_header(GATEWAY_USER, gateway_password()))
}

/// The Base64 payload of the gateway credential, which must never reach vLLM.
fn gateway_secret() -> String {
    basic_auth_header(GATEWAY_USER, gateway_password())
        .trim_start_matches("Basic ")
        .to_owned()
}

/// The Bearer token `credential_injection` adds toward vLLM in these tests.
fn injected_bearer() -> String {
    let key = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    format!("Bearer {key}")
}

/// A Claude Code create-message body.
fn claude_code_body() -> serde_json::Value {
    serde_json::json!({
        "model": "qwen3-8b",
        "max_tokens": 1024,
        "system": "You are a coding assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    })
}

/// An OpenCode or pi Chat Completions body.
fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "List the files here."}],
    })
}

// -----------------------------------------------------------------------------
// Config shape
// -----------------------------------------------------------------------------

#[test]
fn sandboxed_config_builds_pipeline() {
    let db = TempSqlite::new("openshell_sandboxed_build");
    let config = sandboxed_config(free_port(), free_port(), 19961, db.url());
    let _pipeline = build_pipeline(&config);
}

#[test]
fn sandboxed_listeners_bind_a_non_loopback_address() {
    // Checked on the unpatched YAML: the test loader rewrites every listener to
    // loopback, which would hide a regression here.
    let config = Config::from_yaml(&example_yaml()).expect("example config should parse");
    assert_eq!(
        config.listeners.len(),
        2,
        "one listener for messages/chat, one for Codex"
    );
    for listener in &config.listeners {
        assert!(
            !listener.address.starts_with("127.") && !listener.address.starts_with("[::1]"),
            "listener {} binds {}; sandboxes reach the host through host.openshell.internal, \
             and OpenShell never authorizes egress to loopback",
            listener.name,
            listener.address
        );
    }
}

// -----------------------------------------------------------------------------
// Messages and Chat Completions listener (Claude Code, OpenCode, pi)
// -----------------------------------------------------------------------------

#[test]
fn messages_listener_rejects_unauthenticated_request() {
    let backend = start_header_echo_backend();
    let db = TempSqlite::new("openshell_sandboxed_messages_401");
    let config = sandboxed_config(free_port(), free_port(), backend.port(), db.url());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &claude_code_body().to_string(), "X-Unused: 1"),
    );

    assert_eq!(
        parse_status(&raw),
        401,
        "a sandboxed client without the gateway credential must be rejected"
    );
}

#[test]
fn messages_listener_forwards_claude_code_body_unchanged() {
    let backend = start_capturing_backend(r#"{"type":"message","role":"assistant","content":[]}"#);
    let db = TempSqlite::new("openshell_sandboxed_messages_body");
    let config = sandboxed_config(free_port(), free_port(), backend.port(), db.url());
    let proxy = start_proxy(&config);

    let request = claude_code_body();
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "authenticated Claude Code request should return 200"
    );
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    assert_eq!(forwarded, request, "vLLM must receive the Anthropic body unchanged");
}

#[test]
fn messages_listener_swaps_gateway_credential_for_vllm_bearer() {
    let backend = start_header_echo_backend();
    let db = TempSqlite::new("openshell_sandboxed_messages_creds");
    let config = sandboxed_config(free_port(), free_port(), backend.port(), db.url());
    let proxy = start_proxy(&config);

    for (path, body) in [
        ("/v1/messages", claude_code_body()),
        ("/v1/chat/completions", chat_body()),
    ] {
        let raw = http_send(
            proxy.addr(),
            &json_post_with_header(path, &body.to_string(), &gateway_auth_line()),
        );
        let echoed = parse_body(&raw);

        assert_eq!(parse_status(&raw), 200, "{path} should return 200: {raw}");
        assert!(
            !echoed.contains(&gateway_secret()),
            "{path}: the gateway credential must be stripped before vLLM: {echoed}"
        );
        assert!(
            echoed.contains(&injected_bearer()),
            "{path}: vLLM must receive the server-owned Bearer token: {echoed}"
        );
    }
}

// -----------------------------------------------------------------------------
// Codex listener
// -----------------------------------------------------------------------------

#[test]
fn codex_listener_rejects_unauthenticated_request() {
    let backend = start_header_echo_backend();
    let codex_port = free_port();
    let db = TempSqlite::new("openshell_sandboxed_codex_401");
    let config = sandboxed_config(free_port(), codex_port, backend.port(), db.url());
    let _proxy = start_proxy(&config);
    let codex_addr = format!("127.0.0.1:{codex_port}");
    wait_for_tcp(&codex_addr);

    let body = serde_json::json!({"model": "qwen3-8b", "input": "Hi"});
    let raw = http_send(
        &codex_addr,
        &json_post_with_header("/v1/responses", &body.to_string(), "X-Unused: 1"),
    );

    assert_eq!(
        parse_status(&raw),
        401,
        "the Codex listener must require the same gateway credential"
    );
}

#[test]
fn codex_listener_injects_vllm_bearer_inside_the_iterative_router() {
    // `basic_auth` strips the caller's `Authorization` at the head of the
    // chain, so the inference sub-request carries the vLLM key only because the
    // IRR step injects it. Without that step keyed vLLM answers 401.
    let backend_response = serde_json::json!({
        "id": "resp_sandboxed",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}]
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let codex_port = free_port();
    let db = TempSqlite::new("openshell_sandboxed_codex_creds");
    let config = sandboxed_config(free_port(), codex_port, model.port(), db.url());
    let _proxy = start_proxy(&config);
    let codex_addr = format!("127.0.0.1:{codex_port}");
    wait_for_tcp(&codex_addr);

    let request = serde_json::json!({"model": "qwen3-8b", "input": "Reply with OK."});
    let raw = http_send(
        &codex_addr,
        &json_post_with_header("/v1/responses", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "authenticated Codex request should return 200: {raw}"
    );
    let inference: Vec<_> = model
        .requests()
        .into_iter()
        .filter(|request| request.method == "POST")
        .collect();
    assert_eq!(inference.len(), 1, "one inference round reaches vLLM");
    assert_eq!(
        inference[0].uri, "/v1/responses",
        "Codex stays on the native Responses path"
    );
    let headers = inference[0].headers.to_ascii_lowercase();
    assert!(
        !inference[0].headers.contains(&gateway_secret()),
        "the gateway credential must be stripped before vLLM: {headers}"
    );
    assert!(
        headers.contains(&injected_bearer().to_ascii_lowercase()),
        "vLLM must receive the server-owned Bearer token: {headers}"
    );
}
