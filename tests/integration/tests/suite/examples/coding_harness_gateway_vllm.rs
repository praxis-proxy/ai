// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the unified coding-harness example config.
//!
//! `coding-harness-gateway-vllm.yaml` serves all three supported coding CLIs
//! from one configuration, one proxy process, and one vLLM backend:
//!
//! ```text
//! Codex       -> :8081 /v1/responses        -> vLLM /v1/responses
//! Claude Code -> :8080 /v1/messages         -> vLLM /v1/messages
//! OpenCode    -> :8080 /v1/chat/completions -> vLLM /v1/chat/completions
//! ```
//!
//! These tests assert the four properties the config exists to guarantee:
//!
//! 1. The two chains stay split. The Responses chain terminates in an `iterative_request_router`; the Claude Code /
//!    OpenCode chain contains no IRR at all. Merging them would route native `/v1/messages` and `/v1/chat/completions`
//!    SSE through the IRR, which buffers a sub-request response unless a filter selects
//!    `SubRequestResponseMode::Streaming` — and neither native path selects it. See
//!    `messages_chain_has_no_irr_or_responses_filters`.
//! 2. Each client's native wire format reaches the backend untranslated, on its own path.
//! 3. Codex's rich client tools are lowered to plain functions on the wire the backend sees.
//! 4. Both listeners share one gateway credential, strip every client credential, and inject the backend's own Bearer
//!    token.
//!
//! `basic_auth` and `credential_injection` both resolve secrets at pipeline
//! build time. `std::env::set_var` is `unsafe` (and `unsafe_code` is denied
//! workspace-wide), so instead of mutating the environment the gateway password
//! is inlined and `VLLM_API_KEY` is repointed to `CARGO_PKG_NAME` — always set
//! by Cargo for a test binary (mirrors `anthropic_messages_native_vllm.rs`).

use praxis_core::config::Config;
use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, assert_error_is_anthropic_shaped, assert_error_is_openai_shaped,
    basic_auth_header, build_pipeline, example_config_path, free_port, http_send, json_post_with_header, parse_body,
    parse_status, start_capturing_backend, start_proxy, start_uri_echo_backend, wait_for_http,
};

const CONFIG: &str = "coding-harness-gateway-vllm.yaml";

/// The gateway `basic_auth` username the example config configures.
const GATEWAY_USER: &str = "gateway";

/// Ports the example binds, replaced with free ports before the proxy starts.
const MESSAGES_LISTENER: &str = "127.0.0.1:8080";
const RESPONSES_LISTENER: &str = "127.0.0.1:8081";
const VLLM_ENDPOINT: &str = "127.0.0.1:8000";

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

/// The `Authorization` header line all three CLIs present to the gateway.
fn gateway_auth_line() -> String {
    format!("Authorization: {}", basic_auth_header(GATEWAY_USER, gateway_password()))
}

/// Addresses of a started two-listener gateway.
struct Ports {
    /// Listener serving Claude Code and OpenCode.
    messages: u16,
    /// Listener serving Codex.
    responses: u16,
}

/// Read the example and rewrite its two listener ports, its backend endpoint,
/// its store URL, and both build-time secrets.
///
/// The port rewrite goes through placeholders rather than
/// `praxis_test_utils::patch_yaml` because this config carries three distinct
/// `127.0.0.1:<port>` literals: substituting them in place risks a freshly
/// allocated port colliding with a literal that has not been replaced yet.
fn coding_harness_config(ports: &Ports, backend_port: u16, db_url: &str) -> Config {
    let path = example_config_path(CONFIG);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    let staged = yaml
        .replace(VLLM_ENDPOINT, "@@VLLM@@")
        .replace(RESPONSES_LISTENER, "@@RESPONSES@@")
        .replace(MESSAGES_LISTENER, "@@MESSAGES@@");
    let patched = staged
        .replace("@@VLLM@@", &format!("127.0.0.1:{backend_port}"))
        .replace("@@RESPONSES@@", &format!("127.0.0.1:{}", ports.responses))
        .replace("@@MESSAGES@@", &format!("127.0.0.1:{}", ports.messages))
        .replace("sqlite://responses.db?mode=rwc", db_url)
        .replace("env_var: VLLM_API_KEY", "env_var: CARGO_PKG_NAME")
        .replace(
            "env_var: GATEWAY_AUTH_PASSWORD",
            &format!("password: {}", gateway_password()),
        );

    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse {CONFIG}: {e}"))
}

/// Load the example for shape assertions, with ports that are never bound.
fn shape_config() -> Config {
    let ports = Ports {
        messages: 29_940,
        responses: 29_941,
    };
    let db = TempSqlite::new("coding_harness_shape");
    coding_harness_config(&ports, 29_942, db.url())
}

/// Look up a chain by name.
fn chain<'a>(config: &'a Config, name: &str) -> &'a praxis_core::config::FilterChainConfig {
    config
        .filter_chains
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("config should contain a {name} chain"))
}

/// Filter type names of a chain, in order.
fn filter_types(chain: &praxis_core::config::FilterChainConfig) -> Vec<&str> {
    chain.filters.iter().map(|f| f.filter_type.as_str()).collect()
}

// -----------------------------------------------------------------------------
// Chain shape
// -----------------------------------------------------------------------------

#[test]
fn coding_harness_config_parses_two_listeners() {
    let config = shape_config();

    assert_eq!(config.listeners.len(), 2, "one listener per wire protocol group");
    let names: Vec<&str> = config.listeners.iter().map(|l| &*l.name).collect();
    assert_eq!(
        names,
        ["messages-chat-gateway", "responses-gateway"],
        "listener names should be stable; the docs reference them"
    );
    assert_eq!(config.filter_chains.len(), 2, "one chain per listener");
}

#[test]
fn example_config_builds_pipeline() {
    let config = shape_config();
    let _pipeline = build_pipeline(&config);
}

#[test]
fn messages_chain_has_no_irr_or_responses_filters() {
    // This is the regression guard for the whole two-listener design. An
    // `iterative_request_router` buffers its sub-request response unless a
    // filter selects `SubRequestResponseMode::Streaming`, and native
    // `/v1/messages` and `/v1/chat/completions` select nothing. Folding this
    // chain into the Responses chain would therefore leave Claude Code and
    // OpenCode rendering nothing until each turn completed.
    let config = shape_config();
    let types = filter_types(chain(&config, "claude-and-opencode"));

    assert_eq!(
        types,
        [
            "basic_auth",
            "ai_operation",
            "anthropic_messages_request",
            "anthropic_messages_protocol",
            "headers",
            "router",
            "credential_injection",
            "load_balancer",
        ],
        "the Claude Code / OpenCode chain must stay a flat native passthrough, in order"
    );

    for banned in [
        "iterative_request_router",
        "openai_stream_events",
        "openai_agentic_loop",
        "openai_client_tool_compat",
        "openai_responses_proxy",
        "anthropic_messages_to_chat_completions",
        "anthropic_messages_to_chat_completions_stream",
        "responses_to_chat_completions",
    ] {
        assert!(
            !types.contains(&banned),
            "{banned} must not appear on the native streaming chain: {types:?}"
        );
    }
}

#[test]
fn responses_chain_terminates_in_the_iterative_router() {
    let config = shape_config();
    let types = filter_types(chain(&config, "codex-responses"));

    assert_eq!(
        types,
        [
            "basic_auth",
            "ai_operation",
            "openai_responses_request",
            "openai_responses_request",
            "state_owner",
            "openai_response_store",
            "openai_responses_rehydrate",
            "iterative_request_router",
        ],
        "the Codex chain must build Responses state before the router, in order"
    );
    assert_eq!(
        types.last(),
        Some(&"iterative_request_router"),
        "the IRR is a terminal filter, so nothing may follow it"
    );
}

#[test]
fn responses_step_runs_stream_events_before_client_tool_compat() {
    // `openai_client_tool_compat` fails closed with HTTP 500 on a streaming
    // Responses request unless `openai_stream_events` precedes it in the same
    // IRR step to restore the lowered calls live in the stream.
    let config = shape_config();
    let irr = chain(&config, "codex-responses")
        .filters
        .iter()
        .find(|f| f.filter_type == "iterative_request_router")
        .expect("the Codex chain should contain an iterative_request_router");

    let steps = irr
        .config
        .get("steps")
        .and_then(serde_yaml::Value::as_sequence)
        .expect("the IRR should declare steps");
    let step_filters: Vec<&str> = steps
        .iter()
        .filter_map(|step| step.get("filters"))
        .filter_map(serde_yaml::Value::as_sequence)
        .flatten()
        .filter_map(|f| f.get("filter"))
        .filter_map(serde_yaml::Value::as_str)
        .collect();

    assert_eq!(
        step_filters,
        [
            "openai_stream_events",
            "openai_agentic_loop",
            "openai_client_tool_compat",
            "openai_responses_proxy",
            "router",
            "credential_injection",
            "load_balancer",
        ],
        "inference-step order is load-bearing and test-locked"
    );
}

#[test]
fn anthropic_request_processing_needs_no_path_condition() {
    // The processor is keyed to the create-message operation, so OpenCode's
    // `/v1/chat/completions` traffic and the bodyless `GET /v1/models` probe
    // are released untouched on this shared listener. The separate validator
    // this replaced rejected both and had to be gated to `/v1/messages`;
    // carrying that condition forward would be redundant scoping that hides
    // where the rule lives.
    let config = shape_config();
    let processor = chain(&config, "claude-and-opencode")
        .filters
        .iter()
        .find(|f| f.filter_type == "anthropic_messages_request")
        .expect("chain should contain anthropic_messages_request");

    assert!(
        processor.conditions.is_empty(),
        "the operation keys the processor, so no path condition should be needed"
    );
}

#[test]
fn both_listeners_authenticate_before_anything_else() {
    let config = shape_config();

    for name in ["claude-and-opencode", "codex-responses"] {
        let types = filter_types(chain(&config, name));
        assert_eq!(
            types.first(),
            Some(&"basic_auth"),
            "{name} must authenticate the client before any routing or body work"
        );
    }
}

// -----------------------------------------------------------------------------
// Native passthrough
// -----------------------------------------------------------------------------

#[test]
fn claude_code_message_body_reaches_vllm_unchanged() {
    let backend = start_capturing_backend(r#"{"type":"message","role":"assistant","content":[]}"#);
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_messages");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    // `system` and `max_tokens` are exactly the fields a Chat Completions
    // translation would rewrite or drop.
    let request = serde_json::json!({
        "model": "qwen3-8b",
        "max_tokens": 1024,
        "system": "You are a coding assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "native Anthropic request should return 200");
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    assert_eq!(
        forwarded, request,
        "backend must receive the Anthropic body byte-for-byte, with no translation"
    );

    drop(proxy);
}

#[test]
fn opencode_chat_completions_share_the_messages_listener() {
    // OpenCode speaks `/v1/chat/completions`. It reaches the same vLLM backend
    // over the same listener and the same gateway credential as Claude Code,
    // with its path preserved and no Anthropic validation applied to it.
    let backend = start_uri_echo_backend();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_chat");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    let body = r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}]}"#;
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/chat/completions", body, &gateway_auth_line()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "OpenCode traffic must not be rejected by Anthropic validation"
    );
    assert_eq!(
        parse_body(&raw),
        "/v1/chat/completions",
        "the Chat Completions path must reach the backend unchanged"
    );

    drop(proxy);
}

#[test]
fn bodyless_model_probe_passes_through() {
    let backend = start_uri_echo_backend();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_probe");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &format!(
            "GET /v1/models HTTP/1.1\r\n\
             Host: localhost\r\n\
             {}\r\n\
             Connection: close\r\n\r\n",
            gateway_auth_line()
        ),
    );

    assert_eq!(parse_status(&raw), 200, "bodyless startup probe should return 200");
    assert_eq!(
        parse_body(&raw),
        "/v1/models",
        "model-discovery path must reach the backend unchanged"
    );

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Error shape on the Claude Code / OpenCode listener
// -----------------------------------------------------------------------------

/// OpenCode parses OpenAI-shaped errors, so a proxy-generated failure on
/// `/v1/chat/completions` must not fall back to RFC 9457 problem details.
///
/// `anthropic_messages_request` installs the Anthropic formatter only on the
/// Anthropic Messages surface and leaves everything else entirely alone, so
/// `/v1/chat/completions` reaches core's error path with no formatter from it.
/// Only the head-driven `ai_operation` covers this protocol.
#[test]
fn opencode_chat_completions_failure_keeps_the_openai_error_shape() {
    // Nothing listens on this port, so the failure is generated by the proxy
    // rather than returned by a backend.
    let dead_port = free_port();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_chat_error");
    let config = coding_harness_config(&ports, dead_port, db.url());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/chat/completions",
            r#"{"model":"qwen3-8b","messages":[{"role":"user","content":"Hi"}]}"#,
            &gateway_auth_line(),
        ),
    );

    assert_error_is_openai_shaped(&raw);

    drop(proxy);
}

/// Claude Code keeps Anthropic-shaped errors on the same listener. Guards the
/// head-driven classifier against regressing the protocol it already served.
#[test]
fn claude_code_messages_failure_keeps_the_anthropic_error_shape() {
    let dead_port = free_port();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_messages_error");
    let config = coding_harness_config(&ports, dead_port, db.url());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_header(
            "/v1/messages",
            r#"{"model":"qwen3-8b","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#,
            &gateway_auth_line(),
        ),
    );

    assert_error_is_anthropic_shaped(&raw);

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Codex: rich client tools over the Responses listener
// -----------------------------------------------------------------------------

#[test]
fn codex_custom_tool_is_lowered_for_the_responses_backend() {
    let backend_response = serde_json::json!({
        "id": "resp_codex",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_abc",
            "call_id": "call_abc",
            "name": "apply_patch",
            "arguments": r#"{"input":"*** Begin Patch"}"#,
            "status": "completed"
        }]
    });
    let backend = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_codex");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    let responses_addr = format!("127.0.0.1:{}", ports.responses);
    wait_for_http(&responses_addr);

    let request = serde_json::json!({
        "model": "qwen3-8b",
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }],
    });
    let raw = http_send(
        &responses_addr,
        &json_post_with_header("/v1/responses", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "Codex Responses request should return 200");

    let captured = backend.requests();
    let inference = captured
        .iter()
        .find(|r| r.uri.starts_with("/v1/responses"))
        .expect("the backend should receive a /v1/responses inference request");
    let forwarded: serde_json::Value =
        serde_json::from_str(&inference.body).expect("forwarded backend body should be JSON");
    let tool = &forwarded["tools"][0];

    assert_eq!(
        tool["type"], "function",
        "the rich `custom` tool must be lowered to a plain function the backend accepts: {forwarded}"
    );
    assert_eq!(
        tool["name"], "apply_patch",
        "the lowered tool keeps the client's tool name: {forwarded}"
    );

    // The client must never see the lowered shape: it is restored on the way back.
    let client_view: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client body should be JSON");
    assert_eq!(
        client_view["output"][0]["type"], "custom_tool_call",
        "the backend's function_call must be restored to the canonical typed item: {client_view}"
    );

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Credential isolation
// -----------------------------------------------------------------------------

#[test]
fn messages_listener_strips_client_credentials_and_injects_backend_bearer() {
    let injected = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    let backend = StatefulCapturingBackend::new(vec![(
        200,
        r#"{"type":"message","role":"assistant","content":[]}"#.to_owned(),
    )])
    .start_with_shutdown();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_creds");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    // The client presents two distinct credentials: the native Anthropic
    // `x-api-key` and the gateway `Authorization: Basic ...`. Neither may reach
    // the backend; only the injected server-owned Bearer token may.
    let gateway = basic_auth_header(GATEWAY_USER, gateway_password());
    let gateway_secret = gateway.trim_start_matches("Basic ").to_owned();
    let body = r#"{"model":"qwen3-8b","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
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
    let headers = backend
        .requests()
        .first()
        .map(|r| r.headers.clone())
        .expect("the backend should receive the request");

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
fn responses_listener_injects_the_same_backend_bearer() {
    let injected = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    let backend_response = serde_json::json!({
        "id": "resp_creds",
        "object": "response",
        "status": "completed",
        "output": [],
    });
    let backend = StatefulCapturingBackend::new(vec![(200, backend_response.to_string())]).start_with_shutdown();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_codex_creds");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    let responses_addr = format!("127.0.0.1:{}", ports.responses);
    wait_for_http(&responses_addr);

    let gateway = basic_auth_header(GATEWAY_USER, gateway_password());
    let gateway_secret = gateway.trim_start_matches("Basic ").to_owned();
    let body = r#"{"model":"qwen3-8b","input":"Hi"}"#;
    let raw = http_send(
        &responses_addr,
        &json_post_with_header("/v1/responses", body, &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "Codex request should return 200");
    let headers = backend
        .requests()
        .first()
        .map(|r| r.headers.clone())
        .expect("the backend should receive the request");

    assert!(
        !headers.contains(&gateway_secret),
        "the gateway Basic credential must be stripped before reaching the backend: {headers}"
    );
    assert!(
        headers.contains(&format!("Bearer {injected}")),
        "both listeners must present the same server-owned backend credential: {headers}"
    );

    drop(proxy);
}

#[test]
fn both_listeners_reject_an_unauthenticated_request() {
    let backend = start_uri_echo_backend();
    let ports = Ports {
        messages: free_port(),
        responses: free_port(),
    };
    let db = TempSqlite::new("coding_harness_unauth");
    let config = coding_harness_config(&ports, backend.port(), db.url());
    let proxy = start_proxy(&config);

    let responses_addr = format!("127.0.0.1:{}", ports.responses);
    wait_for_http(&responses_addr);

    let messages_body = r#"{"model":"qwen3-8b","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#;
    let messages_raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", messages_body, "X-Unused: 1"),
    );
    let responses_raw = http_send(
        &responses_addr,
        &json_post_with_header("/v1/responses", r#"{"model":"qwen3-8b","input":"Hi"}"#, "X-Unused: 1"),
    );

    assert_eq!(
        parse_status(&messages_raw),
        401,
        "the messages listener must reject a caller with no gateway credential"
    );
    assert_eq!(
        parse_status(&responses_raw),
        401,
        "the Responses listener must reject a caller with no gateway credential"
    );

    drop(proxy);
}

// -----------------------------------------------------------------------------
// Client-facing contract
// -----------------------------------------------------------------------------

#[test]
fn codex_must_not_be_configured_with_env_key() {
    // Documentation guard, asserted against the config the doc describes.
    //
    // Codex builds its provider headers (`http_headers`, `env_http_headers`)
    // independently of `env_key`, and applies the `env_key` bearer with an
    // Append — not a Replace. Setting both would send two `Authorization`
    // headers. The doc therefore tells Codex users to carry the gateway Basic
    // credential in `env_http_headers` and omit `env_key` entirely, which only
    // works because this chain authenticates with `basic_auth`.
    let config = shape_config();
    let auth = chain(&config, "codex-responses")
        .filters
        .first()
        .expect("the Codex chain should not be empty");

    assert_eq!(
        auth.filter_type, "basic_auth",
        "the Codex listener gates on Basic auth, so Codex must present Basic (not Bearer) credentials"
    );
}
