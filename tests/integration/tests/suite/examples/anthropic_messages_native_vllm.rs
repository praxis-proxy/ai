// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the native-vLLM Anthropic Messages example config.
//!
//! `messages-native-vllm.yaml` proxies native Anthropic Messages traffic
//! (`/v1/messages`, `/v1/messages/count_tokens`) straight to a vLLM backend
//! that serves the Anthropic Messages API natively — WITHOUT any request or
//! response body translation. These tests assert three properties:
//!
//! 1. The chain contains the native Anthropic filters and NONE of the Anthropic->Chat-Completions translation or
//!    agentic filters.
//! 2. Request bodies reach the backend byte-for-byte (native passthrough), and native paths (`count_tokens`,
//!    `/v1/models`) are preserved upstream.
//! 3. Client credentials are stripped and the backend's own Bearer token is injected.
//!
//! `credential_injection` resolves its `env_var` at pipeline-build time and
//! errors if the variable is unset. Since `std::env::set_var` is `unsafe` in
//! this edition and `unsafe_code` is denied workspace-wide, the `start_proxy`
//! tests repoint `VLLM_API_KEY` to `CARGO_PKG_NAME` — always set by Cargo for
//! any test binary — instead of mutating the environment (mirrors
//! `aws_sigv4.rs`).

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    thread,
    time::Duration,
};

use praxis_core::config::Config;
use praxis_test_utils::{
    basic_auth_header, free_port, http_send, json_post_with_header, parse_body, parse_status, start_capturing_backend,
    start_header_echo_backend, start_proxy, start_uri_echo_backend,
};

use super::load_example_config;

const CONFIG: &str = "anthropic/messages-native-vllm.yaml";

/// The gateway `basic_auth` username the example config configures.
const GATEWAY_USER: &str = "gateway";

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

/// Build the native-vLLM config with ports patched, `VLLM_API_KEY` repointed to
/// a Cargo-provided variable, and the gateway password inlined so pipeline build
/// succeeds without mutating the environment.
///
/// `basic_auth` and `credential_injection` both resolve secrets at pipeline
/// build time. `std::env::set_var` is `unsafe` (and `unsafe_code` is denied
/// workspace-wide), so instead of setting the environment the gateway password
/// is inlined and `VLLM_API_KEY` is repointed to `CARGO_PKG_NAME` — always set
/// by Cargo for a test binary.
fn native_vllm_config(proxy_port: u16, backend_port: u16) -> Config {
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

// -----------------------------------------------------------------------------
// Chain shape
// -----------------------------------------------------------------------------

#[test]
fn native_vllm_config_parses() {
    let config = load_example_config(CONFIG, 29920, HashMap::from([("127.0.0.1:8000", 29921_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(
        &*config.listeners[0].name, "anthropic-gateway",
        "listener name should be anthropic-gateway"
    );
    assert_eq!(config.filter_chains.len(), 1, "should have 1 filter chain");
    assert_eq!(
        config.filter_chains[0].name, "anthropic-native-vllm",
        "chain name should be anthropic-native-vllm"
    );
}

#[test]
fn native_vllm_chain_is_native_passthrough_not_translation() {
    let config = load_example_config(CONFIG, 29922, HashMap::from([("127.0.0.1:8000", 29923_u16)]));
    let chain = &config.filter_chains[0];
    let types: Vec<&str> = chain.filters.iter().map(|f| f.filter_type.as_str()).collect();

    assert_eq!(
        types,
        [
            "basic_auth",
            "anthropic_messages_format",
            "anthropic_validate",
            "anthropic_messages_protocol",
            "headers",
            "router",
            "credential_injection",
            "load_balancer",
        ],
        "native-vLLM chain must be the native passthrough shape, in order"
    );

    // The whole point of native passthrough: no body translation, no agentic
    // orchestration. Any of these appearing would mean we are not passing the
    // Anthropic wire format through unchanged.
    for banned in [
        "anthropic_messages_to_chat_completions",
        "anthropic_messages_to_chat_completions_stream",
        "anthropic_web_search",
        "responses_to_chat_completions",
        "openai_agentic_loop",
        "openai_mcp_dispatch",
        "mcp",
        "a2a",
    ] {
        assert!(
            !types.contains(&banned),
            "native passthrough chain must not contain the {banned} filter: {types:?}"
        );
    }
}

#[test]
fn native_vllm_validate_is_scoped_to_messages() {
    // `anthropic_validate` rejects bodyless requests, so it must be gated to
    // `/v1/messages` — otherwise the bodyless `GET /v1/models` startup probe
    // would be rejected with 400.
    let config = load_example_config(CONFIG, 29924, HashMap::from([("127.0.0.1:8000", 29925_u16)]));
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
// Native passthrough (bodies and paths)
// -----------------------------------------------------------------------------

#[test]
fn native_vllm_forwards_message_body_unchanged() {
    let backend = start_capturing_backend(r#"{"type":"message","role":"assistant","content":[]}"#);
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    // An Anthropic-native body: `system` and `max_tokens` are exactly the
    // fields a Chat Completions translation would rewrite or drop.
    let request = serde_json::json!({
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "system": "You are a coding assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "native request should return 200");
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    assert_eq!(
        forwarded, request,
        "backend must receive the Anthropic body byte-for-byte, with no translation"
    );

    drop(proxy);
}

#[test]
fn native_vllm_routes_count_tokens_natively() {
    let backend = start_uri_echo_backend();
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "How many tokens is this?"}],
    });
    let raw = http_send(
        proxy.addr(),
        &json_post_with_header("/v1/messages/count_tokens", &request.to_string(), &gateway_auth_line()),
    );

    assert_eq!(parse_status(&raw), 200, "count_tokens should return 200");
    assert_eq!(
        parse_body(&raw),
        "/v1/messages/count_tokens",
        "count_tokens path must be preserved to the backend, not rewritten"
    );
}

#[test]
fn native_vllm_passes_bodyless_models_probe() {
    // `GET /v1/models` carries no body; because `anthropic_validate` is scoped
    // to `/v1/messages`, the probe must route through to the backend unchanged
    // rather than being rejected for an empty body.
    let backend = start_uri_echo_backend();
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend.port());
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

    assert_eq!(
        parse_status(&raw),
        200,
        "bodyless model-discovery probe should return 200"
    );
    assert_eq!(
        parse_body(&raw),
        "/v1/models",
        "model-discovery path must reach the backend unchanged"
    );
}

// -----------------------------------------------------------------------------
// Credential isolation
// -----------------------------------------------------------------------------

#[test]
fn native_vllm_strips_client_credentials_and_injects_backend_bearer() {
    let injected = std::env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME is always set by cargo test");
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend.port());
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
    let echoed = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "credential-injected request should return 200");
    assert!(
        !echoed.contains("client-anthropic-secret"),
        "client x-api-key must be stripped before reaching the backend: {echoed}"
    );
    assert!(
        !echoed.contains(&gateway_secret),
        "the gateway Basic credential must be stripped before reaching the backend: {echoed}"
    );
    assert!(
        echoed.contains(&format!("Bearer {injected}")),
        "backend must receive the injected server-owned Bearer token: {echoed}"
    );
}

#[test]
fn native_vllm_rejects_unauthenticated_gateway_request() {
    // `basic_auth` runs first and gates every path: a caller with no gateway
    // credential is rejected before any routing or credential injection happens.
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend.port());
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
}

// -----------------------------------------------------------------------------
// Native streaming (incremental SSE forwarding, not buffering)
// -----------------------------------------------------------------------------

/// The opening Anthropic stream events the gated backend flushes first.
const STREAM_FIRST_EVENTS: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_native_stream\",\"type\":\"message\",",
    "\"role\":\"assistant\",\"model\":\"claude-opus-4-8\",\"content\":[],\"stop_reason\":null,",
    "\"usage\":{\"input_tokens\":8,\"output_tokens\":0}}}\n\n",
    "event: content_block_start\n",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
);

/// The terminal Anthropic stream events sent only after the test releases the backend.
const STREAM_FINAL_EVENTS: &str = concat!(
    "event: content_block_stop\n",
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},",
    "\"usage\":{\"output_tokens\":3}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[test]
fn native_vllm_streams_sse_incrementally_before_upstream_eof() {
    // A gated backend flushes the opening Anthropic stream events, then holds the
    // connection open (no EOF) until the test releases it. If Praxis buffered the
    // native streaming response, the client could observe nothing until the
    // backend closed — but the backend only closes AFTER the client reports it
    // saw an event, so a buffering proxy would deadlock and the `observed`
    // receive would time out. Observing an event while the upstream is still
    // gated therefore proves Praxis forwards the native SSE stream incrementally.
    let (backend_port, first_sent, release, backend_thread) = start_gated_sse_backend(
        vec![STREAM_FIRST_EVENTS.to_owned()],
        vec![STREAM_FINAL_EVENTS.to_owned()],
    );
    let proxy_port = free_port();
    let config = native_vllm_config(proxy_port, backend_port);
    let proxy = start_proxy(&config);

    let (observed_tx, observed_rx) = mpsc::channel();
    let (complete_tx, complete_rx) = mpsc::channel();
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let raw = read_sse_incrementally(
            &proxy_addr,
            r#"{"model":"claude-opus-4-8","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"Hi"}]}"#,
            "message_start",
            &observed_tx,
        );
        complete_tx.send(raw).expect("test receiver should remain available");
    });

    first_sent
        .recv_timeout(Duration::from_secs(2))
        .expect("backend should flush the opening stream events");
    observed_rx.recv_timeout(Duration::from_secs(2)).expect(
        "client must observe an SSE event while the upstream is still gated; \
         a timeout here means the native response was buffered, not streamed",
    );
    release
        .send(())
        .expect("backend release receiver should remain available");

    let raw = complete_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("client should receive the completed stream");
    assert_eq!(
        parse_status(&raw),
        200,
        "native streaming response should return 200: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains(r#""text":"hel""#),
        "the content delta must pass through natively, unmodified: {body}"
    );
    assert!(
        body.contains("message_stop"),
        "the terminal event must reach the client: {body}"
    );

    client.join().expect("client thread should not panic");
    backend_thread.join().expect("backend thread should not panic");
}

/// Starts a backend that flushes `first_chunks`, holds the connection open until
/// released, then writes `final_chunks` and closes — a chunked `text/event-stream`
/// response whose timing the test controls, to prove incremental forwarding.
fn start_gated_sse_backend(
    first_chunks: Vec<String>,
    final_chunks: Vec<String>,
) -> (u16, mpsc::Receiver<()>, mpsc::Sender<()>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let port = listener.local_addr().expect("backend should have an address").port();
    let (first_sent_tx, first_sent_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_request(&mut stream);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        .expect("response headers should be written");
        for chunk in first_chunks {
            write_chunk(&mut stream, &chunk);
        }
        stream.flush().expect("initial chunks should flush");
        first_sent_tx.send(()).expect("test receiver should remain available");
        release_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("test should release the backend");
        for chunk in final_chunks {
            write_chunk(&mut stream, &chunk);
        }
        stream.write_all(b"0\r\n\r\n").expect("chunked response should finish");
        stream.flush().expect("terminal chunks should flush");
    });
    (port, first_sent_rx, release_tx, handle)
}

/// Drains an HTTP request from `stream` (headers plus any `Content-Length` body).
fn read_request(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("backend read timeout should be set");
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = stream.read(&mut buffer).expect("backend request read should succeed");
        assert!(count > 0, "request must complete before connection closes");
        request.extend_from_slice(&buffer[..count]);
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
            })
            .unwrap_or(0);
        if request.len() >= header_end + 4 + content_length {
            return;
        }
    }
}

/// Writes one HTTP/1.1 chunk (`hex-length CRLF chunk CRLF`).
fn write_chunk(stream: &mut TcpStream, chunk: &str) {
    write!(stream, "{:x}\r\n{chunk}\r\n", chunk.len()).expect("response chunk should be written");
}

/// Sends a native streaming `POST /v1/messages` and reads the response
/// incrementally, firing `observed` the moment `needle` first appears — which,
/// under a streaming proxy, happens before the upstream sends EOF.
fn read_sse_incrementally(proxy_addr: &str, body: &str, needle: &str, observed: &mpsc::Sender<()>) -> String {
    let mut stream = TcpStream::connect(proxy_addr).expect("client should connect to proxy");
    stream
        .set_read_timeout(Some(Duration::from_secs(4)))
        .expect("client read timeout should be set");
    stream
        .write_all(json_post_with_header("/v1/messages", body, &gateway_auth_line()).as_bytes())
        .expect("client request should be written");

    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut notified = false;
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                raw.extend_from_slice(&buffer[..count]);
                if !notified && String::from_utf8_lossy(&raw).contains(needle) {
                    observed.send(()).expect("test receiver should remain available");
                    notified = true;
                }
            },
            Err(error) => panic!("streaming response read failed: {error}"),
        }
    }
    String::from_utf8_lossy(&raw).into_owned()
}
