// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `compact` example config.
//!
//! Verifies that the example pipeline builds, simple requests pass
//! through, and the multi-turn compaction flow works end-to-end.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::TcpStream,
    sync::{Arc, Mutex},
    time::Duration,
};

use praxis_test_utils::{
    Backend, TempSqlite, bind_unique_port, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response for the first turn — stored by response_store.
/// The output text is long enough to exceed a low compact_threshold.
const FIRST_RESPONSE_JSON: &str = r#"{"id":"resp_compact","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Explain TCP vs UDP","output":[{"type":"message","content":[{"type":"output_text","text":"TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability."}]}]}"#;

/// Chat Completions response used for the summarization callout.
const CHAT_COMPLETIONS_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Summary of the conversation."},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60}}"#;

/// Responses API response returned for the main inference call.
const INFERENCE_RESPONSE: &str = r#"{"id":"resp_inf","created_at":2000,"model":"gpt-4.1","object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"QUIC is faster."}]}]}"#;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the compact example config, replacing the SQLite URL and
/// patching listener/backend addresses.
fn load_compact_config(yaml: &str, db_url: &str, proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    let replaced = yaml
        .replace("sqlite://responses.db?mode=rwc", db_url)
        .replace("localhost:11434", &format!("127.0.0.1:{backend_port}"));
    let patched = patch_yaml(
        &replaced,
        proxy_port,
        &HashMap::from([("127.0.0.1:11434", backend_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

/// Start a sequenced backend that:
/// - Returns `first_response` for the first request (summarization callout)
/// - Returns `second_response` for the second request (inference callout)
///
/// The body of the second request is captured and available via the returned
/// `Arc<Mutex<Option<String>>>`.
fn start_sequenced_backend(
    first_response: &'static str,
    second_response: &'static str,
) -> (u16, Arc<Mutex<Option<String>>>) {
    let (listener, port) = bind_unique_port();
    let captured = Arc::new(Mutex::new(None::<String>));
    let capture_slot = Arc::clone(&captured);

    std::thread::spawn(move || {
        let mut call = 0_u32;
        for stream in listener.incoming().flatten() {
            call += 1;
            let body = if call == 1 { first_response } else { second_response };
            let slot = Arc::clone(&capture_slot);
            let body = body.to_owned();
            std::thread::spawn(move || {
                handle_sequenced_request(stream, &body, call, &slot);
            });
        }
    });

    (port, captured)
}

fn handle_sequenced_request(
    mut stream: TcpStream,
    response_body: &str,
    call: u32,
    captured: &Arc<Mutex<Option<String>>>,
) {
    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        let raw = String::from_utf8_lossy(&data);
        if let Some(header_end) = raw.find("\r\n\r\n") {
            let content_length: usize = raw
                .get(..header_end)
                .unwrap_or("")
                .lines()
                .find(|l| l.to_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split_once(':').map(|(_, v)| v.trim().parse().ok()))
                .flatten()
                .unwrap_or(0);
            if data.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }
    let raw = String::from_utf8_lossy(&data);
    let request_body = raw.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
    if call == 2 && !request_body.is_empty() {
        *captured.lock().unwrap() = Some(request_body);
    }
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    );
    drop(stream.write_all(response.as_bytes()));
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn compact_passthrough() {
    let backend_guard = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let config = load_compact_config(&yaml, "sqlite::memory:", proxy_port, backend_guard.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "request without context_management should pass through"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_multi_turn_compaction() {
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("compact");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");

    let config1 = load_compact_config(&yaml, db.url(), proxy_port, backend1.port());
    let proxy1 = start_proxy(&config1);

    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should succeed");

    drop(backend1);
    drop(proxy1);

    let backend2 = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();

    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":50}]}"#,
        ),
    );
    let status2 = parse_status(&raw2);
    assert_eq!(
        status2, 200,
        "second request with compaction should succeed (callout + pipeline completed)"
    );

    drop(proxy2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_verifies_summarization_call_and_compacted_state() {
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("compact_verify");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");

    // First turn: store a response.
    let config1 = load_compact_config(&yaml, db.url(), proxy_port, backend1.port());
    let proxy1 = start_proxy(&config1);
    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should succeed");
    drop(backend1);
    drop(proxy1);

    // Second turn: sequenced backend — first call is summarization, second is inference.
    let (backend_port, captured_inference_body) =
        start_sequenced_backend(CHAT_COMPLETIONS_RESPONSE, INFERENCE_RESPONSE);

    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend_port);
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":50}]}"#,
        ),
    );
    assert_eq!(parse_status(&raw2), 200, "second request should succeed");
    drop(proxy2);

    // The inference request body must contain the compacted state.
    let inference_body = captured_inference_body
        .lock()
        .unwrap()
        .clone()
        .expect("inference request body should have been captured");
    let inference_json: serde_json::Value =
        serde_json::from_str(&inference_body).expect("inference body should be valid JSON");

    // The input should have exactly 2 items: the compacted summary + the current input.
    let input = inference_json["input"].as_array().expect("input should be an array");
    assert_eq!(
        input.len(),
        2,
        "compacted input should have exactly 2 items: summary + current input"
    );

    // The first item should be the translated compaction summary (assistant message).
    assert_eq!(
        input[0]["role"], "assistant",
        "first item should be the compaction summary as an assistant message"
    );
    let content = input[0]["content"]
        .as_str()
        .expect("summary content should be a string");
    assert!(
        content.contains("Previous conversation summary"),
        "summary should be labeled"
    );

    // The second item should be the current user input.
    let second = input[1]["content"]
        .as_str()
        .unwrap_or_else(|| input[1]["content"][0]["text"].as_str().unwrap_or(""));
    assert!(
        second.contains("Compare with QUIC"),
        "second item should be the current user input"
    );
}

#[test]
fn compact_direct_input_skips_reactive_compaction() {
    let backend = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let config = load_compact_config(&yaml, "sqlite::memory:", proxy_port, backend.port());
    let proxy = start_proxy(&config);

    // Send a full conversation in `input` with context_management but
    // no previous_response_id. Reactive compaction is skipped because
    // state.input == state.messages — there is no separable "current
    // turn" to preserve after summarization.
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":[{"role":"user","content":"Explain TCP vs UDP in detail"},{"role":"assistant","content":"TCP is a connection-oriented protocol."},{"role":"user","content":"Compare with QUIC"}],"context_management":[{"type":"compaction","compact_threshold":50}]}"#,
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "direct input without rehydration should pass through without compaction"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint() {
    // Phase 1: store a response via normal inference.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_explicit");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");

    let config1 = load_compact_config(&yaml, db.url(), proxy_port, backend1.port());
    let proxy1 = start_proxy(&config1);
    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should store response");
    drop(backend1);
    drop(proxy1);

    // Phase 2: POST /v1/responses/compact with the stored response_id.
    // The summarization callout goes to inference_url (same backend address).
    let backend2 = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"response_id":"resp_compact","model":"gpt-4.1"}"#,
        ),
    );
    assert_eq!(parse_status(&raw2), 200, "explicit compact should return 200");

    let body = parse_body(&raw2);
    let resp: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(resp["object"], "response", "should be a response object");
    assert_eq!(resp["status"], "completed");
    assert_eq!(
        resp["previous_response_id"], "resp_compact",
        "should reference the original response"
    );
    let output = resp["output"].as_array().expect("output should be an array");
    assert_eq!(output.len(), 1, "output should have one compaction item");
    assert_eq!(output[0]["type"], "compaction");
    assert!(
        output[0].get("encrypted_content").is_some(),
        "compaction item should have encrypted_content"
    );
    drop(proxy2);
}
