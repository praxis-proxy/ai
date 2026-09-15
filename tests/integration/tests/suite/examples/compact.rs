// SPDX-License-Identifier: Apache-2.0
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
use sqlx::Row as _;

use super::openai_file_resolve::{start_file_url_stub, start_files_api_stub};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response for the first turn - stored by response_store.
/// The output text is long enough to exceed a 1000 token compact_threshold.
const FIRST_RESPONSE_JSON: &str = r#"{"id":"resp_compact","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Explain TCP vs UDP","output":[{"type":"message","content":[{"type":"output_text","text":"TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability. TCP is a connection-oriented protocol that provides reliable, ordered delivery of data. It establishes a connection through a three-way handshake before transmitting data. UDP is a connectionless protocol that sends data without establishing a connection first. TCP guarantees delivery through acknowledgments and retransmissions while UDP does not. TCP is used for applications requiring reliability like web browsing and email while UDP is used for real-time applications like video streaming and gaming where speed matters more than reliability."}]}]}"#;

/// Chat Completions response used for the summarization callout.
///
/// Carries the full `CompletionUsage` shape - including
/// `prompt_tokens_details.cached_tokens` and
/// `completion_tokens_details.reasoning_tokens` - so tests can verify those
/// counts are threaded into the compaction `ResponseUsage`.
const CHAT_COMPLETIONS_RESPONSE: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Summary of the conversation."},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60,"prompt_tokens_details":{"cached_tokens":5},"completion_tokens_details":{"reasoning_tokens":3}}}"#;

/// Responses API response returned for the main inference call.
const INFERENCE_RESPONSE: &str = r#"{"id":"resp_inf","created_at":2000,"model":"gpt-4.1","object":"response","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"QUIC is faster."}]}]}"#;

/// Summarization callout response that omits the `usage` object, forcing the
/// compaction usage to fall back to a tiktoken estimate.
const CHAT_COMPLETIONS_NO_USAGE: &str = r#"{"id":"chatcmpl-2","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Summary of the conversation."},"finish_reason":"stop"}]}"#;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the compact example config, replacing the SQLite URL and
/// patching listener/backend addresses.
fn load_compact_config(yaml: &str, db_url: &str, proxy_port: u16, backend_port: u16) -> praxis_core::config::Config {
    load_compact_config_with_files(yaml, db_url, proxy_port, backend_port, None)
}

/// Load the compact example, optionally pointing `files_api_url` at a stub.
fn load_compact_config_with_files(
    yaml: &str,
    db_url: &str,
    proxy_port: u16,
    backend_port: u16,
    files_api_port: Option<u16>,
) -> praxis_core::config::Config {
    let mut replaced = yaml
        .replace("sqlite://responses.db?mode=rwc", db_url)
        .replace("localhost:11434", &format!("127.0.0.1:{backend_port}"));
    if let Some(files_port) = files_api_port {
        replaced = replaced.replace("127.0.0.1:9999", &format!("127.0.0.1:{files_port}"));
    }
    let patched = patch_yaml(
        &replaced,
        proxy_port,
        &HashMap::from([("127.0.0.1:11434", backend_port)]),
    );
    praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse")
}

/// Drop `openai_doc_extract` so a test can assert the resolve-only shape.
fn without_doc_extract(yaml: &str) -> String {
    yaml.replace(
        "      - filter: openai_doc_extract\n        allow_pre_security_callout: true\n        on_unsupported: continue\n\n",
        "",
    )
}

/// Allow the test file-url stub origin so loopback fetches are not SSRF-blocked.
fn with_allowed_file_url_origin(yaml: &str, origin: &str) -> String {
    yaml.replace(
        "        file_url: resolve\n",
        &format!("        file_url: resolve\n        allowed_file_url_origins:\n          - \"{origin}\"\n"),
    )
}

/// True when any object in `value` still carries `key`.
fn json_contains_key(value: &serde_json::Value, key: &str) -> bool {
    match value {
        serde_json::Value::Object(map) => map.contains_key(key) || map.values().any(|v| json_contains_key(v, key)),
        serde_json::Value::Array(items) => items.iter().any(|v| json_contains_key(v, key)),
        _ => false,
    }
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

/// Assert a `usage` object carries every field the OpenAI `ResponseUsage`
/// schema marks as required (including the nested detail objects).
fn assert_response_usage_contract(usage: &serde_json::Value) {
    assert!(usage["input_tokens"].is_u64(), "usage.input_tokens required: {usage}");
    assert!(usage["output_tokens"].is_u64(), "usage.output_tokens required: {usage}");
    assert!(usage["total_tokens"].is_u64(), "usage.total_tokens required: {usage}");
    let input_details = &usage["input_tokens_details"];
    assert!(
        input_details["cached_tokens"].is_u64(),
        "input_tokens_details.cached_tokens required: {usage}"
    );
    assert!(
        input_details["cache_write_tokens"].is_u64(),
        "input_tokens_details.cache_write_tokens required: {usage}"
    );
    assert!(
        usage["output_tokens_details"]["reasoning_tokens"].is_u64(),
        "output_tokens_details.reasoning_tokens required: {usage}"
    );
}

/// Assert a compaction `output` item carries every field the OpenAI
/// `CompactionBody` schema marks as required.
fn assert_compaction_item_contract(item: &serde_json::Value) {
    assert_eq!(item["type"], "compaction", "compaction item type: {item}");
    assert!(
        item["id"].as_str().is_some_and(|id| !id.is_empty()),
        "compaction item requires a non-empty id: {item}"
    );
    assert!(
        item["encrypted_content"].as_str().is_some_and(|c| !c.is_empty()),
        "compaction item requires encrypted_content: {item}"
    );
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
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":1000}]}"#,
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

    // Second turn: sequenced backend - first call is summarization, second is inference.
    let (backend_port, captured_inference_body) =
        start_sequenced_backend(CHAT_COMPLETIONS_RESPONSE, INFERENCE_RESPONSE);

    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend_port);
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":1000}]}"#,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_preserves_resolved_file_url_as_file_data() {
    let file_url_port = start_file_url_stub();
    let origin = format!("http://127.0.0.1:{file_url_port}");
    let yaml = without_doc_extract(&with_allowed_file_url_origin(
        &std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
            .expect("example config should exist"),
        &origin,
    ));
    let inference =
        compact_follow_up_and_capture(&yaml, "compact_file_url", None, &file_url_follow_up_body(file_url_port));

    assert_no_unresolved_file_fields(&inference);
    let current = &inference["input"][1];
    assert_eq!(
        current["content"][0]["type"], "input_file",
        "resolve-only pipeline should keep input_file after compaction"
    );
    assert!(
        current["content"][0]
            .get("file_data")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "resolved file_data must reach the inference backend: {current}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_preserves_extracted_file_id_as_input_text() {
    let files_api_port = start_files_api_stub();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let inference = compact_follow_up_and_capture(&yaml, "compact_file_id", Some(files_api_port), FILE_ID_FOLLOW_UP);

    assert_no_unresolved_file_fields(&inference);
    let current = &inference["input"][1];
    assert_eq!(
        current["content"][0]["type"], "input_text",
        "doc_extract rewrite must survive compaction: {current}"
    );
    let text = current["content"][0]["text"]
        .as_str()
        .expect("extracted input_text should have a text field");
    assert!(
        text.contains("Hello, world!"),
        "extracted text should include file content: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_example_extracts_file_url_and_does_not_restore_it() {
    let file_url_port = start_file_url_stub();
    let origin = format!("http://127.0.0.1:{file_url_port}");
    let yaml = with_allowed_file_url_origin(
        &std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
            .expect("example config should exist"),
        &origin,
    );
    let inference = compact_follow_up_and_capture(
        &yaml,
        "compact_example_file_url",
        None,
        &file_url_follow_up_body(file_url_port),
    );

    assert_no_unresolved_file_fields(&inference);
    let current = &inference["input"][1];
    assert_eq!(
        current["content"][0]["type"], "input_text",
        "example pipeline should extract the fetched file_url: {current}"
    );
    let text = current["content"][0]["text"]
        .as_str()
        .expect("extracted input_text should have a text field");
    assert!(
        text.contains("Hello, world!"),
        "extracted text should include file content: {text}"
    );
}

const FILE_ID_FOLLOW_UP: &str = r#"{"model":"gpt-4.1","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":1000}],"input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"test-file-123"}]}]}"#;

fn file_url_follow_up_body(file_url_port: u16) -> String {
    format!(
        r#"{{"model":"gpt-4.1","previous_response_id":"resp_compact","context_management":[{{"type":"compaction","compact_threshold":1000}}],"input":[{{"type":"message","role":"user","content":[{{"type":"input_file","file_url":"http://127.0.0.1:{file_url_port}/document.txt"}}]}}]}}"#
    )
}

fn assert_no_unresolved_file_fields(inference: &serde_json::Value) {
    let input = inference["input"]
        .as_array()
        .expect("inference input should be an array");
    assert_eq!(
        input.len(),
        2,
        "compacted input should have exactly 2 items: summary + current input"
    );
    assert_eq!(
        input[0]["role"], "assistant",
        "first item should be the compaction summary as an assistant message"
    );
    assert!(
        !json_contains_key(inference, "file_url"),
        "compacted outbound body must not restore file_url: {inference}"
    );
    assert!(
        !json_contains_key(inference, "file_id"),
        "compacted outbound body must not restore file_id: {inference}"
    );
}

/// Store the first compact turn, then send `follow_up` against a sequenced
/// backend and return the captured inference request JSON.
fn compact_follow_up_and_capture(
    yaml: &str,
    db_name: &str,
    files_api_port: Option<u16>,
    follow_up: &str,
) -> serde_json::Value {
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new(db_name);

    let config1 = load_compact_config_with_files(yaml, db.url(), proxy_port, backend1.port(), files_api_port);
    let proxy1 = start_proxy(&config1);
    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should succeed");
    drop(backend1);
    drop(proxy1);

    let (backend_port, captured_inference_body) =
        start_sequenced_backend(CHAT_COMPLETIONS_RESPONSE, INFERENCE_RESPONSE);
    let config2 = load_compact_config_with_files(yaml, db.url(), proxy_port, backend_port, files_api_port);
    let proxy2 = start_proxy(&config2);
    let raw2 = http_send(proxy2.addr(), &json_post("/v1/responses", follow_up));
    assert_eq!(
        parse_status(&raw2),
        200,
        "compaction follow-up should succeed, body: {raw2}"
    );
    drop(proxy2);

    let inference_body = captured_inference_body
        .lock()
        .unwrap()
        .clone()
        .expect("inference request body should have been captured");
    serde_json::from_str(&inference_body).expect("inference body should be valid JSON")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_rejects_invalid_compact_threshold() {
    // Phase 1: store a response so the second turn rehydrates history and
    // the compact filter actually evaluates the compaction config.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_invalid_threshold");
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

    // Phase 2: a rehydrated request with a below-minimum threshold must be
    // rejected before any summarization callout.
    let backend2 = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":50}]}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 400, "threshold below 1000 should return 400");
    assert!(
        raw.contains("invalid_request_error"),
        "response should be invalid_request_error: {raw}"
    );
    assert!(
        raw.contains("at least 1000"),
        "response should explain threshold requirement: {raw}"
    );
    drop(proxy2);
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

    // Send a full conversation in `input` with a valid compaction config
    // whose token count exceeds the threshold, but no previous_response_id.
    // Reactive compaction is still skipped because state.input ==
    // state.messages - there is no separable "current turn" to preserve
    // after summarization - so the request passes through untouched.
    let padding = "context padding ".repeat(700);
    let body = format!(
        r#"{{"model":"gpt-4.1","input":[{{"role":"user","content":"Explain TCP vs UDP in detail"}},{{"role":"assistant","content":"TCP is a connection-oriented protocol. {padding}"}},{{"role":"user","content":"Compare with QUIC"}}],"context_management":[{{"type":"compaction","compact_threshold":1000}}]}}"#
    );
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body));
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
            r#"{"model":"gpt-4.1","previous_response_id":"resp_compact"}"#,
        ),
    );
    assert_eq!(parse_status(&raw2), 200, "explicit compact should return 200");

    let body = parse_body(&raw2);
    let resp: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(
        resp["object"], "response.compaction",
        "should be a response.compaction object per the OpenAI contract"
    );
    // CompactResource required fields: id, object, output, created_at, usage.
    assert!(resp["id"].is_string(), "compaction response must have an id");
    assert!(
        resp["created_at"].is_number(),
        "compaction response must have created_at"
    );
    let output = resp["output"].as_array().expect("output should be an array");
    assert_eq!(output.len(), 1, "output should have one compaction item");
    assert_compaction_item_contract(&output[0]);

    let usage = &resp["usage"];
    assert_response_usage_contract(usage);
    // Usage is threaded through from the summarization callout, not re-estimated.
    // CHAT_COMPLETIONS_RESPONSE reports prompt=50, completion=10, total=60 with
    // cached_tokens=5 and reasoning_tokens=3 in its detail objects.
    assert_eq!(usage["input_tokens"], 50, "input_tokens must reflect callout usage");
    assert_eq!(usage["output_tokens"], 10, "output_tokens must reflect callout usage");
    assert_eq!(usage["total_tokens"], 60, "total_tokens must reflect callout usage");
    assert_eq!(
        usage["input_tokens_details"]["cached_tokens"], 5,
        "cached_tokens must be threaded from the callout's prompt_tokens_details"
    );
    assert_eq!(
        usage["output_tokens_details"]["reasoning_tokens"], 3,
        "reasoning_tokens must be threaded from the callout's completion_tokens_details"
    );
    drop(proxy2);

    // The explicit endpoint persists the compaction record. Read it back and
    // confirm the stored row matches the returned response and is itself
    // contract-shaped.
    let returned_id = resp["id"].as_str().expect("response id should be a string");
    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let row = sqlx::query("SELECT tenant_id, model, response_object, messages FROM openai_responses WHERE id = ?")
        .bind(returned_id)
        .fetch_one(&pool)
        .await
        .expect("compaction record should be persisted");
    pool.close().await;

    let tenant_id: String = row.get("tenant_id");
    let model: String = row.get("model");
    assert_eq!(tenant_id, "default", "compaction record should use the default tenant");
    assert_eq!(model, "gpt-4.1", "compaction record should persist the request model");

    let stored_object: serde_json::Value =
        serde_json::from_str(&row.get::<String, _>("response_object")).expect("response_object should be valid JSON");
    assert_eq!(
        stored_object["object"], "response.compaction",
        "stored response_object should be a response.compaction"
    );
    assert_eq!(
        stored_object["id"], returned_id,
        "stored id should match the returned id"
    );
    assert_response_usage_contract(&stored_object["usage"]);

    let stored_messages: serde_json::Value =
        serde_json::from_str(&row.get::<String, _>("messages")).expect("messages should be valid JSON");
    let items = stored_messages.as_array().expect("messages should be an array");
    assert_eq!(
        items.len(),
        1,
        "persisted messages should hold the single compaction item"
    );
    assert_compaction_item_contract(&items[0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_response_is_valid_follow_up_target() {
    // Regression: the compaction response must be usable as a
    // `previous_response_id`. `rehydrate::validate_response_status` rejects any
    // stored record whose `status` is not "completed" (missing status reads as
    // 'unknown'), so the compaction object must persist `status: "completed"`.
    // Otherwise the id we hand back cannot actually be continued from, breaking
    // the follow-up continuity the endpoint advertises.

    // Phase 1: store a response via normal inference.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_follow_up");
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

    // Phase 2: explicitly compact the stored response and capture the returned id.
    let backend2 = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);
    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","previous_response_id":"resp_compact"}"#,
        ),
    );
    assert_eq!(parse_status(&raw2), 200, "explicit compact should return 200");
    let compaction: serde_json::Value =
        serde_json::from_str(&parse_body(&raw2)).expect("response should be valid JSON");
    assert_eq!(
        compaction["status"], "completed",
        "compaction response must carry status 'completed' to be a valid follow-up target"
    );
    let compaction_id = compaction["id"]
        .as_str()
        .expect("compaction response should have an id")
        .to_owned();
    drop(backend2);
    drop(proxy2);

    // Phase 3: send a follow-up request referencing the compaction id. Rehydrate
    // must accept the record (status == "completed") and continue from it. Before
    // the status fix this failed with 400 "cannot continue from response with
    // status 'unknown'".
    let backend3 = Backend::fixed(INFERENCE_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config3 = load_compact_config(&yaml, db.url(), proxy_port, backend3.port());
    let proxy3 = start_proxy(&config3);
    let follow_up =
        format!(r#"{{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"{compaction_id}"}}"#);
    let raw3 = http_send(proxy3.addr(), &json_post("/v1/responses", &follow_up));
    assert_eq!(
        parse_status(&raw3),
        200,
        "follow-up referencing the compaction id should rehydrate and succeed, body: {raw3}"
    );
    drop(proxy3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_inline_input() {
    // A contract-conforming `{model, input}` request (no previous_response_id)
    // must be accepted and compact the inline conversation directly.
    let backend = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let config = load_compact_config(&yaml, "sqlite::memory:", proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","input":[{"role":"user","content":"Explain TCP vs UDP"},{"role":"assistant","content":"TCP is reliable, UDP is not."}]}"#,
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "standard {{model, input}} compact request should be accepted"
    );

    let body = parse_body(&raw);
    let resp: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(resp["object"], "response.compaction");
    let output = resp["output"].as_array().expect("output should be an array");
    assert_eq!(output.len(), 1, "output should have one compaction item");
    assert_compaction_item_contract(&output[0]);
    assert_response_usage_contract(&resp["usage"]);
    drop(proxy);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_unknown_previous_response_id() {
    // Referencing a response that was never stored must fail before any
    // summarization callout, with a not-found error. In the recommended
    // pipeline `rehydrate` runs ahead of compact and rejects the unknown
    // `previous_response_id` with a 400 "not found", so that is what the
    // client sees end-to-end (compact's own 404 arm is only reachable when
    // rehydrate is absent from the chain).
    let backend = Backend::fixed(CHAT_COMPLETIONS_RESPONSE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    let config = load_compact_config(&yaml, "sqlite::memory:", proxy_port, backend.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","previous_response_id":"resp_never_stored"}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "unknown previous_response_id should return 400"
    );
    assert!(
        raw.contains("not found"),
        "response should explain the id was not found: {raw}"
    );
    drop(proxy);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_fail_closed_on_callout_error() {
    // Phase 1: store a response to reference.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_fail_closed");
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

    // Phase 2: the summarization callout fails. The example config uses the
    // default on_failure=closed, so the request must be rejected with 502.
    let backend2 = Backend::status(500, r#"{"error":{"message":"backend exploded"}}"#)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","previous_response_id":"resp_compact"}"#,
        ),
    );
    assert_eq!(
        parse_status(&raw),
        502,
        "failed callout under fail-closed should return 502"
    );
    assert!(
        raw.contains("summarization callout rejected"),
        "response should explain the callout was rejected: {raw}"
    );
    drop(proxy2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_fail_open_returns_schema_valid_empty_output() {
    // Phase 1: store a response to reference.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_fail_open");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/compact.yaml"))
        .expect("example config should exist");
    // Opt the compact filter into fail-open so a failed callout yields a no-op
    // compaction response rather than a 502.
    let yaml = yaml.replace(
        "        default_model: llama3.2:1b\n",
        "        default_model: llama3.2:1b\n        on_failure: open\n",
    );

    let config1 = load_compact_config(&yaml, db.url(), proxy_port, backend1.port());
    let proxy1 = start_proxy(&config1);
    let raw1 = http_send(
        proxy1.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Explain TCP vs UDP"}"#),
    );
    assert_eq!(parse_status(&raw1), 200, "first request should store response");
    drop(backend1);
    drop(proxy1);

    // Phase 2: the summarization callout fails. Under on_failure=open the
    // endpoint must return a schema-valid response.compaction with an EMPTY
    // output array (no compaction item = the no-op signal), never the raw
    // `{role, content}` messages, which would lack the required output-item
    // fields and break typed OpenAI clients.
    let backend2 = Backend::status(500, r#"{"error":{"message":"backend exploded"}}"#)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","previous_response_id":"resp_compact"}"#,
        ),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "failed callout under fail-open should return 200: {raw}"
    );

    let body = parse_body(&raw);
    let resp: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(
        resp["object"], "response.compaction",
        "fail-open response should still be a response.compaction object"
    );
    let output = resp["output"].as_array().expect("output should be an array");
    assert!(
        output.is_empty(),
        "fail-open output must be empty (no compaction item), got: {output:?}"
    );
    // No summary was produced, so no output tokens were generated.
    assert_response_usage_contract(&resp["usage"]);
    assert_eq!(
        resp["usage"]["output_tokens"], 0,
        "fail-open usage.output_tokens must be 0 (no summary produced)"
    );
    drop(proxy2);

    // The intact conversation must still be persisted so a follow-up request
    // using this response id rehydrates the full history rather than nothing.
    let returned_id = resp["id"].as_str().expect("response id should be a string");
    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let row = sqlx::query("SELECT messages FROM openai_responses WHERE id = ?")
        .bind(returned_id)
        .fetch_one(&pool)
        .await
        .expect("compaction record should be persisted");
    pool.close().await;

    let stored_messages: serde_json::Value =
        serde_json::from_str(&row.get::<String, _>("messages")).expect("messages should be valid JSON");
    let items = stored_messages.as_array().expect("messages should be an array");
    assert!(
        !items.is_empty(),
        "fail-open must persist the intact conversation, not an empty history"
    );
    assert!(
        items.iter().all(|m| m["type"] != "compaction"),
        "fail-open persisted history must not contain a compaction item: {items:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_explicit_endpoint_estimates_usage_when_callout_omits_it() {
    // Phase 1: store a response to reference.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_usage_fallback");
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

    // Phase 2: the callout succeeds but omits `usage`; compaction usage must
    // fall back to a tiktoken estimate of the conversation and summary.
    let backend2 = Backend::fixed(CHAT_COMPLETIONS_NO_USAGE)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend2.port());
    let proxy2 = start_proxy(&config2);

    let raw = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses/compact",
            r#"{"model":"gpt-4.1","previous_response_id":"resp_compact"}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200, "explicit compact should return 200");

    let resp: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be valid JSON");
    let usage = &resp["usage"];
    // Even on the estimated path the emitted usage must satisfy the contract.
    assert_response_usage_contract(usage);
    let input_tokens = usage["input_tokens"].as_u64().expect("input_tokens should be a number");
    let output_tokens = usage["output_tokens"]
        .as_u64()
        .expect("output_tokens should be a number");
    // The stored conversation is large, so the estimate must be non-trivial and
    // must NOT be the callout values (there were none to thread through).
    assert!(
        input_tokens > 100,
        "estimated input_tokens should reflect the long conversation"
    );
    assert!(output_tokens > 0, "estimated output_tokens should reflect the summary");
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        input_tokens + output_tokens,
        "total should be the sum of estimated input and output tokens"
    );
    drop(proxy2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_reactive_with_store_false_does_not_persist() {
    // Phase 1: store the first turn.
    let backend1 = Backend::fixed(FIRST_RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compact_store_false");
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

    // Phase 2: a reactive compaction turn with store:false. The callout
    // summarizes, the compacted request is forwarded to inference, but nothing
    // from this turn - neither the inference response nor an orphan compaction
    // row - may be persisted.
    let (backend_port, _captured) = start_sequenced_backend(CHAT_COMPLETIONS_RESPONSE, INFERENCE_RESPONSE);
    let config2 = load_compact_config(&yaml, db.url(), proxy_port, backend_port);
    let proxy2 = start_proxy(&config2);

    let raw2 = http_send(
        proxy2.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Compare with QUIC","previous_response_id":"resp_compact","context_management":[{"type":"compaction","compact_threshold":1000}],"store":false}"#,
        ),
    );
    assert_eq!(parse_status(&raw2), 200, "compaction turn should succeed");
    drop(proxy2);

    // The database must still contain exactly the first turn's response.
    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let total: i64 = sqlx::query("SELECT COUNT(*) AS n FROM openai_responses")
        .fetch_one(&pool)
        .await
        .expect("count query should run")
        .get("n");
    let inference_persisted: i64 = sqlx::query("SELECT COUNT(*) AS n FROM openai_responses WHERE id = ?")
        .bind("resp_inf")
        .fetch_one(&pool)
        .await
        .expect("lookup query should run")
        .get("n");
    pool.close().await;

    assert_eq!(
        total, 1,
        "store:false turn must not add rows (only the first turn persists)"
    );
    assert_eq!(
        inference_persisted, 0,
        "the store:false inference response must not be persisted"
    );
}
