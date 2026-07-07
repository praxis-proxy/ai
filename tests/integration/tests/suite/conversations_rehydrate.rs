// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for conversation-based rehydration in the
//! Responses API pipeline.
//!
//! Verifies that `POST /v1/responses` with a `conversation` field
//! loads stored conversation items, prepends them to the input,
//! and strips the `conversation` field before forwarding to the
//! backend.

use praxis_core::config::Config;
use praxis_test_utils::{
    Backend, free_port, http_send, json_post, parse_body, parse_status, start_echo_backend, start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rehydrates_from_conversation_string_id() {
    let (proxy, _backend, _db) = start_test_env();

    let conv_id = create_conversation(
        &proxy,
        r#"{"metadata":{},"items":[
            {"id":"item_1","type":"message","role":"user","content":"first turn"},
            {"id":"item_2","type":"message","role":"assistant","content":"first reply"}
        ]}"#,
    );

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "second turn",
        "conversation": conv_id,
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );
    assert_eq!(parse_status(&raw), 200, "responses request should succeed");

    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert!(
        echoed.get("conversation").is_none(),
        "conversation field should be stripped from forwarded body"
    );
    assert_eq!(echoed["model"], "gpt-4.1", "model should be preserved");

    let input = echoed["input"].as_array().expect("input should be an array");
    assert!(
        input.len() >= 3,
        "input should contain stored history + new message, got {input:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rehydrates_from_conversation_object_form() {
    let (proxy, _backend, _db) = start_test_env();

    let conv_id = create_conversation(
        &proxy,
        r#"{"metadata":{},"items":[
            {"id":"item_1","type":"message","role":"user","content":"hello"}
        ]}"#,
    );

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "follow-up",
        "conversation": {"id": conv_id},
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );
    assert_eq!(parse_status(&raw), 200, "responses request should succeed");

    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert!(echoed.get("conversation").is_none(), "conversation should be stripped");

    let input = echoed["input"].as_array().expect("input should be an array");
    assert!(
        input.len() >= 2,
        "input should contain stored item + new message, got {input:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn previous_response_id_takes_precedence_over_conversation() {
    let (proxy, _backend, _db) = start_test_env();

    let conv_id = create_conversation(
        &proxy,
        r#"{"metadata":{},"items":[
            {"id":"item_1","type":"message","role":"user","content":"conv turn"}
        ]}"#,
    );

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "test",
        "previous_response_id": "resp_nonexistent",
        "conversation": conv_id,
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    let status = parse_status(&raw);
    assert!(
        status == 400 || status == 404,
        "should reject with error when previous_response_id is invalid (got {status})"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nonexistent_conversation_returns_error() {
    let (proxy, _backend, _db) = start_test_env();

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "hello",
        "conversation": "conv_does_not_exist",
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );

    let status = parse_status(&raw);
    assert!(
        status == 400 || status == 404,
        "nonexistent conversation should return error (got {status})"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_without_conversation_passes_through() {
    let (proxy, _backend, _db) = start_test_env();

    let body = r#"{"model":"gpt-4.1","input":"no conversation"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));
    assert_eq!(
        parse_status(&raw),
        200,
        "request without conversation should pass through"
    );

    let echoed: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    assert_eq!(echoed["input"], "no conversation", "input should be unchanged");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn append_back_persists_items_after_response() {
    let response_body = serde_json::json!({
        "id": "resp_test123",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "msg_out1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "backend reply"}]
            }
        ]
    });
    let backend = Backend::fixed(&serde_json::to_string(&response_body).unwrap())
        .header("content-type", "application/json")
        .start_with_shutdown();

    let db = TempDb::new();
    let proxy_port = free_port();
    let yaml = pipeline_yaml(proxy_port, backend.port(), &db.url());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);

    let conv_id = create_conversation(
        &proxy,
        r#"{"metadata":{},"items":[
            {"id":"item_seed","type":"message","role":"user","content":"seed turn"}
        ]}"#,
    );

    let body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "follow-up turn",
        "conversation": conv_id,
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&body).unwrap()),
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "responses request should succeed for append-back"
    );

    let mut data_len = 0;
    for _ in 0..40 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let raw = http_send(
            proxy.addr(),
            &format!(
                "GET /v1/conversations/{conv_id}/items?order=asc HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
            ),
        );
        if parse_status(&raw) == 200 {
            let items: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
            if let Some(arr) = items["data"].as_array() {
                data_len = arr.len();
                if data_len >= 2 {
                    break;
                }
            }
        }
    }

    assert!(
        data_len >= 2,
        "conversation should have at least seed + appended items, got {data_len}",
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

struct TempDb {
    path: std::path::PathBuf,
}

impl TempDb {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("praxis_test_{}.db", std::process::id()));
        let unique = path.with_extension(format!("{}.db", free_port()));
        Self { path: unique }
    }

    fn url(&self) -> String {
        format!("sqlite://{}?mode=rwc", self.path.display())
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        drop(std::fs::remove_file(&self.path));
        drop(std::fs::remove_file(self.path.with_extension("db-shm")));
        drop(std::fs::remove_file(self.path.with_extension("db-wal")));
    }
}

fn start_test_env() -> (
    praxis_test_utils::ProxyGuard,
    praxis_test_utils::net::backend::BackendGuard,
    TempDb,
) {
    let echo = start_echo_backend();
    let db = TempDb::new();
    let proxy_port = free_port();
    let yaml = pipeline_yaml(proxy_port, echo.port(), &db.url());
    let config = Config::from_yaml(&yaml).unwrap();
    let proxy = start_proxy(&config);
    (proxy, echo, db)
}

fn pipeline_yaml(proxy_port: u16, backend_port: u16, db_url: &str) -> String {
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_conversations
        backend: sqlite
        database_url: "{db_url}"
        conversations_table: test_conversations
        items_table: test_conversation_items

      - filter: openai_responses_format

      - filter: openai_responses_validate

      - filter: openai_response_store
        backend: sqlite
        database_url: "{db_url}"
        responses_table: test_responses
        conversations_table: test_conversations

      - filter: openai_responses_rehydrate

      - filter: responses_proxy

      - filter: router
        routes:
          - path_prefix: "/"
            cluster: "backend"

      - filter: load_balancer
        clusters:
          - name: "backend"
            endpoints:
              - "127.0.0.1:{backend_port}"
"#
    )
}

fn create_conversation(proxy: &praxis_test_utils::ProxyGuard, body: &str) -> String {
    let raw = http_send(proxy.addr(), &json_post("/v1/conversations", body));
    assert_eq!(parse_status(&raw), 200, "create conversation should succeed");
    let json: serde_json::Value = serde_json::from_str(&parse_body(&raw)).unwrap();
    json["id"].as_str().unwrap().to_owned()
}
