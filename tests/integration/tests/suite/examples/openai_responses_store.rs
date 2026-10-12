// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the `openai_responses_store` example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body, parse_header,
    parse_status, patch_yaml, start_proxy,
};
use sqlx::Row as _;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Backend response matching a real Responses API shape with `input`
/// and `output` fields the store extracts for persistence.
const RESPONSE_JSON: &str = r#"{"id":"resp_abc","created_at":1000,"model":"gpt-4.1","object":"response","input":"Hello","output":[{"type":"message","content":[{"type":"output_text","text":"Hi there"}]}]}"#;

/// Terminal Responses object carried by a streamed `response.completed` event.
/// Distinct id/model from [`RESPONSE_JSON`] so the GET assertions prove the
/// accumulated streaming state — not the finite path — produced the record.
const STREAM_RESPONSE_JSON: &str = r#"{"id":"resp_stream_store","created_at":2000,"model":"gpt-4.1-mini","object":"response","status":"completed","input":"Stream hello","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Streamed reply"}]}]}"#;

/// Table name from the example config.
const RESPONSES_TABLE: &str = "openai_responses";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_store_persists_response_to_sqlite() {
    let backend_guard = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("persist");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "Responses API POST should return 200");
    assert_eq!(
        parse_body(&raw),
        RESPONSE_JSON,
        "response body should match the backend's JSON"
    );

    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT id, tenant_id, created_at, model, input, messages FROM {RESPONSES_TABLE} WHERE id = ?");
    let row: sqlx::sqlite::SqliteRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_abc")
        .fetch_one(&pool)
        .await
        .expect("persisted record should exist in database");
    pool.close().await;

    let id: String = row.get("id");
    let tenant_id: String = row.get("tenant_id");
    let created_at: i64 = row.get("created_at");
    let model: String = row.get("model");

    assert_eq!(id, "resp_abc", "persisted id should match response");
    assert_eq!(tenant_id, "default", "default tenant should be used");
    assert_eq!(created_at, 1000, "persisted created_at should match response");
    assert_eq!(model, "gpt-4.1", "persisted model should match response");

    let input_raw: Vec<u8> = row.get("input");
    let input: serde_json::Value = serde_json::from_slice(&input_raw).expect("input column should be valid JSON");
    assert_eq!(
        input,
        serde_json::json!("Hello"),
        "input should match the response's input field"
    );

    let messages_raw: Vec<u8> = row.get("messages");
    let messages: serde_json::Value =
        serde_json::from_slice(&messages_raw).expect("messages column should be valid JSON");
    let items = messages.as_array().expect("messages should be an array");
    assert_eq!(
        items.len(),
        2,
        "messages should include normalized input plus output for rehydration"
    );
    assert_eq!(
        items[0],
        serde_json::json!({"type": "message", "role": "user", "content": "Hello"}),
        "string input should be normalized as a message item"
    );
    assert_eq!(items[1]["type"], "message", "output item should be preserved");

    drop(proxy);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_store_persists_compressed_payload_to_sqlite() {
    let backend_guard = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("compress");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    // Enable zstd compression by appending a compression block to the store filter.
    let with_compression = yaml.replace(
        "        conversations_table: openai_conversations\n",
        "        conversations_table: openai_conversations\n        compression:\n          algorithm: zstd\n          level: 3\n",
    );
    let patched = patch_yaml(
        &with_compression.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );
    assert_eq!(parse_status(&raw), 200, "Responses API POST should return 200");

    // The stored column must be a raw zstd frame (magic 0x28 0xB5 0x2F 0xFD),
    // not plain JSON.
    const ZSTD_MAGIC: &[u8] = &[0x28, 0xB5, 0x2F, 0xFD];
    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT input, messages FROM {RESPONSES_TABLE} WHERE id = ?");
    let row: sqlx::sqlite::SqliteRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_abc")
        .fetch_one(&pool)
        .await
        .expect("persisted record should exist in database");
    pool.close().await;

    let input_raw: Vec<u8> = row.get("input");
    let messages_raw: Vec<u8> = row.get("messages");
    assert!(
        input_raw.starts_with(ZSTD_MAGIC),
        "input column should be a zstd frame, got prefix: {:?}",
        &input_raw[..input_raw.len().min(4)]
    );
    assert!(
        messages_raw.starts_with(ZSTD_MAGIC),
        "messages column should be a zstd frame, got prefix: {:?}",
        &messages_raw[..messages_raw.len().min(4)]
    );

    // The GET endpoint must transparently decompress and return the response.
    let (status, body) = http_get(proxy.addr(), "/v1/responses/resp_abc", None);
    assert_eq!(status, 200, "GET of stored response should return 200");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("body should be valid JSON");
    assert_eq!(parsed["id"], "resp_abc", "decompressed response id should match");
    assert_eq!(parsed["model"], "gpt-4.1", "decompressed model should match");

    drop(proxy);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_store_persists_streamed_post_then_get_returns_completed_json() {
    // A `stream: true` POST is composed into one logical stream inside the IRR,
    // accumulated by `openai_stream_events`, and persisted by the pre-IRR store.
    // The backend delivers the SSE across multiple chunks so the accumulator
    // must merge state across chunk boundaries before the terminal event.
    let chunks = vec![
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream_store\",\"status\":\"in_progress\"}}\n\n".to_owned(),
        format!("event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{STREAM_RESPONSE_JSON}}}\n\n"),
        "event: done\ndata: [DONE]\n\n".to_owned(),
    ];
    let backend_guard = Backend::chunked(chunks)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("stream_persist");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    // The client receives a valid terminal SSE stream.
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1-mini","input":"Stream hello","stream":true}"#,
        ),
    );
    assert_eq!(parse_status(&raw), 200, "streaming POST should return 200");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "streaming response should keep the text/event-stream content type"
    );
    let body = parse_body(&raw);
    let completed = sse_completed_events(&body);
    assert_eq!(
        completed.len(),
        1,
        "client stream must dispatch exactly one blank-line-terminated response.completed event: {body}"
    );
    let completed_response = &completed[0]["response"];
    assert_eq!(
        completed_response["id"], "resp_stream_store",
        "terminal event should carry the streamed response id"
    );
    assert_eq!(
        completed_response["status"], "completed",
        "terminal event should report the completed status"
    );
    assert_eq!(
        completed_response["output"][0]["content"][0]["text"], "Streamed reply",
        "terminal event should carry the accumulated output text"
    );

    // An immediate ordinary GET returns the completed JSON object built from the
    // accumulated stream — not a stream — with the full output.
    let (status, get_body) = http_get(proxy.addr(), "/v1/responses/resp_stream_store", None);
    assert_eq!(status, 200, "GET of the streamed response should return 200");
    let parsed: serde_json::Value = serde_json::from_str(&get_body).expect("GET body should be valid JSON");
    assert_eq!(
        parsed["id"], "resp_stream_store",
        "GET id should match the streamed response"
    );
    assert_eq!(
        parsed["model"], "gpt-4.1-mini",
        "GET model should match the streamed response"
    );
    assert_eq!(parsed["status"], "completed", "GET should report the completed status");
    assert_eq!(
        parsed["output"][0]["content"][0]["text"], "Streamed reply",
        "GET should return the accumulated output text"
    );

    // The persisted row carries the accumulated terminal object.
    let pool = sqlx::SqlitePool::connect(db.url())
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT id, tenant_id, model FROM {RESPONSES_TABLE} WHERE id = ?");
    let row: sqlx::sqlite::SqliteRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_stream_store")
        .fetch_one(&pool)
        .await
        .expect("streamed response should be persisted in database");
    pool.close().await;

    let id: String = row.get("id");
    let tenant_id: String = row.get("tenant_id");
    let model: String = row.get("model");
    assert_eq!(id, "resp_stream_store", "persisted id should match the stream");
    assert_eq!(tenant_id, "default", "single_tenant owner should be persisted");
    assert_eq!(model, "gpt-4.1-mini", "persisted model should match the stream");

    drop(proxy);
}

#[test]
fn response_store_passes_through_non_responses_traffic() {
    let backend_guard = Backend::fixed("fallback")
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", "sqlite::memory:"),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "Chat Completions body should still route through"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_store_delete_returns_200_after_post() {
    let backend_guard = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("delete_200");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let post_raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );
    assert_eq!(parse_status(&post_raw), 200, "POST should succeed");

    let delete_raw = http_send(
        proxy.addr(),
        "DELETE /v1/responses/resp_abc HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(
        parse_status(&delete_raw),
        200,
        "DELETE of existing response should return 200"
    );

    let body = parse_body(&delete_raw);
    let json: serde_json::Value = serde_json::from_str(&body).expect("body should be valid JSON");
    assert_eq!(json["id"], "resp_abc", "response id should match");
    assert_eq!(json["deleted"], true, "deleted flag should be true");

    drop(proxy);
}

#[test]
fn response_store_delete_nonexistent_returns_404() {
    let backend_guard = Backend::fixed("unused")
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", "sqlite::memory:"),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "DELETE /v1/responses/resp_nonexistent HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(
        parse_status(&raw),
        404,
        "DELETE of nonexistent response should return 404"
    );
}

#[test]
fn response_store_delete_has_json_content_type() {
    let backend_guard = Backend::fixed("unused")
        .header("content-type", "text/plain")
        .start_with_shutdown();
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", "sqlite::memory:"),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        "DELETE /v1/responses/resp_any HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );

    let ct = parse_header(&raw, "content-type");
    assert_eq!(
        ct.as_deref(),
        Some("application/json"),
        "DELETE response should have JSON content type"
    );
}

// -----------------------------------------------------------------------------
// GET Retrieval
// -----------------------------------------------------------------------------

#[test]
fn get_missing_response_returns_404() {
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", "sqlite::memory:"),
        proxy_port,
        &HashMap::new(),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);
    let (status, body) = http_get(proxy.addr(), "/v1/responses/resp_nonexistent", None);

    assert_eq!(status, 404, "GET for missing response should return 404");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("body should be valid JSON");
    assert_eq!(
        parsed["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "404 body should use invalid_request_error type"
    );
}

#[test]
fn get_missing_input_items_returns_404() {
    let proxy_port = free_port();

    let yaml = std::fs::read_to_string(example_config_path("openai/responses/response-store.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", "sqlite::memory:"),
        proxy_port,
        &HashMap::new(),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);
    let (status, body) = http_get(proxy.addr(), "/v1/responses/resp_nonexistent/input_items", None);

    assert_eq!(status, 404, "input_items for missing response should return 404");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("body should be valid JSON");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .expect("error message should be a string")
            .contains("resp_nonexistent"),
        "error message should include the missing ID"
    );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Parse `response.completed` events from a fully accumulated SSE body.
///
/// Splits the body into `\n\n`-delimited frames and returns the parsed `data:`
/// JSON of every frame whose `type` is `response.completed`. Only frames
/// terminated by the required blank-line boundary are considered — SSE dispatches
/// an event only at that boundary, so a truncated final frame is never counted —
/// and malformed `data:` JSON is dropped rather than matched by a naive
/// substring.
fn sse_completed_events(body: &str) -> Vec<serde_json::Value> {
    let mut frames: Vec<&str> = body.split("\n\n").collect();
    // Drop the segment after the final boundary: it is either empty (the body
    // ended with the blank line) or an unterminated, undispatched partial frame.
    frames.pop();

    frames
        .iter()
        .filter_map(|frame| {
            let data = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(|value| value.strip_prefix(' ').unwrap_or(value))
                .collect::<Vec<_>>()
                .join("\n");
            serde_json::from_str::<serde_json::Value>(&data).ok()
        })
        .filter(|event| event["type"] == "response.completed")
        .collect()
}
