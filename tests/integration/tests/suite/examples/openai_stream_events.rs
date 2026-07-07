// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the streaming Responses API example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, example_config_path, free_port, http_send, json_post, parse_body, parse_header, parse_status, patch_yaml,
    start_proxy,
};
use sqlx::Row as _;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const RESPONSE_JSON: &str = r#"{"id":"resp_stream_example","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Hello streaming","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi from stream"}]}]}"#;

const RESPONSES_TABLE: &str = "openai_responses";

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_accumulates_state_and_persists_response_to_sqlite() {
    let sse_body = format!(
        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE_JSON}}}\n\n\
         event: done\ndata: [DONE]\n\n"
    );
    let backend_guard = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200, "streaming request should return 200");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "streaming response should keep text/event-stream content type"
    );

    let body = parse_body(&raw);
    assert!(
        body.contains("data:"),
        "streaming response body should contain SSE data lines: {body}"
    );
    assert!(
        body.contains("response.completed"),
        "streaming response body should contain response.completed event: {body}"
    );

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT id, tenant_id, created_at, model, input, messages FROM {RESPONSES_TABLE} WHERE id = ?");
    let row: sqlx::sqlite::SqliteRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_stream_example")
        .fetch_one(&pool)
        .await
        .expect("streamed response should be persisted in database");
    pool.close().await;

    let id: String = row.get("id");
    let tenant_id: String = row.get("tenant_id");
    let created_at: i64 = row.get("created_at");
    let model: String = row.get("model");

    assert_eq!(id, "resp_stream_example", "persisted id should match stream");
    assert_eq!(tenant_id, "default", "default tenant should be used");
    assert_eq!(created_at, 1000, "persisted created_at should match stream");
    assert_eq!(model, "gpt-4.1", "persisted model should match stream");

    let input_raw: String = row.get("input");
    let input: serde_json::Value = serde_json::from_str(&input_raw).expect("input column should be valid JSON");
    assert_eq!(
        input,
        serde_json::json!("Hello streaming"),
        "persisted input should match terminal response"
    );

    let messages_raw: String = row.get("messages");
    let messages: serde_json::Value =
        serde_json::from_str(&messages_raw).expect("messages column should be valid JSON");
    let items = messages.as_array().expect("messages should be an array");
    assert_eq!(
        items.len(),
        2,
        "messages should include normalized input plus output for rehydration"
    );
    assert_eq!(
        items[0],
        serde_json::json!({"type": "message", "role": "user", "content": "Hello streaming"}),
        "string input should be normalized as a user message"
    );
    assert_eq!(items[1]["type"], "message", "output item should be preserved");

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_incremental_accumulation_before_terminal() {
    let sse_body = [
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,",
        "\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",",
        "\"name\":\"get_weather\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",",
        "\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\"}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",",
        "\"item_id\":\"fc_1\",\"output_index\":0,\"delta\":\"\\\"NYC\\\"}\"}\n\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",",
        "\"item_id\":\"fc_1\",\"output_index\":0,",
        "\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}\n\n",
        &format!(
            "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE_JSON}}}\n\n"
        ),
        "event: done\ndata: [DONE]\n\n",
    ]
    .concat();

    let backend_guard = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_incr");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 200);

    let body = parse_body(&raw);
    assert!(
        body.contains("function_call_arguments.done"),
        "response should contain function_call_arguments.done event: {body}"
    );
    assert!(
        body.contains("response.completed"),
        "response should contain response.completed event: {body}"
    );

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT id FROM {RESPONSES_TABLE} WHERE id = ?");
    let row = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .bind("resp_stream_example")
        .fetch_one(&pool)
        .await
        .expect("terminal response should still be persisted after incremental events");
    pool.close().await;

    let id: String = row.get("id");
    assert_eq!(id, "resp_stream_example");

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_processes_validate_reformatted_error() {
    let error_body =
        r#"{"error":{"message":"model not found","type":"invalid_request_error","code":"model_not_found"}}"#;
    let backend_guard = Backend::status(404, error_body)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_err");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            r#"{"model":"nonexistent","input":"Hello","stream":true}"#,
        ),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "validate filter reformats 404 to 200 SSE for streaming requests"
    );

    let body = parse_body(&raw);
    assert!(
        body.contains("model not found") || body.contains("model_not_found"),
        "error details should be preserved through validate+stream_events: {body}"
    );

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn temp_sqlite_url(test_name: &str) -> (String, std::path::PathBuf) {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("praxis_integ_{test_name}_{}_{nanos}.db", std::process::id()));
    (format!("sqlite://{}?mode=rwc", db_path.display()), db_path)
}

fn cleanup_sqlite_files(db_path: &std::path::Path) {
    drop(std::fs::remove_file(db_path));
    drop(std::fs::remove_file(format!("{}-shm", db_path.display())));
    drop(std::fs::remove_file(format!("{}-wal", db_path.display())));
}
