// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the streaming Responses API example config.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, example_config_path, free_port, http_send, parse_body, parse_header, parse_status, patch_yaml, start_proxy,
};
use sqlx::Row as _;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

const RESPONSE_JSON: &str = r#"{"id":"resp_stream_example","created_at":1000,"model":"gpt-4.1","object":"response","status":"completed","input":"Hello streaming","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi from stream"}]}]}"#;

const RESPONSES_TABLE: &str = "openai_responses";

const OWNER_ASSERTION: &str = "v1.WyJzdHJlYW0tdGVuYW50IiwidXJuOnByYXhpczp0ZXN0IiwiYWxpY2UiXQ";

const STREAMING_EXAMPLES: [(&str, u64); 5] = [
    ("openai/responses/agentic-loop.yaml", 360_000),
    ("agentic/full-flow-agentic.yaml", 300_000),
    ("openai/responses/irr-terminal-streaming.yaml", 360_000),
    ("openai/responses/responses-to-chat-completions.yaml", 660_000),
    ("openai/responses/stream-events.yaml", 360_000),
];

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn streaming_examples_override_short_irr_default_deadline() {
    for (example, minimum_timeout_ms) in STREAMING_EXAMPLES {
        let yaml = std::fs::read_to_string(example_config_path(example)).expect("example config should exist");
        let config: serde_yaml::Value = serde_yaml::from_str(&yaml).expect("example config should be valid YAML");
        let filters = config["filter_chains"][0]["filters"]
            .as_sequence()
            .expect("filter chain should contain filters");
        let irr = filters
            .iter()
            .find(|filter| filter["filter"].as_str() == Some("iterative_request_router"))
            .unwrap_or_else(|| panic!("{example} should contain an iterative_request_router"));
        let overall_timeout_ms = irr["timeout_ms"]
            .as_u64()
            .unwrap_or_else(|| panic!("{example} streaming IRR should configure an overall timeout"));
        let step_timeout_ms = irr["step_timeout_ms"].as_u64().unwrap_or(overall_timeout_ms);

        assert!(
            overall_timeout_ms >= minimum_timeout_ms,
            "{example} IRR overall timeout ({overall_timeout_ms}ms) must be at least {minimum_timeout_ms}ms"
        );
        assert!(
            step_timeout_ms >= minimum_timeout_ms,
            "{example} IRR step timeout ({step_timeout_ms}ms) must be at least {minimum_timeout_ms}ms"
        );
    }
}

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
        &json_post_with_owner(
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
    assert_eq!(tenant_id, "stream-tenant", "trusted owner tenant should be persisted");
    assert_eq!(created_at, 1000, "persisted created_at should match stream");
    assert_eq!(model, "gpt-4.1", "persisted model should match stream");

    let input_raw: Vec<u8> = row.get("input");
    let input: serde_json::Value = serde_json::from_slice(&input_raw).expect("input column should be valid JSON");
    assert_eq!(
        input,
        serde_json::json!("Hello streaming"),
        "persisted input should match terminal response"
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
        &json_post_with_owner(
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
async fn stream_events_forwards_backend_error_transparently() {
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
        &json_post_with_owner(
            "/v1/responses",
            r#"{"model":"nonexistent","input":"Hello","stream":true}"#,
        ),
    );

    assert_eq!(parse_status(&raw), 404, "backend 404 should be forwarded unchanged");
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("application/json"),
        "backend content-type should be forwarded unchanged"
    );

    let body = parse_body(&raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("backend JSON should be forwarded intact");
    assert_eq!(parsed["error"]["message"], "model not found");
    assert_eq!(parsed["error"]["code"], "model_not_found");

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_idle_backend_is_cut_off_by_read_timeout() {
    use std::time::{Duration, Instant};

    let first_event = "event: response.in_progress\ndata: {\"type\":\"response.in_progress\"}\n\n";
    let backend_guard = Backend::chunked(vec![
        first_event.to_owned(),
        "event: response.completed\ndata: {}\n\n".to_owned(),
    ])
    .header("content-type", "text/event-stream")
    .stall_after_first_chunk(Duration::from_secs(10))
    .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_idle");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    // `openai_stream_events` sits after load_balancer in the example so IRR
    // body hooks see the selected peer. Do not also shrink `read_timeout_ms`;
    // that would hide a missing live-body recap.
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", &db_url).replace(
        "              - filter: openai_stream_events\n",
        "              - filter: openai_stream_events\n                timeout_secs: 1\n",
    );
    let patched = patch_yaml(
        &yaml,
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let started = Instant::now();
    let raw = http_send(
        proxy.addr(),
        &json_post_with_owner(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(4),
        "an idle backend after the first SSE event must be cut off by timeout_secs, not held until the 10s stall; elapsed={elapsed:?}"
    );
    assert_eq!(
        parse_status(&raw),
        200,
        "headers should already be committed as SSE: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("response.in_progress"),
        "the first SSE event should reach the client before the idle abort: {body}"
    );
    assert!(
        !body.contains("response.completed"),
        "the stalled backend must not be able to finish the stream after the idle deadline: {body}"
    );

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_fails_closed_when_accumulation_budget_exceeded() {
    // #556: a backend that streams many individually-valid SSE events whose
    // aggregate accumulated state crosses the budget must fail the stream closed —
    // the client sees a terminal error rather than a success, and nothing is
    // persisted. Exercised end-to-end through the proxy with a tiny
    // `max_accumulated_bytes` so a small body trips the aggregate byte ceiling.
    let mut sse_body = String::new();
    for i in 0..10 {
        sse_body.push_str(&format!(
            "event: response.output_item.added\n\
             data: {{\"type\":\"response.output_item.added\",\"output_index\":{i},\
             \"item\":{{\"type\":\"message\",\"id\":\"item_{i}\",\"content\":[]}}}}\n\n"
        ));
    }
    sse_body.push_str(&format!(
        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE_JSON}}}\n\n"
    ));
    sse_body.push_str("event: done\ndata: [DONE]\n\n");

    let backend_guard = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_overflow");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    // Inject a tiny aggregate byte ceiling so the streamed items trip the budget.
    let yaml = yaml.replace(
        "- filter: openai_stream_events\n",
        "- filter: openai_stream_events\n                max_accumulated_bytes: 512\n",
    );
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post_with_owner(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );

    // Streaming headers are already sent when the budget trips mid-body, so the
    // failure surfaces as an in-band terminal error event, not an HTTP status.
    assert_eq!(
        parse_status(&raw),
        200,
        "streaming request returns 200 before the body trips the budget"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("event: error"),
        "budget overflow must terminate the logical stream with an error event: {body}"
    );
    assert!(
        !body.contains("response.completed"),
        "the poisoned terminal must be suppressed, not forwarded as success: {body}"
    );

    let pool = sqlx::SqlitePool::connect(&db_url)
        .await
        .expect("should connect to test database");
    let sql = format!("SELECT COUNT(*) AS n FROM {RESPONSES_TABLE}");
    let row: sqlx::sqlite::SqliteRow = sqlx::query(sqlx::AssertSqlSafe(sql.as_str()))
        .fetch_one(&pool)
        .await
        .expect("count query should succeed");
    let persisted: i64 = row.get("n");
    pool.close().await;
    assert_eq!(persisted, 0, "a budget-overflow stream must not persist any response");

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

fn json_post_with_owner(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         x-authenticated-state-owner: {OWNER_ASSERTION}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

fn cleanup_sqlite_files(db_path: &std::path::Path) {
    drop(std::fs::remove_file(db_path));
    drop(std::fs::remove_file(format!("{}-shm", db_path.display())));
    drop(std::fs::remove_file(format!("{}-wal", db_path.display())));
}
