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
    ("openai/responses/full-flow-agentic.yaml", 300_000),
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
        // Some examples (e.g. full-flow-agentic) express the pipeline as several
        // named chains composed by the listener, so search every chain's filters
        // for the IRR rather than assuming it lives in the first chain.
        let irr = config["filter_chains"]
            .as_sequence()
            .expect("config should declare filter_chains")
            .iter()
            .flat_map(|chain| chain["filters"].as_sequence().map(Vec::as_slice).unwrap_or(&[]))
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

/// A completed response created with stream=true stores its normalized event
/// log; GET /v1/responses/{id}?stream=true replays those exact events, in
/// original sequence order, ending with the terminal event -- without
/// reconstructing deltas. `starting_after` resumes after a cursor, and
/// `starting_after` without `stream=true` is rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_replays_stored_event_log() {
    let sse_body = [
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream_example\",\"status\":\"in_progress\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hi\"}\n\n",
        &format!("event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE_JSON}}}\n\n"),
        "event: done\ndata: [DONE]\n\n",
    ]
    .concat();
    let backend_guard = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_replay");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let created_raw = http_send(
        proxy.addr(),
        &json_post_with_owner(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );
    assert_eq!(
        parse_status(&created_raw),
        200,
        "streaming create should return 200: {created_raw}"
    );
    let created_events = parse_sse_events(&parse_body(&created_raw));
    assert!(
        created_events.iter().any(|(t, _)| t == "response.completed"),
        "the live stream should terminate with response.completed: {created_events:?}"
    );
    assert!(
        created_events.iter().all(|(_, seq)| seq.is_some()),
        "every persisted event should carry a sequence_number: {created_events:?}"
    );

    // Replay serves the stored events in original order, ending with the terminal.
    let replay_raw = http_send(
        proxy.addr(),
        &get_with_owner("/v1/responses/resp_stream_example?stream=true"),
    );
    assert_eq!(parse_status(&replay_raw), 200, "replay should return 200: {replay_raw}");
    assert_eq!(
        parse_header(&replay_raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "replay should be served as an SSE stream"
    );
    let replay_events = parse_sse_events(&parse_body(&replay_raw));
    assert_eq!(
        replay_events, created_events,
        "replay must serve the stored events in original order, not a reconstruction"
    );
    assert_eq!(
        replay_events.last().map(|(t, _)| t.as_str()),
        Some("response.completed"),
        "replay must end with the terminal event: {replay_events:?}"
    );

    // starting_after resumes strictly after the cursor while keeping the terminal.
    let first_seq = created_events[0].1.expect("first replay event has a sequence number");
    let after_raw = http_send(
        proxy.addr(),
        &get_with_owner(&format!(
            "/v1/responses/resp_stream_example?stream=true&starting_after={first_seq}"
        )),
    );
    assert_eq!(
        parse_status(&after_raw),
        200,
        "cursor replay should return 200: {after_raw}"
    );
    let after_events = parse_sse_events(&parse_body(&after_raw));
    assert!(
        !after_events.is_empty(),
        "replay after the first event must still return events"
    );
    // Assert the exact stored suffix, not just the bounds: a replay that dropped
    // middle events after the cursor (e.g. returned only response.completed) would
    // still satisfy a bounds-only check.
    let expected_after: Vec<_> = created_events
        .iter()
        .filter(|(_, seq)| seq.is_some_and(|s| s > first_seq))
        .cloned()
        .collect();
    assert_eq!(
        after_events, expected_after,
        "starting_after must return exactly the stored suffix after the cursor: {after_events:?}"
    );
    assert_eq!(
        after_events.last().map(|(t, _)| t.as_str()),
        Some("response.completed"),
        "the terminal event must survive the cursor: {after_events:?}"
    );

    // starting_after without stream=true is rejected rather than silently
    // returning the plain JSON record.
    let invalid_raw = http_send(
        proxy.addr(),
        &get_with_owner("/v1/responses/resp_stream_example?starting_after=0"),
    );
    assert_eq!(
        parse_status(&invalid_raw),
        400,
        "starting_after without stream must be rejected: {invalid_raw}"
    );

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

/// A classifier-only `openai_responses_request` pass promotes
/// `openai_responses_request.format=openai_responses` for a replay GET (a GET to
/// /v1/responses/{id} is a Responses endpoint) with `stream=false` (no request
/// body). The store filter must force streaming mode for the replay GET instead
/// of selecting the buffered Responses-format response mode, which the runtime
/// would reject with a 500 against the streaming replay body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_replay_streams_under_classifier_only_pipeline() {
    let sse_body = [
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream_example\",\"status\":\"in_progress\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hi\"}\n\n",
        &format!("event: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{RESPONSE_JSON}}}\n\n"),
        "event: done\ndata: [DONE]\n\n",
    ]
    .concat();
    let backend_guard = Backend::fixed(&sse_body)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_replay_legacy");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    // Add a classifier-only `openai_responses_request` pass alongside the managed
    // owner (both run before the store filter). The owner drives create-time
    // persistence via `responses.*` metadata and `x-praxis-*` headers, while the
    // classifier-only pass (`initialize_state: false`) promotes
    // `openai_responses_request.format=openai_responses` (with `stream=false` for
    // the body-less replay GET) that `is_responses_format` reads -- reproducing
    // the buffered-mode trap the fix targets without disabling persistence.
    let owner_classifier = concat!(
        "      - filter: openai_responses_request\n",
        "        on_invalid: reject\n",
        "        headers:\n",
        "          format: x-praxis-ai-format\n",
        "          model: x-praxis-ai-model\n",
        "          stream: x-praxis-ai-stream\n",
        "          mode: x-praxis-responses-mode\n",
    );
    let facts_classifier = "      - filter: openai_responses_request\n        initialize_state: false\n";
    let yaml = yaml.replace(owner_classifier, &format!("{owner_classifier}\n{facts_classifier}"));
    assert!(
        yaml.contains("initialize_state: false") && yaml.contains("filter: openai_responses_request"),
        "both the managed owner and the classifier-only pass must be present so create persists and the replay GET is trapped"
    );
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let created_raw = http_send(
        proxy.addr(),
        &json_post_with_owner(
            "/v1/responses",
            r#"{"model":"gpt-4.1","input":"Hello streaming","stream":true}"#,
        ),
    );
    assert_eq!(
        parse_status(&created_raw),
        200,
        "streaming create should return 200 under the legacy classifier: {created_raw}"
    );
    let created_events = parse_sse_events(&parse_body(&created_raw));

    // The replay GET must stream (200 SSE), not be rejected 500 by a buffered mode.
    let replay_raw = http_send(
        proxy.addr(),
        &get_with_owner("/v1/responses/resp_stream_example?stream=true"),
    );
    assert_eq!(
        parse_status(&replay_raw),
        200,
        "replay must stream, not 500, under the legacy classifier: {replay_raw}"
    );
    assert_eq!(
        parse_header(&replay_raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "replay should be served as an SSE stream"
    );
    let replay_events = parse_sse_events(&parse_body(&replay_raw));
    assert_eq!(
        replay_events, created_events,
        "replay must serve the stored events in original order"
    );

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

/// A response created without stream=true is persisted via the buffered path
/// with no event log; replaying it must return 400 invalid_request_error --
/// never a 404 and never a stream reconstructed from the stored JSON.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_events_replay_rejects_response_without_event_log() {
    let backend_guard = Backend::fixed(RESPONSE_JSON)
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let (db_url, db_path) = temp_sqlite_url("stream_events_no_log");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/stream-events.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", &db_url),
        proxy_port,
        &HashMap::from([("127.0.0.1:8000", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let created_raw = http_send(
        proxy.addr(),
        &json_post_with_owner("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello","stream":false}"#),
    );
    assert_eq!(
        parse_status(&created_raw),
        200,
        "buffered create should succeed: {created_raw}"
    );

    let replay_raw = http_send(
        proxy.addr(),
        &get_with_owner("/v1/responses/resp_stream_example?stream=true"),
    );
    assert_eq!(
        parse_status(&replay_raw),
        400,
        "a response with no event log is not replayable: {replay_raw}"
    );
    let body = parse_body(&replay_raw);
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("400 body should be JSON");
    assert_eq!(parsed["error"]["type"], "invalid_request_error", "{body}");
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("replay"),
        "the error should explain the response is not replayable: {body}"
    );

    drop(proxy);
    cleanup_sqlite_files(&db_path);
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Parse `(event_type, sequence_number)` pairs from a normalized Responses SSE
/// body, skipping the sentinel `[DONE]` line.
fn parse_sse_events(body: &str) -> Vec<(String, Option<i64>)> {
    let mut events = Vec::new();
    for block in body.split("\n\n") {
        let Some(data) = block.lines().find_map(|line| line.strip_prefix("data:").map(str::trim)) else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        let Some(event_type) = value.get("type").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let seq = value.get("sequence_number").and_then(serde_json::Value::as_i64);
        events.push((event_type.to_owned(), seq));
    }
    events
}

fn get_with_owner(path: &str) -> String {
    format!(
        "GET {path} HTTP/1.1\r\nHost: localhost\r\n\
         x-authenticated-state-owner: {OWNER_ASSERTION}\r\nConnection: close\r\n\r\n"
    )
}

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
