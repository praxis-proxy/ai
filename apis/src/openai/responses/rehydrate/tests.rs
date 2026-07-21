// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::sync::Arc;

use bytes::Bytes;
use praxis_filter::{FilterAction, FilterEntry, FilterPipeline};
use serde_json::json;

use super::*;
use crate::store::{
    ConversationRecord, ResponseRecord, ResponseStore, ResponseStoreRegistry, SqliteResponseStore, StoreError,
};

fn default_filter() -> RehydrateFilter {
    RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: None,
    }
}

// -----------------------------------------------------------------------------
// from_config
// -----------------------------------------------------------------------------

#[test]
fn from_config_succeeds() {
    let filter = RehydrateFilter::from_config(&serde_yaml::Value::Null).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_rehydrate",
        "filter name should match convention"
    );
}

#[test]
fn unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unexpected: true").unwrap();
    let result = RehydrateFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "unknown fields should be rejected by deny_unknown_fields"
    );
}

#[test]
fn body_access_is_read_only() {
    let filter = default_filter();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadOnly,
        "filter should use read-only body access"
    );
}

// -----------------------------------------------------------------------------
// Bypass
// -----------------------------------------------------------------------------

#[tokio::test]
async fn skips_non_post_request() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"input":"test"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "non-POST should continue");
}

#[tokio::test]
async fn skips_non_responses_format() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_chat_completions");
    let mut body = Some(Bytes::from(r#"{"messages":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "non-responses format should release"
    );
}

#[tokio::test]
async fn continues_on_non_end_of_stream() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"input":"partial"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-end-of-stream should continue"
    );
}

#[tokio::test]
async fn skips_cancel_request_without_parsing_empty_body() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses/resp_123/cancel");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::new());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "cancel request should bypass rehydrate even with an empty stream-buffer body"
    );
    assert_eq!(body.as_ref().unwrap().len(), 0, "empty body should stay unchanged");
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "ResponsesState should not be set for cancel requests"
    );
}

// -----------------------------------------------------------------------------
// Passthrough
// -----------------------------------------------------------------------------

#[tokio::test]
async fn passthrough_when_no_previous_response_id() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let original = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release when no previous_response_id"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "body should be unchanged"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "ResponsesState should not be set without previous_response_id"
    );
}

#[tokio::test]
async fn passthrough_when_previous_response_id_is_null() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hello","previous_response_id":null}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release when previous_response_id is null"
    );
}

// -----------------------------------------------------------------------------
// Validation + Metadata
// -----------------------------------------------------------------------------

#[tokio::test]
async fn validates_previous_response_and_sets_metadata() {
    let messages = json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]);
    let store = MockStore::with_completed_response("resp_prev", json!("Hello"), messages);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let original = r#"{"model":"gpt-4.1","input":"What next?","previous_response_id":"resp_prev"}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after validation"
    );

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "body should not be modified"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_response_id"),
        Some("resp_prev"),
        "should set previous_response_id metadata"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
    assert_eq!(state.messages[0]["role"], "user", "first stored message");
    assert_eq!(state.messages[1]["role"], "assistant", "second stored message");
    assert_eq!(
        state.messages[2]["content"], "What next?",
        "current input should be last"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_validates_during_cold_request_body_pre_read() {
    let (db_url, db_path) = temp_sqlite_url("rehydrate_cold_pre_read");
    let seeded_store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None)
        .await
        .unwrap();
    seeded_store
        .upsert_response(&ResponseRecord {
            id: "resp_prev".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_prev",
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": "Hi"}]
            }),
            input: json!("Hello"),
            messages: json!([
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi"}
            ]),
        })
        .await
        .unwrap();
    drop(seeded_store);

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: openai_responses_rehydrate
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
    pipeline.add_pipeline_extension(Box::new(ResponseStoreRegistry::new()));

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    pipeline.prepare_extensions(&mut ctx.extensions);

    drop(pipeline.execute_http_request(&mut ctx).await.unwrap());

    let original = r#"{"model":"gpt-4.1","input":"What next?","previous_response_id":"resp_prev"}"#;
    let mut body = Some(Bytes::from(original));

    let action = pipeline
        .execute_http_request_body(&mut ctx, &mut body, true)
        .await
        .unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "on_request should register store so rehydrate finds it in on_request_body"
    );

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "body should not be modified by rehydrate filter"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_response_id"),
        Some("resp_prev"),
        "previous_response_id should be promoted to metadata"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated in pipeline");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
    assert_eq!(state.messages[0]["role"], "user", "first stored message");
    assert_eq!(state.messages[1]["role"], "assistant", "second stored message");
    assert_eq!(
        state.messages[2]["content"], "What next?",
        "current input should be last"
    );

    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

// -----------------------------------------------------------------------------
// Rejections
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rejects_when_previous_response_not_found() {
    let store = MockStore::empty();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_missing"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject with 400"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_status_not_completed() {
    let store = MockStore::with_status("resp_123", "in_progress");
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject non-completed status"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_status_incomplete() {
    let store = MockStore::with_status("resp_123", "incomplete");
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject incomplete status"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_status_failed() {
    let store = MockStore::with_status("resp_123", "failed");
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject failed status"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_store_unavailable() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 500, "should reject with 500 when store unavailable"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_store_not_registered() {
    let registry = ResponseStoreRegistry::new();

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 500, "should reject with 500 when store not registered"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_invalid_json_body() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from("not json"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject invalid JSON"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_non_string_previous_response_id() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":123}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "should reject non-string previous_response_id"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_store_fetch_fails() {
    let store = MockStore::failing();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_123"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 500, "should reject with 500 on store error"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_when_conversation_store_fails() {
    let store = MockStore::failing();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_abc"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 500, "store failure should reject with 500"),
        other => panic!("expected Reject for conversation store failure, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// MCP Tool Recovery
// -----------------------------------------------------------------------------

#[tokio::test]
async fn extracts_mcp_tools_from_previous_response() {
    let output = json!([
        {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]},
        {
            "id": "mcpl_abc",
            "type": "mcp_list_tools",
            "server_label": "my-server",
            "server_url": "http://10.0.0.5:8080/mcp",
            "tools": [
                {"name": "get_weather", "description": "Get weather", "input_schema": {}},
                {"name": "search", "description": "Search docs", "input_schema": {}}
            ]
        }
    ]);
    let usage = json!({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150});
    let store = MockStore::with_output_and_usage("resp_mcp", output, usage);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"input":"Follow up","previous_response_id":"resp_mcp"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(state.previous_tools.len(), 1, "should store full previous tool listing");
    assert_eq!(
        state.previous_tools[0]["server_label"], "my-server",
        "server label should match"
    );
    assert_eq!(
        state.previous_tools[0]["tools"].as_array().unwrap().len(),
        2,
        "should preserve both tools"
    );
    assert_eq!(
        state.previous_tools[0]["tools"][0]["description"], "Get weather",
        "ResponsesState should preserve full tool definitions"
    );
    assert_eq!(
        state.previous_tools[0]["server_url"], "http://10.0.0.5:8080/mcp",
        "server_url should be preserved for cache matching"
    );
}

#[tokio::test]
async fn no_previous_tools_when_output_has_no_mcp_items() {
    let output = json!([
        {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}
    ]);
    let usage = json!({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15});
    let store = MockStore::with_output_and_usage("resp_no_mcp", output, usage);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_no_mcp"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );
    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert!(
        state.previous_tools.is_empty(),
        "should not store previous tools when no mcp_list_tools items"
    );
}

#[tokio::test]
async fn extracts_mcp_tools_from_multiple_servers() {
    let output = json!([
        {
            "id": "mcpl_1",
            "type": "mcp_list_tools",
            "server_label": "weather-server",
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]
        },
        {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]},
        {
            "id": "mcpl_2",
            "type": "mcp_list_tools",
            "server_label": "search-server",
            "tools": [
                {"name": "search", "description": "d", "input_schema": {}},
                {"name": "index", "description": "d", "input_schema": {}}
            ]
        }
    ]);
    let store = MockStore::with_output_and_usage("resp_multi", output, Value::Null);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_multi"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(state.previous_tools.len(), 2, "should have two server entries");
    assert_eq!(
        state.previous_tools[0]["server_label"], "weather-server",
        "first server label"
    );
    assert_eq!(
        state.previous_tools[1]["server_label"], "search-server",
        "second server label"
    );
    assert_eq!(
        state.previous_tools[1]["tools"].as_array().unwrap().len(),
        2,
        "second server should have two tools"
    );
}

#[tokio::test]
async fn deduplicates_mcp_tools_independent_of_tool_order() {
    let mut records = std::collections::HashMap::new();
    records.insert(
        "resp_dedupe".to_owned(),
        ResponseRecord {
            id: "resp_dedupe".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_dedupe",
                "status": "completed",
                "output": [{
                    "id": "mcpl_output",
                    "type": "mcp_list_tools",
                    "server_label": "shared-server",
                    "tools": [
                        {"name": "beta", "description": "d", "input_schema": {}},
                        {"name": "alpha", "description": "d", "input_schema": {}}
                    ]
                }]
            }),
            input: json!("Hello"),
            messages: json!([
                {
                    "id": "mcpl_history",
                    "type": "mcp_list_tools",
                    "server_label": "shared-server",
                    "tools": [
                        {"name": "alpha", "description": "d", "input_schema": {}},
                        {"name": "beta", "description": "d", "input_schema": {}}
                    ]
                }
            ]),
        },
    );
    let store = MockStore {
        records,
        conversations: std::collections::HashMap::new(),
        should_fail: false,
    };
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Next","previous_response_id":"resp_dedupe"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.previous_tools.len(),
        1,
        "ResponsesState should not retain duplicate MCP listings"
    );
}

#[tokio::test]
async fn extracts_mcp_tools_from_stored_history_when_latest_output_has_none() {
    let mut records = std::collections::HashMap::new();
    records.insert(
        "resp_chain".to_owned(),
        ResponseRecord {
            id: "resp_chain".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_chain",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Latest turn"}]}
                ]
            }),
            input: json!("Second turn"),
            messages: json!([
                {"type": "message", "role": "user", "content": "First turn"},
                {
                    "id": "mcpl_earlier",
                    "type": "mcp_list_tools",
                    "server_label": "weather-server",
                    "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]
                },
                {"type": "message", "role": "assistant", "content": "Latest turn"}
            ]),
        },
    );
    let store = MockStore {
        records,
        conversations: std::collections::HashMap::new(),
        should_fail: false,
    };
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"input":"Third turn","previous_response_id":"resp_chain"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after chained rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(state.previous_tools.len(), 1, "should recover one earlier server entry");
    assert_eq!(
        state.previous_tools[0]["server_label"], "weather-server",
        "server label should match"
    );
    let tools = state.previous_tools[0]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1, "should recover one earlier tool");
    assert_eq!(tools[0]["name"], "get_weather", "tool name should match");
    assert!(
        state
            .messages
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("mcp_list_tools")),
        "stored MCP list items should not be replayed as request input"
    );
    assert_eq!(
        state.persisted_messages.len(),
        4,
        "persistence history should keep stored MCP metadata plus current input"
    );
    assert_eq!(
        state.persisted_messages[1]["type"], "mcp_list_tools",
        "persistence history should preserve stored MCP metadata"
    );
}

#[tokio::test]
async fn large_mcp_tool_listing_is_preserved_in_state() {
    let many_tools: Vec<Value> = (0..30)
        .map(|i| json!({"name": format!("very_long_tool_name_number_{i}"), "description": "d", "input_schema": {}}))
        .collect();
    let output = json!([{
        "id": "mcpl_big",
        "type": "mcp_list_tools",
        "server_label": "big-server",
        "tools": many_tools,
    }]);
    let store = MockStore::with_output_and_usage("resp_big", output, Value::Null);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_big"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );
    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.previous_tools.len(),
        1,
        "large listing should not drop previous tools from state"
    );
    assert_eq!(
        state.previous_tools[0]["tools"].as_array().unwrap().len(),
        30,
        "large listing should preserve every tool definition"
    );
}

// -----------------------------------------------------------------------------
// Usage Extraction
// -----------------------------------------------------------------------------

#[tokio::test]
async fn extracts_usage_from_previous_response() {
    let output = json!([{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]);
    let usage = json!({"input_tokens": 500, "output_tokens": 200, "total_tokens": 700});
    let store = MockStore::with_output_and_usage("resp_usage", output, usage.clone());
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_usage"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_usage_input_tokens"),
        Some("500"),
        "input tokens"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_usage_output_tokens"),
        Some("200"),
        "output tokens"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_usage_total_tokens"),
        Some("700"),
        "total tokens"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.previous_usage.as_ref(),
        Some(&usage),
        "previous usage should be stored in ResponsesState"
    );
}

#[tokio::test]
async fn no_usage_metadata_when_usage_missing() {
    let output = json!([{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]);
    let store = MockStore::with_output_and_usage("resp_no_usage", output, Value::Null);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_no_usage"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );
    assert!(
        ctx.get_metadata("responses.previous_usage_input_tokens").is_none(),
        "should not set input tokens"
    );
    assert!(
        ctx.get_metadata("responses.previous_usage_output_tokens").is_none(),
        "should not set output tokens"
    );
    assert!(
        ctx.get_metadata("responses.previous_usage_total_tokens").is_none(),
        "should not set total tokens"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert!(
        state.previous_usage.is_none(),
        "missing usage should not populate previous_usage"
    );
}

#[tokio::test]
async fn extracts_partial_usage_fields() {
    let output = json!([{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]);
    let usage = json!({"input_tokens": 42});
    let store = MockStore::with_output_and_usage("resp_partial", output, usage);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_partial"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_usage_input_tokens"),
        Some("42"),
        "should set input tokens"
    );
    assert!(
        ctx.get_metadata("responses.previous_usage_output_tokens").is_none(),
        "should not set output tokens when missing"
    );
    assert!(
        ctx.get_metadata("responses.previous_usage_total_tokens").is_none(),
        "should not set total tokens when missing"
    );
}

// -----------------------------------------------------------------------------
// Fallback + MCP
// -----------------------------------------------------------------------------

#[tokio::test]
async fn fallback_reconstruction_replays_only_canonical_input_items() {
    let mut records = std::collections::HashMap::new();
    records.insert(
        "resp_mcp_fb".to_owned(),
        ResponseRecord {
            id: "resp_mcp_fb".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_mcp_fb",
                "status": "completed",
                "output": [
                    {
                        "id": "mcpl_fb",
                        "type": "mcp_list_tools",
                        "server_label": "fb-server",
                        "tools": [{"name": "fb_tool", "description": "d", "input_schema": {}}]
                    },
                    {"id": "ws_fb", "type": "web_search_call", "status": "completed"},
                    {"type": "message", "content": [{"type": "output_text", "text": "result"}]}
                ]
            }),
            input: json!("Hello"),
            messages: json!([]),
        },
    );
    let store = MockStore {
        records,
        conversations: std::collections::HashMap::new(),
        should_fail: false,
    };
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Next","previous_response_id":"resp_mcp_fb"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after fallback rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages.len(),
        3,
        "fallback should reconstruct canonical replay items before current input"
    );
    assert_eq!(state.messages[0]["content"], "Hello", "previous input should be first");
    assert!(
        state.messages.iter().all(|item| {
            !matches!(
                item.get("type").and_then(Value::as_str),
                Some("mcp_list_tools" | "web_search_call")
            )
        }),
        "fallback should not replay hosted output items as request input"
    );
    assert_eq!(
        state.messages[1]["type"], "message",
        "previous message output should follow"
    );
    assert_eq!(state.messages[2]["content"], "Next", "current input should be last");
    assert_eq!(
        state.persisted_messages.len(),
        5,
        "persistence history should keep previous input, all output items, and current input"
    );
    assert_eq!(
        state.persisted_messages[1]["type"], "mcp_list_tools",
        "fallback persistence history should preserve MCP list metadata"
    );
    assert_eq!(state.previous_tools.len(), 1, "fallback should populate previous tools");
}

#[test]
fn replay_canonicalizes_defaulted_item_types_and_excludes_unknown_items() {
    let stored = vec![
        json!({"id":"item-1"}),
        json!({"id":"item-2","type":null}),
        json!({"id":"msg-1","role":"assistant","content":"answer"}),
        json!({"id":"hosted-1","type":"web_search_call","status":"completed"}),
        json!({"type":"unknown","content":"not replayable"}),
    ];

    assert_eq!(
        replay_messages_from_stored(&stored),
        vec![
            json!({"id":"item-1","type":"item_reference"}),
            json!({"id":"item-2","type":"item_reference"}),
            json!({"id":"msg-1","type":"message","role":"assistant","content":"answer"}),
        ]
    );
}

// -----------------------------------------------------------------------------
// History Limits
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rejects_stored_history_exceeding_byte_limit() {
    let user_content = "A".repeat(500);
    let assistant_content = "B".repeat(500);
    let messages = json!([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]);
    let store = MockStore::with_completed_response("resp_big", json!("Hello"), messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: 64,
        max_history_items: None,
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_big"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "should reject with 413 for oversized history");
            let body_bytes = r.body.unwrap();
            let body_str = std::str::from_utf8(&body_bytes).unwrap();
            assert!(
                body_str.contains("byte limit"),
                "rejection body should mention byte limit: {body_str}"
            );
        },
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_stored_history_exceeding_item_limit() {
    let messages = json!([
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Turn 3"}
    ]);
    let store = MockStore::with_completed_response("resp_many", json!("Hello"), messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: Some(3),
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_many"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "should reject with 413 for too many items");
            let body_bytes = r.body.unwrap();
            let body_str = std::str::from_utf8(&body_bytes).unwrap();
            assert!(
                body_str.contains("item limit"),
                "rejection body should mention item limit: {body_str}"
            );
        },
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn allows_history_within_limits() {
    let messages = json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]);
    let store = MockStore::with_completed_response("resp_ok", json!("Hello"), messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: Some(10),
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Next","previous_response_id":"resp_ok"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release when history is within limits"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
}

#[tokio::test]
async fn from_config_with_custom_limits() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_history_bytes: 2097152\nmax_history_items: 2").unwrap();
    let filter = RehydrateFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_responses_rehydrate");

    let messages = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "third"}
    ]);
    let store = MockStore::with_completed_response("resp_cfg", json!("hello"), messages);
    let registry = setup_registry(store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"next","previous_response_id":"resp_cfg"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "configured item limit of 2 should reject 3-item history");
        },
        other => panic!("expected Reject for custom max_history_items=2, got {other:?}"),
    }
}

#[test]
fn from_config_rejects_zero_limits() {
    let yaml = serde_yaml::from_str::<serde_yaml::Value>("max_history_bytes: 0").unwrap();
    assert!(RehydrateFilter::from_config(&yaml).is_err());

    let yaml = serde_yaml::from_str::<serde_yaml::Value>("max_history_items: 0").unwrap();
    assert!(RehydrateFilter::from_config(&yaml).is_err());
}

#[tokio::test]
async fn rejects_fallback_reconstruction_exceeding_byte_limit() {
    let large_input = "X".repeat(500);
    let large_output = json!([
        {"type": "message", "content": [{"type": "output_text", "text": "Y".repeat(500)}]}
    ]);
    let mut records = std::collections::HashMap::new();
    records.insert(
        "resp_fallback".to_owned(),
        ResponseRecord {
            id: "resp_fallback".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_fallback",
                "status": "completed",
                "output": large_output,
            }),
            input: json!(large_input),
            messages: json!([]),
        },
    );
    let store = MockStore {
        records,
        conversations: std::collections::HashMap::new(),
        should_fail: false,
    };
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: 64,
        max_history_items: None,
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_fallback"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(
                r.status, 413,
                "fallback reconstruction exceeding byte limit should reject with 413"
            );
        },
        other => panic!("expected Reject for oversized fallback reconstruction, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_fallback_reconstruction_exceeding_item_limit() {
    let input = json!([
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "b"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "c"}]}
    ]);
    let output = json!([
        {"type": "message", "content": [{"type": "output_text", "text": "r1"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "r2"}]}
    ]);
    let mut records = std::collections::HashMap::new();
    records.insert(
        "resp_items".to_owned(),
        ResponseRecord {
            id: "resp_items".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_items",
                "status": "completed",
                "output": output,
            }),
            input,
            messages: json!([]),
        },
    );
    let store = MockStore {
        records,
        conversations: std::collections::HashMap::new(),
        should_fail: false,
    };
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: Some(3),
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"input":"Hi","previous_response_id":"resp_items"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(
                r.status, 413,
                "fallback reconstruction exceeding item limit should reject with 413"
            );
            let body_bytes = r.body.unwrap();
            let body_str = std::str::from_utf8(&body_bytes).unwrap();
            assert!(
                body_str.contains("item limit"),
                "rejection body should mention item limit: {body_str}"
            );
        },
        other => panic!("expected Reject for oversized fallback reconstruction, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Conversation Rehydration
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rehydrates_from_conversation_string_id() {
    let messages = json!([
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply one"}
    ]);
    let store = MockStore::with_conversation("conv_abc", messages);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"turn two","conversation":"conv_abc"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after conversation rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated from conversation");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
    assert_eq!(state.messages[0]["content"], "turn one", "first stored message");
    assert_eq!(state.messages[1]["content"], "reply one", "second stored message");
    assert_eq!(state.messages[2]["content"], "turn two", "current input should be last");

    assert_eq!(
        state.persisted_messages.len(),
        3,
        "persisted_messages should mirror messages for conversation rehydration"
    );
}

#[tokio::test]
async fn rehydrates_from_conversation_object_form() {
    let messages = json!([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"}
    ]);
    let store = MockStore::with_conversation("conv_obj", messages);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"follow up","conversation":{"id":"conv_obj"}}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after conversation object rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated from conversation object");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
    assert_eq!(state.messages[0]["content"], "hello", "first stored message");
    assert_eq!(
        state.messages[2]["content"], "follow up",
        "current input should be last"
    );
}

#[tokio::test]
async fn previous_response_id_takes_precedence_over_conversation() {
    let response_messages = json!([
        {"role": "user", "content": "from response"},
        {"role": "assistant", "content": "response reply"}
    ]);
    let mut store = MockStore::with_completed_response("resp_win", json!("from response"), response_messages);
    store.conversations.insert(
        "conv_lose".to_owned(),
        ConversationRecord {
            conversation_id: "conv_lose".to_owned(),
            tenant_id: "default".to_owned(),
            created_at: 1000,
            metadata: json!({}),
            messages: json!([
                {"role": "user", "content": "from conversation"},
                {"role": "assistant", "content": "conversation reply"}
            ]),
        },
    );
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"next","previous_response_id":"resp_win","conversation":"conv_lose"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release after rehydration"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages[0]["content"], "from response",
        "previous_response_id should take precedence over conversation"
    );
    assert_eq!(
        ctx.get_metadata("responses.previous_response_id"),
        Some("resp_win"),
        "previous_response_id metadata should be set"
    );
}

#[tokio::test]
async fn rejects_when_conversation_not_found() {
    let store = MockStore::empty();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_missing"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "missing conversation should reject with 400"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn tenant_mismatch_rejects_conversation() {
    let store = MockStore::with_conversation("conv_abc", json!([{"role": "user", "content": "hello"}]));
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata(TENANT_METADATA_KEY, "tenant_b");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_abc"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(
            r.status, 400,
            "conversation stored under different tenant should not be found"
        ),
        other => panic!("expected Reject for tenant mismatch, got {other:?}"),
    }

    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "no state should be produced for cross-tenant lookup"
    );
}

#[tokio::test]
async fn rejects_malformed_conversation_empty_object() {
    let store = MockStore::empty();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"model":"gpt-4.1","input":"Hi","conversation":{}}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "empty object conversation should be rejected"),
        other => panic!("expected Reject for malformed conversation, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_malformed_conversation_numeric() {
    let store = MockStore::empty();
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(r#"{"model":"gpt-4.1","input":"Hi","conversation":42}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 400, "numeric conversation should be rejected"),
        other => panic!("expected Reject for malformed conversation, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_conversation_produces_valid_state() {
    let store = MockStore::with_conversation("conv_empty", json!([]));
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"first message","conversation":"conv_empty"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "empty conversation should release successfully"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated for empty conversation");
    assert_eq!(
        state.messages.len(),
        1,
        "messages should contain only the current input"
    );
    assert_eq!(state.messages[0]["content"], "first message", "current input");
}

#[tokio::test]
async fn conversation_rehydration_requires_store_registry() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_123"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => assert_eq!(r.status, 500, "missing store registry should reject with 500"),
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn conversation_null_messages_treated_as_empty() {
    let store = MockStore::with_conversation("conv_null_msgs", Value::Null);
    let registry = setup_registry(store);

    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"test","conversation":"conv_null_msgs"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "null messages should release successfully"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages.len(),
        1,
        "null conversation messages should contribute zero stored items"
    );
}

// -----------------------------------------------------------------------------
// Conversation History Limits
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rejects_conversation_history_exceeding_byte_limit() {
    let user_content = "A".repeat(500);
    let assistant_content = "B".repeat(500);
    let messages = json!([
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]);
    let store = MockStore::with_conversation("conv_big", messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: 64,
        max_history_items: None,
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_big"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(
                r.status, 413,
                "should reject with 413 for oversized conversation history"
            );
            let body_bytes = r.body.unwrap();
            let body_str = std::str::from_utf8(&body_bytes).unwrap();
            assert!(
                body_str.contains("byte limit"),
                "rejection body should mention byte limit: {body_str}"
            );
        },
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn rejects_conversation_history_exceeding_item_limit() {
    let messages = json!([
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Turn 3"}
    ]);
    let store = MockStore::with_conversation("conv_many", messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: Some(3),
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Hi","conversation":"conv_many"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    match action {
        FilterAction::Reject(r) => {
            assert_eq!(r.status, 413, "should reject with 413 for too many conversation items");
            let body_bytes = r.body.unwrap();
            let body_str = std::str::from_utf8(&body_bytes).unwrap();
            assert!(
                body_str.contains("item limit"),
                "rejection body should mention item limit: {body_str}"
            );
        },
        other => panic!("expected Reject, got {other:?}"),
    }
}

#[tokio::test]
async fn allows_conversation_history_within_limits() {
    let messages = json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]);
    let store = MockStore::with_conversation("conv_ok", messages);
    let registry = setup_registry(store);

    let filter = RehydrateFilter {
        max_history_bytes: default_max_history_bytes(),
        max_history_items: Some(10),
    };
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4.1","input":"Next","conversation":"conv_ok"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release when conversation history is within limits"
    );

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("ResponsesState should be populated");
    assert_eq!(
        state.messages.len(),
        3,
        "messages should contain 2 stored + 1 current input"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

struct MockStore {
    records: std::collections::HashMap<String, ResponseRecord>,
    conversations: std::collections::HashMap<String, ConversationRecord>,
    should_fail: bool,
}

impl MockStore {
    fn with_completed_response(id: &str, input: Value, messages: Value) -> Self {
        let mut records = std::collections::HashMap::new();
        records.insert(
            id.to_owned(),
            ResponseRecord {
                id: id.to_owned(),
                tenant_id: "default".to_owned(),
                created_at: 1000,
                model: "gpt-4.1".to_owned(),
                response_object: json!({
                    "id": id,
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]
                }),
                input,
                messages,
            },
        );
        Self {
            records,
            conversations: std::collections::HashMap::new(),
            should_fail: false,
        }
    }

    fn with_conversation(id: &str, messages: Value) -> Self {
        let mut conversations = std::collections::HashMap::new();
        conversations.insert(
            id.to_owned(),
            ConversationRecord {
                conversation_id: id.to_owned(),
                tenant_id: "default".to_owned(),
                created_at: 1000,
                metadata: json!({}),
                messages,
            },
        );
        Self {
            records: std::collections::HashMap::new(),
            conversations,
            should_fail: false,
        }
    }

    fn with_output_and_usage(id: &str, output: Value, usage: Value) -> Self {
        let mut records = std::collections::HashMap::new();
        let mut response_object = json!({
            "id": id,
            "status": "completed",
            "output": output,
        });
        if !usage.is_null() {
            response_object
                .as_object_mut()
                .expect("response_object should be an object")
                .insert("usage".to_owned(), usage);
        }
        records.insert(
            id.to_owned(),
            ResponseRecord {
                id: id.to_owned(),
                tenant_id: "default".to_owned(),
                created_at: 1000,
                model: "gpt-4.1".to_owned(),
                response_object,
                input: json!("Hello"),
                messages: json!([
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"}
                ]),
            },
        );
        Self {
            records,
            conversations: std::collections::HashMap::new(),
            should_fail: false,
        }
    }

    fn with_status(id: &str, status: &str) -> Self {
        let mut records = std::collections::HashMap::new();
        records.insert(
            id.to_owned(),
            ResponseRecord {
                id: id.to_owned(),
                tenant_id: "default".to_owned(),
                created_at: 1000,
                model: "gpt-4.1".to_owned(),
                response_object: json!({"id": id, "status": status}),
                input: json!("Hello"),
                messages: json!([]),
            },
        );
        Self {
            records,
            conversations: std::collections::HashMap::new(),
            should_fail: false,
        }
    }

    fn empty() -> Self {
        Self {
            records: std::collections::HashMap::new(),
            conversations: std::collections::HashMap::new(),
            should_fail: false,
        }
    }

    fn failing() -> Self {
        Self {
            records: std::collections::HashMap::new(),
            conversations: std::collections::HashMap::new(),
            should_fail: true,
        }
    }
}

#[async_trait::async_trait]
impl ResponseStore for MockStore {
    async fn upsert_response(&self, _record: &ResponseRecord) -> Result<(), StoreError> {
        Ok(())
    }

    async fn get_response(&self, tenant_id: &str, id: &str) -> Result<Option<ResponseRecord>, StoreError> {
        if self.should_fail {
            return Err(StoreError::Unavailable("mock failure".to_owned()));
        }
        Ok(self.records.get(id).filter(|r| r.tenant_id == tenant_id).cloned())
    }

    async fn delete_response(&self, _tenant_id: &str, _id: &str) -> Result<bool, StoreError> {
        Ok(false)
    }

    async fn get_conversation(
        &self,
        tenant_id: &str,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError> {
        if self.should_fail {
            return Err(StoreError::Unavailable("mock failure".to_owned()));
        }
        Ok(self
            .conversations
            .get(conversation_id)
            .filter(|c| c.tenant_id == tenant_id)
            .map(|c| ConversationRecord {
                conversation_id: c.conversation_id.clone(),
                tenant_id: c.tenant_id.clone(),
                created_at: c.created_at,
                metadata: c.metadata.clone(),
                messages: c.messages.clone(),
            }))
    }
}

fn setup_registry(store: MockStore) -> ResponseStoreRegistry {
    let registry = ResponseStoreRegistry::new();
    let name: Arc<str> = Arc::from("default");
    registry.register(&name, Arc::new(store)).unwrap();
    registry
}

fn temp_sqlite_url(test_name: &str) -> (String, std::path::PathBuf) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("praxis_{test_name}_{}_{}.db", std::process::id(), nanos));
    (format!("sqlite://{}?mode=rwc", db_path.display()), db_path)
}

fn cleanup_sqlite_file(db_path: &std::path::Path) {
    drop(std::fs::remove_file(db_path));
    drop(std::fs::remove_file(format!("{}-shm", db_path.display())));
    drop(std::fs::remove_file(format!("{}-wal", db_path.display())));
}
