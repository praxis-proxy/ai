// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use praxis_filter::{FilterAction, FilterEntry, FilterPipeline};
use serde_json::json;

use super::*;
use crate::{
    openai::sse::{SseFrame, SseFrameParser},
    store::{
        ConversationRecord, ResponseRecord, ResponseStore, ResponseStoreRegistry, SqliteResponseStore, StoreError,
    },
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
    ctx.set_metadata("responses.response_id", "resp_current");
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
    assert_eq!(state.response_id.as_deref(), Some("resp_current"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_validates_during_cold_request_body_pre_read() {
    let (db_url, db_path) = temp_sqlite_url("rehydrate_cold_pre_read");
    let seeded_store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None)
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
    ctx.set_metadata("responses.response_id", "resp_conversation");
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
    assert_eq!(state.response_id.as_deref(), Some("resp_conversation"));
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
// Response-side previous_response_id restore (issue #932)
// -----------------------------------------------------------------------------

/// A rehydrated state carrying the caller's `previous_response_id`.
fn rehydrated_state(prev_id: &str) -> ResponsesState {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": "What next?",
        "previous_response_id": prev_id,
    }));
    state.history_rehydrated = true;
    state
}

/// A 200 response with a JSON content type.
fn json_ok_response() -> praxis_filter::Response {
    let mut response = crate::test_utils::make_response();
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
}

#[tokio::test]
async fn restores_previous_response_id_into_response_body() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("74"));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        ctx.response_headers_modified,
        "framing headers change should be signalled"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| !resp.headers.contains_key(http::header::CONTENT_LENGTH)),
        "Content-Length must be dropped so core reframes the rewritten body"
    );

    let mut body = Some(Bytes::from(
        r#"{"id":"resp_new","object":"response","status":"completed","previous_response_id":null}"#,
    ));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );

    let patched: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        patched["previous_response_id"], "resp_prev",
        "caller previous_response_id should be restored"
    );
    assert_eq!(patched["id"], "resp_new", "other response fields should be preserved");
    assert_eq!(patched["status"], "completed", "status should be preserved");
}

#[tokio::test]
async fn does_not_restore_without_rehydration() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    // No ResponsesState in extensions: a plain first-turn request.
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue without arming a restore when no state was rehydrated"
    );
    assert!(
        !ctx.response_headers_modified,
        "headers should be untouched when nothing was rehydrated"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite when nothing was armed"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "body should pass through untouched"
    );
}

#[tokio::test]
async fn does_not_restore_for_conversation_continuation() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    // Conversation-based continuation rehydrates history but carries no
    // previous_response_id, so the null echo is correct and must stay.
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4.1",
        "input": "What next?",
        "conversation": "conv_abc",
    }));
    state.history_rehydrated = true;
    ctx.extensions.insert(state);
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue without arming a restore for a conversation continuation"
    );
    assert!(
        !ctx.response_headers_modified,
        "conversation continuation should not rewrite the response"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite for a conversation continuation"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "conversation continuation should leave previous_response_id null"
    );
}

#[tokio::test]
async fn does_not_restore_for_non_success_status() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    response.status = http::StatusCode::BAD_REQUEST;

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue without arming a restore for a non-success status"
    );
    assert!(!ctx.response_headers_modified, "error responses must not be rewritten");

    let original = r#"{"error":{"message":"bad request"}}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite for an error response"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "error body should pass through untouched"
    );
}

// -----------------------------------------------------------------------------
// Streaming (SSE) previous_response_id restore (issue #932, streaming half)
// -----------------------------------------------------------------------------

/// A 200 response with an SSE content type.
fn sse_ok_response() -> praxis_filter::Response {
    let mut response = crate::test_utils::make_response();
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    response
}

/// Re-parse assembled SSE output bytes into frames for assertions.
fn parse_sse_frames(bytes: &[u8]) -> Vec<SseFrame> {
    SseFrameParser::new(1 << 20)
        .parse_chunk(bytes)
        .expect("assembled SSE output should parse")
}

#[tokio::test]
async fn restores_for_streaming_response() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue and arm the streaming restore"
    );

    let input = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_new\",\"object\":\"response\",\"status\":\"in_progress\",\"previous_response_id\":null}}\n",
        "\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"delta\":\"Hi\"}\n",
        "\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_new\",\"object\":\"response\",\"status\":\"completed\",\"previous_response_id\":null}}\n",
        "\n",
    );
    let mut body = Some(Bytes::from_static(input.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );

    let out = body.expect("streaming body should be rewritten in place");
    let frames = parse_sse_frames(&out);
    assert_eq!(frames.len(), 3, "all three frames should survive reconstruction");

    for (idx, event) in [(0_usize, "response.created"), (2, "response.completed")] {
        assert_eq!(frames[idx].event_type.as_deref(), Some(event));
        let data: Value = serde_json::from_slice(&frames[idx].data).unwrap();
        assert_eq!(
            data["response"]["previous_response_id"], "resp_prev",
            "{event} must carry the restored previous_response_id"
        );
        assert_eq!(
            data["response"]["id"], "resp_new",
            "other response fields must be preserved"
        );
    }

    assert_eq!(
        frames[1].event_type.as_deref(),
        Some("response.output_text.delta"),
        "the delta frame (no top-level response object) passes through untouched"
    );
    let delta: Value = serde_json::from_slice(&frames[1].data).unwrap();
    assert!(
        delta.get("previous_response_id").is_none(),
        "a delta frame must not gain a previous_response_id"
    );
    assert_eq!(delta["delta"], "Hi", "delta payload must be preserved");
}

#[tokio::test]
async fn restores_streaming_across_chunk_boundary() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);
    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");

    // A single lifecycle frame split mid-`data:` across two chunks. The first
    // chunk completes no frame (its bytes stay buffered in the parser); the
    // second chunk completes and rewrites the frame.
    let full = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"sequence_number\":0,\"response\":{\"id\":\"resp_new\",\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    );
    let (head, tail) = full.split_at(40);

    let mut assembled = Vec::new();
    let mut body = Some(Bytes::from_static(head.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert!(
        body.is_none(),
        "an incomplete frame must not emit partial bytes (they stay buffered)"
    );

    let mut body = Some(Bytes::from_static(tail.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    if let Some(chunk) = &body {
        assembled.extend_from_slice(chunk);
    }

    let frames = parse_sse_frames(&assembled);
    assert_eq!(frames.len(), 1, "the frame split across chunks should complete once");
    let data: Value = serde_json::from_slice(&frames[0].data).unwrap();
    assert_eq!(
        data["response"]["previous_response_id"], "resp_prev",
        "a frame split across chunk boundaries must still have its id restored"
    );
}

#[tokio::test]
async fn streaming_delta_frames_pass_through_unchanged() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);
    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");

    let input = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n",
        "\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\" there\"}\n",
        "\n",
    );
    let mut body = Some(Bytes::from_static(input.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(input.as_bytes()),
        "delta-only frames must be reconstructed byte-identically"
    );
}

#[tokio::test]
async fn streaming_preserves_id_retry_and_comment_lines() {
    // Finding 2 regression: frames are spliced, not reconstructed. A comment-only
    // heartbeat and a delta frame pass through byte-for-byte, and a lifecycle frame
    // keeps its `id:`, `retry:`, and comment lines while only its `data:` JSON is
    // rewritten. Nothing outside the payload — including CRLF-adjacent SSE fields —
    // may be dropped.
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);
    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");

    let heartbeat = ": keepalive\n\n";
    let lifecycle_in = concat!(
        "id: 42\n",
        "event: response.created\n",
        "retry: 1500\n",
        "data: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        ": trailing note\n",
        "\n",
    );
    let delta = concat!(
        "event: response.output_text.delta\n",
        "id: 43\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n",
        "\n",
    );
    let input = format!("{heartbeat}{lifecycle_in}{delta}");
    let mut body = Some(Bytes::from(input));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );

    let lifecycle_out = concat!(
        "id: 42\n",
        "event: response.created\n",
        "retry: 1500\n",
        "data: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"previous_response_id\":\"resp_prev\"}}\n",
        ": trailing note\n",
        "\n",
    );
    let expected = format!("{heartbeat}{lifecycle_out}{delta}");
    assert_eq!(
        body.as_deref(),
        Some(expected.as_bytes()),
        "id/retry/comment/heartbeat lines must survive; only the lifecycle data payload is rewritten"
    );
}

#[tokio::test]
async fn does_not_restore_for_encoded_streaming_response() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();
    // A gzipped event stream cannot be parsed frame-by-frame, so it must be
    // declined and passed through verbatim (mirrors the encoded-SSE decline in
    // stream_events; see issue #668).
    response
        .headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "an encoded event stream must be left as an untouched passthrough"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| resp.headers.contains_key(http::header::CONTENT_ENCODING)),
        "Content-Encoding must be preserved so the client can still decode the stream"
    );

    let original = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    );
    let mut body = Some(Bytes::from_static(original.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_bytes()),
        "an encoded stream must pass through byte-for-byte with no restore attempt"
    );
}

#[tokio::test]
async fn does_not_restore_for_non_ok_streaming_response() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();
    response.status = http::StatusCode::INTERNAL_SERVER_ERROR;

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a non-200 event stream must not be rewritten"
    );

    let original = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    );
    let mut body = Some(Bytes::from_static(original.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_bytes()),
        "a non-200 event stream passes through unchanged"
    );
}

#[tokio::test]
async fn declines_streaming_response_with_body_validators() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = sse_ok_response();
    // SSE normally carries no body validators, but if a backend sets one the
    // rewritten frames would no longer match its upstream digest. The header phase
    // cannot recompute or safely strip it (Praxis commits headers before the body
    // is rewritten), so the stream is declined and passed through byte-for-byte with
    // its validators intact — the same fail-safe as the non-streaming path.
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("\"abc123\""));
    response.headers.insert(
        http::header::LAST_MODIFIED,
        http::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
    );
    response.headers.insert(
        "content-md5",
        http::HeaderValue::from_static("Q2hlY2sgSW50ZWdyaXR5IQ=="),
    );
    response.headers.insert(
        "digest",
        http::HeaderValue::from_static("sha-256=X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE="),
    );
    response.headers.insert(
        "content-digest",
        http::HeaderValue::from_static("sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"),
    );
    response.headers.insert(
        "repr-digest",
        http::HeaderValue::from_static("sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"),
    );

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a validator-bearing event stream must be declined as an untouched passthrough"
    );

    let headers = &ctx
        .response_header
        .as_ref()
        .expect("response header should still be present")
        .headers;
    for validator in [
        http::header::ETAG.as_str(),
        http::header::LAST_MODIFIED.as_str(),
        "content-md5",
        "digest",
        "content-digest",
        "repr-digest",
    ] {
        assert!(
            headers.contains_key(validator),
            "body validator {validator} must be preserved on a declined SSE stream"
        );
    }

    let original = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    );
    let mut body = Some(Bytes::from_static(original.as_bytes()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_bytes()),
        "a validator-bearing stream must pass through byte-for-byte with no restore attempt"
    );
}

#[tokio::test]
async fn does_not_rewrite_non_response_json_shape() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue and arm the restore for an eligible 2xx JSON response"
    );

    let original = r#"{"object":"list","data":[]}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite for a non-response JSON shape"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "non-response JSON should not gain a previous_response_id"
    );
}

#[tokio::test]
async fn ignores_non_end_of_stream_chunks() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response must arm the restore so the body phase exercises the end-of-stream guard, not the unarmed path"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite for a non-terminal chunk"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "partial chunk should be left untouched until end-of-stream"
    );
}

#[test]
fn response_body_access_is_read_write() {
    let filter = default_filter();
    assert_eq!(
        filter.response_body_access(),
        BodyAccess::ReadWrite,
        "response phase must be able to rewrite the body"
    );
    assert_eq!(
        filter.response_body_mode(),
        BodyMode::Stream,
        "response defaults to streaming; buffering is selected dynamically"
    );
}

#[tokio::test]
async fn does_not_restore_for_encoded_response() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("64"));
    // A backend that honored the client's Accept-Encoding returns a compressed
    // body. The restore step parses the body as JSON, which a compressed body
    // is not, so the response must be declined and passed through with its
    // framing headers intact — never stripped and shipped as mislabeled
    // identity JSON (issue #932 encoded-response corruption).
    response
        .headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response should continue without arming a restore for an encoded response"
    );
    assert!(
        !ctx.response_headers_modified,
        "an encoded response must be left as an untouched passthrough"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| resp.headers.contains_key(http::header::CONTENT_ENCODING)),
        "Content-Encoding must be preserved so the client can still decode the body"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| resp.headers.contains_key(http::header::CONTENT_LENGTH)),
        "Content-Length must be preserved for an untouched passthrough"
    );

    // Valid JSON so a rewrite WOULD change it: proves the decline, not merely an
    // unparseable body, prevented the restore.
    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue without a rewrite for an encoded response"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "an encoded body must pass through byte-for-byte with no restore attempt"
    );
}

#[tokio::test]
async fn declines_identity_content_encoding_conservatively() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    // `Content-Encoding: identity` means "no coding", but the presence-based
    // guard declines it anyway to match the sibling stream_events filters. The
    // only cost is that the previous_response_id echo is not restored for this
    // (RFC 9110-discouraged) response shape; the body is never corrupted.
    response.headers.insert(
        http::header::CONTENT_ENCODING,
        http::HeaderValue::from_static("identity"),
    );

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a Content-Encoding: identity response is declined by the presence-based guard"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a declined identity response passes through unchanged"
    );
}

#[tokio::test]
async fn does_not_restore_when_content_type_missing() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a response without a Content-Type must not be treated as JSON"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a response without a Content-Type passes through unchanged"
    );
}

#[tokio::test]
async fn does_not_restore_for_non_json_content_type() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();
    response
        .headers
        .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("text/plain"));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a non-JSON content type must not be rewritten"
    );

    let original = "plain text body";
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a non-JSON body passes through unchanged"
    );
}

#[tokio::test]
async fn restores_for_json_content_type_with_charset_parameter() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json; charset=utf-8"),
    );

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        ctx.response_headers_modified,
        "a JSON content type with a charset parameter is still eligible"
    );

    let mut body = Some(Bytes::from(
        r#"{"id":"resp_new","object":"response","previous_response_id":null}"#,
    ));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    let patched: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        patched["previous_response_id"], "resp_prev",
        "charset-parameterized JSON should still have previous_response_id restored"
    );
}

#[tokio::test]
async fn restores_for_uppercase_json_content_type() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("APPLICATION/JSON"),
    );

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        ctx.response_headers_modified,
        "content-type matching must be case-insensitive"
    );

    let mut body = Some(Bytes::from(
        r#"{"id":"resp_new","object":"response","previous_response_id":null}"#,
    ));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    let patched: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        patched["previous_response_id"], "resp_prev",
        "uppercase application/json should still have previous_response_id restored"
    );
}

#[tokio::test]
async fn declines_and_preserves_response_body_validators() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    // Validators and integrity digests describe the exact upstream bytes; they
    // would be invalidated the moment the body is re-serialized with the caller's
    // previous_response_id. The proxy cannot recompute them from the body phase
    // (headers are already committed) so it declines the response entirely at
    // eligibility rather than stripping them and shipping a mismatched body — the
    // response passes through byte-identical with every validator intact.
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("65"));
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("\"abc123\""));
    response.headers.insert(
        http::header::LAST_MODIFIED,
        http::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
    );
    response.headers.insert(
        "content-md5",
        http::HeaderValue::from_static("Q2hlY2sgSW50ZWdyaXR5IQ=="),
    );
    response.headers.insert(
        "digest",
        http::HeaderValue::from_static("sha-256=X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE="),
    );
    response.headers.insert(
        "content-digest",
        http::HeaderValue::from_static("sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"),
    );
    response.headers.insert(
        "repr-digest",
        http::HeaderValue::from_static("sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"),
    );
    response
        .headers
        .insert(http::header::CACHE_CONTROL, http::HeaderValue::from_static("no-store"));
    response
        .headers
        .insert("x-request-id", http::HeaderValue::from_static("req_trace_123"));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a validator-bearing response must be an untouched passthrough"
    );

    let headers = &ctx
        .response_header
        .as_ref()
        .expect("response header should still be present")
        .headers;
    for preserved in [
        http::header::CONTENT_LENGTH.as_str(),
        http::header::ETAG.as_str(),
        http::header::LAST_MODIFIED.as_str(),
        "content-md5",
        "digest",
        "content-digest",
        "repr-digest",
    ] {
        assert!(
            headers.contains_key(preserved),
            "declined response must keep its body validator {preserved} intact"
        );
    }
    assert_eq!(
        headers.get(http::header::CACHE_CONTROL).and_then(|v| v.to_str().ok()),
        Some("no-store"),
        "caching-policy headers must be preserved"
    );
    assert_eq!(
        headers.get(http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()),
        Some("application/json"),
        "content-type must be preserved"
    );
    assert_eq!(
        headers.get("x-request-id").and_then(|v| v.to_str().ok()),
        Some("req_trace_123"),
        "routing/tracing headers must be preserved"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a validator-bearing response passes through byte-identical (id not restored)"
    );
}

/// Each body validator / integrity digest must decline the restore on its own.
/// The aggregate test above stays green as long as *any* validator is still
/// checked, so this per-header sweep guards against a refactor that silently
/// drops a single entry from `describes_exact_upstream_bytes` (which would let a
/// response bearing only that header be rewritten and ship a stale validator —
/// the exact issue #932 regression).
#[tokio::test]
async fn each_body_validator_alone_declines_restore() {
    for (name, value) in [
        ("etag", "\"abc123\""),
        ("last-modified", "Wed, 21 Oct 2026 07:28:00 GMT"),
        ("content-md5", "Q2hlY2sgSW50ZWdyaXR5IQ=="),
        ("digest", "sha-256=X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE="),
        (
            "content-digest",
            "sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:",
        ),
        ("repr-digest", "sha-256=:X48E9qOokqqrvdts8nOJRJN3OWDUoyWxBf7kbu9DBPE=:"),
    ] {
        assert_single_validator_declines(name, value).await;
    }
}

/// Drive `on_response` + `on_response_body` for a `200 OK` JSON response carrying
/// exactly one validator header and assert it is declined: not modified, the
/// validator preserved, and the body passed through byte-identical.
async fn assert_single_validator_declines(name: &'static str, value: &'static str) {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    response.headers.insert(name, http::HeaderValue::from_static(value));
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let _action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        !ctx.response_headers_modified,
        "a response carrying only {name} must be declined (untouched passthrough)"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| resp.headers.contains_key(name)),
        "the sole validator {name} must be preserved on the declined response"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let _action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a response carrying only {name} passes through byte-identical (id not restored)"
    );
}

#[tokio::test]
async fn does_not_restore_for_partial_content_status() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    response.status = http::StatusCode::PARTIAL_CONTENT;
    // A `206 Partial Content` body is a fragment of a larger representation and
    // carries a `Content-Range`; rewriting it would be unsound.
    response.headers.insert(
        http::header::CONTENT_RANGE,
        http::HeaderValue::from_static("bytes 0-31/64"),
    );
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("\"partial\""));

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a partial (ranged) response must be an untouched passthrough"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .is_some_and(|resp| resp.headers.contains_key(http::header::CONTENT_RANGE)
                && resp.headers.contains_key(http::header::ETAG)),
        "a declined partial response must keep its Content-Range and validators"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a partial response passes through unchanged"
    );
}

#[tokio::test]
async fn does_not_restore_for_content_range_on_ok() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    // Even on a `200 OK`, a `Content-Range` marks the body as a partial
    // representation; the presence-based guard declines it independently of the
    // status narrowing.
    response.headers.insert(
        http::header::CONTENT_RANGE,
        http::HeaderValue::from_static("bytes 0-31/64"),
    );

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a Content-Range response must be declined regardless of status"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a ranged 200 response passes through unchanged"
    );
}

#[tokio::test]
async fn does_not_restore_for_non_ok_success_status() {
    let filter = default_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut response = json_ok_response();
    // A 2xx that is not `200 OK` (here a background `202 Accepted` handoff) is
    // not a complete Responses resource to rewrite and must pass through.
    response.status = http::StatusCode::ACCEPTED;

    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(rehydrated_state("resp_prev"));
    ctx.response_header = Some(&mut response);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "on_response should continue");
    assert!(
        !ctx.response_headers_modified,
        "a non-200 success response must not be rewritten"
    );

    let original = r#"{"id":"resp_new","object":"response","previous_response_id":null}"#;
    let mut body = Some(Bytes::from(original));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_response_body should continue"
    );
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_bytes(),
        "a 202 Accepted body passes through unchanged"
    );
}

#[test]
fn restore_ignores_absent_body() {
    let mut body: Option<Bytes> = None;
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert!(body.is_none(), "an absent body must stay absent");
}

#[test]
fn restore_ignores_empty_body() {
    let mut body = Some(Bytes::new());
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert!(
        body.as_ref().unwrap().is_empty(),
        "an empty body must be left unchanged"
    );
}

#[test]
fn restore_ignores_malformed_json() {
    let original: &[u8] = b"{not valid json";
    let mut body = Some(Bytes::from_static(original));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original,
        "a malformed JSON body must pass through unchanged"
    );
}

#[test]
fn restore_ignores_json_array() {
    let original: &[u8] = b"[1,2,3]";
    let mut body = Some(Bytes::from_static(original));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original,
        "a JSON array must pass through unchanged"
    );
}

#[test]
fn restore_ignores_json_scalar() {
    let original: &[u8] = b"42";
    let mut body = Some(Bytes::from_static(original));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original,
        "a JSON scalar must pass through unchanged"
    );
}

#[test]
fn restore_ignores_non_response_object() {
    let original: &[u8] = br#"{"object":"list","data":[]}"#;
    let mut body = Some(Bytes::from_static(original));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original,
        "a JSON object that is not a Responses resource must pass through unchanged"
    );
}

#[test]
fn restore_inserts_previous_response_id_when_field_absent() {
    let mut body = Some(Bytes::from_static(br#"{"id":"resp_new","object":"response"}"#));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    let patched: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        patched["previous_response_id"], "resp_prev",
        "a Responses resource missing the field should gain it"
    );
    assert_eq!(patched["id"], "resp_new", "other fields must be preserved");
}

#[test]
fn restore_replaces_null_previous_response_id() {
    let mut body = Some(Bytes::from_static(
        br#"{"id":"resp_new","object":"response","previous_response_id":null}"#,
    ));
    restore_previous_response_id("resp_prev".to_owned(), &mut body);
    let patched: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        patched["previous_response_id"], "resp_prev",
        "a null previous_response_id must be replaced with the caller's id"
    );
}

// -----------------------------------------------------------------------------
// Streaming restore pure functions
// -----------------------------------------------------------------------------

#[test]
fn rewrite_lifecycle_frame_data_injects_into_response_object() {
    let data =
        br#"{"type":"response.created","response":{"id":"resp_new","object":"response","previous_response_id":null}}"#;
    let out = rewrite_lifecycle_frame_data(data, "resp_prev").expect("a lifecycle frame should be rewritten");
    let parsed: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(
        parsed["response"]["previous_response_id"], "resp_prev",
        "the caller id must be injected into the nested response object"
    );
    assert_eq!(parsed["type"], "response.created", "the event type must be preserved");
    assert_eq!(
        parsed["response"]["id"], "resp_new",
        "other response fields must be preserved"
    );
}

#[test]
fn rewrite_lifecycle_frame_data_skips_delta_frame() {
    let data = br#"{"type":"response.output_text.delta","delta":"Hi"}"#;
    assert!(
        rewrite_lifecycle_frame_data(data, "resp_prev").is_none(),
        "a delta frame has no top-level response object and must be skipped"
    );
}

#[test]
fn rewrite_lifecycle_frame_data_skips_non_response_shaped_object() {
    let data = br#"{"type":"x","response":{"object":"list"}}"#;
    assert!(
        rewrite_lifecycle_frame_data(data, "resp_prev").is_none(),
        "a response member with a non-response object must be skipped"
    );
}

#[test]
fn rewrite_lifecycle_frame_data_skips_non_json() {
    assert!(
        rewrite_lifecycle_frame_data(b"{not json", "resp_prev").is_none(),
        "a non-JSON payload must be skipped"
    );
}

fn armed_stream(max_buffer_bytes: usize, prev_id: &str) -> RestorePreviousResponseIdStream {
    RestorePreviousResponseIdStream {
        pending: BytesMut::new(),
        scan_from: 0,
        scan_at_line_start: true,
        max_buffer_bytes,
        previous_response_id: prev_id.to_owned(),
    }
}

#[test]
fn stream_chunk_fails_open_on_buffer_overflow() {
    let mut armed = armed_stream(16, "resp_prev");
    // A single line far exceeding the 16-byte buffer with no frame terminator can
    // never complete, forcing the fail-open flush.
    let original: &[u8] =
        br#"data: {"type":"response.created","response":{"object":"response","previous_response_id":null}}"#;
    let mut body = Some(Bytes::from_static(original));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(!keep, "a buffer overflow must disarm the restore (fail-open)");
    assert_eq!(
        body.as_deref(),
        Some(original),
        "on overflow the chunk must pass through untouched, never corrupted or errored"
    );
}

#[test]
fn stream_chunk_withholds_incomplete_frame_bytes() {
    let mut armed = armed_stream(1 << 20, "resp_prev");
    // No blank line: the frame is incomplete, so no bytes are emitted yet.
    let mut body = Some(Bytes::from_static(b"event: response.created\ndata: {\"response\":"));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "an incomplete frame keeps the restore armed");
    assert!(
        body.is_none(),
        "an incomplete frame must not emit partial bytes; they stay buffered"
    );
}

#[test]
fn stream_chunk_flushes_buffered_prefix_on_overflow() {
    // Finding 1 regression: a partial frame withheld from an earlier chunk lives
    // only in `pending`. When a later chunk overflows the buffer, the fail-open
    // flush must include that buffered prefix — never just the overflowing chunk.
    let mut armed = armed_stream(40, "resp_prev");

    let prefix: &[u8] = b"event: response.created\n";
    let mut body = Some(Bytes::from_static(prefix));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "the first partial frame keeps the restore armed");
    assert!(body.is_none(), "the buffered prefix emits no partial bytes yet");

    let overflow: &[u8] = b"data: {\"partial\":\"loooooong-unterminated\"}";
    let mut body = Some(Bytes::from_static(overflow));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(!keep, "exceeding the buffer limit disarms the restore (fail-open)");

    let mut expected = Vec::new();
    expected.extend_from_slice(prefix);
    expected.extend_from_slice(overflow);
    assert_eq!(
        body.as_deref(),
        Some(expected.as_slice()),
        "fail-open must flush the buffered prefix AND the overflowing chunk — no bytes dropped"
    );
}

#[test]
fn stream_chunk_flushes_unterminated_frame_at_end_of_stream() {
    let mut armed = armed_stream(1 << 20, "resp_prev");
    let tail: &[u8] = b"event: response.completed\ndata: {\"response\":";
    let mut body = Some(Bytes::from_static(tail));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, true);
    assert!(!keep, "end of stream disarms the restore");
    assert_eq!(
        body.as_deref(),
        Some(tail),
        "an unterminated final frame is flushed raw at end of stream, never withheld"
    );
}

#[test]
fn stream_chunk_fails_open_on_oversized_complete_frame() {
    // Cap-enforcement regression: the buffer limit must bound a *complete* frame
    // too, not only an incomplete trailing one. A terminated lifecycle frame larger
    // than the limit must never be JSON-parsed or rewritten — it is flushed raw and
    // the restore disarms (fail-open), so `previous_response_id` stays exactly as the
    // backend sent it (`null`), which proves the oversized frame was not parsed.
    let mut armed = armed_stream(16, "resp_prev");
    let original: &[u8] = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    )
    .as_bytes();
    let mut body = Some(Bytes::from_static(original));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(!keep, "an oversized complete frame must disarm the restore (fail-open)");
    assert_eq!(
        body.as_deref(),
        Some(original),
        "an oversized complete frame is flushed raw and never parsed or rewritten"
    );
}

#[test]
fn stream_chunk_forwards_passthrough_frame_without_copying() {
    // Ownership regression: a chunk that needs no rewrite is forwarded as a
    // zero-copy slice of the original `Bytes`, never copied into a fresh buffer. The
    // forwarded chunk must share the input's backing allocation (same pointer).
    let mut armed = armed_stream(1 << 20, "resp_prev");
    let original: &[u8] = concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n",
        "\n",
    )
    .as_bytes();
    let input = Bytes::from_static(original);
    let input_ptr = input.as_ptr();
    let mut body = Some(input);
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "a complete pass-through frame keeps the restore armed");
    let forwarded = body.expect("a complete frame must be forwarded, not withheld");
    assert_eq!(
        forwarded.as_ref(),
        original,
        "a pass-through frame must be byte-identical"
    );
    assert_eq!(
        forwarded.as_ptr(),
        input_ptr,
        "a pass-through chunk must be forwarded without copying its bytes"
    );
}

#[test]
fn stream_chunk_resumes_scan_without_reprocessing_buffered_bytes() {
    // Amplification regression (P1): a single frame spanning many chunks must be
    // scanned once, never rescanned from byte zero every chunk. The resume cursor
    // (`scan_from`) must always sit at the end of everything buffered so far, so the
    // next chunk only scans its own new bytes. Without this the work is quadratic — a
    // 64 MiB frame in 16 KiB chunks would copy and scan ~128 GiB.
    let mut armed = armed_stream(1 << 20, "resp_restored");
    let head: &[u8] = concat!(
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":",
        "{\"object\":\"response\",\"previous_response_id\":null,\"pad\":\"",
    )
    .as_bytes();
    let mut body = Some(Bytes::copy_from_slice(head));
    assert!(restore_previous_response_id_stream_chunk(&mut armed, &mut body, false));
    assert!(body.is_none(), "the opening partial frame is withheld, not emitted");
    assert_eq!(
        armed.scan_from,
        armed.pending.len(),
        "the resume cursor must sit at the buffer end, proving no rescan from zero"
    );

    for _ in 0..64 {
        let mut body = Some(Bytes::from_static(b"xxxxxxxxxxxxxxxx"));
        assert!(restore_previous_response_id_stream_chunk(&mut armed, &mut body, false));
        assert!(body.is_none(), "an unterminated frame keeps withholding its bytes");
        assert_eq!(
            armed.scan_from,
            armed.pending.len(),
            "each chunk resumes at the prior end; buffered bytes are never rescanned"
        );
    }

    // Close the frame: the whole buffered lifecycle frame is rewritten and emitted,
    // and `pending` is drained.
    let mut body = Some(Bytes::from_static(b"\"}}\n\n"));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "a completed frame keeps the restore armed");
    let forwarded = body.expect("the completed frame must be emitted");
    let text = String::from_utf8(forwarded.to_vec()).expect("utf8");
    assert!(
        text.contains("resp_restored"),
        "the buffered lifecycle frame must have previous_response_id restored"
    );
    assert!(
        !text.contains("null"),
        "the backend's null previous_response_id must be replaced, not left in place"
    );
    assert!(armed.pending.is_empty(), "the emitted frame is drained from pending");
}

#[test]
fn stream_chunk_forwards_completed_frame_and_compacts_pending() {
    // Carry-over path: when a chunk completes a buffered frame and opens a new partial,
    // the completed frame is emitted (rewritten) and dropped off the FRONT of
    // `pending`, leaving only the new partial — the buffer never accumulates forwarded
    // frames.
    let mut armed = armed_stream(1 << 20, "resp_restored");

    let first: &[u8] = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"respo",
    )
    .as_bytes();
    let mut body = Some(Bytes::from_static(first));
    assert!(restore_previous_response_id_stream_chunk(&mut armed, &mut body, false));
    assert!(body.is_none(), "the first partial frame is withheld");

    let rest: &[u8] = concat!(
        "nse\":{\"object\":\"response\",\"previous_response_id\":null}}\n\n",
        "event: response.in_progress\ndata: ",
    )
    .as_bytes();
    let mut body = Some(Bytes::from_static(rest));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "the completed first frame keeps the restore armed");

    let forwarded = body.expect("the completed first frame must be emitted");
    let text = String::from_utf8(forwarded.to_vec()).expect("utf8");
    assert!(
        text.contains("resp_restored"),
        "the completed lifecycle frame must have previous_response_id restored"
    );
    assert!(
        text.starts_with("event: response.created\n"),
        "only the completed first frame is emitted, verbatim except the payload"
    );
    assert_eq!(
        armed.pending.as_ref(),
        b"event: response.in_progress\ndata: ",
        "the forwarded frame is dropped off the front; only the new partial remains"
    );
    assert_eq!(
        armed.scan_from,
        armed.pending.len(),
        "the new partial is fully scanned; the next chunk resumes at its end"
    );
}

#[test]
fn stream_chunk_rewrites_lifecycle_frame_terminated_by_bare_cr_at_end_of_stream() {
    // Terminator regression (P2): a valid lifecycle frame whose only line endings are
    // bare carriage returns — ending in a blank `\r\r` — must still be detected and
    // rewritten at end-of-stream. Mid-stream a trailing lone `\r` is ambiguous (it may
    // still become `\r\n`), but at end-of-stream no continuation byte can follow, so the
    // frame completes and `previous_response_id` is restored rather than flushed raw with
    // the backend's `null`.
    let mut armed = armed_stream(1 << 20, "resp_restored");
    let frame: &[u8] = concat!(
        "event: response.completed\r",
        "data: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\r",
        "\r",
    )
    .as_bytes();
    let mut body = Some(Bytes::from_static(frame));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, true);
    assert!(!keep, "end of stream disarms the restore");
    let forwarded = body.expect("the final CR-terminated frame must be emitted");
    let text = String::from_utf8(forwarded.to_vec()).expect("utf8");
    assert!(
        text.contains("resp_restored"),
        "a bare-CR-terminated lifecycle frame must have previous_response_id restored at end of stream"
    );
    assert!(
        !text.contains("null"),
        "the backend's null previous_response_id must be replaced, not flushed raw"
    );
    assert!(
        text.starts_with("event: response.completed\r"),
        "the frame's original CR line endings are preserved verbatim outside the payload"
    );
}

#[test]
fn stream_chunk_defers_ambiguous_trailing_cr_across_chunk_boundary() {
    // The bare-CR fix must stay scoped to end-of-stream: mid-stream a trailing lone `\r`
    // is still ambiguous, because the next chunk may begin with `\n` to form a single
    // `\r\n` terminator. Splitting a `\r\n\r\n` frame boundary as `...\r\n\r` + `\n` must
    // yield exactly one complete frame, never a corrupted early split.
    let mut armed = armed_stream(1 << 20, "resp_restored");
    let head: &[u8] = concat!(
        "event: response.completed\r\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\r\n",
        "\r",
    )
    .as_bytes();
    let mut body = Some(Bytes::from_static(head));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "an ambiguous trailing CR keeps the restore armed");
    assert!(
        body.is_none(),
        "the frame boundary is ambiguous until the next byte; nothing is emitted yet"
    );

    let mut body = Some(Bytes::from_static(b"\n"));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);
    assert!(keep, "the completed frame keeps the restore armed");
    let forwarded = body.expect("the completed frame must be emitted once the CRLF resolves");
    let text = String::from_utf8(forwarded.to_vec()).expect("utf8");
    assert!(
        text.contains("resp_restored"),
        "the frame completes as one CRLF boundary and gets previous_response_id restored"
    );
    assert!(
        text.ends_with("\r\n\r\n"),
        "the split \\r\\n\\r + \\n must rejoin into a single CRLF blank-line terminator"
    );
    assert!(armed.pending.is_empty(), "the emitted frame is drained from pending");
}

#[test]
fn stream_chunk_restores_when_chunk_completes_valid_frame_and_carries_more() {
    // Cap regression (P2): whether previous_response_id is restored must NOT depend on
    // how the transport chunked the stream. `max_buffer_bytes` bounds a single SSE frame,
    // never the transient `pending + incoming` sum. A chunk that finishes a valid carried
    // lifecycle frame and carries the start of the next frame must be processed in full —
    // the completed frame rewritten and forwarded, the trailing partial retained — even
    // when `pending + incoming` transiently exceeds the per-frame budget.
    let frame1: &[u8] = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"previous_response_id\":null}}\n",
        "\n",
    )
    .as_bytes();
    let frame2_partial: &[u8] = b"event: response.in_progress\ndata: {\"type\":\"response.in_prog";

    // The cap admits one whole frame (frame1) but is smaller than `pending + incoming`,
    // so a sum-based pre-check would wrongly fail open on the completing chunk.
    let cap = frame1.len() + 8;
    assert!(frame1.len() <= cap, "frame1 must fit the per-frame budget");
    assert!(frame2_partial.len() <= cap, "the retained partial must fit the budget");
    assert!(
        cap < frame1.len() + frame2_partial.len(),
        "pending + incoming must exceed the cap to exercise the regression"
    );

    let mut armed = armed_stream(cap, "resp_restored");

    // Chunk 1: an incomplete opening frame (cut before its blank-line terminator) is
    // withheld and retained within the cap.
    let split = frame1.len() - 6;
    let mut body = Some(Bytes::copy_from_slice(&frame1[..split]));
    assert!(restore_previous_response_id_stream_chunk(&mut armed, &mut body, false));
    assert!(body.is_none(), "the opening partial frame is withheld");
    assert!(
        armed.pending.len() <= armed.max_buffer_bytes,
        "the retained buffer stays within the cap"
    );

    // Chunk 2: finishes frame1 and carries the start of frame2. `pending + incoming`
    // exceeds the cap, but frame1 itself does not — it must be restored, not failed open.
    let mut chunk2 = Vec::new();
    chunk2.extend_from_slice(&frame1[split..]);
    chunk2.extend_from_slice(frame2_partial);
    assert!(
        armed.pending.len() + chunk2.len() > armed.max_buffer_bytes,
        "the transient pending + incoming sum must exceed the cap here"
    );
    let mut body = Some(Bytes::from(chunk2));
    let keep = restore_previous_response_id_stream_chunk(&mut armed, &mut body, false);

    assert!(
        keep,
        "a valid completed frame keeps the restore armed regardless of chunking"
    );
    let forwarded = body.expect("the completed frame must be forwarded");
    let text = String::from_utf8(forwarded.to_vec()).expect("utf8");
    assert!(
        text.contains("\"previous_response_id\":\"resp_restored\""),
        "the completed lifecycle frame has previous_response_id restored: {text}"
    );
    assert!(
        !text.contains("\"previous_response_id\":null"),
        "the stale null id must be replaced, not forwarded"
    );
    assert!(
        !text.contains("response.in_progress"),
        "only the completed frame is forwarded; the trailing partial stays withheld"
    );
    assert_eq!(
        armed.pending.as_ref(),
        frame2_partial,
        "only the trailing partial frame is retained across the callback"
    );
    assert!(
        armed.pending.len() <= armed.max_buffer_bytes,
        "the retained partial stays within the cap"
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
