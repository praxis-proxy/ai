// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_response_store` filter.

use std::{
    num::{NonZeroU32, NonZeroU64},
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterEntry, FilterPipeline, HttpFilter as _, HttpFilterContext,
    StreamingTerminalResponse, parse_filter_config,
};
use serde_json::json;

use super::{
    ResponseStoreFilter,
    config::{ResponseStoreConfig, validate_config},
};
use crate::{
    openai::{
        include::{IncludeField, IncludeFields},
        responses::state::ResponsesState,
    },
    service::responses::{
        InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, input_items::DEFAULT_PAGE_LIMIT, list_input_items,
    },
    store::{
        DEFAULT_STORE_NAME, PersistedStateBackend, ResponseRecord, ResponseStore as _, ResponseStoreRegistry,
        SqliteResponseStore,
    },
};

// -----------------------------------------------------------------------------
// from_config
// -----------------------------------------------------------------------------

#[test]
fn valid_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let filter = ResponseStoreFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_response_store",
        "filter should parse successfully"
    );
}

#[test]
fn config_with_zstd_compression_parses_and_validates() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
compression:
  algorithm: zstd
  level: 5
"#,
    )
    .unwrap();
    let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", &yaml).unwrap();
    validate_config(&cfg).expect("valid zstd compression config should validate");
    let compression = cfg.compression.expect("compression should be present");
    assert_eq!(compression.level, Some(5), "configured level should round-trip");
}

#[test]
fn config_with_invalid_compression_level_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
compression:
  algorithm: none
  level: 3
"#,
    )
    .unwrap();
    let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", &yaml).unwrap();
    let err = validate_config(&cfg).expect_err("level with algorithm none should be rejected");
    assert!(
        format!("{err}").contains("level"),
        "error should mention the offending level field: {err}"
    );
}

#[test]
fn empty_database_url_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: ""
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "empty database_url should be rejected");
}

#[test]
fn database_url_path_traversal_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite://../responses.db?mode=rwc"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "database_url with .. traversal should be rejected");
}

#[test]
fn database_url_encoded_path_traversal_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite://data/%2e%2e/responses.db?mode=rwc"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "database_url with percent-encoded .. traversal should be rejected"
    );
}

#[test]
fn database_url_encoded_slash_traversal_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite://..%2F..%2Fetc%2Fresponses.db?mode=rwc"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "database_url with percent-encoded slash traversal should be rejected"
    );
}

#[test]
fn missing_backend_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "missing backend should be rejected");
}

#[test]
fn invalid_responses_table_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: bad-name
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "invalid responses_table should be rejected");
}

#[test]
fn invalid_conversations_table_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: bad-name
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "invalid conversations_table should be rejected");
}

#[test]
fn duplicate_table_names_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: same_table
conversations_table: same_table
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "duplicate table names should be rejected");
}

#[test]
fn duplicate_table_names_rejected_case_insensitively() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: Responses
conversations_table: responses
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "case-insensitive duplicate table names should be rejected"
    );
}

#[test]
fn unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
unknown_extra_field: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "unknown field should be rejected by deny_unknown_fields"
    );
}

// -----------------------------------------------------------------------------
// Filter Trait Declarations
// -----------------------------------------------------------------------------

#[test]
fn name_returns_openai_response_store() {
    let filter = make_filter();
    assert_eq!(
        filter.name(),
        "openai_response_store",
        "name should be openai_response_store"
    );
}

#[test]
fn response_body_access_is_read_only() {
    let filter = make_filter();
    assert_eq!(
        filter.response_body_access(),
        BodyAccess::ReadOnly,
        "response body access should be ReadOnly"
    );
}

#[test]
fn declares_dual_phase_request_body_access() {
    let filter = make_filter();
    assert_eq!(filter.request_body_access(), BodyAccess::ReadOnly);
    assert_eq!(filter.bound_upstream_request_body_access(), BodyAccess::ReadOnly);
}

#[test]
fn response_body_mode_defaults_to_stream() {
    let filter = make_filter();
    assert_eq!(
        filter.response_body_mode(),
        BodyMode::Stream,
        "streaming requests must not inherit a pipeline-level StreamBuffer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_selects_bounded_stream_buffer_for_non_streaming_responses() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Continue), "request should continue");
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::StreamBuffer {
            max_bytes: Some(67_108_864)
        },
        "non-streaming Responses requests should remain bounded"
    );
}

// -----------------------------------------------------------------------------
// on_request Bypass
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_skips_for_get_to_unrelated_path() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/models");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip for GET to unrelated path"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_skips_delete_on_unrelated_path() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "DELETE to unrelated path should continue"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_rejects_persistable_request_when_store_not_provisioned() {
    // The filter resolves the provisioned store from the context registry. A
    // persistable Responses create with no store provisioned fails fast with a
    // 500 rather than running inference it cannot persist.
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 500,
        "a persistable request without a provisioned store must reject with 500"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_continues_for_previous_response_id_with_store_provisioned() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.store", "false");
    ctx.set_metadata("openai_responses_format.has_previous_response_id", "true");

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "rehydrate with a provisioned store should continue"
    );
}

// -----------------------------------------------------------------------------
// Exchange-scoped persistence arming
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_body_arms_persistence_for_persisted_response() {
    // A Responses create that will persist its response must publish the
    // exchange-scoped persistence-armed marker so a downstream approval pause can
    // tell that THIS response will be stored, not merely that a store exists in
    // the pipeline.
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    // openai_responses_request creates ResponsesState earlier in this body phase.
    ctx.extensions.insert(ResponsesState::default());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"Hi"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "request body phase should continue"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().store_persist_armed,
        "a persisted Responses create must arm persistence for this exchange"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_body_does_not_arm_persistence_when_store_false() {
    // store=false will not persist, so persistence must not be armed. mcp_dispatch
    // rejects such an approval with a client-facing 400 before this point, but the
    // marker must stay honest regardless.
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(ResponsesState::default());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.store", "false");
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"Hi","store":false}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "request body phase should continue"
    );
    assert!(
        !ctx.extensions.get::<ResponsesState>().unwrap().store_persist_armed,
        "store=false must not arm persistence"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_body_does_not_arm_persistence_for_rehydrate_only() {
    // A store=false request with previous_response_id needs the store so rehydrate
    // can read history, but it will not persist a new response, so it must not arm
    // persistence. Arming on store presence alone would resurrect the
    // pipeline-scoped bug this marker exists to fix.
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    ctx.extensions.insert(ResponsesState::default());
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.store", "false");
    ctx.set_metadata("openai_responses_format.has_previous_response_id", "true");
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"Hi","store":false,"previous_response_id":"resp_prev"}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "request body phase should continue"
    );
    assert!(
        !ctx.extensions.get::<ResponsesState>().unwrap().store_persist_armed,
        "a rehydrate-only request that will not persist must not arm persistence"
    );
}

// -----------------------------------------------------------------------------
// on_response
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_skips_when_format_metadata_absent() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when format metadata is absent"
    );
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::Stream,
        "body mode should remain Stream when skipped"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_sets_skip_persist_for_non_2xx() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;

    let mut resp = crate::test_utils::make_response();
    resp.status = http::StatusCode::INTERNAL_SERVER_ERROR;
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should continue for non-2xx");
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "should set skip_persist for non-2xx responses"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_sets_skip_persist_for_non_json_content_type() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue for non-JSON content type"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "should set skip_persist for non-JSON responses"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_continues_for_json_200() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "should continue for JSON 200");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_accepts_mixed_case_json_content_type() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;

    let mut resp = crate::test_utils::make_response();
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        "Application/JSON; charset=utf-8".parse().unwrap(),
    );
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue for mixed-case JSON content type"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_continues_for_event_stream_200() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(&filter, &mut ctx).await;

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue for event-stream 200"
    );
    assert!(
        ctx.get_metadata("responses.skip_persist").is_none(),
        "should not skip persist for event-stream content type"
    );
}

// -----------------------------------------------------------------------------
// on_response_body
// -----------------------------------------------------------------------------

#[test]
fn on_response_body_releases_skipped_non_end_of_stream() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from_static(b"partial"));

    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release non-persisted non-end-of-stream chunks"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_streaming_persist_on_parse_error() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(&filter, &mut ctx).await;

    ctx.set_metadata("responses.stream_parse_error", "true");
    ctx.extensions.insert(ResponsesState {
        response_object: json!({"id": "resp_err", "created_at": 1, "model": "gpt-4.1"}),
        ..Default::default()
    });

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip persistence when stream had parse errors"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_streaming_persist_on_incomplete_stream() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(&filter, &mut ctx).await;

    ctx.set_metadata("responses.stream_incomplete", "true");

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip persistence when stream was incomplete"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_streaming_persist_when_no_state() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(&filter, &mut ctx).await;

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip persistence when ResponsesState is absent"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_streaming_response_at_eos() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let owner = crate::StateOwner::from_trusted_parts("tenant-stream", "issuer-a", "alice").unwrap();
    ctx.extensions.insert(owner.clone());
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(&filter, &mut ctx).await;

    let response_json = json!({
        "id": "resp_stream_unit",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "Streamed reply"}]
    });
    ctx.extensions.insert(ResponsesState {
        response_object: response_json.clone(),
        persisted_messages: vec![json!({"role": "user", "content": "Hello"})],
        ..Default::default()
    });

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after persisting streaming response"
    );

    let record = store
        .get_response(&owner, "resp_stream_unit")
        .await
        .expect("get_response should succeed")
        .expect("record should exist after streaming persist");

    assert_eq!(record.id, "resp_stream_unit", "persisted ID should match");
    assert_eq!(record.created_at, 1_719_900_000, "persisted created_at should match");
    assert_eq!(record.model, "gpt-4.1", "persisted model should match");
    assert_eq!(
        record.owner, owner,
        "streaming persistence must retain the initiating owner"
    );
    let same_tenant_other = crate::StateOwner::from_trusted_parts("tenant-stream", "issuer-a", "bob").unwrap();
    assert!(
        store
            .get_response(&same_tenant_other, "resp_stream_unit")
            .await
            .expect("cross-owner lookup should succeed")
            .is_none(),
        "streaming state must be hidden from another owner in the same tenant"
    );
    assert_eq!(
        record.response_object, response_json,
        "persisted response_object should match the accumulated state"
    );
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Streamed reply"}
        ]),
        "persisted messages should combine persisted input history with streamed output"
    );
}

// Record-assembly unit tests (null / missing-field / streaming-history cases)
// moved to the service layer: `crate::service::responses` tests, where they run
// against plain values with no pipeline or database.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_releases_when_skip_persist_is_true() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    ctx.set_metadata("responses.skip_persist", "true");
    let mut body = Some(Bytes::from_static(b"{}"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "should release when skip_persist is true"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_releases_streaming_request_before_eos() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    let request_action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "streaming request should pass request phase"
    );

    let mut body = Some(Bytes::from_static(b"event: response.output_text.delta\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "streaming response chunks should release before EOS"
    );
}

// -----------------------------------------------------------------------------
// #937: persist the streaming terminal frame before it is released downstream
// -----------------------------------------------------------------------------

/// A `ResponseStore` fake that counts `upsert_response` calls and can be armed to
/// fail every write.
///
/// #937 tests use it to prove the store persists the streaming terminal frame
/// exactly once (before the frame is released) and fails closed when that
/// persistence errors. The real `SqliteResponseStore` cannot force an upsert
/// failure or observe call counts.
struct RecordingResponseStore {
    upserts: std::sync::atomic::AtomicUsize,
    fail: bool,
    records: std::sync::Mutex<std::collections::HashMap<String, ResponseRecord>>,
    /// In-memory replay logs keyed by response id, kept ordered by
    /// `sequence_number`. Mirrors the real backend's owner-gated, insert-if-absent
    /// semantics so replay seam tests observe truthful behavior.
    events: std::sync::Mutex<std::collections::HashMap<String, Vec<crate::store::ResponseEventRecord>>>,
    /// When true, `append_events` fails to prove the terminal seam fails closed.
    fail_events: bool,
}

impl RecordingResponseStore {
    fn new(fail: bool) -> Self {
        Self {
            upserts: std::sync::atomic::AtomicUsize::new(0),
            fail,
            records: std::sync::Mutex::new(std::collections::HashMap::new()),
            events: std::sync::Mutex::new(std::collections::HashMap::new()),
            fail_events: false,
        }
    }

    /// A store whose response upserts succeed but whose event-log appends fail,
    /// used to prove the replay flush fails closed after the record is durable.
    fn new_failing_events() -> Self {
        Self {
            fail_events: true,
            ..Self::new(false)
        }
    }

    fn upsert_count(&self) -> usize {
        self.upserts.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Number of persisted replay events for a response.
    fn event_count(&self, response_id: &str) -> usize {
        self.events
            .lock()
            .expect("events mutex should not be poisoned")
            .get(response_id)
            .map_or(0, Vec::len)
    }
}

#[expect(
    clippy::significant_drop_tightening,
    reason = "each mock method is one short mutex-guarded critical section; holding the guard is what keeps the multi-step ops atomic"
)]
#[async_trait::async_trait]
impl crate::store::ResponseStore for RecordingResponseStore {
    async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), crate::store::StoreError> {
        self.upserts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.fail {
            return Err(crate::store::StoreError::Unavailable(
                "recording store: forced failure".to_owned(),
            ));
        }
        self.records
            .lock()
            .expect("records mutex should not be poisoned")
            .insert(record.id.clone(), record.clone());
        Ok(())
    }

    async fn get_response(
        &self,
        owner: &crate::StateOwner,
        id: &str,
    ) -> Result<Option<ResponseRecord>, crate::store::StoreError> {
        Ok(self
            .records
            .lock()
            .expect("records mutex should not be poisoned")
            .get(id)
            .filter(|record| record.owner == *owner)
            .cloned())
    }

    async fn delete_response(&self, _owner: &crate::StateOwner, _id: &str) -> Result<bool, crate::store::StoreError> {
        Ok(false)
    }

    async fn get_conversation(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
    ) -> Result<Option<crate::store::ConversationRecord>, crate::store::StoreError> {
        Ok(None)
    }

    async fn record_pending_approvals(
        &self,
        _owner: &crate::StateOwner,
        _response_id: &str,
        _records: &[crate::store::PendingApprovalRecord],
        _created_at: i64,
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    async fn get_pending_approvals(
        &self,
        _owner: &crate::StateOwner,
        _response_id: &str,
        _approval_ids: &[&str],
    ) -> Result<Vec<crate::store::PendingApprovalRecord>, crate::store::StoreError> {
        Ok(Vec::new())
    }

    async fn consume_approvals(
        &self,
        _owner: &crate::StateOwner,
        _response_id: &str,
        _approval_ids: &[&str],
        _consumed_at: i64,
    ) -> Result<Option<usize>, crate::store::StoreError> {
        Ok(None)
    }

    async fn append_events(
        &self,
        owner: &crate::StateOwner,
        response_id: &str,
        events: &[crate::store::ResponseEventRecord],
    ) -> Result<(), crate::store::StoreError> {
        if self.fail_events {
            return Err(crate::store::StoreError::Unavailable(
                "recording store: forced event-log failure".to_owned(),
            ));
        }
        // Emulate the backend's EXISTS gate: silently drop rows whose parent
        // response does not exist under the same owner.
        let parent_exists = self
            .records
            .lock()
            .expect("records mutex should not be poisoned")
            .get(response_id)
            .is_some_and(|record| record.owner == *owner);
        if !parent_exists {
            return Ok(());
        }
        let mut logs = self.events.lock().expect("events mutex should not be poisoned");
        let log = logs.entry(response_id.to_owned()).or_default();
        for event in events {
            if event.owner != *owner {
                continue;
            }
            // Insert-if-absent by sequence_number.
            if log.iter().any(|e| e.sequence_number == event.sequence_number) {
                continue;
            }
            log.push(event.clone());
        }
        log.sort_by_key(|e| e.sequence_number);
        Ok(())
    }

    async fn list_events_after(
        &self,
        owner: &crate::StateOwner,
        response_id: &str,
        after: Option<u64>,
        limit: u32,
    ) -> Result<Vec<crate::store::ResponseEventRecord>, crate::store::StoreError> {
        let logs = self.events.lock().expect("events mutex should not be poisoned");
        let Some(log) = logs.get(response_id) else {
            return Ok(Vec::new());
        };
        Ok(log
            .iter()
            .filter(|e| e.owner == *owner)
            .filter(|e| after.is_none_or(|n| e.sequence_number > n))
            .take(limit as usize)
            .cloned()
            .collect())
    }

    async fn event_log_status(
        &self,
        owner: &crate::StateOwner,
        response_id: &str,
    ) -> Result<crate::store::EventLogStatus, crate::store::StoreError> {
        let logs = self.events.lock().expect("events mutex should not be poisoned");
        let Some(log) = logs.get(response_id) else {
            return Ok(crate::store::EventLogStatus::Absent);
        };
        let owned: Vec<&crate::store::ResponseEventRecord> = log.iter().filter(|e| e.owner == *owner).collect();
        let Some(max_sequence) = owned.iter().map(|e| e.sequence_number).max() else {
            return Ok(crate::store::EventLogStatus::Absent);
        };
        if owned.iter().any(|e| e.terminal) {
            Ok(crate::store::EventLogStatus::Replayable { max_sequence })
        } else {
            Ok(crate::store::EventLogStatus::Incomplete { max_sequence })
        }
    }
}

/// Register a store as the default store in the request context so the filter
/// resolves it instead of a real backend.
fn install_recording_store(ctx: &mut HttpFilterContext<'_>, store: Arc<dyn PersistedStateBackend>) {
    let registry = ResponseStoreRegistry::new();
    registry
        .register(&Arc::from(DEFAULT_STORE_NAME), store)
        .expect("recording store should register");
    ctx.extensions.insert(registry);
}

/// Build a streaming request context whose accumulated state carries a canonical
/// `response_object`, optionally already marked as having emitted its terminal
/// `response.completed` frame (`terminal_emitted`).
///
/// Uses [`make_owned_filter_context`], which installs the default
/// [`test_owner`] before the request phase so `capture_persistence_owner`
/// records the immutable owner (#1197); without it streaming persistence would
/// fail closed with a `Reject` instead of writing the record.
///
/// [`make_owned_filter_context`]: crate::test_utils::make_owned_filter_context
/// [`test_owner`]: crate::test_utils::test_owner
async fn armed_streaming_ctx<'a>(
    filter: &ResponseStoreFilter,
    req: &'a praxis_filter::Request,
    store: Arc<dyn PersistedStateBackend>,
    response_id: &str,
    terminal_emitted: bool,
) -> HttpFilterContext<'a> {
    let mut ctx = crate::test_utils::make_owned_filter_context(req);
    install_recording_store(&mut ctx, store);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    run_request_phase(filter, &mut ctx).await;

    ctx.extensions.insert(ResponsesState {
        response_object: json!({
            "id": response_id,
            "created_at": 1_719_900_100_i64,
            "model": "gpt-4.1",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": "Done"}]
        }),
        persisted_messages: vec![json!({"role": "user", "content": "Hi"})],
        logical_stream_terminal_emitted: terminal_emitted,
        ..Default::default()
    });
    ctx
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_terminal_frame_persists_before_eos_release() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_937_terminal", true).await;

    // The deferred terminal `response.completed` frame arrives as a
    // non-end-of-stream chunk. It must be persisted BEFORE it is released.
    let mut terminal = Some(Bytes::from_static(
        b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
    ));
    let action = filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "the terminal frame chunk is still released, only after persistence"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "the streaming response must be persisted at the terminal frame, before release"
    );
    assert!(
        store
            .get_response(&crate::test_utils::test_owner("default"), "resp_937_terminal")
            .await
            .unwrap()
            .is_some(),
        "the record must be durable before the client can observe response.completed"
    );

    // The subsequent empty end-of-stream callback must not persist again.
    let mut eos_body: Option<Bytes> = None;
    let eos_action = filter.on_response_body(&mut ctx, &mut eos_body, true).unwrap();
    assert!(
        matches!(eos_action, FilterAction::Continue),
        "end-of-stream after a persisted terminal frame should continue"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "end-of-stream must not trigger a redundant second persist"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_local_terminal_persists_before_release_and_only_once() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_410_local", false).await;
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .local_stream_terminal_emitted = true;

    let mut terminal = Some(Bytes::from_static(b"event: response.completed\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(matches!(action, FilterAction::Release));
    assert_eq!(
        store.upsert_count(),
        1,
        "local completion must be durable before release"
    );

    let mut eos_body = None;
    drop(filter.on_response_body(&mut ctx, &mut eos_body, true).unwrap());
    assert_eq!(
        store.upsert_count(),
        1,
        "EOS must not repeat the local completion write"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_local_terminal_at_eos_still_persists() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_410_local_eos", false).await;
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .local_stream_terminal_emitted = true;

    let mut terminal = Some(Bytes::from_static(b"event: response.completed\n\n"));
    drop(filter.on_response_body(&mut ctx, &mut terminal, true).unwrap());
    assert_eq!(store.upsert_count(), 1, "an EOS terminal must still be durable");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_error_at_eos_does_not_persist_completed_upstream_snapshot() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_410_failed_irr", false).await;

    // IRR can retain an upstream `status: completed` snapshot while replacing
    // its deferred response.completed with a client-visible SSE error. The
    // inner step's error metadata does not reach this outer filter.
    let mut error = Some(Bytes::from_static(b"event: error\ndata: {\"type\":\"error\"}\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut error, false).unwrap();
    assert!(matches!(action, FilterAction::Release));
    let mut eos_body = None;
    drop(filter.on_response_body(&mut ctx, &mut eos_body, true).unwrap());
    assert_eq!(store.upsert_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_error_after_replay_decoder_overflow_does_not_persist() {
    let filter = ResponseStoreFilter::with_bounds(NonZeroU32::new(64).unwrap(), NonZeroU64::new(1).unwrap());
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_410_overflow_error", false).await;

    // The first client-visible frame exceeds the replay decoder's byte limit.
    // Its poisoned decoder cannot decode the later error, which can also be
    // split across response-body callbacks.
    let created = format!(
        "event: response.created\ndata: {{\"type\":\"response.created\",\"sequence_number\":0,\"padding\":\"{}\"}}\n\n",
        "x".repeat(3072)
    );
    let mut body = Some(Bytes::from(created));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
        FilterAction::Release
    ));
    for part in [b"event: er".as_slice(), b"ror\ndata: {\"type\":\"error\"}\n\n"] {
        let mut body = Some(Bytes::copy_from_slice(part));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
            FilterAction::Release
        ));
    }
    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_eq!(
        store.upsert_count(),
        0,
        "the hidden completed snapshot must not be persisted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_terminal_frame_persist_failure_fails_closed() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(true));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_937_failclosed", true).await;

    let mut terminal = Some(Bytes::from_static(
        b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
    ));
    let result = filter.on_response_body(&mut ctx, &mut terminal, false);
    assert!(
        result.is_err(),
        "a persistence failure on the terminal frame must fail closed so the client never observes response.completed"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "the failing upsert must have been attempted for the terminal frame"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_terminal_fail_open_error_is_not_retried_at_eos() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(true));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_410_open", true).await;

    let mut terminal = Some(Bytes::from_static(b"event: response.completed\n\n"));
    assert!(filter.on_response_body(&mut ctx, &mut terminal, false).is_err());
    // A failure_mode: open pipeline suppresses the error above and delivers
    // the chunk. EOS must not retry state already consumed by that write.
    let mut eos_body = None;
    let eos_action = filter.on_response_body(&mut ctx, &mut eos_body, true).unwrap();
    assert!(matches!(eos_action, FilterAction::Continue));
    assert_eq!(store.upsert_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_terminal_frame_propagates_persist_rejection() {
    // #937 (latest-main integration): `persist_from_streaming_state` returns
    // `Ok(FilterAction::Reject(_))` when the immutable owner/request state was
    // never captured (#1197). The deferred-terminal branch must propagate that
    // rejection instead of unconditionally releasing `response.completed`, so a
    // client never observes completion for a record that was refused persistence.
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));

    // Deliberately skip the request phase so no `ResponseStoreRequestState`
    // (owner + input) is captured; streaming persistence then fails closed.
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);
    install_recording_store(&mut ctx, store_dyn);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "true");
    ctx.extensions.insert(ResponsesState {
        response_object: json!({
            "id": "resp_937_reject",
            "created_at": 1_719_900_100_i64,
            "model": "gpt-4.1",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": "Done"}]
        }),
        logical_stream_terminal_emitted: true,
        ..Default::default()
    });

    let mut terminal = Some(Bytes::from_static(
        b"event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
    ));
    let rejection = expect_reject(filter.on_response_body(&mut ctx, &mut terminal, false).unwrap());
    assert_eq!(
        rejection.status, 500,
        "a fail-closed persist rejection must propagate, not release response.completed"
    );
    assert_has_json_content_type(&rejection);
    assert_eq!(
        store.upsert_count(),
        0,
        "the missing-owner guard rejects before any upsert is attempted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_without_terminal_signal_persists_only_at_eos() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    // No terminal frame observed through this filter (e.g. a plain single-round
    // stream): the signal stays unset and the EOS fallback still persists.
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_937_fallback", false).await;

    let mut chunk = Some(Bytes::from_static(b"event: response.output_text.delta\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "content chunks without a terminal signal release without persisting"
    );
    assert_eq!(
        store.upsert_count(),
        0,
        "no persistence should occur before EOS without a terminal signal"
    );

    let mut eos_body: Option<Bytes> = None;
    let eos_action = filter.on_response_body(&mut ctx, &mut eos_body, true).unwrap();
    assert!(
        matches!(eos_action, FilterAction::Continue),
        "end-of-stream should persist from accumulated state and continue"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "the EOS fallback persists exactly once for non-deferred streams"
    );
}

// -----------------------------------------------------------------------------
// Replay event-log capture → terminal-seam flush
// -----------------------------------------------------------------------------

/// Three canonical SSE events (created, one delta, completed) in one chunk, the
/// terminal event last. Each carries the numeric `sequence_number` capture keys
/// on.
const SEAM_EVENTS_CHUNK: &[u8] = b"event: response.created\n\
data: {\"type\":\"response.created\",\"sequence_number\":0}\n\n\
event: response.output_text.delta\n\
data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1}\n\n\
event: response.completed\n\
data: {\"type\":\"response.completed\",\"sequence_number\":2}\n\n";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_events_persist_at_terminal_seam() {
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_seam", true).await;

    // The deferred terminal frame arrives with the captured events; the seam
    // persists the record and then flushes the whole log before release.
    let mut chunk = Some(Bytes::from_static(SEAM_EVENTS_CHUNK));
    let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "the terminal chunk is released after the record and log persist"
    );
    assert_eq!(store.upsert_count(), 1, "the JSON record persists exactly once");
    assert_eq!(
        store.event_count("resp_seam"),
        3,
        "all captured events flush together at the terminal seam"
    );

    let events = store
        .list_events_after(&crate::test_utils::test_owner("default"), "resp_seam", None, 10)
        .await
        .unwrap();
    assert_eq!(
        events.iter().map(|e| e.sequence_number).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "events persist in ascending sequence order"
    );
    assert!(
        events.last().unwrap().terminal,
        "the terminal event must be marked terminal so the log is replayable"
    );
    assert!(
        events.iter().take(2).all(|e| !e.terminal),
        "non-terminal events must not be flagged terminal"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_event_flush_failure_fails_closed() {
    let filter = make_filter();
    // Record upserts succeed; the event-log append fails.
    let store = Arc::new(RecordingResponseStore::new_failing_events());
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_seam_fail", true).await;

    let mut chunk = Some(Bytes::from_static(SEAM_EVENTS_CHUNK));
    let result = filter.on_response_body(&mut ctx, &mut chunk, false);
    assert!(
        result.is_err(),
        "a replay-log append failure must fail closed so the client never observes response.completed"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "the record persists before the failing event flush is attempted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_events_over_budget_are_not_persisted() {
    // A one-event count bound: the second event trips the bound, abandoning the
    // whole log so the response stays retrievable as JSON but is not replayable.
    let filter = ResponseStoreFilter::with_bounds(NonZeroU32::new(1).unwrap(), super::config::DEFAULT_MAX_EVENT_BYTES);
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_seam_budget", true).await;

    let mut chunk = Some(Bytes::from_static(SEAM_EVENTS_CHUNK));
    let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "an over-budget capture never withholds the client bytes"
    );
    assert_eq!(
        store.upsert_count(),
        1,
        "the JSON record still persists when the log is abandoned"
    );
    assert_eq!(
        store.event_count("resp_seam_budget"),
        0,
        "an over-budget log is dropped entirely so no partial replay is served"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_single_event_over_line_default_is_captured() {
    // A single terminal event larger than the SSE decoder's 1 MiB default
    // per-line cap but well within the 16 MiB replay byte budget. Before the
    // capture decoder raised `max_line_bytes` alongside `max_record_bytes`, a
    // >1 MiB `data:` line poisoned the decoder and abandoned the whole log; the
    // event must now be captured and remain replayable.
    let filter = make_filter();
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_big_event", true).await;

    // ~2 MiB single-line terminal event: one `data: <json>` line whose padding
    // pushes it past the 1 MiB default per-line limit but under 16 MiB.
    let pad = "x".repeat(2 * 1024 * 1024);
    let big_event = format!(
        "event: response.completed\ndata: {{\"type\":\"response.completed\",\"sequence_number\":0,\"pad\":\"{pad}\"}}\n\n"
    );
    let mut chunk = Some(Bytes::from(big_event));
    let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "a within-budget large event is captured and released at the terminal seam"
    );
    assert_eq!(
        store.event_count("resp_big_event"),
        1,
        "a >1 MiB event under the byte budget must be captured, not dropped by a stale 1 MiB line cap"
    );

    let events = store
        .list_events_after(&crate::test_utils::test_owner("default"), "resp_big_event", None, 10)
        .await
        .unwrap();
    assert!(
        events.last().is_some_and(|e| e.terminal),
        "the large terminal event must remain marked terminal so the log is replayable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_event_within_budget_but_over_raw_framing_is_captured() {
    // The configured `max_event_bytes` bounds the JSON `data` payload only, but
    // the SSE decoder additionally counts framing: `max_record_bytes` sums the
    // `event:` type value alongside the payload. Sizing the decoder to exactly
    // the budget rejected a within-budget payload once framing pushed the record
    // past it -- poisoning the decoder and abandoning the whole log so a
    // completed response returned 400 on replay. The framing headroom must let
    // the event decode and reach the authoritative payload check.
    let budget: u64 = 100;
    let filter =
        ResponseStoreFilter::with_bounds(super::config::DEFAULT_MAX_EVENT_COUNT, NonZeroU64::new(budget).unwrap());
    let store = Arc::new(RecordingResponseStore::new(false));
    let store_dyn: Arc<dyn PersistedStateBackend> = Arc::<RecordingResponseStore>::clone(&store);

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = armed_streaming_ctx(&filter, &req, store_dyn, "resp_framing_boundary", true).await;

    // A `data` payload within the byte budget whose SSE record (payload plus the
    // `event: response.completed` type value) exceeds the raw budget. A decoder
    // sized to exactly `max_event_bytes` reports `RecordTooLarge` here.
    let event_type = "response.completed";
    let prefix = r#"{"type":"response.completed","sequence_number":0,"p":""#;
    let suffix = r#""}"#;
    let target_data_len = 90_usize;
    let pad_len = target_data_len - prefix.len() - suffix.len();
    let data = format!("{prefix}{}{suffix}", "x".repeat(pad_len));
    assert_eq!(data.len(), target_data_len, "test payload is the intended size");
    assert!(
        data.len() as u64 <= budget,
        "the JSON payload must be within the configured byte budget"
    );
    assert!(
        (event_type.len() + data.len()) as u64 > budget,
        "the SSE record framing must exceed the raw budget so it exercises the headroom"
    );

    let event = format!("event: {event_type}\ndata: {data}\n\n");
    let mut chunk = Some(Bytes::from(event));
    let action = filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        matches!(action, FilterAction::Release),
        "a within-budget event is captured and released at the terminal seam"
    );
    assert_eq!(
        store.event_count("resp_framing_boundary"),
        1,
        "an event within the payload budget must survive SSE framing, not be dropped as RecordTooLarge"
    );

    let events = store
        .list_events_after(
            &crate::test_utils::test_owner("default"),
            "resp_framing_boundary",
            None,
            10,
        )
        .await
        .unwrap();
    assert!(
        events.last().is_some_and(|e| e.terminal),
        "the terminal event must remain terminal so the log is replayable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_buffers_persistable_non_end_of_stream() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;

    let mut body = Some(Bytes::from_static(b"{\"id\":\"resp_partial\""));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "persistable response should remain buffered before EOS"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_body_is_none() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let mut body: Option<Bytes> = None;

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when body is None"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_continues_when_terminal_body_is_none() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    drop(filter.on_request(&mut ctx).await.unwrap());

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    drop(filter.on_response(&mut ctx).await.unwrap());

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip terminal response with no body"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_body_is_empty() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let mut body = Some(Bytes::new());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when body is empty"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_body_is_invalid_json() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let mut body = Some(Bytes::from_static(b"not json {{{"));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when body is invalid JSON"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_id_field_missing() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let body_json = json!({"created_at": 1000, "model": "gpt-4.1"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when id field is missing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_created_at_field_missing() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let body_json = json!({"id": "resp_test", "model": "gpt-4.1"});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when created_at field is missing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_skips_when_model_field_missing() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    run_request_phase(&filter, &mut ctx).await;
    let body_json = json!({"id": "resp_test", "created_at": 1000});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should skip when model field is missing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_valid_response() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    drop(filter.on_request(&mut ctx).await.unwrap());


    let body_json = json!({
        "id": "resp_test123",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": [{"role": "user", "content": "Hello"}],
        "output": [{"type": "message", "content": "Hello"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after spawning persist task"
    );

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_test123")
        .await
        .expect("get_response should succeed")
        .expect("record should exist after persist");

    assert_eq!(record.id, "resp_test123", "persisted ID should match");
    assert_eq!(record.created_at, 1_719_900_000, "persisted created_at should match");
    assert_eq!(record.model, "gpt-4.1", "persisted model should match");
    assert_eq!(
        record.owner.tenant_id(),
        "default",
        "persisted tenant_id should be default"
    );
    assert_eq!(
        record.response_object, body_json,
        "persisted response_object should match the full JSON"
    );
    assert_eq!(
        record.input, body_json["input"],
        "persisted input should be extracted from the response"
    );
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Hello"},
            {"type": "message", "content": "Hello"}
        ]),
        "persisted messages should preserve input before output for rehydration"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_string_input_as_message_item() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    drop(filter.on_request(&mut ctx).await.unwrap());


    let body_json = json!({
        "id": "resp_string_input",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": "Hello",
        "output": [{"type": "message", "content": "Hi"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after spawning persist task"
    );

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_string_input")
        .await
        .expect("get_response should succeed")
        .expect("record should exist after persist");

    assert_eq!(
        record.input, body_json["input"],
        "persisted input should preserve the response input"
    );
    assert_eq!(
        record.messages,
        json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "content": "Hi"}
        ]),
        "persisted messages should normalize string input before output"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_uses_request_input_when_response_omits_input() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.current_filter_id = Some(7);

    let request_input = json!([{"role": "user", "content": "Captured request input"}]);
    let request_json = json!({
        "model": "gpt-4.1",
        "input": request_input
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_action = filter.on_request_body(&mut ctx, &mut request_body, true).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request body phase should capture input and continue"
    );


    let response_json = json!({
        "id": "resp_no_echoed_input",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "content": "Stored output"}]
    });
    let mut response_body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));

    ctx.current_filter_id = Some(7);
    let response_action = filter.on_response_body(&mut ctx, &mut response_body, true).unwrap();
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response body phase should persist and continue"
    );

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_no_echoed_input")
        .await
        .expect("get_response should succeed")
        .expect("record should exist after persist");

    assert_eq!(
        record.response_object, response_json,
        "stored response object should remain the backend response"
    );
    assert_eq!(
        record.input, request_input,
        "stored input should come from the original request"
    );
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Captured request input"},
            {"type": "message", "content": "Stored output"}
        ]),
        "stored messages should combine request input with response output"
    );
}

// `streaming_record_uses_request_input_when_state_messages_are_empty` and
// `streaming_record_preserves_mcp_metadata_from_persisted_messages` moved to the
// service layer: `crate::service::responses` tests exercise `build_record`
// directly against plain values.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_persists_after_format_request_body_classification() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_persists_after_format_request_body_classification");

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: router
  routes:
    - path_prefix: "/"
      cluster: test-backend
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: test-backend
      endpoints: ["127.0.0.1:3001"]
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(file_store_registry(&db_url).await);

    let request_json = json!({
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "Hello"}]
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut request_body, true)
        .await
        .unwrap();
    assert!(
        matches!(request_body_action, FilterAction::Release),
        "format classifier should release the buffered request body"
    );
    assert_eq!(
        ctx.get_metadata("openai_responses_format.format"),
        Some("openai_responses"),
        "format classifier should write metadata before store filter runs"
    );
    ctx.buffered_request_body = request_body.clone();

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request phase should continue after initializing the store"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    let response_action = pipeline.execute_http_response(&mut ctx).await.unwrap();
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response phase should continue and arm persistence buffering"
    );
    ctx.response_header = None;

    let response_json = json!({
        "id": "resp_pipeline",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "content": "Hi"}]
    });
    let mut response_body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let response_body_action = pipeline
        .execute_http_response_body(&mut ctx, &mut response_body, true)
        .unwrap();
    assert!(
        matches!(response_body_action, FilterAction::Continue),
        "response body phase should persist and continue"
    );

    let store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_pipeline")
        .await
        .unwrap()
        .expect("pipeline should persist the response after body classification");
    assert_eq!(record.response_object, response_json);
    assert_eq!(record.input, request_json["input"]);
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Hello"},
            {"type": "message", "content": "Hi"}
        ])
    );

    drop(store);
    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

#[cfg(feature = "openai-conversations")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_persists_chunked_response_with_unarmed_conversations_filter() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_persists_chunked_with_conversations");

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: openai_conversations
  backend: sqlite
  database_url: "{db_url}"
  conversations_table: test_conversations_api
  items_table: test_conversation_items
"#
    ))
    .unwrap();
    let mut registry = crate::test_utils::make_ai_registry();
    praxis_filter::register_filters!(
        @register registry,
        http "openai_conversations" => crate::openai::OpenaiConversationsFilter::from_config
    );
    let pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    // Provision the store the registry-only filter resolves at request time.
    ctx.extensions.insert(file_store_registry(&db_url).await);
    let request_json = json!({
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "Hello"}]
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    assert!(matches!(
        pipeline
            .execute_http_request_body(&mut ctx, &mut request_body, true)
            .await
            .unwrap(),
        FilterAction::Release
    ));
    assert!(matches!(
        pipeline.execute_http_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    assert!(matches!(
        pipeline.execute_http_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.response_header = None;

    let response_json = json!({
        "id": "resp_chunked_pipeline",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "content": "Hi"}]
    });
    let response_bytes = serde_json::to_vec(&response_json).unwrap();
    let split_at = response_bytes.len() / 2;
    let full_body = Bytes::from(response_bytes);

    let mut first_body = Some(full_body.slice(..split_at));
    let first_action = pipeline
        .execute_http_response_body(&mut ctx, &mut first_body, false)
        .unwrap();

    // `StreamBuffer` releases the first chunk unchanged when a filter returns
    // `Release`; subsequent callbacks then receive only the tail. When every
    // filter continues, the protocol retains the first chunk and presents the
    // frozen aggregate at EOS. Mirror both paths so the persistence assertion
    // catches a premature release rather than merely checking the action.
    let eos_body = if matches!(&first_action, FilterAction::Release) {
        full_body.slice(split_at..)
    } else {
        full_body.clone()
    };
    let mut second_body = Some(eos_body);
    assert!(matches!(
        pipeline
            .execute_http_response_body(&mut ctx, &mut second_body, true)
            .unwrap(),
        FilterAction::Continue
    ));

    let store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_chunked_pipeline")
        .await
        .unwrap()
        .expect("chunked response must be persisted when Conversations is unarmed");
    assert_eq!(record.response_object, response_json);
    assert!(matches!(&first_action, FilterAction::Continue));

    drop(store);
    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_persists_streaming_response_from_accumulated_state() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_persists_streaming_response_from_accumulated_state");

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: router
  routes:
    - path_prefix: "/"
      cluster: test-backend
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: test-backend
      endpoints: ["127.0.0.1:3001"]
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(file_store_registry(&db_url).await);

    let request_json = json!({
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "Hello"}],
        "stream": true
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut request_body, true)
        .await
        .unwrap();
    assert!(
        matches!(request_body_action, FilterAction::Release),
        "format classifier should release the buffered request body"
    );
    assert_eq!(
        ctx.get_metadata("openai_responses_format.stream"),
        Some("true"),
        "format classifier should detect stream=true"
    );
    ctx.buffered_request_body = request_body.clone();

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request phase should continue and initialize the store for streaming"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    let response_action = pipeline.execute_http_response(&mut ctx).await.unwrap();
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response phase should continue for event-stream content type"
    );
    ctx.response_header = None;

    let mut chunk = Some(Bytes::from_static(b"event: response.output_text.delta\ndata: {}\n\n"));
    let chunk_action = pipeline
        .execute_http_response_body(&mut ctx, &mut chunk, false)
        .unwrap();
    assert!(
        matches!(chunk_action, FilterAction::Release),
        "intermediate streaming chunks should release immediately"
    );

    let response_json = json!({
        "id": "resp_stream_pipeline",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "Streamed reply"}]
    });
    ctx.extensions.insert(ResponsesState {
        response_object: response_json.clone(),
        persisted_messages: vec![json!({"role": "user", "content": "Hello"})],
        ..Default::default()
    });

    let mut eos_body: Option<Bytes> = None;
    let eos_action = pipeline
        .execute_http_response_body(&mut ctx, &mut eos_body, true)
        .unwrap();
    assert!(
        matches!(eos_action, FilterAction::Continue),
        "EOS should persist from accumulated state and continue"
    );

    let store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_stream_pipeline")
        .await
        .unwrap()
        .expect("streaming pipeline should persist the response from accumulated ResponsesState");
    assert_eq!(record.response_object, response_json);
    assert_eq!(
        record.input, request_json["input"],
        "stored input should come from the original streaming request"
    );
    assert_eq!(
        record.messages,
        json!([
            {"role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Streamed reply"}
        ]),
        "stored messages should combine persisted input history with streamed output"
    );

    drop(store);
    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_non_responses_post_does_not_open_sqlite_store() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_non_responses_post_does_not_open_sqlite_store");

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: router
  routes:
    - path_prefix: "/"
      cluster: test-backend
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: test-backend
      endpoints: ["127.0.0.1:3001"]
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);

    let request_json = json!({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut request_body, true)
        .await
        .unwrap();
    assert!(
        matches!(request_body_action, FilterAction::Release),
        "format classifier should release the buffered request body"
    );
    assert_eq!(
        ctx.get_metadata("openai_responses_format.format"),
        Some("openai_chat_completions"),
        "format classifier should mark Chat Completions traffic"
    );
    ctx.buffered_request_body = request_body.clone();

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request phase should continue without opening the response store"
    );
    assert!(
        !db_path.exists(),
        "non-Responses POST should not create the SQLite response store file"
    );

    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_persists_rehydrated_messages_when_response_omits_input() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_persists_rehydrated_messages");
    let seeded_store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    seeded_store
        .upsert_response(&ResponseRecord {
            id: "resp_prev".to_owned(),
            owner: crate::test_utils::test_owner("default"),
            created_at: 1_719_800_000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_prev",
                "created_at": 1_719_800_000,
                "model": "gpt-4.1",
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
- filter: router
  routes:
    - path_prefix: "/"
      cluster: test-backend
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: openai_responses_rehydrate
- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: test-backend
      endpoints: ["127.0.0.1:3001"]
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
    pipeline.add_pipeline_extension(Box::new(file_store_registry(&db_url).await));

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    pipeline.prepare_extensions(&mut ctx.extensions);

    let request_json = json!({
        "model": "gpt-4.1",
        "input": "What next?",
        "previous_response_id": "resp_prev"
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut request_body, true)
        .await
        .unwrap();
    assert!(
        matches!(request_body_action, FilterAction::Release),
        "request body phase should classify, register the store, and rehydrate"
    );
    ctx.buffered_request_body = request_body.clone();

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request phase should continue after pre-read rehydration"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    let response_action = pipeline.execute_http_response(&mut ctx).await.unwrap();
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response phase should arm persistence buffering"
    );
    ctx.response_header = None;

    let response_json = json!({
        "id": "resp_next",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "Next answer"}]
    });
    let mut response_body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let response_body_action = pipeline
        .execute_http_response_body(&mut ctx, &mut response_body, true)
        .unwrap();
    assert!(
        matches!(response_body_action, FilterAction::Continue),
        "response body phase should persist and continue"
    );

    let store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_next")
        .await
        .unwrap()
        .expect("pipeline should persist the rehydrated response");
    assert_eq!(
        record.input, request_json["input"],
        "stored input should remain the current request input"
    );
    assert_eq!(
        record.messages,
        json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi"},
            {"type": "message", "role": "user", "content": "What next?"},
            {"type": "message", "role": "assistant", "content": "Next answer"}
        ]),
        "stored messages should preserve previous turns, current input, and output"
    );

    drop(store);
    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pipeline_persists_fallback_mcp_metadata_for_future_rehydrate() {
    let (db_url, db_path) = temp_sqlite_url("pipeline_persists_fallback_mcp_metadata");
    let seeded_store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    seeded_store
        .upsert_response(&ResponseRecord {
            id: "resp_prev".to_owned(),
            owner: crate::test_utils::test_owner("default"),
            created_at: 1_719_800_000,
            model: "gpt-4.1".to_owned(),
            response_object: json!({
                "id": "resp_prev",
                "created_at": 1_719_800_000,
                "model": "gpt-4.1",
                "status": "completed",
                "output": [
                    {
                        "id": "mcpl_prev",
                        "type": "mcp_list_tools",
                        "server_label": "weather-server",
                        "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]
                    },
                    {"type": "message", "role": "assistant", "content": "Tools loaded"}
                ]
            }),
            input: json!("Hello"),
            messages: json!([]),
        })
        .await
        .unwrap();
    drop(seeded_store);

    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(&format!(
        r#"
- filter: openai_responses_format
- filter: router
  routes:
    - path_prefix: "/"
      cluster: test-backend
- filter: openai_response_store
  backend: sqlite
  database_url: "{db_url}"
  responses_table: test_responses
  conversations_table: test_conversations
- filter: openai_responses_rehydrate
- filter: load_balancer
  cluster_source: bound_upstream
  clusters:
    - name: test-backend
      endpoints: ["127.0.0.1:3001"]
"#
    ))
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
    pipeline.add_pipeline_extension(Box::new(file_store_registry(&db_url).await));

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    pipeline.prepare_extensions(&mut ctx.extensions);

    let request_json = json!({
        "model": "gpt-4.1",
        "input": "What next?",
        "previous_response_id": "resp_prev"
    });
    let mut request_body = Some(Bytes::from(serde_json::to_vec(&request_json).unwrap()));
    let request_body_action = pipeline
        .execute_http_request_body(&mut ctx, &mut request_body, true)
        .await
        .unwrap();
    assert!(
        matches!(request_body_action, FilterAction::Release),
        "request body phase should classify, register the store, and rehydrate"
    );
    ctx.buffered_request_body = request_body.clone();

    let request_action = pipeline.execute_http_request(&mut ctx).await.unwrap();
    assert!(
        matches!(request_action, FilterAction::Continue),
        "request phase should continue after pre-read rehydration"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);
    let response_action = pipeline.execute_http_response(&mut ctx).await.unwrap();
    assert!(
        matches!(response_action, FilterAction::Continue),
        "response phase should arm persistence buffering"
    );
    ctx.response_header = None;

    let response_json = json!({
        "id": "resp_next",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": "Next answer"}]
    });
    let mut response_body = Some(Bytes::from(serde_json::to_vec(&response_json).unwrap()));
    let response_body_action = pipeline
        .execute_http_response_body(&mut ctx, &mut response_body, true)
        .unwrap();
    assert!(
        matches!(response_body_action, FilterAction::Continue),
        "response body phase should persist and continue"
    );

    let store = SqliteResponseStore::new(&db_url, "test_responses", "test_conversations", None, None, None)
        .await
        .unwrap();
    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_next")
        .await
        .unwrap()
        .expect("pipeline should persist the rehydrated response");
    assert_eq!(
        record.messages,
        json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {
                "id": "mcpl_prev",
                "type": "mcp_list_tools",
                "server_label": "weather-server",
                "tools": [{"name": "get_weather", "description": "d", "input_schema": {}}]
            },
            {"type": "message", "role": "assistant", "content": "Tools loaded"},
            {"type": "message", "role": "user", "content": "What next?"},
            {"type": "message", "role": "assistant", "content": "Next answer"}
        ]),
        "stored messages should preserve fallback MCP metadata for later continuations"
    );

    drop(store);
    drop(pipeline);
    cleanup_sqlite_file(&db_path);
}

// -----------------------------------------------------------------------------
// Postgres Config
// -----------------------------------------------------------------------------

fn postgres_config_yaml(database_url: &str, extra: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&format!(
        r#"
backend: postgres
database_url: "{database_url}"
responses_table: responses
conversations_table: conversations
{extra}
"#
    ))
    .unwrap()
}

#[test]
fn valid_postgres_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let filter = ResponseStoreFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_response_store",
        "postgres config should parse successfully"
    );
}

#[test]
fn postgres_config_accepts_postgresql_scheme() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgresql://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "postgresql:// scheme should be accepted");
}

#[test]
fn postgres_config_rejects_loopback_database_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@127.0.0.1:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "loopback postgres hosts should be rejected by default");
}

#[test]
fn postgres_config_rejects_legacy_ipv4_local_database_hosts() {
    for host in [
        "127.1",
        "2130706433",
        "0x7f.0.0.1",
        "0177.0.0.1",
        "0",
        "0xa9fea9fe",
        "0x0a000005",
    ] {
        let yaml = postgres_config_yaml(&format!("postgres://user:pass@{host}:5432/praxis"), "");
        let result = ResponseStoreFilter::from_config(&yaml);
        assert!(
            result.is_err(),
            "legacy IPv4 local-sensitive postgres host should be rejected by default: {host}"
        );
    }
}

#[test]
fn postgres_config_rejects_localhost_database_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@LOCALHOST.:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "localhost postgres hosts should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_ipv6_loopback_database_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@[::1]:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "IPv6 loopback postgres hosts should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_link_local_database_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@169.254.169.254:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "link-local postgres hosts should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_private_database_hosts() {
    for host in ["10.0.0.5", "172.16.0.1", "192.168.1.10", "[fd00::1]"] {
        let yaml = postgres_config_yaml(&format!("postgres://user:pass@{host}:5432/praxis"), "");
        let result = ResponseStoreFilter::from_config(&yaml);
        assert!(
            result.is_err(),
            "private postgres hosts should be rejected by default: {host}"
        );
    }
}

#[test]
fn postgres_config_rejects_dns_database_hosts_without_private_database_url_opt_in() {
    let yaml = postgres_config_yaml("postgres://user:pass@db.example.net:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "DNS postgres hosts should be rejected by default to avoid DNS rebinding"
    );
}

#[test]
fn postgres_config_allows_dns_database_hosts_with_private_database_url_opt_in() {
    let yaml = postgres_config_yaml(
        "postgres://user:pass@db.example.net:5432/praxis",
        "allow_private_database_url: true",
    );
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit private database URL opt-in should allow DNS hosts"
    );
}

#[test]
fn postgres_config_rejects_unspecified_database_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@0.0.0.0:5432/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "unspecified postgres hosts should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_hostaddr_loopback_override() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?hostaddr=127.0.0.1"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "hostaddr loopback override should be rejected");
}

#[test]
fn postgres_config_rejects_host_loopback_override() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?host=127.0.0.1"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "host loopback override should be rejected");
}

#[test]
fn postgres_config_rejects_legacy_ipv4_host_override() {
    for host in [
        "127.1",
        "2130706433",
        "0x7f.0.0.1",
        "0177.0.0.1",
        "0",
        "0xa9fea9fe",
        "0x0a000005",
    ] {
        let yaml = postgres_config_yaml(&format!("postgres://user:pass@1.2.3.4:5432/praxis?host={host}"), "");
        let result = ResponseStoreFilter::from_config(&yaml);
        assert!(
            result.is_err(),
            "legacy IPv4 local-sensitive host override should be rejected by default: {host}"
        );
    }
}

#[test]
fn postgres_config_rejects_host_localhost_override() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?host=localhost"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "host localhost override should be rejected");
}

#[test]
fn postgres_config_rejects_mixed_case_host_query_as_missing_explicit_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres:///?HoSt=1.2.3.4"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "mixed-case host query should not satisfy explicit host validation"
    );
}

#[test]
fn postgres_config_rejects_hostaddr_unspecified_override() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?hostaddr=::"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "hostaddr unspecified override should be rejected");
}

#[test]
fn postgres_config_rejects_ipv4_mapped_link_local_hostaddr() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?hostaddr=::ffff:169.254.169.254"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "IPv4-mapped metadata hostaddr should be rejected");
}

#[test]
fn postgres_config_allows_loopback_with_private_database_url_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@127.0.0.1:5432/praxis"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit private database URL opt-in should allow loopback"
    );
}

#[test]
fn postgres_config_allows_legacy_ipv4_with_private_database_url_opt_in() {
    let yaml = postgres_config_yaml(
        "postgres://user:pass@127.1:5432/praxis",
        "allow_private_database_url: true",
    );
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit private database URL opt-in should allow legacy IPv4 loopback"
    );
}

#[test]
fn postgres_config_allows_localhost_with_private_database_url_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@localhost:5432/praxis"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit private database URL opt-in should allow localhost"
    );
}

#[test]
fn postgres_config_allows_private_with_private_database_url_opt_in() {
    for host in ["10.0.0.5", "[fd00::1]"] {
        let yaml = postgres_config_yaml(
            &format!("postgres://user:pass@{host}:5432/praxis"),
            "allow_private_database_url: true",
        );
        let result = ResponseStoreFilter::from_config(&yaml);
        assert!(
            result.is_ok(),
            "explicit private database URL opt-in should allow private hosts: {host}"
        );
    }
}

#[test]
fn postgres_config_rejects_unspecified_with_private_database_url_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@0.0.0.0:5432/praxis"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "unspecified database hosts must remain blocked after the private-target opt-in"
    );
}

#[test]
fn postgres_config_rejects_socket_host_without_private_database_url_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?host=%2Fvar%2Frun%2Fpostgresql"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "Unix socket host override should require explicit opt-in"
    );
}

#[test]
fn postgres_config_allows_socket_host_with_private_database_url_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?host=%2Fvar%2Frun%2Fpostgresql"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit private database URL opt-in should allow Unix sockets"
    );
}

#[test]
fn postgres_config_allows_socket_host_with_empty_authority_and_port_with_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@:5433/praxis?host=%2Fvar%2Frun%2Fpostgresql"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "query host should supply the socket target when authority host is empty"
    );
}

#[test]
fn postgres_config_rejects_socket_host_path_traversal_with_opt_in() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?host=%2Fvar%2Frun%2F..%2Fpostgresql"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "Unix socket host traversal should be rejected even with opt-in"
    );
}

#[test]
fn postgres_config_rejects_missing_explicit_host() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@/praxis"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "postgres database_url should not rely on environment/default host"
    );
}

#[test]
fn postgres_config_with_ssl_mode_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: require
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "postgres config with ssl_mode should parse");
}

#[test]
fn postgres_config_with_ssl_root_cert_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-ca
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "postgres config with ssl_mode and ssl_root_cert should parse"
    );
}

#[test]
fn postgres_config_with_url_verify_sslmode_and_ssl_root_cert_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full"
responses_table: responses
conversations_table: conversations
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "URL sslmode=verify-full should allow ssl_root_cert");
}

#[test]
fn postgres_config_with_url_sslrootcert_and_verified_sslmode_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full&sslrootcert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "URL sslrootcert should parse when effective sslmode verifies certificates"
    );
}

#[test]
fn postgres_config_with_url_ssl_root_cert_alias_and_verified_sslmode_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?ssl-mode=verify-ca&ssl-root-cert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "URL ssl-root-cert alias should parse when effective sslmode verifies certificates"
    );
}

#[test]
fn postgres_config_accepts_ssl_root_cert_without_explicit_ssl_mode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "ssl_root_cert should be accepted when default ssl_mode is verify-full"
    );
}

#[test]
fn postgres_config_rejects_ssl_root_cert_with_non_verified_ssl_mode() {
    for ssl_mode in ["ssl_mode: disable", "ssl_mode: prefer", "ssl_mode: require"] {
        let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
            r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
{ssl_mode}
ssl_root_cert: /path/to/ca.pem
"#
        ))
        .unwrap();
        let result = ResponseStoreFilter::from_config(&yaml);
        assert!(
            result.is_err(),
            "ssl_root_cert should require verify-ca or verify-full, got {ssl_mode:?}"
        );
    }
}

#[test]
fn postgres_config_rejects_ssl_root_cert_with_non_verified_url_sslmode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require"
responses_table: responses
conversations_table: conversations
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "URL sslmode=require should not allow ssl_root_cert");
}

#[test]
fn postgres_config_mixed_case_url_sslmode_falls_through_to_verified_default() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?SSLMODE=verify-full&sslrootcert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "mixed-case SSLMODE is not recognized but default verify-full satisfies sslrootcert"
    );
}

#[test]
fn postgres_config_rejects_url_sslrootcert_without_verified_ssl_mode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&sslrootcert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "URL sslrootcert should require verify-ca or verify-full"
    );
}

#[test]
fn postgres_config_rejects_url_sslrootcert_when_last_sslmode_is_not_verified() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full&sslmode=require&sslrootcert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "last URL sslmode should match the effective sqlx option"
    );
}

#[test]
fn postgres_config_explicit_ssl_mode_overrides_url_sslmode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full"
responses_table: responses
conversations_table: conversations
ssl_mode: require
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "explicit ssl_mode=require should override URL sslmode=verify-full"
    );
}

#[test]
fn postgres_config_explicit_verified_ssl_mode_allows_url_sslrootcert() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&sslrootcert=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "explicit verified ssl_mode should override URL sslmode=require"
    );
}

#[test]
fn postgres_config_rejects_ssl_root_cert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-ca
ssl_root_cert: ../ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl_root_cert path traversal should be rejected");
}

#[test]
fn postgres_config_rejects_url_sslrootcert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full&sslrootcert=../ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "URL sslrootcert path traversal should be rejected");
}

#[test]
fn postgres_config_rejects_url_encoded_sslrootcert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full&sslrootcert=%2e%2e%2fca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "URL sslrootcert percent-encoded path traversal should be rejected"
    );
}

#[test]
fn postgres_config_rejects_url_sslcert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&sslcert=../client.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "URL sslcert path traversal should be rejected");
}

#[test]
fn postgres_config_rejects_url_sslkey_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&sslkey=../client.key"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "URL sslkey path traversal should be rejected");
}

#[test]
fn postgres_config_rejects_long_responses_table() {
    let responses_table = "r".repeat(64);
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: {responses_table}
conversations_table: conversations
"#
    ))
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "postgres responses_table above 63 bytes should be rejected"
    );
}

#[test]
fn postgres_config_rejects_long_conversations_table_for_index_name() {
    let conversations_table = "c".repeat(50);
    let yaml: serde_yaml::Value = serde_yaml::from_str(&format!(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: {conversations_table}
"#
    ))
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "postgres conversations_table above index-safe length should be rejected"
    );
}

#[test]
fn sqlite_config_rejects_ssl_mode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
ssl_mode: require
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl_mode should be rejected for sqlite backend");
}

#[test]
fn sqlite_config_rejects_ssl_root_cert() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
ssl_root_cert: /path/to/ca.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl_root_cert should be rejected for sqlite backend");
}

#[test]
fn sqlite_config_rejects_ssl_client_cert() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
ssl_client_cert: /path/to/client.pem
ssl_client_key: /path/to/client.key
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl_client_cert should be rejected for sqlite backend");
}

#[test]
fn sqlite_config_rejects_require_certificate_authentication() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
require_certificate_authentication: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "require_certificate_authentication should be rejected for sqlite backend"
    );
}

#[test]
fn sqlite_config_rejects_allow_private_database_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
allow_private_database_url: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "allow_private_database_url should be rejected for sqlite backend"
    );
}

#[test]
fn postgres_config_accepts_client_cert_and_key() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
ssl_root_cert: /etc/pki/ca.pem
ssl_client_cert: /etc/pki/client.pem
ssl_client_key: /etc/pki/client.key
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "client cert + key with verify-full should parse");
}

#[test]
fn postgres_config_rejects_client_cert_without_key() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
ssl_client_cert: /etc/pki/client.pem
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "client cert without key should be rejected");
}

#[test]
fn postgres_config_rejects_client_cert_with_unverified_mode() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: require
ssl_client_cert: /etc/pki/client.pem
ssl_client_key: /etc/pki/client.key
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "client cert with unverified ssl_mode should be rejected"
    );
}

#[test]
fn postgres_config_accepts_compliance_profile() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
ssl_root_cert: /etc/pki/ca.pem
ssl_client_cert: /etc/pki/client.pem
ssl_client_key: /etc/pki/client.key
require_certificate_authentication: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "well-formed compliance profile should parse");
}

#[test]
fn postgres_config_compliance_rejects_password_in_url() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user:secret@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
ssl_client_cert: /etc/pki/client.pem
ssl_client_key: /etc/pki/client.key
require_certificate_authentication: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "compliance profile must reject a password in database_url"
    );
}

#[test]
fn postgres_config_compliance_requires_verify_full() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-ca
ssl_root_cert: /etc/pki/ca.pem
ssl_client_cert: /etc/pki/client.pem
ssl_client_key: /etc/pki/client.key
require_certificate_authentication: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "compliance profile must require ssl_mode verify-full");
}

#[test]
fn postgres_config_compliance_requires_client_cert() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://cert-user@1.2.3.4:5432/praxis"
responses_table: responses
conversations_table: conversations
ssl_mode: verify-full
require_certificate_authentication: true
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "compliance profile must require a client certificate and key"
    );
}

#[test]
fn postgres_url_without_postgres_scheme_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "non-postgres URL should be rejected for postgres backend"
    );
}

#[test]
fn postgres_config_rejects_percent_encoded_loopback() {
    let yaml = postgres_config_yaml("postgres://user@%31%32%37.0.0.1/db", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "percent-encoded loopback host should be rejected");
}

#[test]
fn postgres_config_rejects_octal_loopback() {
    let yaml = postgres_config_yaml("postgres://user@0177.0.0.1/db", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "octal loopback host should be rejected via legacy IPv4 parsing"
    );
}

#[test]
fn postgres_config_rejects_hex_loopback() {
    let yaml = postgres_config_yaml("postgres://user@0x7f.0.0.1/db", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "hex loopback host should be rejected via legacy IPv4 parsing"
    );
}

#[test]
fn postgres_config_rejects_ipv6_bracketed_loopback() {
    let yaml = postgres_config_yaml("postgres://user@[::1]/db", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "bracketed IPv6 loopback host should be rejected");
}

#[test]
fn postgres_config_rejects_socket_path_with_traversal() {
    let yaml = postgres_config_yaml(
        "postgres://user@1.2.3.4/db?host=%2Fvar%2Frun%2F..%2F..%2Fetc%2Fdb",
        "allow_private_database_url: true",
    );
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "socket path with directory traversal should be rejected"
    );
}

#[test]
fn postgres_config_rejects_hostaddr_param_with_loopback() {
    let yaml = postgres_config_yaml("postgres://user@8.8.8.8/db?hostaddr=127.0.0.1", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "hostaddr query param with loopback should be rejected");
}

#[test]
fn sqlite_mode_memory_query_param_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite:///path?mode=memory"
responses_table: responses
conversations_table: conversations
"#,
    )
    .expect("YAML should parse");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "sqlite URL with mode=memory query param should be accepted as in-memory"
    );
}

#[test]
fn postgres_config_rejects_empty_host() {
    let yaml = postgres_config_yaml("postgres:///mydb", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "postgres URL with no host should be rejected");
}

// -----------------------------------------------------------------------------
// GET /v1/responses/{id}
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_returns_200_when_found() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_found", "default", json!([{"id": "item_1", "type": "message"}])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_found");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "should return 200 for found response");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        body["status"], "completed",
        "body should contain the stored response_object"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_returns_404_when_not_found() {
    let filter = make_filter();
    let registry = store_registry().await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_nonexistent");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 404, "should return 404 for missing response");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "error type should be invalid_request_error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_tenant_isolation() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_tenant", "tenant_a", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_tenant");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 404,
        "should return 404 when response belongs to a different tenant"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_endpoints_hide_same_tenant_cross_owner_resources() {
    let filter = make_filter();
    let owner = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "alice").unwrap();
    let other = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "bob").unwrap();
    let registry = init_store_and_seed_owner(
        "resp_owner_private",
        owner.clone(),
        json!([{"id": "item_private", "type": "message"}]),
    )
    .await;

    for (method, path) in [
        (http::Method::GET, "/v1/responses/resp_owner_private"),
        (http::Method::GET, "/v1/responses/resp_owner_private/input_items"),
        (http::Method::DELETE, "/v1/responses/resp_owner_private"),
    ] {
        let req = crate::test_utils::make_request(method, path);
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.extensions.insert(other.clone());
        ctx.extensions.insert(registry.clone());
        let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());
        assert_eq!(rejection.status, 404, "wrong-owner access must look absent: {path}");
    }

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_owner_private");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(owner);
    ctx.extensions.insert(registry);
    let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(rejection.status, 200, "wrong-owner delete must not mutate the response");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_state_access_fails_closed_without_an_owner() {
    let filter = make_filter();
    let registry = store_registry().await;
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_private");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(registry);

    let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());

    assert_eq!(rejection.status, 401);
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_state_owner");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unauthorized_and_missing_responses_are_externally_identical() {
    let owned_filter = make_filter();
    let empty_filter = make_filter();
    let owner = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "alice").unwrap();
    let other = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "bob").unwrap();
    let seeded = init_store_and_seed_owner("resp_indistinguishable", owner, json!([])).await;
    let empty = store_registry().await;
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_indistinguishable");

    let mut unauthorized_ctx = crate::test_utils::make_filter_context(&req);
    unauthorized_ctx.extensions.insert(other.clone());
    unauthorized_ctx.extensions.insert(seeded);
    let unauthorized = expect_reject(owned_filter.on_request(&mut unauthorized_ctx).await.unwrap());
    let mut missing_ctx = crate::test_utils::make_filter_context(&req);
    missing_ctx.extensions.insert(other);
    missing_ctx.extensions.insert(empty);
    let missing = expect_reject(empty_filter.on_request(&mut missing_ctx).await.unwrap());

    assert_eq!(unauthorized.status, missing.status);
    assert_eq!(unauthorized.headers, missing.headers);
    assert_eq!(unauthorized.body, missing.body);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_trailing_slash_handled() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_slash", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_slash/");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 200,
        "trailing slash should be stripped and response found"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_unrelated_path_continues() {
    let filter = make_filter();

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/chat/completions");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "GET to unrelated path should continue"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_stream_true_without_log_returns_400() {
    let filter = make_filter();
    // Seeded with no replay event log (a legacy or non-streamed record), so a
    // replay request is a client error, not a 200 empty stream.
    let registry = init_store_and_seed("resp_stream", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_stream?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 400, "replay without an event log should return 400");
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no replayable event stream"),
        "message should explain the response is not replayable: {body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_rejects_include() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_inc", "default", json!([])).await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_inc?include%5B%5D=reasoning.encrypted_content",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 400, "include[] should return 400");
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_accepts_stream_false() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_sf", "default", json!([{"id": "item_1", "type": "message"}])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_sf?stream=false");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "stream=false should return 200");
    assert_has_json_content_type(&rejection);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_response_no_query_still_returns_200() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_nq", "default", json!([{"id": "item_1", "type": "message"}])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_nq");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "no query should still return 200");
}

// -----------------------------------------------------------------------------
// GET /v1/responses/{id}?stream=true — SSE replay
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_returns_events_in_order_ending_with_terminal() {
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    let registry = seed_response_with_events(
        "resp_replay",
        owner.clone(),
        vec![
            event_record("resp_replay", &owner, 0, "response.created", false),
            event_record("resp_replay", &owner, 1, "response.output_text.delta", false),
            event_record("resp_replay", &owner, 2, "response.completed", true),
        ],
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_replay?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let streaming = expect_streaming(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(streaming.status, 200, "a replayable log streams with 200");
    assert_eq!(
        streaming.headers.get(http::header::CONTENT_TYPE).unwrap(),
        "text/event-stream",
        "replay uses the SSE content type"
    );
    assert_eq!(
        streaming.headers.get(http::header::CACHE_CONTROL).unwrap(),
        "no-store",
        "replay must not be cached"
    );

    let body = drain_replay_body(streaming).await;
    let text = String::from_utf8(body).unwrap();
    let types: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("event: ")).collect();
    assert_eq!(
        types,
        vec!["response.created", "response.output_text.delta", "response.completed"],
        "events replay in original order, terminal last: {text}"
    );
    assert!(
        text.contains("\"sequence_number\":2"),
        "the terminal event payload is replayed verbatim: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_in_legacy_responses_format_pipeline_streams_not_buffers() {
    // A legacy `openai_responses_format` classifier pipeline marks the request
    // Responses-format with `stream=false` (the flag is read from a POST body the
    // GET never carries). Without forcing streaming mode for replay GETs, the
    // buffered-mode override would run and the runtime would reject the streaming
    // replay body with a 500. The replay must still stream and the response body
    // mode must stay `Stream`.
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    let registry = seed_response_with_events(
        "resp_legacy_replay",
        owner.clone(),
        vec![
            event_record("resp_legacy_replay", &owner, 0, "response.created", false),
            event_record("resp_legacy_replay", &owner, 1, "response.completed", true),
        ],
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_legacy_replay?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);
    // The legacy classifier promotes these facts before the store filter runs.
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");

    let streaming = expect_streaming(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(
        streaming.status, 200,
        "the replay still streams under a legacy classifier pipeline"
    );
    assert_eq!(
        streaming.headers.get(http::header::CONTENT_TYPE).unwrap(),
        "text/event-stream",
        "replay uses the SSE content type"
    );
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::Stream,
        "a replay GET must force streaming mode instead of the buffered Responses-format override"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_starting_after_skips_earlier_events() {
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    let registry = seed_response_with_events(
        "resp_replay_cursor",
        owner.clone(),
        vec![
            event_record("resp_replay_cursor", &owner, 0, "response.created", false),
            event_record("resp_replay_cursor", &owner, 1, "response.output_text.delta", false),
            event_record("resp_replay_cursor", &owner, 2, "response.completed", true),
        ],
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_replay_cursor?stream=true&starting_after=1",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let streaming = expect_streaming(filter.on_request(&mut ctx).await.unwrap());
    let text = String::from_utf8(drain_replay_body(streaming).await).unwrap();
    let types: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("event: ")).collect();
    assert_eq!(
        types,
        vec!["response.completed"],
        "starting_after=1 returns only events with sequence_number > 1: {text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_missing_response_returns_404() {
    let filter = make_filter();
    let registry = store_registry().await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_absent?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(rejection.status, 404, "replay of a missing response is a 404");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_incomplete_log_returns_400() {
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    // Events exist but none is terminal: the log is incomplete, so it must never
    // be served as a complete replay.
    let registry = seed_response_with_events(
        "resp_incomplete",
        owner.clone(),
        vec![
            event_record("resp_incomplete", &owner, 0, "response.created", false),
            event_record("resp_incomplete", &owner, 1, "response.output_text.delta", false),
        ],
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_incomplete?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(rejection.status, 400, "an incomplete log is not replayable");
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no replayable event stream"),
        "message should explain the log is not replayable: {body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_cross_owner_returns_404() {
    let filter = make_filter();
    let owner = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "alice").unwrap();
    let registry = seed_response_with_events(
        "resp_private_replay",
        owner.clone(),
        vec![event_record(
            "resp_private_replay",
            &owner,
            0,
            "response.completed",
            true,
        )],
    )
    .await;

    // A different owner in the same tenant must not see the response or its log.
    let other = crate::StateOwner::from_trusted_parts("tenant-a", "issuer-a", "bob").unwrap();
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_private_replay?stream=true");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.extensions.insert(other);
    ctx.extensions.insert(registry);

    let rejection = expect_reject(filter.on_request(&mut ctx).await.unwrap());
    assert_eq!(rejection.status, 404, "another owner's replay log must look absent");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_truncated_by_concurrent_delete_errors() {
    // A concurrent DELETE can remove the event log after the first page streams but
    // before the terminal event is reached. The replay must surface an error rather
    // than a clean EOF that would hide the missing terminal frame from the client.
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    let (registry, store) = empty_registry().await;

    let record = ResponseRecord {
        id: "resp_truncated".to_owned(),
        owner: owner.clone(),
        created_at: 1000,
        model: "gpt-4.1".to_owned(),
        response_object: json!({"status": "completed"}),
        input: json!([]),
        messages: json!([{"role": "user", "content": "hello"}]),
    };
    store
        .upsert_response(&record)
        .await
        .expect("seed response should succeed");

    // Put the terminal event on a later page so the first page is non-terminal and
    // a second read is required to reach the terminal boundary.
    let total = u64::from(super::filter::REPLAY_PAGE_LIMIT) + 50;
    let terminal = total - 1;
    let events: Vec<_> = (0..total)
        .map(|seq| {
            let is_terminal = seq == terminal;
            let event_type = if is_terminal {
                "response.completed"
            } else {
                "response.output_text.delta"
            };
            event_record("resp_truncated", &owner, seq, event_type, is_terminal)
        })
        .collect();
    store
        .append_events(&owner, "resp_truncated", &events)
        .await
        .expect("seed replay events should succeed");

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_truncated?stream=true");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let mut streaming = expect_streaming(filter.on_request(&mut ctx).await.unwrap());

    // The first page streams non-terminal events; the terminal is still ahead.
    let first = streaming
        .body
        .next_chunk()
        .await
        .expect("first page should not error")
        .expect("first page should carry events");
    assert!(
        !std::str::from_utf8(&first).unwrap().contains("response.completed"),
        "the terminal event must not be on the first page"
    );

    // A concurrent DELETE removes the whole log before the terminal is reached.
    let deleted = store
        .delete_response(&owner, "resp_truncated")
        .await
        .expect("delete should succeed");
    assert!(deleted, "the seeded response is deleted");

    // The next read must error rather than ending cleanly without the terminal.
    let error = streaming
        .body
        .next_chunk()
        .await
        .expect_err("a log truncated before its terminal must error, not EOF");
    assert!(
        error.to_string().contains("truncated before its terminal event"),
        "the error explains the mid-replay truncation: {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_starting_after_terminal_returns_empty_body() {
    // A cursor at or beyond the terminal is a legitimate empty tail, not a
    // truncation: the replay ends cleanly with an empty body, never an error.
    let filter = make_filter();
    let owner = crate::test_utils::test_owner("default");
    let registry = seed_response_with_events(
        "resp_replay_tail",
        owner.clone(),
        vec![
            event_record("resp_replay_tail", &owner, 0, "response.created", false),
            event_record("resp_replay_tail", &owner, 1, "response.completed", true),
        ],
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_replay_tail?stream=true&starting_after=1",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let mut streaming = expect_streaming(filter.on_request(&mut ctx).await.unwrap());
    let chunk = streaming
        .body
        .next_chunk()
        .await
        .expect("a cursor beyond the terminal returns an empty body, not an error");
    assert!(chunk.is_none(), "no events remain after the terminal: {chunk:?}");
}

// -----------------------------------------------------------------------------
// GET /v1/responses/{id}/input_items
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_returns_200_when_found() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_items",
        "default",
        json!([
            {"id": "item_1", "type": "message", "content": "hello"},
            {"id": "item_2", "type": "message", "content": "world"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_items/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "should return 200 for input items");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["object"], "list", "should have list object type");
    assert_eq!(body["data"].as_array().unwrap().len(), 2, "should have 2 items");
    assert_eq!(body["has_more"], false, "should have no more items");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_returns_404_when_not_found() {
    let filter = make_filter();
    let registry = store_registry().await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_missing/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 404,
        "should return 404 when response not found for input_items"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_rejects_invalid_query_params() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_qp", "default", json!(["hello"])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_qp/input_items?limit=abc");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 400, "should return 400 for invalid limit");
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_rejects_key_only_query_param() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_ko", "default", json!(["hello"])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_ko/input_items?limit");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 400, "should return 400 for key-only param");
    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_with_limit_and_order() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_page",
        "default",
        json!([
            {"id": "item_1", "type": "message"},
            {"id": "item_2", "type": "message"},
            {"id": "item_3", "type": "message"},
            {"id": "item_4", "type": "message"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_page/input_items?limit=2&order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "should return 200 for paginated input items");

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 2, "should limit to 2 items");
    assert_eq!(body["has_more"], true, "should indicate more items exist");
    assert_eq!(
        body["first_id"], "item_1",
        "first_id should be item_1 in ascending order"
    );
    assert_eq!(body["last_id"], "item_2", "last_id should be item_2 in ascending order");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_with_cursor() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_cursor",
        "default",
        json!([
            {"id": "item_1", "type": "message"},
            {"id": "item_2", "type": "message"},
            {"id": "item_3", "type": "message"},
            {"id": "item_4", "type": "message"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_cursor/input_items?after=item_2&limit=2&order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "should return 200 for cursor-based pagination");

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "should return 2 items after cursor");
    assert_eq!(data[0]["id"], "item_3", "first item should be item_3 after item_2");
    assert_eq!(data[1]["id"], "item_4", "second item should be item_4");
    assert_eq!(body["has_more"], false, "should indicate no more items after this page");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_with_malformed_cursor_returns_400() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_bad_cursor",
        "default",
        json!([
            {"id": "item_1", "type": "message"},
            {"id": "item_2", "type": "message"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_bad_cursor/input_items?after=not-a-cursor",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 400, "malformed cursor should return 400");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "malformed cursor should return an invalid request error"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("invalid input_items cursor")),
        "error message should explain the invalid input_items cursor"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_last_id_falls_back_to_numeric_cursor_for_non_object_items() {
    let filter = make_filter();
    // Plain string entries can't carry a synthetic `id` (there's no
    // object to attach it to), so `last_id` must fall back to the
    // page's numeric cursor instead of staying `null`.
    let registry = init_store_and_seed("resp_non_object", "default", json!(["first", "second"])).await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_non_object/input_items?limit=1&order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["has_more"], true);
    assert_eq!(
        body["last_id"], "1",
        "last_id should fall back to the numeric cursor when the last item has no ID"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_pagination_usable_for_id_less_array_items() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_no_ids",
        "default",
        json!([
            {"type": "message", "role": "user", "content": "first"},
            {"type": "message", "role": "assistant", "content": "second"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_no_ids/input_items?limit=1&order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry.clone());
    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["has_more"], true, "second item should remain on the next page");
    let last_id = body["last_id"]
        .as_str()
        .expect("last_id must be a usable cursor, not null, even when input items have no ID field");

    let follow_up = crate::test_utils::make_request(
        http::Method::GET,
        &format!("/v1/responses/resp_no_ids/input_items?limit=1&order=asc&after={last_id}"),
    );
    let mut follow_up_ctx = crate::test_utils::make_owned_filter_context(&follow_up);
    follow_up_ctx.extensions.insert(registry);
    let follow_up_action = filter.on_request(&mut follow_up_ctx).await.unwrap();
    let follow_up_rejection = expect_reject(follow_up_action);
    assert_eq!(follow_up_rejection.status, 200);

    let follow_up_body: serde_json::Value =
        serde_json::from_slice(follow_up_rejection.body.as_deref().unwrap()).unwrap();
    let data = follow_up_body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "the after cursor should resolve to the second item");
    assert_eq!(data[0]["content"], "second");
    assert_eq!(follow_up_body["has_more"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_hides_compaction_items() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_compact_hidden",
        "default",
        json!([
            {"type": "compaction", "id": "compact_1", "encrypted_content": "c3VtbWFyeQ=="},
            {"type": "message", "role": "user", "content": "hello"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_compact_hidden/input_items?order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);
    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "compaction item should be hidden");
    assert_eq!(data[0]["content"], "hello");
    assert!(
        !data
            .iter()
            .any(|item| item.get("type").and_then(|t| t.as_str()) == Some("compaction")),
        "no compaction items should appear in input_items"
    );
}

// -----------------------------------------------------------------------------
// DELETE
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_existing_response_returns_200() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_del1", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_del1");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "existing response should return 200");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().expect("body should be present"))
        .expect("body should be valid JSON");
    assert_eq!(body["id"], "resp_del1", "body should contain the response id");
    assert_eq!(
        body["object"], "response.deleted",
        "body should have object=response.deleted"
    );
    assert_eq!(body["deleted"], true, "body should have deleted=true");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_nonexistent_response_returns_404() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_exists", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_missing");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 404, "nonexistent response should return 404");
    assert_has_json_content_type(&rejection);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().expect("body should be present"))
        .expect("body should be valid JSON");
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "error type should be invalid_request_error"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("resp_missing")),
        "error message should reference the missing id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_is_idempotent() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_idem", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_idem");
    let mut ctx1 = crate::test_utils::make_owned_filter_context(&req);
    ctx1.extensions.insert(registry.clone());
    let action1 = filter.on_request(&mut ctx1).await.unwrap();
    let r1 = expect_reject(action1);
    assert_eq!(r1.status, 200, "first delete should return 200");

    let mut ctx2 = crate::test_utils::make_owned_filter_context(&req);
    ctx2.extensions.insert(registry);
    let action2 = filter.on_request(&mut ctx2).await.unwrap();
    let r2 = expect_reject(action2);
    assert_eq!(r2.status, 404, "second delete should return 404");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_cross_tenant_returns_404() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_tenant", "tenant_a", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_tenant");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 404, "delete from wrong tenant should return 404");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_uses_trusted_owner_context() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_tmeta", "tenant_x", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_tmeta");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);
    ctx.extensions.insert(crate::test_utils::test_owner("tenant_x"));

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "delete with matching owner should return 200");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_rejects_when_store_unavailable() {
    let filter = make_filter();

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_gone");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 500,
        "DELETE should reject with 500 when store is unavailable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_continues_when_store_unavailable() {
    let filter = make_filter();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses/resp_abc/cancel");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /cancel should continue on_request when store is unavailable"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /cancel should continue on_response when store is unavailable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn input_tokens_continues_when_store_unavailable() {
    let filter = make_filter();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses/input_tokens");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /input_tokens should continue on_request when store is unavailable"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /input_tokens should continue on_response when store is unavailable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compact_continues_when_store_unavailable() {
    let filter = make_filter();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses/compact");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /compact should continue on_request when store is unavailable"
    );

    let mut resp = crate::test_utils::make_response();
    resp.headers
        .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
    ctx.response_header = Some(&mut resp);

    let action = filter.on_response(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "POST /compact should continue on_response when store is unavailable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_response_has_json_content_type() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_ct", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::DELETE, "/v1/responses/resp_ct");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_has_json_content_type(&rejection);
}

// -----------------------------------------------------------------------------
// extract_response_id
// -----------------------------------------------------------------------------

#[test]
fn extract_response_id_valid() {
    assert_eq!(
        super::filter::extract_response_id("/v1/responses/resp_abc"),
        Some("resp_abc"),
        "should extract ID from valid path"
    );
}

#[test]
fn extract_response_id_trailing_slash() {
    assert_eq!(
        super::filter::extract_response_id("/v1/responses/resp_abc/"),
        Some("resp_abc"),
        "should extract ID with trailing slash"
    );
}

#[test]
fn extract_response_id_no_id() {
    assert_eq!(
        super::filter::extract_response_id("/v1/responses"),
        None,
        "should return None without ID segment"
    );
}

#[test]
fn extract_response_id_sub_resource() {
    assert_eq!(
        super::filter::extract_response_id("/v1/responses/resp_abc/input_items"),
        None,
        "should return None for sub-resource path"
    );
}

#[test]
fn extract_response_id_unrelated_path() {
    assert_eq!(
        super::filter::extract_response_id("/v1/chat/completions"),
        None,
        "should return None for unrelated path"
    );
}

#[test]
fn extract_response_id_empty_id_segment() {
    assert_eq!(
        super::filter::extract_response_id("/v1/responses/"),
        None,
        "should return None for empty ID segment"
    );
}

// -----------------------------------------------------------------------------
// parse_query_params
// -----------------------------------------------------------------------------

#[test]
fn parse_query_params_empty() {
    let params = super::filter::parse_query_params(None).unwrap();
    assert!(params.cursor.is_none(), "cursor should be None for empty query");
    assert_eq!(params.limit, 20, "limit should default to 20");
    assert_eq!(params.order, Order::Descending, "order should default to Descending");
}

#[test]
fn parse_query_params_all_fields() {
    let params = super::filter::parse_query_params(Some("after=5&limit=10&order=asc")).unwrap();
    assert_eq!(
        params.cursor.as_deref(),
        Some("5"),
        "cursor should be parsed from after param"
    );
    assert_eq!(params.limit, 10, "limit should be parsed from query");
    assert_eq!(params.order, Order::Ascending, "order should be parsed as Ascending");
}

#[test]
fn parse_query_params_invalid_limit_rejected() {
    let err = super::filter::parse_query_params(Some("limit=abc")).unwrap_err();
    assert!(
        err.contains("Invalid value for 'limit'"),
        "should reject non-numeric limit: {err}"
    );
}

#[test]
fn parse_query_params_zero_limit_rejected() {
    let err = super::filter::parse_query_params(Some("limit=0")).unwrap_err();
    assert!(err.contains("must be between 1 and"), "should reject zero limit: {err}");
}

#[test]
fn parse_query_params_above_max_limit_rejected() {
    let err = super::filter::parse_query_params(Some("limit=101")).unwrap_err();
    assert!(
        err.contains("must be between 1 and"),
        "should reject above-max limit: {err}"
    );
}

#[test]
fn parse_query_params_decodes_percent_encoded_cursor() {
    let params = super::filter::parse_query_params(Some("after=item%5F1")).unwrap();
    assert_eq!(
        params.cursor.as_deref(),
        Some("item_1"),
        "percent-encoded cursor should be decoded"
    );
}

#[test]
fn parse_query_params_empty_after_rejected() {
    let err = super::filter::parse_query_params(Some("after=")).unwrap_err();
    assert!(
        err.contains("cursor must not be empty"),
        "should reject empty after: {err}"
    );
}

#[test]
fn parse_query_params_unknown_order_rejected() {
    let err = super::filter::parse_query_params(Some("order=random")).unwrap_err();
    assert!(
        err.contains("Invalid value for 'order'"),
        "should reject unknown order value: {err}"
    );
}

#[test]
fn parse_query_params_unknown_parameter_rejected() {
    let err = super::filter::parse_query_params(Some("foo=bar")).unwrap_err();
    assert!(
        err.contains("Unknown query parameter"),
        "should reject unknown parameter: {err}"
    );
}

#[test]
fn parse_query_params_boundary_limits_accepted() {
    let one = super::filter::parse_query_params(Some("limit=1")).unwrap();
    assert_eq!(one.limit, 1, "limit=1 should be accepted");
    let max = super::filter::parse_query_params(Some("limit=100")).unwrap();
    assert_eq!(max.limit, 100, "limit=100 should be accepted");
}

#[test]
fn parse_query_params_key_only_limit_rejected() {
    let err = super::filter::parse_query_params(Some("limit")).unwrap_err();
    assert!(err.contains("Missing value"), "should reject key-only limit: {err}");
}

#[test]
fn parse_query_params_key_only_order_rejected() {
    let err = super::filter::parse_query_params(Some("order")).unwrap_err();
    assert!(err.contains("Missing value"), "should reject key-only order: {err}");
}

#[test]
fn parse_query_params_key_only_unknown_ignored() {
    let params = super::filter::parse_query_params(Some("order=asc&foo")).unwrap();
    assert_eq!(
        params.order,
        Order::Ascending,
        "unknown key-only param should be ignored"
    );
}

// -----------------------------------------------------------------------------
// parse_get_response_query
// -----------------------------------------------------------------------------

#[test]
fn parse_get_response_query_no_query() {
    let parsed = super::filter::parse_get_response_query(None).unwrap();
    assert!(!parsed.stream, "no query string should default to non-stream");
    assert_eq!(parsed.starting_after, None, "no query string should have no cursor");
}

#[test]
fn parse_get_response_query_empty_query() {
    let parsed = super::filter::parse_get_response_query(Some("")).unwrap();
    assert!(!parsed.stream, "empty query string should default to non-stream");
}

#[test]
fn parse_get_response_query_stream_false_accepted() {
    let parsed = super::filter::parse_get_response_query(Some("stream=false")).unwrap();
    assert!(!parsed.stream, "stream=false should parse as non-stream");
}

#[test]
fn parse_get_response_query_stream_true_accepted() {
    let parsed = super::filter::parse_get_response_query(Some("stream=true")).unwrap();
    assert!(parsed.stream, "stream=true should now be accepted as a replay request");
    assert_eq!(parsed.starting_after, None, "no cursor without starting_after");
}

#[test]
fn parse_get_response_query_stream_invalid_value_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream=maybe")).unwrap_err();
    assert!(
        err.contains("must be 'true' or 'false'"),
        "stream=maybe should be rejected as invalid boolean: {err}"
    );
}

#[test]
fn parse_get_response_query_starting_after_with_stream_accepted() {
    let parsed = super::filter::parse_get_response_query(Some("stream=true&starting_after=5")).unwrap();
    assert!(parsed.stream, "stream=true should be set");
    assert_eq!(
        parsed.starting_after,
        Some(5),
        "starting_after=5 should parse as cursor 5"
    );
}

#[test]
fn parse_get_response_query_starting_after_order_independent() {
    let parsed = super::filter::parse_get_response_query(Some("starting_after=9&stream=true")).unwrap();
    assert!(parsed.stream, "stream=true should be set regardless of order");
    assert_eq!(
        parsed.starting_after,
        Some(9),
        "cursor should parse regardless of order"
    );
}

#[test]
fn parse_get_response_query_starting_after_without_stream_rejected() {
    let err = super::filter::parse_get_response_query(Some("starting_after=5")).unwrap_err();
    assert!(
        err.contains("requires 'stream=true'"),
        "starting_after without stream=true should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_starting_after_with_stream_false_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream=false&starting_after=5")).unwrap_err();
    assert!(
        err.contains("requires 'stream=true'"),
        "starting_after with stream=false should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_starting_after_empty_value_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream=true&starting_after=")).unwrap_err();
    assert!(
        err.contains("cursor must not be empty"),
        "empty starting_after should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_starting_after_non_numeric_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream=true&starting_after=abc")).unwrap_err();
    assert!(
        err.contains("not a valid integer"),
        "non-numeric starting_after should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_include_bracket_rejected() {
    let err = super::filter::parse_get_response_query(Some("include[]=file_search_call_results.results")).unwrap_err();
    assert!(
        err.contains("'include' parameter is not supported"),
        "include[] should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_include_percent_encoded_rejected() {
    let err = super::filter::parse_get_response_query(Some("include%5B%5D=usage")).unwrap_err();
    assert!(
        err.contains("'include' parameter is not supported"),
        "percent-encoded include[] should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_include_bare_rejected() {
    let err = super::filter::parse_get_response_query(Some("include=usage")).unwrap_err();
    assert!(
        err.contains("'include' parameter is not supported"),
        "bare include should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_include_obfuscation_rejected() {
    let err = super::filter::parse_get_response_query(Some("include_obfuscation=true")).unwrap_err();
    assert!(
        err.contains("'include_obfuscation' parameter is not supported"),
        "include_obfuscation should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_unknown_param_rejected() {
    let err = super::filter::parse_get_response_query(Some("foo=bar")).unwrap_err();
    assert!(
        err.contains("Unknown query parameter"),
        "unknown param should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_key_only_known_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream")).unwrap_err();
    assert!(
        err.contains("Missing value"),
        "key-only known param should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_key_only_include_rejected() {
    let err = super::filter::parse_get_response_query(Some("include")).unwrap_err();
    assert!(
        err.contains("Missing value"),
        "key-only include should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_key_only_unknown_rejected() {
    let err = super::filter::parse_get_response_query(Some("foo")).unwrap_err();
    assert!(
        err.contains("Unknown query parameter"),
        "key-only unknown param should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_percent_encoded_stream_false_accepted() {
    let parsed = super::filter::parse_get_response_query(Some("stream=%66alse")).unwrap();
    assert!(
        !parsed.stream,
        "percent-encoded stream=false should parse as non-stream"
    );
}

#[test]
fn parse_get_response_query_invalid_utf8_key_rejected() {
    let err = super::filter::parse_get_response_query(Some("%FF=x")).unwrap_err();
    assert!(
        err.contains("Invalid percent-encoding in query parameter key"),
        "invalid UTF-8 key should be rejected: {err}"
    );
}

#[test]
fn parse_get_response_query_invalid_utf8_value_rejected() {
    let err = super::filter::parse_get_response_query(Some("stream=%FF")).unwrap_err();
    assert!(
        err.contains("Invalid percent-encoding in value"),
        "invalid UTF-8 value should be rejected: {err}"
    );
}

#[test]
fn known_params_and_parser_match_arms_in_sync() {
    for &param in super::filter::GET_RESPONSE_KNOWN_PARAMS {
        let mut parsed = super::filter::GetResponseQuery::default();
        let result = super::filter::apply_get_response_param(&mut parsed, param, "__probe__");
        // Every known param is handled by an explicit match arm, so the probe
        // never falls through to the "Unknown query parameter" catch-all.
        // `stream`/`starting_after` reject the non-boolean/non-numeric probe;
        // the unsupported params reject with their own message. Either way the
        // error must not be the unknown-parameter fallthrough.
        if let Err(err) = result {
            assert!(
                !err.contains("Unknown"),
                "known param '{param}' should not produce an 'Unknown' error: {err}"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// list_input_items (direct unit tests)
// -----------------------------------------------------------------------------

fn make_record_with_input(input: serde_json::Value) -> ResponseRecord {
    ResponseRecord {
        id: "resp_test".to_owned(),
        owner: crate::test_utils::test_owner("default"),
        created_at: 1000,
        model: "gpt-4.1".to_owned(),
        response_object: json!({}),
        input,
        messages: json!([]),
    }
}

#[test]
fn list_input_items_scalar_input_normalized_to_message_item() {
    let record = make_record_with_input(json!("Hello"));
    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 1, "string input should produce one message item");
    assert_eq!(
        page.data[0],
        json!({
            "id": "msg_resp_test_input_0",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}]
        }),
        "string input should be normalized to a user message resource"
    );
    assert!(!page.has_more);
}

#[test]
fn list_input_items_null_input_returns_empty_page() {
    let record = make_record_with_input(json!(null));
    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(page.data.is_empty(), "null input should yield no items");
    assert!(!page.has_more);
    assert!(page.next_cursor.is_none());
}

#[test]
fn list_input_items_empty_array_returns_empty_page() {
    let record = make_record_with_input(json!([]));
    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(page.data.is_empty(), "empty array input should yield empty page");
    assert!(!page.has_more);
    assert!(page.next_cursor.is_none());
}

#[test]
fn list_input_items_ascending_preserves_natural_order() {
    let record = make_record_with_input(json!([
        {"id": "a", "val": 1},
        {"id": "b", "val": 2},
        {"id": "c", "val": 3}
    ]));
    let params = ListParams {
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data[0]["id"], "a");
    assert_eq!(page.data[1]["id"], "b");
    assert_eq!(page.data[2]["id"], "c");
}

#[test]
fn list_input_items_descending_reverses_order() {
    let record = make_record_with_input(json!([
        {"id": "a", "val": 1},
        {"id": "b", "val": 2},
        {"id": "c", "val": 3}
    ]));
    let params = ListParams {
        order: Order::Descending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data[0]["id"], "c");
    assert_eq!(page.data[1]["id"], "b");
    assert_eq!(page.data[2]["id"], "a");
}

#[test]
fn list_input_items_limit_zero_clamped_to_one() {
    let record = make_record_with_input(json!([
        {"id": "a"},
        {"id": "b"},
        {"id": "c"}
    ]));
    let params = ListParams {
        limit: 0,
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 1, "limit=0 should be clamped to 1");
    assert!(page.has_more);
}

#[test]
fn list_input_items_limit_above_max_clamped() {
    let items: Vec<serde_json::Value> = (0..5).map(|i| json!({"id": format!("item_{i}")})).collect();
    let record = make_record_with_input(json!(items));
    let params = ListParams {
        limit: MAX_PAGE_LIMIT + 50,
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(
        page.data.len(),
        5,
        "should return all items when limit exceeds count after clamping"
    );
    assert!(!page.has_more);
}

#[test]
fn list_input_items_cursor_after_last_returns_empty_page() {
    let record = make_record_with_input(json!([
        {"id": "a"},
        {"id": "b"}
    ]));
    let params = ListParams {
        cursor: Some("b".to_owned()),
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(page.data.is_empty(), "cursor after last item should yield empty page");
    assert!(!page.has_more);
}

#[test]
fn list_input_items_cursor_after_first_skips_it() {
    let record = make_record_with_input(json!([
        {"id": "a"},
        {"id": "b"},
        {"id": "c"}
    ]));
    let params = ListParams {
        cursor: Some("a".to_owned()),
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 2);
    assert_eq!(page.data[0]["id"], "b");
    assert_eq!(page.data[1]["id"], "c");
}

#[test]
fn list_input_items_numeric_cursor_fallback() {
    let record = make_record_with_input(json!([
        {"val": 1},
        {"val": 2},
        {"val": 3}
    ]));
    let params = ListParams {
        cursor: Some("1".to_owned()),
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 2, "numeric cursor 1 should skip first item");
    assert_eq!(page.data[0]["val"], 2);
    assert_eq!(page.data[1]["val"], 3);
}

#[test]
fn list_input_items_invalid_cursor_returns_error() {
    let record = make_record_with_input(json!([
        {"val": 1},
        {"val": 2}
    ]));
    let params = ListParams {
        cursor: Some("not_a_cursor".to_owned()),
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let result = list_input_items(&record, &params, includes);
    assert!(result.is_err(), "non-numeric cursor not matching any ID should error");
}

#[test]
fn list_input_items_next_cursor_uses_item_id() {
    let record = make_record_with_input(json!([
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "d"}
    ]));
    let params = ListParams {
        limit: 2,
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(page.has_more);
    assert_eq!(
        page.next_cursor.as_deref(),
        Some("b"),
        "next_cursor should use the last item's ID"
    );
}

#[test]
fn list_input_items_assigns_synthetic_ids_to_id_less_array_items() {
    let record = make_record_with_input(json!([
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "message", "role": "assistant", "content": "hello"}
    ]));
    // Default order is descending, but synthetic IDs must stay tied to
    // each item's original stored position, not its display position.
    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(
        page.data[0]["id"], "msg_resp_test_input_1",
        "the assistant message (original index 1) should surface first in descending order"
    );
    assert_eq!(
        page.data[1]["id"], "msg_resp_test_input_0",
        "the user message (original index 0) should surface second in descending order"
    );
}

#[test]
fn list_input_items_mixed_existing_and_synthetic_ids_paginate_correctly() {
    let record = make_record_with_input(json!([
        {"id": "a", "type": "message"},
        {"type": "message", "content": "hi"}
    ]));
    let base_params = ListParams {
        limit: 1,
        order: Order::Ascending,
        ..Default::default()
    };

    let includes = IncludeFields::default();
    let page1 = list_input_items(&record, &base_params, includes).unwrap();
    assert_eq!(
        page1.data[0]["id"], "a",
        "the item's existing id must be preserved, not overwritten"
    );
    assert!(page1.has_more);
    let cursor_after_a = page1.next_cursor.clone();
    assert_eq!(
        cursor_after_a.as_deref(),
        Some("a"),
        "next_cursor should reuse the item's existing id"
    );

    let page2 = list_input_items(
        &record,
        &ListParams {
            cursor: cursor_after_a,
            ..base_params.clone()
        },
        includes,
    )
    .unwrap();
    assert_eq!(
        page2.data[0]["id"], "msg_resp_test_input_1",
        "the id-less second item should get a synthetic id keyed by its original index"
    );
    assert!(!page2.has_more, "the mixed array is exhausted after the second item");
    assert!(page2.next_cursor.is_none());

    // Paginating again using the synthetic id itself as the cursor must still resolve it as an
    // ID-based cursor (not fall through to the numeric fallback), correctly yielding an empty
    // page since it was the last item.
    let page3 = list_input_items(
        &record,
        &ListParams {
            cursor: Some("msg_resp_test_input_1".to_owned()),
            ..base_params
        },
        includes,
    )
    .unwrap();
    assert!(
        page3.data.is_empty(),
        "cursor after the synthetic id of the last item should yield an empty page"
    );
    assert!(!page3.has_more);
}

#[test]
fn list_input_items_next_cursor_uses_synthetic_id_when_input_lacks_ids() {
    let record = make_record_with_input(json!([
        {"val": 1},
        {"val": 2},
        {"val": 3},
        {"val": 4}
    ]));
    let params = ListParams {
        limit: 2,
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(page.has_more);
    assert_eq!(
        page.next_cursor.as_deref(),
        Some("msg_resp_test_input_1"),
        "next_cursor should use the synthetic ID assigned to ID-less items, not a raw numeric offset"
    );
}

#[test]
fn list_input_items_numeric_cursor_still_works_as_fallback_for_synthetic_ids() {
    let record = make_record_with_input(json!([
        {"val": 1},
        {"val": 2},
        {"val": 3},
        {"val": 4}
    ]));
    let params = ListParams {
        cursor: Some("2".to_owned()),
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(
        page.data.len(),
        2,
        "numeric offset cursor should still work even though items now carry synthetic IDs"
    );
    assert_eq!(page.data[0]["val"], 3);
    assert_eq!(page.data[1]["val"], 4);
}

#[test]
fn list_input_items_no_next_cursor_when_no_more() {
    let record = make_record_with_input(json!([{"id": "a"}, {"id": "b"}]));
    let params = ListParams {
        limit: 10,
        order: Order::Ascending,
        ..Default::default()
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert!(!page.has_more);
    assert!(
        page.next_cursor.is_none(),
        "next_cursor should be None when all items fit"
    );
}

#[test]
fn list_input_items_default_limit_is_default_page_limit() {
    let params = ListParams::default();
    assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
    assert_eq!(params.order, Order::Descending);
    assert!(params.cursor.is_none());
}

#[test]
fn list_input_items_object_input_wrapped_as_single_item() {
    let record = make_record_with_input(json!({"type": "message", "role": "user", "content": "hi"}));
    let params = ListParams::default();
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 1);
    assert_eq!(page.data[0]["type"], "message");
}

#[test]
fn list_input_items_descending_cursor_operates_on_reversed_order() {
    let record = make_record_with_input(json!([
        {"id": "a"},
        {"id": "b"},
        {"id": "c"},
        {"id": "d"}
    ]));
    let params = ListParams {
        cursor: Some("c".to_owned()),
        limit: 2,
        order: Order::Descending,
    };
    let includes = IncludeFields::default();
    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 2, "should return items after cursor in reversed order");
    assert_eq!(page.data[0]["id"], "b");
    assert_eq!(page.data[1]["id"], "a");
    assert!(!page.has_more);
}

// -----------------------------------------------------------------------------
// on_request_body edge cases
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_body_non_end_of_stream_does_not_process() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1","input":"Hi"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-EOS request body should just continue"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_request_body_non_post_at_eos_does_not_extract_input() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_123");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let mut body = Some(Bytes::from_static(b"{}"));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-POST request body at EOS should continue"
    );
}

// -----------------------------------------------------------------------------
// on_response_body message assembly edge cases
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_null_output() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_null_output",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": [{"role": "user", "content": "Hello"}],
        "output": null
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_null_output")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(
        record.messages,
        json!([{"role": "user", "content": "Hello"}]),
        "null output should not appear in messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_no_output_field() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_no_output",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": [{"role": "user", "content": "Hello"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_no_output")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(
        record.messages,
        json!([{"role": "user", "content": "Hello"}]),
        "missing output field should not appear in messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_non_array_output() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_object_output",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": "Hello",
        "output": {"type": "message", "content": "Hi"}
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_object_output")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(
        record.messages,
        json!([
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "content": "Hi"}
        ]),
        "non-array output should be pushed as a single item"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_object_input() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_object_input",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": {"type": "custom_item", "data": "some data"},
        "output": [{"type": "message", "content": "Result"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_object_input")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(
        record.messages,
        json!([
            {"type": "custom_item", "data": "some data"},
            {"type": "message", "content": "Result"}
        ]),
        "object input should be pushed as a single item in messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_null_input() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_null_input",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": null,
        "output": [{"type": "message", "content": "Result"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_null_input")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(
        record.messages,
        json!([{"type": "message", "content": "Result"}]),
        "null input should not appear in messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_response_with_no_input_field() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    drop(filter.on_request(&mut ctx).await.unwrap());

    let body_json = json!({
        "id": "resp_missing_input",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "output": [{"type": "message", "content": "Result"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&crate::test_utils::test_owner("default"), "resp_missing_input")
        .await
        .unwrap()
        .expect("record should exist");
    assert_eq!(record.input, json!(null), "missing input should default to null");
    assert_eq!(
        record.messages,
        json!([{"type": "message", "content": "Result"}]),
        "missing input should not appear in messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn on_response_body_persists_uses_trusted_owner_context() {
    let filter = make_filter();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    let store = install_store(&mut ctx).await;
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    let owner = crate::StateOwner::from_trusted_parts("custom_tenant", "issuer-a", "alice").unwrap();
    ctx.extensions.insert(owner.clone());
    drop(filter.on_request(&mut ctx).await.unwrap());
    let same_tenant_other = crate::StateOwner::from_trusted_parts("custom_tenant", "issuer-a", "bob").unwrap();
    ctx.extensions.insert(same_tenant_other.clone());

    let body_json = json!({
        "id": "resp_tenant_body",
        "created_at": 1_719_900_000,
        "model": "gpt-4.1",
        "status": "completed",
        "input": "Hi",
        "output": []
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let record = store
        .get_response(&owner, "resp_tenant_body")
        .await
        .unwrap()
        .expect("record should exist under custom tenant");
    assert_eq!(record.owner, owner);

    let outsider_record = store
        .get_response(&same_tenant_other, "resp_tenant_body")
        .await
        .unwrap();
    assert!(
        outsider_record.is_none(),
        "buffered response must retain the complete initiating owner"
    );
}

// -----------------------------------------------------------------------------
// GET /v1/responses/{id}/input_items edge cases
// -----------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_trailing_slash_handled() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_slash_items",
        "default",
        json!([{"id": "item_1", "type": "message"}]),
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_slash_items/input_items/");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "trailing slash on input_items should be handled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_with_scalar_input() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_scalar", "default", json!("hello world")).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_scalar/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert_eq!(
        body["data"].as_array().unwrap().len(),
        1,
        "string input should produce one message item"
    );
    assert_eq!(
        body["data"][0],
        json!({
            "id": "msg_resp_scalar_input_0",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello world"}]
        }),
        "string input should be normalized to a user message resource"
    );
    assert_eq!(
        body["first_id"], "msg_resp_scalar_input_0",
        "first_id should use the synthetic message id"
    );
    assert_eq!(
        body["last_id"], "msg_resp_scalar_input_0",
        "last_id should use the synthetic message id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_default_descending_order() {
    let filter = make_filter();
    let registry = init_store_and_seed(
        "resp_desc",
        "default",
        json!([
            {"id": "item_1", "type": "message"},
            {"id": "item_2", "type": "message"},
            {"id": "item_3", "type": "message"}
        ]),
    )
    .await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_desc/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data[0]["id"], "item_3", "default descending order should reverse items");
    assert_eq!(data[1]["id"], "item_2");
    assert_eq!(data[2]["id"], "item_1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_with_empty_input() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_empty_input", "default", json!([])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_empty_input/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200);

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    assert!(body["data"].as_array().unwrap().is_empty());
    assert_eq!(body["has_more"], false);
    assert_eq!(body["first_id"], json!(null), "empty data should have null first_id");
    assert_eq!(body["last_id"], json!(null), "empty data should have null last_id");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_tenant_isolation() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_tenant_items", "tenant_a", json!([{"id": "item_1"}])).await;

    let req = crate::test_utils::make_request(http::Method::GET, "/v1/responses/resp_tenant_items/input_items");
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(
        rejection.status, 404,
        "input_items should return 404 when response belongs to different tenant"
    );
}

// -----------------------------------------------------------------------------
// GET /v1/responses/{id}/input_items include projection
// -----------------------------------------------------------------------------

/// Stored input covering a top-level include-gated field (reasoning
/// `encrypted_content`) and a nested one (message `input_image.image_url`).
fn include_fixture_input() -> serde_json::Value {
    json!([
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "stored-secret"
        },
        {
            "id": "msg_1",
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"},
                {"type": "input_text", "text": "keep me"}
            ]
        }
    ])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_omits_include_gated_fields_when_not_requested() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_include_default", "default", include_fixture_input()).await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_include_default/input_items?order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "input_items without include should return 200");

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "both stored items should be listed");
    assert!(
        data[0].get("encrypted_content").is_none(),
        "unrequested encrypted reasoning must be omitted"
    );
    assert!(
        data[1]["content"][0].get("image_url").is_none(),
        "unrequested input-image URLs must be omitted"
    );
    assert_eq!(
        data[1]["content"][1]["text"], "keep me",
        "content not gated by include must survive projection"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_include_reveals_requested_field_for_both_sdk_encodings() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_include_reveal", "default", include_fixture_input()).await;

    for query in [
        "include=reasoning.encrypted_content&order=asc",
        "include%5B%5D=reasoning.encrypted_content&order=asc",
    ] {
        let path = format!("/v1/responses/resp_include_reveal/input_items?{query}");
        let req = crate::test_utils::make_request(http::Method::GET, &path);
        let mut ctx = crate::test_utils::make_owned_filter_context(&req);
        ctx.extensions.insert(registry.clone());

        let action = filter.on_request(&mut ctx).await.unwrap();
        let rejection = expect_reject(action);
        assert_eq!(rejection.status, 200, "include should be accepted for {query}");

        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        let data = body["data"].as_array().unwrap();
        assert_eq!(
            data[0]["encrypted_content"], "stored-secret",
            "{query} should reveal the requested stored field"
        );
        assert!(
            data[1]["content"][0].get("image_url").is_none(),
            "{query} must not reveal include-gated fields it did not request"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_include_applies_to_paginated_window() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_include_page", "default", include_fixture_input()).await;

    let req = crate::test_utils::make_request(
        http::Method::GET,
        "/v1/responses/resp_include_page/input_items?include=reasoning.encrypted_content&limit=1&order=asc",
    );
    let mut ctx = crate::test_utils::make_owned_filter_context(&req);
    ctx.extensions.insert(registry);

    let action = filter.on_request(&mut ctx).await.unwrap();
    let rejection = expect_reject(action);
    assert_eq!(rejection.status, 200, "include should combine with pagination params");

    let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(
        data.len(),
        1,
        "limit should still bound the page when include is present"
    );
    assert_eq!(body["has_more"], true, "a second page should remain available");
    assert_eq!(
        body["last_id"], "reasoning_1",
        "projection must not strip the item ID used as the pagination cursor"
    );
    assert_eq!(
        data[0]["encrypted_content"], "stored-secret",
        "requested field should be present on the paginated item"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_input_items_rejects_malformed_include_values() {
    let filter = make_filter();
    let registry = init_store_and_seed("resp_include_bad", "default", include_fixture_input()).await;

    for (query, expected) in [
        ("include=future.secret_field", "unsupported include value"),
        ("include%5B%5D=future.secret_field", "unsupported include value"),
        ("include", "requires a value"),
    ] {
        let path = format!("/v1/responses/resp_include_bad/input_items?{query}");
        let req = crate::test_utils::make_request(http::Method::GET, &path);
        let mut ctx = crate::test_utils::make_owned_filter_context(&req);
        ctx.extensions.insert(registry.clone());

        let action = filter.on_request(&mut ctx).await.unwrap();
        let rejection = expect_reject(action);
        assert_eq!(rejection.status, 400, "{query} should be rejected as invalid input");

        let body: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains(expected),
            "{query} should explain the problem, got '{message}'"
        );
    }
}

#[test]
fn list_input_items_projects_each_page_item_with_requested_includes() {
    let record = make_record_with_input(include_fixture_input());
    let mut includes = IncludeFields::default();
    includes.insert(IncludeField::ReasoningEncryptedContent);
    let params = ListParams {
        cursor: None,
        limit: 1,
        order: Order::Ascending,
    };

    let page = list_input_items(&record, &params, includes).unwrap();
    assert_eq!(page.data.len(), 1, "limit should bound the projected page");
    assert!(page.has_more, "a second item should remain");
    assert_eq!(
        page.data[0]["encrypted_content"], "stored-secret",
        "requested include should preserve the encrypted reasoning content"
    );

    let next_params = ListParams {
        cursor: page.next_cursor.clone(),
        limit: 1,
        order: Order::Ascending,
    };
    let next_page = list_input_items(&record, &next_params, includes).unwrap();
    assert!(
        next_page.data[0]["content"][0].get("image_url").is_none(),
        "include values that were not requested must stay projected out on later pages"
    );
    assert_eq!(
        next_page.data[0]["content"][1]["text"], "keep me",
        "non-gated content must survive projection"
    );
    assert_eq!(
        record.input[1]["content"][0]["image_url"], "https://example.com/image.png",
        "projection must not mutate the stored record"
    );
}

// -----------------------------------------------------------------------------
// Additional config validation edge cases
// -----------------------------------------------------------------------------

#[test]
fn sqlite_colon_memory_variant_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite://:memory:"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "sqlite://:memory: variant should be accepted as in-memory database"
    );
}

#[test]
fn sqlite_database_url_without_double_slash_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite:test.db?mode=rwc"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "sqlite URL without // prefix should be accepted");
}

#[test]
fn sqlite_memory_with_extra_query_params_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite:///tmp/test.db?mode=memory&cache=shared"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "sqlite URL with mode=memory among other query params should be accepted"
    );
}

#[test]
fn postgres_config_accepts_public_ipv4_host() {
    let yaml = postgres_config_yaml("postgres://user:pass@1.2.3.4:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "public IPv4 postgres host should be accepted");
}

#[test]
fn postgres_config_rejects_ipv6_unique_local() {
    let yaml = postgres_config_yaml("postgres://user:pass@[fd12:3456:789a::1]:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "IPv6 unique-local postgres host should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_ipv6_unspecified() {
    let yaml = postgres_config_yaml("postgres://user:pass@[::]:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "IPv6 unspecified postgres host should be rejected by default"
    );
}

#[test]
fn postgres_config_rejects_ipv4_mapped_loopback_in_authority() {
    let yaml = postgres_config_yaml("postgres://user:pass@[::ffff:127.0.0.1]:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "IPv4-mapped loopback in authority should be rejected");
}

#[test]
fn postgres_config_url_fragment_not_treated_as_query() {
    let yaml = postgres_config_yaml(
        "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=verify-full#sslrootcert=/path/to/ca.pem",
        "ssl_root_cert: /path/to/ca.pem",
    );
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "URL fragment should not be parsed as query parameters");
}

#[test]
fn postgres_config_rejects_ssl_cert_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&ssl-cert=../client.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl-cert alias with path traversal should be rejected");
}

#[test]
fn postgres_config_rejects_ssl_key_path_traversal() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?sslmode=require&ssl-key=../client.key"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "ssl-key alias with path traversal should be rejected");
}

#[test]
fn postgres_config_with_ssl_ca_alias_and_verified_sslmode_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?ssl-mode=verify-full&ssl-ca=/path/to/ca.pem"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_ok(),
        "ssl-ca alias should be recognized as root cert parameter"
    );
}

#[test]
fn postgres_config_rejects_hostaddr_with_invalid_ip() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pass@1.2.3.4:5432/praxis?hostaddr=not-an-ip"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_err(), "hostaddr with invalid IP should be rejected");
}

#[test]
fn postgres_config_accepts_ipv6_public_host() {
    let yaml = postgres_config_yaml("postgres://user:pass@[2606:4700::1]:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(result.is_ok(), "public IPv6 postgres host should be accepted");
}

#[test]
fn postgres_config_rejects_ipv6_link_local() {
    let yaml = postgres_config_yaml("postgres://user:pass@[fe80::1]:5432/praxis", "");
    let result = ResponseStoreFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "IPv6 link-local postgres host should be rejected by default"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> ResponseStoreFilter {
    ResponseStoreFilter::with_bounds(
        super::config::DEFAULT_MAX_EVENT_COUNT,
        super::config::DEFAULT_MAX_EVENT_BYTES,
    )
}

/// Build an in-memory default store plus a registry holding it, without touching
/// a context. The filter resolves the same backend once the registry is
/// installed into a request context, mirroring serving-runtime provisioning.
async fn empty_registry() -> (ResponseStoreRegistry, Arc<dyn PersistedStateBackend>) {
    let store: Arc<dyn PersistedStateBackend> = Arc::new(
        SqliteResponseStore::new(
            "sqlite::memory:",
            "test_responses",
            "test_conversations",
            None,
            None,
            None,
        )
        .await
        .expect("in-memory sqlite store should build"),
    );
    let registry = ResponseStoreRegistry::new();
    registry
        .register(&Arc::from(DEFAULT_STORE_NAME), Arc::clone(&store))
        .expect("default store should register");
    (registry, store)
}

/// A registry holding an empty in-memory default store.
async fn store_registry() -> ResponseStoreRegistry {
    empty_registry().await.0
}

/// Install an empty in-memory default store into the context, returning the
/// backing store for direct seeding.
async fn install_store(ctx: &mut HttpFilterContext<'_>) -> Arc<dyn PersistedStateBackend> {
    let (registry, store) = empty_registry().await;
    ctx.extensions.insert(registry);
    store
}

async fn run_request_phase(filter: &ResponseStoreFilter, ctx: &mut HttpFilterContext<'_>) {
    // The happy path needs a provisioned store; install an empty one unless the
    // test already installed (and possibly seeded) its own.
    if ctx.extensions.get::<ResponseStoreRegistry>().is_none() {
        install_store(ctx).await;
    }
    let action = filter.on_request(ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "request phase should continue"
    );
}

fn temp_sqlite_url(test_name: &str) -> (String, PathBuf) {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let db_path = std::env::temp_dir().join(format!("praxis_{test_name}_{}_{}.db", std::process::id(), nanos));
    (format!("sqlite://{}?mode=rwc", db_path.display()), db_path)
}

fn cleanup_sqlite_file(db_path: &PathBuf) {
    drop(std::fs::remove_file(db_path));
    drop(std::fs::remove_file(format!("{}-shm", db_path.display())));
    drop(std::fs::remove_file(format!("{}-wal", db_path.display())));
}

/// A registry whose default store is backed by the file at `db_url`, matching a
/// pipeline test's `test_responses`/`test_conversations` tables so a store opened
/// separately for verification observes the same rows.
async fn file_store_registry(db_url: &str) -> ResponseStoreRegistry {
    let store: Arc<dyn PersistedStateBackend> = Arc::new(
        SqliteResponseStore::new(db_url, "test_responses", "test_conversations", None, None, None)
            .await
            .expect("file-backed sqlite store should build"),
    );
    let registry = ResponseStoreRegistry::new();
    registry
        .register(&Arc::from(DEFAULT_STORE_NAME), store)
        .expect("default store should register");
    registry
}

/// A registry whose default store is seeded with one completed response owned by
/// `tenant_id`. Install it into a request context to expose the record.
async fn init_store_and_seed(id: &str, tenant_id: &str, input: serde_json::Value) -> ResponseStoreRegistry {
    init_store_and_seed_owner(id, crate::test_utils::test_owner(tenant_id), input).await
}

async fn init_store_and_seed_owner(
    id: &str,
    owner: crate::StateOwner,
    input: serde_json::Value,
) -> ResponseStoreRegistry {
    let (registry, store) = empty_registry().await;
    let record = ResponseRecord {
        id: id.to_owned(),
        owner,
        created_at: 1000,
        model: "gpt-4.1".to_owned(),
        response_object: json!({"status": "completed"}),
        input,
        messages: json!([{"role": "user", "content": "hello"}]),
    };
    store
        .upsert_response(&record)
        .await
        .expect("seed response should succeed");
    registry
}

/// Build one replay event-log row for seeding a store directly.
fn event_record(
    response_id: &str,
    owner: &crate::StateOwner,
    sequence_number: u64,
    event_type: &str,
    terminal: bool,
) -> crate::store::ResponseEventRecord {
    crate::store::ResponseEventRecord {
        response_id: response_id.to_owned(),
        owner: owner.clone(),
        sequence_number,
        event_type: event_type.to_owned(),
        payload: serde_json::to_vec(&json!({"type": event_type, "sequence_number": sequence_number}))
            .expect("event payload serializes"),
        terminal,
        created_at: 1000,
    }
}

/// A registry whose default store holds one completed response owned by `owner`
/// plus the given replay event log. Backs the `?stream=true` replay tests.
async fn seed_response_with_events(
    id: &str,
    owner: crate::StateOwner,
    events: Vec<crate::store::ResponseEventRecord>,
) -> ResponseStoreRegistry {
    let (registry, store) = empty_registry().await;
    let record = ResponseRecord {
        id: id.to_owned(),
        owner: owner.clone(),
        created_at: 1000,
        model: "gpt-4.1".to_owned(),
        response_object: json!({"status": "completed"}),
        input: json!([]),
        messages: json!([{"role": "user", "content": "hello"}]),
    };
    store
        .upsert_response(&record)
        .await
        .expect("seed response should succeed");
    store
        .append_events(&owner, id, &events)
        .await
        .expect("seed replay events should succeed");
    registry
}

fn expect_reject(action: FilterAction) -> praxis_filter::Rejection {
    match action {
        FilterAction::Reject(r) => r,
        other => panic!("expected Reject, got {other:?}"),
    }
}

/// Extract the boxed streaming terminal response from a replay action.
fn expect_streaming(action: FilterAction) -> Box<StreamingTerminalResponse> {
    match action {
        FilterAction::StreamingTerminalResponse(s) => s,
        other => panic!("expected StreamingTerminalResponse, got {other:?}"),
    }
}

/// Drive a replay body to completion, concatenating every chunk into one buffer.
async fn drain_replay_body(mut streaming: Box<StreamingTerminalResponse>) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(chunk) = streaming
        .body
        .next_chunk()
        .await
        .expect("replay chunk should not error")
    {
        out.extend_from_slice(&chunk);
    }
    out
}

fn assert_has_json_content_type(rejection: &praxis_filter::Rejection) {
    let has_ct = rejection
        .headers
        .iter()
        .any(|(k, v)| k == "content-type" && v == "application/json");
    assert!(has_ct, "rejection should have application/json content-type");
}

// -----------------------------------------------------------------------------
// Pool Config
// -----------------------------------------------------------------------------

#[test]
fn config_accepts_pool_options() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
pool:
  max_connections: 20
  min_connections: 2
  idle_timeout_secs: 300
  acquire_timeout_secs: 15
"#,
    )
    .unwrap();
    let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", &yaml).unwrap();
    validate_config(&cfg).unwrap();

    let pool = cfg.pool.expect("pool config should be present");
    assert_eq!(pool.max_connections, Some(20));
    assert_eq!(pool.min_connections, Some(2));
    assert_eq!(pool.idle_timeout_secs, Some(300));
    assert_eq!(pool.acquire_timeout_secs, Some(15));
}

#[test]
fn config_accepts_partial_pool_options() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
pool:
  max_connections: 50
"#,
    )
    .unwrap();
    let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", &yaml).unwrap();
    validate_config(&cfg).unwrap();

    let pool = cfg.pool.expect("pool config should be present");
    assert_eq!(pool.max_connections, Some(50));
    assert!(pool.min_connections.is_none());
    assert!(pool.idle_timeout_secs.is_none());
    assert!(pool.acquire_timeout_secs.is_none());
}

#[test]
fn config_omitted_pool_yields_none() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: responses
conversations_table: conversations
"#,
    )
    .unwrap();
    let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", &yaml).unwrap();
    validate_config(&cfg).unwrap();
    assert!(cfg.pool.is_none(), "omitted pool should be None");
}

#[test]
fn postgres_uppercase_table_name_rejected_at_config_load() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: postgres
database_url: "postgres://user:pw@1.2.3.4/praxis"
allow_private_database_url: true
responses_table: OpenAIResponses
conversations_table: openai_conversations
"#,
    )
    .unwrap();

    let err = ResponseStoreFilter::from_config(&yaml).map(|_| ()).unwrap_err();
    assert!(
        err.to_string().contains("must be lowercase"),
        "PostgreSQL uppercase table name should be rejected at config load, got: {err}"
    );
}

#[test]
fn sqlite_uppercase_table_name_still_accepted() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
backend: sqlite
database_url: "sqlite::memory:"
responses_table: OpenAIResponses
conversations_table: openai_conversations
"#,
    )
    .unwrap();

    assert!(
        ResponseStoreFilter::from_config(&yaml).is_ok(),
        "SQLite compares names case-insensitively, so uppercase must still be accepted"
    );
}

// -----------------------------------------------------------------------------
// InputItemPage serialization & allocation tests
// -----------------------------------------------------------------------------

#[test]
fn input_item_page_serialization_empty_page() {
    let page = InputItemPage {
        data: vec![],
        next_cursor: None,
        has_more: false,
    };
    let json_str = serde_json::to_string(&page).expect("empty page should serialize");
    let val: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
    assert_eq!(
        val,
        json!({
            "object": "list",
            "data": [],
            "has_more": false,
            "first_id": null,
            "last_id": null
        }),
        "empty page serialization must match expected list schema"
    );
}

#[test]
fn input_item_page_serialization_full_page() {
    let items: Vec<serde_json::Value> = (0..20)
        .map(|i| {
            json!({
                "id": format!("msg_item_{i}"),
                "type": "message",
                "role": "user",
                "content": format!("content {i}")
            })
        })
        .collect();
    let page = InputItemPage {
        data: items.clone(),
        next_cursor: Some("msg_item_19".to_owned()),
        has_more: true,
    };
    let json_str = serde_json::to_string(&page).expect("full page should serialize");
    let val: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
    assert_eq!(
        val,
        json!({
            "object": "list",
            "data": items,
            "has_more": true,
            "first_id": "msg_item_0",
            "last_id": "msg_item_19"
        }),
        "full page serialization must match expected list schema"
    );
}

#[test]
fn input_item_page_serialization_maximum_size_100_items() {
    let items: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("msg_max_{i}"),
                "type": "message",
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": [
                    {
                        "type": "input_text",
                        "text": format!("Nontrivial text payload for item {i} with additional metadata and details.")
                    }
                ],
                "metadata": {
                    "index": i,
                    "tag": "max_page_test"
                }
            })
        })
        .collect();
    let page = InputItemPage {
        data: items.clone(),
        next_cursor: Some("msg_max_99".to_owned()),
        has_more: true,
    };
    let json_str = serde_json::to_string(&page).expect("100-item page should serialize");
    let val: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
    assert_eq!(
        val,
        json!({
            "object": "list",
            "data": items,
            "has_more": true,
            "first_id": "msg_max_0",
            "last_id": "msg_max_99"
        }),
        "maximum size 100-item page serialization must match expected list schema"
    );
}

#[test]
fn input_item_page_serialization_non_object_and_cursor_fallback() {
    let page = InputItemPage {
        data: vec![json!("plain string input"), json!(12345)],
        next_cursor: Some("2".to_owned()),
        has_more: true,
    };
    let json_str = serde_json::to_string(&page).expect("non-object page should serialize");
    let val: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
    assert_eq!(
        val,
        json!({
            "object": "list",
            "data": ["plain string input", 12345],
            "has_more": true,
            "first_id": null,
            "last_id": "2"
        }),
        "non-object page serialization must fall back last_id to next_cursor"
    );
}

#[test]
fn input_item_page_serialization_100_item_nontrivial_allocation_evidence() {
    let items: Vec<serde_json::Value> = (0..100)
        .map(|i| {
            json!({
                "id": format!("msg_nontrivial_{i}"),
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": format!("Nontrivial text payload for item {i}: {}", "x".repeat(200))
                    }
                ],
                "metadata": {
                    "item_index": i,
                    "nested_info": {
                        "key_a": "value_a",
                        "key_b": 42
                    }
                }
            })
        })
        .collect();

    let page = InputItemPage {
        data: items,
        next_cursor: Some("msg_nontrivial_99".to_owned()),
        has_more: true,
    };

    // Legacy pattern: serde_json::json! deep-copies page.data into a second Value tree
    let legacy_fn = |p: &InputItemPage| -> Vec<u8> {
        let first_id = p.data.first().and_then(|v| v.get("id")).and_then(|v| v.as_str());
        let last_id = p
            .data
            .last()
            .and_then(|v| v.get("id"))
            .and_then(|v| v.as_str())
            .or(p.next_cursor.as_deref());

        let body = serde_json::json!({
            "object": "list",
            "data": p.data,
            "has_more": p.has_more,
            "first_id": first_id,
            "last_id": last_id,
        });
        serde_json::to_vec(&body).unwrap()
    };

    // New pattern: direct serialization from &page without second Value or Vec<Value> tree
    let direct_fn = |p: &InputItemPage| -> Vec<u8> { serde_json::to_vec(p).unwrap() };

    let legacy_bytes = legacy_fn(&page);
    let direct_bytes = direct_fn(&page);

    assert_eq!(
        legacy_bytes, direct_bytes,
        "direct serialization output must match legacy json! output byte-for-byte"
    );

    let legacy_allocs = allocation_counter::measure(|| {
        std::hint::black_box(legacy_fn(&page));
    });

    let direct_allocs = allocation_counter::measure(|| {
        std::hint::black_box(direct_fn(&page));
    });

    assert!(
        direct_allocs.count_total < legacy_allocs.count_total,
        "direct serialization must perform fewer allocations: direct={} legacy={}",
        direct_allocs.count_total,
        legacy_allocs.count_total
    );

    assert!(
        direct_allocs.bytes_total < legacy_allocs.bytes_total,
        "direct serialization must allocate fewer bytes: direct={} legacy={}",
        direct_allocs.bytes_total,
        legacy_allocs.bytes_total
    );
}

/// The response-store filter only exercises the response half, so this recording
/// double leaves the conversation-item surface unsupported.
#[async_trait::async_trait]
impl crate::store::ConversationItemStore for RecordingResponseStore {
    async fn upsert_conversation(
        &self,
        _record: &crate::store::ConversationRecord,
    ) -> Result<(), crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn update_conversation_messages(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _messages: &serde_json::Value,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn update_conversation_metadata(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _metadata: &serde_json::Value,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn compare_and_swap_conversation_messages(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _expected_messages: &serde_json::Value,
        _messages: &serde_json::Value,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn get_conversation(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
    ) -> Result<Option<crate::store::ConversationRecord>, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn delete_conversation(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn create_conversation_items(
        &self,
        _items: &[crate::store::ConversationItemRecord],
    ) -> Result<(), crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn list_conversation_items(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _after_item_id: Option<&str>,
        _limit: u32,
        _ascending: bool,
    ) -> Result<Vec<crate::store::ConversationItemRecord>, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn get_existing_conversation_item_ids(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _item_ids: &[&str],
    ) -> Result<Vec<String>, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn get_conversation_item(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _item_id: &str,
    ) -> Result<Option<crate::store::ConversationItemRecord>, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn delete_conversation_item(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _item_id: &str,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn conversation_item_position(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _item_id: &str,
    ) -> Result<Option<i64>, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn max_item_position(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
    ) -> Result<i64, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn create_items_and_sync_messages(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _items: &[crate::store::ConversationItemRecord],
    ) -> Result<(), crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }

    async fn delete_item_and_sync_messages(
        &self,
        _owner: &crate::StateOwner,
        _conversation_id: &str,
        _item_id: &str,
    ) -> Result<bool, crate::store::StoreError> {
        Err(crate::store::StoreError::Unavailable(
            "recording store has no conversation items".to_owned(),
        ))
    }
}
