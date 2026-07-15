// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the Responses proxy filter.

use bytes::Bytes;
use http::Method;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, HttpFilter};
use serde_json::json;

use super::super::state::ResponsesState;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn from_config_accepts_null() {
    let yaml = serde_yaml::Value::Null;
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_proxy",
        "filter name should be openai_responses_proxy"
    );
}

#[test]
fn from_config_accepts_empty_mapping() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    assert_eq!(
        filter.name(),
        "openai_responses_proxy",
        "filter name should be openai_responses_proxy"
    );
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
    let result = super::ResponsesProxyFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

#[test]
fn body_access_is_read_write() {
    let filter = make_filter();
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadWrite,
        "openai_responses_proxy must declare ReadWrite to modify the body"
    );
}

#[test]
fn body_mode_is_stream_buffer() {
    let filter = make_filter();
    assert!(
        matches!(filter.request_body_mode(), BodyMode::StreamBuffer { .. }),
        "openai_responses_proxy must use StreamBuffer to receive complete body at EOS"
    );
}

#[tokio::test]
async fn on_request_returns_continue() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_request should return Continue"
    );
}

#[tokio::test]
async fn passthrough_without_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = r#"{"model":"gpt-4o","input":"hello"}"#;
    let mut body = Some(Bytes::from(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue without ResponsesState"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_bytes()),
        "body should be unchanged when no state is present"
    );
}

#[tokio::test]
async fn not_end_of_stream_continues() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"input":"partial"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-EOS should return Continue"
    );
}

#[tokio::test]
async fn rebuilds_body_with_conversation_history() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "What did I say?",
        "previous_response_id": "resp_abc123"
    });

    let mut state = ResponsesState::from_request_body(request_body);
    let stored_history = vec![
        json!({"role": "user", "content": "Hello"}),
        json!({"role": "assistant", "content": "Hi there!"}),
    ];
    state.messages.splice(0..0, stored_history);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"What did I say?","previous_response_id":"resp_abc123"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "should continue after rebuilding body"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["model"], "gpt-4o", "model should be preserved");

    let input = rebuilt["input"].as_array().unwrap();
    assert_eq!(input.len(), 3, "input should contain stored history + new message");
    assert_eq!(input[0]["content"], "Hello", "first message should be stored history");
    assert_eq!(
        input[1]["content"], "Hi there!",
        "second message should be stored history"
    );

    assert!(
        rebuilt.get("previous_response_id").is_none(),
        "previous_response_id should be stripped from outbound body"
    );
}

#[tokio::test]
async fn updates_content_length_header() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let has_content_length = ctx
        .extra_request_headers
        .iter()
        .any(|(k, _)| k.as_ref() == "content-length");
    assert!(
        has_content_length,
        "content-length header should be set after body rebuild"
    );

    let cl_value: usize = ctx
        .extra_request_headers
        .iter()
        .find(|(k, _)| k.as_ref() == "content-length")
        .map(|(_, v)| v.parse().unwrap())
        .unwrap();
    assert_eq!(
        cl_value,
        body.as_ref().unwrap().len(),
        "content-length should match rebuilt body size"
    );
}

#[tokio::test]
async fn preserves_other_request_fields() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "temperature": 0.7,
        "stream": true,
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","temperature":0.7,"stream":true,"previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["temperature"], 0.7, "temperature should be preserved");
    assert_eq!(rebuilt["stream"], true, "stream should be preserved");
    assert_eq!(rebuilt["model"], "gpt-4o", "model should be preserved");
}

#[tokio::test]
async fn rejects_oversized_rebuilt_body_with_413() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 16").unwrap();
    let filter = super::ResponsesProxyFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.messages.splice(
        0..0,
        vec![json!({"role": "user", "content": "a]long message that exceeds the tiny limit"})],
    );
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","previous_response_id":"resp_abc123"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 413),
        "should reject with 413 when rebuilt body exceeds max_body_bytes"
    );
}

#[tokio::test]
async fn strips_conversation_from_outbound_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "conversation": {"id": "conv_abc123"}
    });
    let state = ResponsesState::from_request_body(request_body);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","conversation":{"id":"conv_abc123"}}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.get("conversation").is_none(),
        "conversation should be stripped from outbound body"
    );
    assert_eq!(rebuilt["model"], "gpt-4o");
}

#[tokio::test]
async fn strips_both_previous_response_id_and_conversation() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "previous_response_id": "resp_abc123",
        "conversation": "conv_xyz789"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","previous_response_id":"resp_abc123","conversation":"conv_xyz789"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.get("previous_response_id").is_none(),
        "previous_response_id should be stripped"
    );
    assert!(rebuilt.get("conversation").is_none(), "conversation should be stripped");
}

#[tokio::test]
async fn passthrough_strips_conversation_from_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from(r#"{"model":"gpt-4.1","input":"hello","conversation":42}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let parsed: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        parsed.get("conversation").is_none(),
        "conversation should be stripped even without ResponsesState"
    );
    assert_eq!(parsed["model"], "gpt-4.1", "other fields should be preserved");
}

#[tokio::test]
async fn passthrough_none_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body: Option<Bytes> = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "None body should return Continue"
    );
    assert!(body.is_none(), "None body should remain None");
}

#[tokio::test]
async fn passthrough_invalid_json_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = b"not valid json {{{";
    let mut body = Some(Bytes::from_static(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "invalid JSON body should return Continue"
    );
    assert_eq!(
        body.as_deref(),
        Some(original.as_slice()),
        "invalid JSON body should pass through unchanged"
    );
}

#[tokio::test]
async fn rebuild_non_object_request_body_passes_through() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = ResponsesState {
        request_body: json!(["not", "an", "object"]),
        ..Default::default()
    };
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(r#"["not","an","object"]"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-object request_body should continue"
    );

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(
        rebuilt.is_array(),
        "non-object request_body should pass through without modification"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    super::ResponsesProxyFilter::from_config(&serde_yaml::Value::Null).unwrap()
}
