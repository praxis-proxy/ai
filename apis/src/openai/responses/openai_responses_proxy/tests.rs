// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the Responses proxy filter.

use base64::Engine as _;
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
async fn initialized_state_preserves_scalar_input_on_first_pass() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({"model":"gpt-4.1","input":"hello"});
    ctx.extensions.insert(ResponsesState::from_request_body(request_body));
    let original = br#"{
  "model": "gpt-4.1",
  "input": "hello"
}"#;
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(
        ctx.request_headers_to_set.is_empty(),
        "byte-exact first-pass forwarding must not synthesize content-length"
    );
}

#[tokio::test]
async fn provider_previous_response_id_is_byte_exact_without_rehydrate() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = br#"{
  "model": "gpt-4.1",
  "input": "continue",
  "previous_response_id": "resp_provider"
}"#;
    let parsed = serde_json::from_slice(original).unwrap();
    ctx.extensions.insert(ResponsesState::from_request_body(parsed));
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(ctx.request_headers_to_set.is_empty());
}

#[tokio::test]
async fn rebuild_serializes_from_state_request_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = json!({"model":"client-model","input":"hello"});
    let mut state = ResponsesState::from_request_body(original);
    state
        .messages
        .splice(0..0, [json!({"type":"message","role":"assistant","content":"history"})]);
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(br#"{"model":"client-model","input":"hello"}"#.as_slice()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["model"], "client-model", "serializes from state.request_body");
    assert_eq!(outbound["input"].as_array().unwrap().len(), 2);
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
    state.history_rehydrated = true;
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
async fn does_not_set_content_length_header() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-length"),
        "filter must not set content-length (core handles framing)"
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
    state.history_rehydrated = true;
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
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
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
    state.history_rehydrated = true;
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
    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-length"),
        "filter must not set content-length (core handles framing)"
    );
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

#[tokio::test]
async fn tool_choice_serialized_from_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "hello",
        "tool_choice": "auto"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.tool_choice = json!("none");
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"hello","tool_choice":"auto"}"#,
    ));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rebuilt["tool_choice"], "none",
        "tool_choice should come from state, not from original request_body"
    );
}

#[tokio::test]
async fn rebuild_does_not_set_content_type_header() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let request_body = json!({
        "model": "gpt-4o",
        "input": "test",
        "previous_response_id": "resp_abc123"
    });
    let mut state = ResponsesState::from_request_body(request_body);
    state.history_rehydrated = true;
    state
        .messages
        .splice(0..0, vec![json!({"role": "user", "content": "stored"})]);
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"test","previous_response_id":"resp_abc123"}"#,
    ));
    let _action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        ctx.extra_request_headers
            .iter()
            .all(|(k, _)| k.as_ref() != "content-type"),
        "filter must not set content-type (core handles framing)"
    );
}

// -----------------------------------------------------------------------------
// messages_for_backend / compaction_to_assistant_message
// -----------------------------------------------------------------------------

#[test]
fn messages_for_backend_borrows_when_no_compaction() {
    let msgs = vec![
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "hi"}),
    ];
    let result = super::messages_for_backend(&msgs);
    assert!(
        matches!(result, std::borrow::Cow::Borrowed(_)),
        "should borrow when no compaction items"
    );
    assert_eq!(result.len(), 2);
}

#[test]
fn messages_for_backend_translates_compaction_item() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("summary text");
    let msgs = vec![json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded})];
    let result = super::messages_for_backend(&msgs);
    assert!(matches!(result, std::borrow::Cow::Owned(_)));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["role"], "assistant");
    assert!(result[0]["content"].as_str().unwrap().contains("summary text"));
}

#[test]
fn messages_for_backend_mixed_items() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("ctx");
    let msgs = vec![
        json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded}),
        json!({"role": "user", "content": "hello"}),
    ];
    let result = super::messages_for_backend(&msgs);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0]["role"], "assistant");
    assert_eq!(result[1]["role"], "user");
}

#[test]
fn compaction_to_assistant_message_decodes_encrypted_content() {
    let encoded = base64::engine::general_purpose::STANDARD.encode("decoded summary");
    let item = json!({"type": "compaction", "id": "c_1", "encrypted_content": encoded});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
    assert!(msg["content"].as_str().unwrap().contains("decoded summary"));
}

#[test]
fn compaction_to_assistant_message_handles_missing_content() {
    let item = json!({"type": "compaction", "id": "c_1"});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
    assert!(
        msg["content"]
            .as_str()
            .unwrap()
            .contains("[Previous conversation summary]")
    );
}

#[test]
fn compaction_to_assistant_message_handles_invalid_base64() {
    let item = json!({"type": "compaction", "id": "c_1", "encrypted_content": "not!valid!base64!!!"});
    let msg = super::compaction_to_assistant_message(&item);
    assert_eq!(msg["role"], "assistant");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    super::ResponsesProxyFilter::from_config(&serde_yaml::Value::Null).unwrap()
}
