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
async fn initialized_state_preserves_scalar_input_on_first_pass() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request_body = json!({
        "model": "gpt-4o",
        "input": "keep this representation"
    });
    ctx.extensions
        .insert(ResponsesState::from_request_body(request_body.clone()));
    let mut body = Some(Bytes::from(serde_json::to_vec(&request_body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rebuilt["input"], "keep this representation",
        "an unmodified first pass must preserve the client's scalar input"
    );
}

#[tokio::test]
async fn deferred_file_search_first_pass_preserves_body_bytes_without_rebuild() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = br#"{
  "model": "gpt-4.1",
  "input": [{"type":"message", "role":"user", "content":"keep formatting"}],
  "tools": [{"type":"file_search", "vector_store_ids":["vs_1"]}]
}"#;
    let parsed = serde_json::from_slice(original).unwrap();
    ctx.extensions
        .insert(ResponsesState::from_file_search_request_body(parsed));
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(
        ctx.extra_request_headers.is_empty(),
        "byte-exact first-pass forwarding must not synthesize content-length"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.has_deferred_history());
    assert!(state.messages.is_empty());
    assert!(state.persisted_messages.is_empty());
}

#[tokio::test]
async fn provider_previous_response_id_is_byte_exact_without_rehydrate() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = br#"{
  "model": "gpt-4.1",
  "input": "continue",
  "tools": [{"type":"file_search", "vector_store_ids":["vs_1"]}],
  "previous_response_id": "resp_provider"
}"#;
    let parsed = serde_json::from_slice(original).unwrap();
    ctx.extensions.insert(ResponsesState::from_request_body(parsed));
    let mut body = Some(Bytes::copy_from_slice(original));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body.as_deref(), Some(original.as_slice()));
    assert!(ctx.extra_request_headers.is_empty());
}

#[tokio::test]
async fn previous_history_id_passes_through_when_rehydrate_did_not_run() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let request = json!({
        "model": "gpt-4.1",
        "input": "search",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "previous_response_id": null,
        "conversation": null
    });
    ctx.extensions
        .insert(ResponsesState::from_file_search_request_body(request.clone()));
    let mut body = Some(Bytes::from(serde_json::to_vec(&request).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(outbound["previous_response_id"].is_null());
    assert!(outbound.get("conversation").is_none());
    assert_eq!(outbound["input"], "search");
    assert_eq!(outbound["tools"], request["tools"]);
    assert_eq!(
        ctx.extra_request_headers
            .iter()
            .find(|(name, _value)| name.as_ref() == "content-length")
            .map(|(_name, value)| value.parse::<usize>().unwrap()),
        body.as_ref().map(Bytes::len)
    );
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().has_deferred_history());
}

#[tokio::test]
async fn rebuild_uses_request_body_mutated_after_state_initialization() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original = json!({
        "model": "client-model",
        "input": "hello",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
    });
    let mut state = ResponsesState::from_request_body(original);
    state.messages.splice(
        0..0,
        [json!({"type": "message", "role": "assistant", "content": "history"})],
    );
    ctx.extensions.insert(state);
    let rewritten = json!({
        "model": "backend-model",
        "input": "hello",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&rewritten).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["model"], "backend-model", "later body rewrites must survive");
    assert_eq!(
        outbound["input"].as_array().unwrap().len(),
        2,
        "state history should still replace input"
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
    state.history_rehydrated = true;
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
    assert_eq!(
        rebuilt["input"], "hello",
        "stripping conversation must not normalize unchanged scalar input"
    );
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
    assert_eq!(
        ctx.extra_request_headers
            .iter()
            .find(|(name, _value)| name.as_ref() == "content-length")
            .map(|(_name, value)| value.parse::<usize>().unwrap()),
        body.as_ref().map(Bytes::len)
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
async fn continuation_resets_forced_tool_choice_to_state_value() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::from_request_body(json!({
        "model":"gpt-4o",
        "input":"search",
        "tool_choice":{"type":"file_search"}
    }));
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!("auto"));
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(
        r#"{"model":"gpt-4o","input":"search","tool_choice":{"type":"file_search"}}"#,
    ));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["tool_choice"], "auto");
}

#[tokio::test]
async fn continuation_forwards_only_remaining_max_tool_calls_without_mutating_policy() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::from_file_search_request_body(json!({
        "model":"gpt-4.1",
        "input":"search",
        "tools":[{"type":"file_search","vector_store_ids":["vs_1"]}],
        "tool_choice":{"type":"file_search"},
        "max_tool_calls":3
    }));
    state.materialize_deferred_history();
    state.replace_output_items(vec![
        json!({"type":"file_search_call","id":"fs_1","status":"completed"}),
        json!({"type":"web_search_call","id":"ws_1","status":"completed"}),
        json!({"type":"message","id":"msg_1","content":[]}),
    ]);
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!("auto"));
    ctx.extensions.insert(state);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let rebuilt: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rebuilt["input"].as_array().unwrap().len(), 1);
    assert_eq!(rebuilt["tools"][0]["type"], "file_search");
    assert_eq!(rebuilt["tool_choice"], "auto");
    assert_eq!(rebuilt["max_tool_calls"], 1);
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.max_tool_calls, Some(3));
    assert_eq!(state.request_body["max_tool_calls"], 3);
}

#[tokio::test]
async fn exhausted_max_tool_calls_allows_answer_round_without_serializing_zero() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original_choice = json!({"type": "file_search"});
    let tools = json!([{"type": "file_search", "vector_store_ids": ["vs_1"]}]);
    let mut state = ResponsesState::from_file_search_request_body(json!({
        "model": "gpt-4.1",
        "input": "search",
        "tools": tools,
        "tool_choice": original_choice,
        "max_tool_calls": 1,
    }));
    state.materialize_deferred_history();
    state.replace_output_items(vec![json!({
        "type": "file_search_call",
        "id": "fs_1",
        "status": "completed"
    })]);
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "file_search"}]
    }));
    ctx.extensions.insert(state);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        outbound,
        json!({
            "model":"gpt-4.1",
            "tool_choice":"none",
            "input":[{"type":"message","role":"user","content":"search"}],
            "tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]
        })
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.max_tool_calls, Some(1));
    assert_eq!(state.tool_choice, original_choice);
    assert_eq!(state.request_body["max_tool_calls"], 1);
}

#[tokio::test]
async fn exhausted_builtin_budget_preserves_narrow_custom_tool_scope() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original_choice = json!({
        "type": "allowed_tools",
        "mode": "required",
        "tools": [
            {"type": "file_search"},
            {"type": "function", "name": "lookup"}
        ]
    });
    let mut state = ResponsesState::from_file_search_request_body(json!({
        "model": "gpt-4.1",
        "input": "search",
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
            {"type": "function", "name": "outside_scope", "parameters": {"type": "object"}}
        ],
        "tool_choice": original_choice,
        "max_tool_calls": 1,
    }));
    state.materialize_deferred_history();
    state.replace_output_items(vec![json!({
        "type": "file_search_call",
        "id": "fs_1",
        "status": "completed"
    })]);
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "file_search"},
            {"type": "function", "name": "lookup"}
        ]
    }));
    ctx.extensions.insert(state);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(outbound.get("max_tool_calls").is_none());
    assert_eq!(
        outbound["tool_choice"],
        json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [{"type": "function", "name": "lookup"}]
        })
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_choice, original_choice);
}

#[tokio::test]
async fn continuation_restores_fields_moved_out_of_deferred_fallback_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let original_choice = json!({
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "file_search"}, {"type": "function", "name": "other"}]
    });
    let continuation_choice = json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "file_search"}, {"type": "function", "name": "other"}]
    });
    let mut state = ResponsesState::from_file_search_request_body(json!({
        "model": "gpt-4.1",
        "input": "search",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "context_management": {"type": "custom", "payload": "large"},
        "conversation": {"id": ""},
        "include": ["file_search_call.results"],
        "previous_response_id": "",
        "tool_choice": original_choice,
    }));
    state.materialize_deferred_history();
    state.iteration = 1;
    state.continuation_tool_choice = Some(continuation_choice.clone());
    ctx.extensions.insert(state);
    let mut body = None;

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let outbound: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(outbound["input"][0]["content"], "search");
    assert_eq!(outbound["tools"][0]["type"], "file_search");
    assert_eq!(outbound["context_management"]["payload"], "large");
    assert_eq!(outbound["include"], json!(["file_search_call.results"]));
    assert_eq!(outbound["tool_choice"], continuation_choice);
    assert!(outbound.get("conversation").is_none());
    assert_eq!(outbound["previous_response_id"], "");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_choice, original_choice);
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    super::ResponsesProxyFilter::from_config(&serde_yaml::Value::Null).unwrap()
}
