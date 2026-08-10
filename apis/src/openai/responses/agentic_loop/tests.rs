// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the agentic loop filter.

use bytes::Bytes;
use http::Method;
use praxis_filter::{FilterAction, HttpFilter};
use serde_json::{Value, json};

use super::super::state::ResponsesState;
use crate::test_utils::{make_filter_context, make_request};

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn from_config_accepts_null() {
    let yaml = serde_yaml::Value::Null;
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "agentic_loop");
}

#[test]
fn from_config_accepts_empty_mapping() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "agentic_loop");
}

#[test]
fn from_config_accepts_custom_values() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 5").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "agentic_loop");
}

#[test]
fn from_config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
    let result = super::AgenticLoopFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

#[test]
fn from_config_rejects_zero_max_infer_iters() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 0").unwrap();
    let result = super::AgenticLoopFilter::from_config(&yaml);
    assert!(result.is_err(), "max_infer_iters=0 should be rejected");
}

// -----------------------------------------------------------------------------
// Passthrough Without State
// -----------------------------------------------------------------------------

#[tokio::test]
async fn passthrough_without_state_on_request_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not write filter_results without state"
    );
}

#[test]
fn passthrough_without_state_on_response_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not write filter_results without state"
    );
}

// -----------------------------------------------------------------------------
// on_request_body Bookkeeping
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_body_clears_stale_tool_calls() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "on_request_body must clear stale tool_calls from previous round"
    );
    assert!(
        ctx.filter_results.is_empty(),
        "on_request_body should not set filter_results"
    );
}

#[tokio::test]
async fn tool_choice_preserved_on_first_iteration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.tool_choice = json!("required");
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.tool_choice,
        json!("required"),
        "tool_choice should be preserved on first iteration (iteration=0)"
    );
}

#[tokio::test]
async fn tool_choice_reset_after_first_iteration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.tool_choice = json!("required");
    state.iteration = 1;
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.tool_choice,
        json!("auto"),
        "tool_choice should be reset to auto after first iteration"
    );
    assert_eq!(
        state.request_body["tool_choice"], "auto",
        "tool_choice should be inserted into request_body for proxy serialization"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Content-Type on Re-entry
// -----------------------------------------------------------------------------

#[tokio::test]
async fn sets_content_type_on_reentry() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.iteration = 1;
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let has_content_type = ctx
        .request_headers_to_set
        .iter()
        .any(|(k, v)| k == http::header::CONTENT_TYPE && v == "application/json");
    assert!(has_content_type, "IRR re-entry must set content-type: application/json");
}

#[tokio::test]
async fn does_not_set_content_type_on_first_pass() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let has_content_type = ctx
        .request_headers_to_set
        .iter()
        .any(|(k, _)| k == http::header::CONTENT_TYPE);
    assert!(
        !has_content_type,
        "first pass relies on client content-type, filter must not set it"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Parallel Tool Calls
// -----------------------------------------------------------------------------

#[tokio::test]
async fn forces_parallel_tool_calls_false() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({"model": "gpt-4o", "input": "test", "tools": [{"type": "function"}]});
    let mut state = ResponsesState::from_request_body(body);
    assert!(state.parallel_tool_calls, "default should be true");
    assert_eq!(
        state.request_body.get("parallel_tool_calls"),
        None,
        "client did not set parallel_tool_calls"
    );

    state.iteration = 0;
    ctx.extensions.insert(state);
    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.parallel_tool_calls, "should be forced to false");
    assert_eq!(
        state.request_body["parallel_tool_calls"], false,
        "request_body should contain parallel_tool_calls=false for proxy serialization"
    );
    assert!(
        state.request_body_requires_rebuild(),
        "inserting parallel_tool_calls must require proxy serialization"
    );
}

#[tokio::test]
async fn preserves_unmodified_parallel_tool_calls_false() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({
        "model": "gpt-4o",
        "input": "test",
        "parallel_tool_calls": false,
        "tools": [{"type": "function"}]
    });
    ctx.extensions.insert(ResponsesState::from_request_body(body));

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        !state.request_body_requires_rebuild(),
        "an already-disabled request should retain byte-exact passthrough"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Reject Streaming
// -----------------------------------------------------------------------------

#[tokio::test]
async fn rejects_streaming_request() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({"model": "gpt-4o", "input": "test", "stream": true});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 400),
        "stream:true should produce a 400 rejection"
    );
}

#[tokio::test]
async fn streaming_rejection_preserves_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({"model": "gpt-4o", "input": "test", "stream": true});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "ResponsesState must remain in extensions after streaming rejection"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: No Tool Calls → Done
// -----------------------------------------------------------------------------

#[test]
fn no_tool_calls_sets_done() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
}

#[test]
fn state_survives_done_path() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>();
    assert!(
        state.is_some(),
        "ResponsesState must remain in extensions after done so downstream filters can read it"
    );
}

#[test]
fn non_end_of_stream_passes_through() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, false).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not set filter_results on non-end-of-stream chunks"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Tool Calls Present → Loop
// -----------------------------------------------------------------------------

#[test]
fn tool_calls_set_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": "{}",
    })]);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should be incremented");
}

#[test]
fn any_tool_type_sets_loop() {
    for tool_type in ["function", "mcp", "web_search", "file_search", "custom_tool"] {
        let filter = make_filter();
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);

        let state = make_state_with_tool_calls(vec![json!({
            "type": tool_type,
            "call_id": "call_1",
        })]);
        ctx.extensions.insert(state);

        drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
        assert_action(&ctx, "loop");
    }
}

// -----------------------------------------------------------------------------
// on_response_body: Config Defaults
// -----------------------------------------------------------------------------

#[test]
fn default_config_has_max_infer_iters_ten() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.iteration = 9;
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 10, "iteration should have incremented to 10");
    state.tool_calls = vec![json!({"type": "function", "call_id": "call_2", "name": "test"})];
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 508),
        "iteration 10 at default limit should produce 508 rejection"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Iteration Limit
// -----------------------------------------------------------------------------

#[test]
fn max_infer_iters_one_allows_exactly_one_loop() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 1").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "should have incremented to 1");

    state.tool_calls = vec![json!({"type": "function", "call_id": "call_2", "name": "test"})];
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 508),
        "second round at iteration limit should produce 508 rejection"
    );
}

#[test]
fn iteration_limit_returns_508_error() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 2").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.iteration = 2;
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 508),
        "iteration limit should produce a 508 rejection"
    );

    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "ResponsesState should be preserved after iteration limit rejection"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Multiple Function Calls
// -----------------------------------------------------------------------------

#[test]
fn multiple_function_calls_returns_error() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "get_time",
                "arguments": r#"{"timezone":"PST"}"#,
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 400),
        "multiple function calls should produce a 400 rejection"
    );

    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "ResponsesState should be preserved after rejection"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Reasoning Items
// -----------------------------------------------------------------------------

#[test]
fn reasoning_items_preserved_in_messages() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "thinking..."}]
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "single function call with reasoning should continue"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_calls.len(), 1, "only function_call goes to tool_calls");

    let msg_types: Vec<&str> = state
        .messages
        .iter()
        .filter_map(|m| m.get("type").and_then(Value::as_str))
        .collect();
    assert!(
        msg_types.contains(&"reasoning"),
        "reasoning item should be in messages: {msg_types:?}"
    );
    assert!(
        msg_types.contains(&"function_call"),
        "function_call item should be in messages: {msg_types:?}"
    );

    let persisted_types: Vec<&str> = state
        .persisted_messages
        .iter()
        .filter_map(|m| m.get("type").and_then(Value::as_str))
        .collect();
    assert!(
        persisted_types.contains(&"reasoning"),
        "reasoning item should be in persisted_messages"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Finish Reason Length
// -----------------------------------------------------------------------------

#[test]
fn finish_reason_length_exits_as_incomplete() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.response_object = json!({
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    });
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(
        status,
        Some("incomplete"),
        "should set incomplete status on finish_reason length"
    );
}

#[test]
fn finish_reason_length_passes_body_unchanged() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
            "status": "completed"
        }]
    });
    let original_bytes = serde_json::to_vec(&response_body).unwrap();
    let mut body = Some(Bytes::from(original_bytes.clone()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "model-owned incomplete should continue, not reject"
    );
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(
        status,
        Some("incomplete"),
        "should set incomplete metadata for model-owned reason"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Iteration Counter
// -----------------------------------------------------------------------------

#[test]
fn iteration_incremented_on_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should increment from 0 to 1");
}

// -----------------------------------------------------------------------------
// on_response_body: Filter Results Schema
// -----------------------------------------------------------------------------

#[test]
fn filter_results_schema_for_irr_consumers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

    let results = ctx
        .filter_results
        .get("agentic_loop")
        .expect("IRR consumers require agentic_loop entry");
    let action = results.get("action").expect("IRR consumers require action key");
    assert!(
        action == "loop" || action == "done",
        "action must be 'loop' or 'done', got: {action}"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Body Extraction (non-streaming)
// -----------------------------------------------------------------------------

#[test]
fn extracts_tool_calls_from_non_streaming_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_calls.len(), 1);
    assert_eq!(state.tool_calls[0]["call_id"], "call_1");
}

#[test]
fn appends_function_calls_to_messages() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages.len(), 2, "original input + function_call");
    assert_eq!(state.messages[1]["type"], "function_call");
    assert_eq!(
        state.persisted_messages.len(),
        2,
        "original input + function_call in persisted_messages"
    );
}

#[test]
fn skips_extraction_when_body_is_none() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "done");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.tool_calls.is_empty(), "should not extract from None body");
    assert_eq!(state.messages.len(), 1, "only the original normalized input");
}

#[test]
fn ignores_non_completed_function_calls() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "status": "in_progress"
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}]
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_action(&ctx, "done");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.tool_calls.is_empty(), "non-completed calls should be ignored");
}

#[test]
fn stores_response_object_from_body() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": []
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["id"], "resp_1");
    assert_eq!(state.response_object["status"], "completed");
}

#[test]
fn parse_failure_clears_stale_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_stale",
        "name": "leftover",
    })]);
    state.response_object = json!({"id": "resp_old", "status": "completed"});
    ctx.extensions.insert(state);

    let invalid: &[u8] = b"not valid json";
    let mut body = Some(Bytes::from(invalid));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.response_object.is_null(),
        "parse failure must clear stale response_object"
    );
    assert!(state.tool_calls.is_empty(), "parse failure must clear stale tool_calls");
    assert_action(&ctx, "done");
}

// -----------------------------------------------------------------------------
// on_response_body: Usage Accumulation
// -----------------------------------------------------------------------------

#[test]
fn accumulates_usage_across_rounds() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let round1 = json!({
        "id": "resp_1",
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
            "status": "completed"
        }],
        "usage": {"input_tokens": 100, "output_tokens": 50}
    });
    let mut body1 = Some(Bytes::from(serde_json::to_vec(&round1).unwrap()));
    drop(filter.on_response_body(&mut ctx, &mut body1, true).unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.usage["input_tokens"], 100);
    assert_eq!(state.usage["output_tokens"], 50);

    state.tool_calls.clear();
    ctx.extensions.insert(state);

    let round2 = json!({
        "id": "resp_2",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 200, "output_tokens": 75}
    });
    let mut body2 = Some(Bytes::from(serde_json::to_vec(&round2).unwrap()));
    drop(filter.on_response_body(&mut ctx, &mut body2, true).unwrap());
    assert_action(&ctx, "done");

    let terminal: Value = serde_json::from_slice(body2.as_ref().unwrap()).unwrap();
    assert_eq!(
        terminal["usage"]["input_tokens"], 300,
        "input_tokens should sum across rounds"
    );
    assert_eq!(
        terminal["usage"]["output_tokens"], 125,
        "output_tokens should sum across rounds"
    );
}

// -----------------------------------------------------------------------------
// Example Config Parse
// -----------------------------------------------------------------------------

#[test]
fn example_config_agentic_loop_parses() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("examples/configs/openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let config: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();

    let filters = config["filter_chains"][0]["filters"]
        .as_sequence()
        .expect("should have filters array");
    let irr = filters
        .iter()
        .find(|f| f["filter"].as_str() == Some("iterative_request_router"))
        .expect("should have iterative_request_router filter");
    let inference_step = irr["steps"]
        .as_sequence()
        .expect("should have steps array")
        .iter()
        .find(|s| s["name"].as_str() == Some("inference"))
        .expect("should have inference step");
    let step_filters = inference_step["filters"]
        .as_sequence()
        .expect("inference step should have filters");
    let al_config = step_filters
        .iter()
        .find(|f| f["filter"].as_str() == Some("agentic_loop"))
        .expect("inference step should have agentic_loop filter");
    let filter = super::AgenticLoopFilter::from_config(al_config).unwrap();
    assert_eq!(filter.name(), "agentic_loop");
}

// -----------------------------------------------------------------------------
// on_response_body: web_search_call Extraction
// -----------------------------------------------------------------------------

#[test]
fn web_search_call_extracted_to_web_search_calls_not_tool_calls() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "rust async"}
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "web_search_call must not appear in tool_calls"
    );
    assert_eq!(
        state.web_search_calls.len(),
        1,
        "web_search_call must appear in web_search_calls"
    );
    assert_eq!(state.web_search_calls[0]["id"], "ws_1");
}

#[test]
fn web_search_call_triggers_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.web_search_calls = vec![json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "loop");
}

#[test]
fn web_search_call_alone_increments_iteration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.web_search_calls = vec![json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    })];
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should increment from 0 to 1");
}

#[test]
fn mixed_function_and_web_search_calls() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "status": "completed"
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "weather SF"}
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_calls.len(), 1, "one function_call in tool_calls");
    assert_eq!(
        state.web_search_calls.len(),
        1,
        "one web_search_call in web_search_calls"
    );
    assert_eq!(state.tool_calls[0]["call_id"], "call_1");
    assert_eq!(state.web_search_calls[0]["id"], "ws_1");
}

#[test]
fn web_search_call_subject_to_iteration_limit() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 2").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.web_search_calls = vec![json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    })];
    state.iteration = 2;
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 508),
        "web_search_call at iteration limit should produce 508 rejection"
    );
}

#[tokio::test]
async fn web_search_calls_cleared_on_prepare() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.web_search_calls = vec![json!({
        "type": "web_search_call",
        "id": "ws_stale",
        "action": {"type": "search", "query": "old query"}
    })];
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.web_search_calls.is_empty(),
        "on_request_body must clear stale web_search_calls from previous round"
    );
}

#[test]
fn web_search_call_appended_to_messages_and_persisted() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "test query"}
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let msg_types: Vec<&str> = state
        .messages
        .iter()
        .filter_map(|m| m.get("type").and_then(Value::as_str))
        .collect();
    assert!(
        msg_types.contains(&"web_search_call"),
        "web_search_call should be in messages: {msg_types:?}"
    );

    let persisted_types: Vec<&str> = state
        .persisted_messages
        .iter()
        .filter_map(|m| m.get("type").and_then(Value::as_str))
        .collect();
    assert!(
        persisted_types.contains(&"web_search_call"),
        "web_search_call should be in persisted_messages: {persisted_types:?}"
    );

    assert!(
        state
            .accumulated_output
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call")),
        "web_search_call should be in accumulated_output"
    );
}

#[test]
fn web_search_call_does_not_count_as_function_call_for_limit() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "query 1"}
            },
            {
                "type": "web_search_call",
                "id": "ws_2",
                "status": "completed",
                "action": {"type": "search", "query": "query 2"}
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "multiple web_search_calls should not trigger the one-function-call limit"
    );
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.web_search_calls.len(), 2);
    assert!(state.tool_calls.is_empty());
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    super::AgenticLoopFilter::from_config(&serde_yaml::Value::Null).unwrap()
}

fn make_state_with_tool_calls(tool_calls: Vec<Value>) -> ResponsesState {
    let body = json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.tool_calls = tool_calls;
    state
}

fn assert_action(ctx: &praxis_filter::HttpFilterContext<'_>, expected: &str) {
    let results = ctx
        .filter_results
        .get("agentic_loop")
        .expect("filter_results should contain agentic_loop entry");
    let action = results.get("action").expect("should have action key");
    assert_eq!(action, expected, "agentic_loop action mismatch");
}
