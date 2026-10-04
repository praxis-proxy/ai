// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the agentic loop filter.

use bytes::Bytes;
use http::Method;
use praxis_filter::{FilterAction, HttpFilter, SubRequestResponseMode, TrustedHeaderMutation};
use serde_json::{Value, json};

use super::super::state::ResponsesState;
#[cfg(feature = "openai-mcp-tools")]
use crate::openai::responses::state::DeferredMcpConnector;
use crate::{
    openai::responses::state::{DispatchFailure, FileSearchAssignment, McpApprovalState, SynthesisKind},
    test_utils::{make_filter_context, make_request},
};

#[test]
fn buffered_round_moves_public_output_payload_without_reallocating_it() {
    let mut response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "client_tool",
            "arguments": "x".repeat(256 * 1024),
            "status": "completed"
        }]
    });
    let payload_ptr = response["output"][0]["arguments"].as_str().unwrap().as_ptr();
    let mut state = ResponsesState::default();
    super::collect_output_items(&mut response, &mut state, &[]);

    assert!(response["output"].as_array().unwrap().is_empty());
    assert_eq!(
        state.accumulated_output[0]["arguments"].as_str().unwrap().as_ptr(),
        payload_ptr
    );
    assert_eq!(state.selected_tool_calls()[0]["id"], "fc_1");
    assert_ne!(state.messages[0]["arguments"].as_str().unwrap().as_ptr(), payload_ptr);
    assert_ne!(
        state.persisted_messages[0]["arguments"].as_str().unwrap().as_ptr(),
        payload_ptr
    );
}

#[test]
fn indexed_buffered_collection_allocates_less_than_payload_dispatch_copies() {
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_large",
            "call_id": "call_large",
            "name": "server__lookup",
            "arguments": "x".repeat(256 * 1024),
            "status": "completed"
        }]
    });
    let mut old_response = response.clone();
    let mut new_response = response;
    let mut old_state = ResponsesState::default();
    let mut new_state = ResponsesState::default();
    let mut old_dispatch = Vec::new();

    let old = allocation_counter::measure(|| {
        for item in old_response["output"].as_array_mut().unwrap().iter() {
            old_state.accumulated_output.push(item.clone());
            old_dispatch.push(item.clone());
            old_state.messages.push(item.clone());
            old_state.persisted_messages.push(item.clone());
        }
        std::hint::black_box(&old_dispatch);
    });
    let new = allocation_counter::measure(|| {
        super::collect_output_items(&mut new_response, &mut new_state, &[]);
        std::hint::black_box(&new_state);
    });

    assert_eq!(new_state.selected_tool_calls().len(), 1);
    assert_eq!(new_state.messages, old_state.messages);
    assert_eq!(new_state.persisted_messages, old_state.persisted_messages);
    assert_eq!(new_state.accumulated_output, old_state.accumulated_output);
    assert!(
        old.bytes_total >= new.bytes_total + 2 * 256 * 1024,
        "removing the accumulated and dispatch deep copies should save two payloads: old={} new={}",
        old.bytes_total,
        new.bytes_total
    );
}

#[test]
fn streamed_round_moves_public_output_payload_without_reallocating_it() {
    let mut state = ResponsesState {
        response_object: json!({
            "object": "response",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "client_tool",
                "arguments": "x".repeat(256 * 1024),
                "status": "completed"
            }]
        }),
        ..ResponsesState::default()
    };
    let payload_ptr = state.response_object["output"][0]["arguments"]
        .as_str()
        .unwrap()
        .as_ptr();
    super::collect_streaming_output_items(&mut state).unwrap();

    assert!(state.output_items().is_empty());
    assert_eq!(
        state.accumulated_output[0]["arguments"].as_str().unwrap().as_ptr(),
        payload_ptr
    );
    assert_eq!(state.selected_tool_calls()[0]["id"], "fc_1");
}

#[test]
fn buffered_round_assignments_keep_output_and_history_order_across_rounds() {
    let mut state = ResponsesState::default();
    let mut first = json!({"object": "response", "output": [
        {"type": "reasoning", "id": "r_1", "summary": []},
        {"type": "function_call", "id": "fc_1", "call_id": "c_1", "name": "server__lookup", "arguments": "{}", "status": "completed"},
        {"type": "function_call", "id": "fc_partial", "call_id": "c_partial", "name": "client", "status": "in_progress"},
        {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "query": "q"}},
        {"type": "tool_search_call", "id": "ts_1", "status": "completed"}
    ]});
    super::collect_output_items(&mut first, &mut state, &[]);
    assert_eq!(state.selected_tool_calls().len(), 1);
    assert_eq!(state.selected_web_search_calls().len(), 1);
    assert_eq!(state.selected_tool_search_calls().len(), 1);
    let first_assignment = state.tool_calls[0].clone();

    let tool_output = json!({"type": "function_call_output", "call_id": "c_1", "output": "found"});
    state.messages.push(tool_output.clone());
    state.persisted_messages.push(tool_output);
    super::prepare_iteration(&mut state);
    let mut second = json!({"object": "response", "output": [
        {"type": "reasoning", "id": "r_2", "summary": []},
        {"type": "function_call", "id": "fc_2", "call_id": "c_2", "name": "client", "arguments": "{}", "status": "completed"}
    ]});
    super::collect_output_items(&mut second, &mut state, &[]);

    let ids: Vec<_> = state
        .accumulated_output
        .iter()
        .map(|item| item["id"].as_str())
        .collect();
    assert_eq!(
        ids,
        vec![
            Some("r_1"),
            Some("fc_1"),
            Some("fc_partial"),
            Some("ws_1"),
            Some("ts_1"),
            Some("r_2"),
            Some("fc_2")
        ]
    );
    let message_types: Vec<_> = state.messages.iter().map(|item| item["type"].as_str()).collect();
    assert_eq!(
        message_types,
        vec![
            Some("reasoning"),
            Some("function_call"),
            Some("function_call_output"),
            Some("reasoning"),
            Some("function_call")
        ]
    );
    let persisted_types: Vec<_> = state
        .persisted_messages
        .iter()
        .map(|item| item["type"].as_str())
        .collect();
    assert_eq!(
        persisted_types,
        vec![
            Some("reasoning"),
            Some("function_call"),
            Some("web_search_call"),
            Some("tool_search_call"),
            Some("function_call_output"),
            Some("reasoning"),
            Some("function_call")
        ]
    );
    assert_eq!(state.selected_tool_calls().len(), 1);
    assert_eq!(state.selected_tool_calls()[0]["id"], "fc_2");
    assert!(
        first_assignment
            .resolve(&state.accumulated_output, "web_search_call")
            .is_none()
    );
}

#[test]
fn streamed_round_assignments_keep_one_output_owner_across_rounds() {
    let mut state = ResponsesState {
        response_object: json!({"object": "response", "status": "completed", "output": [
            {"type": "function_call", "id": "fc_1", "call_id": "c_1", "name": "server__lookup", "arguments": "{}", "status": "completed"},
            {"type": "web_search_call", "id": "ws_1", "action": {"type": "search", "query": "first"}}
        ]}),
        ..ResponsesState::default()
    };
    super::collect_streaming_output_items(&mut state).unwrap();
    assert!(state.output_items().is_empty());
    assert_eq!(state.selected_tool_calls()[0]["id"], "fc_1");
    assert_eq!(state.selected_web_search_calls()[0]["id"], "ws_1");
    let first_assignment = state.tool_calls[0].clone();

    super::prepare_iteration(&mut state);
    state.response_object = json!({"object": "response", "status": "completed", "output": [
        {"type": "reasoning", "id": "r_2", "summary": []},
        {"type": "function_call", "id": "fc_2", "call_id": "c_2", "name": "client", "arguments": "{}", "status": "completed"},
        {"type": "tool_search_call", "id": "ts_2", "status": "completed"}
    ]});
    super::collect_streaming_output_items(&mut state).unwrap();

    assert_eq!(state.current_round_output_start, Some(2));
    assert!(state.output_items().is_empty());
    assert_eq!(state.selected_tool_calls().len(), 1);
    assert_eq!(state.selected_tool_calls()[0]["id"], "fc_2");
    assert!(state.selected_web_search_calls().is_empty());
    assert_eq!(state.selected_tool_search_calls()[0]["id"], "ts_2");
    assert_eq!(
        state
            .accumulated_output
            .iter()
            .map(|item| item["id"].as_str())
            .collect::<Vec<_>>(),
        [Some("fc_1"), Some("ws_1"), Some("r_2"), Some("fc_2"), Some("ts_2")]
    );
    assert_eq!(
        state
            .messages
            .iter()
            .map(|item| item["id"].as_str())
            .collect::<Vec<_>>(),
        [Some("fc_1"), Some("r_2"), Some("fc_2")]
    );
    assert_eq!(
        state
            .persisted_messages
            .iter()
            .map(|item| item["id"].as_str())
            .collect::<Vec<_>>(),
        [Some("fc_1"), Some("ws_1"), Some("r_2"), Some("fc_2"), Some("ts_2")]
    );
    state.tool_calls.push(first_assignment);
    assert_eq!(
        state.selected_tool_calls().len(),
        1,
        "a stale prior-round selection cannot dispatch"
    );
}

#[test]
fn stale_tool_search_selection_cannot_rewrite_prior_output_or_history() {
    let prior = json!({"type": "tool_search_call", "id": "ts_duplicate", "status": "completed", "action": "prior"});
    let current = json!({"type": "tool_search_call", "id": "ts_duplicate", "status": "completed", "action": "current"});
    let mut state = ResponsesState {
        accumulated_output: vec![prior.clone(), current.clone()],
        persisted_messages: vec![prior.clone(), current],
        max_tool_calls: Some(0),
        ..ResponsesState::default()
    };
    state.select_test_output("tool_search_call", vec![state.accumulated_output[1].clone()]);
    state.accumulated_output[1] = json!({"type": "function_call", "id": "ts_duplicate"});

    super::mark_over_budget_tool_searches_incomplete(&mut state);

    assert!(state.tool_search_calls.is_empty());
    assert_eq!(state.accumulated_output[0], prior);
    assert_eq!(state.persisted_messages[0]["status"], "completed");
    assert_eq!(state.persisted_messages[1]["status"], "completed");
}

#[test]
fn over_budget_tool_search_reusing_prior_id_marks_only_current_round() {
    let prior = json!({"type": "tool_search_call", "id": "ts_duplicate", "status": "completed", "action": "prior"});
    let current = json!({"type": "tool_search_call", "id": "ts_duplicate", "status": "completed", "action": "current"});
    let mut state = ResponsesState {
        accumulated_output: vec![prior.clone(), current.clone()],
        persisted_messages: vec![prior.clone(), current.clone()],
        current_round_output_start: Some(1),
        max_tool_calls: Some(0),
        ..ResponsesState::default()
    };
    state.select_test_output("tool_search_call", vec![current]);

    super::mark_over_budget_tool_searches_incomplete(&mut state);

    assert!(state.tool_search_calls.is_empty());
    assert_eq!(state.accumulated_output[0], prior);
    assert_eq!(state.persisted_messages[0], prior);
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
    assert_eq!(state.persisted_messages[1]["status"], "incomplete");
}

// -----------------------------------------------------------------------------
// Config Parsing
// -----------------------------------------------------------------------------

#[test]
fn from_config_accepts_null() {
    let yaml = serde_yaml::Value::Null;
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_agentic_loop");
}

#[test]
fn from_config_accepts_empty_mapping() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_agentic_loop");
}

#[test]
fn from_config_accepts_custom_values() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 5").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_agentic_loop");
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

#[test]
fn from_config_rejects_out_of_range_retained_byte_budgets() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_retained_bytes: 4095").unwrap();
    let expected = "openai_agentic_loop: max_retained_bytes must be in 4096..=268435456, got 4095";

    let filter_error = super::AgenticLoopFilter::from_config(&yaml)
        .err()
        .expect("filter config should reject an out-of-range byte budget");
    assert!(filter_error.to_string().contains(expected));

    let policy_error = super::AgenticBudgetPolicy::from_config(&yaml)
        .err()
        .expect("budget policy should reject an out-of-range byte budget");
    assert!(policy_error.to_string().contains(expected));
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

#[tokio::test]
async fn deferred_tool_limit_completes_after_request_side_dispatchers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        deferred_tool_limit_completion: true,
        response_object: json!({"id":"resp_limit", "object":"response", "status":"completed", "output":[]}),
        accumulated_output: vec![json!({
            "type":"web_search_call", "id":"ws_rejected", "status":"failed",
            "error":"max_tool_calls exhausted"
        })],
        ..ResponsesState::default()
    });

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("the last request-side filter must complete the response locally");
    };
    assert_eq!(response.status, 200);
    let body: Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["output"][0]["id"], "ws_rejected");
    assert_eq!(ctx.filter_results["openai_agentic_loop"].get("action"), Some("done"));
    assert!(
        !ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .deferred_tool_limit_completion
    );
}

#[tokio::test]
async fn streamed_tool_limit_completion_restores_reentry_response_template() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        deferred_tool_limit_completion: true,
        deferred_stream_done: true,
        request_body: json!({"stream": true}),
        response_object: Value::Null,
        local_completion_response_template: json!({
            "id":"resp_limit", "object":"response", "status":"completed", "output":[]
        }),
        accumulated_output: vec![json!({
            "type":"web_search_call", "id":"ws_rejected", "status":"failed",
            "error":"max_tool_calls exhausted"
        })],
        ..ResponsesState::default()
    });

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("the streaming tool-limit response must complete locally");
    };
    assert_eq!(response.status, 200);
    let body = std::str::from_utf8(response.body.as_deref().unwrap()).unwrap();
    assert!(
        body.contains("event: response.completed"),
        "the local stream must include a response.completed terminal: {body}"
    );
    assert!(
        body.contains("\"id\":\"ws_rejected\""),
        "the terminal snapshot must include accumulated output: {body}"
    );
    assert!(
        body.ends_with("data: [DONE]\n\n"),
        "the local stream must preserve the deferred done sentinel: {body}"
    );
}

#[tokio::test]
async fn deferred_mcp_approval_completes_after_sibling_dispatchers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        mcp_approval_state: McpApprovalState::ApprovalPendingThenReturn,
        response_object: json!({"id":"resp_approval", "object":"response", "status":"completed", "output":[]}),
        accumulated_output: vec![json!({
            "type":"mcp_approval_request", "id":"approval_1", "name":"dangerous"
        })],
        ..ResponsesState::default()
    });

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("the trailing loop filter must complete the approval response locally");
    };
    assert_eq!(response.status, 200);
    let body: Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["output"][0]["id"], "approval_1");
    assert_eq!(ctx.filter_results["openai_agentic_loop"].get("action"), Some("done"));
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().mcp_approval_state,
        McpApprovalState::None
    );
}

#[tokio::test]
async fn streamed_mcp_approval_completion_restores_reentry_response_template() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        mcp_approval_state: McpApprovalState::ApprovalPendingThenReturn,
        request_body: json!({"stream": true}),
        response_object: Value::Null,
        local_completion_response_template: json!({
            "id":"resp_approval", "object":"response", "status":"completed", "output":[]
        }),
        accumulated_output: vec![json!({
            "type":"mcp_approval_request", "id":"approval_1", "name":"dangerous"
        })],
        ..ResponsesState::default()
    });

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("the streaming approval response must complete locally");
    };
    assert_eq!(response.status, 200);
    let body = std::str::from_utf8(response.body.as_deref().unwrap()).unwrap();
    assert!(
        body.contains("event: response.completed"),
        "the local stream must include a response.completed terminal: {body}"
    );
    assert!(
        body.contains("\"id\":\"approval_1\""),
        "the terminal snapshot must include the approval request: {body}"
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
// on_request: fail closed on unsafe terminal-streaming configuration
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_rejects_typed_streaming_without_logical_stream() {
    // openai_responses_proxy already selected the typed streaming transport for
    // this round, but openai_stream_events published no logical-stream marker
    // (the filter is absent from this step, so nothing armed a finalizer). A
    // loop-terminal error could not reach the client, so this must fail closed
    // before dispatch.
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);

    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("unsafe agentic terminal streaming must fail closed before dispatch");
    };
    assert_eq!(
        rejection.status, 500,
        "server misconfiguration must fail closed with 500, not commit a truncatable stream"
    );
    let error_body = std::str::from_utf8(rejection.body.as_deref().unwrap()).unwrap();
    assert!(
        error_body.contains("server_error"),
        "rejection must carry the server_error code: {error_body}"
    );
}

#[tokio::test]
async fn on_request_allows_typed_streaming_with_logical_stream() {
    // openai_stream_events armed a logical-stream finalizer this round, so a
    // loop-terminal error can still reach the client: proceed.
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.set_metadata("responses.logical_stream", "true");

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "agentic terminal streaming with a logical-stream finalizer must proceed"
    );
    // The marker is consumed so a stale "true" cannot satisfy a later IRR step
    // whose own filters did not re-publish it.
    assert_eq!(ctx.get_metadata("responses.logical_stream"), Some("false"));
}

#[tokio::test]
async fn on_request_allows_buffered_mode_without_logical_stream() {
    // A buffered sub-request retains full loop-terminal error handling, so the
    // fail-closed check does not apply even without a logical-stream finalizer.
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "a buffered sub-request must not be rejected by the streaming fail-closed check"
    );
}

#[tokio::test]
async fn selected_adapter_enforces_deferred_streaming_guard() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);

    let rejection =
        super::enforce_agentic_stream_guard(&mut ctx).expect("the selected adapter must reject unsafe typed streaming");
    assert_eq!(rejection.status, 500);
}

#[tokio::test]
async fn selected_adapter_accepts_deferred_streaming_with_logical_stream() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.logical_stream", "true");

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);

    assert!(
        super::enforce_agentic_stream_guard(&mut ctx).is_none(),
        "an armed logical stream should satisfy the deferred guard"
    );
    assert_eq!(ctx.get_metadata("responses.logical_stream"), Some("false"));
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
        state.original_tool_choice,
        Some(json!("required")),
        "client-visible tool_choice should survive internal continuation resets"
    );
    assert_eq!(
        state.request_body["tool_choice"], "auto",
        "tool_choice should be inserted into request_body for proxy serialization"
    );
}

#[test]
fn explicit_auto_tool_choice_invalidates_stable_charge_on_reentry() {
    let mut state = ResponsesState::from_request_body(json!({"input":[], "tool_choice":"auto"}));
    state.iteration = 1;
    let stable_before = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
    let revision_before = state.replay_stable_payload_revision;

    super::prepare_iteration(&mut state);

    assert_eq!(state.tool_choice, json!("auto"));
    assert_eq!(state.original_tool_choice, Some(json!("auto")));
    assert_eq!(state.request_body["tool_choice"], "auto");
    assert!(state.replay_stable_payload_revision > revision_before);
    assert!(state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap() > stable_before);
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

#[test]
fn continuation_headers_keep_order_in_the_legacy_queue() {
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    super::queue_continuation_header(&mut ctx, http::header::AUTHORIZATION, "Bearer token".parse().unwrap());
    super::queue_continuation_header(
        &mut ctx,
        http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );

    assert_eq!(
        ctx.request_headers_to_set,
        vec![
            (http::header::AUTHORIZATION, "Bearer token".parse().unwrap()),
            (http::header::CONTENT_TYPE, "application/json".parse().unwrap()),
        ],
        "continuation headers must queue into the legacy list in call order"
    );
    assert!(
        ctx.pre_read_mutations.is_empty(),
        "an empty ordered log must stay empty when no pre-read mutations exist"
    );
}

#[test]
fn continuation_headers_join_an_active_ordered_log() {
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(
        http::header::ACCEPT,
        "application/json".parse().unwrap(),
    ));
    super::queue_continuation_header(&mut ctx, http::header::AUTHORIZATION, "Bearer token".parse().unwrap());
    super::queue_continuation_header(
        &mut ctx,
        http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );

    assert_eq!(
        ctx.request_headers_to_set.len(),
        2,
        "both continuation headers must reach the legacy queue"
    );
    assert_eq!(
        ctx.pre_read_mutations.len(),
        3,
        "both continuation headers must also join the pre-existing ordered log"
    );
    for ((name, value), mutation) in ctx
        .request_headers_to_set
        .iter()
        .zip(ctx.pre_read_mutations.iter().skip(1))
    {
        let TrustedHeaderMutation::Set(ordered_name, ordered_value) = mutation else {
            panic!("continuation header must be an ordered Set");
        };
        assert_eq!(
            (name, value),
            (ordered_name, ordered_value),
            "legacy queue and ordered log must carry identical continuation headers in the same order"
        );
    }
    assert_eq!(
        ctx.request_headers_to_set[0].0,
        http::header::AUTHORIZATION,
        "the first queued continuation header must be authorization"
    );
    assert_eq!(
        ctx.request_headers_to_set[1].0,
        http::header::CONTENT_TYPE,
        "the second queued continuation header must be content-type"
    );
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
async fn preserves_default_parallel_tool_calls_true() {
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
    assert!(state.parallel_tool_calls, "the API default should be preserved");
    assert_eq!(
        state.request_body.get("parallel_tool_calls"),
        None,
        "an omitted field should stay omitted for byte-exact passthrough"
    );
    assert!(
        !state.request_body_requires_rebuild(),
        "preserving caller intent must not require proxy serialization"
    );
}

#[tokio::test]
async fn preserves_explicit_parallel_tool_calls_true() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let body = json!({"model": "gpt-4o", "input": "test", "parallel_tool_calls": true});
    ctx.extensions.insert(ResponsesState::from_request_body(body));

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.parallel_tool_calls);
    assert_eq!(state.request_body["parallel_tool_calls"], true);
    assert!(!state.request_body_requires_rebuild());
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
// on_request_body: Streaming
// -----------------------------------------------------------------------------

#[tokio::test]
async fn accepts_streaming_request() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({"model": "gpt-4o", "input": "test", "stream": true});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "stream:true should continue into IRR streaming"
    );
}

#[tokio::test]
async fn streaming_request_preserves_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let body = json!({"model": "gpt-4o", "input": "test", "stream": true});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "ResponsesState must remain in extensions for streaming response accumulation"
    );
}

#[test]
fn incomplete_stream_does_not_dispatch_accumulated_tool_call() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({
        "id": "resp_incomplete",
        "object": "response",
        "status": "incomplete",
        "output": [
        {"type": "function_call", "call_id": "call_partial", "name": "must_not_run", "arguments": "{}", "status": "completed"},
        {"type": "tool_search_call", "id": "tsc_partial", "status": "completed"},
        {
            "type": "message",
            "id": "msg_partial",
            "status": "incomplete",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "partial"}]
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a terminal incomplete stream must yield Continue so the committed SSE stream is forwarded"
    );
    assert_action(&ctx, "done");
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty(),
        "a tool from a truncated stream must not remain dispatchable"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .tool_search_calls
            .is_empty(),
        "a tool_search_call from a truncated stream must not remain dispatchable"
    );
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().accumulated_output[2]["id"],
        "msg_partial",
        "partial output from an incomplete response must remain client-visible"
    );
}

#[test]
fn failed_stream_does_not_dispatch_accumulated_tool_call() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({
        "id": "resp_failed",
        "object": "response",
        "status": "failed",
        "output": [
        {"type": "function_call", "call_id": "call_failed", "name": "must_not_run", "arguments": "{}", "status": "completed"},
        {
            "type": "message",
            "id": "msg_before_failure",
            "status": "incomplete",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "before failure"}]
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a terminal failed stream must yield Continue so the committed SSE stream is forwarded"
    );
    assert_action(&ctx, "done");
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty(),
        "a provider-failed response must not authorize tool execution"
    );
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().accumulated_output[1]["id"],
        "msg_before_failure",
        "partial output preceding a provider failure must remain client-visible"
    );
}

#[test]
fn streamed_web_search_call_is_available_to_dispatch_filter() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({
        "id": "resp_search",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Praxis"}
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a streamed web_search_call round must yield Continue while it loops for dispatch"
    );
    assert_action(&ctx, "loop");
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().web_search_calls.len(),
        1,
        "streamed web-search calls must be visible to openai_web_search"
    );
    assert!(
        !ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .messages
            .iter()
            .any(|m| m.get("type").and_then(Value::as_str) == Some("web_search_call")),
        "a hosted web_search_call is not a valid OpenResponses input item and must not \
         enter model-facing messages (issue #808)"
    );
}

#[test]
fn multiple_streamed_client_function_calls_end_done() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({"id": "resp_multiple", "object": "response", "status": "completed", "output": [
        {"type": "function_call", "call_id": "call_1", "name": "first", "status": "completed"},
        {"type": "function_call", "call_id": "call_2", "name": "second", "status": "completed"}
    ]});
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a streamed round of client calls completes like any other stream"
    );
    // Client `function_call`s resolve to no local dispatcher, so the owner ends
    // the loop and returns the calls to the client rather than looping to the
    // iteration cap (issue #1046 §3.4).
    assert_action(&ctx, "done");
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.len() == 2,
        "both client calls must remain visible for the client to execute"
    );
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), None);
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn mixed_streamed_function_call_ownership_fails_before_dispatch() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com", "require_approval":"never"}),
    );
    state.response_object = json!({"id": "resp_multiple", "object": "response", "status": "completed", "output": [
        {"type": "function_call", "call_id": "call_1", "name": "server__lookup", "status": "completed"},
        {"type": "function_call", "call_id": "call_2", "name": "client", "status": "completed"}
    ]});
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let mut body: Option<Bytes> = None;
    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "a mixed streamed batch must terminate through the logical SSE error path"
    );
    assert_action(&ctx, "done");
    assert!(
        body.is_none(),
        "a streamed SSE error must not serialize JSON onto the committed stream"
    );
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn streamed_mcp_ownership_index_rejects_near_retained_limit() {
    let make_state = || {
        let mut state = ResponsesState::from_request_body(json!({
            "model": "gpt-4o",
            "input": "test",
            "stream": true
        }));
        state.mcp_tool_map.insert(
            ("server".to_owned(), "lookup".to_owned()),
            json!({"server_label": "server", "require_approval": "never"}),
        );
        for index in 0..128 {
            state.mcp_tool_map.insert(
                (
                    format!("other_server_{index}"),
                    "long_tool_name_for_index_budget".to_owned(),
                ),
                json!({"require_approval": "never"}),
            );
        }
        let call = json!({
            "type": "function_call",
            "id": "fc_lookup",
            "call_id": "call_lookup",
            "name": "server__lookup",
            "arguments": "{}",
            "status": "completed"
        });
        // The canonical collector records the function-call selection after
        // moving the final response output.
        state.response_object = json!({
            "id": "resp_index_budget",
            "object": "response",
            "status": "completed",
            "output": [call]
        });
        state
    };

    // Measure the state after the streaming collector moves the output. The
    // reverse index is a separate allocation even though the state still fits.
    let mut projected = make_state();
    super::collect_streaming_output_items(&mut projected).unwrap();
    projected.apply_retained_payload_limit(usize::MAX);
    let retained = projected.retained_payload_bytes().unwrap();
    let index_charge = super::mcp_tool_index_charge(&projected).unwrap();
    let limit = retained + index_charge - 1;
    assert!(index_charge > 4_096);
    assert!(limit > retained);

    let mut state = make_state();
    state.apply_retained_payload_limit(limit);
    assert!(state.can_retain_payload(0), "the request itself fits before indexing");
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn streaming_iteration_limit_ends_with_sse_error() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 1").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.iteration = 1;
    // A dispatchable MCP call so the round would loop but for the iteration cap.
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label": "server", "server_url": "http://example.com", "require_approval": "never"}),
    );
    state.response_object = json!({"id": "resp_limit", "object": "response", "status": "completed", "output": [
        {"type": "function_call", "call_id": "call_limit", "name": "server__lookup", "status": "completed"},
        {"type": "tool_search_call", "id": "tsc_limit", "status": "completed"}
    ]});
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "an exhausted streamed loop must yield Continue after selecting the terminal SSE error"
    );
    assert_action(&ctx, "done");
    assert_eq!(
        ctx.get_metadata("responses.stream_error_code"),
        Some("server_error"),
        "an exhausted committed stream must end with an SSE error"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty(),
        "iteration-limit errors must not leave calls dispatchable"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .tool_search_calls
            .is_empty(),
        "iteration-limit errors must not leave tool_search_calls dispatchable"
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
fn buffered_terminal_rewrite_clears_stale_representation_headers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();
    for (name, value) in [
        (http::header::CONTENT_ENCODING, "gzip"),
        (http::header::CONTENT_LENGTH, "123"),
        (http::header::CONTENT_RANGE, "bytes 0-122/123"),
        (http::header::ETAG, "\"upstream\""),
        (http::header::LAST_MODIFIED, "Wed, 21 Oct 2015 07:28:00 GMT"),
    ] {
        response.headers.insert(name, value.parse().unwrap());
    }
    let mut ctx = make_filter_context(&req);
    ctx.response_header = Some(&mut response);
    ctx.extensions.insert(ResponsesState::default());
    let response_body = json!({
        "id":"resp_headers",
        "object":"response",
        "status":"completed",
        "output":[]
    });
    let mut body = Some(Bytes::from(response_body.to_string()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "a buffered terminal response should continue after finalization"
    );
    assert!(
        ctx.response_headers_modified,
        "rewriting the buffered representation must signal header mutation"
    );
    for name in [
        http::header::CONTENT_ENCODING,
        http::header::CONTENT_LENGTH,
        http::header::CONTENT_RANGE,
        http::header::ETAG,
        http::header::LAST_MODIFIED,
    ] {
        assert!(
            ctx.response_header
                .as_ref()
                .is_some_and(|response| !response.headers.contains_key(&name)),
            "rewritten response retained stale representation header {name}"
        );
    }
}

#[test]
fn buffered_parse_failure_preserves_body_and_representation_headers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut response = crate::test_utils::make_response();
    for (name, value) in [
        (http::header::CONTENT_ENCODING, "gzip"),
        (http::header::CONTENT_LENGTH, "5"),
        (http::header::CONTENT_RANGE, "bytes 0-4/5"),
        (http::header::ETAG, "\"upstream\""),
        (http::header::LAST_MODIFIED, "Wed, 21 Oct 2015 07:28:00 GMT"),
    ] {
        response.headers.insert(name, value.parse().unwrap());
    }
    let mut ctx = make_filter_context(&req);
    ctx.response_header = Some(&mut response);
    ctx.extensions.insert(ResponsesState::default());
    let original = Bytes::from_static(b"opaque upstream bytes");
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "an unrecognized buffered response should pass through"
    );
    assert_eq!(body, Some(original), "passthrough must preserve the upstream body");
    assert!(
        !ctx.response_headers_modified,
        "passthrough must not report a representation-header rewrite"
    );
    for name in [
        http::header::CONTENT_ENCODING,
        http::header::CONTENT_LENGTH,
        http::header::CONTENT_RANGE,
        http::header::ETAG,
        http::header::LAST_MODIFIED,
    ] {
        assert!(
            ctx.response_header
                .as_ref()
                .is_some_and(|response| response.headers.contains_key(&name)),
            "passthrough removed upstream representation header {name}"
        );
    }
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

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn tool_calls_set_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should be incremented");
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn any_dispatchable_call_type_sets_loop() {
    // An MCP `function_call`, a hosted `web_search_call`, and a hosted
    // `file_search_call` each resolve to a local dispatcher, so any one of them
    // must signal `loop`. Client (non-MCP) `function_call`s are covered by the
    // done-path tests: they resolve to no dispatcher and must not loop.
    let mcp = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);

    let mut web = make_state_with_tool_calls(vec![]);
    web.response_object = json!({
        "object": "response",
        "status": "completed",
        "output": [{"type": "web_search_call", "id": "ws_1", "status": "completed"}],
    });

    // A pending hosted file_search_call is recorded by the parse owner as a
    // `FileSearchAssignment` (issue #1046), so drive it through the response
    // object the owner drains rather than a removed state vector.
    let mut file = make_state_with_tool_calls(vec![]);
    file.response_object = json!({
        "object": "response",
        "status": "completed",
        "output": [{"type": "file_search_call", "id": "fs_1", "status": "searching"}],
    });

    for state in [mcp, web, file] {
        let filter = make_filter();
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);
        ctx.extensions.insert(state);

        drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
        assert_action(&ctx, "loop");
    }
}

#[test]
fn completed_file_search_call_sets_done() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    // A completed file_search_call is already terminal: the parse owner records
    // no `FileSearchAssignment` for it, so `has_dispatchable_calls` exits the loop
    // (issue #1046).
    let mut state = make_state_with_tool_calls(vec![]);
    state.response_object = json!({
        "id":"resp_1", "object":"response", "status":"completed",
        "output": [{"type": "file_search_call", "id": "fs_1", "status": "completed"}]
    });
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().iteration, 0);
}

// -----------------------------------------------------------------------------
// on_response_body: Config Defaults
// -----------------------------------------------------------------------------

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn default_config_has_max_infer_iters_ten() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);
    state.iteration = 9;
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 10, "iteration should have incremented to 10");
    state.response_object["output"] = json!([mcp_tool_call("call_2")]);
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

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn max_infer_iters_one_allows_exactly_one_loop() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 1").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "should have incremented to 1");

    state.response_object["output"] = json!([mcp_tool_call("call_2")]);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(r) if r.status == 508),
        "second round at iteration limit should produce 508 rejection"
    );
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn iteration_limit_returns_508_error() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 2").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);
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
fn multiple_client_function_calls_are_preserved() {
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
        matches!(&action, FilterAction::Continue),
        "multiple client function calls should be returned without rejection"
    );
    // Both calls are client-owned (no MCP tool map), so the loop ends and
    // returns them to the client rather than looping (issue #1046 §3.4).
    assert_action(&ctx, "done");
    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.len(), 2);
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn mixed_buffered_function_call_ownership_returns_upstream_error() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = make_state_with_tool_calls(vec![]);
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com", "require_approval":"never"}),
    );
    ctx.extensions.insert(state);
    let response = json!({
        "id":"resp_mixed",
        "object":"response",
        "status":"completed",
        "output":[
            {"type":"function_call", "call_id":"c1", "name":"server__lookup", "status":"completed"},
            {"type":"function_call", "call_id":"c2", "name":"client_lookup", "status":"completed"}
        ]
    });
    let mut body = Some(Bytes::from(response.to_string()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("mixed ownership must fail before server-side dispatch");
    };
    assert_eq!(
        rejection.status, 502,
        "the backend response, not the client request, is invalid"
    );
    assert!(ctx.extensions.get::<ResponsesState>().is_some());
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn mixed_buffered_custom_and_mcp_calls_return_upstream_error() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = make_state_with_tool_calls(vec![]);
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com", "require_approval":"never"}),
    );
    ctx.extensions.insert(state);
    let response = json!({
        "id":"resp_mixed_custom",
        "object":"response",
        "status":"completed",
        "output":[
            {"type":"function_call", "call_id":"c1", "name":"server__lookup", "status":"completed"},
            {"type":"custom_tool_call", "call_id":"c2", "name":"client_code", "input":"echo hello"}
        ]
    });
    let mut body = Some(Bytes::from(response.to_string()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("mixed custom/MCP ownership must fail before server-side dispatch");
    };
    assert_eq!(rejection.status, 502);
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn all_client_executed_call_types_conflict_with_mcp_dispatch() {
    for client_call in [
        json!({"type":"apply_patch_call", "call_id":"c2"}),
        json!({"type":"computer_call", "call_id":"c2"}),
        json!({"type":"local_shell_call", "call_id":"c2"}),
        json!({"type":"shell_call", "call_id":"c2", "environment":{"type":"local"}}),
        json!({"type":"tool_search_call", "call_id":"c2", "execution":"client"}),
    ] {
        let mut state = make_state_with_tool_calls(vec![json!({
            "type":"function_call", "call_id":"c1", "name":"server__lookup", "status":"completed"
        })]);
        state.mcp_tool_map.insert(
            ("server".to_owned(), "lookup".to_owned()),
            json!({"server_label":"server", "server_url":"http://example.com"}),
        );
        // The owner scans this round's accumulated output (the parse owner drains
        // the streamed round out of `response_object` before the check runs), so
        // the client-owned call must sit in `accumulated_output` from index
        // the explicit `current_round_output_start` onward.
        state.accumulated_output.push(client_call.clone());
        state.current_round_output_start = Some(0);

        assert!(
            super::has_mixed_function_call_ownership(&state),
            "client-owned call was not protected: {client_call}"
        );
    }
}

#[test]
fn hosted_container_shell_call_does_not_conflict_with_mcp_dispatch() {
    let mut state = make_state_with_tool_calls(vec![json!({
        "type":"function_call", "call_id":"c1", "name":"server__lookup", "status":"completed"
    })]);
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com"}),
    );
    // A container-referenced shell_call is hosted, not client-executed, so even
    // in this round's accumulated output it must not trip the mixed-ownership
    // guard.
    state.accumulated_output.push(json!({
        "type":"shell_call", "call_id":"c2",
        "environment":{"type":"container_reference", "container_id":"cntr_1"}
    }));
    state.current_round_output_start = Some(0);

    assert!(
        !super::has_mixed_function_call_ownership(&state),
        "a hosted container shell call must remain server-owned"
    );
}

#[test]
fn hosted_tool_search_conflicts_with_client_function_call() {
    let mut state = make_state_with_tool_calls(vec![json!({
        "type":"function_call", "call_id":"c1", "name":"get_weather", "status":"completed"
    })]);
    state.select_test_output(
        "tool_search_call",
        vec![json!({
            "type":"tool_search_call", "id":"tsc_1", "status":"completed"
        })],
    );

    assert!(
        super::has_mixed_function_call_ownership(&state),
        "hosted tool_search_call mixed with a client function_call must be rejected"
    );
}

#[test]
fn hosted_tool_search_alone_is_not_mixed_ownership() {
    let mut state = make_state_with_tool_calls(vec![]);
    state.select_test_output(
        "tool_search_call",
        vec![json!({
            "type":"tool_search_call", "id":"tsc_1", "status":"completed"
        })],
    );

    assert!(
        !super::has_mixed_function_call_ownership(&state),
        "a hosted tool_search_call by itself must still loop for deferred discovery"
    );
}

#[test]
fn mixed_ownership_incomplete_response_is_preserved() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = make_state_with_tool_calls(vec![]);
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com", "require_approval":"never"}),
    );
    ctx.extensions.insert(state);
    let response = json!({
        "id":"resp_incomplete",
        "object":"response",
        "status":"incomplete",
        "incomplete_details":{"reason":"max_output_tokens"},
        "output":[
            {"type":"function_call", "call_id":"c1", "name":"server__lookup", "status":"completed"},
            {"type":"function_call", "call_id":"c2", "name":"client_lookup", "status":"completed"}
        ]
    });
    let mut body = Some(Bytes::from(response.to_string()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
    assert_eq!(ctx.get_metadata("responses.status"), Some("incomplete"));
    let returned: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(returned["status"], "incomplete");
    assert_eq!(returned["incomplete_details"]["reason"], "max_output_tokens");
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

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn iteration_incremented_on_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_mcp_tool_calls(vec![mcp_tool_call("call_1")]);
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
        .get("openai_agentic_loop")
        .expect("IRR consumers require openai_agentic_loop entry");
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
    // The call is extracted regardless of ownership; being client-owned it ends
    // the loop as `done` rather than looping (issue #1046 §3.4).
    assert_action(&ctx, "done");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_calls.len(), 1);
    assert_eq!(state.tool_calls[0].item_id, "fc_1");
    assert_eq!(
        serde_json::from_slice::<Value>(body.as_ref().unwrap()).unwrap()["output"][0]["call_id"],
        "call_1"
    );
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
fn appends_provider_compaction_to_replay_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    ctx.extensions.insert(make_state_with_tool_calls(vec![]));
    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [{
            "type": "compaction",
            "id": "cmp_provider",
            "encrypted_content": "provider-state"
        }]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages[1]["type"], "compaction");
    assert_eq!(state.persisted_messages[1]["id"], "cmp_provider");
    assert!(
        state.provider_compaction_ids.contains("cmp_provider"),
        "provider compaction IDs must include replayable response output"
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

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn accumulates_usage_across_rounds() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    // A dispatchable MCP call so round 1 loops and usage carries into round 2.
    let state = make_state_with_mcp_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let round1 = json!({
        "id": "resp_1",
        "object": "response",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "server__lookup",
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
        .find(|f| f["filter"].as_str() == Some("openai_agentic_loop"))
        .expect("inference step should have openai_agentic_loop filter");
    let filter = super::AgenticLoopFilter::from_config(al_config).unwrap();
    assert_eq!(filter.name(), "openai_agentic_loop");
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
    assert_eq!(state.selected_web_search_calls()[0]["id"], "ws_1");
}

#[test]
fn web_search_call_triggers_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.response_object = json!({"object": "response", "status": "completed", "output": [{
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    }]});
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
    state.response_object = json!({"object": "response", "status": "completed", "output": [{
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    }]});
    ctx.extensions.insert(state);

    drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should increment from 0 to 1");
}

#[test]
fn mixed_client_function_and_web_search_calls_fail_before_dispatch() {
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

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502),
        "mixed client/server ownership must fail before web-search side effects"
    );
}

#[test]
fn mixed_client_function_and_tool_search_calls_fail_before_dispatch() {
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
                "type": "tool_search_call",
                "id": "tsc_1",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502),
        "mixed client function_call and hosted tool_search_call must fail before deferred discovery"
    );
}

#[test]
fn mixed_client_function_and_pending_file_search_fail_before_dispatch() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(make_state_with_tool_calls(vec![]));
    let response_body = json!({
        "id":"resp_1", "object":"response", "status":"completed",
        "output":[
            {
                "type":"function_call", "id":"fc_1", "call_id":"call_1",
                "name":"get_weather", "arguments":"{}", "status":"completed"
            },
            {"type":"file_search_call", "id":"fs_1", "status":"searching"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502),
        "mixed client/file-search ownership must fail before local side effects"
    );
}

#[test]
fn web_search_call_subject_to_iteration_limit() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 2").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.response_object = json!({"object": "response", "status": "completed", "output": [{
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    }]});
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
    state.select_test_output(
        "web_search_call",
        vec![json!({
            "type": "web_search_call",
            "id": "ws_stale",
            "action": {"type": "search", "query": "old query"}
        })],
    );
    ctx.extensions.insert(state);

    drop(filter.on_request_body(&mut ctx, &mut None, true).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.web_search_calls.is_empty(),
        "on_request_body must clear stale web_search_calls from previous round"
    );
}

#[test]
fn web_search_call_excluded_from_messages_but_persisted() {
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
        !msg_types.contains(&"web_search_call"),
        "web_search_call must NOT be forwarded to the backend via messages: {msg_types:?}"
    );

    assert!(
        state
            .selected_web_search_calls()
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("ws_1")),
        "web_search_call should be queued in web_search_calls for dispatch"
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

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn tool_search_call_queued_for_deferred_discovery() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.deferred_mcp = vec![pending_deferred_connector()];
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "tool_search_call",
                "id": "tsc_1",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "tool_search_call should continue so dispatch can loop"
    );
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_search_calls.len(), 1);
    assert!(
        state
            .messages
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("tool_search_call")),
        "tool_search_call should not enter backend messages"
    );
    assert!(
        state
            .persisted_messages
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("tool_search_call")),
        "tool_search_call should be persisted"
    );
}

#[test]
fn client_executed_tool_search_call_is_returned_without_deferred_discovery() {
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
                "type": "tool_search_call",
                "id": "tsc_client",
                "status": "completed",
                "execution": "client"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a client-owned search must not reject the round"
    );
    assert_action(&ctx, "done");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_search_calls.is_empty(),
        "client-executed tool_search_call must not queue server-side tools/list"
    );
    assert!(
        state
            .messages
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("tool_search_call")),
        "tool_search_call should not enter backend messages"
    );
    assert!(
        state
            .persisted_messages
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_client")),
        "client-executed tool_search_call should still be persisted"
    );
    let returned: Value = serde_json::from_slice(body.as_ref().expect("finalized body")).unwrap();
    assert!(
        returned["output"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_client")),
        "client-executed tool_search_call remains caller-visible"
    );
}

#[test]
fn hosted_tool_search_does_not_loop_when_max_tool_calls_is_exhausted() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![]);
    state.max_tool_calls = Some(0);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_1",
        "object": "response",
        "output": [
            {
                "type": "tool_search_call",
                "id": "tsc_over_budget",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_search_calls.is_empty(),
        "an over-budget hosted search must not remain dispatchable"
    );
    let returned: Value = serde_json::from_slice(body.as_ref().expect("finalized body")).unwrap();
    assert_eq!(
        returned["output"][0]["id"], "tsc_over_budget",
        "an over-budget hosted search is still returned to the caller"
    );
    assert_eq!(
        returned["output"][0]["status"], "incomplete",
        "over-budget hosted searches must not remain completed"
    );
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn hosted_tool_search_cap_keeps_only_first_of_two_calls() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = make_state_with_tool_calls(vec![]);
    state.max_tool_calls = Some(1);
    state.deferred_mcp.push(pending_deferred_connector());
    ctx.extensions.insert(state);
    let response = json!({
        "id": "resp_two_searches",
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "tool_search_call", "id": "ts_first", "status": "completed"},
            {"type": "tool_search_call", "id": "ts_second", "status": "completed"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "loop");
    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.tool_search_calls.len(), 1);
    assert_eq!(state.selected_tool_search_calls()[0]["id"], "ts_first");
    assert_eq!(state.accumulated_output[0]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
    let stored_calls: Vec<_> = state
        .persisted_messages
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("tool_search_call"))
        .collect();
    assert_eq!(stored_calls.len(), 2);
    assert_eq!(stored_calls[0]["status"], "completed");
    assert_eq!(stored_calls[1]["status"], "incomplete");
    let mut public_body = None;
    state.finalize_response_body(&mut public_body).unwrap();
    let public: Value = serde_json::from_slice(public_body.as_ref().unwrap()).unwrap();
    assert_eq!(public["output"][0]["status"], "completed");
    assert_eq!(public["output"][1]["status"], "incomplete");
}

#[test]
fn incomplete_tool_search_call_is_not_queued_for_deferred_discovery() {
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
                "type": "tool_search_call",
                "id": "tsc_in_progress",
                "status": "in_progress"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "an in-progress search must not reject the round"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_search_calls.is_empty(),
        "only completed tool_search_call items may trigger tools/list"
    );
    let returned: Value = serde_json::from_slice(body.as_ref().expect("finalized body")).unwrap();
    assert!(
        returned["output"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_in_progress")),
        "the in-progress item remains client-visible"
    );
}

#[cfg(feature = "openai-mcp-tools")]
#[test]
fn streamed_tool_search_call_is_queued_for_deferred_discovery() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.deferred_mcp = vec![pending_deferred_connector()];
    state.response_object = json!({
        "id": "resp_search",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "tool_search_call",
            "id": "tsc_1",
            "status": "completed"
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a streamed tool_search_call round must yield Continue while it loops for discovery"
    );
    assert_action(&ctx, "loop");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.tool_search_calls.len(),
        1,
        "streamed tool_search_call must be visible to openai_mcp_dispatch"
    );
    assert!(
        !state
            .messages
            .iter()
            .any(|m| m.get("type").and_then(Value::as_str) == Some("tool_search_call")),
        "a hosted tool_search_call is not a valid OpenResponses input item and must not \
         enter model-facing messages"
    );
}

#[test]
fn streamed_client_executed_tool_search_call_is_returned_without_deferred_discovery() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({
        "id": "resp_search",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "tool_search_call",
            "id": "tsc_client",
            "status": "completed",
            "execution": "client"
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_search_calls.is_empty(),
        "a streamed client-executed search must not queue tools/list"
    );
    assert!(
        state
            .persisted_messages
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_client")),
        "the client-executed search must still be stored"
    );
    assert!(
        state
            .accumulated_output
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_client")),
        "the client-executed search remains caller-visible"
    );
    assert!(
        !state
            .messages
            .iter()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("tool_search_call")),
        "tool_search_call should not enter backend messages"
    );
}

#[test]
fn streamed_incomplete_tool_search_call_is_not_queued() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.response_object = json!({
        "id": "resp_search",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "tool_search_call",
            "id": "tsc_incomplete",
            "status": "incomplete"
        }]
    });
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .tool_search_calls
            .is_empty(),
        "incomplete streamed searches must not trigger tools/list"
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

#[test]
fn status_less_function_call_is_collected() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_status_less",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let results = ctx.filter_results.get("openai_agentic_loop").unwrap();
    let action = results.get("action").unwrap();

    assert_eq!(
        state.tool_calls.len(),
        1,
        "Status-less function call should be extracted to tool_calls"
    );
    assert_eq!(
        action, "done",
        "an unresolved client-owned function call must be returned without a server loop"
    );
}

#[test]
fn status_null_function_call_is_collected() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_status_null",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "status": null
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let results = ctx.filter_results.get("openai_agentic_loop").unwrap();
    let action = results.get("action").unwrap();

    assert_eq!(
        state.tool_calls.len(),
        1,
        "Function call with null status should be extracted to tool_calls"
    );
    assert_eq!(
        action, "done",
        "an unresolved client-owned function call must be returned without a server loop"
    );
}

#[test]
fn malformed_function_call_status_is_ignored() {
    let malformed_statuses = vec![
        json!(123),
        json!({}),
        json!(["completed"]),
        json!("failed"),
        json!("in_progress"),
    ];

    for malformed_status in malformed_statuses {
        let filter = make_filter();
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);

        let state = make_state_with_tool_calls(vec![]);
        ctx.extensions.insert(state);

        let response_body = json!({
            "id": "resp_malformed_status",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                    "status": malformed_status
                }
            ]
        });
        let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

        drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

        let state = ctx.extensions.get::<ResponsesState>().unwrap();
        assert!(
            state.tool_calls.is_empty(),
            "Function call with malformed status `{malformed_status}` should be ignored"
        );
    }
}

// -----------------------------------------------------------------------------
// on_response_body: Cross-Dispatcher Ordering (#1046)
// -----------------------------------------------------------------------------

/// Boundary test 4: one model round emitting a `web_search_call`, a private
/// `function_call(name=file_search)`, and an MCP `function_call`, in that order,
/// is parsed by the sole owner so that `accumulated_output` preserves the model's
/// ordering while each dispatcher's target is recorded against the correct
/// absolute index. Proves model output ordering survives all three dispatchers.
#[cfg(feature = "openai-mcp-tools")]
#[test]
fn model_output_ordering_survives_all_three_dispatchers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    // A configured hosted file-search tool so the owner normalizes the private
    // `function_call(name=file_search)` into a canonical `file_search_call`.
    let mut state = make_state_with_mcp_tool_calls(vec![]);
    state.tools = vec![json!({"type": "file_search", "vector_store_ids": ["vs_1"]})];
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_ordering",
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {
                "type": "function_call",
                "id": "fc_fs",
                "call_id": "call_fs",
                "name": "file_search",
                "arguments": "{\"query\": \"quarterly revenue\"}",
                "status": "completed"
            },
            {
                "type": "function_call",
                "id": "fc_mcp",
                "call_id": "call_mcp",
                "name": "server__lookup",
                "arguments": "{}",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a round with dispatchable calls continues into request re-entry"
    );
    // Every dispatcher has work, so the sole owner signals one continuation.
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    // Accumulator preserves the model's emission order verbatim.
    assert_eq!(state.accumulated_output.len(), 3);
    assert_eq!(state.accumulated_output[0]["type"], "web_search_call");
    assert_eq!(state.accumulated_output[0]["id"], "ws_1");
    // The private function_call was normalized in place, keeping its position.
    assert_eq!(state.accumulated_output[1]["type"], "file_search_call");
    assert_eq!(state.accumulated_output[1]["id"], "fc_fs");
    assert_eq!(state.accumulated_output[1]["status"], "searching");
    assert_eq!(state.accumulated_output[1]["queries"][0], "quarterly revenue");
    assert_eq!(state.accumulated_output[2]["type"], "function_call");
    assert_eq!(state.accumulated_output[2]["id"], "fc_mcp");

    // Each dispatcher's target is recorded against its absolute index.
    assert_eq!(state.web_search_calls.len(), 1, "web_search dispatcher target recorded");
    assert_eq!(state.selected_web_search_calls()[0]["id"], "ws_1");
    assert_eq!(
        state.file_search_assignments.len(),
        1,
        "file_search dispatcher target recorded"
    );
    assert_eq!(
        state.file_search_assignments[0].output_index, 1,
        "file_search assignment points at its absolute accumulator index"
    );
    assert_eq!(
        state.file_search_assignments[0].synthesis,
        SynthesisKind::Private,
        "a normalized private call records the private synthesis origin"
    );
    assert_eq!(state.tool_calls.len(), 1, "MCP dispatcher target recorded");
    assert_eq!(state.selected_tool_calls()[0]["name"], "server__lookup");
}

#[test]
fn streamed_provider_conversation_marks_persisted_history() {
    let mut state = ResponsesState {
        conversation: Some(json!({"id": "conv_native"})),
        messages: vec![json!({"role": "user", "content": "weather in SF"})],
        response_object: json!({
            "output": [{
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_123",
                "name": "weather__get_weather",
                "arguments": "{}",
                "status": "completed"
            }]
        }),
        ..ResponsesState::default()
    };

    super::collect_streaming_output_items(&mut state).unwrap();

    assert_eq!(
        state.provider_history_len,
        state.messages.len(),
        "provider history must include the streamed function call"
    );
    assert_eq!(
        state.provider_history_len, 2,
        "prompt and function call should be persisted"
    );
}

#[test]
fn appends_streamed_provider_compaction_to_replay_state() {
    let compaction = json!({
        "type": "compaction",
        "id": "cmp_streamed",
        "encrypted_content": "provider-state"
    });
    let input = json!({"type": "message", "role": "user", "content": "continue"});
    let mut state = ResponsesState {
        messages: vec![input.clone()],
        persisted_messages: vec![input],
        response_object: json!({"output": [compaction]}),
        ..ResponsesState::default()
    };

    super::collect_streaming_output_items(&mut state).unwrap();

    assert_eq!(state.messages[1]["type"], "compaction");
    assert_eq!(state.persisted_messages[1]["id"], "cmp_streamed");
    assert!(
        state.provider_compaction_ids.contains("cmp_streamed"),
        "streamed provider compaction IDs must be retained for replay"
    );
    assert_eq!(state.accumulated_output[0]["id"], "cmp_streamed");
}

/// Regression (#955): the sole owner stamps a stable synthetic id on every
/// id-less output item before accumulation, so the public response never ships an
/// item without an id. A private `function_call(name=file_search)` that arrives
/// without an id keeps a durable identity derived from its `call_id`.
#[test]
fn owner_assigns_ids_to_id_less_output_items() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    // A configured hosted file-search tool so the owner normalizes the private
    // `function_call(name=file_search)` into a canonical `file_search_call`.
    let mut state = make_state_with_mcp_tool_calls(vec![]);
    state.tools = vec![json!({"type": "file_search", "vector_store_ids": ["vs_1"]})];
    ctx.extensions.insert(state);

    let response_body = json!({
        "id": "resp_idless",
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}]
            },
            {
                "type": "function_call",
                "call_id": "call_fs",
                "name": "file_search",
                "arguments": "{\"query\": \"revenue\"}",
                "status": "completed"
            }
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));

    let _action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    assert_eq!(state.accumulated_output.len(), 3);
    assert!(
        state.accumulated_output[0]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("rs_")),
        "id-less reasoning item receives a synthetic rs_ id, got: {:?}",
        state.accumulated_output[0]["id"]
    );
    assert!(
        state.accumulated_output[1]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_")),
        "id-less message item receives a synthetic msg_ id, got: {:?}",
        state.accumulated_output[1]["id"]
    );
    // The normalized file-search call keeps a durable id derived from its call_id.
    assert_eq!(
        state.accumulated_output[2]["type"], "file_search_call",
        "the private function_call was normalized in place"
    );
    assert_eq!(
        state.accumulated_output[2]["id"], "fs_call_fs",
        "a normalized id-less file-search call keeps an id derived from its call_id"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Dispatch Failure Conversion (#1046)
// -----------------------------------------------------------------------------

/// Boundary test 7 (buffered): a request-phase dispatcher's recorded
/// [`DispatchFailure`] is converted by the sole owner, before the next inference
/// request, into a buffered JSON rejection carrying the dispatcher's status/code
/// and clearing this round's dispatch bookkeeping. A `Reject` here short-circuits
/// the IRR before any further backend call.
#[tokio::test]
async fn dispatch_failure_buffered_rejects_with_json_envelope() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({"model": "gpt-4o", "input": "test"}));
    // Stale per-round bookkeeping the conversion must drop.
    state.select_test_output(
        "function_call",
        vec![json!({"type": "function_call", "name": "server__lookup", "status": "completed"})],
    );
    state.select_test_output(
        "web_search_call",
        vec![json!({"type": "web_search_call", "id": "ws_1"})],
    );
    state.file_search_assignments = vec![FileSearchAssignment {
        output_index: 0,
        item_id: "fs_stale".to_owned(),
        synthesis: SynthesisKind::Native,
    }];
    state.dispatch_failure = Some(DispatchFailure {
        status: 502,
        code: "server_error",
        message: "vector store search failed".to_owned(),
    });
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("a recorded dispatch failure must reject before the next inference call");
    };
    assert_eq!(response.status, 502);
    let body: Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "server_error");
    assert_eq!(body["error"]["message"], "vector store search failed");
    assert_eq!(ctx.filter_results["openai_agentic_loop"].get("action"), Some("done"));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.tool_calls.is_empty(), "dispatch failure clears stale tool_calls");
    assert!(
        state.web_search_calls.is_empty(),
        "dispatch failure clears stale web_search_calls"
    );
    assert!(
        state.file_search_assignments.is_empty(),
        "dispatch failure clears stale file_search assignments"
    );
}

/// Boundary test 7 (streaming): once `text/event-stream` is committed, the owner
/// converts a recorded [`DispatchFailure`] into a terminal SSE `error` frame on
/// the live logical stream (never a JSON envelope) and suppresses persistence of
/// the failed round. The `Reject` still short-circuits before the next inference
/// call.
#[tokio::test]
async fn dispatch_failure_streaming_emits_sse_error_frame() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.dispatch_failure = Some(DispatchFailure {
        status: 502,
        code: "server_error",
        message: "vector store search failed".to_owned(),
    });
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("a committed stream must terminate through an SSE error reject");
    };
    assert_eq!(response.status, 200, "a committed stream keeps its 200 status");
    let content_type = response.headers.iter().find(|(k, _)| k == "content-type");
    assert_eq!(content_type.map(|(_, v)| v.as_str()), Some("text/event-stream"));
    let body = String::from_utf8(response.body.as_ref().unwrap().to_vec()).unwrap();
    assert!(body.contains("event: error"), "terminal frame is an SSE error: {body}");
    assert!(
        body.contains("vector store search failed"),
        "SSE error carries the dispatcher message: {body}"
    );
    assert!(
        !body.contains("[DONE]"),
        "an SSE error frame terminates the stream without a [DONE] sentinel: {body}"
    );
    assert_eq!(ctx.filter_results["openai_agentic_loop"].get("action"), Some("done"));
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "the failed streamed round must not be persisted"
    );
}

/// A locally-detected security-context failure preempts a generic dispatch failure:
/// the loop owner converts `security_failure` BEFORE `dispatch_failure`, so the client
/// sees the 401 security terminal, never the 502 dispatch terminal.
#[tokio::test]
async fn security_failure_preempts_dispatch_failure() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({"model": "gpt-4o", "input": "test"}));
    state.record_security_failure(DispatchFailure {
        status: 401,
        code: "missing_callout_context",
        message: "no creds".to_owned(),
    });
    state.dispatch_failure = Some(DispatchFailure {
        status: 502,
        code: "server_error",
        message: "bad gateway".to_owned(),
    });
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    let FilterAction::Reject(response) = action else {
        panic!("a recorded security failure must reject before the next inference call");
    };
    assert_eq!(
        response.status, 401,
        "security terminal preempts the 502 dispatch terminal"
    );
    let body: Value = serde_json::from_slice(response.body.as_ref().unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "missing_callout_context");
    assert_eq!(body["error"]["message"], "no creds");
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
    state.select_test_output("function_call", tool_calls);
    state
}

#[cfg(feature = "openai-mcp-tools")]
fn pending_deferred_connector() -> DeferredMcpConnector {
    DeferredMcpConnector {
        authorization: None,
        allowed_tools: None,
        connector_id: "corp_drive".to_owned(),
        headers: None,
        max_rewritten_body_bytes: 67_108_864,
        max_tools: 128,
        require_approval: None,
        server_label: "drive".to_owned(),
        server_url: "https://drive.example.com/mcp".to_owned(),
        timeout: std::time::Duration::from_secs(5),
    }
}

/// A completed `function_call` whose encoded name (`server__lookup`) resolves to
/// the MCP tool registered by [`make_state_with_mcp_tool_calls`], so the owner
/// classifies it as a dispatchable server-owned call and signals `loop`.
///
/// Tests that drive the loop through this call are gated on
/// `openai-mcp-tools`, the feature that gives the owner an MCP dispatcher;
/// without it the same call resolves to no dispatcher and the owner
/// correctly signals `done` (the FIPS build ships that way).
#[cfg(feature = "openai-mcp-tools")]
fn mcp_tool_call(call_id: &str) -> Value {
    json!({
        "type": "function_call",
        "call_id": call_id,
        "name": "server__lookup",
        "arguments": "{}",
        "status": "completed",
    })
}

/// State carrying dispatchable MCP tool calls: the given calls plus a one-entry
/// `mcp_tool_map` so `server__lookup` resolves under an auto-approval policy.
///
/// Client (non-MCP) `function_call`s resolve to no dispatcher and signal `done`;
/// tests that need the loop to continue must drive it with a dispatchable call.
fn make_state_with_mcp_tool_calls(tool_calls: Vec<Value>) -> ResponsesState {
    let mut state = make_state_with_tool_calls(vec![]);
    state.response_object = json!({
        "object": "response",
        "status": "completed",
    });
    state.response_object["output"] = Value::Array(tool_calls);
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label": "server", "server_url": "http://example.com", "require_approval": "never"}),
    );
    state
}

fn assert_action(ctx: &praxis_filter::HttpFilterContext<'_>, expected: &str) {
    let results = ctx
        .filter_results
        .get("openai_agentic_loop")
        .expect("filter_results should contain openai_agentic_loop entry");
    let action = results.get("action").expect("should have action key");
    assert_eq!(action, expected, "openai_agentic_loop action mismatch");
}

#[tokio::test]
async fn on_request_rejects_oversized_initial_state_with_413() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_retained_bytes: 4096").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "x".repeat(2_000)
    }));
    state.store_persist_armed = true;
    ctx.extensions.insert(state);

    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("oversized starting state must be rejected before inference");
    };
    assert_eq!(rejection.status, 413);
    assert_action(&ctx, "done");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.retained_payload_failed);
    assert!(!state.store_persist_armed);
    assert!(state.input.is_empty());
    assert!(state.messages.is_empty());
    assert!(state.persisted_messages.is_empty());
    assert!(state.response_object.is_null());
}

#[tokio::test]
#[cfg(feature = "store")]
async fn on_request_charges_response_store_input_snapshot() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_retained_bytes: 4096").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_metadata("responses.store_request_payload_bytes", "4096");
    ctx.extensions.insert(ResponsesState::default());

    let action = filter.on_request(&mut ctx).await.unwrap();

    let FilterAction::Reject(rejection) = action else {
        panic!("the independently retained store input must be admitted before inference");
    };
    assert_eq!(rejection.status, 413);
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.retained_external_payload_bytes, 0);
    assert_eq!(
        super::super::store::retained_request_payload_bytes(&ctx),
        Some(0),
        "the rejected store snapshot must be released"
    );
}

#[tokio::test]
async fn smallest_loop_retained_limit_wins_for_one_request() {
    let lower: serde_yaml::Value = serde_yaml::from_str("max_retained_bytes: 8192").unwrap();
    let higher: serde_yaml::Value = serde_yaml::from_str("max_retained_bytes: 16384").unwrap();
    let lower = super::AgenticLoopFilter::from_config(&lower).unwrap();
    let higher = super::AgenticLoopFilter::from_config(&higher).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::default());

    assert!(matches!(
        lower.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(matches!(
        higher.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_limit(),
        Some(8192)
    );
}

#[test]
fn only_one_loop_instance_claims_each_router_response() {
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    assert!(super::claim_response_round(&mut ctx, Some(0)));
    assert!(!super::claim_response_round(&mut ctx, Some(0)));
    assert!(super::claim_response_round(&mut ctx, Some(1)));
    assert!(!super::claim_response_round(&mut ctx, Some(1)));
}

#[test]
fn cumulative_buffered_round_overflow_is_transactional() {
    let mut state = ResponsesState::default();
    let response = |id: &str, fill: char| {
        Bytes::from(
            serde_json::to_vec(&json!({
                "id": id,
                "object": "response",
                "status": "completed",
                "output": [{
                    "id": format!("rs_{id}"),
                    "type": "reasoning",
                    "summary": fill.to_string().repeat(600)
                }]
            }))
            .unwrap(),
        )
    };

    super::extract_tool_calls_from_body(&response("one", 'a'), &mut state).unwrap();
    assert_eq!(state.accumulated_output.len(), 1);
    let retained_after_first = state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(retained_after_first + response("two", 'b').len() - 1);

    let failure = super::extract_tool_calls_from_body(&response("two", 'b'), &mut state).unwrap_err();
    assert_eq!(failure.status, 502);
    assert_eq!(state.accumulated_output.len(), 1, "second round committed nothing");
    assert_eq!(state.response_object["id"], "one");
    assert_eq!(state.retained_payload_bytes().unwrap(), retained_after_first);
}

fn large_completed_tool_search_cases() -> [(Value, usize); 2] {
    [
        (
            json!({
                "type": "tool_search_call",
                "id": "tsc_hosted",
                "status": "completed",
                "results": "x".repeat(2_048)
            }),
            3,
        ),
        (
            json!({
                "type": "tool_search_call",
                "id": "tsc_client",
                "status": "completed",
                "execution": "client",
                "results": "x".repeat(2_048)
            }),
            2,
        ),
    ]
}

#[test]
fn buffered_tool_search_preflight_charges_every_retained_copy() {
    for (item, copies) in large_completed_tool_search_cases() {
        let response = json!({"object": "response", "output": [item]});
        let body = Bytes::from(serde_json::to_vec(&response).unwrap());
        let baseline = ResponsesState::default().retained_payload_bytes().unwrap();
        let response_bytes = super::super::state::retained_json_bytes(&response).unwrap();
        let item_bytes = super::super::state::retained_json_bytes(&response["output"][0]).unwrap();
        let selection_id_bytes = if copies == 3 {
            response["output"][0]["id"].as_str().unwrap().len()
        } else {
            0
        };
        let exact_peak = baseline + response_bytes + item_bytes + selection_id_bytes;

        let mut rejected = ResponsesState::default();
        rejected.apply_retained_payload_limit(exact_peak - 1);
        let failure = super::extract_tool_calls_from_body(&body, &mut rejected).unwrap_err();
        assert_eq!(failure.status, 502);
        assert!(
            rejected.response_object.is_null(),
            "preflight must leave the prior response intact"
        );
        assert!(
            rejected.accumulated_output.is_empty(),
            "preflight must not commit caller output"
        );
        assert!(
            rejected.tool_search_calls.is_empty(),
            "preflight must not queue discovery"
        );
        assert!(
            rejected.persisted_messages.is_empty(),
            "preflight must not persist the item"
        );

        let mut admitted = ResponsesState::default();
        admitted.apply_retained_payload_limit(exact_peak);
        super::extract_tool_calls_from_body(&body, &mut admitted).unwrap();
        assert!(admitted.retained_payload_bytes().unwrap() <= exact_peak);
        assert_eq!(admitted.accumulated_output.len(), 1);
        assert_eq!(admitted.tool_search_calls.len(), usize::from(copies == 3));
        assert_eq!(admitted.persisted_messages.len(), 1);
    }
}

#[test]
fn streamed_tool_search_preflight_charges_every_retained_copy() {
    for (item, copies) in large_completed_tool_search_cases() {
        let response = json!({"object": "response", "output": [item]});
        let response_bytes = super::super::state::retained_json_bytes(&response).unwrap();
        let output_bytes = super::super::state::retained_json_bytes(&response["output"]).unwrap();
        let item_bytes = super::super::state::retained_json_bytes(&response["output"][0]).unwrap();
        let mut rejected = ResponsesState {
            response_object: response.clone(),
            ..ResponsesState::default()
        };
        let baseline_without_response = rejected.retained_payload_bytes().unwrap() - response_bytes;
        let selection_id_bytes = if copies == 3 {
            response["output"][0]["id"].as_str().unwrap().len()
        } else {
            0
        };
        let exact_retained =
            baseline_without_response + response_bytes - output_bytes + 2 + 2 * item_bytes + selection_id_bytes;
        rejected.apply_retained_payload_limit(exact_retained - 1);
        assert!(!super::streaming_collection_retention_fits(&rejected));
        let failure = super::collect_streaming_output_items(&mut rejected).unwrap_err();
        assert_eq!(failure.status, 502);
        assert!(rejected.retained_payload_failed);
        assert!(rejected.accumulated_output.is_empty());
        assert!(rejected.tool_search_calls.is_empty());
        assert!(rejected.persisted_messages.is_empty());

        let mut admitted = ResponsesState {
            response_object: response,
            ..ResponsesState::default()
        };
        admitted.apply_retained_payload_limit(exact_retained);
        super::collect_streaming_output_items(&mut admitted).unwrap();
        assert_eq!(admitted.retained_payload_bytes().unwrap(), exact_retained);
        assert_eq!(admitted.accumulated_output.len(), 1);
        assert_eq!(admitted.tool_search_calls.len(), usize::from(copies == 3));
        assert_eq!(admitted.persisted_messages.len(), 1);
    }
}

#[test]
fn compaction_collection_preflights_history_copies_and_provenance_id() {
    let id = format!("cmp_{}", "x".repeat(4_096));
    let response = json!({
        "object": "response",
        "output": [{"type": "compaction", "id": id, "encrypted_content": "opaque"}]
    });
    let response_bytes = super::super::state::retained_json_bytes(&response).unwrap();
    let output_bytes = super::super::state::retained_json_bytes(&response["output"]).unwrap();
    let item_bytes = super::super::state::retained_json_bytes(&response["output"][0]).unwrap();
    let id_bytes = response["output"][0]["id"].as_str().unwrap().len();
    let baseline = ResponsesState::default().retained_payload_bytes().unwrap();

    let body = Bytes::from(serde_json::to_vec(&response).unwrap());
    let buffered_peak = baseline + response_bytes + 2 * item_bytes + id_bytes;
    let mut buffered = ResponsesState::default();
    buffered.apply_retained_payload_limit(buffered_peak - 1);
    assert!(super::extract_tool_calls_from_body(&body, &mut buffered).is_err());
    assert!(buffered.accumulated_output.is_empty());
    assert!(buffered.provider_compaction_ids.is_empty());

    let mut streamed = ResponsesState {
        response_object: response,
        ..ResponsesState::default()
    };
    let streamed_retained = baseline - 4 + response_bytes - output_bytes + 2 + 3 * item_bytes + id_bytes;
    streamed.apply_retained_payload_limit(streamed_retained - 1);
    assert!(!super::streaming_collection_retention_fits(&streamed));
    assert!(streamed.provider_compaction_ids.is_empty());
    assert_eq!(streamed.response_object["output"].as_array().unwrap().len(), 1);
}

#[test]
fn buffered_parse_peak_accepts_exact_budget_without_charging_framework_body() {
    let response = json!({
        "id": "resp_exact",
        "object": "response",
        "status": "completed",
        "output": [{
            "id": "msg_exact",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "retained"}]
        }]
    });
    let body = Bytes::from(serde_json::to_vec(&response).unwrap());
    let baseline = ResponsesState::default().retained_payload_bytes().unwrap();
    let response_bytes = super::super::state::retained_json_bytes(&response).unwrap();
    let usage_projection = super::super::state::retained_json_bytes(&Value::Null).unwrap();
    let exact_peak = baseline + response_bytes + usage_projection;

    let mut state = ResponsesState::default();
    state.apply_retained_payload_limit(exact_peak);
    super::extract_tool_calls_from_body(&body, &mut state).unwrap();

    assert!(state.retained_payload_bytes().unwrap() <= exact_peak);
}

#[test]
fn buffered_numeric_projection_counts_normalization_only_outside_strings() {
    let ordinary = br#"{"quoted":"1e15","escaped":"\\\"1e15","numbers":[1,2,3]}"#;
    assert_eq!(
        super::buffered_parsed_json_bytes_upper_bound(ordinary),
        Some(ordinary.len())
    );

    let scientific = br#"{"object":"response","output":[],"numbers":[1e15,-0,18446744073709551616]}"#;
    let parsed: Value = serde_json::from_slice(scientific).unwrap();
    let exact = super::super::state::retained_json_bytes(&parsed).unwrap();
    let bound = super::buffered_parsed_json_bytes_upper_bound(scientific).unwrap();
    assert!(
        bound >= exact,
        "numeric normalization must fit the pre-parse reservation"
    );
    assert!(
        bound > scientific.len(),
        "scientific notation expands in the parsed owner"
    );
}

#[test]
fn buffered_numeric_projection_covers_plain_decimal_float_expansion() {
    let wire = format!(r#"{{"numbers":[{}]}}"#, vec!["12345678901234567.0"; 1_000].join(","));
    let parsed: Value = serde_json::from_str(&wire).unwrap();
    let normalized = serde_json::to_vec(&parsed).unwrap();
    let bound = super::buffered_parsed_json_bytes_upper_bound(wire.as_bytes()).unwrap();
    assert!(normalized.len() > wire.len());
    assert!(
        bound >= normalized.len(),
        "decimal float normalization must fit the pre-parse reservation"
    );
}

#[test]
fn private_file_search_normalization_reserves_nested_numeric_expansion() {
    // The outer response stores arguments as a string. Parsing that string
    // creates a second JSON owner whose scientific numbers become much larger.
    let arguments = format!(r#"{{"query":"q","ignored":[{}]}}"#, vec!["1e15"; 5_000].join(","));
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "name": "file_search",
            "call_id": "call_numeric",
            "arguments": arguments,
        }],
    });
    let nested: Value = serde_json::from_str(&arguments).unwrap();
    let nested_bytes = super::super::state::retained_json_bytes(&nested).unwrap();
    assert!(nested_bytes > arguments.len() * 2);

    let staging = super::output_normalization_staging_bytes(&response, true).unwrap();
    assert!(
        staging >= nested_bytes + arguments.len(),
        "staging must cover the parsed argument tree and a possible query copy"
    );
}

#[test]
fn buffered_numeric_expansion_rejects_before_parsed_tree_allocation() {
    let numbers = vec!["1e15"; 1_024].join(",");
    let body = Bytes::from(format!(r#"{{"object":"response","output":[],"numbers":[{numbers}]}}"#));
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    let parsed_bytes = super::super::state::retained_json_bytes(&parsed).unwrap();
    let baseline = ResponsesState::default().retained_payload_bytes().unwrap();
    let limit = baseline + body.len() + 1;
    assert!(parsed_bytes > body.len() + 1);

    let mut state = ResponsesState::default();
    state.apply_retained_payload_limit(limit);
    assert!(
        state.can_retain_payload(body.len()),
        "the old raw-length guard would admit parsing"
    );
    let mut rejected = false;
    let allocation = allocation_counter::measure(|| {
        rejected = super::extract_tool_calls_from_body(&body, &mut state).is_err();
    });
    assert!(rejected);
    assert!(
        allocation.bytes_total < 1_024,
        "numeric-heavy Value must be rejected before it is allocated: {allocation:?}"
    );
    assert!(state.response_object.is_null());
    assert!(state.accumulated_output.is_empty());
}

#[test]
fn buffered_usage_projection_rejects_before_copying_large_usage() {
    let body = Bytes::from(
        serde_json::to_vec(&json!({
            "object": "response",
            "output": [],
            "usage": {"detail": "x".repeat(2_000_000)}
        }))
        .unwrap(),
    );
    let limit = 3_000_000;
    assert!(body.len() < limit, "the initial parsed response fits");
    let mut state = ResponsesState::default();
    state.apply_retained_payload_limit(limit);

    let mut rejected = false;
    let allocation = allocation_counter::measure(|| {
        rejected = super::extract_tool_calls_from_body(&body, &mut state).is_err();
    });

    assert!(rejected, "the duplicate usage owner exceeds the budget");
    assert!(
        allocation.bytes_max < 3_000_000,
        "reject before the second large usage allocation: {allocation:?}"
    );
    assert!(state.response_object.is_null());
    assert!(state.usage.is_null());
}

#[test]
fn buffered_usage_replacement_charges_commit_peak_with_copied_output() {
    let old_usage = json!({"detail": "a".repeat(1_000_000)});
    let response = json!({
        "object": "response",
        "output": [{
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "x".repeat(2_000_000)}]
        }],
        "usage": {"detail": "b".repeat(1_500_000)}
    });
    let body = Bytes::from(serde_json::to_vec(&response).unwrap());
    let mut state = ResponsesState {
        usage: old_usage,
        ..ResponsesState::default()
    };
    let baseline = state.retained_payload_bytes().unwrap();
    let response_bytes = super::super::state::retained_json_bytes(&response).unwrap();
    let item_bytes = super::super::state::retained_json_bytes(&response["output"][0]).unwrap();
    let old_usage_bytes = super::super::state::retained_json_bytes(&state.usage).unwrap();
    let incoming_usage_bytes = super::super::state::retained_json_bytes(&response["usage"]).unwrap();
    let usage_growth = incoming_usage_bytes - old_usage_bytes;
    let limit = baseline + response_bytes + 2 * item_bytes + usage_growth + 64_000;
    state.apply_retained_payload_limit(limit);
    assert!(
        state.can_retain_payload(response_bytes + old_usage_bytes + incoming_usage_bytes),
        "the usage projection alone fits; copied output makes commit unsafe"
    );

    assert!(super::extract_tool_calls_from_body(&body, &mut state).is_err());
    assert!(state.accumulated_output.is_empty());
    assert!(state.response_object.is_null());
    assert_eq!(state.usage["detail"].as_str().unwrap().len(), 1_000_000);
}

#[test]
fn buffered_parse_projection_is_rejected_before_state_mutation() {
    let body = Bytes::from(
        serde_json::to_vec(&json!({
            "object": "response",
            "output": [{"type": "message", "content": "x".repeat(8_192)}]
        }))
        .unwrap(),
    );
    let mut state = ResponsesState::default();
    let retained = state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(retained + body.len() - 1);

    let failure = super::extract_tool_calls_from_body(&body, &mut state).unwrap_err();

    assert_eq!(failure.status, 502);
    assert!(state.response_object.is_null());
    assert!(state.accumulated_output.is_empty());
}

#[test]
fn file_search_argument_normalization_is_reserved_before_allocation() {
    let response = json!({
        "object": "response",
        "output": [{
            "type": "function_call",
            "name": "file_search",
            "call_id": "call_large",
            "arguments": serde_json::json!({"query": "x".repeat(4096)}).to_string()
        }]
    });
    let body = Bytes::from(serde_json::to_vec(&response).unwrap());
    let mut state = ResponsesState {
        tools: vec![json!({"type": "file_search"})],
        ..ResponsesState::default()
    };
    let retained = state.retained_payload_bytes().unwrap();
    // The parsed response fits, but its arguments parser/query copy does not.
    state.apply_retained_payload_limit(retained + body.len());

    let failure = super::extract_tool_calls_from_body(&body, &mut state).unwrap_err();

    assert_eq!(failure.status, 502);
    assert!(state.response_object.is_null());
    assert!(state.accumulated_output.is_empty());
}

#[test]
fn over_budget_file_search_reconciles_persisted_history_in_both_collectors() {
    let file = json!({
        "type": "file_search_call",
        "id": "fs_over_budget",
        "status": "searching",
        "results": [{"file_id": "file_partial"}]
    });

    for streaming in [false, true] {
        let mut state = ResponsesState {
            max_tool_calls: Some(0),
            ..ResponsesState::default()
        };
        if streaming {
            state.response_object = json!({"output": [file.clone()]});
            super::collect_streaming_output_items(&mut state).unwrap();
        } else {
            let mut response = json!({"output": [file.clone()]});
            super::collect_output_items(&mut response, &mut state, &[]);
        }

        assert_eq!(state.accumulated_output[0]["status"], "incomplete");
        assert!(state.accumulated_output[0].get("results").is_none());
        assert_eq!(state.persisted_messages[0]["status"], "incomplete");
        assert!(state.persisted_messages[0].get("results").is_none());
        assert!(state.file_search_assignments.is_empty());
    }
}

#[test]
fn over_budget_file_search_with_reused_id_updates_only_rejected_history_item() {
    let mut state = ResponsesState {
        max_tool_calls: Some(1),
        ..ResponsesState::default()
    };
    let mut response = json!({"output": [
        {"type": "file_search_call", "id": "fs_reused", "status": "searching", "results": ["first"]},
        {"type": "file_search_call", "id": "fs_reused", "status": "searching", "results": ["second"]}
    ]});

    super::collect_output_items(&mut response, &mut state, &[]);

    assert_eq!(state.file_search_assignments.len(), 1);
    assert_eq!(state.persisted_messages[0]["status"], "searching");
    assert_eq!(state.persisted_messages[0]["results"], json!(["first"]));
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
    assert_eq!(state.persisted_messages[1]["status"], "incomplete");
    assert!(state.persisted_messages[1].get("results").is_none());
}

#[tokio::test]
async fn dispatcher_failure_does_not_become_retained_budget_failure() {
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.accumulated_output.push(json!({
        "type": "file_search_call",
        "id": "fs_executed",
        "status": "completed",
        "results": []
    }));
    state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Private)];

    let failure = DispatchFailure {
        status: 502,
        code: "server_error",
        message: "MCP call validation failed".to_owned(),
    };
    let action = super::finish_response_failure(&mut ctx, state, &failure, false).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.retained_payload_failed);
    assert_eq!(
        state.pending_local_tool_synthesis,
        vec![(0, SynthesisKind::Private)],
        "unrelated dispatcher failures must preserve already-executed tool synthesis"
    );
}
