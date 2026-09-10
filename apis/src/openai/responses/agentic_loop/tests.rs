// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the agentic loop filter.

use bytes::Bytes;
use http::Method;
use praxis_filter::{FilterAction, HttpFilter, SubRequestResponseMode};
use serde_json::{Value, json};

use super::super::state::ResponsesState;
use crate::{
    openai::responses::state::McpApprovalState,
    test_utils::{make_filter_context, make_request},
};

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
    // (logical_stream: false or the filter is absent). A loop-terminal error
    // could not reach the client, so this must fail closed before dispatch.
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
    state.tool_calls.push(json!({
        "type": "function_call",
        "call_id": "call_partial",
        "name": "must_not_run",
        "arguments": "{}",
        "status": "completed"
    }));
    state.tool_search_calls.push(json!({
        "type": "tool_search_call",
        "id": "tsc_partial",
        "status": "completed"
    }));
    state.response_object = json!({
        "id": "resp_incomplete",
        "object": "response",
        "status": "incomplete",
        "output": [{
            "type": "message",
            "id": "msg_partial",
            "status": "incomplete",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "partial"}]
        }]
    });
    let partial_output = state
        .response_object
        .get("output")
        .and_then(Value::as_array)
        .expect("incomplete response should have output")
        .clone();
    state.output_items_mut().clone_from(&partial_output);
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
        ctx.extensions.get::<ResponsesState>().unwrap().accumulated_output[0]["id"],
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
    state.tool_calls.push(json!({
        "type": "function_call",
        "call_id": "call_failed",
        "name": "must_not_run",
        "arguments": "{}",
        "status": "completed"
    }));
    state.response_object = json!({
        "id": "resp_failed",
        "object": "response",
        "status": "failed",
        "output": [{
            "type": "message",
            "id": "msg_before_failure",
            "status": "incomplete",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "before failure"}]
        }]
    });
    let partial_output = state
        .response_object
        .get("output")
        .and_then(Value::as_array)
        .expect("failed response should retain partial output")
        .clone();
    state.output_items_mut().clone_from(&partial_output);
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
        ctx.extensions.get::<ResponsesState>().unwrap().accumulated_output[0]["id"],
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
fn multiple_streamed_client_function_calls_remain_dispatchable() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "test",
        "stream": true
    }));
    state.tool_calls = vec![
        json!({"type": "function_call", "call_id": "call_1", "name": "first", "status": "completed"}),
        json!({"type": "function_call", "call_id": "call_2", "name": "second", "status": "completed"}),
    ];
    state.response_object = json!({"id": "resp_multiple", "object": "response", "status": "completed", "output": []});
    ctx.set_metadata("responses.stream_completion", "terminal");
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a streamed batch must remain available to its owning dispatch filter"
    );
    assert_action(&ctx, "loop");
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.len() == 2,
        "both streamed calls must remain available"
    );
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), None);
}

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
    state.tool_calls = vec![
        json!({"type": "function_call", "call_id": "call_1", "name": "server__lookup", "status": "completed"}),
        json!({"type": "function_call", "call_id": "call_2", "name": "client", "status": "completed"}),
    ];
    state.mcp_tool_map.insert(
        ("server".to_owned(), "lookup".to_owned()),
        json!({"server_label":"server", "server_url":"http://example.com", "require_approval":"never"}),
    );
    state.response_object = json!({"id": "resp_multiple", "object": "response", "status": "completed", "output": []});
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
    state.tool_calls = vec![json!({
        "type": "function_call",
        "call_id": "call_limit",
        "name": "must_not_run",
        "status": "completed"
    })];
    state.tool_search_calls = vec![json!({
        "type": "tool_search_call",
        "id": "tsc_limit",
        "status": "completed"
    })];
    state.response_object = json!({"id": "resp_limit", "object": "response", "status": "completed", "output": []});
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
    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.len(), 2);
}

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
        state.response_object = json!({"output":[client_call.clone()]});

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
    state.response_object = json!({"output":[{
        "type":"shell_call", "call_id":"c2",
        "environment":{"type":"container_reference", "container_id":"cntr_1"}
    }]});

    assert!(!super::has_mixed_function_call_ownership(&state));
}

#[test]
fn hosted_tool_search_conflicts_with_client_function_call() {
    let mut state = make_state_with_tool_calls(vec![json!({
        "type":"function_call", "call_id":"c1", "name":"get_weather", "status":"completed"
    })]);
    state.tool_search_calls = vec![json!({
        "type":"tool_search_call", "id":"tsc_1", "status":"completed"
    })];

    assert!(
        super::has_mixed_function_call_ownership(&state),
        "hosted tool_search_call mixed with a client function_call must be rejected"
    );
}

#[test]
fn hosted_tool_search_alone_is_not_mixed_ownership() {
    let mut state = make_state_with_tool_calls(vec![]);
    state.tool_search_calls = vec![json!({
        "type":"tool_search_call", "id":"tsc_1", "status":"completed"
    })];

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
            .web_search_calls
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

#[test]
fn tool_search_call_queued_for_deferred_discovery() {
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
    assert!(
        state
            .accumulated_output
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("tsc_in_progress")),
        "the in-progress item remains client-visible in accumulated output"
    );
}

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
        .get("openai_agentic_loop")
        .expect("filter_results should contain openai_agentic_loop entry");
    let action = results.get("action").expect("should have action key");
    assert_eq!(action, expected, "openai_agentic_loop action mismatch");
}
