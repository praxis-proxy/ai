// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the agentic loop filter.

use http::Method;
use praxis_filter::{FilterAction, HttpFilter};
use serde_json::json;

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
async fn passthrough_without_state() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not write filter_results without state"
    );
}

// -----------------------------------------------------------------------------
// No Tool Calls
// -----------------------------------------------------------------------------

#[tokio::test]
async fn no_tool_calls_sets_done() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "done");
}

#[tokio::test]
async fn state_survives_done_path() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![]);
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>();
    assert!(
        state.is_some(),
        "ResponsesState must remain in extensions after done so downstream filters can read it"
    );
}

// -----------------------------------------------------------------------------
// Tool Calls Present → Loop
// -----------------------------------------------------------------------------

#[tokio::test]
async fn tool_calls_set_loop() {
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

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert_action(&ctx, "loop");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should be incremented");
    assert!(state.tool_calls.is_empty(), "tool_calls should be cleared");
}

#[tokio::test]
async fn any_tool_type_sets_loop() {
    for tool_type in ["function", "mcp", "web_search", "file_search", "custom_tool"] {
        let filter = make_filter();
        let req = make_request(Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(&req);

        let state = make_state_with_tool_calls(vec![json!({
            "type": tool_type,
            "call_id": "call_1",
        })]);
        ctx.extensions.insert(state);

        drop(filter.on_request(&mut ctx).await.unwrap());
        assert_action(&ctx, "loop");
    }
}

// -----------------------------------------------------------------------------
// Config Defaults
// -----------------------------------------------------------------------------

#[tokio::test]
async fn default_config_has_max_infer_iters_ten() {
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

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    state.tool_calls = vec![json!({"type": "function", "call_id": "call_2", "name": "test"})];
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");
}

// -----------------------------------------------------------------------------
// Iteration Limit
// -----------------------------------------------------------------------------

#[tokio::test]
async fn max_infer_iters_one_allows_exactly_one_loop() {
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

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "loop");

    let mut state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "should have incremented to 1");

    state.tool_calls = vec![json!({"type": "function", "call_id": "call_2", "name": "test"})];
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(status, Some("incomplete"), "should mark as incomplete at limit");
}

#[tokio::test]
async fn iteration_limit_exits_as_incomplete() {
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

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(
        status,
        Some("incomplete"),
        "should set incomplete status when iteration limit reached"
    );
}

// -----------------------------------------------------------------------------
// Request-Level max_tool_calls
// -----------------------------------------------------------------------------

#[tokio::test]
async fn max_tool_calls_exits_as_incomplete() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.max_tool_calls = Some(2);
    state.iteration = 2;
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(
        status,
        Some("incomplete"),
        "should set incomplete status when request max_tool_calls reached"
    );
}

#[tokio::test]
async fn max_tool_calls_below_config_limit_takes_precedence() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 10").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.max_tool_calls = Some(1);
    state.iteration = 1;
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");
}

#[tokio::test]
async fn max_tool_calls_none_defers_to_config() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_infer_iters: 5").unwrap();
    let filter = super::AgenticLoopFilter::from_config(&yaml).unwrap();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.max_tool_calls = None;
    state.iteration = 3;
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "loop");
}

// -----------------------------------------------------------------------------
// Finish Reason Length
// -----------------------------------------------------------------------------

#[tokio::test]
async fn finish_reason_length_exits_as_incomplete() {
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

    drop(filter.on_request(&mut ctx).await.unwrap());
    assert_action(&ctx, "done");

    let status = ctx.get_metadata("responses.status");
    assert_eq!(
        status,
        Some("incomplete"),
        "should set incomplete status on finish_reason length"
    );
}

// -----------------------------------------------------------------------------
// Tool Choice Reset
// -----------------------------------------------------------------------------

#[tokio::test]
async fn tool_choice_reset_after_first_iteration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.tool_choice = json!("required");
    state.iteration = 1;
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.tool_choice,
        json!("auto"),
        "tool_choice should be reset to auto after first iteration"
    );
}

#[tokio::test]
async fn tool_choice_preserved_on_first_iteration() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let mut state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    state.tool_choice = json!("required");
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.tool_choice,
        json!("required"),
        "tool_choice should be preserved on first iteration (iteration goes 0->1)"
    );
}

// -----------------------------------------------------------------------------
// Multi-Iteration State
// -----------------------------------------------------------------------------

#[tokio::test]
async fn tool_calls_cleared_after_dispatch() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "tool_calls must be empty after dispatch to prevent stale duplicates"
    );
}

#[tokio::test]
async fn iteration_incremented_on_loop() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.iteration, 1, "iteration should increment from 0 to 1");
}

// -----------------------------------------------------------------------------
// Filter Results Schema
// -----------------------------------------------------------------------------

#[tokio::test]
async fn filter_results_schema_for_branch_consumers() {
    let filter = make_filter();
    let req = make_request(Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);

    let state = make_state_with_tool_calls(vec![json!({
        "type": "function",
        "call_id": "call_1",
        "name": "test",
    })]);
    ctx.extensions.insert(state);

    drop(filter.on_request(&mut ctx).await.unwrap());

    let results = ctx
        .filter_results
        .get("agentic_loop")
        .expect("branch consumers require agentic_loop entry");
    let action = results.get("action").expect("branch consumers require action key");
    assert!(
        action == "loop" || action == "done",
        "action must be 'loop' or 'done', got: {action}"
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
    let al_config = filters
        .iter()
        .find(|f| f["filter"].as_str() == Some("agentic_loop"))
        .expect("should have agentic_loop filter");
    let filter = super::AgenticLoopFilter::from_config(al_config).unwrap();
    assert_eq!(filter.name(), "agentic_loop");
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

fn make_filter() -> Box<dyn HttpFilter> {
    super::AgenticLoopFilter::from_config(&serde_yaml::Value::Null).unwrap()
}

fn make_state_with_tool_calls(tool_calls: Vec<serde_json::Value>) -> ResponsesState {
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
