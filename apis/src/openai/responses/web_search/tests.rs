// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the `openai_web_search` filter.

use super::*;

// -----------------------------------------------------------------------------
// Helper: build filter from YAML
// -----------------------------------------------------------------------------

fn make_filter_yaml(provider: &str, api_key: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&format!(
        r#"
provider: {provider}
api_key: "{api_key}"
default_context_size: medium
timeout_ms: 5000
"#,
    ))
    .unwrap()
}

// -----------------------------------------------------------------------------
// from_config tests
// -----------------------------------------------------------------------------

#[test]
fn from_config_brave() {
    let yaml = make_filter_yaml("brave", "brave-test-key");
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_ok(), "should build filter from valid brave config");
}

#[test]
fn from_config_tavily() {
    let yaml = make_filter_yaml("tavily", "tvly-test-key");
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_ok(), "should build filter from valid tavily config");
}

#[test]
fn from_config_missing_provider() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
api_key: "test-key"
"#,
    )
    .unwrap();
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_err(), "should reject config without provider");
}

#[test]
fn from_config_missing_api_key() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider: brave
"#,
    )
    .unwrap();
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_err(), "should reject config without api_key");
}

#[test]
fn from_config_empty_api_key() {
    let yaml = make_filter_yaml("brave", "");
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_err(), "should reject empty api_key");
}

#[test]
fn from_config_unknown_provider() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider: google
api_key: "test-key"
"#,
    )
    .unwrap();
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_err(), "should reject unknown provider");
}

#[test]
fn from_config_unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider: brave
api_key: "test-key"
unknown_field: true
"#,
    )
    .unwrap();
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(filter.is_err(), "should reject unknown config fields");
}

#[test]
fn from_config_rejects_max_body_bytes() {
    // openai_web_search is a read-only reader that produces no request body,
    // so it carries no per-filter raw-body cap: raw request size is governed
    // by the pipeline's body_limits. A stale `max_body_bytes` is a hard error
    // rather than a silently bypassable knob (the core merges sibling buffer
    // modes to the larger limit).
    let yaml: serde_yaml::Value = serde_yaml::from_str(
        r#"
provider: brave
api_key: "test-key"
max_body_bytes: 1024
"#,
    )
    .unwrap();
    let filter = WebSearchFilter::from_config(&yaml);
    assert!(
        filter.is_err(),
        "openai_web_search must reject max_body_bytes; raw body size is governed by the pipeline's body_limits"
    );
}

// -----------------------------------------------------------------------------
// Filter trait tests
// -----------------------------------------------------------------------------

#[test]
fn filter_name() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_web_search");
}

#[tokio::test]
async fn on_request_is_noop() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "on_request should always continue"
    );
}

// -----------------------------------------------------------------------------
// emit_status tests
// -----------------------------------------------------------------------------

#[test]
fn emit_status_uses_valid_key() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    emit_status(&mut ctx, "ws_proactive", "searching");

    let results = ctx.filter_results.get("openai_web_search").unwrap();
    assert_eq!(
        results.get("web_search_call_ws_proactive"),
        Some("searching"),
        "status should be stored with underscore-separated key"
    );
}

// -----------------------------------------------------------------------------
// on_response_body: Loop Signaling
// -----------------------------------------------------------------------------

#[test]
fn on_response_body_signals_loop_when_web_search_calls_present() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test query"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let results = ctx
        .filter_results
        .get("openai_web_search")
        .expect("should have openai_web_search entry");
    assert_eq!(results.get("action"), Some("loop"));
}

#[test]
fn on_response_body_signals_done_when_no_web_search_calls() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let results = ctx
        .filter_results
        .get("openai_web_search")
        .expect("should have openai_web_search entry");
    assert_eq!(results.get("action"), Some("done"));
}

#[test]
fn on_response_body_passthrough_without_state() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not write filter_results without state"
    );
}

#[test]
fn on_response_body_passthrough_on_non_end_of_stream() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_response_body(&mut ctx, &mut None, false).unwrap();
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.filter_results.is_empty(),
        "should not set filter_results on non-end-of-stream"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Passthrough
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_body_passthrough_without_state() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

#[tokio::test]
async fn on_request_body_passthrough_when_no_web_search_calls() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let state = ResponsesState::from_request_body(body);
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.web_search_calls.is_empty());
}

#[tokio::test]
async fn on_request_body_passthrough_on_non_end_of_stream() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_1",
        "action": {"type": "search", "query": "test"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, false).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
}

// -----------------------------------------------------------------------------
// on_request_body: Search Execution
// -----------------------------------------------------------------------------

fn make_filter_yaml_with_base_url(provider: &str, api_key: &str, base_url: &str) -> serde_yaml::Value {
    serde_yaml::from_str(&format!(
        r#"
provider: {provider}
api_key: "{api_key}"
default_context_size: medium
timeout_ms: 5000
base_url: "{base_url}"
allow_private_base_url: true
"#,
    ))
    .unwrap()
}

fn spawn_brave_mock(listener: std::net::TcpListener) {
    use std::io::{Read as _, Write as _};
    let body = serde_json::json!({
        "web": {
            "results": [{
                "title": "Rust Lang",
                "url": "https://rust-lang.org",
                "description": "Systems programming language"
            }]
        }
    })
    .to_string();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _n = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
}

/// Serve one `(status, body)` per sequential search callout.
///
/// Each entry answers exactly one connection, letting a single test drive a
/// mixed batch where earlier calls succeed and later calls fail.
fn spawn_search_responses(listener: std::net::TcpListener, responses: Vec<(u16, String)>) {
    use std::io::{Read as _, Write as _};
    std::thread::spawn(move || {
        for (status, body) in responses {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap();
            let response = format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
    });
}

/// A single Brave web result body for search-execution tests.
fn brave_ok_body() -> String {
    serde_json::json!({
        "web": {"results": [{
            "title": "Rust Lang",
            "url": "https://rust-lang.org",
            "description": "Systems programming language"
        }]}
    })
    .to_string()
}

#[tokio::test]
async fn on_request_body_executes_search_and_populates_state() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({
        "model": "gpt-4o",
        "input": "test",
        "include": ["web_search_call.action.sources"],
    });
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_exec_1",
        "action": {"type": "search", "query": "rust language"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.web_search_calls.is_empty(),
        "calls should be cleared after execution"
    );
    assert!(!state.messages.is_empty(), "tool result should be appended to messages");
    assert!(
        !state.persisted_messages.is_empty(),
        "tool result should be appended to persisted_messages"
    );
    assert!(
        !state.accumulated_output.is_empty(),
        "output item should be appended to accumulated_output"
    );

    let output = &state.accumulated_output[0];
    assert_eq!(output["type"], "web_search_call");
    assert_eq!(output["id"], "ws_exec_1");
    assert_eq!(output["status"], "completed");
    assert_eq!(output["action"]["query"], "rust language");
    assert!(output.get("sources").is_none(), "no top-level sources");
    let sources = output["action"]["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0]["type"], "url");
    assert_eq!(sources[0]["url"], "https://rust-lang.org");
}

#[tokio::test]
async fn on_request_body_omits_sources_without_include() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_exec_2",
        "action": {"type": "search", "query": "rust language"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let output = &state.accumulated_output[0];
    assert_eq!(output["action"]["query"], "rust language");
    assert!(
        output["action"].get("sources").is_none(),
        "sources omitted when not in include"
    );
}

#[tokio::test]
async fn on_request_body_missing_query_produces_incomplete_status() {
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_no_query",
        "action": {"type": "search"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.web_search_calls.is_empty());

    let output = &state.accumulated_output[0];
    assert_eq!(
        output["status"], "incomplete",
        "missing query should produce incomplete status"
    );

    let bridge_call = &state.messages[state.messages.len() - 2];
    assert_eq!(bridge_call["type"], "function_call");
    assert_eq!(bridge_call["arguments"], r#"{}"#);
    assert_eq!(
        bridge_call["status"], "completed",
        "the synthetic function call was fully generated; its output carries the incomplete execution result"
    );
    let bridge_output = state.messages.last().unwrap();
    assert_eq!(bridge_output["type"], "function_call_output");
    assert_eq!(
        bridge_output["output"],
        "Web search could not run because the query was missing."
    );
    assert_eq!(
        state.persisted_messages.last(),
        Some(bridge_output),
        "persisted history must contain the truthful model continuation"
    );
}

#[tokio::test]
async fn on_request_body_provider_failure_produces_failed_item_and_truthful_input() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_search_responses(listener, vec![(503, String::new())]);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    // The response phase already accumulated the model's placeholder call.
    state.accumulated_output = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_fail_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust language"}
    })];
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_fail_1",
        "action": {"type": "search", "query": "rust language"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a provider failure must never reject the Response"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.web_search_calls.is_empty(),
        "calls should be cleared after execution"
    );

    // The placeholder is replaced in place: exactly one public item, marked failed.
    assert_eq!(
        state.accumulated_output.len(),
        1,
        "the failed outcome must replace the placeholder, not duplicate it"
    );
    let output = &state.accumulated_output[0];
    assert_eq!(output["type"], "web_search_call");
    assert_eq!(output["id"], "ws_fail_1");
    assert_eq!(output["status"], "failed");
    assert_eq!(output["action"]["query"], "rust language");

    // The model receives the bounded failure notice through a backend-valid
    // function_call_output bridge, never an invalid hosted web_search_call (#808).
    assert!(
        state
            .messages
            .iter()
            .all(|m| m.get("type").and_then(Value::as_str) != Some("web_search_call")),
        "failure continuation must not feed the model a hosted web_search_call: {:?}",
        state.messages
    );
    let output = state.messages.last().unwrap();
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["output"], "Web search unavailable.");
    assert_eq!(
        state.persisted_messages.last().unwrap()["output"],
        "Web search unavailable.",
        "persisted history mirrors the model input"
    );
}

#[tokio::test]
async fn on_request_body_empty_results_remain_completed() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let empty_body = serde_json::json!({"web": {"results": []}}).to_string();
    spawn_search_responses(listener, vec![(200, empty_body)]);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_empty_1",
        "action": {"type": "search", "query": "rust language"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let output = &state.accumulated_output[0];
    assert_eq!(
        output["status"], "completed",
        "a successful zero-result search stays completed"
    );
    let output = state.messages.last().unwrap();
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["output"], "No search results found.");
}

#[tokio::test]
async fn on_request_body_mixed_batch_preserves_completed_and_failed() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_search_responses(listener, vec![(200, brave_ok_body()), (503, String::new())]);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    // Both placeholders were accumulated during the response phase.
    state.accumulated_output = vec![
        serde_json::json!({
            "type": "web_search_call",
            "id": "ws_ok",
            "status": "completed",
            "action": {"type": "search", "query": "rust language"}
        }),
        serde_json::json!({
            "type": "web_search_call",
            "id": "ws_fail",
            "status": "completed",
            "action": {"type": "search", "query": "rust crates"}
        }),
    ];
    state.web_search_calls = vec![
        serde_json::json!({
            "type": "web_search_call",
            "id": "ws_ok",
            "action": {"type": "search", "query": "rust language"}
        }),
        serde_json::json!({
            "type": "web_search_call",
            "id": "ws_fail",
            "action": {"type": "search", "query": "rust crates"}
        }),
    ];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.accumulated_output.len(),
        2,
        "each placeholder is replaced in place, so no duplicates are produced"
    );

    let completed = &state.accumulated_output[0];
    assert_eq!(completed["id"], "ws_ok");
    assert_eq!(completed["status"], "completed");

    let failed = &state.accumulated_output[1];
    assert_eq!(failed["id"], "ws_fail");
    assert_eq!(failed["status"], "failed");

    // Each call bridges its own outcome through a function_call_output, appended
    // after the original input in call order.
    let outputs: Vec<&Value> = state
        .messages
        .iter()
        .filter(|m| m.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .collect();
    assert_eq!(outputs.len(), 2, "one function_call_output per call");
    assert!(
        outputs[0]["output"].as_str().unwrap().contains("Rust Lang"),
        "the completed call feeds its results to the model: {:?}",
        outputs[0]
    );
    assert_eq!(
        outputs[1]["output"], "Web search unavailable.",
        "the failed call feeds the bounded notice to the model"
    );
}

// -----------------------------------------------------------------------------
// on_request_body: Backend-Valid Continuation (issue #808)
// -----------------------------------------------------------------------------

#[tokio::test]
async fn on_request_body_appends_backend_valid_continuation() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![serde_json::json!({
        "type": "web_search_call",
        "id": "ws_bridge_1",
        "action": {"type": "search", "query": "rust language"}
    })];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "search execution should continue"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    for scope in [&state.messages, &state.persisted_messages] {
        assert!(
            scope
                .iter()
                .all(|m| m.get("type").and_then(Value::as_str) != Some("web_search_call")),
            "continuation must not contain hosted web_search_call items: {scope:?}"
        );
    }

    let call = state
        .messages
        .iter()
        .find(|m| m.get("type").and_then(Value::as_str) == Some("function_call"))
        .expect("continuation should include a synthetic function_call");
    assert_eq!(call["name"], "web_search");
    let call_id = call["call_id"].as_str().expect("function_call needs a call_id");

    assert!(
        call_id.len() <= 64,
        "bridge call_id must be <= 64 chars, got {}: {call_id}",
        call_id.len()
    );
    assert_ne!(call_id, "ws_bridge_1", "bridge must not reuse the unbounded hosted id");

    let output = state
        .messages
        .iter()
        .find(|m| m.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .expect("continuation should include a function_call_output");
    assert_eq!(output["call_id"], call_id, "output must reference the call");
    let text = output["output"].as_str().expect("output should be a string");
    assert!(
        text.contains("rust-lang.org"),
        "search results must reach the model: {text}"
    );

    let public = &state.accumulated_output[0];
    assert_eq!(
        public["type"], "web_search_call",
        "public web_search_call output is retained for the response only"
    );
    assert_eq!(public["id"], "ws_bridge_1");
}

#[tokio::test]
async fn web_search_continuation_serializes_backend_valid_input() {
    use crate::openai::responses::{AgenticLoopFilter, openai_responses_proxy::ResponsesProxyFilter};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    spawn_brave_mock(listener);

    let agentic_loop = AgenticLoopFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let web_search = WebSearchFilter::from_config(&make_filter_yaml_with_base_url(
        "brave",
        "test-key",
        &format!("http://{addr}"),
    ))
    .unwrap();
    let proxy = ResponsesProxyFilter::from_config(&serde_yaml::Value::Null).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "search rust",
        "tools": [{"type": "web_search"}],
    });
    ctx.extensions.insert(ResponsesState::from_request_body(request_body));

    // Response phase: the model emits a web_search_call; agentic_loop extracts it.
    let response_body = serde_json::json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-4.1",
        "output": [{
            "type": "web_search_call",
            "id": "ws_e2e_1",
            "status": "completed",
            "action": {"type": "search", "query": "rust language"}
        }]
    });
    let mut resp = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));
    let action = agentic_loop.on_response_body(&mut ctx, &mut resp, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "agentic_loop should signal loop"
    );

    // Re-entry request phase: web_search executes the search and bridges results,
    // then the proxy rebuilds the outbound body from continuation state.
    let action = web_search.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "web_search dispatch should continue"
    );
    let mut body = Some(Bytes::from(br#"{"model":"gpt-4.1","input":"search rust"}"#.to_vec()));
    let action = proxy.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "proxy body rebuild should continue"
    );

    let serialized: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let input = serialized["input"]
        .as_array()
        .expect("rebuilt body should carry an input array");
    assert!(
        input
            .iter()
            .all(|m| m.get("type").and_then(Value::as_str) != Some("web_search_call")),
        "#808: backend input must not contain hosted web_search_call items: {input:?}"
    );
    assert!(
        input
            .iter()
            .any(|m| m.get("type").and_then(Value::as_str) == Some("function_call_output")),
        "backend input should carry search results via function_call_output: {input:?}"
    );
}

// -----------------------------------------------------------------------------
// Output formatting tests
// -----------------------------------------------------------------------------

#[test]
fn build_output_item_completed_with_sources_included() {
    let item = build_output_item("ws_123", "completed", "test query", &[], true);
    assert_eq!(item["type"], "web_search_call");
    assert_eq!(item["id"], "ws_123");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["action"]["type"], "search");
    assert_eq!(item["action"]["query"], "test query");
    let sources = item["action"]["sources"].as_array().unwrap();
    assert!(sources.is_empty(), "no sources when results empty");
}

#[test]
fn build_output_item_omits_sources_when_not_included() {
    let results = vec![SearchResult {
        title: "Rust Lang".into(),
        url: "https://rust-lang.org".into(),
        snippet: "Systems".into(),
    }];
    let item = build_output_item("ws_123", "completed", "test query", &results, false);
    assert_eq!(item["action"]["type"], "search");
    assert_eq!(item["action"]["query"], "test query");
    assert!(
        item["action"].get("sources").is_none(),
        "sources omitted when include_sources is false"
    );
}

#[test]
fn build_output_item_with_results() {
    let results = vec![
        SearchResult {
            title: "Rust Lang".into(),
            url: "https://rust-lang.org".into(),
            snippet: "Systems".into(),
        },
        SearchResult {
            title: "Crates.io".into(),
            url: "https://crates.io".into(),
            snippet: "Packages".into(),
        },
    ];
    let item = build_output_item("ws_123", "completed", "search query", &results, true);
    assert!(item.get("sources").is_none(), "no top-level sources");
    let sources = item["action"]["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0]["type"], "url");
    assert_eq!(sources[0]["url"], "https://rust-lang.org");
    assert!(sources[0].get("title").is_none(), "title excluded from url source");
    assert_eq!(sources[1]["type"], "url");
    assert_eq!(sources[1]["url"], "https://crates.io");
}

#[test]
fn build_tool_result_messages_empty() {
    let [call, output] = build_tool_result_messages("ws_123", "completed", "rust", &[]);
    assert_eq!(
        call["type"], "function_call",
        "continuation bridge is a backend-valid function_call/function_call_output pair, never a hosted web_search_call"
    );
    assert_eq!(call["call_id"], "ws_123");
    assert_eq!(call["name"], "web_search");
    assert_eq!(call["arguments"], r#"{"query":"rust"}"#);
    assert_eq!(call["status"], "completed");
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["call_id"], "ws_123");
    assert_eq!(output["output"], "No search results found.");
}

#[test]
fn build_tool_result_messages_with_results() {
    let results = vec![SearchResult {
        title: "Example".into(),
        url: "https://example.com".into(),
        snippet: "A description".into(),
    }];
    let [call, output] = build_tool_result_messages("ws_123", "completed", "example query", &results);
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["arguments"], r#"{"query":"example query"}"#);
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["call_id"], "ws_123");
    let text = output["output"].as_str().unwrap();
    assert!(text.contains("[1] Example"));
    assert!(text.contains("https://example.com"));
    assert!(text.contains("A description"));
}

// -----------------------------------------------------------------------------
// Bridge call_id bounding (issue #808)
// -----------------------------------------------------------------------------

#[test]
fn bridge_call_id_is_bounded_for_unbounded_source_id() {
    // OpenAI web-search ids have no maximum length, but a synthetic function
    // call_id must stay within the OpenResponses 64-char limit.
    let long_source = format!("ws_{}", "a".repeat(4096));
    let id = bridge_call_id(&long_source, "rust language", 0);
    assert!(
        id.len() <= 64,
        "bridge call_id must be <= 64 chars for an unbounded source id, got {}: {id}",
        id.len()
    );
}

#[test]
fn bridge_call_id_is_deterministic() {
    assert_eq!(
        bridge_call_id("ws_1", "rust", 0),
        bridge_call_id("ws_1", "rust", 0),
        "identical inputs must yield the same bridge call_id"
    );
}

#[test]
fn bridge_call_id_is_unique_for_duplicate_source_ids() {
    // Two calls in one turn sharing a source id must not collide, or their
    // function_call_output pairs would be ambiguous.
    assert_ne!(
        bridge_call_id("ws_dup", "rust", 0),
        bridge_call_id("ws_dup", "rust", 1),
        "duplicate source ids must produce distinct bridge call_ids per index"
    );
}

#[test]
fn bridge_call_id_is_unique_for_absent_source_ids() {
    // Absent ids collapse to the "ws_unknown" fallback; the call index still
    // disambiguates each bridge.
    assert_ne!(
        bridge_call_id("ws_unknown", "rust", 0),
        bridge_call_id("ws_unknown", "rust", 1),
        "absent source ids must still produce distinct bridge call_ids per index"
    );
}

#[test]
fn build_failed_tool_result_messages_carry_bounded_notice() {
    let [call, output] = build_failed_tool_result_messages("ws_123", "rust");
    assert_eq!(
        call["type"], "function_call",
        "failure bridge is a backend-valid function_call/function_call_output pair, never a hosted web_search_call"
    );
    assert_eq!(call["call_id"], "ws_123");
    assert_eq!(call["name"], "web_search");
    assert_eq!(call["arguments"], r#"{"query":"rust"}"#);
    assert_eq!(call["status"], "completed");
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["call_id"], "ws_123");
    assert_eq!(
        output["output"], "Web search unavailable.",
        "failed tool result must feed the model the bounded notice"
    );
}

#[test]
fn upsert_output_item_replaces_matching_web_search_call() {
    let mut accumulated = vec![
        serde_json::json!({"type": "message", "id": "msg_1"}),
        serde_json::json!({"type": "web_search_call", "id": "ws_1", "status": "completed"}),
    ];
    upsert_output_item(
        &mut accumulated,
        serde_json::json!({"type": "web_search_call", "id": "ws_1", "status": "failed"}),
    );
    assert_eq!(accumulated.len(), 2, "matching id replaces rather than appends");
    assert_eq!(accumulated[1]["status"], "failed");
}

#[test]
fn upsert_output_item_appends_when_no_match() {
    let mut accumulated = vec![serde_json::json!({"type": "web_search_call", "id": "ws_1"})];
    upsert_output_item(
        &mut accumulated,
        serde_json::json!({"type": "web_search_call", "id": "ws_2", "status": "failed"}),
    );
    assert_eq!(accumulated.len(), 2, "a new id appends a fresh item");
    assert_eq!(accumulated[1]["id"], "ws_2");
}

#[test]
fn build_tool_result_messages_incomplete_reports_not_performed() {
    // A non-dispatched call (over-budget or missing query) must not be
    // misrepresented to the model as a completed search with no results.
    let [call, output] = build_tool_result_messages("ws_123", "incomplete", "rust", &[]);
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["call_id"], "ws_123");
    assert_eq!(output["type"], "function_call_output");
    assert_eq!(output["output"], "Web search not performed.");
}

#[test]
fn format_search_results_multiple() {
    let results = vec![
        SearchResult {
            title: "First".into(),
            url: "https://first.com".into(),
            snippet: "First result".into(),
        },
        SearchResult {
            title: "Second".into(),
            url: "https://second.com".into(),
            snippet: "Second result".into(),
        },
    ];
    let formatted = format_search_results(&results);
    assert!(formatted.contains("[1] First"));
    assert!(formatted.contains("[2] Second"));
    assert!(formatted.contains("\n\n"), "results should be separated by blank line");
}

// -----------------------------------------------------------------------------
// Call budget: max_tool_calls and the server-side per-continuation cap
// -----------------------------------------------------------------------------

/// Brave mock that serves every connection and counts dispatched requests.
fn spawn_counting_brave_mock(listener: std::net::TcpListener) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
    use std::{
        io::{Read as _, Write as _},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    let counter = Arc::new(AtomicUsize::new(0));
    let thread_counter = Arc::clone(&counter);
    let body = serde_json::json!({
        "web": {
            "results": [{
                "title": "Rust Lang",
                "url": "https://rust-lang.org",
                "description": "Systems programming language"
            }]
        }
    })
    .to_string();
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap_or(0);
            thread_counter.fetch_add(1, Ordering::SeqCst);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _written = stream.write_all(response.as_bytes());
        }
    });
    counter
}

fn web_search_call(id: &str, query: &str) -> Value {
    serde_json::json!({
        "type": "web_search_call",
        "id": id,
        "action": {"type": "search", "query": query}
    })
}

/// Find the `function_call_output` bridged for `query` in a message list.
///
/// The backend-valid continuation is a `function_call`/`function_call_output`
/// pair keyed by a bounded hash id (issue #808), so the tool result is located
/// by the query carried in the `function_call` arguments rather than by the
/// unbounded public `web_search_call.id`.
fn find_bridge_output<'a>(messages: &'a [Value], query: &str) -> Option<&'a Value> {
    let needle = serde_json::json!({ "query": query }).to_string();
    let call_id = messages.iter().find_map(|m| {
        (m["type"] == "function_call" && m["arguments"] == needle)
            .then(|| m["call_id"].as_str())
            .flatten()
    })?;
    messages
        .iter()
        .find(|m| m["type"] == "function_call_output" && m["call_id"] == call_id)
}

#[test]
fn remaining_budget_uses_server_cap_without_max_tool_calls() {
    let state = ResponsesState::from_request_body(serde_json::json!({"model": "gpt-4o", "input": "x"}));
    assert_eq!(
        remaining_web_search_budget(&state),
        MAX_WEB_SEARCH_CALLS_PER_CONTINUATION,
        "omitting max_tool_calls falls back to the server hard cap"
    );
}

#[test]
fn remaining_budget_caps_large_max_tool_calls_at_server_cap() {
    let state =
        ResponsesState::from_request_body(serde_json::json!({"model": "gpt-4o", "input": "x", "max_tool_calls": 1000}));
    assert_eq!(
        remaining_web_search_budget(&state),
        MAX_WEB_SEARCH_CALLS_PER_CONTINUATION,
        "a large client budget is still bounded by the server cap"
    );
}

#[test]
fn remaining_budget_subtracts_prior_executions() {
    let mut state =
        ResponsesState::from_request_body(serde_json::json!({"model": "gpt-4o", "input": "x", "max_tool_calls": 5}));
    assert_eq!(remaining_web_search_budget(&state), 5);
    state.web_search_calls_executed = 2;
    assert_eq!(
        remaining_web_search_budget(&state),
        3,
        "prior dispatches reduce the remaining client allowance"
    );
}

#[test]
fn remaining_budget_saturates_when_exhausted() {
    let mut state =
        ResponsesState::from_request_body(serde_json::json!({"model": "gpt-4o", "input": "x", "max_tool_calls": 2}));
    state.web_search_calls_executed = 5;
    assert_eq!(
        remaining_web_search_budget(&state),
        0,
        "an over-budget count saturates to zero remaining searches"
    );
}

#[tokio::test]
async fn on_request_body_honors_client_max_tool_calls() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let count = spawn_counting_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test", "max_tool_calls": 1});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![web_search_call("ws_a", "first"), web_search_call("ws_b", "second")];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    assert_eq!(
        count.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "only one provider request should be dispatched under max_tool_calls=1"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.web_search_calls_executed, 1, "one search recorded as executed");
    assert!(
        state.web_search_calls.is_empty(),
        "pending calls cleared after execution"
    );
    assert_eq!(state.accumulated_output.len(), 2, "both calls produce an output item");
    assert_eq!(state.accumulated_output[0]["id"], "ws_a");
    assert_eq!(state.accumulated_output[0]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["id"], "ws_b");
    assert_eq!(
        state.accumulated_output[1]["status"], "incomplete",
        "the over-budget call is surfaced as incomplete, not executed"
    );
    assert_eq!(
        state.accumulated_output[1]["action"]["query"], "second",
        "the declined query is preserved in the incomplete item"
    );

    // The model-facing bridge (state.messages) and the durable rehydration
    // history (persisted_messages) must tell the model the truth about the
    // over-budget call: it was not performed, not a completed empty search.
    let bridge = find_bridge_output(&state.messages, "second").expect("ws_b bridge present");
    assert_eq!(
        bridge["output"], "Web search not performed.",
        "the over-budget bridge must not fabricate a no-results outcome"
    );
    let persisted = find_bridge_output(&state.persisted_messages, "second").expect("ws_b persisted");
    assert_eq!(
        persisted["output"], "Web search not performed.",
        "durable history must not persist a false completed outcome"
    );
    // The dispatched call remains a truthful bridge carrying real results.
    let dispatched = find_bridge_output(&state.messages, "first").expect("ws_a bridge present");
    assert_ne!(
        dispatched["output"], "Web search not performed.",
        "the dispatched call must carry a real search outcome"
    );
    assert!(
        dispatched["output"].as_str().unwrap().contains("Rust Lang"),
        "the dispatched bridge carries the provider results"
    );
}

#[tokio::test]
async fn on_request_body_budget_spans_iterations() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let count = spawn_counting_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test", "max_tool_calls": 2});
    let mut state = ResponsesState::from_request_body(body);
    // Simulate one search already dispatched in a prior IRR iteration.
    state.web_search_calls_executed = 1;
    state.web_search_calls = vec![web_search_call("ws_c", "third"), web_search_call("ws_d", "fourth")];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    assert_eq!(
        count.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "only the single remaining budget slot is dispatched this round"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.web_search_calls_executed, 2,
        "cumulative executions reach but do not exceed the client budget"
    );
    assert_eq!(state.accumulated_output[0]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
}

#[tokio::test]
async fn on_request_body_exhausted_budget_dispatches_nothing() {
    // Budget is already spent, so no live provider is contacted.
    let yaml = make_filter_yaml("brave", "test-key");
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test", "max_tool_calls": 2});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls_executed = 2;
    state.web_search_calls = vec![web_search_call("ws_e", "fifth")];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.web_search_calls_executed, 2, "no further searches executed");
    assert!(state.web_search_calls.is_empty());
    assert_eq!(state.accumulated_output.len(), 1);
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
    assert_eq!(state.accumulated_output[0]["action"]["query"], "fifth");
}

#[tokio::test]
async fn on_request_body_without_max_tool_calls_dispatches_all_under_cap() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let count = spawn_counting_brave_mock(listener);

    let yaml = make_filter_yaml_with_base_url("brave", "test-key", &format!("http://{addr}"));
    let filter = WebSearchFilter::from_config(&yaml).unwrap();

    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let body = serde_json::json!({"model": "gpt-4o", "input": "test"});
    let mut state = ResponsesState::from_request_body(body);
    state.web_search_calls = vec![web_search_call("ws_f", "a"), web_search_call("ws_g", "b")];
    ctx.extensions.insert(state);

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));

    assert_eq!(
        count.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "both searches dispatch when the client omits max_tool_calls and the count is under the server cap"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.web_search_calls_executed, 2);
    assert!(
        state
            .accumulated_output
            .iter()
            .all(|item| item["status"] == "completed"),
        "all searches completed under the server cap"
    );
}
