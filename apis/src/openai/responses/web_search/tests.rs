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
fn build_tool_result_message_empty() {
    let msg = build_tool_result_message("ws_123", &[]);
    assert_eq!(msg["type"], "web_search_call");
    assert_eq!(msg["id"], "ws_123");
    assert_eq!(msg["status"], "completed");
    assert_eq!(msg["output"], "No search results found.");
}

#[test]
fn build_tool_result_message_with_results() {
    let results = vec![SearchResult {
        title: "Example".into(),
        url: "https://example.com".into(),
        snippet: "A description".into(),
    }];
    let msg = build_tool_result_message("ws_123", &results);
    let output = msg["output"].as_str().unwrap();
    assert!(output.contains("[1] Example"));
    assert!(output.contains("https://example.com"));
    assert!(output.contains("A description"));
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
