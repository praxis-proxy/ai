// SPDX-License-Identifier: MIT
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
// Output formatting tests
// -----------------------------------------------------------------------------

#[test]
fn build_output_item_completed() {
    let item = build_output_item("ws_123", "completed", "test query", &[]);
    assert_eq!(item["type"], "web_search_call");
    assert_eq!(item["id"], "ws_123");
    assert_eq!(item["status"], "completed");
    assert_eq!(item["action"]["type"], "search");
    assert_eq!(item["action"]["query"], "test query");
    assert!(item.get("sources").is_none(), "no sources when results empty");
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
    let item = build_output_item("ws_123", "completed", "search query", &results);
    let sources = item["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0]["title"], "Rust Lang");
    assert_eq!(sources[0]["url"], "https://rust-lang.org");
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
