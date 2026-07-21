// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_mcp_dispatch` filter.

use std::collections::HashMap;

use bytes::Bytes;
use praxis_filter::FilterAction;
use serde_json::json;

use super::{
    McpDispatchFilter, build_error_result, build_success_result, content_blocks_to_text, execute_mcp_calls,
    execute_single_call, extract_arguments, extract_call_id, extract_mcp_tool_calls, find_approval_required,
    is_mcp_tool_call, parse_call_arguments, process_call_result, resolve_tool_entry,
};
use crate::{
    openai::responses::{
        mcp_dispatch::{
            approval::{ApprovalPolicy, parse_approval_policy, requires_approval},
            config::{McpDispatchConfig, build_config},
        },
        state::ResponsesState,
    },
    test_utils::{make_filter_context, make_request},
};

// =========================================================================
// Approval Policy Parsing
// =========================================================================

#[test]
fn parse_approval_always() {
    let def = json!({"require_approval": "always"});
    assert_eq!(parse_approval_policy(&def), ApprovalPolicy::Always);
}

#[test]
fn parse_approval_never() {
    let def = json!({"require_approval": "never"});
    assert_eq!(parse_approval_policy(&def), ApprovalPolicy::Never);
}

#[test]
fn parse_approval_absent_defaults_to_always() {
    let def = json!({});
    assert_eq!(parse_approval_policy(&def), ApprovalPolicy::Always);
}

#[test]
fn parse_approval_filter() {
    let def = json!({
        "require_approval": {
            "always": {"tool_names": ["dangerous_tool"]},
            "never": {"tool_names": ["safe_tool"]}
        }
    });
    match parse_approval_policy(&def) {
        ApprovalPolicy::Filter { always, never } => {
            assert_eq!(always, vec!["dangerous_tool"]);
            assert_eq!(never, vec!["safe_tool"]);
        },
        other => panic!("expected Filter, got {other:?}"),
    }
}

#[test]
fn parse_approval_filter_flat_array_fallback() {
    let def = json!({
        "require_approval": {
            "always": ["dangerous_tool"],
            "never": ["safe_tool"]
        }
    });
    match parse_approval_policy(&def) {
        ApprovalPolicy::Filter { always, never } => {
            assert_eq!(always, vec!["dangerous_tool"]);
            assert_eq!(never, vec!["safe_tool"]);
        },
        other => panic!("expected Filter, got {other:?}"),
    }
}

#[test]
fn parse_approval_unrecognized_string_defaults_to_always() {
    let def = json!({"require_approval": "maybe"});
    assert_eq!(parse_approval_policy(&def), ApprovalPolicy::Always);
}

// =========================================================================
// Approval Evaluation
// =========================================================================

#[test]
fn requires_approval_always() {
    assert!(requires_approval(&ApprovalPolicy::Always, "any_tool"));
}

#[test]
fn requires_approval_never() {
    assert!(!requires_approval(&ApprovalPolicy::Never, "any_tool"));
}

#[test]
fn requires_approval_filter_always_list() {
    let policy = ApprovalPolicy::Filter {
        always: vec!["dangerous".to_owned()],
        never: vec![],
    };
    assert!(requires_approval(&policy, "dangerous"));
}

#[test]
fn requires_approval_filter_never_list() {
    let policy = ApprovalPolicy::Filter {
        always: vec![],
        never: vec!["safe".to_owned()],
    };
    assert!(!requires_approval(&policy, "safe"));
}

#[test]
fn requires_approval_filter_always_takes_precedence() {
    let policy = ApprovalPolicy::Filter {
        always: vec!["tool_x".to_owned()],
        never: vec!["tool_x".to_owned()],
    };
    assert!(
        requires_approval(&policy, "tool_x"),
        "always should take precedence over never"
    );
}

#[test]
fn requires_approval_filter_unlisted_defaults_to_true() {
    let policy = ApprovalPolicy::Filter {
        always: vec![],
        never: vec!["other".to_owned()],
    };
    assert!(
        requires_approval(&policy, "unknown_tool"),
        "unlisted tools should default to requiring approval"
    );
}

// =========================================================================
// extract_tool_names edge cases (via parse_approval_policy)
// =========================================================================

#[test]
fn parse_approval_filter_absent_sub_keys() {
    let def = json!({"require_approval": {}});
    match parse_approval_policy(&def) {
        ApprovalPolicy::Filter { always, never } => {
            assert!(always.is_empty());
            assert!(never.is_empty());
        },
        other => panic!("expected Filter, got {other:?}"),
    }
}

#[test]
fn parse_approval_filter_non_object_non_array_value() {
    let def = json!({"require_approval": {"always": 42, "never": true}});
    match parse_approval_policy(&def) {
        ApprovalPolicy::Filter { always, never } => {
            assert!(always.is_empty(), "numeric value should produce empty list");
            assert!(never.is_empty(), "boolean value should produce empty list");
        },
        other => panic!("expected Filter, got {other:?}"),
    }
}

// =========================================================================
// HttpFilter trait method coverage
// =========================================================================

#[test]
fn filter_response_body_access() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert_eq!(filter.response_body_access(), praxis_filter::BodyAccess::ReadOnly);
}

#[test]
fn filter_response_body_mode() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("max_body_bytes: 1024").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert!(
        matches!(
            filter.response_body_mode(),
            praxis_filter::BodyMode::StreamBuffer { max_bytes: Some(1024) }
        ),
        "should return StreamBuffer with configured max_bytes"
    );
}

// =========================================================================
// MCP Tool Call Identification
// =========================================================================

fn sample_tool_map() -> HashMap<(String, String), serde_json::Value> {
    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "weather",
            "server_url": "http://weather.example.com/mcp",
            "headers": null,
            "authorization": null,
            "tool_definition": {"name": "get_weather"},
            "require_approval": null,
        }),
    );
    map.insert(
        ("docs".to_owned(), "search_docs".to_owned()),
        json!({
            "server_label": "docs",
            "server_url": "http://docs.example.com/mcp",
            "headers": null,
            "authorization": null,
            "tool_definition": {"name": "search_docs"},
            "require_approval": null,
        }),
    );
    map
}

#[test]
fn is_mcp_tool_call_matches_known_tool() {
    let tool_map = sample_tool_map();
    let tc = json!({"name": "get_weather", "call_id": "call_1"});
    assert!(is_mcp_tool_call(&tc, &tool_map));
}

#[test]
fn is_mcp_tool_call_rejects_unknown_tool() {
    let tool_map = sample_tool_map();
    let tc = json!({"name": "my_function", "call_id": "call_2"});
    assert!(!is_mcp_tool_call(&tc, &tool_map));
}

#[test]
fn is_mcp_tool_call_rejects_missing_name() {
    let tool_map = sample_tool_map();
    let tc = json!({"call_id": "call_3"});
    assert!(!is_mcp_tool_call(&tc, &tool_map));
}

#[test]
fn extract_mcp_tool_calls_filters_correctly() {
    let tool_map = sample_tool_map();
    let tool_calls = vec![
        json!({"name": "get_weather", "call_id": "call_1"}),
        json!({"name": "my_function", "call_id": "call_2"}),
        json!({"name": "search_docs", "call_id": "call_3"}),
    ];
    let mcp_calls = extract_mcp_tool_calls(&tool_calls, &tool_map);
    assert_eq!(mcp_calls.len(), 2, "should extract only MCP tool calls");
    assert_eq!(mcp_calls[0]["name"], "get_weather");
    assert_eq!(mcp_calls[1]["name"], "search_docs");
}

#[test]
fn extract_mcp_tool_calls_empty_when_no_match() {
    let tool_map = sample_tool_map();
    let tool_calls = vec![json!({"name": "my_function", "call_id": "call_1"})];
    let mcp_calls = extract_mcp_tool_calls(&tool_calls, &tool_map);
    assert!(mcp_calls.is_empty());
}

// =========================================================================
// Approval Pre-check
// =========================================================================

#[test]
fn find_approval_required_returns_none_when_all_never() {
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    let calls = vec![
        json!({"name": "get_weather", "call_id": "call_1"}),
        json!({"name": "search_docs", "call_id": "call_2"}),
    ];
    assert!(find_approval_required(&calls, &tool_map).is_none());
}

#[test]
fn find_approval_required_returns_first_when_absent() {
    let tool_map = sample_tool_map();
    let calls = vec![
        json!({"name": "get_weather", "call_id": "call_1"}),
        json!({"name": "search_docs", "call_id": "call_2"}),
    ];
    let pending = find_approval_required(&calls, &tool_map).unwrap();
    assert_eq!(pending.tool_name, "get_weather");
}

#[test]
fn find_approval_required_returns_first_requiring() {
    let mut tool_map = sample_tool_map();
    tool_map
        .get_mut(&("weather".to_owned(), "get_weather".to_owned()))
        .unwrap()["require_approval"] = json!("never");
    tool_map
        .get_mut(&("docs".to_owned(), "search_docs".to_owned()))
        .unwrap()["require_approval"] = json!("always");

    let calls = vec![
        json!({"name": "get_weather", "call_id": "call_1"}),
        json!({"name": "search_docs", "call_id": "call_2", "arguments": {"query": "rust"}}),
    ];
    let pending = find_approval_required(&calls, &tool_map).unwrap();
    assert_eq!(pending.tool_name, "search_docs");
    assert_eq!(pending.call_id, "call_2");
    assert_eq!(pending.server_label, "docs");
}

#[test]
fn find_approval_required_defaults_to_approval_when_absent() {
    let tool_map = sample_tool_map();
    let calls = vec![json!({"name": "get_weather", "call_id": "call_1"})];
    assert!(
        find_approval_required(&calls, &tool_map).is_some(),
        "absent require_approval should default to requiring approval"
    );
}

#[test]
fn find_approval_required_ambiguous_tool_requires_approval() {
    let mut tool_map = sample_tool_map();
    tool_map.insert(
        ("other_weather".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "other_weather",
            "server_url": "http://other.example.com/mcp",
            "headers": null,
            "authorization": null,
            "tool_definition": {"name": "get_weather"},
            "require_approval": "never",
        }),
    );
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    let calls = vec![json!({"name": "get_weather", "call_id": "call_1"})];
    let pending = find_approval_required(&calls, &tool_map);
    assert!(
        pending.is_some(),
        "ambiguous tool name should require approval even when all servers say never"
    );
    let pending = pending.unwrap();
    assert_eq!(pending.tool_name, "get_weather");
    assert_eq!(pending.server_label, "unknown");
}

// =========================================================================
// Result Construction
// =========================================================================

#[test]
fn build_success_result_message_format() {
    let result = build_success_result("call_1", "weather", "get_weather", "{}", "Sunny, 22°C", false);

    assert_eq!(result.message["type"], "function_call_output");
    assert_eq!(result.message["call_id"], "call_1");
    assert_eq!(result.message["output"], "Sunny, 22°C");
    assert!(
        result.message.get("is_error").is_none(),
        "should not have is_error field"
    );
}

#[test]
fn build_success_result_with_tool_error() {
    let result = build_success_result("call_1", "weather", "get_weather", "{}", "Not found", true);

    assert_eq!(result.message["type"], "function_call_output");
    assert_eq!(result.message["output"], "Error: Not found");
    assert!(result.output_item["approval_request_id"].is_null());
    assert_eq!(
        result.output_item["error"], "Not found",
        "error field should contain the error text"
    );
}

#[test]
fn build_success_result_output_item_format() {
    let result = build_success_result(
        "call_1",
        "weather",
        "get_weather",
        "{\"city\":\"Paris\"}",
        "result",
        false,
    );

    assert_eq!(result.output_item["type"], "mcp_call");
    assert_eq!(result.output_item["id"], "call_1");
    assert!(result.output_item["approval_request_id"].is_null());
    assert_eq!(result.output_item["server_label"], "weather");
    assert_eq!(result.output_item["name"], "get_weather");
    assert_eq!(result.output_item["output"], "result");
    assert!(
        result.output_item.get("error").is_none(),
        "should not have error field on success"
    );
}

#[test]
fn build_error_result_includes_error_message() {
    let result = build_error_result("call_1", "weather", "get_weather", "{}", "connection refused");

    assert_eq!(result.message["type"], "function_call_output");
    assert_eq!(result.message["output"], "Error: connection refused");
    assert!(
        result.message.get("is_error").is_none(),
        "should not have is_error field"
    );
    assert!(result.output_item["approval_request_id"].is_null());
    assert_eq!(result.output_item["output"], "");
    assert_eq!(result.output_item["error"], "connection refused");
}

// =========================================================================
// Arguments Parsing
// =========================================================================

#[test]
fn arguments_string_is_parsed_to_object() {
    // Verify that JSON string arguments can be parsed
    let args_str = r#"{"city": "Paris"}"#;
    let parsed: serde_json::Value = serde_json::from_str(args_str).unwrap();
    assert!(parsed.is_object());
    assert_eq!(parsed["city"], "Paris");
}

// =========================================================================
// Config
// =========================================================================

#[test]
fn config_defaults() {
    let yaml = serde_yaml::from_str::<McpDispatchConfig>("{}").unwrap();
    assert_eq!(yaml.timeout_ms, 30_000);
    assert_eq!(yaml.max_body_bytes, praxis_filter::body::DEFAULT_JSON_BODY_MAX_BYTES);
}

#[test]
fn config_custom_timeout() {
    let yaml = serde_yaml::from_str::<McpDispatchConfig>("timeout_ms: 60000").unwrap();
    assert_eq!(yaml.timeout_ms, 60_000);
}

#[test]
fn config_rejects_unknown_fields() {
    let result = serde_yaml::from_str::<McpDispatchConfig>("unknown_field: true");
    assert!(result.is_err(), "should reject unknown fields");
}

#[test]
fn config_rejects_zero_timeout() {
    let cfg = serde_yaml::from_str::<McpDispatchConfig>("timeout_ms: 0").unwrap();
    let result = build_config(cfg);
    assert!(result.is_err(), "timeout_ms: 0 should be rejected");
}

// =========================================================================
// Content Block Conversion
// =========================================================================

#[test]
fn content_blocks_to_text_extracts_text() {
    let blocks = vec![rmcp::model::ContentBlock::text("hello world")];
    let text = content_blocks_to_text(&blocks);
    assert_eq!(text, "hello world");
}

#[test]
fn content_blocks_to_text_joins_multiple() {
    let blocks = vec![
        rmcp::model::ContentBlock::text("line 1"),
        rmcp::model::ContentBlock::text("line 2"),
    ];
    let text = content_blocks_to_text(&blocks);
    assert_eq!(text, "line 1\nline 2");
}

#[test]
fn content_blocks_to_text_skips_non_text() {
    let blocks = vec![
        rmcp::model::ContentBlock::text("text content"),
        rmcp::model::ContentBlock::image("base64data", "image/png"),
        rmcp::model::ContentBlock::resource(rmcp::model::ResourceContents::TextResourceContents {
            uri: "file://test".to_owned(),
            mime_type: None,
            text: "resource".to_owned(),
            meta: None,
        }),
    ];
    let text = content_blocks_to_text(&blocks);
    assert_eq!(text, "text content", "should skip non-text content types");
}

// =========================================================================
// resolve_tool_entry
// =========================================================================

#[test]
fn resolve_tool_entry_returns_entry_for_unique_tool() {
    let map = sample_tool_map();
    let entry = resolve_tool_entry(&map, "get_weather", "call_1").unwrap();
    assert_eq!(entry.get("server_label").unwrap(), "weather");
}

#[test]
fn resolve_tool_entry_returns_none_for_unknown_tool() {
    let map = sample_tool_map();
    let result = resolve_tool_entry(&map, "nonexistent", "call_1");
    assert!(matches!(result, Err(None)), "unknown tool should return Err(None)");
}

#[test]
fn resolve_tool_entry_returns_error_for_ambiguous_tool() {
    let mut map = sample_tool_map();
    map.insert(
        ("other-server".to_owned(), "get_weather".to_owned()),
        serde_json::json!({"server_label": "other", "server_url": "http://other/mcp"}),
    );
    let result = resolve_tool_entry(&map, "get_weather", "call_1");
    let err = result.unwrap_err().expect("should return error result for ambiguity");
    assert!(
        err.output_item["error"].as_str().unwrap().contains("ambiguous"),
        "error should mention ambiguity"
    );
}

// =========================================================================
// parse_call_arguments
// =========================================================================

#[test]
fn parse_call_arguments_object_passthrough() {
    let tc = serde_json::json!({"name": "tool", "arguments": {"key": "value"}});
    let (args, args_str) = parse_call_arguments(&tc, "c1", "srv", "tool").unwrap();
    assert!(args.is_object());
    assert!(args_str.contains("key"));
}

#[test]
fn parse_call_arguments_string_parsed() {
    let tc = serde_json::json!({"name": "tool", "arguments": "{\"a\": 1}"});
    let (args, _) = parse_call_arguments(&tc, "c1", "srv", "tool").unwrap();
    assert_eq!(args["a"], 1);
}

#[test]
fn parse_call_arguments_malformed_string_returns_error() {
    let tc = serde_json::json!({"name": "tool", "arguments": "not-json"});
    let err = parse_call_arguments(&tc, "c1", "srv", "tool").unwrap_err();
    assert!(err.output_item["error"].as_str().unwrap().contains("malformed"));
}

#[test]
fn parse_call_arguments_absent_defaults_to_empty_object() {
    let tc = serde_json::json!({"name": "tool"});
    let (args, _) = parse_call_arguments(&tc, "c1", "srv", "tool").unwrap();
    assert!(args.is_object());
    assert!(args.as_object().unwrap().is_empty());
}

// =========================================================================
// process_call_result
// =========================================================================

#[test]
fn process_call_result_success() {
    let call_result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text("hello")]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}");
    assert_eq!(result.message["output"], "hello");
    assert_eq!(result.output_item["type"], "mcp_call");
    assert!(result.output_item.get("error").is_none() || result.output_item["error"].is_null());
}

#[test]
fn process_call_result_tool_error() {
    let mut call_result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text("oops")]);
    call_result.is_error = Some(true);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}");
    assert!(result.message["output"].as_str().unwrap().starts_with("Error:"));
    assert_eq!(result.output_item["error"], "oops");
}

#[test]
fn process_call_result_transport_error() {
    let err = crate::mcp_client::McpClientError::CallTool {
        url: "http://example.com/mcp".to_owned(),
        tool_name: "tool".to_owned(),
        source: Box::new(std::io::Error::other("transport error")),
    };
    let result = process_call_result(Err(err), "c1", "srv", "tool", "{}");
    assert!(result.message["output"].as_str().unwrap().contains("Error:"));
    assert!(
        result.output_item["error"]
            .as_str()
            .unwrap()
            .contains("tools/call failed")
    );
}

// =========================================================================
// extract_call_id / extract_arguments
// =========================================================================

#[test]
fn extract_call_id_from_call_id_field() {
    let tc = serde_json::json!({"call_id": "abc"});
    assert_eq!(extract_call_id(&tc), "abc");
}

#[test]
fn extract_call_id_from_id_field() {
    let tc = serde_json::json!({"id": "xyz"});
    assert_eq!(extract_call_id(&tc), "xyz");
}

#[test]
fn extract_call_id_defaults_to_unknown() {
    let tc = serde_json::json!({});
    assert_eq!(extract_call_id(&tc), "unknown");
}

#[test]
fn extract_arguments_present() {
    let tc = serde_json::json!({"arguments": {"a": 1}});
    let args = extract_arguments(&tc);
    assert!(args.contains("\"a\""));
}

#[test]
fn extract_arguments_absent() {
    let tc = serde_json::json!({});
    assert_eq!(extract_arguments(&tc), "");
}

// =========================================================================
// from_config
// =========================================================================

#[test]
fn from_config_minimal() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert_eq!(filter.name(), "openai_mcp_dispatch");
}

#[test]
fn from_config_with_all_fields() {
    let config =
        serde_yaml::from_str::<serde_yaml::Value>("timeout_ms: 5000\nmax_body_bytes: 1048576\nallow_loopback: true")
            .unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert_eq!(filter.name(), "openai_mcp_dispatch");
}

#[test]
fn from_config_rejects_zero_timeout() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("timeout_ms: 0").unwrap();
    assert!(McpDispatchFilter::from_config(&config).is_err());
}

// =========================================================================
// execute_single_call (async)
// =========================================================================

#[tokio::test]
async fn execute_single_call_missing_name_returns_none() {
    let map = sample_tool_map();
    let tc = json!({"call_id": "c1"});
    let timeout = std::time::Duration::from_millis(100);
    assert!(execute_single_call(&tc, &map, timeout, true).await.is_none());
}

#[tokio::test]
async fn execute_single_call_unknown_tool_returns_none() {
    let map = sample_tool_map();
    let tc = json!({"name": "nonexistent", "call_id": "c1"});
    let timeout = std::time::Duration::from_millis(100);
    assert!(execute_single_call(&tc, &map, timeout, true).await.is_none());
}

#[tokio::test]
async fn execute_single_call_ambiguous_returns_error() {
    let mut map = sample_tool_map();
    map.insert(
        ("other".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "other",
            "server_url": "http://other.example.com/mcp",
        }),
    );
    let tc = json!({"name": "get_weather", "call_id": "c1"});
    let timeout = std::time::Duration::from_millis(100);
    let result = execute_single_call(&tc, &map, timeout, true).await.unwrap();
    assert!(result.output_item["error"].as_str().unwrap().contains("ambiguous"));
}

#[tokio::test]
async fn execute_single_call_malformed_args_returns_error() {
    let map = sample_tool_map();
    let tc = json!({"name": "get_weather", "call_id": "c1", "arguments": "not-json"});
    let timeout = std::time::Duration::from_millis(100);
    let result = execute_single_call(&tc, &map, timeout, true).await.unwrap();
    assert!(result.output_item["error"].as_str().unwrap().contains("malformed"));
}

#[tokio::test]
async fn execute_single_call_connection_error() {
    let map = sample_tool_map();
    let tc = json!({"name": "get_weather", "call_id": "c1", "arguments": {"city": "Paris"}});
    let timeout = std::time::Duration::from_millis(200);
    let result = execute_single_call(&tc, &map, timeout, true).await.unwrap();
    assert!(
        result.message["output"].as_str().unwrap().starts_with("Error:"),
        "should report connection/timeout error"
    );
}

// =========================================================================
// execute_mcp_calls (async)
// =========================================================================

#[tokio::test]
async fn execute_mcp_calls_empty_input() {
    let map = sample_tool_map();
    let timeout = std::time::Duration::from_millis(100);
    let map = std::sync::Arc::new(map);
    let results = execute_mcp_calls(&[], &map, false, timeout, true).await;
    assert!(results.is_empty());
}

#[tokio::test]
async fn execute_mcp_calls_sequential() {
    let map = std::sync::Arc::new(sample_tool_map());
    let calls = vec![json!({"name": "get_weather", "call_id": "c1", "arguments": {}})];
    let timeout = std::time::Duration::from_millis(200);
    let results = execute_mcp_calls(&calls, &map, false, timeout, true).await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].output_item["type"], "mcp_call");
}

#[tokio::test]
async fn execute_mcp_calls_parallel() {
    let map = std::sync::Arc::new(sample_tool_map());
    let calls = vec![json!({"name": "get_weather", "call_id": "c1", "arguments": {}})];
    let timeout = std::time::Duration::from_millis(200);
    let results = execute_mcp_calls(&calls, &map, true, timeout, true).await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].output_item["type"], "mcp_call");
}

#[tokio::test]
async fn execute_mcp_calls_emits_error_for_unknown_tools() {
    let map = std::sync::Arc::new(sample_tool_map());
    let calls = vec![json!({"name": "nonexistent", "call_id": "c1"})];
    let timeout = std::time::Duration::from_millis(100);
    let results = execute_mcp_calls(&calls, &map, false, timeout, true).await;
    assert_eq!(results.len(), 1);
    assert!(results[0].output_item["error"].as_str().unwrap().contains("no result"));
}

#[tokio::test]
async fn execute_mcp_calls_emits_error_for_unknown_tool_without_call_id() {
    let map = std::sync::Arc::new(sample_tool_map());
    let calls = vec![json!({"name": "nonexistent"})];
    let timeout = std::time::Duration::from_millis(100);
    let results = execute_mcp_calls(&calls, &map, false, timeout, true).await;
    assert_eq!(results.len(), 1, "must emit error even without call_id");
    assert_eq!(results[0].output_item["id"], "unknown");
    assert!(results[0].output_item["error"].as_str().unwrap().contains("no result"));
}

// =========================================================================
// process_call_result: non-text content blocks
// =========================================================================

#[test]
fn process_call_result_empty_content_produces_empty_output() {
    let call_result = rmcp::model::CallToolResult::success(vec![]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}");
    assert_eq!(result.message["output"], "");
}

#[test]
fn process_call_result_multi_text_joins_with_newline() {
    let call_result = rmcp::model::CallToolResult::success(vec![
        rmcp::model::ContentBlock::text("hello"),
        rmcp::model::ContentBlock::text("world"),
    ]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}");
    assert_eq!(result.message["output"], "hello\nworld");
}

// =========================================================================
// on_response_body (HttpFilter trait)
// =========================================================================

fn make_dispatch_filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    McpDispatchFilter::from_config(&yaml).unwrap()
}

#[test]
fn on_response_body_not_end_of_stream_returns_release() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from("data"));
    let result = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(matches!(result, FilterAction::Release));
}

#[test]
fn on_response_body_no_state_returns_continue() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
}

#[test]
fn on_response_body_no_mcp_calls_returns_continue() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::default());
    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
}

#[test]
fn on_response_body_with_mcp_calls_sets_execute_metadata() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    let state = ResponsesState {
        mcp_tool_map: tool_map,
        tool_calls: vec![json!({"name": "get_weather", "call_id": "c1"})],
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
    assert_eq!(
        ctx.filter_metadata.get("openai_mcp_dispatch.action"),
        Some(&"execute_mcp".to_owned())
    );
}

#[test]
fn on_response_body_approval_required_sets_done_metadata() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "get_weather", "call_id": "c1"})],
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
    assert_eq!(
        ctx.filter_metadata.get("openai_mcp_dispatch.action"),
        Some(&"done".to_owned())
    );
}

// =========================================================================
// on_request (HttpFilter trait)
// =========================================================================

#[tokio::test]
async fn on_request_no_state_returns_continue() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let result = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(result, FilterAction::Continue));
}

#[tokio::test]
async fn on_request_no_mcp_calls_returns_continue() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState::default());
    let result = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(result, FilterAction::Continue));
}

#[tokio::test]
async fn on_request_executes_and_appends_results() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "get_weather", "call_id": "c1", "arguments": {}})],
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    let result = filter.on_request(&mut ctx).await.unwrap();
    assert!(matches!(result, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.messages.is_empty(), "should append result messages");
    assert!(!state.output_items().is_empty(), "should append output items");
    assert!(state.tool_calls.is_empty(), "should clear executed MCP tool calls");
}
