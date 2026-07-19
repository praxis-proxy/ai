// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the tool parse filter.

use bytes::Bytes;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, HttpFilter};

use super::ToolParseFilter;

// =============================================================================
// Config Parsing
// =============================================================================

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = ToolParseFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_tool_parse", "filter name");
}

#[test]
fn full_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 1048576").unwrap();
    let filter = ToolParseFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_tool_parse", "filter name");
}

#[test]
fn unknown_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("typo_field: true").unwrap();
    let result = ToolParseFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown field should be rejected");
}

#[test]
fn zero_max_body_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 0").unwrap();
    let result = ToolParseFilter::from_config(&yaml);
    assert!(result.is_err(), "max_body_bytes=0 should be rejected");
}

#[test]
fn oversized_max_body_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_body_bytes: 999999999999").unwrap();
    let result = ToolParseFilter::from_config(&yaml);
    assert!(result.is_err(), "oversized max_body_bytes should be rejected");
}

// =============================================================================
// Body Access
// =============================================================================

#[test]
fn body_access_is_read_only() {
    let filter = make_filter("{}");
    assert_eq!(
        filter.request_body_access(),
        BodyAccess::ReadOnly,
        "should use read-only body access"
    );
}

#[test]
fn body_mode_is_stream_buffer() {
    let filter = make_filter("{}");
    assert!(
        matches!(filter.request_body_mode(), BodyMode::StreamBuffer { .. }),
        "should use StreamBuffer body mode"
    );
}

// =============================================================================
// on_request_body: No Tools
// =============================================================================

#[tokio::test]
async fn no_tools_skips_metadata() {
    let (action, ctx) = run_filter("{}", r#"{"model":"gpt-4.1","input":"test"}"#).await;
    assert!(matches!(action, FilterAction::Continue), "no tools should continue");
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "no tools => no has_tools metadata"
    );
}

// =============================================================================
// on_request_body: Non-Responses Path
// =============================================================================

#[tokio::test]
async fn non_responses_path_skips_parsing() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/chat/completions");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(r#"{"tools":[{"type":"function","name":"calc"}]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-Responses path should continue without parsing"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "non-Responses path should not set tool metadata"
    );
}

#[tokio::test]
async fn anthropic_messages_path_skips_parsing() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/messages");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(r#"{"tools":[{"type":"function","name":"calc"}]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "Anthropic Messages path should continue without parsing"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "Anthropic Messages path should not set tool metadata"
    );
}

// =============================================================================
// on_request_body: Metadata
// =============================================================================

#[tokio::test]
async fn function_tools_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_tools")
            .map(String::as_str),
        Some("true"),
        "should set has_tools"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.function_count")
            .map(String::as_str),
        Some("1"),
        "should set function_count"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_web_search"),
        "should not set has_web_search"
    );
}

#[tokio::test]
async fn nameless_function_tool_still_sets_has_tools() {
    let body = r#"{"input":"test","tools":[{"type":"function"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_tools")
            .map(String::as_str),
        Some("true"),
        "nameless function tool should still set has_tools metadata"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.function_count")
            .map(String::as_str),
        Some("1"),
        "nameless function tool should still count the discriminator"
    );
}

#[tokio::test]
async fn web_search_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"web_search"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_web_search")
            .map(String::as_str),
        Some("true"),
        "should set has_web_search"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_tools")
            .map(String::as_str),
        Some("true"),
        "web_search counts as has_tools"
    );
}

#[tokio::test]
async fn web_search_context_size_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"web_search","search_context_size":"high"}]}"#;
    let (_action, ctx) = run_filter("{}", body).await;

    assert_eq!(
        ctx.filter_metadata
            .get("tool_parse.search_context_size")
            .map(String::as_str),
        Some("high"),
        "should set search_context_size from tool entry"
    );
}

#[tokio::test]
async fn web_search_no_context_size_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"web_search"}]}"#;
    let (_action, ctx) = run_filter("{}", body).await;

    assert!(
        !ctx.filter_metadata.contains_key("tool_parse.search_context_size"),
        "should not set search_context_size when absent"
    );
}

#[tokio::test]
async fn file_search_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_file_search")
            .map(String::as_str),
        Some("true"),
        "should set has_file_search"
    );
}

#[tokio::test]
async fn mcp_tool_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"mcp","server_label":"srv"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata.get("openai_tool_parse.has_mcp").map(String::as_str),
        Some("true"),
        "should set has_mcp"
    );
}

#[tokio::test]
async fn tool_choice_metadata() {
    let body = r#"{"input":"test","tool_choice":"required","tools":[{"type":"function","name":"f"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice")
            .map(String::as_str),
        Some("required"),
        "should set tool_choice"
    );
}

#[tokio::test]
async fn tool_choice_specific_metadata() {
    let body = r#"{"input":"test","tool_choice":{"type":"function","name":"calc"},"tools":[{"type":"function","name":"calc"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice")
            .map(String::as_str),
        Some("calc"),
        "specific tool_choice should use tool name"
    );
}

#[tokio::test]
async fn tool_choice_hosted_type_metadata() {
    let body = r#"{"input":"test","tool_choice":{"type":"file_search"},"tools":[{"type":"file_search"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice")
            .map(String::as_str),
        Some("file_search"),
        "hosted tool_choice should use tool type"
    );
}

#[tokio::test]
async fn tool_choice_allowed_tools_metadata_uses_mode() {
    let body = r#"{
        "input":"test",
        "tool_choice":{
            "type":"allowed_tools",
            "mode":"required",
            "tools":[{"type":"function","name":"calc"}]
        },
        "tools":[{"type":"function","name":"calc"}]
    }"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice")
            .map(String::as_str),
        Some("required"),
        "allowed_tools tool_choice should promote mode"
    );
}

#[tokio::test]
async fn tool_choice_mcp_metadata() {
    let body = r#"{
        "input":"test",
        "tool_choice":{"type":"mcp","server_label":"deepwiki","name":"search"},
        "tools":[{"type":"mcp","server_label":"deepwiki","server_url":"http://localhost:8001/mcp"}]
    }"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice")
            .map(String::as_str),
        Some("mcp"),
        "MCP tool_choice should promote as 'mcp'"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice_type")
            .map(String::as_str),
        Some("mcp"),
        "MCP tool_choice_type should be 'mcp'"
    );
}

#[tokio::test]
async fn tool_choice_type_metadata_for_function() {
    let body = r#"{"input":"test","tool_choice":{"type":"function","name":"calc"},"tools":[{"type":"function","name":"calc"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.tool_choice_type")
            .map(String::as_str),
        Some("function"),
        "function tool_choice_type should be 'function'"
    );
}

#[tokio::test]
async fn tool_choice_type_metadata_absent_for_string_choices() {
    let body = r#"{"input":"test","tool_choice":"required","tools":[{"type":"function","name":"f"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice_type"),
        "string tool_choice should not set tool_choice_type"
    );
}

#[tokio::test]
async fn tool_choice_mcp_filter_results() {
    let body = r#"{
        "input":"test",
        "tool_choice":{"type":"mcp","server_label":"deepwiki","name":"search"},
        "tools":[{"type":"mcp","server_label":"deepwiki","server_url":"http://localhost:8001/mcp"}]
    }"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(
        results.matches("tool_choice", "mcp"),
        "MCP tool_choice filter result should be 'mcp'"
    );
    assert!(
        results.matches("tool_choice_type", "mcp"),
        "MCP tool_choice_type filter result should be 'mcp'"
    );
}

#[tokio::test]
async fn tool_choice_without_tools_not_promoted() {
    let body = r#"{"input":"test","tool_choice":"none"}"#;
    let (action, ctx) = run_filter("{}", body).await;

    assert!(matches!(action, FilterAction::Continue), "no tools should continue");
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice"),
        "tool_choice without tools should not be promoted"
    );
}

#[tokio::test]
async fn oversized_tool_choice_not_promoted() {
    let long_name = "x".repeat(300);
    let body = format!(r#"{{"input":"test","tool_choice":"{long_name}","tools":[{{"type":"function","name":"f"}}]}}"#);
    let (action, ctx) = run_filter("{}", &body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice"),
        "oversized tool_choice should not be promoted to metadata"
    );
}

#[tokio::test]
async fn unsafe_tool_choice_not_promoted() {
    let body = r#"{"input":"test","tool_choice":"bad\nvalue","tools":[{"type":"function","name":"f"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice"),
        "unsafe tool_choice should not be promoted to metadata"
    );
    assert!(
        results.get("tool_choice").is_none(),
        "unsafe tool_choice should not be promoted to filter results"
    );
    assert!(results.matches("has_tools", "true"), "safe facts still promote");
}

#[tokio::test]
async fn oversized_object_tool_choice_type_not_promoted() {
    let long_type = "x".repeat(300);
    let body = format!(
        r#"{{"input":"test","tool_choice":{{"type":"{long_type}"}},"tools":[{{"type":"function","name":"f"}}]}}"#
    );
    let (action, ctx) = run_filter("{}", &body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice_type"),
        "oversized tool_choice.type should not be promoted to metadata"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice"),
        "oversized hosted tool_choice value should not be promoted"
    );
}

#[tokio::test]
async fn control_char_object_tool_choice_type_not_promoted() {
    let body = r#"{"input":"test","tool_choice":{"type":"bad\ntype"},"tools":[{"type":"function","name":"f"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice_type"),
        "control-char tool_choice.type should not be promoted to metadata"
    );
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.tool_choice"),
        "control-char hosted tool_choice value should not be promoted"
    );
    assert!(
        ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "safe facts still promote"
    );
}

// =============================================================================
// on_request_body: Filter Results
// =============================================================================

#[tokio::test]
async fn filter_results_for_function_tools() {
    let body = r#"{"input":"test","tools":[{"type":"function","name":"calc"}],"tool_choice":"required"}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(results.matches("has_tools", "true"), "has_tools result");
    assert!(results.matches("function_count", "1"), "function_count result");
    assert!(results.matches("tool_choice", "required"), "tool_choice result");
}

#[tokio::test]
async fn filter_results_for_web_search() {
    let body = r#"{"input":"test","tools":[{"type":"web_search"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(results.matches("has_tools", "true"), "has_tools result");
    assert!(results.matches("has_web_search", "true"), "has_web_search result");
}

#[tokio::test]
async fn filter_results_for_mcp() {
    let body = r#"{"input":"test","tools":[{"type":"mcp","server_label":"s"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(results.matches("has_mcp", "true"), "has_mcp result");
}

// =============================================================================
// on_request_body: Hosted Tool Metadata
// =============================================================================

#[tokio::test]
async fn code_interpreter_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"code_interpreter"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_code_interpreter")
            .map(String::as_str),
        Some("true"),
        "should set has_code_interpreter"
    );
    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_tools")
            .map(String::as_str),
        Some("true"),
        "code_interpreter counts as has_tools"
    );
}

#[tokio::test]
async fn computer_use_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"computer_use"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_computer_use")
            .map(String::as_str),
        Some("true"),
        "should set has_computer_use"
    );
}

#[tokio::test]
async fn image_generation_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"image_generation"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_image_generation")
            .map(String::as_str),
        Some("true"),
        "should set has_image_generation"
    );
}

#[tokio::test]
async fn tool_search_metadata() {
    let body = r#"{"input":"test","tools":[{"type":"tool_search"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(
        ctx.filter_metadata
            .get("openai_tool_parse.has_tool_search")
            .map(String::as_str),
        Some("true"),
        "should set has_tool_search"
    );
}

#[tokio::test]
async fn code_interpreter_filter_results() {
    let body = r#"{"input":"test","tools":[{"type":"code_interpreter"}]}"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");
    let results = &ctx.filter_results["openai_tool_parse"];

    assert!(
        results.matches("has_code_interpreter", "true"),
        "has_code_interpreter filter result"
    );
}

// =============================================================================
// on_request_body: Edge Cases
// =============================================================================

#[tokio::test]
async fn empty_body_produces_no_metadata() {
    let (action, ctx) = run_filter("{}", "").await;
    assert!(matches!(action, FilterAction::Continue), "empty body should continue");
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "empty body should produce no tool metadata"
    );
}

#[tokio::test]
async fn invalid_json_produces_no_metadata() {
    let (action, ctx) = run_filter("{}", "not json").await;
    assert!(matches!(action, FilterAction::Continue), "invalid JSON should continue");
    assert!(
        !ctx.filter_metadata.contains_key("openai_tool_parse.has_tools"),
        "invalid JSON should produce no tool metadata"
    );
}

#[tokio::test]
async fn not_end_of_stream_continues() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = Some(Bytes::from(r#"{"tools":[]}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, false).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "non-EOS should continue");
}

#[tokio::test]
async fn mixed_tools_all_metadata_set() {
    let body = r#"{
        "input": "test",
        "tools": [
            {"type": "function", "name": "f1"},
            {"type": "function", "name": "f2"},
            {"type": "web_search"},
            {"type": "file_search", "vector_store_ids": ["vs"]},
            {"type": "code_interpreter"},
            {"type": "mcp", "server_label": "s"}
        ],
        "tool_choice": "auto"
    }"#;
    let (action, ctx) = run_filter("{}", body).await;
    assert!(matches!(action, FilterAction::Release), "has tools should release");

    assert_eq!(ctx.filter_metadata["openai_tool_parse.has_tools"], "true", "has_tools");
    assert_eq!(
        ctx.filter_metadata["openai_tool_parse.function_count"], "2",
        "function_count"
    );
    assert_eq!(
        ctx.filter_metadata["openai_tool_parse.has_web_search"], "true",
        "has_web_search"
    );
    assert_eq!(
        ctx.filter_metadata["openai_tool_parse.has_file_search"], "true",
        "has_file_search"
    );
    assert_eq!(
        ctx.filter_metadata["openai_tool_parse.has_code_interpreter"], "true",
        "has_code_interpreter"
    );
    assert_eq!(ctx.filter_metadata["openai_tool_parse.has_mcp"], "true", "has_mcp");
    assert_eq!(
        ctx.filter_metadata["openai_tool_parse.tool_choice"], "auto",
        "tool_choice"
    );
}

// =============================================================================
// on_request: Restore Filter Results from Metadata
// =============================================================================

#[tokio::test]
async fn restore_presence_flags_from_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    ctx.set_metadata("openai_tool_parse.has_tools", "true");
    ctx.set_metadata("openai_tool_parse.has_web_search", "true");
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    drop(filter.on_request(&mut ctx).await.unwrap());

    let results = &ctx.filter_results["openai_tool_parse"];
    assert!(results.matches("has_tools", "true"), "has_tools restored");
    assert!(results.matches("has_web_search", "true"), "has_web_search restored");
    assert!(results.matches("has_mcp", "true"), "has_mcp restored");
}

#[tokio::test]
async fn restore_presence_flags_absent_when_no_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(
        !ctx.filter_results.contains_key("openai_tool_parse"),
        "no metadata => no filter_results"
    );
}

#[tokio::test]
async fn restore_function_count_from_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    ctx.set_metadata("openai_tool_parse.function_count", "3");

    drop(filter.on_request(&mut ctx).await.unwrap());

    let results = &ctx.filter_results["openai_tool_parse"];
    assert!(results.matches("function_count", "3"), "function_count restored");
}

#[tokio::test]
async fn restore_function_count_absent_when_no_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(
        ctx.filter_results
            .get("openai_tool_parse")
            .and_then(|r| r.get("function_count"))
            .is_none(),
        "no metadata => no function_count result"
    );
}

#[tokio::test]
async fn restore_tool_choice_from_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    ctx.set_metadata("openai_tool_parse.tool_choice", "required");
    ctx.set_metadata("openai_tool_parse.tool_choice_type", "function");

    drop(filter.on_request(&mut ctx).await.unwrap());

    let results = &ctx.filter_results["openai_tool_parse"];
    assert!(results.matches("tool_choice", "required"), "tool_choice restored");
    assert!(
        results.matches("tool_choice_type", "function"),
        "tool_choice_type restored"
    );
}

#[tokio::test]
async fn restore_tool_choice_absent_when_no_metadata() {
    let filter = make_filter("{}");
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);

    drop(filter.on_request(&mut ctx).await.unwrap());

    assert!(
        ctx.filter_results
            .get("openai_tool_parse")
            .and_then(|r| r.get("tool_choice"))
            .is_none(),
        "no metadata => no tool_choice result"
    );
}

// =============================================================================
// Test Utilities
// =============================================================================

/// Run the filter's `on_request_body` and return the action and context.
async fn run_filter(config_yaml: &str, body_str: &str) -> (FilterAction, praxis_filter::HttpFilterContext<'static>) {
    let filter = make_filter(config_yaml);
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let req: &'static praxis_filter::Request = Box::leak(Box::new(req));
    let mut ctx = crate::test_utils::make_filter_context(req);
    let mut body = if body_str.is_empty() {
        None
    } else {
        Some(Bytes::from(body_str.to_owned()))
    };

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    (action, ctx)
}

/// Build a `ToolParseFilter` from a YAML snippet.
fn make_filter(yaml_str: &str) -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml_str).unwrap();
    ToolParseFilter::from_config(&yaml).unwrap()
}
