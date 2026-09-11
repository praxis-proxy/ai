// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_mcp_dispatch` filter.

use std::{collections::HashMap, sync::Arc};

use bytes::Bytes;
use praxis_filter::FilterAction;
use serde_json::json;

use super::{
    McpDispatchFilter, McpExecutionOptions, admitted_result_limits, build_error_result, build_success_result,
    content_blocks_to_output, execute_mcp_calls, execute_single_call, extract_arguments, extract_call_id,
    extract_mcp_tool_calls, find_by_encoded_name, is_mcp_tool_call, mcp_call_ids_are_unique_and_new,
    normalize_arguments, parse_call_arguments, partition_calls_by_approval, process_call_result,
    push_result_within_budget, resolve_tool_entry, result_payload_limit,
};
use crate::{
    openai::responses::{
        DEFAULT_TENANT_ID,
        mcp_dispatch::{
            approval::{
                ApprovalError, ApprovalPolicy, ResolvedApproval, build_approved_tool_call, build_denial_message,
                extract_approval_responses, is_approval_response, parse_approval_policy, parse_approval_response,
                requires_approval, resolve_approval, target_fingerprint,
            },
            config::{McpDispatchConfig, build_config},
        },
        mcp_tool_resolve::{McpToolIndex, encode_function_name},
        state::{McpApprovalState, ResponsesState},
    },
    store::{PendingApprovalRecord, ResponseStore, ResponseStoreRegistry, SqliteResponseStore},
    test_utils::{make_filter_context, make_request},
};

/// Borrow owned test tool calls the way the filter passes them:
/// the dispatch and approval paths take calls by reference.
fn call_refs(calls: &[serde_json::Value]) -> Vec<&serde_json::Value> {
    calls.iter().collect()
}

const TEST_MAX_RESULT_BYTES: usize = 1_048_576;
const TEST_MAX_TOTAL_RESULT_BYTES: usize = 8_388_608;

#[test]
fn rejected_calls_do_not_dilute_admitted_result_allowance() {
    let (per_result, execution_batch) = admitted_result_limits(1, 31, 1_048_576, 8_388_608).unwrap();

    assert_eq!(per_result, 1_048_576);
    assert_eq!(execution_batch, 8_388_608 - 31 * 1_024);
}

#[test]
fn mcp_call_ids_must_be_present_nonempty_and_unique() {
    let distinct = vec![json!({"call_id": "call_1"}), json!({"call_id": "call_2"})];
    assert!(mcp_call_ids_are_unique_and_new(&call_refs(&distinct), &[]));

    let duplicate = vec![json!({"call_id": "call_1"}), json!({"call_id": "call_1"})];
    assert!(!mcp_call_ids_are_unique_and_new(&call_refs(&duplicate), &[]));

    let missing = vec![json!({"id": "item_only"})];
    assert!(!mcp_call_ids_are_unique_and_new(&call_refs(&missing), &[]));

    let empty = vec![json!({"call_id": ""})];
    assert!(!mcp_call_ids_are_unique_and_new(&call_refs(&empty), &[]));

    let reused = vec![json!({"call_id": "call_1"})];
    assert!(!mcp_call_ids_are_unique_and_new(
        &call_refs(&reused),
        &[json!({"type": "mcp_call", "id": "call_1", "output": "prior result"})]
    ));
}

fn execution_options(parallel: bool, timeout: std::time::Duration) -> McpExecutionOptions {
    McpExecutionOptions {
        parallel,
        max_parallel_calls: 8,
        max_result_bytes: TEST_MAX_RESULT_BYTES,
        max_total_result_bytes: TEST_MAX_TOTAL_RESULT_BYTES,
        timeout,
        allow_loopback: true,
    }
}

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
    assert_eq!(filter.response_body_access(), praxis_filter::BodyAccess::ReadWrite);
}

#[test]
fn filter_request_body_access() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert_eq!(filter.request_body_access(), praxis_filter::BodyAccess::ReadOnly);
}

#[test]
fn filter_response_body_mode() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert!(
        matches!(filter.response_body_mode(), praxis_filter::BodyMode::Stream),
        "agentic responses must remain stream-compatible"
    );
}

#[test]
fn filter_request_body_mode() {
    let config = serde_yaml::from_str::<serde_yaml::Value>("{}").unwrap();
    let filter = McpDispatchFilter::from_config(&config).unwrap();
    assert!(
        matches!(
            filter.request_body_mode(),
            praxis_filter::BodyMode::StreamBuffer {
                max_bytes: Some(praxis_filter::body::MAX_JSON_BODY_BYTES)
            }
        ),
        "should buffer up to the absolute ceiling; body_limits governs the raw cap"
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

fn lossy_collision_tool_map() -> HashMap<(String, String), serde_json::Value> {
    let mut map = HashMap::new();
    map.insert(
        ("my.server".to_owned(), "get".to_owned()),
        json!({
            "server_label": "my.server",
            "server_url": "http://a.example.com/mcp",
            "headers": null, "authorization": null,
            "tool_definition": {"name": "get"},
            "require_approval": "never",
        }),
    );
    map.insert(
        ("my_server".to_owned(), "get".to_owned()),
        json!({
            "server_label": "my_server",
            "server_url": "http://b.example.com/mcp",
            "headers": null, "authorization": null,
            "tool_definition": {"name": "get"},
            "require_approval": "never",
        }),
    );
    map
}

#[test]
fn is_mcp_tool_call_matches_known_tool() {
    let tool_map = sample_tool_map();
    let tc = json!({"name": "weather__get_weather", "call_id": "call_1"});
    assert!(is_mcp_tool_call(&tc, &McpToolIndex::new(&tool_map)));
}

#[test]
fn is_mcp_tool_call_rejects_raw_tool_name() {
    let tool_map = sample_tool_map();
    let tc = json!({"name": "get_weather", "call_id": "call_1"});
    assert!(
        !is_mcp_tool_call(&tc, &McpToolIndex::new(&tool_map)),
        "raw tool name should not match; inference returns encoded names"
    );
}

#[test]
fn is_mcp_tool_call_rejects_unknown_tool() {
    let tool_map = sample_tool_map();
    let tc = json!({"name": "my_function", "call_id": "call_2"});
    assert!(!is_mcp_tool_call(&tc, &McpToolIndex::new(&tool_map)));
}

#[test]
fn is_mcp_tool_call_rejects_missing_name() {
    let tool_map = sample_tool_map();
    let tc = json!({"call_id": "call_3"});
    assert!(!is_mcp_tool_call(&tc, &McpToolIndex::new(&tool_map)));
}

#[test]
fn extract_mcp_tool_calls_filters_correctly() {
    let tool_map = sample_tool_map();
    let tool_calls = vec![
        json!({"name": "weather__get_weather", "call_id": "call_1"}),
        json!({"name": "my_function", "call_id": "call_2"}),
        json!({"name": "docs__search_docs", "call_id": "call_3"}),
    ];
    let mcp_calls = extract_mcp_tool_calls(&tool_calls, &McpToolIndex::new(&tool_map));
    assert_eq!(mcp_calls.len(), 2, "should extract only MCP tool calls");
    assert_eq!(mcp_calls[0]["name"], "weather__get_weather");
    assert_eq!(mcp_calls[1]["name"], "docs__search_docs");
}

// =========================================================================
// find_by_encoded_name
// =========================================================================

#[test]
fn find_by_encoded_name_matches_via_encoding() {
    let map = sample_tool_map();
    let result = find_by_encoded_name(&McpToolIndex::new(&map), "weather__get_weather");
    assert!(result.is_some(), "should find entry by encoded name");
    let (key, entry) = result.unwrap();
    assert_eq!(key.0, "weather", "key should have original label");
    assert_eq!(key.1, "get_weather", "key should have original tool name");
    assert_eq!(entry["server_label"], "weather");
}

#[test]
fn find_by_encoded_name_rejects_raw_name() {
    let map = sample_tool_map();
    assert!(
        find_by_encoded_name(&McpToolIndex::new(&map), "get_weather").is_none(),
        "raw tool name should not match; lookup is by encoded name"
    );
}

#[test]
fn extract_mcp_tool_calls_empty_when_no_match() {
    let tool_map = sample_tool_map();
    let tool_calls = vec![json!({"name": "my_function", "call_id": "call_1"})];
    let mcp_calls = extract_mcp_tool_calls(&tool_calls, &McpToolIndex::new(&tool_map));
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
        json!({"name": "weather__get_weather", "call_id": "call_1"}),
        json!({"name": "docs__search_docs", "call_id": "call_2"}),
    ];
    let (pending, ungated) = partition_calls_by_approval(call_refs(&calls), &tool_map);
    assert!(pending.is_empty());
    assert_eq!(ungated.len(), 2);
}

#[test]
fn find_approval_required_returns_first_when_absent() {
    let tool_map = sample_tool_map();
    let calls = vec![
        json!({"name": "weather__get_weather", "call_id": "call_1"}),
        json!({"name": "docs__search_docs", "call_id": "call_2"}),
    ];
    let (pending, ungated) = partition_calls_by_approval(call_refs(&calls), &tool_map);
    assert_eq!(pending.len(), 2);
    assert!(ungated.is_empty());
    assert_eq!(pending[0].tool_name, "get_weather");
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
        json!({"name": "weather__get_weather", "call_id": "call_1"}),
        json!({"name": "docs__search_docs", "call_id": "call_2", "arguments": {"query": "rust"}}),
    ];
    let (pending, ungated) = partition_calls_by_approval(call_refs(&calls), &tool_map);
    assert_eq!(pending.len(), 1);
    assert_eq!(ungated.len(), 1);
    assert_eq!(pending[0].tool_name, "search_docs");
    assert_eq!(pending[0].call_id, "call_2");
    assert_eq!(pending[0].server_label, "docs");
}

#[test]
fn find_approval_required_defaults_to_approval_when_absent() {
    let tool_map = sample_tool_map();
    let calls = vec![json!({"name": "weather__get_weather", "call_id": "call_1"})];
    assert!(
        !partition_calls_by_approval(call_refs(&calls), &tool_map).0.is_empty(),
        "absent require_approval should default to requiring approval"
    );
}

#[test]
fn find_approval_required_ambiguous_tool_requires_approval() {
    let tool_map = lossy_collision_tool_map();
    let calls = vec![json!({"name": "my_server__get", "call_id": "call_1"})];
    let (pending, ungated) = partition_calls_by_approval(call_refs(&calls), &tool_map);
    assert!(
        !pending.is_empty(),
        "ambiguous encoded name should require approval even when all servers say never"
    );
    assert!(ungated.is_empty());
    assert_eq!(pending[0].tool_name, "my_server__get");
    assert_eq!(pending[0].server_label, "unknown");
}

// =========================================================================
// Result Construction
// =========================================================================

#[test]
fn build_success_result_message_format() {
    let result = build_success_result("call_1", "weather", "get_weather", "{}", "Sunny, 22°C", false, None);

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
    let result = build_success_result("call_1", "weather", "get_weather", "{}", "Not found", true, None);

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
        None,
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
    let result = build_error_result("call_1", "weather", "get_weather", "{}", "connection refused", None);

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
    assert_eq!(yaml.max_calls_per_round, 32);
    assert_eq!(yaml.max_parallel_calls, 8);
    assert_eq!(yaml.max_result_bytes, TEST_MAX_RESULT_BYTES);
    assert_eq!(yaml.max_total_result_bytes, TEST_MAX_TOTAL_RESULT_BYTES);
}

#[test]
fn config_rejects_legacy_max_body_bytes() {
    // Raw body size is governed by body_limits, not per-filter. This
    // read-only dispatcher never produced a body, so the knob was removed
    // entirely and is now rejected as an unknown field.
    let result = serde_yaml::from_str::<McpDispatchConfig>("max_body_bytes: 1024");
    assert!(result.is_err(), "legacy max_body_bytes should be rejected");
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

#[test]
fn config_rejects_zero_call_limits() {
    let per_round = serde_yaml::from_str::<McpDispatchConfig>("max_calls_per_round: 0").unwrap();
    assert!(build_config(per_round).is_err());
    let parallel = serde_yaml::from_str::<McpDispatchConfig>("max_parallel_calls: 0").unwrap();
    assert!(build_config(parallel).is_err());
}

#[test]
fn config_accepts_absolute_call_limit_ceilings() {
    let cfg = serde_yaml::from_str::<McpDispatchConfig>("max_calls_per_round: 1024\nmax_parallel_calls: 64").unwrap();
    assert!(build_config(cfg).is_ok());
}

#[test]
fn config_rejects_call_limits_above_absolute_ceilings() {
    let per_round = serde_yaml::from_str::<McpDispatchConfig>("max_calls_per_round: 1025").unwrap();
    assert!(build_config(per_round).is_err());
    let parallel = serde_yaml::from_str::<McpDispatchConfig>("max_parallel_calls: 65").unwrap();
    assert!(build_config(parallel).is_err());
}

#[test]
fn config_validates_result_byte_limits_and_absolute_ceilings() {
    let valid =
        serde_yaml::from_str::<McpDispatchConfig>("max_result_bytes: 16777216\nmax_total_result_bytes: 67108864")
            .unwrap();
    assert!(build_config(valid).is_ok());

    for yaml in [
        "max_result_bytes: 0",
        "max_result_bytes: 1023",
        "max_result_bytes: 16777217\nmax_total_result_bytes: 16777217",
        "max_result_bytes: 1024\nmax_total_result_bytes: 512",
        "max_calls_per_round: 16\nmax_result_bytes: 1024\nmax_total_result_bytes: 15360",
        "max_total_result_bytes: 67108865",
    ] {
        assert!(
            build_config(serde_yaml::from_str(yaml).unwrap()).is_err(),
            "accepted: {yaml}"
        );
    }
}

// =========================================================================
// Content Block Conversion
// =========================================================================

#[test]
fn content_blocks_to_output_extracts_text() {
    let blocks = vec![rmcp::model::ContentBlock::text("hello world")];
    let text = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn content_blocks_to_output_joins_multiple_text() {
    let blocks = vec![
        rmcp::model::ContentBlock::text("line 1"),
        rmcp::model::ContentBlock::text("line 2"),
    ];
    let text = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();
    assert_eq!(text, "line 1\nline 2");
}

#[test]
fn content_blocks_to_output_empty_is_empty_string() {
    let text = content_blocks_to_output(&[], TEST_MAX_RESULT_BYTES).unwrap();
    assert_eq!(text, "", "empty content is genuinely empty, not data loss");
}

#[test]
fn content_blocks_to_output_rejects_oversized_text_before_joining() {
    let blocks = vec![
        rmcp::model::ContentBlock::text("1234"),
        rmcp::model::ContentBlock::text("5"),
    ];
    let error = content_blocks_to_output(&blocks, 5).unwrap_err();
    assert!(error.contains("per-result byte limit"));
}

#[test]
fn retained_result_batch_stops_at_aggregate_byte_limit() {
    let first = build_success_result("c1", "srv", "tool", "{}", "result", false, None);
    let second = build_success_result("c2", "srv", "tool", "{}", "result", false, None);
    let limit = first.retained_bytes().unwrap();
    let mut retained_bytes = 0;
    let mut results = Vec::new();

    push_result_within_budget(&mut results, &mut retained_bytes, first, limit, limit).unwrap();
    assert!(push_result_within_budget(&mut results, &mut retained_bytes, second, limit, limit).is_err());
    assert_eq!(
        results.len(),
        1,
        "the over-budget result must never enter retained state"
    );
}

#[test]
fn content_blocks_to_output_preserves_non_text_losslessly() {
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
    let output = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();

    let recovered: Vec<rmcp::model::ContentBlock> =
        serde_json::from_str(&output).expect("output must be valid JSON content array");
    assert_eq!(
        recovered, blocks,
        "#807: mixed text/non-text output must round-trip losslessly so no MCP content block is dropped"
    );
}

#[test]
fn content_blocks_to_output_preserves_audio_losslessly() {
    let blocks = vec![rmcp::model::ContentBlock::audio("base64audiodata", "audio/wav")];
    let output = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();

    assert!(
        output.contains("\"type\":\"audio\""),
        "audio block serialized as JSON: {output}"
    );
    assert!(output.contains("base64audiodata"), "audio data preserved: {output}");
    assert!(output.contains("audio/wav"), "audio mime type preserved: {output}");

    let recovered: Vec<rmcp::model::ContentBlock> =
        serde_json::from_str(&output).expect("output must be valid JSON content array");
    assert_eq!(
        recovered, blocks,
        "audio content must round-trip losslessly, not be dropped"
    );
}

#[test]
fn content_blocks_to_output_preserves_resource_link_losslessly() {
    let blocks = vec![rmcp::model::ContentBlock::resource_link(
        rmcp::model::Resource::new("file:///test.txt", "test.txt")
            .with_mime_type("text/plain")
            .with_size(100),
    )];
    let output = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();

    assert!(
        output.contains("\"type\":\"resource_link\""),
        "resource_link block serialized as JSON: {output}"
    );
    assert!(output.contains("file:///test.txt"), "resource uri preserved: {output}");
    assert!(output.contains("test.txt"), "resource name preserved: {output}");

    let recovered: Vec<rmcp::model::ContentBlock> =
        serde_json::from_str(&output).expect("output must be valid JSON content array");
    assert_eq!(
        recovered, blocks,
        "resource_link content must round-trip losslessly, not be dropped"
    );
}

#[test]
fn content_blocks_to_output_preserves_mixed_content_ordered() {
    // A single result carrying text, audio, an embedded resource, and a
    // resource link must serialize as the compact JSON content array and
    // deserialize back to the exact same ordered blocks, so no variant is
    // dropped or reordered when several non-text kinds appear together.
    let blocks = vec![
        rmcp::model::ContentBlock::text("summary line"),
        rmcp::model::ContentBlock::audio("base64audiodata", "audio/wav"),
        rmcp::model::ContentBlock::resource(rmcp::model::ResourceContents::TextResourceContents {
            uri: "file:///embedded.txt".to_owned(),
            mime_type: Some("text/plain".to_owned()),
            text: "embedded resource body".to_owned(),
            meta: None,
        }),
        rmcp::model::ContentBlock::resource_link(
            rmcp::model::Resource::new("file:///linked.png", "linked.png")
                .with_mime_type("image/png")
                .with_size(2048),
        ),
    ];
    let output = content_blocks_to_output(&blocks, TEST_MAX_RESULT_BYTES).unwrap();

    let recovered: Vec<rmcp::model::ContentBlock> =
        serde_json::from_str(&output).expect("output must be valid JSON content array");
    assert_eq!(
        recovered, blocks,
        "#1020: mixed text/audio/embedded-resource/resource-link content must \
         deserialize back to the original ordered array"
    );
}

#[test]
fn process_call_result_preserves_mixed_content_in_both_representations() {
    // The model-facing `function_call_output` and the client-facing
    // `mcp_call.output` are both fed from the same serializer, so a mixed
    // non-text result must reach both representations identically and
    // losslessly — neither may silently drop a variant.
    let blocks = vec![
        rmcp::model::ContentBlock::text("caption"),
        rmcp::model::ContentBlock::audio("base64audiodata", "audio/wav"),
        rmcp::model::ContentBlock::resource_link(
            rmcp::model::Resource::new("file:///doc.pdf", "doc.pdf").with_mime_type("application/pdf"),
        ),
    ];
    let call_result = rmcp::model::CallToolResult::success(blocks.clone());
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);

    let model_facing = result.message["output"]
        .as_str()
        .expect("function_call_output output is a string");
    let client_facing = result.output_item["output"]
        .as_str()
        .expect("mcp_call output is a string");
    assert_eq!(
        model_facing, client_facing,
        "model-facing and client-facing representations must carry identical content"
    );

    let recovered: Vec<rmcp::model::ContentBlock> =
        serde_json::from_str(client_facing).expect("client-facing output must be a JSON content array");
    assert_eq!(
        recovered, blocks,
        "both representations must round-trip the mixed content to the original ordered array"
    );
}

// =========================================================================
// resolve_tool_entry
// =========================================================================

#[test]
fn resolve_tool_entry_returns_entry_for_unique_tool() {
    let map = sample_tool_map();
    let (key, entry) = resolve_tool_entry(&McpToolIndex::new(&map), "weather__get_weather", "call_1", None).unwrap();
    assert_eq!(entry.get("server_label").unwrap(), "weather");
    assert_eq!(key.1, "get_weather", "key should contain the original tool name");
}

#[test]
fn resolve_tool_entry_returns_none_for_unknown_tool() {
    let map = sample_tool_map();
    let result = resolve_tool_entry(&McpToolIndex::new(&map), "nonexistent", "call_1", None);
    assert!(matches!(result, Err(None)), "unknown tool should return Err(None)");
}

#[test]
fn resolve_tool_entry_returns_error_for_ambiguous_tool() {
    let map = lossy_collision_tool_map();
    let result = resolve_tool_entry(&McpToolIndex::new(&map), "my_server__get", "call_1", None);
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
    let (args, args_str) = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap();
    assert!(args.is_object());
    assert!(args_str.contains("key"));
}

#[test]
fn parse_call_arguments_string_parsed() {
    let tc = serde_json::json!({"name": "tool", "arguments": "{\"a\": 1}"});
    let (args, _) = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap();
    assert_eq!(args["a"], 1);
}

#[test]
fn parse_call_arguments_malformed_string_returns_error() {
    let tc = serde_json::json!({"name": "tool", "arguments": "not-json"});
    let err = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap_err();
    assert!(err.output_item["error"].as_str().unwrap().contains("malformed"));
}

#[test]
fn parse_call_arguments_absent_defaults_to_empty_object() {
    let tc = serde_json::json!({"name": "tool"});
    let (args, args_str) = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap();
    assert!(args.is_object());
    assert!(args.as_object().unwrap().is_empty());
    assert_eq!(
        args_str, "{}",
        "absent arguments keep the canonical empty-object string"
    );
}

#[test]
fn parse_call_arguments_string_not_double_encoded() {
    let tc = serde_json::json!({"name": "tool", "arguments": "{\"a\": 1}"});
    let (_, args_str) = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap();
    assert_eq!(
        args_str, "{\"a\": 1}",
        "string arguments keep their original representation verbatim"
    );
}

#[test]
fn parse_call_arguments_malformed_string_error_keeps_raw_arguments() {
    let tc = serde_json::json!({"name": "tool", "arguments": "not-json"});
    let err = parse_call_arguments(&tc, "c1", "srv", "tool", None).unwrap_err();
    assert_eq!(
        err.output_item["arguments"], "not-json",
        "the malformed raw string must survive into the error body"
    );
}

// =========================================================================
// process_call_result
// =========================================================================

#[test]
fn process_call_result_success() {
    let call_result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text("hello")]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);
    assert_eq!(result.message["output"], "hello");
    assert_eq!(result.output_item["type"], "mcp_call");
    assert!(result.output_item.get("error").is_none() || result.output_item["error"].is_null());
}

#[test]
fn process_call_result_tool_error() {
    let mut call_result = rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text("oops")]);
    call_result.is_error = Some(true);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);
    assert!(result.message["output"].as_str().unwrap().starts_with("Error:"));
    assert_eq!(result.output_item["error"], "oops");
}

#[test]
fn process_call_result_transport_error() {
    let err = crate::mcp_client::McpClientError::CallTool {
        url: crate::mcp_client::McpDisplayUrl::from_uri(&"http://example.com/mcp".parse().unwrap()),
        tool_name: "tool".to_owned(),
    };
    let result = process_call_result(Err(err), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);
    assert!(result.message["output"].as_str().unwrap().contains("Error:"));
    assert!(
        result.output_item["error"]
            .as_str()
            .unwrap()
            .contains("tools/call failed")
    );
}

// =========================================================================
// normalize_arguments
// =========================================================================

#[test]
fn normalize_arguments_parses_json_string() {
    let raw = serde_json::json!("{\"city\":\"Paris\"}");
    let (parsed, canonical) = normalize_arguments(&raw).unwrap();
    assert_eq!(parsed["city"], "Paris");
    assert_eq!(canonical, "{\"city\":\"Paris\"}");
}

#[test]
fn normalize_arguments_object_passthrough() {
    let raw = serde_json::json!({"city": "Paris"});
    let (parsed, canonical) = normalize_arguments(&raw).unwrap();
    assert_eq!(parsed["city"], "Paris");
    assert!(canonical.contains("Paris"));
}

#[test]
fn normalize_arguments_malformed_string_returns_error() {
    let raw = serde_json::json!("not-json");
    assert!(normalize_arguments(&raw).is_err());
}

#[test]
fn normalize_arguments_empty_object_string() {
    let raw = serde_json::json!("{}");
    let (parsed, canonical) = normalize_arguments(&raw).unwrap();
    assert!(parsed.is_object());
    assert_eq!(canonical, "{}");
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

#[test]
fn extract_arguments_string_not_double_encoded() {
    let tc = serde_json::json!({"arguments": "{\"city\":\"Paris\"}"});
    let args = extract_arguments(&tc);
    assert_eq!(
        args, "{\"city\":\"Paris\"}",
        "string arguments must not be double-encoded"
    );
}

#[test]
fn extract_arguments_malformed_string_passes_through() {
    let tc = serde_json::json!({"arguments": "not-json"});
    let args = extract_arguments(&tc);
    assert_eq!(args, "not-json", "malformed string should pass through unchanged");
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
    let config = serde_yaml::from_str::<serde_yaml::Value>(
        "timeout_ms: 5000\nallow_loopback: true\nmax_calls_per_round: 16\nmax_parallel_calls: 4\nmax_result_bytes: 2048\nmax_total_result_bytes: 16384",
    )
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
    assert!(
        execute_single_call(&tc, &McpToolIndex::new(&map), TEST_MAX_RESULT_BYTES, timeout, true)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn execute_single_call_unknown_tool_returns_none() {
    let map = sample_tool_map();
    let tc = json!({"name": "nonexistent", "call_id": "c1"});
    let timeout = std::time::Duration::from_millis(100);
    assert!(
        execute_single_call(&tc, &McpToolIndex::new(&map), TEST_MAX_RESULT_BYTES, timeout, true)
            .await
            .is_none()
    );
}

#[tokio::test]
async fn execute_single_call_ambiguous_returns_error() {
    let map = lossy_collision_tool_map();
    let tc = json!({"name": "my_server__get", "call_id": "c1"});
    let timeout = std::time::Duration::from_millis(100);
    let result = execute_single_call(&tc, &McpToolIndex::new(&map), TEST_MAX_RESULT_BYTES, timeout, true)
        .await
        .unwrap();
    assert!(result.output_item["error"].as_str().unwrap().contains("ambiguous"));
}

#[tokio::test]
async fn execute_single_call_malformed_args_returns_error() {
    let map = sample_tool_map();
    let tc = json!({"name": "weather__get_weather", "call_id": "c1", "arguments": "not-json"});
    let timeout = std::time::Duration::from_millis(100);
    let result = execute_single_call(&tc, &McpToolIndex::new(&map), TEST_MAX_RESULT_BYTES, timeout, true)
        .await
        .unwrap();
    assert!(result.output_item["error"].as_str().unwrap().contains("malformed"));
}

#[tokio::test]
async fn execute_single_call_connection_error() {
    let map = sample_tool_map();
    let tc = json!({"name": "weather__get_weather", "call_id": "c1", "arguments": {"city": "Paris"}});
    let timeout = std::time::Duration::from_millis(200);
    let result = execute_single_call(&tc, &McpToolIndex::new(&map), TEST_MAX_RESULT_BYTES, timeout, true)
        .await
        .unwrap();
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
    let results = execute_mcp_calls(&[], &McpToolIndex::new(&map), execution_options(false, timeout))
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn execute_mcp_calls_sequential() {
    let map = sample_tool_map();
    let calls = vec![json!({"name": "weather__get_weather", "call_id": "c1", "arguments": {}})];
    let timeout = std::time::Duration::from_millis(200);
    let results = execute_mcp_calls(
        &call_refs(&calls),
        &McpToolIndex::new(&map),
        execution_options(false, timeout),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].output_item["type"], "mcp_call");
}

#[tokio::test]
async fn execute_mcp_calls_parallel() {
    let map = sample_tool_map();
    let calls = vec![json!({"name": "weather__get_weather", "call_id": "c1", "arguments": {}})];
    let timeout = std::time::Duration::from_millis(200);
    let results = execute_mcp_calls(
        &call_refs(&calls),
        &McpToolIndex::new(&map),
        execution_options(true, timeout),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].output_item["type"], "mcp_call");
}

#[tokio::test]
async fn execute_mcp_calls_parallel_preserves_order_across_bounded_chunks() {
    let map = sample_tool_map();
    let calls = vec![
        json!({"name":"missing_a", "call_id":"c1"}),
        json!({"name":"missing_b", "call_id":"c2"}),
        json!({"name":"missing_c", "call_id":"c3"}),
    ];
    let timeout = std::time::Duration::from_millis(100);

    let mut options = execution_options(true, timeout);
    options.max_parallel_calls = 2;
    let results = execute_mcp_calls(&call_refs(&calls), &McpToolIndex::new(&map), options)
        .await
        .unwrap();

    let ids: Vec<&str> = results
        .iter()
        .filter_map(|result| result.output_item["id"].as_str())
        .collect();
    assert_eq!(ids, vec!["c1", "c2", "c3"]);
}

#[tokio::test]
async fn execute_mcp_calls_emits_error_for_unknown_tools() {
    let map = sample_tool_map();
    let calls = vec![json!({"name": "nonexistent", "call_id": "c1"})];
    let timeout = std::time::Duration::from_millis(100);
    let results = execute_mcp_calls(
        &call_refs(&calls),
        &McpToolIndex::new(&map),
        execution_options(false, timeout),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].output_item["error"].as_str().unwrap().contains("no result"));
}

#[tokio::test]
async fn execute_mcp_calls_emits_error_for_unknown_tool_without_call_id() {
    let map = sample_tool_map();
    let calls = vec![json!({"name": "nonexistent"})];
    let timeout = std::time::Duration::from_millis(100);
    let results = execute_mcp_calls(
        &call_refs(&calls),
        &McpToolIndex::new(&map),
        execution_options(false, timeout),
    )
    .await
    .unwrap();
    assert_eq!(results.len(), 1, "must emit error even without call_id");
    assert_eq!(results[0].output_item["id"], "unknown");
    assert!(results[0].output_item["error"].as_str().unwrap().contains("no result"));
}

#[tokio::test]
async fn execute_mcp_calls_rejects_an_aggregate_result_overflow() {
    let map = sample_tool_map();
    let calls = vec![json!({"name":"nonexistent", "call_id":"c1"})];
    let timeout = std::time::Duration::from_millis(100);
    let mut options = execution_options(false, timeout);
    options.max_total_result_bytes = 1;

    assert!(
        execute_mcp_calls(&call_refs(&calls), &McpToolIndex::new(&map), options)
            .await
            .is_err()
    );
}

// =========================================================================
// process_call_result: non-text content blocks
// =========================================================================

#[test]
fn process_call_result_empty_content_produces_empty_output() {
    let call_result = rmcp::model::CallToolResult::success(vec![]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);
    assert_eq!(result.message["output"], "");
}

#[test]
fn process_call_result_multi_text_joins_with_newline() {
    let call_result = rmcp::model::CallToolResult::success(vec![
        rmcp::model::ContentBlock::text("hello"),
        rmcp::model::ContentBlock::text("world"),
    ]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);
    assert_eq!(result.message["output"], "hello\nworld");
}

#[test]
fn process_call_result_image_content_is_preserved() {
    let call_result =
        rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::image("base64data", "image/png")]);
    let result = process_call_result(Ok(call_result), "c1", "srv", "tool", "{}", None, TEST_MAX_RESULT_BYTES);

    let model_output = result.message["output"].as_str().unwrap();
    assert!(
        !model_output.is_empty(),
        "#807 regression: image-only result must not produce empty model output (data loss)"
    );
    assert!(model_output.contains("base64data"), "image data must be preserved");
    assert!(model_output.contains("image/png"), "image mime type must be preserved");

    let client_output = result.output_item["output"].as_str().unwrap();
    assert!(
        !client_output.is_empty(),
        "image-only result must not produce empty client output"
    );
    assert!(
        result.output_item.get("error").is_none() || result.output_item["error"].is_null(),
        "preserved content must not be reported as an error"
    );
}

// =========================================================================
// on_response_body (HttpFilter trait)
// =========================================================================

fn make_dispatch_filter() -> Box<dyn praxis_filter::HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    McpDispatchFilter::from_config(&yaml).unwrap()
}

fn assert_dispatch_action(ctx: &praxis_filter::HttpFilterContext<'_>, expected: &str) {
    assert_eq!(
        ctx.filter_results
            .get("openai_mcp_dispatch")
            .and_then(|results| results.get("action")),
        Some(expected),
        "unexpected MCP dispatch action"
    );
}

#[test]
fn on_response_body_not_end_of_stream_continues_to_stream_parser() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from("data"));
    let result = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(result, FilterAction::Continue),
        "stream chunks must reach the downstream openai_stream_events filter"
    );
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
    assert_dispatch_action(&ctx, "done");
}

#[test]
fn streamed_result_limit_error_uses_next_logical_sequence() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        logical_stream_sequence: 9,
        request_body: json!({"stream":true}),
        ..ResponsesState::default()
    });

    let FilterAction::Reject(response) = McpDispatchFilter::result_limit_action(&mut ctx).unwrap() else {
        panic!("streamed result limit must finish with a local SSE response");
    };
    let text = std::str::from_utf8(response.body.as_ref().unwrap()).unwrap();
    let data = text.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
    let payload: serde_json::Value = serde_json::from_str(data).unwrap();

    assert_eq!(payload["sequence_number"], 9);
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().logical_stream_sequence,
        10
    );
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
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
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
    assert_dispatch_action(&ctx, "loop");
}

#[test]
fn on_response_body_rejects_duplicate_mcp_call_ids_before_dispatch() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"duplicate"}),
            json!({"name":"docs__search_docs", "call_id":"duplicate"}),
        ],
        ..ResponsesState::default()
    });

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502));
    assert!(ctx.get_metadata("openai_mcp_dispatch.action").is_none());
}

#[test]
fn on_response_body_rejects_mcp_call_id_reused_from_prior_round() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        tool_calls: vec![json!({"name":"weather__get_weather", "call_id":"reused"})],
        accumulated_output: vec![json!({
            "type":"mcp_call", "id":"reused", "name":"get_weather", "output":"prior result"
        })],
        ..ResponsesState::default()
    });

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502));
    assert!(ctx.get_metadata("openai_mcp_dispatch.action").is_none());
}

#[test]
fn on_response_body_ends_stream_for_missing_mcp_call_id() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        request_body: json!({"stream":true}),
        mcp_tool_map: tool_map,
        tool_calls: vec![json!({"name":"weather__get_weather"})],
        ..ResponsesState::default()
    });

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_dispatch_action(&ctx, "done");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty());
}

#[test]
fn on_response_body_rejects_mcp_batch_over_hard_cap() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_calls_per_round: 1").unwrap();
    let filter = McpDispatchFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"c1"}),
            json!({"name":"docs__search_docs", "call_id":"c2"}),
        ],
        ..ResponsesState::default()
    });

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(&action, FilterAction::Reject(rejection) if rejection.status == 502));
}

#[test]
fn on_response_body_ends_stream_for_mcp_batch_over_hard_cap() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_calls_per_round: 1").unwrap();
    let filter = McpDispatchFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        request_body: json!({"stream":true}),
        mcp_tool_map: tool_map,
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"c1"}),
            json!({"name":"docs__search_docs", "call_id":"c2"}),
        ],
        ..ResponsesState::default()
    });

    let action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_dispatch_action(&ctx, "done");
    assert_eq!(ctx.get_metadata("responses.stream_error_code"), Some("server_error"));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty());
}

#[tokio::test]
async fn on_response_body_approval_required_sets_done_metadata() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        // The store filter armed persistence for this exchange.
        store_persist_armed: true,
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    register_store(&mut ctx, make_approval_store().await);
    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
    assert_eq!(
        ctx.filter_metadata.get("openai_mcp_dispatch.action"),
        Some(&"done".to_owned())
    );
    assert_dispatch_action(&ctx, "done");
}

#[tokio::test]
async fn on_response_body_approval_emits_correct_arguments() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({
            "name": "weather__get_weather",
            "call_id": "c1",
            "arguments": "{\"city\":\"Paris\"}"
        })],
        // The store filter armed persistence for this exchange.
        store_persist_armed: true,
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    register_store(&mut ctx, make_approval_store().await);
    let mut body = None;
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output.len(), 1);
    let event = &state.accumulated_output[0];
    assert_eq!(event["type"], "mcp_approval_request");
    assert_eq!(
        event["arguments"], "{\"city\":\"Paris\"}",
        "approval event arguments must not be double-encoded"
    );
}

#[tokio::test]
async fn on_response_body_approval_serializes_approval_request_into_body() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({
            "name": "weather__get_weather",
            "call_id": "c1",
            "arguments": "{\"city\":\"Paris\"}"
        })],
        response_object: json!({
            "id": "resp_123",
            "output": []
        }),
        // The store filter armed persistence for this exchange.
        store_persist_armed: true,
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    register_store(&mut ctx, make_approval_store().await);

    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));

    let bytes = body.expect("response body should be serialized with approval request");
    let response_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let output = response_json["output"].as_array().expect("output should be an array");
    assert_eq!(output.len(), 1, "output array should contain 1 item");
    assert_eq!(
        output[0]["type"], "mcp_approval_request",
        "output item should be mcp_approval_request"
    );
    assert_eq!(output[0]["id"], "c1");
}

#[test]
fn on_response_body_approval_required_with_store_disabled_is_rejected() {
    // An approval round trip is only completable if this response is persisted:
    // the resume turn correlates its mcp_approval_response to the server-owned
    // pending record via previous_response_id. With store=false nothing is
    // persisted, so the proxy must fail closed here instead of emitting an
    // mcp_approval_request the mandatory follow-up can never resume.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        request_body: json!({"model": "gpt-4.1", "store": false}),
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let rejection = expect_reject(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_eq!(
        rejection.status, 400,
        "combining require_approval with store=false is a client error"
    );
    assert!(
        reject_message(&rejection).contains("store"),
        "the error should explain approvals require store=true: {}",
        reject_message(&rejection)
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.pending_approvals.is_empty(),
        "no pending approval may be recorded when the response won't be stored"
    );
    assert!(
        state.accumulated_output.is_empty(),
        "no mcp_approval_request may be emitted when it can never be resumed"
    );
}

#[tokio::test]
async fn on_response_body_approval_required_with_store_enabled_still_emits() {
    // The gate is specifically store=false or a store that did not arm
    // persistence for this exchange, not "the store field is present": an
    // explicit store=true with the store filter armed must still pause for
    // approval and record the server-owned pending row.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        request_body: json!({"model": "gpt-4.1", "store": true}),
        // The store filter armed persistence for this exchange during the
        // request phase.
        store_persist_armed: true,
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    register_store(&mut ctx, make_approval_store().await);
    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(result, FilterAction::Continue),
        "store=true must still pause for approval"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.pending_approvals.len(),
        1,
        "store=true records the server-owned pending approval"
    );
}

#[test]
fn on_response_body_approval_required_without_registered_store_is_rejected() {
    // store defaults to true, but no store filter ran to arm persistence for
    // this exchange (store_persist_armed stays false) and no backend is
    // registered. Persisting the server-owned pending record is then impossible,
    // so the mandatory mcp_approval_response follow-up could never resume via
    // previous_response_id. The proxy must fail closed here instead of emitting
    // an mcp_approval_request that can never be resumed. This mirrors the resume
    // path, which returns the same 500 when the store is unavailable.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        // No explicit store field: it defaults to true, so the client opted in.
        request_body: json!({"model": "gpt-4.1"}),
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    // Deliberately do NOT register a ResponseStoreRegistry.
    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let rejection = expect_reject(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_eq!(
        rejection.status, 500,
        "a missing store backend is a server configuration error, not a client error"
    );
    assert!(
        reject_message(&rejection).contains("store"),
        "the error should explain a store is required to persist the approval: {}",
        reject_message(&rejection)
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.pending_approvals.is_empty(),
        "no pending approval may be recorded when no store can persist it"
    );
    assert!(
        state.accumulated_output.is_empty(),
        "no mcp_approval_request may be emitted when it can never be resumed"
    );
}

#[tokio::test]
async fn on_response_body_approval_required_registered_but_not_armed_is_rejected() {
    // Registry membership is pipeline-scoped, not exchange-scoped: a store
    // backend can be registered (by a store filter that is response-conditioned,
    // mis-ordered relative to dispatch, or otherwise skips persistence for THIS
    // exchange) while this response is never persisted. Emitting an
    // mcp_approval_request in that state strands the client with an approval the
    // mandatory follow-up can never resume. The proxy must gate on the
    // exchange-scoped persistence-armed marker the store filter sets only when it
    // will persist THIS response, not on registry membership alone.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        // store defaults to true (client opted in), yet the store filter never
        // armed persistence for this exchange (store_persist_armed stays false).
        request_body: json!({"model": "gpt-4.1"}),
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    // A backend IS registered in the pipeline registry, but that only proves
    // pipeline-scoped membership — not that persistence will run for this
    // exchange.
    register_store(&mut ctx, make_approval_store().await);
    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let rejection = expect_reject(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert_eq!(
        rejection.status, 500,
        "a registered-but-unarmed store cannot persist this exchange; fail closed"
    );
    assert!(
        reject_message(&rejection).contains("store"),
        "the error should explain a store is required to persist the approval: {}",
        reject_message(&rejection)
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.pending_approvals.is_empty(),
        "no pending approval may be recorded when this exchange will not persist"
    );
    assert!(
        state.accumulated_output.is_empty(),
        "no mcp_approval_request may be emitted when it can never be resumed"
    );
}

#[tokio::test]
async fn on_response_body_unresumable_approval_on_committed_stream_emits_sse_error() {
    // The persistence guard fires from the response-body hook. On a committed
    // stream (subrequest_response_mode == Streaming) the SSE headers and model
    // events are already on the wire, so returning FilterAction::Reject would be
    // converted into a transport error after a truncated success — the client
    // never learns why. Instead the guard must hand a terminal error to the
    // logical-stream finalizer via the committed-stream metadata and skip
    // persistence, without emitting an unresumable mcp_approval_request.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.set_subrequest_response_mode(praxis_filter::SubRequestResponseMode::Streaming);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1"})],
        // store defaults to true (client opted in), yet the store filter never
        // armed persistence for this exchange, so the approval is unresumable.
        request_body: json!({"model": "gpt-4.1", "stream": true}),
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    register_store(&mut ctx, make_approval_store().await);
    let mut body = Some(Bytes::from(r#"{"id":"resp_123","output":[]}"#));
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(result, FilterAction::Continue),
        "a committed stream must not fail closed with FilterAction::Reject; the error \
         reaches the client through the logical-stream finalizer instead"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_error_code"),
        Some("server_error"),
        "the committed-stream error code must be published for the finalizer"
    );
    assert!(
        ctx.get_metadata("responses.stream_error_message")
            .is_some_and(|m| m.contains("store")),
        "the committed-stream error message must explain a store is required"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "an unresumable approval must not persist the committed response"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.pending_approvals.is_empty(),
        "no pending approval may be recorded when this exchange will not persist"
    );
    assert!(
        state.accumulated_output.is_empty(),
        "no mcp_approval_request may be emitted when it can never be resumed"
    );
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "response and request phases of approval-only sibling re-entry"
)]
async fn approval_only_batch_returns_after_web_search_sibling_reentry() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"c1", "arguments":"{}"}),
            json!({"name":"docs__search_docs", "call_id":"c2", "arguments":"{}"}),
        ],
        web_search_calls: vec![json!({"type":"web_search_call", "id":"ws_1", "status":"completed"})],
        response_object: json!({"id":"resp_batch", "output":[]}),
        // Approvals are server-owned: arm persistence so the round is resumable.
        store_persist_armed: true,
        ..ResponsesState::default()
    });
    register_store(&mut ctx, make_approval_store().await);
    let mut body = Some(Bytes::from_static(br#"{"id":"resp_batch","output":[]}"#));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_dispatch_action(&ctx, "done");
    let response: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let approvals = response["output"].as_array().unwrap();
    assert_eq!(approvals.len(), 2, "every gated batch member must remain pending");
    assert!(approvals.iter().all(|item| item["type"] == "mcp_approval_request"));
    assert!(
        ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.is_empty(),
        "gated calls must not survive into a sibling dispatcher's re-entry"
    );
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().mcp_approval_state,
        McpApprovalState::ApprovalPendingThenReturn
    );

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.web_search_calls.len(),
        1,
        "MCP must leave the sibling queue intact"
    );
    assert_eq!(state.mcp_approval_state, McpApprovalState::ApprovalPendingThenReturn);
}

#[test]
fn oversized_completed_result_becomes_a_bounded_per_call_error() {
    let call = json!({
        "name": "tool_name_far_beyond_the_normal_schema_boundary_but_still_bounded_by_the_fallback",
        "call_id": "call_id_far_beyond_the_normal_schema_boundary_but_still_bounded_by_the_fallback"
    });
    let result = build_success_result("c1", "srv", "tool", "{}", &"x".repeat(4096), false, None);

    let bounded = super::fit_result_or_limit_error(&call, result, super::MIN_RETAINED_RESULT_BYTES);

    assert!(bounded.retained_bytes().unwrap() <= super::MIN_RETAINED_RESULT_BYTES);
    assert!(
        bounded.output_item["error"]
            .as_str()
            .unwrap()
            .contains("retained-byte limit")
    );
    assert_eq!(result_payload_limit(4096), 1024);
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "response and request phases of the quota lifecycle are asserted together"
)]
async fn over_budget_gated_call_is_rejected_without_an_approval_request() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    tool_map
        .get_mut(&("weather".to_owned(), "get_weather".to_owned()))
        .unwrap()["require_approval"] = json!("always");
    let web = json!({"type":"web_search_call", "id":"ws_1", "status":"completed"});
    let gated = json!({
        "type":"function_call", "name":"weather__get_weather",
        "call_id":"mcp_gated", "arguments":"{}", "status":"completed"
    });
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        max_tool_calls: Some(1),
        tool_calls: vec![gated.clone()],
        web_search_calls: vec![web.clone()],
        accumulated_output: vec![web.clone(), gated.clone()],
        response_object: json!({"id":"resp_budget", "object":"response", "status":"completed", "output":[web, gated]}),
        ..ResponsesState::default()
    });

    let response_action = filter.on_response_body(&mut ctx, &mut None, true).unwrap();
    assert!(matches!(response_action, FilterAction::Continue));
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .accumulated_output
            .iter()
            .all(|item| item["type"] != "mcp_approval_request")
    );

    let request_action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();
    assert!(matches!(request_action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.accumulated_output.iter().any(|item| {
        item["type"] == "mcp_call"
            && item["id"] == "mcp_gated"
            && item["error"]
                .as_str()
                .is_some_and(|error| error.contains("max_tool_calls"))
    }));
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the response and request phases of one approval lifecycle must be asserted together"
)]
async fn mixed_approval_batch_executes_ungated_sibling_before_returning_approval() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    tool_map
        .get_mut(&("weather".to_owned(), "get_weather".to_owned()))
        .unwrap()["require_approval"] = json!("always");
    tool_map
        .get_mut(&("docs".to_owned(), "search_docs".to_owned()))
        .unwrap()["require_approval"] = json!("never");
    let calls = vec![
        json!({"type":"function_call", "name":"weather__get_weather", "call_id":"c1", "arguments":"{}"}),
        json!({"type":"function_call", "name":"docs__search_docs", "call_id":"c2", "arguments":"{bad"}),
    ];
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        max_tool_calls: Some(2),
        tool_calls: calls.clone(),
        accumulated_output: calls.clone(),
        response_object: json!({"id":"resp_batch", "object":"response", "status":"completed", "output":calls}),
        // Approvals are server-owned: arm persistence so the round is resumable.
        store_persist_armed: true,
        ..ResponsesState::default()
    });
    register_store(&mut ctx, make_approval_store().await);
    let mut response_body = Some(Bytes::from_static(br#"{"id":"resp_batch","output":[]}"#));

    let response_action = filter.on_response_body(&mut ctx, &mut response_body, true).unwrap();
    assert!(matches!(response_action, FilterAction::Continue));
    assert_dispatch_action(&ctx, "loop");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.mcp_approval_state, McpApprovalState::ExecuteUngatedThenReturn);
    assert_eq!(state.tool_calls.len(), 1);
    assert_eq!(state.tool_calls[0]["call_id"], "c2");

    let mut request_body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let request_action = filter.on_request_body(&mut ctx, &mut request_body, true).await.unwrap();
    assert!(matches!(request_action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let output = &state.accumulated_output;
    assert!(
        output
            .iter()
            .any(|item| item["type"] == "mcp_approval_request" && item["id"] == "c1")
    );
    assert!(
        output
            .iter()
            .any(|item| item["type"] == "mcp_call" && item["id"] == "c2")
    );
    assert_eq!(state.mcp_approval_state, McpApprovalState::ExecuteUngatedThenReturn);
}

// =========================================================================
// on_request (HttpFilter trait)
// =========================================================================

#[tokio::test]
async fn on_request_no_state_returns_continue() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let result = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
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
async fn on_request_body_executes_and_appends_results_before_proxy_serialization() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let state = ResponsesState {
        mcp_tool_map: sample_tool_map(),
        tool_calls: vec![json!({"name": "weather__get_weather", "call_id": "c1", "arguments": {}})],
        ..ResponsesState::default()
    };
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let result = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(result, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.messages.is_empty(), "should append result messages");
    assert!(
        !state.accumulated_output.is_empty(),
        "should append output items to accumulated_output"
    );
    assert!(state.tool_calls.is_empty(), "should clear executed MCP tool calls");
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the two-phase response-wide budget lifecycle is asserted together"
)]
async fn on_request_body_enforces_exhausted_max_tool_calls_without_side_effects() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        max_tool_calls: Some(0),
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"c1", "arguments":{}}),
            json!({"name":"docs__search_docs", "call_id":"c2", "arguments":{}}),
        ],
        response_object: json!({"id":"resp_limit", "object":"response", "status":"completed", "output":[]}),
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.tool_calls.is_empty());
    assert_eq!(state.accumulated_output.len(), 2);
    assert!(state.accumulated_output.iter().all(|item| {
        item["error"]
            .as_str()
            .is_some_and(|error| error.contains("max_tool_calls"))
    }));
    assert_eq!(
        state.mcp_approval_state,
        McpApprovalState::ToolLimitExceededThenReturn,
        "the trailing agentic-loop filter owns local completion"
    );
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "mixed dispatcher state and retained output assertions"
)]
async fn deferred_web_limit_retains_mcp_siblings_before_local_completion() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut tool_map = sample_tool_map();
    for entry in tool_map.values_mut() {
        entry["require_approval"] = json!("never");
    }
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        max_tool_calls: Some(1),
        deferred_tool_limit_completion: true,
        tool_calls: vec![
            json!({"name":"weather__get_weather", "call_id":"mcp_1", "arguments":{}}),
            json!({"name":"docs__search_docs", "call_id":"mcp_2", "arguments":{}}),
        ],
        accumulated_output: vec![
            json!({"type":"web_search_call", "id":"ws_1", "status":"completed"}),
            json!({"type":"web_search_call", "id":"ws_2", "status":"failed"}),
        ],
        response_object: json!({"id":"resp_mixed", "object":"response", "status":"completed", "output":[]}),
        ..ResponsesState::default()
    });

    let action = filter.on_request_body(&mut ctx, &mut None, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output.len(), 4);
    assert_eq!(state.accumulated_output[0]["id"], "ws_1");
    assert_eq!(state.accumulated_output[1]["id"], "ws_2");
    assert!(state.accumulated_output[2..].iter().all(|item| {
        item["type"] == "mcp_call"
            && item["error"]
                .as_str()
                .is_some_and(|error| error.contains("max_tool_calls"))
    }));
    assert!(state.deferred_tool_limit_completion);
    assert_eq!(state.mcp_approval_state, McpApprovalState::ToolLimitExceededThenReturn);
}

// =========================================================================
// Resolve → Dispatch roundtrip (end-to-end data contract)
// =========================================================================

fn resolver_style_tool_map(
    label: &str,
    tool_name: &str,
    server_url: &str,
) -> HashMap<(String, String), serde_json::Value> {
    let mut map = HashMap::new();
    map.insert(
        (label.to_owned(), tool_name.to_owned()),
        json!({
            "server_label": label,
            "server_url": server_url,
            "headers": null,
            "authorization": null,
            "require_approval": "never",
            "tool_definition": {
                "name": tool_name,
                "description": "Get weather for a city",
                "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }),
    );
    map
}

#[test]
fn resolve_to_dispatch_encoded_name_roundtrip() {
    let (label, tool_name, url) = ("weather", "get_weather", "http://weather.example.com/mcp");
    let encoded = encode_function_name(label, tool_name);
    assert_eq!(encoded, "weather__get_weather");

    let tool_map = resolver_style_tool_map(label, tool_name, url);
    let tool_calls = vec![
        json!({"name": encoded, "call_id": "c1"}),
        json!({"name": "plain_function", "call_id": "c2"}),
    ];

    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map.clone(),
        tool_calls: tool_calls.clone(),
        ..ResponsesState::default()
    });

    let mut body = None;
    let result = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(matches!(result, FilterAction::Continue));
    assert_eq!(
        ctx.filter_metadata.get("openai_mcp_dispatch.action"),
        Some(&"execute_mcp".to_owned()),
    );
    assert_dispatch_action(&ctx, "loop");

    let (key, entry) = find_by_encoded_name(&McpToolIndex::new(&tool_map), &encoded).unwrap();
    assert_eq!((key.0.as_str(), key.1.as_str()), (label, tool_name));
    assert_eq!(entry["server_url"], url);

    let mcp_calls = extract_mcp_tool_calls(&tool_calls, &McpToolIndex::new(&tool_map));
    assert_eq!(mcp_calls.len(), 1);
    assert_eq!(mcp_calls[0]["name"], encoded);
}

#[tokio::test]
async fn resolve_to_dispatch_execute_with_original_name() {
    let label = "weather";
    let tool_name = "get_weather";
    let encoded_name = encode_function_name(label, tool_name);
    let tool_map = resolver_style_tool_map(label, tool_name, "http://192.0.2.1:1/mcp");

    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: tool_map,
        tool_calls: vec![json!({"name": encoded_name, "call_id": "c1", "arguments": "{\"city\":\"NYC\"}"})],
        ..ResponsesState::default()
    });

    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let result = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(result, FilterAction::Continue));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.messages.is_empty(), "should append result messages");

    let output = &state.accumulated_output;
    assert!(!output.is_empty(), "should append output items to accumulated_output");
    assert_eq!(output[0]["type"], "mcp_call");
    assert_eq!(
        output[0]["name"], tool_name,
        "should use original tool name, not encoded"
    );
    assert_eq!(output[0]["server_label"], label);
    assert!(state.tool_calls.is_empty(), "should clear executed MCP tool calls");
}

// =========================================================================
// Approval Response Round Trip
// =========================================================================

/// A tool map whose single entry requires approval and points at an
/// unreachable server, so an approved call resolves and executes (failing
/// fast with a connection error) without contacting a real host.
/// The resolved tool-map entry for the `weather` server shared by the approval
/// tests.
///
/// A loopback URL under the default `allow_loopback=false` policy makes any
/// executed approved call fail closed instantly via the SSRF guard — no
/// network round trip, no timeout wait. The resume/consume/inject logic is
/// what these tests exercise; `build_error_result` still preserves the
/// call identity (id, name, `server_label`, arguments, `approval_request_id`)
/// that the "preserved call" assertions read, so a fast SSRF failure and a
/// real tool response are equivalent for their purpose.
fn weather_entry() -> serde_json::Value {
    json!({
        "server_label": "weather",
        "server_url": "http://127.0.0.1:1/mcp",
        "headers": null,
        "authorization": null,
        "tool_definition": {"name": "get_weather"},
        "require_approval": "always",
    })
}

fn approval_tool_map() -> HashMap<(String, String), serde_json::Value> {
    let mut map = HashMap::new();
    map.insert(("weather".to_owned(), "get_weather".to_owned()), weather_entry());
    map
}

/// The stored `mcp_approval_request` Request 1 emitted, as it survives into
/// Request 2's `persisted_messages` (the source of truth for target binding).
///
/// Carries the `target_fingerprint` of [`weather_entry`] so resolution against
/// [`approval_tool_map`] binds the same target identity it was granted for; a
/// missing or divergent fingerprint fails closed.
fn stored_approval_request(id: &str, label: &str, name: &str, arguments: &str) -> serde_json::Value {
    json!({
        "type": "mcp_approval_request",
        "id": id,
        "name": name,
        "server_label": label,
        "arguments": arguments,
        "target_fingerprint": target_fingerprint(&weather_entry()),
    })
}

/// A client-supplied `mcp_approval_response` decision.
fn approval_response(id: &str, approve: bool, reason: Option<&str>) -> serde_json::Value {
    let mut item = json!({
        "type": "mcp_approval_response",
        "approval_request_id": id,
        "approve": approve,
    });
    if let Some(reason) = reason {
        item["reason"] = json!(reason);
    }
    item
}

/// Build a server-owned pending-approval record as the proxy would have written
/// it when it emitted the `mcp_approval_request`.
fn pending_record(id: &str, label: &str, name: &str, arguments: &str, fingerprint: String) -> PendingApprovalRecord {
    PendingApprovalRecord {
        approval_id: id.to_owned(),
        server_label: label.to_owned(),
        tool_name: name.to_owned(),
        arguments: arguments.to_owned(),
        target_fingerprint: fingerprint,
    }
}

/// The response id that issued the seeded approvals. A pending approval is
/// scoped to its originating response, so resume tests set
/// [`ResponsesState::previous_response_id`] to this value to be in scope.
const APPROVAL_PREV_ID: &str = "resp_prev";

/// Seed `store` with the pending approval the proxy would have recorded for a
/// weather call under `response_id`. Its fingerprint binds to [`weather_entry`]
/// so a resume against [`approval_tool_map`] matches; correlation is
/// server-owned, never inferred from conversation history.
async fn seed_weather_approval(store: &dyn ResponseStore, response_id: &str, id: &str, arguments: &str) {
    let record = pending_record(
        id,
        "weather",
        "get_weather",
        arguments,
        target_fingerprint(&weather_entry()),
    );
    store
        .record_pending_approvals(DEFAULT_TENANT_ID, response_id, std::slice::from_ref(&record), 1000)
        .await
        .expect("seeding the pending approval should succeed");
}

/// A fresh in-memory SQLite store for the approval-consumption path.
async fn make_approval_store() -> Arc<dyn ResponseStore> {
    Arc::new(
        SqliteResponseStore::new("sqlite::memory:", "resp", "conv", None, None)
            .await
            .expect("in-memory store should initialize"),
    )
}

/// Insert a registry exposing `store` under the default name into `ctx`.
fn register_store(ctx: &mut praxis_filter::HttpFilterContext<'_>, store: Arc<dyn ResponseStore>) {
    let registry = ResponseStoreRegistry::new();
    registry
        .register(&Arc::from("default"), store)
        .expect("register should succeed");
    ctx.extensions.insert(registry);
}

/// Unwrap a rejecting filter action.
fn expect_reject(action: FilterAction) -> praxis_filter::Rejection {
    match action {
        FilterAction::Reject(rejection) => rejection,
        _ => panic!("expected FilterAction::Reject, resume did not fail closed"),
    }
}

/// Parse the `error.message` from a non-streaming rejection body.
fn reject_message(rejection: &praxis_filter::Rejection) -> String {
    let body = rejection.body.clone().expect("rejection carries a body");
    let parsed: serde_json::Value = serde_json::from_slice(&body).expect("rejection body is JSON");
    parsed["error"]["message"]
        .as_str()
        .expect("error.message is a string")
        .to_owned()
}

#[tokio::test]
async fn resume_approval_approve_executes_once_and_preserves_call() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let store = make_approval_store().await;
    let args = "{\"city\":\"Paris\"}";
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", args).await;
    register_store(&mut ctx, store);

    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });

    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let result = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(result, FilterAction::Continue),
        "approval should resume, not reject"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        !state.messages.iter().any(is_approval_response),
        "mcp_approval_response must be stripped from backend-bound messages"
    );
    assert_eq!(state.accumulated_output.len(), 1, "exactly one tools/call must run");
    let call = &state.accumulated_output[0];
    assert_eq!(call["type"], "mcp_call");
    assert_eq!(call["id"], "call_1", "call id preserved across the round trip");
    assert_eq!(call["name"], "get_weather", "original tool name preserved");
    assert_eq!(call["server_label"], "weather", "server label preserved");
    assert!(
        call["arguments"].as_str().unwrap().contains("Paris"),
        "arguments preserved from the stored request"
    );
    assert_eq!(
        call["approval_request_id"], "call_1",
        "mcp_call must reference the authorizing approval"
    );
    assert!(state.tool_calls.is_empty(), "executed approved call must be cleared");
}

#[tokio::test]
async fn resume_approval_deny_resumes_without_tool_call() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;
    register_store(&mut ctx, store);

    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", false, Some("too risky"))],
        ..ResponsesState::default()
    });

    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let result = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(result, FilterAction::Continue),
        "denial should resume inference"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.accumulated_output.is_empty(),
        "denial must perform zero tools/call (no mcp_call output)"
    );
    assert!(
        !state.messages.iter().any(is_approval_response),
        "mcp_approval_response must be stripped"
    );
    assert!(
        !state.messages.iter().any(|m| m["type"] == "mcp_call"),
        "denial must not fabricate an mcp_call item"
    );
    let denial = state
        .messages
        .iter()
        .find(|m| m["type"] == "function_call_output")
        .expect("denial should append a function_call_output");
    assert_eq!(denial["call_id"], "call_1", "denial correlates to the original call id");
    let output = denial["output"].as_str().unwrap();
    assert!(
        output.contains("denied"),
        "denial output should state the tool was denied"
    );
    assert!(output.contains("too risky"), "denial output should surface the reason");
    assert!(
        state
            .persisted_messages
            .iter()
            .any(|m| m["type"] == "function_call_output"),
        "denial should be persisted for the durable trace"
    );
}

#[tokio::test]
async fn resume_approval_replay_is_rejected() {
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;
    let filter = make_dispatch_filter();

    // First request approves and executes.
    let req1 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx1 = make_filter_context(&req1);
    register_store(&mut ctx1, Arc::clone(&store));
    ctx1.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body1 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let first = filter.on_request_body(&mut ctx1, &mut body1, true).await.unwrap();
    assert!(matches!(first, FilterAction::Continue), "first approval should succeed");

    // Second request replays the same approval and must fail closed.
    let req2 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx2 = make_filter_context(&req2);
    register_store(&mut ctx2, Arc::clone(&store));
    ctx2.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body2 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx2, &mut body2, true).await.unwrap());
    assert_eq!(rejection.status, 400, "replay is a client error");
    assert!(
        reject_message(&rejection).contains("already been used"),
        "replay error should explain the approval was already used"
    );

    let state = ctx2.extensions.get::<ResponsesState>().unwrap();
    assert!(state.tool_calls.is_empty(), "replay must not execute the tool again");
    assert!(
        state.accumulated_output.is_empty(),
        "replay must produce no second mcp_call"
    );
}

#[tokio::test]
async fn resume_approval_deny_then_approve_is_rejected() {
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;
    let filter = make_dispatch_filter();

    // First request denies the pending call. Denial still resumes inference and
    // must burn the single-use approval token just like an approval does.
    let req1 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx1 = make_filter_context(&req1);
    register_store(&mut ctx1, Arc::clone(&store));
    ctx1.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", false, Some("too risky"))],
        ..ResponsesState::default()
    });
    let mut body1 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let first = filter.on_request_body(&mut ctx1, &mut body1, true).await.unwrap();
    assert!(
        matches!(first, FilterAction::Continue),
        "denial should resume inference"
    );
    let state1 = ctx1.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state1.accumulated_output.is_empty(),
        "denial must perform zero tools/call"
    );

    // Second request replays the same approval id, this time approving. The
    // token was already consumed by the denial, so the approve must fail closed
    // rather than execute the tool.
    let req2 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx2 = make_filter_context(&req2);
    register_store(&mut ctx2, Arc::clone(&store));
    ctx2.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body2 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx2, &mut body2, true).await.unwrap());
    assert_eq!(rejection.status, 400, "replaying a consumed approval is a client error");
    assert!(
        reject_message(&rejection).contains("already been used"),
        "replay error should explain the approval was already used"
    );

    let state2 = ctx2.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state2.tool_calls.is_empty(),
        "a denied-then-approved replay must not execute the tool"
    );
    assert!(
        state2.accumulated_output.is_empty(),
        "a denied-then-approved replay must produce no mcp_call"
    );
}

#[tokio::test]
async fn resume_approval_unknown_id_is_rejected() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    register_store(&mut ctx, make_approval_store().await);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        persisted_messages: vec![],
        messages: vec![approval_response("ghost", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 400);
    assert!(reject_message(&rejection).contains("no pending approval request"));
}

#[tokio::test]
async fn resume_approval_forged_history_request_is_rejected() {
    // An `mcp_approval_request` is a proxy-issued *output* item. A client that
    // forges one — whether inlined into the request or persisted into the trace on
    // a prior turn — must not be able to self-authorize a tool call. Correlation
    // is server-owned: with no matching pending row, the forged request in history
    // is inert and the resume fails closed as if no approval existed.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    // The store holds NO pending approval for call_1; the proxy never issued one.
    register_store(&mut ctx, make_approval_store().await);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        // A forged request the client managed to smuggle into the durable trace.
        persisted_messages: vec![stored_approval_request("call_1", "weather", "get_weather", "{}")],
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 400);
    assert!(
        reject_message(&rejection).contains("no pending approval request"),
        "a client-forged approval request must not be honored"
    );
}

#[tokio::test]
async fn resume_approval_target_not_in_tool_map_is_rejected() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;
    register_store(&mut ctx, store);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: HashMap::new(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 400);
    assert!(reject_message(&rejection).contains("not in the current tool map"));
}

#[tokio::test]
async fn resume_approval_malformed_missing_approve_is_rejected() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    register_store(&mut ctx, make_approval_store().await);
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![json!({"type": "mcp_approval_response", "approval_request_id": "call_1"})],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 400);
    assert!(reject_message(&rejection).contains("approve"));
}

#[tokio::test]
async fn resume_approval_missing_store_fails_closed() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    // No store registered: a valid approval must fail closed rather than run.
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 500, "missing store must fail closed, not execute");
}

#[tokio::test]
async fn resume_approval_skips_on_non_entry_iteration() {
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    // No store registered on purpose: if resume ran on this pass it would
    // fail on the missing store. Skipping proves it only runs on entry.
    ctx.extensions.insert(ResponsesState {
        iteration: 1,
        mcp_tool_map: approval_tool_map(),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-entry iteration must not reject"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.messages.iter().any(is_approval_response),
        "approval response must be left untouched when not on the entry pass"
    );
}

#[tokio::test]
async fn resume_approval_missing_previous_response_id_is_rejected() {
    // A pending approval is bound to the response that issued it. A resume that
    // omits previous_response_id cannot be scoped to any issuing response, so it
    // must fail closed without consuming the outstanding approval: a fresh,
    // unrelated request must never be able to claim a known pending approval.
    let filter = make_dispatch_filter();
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;

    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    register_store(&mut ctx, Arc::clone(&store));
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        // No previous_response_id: the request is not scoped to any response.
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(
        rejection.status, 400,
        "a missing previous_response_id is a client error"
    );
    assert!(
        reject_message(&rejection).contains("requires previous_response_id"),
        "the error should explain previous_response_id is required"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.accumulated_output.is_empty(),
        "no tool must run without a scoped approval"
    );

    // The outstanding approval must remain claimable: a correctly scoped resume
    // still succeeds, proving the unscoped attempt burned nothing.
    let req2 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx2 = make_filter_context(&req2);
    register_store(&mut ctx2, Arc::clone(&store));
    ctx2.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body2 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let action = filter.on_request_body(&mut ctx2, &mut body2, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "the correctly scoped resume should still succeed"
    );
    let state2 = ctx2.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state2.accumulated_output.len(),
        1,
        "the still-outstanding approval should execute exactly once"
    );
}

#[tokio::test]
async fn resume_approval_wrong_previous_response_id_is_rejected() {
    // An approval issued by response R1 must not be consumable by a resume that
    // claims a different originating response R2. Scoping is server-owned: under
    // R2 there is no matching pending row, so the resume fails closed and R1's
    // approval stays outstanding.
    let filter = make_dispatch_filter();
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;

    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    register_store(&mut ctx, Arc::clone(&store));
    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some("resp_other".to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(
        rejection.status, 400,
        "an approval scoped to a different response is a client error"
    );
    assert!(
        reject_message(&rejection).contains("no pending approval request"),
        "a cross-response approval must be treated as unknown"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.accumulated_output.is_empty(),
        "no tool must run for a cross-response approval"
    );

    // The approval issued by APPROVAL_PREV_ID must remain claimable.
    let req2 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx2 = make_filter_context(&req2);
    register_store(&mut ctx2, Arc::clone(&store));
    ctx2.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body2 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let action = filter.on_request_body(&mut ctx2, &mut body2, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "the correctly scoped resume should succeed"
    );
    let state2 = ctx2.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state2.accumulated_output.len(),
        1,
        "the correctly scoped approval should execute exactly once"
    );
}

#[tokio::test]
async fn resume_approval_batch_exceeding_cap_is_rejected() {
    // The agentic loop issues exactly one function call per round, so a resume
    // turn carries a single approval. A larger batch is rejected before any
    // store work: it both violates that invariant and, left unbounded, could
    // exceed PostgreSQL's 16-bit Bind parameter ceiling in the consume query.
    let filter = make_dispatch_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let store = make_approval_store().await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_1", "{}").await;
    seed_weather_approval(store.as_ref(), APPROVAL_PREV_ID, "call_2", "{}").await;
    register_store(&mut ctx, Arc::clone(&store));

    ctx.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![
            approval_response("call_1", true, None),
            approval_response("call_2", true, None),
        ],
        ..ResponsesState::default()
    });

    let mut body = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let rejection = expect_reject(filter.on_request_body(&mut ctx, &mut body, true).await.unwrap());
    assert_eq!(rejection.status, 400, "an oversized approval batch is a client error");
    assert!(
        reject_message(&rejection).contains("at most"),
        "the error should state the per-request approval cap: {}",
        reject_message(&rejection)
    );

    // Nothing was consumed or applied: the batch fails closed before any side
    // effect, so a corrected single-approval retry can still resume.
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "no approved tool call may be injected when the batch is rejected"
    );
    assert!(
        state.accumulated_output.is_empty(),
        "no tool may run when the batch is rejected"
    );

    // The still-outstanding approval remains claimable by a compliant retry.
    let req2 = make_request(http::Method::POST, "/v1/responses");
    let mut ctx2 = make_filter_context(&req2);
    register_store(&mut ctx2, Arc::clone(&store));
    ctx2.extensions.insert(ResponsesState {
        mcp_tool_map: approval_tool_map(),
        previous_response_id: Some(APPROVAL_PREV_ID.to_owned()),
        messages: vec![approval_response("call_1", true, None)],
        ..ResponsesState::default()
    });
    let mut body2 = Some(Bytes::from_static(br#"{"model":"gpt-4.1"}"#));
    let action = filter.on_request_body(&mut ctx2, &mut body2, true).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "a single-approval retry should resume after the oversized batch was rejected"
    );
    let state2 = ctx2.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state2.accumulated_output.len(),
        1,
        "the compliant retry executes the approved call exactly once"
    );
}

// -----------------------------------------------------------------------------
// approval.rs pure functions
// -----------------------------------------------------------------------------

#[test]
fn resolve_approval_binds_complete_stored_call() {
    let map = approval_tool_map();
    let input = parse_approval_response(&approval_response("call_1", true, Some("go ahead"))).expect("parse");
    let pending = pending_record(
        "call_1",
        "weather",
        "get_weather",
        "{\"city\":\"Paris\"}",
        target_fingerprint(&weather_entry()),
    );
    let resolved = resolve_approval(&input, &pending, &map).expect("should resolve");
    assert_eq!(
        resolved,
        ResolvedApproval {
            approval_id: "call_1".to_owned(),
            approve: true,
            reason: Some("go ahead".to_owned()),
            server_label: "weather".to_owned(),
            tool_name: "get_weather".to_owned(),
            encoded_name: "weather__get_weather".to_owned(),
            arguments: "{\"city\":\"Paris\"}".to_owned(),
        }
    );
}

#[test]
fn parse_approval_response_missing_approval_request_id_is_malformed() {
    let response = json!({"type": "mcp_approval_response", "approve": true});
    let err = parse_approval_response(&response).unwrap_err();
    assert!(matches!(err, ApprovalError::Malformed(_)));
    assert!(err.message().contains("approval_request_id"));
}

#[test]
fn parse_approval_response_missing_approve_is_malformed() {
    let response = json!({"type": "mcp_approval_response", "approval_request_id": "call_1"});
    let err = parse_approval_response(&response).unwrap_err();
    assert!(matches!(err, ApprovalError::Malformed(_)));
    assert!(err.message().contains("approve"));
}

#[test]
fn resolve_approval_target_missing_from_map_is_unresolvable() {
    let map = HashMap::new();
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record(
        "call_1",
        "weather",
        "get_weather",
        "{}",
        target_fingerprint(&weather_entry()),
    );
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(matches!(err, ApprovalError::TargetUnresolvable(_)));
}

#[test]
fn resolve_approval_ambiguous_encoded_target_is_rejected() {
    // Two distinct (label, name) pairs collide to the same encoded name. The
    // ambiguity is caught while binding the target, before the fingerprint check.
    let map = lossy_collision_tool_map();
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record("call_1", "my.server", "get", "{}", "unchecked".to_owned());
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(matches!(err, ApprovalError::AmbiguousTarget(_)));
}

#[test]
fn resolve_approval_diverging_identity_is_unresolvable() {
    // Stored ("weather.v1","get_weather") encodes identically to the single
    // tool-map entry ("weather_v1","get_weather") but is not the same target.
    let mut map = HashMap::new();
    map.insert(
        ("weather_v1".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "weather_v1",
            "server_url": "http://192.0.2.1:1/mcp",
            "headers": null,
            "authorization": null,
            "tool_definition": {"name": "get_weather"},
            "require_approval": "always",
        }),
    );
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record("call_1", "weather.v1", "get_weather", "{}", "unchecked".to_owned());
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(
        matches!(err, ApprovalError::TargetUnresolvable(_)),
        "identity divergence must fail closed even when the encoded name collides"
    );
}

#[test]
fn resolve_approval_rejects_redirected_target() {
    // The approval was granted for the loopback weather server. The resume turn
    // keeps the same (server_label, tool_name) but points the URL elsewhere, so
    // the fingerprint diverges and the call must fail closed.
    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "weather",
            "server_url": "http://evil.example/mcp",
            "headers": null,
            "authorization": null,
            "tool_definition": {"name": "get_weather"},
            "require_approval": "always",
        }),
    );
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record(
        "call_1",
        "weather",
        "get_weather",
        "{}",
        target_fingerprint(&weather_entry()),
    );
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(
        matches!(err, ApprovalError::TargetIdentityMismatch(_)),
        "a redirected target must fail closed: {err:?}"
    );
    assert!(err.message().contains("different target"));
}

#[test]
fn resolve_approval_rejects_swapped_authorization() {
    // Same URL, but the resume turn injects a different authorization credential.
    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        json!({
            "server_label": "weather",
            "server_url": "http://127.0.0.1:1/mcp",
            "headers": null,
            "authorization": "Bearer attacker-token",
            "tool_definition": {"name": "get_weather"},
            "require_approval": "always",
        }),
    );
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record(
        "call_1",
        "weather",
        "get_weather",
        "{}",
        target_fingerprint(&weather_entry()),
    );
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(
        matches!(err, ApprovalError::TargetIdentityMismatch(_)),
        "a swapped authorization credential must fail closed: {err:?}"
    );
}

#[test]
fn resolve_approval_missing_stored_fingerprint_fails_closed() {
    // A pending record that carries no target fingerprint cannot be verified, so
    // it must be rejected rather than trusted.
    let map = approval_tool_map();
    let input = parse_approval_response(&approval_response("call_1", true, None)).expect("parse");
    let pending = pending_record("call_1", "weather", "get_weather", "{}", String::new());
    let err = resolve_approval(&input, &pending, &map).unwrap_err();
    assert!(
        matches!(err, ApprovalError::TargetIdentityMismatch(_)),
        "a fingerprint-less pending approval must fail closed: {err:?}"
    );
    assert!(err.message().contains("missing its target fingerprint"));
}

#[test]
fn target_fingerprint_is_header_order_independent_and_identity_sensitive() {
    let base = json!({
        "server_url": "http://127.0.0.1:1/mcp",
        "authorization": "Bearer token",
        "headers": {"X-A": "1", "X-B": "2"},
    });
    // Same fields, headers in a different insertion order → identical digest.
    let reordered = json!({
        "server_url": "http://127.0.0.1:1/mcp",
        "authorization": "Bearer token",
        "headers": {"X-B": "2", "X-A": "1"},
    });
    assert_eq!(
        target_fingerprint(&base),
        target_fingerprint(&reordered),
        "header key order must not change the fingerprint"
    );

    for mutated in [
        json!({"server_url": "http://127.0.0.1:2/mcp", "authorization": "Bearer token", "headers": {"X-A": "1", "X-B": "2"}}),
        json!({"server_url": "http://127.0.0.1:1/mcp", "authorization": "Bearer other", "headers": {"X-A": "1", "X-B": "2"}}),
        json!({"server_url": "http://127.0.0.1:1/mcp", "authorization": "Bearer token", "headers": {"X-A": "1", "X-B": "9"}}),
        json!({"server_url": "http://127.0.0.1:1/mcp", "authorization": "Bearer token", "headers": {"X-A": "1"}}),
        json!({"server_url": "http://127.0.0.1:1/mcp", "authorization": "Bearer token", "headers": {"X-A": "1", "X-B": "2"}, "connector_id": "c1"}),
    ] {
        assert_ne!(
            target_fingerprint(&base),
            target_fingerprint(&mutated),
            "a target identity change must change the fingerprint: {mutated}"
        );
    }
}

#[test]
fn target_fingerprint_fails_closed_on_case_insensitive_duplicate_headers() {
    // The MCP transport lowercases header names (via `http::HeaderName`) and keys
    // its header map by the normalized name, so among case-insensitive duplicates
    // the last one in JSON order wins. A case-sensitive key sort hashes both
    // orderings identically, so reversing the two names transmits a different
    // value while preserving the digest — an old approval would still authorize a
    // redirected header. Such an ambiguous target must fingerprint to the empty
    // fail-closed sentinel so it can never match on resume.
    let forward = json!({
        "server_url": "http://127.0.0.1:1/mcp",
        "authorization": "Bearer token",
        "headers": {"X-Target-Tenant": "A", "x-target-tenant": "B"},
    });
    let reversed = json!({
        "server_url": "http://127.0.0.1:1/mcp",
        "authorization": "Bearer token",
        "headers": {"x-target-tenant": "B", "X-Target-Tenant": "A"},
    });
    assert!(
        target_fingerprint(&forward).is_empty(),
        "case-insensitive duplicate header names must fail closed"
    );
    assert!(
        target_fingerprint(&reversed).is_empty(),
        "case-insensitive duplicate header names must fail closed"
    );
}

#[test]
fn build_approved_tool_call_shapes_a_function_call() {
    let resolved = ResolvedApproval {
        approval_id: "call_1".to_owned(),
        approve: true,
        reason: None,
        server_label: "weather".to_owned(),
        tool_name: "get_weather".to_owned(),
        encoded_name: "weather__get_weather".to_owned(),
        arguments: "{\"city\":\"Paris\"}".to_owned(),
    };
    let tc = build_approved_tool_call(&resolved);
    assert_eq!(tc["type"], "function_call");
    assert_eq!(tc["name"], "weather__get_weather");
    assert_eq!(tc["call_id"], "call_1");
    assert_eq!(tc["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(tc["approval_request_id"], "call_1");
}

#[test]
fn build_denial_message_is_schema_valid_function_call_output() {
    let denial = build_denial_message("call_1", Some("too risky"));
    assert_eq!(denial["type"], "function_call_output");
    assert_eq!(denial["call_id"], "call_1");
    let output = denial["output"].as_str().unwrap();
    assert!(output.contains("denied"));
    assert!(output.contains("too risky"));
    // Never a fabricated mcp_call with an invalid "denied" status.
    assert_ne!(denial["type"], "mcp_call");
    assert!(denial.get("status").is_none());
}

#[test]
fn build_denial_message_without_reason_omits_reason_clause() {
    let denial = build_denial_message("call_1", None);
    assert_eq!(denial["output"], "Tool call was denied by the user.");
}

#[test]
fn build_denial_message_blank_reason_is_ignored() {
    let denial = build_denial_message("call_1", Some("   "));
    assert_eq!(denial["output"], "Tool call was denied by the user.");
}

#[test]
fn extract_approval_responses_filters_only_responses() {
    let messages = vec![
        json!({"type": "message", "role": "user"}),
        approval_response("call_1", true, None),
        json!({"type": "function_call_output", "call_id": "x"}),
        approval_response("call_2", false, None),
    ];
    let responses = extract_approval_responses(&messages);
    assert_eq!(responses.len(), 2);
    assert!(responses.iter().copied().all(is_approval_response));
}
