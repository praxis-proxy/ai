// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the `openai_mcp_tool_resolve` filter.

use super::*;

// =========================================================================
// Config Parsing
// =========================================================================

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_mcp_tool_resolve", "filter name");
}

#[test]
fn config_with_custom_timeout() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_ms: 10000").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_mcp_tool_resolve", "filter name");
}

#[test]
fn config_rejects_unknown_fields() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("unknown_field: true").unwrap();
    assert!(
        McpToolResolveFilter::from_config(&yaml).is_err(),
        "unknown fields should be rejected"
    );
}

#[test]
fn config_rejects_zero_timeout() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_ms: 0").unwrap();
    assert!(
        McpToolResolveFilter::from_config(&yaml).is_err(),
        "timeout_ms: 0 should be rejected"
    );
}

#[test]
fn config_rejects_zero_max_servers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_servers: 0").unwrap();
    assert!(
        McpToolResolveFilter::from_config(&yaml).is_err(),
        "max_servers: 0 should be rejected"
    );
}

#[test]
fn config_with_custom_max_servers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_servers: 5").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_mcp_tool_resolve", "filter name");
}

#[test]
fn config_rejects_zero_max_tools() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tools: 0").unwrap();
    assert!(
        McpToolResolveFilter::from_config(&yaml).is_err(),
        "max_tools: 0 should be rejected"
    );
}

#[test]
fn config_with_custom_max_tools() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tools: 50").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_mcp_tool_resolve", "filter name");
}

// =========================================================================
// MCP Entry Extraction
// =========================================================================

#[test]
fn extract_mcp_entries_from_mixed_tools() {
    let body = serde_json::json!({
        "input": "test",
        "tools": [
            {"type": "function", "name": "calc"},
            {"type": "mcp", "server_label": "weather", "server_url": "http://localhost:8001/mcp"},
            {"type": "web_search"},
            {"type": "mcp", "server_label": "calendar", "server_url": "http://localhost:8002/mcp"}
        ]
    });
    let entries = extract_mcp_entries(body.to_string().as_bytes());

    assert_eq!(entries.len(), 2, "should extract 2 MCP entries");
    assert_eq!(
        entries[0]["server_label"].as_str(),
        Some("weather"),
        "first entry server_label"
    );
    assert_eq!(
        entries[1]["server_label"].as_str(),
        Some("calendar"),
        "second entry server_label"
    );
}

#[test]
fn extract_mcp_entries_empty_when_no_mcp() {
    let body = serde_json::json!({
        "input": "test",
        "tools": [{"type": "function", "name": "calc"}]
    });
    let entries = extract_mcp_entries(body.to_string().as_bytes());
    assert!(entries.is_empty(), "should find no MCP entries");
}

#[test]
fn extract_mcp_entries_handles_no_tools() {
    let body = serde_json::json!({"input": "test"});
    let entries = extract_mcp_entries(body.to_string().as_bytes());
    assert!(entries.is_empty(), "should handle missing tools array");
}

#[test]
fn extract_mcp_entries_handles_invalid_json() {
    let entries = extract_mcp_entries(b"not json");
    assert!(entries.is_empty(), "should handle invalid JSON");
}

// =========================================================================
// Cache Matching
// =========================================================================

#[test]
fn cache_hit_when_all_allowed_tools_present() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [
            {"name": "get_weather"},
            {"name": "get_forecast"}
        ]
    })];
    let allowed = vec!["get_weather".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", url, Some(&allowed));
    assert!(result.is_some(), "should hit cache");
    assert_eq!(result.unwrap().len(), 2, "should return full cached listing");
}

#[test]
fn cache_miss_when_allowed_tool_not_in_cache() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [{"name": "get_weather"}]
    })];
    let allowed = vec!["unknown_tool".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", url, Some(&allowed));
    assert!(result.is_none(), "should miss cache for unknown tool");
}

#[test]
fn cache_miss_when_unrestricted_allowed_tools() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [{"name": "get_weather"}, {"name": "get_forecast"}]
    })];

    let result = find_cached_listing(Some(&previous), "weather", url, None);
    assert!(
        result.is_none(),
        "unrestricted entries must miss to avoid reusing partial listings"
    );
}

#[test]
fn cache_miss_when_unrestricted_widens_narrow_cached_listing() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [{"name": "get_weather"}]
    })];

    let result = find_cached_listing(Some(&previous), "weather", url, None);
    assert!(
        result.is_none(),
        "unrestricted must miss when cached listing is a narrow subset"
    );
}

#[test]
fn cache_miss_when_wrong_server_label() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [{"name": "get_weather"}]
    })];
    let allowed = vec!["get_weather".to_owned()];

    let result = find_cached_listing(Some(&previous), "calendar", url, Some(&allowed));
    assert!(result.is_none(), "should miss cache for different server");
}

#[test]
fn cache_miss_when_no_previous_tools() {
    let allowed = vec!["get_weather".to_owned()];
    let result = find_cached_listing(None, "weather", "http://10.0.0.5/mcp", Some(&allowed));
    assert!(result.is_none(), "should miss when no previous_tools");
}

#[test]
fn cache_miss_when_server_url_changed() {
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": "http://10.0.0.5/mcp",
        "tools": [{"name": "get_weather"}]
    })];
    let allowed = vec!["get_weather".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", "http://10.0.0.99/mcp", Some(&allowed));
    assert!(
        result.is_none(),
        "should miss cache when server_url differs from cached entry"
    );
}

#[test]
fn cache_miss_when_continuation_changes_allowed_tools() {
    let url = "http://10.0.0.5/mcp";
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": url,
        "tools": [{"name": "get_weather"}]
    })];
    let new_allowed = vec!["get_forecast".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", url, Some(&new_allowed));
    assert!(
        result.is_none(),
        "cache should miss when continuation requests a tool not in the cached listing"
    );
}

// =========================================================================
// Allowed Tools Filter
// =========================================================================

#[test]
fn filter_keeps_only_allowed_tools() {
    let tools = vec![
        serde_json::json!({"name": "get_weather"}),
        serde_json::json!({"name": "get_forecast"}),
        serde_json::json!({"name": "get_alerts"}),
    ];
    let allowed = AllowedTools {
        names: Some(vec!["get_weather".to_owned(), "get_alerts".to_owned()]),
        read_only: None,
    };

    let filtered = apply_allowed_tools_filter(tools, &allowed);
    assert_eq!(filtered.len(), 2, "should keep only allowed tools");
    assert_eq!(filtered[0]["name"], "get_weather", "first kept tool");
    assert_eq!(filtered[1]["name"], "get_alerts", "second kept tool");
}

#[test]
fn filter_returns_all_when_unrestricted() {
    let tools = vec![serde_json::json!({"name": "a"}), serde_json::json!({"name": "b"})];
    let filtered = apply_allowed_tools_filter(tools, &AllowedTools::unrestricted());
    assert_eq!(filtered.len(), 2, "should return all tools when no filter");
}

#[test]
fn filter_returns_empty_when_no_match() {
    let tools = vec![serde_json::json!({"name": "a"})];
    let allowed = AllowedTools {
        names: Some(vec!["nonexistent".to_owned()]),
        read_only: None,
    };
    let filtered = apply_allowed_tools_filter(tools, &allowed);
    assert!(filtered.is_empty(), "should return empty when no match");
}

#[test]
fn filter_read_only_keeps_annotated_tools() {
    let tools = vec![
        serde_json::json!({"name": "read_data", "annotations": {"readOnlyHint": true}}),
        serde_json::json!({"name": "write_data", "annotations": {"readOnlyHint": false}}),
        serde_json::json!({"name": "no_annotation"}),
    ];
    let allowed = AllowedTools {
        names: None,
        read_only: Some(true),
    };
    let filtered = apply_allowed_tools_filter(tools, &allowed);
    assert_eq!(filtered.len(), 1, "should keep only read-only tools");
    assert_eq!(filtered[0]["name"], "read_data", "kept tool");
}

#[test]
fn filter_read_only_list_applies_both() {
    let tools = vec![
        serde_json::json!({"name": "read_data", "annotations": {"readOnlyHint": true}}),
        serde_json::json!({"name": "write_data", "annotations": {"readOnlyHint": false}}),
        serde_json::json!({"name": "other_read", "annotations": {"readOnlyHint": true}}),
    ];
    let allowed = AllowedTools {
        names: Some(vec!["read_data".to_owned(), "write_data".to_owned()]),
        read_only: Some(true),
    };
    let filtered = apply_allowed_tools_filter(tools, &allowed);
    assert_eq!(filtered.len(), 1, "name + read_only intersection");
    assert_eq!(filtered[0]["name"], "read_data", "only read_data passes both filters");
}

#[test]
fn filter_writable_only_excludes_read_only_tools() {
    let tools = vec![
        serde_json::json!({"name": "read_data", "annotations": {"readOnlyHint": true}}),
        serde_json::json!({"name": "write_data", "annotations": {"readOnlyHint": false}}),
        serde_json::json!({"name": "no_annotation"}),
    ];
    let allowed = AllowedTools {
        names: None,
        read_only: Some(false),
    };
    let filtered = apply_allowed_tools_filter(tools, &allowed);
    assert_eq!(filtered.len(), 2, "should keep writable + unannotated tools");
    assert_eq!(filtered[0]["name"], "write_data");
    assert_eq!(filtered[1]["name"], "no_annotation");
}

// =========================================================================
// Pipeline Regression: no spurious ResponsesState creation
// =========================================================================

/// Build a minimal MCP request body targeting `server_url`.
fn mcp_body(server_url: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "mcp", "server_label": "weather",
                    "server_url": server_url, "allowed_tools": ["get_weather"]}]
    })
}

/// Cache-hit path populates `mcp_tool_map` on an existing state
/// without corrupting `request_body`.
#[tokio::test]
async fn cache_hit_populates_mcp_tool_map_on_existing_state() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let server_url = "http://10.0.0.5/mcp";
    let body_json = mcp_body(server_url);
    let mut state = ResponsesState::from_request_body(body_json.clone());
    state.previous_tools = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": server_url,
        "tools": [{"name": "get_weather", "description": "Get weather"}]
    })];
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    assert!(matches!(action, FilterAction::Continue), "cache hit should continue");

    let state = ctx.extensions.get::<ResponsesState>().expect("state should exist");
    assert!(
        state
            .mcp_tool_map
            .contains_key(&("weather".to_owned(), "get_weather".to_owned())),
        "tool should be in map"
    );
    assert!(!state.request_body.is_null(), "request_body must not be null");
}

/// A cache hit must still reject a blocked `server_url` — a
/// continuation cannot smuggle an SSRF URL through cached tools.
#[tokio::test]
async fn cache_hit_with_blocked_url_still_rejected() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = mcp_body("http://127.0.0.1/mcp");
    let mut state = ResponsesState::from_request_body(body_json.clone());
    state.previous_tools = vec![serde_json::json!({
        "server_label": "weather",
        "tools": [{"name": "get_weather", "description": "Get weather"}]
    })];
    ctx.extensions.insert(state);

    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "cache hit with blocked URL must still be rejected"
    );
}

/// Without a pre-existing `ResponsesState`, the filter must NOT
/// insert a default one (whose `request_body` is `Null`).
#[tokio::test]
async fn no_spurious_state_creation_without_existing_state() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = mcp_body("http://127.0.0.1:1/mcp");
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "should reject (SSRF-blocked loopback)"
    );
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "no state should be inserted"
    );
}

/// MCP entries with `connector_id` but no `server_url` should
/// be skipped, not rejected — the backend handles connector
/// resolution.
#[tokio::test]
async fn connector_id_entry_without_server_url_is_skipped() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "mcp", "server_label": "remote",
                    "connector_id": "mcp_conn_abc123"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "connector_id entry should be skipped, not rejected"
    );
}

#[tokio::test]
async fn missing_server_label_skips_resolution() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "mcp", "server_url": "http://10.0.0.5/mcp"}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "MCP entry without server_label should be skipped"
    );
}

#[tokio::test]
async fn defer_loading_true_skips_eager_resolution() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "mcp", "server_label": "deferred",
                    "server_url": "http://10.0.0.5/mcp",
                    "defer_loading": true}]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "defer_loading: true should skip eager tools/list"
    );
}

/// `write_state` creates `ResponsesState` from body when none
/// exists, preserving `request_body` and populating `mcp_tool_map`.
#[test]
fn write_state_creates_state_when_missing() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let body_json = mcp_body("http://10.0.0.5/mcp");
    let body_bytes = serde_json::to_vec(&body_json).unwrap();

    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        serde_json::json!({"tool": true}),
    );
    write_state(&mut ctx, &body_bytes, map);

    let state = ctx.extensions.get::<ResponsesState>().expect("state should be created");
    assert!(
        state
            .mcp_tool_map
            .contains_key(&("weather".to_owned(), "get_weather".to_owned())),
        "tool should be in map"
    );
    assert_eq!(
        state.request_body["model"], "gpt-4o",
        "request_body should be parsed from body"
    );
    assert!(!state.request_body.is_null(), "request_body must not be null");
}

/// `write_state` updates existing state without replacing it.
#[test]
fn write_state_updates_existing_state() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let body_json = mcp_body("http://10.0.0.5/mcp");
    let body_bytes = serde_json::to_vec(&body_json).unwrap();

    let mut state = ResponsesState::from_request_body(body_json);
    state.previous_response_id = Some("resp_existing".to_owned());
    ctx.extensions.insert(state);

    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        serde_json::json!({"tool": true}),
    );
    write_state(&mut ctx, &body_bytes, map);

    let state = ctx.extensions.get::<ResponsesState>().expect("state should exist");
    assert!(
        state
            .mcp_tool_map
            .contains_key(&("weather".to_owned(), "get_weather".to_owned())),
        "tool should be in map"
    );
    assert_eq!(
        state.previous_response_id.as_deref(),
        Some("resp_existing"),
        "existing fields should be preserved"
    );
}

// =========================================================================
// Allowed Tools Extraction
// =========================================================================

#[test]
fn extract_allowed_tools_string_array() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": ["get_weather", "get_forecast"]
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(
        allowed.as_names().unwrap(),
        &["get_weather", "get_forecast"],
        "should extract allowed_tools"
    );
}

#[test]
fn extract_allowed_tools_filter_object() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": {"tool_names": ["get_weather"]}
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(
        allowed.as_names().unwrap(),
        &["get_weather"],
        "should extract tool_names from filter object"
    );
}

#[test]
fn extract_allowed_tools_read_only() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": {"read_only": true}
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(allowed.read_only, Some(true), "should be read_only");
    assert!(allowed.as_names().is_none(), "should have no name list");
}

#[test]
fn extract_allowed_tools_read_only_with_names() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": {"read_only": true, "tool_names": ["get_weather"]}
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(allowed.read_only, Some(true), "should be read_only");
    assert_eq!(
        allowed.as_names().unwrap(),
        &["get_weather"],
        "should also have name list"
    );
}

#[test]
fn extract_allowed_tools_read_only_false_filters_writable() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": {"read_only": false}
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(
        allowed.read_only,
        Some(false),
        "read_only: false should filter for writable tools"
    );
    assert!(allowed.as_names().is_none(), "should have no name list");
}

#[test]
fn extract_allowed_tools_read_only_false_with_names() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv",
        "allowed_tools": {"read_only": false, "tool_names": ["a"]}
    });
    let allowed = extract_allowed_tools(&entry);
    assert_eq!(
        allowed.read_only,
        Some(false),
        "read_only: false should filter for writable tools"
    );
    assert_eq!(allowed.as_names().unwrap(), &["a"], "should have name list");
}

#[test]
fn extract_allowed_tools_unrestricted_when_absent() {
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "srv"
    });
    assert!(
        extract_allowed_tools(&entry).as_names().is_none(),
        "should be unrestricted when absent"
    );
}

// =========================================================================
// Cache Matching: real mcp_list_tools shape
// =========================================================================

#[test]
fn cache_hit_when_cached_entry_has_no_server_url() {
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "tools": [{"name": "get_weather"}]
    })];
    let allowed = vec!["get_weather".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", "http://10.0.0.5/mcp", Some(&allowed));
    assert!(
        result.is_some(),
        "real mcp_list_tools items lack server_url; label-only match"
    );
}

#[test]
fn cache_miss_when_same_label_different_server_url() {
    let previous = vec![serde_json::json!({
        "server_label": "weather",
        "server_url": "http://10.0.0.1/mcp",
        "tools": [{"name": "get_weather"}]
    })];
    let allowed = vec!["get_weather".to_owned()];

    let result = find_cached_listing(Some(&previous), "weather", "http://10.0.0.99/mcp", Some(&allowed));
    assert!(
        result.is_none(),
        "different server_url with same label should not match"
    );
}

// =========================================================================
// Duplicate Tool Names Across Servers
// =========================================================================

#[test]
fn duplicate_tool_name_across_servers_accepted() {
    let tools = vec![serde_json::json!({"name": "shared_tool"})];
    let mut tool_map: HashMap<(String, String), serde_json::Value> = HashMap::new();

    let entry_a = serde_json::json!({"server_label": "server_a", "server_url": "http://10.0.0.1/mcp"});
    insert_tools(tools.clone(), &entry_a, &mut tool_map);

    let entry_b = serde_json::json!({"server_label": "server_b", "server_url": "http://10.0.0.2/mcp"});
    insert_tools(tools, &entry_b, &mut tool_map);

    assert_eq!(
        tool_map.len(),
        2,
        "same tool name from different servers should coexist"
    );
    assert!(tool_map.contains_key(&("server_a".to_owned(), "shared_tool".to_owned())));
    assert!(tool_map.contains_key(&("server_b".to_owned(), "shared_tool".to_owned())));
}

// =========================================================================
// has_entry_credentials
// =========================================================================

#[test]
fn has_credentials_with_authorization() {
    let entry = serde_json::json!({"server_label": "s", "server_url": "http://10.0.0.1/mcp", "authorization": "tok"});
    assert!(has_entry_credentials(&entry));
}

#[test]
fn has_credentials_with_headers() {
    let entry =
        serde_json::json!({"server_label": "s", "server_url": "http://10.0.0.1/mcp", "headers": {"x-key": "val"}});
    assert!(has_entry_credentials(&entry));
}

#[test]
fn no_credentials_without_auth_or_headers() {
    let entry = serde_json::json!({"server_label": "s", "server_url": "http://10.0.0.1/mcp"});
    assert!(!has_entry_credentials(&entry));
}

#[test]
fn no_credentials_with_empty_headers() {
    let entry = serde_json::json!({"server_label": "s", "server_url": "http://10.0.0.1/mcp", "headers": {}});
    assert!(!has_entry_credentials(&entry));
}

#[test]
fn same_server_different_auth_both_detected() {
    let url = "http://10.0.0.1/mcp";
    let entry_a = serde_json::json!({"server_label": "s", "server_url": url, "authorization": "tok_a"});
    let entry_b = serde_json::json!({"server_label": "s", "server_url": url, "authorization": "tok_b"});
    assert!(has_entry_credentials(&entry_a), "entry A has credentials");
    assert!(has_entry_credentials(&entry_b), "entry B has credentials");
}

// =========================================================================
// insert_tools preserves dispatch config
// =========================================================================

#[test]
fn insert_tools_preserves_authorization_and_require_approval() {
    let tools = vec![serde_json::json!({"name": "run_query"})];
    let entry = serde_json::json!({
        "type": "mcp",
        "server_label": "db",
        "server_url": "http://10.0.0.5/mcp",
        "authorization": "tok_secret",
        "require_approval": "always",
        "headers": {"x-custom": "val"}
    });
    let mut tool_map: HashMap<(String, String), serde_json::Value> = HashMap::new();
    insert_tools(tools, &entry, &mut tool_map);

    let val = &tool_map[&("db".to_owned(), "run_query".to_owned())];
    assert_eq!(
        val.get("authorization").and_then(serde_json::Value::as_str),
        Some("tok_secret"),
        "authorization must be preserved for dispatch"
    );
    assert_eq!(
        val.get("require_approval").and_then(serde_json::Value::as_str),
        Some("always"),
        "require_approval must be preserved for dispatch"
    );
}

// =========================================================================
// connector_id pass-through: body preserved
// =========================================================================

#[tokio::test]
async fn connector_id_entry_preserves_body() {
    let filter = McpToolResolveFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "previous_response_id": "resp_abc",
        "tools": [{"type": "mcp", "server_label": "remote",
                    "connector_id": "mcp_conn_abc123"}]
    });
    let body_bytes = serde_json::to_vec(&body_json).unwrap();
    let mut body = Some(Bytes::from(body_bytes.clone()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "connector_id entry should continue"
    );
    assert_eq!(
        body.as_ref().map(AsRef::as_ref),
        Some(body_bytes.as_slice()),
        "body should be unchanged"
    );
}

// =========================================================================
// Distinct Server Counting
// =========================================================================

#[test]
fn count_distinct_servers_dedupes_same_label_url() {
    let entries = vec![
        serde_json::json!({"server_label": "a", "server_url": "http://10.0.0.1/mcp"}),
        serde_json::json!({"server_label": "a", "server_url": "http://10.0.0.1/mcp", "allowed_tools": ["x"]}),
        serde_json::json!({"server_label": "b", "server_url": "http://10.0.0.2/mcp"}),
    ];
    assert_eq!(
        count_distinct_servers(&entries),
        2,
        "same (label, url) should count once"
    );
}

#[test]
fn count_distinct_servers_excludes_connector_and_deferred() {
    let entries = vec![
        serde_json::json!({"server_label": "a", "connector_id": "conn_1"}),
        serde_json::json!({"server_label": "b", "server_url": "http://10.0.0.1/mcp", "defer_loading": true}),
        serde_json::json!({"server_label": "c", "server_url": "http://10.0.0.2/mcp"}),
    ];
    assert_eq!(
        count_distinct_servers(&entries),
        1,
        "only resolvable entries should be counted"
    );
}

/// Apply `allowed_tools` filtering and insert results into the
/// tool map — mirrors the per-entry flow in `resolve_all_entries`.
fn filter_and_insert(
    tools: Vec<serde_json::Value>,
    entry: &serde_json::Value,
    tool_map: &mut HashMap<(String, String), serde_json::Value>,
) {
    let allowed = extract_allowed_tools(entry);
    insert_tools(apply_allowed_tools_filter(tools, &allowed), entry, tool_map);
}

#[test]
fn same_server_different_allowed_tools_both_inserted() {
    let tools = vec![
        serde_json::json!({"name": "tool_a"}),
        serde_json::json!({"name": "tool_b"}),
    ];
    let mut tool_map: HashMap<(String, String), serde_json::Value> = HashMap::new();

    let entry_a = serde_json::json!({
        "server_label": "s", "server_url": "http://10.0.0.1/mcp",
        "allowed_tools": ["tool_a"], "require_approval": "always"
    });
    let entry_b = serde_json::json!({
        "server_label": "s", "server_url": "http://10.0.0.1/mcp",
        "allowed_tools": ["tool_b"], "require_approval": "never"
    });

    filter_and_insert(tools.clone(), &entry_a, &mut tool_map);
    filter_and_insert(tools, &entry_b, &mut tool_map);

    let key_a = ("s".to_owned(), "tool_a".to_owned());
    let key_b = ("s".to_owned(), "tool_b".to_owned());
    assert!(tool_map.contains_key(&key_a), "tool_a from entry A should be present");
    assert!(tool_map.contains_key(&key_b), "tool_b from entry B should be present");
    assert_eq!(
        tool_map[&key_b]
            .get("require_approval")
            .and_then(serde_json::Value::as_str),
        Some("never"),
        "entry B's config should be on tool_b"
    );
}

// =========================================================================
// Max Servers Cap
// =========================================================================

#[tokio::test]
async fn max_servers_exceeded_returns_rejection() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_servers: 1").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [
            {"type": "mcp", "server_label": "a", "server_url": "http://10.0.0.1/mcp"},
            {"type": "mcp", "server_label": "b", "server_url": "http://10.0.0.2/mcp"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "exceeding max_servers should produce a rejection"
    );
}

#[tokio::test]
async fn connector_only_entries_not_counted_against_max_servers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_servers: 1").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [
            {"type": "mcp", "server_label": "a", "connector_id": "conn_1"},
            {"type": "mcp", "server_label": "b", "connector_id": "conn_2"}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "connector-only entries should not count against max_servers"
    );
}

#[tokio::test]
async fn deferred_entries_not_counted_against_max_servers() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_servers: 1").unwrap();
    let filter = McpToolResolveFilter::from_config(&yaml).unwrap();
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    ctx.set_metadata("openai_tool_parse.has_mcp", "true");

    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [
            {"type": "mcp", "server_label": "a", "server_url": "http://10.0.0.1/mcp", "defer_loading": true},
            {"type": "mcp", "server_label": "b", "server_url": "http://10.0.0.2/mcp", "defer_loading": true}
        ]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&body_json).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "deferred entries should not count against max_servers"
    );
}

// =========================================================================
// previous_response_id preservation
// =========================================================================

#[test]
fn write_state_skips_state_creation_with_previous_response_id() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);
    let body_json = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "previous_response_id": "resp_abc",
        "tools": [{"type": "mcp", "server_label": "w", "server_url": "http://10.0.0.5/mcp"}]
    });
    let body_bytes = serde_json::to_vec(&body_json).unwrap();

    let mut map = HashMap::new();
    map.insert(
        ("w".to_owned(), "get_weather".to_owned()),
        serde_json::json!({"tool": true}),
    );
    write_state(&mut ctx, &body_bytes, map);

    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("state should be created even with previous_response_id");
    assert!(!state.mcp_tool_map.is_empty(), "tool map should be populated");
}

// =========================================================================
// Body Rewrite: MCP to Function
// =========================================================================

/// Build a tool map with a single `(weather, get_weather)` entry.
fn weather_tool_map() -> HashMap<(String, String), serde_json::Value> {
    let mut map = HashMap::new();
    map.insert(
        ("weather".to_owned(), "get_weather".to_owned()),
        serde_json::json!({
            "server_label": "weather",
            "tool_definition": {
                "name": "get_weather",
                "description": "Get current weather",
                "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }),
    );
    map
}

/// Build a [`Resolution`] from raw MCP definitions, pre-building
/// function tools for `per_entry` and dispatch entries for
/// `tool_map`, using `"weather"` as the server label.
fn make_resolution(raw_per_entry: &[Vec<serde_json::Value>]) -> Resolution {
    let label = "weather";
    let mut tool_map = HashMap::new();
    let mut per_entry = Vec::new();
    for raw_tools in raw_per_entry {
        let function_tools: Vec<serde_json::Value> = raw_tools
            .iter()
            .map(|def| mcp_tool_to_function_tool(label, def))
            .collect();
        for tool in raw_tools {
            let name = tool
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown");
            tool_map.insert(
                (label.to_owned(), name.to_owned()),
                serde_json::json!({"server_label": label, "tool_definition": tool}),
            );
        }
        per_entry.push(function_tools);
    }
    Resolution { per_entry, tool_map }
}

/// Pre-built function tools in `per_entry` are moved into the
/// rewritten array; non-MCP tools pass through unchanged.
#[test]
fn rewrite_tools_array_converts_mcp_to_function() {
    let tools = vec![
        serde_json::json!({"type": "function", "name": "calc"}),
        serde_json::json!({"type": "mcp", "server_label": "weather"}),
    ];
    let raw = vec![vec![serde_json::json!({
        "name": "get_weather",
        "description": "Get current weather",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}
    })]];
    let resolution = make_resolution(&raw);
    let (rewritten, generated) = rewrite_tools_array(tools, resolution.per_entry);

    assert_eq!(rewritten.len(), 2, "should have 2 tools");
    assert_eq!(rewritten[0]["name"], "calc", "first tool unchanged");
    assert_eq!(rewritten[1]["type"], "function", "MCP converted to function");
    assert_eq!(rewritten[1]["name"], "weather__get_weather", "prefixed name");
    assert_eq!(
        rewritten[1]["description"], "Get current weather",
        "description preserved from MCP definition"
    );
    assert!(
        rewritten[1]["parameters"]["properties"]["city"].is_object(),
        "inputSchema mapped to parameters"
    );
    assert!(
        generated.contains("weather__get_weather"),
        "generated names should track rewritten tools"
    );
}

/// Unresolved MCP entries (empty `per_entry`) are left unchanged.
#[test]
fn rewrite_tools_array_preserves_unresolved_mcp() {
    let tools = vec![serde_json::json!({
        "type": "mcp",
        "server_label": "unknown",
        "server_url": "http://10.0.0.99/mcp"
    })];

    let (rewritten, generated) = rewrite_tools_array(tools, vec![Vec::new()]);

    assert_eq!(rewritten.len(), 1, "should preserve unresolved entry");
    assert_eq!(rewritten[0]["type"], "mcp", "unresolved MCP left unchanged");
    assert!(generated.is_empty(), "no generated names for unresolved");
}

/// Missing `inputSchema` defaults to `{"type":"object"}`.
#[test]
fn mcp_tool_to_function_tool_adds_default_parameters() {
    let definition = serde_json::json!({"name": "simple_tool", "description": "Does something"});

    let function_tool = mcp_tool_to_function_tool("srv", &definition);

    assert_eq!(function_tool["type"], "function", "type set to function");
    assert_eq!(
        function_tool["name"], "srv__simple_tool",
        "name encoded with label prefix"
    );
    assert_eq!(function_tool["description"], "Does something", "description preserved");
    assert_eq!(
        function_tool["parameters"]["type"], "object",
        "default parameters when inputSchema absent"
    );
}

/// A single MCP entry resolving to multiple tools expands into
/// multiple function tools in the output.
#[test]
fn rewrite_tools_array_expands_multiple_tools() {
    let tools = vec![serde_json::json!({
        "type": "mcp",
        "server_label": "math",
        "server_url": "http://10.0.0.5/mcp"
    })];

    let per_entry = vec![vec![
        mcp_tool_to_function_tool(
            "math",
            &serde_json::json!({"name": "add", "description": "Add numbers"}),
        ),
        mcp_tool_to_function_tool(
            "math",
            &serde_json::json!({"name": "subtract", "description": "Subtract numbers"}),
        ),
    ]];

    let (rewritten, generated) = rewrite_tools_array(tools, per_entry);

    assert_eq!(rewritten.len(), 2, "one MCP entry expands to multiple function tools");
    let names: Vec<&str> = rewritten.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"math__add"), "should have add tool");
    assert!(names.contains(&"math__subtract"), "should have subtract tool");
    assert_eq!(generated.len(), 2, "both names tracked as generated");
}

// =========================================================================
// Body Rewrite: input_schema fallback (cached tool format)
// =========================================================================

/// Falls back to `input_schema` when `inputSchema` is absent
/// (cached API listing format).
#[test]
fn mcp_tool_to_function_tool_reads_input_schema_snake_case() {
    let definition = serde_json::json!({
        "name": "cached_tool",
        "description": "Cached from API",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    });

    let function_tool = mcp_tool_to_function_tool("srv", &definition);

    assert_eq!(function_tool["type"], "function", "type set to function");
    assert_eq!(function_tool["name"], "srv__cached_tool", "name encoded");
    assert_eq!(
        function_tool["parameters"]["properties"]["q"]["type"], "string",
        "should read input_schema when inputSchema is absent"
    );
}

/// `inputSchema` takes precedence over `input_schema` when both
/// fields are present.
#[test]
fn mcp_tool_to_function_tool_prefers_input_schema_camel_case() {
    let definition = serde_json::json!({
        "name": "both_schemas",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}},
        "input_schema": {"type": "object", "properties": {"b": {"type": "string"}}}
    });

    let function_tool = mcp_tool_to_function_tool("srv", &definition);

    assert!(
        function_tool["parameters"]["properties"]["a"].is_object(),
        "inputSchema (camelCase) should take precedence"
    );
}

// =========================================================================
// Function Name Encoding
// =========================================================================

/// Invalid characters (dots, slashes) are replaced with
/// underscores to satisfy the `^[a-zA-Z0-9_-]+$` schema.
#[test]
fn encode_function_name_sanitizes_invalid_chars() {
    let name = encode_function_name("my.server", "get/weather");
    assert_eq!(name, "my_server__get_weather", "dots and slashes replaced with _");
}

/// Combined label + tool name exceeding 64 characters is truncated
/// to the `MAX_FUNCTION_NAME_LEN` limit.
#[test]
fn encode_function_name_truncates_long_names() {
    let long_label = "a".repeat(40);
    let long_tool = "b".repeat(40);
    let name = encode_function_name(&long_label, &long_tool);
    assert!(
        name.len() <= MAX_FUNCTION_NAME_LEN,
        "name should be truncated to {} chars, got {}",
        MAX_FUNCTION_NAME_LEN,
        name.len()
    );
}

/// Alphanumeric characters, hyphens, and underscores pass through
/// without sanitization.
#[test]
fn encode_function_name_preserves_valid_chars() {
    let name = encode_function_name("label-1", "tool_2");
    assert_eq!(name, "label-1__tool_2", "valid chars preserved");
}

// =========================================================================
// tool_choice Rewrite
// =========================================================================

/// Named MCP `tool_choice` is translated to a function-typed
/// choice with the encoded name.
#[test]
fn rewrite_tool_choice_translates_mcp_to_function() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({"type": "mcp", "server_label": "weather", "name": "get_weather"}),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let choice = &obj["tool_choice"];
    assert_eq!(choice["type"], "function", "type rewritten to function");
    assert_eq!(choice["name"], "weather__get_weather", "name encoded");
}

/// Non-object `tool_choice` values (like `"auto"`) are left
/// unchanged.
#[test]
fn rewrite_tool_choice_ignores_non_mcp() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert("tool_choice".to_owned(), serde_json::json!("auto"));

    rewrite_tool_choice(&mut obj, &tool_map);

    assert_eq!(obj["tool_choice"], "auto", "non-MCP tool_choice unchanged");
}

/// Missing `tool_choice` field is a no-op.
#[test]
fn rewrite_tool_choice_ignores_missing() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();

    rewrite_tool_choice(&mut obj, &tool_map);

    assert!(obj.get("tool_choice").is_none(), "no tool_choice inserted");
}

/// Named MCP `tool_choice` referencing a tool not in `tool_map`
/// is left unchanged.
#[test]
fn rewrite_tool_choice_skips_unknown_mcp_tool() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({"type": "mcp", "server_label": "weather", "name": "nonexistent"}),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    assert_eq!(
        obj["tool_choice"]["type"], "mcp",
        "unknown MCP tool_choice left unchanged"
    );
}

// =========================================================================
// State Sync: write_state synchronizes request_body and tools
// =========================================================================

/// Build a rewritten body with function tools and `tool_choice`
/// already translated from MCP form.
fn rewritten_body_json() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "function", "name": "weather__get_weather", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "name": "weather__get_weather"}
    })
}

/// `write_state` synchronizes `request_body`, `tools`, and
/// `tool_choice` on an existing `ResponsesState`.
#[test]
fn write_state_syncs_request_body_and_tools_on_existing_state() {
    let req = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let mut ctx = crate::test_utils::make_filter_context(&req);

    let original_body = serde_json::json!({
        "model": "gpt-4o", "input": "test",
        "tools": [{"type": "mcp", "server_label": "weather"}],
        "tool_choice": {"type": "mcp", "server_label": "weather", "name": "get_weather"}
    });
    ctx.extensions.insert(ResponsesState::from_request_body(original_body));

    let body_bytes = serde_json::to_vec(&rewritten_body_json()).unwrap();
    write_state(&mut ctx, &body_bytes, weather_tool_map());

    let state = ctx.extensions.get::<ResponsesState>().expect("state should exist");
    assert_eq!(state.tools.len(), 1, "tools synced from rewritten body");
    assert_eq!(state.tools[0]["type"], "function", "synced tool is function type");
    assert_eq!(state.tool_choice["type"], "function", "tool_choice synced");
    assert_eq!(
        state.request_body["tools"][0]["type"], "function",
        "request_body rewritten"
    );
}

// =========================================================================
// Per-entry expansion respects allowed_tools
// =========================================================================

/// Per-entry `per_entry` lists are consumed independently,
/// respecting per-entry `allowed_tools` filtering.
#[test]
fn rewrite_per_entry_respects_allowed_tools_filter() {
    let tools = vec![
        serde_json::json!({"type": "mcp", "server_label": "s", "allowed_tools": ["tool_a"]}),
        serde_json::json!({"type": "mcp", "server_label": "s", "allowed_tools": ["tool_b"]}),
    ];

    let per_entry = vec![
        vec![mcp_tool_to_function_tool(
            "s",
            &serde_json::json!({"name": "tool_a", "description": "A"}),
        )],
        vec![mcp_tool_to_function_tool(
            "s",
            &serde_json::json!({"name": "tool_b", "description": "B"}),
        )],
    ];

    let (rewritten, _) = rewrite_tools_array(tools, per_entry);

    assert_eq!(rewritten.len(), 2, "each entry expands independently");
    assert_eq!(rewritten[0]["name"], "s__tool_a", "first entry only has tool_a");
    assert_eq!(rewritten[1]["name"], "s__tool_b", "second entry only has tool_b");
}

// =========================================================================
// Server-level MCP tool_choice → allowed_tools
// =========================================================================

/// Server-level MCP `tool_choice` `{"type":"mcp","server_label":"X"}`
/// (without `name`) rewrites to an `allowed_tools` form scoped to
/// that server's generated functions.
#[test]
fn rewrite_tool_choice_server_level_becomes_allowed_tools() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({"type": "mcp", "server_label": "weather"}),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let choice = &obj["tool_choice"];
    assert_eq!(
        choice["type"], "allowed_tools",
        "server-level MCP tool_choice should become allowed_tools"
    );
    assert_eq!(choice["mode"], "required", "mode should be required");
    let tool_refs = choice["tools"].as_array().expect("tools should be array");
    assert!(
        tool_refs.iter().all(|t| t["type"] == "function"),
        "all entries should be function type"
    );
    assert!(
        tool_refs.iter().any(|t| t["name"] == "weather__get_weather"),
        "should include weather server's function"
    );
}

/// Server-level MCP `tool_choice` for an unknown server is left
/// unchanged when no tools from that server exist.
#[test]
fn rewrite_tool_choice_server_level_unknown_label_unchanged() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({"type": "mcp", "server_label": "nonexistent"}),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    assert_eq!(
        obj["tool_choice"]["type"], "mcp",
        "unknown server-level MCP tool_choice left unchanged"
    );
}

// =========================================================================
// MCP selectors inside allowed_tools tool_choice
// =========================================================================

/// MCP selectors inside an `allowed_tools` `tool_choice` are
/// expanded to their generated function equivalents.
#[test]
fn rewrite_tool_choice_expands_mcp_selectors_in_allowed_tools() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "mcp", "server_label": "weather"},
                {"type": "function", "name": "calc"}
            ]
        }),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let choice = &obj["tool_choice"];
    assert_eq!(choice["type"], "allowed_tools", "type preserved");
    assert_eq!(choice["mode"], "auto", "mode preserved");
    let tool_refs = choice["tools"].as_array().expect("tools should be array");
    let names: Vec<&str> = tool_refs.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"calc"), "non-MCP selector preserved");
    assert!(
        names.contains(&"weather__get_weather"),
        "MCP selector expanded to function"
    );
}

/// Named MCP selectors inside `allowed_tools` are translated to
/// their specific function equivalent.
#[test]
fn rewrite_tool_choice_translates_named_mcp_selector_in_allowed_tools() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "mcp", "server_label": "weather", "name": "get_weather"}
            ]
        }),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let tool_refs = obj["tool_choice"]["tools"].as_array().expect("tools array");
    assert_eq!(tool_refs.len(), 1, "one selector → one function ref");
    assert_eq!(tool_refs[0]["type"], "function", "translated to function");
    assert_eq!(tool_refs[0]["name"], "weather__get_weather", "name encoded");
}

/// `allowed_tools` `tool_choice` without MCP selectors is left
/// unchanged.
#[test]
fn rewrite_tool_choice_allowed_tools_without_mcp_unchanged() {
    let tool_map = weather_tool_map();
    let original = serde_json::json!({
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "calc"}]
    });
    let mut obj = serde_json::Map::new();
    obj.insert("tool_choice".to_owned(), original.clone());

    rewrite_tool_choice(&mut obj, &tool_map);

    assert_eq!(obj["tool_choice"], original, "no MCP selectors → unchanged");
}

/// An unresolved named MCP selector inside `allowed_tools` is
/// preserved when no matching entry exists in `tool_map`.
#[test]
fn rewrite_tool_choice_preserves_unresolved_named_mcp_selector() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "mcp", "server_label": "weather", "name": "nonexistent"},
                {"type": "function", "name": "calc"}
            ]
        }),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let tool_refs = obj["tool_choice"]["tools"].as_array().expect("tools array");
    assert_eq!(tool_refs.len(), 2, "unresolved selector preserved alongside function");
    let mcp_ref = tool_refs.iter().find(|t| t["type"] == "mcp");
    assert!(mcp_ref.is_some(), "unresolved MCP selector should be preserved");
    assert_eq!(
        mcp_ref.unwrap()["name"],
        "nonexistent",
        "original selector fields preserved"
    );
}

/// An unresolved server-level MCP selector inside `allowed_tools`
/// is preserved when no tools from that server exist in `tool_map`.
#[test]
fn rewrite_tool_choice_preserves_unresolved_server_mcp_selector() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "mcp", "server_label": "unknown_server"}
            ]
        }),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let tool_refs = obj["tool_choice"]["tools"].as_array().expect("tools array");
    assert_eq!(tool_refs.len(), 1, "unresolved server selector preserved");
    assert_eq!(tool_refs[0]["type"], "mcp", "original MCP type preserved");
    assert_eq!(
        tool_refs[0]["server_label"], "unknown_server",
        "original server_label preserved"
    );
}

/// Mixed resolved and unresolved MCP selectors: resolved ones
/// expand, unresolved ones are preserved.
#[test]
fn rewrite_tool_choice_mixed_resolved_and_unresolved_selectors() {
    let tool_map = weather_tool_map();
    let mut obj = serde_json::Map::new();
    obj.insert(
        "tool_choice".to_owned(),
        serde_json::json!({
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "mcp", "server_label": "weather"},
                {"type": "mcp", "server_label": "deferred_server"}
            ]
        }),
    );

    rewrite_tool_choice(&mut obj, &tool_map);

    let choice = &obj["tool_choice"];
    let tool_refs = choice["tools"].as_array().expect("tools array");
    let has_expanded = tool_refs
        .iter()
        .any(|t| t["type"] == "function" && t["name"].as_str().is_some_and(|n| n.starts_with("weather__")));
    assert!(has_expanded, "resolved MCP selector should expand");
    let has_preserved = tool_refs
        .iter()
        .any(|t| t["type"] == "mcp" && t["server_label"] == "deferred_server");
    assert!(has_preserved, "unresolved MCP selector should be preserved");
}

// =========================================================================
// Name Collision Detection
// =========================================================================

/// Duplicate generated function names are detected and rejected.
#[test]
fn detect_name_collisions_rejects_generated_duplicates() {
    let tools = vec![
        serde_json::json!({"type": "function", "name": "my_server__get"}),
        serde_json::json!({"type": "function", "name": "my_server__get"}),
    ];
    let generated = HashSet::from(["my_server__get".to_owned()]);

    let result = detect_name_collisions(&tools, &generated);
    assert!(result.is_err(), "duplicate generated names should be rejected");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("my_server__get"),
        "error should identify the colliding name"
    );
}

/// Distinct function names pass collision detection.
#[test]
fn detect_name_collisions_accepts_distinct_names() {
    let tools = vec![
        serde_json::json!({"type": "function", "name": "server_a__tool"}),
        serde_json::json!({"type": "function", "name": "server_b__tool"}),
    ];
    let generated = HashSet::from(["server_a__tool".to_owned(), "server_b__tool".to_owned()]);

    assert!(
        detect_name_collisions(&tools, &generated).is_ok(),
        "distinct names should pass collision check"
    );
}

/// Non-function tools are skipped during collision detection.
#[test]
fn detect_name_collisions_ignores_non_function_tools() {
    let tools = vec![
        serde_json::json!({"type": "mcp", "server_label": "x"}),
        serde_json::json!({"type": "function", "name": "only_one"}),
    ];
    let generated = HashSet::from(["only_one".to_owned()]);

    assert!(
        detect_name_collisions(&tools, &generated).is_ok(),
        "non-function tools should be skipped"
    );
}

/// Client-supplied duplicate function names are not rejected when
/// no generated name is involved.
#[test]
fn detect_name_collisions_ignores_client_duplicates() {
    let tools = vec![
        serde_json::json!({"type": "function", "name": "client_func"}),
        serde_json::json!({"type": "function", "name": "client_func"}),
    ];
    let generated = HashSet::new();

    assert!(
        detect_name_collisions(&tools, &generated).is_ok(),
        "client-only duplicates should pass — backend validates these"
    );
}

/// A generated name colliding with a client-supplied name is
/// rejected.
#[test]
fn detect_name_collisions_rejects_generated_vs_client_collision() {
    let tools = vec![
        serde_json::json!({"type": "function", "name": "my_func"}),
        serde_json::json!({"type": "function", "name": "my_func"}),
    ];
    let generated = HashSet::from(["my_func".to_owned()]);

    assert!(
        detect_name_collisions(&tools, &generated).is_err(),
        "generated name colliding with client name should be rejected"
    );
}

/// Lossy sanitization can cause collisions when special characters
/// in labels map to the same encoded name.
#[test]
fn encode_function_name_lossy_collision_example() {
    let name_a = encode_function_name("my.server", "get");
    let name_b = encode_function_name("my_server", "get");
    assert_eq!(name_a, name_b, "dots sanitized to underscores create identical names");
}
