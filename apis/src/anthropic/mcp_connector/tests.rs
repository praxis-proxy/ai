// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for the Anthropic MCP connector.

use std::sync::Arc;

use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde_json::json;

use super::{config::*, request::*, response::*, *};
use crate::{
    mcp_core::execute::McpCallOutcome as _,
    test_utils::{capture_log_output, make_filter_context, make_request, make_response},
};

// -----------------------------------------------------------------------------
// Test MCP Server
// -----------------------------------------------------------------------------

/// Input for both test tools.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct CityRequest {
    /// City to act on.
    city: String,
}

/// MCP server exposing a read tool and a write tool.
#[derive(Debug, Clone)]
struct WeatherServer {
    /// Generated tool handlers.
    tool_router: ToolRouter<Self>,
}

#[expect(clippy::unused_self, reason = "rmcp macro-generated code")]
#[tool_router]
impl WeatherServer {
    /// Build the server.
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Read-style tool.
    #[tool(description = "Get current weather for a city")]
    fn get_weather(&self, Parameters(req): Parameters<CityRequest>) -> String {
        format!("sunny in {}", req.city)
    }

    /// Write-style tool.
    #[tool(description = "Delete a city")]
    fn delete_city(&self, Parameters(req): Parameters<CityRequest>) -> String {
        format!("deleted {}", req.city)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for WeatherServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }
}

/// Start the weather server on a loopback port.
async fn start_weather_server() -> (String, tokio_util::sync::CancellationToken) {
    let ct = tokio_util::sync::CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        .with_sse_keep_alive(None)
        .with_cancellation_token(ct.child_token());
    let service: StreamableHttpService<WeatherServer, LocalSessionManager> =
        StreamableHttpService::new(|| Ok(WeatherServer::new()), Arc::default(), config);
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let shutdown = ct.clone();
    tokio::spawn(async move {
        drop(
            axum::serve(listener, router)
                .with_graceful_shutdown(async move { shutdown.cancelled_owned().await })
                .await,
        );
    });
    (format!("http://{addr}/mcp"), ct)
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Parse YAML into a filter config value.
fn yaml(text: &str) -> serde_yaml::Value {
    serde_yaml::from_str(text).unwrap()
}

/// Validated config with one connector at `url`.
fn config_with_connector(url: &str, extra: &str) -> ValidatedConfig {
    let raw: AnthropicMcpConnectorConfig = serde_yaml::from_value(yaml(&format!(
        "connectors:\n  - id: weather\n    server_url: {url}\n    tool_policy: allow_all\n{extra}"
    )))
    .unwrap();
    build_config(raw).unwrap()
}

/// Filter dialing loopback, with one connector at `url`.
fn filter_for(url: &str, extra: &str) -> Box<dyn HttpFilter> {
    AnthropicMcpConnectorFilter::from_config_allow_private(&yaml(&format!(
        "connectors:\n  - id: weather\n    server_url: {url}\n    tool_policy: allow_all\n{extra}"
    )))
    .unwrap()
}

/// Map from a JSON object literal.
fn object(value: Value) -> Map<String, Value> {
    let Value::Object(object) = value else {
        panic!("expected an object");
    };
    object
}

/// A request body using the connector against `url`.
fn connector_body(url: &str, toolset: Value) -> Value {
    json!({
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "mcp_servers": [{"type": "url", "url": url, "name": "weather"}],
        "tools": [{"name": "local", "input_schema": {"type": "object"}}, toolset],
    })
}

/// Body bytes of a rejection.
fn rejection_body(action: &FilterAction) -> (u16, Value) {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected a rejection");
    };
    let body = rejection
        .body
        .as_ref()
        .map(|body| serde_json::from_slice(body).unwrap());
    (rejection.status, body.unwrap_or(Value::Null))
}

/// Connector state with one resolved `weather/get_weather` tool.
fn resolved_state() -> ConnectorState {
    let mut tool_map = HashMap::new();
    insert_tools(
        vec![json!({"name": "get_weather", "inputSchema": {"type": "object"}})],
        &json!({"server_label": "weather", "server_url": "http://127.0.0.1:1/mcp"}),
        &mut tool_map,
    );
    ConnectorState {
        owner: None,
        tool_map: index_resolved_tools(tool_map),
        admission: McpAdmissionDecisions::default(),
        has_connector: false,
        earlier: Vec::new(),
        usage: RoundUsage::default(),
        pending: None,
    }
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

#[test]
fn empty_config_builds() {
    assert!(AnthropicMcpConnectorFilter::from_config(&serde_yaml::Value::Null).is_ok());
}

#[test]
fn config_rejects_invalid_limits_and_headers() {
    for (text, needle) in [
        ("timeout_ms: 0", "timeout_ms"),
        ("max_servers: 0", "max_servers"),
        ("max_result_bytes: 10", "max_result_bytes"),
        ("max_parallel_calls: 0", "max_parallel_calls"),
        ("forward_headers: [authorization]", "forward_headers"),
        ("user_credential: \"\"", "user_credential"),
        ("unknown_field: 1", "unknown"),
    ] {
        let Err(error) = AnthropicMcpConnectorFilter::from_config(&yaml(text)) else {
            panic!("{text} should be rejected");
        };
        assert!(error.to_string().contains(needle), "{text}: {error}");
    }
}

#[test]
fn plain_builtin_rejects_outbound_chain() {
    let Err(error) = AnthropicMcpConnectorFilter::from_config(&yaml("outbound_chain: mcp_out")) else {
        panic!("outbound_chain needs chain binding");
    };
    assert!(error.to_string().contains("chain-binding"), "{error}");
}

#[test]
fn connectors_sharing_a_url_are_rejected() {
    let text = "connectors:\n  - id: a\n    server_url: https://mcp.example.com/mcp\n  - id: b\n    server_url: https://MCP.example.com/mcp";
    let Err(error) = AnthropicMcpConnectorFilter::from_config(&yaml(text)) else {
        panic!("duplicate connector URLs should be rejected");
    };
    assert!(error.to_string().contains("shares its server_url"), "{error}");
}

// -----------------------------------------------------------------------------
// Connector Parsing
// -----------------------------------------------------------------------------

#[test]
fn body_without_connector_parses_to_none() {
    let body = object(json!({"model": "m", "tools": [{"name": "x", "input_schema": {}}]}));
    assert!(parse_connector(&body, 10).unwrap().is_none());
    let body = object(json!({"model": "m", "mcp_servers": []}));
    assert!(parse_connector(&body, 10).unwrap().is_none());
}

#[test]
fn toolset_selection_follows_default_and_per_tool_configs() {
    let allowlist = object(connector_body(
        "https://mcp.example.com/mcp",
        json!({
            "type": "mcp_toolset", "mcp_server_name": "weather",
            "default_config": {"enabled": false},
            "configs": {"get_weather": {"enabled": true}},
        }),
    ));
    let parsed = parse_connector(&allowlist, 10).unwrap().unwrap();
    let toolset = &parsed.toolsets[0];
    assert_eq!(toolset.tools_index, 1);
    assert!(toolset.is_enabled("get_weather"));
    assert!(!toolset.is_enabled("delete_city"));

    let denylist = object(connector_body(
        "https://mcp.example.com/mcp",
        json!({
            "type": "mcp_toolset", "mcp_server_name": "weather",
            "configs": {"delete_city": {"enabled": false}},
        }),
    ));
    let parsed = parse_connector(&denylist, 10).unwrap().unwrap();
    assert!(parsed.toolsets[0].is_enabled("get_weather"));
    assert!(!parsed.toolsets[0].is_enabled("delete_city"));
}

#[test]
fn malformed_connectors_are_rejected_without_echoing_values() {
    let server = |extra: Value| {
        let mut entry = json!({"type": "url", "url": "https://secret-host.example/mcp", "name": "weather"});
        entry.as_object_mut().unwrap().extend(object(extra));
        entry
    };
    let toolset = json!({"type": "mcp_toolset", "mcp_server_name": "weather"});
    let cases = [
        (
            json!({"mcp_servers": [server(json!({"type": "sse"}))], "tools": [toolset]}),
            "type must be",
        ),
        (json!({"mcp_servers": [server(json!({}))]}), "referenced by exactly one"),
        (
            json!({"mcp_servers": [server(json!({}))], "tools": [{"type": "mcp_toolset", "mcp_server_name": "other"}]}),
            "does not match any",
        ),
        (
            json!({"mcp_servers": [server(json!({})), server(json!({}))], "tools": [toolset]}),
            "duplicates",
        ),
        (
            json!({"mcp_servers": [server(json!({}))], "tools": [toolset, toolset]}),
            "already uses",
        ),
        (
            json!({"mcp_servers": [server(json!({}))], "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather", "default_config": {"defer_loading": true}}]}),
            "defer_loading",
        ),
        (
            json!({"mcp_servers": [server(json!({}))], "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather", "tools": []}]}),
            "pinned",
        ),
        (
            json!({"mcp_servers": [server(json!({"tool_configuration": {}}))], "tools": [toolset]}),
            "deprecated",
        ),
        (
            json!({"mcp_servers": [server(json!({"extra": 1}))], "tools": [toolset]}),
            "unsupported field",
        ),
        (json!({"mcp_servers": {}}), "must be an array"),
    ];
    for (body, needle) in cases {
        let error = parse_connector(&object(body.clone()), 10).unwrap_err();
        assert!(error.contains(needle), "{body}: {error}");
        assert!(!error.contains("secret-host"), "error echoed a client value: {error}");
    }
}

#[test]
fn too_many_servers_are_rejected() {
    let body = object(json!({
        "mcp_servers": [
            {"type": "url", "url": "https://a.example/mcp", "name": "a"},
            {"type": "url", "url": "https://b.example/mcp", "name": "b"},
        ],
        "tools": [
            {"type": "mcp_toolset", "mcp_server_name": "a"},
            {"type": "mcp_toolset", "mcp_server_name": "b"},
        ],
    }));
    let error = parse_connector(&body, 1).unwrap_err();
    assert!(error.contains("at most 1"), "{error}");
}

// -----------------------------------------------------------------------------
// Server Gating
// -----------------------------------------------------------------------------

#[test]
fn server_matches_a_connector_by_normalized_url() {
    let config = config_with_connector("https://mcp.example.com", "");
    let server = ServerDecl {
        name: "weather".to_owned(),
        url: "https://MCP.example.com/".to_owned(),
        authorization_token: Some("token".to_owned()),
    };
    let entry = server_entry(0, &server, &config).unwrap();
    assert_eq!(entry["connector_id"], "weather");
    assert_eq!(entry["server_url"], "https://mcp.example.com");
    assert_eq!(entry["server_label"], "weather");
    assert_eq!(entry["authorization"], "token");
}

#[test]
fn unmatched_server_is_rejected_unless_client_servers_are_allowed() {
    let server = ServerDecl {
        name: "other".to_owned(),
        url: "https://elsewhere.example/mcp".to_owned(),
        authorization_token: None,
    };
    let gated = config_with_connector("https://mcp.example.com/mcp", "");
    let error = server_entry(2, &server, &gated).unwrap_err();
    assert!(
        error.contains("mcp_servers[2].url is not an MCP server configured"),
        "{error}"
    );
    assert!(!error.contains("elsewhere"), "{error}");

    let open = config_with_connector("https://mcp.example.com/mcp", "allow_client_servers: true");
    let entry = server_entry(2, &server, &open).unwrap();
    assert_eq!(entry["server_url"], "https://elsewhere.example/mcp");
    assert!(entry.get("connector_id").is_none());
}

// -----------------------------------------------------------------------------
// Tool Rewriting
// -----------------------------------------------------------------------------

#[test]
fn rewrite_replaces_toolset_in_place_and_keeps_cache_control() {
    let mut body = object(json!({
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/mcp", "name": "weather"}],
        "tools": [
            {"name": "first", "input_schema": {"type": "object"}},
            {"type": "mcp_toolset", "mcp_server_name": "weather", "cache_control": {"type": "ephemeral"}},
            {"name": "last", "input_schema": {"type": "object"}},
        ],
    }));
    let parsed = parse_connector(&body, 10).unwrap().unwrap();
    let generated = vec![vec![
        mcp_tool_to_custom_tool(
            "weather",
            &json!({"name": "get_weather", "description": "d", "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}}),
        ),
        mcp_tool_to_custom_tool("weather", &json!({"name": "delete_city"})),
    ]];
    let names = rewrite_tools(&mut body, &parsed.toolsets, generated).unwrap();

    assert_eq!(names, ["weather__get_weather", "weather__delete_city"]);
    assert!(!body.contains_key("mcp_servers"));
    let tools = body["tools"].as_array().unwrap();
    let order: Vec<&str> = tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    assert_eq!(order, ["first", "weather__get_weather", "weather__delete_city", "last"]);
    assert_eq!(tools[1]["description"], "d");
    assert_eq!(tools[1]["input_schema"]["properties"]["city"]["type"], "string");
    assert!(tools[1].get("cache_control").is_none());
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["input_schema"], json!({"type": "object"}));
}

#[test]
fn rewrite_rejects_collision_with_client_tool() {
    let mut body = object(json!({
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/mcp", "name": "weather"}],
        "tools": [
            {"name": "weather__get_weather", "input_schema": {"type": "object"}},
            {"type": "mcp_toolset", "mcp_server_name": "weather"},
        ],
    }));
    let parsed = parse_connector(&body, 10).unwrap().unwrap();
    let generated = vec![vec![mcp_tool_to_custom_tool(
        "weather",
        &json!({"name": "get_weather"}),
    )]];
    let error = rewrite_tools(&mut body, &parsed.toolsets, generated).unwrap_err();
    assert!(error.contains("collides"), "{error}");
}

#[test]
fn rewrite_drops_empty_tools() {
    let mut body = object(json!({
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/mcp", "name": "weather"}],
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather"}],
    }));
    let parsed = parse_connector(&body, 10).unwrap().unwrap();
    rewrite_tools(&mut body, &parsed.toolsets, vec![Vec::new()]).unwrap();
    assert!(!body.contains_key("tools"));
}

#[test]
fn rewrite_and_history_keep_member_order() {
    let mut body = object(json!({
        "model": "m",
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "weather"}],
        "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/mcp", "name": "weather"}],
        "messages": [
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "a", "name": "get_weather", "server_name": "weather", "input": {}},
                {"type": "mcp_tool_result", "tool_use_id": "a", "content": "sunny"},
            ], "id": "kept"},
            {"content": "thanks", "role": "user", "z": 1},
        ],
        "stream": false,
        "extra": 1,
    }));
    let parsed = parse_connector(&body, 10).unwrap().unwrap();
    let generated = vec![vec![mcp_tool_to_custom_tool(
        "weather",
        &json!({"name": "get_weather"}),
    )]];
    rewrite_tools(&mut body, &parsed.toolsets, generated).unwrap();
    convert_mcp_history(&mut body).unwrap();

    let keys = |object: &Map<String, Value>| object.keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        keys(&body),
        ["model", "tools", "messages", "stream", "extra"],
        "removing mcp_servers must not move later members"
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(
        keys(messages[0].as_object().unwrap()),
        ["role", "content", "id"],
        "a split assistant turn keeps its member order"
    );
    assert_eq!(
        keys(messages[1].as_object().unwrap()),
        ["content", "role", "z"],
        "carried results are prepended without moving `content`"
    );
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
}

// -----------------------------------------------------------------------------
// History Replay
// -----------------------------------------------------------------------------

#[test]
fn replayed_mcp_blocks_become_tool_turns() {
    let mut body = object(json!({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "checking"},
            {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "get_weather", "server_name": "weather", "input": {"city": "Paris"}},
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "is_error": false, "content": [{"type": "text", "text": "sunny"}]},
            {"type": "text", "text": "It is sunny."},
        ]},
        {"role": "user", "content": "and Rome?"},
    ]}));
    assert!(has_mcp_history(&body));
    convert_mcp_history(&mut body).unwrap();

    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "mcptoolu_1", "name": "weather__get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "sunny"}]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is sunny."}]},
            {"role": "user", "content": "and Rome?"},
        ])
    );
}

#[test]
fn trailing_mcp_results_merge_into_the_next_user_turn() {
    let mut body = object(json!({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "mcp_tool_use", "id": "a", "name": "get_weather", "server_name": "weather", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "a", "is_error": true, "content": "boom"},
        ]},
        {"role": "user", "content": "try again"},
    ]}));
    convert_mcp_history(&mut body).unwrap();
    assert_eq!(
        body["messages"][2],
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "boom", "is_error": true},
            {"type": "text", "text": "try again"},
        ]})
    );
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
}

#[test]
fn trailing_mcp_results_at_the_end_get_their_own_user_turn() {
    let mut body = object(json!({"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "mcp_tool_use", "id": "a", "name": "get_weather", "server_name": "weather", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "sunny"}]},
        ]},
    ]}));
    convert_mcp_history(&mut body).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"][0]["type"], "tool_result");
}

#[test]
fn orphan_mcp_result_is_rejected() {
    let mut body = object(json!({"messages": [
        {"role": "assistant", "content": [
            {"type": "mcp_tool_result", "tool_use_id": "missing", "content": "x"},
        ]},
    ]}));
    let error = convert_mcp_history(&mut body).unwrap_err();
    assert!(error.contains("answers no earlier mcp_tool_use"), "{error}");
}

// -----------------------------------------------------------------------------
// Results and Responses
// -----------------------------------------------------------------------------

/// Identity for result-builder tests.
fn identity(call_id: &str) -> crate::mcp_core::execute::McpCallIdentity<'_> {
    crate::mcp_core::execute::McpCallIdentity {
        call_id,
        server_label: "weather",
        tool_name: "get_weather",
        arguments: "{}",
        approval_request_id: None,
    }
}

#[test]
fn results_render_for_the_model_and_the_client() {
    let ok = McpResult::success(&identity("t1"), "sunny", false);
    assert_eq!(
        ok.model_block(),
        json!({"type": "tool_result", "tool_use_id": "t1", "content": "sunny"})
    );
    assert_eq!(
        ok.client_block(),
        json!({"type": "mcp_tool_result", "tool_use_id": "t1", "is_error": false, "content": [{"type": "text", "text": "sunny"}]})
    );
    let failed = McpResult::error(&identity("t2"), "boom");
    assert_eq!(failed.model_block()["is_error"], true);
    assert_eq!(failed.client_block()["is_error"], true);
}

#[test]
fn retained_bytes_cover_both_representations() {
    let small = McpResult::success(&identity("t"), "x", false);
    let large = McpResult::success(&identity("t"), &"x".repeat(1_000), false);
    assert!(large.retained_bytes().unwrap() >= 2_000);
    assert!(large.retained_bytes() > small.retained_bytes());
}

#[test]
fn only_tool_use_rounds_with_mcp_calls_loop() {
    let state = resolved_state();
    let index = McpToolIndex::Resolved(&state.tool_map);
    let round = |stop: &str, name: &str| {
        json!({"type": "message", "role": "assistant", "stop_reason": stop,
               "content": [{"type": "tool_use", "id": "t", "name": name, "input": {}}]})
    };
    assert!(round_calls_mcp(
        &round("tool_use", "weather__get_weather"),
        &index,
        &state.admission
    ));
    assert!(round_calls_mcp(
        &round("end_turn", "weather__get_weather"),
        &index,
        &state.admission
    ));
    assert!(!round_calls_mcp(
        &round("max_tokens", "weather__get_weather"),
        &index,
        &state.admission
    ));
    assert!(!round_calls_mcp(&round("tool_use", "local"), &index, &state.admission));
}

#[test]
fn client_content_inlines_results_after_each_mcp_call() {
    let state = resolved_state();
    let index = McpToolIndex::Resolved(&state.tool_map);
    let content = vec![
        json!({"type": "text", "text": "checking"}),
        json!({"type": "tool_use", "id": "t1", "name": "weather__get_weather", "input": {"city": "Paris"}}),
        json!({"type": "tool_use", "id": "t2", "name": "local", "input": {}}),
    ];
    assert!(has_client_tool_use(&content, &index, &state.admission));
    let blocks = client_content(
        &content,
        &index,
        &state.admission,
        vec![McpResult::success(&identity("t1"), "sunny", false)],
    );
    assert_eq!(
        blocks,
        vec![
            json!({"type": "text", "text": "checking"}),
            json!({"type": "mcp_tool_use", "id": "t1", "name": "get_weather", "server_name": "weather", "input": {"city": "Paris"}}),
            json!({"type": "mcp_tool_result", "tool_use_id": "t1", "is_error": false, "content": [{"type": "text", "text": "sunny"}]}),
            json!({"type": "tool_use", "id": "t2", "name": "local", "input": {}}),
        ]
    );
}

#[test]
fn final_response_prepends_earlier_blocks_and_sums_usage() {
    let usage = RoundUsage {
        input_tokens: 10,
        output_tokens: 3,
    };
    let response = object(json!({
        "type": "message", "content": [{"type": "text", "text": "done"}],
        "usage": {"input_tokens": 20, "output_tokens": 5},
    }));
    let rebuilt = final_response(response, vec![json!({"type": "mcp_tool_use", "id": "t"})], usage);
    assert_eq!(rebuilt["content"][0]["type"], "mcp_tool_use");
    assert_eq!(rebuilt["content"][1]["text"], "done");
    assert_eq!(rebuilt["usage"], json!({"input_tokens": 30, "output_tokens": 8}));
}

#[test]
fn beta_tokens_for_the_connector_are_stripped() {
    use crate::anthropic::protocol::strip_mcp_client_betas;
    assert_eq!(strip_mcp_client_betas("files-api-2025-04-14"), None);
    assert_eq!(
        strip_mcp_client_betas("mcp-client-2025-11-20, files-api-2025-04-14").as_deref(),
        Some("files-api-2025-04-14")
    );
    assert_eq!(strip_mcp_client_betas("mcp-client-2026-09-15").as_deref(), Some(""));
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

#[tokio::test]
async fn first_round_resolves_connector_into_plain_tools() {
    let (url, ct) = start_weather_server().await;
    let filter = filter_for(&url, "");
    let mut request = make_request(http::Method::POST, "/v1/messages");
    request.headers.insert(
        "anthropic-beta",
        HeaderValue::from_static("mcp-client-2025-11-20,files-api-2025-04-14"),
    );
    let mut ctx = make_filter_context(&request);
    let toolset = json!({
        "type": "mcp_toolset", "mcp_server_name": "weather",
        "configs": {"delete_city": {"enabled": false}},
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&connector_body(&url, toolset)).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    ct.cancel();

    assert!(matches!(action, FilterAction::Continue), "connector should resolve");
    let rewritten: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert!(rewritten.get("mcp_servers").is_none());
    let names: Vec<&str> = rewritten["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["local", "weather__get_weather"]);
    assert_eq!(
        ctx.filter_metadata.get(MCP_CONNECTOR_METADATA_KEY),
        Some(MCP_CONNECTOR_RESOLVED)
    );
    let state = ctx.extensions.get::<ConnectorState>().unwrap();
    assert!(state.has_connector);
    assert_eq!(state.tool_map.len(), 1);
    assert!(
        ctx.request_headers_to_set
            .iter()
            .any(|(name, value)| name == "anthropic-beta" && value == "files-api-2025-04-14"),
        "the connector beta token should be stripped"
    );
}

#[tokio::test]
async fn unconfigured_server_is_rejected_before_dialing() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let body = connector_body(
        "http://127.0.0.1:1/mcp",
        json!({"type": "mcp_toolset", "mcp_server_name": "weather"}),
    );
    let mut body = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let (status, error) = rejection_body(&action);
    assert_eq!(status, 400);
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(ctx.extensions.get::<ConnectorState>().is_none());
}

#[tokio::test]
async fn streaming_connector_resolves_and_delivers_message_start_before_eof() {
    let (url, ct) = start_weather_server().await;
    let filter = filter_for(&url, "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.current_filter_id = Some(0);
    let mut body = connector_body(&url, json!({"type":"mcp_toolset","mcp_server_name":"weather"}));
    body["stream"] = Value::Bool(true);
    let mut body = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));
    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
    ct.cancel();
    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Streaming);
    let mut chunk = Some(Bytes::from_static(
        b"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
    ));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap(),
        FilterAction::Continue
    ));
    assert!(
        std::str::from_utf8(&chunk.unwrap())
            .unwrap()
            .contains("event: message_start")
    );
}

#[tokio::test]
async fn requests_without_the_connector_pass_through() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let original = Bytes::from_static(
        br#"{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
    );
    let mut body = Some(original.clone());

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body, Some(original), "body must be untouched");
    assert_eq!(ctx.subrequest_response_mode(), SubRequestResponseMode::Streaming);
    assert!(ctx.extensions.get::<ConnectorState>().is_none());
}

#[tokio::test]
async fn unreachable_server_fails_with_api_error() {
    let url = "http://127.0.0.1:1/mcp";
    let filter = filter_for(url, "timeout_ms: 2000");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let body = connector_body(url, json!({"type": "mcp_toolset", "mcp_server_name": "weather"}));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    let (status, error) = rejection_body(&action);
    assert_eq!(status, 502);
    assert_eq!(error["error"]["type"], "api_error");
    assert!(!error["error"]["message"].as_str().unwrap().contains("127.0.0.1"));
}

#[tokio::test]
async fn protected_mode_excludes_a_denied_connector_without_dialing() {
    let url = "http://127.0.0.1:1/mcp";
    let filter = AnthropicMcpConnectorFilter::from_config_allow_private(&yaml(&format!(
        "connectors:\n  - id: weather\n    server_url: {url}\ntool_admission:\n  mode: protected"
    )))
    .unwrap();
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let body = connector_body(url, json!({"type": "mcp_toolset", "mcp_server_name": "weather"}));
    let mut body = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "an excluded server is not a failure"
    );
    let rewritten: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        rewritten["tools"],
        json!([{"name": "local", "input_schema": {"type": "object"}}])
    );
    let state = ctx.extensions.get::<ConnectorState>().unwrap();
    assert!(state.tool_map.is_empty());
    assert!(state.admission.is_denied_encoded("weather__get_weather"));
}

#[tokio::test]
async fn replayed_history_is_converted_without_a_connector() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let body = json!({"model": "m", "max_tokens": 8, "messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "mcp_tool_use", "id": "a", "name": "get_weather", "server_name": "weather", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "a", "content": "sunny"},
            {"type": "text", "text": "sunny"},
        ]},
        {"role": "user", "content": "thanks"},
    ]});
    let mut body = Some(Bytes::from(serde_json::to_vec(&body).unwrap()));

    let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let rewritten: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(rewritten["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(rewritten["messages"][2]["content"][0]["type"], "tool_result");
    assert!(ctx.extensions.get::<ConnectorState>().is_none());
}

#[test]
fn mcp_round_publishes_loop_and_marks_pending() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut response = make_response();
    let mut ctx = make_filter_context(&request);
    ctx.response_header = Some(&mut response);
    ctx.extensions.insert(resolved_state());
    let round = json!({"type": "message", "role": "assistant", "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "t", "name": "weather__get_weather", "input": {"city": "Paris"}}],
        "usage": {"input_tokens": 1, "output_tokens": 1}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&round).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(
        ctx.filter_results
            .get(FILTER_NAME)
            .and_then(|results| results.get("action")),
        Some("loop")
    );
    assert!(matches!(
        ctx.extensions.get::<ConnectorState>().unwrap().pending,
        Some(PendingRound::Buffered)
    ));
}

#[test]
fn final_round_returns_mcp_blocks_from_earlier_rounds() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut response = make_response();
    let mut ctx = make_filter_context(&request);
    ctx.response_header = Some(&mut response);
    let mut state = resolved_state();
    state.earlier = vec![
        json!({"type": "mcp_tool_use", "id": "t", "name": "get_weather", "server_name": "weather", "input": {}}),
        json!({"type": "mcp_tool_result", "tool_use_id": "t", "is_error": false, "content": [{"type": "text", "text": "sunny"}]}),
    ];
    state.usage = RoundUsage {
        input_tokens: 7,
        output_tokens: 2,
    };
    ctx.extensions.insert(state);
    let round = json!({"type": "message", "role": "assistant", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "It is sunny."}],
        "usage": {"input_tokens": 9, "output_tokens": 4}});
    let mut body = Some(Bytes::from(serde_json::to_vec(&round).unwrap()));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    let rebuilt: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let kinds: Vec<&str> = rebuilt["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|block| block["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["mcp_tool_use", "mcp_tool_result", "text"]);
    assert_eq!(rebuilt["usage"], json!({"input_tokens": 16, "output_tokens": 6}));
    assert_eq!(
        ctx.filter_results
            .get(FILTER_NAME)
            .and_then(|results| results.get("action")),
        Some("done")
    );
}

#[test]
fn responses_without_connector_state_are_untouched() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    let original = Bytes::from_static(br#"{"type":"message","content":[]}"#);
    let mut body = Some(original.clone());

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert!(matches!(action, FilterAction::Continue));
    assert_eq!(body, Some(original));
    assert!(!ctx.filter_results.contains_key(FILTER_NAME));
}

#[tokio::test(flavor = "multi_thread")]
async fn authorization_token_never_reaches_logs() {
    let (url, ct) = start_weather_server().await;
    let filter = filter_for(&url, "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut body = connector_body(&url, json!({"type": "mcp_toolset", "mcp_server_name": "weather"}));
    body["mcp_servers"][0]["authorization_token"] = json!("SECRET-TOKEN-MARKER");
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap());

    let handle = tokio::runtime::Handle::current();
    let logs = tokio::task::block_in_place(|| {
        capture_log_output(|| {
            handle.block_on(async {
                let mut ctx = make_filter_context(&request);
                let mut body = Some(bytes);
                let action = filter.on_request_body(&mut ctx, &mut body, true).await.unwrap();
                assert!(matches!(action, FilterAction::Continue));
                let rewritten: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
                assert!(!rewritten.to_string().contains("SECRET-TOKEN-MARKER"));
            });
        })
    });
    ct.cancel();

    assert!(!logs.contains("SECRET-TOKEN-MARKER"), "the token leaked into logs");
}

#[test]
fn stream_state_is_instance_local_and_released_after_a_terminal_error() {
    let filter = filter_for("https://mcp.example.com/mcp", "");
    let request = make_request(http::Method::POST, "/v1/messages");
    let mut ctx = make_filter_context(&request);
    ctx.extensions.insert(resolved_state());
    apply_transport(&mut ctx, true);
    ctx.current_filter_id = Some(3);
    ctx.insert_filter_state(streaming::StreamRound::new(128));
    let prefix = Bytes::from_static(b"data: {\"type\":\"mes");
    let mut body = Some(prefix.clone());
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
        FilterAction::Continue
    ));
    assert!(body.is_none());
    // A different connector instance in the same context never reads the prefix.
    ctx.current_filter_id = Some(4);
    body = Some(prefix.clone());
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(body, Some(prefix));
    ctx.current_filter_id = Some(3);
    body = Some(Bytes::from_static(b"sage_start\",\"message\":{\"id\":\"msg_1\"}}\n\n"));
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, false).unwrap(),
        FilterAction::Continue
    ));
    assert!(
        std::str::from_utf8(body.as_ref().unwrap())
            .unwrap()
            .starts_with("event: message_start")
    );
    body = Some(Bytes::from(vec![b'x'; 129]));
    assert!(filter.on_response_body(&mut ctx, &mut body, false).is_err());
    // The router cancels the upstream before running the final callback.
    body = None;
    assert!(matches!(
        filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
        FilterAction::Continue
    ));
    assert!(
        std::str::from_utf8(body.as_ref().unwrap())
            .unwrap()
            .starts_with("event: error")
    );
    assert!(ctx.get_filter_state::<streaming::StreamRound>().is_none());
}

#[test]
fn fragmented_tool_input_allocation_scales_with_event_size() {
    use crate::anthropic::tool_stream::{BlockPresentation, MessagesStream};
    let allocated = |size| {
        let argument = "é".repeat(size);
        let start = json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"weather__get_weather","input":{}}});
        let delta = json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json": serde_json::to_string(&json!({"city":argument})).unwrap()}});
        let wire = format!(
            "data: {{\"type\":\"message_start\",\"message\":{{}}}}\r\n\r\ndata: {start}\r\n\r\ndata: {delta}\r\n\r\n"
        );
        allocation_counter::measure(|| {
            let mut stream = MessagesStream::new(1 << 20, 1 << 20);
            let mut output = Vec::new();
            for chunk in wire.as_bytes().chunks(17) {
                stream
                    .write_chunk(chunk, false, &mut output, |_| BlockPresentation::Forward)
                    .unwrap();
            }
            assert!(std::str::from_utf8(&output).unwrap().contains(&argument));
        })
        .bytes_total
    };
    let small = allocated(2048);
    let large = allocated(8192);
    assert!(
        large < 5 * small,
        "fourfold input grew allocations from {small} to {large}"
    );
}

#[test]
fn model_replay_preserves_initial_text_and_streamed_citations() {
    use crate::anthropic::tool_stream::{BlockPresentation, MessagesStream};
    let mut stream = MessagesStream::new(8192, 8192);
    let citation = json!({"type":"char_location","cited_text":"Paris","start_char_index":0,"end_char_index":5});
    for frame in [
        json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant"}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"First"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" second"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":citation}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),
        json!({"type":"message_stop"}),
    ] {
        stream
            .write_chunk(format!("data: {frame}\n\n").as_bytes(), false, &mut Vec::new(), |_| {
                BlockPresentation::Forward
            })
            .unwrap();
    }
    assert_eq!(
        stream.reconstruct_message().unwrap()["content"][0],
        json!({"type":"text","text":"First second","citations":[citation]})
    );
}
