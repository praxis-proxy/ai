// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the `client-tool-compat-chat-completions` example config
//! (GitHub issue #1206).
//!
//! These tests drive the composed pipeline end to end: a rich Codex-style
//! Responses client talks to a **function-only Chat Completions backend** through
//! `POST /v1/responses`. The inference step chains
//! `openai_client_tool_compat -> responses_to_chat_completions -> path_rewrite`,
//! so the mock backend receives `POST /v1/chat/completions` with Chat-shaped
//! `function` tools only, while the client keeps talking rich Responses.
//!
//! They assert both directions of the composition:
//!
//! - request phase: rich `custom`, `namespace`, local `shell`, and client-executed `tool_search` declarations are
//!   lowered to private Chat `function` tools (`{"type":"function","function":{"name":..}}`) on the wire the Chat
//!   backend receives — no rich type and no `/v1/responses` path leaks downstream;
//! - response phase: the backend's Chat Completions `tool_calls` are translated back to Responses `function_call` items
//!   by `responses_to_chat_completions` and then restored to their canonical typed items (`custom_tool_call`,
//!   namespaced `function_call`, `shell_call`, `tool_search_call`) by `openai_client_tool_compat`, with the original
//!   rich `tools` echoed back and no private lowered name leaked.
//!
//! A request that declares no rich client tools stays a transparent passthrough,
//! so plain-function Chat traffic is unchanged. A client tool named a reserved
//! hosted call (`file_search` / `web_search`) fails closed with HTTP 400 before
//! any upstream call.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, StatefulCapturingBackend, TempSqlite, build_pipeline, example_config_path, free_port, http_send,
    json_post, parse_body, parse_header, parse_status, patch_yaml, start_proxy,
};

const EXAMPLE: &str = "openai/responses/client-tool-compat-chat-completions.yaml";

// -----------------------------------------------------------------------------
// Config loader
// -----------------------------------------------------------------------------

/// Load the composed example, pointing the proxy at `proxy_port`, the Chat
/// backend at `model_port`, and the response store at a private temp SQLite
/// database so tests do not share persisted state.
fn load_config(proxy_port: u16, model_port: u16, db_url: &str) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read composed example");
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse composed config")
}

/// Like [`load_config`] but with `openai_stream_events` removed from the inference
/// step, so the streaming-restoration owner is never armed. Exercises the
/// missing-owner fail-closed path (#1159) under the composed pipeline.
fn load_config_without_stream_owner(proxy_port: u16, model_port: u16, db_url: &str) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read composed example");
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db_url);
    // Drop the whole `openai_stream_events` block (the filter line plus its
    // config/comment lines) up to the next `- filter:` entry, so the remaining
    // filters keep their order and the YAML stays well formed.
    let mut skipping = false;
    let yaml = yaml
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            if trimmed == "- filter: openai_stream_events" {
                skipping = true;
                return false;
            }
            if skipping {
                if trimmed.starts_with("- filter:") {
                    skipping = false;
                    return true;
                }
                return false;
            }
            true
        })
        .collect::<Vec<_>>()
        .join("\n");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse composed config without stream owner")
}

/// One Chat Completions success carrying a single function `tool_call`.
fn chat_tool_call_response(id: &str, call_id: &str, name: &str, arguments: &str) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "model": "gpt-4.1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    })
    .to_string()
}

// -----------------------------------------------------------------------------
// Pipeline build
// -----------------------------------------------------------------------------

#[test]
fn example_config_builds_pipeline() {
    let db = TempSqlite::new("compat_chat_build");
    let config = load_config(free_port(), 19961, db.url());
    let _pipeline = build_pipeline(&config);
}

// -----------------------------------------------------------------------------
// Round trip: custom tool lowered to a Chat function, restored from Chat tool_calls
// -----------------------------------------------------------------------------

/// A `custom` client tool is lowered to a Chat `function` the backend accepts, and
/// the backend's Chat Completions `tool_calls` reply is translated to a Responses
/// `function_call` and then restored to a `custom_tool_call` with the original
/// `tools` echoed back — all while the client only ever speaks `POST /v1/responses`
/// and the backend only ever speaks `POST /v1/chat/completions`.
#[test]
fn custom_tool_round_trip_over_chat_backend() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response(
            "chatcmpl_custom",
            "call_abc",
            "apply_patch",
            r#"{"input":"*** Begin Patch"}"#,
        ),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_custom");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "custom round trip should return 200: {raw}");

    // Request phase: the backend saw a Chat Completions request with a lowered
    // Chat `function` tool, never a `custom` tool or a `/v1/responses` path.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "one inference round for a single client tool call");
    assert_eq!(model_reqs[0].method, "POST", "backend is called with POST");
    assert_eq!(
        model_reqs[0].uri, "/v1/chat/completions",
        "the composed pipeline rewrites the path to the Chat Completions endpoint"
    );
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    assert!(
        backend_body.get("input").is_none(),
        "Responses-only field `input` must not reach the Chat backend: {backend_body}"
    );
    assert!(
        backend_body["messages"].is_array(),
        "the Chat backend receives translated messages: {backend_body}"
    );
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(backend_tools.len(), 1, "one tool lowered: {backend_body}");
    assert_eq!(
        backend_tools[0]["type"], "function",
        "custom lowered to a Chat function: {backend_body}"
    );
    assert_eq!(
        backend_tools[0]["function"]["name"], "apply_patch",
        "lowered Chat function name preserved: {backend_body}"
    );
    assert!(
        backend_tools.iter().all(|tool| tool["type"] != "custom"),
        "no rich custom tool may reach the Chat backend: {backend_body}"
    );

    // Response phase: the client sees the restored typed item and the echoed
    // original rich tool.
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output.len(), 1, "one restored output item: {response}");
    assert_eq!(
        output[0]["type"], "custom_tool_call",
        "the translated function_call is restored to a custom_tool_call: {response}"
    );
    assert_eq!(output[0]["name"], "apply_patch", "restored custom name");
    assert_eq!(
        output[0]["call_id"], "call_abc",
        "restored call_id from the Chat tool_call id"
    );
    assert_eq!(
        output[0]["input"], "*** Begin Patch",
        "custom input unwrapped from the lowered arguments: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "custom",
        "the client sees its original custom tool, not the lowered function: {response}"
    );
    let body = parse_body(&raw);
    assert!(
        !body.contains("chat.completion"),
        "no raw Chat framing may leak to the Responses client: {body}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: local shell lowered to a Chat function, restored to a shell_call
// -----------------------------------------------------------------------------

/// A local `shell` client tool is lowered to a Chat `function` named `shell`, and
/// the backend's Chat `tool_calls` reply is restored to a `shell_call` stamped with
/// a `local` environment so it stays client-executed.
#[test]
fn shell_tool_round_trip_over_chat_backend() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response("chatcmpl_shell", "call_sh1", "shell", r#"{"commands":["ls","-la"]}"#),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_shell");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "List the files.",
        "tools": [{"type": "shell", "environment": {"type": "local"}}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "shell round trip should return 200: {raw}");

    let model_reqs = model.requests();
    assert_eq!(model_reqs[0].uri, "/v1/chat/completions", "path rewritten to Chat");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(backend_tools[0]["type"], "function", "shell lowered to a Chat function");
    assert_eq!(
        backend_tools[0]["function"]["name"], "shell",
        "lowered shell name preserved: {backend_body}"
    );
    assert!(
        backend_tools
            .iter()
            .all(|tool| tool["type"] != "shell" && tool["type"] != "local_shell"),
        "no rich shell tool may reach the Chat backend: {backend_body}"
    );

    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(output[0]["type"], "shell_call", "function_call restored to shell_call");
    assert_eq!(output[0]["call_id"], "call_sh1", "restored call_id");
    assert_eq!(
        output[0]["environment"]["type"], "local",
        "the restored shell call is stamped local so it stays client-executed: {response}"
    );
    assert_eq!(
        output[0]["action"]["commands"],
        serde_json::json!(["ls", "-la"]),
        "shell commands round-trip through the lowered arguments: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "shell",
        "the client sees its original shell tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: namespace member lowered to a flat Chat function, restored namespaced
// -----------------------------------------------------------------------------

/// A `namespace` member is lowered to a flat private Chat `function`, and the
/// backend's Chat `tool_calls` reply is restored to its namespaced `function_call`
/// form with the member name and `namespace` recovered and no private lowered name
/// leaked.
#[test]
fn namespace_tool_round_trip_over_chat_backend() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response(
            "chatcmpl_ns",
            "call_ns1",
            "agentic_ns__utils__read_file",
            r#"{"path":"/etc/hosts"}"#,
        ),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_namespace");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Read the hosts file.",
        "tools": [{
            "type": "namespace",
            "name": "utils",
            "description": "Local filesystem utilities.",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }
            }]
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "namespace round trip should return 200: {raw}");

    let model_reqs = model.requests();
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(
        backend_tools[0]["type"], "function",
        "namespace member lowered to a Chat function"
    );
    assert_eq!(
        backend_tools[0]["function"]["name"], "agentic_ns__utils__read_file",
        "namespace member flattened to a private Chat function name: {backend_body}"
    );

    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(
        output[0]["type"], "function_call",
        "a namespaced call restores to a function_call, not a custom item: {response}"
    );
    assert_eq!(output[0]["name"], "read_file", "restored member name");
    assert_eq!(output[0]["namespace"], "utils", "restored namespace");
    assert_eq!(output[0]["call_id"], "call_ns1", "restored call_id");
    let body = parse_body(&raw);
    assert!(
        !body.contains("agentic_ns__"),
        "the private lowered namespace name must never reach the client: {body}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "namespace",
        "the client sees its original namespace tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Round trip: client tool_search lowered to a Chat function, restored client-side
// -----------------------------------------------------------------------------

/// A client-executed `tool_search` tool is lowered to a Chat `function`, and the
/// backend's Chat `tool_calls` reply is restored to a `tool_search_call` with
/// `execution: client` and the original `tool_search` tool echoed back.
#[test]
fn tool_search_round_trip_over_chat_backend() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response("chatcmpl_ts", "call_ts1", "tool_search", r#"{"query":"weather"}"#),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_tool_search");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Find a weather tool.",
        "tools": [{"type": "tool_search"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "tool_search round trip should return 200: {raw}"
    );

    let model_reqs = model.requests();
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");
    assert_eq!(
        backend_tools[0]["type"], "function",
        "tool_search lowered to a Chat function"
    );
    assert_eq!(
        backend_tools[0]["function"]["name"], "tool_search",
        "lowered tool_search name preserved: {backend_body}"
    );

    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(
        output[0]["type"], "tool_search_call",
        "function_call restored to tool_search_call: {response}"
    );
    assert_eq!(output[0]["execution"], "client", "restored client execution marker");
    assert_eq!(output[0]["call_id"], "call_ts1", "restored call_id");
    assert_eq!(
        output[0]["arguments"],
        serde_json::json!({"query": "weather"}),
        "tool_search arguments parsed back to a structured object: {response}"
    );
    let echoed_tools = response["tools"].as_array().expect("echoed tools present");
    assert_eq!(
        echoed_tools[0]["type"], "tool_search",
        "the client sees its original tool_search tool: {response}"
    );
}

// -----------------------------------------------------------------------------
// Request phase: every rich client tool type lowers to a Chat function
// -----------------------------------------------------------------------------

/// A request mixing `custom`, `namespace`, local `shell`, and client-executed
/// `tool_search` declarations reaches the Chat backend as Chat `function` tools
/// only — proving the outbound lowering covers all four rich shapes through the
/// r2c translation.
#[test]
fn all_client_tool_types_lower_to_chat_functions() {
    let empty_completion = serde_json::json!({
        "id": "chatcmpl_empty",
        "object": "chat.completion",
        "model": "gpt-4.1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    });
    let model = StatefulCapturingBackend::new(vec![(200, empty_completion.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_all_types");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Use the available tools.",
        "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Apply a unified diff."},
            {
                "type": "namespace",
                "name": "utils",
                "description": "Local filesystem utilities.",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": false
                    }
                }]
            },
            {"type": "shell", "environment": {"type": "local"}},
            {"type": "tool_search"}
        ]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "mixed rich-tool request should return 200: {raw}"
    );

    let model_reqs = model.requests();
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry lowered tools");

    assert!(
        backend_tools.iter().all(|tool| tool["type"] == "function"),
        "every rich client tool must reach the Chat backend as a Chat function: {backend_body}"
    );
    let names: Vec<&str> = backend_tools
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .collect();
    assert!(names.contains(&"apply_patch"), "custom lowered: {names:?}");
    assert!(
        names.contains(&"agentic_ns__utils__read_file"),
        "namespace member flattened: {names:?}"
    );
    assert!(names.contains(&"shell"), "shell lowered: {names:?}");
    assert!(names.contains(&"tool_search"), "tool_search lowered: {names:?}");
}

// -----------------------------------------------------------------------------
// Native passthrough: a plain function tool is unchanged in both directions
// -----------------------------------------------------------------------------

/// A request that declares only a plain `function` tool is a transparent compat
/// passthrough: the Chat backend receives the tool as a Chat `function` (from the
/// r2c translation) and the client receives the model's `function_call` reply
/// without any typed-item restoration.
#[test]
fn native_function_tool_passes_through_unchanged() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response("chatcmpl_native", "call_weather", "get_weather", r#"{"location":"SF"}"#),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_native");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": false
            }
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "native passthrough should return 200: {raw}");

    let model_reqs = model.requests();
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("backend request body should be JSON");
    let backend_tools = backend_body["tools"]
        .as_array()
        .expect("backend request should carry the function tool");
    assert_eq!(backend_tools.len(), 1, "one tool forwarded: {backend_body}");
    assert_eq!(
        backend_tools[0]["function"]["name"], "get_weather",
        "function name unchanged through the Chat translation: {backend_body}"
    );

    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    let output = response["output"].as_array().expect("output array present");
    assert_eq!(
        output[0]["type"], "function_call",
        "a plain function_call must not be restyled: {response}"
    );
    assert_eq!(output[0]["name"], "get_weather", "function name preserved");
    assert!(
        output.iter().all(|item| item["type"] != "custom_tool_call"),
        "native traffic must not gain a restored typed item: {response}"
    );
}

// -----------------------------------------------------------------------------
// Fail closed: a client tool named a reserved hosted call is rejected (C5, #1206)
// -----------------------------------------------------------------------------

/// A rich client tool named exactly `file_search` or `web_search` collides with a
/// hosted-tool call name a downstream filter re-routes by name, so compat fails
/// closed with HTTP 400 before any upstream call rather than lowering it onto the
/// Chat wire. The backend must never be contacted.
#[test]
fn client_tool_named_reserved_hosted_name_fails_closed() {
    for hosted in ["file_search", "web_search"] {
        let model = StatefulCapturingBackend::new(vec![(
            200,
            chat_tool_call_response("chatcmpl_never", "call_never", hosted, "{}"),
        )])
        .start_with_shutdown();
        let proxy_port = free_port();
        let db = TempSqlite::new(&format!("compat_chat_reserved_{hosted}"));
        let config = load_config(proxy_port, model.port(), db.url());
        let proxy = start_proxy(&config);

        let request = serde_json::json!({
            "model": "gpt-4.1",
            "input": "Use the tool.",
            "tools": [{
                "type": "custom",
                "name": hosted,
                "description": "A client tool that collides with a reserved hosted name."
            }]
        });
        let raw = http_send(
            proxy.addr(),
            &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
        );

        assert_eq!(
            parse_status(&raw),
            400,
            "a client tool named the reserved hosted call `{hosted}` must fail closed: {raw}"
        );
        let body = parse_body(&raw);
        assert!(
            body.contains("reserved") && body.contains(hosted),
            "the rejection names the reserved hosted collision `{hosted}`: {body}"
        );
        assert!(
            model.requests().is_empty(),
            "the backend must never be contacted when the request fails closed (`{hosted}`)"
        );
    }
}

// -----------------------------------------------------------------------------
// Fail closed: streaming rich client tool without the streaming owner
// -----------------------------------------------------------------------------

/// A streaming request that declares a rich `custom` client tool fails closed with
/// HTTP 500 before any upstream call when the `openai_stream_events` logical SSE
/// owner is NOT in the pipeline to restore the lowered calls — an operator
/// misconfiguration. The backend must never be contacted.
#[test]
fn streaming_rich_client_tool_without_stream_owner_fails_closed() {
    let model = StatefulCapturingBackend::new(vec![(
        200,
        chat_tool_call_response("chatcmpl_never", "call_never", "apply_patch", "{}"),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_streaming_no_owner");
    let config = load_config_without_stream_owner(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        500,
        "streaming rich client tools without openai_stream_events must fail closed: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("openai_stream_events"),
        "the rejection names the missing streaming owner: {body}"
    );
    assert!(
        model.requests().is_empty(),
        "the backend must never be contacted when the request fails closed"
    );
}

// -----------------------------------------------------------------------------
// Streaming: a custom client tool is restored live from a Chat SSE tool-call
// -----------------------------------------------------------------------------

/// A streaming request that declares a rich `custom` client tool has the tool
/// lowered to a Chat `function` on the wire; the Chat backend streams a
/// `chat.completion.chunk` tool-call lifecycle that `responses_to_chat_completions`
/// translates into a Responses SSE lifecycle, which `openai_stream_events` restores
/// live to a `custom_tool_call` with the single string parameter unwrapped into the
/// plain-string `input` (#1159). No raw Chat framing and no bare un-restored
/// `function_call` output leaks to the client.
#[test]
fn streaming_custom_tool_restores_over_chat_backend() {
    let chunks = vec![
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"apply_patch\"}}]}}]}\n\n".to_owned(),
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"input\\\":\\\"*** Begin Patch\\\"}\"}}]}}]}\n\n".to_owned(),
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_owned(),
        "data: [DONE]\n\n".to_owned(),
    ];
    let model = Backend::chunked(chunks)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_stream_custom");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Apply the patch.",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a unified diff to the workspace."
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming custom restore should return 200: {raw}"
    );
    assert_eq!(
        parse_header(&raw, "content-type").as_deref(),
        Some("text/event-stream"),
        "streaming response keeps the SSE content type"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("\"type\":\"custom_tool_call\""),
        "the streamed function_call is restored to a custom_tool_call: {body}"
    );
    assert!(
        body.contains("*** Begin Patch"),
        "the custom input is unwrapped and streamed to the client: {body}"
    );
    assert!(
        !body.contains("chat.completion.chunk"),
        "no raw Chat framing may leak to the Responses client: {body}"
    );
}

// -----------------------------------------------------------------------------
// Streaming: a namespace member never leaks its private lowered name over Chat SSE
// -----------------------------------------------------------------------------

/// A streaming request that declares a `namespace` member has the member lowered to
/// a flat private Chat `function` name; the Chat backend streams a tool-call for
/// that private name, `responses_to_chat_completions` translates it, and
/// `openai_stream_events` restores it in place to the bare member name with its
/// `namespace` re-added. The private lowered name must never appear in the streamed
/// body (#1159).
#[test]
fn streaming_namespace_member_never_leaks_over_chat_backend() {
    let chunks = vec![
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_ns1\",\"type\":\"function\",\"function\":{\"name\":\"agentic_ns__utils__read_file\"}}]}}]}\n\n".to_owned(),
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\"}}]}}]}\n\n".to_owned(),
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n".to_owned(),
        "data: [DONE]\n\n".to_owned(),
    ];
    let model = Backend::chunked(chunks)
        .header("content-type", "text/event-stream")
        .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_stream_namespace");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "stream": true,
        "input": "Read the hosts file.",
        "tools": [{
            "type": "namespace",
            "name": "utils",
            "description": "Local filesystem utilities.",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }
            }]
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming namespace restore should return 200: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        !body.contains("agentic_ns__"),
        "the private lowered namespace name must never reach the client on the stream: {body}"
    );
    assert!(
        body.contains("\"name\":\"read_file\""),
        "the restored namespaced call carries the bare member name: {body}"
    );
    assert!(
        body.contains("\"namespace\":\"utils\""),
        "the restored namespaced call re-adds its namespace: {body}"
    );
    assert!(
        !body.contains("chat.completion.chunk"),
        "no raw Chat framing may leak to the Responses client: {body}"
    );
}

// -----------------------------------------------------------------------------
// Continuation: a discovered tool becomes callable on the next Chat turn
// -----------------------------------------------------------------------------

/// A full search -> result -> discovered-call round trip over the Chat backend.
/// Turn 1 lowers a client `tool_search` to a Chat function and restores the
/// backend's Chat tool-call to a `tool_search_call`. The client executes the search
/// and returns a `tool_search_output` carrying a discovered `custom` tool. Turn 2
/// must hoist the discovered tool into the Chat backend's declared functions so a
/// function-only backend can invoke it, and restore the returned Chat tool-call to a
/// `custom_tool_call`.
#[test]
fn discovered_tool_round_trip_over_chat_backend() {
    let model = StatefulCapturingBackend::new(vec![
        (
            200,
            chat_tool_call_response("chatcmpl_ts", "call_ts1", "tool_search", r#"{"query":"apply a patch"}"#),
        ),
        (
            200,
            chat_tool_call_response(
                "chatcmpl_ap",
                "call_ap1",
                "apply_patch",
                r#"{"input":"*** Begin Patch"}"#,
            ),
        ),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let db = TempSqlite::new("compat_chat_discovered");
    let config = load_config(proxy_port, model.port(), db.url());
    let proxy = start_proxy(&config);

    // Turn 1: the client declares tool_search and receives a restored
    // tool_search_call.
    let turn1 = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Find and apply a patch.",
        "tools": [{"type": "tool_search"}]
    });
    let raw1 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&turn1).unwrap()),
    );
    assert_eq!(parse_status(&raw1), 200, "turn 1 should return 200: {raw1}");
    let response1: serde_json::Value = serde_json::from_str(&parse_body(&raw1)).expect("turn 1 response is JSON");
    let search_call = response1["output"][0].clone();
    assert_eq!(
        search_call["type"], "tool_search_call",
        "turn 1 restores a tool_search_call: {response1}"
    );

    // Turn 2: the client executed the search and returns a discovered `custom`
    // tool, then re-declares tool_search for the continuation.
    let turn2 = serde_json::json!({
        "model": "gpt-4.1",
        "input": [
            {"type": "message", "role": "user", "content": "Find and apply a patch."},
            search_call,
            {
                "type": "tool_search_output",
                "call_id": "call_ts1",
                "status": "completed",
                "tools": [{
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a unified diff.",
                    "format": {"type": "text"}
                }]
            }
        ],
        "tools": [{"type": "tool_search"}]
    });
    let raw2 = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&turn2).unwrap()),
    );
    assert_eq!(parse_status(&raw2), 200, "turn 2 should return 200: {raw2}");

    // Request phase of turn 2: the Chat backend must see the discovered
    // `apply_patch` tool declared as a Chat function so it can call it.
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 2, "two inference rounds");
    assert_eq!(model_reqs[1].uri, "/v1/chat/completions", "turn 2 is a Chat request");
    let backend_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("turn 2 backend body is JSON");
    let backend_tools = backend_body["tools"].as_array().expect("turn 2 backend tools");
    let hoisted = backend_tools
        .iter()
        .find(|tool| tool["function"]["name"] == "apply_patch")
        .unwrap_or_else(|| panic!("the discovered tool is hoisted for the Chat backend: {backend_body}"));
    assert_eq!(
        hoisted["type"], "function",
        "the discovered custom tool is lowered to a Chat function: {backend_body}"
    );

    // Response phase of turn 2: the returned Chat tool-call is restored to a
    // custom_tool_call for the discovered tool.
    let response2: serde_json::Value = serde_json::from_str(&parse_body(&raw2)).expect("turn 2 response is JSON");
    let output = response2["output"].as_array().expect("turn 2 output array");
    assert_eq!(
        output[0]["type"], "custom_tool_call",
        "the discovered call is restored to a custom_tool_call: {response2}"
    );
    assert_eq!(output[0]["name"], "apply_patch", "restored discovered tool name");
    assert_eq!(output[0]["input"], "*** Begin Patch", "restored freeform input");
}
