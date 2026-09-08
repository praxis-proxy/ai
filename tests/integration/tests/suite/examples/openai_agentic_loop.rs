// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the openai_agentic_loop filter with
//! `iterative_request_router`.
//!
//! These tests verify that IRR, request-supplied MCP resolution,
//! MCP dispatch, and the agentic inference loop function together.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send,
    json_post, parse_body, parse_status, patch_yaml, start_mcp_mock_server_with_config, start_proxy,
};

// -----------------------------------------------------------------------------
// Pipeline Build
// -----------------------------------------------------------------------------

#[test]
fn example_config_builds_pipeline() {
    let config = load_agentic_config(free_port(), 19901);
    let _pipeline = build_pipeline(&config);
}

// -----------------------------------------------------------------------------
// Single-Pass
// -----------------------------------------------------------------------------

#[test]
fn single_pass_completes_through_irr() {
    let response = r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#;
    let model = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = load_agentic_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "single-pass request through IRR should return 200"
    );

    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "model backend should receive one request");
    let model_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("model request body should be valid JSON");
    assert_eq!(
        model_body["parallel_tool_calls"], false,
        "first inference must disable parallel tool calls when the client omits the field"
    );
}

#[test]
fn explicit_false_preserves_original_request_bytes() {
    let response = r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#;
    let model = StatefulCapturingBackend::new(vec![(200, response.to_owned())]).start_with_shutdown();
    let proxy_port = free_port();

    let config = load_agentic_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let body = r#"{ "model": "gpt-4.1", "input": [{"role":"user","content":"Hello"}], "parallel_tool_calls": false }"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200);
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 1, "model backend should receive one request");
    assert_eq!(
        model_reqs[0].body, body,
        "an already-disabled request should retain byte-exact passthrough"
    );
}

#[test]
fn client_function_call_returns_without_server_execution() {
    let function_response = serde_json::json!({
        "id": "resp_client_tool",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_client",
            "call_id": "call_client",
            "name": "get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    });
    let model = StatefulCapturingBackend::new(vec![(
        200,
        serde_json::to_string(&function_response).expect("serialize function response"),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_agentic_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post(
            "/v1/responses",
            &serde_json::to_string(&request).expect("serialize client function request"),
        ),
    );

    assert_eq!(parse_status(&raw), 200);
    let response: serde_json::Value =
        serde_json::from_str(&parse_body(&raw)).expect("client function response should be JSON");
    assert_eq!(response["id"], "resp_client_tool");
    assert_eq!(
        model.requests().len(),
        1,
        "client-side function calls must return to the client without an internal loop"
    );
}

// -----------------------------------------------------------------------------
// IRR Rejection Preservation (regression for #663)
// -----------------------------------------------------------------------------
//
// The agentic-loop filter rejects (400/508) from its `on_response_body` hook,
// which runs inside IRR. These tests assert IRR surfaces that rejection as a
// client-visible status instead of aborting the response body.
// https://github.com/praxis-proxy/ai/issues/663

#[test]
fn multiple_function_calls_returns_client_visible_400() {
    let response = serde_json::json!({
        "id": "resp_parallel_calls",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "get_time",
                "arguments": r#"{"timezone":"PST"}"#,
                "status": "completed"
            }
        ]
    });
    let model = StatefulCapturingBackend::new(vec![(200, response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_agentic_rejection_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        400,
        "IRR must preserve the response-body rejection status: {raw}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("rejection body should be valid JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "openai_agentic_loop supports exactly one function call per round"
    );
    assert_eq!(model.requests().len(), 1, "the rejection must stop iteration");
}

#[test]
fn iteration_limit_returns_client_visible_508() {
    let function_response = |id: &str, call_id: &str| {
        serde_json::json!({
            "id": id,
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": format!("fc_{call_id}"),
                "call_id": call_id,
                "name": "get_weather",
                "arguments": r#"{"location":"SF"}"#,
                "status": "completed"
            }]
        })
        .to_string()
    };
    let model = StatefulCapturingBackend::new(vec![
        (200, function_response("resp_1", "call_1")),
        (200, function_response("resp_2", "call_2")),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_agentic_rejection_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        508,
        "IRR must preserve the iteration-limit rejection status: {raw}"
    );
    let body: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("rejection body should be valid JSON");
    assert_eq!(body["error"]["type"], "server_error");
    assert_eq!(body["error"]["message"], "agentic loop iteration limit exceeded");
    assert_eq!(model.requests().len(), 2, "one loop is allowed before the limit");
}

// -----------------------------------------------------------------------------
// Round-Trip: Resolve MCP → Inference → tools/call → Inference
// -----------------------------------------------------------------------------

#[test]
fn round_trip_captures_tool_and_model_requests() {
    let first_response = serde_json::json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_abc",
            "name": "weather__get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "The weather in SF is 72F and sunny."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": false
                })),
        ],
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "parallel_tool_calls": true,
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": mcp_url,
            "allowed_tools": ["get_weather"],
            "require_approval": "never"
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "round-trip should return 200");
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be valid JSON");
    assert_eq!(
        response["id"], "resp_2",
        "final response should be the second model response"
    );

    // -------------------------------------------------------------------------
    // Assert request-supplied MCP discovery and execution
    // -------------------------------------------------------------------------
    assert!(
        mcp.method_count("tools/list") >= 1,
        "MCP resolver should call tools/list on the request-supplied server"
    );
    assert_eq!(mcp.method_count("tools/call"), 1, "MCP dispatch should call one tool");
    assert_eq!(mcp.last_tool_call_name().as_deref(), Some("get_weather"));

    let mcp_requests = mcp.received_requests();
    let call = mcp_requests
        .iter()
        .find(|request| request.json_rpc_method.as_deref() == Some("tools/call"))
        .expect("MCP server should receive tools/call");
    let call_body: serde_json::Value = serde_json::from_str(&call.body).expect("tools/call body should be JSON");
    assert_eq!(call_body["params"]["arguments"]["location"], "SF");

    // -------------------------------------------------------------------------
    // Assert resolved first request and tool-enriched second request
    // -------------------------------------------------------------------------
    let model_reqs = model.requests();
    assert_eq!(model_reqs.len(), 2, "model backend should receive exactly two requests");

    let first_model_body: serde_json::Value =
        serde_json::from_str(&model_reqs[0].body).expect("first model request body should be valid JSON");
    assert_eq!(
        first_model_body["parallel_tool_calls"], false,
        "first inference must override parallel_tool_calls=true"
    );
    let resolved_tools = first_model_body["tools"]
        .as_array()
        .expect("first model request should contain resolved tools");
    assert!(
        resolved_tools
            .iter()
            .any(|tool| tool["type"] == "function" && tool["name"] == "weather__get_weather"),
        "MCP resolver should expose the request-supplied MCP tool as an encoded function"
    );

    let second_model_req = &model_reqs[1];

    let model_body: serde_json::Value =
        serde_json::from_str(&second_model_req.body).expect("second model request body should be valid JSON");
    let input = model_body["input"]
        .as_array()
        .expect("second model request input should be an array");

    let has_function_call = input.iter().any(|item| item["type"] == "function_call");
    let has_function_call_output = input.iter().any(|item| item["type"] == "function_call_output");
    assert!(
        has_function_call,
        "second model request input should contain a function_call item"
    );
    assert!(
        has_function_call_output,
        "second model request input should contain a function_call_output item"
    );
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("function_call_output should be present");
    assert!(
        function_output["output"]
            .as_str()
            .is_some_and(|output| output.contains("mock result for get_weather")),
        "second inference should receive the MCP tools/call result"
    );

    // -------------------------------------------------------------------------
    // Assert openai_agentic_loop bookkeeping in second request
    // -------------------------------------------------------------------------
    assert_eq!(
        model_body["parallel_tool_calls"], false,
        "openai_agentic_loop must force parallel_tool_calls=false on re-entry"
    );
    assert_eq!(
        model_body["tool_choice"], "auto",
        "openai_agentic_loop must reset tool_choice to auto on re-entry"
    );
}

#[test]
fn streaming_mcp_round_trip_uses_one_logical_sse_response() {
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_stream_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_stream_1",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_stream_1",
                    "call_id": "call_stream_1",
                    "name": "weather__get_weather",
                    "arguments": "",
                    "status": "in_progress"
                },
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.function_call_arguments.delta",
            serde_json::json!({
                "response_id": "resp_stream_1",
                "item_id": "fc_stream_1",
                "output_index": 0,
                "delta": r#"{"location":"SF"}"#,
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.function_call_arguments.done",
            serde_json::json!({
                "response_id": "resp_stream_1",
                "item_id": "fc_stream_1",
                "output_index": 0,
                "arguments": r#"{"location":"SF"}"#,
                "sequence_number": 3
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_stream_1",
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "id": "fc_stream_1",
                        "call_id": "call_stream_1",
                        "name": "weather__get_weather",
                        "arguments": r#"{"location":"SF"}"#,
                        "status": "completed"
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
                },
                "sequence_number": 4
            }),
        ),
    ];
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_stream_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_stream_2",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_stream_2", "role": "assistant", "status": "in_progress", "content": []},
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_stream_2",
                "item_id": "msg_stream_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "The weather in SF is sunny.",
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_stream_2",
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "msg_stream_2",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "The weather in SF is sunny."}]
                    }],
                    "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}
                },
                "sequence_number": 3
            }),
        ),
    ];
    let (model_port, model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": false
                })),
        ],
        ..McpMockConfig::default()
    });
    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "stream": true,
        "store": false,
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": format!("http://127.0.0.1:{}/mcp", mcp.port()),
            "allowed_tools": ["get_weather"],
            "require_approval": "never"
        }]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(
        parse_status(&raw),
        200,
        "streamed agentic request should return 200 (model requests: {}, MCP list: {}, MCP calls: {}): {raw}",
        model_requests
            .lock()
            .expect("model request lock should not be poisoned")
            .len(),
        mcp.method_count("tools/list"),
        mcp.method_count("tools/call"),
    );
    // #985: verify the whole synthesized SSE event history is spec-conformant
    // end to end — correct event types/shapes, one created + one terminal in
    // lifecycle order, ids/indices consistent across rounds, a unique/monotonic/
    // contiguous sequence, no duplicate items, and a terminal snapshot that
    // agrees with the incremental event history.
    let frames = assert_logical_stream_conformance(&body, "resp_stream_1");

    // Exact per-event sequence numbers across the logical stream. The shared
    // check already proved the series is contiguous 0..N, so pinning each named
    // event to its number also fixes the emission order.
    assert_eq!(
        frame_seq(sole_event(&frames, "response.created")),
        0,
        "created seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "function_call")),
        1,
        "function_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.function_call_arguments.delta")),
        2,
        "function_call arguments delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.function_call_arguments.done")),
        3,
        "function_call arguments done seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "mcp_call")),
        4,
        "mcp_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.mcp_call.in_progress")),
        5,
        "mcp_call in_progress seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.mcp_call.completed")),
        6,
        "mcp_call completed seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.done", "mcp_call")),
        7,
        "mcp_call done seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "message")),
        8,
        "resumed message added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.output_text.delta")),
        9,
        "resumed output_text delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.completed")),
        10,
        "terminal seq: {body}"
    );

    // #276: the locally executed MCP call is synthesized as incremental events
    // (it is present in neither upstream stream). It succeeds, so it emits
    // in_progress then completed (never failed), and every event for it shares
    // the reserved output index 1 and item id.
    let mcp_added = item_frame(&frames, "response.output_item.added", "mcp_call");
    let mcp_id = mcp_added.data["item"]["id"]
        .as_str()
        .expect("mcp_call must carry an id")
        .to_owned();
    assert_eq!(mcp_added.data["output_index"], 1, "mcp_call added output_index: {body}");
    for event in ["response.mcp_call.in_progress", "response.mcp_call.completed"] {
        let frame = sole_event(&frames, event);
        assert_eq!(frame.data["output_index"], 1, "{event} output_index: {body}");
        assert_eq!(
            frame.data["item_id"].as_str(),
            Some(mcp_id.as_str()),
            "{event} item_id: {body}"
        );
    }
    let mcp_done = item_frame(&frames, "response.output_item.done", "mcp_call");
    assert_eq!(mcp_done.data["output_index"], 1, "mcp_call done output_index: {body}");
    assert_eq!(
        mcp_done.data["item"]["id"].as_str(),
        Some(mcp_id.as_str()),
        "mcp_call done item id: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.mcp_call.failed"),
        0,
        "a successful MCP call must not fail: {body}"
    );

    // No duplicate output items: three announced (model function call,
    // synthesized MCP call, resumed message) and exactly one MCP done.
    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        3,
        "three items announced: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "one output_item.done: {body}"
    );

    // The resumed model output follows the two tool items at output index 2.
    let message_added = item_frame(&frames, "response.output_item.added", "message");
    assert_eq!(
        message_added.data["output_index"], 2,
        "resumed message output_index: {body}"
    );
    assert_eq!(
        message_added.data["item"]["id"], "msg_stream_2",
        "resumed message id: {body}"
    );
    let text_delta = sole_event(&frames, "response.output_text.delta");
    assert_eq!(text_delta.data["output_index"], 2, "resumed text output_index: {body}");
    assert_eq!(
        text_delta.data["content_index"], 0,
        "resumed text content_index: {body}"
    );
    assert_eq!(
        text_delta.data["item_id"], "msg_stream_2",
        "resumed text item id: {body}"
    );

    // The terminal snapshot agrees with the incremental history item-for-item.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 3, "terminal output must snapshot all three items: {body}");
    assert_eq!(output[0]["type"], "function_call", "terminal[0] type: {body}");
    assert_eq!(output[0]["id"], "fc_stream_1", "terminal[0] id: {body}");
    assert_eq!(output[0]["call_id"], "call_stream_1", "terminal[0] call_id: {body}");
    assert_eq!(
        output[0]["arguments"], r#"{"location":"SF"}"#,
        "terminal[0] arguments: {body}"
    );
    assert_eq!(output[1]["type"], "mcp_call", "terminal[1] type: {body}");
    assert_eq!(
        output[1]["id"].as_str(),
        Some(mcp_id.as_str()),
        "terminal[1] id must match synthesized mcp_call: {body}"
    );
    assert!(
        output[1]["name"]
            .as_str()
            .is_some_and(|name| name.contains("get_weather")),
        "terminal[1] name must be the MCP tool: {body}"
    );
    assert!(
        output[1]["output"]
            .as_str()
            .is_some_and(|text| text.contains("mock result for get_weather")),
        "terminal[1] must carry the MCP result: {body}"
    );
    assert_eq!(output[2]["type"], "message", "terminal[2] type: {body}");
    assert_eq!(output[2]["id"], "msg_stream_2", "terminal[2] id: {body}");
    assert_eq!(
        output[2]["content"][0]["text"], "The weather in SF is sunny.",
        "terminal[2] assistant text: {body}"
    );
    let terminal = sole_event(&frames, "response.completed");
    assert_eq!(
        terminal.data["response"]["id"], "resp_stream_1",
        "terminal logical id: {body}"
    );
    assert_eq!(
        terminal.data["response"]["usage"]["total_tokens"], 41,
        "terminal usage total_tokens: {body}"
    );

    assert_eq!(
        mcp.method_count("tools/call"),
        1,
        "MCP tool should execute exactly once"
    );

    model_thread.join().expect("streaming model thread should finish");
    let second_request: serde_json::Value = {
        let requests = model_requests
            .lock()
            .expect("model request lock should not be poisoned");
        assert_eq!(requests.len(), 2, "IRR should make two streamed model requests");
        serde_json::from_str(&requests[1]).expect("second request should be JSON")
    };
    let input = second_request["input"]
        .as_array()
        .expect("second request input should be an array");
    assert!(
        input.iter().any(|item| item["type"] == "function_call"),
        "second inference should receive the streamed function call"
    );
    assert!(
        input.iter().any(|item| item["type"] == "function_call_output"),
        "second inference should receive the MCP result"
    );

    // #985 (criterion 3): the call identity stays consistent across every round of
    // the logical stream. The model announces the function call, the proxy runs it
    // locally as an mcp_call that reuses that call_id, the terminal snapshot keeps
    // it, and the resumed request feeds the result back keyed by the same call_id.
    let model_call_id = item_frame(&frames, "response.output_item.added", "function_call").data["item"]["call_id"]
        .as_str()
        .expect("model function_call must carry a call_id");
    assert_eq!(model_call_id, "call_stream_1", "model function_call call_id: {body}");
    assert_eq!(
        output[0]["call_id"].as_str(),
        Some(model_call_id),
        "terminal function_call call_id must match the announced call: {body}"
    );
    assert_eq!(
        mcp_id.as_str(),
        model_call_id,
        "synthesized mcp_call id must reuse the model call_id: {body}"
    );
    let function_call_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("resumed request must include the function_call_output");
    assert_eq!(
        function_call_output["call_id"].as_str(),
        Some(model_call_id),
        "resumed function_call_output must be keyed by the same call_id: {body}"
    );
}

#[test]
fn streaming_mcp_failure_synthesizes_failed_progress_in_one_logical_response() {
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_mcp_fail_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_mcp_fail_1",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_fail_1",
                    "call_id": "call_fail_1",
                    "name": "weather__get_weather",
                    "arguments": "",
                    "status": "in_progress"
                },
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.function_call_arguments.delta",
            serde_json::json!({
                "response_id": "resp_mcp_fail_1",
                "item_id": "fc_fail_1",
                "output_index": 0,
                "delta": r#"{"location":"SF"}"#,
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.function_call_arguments.done",
            serde_json::json!({
                "response_id": "resp_mcp_fail_1",
                "item_id": "fc_fail_1",
                "output_index": 0,
                "arguments": r#"{"location":"SF"}"#,
                "sequence_number": 3
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_mcp_fail_1",
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "id": "fc_fail_1",
                        "call_id": "call_fail_1",
                        "name": "weather__get_weather",
                        "arguments": r#"{"location":"SF"}"#,
                        "status": "completed"
                    }],
                    "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
                },
                "sequence_number": 4
            }),
        ),
    ];
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_mcp_fail_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_mcp_fail_2",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_fail_2", "role": "assistant", "status": "in_progress", "content": []},
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_mcp_fail_2",
                "item_id": "msg_fail_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "The weather service is unavailable.",
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_mcp_fail_2",
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "msg_fail_2",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "The weather service is unavailable."}]
                    }],
                    "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}
                },
                "sequence_number": 3
            }),
        ),
    ];
    let (model_port, model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    // The advertised tool is configured to return an `isError: true` result, so
    // the dispatch filter records the mcp_call as failed (non-null `error`).
    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": false
                })),
        ],
        failing_tools: vec!["get_weather".to_owned()],
        ..McpMockConfig::default()
    });
    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "stream": true,
        "store": false,
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": format!("http://127.0.0.1:{}/mcp", mcp.port()),
            "allowed_tools": ["get_weather"],
            "require_approval": "never"
        }]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(
        parse_status(&raw),
        200,
        "streamed failing MCP request should return 200: {raw}"
    );

    // #985: the whole synthesized SSE history stays spec-conformant even when the
    // local tool fails — one logical lifecycle, consistent ids/indices, a
    // contiguous sequence, and a terminal snapshot that agrees with the events.
    let frames = assert_logical_stream_conformance(&body, "resp_mcp_fail_1");

    // Exact per-event sequence numbers: the failed mcp_call emits `failed`
    // (never `completed`) in the terminal slot of its progress lifecycle.
    assert_eq!(
        frame_seq(sole_event(&frames, "response.created")),
        0,
        "created seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "function_call")),
        1,
        "function_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.function_call_arguments.delta")),
        2,
        "function_call arguments delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.function_call_arguments.done")),
        3,
        "function_call arguments done seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "mcp_call")),
        4,
        "mcp_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.mcp_call.in_progress")),
        5,
        "mcp_call in_progress seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.mcp_call.failed")),
        6,
        "mcp_call failed seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.done", "mcp_call")),
        7,
        "mcp_call done seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "message")),
        8,
        "resumed message added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.output_text.delta")),
        9,
        "resumed output_text delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.completed")),
        10,
        "terminal seq: {body}"
    );

    // #276: the failed local mcp_call emits in_progress then failed (never
    // completed), all sharing the reserved output index 1 and its item id.
    let mcp_added = item_frame(&frames, "response.output_item.added", "mcp_call");
    let mcp_id = mcp_added.data["item"]["id"]
        .as_str()
        .expect("mcp_call must carry an id")
        .to_owned();
    assert_eq!(mcp_added.data["output_index"], 1, "mcp_call added output_index: {body}");
    assert_eq!(
        event_count(&frames, "response.mcp_call.completed"),
        0,
        "a failed MCP call must not emit completed: {body}"
    );
    for event in ["response.mcp_call.in_progress", "response.mcp_call.failed"] {
        let frame = sole_event(&frames, event);
        assert_eq!(frame.data["output_index"], 1, "{event} output_index: {body}");
        assert_eq!(
            frame.data["item_id"].as_str(),
            Some(mcp_id.as_str()),
            "{event} item_id: {body}"
        );
    }
    let mcp_done = item_frame(&frames, "response.output_item.done", "mcp_call");
    assert_eq!(mcp_done.data["output_index"], 1, "mcp_call done output_index: {body}");
    assert_eq!(
        mcp_done.data["item"]["id"].as_str(),
        Some(mcp_id.as_str()),
        "mcp_call done item id: {body}"
    );

    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        3,
        "three items announced: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "one output_item.done: {body}"
    );

    // The terminal snapshot carries the failed mcp_call with its non-null error
    // and agrees item-for-item with the incremental history.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 3, "terminal output must snapshot all three items: {body}");
    assert_eq!(output[1]["type"], "mcp_call", "terminal[1] type: {body}");
    assert_eq!(
        output[1]["id"].as_str(),
        Some(mcp_id.as_str()),
        "terminal[1] id: {body}"
    );
    assert!(
        output[1]["error"]
            .as_str()
            .is_some_and(|error| error.contains("mock failure for get_weather")),
        "terminal[1] must carry the MCP error: {body}"
    );
    assert_eq!(output[2]["type"], "message", "terminal[2] type: {body}");
    assert_eq!(
        output[2]["content"][0]["text"], "The weather service is unavailable.",
        "terminal[2] assistant text: {body}"
    );
    assert_eq!(
        sole_event(&frames, "response.completed").data["response"]["id"],
        "resp_mcp_fail_1",
        "terminal logical id: {body}"
    );

    assert_eq!(
        mcp.method_count("tools/call"),
        1,
        "MCP tool should execute exactly once"
    );

    model_thread.join().expect("streaming model thread should finish");

    // #985 (criterion 3): a failed local tool keeps the call identity consistent
    // across rounds too — the failed mcp_call reuses the model's call_id and the
    // resumed request feeds the error result back keyed by the same call_id.
    let second_request: serde_json::Value = {
        let requests = model_requests
            .lock()
            .expect("model request lock should not be poisoned");
        assert_eq!(requests.len(), 2, "IRR should make two streamed model requests");
        serde_json::from_str(&requests[1]).expect("second request should be JSON")
    };
    let input = second_request["input"]
        .as_array()
        .expect("second request input should be an array");
    let model_call_id = item_frame(&frames, "response.output_item.added", "function_call").data["item"]["call_id"]
        .as_str()
        .expect("model function_call must carry a call_id");
    assert_eq!(model_call_id, "call_fail_1", "model function_call call_id: {body}");
    assert_eq!(
        output[0]["call_id"].as_str(),
        Some(model_call_id),
        "terminal function_call call_id must match the announced call: {body}"
    );
    assert_eq!(
        mcp_id.as_str(),
        model_call_id,
        "synthesized mcp_call id must reuse the model call_id: {body}"
    );
    let function_call_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("resumed request must include the function_call_output");
    assert_eq!(
        function_call_output["call_id"].as_str(),
        Some(model_call_id),
        "resumed function_call_output must be keyed by the same call_id: {body}"
    );
}

// -----------------------------------------------------------------------------
// Two consecutive tool rounds: accumulated output and usage (issue #983)
// -----------------------------------------------------------------------------
//
// One client Responses request drives model -> tool -> model -> tool -> model
// final output. Both MCP tools execute exactly once, each result feeds the next
// inference round, the terminal response lists every function call, server-tool
// result, and the final assistant message in stable order, and the reported
// token usage is the exact sum of all three inference rounds. The buffered and
// streaming variants assert the SAME accumulated output and usage to prove the
// two transports are equivalent.
//
// This flow cannot be an inference fixture: the replay harness binds exactly one
// upstream exchange per client turn and rejects MCP callout filters as not
// replay-contained, so a single request that fans out to three upstream rounds
// must be a functional integration test.

/// Per-round token usage as `(input, output, total)`. Distinct values make the
/// accumulated sum unambiguous, and each round's total equals input + output so
/// the summed `total_tokens` (merged as an independent field) also equals the
/// summed input plus output.
const ROUND1_USAGE: (u64, u64, u64) = (11, 4, 15);
const ROUND2_USAGE: (u64, u64, u64) = (22, 5, 27);
const ROUND3_USAGE: (u64, u64, u64) = (33, 6, 39);

/// Usage accumulated across all three inference rounds.
const EXPECTED_INPUT_TOKENS: u64 = ROUND1_USAGE.0 + ROUND2_USAGE.0 + ROUND3_USAGE.0;
const EXPECTED_OUTPUT_TOKENS: u64 = ROUND1_USAGE.1 + ROUND2_USAGE.1 + ROUND3_USAGE.1;
const EXPECTED_TOTAL_TOKENS: u64 = ROUND1_USAGE.2 + ROUND2_USAGE.2 + ROUND3_USAGE.2;

/// Output item `type` values a two-tool-round terminal response must expose, in
/// stable chronological order: each tool round contributes a function call then
/// its server-tool result, followed by the final assistant message.
const EXPECTED_OUTPUT_TYPES: [&str; 5] = ["function_call", "mcp_call", "function_call", "mcp_call", "message"];

/// Final assistant text emitted by the terminal inference round.
const FINAL_TEXT: &str = "SF is 72F and it is 3pm PST.";

#[test]
fn two_tool_rounds_accumulate_output_and_usage() {
    // Round 1: the model asks to call the weather tool.
    let first_response = serde_json::json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_weather",
            "name": "weather__get_weather",
            "arguments": r#"{"location":"SF"}"#,
            "status": "completed"
        }],
        "usage": usage_json(ROUND1_USAGE)
    });
    // Round 2: after the weather result, the model asks to call the time tool.
    let second_response = serde_json::json!({
        "id": "resp_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_2",
            "call_id": "call_time",
            "name": "weather__get_time",
            "arguments": r#"{"timezone":"PST"}"#,
            "status": "completed"
        }],
        "usage": usage_json(ROUND2_USAGE)
    });
    // Round 3: the model emits the final assistant message and the loop exits.
    let final_response = serde_json::json!({
        "id": "resp_3",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": FINAL_TEXT}]
        }],
        "usage": usage_json(ROUND3_USAGE)
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
        (200, serde_json::to_string(&final_response).unwrap()),
    ])
    .start_with_shutdown();

    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(object_schema("location")),
            McpToolFixture::new("get_time")
                .with_description("Get the current time for a timezone")
                .with_input_schema(object_schema("timezone")),
        ],
        ..McpMockConfig::default()
    });

    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model.port());
    let proxy = start_proxy(&config);

    let mcp_url = format!("http://127.0.0.1:{}/mcp", mcp.port());
    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather and time in SF?",
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": mcp_url,
            "allowed_tools": ["get_weather", "get_time"],
            "require_approval": "never"
        }]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "two-round agentic request should return 200: {raw}"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be valid JSON");

    // Three inference rounds ran: model -> tool -> model -> tool -> model.
    let model_reqs = model.requests();
    assert_eq!(
        model_reqs.len(),
        3,
        "model backend should receive exactly three requests"
    );

    // Both tools executed exactly once.
    assert_eq!(mcp.method_count("tools/call"), 2, "exactly two MCP tool calls total");
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        1,
        "get_weather must execute exactly once"
    );
    assert_eq!(mcp.tool_call_count("get_time"), 1, "get_time must execute exactly once");

    // The terminal response lists every function call, server-tool result, and
    // the final message in stable chronological order.
    let output = response["output"].as_array().expect("final response output array");
    let output_types: Vec<&str> = output.iter().map(output_item_type).collect();
    assert_eq!(
        output_types, EXPECTED_OUTPUT_TYPES,
        "terminal output must interleave both tool rounds then the final message: {output:#?}"
    );
    let mcp_calls: Vec<&serde_json::Value> = output.iter().filter(|item| item["type"] == "mcp_call").collect();
    assert!(
        mcp_calls[0]["output"]
            .as_str()
            .is_some_and(|out| out.contains("mock result for get_weather")),
        "first server-tool result must be the weather call: {:#?}",
        mcp_calls[0]
    );
    assert!(
        mcp_calls[1]["output"]
            .as_str()
            .is_some_and(|out| out.contains("mock result for get_time")),
        "second server-tool result must be the time call: {:#?}",
        mcp_calls[1]
    );
    let message = output
        .last()
        .expect("terminal output should end with the assistant message");
    assert_eq!(
        message["content"][0]["text"], FINAL_TEXT,
        "final message text must survive: {message:#?}"
    );

    // The buffered terminal keeps the last round's backend response id.
    assert_eq!(response["id"], "resp_3", "buffered terminal keeps the last round id");

    // Reported usage equals the exact sum of all three inference rounds.
    assert_eq!(
        response["usage"]["input_tokens"], EXPECTED_INPUT_TOKENS,
        "input tokens must sum across rounds"
    );
    assert_eq!(
        response["usage"]["output_tokens"], EXPECTED_OUTPUT_TOKENS,
        "output tokens must sum across rounds"
    );
    assert_eq!(
        response["usage"]["total_tokens"], EXPECTED_TOTAL_TOKENS,
        "total tokens must sum across rounds"
    );

    // Each tool result was supplied to the following inference round.
    let second_input = request_input(&model_reqs[1].body);
    assert!(
        second_input.iter().any(is_weather_result),
        "round 2 input must carry the weather result: {second_input:#?}"
    );
    let third_input = request_input(&model_reqs[2].body);
    assert!(
        third_input.iter().any(is_time_result),
        "round 3 input must carry the time result: {third_input:#?}"
    );
}

#[test]
fn two_tool_rounds_streaming_matches_buffered() {
    // A streamed function-call turn: created -> item added -> arg deltas -> done
    // -> completed (carrying the round's usage). Mirrors the buffered rounds so
    // the streaming and buffered variants accumulate identical output and usage.
    let fc_turn = |resp_id: &str, item_id: &str, call_id: &str, name: &str, args: &str, usage: (u64, u64, u64)| {
        vec![
            sse_event(
                "response.created",
                serde_json::json!({
                    "response": {"id": resp_id, "object": "response", "status": "in_progress", "output": []},
                    "sequence_number": 0
                }),
            ),
            sse_event(
                "response.output_item.added",
                serde_json::json!({
                    "response_id": resp_id,
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": "",
                        "status": "in_progress"
                    },
                    "sequence_number": 1
                }),
            ),
            sse_event(
                "response.function_call_arguments.delta",
                serde_json::json!({
                    "response_id": resp_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "delta": args,
                    "sequence_number": 2
                }),
            ),
            sse_event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "response_id": resp_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "arguments": args,
                    "sequence_number": 3
                }),
            ),
            sse_event(
                "response.completed",
                serde_json::json!({
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "status": "completed",
                        "output": [{
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": args,
                            "status": "completed"
                        }],
                        "usage": usage_json(usage)
                    },
                    "sequence_number": 4
                }),
            ),
        ]
    };
    let final_turn = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_stream_3", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_stream_3",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_stream_3", "role": "assistant", "status": "in_progress", "content": []},
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_stream_3",
                "item_id": "msg_stream_3",
                "output_index": 0,
                "content_index": 0,
                "delta": FINAL_TEXT,
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_stream_3",
                    "object": "response",
                    "status": "completed",
                    "output": [{
                        "type": "message",
                        "id": "msg_stream_3",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": FINAL_TEXT}]
                    }],
                    "usage": usage_json(ROUND3_USAGE)
                },
                "sequence_number": 3
            }),
        ),
    ];
    let (model_port, model_requests, model_thread) = start_streaming_model(vec![
        fc_turn(
            "resp_stream_1",
            "fc_stream_1",
            "call_weather",
            "weather__get_weather",
            r#"{"location":"SF"}"#,
            ROUND1_USAGE,
        ),
        fc_turn(
            "resp_stream_2",
            "fc_stream_2",
            "call_time",
            "weather__get_time",
            r#"{"timezone":"PST"}"#,
            ROUND2_USAGE,
        ),
        final_turn,
    ]);

    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![
            McpToolFixture::new("get_weather")
                .with_description("Get the weather for a location")
                .with_input_schema(object_schema("location")),
            McpToolFixture::new("get_time")
                .with_description("Get the current time for a timezone")
                .with_input_schema(object_schema("timezone")),
        ],
        ..McpMockConfig::default()
    });
    let proxy_port = free_port();
    let config = load_loopback_mcp_config(proxy_port, model_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather and time in SF?",
        "stream": true,
        "store": false,
        "tools": [{
            "type": "mcp",
            "server_label": "weather",
            "server_url": format!("http://127.0.0.1:{}/mcp", mcp.port()),
            "allowed_tools": ["get_weather", "get_time"],
            "require_approval": "never"
        }]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(
        parse_status(&raw),
        200,
        "streamed two-round request should return 200: {raw}"
    );
    assert_eq!(
        body.matches("event: response.created").count(),
        1,
        "the whole loop must expose one logical response.created: {body}"
    );
    assert_eq!(
        body.matches("event: response.completed").count(),
        1,
        "intermediate round completions must be suppressed: {body}"
    );
    assert!(
        body.contains(FINAL_TEXT),
        "the terminal inference text should reach the stream: {body}"
    );
    assert!(
        !body.contains("resp_stream_2") && !body.contains("resp_stream_3"),
        "resumed rounds must retain the first logical response id: {body}"
    );

    // Both tools executed exactly once.
    assert_eq!(mcp.method_count("tools/call"), 2, "exactly two MCP tool calls total");
    assert_eq!(
        mcp.tool_call_count("get_weather"),
        1,
        "get_weather must execute exactly once"
    );
    assert_eq!(mcp.tool_call_count("get_time"), 1, "get_time must execute exactly once");

    // The single terminal response.completed carries the accumulated output and
    // usage — the same order and sums as the buffered variant.
    let completed = extract_completed_response(&body);
    assert_eq!(completed["id"], "resp_stream_1", "streaming keeps the first round id");
    let output = completed["output"].as_array().expect("terminal streamed output array");
    let output_types: Vec<&str> = output.iter().map(output_item_type).collect();
    assert_eq!(
        output_types, EXPECTED_OUTPUT_TYPES,
        "streamed terminal output must match the buffered accumulated order: {output:#?}"
    );
    assert_eq!(
        completed["usage"]["input_tokens"], EXPECTED_INPUT_TOKENS,
        "streamed input tokens must sum across rounds"
    );
    assert_eq!(
        completed["usage"]["output_tokens"], EXPECTED_OUTPUT_TOKENS,
        "streamed output tokens must sum across rounds"
    );
    assert_eq!(
        completed["usage"]["total_tokens"], EXPECTED_TOTAL_TOKENS,
        "streamed total tokens must sum across rounds"
    );

    model_thread.join().expect("streaming model thread should finish");
    let (second_input, third_input) = {
        let requests = model_requests
            .lock()
            .expect("model request lock should not be poisoned");
        assert_eq!(requests.len(), 3, "IRR should make three streamed model requests");
        (request_input(&requests[1]), request_input(&requests[2]))
    };
    assert!(
        second_input.iter().any(is_weather_result),
        "round 2 input must carry the weather result: {second_input:#?}"
    );
    assert!(
        third_input.iter().any(is_time_result),
        "round 3 input must carry the time result: {third_input:#?}"
    );
}

// -----------------------------------------------------------------------------
// Round-Trip: Web Search via IRR
// -----------------------------------------------------------------------------

#[test]
fn web_search_round_trip_executes_and_re_enters_inference() {
    let first_response = serde_json::json!({
        "id": "resp_ws_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust 2025 edition"}
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_ws_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Rust 2025 brings great features."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    spawn_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "web search round-trip should return 200");
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_ws_2",
        "final response should be the second model response after web search"
    );

    // A successful search updates the model's placeholder in place, so the public
    // response carries exactly one completed web_search_call for ws_1.
    let output = response["output"].as_array().expect("final response output array");
    let search_calls: Vec<&serde_json::Value> =
        output.iter().filter(|item| item["type"] == "web_search_call").collect();
    assert_eq!(
        search_calls.len(),
        1,
        "final response must contain exactly one web_search_call, got: {output:#?}"
    );
    assert_eq!(search_calls[0]["id"], "ws_1");
    assert_eq!(search_calls[0]["status"], "completed");

    let model_reqs = model.requests();
    assert_eq!(
        model_reqs.len(),
        2,
        "model backend should receive exactly two requests (initial + post-search)"
    );

    let second_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("second model request should be valid JSON");
    let input = second_body["input"]
        .as_array()
        .expect("second model request input should be an array");

    // #808: a hosted web_search_call is not a valid OpenResponses input item
    // (vLLM's Harmony conversion rejects it with HTTP 400), so the continuation
    // must never forward it to the inference backend.
    assert!(
        input.iter().all(|item| item["type"] != "web_search_call"),
        "second inference input must not contain hosted web_search_call items: {input:?}"
    );

    // The search result reaches the model through a backend-valid
    // function_call / function_call_output bridge instead.
    let has_web_search_call = input
        .iter()
        .any(|item| item["type"] == "function_call" && item["name"] == "web_search");
    assert!(
        has_web_search_call,
        "second inference input should carry a synthetic web_search function_call: {input:?}"
    );
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("second inference input should contain a function_call_output");
    assert!(
        function_output["output"]
            .as_str()
            .is_some_and(|output| output.contains("blog.rust-lang.org")),
        "second inference should receive the web search results: {function_output:?}"
    );
}

#[test]
fn web_search_provider_failure_continues_loop_with_failed_result() {
    let first_response = serde_json::json!({
        "id": "resp_ws_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust 2025 edition"}
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_ws_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I could not search, but here is what I know."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    spawn_failing_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "a provider failure must not reject the Response"
    );
    let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_ws_2",
        "the loop must continue to a second inference after the search fails"
    );

    // The public response must carry exactly one web_search_call for ws_1, marked
    // failed — not a contradictory completed placeholder plus a failed duplicate.
    let output = response["output"].as_array().expect("final response output array");
    let search_calls: Vec<&serde_json::Value> =
        output.iter().filter(|item| item["type"] == "web_search_call").collect();
    assert_eq!(
        search_calls.len(),
        1,
        "final response must contain exactly one web_search_call, got: {output:#?}"
    );
    assert_eq!(search_calls[0]["id"], "ws_1");
    assert_eq!(
        search_calls[0]["status"], "failed",
        "the single web_search_call must reflect the failed outcome"
    );

    let model_reqs = model.requests();
    assert_eq!(
        model_reqs.len(),
        2,
        "model backend should receive two requests (initial + post-failure)"
    );

    let second_body: serde_json::Value =
        serde_json::from_str(&model_reqs[1].body).expect("second model request should be valid JSON");
    let input = second_body["input"]
        .as_array()
        .expect("second model request input should be an array");
    // The model receives the failure through a backend-valid function_call_output
    // bridge carrying the bounded notice — never a hosted web_search_call, which
    // is not a valid OpenResponses input (issue #808).
    assert!(
        input.iter().all(|item| item["type"] != "web_search_call"),
        "the continuation must not feed the model a hosted web_search_call: {input:#?}"
    );
    let has_failure_notice = input
        .iter()
        .any(|item| item["type"] == "function_call_output" && item["output"] == "Web search unavailable.");
    assert!(
        has_failure_notice,
        "the model must receive a truthful failure notice via function_call_output: {input:#?}"
    );
}

#[test]
fn streaming_web_search_round_trip_resumes_one_logical_response() {
    let search_call = serde_json::json!({
        "type": "web_search_call",
        "id": "ws_stream_1",
        "status": "completed",
        "action": {"type": "search", "query": "Rust 2025 edition"}
    });
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_stream_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_ws_stream_1",
                "output_index": 0,
                "item": search_call,
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_stream_1",
                    "object": "response",
                    "status": "completed",
                    "output": [search_call],
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let final_message = serde_json::json!({
        "type": "message",
        "id": "msg_ws_stream_2",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Rust search completed."}]
    });
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_stream_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_ws_stream_2",
                "item_id": "msg_ws_stream_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "Rust search completed.",
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_stream_2",
                    "object": "response",
                    "status": "completed",
                    "output": [final_message],
                    "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let (model_port, model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    let search_calls = spawn_search_mock(search_listener);
    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model_port, search_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search_preview"}]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streamed web search should return 200: {raw}");

    // #985: verify the whole synthesized SSE event history is spec-conformant
    // end to end for a model-declared, locally executed web_search_call.
    let frames = assert_logical_stream_conformance(&body, "resp_ws_stream_1");

    // Exact per-event sequence numbers across the logical stream (contiguous
    // 0..7), which also fixes the emission order.
    assert_eq!(
        frame_seq(sole_event(&frames, "response.created")),
        0,
        "created seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "web_search_call")),
        1,
        "web_search_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.web_search_call.in_progress")),
        2,
        "web_search_call in_progress seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.web_search_call.searching")),
        3,
        "web_search_call searching seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.web_search_call.completed")),
        4,
        "web_search_call completed seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.done", "web_search_call")),
        5,
        "web_search_call done seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.output_text.delta")),
        6,
        "resumed output_text delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.completed")),
        7,
        "terminal seq: {body}"
    );

    // #276: the model announces the web_search_call via output_item.added but
    // never streams its progress events; the proxy synthesizes the missing
    // in_progress -> searching -> completed lifecycle without a second added, all
    // at the reserved output index 0 and item id ws_stream_1.
    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        1,
        "the model-streamed output_item.added must not be duplicated: {body}"
    );
    let ws_added = item_frame(&frames, "response.output_item.added", "web_search_call");
    assert_eq!(
        ws_added.data["output_index"], 0,
        "web_search_call added output_index: {body}"
    );
    assert_eq!(
        ws_added.data["item"]["id"], "ws_stream_1",
        "web_search_call added id: {body}"
    );
    for event in [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ] {
        let frame = sole_event(&frames, event);
        assert_eq!(frame.data["output_index"], 0, "{event} output_index: {body}");
        assert_eq!(frame.data["item_id"], "ws_stream_1", "{event} item_id: {body}");
    }
    let ws_done = item_frame(&frames, "response.output_item.done", "web_search_call");
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "one output_item.done: {body}"
    );
    assert_eq!(
        ws_done.data["output_index"], 0,
        "web_search_call done output_index: {body}"
    );
    assert_eq!(
        ws_done.data["item"]["id"], "ws_stream_1",
        "web_search_call done id: {body}"
    );

    // The resumed model output follows the single tool item at output index 1.
    let text_delta = sole_event(&frames, "response.output_text.delta");
    assert_eq!(text_delta.data["output_index"], 1, "resumed text output_index: {body}");
    assert_eq!(
        text_delta.data["content_index"], 0,
        "resumed text content_index: {body}"
    );
    assert_eq!(
        text_delta.data["item_id"], "msg_ws_stream_2",
        "resumed text item id: {body}"
    );

    // The terminal snapshot agrees with the incremental history item-for-item.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 2, "terminal output must snapshot both items: {body}");
    assert_eq!(output[0]["type"], "web_search_call", "terminal[0] type: {body}");
    assert_eq!(output[0]["id"], "ws_stream_1", "terminal[0] id: {body}");
    assert_eq!(output[0]["status"], "completed", "terminal[0] status: {body}");
    assert_eq!(
        output[0]["action"]["query"], "Rust 2025 edition",
        "terminal[0] query: {body}"
    );
    assert_eq!(output[1]["type"], "message", "terminal[1] type: {body}");
    assert_eq!(output[1]["id"], "msg_ws_stream_2", "terminal[1] id: {body}");
    assert_eq!(
        output[1]["content"][0]["text"], "Rust search completed.",
        "terminal[1] assistant text: {body}"
    );
    assert_eq!(
        sole_event(&frames, "response.completed").data["response"]["id"],
        "resp_ws_stream_1",
        "terminal logical id: {body}"
    );

    model_thread.join().expect("streaming model thread should finish");
    assert_eq!(
        search_calls.load(Ordering::SeqCst),
        1,
        "the web search must execute exactly once"
    );
    let requests = model_requests
        .lock()
        .expect("model request lock should not be poisoned");
    assert_eq!(requests.len(), 2, "web search should trigger a second model stream");
    let second_request: serde_json::Value =
        serde_json::from_str(&requests[1]).expect("second model request should be JSON");
    drop(requests);
    let input = second_request["input"]
        .as_array()
        .expect("second model request input should be an array");

    // #808: a hosted web_search_call is not a valid OpenResponses input item, so
    // the streamed continuation must never forward it to the inference backend.
    assert!(
        input.iter().all(|item| item["type"] != "web_search_call"),
        "second inference input must not contain hosted web_search_call items: {input:?}"
    );

    // The completed search reaches the model through a backend-valid
    // function_call / function_call_output bridge instead.
    let has_web_search_call = input
        .iter()
        .any(|item| item["type"] == "function_call" && item["name"] == "web_search");
    assert!(
        has_web_search_call,
        "second inference input should carry a synthetic web_search function_call: {input:?}"
    );
    let function_output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("second inference input should contain a function_call_output");
    assert!(
        function_output["output"]
            .as_str()
            .is_some_and(|output| output.contains("blog.rust-lang.org")),
        "second inference should receive the web search results: {function_output:?}"
    );

    // #985 (criterion 3): the synthetic bridge is internally consistent — the
    // web_search function_call and its function_call_output share one call_id, so
    // the resumed round refers unambiguously to the same locally executed search.
    let bridge_call = input
        .iter()
        .find(|item| item["type"] == "function_call" && item["name"] == "web_search")
        .expect("resumed request must include the web_search function_call");
    let bridge_call_id = bridge_call["call_id"]
        .as_str()
        .expect("bridge function_call must carry a call_id");
    assert_eq!(
        function_output["call_id"].as_str(),
        Some(bridge_call_id),
        "bridge function_call_output must reuse the bridge function_call call_id: {input:?}"
    );
}

#[test]
fn streaming_web_search_suppresses_premature_round_zero_done() {
    // #276 (finding): the model announces a web_search_call AND emits its
    // output_item.done in round 0 without ever streaming the tool's progress
    // lifecycle. The proxy executes the search locally and synthesizes
    // in_progress -> searching -> completed -> done. The premature round-0 done
    // must be suppressed so the client sees exactly one, correctly ordered done
    // for the item (added -> in_progress -> searching -> completed -> done), never
    // added -> done -> ...progress... -> done.
    let search_call = serde_json::json!({
        "type": "web_search_call",
        "id": "ws_premature_1",
        "status": "completed",
        "action": {"type": "search", "query": "Rust 2025 edition"}
    });
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_premature_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_ws_premature_1",
                "output_index": 0,
                "item": search_call,
                "sequence_number": 1
            }),
        ),
        // The model prematurely finalizes the tool item before any progress event.
        sse_event(
            "response.output_item.done",
            serde_json::json!({
                "response_id": "resp_ws_premature_1",
                "output_index": 0,
                "item": search_call,
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_premature_1",
                    "object": "response",
                    "status": "completed",
                    "output": [search_call],
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
                },
                "sequence_number": 3
            }),
        ),
    ];
    let final_message = serde_json::json!({
        "type": "message",
        "id": "msg_ws_premature_2",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Rust search completed."}]
    });
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_premature_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_ws_premature_2",
                "item_id": "msg_ws_premature_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "Rust search completed.",
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_premature_2",
                    "object": "response",
                    "status": "completed",
                    "output": [final_message],
                    "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let (model_port, _model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    let search_calls = spawn_search_mock(search_listener);
    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model_port, search_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search_preview"}]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streamed web search should return 200: {raw}");

    // The full synthesized history stays conformant despite the premature done.
    let frames = assert_logical_stream_conformance(&body, "resp_ws_premature_1");

    // Exactly one output_item.done for the whole logical stream, and it belongs to
    // the web_search_call. Because the premature round-0 done is suppressed, the
    // item is neither finalized twice nor announced twice.
    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        1,
        "the web_search_call must be announced exactly once: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "the premature round-0 output_item.done must be suppressed, leaving one ordered done: {body}"
    );
    let ws_done = item_frame(&frames, "response.output_item.done", "web_search_call");
    assert_eq!(
        ws_done.data["item"]["id"], "ws_premature_1",
        "the single done belongs to the web_search_call: {body}"
    );

    // The single done follows the synthesized progress lifecycle in order.
    let in_progress = frame_seq(sole_event(&frames, "response.web_search_call.in_progress"));
    let searching = frame_seq(sole_event(&frames, "response.web_search_call.searching"));
    let completed = frame_seq(sole_event(&frames, "response.web_search_call.completed"));
    let done = frame_seq(ws_done);
    assert!(
        in_progress < searching && searching < completed && completed < done,
        "progress lifecycle must precede the single done (in_progress={in_progress}, searching={searching}, completed={completed}, done={done}): {body}"
    );

    // The terminal snapshot agrees with the incremental history item-for-item.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 2, "terminal output must snapshot both items: {body}");
    assert_eq!(output[0]["type"], "web_search_call", "terminal[0] type: {body}");
    assert_eq!(output[0]["id"], "ws_premature_1", "terminal[0] id: {body}");
    assert_eq!(output[0]["status"], "completed", "terminal[0] status: {body}");
    assert_eq!(output[1]["type"], "message", "terminal[1] type: {body}");

    model_thread.join().expect("streaming model thread should finish");
    assert_eq!(
        search_calls.load(Ordering::SeqCst),
        1,
        "the web search must execute exactly once despite the premature done"
    );
}

#[test]
fn streaming_web_search_partial_in_band_lifecycle_synthesizes_missing_phases() {
    // #276 (partial lifecycle): the model announces the web_search_call AND streams
    // only its `in_progress` progress event in-band, never `searching` or
    // `completed`. Each phase is a distinct API lifecycle event, so the proxy must
    // fill in exactly the missing `searching` and `completed` phases when it
    // resumes after the local search — without repeating the `in_progress` already
    // delivered in-band or the `output_item.added` the model streamed. A single
    // "lifecycle streamed" flag would treat the whole lifecycle as delivered and
    // drop the still-owed `searching` phase.
    let search_call = serde_json::json!({
        "type": "web_search_call",
        "id": "ws_partial_1",
        "status": "completed",
        "action": {"type": "search", "query": "Rust 2025 edition"}
    });
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_partial_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_ws_partial_1",
                "output_index": 0,
                "item": search_call,
                "sequence_number": 1
            }),
        ),
        // Only the leading progress event is streamed in-band; `searching` and
        // `completed` never arrive from the model.
        sse_event(
            "response.web_search_call.in_progress",
            serde_json::json!({
                "item_id": "ws_partial_1",
                "output_index": 0,
                "sequence_number": 2
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_partial_1",
                    "object": "response",
                    "status": "completed",
                    "output": [search_call],
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
                },
                "sequence_number": 3
            }),
        ),
    ];
    let final_message = serde_json::json!({
        "type": "message",
        "id": "msg_ws_partial_2",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Rust search completed."}]
    });
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_partial_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_ws_partial_2",
                "item_id": "msg_ws_partial_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "Rust search completed.",
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_partial_2",
                    "object": "response",
                    "status": "completed",
                    "output": [final_message],
                    "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let (model_port, _model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    let search_calls = spawn_search_mock(search_listener);
    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model_port, search_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search_preview"}]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(parse_status(&raw), 200, "streamed web search should return 200: {raw}");

    // The full synthesized history stays conformant despite the partial lifecycle.
    let frames = assert_logical_stream_conformance(&body, "resp_ws_partial_1");

    // The item is announced once, and its `in_progress` phase reaches the client
    // exactly once — the in-band one from round 0, never re-synthesized on resume.
    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        1,
        "the web_search_call must be announced exactly once: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.web_search_call.in_progress"),
        1,
        "the in-band in_progress phase must not be re-synthesized on resume: {body}"
    );
    // The phases the model never streamed are filled in — the middle `searching`
    // phase is not dropped just because `in_progress` already streamed.
    assert_eq!(
        event_count(&frames, "response.web_search_call.searching"),
        1,
        "the missing `searching` phase must be synthesized exactly once: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.web_search_call.completed"),
        1,
        "the missing `completed` phase must be synthesized exactly once: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "the web_search_call must be finalized exactly once: {body}"
    );

    // The in-band `in_progress` (round 0) precedes the synthesized `searching`,
    // proving the resumed round supplied only the missing tail of the lifecycle.
    let in_progress = frame_seq(sole_event(&frames, "response.web_search_call.in_progress"));
    let searching = frame_seq(sole_event(&frames, "response.web_search_call.searching"));
    let completed = frame_seq(sole_event(&frames, "response.web_search_call.completed"));
    let done = frame_seq(item_frame(&frames, "response.output_item.done", "web_search_call"));
    assert_eq!(
        in_progress, 2,
        "the in-band in_progress keeps its round-0 sequence: {body}"
    );
    assert!(
        in_progress < searching && searching < completed && completed < done,
        "the missing phases must be ordered after the in-band in_progress (in_progress={in_progress}, searching={searching}, completed={completed}, done={done}): {body}"
    );

    // The terminal snapshot agrees with the incremental history item-for-item.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 2, "terminal output must snapshot both items: {body}");
    assert_eq!(output[0]["type"], "web_search_call", "terminal[0] type: {body}");
    assert_eq!(output[0]["id"], "ws_partial_1", "terminal[0] id: {body}");
    assert_eq!(output[0]["status"], "completed", "terminal[0] status: {body}");
    assert_eq!(output[1]["type"], "message", "terminal[1] type: {body}");

    model_thread.join().expect("streaming model thread should finish");
    assert_eq!(
        search_calls.load(Ordering::SeqCst),
        1,
        "the web search must execute exactly once"
    );
}

#[test]
fn streaming_web_search_failure_synthesizes_partial_progress_in_one_logical_response() {
    // The model announces the web_search_call as completed, but the local search
    // provider fails, so the proxy must mark the call failed and synthesize a
    // partial progress lifecycle (in_progress -> searching, then no completed).
    let search_call = serde_json::json!({
        "type": "web_search_call",
        "id": "ws_fail_1",
        "status": "completed",
        "action": {"type": "search", "query": "Rust 2025 edition"}
    });
    let first_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_fail_1", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_item.added",
            serde_json::json!({
                "response_id": "resp_ws_fail_1",
                "output_index": 0,
                "item": search_call,
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_fail_1",
                    "object": "response",
                    "status": "completed",
                    "output": [search_call],
                    "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let final_message = serde_json::json!({
        "type": "message",
        "id": "msg_ws_fail_2",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "I could not complete the search."}]
    });
    let second_response = vec![
        sse_event(
            "response.created",
            serde_json::json!({
                "response": {"id": "resp_ws_fail_2", "object": "response", "status": "in_progress", "output": []},
                "sequence_number": 0
            }),
        ),
        sse_event(
            "response.output_text.delta",
            serde_json::json!({
                "response_id": "resp_ws_fail_2",
                "item_id": "msg_ws_fail_2",
                "output_index": 0,
                "content_index": 0,
                "delta": "I could not complete the search.",
                "sequence_number": 1
            }),
        ),
        sse_event(
            "response.completed",
            serde_json::json!({
                "response": {
                    "id": "resp_ws_fail_2",
                    "object": "response",
                    "status": "completed",
                    "output": [final_message],
                    "usage": {"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}
                },
                "sequence_number": 2
            }),
        ),
    ];
    let (model_port, _model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);
    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    spawn_failing_search_mock(search_listener);
    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model_port, search_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust 2025 edition features",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search_preview"}]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(
        parse_status(&raw),
        200,
        "streamed failing web search should return 200: {raw}"
    );

    // #985: even a failed local search keeps the logical stream conformant.
    let frames = assert_logical_stream_conformance(&body, "resp_ws_fail_1");

    // Exact per-event sequence numbers: web_search_call has no conformant failed
    // event, so a failed search emits in_progress and searching but never
    // completed; its failure surfaces through the item status alone.
    assert_eq!(
        frame_seq(sole_event(&frames, "response.created")),
        0,
        "created seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.added", "web_search_call")),
        1,
        "web_search_call added seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.web_search_call.in_progress")),
        2,
        "web_search_call in_progress seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.web_search_call.searching")),
        3,
        "web_search_call searching seq: {body}"
    );
    assert_eq!(
        frame_seq(item_frame(&frames, "response.output_item.done", "web_search_call")),
        4,
        "web_search_call done seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.output_text.delta")),
        5,
        "resumed output_text delta seq: {body}"
    );
    assert_eq!(
        frame_seq(sole_event(&frames, "response.completed")),
        6,
        "terminal seq: {body}"
    );

    // #276: no completed progress event for a failed search, and the single tool
    // item keeps output index 0 and item id ws_fail_1 throughout.
    assert_eq!(
        event_count(&frames, "response.web_search_call.completed"),
        0,
        "a failed web search must not emit completed: {body}"
    );
    for event in [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
    ] {
        let frame = sole_event(&frames, event);
        assert_eq!(frame.data["output_index"], 0, "{event} output_index: {body}");
        assert_eq!(frame.data["item_id"], "ws_fail_1", "{event} item_id: {body}");
    }
    assert_eq!(
        event_count(&frames, "response.output_item.added"),
        1,
        "one item announced: {body}"
    );
    assert_eq!(
        event_count(&frames, "response.output_item.done"),
        1,
        "one output_item.done: {body}"
    );

    // The terminal snapshot carries the failed web_search_call and the resumed
    // message, agreeing item-for-item with the incremental history.
    let output = terminal_output(&frames);
    assert_eq!(output.len(), 2, "terminal output must snapshot both items: {body}");
    assert_eq!(output[0]["type"], "web_search_call", "terminal[0] type: {body}");
    assert_eq!(output[0]["id"], "ws_fail_1", "terminal[0] id: {body}");
    assert_eq!(
        output[0]["status"], "failed",
        "terminal[0] status must be failed: {body}"
    );
    assert_eq!(output[1]["type"], "message", "terminal[1] type: {body}");
    assert_eq!(
        output[1]["content"][0]["text"], "I could not complete the search.",
        "terminal[1] assistant text: {body}"
    );
    assert_eq!(
        sole_event(&frames, "response.completed").data["response"]["id"],
        "resp_ws_fail_1",
        "terminal logical id: {body}"
    );

    model_thread.join().expect("streaming model thread should finish");
}

// -----------------------------------------------------------------------------
// Fail closed: terminal streaming without a logical-stream finalizer
// -----------------------------------------------------------------------------

#[test]
fn terminal_streaming_without_logical_stream_fails_closed_before_dispatch() {
    // openai_responses_proxy keeps terminal_streaming: true, but openai_stream_events
    // is reconfigured with logical_stream: false. Typed streaming commits
    // response.completed to the client as it arrives, so a loop-terminal error
    // detected later by openai_agentic_loop could not reach the client. The loop
    // must therefore reject before any backend request rather than forward a
    // truncatable success.
    let (model_port, model_requests, _model_thread) = start_streaming_model(vec![vec![sse_event(
        "response.completed",
        serde_json::json!({
            "response": {"id": "resp_unreached", "object": "response", "status": "completed", "output": []},
            "sequence_number": 0
        }),
    )]]);
    let proxy_port = free_port();
    let config = load_agentic_config_without_logical_stream(proxy_port, model_port);
    let proxy = start_proxy(&config);
    let request = serde_json::json!({
        "model": "gpt-4.1",
        "input": "What is the weather in SF?",
        "stream": true,
        "store": false
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        500,
        "unsafe terminal streaming without logical_stream must fail closed with 500: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("server_error"),
        "the rejection must carry the server_error code: {body}"
    );
    assert!(
        model_requests
            .lock()
            .expect("model request lock should not be poisoned")
            .is_empty(),
        "the loop must reject before dispatching any backend request"
    );
}

/// Serve web-search results and count every accepted connection.
///
/// The returned counter exposes exactly-once execution to the caller: a single
/// search callout yields one connection, so a duplicate local execution would
/// push the count past one. The listener loops so a second (erroneous) callout
/// is accepted and counted rather than failing obliquely with a refused
/// connection.
fn spawn_search_mock(listener: TcpListener) -> Arc<AtomicUsize> {
    use std::io::{Read as _, Write as _};
    let body = serde_json::json!({
        "web": {
            "results": [{
                "title": "Rust 2025 Edition",
                "url": "https://blog.rust-lang.org/2025",
                "description": "The Rust 2025 edition is here."
            }]
        }
    })
    .to_string();
    let connections = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&connections);
    thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            counter.fetch_add(1, Ordering::SeqCst);
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap_or(0);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            // Ignore write failures so one closed client connection does not tear
            // down the accept loop; a named `_` binding avoids both the
            // `let_underscore_drop` and `unused_result_ok` lints.
            let _written = stream.write_all(response.as_bytes());
        }
    });
    connections
}

/// Serve a single 5xx so the search client maps the callout to a failed outcome.
fn spawn_failing_search_mock(listener: TcpListener) {
    use std::io::{Read as _, Write as _};
    let body = r#"{"error":"service unavailable"}"#;
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 4096];
        let _n = stream.read(&mut buf).unwrap();
        let response = format!(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
}

/// Search mock that serves every connection and counts dispatched requests.
///
/// Returns a shared counter so a test can assert exactly how many provider
/// requests the web search filter issued across an agentic-loop continuation.
fn spawn_counting_search_mock(listener: TcpListener) -> Arc<AtomicUsize> {
    use std::io::{Read as _, Write as _};
    let counter = Arc::new(AtomicUsize::new(0));
    let thread_counter = Arc::clone(&counter);
    let body = serde_json::json!({
        "web": {
            "results": [{
                "title": "Rust 2025 Edition",
                "url": "https://blog.rust-lang.org/2025",
                "description": "The Rust 2025 edition is here."
            }]
        }
    })
    .to_string();
    thread::spawn(move || {
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

#[test]
fn web_search_caps_multiple_calls_within_one_round() {
    // The model requests two searches in a single turn while the client caps
    // built-in tool calls at one. Exactly one provider request must be
    // dispatched and the excess call must be surfaced as incomplete.
    let first_response = serde_json::json!({
        "id": "resp_ws_cap_1",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_cap_a",
                "status": "completed",
                "action": {"type": "search", "query": "Rust 2025 edition"}
            },
            {
                "type": "web_search_call",
                "id": "ws_cap_b",
                "status": "completed",
                "action": {"type": "search", "query": "Rust async runtime"}
            }
        ]
    });
    let second_response = serde_json::json!({
        "id": "resp_ws_cap_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here is what I found."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    let search_count = spawn_counting_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Search for Rust news",
        "max_tool_calls": 1,
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "capped web search round-trip should return 200"
    );

    assert_eq!(
        search_count.load(Ordering::SeqCst),
        1,
        "only one provider request may be dispatched under max_tool_calls=1"
    );

    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_ws_cap_2",
        "loop should still complete and return the final model response"
    );

    let output = response["output"]
        .as_array()
        .expect("response output should be an array");
    let incomplete = output.iter().any(|item| {
        item["type"] == "web_search_call"
            && item["status"] == "incomplete"
            && item["action"]["query"] == "Rust async runtime"
    });
    assert!(
        incomplete,
        "the over-budget web_search_call must be surfaced as incomplete, not executed"
    );

    assert_eq!(
        model.requests().len(),
        2,
        "model backend should receive the initial request plus one post-search continuation"
    );
}

#[test]
fn web_search_budget_persists_across_loop_iterations() {
    // The client caps built-in tool calls at one, but the model requests a
    // *new* web search in a *later* loop iteration. The executed count lives
    // in ResponsesState, which survives IRR re-entries, so the second-round
    // search must be declined even though each individual round contains only
    // a single call. A per-iteration counter would reset and wrongly dispatch
    // twice.
    let first_response = serde_json::json!({
        "id": "resp_persist_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_persist_a",
            "status": "completed",
            "action": {"type": "search", "query": "Rust 2025 edition"}
        }]
    });
    let second_response = serde_json::json!({
        "id": "resp_persist_2",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "web_search_call",
            "id": "ws_persist_b",
            "status": "completed",
            "action": {"type": "search", "query": "Rust async runtime"}
        }]
    });
    let third_response = serde_json::json!({
        "id": "resp_persist_3",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Here is what I found."}]
        }]
    });

    let model = StatefulCapturingBackend::new(vec![
        (200, serde_json::to_string(&first_response).unwrap()),
        (200, serde_json::to_string(&second_response).unwrap()),
        (200, serde_json::to_string(&third_response).unwrap()),
    ])
    .start_with_shutdown();

    let search_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let search_port = search_listener.local_addr().unwrap().port();
    let search_count = spawn_counting_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_web_search_config(proxy_port, model.port(), search_port);
    let proxy = start_proxy(&config);

    let request_body = serde_json::json!({
        "model": "gpt-4.1",
        "input": "Research Rust news",
        "max_tool_calls": 1,
        "tools": [{"type": "web_search_preview"}]
    });
    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request_body).unwrap()),
    );

    assert_eq!(parse_status(&raw), 200, "multi-round web search should return 200");

    assert_eq!(
        search_count.load(Ordering::SeqCst),
        1,
        "the executed budget must persist across iterations: only the first-round search dispatches"
    );

    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("response should be JSON");
    assert_eq!(
        response["id"], "resp_persist_3",
        "loop should complete and return the final model response"
    );

    let output = response["output"]
        .as_array()
        .expect("response output should be an array");
    let second_round_incomplete = output.iter().any(|item| {
        item["type"] == "web_search_call"
            && item["status"] == "incomplete"
            && item["action"]["query"] == "Rust async runtime"
    });
    assert!(
        second_round_incomplete,
        "the second-iteration search must be declined as incomplete once the budget is spent"
    );

    assert_eq!(
        model.requests().len(),
        3,
        "model backend should receive three requests across the two-search loop"
    );
}

// -----------------------------------------------------------------------------
// #985: logical-stream SSE conformance helpers
//
// These parse a raw logical response body into ordered event frames and assert
// the tool-agnostic conformance invariants from issue #985: correct event
// types/shapes, a single created + single terminal in lifecycle order,
// consistent ids/indices across rounds, a unique/monotonic/contiguous sequence,
// no duplicate items, and a terminal snapshot that agrees with the incremental
// event history. Each round-trip test layers its tool-specific assertions on top
// of the returned frames.
// -----------------------------------------------------------------------------

/// One parsed SSE `event:`/`data:` frame from a logical response stream.
struct SseFrame {
    /// The `event:` type (e.g. `response.output_item.added`).
    event_type: String,
    /// The parsed `data:` JSON payload.
    data: serde_json::Value,
}

/// Terminal event types that close a logical response stream.
const TERMINAL_EVENTS: [&str; 3] = ["response.completed", "response.failed", "response.incomplete"];

/// Split a raw SSE body into ordered event frames.
///
/// Production encodes each event as `event: <type>\ndata: <single-line-json>\n\n`
/// plus an optional trailing `data: [DONE]\n\n` sentinel. This returns one
/// [`SseFrame`] per typed `event:` block in emission order; the `[DONE]`
/// sentinel is skipped (its position is validated separately). Panics on a block
/// missing an `event:` line or whose `data:` is not JSON, matching the test-only
/// `expect`/`unwrap` style in this suite.
fn parse_sse_frames(body: &str) -> Vec<SseFrame> {
    let mut frames = Vec::new();
    for block in body.split("\n\n") {
        let block = block.trim_matches('\n');
        if block.is_empty() {
            continue;
        }
        let mut event_type: Option<String> = None;
        let mut data_lines: Vec<&str> = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event_type = Some(rest.to_owned());
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data_lines.push(rest);
            } else if line.starts_with(':') {
                // SSE comment line; ignore per the SSE spec.
            }
        }
        let data = data_lines.join("\n");
        if data == "[DONE]" {
            continue;
        }
        let event_type = event_type.unwrap_or_else(|| panic!("SSE block without an event: line: {block:?}"));
        let data: serde_json::Value =
            serde_json::from_str(&data).unwrap_or_else(|err| panic!("SSE data is not JSON ({err}): {data:?}"));
        frames.push(SseFrame { event_type, data });
    }
    frames
}

/// The single frame of `event_type`, panicking unless exactly one exists.
fn sole_event<'a>(frames: &'a [SseFrame], event_type: &str) -> &'a SseFrame {
    let mut matching = frames.iter().filter(|frame| frame.event_type == event_type);
    let frame = matching.next().unwrap_or_else(|| panic!("no {event_type} frame found"));
    assert!(matching.next().is_none(), "expected exactly one {event_type} frame");
    frame
}

/// The single `event_type` frame whose `item.type` equals `item_type`.
///
/// `output_item.added`/`output_item.done` can appear for several items in one
/// stream, so callers disambiguate by the wrapped item's type.
fn item_frame<'a>(frames: &'a [SseFrame], event_type: &str, item_type: &str) -> &'a SseFrame {
    let mut matching = frames
        .iter()
        .filter(|frame| frame.event_type == event_type && frame.data["item"]["type"] == item_type);
    let frame = matching
        .next()
        .unwrap_or_else(|| panic!("no {event_type} frame for item type {item_type}"));
    assert!(
        matching.next().is_none(),
        "expected exactly one {event_type} frame for item type {item_type}"
    );
    frame
}

/// The `sequence_number` carried by a frame.
fn frame_seq(frame: &SseFrame) -> u64 {
    frame.data["sequence_number"]
        .as_u64()
        .unwrap_or_else(|| panic!("frame {} must carry a numeric sequence_number", frame.event_type))
}

/// Count of frames of `event_type`.
fn event_count(frames: &[SseFrame], event_type: &str) -> usize {
    frames.iter().filter(|frame| frame.event_type == event_type).count()
}

/// The terminal event's `response.output` snapshot array.
fn terminal_output(frames: &[SseFrame]) -> &Vec<serde_json::Value> {
    let terminal = frames
        .iter()
        .find(|frame| TERMINAL_EVENTS.contains(&frame.event_type.as_str()))
        .expect("logical stream must contain a terminal event");
    terminal.data["response"]["output"]
        .as_array()
        .expect("terminal response.output must be an array")
}

/// Assert the tool-agnostic issue #985 conformance invariants over a logical
/// response stream and return the parsed frames for tool-specific follow-up.
fn assert_logical_stream_conformance(body: &str, expected_response_id: &str) -> Vec<SseFrame> {
    let frames = parse_sse_frames(body);
    assert!(
        !frames.is_empty(),
        "logical stream must emit at least one event: {body}"
    );

    // (2) Lifecycle order: exactly one response.created, first; exactly one
    // terminal event, last; any [DONE] sentinel strictly after the terminal.
    assert_eq!(
        event_count(&frames, "response.created"),
        1,
        "exactly one response.created must be emitted: {body}"
    );
    assert_eq!(
        frames[0].event_type, "response.created",
        "response.created must be the first emitted event: {body}"
    );
    let terminal_count = frames
        .iter()
        .filter(|frame| TERMINAL_EVENTS.contains(&frame.event_type.as_str()))
        .count();
    assert_eq!(terminal_count, 1, "exactly one terminal event must be emitted: {body}");
    let last = &frames[frames.len() - 1];
    assert!(
        TERMINAL_EVENTS.contains(&last.event_type.as_str()),
        "the terminal event must be the last emitted event: {body}"
    );
    if let Some(done_pos) = body.find("data: [DONE]") {
        let terminal_pos = body
            .rfind(&format!("event: {}", last.event_type))
            .expect("terminal event must appear in the raw body");
        assert!(
            terminal_pos < done_pos,
            "the [DONE] sentinel must appear after the terminal event: {body}"
        );
    }

    // (4) Sequence numbers: every event carries one; the emission-order series is
    // exactly 0..=N (unique, strictly increasing, contiguous); the terminal
    // carries the maximum. A single seq 0 on response.created proves the
    // suppressed resumed-round response.created consumed no number.
    let sequences: Vec<u64> = frames.iter().map(frame_seq).collect();
    let expected: Vec<u64> = (0..sequences.len() as u64).collect();
    assert_eq!(
        sequences, expected,
        "sequence numbers must be unique, strictly increasing, and contiguous from 0: {body}"
    );
    assert_eq!(
        frame_seq(last),
        sequences.len() as u64 - 1,
        "the terminal event must carry the final (maximum) sequence number: {body}"
    );

    // (3) Response id: one stable logical id on every id-bearing event; the
    // resumed-round id never leaks.
    for frame in &frames {
        if let Some(response_id) = frame.data.get("response_id").and_then(serde_json::Value::as_str) {
            assert_eq!(
                response_id, expected_response_id,
                "every response_id must be the stable logical id ({}): {body}",
                frame.event_type
            );
        }
        if let Some(id) = frame.data["response"]["id"].as_str() {
            assert_eq!(
                id, expected_response_id,
                "every response.id must be the stable logical id ({}): {body}",
                frame.event_type
            );
        }
    }
    assert_eq!(
        frames[0].data["response"]["id"].as_str(),
        Some(expected_response_id),
        "response.created must carry the logical response id: {body}"
    );

    // (5) No duplicate output_item.added for any item id.
    let mut added_ids: HashMap<String, usize> = HashMap::new();
    for frame in frames
        .iter()
        .filter(|frame| frame.event_type == "response.output_item.added")
    {
        if let Some(id) = frame.data["item"]["id"].as_str() {
            *added_ids.entry(id.to_owned()).or_default() += 1;
        }
    }
    for (id, count) in &added_ids {
        assert_eq!(
            *count, 1,
            "item {id} must be announced by exactly one output_item.added: {body}"
        );
    }

    // (3 + 6) Non-overlapping output indices and terminal snapshot agreement:
    // each output_index maps to exactly one item id, the announced indices are
    // contiguous 0..len, and terminal output[i] matches the id (and, when the
    // incremental events announced it, the type) seen at output_index i.
    let mut index_ids: std::collections::BTreeMap<u64, String> = std::collections::BTreeMap::new();
    let mut index_types: std::collections::BTreeMap<u64, String> = std::collections::BTreeMap::new();
    for frame in &frames {
        let Some(output_index) = frame.data.get("output_index").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let id = frame.data["item"]["id"]
            .as_str()
            .or_else(|| frame.data["item_id"].as_str());
        if let Some(id) = id {
            if let Some(existing) = index_ids.get(&output_index) {
                assert_eq!(
                    existing, id,
                    "output_index {output_index} must map to a single item id: {body}"
                );
            } else {
                index_ids.insert(output_index, id.to_owned());
            }
        }
        if let Some(item_type) = frame.data["item"]["type"].as_str() {
            index_types.insert(output_index, item_type.to_owned());
        }
    }
    let output = terminal_output(&frames);
    assert_eq!(
        output.len(),
        index_ids.len(),
        "terminal output must contain exactly the incrementally announced items: {body}"
    );
    for (position, (index, id)) in index_ids.iter().enumerate() {
        assert_eq!(
            *index, position as u64,
            "announced output indices must be contiguous from 0: {body}"
        );
        assert_eq!(
            output[position]["id"].as_str(),
            Some(id.as_str()),
            "terminal output[{position}] id must match the item announced at output_index {index}: {body}"
        );
        if let Some(expected_type) = index_types.get(index) {
            assert_eq!(
                output[position]["type"].as_str(),
                Some(expected_type.as_str()),
                "terminal output[{position}] type must match the announced type: {body}"
            );
        }
    }

    // (7) Every locally executed tool item is finalized by exactly one
    // output_item.done. #276's synthesis owns the `output_item.done` envelope for
    // mcp_call / web_search_call / mcp_approval_request items: a backend round cut
    // off before the envelope (Finding 1) would leave such an item announced yet
    // never finalized (a dropped done), and a duplicated envelope would finalize it
    // twice. Model-streamed items (e.g. the trailing assistant message) may instead
    // be finalized by the terminal response snapshot, so they are excluded here.
    //
    // This holds because none of these scenarios mutate a local item's content
    // across rounds. When a previously delivered item legitimately changes (e.g. a
    // web_search_call that gains `action.sources` after further local execution),
    // the synthesis re-emits a fresh `output_item.done` carrying the updated item —
    // a second `done` for the same id by design, covered by the
    // `logical_stream_reemits_outcome_when_local_item_gains_sources` unit test. A
    // future harness scenario that drives such a mutation must relax this to assert
    // the item's final `done` is terminal rather than a bare count of one.
    const LOCAL_TOOL_TYPES: [&str; 3] = ["mcp_call", "web_search_call", "mcp_approval_request"];
    let local_added_ids: std::collections::BTreeSet<String> = frames
        .iter()
        .filter(|frame| frame.event_type == "response.output_item.added")
        .filter(|frame| {
            frame.data["item"]["type"]
                .as_str()
                .is_some_and(|item_type| LOCAL_TOOL_TYPES.contains(&item_type))
        })
        .filter_map(|frame| frame.data["item"]["id"].as_str().map(str::to_owned))
        .collect();
    for id in &local_added_ids {
        let done_count = frames
            .iter()
            .filter(|frame| frame.event_type == "response.output_item.done")
            .filter(|frame| frame.data["item"]["id"].as_str() == Some(id.as_str()))
            .count();
        assert_eq!(
            done_count, 1,
            "local tool item {id} must be finalized by exactly one output_item.done: {body}"
        );
    }

    frames
}

/// Encode a typed Responses event as one SSE frame.
fn sse_event(event_type: &str, mut payload: serde_json::Value) -> String {
    payload
        .as_object_mut()
        .expect("SSE payload should be an object")
        .insert("type".to_owned(), serde_json::Value::String(event_type.to_owned()));
    format!("event: {event_type}\ndata: {payload}\n\n")
}

/// Handle returned by the synthetic streaming model backend.
type StreamingModel = (u16, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>);

/// Start a two-turn model backend that emits each SSE event as a chunk.
fn start_streaming_model(responses: Vec<Vec<String>>) -> StreamingModel {
    let listener = TcpListener::bind("127.0.0.1:0").expect("streaming model should bind");
    let port = listener
        .local_addr()
        .expect("streaming model should have an address")
        .port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let handle = thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().expect("streaming model should accept request");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("streaming model should set read timeout");
            let request = read_json_request(&mut stream);
            captured
                .lock()
                .expect("model request lock should not be poisoned")
                .push(request);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .expect("streaming model should write response headers");
            for event in response {
                write!(stream, "{:x}\r\n{event}\r\n", event.len()).expect("streaming model should write event chunk");
                stream.flush().expect("streaming model should flush event chunk");
            }
            stream
                .write_all(b"0\r\n\r\n")
                .expect("streaming model should finish chunked response");
        }
    });
    (port, requests, handle)
}

/// Read one content-length JSON request and return its body.
fn read_json_request(stream: &mut TcpStream) -> String {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = stream.read(&mut buffer).expect("streaming model should read request");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&raw);
        let Some((headers, body)) = text.split_once("\r\n\r\n") else {
            continue;
        };
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if body.len() >= content_length {
            return body.get(..content_length).unwrap_or_default().to_owned();
        }
    }
    String::new()
}

fn load_web_search_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!(
            "api_key: test-key\n                base_url: http://127.0.0.1:{search_port}\n                allow_private_base_url: true"
        ),
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse web search config")
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Build an OpenAI Responses `usage` object from `(input, output, total)`.
fn usage_json((input, output, total): (u64, u64, u64)) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": total
    })
}

/// A minimal single-string-property MCP tool input schema.
fn object_schema(property: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {property: {"type": "string"}},
        "required": [property],
        "additionalProperties": false
    })
}

/// The `type` of a Responses output item, or `""` when absent.
fn output_item_type(item: &serde_json::Value) -> &str {
    item["type"].as_str().unwrap_or_default()
}

/// Parse a captured model request body and return its `input` array.
fn request_input(body: &str) -> Vec<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(body).expect("model request body should be valid JSON");
    value["input"].as_array().cloned().unwrap_or_default()
}

/// True when `item` is the `function_call_output` bridge carrying the weather
/// tool result fed back to the model.
fn is_weather_result(item: &serde_json::Value) -> bool {
    item["type"] == "function_call_output"
        && item["output"]
            .as_str()
            .is_some_and(|out| out.contains("mock result for get_weather"))
}

/// True when `item` is the `function_call_output` bridge carrying the time tool
/// result fed back to the model.
fn is_time_result(item: &serde_json::Value) -> bool {
    item["type"] == "function_call_output"
        && item["output"]
            .as_str()
            .is_some_and(|out| out.contains("mock result for get_time"))
}

/// Extract the `response` object from the single terminal `response.completed`
/// SSE frame in a client stream.
fn extract_completed_response(body: &str) -> serde_json::Value {
    let mut lines = body.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "event: response.completed" {
            continue;
        }
        for data_line in lines.by_ref() {
            if let Some(payload) = data_line.strip_prefix("data: ") {
                let event: serde_json::Value =
                    serde_json::from_str(payload.trim()).expect("response.completed data should be valid JSON");
                return event["response"].clone();
            }
            if data_line.trim().is_empty() {
                break;
            }
        }
    }
    panic!("no response.completed event found in stream: {body}");
}

fn patch_web_search_api_key(yaml: &str) -> String {
    yaml.replace("api_key: ${WEB_SEARCH_API_KEY}", "api_key: test-key")
}

fn load_agentic_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = patch_web_search_api_key(&yaml);
    praxis_core::config::Config::from_yaml(&yaml).expect("parse agentic-loop config")
}

fn load_agentic_config_without_logical_stream(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = patch_web_search_api_key(&yaml);
    // Disable logical_stream on the real openai_stream_events filter (the two-line
    // `- filter:`/`logical_stream:` pair). The doc comment above it also contains
    // the literal `logical_stream: true`, so match the filter line too to avoid
    // rewriting the comment instead of the config.
    let enabled = "- filter: openai_stream_events\n                logical_stream: true";
    let disabled = "- filter: openai_stream_events\n                logical_stream: false";
    let patched = yaml.replacen(enabled, disabled, 1);
    assert_ne!(
        patched, yaml,
        "expected to disable logical_stream in agentic-loop.yaml; its openai_stream_events block may have changed"
    );
    praxis_core::config::Config::from_yaml(&patched).expect("parse agentic-loop config without logical_stream")
}

fn load_agentic_rejection_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop-fixture.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop fixture");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    // Route action=loop back to inference so the loop re-enters and can reach the
    // iteration limit (508). Without this, IRR terminates after the first pass via
    // the default `done: true` branch.
    let terminal_on_result = "            on_result:\n              - default: true\n                done: true";
    let looping_on_result = "            on_result:\n              - filter: openai_agentic_loop\n                key: action\n                value: loop\n                next: inference\n              - default: true\n                done: true";
    let patched = yaml.replacen(terminal_on_result, looping_on_result, 1);
    assert_ne!(
        patched, yaml,
        "expected to inject the loop action into agentic-loop-fixture.yaml; its on_result block may have changed"
    );
    praxis_core::config::Config::from_yaml(&patched).expect("parse agentic-loop rejection config")
}

fn load_loopback_mcp_config(proxy_port: u16, model_port: u16) -> praxis_core::config::Config {
    let path = example_config_path("openai/responses/agentic-loop.yaml");
    let yaml = std::fs::read_to_string(path).expect("read agentic-loop example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    let yaml = patch_web_search_api_key(&yaml);
    let yaml = yaml.replacen(
        "      - filter: openai_mcp_tool_resolve\n",
        "      - filter: openai_mcp_tool_resolve\n        allow_loopback: true\n",
        1,
    );
    let yaml = yaml.replacen(
        "              - filter: openai_mcp_dispatch\n",
        "              - filter: openai_mcp_dispatch\n                allow_loopback: true\n",
        1,
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse loopback MCP config")
}
