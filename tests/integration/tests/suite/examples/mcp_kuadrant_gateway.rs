// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Config-validity functional test for the mcp-kuadrant-gateway example config.
//!
//! The live end-to-end acceptance (per-user discovery filtering + per-call
//! authorization through a real Kuadrant MCP Gateway) runs in the gated Kuadrant
//! lane; see `tests/integration/kuadrant/` and
//! `tests/integration/sdk/openai/test_openai_responses_mcp_kuadrant.py`. This test
//! only proves the example config loads and its pipeline builds, so the example
//! stays valid under normal CI without the live gateway.

use std::collections::HashMap;

use praxis_test_utils::{
    McpMockConfig, McpToolFixture, StatefulCapturingBackend, free_port, http_send, load_example_config, parse_status,
    start_mcp_mock_server_with_config, start_proxy,
};
use serde_json::json;

#[test]
fn example_config_loads_and_pipeline_builds() {
    let proxy_port = free_port();
    // The model endpoint and gateway connector are not dialed at startup, so the
    // pipeline builds without the live stack; mapping the model endpoint keeps the
    // rewrite explicit.
    let config = load_example_config(
        "openai/responses/mcp-kuadrant-gateway.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", free_port())]),
    );
    // start_proxy builds the full AI filter pipeline (incl. the MCP connector and
    // credential slots); an invalid connector/slot config would fail here.
    let _proxy = start_proxy(&config);

    // Discovery and execution share the same Kuadrant target and bearer slot.
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/configs/openai/responses/mcp-kuadrant-gateway.yaml"
    ))
    .expect("example config present");
    assert!(
        raw.contains("id: trusted-mcp"),
        "connector id trusted-mcp must be declared"
    );
    assert_eq!(raw.matches("filter: openai_mcp_connector").count(), 1);
    assert_eq!(
        raw.matches("user_credential: mcp_gateway").count(),
        1,
        "the connector must declare one per-user bearer slot"
    );
}

#[test]
fn example_forwards_credentials_on_discovery_and_dispatch() {
    let model = StatefulCapturingBackend::new(vec![
        (
            200,
            json!({
                "id": "resp_mcp_call",
                "created_at": 1000,
                "model": "gpt-4.1",
                "object": "response",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "id": "fc_mcp_call",
                    "call_id": "call_mcp_weather",
                    "name": "weather__get_weather",
                    "arguments": r#"{"location":"SF"}"#,
                    "status": "completed"
                }]
            })
            .to_string(),
        ),
        (
            200,
            json!({
                "id": "resp_mcp_final",
                "created_at": 2000,
                "model": "gpt-4.1",
                "object": "response",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The weather in SF is 72F."}]
                }]
            })
            .to_string(),
        ),
    ])
    .start_with_shutdown();
    let mcp = start_mcp_mock_server_with_config(McpMockConfig {
        tools: vec![McpToolFixture::new("get_weather")],
        ..McpMockConfig::default()
    });

    let config = load_example_config(
        "openai/responses/mcp-kuadrant-gateway.yaml",
        free_port(),
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8000", mcp.port())]),
    );
    let proxy = start_proxy(&config);
    let body = json!({"model":"gpt-4.1", "input":"weather?", "store":false,
        "tools":[{"type":"mcp", "server_label":"weather", "connector_id":"trusted-mcp", "require_approval":"never"}]})
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nx-auth-tenant: acme\r\nx-auth-user: userA\r\nx-user-mcp-key: test-key\r\nx-mcp-authorized: test-assertion\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let raw = http_send(proxy.addr(), &request);
    assert_eq!(parse_status(&raw), 200, "{raw}");
    let requests = mcp.received_requests();
    for method in ["tools/list", "tools/call"] {
        let request = requests
            .iter()
            .find(|request| request.json_rpc_method.as_deref() == Some(method))
            .expect("MCP method received");
        for (header, expected) in [
            ("authorization", "Bearer test-key"),
            ("x-mcp-authorized", "test-assertion"),
        ] {
            assert_eq!(
                request
                    .headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(header))
                    .map(|(_, value)| value.as_str()),
                Some(expected),
                "{method}: {header}"
            );
        }
    }
    assert_eq!(mcp.tool_call_count("get_weather"), 1);
}
