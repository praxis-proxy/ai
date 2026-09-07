// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the full-flow-agentic example config.

use std::collections::HashMap;

use praxis_test_utils::{
    StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_backend_with_shutdown, start_proxy, start_stateful_backend,
};
use serde_json::{Value, json};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn load_full_flow_agentic_config(
    proxy_port: u16,
    port_map: &HashMap<&str, u16>,
) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new("full_flow_agentic");
    let path = example_config_path("openai/responses/full-flow-agentic.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let yaml = yaml.replace("sqlite://responses.db?mode=rwc", db.url());
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    let config = praxis_core::config::Config::from_yaml(&patched)
        .unwrap_or_else(|e| panic!("parse full-flow-agentic.yaml: {e}"));
    (config, db)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn full_flow_agentic_single_pass_completes() {
    let backend =
        start_backend_with_shutdown(r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#);
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "single-pass request through IRR should succeed"
    );
}

#[test]
fn full_flow_agentic_file_search_round_trip() {
    let first_model_response = json!({
        "id": "resp_search",
        "object": "response",
        "status": "completed",
        "output": [{
            "id": "fs_1",
            "type": "file_search_call",
            "status": "searching",
            "queries": ["What were the Q4 results?"]
        }],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    });
    let final_model_response = json!({
        "id": "resp_final",
        "object": "response",
        "status": "completed",
        "output": [{
            "id": "msg_final",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": "Q4 revenue was $42 million <|file-q4|>",
                "annotations": []
            }]
        }],
        "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}
    });
    let model = start_stateful_backend(vec![
        (200, first_model_response.to_string()),
        (200, final_model_response.to_string()),
    ]);
    let search_response = json!({
        "data": [{
            "file_id": "file-q4",
            "filename": "q4-results.txt",
            "score": 0.99,
            "content": [{"type": "text", "text": "Q4 revenue was $42 million."}],
            "attributes": null
        }]
    });
    let search = StatefulCapturingBackend::new(vec![(200, search_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:3002", search.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "gpt-4.1",
        "input": "What do the documents say about Q4?",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}]
    });
    let request = json_post("/v1/responses", &request.to_string()).replacen(
        "Content-Type: application/json",
        "Authorization: Bearer search-key\r\nContent-Type: application/json",
        1,
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "file search round trip failed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("response should be JSON");
    assert_eq!(response["id"], "resp_final");
    assert_eq!(response["output"][0]["type"], "file_search_call");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][1]["type"], "message");

    let search_requests = search.requests();
    let search_callouts: Vec<_> = search_requests.iter().filter(|r| r.method == "POST").collect();
    assert_eq!(search_callouts.len(), 1, "expected one vector store callout");
    assert!(
        search_callouts[0]
            .headers
            .to_lowercase()
            .contains("authorization: bearer search-key"),
        "vector store callout should forward the authorization header: {}",
        search_callouts[0].headers,
    );
}

#[test]
fn full_flow_agentic_without_tools_passthrough() {
    let response = r#"{"id":"resp_456","object":"response","output":[{"id":"msg_456","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}]}"#;
    let backend = start_backend_with_shutdown(response);
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", r#"{"model":"gpt-4.1","input":"Hello"}"#),
    );

    assert_eq!(parse_status(&raw), 200, "request failed: {raw}");
    assert_eq!(parse_body(&raw), response, "request without tools should pass through");
}

#[test]
fn full_flow_agentic_rejects_non_responses_path() {
    let backend =
        start_backend_with_shutdown(r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#);
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), "GET /v1/prompts HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");

    let status = parse_status(&raw);
    assert_ne!(status, 200, "non-responses path should not reach a backend: {raw}");
}

#[test]
fn full_flow_agentic_rejects_responses_subpath() {
    let backend =
        start_backend_with_shutdown(r#"{"id":"resp_1","object":"response","status":"completed","output":[]}"#);
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses/resp_123/cancel", r#"{"model":"gpt-4.1"}"#),
    );

    let status = parse_status(&raw);
    assert_ne!(status, 200, "subpath should not reach inference backend: {raw}");
}

#[test]
fn full_flow_agentic_connection_nominated_header_not_forwarded() {
    let first_model_response = json!({
        "id": "resp_conn",
        "object": "response",
        "status": "completed",
        "output": [{
            "id": "fs_conn",
            "type": "file_search_call",
            "status": "searching",
            "queries": ["test query"]
        }],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    });
    let final_model_response = json!({
        "id": "resp_conn_final",
        "object": "response",
        "status": "completed",
        "output": [{
            "id": "msg_conn",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}]
        }],
        "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}
    });
    let model = start_stateful_backend(vec![
        (200, first_model_response.to_string()),
        (200, final_model_response.to_string()),
    ]);
    let search_response = json!({
        "data": [{
            "file_id": "file-conn",
            "filename": "test.txt",
            "score": 0.9,
            "content": [{"type": "text", "text": "test content"}],
            "attributes": null
        }]
    });
    let search = StatefulCapturingBackend::new(vec![(200, search_response.to_string())]).start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_full_flow_agentic_config(
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:3002", search.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "gpt-4.1",
        "input": "test",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_test"]}]
    });
    let request = json_post("/v1/responses", &request.to_string()).replacen(
        "Content-Type: application/json",
        "Authorization: Bearer secret\r\nConnection: authorization\r\nContent-Type: application/json",
        1,
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "round trip should succeed: {raw}");
    let search_requests = search.requests();
    let search_callouts: Vec<_> = search_requests.iter().filter(|r| r.method == "POST").collect();
    assert_eq!(search_callouts.len(), 1, "expected one vector store callout");
    assert!(
        !search_callouts[0].headers.to_lowercase().contains("authorization"),
        "connection-nominated authorization must not be forwarded to vector store: {}",
        search_callouts[0].headers,
    );
}
