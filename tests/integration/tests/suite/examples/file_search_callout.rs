// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration tests for the file-search-callout example config.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    ProxyGuard, free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_backend_with_shutdown,
    start_capturing_backend, start_proxy, start_stateful_backend,
};
use serde_json::{Value, json};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Load the example with its environment reference replaced by a test key.
fn load_file_search_callout_config(proxy_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let path = praxis_test_utils::example_config_path("openai/responses/file-search-callout.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse file-search-callout.yaml: {e}"))
}

/// Start a proxy whose test pipeline includes the Pingora subrequest connector.
fn start_file_search_proxy(config: &Config) -> ProxyGuard {
    start_proxy(config)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn file_search_callout_example_runs_model_search_model_round_trip() {
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
        // `start_proxy` probes `/` before returning the guard.
        (200, r#"{"status":"ready"}"#.to_owned()),
        (200, first_model_response.to_string()),
        (200, final_model_response.to_string()),
    ]);
    let search = start_capturing_backend(
        &json!({
            "data": [{
                "file_id": "file-q4",
                "filename": "q4-results.txt",
                "score": 0.99,
                "content": [{"type": "text", "text": "Q4 revenue was $42 million."}],
                "attributes": null
            }]
        })
        .to_string(),
    );
    let proxy_port = free_port();
    let config = load_file_search_callout_config(
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_file_search_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "What do the uploaded documents say about Q4 results?",
        "include": ["file_search_call.results"],
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}]
    });
    let request = json_post("/v1/responses", &request.to_string()).replacen(
        "Content-Type: application/json",
        "Authorization: Bearer inference-key\r\nX-Tenant-Id: tenant-q4\r\nContent-Type: application/json",
        1,
    );
    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200, "round trip failed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("final response should be JSON");
    assert_eq!(response["id"], "resp_final");
    assert_eq!(response["output"][0]["type"], "file_search_call");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][0]["results"][0]["file_id"], "file-q4");
    assert_eq!(response["output"][1]["type"], "message");
    assert_eq!(
        response["output"][1]["content"][0]["text"],
        "Q4 revenue was $42 million"
    );
    assert_eq!(
        response["output"][1]["content"][0]["annotations"][0]["file_id"],
        "file-q4"
    );
    assert_eq!(
        response["usage"],
        json!({"input_tokens": 30, "output_tokens": 12, "total_tokens": 42})
    );

    let search_request: Value = serde_json::from_str(&search.body()).expect("search request should be JSON");
    assert_eq!(search_request["query"], "What were the Q4 results?");
    assert_eq!(search_request["rewrite_query"], false);

    let model_requests = model.requests();
    let inference_requests = model_requests
        .iter()
        .filter(|request| request.starts_with("POST /v1/responses "))
        .collect::<Vec<_>>();
    assert_eq!(inference_requests.len(), 2);
    for request in inference_requests {
        let lowercase = request.to_ascii_lowercase();
        assert!(lowercase.contains("authorization: bearer inference-key\r\n"));
        assert!(lowercase.contains("x-tenant-id: tenant-q4\r\n"));
    }
}

#[test]
fn file_search_callout_example_without_tools_passthrough() {
    let response = r#"{"id":"resp_456","object":"response","output":[{"id":"msg_456","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}]}"#;
    let backend = start_backend_with_shutdown(response);
    let proxy_port = free_port();

    let config = load_file_search_callout_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_file_search_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "request failed: {raw}");
    assert_eq!(parse_body(&raw), response, "request should reach inference backend");
}

#[test]
fn file_search_callout_example_rejects_streaming_before_inference() {
    let backend = start_backend_with_shutdown(r#"{"id":"unexpected"}"#);
    let proxy_port = free_port();
    let config = load_file_search_callout_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_file_search_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello","stream":true}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 400, "streaming should be rejected: {raw}");
    assert!(
        parse_body(&raw).contains("stream=true is not supported"),
        "rejection should explain the pipeline limitation: {raw}"
    );
}

#[test]
fn file_search_callout_example_rejects_parallel_client_function_call() {
    let mixed_response = json!({
        "id": "resp_mixed",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "fs_1",
                "type": "file_search_call",
                "status": "searching",
                "queries": ["What were the Q4 results?"]
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_tax",
                "arguments": "{}",
                "status": "completed"
            }
        ]
    });
    let model = start_stateful_backend(vec![
        (200, r#"{"status":"ready"}"#.to_owned()),
        (200, mixed_response.to_string()),
    ]);
    let search = start_capturing_backend(&json!({"data": []}).to_string());
    let proxy_port = free_port();
    let config = load_file_search_callout_config(
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_file_search_proxy(&config);
    let request = json!({
        "model": "llama-3.3-70b",
        "input": "Combine the report and tax lookup.",
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_q4"]},
            {"type": "function", "name": "lookup_tax", "parameters": {"type": "object"}}
        ]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(parse_status(&raw), 502, "mixed-tool response should be rejected: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("mixed response should be JSON");
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("cannot combine")),
        "rejection should explain the unsupported combination: {response}"
    );
    let inference_calls = model
        .requests()
        .into_iter()
        .filter(|request| request.starts_with("POST /v1/responses "))
        .count();
    assert_eq!(
        inference_calls, 1,
        "client function output must arrive before reinference"
    );
}
