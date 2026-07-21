// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional test for the vLLM Agentic API passthrough example.

use std::collections::HashMap;

use praxis_core::config::Config;
use praxis_test_utils::{
    Recording, free_port, http_send, json_post, parse_body, parse_status, patch_yaml, start_capturing_backend,
    start_proxy,
};

fn load_agentic_api_config(proxy_port: u16, port_map: &HashMap<&str, u16>) -> Config {
    let path = praxis_test_utils::example_config_path("openai/responses/vllm-agentic-api.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = patch_yaml(&yaml, proxy_port, port_map);
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse vllm-agentic-api.yaml: {e}"))
}

#[test]
fn vllm_agentic_api_example_preserves_previous_response_id() {
    let backend =
        start_capturing_backend(r#"{"id":"resp_agentic_456","object":"response","status":"completed","output":[]}"#);
    let proxy_port = free_port();
    let config = load_agentic_api_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);
    let body = r#"{
        "model":"open-model",
        "input":"Continue the previous answer",
        "previous_response_id":"resp_agentic_123",
        "tools":[{"type":"web_search"}]
    }"#;

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(
        parse_status(&raw),
        200,
        "Agentic API upstream should receive the request"
    );
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured request body should be valid JSON");
    assert_eq!(forwarded["previous_response_id"], "resp_agentic_123");
    assert_eq!(forwarded["tools"], serde_json::json!([{"type":"web_search"}]));
    assert_eq!(forwarded["input"], "Continue the previous answer");
}

#[test]
fn vllm_agentic_api_example_replays_web_search_recording() {
    let recording = Recording::load("agentic_api/web_search_nonstreaming.json");
    let response_body = recording.response_body();
    let request_body = recording.request_body();
    let backend = start_capturing_backend(&response_body);
    let proxy_port = free_port();
    let config = load_agentic_api_config(proxy_port, &HashMap::from([("127.0.0.1:3001", backend.port())]));
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request_body));

    assert_eq!(
        parse_status(&raw),
        200,
        "recorded Agentic API response should reach the client"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parse_body(&raw)).expect("response body should be JSON"),
        recording.response.expect("recording should contain a JSON response"),
        "Praxis must relay Agentic API web-search output unchanged"
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&backend.body()).expect("captured request body should be valid JSON"),
        recording.request,
        "Praxis must forward the recorded web-search request unchanged"
    );
}
