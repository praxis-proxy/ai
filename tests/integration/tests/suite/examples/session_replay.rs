// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for stored-session replay fixtures.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, SessionReplay, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};

use super::load_example_config;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn replay_claude_messages_session_through_protocol_example() {
    let replay = SessionReplay::load("replay/claude/messages-basic.json");
    let turn = replay.single_turn();
    let backend_guard = Backend::fixed(&turn.response_body())
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .start_with_shutdown();
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-protocol.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post(turn.path(), &turn.request_body()));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("client body should be JSON");

    assert_eq!(status, 200, "Claude replay request should return 200");
    assert_eq!(
        &response, &turn.response,
        "client response should match the replayed Claude fixture response"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_codex_responses_session_through_full_flow_example() {
    let replay = SessionReplay::load("replay/codex/responses-basic.json");
    let turn = replay.single_turn();
    let backend_guard = Backend::fixed(&turn.response_body())
        .header("content-type", "application/json")
        .start_with_shutdown();
    let proxy_port = free_port();

    let db = TempSqlite::new("session_replay");
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/full-flow.yaml"))
        .expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        proxy_port,
        &HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post(turn.path(), &turn.request_body()));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let response: serde_json::Value = serde_json::from_str(&body).expect("client body should be JSON");

    assert_eq!(status, 200, "Codex replay request should return 200");
    assert_eq!(
        &response, &turn.response,
        "client response should match the replayed Codex fixture response"
    );

    let response_id = turn
        .response
        .get("id")
        .and_then(serde_json::Value::as_str)
        .expect("Codex replay response should have an id");
    let (get_status, get_body) = http_get(proxy.addr(), &format!("/v1/responses/{response_id}"), None);
    let stored: serde_json::Value = serde_json::from_str(&get_body).expect("stored response should be JSON");

    assert_eq!(get_status, 200, "replayed response should be retrievable");
    assert_eq!(
        stored, turn.response,
        "stored response should match the replayed Codex fixture response"
    );

    drop(proxy);
}
