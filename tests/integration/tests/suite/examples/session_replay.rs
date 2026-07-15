// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for stored-session replay fixtures.

use std::collections::HashMap;

use praxis_test_utils::{
    Backend, SessionReplay, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_capturing_backend, start_echo_backend, start_proxy,
};
use serde_json::json;

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

#[test]
fn replay_claude_messages_image_session_through_protocol_example() {
    let replay = SessionReplay::load("replay/claude/messages-image.json");
    let turn = replay.single_turn();
    let backend_guard = start_echo_backend();
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
    let forwarded: serde_json::Value = serde_json::from_str(&body).expect("echoed request body should be JSON");

    assert_eq!(status, 200, "Claude image replay request should return 200");
    assert_eq!(
        forwarded, turn.request,
        "backend should receive the image-bearing Claude request unchanged"
    );
    assert_eq!(
        forwarded["messages"][0]["content"][1]["source"]["media_type"], "image/png",
        "forwarded request should preserve the image content block"
    );

    drop(proxy);
}

#[test]
fn replay_claude_messages_tool_cycle_preserves_source_records() {
    let replay = SessionReplay::load("replay/claude/messages-tool-cycle.json");
    assert_eq!(
        replay.turns.len(),
        2,
        "tool-cycle replay should cover both request phases"
    );
    assert_eq!(
        replay.turns[0]
            .source_records
            .as_ref()
            .expect("first turn should preserve source records")
            .len(),
        3,
        "first turn should preserve user request plus split assistant text and tool_use records"
    );
    assert_eq!(
        replay.turns[1]
            .source_records
            .as_ref()
            .expect("second turn should preserve source records")
            .len(),
        2,
        "second turn should preserve tool_result and final assistant records"
    );
    for turn in &replay.turns {
        let backend_guard = start_capturing_backend(&turn.response_body());
        let proxy_port = free_port();

        let config = load_example_config(
            "anthropic/messages-protocol.yaml",
            proxy_port,
            HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
        );
        let proxy = start_proxy(&config);

        let raw = http_send(proxy.addr(), &json_post(turn.path(), &turn.request_body()));
        let response: serde_json::Value = serde_json::from_str(&parse_body(&raw)).expect("client body should be JSON");

        assert_eq!(parse_status(&raw), 200, "{} should return 200", turn.name);
        assert_eq!(
            response, turn.response,
            "{} client response should match the replay fixture response",
            turn.name
        );
        let forwarded: serde_json::Value =
            serde_json::from_str(&backend_guard.body()).expect("captured backend body should be JSON");
        assert_eq!(
            forwarded, turn.request,
            "{} backend request should match the replay fixture request",
            turn.name
        );

        drop(proxy);
    }
    assert_eq!(
        replay.turns[0].response["content"][0]["type"], "text",
        "first turn should preserve assistant text before the tool request"
    );
    assert_eq!(
        replay.turns[0].response["content"][1]["type"], "tool_use",
        "first turn should replay the assistant tool request"
    );
    assert_eq!(
        replay.turns[1].request["messages"][1]["content"][1]["type"], "tool_use",
        "second turn should include the assistant tool_use history required by Anthropic"
    );
    assert_eq!(
        replay.turns[1].request["messages"][1]["content"][1]["id"], "toolu_replay_bash_01",
        "second turn assistant history should expose the tool_use id"
    );
    assert_eq!(
        replay.turns[1].request["messages"][2]["content"][0]["type"], "tool_result",
        "second turn should preserve the client tool result request shape"
    );
    assert_eq!(
        replay.turns[1].request["messages"][2]["content"][0]["tool_use_id"],
        replay.turns[1].request["messages"][1]["content"][1]["id"],
        "tool_result should refer to the preceding assistant tool_use"
    );
    assert_eq!(
        replay.turns[1].source_records.as_ref().expect("source records")[0]["message"]["content"][0]["tool_use_id"],
        "toolu_replay_bash_01",
        "source records should retain the original tool_result linkage"
    );
}

#[test]
fn replay_claude_messages_thinking_session_through_protocol_example() {
    let replay = SessionReplay::load("replay/claude/messages-thinking.json");
    let turn = replay.single_turn();
    let backend_guard = start_capturing_backend(&turn.response_body());
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
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend_guard.body()).expect("captured backend body should be JSON");

    assert_eq!(status, 200, "Claude thinking replay request should return 200");
    assert_eq!(
        response, turn.response,
        "client response should match the replayable Anthropic response"
    );
    assert_eq!(
        forwarded, turn.request,
        "backend should receive the Claude request unchanged"
    );
    assert_eq!(
        turn.response["content"][0]["type"], "text",
        "fixture response should contain the replayable visible answer"
    );
    assert!(
        turn.response["content"][0]["text"]
            .as_str()
            .expect("fixture response text")
            .contains("NPV = $500,000"),
        "fixture should preserve Claude's visible NPV result"
    );

    let source_records = turn.source_records.as_ref().expect("thinking fixture source records");
    assert_eq!(
        source_records[1]["message"]["content"][0]["type"], "thinking",
        "source records should preserve Claude Code thinking records"
    );
    assert_eq!(
        source_records[2]["message"]["content"][0]["type"], "text",
        "source records should preserve the visible assistant response record"
    );
    assert_eq!(
        source_records[1]["message"]["id"], source_records[2]["message"]["id"],
        "split thinking and text records should retain their shared Claude message id"
    );

    drop(proxy);
}

#[test]
fn replay_claude_messages_image_session_through_chat_completions_translation_example() {
    let replay = SessionReplay::load("replay/claude/messages-image.json");
    let turn = replay.single_turn();
    let assistant_text = turn.response["content"][0]["text"]
        .as_str()
        .expect("fixture response should contain assistant text");
    let chat_response = json!({
        "id": "chatcmpl_replay_image",
        "object": "chat.completion",
        "model": turn.request["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 202, "total_tokens": 205}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post(turn.path(), &turn.request_body()));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let transformed: serde_json::Value = serde_json::from_str(&body).expect("client body should be JSON");
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    let image_data = turn.request["messages"][0]["content"][1]["source"]["data"]
        .as_str()
        .expect("fixture image data should be base64");

    assert_eq!(status, 200, "Claude image replay translation should return 200");
    assert_eq!(
        forwarded["messages"][0]["content"][1]["type"], "image_url",
        "backend should receive a Chat Completions image content part"
    );
    assert_eq!(
        forwarded["messages"][0]["content"][1]["image_url"]["url"],
        format!("data:image/png;base64,{image_data}"),
        "base64 Anthropic image should translate to Chat Completions data URL"
    );
    assert_eq!(
        transformed["content"][0]["text"], assistant_text,
        "Chat Completions response should translate back to Anthropic text content"
    );
    assert_eq!(
        transformed["stop_reason"], "end_turn",
        "Chat Completions stop finish reason should translate to Anthropic end_turn"
    );

    drop(proxy);
}

#[test]
fn replay_claude_messages_thinking_fixture_translates_visible_text_for_openai() {
    let replay = SessionReplay::load("replay/claude/messages-thinking.json");
    let turn = replay.single_turn();
    let visible_text = turn.response["content"][0]["text"]
        .as_str()
        .expect("fixture response should contain final assistant text");
    let chat_response = json!({
        "id": "chatcmpl_replay_thinking",
        "object": "chat.completion",
        "model": turn.request["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": visible_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
    });
    let backend = start_capturing_backend(&chat_response.to_string());
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend.port())]),
    );
    let proxy = start_proxy(&config);

    let raw = http_send(proxy.addr(), &json_post(turn.path(), &turn.request_body()));
    let status = parse_status(&raw);
    let body = parse_body(&raw);
    let transformed: serde_json::Value = serde_json::from_str(&body).expect("client body should be JSON");
    let forwarded: serde_json::Value =
        serde_json::from_str(&backend.body()).expect("captured backend body should be JSON");
    let messages = forwarded["messages"]
        .as_array()
        .expect("OpenAI request should contain messages");

    assert_eq!(status, 200, "Claude thinking replay translation should return 200");
    assert_eq!(
        messages.len(),
        1,
        "Claude Code NPV fixture should translate its single user replay request"
    );
    assert_eq!(
        messages[0]["role"], "user",
        "user prompt should be forwarded to Chat Completions"
    );
    assert_eq!(
        messages[0]["content"], turn.request["messages"][0]["content"],
        "user prompt text should be forwarded to Chat Completions"
    );
    assert_eq!(
        transformed["content"][0]["text"], visible_text,
        "Chat Completions response should translate back to Anthropic text content"
    );
    assert_eq!(
        transformed["stop_reason"], "end_turn",
        "Chat Completions stop finish reason should translate to Anthropic end_turn"
    );

    drop(proxy);
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
