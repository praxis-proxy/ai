// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for stored-session replay fixtures.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    time::Duration,
};

use praxis_test_utils::{
    Backend, SessionReplay, TempSqlite, example_config_path, free_port, http_get, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_echo_backend, start_proxy,
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
    let backend = start_capturing_backend(chat_response.to_string());
    let proxy_port = free_port();

    let config = load_example_config(
        "anthropic/messages-to-openai.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:8000", backend.port)]),
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

struct CapturingBackend {
    port: u16,
    body: Arc<Mutex<Option<String>>>,
}

impl CapturingBackend {
    fn body(&self) -> String {
        self.body
            .lock()
            .expect("captured body mutex should not be poisoned")
            .clone()
            .expect("backend should capture one request body")
    }
}

fn start_capturing_backend(response_body: String) -> CapturingBackend {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind capturing backend");
    let port = listener.local_addr().expect("capturing backend address").port();
    let body = Arc::new(Mutex::new(None));
    let captured = Arc::clone(&body);

    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("set read timeout");
            let request_body = read_request_body(&mut stream);
            if !request_body.is_empty() {
                *captured.lock().expect("captured body mutex should not be poisoned") = Some(request_body);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _sent = stream.write_all(response.as_bytes());
        }
    });

    CapturingBackend { port, body }
}

fn read_request_body(stream: &mut TcpStream) -> String {
    let mut data = Vec::new();
    let mut buf = [0_u8; 4096];

    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
        }
        if request_body_complete(&data) {
            break;
        }
    }

    String::from_utf8_lossy(&data)
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .to_owned()
}

fn request_body_complete(data: &[u8]) -> bool {
    let raw = String::from_utf8_lossy(data);
    let Some(header_section) = raw.split("\r\n\r\n").next() else {
        return false;
    };
    let content_length = header_section
        .lines()
        .find(|line| line.to_lowercase().starts_with("content-length:"))
        .and_then(|line| line.split_once(':').map(|(_, value)| value))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);

    data.len() >= header_section.len() + 4 + content_length
}
