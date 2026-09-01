// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the Anthropic Messages web-search loop.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
};

use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_body, parse_status,
    patch_yaml, start_proxy,
};
use serde_json::{Value, json};

struct SearchStub {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl SearchStub {
    fn start(response: &Value) -> Self {
        Self::start_many(std::slice::from_ref(response))
    }

    fn start_many(responses: &[Value]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind You.com stub");
        let port = listener.local_addr().expect("stub address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let bodies = responses.iter().map(Value::to_string).collect::<Vec<_>>();
        std::thread::spawn(move || {
            for body in bodies {
                let (mut stream, _) = listener.accept().expect("accept search request");
                captured
                    .lock()
                    .expect("capture search request")
                    .push(read_http_request(&mut stream));
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).expect("write search response");
            }
        });
        Self { port, requests }
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("read search requests").len()
    }

    fn last_request(&self) -> String {
        self.requests
            .lock()
            .expect("read search requests")
            .last()
            .expect("search request exists")
            .clone()
    }

    fn last_json(&self) -> Value {
        let request = self.last_request();
        let (_, body) = request.split_once("\r\n\r\n").expect("search request body");
        serde_json::from_str(body).expect("search request JSON")
    }

    fn request_json(&self, index: usize) -> Value {
        let request = self.requests.lock().expect("read search requests")[index].clone();
        let (_, body) = request.split_once("\r\n\r\n").expect("search request body");
        serde_json::from_str(body).expect("search request JSON")
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut request = Vec::new();
    loop {
        let mut buffer = [0_u8; 4096];
        let count = stream.read(&mut buffer).expect("read search request");
        assert!(count > 0, "search request closed before its body arrived");
        request.extend_from_slice(&buffer[..count]);

        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
        let content_length = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .map(str::trim)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        if request.len() >= body_start + content_length {
            return String::from_utf8(request).expect("search request UTF-8");
        }
    }
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
    ))
    .expect("parse web-search fixture")
}

fn load_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    load_config_with_limits(proxy_port, model_port, search_port, None, None)
}

fn load_config_with_max_state_bytes(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    max_state_bytes: usize,
) -> praxis_core::config::Config {
    load_config_with_limits(proxy_port, model_port, search_port, Some(max_state_bytes), None)
}

fn load_config_with_limits(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    max_state_bytes: Option<usize>,
    max_body_bytes: Option<usize>,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path("anthropic/messages-web-search.yaml"))
        .expect("read Messages web-search example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", model_port)]));
    let mut yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!("api_key: test-key\n                base_url: http://127.0.0.1:{search_port}"),
    );
    if let Some(max_state_bytes) = max_state_bytes {
        yaml = yaml.replace(
            "max_iterations: 6",
            &format!("max_iterations: 6\n        max_state_bytes: {max_state_bytes}"),
        );
    }
    if let Some(max_body_bytes) = max_body_bytes {
        yaml = yaml.replace(
            "timeout_ms: 10000",
            &format!("timeout_ms: 10000\n                max_body_bytes: {max_body_bytes}"),
        );
    }
    praxis_core::config::Config::from_yaml(&yaml).expect("parse Messages web-search example")
}

#[test]
fn messages_web_search_round_trip_re_enters_the_model() {
    let fixture = fixture();
    let mut first_model_response = fixture["first_model_response"].clone();
    let tool_use = first_model_response["content"][0].clone();
    first_model_response["content"] = json!([
        {"type":"text","text":"I will search before answering."},
        tool_use
    ]);
    let model = StatefulCapturingBackend::new(vec![
        (200, first_model_response.to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let search = SearchStub::start(&fixture["search_response"]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(parse_status(&raw), 200);
    let client_response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(client_response, fixture["final_model_response"]);

    let requests = model.requests();
    assert_eq!(requests.len(), 2, "model should receive two Messages requests");
    assert_eq!(requests[0].uri, "/v1/messages");
    assert_eq!(requests[1].uri, "/v1/messages");
    let second: Value = serde_json::from_str(&requests[1].body).expect("second model request JSON");
    assert_eq!(second["model"], fixture["initial_request"]["model"]);
    assert_eq!(second["tools"], fixture["initial_request"]["tools"]);
    let messages = second["messages"].as_array().expect("Messages history");
    assert_eq!(messages[messages.len() - 2]["content"], first_model_response["content"]);
    assert_eq!(messages[messages.len() - 1]["content"][0]["type"], "tool_result");
    assert_eq!(
        messages[messages.len() - 1]["content"][0]["tool_use_id"],
        "toolu_web_search_01"
    );
    assert!(
        messages[messages.len() - 1]["content"][0]["content"]
            .as_str()
            .is_some_and(|content| content.contains("Potato - Wikipedia"))
    );
    assert_eq!(search.request_count(), 1);
    assert_eq!(search.last_json()["query"], "potato");
    assert!(
        search
            .last_request()
            .to_ascii_lowercase()
            .contains("x-api-key: test-key")
    );
}

#[test]
fn caller_anthropic_headers_are_preserved_across_model_reentry() {
    let fixture = fixture();
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let search = SearchStub::start(&fixture["search_response"]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));
    let body = fixture["initial_request"].to_string();
    let request = json_post_with_headers(
        "/v1/messages",
        &body,
        &[
            ("anthropic-version", "2024-01-01"),
            ("anthropic-beta", "test-beta-2026-01-01"),
        ],
    );

    let raw = http_send(proxy.addr(), &request);

    assert_eq!(parse_status(&raw), 200);
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "model should receive two Messages requests");
    for (index, request) in requests.iter().enumerate() {
        let headers = request.headers.to_ascii_lowercase();
        assert!(
            headers.contains("anthropic-version: 2024-01-01"),
            "model request {index} should preserve the caller's anthropic-version; headers: {}",
            request.headers
        );
        assert!(
            headers.contains("anthropic-beta: test-beta-2026-01-01"),
            "model request {index} should preserve the caller's anthropic-beta; headers: {}",
            request.headers
        );
    }
}

#[test]
fn non_success_tool_use_response_passes_through_without_search_or_reentry() {
    let fixture = fixture();
    let upstream = fixture["first_model_response"].clone();
    let model = StatefulCapturingBackend::new(vec![(429, upstream.to_string())]).start_with_shutdown();
    let search = SearchStub::start(&fixture["search_response"]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(parse_status(&raw), 429);
    assert_eq!(
        serde_json::from_str::<Value>(&parse_body(&raw)).expect("client response JSON"),
        upstream
    );
    assert_eq!(model.requests().len(), 1, "429 must not re-enter the model");
    assert_eq!(search.request_count(), 0, "429 must not dispatch a managed search");
}

#[test]
fn two_sequential_web_searches_retain_ordered_tool_history() {
    let fixture = fixture();
    let first_response = fixture["first_model_response"].clone();
    let second_response = json!({
        "id":"msg_web_search_02",
        "type":"message",
        "role":"assistant",
        "model":"openai/gpt-oss-20b",
        "content":[{
            "type":"tool_use",
            "id":"toolu_web_search_02",
            "name":"WebSearch",
            "input":{"query":"potato cultivation"}
        }],
        "stop_reason":"tool_use",
        "stop_sequence":null,
        "usage":{"input_tokens":74,"output_tokens":9}
    });
    let final_response = fixture["final_model_response"].clone();
    let model = StatefulCapturingBackend::new(vec![
        (200, first_response.to_string()),
        (200, second_response.to_string()),
        (200, final_response.to_string()),
    ])
    .start_with_shutdown();
    let second_search_response = json!({
        "results":{"web":[{
            "title":"Growing potatoes",
            "url":"https://example.com/growing-potatoes",
            "description":"Potatoes prefer cool weather and loose soil."
        }],"news":[]}
    });
    let search = SearchStub::start_many(&[fixture["search_response"].clone(), second_search_response]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(parse_status(&raw), 200);
    let requests = model.requests();
    assert_eq!(requests.len(), 3, "two searches require three model requests");
    assert_eq!(search.request_count(), 2, "each managed call requires one search");
    assert_eq!(search.request_json(0)["query"], "potato");
    assert_eq!(search.request_json(1)["query"], "potato cultivation");

    let third: Value = serde_json::from_str(&requests[2].body).expect("third model request JSON");
    let messages = third["messages"].as_array().expect("Messages history");
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[1]["content"], first_response["content"]);
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_web_search_01");
    assert_eq!(messages[3]["content"], second_response["content"]);
    assert_eq!(messages[4]["content"][0]["tool_use_id"], "toolu_web_search_02");
}

#[test]
fn streaming_request_is_rejected_before_model_or_search() {
    let fixture = fixture();
    let model =
        StatefulCapturingBackend::new(vec![(200, fixture["final_model_response"].to_string())]).start_with_shutdown();
    let search = SearchStub::start(&fixture["search_response"]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));
    let mut request = fixture["initial_request"].clone();
    request["stream"] = Value::Bool(true);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request.to_string()));

    assert_eq!(parse_status(&raw), 400);
    assert!(parse_body(&raw).contains("streaming is not supported"));
    assert!(model.requests().is_empty(), "streaming request must not reach model");
    assert_eq!(search.request_count(), 0, "streaming request must not call search");
}

#[test]
fn state_limit_rejects_before_large_search_result_reenters_model() {
    let fixture = fixture();
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let mut large_search_response = fixture["search_response"].clone();
    large_search_response["results"]["web"][0]["description"] = Value::String("x".repeat(40_000));
    let search = SearchStub::start(&large_search_response);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_max_state_bytes(
        proxy_port,
        model.port(),
        search.port(),
        20_000,
    ));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(parse_status(&raw), 413);
    assert_eq!(
        model.requests().len(),
        1,
        "oversized retained state must halt before model re-entry"
    );
    assert_eq!(
        search.request_count(),
        1,
        "the result must trigger retained-state growth"
    );
}

#[test]
fn body_limit_rejects_before_large_rebuilt_request_reenters_model() {
    let fixture = fixture();
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let mut large_search_response = fixture["search_response"].clone();
    large_search_response["results"]["web"][0]["description"] = Value::String("x".repeat(40_000));
    let search = SearchStub::start(&large_search_response);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_limits(
        proxy_port,
        model.port(),
        search.port(),
        None,
        Some(20_000),
    ));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(parse_status(&raw), 413);
    assert_eq!(
        model.requests().len(),
        1,
        "oversized rebuilt body must halt before model re-entry"
    );
    assert_eq!(search.request_count(), 1, "the result must trigger rebuilt-body growth");
}

fn json_post_with_headers(path: &str, body: &str, headers: &[(&str, &str)]) -> String {
    let mut extra = String::new();
    for (name, value) in headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         {extra}\
         \r\n\
         {body}",
        body.len(),
    )
}
