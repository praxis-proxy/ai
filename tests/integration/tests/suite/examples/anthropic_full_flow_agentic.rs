// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the Anthropic Messages full-flow agentic example
//! (`anthropic/full-flow-agentic.yaml`).
//!
//! A single example config serves the server-owned web-search loop in BOTH
//! modes via `terminal_streaming: true`, selected per request from the client's
//! `stream` flag:
//!
//!   * `stream: true`  — the managed `WebSearch` tool-use block is suppressed, intermediate rounds stay internal, and
//!     the terminal Messages response is delivered incrementally as one coherent client-visible SSE lifecycle across
//!     IRR re-entry.
//!   * `stream: false` — the same loop runs buffered and returns one final Anthropic Messages JSON object.
//!
//! The buffered tests exercise the JSON path; the streaming tests exercise the
//! incremental SSE path. Both run against the one unified example.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_body, parse_status,
    patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "anthropic/full-flow-agentic.yaml";
const TOOL_USE_ID: &str = "toolu_web_search_01";

// -----------------------------------------------------------------------------
// SSE builders (native Anthropic Messages lifecycle)
// -----------------------------------------------------------------------------

/// Encode one native Messages SSE event.
fn sse(event_type: &str, data: &Value) -> String {
    format!("event: {event_type}\ndata: {data}\n\n")
}

/// A `message_start` frame carrying a stable message id.
fn message_start(id: &str) -> String {
    sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": "openai/gpt-oss-20b",
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 20, "output_tokens": 0}
            }
        }),
    )
}

/// A text content block's start/delta/stop frames at a backend `index`.
fn text_block(index: u64, text: &str) -> String {
    let mut bytes = sse(
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""}
        }),
    );
    bytes.push_str(&sse(
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text}
        }),
    ));
    bytes.push_str(&sse(
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    ));
    bytes
}

/// A managed `WebSearch` `tool_use` block's start/delta/stop frames.
fn web_search_block(index: u64, id: &str, query: &str) -> String {
    let partial_json = json!({"query": query}).to_string();
    let mut bytes = sse(
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": "WebSearch", "input": {}}
        }),
    );
    bytes.push_str(&sse(
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json}
        }),
    ));
    bytes.push_str(&sse(
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    ));
    bytes
}

/// A `message_delta` frame carrying a stop reason and output-token count.
fn message_delta(stop_reason: &str, output_tokens: u64) -> String {
    sse(
        "message_delta",
        &json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        }),
    )
}

/// A `message_stop` frame.
fn message_stop() -> String {
    sse("message_stop", &json!({"type": "message_stop"}))
}

/// A complete managed-search round: `message_start` -> suppressed `WebSearch`
/// tool-use -> `message_delta`(tool_use) -> `message_stop`.
fn search_round(id: &str, tool_use_id: &str, query: &str, output_tokens: u64) -> String {
    let mut round = message_start(id);
    round.push_str(&web_search_block(0, tool_use_id, query));
    round.push_str(&message_delta("tool_use", output_tokens));
    round.push_str(&message_stop());
    round
}

/// A complete terminal round: `message_start` -> text answer ->
/// `message_delta`(end_turn) -> `message_stop`.
fn answer_round(id: &str, text: &str, output_tokens: u64) -> String {
    let mut round = message_start(id);
    round.push_str(&text_block(0, text));
    round.push_str(&message_delta("end_turn", output_tokens));
    round.push_str(&message_stop());
    round
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

/// The buffered non-streaming web-search fixture (initial request, model rounds,
/// and search response).
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/anthropic/messages/web_search_nonstreaming.json"
    ))
    .expect("parse web-search fixture")
}

/// The initial client request with a `WebSearch` tool and `stream: true`.
fn streaming_request() -> String {
    json!({
        "model": "openai/gpt-oss-20b",
        "max_tokens": 1024,
        "stream": true,
        "messages": [{"role": "user", "content": "Use web search to look up potato, then summarize."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }]
    })
    .to_string()
}

/// A You.com-shaped search response body.
fn search_results() -> Value {
    json!({
        "results": {
            "web": [{
                "title": "Potato - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Potato",
                "description": "The potato is a starchy tuber native to the Americas."
            }],
            "news": []
        }
    })
}

// -----------------------------------------------------------------------------
// Config loaders
// -----------------------------------------------------------------------------

/// Read the unified example and rewrite the proxy/backend ports and the search
/// provider endpoint so the loop calls the local stubs.
fn base_example_yaml(proxy_port: u16, model_port: u16, search_port: u16) -> String {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read full-flow-agentic example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", model_port)]));
    yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!(
            "api_key: test-key\n                base_url: http://127.0.0.1:{search_port}\n                allow_private_base_url: true"
        ),
    )
}

fn load_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    praxis_core::config::Config::from_yaml(&base_example_yaml(proxy_port, model_port, search_port))
        .expect("parse full-flow-agentic example")
}

/// Override the IRR `max_iterations` so a test can drive the loop into the
/// router's iteration ceiling.
fn load_config_with_max_iterations(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    max_iterations: u32,
) -> praxis_core::config::Config {
    let yaml = base_example_yaml(proxy_port, model_port, search_port)
        .replace("max_iterations: 6", &format!("max_iterations: {max_iterations}"));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse full-flow-agentic example")
}

/// Override the IRR `timeout_ms` so a round that stalls mid-stream trips the
/// router deadline (an abnormal stream termination).
fn load_config_with_timeout(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    timeout_ms: u32,
) -> praxis_core::config::Config {
    let yaml = base_example_yaml(proxy_port, model_port, search_port)
        .replace("timeout_ms: 90000", &format!("timeout_ms: {timeout_ms}"));
    praxis_core::config::Config::from_yaml(&yaml).expect("parse full-flow-agentic example")
}

fn load_config_with_max_state_bytes(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    max_state_bytes: usize,
) -> praxis_core::config::Config {
    load_config_with_limits(proxy_port, model_port, search_port, Some(max_state_bytes), None)
}

/// Optionally inject an IRR `max_state_bytes` and/or a filter `max_body_bytes`
/// cap so a test can assert the loop halts before an oversized re-entry.
fn load_config_with_limits(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    max_state_bytes: Option<usize>,
    max_body_bytes: Option<usize>,
) -> praxis_core::config::Config {
    let mut yaml = base_example_yaml(proxy_port, model_port, search_port);
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
    praxis_core::config::Config::from_yaml(&yaml).expect("parse full-flow-agentic example")
}

// -----------------------------------------------------------------------------
// You.com search stub (serves ordered responses, captures each request)
// -----------------------------------------------------------------------------

struct SearchStub {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl SearchStub {
    fn start(response: &Value) -> Self {
        Self::start_many(std::slice::from_ref(response))
    }

    fn start_many(responses: &[Value]) -> Self {
        Self::start_many_with_status(responses, "200 OK")
    }

    /// Serve a single failing HTTP response so the loop maps the provider
    /// callout to a failed outcome and continues with an error tool result.
    fn start_failing() -> Self {
        Self::start_many_with_status(&[json!({"error": "service unavailable"})], "503 Service Unavailable")
    }

    fn start_many_with_status(responses: &[Value], status_line: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind search stub");
        let port = listener.local_addr().expect("stub address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let bodies = responses.iter().map(Value::to_string).collect::<Vec<_>>();
        let status_line = status_line.to_owned();
        thread::spawn(move || {
            for body in bodies {
                let (mut stream, _) = listener.accept().expect("accept search request");
                captured
                    .lock()
                    .expect("capture search request")
                    .push(read_full_request(&mut stream));
                let response = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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

    fn request_json(&self, index: usize) -> Value {
        let request = self.requests.lock().expect("read search requests")[index].clone();
        let (_, body) = request.split_once("\r\n\r\n").expect("search request body");
        serde_json::from_str(body).expect("search request JSON")
    }

    fn last_json(&self) -> Value {
        let request = self.last_request();
        let (_, body) = request.split_once("\r\n\r\n").expect("search request body");
        serde_json::from_str(body).expect("search request JSON")
    }

    fn query(&self) -> String {
        self.request_json(0)["query"].as_str().expect("search query").to_owned()
    }
}

// -----------------------------------------------------------------------------
// Streaming model backend (sequential SSE rounds, capturing requests)
// -----------------------------------------------------------------------------

/// A native Messages backend that serves sequential SSE rounds over chunked
/// transfer encoding and captures each round's request.
struct StreamingModel {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl StreamingModel {
    /// Serve each entry as one round's complete SSE body.
    fn start(rounds: Vec<String>) -> Self {
        Self::start_fragmented(rounds.into_iter().map(|round| vec![round]).collect())
    }

    /// Serve each round as an ordered list of transport fragments so a single
    /// SSE event can be split across chunks.
    fn start_fragmented(rounds: Vec<Vec<String>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind streaming model");
        let port = listener.local_addr().expect("model address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        thread::spawn(move || {
            for fragments in rounds {
                let (mut stream, _) = listener.accept().expect("accept model round");
                let request = read_full_request(&mut stream);
                captured.lock().expect("capture model request").push(request);
                write_stream_headers(&mut stream, "text/event-stream");
                for fragment in fragments {
                    write_chunk(&mut stream, &fragment);
                    stream.flush().expect("flush model chunk");
                }
                stream.write_all(b"0\r\n\r\n").expect("finish chunked round");
                stream.flush().expect("flush terminal chunk");
            }
        });
        Self { port, requests }
    }

    /// Serve `rounds` as normal SSE rounds, then serve one additional re-entry
    /// round that returns a non-2xx status with a plain JSON body. This models a
    /// backend that fails a later round after the client-visible stream has
    /// already started, so the raw error body must not be forwarded downstream.
    fn start_then_error(rounds: Vec<String>, status_line: &str, error_body: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind streaming model");
        let port = listener.local_addr().expect("model address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let status_line = status_line.to_owned();
        thread::spawn(move || {
            for round in rounds {
                let (mut stream, _) = listener.accept().expect("accept model round");
                let request = read_full_request(&mut stream);
                captured.lock().expect("capture model request").push(request);
                write_stream_headers(&mut stream, "text/event-stream");
                write_chunk(&mut stream, &round);
                stream.write_all(b"0\r\n\r\n").expect("finish chunked round");
                stream.flush().expect("flush terminal chunk");
            }
            let (mut stream, _) = listener.accept().expect("accept failing re-entry round");
            let request = read_full_request(&mut stream);
            captured.lock().expect("capture model request").push(request);
            let response = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{error_body}",
                error_body.len()
            );
            stream
                .write_all(response.as_bytes())
                .expect("write failing re-entry round");
        });
        Self { port, requests }
    }

    /// Serve `rounds` as normal SSE rounds, then accept one more connection, send
    /// `partial` (an incomplete SSE round) and hold the connection open without
    /// ever completing it. A short IRR `timeout_ms` then expires that round
    /// mid-stream, which Praxis surfaces as an abnormal stream termination
    /// (`DeadlineExceeded`) rather than a clean upstream EOF.
    fn start_then_hang(rounds: Vec<String>, partial: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind streaming model");
        let port = listener.local_addr().expect("model address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        thread::spawn(move || {
            for round in rounds {
                let (mut stream, _) = listener.accept().expect("accept model round");
                let request = read_full_request(&mut stream);
                captured.lock().expect("capture model request").push(request);
                write_stream_headers(&mut stream, "text/event-stream");
                write_chunk(&mut stream, &round);
                stream.write_all(b"0\r\n\r\n").expect("finish chunked round");
                stream.flush().expect("flush terminal chunk");
            }
            let (mut stream, _) = listener.accept().expect("accept stalling round");
            let request = read_full_request(&mut stream);
            captured.lock().expect("capture model request").push(request);
            write_stream_headers(&mut stream, "text/event-stream");
            write_chunk(&mut stream, &partial);
            stream.flush().expect("flush partial chunk");
            // Never complete the round: hold the connection open past the IRR
            // deadline so the round terminates abnormally mid-stream. The process
            // exits and reaps this thread when the test ends.
            thread::sleep(Duration::from_secs(30));
            drop(stream);
        });
        Self { port, requests }
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("read model requests").len()
    }

    fn request_json(&self, index: usize) -> Value {
        let request = self.requests.lock().expect("read model requests")[index].clone();
        let (_, body) = request.split_once("\r\n\r\n").expect("model request body");
        serde_json::from_str(body).expect("model request JSON")
    }
}

// -----------------------------------------------------------------------------
// Buffered-mode tests (stream: false — one final JSON object)
// -----------------------------------------------------------------------------

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
fn provider_failure_appends_is_error_tool_result_and_re_enters_model() {
    let fixture = fixture();
    let model = StatefulCapturingBackend::new(vec![
        (200, fixture["first_model_response"].to_string()),
        (200, fixture["final_model_response"].to_string()),
    ])
    .start_with_shutdown();
    let search = SearchStub::start_failing();
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/messages", &fixture["initial_request"].to_string()),
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "a provider failure must not reject the request"
    );
    let client_response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        client_response, fixture["final_model_response"],
        "the loop must return the model's post-failure answer"
    );

    let requests = model.requests();
    assert_eq!(
        requests.len(),
        2,
        "the loop must re-enter the model after the search fails"
    );
    let second: Value = serde_json::from_str(&requests[1].body).expect("second model request JSON");
    let messages = second["messages"].as_array().expect("Messages history");
    let tool_result = &messages[messages.len() - 1]["content"][0];
    assert_eq!(tool_result["type"], "tool_result");
    assert_eq!(tool_result["tool_use_id"], "toolu_web_search_01");
    assert_eq!(
        tool_result["is_error"], true,
        "a provider failure must produce a truthful is_error result"
    );
    assert_eq!(
        tool_result["content"], "Web search unavailable.",
        "the model must receive the bounded failure notice"
    );
    assert_eq!(
        search.request_count(),
        1,
        "the failed search still counts as one callout"
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
fn stream_false_preserves_buffered_loop() {
    // With terminal_streaming enabled but stream:false, the buffered loop is
    // unchanged: the backend serves JSON and the client receives one JSON body.
    let first = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "openai/gpt-oss-20b",
        "content": [{"type": "tool_use", "id": TOOL_USE_ID, "name": "WebSearch", "input": {"query": "potato"}}],
        "stop_reason": "tool_use", "stop_sequence": null, "usage": {"input_tokens": 20, "output_tokens": 8}
    });
    let final_answer = json!({
        "id": "msg_2", "type": "message", "role": "assistant", "model": "openai/gpt-oss-20b",
        "content": [{"type": "text", "text": "Potato is a tuber."}],
        "stop_reason": "end_turn", "stop_sequence": null, "usage": {"input_tokens": 74, "output_tokens": 18}
    });
    let model = StatefulCapturingBackend::new(vec![(200, first.to_string()), (200, final_answer.to_string())])
        .start_with_shutdown();
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));
    let mut request: Value = serde_json::from_str(&streaming_request()).expect("request JSON");
    request["stream"] = Value::Bool(false);

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request.to_string()));

    assert_eq!(parse_status(&raw), 200, "the buffered loop returns 200: {raw}");
    let client_response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        client_response, final_answer,
        "the buffered loop returns the final JSON answer"
    );
    assert_eq!(model.requests().len(), 2, "the buffered loop re-enters the model");
    assert_eq!(search.request_count(), 1, "the buffered loop dispatches one search");
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

// -----------------------------------------------------------------------------
// Streaming-mode tests (stream: true — one coherent SSE lifecycle)
// -----------------------------------------------------------------------------

#[test]
fn terminal_answer_streams_as_one_lifecycle() {
    let model = StreamingModel::start(vec![answer_round("msg_1", "Potato is a tuber.", 12)]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(parse_status(&raw), 200, "a streamed terminal answer returns 200: {raw}");
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "one message_start: {body}"
    );
    assert!(
        body.contains("event: content_block_start"),
        "text block forwarded: {body}"
    );
    assert!(
        body.contains("\"text\":\"Potato is a tuber.\""),
        "text delta forwarded: {body}"
    );
    assert!(
        body.contains("event: message_delta"),
        "terminal message_delta emitted: {body}"
    );
    assert!(
        body.contains("\"output_tokens\":12"),
        "terminal output tokens carried: {body}"
    );
    assert_eq!(
        body.matches("event: message_stop").count(),
        1,
        "one message_stop: {body}"
    );
    assert_eq!(model.request_count(), 1, "a direct answer needs one model round");
    assert_eq!(search.request_count(), 0, "no managed search on a direct answer");
}

#[test]
fn managed_web_search_is_suppressed_and_final_answer_streams() {
    let model = StreamingModel::start(vec![
        search_round("msg_1", TOOL_USE_ID, "potato", 8),
        answer_round("msg_2", "Potato is a starchy tuber.", 18),
    ]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(parse_status(&raw), 200, "the streamed loop returns 200: {raw}");
    let body = parse_body(&raw);
    // One coherent lifecycle: a single message_start and message_stop span both
    // rounds, and the intermediate managed round is fully internal.
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "one message_start: {body}"
    );
    assert_eq!(
        body.matches("event: message_stop").count(),
        1,
        "one message_stop: {body}"
    );
    assert_eq!(
        body.matches("event: message_delta").count(),
        1,
        "only the terminal message_delta: {body}"
    );
    assert!(
        !body.contains("WebSearch"),
        "the managed WebSearch block is suppressed: {body}"
    );
    assert!(
        !body.contains("\"type\":\"tool_use\""),
        "no tool_use content block reaches the client: {body}"
    );
    assert!(!body.contains(TOOL_USE_ID), "the managed tool id never leaks: {body}");
    assert!(
        body.contains("\"text\":\"Potato is a starchy tuber.\""),
        "final answer streamed: {body}"
    );
    assert!(
        body.contains("\"output_tokens\":26"),
        "terminal aggregates 8 + 18 output tokens: {body}"
    );

    // The loop re-entered the model with the appended tool result.
    assert_eq!(model.request_count(), 2, "the managed search re-enters the model");
    assert_eq!(search.request_count(), 1, "one managed search callout");
    assert_eq!(search.query(), "potato", "the reconstructed query drove the search");
    let second = model.request_json(1);
    let messages = second["messages"].as_array().expect("re-entry Messages history");
    let tool_result = &messages[messages.len() - 1]["content"][0];
    assert_eq!(tool_result["type"], "tool_result", "the re-entry appends a tool result");
    assert_eq!(
        tool_result["tool_use_id"], TOOL_USE_ID,
        "the tool result matches the managed call"
    );
    assert!(
        tool_result["content"]
            .as_str()
            .is_some_and(|content| content.contains("Potato - Wikipedia")),
        "the search result reaches the model: {tool_result}"
    );
}

#[test]
fn terminal_sse_reaches_client_before_upstream_completes() {
    // The first flush delivers a complete, forwardable text delta; the terminal
    // frames are gated until the client has observed it.
    let mut opening = message_start("msg_1");
    opening.push_str(&sse(
        "content_block_start",
        &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ));
    opening.push_str(&sse(
        "content_block_delta",
        &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Potato"}}),
    ));
    let mut tail = sse("content_block_stop", &json!({"type": "content_block_stop", "index": 0}));
    tail.push_str(&message_delta("end_turn", 9));
    tail.push_str(&message_stop());

    let (backend_port, first_sent, release, backend_thread) = start_gated_backend(vec![opening], vec![tail]);
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, backend_port, free_port()));
    let (observed_tx, observed_rx) = mpsc::channel();
    let (complete_tx, complete_rx) = mpsc::channel();
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let raw = read_response_incrementally(&proxy_addr, &streaming_request(), "\"text\":\"Potato\"", &observed_tx);
        complete_tx.send(raw).expect("test receiver should remain available");
    });

    first_sent
        .recv_timeout(Duration::from_secs(2))
        .expect("backend should send the opening frames");
    observed_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("client should observe the text delta while upstream is still gated");
    release
        .send(())
        .expect("backend release receiver should remain available");

    let raw = complete_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("client should receive the completed stream");
    assert_eq!(parse_status(&raw), 200, "terminal stream should return 200: {raw}");
    let body = parse_body(&raw);
    assert!(
        body.contains("event: message_stop"),
        "terminal message_stop should arrive: {body}"
    );
    client.join().expect("client thread should not panic");
    backend_thread.join().expect("backend thread should not panic");
}

#[test]
fn fragmented_message_start_reassembles() {
    // Split the message_start event across two transport chunks; the client must
    // still observe one intact message_start.
    let full = answer_round("msg_1", "Potato.", 5);
    let split = full.len() / 2;
    let (head, rest) = full.split_at(split);
    let model = StreamingModel::start_fragmented(vec![vec![head.to_owned(), rest.to_owned()]]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(parse_status(&raw), 200, "a fragmented stream returns 200: {raw}");
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "fragments reassemble to one message_start: {body}"
    );
    assert!(
        body.contains("\"text\":\"Potato.\""),
        "the fragmented answer is delivered intact: {body}"
    );
    assert!(
        body.contains("event: message_stop"),
        "the terminal frames still arrive: {body}"
    );
}

#[test]
fn provider_failure_streams_is_error_then_final_answer() {
    let model = StreamingModel::start(vec![
        search_round("msg_1", TOOL_USE_ID, "potato", 8),
        answer_round("msg_2", "I could not search, but potatoes are tubers.", 20),
    ]);
    let search = SearchStub::start_failing();
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "a provider failure must not reject the stream: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("\"text\":\"I could not search, but potatoes are tubers.\""),
        "final answer streamed: {body}"
    );
    assert_eq!(
        model.request_count(),
        2,
        "the loop re-enters the model after a failed search"
    );
    assert_eq!(
        search.request_count(),
        1,
        "the failed search still counts as one callout"
    );
    let second = model.request_json(1);
    let messages = second["messages"].as_array().expect("re-entry Messages history");
    let tool_result = &messages[messages.len() - 1]["content"][0];
    assert_eq!(
        tool_result["is_error"], true,
        "a provider failure produces a truthful is_error result"
    );
    assert_eq!(
        tool_result["content"], "Web search unavailable.",
        "the model receives the bounded failure notice"
    );
}

#[test]
fn malformed_upstream_frame_fails_closed_with_error_event() {
    // A content_block_start without an index is malformed: the stream must fail
    // closed with exactly one terminal error event and leak no raw bytes.
    let mut round = message_start("msg_1");
    round.push_str(&sse(
        "content_block_start",
        &json!({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
    ));
    round.push_str(&text_block(0, "should never appear"));
    round.push_str(&message_delta("end_turn", 5));
    round.push_str(&message_stop());
    let model = StreamingModel::start(vec![round]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the malformed frame: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("event: error"),
        "a terminal error event is emitted: {body}"
    );
    assert!(
        body.contains("\"type\":\"api_error\""),
        "malformed framing maps to api_error: {body}"
    );
    assert!(
        !body.contains("should never appear"),
        "no raw upstream bytes leak after failure: {body}"
    );
}

#[test]
fn premature_clean_eof_fails_closed_with_error_event() {
    // A clean transport EOF (the chunked body ends normally) can still leave the
    // Messages lifecycle logically incomplete: here the round streams a stop
    // reason via `message_delta` but the upstream closes before `message_stop`.
    // The filter must not fabricate a successful terminal from a truncated round;
    // it must fail closed with one coherent Anthropic `error` event.
    let mut round = message_start("msg_1");
    round.push_str(&text_block(0, "Potato is a tuber."));
    round.push_str(&message_delta("end_turn", 9));
    // Intentionally omit `message_stop`: the round is truncated on a clean EOF.
    let model = StreamingModel::start(vec![round]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the round truncates: {raw}"
    );
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "the forwarded message_start reaches the client before truncation: {body}"
    );
    assert!(
        body.contains("event: error"),
        "a truncated round emits a coherent terminal error event: {body}"
    );
    assert!(
        body.contains("\"type\":\"api_error\""),
        "an incomplete stream maps to api_error: {body}"
    );
    assert!(
        body.contains("web search response stream ended before completion"),
        "the error reports the truncated-stream reason: {body}"
    );
    assert!(
        !body.contains("event: message_stop"),
        "a truncated round must not fabricate a terminal message_stop: {body}"
    );
    assert!(
        !body.contains("event: message_delta"),
        "the gated terminal message_delta is never emitted for a truncated round: {body}"
    );
    assert_eq!(
        model.request_count(),
        1,
        "a truncated terminal round ends the loop without re-entry"
    );
    assert_eq!(
        search.request_count(),
        0,
        "a truncated round with no managed search dispatches none"
    );
}

#[test]
fn iteration_ceiling_streams_terminal_error_event() {
    // With max_iterations=2 the loop may search once (round 0) and re-enter, but
    // the second managed round (round 1) is at the router's iteration ceiling:
    // re-entering would meet the limit and the IRR would abort the stream with an
    // abrupt EOF. The filter must detect that and terminate the round with one
    // coherent Anthropic `error` event instead of a truncated stream.
    let model = StreamingModel::start(vec![
        search_round("msg_1", TOOL_USE_ID, "potato", 8),
        search_round("msg_2", TOOL_USE_ID, "tuber", 8),
    ]);
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_max_iterations(
        proxy_port,
        model.port(),
        search.port(),
        2,
    ));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the ceiling is reached: {raw}"
    );
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "the first round's message_start is forwarded exactly once: {body}"
    );
    assert!(
        body.contains("event: error"),
        "the ceiling emits a coherent terminal error event, not an abrupt EOF: {body}"
    );
    assert!(
        body.contains("\"type\":\"api_error\""),
        "the iteration ceiling maps to api_error: {body}"
    );
    assert!(
        body.contains("web search exceeded the maximum number of search iterations"),
        "the client learns the search loop hit its ceiling: {body}"
    );
    assert!(
        !body.contains("WebSearch") && !body.contains("\"type\":\"tool_use\""),
        "no managed search block leaks even at the ceiling: {body}"
    );
    assert!(!body.contains(TOOL_USE_ID), "the managed tool id never leaks: {body}");
    assert_eq!(
        model.request_count(),
        2,
        "the loop searches once then re-enters into the ceiling round"
    );
    assert_eq!(
        search.request_count(),
        1,
        "only the first round dispatches a search; the ceiling round does not"
    );
}

#[test]
fn later_round_non_success_fails_closed_with_error_event() {
    // Round 0 streams a managed WebSearch call, so the client is mid-stream on a
    // committed 200 SSE lifecycle. The re-entry round then returns a non-2xx
    // status; forwarding its raw error body into the open stream would corrupt
    // it, so the loop must fail closed to one coherent terminal error event.
    //
    // In the live streaming pump the SSE parser is what fails closed: the
    // status/encoding guard reads a response header the IRR streaming body phase
    // does not set, so the raw non-SSE error body reaches the transform and is
    // rejected there (it never parses as forwardable events). This exercises the
    // end-to-end property regardless of which layer catches it.
    let model = StreamingModel::start_then_error(
        vec![search_round("msg_1", TOOL_USE_ID, "potato", 8)],
        "429 Too Many Requests",
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "slow down"}}).to_string(),
    );
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed on the first round before the later round fails: {raw}"
    );
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "the first round's message_start is forwarded exactly once: {body}"
    );
    assert!(
        body.contains("event: error"),
        "a later-round non-success fails closed to a terminal error event: {body}"
    );
    assert!(
        body.contains("\"type\":\"api_error\""),
        "an untransformable later round maps to api_error: {body}"
    );
    assert!(
        !body.contains("overloaded_error") && !body.contains("slow down"),
        "no raw upstream error body leaks into the already-open stream: {body}"
    );
    assert!(
        !body.contains("WebSearch") && !body.contains("\"type\":\"tool_use\""),
        "the managed search block stays suppressed: {body}"
    );
    assert_eq!(
        model.request_count(),
        2,
        "the loop searches once then re-enters into the failing round"
    );
    assert_eq!(
        search.request_count(),
        1,
        "only the first round dispatches a search before the failure"
    );
}

#[test]
fn mid_stream_termination_fails_closed_with_error_event() {
    // Round 0 forwards message_start, then the backend stalls without ever
    // completing the round. A short IRR timeout expires the round mid-stream,
    // which Praxis surfaces as an abnormal stream termination. The filter must
    // convert that into one coherent terminal error event AND mark the
    // termination handled, so the router forwards the completion bytes; otherwise
    // the completion output is discarded and the client stream ends in an abrupt
    // EOF with no terminal event.
    let model = StreamingModel::start_then_hang(vec![], message_start("msg_1"));
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_timeout(proxy_port, model.port(), search.port(), 1500));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the round stalls: {raw}"
    );
    let body = parse_body(&raw);
    assert_eq!(
        body.matches("event: message_start").count(),
        1,
        "the forwarded message_start reaches the client before the stall: {body}"
    );
    assert!(
        body.contains("event: error"),
        "an abnormal termination emits a coherent terminal error event, not an abrupt EOF: {body}"
    );
    assert!(
        body.contains("\"type\":\"api_error\""),
        "a stream termination maps to api_error: {body}"
    );
    assert!(
        body.contains("web search exceeded the configured deadline"),
        "a deadline termination reports the deadline reason to the client: {body}"
    );
    assert_eq!(
        search.request_count(),
        0,
        "a round that stalls before completing dispatches no search"
    );
}

#[test]
fn round_zero_non_success_passes_through_untransformed() {
    // The very first round returns a non-2xx JSON error before any SSE bytes
    // reach the client. No stream has been committed, so the raw upstream error
    // must pass through unchanged rather than being transformed into an SSE error
    // event under the original JSON status and headers.
    //
    // `start_then_error` with no preceding rounds serves the error on the first
    // accepted connection, i.e. round 0.
    let model = StreamingModel::start_then_error(
        vec![],
        "429 Too Many Requests",
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "slow down"}}).to_string(),
    );
    let search = SearchStub::start(&search_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = read_response_to_end(proxy.addr(), &streaming_request());

    assert_eq!(
        parse_status(&raw),
        429,
        "the round-0 upstream status passes through unchanged: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        body.contains("overloaded_error") && body.contains("slow down"),
        "the raw upstream error body passes through untransformed: {body}"
    );
    assert!(
        !body.contains("event: error") && !body.contains("event: message_start"),
        "a round-0 non-success is not transformed into an SSE lifecycle: {body}"
    );
    assert_eq!(
        model.request_count(),
        1,
        "a round-0 failure ends the loop without re-entry"
    );
    assert_eq!(
        search.request_count(),
        0,
        "no search is dispatched when the first round fails"
    );
}

#[test]
fn downstream_cancellation_closes_upstream_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let backend_port = listener.local_addr().expect("backend should have an address").port();
    let (first_sent_tx, first_sent_rx) = mpsc::channel();
    let (client_dropped_tx, client_dropped_rx) = mpsc::channel();
    let (cancelled_tx, cancelled_rx) = mpsc::channel();
    let backend_thread = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_full_request(&mut stream);
        write_stream_headers(&mut stream, "text/event-stream");
        let mut opening = message_start("msg_1");
        opening.push_str(&sse(
            "content_block_start",
            &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ));
        opening.push_str(&sse(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Potato"}}),
        ));
        write_chunk(&mut stream, &opening);
        stream.flush().expect("opening frames should flush");
        first_sent_tx.send(()).expect("test receiver should remain available");

        client_dropped_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("client should drop the downstream stream");
        thread::sleep(Duration::from_millis(50));
        stream
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("backend write timeout should be set");
        // Flood with complete, forwardable text deltas: the proxy relays each to
        // the dropped client, detects the closed downstream, and tears down this
        // upstream exchange. (Raw non-SSE bytes would only be buffered, never
        // written downstream, so the closure would go unnoticed.)
        let filler = "x".repeat(16 * 1024);
        let delta = sse(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": filler}}),
        );
        let mut upstream_closed = false;
        for _ in 0..256 {
            if write_chunk_checked(&mut stream, &delta)
                .and_then(|()| stream.flush())
                .is_err()
            {
                upstream_closed = true;
                break;
            }
        }
        cancelled_tx
            .send(upstream_closed)
            .expect("test receiver should remain available");
    });
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, backend_port, free_port()));
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let mut stream = connect_and_send(&proxy_addr, &streaming_request());
        let mut received = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !String::from_utf8_lossy(&received).contains("\"text\":\"Potato\"") {
            let count = stream
                .read(&mut buffer)
                .expect("streaming response read should succeed");
            assert!(count > 0, "stream should not end before the first event");
            received.extend_from_slice(&buffer[..count]);
        }
        drop(stream);
        client_dropped_tx
            .send(())
            .expect("backend cancellation receiver should remain available");
    });

    first_sent_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("backend should send the first event");
    client.join().expect("client thread should not panic");
    assert!(
        cancelled_rx
            .recv_timeout(Duration::from_secs(8))
            .expect("backend should observe cancellation"),
        "dropping the downstream response must close the upstream streaming exchange"
    );
    backend_thread.join().expect("backend thread should not panic");
}

// -----------------------------------------------------------------------------
// Transport helpers
// -----------------------------------------------------------------------------

fn start_gated_backend(
    first_chunks: Vec<String>,
    final_chunks: Vec<String>,
) -> (u16, mpsc::Receiver<()>, mpsc::Sender<()>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let port = listener.local_addr().expect("backend should have an address").port();
    let (first_sent_tx, first_sent_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_full_request(&mut stream);
        write_stream_headers(&mut stream, "text/event-stream");
        for chunk in first_chunks {
            write_chunk(&mut stream, &chunk);
        }
        stream.flush().expect("initial chunks should flush");
        first_sent_tx.send(()).expect("test receiver should remain available");
        release_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("test should release the backend");
        for chunk in final_chunks {
            write_chunk(&mut stream, &chunk);
        }
        stream.write_all(b"0\r\n\r\n").expect("chunked response should finish");
        stream.flush().expect("terminal chunks should flush");
    });
    (port, first_sent_rx, release_tx, handle)
}

/// Read a full HTTP request (headers + `Content-Length` body) from a stream.
fn read_full_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("backend read timeout should be set");
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = stream.read(&mut buffer).expect("request read should succeed");
        assert!(count > 0, "request must complete before the connection closes");
        request.extend_from_slice(&buffer[..count]);
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
            })
            .unwrap_or(0);
        if request.len() >= header_end + 4 + content_length {
            return String::from_utf8_lossy(&request).into_owned();
        }
    }
}

fn write_stream_headers(stream: &mut TcpStream, content_type: &str) {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )
    .expect("response headers should be written");
}

fn write_chunk(stream: &mut TcpStream, chunk: &str) {
    write!(stream, "{:x}\r\n{chunk}\r\n", chunk.len()).expect("response chunk should be written");
}

/// Write one chunked-transfer frame, surfacing any transport error to the caller.
fn write_chunk_checked(stream: &mut TcpStream, chunk: &str) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n{chunk}\r\n", chunk.len())
}

fn connect_and_send(proxy_addr: &str, body: &str) -> TcpStream {
    let mut stream = TcpStream::connect(proxy_addr).expect("client should connect to proxy");
    stream
        .set_read_timeout(Some(Duration::from_secs(4)))
        .expect("client read timeout should be set");
    stream
        .write_all(json_post("/v1/messages", body).as_bytes())
        .expect("client request should be written");
    stream
}

fn read_response_incrementally(proxy_addr: &str, body: &str, needle: &str, observed: &mpsc::Sender<()>) -> String {
    let mut stream = connect_and_send(proxy_addr, body);
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 1024];
    let mut notified = false;
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                raw.extend_from_slice(&buffer[..count]);
                if !notified && String::from_utf8_lossy(&raw).contains(needle) {
                    observed.send(()).expect("test receiver should remain available");
                    notified = true;
                }
            },
            Err(error) => panic!("streaming response read failed: {error}"),
        }
    }
    String::from_utf8_lossy(&raw).into_owned()
}

fn read_response_to_end(proxy_addr: &str, body: &str) -> String {
    let mut stream = connect_and_send(proxy_addr, body);
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("proxy response should end");
    String::from_utf8_lossy(&raw).into_owned()
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
