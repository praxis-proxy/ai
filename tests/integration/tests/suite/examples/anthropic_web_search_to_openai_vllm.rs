// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional coverage for the Anthropic Messages web-search -> Chat Completions
//! vLLM example config (`anthropic/web-search-to-openai-vllm.yaml`).
//!
//! This config combines two transformations in one pipeline:
//!
//!   * the server-owned Anthropic `WebSearch` loop from `full-flow-agentic.yaml`, and
//!   * the Anthropic <-> Chat Completions translation from `messages-to-openai-vllm.yaml`.
//!
//! A client speaks native Anthropic Messages while the model runs on a
//! Chat-Completions-only vLLM backend. Each round the Anthropic request is
//! translated to Chat Completions; when the model returns a `WebSearch` function
//! call, the translation layer rebuilds it as an Anthropic `tool_use` block,
//! `anthropic_web_search` runs the search (Tavily), appends the result, and
//! re-enters the model, which then answers in text.
//!
//! The nightly GPU suite exercises this same config against a live vLLM model
//! (with a stubbed and a live Tavily provider) via
//! `tests/integration/sdk/anthropic/test_anthropic_web_search_vllm.py`. This CPU
//! test pins the deterministic wiring with scripted stubs so the combined
//! translation + web-search loop keeps working without a GPU.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use praxis_test_utils::{
    StatefulCapturingBackend, example_config_path, free_port, http_send, json_post, parse_body, parse_status,
    patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "anthropic/web-search-to-openai-vllm.yaml";
const TOOL_CALL_ID: &str = "call_web_search_01";

// -----------------------------------------------------------------------------
// Backend fixtures (OpenAI Chat Completions, the shape the vLLM backend serves)
// -----------------------------------------------------------------------------

/// A Chat Completions round that forces a single `WebSearch` function call. The
/// translation layer rebuilds this as an Anthropic `tool_use` block for
/// `anthropic_web_search` to classify and dispatch.
fn chat_tool_call_round(call_id: &str, query: &str) -> String {
    json!({
        "id": "chatcmpl-round0",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "Qwen/Qwen3-8B",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "WebSearch", "arguments": json!({"query": query}).to_string()}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    })
    .to_string()
}

/// A terminal Chat Completions round that answers in plain text.
fn chat_answer_round(text: &str) -> String {
    json!({
        "id": "chatcmpl-round1",
        "object": "chat.completion",
        "created": 1_700_000_001,
        "model": "Qwen/Qwen3-8B",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 74, "completion_tokens": 18, "total_tokens": 92}
    })
    .to_string()
}

/// A Tavily-shaped search response body (`results[].{title,url,content}`).
fn tavily_results() -> Value {
    json!({
        "results": [{
            "title": "Potato - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Potato",
            "content": "The potato is a starchy tuber native to the Americas."
        }]
    })
}

/// The initial buffered client request: native Anthropic Messages declaring the
/// `WebSearch` tool and forcing it via `tool_choice`, exactly as a live client
/// would to make a small model reliably open the managed loop.
fn client_request() -> String {
    json!({
        "model": "Qwen/Qwen3-8B",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Use web search to look up potato, then summarize."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }],
        "tool_choice": {"type": "tool", "name": "WebSearch"}
    })
    .to_string()
}

// -----------------------------------------------------------------------------
// Config loader
// -----------------------------------------------------------------------------

/// Read the example and rewrite the proxy/backend ports, the Tavily endpoint, and
/// the backend credential env var so the loop calls the local stubs without
/// mutating the environment.
fn load_config(proxy_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    load_config_full(proxy_port, model_port, search_port, None, None)
}

/// Same as [`load_config`], but optionally shrinks the translator's
/// `max_body_bytes` so a test can exercise the translator's raw-response size
/// ceiling (the streaming error path) without fabricating a multi-megabyte body.
fn load_config_with_translator_limit(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    translator_max_body_bytes: Option<usize>,
) -> praxis_core::config::Config {
    load_config_full(proxy_port, model_port, search_port, translator_max_body_bytes, None)
}

/// Same as [`load_config`], but injects `max_body_bytes` on the
/// `anthropic_web_search` loop authority and leaves the translator at its 1 MiB
/// default. This reproduces the response size-limit bypass exactly: the translator
/// shrinks an oversized upstream error before web_search's ceiling would see it, so
/// the loop must reject the round from its OWN limit measured on the raw bytes.
fn load_config_with_web_search_limit(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    web_search_max_body_bytes: usize,
) -> praxis_core::config::Config {
    load_config_full(
        proxy_port,
        model_port,
        search_port,
        None,
        Some(web_search_max_body_bytes),
    )
}

/// Read the example and rewrite the proxy/backend ports, the Tavily endpoint, and
/// the backend credential env var so the loop calls the local stubs without
/// mutating the environment, optionally overriding either filter's byte ceiling.
fn load_config_full(
    proxy_port: u16,
    model_port: u16,
    search_port: u16,
    translator_max_body_bytes: Option<usize>,
    web_search_max_body_bytes: Option<usize>,
) -> praxis_core::config::Config {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("read web-search-to-openai-vllm example");
    let yaml = patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:8000", model_port)]));
    let yaml = match translator_max_body_bytes {
        Some(limit) => yaml.replace("max_body_bytes: 1048576", &format!("max_body_bytes: {limit}")),
        None => yaml,
    };
    // The example omits `max_body_bytes` on `anthropic_web_search` (it defaults to
    // the shared JSON ceiling); anchor on its unique `provider: tavily` line to add
    // one just for the test, at the block's 16-space indent.
    let yaml = match web_search_max_body_bytes {
        Some(limit) => yaml.replace(
            "                provider: tavily",
            &format!("                provider: tavily\n                max_body_bytes: {limit}"),
        ),
        None => yaml,
    };
    // Point the Tavily provider at the local body-authenticated stub with a known
    // key so the test can assert the credential travels in the request body.
    let yaml = yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!("api_key: test-key\n                base_url: http://127.0.0.1:{search_port}"),
    );
    // `credential_injection` and other secret-bearing filters resolve at
    // pipeline-build time; repoint the backend Bearer to a Cargo-provided var so
    // the build succeeds without `unsafe` `set_var`.
    let yaml = yaml.replace("env_var: VLLM_API_KEY", "env_var: CARGO_PKG_NAME");
    // The provider callout targets a loopback mock, so the executor's SSRF check
    // requires the operator opt-in on the outbound pipeline.
    let yaml = yaml.replace(
        "allow_private_endpoints: true",
        "allow_private_endpoints: true\n  allow_private_upstreams: true",
    );
    praxis_core::config::Config::from_yaml(&yaml).expect("parse web-search-to-openai-vllm example")
}

// -----------------------------------------------------------------------------
// Tavily search stub (body-authenticated, captures each request)
// -----------------------------------------------------------------------------

struct TavilyStub {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl TavilyStub {
    fn start(response: &Value) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind Tavily stub");
        let port = listener.local_addr().expect("stub address").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let body = response.to_string();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept search request");
            captured
                .lock()
                .expect("capture search request")
                .push(read_full_request(&mut stream));
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).expect("write search response");
        });
        Self { port, requests }
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn request_count(&self) -> usize {
        self.requests.lock().expect("read search requests").len()
    }

    fn body_json(&self) -> Value {
        let request = self.requests.lock().expect("read search requests")[0].clone();
        let (_, body) = request.split_once("\r\n\r\n").expect("search request body");
        serde_json::from_str(body).expect("search request JSON")
    }
}

/// Read a full HTTP request (headers + `Content-Length` body) from a stream.
fn read_full_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("stub read timeout should be set");
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

// -----------------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------------

#[test]
fn web_search_loop_translates_chat_completions_round_trip() {
    // Round 0 forces a WebSearch call; round 1 answers in text after the search
    // result is appended. The backend serves Chat Completions; the translation
    // filters convert each response back into Anthropic Messages for
    // `anthropic_web_search` to classify.
    let model = StatefulCapturingBackend::new(vec![
        (200, chat_tool_call_round(TOOL_CALL_ID, "potato")),
        (200, chat_answer_round("Potato is a starchy tuber.")),
    ])
    .start_with_shutdown();
    let search = TavilyStub::start(&tavily_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request()));

    // The client receives one translated Anthropic Messages answer.
    assert_eq!(parse_status(&raw), 200, "the buffered loop returns 200: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response JSON");
    assert_eq!(
        response["type"], "message",
        "the Chat Completion must be translated back into an Anthropic message: {response}"
    );
    assert_eq!(
        response["content"][0]["text"], "Potato is a starchy tuber.",
        "the translated answer must carry the model's terminal text: {response}"
    );

    // Exactly one managed Tavily search, body-authenticated with the configured
    // key and the reconstructed query.
    assert_eq!(search.request_count(), 1, "the loop dispatches exactly one search");
    let search_body = search.body_json();
    assert_eq!(
        search_body["query"], "potato",
        "the reconstructed query drove the search"
    );
    assert_eq!(
        search_body["api_key"], "test-key",
        "the Tavily key must travel in the request body"
    );

    // The backend saw two Chat Completions rounds; the re-entry carries the
    // translated tool result so the model can answer from the search output.
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "the managed search re-enters the model exactly once");
    assert_eq!(
        requests[0].uri, "/v1/chat/completions",
        "round 0 hits the Chat endpoint"
    );
    assert_eq!(
        requests[1].uri, "/v1/chat/completions",
        "round 1 hits the Chat endpoint"
    );
    let reentry: Value = serde_json::from_str(&requests[1].body).expect("re-entry request JSON");
    let messages = reentry["messages"].as_array().expect("re-entry Chat messages");
    let tool_message = messages
        .iter()
        .find(|m| m["role"] == "tool")
        .expect("the re-entry must append a Chat `tool` message for the search result");
    assert_eq!(
        tool_message["tool_call_id"], TOOL_CALL_ID,
        "the tool result must reference the managed call id"
    );
    assert!(
        tool_message["content"]
            .as_str()
            .is_some_and(|content| content.contains("Potato - Wikipedia")),
        "the Tavily result must reach the model on re-entry: {tool_message}"
    );
}

#[test]
fn streaming_non2xx_round_yields_single_anthropic_error() {
    // Regression: a streaming request whose first model round returns a non-2xx
    // JSON error must reach the client as exactly one Anthropic error object.
    //
    // The buffered translator ratchets the response body mode to `StreamBuffer`,
    // but this streaming-composed IRR pipeline ignores that per-request ratchet
    // and delivers the error round as raw `Stream` chunks. Before the fix the raw
    // upstream error streamed through and a second, empty-input transform was
    // appended -- two concatenated JSON objects, invalid for the client. The
    // translator now accumulates the error round and transforms it once.
    let model = StatefulCapturingBackend::new(vec![(
        429,
        json!({"error": {"message": "rate limited", "type": "rate_limit_error", "code": "rate_limit_exceeded"}})
            .to_string(),
    )])
    .start_with_shutdown();
    let search = TavilyStub::start(&tavily_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config(proxy_port, model.port(), search.port()));

    let request = json!({
        "model": "Qwen/Qwen3-8B",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Use web search to look up potato."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }],
        "tool_choice": {"type": "tool", "name": "WebSearch"},
        "stream": true
    })
    .to_string();
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request));

    assert_eq!(
        parse_status(&raw),
        429,
        "the upstream error status passes through: {raw}"
    );
    let body = parse_body(&raw);
    // `from_str` rejects trailing data, so a successful parse proves the body is a
    // single JSON object rather than the raw error concatenated with a second one.
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["type"], "error",
        "the client receives an Anthropic error envelope: {body}"
    );
    assert_eq!(
        response["error"]["type"], "rate_limit_error",
        "the upstream 429 maps to an Anthropic rate_limit_error: {body}"
    );
    assert_eq!(
        response["error"]["message"], "rate limited",
        "the real upstream message survives (not an empty-input fallback): {body}"
    );
    assert_eq!(
        search.request_count(),
        0,
        "the round-0 error terminates the loop before any search dispatch"
    );
}

/// A JSON 429 error body padded past a size ceiling so the transformed Anthropic
/// envelope would be far smaller than the raw upstream response.
fn oversized_rate_limit_body(pad_bytes: usize) -> String {
    json!({
        "error": {
            "message": "rate limited",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "detail": "x".repeat(pad_bytes)
        }
    })
    .to_string()
}

#[test]
fn buffered_oversized_error_is_rejected_before_translation_shrinks_it() {
    // Response size-limit bypass regression (the reviewer's exact repro): the
    // `anthropic_web_search` loop is capped at 4 KiB while the translator keeps its
    // 1 MiB default. On the composed pipeline the translator runs BEFORE
    // `anthropic_web_search` on the response path, so it normalizes an 8 KiB upstream
    // error into a small Anthropic envelope; web_search's own ceiling would then
    // measure the shrunk body and let it through. The loop must instead enforce its
    // ceiling on the RAW upstream size (recorded by the translator) and reject with a
    // 502 rather than forward the normalized 429.
    let model = StatefulCapturingBackend::new(vec![(429, oversized_rate_limit_body(8192))]).start_with_shutdown();
    let search = TavilyStub::start(&tavily_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_web_search_limit(
        proxy_port,
        model.port(),
        search.port(),
        4096,
    ));

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request()));

    assert_eq!(
        parse_status(&raw),
        502,
        "the raw oversized error is rejected with the configured 502, not a normalized 429: {raw}"
    );
    let body = parse_body(&raw);
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["type"], "error",
        "the client receives an Anthropic error envelope: {body}"
    );
    assert_eq!(
        response["error"]["type"], "api_error",
        "an oversized upstream body maps to an api_error, not the upstream rate_limit_error: {body}"
    );
    assert_eq!(
        search.request_count(),
        0,
        "the oversized error terminates the loop before any search dispatch"
    );
}

#[test]
fn buffered_expanding_translation_is_rejected_over_web_search_limit() {
    // Complementary bypass: the Chat Completions -> Anthropic translation EXPANDS a
    // successful response (the Anthropic envelope is larger than the Chat one). The
    // loop must reject when the translated body it buffers exceeds max_body_bytes,
    // even though the raw upstream round did not. Pin the web-search limit to the
    // exact raw Chat length so the raw round passes and only the larger translated
    // body can trip the ceiling -- checking raw size alone would wrongly return 200.
    let answer = chat_answer_round("Potato is a starchy tuber native to the Americas.");
    let raw_len = answer.len();
    let model = StatefulCapturingBackend::new(vec![(200, answer)]).start_with_shutdown();
    let search = TavilyStub::start(&tavily_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_web_search_limit(
        proxy_port,
        model.port(),
        search.port(),
        raw_len,
    ));

    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &client_request()));

    assert_eq!(
        parse_status(&raw),
        502,
        "the expanded translated body exceeds the web-search limit and must be rejected (raw_len={raw_len}): {raw}"
    );
    let body = parse_body(&raw);
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["error"]["type"], "api_error",
        "an oversized translated body maps to an api_error: {body}"
    );
    assert_eq!(
        search.request_count(),
        0,
        "the oversized translated round is rejected before any search dispatch"
    );
}

#[test]
fn streaming_oversized_error_fails_closed_with_json() {
    // The streaming error accumulator must fail closed on an oversized round rather
    // than silently truncating it -- and it must do so with a valid JSON error, not
    // an SSE frame.
    //
    // This round is pre-SSE: a streaming request whose first model round returns a
    // JSON non-2xx has its 429 status and `application/json` content-type committed
    // in the header phase. A body-phase `Reject` here would become a stream
    // termination that `anthropic_web_search` renders as an `event: error` SSE frame
    // under those JSON headers -- a body the Anthropic SDK cannot parse. Instead the
    // translator emits exactly one JSON `api_error`, which the loop passes through
    // unchanged, so the client receives a structured error it can decode.
    let model = StatefulCapturingBackend::new(vec![(429, oversized_rate_limit_body(8192))]).start_with_shutdown();
    let search = TavilyStub::start(&tavily_results());
    let proxy_port = free_port();
    let proxy = start_proxy(&load_config_with_translator_limit(
        proxy_port,
        model.port(),
        search.port(),
        Some(4096),
    ));

    let request = json!({
        "model": "Qwen/Qwen3-8B",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Use web search to look up potato."}],
        "tools": [{
            "name": "WebSearch",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }],
        "tool_choice": {"type": "tool", "name": "WebSearch"},
        "stream": true
    })
    .to_string();
    let raw = http_send(proxy.addr(), &json_post("/v1/messages", &request));

    assert_eq!(
        parse_status(&raw),
        429,
        "the committed upstream status passes through the fail-closed body: {raw}"
    );
    let body = parse_body(&raw);
    assert!(
        !body.contains("event: error"),
        "the pre-SSE overflow must fail closed as JSON, never an SSE error frame: {raw}"
    );
    // `from_str` rejects trailing data, so a successful parse proves the body is one
    // JSON object rather than SSE framing or two concatenated errors.
    let response: Value =
        serde_json::from_str(&body).unwrap_or_else(|error| panic!("body must be one JSON object ({error}): {body}"));
    assert_eq!(
        response["type"], "error",
        "the client receives an Anthropic error envelope: {body}"
    );
    assert_eq!(
        response["error"]["type"], "api_error",
        "an oversized round is an api_error, not the normalized upstream rate_limit_error: {body}"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("max_body_bytes")),
        "the JSON error names the exceeded ceiling: {body}"
    );
    assert!(
        !body.contains("rate_limit_error"),
        "the suppressed upstream body must not leak its rate_limit_error type: {raw}"
    );
    assert_eq!(
        search.request_count(),
        0,
        "the oversized error terminates the loop before any search dispatch"
    );
}
