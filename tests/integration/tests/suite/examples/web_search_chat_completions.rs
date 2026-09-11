// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for Responses web search through a Chat Completions backend.
//!
//! The single `web-search-chat-completions.yaml` example serves both finite
//! (`"stream": false`) and streaming (`"stream": true`) requests. These tests
//! exercise the finite round-trip, the streaming round-trip (translated Chat
//! Completions SSE, restored `web_search_call`, dispatched search, resumed
//! inference, one coherent Responses SSE lifecycle), and the fail-closed guard
//! when the logical-stream finalizer is absent from a streaming pipeline.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use praxis_test_utils::{
    StatefulCapturingBackend, build_pipeline, example_config_path, free_port, http_send, json_post, parse_body,
    parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "openai/responses/web-search-chat-completions.yaml";

/// Read the example config and point its model backend and search provider at
/// the given local mock ports.
fn patched_yaml(listener_port: u16, model_port: u16, search_port: u16) -> String {
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let yaml = patch_yaml(&yaml, listener_port, &HashMap::from([("127.0.0.1:3001", model_port)]));
    yaml.replace(
        "api_key: ${WEB_SEARCH_API_KEY}",
        &format!(
            "api_key: test-key\n                base_url: http://127.0.0.1:{search_port}\n                allow_private_base_url: true"
        ),
    )
}

fn load_test_config(listener_port: u16, model_port: u16, search_port: u16) -> praxis_core::config::Config {
    praxis_core::config::Config::from_yaml(&patched_yaml(listener_port, model_port, search_port))
        .expect("patched config should parse")
}

/// Load the example with the logical-stream finalizer removed, modelling a
/// streaming-capable pipeline that is missing `openai_stream_events`. Without the
/// finalizer inside the iterative router nothing publishes the
/// `responses.logical_stream` marker, so `openai_agentic_loop` must fail closed
/// before any typed-streaming dispatch.
fn load_test_config_without_stream_events(
    listener_port: u16,
    model_port: u16,
    search_port: u16,
) -> praxis_core::config::Config {
    // Drop the whole `openai_stream_events` filter entry (and the blank line that
    // follows it) from the inference step. The header doc comment mentions the
    // filter by name but never as a `- filter:` list entry, so the match is
    // unambiguous.
    let stream_events = "              - filter: openai_stream_events\n\n";
    let yaml = patched_yaml(listener_port, model_port, search_port);
    assert!(
        yaml.contains(stream_events),
        "expected the openai_stream_events filter entry in the example; its block may have changed"
    );
    let yaml = yaml.replace(stream_events, "");
    praxis_core::config::Config::from_yaml(&yaml).expect("patched config should parse")
}

#[test]
fn web_search_chat_completions_example_builds() {
    let config = load_test_config(free_port(), 19_301, 19_302);
    let _pipeline = build_pipeline(&config);
}

#[test]
fn web_search_chat_completions_completes_model_search_model_round_trip() {
    let first_response = json!({
        "id": "chatcmpl_search",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"Praxis Proxy latest release\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    });
    let second_response = json!({
        "id": "chatcmpl_answer",
        "object": "chat.completion",
        "model": "chat-only-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Praxis Proxy has a current release."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
    });
    let model = StatefulCapturingBackend::new(vec![
        (200, first_response.to_string()),
        (200, second_response.to_string()),
    ])
    .start_with_shutdown();
    let search = StatefulCapturingBackend::new(vec![(
        200,
        json!({
            "web": {"results": [{
                "title": "Praxis Proxy releases",
                "url": "https://github.com/praxis-proxy/praxis/releases",
                "description": "Current Praxis Proxy releases."
            }]}
        })
        .to_string(),
    )])
    .start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), search.port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "tools": [{
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "FR"}
        }],
        "tool_choice": {"type": "web_search"},
        "include": ["web_search_call.action.sources"],
        "store": false
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(parse_status(&raw), 200, "round trip should succeed: {raw}");
    let response: Value = serde_json::from_str(&parse_body(&raw)).expect("client response should be JSON");
    assert_eq!(response["output"].as_array().map(Vec::len), Some(2));
    assert_eq!(response["output"][0]["type"], "web_search_call");
    assert_eq!(response["output"][0]["id"], "call_search_1");
    assert_eq!(response["output"][0]["status"], "completed");
    assert_eq!(response["output"][0]["action"]["query"], "Praxis Proxy latest release");
    assert_eq!(
        response["output"][0]["action"]["sources"][0]["url"],
        "https://github.com/praxis-proxy/praxis/releases"
    );
    assert_eq!(
        response["output"][1]["content"][0]["text"],
        "Praxis Proxy has a current release."
    );
    assert_eq!(response["tools"], request["tools"]);
    assert_eq!(response["tool_choice"], request["tool_choice"]);
    assert_eq!(response["usage"]["input_tokens"], 36);
    assert_eq!(response["usage"]["output_tokens"], 13);
    assert_eq!(response["usage"]["total_tokens"], 49);

    let model_requests = model.requests();
    assert_eq!(model_requests.len(), 2, "web search should drive two model calls");
    assert!(
        model_requests
            .iter()
            .all(|captured| captured.uri == "/v1/chat/completions")
    );
    let first_forwarded: Value = serde_json::from_str(&model_requests[0].body).expect("first request should be JSON");
    assert_eq!(first_forwarded["tools"][0]["type"], "function");
    assert_eq!(first_forwarded["tools"][0]["function"]["name"], "web_search");
    assert_eq!(
        first_forwarded["tools"][0]["function"]["parameters"]["required"],
        json!(["query"])
    );
    assert_eq!(
        first_forwarded["tool_choice"],
        json!({"type": "function", "function": {"name": "web_search"}})
    );
    assert!(first_forwarded["tools"][0].get("search_context_size").is_none());
    assert!(first_forwarded["tools"][0].get("user_location").is_none());

    let second_forwarded: Value = serde_json::from_str(&model_requests[1].body).expect("second request should be JSON");
    assert_eq!(second_forwarded["tool_choice"], "auto");
    let messages = second_forwarded["messages"]
        .as_array()
        .expect("messages should be an array");
    assert!(messages.iter().any(|message| {
        message["tool_calls"][0]["function"]["name"] == "web_search"
            && message["tool_calls"][0]["function"]["arguments"] == "{\"query\":\"Praxis Proxy latest release\"}"
    }));
    assert!(messages.iter().any(|message| {
        message["role"] == "tool"
            && message["content"]
                .as_str()
                .is_some_and(|content| content.contains("Praxis Proxy releases"))
    }));

    let search_requests = search.requests();
    assert_eq!(search_requests.len(), 1);
    assert!(search_requests[0].uri.contains("q=Praxis%20Proxy%20latest%20release"));
    assert!(search_requests[0].uri.contains("count=10"));
}

#[test]
fn web_search_chat_completions_streams_dispatch_once_as_one_logical_response() {
    // Round 1: the Chat backend streams a private `web_search` tool call.
    let first_response = vec![
        chat_chunk(
            r#"{"id":"chatcmpl_ws","object":"chat.completion.chunk","model":"chat-only-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_ws_1","type":"function","function":{"name":"web_search","arguments":""}}]}}]}"#,
        ),
        chat_chunk(
            r#"{"id":"chatcmpl_ws","object":"chat.completion.chunk","model":"chat-only-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":\"Praxis Proxy latest release\"}"}}]}}]}"#,
        ),
        chat_chunk(
            r#"{"id":"chatcmpl_ws","object":"chat.completion.chunk","model":"chat-only-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ),
        "data: [DONE]\n\n".to_owned(),
    ];
    // Round 2: after the search result is bridged in, the model streams the
    // final assistant answer.
    let second_response = vec![
        chat_chunk(
            r#"{"id":"chatcmpl_ans","object":"chat.completion.chunk","model":"chat-only-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Praxis Proxy shipped a new release."}}]}"#,
        ),
        chat_chunk(
            r#"{"id":"chatcmpl_ans","object":"chat.completion.chunk","model":"chat-only-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ),
        "data: [DONE]\n\n".to_owned(),
    ];
    let (model_port, model_requests, model_thread) = start_streaming_model(vec![first_response, second_response]);

    let search_listener = TcpListener::bind("127.0.0.1:0").expect("search mock should bind");
    let search_port = search_listener.local_addr().expect("search mock address").port();
    let search_hits = spawn_counting_search_mock(search_listener);

    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model_port, search_port);
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search"}]
    });

    let raw = http_send(
        proxy.addr(),
        &json_post("/v1/responses", &serde_json::to_string(&request).unwrap()),
    );
    let body = parse_body(&raw);

    assert_eq!(
        parse_status(&raw),
        200,
        "streaming web-search round-trip should return 200 (model requests: {}, search hits: {}): {raw}",
        model_requests
            .lock()
            .expect("model request lock should not be poisoned")
            .len(),
        search_hits.load(Ordering::SeqCst),
    );

    // AC1: the configured web-search provider is dispatched exactly once.
    assert_eq!(
        search_hits.load(Ordering::SeqCst),
        1,
        "the search provider must be dispatched exactly once: {body}"
    );

    // AC3: the client observes ONE coherent Responses SSE lifecycle across the
    // model round, the search, and the resumed model output.
    assert_eq!(
        body.matches("event: response.created").count(),
        1,
        "one logical stream must expose exactly one response.created event: {body}"
    );
    assert_eq!(
        body.matches("event: response.completed").count(),
        1,
        "the intermediate per-round completion must be suppressed: {body}"
    );
    // None of the upstream Chat Completions framing leaks to the client.
    assert!(
        !body.contains("chat.completion.chunk"),
        "raw Chat framing leaked to a Responses client: {body}"
    );
    assert!(!body.contains("[DONE]"), "raw Chat sentinel leaked: {body}");
    // The resumed inference text reaches the same stream.
    assert!(
        body.contains("Praxis Proxy shipped a new release."),
        "resumed inference text should reach the client stream: {body}"
    );

    // AC4: the terminal response.completed carries the completed web_search_call
    // and the final assistant message.
    let completed = body
        .split("\n\n")
        .find(|frame| frame.starts_with("event: response.completed\n"))
        .expect("stream should contain a terminal response.completed event");
    let data = completed
        .lines()
        .nth(1)
        .and_then(|line| line.strip_prefix("data: "))
        .expect("terminal event should carry a data line");
    let terminal: Value = serde_json::from_str(data).expect("terminal event data should be JSON");
    assert_eq!(terminal["response"]["status"], "completed");
    let output = terminal["response"]["output"]
        .as_array()
        .expect("terminal response should carry an output array");
    let search_calls: Vec<&Value> = output.iter().filter(|item| item["type"] == "web_search_call").collect();
    assert_eq!(
        search_calls.len(),
        1,
        "terminal output must contain exactly one web_search_call: {output:#?}"
    );
    assert_eq!(search_calls[0]["status"], "completed");
    assert!(
        output.iter().any(|item| {
            item["type"] == "message" && item["content"][0]["text"] == "Praxis Proxy shipped a new release."
        }),
        "terminal output must contain the final assistant message: {output:#?}"
    );

    // AC2: the search result is bridged into the resumed inference request.
    model_thread.join().expect("streaming model thread should finish");
    // The model thread has joined, so take the captured bodies out of the shared
    // log; the lock guard is released with the statement instead of being held
    // across the request assertions below.
    let requests = std::mem::take(
        &mut *model_requests
            .lock()
            .expect("model request lock should not be poisoned"),
    );
    assert_eq!(
        requests.len(),
        2,
        "web search should drive exactly two streamed model requests"
    );
    let first: Value = serde_json::from_str(&requests[0]).expect("first request should be JSON");
    assert_eq!(
        first["stream"], true,
        "the translated request must ask the backend to stream"
    );
    assert_eq!(
        first["tools"][0]["function"]["name"], "web_search",
        "web_search must be exposed to the Chat backend as a private function tool"
    );

    let second: Value = serde_json::from_str(&requests[1]).expect("second request should be JSON");
    let messages = second["messages"].as_array().expect("second request messages array");
    // #808: a hosted web_search_call is not a valid Chat input, so the
    // continuation bridges it as a function_call / tool result pair instead.
    assert!(
        messages.iter().any(|message| {
            message["tool_calls"][0]["function"]["name"] == "web_search"
                && message["tool_calls"][0]["function"]["arguments"] == "{\"query\":\"Praxis Proxy latest release\"}"
        }),
        "second inference should carry the bridged web_search function_call: {messages:#?}"
    );
    assert!(
        messages.iter().any(|message| {
            message["role"] == "tool"
                && message["content"]
                    .as_str()
                    .is_some_and(|content| content.contains("Praxis Proxy releases"))
        }),
        "second inference should carry the search result as a tool message: {messages:#?}"
    );
}

#[test]
fn web_search_chat_completions_streaming_fails_closed_without_stream_events() {
    // openai_responses_to_chat_completions always advertises the streaming subrequest
    // capability and selects the streaming transport for an effective
    // `"stream": true` request. Without `openai_stream_events` in the inference
    // step nothing publishes the `responses.logical_stream` marker, so typed
    // streaming would commit `response.completed` to the client as it arrives and
    // a loop-terminal error detected later by `openai_agentic_loop` could not
    // reach the client. The loop must therefore fail closed before any backend
    // dispatch rather than forward a truncatable typed stream.
    let model = StatefulCapturingBackend::new(vec![(200, "unexpected".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config_without_stream_events(proxy_port, model.port(), free_port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "Find the latest Praxis Proxy release.",
        "stream": true,
        "store": false,
        "tools": [{"type": "web_search"}]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(
        parse_status(&raw),
        500,
        "a streaming request without a logical-stream finalizer must fail closed before dispatch: {raw}"
    );
    assert!(
        parse_body(&raw).contains("server_error"),
        "the rejection must carry the server_error code: {}",
        parse_body(&raw)
    );
    assert!(
        model.requests().is_empty(),
        "an unsafe streaming request must not reach the model backend"
    );
}

#[test]
fn web_search_chat_completions_rejects_function_name_collision_before_forwarding() {
    let model = StatefulCapturingBackend::new(vec![(200, "unexpected".to_owned())]).start_with_shutdown();
    let proxy_port = free_port();
    let config = load_test_config(proxy_port, model.port(), free_port());
    let proxy = start_proxy(&config);
    let request = json!({
        "model": "chat-only-model",
        "input": "search",
        "tools": [
            {"type": "web_search"},
            {"type": "function", "name": "web_search", "parameters": {"type": "object"}}
        ]
    });

    let raw = http_send(proxy.addr(), &json_post("/v1/responses", &request.to_string()));

    assert_eq!(
        parse_status(&raw),
        400,
        "collision should fail before forwarding: {raw}"
    );
    assert!(parse_body(&raw).contains("conflicts with the synthesized web_search function"));
    assert!(
        model.requests().is_empty(),
        "collision must not reach the model backend"
    );
}

/// Encode one Chat Completions SSE frame (no `event:` line, `data:` payload).
fn chat_chunk(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// Handle returned by the synthetic streaming model backend.
type StreamingModel = (u16, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>);

/// Start a two-turn Chat Completions backend that emits each SSE frame as a
/// separate chunk and records the request body it received each round.
fn start_streaming_model(responses: Vec<Vec<String>>) -> StreamingModel {
    let listener = TcpListener::bind("127.0.0.1:0").expect("streaming model should bind");
    let port = listener
        .local_addr()
        .expect("streaming model should have an address")
        .port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&requests);
    let handle = thread::spawn(move || {
        for response in responses {
            let (mut stream, _) = listener.accept().expect("streaming model should accept request");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("streaming model should set read timeout");
            let request = read_json_request(&mut stream);
            captured
                .lock()
                .expect("model request lock should not be poisoned")
                .push(request);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .expect("streaming model should write response headers");
            for frame in response {
                write!(stream, "{:x}\r\n{frame}\r\n", frame.len()).expect("streaming model should write frame chunk");
                stream.flush().expect("streaming model should flush frame chunk");
            }
            stream
                .write_all(b"0\r\n\r\n")
                .expect("streaming model should finish chunked response");
        }
    });
    (port, requests, handle)
}

/// Read one content-length JSON request and return its body.
fn read_json_request(stream: &mut TcpStream) -> String {
    let mut raw = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = stream.read(&mut buffer).expect("streaming model should read request");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
        let text = String::from_utf8_lossy(&raw);
        let Some((headers, body)) = text.split_once("\r\n\r\n") else {
            continue;
        };
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if body.len() >= content_length {
            return body.get(..content_length).unwrap_or_default().to_owned();
        }
    }
    String::new()
}

/// Serve Brave-style search results and count how many times the provider is hit.
fn spawn_counting_search_mock(listener: TcpListener) -> Arc<AtomicUsize> {
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&hits);
    let body = json!({
        "web": {
            "results": [{
                "title": "Praxis Proxy releases",
                "url": "https://github.com/praxis-proxy/praxis/releases",
                "description": "Current Praxis Proxy releases."
            }]
        }
    })
    .to_string();
    thread::spawn(move || {
        loop {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let mut buf = [0_u8; 4096];
            let _read = stream.read(&mut buf).unwrap_or(0);
            counter.fetch_add(1, Ordering::SeqCst);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _sent = stream.write_all(response.as_bytes());
        }
    });
    hits
}
