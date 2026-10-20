// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional test for the committed-stream retained-payload overflow example.
//!
//! The `budget-overflow-streaming.yaml` config installs a tiny
//! `openai_agentic_loop.max_retained_bytes`. The backend commits the client
//! stream with `response.created`, then streams an output frame whose weighted
//! ledger charge exceeds the remaining budget. The committed stream must
//! terminate with one SSE `error` event, suppressing `response.completed` and
//! the `[DONE]` sentinel.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    thread,
    time::Duration,
};

use praxis_test_utils::{free_port, json_post, load_example_config, parse_body, parse_status, start_proxy};

const EXAMPLE: &str = "openai/responses/budget-overflow-streaming.yaml";

const CREATED_EVENT: &str = concat!(
    "event: response.created\n",
    "data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",",
    "\"object\":\"response\",\"created_at\":0,\"model\":\"gpt-4.1\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
);

#[test]
fn committed_stream_overflow_terminates_with_sse_error() {
    // The 16 KiB raw frame is charged at 32x, plus a transient parsing peak, so
    // its ledger charge exceeds the configured 128 KiB budget.
    let oversized = "A".repeat(16_384);
    let delta_event = format!(
        "event: response.output_text.delta\n\
         data: {{\"type\":\"response.output_text.delta\",\"sequence_number\":1,\
         \"content_index\":0,\"output_index\":0,\"response_id\":\"resp_1\",\"delta\":\"{oversized}\"}}\n\n"
    );
    let completed_event = format!(
        "event: response.completed\n\
         data: {{\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{{\"id\":\"resp_1\",\
         \"object\":\"response\",\"model\":\"gpt-4.1\",\"status\":\"completed\",\"output\":[{{\"type\":\"message\",\
         \"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{{\"type\":\"output_text\",\
         \"annotations\":[],\"text\":\"{oversized}\"}}]}}]}}}}\n\n"
    );

    let backend_port = start_streaming_backend(vec![CREATED_EVENT.to_owned(), delta_event, completed_event]);
    let proxy = start_example_proxy(backend_port);
    let raw = read_response_to_end(
        proxy.addr(),
        r#"{"model":"gpt-4.1","input":"Reply briefly.","stream":true}"#,
    );
    let body = parse_body(&raw);
    let events = sse_event_names(&body);

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the overflow, so the status stays 200: {raw}"
    );
    assert!(
        raw.to_ascii_lowercase().contains("text/event-stream"),
        "the committed SSE content type must stay: {raw}"
    );
    assert!(
        body.contains("response.created"),
        "the committing created event must reach the client: {body}"
    );
    assert!(
        body.contains("event: error"),
        "the committed stream must terminate with a terminal SSE error event: {body}"
    );
    let error_count = events.iter().filter(|event| event.as_str() == "error").count();
    assert_eq!(
        error_count, 1,
        "the committed stream must emit exactly one error event: {events:?}"
    );
    assert_eq!(
        events.last().map(String::as_str),
        Some("error"),
        "the terminal error must be the final SSE event: {events:?}"
    );
    assert!(
        body.contains("processing limit set by the platform administrator"),
        "the terminal error must use the sanitized processing-limit message: {body}"
    );
    assert!(
        !body.contains("max_retained_bytes"),
        "the terminal error must not expose the budget configuration: {body}"
    );
    assert!(
        !body.contains("response.completed"),
        "response.completed must be suppressed after the committed-stream overflow: {body}"
    );
    assert!(
        !body.contains("[DONE]"),
        "the [DONE] sentinel must be suppressed after the committed-stream overflow: {body}"
    );
}

fn sse_event_names(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter_map(|frame| {
            frame
                .lines()
                .find_map(|line| line.strip_prefix("event:").map(str::trim))
                .filter(|event| !event.is_empty())
                .map(str::to_owned)
        })
        .collect()
}

fn start_example_proxy(backend_port: u16) -> praxis_test_utils::ProxyGuard {
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend_port)]));
    start_proxy(&config)
}

/// Serve one SSE response composed of the supplied frames, then close.
fn start_streaming_backend(chunks: Vec<String>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let port = listener.local_addr().expect("backend should have an address").port();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_request(&mut stream);
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        .expect("response headers should be written");
        for chunk in chunks {
            write!(stream, "{:x}\r\n{chunk}\r\n", chunk.len()).expect("response chunk should be written");
            stream.flush().expect("chunk should flush");
        }
        // The client may already have disconnected after the terminal error, so
        // the final empty chunk is best-effort.
        drop(stream.write_all(b"0\r\n\r\n"));
        drop(stream.flush());
    });
    port
}

fn read_request(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("backend read timeout should be set");
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = stream.read(&mut buffer).expect("backend request read should succeed");
        assert!(count > 0, "request must complete before connection closes");
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
            return;
        }
    }
}

fn read_response_to_end(proxy_addr: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(proxy_addr).expect("client should connect to proxy");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("client read timeout should be set");
    stream
        .write_all(json_post("/v1/responses", body).as_bytes())
        .expect("client request should be written");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("proxy response should end");
    String::from_utf8_lossy(&raw).into_owned()
}
