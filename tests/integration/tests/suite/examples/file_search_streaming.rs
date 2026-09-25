// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional integration tests for the streaming file-search-dispatch example config.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        mpsc::{self, Sender},
    },
    thread,
    time::Duration,
};

use praxis_test_utils::{
    free_port, json_post, load_example_config, parse_body, parse_status, start_capturing_backend, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "openai/responses/file-search-streaming.yaml";

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn connect_and_send(proxy_addr: &str, body: &str) -> TcpStream {
    let mut stream = TcpStream::connect(proxy_addr).expect("client should connect to proxy");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("client read timeout should be set");
    stream
        .write_all(json_post("/v1/responses", body).as_bytes())
        .expect("client request should be written");
    stream
}

fn read_response_to_end(proxy_addr: &str, body: &str) -> String {
    let mut stream = connect_and_send(proxy_addr, body);
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("proxy response should end");
    String::from_utf8_lossy(&raw).into_owned()
}

fn read_request(stream: &mut TcpStream) {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
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

struct MultiRoundBackend {
    port: u16,
    request_count: Arc<Mutex<usize>>,
    thread: Option<thread::JoinHandle<()>>,
    shutdown_tx: Option<Sender<()>>,
}

impl MultiRoundBackend {
    fn start(rounds: Vec<Vec<String>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
        let port = listener.local_addr().expect("backend should have an address").port();
        let request_count = Arc::new(Mutex::new(0_usize));
        let request_count_clone = Arc::clone(&request_count);
        let (shutdown_tx, shutdown_rx) = mpsc::channel();

        let thread = thread::spawn(move || {
            listener
                .set_nonblocking(true)
                .expect("listener should be set to non-blocking");

            for chunks in rounds {
                // Wait for the next connection with shutdown check
                let stream = loop {
                    if shutdown_rx.try_recv().is_ok() {
                        return;
                    }
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(50));
                        },
                        Err(e) => panic!("backend accept failed: {e}"),
                    }
                };

                *request_count_clone.lock().expect("request count lock should succeed") += 1;

                let mut stream = stream;
                read_request(&mut stream);
                write_stream_headers(&mut stream, "text/event-stream");

                for chunk in chunks {
                    write_chunk(&mut stream, &chunk);
                }

                stream.write_all(b"0\r\n\r\n").expect("chunked response should finish");
                stream.flush().expect("round should flush");
            }
        });

        Self {
            port,
            request_count,
            thread: Some(thread),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    fn port(&self) -> u16 {
        self.port
    }

    fn request_count(&self) -> usize {
        *self.request_count.lock().expect("request count lock should succeed")
    }
}

impl Drop for MultiRoundBackend {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _unused = tx.send(());
        }
        if let Some(thread) = self.thread.take() {
            drop(thread.join());
        }
    }
}

fn build_private_file_search_round_sse() -> Vec<String> {
    // Round 0: model emits a private function_call named "file_search"
    // The filter will suppress this and synthesize lifecycle events
    vec![
        concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_private\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_priv\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"Q4 results\\\"}\",\"status\":\"completed\"},\"sequence_number\":1}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_priv\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"Q4 results\\\"}\",\"status\":\"completed\"},\"sequence_number\":2}\n\n"
        ).to_owned(),
        concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_private\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"fc_priv\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"Q4 results\\\"}\",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n"
        ).to_owned(),
    ]
}

fn build_terminal_round_sse_with_citation() -> Vec<String> {
    // Round 1: terminal message with citation-annotated output
    vec![
        concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_final\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_final\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]},\"sequence_number\":1}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Q4 revenue was $42 million <|file-q4|>\",\"sequence_number\":2}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_final\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Q4 revenue was $42 million <|file-q4|>\"}]},\"sequence_number\":3}\n\n"
        ).to_owned(),
        concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_final\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_final\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Q4 revenue was $42 million <|file-q4|>\"}]}],\"usage\":{\"input_tokens\":20,\"output_tokens\":7,\"total_tokens\":27}}}\n\n"
        ).to_owned(),
    ]
}

fn build_native_hybrid_round_sse() -> Vec<String> {
    // Round 0: native file_search_call(searching) opening passes, pending done suppressed
    vec![
        concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_hybrid\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fs_native\",\"type\":\"file_search_call\",\"status\":\"searching\",\"queries\":[\"Q4 results\"]},\"sequence_number\":1}\n\n"
        ).to_owned(),
        concat!(
            "event: response.file_search_call.in_progress\n",
            "data: {\"type\":\"response.file_search_call.in_progress\",\"output_index\":0,\"item_id\":\"fs_native\",\"sequence_number\":2}\n\n"
        ).to_owned(),
        concat!(
            "event: response.file_search_call.searching\n",
            "data: {\"type\":\"response.file_search_call.searching\",\"output_index\":0,\"item_id\":\"fs_native\",\"sequence_number\":3}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fs_native\",\"type\":\"file_search_call\",\"status\":\"searching\",\"queries\":[\"Q4 results\"]},\"sequence_number\":4}\n\n"
        ).to_owned(),
        concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_hybrid\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"fs_native\",\"type\":\"file_search_call\",\"status\":\"searching\",\"queries\":[\"Q4 results\"]}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n"
        ).to_owned(),
    ]
}

fn build_native_passthrough_round_sse() -> Vec<String> {
    // Provider resolves the call mid-stream with completed status
    vec![
        concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_pass\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fs_pass\",\"type\":\"file_search_call\",\"status\":\"searching\",\"queries\":[\"Q4 results\"]},\"sequence_number\":1}\n\n"
        ).to_owned(),
        concat!(
            "event: response.file_search_call.completed\n",
            "data: {\"type\":\"response.file_search_call.completed\",\"output_index\":0,\"item_id\":\"fs_pass\",\"sequence_number\":2}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fs_pass\",\"type\":\"file_search_call\",\"status\":\"completed\",\"queries\":[\"Q4 results\"],\"results\":[{\"file_id\":\"file-q4\",\"filename\":\"q4.txt\",\"score\":0.99,\"content\":[{\"type\":\"text\",\"text\":\"Revenue $42M\"}]}]},\"sequence_number\":3}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_pass\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]},\"sequence_number\":4}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"content_index\":0,\"delta\":\"Based on the search, Q4 revenue was $42M.\",\"sequence_number\":5}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_pass\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Based on the search, Q4 revenue was $42M.\"}]},\"sequence_number\":6}\n\n"
        ).to_owned(),
        concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_pass\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"fs_pass\",\"type\":\"file_search_call\",\"status\":\"completed\",\"queries\":[\"Q4 results\"],\"results\":[{\"file_id\":\"file-q4\",\"filename\":\"q4.txt\",\"score\":0.99,\"content\":[{\"type\":\"text\",\"text\":\"Revenue $42M\"}]}]},{\"id\":\"msg_pass\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Based on the search, Q4 revenue was $42M.\"}]}],\"usage\":{\"input_tokens\":20,\"output_tokens\":10,\"total_tokens\":30}}}\n\n"
        ).to_owned(),
    ]
}

fn build_multi_round_first_search() -> Vec<String> {
    build_private_file_search_round_sse()
}

fn build_multi_round_second_search() -> Vec<String> {
    // Round 1: another private file_search for multi-round test
    vec![
        concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2nd\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_2nd\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"margins\\\"}\",\"status\":\"completed\"},\"sequence_number\":1}\n\n"
        ).to_owned(),
        concat!(
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_2nd\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"margins\\\"}\",\"status\":\"completed\"},\"sequence_number\":2}\n\n"
        ).to_owned(),
        concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2nd\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"fc_2nd\",\"type\":\"function_call\",\"name\":\"file_search\",\"arguments\":\"{\\\"query\\\":\\\"margins\\\"}\",\"status\":\"completed\"}],\"usage\":{\"input_tokens\":10,\"output_tokens\":5,\"total_tokens\":15}}}\n\n"
        ).to_owned(),
    ]
}

fn build_multi_round_terminal() -> Vec<String> {
    build_terminal_round_sse_with_citation()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn private_call_full_loop_synthesizes_lifecycle() {
    let model = MultiRoundBackend::start(vec![
        build_private_file_search_round_sse(),
        build_terminal_round_sse_with_citation(),
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
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "What do the uploaded documents say about Q4 results?",
        "include": ["file_search_call.results"],
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}],
        "stream": true
    });

    let raw = read_response_to_end(proxy.addr(), &request.to_string());

    assert_eq!(parse_status(&raw), 200, "streaming request should succeed: {raw}");
    let body = parse_body(&raw);

    // Private function_call must be suppressed
    assert!(
        !body.contains(r#""type":"function_call""#),
        "private function_call should be suppressed on the wire"
    );

    // Synthesized lifecycle must be present
    assert!(
        body.contains(r#""type":"file_search_call""#) && body.contains("response.output_item.added"),
        "synthesized lifecycle should include output_item.added with file_search_call: {body}"
    );
    assert!(
        body.contains("response.file_search_call.in_progress"),
        "synthesized lifecycle should include in_progress: {body}"
    );
    assert!(
        body.contains("response.file_search_call.searching"),
        "synthesized lifecycle should include searching: {body}"
    );
    assert!(
        body.contains("response.file_search_call.completed"),
        "synthesized lifecycle should include completed: {body}"
    );

    // Exactly one response.created and one response.completed on the logical stream
    let created_count = body.matches("event: response.created").count();
    let completed_count = body.matches("event: response.completed").count();
    assert_eq!(
        created_count, 1,
        "logical stream should emit exactly one response.created"
    );
    assert_eq!(
        completed_count, 1,
        "logical stream should emit exactly one response.completed"
    );

    // Terminal response.completed should carry the annotated answer
    assert!(
        body.contains("Q4 revenue was $42 million"),
        "terminal output should include the answer: {body}"
    );

    // Search bridging assertion
    let search_request: Value = serde_json::from_str(&search.body()).expect("search request should be JSON");
    assert_eq!(search_request["query"], "Q4 results");

    // Model round count
    assert_eq!(model.request_count(), 2, "should have made 2 model requests");
}

#[test]
fn closed_search_failure_emits_error_no_done() {
    let model = MultiRoundBackend::start(vec![build_private_file_search_round_sse()]);
    let search_error = json!({
        "error": {
            "message": "Rate limit reached for vector store search",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    })
    .to_string();

    // Craft a non-200 search backend
    let listener = TcpListener::bind("127.0.0.1:0").expect("search backend should bind");
    let search_port = listener.local_addr().expect("search should have an address").port();
    let error_body = search_error.clone();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("search should accept");
        read_request(&mut stream);
        write!(
            stream,
            "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            error_body.len(),
            error_body
        )
        .expect("error response should be written");
        stream.flush().expect("error response should flush");
    });

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search_port)]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "What do the uploaded documents say about Q4 results?",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}],
        "stream": true
    });

    let raw = read_response_to_end(proxy.addr(), &request.to_string());

    assert_eq!(parse_status(&raw), 200, "SSE commits 200 before failure: {raw}");
    let body = parse_body(&raw);

    // Single event: error
    assert!(
        body.contains("event: error"),
        "closed failure should emit error event: {body}"
    );

    // No [DONE] marker
    assert!(
        !body.contains("data: [DONE]"),
        "error stream should not emit [DONE]: {body}"
    );

    // No synthesized response.completed
    assert!(
        !body.contains("event: response.completed"),
        "error stream should not emit response.completed: {body}"
    );

    // Should have only made one model request
    assert_eq!(model.request_count(), 1, "closed failure should not reinfer");
}

#[test]
fn multi_round_two_searches_then_terminal() {
    let model = MultiRoundBackend::start(vec![
        build_multi_round_first_search(),
        build_multi_round_second_search(),
        build_multi_round_terminal(),
    ]);
    let search = start_capturing_backend(
        &json!({
            "data": [{
                "file_id": "file-q4",
                "filename": "q4.txt",
                "score": 0.99,
                "content": [{"type": "text", "text": "Revenue data."}],
                "attributes": null
            }]
        })
        .to_string(),
    );

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "Q4 and margins",
        "include": ["file_search_call.results"],
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}],
        "stream": true
    });

    let raw = read_response_to_end(proxy.addr(), &request.to_string());

    assert_eq!(parse_status(&raw), 200, "multi-round should succeed: {raw}");
    let body = parse_body(&raw);

    // Exactly one logical response.created and response.completed
    let created_count = body.matches("event: response.created").count();
    let completed_count = body.matches("event: response.completed").count();
    assert_eq!(
        created_count, 1,
        "logical stream should emit exactly one response.created"
    );
    assert_eq!(
        completed_count, 1,
        "logical stream should emit exactly one response.completed"
    );

    // Model round count should be 3
    assert_eq!(
        model.request_count(),
        3,
        "should have made 3 model requests (2 searches + 1 terminal)"
    );

    // Should see synthesis for both searches
    let lifecycle_count = body.matches("response.file_search_call.searching").count();
    assert!(
        lifecycle_count >= 2,
        "should synthesize lifecycle for both searches: {body}"
    );
}

#[test]
fn native_hybrid_suppresses_pending_done_synthesizes_tail() {
    let model = MultiRoundBackend::start(vec![
        build_native_hybrid_round_sse(),
        build_terminal_round_sse_with_citation(),
    ]);
    let search = start_capturing_backend(
        &json!({
            "data": [{
                "file_id": "file-q4",
                "filename": "q4.txt",
                "score": 0.99,
                "content": [{"type": "text", "text": "Q4 data."}],
                "attributes": null
            }]
        })
        .to_string(),
    );

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", search.port())]),
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "Q4 results",
        "include": ["file_search_call.results"],
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}],
        "stream": true
    });

    let raw = read_response_to_end(proxy.addr(), &request.to_string());

    assert_eq!(parse_status(&raw), 200, "native hybrid should succeed: {raw}");
    let body = parse_body(&raw);

    // Native opening events should pass through
    assert!(
        body.contains("response.file_search_call.in_progress"),
        "native opening should include in_progress: {body}"
    );
    assert!(
        body.contains("response.file_search_call.searching"),
        "native opening should include searching: {body}"
    );

    // Synthesized completed tail must be present
    assert!(
        body.contains("response.file_search_call.completed"),
        "synthesized tail should include completed: {body}"
    );

    // Exactly one logical response.created and response.completed
    let created_count = body.matches("event: response.created").count();
    let completed_count = body.matches("event: response.completed").count();
    assert_eq!(
        created_count, 1,
        "logical stream should emit exactly one response.created"
    );
    assert_eq!(
        completed_count, 1,
        "logical stream should emit exactly one response.completed"
    );

    assert_eq!(model.request_count(), 2, "should have made 2 model requests");
}

#[test]
fn native_passthrough_no_synthesized_tail() {
    let model = MultiRoundBackend::start(vec![build_native_passthrough_round_sse()]);

    let proxy_port = free_port();
    let config = load_example_config(
        EXAMPLE,
        proxy_port,
        HashMap::from([("127.0.0.1:3001", model.port()), ("127.0.0.1:8001", 9999)]), // search unused
    );
    let proxy = start_proxy(&config);

    let request = json!({
        "model": "llama-3.3-70b",
        "input": "Q4 results",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_q4"]}],
        "stream": true
    });

    let raw = read_response_to_end(proxy.addr(), &request.to_string());

    assert_eq!(parse_status(&raw), 200, "native passthrough should succeed: {raw}");
    let body = parse_body(&raw);

    // Provider emitted file_search_call.completed mid-stream
    assert!(
        body.contains("response.file_search_call.completed"),
        "provider should emit completed: {body}"
    );

    // F1: a provider-completed native call streams its full lifecycle live; EOS must NOT
    // re-synthesize a tail, or the client would see the terminal events twice. Anchor on
    // the `event:` line so each SSE frame is counted exactly once.
    let fs_completed_frames = body.matches("event: response.file_search_call.completed").count();
    assert_eq!(
        fs_completed_frames, 1,
        "native completed must appear exactly once (no duplicate synthesized tail): {body}"
    );
    let item_done_frames = body.matches("event: response.output_item.done").count();
    assert_eq!(
        item_done_frames, 2,
        "exactly one output_item.done per live item (file_search_call + message), no synthesized duplicate: {body}"
    );

    // Exactly one logical response.created and response.completed
    let created_count = body.matches("event: response.created").count();
    let completed_count = body.matches("event: response.completed").count();
    assert_eq!(
        created_count, 1,
        "logical stream should emit exactly one response.created"
    );
    assert_eq!(
        completed_count, 1,
        "logical stream should emit exactly one response.completed"
    );

    // Only one model request since provider resolved it
    assert_eq!(
        model.request_count(),
        1,
        "native passthrough should only make 1 model request"
    );
}
