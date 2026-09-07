// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for filter-selected terminal Responses streaming in IRR.

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    thread,
    time::Duration,
};

use praxis_test_utils::{free_port, json_post, load_example_config, parse_body, parse_status, start_proxy};

const EXAMPLE: &str = "openai/responses/irr-terminal-streaming.yaml";
const FIRST_EVENT: &str = concat!(
    "event: response.output_text.delta\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hel\"}\n\n",
);
const FINAL_EVENT: &str = concat!(
    "event: response.completed\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",",
    "\"object\":\"response\",\"status\":\"completed\",\"output\":[]}}\n\n",
);

#[test]
fn sse_event_reaches_client_before_upstream_completes() {
    let (backend_port, first_sent, release, backend_thread) = start_gated_backend(
        vec![
            "event: response.output_text.delta\nda".to_owned(),
            "ta: {\"type\":\"response.output_text.delta\",\"delta\":\"hel\"}\n\n".to_owned(),
        ],
        vec![FINAL_EVENT.to_owned()],
        "text/event-stream",
    );
    let proxy = start_example_proxy(backend_port);
    let (observed_tx, observed_rx) = mpsc::channel();
    let (complete_tx, complete_rx) = mpsc::channel();
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let raw = read_response_incrementally(
            &proxy_addr,
            r#"{"model":"gpt-4.1","input":"hello","stream":true}"#,
            "\"delta\":\"hel\"}\n\n",
            &observed_tx,
        );
        complete_tx.send(raw).expect("test receiver should remain available");
    });

    first_sent
        .recv_timeout(Duration::from_secs(2))
        .expect("backend should send the fragmented first event");
    observed_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("client should observe one complete SSE event while upstream is still gated");
    release
        .send(())
        .expect("backend release receiver should remain available");

    let raw = complete_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("client should receive the completed stream");
    assert_eq!(parse_status(&raw), 200, "terminal stream should return 200: {raw}");
    let body = parse_body(&raw);
    assert!(
        body.contains(FIRST_EVENT),
        "fragmented event should pass through intact: {body}"
    );
    assert!(
        body.contains("response.completed"),
        "terminal event should reach the client: {body}"
    );

    client.join().expect("client thread should not panic");
    backend_thread.join().expect("backend thread should not panic");
}

#[test]
fn stream_false_preserves_buffered_irr_response() {
    let (backend_port, first_sent, release, backend_thread) = start_gated_backend(
        vec![r#"{"id":"resp_buffered","object":"res"#.to_owned()],
        vec![r#"ponse","status":"completed"}"#.to_owned()],
        "application/json",
    );
    let proxy = start_example_proxy(backend_port);
    let (first_byte_tx, first_byte_rx) = mpsc::channel();
    let (complete_tx, complete_rx) = mpsc::channel();
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let mut stream = connect_and_send(&proxy_addr, r#"{"model":"gpt-4.1","input":"hello","stream":false}"#);
        let mut first = [0_u8; 1024];
        let count = stream.read(&mut first).expect("proxy response read should succeed");
        first_byte_tx
            .send(())
            .expect("first-byte receiver should remain available");
        let mut raw = first[..count].to_vec();
        stream.read_to_end(&mut raw).expect("proxy response should complete");
        complete_tx
            .send(String::from_utf8_lossy(&raw).into_owned())
            .expect("test receiver should remain available");
    });

    first_sent
        .recv_timeout(Duration::from_secs(2))
        .expect("backend should send the first JSON chunk");
    let arrived_while_incomplete = first_byte_rx.recv_timeout(Duration::from_millis(250)).is_ok();
    release
        .send(())
        .expect("backend release receiver should remain available");

    let raw = complete_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("buffered response should complete after upstream EOF");
    assert!(
        !arrived_while_incomplete,
        "stream=false must not expose headers or body before the upstream response is complete"
    );
    assert_eq!(parse_status(&raw), 200, "buffered response should return 200: {raw}");
    assert_eq!(
        parse_body(&raw),
        r#"{"id":"resp_buffered","object":"response","status":"completed"}"#,
        "buffered body should be preserved"
    );

    client.join().expect("client thread should not panic");
    backend_thread.join().expect("backend thread should not panic");
}

#[test]
fn downstream_cancellation_closes_terminal_upstream_stream() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let backend_port = listener.local_addr().expect("backend should have an address").port();
    let (first_sent_tx, first_sent_rx) = mpsc::channel();
    let (client_dropped_tx, client_dropped_rx) = mpsc::channel();
    let (cancelled_tx, cancelled_rx) = mpsc::channel();
    let backend_thread = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_request(&mut stream);
        write_stream_headers(&mut stream, "text/event-stream");
        write_chunk(&mut stream, FIRST_EVENT);
        stream.flush().expect("first event should flush");
        first_sent_tx.send(()).expect("test receiver should remain available");

        client_dropped_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("client should drop the downstream stream");
        thread::sleep(Duration::from_millis(50));
        let payload = "x".repeat(64 * 1024);
        let mut upstream_closed = false;
        for _ in 0..256 {
            if write!(stream, "{:x}\r\n{payload}\r\n", payload.len())
                .and_then(|()| stream.flush())
                .is_err()
            {
                upstream_closed = true;
                break;
            }
        }
        if !upstream_closed {
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .expect("backend read timeout should be set");
            let mut byte = [0_u8; 1];
            upstream_closed = matches!(stream.read(&mut byte), Ok(0));
        }
        cancelled_tx
            .send(upstream_closed)
            .expect("test receiver should remain available");
    });
    let proxy = start_example_proxy(backend_port);
    let proxy_addr = proxy.addr().to_owned();

    let client = thread::spawn(move || {
        let mut stream = connect_and_send(&proxy_addr, r#"{"model":"gpt-4.1","input":"hello","stream":true}"#);
        let mut received = Vec::new();
        let mut buffer = [0_u8; 1024];
        while !String::from_utf8_lossy(&received).contains("response.output_text.delta") {
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
            .recv_timeout(Duration::from_secs(5))
            .expect("backend should observe cancellation"),
        "dropping the downstream response must close the upstream streaming exchange"
    );
    backend_thread.join().expect("backend thread should not panic");
}

#[test]
fn late_upstream_failure_does_not_replace_committed_sse() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let backend_port = listener.local_addr().expect("backend should have an address").port();
    let backend_thread = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_request(&mut stream);
        write_stream_headers(&mut stream, "text/event-stream");
        write_chunk(&mut stream, FIRST_EVENT);
        stream.flush().expect("first event should flush");
        stream
            .write_all(b"20\r\nincomplete")
            .expect("malformed late chunk should be written");
    });
    let proxy = start_example_proxy(backend_port);
    let raw = read_response_to_end(proxy.addr(), r#"{"model":"gpt-4.1","input":"hello","stream":true}"#);

    assert_eq!(
        parse_status(&raw),
        200,
        "headers are committed before the late failure: {raw}"
    );
    assert!(
        raw.contains("response.output_text.delta"),
        "the event delivered before the transport failure must be preserved: {raw}"
    );
    assert!(
        !raw.to_ascii_lowercase().contains("bad gateway") && !raw.contains("\"error\""),
        "a late failure must not replace the committed SSE response: {raw}"
    );
    backend_thread.join().expect("backend thread should not panic");
}

fn start_example_proxy(backend_port: u16) -> praxis_test_utils::ProxyGuard {
    let proxy_port = free_port();
    let config = load_example_config(EXAMPLE, proxy_port, HashMap::from([("127.0.0.1:3001", backend_port)]));
    start_proxy(&config)
}

fn start_gated_backend(
    first_chunks: Vec<String>,
    final_chunks: Vec<String>,
    content_type: &'static str,
) -> (u16, mpsc::Receiver<()>, mpsc::Sender<()>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("backend should bind");
    let port = listener.local_addr().expect("backend should have an address").port();
    let (first_sent_tx, first_sent_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("backend should accept request");
        read_request(&mut stream);
        write_stream_headers(&mut stream, content_type);
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

fn connect_and_send(proxy_addr: &str, body: &str) -> TcpStream {
    let mut stream = TcpStream::connect(proxy_addr).expect("client should connect to proxy");
    stream
        .set_read_timeout(Some(Duration::from_secs(4)))
        .expect("client read timeout should be set");
    stream
        .write_all(json_post("/v1/responses", body).as_bytes())
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
