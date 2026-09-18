// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Scripted HTTP backend for integration testing of the
//! `POST /v1/responses` flow.
//!
//! The backend accepts HTTP requests at the configured path, captures
//! the full request (method, headers, body), and replays a deterministic
//! sequence of SSE or JSON responses. `WebSocket` upgrade attempts are
//! reported as `HttpBackendEvent::WebSocketUpgrade` so callers can fail
//! fast when the test boundary forbids the transport.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use http::HeaderMap;
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::TcpListener,
    sync::mpsc,
    task::JoinSet,
};
use tracing::debug;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum HTTP request-head size accepted by the scripted backend.
const MAX_HEAD_BYTES: usize = 16_384; // 16 KiB
/// Maximum time to wait for a complete HTTP request head.
const REQUEST_HEAD_TIMEOUT: Duration = Duration::from_secs(5);
/// Maximum HTTP request-body size captured or drained by the backend.
const MAX_REQUEST_BODY_BYTES: usize = 16_777_216; // 16 MiB

/// A captured HTTP request from a test client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedHttpRequest {
    /// Full request body.
    pub body: Bytes,
    /// Request headers observed by the backend.
    pub headers: HeaderMap,
    /// HTTP method used by the client (e.g. `"POST"`).
    pub method: String,
    /// Path component of the request target (without query string).
    pub path: String,
    /// Query string portion of the request target, if any.
    pub query: Option<String>,
}

/// An observation emitted by the scripted HTTP backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpBackendEvent {
    /// A complete HTTP request was received.
    Request(CapturedHttpRequest),
    /// A matching request declared a body larger than the capture ceiling.
    RequestTooLarge {
        /// Declared request-body size.
        body_bytes: usize,
        /// Maximum body size accepted by this backend.
        max_body_bytes: usize,
    },
    /// A valid request arrived after every scripted response was consumed.
    ScriptExhausted {
        /// Zero-based turn index requested by the client.
        turn: usize,
    },
    /// The backend received an unexpected HTTP request (wrong method, path,
    /// or unexpected upgrade attempt) before the configured path matched.
    UnexpectedRequest {
        /// Request method parsed from the request line.
        method: String,
        /// Request path parsed from the request line.
        path: String,
    },
    /// The client attempted a `WebSocket` upgrade on this listener. The
    /// backend always rejects it because the test boundary is HTTP-only.
    WebSocketUpgrade {
        /// Request method parsed from the request line.
        method: String,
        /// Request path parsed from the request line.
        path: String,
        /// Observed upgrade header value.
        upgrade: String,
    },
}

/// A deterministic response action performed after each request turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HttpServerAction {
    /// Respond with a non-streaming JSON body.
    Json {
        /// JSON body to return.
        body: String,
        /// HTTP status code (e.g. `200`).
        status: u16,
    },
    /// Respond with a chunked transfer-encoded body (SSE-style frames).
    StreamSse {
        /// Pre-rendered SSE event payload lines, each terminated with `\n\n`.
        events: Vec<String>,
        /// Delay after each non-final event; zero keeps the fast fixture path.
        inter_event_delay: Duration,
    },
}

/// RAII guard for a scripted HTTP backend.
///
/// Dropping the guard stops the listener and aborts all connection tasks.
pub struct HttpBackendGuard {
    /// Stream of backend observations.
    events: mpsc::Receiver<HttpBackendEvent>,
    /// Listener task, which owns all connection tasks.
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Port allocated by the operating system.
    port: u16,
    /// Listener shutdown signal.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl HttpBackendGuard {
    /// Wait for the next backend observation.
    pub async fn next_event(&mut self) -> Option<HttpBackendEvent> {
        self.events.recv().await
    }

    /// Return the allocated port.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for HttpBackendGuard {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _sent = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// Start an HTTP backend that replays `script` for each sequential request.
///
/// The backend accepts HTTP/1.1 `POST` (or other configured method) requests
/// at the configured exact path; every accepted request triggers the next scripted action.
/// `WebSocket` upgrade attempts are reported as `HttpBackendEvent::WebSocketUpgrade`
/// and the connection is closed with a 400 response.
///
/// # Panics
///
/// Panics if the loopback listener cannot bind or report its local address.
pub async fn start_scripted_http_backend(
    expected_method: &str,
    expected_path: &str,
    script: Vec<HttpServerAction>,
) -> HttpBackendGuard {
    start_scripted_http_backend_turns(expected_method, expected_path, vec![script]).await
}

/// Start an HTTP backend that replays one action sequence for each
/// sequential request.
///
/// Each `turn` element is a list of [`HttpServerAction`] entries. The
/// backend advances to the next turn after the previous one has been
/// delivered. `Json` and `StreamSse` actions both consume exactly one request.
///
/// # Panics
///
/// Panics if the loopback listener cannot bind or report its local address.
pub async fn start_scripted_http_backend_turns(
    expected_method: &str,
    expected_path: &str,
    turns: Vec<Vec<HttpServerAction>>,
) -> HttpBackendGuard {
    let listener = bind_listener().await;
    let port = listener
        .local_addr()
        .expect("scripted HTTP backend should have a local address")
        .port();
    let (event_tx, event_rx) = mpsc::channel(256);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let state = Arc::new(ScriptState::new(expected_method, expected_path, turns));
    let handle = tokio::spawn(run_listener(listener, state, event_tx, shutdown_rx));
    debug!(port, "scripted HTTP backend listening");

    HttpBackendGuard {
        events: event_rx,
        handle: Some(handle),
        port,
        shutdown: Some(shutdown_tx),
    }
}

/// Mutable accumulator shared across [`apply_header_line`] calls.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the independent header facts make request validation explicit"
)]
struct HeaderParseState {
    /// Declared request body length.
    body_bytes: usize,
    /// Whether any `Connection` header listed the `upgrade` token.
    connection_upgrade: bool,
    /// Parsed header map.
    headers: HeaderMap,
    /// Whether the request head contains a malformed header or content length.
    malformed: bool,
    /// Observed `Upgrade` header value (preserved verbatim).
    upgrade_value: String,
    /// Whether any `Upgrade` header carried the `websocket` token.
    websocket_upgrade: bool,
}

/// Aggregated facts extracted from a request head's header lines.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the independent header facts make request validation explicit"
)]
struct ParsedHeaders {
    /// Declared request body length.
    body_bytes: usize,
    /// Whether any `Connection` header listed the `upgrade` token.
    connection_upgrade: bool,
    /// Parsed header map.
    headers: HeaderMap,
    /// Whether the request head contains a malformed header or content length.
    malformed: bool,
    /// Observed `Upgrade` header value (preserved verbatim).
    upgrade_value: String,
    /// Whether any `Upgrade` header carried the `websocket` token.
    websocket_upgrade: bool,
}

/// Parsed facts needed before reading a request body.
#[derive(Clone, Debug)]
struct PeekedHttpRequest {
    /// Declared request body length.
    body_bytes: usize,
    /// Number of bytes through the terminating blank header line.
    head_bytes: usize,
    /// Parsed request headers.
    headers: HeaderMap,
    /// Whether the request head contains a malformed header or content length.
    malformed: bool,
    /// Request method.
    method: String,
    /// Request path component.
    path: String,
    /// Optional query string from the request target.
    query: Option<String>,
    /// Observed `Upgrade` header value.
    upgrade_value: String,
    /// Whether the request declares a `WebSocket` upgrade.
    websocket_upgrade: bool,
}

/// Method and path accepted by the scripted backend.
#[derive(Debug)]
struct RequestExpectation {
    /// Required HTTP method.
    method: String,
    /// Required request path without a query string.
    path: String,
}

/// Shared immutable script plus the globally claimed turn index.
#[derive(Debug)]
struct ScriptState {
    /// Method and path accepted by the backend.
    expectation: RequestExpectation,
    /// Next response turn claimed by a valid request.
    turn_index: AtomicUsize,
    /// Ordered response actions, shared across every connection.
    turns: Box<[Vec<HttpServerAction>]>,
}

impl ScriptState {
    /// Build listener-owned state for one backend instance.
    fn new(expected_method: &str, expected_path: &str, turns: Vec<Vec<HttpServerAction>>) -> Self {
        Self {
            expectation: RequestExpectation {
                method: expected_method.to_owned(),
                path: expected_path.to_owned(),
            },
            turn_index: AtomicUsize::new(0),
            turns: turns.into_boxed_slice(),
        }
    }

    /// Return whether a parsed request satisfies the configured contract.
    fn expects(&self, request: &PeekedHttpRequest) -> bool {
        request.method == self.expectation.method && request.path == self.expectation.path
    }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Bind an ephemeral IPv4 loopback listener for one backend instance.
async fn bind_listener() -> TcpListener {
    TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("scripted HTTP backend should bind")
}

/// Accept connections until the guard signals shutdown.
async fn run_listener(
    listener: TcpListener,
    state: Arc<ScriptState>,
    event_tx: mpsc::Sender<HttpBackendEvent>,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    let mut connections = JoinSet::new();

    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let Ok((stream, peer)) = accepted else {
                    break;
                };
                debug!(%peer, "scripted HTTP backend accepted connection");
                connections.spawn(Box::pin(handle_connection(
                    stream,
                    Arc::clone(&state),
                    event_tx.clone(),
                )));
            },
            completed = connections.join_next(), if !connections.is_empty() => {
                if completed.is_none() {
                    break;
                }
            },
        }
    }

    connections.abort_all();
    while connections.join_next().await.is_some() {}
}

/// Complete one HTTP request/response cycle against a single connection.
#[expect(
    clippy::large_stack_frames,
    reason = "bounded test-backend connection state is only slightly above the workspace threshold"
)]
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    state: Arc<ScriptState>,
    event_tx: mpsc::Sender<HttpBackendEvent>,
) {
    let Some(peek) = peek_http_request(&stream, REQUEST_HEAD_TIMEOUT).await else {
        return;
    };

    if peek.malformed {
        reject_malformed_request(&mut stream, peek.head_bytes).await;
        return;
    }

    if peek.websocket_upgrade {
        let _handled = handle_websocket_upgrade(&mut stream, &event_tx, &peek).await;
        return;
    }

    if !state.expects(&peek) {
        reject_unexpected_request(&mut stream, &event_tx, &peek).await;
        return;
    }
    if peek.body_bytes > MAX_REQUEST_BODY_BYTES {
        reject_oversized_request(&mut stream, &event_tx, peek.body_bytes).await;
        return;
    }

    let Some(request) = read_full_request(&mut stream, peek).await else {
        return;
    };
    if event_tx.send(HttpBackendEvent::Request(request)).await.is_err() {
        return;
    }

    respond_to_turn(&mut stream, &state, &event_tx).await;
}

/// Reject a syntactically malformed request head.
async fn reject_malformed_request(stream: &mut tokio::net::TcpStream, head_bytes: usize) {
    let mut head = vec![0_u8; head_bytes];
    if stream.read_exact(&mut head).await.is_err() {
        return;
    }
    let _written = stream
        .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await;
}

/// Reject an oversized matching request without allocating its declared body.
async fn reject_oversized_request(
    stream: &mut tokio::net::TcpStream,
    event_tx: &mpsc::Sender<HttpBackendEvent>,
    body_bytes: usize,
) {
    let _queued = event_tx
        .send(HttpBackendEvent::RequestTooLarge {
            body_bytes,
            max_body_bytes: MAX_REQUEST_BODY_BYTES,
        })
        .await;
    let _written = stream
        .write_all(b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await;
}

/// Reject a request that does not match the backend's method and path contract.
async fn reject_unexpected_request(
    stream: &mut tokio::net::TcpStream,
    event_tx: &mpsc::Sender<HttpBackendEvent>,
    peek: &PeekedHttpRequest,
) {
    let event = HttpBackendEvent::UnexpectedRequest {
        method: peek.method.clone(),
        path: peek.path.clone(),
    };
    if event_tx.send(event).await.is_err() || !drain_unexpected_request(stream, peek.head_bytes, peek.body_bytes).await
    {
        return;
    }
    let _written = stream
        .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await;
}

/// Claim and execute the next global response turn.
async fn respond_to_turn(
    stream: &mut tokio::net::TcpStream,
    state: &ScriptState,
    event_tx: &mpsc::Sender<HttpBackendEvent>,
) {
    let index = state.turn_index.fetch_add(1, Ordering::SeqCst);
    let Some(script) = state.turns.get(index) else {
        let _queued = event_tx.send(HttpBackendEvent::ScriptExhausted { turn: index }).await;
        let _written = stream
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await;
        return;
    };
    if !execute_turn_actions(stream, script).await {
        debug!(turn = index, "scripted HTTP backend did not complete response");
    }
}

/// Handle a `WebSocket` upgrade attempt by sending a rejection response.
///
/// Returns `false` if the stream could not be drained or the rejection
/// could not be written.
async fn handle_websocket_upgrade(
    stream: &mut tokio::net::TcpStream,
    event_tx: &mpsc::Sender<HttpBackendEvent>,
    peek: &PeekedHttpRequest,
) -> bool {
    if event_tx
        .send(HttpBackendEvent::WebSocketUpgrade {
            method: peek.method.clone(),
            path: peek.path.clone(),
            upgrade: peek.upgrade_value.clone(),
        })
        .await
        .is_err()
    {
        return false;
    }
    if !drain_unexpected_request(stream, peek.head_bytes, peek.body_bytes).await {
        return false;
    }
    stream
        .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await
        .is_ok()
}

/// Execute the response action for a single turn, returning whether it was sent.
async fn execute_turn_actions(stream: &mut tokio::net::TcpStream, script: &[HttpServerAction]) -> bool {
    let Some(action) = script.first() else {
        return false;
    };
    match action {
        HttpServerAction::StreamSse {
            events,
            inter_event_delay,
        } => write_sse_response(stream, events, *inter_event_delay).await,
        HttpServerAction::Json { status, body } => write_json_response(stream, *status, body).await,
    }
}

/// Write a chunked transfer-encoded SSE response.
async fn write_sse_response(
    stream: &mut tokio::net::TcpStream,
    events: &[String],
    inter_event_delay: Duration,
) -> bool {
    let head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";
    if stream.write_all(head).await.is_err() {
        return false;
    }
    for (index, event) in events.iter().enumerate() {
        if !write_sse_chunk(stream, event).await {
            return false;
        }
        if index + 1 < events.len() && !inter_event_delay.is_zero() {
            tokio::time::sleep(inter_event_delay).await;
        }
    }
    // Terminating chunk
    if stream.write_all(b"0\r\n\r\n").await.is_err() {
        return false;
    }
    stream.flush().await.is_ok()
}

/// Write and flush one chunked SSE frame.
async fn write_sse_chunk(stream: &mut tokio::net::TcpStream, event: &str) -> bool {
    let payload = if event.ends_with("\n\n") {
        event.to_owned()
    } else {
        format!("{event}\n")
    };
    let hex_len = format!("{:x}\r\n", payload.len());
    if stream.write_all(hex_len.as_bytes()).await.is_err()
        || stream.write_all(payload.as_bytes()).await.is_err()
        || stream.write_all(b"\r\n").await.is_err()
    {
        return false;
    }
    stream.flush().await.is_ok()
}

/// Write a non-streaming JSON response with a known content length.
async fn write_json_response(stream: &mut tokio::net::TcpStream, status: u16, body: &str) -> bool {
    let reason = status_reason(status);
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    if stream.write_all(head.as_bytes()).await.is_err() {
        return false;
    }
    if stream.write_all(body.as_bytes()).await.is_err() {
        return false;
    }
    stream.flush().await.is_ok()
}

/// Map an HTTP status code to a standard reason phrase.
fn status_reason(status: u16) -> &'static str {
    match status {
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

/// Inspect an HTTP head without consuming bytes from the TCP stream.
async fn peek_http_request(stream: &tokio::net::TcpStream, timeout: Duration) -> Option<PeekedHttpRequest> {
    let head_bytes = peek_head_bytes(stream, timeout).await?;
    let head = std::str::from_utf8(&head_bytes).ok()?;
    let mut lines = head.split("\r\n");
    let (method, path, query) = parse_request_line(lines.next()?)?;
    let header_lines: Vec<&str> = lines.collect();
    let parsed_headers = parse_request_headers(&header_lines);
    let websocket_upgrade = method == "GET" && parsed_headers.connection_upgrade && parsed_headers.websocket_upgrade;

    Some(PeekedHttpRequest {
        body_bytes: parsed_headers.body_bytes,
        head_bytes: head_bytes.len(),
        headers: parsed_headers.headers,
        malformed: parsed_headers.malformed,
        method,
        path,
        query,
        upgrade_value: parsed_headers.upgrade_value,
        websocket_upgrade,
    })
}

/// Wait until the HTTP head terminator is available in the stream's peek buffer.
async fn peek_head_bytes(stream: &tokio::net::TcpStream, timeout: Duration) -> Option<Vec<u8>> {
    let mut buffer = vec![0_u8; MAX_HEAD_BYTES];
    let head_bytes = tokio::time::timeout(timeout, async {
        loop {
            let count = stream.peek(&mut buffer).await.ok()?;
            if count == 0 {
                return None;
            }
            if let Some(offset) = buffer[..count].windows(4).position(|window| window == b"\r\n\r\n") {
                return Some(offset + 4);
            }
            if count == MAX_HEAD_BYTES {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .ok()??;
    Some(buffer[..head_bytes].to_vec())
}

/// Parse the request line into method, path, and optional query string.
fn parse_request_line(line: &str) -> Option<(String, String, Option<String>)> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();
    let version = parts.next()?;
    if !version.starts_with("HTTP/") || parts.next().is_some() {
        return None;
    }
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_owned(), Some(query.to_owned())),
        None => (target, None),
    };
    Some((method, path, query))
}

/// Parse all header lines and aggregate the connection, upgrade, body length, and header map.
fn parse_request_headers(lines: &[&str]) -> ParsedHeaders {
    let mut state = HeaderParseState::default();
    for line in lines.iter().take_while(|line| !line.is_empty()) {
        apply_header_line(&mut state, line);
    }
    ParsedHeaders {
        body_bytes: state.body_bytes,
        connection_upgrade: state.connection_upgrade,
        headers: state.headers,
        malformed: state.malformed,
        upgrade_value: state.upgrade_value,
        websocket_upgrade: state.websocket_upgrade,
    }
}

/// Apply one header line to the parse state.
fn apply_header_line(state: &mut HeaderParseState, line: &str) {
    let Some((name, value)) = line.split_once(':') else {
        state.malformed = true;
        return;
    };
    let trimmed_name = name.trim();
    let trimmed_value = value.trim();
    let Ok(header_name) = http::header::HeaderName::from_bytes(trimmed_name.as_bytes()) else {
        state.malformed = true;
        return;
    };
    let Ok(header_value) = http::header::HeaderValue::from_str(trimmed_value) else {
        state.malformed = true;
        return;
    };
    state.headers.append(header_name, header_value);
    if name.eq_ignore_ascii_case("connection") {
        state.connection_upgrade |= value
            .split(',')
            .any(|token| token.trim().eq_ignore_ascii_case("upgrade"));
    } else if name.eq_ignore_ascii_case("upgrade") {
        trimmed_value.clone_into(&mut state.upgrade_value);
        state.websocket_upgrade |= state.upgrade_value.eq_ignore_ascii_case("websocket");
    } else if name.eq_ignore_ascii_case("content-length") {
        match trimmed_value.parse() {
            Ok(body_bytes) => state.body_bytes = body_bytes,
            Err(_) => state.malformed = true,
        }
    }
}

/// Consume an unexpected request so the HTTP rejection is delivered cleanly.
async fn drain_unexpected_request(
    stream: &mut tokio::net::TcpStream,
    head_bytes: usize,
    mut body_bytes: usize,
) -> bool {
    if body_bytes > MAX_REQUEST_BODY_BYTES {
        return false;
    }

    let mut head = vec![0_u8; head_bytes];
    if stream.read_exact(&mut head).await.is_err() {
        return false;
    }
    let mut buffer = vec![0_u8; 8_192];
    while body_bytes > 0 {
        let chunk = body_bytes.min(buffer.len());
        if stream.read_exact(&mut buffer[..chunk]).await.is_err() {
            return false;
        }
        body_bytes -= chunk;
    }
    true
}

/// Read a full request (head + body) and emit a captured copy.
async fn read_full_request(stream: &mut tokio::net::TcpStream, peek: PeekedHttpRequest) -> Option<CapturedHttpRequest> {
    let request_bytes = peek.head_bytes.checked_add(peek.body_bytes)?;
    let mut buffer = vec![0_u8; request_bytes];
    stream.read_exact(&mut buffer).await.ok()?;

    let method = peek.method;
    let path = peek.path;
    let query = peek.query;
    let headers = peek.headers;
    let body = Bytes::copy_from_slice(buffer.get(peek.head_bytes..)?);
    Some(CapturedHttpRequest {
        body,
        headers,
        method,
        path,
        query,
    })
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    use super::*;

    /// Capture a request body exactly once and preserve its parsed metadata.
    #[tokio::test]
    async fn scripted_http_backend_captures_complete_request() {
        let mut backend = start_scripted_http_backend(
            "POST",
            "/v1/chat/completions",
            vec![HttpServerAction::Json {
                status: 200,
                body: r#"{"ok":true}"#.to_owned(),
            }],
        )
        .await;
        let body = r#"{"model":"test-model"}"#;

        let response = send_request(&backend, "POST", "/v1/chat/completions?stream=true", body).await;

        assert!(
            response.contains("200 OK"),
            "backend should return its scripted response: {response}"
        );
        let event = next_event(&mut backend).await;
        let HttpBackendEvent::Request(request) = event else {
            panic!("expected captured request, got {event:?}");
        };
        assert_eq!(request.method, "POST", "captured method should be preserved");
        assert_eq!(
            request.path, "/v1/chat/completions",
            "captured path should exclude query"
        );
        assert_eq!(
            request.query.as_deref(),
            Some("stream=true"),
            "query should be preserved separately"
        );
        assert_eq!(request.body, Bytes::from(body), "captured body should match exactly");
    }

    /// Advance response turns globally when each request uses a new connection.
    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "complete two-connection response-turn test")]
    async fn scripted_http_backend_advances_turns_across_connections() {
        let mut backend = start_scripted_http_backend_turns(
            "POST",
            "/v1/chat/completions",
            vec![
                vec![HttpServerAction::Json {
                    status: 200,
                    body: r#"{"turn":1}"#.to_owned(),
                }],
                vec![HttpServerAction::Json {
                    status: 200,
                    body: r#"{"turn":2}"#.to_owned(),
                }],
            ],
        )
        .await;

        let first = send_request(&backend, "POST", "/v1/chat/completions", "{}").await;
        let second = send_request(&backend, "POST", "/v1/chat/completions", "{}").await;

        assert!(
            first.ends_with(r#"{"turn":1}"#),
            "first connection should receive turn one: {first}"
        );
        assert!(
            second.ends_with(r#"{"turn":2}"#),
            "second connection should receive turn two: {second}"
        );
        assert!(
            matches!(next_event(&mut backend).await, HttpBackendEvent::Request(_)),
            "first connection should emit a captured request"
        );
        assert!(
            matches!(next_event(&mut backend).await, HttpBackendEvent::Request(_)),
            "second connection should emit a captured request"
        );
    }

    /// Reject an unexpected path without consuming a scripted turn.
    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "complete rejection and turn-preservation test")]
    async fn scripted_http_backend_rejects_unexpected_request() {
        let mut backend = start_scripted_http_backend(
            "POST",
            "/v1/chat/completions",
            vec![HttpServerAction::Json {
                status: 200,
                body: r#"{"turn":1}"#.to_owned(),
            }],
        )
        .await;

        let rejected = send_request(&backend, "POST", "/v1/responses", "{}").await;
        let accepted = send_request(&backend, "POST", "/v1/chat/completions", "{}").await;

        assert!(
            rejected.contains("400 Bad Request"),
            "wrong path should be rejected: {rejected}"
        );
        assert!(
            accepted.ends_with(r#"{"turn":1}"#),
            "rejection should not consume turn one: {accepted}"
        );
        assert!(
            matches!(
                next_event(&mut backend).await,
                HttpBackendEvent::UnexpectedRequest { method, path }
                    if method == "POST" && path == "/v1/responses"
            ),
            "wrong path should emit a test-visible event"
        );
        assert!(
            matches!(next_event(&mut backend).await, HttpBackendEvent::Request(_)),
            "accepted request should emit a captured event"
        );
    }

    /// Report script exhaustion instead of silently reusing or dropping a turn.
    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "complete script-exhaustion response-boundary test")]
    async fn scripted_http_backend_reports_script_exhaustion() {
        let mut backend = start_scripted_http_backend(
            "POST",
            "/v1/chat/completions",
            vec![HttpServerAction::Json {
                status: 200,
                body: r#"{"turn":1}"#.to_owned(),
            }],
        )
        .await;

        let first = send_request(&backend, "POST", "/v1/chat/completions", "{}").await;
        let exhausted = send_request(&backend, "POST", "/v1/chat/completions", "{}").await;

        assert!(
            first.contains("200 OK"),
            "first scripted response should succeed: {first}"
        );
        assert!(
            exhausted.contains("500 Internal Server Error"),
            "exhaustion should fail: {exhausted}"
        );
        assert!(
            matches!(next_event(&mut backend).await, HttpBackendEvent::Request(_)),
            "first request should emit a captured event"
        );
        assert!(
            matches!(next_event(&mut backend).await, HttpBackendEvent::Request(_)),
            "exhausted request should still emit a captured event"
        );
        assert!(
            matches!(
                next_event(&mut backend).await,
                HttpBackendEvent::ScriptExhausted { turn: 1 }
            ),
            "exhaustion should identify the missing turn"
        );
    }

    /// Reject a declared body above the capture ceiling before allocating it.
    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "complete oversize request/response boundary test")]
    async fn scripted_http_backend_rejects_oversized_request() {
        let mut backend = start_scripted_http_backend("POST", "/v1/responses", vec![]).await;
        let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, backend.port()))
            .await
            .expect("test client should connect");
        let request = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_REQUEST_BODY_BYTES + 1
        );
        stream
            .write_all(request.as_bytes())
            .await
            .expect("request head should be written");
        stream
            .shutdown()
            .await
            .expect("oversized request body should be closed");
        let mut response = [0_u8; 256];
        let read = stream
            .read(&mut response)
            .await
            .expect("oversize rejection should start");
        assert!(
            String::from_utf8_lossy(&response[..read]).contains("413 Payload Too Large"),
            "oversized request should receive a bounded 413 response"
        );
        assert!(
            matches!(
                next_event(&mut backend).await,
                HttpBackendEvent::RequestTooLarge {
                    body_bytes,
                    max_body_bytes
                } if body_bytes == MAX_REQUEST_BODY_BYTES + 1 && max_body_bytes == MAX_REQUEST_BODY_BYTES
            ),
            "oversized request should emit its declared and accepted sizes"
        );
    }

    /// Reject a non-numeric `Content-Length` instead of accepting an empty body.
    #[tokio::test]
    async fn scripted_http_backend_rejects_invalid_content_length() {
        let backend = start_scripted_http_backend(
            "POST",
            "/v1/responses",
            vec![HttpServerAction::Json {
                status: 200,
                body: r#"{"ok":true}"#.to_owned(),
            }],
        )
        .await;
        let response = send_raw_request(
            &backend,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: nope\r\nConnection: close\r\n\r\n",
        )
        .await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "invalid Content-Length should receive 400: {response}"
        );
    }

    /// Reject a malformed header instead of silently accepting it.
    #[tokio::test]
    async fn scripted_http_backend_rejects_malformed_headers() {
        let backend = start_scripted_http_backend("POST", "/v1/responses", vec![]).await;
        let response = send_raw_request(
            &backend,
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nnot-a-header\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "malformed header should receive 400: {response}"
        );
    }

    /// Reject malformed request lines before the backend claims a scripted turn.
    #[test]
    fn scripted_http_backend_rejects_malformed_request_lines() {
        assert!(
            parse_request_line("POST").is_none(),
            "request line without a target should be rejected"
        );
        assert!(
            parse_request_line("POST /v1/responses HTTP/1.1 extra").is_none(),
            "request line with trailing fields should be rejected"
        );
    }

    /// Time out an incomplete request head without reading unbounded client data.
    #[tokio::test]
    async fn scripted_http_backend_times_out_incomplete_request_head() {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("test listener should bind");
        let client = tokio::net::TcpStream::connect(listener.local_addr().expect("listener address"))
            .await
            .expect("test client should connect");
        let (server, _) = listener.accept().await.expect("listener should accept test client");

        assert!(
            peek_head_bytes(&server, Duration::from_millis(10)).await.is_none(),
            "incomplete request head should time out"
        );
        drop(client);
    }

    /// Map configured status codes to the reason phrases emitted on the wire.
    #[test]
    fn scripted_http_backend_uses_expected_status_reasons() {
        assert_eq!(status_reason(201), "Created", "201 should use Created");
        assert_eq!(status_reason(408), "Request Timeout", "408 should use Request Timeout");
        assert_eq!(
            status_reason(503),
            "Service Unavailable",
            "503 should use Service Unavailable"
        );
        assert_eq!(status_reason(418), "OK", "unmapped statuses use the fallback reason");
    }

    /// Reject `WebSocket` upgrade attempts so the HTTP-only test boundary holds.
    #[tokio::test]
    async fn scripted_http_backend_rejects_websocket_upgrades() {
        let mut backend = start_scripted_http_backend("POST", "/v1/responses", vec![]).await;
        let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, backend.port()))
            .await
            .unwrap();
        let request =
            "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\r\n";
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .expect("HTTP rejection should complete within five seconds")
            .unwrap();
        let response_text = String::from_utf8_lossy(&response);
        assert!(
            response_text.contains("400"),
            "rejection should be 400: {response_text}"
        );

        let event = tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .expect("backend event should arrive within five seconds")
            .expect("backend event channel should remain open");
        assert!(
            matches!(event, HttpBackendEvent::WebSocketUpgrade { .. }),
            "expected WebSocketUpgrade event, got {event:?}"
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Receive one backend event within the test deadline.
    async fn next_event(backend: &mut HttpBackendGuard) -> HttpBackendEvent {
        tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .expect("backend event should arrive within five seconds")
            .expect("backend event channel should remain open")
    }

    /// Send raw bytes over a fresh connection and collect the response.
    async fn send_raw_request(backend: &HttpBackendGuard, request: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, backend.port()))
            .await
            .expect("test client should connect");
        stream
            .write_all(request.as_bytes())
            .await
            .expect("raw test request should be written");
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .expect("test response should complete within five seconds")
            .expect("test response should be readable");
        String::from_utf8(response).expect("test response should be UTF-8")
    }

    /// Send one complete HTTP request over a fresh connection and collect the response.
    async fn send_request(backend: &HttpBackendGuard, method: &str, path: &str, body: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, backend.port()))
            .await
            .expect("test client should connect");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .await
            .expect("test request should be written");
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .expect("test response should complete within five seconds")
            .expect("test response should be readable");
        String::from_utf8(response).expect("test response should be UTF-8")
    }
}
