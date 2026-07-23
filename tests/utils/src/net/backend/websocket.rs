// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Scripted `WebSocket` backend for integration testing.

use std::time::Duration;

use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use http::{HeaderMap, Method};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::TcpListener,
    sync::mpsc,
    task::JoinSet,
};
use tokio_tungstenite::{
    WebSocketStream, accept_hdr_async,
    tungstenite::{Message, protocol::CloseFrame},
};
use tracing::debug;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum HTTP request-head size accepted by the scripted backend.
const MAX_HEAD_BYTES: usize = 16_384; // 16 KiB
/// Maximum unexpected HTTP request-body size drained before rejection.
const MAX_UNEXPECTED_BODY_BYTES: usize = 16_777_216; // 16 MiB

/// A message captured from a test client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturedWsMessage {
    /// UTF-8 text message.
    Text(String),
    /// Binary message.
    Binary(Bytes),
    /// Ping control message.
    Ping(Bytes),
    /// Pong control message.
    Pong(Bytes),
    /// Close control message.
    Close {
        /// Numeric close code, when supplied.
        code: Option<u16>,
        /// Close reason, when supplied.
        reason: Option<String>,
    },
}

/// An observation emitted by the scripted backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WsBackendEvent {
    /// A completed `WebSocket` HTTP handshake.
    Handshake {
        /// Request headers observed by the backend.
        headers: HeaderMap,
        /// Request method observed by the backend.
        method: Method,
        /// Path and query observed by the backend.
        path: String,
    },
    /// A message received from the connected client.
    ClientMessage(CapturedWsMessage),
    /// A non-upgrade HTTP request sent to the backend listener.
    UnexpectedHttpRequest {
        /// Request method parsed from the request line.
        method: String,
        /// Request target parsed from the request line.
        path: String,
    },
}

/// A deterministic action performed after the first client data message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WsServerAction {
    /// Send a binary message.
    Binary(Bytes),
    /// Send a UTF-8 text message.
    Text(String),
    /// Send a ping control message.
    Ping(Bytes),
    /// Wait while continuing to service client control frames.
    Delay(Duration),
    /// Send a close control message and end the connection.
    Close {
        /// Numeric close code.
        code: u16,
        /// Human-readable close reason.
        reason: String,
    },
    /// Drop the TCP connection without a `WebSocket` close frame.
    Disconnect,
}

/// RAII guard for a scripted asynchronous `WebSocket` backend.
///
/// Dropping the guard stops the listener and aborts all connection tasks.
pub struct WsBackendGuard {
    /// Stream of backend observations.
    events: mpsc::Receiver<WsBackendEvent>,
    /// Listener task, which owns all connection tasks.
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Port allocated by the operating system.
    port: u16,
    /// Listener shutdown signal.
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl WsBackendGuard {
    /// Wait for the next handshake or client-message observation.
    pub async fn next_event(&mut self) -> Option<WsBackendEvent> {
        self.events.recv().await
    }

    /// Return the allocated port.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for WsBackendGuard {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _sent = shutdown.send(());
        }
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// Start a `WebSocket` backend that replays `script` after the first
/// client text or binary message.
///
/// Each accepted connection receives an independent copy of the script.
/// The returned guard exposes handshake and client-message observations.
///
/// # Panics
///
/// Panics if the loopback listener cannot bind or report its local address.
pub async fn start_scripted_websocket_backend(script: Vec<WsServerAction>) -> WsBackendGuard {
    start_scripted_websocket_backend_turns(vec![script]).await
}

/// Start a backend that replays one action sequence for each sequential client
/// data message on a connection.
///
/// This supports protocols that perform a prewarm request before the real
/// turn. A turn begins after the previous action sequence completes. Data
/// messages received during a scripted delay are captured but do not start a
/// nested turn. Each accepted connection receives an independent copy of
/// `turns`.
///
/// # Panics
///
/// Panics if the loopback listener cannot bind or report its local address.
pub async fn start_scripted_websocket_backend_turns(turns: Vec<Vec<WsServerAction>>) -> WsBackendGuard {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("scripted WebSocket backend should bind");
    let port = listener
        .local_addr()
        .expect("scripted WebSocket backend should have a local address")
        .port();
    let (event_tx, event_rx) = mpsc::channel(256);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    let handle = tokio::spawn(run_listener(listener, turns, event_tx, shutdown_rx));
    debug!(port, "scripted WebSocket backend listening");

    WsBackendGuard {
        events: event_rx,
        handle: Some(handle),
        port,
        shutdown: Some(shutdown_tx),
    }
}

/// Accept connections until the guard signals shutdown.
async fn run_listener(
    listener: TcpListener,
    turns: Vec<Vec<WsServerAction>>,
    event_tx: mpsc::Sender<WsBackendEvent>,
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
                debug!(%peer, "scripted WebSocket backend accepted connection");
                connections.spawn(Box::pin(handle_connection(stream, turns.clone(), event_tx.clone())));
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

/// Complete a handshake, capture client messages, and run one script.
async fn handle_connection(
    stream: tokio::net::TcpStream,
    turns: Vec<Vec<WsServerAction>>,
    event_tx: mpsc::Sender<WsBackendEvent>,
) {
    let Some(mut socket) = Box::pin(accept_connection(stream, &event_tx)).await else {
        return;
    };
    let mut next_turn = 0;

    while let Some(result) = socket.next().await {
        let Ok(message) = result else {
            return;
        };
        let starts_script = matches!(message, Message::Text(_) | Message::Binary(_));
        if !capture_and_service_message(&mut socket, &event_tx, message).await {
            return;
        }
        if starts_script && let Some(script) = turns.get(next_turn) {
            next_turn += 1;
            if !run_script(&mut socket, &event_tx, script).await {
                return;
            }
        }
    }
}

/// Validate the HTTP request and complete a `WebSocket` handshake.
#[expect(
    clippy::too_many_lines,
    reason = "preflight rejection and handshake capture form one ownership boundary"
)]
async fn accept_connection(
    mut stream: tokio::net::TcpStream,
    event_tx: &mpsc::Sender<WsBackendEvent>,
) -> Option<WebSocketStream<tokio::net::TcpStream>> {
    let request = Box::pin(peek_http_request(&stream)).await?;
    if !request.websocket_upgrade {
        let _queued = event_tx.try_send(WsBackendEvent::UnexpectedHttpRequest {
            method: request.method,
            path: request.path,
        });
        if !Box::pin(drain_unexpected_request(
            &mut stream,
            request.head_bytes,
            request.body_bytes,
        ))
        .await
        {
            return None;
        }
        let _written = stream
            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .await;
        return None;
    }

    let handshake_events = event_tx.clone();
    #[expect(
        clippy::result_large_err,
        reason = "Tungstenite's callback requires its HTTP error response type"
    )]
    let callback = move |request: &http::Request<()>, response| {
        // The callback borrows its request. Owning the header map is required
        // because test assertions run after the handshake completes.
        let event = WsBackendEvent::Handshake {
            headers: request.headers().clone(),
            method: request.method().clone(),
            path: request.uri().to_string(),
        };
        let _queued = handshake_events.try_send(event);
        Ok(response)
    };
    accept_hdr_async(stream, callback).await.ok()
}

/// Parsed facts needed before handing the stream to Tungstenite.
struct PeekedHttpRequest {
    /// Declared request body length.
    body_bytes: usize,
    /// Number of bytes through the terminating blank header line.
    head_bytes: usize,
    /// Request method.
    method: String,
    /// Request target.
    path: String,
    /// Whether the request declares a `WebSocket` upgrade.
    websocket_upgrade: bool,
}

/// Inspect an HTTP head without consuming bytes from the TCP stream.
#[expect(
    clippy::too_many_lines,
    reason = "the bounded parser keeps all unexpected-request checks together"
)]
async fn peek_http_request(stream: &tokio::net::TcpStream) -> Option<PeekedHttpRequest> {
    let mut buffer = vec![0_u8; MAX_HEAD_BYTES];
    let head_bytes = tokio::time::timeout(Duration::from_secs(5), async {
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
    let head = std::str::from_utf8(&buffer[..head_bytes]).ok()?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next()?.split_whitespace();
    let method = request_line.next()?.to_owned();
    let path = request_line.next()?.to_owned();
    let mut connection_upgrade = false;
    let mut websocket_upgrade = false;
    let mut body_bytes = 0;

    for line in lines.take_while(|line| !line.is_empty()) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("connection") {
            connection_upgrade |= value
                .split(',')
                .any(|token| token.trim().eq_ignore_ascii_case("upgrade"));
        } else if name.eq_ignore_ascii_case("upgrade") {
            websocket_upgrade |= value.trim().eq_ignore_ascii_case("websocket");
        } else if name.eq_ignore_ascii_case("content-length") {
            body_bytes = value.trim().parse().ok()?;
        }
    }
    let websocket_upgrade = method == "GET" && connection_upgrade && websocket_upgrade;

    Some(PeekedHttpRequest {
        body_bytes,
        head_bytes,
        method,
        path,
        websocket_upgrade,
    })
}

/// Consume an unexpected request so the HTTP rejection is delivered cleanly.
async fn drain_unexpected_request(
    stream: &mut tokio::net::TcpStream,
    head_bytes: usize,
    mut body_bytes: usize,
) -> bool {
    if body_bytes > MAX_UNEXPECTED_BODY_BYTES {
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

/// Execute scripted actions, servicing control frames during delays.
#[expect(
    clippy::too_many_lines,
    reason = "each explicit script action has distinct protocol behavior"
)]
async fn run_script(
    socket: &mut WebSocketStream<tokio::net::TcpStream>,
    event_tx: &mpsc::Sender<WsBackendEvent>,
    script: &[WsServerAction],
) -> bool {
    for action in script {
        match action {
            WsServerAction::Binary(data) => {
                if socket.send(Message::Binary(data.clone())).await.is_err() {
                    return false;
                }
            },
            WsServerAction::Text(text) => {
                if socket.send(Message::Text(text.as_str().into())).await.is_err() {
                    return false;
                }
            },
            WsServerAction::Ping(payload) => {
                if socket.send(Message::Ping(payload.clone())).await.is_err() {
                    return false;
                }
            },
            WsServerAction::Delay(duration) => {
                if !delay_while_servicing(socket, event_tx, *duration).await {
                    return false;
                }
            },
            WsServerAction::Close { code, reason } => {
                let frame = CloseFrame {
                    code: (*code).into(),
                    reason: reason.as_str().into(),
                };
                let _sent = socket.send(Message::Close(Some(frame))).await;
                return false;
            },
            WsServerAction::Disconnect => return false,
        }
    }
    true
}

/// Wait for a scripted duration without starving client control frames.
async fn delay_while_servicing(
    socket: &mut WebSocketStream<tokio::net::TcpStream>,
    event_tx: &mpsc::Sender<WsBackendEvent>,
    duration: Duration,
) -> bool {
    let delay = tokio::time::sleep(duration);
    tokio::pin!(delay);

    loop {
        tokio::select! {
            () = &mut delay => return true,
            incoming = socket.next() => {
                let Some(Ok(message)) = incoming else {
                    return false;
                };
                if !capture_and_service_message(socket, event_tx, message).await {
                    return false;
                }
            },
        }
    }
}

/// Capture one client message and flush automatic control replies.
async fn capture_and_service_message(
    socket: &mut WebSocketStream<tokio::net::TcpStream>,
    event_tx: &mpsc::Sender<WsBackendEvent>,
    message: Message,
) -> bool {
    let (captured, is_control, is_close) = match message {
        Message::Text(text) => (CapturedWsMessage::Text(text.to_string()), false, false),
        Message::Binary(data) => (CapturedWsMessage::Binary(data), false, false),
        Message::Ping(data) => (CapturedWsMessage::Ping(data), true, false),
        Message::Pong(data) => (CapturedWsMessage::Pong(data), false, false),
        Message::Close(frame) => {
            let (code, reason) = frame.map_or((None, None), |frame| {
                (Some(frame.code.into()), Some(frame.reason.to_string()))
            });
            (CapturedWsMessage::Close { code, reason }, true, true)
        },
        Message::Frame(_) => return true,
    };
    let _queued = event_tx.try_send(WsBackendEvent::ClientMessage(captured));

    if is_control && socket.flush().await.is_err() {
        return false;
    }
    !is_close
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use std::time::Duration;

    use futures::{SinkExt as _, StreamExt as _};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    use super::*;

    /// Capture the opening handshake and the first client data message.
    #[tokio::test]
    async fn scripted_backend_captures_handshake_and_client_text() {
        let mut backend = start_scripted_websocket_backend(vec![
            WsServerAction::Text(r#"{"type":"response.created"}"#.to_owned()),
            WsServerAction::Close {
                code: 1000,
                reason: "complete".to_owned(),
            },
        ])
        .await;

        let url = format!("ws://127.0.0.1:{}/v1/responses?model=gpt-5", backend.port());
        let (mut socket, response) = connect_async(&url).await.unwrap();
        assert_eq!(
            response.status(),
            http::StatusCode::SWITCHING_PROTOCOLS,
            "the scripted backend should accept a valid opening handshake"
        );

        socket
            .send(Message::Text(r#"{"type":"response.create"}"#.into()))
            .await
            .unwrap();
        assert_eq!(
            next_message(&mut socket).await.into_text().unwrap(),
            r#"{"type":"response.created"}"#,
            "the backend should replay its scripted text frame"
        );
        assert!(
            next_message(&mut socket).await.is_close(),
            "the backend should replay its scripted close frame"
        );

        let handshake = next_event(&mut backend).await;
        let WsBackendEvent::Handshake { headers, method, path } = handshake else {
            panic!("expected handshake event, got {handshake:?}");
        };
        assert_eq!(method, Method::GET, "the backend should observe the GET method");
        assert_eq!(
            path, "/v1/responses?model=gpt-5",
            "the backend should preserve the request target"
        );
        assert_eq!(
            headers.get(http::header::UPGRADE).unwrap(),
            "websocket",
            "the backend should observe the WebSocket upgrade protocol"
        );
        assert_eq!(
            next_event(&mut backend).await,
            WsBackendEvent::ClientMessage(CapturedWsMessage::Text(r#"{"type":"response.create"}"#.to_owned())),
            "the backend should capture the client text frame"
        );
    }

    /// Service client control traffic while a scripted turn is delayed.
    #[tokio::test]
    async fn scripted_backend_services_ping_during_delay() {
        let backend = start_scripted_websocket_backend(vec![
            WsServerAction::Delay(Duration::from_millis(50)),
            WsServerAction::Text("ready".to_owned()),
        ])
        .await;
        let url = format!("ws://127.0.0.1:{}/v1/responses", backend.port());
        let (mut socket, _) = connect_async(&url).await.unwrap();

        socket.send(Message::Text("start".into())).await.unwrap();
        socket.send(Message::Ping(vec![1, 2, 3].into())).await.unwrap();

        assert_eq!(
            next_message(&mut socket).await,
            Message::Pong(vec![1, 2, 3].into()),
            "the backend should answer a ping during a delay"
        );
        assert_eq!(
            next_message(&mut socket).await.into_text().unwrap(),
            "ready",
            "the script should resume after the delay"
        );
    }

    /// Reject non-upgrade HTTP traffic without leaving the client hanging.
    #[tokio::test]
    async fn scripted_backend_records_and_rejects_unexpected_http_request() {
        let mut backend = start_scripted_websocket_backend(Vec::new()).await;
        let mut stream = tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, backend.port()))
            .await
            .unwrap();

        stream
            .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .expect("HTTP rejection should complete within five seconds")
            .unwrap();

        assert!(
            String::from_utf8(response).unwrap().starts_with("HTTP/1.1 400"),
            "the backend should return a bounded HTTP rejection"
        );
        assert_eq!(
            next_event(&mut backend).await,
            WsBackendEvent::UnexpectedHttpRequest {
                method: "POST".to_owned(),
                path: "/v1/responses".to_owned(),
            },
            "the backend should report the rejected method and path"
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Receive one observation without allowing a broken listener to hang.
    async fn next_event(backend: &mut WsBackendGuard) -> WsBackendEvent {
        tokio::time::timeout(Duration::from_secs(5), backend.next_event())
            .await
            .expect("backend observation should arrive within five seconds")
            .expect("backend observation channel should remain open")
    }

    /// Receive one message without allowing a broken test backend to hang.
    async fn next_message<S>(socket: &mut WebSocketStream<S>) -> Message
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("WebSocket receive should complete within five seconds")
            .expect("WebSocket stream should remain open")
            .expect("WebSocket message should be valid")
    }
}
