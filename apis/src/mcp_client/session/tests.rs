// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the MCP session lifecycle client.
//!
//! The scripted backend speaks raw HTTP/1.1 so each test controls framing,
//! status, headers, and whether an SSE response stays open.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use http::{HeaderMap, HeaderValue};
use praxis_core::connectivity::{Upstream, UrlResolutionPolicy, prepare_url_target_with_policy};
use praxis_filter::{RequestExtensions, StagedUpstream, SubrequestRuntime};
use rmcp::model::{ErrorCode, ErrorData, JsonRpcError, JsonRpcVersion2_0};
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::TcpListener,
};

use super::*;
use crate::mcp_client::subrequest_transport::build_bare_outbound_pipeline;

// -----------------------------------------------------------------------------
// Scripted Backend
// -----------------------------------------------------------------------------

/// One request the scripted backend received.
#[derive(Clone, Debug)]
struct Received {
    /// HTTP method.
    method: String,
    /// Lower-cased header names and their values, in order.
    headers: Vec<(String, String)>,
    /// Request body.
    body: String,
}

impl Received {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// A scripted reply: raw bytes written after the request is read, then the
/// connection is held open for `hold` before it closes.
#[derive(Clone)]
struct Reply {
    bytes: Vec<u8>,
    hold: Duration,
}

impl Reply {
    fn closed(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
            hold: Duration::ZERO,
        }
    }
}

/// Start a backend answering connection `n` with `replies[n]` (the last reply
/// repeats) and recording every request.
async fn scripted_backend(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<Received>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        let mut served = 0_usize;
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let reply = replies[served.min(replies.len() - 1)].clone();
            served += 1;
            let log = Arc::clone(&log);
            tokio::spawn(async move {
                let Some(request) = read_request(&mut stream).await else {
                    return;
                };
                log.lock().unwrap().push(request);
                if stream.write_all(&reply.bytes).await.is_err() {
                    return;
                }
                let _flushed = stream.flush().await;
                tokio::time::sleep(reply.hold).await;
            });
        }
    });
    (format!("http://{addr}/mcp"), received)
}

/// Read one HTTP/1.1 request with a `Content-Length` body.
async fn read_request(stream: &mut tokio::net::TcpStream) -> Option<Received> {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    let head_end = loop {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..read]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let method = lines.next()?.split(' ').next()?.to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect();
    let length = headers
        .iter()
        .find(|(name, _)| name == "content-length")
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some(Received {
        method,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

/// A complete HTTP/1.1 response with `Content-Length`.
fn http_response(status: &str, headers: &[(&str, &str)], body: &str) -> Vec<u8> {
    let mut out = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
    // RFC 9110 §8.6: a server MUST NOT send Content-Length in a 204 response.
    if !status.starts_with("204") {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    for (name, value) in headers {
        out.push_str(&format!("{name}: {value}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(body);
    out.into_bytes()
}

/// An `initialize` result for the id the backend received, per MCP.
fn init_result(request: &str, version: &str) -> String {
    let id = serde_json::from_str::<serde_json::Value>(request).unwrap()["id"].clone();
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": version,
            "capabilities": {},
            "serverInfo": { "name": "scripted", "version": "1" },
        },
    })
    .to_string()
}

// -----------------------------------------------------------------------------
// Client Under Test
// -----------------------------------------------------------------------------

/// Stages the scripted backend as the dial target, as a chain's cluster
/// selector would.
struct StagedBackend(Upstream);

impl McpExchangeStage for StagedBackend {
    fn stage(&self, extensions: &mut RequestExtensions) {
        extensions.insert(StagedUpstream(self.0.clone()));
    }

    fn follow_up(&self) -> Arc<dyn McpExchangeStage> {
        Arc::new(Self(self.0.clone()))
    }
}

async fn staged(url: &str) -> Arc<dyn McpExchangeStage> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let target = prepare_url_target_with_policy(url, deadline, |_addrs| Ok(()), UrlResolutionPolicy::ClientPerCall)
        .await
        .unwrap();
    Arc::new(StagedBackend(StagedUpstream::from_prepared_target(&target).unwrap().0))
}

/// An outbound chain whose first filter selects streamed replies, as the
/// broker's backend selector does.
fn streaming_pipeline() -> Arc<FilterPipeline> {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_filter::register_filters!(
        @register registry,
        http "openai_mcp_streaming_selector" => crate::McpStreamingSelectorFilter::from_config
    );
    let mut entries: Vec<praxis_filter::FilterEntry> =
        serde_yaml::from_str("- filter: openai_mcp_streaming_selector\n").unwrap();
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
    pipeline.set_allow_private_upstreams(true);
    Arc::new(pipeline)
}

fn session_client(max_response_bytes: usize) -> McpSessionClient {
    McpSessionClient::new(streaming_pipeline(), max_response_bytes, Duration::from_secs(5))
}

#[tokio::test]
async fn buffering_chain_still_reads_a_closed_sse_reply() {
    let (url, received) = scripted_backend(vec![Reply::closed(http_response("202 Accepted", &[], ""))]).await;
    let client = McpSessionClient::new(
        build_bare_outbound_pipeline(true).unwrap(),
        1 << 20,
        Duration::from_secs(5),
    );

    client
        .send_initialized(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-1")),
        )
        .await
        .unwrap();

    assert_eq!(
        received.lock().unwrap().len(),
        1,
        "a chain without streaming still works"
    );
}

fn callout(timeout: Duration) -> McpSessionCallout {
    let mut forwarded_headers = HeaderMap::new();
    forwarded_headers.append("authorization", HeaderValue::from_static("Bearer first"));
    forwarded_headers.append("x-tenant", HeaderValue::from_static("a"));
    forwarded_headers.append("x-tenant", HeaderValue::from_static("b"));
    McpSessionCallout {
        client: crate::subrequest::isolated_client(4),
        deadline: Instant::now() + timeout,
        depth: 0,
        downstream: SubrequestRuntime::new(None, false, None, Instant::now()),
        forwarded_headers,
        trace_context: None,
    }
}

fn session(id: Option<&'static str>) -> McpSessionInit {
    McpSessionInit {
        protocol_version: Box::from("2025-06-18"),
        session_id: id.map(HeaderValue::from_static),
    }
}

/// Answer `initialize` from the request the backend actually received.
async fn initialize_with(
    reply: impl Fn(&str) -> Reply + Send + Sync + 'static,
    max_response_bytes: usize,
    timeout: Duration,
) -> (Result<McpSessionInit, McpSessionError>, Vec<Received>) {
    initialize_via(session_client(max_response_bytes), reply, timeout).await
}

/// A client over a chain that never selects streaming, as when an operator
/// filter buffers response bodies.
fn buffering_client(max_response_bytes: usize) -> McpSessionClient {
    McpSessionClient::new(
        build_bare_outbound_pipeline(true).unwrap(),
        max_response_bytes,
        Duration::from_secs(5),
    )
}

/// Run `initialize` with `client` against a backend answering with `reply`.
async fn initialize_via(
    client: McpSessionClient,
    reply: impl Fn(&str) -> Reply + Send + Sync + 'static,
    timeout: Duration,
) -> (Result<McpSessionInit, McpSessionError>, Vec<Received>) {
    // The reply must echo the random request id, so the backend answers in
    // two phases: record the request, then build the reply from it.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return;
        };
        let Some(request) = read_request(&mut stream).await else {
            return;
        };
        let reply = reply(&request.body);
        log.lock().unwrap().push(request);
        let _written = stream.write_all(&reply.bytes).await;
        let _flushed = stream.flush().await;
        tokio::time::sleep(reply.hold).await;
    });
    let url = format!("http://{addr}/mcp");
    let result = client
        .initialize("/mcp", staged(&url).await, &callout(timeout), "2025-06-18")
        .await
        .map_err(|failure| failure.error);
    let received = received.lock().unwrap().clone();
    (result, received)
}

// -----------------------------------------------------------------------------
// initialize
// -----------------------------------------------------------------------------

#[tokio::test]
async fn json_initialize_captures_session_and_version() {
    let (result, received) = initialize_with(
        |body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json"), ("Mcp-Session-Id", "s-1")],
                &init_result(body, "2025-06-18"),
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    let init = result.unwrap();
    assert_eq!(init.session_id.unwrap(), "s-1", "session id must be captured");
    assert_eq!(&*init.protocol_version, "2025-06-18", "version must be captured");

    let request = &received[0];
    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(body["method"], "initialize", "method must be initialize");
    assert_eq!(
        body["params"]["protocolVersion"], "2025-06-18",
        "version must be offered"
    );
    assert_eq!(
        body["params"]["clientInfo"]["name"], CLIENT_NAME,
        "client name must be sent"
    );
    assert!(
        body["id"].as_str().unwrap().starts_with(INITIALIZE_REQUEST_ID_PREFIX),
        "the request id must carry the prefix: {body}"
    );
    assert_eq!(
        request.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
    let tenants: Vec<&str> = request
        .headers
        .iter()
        .filter(|(name, _)| name == "x-tenant")
        .map(|(_, value)| value.as_str())
        .collect();
    assert_eq!(tenants, vec!["a", "b"], "every forwarded value must be sent");
    assert_eq!(
        request.header("accept"),
        Some("text/event-stream, application/json"),
        "the client must accept both reply forms"
    );
    assert!(
        request.header("mcp-session-id").is_none(),
        "initialize carries no session"
    );
}

#[tokio::test]
async fn sessionless_backend_is_accepted() {
    let (result, _) = initialize_with(
        |body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json")],
                &init_result(body, "2025-03-26"),
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        result.unwrap().session_id.is_none(),
        "backend without a session yields None"
    );
}

#[tokio::test]
async fn sse_initialize_completes_while_stream_stays_open() {
    // MCP Streamable HTTP lets a server keep the SSE response open after its
    // JSON-RPC response; the handshake must not wait for the stream to close.
    let (result, _) = initialize_with(
        |body| {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nMcp-Session-Id: s-2\r\n\
                        Transfer-Encoding: chunked\r\n\r\n";
            let events = format!(
                ": keepalive\n\nevent: message\ndata: {{\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{{\"level\":\"info\",\"data\":\"x\"}}}}\n\ndata: {}\n\n",
                init_result(body, "2025-06-18")
            );
            let chunk = format!("{:x}\r\n{events}\r\n", events.len());
            Reply {
                bytes: [head.as_bytes(), chunk.as_bytes()].concat(),
                hold: Duration::from_secs(30),
            }
        },
        1 << 20,
        Duration::from_secs(3),
    )
    .await;

    let init = result.unwrap();
    assert_eq!(init.session_id.unwrap(), "s-2", "SSE session must be captured");
    assert_eq!(&*init.protocol_version, "2025-06-18", "SSE reply must be found");
}

#[tokio::test]
async fn sse_initialize_without_reply_is_malformed() {
    let (result, _) = initialize_with(
        |_body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "text/event-stream")],
                "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n\n",
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::Malformed(_))),
        "a stream that ends without the reply must be malformed: {result:?}"
    );
}

#[tokio::test]
async fn sse_reply_without_terminating_blank_line_is_discarded() {
    // SSE dispatches an event only at a blank line; one cut off by EOF is not
    // an event.
    let (result, _) = initialize_with(
        |body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "text/event-stream")],
                &format!("data: {}", init_result(body, "2025-06-18")),
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::Malformed(_))),
        "an unterminated event must not complete the handshake: {result:?}"
    );
}

#[tokio::test]
async fn sse_reply_accepts_cr_framing_and_bom() {
    for framing in ["\r", "\r\n"] {
        let (result, _) = initialize_with(
            move |body| {
                let reply = init_result(body, "2025-06-18");
                Reply::closed(http_response(
                    "200 OK",
                    &[("Content-Type", "text/event-stream")],
                    &format!("\u{feff}event: message{framing}data: {reply}{framing}{framing}"),
                ))
            },
            1 << 20,
            Duration::from_secs(5),
        )
        .await;
        assert!(result.is_ok(), "{framing:?} framing with a BOM must parse: {result:?}");
    }
}

/// An SSE `initialize` reply preceded by a field SSE requires clients to
/// ignore, a colonless field, and a notification carrying `marker`.
fn sse_with_extras(body: &str, marker: &str) -> Reply {
    let notification = format!(
        r#"{{"jsonrpc":"2.0","method":"notifications/message","params":{{"level":"info","data":"{marker}"}}}}"#
    );
    let events = format!(
        "x-vendor: 1\nretry\nevent: message\ndata: {notification}\n\nid: 7\nx-vendor: 2\ndata: {}\n\n",
        init_result(body, "2025-06-18")
    );
    Reply::closed(http_response(
        "200 OK",
        &[("Content-Type", "text/event-stream")],
        &events,
    ))
}

#[tokio::test]
async fn sse_reply_ignores_unknown_and_colonless_fields() {
    let (result, _) = initialize_with(|body| sse_with_extras(body, "x"), 1 << 20, Duration::from_secs(5)).await;

    assert_eq!(
        result.unwrap().protocol_version.as_ref(),
        "2025-06-18",
        "SSE requires unknown fields to be ignored"
    );
}

#[tokio::test]
async fn buffered_sse_reply_ignores_unknown_fields() {
    let (result, _) = initialize_via(
        buffering_client(1 << 20),
        |body| sse_with_extras(body, "x"),
        Duration::from_secs(5),
    )
    .await;

    assert!(
        result.is_ok(),
        "a buffered SSE reply must parse the same way: {result:?}"
    );
}

#[tokio::test]
async fn buffered_sse_reply_discards_unterminated_event() {
    let (result, _) = initialize_via(
        buffering_client(1 << 20),
        |body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "text/event-stream")],
                &format!("data: {}", init_result(body, "2025-06-18")),
            ))
        },
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::Malformed(_))),
        "a buffered event cut off at end of stream is not an event: {result:?}"
    );
}

#[tokio::test]
async fn buffered_reply_over_the_limit_is_too_large() {
    // Between the limit and the executor's doubled streaming backstop.
    let (result, _) = initialize_via(
        buffering_client(16 * 1024),
        |body| {
            let mut reply: serde_json::Value = serde_json::from_str(&init_result(body, "2025-06-18")).unwrap();
            reply["result"]["serverInfo"]["padding"] = serde_json::json!("x".repeat(20 * 1024));
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json")],
                &reply.to_string(),
            ))
        },
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::ResponseTooLarge { limit: 16 * 1024 }),
        "a buffered reply must respect the exact limit"
    );
}

#[tokio::test]
async fn truncated_json_reply_is_a_transport_failure() {
    let (result, _) = initialize_with(
        |_body| {
            Reply::closed(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n{\"jsonrpc\""
                    .to_vec(),
            )
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::Transport),
        "a cut-off body is not an oversized one"
    );
}

#[test]
fn notification_content_before_the_reply_is_not_logged() {
    const MARKER: &str = "secret-notification-marker";
    let mut result = None;
    let logs = crate::test_utils::capture_log_output(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        result = Some(runtime.block_on(initialize_with(
            |body| sse_with_extras(body, MARKER),
            1 << 20,
            Duration::from_secs(5),
        )));
    });

    assert!(result.unwrap().0.is_ok(), "the handshake must complete");
    assert!(
        !logs.contains(MARKER),
        "notification content must not reach logs: {logs}"
    );
}

/// An SSE reply followed by a large keepalive comment, written as one HTTP
/// chunk or as two chunks a moment apart; the stream stays open.
fn reply_then_keepalive(body: &str, split: bool) -> Reply {
    let event = format!("data: {}\n\n", init_result(body, "2025-06-18"));
    let keepalive = format!(": {}\n\n", "k".repeat(8 * 1024));
    let chunk = |payload: &str| format!("{:x}\r\n{payload}\r\n", payload.len());
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
    let bytes = if split {
        [head, &chunk(&event)].concat()
    } else {
        [head, &chunk(&format!("{event}{keepalive}"))].concat()
    };
    Reply {
        bytes: bytes.into_bytes(),
        hold: Duration::from_secs(30),
    }
}

#[tokio::test]
async fn sse_limit_does_not_depend_on_chunking() {
    // The reply fits in 4 KiB; the keepalive after it does not.
    let mut results = Vec::new();
    for split in [false, true] {
        let (result, _) = initialize_with(
            move |body| reply_then_keepalive(body, split),
            4 * 1024,
            Duration::from_secs(3),
        )
        .await;
        results.push(result.map(|init| init.protocol_version));
    }

    assert_eq!(
        results[0], results[1],
        "the same bytes must give the same result in one chunk or two"
    );
    assert!(
        results[0].is_ok(),
        "a reply within the limit must complete: {results:?}"
    );
}

#[tokio::test]
async fn reply_past_the_limit_before_the_reply_event_is_too_large() {
    let (result, _) = initialize_with(
        |body| {
            let events = format!(
                ": {}\n\ndata: {}\n\n",
                "k".repeat(8 * 1024),
                init_result(body, "2025-06-18")
            );
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "text/event-stream")],
                &events,
            ))
        },
        4 * 1024,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::ResponseTooLarge { limit: 4 * 1024 }),
        "bytes before the reply count toward the limit"
    );
}

#[tokio::test]
async fn stall_at_a_shorter_step_timeout_is_a_timeout() {
    let client = McpSessionClient::new(streaming_pipeline(), 1 << 20, Duration::from_millis(300));
    let (result, _) = initialize_via(
        client,
        |_body| Reply {
            bytes: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec(),
            hold: Duration::from_secs(30),
        },
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::Timeout),
        "the step timeout is the exchange deadline when it is earlier"
    );
}

#[tokio::test]
async fn open_stream_without_reply_times_out() {
    let (result, _) = initialize_with(
        |_body| Reply {
            bytes: b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec(),
            hold: Duration::from_secs(30),
        },
        1 << 20,
        Duration::from_millis(300),
    )
    .await;

    assert_eq!(result, Err(McpSessionError::Timeout), "the deadline must still apply");
}

#[tokio::test]
async fn oversized_initialize_is_too_large() {
    let (result, _) = initialize_with(
        |body| {
            let mut reply: serde_json::Value = serde_json::from_str(&init_result(body, "2025-06-18")).unwrap();
            reply["result"]["serverInfo"]["padding"] = serde_json::json!("x".repeat(64 * 1024));
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json")],
                &reply.to_string(),
            ))
        },
        16 * 1024,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::ResponseTooLarge { .. })),
        "an oversized reply must be classified: {result:?}"
    );
}

#[tokio::test]
async fn oversized_sse_initialize_is_too_large() {
    let (result, _) = initialize_with(
        |_body| {
            let event = format!("data: {{\"padding\":\"{}\"}}\n\n", "x".repeat(64 * 1024));
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "text/event-stream")],
                &event,
            ))
        },
        16 * 1024,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::ResponseTooLarge { .. })),
        "an oversized SSE event must be classified: {result:?}"
    );
}

#[tokio::test]
async fn non_success_status_is_rejected() {
    let (result, _) = initialize_with(
        |_body| {
            Reply::closed(http_response(
                "401 Unauthorized",
                &[("Content-Type", "application/json"), ("WWW-Authenticate", "Bearer")],
                "{}",
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::Rejected { status: 401 }),
        "401 must map to Rejected"
    );
}

#[tokio::test]
async fn json_rpc_error_code_is_preserved() {
    let (result, _) = initialize_with(
        |body| {
            let id = serde_json::from_str::<serde_json::Value>(body).unwrap()["id"].clone();
            let reply = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32602, "message": "secret detail" },
            });
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json")],
                &reply.to_string(),
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::JsonRpc { code: -32602 }),
        "JSON-RPC error code must be preserved"
    );
}

#[tokio::test]
async fn non_json_reply_is_malformed() {
    let (result, _) = initialize_with(
        |_body| {
            Reply::closed(http_response(
                "200 OK",
                &[("Content-Type", "application/json")],
                "not json",
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(result, Err(McpSessionError::Malformed(_))),
        "malformed JSON must be classified: {result:?}"
    );
}

#[tokio::test]
async fn duplicate_session_headers_are_rejected() {
    let (result, _) = initialize_with(
        |body| {
            Reply::closed(http_response(
                "200 OK",
                &[
                    ("Content-Type", "application/json"),
                    ("Mcp-Session-Id", "a"),
                    ("Mcp-Session-Id", "b"),
                ],
                &init_result(body, "2025-06-18"),
            ))
        },
        1 << 20,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        Err(McpSessionError::Malformed("multiple Mcp-Session-Id headers")),
        "an ambiguous session id must be rejected"
    );
}

#[tokio::test]
async fn unreachable_backend_is_a_transport_failure() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let stage = staged(&url).await;
    drop(listener);

    let result = session_client(1 << 20)
        .initialize("/mcp", stage, &callout(Duration::from_secs(5)), "2025-06-18")
        .await;

    // The executor answers a refused dial with a synthesized 502.
    assert_eq!(
        result,
        Err(McpInitializeError::from(McpSessionError::Rejected { status: 502 })),
        "a refused dial must fail without a session"
    );
}

// -----------------------------------------------------------------------------
// Server Requests During initialize
// -----------------------------------------------------------------------------

/// How the ping backend answers each POST that is not `initialize`.
#[derive(Clone, Copy)]
enum AnswerReply {
    /// `202 Accepted`, as MCP requires for an accepted response.
    Accepted,
    /// A complete response with this status line.
    Status(&'static str),
    /// `202 Accepted` after this delay.
    After(Duration),
    /// No response; the connection stays open.
    Never,
}

/// A backend whose `initialize` stream carries server requests.
#[derive(Clone)]
struct PingScript {
    /// Server request events written before the `initialize` result.
    requests: Vec<String>,
    /// Hold the result until every request was answered.
    wait_for_answers: bool,
    /// How the answer POSTs are answered.
    answer: AnswerReply,
    /// End the stream after the result instead of holding it open.
    close_after_result: bool,
}

impl PingScript {
    fn waiting(requests: Vec<String>) -> Self {
        Self {
            requests,
            wait_for_answers: true,
            answer: AnswerReply::Accepted,
            close_after_result: false,
        }
    }
}

/// A `ping` request with `id`.
fn ping(id: &serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": "ping" }).to_string()
}

/// One HTTP/1.1 chunk carrying one SSE event with `data`.
fn sse_chunk(data: &str) -> String {
    let event = format!("data: {data}\n\n");
    format!("{:x}\r\n{event}\r\n", event.len())
}

/// Start a backend that serves `initialize` on an SSE stream with `script`'s
/// server requests and records every request, answer POSTs included.
async fn ping_backend(script: PingScript) -> (String, Arc<Mutex<Vec<Received>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let answered = Arc::new(tokio::sync::Notify::new());
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let (log, answered, script) = (Arc::clone(&log), Arc::clone(&answered), script.clone());
            tokio::spawn(async move {
                let Some(request) = read_request(&mut stream).await else {
                    return;
                };
                let body = request.body.clone();
                let is_initialize = serde_json::from_str::<serde_json::Value>(&body).unwrap()["method"] == "initialize";
                log.lock().unwrap().push(request);
                if is_initialize {
                    serve_initialize(&mut stream, &body, &script, &answered).await;
                } else {
                    answered.notify_one();
                    serve_answer(&mut stream, script.answer).await;
                }
            });
        }
    });
    (format!("http://{addr}/mcp"), received)
}

/// Write the `initialize` stream: the server requests, then the result.
async fn serve_initialize(
    stream: &mut tokio::net::TcpStream,
    body: &str,
    script: &PingScript,
    answered: &tokio::sync::Notify,
) {
    // A stream that ends closes its connection, so the pool does not reuse it
    // for an answer this backend would never read.
    let connection = if script.close_after_result {
        "Connection: close\r\n"
    } else {
        ""
    };
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nMcp-Session-Id: s-ping\r\n{connection}\
         Transfer-Encoding: chunked\r\n\r\n"
    );
    let _written = stream.write_all(head.as_bytes()).await;
    for request in &script.requests {
        let _written = stream.write_all(sse_chunk(request).as_bytes()).await;
        let _flushed = stream.flush().await;
        if script.wait_for_answers {
            answered.notified().await;
        }
    }
    let _written = stream
        .write_all(sse_chunk(&init_result(body, "2025-06-18")).as_bytes())
        .await;
    if script.close_after_result {
        let _written = stream.write_all(b"0\r\n\r\n").await;
    }
    let _flushed = stream.flush().await;
    tokio::time::sleep(Duration::from_secs(30)).await;
}

/// Answer one answer POST as scripted.
async fn serve_answer(stream: &mut tokio::net::TcpStream, reply: AnswerReply) {
    let bytes = match reply {
        AnswerReply::Accepted => http_response("202 Accepted", &[], ""),
        AnswerReply::Status(status) => http_response(status, &[("Content-Type", "text/plain")], "no"),
        AnswerReply::After(delay) => {
            tokio::time::sleep(delay).await;
            http_response("202 Accepted", &[], "")
        },
        AnswerReply::Never => {
            tokio::time::sleep(Duration::from_secs(30)).await;
            return;
        },
    };
    let _written = stream.write_all(&bytes).await;
    let _flushed = stream.flush().await;
}

/// Run `initialize` with `client` against a ping backend.
async fn initialize_pinged(
    client: McpSessionClient,
    script: PingScript,
    timeout: Duration,
) -> (Result<McpSessionInit, McpInitializeError>, Vec<Received>) {
    let (url, received) = ping_backend(script).await;
    let result = client
        .initialize("/mcp", staged(&url).await, &callout(timeout), "2025-06-18")
        .await;
    let received = received.lock().unwrap().clone();
    (result, received)
}

/// A failure after the ping backend issued its session, which the caller
/// must still terminate.
fn half_made(error: McpSessionError) -> Result<McpSessionInit, McpInitializeError> {
    Err(McpInitializeError {
        error,
        issued_session: Some(HeaderValue::from_static("s-ping")),
    })
}

/// The answer POSTs a ping backend received, in order.
fn answers(received: &[Received]) -> Vec<&Received> {
    received
        .iter()
        .filter(|request| !request.body.contains("\"initialize\""))
        .collect()
}

#[tokio::test]
async fn ping_before_the_result_is_answered_on_the_session() {
    let script = PingScript::waiting(vec![ping(&serde_json::json!("ping-1"))]);
    let (result, received) = initialize_pinged(session_client(1 << 20), script, Duration::from_secs(5)).await;

    let init = result.unwrap();
    assert_eq!(init.session_id.unwrap(), "s-ping", "the session is established");
    let answers = answers(&received);
    assert_eq!(answers.len(), 1, "the ping is answered once");
    let pong = answers[0];
    assert_eq!(pong.method, "POST", "the answer is a new POST");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&pong.body).unwrap(),
        serde_json::json!({ "jsonrpc": "2.0", "id": "ping-1", "result": {} }),
        "the pong echoes the ping id with an empty result"
    );
    assert_eq!(
        pong.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
    assert_eq!(pong.header("mcp-session-id"), Some("s-ping"), "the session id is sent");
    assert_eq!(
        pong.header("content-type"),
        Some("application/json"),
        "the body is JSON"
    );
    assert!(
        pong.header("mcp-protocol-version").is_none(),
        "no version is negotiated before the initialize result"
    );
}

#[tokio::test]
async fn numeric_ping_ids_are_echoed() {
    let script = PingScript::waiting(vec![ping(&serde_json::json!(42))]);
    let (result, received) = initialize_pinged(session_client(1 << 20), script, Duration::from_secs(5)).await;

    assert!(result.is_ok(), "the handshake must complete: {result:?}");
    let body: serde_json::Value = serde_json::from_str(&answers(&received)[0].body).unwrap();
    assert_eq!(body["id"], 42, "a numeric id keeps its type");
}

#[tokio::test]
async fn ping_with_params_is_still_a_ping() {
    let ping = r#"{"jsonrpc":"2.0","id":3,"method":"ping","params":{"_meta":{"k":"v"}}}"#.to_owned();
    let (result, received) = initialize_pinged(
        session_client(1 << 20),
        PingScript::waiting(vec![ping]),
        Duration::from_secs(5),
    )
    .await;

    assert!(result.is_ok(), "the handshake must complete: {result:?}");
    let body: serde_json::Value = serde_json::from_str(&answers(&received)[0].body).unwrap();
    assert_eq!(
        body["result"],
        serde_json::json!({}),
        "a ping with params gets an empty result"
    );
}

#[tokio::test]
async fn other_server_requests_get_method_not_found() {
    let roots = r#"{"jsonrpc":"2.0","id":7,"method":"roots/list"}"#.to_owned();
    let custom = r#"{"jsonrpc":"2.0","id":"x","method":"vendor/probe","params":{}}"#.to_owned();
    let (result, received) = initialize_pinged(
        session_client(1 << 20),
        PingScript::waiting(vec![roots, custom]),
        Duration::from_secs(5),
    )
    .await;

    assert!(result.is_ok(), "the handshake must complete: {result:?}");
    let bodies: Vec<serde_json::Value> = answers(&received)
        .iter()
        .map(|request| serde_json::from_str(&request.body).unwrap())
        .collect();
    assert_eq!(bodies.len(), 2, "each request is answered");
    for (body, id) in bodies.iter().zip([serde_json::json!(7), serde_json::json!("x")]) {
        assert_eq!(body["id"], id, "the answer echoes the request id");
        assert_eq!(body["error"]["code"], -32601, "the method is not available: {body}");
        assert!(body.get("result").is_none(), "an error carries no result");
    }
}

#[tokio::test]
async fn unanswered_pong_post_times_out_at_the_deadline() {
    // The backend sends its result without waiting, but never answers the
    // pong POST. Answers are sent before the stream is read further, so the
    // handshake ends at its deadline.
    let script = PingScript {
        wait_for_answers: false,
        answer: AnswerReply::Never,
        ..PingScript::waiting(vec![ping(&serde_json::json!(1))])
    };
    let started = Instant::now();
    let (result, received) = initialize_pinged(session_client(1 << 20), script, Duration::from_millis(500)).await;

    assert_eq!(
        result,
        half_made(McpSessionError::Timeout),
        "an unanswered pong POST is a timeout"
    );
    assert!(started.elapsed() < Duration::from_secs(4), "the deadline ends the wait");
    assert_eq!(answers(&received).len(), 1, "the pong was sent");
}

#[tokio::test]
async fn rejected_pong_post_fails_the_handshake() {
    for (status, code) in [("400 Bad Request", 400), ("503 Service Unavailable", 503)] {
        let script = PingScript {
            answer: AnswerReply::Status(status),
            ..PingScript::waiting(vec![ping(&serde_json::json!(1))])
        };
        let (result, _) = initialize_pinged(session_client(1 << 20), script, Duration::from_secs(5)).await;

        assert_eq!(
            result,
            half_made(McpSessionError::Rejected { status: code }),
            "a rejected pong fails the handshake with its status"
        );
    }
}

#[tokio::test]
async fn pong_to_an_unreachable_endpoint_fails_the_handshake() {
    /// Stages the backend for `initialize` and a closed port for answers.
    struct DeadFollowUp(Upstream, Arc<dyn McpExchangeStage>);
    impl McpExchangeStage for DeadFollowUp {
        fn stage(&self, extensions: &mut RequestExtensions) {
            extensions.insert(StagedUpstream(self.0.clone()));
        }

        fn follow_up(&self) -> Arc<dyn McpExchangeStage> {
            Arc::clone(&self.1)
        }
    }

    let (url, received) = ping_backend(PingScript::waiting(vec![ping(&serde_json::json!(1))])).await;
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead = format!("http://{}/mcp", closed.local_addr().unwrap());
    drop(closed);
    let deadline = Instant::now() + Duration::from_secs(5);
    let target = prepare_url_target_with_policy(&url, deadline, |_addrs| Ok(()), UrlResolutionPolicy::ClientPerCall)
        .await
        .unwrap();
    let stage = Arc::new(DeadFollowUp(
        StagedUpstream::from_prepared_target(&target).unwrap().0,
        staged(&dead).await,
    ));

    let result = session_client(1 << 20)
        .initialize("/mcp", stage, &callout(Duration::from_secs(5)), "2025-06-18")
        .await;

    // The executor answers a refused dial with a synthesized 502.
    assert_eq!(
        result,
        half_made(McpSessionError::Rejected { status: 502 }),
        "an undeliverable pong fails the handshake"
    );
    assert!(
        answers(&received.lock().unwrap()).is_empty(),
        "the pong is dialed to the follow-up stage, not the stream's backend"
    );
}

#[tokio::test]
async fn more_server_requests_than_the_cap_are_malformed() {
    let pings = (0..=MAX_SERVER_REQUESTS).map(|n| ping(&serde_json::json!(n))).collect();
    let (result, received) = initialize_pinged(
        session_client(1 << 20),
        PingScript::waiting(pings),
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        half_made(McpSessionError::Malformed("too many server requests")),
        "requests past the cap end the handshake"
    );
    assert_eq!(
        answers(&received).len(),
        MAX_SERVER_REQUESTS,
        "only the capped requests are answered"
    );
}

#[tokio::test]
async fn ping_round_trip_stays_within_the_deadline() {
    let script = PingScript {
        answer: AnswerReply::After(Duration::from_secs(10)),
        ..PingScript::waiting(vec![ping(&serde_json::json!(1))])
    };
    let started = Instant::now();
    let (result, _) = initialize_pinged(session_client(1 << 20), script, Duration::from_millis(500)).await;

    assert_eq!(
        result,
        half_made(McpSessionError::Timeout),
        "a slow pong ends at the deadline"
    );
    // The pong would take 10s; the handshake must end near its 500ms budget.
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the pong does not extend the handshake budget: {:?}",
        started.elapsed()
    );

    let script = PingScript {
        answer: AnswerReply::After(Duration::from_millis(100)),
        ..PingScript::waiting(vec![ping(&serde_json::json!(1))])
    };
    let started = Instant::now();
    let (result, _) = initialize_pinged(session_client(1 << 20), script, Duration::from_secs(2)).await;
    assert!(result.is_ok(), "a pong within the budget completes: {result:?}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the whole exchange fits the deadline"
    );
}

#[tokio::test]
async fn ping_events_count_toward_the_size_limit() {
    let padded = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"ping","params":{{"_meta":{{"pad":"{}"}}}}}}"#,
        "p".repeat(4096)
    );
    let (result, received) = initialize_pinged(
        session_client(2048),
        PingScript::waiting(vec![padded]),
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        result,
        half_made(McpSessionError::ResponseTooLarge { limit: 2048 }),
        "bytes before the reply still count"
    );
    assert!(answers(&received).is_empty(), "a ping past the limit is not answered");
}

#[tokio::test]
async fn buffering_chain_answers_pings_after_the_stream_ends() {
    // A chain that buffers delivers the stream only after it ends; the server
    // here did not wait, so the result is present and the ping still answered.
    let script = PingScript {
        wait_for_answers: false,
        close_after_result: true,
        ..PingScript::waiting(vec![ping(&serde_json::json!("late"))])
    };
    let (result, received) = initialize_pinged(buffering_client(1 << 20), script, Duration::from_secs(5)).await;

    assert!(result.is_ok(), "the buffered handshake completes: {result:?}");
    let body: serde_json::Value = serde_json::from_str(&answers(&received)[0].body).unwrap();
    assert_eq!(body["id"], "late", "the buffered ping is answered");
}

#[tokio::test]
async fn buffering_chain_times_out_on_a_server_waiting_for_its_pong() {
    let script = PingScript::waiting(vec![ping(&serde_json::json!(1))]);
    let (result, received) = initialize_pinged(buffering_client(1 << 20), script, Duration::from_millis(500)).await;

    // The buffered response head never arrives, so the executor ends the
    // exchange at its deadline with a synthesized 504, as before this change.
    assert_eq!(
        result,
        Err(McpInitializeError::from(McpSessionError::Rejected { status: 504 })),
        "the stream never ends"
    );
    assert!(
        answers(&received).is_empty(),
        "nothing can be answered before the stream ends"
    );
}

#[test]
fn ping_content_is_not_logged() {
    const MARKER: &str = "secret-ping-marker";
    let ping =
        format!(r#"{{"jsonrpc":"2.0","id":"{MARKER}","method":"ping","params":{{"_meta":{{"m":"{MARKER}"}}}}}}"#);
    let mut result = None;
    let logs = crate::test_utils::capture_log_output(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        result = Some(runtime.block_on(initialize_pinged(
            session_client(1 << 20),
            PingScript::waiting(vec![ping]),
            Duration::from_secs(5),
        )));
    });

    let (result, received) = result.unwrap();
    assert!(result.is_ok(), "the handshake must complete: {result:?}");
    assert!(answers(&received)[0].body.contains(MARKER), "the pong echoes the id");
    assert!(!logs.contains(MARKER), "ping content must not reach logs: {logs}");
}

// -----------------------------------------------------------------------------
// notifications/initialized and DELETE
// -----------------------------------------------------------------------------

#[tokio::test]
async fn initialized_notification_carries_session_and_version() {
    let (url, received) = scripted_backend(vec![Reply::closed(http_response("202 Accepted", &[], ""))]).await;

    session_client(1 << 20)
        .send_initialized(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-9")),
        )
        .await
        .unwrap();

    let request = received.lock().unwrap()[0].clone();
    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(body["method"], "notifications/initialized", "notification method");
    assert!(body.get("id").is_none(), "a notification has no id");
    assert_eq!(request.header("mcp-session-id"), Some("s-9"), "session id must be sent");
    assert_eq!(
        request.header("mcp-protocol-version"),
        Some("2025-06-18"),
        "negotiated version must be sent"
    );
    assert_eq!(
        request.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
}

#[tokio::test]
async fn oversized_initialized_reply_is_too_large() {
    let (url, _) = scripted_backend(vec![Reply::closed(http_response(
        "200 OK",
        &[("Content-Type", "application/json")],
        &"x".repeat(8 * 1024),
    ))])
    .await;

    let result = session_client(4 * 1024)
        .send_initialized(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-9")),
        )
        .await;

    assert_eq!(
        result,
        Err(McpSessionError::ResponseTooLarge { limit: 4 * 1024 }),
        "the reply to notifications/initialized is bounded too"
    );
}

#[tokio::test]
async fn rejected_initialized_notification_keeps_status() {
    let (url, _) = scripted_backend(vec![Reply::closed(http_response("400 Bad Request", &[], ""))]).await;

    let result = session_client(1 << 20)
        .send_initialized(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-9")),
        )
        .await;

    assert_eq!(
        result,
        Err(McpSessionError::Rejected { status: 400 }),
        "status must be kept"
    );
}

#[tokio::test]
async fn terminate_sends_delete_with_session_credentials() {
    let (url, received) = scripted_backend(vec![Reply::closed(http_response("204 No Content", &[], ""))]).await;

    session_client(1 << 20)
        .terminate(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-3")),
        )
        .await
        .unwrap();

    let request = received.lock().unwrap()[0].clone();
    assert_eq!(request.method, "DELETE", "termination is a DELETE");
    assert_eq!(request.header("mcp-session-id"), Some("s-3"), "session id must be sent");
    assert_eq!(
        request.header("mcp-protocol-version"),
        Some("2025-06-18"),
        "version must be sent"
    );
    assert_eq!(
        request.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
}

#[tokio::test]
async fn terminate_issued_sends_delete_without_a_version() {
    let (url, received) = scripted_backend(vec![Reply::closed(http_response("204 No Content", &[], ""))]).await;

    session_client(1 << 20)
        .terminate_issued(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &HeaderValue::from_static("s-half"),
        )
        .await
        .unwrap();

    let request = received.lock().unwrap()[0].clone();
    assert_eq!(request.method, "DELETE", "termination is a DELETE");
    assert_eq!(
        request.header("mcp-session-id"),
        Some("s-half"),
        "session id must be sent"
    );
    assert!(
        request.header("mcp-protocol-version").is_none(),
        "no version was negotiated for an issued session"
    );
    assert_eq!(
        request.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
}

#[tokio::test]
async fn terminate_issued_reports_other_failures() {
    let (url, _) = scripted_backend(vec![Reply::closed(http_response("500 Internal Server Error", &[], ""))]).await;

    let result = session_client(1 << 20)
        .terminate_issued(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &HeaderValue::from_static("s-half"),
        )
        .await;

    assert_eq!(result, Err(McpSessionError::Rejected { status: 500 }), "status is kept");
}

#[tokio::test]
async fn sessionless_failure_reports_no_issued_session() {
    // A head without a session id leaves nothing on the server to end.
    let (plain, _) = scripted_backend(vec![Reply::closed(http_response(
        "200 OK",
        &[("Content-Type", "text/event-stream")],
        "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n\n",
    ))])
    .await;

    let result = session_client(1 << 20)
        .initialize(
            "/mcp",
            staged(&plain).await,
            &callout(Duration::from_secs(5)),
            "2025-06-18",
        )
        .await;

    assert_eq!(
        result,
        Err(McpInitializeError::from(McpSessionError::Malformed(
            "stream has no reply"
        ))),
        "a sessionless failure reports no session to end"
    );
}

#[tokio::test]
async fn terminate_treats_gone_and_unsupported_as_done() {
    for status in ["404 Not Found", "405 Method Not Allowed"] {
        let (url, _) = scripted_backend(vec![Reply::closed(http_response(status, &[], ""))]).await;
        let result = session_client(1 << 20)
            .terminate(
                "/mcp",
                staged(&url).await,
                &callout(Duration::from_secs(5)),
                &session(Some("s-3")),
            )
            .await;
        assert_eq!(result, Ok(()), "{status} leaves nothing to terminate");
    }
}

#[tokio::test]
async fn terminate_reports_other_failures() {
    let (url, _) = scripted_backend(vec![Reply::closed(http_response("500 Internal Server Error", &[], ""))]).await;

    let result = session_client(1 << 20)
        .terminate(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(Some("s-3")),
        )
        .await;

    assert_eq!(
        result,
        Err(McpSessionError::Rejected { status: 500 }),
        "500 must be reported"
    );
}

#[tokio::test]
async fn terminate_without_session_sends_nothing() {
    let (url, received) = scripted_backend(vec![Reply::closed(http_response("204 No Content", &[], ""))]).await;

    session_client(1 << 20)
        .terminate(
            "/mcp",
            staged(&url).await,
            &callout(Duration::from_secs(5)),
            &session(None),
        )
        .await
        .unwrap();

    assert!(received.lock().unwrap().is_empty(), "no session, no DELETE");
}

// -----------------------------------------------------------------------------
// Reply Validation
// -----------------------------------------------------------------------------

fn request_id() -> RequestId {
    RequestId::String(Arc::from("praxis-initialize-test"))
}

fn reply(value: serde_json::Value) -> ServerJsonRpcMessage {
    serde_json::from_value(value).unwrap()
}

#[test]
fn reply_to_another_request_id_is_rejected() {
    let message = reply(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "praxis-initialize-other",
        "result": { "protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": { "name": "s", "version": "1" } },
    }));
    assert_eq!(
        initialize_result_version(&message, &request_id()),
        Err(McpSessionError::Malformed("initialize reply id does not match")),
        "a reply must answer this handshake's own id"
    );
}

#[test]
fn error_for_another_request_id_is_rejected() {
    let message = ServerJsonRpcMessage::Error(JsonRpcError {
        jsonrpc: JsonRpcVersion2_0,
        id: Some(RequestId::Number(7)),
        error: ErrorData::new(ErrorCode(-32600), "x", None),
    });
    assert_eq!(
        initialize_result_version(&message, &request_id()),
        Err(McpSessionError::Malformed("initialize reply id does not match")),
        "an error for another request is not this handshake's answer"
    );
}

#[test]
fn result_without_server_info_is_not_an_initialize_result() {
    let message = reply(serde_json::json!({
        "jsonrpc": "2.0",
        "id": "praxis-initialize-test",
        "result": { "protocolVersion": "2025-06-18" },
    }));
    assert!(
        matches!(
            initialize_result_version(&message, &request_id()),
            Err(McpSessionError::Malformed(_))
        ),
        "MCP requires capabilities and serverInfo in an initialize result"
    );
}

#[test]
fn invalid_protocol_version_is_rejected() {
    for version in ["", "2025-06-18\u{7f}", &"9".repeat(MAX_PROTOCOL_VERSION_LEN + 1)] {
        let message = reply(serde_json::json!({
            "jsonrpc": "2.0",
            "id": "praxis-initialize-test",
            "result": { "protocolVersion": version, "capabilities": {}, "serverInfo": { "name": "s", "version": "1" } },
        }));
        assert_eq!(
            initialize_result_version(&message, &request_id()),
            Err(McpSessionError::Malformed("invalid protocolVersion")),
            "version {version:?} must be rejected"
        );
    }
}

#[test]
fn session_ids_must_be_visible_ascii_within_bounds() {
    let mut headers = HeaderMap::new();
    headers.insert(MCP_SESSION_ID, HeaderValue::from_static("a b"));
    assert!(response_session_id(&headers).is_err(), "a space must be rejected");
    let long = "a".repeat(MAX_SESSION_ID_LEN + 1);
    headers.insert(MCP_SESSION_ID, HeaderValue::from_str(&long).unwrap());
    assert!(
        response_session_id(&headers).is_err(),
        "an oversized id must be rejected"
    );
    headers.insert(MCP_SESSION_ID, HeaderValue::from_static("ok-1"));
    assert_eq!(
        response_session_id(&headers).unwrap().unwrap(),
        "ok-1",
        "a valid id is kept"
    );
}

#[test]
fn initialize_request_ids_are_unique() {
    let first = initialize_request_id().unwrap();
    assert!(
        first.starts_with(INITIALIZE_REQUEST_ID_PREFIX),
        "the id must carry the prefix"
    );
    assert_ne!(
        first,
        initialize_request_id().unwrap(),
        "each handshake must get its own id"
    );
}

#[test]
fn errors_carry_no_server_text() {
    let rendered = McpSessionError::JsonRpc { code: -32602 }.to_string();
    assert!(
        !rendered.contains("secret"),
        "server messages never reach the error text"
    );
}

// -----------------------------------------------------------------------------
// tools/list
// -----------------------------------------------------------------------------

/// A `tools/list` result for the request the backend received.
fn tools_result(request: &serde_json::Value, tools: &serde_json::Value, next: Option<&str>) -> String {
    let mut result = serde_json::json!({ "tools": tools });
    if let Some(next) = next {
        result["nextCursor"] = serde_json::json!(next);
    }
    serde_json::json!({ "jsonrpc": "2.0", "id": request["id"], "result": result }).to_string()
}

/// Start a backend answering `tools/list` with `reply`, built from the
/// request it received. With `ping_first`, the reply arrives on an SSE stream
/// after a `ping` the client must answer first.
async fn listing_backend(
    ping_first: bool,
    reply: impl Fn(&serde_json::Value) -> String + Send + Sync + 'static,
) -> (String, Arc<Mutex<Vec<Received>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let received = Arc::new(Mutex::new(Vec::new()));
    let answered = Arc::new(tokio::sync::Notify::new());
    let reply = Arc::new(reply);
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let (log, answered, reply) = (Arc::clone(&log), Arc::clone(&answered), Arc::clone(&reply));
            tokio::spawn(async move {
                let Some(request) = read_request(&mut stream).await else {
                    return;
                };
                let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
                log.lock().unwrap().push(request);
                if body["method"] != "tools/list" {
                    answered.notify_one();
                    let _written = stream.write_all(&http_response("202 Accepted", &[], "")).await;
                    return;
                }
                if !ping_first {
                    let bytes = http_response("200 OK", &[("Content-Type", "application/json")], &reply(&body));
                    let _written = stream.write_all(&bytes).await;
                    return;
                }
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
                let _written = stream.write_all(head.as_bytes()).await;
                let _written = stream
                    .write_all(sse_chunk(&ping(&serde_json::json!("ping-l"))).as_bytes())
                    .await;
                let _flushed = stream.flush().await;
                answered.notified().await;
                let _written = stream.write_all(sse_chunk(&reply(&body)).as_bytes()).await;
                let _flushed = stream.flush().await;
                tokio::time::sleep(Duration::from_secs(30)).await;
            });
        }
    });
    (format!("http://{addr}/mcp"), received)
}

fn object_tool(name: &str) -> serde_json::Value {
    serde_json::json!({ "name": name, "inputSchema": { "type": "object" } })
}

async fn list_via(
    url: &str,
    scope: McpListingScope<'_>,
    cursor: Option<&str>,
) -> Result<McpToolPage<serde_json::Value>, McpSessionError> {
    session_client(1 << 20)
        .list_tools(
            "/mcp",
            staged(url).await,
            &callout(Duration::from_secs(5)),
            scope,
            cursor.map(str::to_owned),
        )
        .await
}

#[tokio::test]
async fn session_listing_sends_the_session_and_reads_the_page() {
    let (url, received) = listing_backend(false, |request| {
        tools_result(
            request,
            &serde_json::json!([object_tool("a"), object_tool("b")]),
            Some("next"),
        )
    })
    .await;
    let session = session(Some("s-1"));

    let page = list_via(&url, McpListingScope::Session(&session), Some("c1"))
        .await
        .unwrap();

    let names: Vec<&str> = page.tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(page.next_cursor.as_deref(), Some("next"), "the next cursor is returned");
    let received = received.lock().unwrap();
    let request = &received[0];
    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    assert_eq!(body["method"], "tools/list");
    assert_eq!(body["params"]["cursor"], "c1", "the cursor is sent");
    assert!(
        body["params"].get("_meta").is_none(),
        "no stateless metadata on a session"
    );
    assert!(
        body["id"].as_str().unwrap().starts_with(TOOLS_LIST_REQUEST_ID_PREFIX),
        "a fresh id that no client id collides with"
    );
    assert_eq!(request.header("mcp-session-id"), Some("s-1"));
    assert_eq!(request.header("mcp-protocol-version"), Some("2025-06-18"));
    assert_eq!(
        request.header("authorization"),
        Some("Bearer first"),
        "credential forwarded"
    );
    assert!(request.header("mcp-method").is_none());
}

#[tokio::test]
async fn stateless_listing_declares_the_profile() {
    let (url, received) = listing_backend(false, |request| {
        tools_result(request, &serde_json::json!([object_tool("a")]), None)
    })
    .await;

    let page = list_via(
        &url,
        McpListingScope::Stateless {
            protocol_version: "2026-07-28",
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(page.tools.len(), 1);
    assert!(page.next_cursor.is_none(), "the last page has no cursor");
    let received = received.lock().unwrap();
    let request = &received[0];
    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    let meta = &body["params"]["_meta"];
    assert_eq!(meta["io.modelcontextprotocol/protocolVersion"], "2026-07-28");
    assert_eq!(meta["io.modelcontextprotocol/clientInfo"]["name"], CLIENT_NAME);
    assert!(meta["io.modelcontextprotocol/clientCapabilities"].is_object());
    assert!(body["params"].get("cursor").is_none(), "the first page sends no cursor");
    assert_eq!(request.header("mcp-protocol-version"), Some("2026-07-28"));
    assert_eq!(request.header("mcp-method"), Some("tools/list"));
    assert!(request.header("mcp-session-id").is_none(), "no session");
}

#[tokio::test]
async fn listing_errors_keep_their_class() {
    let (url, _) = listing_backend(false, |request| {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": { "code": -32601, "message": "no tools here" },
        })
        .to_string()
    })
    .await;
    let session = session(Some("s-1"));
    assert_eq!(
        list_via(&url, McpListingScope::Session(&session), None).await,
        Err(McpSessionError::JsonRpc { code: -32601 }),
        "a JSON-RPC error keeps its code"
    );

    let (url, _) = listing_backend(false, |request| {
        tools_result(request, &serde_json::json!([{ "name": "no_schema" }]), None)
    })
    .await;
    assert_eq!(
        list_via(&url, McpListingScope::Session(&session), None).await,
        Err(McpSessionError::Malformed(
            "tools/list reply is not a tools/list result"
        )),
        "a tool without an input schema is not an MCP tool"
    );

    let (url, _) = listing_backend(false, |_request| {
        tools_result(&serde_json::json!({ "id": "other" }), &serde_json::json!([]), None)
    })
    .await;
    assert_eq!(
        list_via(&url, McpListingScope::Session(&session), None).await,
        Err(McpSessionError::Malformed("tools/list reply id does not match"))
    );
}

#[tokio::test]
async fn ping_during_a_listing_is_answered_with_the_negotiated_version() {
    let (url, received) = listing_backend(true, |request| {
        tools_result(request, &serde_json::json!([object_tool("a")]), None)
    })
    .await;
    let session = session(Some("s-1"));

    let page = list_via(&url, McpListingScope::Session(&session), None).await.unwrap();

    assert_eq!(page.tools.len(), 1, "the reply after the ping is read");
    let received = received.lock().unwrap();
    let pong = received
        .iter()
        .find(|request| !request.body.contains("tools/list"))
        .expect("the ping is answered");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&pong.body).unwrap(),
        serde_json::json!({ "jsonrpc": "2.0", "id": "ping-l", "result": {} })
    );
    assert_eq!(pong.header("mcp-session-id"), Some("s-1"));
    assert_eq!(
        pong.header("mcp-protocol-version"),
        Some("2025-06-18"),
        "answers after initialization carry the negotiated version"
    );
}
