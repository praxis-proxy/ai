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
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
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
    let result = session_client(max_response_bytes)
        .initialize("/mcp", staged(&url).await, &callout(timeout), "2025-06-18")
        .await;
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
                "data: {\"jsonrpc\":\"2.0\",\"method\":\"ping\",\"id\":9}\n\n",
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
        Err(McpSessionError::Rejected { status: 502 }),
        "a refused dial must fail without a session"
    );
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
