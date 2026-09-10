// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Size-bounded HTTP adapter for RMCP tool-call responses.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use bytes::{Bytes, BytesMut};
use futures::{StreamExt as _, future, stream::BoxStream};
use http::{HeaderName, HeaderValue, header::WWW_AUTHENTICATE};
use reqwest::header::ACCEPT;
use rmcp::{
    model::{ClientJsonRpcMessage, ClientRequest, JsonRpcMessage, ServerJsonRpcMessage},
    transport::{
        common::http_header::{EVENT_STREAM_MIME_TYPE, HEADER_LAST_EVENT_ID, HEADER_SESSION_ID, JSON_MIME_TYPE},
        streamable_http_client::{
            AuthRequiredError, InsufficientScopeError, SseError, StreamableHttpClient, StreamableHttpError,
            StreamableHttpPostResponse,
        },
    },
};
use sse_stream::{Sse, SseStream};

/// Wire-size allowance for MCP initialization and control responses.
const MAX_CONTROL_RESPONSE_BYTES: usize = 1_048_576;
/// JSON-RPC envelope allowance above the configured retained tool-result cap.
const MAX_TOOL_RESULT_ENVELOPE_BYTES: usize = 65_536;
/// Worst-case JSON string encoding expansion (`\u00XX` for one input byte).
const MAX_JSON_STRING_EXPANSION: usize = 6;

/// Reqwest transport that bounds JSON bodies before deserialization.
#[derive(Clone)]
pub(super) struct BoundedMcpHttpClient {
    /// Pinned client that performs the outbound request.
    inner: reqwest::Client,
    /// Raw tool-call response ceiling applied before JSON or SSE parsing.
    max_tool_response_bytes: usize,
}

impl BoundedMcpHttpClient {
    /// Wrap a pinned reqwest client with one response-body ceiling.
    pub(super) fn new(inner: reqwest::Client, max_response_bytes: usize) -> Self {
        Self {
            inner,
            max_tool_response_bytes: max_response_bytes
                .saturating_mul(MAX_JSON_STRING_EXPANSION)
                .saturating_add(MAX_TOOL_RESULT_ENVELOPE_BYTES),
        }
    }

    /// Largest SSE event accepted by the underlying RMCP transport.
    pub(super) fn max_sse_event_size(&self) -> usize {
        self.max_tool_response_bytes.max(MAX_CONTROL_RESPONSE_BYTES)
    }

    /// Select a wire ceiling without applying the result-specific cap to the
    /// initialization handshake.
    fn response_limit(&self, message: &ClientJsonRpcMessage, max_sse_event_size: usize) -> usize {
        let configured = match message {
            ClientJsonRpcMessage::Request(request) if matches!(request.request, ClientRequest::CallToolRequest(_)) => {
                self.max_tool_response_bytes
            },
            _ => MAX_CONTROL_RESPONSE_BYTES,
        };
        configured.min(max_sse_event_size)
    }
}

/// Error used by the byte stream before SSE parsing.
#[derive(Debug, thiserror::Error)]
enum BoundedBodyError {
    /// Reqwest failed while streaming the response.
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    /// The response crossed its configured byte ceiling.
    #[error("MCP response exceeded the configured limit of {limit} bytes")]
    TooLarge {
        /// Configured byte ceiling crossed by the stream.
        limit: usize,
    },
}

/// Incremental raw-size accounting for one SSE event.
#[derive(Debug)]
struct SseEventSizeLimiter {
    /// Maximum raw bytes retained for one event.
    limit: usize,
    /// Bytes observed on completed non-empty lines in the current event.
    event_size: usize,
    /// Bytes observed on the current incomplete line.
    line_size: usize,
    /// Whether a trailing carriage return may be followed by a line feed.
    previous_was_cr: bool,
}

impl SseEventSizeLimiter {
    /// Start an event-size limiter at the given ceiling.
    fn new(limit: usize) -> Self {
        Self {
            limit,
            event_size: 0,
            line_size: 0,
            previous_was_cr: false,
        }
    }

    /// Account for one raw transport chunk.
    fn observe(&mut self, chunk: &[u8]) -> Result<(), ()> {
        for &byte in chunk {
            if self.previous_was_cr {
                self.previous_was_cr = false;
                if byte == b'\n' {
                    self.finish_line(2)?;
                    continue;
                }
                self.finish_line(1)?;
            }
            match byte {
                b'\r' => {
                    self.previous_was_cr = true;
                    self.check_limit()?;
                },
                b'\n' => self.finish_line(1)?,
                _ => {
                    self.line_size = self.line_size.saturating_add(1);
                    self.check_limit()?;
                },
            }
        }
        Ok(())
    }

    /// Complete the current line and reset at an empty event delimiter.
    fn finish_line(&mut self, delimiter_bytes: usize) -> Result<(), ()> {
        if self.line_size == 0 {
            if self.event_size.saturating_add(delimiter_bytes) > self.limit {
                return Err(());
            }
            self.event_size = 0;
        } else {
            self.event_size = self
                .event_size
                .saturating_add(self.line_size)
                .saturating_add(delimiter_bytes);
        }
        self.line_size = 0;
        self.check_limit()
    }

    /// Reject when the current event has crossed its ceiling.
    fn check_limit(&self) -> Result<(), ()> {
        let pending_carriage_return = usize::from(self.previous_was_cr);
        (self
            .event_size
            .saturating_add(self.line_size)
            .saturating_add(pending_carriage_return)
            <= self.limit)
            .then_some(())
            .ok_or(())
    }
}

/// Attach transport-provided headers to one request.
fn apply_headers(
    mut request: reqwest::RequestBuilder,
    headers: HashMap<HeaderName, HeaderValue>,
) -> Result<reqwest::RequestBuilder, StreamableHttpError<reqwest::Error>> {
    for (name, value) in headers {
        let reserved = name == ACCEPT
            || name.as_str().eq_ignore_ascii_case(HEADER_SESSION_ID)
            || name.as_str().eq_ignore_ascii_case(HEADER_LAST_EVENT_ID);
        if reserved {
            return Err(StreamableHttpError::ReservedHeaderConflict(name.to_string()));
        }
        request = request.header(name, value);
    }
    Ok(request)
}

/// Read one response body while rejecting before retained bytes cross `limit`.
async fn collect_bounded(
    response: reqwest::Response,
    limit: usize,
) -> Result<Bytes, StreamableHttpError<reqwest::Error>> {
    let limit_u64 = u64::try_from(limit).unwrap_or(u64::MAX);
    if response.content_length().is_some_and(|length| length > limit_u64) {
        return Err(response_too_large(limit));
    }
    let mut body = BytesMut::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(StreamableHttpError::Client)?;
        let next = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| response_too_large(limit))?;
        if next > limit {
            return Err(response_too_large(limit));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

/// Build a non-sensitive RMCP transport error for an oversized body.
fn response_too_large(limit: usize) -> StreamableHttpError<reqwest::Error> {
    StreamableHttpError::UnexpectedServerResponse(Cow::Owned(format!(
        "MCP response exceeded the configured limit of {limit} bytes"
    )))
}

/// Extract the optional `scope` parameter from a bearer challenge.
fn required_scope(header: &str) -> Option<String> {
    let lower = header.to_ascii_lowercase();
    let start = lower.find("scope=")?.saturating_add("scope=".len());
    let value = header.get(start..)?;
    if let Some(quoted) = value.strip_prefix('"') {
        return quoted
            .find('"')
            .and_then(|end| quoted.get(..end))
            .map(ToOwned::to_owned);
    }
    let end = value
        .find(|character: char| character == ',' || character == ';' || character.is_whitespace())
        .unwrap_or(value.len());
    (end > 0).then(|| value.get(..end).map(ToOwned::to_owned)).flatten()
}

/// Preserve RMCP compatibility for empty successful notification responses.
fn accepts_empty_success(
    status: reqwest::StatusCode,
    content_length: Option<u64>,
    message: &ClientJsonRpcMessage,
) -> bool {
    status.is_success()
        && content_length == Some(0)
        && matches!(
            message,
            ClientJsonRpcMessage::Notification(_) | ClientJsonRpcMessage::Response(_) | ClientJsonRpcMessage::Error(_)
        )
}

/// Parse an SSE response while bounding each event independently.
fn bounded_sse_response(response: reqwest::Response, limit: usize) -> BoxStream<'static, Result<Sse, SseError>> {
    let stream = response
        .bytes_stream()
        .scan((SseEventSizeLimiter::new(limit), false), move |state, item| {
            let output = if state.1 {
                None
            } else {
                Some(match item {
                    Ok(chunk) => {
                        if state.0.observe(&chunk).is_ok() {
                            Ok(chunk)
                        } else {
                            state.1 = true;
                            Err(BoundedBodyError::TooLarge { limit })
                        }
                    },
                    Err(error) => {
                        state.1 = true;
                        Err(BoundedBodyError::Reqwest(error))
                    },
                })
            };
            future::ready(output)
        });
    SseStream::from_bytes_stream(stream).boxed()
}

impl StreamableHttpClient for BoundedMcpHttpClient {
    type Error = reqwest::Error;

    async fn post_message(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        self.post_message_with_max_sse_event_size(
            uri,
            message,
            session_id,
            auth_header,
            custom_headers,
            self.max_sse_event_size(),
        )
        .await
    }

    #[expect(
        clippy::too_many_lines,
        reason = "HTTP status and content-type lifecycle is intentionally linear"
    )]
    #[expect(
        clippy::large_stack_frames,
        reason = "the async RMCP transport future owns reqwest and JSON-RPC state"
    )]
    async fn post_message_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        message: ClientJsonRpcMessage,
        session_id: Option<Arc<str>>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<StreamableHttpPostResponse, StreamableHttpError<Self::Error>> {
        let limit = self.response_limit(&message, max_sse_event_size);
        let mut request = self
            .inner
            .post(uri.as_ref())
            .header(ACCEPT, [EVENT_STREAM_MIME_TYPE, JSON_MIME_TYPE].join(", "));
        if let Some(auth_header) = auth_header {
            request = request.bearer_auth(auth_header);
        }
        let had_session = session_id.is_some();
        if let Some(session_id) = session_id {
            request = request.header(HEADER_SESSION_ID, session_id.as_ref());
        }
        let response = apply_headers(request, custom_headers)?
            .json(&message)
            .send()
            .await
            .map_err(StreamableHttpError::Client)?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED
            && let Some(header) = response.headers().get(WWW_AUTHENTICATE)
        {
            let challenge = header.to_str().map_err(|_error| {
                StreamableHttpError::UnexpectedServerResponse(Cow::Borrowed("invalid www-authenticate header value"))
            })?;
            return Err(StreamableHttpError::AuthRequired(AuthRequiredError::new(
                challenge.to_owned(),
            )));
        }
        if status == reqwest::StatusCode::FORBIDDEN
            && let Some(header) = response.headers().get(WWW_AUTHENTICATE)
        {
            let challenge = header.to_str().map_err(|_error| {
                StreamableHttpError::UnexpectedServerResponse(Cow::Borrowed("invalid www-authenticate header value"))
            })?;
            return Err(StreamableHttpError::InsufficientScope(InsufficientScopeError::new(
                challenge.to_owned(),
                required_scope(challenge),
            )));
        }
        if matches!(status, reqwest::StatusCode::ACCEPTED | reqwest::StatusCode::NO_CONTENT) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        if status == reqwest::StatusCode::NOT_FOUND && had_session {
            return Err(StreamableHttpError::SessionExpired);
        }
        if accepts_empty_success(status, response.content_length(), &message) {
            return Ok(StreamableHttpPostResponse::Accepted);
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
        let returned_session = response
            .headers()
            .get(HEADER_SESSION_ID)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        if !status.is_success() {
            let body = collect_bounded(response, limit).await?;
            if content_type
                .as_deref()
                .is_some_and(|value| value.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()))
                && let Ok(message @ JsonRpcMessage::Error(_)) = serde_json::from_slice(&body)
            {
                return Ok(StreamableHttpPostResponse::Json(message, returned_session));
            }
            // Our callers use `ServiceExt::serve`, whose rmcp lifecycle is
            // always `Initialize`; they never send the `DiscoverRequest` that
            // rmcp's reqwest-only legacy-discovery fallback recognizes.
            return Err(StreamableHttpError::UnexpectedServerResponse(Cow::Owned(format!(
                "HTTP {status}"
            ))));
        }
        match content_type.as_deref() {
            Some(content_type) if content_type.as_bytes().starts_with(EVENT_STREAM_MIME_TYPE.as_bytes()) => Ok(
                StreamableHttpPostResponse::Sse(bounded_sse_response(response, limit), returned_session),
            ),
            Some(content_type) if content_type.as_bytes().starts_with(JSON_MIME_TYPE.as_bytes()) => {
                let body = collect_bounded(response, limit).await?;
                if body.is_empty() {
                    return Ok(StreamableHttpPostResponse::Accepted);
                }
                match serde_json::from_slice::<ServerJsonRpcMessage>(&body) {
                    Ok(message) => Ok(StreamableHttpPostResponse::Json(message, returned_session)),
                    Err(error) => {
                        tracing::warn!(%error, "could not parse bounded MCP JSON response; treating as accepted");
                        Ok(StreamableHttpPostResponse::Accepted)
                    },
                }
            },
            _ => Err(StreamableHttpError::UnexpectedContentType(content_type)),
        }
    }

    async fn delete_session(
        &self,
        uri: Arc<str>,
        session_id: Arc<str>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<(), StreamableHttpError<Self::Error>> {
        <reqwest::Client as StreamableHttpClient>::delete_session(
            &self.inner,
            uri,
            session_id,
            auth_header,
            custom_headers,
        )
        .await
    }

    async fn get_stream(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        self.get_stream_with_max_sse_event_size(
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
            self.max_sse_event_size(),
        )
        .await
    }

    async fn get_stream_with_max_sse_event_size(
        &self,
        uri: Arc<str>,
        session_id: Option<Arc<str>>,
        last_event_id: Option<String>,
        auth_header: Option<String>,
        custom_headers: HashMap<HeaderName, HeaderValue>,
        max_sse_event_size: usize,
    ) -> Result<BoxStream<'static, Result<Sse, SseError>>, StreamableHttpError<Self::Error>> {
        // A session GET stream carries server requests and notifications as
        // well as request-scoped responses resumed by RMCP. It therefore needs
        // the transport's control-event allowance. Tool-result retention is
        // still enforced after decoding by the dispatcher's per-call cap.
        <reqwest::Client as StreamableHttpClient>::get_stream_with_max_sse_event_size(
            &self.inner,
            uri,
            session_id,
            last_event_id,
            auth_header,
            custom_headers,
            max_sse_event_size,
        )
        .await
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test setup and assertions")]
mod tests {
    use std::io::{Read as _, Write as _};

    use super::*;

    /// Serve one raw HTTP response and return its loopback URL.
    fn serve_once(response: Vec<u8>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let address = listener.local_addr().expect("listener should have an address");
        std::thread::spawn(move || {
            let (mut socket, _peer) = listener.accept().expect("request should connect");
            let mut request = [0_u8; 1024];
            let _read = socket.read(&mut request).expect("request should be readable");
            socket.write_all(&response).expect("response should be writable");
        });
        format!("http://{address}/mcp")
    }

    #[tokio::test]
    async fn content_length_is_rejected_before_body_collection() {
        let body = vec![b'x'; 64];
        let mut response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        response.extend_from_slice(&body);
        let url = serve_once(response);
        let response = reqwest::Client::new()
            .get(url)
            .send()
            .await
            .expect("request should succeed");

        let error = collect_bounded(response, 16)
            .await
            .expect_err("declared oversized body must fail");

        assert!(error.to_string().contains("limit of 16 bytes"));
    }

    #[tokio::test]
    async fn chunked_body_is_rejected_while_streaming() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n4\r\n1234\r\n4\r\n5678\r\n0\r\n\r\n".to_vec();
        let url = serve_once(response);
        let response = reqwest::Client::new()
            .get(url)
            .send()
            .await
            .expect("request should succeed");

        let error = collect_bounded(response, 6)
            .await
            .expect_err("streamed oversized body must fail");

        assert!(error.to_string().contains("limit of 6 bytes"));
    }

    #[test]
    fn sse_limit_resets_between_events() {
        let mut limiter = SseEventSizeLimiter::new(12);

        for _ in 0..100 {
            limiter
                .observe(b"data: ok\n\n")
                .expect("each event stays within the limit");
        }
    }

    #[test]
    fn sse_limit_applies_across_chunks_of_one_event() {
        let mut limiter = SseEventSizeLimiter::new(12);

        limiter.observe(b"data: ").expect("partial event is within the limit");
        assert!(
            limiter.observe(b"1234567").is_err(),
            "one split oversized event must fail"
        );
    }

    #[test]
    fn sse_limit_counts_both_crlf_bytes() {
        let mut limiter = SseEventSizeLimiter::new(11);

        limiter.observe(b"a\r\nb\r\nc\r\n").expect("nine bytes fit");
        assert!(
            limiter.observe(b"d\r\n").is_err(),
            "the fourth three-byte CRLF line must cross the limit"
        );
    }

    #[test]
    fn empty_success_notification_is_accepted_without_content_type() {
        let message: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc":"2.0", "method":"notifications/initialized"
        }))
        .expect("notification should deserialize");

        assert!(accepts_empty_success(reqwest::StatusCode::OK, Some(0), &message));
    }

    #[test]
    fn tool_result_limit_does_not_cap_initialization_response() {
        let client = BoundedMcpHttpClient::new(reqwest::Client::new(), 16);
        let initialize: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc":"2.0", "id":1, "method":"initialize",
            "params":{"protocolVersion":"2025-03-26", "capabilities":{}, "clientInfo":{"name":"test", "version":"1"}}
        }))
        .expect("initialize request should deserialize");
        let call: ClientJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc":"2.0", "id":2, "method":"tools/call",
            "params":{"name":"echo", "arguments":{}}
        }))
        .expect("tool call should deserialize");

        assert_eq!(
            client.response_limit(&initialize, usize::MAX),
            MAX_CONTROL_RESPONSE_BYTES
        );
        assert_eq!(
            client.response_limit(&call, usize::MAX),
            16 * MAX_JSON_STRING_EXPANSION + MAX_TOOL_RESULT_ENVELOPE_BYTES
        );
        assert_eq!(
            client.max_sse_event_size(),
            MAX_CONTROL_RESPONSE_BYTES,
            "session GET delivery must retain the control-event allowance"
        );
    }

    #[test]
    fn tool_result_wire_limit_allows_worst_case_json_escaping() {
        let decoded_limit = 4_194_304;
        let client = BoundedMcpHttpClient::new(reqwest::Client::new(), decoded_limit);

        assert_eq!(
            client.max_tool_response_bytes,
            decoded_limit * MAX_JSON_STRING_EXPANSION + MAX_TOOL_RESULT_ENVELOPE_BYTES
        );
        assert!(
            client.max_tool_response_bytes
                >= serde_json::to_vec(&"\u{1}".repeat(decoded_limit))
                    .expect("JSON string serialization should succeed")
                    .len(),
            "the wire cap must admit a decoded payload whose bytes all require six-byte JSON escapes"
        );
    }
}
