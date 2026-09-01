// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared HTTP client for OpenAI-compatible API callouts.
//!
//! Provides URL construction, SSRF-safe base-URL validation,
//! resource-ID path-segment encoding, header forwarding, bounded
//! JSON and byte reads, and normalized error mapping. Used by
//! [`FilesApiClient`] and vector-store search.
//!
//! All requests route through the [`SubRequestClient`] from
//! praxis-core for connection pooling, TLS, admission control,
//! and response body size limits.
//!
//! Each consuming filter retains its own [`ApiClient`] instance.
//!
//! [`FilesApiClient`]: super::responses::file_resolve
//! [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient

pub(crate) mod error;
pub(crate) mod url;

use std::time::Duration;

use bytes::Bytes;
use http::HeaderMap;

pub(crate) use self::{
    error::ApiClientError,
    url::{resource_url, validate_base_url, validate_forward_headers},
};
use crate::subrequest::{self, SubRequest, SubRequestClient, SubRequestError, SubResponse};

/// Configuration for constructing an [`ApiClient`].
///
/// Assembled programmatically by each consuming filter from its
/// own validated YAML config — no shared YAML schema.
pub(crate) struct ApiClientConfig {
    /// Base URL of the API endpoint (trailing slash stripped).
    pub api_base_url: String,
    /// Sub-request client for bounded execution.
    pub client: SubRequestClient,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Maximum response body bytes.
    pub max_response_bytes: usize,
    /// Header names to forward from the original request.
    pub forward_header_names: Vec<http::HeaderName>,
}

/// Shared HTTP client for OpenAI-compatible API callouts.
///
/// All requests route through the [`SubRequestClient`] from
/// praxis-core for connection pooling, TLS, admission control,
/// and response body size limits.
///
/// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
pub(crate) struct ApiClient {
    /// Base URL of the API endpoint (trailing slash stripped).
    api_base_url: String,
    /// Sub-request client for bounded execution.
    client: SubRequestClient,
    /// Per-request timeout.
    timeout: Duration,
    /// Maximum response body bytes for JSON requests.
    max_response_bytes: usize,
    /// Header names to forward from the original downstream
    /// request.
    forward_header_names: Vec<http::HeaderName>,
}

/// Map a [`SubRequestError`] to an [`ApiClientError`].
fn map_subrequest_error(err: SubRequestError) -> ApiClientError {
    match err {
        SubRequestError::ResponseTooLarge { limit, .. } => ApiClientError::ResponseTooLarge { limit },
        source => ApiClientError::Transport { source },
    }
}

impl ApiClient {
    /// Build a new client from validated configuration.
    ///
    /// The base URL should already be validated with
    /// [`validate_base_url`].
    pub(crate) fn new(config: ApiClientConfig) -> Self {
        let ApiClientConfig {
            api_base_url,
            client,
            timeout,
            max_response_bytes,
            forward_header_names,
        } = config;

        Self {
            api_base_url: api_base_url.trim_end_matches('/').to_owned(),
            client,
            timeout,
            max_response_bytes,
            forward_header_names,
        }
    }

    /// Return the validated base URL.
    pub(crate) fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    /// Return the configured maximum response size.
    pub(crate) fn max_response_bytes(&self) -> usize {
        self.max_response_bytes
    }

    /// Return the configured per-request timeout.
    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Build a resource URL from the configured base, a path
    /// prefix, a resource ID, and an optional suffix.
    ///
    /// See [`resource_url`] for encoding and validation behavior.
    pub(crate) fn resource_url(
        &self,
        path_prefix: &str,
        resource_id: &str,
        suffix: Option<&str>,
    ) -> Result<String, ApiClientError> {
        resource_url(&self.api_base_url, path_prefix, resource_id, suffix)
    }

    /// Shared sub-request client used for callouts.
    pub(crate) fn subrequest_client(&self) -> &SubRequestClient {
        &self.client
    }

    /// Send a GET request and return a bounded HTTP response.
    pub(crate) async fn get(
        &self,
        url: &str,
        request_headers: &HeaderMap,
        max_response_bytes: usize,
    ) -> Result<SubResponse, ApiClientError> {
        let headers = self.build_header_map(request_headers);
        self.execute_url(url, http::Method::GET, headers, Bytes::new(), max_response_bytes)
            .await
    }

    /// Send a pre-serialized JSON body and return the bounded HTTP
    /// response.
    pub(crate) async fn post_json_bytes(
        &self,
        url: String,
        body: Vec<u8>,
        request_headers: &HeaderMap,
    ) -> Result<SubResponse, ApiClientError> {
        let mut headers = self.build_header_map(request_headers);
        headers.remove(http::header::CONTENT_TYPE);
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        self.execute_url(
            &url,
            http::Method::POST,
            headers,
            Bytes::from(body),
            self.max_response_bytes,
        )
        .await
    }

    /// Send a GET request and return the response body with
    /// bounded reads.
    ///
    /// The Pingora connector does not follow redirects because it
    /// connects to a specific peer. Redirect responses are returned
    /// to the caller as bounded HTTP responses.
    pub(crate) async fn get_bytes(
        &self,
        url: &str,
        request_headers: &HeaderMap,
        max_bytes: usize,
    ) -> Result<Bytes, ApiClientError> {
        let response = self.get(url, request_headers, max_bytes).await?;
        Ok(response.body)
    }

    /// Copy configured headers from the original downstream
    /// request into a [`HeaderMap`] for forwarding.
    pub(crate) fn forward_headers(&self, request_headers: &HeaderMap) -> Vec<(http::HeaderName, http::HeaderValue)> {
        let mut headers = Vec::new();
        for name in &self.forward_header_names {
            if let Some(value) = request_headers.get(name) {
                headers.push((name.clone(), value.clone()));
            }
        }
        headers
    }

    /// Build a [`HeaderMap`] from forwarded headers.
    fn build_header_map(&self, request_headers: &HeaderMap) -> HeaderMap {
        let mut map = HeaderMap::new();
        for name in &self.forward_header_names {
            if let Some(value) = request_headers.get(name) {
                map.insert(name.clone(), value.clone());
            }
        }
        map
    }

    /// Parse the URL, build a [`SubRequest`], and execute it with
    /// the caller's response-size limit.
    #[expect(
        clippy::too_many_arguments,
        reason = "the request's independently owned transport fields stay explicit"
    )]
    async fn execute_url(
        &self,
        url: &str,
        method: http::Method,
        headers: HeaderMap,
        body: Bytes,
        max_response_bytes: usize,
    ) -> Result<SubResponse, ApiClientError> {
        let request = SubRequest {
            method,
            uri: http::Uri::default(),
            headers,
            body,
        };

        let mut response = subrequest::execute_url(&self.client, url, request, max_response_bytes, self.timeout)
            .await
            .map_err(map_subrequest_error)?;
        sanitize_response_headers(&mut response.headers);
        Ok(response)
    }
}

/// Retain the safe response metadata required by callout consumers.
fn sanitize_response_headers(headers: &mut HeaderMap) {
    let mut sanitized = HeaderMap::new();
    for name in [
        "content-type",
        "retry-after",
        "x-request-id",
        "request-id",
        "openai-request-id",
    ] {
        for value in headers.get_all(name) {
            sanitized.append(http::HeaderName::from_static(name), value.clone());
        }
    }
    *headers = sanitized;
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::{SocketAddr, TcpListener, TcpStream},
        thread::JoinHandle,
    };

    use super::*;

    fn bind_test_server() -> (TcpListener, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        (listener, addr)
    }

    fn capture_request(listener: TcpListener, response_body: &str) -> JoinHandle<String> {
        let body = response_body.to_owned();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_request(&mut stream);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8(request).unwrap()
        })
    }

    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buf = [0_u8; 4096];

        loop {
            let n = stream.read(&mut buf).unwrap();
            assert!(n > 0, "connection closed before the complete request arrived");
            request.extend_from_slice(&buf[..n]);

            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
                continue;
            };
            let body_start = header_end + 4;
            let headers = std::str::from_utf8(&request[..header_end]).unwrap();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);

            if request.len() >= body_start + content_length {
                return request;
            }
        }
    }

    use praxis_core::subrequest::SubRequestConnector;

    fn test_client(base_url: &str) -> ApiClient {
        ApiClient::new(ApiClientConfig {
            api_base_url: base_url.to_owned(),
            client: SubRequestClient::new(SubRequestConnector::new(4, None)),
            timeout: Duration::from_millis(1_000),
            max_response_bytes: 1_048_576,
            forward_header_names: Vec::new(),
        })
    }

    #[test]
    fn new_strips_trailing_slash() {
        let client = test_client("http://ogx:8321/");
        assert_eq!(client.api_base_url(), "http://ogx:8321");
    }

    #[test]
    fn forward_headers_copies_configured_headers() {
        let client = ApiClient::new(ApiClientConfig {
            api_base_url: "http://ogx:8321".to_owned(),
            client: SubRequestClient::new(SubRequestConnector::new(4, None)),
            timeout: Duration::from_millis(1_000),
            max_response_bytes: 1_048_576,
            forward_header_names: vec![
                http::header::AUTHORIZATION,
                http::HeaderName::from_static("x-tenant-id"),
            ],
        });

        let mut request_headers = HeaderMap::new();
        request_headers.insert(http::header::AUTHORIZATION, "Bearer token".parse().unwrap());
        request_headers.insert("x-tenant-id", "tenant-1".parse().unwrap());
        request_headers.insert("x-unrelated", "ignored".parse().unwrap());

        let forwarded = client.forward_headers(&request_headers);

        assert_eq!(forwarded.len(), 2, "only configured headers should be forwarded");
        assert!(
            forwarded
                .iter()
                .any(|(n, v)| n == "authorization" && v == "Bearer token"),
            "authorization header should be forwarded"
        );
        assert!(
            forwarded.iter().any(|(n, v)| n == "x-tenant-id" && v == "tenant-1"),
            "x-tenant-id header should be forwarded"
        );
    }

    #[test]
    fn resource_url_delegates_to_url_module() {
        let client = test_client("http://ogx:8321");
        let url = client.resource_url("v1/files", "file-abc", Some("content")).unwrap();
        assert_eq!(url, "http://ogx:8321/v1/files/file-abc/content");
    }

    #[tokio::test]
    async fn get_bytes_preserves_redirect_without_following_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let body = client
            .get_bytes(
                &format!("http://{address}/v1/files/test/content"),
                &HeaderMap::new(),
                1024,
            )
            .await
            .unwrap();

        assert!(body.is_empty(), "redirect response should not contact its target");
    }

    #[tokio::test]
    async fn get_bytes_transport_failure_returns_callout_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            drop(stream);
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .get_bytes(
                &format!("http://{address}/v1/files/test/content"),
                &HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::Transport { .. }),
            "transport errors should remain typed"
        );
    }

    #[tokio::test]
    async fn get_bytes_rejects_response_exceeding_per_request_limit() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n0123456789abcdef")
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .get_bytes(&format!("http://{address}/v1/files/test/content"), &HeaderMap::new(), 8)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::ResponseTooLarge { .. }),
            "responses exceeding per-request max_bytes should be rejected as ResponseTooLarge: {err:?}"
        );
    }

    #[tokio::test]
    async fn get_bytes_oversized_non_2xx_is_response_too_large() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            let body = vec![b'x'; 64];
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .get_bytes(&format!("http://{address}/v1/files/test/content"), &HeaderMap::new(), 8)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::ResponseTooLarge { .. }),
            "oversized response body should be ResponseTooLarge regardless of status: {err:?}"
        );
    }

    #[tokio::test]
    async fn get_returns_valid_json_without_decoding() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            let body = r#"{"id":"file-abc","content_type":"text/plain"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let response = client
            .get(&format!("http://{address}/v1/files/file-abc"), &HeaderMap::new(), 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&response.body).unwrap();

        assert_eq!(json["id"].as_str().unwrap(), "file-abc");
        assert_eq!(json["content_type"].as_str().unwrap(), "text/plain");
    }

    #[tokio::test]
    async fn get_preserves_invalid_json_without_decoding() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nnot-json!!!")
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let response = client
            .get(&format!("http://{address}/v1/files/file-abc"), &HeaderMap::new(), 1024)
            .await
            .unwrap();

        assert_eq!(response.body, "not-json!!!");
    }

    #[tokio::test]
    async fn post_json_bytes_sends_body_and_preserves_response() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, r#"{"results":[]}"#);
        let client = test_client(&format!("http://{address}"));

        let response = client
            .post_json_bytes(
                format!("http://{address}/v1/vector_stores/vs-123/search"),
                br#"{"query":"test"}"#.to_vec(),
                &HeaderMap::new(),
            )
            .await
            .unwrap();

        assert_eq!(response.body, br#"{"results":[]}"#.as_slice());

        let request = captured.join().unwrap();
        let request_lower = request.to_lowercase();
        assert!(request.starts_with("POST"), "should be a POST request");
        assert!(
            request_lower.contains("content-type: application/json"),
            "should have JSON content-type: {request}"
        );
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        assert_eq!(body, r#"{"query":"test"}"#, "serialized JSON body should be sent");
    }

    #[tokio::test]
    async fn post_json_bytes_preserves_invalid_json() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, "not-json!!!");
        let client = test_client(&format!("http://{address}"));

        let response = client
            .post_json_bytes(
                format!("http://{address}/v1/vector_stores/vs-123/search"),
                br#"{"query":"test"}"#.to_vec(),
                &HeaderMap::new(),
            )
            .await
            .unwrap();

        captured.join().unwrap();
        assert_eq!(response.body, "not-json!!!");
    }

    #[tokio::test]
    async fn post_json_strips_forwarded_content_type() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, r#"{"ok":true}"#);

        let client = ApiClient::new(ApiClientConfig {
            api_base_url: format!("http://{address}"),
            client: SubRequestClient::new(SubRequestConnector::new(4, None)),
            timeout: Duration::from_millis(1_000),
            max_response_bytes: 1_048_576,
            forward_header_names: vec![http::header::CONTENT_TYPE],
        });

        let mut headers = HeaderMap::new();
        headers.insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());

        client
            .post_json_bytes(format!("http://{address}/v1/search"), b"{}".to_vec(), &headers)
            .await
            .unwrap();

        let req = captured.join().unwrap();
        let req_lower = req.to_lowercase();
        let ct_count = req_lower.matches("content-type:").count();
        assert_eq!(ct_count, 1, "exactly one content-type header, got {ct_count}");
        assert!(
            req_lower.contains("content-type: application/json"),
            "should be application/json: {req}"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the status matrix keeps every response-preservation case visible"
    )]
    async fn get_preserves_valid_http_statuses_and_bounded_bodies() {
        for (status, location) in [
            (301_u16, Some("https://example.invalid/redirect")),
            (302_u16, Some("https://example.invalid/redirect")),
            (401_u16, None),
            (403_u16, None),
            (404_u16, None),
            (429_u16, None),
            (500_u16, None),
            (503_u16, None),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let body = format!(r#"{{"error":{{"message":"status-{status}"}}}}"#);
            let response_body = body.clone();
            std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 4096];
                let _read = stream.read(&mut request).unwrap();
                let location = location.map_or_else(String::new, |value| format!("Location: {value}\r\n"));
                let response = format!(
                    "HTTP/1.1 {status} Status\r\n{location}Content-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            });
            let client = test_client(&format!("http://{address}"));

            let response = client
                .get(&format!("http://{address}/v1/files/file-abc"), &HeaderMap::new(), 1024)
                .await
                .unwrap();

            assert_eq!(response.status, status, "status {status} should be preserved");
            assert_eq!(response.body, body, "status {status} body should be preserved");
            assert!(
                response.headers.get(http::header::LOCATION).is_none(),
                "Location must not be exposed"
            );
        }
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the header policy test lists the allowed and rejected boundary values"
    )]
    async fn get_exposes_only_safe_response_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nRetry-After: 30\r\nX-Request-Id: req_123\r\nConnection: x-remove-me\r\nX-Remove-Me: should-not-pass\r\nX-Praxis-Private: should-not-pass\r\nSet-Cookie: session=secret\r\nCookie: session=secret\r\nAuthorization: Bearer secret\r\nX-Api-Key: secret\r\nX-Auth-Token: secret\r\nX-Unknown-Provider: should-not-pass\r\nContent-Length: 2\r\n\r\n{}",
                )
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let response = client
            .get(&format!("http://{address}/v1/files/file-abc"), &HeaderMap::new(), 1024)
            .await
            .unwrap();

        assert_eq!(response.headers[http::header::CONTENT_TYPE], "application/json");
        assert_eq!(response.headers[http::header::RETRY_AFTER], "30");
        assert_eq!(response.headers["x-request-id"], "req_123");
        for name in [
            "connection",
            "x-remove-me",
            "x-praxis-private",
            "set-cookie",
            "cookie",
            "authorization",
            "x-api-key",
            "x-auth-token",
            "x-unknown-provider",
        ] {
            assert!(response.headers.get(name).is_none(), "{name} must not be exposed");
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "each typed transport variant needs an independent source-redaction assertion"
    )]
    fn transport_errors_preserve_kind_without_rendering_source_details() {
        let connect = map_subrequest_error(SubRequestError::Connect("attacker-controlled".to_owned()));
        assert!(matches!(
            connect,
            ApiClientError::Transport {
                source: SubRequestError::Connect(_)
            }
        ));
        assert!(!connect.to_string().contains("attacker-controlled"));

        let io = map_subrequest_error(SubRequestError::Io("attacker-controlled".to_owned()));
        assert!(matches!(
            io,
            ApiClientError::Transport {
                source: SubRequestError::Io(_)
            }
        ));
        assert!(!io.to_string().contains("attacker-controlled"));

        let admission = map_subrequest_error(SubRequestError::AdmissionTimeout { max_connections: 1 });
        assert!(matches!(
            admission,
            ApiClientError::Transport {
                source: SubRequestError::AdmissionTimeout { .. }
            }
        ));

        let circuit = map_subrequest_error(SubRequestError::CircuitOpen {
            peer: "attacker-controlled".to_owned(),
        });
        assert!(matches!(
            circuit,
            ApiClientError::Transport {
                source: SubRequestError::CircuitOpen { .. }
            }
        ));
        assert!(!circuit.to_string().contains("attacker-controlled"));

        let deadline = map_subrequest_error(SubRequestError::DeadlineExceeded);
        assert!(matches!(
            deadline,
            ApiClientError::Transport {
                source: SubRequestError::DeadlineExceeded
            }
        ));
    }

    #[test]
    fn display_invalid_resource_id() {
        let err = ApiClientError::InvalidResourceId {
            resource_id: "../etc/passwd".to_owned(),
            detail: "path traversal".to_owned(),
        };
        assert_eq!(err.to_string(), "invalid resource id '../etc/passwd': path traversal");
    }

    #[test]
    fn display_response_too_large() {
        let err = ApiClientError::ResponseTooLarge { limit: 1024 };
        assert_eq!(err.to_string(), "response exceeds size limit (1024 bytes)");
    }

    #[tokio::test]
    async fn get_bytes_above_one_mib_succeeds() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let payload_size: usize = 1_200_000;
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            let body = vec![0x42_u8; payload_size];
            let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {payload_size}\r\nConnection: close\r\n\r\n");
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let bytes = client
            .get_bytes(
                &format!("http://{address}/v1/files/big/content"),
                &HeaderMap::new(),
                2_000_000,
            )
            .await
            .unwrap();

        assert_eq!(bytes.len(), payload_size, "should receive full >1 MiB payload");
    }

    fn slow_body_server(listener: TcpListener) {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _n = stream.read(&mut buf).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\na")
                .unwrap();
            stream.flush().unwrap();
            std::thread::park_timeout(Duration::from_millis(250));
            let _result = stream.write_all(b"bcde");
        });
    }

    #[tokio::test]
    async fn get_bytes_timeout_covers_response_body() {
        let (listener, addr) = bind_test_server();
        slow_body_server(listener);

        let client = ApiClient::new(ApiClientConfig {
            api_base_url: format!("http://{addr}"),
            client: SubRequestClient::new(SubRequestConnector::new(4, None)),
            timeout: Duration::from_millis(50),
            max_response_bytes: 1_048_576,
            forward_header_names: Vec::new(),
        });

        let err = client
            .get_bytes(&format!("http://{addr}/v1/files/slow/content"), &HeaderMap::new(), 1024)
            .await
            .unwrap_err();

        assert!(
            matches!(
                &err,
                ApiClientError::Transport {
                    source: SubRequestError::DeadlineExceeded
                }
            ),
            "slow body should fail before completing: {err}"
        );
    }
}
