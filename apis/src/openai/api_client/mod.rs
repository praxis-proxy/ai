// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared HTTP client for OpenAI-compatible API callouts.
//!
//! Provides URL construction, SSRF-safe base-URL validation,
//! resource-ID path-segment encoding, header forwarding, bounded
//! JSON and byte reads, and normalized error mapping. Used by
//! [`FilesApiClient`] and vector-store search.
//!
//! All requests route through `praxis_core::callout::CalloutClient`
//! for circuit breaking, callout-depth protection, timeout, and
//! bounded reads via `CalloutConfig::max_response_bytes`. Content
//! downloads (`get_bytes`) apply a tighter per-request `max_bytes`
//! check post-collection so callers can enforce file-level budgets.
//!
//! Each consuming filter retains its own [`ApiClient`] instance so
//! circuit-breaker state is isolated per filter.
//!
//! [`FilesApiClient`]: super::responses::file_resolve

pub(crate) mod error;
pub(crate) mod url;

use std::time::Duration;

use praxis_core::callout::{CalloutClient, CalloutConfig, CalloutRequest, CalloutResult};

pub(crate) use self::{
    error::ApiClientError,
    url::{resource_url, validate_base_url, validate_forward_headers},
};

/// Configuration for constructing an [`ApiClient`].
///
/// Assembled programmatically by each consuming filter from its
/// own validated YAML config — no shared YAML schema.
pub(crate) struct ApiClientConfig {
    /// Base URL of the API endpoint (trailing slash stripped).
    pub api_base_url: String,
    /// Transport configuration for the underlying
    /// [`CalloutClient`].
    pub callout_config: CalloutConfig,
    /// Header names to forward from the original request.
    pub forward_header_names: Vec<http::HeaderName>,
}

/// Shared HTTP client for OpenAI-compatible API callouts.
///
/// All requests route through [`CalloutClient`] for circuit
/// breaking, callout-depth protection, and bounded reads via
/// [`CalloutConfig::max_response_bytes`].
///
/// Content downloads (`get_bytes`) apply a tighter per-request
/// `max_bytes` check after collection to return
/// [`ApiClientError::ResponseTooLarge`]. The client-wide
/// `max_response_bytes` caps memory during collection; the
/// per-request limit lets callers enforce stricter file-level
/// budgets.
pub(crate) struct ApiClient {
    /// Base URL of the API endpoint (trailing slash stripped).
    api_base_url: String,
    /// Callout client shared by all request paths.
    client: CalloutClient,
    /// Header names to forward from the original downstream request.
    forward_header_names: Vec<http::HeaderName>,
    /// Per-request timeout (from the callout config).
    timeout: Duration,
}

impl ApiClient {
    /// Build a new client from validated configuration.
    ///
    /// The base URL should already be validated with
    /// [`validate_base_url`]. This constructor strips trailing
    /// slashes and builds the transport client.
    ///
    /// # Errors
    ///
    /// Returns an error if the callout client cannot be
    /// constructed.
    pub(crate) fn new(config: ApiClientConfig) -> Result<Self, String> {
        let ApiClientConfig {
            api_base_url,
            callout_config,
            forward_header_names,
        } = config;

        let timeout = Duration::from_millis(callout_config.timeout_ms);

        let client = CalloutClient::new(callout_config).map_err(|e| format!("failed to build callout client: {e}"))?;

        Ok(Self {
            api_base_url: api_base_url.trim_end_matches('/').to_owned(),
            client,
            forward_header_names,
            timeout,
        })
    }

    /// Return the validated base URL.
    pub(crate) fn api_base_url(&self) -> &str {
        &self.api_base_url
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

    /// Send a GET request via the callout client and parse the
    /// response body as JSON.
    pub(crate) async fn get_json(
        &self,
        url: String,
        request_headers: &http::HeaderMap,
    ) -> Result<serde_json::Value, ApiClientError> {
        let headers = self.forward_headers(request_headers);
        let request = CalloutRequest {
            body: None,
            depth: 0,
            headers,
            method: http::Method::GET,
            url,
        };

        let response = execute_callout(&self.client, request).await?;
        serde_json::from_slice(&response.body).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("JSON decode failed: {e}"),
        })
    }

    /// Send a POST request with a JSON body via the callout client
    /// and parse the response body as JSON.
    pub(crate) async fn post_json(
        &self,
        url: String,
        body: &serde_json::Value,
        request_headers: &http::HeaderMap,
    ) -> Result<serde_json::Value, ApiClientError> {
        let serialized = serde_json::to_vec(body).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("request body serialization failed: {e}"),
        })?;

        let response = self.post_json_bytes(url, serialized, request_headers).await?;

        serde_json::from_slice(&response).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("JSON decode failed: {e}"),
        })
    }

    /// Send a pre-serialized JSON body and return the bounded raw response.
    ///
    /// This supports consumers that enforce domain-specific serialization and
    /// decode limits while retaining the shared transport, header, timeout,
    /// circuit-breaker, and status handling.
    pub(crate) async fn post_json_bytes(
        &self,
        url: String,
        body: Vec<u8>,
        request_headers: &http::HeaderMap,
    ) -> Result<Vec<u8>, ApiClientError> {
        let mut headers = self.forward_headers(request_headers);
        headers.retain(|(name, _)| name != http::header::CONTENT_TYPE);
        headers.push((
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        ));

        let request = CalloutRequest {
            body: Some(body),
            depth: 0,
            headers,
            method: http::Method::POST,
            url,
        };

        let response = execute_callout(&self.client, request).await?;
        Ok(response.body)
    }

    /// Send a GET request via the callout client and return the
    /// response body, rejecting it when it exceeds `max_bytes`.
    ///
    /// Memory during collection is bounded by
    /// [`CalloutConfig::max_response_bytes`] (the client-wide
    /// limit). The per-request `max_bytes` is checked
    /// post-collection so callers can enforce tighter file-level
    /// budgets while still returning
    /// [`ApiClientError::ResponseTooLarge`].
    ///
    /// The configured timeout spans the full request including
    /// body transfer.
    pub(crate) async fn get_bytes(
        &self,
        url: &str,
        request_headers: &http::HeaderMap,
        max_bytes: usize,
    ) -> Result<Vec<u8>, ApiClientError> {
        let headers = self.forward_headers(request_headers);
        let request = CalloutRequest {
            body: None,
            depth: 0,
            headers,
            method: http::Method::GET,
            url: url.to_owned(),
        };

        let response = tokio::time::timeout(self.timeout, execute_callout(&self.client, request))
            .await
            .map_err(|_elapsed| ApiClientError::CalloutFailed {
                detail: "content download timed out".to_owned(),
            })??;

        if response.body.len() > max_bytes {
            return Err(ApiClientError::ResponseTooLarge { limit: max_bytes });
        }

        Ok(response.body)
    }

    /// Copy configured headers from the original downstream
    /// request for forwarding to the external API.
    pub(crate) fn forward_headers(
        &self,
        request_headers: &http::HeaderMap,
    ) -> Vec<(http::HeaderName, http::HeaderValue)> {
        let mut headers = Vec::new();
        for name in &self.forward_header_names {
            if let Some(value) = request_headers.get(name) {
                headers.push((name.clone(), value.clone()));
            }
        }
        headers
    }
}

/// Execute a callout and map non-success outcomes to
/// [`ApiClientError`].
async fn execute_callout(
    client: &CalloutClient,
    request: CalloutRequest,
) -> Result<praxis_core::callout::CalloutResponse, ApiClientError> {
    match client.execute(request).await {
        CalloutResult::Success(r) => Ok(r),
        CalloutResult::Failed => Err(ApiClientError::CalloutFailed {
            detail: "callout failed (fail-open)".to_owned(),
        }),
        CalloutResult::Rejected(rejection) => Err(ApiClientError::CalloutFailed {
            detail: format!("callout rejected with status {}", rejection.status),
        }),
    }
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

    use praxis_core::callout::{CalloutConfig, FailureMode};

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

    fn test_client(base_url: &str) -> ApiClient {
        test_client_with_limits(base_url, CalloutConfig::default().max_response_bytes, 1_000)
    }

    fn test_client_with_limits(base_url: &str, max_response_bytes: usize, timeout_ms: u64) -> ApiClient {
        ApiClient::new(ApiClientConfig {
            api_base_url: base_url.to_owned(),
            callout_config: CalloutConfig {
                failure_mode: FailureMode::Closed,
                max_response_bytes,
                timeout_ms,
                ..CalloutConfig::default()
            },
            forward_header_names: Vec::new(),
        })
        .unwrap()
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
            callout_config: CalloutConfig {
                failure_mode: FailureMode::Closed,
                ..CalloutConfig::default()
            },
            forward_header_names: vec![
                http::header::AUTHORIZATION,
                http::HeaderName::from_static("x-tenant-id"),
            ],
        })
        .unwrap();

        let mut request_headers = http::HeaderMap::new();
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
    async fn get_bytes_does_not_follow_redirects() {
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

        let err = client
            .get_bytes(
                &format!("http://{address}/v1/files/test/content"),
                &http::HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "redirect response should be rejected without contacting its target"
        );
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
                &http::HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "transport errors should be mapped to CalloutFailed"
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
            .get_bytes(
                &format!("http://{address}/v1/files/test/content"),
                &http::HeaderMap::new(),
                8,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::ResponseTooLarge { .. }),
            "responses exceeding per-request max_bytes should be rejected"
        );
    }

    #[tokio::test]
    async fn get_json_parses_valid_json() {
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

        let json = client
            .get_json(format!("http://{address}/v1/files/file-abc"), &http::HeaderMap::new())
            .await
            .unwrap();

        assert_eq!(json["id"].as_str().unwrap(), "file-abc");
        assert_eq!(json["content_type"].as_str().unwrap(), "text/plain");
    }

    #[tokio::test]
    async fn get_json_returns_decode_error_on_invalid_json() {
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

        let err = client
            .get_json(format!("http://{address}/v1/files/file-abc"), &http::HeaderMap::new())
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::DecodeFailed { .. }),
            "invalid JSON should return a decode error"
        );
    }

    #[tokio::test]
    async fn post_json_sends_body_and_parses_response() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, r#"{"results":[]}"#);
        let client = test_client(&format!("http://{address}"));

        let request_body = serde_json::json!({"query": "test"});
        let json = client
            .post_json(
                format!("http://{address}/v1/vector_stores/vs-123/search"),
                &request_body,
                &http::HeaderMap::new(),
            )
            .await
            .unwrap();

        assert!(json["results"].as_array().unwrap().is_empty());

        let request = captured.join().unwrap();
        assert!(request.starts_with("POST"), "should be a POST request");
        assert!(
            request.contains("content-type: application/json"),
            "should have JSON content-type: {request}"
        );
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        assert_eq!(body, r#"{"query":"test"}"#, "serialized JSON body should be sent");
    }

    #[tokio::test]
    async fn post_json_returns_decode_error_on_invalid_json() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, "not-json!!!");
        let client = test_client(&format!("http://{address}"));

        let err = client
            .post_json(
                format!("http://{address}/v1/vector_stores/vs-123/search"),
                &serde_json::json!({"query": "test"}),
                &http::HeaderMap::new(),
            )
            .await
            .unwrap_err();

        captured.join().unwrap();
        assert!(
            matches!(err, ApiClientError::DecodeFailed { .. }),
            "invalid JSON should return a decode error"
        );
    }

    #[tokio::test]
    async fn post_json_strips_forwarded_content_type() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, r#"{"ok":true}"#);

        let client = ApiClient::new(ApiClientConfig {
            api_base_url: format!("http://{address}"),
            callout_config: CalloutConfig {
                failure_mode: FailureMode::Closed,
                timeout_ms: 1_000,
                ..CalloutConfig::default()
            },
            forward_header_names: vec![http::header::CONTENT_TYPE],
        })
        .unwrap();

        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());

        client
            .post_json(format!("http://{address}/v1/search"), &serde_json::json!({}), &headers)
            .await
            .unwrap();

        let req = captured.join().unwrap();
        let ct_count = req.matches("content-type:").count();
        assert_eq!(ct_count, 1, "exactly one content-type header, got {ct_count}");
        assert!(
            req.contains("content-type: application/json"),
            "should be application/json"
        );
    }

    #[tokio::test]
    async fn get_json_non_2xx_returns_callout_failed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            let body = r#"{"error":"not found"}"#;
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .get_json(format!("http://{address}/v1/files/missing"), &http::HeaderMap::new())
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "non-2xx JSON response should map to CalloutFailed"
        );
    }

    #[tokio::test]
    async fn get_bytes_non_2xx_returns_callout_failed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        });
        let client = test_client(&format!("http://{address}"));

        let err = client
            .get_bytes(
                &format!("http://{address}/v1/files/test/content"),
                &http::HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "non-2xx byte download should map to CalloutFailed via callout client"
        );
    }

    #[test]
    fn display_callout_failed() {
        let err = ApiClientError::CalloutFailed {
            detail: "connection refused".to_owned(),
        };
        assert_eq!(err.to_string(), "API callout failed: connection refused");
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

    #[test]
    fn display_decode_failed() {
        let err = ApiClientError::DecodeFailed {
            detail: "expected value at line 1".to_owned(),
        };
        assert_eq!(err.to_string(), "response decode failed: expected value at line 1");
    }

    #[tokio::test]
    async fn get_bytes_above_one_mib_succeeds() {
        let (listener, address) = bind_test_server();
        let payload_size: usize = 1_200_000;
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _read = stream.read(&mut buf).unwrap();
            let body = vec![0x42_u8; payload_size];
            let header = format!("HTTP/1.1 200 OK\r\nContent-Length: {payload_size}\r\nConnection: close\r\n\r\n");
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });
        let client = test_client_with_limits(&format!("http://{address}"), 2_000_000, 5_000);

        let bytes = client
            .get_bytes(
                &format!("http://{address}/v1/files/big/content"),
                &http::HeaderMap::new(),
                2_000_000,
            )
            .await
            .unwrap();

        assert_eq!(bytes.len(), payload_size, "should receive full >1 MiB payload");
    }

    #[tokio::test]
    async fn get_bytes_client_wide_limit_rejects_oversized_chunked_response() {
        let (listener, address) = bind_test_server();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _read = stream.read(&mut buf).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.write_all(&vec![0x41_u8; 256]).unwrap();
        });
        let client = test_client_with_limits(&format!("http://{address}"), 64, 1_000);

        let err = client
            .get_bytes(
                &format!("http://{address}/v1/files/huge/content"),
                &http::HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "response exceeding client-wide max_response_bytes should be rejected by callout client"
        );
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
        let client = test_client_with_limits(&format!("http://{addr}"), 1024, 50);

        let err = client
            .get_bytes(
                &format!("http://{addr}/v1/files/slow/content"),
                &http::HeaderMap::new(),
                1024,
            )
            .await
            .unwrap_err();

        assert!(
            matches!(&err, ApiClientError::CalloutFailed { .. }),
            "slow body should fail before completing: {err}"
        );
    }
}
