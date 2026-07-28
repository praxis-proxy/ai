// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared HTTP client for OpenAI-compatible API callouts.
//!
//! Provides URL construction, SSRF-safe base-URL validation,
//! resource-ID path-segment encoding, header forwarding, bounded
//! JSON and byte reads, and normalized error mapping. Used by
//! [`FilesApiClient`] and vector-store search.
//!
//! All requests route through the Pingora-native
//! [`SubRequestConnector`] for connection pooling and TLS.
//!
//! Each consuming filter retains its own [`ApiClient`] instance.
//!
//! [`FilesApiClient`]: super::responses::file_resolve
//! [`SubRequestConnector`]: praxis_core::subrequest::SubRequestConnector

pub(crate) mod error;
pub(crate) mod url;

use std::time::Duration;

use bytes::Bytes;
use http::HeaderMap;

pub(crate) use self::{
    error::ApiClientError,
    url::{resource_url, validate_base_url, validate_forward_headers},
};
use crate::subrequest::{self, SubRequest, SubRequestConnector, SubRequestError, SubResponse};

/// Configuration for constructing an [`ApiClient`].
///
/// Assembled programmatically by each consuming filter from its
/// own validated YAML config — no shared YAML schema.
pub(crate) struct ApiClientConfig {
    /// Base URL of the API endpoint (trailing slash stripped).
    pub api_base_url: String,
    /// Pingora-native HTTP connector.
    pub connector: SubRequestConnector,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Maximum response body bytes.
    pub max_response_bytes: usize,
    /// Header names to forward from the original request.
    pub forward_header_names: Vec<http::HeaderName>,
}

/// Shared HTTP client for OpenAI-compatible API callouts.
///
/// All requests route through the Pingora-native
/// [`SubRequestConnector`] for connection pooling and TLS.
///
/// [`SubRequestConnector`]: praxis_core::subrequest::SubRequestConnector
pub(crate) struct ApiClient {
    /// Base URL of the API endpoint (trailing slash stripped).
    api_base_url: String,
    /// Pingora-native HTTP connector.
    connector: SubRequestConnector,
    /// Per-request timeout.
    timeout: Duration,
    /// Maximum response body bytes for JSON requests.
    max_response_bytes: usize,
    /// Header names to forward from the original downstream
    /// request.
    forward_header_names: Vec<http::HeaderName>,
}

/// Map a [`SubRequestError`] to an [`ApiClientError`], preserving
/// the `ResponseTooLarge` variant for 2xx responses so callers can
/// distinguish genuine size violations from backend failures.
///
/// Non-2xx oversized responses map to `CalloutFailed` — the
/// backend error takes precedence over the size limit.
fn map_subrequest_error(err: SubRequestError) -> ApiClientError {
    match err {
        SubRequestError::ResponseTooLarge { status, .. } if !(200..300).contains(&(status as usize)) => {
            ApiClientError::CalloutFailed {
                detail: format!("callout rejected with status {status} (response body exceeded limit)"),
            }
        },
        SubRequestError::ResponseTooLarge { limit, .. } => ApiClientError::ResponseTooLarge { limit },
        other => ApiClientError::CalloutFailed {
            detail: other.to_string(),
        },
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
            connector,
            timeout,
            max_response_bytes,
            forward_header_names,
        } = config;

        Self {
            api_base_url: api_base_url.trim_end_matches('/').to_owned(),
            connector,
            timeout,
            max_response_bytes,
            forward_header_names,
        }
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

    /// Send a GET request and parse the response body as JSON.
    pub(crate) async fn get_json(
        &self,
        url: String,
        request_headers: &HeaderMap,
    ) -> Result<serde_json::Value, ApiClientError> {
        let headers = self.build_header_map(request_headers);
        let response = self.execute_url(&url, http::Method::GET, headers, Bytes::new()).await?;
        serde_json::from_slice(&response.body).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("JSON decode failed: {e}"),
        })
    }

    /// Send a POST request with a JSON body and parse the response
    /// body as JSON.
    pub(crate) async fn post_json(
        &self,
        url: String,
        body: &serde_json::Value,
        request_headers: &HeaderMap,
    ) -> Result<serde_json::Value, ApiClientError> {
        let serialized = serde_json::to_vec(body).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("request body serialization failed: {e}"),
        })?;

        let response = self.post_json_bytes(url, serialized, request_headers).await?;

        serde_json::from_slice(&response).map_err(|e| ApiClientError::DecodeFailed {
            detail: format!("JSON decode failed: {e}"),
        })
    }

    /// Send a pre-serialized JSON body and return the bounded raw
    /// response.
    pub(crate) async fn post_json_bytes(
        &self,
        url: String,
        body: Vec<u8>,
        request_headers: &HeaderMap,
    ) -> Result<Bytes, ApiClientError> {
        let mut headers = self.build_header_map(request_headers);
        headers.remove(http::header::CONTENT_TYPE);
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let response = self
            .execute_url(&url, http::Method::POST, headers, Bytes::from(body))
            .await?;
        Ok(response.body)
    }

    /// Send a GET request and return the response body with
    /// bounded reads.
    ///
    /// The Pingora connector does not follow redirects (it
    /// connects to a specific peer), matching the redirect-
    /// rejection behavior of the previous `reqwest` path.
    pub(crate) async fn get_bytes(
        &self,
        url: &str,
        request_headers: &HeaderMap,
        max_bytes: usize,
    ) -> Result<Bytes, ApiClientError> {
        let headers = self.build_header_map(request_headers);

        let (target, uri) = subrequest::parse_url(url).map_err(|e| ApiClientError::CalloutFailed {
            detail: format!("content download failed: {e}"),
        })?;

        let request = SubRequest {
            method: http::Method::GET,
            uri,
            headers,
            body: Bytes::new(),
        };

        let response = subrequest::execute(&self.connector, &target, &request, max_bytes, self.timeout)
            .await
            .map_err(map_subrequest_error)?;

        if response.status < 200 || response.status >= 300 {
            return Err(ApiClientError::CalloutFailed {
                detail: format!("content download failed: {}", response.status),
            });
        }

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

    /// Parse the URL, build a [`SubRequest`], execute via the
    /// connector, and check for non-2xx status.
    async fn execute_url(
        &self,
        url: &str,
        method: http::Method,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<SubResponse, ApiClientError> {
        let (target, uri) =
            subrequest::parse_url(url).map_err(|e| ApiClientError::CalloutFailed { detail: e.to_string() })?;

        let request = SubRequest {
            method,
            uri,
            headers,
            body,
        };

        let response = subrequest::execute(
            &self.connector,
            &target,
            &request,
            self.max_response_bytes,
            self.timeout,
        )
        .await
        .map_err(map_subrequest_error)?;

        if response.status < 200 || response.status >= 300 {
            return Err(ApiClientError::CalloutFailed {
                detail: format!("callout rejected with status {}", response.status),
            });
        }

        Ok(response)
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
        ApiClient::new(ApiClientConfig {
            api_base_url: base_url.to_owned(),
            connector: SubRequestConnector::new(4, None),
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
            connector: SubRequestConnector::new(4, None),
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
                &HeaderMap::new(),
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
                &HeaderMap::new(),
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
            .get_bytes(&format!("http://{address}/v1/files/test/content"), &HeaderMap::new(), 8)
            .await
            .unwrap_err();

        assert!(
            matches!(err, ApiClientError::ResponseTooLarge { .. }),
            "responses exceeding per-request max_bytes should be rejected as ResponseTooLarge: {err:?}"
        );
    }

    #[tokio::test]
    async fn get_bytes_oversized_non_2xx_is_callout_failed_not_too_large() {
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
            matches!(err, ApiClientError::CalloutFailed { .. }),
            "non-2xx oversized response should be CalloutFailed, not ResponseTooLarge: {err:?}"
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
            .get_json(format!("http://{address}/v1/files/file-abc"), &HeaderMap::new())
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
            .get_json(format!("http://{address}/v1/files/file-abc"), &HeaderMap::new())
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
                &HeaderMap::new(),
            )
            .await
            .unwrap();

        assert!(json["results"].as_array().unwrap().is_empty());

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
    async fn post_json_returns_decode_error_on_invalid_json() {
        let (listener, address) = bind_test_server();
        let captured = capture_request(listener, "not-json!!!");
        let client = test_client(&format!("http://{address}"));

        let err = client
            .post_json(
                format!("http://{address}/v1/vector_stores/vs-123/search"),
                &serde_json::json!({"query": "test"}),
                &HeaderMap::new(),
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
            connector: SubRequestConnector::new(4, None),
            timeout: Duration::from_millis(1_000),
            max_response_bytes: 1_048_576,
            forward_header_names: vec![http::header::CONTENT_TYPE],
        });

        let mut headers = HeaderMap::new();
        headers.insert(http::header::CONTENT_TYPE, "text/plain".parse().unwrap());

        client
            .post_json(format!("http://{address}/v1/search"), &serde_json::json!({}), &headers)
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
            .get_json(format!("http://{address}/v1/files/missing"), &HeaderMap::new())
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
                &HeaderMap::new(),
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
            connector: SubRequestConnector::new(4, None),
            timeout: Duration::from_millis(50),
            max_response_bytes: 1_048_576,
            forward_header_names: Vec::new(),
        });

        let err = client
            .get_bytes(&format!("http://{addr}/v1/files/slow/content"), &HeaderMap::new(), 1024)
            .await
            .unwrap_err();

        assert!(
            matches!(&err, ApiClientError::CalloutFailed { .. }),
            "slow body should fail before completing: {err}"
        );
    }
}
