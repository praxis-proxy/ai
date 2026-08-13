// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! URL resolution and bounded execution for sub-requests.
//!
//! Provides [`execute_url`], which preserves the URL authority for
//! HTTP virtual hosting, resolves every address for fallback, and
//! bounds DNS plus the HTTP exchange with one overall deadline.
//!
//! Types ([`SubRequestClient`], [`SubRequest`], [`SubResponse`],
//! [`SubRequestError`]) are re-exported from [`praxis_core::subrequest`].

use std::{future::Future, net::SocketAddr, time::Duration};

use pingora_core::upstreams::peer::HttpPeer;
pub(crate) use praxis_core::subrequest::{SubRequest, SubRequestClient, SubRequestError, SubResponse};
use tracing::debug;

/// Parsed URL components needed to resolve and execute a request.
#[derive(Debug)]
struct ParsedUrl {
    /// Whether to establish a TLS connection.
    tls: bool,
    /// DNS hostname or literal address.
    host: String,
    /// Destination TCP port.
    port: u16,
    /// TLS server name, empty for cleartext HTTP.
    sni: String,
    /// Original URL authority for HTTP virtual hosting.
    authority: http::HeaderValue,
    /// Path and query sent to the upstream.
    uri: http::Uri,
}

/// Extract scheme, TLS flag, host, port, SNI, authority, and path.
#[expect(clippy::too_many_lines, reason = "sequential URL component extraction")]
fn parse_url_components(url: &str) -> Result<ParsedUrl, SubRequestError> {
    let parsed: http::Uri = url
        .parse()
        .map_err(|e| SubRequestError::InvalidRequest(format!("{e}: {url}")))?;

    let tls = match parsed.scheme_str().unwrap_or("http") {
        "https" => true,
        "http" => false,
        other => {
            return Err(SubRequestError::InvalidRequest(format!(
                "unsupported scheme '{other}': {url}"
            )));
        },
    };

    let authority = parsed
        .authority()
        .ok_or_else(|| SubRequestError::InvalidRequest(format!("missing host: {url}")))?;
    let host = authority.host().trim_start_matches('[').trim_end_matches(']');
    let port = authority.port_u16().unwrap_or(if tls { 443 } else { 80 });
    let sni = if tls { host.to_owned() } else { String::new() };
    let authority = http::HeaderValue::from_str(authority.as_str())
        .map_err(|e| SubRequestError::InvalidRequest(format!("invalid authority: {e}")))?;

    let path_and_query = parsed.path_and_query().map_or("/", |pq| pq.as_str());
    let uri: http::Uri = path_and_query
        .parse()
        .map_err(|e| SubRequestError::InvalidRequest(format!("bad path: {e}")))?;

    Ok(ParsedUrl {
        tls,
        host: host.to_owned(),
        port,
        sni,
        authority,
        uri,
    })
}

/// Resolve every address so callers can fall back across address families.
async fn resolve_addrs(host: &str, port: u16) -> Result<Vec<SocketAddr>, SubRequestError> {
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| SubRequestError::Connect(format!("DNS resolution failed for {host}: {e}")))?
        .collect::<Vec<_>>();

    if addrs.is_empty() {
        return Err(SubRequestError::Connect(format!("no addresses resolved for {host}")));
    }

    Ok(addrs)
}

/// Enforce the deadline around DNS resolution and every connection attempt.
async fn with_deadline<T>(
    timeout: Duration,
    operation: impl Future<Output = Result<T, SubRequestError>>,
) -> Result<T, SubRequestError> {
    tokio::time::timeout(timeout, operation)
        .await
        .map_err(|_elapsed| SubRequestError::DeadlineExceeded)?
}

/// Parse and execute a full-URL sub-request.
///
/// The configured timeout covers URL resolution and the complete HTTP
/// exchange. All resolved addresses are tried in order when connecting,
/// while the original URL authority is preserved in `Host`.
pub(crate) async fn execute_url(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
) -> Result<SubResponse, SubRequestError> {
    with_deadline(
        timeout,
        Box::pin(execute_url_inner(client, url, request, max_response_bytes, timeout)),
    )
    .await
}

/// Resolve DNS and execute under the deadline enforced by [`execute_url`].
async fn execute_url_inner(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
) -> Result<SubResponse, SubRequestError> {
    let parsed = parse_url_components(url)?;
    let addrs = resolve_addrs(&parsed.host, parsed.port).await?;
    execute_resolved_url(client, parsed, request, &addrs, max_response_bytes, timeout).await
}

/// Try each resolved address until one connects successfully.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper requires parsed target and execution limits"
)]
async fn execute_resolved_url(
    client: &SubRequestClient,
    parsed: ParsedUrl,
    mut request: SubRequest,
    addrs: &[SocketAddr],
    max_response_bytes: usize,
    timeout: Duration,
) -> Result<SubResponse, SubRequestError> {
    request.uri = parsed.uri;
    if !request.headers.contains_key(http::header::HOST) {
        request.headers.insert(http::header::HOST, parsed.authority);
    }

    let mut last_connect_error = None;
    for addr in addrs {
        let peer = HttpPeer::new(*addr, parsed.tls, parsed.sni.clone());
        debug!(host = %parsed.host, %addr, "sub-request: trying resolved address");

        match client.execute(&peer, &request, max_response_bytes, timeout, None).await {
            Ok(response) => return Ok(response),
            Err(SubRequestError::Connect(error)) => {
                debug!(host = %parsed.host, %addr, %error, "sub-request: connect failed, trying next address");
                last_connect_error = Some(error);
            },
            Err(error) => return Err(error),
        }
    }

    Err(SubRequestError::Connect(last_connect_error.map_or_else(
        || format!("no addresses resolved for {}", parsed.host),
        |error| format!("all resolved addresses for {} failed: {error}", parsed.host),
    )))
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
    use std::io::{Read as _, Write as _};

    use bytes::Bytes;
    use http::HeaderMap;
    use praxis_core::subrequest::SubRequestConnector;

    use super::*;

    fn test_client() -> SubRequestClient {
        SubRequestClient::new(SubRequestConnector::new(4, None))
    }

    fn empty_request() -> SubRequest {
        SubRequest {
            method: http::Method::GET,
            uri: http::Uri::default(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        }
    }

    fn capture_raw_request(listener: std::net::TcpListener) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let n = stream.read(&mut buf).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
            String::from_utf8_lossy(&buf[..n]).into_owned()
        })
    }

    #[test]
    fn parse_url_https() {
        let parsed = parse_url_components("https://127.0.0.1:8443/v1/search?q=test").unwrap();
        assert!(parsed.tls, "HTTPS should enable TLS");
        assert_eq!(parsed.port, 8443);
        assert_eq!(parsed.uri.path(), "/v1/search");
        assert_eq!(parsed.uri.query(), Some("q=test"));
    }

    #[test]
    fn parse_url_preserves_hostname_authority() {
        let parsed = parse_url_components("https://api.example.com:8443/v1/search").unwrap();
        assert_eq!(parsed.host, "api.example.com");
        assert_eq!(parsed.authority, "api.example.com:8443");
        assert_eq!(parsed.sni, "api.example.com");
    }

    #[test]
    fn parse_url_ipv6_loopback() {
        let parsed = parse_url_components("http://[::1]:9090/metrics").unwrap();
        assert!(!parsed.tls);
        assert_eq!(parsed.host, "::1");
        assert_eq!(parsed.port, 9090);
        assert_eq!(parsed.authority, "[::1]:9090");
        assert_eq!(parsed.uri.path(), "/metrics");
    }

    #[test]
    fn parse_url_missing_host_returns_error() {
        assert!(parse_url_components("/relative/path").is_err());
    }

    #[test]
    fn parse_url_invalid_returns_error() {
        assert!(parse_url_components("://bad").is_err());
    }

    #[test]
    fn parse_url_root_path() {
        let parsed = parse_url_components("https://127.0.0.1").unwrap();
        assert_eq!(parsed.uri.path(), "/");
    }

    #[test]
    fn parse_url_rejects_unsupported_schemes() {
        for url in ["ftp://127.0.0.1/data.csv", "file:///etc/passwd"] {
            let err = parse_url_components(url).unwrap_err();
            assert!(
                err.to_string().contains("sub-request"),
                "scheme should be rejected: {err}"
            );
        }
    }

    #[tokio::test]
    async fn resolve_addrs_unresolvable_host_returns_connect_error() {
        let result = resolve_addrs("this-host-does-not-exist.invalid", 443).await;
        assert!(
            matches!(result, Err(SubRequestError::Connect(_))),
            "unresolvable host should return Connect error: {result:?}"
        );
    }

    #[tokio::test]
    async fn deadline_bounds_resolution_and_exchange() {
        let result = with_deadline(
            Duration::from_millis(10),
            std::future::pending::<Result<(), SubRequestError>>(),
        )
        .await;
        assert!(matches!(result, Err(SubRequestError::DeadlineExceeded)));
    }

    #[tokio::test]
    async fn execute_falls_back_when_first_address_refuses() {
        let bad_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let bad_addr = bad_listener.local_addr().unwrap();
        drop(bad_listener);

        let good_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let good_addr = good_listener.local_addr().unwrap();
        let captured = capture_raw_request(good_listener);

        let parsed = parse_url_components(&format!("http://example.test:{}/test", good_addr.port())).unwrap();
        let response = execute_resolved_url(
            &test_client(),
            parsed,
            empty_request(),
            &[bad_addr, good_addr],
            1024,
            Duration::from_secs(5),
        )
        .await
        .unwrap();

        assert_eq!(response.status, 200);
        let _request = captured.join().unwrap();
    }

    #[tokio::test]
    async fn execute_sends_original_authority_as_host_header() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = capture_raw_request(listener);
        let authority = format!("my-virtual-host.example.com:{}", addr.port());
        let parsed = parse_url_components(&format!("http://{authority}/test")).unwrap();

        execute_resolved_url(
            &test_client(),
            parsed,
            empty_request(),
            &[addr],
            1024,
            Duration::from_secs(5),
        )
        .await
        .unwrap();

        let wire = captured.join().unwrap().to_lowercase();
        assert!(
            wire.contains(&format!("host: {authority}")),
            "Host header should use original authority, not resolved IP: {wire}"
        );
    }
}
