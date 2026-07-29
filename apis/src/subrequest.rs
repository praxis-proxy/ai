// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! **Temporary** sub-request execution — to be replaced by
//! [praxis-core#826](https://github.com/praxis-proxy/praxis/issues/826).
//!
//! Provides URL-to-peer parsing and an HTTP execution function
//! that replaces `praxis_core::callout::CalloutClient` with
//! Pingora-native transport — getting connection pooling and TLS
//! without a separate HTTP stack.
//!
//! Once praxis-core exposes the executor and typed errors through
//! `SubRequestConnector`, this module should shrink to URL
//! validation, `TargetPeer` construction, and domain error
//! mapping.
//!
//! Types ([`SubRequestConnector`], [`SubRequest`], [`SubResponse`])
//! are re-exported from [`praxis_core::subrequest`].
//!
//! [`Connector`]: pingora_core::connectors::http::Connector

use std::time::Duration;

use bytes::Bytes;
use http::HeaderMap;
use pingora_core::{protocols::http::client::HttpSession, upstreams::peer::HttpPeer};
pub(crate) use praxis_core::subrequest::{SubRequest, SubRequestConnector, SubResponse};
use thiserror::Error;
use tracing::{debug, warn};

// -----------------------------------------------------------------------------
// Error
// -----------------------------------------------------------------------------

/// Errors from sub-request execution.
#[derive(Debug, Error)]
pub(crate) enum SubRequestError {
    /// Failed to connect to the upstream.
    #[error("sub-request connect error: {0}")]
    Connect(String),

    /// Failed to write the request or read the response.
    #[error("sub-request I/O error: {0}")]
    Io(String),

    /// Response body exceeded the size limit.
    #[error(
        "sub-request response body exceeded limit \
         ({actual} > {limit} bytes, status {status})"
    )]
    ResponseTooLarge {
        /// Actual bytes received before truncation.
        actual: usize,
        /// Configured limit.
        limit: usize,
        /// HTTP status code at the time of truncation.
        status: u16,
    },

    /// The overall deadline expired.
    #[error("sub-request deadline exceeded")]
    DeadlineExceeded,

    /// The URL could not be parsed into a valid peer address.
    #[error("invalid sub-request URL: {0}")]
    InvalidUrl(String),
}

// -----------------------------------------------------------------------------
// TargetPeer
// -----------------------------------------------------------------------------

/// Parsed target for sub-request execution.
///
/// Holds URL components needed to create an [`HttpPeer`] and set
/// the Host header. DNS resolution is deferred to execution time
/// so it is both async and fallible.
#[derive(Debug, Clone)]
pub(crate) struct TargetPeer {
    /// Hostname or IP to connect to.
    pub(crate) host: String,
    /// TCP port.
    pub(crate) port: u16,
    /// Whether TLS is enabled.
    pub(crate) tls: bool,
    /// TLS SNI hostname.
    sni: String,
    /// Original URL authority for the HTTP Host header.
    authority: String,
}

// -----------------------------------------------------------------------------
// URL → (TargetPeer, Uri)
// -----------------------------------------------------------------------------

/// Parse a full URL into a [`TargetPeer`] and a path-only
/// [`http::Uri`].
///
/// Extracts scheme (→ TLS), host, port (defaults 443/80), and
/// SNI from the URL. The returned URI contains only the path and
/// query components. Only `http` and `https` schemes are accepted.
///
/// DNS resolution is NOT performed here — it happens inside
/// [`execute`] so it can be async and fallible.
#[expect(clippy::too_many_lines, reason = "sequential URL component extraction")]
pub(crate) fn parse_url(url: &str) -> Result<(TargetPeer, http::Uri), SubRequestError> {
    let parsed: http::Uri = url
        .parse()
        .map_err(|e| SubRequestError::InvalidUrl(format!("{e}: {url}")))?;

    let scheme = parsed.scheme_str().unwrap_or("http");

    let tls = match scheme {
        "https" => true,
        "http" => false,
        other => {
            return Err(SubRequestError::InvalidUrl(format!(
                "unsupported scheme '{other}': {url}"
            )));
        },
    };

    let authority = parsed
        .authority()
        .ok_or_else(|| SubRequestError::InvalidUrl(format!("missing host: {url}")))?;

    let host = authority.host().trim_start_matches('[').trim_end_matches(']');
    let default_port = if tls { 443 } else { 80 };
    let port = authority.port_u16().unwrap_or(default_port);

    let sni = if tls { host.to_owned() } else { String::new() };

    let target = TargetPeer {
        host: host.to_owned(),
        port,
        tls,
        sni,
        authority: authority.to_string(),
    };

    let path_and_query = parsed.path_and_query().map_or("/", |pq| pq.as_str());
    let uri: http::Uri = path_and_query
        .parse()
        .map_err(|e| SubRequestError::InvalidUrl(format!("bad path: {e}")))?;

    Ok((target, uri))
}

// -----------------------------------------------------------------------------
// Execute
// -----------------------------------------------------------------------------

/// Execute a sub-request using Pingora's [`Connector`].
///
/// Resolves DNS asynchronously, connects to the target, sends
/// `request`, reads the full response (bounded by
/// `max_response_bytes`), and returns a [`SubResponse`].  The
/// overall operation — including DNS — is bounded by `timeout`.
///
/// [`Connector`]: pingora_core::connectors::http::Connector
pub(crate) async fn execute(
    connector: &SubRequestConnector,
    target: &TargetPeer,
    request: &SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
) -> Result<SubResponse, SubRequestError> {
    tokio::time::timeout(
        timeout,
        Box::pin(execute_inner(connector, target, request, max_response_bytes, timeout)),
    )
    .await
    .map_err(|_elapsed| SubRequestError::DeadlineExceeded)?
}

/// Resolve DNS and perform the HTTP exchange under the deadline
/// enforced by [`execute`].
#[expect(clippy::large_stack_frames, reason = "Pingora session types are large")]
#[expect(clippy::too_many_lines, reason = "sequential HTTP exchange steps")]
#[expect(clippy::cognitive_complexity, reason = "mirrors praxis-filter execute_inner")]
async fn execute_inner(
    connector: &SubRequestConnector,
    target: &TargetPeer,
    request: &SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
) -> Result<SubResponse, SubRequestError> {
    let _permit = connector.acquire_permit().await;
    let addrs = resolve_addrs(target).await?;
    let (mut session, reused, peer) = connect_to_first_reachable(connector, target, &addrs, timeout).await?;

    debug!(
        authority = target.authority,
        reused,
        method = %request.method,
        uri = %request.uri,
        "sub-request: connected"
    );

    session.set_read_timeout(Some(min_timeout(peer.options.read_timeout, timeout)));
    session.set_write_timeout(Some(min_timeout(peer.options.write_timeout, timeout)));

    let path = request
        .uri
        .path_and_query()
        .map_or(b"/".as_slice(), |pq| pq.as_str().as_bytes());
    let mut req_header = pingora_http::RequestHeader::build(request.method.clone(), path, None)
        .map_err(|e| SubRequestError::Io(e.to_string()))?;

    for (name, value) in &request.headers {
        let _append = req_header.append_header(name.clone(), value.clone());
    }

    ensure_host_header(&mut req_header, &target.authority)?;

    if !request.body.is_empty() || empty_body_needs_framing(&request.method) {
        let _cl = req_header.insert_header("Content-Length", request.body.len().to_string());
    }

    session
        .write_request_header(Box::new(req_header))
        .await
        .map_err(|e| SubRequestError::Io(e.to_string()))?;

    if !request.body.is_empty() {
        session
            .write_request_body(request.body.clone(), true)
            .await
            .map_err(|e| SubRequestError::Io(e.to_string()))?;
    }

    session
        .finish_request_body()
        .await
        .map_err(|e| SubRequestError::Io(e.to_string()))?;

    session
        .read_response_header()
        .await
        .map_err(|e| SubRequestError::Io(e.to_string()))?;

    let resp_header = session
        .response_header()
        .ok_or_else(|| SubRequestError::Io("no response header received".to_owned()))?;

    let status = resp_header.status.as_u16();
    if !(100..=599).contains(&status) {
        session.shutdown().await;
        return Err(SubRequestError::Io(format!(
            "upstream returned unsupported HTTP status {status}"
        )));
    }
    let mut resp_headers = HeaderMap::new();
    for (name, value) in &resp_header.headers {
        if let Ok(v) = http::header::HeaderValue::from_bytes(value.as_bytes()) {
            resp_headers.append(name.clone(), v);
        }
    }

    let mut body_buf = Vec::new();
    while !session.response_done() {
        match session.read_response_body().await {
            Ok(Some(chunk)) => {
                if body_buf.len() + chunk.len() > max_response_bytes {
                    warn!(
                        current = body_buf.len(),
                        chunk = chunk.len(),
                        limit = max_response_bytes,
                        status,
                        "sub-request response body exceeded limit"
                    );
                    session.shutdown().await;
                    return Err(SubRequestError::ResponseTooLarge {
                        actual: body_buf.len() + chunk.len(),
                        limit: max_response_bytes,
                        status,
                    });
                }
                body_buf.extend_from_slice(&chunk);
            },
            Ok(None) => break,
            Err(e) => {
                session.shutdown().await;
                return Err(SubRequestError::Io(e.to_string()));
            },
        }
    }

    debug!(status, body_bytes = body_buf.len(), "sub-request: response received");

    connector.connector().release_http_session(session, &peer, None).await;

    Ok(SubResponse {
        status,
        headers: resp_headers,
        body: Bytes::from(body_buf),
    })
}

/// Methods whose empty payload is commonly rejected without
/// explicit framing.
fn empty_body_needs_framing(method: &http::Method) -> bool {
    matches!(*method, http::Method::POST | http::Method::PUT | http::Method::PATCH)
}

/// Resolve all addresses for a [`TargetPeer`].
///
/// Uses `tokio::net::lookup_host` for async, fallible resolution.
/// Returns every address so callers can fall back on dual-stack
/// hosts where the first address family is unreachable.
async fn resolve_addrs(target: &TargetPeer) -> Result<Vec<std::net::SocketAddr>, SubRequestError> {
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((target.host.as_str(), target.port))
        .await
        .map_err(|e| SubRequestError::Connect(format!("DNS resolution failed for {}: {e}", target.host)))?
        .collect();

    if addrs.is_empty() {
        return Err(SubRequestError::Connect(format!(
            "no addresses resolved for {}",
            target.host
        )));
    }

    Ok(addrs)
}

/// Try each resolved address in order and return the first
/// successful connection.
///
/// On dual-stack hosts this allows falling back from an
/// unreachable IPv6 address to a working IPv4 address — a
/// property the previous `reqwest`-based client provided via
/// Happy Eyeballs.
#[expect(clippy::large_stack_frames, reason = "Pingora session types are large")]
async fn connect_to_first_reachable(
    connector: &SubRequestConnector,
    target: &TargetPeer,
    addrs: &[std::net::SocketAddr],
    timeout: Duration,
) -> Result<(HttpSession, bool, HttpPeer), SubRequestError> {
    let mut last_error = None;

    for addr in addrs {
        let mut peer = HttpPeer::new(addr.to_string(), target.tls, target.sni.clone());
        clamp_peer_timeouts(&mut peer, timeout);

        match Box::pin(connector.connector().get_http_session(&peer)).await {
            Ok((session, reused)) => return Ok((session, reused, peer)),
            Err(e) => {
                debug!(
                    address = %addr,
                    error = %e,
                    "sub-request: connect failed, trying next address"
                );
                last_error = Some(e);
            },
        }
    }

    Err(SubRequestError::Connect(last_error.map_or_else(
        || format!("no addresses resolved for {}", target.host),
        |e| format!("all resolved addresses for {} failed: {e}", target.host),
    )))
}

/// Ensure HTTP/1.1 virtual hosting and HTTP/2 `:authority` are
/// valid — using the original URL authority, not the resolved IP.
fn ensure_host_header(request: &mut pingora_http::RequestHeader, authority: &str) -> Result<(), SubRequestError> {
    if !request.headers.contains_key(http::header::HOST) {
        request
            .insert_header(http::header::HOST, authority)
            .map_err(|error| SubRequestError::Io(error.to_string()))?;
    }
    Ok(())
}

/// Clamp connect timeouts to the remaining overall deadline.
fn clamp_peer_timeouts(peer: &mut HttpPeer, deadline: Duration) {
    peer.options.connection_timeout = Some(min_timeout(peer.options.connection_timeout, deadline));
    peer.options.total_connection_timeout = Some(min_timeout(peer.options.total_connection_timeout, deadline));
}

/// Keep an operator-configured timeout when it is stricter than
/// the deadline.
fn min_timeout(configured: Option<Duration>, deadline: Duration) -> Duration {
    configured.map_or(deadline, |configured| configured.min(deadline))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

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

    use super::*;

    #[test]
    fn parse_url_https() {
        let (target, uri) = parse_url("https://127.0.0.1:8443/v1/search?q=test").unwrap();
        assert!(target.tls, "HTTPS should enable TLS");
        assert_eq!(uri.path(), "/v1/search");
        assert_eq!(uri.query(), Some("q=test"));
    }

    #[test]
    fn parse_url_http_with_port() {
        let (target, uri) = parse_url("http://127.0.0.1:8080/health").unwrap();
        assert!(!target.tls, "HTTP should disable TLS");
        assert_eq!(uri.path(), "/health");
    }

    #[test]
    fn parse_url_default_ports() {
        let (https_target, _) = parse_url("https://127.0.0.1/path").unwrap();
        assert_eq!(https_target.host, "127.0.0.1");
        assert_eq!(https_target.port, 443);

        let (http_target, _) = parse_url("http://127.0.0.1/path").unwrap();
        assert_eq!(http_target.host, "127.0.0.1");
        assert_eq!(http_target.port, 80);
    }

    #[test]
    fn parse_url_preserves_hostname_authority() {
        let (target, _) = parse_url("https://api.example.com/v1/search").unwrap();
        assert_eq!(target.authority, "api.example.com");
        assert_eq!(target.host, "api.example.com");
        assert_eq!(target.port, 443);
    }

    #[test]
    fn parse_url_preserves_hostname_authority_with_port() {
        let (target, _) = parse_url("https://api.example.com:8443/path").unwrap();
        assert_eq!(target.authority, "api.example.com:8443");
        assert_eq!(target.host, "api.example.com");
        assert_eq!(target.port, 8443);
    }

    #[test]
    fn parse_url_ipv6_loopback() {
        let (target, uri) = parse_url("http://[::1]:9090/metrics").unwrap();
        assert_eq!(target.host, "::1");
        assert_eq!(target.port, 9090);
        assert_eq!(target.authority, "[::1]:9090");
        assert_eq!(uri.path(), "/metrics");
    }

    #[test]
    fn parse_url_ipv6_default_port() {
        let (target, _) = parse_url("https://[::1]/path").unwrap();
        assert_eq!(target.host, "::1");
        assert_eq!(target.port, 443);
        assert!(target.tls);
    }

    #[tokio::test]
    async fn resolve_addrs_handles_ipv6() {
        let target = TargetPeer {
            host: "::1".to_owned(),
            port: 8080,
            tls: false,
            sni: String::new(),
            authority: "[::1]:8080".to_owned(),
        };
        let addrs = resolve_addrs(&target).await.unwrap();
        assert!(!addrs.is_empty(), "should resolve at least one address");
        assert!(
            addrs.iter().any(|a| a.to_string().contains("::1")),
            "resolved addresses should contain IPv6 loopback: {addrs:?}"
        );
    }

    #[test]
    fn parse_url_missing_host_returns_error() {
        assert!(parse_url("/relative/path").is_err());
    }

    #[test]
    fn parse_url_invalid_returns_error() {
        assert!(parse_url("://bad").is_err());
    }

    #[test]
    fn parse_url_root_path() {
        let (_, uri) = parse_url("https://127.0.0.1").unwrap();
        assert_eq!(uri.path(), "/");
    }

    #[test]
    fn parse_url_rejects_ftp_scheme() {
        let err = parse_url("ftp://127.0.0.1/data.csv").unwrap_err();
        assert!(
            err.to_string().contains("unsupported scheme"),
            "ftp should be rejected: {err}"
        );
    }

    #[test]
    fn parse_url_rejects_file_scheme() {
        let err = parse_url("file:///etc/passwd").unwrap_err();
        assert!(
            err.to_string().contains("invalid sub-request URL"),
            "file:// should be rejected: {err}"
        );
    }

    #[test]
    fn connector_clone_shares_pool() {
        let a = SubRequestConnector::new(16, None);
        let b = a.clone();
        assert!(
            std::ptr::eq(a.connector(), b.connector()),
            "cloned connectors should share the same pool"
        );
    }

    #[test]
    fn connector_debug_does_not_panic() {
        let connector = SubRequestConnector::new(8, None);
        let debug = format!("{connector:?}");
        assert!(
            debug.contains("SubRequestConnector"),
            "debug output should contain type name"
        );
    }

    #[test]
    fn empty_body_framing_for_entity_methods() {
        assert!(empty_body_needs_framing(&http::Method::POST));
        assert!(empty_body_needs_framing(&http::Method::PUT));
        assert!(empty_body_needs_framing(&http::Method::PATCH));
        assert!(!empty_body_needs_framing(&http::Method::GET));
        assert!(!empty_body_needs_framing(&http::Method::HEAD));
    }

    #[test]
    fn min_timeout_preserves_stricter_limit() {
        assert_eq!(
            min_timeout(Some(Duration::from_secs(1)), Duration::from_secs(10)),
            Duration::from_secs(1)
        );
        assert_eq!(
            min_timeout(Some(Duration::from_secs(20)), Duration::from_secs(10)),
            Duration::from_secs(10)
        );
        assert_eq!(min_timeout(None, Duration::from_secs(10)), Duration::from_secs(10));
    }

    #[tokio::test]
    async fn deadline_bounds_the_complete_exchange() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        let connector = SubRequestConnector::new(1, None);
        let (target, uri) = parse_url(&format!("http://{address}/")).unwrap();
        let request = SubRequest {
            method: http::Method::GET,
            uri,
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let started = std::time::Instant::now();
        let result = Box::pin(execute(&connector, &target, &request, 1024, Duration::from_millis(10))).await;
        let elapsed = started.elapsed();
        backend.abort();

        assert!(result.is_err(), "a backend that never responds must time out");
        assert!(
            elapsed < Duration::from_millis(500),
            "exchange exceeded its deadline: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn unresolvable_host_returns_connect_error() {
        let connector = SubRequestConnector::new(1, None);
        let (target, uri) = parse_url("https://this-host-does-not-exist.invalid/path").unwrap();
        let request = SubRequest {
            method: http::Method::GET,
            uri,
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let result = Box::pin(execute(&connector, &target, &request, 1024, Duration::from_secs(5))).await;

        assert!(
            matches!(result, Err(SubRequestError::Connect(_))),
            "unresolvable host should return Connect error, not panic: {result:?}"
        );
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

    #[tokio::test]
    async fn execute_sends_authority_as_host_header() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = capture_raw_request(listener);

        let connector = SubRequestConnector::new(1, None);
        let target = TargetPeer {
            host: "127.0.0.1".to_owned(),
            port: addr.port(),
            tls: false,
            sni: String::new(),
            authority: format!("my-virtual-host.example.com:{}", addr.port()),
        };
        let request = SubRequest {
            method: http::Method::GET,
            uri: "/test".parse().unwrap(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        };

        let _response = Box::pin(execute(&connector, &target, &request, 1024, Duration::from_secs(5)))
            .await
            .unwrap();

        let wire = captured.join().unwrap().to_lowercase();
        let expected = format!("host: my-virtual-host.example.com:{}", addr.port());
        assert!(
            wire.contains(&expected),
            "Host header should use original authority, not resolved IP: {wire}"
        );
    }

    #[tokio::test]
    async fn connect_falls_back_when_first_address_refuses() {
        let bad_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let bad_addr = bad_listener.local_addr().unwrap();
        drop(bad_listener);

        let good_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let good_addr = good_listener.local_addr().unwrap();

        let connector = SubRequestConnector::new(1, None);
        let target = TargetPeer {
            host: "127.0.0.1".to_owned(),
            port: good_addr.port(),
            tls: false,
            sni: String::new(),
            authority: format!("127.0.0.1:{}", good_addr.port()),
        };

        let addrs = vec![bad_addr, good_addr];
        let result = connect_to_first_reachable(&connector, &target, &addrs, Duration::from_secs(5)).await;

        drop(good_listener);
        assert!(result.is_ok(), "should connect via fallback to second address");
    }

    #[tokio::test]
    async fn connect_fails_when_all_addresses_refuse() {
        let l1 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let a1 = l1.local_addr().unwrap();
        drop(l1);
        let l2 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let a2 = l2.local_addr().unwrap();
        drop(l2);

        let connector = SubRequestConnector::new(1, None);
        let target = TargetPeer {
            host: "127.0.0.1".to_owned(),
            port: a1.port(),
            tls: false,
            sni: String::new(),
            authority: format!("127.0.0.1:{}", a1.port()),
        };

        let result = connect_to_first_reachable(&connector, &target, &[a1, a2], Duration::from_secs(2)).await;

        let Err(err) = result else {
            panic!("should fail when all addresses refuse connections");
        };
        let msg = err.to_string();
        assert!(
            msg.contains("all resolved"),
            "should report all addresses failed: {msg}"
        );
    }
}
