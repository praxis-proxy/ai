// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! URL resolution and bounded execution for sub-requests.
//!
//! Provides [`execute_url`], which preserves the URL authority for
//! HTTP virtual hosting, resolves every address for fallback, and
//! bounds DNS plus the HTTP exchange with one overall deadline.
//! [`execute_configured_url`] does the same for operator-configured
//! targets, resolving through the proxy's shared DNS cache.
//!
//! Types ([`SubRequestClient`], [`SubRequest`], [`SubResponse`],
//! [`SubRequestError`]) are re-exported from [`praxis_core::subrequest`].

use std::{future::Future, net::SocketAddr, time::Duration};

use pingora_core::upstreams::peer::HttpPeer;
pub use praxis_core::subrequest::{FrameworkHeaders, SubRequest, SubRequestClient, SubRequestError, SubResponse};
use tracing::debug;

use crate::callout_target::{AddressPolicy, validate_http_target, validate_resolved_addrs};

/// Build an isolated client after installing the process-wide crypto provider.
///
/// Constructing a connector creates a rustls client configuration, so provider
/// installation must happen at this boundary rather than relying on a binary
/// entry point having run first.
pub(crate) fn isolated_client(pool_size: usize) -> SubRequestClient {
    praxis_tls::provider::install();
    SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(pool_size, None))
}

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
    /// Path with query values replaced by `[REDACTED]`, safe for logging.
    redacted_uri: String,
}

/// Build a log-safe version of a path-and-query string by replacing each
/// query-parameter value with `[REDACTED]`. Returns the path unchanged
/// when there is no query string.
fn redact_path_query(pq: &str) -> String {
    let Some((path, query)) = pq.split_once('?') else {
        return pq.to_owned();
    };
    let redacted: Vec<_> = query
        .split('&')
        .map(|pair| {
            pair.split_once('=')
                .map_or_else(|| pair.to_owned(), |(key, _)| format!("{key}=[REDACTED]"))
        })
        .collect();
    format!("{path}?{}", redacted.join("&"))
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
    let redacted_uri = redact_path_query(path_and_query);
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
        redacted_uri,
    })
}

/// How a sub-request target's hostname is resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Resolution {
    /// Resolve on every call. Required for URLs a client can choose, so
    /// arbitrary hostnames never fill the shared, bounded cache.
    PerCall,
    /// Resolve through the proxy's upstream DNS cache: single-flight,
    /// answers kept for 60 seconds, and the last good answer served when
    /// resolution fails only for lack of descriptors or memory.
    Shared,
}

/// Resolve every address so callers can fall back across address families.
async fn resolve_addrs(host: &str, port: u16, resolution: Resolution) -> Result<Vec<SocketAddr>, SubRequestError> {
    let addrs = match resolution {
        Resolution::PerCall => tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| SubRequestError::Connect(format!("DNS resolution failed for {host}: {e}")))?
            .collect::<Vec<_>>(),
        Resolution::Shared => {
            let target = if host.contains(':') {
                format!("[{host}]:{port}")
            } else {
                format!("{host}:{port}")
            };
            praxis_core::connectivity::peer::resolve_addresses(&target)
                .await
                .map_err(|e| SubRequestError::Connect(format!("DNS resolution failed for {host}: {e}")))?
                .to_vec()
        },
    };

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
/// while the original URL authority is preserved in `Host`. Admission
/// control and per-peer circuit breaking are inherited from `client`.
///
/// # Errors
///
/// Returns [`SubRequestError`] when the URL cannot be parsed, DNS
/// resolution or connect fails, the deadline is exceeded, admission
/// or circuit breaking rejects the call, the response exceeds
/// `max_response_bytes`, or I/O fails during the exchange.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's transport policy and execution bounds remain explicit"
)]
pub async fn execute_url(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
) -> Result<SubResponse, SubRequestError> {
    execute_url_with_framework(client, url, request, max_response_bytes, timeout, address_policy, None).await
}

/// [`execute_url`] for a target the operator configured, resolved through
/// the proxy's shared DNS cache instead of on every call.
///
/// Callouts made on every request (metering balance checks and usage
/// reports) would otherwise run a blocking lookup, and hold its sockets,
/// per request. Never pass a URL a client can choose: arbitrary hostnames
/// would crowd upstream entries out of the bounded cache. Address-policy
/// validation still runs on every call.
///
/// # Errors
///
/// Same as [`execute_url`].
#[expect(
    clippy::too_many_arguments,
    reason = "the request's transport policy and execution bounds remain explicit"
)]
pub async fn execute_configured_url(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
) -> Result<SubResponse, SubRequestError> {
    with_deadline(
        timeout,
        Box::pin(resolve_and_execute_url(
            client,
            url,
            request,
            max_response_bytes,
            timeout,
            address_policy,
            None,
            Resolution::Shared,
        )),
    )
    .await
}

/// Parse and execute a full-URL sub-request carrying framework headers.
///
/// This is used by the generic callout filter to retain depth propagation
/// while sharing the same DNS pinning and address-policy enforcement as the
/// provider-specific clients.
///
/// # Errors
///
/// Returns [`SubRequestError`] for invalid URLs, resolution or connect
/// failures, policy violations, timeouts, bounded-read failures, or I/O.
#[expect(
    clippy::too_many_arguments,
    reason = "framework metadata is an additional explicit transport input"
)]
pub async fn execute_url_with_framework(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
    framework_headers: Option<&FrameworkHeaders>,
) -> Result<SubResponse, SubRequestError> {
    with_deadline(
        timeout,
        Box::pin(resolve_and_execute_url(
            client,
            url,
            request,
            max_response_bytes,
            timeout,
            address_policy,
            framework_headers,
            Resolution::PerCall,
        )),
    )
    .await
}

/// Resolve DNS and execute the request against the validated addresses.
#[expect(
    clippy::too_many_arguments,
    reason = "the resolution and execution inputs remain explicit"
)]
async fn resolve_and_execute_url(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
    framework_headers: Option<&FrameworkHeaders>,
    resolution: Resolution,
) -> Result<SubResponse, SubRequestError> {
    validate_http_target("sub-request", url).map_err(|error| SubRequestError::InvalidRequest(error.to_string()))?;
    let parsed = parse_url_components(url)?;
    let addrs = resolve_addrs(&parsed.host, parsed.port, resolution).await?;
    execute_with_addresses(
        client,
        parsed,
        request,
        max_response_bytes,
        timeout,
        address_policy,
        framework_headers,
        addrs,
    )
    .await
}

/// Validate addresses and try each one until the request succeeds.
///
/// Keeping this boundary separate gives tests a controlled DNS result set
/// without changing production resolution behavior.
#[expect(
    clippy::too_many_arguments,
    reason = "the subrequest transport inputs remain explicit"
)]
async fn execute_with_addresses(
    client: &SubRequestClient,
    parsed: ParsedUrl,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
    framework_headers: Option<&FrameworkHeaders>,
    addrs: Vec<SocketAddr>,
) -> Result<SubResponse, SubRequestError> {
    let addrs = validate_resolved_addrs("sub-request", &addrs, address_policy)
        .map_err(|error| SubRequestError::Connect(error.to_string()))?;
    let mut request = request;
    debug!(
        host = %parsed.host,
        uri = %parsed.redacted_uri,
        method = %request.method,
        "sub-request: dispatching"
    );
    request.uri = parsed.uri;
    request.headers.insert(http::header::HOST, parsed.authority);

    let mut last_connect_error = None;
    for addr in &addrs {
        let peer = HttpPeer::new(*addr, parsed.tls, parsed.sni.clone());
        debug!(host = %parsed.host, %addr, "sub-request: trying resolved address");

        match Box::pin(client.execute(&peer, &request, max_response_bytes, timeout, framework_headers)).await {
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

/// Test-only entry point that substitutes controlled DNS results.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "the test seam mirrors the production subrequest transport inputs"
)]
async fn execute_url_with_test_addresses(
    client: &SubRequestClient,
    url: &str,
    request: SubRequest,
    max_response_bytes: usize,
    timeout: Duration,
    address_policy: AddressPolicy,
    framework_headers: Option<&FrameworkHeaders>,
    addrs: Vec<SocketAddr>,
) -> Result<SubResponse, SubRequestError> {
    validate_http_target("sub-request", url).map_err(|error| SubRequestError::InvalidRequest(error.to_string()))?;
    let parsed = parse_url_components(url)?;
    execute_with_addresses(
        client,
        parsed,
        request,
        max_response_bytes,
        timeout,
        address_policy,
        framework_headers,
        addrs,
    )
    .await
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

    use super::*;

    fn test_client() -> SubRequestClient {
        isolated_client(4)
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
        assert!(!parsed.tls, "HTTP URL should not enable TLS");
        assert_eq!(parsed.host, "::1");
        assert_eq!(parsed.port, 9090);
        assert_eq!(parsed.authority, "[::1]:9090");
        assert_eq!(parsed.uri.path(), "/metrics");
    }

    #[test]
    fn parse_url_missing_host_returns_error() {
        assert!(
            parse_url_components("/relative/path").is_err(),
            "relative URL should be rejected"
        );
    }

    #[test]
    fn parse_url_invalid_returns_error() {
        assert!(
            parse_url_components("://bad").is_err(),
            "malformed URL should be rejected"
        );
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
        let result = resolve_addrs("this-host-does-not-exist.invalid", 443, Resolution::PerCall).await;
        assert!(
            matches!(result, Err(SubRequestError::Connect(_))),
            "unresolvable host should return Connect error: {result:?}"
        );
    }

    #[tokio::test]
    async fn shared_resolution_reuses_a_cached_failure_and_per_call_does_not() {
        let host = "praxis-ai-subrequest-shared-resolution.invalid";

        let first = resolve_addrs(host, 443, Resolution::Shared).await;
        assert!(
            matches!(&first, Err(SubRequestError::Connect(_))),
            "unresolvable host should return Connect error: {first:?}"
        );
        let second = resolve_addrs(host, 443, Resolution::Shared).await;
        assert!(
            matches!(&second, Err(SubRequestError::Connect(detail)) if detail.contains("recently failed")),
            "a repeat lookup must come from the shared cache: {second:?}"
        );
        let per_call = resolve_addrs(host, 443, Resolution::PerCall).await;
        assert!(
            matches!(&per_call, Err(SubRequestError::Connect(detail)) if !detail.contains("recently failed")),
            "per-call resolution must never consult the shared cache: {per_call:?}"
        );
    }

    #[tokio::test]
    async fn shared_resolution_resolves_hostnames() {
        let addrs = resolve_addrs("localhost", 8080, Resolution::Shared).await.unwrap();
        assert!(
            addrs.iter().all(|addr| addr.ip().is_loopback() && addr.port() == 8080),
            "localhost should resolve to loopback on the requested port: {addrs:?}"
        );
    }

    #[tokio::test]
    async fn shared_resolution_accepts_ip_literals() {
        let v4 = resolve_addrs("127.0.0.1", 8080, Resolution::Shared).await.unwrap();
        assert_eq!(v4, vec!["127.0.0.1:8080".parse::<SocketAddr>().unwrap()]);

        let v6 = resolve_addrs("::1", 8080, Resolution::Shared).await.unwrap();
        assert_eq!(
            v6,
            vec!["[::1]:8080".parse::<SocketAddr>().unwrap()],
            "a bracket-stripped IPv6 host must be rebracketed for the shared resolver"
        );
    }

    #[tokio::test]
    async fn configured_url_resolves_and_keeps_the_authority() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = capture_raw_request(listener);

        let response = execute_configured_url(
            &test_client(),
            &format!("http://localhost:{port}/test"),
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
        )
        .await
        .unwrap();

        assert_eq!(response.status, 200);
        let wire = captured.join().unwrap().to_lowercase();
        assert!(
            wire.contains(&format!("host: localhost:{port}")),
            "Host header should use the configured authority: {wire}"
        );
    }

    #[tokio::test]
    async fn configured_url_enforces_address_policy_after_cached_resolution() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!("http://localhost:{port}/test");

        let _warm = resolve_addrs("localhost", port, Resolution::Shared).await.unwrap();
        let result = execute_configured_url(
            &test_client(),
            &url,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::PublicOnly,
        )
        .await;

        assert!(
            matches!(&result, Err(SubRequestError::Connect(detail)) if detail.contains("blocked non-public address")),
            "a cached answer must still pass address policy: {result:?}"
        );
        assert!(
            matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "a rejected address set must not be dialed"
        );
    }

    #[tokio::test]
    async fn deadline_bounds_resolution_and_exchange() {
        let result = with_deadline(
            Duration::from_millis(10),
            std::future::pending::<Result<(), SubRequestError>>(),
        )
        .await;
        assert!(
            matches!(result, Err(SubRequestError::DeadlineExceeded)),
            "deadline should bound the pending operation"
        );
    }

    #[tokio::test]
    async fn execute_falls_back_when_first_address_refuses() {
        let bad_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let bad_addr = bad_listener.local_addr().unwrap();
        drop(bad_listener);

        let good_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let good_addr = good_listener.local_addr().unwrap();
        let captured = capture_raw_request(good_listener);

        let url = format!("http://example.test:{}/test", good_addr.port());
        let response = Box::pin(execute_url_with_test_addresses(
            &test_client(),
            &url,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
            None,
            vec![bad_addr, good_addr],
        ))
        .await
        .unwrap();

        assert_eq!(response.status, 200);
        let _request = captured.join().unwrap();
    }

    #[tokio::test]
    async fn execute_rejects_mixed_resolved_addresses_before_dialing() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let private_addr = listener.local_addr().unwrap();
        let public_addr = "8.8.8.8:443".parse().unwrap();
        let url = format!("http://example.test:{}/test", private_addr.port());

        let result = Box::pin(execute_url_with_test_addresses(
            &test_client(),
            &url,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::PublicOnly,
            None,
            vec![public_addr, private_addr],
        ))
        .await;

        assert!(
            matches!(&result, Err(SubRequestError::Connect(detail)) if detail.contains("blocked non-public address")),
            "mixed public/private answers must be rejected before transport: {result:?}"
        );
        assert!(
            matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "a rejected address set must not be dialed"
        );
    }

    #[tokio::test]
    async fn execute_sends_original_authority_as_host_header() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = capture_raw_request(listener);
        let authority = format!("my-virtual-host.example.com:{}", addr.port());
        let parsed = parse_url_components(&format!("http://{authority}/test")).unwrap();

        Box::pin(execute_with_addresses(
            &test_client(),
            parsed,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
            None,
            vec![addr],
        ))
        .await
        .unwrap();

        let wire = captured.join().unwrap().to_lowercase();
        assert!(
            wire.contains(&format!("host: {authority}")),
            "Host header should use original authority, not resolved IP: {wire}"
        );
    }

    #[tokio::test]
    #[expect(clippy::too_many_lines, reason = "sequential setup, execution, and wire assertions")]
    async fn execute_overwrites_conflicting_host_header() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = capture_raw_request(listener);
        let authority = format!("expected.example.com:{}", addr.port());
        let parsed = parse_url_components(&format!("http://{authority}/test")).unwrap();
        let mut request = empty_request();
        request.headers.insert(
            http::header::HOST,
            http::HeaderValue::from_static("attacker.example.com"),
        );

        Box::pin(execute_with_addresses(
            &test_client(),
            parsed,
            request,
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
            None,
            vec![addr],
        ))
        .await
        .unwrap();

        let wire = captured.join().unwrap().to_lowercase();
        let host_headers = wire
            .lines()
            .filter(|line| line.starts_with("host:"))
            .collect::<Vec<_>>();
        assert_eq!(
            host_headers,
            [format!("host: {authority}").as_str()],
            "only the URL authority should be sent as Host: {wire}"
        );
    }

    #[test]
    fn redact_path_query_preserves_path_without_query() {
        assert_eq!(redact_path_query("/v1/files"), "/v1/files");
    }

    #[test]
    fn redact_path_query_replaces_values() {
        assert_eq!(
            redact_path_query("/blob?sig=SECRET&se=2026-01-01"),
            "/blob?sig=[REDACTED]&se=[REDACTED]"
        );
    }

    #[test]
    fn redact_path_query_preserves_valueless_keys() {
        assert_eq!(redact_path_query("/path?flag"), "/path?flag");
    }

    #[tokio::test(flavor = "current_thread")]
    #[expect(clippy::too_many_lines, reason = "inline tracing capture layer and assertions")]
    async fn dispatch_log_event_redacts_query_values() {
        use std::sync::{Arc, Mutex};

        use tracing_subscriber::layer::SubscriberExt as _;

        #[derive(Clone)]
        struct EventCapture(Arc<Mutex<Vec<String>>>);

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for EventCapture {
            fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
                struct FieldCollector {
                    message: Option<String>,
                    uri: Option<String>,
                }
                impl tracing::field::Visit for FieldCollector {
                    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                        match field.name() {
                            "message" => self.message = Some(format!("{value:?}")),
                            "uri" => self.uri = Some(format!("{value:?}")),
                            _ => {},
                        }
                    }

                    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                        match field.name() {
                            "message" => self.message = Some(value.to_owned()),
                            "uri" => self.uri = Some(value.to_owned()),
                            _ => {},
                        }
                    }
                }
                let mut collector = FieldCollector {
                    message: None,
                    uri: None,
                };
                event.record(&mut collector);
                if collector.message.as_deref() == Some("sub-request: dispatching")
                    && let Some(uri) = collector.uri
                {
                    self.0.lock().unwrap().push(uri);
                }
            }
        }

        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = EventCapture(Arc::clone(&events));
        let subscriber = tracing_subscriber::registry().with(capture);

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let _captured = capture_raw_request(listener);

        let client = test_client();
        let url = format!("http://example.test:{}/blob?sig=SECRET&token=s3cret", addr.port());

        // Callsite interest is cached process-wide, and while this is the only
        // scoped dispatcher, a callsite first reached from another test's
        // thread caches that thread's "nobody listening" answer. Register the
        // callsite before capturing (creating the dispatcher raises the max
        // level so it can register), then recompute interest with the capture
        // installed.
        let dispatch = tracing::Dispatch::new(subscriber);
        let refused = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let _warm_up = Box::pin(execute_url_with_test_addresses(
            &client,
            &url,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
            None,
            vec![refused],
        ))
        .await;
        let _guard = tracing::dispatcher::set_default(&dispatch);
        tracing::callsite::rebuild_interest_cache();
        let _result = Box::pin(execute_url_with_test_addresses(
            &client,
            &url,
            empty_request(),
            1024,
            Duration::from_secs(5),
            AddressPolicy::AllowPrivate,
            None,
            vec![addr],
        ))
        .await;

        let logged_uris = events.lock().unwrap();
        assert!(
            !logged_uris.is_empty(),
            "dispatch event with uri field should be emitted"
        );
        for uri in logged_uris.iter() {
            assert!(
                !uri.contains("SECRET") && !uri.contains("s3cret"),
                "signed URL secrets must not appear in logged URI: {uri}"
            );
            assert!(
                uri.contains("[REDACTED]"),
                "query values should be replaced with [REDACTED]: {uri}"
            );
        }
    }
}
