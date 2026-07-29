// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! URL-to-peer parsing and DNS resolution for sub-request execution.
//!
//! Provides [`resolve_url`] which converts a full URL into an
//! [`HttpPeer`] (with async DNS resolution) and a path-only [`Uri`],
//! suitable for passing to [`SubRequestClient::execute`].
//!
//! Types ([`SubRequestClient`], [`SubRequest`], [`SubResponse`],
//! [`SubRequestError`]) are re-exported from [`praxis_core::subrequest`].
//!
//! [`HttpPeer`]: pingora_core::upstreams::peer::HttpPeer
//! [`Uri`]: http::Uri

use pingora_core::upstreams::peer::HttpPeer;
pub(crate) use praxis_core::subrequest::{SubRequest, SubRequestClient, SubRequestError, SubResponse};
use tracing::debug;

// -----------------------------------------------------------------------------
// URL → (HttpPeer, Uri)
// -----------------------------------------------------------------------------

/// Extract scheme, TLS flag, host, port, SNI, and path from a URL.
fn parse_url_components(url: &str) -> Result<(bool, String, u16, String, http::Uri), SubRequestError> {
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

    let path_and_query = parsed.path_and_query().map_or("/", |pq| pq.as_str());
    let uri: http::Uri = path_and_query
        .parse()
        .map_err(|e| SubRequestError::InvalidRequest(format!("bad path: {e}")))?;

    Ok((tls, host.to_owned(), port, sni, uri))
}

/// Parse a full URL and resolve DNS to produce an [`HttpPeer`]
/// and a path-only [`http::Uri`].
///
/// Extracts scheme (→ TLS), host, port (defaults 443/80), and
/// SNI from the URL. The returned URI contains only the path and
/// query components. Only `http` and `https` schemes are accepted.
///
/// DNS resolution is performed asynchronously via
/// [`tokio::net::lookup_host`]. When multiple addresses resolve
/// (e.g. dual-stack hosts), the first is used — Pingora's
/// [`SubRequestClient`] handles retries at the transport layer.
pub(crate) async fn resolve_url(url: &str) -> Result<(HttpPeer, http::Uri), SubRequestError> {
    let (tls, host, port, sni, uri) = parse_url_components(url)?;

    let addr = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|e| SubRequestError::Connect(format!("DNS resolution failed for {host}: {e}")))?
        .next()
        .ok_or_else(|| SubRequestError::Connect(format!("no addresses resolved for {host}")))?;

    debug!(%host, %addr, "sub-request: resolved");

    Ok((HttpPeer::new(addr, tls, sni), uri))
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
    use pingora_core::upstreams::peer::Peer as _;

    use super::*;

    #[tokio::test]
    async fn resolve_url_https() {
        let (peer, uri) = resolve_url("https://127.0.0.1:8443/v1/search?q=test").await.unwrap();
        assert!(peer.is_tls(), "HTTPS should enable TLS");
        assert_eq!(uri.path(), "/v1/search");
        assert_eq!(uri.query(), Some("q=test"));
    }

    #[tokio::test]
    async fn resolve_url_http_with_port() {
        let (peer, uri) = resolve_url("http://127.0.0.1:8080/health").await.unwrap();
        assert!(!peer.is_tls(), "HTTP should disable TLS");
        assert_eq!(uri.path(), "/health");
    }

    #[tokio::test]
    async fn resolve_url_ipv6_loopback() {
        let (peer, uri) = resolve_url("http://[::1]:9090/metrics").await.unwrap();
        assert!(!peer.is_tls());
        assert_eq!(uri.path(), "/metrics");
        let addr = peer.address().to_string();
        assert!(addr.contains("::1"), "should contain IPv6 loopback: {addr}");
    }

    #[tokio::test]
    async fn resolve_url_ipv6_default_port() {
        let (peer, _) = resolve_url("https://[::1]/path").await.unwrap();
        assert!(peer.is_tls());
    }

    #[test]
    fn resolve_url_missing_host_returns_error() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(resolve_url("/relative/path"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_url_invalid_returns_error() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(resolve_url("://bad"));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resolve_url_root_path() {
        let (_, uri) = resolve_url("https://127.0.0.1").await.unwrap();
        assert_eq!(uri.path(), "/");
    }

    #[tokio::test]
    async fn resolve_url_rejects_ftp_scheme() {
        let err = resolve_url("ftp://127.0.0.1/data.csv").await.unwrap_err();
        assert!(
            err.to_string().contains("unsupported scheme"),
            "ftp should be rejected: {err}"
        );
    }

    #[test]
    fn resolve_url_rejects_file_scheme() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt.block_on(resolve_url("file:///etc/passwd")).unwrap_err();
        assert!(
            err.to_string().contains("sub-request"),
            "file:// should be rejected: {err}"
        );
    }

    #[tokio::test]
    async fn resolve_url_unresolvable_host_returns_connect_error() {
        let result = resolve_url("https://this-host-does-not-exist.invalid/path").await;
        assert!(
            matches!(result, Err(SubRequestError::Connect(_))),
            "unresolvable host should return Connect error: {result:?}"
        );
    }

    #[tokio::test]
    async fn resolve_url_localhost() {
        let (peer, uri) = resolve_url("http://127.0.0.1:8080/path").await.unwrap();
        assert!(!peer.is_tls());
        assert_eq!(uri.path(), "/path");
    }
}
