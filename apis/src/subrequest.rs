// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Policy-aware URL execution for outbound sub-requests.
//!
//! Praxis core owns URL parsing, DNS resolution, address-set validation, the
//! frozen-peer fallback, the single operation deadline, and the response bound
//! through [`SubRequestClient::execute_url`] and
//! [`prepare_url_target_with_policy`]. This module only adapts the AI callout
//! surface onto that executor:
//!
//! - [`address_policy_hook`] bridges the AI [`AddressPolicy`] into core's post-resolution validation hook, so the
//!   complete resolved address set is rejected atomically before any dial (closing the DNS-rebinding gap).
//! - [`map_url_error`] maps core's [`UrlSubRequestError`] back to [`SubRequestError`] for callers that already branch
//!   on that type, keeping the credential-safe diagnostics (`recently failed`, `blocked non-public address`) those
//!   callers assert on.
//!
//! Callers select the DNS resolution policy by the destination hostname's
//! provenance, not by the helper name: [`UrlResolutionPolicy::OperatorCached`]
//! for operator-configured hosts (metering, web/file search, MCP connectors,
//! compaction, token endpoints, generic callouts) and
//! [`UrlResolutionPolicy::ClientPerCall`] for client-selected hosts (input
//! file URLs, direct MCP `server_url` targets).
//!
//! Types ([`SubRequestClient`], [`SubRequest`], [`SubResponse`],
//! [`SubRequestError`], [`UrlSubRequestError`], [`FrameworkHeaders`]) and the
//! [`UrlResolutionPolicy`] selector are re-exported from Praxis core.
//!
//! [`prepare_url_target_with_policy`]: praxis_core::connectivity::prepare_url_target_with_policy

use std::net::SocketAddr;

use praxis_core::connectivity::UrlTargetError;
pub use praxis_core::{
    connectivity::UrlResolutionPolicy,
    subrequest::{FrameworkHeaders, SubRequest, SubRequestClient, SubRequestError, SubResponse, UrlSubRequestError},
};

use crate::callout_target::{AddressPolicy, validate_resolved_addrs};

/// Build an isolated client after installing the process-wide crypto provider.
///
/// Constructing a connector creates a rustls client configuration, so provider
/// installation must happen at this boundary rather than relying on a binary
/// entry point having run first.
pub(crate) fn isolated_client(pool_size: usize) -> SubRequestClient {
    praxis_tls::provider::install();
    SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(pool_size, None))
}

/// Bridge the AI [`AddressPolicy`] into core's post-resolution validation hook.
///
/// The returned closure is called at most once by
/// [`SubRequestClient::execute_url`] / [`prepare_url_target_with_policy`], on
/// the complete normalized and deduplicated address set, before any connection
/// attempt. Returning `Err` prevents every dial, so a mixed public/private DNS
/// answer cannot become an address-selection bypass. The error box carries the
/// [`validate_resolved_addrs`] diagnostic (e.g. `blocked non-public address`),
/// which core surfaces unchanged through [`UrlTargetError::PolicyRejected`].
///
/// [`prepare_url_target_with_policy`]: praxis_core::connectivity::prepare_url_target_with_policy
pub fn address_policy_hook(
    address_policy: AddressPolicy,
) -> impl FnOnce(&[SocketAddr]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send {
    move |addrs: &[SocketAddr]| validate_resolved_addrs("sub-request", addrs, address_policy).map(|_| ())
}

/// Map core's [`UrlSubRequestError`] back to [`SubRequestError`].
///
/// Callers that predate the policy-aware executor branch on [`SubRequestError`];
/// this preserves that surface. The mapping keeps the credential-safe
/// diagnostics intact: a resolver `recently failed` cache hit stays a
/// [`SubRequestError::Connect`], and a policy rejection surfaces the hook's own
/// `blocked non-public address` message (from the boxed source, not
/// [`UrlTargetError`]'s generic display).
#[must_use]
pub fn map_url_error(error: UrlSubRequestError) -> SubRequestError {
    match error {
        UrlSubRequestError::Exchange(inner) => inner,
        UrlSubRequestError::DeadlineExceeded => SubRequestError::DeadlineExceeded,
        UrlSubRequestError::Target(target) => map_target_error(target),
        // `UrlSubRequestError` is `#[non_exhaustive]`; a future variant keeps a
        // bounded, credential-safe fallback rather than leaking its display.
        _ => SubRequestError::Connect("sub-request URL preparation failed".to_owned()),
    }
}

/// Map a [`UrlTargetError`] (preparation failure) to [`SubRequestError`].
fn map_target_error(error: UrlTargetError) -> SubRequestError {
    match error {
        // A structurally invalid target (bad scheme, userinfo, fragment,
        // missing host, bad port/literal) is a request error, not transport.
        UrlTargetError::InvalidTarget(inner) => SubRequestError::InvalidRequest(inner.to_string()),
        // DNS failure, including a shared-cache `recently failed` replay.
        UrlTargetError::Resolve(inner) => SubRequestError::Connect(inner.to_string()),
        // Surface the validation hook's own message (`blocked non-public
        // address ...`), not the wrapper's generic display.
        UrlTargetError::PolicyRejected(source) => SubRequestError::Connect(source.to_string()),
        UrlTargetError::DeadlineExceeded => SubRequestError::DeadlineExceeded,
        // `UrlTargetError` is `#[non_exhaustive]`; keep a bounded fallback.
        _ => SubRequestError::Connect("sub-request target preparation failed".to_owned()),
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
        time::Duration,
    };

    use bytes::Bytes;
    use http::HeaderMap;
    use praxis_core::connectivity::{InvalidTarget, peer::AddressResolutionError};

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
    fn map_url_error_unwraps_exchange() {
        let mapped = map_url_error(UrlSubRequestError::Exchange(SubRequestError::ResponseTooLarge {
            actual: 4096,
            limit: 1024,
        }));
        assert!(
            matches!(mapped, SubRequestError::ResponseTooLarge { limit: 1024, .. }),
            "an exchange error must pass through unchanged: {mapped:?}"
        );
    }

    #[test]
    fn map_url_error_maps_deadline() {
        assert!(
            matches!(
                map_url_error(UrlSubRequestError::DeadlineExceeded),
                SubRequestError::DeadlineExceeded
            ),
            "the operation deadline must map to DeadlineExceeded"
        );
        assert!(
            matches!(
                map_url_error(UrlSubRequestError::Target(UrlTargetError::DeadlineExceeded)),
                SubRequestError::DeadlineExceeded
            ),
            "a preparation deadline must also map to DeadlineExceeded"
        );
    }

    #[test]
    fn map_url_error_maps_invalid_target_to_request_error() {
        let mapped = map_url_error(UrlSubRequestError::Target(UrlTargetError::InvalidTarget(
            InvalidTarget::UnsupportedScheme("ftp".to_owned()),
        )));
        assert!(
            matches!(mapped, SubRequestError::InvalidRequest(_)),
            "a structurally invalid target must map to InvalidRequest: {mapped:?}"
        );
    }

    #[test]
    fn map_url_error_preserves_recently_failed_diagnostic() {
        let resolve = AddressResolutionError::RecentFailure {
            address: "example.invalid:443".to_owned(),
            message: "cached".to_owned(),
        };
        let mapped = map_url_error(UrlSubRequestError::Target(UrlTargetError::Resolve(resolve)));
        assert!(
            matches!(&mapped, SubRequestError::Connect(detail) if detail.contains("recently failed")),
            "a cached resolver failure must keep its diagnostic: {mapped:?}"
        );
    }

    #[test]
    fn map_url_error_preserves_policy_rejection_diagnostic() {
        let source: Box<dyn std::error::Error + Send + Sync> =
            "sub-request: target resolved to blocked non-public address 127.0.0.1".into();
        let mapped = map_url_error(UrlSubRequestError::Target(UrlTargetError::PolicyRejected(source)));
        assert!(
            matches!(&mapped, SubRequestError::Connect(detail) if detail.contains("blocked non-public address")),
            "a policy rejection must surface the hook's own message: {mapped:?}"
        );
    }

    #[tokio::test]
    async fn operator_cached_execute_keeps_the_url_authority() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = capture_raw_request(listener);

        let response = test_client()
            .execute_url(
                &format!("http://localhost:{port}/test"),
                empty_request(),
                address_policy_hook(AddressPolicy::AllowPrivate),
                UrlResolutionPolicy::OperatorCached,
                Duration::from_secs(5),
                1024,
                None,
            )
            .await
            .map_err(map_url_error)
            .unwrap();

        assert_eq!(response.status, 200);
        let wire = captured.join().unwrap().to_lowercase();
        assert!(
            wire.contains(&format!("host: localhost:{port}")),
            "Host header should use the URL authority: {wire}"
        );
    }

    #[tokio::test]
    async fn address_policy_hook_rejects_loopback_under_public_only() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();

        let result = test_client()
            .execute_url(
                &format!("http://127.0.0.1:{port}/test"),
                empty_request(),
                address_policy_hook(AddressPolicy::PublicOnly),
                UrlResolutionPolicy::OperatorCached,
                Duration::from_secs(5),
                1024,
                None,
            )
            .await
            .map_err(map_url_error);

        assert!(
            matches!(&result, Err(SubRequestError::Connect(detail)) if detail.contains("blocked non-public address")),
            "the address policy must reject a loopback target before dialing: {result:?}"
        );
        assert!(
            matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "a rejected address set must not be dialed"
        );
    }

    #[tokio::test]
    async fn client_per_call_policy_executes_a_literal_target() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let captured = capture_raw_request(listener);

        let response = test_client()
            .execute_url(
                &format!("http://127.0.0.1:{port}/test"),
                empty_request(),
                address_policy_hook(AddressPolicy::AllowPrivate),
                UrlResolutionPolicy::ClientPerCall,
                Duration::from_secs(5),
                1024,
                None,
            )
            .await
            .map_err(map_url_error)
            .unwrap();

        assert_eq!(response.status, 200, "a client-selected literal must execute");
        let _wire = captured.join().unwrap();
    }
}
