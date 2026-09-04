// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! URL construction, resource-ID encoding, SSRF validation, and
//! forward-header validation for OpenAI-compatible API clients.
//!
//! These functions are shared by every filter that makes callouts
//! to an OpenAI-compatible API (Files API, vector-store search).
//! Each filter calls the validation helpers during its own config
//! validation phase, passing its filter name for error messages.

use std::collections::HashSet;

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use praxis_filter::FilterError;

use super::error::ApiClientError;
use crate::callout_target::{AddressPolicy, validate_configured_http_target};

/// Characters that could let a client-supplied resource ID escape
/// its single URL path segment.
///
/// Encoding path separators, query/fragment delimiters, `%`, and
/// URL parser special characters keeps the ID opaque when it is
/// appended to a path prefix like `/v1/files/` or
/// `/v1/vector_stores/`. Dots are encoded as an additional defense
/// against path normalization; exact `.` and `..` IDs are rejected
/// by [`resource_url`].
pub(crate) const RESOURCE_ID_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'.')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'[')
    .add(b'\\')
    .add(b']')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'}');

/// Build a resource URL with the ID encoded as a single path
/// segment.
///
/// Produces `{api_base_url}/{path_prefix}/{encoded_id}` when
/// `suffix` is `None`, or
/// `{api_base_url}/{path_prefix}/{encoded_id}/{suffix}` when
/// provided.
///
/// Rejects exact `.` and `..` resource IDs before encoding to
/// prevent path normalization attacks regardless of server
/// behavior.
pub(crate) fn resource_url(
    api_base_url: &str,
    path_prefix: &str,
    resource_id: &str,
    suffix: Option<&str>,
) -> Result<String, ApiClientError> {
    if matches!(resource_id, "." | "..") {
        return Err(ApiClientError::InvalidResourceId {
            resource_id: resource_id.to_owned(),
            detail: "dot path segments are not valid resource IDs".to_owned(),
        });
    }

    let encoded_id = utf8_percent_encode(resource_id, RESOURCE_ID_ENCODE_SET);
    match suffix {
        Some(s) => Ok(format!("{api_base_url}/{path_prefix}/{encoded_id}/{s}")),
        None => Ok(format!("{api_base_url}/{path_prefix}/{encoded_id}")),
    }
}

// -----------------------------------------------------------------------------
// Base URL validation (SSRF)
// -----------------------------------------------------------------------------

/// Validate a base URL against SSRF-sensitive targets.
///
/// Checks scheme, embedded credentials, query strings, fragments,
/// and literal host address. DNS names are accepted because the shared
/// transport validates and pins every resolved address at connect time.
///
/// `filter_name` is used as a prefix in error messages so each
/// consuming filter reports its own name.
pub(crate) fn validate_base_url(filter_name: &str, url: &str, allow_private: bool) -> Result<(), FilterError> {
    let parsed = validate_configured_http_target(filter_name, url, AddressPolicy::from_allow_private(allow_private))?;
    if parsed.query().is_some() {
        return Err(format!("{filter_name}: base URL must not contain a query string").into());
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Forward-header validation
// -----------------------------------------------------------------------------

/// Validate and normalize headers forwarded across an external API
/// security boundary.
///
/// Normalizes header names to lowercase, rejects transport and
/// internal headers, and rejects duplicates.
pub(crate) fn validate_forward_headers(filter_name: &str, headers: &mut [String]) -> Result<(), FilterError> {
    let mut seen = HashSet::with_capacity(headers.len());

    for configured in headers {
        let name = http::HeaderName::from_bytes(configured.as_bytes()).map_err(|e| -> FilterError {
            format!("{filter_name}: invalid 'forward_headers' entry '{configured}': {e}").into()
        })?;
        let normalized = name.as_str();

        if is_blocked_forward_header(normalized) {
            return Err(format!(
                "{filter_name}: 'forward_headers' must not include transport or internal header '{normalized}'"
            )
            .into());
        }

        if !seen.insert(normalized.to_owned()) {
            return Err(format!("{filter_name}: duplicate 'forward_headers' entry '{normalized}'").into());
        }

        normalized.clone_into(configured);
    }

    Ok(())
}

/// Return whether a header is unsafe to copy from the client
/// request to a newly constructed external API request.
fn is_blocked_forward_header(name: &str) -> bool {
    name.starts_with("x-praxis-")
        || name.starts_with("x-ext-protocol-")
        || name.starts_with("x-ext-agent-")
        || name.starts_with("x-mcp-")
        || name.starts_with("x-a2a-")
        || matches!(
            name,
            "connection"
                | "content-length"
                | "host"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "proxy-connection"
                | "te"
                | "trailer"
                | "transfer-encoding"
                | "upgrade"
        )
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
    use super::*;

    // -- resource_url --------------------------------------------------------

    #[test]
    fn resource_url_encodes_id_as_single_path_segment() {
        let url = resource_url("http://ogx:8321", "v1/files", "../admin?x#y", None).unwrap();

        assert!(
            url.starts_with("http://ogx:8321/v1/files/"),
            "URL should use the resource path prefix: {url}"
        );
        assert!(
            !url.contains("../admin"),
            "raw path traversal should not appear in URL: {url}"
        );
        assert!(!url.contains("?x"), "query delimiter should be encoded: {url}");
        assert!(!url.contains("#y"), "fragment delimiter should be encoded: {url}");
        assert!(url.contains("%2F"), "slash should be encoded: {url}");
        assert!(url.contains("%3F"), "question mark should be encoded: {url}");
        assert!(url.contains("%23"), "fragment marker should be encoded: {url}");
    }

    #[test]
    fn resource_url_rejects_exact_dot_dot_segment() {
        let err = resource_url("http://ogx:8321", "v1/files", "..", None).unwrap_err();

        assert!(
            matches!(err, ApiClientError::InvalidResourceId { .. }),
            "exact dot-dot resource id should be rejected before URL construction"
        );
    }

    #[test]
    fn resource_url_rejects_exact_dot_segment() {
        let err = resource_url("http://ogx:8321", "v1/files", ".", None).unwrap_err();

        assert!(
            matches!(err, ApiClientError::InvalidResourceId { .. }),
            "exact dot resource id should be rejected"
        );
    }

    #[test]
    fn resource_url_preserves_base_path_and_adds_suffix() {
        let url = resource_url("http://ogx:8321/files-api", "v1/files", "file-abc", Some("content")).unwrap();

        assert_eq!(
            url, "http://ogx:8321/files-api/v1/files/file-abc/content",
            "URL should preserve configured base path and append suffix"
        );
    }

    #[test]
    fn resource_url_without_suffix() {
        let url = resource_url("http://ogx:8321", "v1/files", "file-abc", None).unwrap();

        assert_eq!(url, "http://ogx:8321/v1/files/file-abc");
    }

    // -- SSRF validation -----------------------------------------------------

    #[test]
    fn ssrf_rejects_loopback_ipv4() {
        assert!(
            validate_base_url("test", "http://127.0.0.1:8321", false).is_err(),
            "loopback IPv4 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_loopback_ipv6() {
        assert!(
            validate_base_url("test", "http://[::1]:8321", false).is_err(),
            "loopback IPv6 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_private_ipv4() {
        assert!(
            validate_base_url("test", "http://10.0.0.1:8321", false).is_err(),
            "private IPv4 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_link_local_ipv4() {
        assert!(
            validate_base_url("test", "http://169.254.169.254", false).is_err(),
            "link-local IPv4 (metadata endpoint) should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_cgnat_ipv4() {
        assert!(
            validate_base_url("test", "http://100.64.0.1:8321", false).is_err(),
            "CGNAT IPv4 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_localhost_name() {
        assert!(
            validate_base_url("test", "http://localhost:8321", false).is_err(),
            "localhost name should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_allows_dns_name_for_connect_time_validation() {
        assert!(
            validate_base_url("test", "http://ogx:8321", false).is_ok(),
            "DNS names are classified and pinned by the transport at connect time"
        );
    }

    #[test]
    fn ssrf_rejects_legacy_octal_loopback() {
        assert!(
            validate_base_url("test", "http://0177.0.0.1:8321", false).is_err(),
            "octal-encoded loopback should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_shared_special_use_range() {
        assert!(
            validate_base_url("test", "http://192.0.2.1:8321", false).is_err(),
            "documentation ranges should be rejected by shared IP classification"
        );
    }

    #[test]
    fn ssrf_rejects_shared_cloud_metadata_endpoint() {
        assert!(
            validate_base_url("test", "http://100.100.100.200:8321", false).is_err(),
            "cloud metadata endpoints should be rejected by shared IP classification"
        );
    }

    #[test]
    fn ssrf_rejects_ipv4_mapped_ipv6_loopback() {
        assert!(
            validate_base_url("test", "http://[::ffff:127.0.0.1]:8321", false).is_err(),
            "IPv4-mapped IPv6 loopback should be rejected"
        );
    }

    #[test]
    fn ssrf_allows_public_ipv4() {
        assert!(
            validate_base_url("test", "http://8.8.8.8:8321", false).is_ok(),
            "public IPv4 should be allowed"
        );
    }

    #[test]
    fn ssrf_allows_public_ipv6() {
        assert!(
            validate_base_url("test", "https://[2606:4700:4700::1111]:8321", false).is_ok(),
            "bracketed public IPv6 should be recognized as an IP literal"
        );
    }

    #[test]
    fn ssrf_allows_private_with_override() {
        assert!(
            validate_base_url("test", "http://127.0.0.1:8321", true).is_ok(),
            "loopback should be allowed with allow_private"
        );
    }

    #[test]
    fn ssrf_allows_dns_with_override() {
        assert!(
            validate_base_url("test", "http://ogx:8321", true).is_ok(),
            "DNS name should be allowed with allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_non_http_scheme() {
        assert!(
            validate_base_url("test", "ftp://ogx:8321", false).is_err(),
            "non-http scheme should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_embedded_credentials() {
        assert!(
            validate_base_url("test", "http://user:password@ogx:8321", true).is_err(),
            "embedded URL credentials should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_query_string() {
        assert!(
            validate_base_url("test", "http://ogx:8321/base?tenant=abc", true).is_err(),
            "query strings should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_fragment() {
        assert!(
            validate_base_url("test", "http://ogx:8321/base#v2", true).is_err(),
            "fragments should be rejected"
        );
    }

    // -- forward-header validation -------------------------------------------

    #[test]
    fn forward_headers_are_normalized() {
        let mut headers = vec!["Authorization".to_owned(), "X-Tenant-ID".to_owned()];
        validate_forward_headers("test", &mut headers).unwrap();

        assert_eq!(
            headers,
            vec!["authorization", "x-tenant-id"],
            "forwarded header names should be normalized"
        );
    }

    #[test]
    fn invalid_forward_header_rejected() {
        let mut headers = vec!["bad header".to_owned()];
        assert!(
            validate_forward_headers("test", &mut headers).is_err(),
            "syntactically invalid forwarded header names should be rejected"
        );
    }

    #[test]
    fn unsafe_forward_headers_rejected() {
        for name in [
            "host",
            "content-length",
            "transfer-encoding",
            "proxy-authorization",
            "x-praxis-route",
        ] {
            let mut headers = vec![name.to_owned()];
            assert!(
                validate_forward_headers("test", &mut headers).is_err(),
                "unsafe forwarded header '{name}' should be rejected"
            );
        }
    }

    #[test]
    fn duplicate_forward_headers_rejected_case_insensitively() {
        let mut headers = vec!["Authorization".to_owned(), "authorization".to_owned()];
        assert!(
            validate_forward_headers("test", &mut headers).is_err(),
            "duplicate forwarded header names should be rejected after normalization"
        );
    }
}
