// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared outbound hop-by-hop HTTP header policy.
//!
//! RFC 9110 hop-by-hop field names plus the obsolete `Proxy-Connection`
//! name. Outbound paths that copy headers onto a newly constructed request
//! must consult [`is_hop_by_hop`] and strip every field named by a
//! [`Connection`](http::header::CONNECTION) token list before dropping
//! `Connection` itself.
//!
//! Callers keep path-specific denylists (`Host`, `Content-Length`, cookies,
//! internal `x-praxis-*` prefixes) on top of this predicate. Do not copy
//! those extras into this module: they are not hop-by-hop.

use http::{HeaderMap, HeaderName};

/// Whether `name` is a hop-by-hop field that must not be copied onto a
/// newly constructed outbound request.
///
/// `name` must be a lowercase HTTP field name, as produced by
/// [`HeaderName::as_str`](http::HeaderName::as_str).
///
/// ```
/// assert!(praxis_ai_apis::http_hop::is_hop_by_hop("keep-alive"));
/// assert!(praxis_ai_apis::http_hop::is_hop_by_hop("proxy-connection"));
/// assert!(!praxis_ai_apis::http_hop::is_hop_by_hop("authorization"));
/// assert!(!praxis_ai_apis::http_hop::is_hop_by_hop("host"));
/// ```
#[must_use]
pub fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name,
        "connection"
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

/// Iterate hop-by-hop field names listed in one `Connection` header value.
///
/// Splits on commas, trims ASCII whitespace, and drops empty tokens.
/// Token comparison is the caller's responsibility; see
/// [`connection_nominates`].
pub fn connection_tokens(value: &str) -> impl Iterator<Item = &str> {
    value.split(',').map(str::trim).filter(|token| !token.is_empty())
}

/// Whether one `Connection` header value nominates `name` as hop-by-hop.
#[must_use]
pub fn connection_nominates(value: &str, name: &str) -> bool {
    connection_tokens(value).any(|token| token.eq_ignore_ascii_case(name))
}

/// Whether any `Connection` value on `headers` nominates `name`.
///
/// Invalid (non-text) `Connection` values are ignored rather than failing
/// open: a header that cannot be parsed as tokens cannot nominate fields.
#[must_use]
pub fn connection_nominates_header(headers: &HeaderMap, name: &HeaderName) -> bool {
    headers
        .get_all(http::header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .any(|value| connection_nominates(value, name.as_str()))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use http::{HeaderMap, HeaderName, HeaderValue};

    use super::{connection_nominates, connection_nominates_header, connection_tokens, is_hop_by_hop};

    #[test]
    fn hop_by_hop_covers_rfc_names_and_proxy_connection() {
        for name in [
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "proxy-connection",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        ] {
            assert!(is_hop_by_hop(name), "{name} is hop-by-hop");
        }
        for name in ["authorization", "cookie", "content-length", "host", "x-custom"] {
            assert!(!is_hop_by_hop(name), "{name} is not hop-by-hop");
        }
    }

    #[test]
    fn connection_tokens_split_trim_and_drop_empty() {
        let tokens: Vec<&str> = connection_tokens(" x-smuggle , , Keep-Alive ").collect();
        assert_eq!(
            tokens,
            ["x-smuggle", "Keep-Alive"],
            "Connection tokens should be comma-split and trimmed"
        );
    }

    #[test]
    fn connection_nominates_is_case_insensitive() {
        assert!(
            connection_nominates("X-Smuggle, close", "x-smuggle"),
            "Connection tokens are case-insensitive"
        );
        assert!(
            !connection_nominates("close", "x-custom"),
            "unrelated names must not be treated as nominated"
        );
    }

    #[test]
    fn connection_nominates_header_reads_every_connection_value() {
        let mut headers = HeaderMap::new();
        headers.append(http::header::CONNECTION, HeaderValue::from_static("X-Hop-One"));
        headers.append(
            http::header::CONNECTION,
            HeaderValue::from_static("x-hop-two, keep-alive"),
        );
        let hop_one = HeaderName::from_static("x-hop-one");
        let hop_two = HeaderName::from_static("x-hop-two");
        let custom = HeaderName::from_static("x-custom");
        assert!(
            connection_nominates_header(&headers, &hop_one),
            "first Connection value should nominate x-hop-one"
        );
        assert!(
            connection_nominates_header(&headers, &hop_two),
            "second Connection value should nominate x-hop-two"
        );
        assert!(
            !connection_nominates_header(&headers, &custom),
            "x-custom is not listed in Connection"
        );
    }
}
