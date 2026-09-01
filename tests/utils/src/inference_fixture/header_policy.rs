// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Central replay-boundary HTTP header safety policy.

use std::collections::{BTreeMap, BTreeSet};

use http::{HeaderMap, HeaderName, HeaderValue, header};

use super::FixtureError;

/// Header names retained in committed wire fixtures.
const FIXTURE_HEADERS: [&str; 5] = [
    "content-type",
    "anthropic-version",
    "openai-beta",
    "request-id",
    "x-request-id",
];

/// Converts recorded headers to a transport-safe HTTP map.
pub(super) fn recorded_transport_headers(source: &BTreeMap<String, Vec<String>>) -> Result<HeaderMap, FixtureError> {
    let nominated = recorded_connection_nominations(source);
    let mut headers = HeaderMap::new();
    for (name, values) in source {
        let normalized = name.to_ascii_lowercase();
        if is_transport_unsafe(&normalized, &nominated) {
            continue;
        }
        let name = HeaderName::from_bytes(normalized.as_bytes())
            .map_err(|_source| runtime_error("replay header name is invalid"))?;
        for value in values {
            let value =
                HeaderValue::from_str(value).map_err(|_source| runtime_error("replay header value is invalid"))?;
            headers.append(&name, value);
        }
    }
    Ok(headers)
}

/// Validates every representable recorded header before a network boundary.
pub(super) fn validate_recorded_headers(source: &BTreeMap<String, Vec<String>>) -> Result<(), FixtureError> {
    for (name, values) in source {
        HeaderName::from_bytes(name.as_bytes()).map_err(|_source| runtime_error("replay header name is invalid"))?;
        for value in values {
            HeaderValue::from_str(value).map_err(|_source| runtime_error("replay header value is invalid"))?;
        }
    }
    for nomination in recorded_connection_nominations(source) {
        HeaderName::from_bytes(nomination.as_bytes())
            .map_err(|_source| runtime_error("replay Connection nomination is invalid"))?;
    }
    Ok(())
}

/// Projects recorded headers to the deterministic fixture allowlist.
pub(super) fn recorded_fixture_headers(source: BTreeMap<String, Vec<String>>) -> BTreeMap<String, Vec<String>> {
    let nominated = recorded_connection_nominations(&source);
    let mut headers = BTreeMap::new();
    for (name, mut values) in source {
        let normalized = name.to_ascii_lowercase();
        if is_transport_unsafe(&normalized, &nominated) || !is_fixture_header(&normalized) {
            continue;
        }
        headers.entry(normalized).or_insert_with(Vec::new).append(&mut values);
    }
    headers
}

/// Projects only canonical fixture header names without cloning their values.
pub(super) fn recorded_fixture_header_names(source: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let nominated = recorded_connection_nominations(source);
    source
        .keys()
        .map(|name| name.to_ascii_lowercase())
        .filter(|name| !is_transport_unsafe(name, &nominated) && is_fixture_header(name))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Captures an HTTP header map through the deterministic fixture allowlist.
pub(super) fn http_fixture_headers(headers: &HeaderMap) -> Result<BTreeMap<String, Vec<String>>, FixtureError> {
    let nominated = http_connection_nominations(headers)?;
    let mut captured = BTreeMap::new();
    for name in FIXTURE_HEADERS {
        if is_transport_unsafe(name, &nominated) {
            continue;
        }
        let mut values = Vec::new();
        for value in headers.get_all(name) {
            values.push(
                value
                    .to_str()
                    .map_err(|_source| runtime_error("replay header value is not text"))?
                    .to_owned(),
            );
        }
        if !values.is_empty() {
            captured.insert(name.to_owned(), values);
        }
    }
    Ok(captured)
}

/// Builds provider request headers without forwarding inbound credentials or hop state.
///
/// Configured outbound values replace all inbound values for the same name and
/// preserve their original per-name order. Credential headers are allowed only
/// in the configured map, which is never projected into a fixture.
pub(super) fn provider_request_headers(inbound: &HeaderMap, configured: &HeaderMap) -> Result<HeaderMap, FixtureError> {
    validate_provider_headers(configured)?;
    let mut forwarded = safe_http_headers(inbound)?;
    for name in configured.keys() {
        forwarded.remove(name);
        for value in configured.get_all(name) {
            forwarded.append(name, value.clone());
        }
    }
    Ok(forwarded)
}

/// Builds downstream response headers without provider credentials or hop state.
pub(super) fn provider_response_headers(headers: &HeaderMap) -> Result<HeaderMap, FixtureError> {
    safe_http_headers(headers)
}

/// Validates configured outbound headers before a recording listener is bound.
pub(super) fn validate_provider_headers(headers: &HeaderMap) -> Result<(), FixtureError> {
    for name in headers.keys() {
        let normalized = name.as_str().to_ascii_lowercase();
        if normalized == "host" || is_framing_or_hop_header(&normalized) {
            return Err(runtime_error("recording provider target is invalid"));
        }
    }
    Ok(())
}

/// Returns whether one candidate byte string contains a configured credential.
///
/// Values under known credential names and values explicitly marked sensitive
/// are protected. In addition to the exact configured value, supported HTTP
/// authorization values protect their raw credential portion so a provider
/// cannot evade detection by omitting the scheme while echoing the secret.
pub(super) fn contains_configured_credential(configured: &HeaderMap, candidate: &[u8]) -> bool {
    configured.keys().any(|name| {
        configured.get_all(name).iter().any(|value| {
            if !is_credential_header(name.as_str()) && !value.is_sensitive() {
                return false;
            }
            let value = value.as_bytes();
            contains_nonempty(candidate, value)
                || is_authorization_header(name.as_str())
                    && authorization_credential(value)
                        .is_some_and(|credential| contains_nonempty(candidate, credential))
        })
    })
}

/// Returns whether any safe response header value reflects a credential.
pub(super) fn headers_contain_configured_credential(configured: &HeaderMap, response: &HeaderMap) -> bool {
    response
        .values()
        .any(|value| contains_configured_credential(configured, value.as_bytes()))
}

/// Returns whether a header name is known to carry credentials.
pub(super) fn is_credential_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization" | "proxy-authorization" | "cookie" | "set-cookie" | "x-api-key" | "api-key" | "x-goog-api-key"
    )
}

/// Returns whether a header uses the HTTP authorization value grammar.
fn is_authorization_header(name: &str) -> bool {
    name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("proxy-authorization")
}

/// Borrows the credential portion of a supported authorization value.
fn authorization_credential(value: &[u8]) -> Option<&[u8]> {
    let separator = value.iter().position(u8::is_ascii_whitespace)?;
    let scheme = value.get(..separator)?;
    if ![b"bearer".as_slice(), b"basic".as_slice()]
        .into_iter()
        .any(|supported| scheme.eq_ignore_ascii_case(supported))
    {
        return None;
    }
    let credential = value.get(separator..)?.trim_ascii();
    (!credential.is_empty()).then_some(credential)
}

/// Searches for a nonempty exact byte substring without allocating.
fn contains_nonempty(candidate: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && candidate.windows(needle.len()).any(|window| window == needle)
}

/// Extracts every mixed-case, repeated recorded `Connection` nomination.
fn recorded_connection_nominations(headers: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("connection"))
        .flat_map(|(_, values)| values)
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Extracts every repeated HTTP `Connection` nomination.
fn http_connection_nominations(headers: &HeaderMap) -> Result<BTreeSet<String>, FixtureError> {
    let mut nominated = BTreeSet::new();
    for value in headers.get_all(header::CONNECTION) {
        let value = value
            .to_str()
            .map_err(|_source| runtime_error("replay Connection header is not text"))?;
        nominated.extend(
            value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_ascii_lowercase),
        );
    }
    Ok(nominated)
}

/// Copies headers safe for one provider transport boundary.
fn safe_http_headers(headers: &HeaderMap) -> Result<HeaderMap, FixtureError> {
    let nominated = http_connection_nominations(headers)?;
    let mut forwarded = HeaderMap::new();
    for name in headers.keys() {
        let normalized = name.as_str().to_ascii_lowercase();
        if is_transport_unsafe(&normalized, &nominated) {
            continue;
        }
        for value in headers.get_all(name) {
            forwarded.append(name, value.clone());
        }
    }
    Ok(forwarded)
}

/// Returns whether a name must not cross a replay transport boundary.
fn is_transport_unsafe(name: &str, nominated: &BTreeSet<String>) -> bool {
    name == "host" || is_credential_header(name) || is_framing_or_hop_header(name) || nominated.contains(name)
}

/// Returns whether a name controls message framing or one HTTP hop.
fn is_framing_or_hop_header(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "content-length"
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

/// Returns whether a normalized name belongs in a fixture.
fn is_fixture_header(name: &str) -> bool {
    FIXTURE_HEADERS.contains(&name)
}

/// Creates a static replay error with no header contents.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::collections::BTreeMap;

    use http::{HeaderMap, HeaderValue, header};

    use super::{
        contains_configured_credential, http_fixture_headers, provider_response_headers, recorded_fixture_headers,
        recorded_transport_headers,
    };

    #[test]
    fn credential_scan_handles_repeated_empty_and_basic_authorization_values() {
        let mut repeated = HeaderMap::new();
        repeated.append("x-api-key", HeaderValue::from_static("first-configured-value"));
        repeated.append("x-api-key", HeaderValue::from_static("second-configured-value"));
        assert!(contains_configured_credential(
            &repeated,
            b"reflected second-configured-value"
        ));

        let mut empty = HeaderMap::new();
        empty.append("x-api-key", HeaderValue::from_static(""));
        assert!(!contains_configured_credential(&empty, b""));
        assert!(!contains_configured_credential(&empty, b"unrelated response"));

        let mut basic = HeaderMap::new();
        basic.append(header::AUTHORIZATION, HeaderValue::from_static("Basic dXNlcjpwYXNz"));
        assert!(contains_configured_credential(
            &basic,
            b"reflected dXNlcjpwYXNz without scheme"
        ));
    }

    #[test]
    fn credential_scan_honors_sensitive_custom_headers_without_matching_ordinary_values() {
        let mut configured = HeaderMap::new();
        configured.append("openai-beta", HeaderValue::from_static("ordinary-preview-value"));
        let mut secret = HeaderValue::from_static("custom-provider-secret");
        secret.set_sensitive(true);
        configured.append("x-auth-token", secret);

        assert!(contains_configured_credential(
            &configured,
            b"provider reflected custom-provider-secret in JSON"
        ));
        assert!(!contains_configured_credential(
            &configured,
            b"provider returned ordinary-preview-value in JSON"
        ));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one regression scenario verifies wire and fixture policy against repeated nominations and credentials"
    )]
    fn provider_response_strips_host_credentials_and_repeated_nominations_from_wire_and_fixture() {
        let mut source = HeaderMap::new();
        source.insert(header::HOST, HeaderValue::from_static("provider.internal"));
        source.append(header::CONNECTION, HeaderValue::from_static("X-Hop-One"));
        source.append(header::CONNECTION, HeaderValue::from_static("x-hop-two, keep-alive"));
        source.insert("x-hop-one", HeaderValue::from_static("remove-one"));
        source.insert("x-hop-two", HeaderValue::from_static("remove-two"));
        source.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer secret"));
        source.insert(header::SET_COOKIE, HeaderValue::from_static("session=secret"));
        source.insert("x-api-key", HeaderValue::from_static("secret"));
        source.append("request-id", HeaderValue::from_static("safe-first"));
        source.append("request-id", HeaderValue::from_static("safe-second"));

        let forwarded = provider_response_headers(&source).unwrap();
        let captured = http_fixture_headers(&source).unwrap();
        assert_eq!(
            forwarded
                .get_all("request-id")
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect::<Vec<_>>(),
            ["safe-first", "safe-second"]
        );
        assert_eq!(
            captured.get("request-id"),
            Some(&vec!["safe-first".to_owned(), "safe-second".to_owned()])
        );
        for removed in [
            "host",
            "connection",
            "keep-alive",
            "x-hop-one",
            "x-hop-two",
            "authorization",
            "set-cookie",
            "x-api-key",
        ] {
            assert!(!forwarded.contains_key(removed), "wire retained {removed}");
            assert!(!captured.contains_key(removed), "fixture retained {removed}");
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one table exercises every centralized credential, hop, nomination, and repeated-header rule"
    )]
    fn every_projection_removes_mixed_case_credentials_hop_headers_and_all_connection_nominations() {
        let source = BTreeMap::from([
            (
                "Connection".to_owned(),
                vec!["X-Remove-One, keep-alive, X-Request-Id".to_owned()],
            ),
            ("connection".to_owned(), vec!["x-remove-two".to_owned()]),
            ("X-Remove-One".to_owned(), vec!["one".to_owned()]),
            ("x-remove-two".to_owned(), vec!["two".to_owned()]),
            ("AUTHORIZATION".to_owned(), vec!["Bearer secret".to_owned()]),
            ("Proxy-Authorization".to_owned(), vec!["Basic secret".to_owned()]),
            ("Cookie".to_owned(), vec!["session=secret".to_owned()]),
            ("Set-Cookie".to_owned(), vec!["session=secret".to_owned()]),
            ("X-Api-Key".to_owned(), vec!["secret".to_owned()]),
            ("Api-Key".to_owned(), vec!["secret".to_owned()]),
            ("X-Goog-Api-Key".to_owned(), vec!["secret".to_owned()]),
            ("X-Safe".to_owned(), vec!["first".to_owned(), "second".to_owned()]),
            (
                "X-Request-Id".to_owned(),
                vec!["request-a".to_owned(), "request-b".to_owned()],
            ),
            ("Request-Id".to_owned(), vec!["safe-a".to_owned(), "safe-b".to_owned()]),
        ]);

        let transport = recorded_transport_headers(&source).unwrap();
        assert_eq!(
            transport
                .get_all("x-safe")
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        for removed in [
            "connection",
            "keep-alive",
            "x-remove-one",
            "x-remove-two",
            "x-request-id",
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
        ] {
            assert!(!transport.contains_key(removed), "transport retained {removed}");
        }

        let fixture = recorded_fixture_headers(source);
        assert_eq!(
            fixture.get("request-id"),
            Some(&vec!["safe-a".to_owned(), "safe-b".to_owned()])
        );
        assert!(!fixture.contains_key("x-request-id"));
        assert!(!fixture.contains_key("x-safe"));
        for removed in [
            "connection",
            "x-remove-one",
            "x-remove-two",
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
        ] {
            assert!(!fixture.contains_key(removed), "fixture retained {removed}");
        }
    }
}
