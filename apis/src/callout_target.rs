// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared target and address policy for outbound HTTP callouts.
//!
//! Operator-configured targets are syntax-checked when a pipeline is built
//! and their address policy is enforced again after DNS resolution, directly
//! before the validated socket addresses are handed to the transport.  This
//! closes the DNS-rebinding gap left by startup-only URL validation.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use praxis_core::connectivity::normalize_mapped_ipv4;
use praxis_filter::FilterError;

use crate::openai::url_security::is_non_public_ip;

/// Whether a configured callout may connect to non-public addresses.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AddressPolicy {
    /// Only publicly routable addresses are accepted.
    #[default]
    PublicOnly,
    /// Private, loopback, link-local, and other non-public addresses are
    /// accepted because the operator explicitly opted in.
    AllowPrivate,
}

impl AddressPolicy {
    /// Build a policy from a filter's explicit private-target opt-in.
    #[must_use]
    pub const fn from_allow_private(allow_private: bool) -> Self {
        if allow_private {
            Self::AllowPrivate
        } else {
            Self::PublicOnly
        }
    }

    /// Return whether non-public addresses are allowed.
    #[must_use]
    pub const fn allows_private(self) -> bool {
        matches!(self, Self::AllowPrivate)
    }
}

/// Validate an operator-configured HTTP target's URL structure.
///
/// Address classification is deliberately deferred until immediately after
/// DNS resolution.  Embedded credentials are always rejected, regardless of
/// address policy.
///
/// # Errors
///
/// Returns [`FilterError`] when the URL is malformed, is not HTTP(S), lacks a
/// host, contains userinfo, or contains a fragment.
pub fn validate_http_target(filter_name: &str, raw: &str) -> Result<url::Url, FilterError> {
    let parsed = url::Url::parse(raw)
        .map_err(|error| -> FilterError { format!("{filter_name}: target URL is not valid: {error}").into() })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!("{filter_name}: target URL must use http or https").into());
    }
    if parsed.host().is_none() {
        return Err(format!("{filter_name}: target URL must include a host").into());
    }
    if raw_authority_contains_userinfo_delimiter(raw) || !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("{filter_name}: target URL must not contain embedded credentials").into());
    }
    if parsed.fragment().is_some() {
        return Err(format!("{filter_name}: target URL must not contain a fragment").into());
    }
    Ok(parsed)
}

/// Return whether the raw URL authority contains the userinfo delimiter.
///
/// `url::Url` normalizes empty userinfo (`http://@example.com`) to an empty
/// username with no password, so checking the parsed credential values alone
/// cannot distinguish it from a URL without userinfo.  Limit the scan to the
/// authority so `@` in a path, query, or fragment is not treated as userinfo.
fn raw_authority_contains_userinfo_delimiter(raw: &str) -> bool {
    let Some((_, authority_and_rest)) = raw.split_once("://") else {
        return false;
    };
    let authority = authority_and_rest.split(['/', '?', '#']).next().unwrap_or_default();
    authority.contains('@')
}

/// Validate a configured target, including address literals and localhost
/// aliases that can be classified without DNS.
///
/// # Errors
///
/// Returns [`FilterError`] for an invalid HTTP target or when a statically
/// classifiable host violates `policy`.
pub fn validate_configured_http_target(
    filter_name: &str,
    raw: &str,
    policy: AddressPolicy,
) -> Result<url::Url, FilterError> {
    let parsed = validate_http_target(filter_name, raw)?;
    let host = parsed.host_str().unwrap_or_default();
    let host_without_brackets = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    if !policy.allows_private()
        && host_without_brackets
            .trim_end_matches('.')
            .eq_ignore_ascii_case("localhost")
    {
        return Err(
            format!("{filter_name}: target URL targets localhost; enable the private-target opt-in to allow").into(),
        );
    }
    if let Ok(ip) = host_without_brackets.parse::<IpAddr>() {
        validate_ip(filter_name, ip, policy)?;
    } else if let Some(ip) = parse_legacy_ipv4_host(host_without_brackets) {
        validate_ip(filter_name, IpAddr::V4(ip), policy)?;
    }
    Ok(parsed)
}

/// Parse legacy IPv4 literals accepted by common libc resolvers.
fn parse_legacy_ipv4_host(host: &str) -> Option<Ipv4Addr> {
    let host = host.trim_end_matches('.');
    let parts: Vec<_> = host.split('.').collect();
    if parts.is_empty() || parts.len() > 4 || parts.iter().any(|part| part.is_empty()) {
        return None;
    }

    let mut numbers = Vec::with_capacity(parts.len());
    for part in parts {
        numbers.push(parse_legacy_ipv4_number(part)?);
    }

    let addr = match numbers.as_slice() {
        [a] => *a,
        [a, b] if *a <= 0xFF && *b <= 0x00FF_FFFF => (*a << 24) | *b,
        [a, b, c] if *a <= 0xFF && *b <= 0xFF && *c <= 0xFFFF => (*a << 24) | (*b << 16) | *c,
        [a, b, c, d] if numbers.iter().all(|part| *part <= 0xFF) => (*a << 24) | (*b << 16) | (*c << 8) | *d,
        _ => return None,
    };

    Some(Ipv4Addr::from(addr))
}

/// Parse a decimal, octal, or hexadecimal legacy IPv4 component.
fn parse_legacy_ipv4_number(part: &str) -> Option<u32> {
    let (digits, radix) = part.strip_prefix("0x").or_else(|| part.strip_prefix("0X")).map_or_else(
        || {
            if part.len() > 1 && part.starts_with('0') {
                (part.get(1..).unwrap_or_default(), 8)
            } else {
                (part, 10)
            }
        },
        |digits| (digits, 16),
    );

    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }

    u32::from_str_radix(digits, radix).ok()
}

/// Validate every address returned by one DNS lookup.
///
/// The complete answer set is rejected when any address is disallowed. This
/// prevents a mixed public/private DNS response from becoming an address-
/// selection bypass.
///
/// # Errors
///
/// Returns [`FilterError`] when the set is empty or any address violates
/// `policy`.
pub fn validate_resolved_addrs(
    filter_name: &str,
    addrs: &[SocketAddr],
    policy: AddressPolicy,
) -> Result<Vec<SocketAddr>, FilterError> {
    if addrs.is_empty() {
        return Err(format!("{filter_name}: DNS returned no addresses").into());
    }

    let mut validated = Vec::with_capacity(addrs.len());
    let mut seen = std::collections::HashSet::with_capacity(addrs.len());
    for addr in addrs {
        let ip = normalize_mapped_ipv4(addr.ip());
        validate_ip(filter_name, ip, policy)?;
        let normalized = SocketAddr::new(ip, addr.port());
        if seen.insert(normalized) {
            validated.push(normalized);
        }
    }
    Ok(validated)
}

/// Validate one literal or DNS-resolved address.
///
/// # Errors
///
/// Returns [`FilterError`] when `ip` is non-public under
/// [`AddressPolicy::PublicOnly`].
pub fn validate_ip(filter_name: &str, ip: IpAddr, policy: AddressPolicy) -> Result<(), FilterError> {
    let ip = normalize_mapped_ipv4(ip);
    if !policy.allows_private() && is_non_public_ip(&ip) {
        return Err(format!(
            "{filter_name}: target resolved to blocked non-public address {ip}; set the filter's private-target opt-in to true to allow"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn target_rejects_userinfo_for_both_policies() {
        for target in [
            "https://user:pass@example.com/path",
            "http://@example.com",
            "http://:@example.com",
        ] {
            assert!(
                validate_http_target("test", target).is_err(),
                "{target} should be rejected"
            );
        }
    }

    #[test]
    fn target_allows_at_sign_outside_authority() {
        for target in ["https://example.com/@path", "https://example.com/?q=@value"] {
            assert!(
                validate_http_target("test", target).is_ok(),
                "{target} should be accepted"
            );
        }
    }

    #[test]
    fn public_only_rejects_legacy_ipv4_literals() {
        for host in ["127.1", "2130706433", "0x7f.0.0.1", "0177.0.0.1", "0x7f000001"] {
            assert!(
                validate_configured_http_target("test", &format!("http://{host}:8080"), AddressPolicy::PublicOnly)
                    .is_err(),
                "legacy IPv4 literal {host} should be rejected"
            );
        }
    }

    #[test]
    fn mixed_dns_answer_is_rejected() {
        let addrs = ["8.8.8.8:443".parse().unwrap(), "169.254.169.254:443".parse().unwrap()];
        assert!(
            validate_resolved_addrs("test", &addrs, AddressPolicy::PublicOnly).is_err(),
            "mixed public and private DNS answers should be rejected"
        );
    }

    #[test]
    fn private_opt_in_accepts_private_answers() {
        let addrs = ["127.0.0.1:8080".parse().unwrap(), "10.0.0.1:8080".parse().unwrap()];
        assert_eq!(
            validate_resolved_addrs("test", &addrs, AddressPolicy::AllowPrivate).unwrap(),
            addrs
        );
    }

    #[test]
    fn mapped_loopback_is_rejected() {
        let addrs = ["[::ffff:127.0.0.1]:80".parse().unwrap()];
        assert!(
            validate_resolved_addrs("test", &addrs, AddressPolicy::PublicOnly).is_err(),
            "IPv4-mapped loopback answers should be rejected"
        );
    }

    #[cfg(any(feature = "callout-rustls", feature = "callout-native-tls"))]
    #[allow(clippy::too_many_lines, reason = "test helper: cert generation + server setup")]
    fn spawn_tls_server() -> (u16, reqwest::Certificate, tokio::task::JoinHandle<()>) {
        use std::sync::Arc;

        let ca_key = rcgen::KeyPair::generate().unwrap();
        let mut ca_params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params.distinguished_name.push(rcgen::DnType::CommonName, "test-ca");
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();
        let issuer = rcgen::Issuer::from_params(&ca_params, &ca_key);

        let server_key = rcgen::KeyPair::generate().unwrap();
        let mut server_params = rcgen::CertificateParams::new(vec!["localhost".to_owned()]).unwrap();
        server_params.is_ca = rcgen::IsCa::NoCa;
        let server_cert = server_params.signed_by(&server_key, &issuer).unwrap();

        let rustls_certs = vec![
            rustls::pki_types::CertificateDer::from(server_cert.der().to_vec()),
            rustls::pki_types::CertificateDer::from(ca_cert.der().to_vec()),
        ];
        let rustls_key = rustls::pki_types::PrivateKeyDer::try_from(server_key.serialize_der()).unwrap();

        let tls_config = Arc::new(
            rustls::ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(rustls_certs, rustls_key)
                .unwrap(),
        );

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();

        let handle = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

            let acceptor = tokio_rustls::TlsAcceptor::from(tls_config);
            let (tcp, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(tcp).await.unwrap();

            let mut buf = vec![0_u8; 4096];
            let n = tls.read(&mut buf).await.unwrap();
            assert!(n > 0, "expected HTTP request bytes");

            let body = "OK";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            tls.write_all(response.as_bytes()).await.unwrap();
            tls.shutdown().await.unwrap();
        });

        let root_cert = reqwest::Certificate::from_der(ca_cert.der()).unwrap();
        (port, root_cert, handle)
    }

    #[cfg(any(feature = "callout-rustls", feature = "callout-native-tls"))]
    #[tokio::test]
    async fn pinned_client_completes_tls_handshake() {
        praxis_tls::provider::install();

        let (port, root_cert, server_handle) = spawn_tls_server();

        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(std::time::Duration::from_secs(5))
            .tls_certs_only([root_cert])
            .resolve_to_addrs("localhost", &[addr])
            .build()
            .unwrap();

        let resp = client
            .get(format!("https://localhost:{port}/healthz"))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        assert_eq!(resp.text().await.unwrap(), "OK");

        server_handle.await.unwrap();
    }
}
