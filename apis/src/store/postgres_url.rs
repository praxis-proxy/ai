// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared `PostgreSQL` URL parsing and SSRF validation.
//!
//! Both `openai_response_store` and `openai_conversations` validate
//! `PostgreSQL` connection URLs against the same SSRF policy. This
//! module centralizes the target parser, host/IP classification,
//! Unix socket checks, and legacy IPv4 handling so the two filters
//! cannot drift.

use std::{
    borrow::Cow,
    net::{IpAddr, Ipv4Addr},
};

use percent_encoding::percent_decode_str;
use praxis_filter::{FilterError, has_dot_dot_traversal};

use crate::openai::url_security::is_non_public_ip;

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Validate a `PostgreSQL` connection URL against SSRF-sensitive targets.
///
/// Checks the URL scheme, authority host, `host` and `hostaddr` query
/// parameters, and rejects local-sensitive targets unless `allow_private`
/// is set.
///
/// `filter_name` is used as a prefix in error messages so each
/// consuming filter reports its own name.
pub(crate) fn validate_postgres_database_url(
    filter_name: &str,
    database_url: &str,
    allow_private: bool,
) -> Result<(), FilterError> {
    let Some(after_scheme) = postgres_url_after_scheme(database_url) else {
        return Err(
            format!("{filter_name}: postgres database_url must start with 'postgres://' or 'postgresql://'").into(),
        );
    };

    let mut has_explicit_target = false;
    if let Some(host) = postgres_authority_host(after_scheme) {
        has_explicit_target = true;
        validate_postgres_host_value(filter_name, "host", &host, allow_private)?;
    }

    for (key, value) in postgres_query_params(database_url) {
        if is_postgres_hostaddr_param(&key) {
            has_explicit_target = true;
            validate_postgres_hostaddr(filter_name, &value, allow_private)?;
        } else if is_postgres_host_param(&key) {
            has_explicit_target = true;
            validate_postgres_host_value(filter_name, &key, &value, allow_private)?;
        }
    }

    if !has_explicit_target {
        return Err(format!("{filter_name}: postgres database_url must include an explicit host").into());
    }

    Ok(())
}

/// Re-validate only the `PostgreSQL` host/IP portions of the connection
/// URL immediately before `SQLx` resolves and connects.
///
/// Full config validation runs once at construction time. This narrower
/// check guards against DNS rebinding between validation and connection
/// by re-checking the SSRF-sensitive host rules on every retry without
/// redundantly re-validating immutable fields (table names, SSL config,
/// URL scheme).
pub(crate) fn revalidate_postgres_host(
    filter_name: &str,
    database_url: &str,
    allow_private: bool,
) -> Result<(), FilterError> {
    let Some(after_scheme) = postgres_url_after_scheme(database_url) else {
        return Ok(());
    };
    if let Some(host) = postgres_authority_host(after_scheme) {
        validate_postgres_host_value(filter_name, "host", &host, allow_private)?;
    }
    for (key, value) in postgres_query_params(database_url) {
        if is_postgres_hostaddr_param(&key) {
            validate_postgres_hostaddr(filter_name, &value, allow_private)?;
        } else if is_postgres_host_param(&key) {
            validate_postgres_host_value(filter_name, &key, &value, allow_private)?;
        }
    }
    Ok(())
}

/// Validate `PostgreSQL` TLS file paths embedded in the connection URL.
pub(crate) fn validate_postgres_url_tls_file_params(filter_name: &str, database_url: &str) -> Result<(), FilterError> {
    for (key, value) in postgres_query_params(database_url) {
        if is_postgres_tls_file_param(&key) && has_dot_dot_traversal(&value) {
            return Err(
                format!("{filter_name}: database_url parameter '{key}' must not contain '..' path traversal").into(),
            );
        }
    }
    Ok(())
}

/// Extract a raw `sslmode` value from a `PostgreSQL` URL query string.
pub(crate) fn postgres_url_sslmode(database_url: &str) -> Option<String> {
    postgres_query_params(database_url)
        .filter(|(key, _)| is_postgres_sslmode_param(key))
        .map(|(_, value)| value.into_owned())
        .last()
}

/// Return whether a `PostgreSQL` URL contains a root CA certificate parameter.
pub(crate) fn has_postgres_url_ssl_root_cert(database_url: &str) -> bool {
    postgres_query_params(database_url).any(|(key, _)| is_postgres_ssl_root_cert_param(&key))
}

/// Return whether an `sslmode` value enables certificate verification.
pub(crate) fn is_verified_postgres_sslmode(value: &str) -> bool {
    value.eq_ignore_ascii_case("verify-ca") || value.eq_ignore_ascii_case("verify-full")
}

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

/// Return the URL portion after an accepted `PostgreSQL` scheme.
fn postgres_url_after_scheme(database_url: &str) -> Option<&str> {
    database_url
        .strip_prefix("postgres://")
        .or_else(|| database_url.strip_prefix("postgresql://"))
}

/// Extract the authority host from a `PostgreSQL` URL.
fn postgres_authority_host(after_scheme: &str) -> Option<Cow<'_, str>> {
    let authority = after_scheme.split(['/', '?', '#']).next().unwrap_or_default();
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, host)| host);
    if host_port.is_empty() {
        return None;
    }

    let host = if let Some(bracketed) = host_port.strip_prefix('[') {
        bracketed.split_once(']').map_or(host_port, |(host, _)| host)
    } else {
        host_port.rsplit_once(':').map_or(host_port, |(host, port)| {
            if port.bytes().all(|b| b.is_ascii_digit()) {
                host
            } else {
                host_port
            }
        })
    };
    if host.is_empty() {
        return None;
    }

    Some(percent_decode_str(host).decode_utf8_lossy())
}

/// Validate a `PostgreSQL` host value from authority or query params.
fn validate_postgres_host_value(
    filter_name: &str,
    kind: &str,
    host: &str,
    allow_private: bool,
) -> Result<(), FilterError> {
    if host.is_empty() {
        return Err(format!("{filter_name}: database_url {kind} must not be empty").into());
    }
    if host.starts_with('/') {
        return validate_postgres_socket_path(filter_name, kind, host, allow_private);
    }
    if !allow_private && is_postgres_localhost_name(host) {
        return Err(format!(
            "{filter_name}: database_url {kind} targets localhost; \
             set allow_private_database_url: true to allow"
        )
        .into());
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        validate_postgres_ip_target(filter_name, kind, ip, allow_private)?;
    } else if let Some(ip) = parse_legacy_ipv4_host(host) {
        validate_postgres_ip_target(filter_name, kind, IpAddr::V4(ip), allow_private)?;
    } else {
        validate_postgres_dns_target(filter_name, kind, host, allow_private)?;
    }
    Ok(())
}

/// Validate a `PostgreSQL` `hostaddr` query parameter.
fn validate_postgres_hostaddr(filter_name: &str, value: &str, allow_private: bool) -> Result<(), FilterError> {
    let ip = value
        .parse::<IpAddr>()
        .map_err(|e| format!("{filter_name}: database_url parameter 'hostaddr' must be a valid IP address: {e}"))?;
    validate_postgres_ip_target(filter_name, "hostaddr", ip, allow_private)
}

/// Validate a `PostgreSQL` IP target against the shared SSRF policy.
fn validate_postgres_ip_target(
    filter_name: &str,
    kind: &str,
    ip: IpAddr,
    allow_private: bool,
) -> Result<(), FilterError> {
    if !allow_private && is_non_public_ip(&ip) {
        return Err(format!(
            "{filter_name}: database_url {kind} targets a local-sensitive address; \
             set allow_private_database_url: true to allow"
        )
        .into());
    }
    Ok(())
}

/// Reject a `PostgreSQL` DNS hostname unless private targets are opted in.
fn validate_postgres_dns_target(
    filter_name: &str,
    kind: &str,
    host: &str,
    allow_private: bool,
) -> Result<(), FilterError> {
    if allow_private {
        return Ok(());
    }

    Err(format!(
        "{filter_name}: database_url {kind} host '{host}' is a DNS name; \
         use a literal IP address or set allow_private_database_url: true to allow DNS targets"
    )
    .into())
}

/// Validate a `PostgreSQL` Unix socket path.
fn validate_postgres_socket_path(
    filter_name: &str,
    kind: &str,
    path: &str,
    allow_private: bool,
) -> Result<(), FilterError> {
    if has_dot_dot_traversal(path) {
        return Err(format!("{filter_name}: database_url {kind} must not contain '..' path traversal").into());
    }
    if !allow_private {
        return Err(format!(
            "{filter_name}: database_url {kind} targets a Unix socket; \
             set allow_private_database_url: true to allow"
        )
        .into());
    }
    Ok(())
}

/// Return whether a host name resolves through the local loopback alias.
fn is_postgres_localhost_name(host: &str) -> bool {
    host.trim_end_matches('.').eq_ignore_ascii_case("localhost")
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

/// Iterate decoded query parameters from a `PostgreSQL` URL.
fn postgres_query_params(database_url: &str) -> impl Iterator<Item = (Cow<'_, str>, Cow<'_, str>)> + '_ {
    database_url
        .split_once('?')
        .map(|(_, query)| query.split_once('#').map_or(query, |(q, _)| q))
        .into_iter()
        .flat_map(|query| query.split('&'))
        .filter(|param| !param.is_empty())
        .map(|param| {
            let (key, value) = param.split_once('=').map_or((param, ""), |(k, v)| (k, v));
            (
                percent_decode_str(key).decode_utf8_lossy(),
                percent_decode_str(value).decode_utf8_lossy(),
            )
        })
}

/// Return whether a query key configures `PostgreSQL` host by address.
fn is_postgres_hostaddr_param(key: &str) -> bool {
    key == "hostaddr"
}

/// Return whether a query key configures `PostgreSQL` host.
fn is_postgres_host_param(key: &str) -> bool {
    key == "host"
}

/// Return whether a query key configures `PostgreSQL` TLS mode.
fn is_postgres_sslmode_param(key: &str) -> bool {
    key == "sslmode" || key == "ssl-mode"
}

/// Return whether a query key configures the `PostgreSQL` TLS root CA file.
fn is_postgres_ssl_root_cert_param(key: &str) -> bool {
    key == "sslrootcert" || key == "ssl-root-cert" || key == "ssl-ca"
}

/// Return whether a query key configures any `PostgreSQL` TLS file path.
fn is_postgres_tls_file_param(key: &str) -> bool {
    is_postgres_ssl_root_cert_param(key) || key == "sslcert" || key == "ssl-cert" || key == "sslkey" || key == "ssl-key"
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

    const FILTER: &str = "test_filter";

    // -- validate_postgres_database_url -----------------------------------------

    #[test]
    fn rejects_missing_scheme() {
        let err = validate_postgres_database_url(FILTER, "1.2.3.4:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("must start with"),
            "non-postgres scheme should be rejected"
        );
    }

    #[test]
    fn accepts_postgres_scheme() {
        validate_postgres_database_url(FILTER, "postgres://1.2.3.4:5432/db", false).unwrap();
    }

    #[test]
    fn accepts_postgresql_scheme() {
        validate_postgres_database_url(FILTER, "postgresql://1.2.3.4:5432/db", false).unwrap();
    }

    #[test]
    fn rejects_missing_explicit_host() {
        let err = validate_postgres_database_url(FILTER, "postgres:///db", false).unwrap_err();
        assert!(
            err.to_string().contains("explicit host"),
            "URL without host should be rejected"
        );
    }

    #[test]
    fn accepts_public_ipv4() {
        validate_postgres_database_url(FILTER, "postgres://user:pass@1.2.3.4:5432/db", false).unwrap();
    }

    // -- IP classification (shared is_non_public_ip policy) --------------------

    #[test]
    fn rejects_loopback_ipv4() {
        let err = validate_postgres_database_url(FILTER, "postgres://127.0.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "IPv4 loopback should be rejected"
        );
    }

    #[test]
    fn rejects_loopback_ipv6() {
        let err = validate_postgres_database_url(FILTER, "postgres://[::1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "IPv6 loopback should be rejected"
        );
    }

    #[test]
    fn rejects_private_ipv4() {
        for host in ["10.0.0.1", "172.16.0.1", "192.168.1.10"] {
            let err = validate_postgres_database_url(FILTER, &format!("postgres://{host}:5432/db"), false).unwrap_err();
            assert!(err.to_string().contains("local-sensitive"), "private {host}");
        }
    }

    #[test]
    fn rejects_link_local_ipv4() {
        let err = validate_postgres_database_url(FILTER, "postgres://169.254.1.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "link-local IPv4 should be rejected"
        );
    }

    #[test]
    fn rejects_unspecified_ipv4() {
        let err = validate_postgres_database_url(FILTER, "postgres://0.0.0.0:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "unspecified IPv4 should be rejected"
        );
    }

    #[test]
    fn rejects_cgnat_range() {
        let err = validate_postgres_database_url(FILTER, "postgres://100.64.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "CGNAT 100.64.0.0/10 should be rejected"
        );
    }

    #[test]
    fn rejects_tailscale_magic_dns() {
        let err = validate_postgres_database_url(FILTER, "postgres://100.100.100.200:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "Tailscale MagicDNS endpoint should be rejected"
        );
    }

    #[test]
    fn rejects_aws_imds_metadata() {
        let err = validate_postgres_database_url(FILTER, "postgres://169.254.169.254:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "AWS IMDS metadata endpoint should be rejected"
        );
    }

    #[test]
    fn rejects_multicast_ipv4() {
        let err = validate_postgres_database_url(FILTER, "postgres://224.0.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "multicast IPv4 should be rejected"
        );
    }

    #[test]
    fn rejects_documentation_range() {
        let err = validate_postgres_database_url(FILTER, "postgres://192.0.2.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "TEST-NET documentation range should be rejected"
        );
    }

    #[test]
    fn rejects_class_e_ipv4() {
        let err = validate_postgres_database_url(FILTER, "postgres://240.0.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "class E (240+) should be rejected"
        );
    }

    #[test]
    fn rejects_ipv6_unique_local() {
        let err = validate_postgres_database_url(FILTER, "postgres://[fd00::1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "IPv6 unique-local should be rejected"
        );
    }

    #[test]
    fn rejects_ipv6_link_local() {
        let err = validate_postgres_database_url(FILTER, "postgres://[fe80::1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "IPv6 link-local should be rejected"
        );
    }

    #[test]
    fn rejects_ipv6_site_local() {
        let err = validate_postgres_database_url(FILTER, "postgres://[fec0::1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "deprecated site-local IPv6 should be rejected"
        );
    }

    #[test]
    fn rejects_ipv4_mapped_loopback() {
        let err = validate_postgres_database_url(FILTER, "postgres://[::ffff:127.0.0.1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "IPv4-mapped loopback should be rejected"
        );
    }

    #[test]
    fn rejects_nat64_embedded_loopback() {
        let err = validate_postgres_database_url(FILTER, "postgres://[64:ff9b::127.0.0.1]:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "NAT64-embedded private IPv4 should be rejected"
        );
    }

    // -- Legacy IPv4 parsing ---------------------------------------------------

    #[test]
    fn rejects_legacy_ipv4_loopback_variants() {
        for host in [
            "127.1",
            "2130706433",
            "0x7f.0.0.1",
            "0177.0.0.1",
            "0",
            "0xa9fea9fe",
            "0x0a000005",
        ] {
            let err = validate_postgres_database_url(FILTER, &format!("postgres://{host}:5432/db"), false).unwrap_err();
            assert!(
                err.to_string().contains("local-sensitive"),
                "legacy IPv4 {host} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_legacy_ipv4_cgnat() {
        let err = validate_postgres_database_url(FILTER, "postgres://0x64400001:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "legacy hex CGNAT should be rejected"
        );
    }

    // -- DNS and localhost -----------------------------------------------------

    #[test]
    fn rejects_dns_name() {
        let err = validate_postgres_database_url(FILTER, "postgres://db.example.net:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("DNS name"),
            "DNS hostname should be rejected without opt-in"
        );
    }

    #[test]
    fn rejects_localhost() {
        let err = validate_postgres_database_url(FILTER, "postgres://LOCALHOST.:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("localhost"),
            "localhost hostname should be rejected"
        );
    }

    // -- Query parameter overrides ---------------------------------------------

    #[test]
    fn rejects_hostaddr_loopback_override() {
        let err =
            validate_postgres_database_url(FILTER, "postgres://1.2.3.4:5432/db?hostaddr=127.0.0.1", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "hostaddr loopback override should be rejected"
        );
    }

    #[test]
    fn rejects_host_loopback_override() {
        let err =
            validate_postgres_database_url(FILTER, "postgres://1.2.3.4:5432/db?host=127.0.0.1", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "host query param loopback override should be rejected"
        );
    }

    #[test]
    fn rejects_hostaddr_cgnat() {
        let err = validate_postgres_database_url(FILTER, "postgres://1.2.3.4:5432/db?hostaddr=100.64.0.1", false)
            .unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "CGNAT hostaddr override should be rejected"
        );
    }

    #[test]
    fn rejects_hostaddr_cloud_metadata() {
        let err = validate_postgres_database_url(FILTER, "postgres://1.2.3.4:5432/db?hostaddr=100.100.100.200", false)
            .unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "cloud metadata hostaddr override should be rejected"
        );
    }

    // -- Unix socket -----------------------------------------------------------

    #[test]
    fn rejects_unix_socket() {
        let err = validate_postgres_database_url(FILTER, "postgres:///db?host=/var/run/postgresql", false).unwrap_err();
        assert!(
            err.to_string().contains("Unix socket"),
            "Unix socket path should be rejected without opt-in"
        );
    }

    #[test]
    fn rejects_socket_path_traversal_even_with_allow_private() {
        let err =
            validate_postgres_database_url(FILTER, "postgres:///db?host=/var/run/../postgresql", true).unwrap_err();
        assert!(
            err.to_string().contains("path traversal"),
            "socket path traversal should be rejected even with opt-in"
        );
    }

    // -- allow_private opt-in --------------------------------------------------

    #[test]
    fn allows_loopback_with_opt_in() {
        validate_postgres_database_url(FILTER, "postgres://127.0.0.1:5432/db", true).unwrap();
    }

    #[test]
    fn allows_dns_with_opt_in() {
        validate_postgres_database_url(FILTER, "postgres://db.example.net:5432/db", true).unwrap();
    }

    #[test]
    fn allows_cgnat_with_opt_in() {
        validate_postgres_database_url(FILTER, "postgres://100.64.0.1:5432/db", true).unwrap();
    }

    #[test]
    fn allows_unix_socket_with_opt_in() {
        validate_postgres_database_url(FILTER, "postgres:///db?host=/var/run/postgresql", true).unwrap();
    }

    // -- revalidate_postgres_host ----------------------------------------------

    #[test]
    fn revalidate_rejects_private_ip() {
        let err = revalidate_postgres_host(FILTER, "postgres://10.0.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "revalidation should reject private IP"
        );
    }

    #[test]
    fn revalidate_rejects_cgnat() {
        let err = revalidate_postgres_host(FILTER, "postgres://100.64.0.1:5432/db", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "revalidation should reject CGNAT"
        );
    }

    #[test]
    fn revalidate_rejects_hostaddr_param() {
        let err =
            revalidate_postgres_host(FILTER, "postgres://1.2.3.4:5432/db?hostaddr=192.168.0.1", false).unwrap_err();
        assert!(
            err.to_string().contains("local-sensitive"),
            "revalidation should reject private hostaddr param"
        );
    }

    #[test]
    fn revalidate_accepts_public_ip() {
        revalidate_postgres_host(FILTER, "postgres://1.2.3.4:5432/db", false).unwrap();
    }

    #[test]
    fn revalidate_skips_non_postgres_url() {
        revalidate_postgres_host(FILTER, "sqlite::memory:", false).unwrap();
    }

    // -- filter_name propagation -----------------------------------------------

    #[test]
    fn error_messages_include_filter_name() {
        let err =
            validate_postgres_database_url("openai_response_store", "postgres://127.0.0.1/db", false).unwrap_err();
        assert!(
            err.to_string().starts_with("openai_response_store:"),
            "error should carry the filter name prefix"
        );

        let err = validate_postgres_database_url("openai_conversations", "postgres://127.0.0.1/db", false).unwrap_err();
        assert!(
            err.to_string().starts_with("openai_conversations:"),
            "error should carry the filter name prefix"
        );
    }

    // -- TLS file validation ---------------------------------------------------

    #[test]
    fn rejects_tls_file_path_traversal() {
        let err = validate_postgres_url_tls_file_params(FILTER, "postgres://1.2.3.4/db?sslrootcert=../../etc/ca.pem")
            .unwrap_err();
        assert!(
            err.to_string().contains("path traversal"),
            "sslrootcert path traversal should be rejected"
        );
    }

    #[test]
    fn rejects_sslkey_path_traversal() {
        let err = validate_postgres_url_tls_file_params(FILTER, "postgres://1.2.3.4/db?sslkey=../../etc/key.pem")
            .unwrap_err();
        assert!(
            err.to_string().contains("path traversal"),
            "sslkey path traversal should be rejected"
        );
    }

    #[test]
    fn accepts_clean_tls_file_params() {
        validate_postgres_url_tls_file_params(
            FILTER,
            "postgres://1.2.3.4/db?sslrootcert=/etc/ca.pem&sslkey=/etc/key.pem",
        )
        .unwrap();
    }
}
