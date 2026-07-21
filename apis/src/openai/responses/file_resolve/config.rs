// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the `openai_file_resolve` filter.

use std::{
    collections::HashSet,
    net::{IpAddr, Ipv4Addr},
};

use praxis_core::connectivity::normalize_mapped_ipv4;
use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

/// Default HTTP timeout for Files API callout requests (30 000 ms).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default maximum number of file references resolved per request.
const DEFAULT_MAX_FILE_REFERENCES: usize = 32;

/// Maximum configurable file reference count per request.
const MAX_CONFIGURABLE_FILE_REFERENCES: usize = 128;

/// Maximum allowed timeout (300 000 ms / 5 minutes).
const MAX_TIMEOUT_MS: u64 = 300_000;

/// Behavior when a referenced file cannot be fetched.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnMissing {
    /// Leave the `file_id` reference unchanged and continue.
    #[default]
    Continue,

    /// Return an error response to the client.
    Reject,
}

/// YAML configuration for the [`FileResolveFilter`].
///
/// [`FileResolveFilter`]: super::FileResolveFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileResolveConfig {
    /// Allow `files_api_url` to target private, loopback, link-local,
    /// or DNS-name hosts.  Default `false` rejects SSRF-sensitive
    /// targets; set to `true` in development or when the Files API is
    /// an internal service on a private network.
    #[serde(default)]
    pub allow_private_files_api_url: bool,

    /// Allow Files API callouts from the `StreamBuffer` pre-read
    /// phase, before header-phase security filters execute.
    ///
    /// This must be explicitly enabled only when an outer trust
    /// boundary authenticates and authorizes requests before they
    /// reach this listener. Forwarded headers are the original
    /// downstream values, not mutations from request filters.
    #[serde(default)]
    pub allow_pre_security_callout: bool,

    /// Base URL of the Files API (OGX) endpoint.
    ///
    /// Example: `http://ogx:8321`
    pub files_api_url: String,

    /// Headers to forward from the original request to the
    /// Files API for authentication and tenant isolation. No
    /// downstream headers are forwarded by default.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Maximum number of distinct content-part / `file_id` pairs to
    /// resolve in one request, including rehydrated history.
    #[serde(default = "default_max_file_references")]
    pub max_file_references: usize,

    /// Behavior when a referenced file cannot be fetched.
    #[serde(default)]
    pub on_missing: OnMissing,

    /// HTTP timeout in milliseconds for Files API callout requests.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Default maximum file reference count.
fn default_max_file_references() -> usize {
    DEFAULT_MAX_FILE_REFERENCES
}

/// Default timeout in milliseconds.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Validate the parsed configuration.
pub(crate) fn validate_config(mut cfg: FileResolveConfig) -> Result<FileResolveConfig, FilterError> {
    if cfg.files_api_url.is_empty() {
        return Err("openai_file_resolve: 'files_api_url' must not be empty".into());
    }

    if cfg.files_api_url.ends_with('/') {
        return Err("openai_file_resolve: 'files_api_url' must not end with '/'".into());
    }

    validate_files_api_url(&cfg.files_api_url, cfg.allow_private_files_api_url)?;
    validate_forward_headers(&mut cfg.forward_headers)?;
    validate_limits(&cfg)?;
    validate_pre_security_callout(&cfg)?;

    Ok(cfg)
}

/// Require explicit acknowledgement of the pre-read security boundary.
fn validate_pre_security_callout(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if !cfg.allow_pre_security_callout {
        return Err(
            "openai_file_resolve: 'allow_pre_security_callout' must be true because StreamBuffer body callouts run before header-phase security filters; place authentication and authorization in an outer trust boundary"
                .into(),
        );
    }
    Ok(())
}

/// Validate numeric limits applied while buffering and resolving.
fn validate_limits(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if cfg.max_body_bytes == 0 {
        return Err("openai_file_resolve: 'max_body_bytes' must be greater than 0".into());
    }

    if cfg.max_body_bytes > MAX_JSON_BODY_BYTES {
        return Err(format!(
            "openai_file_resolve: 'max_body_bytes' ({}) exceeds maximum ({MAX_JSON_BODY_BYTES})",
            cfg.max_body_bytes
        )
        .into());
    }

    validate_resolution_limits(cfg)
}

/// Validate callout count and time limits.
fn validate_resolution_limits(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if cfg.max_file_references == 0 {
        return Err("openai_file_resolve: 'max_file_references' must be greater than 0".into());
    }

    if cfg.max_file_references > MAX_CONFIGURABLE_FILE_REFERENCES {
        return Err(format!(
            "openai_file_resolve: 'max_file_references' ({}) exceeds maximum ({MAX_CONFIGURABLE_FILE_REFERENCES})",
            cfg.max_file_references
        )
        .into());
    }

    if cfg.timeout_ms == 0 {
        return Err("openai_file_resolve: 'timeout_ms' must be greater than 0".into());
    }

    if cfg.timeout_ms > MAX_TIMEOUT_MS {
        return Err(format!(
            "openai_file_resolve: 'timeout_ms' ({}) exceeds maximum ({MAX_TIMEOUT_MS})",
            cfg.timeout_ms
        )
        .into());
    }

    Ok(())
}

/// Validate `files_api_url` against SSRF-sensitive targets.
fn validate_files_api_url(url: &str, allow_private: bool) -> Result<(), FilterError> {
    if url.contains('#') {
        return Err("openai_file_resolve: 'files_api_url' must not contain a fragment".into());
    }

    let uri: http::Uri = url.parse().map_err(|e: http::uri::InvalidUri| -> FilterError {
        format!("openai_file_resolve: 'files_api_url' is not a valid URL: {e}").into()
    })?;

    match uri.scheme_str() {
        Some("http" | "https") => {},
        _ => {
            return Err("openai_file_resolve: 'files_api_url' must use http or https scheme".into());
        },
    }

    if uri
        .authority()
        .is_some_and(|authority| authority.as_str().contains('@'))
    {
        return Err("openai_file_resolve: 'files_api_url' must not contain embedded credentials".into());
    }

    if uri.query().is_some() {
        return Err("openai_file_resolve: 'files_api_url' must not contain a query string".into());
    }

    let host = uri
        .host()
        .ok_or_else(|| -> FilterError { "openai_file_resolve: 'files_api_url' must include a host".into() })?;

    validate_files_api_host(host, allow_private)
}

/// Validate and normalize headers forwarded across the Files API
/// security boundary.
fn validate_forward_headers(headers: &mut [String]) -> Result<(), FilterError> {
    let mut seen = HashSet::with_capacity(headers.len());

    for configured in headers {
        let name = http::HeaderName::from_bytes(configured.as_bytes()).map_err(|e| -> FilterError {
            format!("openai_file_resolve: invalid 'forward_headers' entry '{configured}': {e}").into()
        })?;
        let normalized = name.as_str();

        if is_blocked_forward_header(normalized) {
            return Err(format!(
                "openai_file_resolve: 'forward_headers' must not include transport or internal header '{normalized}'"
            )
            .into());
        }

        if !seen.insert(normalized.to_owned()) {
            return Err(format!("openai_file_resolve: duplicate 'forward_headers' entry '{normalized}'").into());
        }

        normalized.clone_into(configured);
    }

    Ok(())
}

/// Return whether a header is unsafe to copy from the client request
/// to a newly constructed Files API request.
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

/// Validate a `files_api_url` host value against SSRF-sensitive targets.
fn validate_files_api_host(host: &str, allow_private: bool) -> Result<(), FilterError> {
    if !allow_private && is_localhost_name(host) {
        return Err("openai_file_resolve: 'files_api_url' targets localhost; \
             set allow_private_files_api_url: true to allow"
            .into());
    }

    let ip_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = ip_host.parse::<IpAddr>() {
        validate_files_api_ip(ip, allow_private)?;
    } else if let Some(ip) = parse_legacy_ipv4_host(host) {
        validate_files_api_ip(IpAddr::V4(ip), allow_private)?;
    } else {
        validate_files_api_dns(host, allow_private)?;
    }

    Ok(())
}

/// Validate an IP target against SSRF-sensitive ranges.
fn validate_files_api_ip(ip: IpAddr, allow_private: bool) -> Result<(), FilterError> {
    let ip = normalize_mapped_ipv4(ip);
    if !allow_private && is_ssrf_sensitive_ip(&ip) {
        return Err(
            "openai_file_resolve: 'files_api_url' targets a local-sensitive address; \
             set allow_private_files_api_url: true to allow"
                .into(),
        );
    }
    Ok(())
}

/// Reject DNS hostnames unless private targets are opted in.
fn validate_files_api_dns(host: &str, allow_private: bool) -> Result<(), FilterError> {
    if allow_private {
        return Ok(());
    }
    Err(format!(
        "openai_file_resolve: 'files_api_url' host '{host}' is a DNS name; \
         use a literal IP address or set allow_private_files_api_url: true to allow DNS targets"
    )
    .into())
}

/// Return whether an IP address is SSRF-sensitive (loopback, private,
/// link-local, CGNAT/shared, current-network, or unspecified).
fn is_ssrf_sensitive_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.octets()[0] == 0
                || is_cgnat(*v4)
        },
        IpAddr::V6(v6) => v6.is_loopback() || v6.is_unique_local() || v6.is_unicast_link_local() || v6.is_unspecified(),
    }
}

/// Return whether an IPv4 address is in the shared CGNAT range
/// (`100.64.0.0/10`, RFC 6598).
fn is_cgnat(ip: Ipv4Addr) -> bool {
    u32::from(ip) & 0xFFC0_0000 == 0x6440_0000
}

/// Return whether a host name is a localhost alias.
fn is_localhost_name(host: &str) -> bool {
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

    const MINIMAL_YAML: &str = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
"#;

    #[test]
    fn minimal_config_parses() {
        let cfg: FileResolveConfig = serde_yaml::from_str(MINIMAL_YAML).unwrap();
        let validated = validate_config(cfg).unwrap();
        assert_eq!(validated.files_api_url, "http://ogx:8321", "files_api_url should match");
        assert_eq!(
            validated.max_body_bytes, MAX_JSON_BODY_BYTES,
            "max_body_bytes should default to 64 MiB"
        );
        assert_eq!(
            validated.timeout_ms, DEFAULT_TIMEOUT_MS,
            "timeout_ms should default to 30000"
        );
        assert_eq!(
            validated.max_file_references, DEFAULT_MAX_FILE_REFERENCES,
            "max_file_references should default to 32"
        );
        assert_eq!(
            validated.on_missing,
            OnMissing::Continue,
            "on_missing should default to continue"
        );
        assert_eq!(
            validated.forward_headers,
            Vec::<String>::new(),
            "forward_headers should default to empty"
        );
    }

    #[test]
    fn full_config_parses() {
        let yaml = r#"
files_api_url: "http://files:9090"
allow_private_files_api_url: true
allow_pre_security_callout: true
forward_headers:
  - authorization
  - x-custom-tenant
max_body_bytes: 1048576
max_file_references: 16
on_missing: reject
timeout_ms: 10000
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_config(cfg).unwrap();
        assert_eq!(
            validated.files_api_url, "http://files:9090",
            "files_api_url should match"
        );
        assert_eq!(
            validated.forward_headers,
            vec!["authorization", "x-custom-tenant"],
            "forward_headers should match"
        );
        assert_eq!(validated.max_body_bytes, 1_048_576, "max_body_bytes should match");
        assert_eq!(validated.max_file_references, 16, "max_file_references should match");
        assert_eq!(validated.on_missing, OnMissing::Reject, "on_missing should match");
        assert_eq!(validated.timeout_ms, 10_000, "timeout_ms should match");
    }

    #[test]
    fn deny_unknown_fields_rejects_typo() {
        let yaml = r#"files_api_url: "http://ogx:8321"
on_mising: reject"#;
        let result: Result<FileResolveConfig, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "typo in config field should be rejected");
    }

    #[test]
    fn empty_files_api_url_rejected() {
        let yaml = "files_api_url: ''";
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "empty files_api_url should be rejected");
    }

    #[test]
    fn trailing_slash_files_api_url_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321/""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "trailing slash should be rejected");
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321"
max_body_bytes: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "zero max_body_bytes should be rejected");
    }

    #[test]
    fn zero_timeout_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321"
timeout_ms: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "zero timeout should be rejected");
    }

    #[test]
    fn zero_max_file_references_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321"
max_file_references: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "zero max_file_references should be rejected"
        );
    }

    #[test]
    fn max_file_references_above_ceiling_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321"
max_file_references: 129"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "max_file_references above the ceiling should be rejected"
        );
    }

    #[test]
    fn timeout_above_ceiling_rejected() {
        let yaml = r#"files_api_url: "http://ogx:8321"
timeout_ms: 300001"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "timeout above ceiling should be rejected");
    }

    #[test]
    fn valid_config_passes() {
        let cfg = FileResolveConfig {
            allow_private_files_api_url: true,
            allow_pre_security_callout: true,
            files_api_url: "http://ogx:8321".to_owned(),
            forward_headers: Vec::new(),
            max_body_bytes: MAX_JSON_BODY_BYTES,
            max_file_references: DEFAULT_MAX_FILE_REFERENCES,
            on_missing: OnMissing::Continue,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        };
        assert!(validate_config(cfg).is_ok(), "valid config should pass validation");
    }

    #[test]
    fn forward_headers_are_normalized() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
forward_headers:
  - Authorization
  - X-Tenant-ID
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_config(cfg).unwrap();

        assert_eq!(
            validated.forward_headers,
            vec!["authorization", "x-tenant-id"],
            "forwarded header names should be normalized once during validation"
        );
    }

    #[test]
    fn invalid_forward_header_rejected() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
forward_headers: ["bad header"]
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
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
            let yaml = format!(
                "files_api_url: \"http://ogx:8321\"\nallow_private_files_api_url: true\nforward_headers: [\"{name}\"]"
            );
            let cfg: FileResolveConfig = serde_yaml::from_str(&yaml).unwrap();

            assert!(
                validate_config(cfg).is_err(),
                "unsafe forwarded header '{name}' should be rejected"
            );
        }
    }

    #[test]
    fn duplicate_forward_headers_rejected_case_insensitively() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
forward_headers: ["Authorization", "authorization"]
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "duplicate forwarded header names should be rejected after normalization"
        );
    }

    #[test]
    fn pre_security_callout_requires_explicit_opt_in() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "pre-security external callouts should be disabled by default"
        );
    }

    // -------------------------------------------------------------------------
    // SSRF validation
    // -------------------------------------------------------------------------

    #[test]
    fn ssrf_rejects_loopback_ipv4() {
        let yaml = r#"files_api_url: "http://127.0.0.1:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(
            result.is_err(),
            "loopback IPv4 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_loopback_ipv6() {
        let yaml = r#"files_api_url: "http://[::1]:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(
            result.is_err(),
            "loopback IPv6 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_private_ipv4() {
        let yaml = r#"files_api_url: "http://10.0.0.1:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "private IPv4 should be rejected without allow_private");
    }

    #[test]
    fn ssrf_rejects_link_local_ipv4() {
        let yaml = r#"files_api_url: "http://169.254.169.254""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(
            result.is_err(),
            "link-local IPv4 (metadata endpoint) should be rejected"
        );
    }

    #[test]
    fn ssrf_rejects_cgnat_ipv4() {
        let yaml = r#"files_api_url: "http://100.64.0.1:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "CGNAT IPv4 should be rejected without allow_private");
    }

    #[test]
    fn ssrf_rejects_localhost_name() {
        let yaml = r#"files_api_url: "http://localhost:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(
            result.is_err(),
            "localhost name should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_rejects_dns_name() {
        let yaml = r#"files_api_url: "http://ogx:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "DNS name should be rejected without allow_private");
    }

    #[test]
    fn ssrf_rejects_legacy_octal_loopback() {
        let yaml = r#"files_api_url: "http://0177.0.0.1:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "octal-encoded loopback should be rejected");
    }

    #[test]
    fn ssrf_rejects_ipv4_mapped_ipv6_loopback() {
        let yaml = r#"files_api_url: "http://[::ffff:127.0.0.1]:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "IPv4-mapped IPv6 loopback should be rejected");
    }

    #[test]
    fn ssrf_allows_public_ipv4() {
        let yaml = r#"files_api_url: "http://203.0.113.1:8321"
allow_pre_security_callout: true"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_ok(), "public IPv4 should be allowed");
    }

    #[test]
    fn ssrf_allows_public_ipv6() {
        let yaml = r#"files_api_url: "https://[2606:4700:4700::1111]:8321"
allow_pre_security_callout: true"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_ok(),
            "bracketed public IPv6 should be recognized as an IP literal"
        );
    }

    #[test]
    fn ssrf_allows_private_with_override() {
        let yaml = r#"
files_api_url: "http://127.0.0.1:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_ok(),
            "loopback should be allowed with allow_private_files_api_url"
        );
    }

    #[test]
    fn ssrf_allows_dns_with_override() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_ok(),
            "DNS name should be allowed with allow_private_files_api_url"
        );
    }

    #[test]
    fn ssrf_rejects_non_http_scheme() {
        let yaml = r#"files_api_url: "ftp://ogx:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "non-http scheme should be rejected");
    }

    #[test]
    fn files_api_url_rejects_embedded_credentials() {
        let yaml = r#"
files_api_url: "http://user:password@ogx:8321"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "embedded URL credentials should be rejected"
        );
    }

    #[test]
    fn files_api_url_rejects_query_string() {
        let yaml = r#"
files_api_url: "http://ogx:8321/base?tenant=abc"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "query strings would make appended Files API paths ambiguous"
        );
    }

    #[test]
    fn files_api_url_rejects_fragment() {
        let yaml = r#"
files_api_url: "http://ogx:8321/base#v2"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "fragments would hide appended Files API paths from the HTTP request"
        );
    }
}
