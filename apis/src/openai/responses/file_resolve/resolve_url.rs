// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! URL resolution and SSRF validation for `file_url` references.

use std::net::{IpAddr, SocketAddr};

use praxis_core::connectivity::normalize_mapped_ipv4;

use super::resolve::{
    ResolveError, ResolvedFile, infer_mime_from_filename, max_content_bytes_for_data_url, read_bounded_body,
};

/// A validated, normalized origin (scheme + host + port) for allowlist matching.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NormalizedOrigin {
    /// URL scheme (`http` or `https`).
    pub scheme: String,
    /// Canonical host (lowercase; brackets preserved for IPv6).
    pub host: String,
    /// Effective port (443 for https, 80 for http if not explicit).
    pub port: u16,
}

impl NormalizedOrigin {
    /// Parse and validate an origin string.
    ///
    /// Accepts `scheme://host[:port]`. Rejects paths (other than `/`),
    /// queries, fragments, and credentials. Normalizes default ports
    /// and lowercases the host.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let url = url::Url::parse(raw).map_err(|e| format!("invalid origin URL: {e}"))?;
        let scheme = url.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(format!("origin must use http or https scheme, got '{scheme}'"));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err("origin must not contain credentials".to_owned());
        }
        if url.path() != "/" {
            return Err(format!("origin must not contain a path, got '{}'", url.path()));
        }
        if url.query().is_some() {
            return Err("origin must not contain a query string".to_owned());
        }
        if url.fragment().is_some() {
            return Err("origin must not contain a fragment".to_owned());
        }
        let host = url
            .host_str()
            .ok_or_else(|| "origin must include a host".to_owned())?
            .to_ascii_lowercase();
        let default_port = if scheme == "https" { 443 } else { 80 };
        let port = url.port().unwrap_or(default_port);

        let origin = Self {
            scheme: scheme.to_owned(),
            host,
            port,
        };
        origin.reject_unconditionally_blocked_ips()?;
        Ok(origin)
    }

    /// Check whether a request URL's origin matches this allowlist entry.
    pub fn matches_url(&self, url: &url::Url) -> bool {
        let url_scheme = url.scheme();
        let Some(url_host) = url.host_str() else {
            return false;
        };
        let default_port = if url_scheme == "https" { 443 } else { 80 };
        let url_port = url.port().unwrap_or(default_port);
        self.scheme == url_scheme && self.host == url_host.to_ascii_lowercase() && self.port == url_port
    }

    /// Reject IPs that are never valid in an allowlist: unspecified,
    /// multicast, and cloud metadata.
    fn reject_unconditionally_blocked_ips(&self) -> Result<(), String> {
        // Strip brackets for IPv6 addresses
        let host_without_brackets = self
            .host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(&self.host);

        let ip: IpAddr = match host_without_brackets.parse() {
            Ok(ip) => normalize_mapped_ipv4(ip),
            Err(_) => return Ok(()),
        };
        if ip.is_unspecified() {
            return Err("origin must not target an unspecified address".to_owned());
        }
        if ip.is_multicast() {
            return Err("origin must not target a multicast address".to_owned());
        }
        if is_cloud_metadata(&ip) {
            return Err("origin must not target a cloud metadata endpoint".to_owned());
        }
        Ok(())
    }
}

/// Cloud metadata and credential endpoints that are never permitted.
fn is_cloud_metadata(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            *v4 == std::net::Ipv4Addr::new(169, 254, 169, 254)
                || *v4 == std::net::Ipv4Addr::new(169, 254, 170, 2)
                || *v4 == std::net::Ipv4Addr::new(169, 254, 170, 23)
                || *v4 == std::net::Ipv4Addr::new(100, 100, 100, 200)
        },
        IpAddr::V6(v6) => {
            const AWS_IMDS_V6: std::net::Ipv6Addr = std::net::Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0254);
            const AWS_ECS_CREDS_V6: std::net::Ipv6Addr = std::net::Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0023);
            *v6 == AWS_IMDS_V6 || *v6 == AWS_ECS_CREDS_V6
        },
    }
}

/// IPs that are always blocked regardless of allowlist.
fn is_unconditionally_blocked(ip: &IpAddr) -> bool {
    ip.is_unspecified() || ip.is_multicast() || is_cloud_metadata(ip)
}

/// IPs that are blocked unless the origin is allowlisted.
fn is_private_range(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.octets()[0] == 0
                || is_cgnat(*v4)
                || is_special_use_v4(*v4)
        },
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unique_local()
                || is_unicast_link_local_v6(v6)
                || is_site_local_v6(v6)
                || is_special_use_v6(*v6)
        },
    }
}

/// Check if an IPv4 address is in the CGNAT range (100.64.0.0/10).
fn is_cgnat(ip: std::net::Ipv4Addr) -> bool {
    u32::from(ip) & 0xFFC0_0000 == 0x6440_0000
}

/// Check if an IPv6 address is in the unicast link-local range (`fe80::/10`).
fn is_unicast_link_local_v6(v6: &std::net::Ipv6Addr) -> bool {
    let [a, b, ..] = v6.octets();
    a == 0xFE && (b & 0xC0) == 0x80
}

/// Deprecated site-local range (`fec0::/10`, RFC 3879).
fn is_site_local_v6(v6: &std::net::Ipv6Addr) -> bool {
    let [a, b, ..] = v6.octets();
    a == 0xFE && (b & 0xC0) == 0xC0
}

/// Non-globally-reachable IPv4 special-use ranges per IANA registry.
fn is_special_use_v4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    let n = u32::from(ip);
    // 192.0.0.0/24 (IETF Protocol Assignments) + 192.0.2.0/24 (TEST-NET-1)
    (o[0] == 192 && o[1] == 0 && (o[2] == 0 || o[2] == 2))
    // 198.18.0.0/15 — Benchmarking
    || (n & 0xFFFE_0000 == 0xC612_0000)
    // 198.51.100.0/24 — Documentation (TEST-NET-2)
    || (o[0] == 198 && o[1] == 51 && o[2] == 100)
    // 203.0.113.0/24 — Documentation (TEST-NET-3)
    || (o[0] == 203 && o[1] == 0 && o[2] == 113)
    // 240.0.0.0/4 — Reserved for future use
    || o[0] >= 240
}

/// Non-globally-reachable IPv6 special-use ranges per IANA registry.
fn is_special_use_v6(v6: std::net::Ipv6Addr) -> bool {
    let s = v6.segments();
    // 100::/64 — Discard-Only (RFC 6666)
    (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0)
    // 2001:db8::/32 — Documentation (RFC 3849)
    || (s[0] == 0x2001 && s[1] == 0x0DB8)
}

/// Validate that a string is a well-formed MIME type (`token/token`).
///
/// Uses the RFC 2045 token production but additionally excludes characters
/// that are URI-structural (`#`, `%`) or non-printable in data URIs
/// (`^`, `` ` ``, `|`), preventing corruption when the type is interpolated
/// into `data:{type};base64,...`.
fn is_valid_mime_type(s: &str) -> bool {
    /// RFC 2045 token characters minus URI-unsafe ones (`#%^``|`).
    fn is_data_uri_safe_token(c: char) -> bool {
        c.is_ascii() && !c.is_ascii_control() && !b" \t\"(),/:;<=>?@[\\]{}#%^`|".contains(&(c as u8))
    }

    let Some((type_part, subtype)) = s.split_once('/') else {
        return false;
    };
    !type_part.is_empty()
        && !subtype.is_empty()
        && type_part.chars().all(is_data_uri_safe_token)
        && subtype.chars().all(is_data_uri_safe_token)
}

/// Check if an IP should be blocked for `file_url` fetches.
pub(crate) fn is_file_url_ssrf_blocked(ip: &IpAddr, allow_private: bool) -> bool {
    let ip = &normalize_mapped_ipv4(*ip);
    if is_unconditionally_blocked(ip) {
        return true;
    }
    if !allow_private && is_private_range(ip) {
        return true;
    }
    false
}

/// Redact sensitive parts of a URL for safe logging.
///
/// Strips credentials entirely and replaces query parameter values with `[REDACTED]`.
pub(crate) fn redact_url(raw: &str) -> String {
    let Ok(mut url) = url::Url::parse(raw) else {
        return "<invalid URL>".to_owned();
    };
    // Clear credentials (error only occurs if cannot-be-a-base, which was already validated)
    #[expect(clippy::let_underscore_must_use, reason = "Result<(), ()> intentionally ignored")]
    {
        let _ = url.set_username("");
        let _ = url.set_password(None);
    }

    // Redact query values
    if url.query().is_some() {
        let redacted_pairs: Vec<_> = url
            .query_pairs()
            .map(|(key, _value)| format!("{key}=[REDACTED]"))
            .collect();
        if !redacted_pairs.is_empty() {
            url.set_query(Some(&redacted_pairs.join("&")));
        }
    }

    url.to_string()
}

/// Validate a `file_url`: must be http/https, no credentials, no fragment.
pub(crate) fn validate_file_url(raw: &str) -> Result<url::Url, ResolveError> {
    let url = url::Url::parse(raw).map_err(|e| {
        tracing::debug!(error = %e, "invalid file_url");
        ResolveError::FileUrlBlocked { label: redact_url(raw) }
    })?;

    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(ResolveError::FileUrlBlocked { label: redact_url(raw) });
    }

    if !url.username().is_empty() || url.password().is_some() {
        return Err(ResolveError::FileUrlBlocked { label: redact_url(raw) });
    }

    if url.fragment().is_some() {
        return Err(ResolveError::FileUrlBlocked { label: redact_url(raw) });
    }

    Ok(url)
}

/// Sanitize a filename by stripping path separators and control characters.
///
/// Returns `None` if the result is empty or exceeds 255 bytes.
pub(crate) fn sanitize_filename(raw: &str) -> Option<String> {
    let sanitized: String = raw
        .chars()
        .filter(|&c| c != '/' && c != '\\' && c >= '\x20' && c != '\x7F')
        .collect();

    if sanitized.is_empty() {
        return None;
    }

    // Bound at 255 bytes on UTF-8 boundary
    if sanitized.len() <= 255 {
        Some(sanitized)
    } else {
        // Find the last valid UTF-8 boundary within 255 bytes
        let mut len = 255;
        while len > 0 && !sanitized.is_char_boundary(len) {
            len -= 1;
        }
        if len == 0 {
            None
        } else {
            sanitized.get(..len).map(ToOwned::to_owned)
        }
    }
}

/// Resolves `file_url` references with DNS pinning and SSRF protection.
pub(crate) struct FileUrlResolver {
    /// Origins that permit resolution to private/loopback addresses.
    pub(crate) allowed_private_origins: Vec<NormalizedOrigin>,
}

impl FileUrlResolver {
    /// Resolve a `file_url` with SSRF protection and DNS pinning.
    #[expect(clippy::too_many_lines, reason = "main resolution flow")]
    pub(crate) async fn resolve_url(
        &self,
        url: &str,
        deadline: tokio::time::Instant,
        max_resolved_bytes: usize,
    ) -> Result<ResolvedFile, ResolveError> {
        let parsed_url = validate_file_url(url)?;
        let label = redact_url(url);

        // Check if origin is allowlisted for private addresses
        let allow_private = self
            .allowed_private_origins
            .iter()
            .any(|origin| origin.matches_url(&parsed_url));

        // Resolve and validate addresses
        let (host, addrs) = resolve_and_pin_url(&parsed_url, allow_private, &label, deadline).await?;

        // Check deadline before making the request
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(ResolveError::FileUrlFailed {
                label,
                detail: "deadline exceeded before fetch".to_owned(),
            });
        }
        let remaining = deadline - now;

        // Build pinned client
        let client = build_pinned_client(&host, &addrs, remaining).map_err(|e| ResolveError::FileUrlFailed {
            label: label.clone(),
            detail: format!("failed to build HTTP client: {e}"),
        })?;

        // Send GET with only x-praxis-callout-depth header
        let mut request = client.get(parsed_url.as_str());
        request = request.header("x-praxis-callout-depth", "1");

        let response = request.send().await.map_err(|e| {
            let detail = if e.is_timeout() {
                "fetch timed out".to_owned()
            } else {
                "fetch failed".to_owned()
            };
            ResolveError::FileUrlFailed {
                label: label.clone(),
                detail,
            }
        })?;

        // Check success status
        if !response.status().is_success() {
            return Err(ResolveError::FileUrlFailed {
                label,
                detail: format!("server returned {}", response.status()),
            });
        }

        // Validate Content-Type
        let content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .and_then(|ct| {
                let mime = ct.split(';').next().map(str::trim)?;
                is_valid_mime_type(mime).then_some(mime)
            })
            .or_else(|| {
                // Fall back to URL path extension
                infer_mime_from_filename(parsed_url.path().rsplit('/').next())
            })
            .unwrap_or("application/octet-stream")
            .to_owned();

        // Compute max raw bytes after reserving data URI prefix
        let max_content_bytes = max_content_bytes_for_data_url(max_resolved_bytes, &content_type).ok_or_else(|| {
            ResolveError::FileUrlFailed {
                label: label.clone(),
                detail: "content type prefix exceeds budget".to_owned(),
            }
        })?;

        // Check Content-Length against max raw bytes
        if let Some(cl) = response.content_length()
            && usize::try_from(cl).unwrap_or(usize::MAX) > max_content_bytes
        {
            return Err(ResolveError::FileUrlFailed {
                label,
                detail: format!("Content-Length {cl} exceeds limit"),
            });
        }

        // Read bounded body
        let content = read_bounded_body(response, &label, max_content_bytes, max_resolved_bytes).await?;

        // Derive filename
        let filename = parsed_url
            .path_segments()
            .and_then(|mut segments| segments.next_back())
            .and_then(sanitize_filename);

        // Encode as base64
        let base64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &content);

        Ok(ResolvedFile {
            base64,
            content_type,
            filename,
        })
    }
}

/// Resolve DNS and validate all addresses.
#[expect(clippy::too_many_lines, reason = "DNS resolution + validation flow")]
async fn resolve_and_pin_url(
    url: &url::Url,
    allow_private: bool,
    label: &str,
    deadline: tokio::time::Instant,
) -> Result<(String, Vec<SocketAddr>), ResolveError> {
    let Some(host) = url.host_str() else {
        return Err(ResolveError::FileUrlBlocked {
            label: label.to_owned(),
        });
    };

    // Check if host is an IP literal
    if let Ok(ip) = host.parse::<IpAddr>() {
        let normalized = normalize_mapped_ipv4(ip);
        if is_file_url_ssrf_blocked(&normalized, allow_private) {
            return Err(ResolveError::FileUrlBlocked {
                label: label.to_owned(),
            });
        }
        // IP literals don't need DNS resolution, return empty addrs for no pinning
        return Ok((host.to_owned(), Vec::new()));
    }

    // Check blocked hostnames
    if !allow_private && is_blocked_hostname(host) {
        return Err(ResolveError::FileUrlBlocked {
            label: label.to_owned(),
        });
    }

    // DNS resolution
    let port = url.port_or_known_default().unwrap_or(80);
    let lookup = format!("{host}:{port}");

    let addrs: Vec<SocketAddr> = tokio::time::timeout_at(deadline, tokio::net::lookup_host(&lookup))
        .await
        .map_err(|e| {
            tracing::debug!(error = %e, "DNS resolution deadline exceeded");
            ResolveError::FileUrlFailed {
                label: label.to_owned(),
                detail: "DNS resolution timed out".to_owned(),
            }
        })?
        .map_err(|e| ResolveError::FileUrlFailed {
            label: label.to_owned(),
            detail: format!("DNS resolution failed: {e}"),
        })?
        .collect();

    if addrs.is_empty() {
        return Err(ResolveError::FileUrlFailed {
            label: label.to_owned(),
            detail: "DNS returned zero addresses".to_owned(),
        });
    }

    // Deduplicate and validate all IPs
    let mut seen = std::collections::HashSet::new();
    let mut validated = Vec::new();
    for addr in addrs {
        let ip = normalize_mapped_ipv4(addr.ip());
        if seen.insert(ip) {
            if is_file_url_ssrf_blocked(&ip, allow_private) {
                return Err(ResolveError::FileUrlBlocked {
                    label: label.to_owned(),
                });
            }
            validated.push(SocketAddr::new(ip, addr.port()));
        }
    }

    Ok((host.to_owned(), validated))
}

/// Check if a hostname is blocked (localhost and *.localhost).
fn is_blocked_hostname(host: &str) -> bool {
    let lower = host.to_ascii_lowercase();
    lower == "localhost" || lower.ends_with(".localhost")
}

/// Build a per-request pinned reqwest client.
fn build_pinned_client(
    host: &str,
    addrs: &[SocketAddr],
    timeout: std::time::Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout);

    if !addrs.is_empty() {
        builder = builder.resolve_to_addrs(host, addrs);
    }

    builder.build()
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

    #[test]
    fn parse_valid_https_origin() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        assert_eq!(origin.scheme, "https");
        assert_eq!(origin.host, "files.example.com");
        assert_eq!(origin.port, 443, "default https port should be 443");
    }

    #[test]
    fn parse_valid_http_origin_with_port() {
        let origin = NormalizedOrigin::parse("http://files.internal:8443").unwrap();
        assert_eq!(origin.scheme, "http");
        assert_eq!(origin.host, "files.internal");
        assert_eq!(origin.port, 8443);
    }

    #[test]
    fn parse_normalizes_host_case() {
        let origin = NormalizedOrigin::parse("https://Files.EXAMPLE.Com").unwrap();
        assert_eq!(origin.host, "files.example.com", "host should be lowercased");
    }

    #[test]
    fn parse_normalizes_default_port() {
        let with_port = NormalizedOrigin::parse("https://example.com:443").unwrap();
        let without_port = NormalizedOrigin::parse("https://example.com").unwrap();
        assert_eq!(with_port, without_port, "explicit default port should normalize");
    }

    #[test]
    fn parse_rejects_non_http_scheme() {
        assert!(NormalizedOrigin::parse("ftp://example.com").is_err());
        assert!(NormalizedOrigin::parse("file:///tmp/x").is_err());
    }

    #[test]
    fn parse_rejects_credentials() {
        assert!(NormalizedOrigin::parse("https://user@example.com").is_err());
        assert!(NormalizedOrigin::parse("https://user:pass@example.com").is_err());
    }

    #[test]
    fn parse_rejects_path() {
        assert!(NormalizedOrigin::parse("https://example.com/files").is_err());
    }

    #[test]
    fn parse_accepts_root_path() {
        assert!(NormalizedOrigin::parse("https://example.com/").is_ok());
    }

    #[test]
    fn parse_rejects_query() {
        assert!(NormalizedOrigin::parse("https://example.com?q=1").is_err());
    }

    #[test]
    fn parse_rejects_fragment() {
        assert!(NormalizedOrigin::parse("https://example.com#frag").is_err());
    }

    #[test]
    fn parse_rejects_cloud_metadata_ipv4() {
        assert!(NormalizedOrigin::parse("http://169.254.169.254").is_err());
        assert!(NormalizedOrigin::parse("http://100.100.100.200").is_err());
    }

    #[test]
    fn parse_rejects_cloud_metadata_ipv6() {
        assert!(NormalizedOrigin::parse("http://[fd00:ec2::254]").is_err());
    }

    #[test]
    fn parse_rejects_unspecified() {
        assert!(NormalizedOrigin::parse("http://0.0.0.0").is_err());
        assert!(NormalizedOrigin::parse("http://[::]").is_err());
    }

    #[test]
    fn parse_rejects_multicast() {
        assert!(NormalizedOrigin::parse("http://224.0.0.1").is_err());
    }

    #[test]
    fn parse_allows_private_ip_in_allowlist() {
        assert!(
            NormalizedOrigin::parse("http://10.0.0.1:8080").is_ok(),
            "private IPs should be allowed in the allowlist"
        );
    }

    #[test]
    fn parse_allows_loopback_in_allowlist() {
        assert!(
            NormalizedOrigin::parse("http://127.0.0.1:8080").is_ok(),
            "loopback should be allowed in the allowlist"
        );
    }

    #[test]
    fn matches_url_exact() {
        let origin = NormalizedOrigin::parse("https://files.example.com:8443").unwrap();
        let url = url::Url::parse("https://files.example.com:8443/path/to/file.pdf?token=abc").unwrap();
        assert!(origin.matches_url(&url), "exact origin should match");
    }

    #[test]
    fn matches_url_default_port() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        let url = url::Url::parse("https://files.example.com/file.pdf").unwrap();
        assert!(origin.matches_url(&url), "default-port origin should match");
    }

    #[test]
    fn matches_url_case_insensitive() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        let url = url::Url::parse("https://Files.Example.Com/file.pdf").unwrap();
        assert!(origin.matches_url(&url), "host matching should be case-insensitive");
    }

    #[test]
    fn no_match_wrong_scheme() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        let url = url::Url::parse("http://files.example.com/file.pdf").unwrap();
        assert!(!origin.matches_url(&url), "scheme mismatch should not match");
    }

    #[test]
    fn no_match_wrong_port() {
        let origin = NormalizedOrigin::parse("https://files.example.com:8443").unwrap();
        let url = url::Url::parse("https://files.example.com:9443/file.pdf").unwrap();
        assert!(!origin.matches_url(&url), "port mismatch should not match");
    }

    #[test]
    fn no_match_wrong_host() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        let url = url::Url::parse("https://other.example.com/file.pdf").unwrap();
        assert!(!origin.matches_url(&url), "host mismatch should not match");
    }

    // SSRF IP validation tests
    #[test]
    fn ssrf_blocks_loopback_ipv4() {
        assert!(
            is_file_url_ssrf_blocked(&"127.0.0.1".parse().unwrap(), false),
            "loopback IPv4 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_loopback_ipv6() {
        assert!(
            is_file_url_ssrf_blocked(&"::1".parse().unwrap(), false),
            "loopback IPv6 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_private_10() {
        assert!(
            is_file_url_ssrf_blocked(&"10.0.0.1".parse().unwrap(), false),
            "10.x should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_private_172() {
        assert!(
            is_file_url_ssrf_blocked(&"172.16.0.1".parse().unwrap(), false),
            "172.16-31.x should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_private_192() {
        assert!(
            is_file_url_ssrf_blocked(&"192.168.1.1".parse().unwrap(), false),
            "192.168.x should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_link_local_ipv4() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.1.1".parse().unwrap(), false),
            "link-local IPv4 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_link_local_ipv6() {
        assert!(
            is_file_url_ssrf_blocked(&"fe80::1".parse().unwrap(), false),
            "link-local IPv6 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_cgnat() {
        assert!(
            is_file_url_ssrf_blocked(&"100.64.0.1".parse().unwrap(), false),
            "CGNAT 100.64.0.0/10 should be blocked without allowlist"
        );
        assert!(
            is_file_url_ssrf_blocked(&"100.127.255.254".parse().unwrap(), false),
            "CGNAT upper range should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_unspecified_ipv4() {
        assert!(
            is_file_url_ssrf_blocked(&"0.0.0.0".parse().unwrap(), true),
            "unspecified IPv4 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_unspecified_ipv6() {
        assert!(
            is_file_url_ssrf_blocked(&"::".parse().unwrap(), true),
            "unspecified IPv6 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_multicast_ipv4() {
        assert!(
            is_file_url_ssrf_blocked(&"224.0.0.1".parse().unwrap(), true),
            "multicast IPv4 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_multicast_ipv6() {
        assert!(
            is_file_url_ssrf_blocked(&"ff02::1".parse().unwrap(), true),
            "multicast IPv6 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_aws_imds_v4() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.169.254".parse().unwrap(), true),
            "AWS IMDS v4 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_aws_imds_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"fd00:ec2::254".parse().unwrap(), true),
            "AWS IMDS v6 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_alibaba_metadata() {
        assert!(
            is_file_url_ssrf_blocked(&"100.100.100.200".parse().unwrap(), true),
            "Alibaba metadata should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_ipv4_mapped_loopback() {
        assert!(
            is_file_url_ssrf_blocked(&"::ffff:127.0.0.1".parse().unwrap(), false),
            "IPv4-mapped loopback should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_unique_local_ipv6() {
        assert!(
            is_file_url_ssrf_blocked(&"fc00::1".parse().unwrap(), false),
            "unique local IPv6 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_allows_public_ipv4() {
        assert!(
            !is_file_url_ssrf_blocked(&"8.8.8.8".parse().unwrap(), false),
            "public IPv4 should be allowed"
        );
    }

    #[test]
    fn ssrf_allows_public_ipv6() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:4860:4860::8888".parse().unwrap(), false),
            "public IPv6 should be allowed"
        );
    }

    #[test]
    fn ssrf_allows_private_with_allowlist() {
        assert!(
            !is_file_url_ssrf_blocked(&"10.0.0.1".parse().unwrap(), true),
            "private IP should be allowed with allowlist match"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"127.0.0.1".parse().unwrap(), true),
            "loopback should be allowed with allowlist match"
        );
    }

    #[test]
    fn ssrf_blocks_cloud_metadata_even_with_allowlist() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.169.254".parse().unwrap(), true),
            "cloud metadata should be blocked even with allowlist"
        );
        assert!(
            is_file_url_ssrf_blocked(&"fd00:ec2::254".parse().unwrap(), true),
            "cloud metadata IPv6 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_ecs_credential_endpoint() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.170.2".parse().unwrap(), true),
            "ECS credential endpoint should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_eks_credential_endpoint() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.170.23".parse().unwrap(), true),
            "EKS credential endpoint should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_ecs_credential_endpoint_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"fd00:ec2::23".parse().unwrap(), true),
            "ECS/EKS credential IPv6 endpoint should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_site_local_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"fec0::1".parse().unwrap(), false),
            "site-local fec0::/10 should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_site_local_v6_upper() {
        assert!(
            is_file_url_ssrf_blocked(&"feff::1".parse().unwrap(), false),
            "site-local upper range feff:: should be blocked without allowlist"
        );
    }

    #[test]
    fn ssrf_allows_site_local_v6_with_allowlist() {
        assert!(
            !is_file_url_ssrf_blocked(&"fec0::1".parse().unwrap(), true),
            "site-local should be allowed with allowlist"
        );
    }

    // Special-use range tests
    #[test]
    fn ssrf_blocks_benchmarking_range() {
        assert!(
            is_file_url_ssrf_blocked(&"198.18.0.1".parse().unwrap(), false),
            "198.18.0.0/15 benchmarking should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"198.19.255.254".parse().unwrap(), false),
            "198.19.x upper range should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_ietf_protocol_assignments() {
        assert!(
            is_file_url_ssrf_blocked(&"192.0.0.1".parse().unwrap(), false),
            "192.0.0.0/24 IETF assignments should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_documentation_ranges() {
        assert!(
            is_file_url_ssrf_blocked(&"192.0.2.1".parse().unwrap(), false),
            "TEST-NET-1 should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"198.51.100.1".parse().unwrap(), false),
            "TEST-NET-2 should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"203.0.113.1".parse().unwrap(), false),
            "TEST-NET-3 should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_reserved_v4() {
        assert!(
            is_file_url_ssrf_blocked(&"240.0.0.1".parse().unwrap(), false),
            "240.0.0.0/4 reserved should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"255.255.255.254".parse().unwrap(), false),
            "255.x reserved upper should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_discard_only_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"100::1".parse().unwrap(), false),
            "100::/64 discard-only should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_documentation_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:db8::1".parse().unwrap(), false),
            "2001:db8::/32 documentation should be blocked"
        );
    }

    #[test]
    fn ssrf_allows_special_use_with_allowlist() {
        assert!(
            !is_file_url_ssrf_blocked(&"198.18.0.1".parse().unwrap(), true),
            "benchmarking range should be allowed with allowlist"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:db8::1".parse().unwrap(), true),
            "documentation v6 should be allowed with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_zero_octet() {
        assert!(
            is_file_url_ssrf_blocked(&"0.1.2.3".parse().unwrap(), false),
            "IPv4 with leading zero octet should be blocked without allowlist"
        );
    }

    // MIME validation tests
    #[test]
    fn mime_valid_simple() {
        assert!(is_valid_mime_type("text/plain"));
        assert!(is_valid_mime_type("application/octet-stream"));
        assert!(is_valid_mime_type("image/png"));
    }

    #[test]
    fn mime_rejects_comma_separated() {
        assert!(
            !is_valid_mime_type("text/plain,application/json"),
            "comma in MIME should be rejected"
        );
    }

    #[test]
    fn mime_rejects_no_slash() {
        assert!(
            !is_valid_mime_type("textplain"),
            "MIME without slash should be rejected"
        );
    }

    #[test]
    fn mime_rejects_empty_parts() {
        assert!(!is_valid_mime_type("/plain"), "empty type should be rejected");
        assert!(!is_valid_mime_type("text/"), "empty subtype should be rejected");
    }

    #[test]
    fn mime_rejects_spaces() {
        assert!(!is_valid_mime_type("text /plain"), "space in type should be rejected");
    }

    #[test]
    fn mime_rejects_semicolon_in_type() {
        assert!(
            !is_valid_mime_type("text/plain;charset=utf-8"),
            "semicolon should have been stripped before validation"
        );
    }

    #[test]
    fn mime_rejects_data_uri_unsafe_chars() {
        assert!(!is_valid_mime_type("text/x-foo#bar"), "# corrupts data URI fragment");
        assert!(!is_valid_mime_type("text/x%2Fplain"), "% triggers percent-encoding");
        assert!(!is_valid_mime_type("text/x^plain"), "^ is not URI-safe");
        assert!(!is_valid_mime_type("text/x`plain"), "backtick is not URI-safe");
        assert!(!is_valid_mime_type("text/x|plain"), "pipe is not URI-safe");
    }

    // RedactedUrl tests
    #[test]
    fn redact_url_strips_credentials() {
        let redacted = redact_url("https://user:pass@example.com/path");
        assert!(!redacted.contains("user"), "username should be stripped");
        assert!(!redacted.contains("pass"), "password should be stripped");
        assert!(redacted.contains("example.com"), "host should remain");
    }

    #[test]
    fn redact_url_replaces_query_values() {
        let redacted = redact_url("https://example.com/file?token=secret&id=123");
        assert!(redacted.contains("token=[REDACTED]"), "query key should remain");
        assert!(!redacted.contains("secret"), "query value should be redacted");
        assert!(
            redacted.contains("id=[REDACTED]"),
            "all query values should be redacted"
        );
    }

    #[test]
    fn redact_url_preserves_path() {
        let redacted = redact_url("https://example.com/sensitive/path");
        assert!(redacted.contains("/sensitive/path"), "path should be preserved");
    }

    #[test]
    fn redact_url_handles_invalid_url() {
        let redacted = redact_url("not a url");
        assert_eq!(redacted, "<invalid URL>", "invalid URL should return placeholder");
    }

    #[test]
    fn redact_url_no_query_unchanged() {
        let redacted = redact_url("https://example.com/file.pdf");
        assert_eq!(
            redacted, "https://example.com/file.pdf",
            "URL without query should remain unchanged except for redaction"
        );
    }

    // validate_file_url tests
    #[test]
    fn validate_file_url_accepts_http() {
        assert!(
            validate_file_url("http://example.com/file.pdf").is_ok(),
            "http scheme should be accepted"
        );
    }

    #[test]
    fn validate_file_url_accepts_https() {
        assert!(
            validate_file_url("https://example.com/file.pdf").is_ok(),
            "https scheme should be accepted"
        );
    }

    #[test]
    fn validate_file_url_rejects_ftp() {
        assert!(
            validate_file_url("ftp://example.com/file.pdf").is_err(),
            "ftp scheme should be rejected"
        );
    }

    #[test]
    fn validate_file_url_rejects_file_scheme() {
        assert!(
            validate_file_url("file:///etc/passwd").is_err(),
            "file scheme should be rejected"
        );
    }

    #[test]
    fn validate_file_url_rejects_data_uri() {
        assert!(
            validate_file_url("data:text/plain;base64,SGVsbG8=").is_err(),
            "data URI should be rejected"
        );
    }

    #[test]
    fn validate_file_url_rejects_credentials() {
        assert!(
            validate_file_url("https://user:pass@example.com/file.pdf").is_err(),
            "credentials should be rejected"
        );
    }

    #[test]
    fn validate_file_url_rejects_username_only() {
        assert!(
            validate_file_url("https://user@example.com/file.pdf").is_err(),
            "username without password should be rejected"
        );
    }

    #[test]
    fn validate_file_url_rejects_fragment() {
        assert!(
            validate_file_url("https://example.com/file.pdf#page=2").is_err(),
            "fragment should be rejected"
        );
    }

    #[test]
    fn validate_file_url_preserves_query() {
        let url = validate_file_url("https://example.com/file.pdf?token=abc&expires=123").unwrap();
        assert!(
            url.query().is_some(),
            "query string should be preserved for signed URLs"
        );
    }

    #[test]
    fn validate_file_url_malformed() {
        assert!(
            validate_file_url("not a url").is_err(),
            "malformed URL should be rejected"
        );
    }

    // sanitize_filename tests
    #[test]
    fn sanitize_filename_strips_path_separators() {
        assert_eq!(
            sanitize_filename("dir/file.txt"),
            Some("dirfile.txt".to_owned()),
            "forward slash should be stripped"
        );
        assert_eq!(
            sanitize_filename(r"dir\file.txt"),
            Some("dirfile.txt".to_owned()),
            "backslash should be stripped"
        );
    }

    #[test]
    fn sanitize_filename_removes_control_chars() {
        assert_eq!(
            sanitize_filename("file\x00\x1F.txt"),
            Some("file.txt".to_owned()),
            "control characters should be removed"
        );
    }

    #[test]
    fn sanitize_filename_bounds_length() {
        let long = "a".repeat(300);
        let sanitized = sanitize_filename(&long).unwrap();
        assert!(
            sanitized.len() <= 255,
            "filename should be bounded at 255 bytes on UTF-8 boundary"
        );
        assert!(
            sanitized.is_char_boundary(sanitized.len()),
            "truncation should respect UTF-8 boundaries"
        );
    }

    #[test]
    fn sanitize_filename_empty_input() {
        assert_eq!(sanitize_filename(""), None, "empty input should return None");
    }

    #[test]
    fn sanitize_filename_only_separators() {
        assert_eq!(
            sanitize_filename("///\\\\\\"),
            None,
            "input with only separators should return None"
        );
    }
}
