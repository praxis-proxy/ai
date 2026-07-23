// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! URL resolution and SSRF validation for `file_url` references.

use std::net::{IpAddr, SocketAddr};

use praxis_core::connectivity::normalize_mapped_ipv4;

use super::resolve::{ResolveError, ResolvedFile, infer_mime_from_filename, max_content_bytes_for_data_url};
use crate::openai::url_security::{is_cloud_metadata, is_file_url_ssrf_blocked};

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

/// Redact sensitive parts of a URL for safe logging.
///
/// Strips credentials and fragments, and replaces query parameter values
/// with `[REDACTED]`.
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

    url.set_fragment(None);

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

/// Extract filename from a `Content-Disposition` header value (RFC 6266).
///
/// `filename*` (RFC 5987) always takes precedence over `filename` regardless
/// of parameter order. Parameter names are matched case-insensitively.
/// Quoted values may contain embedded semicolons.
fn parse_content_disposition_filename(header: &str) -> Option<String> {
    let params = split_header_params(header);
    // First pass: prefer filename* (RFC 5987 extended notation)
    for (name, value) in &params {
        if name.eq_ignore_ascii_case("filename*")
            && let Some(decoded) = decode_rfc5987(value)
            && !decoded.is_empty()
        {
            return Some(decoded);
        }
    }
    // Second pass: fall back to filename
    for (name, value) in &params {
        if name.eq_ignore_ascii_case("filename") {
            let unquoted = unquote(value);
            if !unquoted.is_empty() {
                return Some(unquoted);
            }
        }
    }
    None
}

/// Split a header value into `(name, value)` parameter pairs, respecting
/// quoted strings so that semicolons inside quotes do not split.
fn split_header_params(header: &str) -> Vec<(String, String)> {
    split_on_unquoted_semicolons(header)
        .into_iter()
        .skip(1)
        .filter_map(|seg| {
            let seg = seg.trim();
            let (name, value) = seg.split_once('=')?;
            let name = name.trim();
            (!name.is_empty()).then(|| (name.to_owned(), value.trim().to_owned()))
        })
        .collect()
}

/// Split on `;` outside of double-quoted strings.
fn split_on_unquoted_semicolons(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    let mut prev_backslash = false;
    for (i, c) in s.char_indices() {
        if prev_backslash {
            prev_backslash = false;
            continue;
        }
        match c {
            '\\' if in_quote => prev_backslash = true,
            '"' => in_quote = !in_quote,
            ';' if !in_quote => {
                parts.push(s.get(start..i).unwrap_or_default());
                start = i + 1;
            },
            _ => {},
        }
    }
    parts.push(s.get(start..).unwrap_or_default());
    parts
}

/// Decode an RFC 5987 `charset'language'percent-encoded` value.
fn decode_rfc5987(raw: &str) -> Option<String> {
    let (_, after_charset) = raw.split_once('\'')?;
    let (_, encoded) = after_charset.split_once('\'')?;
    Some(percent_decode_utf8(encoded))
}

/// Strip surrounding double-quotes and unescape backslash-escaped characters.
fn unquote(s: &str) -> String {
    let s = s.trim();
    let inner = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')).unwrap_or(s);
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(escaped) = chars.next() {
                out.push(escaped);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Percent-decode a UTF-8 string, replacing invalid sequences with `_`.
fn percent_decode_utf8(input: &str) -> String {
    let mut bytes = Vec::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if hex.len() == 2
                && let Ok(byte) = u8::from_str_radix(&hex, 16)
            {
                bytes.push(byte);
                continue;
            }
            bytes.push(b'_');
        } else {
            let mut buf = [0_u8; 4];
            bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    String::from_utf8(bytes).unwrap_or_default()
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
            return Err(ResolveError::TooLarge {
                reference: label,
                limit: max_resolved_bytes,
            });
        }

        // Derive filename before consuming body: prefer Content-Disposition, fall back to URL path
        let filename = response
            .headers()
            .get(http::header::CONTENT_DISPOSITION)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_content_disposition_filename)
            .and_then(|f| sanitize_filename(&f))
            .or_else(|| {
                parsed_url
                    .path_segments()
                    .and_then(|mut segments| segments.next_back())
                    .and_then(sanitize_filename)
            });

        // Read bounded body
        let content = read_bounded_url_body(response, &label, max_content_bytes, max_resolved_bytes).await?;

        // Encode as base64
        let base64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &content);

        Ok(ResolvedFile {
            base64,
            content_type,
            filename,
        })
    }
}

/// Read a file URL response body while preserving URL-specific error context.
async fn read_bounded_url_body(
    mut response: reqwest::Response,
    label: &str,
    max_content_bytes: usize,
    limit: usize,
) -> Result<Vec<u8>, ResolveError> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| ResolveError::FileUrlFailed {
        label: label.to_owned(),
        detail: if e.is_timeout() {
            "content download timed out".to_owned()
        } else {
            format!("content download read error: {e}")
        },
    })? {
        if chunk.len() > max_content_bytes.saturating_sub(body.len()) {
            return Err(ResolveError::TooLarge {
                reference: label.to_owned(),
                limit,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
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

    let validated = validate_resolved_addrs(addrs, allow_private, label)?;
    Ok((host.to_owned(), validated))
}

/// Validate and deduplicate every address returned for one DNS lookup.
fn validate_resolved_addrs(
    addrs: Vec<SocketAddr>,
    allow_private: bool,
    label: &str,
) -> Result<Vec<SocketAddr>, ResolveError> {
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

    Ok(validated)
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
    configure_pinned_client(reqwest::Client::builder(), host, addrs, timeout)
}

/// Apply proxy, redirect, timeout, and DNS-pinning policy to a client builder.
fn configure_pinned_client(
    builder: reqwest::ClientBuilder,
    host: &str,
    addrs: &[SocketAddr],
    timeout: std::time::Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    let mut builder = builder
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
    use std::{
        io::{Read as _, Write as _},
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
    };

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
        assert!(
            NormalizedOrigin::parse("ftp://example.com").is_err(),
            "ftp should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("file:///tmp/x").is_err(),
            "file scheme should be rejected"
        );
    }

    #[test]
    fn parse_rejects_credentials() {
        assert!(
            NormalizedOrigin::parse("https://user@example.com").is_err(),
            "username should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("https://user:pass@example.com").is_err(),
            "user:pass should be rejected"
        );
    }

    #[test]
    fn parse_rejects_path() {
        assert!(
            NormalizedOrigin::parse("https://example.com/files").is_err(),
            "path should be rejected"
        );
    }

    #[test]
    fn parse_accepts_root_path() {
        assert!(
            NormalizedOrigin::parse("https://example.com/").is_ok(),
            "root path should be accepted"
        );
    }

    #[test]
    fn parse_rejects_query() {
        assert!(
            NormalizedOrigin::parse("https://example.com?q=1").is_err(),
            "query should be rejected"
        );
    }

    #[test]
    fn parse_rejects_fragment() {
        assert!(
            NormalizedOrigin::parse("https://example.com#frag").is_err(),
            "fragment should be rejected"
        );
    }

    #[test]
    fn parse_rejects_cloud_metadata_ipv4() {
        assert!(
            NormalizedOrigin::parse("http://169.254.169.254").is_err(),
            "IMDS should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("http://100.100.100.200").is_err(),
            "Alibaba metadata should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("http://169.254.0.23").is_err(),
            "Tencent metadata 169.254.0.23 should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("http://169.254.10.10").is_err(),
            "Tencent metadata 169.254.10.10 should be rejected"
        );
    }

    #[test]
    fn parse_rejects_cloud_metadata_ipv6() {
        assert!(
            NormalizedOrigin::parse("http://[fd00:ec2::254]").is_err(),
            "IMDS v6 should be rejected"
        );
    }

    #[test]
    fn parse_rejects_unspecified() {
        assert!(
            NormalizedOrigin::parse("http://0.0.0.0").is_err(),
            "unspecified v4 should be rejected"
        );
        assert!(
            NormalizedOrigin::parse("http://[::]").is_err(),
            "unspecified v6 should be rejected"
        );
    }

    #[test]
    fn parse_rejects_multicast() {
        assert!(
            NormalizedOrigin::parse("http://224.0.0.1").is_err(),
            "multicast should be rejected"
        );
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
    fn ssrf_blocks_tencent_metadata() {
        assert!(
            is_file_url_ssrf_blocked(&"169.254.0.23".parse().unwrap(), true),
            "Tencent metadata 169.254.0.23 should be blocked even with allowlist"
        );
        assert!(
            is_file_url_ssrf_blocked(&"169.254.10.10".parse().unwrap(), true),
            "Tencent metadata 169.254.10.10 should be blocked even with allowlist"
        );
    }

    #[test]
    fn ssrf_blocks_tencent_metadata_via_nat64() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::a9fe:0017".parse().unwrap(), true),
            "Tencent 169.254.0.23 via NAT64 should be blocked even with allowlist"
        );
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::a9fe:0a0a".parse().unwrap(), true),
            "Tencent 169.254.10.10 via NAT64 should be blocked even with allowlist"
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

    // IPv4 special-use range tests
    #[test]
    fn ssrf_blocks_benchmarking_range() {
        assert!(
            is_file_url_ssrf_blocked(&"198.18.0.1".parse().unwrap(), false),
            "198.18.0.0/15 lower bound should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"198.19.255.254".parse().unwrap(), false),
            "198.18.0.0/15 upper bound should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_ietf_protocol_assignments() {
        assert!(
            is_file_url_ssrf_blocked(&"192.0.0.1".parse().unwrap(), false),
            "192.0.0.0/24 should be blocked"
        );
    }

    #[test]
    fn ssrf_allows_globally_reachable_in_192_0_0() {
        assert!(
            !is_file_url_ssrf_blocked(&"192.0.0.9".parse().unwrap(), false),
            "192.0.0.9 is globally reachable per IANA"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"192.0.0.10".parse().unwrap(), false),
            "192.0.0.10 is globally reachable per IANA"
        );
    }

    #[test]
    fn ssrf_blocks_6to4_relay_anycast() {
        assert!(
            is_file_url_ssrf_blocked(&"192.88.99.1".parse().unwrap(), false),
            "192.88.99.0/24 deprecated 6to4 relay should be blocked"
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
            "240.0.0.0/4 should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"255.255.255.254".parse().unwrap(), false),
            "255.x reserved should be blocked"
        );
    }

    // IPv6 special-use range tests
    #[test]
    fn ssrf_blocks_discard_only_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"100::".parse().unwrap(), false),
            "100::/64 discard-only should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"100:0:0:1::1".parse().unwrap(), false),
            "100:0:0:1::/64 dummy prefix should be blocked"
        );
    }

    #[test]
    fn ssrf_allows_outside_discard_prefix() {
        assert!(
            !is_file_url_ssrf_blocked(&"100:0:0:2::1".parse().unwrap(), false),
            "100:0:0:2:: is outside the registered discard prefixes"
        );
    }

    #[test]
    fn ssrf_blocks_nat64_local_use() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b:1::1".parse().unwrap(), false),
            "64:ff9b:1::/48 local-use NAT64 should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_6to4_v6() {
        assert!(
            is_file_url_ssrf_blocked(&"2002:c0a8:1::1".parse().unwrap(), false),
            "6to4 2002::/16 should be blocked (deprecated RFC 7526)"
        );
    }

    // 2001::/23 IETF Protocol Assignments
    #[test]
    fn ssrf_blocks_2001_non_global() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:5::1".parse().unwrap(), false),
            "2001:5:: is non-global within 2001::/23"
        );
        assert!(
            is_file_url_ssrf_blocked(&"2001:1ff::1".parse().unwrap(), false),
            "2001:1ff:: is at the edge of 2001::/23"
        );
    }

    #[test]
    fn ssrf_blocks_2001_benchmarking() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:2:0::1".parse().unwrap(), false),
            "2001:2::/48 benchmarking should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_2001_documentation() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:db8::1".parse().unwrap(), false),
            "2001:db8::/32 documentation should be blocked"
        );
    }

    #[test]
    fn ssrf_blocks_2001_teredo() {
        assert!(
            is_file_url_ssrf_blocked(&"2001::1".parse().unwrap(), false),
            "2001::/32 Teredo has N/A global reachability, treat as transition range"
        );
    }

    #[test]
    fn ssrf_allows_2001_anycast_128s() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:1::1".parse().unwrap(), false),
            "2001:1::1/128 PCP anycast is globally reachable"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:1::2".parse().unwrap(), false),
            "2001:1::2/128 TURN anycast is globally reachable"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:1::3".parse().unwrap(), false),
            "2001:1::3/128 DNS-SD anycast is globally reachable"
        );
    }

    #[test]
    fn ssrf_blocks_2001_1_non_anycast() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:1::4".parse().unwrap(), false),
            "2001:1::4 is not a registered anycast address"
        );
        assert!(
            is_file_url_ssrf_blocked(&"2001:1::100".parse().unwrap(), false),
            "2001:1::100 is not a registered anycast address"
        );
    }

    #[test]
    fn ssrf_allows_2001_amt() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:3::1".parse().unwrap(), false),
            "2001:3::/32 AMT is globally reachable"
        );
    }

    #[test]
    fn ssrf_allows_2001_as112() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:4:112::1".parse().unwrap(), false),
            "2001:4:112::/48 AS112-v6 is globally reachable"
        );
    }

    #[test]
    fn ssrf_allows_2001_orchidv2() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:20::1".parse().unwrap(), false),
            "2001:20::/28 ORCHIDv2 is globally reachable"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:2f::1".parse().unwrap(), false),
            "2001:2f:: is at the edge of ORCHIDv2"
        );
    }

    #[test]
    fn ssrf_allows_2001_drone_remote_id() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:30::1".parse().unwrap(), false),
            "2001:30::/28 Drone Remote ID is globally reachable"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:3f::1".parse().unwrap(), false),
            "2001:3f:: is at the edge of Drone Remote ID"
        );
    }

    #[test]
    fn ssrf_blocks_between_global_exceptions() {
        assert!(
            is_file_url_ssrf_blocked(&"2001:40::1".parse().unwrap(), false),
            "2001:40:: is outside all globally reachable exceptions"
        );
    }

    #[test]
    fn ssrf_allows_outside_2001_23() {
        assert!(
            !is_file_url_ssrf_blocked(&"2001:200::1".parse().unwrap(), false),
            "2001:200:: is outside 2001::/23"
        );
    }

    // 3fff::/20 and 5f00::/16
    #[test]
    fn ssrf_blocks_documentation_rfc9637() {
        assert!(
            is_file_url_ssrf_blocked(&"3fff::1".parse().unwrap(), false),
            "3fff::1 documentation prefix (RFC 9637) should be blocked"
        );
        assert!(
            is_file_url_ssrf_blocked(&"3fff:0fff::1".parse().unwrap(), false),
            "3fff:0fff:: is inside 3fff::/20"
        );
    }

    #[test]
    fn ssrf_allows_outside_3fff_20() {
        assert!(
            !is_file_url_ssrf_blocked(&"3fff:1000::1".parse().unwrap(), false),
            "3fff:1000:: is outside 3fff::/20"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"3ffe::1".parse().unwrap(), false),
            "3ffe:: is outside 3fff::/20"
        );
    }

    #[test]
    fn ssrf_blocks_srv6_sid() {
        assert!(
            is_file_url_ssrf_blocked(&"5f00::1".parse().unwrap(), false),
            "5f00::/16 SRv6 SID should be blocked"
        );
    }

    // NAT64 Well-Known Prefix (64:ff9b::/96) — embedded IPv4 policy (RFC 6052)
    #[test]
    fn ssrf_blocks_nat64_embedded_metadata() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::a9fe:a9fe".parse().unwrap(), false),
            "64:ff9b::169.254.169.254 embeds cloud metadata"
        );
    }

    #[test]
    fn ssrf_blocks_nat64_embedded_loopback() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::7f00:1".parse().unwrap(), false),
            "64:ff9b::127.0.0.1 embeds loopback"
        );
    }

    #[test]
    fn ssrf_blocks_nat64_embedded_private() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::c0a8:1".parse().unwrap(), false),
            "64:ff9b::192.168.0.1 embeds private range"
        );
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::a9fe:aa02".parse().unwrap(), false),
            "64:ff9b::169.254.170.2 embeds ECS credential endpoint"
        );
    }

    #[test]
    fn ssrf_allows_nat64_embedded_public() {
        assert!(
            !is_file_url_ssrf_blocked(&"64:ff9b::808:808".parse().unwrap(), false),
            "64:ff9b::8.8.8.8 embeds public DNS, should be allowed"
        );
    }

    #[test]
    fn ssrf_allows_nat64_embedded_private_with_allowlist() {
        assert!(
            !is_file_url_ssrf_blocked(&"64:ff9b::c0a8:1".parse().unwrap(), true),
            "64:ff9b::192.168.0.1 should be allowed with allowlist override"
        );
    }

    #[test]
    fn ssrf_blocks_nat64_embedded_metadata_even_with_allowlist() {
        assert!(
            is_file_url_ssrf_blocked(&"64:ff9b::a9fe:a9fe".parse().unwrap(), true),
            "cloud metadata via NAT64 is unconditionally blocked"
        );
    }

    // Allowlist override
    #[test]
    fn ssrf_allows_special_use_with_allowlist() {
        assert!(
            !is_file_url_ssrf_blocked(&"198.18.0.1".parse().unwrap(), true),
            "198.18.0.0/15 should be allowed with allowlist override"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2001:db8::1".parse().unwrap(), true),
            "2001:db8::/32 should be allowed with allowlist override"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"64:ff9b:1::1".parse().unwrap(), true),
            "64:ff9b:1::/48 should be allowed with allowlist override"
        );
        assert!(
            !is_file_url_ssrf_blocked(&"2002:c0a8:1::1".parse().unwrap(), true),
            "2002::/16 (6to4) should be allowed with allowlist override"
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
        assert!(is_valid_mime_type("text/plain"), "text/plain should be valid");
        assert!(
            is_valid_mime_type("application/octet-stream"),
            "application/octet-stream should be valid"
        );
        assert!(is_valid_mime_type("image/png"), "image/png should be valid");
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

    #[test]
    fn redact_url_strips_fragment() {
        let redacted = redact_url("https://example.com/file#access_token=secret");
        assert!(
            !redacted.contains("access_token"),
            "fragment should be stripped entirely"
        );
        assert!(!redacted.contains('#'), "fragment delimiter should not remain");
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

    // Content-Disposition filename tests
    #[test]
    fn content_disposition_quoted_filename() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename=\"report.pdf\""),
            Some("report.pdf".to_owned()),
            "should extract quoted filename"
        );
    }

    #[test]
    fn content_disposition_unquoted_filename() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename=report.pdf"),
            Some("report.pdf".to_owned()),
            "should extract unquoted filename"
        );
    }

    #[test]
    fn content_disposition_star_notation() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf"),
            Some("résumé.pdf".to_owned()),
            "should decode RFC 5987 filename*"
        );
    }

    #[test]
    fn content_disposition_star_preferred_over_plain() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename*=UTF-8''correct.pdf; filename=\"fallback.pdf\""),
            Some("correct.pdf".to_owned()),
            "filename* should take precedence over filename"
        );
    }

    #[test]
    fn content_disposition_plain_before_star() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename=\"fallback.pdf\"; filename*=UTF-8''correct.pdf"),
            Some("correct.pdf".to_owned()),
            "filename* should win even when filename appears first"
        );
    }

    #[test]
    fn content_disposition_case_insensitive() {
        assert_eq!(
            parse_content_disposition_filename("attachment; FILENAME=\"report.pdf\""),
            Some("report.pdf".to_owned()),
            "parameter names should be case-insensitive"
        );
        assert_eq!(
            parse_content_disposition_filename("attachment; FileName*=UTF-8''r%C3%A9sum%C3%A9.pdf"),
            Some("résumé.pdf".to_owned()),
            "filename* should be case-insensitive"
        );
    }

    #[test]
    fn content_disposition_non_empty_language_tag() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename*=UTF-8'en'actual.pdf"),
            Some("actual.pdf".to_owned()),
            "non-empty language tag should be accepted"
        );
    }

    #[test]
    fn content_disposition_quoted_semicolon() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename=\"file;name.pdf\""),
            Some("file;name.pdf".to_owned()),
            "semicolons inside quotes should not split the value"
        );
    }

    #[test]
    fn content_disposition_inline() {
        assert_eq!(
            parse_content_disposition_filename("inline"),
            None,
            "inline without filename should return None"
        );
    }

    #[test]
    fn content_disposition_empty_filename() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename=\"\""),
            None,
            "empty quoted filename should return None"
        );
    }

    // Coverage: matches_url with no host (cannot-be-a-base URL)
    #[test]
    fn matches_url_no_host() {
        let origin = NormalizedOrigin::parse("https://example.com").unwrap();
        let url = url::Url::parse("data:text/plain,hello").unwrap();
        assert!(!origin.matches_url(&url), "cannot-be-a-base URL has no host");
    }

    // Coverage: backslash escape inside quoted Content-Disposition value
    #[test]
    fn content_disposition_escaped_quote_in_filename() {
        assert_eq!(
            parse_content_disposition_filename(r#"attachment; filename="file\"name.pdf""#),
            Some("file\"name.pdf".to_owned()),
            "backslash-escaped quote inside filename should be unescaped"
        );
    }

    // Coverage: invalid percent sequence in percent_decode_utf8
    #[test]
    fn content_disposition_invalid_percent_encoding() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename*=UTF-8''file%ZZname.pdf"),
            Some("file_name.pdf".to_owned()),
            "invalid percent sequence should produce underscore replacement"
        );
    }

    // Coverage: multi-byte UTF-8 truncation in sanitize_filename
    #[test]
    fn sanitize_filename_multibyte_truncation() {
        // 254 ASCII bytes + 2-byte UTF-8 char = 256 bytes, exceeds 255
        let mut name = "a".repeat(254);
        name.push('\u{00E9}'); // é is 2 bytes in UTF-8
        let sanitized = sanitize_filename(&name).unwrap();
        assert!(
            sanitized.len() <= 255,
            "multi-byte truncation should stay within 255 bytes"
        );
        assert_eq!(sanitized.len(), 254, "should truncate to 254 (before the 2-byte char)");
    }

    // Coverage: Content-Disposition with no = sign in a segment
    #[test]
    fn content_disposition_no_equals() {
        assert_eq!(
            parse_content_disposition_filename("attachment; noequals"),
            None,
            "parameter without = should be skipped"
        );
    }

    // Coverage: Content-Disposition with empty parameter name
    #[test]
    fn content_disposition_empty_param_name() {
        assert_eq!(
            parse_content_disposition_filename("attachment; =value"),
            None,
            "empty parameter name should be skipped"
        );
    }

    // Coverage: percent_decode_utf8 truncated sequence (single char after %)
    #[test]
    fn content_disposition_truncated_percent() {
        assert_eq!(
            parse_content_disposition_filename("attachment; filename*=UTF-8''file%2"),
            Some("file_".to_owned()),
            "truncated percent sequence should produce underscore"
        );
    }

    // Coverage: MIME type edge cases for is_valid_mime_type
    #[test]
    fn mime_rejects_double_slash() {
        assert!(
            !is_valid_mime_type("text/plain/extra"),
            "MIME with multiple slashes should be rejected (subtype contains /)"
        );
    }

    #[test]
    fn mime_accepts_with_dash_dot_plus() {
        assert!(
            is_valid_mime_type("application/vnd.openxml-officedocument+xml"),
            "MIME with dashes, dots, and plus should be accepted"
        );
    }

    // Coverage: is_blocked_hostname
    #[test]
    fn blocked_hostname_localhost() {
        assert!(is_blocked_hostname("localhost"), "localhost should be blocked");
        assert!(
            is_blocked_hostname("app.localhost"),
            ".localhost subdomain should be blocked"
        );
        assert!(
            !is_blocked_hostname("example.com"),
            "non-localhost should not be blocked"
        );
    }

    // Coverage: NormalizedOrigin::parse with non-IP hostname
    #[test]
    fn parse_accepts_dns_hostname() {
        let origin = NormalizedOrigin::parse("https://files.example.com").unwrap();
        assert_eq!(origin.host, "files.example.com", "hostname should be preserved");
    }

    // Coverage: validate_file_url accepts localhost (validation doesn't check hostname)
    #[test]
    fn validate_file_url_passes_localhost() {
        assert!(
            validate_file_url("https://localhost/file.pdf").is_ok(),
            "validate_file_url only checks scheme/credentials/fragment, not hostname"
        );
    }

    #[tokio::test]
    async fn file_url_resolver_blocks_legacy_ipv4_loopback_encodings() {
        let resolver = FileUrlResolver {
            allowed_private_origins: vec![],
        };

        for url in [
            "http://0177.0.0.1/file.txt",
            "http://127.1/file.txt",
            "http://2130706433/file.txt",
            "http://0x7f000001/file.txt",
        ] {
            let result = resolver
                .resolve_url(
                    url,
                    tokio::time::Instant::now() + std::time::Duration::from_secs(1),
                    1024,
                )
                .await;

            assert!(
                matches!(result, Err(ResolveError::FileUrlBlocked { .. })),
                "URL-parser canonicalization must not let legacy IPv4 loopback bypass SSRF checks: {url}"
            );
        }
    }

    struct StalledBodyServer {
        address: SocketAddr,
        headers_sent: Arc<AtomicBool>,
        release: mpsc::Sender<()>,
        thread: std::thread::JoinHandle<()>,
    }

    fn start_stalled_body_server() -> StalledBodyServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let headers_sent = Arc::new(AtomicBool::new(false));
        let server_headers_sent = Arc::clone(&headers_sent);
        let (release, release_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
            stream.flush().unwrap();
            server_headers_sent.store(true, Ordering::Release);
            let _released = release_rx.recv_timeout(std::time::Duration::from_secs(5));
        });
        StalledBodyServer {
            address,
            headers_sent,
            release,
            thread,
        }
    }

    #[tokio::test]
    async fn pinned_client_times_out_while_response_body_is_stalled() {
        let server = start_stalled_body_server();
        let resolver = FileUrlResolver {
            allowed_private_origins: vec![NormalizedOrigin::parse(&format!("http://{}", server.address)).unwrap()],
        };
        let result = resolver
            .resolve_url(
                &format!("http://{}/file.txt", server.address),
                tokio::time::Instant::now() + std::time::Duration::from_millis(500),
                1024,
            )
            .await;

        server.release.send(()).unwrap();
        server.thread.join().unwrap();
        assert!(
            server.headers_sent.load(Ordering::Acquire),
            "the response headers must arrive before the body-read timeout"
        );
        match result {
            Err(ResolveError::FileUrlFailed { detail, .. }) => {
                assert!(
                    detail.contains("timed out"),
                    "a body that stalls after headers must report a timeout"
                );
            },
            Err(other) => panic!("expected URL timeout for a stalled body, got {other}"),
            Ok(_) => panic!("a body that stalls after headers must fail at the shared deadline"),
        }
    }

    #[test]
    fn dns_rebinding_with_mixed_public_and_private_answers_is_blocked() {
        let addrs = vec!["93.184.216.34:443".parse().unwrap(), "127.0.0.1:443".parse().unwrap()];

        let result = validate_resolved_addrs(addrs, false, "https://files.example/document.pdf");

        assert!(
            matches!(result, Err(ResolveError::FileUrlBlocked { .. })),
            "one private DNS answer must reject the entire pinned address set"
        );
    }

    #[tokio::test]
    async fn pinned_client_uses_only_validated_dns_addresses() {
        let target_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let target_address = target_listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = target_listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
        });

        let client = build_pinned_client(
            "files.example.test",
            &[target_address],
            std::time::Duration::from_secs(5),
        )
        .unwrap();
        let response = client
            .get(format!("http://files.example.test:{}/file.txt", target_address.port()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"ok");
        server.join().unwrap();
    }

    fn start_redirect_server(target_address: SocketAddr) -> (SocketAddr, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            let response = format!(
                "HTTP/1.1 302 Found\r\nLocation: http://{target_address}/secret\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (address, server)
    }

    #[tokio::test]
    async fn file_url_resolver_does_not_follow_redirects() {
        let target_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        target_listener.set_nonblocking(true).unwrap();
        let target_address = target_listener.local_addr().unwrap();
        let (redirect_address, server) = start_redirect_server(target_address);

        let resolver = FileUrlResolver {
            allowed_private_origins: vec![NormalizedOrigin::parse(&format!("http://{redirect_address}")).unwrap()],
        };
        let result = resolver
            .resolve_url(
                &format!("http://{redirect_address}/file.txt"),
                tokio::time::Instant::now() + std::time::Duration::from_secs(5),
                1024,
            )
            .await;
        server.join().unwrap();

        match result {
            Err(ResolveError::FileUrlFailed { detail, .. }) => {
                assert!(detail.contains("302"), "redirect response should be rejected directly");
            },
            Err(other) => panic!("expected URL fetch failure for redirect, got {other}"),
            Ok(_) => panic!("redirect must not be followed"),
        }
        assert!(
            matches!(target_listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "redirect target must not receive a connection"
        );
    }

    #[tokio::test]
    async fn pinned_client_disables_configured_proxy() {
        let target_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let target_address = target_listener.local_addr().unwrap();
        let proxy_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        proxy_listener.set_nonblocking(true).unwrap();
        let proxy_address = proxy_listener.local_addr().unwrap();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = target_listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _read = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
        });

        let builder = reqwest::Client::builder().proxy(reqwest::Proxy::all(format!("http://{proxy_address}")).unwrap());
        let client = configure_pinned_client(builder, "127.0.0.1", &[], std::time::Duration::from_secs(5)).unwrap();
        let response = client
            .get(format!("http://{target_address}/file.txt"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"ok");
        server.join().unwrap();

        assert!(
            matches!(proxy_listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "configured proxy must not receive a connection"
        );
    }

    // Coverage: NormalizedOrigin default HTTP port
    #[test]
    fn parse_normalizes_http_default_port() {
        let origin = NormalizedOrigin::parse("http://example.com").unwrap();
        assert_eq!(origin.port, 80, "HTTP default port should be 80");
    }
}
