// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! URL resolution and SSRF validation for `file_url` references.

use std::net::IpAddr;

use praxis_core::connectivity::normalize_mapped_ipv4;

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
    #[cfg_attr(not(test), expect(dead_code, reason = "used in Task 3"))]
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

/// Cloud metadata endpoints that are never permitted.
fn is_cloud_metadata(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            *v4 == std::net::Ipv4Addr::new(169, 254, 169, 254) || *v4 == std::net::Ipv4Addr::new(100, 100, 100, 200)
        },
        IpAddr::V6(v6) => {
            const AWS_IMDS_V6: std::net::Ipv6Addr = std::net::Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0254);
            *v6 == AWS_IMDS_V6
        },
    }
}

/// Resolves `file_url` references with DNS pinning and SSRF protection.
#[expect(dead_code, reason = "used in Task 3")] // Used in Task 3
pub(crate) struct FileUrlResolver {
    /// Origins that permit resolution to private/loopback addresses.
    pub(crate) allowed_private_origins: Vec<NormalizedOrigin>,
    /// Overall timeout for URL resolution.
    pub(crate) timeout: std::time::Duration,
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
}
