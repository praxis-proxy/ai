// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Domain allow-list enforced on proxy-executed web-search results.
//!
//! Search providers differ in whether and how they accept a domain
//! restriction, so the proxy enforces the client's allow-list itself on every
//! provider result before it can reach the model or a public source list
//! (issue #1768).

use url::Host;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum entries accepted in one `allowed_domains` list.
///
/// Matches the hosted OpenAI web-search limit and bounds the per-result
/// matching work a client can request.
pub(crate) const MAX_ALLOWED_DOMAINS: usize = 100;

// -----------------------------------------------------------------------------
// AllowedDomains
// -----------------------------------------------------------------------------

/// A normalized, non-empty domain allow-list.
///
/// A result is permitted when its URL host equals an allowed host or is a
/// subdomain of an allowed domain. An empty client list means "unrestricted"
/// and is represented by the absence of an [`AllowedDomains`], never by an
/// empty value: an empty value permits nothing, which callers use to fail closed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AllowedDomains {
    /// Parsed hosts, sorted and deduplicated so equal lists compare equal.
    hosts: Vec<Host<String>>,
}

impl AllowedDomains {
    /// Parse client-supplied domain entries.
    ///
    /// Each entry may carry an optional `http://`/`https://` prefix, a trailing
    /// `/`, or a trailing root `.`; the remainder must be a bare host. Domains
    /// are lowercased and IDNA-normalized exactly as result URLs are, so the
    /// comparison is like-for-like. Returns `Ok(None)` for an empty list.
    ///
    /// # Errors
    ///
    /// Returns a client-facing message when the list exceeds
    /// [`MAX_ALLOWED_DOMAINS`] or an entry is not a bare host.
    pub(crate) fn parse<S: AsRef<str>>(entries: impl Iterator<Item = S>) -> Result<Option<Self>, String> {
        // Parse at most one entry past the cap so an oversized list is rejected
        // without normalizing every entry first.
        let mut hosts = entries
            .take(MAX_ALLOWED_DOMAINS.saturating_add(1))
            .map(|entry| {
                parse_host(entry.as_ref()).ok_or_else(|| {
                    "filters.allowed_domains entries must be bare domain names such as `example.com`.".to_owned()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if hosts.len() > MAX_ALLOWED_DOMAINS {
            return Err(format!(
                "filters.allowed_domains accepts at most {MAX_ALLOWED_DOMAINS} domains."
            ));
        }
        hosts.sort();
        hosts.dedup();
        Ok((!hosts.is_empty()).then_some(Self { hosts }))
    }

    /// Whether `url`'s host is an allowed host or a subdomain of an allowed domain.
    ///
    /// A URL that does not parse or has no host is never permitted.
    pub(crate) fn permits_url(&self, url: &str) -> bool {
        let Ok(parsed) = url::Url::parse(url) else {
            return false;
        };
        match parsed.host() {
            Some(Host::Domain(host)) => {
                let host = host.strip_suffix('.').unwrap_or(host);
                self.hosts.iter().any(|allowed| match allowed {
                    Host::Domain(domain) => is_same_or_subdomain(host, domain),
                    Host::Ipv4(_) | Host::Ipv6(_) => false,
                })
            },
            Some(Host::Ipv4(ip)) => self.hosts.contains(&Host::Ipv4(ip)),
            Some(Host::Ipv6(ip)) => self.hosts.contains(&Host::Ipv6(ip)),
            None => false,
        }
    }
}

/// Parse one allow-list entry into a normalized host.
fn parse_host(entry: &str) -> Option<Host<String>> {
    let entry = entry.trim();
    let entry = strip_prefix_ignore_ascii_case(entry, "https://")
        .or_else(|| strip_prefix_ignore_ascii_case(entry, "http://"))
        .unwrap_or(entry);
    let entry = entry.strip_suffix('/').unwrap_or(entry);
    let entry = entry.strip_suffix('.').unwrap_or(entry);
    if entry.is_empty() {
        return None;
    }
    Host::parse(entry).ok()
}

/// Strip an ASCII `prefix` from `value`, ignoring ASCII case.
fn strip_prefix_ignore_ascii_case<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    let head = value.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| value.get(prefix.len()..))
        .flatten()
}

/// Whether `host` is `domain` itself or one of its subdomains.
fn is_same_or_subdomain(host: &str, domain: &str) -> bool {
    host.strip_suffix(domain)
        .is_some_and(|prefix| prefix.is_empty() || prefix.ends_with('.'))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    fn allowed(entries: &[&str]) -> AllowedDomains {
        AllowedDomains::parse(entries.iter().copied())
            .unwrap()
            .expect("non-empty list")
    }

    #[test]
    fn permits_exact_host_and_subdomains() {
        let domains = allowed(&["example.org"]);
        assert!(domains.permits_url("https://example.org/a"), "exact host is allowed");
        assert!(
            domains.permits_url("https://docs.example.org/a"),
            "subdomain is allowed"
        );
        assert!(
            domains.permits_url("http://a.b.example.org"),
            "nested subdomain is allowed"
        );
        assert!(
            domains.permits_url("https://example.org./a"),
            "root-dot host is allowed"
        );
    }

    #[test]
    fn rejects_lookalike_and_unrelated_hosts() {
        let domains = allowed(&["example.org"]);
        assert!(!domains.permits_url("https://outside.example/result"), "unrelated host");
        assert!(
            !domains.permits_url("https://badexample.org/"),
            "suffix without a label boundary"
        );
        assert!(
            !domains.permits_url("https://example.org.evil.test/"),
            "allowed domain as a left-hand label"
        );
        assert!(
            !domains.permits_url("https://user@evil.test/example.org"),
            "path is not the host"
        );
        assert!(!domains.permits_url("not a url"), "unparseable URL");
        assert!(!domains.permits_url("mailto:a@example.org"), "URL without a host");
    }

    #[test]
    fn normalizes_entries_like_result_hosts() {
        let domains = allowed(&[" HTTPS://Example.ORG/ ", "bücher.example."]);
        assert!(
            domains.permits_url("https://www.example.org/"),
            "case and scheme are normalized"
        );
        assert!(
            domains.permits_url("https://xn--bcher-kva.example/"),
            "IDN entry matches its punycode host"
        );
        assert_eq!(
            domains,
            allowed(&["bücher.example", "example.org", "example.org"]),
            "equivalent lists compare equal regardless of order or duplicates"
        );
    }

    #[test]
    fn ip_hosts_match_exactly() {
        let domains = allowed(&["192.0.2.1", "[2001:db8::1]"]);
        assert!(domains.permits_url("http://192.0.2.1/x"), "exact IPv4");
        assert!(domains.permits_url("http://[2001:db8::1]/x"), "exact IPv6");
        assert!(!domains.permits_url("http://192.0.2.2/x"), "different IPv4");
    }

    #[test]
    fn empty_list_is_unrestricted_and_default_permits_nothing() {
        assert!(
            AllowedDomains::parse(std::iter::empty::<&str>()).unwrap().is_none(),
            "an empty client list means no restriction"
        );
        assert!(
            !AllowedDomains::default().permits_url("https://example.org/"),
            "the empty fail-closed value permits nothing"
        );
    }

    #[test]
    fn rejects_malformed_entries_and_oversized_lists() {
        for entry in [
            "",
            " ",
            "/",
            "example.org/path",
            "example.org:443",
            "a b.example",
            "user@example.org",
        ] {
            assert!(
                AllowedDomains::parse(std::iter::once(entry)).is_err(),
                "entry {entry:?} must be rejected"
            );
        }
        let many: Vec<String> = (0..=MAX_ALLOWED_DOMAINS).map(|i| format!("d{i}.example")).collect();
        assert!(
            AllowedDomains::parse(many.iter().map(String::as_str)).is_err(),
            "lists over the cap must be rejected"
        );
    }
}
