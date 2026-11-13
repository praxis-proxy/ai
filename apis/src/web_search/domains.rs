// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Domain filters enforced on proxy-executed web-search results.
//!
//! Search providers differ in whether and how they accept a domain
//! restriction, so the proxy enforces the client's `filters.allowed_domains`
//! and `filters.blocked_domains` itself on every provider result before it can
//! reach the model or a public source list (issue #1768).

use url::Host;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum entries accepted in one `filters` domain list.
///
/// Matches the hosted OpenAI limit per list (`allowed_domains` or
/// `blocked_domains`) and bounds the per-result matching work a client can
/// request.
pub(crate) const MAX_DOMAINS_PER_LIST: usize = 100;

// -----------------------------------------------------------------------------
// DomainField
// -----------------------------------------------------------------------------

/// One client-declared `filters` domain list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DomainField {
    /// `filters.allowed_domains`: only these hosts and their subdomains are permitted.
    Allowed,
    /// `filters.blocked_domains`: these hosts and their subdomains are never permitted.
    Blocked,
}

impl DomainField {
    /// The JSON member name, used in client-facing messages.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed_domains",
            Self::Blocked => "blocked_domains",
        }
    }
}

// -----------------------------------------------------------------------------
// DomainList
// -----------------------------------------------------------------------------

/// A normalized, non-empty host list from one `filters` member.
///
/// A host matches when it equals an entry or is a subdomain of a domain entry.
/// Entries and result hosts share the same WHATWG/IDNA host parsing, so
/// `HTTPS://Example.ORG/` and `example.org` name the same domain; IP entries
/// match their exact address only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DomainList {
    /// Parsed hosts, sorted and deduplicated so equal lists compare equal.
    hosts: Vec<Host<String>>,
}

impl DomainList {
    /// Parse client-supplied domain entries.
    ///
    /// Each entry may carry an optional `http://`/`https://` prefix, a trailing
    /// `/`, or a trailing root `.`; the remainder must be a bare host. Returns
    /// `Ok(None)` for an empty list, which restricts nothing.
    ///
    /// # Errors
    ///
    /// Returns a client-facing message naming `field` when the list exceeds
    /// [`MAX_DOMAINS_PER_LIST`] or an entry is not a bare host.
    pub(crate) fn parse<S: AsRef<str>>(
        field: DomainField,
        entries: impl Iterator<Item = S>,
    ) -> Result<Option<Self>, String> {
        // Parse at most one entry past the cap so an oversized list is rejected
        // without normalizing every entry first.
        let mut hosts = entries
            .take(MAX_DOMAINS_PER_LIST.saturating_add(1))
            .map(|entry| {
                parse_host(entry.as_ref()).ok_or_else(|| {
                    format!(
                        "filters.{} entries must be bare domain names such as `example.com`.",
                        field.as_str()
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if hosts.len() > MAX_DOMAINS_PER_LIST {
            return Err(format!(
                "filters.{} accepts at most {MAX_DOMAINS_PER_LIST} domains.",
                field.as_str()
            ));
        }
        hosts.sort();
        hosts.dedup();
        Ok((!hosts.is_empty()).then_some(Self { hosts }))
    }

    /// Whether `url`'s host equals an entry or is a subdomain of a domain entry.
    ///
    /// A URL that does not parse or has no host never matches.
    pub(crate) fn contains_url(&self, url: &str) -> bool {
        let Ok(parsed) = url::Url::parse(url) else {
            return false;
        };
        match parsed.host() {
            Some(Host::Domain(host)) => {
                let host = host.strip_suffix('.').unwrap_or(host);
                self.hosts.iter().any(|entry| match entry {
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

// -----------------------------------------------------------------------------
// DomainFilters
// -----------------------------------------------------------------------------

/// The client's web-search domain filters, applied to every provider result.
///
/// The allow-list bounds which hosts may appear, then the block-list removes
/// hosts from that set, so a host matching both lists is blocked. An absent,
/// null, or empty list restricts nothing, matching hosted OpenAI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DomainFilters {
    /// The request declared no restriction: every provider result is permitted.
    Unrestricted,
    /// The request declared at least one non-empty list.
    Restricted {
        /// `filters.allowed_domains`, when non-empty. `None` permits every host.
        allowed: Option<DomainList>,
        /// `filters.blocked_domains`, when non-empty. `None` blocks no host.
        blocked: Option<DomainList>,
    },
    /// Fail closed: no result is permitted.
    ///
    /// Re-entry uses this when the request's declared filters could not be
    /// enforced, so a constrained search never silently widens to unrestricted.
    DenyAll,
}

impl DomainFilters {
    /// Combine one tool's parsed `filters` members.
    ///
    /// A tool that restricts nothing is [`DomainFilters::Unrestricted`], so
    /// equivalent declarations on different web-search tools compare equal.
    pub(crate) fn new(allowed: Option<DomainList>, blocked: Option<DomainList>) -> Self {
        if allowed.is_none() && blocked.is_none() {
            Self::Unrestricted
        } else {
            Self::Restricted { allowed, blocked }
        }
    }

    /// Whether `url` is permitted by the declared filters.
    pub(crate) fn permits_url(&self, url: &str) -> bool {
        match self {
            Self::Unrestricted => true,
            Self::DenyAll => false,
            Self::Restricted { allowed, blocked } => {
                allowed.as_ref().is_none_or(|allowed| allowed.contains_url(url))
                    && !blocked.as_ref().is_some_and(|blocked| blocked.contains_url(url))
            },
        }
    }
}

/// Parse one allow-list or block-list entry into a normalized host.
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

    fn list(entries: &[&str]) -> DomainList {
        DomainList::parse(DomainField::Allowed, entries.iter().copied())
            .unwrap()
            .expect("non-empty list")
    }

    fn permitted(allowed: Option<&[&str]>, blocked: Option<&[&str]>, url: &str) -> bool {
        DomainFilters::new(allowed.map(list), blocked.map(list)).permits_url(url)
    }

    #[test]
    fn permits_exact_host_and_subdomains() {
        let domains = list(&["example.org"]);
        assert!(domains.contains_url("https://example.org/a"), "exact host is allowed");
        assert!(
            domains.contains_url("https://docs.example.org/a"),
            "subdomain is allowed"
        );
        assert!(
            domains.contains_url("http://a.b.example.org"),
            "nested subdomain is allowed"
        );
        assert!(
            domains.contains_url("https://example.org./a"),
            "root-dot host is allowed"
        );
    }

    #[test]
    fn rejects_lookalike_and_unrelated_hosts() {
        let domains = list(&["example.org"]);
        assert!(
            !domains.contains_url("https://outside.example/result"),
            "unrelated host"
        );
        assert!(
            !domains.contains_url("https://badexample.org/"),
            "suffix without a label boundary"
        );
        assert!(
            !domains.contains_url("https://example.org.evil.test/"),
            "allowed domain as a left-hand label"
        );
        assert!(
            !domains.contains_url("https://user@evil.test/example.org"),
            "path is not the host"
        );
        assert!(!domains.contains_url("not a url"), "unparseable URL");
        assert!(!domains.contains_url("mailto:a@example.org"), "URL without a host");
    }

    #[test]
    fn normalizes_entries_like_result_hosts() {
        let domains = list(&[" HTTPS://Example.ORG/ ", "bücher.example."]);
        assert!(
            domains.contains_url("https://www.example.org/"),
            "case and scheme are normalized"
        );
        assert!(
            domains.contains_url("https://xn--bcher-kva.example/"),
            "IDN entry matches its punycode host"
        );
        assert_eq!(
            domains,
            list(&["bücher.example", "example.org", "example.org"]),
            "equivalent lists compare equal regardless of order or duplicates"
        );
    }

    #[test]
    fn ip_hosts_match_exactly() {
        let domains = list(&["192.0.2.1", "[2001:db8::1]"]);
        assert!(domains.contains_url("http://192.0.2.1/x"), "exact IPv4");
        assert!(domains.contains_url("http://[2001:db8::1]/x"), "exact IPv6");
        assert!(!domains.contains_url("http://192.0.2.2/x"), "different IPv4");
    }

    #[test]
    fn empty_list_is_unrestricted_and_deny_all_permits_nothing() {
        assert!(
            DomainList::parse(DomainField::Allowed, std::iter::empty::<&str>())
                .unwrap()
                .is_none(),
            "an empty client list means no restriction"
        );
        assert_eq!(
            DomainFilters::new(None, None),
            DomainFilters::Unrestricted,
            "a tool without lists restricts nothing"
        );
        assert!(
            !DomainFilters::DenyAll.permits_url("https://example.org/"),
            "the fail-closed value permits nothing"
        );
    }

    #[test]
    fn allowed_list_bounds_and_blocked_list_subtracts() {
        assert!(
            !permitted(Some(&["example.org"]), None, "https://outside.example/"),
            "a host outside the allow-list is not permitted"
        );
        assert!(
            !permitted(None, Some(&["example.org"]), "https://docs.example.org/"),
            "a subdomain of a blocked domain is not permitted"
        );
        assert!(
            permitted(None, Some(&["example.org"]), "https://badexample.org/"),
            "a suffix without a label boundary is not blocked"
        );
        assert!(
            !permitted(
                Some(&["example.org"]),
                Some(&["docs.example.org"]),
                "https://docs.example.org/"
            ),
            "the block-list subtracts from the allow-list"
        );
        assert!(
            permitted(
                Some(&["example.org"]),
                Some(&["docs.example.org"]),
                "https://www.example.org/"
            ),
            "a host inside the allow-list and outside the block-list is permitted"
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
            for field in [DomainField::Allowed, DomainField::Blocked] {
                assert!(
                    DomainList::parse(field, std::iter::once(entry)).is_err(),
                    "{field:?} entry {entry:?} must be rejected"
                );
            }
        }
        let many: Vec<String> = (0..=MAX_DOMAINS_PER_LIST).map(|i| format!("d{i}.example")).collect();
        for field in [DomainField::Allowed, DomainField::Blocked] {
            assert!(
                DomainList::parse(field, many.iter().map(String::as_str)).is_err(),
                "{field:?} lists over the cap must be rejected"
            );
        }
    }

    #[test]
    fn rejection_messages_name_the_offending_field() {
        let error = DomainList::parse(DomainField::Blocked, std::iter::once("a b")).unwrap_err();
        assert!(
            error.contains("filters.blocked_domains"),
            "the message names blocked_domains: {error}"
        );
        let many: Vec<String> = (0..=MAX_DOMAINS_PER_LIST).map(|i| format!("d{i}.example")).collect();
        let error = DomainList::parse(DomainField::Blocked, many.iter().map(String::as_str)).unwrap_err();
        assert!(
            error.contains("filters.blocked_domains"),
            "the cap message names blocked_domains: {error}"
        );
    }
}
