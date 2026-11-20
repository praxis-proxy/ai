// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral domain filters for proxy-executed web-search results.
//!
//! Search providers differ in whether and how they accept a domain
//! restriction, so the proxy enforces the client's allow-list and block-list
//! itself on every provider result before it can reach the model or a public
//! source list (issue #1768). Protocol adapters own the wire spelling, list
//! size limits, and rejection responses; this module owns URL parsing and host
//! matching so every web-search protocol enforces the same semantics.

// The OpenAI Responses adapter is the only consumer today, but any web-search
// protocol adapter may use this matcher without enabling that feature.
#![cfg_attr(
    all(not(feature = "openai-responses"), not(test)),
    expect(
        dead_code,
        reason = "shared matcher: the OpenAI Responses adapter is the only consumer today"
    )
)]

use url::{Host, Url};

// -----------------------------------------------------------------------------
// InvalidDomainEntry
// -----------------------------------------------------------------------------

/// A declared domain entry that is not a bare host name.
///
/// Adapters map this to their protocol's rejection response; the error carries
/// no client-supplied text so it can never be echoed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InvalidDomainEntry;

// -----------------------------------------------------------------------------
// DomainList
// -----------------------------------------------------------------------------

/// A normalized, non-empty host list from one declared allow-list or block-list.
///
/// A host matches when it equals an entry or is a subdomain of a domain entry.
/// Entries and result hosts share the same WHATWG/IDNA host parsing, so
/// `HTTPS://Example.ORG/` and `example.org` name the same domain; IP entries
/// match their exact address only. Wildcard entries are rejected: a stored
/// `*.example.org` would never match a URL host, so accepting it would silently
/// enforce nothing.
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
    /// `Ok(None)` for an empty list, which restricts nothing. Protocol adapters
    /// enforce their own entry cap before calling this.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidDomainEntry`] when an entry is not a bare host or
    /// contains a wildcard.
    pub(crate) fn parse<S: AsRef<str>>(entries: impl Iterator<Item = S>) -> Result<Option<Self>, InvalidDomainEntry> {
        let mut hosts = entries
            .map(|entry| parse_host(entry.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        hosts.sort();
        hosts.dedup();
        Ok((!hosts.is_empty()).then_some(Self { hosts }))
    }

    /// Whether an already-parsed result host matches this list.
    fn contains_host(&self, host: &Host<&str>) -> bool {
        self.hosts.iter().any(|entry| match (entry, host) {
            (Host::Domain(domain), Host::Domain(host)) => is_same_or_subdomain(host, domain),
            (Host::Ipv4(entry), Host::Ipv4(host)) => entry == host,
            (Host::Ipv6(entry), Host::Ipv6(host)) => entry == host,
            _ => false,
        })
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
    ///
    /// Any restriction requires a verifiable result host: a URL that is not an
    /// absolute `http`/`https` URL with a matchable host is dropped, so a
    /// block-list-only request cannot retain a malformed, scheme-relative, or
    /// opaque-scheme provider URL.
    pub(crate) fn permits_url(&self, url: &str) -> bool {
        let Self::Restricted { allowed, blocked } = self else {
            return matches!(self, Self::Unrestricted);
        };
        let Ok(parsed) = Url::parse(url) else {
            return false;
        };
        // A non-web scheme keeps the raw host spelling (for example
        // `custom://EXAMPLE.ORG`), so it cannot be compared reliably against a
        // normalized list entry.
        if !matches!(parsed.scheme(), "http" | "https") {
            return false;
        }
        let Some(host) = parsed.host().and_then(normalized_host) else {
            return false;
        };
        allowed.as_ref().is_none_or(|allowed| allowed.contains_host(&host))
            && !blocked.as_ref().is_some_and(|blocked| blocked.contains_host(&host))
    }
}

/// Parse one allow-list or block-list entry into a normalized host.
fn parse_host(entry: &str) -> Result<Host<String>, InvalidDomainEntry> {
    let entry = entry.trim();
    let entry = strip_prefix_ignore_ascii_case(entry, "https://")
        .or_else(|| strip_prefix_ignore_ascii_case(entry, "http://"))
        .unwrap_or(entry);
    let entry = entry.strip_suffix('/').unwrap_or(entry);
    match Host::parse(entry).map_err(|_parse_error| InvalidDomainEntry)? {
        Host::Domain(mut domain) => {
            // IDNA conversion and percent-decoding happen during parsing, so
            // the root dot can only be removed afterwards.
            let normalized = normalize_domain(&domain).ok_or(InvalidDomainEntry)?;
            domain.truncate(normalized.len());
            Ok(Host::Domain(domain))
        },
        host @ (Host::Ipv4(_) | Host::Ipv6(_)) => Ok(host),
    }
}

/// Normalize a parsed result host for comparison with list entries.
fn normalized_host(host: Host<&str>) -> Option<Host<&str>> {
    match host {
        Host::Domain(domain) => normalize_domain(domain).map(Host::Domain),
        host @ (Host::Ipv4(_) | Host::Ipv6(_)) => Some(host),
    }
}

/// Remove the root label from a parsed domain, rejecting spellings that no
/// result host can enforce.
///
/// Parsing happens first because it performs IDNA conversion and decodes
/// percent-encoded dots and wildcards. Only one trailing root dot is valid; an
/// empty label, a leftover trailing dot, or a wildcard is not a domain this
/// matcher can compare against a URL host.
fn normalize_domain(domain: &str) -> Option<&str> {
    let domain = domain.strip_suffix('.').unwrap_or(domain);
    if domain.is_empty() || domain.contains('*') || domain.split('.').any(str::is_empty) {
        return None;
    }
    Some(domain)
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
        DomainList::parse(entries.iter().copied())
            .unwrap()
            .expect("non-empty list")
    }

    fn permitted(allowed: Option<&[&str]>, blocked: Option<&[&str]>, url: &str) -> bool {
        DomainFilters::new(allowed.map(list), blocked.map(list)).permits_url(url)
    }

    #[test]
    fn permits_exact_host_and_subdomains() {
        let permits = |url: &str| permitted(Some(&["example.org"]), None, url);
        assert!(permits("https://example.org/a"), "exact host is allowed");
        assert!(permits("https://docs.example.org/a"), "subdomain is allowed");
        assert!(permits("http://a.b.example.org"), "nested subdomain is allowed");
        assert!(permits("https://example.org./a"), "root-dot host is allowed");
    }

    #[test]
    fn rejects_lookalike_and_unrelated_hosts() {
        let permits = |url: &str| permitted(Some(&["example.org"]), None, url);
        assert!(!permits("https://outside.example/result"), "unrelated host");
        assert!(!permits("https://badexample.org/"), "suffix without a label boundary");
        assert!(
            !permits("https://example.org.evil.test/"),
            "allowed domain as a left-hand label"
        );
        assert!(!permits("https://user@evil.test/example.org"), "path is not the host");
        assert!(!permits("not a url"), "unparseable URL");
        assert!(!permits("mailto:a@example.org"), "URL without a host");
    }

    #[test]
    fn normalizes_entries_like_result_hosts() {
        let entries = [" HTTPS://Example.ORG/ ", "bücher.example."];
        assert!(
            permitted(Some(&entries), None, "https://www.example.org/"),
            "case and scheme are normalized"
        );
        assert!(
            permitted(Some(&entries), None, "https://xn--bcher-kva.example/"),
            "IDN entry matches its punycode host"
        );
        let domains = list(&entries);
        assert_eq!(
            domains,
            list(&["bücher.example", "example.org", "example.org"]),
            "equivalent lists compare equal regardless of order or duplicates"
        );
    }

    #[test]
    fn ip_hosts_match_exactly() {
        let entries = ["192.0.2.1", "[2001:db8::1]"];
        assert!(permitted(Some(&entries), None, "http://192.0.2.1/x"), "exact IPv4");
        assert!(permitted(Some(&entries), None, "http://[2001:db8::1]/x"), "exact IPv6");
        assert!(!permitted(Some(&entries), None, "http://192.0.2.2/x"), "different IPv4");
    }

    #[test]
    fn empty_list_is_unrestricted_and_deny_all_permits_nothing() {
        assert!(
            DomainList::parse(std::iter::empty::<&str>()).unwrap().is_none(),
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
    fn rejects_malformed_and_wildcard_entries() {
        for entry in [
            "",
            " ",
            "/",
            "example.org/path",
            "example.org:443",
            "a b.example",
            "user@example.org",
            // A wildcard stored literally would never match a URL host, so it
            // must be rejected instead of silently enforcing nothing. Parsing
            // decodes percent escapes, so the encoded spelling must not slip
            // past the raw check.
            "*.example.org",
            "example.*",
            "*",
            "%2a.example.org",
            "%2A.example.org",
            // Only one root-label dot is valid; empty labels cannot match.
            "example.org..",
            "example..org",
            ".",
            "..",
        ] {
            assert!(
                DomainList::parse(std::iter::once(entry)).is_err(),
                "entry {entry:?} must be rejected"
            );
        }
    }

    #[test]
    fn normalizes_root_dots_after_idna_conversion() {
        // `Host::parse` decodes percent-encoded dots, so the root dot has to be
        // removed after parsing or an escaped entry stores `example.org.` and
        // matches nothing.
        assert!(
            permitted(Some(&["example.org%2e"]), None, "https://example.org/"),
            "encoded root dot matches the root domain"
        );
        assert!(
            permitted(Some(&["example.org%2e"]), None, "https://docs.example.org/"),
            "encoded root dot matches subdomains"
        );
        assert!(
            !permitted(None, Some(&["example.org%2e"]), "https://example.org/"),
            "an encoded root dot in the block-list still blocks the root domain"
        );
        assert!(
            !permitted(None, Some(&["example.org%2e"]), "https://docs.example.org/"),
            "an encoded root dot in the block-list still blocks subdomains"
        );
        assert_eq!(
            list(&["example.org%2e"]),
            list(&["example.org"]),
            "the encoded spelling normalizes to the same list"
        );
        assert_eq!(
            list(&["example.org."]),
            list(&["example.org"]),
            "a single trailing root dot is valid"
        );
        assert!(
            DomainList::parse(std::iter::once("example.org..")).is_err(),
            "a repeated trailing root dot leaves an empty label"
        );
    }

    #[test]
    fn restricted_filters_drop_unverifiable_hosts() {
        // A block-list-only request must not retain a URL whose host cannot be
        // verified: an unparseable, scheme-relative, or opaque-scheme provider
        // URL would otherwise pass the "not blocked" check.
        for url in [
            "not a url",
            "//example.org/result",
            "/relative/path",
            "mailto:a@example.org",
            "https://",
            // A non-web scheme keeps the raw host case, so it cannot be compared
            // against a normalized entry.
            "custom://EXAMPLE.ORG/result",
            "custom://example.org/result",
            "ftp://example.org/result",
            // A repeated root dot or wildcard cannot identify a result host.
            "https://example.org../result",
            "https://*.example.org/result",
            "https://./result",
        ] {
            assert!(
                !permitted(None, Some(&["example.org"]), url),
                "a block-list-only restriction must drop unverifiable URL {url:?}"
            );
            assert!(
                !permitted(Some(&["example.org"]), None, url),
                "an allow-list restriction must drop unverifiable URL {url:?}"
            );
        }
        assert!(
            permitted(None, Some(&["example.org"]), "https://outside.example/"),
            "a verifiable URL outside the block-list is permitted"
        );
        assert!(
            DomainFilters::Unrestricted.permits_url("not a url"),
            "an unrestricted request passes provider results through unchanged"
        );
    }
}
