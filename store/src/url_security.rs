// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared IP classification for outbound HTTP clients and datastore targets.
//!
//! One canonical SSRF and private-range policy for every first-party consumer
//! (outbound file fetches, callout targets, and the Postgres URL validator).
//! SQL-free and dependency-light so the store leaf stays crypto-free.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Convert IPv4-mapped IPv6 addresses (`::ffff:A.B.C.D`) to plain IPv4.
///
/// Native IPv4 and non-mapped IPv6 addresses pass through unchanged.
fn normalize_mapped_ipv4(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(IpAddr::V6(v6), IpAddr::V4),
        v4 @ IpAddr::V4(_) => v4,
    }
}

/// Return whether an IP targets a known cloud metadata or credential endpoint.
pub fn is_cloud_metadata(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            *v4 == Ipv4Addr::new(169, 254, 169, 254)
                || *v4 == Ipv4Addr::new(169, 254, 170, 2)
                || *v4 == Ipv4Addr::new(169, 254, 170, 23)
                || *v4 == Ipv4Addr::new(169, 254, 0, 23)
                || *v4 == Ipv4Addr::new(169, 254, 10, 10)
                || *v4 == Ipv4Addr::new(100, 100, 100, 200)
        },
        IpAddr::V6(v6) => {
            const AWS_IMDS_V6: Ipv6Addr = Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0254);
            const AWS_ECS_CREDS_V6: Ipv6Addr = Ipv6Addr::new(0xFD00, 0x0EC2, 0, 0, 0, 0, 0, 0x0023);
            *v6 == AWS_IMDS_V6 || *v6 == AWS_ECS_CREDS_V6
        },
    }
}

/// Return whether an IP is unsafe even for explicitly allowlisted file URLs.
fn is_unconditionally_blocked(ip: &IpAddr) -> bool {
    ip.is_unspecified() || ip.is_multicast() || is_cloud_metadata(ip)
}

/// Return whether an IP is not publicly routable under the shared policy.
pub fn is_non_public_ip(ip: &IpAddr) -> bool {
    let ip = normalize_mapped_ipv4(*ip);
    if is_unconditionally_blocked(&ip) || is_private_or_special_use(&ip) {
        return true;
    }
    if let IpAddr::V6(v6) = ip
        && let Some(v4) = nat64_embedded_ipv4(&v6)
    {
        return is_non_public_ip(&IpAddr::V4(v4));
    }
    false
}

/// Return whether an IP must be blocked for an untrusted `file_url` fetch.
pub fn is_file_url_ssrf_blocked(ip: &IpAddr, allow_private: bool) -> bool {
    let ip = normalize_mapped_ipv4(*ip);
    if is_unconditionally_blocked(&ip) {
        return true;
    }
    if let IpAddr::V6(v6) = ip
        && let Some(v4) = nat64_embedded_ipv4(&v6)
    {
        let embedded = IpAddr::V4(v4);
        if is_unconditionally_blocked(&embedded) || (!allow_private && is_private_or_special_use(&embedded)) {
            return true;
        }
    }
    !allow_private && is_private_or_special_use(&ip)
}

/// Return whether an IP is private or belongs to a non-global special-use range.
fn is_private_or_special_use(ip: &IpAddr) -> bool {
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

/// Return whether an IPv4 address is in the shared CGNAT range (`100.64.0.0/10`).
fn is_cgnat(ip: Ipv4Addr) -> bool {
    u32::from(ip) & 0xFFC0_0000 == 0x6440_0000
}

/// Return whether an IPv6 address is in the unicast link-local range (`fe80::/10`).
fn is_unicast_link_local_v6(v6: &Ipv6Addr) -> bool {
    let [a, b, ..] = v6.octets();
    a == 0xFE && (b & 0xC0) == 0x80
}

/// Return whether an IPv6 address is in the deprecated site-local range (`fec0::/10`).
fn is_site_local_v6(v6: &Ipv6Addr) -> bool {
    let [a, b, ..] = v6.octets();
    a == 0xFE && (b & 0xC0) == 0xC0
}

/// Return whether an IPv4 address belongs to a non-global IANA special-use range.
fn is_special_use_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    let n = u32::from(ip);
    (o[0] == 192 && o[1] == 0 && o[2] == 0 && o[3] != 9 && o[3] != 10)
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)
        || (o[0] == 192 && o[1] == 88 && o[2] == 99)
        || (n & 0xFFFE_0000 == 0xC612_0000)
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)
        || o[0] >= 240
}

/// Return whether an IPv6 address belongs to a non-global IANA special-use range.
fn is_special_use_v6(v6: Ipv6Addr) -> bool {
    let s = v6.segments();
    (s[0] == 0x0064 && s[1] == 0xFF9B && s[2] == 0x0001)
        || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0)
        || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 1)
        || is_2001_ietf_non_global(s)
        || (s[0] == 0x2001 && s[1] == 0x0DB8)
        || s[0] == 0x2002
        || (s[0] == 0x3FFF && s[1] & 0xF000 == 0)
        || s[0] == 0x5F00
}

/// Return whether an IPv6 address is in a non-global sub-range of `2001::/23`.
fn is_2001_ietf_non_global(s: [u16; 8]) -> bool {
    if s[0] != 0x2001 || s[1] > 0x01FF {
        return false;
    }
    match s[1] {
        0x0001 if s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0 && s[6] == 0 && matches!(s[7], 1..=3) => false,
        0x0003 => false,
        0x0004 if s[2] == 0x0112 => false,
        v if v & 0xFFF0 == 0x0020 || v & 0xFFF0 == 0x0030 => false,
        _ => true,
    }
}

/// Extract an embedded IPv4 address from the NAT64 well-known prefix (`64:ff9b::/96`).
fn nat64_embedded_ipv4(v6: &Ipv6Addr) -> Option<Ipv4Addr> {
    let s = v6.segments();
    (s[0] == 0x0064 && s[1] == 0xFF9B && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0).then(|| {
        let [.., a, b, c, d] = v6.octets();
        Ipv4Addr::new(a, b, c, d)
    })
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::panic, reason = "tests")]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn cloud_metadata_endpoints_flagged() {
        for s in [
            "169.254.169.254",
            "169.254.170.2",
            "169.254.170.23",
            "169.254.0.23",
            "169.254.10.10",
            "100.100.100.200",
            "fd00:ec2::254",
            "fd00:ec2::23",
        ] {
            assert!(is_cloud_metadata(&ip(s)), "{s} should be cloud metadata");
            assert!(is_non_public_ip(&ip(s)), "{s} should be non-public");
            assert!(
                is_file_url_ssrf_blocked(&ip(s), true),
                "{s} blocked even when private allowed"
            );
        }
    }

    #[test]
    fn private_and_loopback_ranges_are_non_public() {
        for s in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fec0::1",
            "fc00::1",
        ] {
            assert!(is_non_public_ip(&ip(s)), "{s} should be non-public");
        }
    }

    #[test]
    fn public_addresses_are_public() {
        for s in ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888"] {
            assert!(!is_non_public_ip(&ip(s)), "{s} should be public");
            assert!(!is_file_url_ssrf_blocked(&ip(s), false), "{s} not blocked");
        }
    }

    #[test]
    fn allow_private_toggle_only_relaxes_private_ranges() {
        // Private range: allowed when allow_private, blocked otherwise.
        assert!(is_file_url_ssrf_blocked(&ip("10.0.0.1"), false));
        assert!(!is_file_url_ssrf_blocked(&ip("10.0.0.1"), true));
        // Unconditionally blocked (metadata/unspecified/multicast) ignores the toggle.
        assert!(is_file_url_ssrf_blocked(&ip("0.0.0.0"), true));
        assert!(is_file_url_ssrf_blocked(&ip("224.0.0.1"), true));
    }

    #[test]
    fn ipv4_mapped_ipv6_is_normalized() {
        assert!(is_non_public_ip(&ip("::ffff:127.0.0.1")));
        assert!(is_file_url_ssrf_blocked(&ip("::ffff:127.0.0.1"), false));
    }

    #[test]
    fn nat64_embedded_ipv4_is_reclassified() {
        // 64:ff9b::/96 embedding a metadata / private v4 is blocked.
        assert!(is_non_public_ip(&ip("64:ff9b::a9fe:a9fe"))); // 169.254.169.254
        assert!(is_non_public_ip(&ip("64:ff9b::c0a8:1"))); // 192.168.0.1
        // Embedding a public v4 stays public.
        assert!(!is_non_public_ip(&ip("64:ff9b::808:808"))); // 8.8.8.8
    }

    #[test]
    fn special_use_ranges_are_non_public() {
        for s in [
            "198.18.0.1",
            "192.0.0.1",
            "192.88.99.1",
            "192.0.2.1",
            "198.51.100.1",
            "203.0.113.1",
            "240.0.0.1",
            "2001:db8::1",
            "2002:c0a8:1::1",
            "3fff::1",
            "5f00::1",
        ] {
            assert!(is_non_public_ip(&ip(s)), "{s} should be non-public special-use");
        }
        // 192.0.0.9 and 192.0.0.10 are global exceptions inside 192.0.0.0/24.
        assert!(!is_non_public_ip(&ip("192.0.0.9")));
        assert!(!is_non_public_ip(&ip("192.0.0.10")));
    }
}
