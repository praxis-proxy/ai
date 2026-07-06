// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! MCP protocol version and profile types.
//!
//! Centralizes the protocol version string so it is not scattered through
//! handlers and tests. Future MCP spec versions can be added to
//! [`SUPPORTED_VERSIONS_CURRENT`] without modifying individual request handlers.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Protocol version implemented by the current broker behavior.
pub(crate) const PROTOCOL_VERSION_CURRENT: &str = "2025-03-26";

/// Protocol version for the MCP 2026-07-28 stateless profile (release candidate).
pub(crate) const PROTOCOL_VERSION_STATELESS_2026_07_28: &str = "2026-07-28";

/// Protocol versions supported by the current profile.
pub(crate) const SUPPORTED_VERSIONS_CURRENT: &[&str] = &[PROTOCOL_VERSION_CURRENT];

/// Protocol versions supported by the stateless profile.
pub(crate) const SUPPORTED_VERSIONS_STATELESS: &[&str] = &[PROTOCOL_VERSION_STATELESS_2026_07_28];

/// All protocol versions this build of Praxis can handle across all profiles.
pub(crate) const SUPPORTED_VERSIONS: &[&str] = &[PROTOCOL_VERSION_CURRENT, PROTOCOL_VERSION_STATELESS_2026_07_28];

// -----------------------------------------------------------------------------
// ProtocolProfile
// -----------------------------------------------------------------------------

/// MCP protocol profile governing session semantics and header requirements.
///
/// The `Current` profile preserves the existing `initialize`/session behavior.
/// The `Stateless` profile implements MCP 2026-07-28 stateless semantics:
/// no protocol sessions, required request metadata headers, and
/// `server/discover` instead of `initialize`.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProtocolProfile {
    /// Current MCP Streamable HTTP behavior: `initialize` handshake,
    /// optional `MCP-Session-Id`, and session-aware DELETE.
    #[default]
    Current,
    /// MCP 2026-07-28 stateless behavior: no protocol sessions,
    /// required `MCP-Protocol-Version` / `Mcp-Method` / `Mcp-Name` headers,
    /// and `server/discover` for capability discovery.
    Stateless,
}

impl ProtocolProfile {
    /// String label for logging and metadata.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Stateless => "stateless",
        }
    }
}

// -----------------------------------------------------------------------------
// Profile Helpers
// -----------------------------------------------------------------------------

/// Returns the default protocol version for the given profile.
pub(crate) fn default_version_for_profile(profile: ProtocolProfile) -> &'static str {
    match profile {
        ProtocolProfile::Current => PROTOCOL_VERSION_CURRENT,
        ProtocolProfile::Stateless => PROTOCOL_VERSION_STATELESS_2026_07_28,
    }
}

/// Returns the supported protocol versions for the given profile.
pub(crate) fn supported_versions_for_profile(profile: ProtocolProfile) -> &'static [&'static str] {
    match profile {
        ProtocolProfile::Current => SUPPORTED_VERSIONS_CURRENT,
        ProtocolProfile::Stateless => SUPPORTED_VERSIONS_STATELESS,
    }
}

/// Returns `true` when `version` is supported by the given profile.
pub(crate) fn is_supported_version_for_profile(profile: ProtocolProfile, version: &str) -> bool {
    supported_versions_for_profile(profile).contains(&version)
}

// -----------------------------------------------------------------------------
// Global Helpers
// -----------------------------------------------------------------------------

/// Returns `true` when `version` appears in [`SUPPORTED_VERSIONS`].
pub(crate) fn is_supported_version(version: &str) -> bool {
    SUPPORTED_VERSIONS.contains(&version)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    #[test]
    fn default_version_for_each_profile_is_supported() {
        for profile in [ProtocolProfile::Current, ProtocolProfile::Stateless] {
            let version = default_version_for_profile(profile);
            assert!(
                is_supported_version(version),
                "default version for {profile:?} must appear in SUPPORTED_VERSIONS"
            );
        }
    }

    #[test]
    fn current_version_is_supported() {
        assert!(
            is_supported_version(PROTOCOL_VERSION_CURRENT),
            "PROTOCOL_VERSION_CURRENT must be supported"
        );
    }

    #[test]
    fn stateless_version_is_supported() {
        assert!(
            is_supported_version(PROTOCOL_VERSION_STATELESS_2026_07_28),
            "PROTOCOL_VERSION_STATELESS_2026_07_28 must be supported"
        );
    }

    #[test]
    fn unknown_version_is_not_supported() {
        assert!(
            !is_supported_version("9999-12-31"),
            "arbitrary version should not be supported"
        );
    }

    #[test]
    fn default_profile_is_current() {
        assert_eq!(
            ProtocolProfile::default(),
            ProtocolProfile::Current,
            "default profile should be Current"
        );
    }

    #[test]
    fn profile_as_str_round_trips() {
        assert_eq!(ProtocolProfile::Current.as_str(), "current");
        assert_eq!(ProtocolProfile::Stateless.as_str(), "stateless");
    }

    #[test]
    fn profile_deserializes_from_yaml() {
        let profile: ProtocolProfile = serde_yaml::from_str("current").unwrap();
        assert_eq!(profile, ProtocolProfile::Current, "should parse 'current'");
    }

    #[test]
    fn stateless_profile_parses_from_yaml() {
        let profile: ProtocolProfile = serde_yaml::from_str("stateless").unwrap();
        assert_eq!(profile, ProtocolProfile::Stateless, "should parse 'stateless'");
    }

    #[test]
    fn profile_rejects_unknown_value() {
        let result = serde_yaml::from_str::<ProtocolProfile>("unknown");
        assert!(result.is_err(), "unknown profile value should fail to parse");
    }

    #[test]
    fn default_profile_remains_current() {
        assert_eq!(
            ProtocolProfile::default(),
            ProtocolProfile::Current,
            "default ProtocolProfile must remain Current for backward compatibility"
        );
    }

    #[test]
    fn stateless_profile_defaults_to_2026_07_28() {
        assert_eq!(
            default_version_for_profile(ProtocolProfile::Stateless),
            PROTOCOL_VERSION_STATELESS_2026_07_28,
            "stateless profile default version should be 2026-07-28"
        );
    }

    #[test]
    fn current_profile_defaults_to_current() {
        assert_eq!(
            default_version_for_profile(ProtocolProfile::Current),
            PROTOCOL_VERSION_CURRENT,
            "current profile default version should be 2025-03-26"
        );
    }

    #[test]
    fn supported_versions_for_current_profile() {
        let versions = supported_versions_for_profile(ProtocolProfile::Current);
        assert!(
            versions.contains(&PROTOCOL_VERSION_CURRENT),
            "current profile should support 2025-03-26"
        );
        assert!(
            !versions.contains(&PROTOCOL_VERSION_STATELESS_2026_07_28),
            "current profile should not support 2026-07-28"
        );
    }

    #[test]
    fn supported_versions_for_stateless_profile() {
        let versions = supported_versions_for_profile(ProtocolProfile::Stateless);
        assert!(
            versions.contains(&PROTOCOL_VERSION_STATELESS_2026_07_28),
            "stateless profile should support 2026-07-28"
        );
        assert!(
            !versions.contains(&PROTOCOL_VERSION_CURRENT),
            "stateless profile should not support 2025-03-26"
        );
    }

    #[test]
    fn is_supported_version_for_current_profile() {
        assert!(
            is_supported_version_for_profile(ProtocolProfile::Current, PROTOCOL_VERSION_CURRENT),
            "2025-03-26 should be supported by current profile"
        );
        assert!(
            !is_supported_version_for_profile(ProtocolProfile::Current, PROTOCOL_VERSION_STATELESS_2026_07_28),
            "2026-07-28 should not be supported by current profile"
        );
    }

    #[test]
    fn supported_versions_equals_union_of_all_profiles() {
        let mut union: Vec<&str> = Vec::new();
        union.extend_from_slice(SUPPORTED_VERSIONS_CURRENT);
        union.extend_from_slice(SUPPORTED_VERSIONS_STATELESS);
        union.sort_unstable();
        union.dedup();

        let mut global = SUPPORTED_VERSIONS.to_vec();
        global.sort_unstable();

        assert_eq!(
            global, union,
            "SUPPORTED_VERSIONS must equal the union of all per-profile version arrays"
        );

        assert_eq!(
            global.len(),
            union.len(),
            "SUPPORTED_VERSIONS must not contain duplicates or extras"
        );
    }

    #[test]
    fn is_supported_version_for_stateless_profile() {
        assert!(
            is_supported_version_for_profile(ProtocolProfile::Stateless, PROTOCOL_VERSION_STATELESS_2026_07_28),
            "2026-07-28 should be supported by stateless profile"
        );
        assert!(
            !is_supported_version_for_profile(ProtocolProfile::Stateless, PROTOCOL_VERSION_CURRENT),
            "2025-03-26 should not be supported by stateless profile"
        );
    }
}
