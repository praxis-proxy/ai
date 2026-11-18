// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared tenant identity resolution for AI filters.
//!
//! Resolves identity from the highest-trust source available and marks
//! `{prefix}*` request headers for removal so tenant claims never leak
//! to an upstream provider.

use std::collections::BTreeMap;

use praxis_filter::{AuthenticatedIdentity, HttpFilterContext};

/// Default header prefix for tenant identity headers.
pub(crate) const DEFAULT_IDENTITY_HEADER_PREFIX: &str = "x-tenant-";

/// Default metadata namespace the `identity_header_guard` filter writes
/// captured identity headers under.
pub(crate) const DEFAULT_IDENTITY_METADATA_NAMESPACE: &str = "identity";

/// Tenant identity resolved from metadata or request headers.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct TenantIdentity {
    /// Value of `{prefix}cost-center`.
    pub cost_center: String,

    /// Value of `{prefix}group`.
    pub group: String,

    /// Value of `{prefix}model`.
    pub model: String,

    /// Value of `{prefix}organization-id`.
    pub organization_id: String,

    /// Value of `{prefix}subscription`.
    pub subscription: String,

    /// Value of `{prefix}username`.
    pub username: String,
}

impl TenantIdentity {
    /// Whether every identity field is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.username.is_empty()
            && self.group.is_empty()
            && self.subscription.is_empty()
            && self.model.is_empty()
            && self.organization_id.is_empty()
            && self.cost_center.is_empty()
    }
}

/// Resolve tenant identity from the highest-trust source available.
///
/// Four tiers, most trusted first:
///
/// 0. An [`AuthenticatedIdentity`] extension published by a policy/auth filter (e.g. the `policy` engine's
///    `identity/jwt` plugin) from a cryptographically verified token. This is the strongest source — its claims never
///    traverse a forgeable request header — so when present it fully determines identity and every lower tier is
///    skipped. The username comes from `username_claim` (else the verified subject id) and the group from
///    `group_claim`; see [`authenticated_identity_fields`].
/// 1. Unnamespaced `{prefix}*` metadata keys, written by an authentication filter from verified credentials (e.g. JWT
///    claims or a validated API key). When any of these are present, every lower tier is ignored entirely so a client
///    cannot spoof the remaining fields via forged headers alongside valid credentials.
/// 2. Namespaced `{namespace}.{prefix}*` metadata keys, written by the `identity_header_guard` filter from captured
///    request headers.
/// 3. Raw `{prefix}*` request headers, set by a trusted upstream auth layer when neither metadata tier is populated.
///
/// Identity headers are always marked for removal so tenant identity
/// never leaks to the upstream provider, regardless of which tier
/// supplied the identity.
///
/// `username_claim` / `group_claim` name the custom claims to read at tier 0;
/// they are ignored by every other tier. `prefix` must already be lowercase.
pub(crate) fn resolve_tenant_identity(
    ctx: &mut HttpFilterContext<'_>,
    prefix: &str,
    identity_namespace: &str,
    username_claim: Option<&str>,
    group_claim: Option<&str>,
) -> TenantIdentity {
    // Tier 0: a policy/auth filter's cryptographically verified subject. When
    // present it fully determines identity; every lower (forgeable) tier is
    // skipped, and identity headers are still stripped so nothing leaks upstream.
    if let Some(identity) = resolve_authenticated_identity(ctx, username_claim, group_claim) {
        strip_identity_headers(ctx, prefix);
        return identity;
    }

    let mut identity = resolve_tenant_identity_from_metadata(ctx, prefix, identity_namespace);

    if identity.is_empty() {
        read_header_identity(ctx, prefix, &mut identity);
    } else {
        strip_identity_headers(ctx, prefix);
    }

    identity
}

/// Tier 0: resolve identity from an [`AuthenticatedIdentity`] extension
/// published by a policy/auth filter from a verified token.
///
/// Returns `None` when no such extension is present, so the caller falls
/// through to the lower, forgeable tiers. When present, the returned identity
/// carries only the username and group ([`authenticated_identity_fields`]); the
/// remaining fields stay empty, as a verified token is not expected to carry
/// billing metadata and the metering state backfills the group from the
/// subscription.
fn resolve_authenticated_identity(
    ctx: &HttpFilterContext<'_>,
    username_claim: Option<&str>,
    group_claim: Option<&str>,
) -> Option<TenantIdentity> {
    let auth = ctx.extensions.get::<AuthenticatedIdentity>()?;
    Some(authenticated_identity_fields(
        auth.subject_id(),
        auth.custom_claims(),
        username_claim,
        group_claim,
    ))
}

/// Select the metered username and group from a verified identity's subject id
/// and custom claims.
///
/// The username is the `username_claim` custom claim when set and non-empty,
/// otherwise the verified subject id (which [`AuthenticatedIdentity`] guarantees
/// is non-empty). The group is the `group_claim` custom claim when set and
/// non-empty, otherwise empty.
///
/// Pure over its inputs so the claim-selection policy is unit testable without
/// constructing an [`AuthenticatedIdentity`], whose constructor is private to
/// Praxis core.
fn authenticated_identity_fields(
    subject_id: &str,
    custom_claims: &BTreeMap<String, String>,
    username_claim: Option<&str>,
    group_claim: Option<&str>,
) -> TenantIdentity {
    let claim = |name: Option<&str>| {
        name.and_then(|name| custom_claims.get(name))
            .filter(|value| !value.is_empty())
            .cloned()
    };

    TenantIdentity {
        username: claim(username_claim).unwrap_or_else(|| subject_id.to_owned()),
        group: claim(group_claim).unwrap_or_default(),
        ..TenantIdentity::default()
    }
}

/// Resolve tenant identity from metadata without consulting raw request headers.
///
/// Verified identity metadata wins over guard-captured metadata, and a partial
/// verified identity cannot be extended from the lower-trust namespace.
pub(crate) fn resolve_tenant_identity_from_metadata(
    ctx: &HttpFilterContext<'_>,
    prefix: &str,
    identity_namespace: &str,
) -> TenantIdentity {
    let mut identity = TenantIdentity::default();
    read_metadata_identity(ctx, prefix, "", &mut identity);

    // Any verified field blocks the lower tiers entirely: an auth filter
    // may map only some claims (e.g. group without username), and a
    // partially verified identity must not be extended by forgeable
    // sources.
    let has_verified_identity = !identity.is_empty();

    if !has_verified_identity {
        read_metadata_identity(ctx, prefix, &format!("{identity_namespace}."), &mut identity);
    }
    identity
}

/// Read the `{namespace}{prefix}*` identity keys from `filter_metadata`,
/// overwriting only the fields the namespace carries.
fn read_metadata_identity(
    ctx: &HttpFilterContext<'_>,
    prefix_lower: &str,
    namespace: &str,
    identity: &mut TenantIdentity,
) {
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}group")) {
        identity.group.clone_from(val);
    }
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}model")) {
        identity.model.clone_from(val);
    }
    if let Some(val) = ctx
        .filter_metadata
        .get(&format!("{namespace}{prefix_lower}cost-center"))
    {
        identity.cost_center.clone_from(val);
    }
    if let Some(val) = ctx
        .filter_metadata
        .get(&format!("{namespace}{prefix_lower}organization-id"))
    {
        identity.organization_id.clone_from(val);
    }
    if let Some(val) = ctx
        .filter_metadata
        .get(&format!("{namespace}{prefix_lower}subscription"))
    {
        identity.subscription.clone_from(val);
    }
    if let Some(val) = ctx.filter_metadata.get(&format!("{namespace}{prefix_lower}username")) {
        identity.username.clone_from(val);
    }
}

/// Read identity from raw `{prefix}*` request headers, marking each one
/// for removal.
///
/// [`http::header::HeaderName::as_str`] already returns the lowercased
/// name, so the pre-lowercased prefix compares directly without
/// per-header allocation.
fn read_header_identity(ctx: &mut HttpFilterContext<'_>, prefix_lower: &str, identity: &mut TenantIdentity) {
    for (key, value) in &ctx.request.headers {
        let Some(suffix) = key.as_str().strip_prefix(prefix_lower) else {
            continue;
        };
        let val = value.to_str().unwrap_or_default();

        match suffix {
            "cost-center" if identity.cost_center.is_empty() => val.clone_into(&mut identity.cost_center),
            "group" if identity.group.is_empty() => val.clone_into(&mut identity.group),
            "model" if identity.model.is_empty() => val.clone_into(&mut identity.model),
            "organization-id" if identity.organization_id.is_empty() => val.clone_into(&mut identity.organization_id),
            "subscription" if identity.subscription.is_empty() => val.clone_into(&mut identity.subscription),
            "username" if identity.username.is_empty() => val.clone_into(&mut identity.username),
            _ => {},
        }

        ctx.request_headers_to_remove.push(key.clone());
    }
}

/// Mark every `{prefix}*` header for removal without reading it, so
/// unused identity headers still never reach the upstream provider.
fn strip_identity_headers(ctx: &mut HttpFilterContext<'_>, prefix_lower: &str) {
    // This scan is required even after trusted metadata resolves identity:
    // otherwise a spoofed lower-trust header with the same prefix would be
    // forwarded upstream. HeaderMap has no prefix index, so the O(n) pass is
    // the security boundary; n is the request's bounded header count.
    for (key, _) in &ctx.request.headers {
        if key.as_str().starts_with(prefix_lower) {
            ctx.request_headers_to_remove.push(key.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claims(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect()
    }

    #[test]
    fn username_falls_back_to_subject_id_when_no_claim_configured() {
        let identity =
            authenticated_identity_fields("user-123", &claims(&[("preferred_username", "alice")]), None, None);

        assert_eq!(
            identity.username, "user-123",
            "unset username_claim meters the subject id"
        );
        assert!(identity.group.is_empty(), "unset group_claim meters no group");
    }

    #[test]
    fn username_and_group_come_from_configured_claims() {
        let identity = authenticated_identity_fields(
            "user-123",
            &claims(&[("preferred_username", "alice"), ("groups", "engineering")]),
            Some("preferred_username"),
            Some("groups"),
        );

        assert_eq!(identity.username, "alice", "the configured username claim is metered");
        assert_eq!(identity.group, "engineering", "the configured group claim is metered");
    }

    #[test]
    fn missing_or_empty_username_claim_falls_back_to_subject_id() {
        let absent = authenticated_identity_fields("user-123", &claims(&[]), Some("preferred_username"), None);
        assert_eq!(
            absent.username, "user-123",
            "an absent claim falls back to the subject id"
        );

        let empty = authenticated_identity_fields(
            "user-123",
            &claims(&[("preferred_username", "")]),
            Some("preferred_username"),
            None,
        );
        assert_eq!(
            empty.username, "user-123",
            "an empty claim falls back to the subject id"
        );
    }

    #[test]
    fn missing_or_empty_group_claim_leaves_group_empty() {
        let absent = authenticated_identity_fields("user-123", &claims(&[]), None, Some("groups"));
        assert!(absent.group.is_empty(), "an absent group claim leaves the group empty");

        let empty = authenticated_identity_fields("user-123", &claims(&[("groups", "")]), None, Some("groups"));
        assert!(empty.group.is_empty(), "an empty group claim leaves the group empty");
    }

    #[test]
    fn tier_zero_carries_no_billing_metadata() {
        let identity = authenticated_identity_fields(
            "user-123",
            &claims(&[("subscription", "sub-1"), ("organization_id", "org-1")]),
            None,
            None,
        );

        assert!(
            identity.subscription.is_empty()
                && identity.organization_id.is_empty()
                && identity.cost_center.is_empty()
                && identity.model.is_empty(),
            "only username and group are read from a verified identity",
        );
    }
}
