// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared tenant identity resolution for AI filters.
//!
//! Resolves identity from the highest-trust source available and marks
//! `{prefix}*` request headers for removal so tenant claims never leak
//! to an upstream provider.

use praxis_filter::HttpFilterContext;

/// Default header prefix for tenant identity headers.
pub(crate) const DEFAULT_IDENTITY_HEADER_PREFIX: &str = "x-tenant-";

/// Default metadata namespace the `identity_header_guard` filter writes
/// captured identity headers under.
pub(crate) const DEFAULT_IDENTITY_METADATA_NAMESPACE: &str = "identity";

/// Tenant identity resolved from metadata or request headers.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct TenantIdentity {
    /// Value of `{prefix}group`.
    pub group: String,

    /// Value of `{prefix}model`.
    pub model: String,

    /// Value of `{prefix}subscription`.
    pub subscription: String,

    /// Value of `{prefix}username`.
    pub username: String,
}

impl TenantIdentity {
    /// Whether every identity field is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.username.is_empty() && self.group.is_empty() && self.subscription.is_empty() && self.model.is_empty()
    }
}

/// Resolve tenant identity from the highest-trust source available.
///
/// Three tiers, most trusted first:
///
/// 1. Unnamespaced `{prefix}*` metadata keys, written by an authentication
///    filter from verified credentials (e.g. JWT claims or a validated API
///    key). When any of these are present, every lower tier is ignored
///    entirely so a client cannot spoof the remaining fields via forged
///    headers alongside valid credentials.
/// 2. Namespaced `{namespace}.{prefix}*` metadata keys, written by the
///    `identity_header_guard` filter from captured request headers.
/// 3. Raw `{prefix}*` request headers, set by a trusted upstream auth
///    layer when neither metadata tier is populated.
///
/// Identity headers are always marked for removal so tenant identity
/// never leaks to the upstream provider, regardless of which tier
/// supplied the identity.
///
/// `prefix` must already be lowercase.
pub(crate) fn resolve_tenant_identity(
    ctx: &mut HttpFilterContext<'_>,
    prefix: &str,
    identity_namespace: &str,
) -> TenantIdentity {
    let mut identity = resolve_tenant_identity_from_metadata(ctx, prefix, identity_namespace);

    if identity.is_empty() {
        read_header_identity(ctx, prefix, &mut identity);
    } else {
        strip_identity_headers(ctx, prefix);
    }

    identity
}

/// Resolve tenant identity from metadata without consulting raw request headers.
///
/// This is the safe lookup path for observability filters: verified identity
/// metadata wins over guard-captured metadata, and a partial verified identity
/// cannot be extended from the lower-trust namespace.
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
            "group" if identity.group.is_empty() => val.clone_into(&mut identity.group),
            "model" if identity.model.is_empty() => val.clone_into(&mut identity.model),
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
    for (key, _) in &ctx.request.headers {
        if key.as_str().starts_with(prefix_lower) {
            ctx.request_headers_to_remove.push(key.clone());
        }
    }
}
