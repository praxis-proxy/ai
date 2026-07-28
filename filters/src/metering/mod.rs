// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! External metering filter: tenant identity handling for metered inference.
//!
//! Removes the tenant identity headers and the client credentials from a
//! request before it reaches the upstream provider. Tenant headers are trusted
//! input from an authenticating layer in front of the proxy, so they must never
//! be forwarded: an upstream that echoes or logs them would leak tenant
//! attribution, and a client that sets them itself must not be able to
//! impersonate a tenant.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests;

use async_trait::async_trait;
use http::header::HeaderName;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};

use self::config::{ExternalMeteringConfig, validate_config};

// -----------------------------------------------------------------------------
// ExternalMeteringFilter
// -----------------------------------------------------------------------------

/// Strips tenant identity headers and client credentials from metered
/// inference requests before they reach the upstream provider.
///
/// # YAML
///
/// ```yaml
/// filter: external_metering
/// identity_header_prefix: "x-tenant-"
/// ```
pub struct ExternalMeteringFilter {
    /// Prefix of the tenant identity headers to strip.
    identity_header_prefix: String,
}

impl ExternalMeteringFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config)?))
    }

    /// Build the concrete filter from parsed YAML config.
    fn build(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: ExternalMeteringConfig = parse_filter_config("external_metering", config)?;
        validate_config(&cfg)?;

        Ok(Self {
            identity_header_prefix: cfg.identity_header_prefix,
        })
    }
}

#[async_trait]
impl HttpFilter for ExternalMeteringFilter {
    fn name(&self) -> &'static str {
        "external_metering"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        strip_identity_headers(ctx, &self.identity_header_prefix);
        strip_client_credentials(ctx);

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Header Removal
// -----------------------------------------------------------------------------

/// Mark every header carrying the tenant identity prefix for removal.
fn strip_identity_headers(ctx: &mut HttpFilterContext<'_>, prefix: &str) {
    let prefix_lower = prefix.to_ascii_lowercase();

    for key in ctx.request.headers.keys() {
        if key.as_str().to_ascii_lowercase().starts_with(prefix_lower.as_str()) {
            ctx.request_headers_to_remove.push(key.clone());
        }
    }
}

/// Mark client-supplied credentials for removal.
///
/// The proxy authenticates to the provider with its own credentials, so a
/// client-supplied key is never useful upstream and forwarding one would let a
/// client bill an account the gateway does not control.
fn strip_client_credentials(ctx: &mut HttpFilterContext<'_>) {
    ctx.request_headers_to_remove.push(http::header::AUTHORIZATION);

    if let Ok(name) = "x-api-key".parse::<HeaderName>() {
        ctx.request_headers_to_remove.push(name);
    }
}
