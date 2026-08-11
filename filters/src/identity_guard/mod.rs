// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Identity header guard filter: captures request headers matching
//! a configured prefix into `filter_metadata` and strips them from
//! the upstream request.
//!
//! Prevents identity headers injected by a trusted auth layer
//! (e.g. `x-tenant-username`) from leaking to upstream LLM
//! providers, while making them available to downstream filters
//! (metering, audit) via metadata.
//!
//! Maps to IPP's `maas-headers-guard` plugin but is generic:
//! the prefix is configurable rather than hardcoded to `x-maas-`.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests;

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use tracing::trace;

use self::config::{IdentityHeaderGuardConfig, validate_config};

// -----------------------------------------------------------------------------
// IdentityHeaderGuardFilter
// -----------------------------------------------------------------------------

/// Captures request headers matching a configured prefix into
/// `filter_metadata` and removes them from the upstream request.
///
/// A client that sets `x-tenant-username: admin` directly is
/// indistinguishable from a gateway that set it legitimately
/// unless this filter strips the headers first. Place it early
/// in the pipeline — before any filter that reads identity from
/// request headers.
///
/// # YAML configuration
///
/// ```yaml
/// filter: identity_header_guard
/// prefix: "x-tenant-"
/// metadata_namespace: "identity"
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_filters::IdentityHeaderGuardFilter;
///
/// let yaml: serde_yaml::Value = serde_yaml::from_str(r#"prefix: "x-tenant-""#).unwrap();
/// let filter = IdentityHeaderGuardFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "identity_header_guard");
/// ```
pub struct IdentityHeaderGuardFilter {
    /// Lowercase prefix to match against header names.
    prefix: String,

    /// Metadata key namespace for captured headers.
    namespace: String,
}

impl IdentityHeaderGuardFilter {
    /// Parse from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: IdentityHeaderGuardConfig = parse_filter_config("identity_header_guard", value)?;
        validate_config(&config).map_err(|e| -> FilterError { e.into() })?;

        Ok(Box::new(Self {
            prefix: config.prefix.to_lowercase(),
            namespace: config.metadata_namespace,
        }))
    }
}

#[async_trait]
impl HttpFilter for IdentityHeaderGuardFilter {
    fn name(&self) -> &'static str {
        "identity_header_guard"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let mut captured = 0_usize;

        for (name, value) in &ctx.request.headers {
            let name_lower = name.as_str().to_lowercase();

            if !name_lower.starts_with(&self.prefix) {
                continue;
            }

            if let Ok(val) = value.to_str() {
                // Namespaced key only. The guard must NOT write
                // unnamespaced keys — jwt_auth writes those from
                // verified claims, and overwriting them here would
                // launder client-spoofed headers into the trusted
                // metadata namespace.
                let namespaced = format!("{}.{}", self.namespace, name_lower);
                ctx.set_metadata(namespaced, val.to_owned());
                captured += 1;
            }

            ctx.request_headers_to_remove.push(name.clone());
        }

        if captured > 0 {
            trace!(captured, prefix = %self.prefix, "identity headers captured and stripped");
        }

        Ok(FilterAction::Continue)
    }
}
