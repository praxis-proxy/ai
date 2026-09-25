// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`GcpAdcFilter`] implementation and `HttpFilter` trait impl.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use http::{HeaderValue, header};
use praxis_ai_apis::token_cache::TokenCache;
use praxis_filter::{FilterError, parse_filter_config};
use tracing::warn;

use super::{
    config::{GcpAdcConfig, validate_config},
    token::{self, TokenSource, resolve_token_source},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Treat a cached token as expired this long before its real expiry, so
/// a token is never injected onto a request that could outlive it in
/// flight. Passed to [`TokenCache::new`] as its safety margin.
const EXPIRY_SKEW: Duration = Duration::from_secs(30);

/// Timeout for a single metadata-server round-trip.
const TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Injects a GCP `OAuth2` access token into outbound requests.
///
/// Experimental: requires the `gcp-adc-filter` cargo feature, which is
/// off by default and activates the `experimental` marker. This filter
/// is a work in progress and its configuration surface may change
/// between releases.
///
/// Acquires a token via Application Default Credentials (GKE metadata
/// server, or a service-account key file) and injects `Authorization:
/// Bearer <token>` on every proxied request, keeping GCP credentials
/// invisible to the downstream client. There is no background refresh
/// thread: caching is cache-through, the same as
/// [`crate::azure::azure_ad`] — see
/// [`praxis_ai_apis::token_cache::TokenCache`] for the exact contract.
///
/// For `source: key_file`, the filter mints tokens itself: it signs a
/// `JWT` assertion with the key file's private key and exchanges it at
/// Google's `OAuth2` token endpoint. The `token_uri` inside the key file
/// is validated at construct time to `https://oauth2.googleapis.com`
/// (or a loopback test fixture), so a tampered key file cannot redirect
/// the signed assertion elsewhere. Key files missing `client_email`,
/// `private_key`, or `token_uri` are rejected as configuration errors.
///
/// Credential-source resolution happens at construct time:
/// `GOOGLE_APPLICATION_CREDENTIALS` is read once when the pipeline is
/// built (reload the config to pick up changes), and a `gcloud` user
/// credential file (`authorized_user`) is rejected as a configuration
/// error rather than silently falling through the ADC chain.
///
/// This filter only injects `Authorization`. Pointing the request at
/// the correct Vertex endpoint (cluster `endpoints` + `tls.sni`) is
/// the operator's responsibility.
///
/// Whenever no valid token can be produced — none cached and the inline
/// fetch fails — the request is rejected with `503` rather than
/// forwarded unauthenticated.
///
/// Metadata requests use a proxy-free, redirect-free client pinned to the
/// complete validated DNS result set. Metadata mode intentionally permits
/// the protocol-owned private endpoint; arbitrary private hosts remain
/// invalid configuration.
///
/// # YAML configuration
///
/// ```yaml
/// filter: gcp_adc
/// source: adc
/// scope: https://www.googleapis.com/auth/cloud-platform
/// ```
pub struct GcpAdcFilter {
    /// Cache-through token cache; see the struct docs and
    /// [`praxis_ai_apis::token_cache`].
    cache: TokenCache<HeaderValue>,

    /// Resolved credential source.
    source: TokenSource,

    /// Optional logical upstream cluster allowlist for credential injection.
    clusters: Vec<String>,

    /// `OAuth2` scope requested with the access token.
    scope: String,

    /// GCE/GKE metadata server host.
    metadata_host: String,

    /// Whether the filter is currently failing closed, so the missing-
    /// token condition is logged on state transitions instead of once
    /// per rejected request.
    failing: AtomicBool,
}

impl GcpAdcFilter {
    /// Build a filter from parsed config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if `service_account` or `metadata_host`
    /// are structurally unsafe, a field is set that its `source` does
    /// not use, or ADC file resolution fails.
    fn new(config: &GcpAdcConfig, application_credentials: Option<&std::path::Path>) -> Result<Self, FilterError> {
        validate_config(config)?;
        let source = resolve_token_source(config, application_credentials)?;
        Ok(Self {
            cache: TokenCache::new(EXPIRY_SKEW),
            source,
            clusters: config.clusters.clone(),
            scope: config.scope.clone(),
            metadata_host: config.metadata_host.clone(),
            failing: AtomicBool::new(false),
        })
    }

    /// Parse YAML config and build a boxed filter instance.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is malformed or credential
    /// resolution fails.
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn praxis_filter::HttpFilter>, FilterError> {
        let config: GcpAdcConfig = parse_filter_config("gcp_adc", config)?;
        Ok(Box::new(Self::new(
            &config,
            std::env::var_os("GOOGLE_APPLICATION_CREDENTIALS")
                .as_deref()
                .map(std::path::Path::new),
        )?))
    }

    /// Whether this filter should inject credentials for the selected upstream.
    fn cluster_is_in_scope(&self, cluster: Option<&str>) -> bool {
        self.clusters.is_empty() || cluster.is_some_and(|selected| self.clusters.iter().any(|name| name == selected))
    }
}

#[async_trait::async_trait]
impl praxis_filter::HttpFilter for GcpAdcFilter {
    fn name(&self) -> &'static str {
        "gcp_adc"
    }

    fn request_body_access(&self) -> praxis_filter::BodyAccess {
        praxis_filter::BodyAccess::None
    }

    fn request_body_mode(&self) -> praxis_filter::BodyMode {
        praxis_filter::BodyMode::Stream
    }

    async fn on_request(
        &self,
        ctx: &mut praxis_filter::HttpFilterContext<'_>,
    ) -> Result<praxis_filter::FilterAction, FilterError> {
        if !self.cluster_is_in_scope(ctx.cluster_name()) {
            return Ok(praxis_filter::FilterAction::Continue);
        }

        let fetched = self
            .cache
            .get_or_refresh(|| {
                token::fetch_pinned(&self.source, &self.metadata_host, &self.scope, TOKEN_REQUEST_TIMEOUT)
            })
            .await;
        match fetched {
            Ok(authorization) => {
                if self.failing.swap(false, Ordering::Relaxed) {
                    warn!("gcp_adc: valid token available again; resuming request forwarding");
                }
                ctx.request_headers_to_set.push((header::AUTHORIZATION, authorization));
                Ok(praxis_filter::FilterAction::Continue)
            },
            Err(error) => {
                if !self.failing.swap(true, Ordering::Relaxed) {
                    warn!(%error, "gcp_adc: no valid token available; rejecting requests with 503 until one is acquired");
                }
                Ok(praxis_filter::FilterAction::Reject(praxis_filter::Rejection::status(
                    503,
                )))
            },
        }
    }
}
