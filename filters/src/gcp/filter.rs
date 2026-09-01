// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`GcpAdcFilter`] implementation and `HttpFilter` trait impl.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use arc_swap::ArcSwap;
use http::header;
use praxis_filter::{FilterError, parse_filter_config};
use tracing::warn;

use super::{
    config::{GcpAdcConfig, validate_config},
    token::resolve_token_source,
};

// -----------------------------------------------------------------------------
// Cached token
// -----------------------------------------------------------------------------

/// A bearer token cached in memory, pre-formatted for injection.
///
/// The only way to construct one is [`CachedToken::new`], which formats
/// the `Authorization` value and marks it sensitive, so a token can
/// never reach the cache unredacted (`HeaderValue`'s `Debug` output
/// redacts sensitive values).
#[derive(Debug)]
pub(super) struct CachedToken {
    /// The complete `Authorization` header value (`"Bearer <token>"`),
    /// marked sensitive so it is redacted from header debug output.
    authorization: http::HeaderValue,

    /// Instant after which the token must not be used. The producer is
    /// responsible for subtracting a safety skew from the real expiry.
    expires_at: Instant,
}

impl CachedToken {
    /// Build a cached token from a raw access token, formatting the
    /// `Authorization` value and marking it sensitive.
    ///
    /// Currently only exercised by tests; token acquisition will call
    /// this once fetch is implemented on top of the core background-task
    /// primitive (praxis#1043).
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the token is not a valid header value.
    #[cfg(test)]
    pub(super) fn new(access_token: &str, expires_at: Instant) -> Result<Self, FilterError> {
        let mut authorization = http::HeaderValue::from_str(&format!("Bearer {access_token}"))
            .map_err(|e| FilterError::from(format!("gcp_adc: token is not a valid header value: {e}")))?;
        authorization.set_sensitive(true);
        Ok(Self {
            authorization,
            expires_at,
        })
    }

    /// Whether the token is still safe to inject at `now`.
    pub(super) fn is_valid(&self, now: Instant) -> bool {
        now < self.expires_at
    }
}

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
/// **Token acquisition is not implemented yet.** This filter currently
/// establishes the configuration surface, credential-source resolution,
/// and the fail-closed request path: the token cache is never populated,
/// so every request is rejected with `503`. Fetch (metadata server, key
/// file) arrives with the shared background-refresh primitive tracked in
/// praxis#555, praxis#1042, and praxis#1043 — filters must not spawn
/// their own refresher threads.
///
/// Once implemented, the filter will acquire a token via Application
/// Default Credentials (GKE metadata or a service-account key file) and
/// inject `Authorization: Bearer <token>` on every proxied request,
/// keeping GCP credentials invisible to the downstream client.
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
/// Until a token is cached, and whenever the cached token is missing
/// or expired, requests are rejected with `503` rather than forwarded
/// unauthenticated.
///
/// # YAML configuration
///
/// ```yaml
/// filter: gcp_adc
/// source: adc
/// scope: https://www.googleapis.com/auth/cloud-platform
/// refresh_ratio: 0.75
/// ```
pub struct GcpAdcFilter {
    /// Lock-free cache of the current token, read on every request.
    /// Nothing populates it until token fetch is implemented on top of
    /// the core background-task primitive (praxis#1043).
    token: Arc<ArcSwap<Option<CachedToken>>>,

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
    /// Returns [`FilterError`] if `refresh_ratio` is out of range,
    /// `service_account` is structurally unsafe, a field is set that its
    /// `source` does not use, or ADC file resolution fails.
    fn new(config: &GcpAdcConfig, application_credentials: Option<&std::path::Path>) -> Result<Self, FilterError> {
        validate_config(config)?;
        // Resolution validates the credential source (file readable,
        // supported `type`) at the config boundary; the resolved source
        // is not stored until token fetch is implemented.
        resolve_token_source(config, application_credentials)?;
        Ok(Self {
            token: Arc::new(ArcSwap::from_pointee(None)),
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

    /// Build a filter around a given cache.
    #[cfg(test)]
    pub(super) fn for_test(token: Option<CachedToken>) -> Self {
        Self {
            token: Arc::new(ArcSwap::from_pointee(token)),
            failing: AtomicBool::new(false),
        }
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
        let cached = self.token.load();
        match cached.as_ref() {
            Some(token) if token.is_valid(Instant::now()) => {
                if self.failing.swap(false, Ordering::Relaxed) {
                    warn!("gcp_adc: valid token available again; resuming request forwarding");
                }
                ctx.request_headers_to_set
                    .push((header::AUTHORIZATION, token.authorization.clone()));
                Ok(praxis_filter::FilterAction::Continue)
            },
            _ => {
                if !self.failing.swap(true, Ordering::Relaxed) {
                    warn!("gcp_adc: no valid cached token; rejecting requests with 503 until one is available");
                }
                Ok(praxis_filter::FilterAction::Reject(praxis_filter::Rejection::status(
                    503,
                )))
            },
        }
    }
}
