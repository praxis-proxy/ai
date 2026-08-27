// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`AzureAdFilter`] — injects Azure AD (Entra ID) bearer tokens into
//! outbound requests to Azure OpenAI and other Azure AI services.
//!
//! **Experimental.** Requires the `azure-ad-filter` cargo feature, which
//! is off by default and activates the `experimental` marker. The
//! configuration surface may change between releases.
//!
//! # Overview
//!
//! Enterprise Azure deployments prohibit static API keys and require an
//! `OAuth2` bearer token from Entra ID. This filter acquires such a token
//! via the **client-credentials** grant, caches it, refreshes it in the
//! background before it expires, and injects `Authorization: Bearer
//! <token>` on every proxied request. The downstream client is unaware
//! of Entra ID.
//!
//! # Scope
//!
//! This filter currently supports the **client-secret** credential only.
//! Managed identity (AKS/IMDS), client certificates (`private_key_jwt`
//! assertions), and OIDC/workload-identity federation are planned
//! follow-ups; they slot into the same token-cache/refresh machinery
//! this filter already uses, with no change to request handling.
//!
//! # Routing vs. authentication
//!
//! Unlike AWS `SigV4` signing, an Entra ID bearer token is **not** bound
//! to the request `Host`, path, or body — it is a standalone credential.
//! This filter therefore does exactly one thing: inject the
//! `Authorization` header. Pointing the request at the correct Azure
//! endpoint (cluster `endpoints` + `tls.sni`, and any host rewrite) is
//! the operator's responsibility, handled the same way as for any other
//! upstream. Keeping authentication and routing separate means this
//! filter has no ordering constraint relative to path-rewrite filters.
//!
//! # Failure behavior
//!
//! The filter **fails closed**: until the first token is acquired, and
//! whenever the cached token is missing or expired, requests are
//! rejected with `503` rather than forwarded unauthenticated. Token
//! acquisition happens asynchronously in the background so that building
//! or hot-reloading a pipeline never blocks on a network round-trip to
//! Entra ID. Repeated acquisition failures back off exponentially (up to
//! 15 minutes) so an unreachable endpoint does not become a tight retry
//! loop.
//!
//! # YAML config
//!
//! ```yaml
//! filter: azure_ad
//! tenant_id: 00000000-0000-0000-0000-000000000000
//! client_id: 11111111-1111-1111-1111-111111111111
//! scope: https://cognitiveservices.azure.com/.default
//! client_secret_env_var: AZURE_CLIENT_SECRET
//! authority_host: login.microsoftonline.com   # optional, for sovereign clouds
//! refresh_ratio: 0.75                          # optional, refresh at 75% of the usable lifetime
//! ```

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use http::{HeaderValue, header};
use praxis_filter::FilterError;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use tracing::warn;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Treat a cached token as expired this long before its real expiry, so
/// a token is never injected onto a request that could outlive it in
/// flight.
const EXPIRY_SKEW: Duration = Duration::from_secs(30);

/// Base delay before retrying after a failed token acquisition. Grows
/// exponentially with consecutive failures, capped at
/// [`MAX_RETRY_BACKOFF`].
const RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// Upper bound on the exponential retry backoff, so a persistently
/// unreachable token endpoint settles into an infrequent poll rather than
/// a tight loop.
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(900);

/// Lower bound on the scheduled refresh delay, so a pathologically small
/// `expires_in` from the token endpoint cannot spin the refresher.
const MIN_REFRESH_DELAY: Duration = Duration::from_secs(1);

/// Timeout for a single token-endpoint round-trip.
const TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// How long [`RefreshHandle::drop`] waits for the refresher thread to
/// exit before giving up and logging.
const JOIN_TIMEOUT: Duration = Duration::from_secs(2);

// -----------------------------------------------------------------------------
// Cached token
// -----------------------------------------------------------------------------

/// A bearer token cached in memory, pre-formatted for injection.
///
/// The token is stored as a ready-to-use `Authorization` header value so
/// the request hot path does no string work — it clones a `HeaderValue`
/// and nothing more.
struct CachedToken {
    /// The complete `Authorization` header value (`"Bearer <token>"`),
    /// marked sensitive so it is redacted from header debug output.
    authorization: HeaderValue,

    /// Instant, already adjusted by [`EXPIRY_SKEW`], after which the
    /// token must not be used.
    expires_at: Instant,
}

impl CachedToken {
    /// Whether the token is still safe to inject at `now`.
    fn is_valid(&self, now: Instant) -> bool {
        now < self.expires_at
    }
}

// -----------------------------------------------------------------------------
// Token endpoint
// -----------------------------------------------------------------------------

/// Successful response body from the Entra ID token endpoint. Extra
/// fields (`token_type`, `ext_expires_in`, …) are ignored.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    /// The JWT access token.
    access_token: String,

    /// Token lifetime in seconds.
    expires_in: u64,
}

/// Acquire a token from the Entra ID token endpoint via the
/// client-credentials grant.
///
/// Returns the ready-to-inject `Authorization` header value and the
/// token's lifetime. Kept free of [`CachedToken`]/[`Instant`] so it can
/// be unit-tested against a local mock endpoint.
///
/// # Errors
///
/// Returns [`FilterError`] if the request fails, the endpoint returns a
/// non-success status, the body cannot be parsed, or the returned token
/// is not a valid header value.
async fn fetch_token(
    client: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    client_secret: &str,
    scope: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "client_credentials")
        .append_pair("client_id", client_id)
        .append_pair("client_secret", client_secret)
        .append_pair("scope", scope)
        .finish();

    let response = client
        .post(token_url)
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|e| FilterError::from(format!("azure_ad: token request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        // The error body can echo the request; surface only the status
        // so a misconfigured secret never lands in logs verbatim.
        return Err(FilterError::from(format!(
            "azure_ad: token endpoint returned HTTP status {status}"
        )));
    }

    let token: TokenResponse = response
        .json()
        .await
        .map_err(|e| FilterError::from(format!("azure_ad: failed to parse token response: {e}")))?;

    let mut authorization = HeaderValue::from_str(&format!("Bearer {}", token.access_token))
        .map_err(|e| FilterError::from(format!("azure_ad: token is not a valid header value: {e}")))?;
    authorization.set_sensitive(true);

    Ok((authorization, Duration::from_secs(token.expires_in)))
}

/// Compute how long to wait before the next refresh: `lifetime * ratio`,
/// floored at [`MIN_REFRESH_DELAY`]. Callers pass the skew-adjusted
/// usable lifetime, not the raw TTL, so the refresh always fires while
/// the cached token is still valid.
fn refresh_delay(lifetime: Duration, ratio: f64) -> Duration {
    lifetime.mul_f64(ratio).max(MIN_REFRESH_DELAY)
}

/// Safety margin to subtract from a token's TTL before caching its
/// expiry. Normally [`EXPIRY_SKEW`], but never more than half the TTL, so
/// a short-lived token stays usable for part of its life instead of being
/// cached already-expired (which would fail every request closed).
fn effective_skew(ttl: Duration) -> Duration {
    EXPIRY_SKEW.min(ttl / 2)
}

/// Exponential backoff after `failures` consecutive fetch failures:
/// `RETRY_BACKOFF * 2^(failures - 1)`, capped at [`MAX_RETRY_BACKOFF`].
fn retry_backoff(failures: u32) -> Duration {
    // Bound the shift so it can never exceed u32 width; the result is
    // capped anyway, so a large shift just saturates to the ceiling.
    let shift = failures.saturating_sub(1).min(20);
    RETRY_BACKOFF.saturating_mul(1_u32 << shift).min(MAX_RETRY_BACKOFF)
}

// -----------------------------------------------------------------------------
// Background refresher
// -----------------------------------------------------------------------------

/// Inputs the background refresher needs to acquire and cache tokens.
struct RefresherParams {
    /// Fully-formed token endpoint URL.
    token_url: String,

    /// Application (client) ID.
    client_id: String,

    /// Client secret, resolved from the configured environment variable.
    client_secret: String,

    /// `OAuth2` scope (e.g. `https://cognitiveservices.azure.com/.default`).
    scope: String,

    /// Fraction of a token's usable lifetime (TTL minus the expiry
    /// safety margin) at which to refresh it.
    refresh_ratio: f64,

    /// Shared cache the filter's hot path reads from.
    shared: Arc<ArcSwap<Option<CachedToken>>>,
}

/// Owns the background refresher thread and stops it on drop.
///
/// Dropping the handle cancels the refresher (the filter pipeline was
/// swapped or shut down) and joins the thread, bounded by
/// [`JOIN_TIMEOUT`]. This is the shutdown signal required for background
/// tasks.
struct RefreshHandle {
    /// Cancellation signal for the refresher loop.
    shutdown: CancellationToken,

    /// Refresher thread join handle.
    thread: Option<JoinHandle<()>>,
}

impl Drop for RefreshHandle {
    #[expect(
        clippy::disallowed_methods,
        reason = "Drop is sync; tokio::time::sleep cannot be used here (mirrors routing::overlay)"
    )]
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(handle) = self.thread.take() {
            let start = Instant::now();
            while !handle.is_finished() {
                if start.elapsed() >= JOIN_TIMEOUT {
                    warn!(
                        timeout_secs = JOIN_TIMEOUT.as_secs(),
                        "azure_ad: token refresher thread did not exit within timeout"
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(handle.join());
        }
    }
}

/// Spawn the background token refresher on a dedicated thread with its
/// own current-thread runtime (mirrors `routing::overlay`), so token
/// acquisition never blocks the pipeline build or the request hot path.
fn spawn_refresher(params: RefresherParams) -> RefreshHandle {
    let shutdown = CancellationToken::new();
    let token = shutdown.clone();

    let thread = std::thread::Builder::new()
        .name("azure-ad-token-refresher".to_owned())
        .spawn(
            move || match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime.block_on(refresh_loop(params, token)),
                Err(error) => {
                    warn!(%error, "azure_ad: failed to create refresher runtime; tokens will not be acquired");
                },
            },
        );

    let thread = match thread {
        Ok(handle) => Some(handle),
        Err(error) => {
            // Without the refresher the cache stays empty and every
            // request fails closed (503) — safe, but log loudly.
            warn!(%error, "azure_ad: failed to spawn token refresher thread; requests will fail closed");
            None
        },
    };

    RefreshHandle { shutdown, thread }
}

/// Acquire a token once and publish it to the shared cache.
///
/// Returns `Some(delay)` — the delay until the next scheduled refresh —
/// on success, or `None` on failure. On failure the cache is left
/// untouched, so a still-valid token keeps serving.
async fn refresh_once(client: &reqwest::Client, params: &RefresherParams) -> Option<Duration> {
    match fetch_token(
        client,
        &params.token_url,
        &params.client_id,
        &params.client_secret,
        &params.scope,
    )
    .await
    {
        Ok((authorization, ttl)) => {
            if ttl <= EXPIRY_SKEW {
                warn!(
                    ttl_secs = ttl.as_secs(),
                    "azure_ad: token TTL is unusually short; validity margin reduced"
                );
            }
            Some(publish_token(params, authorization, ttl))
        },
        Err(error) => {
            warn!(%error, "azure_ad: token refresh failed; will retry");
            None
        },
    }
}

/// Cache a freshly fetched token and return the delay until the next
/// scheduled refresh.
///
/// Both the cached expiry and the refresh schedule are computed from the
/// skew-adjusted usable lifetime. Scheduling from the raw TTL instead
/// (`ratio * ttl`) can land past `ttl - skew`, leaving a window every
/// cycle where the cached token is already invalid but the refresh has
/// not fired yet — deterministic 503s on a healthy token endpoint.
fn publish_token(params: &RefresherParams, authorization: HeaderValue, ttl: Duration) -> Duration {
    let usable = ttl.saturating_sub(effective_skew(ttl));
    params.shared.store(Arc::new(Some(CachedToken {
        authorization,
        expires_at: Instant::now() + usable,
    })));
    refresh_delay(usable, params.refresh_ratio)
}

/// Repeatedly acquire a token, publish it to the shared cache, and sleep
/// until the next refresh — until cancelled.
async fn refresh_loop(params: RefresherParams, shutdown: CancellationToken) {
    let client = match reqwest::Client::builder().timeout(TOKEN_REQUEST_TIMEOUT).build() {
        Ok(client) => client,
        Err(error) => {
            warn!(%error, "azure_ad: failed to build HTTP client; requests will fail closed");
            return;
        },
    };

    let mut failures: u32 = 0;
    loop {
        // Race the fetch against cancellation so a pipeline drop is not
        // blocked behind an in-flight token request (up to its 30s
        // timeout) — `RefreshHandle::drop` only waits `JOIN_TIMEOUT`.
        let refreshed = tokio::select! {
            refreshed = refresh_once(&client, &params) => refreshed,
            () = shutdown.cancelled() => break,
        };
        let delay = if let Some(delay) = refreshed {
            failures = 0;
            delay
        } else {
            failures = failures.saturating_add(1);
            retry_backoff(failures)
        };
        tokio::select! {
            () = tokio::time::sleep(delay) => {},
            () = shutdown.cancelled() => break,
        }
    }
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Injects an Azure AD (Entra ID) bearer token into outbound requests.
///
/// Experimental: requires the `azure-ad-filter` cargo feature, which is
/// off by default and activates the `experimental` marker. This filter
/// is a work in progress and its configuration surface may change
/// between releases.
///
/// See the module docs for scope (client-secret only), the
/// routing-vs-authentication separation, and the fail-closed behavior.
pub struct AzureAdFilter {
    /// Lock-free cache of the current token, populated by the background
    /// refresher and read on every request.
    token: Arc<ArcSwap<Option<CachedToken>>>,

    /// Whether the filter is currently failing closed, so the missing-
    /// token condition is logged on state transitions instead of once
    /// per rejected request.
    failing: AtomicBool,

    /// Background refresher; stops when the filter is dropped.
    _refresh: RefreshHandle,
}

impl AzureAdFilter {
    /// Build a filter from parsed config, resolving the client secret
    /// from its environment variable and spawning the refresher.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if `refresh_ratio` is out of range,
    /// `authority_host` or `tenant_id` contain URL-structural characters,
    /// or the configured secret environment variable is unset or not
    /// UTF-8.
    fn new(config: AzureAdConfig) -> Result<Self, FilterError> {
        validate_config(&config)?;

        let client_secret = std::env::var(&config.client_secret_env_var).map_err(|e| {
            FilterError::from(format!(
                "azure_ad: client_secret_env_var '{}' is not set: {e}",
                config.client_secret_env_var
            ))
        })?;

        let token_url = format!(
            "https://{}/{}/oauth2/v2.0/token",
            config.authority_host, config.tenant_id
        );
        let shared = Arc::new(ArcSwap::from_pointee(None));

        let refresh = spawn_refresher(RefresherParams {
            token_url,
            client_id: config.client_id,
            client_secret,
            scope: config.scope,
            refresh_ratio: config.refresh_ratio,
            shared: Arc::clone(&shared),
        });

        Ok(Self {
            token: shared,
            failing: AtomicBool::new(false),
            _refresh: refresh,
        })
    }

    /// Parse YAML config and build a boxed filter instance.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is malformed or the
    /// configured secret environment variable is unset.
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn praxis_filter::HttpFilter>, FilterError> {
        let config = parse_azure_ad_config(config)?;
        Ok(Box::new(Self::new(config)?))
    }
}

#[async_trait::async_trait]
impl praxis_filter::HttpFilter for AzureAdFilter {
    fn name(&self) -> &'static str {
        "azure_ad"
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
                    warn!("azure_ad: valid token available again; resuming request forwarding");
                }
                // Cheap: clone a pre-formatted, sensitive HeaderValue.
                ctx.request_headers_to_set
                    .push((header::AUTHORIZATION, token.authorization.clone()));
                Ok(praxis_filter::FilterAction::Continue)
            },
            _ => {
                // Fail closed: never forward an unauthenticated request.
                // Log on the state transition, not per rejected request,
                // so a token outage under load cannot flood the logs.
                if !self.failing.swap(true, Ordering::Relaxed) {
                    warn!("azure_ad: no valid cached token; rejecting requests with 503 until one is available");
                }
                Ok(praxis_filter::FilterAction::Reject(praxis_filter::Rejection::status(
                    503,
                )))
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `azure_ad` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AzureAdConfig {
    /// Entra ID directory (tenant) ID.
    pub(crate) tenant_id: String,

    /// Application (client) ID of the app registration.
    pub(crate) client_id: String,

    /// `OAuth2` scope to request (e.g.
    /// `https://cognitiveservices.azure.com/.default`).
    pub(crate) scope: String,

    /// Environment variable holding the client secret.
    pub(crate) client_secret_env_var: String,

    /// Entra ID authority host. Override for sovereign clouds (e.g.
    /// `login.microsoftonline.us`).
    #[serde(default = "default_authority_host")]
    pub(crate) authority_host: String,

    /// Fraction of a token's usable lifetime (TTL minus the expiry
    /// safety margin) at which to refresh it. Must be in the open
    /// interval `(0, 1)`.
    #[serde(default = "default_refresh_ratio")]
    pub(crate) refresh_ratio: f64,
}

/// Validate the config fields [`AzureAdFilter::new`] relies on before it
/// reads the secret and spawns the refresher.
fn validate_config(config: &AzureAdConfig) -> Result<(), FilterError> {
    if !(config.refresh_ratio > 0.0 && config.refresh_ratio < 1.0) {
        return Err(FilterError::from(format!(
            "azure_ad: refresh_ratio must be between 0 and 1 (exclusive), got {}",
            config.refresh_ratio
        )));
    }
    validate_url_component("authority_host", &config.authority_host)?;
    validate_url_component("tenant_id", &config.tenant_id)?;
    Ok(())
}

/// Reject a config value that could break out of its URL component and
/// redirect the token request — which carries the client secret in its
/// body — to an unintended endpoint.
///
/// `authority_host` is interpolated as the URL authority and `tenant_id`
/// as a path segment; either could otherwise inject a scheme, an `@`
/// userinfo host override, or extra path/query/fragment. This does not
/// attempt full hostname validation — it only forbids the characters that
/// change URL structure, which keeps GUID and domain-style tenants valid.
fn validate_url_component(field: &str, value: &str) -> Result<(), FilterError> {
    if value.is_empty() {
        return Err(FilterError::from(format!("azure_ad: {field} must not be empty")));
    }
    let forbidden = |c: char| matches!(c, '/' | '\\' | '?' | '#' | '@') || c.is_whitespace() || c.is_control();
    if value.contains(forbidden) {
        return Err(FilterError::from(format!(
            "azure_ad: {field} '{value}' is invalid: it must be a bare value with no scheme, \
             path, query, '@', or whitespace"
        )));
    }
    Ok(())
}

/// Default value for [`AzureAdConfig::authority_host`].
fn default_authority_host() -> String {
    "login.microsoftonline.com".to_owned()
}

/// Default value for [`AzureAdConfig::refresh_ratio`].
fn default_refresh_ratio() -> f64 {
    0.75
}

/// Parse and validate the `azure_ad` filter's YAML config.
///
/// # Errors
///
/// Returns [`FilterError`] if the YAML is malformed, has unknown fields,
/// or is missing a required field.
pub(crate) fn parse_azure_ad_config(config: &serde_yaml::Value) -> Result<AzureAdConfig, FilterError> {
    praxis_filter::parse_filter_config("azure_ad", config)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use std::io::{Read as _, Write as _};

    use http::Method;
    use praxis_filter::{FilterAction, HttpFilter as _};

    use super::{
        AzureAdFilter, CachedToken, EXPIRY_SKEW, MAX_RETRY_BACKOFF, MIN_REFRESH_DELAY, RETRY_BACKOFF, RefreshHandle,
        RefresherParams, effective_skew, fetch_token, parse_azure_ad_config, refresh_delay, refresh_once,
        retry_backoff, validate_url_component,
    };
    use crate::test_utils::{make_filter_context, make_request};

    fn yaml(body: &str) -> serde_yaml::Value {
        serde_yaml::from_str(body).expect("test YAML must parse")
    }

    // -- Config parsing -------------------------------------------------------

    #[test]
    fn parses_minimal_valid_config() {
        let config = parse_azure_ad_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: https://cognitiveservices.azure.com/.default\n\
             client_secret_env_var: AZURE_CLIENT_SECRET\n",
        ))
        .expect("minimal config should parse");

        assert_eq!(config.tenant_id, "tid");
        assert_eq!(config.client_id, "cid");
        assert_eq!(config.client_secret_env_var, "AZURE_CLIENT_SECRET");
        assert_eq!(
            config.authority_host, "login.microsoftonline.com",
            "authority_host should default"
        );
        assert!(
            (config.refresh_ratio - 0.75).abs() < f64::EPSILON,
            "refresh_ratio should default to 0.75"
        );
    }

    #[test]
    fn parses_optional_authority_host_and_refresh_ratio() {
        let config = parse_azure_ad_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: s\n\
             client_secret_env_var: AZURE_CLIENT_SECRET\n\
             authority_host: login.microsoftonline.us\n\
             refresh_ratio: 0.5\n",
        ))
        .expect("full config should parse");

        assert_eq!(config.authority_host, "login.microsoftonline.us");
        assert!((config.refresh_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_missing_required_field() {
        let err = parse_azure_ad_config(&yaml(
            "client_id: cid\n\
             scope: s\n\
             client_secret_env_var: AZURE_CLIENT_SECRET\n",
        ));
        assert!(err.is_err(), "missing 'tenant_id' should fail to parse");
    }

    #[test]
    fn rejects_unknown_field() {
        let err = parse_azure_ad_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: s\n\
             client_secret_env_var: AZURE_CLIENT_SECRET\n\
             typo_field: oops\n",
        ));
        assert!(err.is_err(), "unknown field should be rejected by deny_unknown_fields");
    }

    #[test]
    fn new_rejects_out_of_range_refresh_ratio() {
        for bad in [0.0, 1.0, 1.5, -0.1] {
            let cfg = super::AzureAdConfig {
                tenant_id: "tid".to_owned(),
                client_id: "cid".to_owned(),
                scope: "s".to_owned(),
                client_secret_env_var: "AZURE_TEST_UNSET_SECRET".to_owned(),
                authority_host: super::default_authority_host(),
                refresh_ratio: bad,
            };
            match AzureAdFilter::new(cfg) {
                Ok(_) => panic!("out-of-range refresh_ratio ({bad}) must be rejected"),
                Err(err) => assert!(
                    format!("{err}").contains("refresh_ratio"),
                    "error must name the offending field, got: {err}"
                ),
            }
        }
    }

    #[test]
    fn from_config_propagates_missing_secret() {
        // refresh_ratio is valid, so construction proceeds to the
        // credential resolution step and fails there.
        let err = AzureAdFilter::from_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: s\n\
             client_secret_env_var: AZURE_TEST_FROM_CONFIG_DEFINITELY_UNSET_SECRET\n",
        ));
        assert!(err.is_err(), "missing secret env var must fail construction");
    }

    // -- URL-component validation ---------------------------------------------

    #[test]
    fn validate_url_component_accepts_guid_and_domain_tenants() {
        validate_url_component("tenant_id", "00000000-0000-0000-0000-000000000000")
            .expect("GUID tenant must be accepted");
        validate_url_component("tenant_id", "contoso.onmicrosoft.com").expect("domain tenant must be accepted");
        validate_url_component("authority_host", "login.microsoftonline.com").expect("bare host must be accepted");
        validate_url_component("authority_host", "login.microsoftonline.us:443").expect("host:port must be accepted");
    }

    #[test]
    fn validate_url_component_rejects_structural_characters() {
        // Each of these could redirect the secret-bearing token POST.
        for bad in [
            "login.microsoftonline.com@evil.com", // '@' userinfo host override
            "https://evil.com",                   // scheme injection
            "host/extra",                         // path injection
            "host?q=1",                           // query injection
            "host#frag",                          // fragment injection
            "host with space",
            "",
        ] {
            assert!(
                validate_url_component("authority_host", bad).is_err(),
                "value {bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn new_rejects_authority_host_with_userinfo_override() {
        let cfg = super::AzureAdConfig {
            tenant_id: "tid".to_owned(),
            client_id: "cid".to_owned(),
            scope: "s".to_owned(),
            client_secret_env_var: "AZURE_TEST_UNSET_SECRET".to_owned(),
            authority_host: "login.microsoftonline.com@evil.com".to_owned(),
            refresh_ratio: 0.75,
        };
        match AzureAdFilter::new(cfg) {
            Ok(_) => panic!("malicious authority_host must be rejected"),
            Err(err) => assert!(
                format!("{err}").contains("authority_host"),
                "error must name the offending field, got: {err}"
            ),
        }
    }

    // -- Pure helpers ---------------------------------------------------------

    #[test]
    fn refresh_delay_is_ttl_times_ratio() {
        let delay = refresh_delay(std::time::Duration::from_secs(3600), 0.75);
        assert_eq!(delay, std::time::Duration::from_secs(2700));
    }

    #[test]
    fn refresh_delay_is_floored() {
        // 1s * 0.75 = 750ms, below the floor.
        let delay = refresh_delay(std::time::Duration::from_secs(1), 0.75);
        assert_eq!(delay, MIN_REFRESH_DELAY);
    }

    #[test]
    fn effective_skew_is_capped_at_half_ttl() {
        use std::time::Duration;
        // Long tokens get the full skew.
        assert_eq!(effective_skew(Duration::from_secs(3600)), EXPIRY_SKEW);
        // Short tokens get at most half their TTL, so they are never
        // cached already-expired.
        assert_eq!(effective_skew(Duration::from_secs(40)), Duration::from_secs(20));
        assert_eq!(effective_skew(Duration::from_secs(10)), Duration::from_secs(5));
    }

    #[test]
    fn retry_backoff_grows_then_caps() {
        assert_eq!(retry_backoff(0), RETRY_BACKOFF, "no failures yet -> base delay");
        assert_eq!(retry_backoff(1), RETRY_BACKOFF, "first failure -> base delay");
        assert_eq!(retry_backoff(2), RETRY_BACKOFF * 2);
        assert_eq!(retry_backoff(3), RETRY_BACKOFF * 4);
        assert_eq!(retry_backoff(1000), MAX_RETRY_BACKOFF, "large failure count -> capped");
    }

    #[test]
    fn cached_token_expiry() {
        let now = std::time::Instant::now();
        let token = CachedToken {
            authorization: http::HeaderValue::from_static("Bearer x"),
            expires_at: now + std::time::Duration::from_secs(60),
        };
        assert!(token.is_valid(now), "token in the future must be valid");
        assert!(
            !token.is_valid(now + std::time::Duration::from_secs(61)),
            "token past expiry must be invalid"
        );
    }

    // -- on_request -----------------------------------------------------------

    /// Build a filter directly around a given cache, without spawning a
    /// refresher thread (the `RefreshHandle` has no thread, so its
    /// `Drop` only cancels a token).
    fn test_filter(token: Option<CachedToken>) -> AzureAdFilter {
        AzureAdFilter {
            token: std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(token)),
            failing: std::sync::atomic::AtomicBool::new(false),
            _refresh: RefreshHandle {
                shutdown: tokio_util::sync::CancellationToken::new(),
                thread: None,
            },
        }
    }

    #[tokio::test]
    async fn on_request_injects_bearer_when_token_valid() {
        let filter = test_filter(Some(CachedToken {
            authorization: http::HeaderValue::from_static("Bearer test-token"),
            expires_at: std::time::Instant::now() + std::time::Duration::from_secs(300),
        }));
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.expect("must not error");
        assert!(matches!(action, FilterAction::Continue));

        let auth = ctx
            .request_headers_to_set
            .iter()
            .find(|(name, _)| *name == http::header::AUTHORIZATION)
            .map(|(_, value)| value.to_str().expect("ascii"));
        assert_eq!(auth, Some("Bearer test-token"), "must inject the cached bearer token");
    }

    #[tokio::test]
    async fn on_request_fails_closed_when_no_token() {
        let filter = test_filter(None);
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "no cached token must fail closed with 503"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "no headers must be set when failing closed"
        );
    }

    #[tokio::test]
    async fn on_request_fails_closed_when_token_expired() {
        let filter = test_filter(Some(CachedToken {
            authorization: http::HeaderValue::from_static("Bearer stale"),
            // Expired well beyond the skew.
            expires_at: std::time::Instant::now() - (EXPIRY_SKEW + std::time::Duration::from_secs(1)),
        }));
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "expired token must fail closed with 503"
        );
    }

    // -- fetch_token against a mock endpoint ----------------------------------

    /// Spawn a one-shot HTTP/1.1 server on loopback that replies with
    /// `body` and returns its bound URL.
    fn mock_token_endpoint(body: &'static str) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/token");
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        (url, handle)
    }

    #[tokio::test]
    async fn fetch_token_parses_bearer_and_ttl() {
        let (url, server) = mock_token_endpoint(r#"{"access_token":"abc123","expires_in":3600}"#);
        let client = reqwest::Client::new();

        let (authorization, ttl) = fetch_token(&client, &url, "cid", "secret", "scope")
            .await
            .expect("mock token fetch must succeed");

        assert_eq!(authorization.to_str().unwrap(), "Bearer abc123");
        assert!(authorization.is_sensitive(), "bearer header must be marked sensitive");
        assert_eq!(ttl, std::time::Duration::from_secs(3600));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn refresh_once_caches_usable_token_for_short_ttl() {
        // A TTL below the full EXPIRY_SKEW must still yield a token that
        // is valid right now — regression for the skew-saturates-to-zero
        // bug that would cache an already-expired token and 503 forever.
        let (url, server) = mock_token_endpoint(r#"{"access_token":"short","expires_in":40}"#);
        let shared = std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(None));
        let params = RefresherParams {
            token_url: url,
            client_id: "cid".to_owned(),
            client_secret: "secret".to_owned(),
            scope: "scope".to_owned(),
            refresh_ratio: 0.75,
            shared: std::sync::Arc::clone(&shared),
        };
        let client = reqwest::Client::new();

        let delay = refresh_once(&client, &params)
            .await
            .expect("short-ttl fetch must succeed");
        assert!(delay >= MIN_REFRESH_DELAY, "refresh delay must respect the floor");

        let cached = shared.load_full();
        let token = cached.as_ref().as_ref().expect("a successful fetch must cache a token");
        let now = std::time::Instant::now();
        assert!(
            token.is_valid(now),
            "a 40s token must be usable now, not cached already-expired"
        );
        // Regression: the refresh must be scheduled from the
        // skew-adjusted usable lifetime. Scheduling from the raw
        // TTL (0.75 * 40s = 30s) lands past the cached expiry
        // (40s - 20s skew = 20s), leaving a guaranteed window of
        // 503s every cycle even with a healthy token endpoint.
        assert!(
            token.is_valid(now + delay),
            "the cached token must still be valid when the scheduled refresh fires"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn refresh_once_returns_none_on_failure() {
        // Closed port -> fetch fails -> no token cached, None returned so
        // the loop applies backoff.
        let shared = std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(None));
        let params = RefresherParams {
            token_url: "http://127.0.0.1:1/token".to_owned(),
            client_id: "cid".to_owned(),
            client_secret: "secret".to_owned(),
            scope: "scope".to_owned(),
            refresh_ratio: 0.75,
            shared: std::sync::Arc::clone(&shared),
        };
        let client = reqwest::Client::new();

        assert!(
            refresh_once(&client, &params).await.is_none(),
            "a failed fetch must return None"
        );
        assert!(shared.load_full().is_none(), "a failed fetch must not cache a token");
    }

    #[tokio::test]
    async fn fetch_token_errors_on_non_success_status() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/token");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 4096];
            let _ = stream.read(&mut buf).unwrap();
            stream
                .write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .unwrap();
        });

        let client = reqwest::Client::new();
        let err = fetch_token(&client, &url, "cid", "secret", "scope")
            .await
            .expect_err("401 must produce an error");
        assert!(format!("{err}").contains("401"), "error must carry the status: {err}");
        server.join().unwrap();
    }
}
