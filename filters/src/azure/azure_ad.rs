// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`AzureAdFilter`] — injects Azure AD (Entra ID) bearer tokens into
//! outbound requests to Azure OpenAI and other Azure AI services.
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
//! Entra ID.
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
//! refresh_ratio: 0.75                          # optional, refresh at 75% of TTL
//! ```

use std::{
    sync::Arc,
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

/// How long to wait before retrying after a failed token acquisition.
const RETRY_BACKOFF: Duration = Duration::from_secs(30);

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

/// Compute how long to wait before the next refresh: `ttl * ratio`,
/// floored at [`MIN_REFRESH_DELAY`].
fn refresh_delay(ttl: Duration, ratio: f64) -> Duration {
    ttl.mul_f64(ratio).max(MIN_REFRESH_DELAY)
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

    /// Fraction of a token's TTL at which to refresh it.
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

/// Acquire a token once, publish it to the shared cache, and return the
/// delay until the next refresh. On failure the cache is left untouched
/// (so a still-valid token keeps serving) and the retry backoff is used.
async fn refresh_once(client: &reqwest::Client, params: &RefresherParams) -> Duration {
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
            let expires_at = Instant::now() + ttl.saturating_sub(EXPIRY_SKEW);
            params.shared.store(Arc::new(Some(CachedToken {
                authorization,
                expires_at,
            })));
            refresh_delay(ttl, params.refresh_ratio)
        },
        Err(error) => {
            warn!(%error, "azure_ad: token refresh failed; will retry");
            RETRY_BACKOFF
        },
    }
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

    loop {
        let delay = refresh_once(&client, &params).await;
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
/// See the module docs for scope (client-secret only), the
/// routing-vs-authentication separation, and the fail-closed behavior.
pub struct AzureAdFilter {
    /// Lock-free cache of the current token, populated by the background
    /// refresher and read on every request.
    token: Arc<ArcSwap<Option<CachedToken>>>,

    /// Background refresher; stops when the filter is dropped.
    _refresh: RefreshHandle,
}

impl AzureAdFilter {
    /// Build a filter from parsed config, resolving the client secret
    /// from its environment variable and spawning the refresher.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if `refresh_ratio` is out of range or the
    /// configured secret environment variable is unset or not UTF-8.
    fn new(config: AzureAdConfig) -> Result<Self, FilterError> {
        if !(config.refresh_ratio > 0.0 && config.refresh_ratio < 1.0) {
            return Err(FilterError::from(format!(
                "azure_ad: refresh_ratio must be between 0 and 1 (exclusive), got {}",
                config.refresh_ratio
            )));
        }

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
        let cached = self.token.load_full();
        match cached.as_ref() {
            Some(token) if token.is_valid(Instant::now()) => {
                // Cheap: clone a pre-formatted, sensitive HeaderValue.
                ctx.request_headers_to_set
                    .push((header::AUTHORIZATION, token.authorization.clone()));
                Ok(praxis_filter::FilterAction::Continue)
            },
            _ => {
                // Fail closed: never forward an unauthenticated request.
                warn!("azure_ad: no valid cached token; rejecting with 503");
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

    /// Fraction of a token's TTL at which to refresh it. Must be in the
    /// open interval `(0, 1)`.
    #[serde(default = "default_refresh_ratio")]
    pub(crate) refresh_ratio: f64,
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
        AzureAdFilter, CachedToken, EXPIRY_SKEW, MIN_REFRESH_DELAY, RefreshHandle, fetch_token, parse_azure_ad_config,
        refresh_delay,
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
