// SPDX-License-Identifier: Apache-2.0
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
//! `OAuth2` bearer token from Entra ID. This filter acquires such a
//! token via the **client-credentials** grant and injects
//! `Authorization: Bearer <token>` on every proxied request. The
//! downstream client is unaware of Entra ID.
//!
//! # Caching: cache-through, not refresh-ahead
//!
//! There is no background refresh thread. Every request checks the
//! cached token; if it is still valid (outside the [`EXPIRY_SKEW`]
//! safety margin) it is used immediately with no network call. If it is
//! missing or stale, that request's handling acquires a fresh token
//! inline before proceeding — see
//! [`praxis_ai_apis::token_cache::TokenCache`] for the exact
//! cache-through/double-checked-locking contract, including how
//! concurrent requests that all observe a stale cache still trigger at
//! most one token-endpoint call.
//!
//! # Scope
//!
//! This filter currently supports the **client-secret** credential only.
//! Managed identity (AKS/IMDS), client certificates (`private_key_jwt`
//! assertions), and OIDC/workload-identity federation are planned
//! follow-ups; they slot into the same cache-through machinery this
//! filter already uses, with no change to request handling.
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
//! The filter **fails closed**: whenever no valid token can be produced
//! — none cached and the inline fetch fails — the request is rejected
//! with `503` rather than forwarded unauthenticated. There is no
//! server-side retry loop; a failed fetch is not cached, so the next
//! request simply tries again.
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
//! allow_private_authority: false               # opt in only for a trusted private authority
//! ```

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use http::{HeaderValue, header};
use praxis_ai_apis::token_cache::TokenCache;
use praxis_filter::FilterError;
use serde::Deserialize;
use tracing::warn;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Treat a cached token as expired this long before its real expiry, so
/// a token is never injected onto a request that could outlive it in
/// flight. Passed to [`TokenCache::new`] as its safety margin.
const EXPIRY_SKEW: Duration = Duration::from_secs(30);

/// Timeout for a single token-endpoint round-trip.
const TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
/// token's lifetime. Kept free of caching concerns so it can be
/// unit-tested against a local mock endpoint, and so it fits
/// [`TokenCache::get_or_refresh`]'s `fetch` closure shape directly.
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
    /// Cache-through token cache; see the module docs and
    /// [`praxis_ai_apis::token_cache`].
    cache: TokenCache<HeaderValue>,

    /// Fully-formed token endpoint URL.
    token_url: String,

    /// Connect-time address policy for the authority endpoint.
    address_policy: praxis_ai_apis::callout_target::AddressPolicy,

    /// Application (client) ID.
    client_id: String,

    /// Client secret, resolved from the configured environment variable.
    client_secret: String,

    /// `OAuth2` scope (e.g. `https://cognitiveservices.azure.com/.default`).
    scope: String,

    /// Whether the filter is currently failing closed, so the missing-
    /// token condition is logged on state transitions instead of once
    /// per rejected request.
    failing: AtomicBool,
}

impl AzureAdFilter {
    /// Build a filter from parsed config, resolving the client secret
    /// from its environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if `authority_host` or `tenant_id` contain
    /// URL-structural characters, the configured secret environment
    /// variable is unset or not UTF-8, or the HTTP client fails to build.
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
        let address_policy =
            praxis_ai_apis::callout_target::AddressPolicy::from_allow_private(config.allow_private_authority);
        praxis_ai_apis::callout_target::validate_configured_http_target("azure_ad", &token_url, address_policy)?;
        Ok(Self {
            cache: TokenCache::new(EXPIRY_SKEW),
            token_url,
            address_policy,
            client_id: config.client_id,
            client_secret,
            scope: config.scope,
            failing: AtomicBool::new(false),
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

    #[expect(
        clippy::too_many_lines,
        reason = "token refresh and fail-closed state transitions stay together"
    )]
    async fn on_request(
        &self,
        ctx: &mut praxis_filter::HttpFilterContext<'_>,
    ) -> Result<praxis_filter::FilterAction, FilterError> {
        let fetched = self
            .cache
            .get_or_refresh(|| async {
                let client = praxis_ai_apis::callout_target::build_pinned_reqwest_client(
                    "azure_ad",
                    &self.token_url,
                    self.address_policy,
                    TOKEN_REQUEST_TIMEOUT,
                )
                .await?;
                fetch_token(
                    &client,
                    &self.token_url,
                    &self.client_id,
                    &self.client_secret,
                    &self.scope,
                )
                .await
            })
            .await;
        match fetched {
            Ok(authorization) => {
                if self.failing.swap(false, Ordering::Relaxed) {
                    warn!("azure_ad: valid token available again; resuming request forwarding");
                }
                // Cheap: clone a pre-formatted, sensitive HeaderValue.
                ctx.request_headers_to_set.push((header::AUTHORIZATION, authorization));
                Ok(praxis_filter::FilterAction::Continue)
            },
            Err(error) => {
                // Fail closed: never forward an unauthenticated request.
                // Log on the state transition, not per rejected request,
                // so a token outage under load cannot flood the logs.
                if !self.failing.swap(true, Ordering::Relaxed) {
                    warn!(%error, "azure_ad: no valid token available; rejecting requests with 503 until one is acquired");
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

    /// Allow a private authority host for a trusted internal identity service.
    #[serde(default)]
    pub(crate) allow_private_authority: bool,
}

/// Validate the config fields [`AzureAdFilter::new`] relies on before it
/// reads the secret.
fn validate_config(config: &AzureAdConfig) -> Result<(), FilterError> {
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

    use super::{AzureAdConfig, AzureAdFilter, fetch_token, parse_azure_ad_config, validate_url_component};
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
    }

    #[test]
    fn parses_optional_authority_host() {
        let config = parse_azure_ad_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: s\n\
             client_secret_env_var: AZURE_CLIENT_SECRET\n\
             authority_host: login.microsoftonline.us\n",
        ))
        .expect("full config should parse");

        assert_eq!(config.authority_host, "login.microsoftonline.us");
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
    fn from_config_propagates_missing_secret() {
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
        let cfg = AzureAdConfig {
            tenant_id: "tid".to_owned(),
            client_id: "cid".to_owned(),
            scope: "s".to_owned(),
            client_secret_env_var: "AZURE_TEST_UNSET_SECRET".to_owned(),
            authority_host: "login.microsoftonline.com@evil.com".to_owned(),
            allow_private_authority: false,
        };
        match AzureAdFilter::new(cfg) {
            Ok(_) => panic!("malicious authority_host must be rejected"),
            Err(err) => assert!(
                format!("{err}").contains("authority_host"),
                "error must name the offending field, got: {err}"
            ),
        }
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

    #[tokio::test]
    async fn token_redirect_does_not_disclose_client_secret() {
        let redirect_target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        redirect_target.set_nonblocking(true).unwrap();
        let target_addr = redirect_target.local_addr().unwrap();

        let redirector = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let redirector_addr = redirector.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = redirector.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            let response = format!(
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{target_addr}/steal\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let filter = filter_at(&format!("http://{redirector_addr}/token"));
        let request = make_request(Method::POST, "/openai/deployments/gpt-4o/chat/completions");
        let mut ctx = make_filter_context(&request);
        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(
            matches!(action, FilterAction::Reject(_)),
            "redirected token fetch must fail closed"
        );
        server.join().unwrap();
        assert!(
            matches!(redirect_target.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock),
            "redirect target must not receive the secret-bearing token request"
        );
    }

    // -- on_request: cache-through end to end ----------------------------------

    /// Build a filter with a real, always-set secret env var (so
    /// construction never fails on credential resolution) and then point
    /// its token endpoint at `token_url` — bypassing `authority_host`/
    /// `tenant_id` URL construction entirely so tests can target a local
    /// mock server directly.
    fn filter_at(token_url: &str) -> AzureAdFilter {
        let config = parse_azure_ad_config(&yaml(
            "tenant_id: tid\n\
             client_id: cid\n\
             scope: scope\n\
             client_secret_env_var: CARGO_PKG_NAME\n\
             allow_private_authority: true\n",
        ))
        .expect("test config must parse");
        let mut filter =
            AzureAdFilter::new(config).expect("construction must succeed with an always-set secret env var");
        filter.token_url = token_url.to_owned();
        filter
    }

    #[tokio::test]
    async fn on_request_injects_bearer_on_first_fetch() {
        let (url, server) = mock_token_endpoint(r#"{"access_token":"fresh","expires_in":3600}"#);
        let filter = filter_at(&url);
        let request = make_request(Method::POST, "/openai/deployments/gpt-4o/chat/completions");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.expect("must not error");
        assert!(matches!(action, FilterAction::Continue));

        let auth = ctx
            .request_headers_to_set
            .iter()
            .find(|(name, _)| *name == http::header::AUTHORIZATION)
            .map(|(_, value)| value.to_str().expect("ascii"));
        assert_eq!(
            auth,
            Some("Bearer fresh"),
            "must inject the freshly fetched bearer token"
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn on_request_reuses_cached_token_without_a_second_fetch() {
        // The mock endpoint accepts exactly one connection; a second
        // on_request call must not attempt a second fetch, or it would
        // fail to connect and 503 instead of continuing.
        let (url, server) = mock_token_endpoint(r#"{"access_token":"once","expires_in":3600}"#);
        let filter = filter_at(&url);
        let request = make_request(Method::POST, "/openai/deployments/gpt-4o/chat/completions");

        let mut first_ctx = make_filter_context(&request);
        let first = filter
            .on_request(&mut first_ctx)
            .await
            .expect("first call must not error");
        assert!(matches!(first, FilterAction::Continue));

        let mut second_ctx = make_filter_context(&request);
        let second = filter
            .on_request(&mut second_ctx)
            .await
            .expect("second call must not error");
        assert!(
            matches!(second, FilterAction::Continue),
            "a still-valid cache must serve the second request without a new connection"
        );
        let auth = second_ctx
            .request_headers_to_set
            .iter()
            .find(|(name, _)| *name == http::header::AUTHORIZATION)
            .map(|(_, value)| value.to_str().expect("ascii"));
        assert_eq!(auth, Some("Bearer once"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn on_request_fails_closed_when_fetch_fails() {
        // A closed local port: the connection is refused immediately, no
        // real network dependency.
        let filter = filter_at("http://127.0.0.1:1/token");
        let request = make_request(Method::POST, "/openai/deployments/gpt-4o/chat/completions");
        let mut ctx = make_filter_context(&request);

        let action = filter.on_request(&mut ctx).await.expect("must reject, not error");
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "a failed fetch must fail closed with 503"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "no headers must be set when failing closed"
        );
    }
}
