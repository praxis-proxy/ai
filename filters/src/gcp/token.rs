// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Credential-source resolution and token fetch for [`GcpAdcFilter`].
//!
//! Two credential flows live here:
//!
//! - [`TokenSource::Metadata`] acquires a token from the GCE/GKE metadata server, which returns one on request for the
//!   VM's attached service account.
//! - [`TokenSource::ServiceAccountKey`] (a parsed `type: service_account` key file) mints a token itself: it signs a
//!   `JWT` assertion with the key file's private key and exchanges it at Google's `OAuth2` token endpoint for a
//!   short-lived access token (`urn:ietf:params:oauth: grant-type:jwt-bearer`). Nothing is cached at this layer —
//!   caching is [`TokenCache`](praxis_ai_apis::token_cache::TokenCache)'s job.

use std::{path::Path, time::Duration};

use http::HeaderValue;
use praxis_filter::FilterError;
use serde::Deserialize;

use super::config::{GcpAdcConfig, GcpAdcSource};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// `OAuth2` grant type for a signed service-account assertion, per
/// the Google `JWT`-bearer flow.
const JWT_BEARER_GRANT: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";

/// Lifetime of a signed assertion. Google accepts up to one hour; a
/// shorter lifetime narrows the window in which a leaked assertion can
/// be replayed against the token endpoint.
const ASSERTION_LIFETIME: Duration = Duration::from_secs(600);

/// The only non-loopback token-endpoint host accepted for
/// [`TokenSource::ServiceAccountKey`]: Google's `OAuth2` token endpoint.
const GOOGLE_TOKEN_HOST: &str = "oauth2.googleapis.com";

// -----------------------------------------------------------------------------
// TokenSource
// -----------------------------------------------------------------------------

/// Resolved credential source used to fetch a token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum TokenSource {
    /// GCE/GKE/Cloud Run metadata server.
    Metadata {
        /// Service account email or `default`.
        service_account: String,
    },

    /// Parsed and validated `type: service_account` key file; the token
    /// is minted by signing a `JWT` assertion with its private key.
    ServiceAccountKey(ServiceAccountKey),
}

/// The fields of a `type: service_account` key file needed to mint an
/// access token, validated at construct time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ServiceAccountKey {
    /// `client_email` — the `iss` claim and the identity the token is
    /// minted for.
    pub client_email: String,

    /// PKCS#1 or PKCS#8 PEM private key used to sign the assertion.
    pub private_key_pem: String,

    /// Fully validated token endpoint URL (`token_uri`).
    pub token_url: String,

    /// Whether [`Self::token_url`] targets the loopback address. Only
    /// true for test fixtures; production key files carry
    /// [`GOOGLE_TOKEN_HOST`]. Selects the address policy of the pinned
    /// client, mirroring how `metadata_host` accepts loopback for tests.
    pub loopback_test: bool,
}

// -----------------------------------------------------------------------------
// Fetch
// -----------------------------------------------------------------------------

/// Response body from a Google `OAuth2` token endpoint (the metadata
/// server's token endpoint and `oauth2.googleapis.com` return the same
/// shape). Extra fields (`token_type`, `scope`, …) are ignored.
#[derive(Debug, Deserialize)]
struct TokenEndpointResponse {
    /// The `OAuth2` access token.
    access_token: String,

    /// Token lifetime in seconds.
    expires_in: u64,
}

/// Acquire a token for `source`.
///
/// Kept free of caching concerns so it fits
/// [`TokenCache::get_or_refresh`](praxis_ai_apis::token_cache::TokenCache::get_or_refresh)'s
/// `fetch` closure shape directly.
///
/// # Errors
///
/// Returns [`FilterError`] if the token request fails, returns a
/// non-success status, or its body cannot be parsed.
#[cfg(test)]
pub(super) async fn fetch(
    client: &reqwest::Client,
    source: &TokenSource,
    metadata_host: &str,
    scope: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    match source {
        TokenSource::Metadata { service_account } => {
            fetch_metadata_token(client, metadata_host, service_account, scope).await
        },
        TokenSource::ServiceAccountKey(key) => fetch_service_account_token(client, key, scope).await,
    }
}

/// Acquire a token through a proxy-free, redirect-free client pinned to
/// every validated metadata-server address returned by one DNS lookup.
///
/// The metadata protocol intentionally targets a private endpoint. The
/// configured host is separately restricted to Google's metadata hostname
/// or a literal loopback test host.
///
/// The service-account key source is likewise pinned; its address policy
/// follows the key file's validated `token_uri`: public-only for
/// [`GOOGLE_TOKEN_HOST`], loopback-allowed only for the loopback test
/// fixture the config validators admit.
pub(super) async fn fetch_pinned(
    source: &TokenSource,
    metadata_host: &str,
    scope: &str,
    timeout: Duration,
) -> Result<(HeaderValue, Duration), FilterError> {
    match source {
        TokenSource::Metadata { service_account } => {
            let url = metadata_token_url(metadata_host, service_account, scope);
            let client = crate::pinned_client::build_pinned_reqwest_client(
                "gcp_adc",
                &url,
                praxis_ai_apis::callout_target::AddressPolicy::AllowPrivate,
                timeout,
            )
            .await?;
            fetch_metadata_token_url(&client, &url).await
        },
        TokenSource::ServiceAccountKey(key) => {
            let policy = if key.loopback_test {
                praxis_ai_apis::callout_target::AddressPolicy::AllowPrivate
            } else {
                praxis_ai_apis::callout_target::AddressPolicy::PublicOnly
            };
            let client =
                crate::pinned_client::build_pinned_reqwest_client("gcp_adc", &key.token_url, policy, timeout).await?;
            fetch_service_account_token(&client, key, scope).await
        },
    }
}

/// Acquire a token from the GCE/GKE metadata server.
#[cfg(test)]
async fn fetch_metadata_token(
    client: &reqwest::Client,
    metadata_host: &str,
    service_account: &str,
    scope: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    let url = metadata_token_url(metadata_host, service_account, scope);
    fetch_metadata_token_url(client, &url).await
}

/// Build the metadata token URL from already validated components.
fn metadata_token_url(metadata_host: &str, service_account: &str, scope: &str) -> String {
    let mut url =
        format!("http://{metadata_host}/computeMetadata/v1/instance/service-accounts/{service_account}/token?scopes=");
    url::form_urlencoded::byte_serialize(scope.as_bytes()).for_each(|piece| url.push_str(piece));
    url
}

/// Send one metadata token request with a caller-configured client.
async fn fetch_metadata_token_url(client: &reqwest::Client, url: &str) -> Result<(HeaderValue, Duration), FilterError> {
    let response = client
        .get(url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: metadata token request failed: {e}")))?;

    parse_token_response(response, "metadata").await
}

/// Mint an access token from a service-account key file: sign a `JWT`
/// assertion and exchange it at the (already validated) token endpoint.
async fn fetch_service_account_token(
    client: &reqwest::Client,
    key: &ServiceAccountKey,
    scope: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    let assertion = sign_assertion(key, scope)?;

    let body = format!(
        "grant_type={JWT_BEARER_GRANT}&assertion={assertion}",
        assertion = url::form_urlencoded::byte_serialize(assertion.as_bytes()).collect::<String>()
    );
    let response = client
        .post(&key.token_url)
        .header(http::header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: service-account token request failed: {e}")))?;

    parse_token_response(response, "service-account key").await
}

/// The `JWT`-bearer assertion claims (RFC 7523 style: `iss` is the
/// service-account email, `aud` the token endpoint, `scope` the
/// requested `OAuth2` scope).
#[derive(serde::Serialize)]
struct AssertionClaims<'a> {
    /// The service-account email asserting the request.
    iss: &'a str,
    /// The `OAuth2` scope requested with the minted token.
    scope: &'a str,
    /// The token endpoint this assertion is valid at.
    aud: &'a str,
    /// Issuance time, seconds since the Unix epoch.
    iat: u64,
    /// Expiry time, seconds since the Unix epoch.
    exp: u64,
}

/// Sign the `JWT`-bearer assertion for `key`.
fn sign_assertion(key: &ServiceAccountKey, scope: &str) -> Result<String, FilterError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| FilterError::from(format!("gcp_adc: system clock before Unix epoch: {e}")))?
        .as_secs();

    let claims = AssertionClaims {
        iss: &key.client_email,
        scope,
        aud: &key.token_url,
        iat: now,
        exp: now.saturating_add(ASSERTION_LIFETIME.as_secs()),
    };

    let signing_key = jsonwebtoken::EncodingKey::from_rsa_pem(key.private_key_pem.as_bytes())
        .map_err(|e| FilterError::from(format!("gcp_adc: invalid private key in credentials file: {e}")))?;
    let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
    jsonwebtoken::encode(&header, &claims, &signing_key)
        .map_err(|e| FilterError::from(format!("gcp_adc: failed to sign service-account assertion: {e}")))
}

/// Parse a token-endpoint response into a sensitive bearer header plus
/// its lifetime. `context` names the endpoint in error messages; the
/// response body is never included (it can carry credential material).
async fn parse_token_response(
    response: reqwest::Response,
    context: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    let status = response.status();
    if !status.is_success() {
        return Err(FilterError::from(format!(
            "gcp_adc: {context} token endpoint returned HTTP status {status}"
        )));
    }

    let token: TokenEndpointResponse = response
        .json()
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: failed to parse {context} token response: {e}")))?;

    let mut authorization = HeaderValue::from_str(&format!("Bearer {}", token.access_token))
        .map_err(|e| FilterError::from(format!("gcp_adc: {context} token is not a valid header value: {e}")))?;
    authorization.set_sensitive(true);

    Ok((authorization, Duration::from_secs(token.expires_in)))
}

// -----------------------------------------------------------------------------
// GoogleApplicationCredentials
// -----------------------------------------------------------------------------

/// Discriminator parse of a Google ADC JSON file; the `service_account`
/// branch additionally needs the three minting fields, which serde only
/// deserializes on demand.
#[derive(Debug, Deserialize)]
struct GoogleApplicationCredentials {
    /// Google credential `type` field.
    #[serde(rename = "type")]
    cred_type: String,

    /// Service-account email (`iss`).
    #[serde(default)]
    client_email: Option<String>,

    /// PEM private key for assertion signing.
    #[serde(default)]
    private_key: Option<String>,

    /// `OAuth2` token endpoint URL.
    #[serde(default)]
    token_uri: Option<String>,
}

// -----------------------------------------------------------------------------
// Resolution
// -----------------------------------------------------------------------------

/// Resolve the runtime token source from config and an optional ADC path.
///
/// `application_credentials` is the value of `GOOGLE_APPLICATION_CREDENTIALS`
/// when called from production, or a test-supplied path. It is never read
/// from the environment here so unit tests stay hermetic.
///
/// # Errors
///
/// Returns [`FilterError`] if a configured file is missing, unreadable,
/// or has an unsupported or incomplete `type`.
pub(super) fn resolve_token_source(
    config: &GcpAdcConfig,
    application_credentials: Option<&Path>,
) -> Result<TokenSource, FilterError> {
    let metadata_source = || TokenSource::Metadata {
        service_account: config.service_account.clone().unwrap_or_else(|| "default".to_owned()),
    };
    match config.source {
        GcpAdcSource::Metadata => Ok(metadata_source()),
        GcpAdcSource::KeyFile => {
            let path = config
                .credentials_file
                .as_deref()
                .ok_or_else(|| FilterError::from("gcp_adc: source key_file requires credentials_file"))?;
            parse_credential_file(Path::new(path))
        },
        GcpAdcSource::Adc => match application_credentials {
            Some(path) => parse_credential_file(path),
            None => Ok(metadata_source()),
        },
    }
}

/// Read a Google ADC JSON file and map its `type` to a [`TokenSource`].
fn parse_credential_file(path: &Path) -> Result<TokenSource, FilterError> {
    let display = path.display();
    let raw = std::fs::read_to_string(path)
        .map_err(|error| FilterError::from(format!("gcp_adc: failed to read credentials file '{display}': {error}")))?;
    let parsed: GoogleApplicationCredentials = serde_json::from_str(&raw).map_err(|error| {
        FilterError::from(format!(
            "gcp_adc: failed to parse credentials file '{display}': {error}"
        ))
    })?;
    match parsed.cred_type.as_str() {
        "service_account" => service_account_source(parsed),
        "authorized_user" => Err(FilterError::from(
            "gcp_adc: gcloud user ADC (authorized_user) is not supported",
        )),
        "external_account" => Err(FilterError::from(
            "gcp_adc: external_account (WIF/STS) is not implemented yet",
        )),
        other => Err(FilterError::from(format!(
            "gcp_adc: unsupported credential type '{other}'"
        ))),
    }
}

/// Build a [`TokenSource::ServiceAccountKey`] from a parsed key file,
/// requiring every field the minting flow needs so an incomplete file
/// fails at config time rather than per request.
fn service_account_source(parsed: GoogleApplicationCredentials) -> Result<TokenSource, FilterError> {
    let client_email = parsed
        .client_email
        .filter(|email| !email.is_empty())
        .ok_or_else(|| FilterError::from("gcp_adc: credentials file is missing client_email"))?;
    super::config::validate_service_account(&client_email)?;
    let private_key_pem = parsed
        .private_key
        .filter(|key| !key.is_empty())
        .ok_or_else(|| FilterError::from("gcp_adc: credentials file is missing private_key"))?;
    let token_uri = parsed
        .token_uri
        .filter(|uri| !uri.is_empty())
        .ok_or_else(|| FilterError::from("gcp_adc: credentials file is missing token_uri"))?;
    let token_url = validate_token_uri(&token_uri)?;
    let loopback_test = token_url.starts_with("http://127.0.0.1");
    Ok(TokenSource::ServiceAccountKey(ServiceAccountKey {
        client_email,
        private_key_pem,
        token_url,
        loopback_test,
    }))
}

/// Validate a key file's `token_uri` to the exact set of endpoints this
/// filter may call: Google's `OAuth2` token endpoint over `HTTPS`, or a
/// `127.0.0.1` `HTTP` address (test fixtures only, same convention as
/// `metadata_host`). Everything else — any other host, any other scheme,
/// embedded credentials, query or fragment — is invalid configuration,
/// so a tampered key file can never point the signed assertion at an
/// attacker's endpoint.
fn validate_token_uri(raw: &str) -> Result<String, FilterError> {
    let parsed = url::Url::parse(raw).map_err(|e| {
        FilterError::from(format!(
            "gcp_adc: credentials file token_uri '{raw}' is not a valid URL: {e}"
        ))
    })?;
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(FilterError::from(format!(
            "gcp_adc: credentials file token_uri '{raw}' must not contain credentials, a query, or a fragment"
        )));
    }
    let host = parsed.host_str().ok_or_else(|| {
        FilterError::from(format!(
            "gcp_adc: credentials file token_uri '{raw}' must include a host"
        ))
    })?;
    let allowed =
        (parsed.scheme() == "https" && host == GOOGLE_TOKEN_HOST) || (parsed.scheme() == "http" && host == "127.0.0.1");
    if !allowed {
        return Err(FilterError::from(format!(
            "gcp_adc: credentials file token_uri '{raw}' is not allowed: only 'https://{GOOGLE_TOKEN_HOST}' \
             (or an http://127.0.0.1 test fixture) may be used as the token endpoint"
        )));
    }
    Ok(parsed.into())
}
