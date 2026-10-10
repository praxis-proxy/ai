// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Credential-source resolution and token fetch for [`GcpAdcFilter`].
//!
//! Two credential flows live here:
//!
//! - [`TokenSource::Metadata`] acquires a token from the GCE/GKE metadata server, which returns one on request for the
//!   VM's attached service account.
//! - [`TokenSource::ServiceAccountKey`] (a parsed `type: service_account` key file) mints a token itself: it signs a
//!   `JWT` assertion with the key file's private key through the system OpenSSL and exchanges it at Google's `OAuth2`
//!   token endpoint for a short-lived access token (`urn:ietf:params:oauth:grant-type:jwt-bearer`). Nothing is cached
//!   at this layer: caching is [`TokenCache`](praxis_ai_apis::token_cache::TokenCache)'s job.

use std::{fmt, path::Path, time::Duration};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::HeaderValue;
use openssl::{
    hash::MessageDigest,
    pkey::{Id, PKey, Private},
    rsa::Padding,
    sign::Signer,
};
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{SubRequest, SubRequestClient, SubResponse, execute_url},
};
use praxis_filter::FilterError;
use serde::Deserialize;
use zeroize::Zeroizing;

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

/// `JOSE` header of every assertion: `RS256` (RSA PKCS#1 v1.5 over
/// SHA-256) is what Google's token endpoint expects for service-account
/// keys.
const ASSERTION_HEADER: &str = r#"{"alg":"RS256","typ":"JWT"}"#;

/// The only token endpoint a key file's `token_uri` may name: Google's
/// `OAuth2` token endpoint.
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

// -----------------------------------------------------------------------------
// TokenSource
// -----------------------------------------------------------------------------

/// Resolved credential source used to fetch a token.
#[derive(Debug)]
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
/// access token, validated and parsed at construct time.
///
/// `Debug` is written by hand so the private key never reaches a log line
/// or an assertion message.
pub(super) struct ServiceAccountKey {
    /// `client_email`: the `iss` claim and the identity the token is
    /// minted for.
    pub client_email: String,

    /// RSA private key that signs the assertion, parsed from the key
    /// file's PEM.
    pub private_key: PKey<Private>,

    /// Validated token endpoint URL (`token_uri`): [`GOOGLE_TOKEN_URL`]
    /// outside this crate's unit tests.
    pub token_url: String,
}

impl fmt::Debug for ServiceAccountKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServiceAccountKey")
            .field("client_email", &self.client_email)
            .field("token_url", &self.token_url)
            .finish_non_exhaustive()
    }
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
    client: &SubRequestClient,
    source: &TokenSource,
    metadata_host: &str,
    scope: &str,
    timeout: Duration,
) -> Result<(HeaderValue, Duration), FilterError> {
    match source {
        TokenSource::Metadata { service_account } => {
            fetch_metadata_token(client, metadata_host, service_account, scope, timeout).await
        },
        TokenSource::ServiceAccountKey(key) => {
            fetch_service_account_token(client, key, scope, timeout, token_address_policy(&key.token_url)).await
        },
    }
}

/// Acquire a token for `source` through `SubRequestClient`.
///
/// The metadata protocol intentionally targets a private endpoint. The
/// configured host is separately restricted to Google's metadata hostname.
///
/// The service-account key source only reaches Google's public token
/// endpoint, so it is public-only.
pub(super) async fn fetch_pinned(
    client: &SubRequestClient,
    source: &TokenSource,
    metadata_host: &str,
    scope: &str,
    timeout: Duration,
) -> Result<(HeaderValue, Duration), FilterError> {
    match source {
        TokenSource::Metadata { service_account } => {
            let url = metadata_token_url(metadata_host, service_account, scope);
            fetch_metadata_token_url(client, &url, timeout, metadata_address_policy(metadata_host)).await
        },
        TokenSource::ServiceAccountKey(key) => {
            fetch_service_account_token(client, key, scope, timeout, token_address_policy(&key.token_url)).await
        },
    }
}

/// Use the production metadata-only policy except for literal loopback mocks
/// compiled into this crate's unit tests.
#[cfg(test)]
fn metadata_address_policy(metadata_host: &str) -> AddressPolicy {
    if metadata_host.split(':').next().unwrap_or(metadata_host) == "127.0.0.1" {
        return AddressPolicy::AllowPrivate;
    }
    AddressPolicy::AllowGoogleMetadata
}

/// Use the metadata-only policy in every production build.
#[cfg(not(test))]
fn metadata_address_policy(_metadata_host: &str) -> AddressPolicy {
    AddressPolicy::AllowGoogleMetadata
}

/// Use the production public-only policy except for the loopback mocks
/// [`validate_token_uri`] admits in this crate's unit tests.
#[cfg(test)]
fn token_address_policy(token_url: &str) -> AddressPolicy {
    if token_url == GOOGLE_TOKEN_URL {
        return AddressPolicy::PublicOnly;
    }
    AddressPolicy::AllowPrivate
}

/// Google's token endpoint is public; production builds reach nothing else.
#[cfg(not(test))]
fn token_address_policy(_token_url: &str) -> AddressPolicy {
    AddressPolicy::PublicOnly
}

/// Acquire a token from the GCE/GKE metadata server.
#[cfg(test)]
async fn fetch_metadata_token(
    client: &SubRequestClient,
    metadata_host: &str,
    service_account: &str,
    scope: &str,
    timeout: Duration,
) -> Result<(HeaderValue, Duration), FilterError> {
    let url = metadata_token_url(metadata_host, service_account, scope);
    fetch_metadata_token_url(client, &url, timeout, AddressPolicy::AllowPrivate).await
}

/// Build the metadata token URL from already validated components.
fn metadata_token_url(metadata_host: &str, service_account: &str, scope: &str) -> String {
    let mut url =
        format!("http://{metadata_host}/computeMetadata/v1/instance/service-accounts/{service_account}/token?scopes=");
    url::form_urlencoded::byte_serialize(scope.as_bytes()).for_each(|piece| url.push_str(piece));
    url
}

/// Send one metadata token request via `SubRequestClient`.
async fn fetch_metadata_token_url(
    client: &SubRequestClient,
    url: &str,
    timeout: Duration,
    address_policy: AddressPolicy,
) -> Result<(HeaderValue, Duration), FilterError> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::HeaderName::from_static("metadata-flavor"),
        HeaderValue::from_static("Google"),
    );
    let request = SubRequest {
        method: http::Method::GET,
        uri: http::Uri::default(),
        headers,
        body: Bytes::new(),
    };

    let response = execute_url(client, url, request, 65_536, timeout, address_policy)
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: metadata token request failed: {e}")))?;

    parse_token_response(&response, "metadata")
}

/// Mint an access token from a service-account key file: sign a `JWT`
/// assertion and exchange it at the (already validated) token endpoint.
async fn fetch_service_account_token(
    client: &SubRequestClient,
    key: &ServiceAccountKey,
    scope: &str,
    timeout: Duration,
    address_policy: AddressPolicy,
) -> Result<(HeaderValue, Duration), FilterError> {
    let assertion = sign_assertion(key, scope)?;

    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", JWT_BEARER_GRANT)
        .append_pair("assertion", &assertion)
        .finish();
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-www-form-urlencoded"),
    );
    let request = SubRequest {
        method: http::Method::POST,
        uri: http::Uri::default(),
        headers,
        body: Bytes::from(body),
    };

    let response = execute_url(client, &key.token_url, request, 65_536, timeout, address_policy)
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: service-account token request failed: {e}")))?;

    parse_token_response(&response, "service-account key")
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

/// Sign the `JWT`-bearer assertion for `key`: base64url header and claims,
/// then their `RS256` signature, computed by the system OpenSSL so it runs
/// inside the validated module on a FIPS host.
fn sign_assertion(key: &ServiceAccountKey, scope: &str) -> Result<String, FilterError> {
    let claims = assertion_claims(key, scope)?;

    let mut assertion = URL_SAFE_NO_PAD.encode(ASSERTION_HEADER);
    assertion.push('.');
    URL_SAFE_NO_PAD.encode_string(claims, &mut assertion);

    let signature = rs256_signature(key, assertion.as_bytes())?;
    assertion.push('.');
    URL_SAFE_NO_PAD.encode_string(signature, &mut assertion);
    Ok(assertion)
}

/// Serialize the assertion claims for `key`, issued now and valid for
/// [`ASSERTION_LIFETIME`].
fn assertion_claims(key: &ServiceAccountKey, scope: &str) -> Result<Vec<u8>, FilterError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| FilterError::from(format!("gcp_adc: system clock before Unix epoch: {e}")))?
        .as_secs();

    serde_json::to_vec(&AssertionClaims {
        iss: &key.client_email,
        scope,
        aud: &key.token_url,
        iat: now,
        exp: now.saturating_add(ASSERTION_LIFETIME.as_secs()),
    })
    .map_err(|e| FilterError::from(format!("gcp_adc: failed to encode service-account assertion: {e}")))
}

/// `RS256` signature of `signing_input` with the key file's private key.
/// OpenSSL errors name the failing operation, never key material.
fn rs256_signature(key: &ServiceAccountKey, signing_input: &[u8]) -> Result<Vec<u8>, FilterError> {
    Signer::new(MessageDigest::sha256(), &key.private_key)
        .and_then(|mut signer| {
            signer.set_rsa_padding(Padding::PKCS1)?;
            signer.sign_oneshot_to_vec(signing_input)
        })
        .map_err(|e| FilterError::from(format!("gcp_adc: failed to sign service-account assertion: {e}")))
}

/// Parse a token-endpoint response into a sensitive bearer header plus
/// its lifetime. `context` names the endpoint in error messages; the
/// response body is never included (it can carry credential material).
fn parse_token_response(response: &SubResponse, context: &str) -> Result<(HeaderValue, Duration), FilterError> {
    let status = http::StatusCode::from_u16(response.status).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
    if !status.is_success() {
        return Err(FilterError::from(format!(
            "gcp_adc: {context} token endpoint returned HTTP status {status}"
        )));
    }

    let token: TokenEndpointResponse = serde_json::from_slice(&response.body)
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
/// deserializes on demand. No `Debug`: it carries the private key.
#[derive(Deserialize)]
struct GoogleApplicationCredentials {
    /// Google credential `type` field.
    #[serde(rename = "type")]
    cred_type: String,

    /// Service-account email (`iss`).
    #[serde(default)]
    client_email: Option<String>,

    /// PEM private key for assertion signing, wiped from memory once
    /// parsed.
    #[serde(default)]
    private_key: Option<Zeroizing<String>>,

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
    let raw = Zeroizing::new(std::fs::read_to_string(path).map_err(|error| {
        FilterError::from(format!("gcp_adc: failed to read credentials file '{display}': {error}"))
    })?);
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
    let private_key = parsed
        .private_key
        .filter(|key| !key.is_empty())
        .ok_or_else(|| FilterError::from("gcp_adc: credentials file is missing private_key"))?;
    let token_uri = parsed
        .token_uri
        .filter(|uri| !uri.is_empty())
        .ok_or_else(|| FilterError::from("gcp_adc: credentials file is missing token_uri"))?;
    let token_url = validate_token_uri(&token_uri)?;
    Ok(TokenSource::ServiceAccountKey(ServiceAccountKey {
        client_email,
        private_key: parse_private_key(&private_key)?,
        token_url,
    }))
}

/// Parse the key file's PEM private key once, at construct time, so a
/// malformed or non-RSA key is a configuration error rather than a failure
/// on every request. Google issues unencrypted PKCS#8; PKCS#1 loads too.
fn parse_private_key(pem: &str) -> Result<PKey<Private>, FilterError> {
    // Without a callback OpenSSL prompts on the controlling terminal for an
    // encrypted key; refusing every passphrase makes it a load error instead.
    let key = PKey::private_key_from_pem_callback(pem.as_bytes(), |_passphrase| Ok(0)).map_err(|e| {
        FilterError::from(format!(
            "gcp_adc: credentials file private_key is not an unencrypted PEM private key: {e}"
        ))
    })?;
    if key.id() != Id::RSA {
        return Err(FilterError::from(
            "gcp_adc: credentials file private_key must be an RSA key (assertions are signed with RS256)",
        ));
    }
    Ok(key)
}

/// Accept only Google's `OAuth2` token endpoint as a key file's
/// `token_uri`, compared after URL normalization, so a tampered key file can
/// never send the signed assertion anywhere else: any other scheme, host,
/// port, path, userinfo, query, or fragment is invalid configuration.
/// Unit-test builds additionally accept a plain `http` literal IPv4 loopback
/// endpoint so the mint can run against an in-process mock.
///
/// Errors never echo the value: a malformed URL may carry credentials.
fn validate_token_uri(raw: &str) -> Result<String, FilterError> {
    let parsed = url::Url::parse(raw)
        .map_err(|e| FilterError::from(format!("gcp_adc: credentials file token_uri is not a valid URL: {e}")))?;
    let is_allowed = parsed.as_str() == GOOGLE_TOKEN_URL;
    #[cfg(test)]
    let is_allowed = is_allowed || is_loopback_test_endpoint(&parsed);
    if !is_allowed {
        return Err(FilterError::from(format!(
            "gcp_adc: credentials file token_uri must be '{GOOGLE_TOKEN_URL}'; no other token endpoint may \
             receive the signed assertion"
        )));
    }
    Ok(parsed.into())
}

/// A plain `http` endpoint on literal `127.0.0.1` with no userinfo, query,
/// or fragment: the in-process token endpoint mock of this crate's tests.
#[cfg(test)]
fn is_loopback_test_endpoint(url: &url::Url) -> bool {
    url.scheme() == "http"
        && url.host() == Some(url::Host::Ipv4(std::net::Ipv4Addr::LOCALHOST))
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
}
