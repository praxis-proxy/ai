// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Credential-source resolution and token fetch for [`GcpAdcFilter`].
//!
//! [`fetch`] acquires a token for the [`Metadata`](TokenSource::Metadata)
//! source from the GCE/GKE metadata server. The
//! [`ServiceAccountKey`](TokenSource::ServiceAccountKey) source (a parsed
//! `type: service_account` key file) is resolved and validated at
//! construct time but its token fetch is not implemented yet — it needs
//! `JWT` signing, which this workspace does not currently depend on — so
//! [`fetch`] returns a clear error for it rather than the cache silently
//! staying empty forever.

use std::{path::Path, time::Duration};

use http::HeaderValue;
use praxis_filter::FilterError;
use serde::Deserialize;

use super::config::{GcpAdcConfig, GcpAdcSource};

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

    /// Parsed `type: service_account` key file. Fetch is not implemented
    /// yet (requires `JWT` signing); [`fetch`] returns an error for this
    /// source so requests fail closed with a clear reason.
    ServiceAccountKey,
}

// -----------------------------------------------------------------------------
// Fetch
// -----------------------------------------------------------------------------

/// Response body from the GCE/GKE metadata server's token endpoint.
/// Extra fields (`token_type`, …) are ignored.
#[derive(Debug, Deserialize)]
struct MetadataTokenResponse {
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
/// Returns [`FilterError`] if the metadata request fails, returns a
/// non-success status, or its body cannot be parsed; or, for
/// [`TokenSource::ServiceAccountKey`], always (not implemented).
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
        TokenSource::ServiceAccountKey => Err(FilterError::from(
            "gcp_adc: token fetch for source key_file is not implemented yet (requires JWT signing); \
             use source: adc or source: metadata on a GCE/GKE instance instead",
        )),
    }
}

/// Acquire a token from the GCE/GKE metadata server.
async fn fetch_metadata_token(
    client: &reqwest::Client,
    metadata_host: &str,
    service_account: &str,
    scope: &str,
) -> Result<(HeaderValue, Duration), FilterError> {
    let mut url =
        format!("http://{metadata_host}/computeMetadata/v1/instance/service-accounts/{service_account}/token?scopes=");
    url::form_urlencoded::byte_serialize(scope.as_bytes()).for_each(|piece| url.push_str(piece));

    let response = client
        .get(url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: metadata token request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        return Err(FilterError::from(format!(
            "gcp_adc: metadata server returned HTTP status {status}"
        )));
    }

    let token: MetadataTokenResponse = response
        .json()
        .await
        .map_err(|e| FilterError::from(format!("gcp_adc: failed to parse metadata token response: {e}")))?;

    let mut authorization = HeaderValue::from_str(&format!("Bearer {}", token.access_token))
        .map_err(|e| FilterError::from(format!("gcp_adc: token is not a valid header value: {e}")))?;
    authorization.set_sensitive(true);

    Ok((authorization, Duration::from_secs(token.expires_in)))
}

// -----------------------------------------------------------------------------
// GoogleApplicationCredentials
// -----------------------------------------------------------------------------

/// Discriminator-only parse of a Google ADC JSON file.
#[derive(Debug, Deserialize)]
struct GoogleApplicationCredentials {
    /// Google credential `type` field.
    #[serde(rename = "type")]
    cred_type: String,
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
/// or has an unsupported `type`.
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
    let raw = std::fs::read_to_string(path).map_err(|error| {
        FilterError::from(format!(
            "gcp_adc: failed to read credentials file '{}': {error}",
            path.display()
        ))
    })?;
    let parsed: GoogleApplicationCredentials = serde_json::from_str(&raw).map_err(|error| {
        FilterError::from(format!(
            "gcp_adc: failed to parse credentials file '{}': {error}",
            path.display()
        ))
    })?;
    match parsed.cred_type.as_str() {
        "service_account" => Ok(TokenSource::ServiceAccountKey),
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
