// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Credential-source resolution for [`GcpAdcFilter`].
//!
//! Token HTTP fetch (metadata server, `JWT` bearer) is added in follow-up
//! PRs. This module only classifies the source at construct time.

use std::path::Path;

use praxis_filter::FilterError;
use serde::Deserialize;

use super::config::{GcpAdcConfig, GcpAdcSource};

// -----------------------------------------------------------------------------
// TokenSource
// -----------------------------------------------------------------------------

/// Resolved credential source used by the background refresher.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum TokenSource {
    /// GCE/GKE/Cloud Run metadata server.
    Metadata {
        /// Service account email or `default`.
        service_account: String,
    },

    /// Parsed `type: service_account` key file. Fetch is not implemented
    /// in this skeleton; the cache stays empty and requests fail closed.
    ServiceAccountKey,
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
