// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the GCP ADC filter.

use praxis_filter::FilterError;
#[cfg(test)]
use praxis_filter::parse_filter_config;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// GcpAdcSource
// -----------------------------------------------------------------------------

/// Where GCP credentials are obtained from.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum GcpAdcSource {
    /// Application Default Credentials: `GOOGLE_APPLICATION_CREDENTIALS` if
    /// set, otherwise the GCE/GKE metadata server.
    #[default]
    Adc,

    /// GCE/GKE/Cloud Run metadata server. Ignores
    /// `GOOGLE_APPLICATION_CREDENTIALS`.
    Metadata,

    /// Service account key JSON file at [`GcpAdcConfig::credentials_file`].
    KeyFile,
}

// -----------------------------------------------------------------------------
// GcpAdcConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `gcp_adc` filter.
///
/// ```yaml
/// filter: gcp_adc
/// source: adc
/// scope: https://www.googleapis.com/auth/cloud-platform
/// refresh_ratio: 0.75
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GcpAdcConfig {
    /// Credential source. Defaults to `adc`.
    #[serde(default)]
    pub source: GcpAdcSource,

    /// `OAuth2` scope requested with the access token.
    #[serde(default = "default_scope")]
    pub scope: String,

    /// Metadata service account email, or `default` (the default). Only
    /// used with the `metadata` and `adc` sources; rejected for
    /// `key_file`. Interpolated into the metadata path, so it must be
    /// `default` or a service-account email (letters, digits, `@`, `.`,
    /// `-`, `_`).
    #[serde(default)]
    pub service_account: Option<String>,

    /// Path to a service-account key JSON file. Required when `source` is
    /// `key_file`; rejected for the other sources (`adc` reads
    /// `GOOGLE_APPLICATION_CREDENTIALS` instead).
    #[serde(default)]
    pub credentials_file: Option<String>,

    /// Fraction of a token's TTL at which to refresh it. Must be in the
    /// open interval `(0, 1)`.
    #[serde(default = "default_refresh_ratio")]
    pub refresh_ratio: f64,
}

/// Default `OAuth2` scope for Vertex AI and other GCP APIs.
fn default_scope() -> String {
    "https://www.googleapis.com/auth/cloud-platform".to_owned()
}

/// Default value for [`GcpAdcConfig::refresh_ratio`].
fn default_refresh_ratio() -> f64 {
    0.75
}

/// Parse and validate the `gcp_adc` filter's YAML config.
///
/// # Errors
///
/// Returns [`FilterError`] if the YAML is malformed or has unknown fields.
#[cfg(test)]
pub(super) fn parse_gcp_adc_config(config: &serde_yaml::Value) -> Result<GcpAdcConfig, FilterError> {
    parse_filter_config("gcp_adc", config)
}

/// Validate fields [`GcpAdcFilter::new`] relies on before resolving
/// credentials. Fields a source does not use are rejected rather than
/// silently ignored, so an auth misconfiguration fails loudly at startup.
///
/// [`GcpAdcFilter::new`]: super::filter::GcpAdcFilter
pub(super) fn validate_config(config: &GcpAdcConfig) -> Result<(), FilterError> {
    if !(config.refresh_ratio > 0.0 && config.refresh_ratio < 1.0) {
        return Err(format!(
            "gcp_adc: refresh_ratio must be between 0 and 1 (exclusive), got {}",
            config.refresh_ratio
        )
        .into());
    }
    if config.scope.is_empty() {
        return Err("gcp_adc: scope must not be empty".into());
    }
    match config.source {
        GcpAdcSource::KeyFile => validate_key_file_fields(config),
        GcpAdcSource::Metadata | GcpAdcSource::Adc => validate_metadata_fields(config),
    }
}

/// `key_file` requires `credentials_file` and does not use
/// `service_account`.
fn validate_key_file_fields(config: &GcpAdcConfig) -> Result<(), FilterError> {
    match config.credentials_file.as_deref() {
        Some(path) if !path.is_empty() => {},
        _ => {
            return Err("gcp_adc: source key_file requires credentials_file".into());
        },
    }
    if config.service_account.is_some() {
        return Err(
            "gcp_adc: service_account is only used with the metadata and adc sources; \
                    remove it for source key_file"
                .into(),
        );
    }
    Ok(())
}

/// `metadata` and `adc` use `service_account` and do not read
/// `credentials_file` (`adc` reads `GOOGLE_APPLICATION_CREDENTIALS`).
fn validate_metadata_fields(config: &GcpAdcConfig) -> Result<(), FilterError> {
    if config.credentials_file.is_some() {
        return Err(
            "gcp_adc: credentials_file is only used with source key_file (the adc source \
                    reads GOOGLE_APPLICATION_CREDENTIALS); remove it or set source: key_file"
                .into(),
        );
    }
    if let Some(service_account) = config.service_account.as_deref() {
        validate_service_account(service_account)?;
    }
    Ok(())
}

/// Accept only a `service_account` value that cannot alter the metadata
/// URL path (`/computeMetadata/v1/instance/service-accounts/{sa}/token`):
/// `default` or a service-account email, i.e. letters, digits, and
/// `@` `.` `-` `_`.
pub(super) fn validate_service_account(value: &str) -> Result<(), FilterError> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || matches!(c, '@' | '.' | '-' | '_');
    if value.is_empty() || !value.chars().all(allowed) {
        return Err(format!(
            "gcp_adc: service_account '{value}' is invalid: it must be 'default' or a \
             service-account email (letters, digits, '@', '.', '-', '_')"
        )
        .into());
    }
    Ok(())
}
