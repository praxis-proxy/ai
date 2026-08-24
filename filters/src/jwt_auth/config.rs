// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the JWT authentication filter.

use serde::Deserialize;

// -----------------------------------------------------------------------------
// JwtAuthConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the JWT auth filter.
///
/// ```yaml
/// filter: jwt_auth
/// jwks_url: "https://keycloak.example.com/realms/ai-gateway/protocol/openid-connect/certs"
/// issuer: "https://keycloak.example.com/realms/ai-gateway"
/// audience: "praxis-gateway"
/// claim_metadata:
///   preferred_username: "x-tenant-username"
///   groups: "x-tenant-group"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct JwtAuthConfig {
    /// URL of the JWKS endpoint (JSON Web Key Set).
    /// The filter fetches public keys from here to verify JWT signatures.
    ///
    /// This channel is the filter's root of trust — it should be
    /// `https://` in any deployment where the proxy-to-`IdP` path is
    /// not already authenticated. A plain `http://` URL (other than
    /// loopback) is accepted but logs a warning at startup.
    pub jwks_url: String,

    /// Expected `iss` (issuer) claim. If set, tokens from other
    /// issuers are rejected.
    #[serde(default)]
    pub issuer: Option<String>,

    /// Expected `aud` (audience) claim. If set, tokens not intended
    /// for this audience are rejected.
    #[serde(default)]
    pub audience: Option<String>,

    /// Maps JWT claim names to `filter_metadata` keys.
    ///
    /// The filter extracts these claims from verified tokens and
    /// writes them to `filter_metadata` for downstream filters
    /// (e.g. `external_metering`). Claims are intentionally NOT
    /// injected as upstream request headers — see the filter docs
    /// for the trusted-channel rationale.
    #[serde(default)]
    pub claim_metadata: std::collections::HashMap<String, String>,

    /// Header to read the bearer token from.
    #[serde(default = "default_token_header")]
    pub token_header: String,

    /// Skip TLS certificate verification when fetching JWKS.
    ///
    /// Defaults to `false` (certificates are verified). Enable only
    /// for in-cluster `IdP`s with self-signed certificates that are
    /// otherwise trusted. Enabling it logs a warning, because the
    /// JWKS fetch is the filter's root of trust and disabling
    /// verification exposes it to MITM key substitution.
    #[serde(default)]
    pub insecure_skip_tls_verify: bool,
}

/// Returns the default bearer token header name (`authorization`).
fn default_token_header() -> String {
    "authorization".to_owned()
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate a [`JwtAuthConfig`], returning an error on missing required fields.
pub(super) fn validate_config(config: &JwtAuthConfig) -> Result<(), String> {
    if config.jwks_url.is_empty() {
        return Err("jwt_auth: jwks_url must not be empty".into());
    }
    if config.claim_metadata.is_empty() {
        return Err("jwt_auth: claim_metadata must have at least one mapping".into());
    }
    Ok(())
}
