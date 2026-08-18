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
/// jwks_url: "http://keycloak:8080/realms/ai-gateway/protocol/openid-connect/certs"
/// issuer: "http://keycloak:8080/realms/ai-gateway"
/// audience: "praxis-gateway"
/// claim_headers:
///   preferred_username: "x-tenant-username"
///   groups: "x-tenant-group"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct JwtAuthConfig {
    /// URL of the JWKS endpoint (JSON Web Key Set).
    /// The filter fetches public keys from here to verify JWT signatures.
    pub jwks_url: String,

    /// Expected `iss` (issuer) claim. If set, tokens from other
    /// issuers are rejected.
    #[serde(default)]
    pub issuer: Option<String>,

    /// Expected `aud` (audience) claim. If set, tokens not intended
    /// for this audience are rejected.
    #[serde(default)]
    pub audience: Option<String>,

    /// Maps JWT claim names to request header names.
    /// The filter extracts these claims from verified tokens and
    /// injects them as request headers for downstream filters.
    #[serde(default)]
    pub claim_headers: std::collections::HashMap<String, String>,

    /// Header to read the bearer token from.
    #[serde(default = "default_token_header")]
    pub token_header: String,
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
    if config.claim_headers.is_empty() {
        return Err("jwt_auth: claim_headers must have at least one mapping".into());
    }
    Ok(())
}
