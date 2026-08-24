// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! JWT authentication filter: validates bearer tokens against a
//! JWKS endpoint and writes verified claims to `filter_metadata`.
//!
//! The filter downloads public keys from the `IdP`'s JWKS endpoint
//! lazily on the first request (and refreshes on unknown `kid` or
//! TTL expiry), validates the JWT signature locally (no per-request
//! callout), and writes configured claims to `filter_metadata` for
//! downstream filters. Claims are deliberately not injected as
//! upstream request headers — see [`JwtAuthFilter`] for why.
//!
//! Works with any OIDC-compliant identity provider (Keycloak,
//! Okta, Azure AD, etc.) that publishes a JWKS endpoint.
//!
//! # Temporary bridge
//!
//! This filter is a self-contained bridge for JWT/OIDC-fronted
//! deployments. JWT validation is expected to move into the core
//! CPEX policy engine; this filter should be retired once that
//! lands (tracked in #708). No other filter should depend on it.

mod config;
mod jwks;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use jsonwebtoken::{TokenData, Validation, decode};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use tracing::debug;

use self::{
    config::{JwtAuthConfig, validate_config},
    jwks::JwksCache,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Bearer token prefix (case-insensitive match).
const BEARER_PREFIX: &str = "bearer ";

// -----------------------------------------------------------------------------
// JwtAuthFilter
// -----------------------------------------------------------------------------

/// Validates JWT bearer tokens against a JWKS endpoint and writes
/// verified claims to `filter_metadata`.
///
/// # How it works
///
/// 1. Extracts the bearer token from the configured header
/// 2. Decodes the JWT header to find the `kid` (key ID)
/// 3. Looks up the public key in the JWKS cache (refreshes if unknown)
/// 4. Validates the signature, expiry, `nbf`, issuer, and audience
/// 5. Writes configured claims to `filter_metadata`
/// 6. Strips the token header so it never reaches the upstream
/// 7. Rejects with 401 if any step fails
///
/// Tokens without a `kid` header are rejected — the JWKS lookup is
/// keyed by `kid`, so single-key `IdP`s that omit it are not supported.
///
/// # Claims go to metadata, not headers
///
/// Verified claims are written to `filter_metadata`, not to upstream
/// request headers. Header injection would happen after
/// `request_headers_to_remove` is applied, so `identity_header_guard`
/// could not strip them and they would leak to the upstream provider.
/// `filter_metadata` is the trusted channel downstream filters read.
///
/// # YAML configuration
///
/// ```yaml
/// filter: jwt_auth
/// jwks_url: "https://keycloak.example.com/realms/ai-gateway/protocol/openid-connect/certs"
/// issuer: "https://keycloak.example.com/realms/ai-gateway"
/// claim_metadata:
///   preferred_username: "x-tenant-username"
///   groups: "x-tenant-group"
/// ```
pub struct JwtAuthFilter {
    /// Cached JWKS keys for signature verification.
    jwks: JwksCache,

    /// Expected issuer (`iss` claim).
    issuer: Option<String>,

    /// Expected audience (`aud` claim).
    audience: Option<String>,

    /// Maps claim names to `filter_metadata` keys.
    claim_metadata: Vec<(String, String)>,

    /// Header to read the bearer token from.
    token_header: String,
}

impl JwtAuthFilter {
    /// Parse from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    /// JWKS keys are fetched lazily on the first request, so a
    /// misconfigured or unreachable endpoint surfaces as 401s at
    /// request time, not a construction error.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: JwtAuthConfig = parse_filter_config("jwt_auth", value)?;
        validate_config(&config).map_err(|e| -> FilterError { e.into() })?;

        let jwks_url = config.jwks_url.clone();
        let claim_metadata: Vec<(String, String)> = config.claim_metadata.into_iter().collect();

        let jwks =
            JwksCache::new(jwks_url, config.insecure_skip_tls_verify).map_err(|e| -> FilterError { e.into() })?;

        Ok(Box::new(Self {
            jwks,
            issuer: config.issuer,
            audience: config.audience,
            claim_metadata,
            token_header: config.token_header.to_lowercase(),
        }))
    }

    /// Extract the JWT from the configured header.
    ///
    /// Handles both `Authorization: Bearer <token>` and raw
    /// `x-api-key: <token>` formats. Strips the "Bearer " prefix
    /// if present, otherwise uses the raw value.
    fn extract_token<'a>(&self, ctx: &'a HttpFilterContext<'_>) -> Option<&'a str> {
        let value = ctx.request.headers.get(&*self.token_header)?;
        let value_str = value.to_str().ok()?;

        if value_str.len() > BEARER_PREFIX.len()
            && value_str
                .get(..BEARER_PREFIX.len())
                .is_some_and(|p| p.eq_ignore_ascii_case(BEARER_PREFIX))
        {
            value_str.get(BEARER_PREFIX.len()..)
        } else if !value_str.is_empty() {
            Some(value_str)
        } else {
            None
        }
    }

    /// Verify a token's signature and claims, returning its claims.
    ///
    /// The accepted algorithms come from the JWKS entry, never from
    /// the token header, so an attacker cannot downgrade to `none`
    /// or force HS256 with the public key as an HMAC secret.
    ///
    /// Kept separate from `on_request` so the sizable `Validation`
    /// and `TokenData` values stay off the request handler's stack
    /// frame.
    fn verify_claims(
        &self,
        token: &str,
        decoding_key: &jsonwebtoken::DecodingKey,
        algorithms: Vec<jsonwebtoken::Algorithm>,
    ) -> Result<serde_json::Value, jsonwebtoken::errors::Error> {
        let Some(first_alg) = algorithms.first().copied() else {
            return Err(jsonwebtoken::errors::ErrorKind::InvalidAlgorithm.into());
        };
        let mut validation = Validation::new(first_alg);
        validation.algorithms = algorithms;
        validation.validate_exp = true;
        validation.validate_nbf = true;
        // Disable the default audience requirement — only enforce
        // when explicitly configured.
        validation.set_audience::<&str>(&[]);
        validation.validate_aud = false;

        if let Some(iss) = &self.issuer {
            validation.set_issuer(&[iss]);
        }
        if let Some(aud) = &self.audience {
            validation.set_audience(&[aud]);
            validation.validate_aud = true;
        }

        let token_data: TokenData<serde_json::Value> = decode(token, decoding_key, &validation)?;
        Ok(token_data.claims)
    }
}

#[async_trait]
impl HttpFilter for JwtAuthFilter {
    fn name(&self) -> &'static str {
        "jwt_auth"
    }

    #[expect(
        clippy::too_many_lines,
        reason = "sequential validation pipeline with early-return branches"
    )]
    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // 1. Extract the bearer token
        let Some(token) = self.extract_token(ctx) else {
            debug!("no bearer token found, rejecting");
            return Ok(reject_unauthorized("missing or malformed bearer token"));
        };

        // 2. Decode the JWT header to get the kid
        let header = match jsonwebtoken::decode_header(token) {
            Ok(h) => h,
            Err(e) => {
                debug!("invalid JWT header: {e}");
                return Ok(reject_unauthorized("invalid token"));
            },
        };

        let Some(kid) = header.kid.as_deref() else {
            debug!("JWT missing kid");
            return Ok(reject_unauthorized("invalid token"));
        };

        // 3. Look up the public key
        let Some((decoding_key, algorithms)) = self.jwks.get_key(kid).await else {
            debug!(kid, "unknown signing key");
            return Ok(reject_unauthorized("invalid token"));
        };

        // 4. Verify signature, expiry, nbf, issuer, and audience. Extracted to a helper so the large `Validation` and
        //    `TokenData` values live in their own stack frame.
        let claims = match self.verify_claims(token, &decoding_key, algorithms) {
            Ok(claims) => claims,
            Err(e) => {
                debug!("JWT validation failed: {e}");
                return Ok(reject_unauthorized("invalid token"));
            },
        };

        // 5. Strip the token header so the JWT doesn't leak to the upstream provider. credential_injection will add the
        //    real provider key later.
        if let Ok(name) = http::HeaderName::from_bytes(self.token_header.as_bytes()) {
            ctx.request_headers_to_remove.push(name);
        }

        // 6. Extract claims to filter_metadata only.
        //
        //    Identity is NOT injected into extra_request_headers
        //    because those are added to the upstream request after
        //    request_headers_to_remove is applied — meaning the
        //    identity_header_guard cannot strip them, and they'd
        //    leak to the upstream provider.
        //
        //    Downstream filters (external_metering) read identity
        //    from filter_metadata, which is the trusted channel.
        for (claim_name, metadata_key) in &self.claim_metadata {
            if let Some(value) = claims.get(claim_name) {
                let metadata_value = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(arr) => {
                        let parts: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                        parts.join(",")
                    },
                    other => other.to_string(),
                };

                ctx.set_metadata(metadata_key.clone(), metadata_value);
            }
        }

        debug!(
            username = claims
                .get("preferred_username")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown"),
            "JWT validated"
        );

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Build a 401 rejection with a `WWW-Authenticate: Bearer` header.
fn reject_unauthorized(message: &'static str) -> FilterAction {
    FilterAction::Reject(Rejection {
        status: 401,
        body: Some(Bytes::from(message)),
        headers: vec![("WWW-Authenticate".to_owned(), "Bearer".to_owned())],
        header_map: None,
        preserve_keepalive: false,
    })
}
