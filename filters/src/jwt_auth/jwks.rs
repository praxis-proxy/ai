// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! JWKS (JSON Web Key Set) fetching and caching.
//!
//! Downloads public keys from the `IdP`'s JWKS endpoint, caches them
//! by `kid` (Key ID), and provides lookup for JWT verification.
//! Keys are refreshed when an unknown `kid` is encountered (with
//! a cooldown to prevent abuse) or when the cache TTL expires.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use jsonwebtoken::{Algorithm, DecodingKey, jwk::JwkSet};
use tokio::sync::RwLock;
use tracing::{debug, warn};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Minimum interval between JWKS refresh attempts.
const REFRESH_COOLDOWN: Duration = Duration::from_secs(30);

/// Default TTL for cached keys.
const DEFAULT_TTL: Duration = Duration::from_secs(300); // 5 minutes

/// Maximum response size for JWKS endpoint.
const MAX_JWKS_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// JwksCache
// -----------------------------------------------------------------------------

/// Thread-safe cache of JWKS decoding keys.
pub(super) struct JwksCache {
    /// Cached keys indexed by `kid`.
    keys: Arc<RwLock<CachedKeys>>,

    /// HTTP client for fetching JWKS.
    client: reqwest::Client,

    /// JWKS endpoint URL.
    url: String,
}

/// Cached JWKS decoding keys and their refresh timestamp.
struct CachedKeys {
    /// Decoding keys by `kid`.
    by_kid: HashMap<String, (DecodingKey, Algorithm)>,

    /// When the cache was last refreshed. `None` means never
    /// refreshed — forces immediate fetch on first request.
    last_refresh: Option<Instant>,
}

impl JwksCache {
    /// Create a new cache. Keys are fetched lazily on the first
    /// request rather than at construction, because the filter
    /// is built during config parsing before the async runtime
    /// is fully available.
    pub(super) fn new(url: String) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            // Accept self-signed certs for JWKS endpoints.
            // In production, the IdP should have a valid cert
            // and this should be removed or made configurable.
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("jwt_auth: failed to build HTTP client: {e}"))?;

        Ok(Self {
            keys: Arc::new(RwLock::new(CachedKeys {
                by_kid: HashMap::new(),
                last_refresh: None,
            })),
            client,
            url,
        })
    }

    /// Look up a decoding key by `kid`. Refreshes when:
    /// - the `kid` is unknown and the cooldown has elapsed, or
    /// - the cache TTL has expired (ensures revoked keys stop working within one TTL window).
    pub(super) async fn get_key(&self, kid: &str) -> Option<(DecodingKey, Algorithm)> {
        // Fast path: key is cached and TTL hasn't expired.
        {
            let keys = self.keys.read().await;
            let ttl_ok = keys.last_refresh.is_some_and(|t| t.elapsed() < DEFAULT_TTL);
            if ttl_ok && let Some(entry) = keys.by_kid.get(kid) {
                return Some(entry.clone());
            }
        }

        // Need refresh — either TTL expired, never fetched, or
        // unknown kid. All paths share the same cooldown to
        // prevent stampede when the IdP is down.
        {
            let keys = self.keys.read().await;
            let cooldown_active = keys.last_refresh.is_some_and(|t| t.elapsed() < REFRESH_COOLDOWN);
            if cooldown_active {
                return keys.by_kid.get(kid).cloned();
            }
        }

        debug!(kid, "refreshing JWKS");
        if let Err(e) = self.refresh().await {
            warn!("JWKS refresh failed: {e}");
            // Use stale cache on failure.
            let keys = self.keys.read().await;
            return keys.by_kid.get(kid).cloned();
        }

        let keys = self.keys.read().await;
        keys.by_kid.get(kid).cloned()
    }

    /// Fetch JWKS from the endpoint and update the cache.
    ///
    /// Updates `last_refresh` on both success and failure so the
    /// cooldown prevents stampede when the `IdP` is down.
    #[expect(
        clippy::too_many_lines,
        reason = "fetch-parse-update pipeline with error handling at each step"
    )]
    async fn refresh(&self) -> Result<(), String> {
        // Update timestamp first so concurrent callers see the
        // cooldown immediately, even if the fetch fails.
        {
            let mut keys = self.keys.write().await;
            keys.last_refresh = Some(Instant::now());
        }

        let resp = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| format!("JWKS fetch failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("JWKS endpoint returned {}", resp.status()));
        }

        let body = resp.bytes().await.map_err(|e| format!("JWKS read failed: {e}"))?;

        if body.len() > MAX_JWKS_BYTES {
            return Err(format!("JWKS response too large: {} bytes", body.len()));
        }

        let jwk_set: JwkSet = serde_json::from_slice(&body).map_err(|e| format!("JWKS parse failed: {e}"))?;

        let mut by_kid = HashMap::new();
        for jwk in &jwk_set.keys {
            let Some(kid) = jwk.common.key_id.as_ref() else {
                continue;
            };
            let algorithm = match jwk.common.key_algorithm.as_ref() {
                Some(jsonwebtoken::jwk::KeyAlgorithm::RS256) => Algorithm::RS256,
                Some(jsonwebtoken::jwk::KeyAlgorithm::RS384) => Algorithm::RS384,
                Some(jsonwebtoken::jwk::KeyAlgorithm::RS512) => Algorithm::RS512,
                Some(jsonwebtoken::jwk::KeyAlgorithm::ES256) => Algorithm::ES256,
                Some(jsonwebtoken::jwk::KeyAlgorithm::ES384) => Algorithm::ES384,
                // Azure AD omits `alg` — infer from key type.
                None => match &jwk.algorithm {
                    jsonwebtoken::jwk::AlgorithmParameters::RSA(_) => Algorithm::RS256,
                    jsonwebtoken::jwk::AlgorithmParameters::EllipticCurve(_) => Algorithm::ES256,
                    _ => continue,
                },
                _ => continue,
            };

            match DecodingKey::from_jwk(jwk) {
                Ok(key) => {
                    by_kid.insert(kid.clone(), (key, algorithm));
                },
                Err(e) => {
                    warn!(kid, "failed to parse JWK: {e}");
                },
            }
        }

        debug!(count = by_kid.len(), "JWKS cache refreshed");

        let mut keys = self.keys.write().await;
        keys.by_kid = by_kid;
        keys.last_refresh = Some(Instant::now());
        drop(keys);

        Ok(())
    }
}
