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
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, warn};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Cooldown after a *successful* refresh, and the ceiling for the
/// failure backoff. Bounds unknown-`kid` refetch abuse when the `IdP`
/// is healthy.
const REFRESH_COOLDOWN: Duration = Duration::from_secs(30);

/// First backoff after a *failed* refresh. Subsequent consecutive
/// failures double it, capped at [`REFRESH_COOLDOWN`], so a transient
/// `IdP` blip recovers in ~1s instead of blocking auth for a full 30s.
const FAILURE_BACKOFF_INITIAL: Duration = Duration::from_secs(1);

/// Default TTL for cached keys.
const DEFAULT_TTL: Duration = Duration::from_secs(300); // 5 minutes

/// Maximum response size for the JWKS endpoint.
const MAX_JWKS_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// JwksCache
// -----------------------------------------------------------------------------

/// Thread-safe cache of JWKS decoding keys.
pub(super) struct JwksCache {
    /// Cached keys indexed by `kid`.
    keys: Arc<RwLock<CachedKeys>>,

    /// Serializes refresh attempts so concurrent cache misses trigger
    /// a single fetch (single-flight) rather than a stampede.
    refresh_lock: Mutex<()>,

    /// HTTP client for fetching JWKS.
    client: reqwest::Client,

    /// JWKS endpoint URL.
    url: String,
}

/// Cached JWKS decoding keys and their refresh timestamp.
struct CachedKeys {
    /// Decoding keys by `kid`, each with the algorithm family that
    /// key may be used with.
    by_kid: HashMap<String, (DecodingKey, Vec<Algorithm>)>,

    /// When the cache was last refreshed. `None` means never
    /// refreshed — forces immediate fetch on first request.
    last_refresh: Option<Instant>,

    /// Cooldown that must elapse since `last_refresh` before another
    /// refresh is attempted. `REFRESH_COOLDOWN` after a success; a
    /// short exponential backoff after a failure.
    retry_after: Duration,

    /// Consecutive failed refreshes, used to grow `retry_after`.
    consecutive_failures: u32,
}

impl JwksCache {
    /// Create a new cache. Keys are fetched lazily on the first
    /// request rather than at construction, because the filter
    /// is built during config parsing before the async runtime
    /// is fully available.
    ///
    /// TLS certificates are verified by default. `insecure_skip_tls_verify`
    /// disables verification for in-cluster `IdP`s with self-signed
    /// certs — it logs a warning because the JWKS fetch is the
    /// filter's root of trust.
    pub(super) fn new(url: String, insecure_skip_tls_verify: bool) -> Result<Self, String> {
        if insecure_skip_tls_verify {
            warn!(
                %url,
                "jwt_auth: TLS certificate verification DISABLED for JWKS fetch \
                 (insecure_skip_tls_verify=true) — vulnerable to MITM key substitution"
            );
        } else if !is_authenticated_url(&url) {
            warn!(
                %url,
                "jwt_auth: JWKS URL is not https:// and not loopback — key delivery is \
                 unauthenticated and vulnerable to MITM key substitution"
            );
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .danger_accept_invalid_certs(insecure_skip_tls_verify)
            .build()
            .map_err(|e| format!("jwt_auth: failed to build HTTP client: {e}"))?;

        Ok(Self {
            keys: Arc::new(RwLock::new(CachedKeys {
                by_kid: HashMap::new(),
                last_refresh: None,
                retry_after: REFRESH_COOLDOWN,
                consecutive_failures: 0,
            })),
            refresh_lock: Mutex::new(()),
            client,
            url,
        })
    }

    /// Look up a decoding key by `kid`. Refreshes when:
    /// - the `kid` is unknown and the cooldown has elapsed, or
    /// - the cache TTL has expired.
    ///
    /// A failed refresh backs off exponentially (starting at
    /// [`FAILURE_BACKOFF_INITIAL`], capped at [`REFRESH_COOLDOWN`]), so a
    /// transient `IdP` blip is retried within ~1s rather than blocking
    /// auth for the full success cooldown, while a sustained outage is
    /// not hammered.
    ///
    /// If a refresh fails (e.g. the `IdP` is unreachable), the last
    /// good cache is served past its TTL rather than failing closed —
    /// a stale-if-error availability tradeoff. This means a revoked
    /// key can keep validating for as long as the `IdP` stays down;
    /// tokens still expire independently via `exp`.
    pub(super) async fn get_key(&self, kid: &str) -> Option<(DecodingKey, Vec<Algorithm>)> {
        // Fast path: key is cached and TTL hasn't expired.
        {
            let keys = self.keys.read().await;
            let ttl_ok = keys.last_refresh.is_some_and(|t| t.elapsed() < DEFAULT_TTL);
            if ttl_ok && let Some(entry) = keys.by_kid.get(kid) {
                return Some(entry.clone());
            }
        }

        // Cooldown gate (lock-free): if we attempted a refresh within
        // the current cooldown, serve whatever is cached instead of
        // refetching. The cooldown is REFRESH_COOLDOWN after a success
        // (bounds unknown-kid abuse) or a short backoff after a failure
        // (fast recovery from a transient IdP blip).
        {
            let keys = self.keys.read().await;
            if keys.last_refresh.is_some_and(|t| t.elapsed() < keys.retry_after) {
                return keys.by_kid.get(kid).cloned();
            }
        }

        // Single-flight: only one task fetches at a time. Waiters
        // re-check the cooldown after acquiring the lock — if the
        // winner already refreshed, they use its result.
        let _guard = self.refresh_lock.lock().await;
        {
            let keys = self.keys.read().await;
            if keys.last_refresh.is_some_and(|t| t.elapsed() < keys.retry_after) {
                return keys.by_kid.get(kid).cloned();
            }
        }

        debug!(kid, "refreshing JWKS");
        match self.refresh().await {
            Ok(()) => {
                // Success resets the backoff to the full cooldown.
                let mut keys = self.keys.write().await;
                keys.consecutive_failures = 0;
                keys.retry_after = REFRESH_COOLDOWN;
                keys.by_kid.get(kid).cloned()
            },
            Err(e) => {
                warn!("JWKS refresh failed: {e}");
                // Grow the backoff (capped) so a transient failure is
                // retried quickly but a sustained outage isn't hammered.
                let mut keys = self.keys.write().await;
                keys.consecutive_failures = keys.consecutive_failures.saturating_add(1);
                let shift = (keys.consecutive_failures - 1).min(5);
                keys.retry_after = FAILURE_BACKOFF_INITIAL.saturating_mul(1 << shift).min(REFRESH_COOLDOWN);
                // Serve stale cache on failure (stale-if-error).
                keys.by_kid.get(kid).cloned()
            },
        }
    }

    /// Fetch JWKS from the endpoint and update the cache.
    ///
    /// Updates `last_refresh` on both success and failure so the
    /// cooldown prevents stampede when the `IdP` is down. Callers
    /// hold `refresh_lock`, so only one fetch runs at a time.
    #[expect(
        clippy::too_many_lines,
        reason = "fetch-validate-parse-update pipeline with error handling at each step"
    )]
    async fn refresh(&self) -> Result<(), String> {
        // Update timestamp first so concurrent callers see the
        // cooldown immediately, even if the fetch fails.
        {
            let mut keys = self.keys.write().await;
            keys.last_refresh = Some(Instant::now());
        }

        let mut resp = self
            .client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| format!("JWKS fetch failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("JWKS endpoint returned {}", resp.status()));
        }

        // Reject oversized responses by the advertised length before
        // reading a single byte, when the server provides it.
        if let Some(len) = resp.content_length()
            && len > MAX_JWKS_BYTES as u64
        {
            return Err(format!("JWKS response too large: {len} bytes (Content-Length)"));
        }

        // Read chunked with a running cap so a chunked/unbounded
        // response can't exhaust memory even without Content-Length.
        let mut body = Vec::new();
        while let Some(chunk) = resp.chunk().await.map_err(|e| format!("JWKS read failed: {e}"))? {
            if body.len() + chunk.len() > MAX_JWKS_BYTES {
                return Err(format!("JWKS response exceeds {MAX_JWKS_BYTES} bytes"));
            }
            body.extend_from_slice(&chunk);
        }

        let jwk_set: JwkSet = serde_json::from_slice(&body).map_err(|e| format!("JWKS parse failed: {e}"))?;

        let mut by_kid = HashMap::new();
        for jwk in &jwk_set.keys {
            let Some(kid) = jwk.common.key_id.as_ref() else {
                continue;
            };
            let Some(algorithms) = algorithms_for(jwk) else {
                continue;
            };

            match DecodingKey::from_jwk(jwk) {
                Ok(key) => {
                    by_kid.insert(kid.clone(), (key, algorithms));
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

/// Determine the algorithm(s) a JWK may be used to verify.
///
/// When the JWK declares an `alg`, that single algorithm is pinned.
/// When it omits `alg` (e.g. Azure AD), the whole family for the key
/// type is allowed — same key, so no algorithm-confusion risk, and it
/// avoids false 401s when the `IdP` signs with e.g. RS512.
fn algorithms_for(jwk: &jsonwebtoken::jwk::Jwk) -> Option<Vec<Algorithm>> {
    use jsonwebtoken::jwk::{AlgorithmParameters, KeyAlgorithm};

    match jwk.common.key_algorithm {
        Some(KeyAlgorithm::RS256) => Some(vec![Algorithm::RS256]),
        Some(KeyAlgorithm::RS384) => Some(vec![Algorithm::RS384]),
        Some(KeyAlgorithm::RS512) => Some(vec![Algorithm::RS512]),
        Some(KeyAlgorithm::ES256) => Some(vec![Algorithm::ES256]),
        Some(KeyAlgorithm::ES384) => Some(vec![Algorithm::ES384]),
        // No `alg` declared — allow the full family for the key type.
        None => match &jwk.algorithm {
            AlgorithmParameters::RSA(_) => Some(vec![Algorithm::RS256, Algorithm::RS384, Algorithm::RS512]),
            AlgorithmParameters::EllipticCurve(_) => Some(vec![Algorithm::ES256, Algorithm::ES384]),
            _ => None,
        },
        _ => None,
    }
}

/// Whether the JWKS URL delivers keys over an authenticated channel:
/// `https://`, or `http://` to a loopback host (in-cluster sidecar).
fn is_authenticated_url(url: &str) -> bool {
    match reqwest::Url::parse(url) {
        Ok(u) => {
            u.scheme() == "https"
                // url::host_str() returns IPv6 hosts bracketed, so match "[::1]".
                || (u.scheme() == "http" && matches!(u.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")))
        },
        // Unparseable URL: treat as unauthenticated so we warn.
        Err(_) => false,
    }
}
