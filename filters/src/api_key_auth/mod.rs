// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! API key authentication filter: validates `sk-oai-*` keys against
//! an external key service (e.g. `maas-api`) and writes verified
//! identity to `filter_metadata`.
//!
//! The filter reads the API key from the configured request header,
//! validates it via an HTTP callout, caches the result, and injects
//! the verified identity for downstream filters (metering, audit).
//!
//! Works with any service implementing the validate contract:
//! `POST {validate_url}` with `{"key": "..."}` → `{"valid": true,
//! "username": "...", "groups": [...]}`.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests;

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tokio::sync::RwLock;
use tracing::debug;

use self::config::{ApiKeyAuthConfig, validate_config};

// -----------------------------------------------------------------------------
// CachedIdentity
// -----------------------------------------------------------------------------

/// Cached validation result for a previously-seen API key.
#[derive(Clone)]
struct CachedIdentity {
    /// Extracted username.
    username: String,

    /// Extracted group (comma-joined if multiple).
    group: String,

    /// Extracted subscription.
    subscription: String,

    /// Stable API-key ID returned by the validation service.
    key_id: Option<String>,

    /// When this entry was cached.
    cached_at: Instant,
}

// -----------------------------------------------------------------------------
// ValidationResponse
// -----------------------------------------------------------------------------

/// JSON response from the key validation endpoint.
#[derive(Debug, Deserialize)]
struct ValidationResponse {
    /// Whether the key is valid.
    valid: bool,

    /// Username associated with the key.
    #[serde(default)]
    username: String,

    /// Groups the user belongs to.
    #[serde(default)]
    groups: Vec<String>,

    /// Subscription name.
    #[serde(default)]
    subscription: String,

    /// Stable API-key ID returned by the validation service.
    #[serde(rename = "keyId", default)]
    key_id: Option<String>,
}

// -----------------------------------------------------------------------------
// ApiKeyAuthFilter
// -----------------------------------------------------------------------------

/// Validates API keys against an external service and injects
/// verified identity into `filter_metadata`.
///
/// # YAML configuration
///
/// ```yaml
/// filter: api_key_auth
/// validate_url: "http://maas-api:8080/internal/v1/api-keys/validate"
/// cache_ttl_seconds: 300
/// ```
pub struct ApiKeyAuthFilter {
    /// HTTP client for validation callouts.
    client: reqwest::Client,

    /// URL of the validation endpoint.
    validate_url: String,

    /// Header to read the API key from.
    token_header: String,

    /// Cache TTL.
    cache_ttl: Duration,

    /// In-memory cache of validated keys (key hash → identity).
    cache: Arc<RwLock<HashMap<[u8; 32], CachedIdentity>>>,

    /// Maximum number of validated identities retained in memory.
    cache_max_entries: usize,
}

impl ApiKeyAuthFilter {
    /// Parse from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config parsing or validation fails.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ApiKeyAuthConfig = parse_filter_config("api_key_auth", value)?;
        validate_config(&cfg).map_err(|e| -> FilterError { e.into() })?;

        // This filter owns a direct reqwest client rather than the shared
        // Praxis subrequest client, so install the process-wide rustls
        // provider at the same construction boundary.
        praxis_tls::provider::install();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_seconds))
            .build()
            .map_err(|e| -> FilterError { format!("api_key_auth: failed to build HTTP client: {e}").into() })?;

        Ok(Box::new(Self {
            client,
            validate_url: cfg.validate_url,
            token_header: cfg.token_header.to_lowercase(),
            cache_ttl: Duration::from_secs(cfg.cache_ttl_seconds),
            cache_max_entries: cfg.cache_max_entries,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }))
    }

    /// Digest the API key for cache lookup without retaining the raw key.
    fn hash_key(key: &str) -> [u8; 32] {
        Sha256::digest(key.as_bytes()).into()
    }

    /// Look up a cached identity, returning `None` if expired.
    async fn cache_get(&self, key_hash: [u8; 32]) -> Option<CachedIdentity> {
        let cache = self.cache.read().await;
        let entry = cache.get(&key_hash)?.clone();
        drop(cache);
        (entry.cached_at.elapsed() < self.cache_ttl).then_some(entry)
    }

    /// Store a validated identity in the cache.
    async fn cache_put(&self, key_hash: [u8; 32], identity: CachedIdentity) {
        let mut cache = self.cache.write().await;
        if !cache.contains_key(&key_hash)
            && cache.len() >= self.cache_max_entries
            && let Some(evicted) = cache.keys().next().copied()
        {
            cache.remove(&evicted);
        }
        cache.insert(key_hash, identity);
    }

    /// Call the validation endpoint.
    #[expect(
        clippy::cognitive_complexity,
        reason = "the validation callout has explicit fail-closed stages"
    )]
    async fn validate_key(&self, key: &str) -> Option<CachedIdentity> {
        let body = serde_json::json!({"key": key});

        let resp = self.client.post(&self.validate_url).json(&body).send().await.ok()?;

        if !resp.status().is_success() {
            debug!(status = %resp.status(), "validation endpoint error");
            return None;
        }

        let result: ValidationResponse = resp.json().await.ok()?;

        if !result.valid {
            debug!("key validation returned valid=false");
            return None;
        }

        if result.username.trim().is_empty() {
            debug!("key validation returned valid=true without a username");
            return None;
        }

        Some(CachedIdentity {
            username: result.username,
            group: result.groups.join(","),
            subscription: result.subscription,
            key_id: result.key_id.filter(|value| !value.trim().is_empty()),
            cached_at: Instant::now(),
        })
    }
}

#[async_trait]
impl HttpFilter for ApiKeyAuthFilter {
    fn name(&self) -> &'static str {
        "api_key_auth"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // 1. Extract the API key.
        let Some(key_value) = ctx.request.headers.get(&*self.token_header) else {
            debug!("no API key header found");
            return Ok(reject_unauthorized("missing API key"));
        };
        let raw = key_value.to_str().unwrap_or_default();
        let key = raw.strip_prefix("Bearer ").unwrap_or(raw);
        if key.is_empty() {
            return Ok(reject_unauthorized("empty API key"));
        }

        let key_hash = Self::hash_key(key);

        // 2. Check cache.
        let identity = if let Some(cached) = self.cache_get(key_hash).await {
            debug!(username = %cached.username, "cache hit");
            cached
        } else {
            // 3. Cache miss — call validation endpoint.
            let Some(validated) = self.validate_key(key).await else {
                return Ok(reject_unauthorized("invalid API key"));
            };
            debug!(username = %validated.username, "key validated");
            self.cache_put(key_hash, validated.clone()).await;
            validated
        };

        // 4. Strip the API key header.
        if let Ok(name) = http::HeaderName::from_bytes(self.token_header.as_bytes()) {
            ctx.request_headers_to_remove.push(name);
        }

        // 5. Write identity to filter_metadata.
        set_identity_metadata(ctx, identity);

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Copy the validated identity into filter metadata for downstream filters.
fn set_identity_metadata(ctx: &mut HttpFilterContext<'_>, identity: CachedIdentity) {
    ctx.set_metadata("x-tenant-username", identity.username);
    ctx.set_metadata("x-tenant-group", identity.group);
    if !identity.subscription.is_empty() {
        ctx.set_metadata("x-tenant-subscription", identity.subscription);
    }
    if let Some(key_id) = identity.key_id {
        ctx.set_metadata("api_key.id", key_id);
    }
}

/// Reject with 401 Unauthorized.
fn reject_unauthorized(message: &'static str) -> FilterAction {
    FilterAction::Reject(Rejection {
        body: Some(Bytes::from(message)),
        header_map: None,
        headers: vec![("WWW-Authenticate".to_owned(), "ApiKey".to_owned())],
        preserve_keepalive: false,
        status: 401,
    })
}
