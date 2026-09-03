// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for protocol-neutral web-search providers.

use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;

use crate::callout_policy;

/// Default callout timeout (10 seconds — search APIs can be slow).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

// -----------------------------------------------------------------------------
// SearchProvider
// -----------------------------------------------------------------------------

/// Supported search backend providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SearchProvider {
    /// Brave Search API.
    Brave,
    /// Tavily Search API.
    Tavily,
    /// You.com Search API.
    You,
}

impl SearchProvider {
    /// Provider name for logging and diagnostics.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Brave => "brave",
            Self::Tavily => "tavily",
            Self::You => "you",
        }
    }
}

// -----------------------------------------------------------------------------
// SearchContextSize
// -----------------------------------------------------------------------------

/// Controls how much surrounding context to include with results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchContextSize {
    /// Minimal context — fewer results, faster.
    Low,
    /// Balanced context (default).
    Medium,
    /// Maximum context — more results, slower.
    High,
}

impl SearchContextSize {
    /// Parse from a string value, returning `None` on unknown.
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    /// Parse from a string value, defaulting to `Medium` on unknown.
    ///
    /// Used at runtime for per-request metadata where rejecting is
    /// not appropriate.
    pub(crate) fn from_str_or_default(s: &str) -> Self {
        Self::from_str(s).unwrap_or(Self::Medium)
    }

    /// Result count hint for search API queries.
    pub(crate) fn result_count(self) -> u32 {
        match self {
            Self::Low => 3,
            Self::Medium => 5,
            Self::High => 10,
        }
    }
}

// -----------------------------------------------------------------------------
// WebSearchFilterConfig (YAML deserialization)
// -----------------------------------------------------------------------------

/// Raw YAML config, deserialized then validated.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WebSearchFilterConfig {
    /// Search backend provider.
    pub(crate) provider: SearchProvider,

    /// API key for the search provider (supports `${ENV_VAR}`).
    /// Wrapped in [`SecretString`] to prevent accidental logging.
    pub(crate) api_key: SecretString,

    /// Default search context size when the client omits it.
    #[serde(default)]
    pub(crate) default_context_size: Option<String>,

    /// Callout timeout in milliseconds.
    #[serde(default)]
    pub(crate) timeout_ms: Option<u64>,

    /// Maximum request body bytes to buffer.
    #[serde(default)]
    pub(crate) max_body_bytes: Option<usize>,

    /// Override the provider's default API base URL.
    #[serde(default)]
    pub(crate) base_url: Option<String>,

    /// Allow a `base_url` that targets local-sensitive addresses.
    ///
    /// DNS targets are unsupported in protected mode (the default):
    /// validation cannot pin the address the HTTP client will eventually
    /// dial, so a `base_url` host must be a public IP literal. Enabling
    /// `allow_private_base_url` also permits DNS results resolving to
    /// local-sensitive addresses, so a hostile or rebound resolution can
    /// send the provider credential to a loopback, private, or
    /// cloud-metadata endpoint.
    #[serde(default)]
    pub(crate) allow_private_base_url: bool,
}

// -----------------------------------------------------------------------------
// OpenAiWebSearchConfig (YAML deserialization)
// -----------------------------------------------------------------------------

// Mirrors `WebSearchFilterConfig` minus `max_body_bytes` and validates through
// the shared `build_config` via `into_shared`. Keep the remaining fields in
// sync with `WebSearchFilterConfig`.

/// Reads the request body but never rewrites it, so `openai_web_search`
/// exposes no `max_body_bytes` knob: raw request body size is governed by the
/// pipeline's `body_limits`, not a per-filter limit (which praxis core merges
/// to the largest sibling buffer and would therefore be bypassable).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiWebSearchConfig {
    /// Search backend provider.
    provider: SearchProvider,

    /// API key for the search provider (supports `${ENV_VAR}`).
    /// Wrapped in [`SecretString`] to prevent accidental logging.
    api_key: SecretString,

    /// Default search context size when the client omits it.
    #[serde(default)]
    default_context_size: Option<String>,

    /// Callout timeout in milliseconds.
    #[serde(default)]
    timeout_ms: Option<u64>,

    /// Override the provider's default API base URL.
    #[serde(default)]
    base_url: Option<String>,

    /// Allow a `base_url` that targets local-sensitive addresses.
    ///
    /// DNS targets are unsupported in protected mode (the default):
    /// validation cannot pin the address the HTTP client will eventually
    /// dial, so a `base_url` host must be a public IP literal. Enabling
    /// `allow_private_base_url` also permits DNS results resolving to
    /// local-sensitive addresses, so a hostile or rebound resolution can
    /// send the provider credential to a loopback, private, or
    /// cloud-metadata endpoint.
    #[serde(default)]
    allow_private_base_url: bool,
}

impl OpenAiWebSearchConfig {
    /// Convert into the shared [`WebSearchFilterConfig`] for validation reuse.
    ///
    /// `max_body_bytes` is fixed to `None`: `openai_web_search` defers raw
    /// request body size to the pipeline's `body_limits` and buffers to the
    /// absolute JSON ceiling, so it carries no per-filter raw-body cap.
    pub(crate) fn into_shared(self) -> WebSearchFilterConfig {
        WebSearchFilterConfig {
            provider: self.provider,
            api_key: self.api_key,
            default_context_size: self.default_context_size,
            timeout_ms: self.timeout_ms,
            max_body_bytes: None,
            base_url: self.base_url,
            allow_private_base_url: self.allow_private_base_url,
        }
    }
}

// -----------------------------------------------------------------------------
// ValidatedConfig (post-validation)
// -----------------------------------------------------------------------------

/// Validated configuration with defaults applied.
#[derive(Clone)]
pub(crate) struct ValidatedConfig {
    /// Search backend provider.
    pub provider: SearchProvider,

    /// Resolved API key.
    pub api_key: SecretString,

    /// Default search context size.
    pub default_context_size: SearchContextSize,

    /// Callout timeout in milliseconds.
    pub timeout_ms: u64,

    /// Maximum request body bytes to buffer.
    pub max_body_bytes: usize,

    /// Override the provider's default API base URL.
    pub base_url: Option<String>,
}

impl std::fmt::Debug for ValidatedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatedConfig")
            .field("provider", &self.provider)
            .field("api_key", &"[REDACTED]")
            .field("default_context_size", &self.default_context_size)
            .field("timeout_ms", &self.timeout_ms)
            .field("max_body_bytes", &self.max_body_bytes)
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Validate raw config and apply defaults.
///
/// # Errors
///
/// Returns [`FilterError`] if the API key is empty or cannot be
/// resolved from environment variables.
pub(crate) fn build_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
) -> Result<ValidatedConfig, FilterError> {
    let api_key = resolve_api_key(filter_name, raw.api_key.expose_secret())?;
    if api_key.is_empty() {
        return Err(FilterError::from(format!("{filter_name}: api_key must not be empty")));
    }
    build_validated_config(filter_name, raw, api_key)
}

/// Apply defaults after resolving and validating the API key.
fn build_validated_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
    api_key: String,
) -> Result<ValidatedConfig, FilterError> {
    if let Some(base_url) = raw.base_url.as_deref() {
        crate::openai::api_client::validate_base_url(filter_name, base_url, raw.allow_private_base_url)?;
    }
    Ok(ValidatedConfig {
        provider: raw.provider,
        api_key: SecretString::from(api_key),
        default_context_size: validate_context_size(filter_name, raw.default_context_size.as_deref())?,
        timeout_ms: callout_policy::validate_timeout_ms(filter_name, raw.timeout_ms, DEFAULT_TIMEOUT_MS)?,
        max_body_bytes: validate_max_body_bytes_field(filter_name, raw.max_body_bytes)?,
        base_url: raw.base_url.clone(),
    })
}

/// Validate `default_context_size`, defaulting to `Medium` when
/// absent and rejecting unknown values.
fn validate_context_size(filter_name: &'static str, raw: Option<&str>) -> Result<SearchContextSize, FilterError> {
    match raw {
        None => Ok(SearchContextSize::Medium),
        Some(s) => SearchContextSize::from_str(s).ok_or_else(|| {
            FilterError::from(format!(
                "{filter_name}: default_context_size must be low, medium, or high, got '{s}'"
            ))
        }),
    }
}

/// Validate `max_body_bytes`, applying the default and delegating
/// to the standard validator that rejects 0 and oversized values.
fn validate_max_body_bytes_field(filter_name: &'static str, raw: Option<usize>) -> Result<usize, FilterError> {
    let value = raw.unwrap_or(MAX_JSON_BODY_BYTES);
    validate_max_body_bytes(filter_name, value)?;
    Ok(value)
}

/// Resolve `${ENV_VAR}` references in the API key string.
fn resolve_api_key(filter_name: &'static str, raw: &str) -> Result<String, FilterError> {
    let trimmed = raw.trim();
    if let Some(var_name) = trimmed.strip_prefix("${").and_then(|s| s.strip_suffix('}')) {
        std::env::var(var_name).map_err(|e| {
            FilterError::from(format!(
                "{filter_name}: environment variable {var_name} not set for api_key: {e}"
            ))
        })
    } else {
        Ok(trimmed.to_owned())
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use praxis_filter::parse_filter_config;
    use secrecy::{ExposeSecret as _, SecretString};

    use super::*;

    fn base_config() -> WebSearchFilterConfig {
        WebSearchFilterConfig {
            provider: SearchProvider::Brave,
            api_key: SecretString::from("test-key-123".to_owned()),
            default_context_size: None,
            timeout_ms: None,
            max_body_bytes: None,
            base_url: None,
            allow_private_base_url: false,
        }
    }

    #[test]
    fn build_config_applies_defaults() {
        let cfg = build_config("openai_web_search", &base_config()).unwrap();
        assert_eq!(cfg.provider, SearchProvider::Brave);
        assert_eq!(cfg.api_key.expose_secret(), "test-key-123");
        assert_eq!(cfg.default_context_size, SearchContextSize::Medium);
        assert_eq!(cfg.timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(cfg.max_body_bytes, MAX_JSON_BODY_BYTES);
    }

    #[test]
    fn build_config_rejects_empty_api_key() {
        let mut cfg = base_config();
        cfg.api_key = SecretString::from(String::new());
        assert!(build_config("openai_web_search", &cfg).is_err());
    }

    #[test]
    fn parse_config_uses_owner_name_for_missing_api_key() {
        let yaml = serde_yaml::from_str("provider: brave").unwrap();
        let error = parse_filter_config::<WebSearchFilterConfig>("anthropic_web_search", &yaml)
            .err()
            .expect("missing api_key should be rejected");
        assert!(
            error
                .to_string()
                .contains("anthropic_web_search: missing field `api_key`"),
            "diagnostic should name the owning filter: {error}"
        );
    }

    #[test]
    fn build_config_rejects_zero_timeout() {
        let mut cfg = base_config();
        cfg.timeout_ms = Some(0);
        assert!(build_config("openai_web_search", &cfg).is_err());
    }

    #[test]
    fn build_config_rejects_invalid_context_size() {
        let mut cfg = base_config();
        cfg.default_context_size = Some("xlarge".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "unknown default_context_size should be rejected"
        );
    }

    #[test]
    fn build_config_custom_values() {
        let mut cfg = base_config();
        cfg.default_context_size = Some("high".into());
        cfg.timeout_ms = Some(15_000);
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(validated.default_context_size, SearchContextSize::High);
        assert_eq!(validated.timeout_ms, 15_000);
    }

    #[test]
    fn build_config_base_url_threaded_through() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://localhost:9999".into());
        cfg.allow_private_base_url = true;
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(validated.base_url.as_deref(), Some("http://localhost:9999"));
    }

    #[test]
    fn build_config_base_url_none_by_default() {
        let validated = build_config("openai_web_search", &base_config()).unwrap();
        assert!(validated.base_url.is_none());
    }

    #[test]
    fn build_config_rejects_loopback_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://127.0.0.1:9999".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "loopback base_url must be rejected without allow_private_base_url (SSRF/credential disclosure)"
        );
    }

    #[test]
    fn build_config_rejects_localhost_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://localhost:9999".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "localhost base_url must be rejected without allow_private_base_url"
        );
    }

    #[test]
    fn build_config_rejects_cloud_metadata_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://169.254.169.254".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "link-local cloud-metadata base_url must be rejected without allow_private_base_url"
        );
    }

    #[test]
    fn build_config_rejects_dns_base_url_without_opt_in() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://internal.search.example:8080".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "DNS base_url must be rejected without allow_private_base_url because the dialed address cannot be pinned"
        );
    }

    #[test]
    fn build_config_rejects_non_http_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("file:///etc/passwd".into());
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "non-http(s) base_url scheme must be rejected"
        );
    }

    #[test]
    fn build_config_rejects_base_url_with_embedded_credentials() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://user:pass@8.8.8.8".into());
        cfg.allow_private_base_url = true;
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "base_url with embedded credentials must be rejected even with allow_private_base_url"
        );
    }

    #[test]
    fn build_config_allows_public_ip_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("https://8.8.8.8".into());
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(
            validated.base_url.as_deref(),
            Some("https://8.8.8.8"),
            "public IP literal base_url should be accepted without the opt-in"
        );
    }

    #[test]
    fn build_config_allows_private_base_url_with_opt_in() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://127.0.0.1:9999".into());
        cfg.allow_private_base_url = true;
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(validated.base_url.as_deref(), Some("http://127.0.0.1:9999"));
    }

    #[test]
    fn resolve_literal_api_key() {
        let result = resolve_api_key("openai_web_search", "my-literal-key").unwrap();
        assert_eq!(result, "my-literal-key");
    }

    #[test]
    fn resolve_literal_api_key_trimmed() {
        let result = resolve_api_key("openai_web_search", "  spaced-key  ").unwrap();
        assert_eq!(result, "spaced-key");
    }

    #[test]
    fn resolve_env_var_syntax_detected() {
        let result = resolve_api_key("openai_web_search", "${DEFINITELY_NOT_SET_KEY_12345}");
        assert!(result.is_err(), "missing env var should fail");
    }

    #[test]
    fn resolve_partial_env_syntax_treated_as_literal() {
        let result = resolve_api_key("openai_web_search", "${INCOMPLETE").unwrap();
        assert_eq!(result, "${INCOMPLETE", "unclosed brace should be literal");
    }

    #[test]
    fn build_config_rejects_zero_max_body_bytes() {
        let mut cfg = base_config();
        cfg.max_body_bytes = Some(0);
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "max_body_bytes=0 should be rejected"
        );
    }

    #[test]
    fn build_config_rejects_oversized_max_body_bytes() {
        let mut cfg = base_config();
        cfg.max_body_bytes = Some(999_999_999_999);
        assert!(
            build_config("openai_web_search", &cfg).is_err(),
            "max_body_bytes above limit should be rejected"
        );
    }

    #[test]
    fn debug_impl_redacts_api_key() {
        let cfg = build_config("openai_web_search", &base_config()).unwrap();
        let debug_output = format!("{cfg:?}");
        assert!(
            debug_output.contains("[REDACTED]"),
            "Debug output should redact api_key"
        );
        assert!(
            !debug_output.contains("test-key-123"),
            "Debug output should not contain the actual api_key"
        );
    }

    #[test]
    fn search_context_size_result_counts() {
        assert_eq!(SearchContextSize::Low.result_count(), 3);
        assert_eq!(SearchContextSize::Medium.result_count(), 5);
        assert_eq!(SearchContextSize::High.result_count(), 10);
    }

    #[test]
    fn search_context_size_parsing() {
        assert_eq!(SearchContextSize::from_str_or_default("low"), SearchContextSize::Low);
        assert_eq!(
            SearchContextSize::from_str_or_default("medium"),
            SearchContextSize::Medium
        );
        assert_eq!(SearchContextSize::from_str_or_default("high"), SearchContextSize::High);
        assert_eq!(
            SearchContextSize::from_str_or_default("unknown"),
            SearchContextSize::Medium
        );
    }

    #[test]
    fn search_provider_as_str() {
        assert_eq!(SearchProvider::Brave.as_str(), "brave");
        assert_eq!(SearchProvider::Tavily.as_str(), "tavily");
        assert_eq!(SearchProvider::You.as_str(), "you");
    }
}
