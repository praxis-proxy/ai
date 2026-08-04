// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration for protocol-neutral web-search providers.

use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;

/// Default callout timeout (10 seconds — search APIs can be slow).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Default HTTP status when the search callout fails in closed mode.
const DEFAULT_STATUS_ON_ERROR: u16 = 502;

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
// FailureMode
// -----------------------------------------------------------------------------

/// What happens when a search callout fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureMode {
    /// Reject the request on search failure (default).
    Closed,
    /// Continue without search results on failure.
    Open,
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

    /// Failure mode for search provider callouts.
    #[serde(default)]
    pub(crate) provider_failure_mode: Option<FailureMode>,

    /// HTTP status code to return when rejecting on error.
    #[serde(default)]
    pub(crate) status_on_error: Option<u16>,

    /// Override the provider's default API base URL.
    #[serde(default)]
    pub(crate) base_url: Option<String>,
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

    /// Failure mode for search callouts.
    pub failure_mode: FailureMode,

    /// HTTP status on error.
    pub status_on_error: u16,

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
            .field("failure_mode", &self.failure_mode)
            .field("status_on_error", &self.status_on_error)
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
    Ok(ValidatedConfig {
        provider: raw.provider,
        api_key: SecretString::from(api_key),
        default_context_size: validate_context_size(filter_name, raw.default_context_size.as_deref())?,
        timeout_ms: validate_timeout_ms(filter_name, raw.timeout_ms)?,
        max_body_bytes: validate_max_body_bytes_field(filter_name, raw.max_body_bytes)?,
        failure_mode: raw.provider_failure_mode.unwrap_or(FailureMode::Closed),
        status_on_error: validate_status_on_error(filter_name, raw.status_on_error)?,
        base_url: raw.base_url.clone(),
    })
}

/// Validate timeout, applying the default and rejecting zero.
fn validate_timeout_ms(filter_name: &'static str, raw: Option<u64>) -> Result<u64, FilterError> {
    let value = raw.unwrap_or(DEFAULT_TIMEOUT_MS);
    if value == 0 {
        return Err(FilterError::from(format!(
            "{filter_name}: timeout_ms must be greater than 0"
        )));
    }
    Ok(value)
}

/// Validate HTTP status code, applying the default and rejecting out-of-range.
fn validate_status_on_error(filter_name: &'static str, raw: Option<u16>) -> Result<u16, FilterError> {
    let value = raw.unwrap_or(DEFAULT_STATUS_ON_ERROR);
    if !(100..=599).contains(&value) {
        return Err(FilterError::from(format!(
            "{filter_name}: status_on_error must be between 100 and 599, got {value}"
        )));
    }
    Ok(value)
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
            provider_failure_mode: None,
            status_on_error: None,
            base_url: None,
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
        assert_eq!(cfg.failure_mode, FailureMode::Closed);
        assert_eq!(cfg.status_on_error, DEFAULT_STATUS_ON_ERROR);
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
    fn parse_config_preserves_provider_failure_mode() {
        let yaml = serde_yaml::from_str("\nprovider: brave\napi_key: test-key\nprovider_failure_mode: open\n").unwrap();

        let raw: WebSearchFilterConfig = parse_filter_config("openai_web_search", &yaml).unwrap();
        let validated = build_config("openai_web_search", &raw).unwrap();

        assert_eq!(validated.failure_mode, FailureMode::Open);
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
    fn build_config_rejects_invalid_status() {
        let mut cfg = base_config();
        cfg.status_on_error = Some(999);
        assert!(build_config("openai_web_search", &cfg).is_err());
    }

    #[test]
    fn build_config_custom_values() {
        let mut cfg = base_config();
        cfg.default_context_size = Some("high".into());
        cfg.timeout_ms = Some(15_000);
        cfg.provider_failure_mode = Some(FailureMode::Open);
        cfg.status_on_error = Some(503);
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(validated.default_context_size, SearchContextSize::High);
        assert_eq!(validated.timeout_ms, 15_000);
        assert_eq!(validated.failure_mode, FailureMode::Open);
        assert_eq!(validated.status_on_error, 503);
    }

    #[test]
    fn build_config_base_url_threaded_through() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://localhost:9999".into());
        let validated = build_config("openai_web_search", &cfg).unwrap();
        assert_eq!(validated.base_url.as_deref(), Some("http://localhost:9999"));
    }

    #[test]
    fn build_config_base_url_none_by_default() {
        let validated = build_config("openai_web_search", &base_config()).unwrap();
        assert!(validated.base_url.is_none());
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
