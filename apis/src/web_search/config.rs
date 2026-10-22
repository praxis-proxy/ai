// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for protocol-neutral web-search providers.

use praxis_core::config::ChainRef;
use praxis_filter::{
    FilterError, body::MAX_JSON_BODY_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;
use zeroize::Zeroizing;

use crate::{
    callout_policy,
    credential_source::{self, CredentialSource, DiagnosticContext, SecretHandle, WatchHandle},
};

/// Default callout timeout (10 seconds — search APIs can be slow).
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Default model-driven web searches accepted from one response round.
#[cfg(feature = "openai-responses")]
pub(crate) const DEFAULT_MAX_CALLS_PER_ROUND: usize = 32;

/// Absolute model-driven web-search calls accepted from one response round.
#[cfg(feature = "openai-responses")]
pub(crate) const MAX_CALLS_PER_ROUND: usize = 1_024;

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
    #[cfg(feature = "openai-responses")]
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

    /// Optional callout-credential slot id. When set, the web-search callout uses the caller's
    /// per-user secret from that slot instead of the shared provider `api_key`. Non-secret (a slot
    /// name). Supported for every provider: each stages the secret as a destination-bound header
    /// credential the executor injects after pinning the provider authority (Brave and You.com
    /// under their provider header, Tavily as an `Authorization: Bearer` token).
    #[serde(default)]
    pub(crate) user_credential: Option<String>,

    /// API key credential source for the search provider: exactly one of
    /// `value`, `env_var`, or `file`. A `file` source is watched for
    /// restart-free rotation; see [`CredentialSource`]. The token never appears
    /// in `Debug` output.
    pub(crate) api_key: CredentialSource,

    /// Default search context size when the client omits it.
    #[serde(default)]
    pub(crate) default_context_size: Option<String>,

    /// Callout timeout in milliseconds. Inside an iterative request router,
    /// the effective timeout is capped by the router's remaining deadline.
    #[serde(default)]
    pub(crate) timeout_ms: Option<u64>,

    /// Maximum request and response body bytes buffered per loop round.
    #[serde(default)]
    pub(crate) max_body_bytes: Option<usize>,

    /// Override the provider's default API base URL.
    #[serde(default)]
    pub(crate) base_url: Option<String>,

    /// Outbound filter chain the provider callout executes through.
    ///
    /// A **named** reference resolves against the top-level `filter_chains`
    /// map, so it binds only when the filter is placed at the top level of a
    /// pipeline (see the `web-search.yaml` example). An **inline** definition
    /// embeds the filters directly and always binds — including when the filter
    /// runs nested as a step of an `iterative_request_router` (the agentic
    /// loop), where the step pipeline's chain map is empty and a named
    /// reference cannot resolve. Use an inline chain for any nested/IRR
    /// placement; a named reference is available only where top-level
    /// `filter_chains` are in scope.
    ///
    /// The chain carries cross-cutting concerns (observability, security,
    /// credential injection) and is bound once at pipeline-build time — a chain
    /// that cannot be built fails config validation. Destination authority,
    /// DNS/SSRF, TLS/SNI, and `Host` are enforced centrally by the executor,
    /// gated by `insecure_options.allow_private_upstreams`.
    ///
    /// Optional. The provider callout always runs through the shared executor;
    /// this chain only adds filters along the way. When omitted it defaults to
    /// an empty inline chain (pure passthrough) via [`default_outbound_chain`],
    /// so every central protection still applies. Provide it only to attach
    /// cross-cutting concerns.
    #[serde(default = "default_outbound_chain")]
    pub(crate) outbound_chain: ChainRef,
}

// -----------------------------------------------------------------------------
// OpenAiWebSearchConfig (YAML deserialization)
// -----------------------------------------------------------------------------

// Mirrors `WebSearchFilterConfig` minus `max_body_bytes` and validates through
// the shared `build_config` via `into_shared`. Keep the remaining fields in
// sync with `WebSearchFilterConfig`.

/// Reads the request body but never rewrites it, so `openai_web_search_dispatch`
/// exposes no `max_body_bytes` knob: raw request body size is governed by the
/// pipeline's `body_limits`, not a per-filter limit (which praxis core merges
/// to the largest sibling buffer and would therefore be bypassable).
#[cfg(feature = "openai-responses")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiWebSearchConfig {
    /// Search backend provider.
    provider: SearchProvider,

    /// Optional callout-credential slot id (see [`WebSearchFilterConfig::user_credential`]).
    #[serde(default)]
    user_credential: Option<String>,

    /// API key credential source for the search provider: exactly one of
    /// `value`, `env_var`, or `file`; see [`CredentialSource`].
    api_key: CredentialSource,

    /// Default search context size when the client omits it.
    #[serde(default)]
    default_context_size: Option<String>,

    /// Callout timeout in milliseconds. Inside an iterative request router,
    /// the effective timeout is capped by the router's remaining deadline.
    #[serde(default)]
    timeout_ms: Option<u64>,

    /// Hard cap on web-search calls processed from one model response (1..=1024; default: 32).
    #[serde(default = "default_max_calls_per_round")]
    pub(crate) max_calls_per_round: usize,

    /// Override the provider's default API base URL.
    #[serde(default)]
    base_url: Option<String>,

    /// Outbound filter chain the provider callout executes through.
    ///
    /// See [`WebSearchFilterConfig::outbound_chain`]; the two configs stay in
    /// sync. Optional — when omitted it defaults to an empty inline passthrough
    /// chain via [`default_outbound_chain`].
    #[serde(default = "default_outbound_chain")]
    pub(crate) outbound_chain: ChainRef,
}

/// Default value for `OpenAiWebSearchConfig::max_calls_per_round`.
#[cfg(feature = "openai-responses")]
fn default_max_calls_per_round() -> usize {
    DEFAULT_MAX_CALLS_PER_ROUND
}

/// Default `outbound_chain`: an empty inline chain (pure passthrough).
///
/// The provider callout always runs through the shared filtered-subrequest
/// executor, which enforces destination authority, DNS/SSRF, TLS/SNI, and
/// `Host` regardless of chain contents. An empty inline chain adds no extra
/// filters, so omitting `outbound_chain` keeps every central protection while
/// requiring no configuration. It is inline (not named) so it also binds when
/// the filter runs nested as an `iterative_request_router` step, where a
/// top-level named-chain reference cannot resolve.
fn default_outbound_chain() -> ChainRef {
    ChainRef::Inline {
        name: "web_search_outbound".to_owned(),
        filters: Vec::new(),
    }
}

#[cfg(feature = "openai-responses")]
impl OpenAiWebSearchConfig {
    /// Convert into the shared [`WebSearchFilterConfig`] for validation reuse.
    ///
    /// `max_body_bytes` is fixed to `None`: `openai_web_search_dispatch` defers raw
    /// request body size to the pipeline's `body_limits` and buffers to the
    /// absolute JSON ceiling, so it carries no per-filter raw-body cap.
    pub(crate) fn into_shared(self) -> WebSearchFilterConfig {
        WebSearchFilterConfig {
            provider: self.provider,
            user_credential: self.user_credential,
            api_key: self.api_key,
            default_context_size: self.default_context_size,
            timeout_ms: self.timeout_ms,
            max_body_bytes: None,
            base_url: self.base_url,
            outbound_chain: self.outbound_chain,
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

    /// Published handle to the resolved API key. A `None` payload (a post-startup
    /// rotation gap) must fail the callout closed — never dial unauthenticated.
    pub api_key: SecretHandle,

    /// Default search context size.
    pub default_context_size: SearchContextSize,

    /// Callout timeout in milliseconds.
    pub timeout_ms: u64,

    /// Maximum request body bytes to buffer.
    pub max_body_bytes: usize,

    /// Override the provider's default API base URL.
    pub base_url: Option<String>,

    /// Configured callout-credential slot id (non-secret), or `None` for the shared key.
    pub user_credential: Option<String>,
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
            .field("user_credential", &self.user_credential)
            .finish()
    }
}

/// Validate raw config, publish the API key credential, and apply defaults.
///
/// The API key is resolved from its [`CredentialSource`] through
/// [`credential_source::build_one`]: the token is validated as a legal HTTP
/// header value and published as the payload, an invalid initial token rejects
/// construction (fail loud at startup), and a `file` source registers a watcher.
/// The returned [`WatchHandle`] (present only for a `file` source) MUST be
/// retained by the caller so rotation keeps publishing.
///
/// # Errors
///
/// Returns [`FilterError`] if the API key source is unreadable, empty, oversized,
/// or does not form a legal HTTP header value, or if any other field is invalid.
pub(crate) fn build_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
) -> Result<(ValidatedConfig, Option<WatchHandle>), FilterError> {
    // Validate that the token forms a legal HTTP header value and publish it,
    // untransformed, as the payload. The owner-qualified diagnostic names the
    // filter so an invalid key is attributable at startup.
    let transform = move |token: &str| -> Result<Zeroizing<String>, FilterError> {
        http::HeaderValue::from_str(token)
            .map_err(|e| FilterError::from(format!("{filter_name}: invalid API key header value: {e}")))?;
        Ok(Zeroizing::new(token.to_owned()))
    };
    let ctx = DiagnosticContext {
        owner: filter_name,
        reference: None,
    };
    let (api_key, watch) = credential_source::build_one(ctx, &raw.api_key, transform)?;
    let validated = build_validated_config(filter_name, raw, api_key)?;
    Ok((validated, watch))
}

/// Apply defaults after resolving and validating the API key.
fn build_validated_config(
    filter_name: &'static str,
    raw: &WebSearchFilterConfig,
    api_key: SecretHandle,
) -> Result<ValidatedConfig, FilterError> {
    if let Some(base_url) = raw.base_url.as_deref() {
        // Structural checks only (scheme, embedded credentials, path). Private-
        // address / SSRF enforcement is deferred to the outbound executor's
        // connect-time `build_peer`, gated by
        // `insecure_options.allow_private_upstreams`. Pass `allow_private = true`
        // here so config validation does not duplicate (or contradict) that
        // runtime gate; `validate_base_url` is shared with other filters and its
        // signature must stay stable.
        crate::openai::api_client::validate_base_url(filter_name, base_url, true)?;
    }
    Ok(ValidatedConfig {
        provider: raw.provider,
        api_key,
        default_context_size: validate_context_size(filter_name, raw.default_context_size.as_deref())?,
        timeout_ms: callout_policy::validate_timeout_ms(filter_name, raw.timeout_ms, DEFAULT_TIMEOUT_MS)?,
        max_body_bytes: validate_max_body_bytes_field(filter_name, raw.max_body_bytes)?,
        base_url: raw.base_url.clone(),
        user_credential: raw.user_credential.clone(),
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

    use super::*;

    fn base_config() -> WebSearchFilterConfig {
        WebSearchFilterConfig {
            provider: SearchProvider::Brave,
            user_credential: None,
            api_key: CredentialSource::Value(Zeroizing::new("test-key-123".to_owned())),
            default_context_size: None,
            timeout_ms: None,
            max_body_bytes: None,
            base_url: None,
            outbound_chain: ChainRef::Named("web_search_outbound".to_owned()),
        }
    }

    #[test]
    fn anthropic_web_search_rejects_terminal_streaming() {
        // `terminal_streaming` was removed: `anthropic_web_search` now selects the
        // streaming transport per request from the client's `stream` field. The
        // shared config carries `deny_unknown_fields`, so a stale config that still
        // sets the flag fails to build — the migration is explicit, never a silent
        // no-op toggle.
        let yaml = serde_yaml::from_str(
            "provider: you\napi_key:\n  value: k\noutbound_chain: web_search_outbound\nterminal_streaming: true",
        )
        .unwrap();
        assert!(
            parse_filter_config::<WebSearchFilterConfig>("anthropic_web_search", &yaml).is_err(),
            "the removed terminal_streaming flag must be rejected so operators migrate their config"
        );
    }

    #[test]
    fn parse_config_defaults_missing_outbound_chain_to_empty_inline() {
        // `outbound_chain` is optional: omitting it defaults to an empty inline
        // chain (pure passthrough), matching `openai_file_search_dispatch`. The
        // callout still runs through the shared executor, which enforces
        // destination authority, DNS/SSRF, TLS/SNI, and `Host` centrally.
        let yaml = serde_yaml::from_str("provider: you\napi_key:\n  value: k").unwrap();
        let raw = parse_filter_config::<WebSearchFilterConfig>("anthropic_web_search", &yaml)
            .expect("omitting outbound_chain must parse via the default");
        match &raw.outbound_chain {
            ChainRef::Inline { filters, .. } => assert!(
                filters.is_empty(),
                "default outbound_chain must be an empty inline chain (pure passthrough)"
            ),
            ChainRef::Named(name) => panic!("default outbound_chain should be inline, got named `{name}`"),
        }
    }

    #[cfg(feature = "openai-responses")]
    #[test]
    fn openai_parse_config_defaults_missing_outbound_chain_to_empty_inline() {
        // The OpenAI-specific config carries its own `#[serde(default)]`, so it
        // must default identically when `outbound_chain` is omitted.
        let yaml = serde_yaml::from_str("provider: brave\napi_key:\n  value: k").unwrap();
        let raw = parse_filter_config::<OpenAiWebSearchConfig>("openai_web_search_dispatch", &yaml)
            .expect("omitting outbound_chain must parse via the default");
        match &raw.outbound_chain {
            ChainRef::Inline { filters, .. } => assert!(
                filters.is_empty(),
                "default outbound_chain must be an empty inline chain (pure passthrough)"
            ),
            ChainRef::Named(name) => panic!("default outbound_chain should be inline, got named `{name}`"),
        }
    }

    #[cfg(feature = "openai-responses")]
    #[test]
    fn openai_web_search_dispatch_rejects_terminal_streaming() {
        let yaml = serde_yaml::from_str("provider: you\napi_key:\n  value: k\nterminal_streaming: true").unwrap();
        assert!(
            parse_filter_config::<OpenAiWebSearchConfig>("openai_web_search_dispatch", &yaml).is_err(),
            "terminal_streaming was removed; openai_web_search_dispatch must reject the unknown field"
        );
    }

    #[test]
    fn build_config_applies_defaults() {
        let (cfg, _watch) = build_config("openai_web_search_dispatch", &base_config()).unwrap();
        assert_eq!(cfg.provider, SearchProvider::Brave);
        assert_eq!(
            cfg.api_key.current().payload.as_deref().map(String::as_str),
            Some("test-key-123")
        );
        assert_eq!(cfg.default_context_size, SearchContextSize::Medium);
        assert_eq!(cfg.timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(cfg.max_body_bytes, MAX_JSON_BODY_BYTES);
    }

    #[test]
    fn build_config_rejects_empty_api_key() {
        let mut cfg = base_config();
        cfg.api_key = CredentialSource::Value(Zeroizing::new(String::new()));
        assert!(build_config("openai_web_search_dispatch", &cfg).is_err());
    }

    #[test]
    fn build_config_invalid_api_key_names_owner() {
        // An initial token that cannot form a legal HTTP header value rejects
        // construction (fail loud at startup), and the diagnostic names the
        // owning filter so an operator can attribute the misconfiguration.
        let mut cfg = base_config();
        cfg.api_key = CredentialSource::Value(Zeroizing::new("invalid\nkey".to_owned()));
        let error = build_config("anthropic_web_search", &cfg)
            .map(|_| ())
            .expect_err("an invalid API key header value must reject construction");
        assert!(
            error
                .to_string()
                .contains("anthropic_web_search: invalid API key header value"),
            "diagnostic should name the owning filter: {error}"
        );
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
        assert!(build_config("openai_web_search_dispatch", &cfg).is_err());
    }

    #[test]
    fn build_config_rejects_invalid_context_size() {
        let mut cfg = base_config();
        cfg.default_context_size = Some("xlarge".into());
        assert!(
            build_config("openai_web_search_dispatch", &cfg).is_err(),
            "unknown default_context_size should be rejected"
        );
    }

    #[test]
    fn build_config_custom_values() {
        let mut cfg = base_config();
        cfg.default_context_size = Some("high".into());
        cfg.timeout_ms = Some(15_000);
        let (validated, _watch) = build_config("openai_web_search_dispatch", &cfg).unwrap();
        assert_eq!(validated.default_context_size, SearchContextSize::High);
        assert_eq!(validated.timeout_ms, 15_000);
    }

    #[test]
    fn build_config_base_url_threaded_through() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://localhost:9999".into());
        let (validated, _watch) = build_config("openai_web_search_dispatch", &cfg).unwrap();
        assert_eq!(validated.base_url.as_deref(), Some("http://localhost:9999"));
    }

    #[test]
    fn build_config_base_url_none_by_default() {
        let (validated, _watch) = build_config("openai_web_search_dispatch", &base_config()).unwrap();
        assert!(validated.base_url.is_none());
    }

    #[test]
    fn build_config_accepts_private_base_url_ssrf_deferred_to_executor() {
        // Private/loopback/link-local destinations are no longer rejected at
        // config-validation time: SSRF enforcement moved to the outbound
        // executor's connect-time `build_peer`, gated by
        // `insecure_options.allow_private_upstreams`. Config validation keeps
        // only structural checks.
        for url in [
            "http://127.0.0.1:9999",
            "http://localhost:9999",
            "http://169.254.1.1",
            "http://internal.search.example:8080",
        ] {
            let mut cfg = base_config();
            cfg.base_url = Some(url.into());
            assert!(
                build_config("openai_web_search_dispatch", &cfg).is_ok(),
                "private base_url `{url}` must pass structural validation; SSRF is enforced at connect time"
            );
        }
    }

    #[test]
    fn build_config_rejects_non_http_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("file:///etc/passwd".into());
        assert!(
            build_config("openai_web_search_dispatch", &cfg).is_err(),
            "non-http(s) base_url scheme must be rejected"
        );
    }

    #[test]
    fn build_config_rejects_base_url_with_embedded_credentials() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://user:pass@8.8.8.8".into());
        assert!(
            build_config("openai_web_search_dispatch", &cfg).is_err(),
            "base_url with embedded credentials must be rejected (structural check)"
        );
    }

    #[test]
    fn build_config_allows_public_ip_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("https://8.8.8.8".into());
        let (validated, _watch) = build_config("openai_web_search_dispatch", &cfg).unwrap();
        assert_eq!(
            validated.base_url.as_deref(),
            Some("https://8.8.8.8"),
            "public IP literal base_url should be accepted without the opt-in"
        );
    }

    #[test]
    fn build_config_allows_private_base_url() {
        let mut cfg = base_config();
        cfg.base_url = Some("http://127.0.0.1:9999".into());
        let (validated, _watch) = build_config("openai_web_search_dispatch", &cfg).unwrap();
        assert_eq!(validated.base_url.as_deref(), Some("http://127.0.0.1:9999"));
    }

    #[test]
    fn build_config_rejects_zero_max_body_bytes() {
        let mut cfg = base_config();
        cfg.max_body_bytes = Some(0);
        assert!(
            build_config("openai_web_search_dispatch", &cfg).is_err(),
            "max_body_bytes=0 should be rejected"
        );
    }

    #[test]
    fn build_config_rejects_oversized_max_body_bytes() {
        let mut cfg = base_config();
        cfg.max_body_bytes = Some(999_999_999_999);
        assert!(
            build_config("openai_web_search_dispatch", &cfg).is_err(),
            "max_body_bytes above limit should be rejected"
        );
    }

    #[test]
    fn debug_impl_redacts_api_key() {
        let (cfg, _watch) = build_config("openai_web_search_dispatch", &base_config()).unwrap();
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

    #[cfg(feature = "openai-responses")]
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

    // -------------------------------------------------------------------------
    // user_credential slot (issue #880)
    // -------------------------------------------------------------------------

    #[test]
    fn brave_accepts_user_credential_slot() {
        let mut cfg = base_config(); // provider = Brave
        cfg.user_credential = Some("brave".to_owned());
        let (validated, _watch) = build_config("anthropic_web_search", &cfg).unwrap();
        assert_eq!(validated.user_credential.as_deref(), Some("brave"));
    }

    #[test]
    fn no_user_credential_is_valid_for_tavily() {
        let mut cfg = base_config();
        cfg.provider = SearchProvider::Tavily;
        cfg.user_credential = None;
        let (validated, _watch) = build_config("anthropic_web_search", &cfg).unwrap();
        assert!(validated.user_credential.is_none());
    }

    #[test]
    fn tavily_accepts_user_credential_slot() {
        // Tavily now authenticates with an `Authorization: Bearer` header
        // credential (issue #1389), so a per-user slot is supported just as it is
        // for the other header-authenticated providers.
        let mut cfg = base_config();
        cfg.provider = SearchProvider::Tavily;
        cfg.user_credential = Some("tav".to_owned());
        let (validated, _watch) = build_config("anthropic_web_search", &cfg).unwrap();
        assert_eq!(validated.user_credential.as_deref(), Some("tav"));
    }

    #[cfg(feature = "openai-responses")]
    #[test]
    fn openai_config_threads_user_credential_through_into_shared() {
        // Locks that `into_shared` does not drop the slot id on the OpenAI path.
        let yaml = serde_yaml::from_str("provider: brave\napi_key:\n  value: k\nuser_credential: brave").unwrap();
        let raw = parse_filter_config::<OpenAiWebSearchConfig>("openai_web_search_dispatch", &yaml).unwrap();
        let (validated, _watch) = build_config("openai_web_search_dispatch", &raw.into_shared()).unwrap();
        assert_eq!(validated.user_credential.as_deref(), Some("brave"));
    }
}
