// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Filter configuration for `file_search_callout`.

use std::time::Duration;

use http::HeaderValue;
use praxis_callout_core::callout::{CalloutConfig, FailureMode};
use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use reqwest::Url;
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;
use zeroize::Zeroizing;

use super::{citations::count_template_placeholders, client::MAX_CONCURRENT_SEARCHES};
use crate::openai::api_client::{self, ApiClient, ApiClientConfig};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default per-result formatting template.
const DEFAULT_ANNOTATION_TEMPLATE: &str = "[{index}] {filename} (score: {score}) cite as <|{file_id}|>\n{content}\n";

/// Default model-facing search context template.
const DEFAULT_CONTEXT_TEMPLATE: &str = "file_search found {num_chunks} chunks for \"{query}\":\n{results}";

/// Default maximum response body size: 10 MiB.
const DEFAULT_MAX_RESPONSE_BYTES: usize = 10_485_760; // 10 MiB

/// Maximum response body size for one callout: 64 MiB.
const MAX_RESPONSE_BYTES: usize = MAX_JSON_BODY_BYTES;

/// Maximum successful wire bytes retained across one execution: 64 MiB.
const MAX_TOTAL_RESPONSE_BYTES: usize = MAX_JSON_BODY_BYTES;

/// Maximum size of one formatting template: 16 `KiB`.
const MAX_TEMPLATE_BYTES: usize = 16_384;

/// Maximum callout timeout: 60 seconds.
const MAX_TIMEOUT_MS: u64 = 60_000;

/// Default query formatting template.
const DEFAULT_SEARCH_TEMPLATE: &str = "{query}";

/// Default callout timeout in milliseconds (5 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 5_000;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Authentication method for OGX requests.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AuthType {
    /// `Authorization: Bearer {api_key}`.
    Bearer,
    /// Do not send an authorization header.
    None,
}

/// Filter configuration from YAML.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileSearchFilterConfig {
    /// Permit credentials to be sent over plaintext HTTP.
    ///
    /// This is intended only for explicitly trusted local development
    /// endpoints. Production authenticated endpoints must use HTTPS.
    #[serde(default)]
    pub allow_insecure_auth_over_http: bool,

    /// Allow URLs that target local-sensitive addresses.
    ///
    /// DNS names are rejected unless this is enabled because validation
    /// cannot pin the address that the HTTP client will eventually dial.
    #[serde(default)]
    pub allow_private_url: bool,

    /// Per-result formatting template.
    #[serde(default = "default_annotation_template")]
    pub annotation_template: String,

    /// Optional Bearer credential (supports `${ENV_VAR}`).
    ///
    /// Wrapped in [`SecretString`] to prevent accidental logging.
    #[serde(default)]
    pub api_key: Option<SecretString>,

    /// Authentication method.
    ///
    /// When omitted, a configured key selects `bearer`; otherwise
    /// authentication defaults to `none`.
    #[serde(default)]
    pub auth_type: Option<AuthType>,

    /// Model-facing search context template.
    #[serde(default = "default_context_template")]
    pub context_template: String,

    /// Maximum response body size in bytes per callout.
    pub max_response_bytes: Option<usize>,

    /// Maximum cumulative successful response bytes per filter execution.
    pub max_total_response_bytes: Option<usize>,

    /// Behaviour when a vector-store callout fails.
    ///
    /// Named `on_error` because `parse_filter_config` strips
    /// `failure_mode` as a pipeline-structural key.
    pub on_error: Option<OnErrorDef>,

    /// Query formatting template.
    #[serde(default = "default_search_template")]
    pub search_template: String,

    /// Whole-call timeout in milliseconds.
    pub timeout_ms: Option<u64>,

    /// Base URL for the OGX vector store API.
    pub vector_store_url: String,
}

/// Callout error handling policy.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnErrorDef {
    /// Log and continue with partial or empty results.
    Ignore,
    /// Return a 502 error to the client.
    Reject,
}

impl OnErrorDef {
    /// Convert to the core failure mode.
    pub fn to_core(self) -> FailureMode {
        match self {
            Self::Ignore => FailureMode::Open,
            Self::Reject => FailureMode::Closed,
        }
    }
}

/// Validated configuration.
pub(crate) struct ValidatedConfig {
    /// Per-result formatting template.
    pub annotation_template: String,

    /// Prevalidated authorization header value.
    pub authorization: Option<HeaderValue>,

    /// Shared OpenAI-compatible API client.
    pub api_client: ApiClient,

    /// Model-facing search context template.
    pub context_template: String,

    /// Failure mode.
    pub failure_mode: FailureMode,

    /// Maximum response body size per callout.
    pub max_response_bytes: usize,

    /// Maximum cumulative successful response bytes.
    pub max_total_response_bytes: usize,

    /// Query formatting template.
    pub search_template: String,

    /// Whole-call timeout.
    pub timeout: Duration,
}

/// Build validated config from filter config.
#[expect(clippy::too_many_lines, reason = "linear security and resource validation")]
pub(crate) fn build_config(cfg: FileSearchFilterConfig) -> Result<ValidatedConfig, FilterError> {
    let vector_store_url = parse_vector_store_url(&cfg.vector_store_url, cfg.allow_private_url)?;
    let api_key = cfg
        .api_key
        .as_ref()
        .map(|key| resolve_api_key(key.expose_secret()))
        .transpose()?;
    let auth_type = effective_auth_type(cfg.auth_type, api_key.is_some());
    validate_auth_transport(
        &vector_store_url,
        auth_type != AuthType::None,
        cfg.allow_insecure_auth_over_http,
    )?;
    let authorization = build_authorization(auth_type, api_key.as_ref())?;
    let failure_mode = cfg.on_error.unwrap_or(OnErrorDef::Reject).to_core();
    let (max_response_bytes, max_total_response_bytes) =
        response_limits(cfg.max_response_bytes, cfg.max_total_response_bytes)?;
    let timeout_ms = validated_timeout(cfg.timeout_ms)?;

    validate_templates(&cfg.search_template, &cfg.annotation_template, &cfg.context_template)?;
    let api_client = build_api_client(
        &vector_store_url,
        authorization.is_some(),
        failure_mode,
        max_response_bytes,
        timeout_ms,
    )?;

    Ok(ValidatedConfig {
        annotation_template: cfg.annotation_template,
        api_client,
        authorization,
        context_template: cfg.context_template,
        failure_mode,
        max_response_bytes,
        max_total_response_bytes,
        search_template: cfg.search_template,
        timeout: Duration::from_millis(timeout_ms),
    })
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

/// Build the shared API client with breaker behavior disabled.
fn build_api_client(
    vector_store_url: &Url,
    has_authorization: bool,
    failure_mode: FailureMode,
    max_response_bytes: usize,
    timeout_ms: u64,
) -> Result<ApiClient, FilterError> {
    // Core currently counts all non-2xx statuses as breaker failures. Leaving
    // the breaker disabled prevents caller-controlled 4xx responses from
    // disabling vector search for unrelated requests.
    ApiClient::new(ApiClientConfig {
        api_base_url: vector_store_url.as_str().to_owned(),
        callout_config: CalloutConfig {
            circuit_breaker: None,
            failure_mode,
            max_depth: 1,
            max_response_bytes,
            pool_max_idle_per_host: 4,
            status_on_error: 502,
            timeout_ms,
        },
        forward_header_names: if has_authorization {
            vec![http::header::AUTHORIZATION]
        } else {
            Vec::new()
        },
    })
    .map_err(|error| -> FilterError {
        format!("openai_file_search_callout: failed to create API client: {error}").into()
    })
}

/// Validate the required context insertion point.
fn validate_context_template(template: &str) -> Result<(), FilterError> {
    if count_template_placeholders(template, "results") != 1 {
        return Err(
            "openai_file_search_callout: context_template must contain exactly one {results} placeholder".into(),
        );
    }
    Ok(())
}

/// Validate all configured formatting templates and their insertion points.
fn validate_templates(search: &str, annotation: &str, context: &str) -> Result<(), FilterError> {
    for (name, template) in [
        ("search_template", search),
        ("annotation_template", annotation),
        ("context_template", context),
    ] {
        if template.len() > MAX_TEMPLATE_BYTES {
            return Err(format!("openai_file_search_callout: {name} exceeds {MAX_TEMPLATE_BYTES} byte limit").into());
        }
    }
    if count_template_placeholders(search, "query") != 1 {
        return Err("openai_file_search_callout: search_template must contain exactly one {query} placeholder".into());
    }
    validate_context_template(context)
}

/// Resolve and validate the per-call and total response limits.
#[expect(clippy::too_many_lines, reason = "paired limits require ordered validation")]
fn response_limits(per_call: Option<usize>, total: Option<usize>) -> Result<(usize, usize), FilterError> {
    let per_call = per_call.unwrap_or(DEFAULT_MAX_RESPONSE_BYTES);
    if per_call == 0 {
        return Err("openai_file_search_callout: max_response_bytes must be greater than 0".into());
    }
    if per_call > MAX_RESPONSE_BYTES {
        return Err(
            format!("openai_file_search_callout: max_response_bytes must not exceed {MAX_RESPONSE_BYTES}").into(),
        );
    }

    let total = match total {
        Some(limit) => limit,
        None => per_call
            .checked_mul(MAX_CONCURRENT_SEARCHES)
            .ok_or_else(|| -> FilterError {
                "openai_file_search_callout: default max_total_response_bytes overflows usize".into()
            })?
            .min(MAX_TOTAL_RESPONSE_BYTES),
    };
    if total == 0 {
        return Err("openai_file_search_callout: max_total_response_bytes must be greater than 0".into());
    }
    if total < per_call {
        return Err("openai_file_search_callout: max_total_response_bytes must be at least max_response_bytes".into());
    }
    if total > MAX_TOTAL_RESPONSE_BYTES {
        return Err(format!(
            "openai_file_search_callout: max_total_response_bytes must not exceed {MAX_TOTAL_RESPONSE_BYTES}"
        )
        .into());
    }
    Ok((per_call, total))
}

/// Resolve and validate the callout timeout.
fn validated_timeout(configured: Option<u64>) -> Result<u64, FilterError> {
    let timeout_ms = configured.unwrap_or(DEFAULT_TIMEOUT_MS);
    if timeout_ms == 0 {
        return Err("openai_file_search_callout: timeout_ms must be greater than 0".into());
    }
    if timeout_ms > MAX_TIMEOUT_MS {
        return Err(format!("openai_file_search_callout: timeout_ms must not exceed {MAX_TIMEOUT_MS}").into());
    }
    Ok(timeout_ms)
}

/// Build and validate the configured authorization header once.
fn build_authorization(
    auth_type: AuthType,
    api_key: Option<&SecretString>,
) -> Result<Option<HeaderValue>, FilterError> {
    match (auth_type, api_key) {
        (AuthType::None, None) => Ok(None),
        (AuthType::None, Some(_)) => {
            Err("openai_file_search_callout: api_key must be omitted when auth_type is none".into())
        },
        (AuthType::Bearer, None) => {
            Err("openai_file_search_callout: api_key is required for the configured auth_type".into())
        },
        (AuthType::Bearer, Some(key)) if key.expose_secret().is_empty() => {
            Err("openai_file_search_callout: api_key must not be empty".into())
        },
        (AuthType::Bearer, Some(key)) => prefixed_authorization("Bearer ", key),
    }
}

/// Resolve the configured authentication type, retaining legacy key behavior.
fn effective_auth_type(configured: Option<AuthType>, has_api_key: bool) -> AuthType {
    configured.unwrap_or(if has_api_key { AuthType::Bearer } else { AuthType::None })
}

/// Build an authorization header while zeroizing the prefixed temporary.
fn prefixed_authorization(prefix: &str, key: &SecretString) -> Result<Option<HeaderValue>, FilterError> {
    let exposed = key.expose_secret();
    let mut value = Zeroizing::new(String::with_capacity(prefix.len().saturating_add(exposed.len())));
    value.push_str(prefix);
    value.push_str(exposed);
    parse_authorization(value.as_str())
}

/// Parse an authorization value without exposing its contents in errors.
fn parse_authorization(value: &str) -> Result<Option<HeaderValue>, FilterError> {
    let mut value = HeaderValue::from_str(value).map_err(|_error| -> FilterError {
        "openai_file_search_callout: api_key contains characters invalid for an HTTP header value".into()
    })?;
    value.set_sensitive(true);
    Ok(Some(value))
}

/// Resolve an exact `${ENV_VAR}` reference or trim a literal API key.
fn resolve_api_key(raw: &str) -> Result<SecretString, FilterError> {
    let trimmed = raw.trim();
    if let Some(var_name) = trimmed.strip_prefix("${").and_then(|value| value.strip_suffix('}')) {
        std::env::var(var_name).map(SecretString::from).map_err(|_error| {
            FilterError::from(format!(
                "openai_file_search_callout: environment variable {var_name} is missing or not valid Unicode for api_key"
            ))
        })
    } else {
        Ok(SecretString::from(trimmed.to_owned()))
    }
}

/// Reject plaintext transport whenever an authorization header is configured.
fn validate_auth_transport(url: &Url, authenticated: bool, allow_insecure: bool) -> Result<(), FilterError> {
    if authenticated && url.scheme() != "https" && !allow_insecure {
        return Err(
            "openai_file_search_callout: authenticated vector_store_url must use https; \
             set allow_insecure_auth_over_http: true only for trusted local development"
                .into(),
        );
    }
    Ok(())
}

/// Parse the URL and reject targets whose dial destination is not safe.
fn parse_vector_store_url(raw: &str, allow_private: bool) -> Result<Url, FilterError> {
    api_client::validate_base_url("openai_file_search_callout", raw, allow_private)?;
    let url = Url::parse(raw).map_err(|error| -> FilterError {
        format!("openai_file_search_callout: vector_store_url is not a valid URL: {error}").into()
    })?;
    Ok(url)
}

/// Default per-result formatting template.
fn default_annotation_template() -> String {
    DEFAULT_ANNOTATION_TEMPLATE.to_owned()
}

/// Default model-facing search context template.
fn default_context_template() -> String {
    DEFAULT_CONTEXT_TEMPLATE.to_owned()
}

/// Default query formatting template.
fn default_search_template() -> String {
    DEFAULT_SEARCH_TEMPLATE.to_owned()
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::unwrap_used, reason = "tests")]
mod tests {
    #[cfg(unix)]
    use std::{ffi::OsString, os::unix::ffi::OsStringExt as _, process::Command};

    use super::*;

    #[test]
    fn config_security_requires_https_for_authentication() {
        let http = Url::parse("http://8.8.8.8").unwrap();
        let https = Url::parse("https://8.8.8.8").unwrap();

        assert!(validate_auth_transport(&http, true, false).is_err());
        assert!(validate_auth_transport(&http, true, true).is_ok());
        assert!(validate_auth_transport(&http, false, false).is_ok());
        assert!(validate_auth_transport(&https, true, false).is_ok());
    }

    #[test]
    fn config_security_enforces_resource_ceilings() {
        assert!(response_limits(Some(MAX_RESPONSE_BYTES.saturating_add(1)), None).is_err());
        assert!(response_limits(None, Some(MAX_TOTAL_RESPONSE_BYTES.saturating_add(1))).is_err());
        assert!(validated_timeout(Some(MAX_TIMEOUT_MS.saturating_add(1))).is_err());

        let oversized = "x".repeat(MAX_TEMPLATE_BYTES.saturating_add(1));
        assert!(validate_templates(&oversized, "{content}", "{results}").is_err());
        assert!(validate_templates("{query}{query}", "{content}", "{results}").is_err());
        assert!(validate_templates("{{query}}", "{content}", "{results}").is_err());
        assert!(validate_templates("{query}", "{content}", "{results}").is_ok());
    }

    #[test]
    fn context_template_requires_one_renderable_results_placeholder() {
        assert!(
            validate_context_template("{{results}}").is_err(),
            "nested braces must not satisfy the results insertion point"
        );
        assert!(
            validate_context_template("before {results} after").is_ok(),
            "one exact parsed placeholder must be accepted"
        );
        assert!(
            validate_context_template("{results} and {results}").is_err(),
            "multiple parsed insertion points must be rejected"
        );
    }

    #[test]
    fn config_security_keeps_resolved_keys_secret_and_headers_sensitive() {
        let key = resolve_api_key("  secret  ").unwrap();
        assert_eq!(key.expose_secret(), "secret");

        let authorization = build_authorization(AuthType::Bearer, Some(&key)).unwrap().unwrap();
        assert!(authorization.is_sensitive());
        assert_eq!(authorization, "Bearer secret");
    }

    #[cfg(unix)]
    #[test]
    fn config_security_non_unicode_environment_key_does_not_leak_value() {
        const CHILD_MARKER: &str = "PRAXIS_FILE_SEARCH_NON_UNICODE_CHILD";
        const KEY_NAME: &str = "PRAXIS_FILE_SEARCH_NON_UNICODE_KEY";
        const SECRET_FRAGMENT: &str = "file-search-secret";

        if std::env::var_os(CHILD_MARKER).is_some() {
            let error = resolve_api_key(&format!("${{{KEY_NAME}}}")).unwrap_err();
            for rendered in [error.to_string(), format!("{error:?}")] {
                assert!(rendered.contains(KEY_NAME));
                assert!(!rendered.contains(SECRET_FRAGMENT));
            }
            return;
        }

        let mut secret = vec![0xFF];
        secret.extend_from_slice(SECRET_FRAGMENT.as_bytes());
        let output = Command::new(std::env::current_exe().unwrap())
            .arg("config_security_non_unicode_environment_key_does_not_leak_value")
            .arg("--nocapture")
            .env(CHILD_MARKER, "1")
            .env(KEY_NAME, OsString::from_vec(secret))
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains(SECRET_FRAGMENT));
        assert!(!String::from_utf8_lossy(&output.stderr).contains(SECRET_FRAGMENT));
    }
}
