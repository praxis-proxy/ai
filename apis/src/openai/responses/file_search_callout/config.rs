// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Filter configuration for `file_search_callout`.

use std::time::Duration;

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use reqwest::Url;
use serde::Deserialize;

use super::client::MAX_CONCURRENT_SEARCHES;
use crate::{
    openai::{
        api_client::{self, ApiClient, ApiClientConfig},
        responses::config_validation::FailureMode,
    },
    subrequest::SubRequestClient,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum response body size: 10 MiB.
const DEFAULT_MAX_RESPONSE_BYTES: usize = 10_485_760; // 10 MiB

/// Maximum response body size for one callout: 64 MiB.
const MAX_RESPONSE_BYTES: usize = MAX_JSON_BODY_BYTES;

/// Maximum successful wire bytes retained across one execution: 64 MiB.
const MAX_TOTAL_RESPONSE_BYTES: usize = MAX_JSON_BODY_BYTES;

/// Default maximum combined router and file-search continuation state: 50 MiB.
const DEFAULT_MAX_STATE_BYTES: usize = 52_428_800;

/// Maximum combined continuation state: 256 MiB.
const MAX_STATE_BYTES: usize = 268_435_456;

/// Maximum callout timeout: 60 seconds.
const MAX_TIMEOUT_MS: u64 = 60_000;

/// Default callout timeout in milliseconds (5 seconds).
const DEFAULT_TIMEOUT_MS: u64 = 5_000;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Filter configuration from YAML.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileSearchFilterConfig {
    /// Allow URLs that target local-sensitive addresses.
    ///
    /// DNS names are rejected unless this is enabled because validation
    /// cannot pin the address that the HTTP client will eventually dial.
    #[serde(default)]
    pub allow_private_url: bool,

    /// Behaviour when a vector-store callout fails.
    pub callout_failure_mode: Option<FailureMode>,

    /// Headers to forward from the original request to the
    /// vector store API for authentication and tenant isolation.
    /// No downstream headers are forwarded by default.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Maximum response body size in bytes per callout.
    pub max_response_bytes: Option<usize>,

    /// Maximum cumulative successful response bytes per filter execution.
    pub max_total_response_bytes: Option<usize>,

    /// Maximum combined iterative-router and file-search continuation
    /// bytes. The filter's value may differ from the enclosing
    /// `iterative_request_router`; the smaller limit wins at runtime.
    pub max_state_bytes: Option<usize>,

    /// Whole-call timeout in milliseconds.
    pub timeout_ms: Option<u64>,

    /// Base URL for the vector store API.
    pub vector_store_url: String,
}

/// Validated configuration.
pub(crate) struct ValidatedConfig {
    /// Shared OpenAI-compatible API client.
    pub api_client: ApiClient,

    /// Search failure handling policy.
    pub failure_mode: FailureMode,

    /// Maximum response body size per callout.
    pub max_response_bytes: usize,

    /// Maximum cumulative successful response bytes.
    pub max_total_response_bytes: usize,

    /// Maximum combined iterative-router and file-search continuation bytes.
    pub max_state_bytes: usize,

    /// Whole-call timeout.
    pub timeout: Duration,
}

/// Build validated config from filter config with a shared sub-request
/// client.
pub(crate) fn build_config_with_client(
    cfg: &FileSearchFilterConfig,
    client: SubRequestClient,
) -> Result<ValidatedConfig, FilterError> {
    let vector_store_url = parse_vector_store_url(&cfg.vector_store_url, cfg.allow_private_url)?;
    let failure_mode = cfg.callout_failure_mode.unwrap_or(FailureMode::Closed);
    let (max_response_bytes, max_total_response_bytes) =
        response_limits(cfg.max_response_bytes, cfg.max_total_response_bytes)?;
    let max_state_bytes = validated_state_limit(cfg.max_state_bytes)?;
    let timeout_ms = validated_timeout(cfg.timeout_ms)?;
    let mut forward_headers = cfg.forward_headers.clone();
    api_client::validate_forward_headers("openai_file_search_callout", &mut forward_headers)?;

    let api_client = build_api_client(
        &vector_store_url,
        client,
        &forward_headers,
        max_response_bytes,
        timeout_ms,
    );

    Ok(ValidatedConfig {
        api_client,
        failure_mode,
        max_response_bytes,
        max_total_response_bytes,
        max_state_bytes,
        timeout: Duration::from_millis(timeout_ms),
    })
}

/// Build validated config with a dedicated per-filter sub-request client.
pub(crate) fn build_config(cfg: &FileSearchFilterConfig) -> Result<ValidatedConfig, FilterError> {
    build_config_with_client(
        cfg,
        SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None)),
    )
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

/// Resolve and validate the combined continuation-state limit.
fn validated_state_limit(configured: Option<usize>) -> Result<usize, FilterError> {
    let limit = configured.unwrap_or(DEFAULT_MAX_STATE_BYTES);
    if limit == 0 {
        return Err("openai_file_search_callout: max_state_bytes must be greater than 0".into());
    }
    if limit > MAX_STATE_BYTES {
        return Err(format!("openai_file_search_callout: max_state_bytes must not exceed {MAX_STATE_BYTES}").into());
    }
    Ok(limit)
}

/// Build the shared API client from the validated URL and sub-request client.
fn build_api_client(
    vector_store_url: &Url,
    client: SubRequestClient,
    forward_headers: &[String],
    max_response_bytes: usize,
    timeout_ms: u64,
) -> ApiClient {
    ApiClient::new(ApiClientConfig {
        api_base_url: vector_store_url.as_str().to_owned(),
        client,
        timeout: Duration::from_millis(timeout_ms),
        max_response_bytes,
        forward_header_names: forward_headers
            .iter()
            .filter_map(|name| http::HeaderName::from_bytes(name.as_bytes()).ok())
            .collect(),
    })
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

/// Parse the URL and reject targets whose dial destination is not safe.
fn parse_vector_store_url(raw: &str, allow_private: bool) -> Result<Url, FilterError> {
    api_client::validate_base_url("openai_file_search_callout", raw, allow_private)?;
    let url = Url::parse(raw).map_err(|error| -> FilterError {
        format!("openai_file_search_callout: vector_store_url is not a valid URL: {error}").into()
    })?;
    Ok(url)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn config_security_enforces_resource_ceilings() {
        assert!(response_limits(Some(MAX_RESPONSE_BYTES.saturating_add(1)), None).is_err());
        assert!(response_limits(None, Some(MAX_TOTAL_RESPONSE_BYTES.saturating_add(1))).is_err());
        assert!(validated_timeout(Some(MAX_TIMEOUT_MS.saturating_add(1))).is_err());
    }
}
