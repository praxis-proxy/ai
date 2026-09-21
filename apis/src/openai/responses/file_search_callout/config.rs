// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Filter configuration for `file_search_callout`.

use std::time::Duration;

use praxis_core::config::ChainRef;
use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use reqwest::Url;
use serde::Deserialize;

use super::client::MAX_CONCURRENT_SEARCHES;
use crate::{callout_policy::OnFailure, openai::api_client, subrequest::SubRequestClient};

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
    /// Outbound filter chain every vector-store sub-request runs through.
    ///
    /// The chain carries cross-cutting concerns (observability, credential
    /// injection, security) and is bound into a prebuilt pipeline at
    /// registration time via [`ChainBindingContext::bind_chain`]. The
    /// destination is supplied by `vector_store_url`, so the chain never needs
    /// an upstream-selecting filter. Private-address gating is centralized in
    /// `insecure_options.allow_private_upstreams`.
    ///
    /// Optional. Callouts always run through the shared sub-request executor;
    /// this chain only adds filters along the way. When omitted it defaults to
    /// an empty inline chain (pure passthrough) via `default_outbound_chain`.
    /// Provide it only to attach cross-cutting concerns.
    ///
    /// When provided, it must be defined **inline** (`outbound_chain: { name:
    /// ..., filters: [...] }`). A `Named` reference to a top-level `filter_chains`
    /// entry is rejected at construction by [`require_inline_outbound_chain`]: this
    /// filter runs nested inside an `iterative_request_router` step, whose pipeline
    /// is built with an empty named-chain map, so a named reference could never
    /// resolve there.
    ///
    /// [`ChainBindingContext::bind_chain`]: praxis_filter::ChainBindingContext::bind_chain
    #[serde(default = "default_outbound_chain")]
    pub outbound_chain: ChainRef,

    /// Behaviour when a vector-store callout fails.
    pub on_failure: Option<OnFailure>,

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

/// Default `outbound_chain` when the field is omitted: an empty inline chain.
///
/// `outbound_chain` is optional. Callouts still run through the shared
/// sub-request executor (that dispatch is unconditional); an empty chain simply
/// applies no extra filters (pure passthrough). Operators supply a chain only to
/// attach cross-cutting concerns such as credential injection, tracing, or
/// request tagging. The name is a label only — inline chains are not looked up,
/// so it never needs to be globally unique.
fn default_outbound_chain() -> ChainRef {
    ChainRef::Inline {
        name: "openai_file_search_callout_outbound".to_owned(),
        filters: Vec::new(),
    }
}

/// Validated configuration.
pub(crate) struct ValidatedConfig {
    /// Vector-store API base URL (trailing slash stripped).
    pub base_url: String,

    /// Shared sub-request transport driving the outbound chain.
    pub subrequest_client: SubRequestClient,

    /// Header names forwarded from the original request to the vector store.
    pub forward_header_names: Vec<http::HeaderName>,

    /// Search failure handling policy.
    pub on_failure: OnFailure,

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
    let base_url = parse_vector_store_url(&cfg.vector_store_url)?;
    let on_failure = cfg.on_failure.unwrap_or(OnFailure::Closed);
    let (max_response_bytes, max_total_response_bytes) =
        response_limits(cfg.max_response_bytes, cfg.max_total_response_bytes)?;
    let max_state_bytes = validated_state_limit(cfg.max_state_bytes)?;
    let timeout_ms = validated_timeout(cfg.timeout_ms)?;
    let mut forward_headers = cfg.forward_headers.clone();
    api_client::validate_forward_headers("openai_file_search_callout", &mut forward_headers)?;
    let forward_header_names = forward_headers
        .iter()
        .filter_map(|name| http::HeaderName::from_bytes(name.as_bytes()).ok())
        .collect();

    Ok(ValidatedConfig {
        base_url,
        subrequest_client: client,
        forward_header_names,
        on_failure,
        max_response_bytes,
        max_total_response_bytes,
        max_state_bytes,
        timeout: Duration::from_millis(timeout_ms),
    })
}

/// Reject a `Named` outbound-chain reference, requiring an inline chain.
///
/// `openai_file_search_callout` runs nested inside an `iterative_request_router`
/// step, and IRR builds each step's pipeline with an empty top-level named-chain
/// map. A `Named` reference (`outbound_chain: my-chain`) therefore can never
/// resolve inside a step and would fail pipeline construction with a confusing
/// "unknown chain" error. Require the chain inline instead
/// (`outbound_chain: { name: ..., filters: [...] }`), which embeds its filters
/// directly and needs no lookup.
///
/// # Errors
///
/// Returns [`FilterError`] when `outbound_chain` is a [`ChainRef::Named`].
pub(crate) fn require_inline_outbound_chain(outbound_chain: &ChainRef) -> Result<(), FilterError> {
    if let ChainRef::Named(name) = outbound_chain {
        return Err(format!(
            "openai_file_search_callout: outbound_chain must be defined inline \
             ({{ name, filters }}); a named reference ('{name}') cannot resolve \
             inside the iterative_request_router step this filter runs in"
        )
        .into());
    }
    Ok(())
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

/// Parse the URL, reject structurally-unsafe targets, and return the
/// normalized base URL with any trailing slash stripped.
///
/// Structural validation always rejects a non-`http(s)` scheme, embedded
/// userinfo, and a query or fragment. Private-address gating is *not* decided
/// here: `allow_private = true` defers every private/loopback decision — for
/// both literal IPs and resolved DNS names — to the connect-time
/// `prepare_url_target` hook, which is the sole SSRF gate and honours the
/// outbound pipeline's `insecure_options.allow_private_upstreams`. Deciding it
/// at startup would be unable to see that flag (it is applied later via
/// `apply_insecure_options`), so a literal private target could never be
/// permitted even with the central opt-in set.
fn parse_vector_store_url(raw: &str) -> Result<String, FilterError> {
    api_client::validate_base_url("openai_file_search_callout", raw, true)?;
    let url = Url::parse(raw).map_err(|error| -> FilterError {
        format!("openai_file_search_callout: vector_store_url is not a valid URL: {error}").into()
    })?;
    Ok(url.as_str().trim_end_matches('/').to_owned())
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

    #[test]
    fn config_rejects_named_outbound_chain() {
        let named = ChainRef::Named("vector-store-outbound".to_owned());
        let error = require_inline_outbound_chain(&named).unwrap_err();
        assert!(
            error.to_string().contains("must be defined inline"),
            "error should explain the inline requirement: {error}"
        );
    }

    #[test]
    fn config_accepts_inline_outbound_chain() {
        let inline = ChainRef::Inline {
            name: "vector-store-outbound".to_owned(),
            filters: Vec::new(),
        };
        require_inline_outbound_chain(&inline).unwrap();
    }

    #[test]
    fn config_defaults_omitted_outbound_chain_to_empty_inline() {
        // `outbound_chain` is optional: omitting it yields an empty inline chain
        // (pure passthrough) rather than a config error.
        let cfg: FileSearchFilterConfig = serde_yaml::from_str("vector_store_url: http://vector-store:8321\n").unwrap();
        assert!(
            matches!(&cfg.outbound_chain, ChainRef::Inline { filters, .. } if filters.is_empty()),
            "omitted outbound_chain should default to an empty inline chain"
        );
        // The default must satisfy the inline-only requirement enforced at build.
        require_inline_outbound_chain(&cfg.outbound_chain).unwrap();
    }
}
