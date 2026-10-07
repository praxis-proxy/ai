// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the Responses request processor filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::OnInvalidBehavior};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Behavior Enums
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// ResponsesClassificationHeaders
// -----------------------------------------------------------------------------

/// Configurable header names for promoted classification facts.
///
/// Transport, credential, API-key, and other internal `x-praxis-*` names
/// are rejected. Each field may use its dedicated default or a custom
/// non-`x-praxis-*` header.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponsesClassificationHeaders {
    /// Header name for the detected format (e.g. `openai_responses`, `openai_chat_completions`).
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-format` remains allowed.
    #[serde(default = "default_format_header")]
    pub format: Option<String>,

    /// Header name for the extracted model value.
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-model` remains allowed. Must not overwrite other
    /// classification facts such as `x-praxis-ai-format`.
    #[serde(default = "default_model_header")]
    pub model: Option<String>,

    /// Header name for the extracted stream flag.
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-ai-stream` remains allowed.
    #[serde(default = "default_stream_header")]
    pub stream: Option<String>,

    /// Header name for the computed mode (`stateless` or `stateful`).
    ///
    /// Must not be a hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` header. Dedicated default
    /// `x-praxis-responses-mode` remains allowed.
    #[serde(default = "default_mode_header")]
    pub mode: Option<String>,
}

impl Default for ResponsesClassificationHeaders {
    fn default() -> Self {
        Self {
            format: default_format_header(),
            model: default_model_header(),
            stream: default_stream_header(),
            mode: default_mode_header(),
        }
    }
}

/// Default format header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_format_header() -> Option<String> {
    Some("x-praxis-ai-format".to_owned())
}

/// Default model header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_model_header() -> Option<String> {
    Some("x-praxis-ai-model".to_owned())
}

/// Default stream header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_stream_header() -> Option<String> {
    Some("x-praxis-ai-stream".to_owned())
}

/// Default mode header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_mode_header() -> Option<String> {
    Some("x-praxis-responses-mode".to_owned())
}

// -----------------------------------------------------------------------------
// ResponsesClassificationConfig
// -----------------------------------------------------------------------------

/// Shared classification and promotion settings for the request processor.
///
/// The create request processor flattens these into [`ResponsesRequestConfig`]
/// so a chain configures promotion headers and `on_invalid` the same way
/// whether the filter runs as a pre-routing fact publisher or the managed owner.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponsesClassificationConfig {
    /// Behavior when the body cannot be classified.
    #[serde(default = "OnInvalidBehavior::default_continue")]
    pub on_invalid: OnInvalidBehavior,

    /// Header names for promoted classification facts.
    ///
    /// Must not be hop-by-hop, framing, Host, credential, API-key, or
    /// other internal `x-praxis-*` names. Dedicated defaults remain allowed.
    #[serde(default)]
    pub headers: ResponsesClassificationHeaders,
}

/// Configuration for the create request processor.
///
/// Extends the shared classification settings with the one option that gates
/// the managed-path lifecycle, so a pre-routing fact publisher and the managed
/// owner can be configured from the same filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent managed-path lifecycle gates: owner, parse-cache, parse-discard"
)]
pub(crate) struct ResponsesRequestConfig {
    /// Classification and promotion settings, shared with the classifier.
    #[serde(flatten)]
    pub shared: ResponsesClassificationConfig,

    /// Whether this entry owns the managed-path request lifecycle.
    ///
    /// On by default, because the stateful Responses filters read the state it
    /// builds. When enabled the filter is the managed owner: it initializes
    /// `ResponsesState` and enforces the managed-path policy that rejects
    /// provider-owned `background`/`prompt` and conflicting history selectors.
    ///
    /// A pre-routing fact publisher sets this to `false`: it classifies the
    /// body, promotes the routing facts, and — when [`Self::cache_parse_for_owner`]
    /// is also set — hands its parse to a later managed entry, but mints no
    /// identifiers, resolves no conversation, and enforces no managed-path
    /// policy — so provider-owned traffic that the router may still bind to a
    /// direct upstream keeps its fields intact.
    ///
    /// Classification metadata, headers, and filter results are published
    /// either way, so routing is unaffected.
    #[serde(default = "default_initialize_state")]
    pub initialize_state: bool,

    /// Whether a pre-routing fact publisher retains its single parse for a later
    /// managed owner.
    ///
    /// Off by default, so a facts publisher drops its parse once the routing
    /// facts are promoted and retains nothing request-sized while the original
    /// body is forwarded. A facts-only chain with no managed owner — a pure
    /// routing gateway, say — keeps this default and leaks no parse.
    ///
    /// Set this to `true` on a pre-routing fact publisher (`initialize_state:
    /// false`) that is followed by the managed owner (`initialize_state: true`)
    /// in the same chain: the publisher then caches its parse in request
    /// extensions and the owner reuses it after binding, so a managed create
    /// body is deserialized exactly once across both phases rather than parsed
    /// again. It has no effect on a managed owner, which consumes its own parse
    /// directly and never publishes one for another entry.
    #[serde(default)]
    pub cache_parse_for_owner: bool,

    /// Whether this entry drops a parse an earlier publisher cached, after binding.
    ///
    /// Off by default. Set this to `true` on an `initialize_state: false` instance
    /// placed on a provider-owned (direct) route whose managed owner never runs —
    /// for example inside a direct-upstream terminal branch. A pre-routing publisher
    /// that set [`Self::cache_parse_for_owner`] hands its single parse to the managed
    /// owner, which consumes it after binding; but a request the router binds direct
    /// skips that owner, so the cached parse would otherwise sit in request
    /// extensions for the whole forward with nothing to consume it. Discarding it at
    /// the header phase — which runs after the route is bound — frees it on the
    /// direct path while managed routes keep the parse for their single
    /// deserialization. Rejected together with `initialize_state: true` (the owner
    /// consumes its own parse) or with `cache_parse_for_owner` on the same entry.
    #[serde(default)]
    pub discard_cached_parse: bool,
}

/// The filter owns the managed-path lifecycle unless a chain opts out.
///
/// Only the create request processor honours this, and that filter is compiled
/// in with the Responses feature.
const fn default_initialize_state() -> bool {
    true
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn build_config(
    filter: &str,
    cfg: ResponsesClassificationConfig,
) -> Result<ResponsesClassificationConfig, FilterError> {
    validate_classification_headers(filter, &cfg.headers)?;
    Ok(cfg)
}

/// Reject `discard_cached_parse` combinations that can never be correct.
///
/// Discarding a cached parse only makes sense on a pre-routing publisher
/// (`initialize_state: false`) that itself does not cache — it drops a parse an
/// *earlier* publisher left for a managed owner the direct route never reaches.
/// A managed owner (`initialize_state: true`) consumes its own cache, and a
/// publisher that caches (`cache_parse_for_owner: true`) is the producer, not the
/// discarder; combining either with discard is a configuration error.
pub(crate) fn validate_request_config(filter: &str, cfg: &ResponsesRequestConfig) -> Result<(), FilterError> {
    if cfg.discard_cached_parse && cfg.initialize_state {
        return Err(FilterError::from(format!(
            "{filter}: discard_cached_parse requires initialize_state: false; a managed owner consumes its own parse"
        )));
    }
    if cfg.discard_cached_parse && cfg.cache_parse_for_owner {
        return Err(FilterError::from(format!(
            "{filter}: discard_cached_parse and cache_parse_for_owner are mutually exclusive; an entry caches a parse or discards one, not both"
        )));
    }
    Ok(())
}

/// Validate dedicated names and reject collisions across header fields.
fn validate_classification_headers(filter: &str, headers: &ResponsesClassificationHeaders) -> Result<(), FilterError> {
    for (field, name, dedicated) in [
        ("format", headers.format.as_deref(), "x-praxis-ai-format"),
        ("model", headers.model.as_deref(), "x-praxis-ai-model"),
        ("stream", headers.stream.as_deref(), "x-praxis-ai-stream"),
        ("mode", headers.mode.as_deref(), "x-praxis-responses-mode"),
    ] {
        crate::promotion::validate_dedicated_promotion_header(filter, field, name, &[dedicated])?;
    }
    crate::promotion::reject_duplicate_promotion_fields(
        filter,
        &[
            ("format", headers.format.as_deref()),
            ("model", headers.model.as_deref()),
            ("stream", headers.stream.as_deref()),
            ("mode", headers.mode.as_deref()),
        ],
    )
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -- Serde defaults -------------------------------------------------------

    #[test]
    fn serde_defaults_responses_format_config() {
        let cfg: ResponsesClassificationConfig = serde_yaml::from_str("{}").unwrap();

        assert_eq!(cfg.on_invalid, OnInvalidBehavior::Continue);
    }

    #[test]
    fn cache_parse_for_owner_defaults_off() {
        let cfg: ResponsesRequestConfig = serde_yaml::from_str("{}").unwrap();
        assert!(
            !cfg.cache_parse_for_owner,
            "a facts publisher must retain no parse unless a chain opts in"
        );
        assert!(cfg.initialize_state, "the managed owner default is unchanged");
    }

    #[test]
    fn cache_parse_for_owner_opt_in_parses() {
        let cfg: ResponsesRequestConfig =
            serde_yaml::from_str("initialize_state: false\ncache_parse_for_owner: true\n").unwrap();
        assert!(!cfg.initialize_state);
        assert!(
            cfg.cache_parse_for_owner,
            "a pre-routing publisher can opt in to the deserialize-once handoff"
        );
    }

    #[test]
    fn discard_cached_parse_defaults_off() {
        let cfg: ResponsesRequestConfig = serde_yaml::from_str("{}").unwrap();
        assert!(
            !cfg.discard_cached_parse,
            "an entry must not discard a cached parse unless a chain opts in"
        );
    }

    #[test]
    fn validate_request_config_accepts_discard_on_a_facts_publisher() {
        let cfg: ResponsesRequestConfig =
            serde_yaml::from_str("initialize_state: false\ndiscard_cached_parse: true\n").unwrap();
        assert!(
            validate_request_config("openai_responses_request", &cfg).is_ok(),
            "a direct-route publisher may discard an earlier cached parse"
        );
    }

    #[test]
    fn validate_request_config_rejects_discard_on_the_managed_owner() {
        let cfg: ResponsesRequestConfig = serde_yaml::from_str("discard_cached_parse: true\n").unwrap();
        let err = validate_request_config("openai_responses_request", &cfg).unwrap_err();
        assert!(
            err.to_string().contains("initialize_state: false"),
            "the managed owner consumes its own parse and cannot discard: {err}"
        );
    }

    #[test]
    fn validate_request_config_rejects_discard_with_caching() {
        let cfg: ResponsesRequestConfig =
            serde_yaml::from_str("initialize_state: false\ncache_parse_for_owner: true\ndiscard_cached_parse: true\n")
                .unwrap();
        let err = validate_request_config("openai_responses_request", &cfg).unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "caching and discarding on one entry must be rejected: {err}"
        );
    }

    #[test]
    fn responses_format_headers_defaults() {
        let h = ResponsesClassificationHeaders::default();
        assert_eq!(h.format.as_deref(), Some("x-praxis-ai-format"));
        assert_eq!(h.model.as_deref(), Some("x-praxis-ai-model"));
        assert_eq!(h.stream.as_deref(), Some("x-praxis-ai-stream"));
        assert_eq!(h.mode.as_deref(), Some("x-praxis-responses-mode"));
    }

    // -- deny_unknown_fields --------------------------------------------------

    #[test]
    fn deny_unknown_fields_responses_format_config() {
        let res = serde_yaml::from_str::<ResponsesClassificationConfig>(
            r#"
bogus: true
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn deny_unknown_fields_responses_format_headers() {
        let res = serde_yaml::from_str::<ResponsesClassificationHeaders>(
            r#"
format: x-test
extra: true
"#,
        );
        assert!(res.is_err());
    }

    // -- build_config ---------------------------------------------------------

    #[test]
    fn build_config_minimal_ok() {
        let cfg: ResponsesClassificationConfig = serde_yaml::from_str("{}").unwrap();
        assert!(build_config("openai_responses_request", cfg).is_ok());
    }

    #[test]
    fn build_config_invalid_header_name_rejected() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: Some("not a valid header!".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("not a valid HTTP header name"),
            "expected invalid header error, got: {err}"
        );
    }

    #[test]
    fn build_config_valid_custom_headers_ok() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: Some("x-custom-format".into()),
                model: Some("x-custom-model".into()),
                stream: Some("x-custom-stream".into()),
                mode: Some("x-custom-mode".into()),
            },
        };
        assert!(build_config("openai_responses_request", cfg).is_ok());
    }

    #[test]
    fn build_config_authorization_header_rejected() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: default_format_header(),
                model: Some("authorization".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("authorization"),
            "authorization promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_api_key_header_rejected() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: default_format_header(),
                model: Some("x-api-key".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-api-key"),
            "x-api-key promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_unrelated_internal_header_rejected() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: Some("x-praxis-route".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-route"),
            "unrelated x-praxis-* promotion header should be rejected: {err}"
        );
    }

    #[test]
    fn build_config_model_header_rejects_format_routing_fact() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: default_format_header(),
                model: Some("x-praxis-ai-format".into()),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-format"),
            "client-derived model must not overwrite format routing: {err}"
        );
    }

    #[test]
    fn build_config_format_header_rejects_model_rewrite_fact() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: Some("x-praxis-ai-effective-model".into()),
                model: default_model_header(),
                stream: default_stream_header(),
                mode: default_mode_header(),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("x-praxis-ai-effective-model"),
            "format fact must not overwrite model-rewrite routing: {err}"
        );
    }

    #[test]
    fn build_config_accepts_dedicated_defaults() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders::default(),
        };
        assert!(
            build_config("openai_responses_request", cfg).is_ok(),
            "dedicated classification defaults should remain allowed"
        );
    }

    #[test]
    fn build_config_rejects_duplicate_promotion_headers() {
        let cfg = ResponsesClassificationConfig {
            on_invalid: OnInvalidBehavior::default_continue(),
            headers: ResponsesClassificationHeaders {
                format: Some("x-foo".into()),
                model: Some("X-Foo".into()),
                stream: Some("x-praxis-ai-stream".into()),
                mode: Some("x-praxis-responses-mode".into()),
            },
        };
        let err = build_config("openai_responses_request", cfg).unwrap_err();
        assert!(
            err.to_string().contains("same header name"),
            "duplicate format and model headers should be rejected: {err}"
        );
    }

    // -- null header disables promotion ---------------------------------------

    #[test]
    fn null_header_disables_promotion() {
        let cfg: ResponsesClassificationConfig = serde_yaml::from_str(
            r#"
headers:
  format: null
  model: null
  stream: null
  mode: null
"#,
        )
        .unwrap();

        assert!(cfg.headers.format.is_none());
        assert!(cfg.headers.model.is_none());
        assert!(cfg.headers.stream.is_none());
        assert!(cfg.headers.mode.is_none());
        assert!(build_config("openai_responses_request", cfg).is_ok());
    }
}
