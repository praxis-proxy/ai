// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Deserialized YAML configuration types for the model rewrite filter.

use std::collections::HashMap;

use praxis_filter::{
    FilterError, body::DEFAULT_JSON_BODY_MAX_BYTES,
    builtins::http::payload_processing::config_validation::validate_max_body_bytes,
};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// ModelRewriteConfig
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the model rewrite filter.
///
/// ```yaml
/// filter: openai_responses_model_rewrite
/// default_model: "llama-3.3-70b"
/// model_aliases:
///   "codex-mini-latest": "llama-3.3-70b"
///   "gpt-4.1-*": "qwen-2.5-72b"
///   "gpt-4.1-mini": "qwen-2.5-72b"
/// max_body_bytes: 10485760
/// on_invalid: continue
/// headers:
///   effective_model: x-praxis-ai-effective-model
///   original_model: x-praxis-ai-original-model
/// ```
///
/// Quote wildcard alias keys in YAML, such as `"gpt-4.1-*"`, so `*` is
/// parsed as a literal character rather than YAML alias syntax. The examples
/// quote all alias keys for consistency.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelRewriteConfig {
    /// Model name to inject when the request body has no `model`
    /// field or when the field is `null`.
    #[serde(default)]
    pub default_model: Option<String>,

    /// Header names for promoted model values.
    #[serde(default)]
    pub headers: ModelRewriteHeaders,

    /// Maximum request body size to buffer before parsing.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Map from client-facing model names or single-wildcard patterns
    /// to backend model names. Quote wildcard keys in YAML. Exact aliases win
    /// before wildcard aliases; wildcard aliases are matched by literal specificity.
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,

    /// Behavior when the body is not valid JSON.
    #[serde(default)]
    pub on_invalid: OnInvalidBehavior,
}

/// Default for `max_body_bytes`.
fn default_max_body_bytes() -> usize {
    DEFAULT_JSON_BODY_MAX_BYTES
}

// -----------------------------------------------------------------------------
// ModelRewriteHeaders
// -----------------------------------------------------------------------------

/// Configurable header names for promoted model values.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelRewriteHeaders {
    /// Header name for the effective (post-rewrite) model value.
    #[serde(default = "default_effective_model_header")]
    pub effective_model: Option<String>,

    /// Header name for the original (pre-rewrite) model value.
    #[serde(default = "default_original_model_header")]
    pub original_model: Option<String>,
}

impl Default for ModelRewriteHeaders {
    fn default() -> Self {
        Self {
            effective_model: default_effective_model_header(),
            original_model: default_original_model_header(),
        }
    }
}

/// Default effective model header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_effective_model_header() -> Option<String> {
    Some("x-praxis-ai-effective-model".to_owned())
}

/// Default original model header name.
#[expect(
    clippy::unnecessary_wraps,
    reason = "serde default functions require Option return type"
)]
fn default_original_model_header() -> Option<String> {
    Some("x-praxis-ai-original-model".to_owned())
}

// -----------------------------------------------------------------------------
// OnInvalidBehavior
// -----------------------------------------------------------------------------

/// Behavior when the request body cannot be parsed as JSON.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum OnInvalidBehavior {
    /// Pass the original body through unchanged.
    #[default]
    Continue,

    /// Return HTTP 400.
    Reject,
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------

/// Validate a parsed config, returning an error for invalid values.
///
/// # Errors
///
/// Returns [`FilterError`] when the config is invalid.
///
/// [`FilterError`]: praxis_filter::FilterError
pub(super) fn validate_config(cfg: &ModelRewriteConfig) -> Result<(), FilterError> {
    if cfg.default_model.is_none() && cfg.model_aliases.is_empty() {
        return Err(
            "openai_responses_model_rewrite: at least one of 'default_model' or 'model_aliases' must be configured"
                .into(),
        );
    }

    if let Some(dm) = &cfg.default_model
        && dm.trim().is_empty()
    {
        return Err("openai_responses_model_rewrite: 'default_model' must not be empty".into());
    }

    validate_aliases(&cfg.model_aliases)?;
    validate_max_body_bytes("openai_responses_model_rewrite", cfg.max_body_bytes)?;
    validate_header_name("effective_model", cfg.headers.effective_model.as_deref())?;
    validate_header_name("original_model", cfg.headers.original_model.as_deref())?;

    Ok(())
}

/// Validate alias map entries.
fn validate_aliases(aliases: &HashMap<String, String>) -> Result<(), FilterError> {
    for (source, target) in aliases {
        if source.is_empty() {
            return Err("openai_responses_model_rewrite: alias source name must not be empty".into());
        }
        if source.chars().filter(|&c| c == '*').count() > 1 {
            return Err(format!(
                "openai_responses_model_rewrite: alias source pattern '{source}' must contain at most one '*'",
            )
            .into());
        }
        if target.is_empty() {
            return Err(
                format!("openai_responses_model_rewrite: alias target for '{source}' must not be empty").into(),
            );
        }
    }
    Ok(())
}

/// Validate a configured header name using the HTTP header-name parser.
fn validate_header_name(field: &str, name: Option<&str>) -> Result<(), FilterError> {
    let Some(name) = name else {
        return Ok(());
    };
    if name.is_empty() {
        return Err(format!("openai_responses_model_rewrite: '{field}' header name must not be empty").into());
    }
    if http::HeaderName::from_bytes(name.as_bytes()).is_err() {
        return Err(
            format!("openai_responses_model_rewrite: '{field}' header name is not a valid HTTP header name").into(),
        );
    }
    Ok(())
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
    fn serde_defaults_model_rewrite_config() {
        let cfg: ModelRewriteConfig = serde_yaml::from_str(
            r#"
default_model: "llama-3.3-70b"
"#,
        )
        .unwrap();

        assert_eq!(cfg.default_model.as_deref(), Some("llama-3.3-70b"));
        assert_eq!(cfg.max_body_bytes, DEFAULT_JSON_BODY_MAX_BYTES);
        assert_eq!(cfg.on_invalid, OnInvalidBehavior::Continue);
        assert!(cfg.model_aliases.is_empty());
    }

    #[test]
    fn on_invalid_behavior_defaults_to_continue() {
        let b = OnInvalidBehavior::default();
        assert_eq!(b, OnInvalidBehavior::Continue);
    }

    #[test]
    fn model_rewrite_headers_defaults() {
        let h = ModelRewriteHeaders::default();
        assert_eq!(h.effective_model.as_deref(), Some("x-praxis-ai-effective-model"));
        assert_eq!(h.original_model.as_deref(), Some("x-praxis-ai-original-model"));
    }

    // -- OnInvalidBehavior serde ----------------------------------------------

    #[test]
    fn on_invalid_behavior_serde_continue() {
        let b: OnInvalidBehavior = serde_yaml::from_str("continue").unwrap();
        assert_eq!(b, OnInvalidBehavior::Continue);
    }

    #[test]
    fn on_invalid_behavior_serde_reject() {
        let b: OnInvalidBehavior = serde_yaml::from_str("reject").unwrap();
        assert_eq!(b, OnInvalidBehavior::Reject);
    }

    // -- deny_unknown_fields --------------------------------------------------

    #[test]
    fn deny_unknown_fields_model_rewrite_config() {
        let res = serde_yaml::from_str::<ModelRewriteConfig>(
            r#"
default_model: "test"
bogus: true
"#,
        );
        assert!(res.is_err());
    }

    #[test]
    fn deny_unknown_fields_model_rewrite_headers() {
        let res = serde_yaml::from_str::<ModelRewriteHeaders>(
            r#"
effective_model: x-test
extra: true
"#,
        );
        assert!(res.is_err());
    }

    // -- validate_config ------------------------------------------------------

    #[test]
    fn validate_no_default_model_no_aliases_rejected() {
        let cfg = ModelRewriteConfig {
            default_model: None,
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: HashMap::new(),
            on_invalid: OnInvalidBehavior::Continue,
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("at least one"),
            "expected 'at least one' error, got: {err}"
        );
    }

    #[test]
    fn validate_empty_default_model_rejected() {
        let cfg = ModelRewriteConfig {
            default_model: Some(String::new()),
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: HashMap::new(),
            on_invalid: OnInvalidBehavior::Continue,
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "expected 'must not be empty' error, got: {err}"
        );
    }

    #[test]
    fn validate_whitespace_only_default_model_rejected() {
        let cfg = ModelRewriteConfig {
            default_model: Some("   ".into()),
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: HashMap::new(),
            on_invalid: OnInvalidBehavior::Continue,
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "expected 'must not be empty' error, got: {err}"
        );
    }

    #[test]
    fn validate_default_model_only_ok() {
        let cfg = ModelRewriteConfig {
            default_model: Some("llama-3.3-70b".into()),
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: HashMap::new(),
            on_invalid: OnInvalidBehavior::Continue,
        };
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn validate_aliases_only_ok() {
        let mut aliases = HashMap::new();
        aliases.insert("gpt-4".into(), "llama-3.3-70b".into());
        let cfg = ModelRewriteConfig {
            default_model: None,
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: aliases,
            on_invalid: OnInvalidBehavior::Continue,
        };
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn validate_both_default_model_and_aliases_ok() {
        let mut aliases = HashMap::new();
        aliases.insert("gpt-4".into(), "llama-3.3-70b".into());
        let cfg = ModelRewriteConfig {
            default_model: Some("default-model".into()),
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: DEFAULT_JSON_BODY_MAX_BYTES,
            model_aliases: aliases,
            on_invalid: OnInvalidBehavior::Continue,
        };
        assert!(validate_config(&cfg).is_ok());
    }

    // -- validate_aliases -----------------------------------------------------

    #[test]
    fn validate_aliases_empty_source_rejected() {
        let mut aliases = HashMap::new();
        aliases.insert(String::new(), "target".into());
        let err = validate_aliases(&aliases).unwrap_err();
        assert!(
            err.to_string().contains("source name must not be empty"),
            "expected empty source error, got: {err}"
        );
    }

    #[test]
    fn validate_aliases_multiple_wildcards_rejected() {
        let mut aliases = HashMap::new();
        aliases.insert("gpt-*-*".into(), "target".into());
        let err = validate_aliases(&aliases).unwrap_err();
        assert!(
            err.to_string().contains("at most one '*'"),
            "expected wildcard error, got: {err}"
        );
    }

    #[test]
    fn validate_aliases_empty_target_rejected() {
        let mut aliases = HashMap::new();
        aliases.insert("gpt-4".into(), String::new());
        let err = validate_aliases(&aliases).unwrap_err();
        assert!(
            err.to_string().contains("target"),
            "expected empty target error, got: {err}"
        );
    }

    #[test]
    fn validate_aliases_single_wildcard_ok() {
        let mut aliases = HashMap::new();
        aliases.insert("gpt-4.1-*".into(), "qwen-2.5-72b".into());
        assert!(validate_aliases(&aliases).is_ok());
    }

    #[test]
    fn validate_aliases_exact_alias_ok() {
        let mut aliases = HashMap::new();
        aliases.insert("codex-mini-latest".into(), "llama-3.3-70b".into());
        assert!(validate_aliases(&aliases).is_ok());
    }

    // -- validate_header_name -------------------------------------------------

    #[test]
    fn validate_header_name_none_ok() {
        assert!(validate_header_name("test", None).is_ok());
    }

    #[test]
    fn validate_header_name_empty_rejected() {
        let err = validate_header_name("test", Some("")).unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "expected empty header error, got: {err}"
        );
    }

    #[test]
    fn validate_header_name_invalid_rejected() {
        let err = validate_header_name("test", Some("not a valid header!")).unwrap_err();
        assert!(
            err.to_string().contains("not a valid HTTP header name"),
            "expected invalid header error, got: {err}"
        );
    }

    #[test]
    fn validate_header_name_valid_accepted() {
        assert!(validate_header_name("test", Some("x-custom-header")).is_ok());
    }

    // -- validate_config: max_body_bytes --------------------------------------

    #[test]
    fn validate_zero_max_body_bytes_rejected() {
        let cfg = ModelRewriteConfig {
            default_model: Some("test-model".into()),
            headers: ModelRewriteHeaders::default(),
            max_body_bytes: 0,
            model_aliases: HashMap::new(),
            on_invalid: OnInvalidBehavior::Continue,
        };
        let err = validate_config(&cfg).unwrap_err();
        assert!(
            err.to_string().contains("must be greater than 0"),
            "expected 'must be greater than 0' error, got: {err}"
        );
    }

    // -- null header disables promotion ---------------------------------------

    #[test]
    fn null_header_disables_promotion() {
        let cfg: ModelRewriteConfig = serde_yaml::from_str(
            r#"
default_model: "test"
headers:
  effective_model: null
  original_model: null
"#,
        )
        .unwrap();

        assert!(cfg.headers.effective_model.is_none());
        assert!(cfg.headers.original_model.is_none());
        assert!(validate_config(&cfg).is_ok());
    }
}
