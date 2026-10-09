// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the Vertex AI dialect filter.

use std::collections::BTreeMap;

use praxis_filter::FilterError;
use serde::Deserialize;

/// Default maximum request/response body size (32 MiB) — long Claude
/// contexts are large, and the request body is buffered to read `model`
/// and `stream` before they move into the URL.
pub(crate) const DEFAULT_MAX_BODY_BYTES: usize = 33_554_432;

/// Default request-URL path prefix stripped to obtain the Vertex
/// publisher model id (`vertex/claude-sonnet-4-5` ->
/// `claude-sonnet-4-5`).
pub(crate) const DEFAULT_MODEL_PREFIX: &str = "vertex/";

/// YAML-facing filter name, shared with the filter module.
pub(super) const FILTER_NAME: &str = "anthropic_messages_to_vertexai_anthropic";

/// YAML configuration for the [`AnthropicMessagesToVertexaiAnthropicFilter`].
///
/// [`AnthropicMessagesToVertexaiAnthropicFilter`]: super::AnthropicMessagesToVertexaiAnthropicFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VertexAnthropicConfig {
    /// GCP project the upstream URL is built against.
    pub project: String,

    /// Vertex location; `global` routes through the multi-region
    /// endpoint and is the recommended default.
    #[serde(default = "default_location")]
    pub location: String,

    /// Prefix stripped from the request body's `model` to obtain the
    /// Vertex publisher model id. Defaults to `vertex/`.
    #[serde(default = "default_model_prefix")]
    pub model_prefix: String,

    /// Snapshot suffixes keyed by Vertex publisher model id, e.g.
    /// `claude-sonnet-4-5: "@20250929"`. A pin is appended to the model
    /// id in the URL only when the publisher id matches its key exactly,
    /// so each model keeps its own snapshot and an alias cannot move to a
    /// new snapshot underneath a stable user-facing name. Models without
    /// an entry are sent unpinned.
    #[serde(default)]
    pub model_pins: BTreeMap<String, String>,

    /// Allowed `anthropic-beta` flag values. An empty list (the
    /// default) forwards the header untouched; a non-empty list keeps
    /// only the listed comma-separated flags and drops the header if
    /// nothing remains.
    #[serde(default)]
    pub beta_allowlist: Vec<String>,

    /// Maximum buffered request/response body size in bytes.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

/// Serde default for [`VertexAnthropicConfig::location`].
fn default_location() -> String {
    "global".to_owned()
}

/// Serde default for [`VertexAnthropicConfig::model_prefix`].
fn default_model_prefix() -> String {
    DEFAULT_MODEL_PREFIX.to_owned()
}

/// Serde default for [`VertexAnthropicConfig::max_body_bytes`].
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

/// Whether a character may appear in a URL path component built from
/// user-supplied or operator-supplied model identity. Deliberately an
/// allowlist so a crafted `model` value can never inject path
/// structure (`/`, `..`), query bytes (`?`, `#`), or the `:rawPredict`
/// verb separator (`:`).
pub(crate) fn is_safe_model_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '@')
}

/// Validate identity components (`project`, `location`, model pins) used
/// verbatim in the upstream URL path.
fn validate_path_component(filter: &str, field: &str, value: &str) -> Result<(), FilterError> {
    if value.is_empty() || !value.chars().all(is_safe_model_char) {
        return Err(FilterError::from(format!(
            "{filter}: {field} '{value}' is invalid: only ASCII letters, digits, and '.', '-', '_', '@' are allowed"
        )));
    }
    Ok(())
}

/// Validate the parsed configuration: every field that lands verbatim
/// in the upstream URL is checked at config time, so a misconfiguration
/// fails at pipeline build rather than per request.
pub(crate) fn build_config(cfg: VertexAnthropicConfig) -> Result<VertexAnthropicConfig, FilterError> {
    praxis_filter::builtins::http::payload_processing::config_validation::validate_max_body_bytes(
        FILTER_NAME,
        cfg.max_body_bytes,
    )?;
    validate_path_component(FILTER_NAME, "project", &cfg.project)?;
    validate_path_component(FILTER_NAME, "location", &cfg.location)?;
    for (model, pin) in &cfg.model_pins {
        validate_path_component(FILTER_NAME, "model_pins key", model)?;
        validate_path_component(FILTER_NAME, &format!("model_pins['{model}']"), pin)?;
    }
    for entry in &cfg.beta_allowlist {
        if entry.trim().is_empty() {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: beta_allowlist entries must not be empty"
            )));
        }
        if entry.contains(['\r', '\n', ',']) {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: beta_allowlist entries must be bare flag values (no commas or newlines)"
            )));
        }
    }
    if cfg.model_prefix.is_empty() {
        return Err(FilterError::from(format!(
            "{FILTER_NAME}: model_prefix must not be empty"
        )));
    }
    Ok(cfg)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    fn config(yaml: &str) -> Result<VertexAnthropicConfig, FilterError> {
        let parsed: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let cfg: VertexAnthropicConfig = serde_yaml::from_value(parsed).unwrap();
        build_config(cfg)
    }

    #[test]
    fn defaults_applied() {
        let cfg = config("project: my-project").unwrap();
        assert_eq!(cfg.location, "global");
        assert_eq!(cfg.model_prefix, "vertex/");
        assert!(cfg.model_pins.is_empty());
        assert!(cfg.beta_allowlist.is_empty());
        assert_eq!(cfg.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }

    #[test]
    fn rejects_path_injection_in_url_components() {
        for yaml in [
            "project: my-project/../evil",
            "project: \"my project\"",
            "project: my-project\nlocation: global?x=1",
            "project: my-project\nlocation: europe:west1",
            "project: my-project\nmodel_pins: {claude-sonnet-4-5: \"@20250929/../x\"}",
            "project: my-project\nmodel_pins: {claude-sonnet-4-5: \"@2025?x=1\"}",
            "project: my-project\nmodel_pins: {\"claude/../x\": \"@20250929\"}",
            "project: my-project\nmodel_pins: {claude-sonnet-4-5: \"\"}",
        ] {
            assert!(config(yaml).is_err(), "must reject: {yaml}");
        }
    }

    #[test]
    fn accepts_valid_components() {
        let cfg = config(
            "project: gcp-jboyer-san-gemini\nlocation: us-east5\nmodel_pins: {claude-sonnet-4-5: '@20250929', claude-opus-4-1: '@20250805'}\nbeta_allowlist: [context-1m-2025-08-07, interleaved-thinking-2025-05-14]",
        )
        .unwrap();
        assert_eq!(
            cfg.model_pins.get("claude-sonnet-4-5").map(String::as_str),
            Some("@20250929")
        );
        assert_eq!(
            cfg.model_pins.get("claude-opus-4-1").map(String::as_str),
            Some("@20250805")
        );
        assert_eq!(cfg.beta_allowlist.len(), 2);
    }

    #[test]
    fn rejects_empty_prefix_and_blank_beta_entry() {
        assert!(config("project: p\nmodel_prefix: \"\"").is_err());
        assert!(config("project: p\nbeta_allowlist: [ok, ' ']").is_err());
    }
}
