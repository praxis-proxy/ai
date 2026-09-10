// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration for the OpenAI Chat Completions to Vertex AI Gemini
//! translation filter.

use praxis_filter::{FilterError, builtins::http::payload_processing::config_validation::validate_max_body_bytes};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default maximum request body size (1 MiB).
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576;

/// Default GCP region for Vertex AI endpoints.
const DEFAULT_REGION: &str = "us-central1";

// -----------------------------------------------------------------------------
// VertexGeminiConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`OpenaiChatCompletionsToVertexaiGeminiFilter`].
///
/// # YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_vertexai_gemini
/// project: my-gcp-project
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_chat_completions_to_vertexai_gemini
/// project: my-gcp-project
/// region: us-central1
/// max_body_bytes: 1048576
/// ```
///
/// [`OpenaiChatCompletionsToVertexaiGeminiFilter`]: super::OpenaiChatCompletionsToVertexaiGeminiFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VertexGeminiConfig {
    /// GCP project ID used to construct the Vertex AI endpoint path.
    pub project: String,

    /// GCP region for the Vertex AI endpoint path.
    #[serde(default = "default_region")]
    pub region: String,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
}

/// Serde default for [`VertexGeminiConfig::region`].
fn default_region() -> String {
    DEFAULT_REGION.to_owned()
}

/// Serde default for [`VertexGeminiConfig::max_body_bytes`].
fn default_max_body_bytes() -> usize {
    DEFAULT_MAX_BODY_BYTES
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// YAML-facing filter name, shared with the filter module.
pub(super) const FILTER_NAME: &str = "openai_chat_completions_to_vertexai_gemini";

/// Validate the parsed configuration.
pub(crate) fn build_config(cfg: VertexGeminiConfig) -> Result<VertexGeminiConfig, FilterError> {
    validate_max_body_bytes(FILTER_NAME, cfg.max_body_bytes)?;
    if cfg.project.is_empty() {
        return Err(FilterError::from(format!("{FILTER_NAME}: project must not be empty")));
    }
    if cfg.region.is_empty() {
        return Err(FilterError::from(format!("{FILTER_NAME}: region must not be empty")));
    }
    validate_path_segment("project", &cfg.project)?;
    validate_path_segment("region", &cfg.region)?;
    Ok(cfg)
}

/// Reject config values that would produce malformed or traversable URL paths.
///
/// Both `project` and `region` are interpolated into the Vertex AI URL
/// path. Characters like `/`, `?`, `#`, and sequences like `..` could
/// alter the request target if present in a misconfigured value.
fn validate_path_segment(field: &str, value: &str) -> Result<(), FilterError> {
    if value.contains('/')
        || value.contains('?')
        || value.contains('#')
        || value.contains("..")
        || value.bytes().any(|b| b.is_ascii_control())
    {
        return Err(FilterError::from(format!(
            "{FILTER_NAME}: {field} must not contain '/', '?', '#', '..', or control characters"
        )));
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn default_config_with_project() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(r#"project: "my-project""#).unwrap();
        let cfg = build_config(yaml).unwrap();
        assert_eq!(cfg.project, "my-project");
        assert_eq!(cfg.region, "us-central1");
        assert_eq!(cfg.max_body_bytes, 1_048_576);
    }

    #[test]
    fn custom_region_and_body_bytes() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(
            r#"
project: "my-project"
region: "europe-west4"
max_body_bytes: 2097152
"#,
        )
        .unwrap();
        let cfg = build_config(yaml).unwrap();
        assert_eq!(cfg.region, "europe-west4");
        assert_eq!(cfg.max_body_bytes, 2_097_152);
    }

    #[test]
    fn missing_project_fails() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(r#"project: """#).unwrap();
        let err = build_config(yaml).unwrap_err();
        assert!(
            err.to_string().contains("project must not be empty"),
            "expected project error, got: {err}"
        );
    }

    #[test]
    fn empty_region_fails() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(
            r#"
project: "my-project"
region: ""
"#,
        )
        .unwrap();
        let err = build_config(yaml).unwrap_err();
        assert!(
            err.to_string().contains("region must not be empty"),
            "expected region error, got: {err}"
        );
    }

    #[test]
    fn omitted_project_fails_deserialization() {
        let result = serde_yaml::from_str::<VertexGeminiConfig>(r#"region: "us-central1""#);
        assert!(result.is_err(), "project is required and must fail when absent");
    }

    #[test]
    fn project_with_path_traversal_rejected() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(r#"project: "my-proj/../../other""#).unwrap();
        let err = build_config(yaml).unwrap_err();
        assert!(
            err.to_string().contains("must not contain"),
            "path traversal should be rejected, got: {err}"
        );
    }

    #[test]
    fn region_with_query_char_rejected() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(
            r#"
project: "my-project"
region: "us-central1?foo=bar"
"#,
        )
        .unwrap();
        let err = build_config(yaml).unwrap_err();
        assert!(
            err.to_string().contains("must not contain"),
            "query char should be rejected, got: {err}"
        );
    }

    #[test]
    fn unknown_field_rejected() {
        let result = serde_yaml::from_str::<VertexGeminiConfig>(
            r#"
project: "my-project"
unknown_field: true
"#,
        );
        assert!(result.is_err(), "unknown fields should be rejected");
    }
}
