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
pub(crate) fn build_config(mut cfg: VertexGeminiConfig) -> Result<VertexGeminiConfig, FilterError> {
    validate_max_body_bytes(FILTER_NAME, cfg.max_body_bytes)?;
    cfg.project = validate_path_segment("project", &cfg.project)?;
    cfg.region = validate_path_segment("region", &cfg.region)?;
    Ok(cfg)
}

/// Trim and validate a config value interpolated into the Vertex AI URL path.
///
/// Returns the trimmed value on success so callers can store the
/// canonical form. Rejects values that are empty or whitespace-only
/// (both produce empty path segments), and values containing characters
/// outside the URL-safe set `[A-Za-z0-9\-._]`. Characters like space,
/// `@`, `[`, `]`, `%`, etc. require percent-encoding and would produce
/// malformed URLs; `/`, `?`, `#`, and `..` could alter routing.
fn validate_path_segment(field: &str, value: &str) -> Result<String, FilterError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(FilterError::from(format!(
            "{FILTER_NAME}: {field} must not be empty or whitespace-only"
        )));
    }
    if trimmed
        .bytes()
        .any(|b| !b.is_ascii_alphanumeric() && b != b'-' && b != b'.' && b != b'_')
    {
        return Err(FilterError::from(format!(
            "{FILTER_NAME}: {field} must contain only alphanumeric, '-', '.', or '_' characters"
        )));
    }
    Ok(trimmed.to_owned())
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
            err.to_string().contains("project"),
            "expected project error, got: {err}"
        );
    }

    #[test]
    fn whitespace_only_project_fails() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(r#"project: "   ""#).unwrap();
        let err = build_config(yaml).unwrap_err();
        assert!(
            err.to_string().contains("project"),
            "whitespace-only project should be rejected, got: {err}"
        );
    }

    #[test]
    fn whitespace_padded_project_is_trimmed() {
        let yaml = serde_yaml::from_str::<VertexGeminiConfig>(r#"project: "  my-project  ""#).unwrap();
        let cfg = build_config(yaml).unwrap();
        assert_eq!(
            cfg.project, "my-project",
            "leading/trailing whitespace must be stripped"
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
        assert!(err.to_string().contains("region"), "expected region error, got: {err}");
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
            err.to_string().contains("alphanumeric"),
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
            err.to_string().contains("alphanumeric"),
            "query char should be rejected, got: {err}"
        );
    }

    #[test]
    fn project_with_url_unsafe_chars_rejected() {
        for bad in ["my@project", "proj[1]", "my%project", "my project"] {
            let yaml = serde_yaml::from_str::<VertexGeminiConfig>(&format!(r#"project: "{bad}""#)).unwrap();
            let err = build_config(yaml).unwrap_err();
            assert!(
                err.to_string().contains("alphanumeric"),
                "'{bad}' should be rejected, got: {err}"
            );
        }
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
