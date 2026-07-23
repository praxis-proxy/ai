// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the `openai_file_resolve` filter.

use praxis_filter::{FilterError, body::MAX_JSON_BODY_BYTES};
use serde::Deserialize;

use super::resolve_url::NormalizedOrigin;
use crate::openai::api_client;

/// Default HTTP timeout for Files API callout requests (30 000 ms).
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// Default maximum number of file references resolved per request.
const DEFAULT_MAX_FILE_REFERENCES: usize = 32;

/// Maximum configurable file reference count per request.
const MAX_CONFIGURABLE_FILE_REFERENCES: usize = 128;

/// Maximum allowed timeout (300 000 ms / 5 minutes).
const MAX_TIMEOUT_MS: u64 = 300_000;

/// Behavior when a referenced file cannot be fetched.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OnMissing {
    /// Leave the `file_id` reference unchanged and continue.
    #[default]
    Continue,

    /// Return an error response to the client.
    Reject,
}

/// Mode for handling `file_url` content parts.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FileUrlMode {
    /// Fetch the URL and inline content as `file_data` (data URI).
    #[default]
    Resolve,

    /// Leave `file_url` unchanged for native-compatible backends.
    Passthrough,
}

/// YAML configuration for the [`FileResolveFilter`].
///
/// [`FileResolveFilter`]: super::FileResolveFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FileResolveConfig {
    /// Allow `files_api_url` to target private, loopback, link-local,
    /// or DNS-name hosts.  Default `false` rejects SSRF-sensitive
    /// targets; set to `true` in development or when the Files API is
    /// an internal service on a private network.
    #[serde(default)]
    pub allow_private_files_api_url: bool,

    /// Allow Files API callouts from the `StreamBuffer` pre-read
    /// phase, before header-phase security filters execute.
    ///
    /// This must be explicitly enabled only when an outer trust
    /// boundary authenticates and authorizes requests before they
    /// reach this listener. Forwarded headers are the original
    /// downstream values, not mutations from request filters.
    #[serde(default)]
    pub allow_pre_security_callout: bool,

    /// Base URL of the Files API endpoint.
    ///
    /// Example: `http://files-api:8321`
    pub files_api_url: String,

    /// Headers to forward from the original request to the
    /// Files API for authentication and tenant isolation. No
    /// downstream headers are forwarded by default.
    #[serde(default)]
    pub forward_headers: Vec<String>,

    /// Maximum body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,

    /// Maximum number of distinct content-part / `file_id` pairs to
    /// resolve in one request, including rehydrated history.
    #[serde(default = "default_max_file_references")]
    pub max_file_references: usize,

    /// Behavior when a referenced file cannot be fetched.
    #[serde(default)]
    pub on_missing: OnMissing,

    /// HTTP timeout in milliseconds for Files API callout requests.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Mode for `file_url` content parts in `input_file`.
    #[serde(default)]
    pub file_url: FileUrlMode,

    /// Exact origins allowed to resolve to private addresses.
    /// Cloud metadata, unspecified, and multicast remain blocked.
    #[serde(default)]
    pub allowed_file_url_origins: Vec<String>,
}

/// Default max body bytes.
fn default_max_body_bytes() -> usize {
    MAX_JSON_BODY_BYTES
}

/// Default maximum file reference count.
fn default_max_file_references() -> usize {
    DEFAULT_MAX_FILE_REFERENCES
}

/// Default timeout in milliseconds.
fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Validate the parsed configuration.
pub(crate) fn validate_config(mut cfg: FileResolveConfig) -> Result<FileResolveConfig, FilterError> {
    if cfg.files_api_url.is_empty() {
        return Err("openai_file_resolve: 'files_api_url' must not be empty".into());
    }

    if cfg.files_api_url.ends_with('/') {
        return Err("openai_file_resolve: 'files_api_url' must not end with '/'".into());
    }

    api_client::validate_base_url(
        "openai_file_resolve",
        &cfg.files_api_url,
        cfg.allow_private_files_api_url,
    )?;
    api_client::validate_forward_headers("openai_file_resolve", &mut cfg.forward_headers)?;
    validate_limits(&cfg)?;
    validate_pre_security_callout(&cfg)?;
    validate_file_url_config(&cfg)?;

    Ok(cfg)
}

/// Require explicit acknowledgement of the pre-read security boundary.
fn validate_pre_security_callout(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if !cfg.allow_pre_security_callout {
        return Err(
            "openai_file_resolve: 'allow_pre_security_callout' must be true because StreamBuffer body callouts run before header-phase security filters; place authentication and authorization in an outer trust boundary"
                .into(),
        );
    }
    Ok(())
}

/// Validate numeric limits applied while buffering and resolving.
fn validate_limits(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if cfg.max_body_bytes == 0 {
        return Err("openai_file_resolve: 'max_body_bytes' must be greater than 0".into());
    }

    if cfg.max_body_bytes > MAX_JSON_BODY_BYTES {
        return Err(format!(
            "openai_file_resolve: 'max_body_bytes' ({}) exceeds maximum ({MAX_JSON_BODY_BYTES})",
            cfg.max_body_bytes
        )
        .into());
    }

    validate_resolution_limits(cfg)
}

/// Validate callout count and time limits.
fn validate_resolution_limits(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if cfg.max_file_references == 0 {
        return Err("openai_file_resolve: 'max_file_references' must be greater than 0".into());
    }

    if cfg.max_file_references > MAX_CONFIGURABLE_FILE_REFERENCES {
        return Err(format!(
            "openai_file_resolve: 'max_file_references' ({}) exceeds maximum ({MAX_CONFIGURABLE_FILE_REFERENCES})",
            cfg.max_file_references
        )
        .into());
    }

    if cfg.timeout_ms == 0 {
        return Err("openai_file_resolve: 'timeout_ms' must be greater than 0".into());
    }

    if cfg.timeout_ms > MAX_TIMEOUT_MS {
        return Err(format!(
            "openai_file_resolve: 'timeout_ms' ({}) exceeds maximum ({MAX_TIMEOUT_MS})",
            cfg.timeout_ms
        )
        .into());
    }

    Ok(())
}

/// Validate `file_url` mode and `allowed_file_url_origins`.
fn validate_file_url_config(cfg: &FileResolveConfig) -> Result<(), FilterError> {
    if cfg.file_url == FileUrlMode::Passthrough && !cfg.allowed_file_url_origins.is_empty() {
        return Err(
            "openai_file_resolve: 'allowed_file_url_origins' cannot be set when 'file_url' is 'passthrough'".into(),
        );
    }

    let mut seen = Vec::new();
    for raw in &cfg.allowed_file_url_origins {
        let origin = NormalizedOrigin::parse(raw).map_err(|e| -> FilterError {
            format!("openai_file_resolve: invalid 'allowed_file_url_origins' entry '{raw}': {e}").into()
        })?;
        if seen.contains(&origin) {
            return Err(format!("openai_file_resolve: duplicate 'allowed_file_url_origins' entry '{raw}'").into());
        }
        seen.push(origin);
    }

    Ok(())
}

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
    use super::*;

    const MINIMAL_YAML: &str = r#"
files_api_url: "http://files-api:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
"#;

    #[test]
    fn minimal_config_parses() {
        let cfg: FileResolveConfig = serde_yaml::from_str(MINIMAL_YAML).unwrap();
        let validated = validate_config(cfg).unwrap();
        assert_eq!(
            validated.files_api_url, "http://files-api:8321",
            "files_api_url should match"
        );
        assert_eq!(
            validated.max_body_bytes, MAX_JSON_BODY_BYTES,
            "max_body_bytes should default to 64 MiB"
        );
        assert_eq!(
            validated.timeout_ms, DEFAULT_TIMEOUT_MS,
            "timeout_ms should default to 30000"
        );
        assert_eq!(
            validated.max_file_references, DEFAULT_MAX_FILE_REFERENCES,
            "max_file_references should default to 32"
        );
        assert_eq!(
            validated.on_missing,
            OnMissing::Continue,
            "on_missing should default to continue"
        );
        assert_eq!(
            validated.forward_headers,
            Vec::<String>::new(),
            "forward_headers should default to empty"
        );
    }

    #[test]
    fn full_config_parses() {
        let yaml = r#"
files_api_url: "http://files:9090"
allow_private_files_api_url: true
allow_pre_security_callout: true
forward_headers:
  - authorization
  - x-custom-tenant
max_body_bytes: 1048576
max_file_references: 16
on_missing: reject
timeout_ms: 10000
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_config(cfg).unwrap();
        assert_eq!(
            validated.files_api_url, "http://files:9090",
            "files_api_url should match"
        );
        assert_eq!(
            validated.forward_headers,
            vec!["authorization", "x-custom-tenant"],
            "forward_headers should match"
        );
        assert_eq!(validated.max_body_bytes, 1_048_576, "max_body_bytes should match");
        assert_eq!(validated.max_file_references, 16, "max_file_references should match");
        assert_eq!(validated.on_missing, OnMissing::Reject, "on_missing should match");
        assert_eq!(validated.timeout_ms, 10_000, "timeout_ms should match");
    }

    #[test]
    fn deny_unknown_fields_rejects_typo() {
        let yaml = r#"files_api_url: "http://files-api:8321"
on_mising: reject"#;
        let result: Result<FileResolveConfig, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "typo in config field should be rejected");
    }

    #[test]
    fn empty_files_api_url_rejected() {
        let yaml = "files_api_url: ''";
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "empty files_api_url should be rejected");
    }

    #[test]
    fn trailing_slash_files_api_url_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321/""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "trailing slash should be rejected");
    }

    #[test]
    fn zero_max_body_bytes_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321"
max_body_bytes: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "zero max_body_bytes should be rejected");
    }

    #[test]
    fn zero_timeout_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321"
timeout_ms: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "zero timeout should be rejected");
    }

    #[test]
    fn zero_max_file_references_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321"
max_file_references: 0"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "zero max_file_references should be rejected"
        );
    }

    #[test]
    fn max_file_references_above_ceiling_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321"
max_file_references: 129"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "max_file_references above the ceiling should be rejected"
        );
    }

    #[test]
    fn timeout_above_ceiling_rejected() {
        let yaml = r#"files_api_url: "http://files-api:8321"
timeout_ms: 300001"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let result = validate_config(cfg);
        assert!(result.is_err(), "timeout above ceiling should be rejected");
    }

    #[test]
    fn valid_config_passes() {
        let cfg = FileResolveConfig {
            allow_private_files_api_url: true,
            allow_pre_security_callout: true,
            files_api_url: "http://files-api:8321".to_owned(),
            forward_headers: Vec::new(),
            max_body_bytes: MAX_JSON_BODY_BYTES,
            max_file_references: DEFAULT_MAX_FILE_REFERENCES,
            on_missing: OnMissing::Continue,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            file_url: FileUrlMode::Resolve,
            allowed_file_url_origins: Vec::new(),
        };
        assert!(validate_config(cfg).is_ok(), "valid config should pass validation");
    }

    #[test]
    fn forward_headers_are_normalized() {
        let yaml = r#"
files_api_url: "http://files-api:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
forward_headers:
  - Authorization
  - X-Tenant-ID
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_config(cfg).unwrap();

        assert_eq!(
            validated.forward_headers,
            vec!["authorization", "x-tenant-id"],
            "forwarded header names should be normalized once during validation"
        );
    }

    #[test]
    fn invalid_forward_header_rejected() {
        let yaml = r#"
files_api_url: "http://files-api:8321"
allow_private_files_api_url: true
forward_headers: ["bad header"]
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "syntactically invalid forwarded header names should be rejected"
        );
    }

    #[test]
    fn unsafe_forward_headers_rejected() {
        for name in [
            "host",
            "content-length",
            "transfer-encoding",
            "proxy-authorization",
            "x-praxis-route",
        ] {
            let yaml = format!(
                "files_api_url: \"http://files-api:8321\"\nallow_private_files_api_url: true\nforward_headers: [\"{name}\"]"
            );
            let cfg: FileResolveConfig = serde_yaml::from_str(&yaml).unwrap();

            assert!(
                validate_config(cfg).is_err(),
                "unsafe forwarded header '{name}' should be rejected"
            );
        }
    }

    #[test]
    fn duplicate_forward_headers_rejected_case_insensitively() {
        let yaml = r#"
files_api_url: "http://files-api:8321"
allow_private_files_api_url: true
forward_headers: ["Authorization", "authorization"]
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "duplicate forwarded header names should be rejected after normalization"
        );
    }

    #[test]
    fn pre_security_callout_requires_explicit_opt_in() {
        let yaml = r#"
files_api_url: "http://files-api:8321"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();

        assert!(
            validate_config(cfg).is_err(),
            "pre-security external callouts should be disabled by default"
        );
    }

    // SSRF validation is tested thoroughly in api_client::url::tests.
    // These tests verify the delegation path through validate_config.

    #[test]
    fn ssrf_rejects_loopback_ipv4() {
        let yaml = r#"files_api_url: "http://127.0.0.1:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "loopback IPv4 should be rejected without allow_private"
        );
    }

    #[test]
    fn ssrf_allows_public_ipv4() {
        let yaml = r#"files_api_url: "http://8.8.8.8:8321"
allow_pre_security_callout: true"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_ok(), "public IPv4 should be allowed");
    }

    #[test]
    fn ssrf_allows_private_with_override() {
        let yaml = r#"
files_api_url: "http://127.0.0.1:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_ok(),
            "loopback should be allowed with allow_private_files_api_url"
        );
    }

    #[test]
    fn ssrf_rejects_non_http_scheme() {
        let yaml = r#"files_api_url: "ftp://files-api:8321""#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_err(), "non-http scheme should be rejected");
    }

    #[test]
    fn files_api_url_rejects_embedded_credentials() {
        let yaml = r#"
files_api_url: "http://user:password@files-api:8321"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "embedded URL credentials should be rejected"
        );
    }

    #[test]
    fn files_api_url_rejects_query_string() {
        let yaml = r#"
files_api_url: "http://files-api:8321/base?tenant=abc"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "query strings would make appended Files API paths ambiguous"
        );
    }

    #[test]
    fn files_api_url_rejects_fragment() {
        let yaml = r#"
files_api_url: "http://files-api:8321/base#v2"
allow_private_files_api_url: true
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "fragments would hide appended Files API paths from the HTTP request"
        );
    }

    #[test]
    fn file_url_defaults_to_resolve() {
        let cfg: FileResolveConfig = serde_yaml::from_str(MINIMAL_YAML).unwrap();
        assert_eq!(cfg.file_url, FileUrlMode::Resolve, "file_url should default to resolve");
    }

    #[test]
    fn file_url_passthrough_accepted() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
file_url: passthrough
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        let validated = validate_config(cfg).unwrap();
        assert_eq!(validated.file_url, FileUrlMode::Passthrough);
    }

    #[test]
    fn allowed_origins_with_passthrough_rejected() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
file_url: passthrough
allowed_file_url_origins:
  - "https://files.internal:8443"
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "allowed_file_url_origins should be rejected with passthrough mode"
        );
    }

    #[test]
    fn valid_allowed_origins_accepted() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
file_url: resolve
allowed_file_url_origins:
  - "https://files.internal:8443"
  - "http://10.0.0.1:9000"
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_ok(), "valid origins should be accepted");
    }

    #[test]
    fn duplicate_origins_rejected() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
allowed_file_url_origins:
  - "https://files.example.com"
  - "https://files.example.com"
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_err(), "duplicate origins should be rejected");
    }

    #[test]
    fn origin_with_path_rejected() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
allowed_file_url_origins:
  - "https://files.example.com/api"
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(validate_config(cfg).is_err(), "origins with paths should be rejected");
    }

    #[test]
    fn origin_cloud_metadata_rejected() {
        let yaml = r#"
files_api_url: "http://ogx:8321"
allow_private_files_api_url: true
allow_pre_security_callout: true
allowed_file_url_origins:
  - "http://169.254.169.254"
"#;
        let cfg: FileResolveConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            validate_config(cfg).is_err(),
            "cloud metadata origins should be rejected"
        );
    }
}
