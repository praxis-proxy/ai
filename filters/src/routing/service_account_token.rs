// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`ServiceAccountTokenFilter`] — reads a login token from a file and
//! attaches it to every outgoing request.
//!
//! # Overview
//!
//! Kubernetes can hand a pod a short-lived login token in a file (a
//! "`serviceAccountToken`") and automatically swap that file out for a
//! fresh one before the old token expires. This filter opens the file and
//! reads it again every time it sends a request — there is no cache and
//! no background refresh job, so whatever is in the file right now is
//! what gets used. A rotated token is picked up on the very next request
//! with no config reload and no restart. The file is small (well under
//! [`MAX_TOKEN_BYTES`]) and backed by memory rather than disk, so reading
//! it on every request is cheap.
//!
//! Add this filter to an `outbound_chain` used by a callout filter (for
//! example, `ai_guardrails`'s `NeMo` provider) so that the outbound
//! request carries proof of identity to a destination that checks
//! Kubernetes tokens, such as a `kube-rbac-proxy` sidecar that asks
//! Kubernetes "is this caller allowed to do this?" before letting the
//! request through.
//!
//! # Security
//!
//! - The token is read straight into a [`Zeroizing`] buffer and is never written to `filter_metadata`, trace logs, or
//!   error messages.
//! - If the token file is missing, empty, or too large, the request is rejected with `503` instead of being sent
//!   without a token.
//! - The header this filter writes to can't be a header Praxis already manages internally, or one that controls the
//!   HTTP connection itself — the same rule [`crate::routing::credential_inject`] uses for its own header.
//!
//! # YAML config
//!
//! ```yaml
//! filter: service_account_token
//! token_file: /var/run/secrets/tokens/nemo/token
//! header: Authorization   # optional, defaults to Authorization
//! prefix: "Bearer "       # optional, defaults to "Bearer "
//! ```

use std::{fs::File, io::Read as _, path::Path};

use async_trait::async_trait;
use http::{HeaderName, HeaderValue, header::AUTHORIZATION};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;
use zeroize::Zeroizing;

use super::descriptor::RESERVED_HEADER_PREFIXES;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum byte length accepted for the configured file path.
const MAX_PATH_LEN: usize = 4096;

/// Maximum byte length accepted for the configured header prefix.
const MAX_PREFIX_LEN: usize = 64;

/// Maximum raw token size accepted from the file.
const MAX_TOKEN_BYTES: usize = 16 * 1024;

/// One byte beyond [`MAX_TOKEN_BYTES`] so an oversized file is detected
/// without reading it into memory in full.
const MAX_TOKEN_READ_BYTES: u64 = MAX_TOKEN_BYTES as u64 + 1;

/// Default header value prefix. Matches the `Authorization: Bearer <jwt>`
/// convention used by Kubernetes bearer-token authenticators.
const DEFAULT_PREFIX: &str = "Bearer ";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `service_account_token` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceAccountTokenConfig {
    /// Path to the projected token file. Re-read on every request; must
    /// exist, be readable, and be non-empty at construction time.
    token_file: String,

    /// Header receiving the assembled value. Defaults to `Authorization`.
    /// Can't be a header Praxis already manages internally (`x-praxis-`,
    /// `x-mcp-`), or one that controls the HTTP connection itself, like
    /// `host`, `content-length`, or `proxy-authorization`.
    #[serde(default)]
    header: Option<String>,

    /// Prefix placed before the token in the header value. Defaults to
    /// `"Bearer "`. Set to `""` to inject the raw token value.
    #[serde(default)]
    prefix: Option<String>,
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Reads a bearer token from a file fresh on every request and injects it
/// into the configured header. See the module documentation for the
/// complete rotation and security model.
pub struct ServiceAccountTokenFilter {
    /// Path to the projected token file, re-read on every request.
    token_file: std::path::PathBuf,
    /// Header receiving the assembled value.
    header_name: HeaderName,
    /// Prefix placed before the token in the header value.
    prefix: String,
}

impl ServiceAccountTokenFilter {
    /// Parse from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if:
    /// - `token_file` is empty, oversized, missing, unreadable, or empty
    /// - `header` is set to a header Praxis manages internally, or one that controls the HTTP connection itself
    /// - `prefix` is oversized
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ServiceAccountTokenConfig = parse_filter_config("service_account_token", value)?;

        validate_bounded("token_file", &cfg.token_file, MAX_PATH_LEN)?;
        let header_name = resolve_header_name(cfg.header.as_deref())?;
        let prefix = cfg.prefix.unwrap_or_else(|| DEFAULT_PREFIX.to_owned());
        validate_prefix(&prefix)?;

        let token_file = std::path::PathBuf::from(&cfg.token_file);
        // Fail fast at construction: a pipeline that can never produce a
        // token should not reach the request path.
        read_token(&token_file)?;

        Ok(Box::new(Self {
            token_file,
            header_name,
            prefix,
        }))
    }
}

#[async_trait]
impl HttpFilter for ServiceAccountTokenFilter {
    fn name(&self) -> &'static str {
        "service_account_token"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let token = match read_token(&self.token_file) {
            Ok(token) => token,
            Err(error) => {
                tracing::warn!(
                    path = %self.token_file.display(),
                    %error,
                    "service_account_token: token unavailable; failing closed"
                );
                return Ok(FilterAction::Reject(Rejection::status(503)));
            },
        };

        let assembled = Zeroizing::new(format!("{}{}", self.prefix, token.as_str()));
        let header_value = HeaderValue::from_str(assembled.as_str()).map_err(|e| -> FilterError {
            format!("service_account_token: assembled header value is not valid HTTP: {e}").into()
        })?;

        // Strip any value already present on this header before setting
        // ours, matching the fail-closed replacement behavior of
        // `credential_inject`.
        ctx.request_headers_to_remove.push(self.header_name.clone());
        ctx.request_headers_to_set
            .push((self.header_name.clone(), header_value));

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Determine the injection header, defaulting to `Authorization`.
///
/// Headers Praxis already manages internally are rejected because Praxis
/// would strip them out before the request ever leaves the proxy, so the
/// token would never actually reach anywhere. Headers that control the
/// HTTP connection itself are rejected too, since writing to them can
/// break how the request gets routed or framed — and `proxy-authorization`
/// specifically would hand the token to a middleman proxy instead of the
/// real destination.
fn resolve_header_name(configured: Option<&str>) -> Result<HeaderName, FilterError> {
    let Some(header) = configured else {
        return Ok(AUTHORIZATION);
    };
    validate_bounded("header", header, MAX_PATH_LEN)?;
    let name: HeaderName = header.parse().map_err(|error| -> FilterError {
        format!("service_account_token: invalid header '{header}': {error}").into()
    })?;
    if RESERVED_HEADER_PREFIXES
        .iter()
        .any(|prefix| name.as_str().starts_with(prefix))
    {
        return Err(
            format!("service_account_token: header '{header}' must not use a reserved internal header prefix").into(),
        );
    }
    if name != AUTHORIZATION && praxis_ai_apis::promotion::is_transport_controlled_header_lowercase(name.as_str()) {
        return Err(format!(
            "service_account_token: header '{header}' is transport-controlled and must not receive the service account token"
        )
        .into());
    }
    Ok(name)
}

/// Validate the configured header-value prefix.
fn validate_prefix(prefix: &str) -> Result<(), FilterError> {
    if prefix.len() > MAX_PREFIX_LEN {
        return Err(format!("service_account_token: prefix must be at most {MAX_PREFIX_LEN} bytes").into());
    }
    Ok(())
}

/// Validate a non-empty value against a byte limit.
fn validate_bounded(field: &str, value: &str, maximum: usize) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > maximum {
        return Err(format!("service_account_token: {field} must contain 1-{maximum} bytes").into());
    }
    Ok(())
}

/// Read and validate the complete token file.
///
/// Reads one byte beyond [`MAX_TOKEN_BYTES`] to distinguish an exact-limit
/// file from an oversized one without buffering an unbounded file in full.
/// The token value is never included in the returned error.
fn read_token(path: &Path) -> Result<Zeroizing<String>, FilterError> {
    let file = File::open(path).map_err(|error| -> FilterError {
        format!("service_account_token: cannot read '{}': {error}", path.display()).into()
    })?;
    let mut content = Zeroizing::new(String::new());
    file.take(MAX_TOKEN_READ_BYTES)
        .read_to_string(&mut content)
        .map_err(|error| -> FilterError {
            format!("service_account_token: cannot read '{}': {error}", path.display()).into()
        })?;
    if content.len() > MAX_TOKEN_BYTES {
        return Err(format!("service_account_token: token file exceeds {MAX_TOKEN_BYTES} bytes").into());
    }
    let token = Zeroizing::new(content.trim().to_owned());
    if token.is_empty() {
        return Err(format!("service_account_token: token file '{}' is empty", path.display()).into());
    }
    Ok(token)
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
    use http::Method;

    use super::*;

    // -------------------------------------------------------------------------
    // Config Validation
    // -------------------------------------------------------------------------

    #[test]
    fn missing_file_rejected_at_construction() {
        let err = parse("token_file: /nonexistent/path/to/token")
            .err()
            .expect("must fail");
        assert!(err.to_string().contains("cannot read"), "{err}");
    }

    #[test]
    fn empty_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "   \n").unwrap();
        let yaml = format!("token_file: {}", path.display());
        let err = parse(&yaml).err().expect("empty file must be rejected");
        assert!(err.to_string().contains("is empty"), "{err}");
    }

    #[test]
    fn oversized_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, vec![b't'; MAX_TOKEN_BYTES + 1]).unwrap();
        let yaml = format!("token_file: {}", path.display());
        let err = parse(&yaml).err().expect("oversized file must be rejected");
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn blank_token_file_path_rejected() {
        let err = parse("token_file: ''").err().expect("must fail");
        assert!(err.to_string().contains("token_file"), "{err}");
    }

    #[test]
    fn reserved_prefix_header_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let yaml = format!("token_file: {}\nheader: x-praxis-internal", path.display());
        let err = parse(&yaml).err().expect("reserved-prefix header must be rejected");
        assert!(err.to_string().contains("reserved internal header prefix"), "{err}");
    }

    #[test]
    fn transport_controlled_header_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        for header in ["host", "content-length", "proxy-authorization"] {
            let yaml = format!("token_file: {}\nheader: {header}", path.display());
            let err = parse(&yaml)
                .err()
                .expect("transport-controlled header must be rejected");
            assert!(err.to_string().contains("transport-controlled"), "{header}: {err}");
        }
    }

    #[test]
    fn oversized_prefix_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let long_prefix = "a".repeat(MAX_PREFIX_LEN + 1);
        let yaml = format!("token_file: {}\nprefix: '{long_prefix}'", path.display());
        let err = parse(&yaml).err().expect("oversized prefix must be rejected");
        assert!(err.to_string().contains("prefix"), "{err}");
    }

    #[test]
    fn valid_minimal_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "tok");
        let yaml = format!("token_file: {}", path.display());
        assert!(parse(&yaml).is_ok(), "valid config must parse");
    }

    // -------------------------------------------------------------------------
    // Request-Time Injection
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn injects_default_authorization_bearer_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "sa-jwt-value");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.request_headers_to_set.len(), 1);
        let (name, value) = &ctx.request_headers_to_set[0];
        assert_eq!(*name, AUTHORIZATION);
        assert_eq!(value.to_str().unwrap(), "Bearer sa-jwt-value");
    }

    #[tokio::test]
    async fn strips_caller_supplied_authorization_before_setting() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "sa-jwt-value");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let mut req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        req.headers
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer smuggled"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert!(ctx.request_headers_to_remove.contains(&AUTHORIZATION));
        assert_eq!(ctx.request_headers_to_set[0].1.to_str().unwrap(), "Bearer sa-jwt-value");
    }

    #[tokio::test]
    async fn custom_header_and_empty_prefix_injects_raw_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "raw-token");
        let yaml = format!("token_file: {}\nheader: x-service-token\nprefix: ''", path.display());
        let f = parse(&yaml).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.request_headers_to_set[0].0.as_str(), "x-service-token");
        assert_eq!(ctx.request_headers_to_set[0].1.to_str().unwrap(), "raw-token");
    }

    #[tokio::test]
    async fn rotation_is_observed_on_the_very_next_request() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "token-a");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");

        let mut first = crate::test_utils::make_filter_context(&req);
        drop(f.on_request(&mut first).await.unwrap());
        assert_eq!(first.request_headers_to_set[0].1, "Bearer token-a");

        // Kubernetes rotates a projected token via an atomic rename, the
        // same pattern used here.
        let replacement = dir.path().join("token.next");
        std::fs::write(&replacement, "token-b\n").unwrap();
        std::fs::rename(replacement, &path).unwrap();

        let mut second = crate::test_utils::make_filter_context(&req);
        drop(f.on_request(&mut second).await.unwrap());
        assert_eq!(
            second.request_headers_to_set[0].1, "Bearer token-b",
            "the very next request must observe the rotated token without any watcher or reload"
        );
    }

    #[tokio::test]
    async fn token_deleted_after_construction_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "token-a");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        std::fs::remove_file(&path).unwrap();

        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Reject(r) if r.status == 503));
        assert!(ctx.request_headers_to_set.is_empty());
    }

    #[tokio::test]
    async fn token_not_in_filter_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_token(&dir, "super-secret-sa-token");
        let f = parse(&format!("token_file: {}", path.display())).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/v1/checks");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        for value in ctx.filter_metadata.values() {
            assert!(
                !value.contains("super-secret-sa-token"),
                "token must not leak into filter_metadata: {value}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn write_token(dir: &tempfile::TempDir, token: &str) -> std::path::PathBuf {
        let path = dir.path().join("token");
        std::fs::write(&path, format!("{token}\n")).unwrap();
        path
    }

    fn parse(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        ServiceAccountTokenFilter::from_config(&val)
    }
}
