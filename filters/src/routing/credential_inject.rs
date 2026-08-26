// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`CredentialInjectFilter`] — injects the credential reference selected
//! by an authorized routing filter into the outgoing request.
//!
//! # Overview
//!
//! `intelligent_route` or `provider_route` writes a selected credential
//! reference into filter metadata when the matched candidate or provider-local
//! route requires a credential:
//!
//! | Metadata key | Example value |
//! |---|---|
//! | `intelligent_route.credential.strategy` | `"bearer_token"` |
//! | `intelligent_route.credential.name` | `"my-api-secret"` |
//! | `intelligent_route.credential.namespace` | `"grid-system"` |
//! | `intelligent_route.credential.key` | `"token"` |
//!
//! This filter reads those keys, looks up the matching token in its configured
//! credential map, removes the incoming `Authorization` value, and sets
//! `Authorization: Bearer <token>` on the upstream request. The filter must
//! appear after the routing filter in the filter chain.
//!
//! # Behaviour
//!
//! | State | Action |
//! |---|---|
//! | No `intelligent_route.credential.name` metadata | No-op — candidate has no credential |
//! | Metadata present, matching entry found | Inject `Authorization: Bearer <token>` |
//! | Metadata present, no matching entry | Reject 503 (fail closed) |
//! | Strategy not `"bearer_token"` | Reject 503 (fail closed) |
//!
//! # Security
//!
//! - Token values are **never** written to `filter_metadata`, tracing spans, or error bodies.
//! - The routing filter writes only the credential *reference* to metadata — the secretRef `name`, `namespace`, and
//!   `key` fields — not the token.
//! - Token values are stored in [`Zeroizing`] wrappers so they are wiped from memory when the filter is dropped.
//! - Credential count, locator fields, source fields, and token bytes are bounded before they enter the request path.
//!
//! # Token sources
//!
//! This filter is the native injection seam for routing credential handling.
//! Tokens can be supplied as inline config values, environment variables, or
//! file paths.  The file source is the production-oriented path for Kubernetes:
//! mount a Secret into the pod and point `file` at the mounted token file.
//! This keeps token bytes out of Praxis `ConfigMap`s without adding Kubernetes
//! API calls to the proxy runtime.
//!
//! # YAML config
//!
//! ```yaml
//! filter: credential_inject
//! credentials:
//!   - name: my-api-secret        # matches intelligent_route.credential.name
//!     namespace: grid-system      # matches intelligent_route.credential.namespace
//!     key: token                  # matches intelligent_route.credential.key
//!     strategy: bearer_token      # optional, defaults to bearer_token
//!     file: /run/secrets/grid-credentials/my-api-secret/token
//!   - name: other-secret
//!     namespace: default
//!     key: api-key
//!     strategy: bearer_token
//!     env_var: OTHER_API_TOKEN    # token from environment variable
//! ```
//!
//! The `name`/`namespace`/`key` triple uniquely identifies a Kubernetes
//! Secret entry and must match what the configuration producer wrote into the
//! routing overlay candidate.

use std::{collections::HashMap, fs::File, io::Read as _};

use async_trait::async_trait;
use http::{HeaderValue, header::AUTHORIZATION};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;
use zeroize::Zeroizing;

use super::metadata::{
    CREDENTIAL_KEY, CREDENTIAL_NAME, CREDENTIAL_NAMESPACE, CREDENTIAL_STRATEGY, STRATEGY_BEARER_TOKEN,
};

/// Maximum credential references accepted by one filter instance.
const MAX_CREDENTIALS: usize = 1024;

/// Maximum byte length for a credential reference component.
const MAX_REFERENCE_LEN: usize = 256;

/// Maximum byte length for an environment variable name or file path.
const MAX_SOURCE_LEN: usize = 4096;

/// Maximum raw credential size accepted from any source.
const MAX_TOKEN_BYTES: usize = 16 * 1024;

/// Maximum bytes read to distinguish an exact-limit file from an oversized one.
const MAX_TOKEN_READ_BYTES: u64 = 16 * 1024 + 1;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `credential_inject` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialInjectConfig {
    /// Credential entries, keyed by secretRef (name/namespace/key).
    credentials: Vec<CredentialEntryConfig>,
}

/// A single configured credential entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialEntryConfig {
    /// Kubernetes Secret name — must match `intelligent_route.credential.name`.
    name: String,

    /// Kubernetes Secret namespace — must match `intelligent_route.credential.namespace`.
    namespace: String,

    /// Key within `Secret.data` — must match `intelligent_route.credential.key`.
    key: String,

    /// Credential strategy.  Currently only `"bearer_token"` is supported.
    #[serde(default = "default_strategy")]
    strategy: String,

    /// Inline token value.  Mutually exclusive with `env_var` and `file`.
    #[serde(default)]
    value: Option<Zeroizing<String>>,

    /// Environment variable holding the token.  Mutually exclusive with `value` and `file`.
    #[serde(default)]
    env_var: Option<String>,

    /// Path to a file containing the token, read once at filter construction.
    ///
    /// The file contents are trimmed of leading/trailing whitespace before use.
    /// The file must exist, be readable, and be non-empty; construction fails
    /// otherwise.  Use this source when the token is mounted from a Kubernetes
    /// Secret volume so that token bytes never appear in Praxis `ConfigMap`s.
    ///
    /// Mutually exclusive with `value` and `env_var`.
    #[serde(default)]
    file: Option<String>,
}

/// Default credential strategy when not specified in config.
fn default_strategy() -> String {
    STRATEGY_BEARER_TOKEN.to_owned()
}

// -----------------------------------------------------------------------------
// Internal types
// -----------------------------------------------------------------------------

/// Lookup key: the tuple `(name, namespace, key)` uniquely identifies a
/// credential reference as written by the configuration producer into the overlay.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CredentialRef {
    /// Kubernetes Secret name.
    name: String,
    /// Kubernetes Secret namespace.
    namespace: String,
    /// Key within `Secret.data`.
    key: String,
}

/// Resolved credential ready for request-time injection.
struct ResolvedCredential {
    /// Full `Authorization` header value ("Bearer {token}"), zeroized on drop.
    ///
    /// A non-zeroized copy is created per-request for `HeaderValue`; it lives
    /// until the request context is dropped.  This is the same accepted residual
    /// as in the Praxis `credential_injection` filter.
    header_value: Zeroizing<String>,
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Replaces caller credentials with the upstream credential selected by
/// `intelligent_route` or `provider_route`.
///
/// Reads `intelligent_route.credential.*` filter metadata written by the preceding
/// routing filter, looks up the configured token, removes caller authorization,
/// and sets exactly one `Authorization: Bearer <token>` value. Token values are
/// never written to metadata, traces, or error bodies. See the module
/// documentation for the complete data flow and configuration.
pub struct CredentialInjectFilter {
    /// Credential reference → resolved injectable credential.
    credentials: HashMap<CredentialRef, ResolvedCredential>,
}

impl CredentialInjectFilter {
    /// Create from YAML config.
    ///
    /// Resolves all credentials (inline values, environment variables, or files) at
    /// construction time; per-request processing is a pure map lookup.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if:
    /// - `credentials` is empty
    /// - any entry has more than one token source (`value`, `env_var`, `file`) or none
    /// - any `env_var` is not set in the environment
    /// - any `file` does not exist, is unreadable, or is empty
    /// - any strategy is not `"bearer_token"`
    /// - the assembled header value is not valid HTTP
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: CredentialInjectConfig = parse_filter_config("credential_inject", config)?;

        if cfg.credentials.is_empty() || cfg.credentials.len() > MAX_CREDENTIALS {
            return Err(format!("credential_inject: credentials must contain 1-{MAX_CREDENTIALS} entries").into());
        }

        let mut credentials = HashMap::with_capacity(cfg.credentials.len());

        for entry in &cfg.credentials {
            validate_strategy(&entry.strategy)?;
            validate_credential_ref(entry)?;
            let resolved = resolve_credential(entry)?;
            let cred_ref = CredentialRef {
                name: entry.name.clone(),
                namespace: entry.namespace.clone(),
                key: entry.key.clone(),
            };
            if credentials.contains_key(&cred_ref) {
                return Err(format!(
                    "credential_inject: duplicate credential entry for '{}/{}/{}'",
                    entry.name, entry.namespace, entry.key
                )
                .into());
            }
            credentials.insert(cred_ref, resolved);
        }

        Ok(Box::new(Self { credentials }))
    }
}

#[async_trait]
impl HttpFilter for CredentialInjectFilter {
    fn name(&self) -> &'static str {
        "credential_inject"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(selected) = selected_credential_ref(ctx) else {
            tracing::debug!("credential_inject: no selected credential; skipping");
            return Ok(FilterAction::Continue);
        };
        let Ok(cred_ref) = selected else {
            return Ok(FilterAction::Reject(Rejection::status(503)));
        };

        let Some(cred) = self.credentials.get(&cred_ref) else {
            // Log the reference identity (not the token) to assist debugging.
            tracing::debug!(
                name = %cred_ref.name,
                namespace = %cred_ref.namespace,
                key = %cred_ref.key,
                "credential_inject: no configured token for selected credential; failing closed"
            );
            return Ok(FilterAction::Reject(Rejection::status(503)));
        };

        tracing::debug!(
            name = %cred_ref.name,
            namespace = %cred_ref.namespace,
            key = %cred_ref.key,
            "credential_inject: injecting bearer credential"
        );

        let header_value = HeaderValue::from_str(cred.header_value.as_str()).map_err(|e| -> FilterError {
            format!("credential_inject: invalid resolved Authorization header: {e}").into()
        })?;
        ctx.request_headers_to_remove.push(AUTHORIZATION);
        ctx.request_headers_to_set.push((AUTHORIZATION, header_value));

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Read one complete credential reference from filter metadata.
///
/// Absence of all four fields means that the selected route requires no
/// credential. Any partial or unsupported reference fails closed.
fn selected_credential_ref(ctx: &HttpFilterContext<'_>) -> Option<Result<CredentialRef, ()>> {
    let strategy = ctx.get_metadata(CREDENTIAL_STRATEGY);
    let name = ctx.get_metadata(CREDENTIAL_NAME);
    let namespace = ctx.get_metadata(CREDENTIAL_NAMESPACE);
    let key = ctx.get_metadata(CREDENTIAL_KEY);

    if strategy.is_none() && name.is_none() && namespace.is_none() && key.is_none() {
        return None;
    }
    if strategy != Some(STRATEGY_BEARER_TOKEN) || name.is_none() || namespace.is_none() || key.is_none() {
        tracing::debug!(
            strategy = strategy.unwrap_or("missing"),
            "credential_inject: incomplete or unsupported credential reference; failing closed"
        );
        return Some(Err(()));
    }

    Some(Ok(CredentialRef {
        name: name.unwrap_or_default().to_owned(),
        namespace: namespace.unwrap_or_default().to_owned(),
        key: key.unwrap_or_default().to_owned(),
    }))
}

/// Validate that the strategy is supported.
fn validate_strategy(strategy: &str) -> Result<(), FilterError> {
    if strategy == STRATEGY_BEARER_TOKEN {
        return Ok(());
    }
    Err(format!(
        "credential_inject: unsupported strategy '{strategy}' \
         (only 'bearer_token' is currently supported)"
    )
    .into())
}

/// Validate the bounded Secret locator fields used as the lookup key.
fn validate_credential_ref(entry: &CredentialEntryConfig) -> Result<(), FilterError> {
    validate_bounded("name", &entry.name, MAX_REFERENCE_LEN)?;
    validate_bounded("namespace", &entry.namespace, MAX_REFERENCE_LEN)?;
    validate_bounded("key", &entry.key, MAX_REFERENCE_LEN)
}

/// Resolve and validate one configured credential.
fn resolve_credential(entry: &CredentialEntryConfig) -> Result<ResolvedCredential, FilterError> {
    let token = resolve_token(entry)?;
    validate_token(&token)?;
    let header_value_str = format!("Bearer {}", &*token);
    HeaderValue::from_str(&header_value_str).map_err(|e| -> FilterError {
        format!(
            "credential_inject: assembled header value invalid for '{}/{}/{}': {e}",
            entry.name, entry.namespace, entry.key
        )
        .into()
    })?;
    Ok(ResolvedCredential {
        header_value: Zeroizing::new(header_value_str),
    })
}

/// Resolve the raw token from inline value, environment variable, or file.
#[expect(
    clippy::too_many_lines,
    reason = "three-way source match with distinct error messages per branch"
)]
fn resolve_token(entry: &CredentialEntryConfig) -> Result<Zeroizing<String>, FilterError> {
    match (&entry.value, &entry.env_var, &entry.file) {
        (Some(val), None, None) => {
            validate_token(val)?;
            Ok(val.clone())
        },
        (None, Some(var), None) => {
            validate_bounded("env_var", var, MAX_SOURCE_LEN)?;
            std::env::var(var).map(Zeroizing::new).map_err(|e| -> FilterError {
                format!(
                    "credential_inject: env var '{var}' not set for '{}/{}/{}': {e}",
                    entry.name, entry.namespace, entry.key
                )
                .into()
            })
        },
        (None, None, Some(path)) => {
            validate_bounded("file", path, MAX_SOURCE_LEN)?;
            let file = File::open(path).map_err(|e| -> FilterError {
                format!(
                    "credential_inject: cannot read file '{path}' for '{}/{}/{}': {e}",
                    entry.name, entry.namespace, entry.key
                )
                .into()
            })?;
            let mut content = Zeroizing::new(String::new());
            file.take(MAX_TOKEN_READ_BYTES)
                .read_to_string(&mut content)
                .map_err(|e| -> FilterError {
                    format!(
                        "credential_inject: cannot read file '{path}' for '{}/{}/{}': {e}",
                        entry.name, entry.namespace, entry.key
                    )
                    .into()
                })?;
            if content.len() > MAX_TOKEN_BYTES {
                return Err(format!("credential_inject: file '{path}' exceeds {MAX_TOKEN_BYTES} bytes").into());
            }
            let token = Zeroizing::new(content.trim().to_owned());
            validate_token(&token)?;
            Ok(token)
        },
        (None, None, None) => Err(format!(
            "credential_inject: '{}/{}/{}' must have exactly one of 'value', 'env_var', or 'file'",
            entry.name, entry.namespace, entry.key
        )
        .into()),
        _ => Err(format!(
            "credential_inject: '{}/{}/{}' has multiple token sources; use exactly one of 'value', 'env_var', or 'file'",
            entry.name, entry.namespace, entry.key
        )
        .into()),
    }
}

/// Validate one non-empty value against a byte limit.
fn validate_bounded(field: &str, value: &str, maximum: usize) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > maximum {
        return Err(format!("credential_inject: {field} must contain 1-{maximum} bytes").into());
    }
    Ok(())
}

/// Validate raw credential bytes before constructing an HTTP header.
fn validate_token(value: &str) -> Result<(), FilterError> {
    validate_bounded("credential token", value, MAX_TOKEN_BYTES)
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
    use http::{HeaderValue, Method};

    use super::*;

    // -------------------------------------------------------------------------
    // Config Validation
    // -------------------------------------------------------------------------

    #[test]
    fn empty_credentials_rejected() {
        let err = parse_err("credentials: []");
        assert!(err.to_string().contains("must contain"), "{err}");
    }

    #[test]
    fn too_many_credentials_rejected() {
        let entries = (0..=MAX_CREDENTIALS)
            .map(|index| format!("  - name: s{index}\n    namespace: ns\n    key: k\n    value: tok\n"))
            .collect::<String>();
        let yaml = format!("credentials:\n{entries}");
        assert!(parse(&yaml).is_err(), "credential count must be bounded");
    }

    #[test]
    fn blank_and_oversized_references_rejected() {
        for name in [String::new(), "n".repeat(MAX_REFERENCE_LEN + 1)] {
            let yaml = format!("credentials:\n  - name: '{name}'\n    namespace: ns\n    key: k\n    value: tok\n");
            assert!(parse(&yaml).is_err(), "invalid credential locator must fail");
        }
    }

    #[test]
    fn both_value_and_env_var_rejected() {
        let err =
            parse_err("credentials:\n  - name: s\n    namespace: ns\n    key: k\n    value: tok\n    env_var: MY_VAR");
        assert!(
            err.to_string().contains("multiple token sources"),
            "value+env_var must report multiple-sources error: {err}"
        );
    }

    #[test]
    fn neither_value_nor_env_var_nor_file_rejected() {
        let err = parse_err("credentials:\n  - name: s\n    namespace: ns\n    key: k");
        assert!(
            err.to_string().contains("must have exactly one of"),
            "no source must be rejected: {err}"
        );
    }

    #[test]
    fn unsupported_strategy_in_config_rejected() {
        let err =
            parse_err("credentials:\n  - name: s\n    namespace: ns\n    key: k\n    strategy: oauth2\n    value: tok");
        assert!(err.to_string().contains("unsupported strategy"), "{err}");
    }

    #[test]
    fn valid_minimal_config() {
        let f = parse("credentials:\n  - name: s\n    namespace: ns\n    key: k\n    value: tok");
        assert!(f.is_ok(), "valid config must parse");
    }

    #[test]
    fn duplicate_credential_ref_rejected() {
        let err = parse_err(concat!(
            "credentials:\n",
            "  - name: s\n    namespace: ns\n    key: k\n    value: token-a\n",
            "  - name: s\n    namespace: ns\n    key: k\n    value: token-b\n",
        ));
        assert!(
            err.to_string().contains("duplicate credential entry"),
            "duplicate secretRef entries must be rejected: {err}"
        );
    }

    #[test]
    fn default_strategy_is_bearer_token() {
        assert_eq!(default_strategy(), STRATEGY_BEARER_TOKEN);
    }

    // -------------------------------------------------------------------------
    // No-Op When No Credential Is Selected
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn no_selected_credential_is_noop() {
        let f = make_filter_with_value("sname", "sns", "skey", "tok");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "no credential metadata → continue"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "no Authorization injected without credential"
        );
    }

    // -------------------------------------------------------------------------
    // Bearer Token Injection
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn bearer_token_with_configured_value_injects_authorization() {
        let f = make_filter_with_value("my-secret", "grid-system", "token", "sk-abc123");
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer customer-token"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "my-secret", "grid-system", "token");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "matched credential must continue"
        );
        assert!(
            ctx.request_headers_to_remove.contains(&AUTHORIZATION),
            "customer Authorization must be removed"
        );
        assert_eq!(ctx.request_headers_to_set.len(), 1, "exactly one header injected");
        let (header_name, header_value) = &ctx.request_headers_to_set[0];
        assert_eq!(*header_name, AUTHORIZATION, "must inject Authorization header");
        assert_eq!(
            header_value.to_str().unwrap(),
            "Bearer sk-abc123",
            "must inject correct Bearer value"
        );
    }

    #[test]
    fn missing_env_var_rejected_at_construction() {
        // Exercises the env_var resolution path without unsafe set_var.
        // Uses a name guaranteed not to exist in the test environment.
        let err = parse_err(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    env_var: DEFINITELY_NOT_SET_GRID_CRED_XYZ123",
        );
        assert!(
            err.to_string().contains("not set"),
            "missing env var must be reported: {err}"
        );
    }

    // -------------------------------------------------------------------------
    // File Source
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn file_source_reads_token_and_injects_authorization() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "file-sourced-token\n").unwrap();
        let yaml = format!(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: {}",
            path.display()
        );
        let f = parse(&yaml).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "s", "ns", "k");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "file credential must route");
        assert_eq!(ctx.request_headers_to_set.len(), 1);
        assert_eq!(
            ctx.request_headers_to_set[0].1.to_str().unwrap(),
            "Bearer file-sourced-token",
            "file-sourced token must be injected (whitespace trimmed)"
        );
    }

    #[test]
    fn missing_file_rejected_at_construction() {
        let err = parse_err(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: /nonexistent/path/to/token.txt",
        );
        assert!(
            err.to_string().contains("cannot read file"),
            "missing file must be reported: {err}"
        );
    }

    #[test]
    fn empty_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "   \n  ").unwrap();
        let yaml = format!(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: {}",
            path.display()
        );
        let err = parse(&yaml).err().expect("empty file must be rejected");
        assert!(
            err.to_string().contains("credential token"),
            "empty file error must be reported: {err}"
        );
    }

    #[test]
    fn oversized_file_rejected_at_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, vec![b't'; MAX_TOKEN_BYTES + 1]).unwrap();
        let yaml = format!(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: {}",
            path.display()
        );
        assert!(parse(&yaml).is_err(), "oversized credential file must fail");
    }

    #[test]
    fn value_and_file_rejected() {
        let err =
            parse_err("credentials:\n  - name: s\n    namespace: ns\n    key: k\n    value: tok\n    file: /some/path");
        assert!(
            err.to_string().contains("multiple token sources"),
            "value+file must be rejected: {err}"
        );
    }

    #[test]
    fn env_var_and_file_rejected() {
        let err = parse_err(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    env_var: MY_VAR\n    file: /some/path",
        );
        assert!(
            err.to_string().contains("multiple token sources"),
            "env_var+file must be rejected: {err}"
        );
    }

    #[tokio::test]
    async fn file_token_not_written_to_filter_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "secret-file-token-xyz").unwrap();
        let yaml = format!(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: {}",
            path.display()
        );
        let f = parse(&yaml).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "s", "ns", "k");
        let _unused = f.on_request(&mut ctx).await.unwrap();
        for value in ctx.filter_metadata.values() {
            assert!(
                !value.contains("secret-file-token-xyz"),
                "file token must not appear in filter_metadata; found in: {value}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Fail Closed
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn missing_configured_token_fails_closed_503() {
        let f = make_filter_with_value("other-secret", "ns", "key", "tok");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        // Metadata references a credential not in the filter's map.
        set_credential_metadata(&mut ctx, "bearer_token", "unknown-secret", "ns", "key");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "missing token must fail closed 503"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "no Authorization on fail-closed path"
        );
    }

    #[tokio::test]
    async fn unsupported_strategy_fails_closed_503() {
        let f = make_filter_with_value("sec", "ns", "key", "tok");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "oauth2", "sec", "ns", "key");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 503),
            "unsupported strategy must fail closed 503"
        );
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "no Authorization on fail-closed path"
        );
    }

    // -------------------------------------------------------------------------
    // Security: Token Not In Metadata
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn token_not_in_filter_metadata_after_injection() {
        let f = make_filter_with_value("sec", "ns", "key", "super-secret-token");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "sec", "ns", "key");

        let _unused = f.on_request(&mut ctx).await.unwrap();

        // Token must not appear anywhere in filter_metadata.
        for value in ctx.filter_metadata.values() {
            assert!(
                !value.contains("super-secret-token"),
                "token must not appear in filter_metadata; found in: {value}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Multi-Credential Selection
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn multiple_credentials_selects_matching_entry_only() {
        let yaml = concat!(
            "credentials:\n",
            "  - name: sec-a\n    namespace: ns\n    key: k\n    value: token-a\n",
            "  - name: sec-b\n    namespace: ns\n    key: k\n    value: token-b\n",
        );
        let f = parse(yaml).unwrap();
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "sec-b", "ns", "k");

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.request_headers_to_set.len(), 1);
        assert_eq!(
            ctx.request_headers_to_set[0].1.to_str().unwrap(),
            "Bearer token-b",
            "must inject token for selected credential sec-b, not sec-a"
        );
    }

    #[tokio::test]
    async fn partial_credential_metadata_fails_closed() {
        let f = make_filter_with_value("sec", "ns", "key", "tok");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata(CREDENTIAL_STRATEGY, STRATEGY_BEARER_TOKEN);
        ctx.set_metadata(CREDENTIAL_NAME, "sec");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(rejection) if rejection.status == 503),
            "partial credential metadata must fail closed"
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn parse(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        CredentialInjectFilter::from_config(&val)
    }

    fn parse_err(yaml: &str) -> FilterError {
        parse(yaml).err().expect("config should have been rejected")
    }

    fn make_filter_with_value(name: &str, namespace: &str, key: &str, token: &str) -> Box<dyn HttpFilter> {
        let yaml =
            format!("credentials:\n  - name: {name}\n    namespace: {namespace}\n    key: {key}\n    value: {token}");
        parse(&yaml).unwrap()
    }

    fn set_credential_metadata(
        ctx: &mut HttpFilterContext<'_>,
        strategy: &str,
        name: &str,
        namespace: &str,
        key: &str,
    ) {
        ctx.set_metadata(CREDENTIAL_STRATEGY, strategy);
        ctx.set_metadata(CREDENTIAL_NAME, name);
        ctx.set_metadata(CREDENTIAL_NAMESPACE, namespace);
        ctx.set_metadata(CREDENTIAL_KEY, key);
    }
}
