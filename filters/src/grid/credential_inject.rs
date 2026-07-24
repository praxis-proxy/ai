// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! [`GridCredentialInjectFilter`] — injects upstream credentials selected by
//! `grid_route` into the outgoing request.
//!
//! # Overview
//!
//! `grid_route` writes the selected candidate's credential reference into
//! filter metadata when the candidate was configured with a `credential`
//! field:
//!
//! | Metadata key | Example value |
//! |---|---|
//! | `grid.route.credential.strategy` | `"bearer_token"` |
//! | `grid.route.credential.name` | `"my-api-secret"` |
//! | `grid.route.credential.namespace` | `"grid-system"` |
//! | `grid.route.credential.key` | `"token"` |
//!
//! This filter reads those keys, looks up the matching token in its configured
//! credential map, and injects `Authorization: Bearer <token>` into the
//! upstream request.  The filter must appear **after** `grid_route` in the
//! filter chain.
//!
//! # Behaviour
//!
//! | State | Action |
//! |---|---|
//! | No `grid.route.credential.name` metadata | No-op — candidate has no credential |
//! | Metadata present, matching entry found | Inject `Authorization: Bearer <token>` |
//! | Metadata present, no matching entry | Reject 503 (fail closed) |
//! | Strategy not `"bearer_token"` | Reject 503 (fail closed) |
//!
//! # Security
//!
//! - Token values are **never** written to `filter_metadata`, tracing spans, or error bodies.
//! - `grid_route` writes only the credential *reference* to metadata — the secretRef `name`, `namespace`, and `key`
//!   fields — not the token.
//! - Token values are stored in [`Zeroizing`] wrappers so they are wiped from memory when the filter is dropped.
//!
//! # Token sources
//!
//! This filter is the native injection seam for Grid credential handling.
//! Tokens can be supplied as inline config values, environment variables, or
//! file paths.  The file source is the production-oriented path for Kubernetes:
//! mount a Secret into the pod and point `file` at the mounted token file.
//! This keeps token bytes out of Praxis `ConfigMap`s without adding Kubernetes
//! API calls to the proxy runtime.
//!
//! # YAML config
//!
//! ```yaml
//! filter: grid_credential_inject
//! credentials:
//!   - name: my-api-secret        # matches grid.route.credential.name
//!     namespace: grid-system      # matches grid.route.credential.namespace
//!     key: token                  # matches grid.route.credential.key
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
//! Secret entry and must match what the Grid operator wrote into the routing
//! overlay candidate.

use std::{borrow::Cow, collections::HashMap};

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;
use zeroize::Zeroizing;

// -----------------------------------------------------------------------------
// Metadata keys (written by grid_route, read by this filter)
// -----------------------------------------------------------------------------

/// Filter metadata key for the selected credential strategy (written by `grid_route`).
const CREDENTIAL_STRATEGY_KEY: &str = "grid.route.credential.strategy";

/// Filter metadata key for the selected credential's Kubernetes Secret name.
const CREDENTIAL_NAME_KEY: &str = "grid.route.credential.name";

/// Filter metadata key for the selected credential's Kubernetes Secret namespace.
const CREDENTIAL_NAMESPACE_KEY: &str = "grid.route.credential.namespace";

/// Filter metadata key for the selected credential's Kubernetes Secret data key.
const CREDENTIAL_KEY_KEY: &str = "grid.route.credential.key";

/// The only currently supported authentication strategy.
const STRATEGY_BEARER_TOKEN: &str = "bearer_token";

/// Lowercase `Authorization` header name for `ctx.extra_request_headers`.
const AUTHORIZATION_HEADER: &str = "authorization";

/// Prefix prepended to the raw token to form the full header value.
const BEARER_PREFIX: &str = "Bearer ";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the `grid_credential_inject` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GridCredentialInjectConfig {
    /// Credential entries, keyed by secretRef (name/namespace/key).
    credentials: Vec<CredentialEntryConfig>,
}

/// A single configured credential entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CredentialEntryConfig {
    /// Kubernetes Secret name — must match `grid.route.credential.name`.
    name: String,

    /// Kubernetes Secret namespace — must match `grid.route.credential.namespace`.
    namespace: String,

    /// Key within `Secret.data` — must match `grid.route.credential.key`.
    key: String,

    /// Credential strategy.  Currently only `"bearer_token"` is supported.
    #[serde(default = "default_strategy")]
    strategy: String,

    /// Inline token value.  Mutually exclusive with `env_var` and `file`.
    #[serde(default)]
    value: Option<String>,

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
/// credential reference as written by the Grid operator into the overlay.
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

/// Injects upstream credentials selected by `grid_route` into the outgoing request.
///
/// Reads `grid.route.credential.*` filter metadata written by `grid_route`, looks up
/// the configured token, and injects `Authorization: Bearer <token>`.  Token values
/// are never written to metadata, traces, or error bodies.  See the module
/// `//!` doc for the full data-flow description and YAML config shape.
pub struct GridCredentialInjectFilter {
    /// Credential reference → resolved injectable credential.
    credentials: HashMap<CredentialRef, ResolvedCredential>,
}

impl GridCredentialInjectFilter {
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
        let cfg: GridCredentialInjectConfig = parse_filter_config("grid_credential_inject", config)?;

        if cfg.credentials.is_empty() {
            return Err("grid_credential_inject: 'credentials' must not be empty".into());
        }

        let mut credentials = HashMap::with_capacity(cfg.credentials.len());

        for entry in &cfg.credentials {
            validate_strategy(&entry.strategy)?;
            let resolved = resolve_credential(entry)?;
            let cred_ref = CredentialRef {
                name: entry.name.clone(),
                namespace: entry.namespace.clone(),
                key: entry.key.clone(),
            };
            if credentials.contains_key(&cred_ref) {
                return Err(format!(
                    "grid_credential_inject: duplicate credential entry for '{}/{}/{}'",
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
impl HttpFilter for GridCredentialInjectFilter {
    fn name(&self) -> &'static str {
        "grid_credential_inject"
    }

    #[expect(
        clippy::too_many_lines,
        reason = "sequential metadata read + strategy check + map lookup + header injection; each branch is a distinct safety boundary"
    )]
    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // If grid_route selected a candidate with no credential, this is a no-op.
        let Some(name) = ctx.get_metadata(CREDENTIAL_NAME_KEY) else {
            tracing::debug!("grid_credential_inject: no selected credential; skipping");
            return Ok(FilterAction::Continue);
        };

        let namespace = ctx.get_metadata(CREDENTIAL_NAMESPACE_KEY).unwrap_or("");
        let key = ctx.get_metadata(CREDENTIAL_KEY_KEY).unwrap_or("");
        let strategy = ctx.get_metadata(CREDENTIAL_STRATEGY_KEY).unwrap_or("");

        if strategy != STRATEGY_BEARER_TOKEN {
            tracing::debug!(
                strategy = strategy,
                "grid_credential_inject: unsupported strategy; failing closed"
            );
            return Ok(FilterAction::Reject(Rejection::status(503)));
        }

        let cred_ref = CredentialRef {
            name: name.to_owned(),
            namespace: namespace.to_owned(),
            key: key.to_owned(),
        };

        let Some(cred) = self.credentials.get(&cred_ref) else {
            // Log the reference identity (not the token) to assist debugging.
            tracing::debug!(
                name = name,
                namespace = namespace,
                key = key,
                "grid_credential_inject: no configured token for selected credential; failing closed"
            );
            return Ok(FilterAction::Reject(Rejection::status(503)));
        };

        tracing::debug!(
            name = name,
            namespace = namespace,
            key = key,
            "grid_credential_inject: injecting bearer credential"
        );

        // Clone the pre-validated header value string for this request.
        // The clone is a plain String (not Zeroizing) because extra_request_headers
        // does not support Zeroize; it lives until the request context is dropped.
        ctx.extra_request_headers.push((
            Cow::Borrowed(AUTHORIZATION_HEADER),
            cred.header_value.as_str().to_owned(),
        ));

        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Validate that the strategy is supported.
fn validate_strategy(strategy: &str) -> Result<(), FilterError> {
    if strategy == STRATEGY_BEARER_TOKEN {
        return Ok(());
    }
    Err(format!(
        "grid_credential_inject: unsupported strategy '{strategy}' \
         (only 'bearer_token' is currently supported)"
    )
    .into())
}

/// Resolve and validate one configured credential.
fn resolve_credential(entry: &CredentialEntryConfig) -> Result<ResolvedCredential, FilterError> {
    let token = resolve_token(entry)?;
    let header_value_str = format!("{BEARER_PREFIX}{}", &*token);
    http::HeaderValue::from_str(&header_value_str).map_err(|e| -> FilterError {
        format!(
            "grid_credential_inject: assembled header value invalid for '{}/{}/{}': {e}",
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
        (Some(val), None, None) => Ok(Zeroizing::new(val.clone())),
        (None, Some(var), None) => std::env::var(var).map(Zeroizing::new).map_err(|e| -> FilterError {
            format!(
                "grid_credential_inject: env var '{var}' not set for '{}/{}/{}': {e}",
                entry.name, entry.namespace, entry.key
            )
            .into()
        }),
        (None, None, Some(path)) => {
            let content = std::fs::read_to_string(path).map_err(|e| -> FilterError {
                format!(
                    "grid_credential_inject: cannot read file '{path}' for '{}/{}/{}': {e}",
                    entry.name, entry.namespace, entry.key
                )
                .into()
            })?;
            let token = content.trim().to_owned();
            if token.is_empty() {
                return Err(format!(
                    "grid_credential_inject: file '{path}' is empty for '{}/{}/{}'",
                    entry.name, entry.namespace, entry.key
                )
                .into());
            }
            Ok(Zeroizing::new(token))
        },
        (None, None, None) => Err(format!(
            "grid_credential_inject: '{}/{}/{}' must have exactly one of 'value', 'env_var', or 'file'",
            entry.name, entry.namespace, entry.key
        )
        .into()),
        _ => Err(format!(
            "grid_credential_inject: '{}/{}/{}' has multiple token sources; use exactly one of 'value', 'env_var', or 'file'",
            entry.name, entry.namespace, entry.key
        )
        .into()),
    }
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

    // ---- Config validation ----

    #[test]
    fn empty_credentials_rejected() {
        let err = parse_err("credentials: []");
        assert!(err.to_string().contains("must not be empty"), "{err}");
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

    // ---- No-op when no selected credential ----

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
            ctx.extra_request_headers.is_empty(),
            "no Authorization injected without credential"
        );
    }

    // ---- Bearer token injection ----

    #[tokio::test]
    async fn bearer_token_with_configured_value_injects_authorization() {
        let f = make_filter_with_value("my-secret", "grid-system", "token", "sk-abc123");
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        set_credential_metadata(&mut ctx, "bearer_token", "my-secret", "grid-system", "token");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "matched credential must continue"
        );
        assert_eq!(ctx.extra_request_headers.len(), 1, "exactly one header injected");
        let (hdr_name, hdr_value) = &ctx.extra_request_headers[0];
        assert_eq!(hdr_name.as_ref(), "authorization", "must inject Authorization header");
        assert_eq!(hdr_value, "Bearer sk-abc123", "must inject correct Bearer value");
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

    // ---- File source ----

    #[tokio::test]
    async fn file_source_reads_token_and_injects_authorization() {
        let path = std::env::temp_dir().join("grid-cred-inject-test-token.txt");
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
        assert_eq!(ctx.extra_request_headers.len(), 1);
        assert_eq!(
            ctx.extra_request_headers[0].1, "Bearer file-sourced-token",
            "file-sourced token must be injected (whitespace trimmed)"
        );
        let _cleanup = std::fs::remove_file(&path);
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
        let path = std::env::temp_dir().join("grid-cred-inject-empty-test.txt");
        std::fs::write(&path, "   \n  ").unwrap();
        let yaml = format!(
            "credentials:\n  - name: s\n    namespace: ns\n    key: k\n    file: {}",
            path.display()
        );
        let err = parse(&yaml).err().expect("empty file must be rejected");
        assert!(
            err.to_string().contains("is empty"),
            "empty file error must be reported: {err}"
        );
        let _cleanup = std::fs::remove_file(&path);
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
        let path = std::env::temp_dir().join("grid-cred-inject-meta-test.txt");
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
        let _cleanup = std::fs::remove_file(&path);
    }

    // ---- Fail closed ----

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
            ctx.extra_request_headers.is_empty(),
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
            ctx.extra_request_headers.is_empty(),
            "no Authorization on fail-closed path"
        );
    }

    // ---- Security: token not in metadata ----

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

    // ---- Multi-credential selection ----

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
        assert_eq!(ctx.extra_request_headers.len(), 1);
        assert_eq!(
            ctx.extra_request_headers[0].1, "Bearer token-b",
            "must inject token for selected credential sec-b, not sec-a"
        );
    }

    // ---- Test utilities ----

    fn parse(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        GridCredentialInjectFilter::from_config(&val)
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
        ctx.set_metadata(CREDENTIAL_STRATEGY_KEY, strategy);
        ctx.set_metadata(CREDENTIAL_NAME_KEY, name);
        ctx.set_metadata(CREDENTIAL_NAMESPACE_KEY, namespace);
        ctx.set_metadata(CREDENTIAL_KEY_KEY, key);
    }
}
