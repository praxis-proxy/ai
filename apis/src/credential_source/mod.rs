// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared credential source (`value | env_var | file`) with bounded reads,
//! restart-free file rotation, and fail-closed publication.
//!
//! Used by the web-search filters (`apis`) and the external metering filter
//! (`filters`).

use std::sync::Arc;

use arc_swap::ArcSwap;
use praxis_filter::FilterError;
use serde::{
    Deserialize,
    de::{self, Deserializer},
};
use zeroize::Zeroizing;

mod file;
mod watch;

pub use watch::{WatchHandle, WatchRegistry};

/// Maximum raw token size (bytes).
pub const MAX_TOKEN_BYTES: usize = 16 * 1024;
/// Maximum length of an env-var name or file path (bytes).
pub const MAX_SOURCE_LEN: usize = 4096;

/// One of exactly three mutually-exclusive credential sources.
///
/// Deserialized from a YAML mapping carrying exactly one of `value`,
/// `env_var`, or `file`; zero keys, two keys, or an unknown key are all
/// deserialization errors. `Deserialize` is implemented manually via a
/// private helper struct because the workspace YAML backend (`yaml_serde`)
/// encodes derived externally-tagged enums as YAML tags rather than
/// single-key mappings.
pub enum CredentialSource {
    /// Inline literal token.
    Value(Zeroizing<String>),
    /// Name of a process environment variable holding the token.
    EnvVar(String),
    /// Path to a projected file holding the token (watched for rotation).
    File(String),
}

impl<'de> Deserialize<'de> for CredentialSource {
    #[expect(
        clippy::too_many_lines,
        reason = "explicit secret-safe shape validation keeps rejected scalar values out of serde diagnostics"
    )]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize through Value so rejecting the removed scalar shorthand
        // never asks serde to render the scalar (which may itself be a secret)
        // in an invalid-type diagnostic.
        let raw = serde_yaml::Value::deserialize(deserializer)?;
        let serde_yaml::Value::Mapping(raw) = raw else {
            return Err(de::Error::custom(
                "credential source must be a mapping with exactly one of `value`, `env_var`, or `file`",
            ));
        };
        let mut value = None;
        let mut env_var = None;
        let mut file = None;
        for (key, raw_value) in raw {
            let Some(key) = key.as_str() else {
                return Err(de::Error::custom("credential source keys must be strings"));
            };
            let Some(raw_value) = raw_value.as_str() else {
                return Err(de::Error::custom(format!("credential source `{key}` must be a string")));
            };
            match key {
                "value" => value = Some(Zeroizing::new(raw_value.to_owned())),
                "env_var" => env_var = Some(raw_value.to_owned()),
                "file" => file = Some(raw_value.to_owned()),
                _ => {
                    return Err(de::Error::unknown_field(key, &["value", "env_var", "file"]));
                },
            }
        }
        match (value, env_var, file) {
            (Some(value), None, None) => Ok(Self::Value(value)),
            (None, Some(env_var), None) => Ok(Self::EnvVar(env_var)),
            (None, None, Some(file)) => Ok(Self::File(file)),
            (None, None, None) => Err(de::Error::custom(
                "credential source requires exactly one of `value`, `env_var`, or `file`",
            )),
            _ => Err(de::Error::custom(
                "credential source must set exactly one of `value`, `env_var`, or `file`",
            )),
        }
    }
}

impl std::fmt::Debug for CredentialSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The inline value is a secret — never render it.
            Self::Value(_) => f.debug_tuple("Value").field(&"[REDACTED]").finish(),
            // An env-var name and a path are non-secret locators.
            Self::EnvVar(name) => f.debug_tuple("EnvVar").field(name).finish(),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

/// A single published credential state. `payload == None` means a
/// **post-startup** reload produced no valid artifact (unreadable / oversized /
/// invalid / removed) — consumers fail closed. At construction the payload is
/// always `Some` (an invalid initial token rejects construction), so `None` is
/// only ever reached via the watcher.
pub struct SecretSnapshot {
    /// The shaped artifact (raw token or pre-assembled header value), or `None`.
    pub payload: Option<Zeroizing<String>>,
}

impl std::fmt::Debug for SecretSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretSnapshot")
            .field("payload", &self.payload.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// A cheap-to-clone handle to the current [`SecretSnapshot`]. Cloning shares the
/// same underlying cell; the watcher swaps the snapshot atomically.
#[derive(Clone)]
pub struct SecretHandle {
    /// Shared, atomically-swappable cell holding the current published snapshot.
    inner: Arc<ArcSwap<SecretSnapshot>>,
}

impl SecretHandle {
    /// Construct a handle whose cell starts at `snapshot`.
    pub fn new(snapshot: SecretSnapshot) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(snapshot)),
        }
    }

    /// Clone the shared cell so the watcher can atomically swap the snapshot.
    pub(crate) fn cell(&self) -> Arc<ArcSwap<SecretSnapshot>> {
        Arc::clone(&self.inner)
    }

    /// Load the current snapshot. Returns a guard the caller borrows `&str` from
    /// via `snapshot.payload.as_deref()` — no per-request deep copy of the secret.
    #[must_use]
    pub fn current(&self) -> arc_swap::Guard<Arc<SecretSnapshot>> {
        self.inner.load()
    }

    /// Build a fixed, non-watched handle for tests.
    #[cfg(test)]
    pub(crate) fn from_snapshot_for_test(payload: Option<Zeroizing<String>>) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(SecretSnapshot { payload })),
        }
    }
}

impl std::fmt::Debug for SecretHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the loaded secret; show only that a handle exists.
        f.debug_struct("SecretHandle").field("payload", &"[REDACTED]").finish()
    }
}

/// Owned, non-secret diagnostic context used to qualify read-path and reload
/// errors that occur BEFORE any token exists (missing file, unreadable,
/// oversized, env var not set) — errors the token-only `transform` cannot
/// annotate. Holds no secret bytes.
#[derive(Clone, Debug)]
pub struct DiagnosticContext {
    /// Owning filter name, e.g. `"credential_inject"`.
    pub owner: &'static str,
    /// Preformatted non-secret label, e.g. `"name/namespace/key"`; `None` for a
    /// single-credential filter (`web_search`).
    pub reference: Option<String>,
}

/// The shaping closure, shared between the initial read and the watcher thread.
pub(crate) type TransformFn = Arc<dyn Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync>;

/// Resolve a source to its raw token (pre-transform), enforcing bounds and
/// reproducing `credential_inject`'s diagnostics via `ctx`.
pub(crate) fn resolve_raw_token(
    source: &CredentialSource,
    ctx: &DiagnosticContext,
) -> Result<Zeroizing<String>, FilterError> {
    match source {
        CredentialSource::Value(value) => {
            let token = value.trim();
            file::validate_token(token, ctx.owner)?;
            Ok(Zeroizing::new(token.to_owned()))
        },
        CredentialSource::EnvVar(name) => {
            file::validate_source_len("env_var", name, ctx.owner)?;
            let owner = ctx.owner;
            let for_ref = match &ctx.reference {
                Some(reference) => format!(" for '{reference}'"),
                None => String::new(),
            };
            let token = Zeroizing::new(std::env::var(name).map_err(|error| -> FilterError {
                let detail = match error {
                    std::env::VarError::NotPresent => "not set",
                    std::env::VarError::NotUnicode(_) => "not valid UTF-8",
                };
                format!("{owner}: env var '{name}' {detail}{for_ref}").into()
            })?);
            file::validate_token(&token, ctx.owner)?;
            Ok(token)
        },
        CredentialSource::File(path) => {
            file::validate_source_len("file", path, ctx.owner)?;
            file::read_projected_token(path, ctx)
        },
    }
}

/// Bridge so `watch.rs` can reuse the file/env bounds check without exposing the
/// `file` submodule publicly.
pub(crate) fn file_validate_source_len_for_registry(path: &str, owner: &str) -> Result<(), FilterError> {
    file::validate_source_len("file", path, owner)
}

/// Build a handle for a single source. `value`/`env_var` publish a fixed
/// snapshot; `file` registers a watcher (see [`WatchRegistry`]) and returns a
/// `WatchHandle`. An invalid initial token — for any source kind — rejects
/// construction (fail loud at startup).
///
/// The `transform` closure must return `Err` for every failure mode and must not
/// panic: it runs on the watcher thread, where a panic terminates rotation and
/// freezes the last-published snapshot (fail-stale) instead of failing closed.
///
/// # Errors
///
/// Returns [`FilterError`] when the source is invalid, unreadable, or produces an
/// empty/oversized token, or when the `transform` rejects the initial token.
pub fn build_one(
    ctx: DiagnosticContext,
    source: &CredentialSource,
    transform: impl Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync + 'static,
) -> Result<(SecretHandle, Option<WatchHandle>), FilterError> {
    if let CredentialSource::File(_) = source {
        let mut registry = WatchRegistry::new();
        let handle = registry.register(ctx, source, transform)?;
        let watch = registry.finish()?;
        Ok((handle, watch))
    } else {
        let raw = resolve_raw_token(source, &ctx)?;
        let payload = transform(&raw)?;
        Ok((SecretHandle::new(SecretSnapshot { payload: Some(payload) }), None))
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Result<CredentialSource, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    #[test]
    fn parses_value_env_var_and_file() {
        assert!(matches!(parse("value: inline").unwrap(), CredentialSource::Value(_)));
        assert!(matches!(parse("env_var: MY_KEY").unwrap(), CredentialSource::EnvVar(v) if v == "MY_KEY"));
        assert!(
            matches!(parse("file: /run/secrets/key").unwrap(), CredentialSource::File(p) if p == "/run/secrets/key")
        );
    }

    #[test]
    fn rejects_multiple_sources() {
        // Externally-tagged enum: a map with two keys is not a single variant.
        assert!(parse("value: a\nenv_var: B").is_err());
    }

    #[test]
    fn rejects_no_source() {
        assert!(parse("{}").is_err());
    }

    #[test]
    fn rejects_unknown_source() {
        assert!(parse("secret_manager: vault").is_err());
    }

    #[test]
    fn rejects_scalar_source_without_echoing_secret() {
        let secret = "scalar-secret-must-not-leak";
        let error = serde_yaml::from_str::<CredentialSource>(secret).unwrap_err();
        assert!(!error.to_string().contains(secret), "{error}");
        assert!(error.to_string().contains("credential source must be a mapping"));
    }

    #[test]
    fn debug_redacts_inline_value() {
        let src = CredentialSource::Value(Zeroizing::new("super-secret".to_owned()));
        let rendered = format!("{src:?}");
        assert!(rendered.contains("[REDACTED]"));
        assert!(!rendered.contains("super-secret"));
    }

    #[test]
    fn debug_shows_non_secret_locators() {
        assert!(format!("{:?}", CredentialSource::EnvVar("MY_KEY".to_owned())).contains("MY_KEY"));
        assert!(format!("{:?}", CredentialSource::File("/run/secrets/key".to_owned())).contains("/run/secrets/key"));
    }

    #[test]
    fn secret_snapshot_debug_redacts_payload() {
        let snap = SecretSnapshot {
            payload: Some(Zeroizing::new("tok-abc".to_owned())),
        };
        let rendered = format!("{snap:?}");
        assert!(rendered.contains("[REDACTED]"), "{rendered}");
        assert!(!rendered.contains("tok-abc"), "{rendered}");
        // None must render distinctly and still leak nothing.
        assert!(format!("{:?}", SecretSnapshot { payload: None }).contains("None"));
    }

    #[test]
    fn secret_handle_current_borrows_payload_and_debug_redacts() {
        let handle = SecretHandle::from_snapshot_for_test(Some(Zeroizing::new("tok-xyz".to_owned())));
        let snap = handle.current();
        assert_eq!(snap.payload.as_deref().map(String::as_str), Some("tok-xyz"));
        let rendered = format!("{handle:?}");
        assert!(rendered.contains("[REDACTED]"));
        assert!(!rendered.contains("tok-xyz"));
    }

    #[test]
    fn diagnostic_context_debug_is_non_secret() {
        let ctx = DiagnosticContext {
            owner: "credential_inject",
            reference: Some("name/namespace/key".to_owned()),
        };
        let rendered = format!("{ctx:?}");
        assert!(rendered.contains("credential_inject"));
        assert!(rendered.contains("name/namespace/key"));
    }

    fn identity_transform() -> impl Fn(&str) -> Result<Zeroizing<String>, FilterError> + Send + Sync + 'static {
        |token: &str| Ok(Zeroizing::new(token.to_owned()))
    }

    #[test]
    fn build_one_value_publishes_transformed_payload() {
        let ctx = DiagnosticContext {
            owner: "test",
            reference: None,
        };
        let src = CredentialSource::Value(Zeroizing::new("inline-tok".to_owned()));
        let (handle, watch) = build_one(ctx, &src, identity_transform()).unwrap();
        assert!(watch.is_none(), "value source must not spawn a watcher");
        assert_eq!(
            handle.current().payload.as_deref().map(String::as_str),
            Some("inline-tok")
        );
    }

    #[test]
    fn build_one_value_trims_surrounding_whitespace() {
        let ctx = DiagnosticContext {
            owner: "test",
            reference: None,
        };
        let src = CredentialSource::Value(Zeroizing::new("  inline-tok \n".to_owned()));
        let (handle, watch) = build_one(ctx, &src, identity_transform()).unwrap();
        assert!(watch.is_none());
        assert_eq!(
            handle.current().payload.as_deref().map(String::as_str),
            Some("inline-tok")
        );
    }

    #[test]
    fn build_one_value_rejects_empty_token_at_construction() {
        let ctx = DiagnosticContext {
            owner: "test",
            reference: None,
        };
        let src = CredentialSource::Value(Zeroizing::new("   ".to_owned()));
        assert!(build_one(ctx, &src, identity_transform()).is_err());
    }

    #[test]
    fn build_one_value_rejects_failing_transform_at_construction() {
        let ctx = DiagnosticContext {
            owner: "web_search",
            reference: None,
        };
        let src = CredentialSource::Value(Zeroizing::new("bad".to_owned()));
        let transform = |_t: &str| -> Result<Zeroizing<String>, FilterError> {
            Err("web_search: invalid API key header value".to_owned().into())
        };
        let err = build_one(ctx, &src, transform).unwrap_err();
        assert!(err.to_string().contains("invalid API key header value"));
    }

    #[test]
    fn build_one_env_var_reads_and_transforms() {
        // Use a var that already exists in every test environment.
        let ctx = DiagnosticContext {
            owner: "test",
            reference: None,
        };
        let src = CredentialSource::EnvVar("CARGO_PKG_NAME".to_owned());
        let (handle, watch) = build_one(ctx, &src, identity_transform()).unwrap();
        assert!(watch.is_none());
        assert_eq!(
            handle.current().payload.as_deref().map(String::as_str),
            Some("praxis-ai-apis")
        );
    }

    #[test]
    fn build_one_env_var_not_set_error_names_reference() {
        let ctx = DiagnosticContext {
            owner: "credential_inject",
            reference: Some("name/namespace/key".to_owned()),
        };
        let src = CredentialSource::EnvVar("DEFINITELY_NOT_SET_CREDSOURCE_12345".to_owned());
        let err = build_one(ctx, &src, identity_transform()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("credential_inject"), "{msg}");
        assert!(msg.contains("name/namespace/key"), "{msg}");
        assert!(msg.contains("not set"), "{msg}");
        assert!(
            !msg.to_lowercase().contains("unicode"),
            "must not fall through to VarError Display: {msg}"
        );
    }
}
