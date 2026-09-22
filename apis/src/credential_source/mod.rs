//! Shared credential source (`value | env_var | file`) with bounded reads,
//! restart-free file rotation, and fail-closed publication.
//!
//! Used by the web-search filters (`apis`) and, imported upward, by
//! `credential_inject` (`filters`). See
//! `docs/superpowers/specs/2026-09-21-shared-credential-source-design.md`.

use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::de::{self, Deserializer};
use serde::Deserialize;
use zeroize::Zeroizing;

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
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Transient parse target: exactly one field is expected to be set.
        // Deliberately does NOT derive `Debug` — it holds the inline secret.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            #[serde(default)]
            value: Option<Zeroizing<String>>,
            #[serde(default)]
            env_var: Option<String>,
            #[serde(default)]
            file: Option<String>,
        }

        let raw = Raw::deserialize(deserializer)?;
        match (raw.value, raw.env_var, raw.file) {
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
    /// Load the current snapshot. Returns a guard the caller borrows `&str` from
    /// via `snapshot.payload.as_deref()` — no per-request deep copy of the secret.
    #[must_use]
    pub fn current(&self) -> arc_swap::Guard<Arc<SecretSnapshot>> {
        self.inner.load()
    }

    /// Build a fixed, non-watched handle for tests.
    #[cfg(test)]
    pub(crate) fn from_snapshot_for_test(payload: Option<Zeroizing<String>>) -> Self {
        Self { inner: Arc::new(ArcSwap::from_pointee(SecretSnapshot { payload })) }
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
        assert!(matches!(parse("file: /run/secrets/key").unwrap(), CredentialSource::File(p) if p == "/run/secrets/key"));
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
        let snap = SecretSnapshot { payload: Some(Zeroizing::new("tok-abc".to_owned())) };
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
}
