//! Shared credential source (`value | env_var | file`) with bounded reads,
//! restart-free file rotation, and fail-closed publication.
//!
//! Used by the web-search filters (`apis`) and, imported upward, by
//! `credential_inject` (`filters`). See
//! `docs/superpowers/specs/2026-09-21-shared-credential-source-design.md`.

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

#[cfg(test)]
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
}
