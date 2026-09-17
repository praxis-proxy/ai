// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the response store filter.

use percent_encoding::percent_decode_str;
use praxis_filter::{FilterError, has_dot_dot_traversal};
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;

#[cfg(feature = "store-postgres")]
use crate::store::postgres_url::{
    self, has_postgres_url_ssl_root_cert, is_verified_postgres_sslmode, postgres_url_sslmode,
    validate_postgres_url_tls_file_params,
};
#[cfg(feature = "store-postgres")]
use crate::store::validate_postgres_table_identifiers;
use crate::store::{PoolConfig, SslMode, validate_table_identifier};

/// Filter name used in SSRF validation error messages.
const FILTER_NAME: &str = "openai_response_store";

// -----------------------------------------------------------------------------
// StorageBackend
// -----------------------------------------------------------------------------

/// Supported storage backends.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StorageBackend {
    /// SQLite backend (file-backed or in-memory). Requires `store-sqlite`.
    Sqlite,

    /// `PostgreSQL` backend. Enabled by default through `store-postgres`.
    Postgres,
}

// -----------------------------------------------------------------------------
// ResponseStoreConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`ResponseStoreFilter`].
///
/// [`ResponseStoreFilter`]: super::ResponseStoreFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResponseStoreConfig {
    /// Storage backend to use.
    pub backend: StorageBackend,

    /// Database connection URL. Wrapped in [`SecretString`] to
    /// prevent accidental logging of credentials.
    pub database_url: SecretString,

    /// Table name for response records.
    pub responses_table: String,

    /// Table name for conversation message records.
    pub conversations_table: String,

    /// TLS mode for `PostgreSQL` connections.
    ///
    /// Only valid when `backend` is `postgres`. Overrides any
    /// `sslmode` parameter in the connection URL.
    #[serde(default)]
    pub ssl_mode: Option<SslMode>,

    /// Path to a PEM-encoded root CA certificate for `PostgreSQL`
    /// TLS verification.
    ///
    /// Only valid when `backend` is `postgres` and the effective
    /// SSL mode is `verify-ca` or `verify-full`.
    #[serde(default)]
    pub ssl_root_cert: Option<SecretString>,

    /// Allow `PostgreSQL` URLs that target local-sensitive addresses.
    ///
    /// By default, DNS names, localhost, loopback, private,
    /// link-local, cloud metadata, unspecified, and Unix socket
    /// targets are rejected. This opt-in is intended for local
    /// development and tests.
    #[serde(default)]
    pub allow_private_database_url: bool,

    /// Connection pool tuning options.
    ///
    /// When omitted, sqlx defaults apply (`max_connections = 10`,
    /// `idle_timeout = 600s`, `acquire_timeout = 30s`).
    #[serde(default)]
    pub pool: Option<PoolConfig>,
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn validate_config(cfg: &ResponseStoreConfig) -> Result<(), FilterError> {
    validate_backend_available(cfg.backend)?;
    let database_url = cfg.database_url.expose_secret();
    if database_url.is_empty() {
        return Err(format!("{FILTER_NAME}: 'database_url' must not be empty").into());
    }
    validate_table_identifier(&cfg.responses_table)
        .map_err(|e| format!("{FILTER_NAME}: invalid responses_table: {e}"))?;
    validate_table_identifier(&cfg.conversations_table)
        .map_err(|e| format!("{FILTER_NAME}: invalid conversations_table: {e}"))?;
    if cfg.responses_table.eq_ignore_ascii_case(&cfg.conversations_table) {
        return Err(format!("{FILTER_NAME}: response and conversation table names must be distinct").into());
    }
    if let Some(pool) = &cfg.pool {
        pool.validate().map_err(|e| format!("{FILTER_NAME}: {e}"))?;
    }
    match cfg.backend {
        StorageBackend::Sqlite => {
            validate_sqlite_database_url(database_url)?;
            reject_postgres_fields(cfg)?;
        },
        StorageBackend::Postgres => {
            #[cfg(feature = "store-postgres")]
            validate_postgres_config(cfg, database_url)?;
        },
    }
    Ok(())
}

/// Reject a configured backend that was not compiled into this binary.
#[cfg_attr(
    all(feature = "store-postgres", feature = "store-sqlite"),
    expect(clippy::unnecessary_wraps, reason = "other feature sets reject unavailable backends")
)]
fn validate_backend_available(backend: StorageBackend) -> Result<(), FilterError> {
    match backend {
        #[cfg(not(feature = "store-sqlite"))]
        StorageBackend::Sqlite => Err(format!(
            "{FILTER_NAME}: backend 'sqlite' is unavailable; rebuild with the 'store-sqlite' feature"
        )
        .into()),
        #[cfg(feature = "store-sqlite")]
        StorageBackend::Sqlite => Ok(()),
        #[cfg(not(feature = "store-postgres"))]
        StorageBackend::Postgres => Err(format!(
            "{FILTER_NAME}: backend 'postgres' is unavailable; rebuild with the 'store-postgres' feature"
        )
        .into()),
        #[cfg(feature = "store-postgres")]
        StorageBackend::Postgres => Ok(()),
    }
}

/// Validate configuration that is specific to the `PostgreSQL` backend.
#[cfg(feature = "store-postgres")]
fn validate_postgres_config(cfg: &ResponseStoreConfig, database_url: &str) -> Result<(), FilterError> {
    postgres_url::validate_postgres_database_url(FILTER_NAME, database_url, cfg.allow_private_database_url)?;
    validate_postgres_table_identifiers(&cfg.responses_table, &cfg.conversations_table)
        .map_err(|e| format!("{FILTER_NAME}: invalid postgres table identifier: {e}"))?;
    validate_postgres_ssl_config(cfg, database_url)
}

/// Reject `..` segments in the SQLite file path to prevent a
/// crafted `database_url` from escaping the intended directory
/// and creating or overwriting files elsewhere on the filesystem.
fn validate_sqlite_database_url(database_url: &str) -> Result<(), FilterError> {
    if is_memory_database_url(database_url) {
        return Ok(());
    }

    let path = sqlite_file_path(database_url).unwrap_or(database_url);
    let path = percent_decode_str(path)
        .decode_utf8()
        .map_err(|e| format!("{FILTER_NAME}: database_url path must be valid UTF-8: {e}"))?;
    if has_dot_dot_traversal(&path) {
        return Err(format!("{FILTER_NAME}: database_url must not contain '..' path traversal").into());
    }
    Ok(())
}

/// Re-validate only the `PostgreSQL` host/IP portions of the
/// connection URL immediately before `SQLx` resolves and connects.
///
/// Full config validation runs once at construction time in
/// [`validate_config`]. This narrower check guards against DNS
/// rebinding between validation and connection by re-checking
/// the SSRF-sensitive host rules on every retry without
/// redundantly re-validating immutable fields (table names, SSL
/// config, URL scheme).
#[cfg(feature = "store-postgres")]
pub(crate) fn revalidate_postgres_host(cfg: &ResponseStoreConfig) -> Result<(), FilterError> {
    let database_url = cfg.database_url.expose_secret();
    postgres_url::revalidate_postgres_host(FILTER_NAME, database_url, cfg.allow_private_database_url)
}

/// Validate `PostgreSQL` TLS options.
#[cfg(feature = "store-postgres")]
fn validate_postgres_ssl_config(cfg: &ResponseStoreConfig, database_url: &str) -> Result<(), FilterError> {
    validate_postgres_url_tls_file_params(FILTER_NAME, database_url)?;

    if let Some(root_cert) = &cfg.ssl_root_cert {
        let root_cert = root_cert.expose_secret();
        if has_dot_dot_traversal(root_cert) {
            return Err(format!("{FILTER_NAME}: ssl_root_cert must not contain '..' path traversal").into());
        }
    }

    if has_postgres_ssl_root_cert(cfg, database_url) && !has_verified_postgres_ssl_mode(cfg, database_url) {
        return Err(format!("{FILTER_NAME}: 'ssl_root_cert' requires ssl_mode 'verify-ca' or 'verify-full'").into());
    }
    Ok(())
}

/// Return whether any configured `PostgreSQL` root CA path is present.
#[cfg(feature = "store-postgres")]
fn has_postgres_ssl_root_cert(cfg: &ResponseStoreConfig, database_url: &str) -> bool {
    cfg.ssl_root_cert.is_some() || has_postgres_url_ssl_root_cert(database_url)
}

/// Return whether the effective `PostgreSQL` SSL mode verifies certificates.
///
/// When no explicit `ssl_mode` is set, the runtime default is
/// [`SslMode::VerifyFull`], so the `None` case is considered verified
/// unless the URL carries a non-verifying `sslmode`.
#[cfg(feature = "store-postgres")]
fn has_verified_postgres_ssl_mode(cfg: &ResponseStoreConfig, database_url: &str) -> bool {
    match cfg.ssl_mode {
        Some(SslMode::VerifyCa | SslMode::VerifyFull) => true,
        Some(SslMode::Disable | SslMode::Prefer | SslMode::Require) => false,
        None => postgres_url_sslmode(database_url)
            .as_deref()
            .is_none_or(is_verified_postgres_sslmode),
    }
}

/// Reject `PostgreSQL`-specific fields when backend is SQLite.
fn reject_postgres_fields(cfg: &ResponseStoreConfig) -> Result<(), FilterError> {
    if cfg.ssl_mode.is_some() {
        return Err(format!("{FILTER_NAME}: 'ssl_mode' is only valid with the 'postgres' backend").into());
    }
    if cfg.ssl_root_cert.is_some() {
        return Err(format!("{FILTER_NAME}: 'ssl_root_cert' is only valid with the 'postgres' backend").into());
    }
    if cfg.allow_private_database_url {
        return Err(
            format!("{FILTER_NAME}: 'allow_private_database_url' is only valid with the 'postgres' backend").into(),
        );
    }
    Ok(())
}

/// Return whether a SQLite URL targets an in-memory database.
fn is_memory_database_url(database_url: &str) -> bool {
    let url = database_url.trim();
    if url == "sqlite::memory:" || url == "sqlite://:memory:" {
        return true;
    }
    url.split_once('?')
        .map_or("", |(_, query)| query)
        .split('&')
        .any(|param| param == "mode=memory")
}

/// Extract the file path component from a SQLite URL.
fn sqlite_file_path(database_url: &str) -> Option<&str> {
    database_url
        .strip_prefix("sqlite://")
        .or_else(|| database_url.strip_prefix("sqlite:"))
        .map(|rest| rest.split_once('?').map_or(rest, |(path, _query)| path))
}

#[cfg(test)]
#[cfg(any(not(feature = "store-postgres"), not(feature = "store-sqlite")))]
#[expect(clippy::allow_attributes, reason = "test-only panic assertions")]
#[allow(clippy::expect_used, reason = "tests")]
mod backend_availability_tests {
    use super::super::ResponseStoreFilter;

    #[cfg(not(feature = "store-sqlite"))]
    #[test]
    fn from_config_rejects_sqlite_when_backend_is_not_compiled() {
        let yaml = serde_yaml::from_str(
            "backend: sqlite\n\
             database_url: 'sqlite::memory:'\n\
             responses_table: responses\n\
             conversations_table: conversations\n",
        )
        .expect("valid YAML");

        let error = ResponseStoreFilter::from_config(&yaml)
            .err()
            .expect("unavailable SQLite backend must fail during construction");

        assert!(error.to_string().contains("'store-sqlite' feature"), "{error}");
    }

    #[cfg(not(feature = "store-postgres"))]
    #[test]
    fn from_config_rejects_postgres_when_backend_is_not_compiled() {
        let yaml = serde_yaml::from_str(
            "backend: postgres\n\
             database_url: 'postgres://user:password@example.com/database'\n\
             responses_table: responses\n\
             conversations_table: conversations\n",
        )
        .expect("valid YAML");

        let error = ResponseStoreFilter::from_config(&yaml)
            .err()
            .expect("unavailable PostgreSQL backend must fail during construction");

        assert!(error.to_string().contains("'store-postgres' feature"), "{error}");
    }
}
