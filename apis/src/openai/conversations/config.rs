// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Configuration types for the conversations filter.

use percent_encoding::percent_decode_str;
use praxis_filter::{FilterError, has_dot_dot_traversal};
use secrecy::{ExposeSecret as _, SecretString};
use serde::Deserialize;

use crate::store::{
    PoolConfig, SslMode,
    postgres_url::{
        self, has_postgres_url_ssl_root_cert, is_verified_postgres_sslmode, postgres_url_sslmode,
        validate_postgres_url_tls_file_params,
    },
    validate_postgres_table_set_identifiers, validate_table_identifier,
};

/// Filter name used in SSRF validation error messages.
const FILTER_NAME: &str = "openai_conversations";

// -----------------------------------------------------------------------------
// StorageBackend
// -----------------------------------------------------------------------------

/// Supported storage backends.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StorageBackend {
    /// SQLite backend (file-backed or in-memory).
    Sqlite,

    /// `PostgreSQL` backend.
    Postgres,
}

// -----------------------------------------------------------------------------
// ConversationsConfig
// -----------------------------------------------------------------------------

/// YAML configuration for the [`OpenaiConversationsFilter`].
///
/// [`OpenaiConversationsFilter`]: super::OpenaiConversationsFilter
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConversationsConfig {
    /// Storage backend to use.
    pub backend: StorageBackend,

    /// Database connection URL. Wrapped in [`SecretString`] to
    /// prevent accidental logging of credentials.
    pub database_url: SecretString,

    /// Table name for conversation records.
    #[serde(default = "default_conversations_table")]
    pub conversations_table: String,

    /// Table name for conversation item records.
    #[serde(default = "default_items_table")]
    pub items_table: String,

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

/// Serde default for [`ConversationsConfig::conversations_table`].
fn default_conversations_table() -> String {
    "openai_conversations".to_owned()
}

/// Serde default for [`ConversationsConfig::items_table`].
fn default_items_table() -> String {
    "openai_conversation_items".to_owned()
}

impl ConversationsConfig {
    /// Return the generated internal responses table name.
    ///
    /// The store constructors require a responses table, but the
    /// conversations filter doesn't use it. This generates a
    /// deterministic name so the DDL runs cleanly; the table
    /// exists but remains empty.
    pub fn responses_table(&self) -> String {
        format!("{}_unused_responses", self.conversations_table)
    }
}

// -----------------------------------------------------------------------------
// Config Validation
// -----------------------------------------------------------------------------

/// Validate the parsed configuration.
pub(crate) fn validate_config(cfg: &ConversationsConfig) -> Result<(), FilterError> {
    let database_url = cfg.database_url.expose_secret();
    if database_url.is_empty() {
        return Err(format!("{FILTER_NAME}: 'database_url' must not be empty").into());
    }
    if let Some(pool) = &cfg.pool {
        pool.validate().map_err(|e| format!("{FILTER_NAME}: {e}"))?;
    }
    let responses_table = validate_table_names(cfg)?;
    match cfg.backend {
        StorageBackend::Sqlite => {
            validate_sqlite_database_url(database_url)?;
            reject_postgres_fields(cfg)?;
        },
        StorageBackend::Postgres => {
            postgres_url::validate_postgres_database_url(FILTER_NAME, database_url, cfg.allow_private_database_url)?;
            validate_postgres_table_set_identifiers(&responses_table, &cfg.conversations_table, Some(&cfg.items_table))
                .map_err(|e| format!("{FILTER_NAME}: invalid postgres table identifier: {e}"))?;
            validate_postgres_ssl_config(cfg, database_url)?;
        },
    }
    Ok(())
}

/// Validate all table name identifiers and uniqueness constraints.
fn validate_table_names(cfg: &ConversationsConfig) -> Result<String, FilterError> {
    validate_table_identifier(&cfg.conversations_table)
        .map_err(|e| format!("{FILTER_NAME}: invalid conversations_table: {e}"))?;
    validate_table_identifier(&cfg.items_table).map_err(|e| format!("{FILTER_NAME}: invalid items_table: {e}"))?;
    if cfg.conversations_table.eq_ignore_ascii_case(&cfg.items_table) {
        return Err(format!("{FILTER_NAME}: conversations and items table names must be distinct").into());
    }
    let responses_table = cfg.responses_table();
    validate_table_identifier(&responses_table)
        .map_err(|e| format!("{FILTER_NAME}: invalid generated responses_table: {e}"))?;
    if responses_table.eq_ignore_ascii_case(&cfg.items_table) {
        return Err(format!("{FILTER_NAME}: generated responses and items table names must be distinct").into());
    }
    Ok(responses_table)
}

/// Reject `..` segments in the SQLite file path.
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
pub(crate) fn revalidate_postgres_host(cfg: &ConversationsConfig) -> Result<(), FilterError> {
    let database_url = cfg.database_url.expose_secret();
    postgres_url::revalidate_postgres_host(FILTER_NAME, database_url, cfg.allow_private_database_url)
}

/// Validate `PostgreSQL` TLS options.
fn validate_postgres_ssl_config(cfg: &ConversationsConfig, database_url: &str) -> Result<(), FilterError> {
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
fn has_postgres_ssl_root_cert(cfg: &ConversationsConfig, database_url: &str) -> bool {
    cfg.ssl_root_cert.is_some() || has_postgres_url_ssl_root_cert(database_url)
}

/// Return whether the effective `PostgreSQL` SSL mode verifies certificates.
///
/// When no explicit `ssl_mode` is set, the runtime default is
/// [`SslMode::VerifyFull`], so the `None` case is considered verified
/// unless the URL carries a non-verifying `sslmode`.
fn has_verified_postgres_ssl_mode(cfg: &ConversationsConfig, database_url: &str) -> bool {
    match cfg.ssl_mode {
        Some(SslMode::VerifyCa | SslMode::VerifyFull) => true,
        Some(SslMode::Disable | SslMode::Prefer | SslMode::Require) => false,
        None => postgres_url_sslmode(database_url)
            .as_deref()
            .is_none_or(is_verified_postgres_sslmode),
    }
}

/// Reject `PostgreSQL`-specific fields when backend is SQLite.
fn reject_postgres_fields(cfg: &ConversationsConfig) -> Result<(), FilterError> {
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
