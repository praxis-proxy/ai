// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SQL-free store configuration shared by every backend.
//!
//! Pool tuning, `PostgreSQL` TLS mode, and table-name validation. These carry no
//! sqlx or crypto dependency, so they live in the contract crate and the
//! backend-free config surface can use them without linking a SQL driver. The
//! sqlx conversions (`PoolOptions` application, `PgSslMode` mapping) stay with
//! the SQL backends.

use std::time::Duration;

use serde::Deserialize;

use crate::types::StoreError;

/// sqlx's implicit `max_connections` when the option is left unset.
///
/// Mirrors sqlx's `PoolOptions::new`, which starts every pool at 10.
/// Validation reasons about this *effective* maximum so a
/// `min_connections` above it is rejected even when `max_connections`
/// is omitted, otherwise the runtime could never reach the requested
/// prewarmed minimum.
pub const DEFAULT_MAX_CONNECTIONS: u32 = 10;

/// Maximum length for a table name identifier.
///
/// SQLite has no identifier length limit, but table names are capped to prevent
/// pathological DDL strings from config input.
pub const MAX_IDENTIFIER_LEN: usize = 128;

/// Connection pool tuning options for store backends.
///
/// All fields are optional. When omitted, the sqlx defaults apply:
/// `max_connections = 10`, `min_connections = 0`,
/// `idle_timeout = 600s (10 min)`, `acquire_timeout = 30s`.
///
/// `min_connections` must not exceed the *effective* maximum: the
/// explicit `max_connections` when set, otherwise the sqlx default of
/// 10. Requesting a larger minimum without also raising
/// `max_connections` is rejected at config load.
///
/// # YAML
///
/// ```yaml
/// pool:
///   max_connections: 20
///   min_connections: 2
///   idle_timeout_secs: 600
///   acquire_timeout_secs: 30
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool.
    #[serde(default)]
    pub max_connections: Option<u32>,

    /// Minimum number of idle connections to maintain.
    #[serde(default)]
    pub min_connections: Option<u32>,

    /// Maximum time (in seconds) a connection can sit idle before
    /// being closed. Set to `0` to disable idle timeout.
    #[serde(default)]
    pub idle_timeout_secs: Option<u64>,

    /// Maximum time (in seconds) to wait when acquiring a connection
    /// from the pool.
    #[serde(default)]
    pub acquire_timeout_secs: Option<u64>,
}

impl PoolConfig {
    /// Reject invalid values that would cause confusing runtime
    /// failures (e.g. a zero-connection pool that deadlocks on the
    /// first query).
    ///
    /// # Errors
    ///
    /// Returns a message describing the invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_connections == Some(0) {
            return Err("pool.max_connections must be at least 1".into());
        }
        if self.acquire_timeout_secs == Some(0) {
            return Err(
                "pool.acquire_timeout_secs must be at least 1 (use idle_timeout_secs: 0 to disable idle timeout)"
                    .into(),
            );
        }
        if let Some(min) = self.min_connections {
            match self.max_connections {
                Some(max) if min > max => {
                    return Err(format!(
                        "pool.min_connections ({min}) must not exceed pool.max_connections ({max})"
                    ));
                },
                None if min > DEFAULT_MAX_CONNECTIONS => {
                    return Err(format!(
                        "pool.min_connections ({min}) must not exceed the default pool.max_connections \
                         ({DEFAULT_MAX_CONNECTIONS}); set pool.max_connections explicitly to raise the maximum"
                    ));
                },
                _ => {},
            }
        }
        Ok(())
    }

    /// Convert `idle_timeout_secs` to a [`Duration`], treating `0`
    /// as "no timeout" (returns `None`).
    #[must_use]
    #[expect(clippy::option_option, reason = "three-valued: unset / disabled(0) / duration")]
    pub fn idle_timeout(&self) -> Option<Option<Duration>> {
        self.idle_timeout_secs.map(|secs| {
            if secs == 0 {
                None
            } else {
                Some(Duration::from_secs(secs))
            }
        })
    }

    /// Convert `acquire_timeout_secs` to a [`Duration`].
    #[must_use]
    pub fn acquire_timeout(&self) -> Option<Duration> {
        self.acquire_timeout_secs.map(Duration::from_secs)
    }
}

/// TLS mode for `PostgreSQL` connections.
///
/// Defaults to [`VerifyFull`](Self::VerifyFull), which requires TLS and
/// verifies both the server certificate chain and hostname. Use [`Disable`](Self::Disable)
/// or [`Prefer`](Self::Prefer) only for local development with an explicit opt-in.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SslMode {
    /// Do not use TLS.
    Disable,

    /// Attempt TLS, falling back to plaintext.
    Prefer,

    /// Require TLS without verifying the server certificate.
    Require,

    /// Require TLS and verify the server certificate chain.
    VerifyCa,

    /// Require TLS and verify both the certificate chain and hostname.
    #[default]
    VerifyFull,
}

/// Validate a store table-name identifier.
///
/// DDL interpolates table names unquoted, so the accepted set is restricted to a
/// leading letter or underscore followed by alphanumerics and underscores.
///
/// # Errors
///
/// Returns [`StoreError::Database`] when the name is empty, too long, or holds a
/// character that is unsafe to interpolate.
pub fn validate_table_identifier(name: &str) -> Result<(), StoreError> {
    if name.is_empty() {
        return Err(StoreError::Database("table name must not be empty".to_owned()));
    }
    if name.len() > MAX_IDENTIFIER_LEN {
        return Err(StoreError::Database(format!(
            "table name exceeds {MAX_IDENTIFIER_LEN} characters: {name}"
        )));
    }
    if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return Err(StoreError::Database(format!(
            "table name must start with a letter or underscore: {name}"
        )));
    }
    // Hyphens are valid in quoted SQLite identifiers but table names are
    // interpolated unquoted, so restrict to alphanumeric + underscore.
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err(StoreError::Database(format!(
            "table name contains invalid characters: {name}"
        )));
    }
    Ok(())
}
