// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Concrete store-backend factories and connection-error redaction.
//!
//! The factories wrap the existing SQL backends (still in apis pre-#1260) as
//! [`praxis_ai_store::StoreBackendFactory`] implementations the lifecycle layer provisions. They
//! own the backend-specific config, classify a build failure so the cache
//! applies the tested policy (SQLite permanent-init failure is unavailable;
//! Postgres transient connect is retryable), compute the dedup key, and close
//! the pool on retirement.

#[cfg(any(feature = "sqlite", feature = "_postgres"))]
use std::time::Duration;

#[cfg(any(feature = "sqlite", feature = "_postgres"))]
use praxis_ai_store::BackendError;

#[cfg(any(feature = "sqlite", feature = "_postgres"))]
use super::redact_connection_error;

/// A permanent build failure, redacted, as a backend-unavailable error.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn permanent(url: &str, message: &str) -> BackendError {
    BackendError::Unavailable(redact_connection_error(url, message))
}

/// Maximum time one backend may spend opening its pool and preparing schemas.
///
/// Pool acquisition timeouts do not bound a schema statement waiting on a
/// database lock, so provisioning needs its own end-to-end deadline.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
const BACKEND_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(30);

/// A transient build failure, redacted, so provisioning retries within budget.
#[cfg(feature = "_postgres")]
fn transient(url: &str, message: &str) -> BackendError {
    BackendError::Transient(redact_connection_error(url, message))
}

/// A stable fingerprint of the pool overrides for the dedup key.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn pool_fingerprint(pool: Option<&crate::PoolConfig>) -> String {
    let pool = pool.cloned().unwrap_or_default();
    format!(
        "{}/{}/{}/{}",
        pool.max_connections.unwrap_or(praxis_ai_store::DEFAULT_MAX_CONNECTIONS),
        pool.min_connections.unwrap_or(praxis_ai_store::DEFAULT_MIN_CONNECTIONS),
        pool.idle_timeout_secs
            .unwrap_or(praxis_ai_store::DEFAULT_IDLE_TIMEOUT_SECS),
        pool.acquire_timeout_secs
            .unwrap_or(praxis_ai_store::DEFAULT_ACQUIRE_TIMEOUT_SECS),
    )
}

/// A stable fingerprint of the compression override for the dedup key.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
fn compression_fingerprint(compression: Option<&praxis_ai_store::StoreCompressionConfig>) -> String {
    use praxis_ai_store::CompressionAlgorithm;

    match compression {
        None
        | Some(praxis_ai_store::StoreCompressionConfig {
            algorithm: CompressionAlgorithm::None,
            level: None,
        }) => "none".to_owned(),
        Some(praxis_ai_store::StoreCompressionConfig {
            algorithm: CompressionAlgorithm::None,
            level: Some(level),
        }) => format!("invalid-none/{level}"),
        Some(praxis_ai_store::StoreCompressionConfig {
            algorithm: CompressionAlgorithm::Zstd,
            level,
        }) => format!("zstd/{}", level.unwrap_or(3)),
    }
}

/// SQLite-backed store-backend factory.
#[cfg(feature = "sqlite")]
mod sqlite {
    use std::{str::FromStr as _, sync::Arc};

    use async_trait::async_trait;
    use percent_encoding::percent_decode_str;
    use praxis_ai_store::{
        BackendNamespaceKey, EffectiveConfigKey, PoolConfig, ProvisionedBackend, RetireBackend, StoreBackendFactory,
        StoreCompressionConfig,
    };
    use secrecy::{ExposeSecret as _, SecretString};
    use serde::Deserialize;
    use serde_json::Value;
    use sqlx::sqlite::SqliteConnectOptions;

    use super::{BACKEND_INITIALIZATION_TIMEOUT, BackendError, permanent};
    use crate::{SqliteResponseStore, sqlite::is_memory_database_url};

    /// Backend id the SQLite factory answers to.
    pub(crate) const BACKEND_ID: &str = "sqlite";

    /// Inline configuration for a SQLite-backed store.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SqliteConfig {
        /// SQLite connection string.
        database_url: SecretString,
        /// Responses table name.
        responses_table: String,
        /// Conversations table name.
        conversations_table: String,
        /// Optional conversation-items table name.
        #[serde(default)]
        items_table: Option<String>,
        /// Optional connection-pool overrides.
        #[serde(default)]
        pool: Option<PoolConfig>,
        /// Optional payload compression for stored JSON columns.
        #[serde(default)]
        compression: Option<StoreCompressionConfig>,
    }

    /// Retirement hook that closes a SQLite pool.
    struct SqliteRetire {
        /// The store whose pool is closed on retirement.
        store: Arc<SqliteResponseStore>,
    }

    #[async_trait]
    impl RetireBackend for SqliteRetire {
        async fn retire(&self) {
            self.store.close().await;
        }
    }

    /// Provisions SQLite-backed persisted-state stores.
    pub(crate) struct SqliteBackendFactory;

    impl SqliteBackendFactory {
        /// Parse the inline config, failing with a config error.
        fn parse(config: &Value) -> Result<SqliteConfig, BackendError> {
            serde_json::from_value(config.clone()).map_err(|e| BackendError::Config(e.to_string()))
        }
    }

    #[async_trait]
    impl StoreBackendFactory for SqliteBackendFactory {
        fn backend_id(&self) -> &str {
            BACKEND_ID
        }

        fn effective_key(&self, config: &Value) -> Result<EffectiveConfigKey, BackendError> {
            let cfg = Self::parse(config)?;
            let url = cfg.database_url.expose_secret();
            let pool = if is_memory_database_url(url) {
                "in-memory-single-connection".to_owned()
            } else {
                super::pool_fingerprint(cfg.pool.as_ref())
            };
            // The pool + url + table names identify one SQLite store instance.
            let key = format!(
                "sqlite\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                url,
                cfg.responses_table,
                cfg.conversations_table,
                cfg.items_table.as_deref().unwrap_or(""),
                pool,
                super::compression_fingerprint(cfg.compression.as_ref()),
            );
            Ok(EffectiveConfigKey::new(key))
        }

        #[expect(
            clippy::too_many_lines,
            reason = "SQLite URI mode, cache, and filename semantics form one namespace decision"
        )]
        fn namespace_key(&self, config: &Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            let cfg = Self::parse(config)?;
            let url = cfg.database_url.expose_secret();
            let sqlite_url = url
                .trim()
                .strip_prefix("sqlite://")
                .or_else(|| url.trim().strip_prefix("sqlite:"))
                .unwrap_or_else(|| url.trim());
            let (database, query) = sqlite_url.split_once('?').unwrap_or((sqlite_url, ""));
            let database = percent_decode_str(database).decode_utf8_lossy();
            let mut memory_mode = false;
            let mut shared_cache = false;
            let mut memory_vfs = false;
            for parameter in query.split('&') {
                let parameter = percent_decode_str(parameter).decode_utf8_lossy();
                if parameter.eq_ignore_ascii_case("mode=memory") {
                    memory_mode = true;
                    shared_cache = true;
                } else if parameter.eq_ignore_ascii_case("cache=private") {
                    shared_cache = false;
                } else if parameter.eq_ignore_ascii_case("cache=shared") {
                    shared_cache = true;
                } else if parameter.eq_ignore_ascii_case("vfs=memdb") {
                    memory_vfs = true;
                }
            }
            let absolute_memdb = memory_vfs && std::path::Path::new(database.as_ref()).is_absolute();
            let private_memory = database.is_empty()
                || database == ":memory:"
                || (database == "file::memory:" && !shared_cache)
                || (memory_mode && !shared_cache)
                || (memory_vfs && !absolute_memdb);
            if private_memory {
                return Ok(None);
            }
            let options = SqliteConnectOptions::from_str(url)
                .map_err(|error| BackendError::Config(format!("invalid SQLite database URL: {error}")))?;
            let (namespace_kind, path) = if memory_vfs {
                ("sqlite-memdb", options.get_filename().to_path_buf())
            } else if memory_mode {
                ("sqlite-memory-mode", options.get_filename().to_path_buf())
            } else if database == "file::memory:" {
                ("sqlite-memory-uri", options.get_filename().to_path_buf())
            } else {
                let path = std::path::absolute(options.get_filename())
                    .map_err(|error| BackendError::Config(format!("cannot resolve SQLite database path: {error}")))?;
                ("sqlite-file", path)
            };
            Ok(Some(BackendNamespaceKey::new(format!(
                "{namespace_kind}\u{1f}{}",
                path.display()
            ))))
        }

        async fn build(&self, config: &Value) -> Result<ProvisionedBackend, BackendError> {
            let cfg = Self::parse(config)?;
            let url = cfg.database_url.expose_secret();
            // SQLite init failure is permanent: fail the build unavailable.
            let store = tokio::time::timeout(
                BACKEND_INITIALIZATION_TIMEOUT,
                SqliteResponseStore::new(
                    url,
                    &cfg.responses_table,
                    &cfg.conversations_table,
                    cfg.items_table.as_deref(),
                    cfg.pool.as_ref(),
                    cfg.compression.as_ref(),
                ),
            )
            .await
            .map_err(|_elapsed| permanent(url, "backend initialization timed out after 30 seconds"))?
            .map_err(|e| permanent(url, &e.to_string()))?;

            let store = Arc::new(store);
            Ok(ProvisionedBackend {
                retire: Arc::new(SqliteRetire {
                    store: Arc::clone(&store),
                }),
                backend: store,
            })
        }
    }
}

/// Postgres-backed store-backend factory.
#[cfg(feature = "_postgres")]
mod postgres {
    use std::{str::FromStr as _, sync::Arc};

    use async_trait::async_trait;
    use praxis_ai_store::{
        BackendNamespaceKey, EffectiveConfigKey, PoolConfig, ProvisionedBackend, RetireBackend, SslMode,
        StoreBackendFactory, StoreCompressionConfig, StoreError,
    };
    use secrecy::{ExposeSecret as _, SecretString};
    use serde::Deserialize;
    use serde_json::Value;
    use sqlx::postgres::PgConnectOptions;

    use super::{BACKEND_INITIALIZATION_TIMEOUT, BackendError, permanent, transient};
    use crate::{PgTlsConfig, PostgresResponseStore, postgres_url};

    /// Backend id the Postgres factory answers to.
    pub(crate) const BACKEND_ID: &str = "postgres";

    /// Inline configuration for a Postgres-backed store.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PostgresConfig {
        /// Postgres connection string.
        database_url: SecretString,
        /// Responses table name.
        responses_table: String,
        /// Conversations table name.
        conversations_table: String,
        /// Optional conversation-items table name.
        #[serde(default)]
        items_table: Option<String>,
        /// Optional connection-pool overrides.
        #[serde(default)]
        pool: Option<PoolConfig>,
        /// TLS verification mode.
        #[serde(default)]
        ssl_mode: Option<SslMode>,
        /// Path to a PEM CA the server certificate is verified against.
        #[serde(default)]
        ssl_root_cert: Option<SecretString>,
        /// Path to a PEM client certificate for mutual TLS.
        #[serde(default)]
        ssl_client_cert: Option<SecretString>,
        /// Path to the PEM client key paired with `ssl_client_cert`.
        #[serde(default)]
        ssl_client_key: Option<SecretString>,
        /// Enforce the certificate-authentication compliance profile.
        #[serde(default)]
        require_certificate_authentication: bool,
        /// Permit a private/loopback database host (opt-in).
        #[serde(default)]
        allow_private_database_url: bool,
        /// Optional payload compression for stored JSON columns.
        #[serde(default)]
        compression: Option<StoreCompressionConfig>,
    }

    /// Retirement hook that closes a Postgres pool.
    struct PostgresRetire {
        /// The store whose pool is closed on retirement.
        store: Arc<PostgresResponseStore>,
    }

    #[async_trait]
    impl RetireBackend for PostgresRetire {
        async fn retire(&self) {
            self.store.close().await;
        }
    }

    /// Provisions Postgres-backed persisted-state stores.
    pub(crate) struct PostgresBackendFactory;

    /// Owned certificate-path material a borrowed [`PgTlsConfig`] outlives.
    type TlsPaths = (Option<String>, Option<String>, Option<String>);

    impl PostgresConfig {
        /// Own the cert-path material so a borrowed `PgTlsConfig` can reference it.
        fn tls_paths(&self) -> TlsPaths {
            (
                self.ssl_root_cert.as_ref().map(|s| s.expose_secret().to_owned()),
                self.ssl_client_cert.as_ref().map(|s| s.expose_secret().to_owned()),
                self.ssl_client_key.as_ref().map(|s| s.expose_secret().to_owned()),
            )
        }

        /// Borrow the owned paths as a `PgTlsConfig`.
        fn tls_config<'a>(&self, paths: &'a TlsPaths) -> PgTlsConfig<'a> {
            PgTlsConfig {
                require_certificate_authentication: self.require_certificate_authentication,
                ssl_client_cert: paths.1.as_deref(),
                ssl_client_key: paths.2.as_deref(),
                ssl_mode: self.ssl_mode,
                ssl_root_cert: paths.0.as_deref(),
            }
        }
    }

    impl PostgresBackendFactory {
        /// Parse the inline config, failing with a config error.
        fn parse(config: &Value) -> Result<PostgresConfig, BackendError> {
            serde_json::from_value(config.clone()).map_err(|e| BackendError::Config(e.to_string()))
        }

        /// Open and validate the pool. A connect failure is transient so the
        /// cache retries within budget. A bad host is a permanent config error.
        async fn connect(cfg: &PostgresConfig) -> Result<PostgresResponseStore, BackendError> {
            let url = cfg.database_url.expose_secret();
            // Re-validate the host on every attempt (guards DNS rebinding).
            postgres_url::revalidate_postgres_host(BACKEND_ID, url, cfg.allow_private_database_url)
                .map_err(|e| BackendError::Config(e.to_string()))?;
            let paths = cfg.tls_paths();
            let tls = cfg.tls_config(&paths);
            // Same fail-closed TLS/auth check the filter runs, before the pool opens.
            tls.validate(BACKEND_ID, url)
                .map_err(|e| BackendError::Config(e.to_string()))?;
            tokio::time::timeout(
                BACKEND_INITIALIZATION_TIMEOUT,
                Box::pin(PostgresResponseStore::new(
                    url,
                    &cfg.responses_table,
                    &cfg.conversations_table,
                    cfg.items_table.as_deref(),
                    &tls,
                    cfg.pool.as_ref(),
                    cfg.compression.as_ref(),
                )),
            )
            .await
            .map_err(|_elapsed| permanent(url, "backend initialization timed out after 30 seconds"))?
            .map_err(|error| classify_initialization_error(url, &error))
        }
    }

    /// Preserve terminal schema diagnostics instead of retrying them as
    /// connectivity failures.
    pub(super) fn classify_initialization_error(url: &str, error: &StoreError) -> BackendError {
        let message = error.to_string();
        if matches!(
            error,
            StoreError::InvalidInput(_) | StoreError::Serialization(_) | StoreError::Unavailable(_)
        ) || message.contains("database recreation required")
        {
            permanent(url, &message)
        } else {
            transient(url, &message)
        }
    }

    #[async_trait]
    impl StoreBackendFactory for PostgresBackendFactory {
        fn backend_id(&self) -> &str {
            BACKEND_ID
        }

        fn effective_key(&self, config: &Value) -> Result<EffectiveConfigKey, BackendError> {
            let cfg = Self::parse(config)?;
            // Key on the cert-path values, not mere presence: distinct trust
            // anchors or client identities at different paths are distinct pools.
            let key = format!(
                "postgres\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{:?}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                cfg.database_url.expose_secret(),
                cfg.responses_table,
                cfg.conversations_table,
                cfg.items_table.as_deref().unwrap_or(""),
                cfg.ssl_mode.unwrap_or_default(),
                cfg.ssl_root_cert.as_ref().map_or("", |s| s.expose_secret()),
                cfg.ssl_client_cert.as_ref().map_or("", |s| s.expose_secret()),
                cfg.ssl_client_key.as_ref().map_or("", |s| s.expose_secret()),
                cfg.require_certificate_authentication,
                super::pool_fingerprint(cfg.pool.as_ref()),
                super::compression_fingerprint(cfg.compression.as_ref()),
            );
            Ok(EffectiveConfigKey::new(key))
        }

        fn namespace_key(&self, config: &Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            let cfg = Self::parse(config)?;
            let options = PgConnectOptions::from_str(cfg.database_url.expose_secret())
                .map_err(|error| BackendError::Config(format!("invalid PostgreSQL database URL: {error}")))?;
            let database = options.get_database().unwrap_or_else(|| options.get_username());
            let endpoint = options
                .get_socket()
                .map_or_else(|| options.get_host().to_owned(), |socket| socket.display().to_string());
            Ok(Some(BackendNamespaceKey::new(format!(
                "postgres\u{1f}{endpoint}\u{1f}{}\u{1f}{database}\u{1f}{}",
                options.get_port(),
                options.get_options().unwrap_or_default()
            ))))
        }

        fn validate_config(&self, config: &Value) -> Result<(), BackendError> {
            let cfg = Self::parse(config)?;
            let url = cfg.database_url.expose_secret();
            // Run the SSRF-sensitive host check at construction so a private or
            // loopback host fails fatally at startup rather than passing here and
            // failing later in async provisioning, where the permanent error
            // would otherwise loop with readiness stuck at failed.
            postgres_url::revalidate_postgres_host(BACKEND_ID, url, cfg.allow_private_database_url)
                .map_err(|e| BackendError::Config(e.to_string()))?;
            let paths = cfg.tls_paths();
            let tls = cfg.tls_config(&paths);
            // Run the filter's fail-closed TLS/auth check at pipeline construction.
            tls.validate(BACKEND_ID, url)
                .map_err(|e| BackendError::Config(e.to_string()))
        }

        async fn build(&self, config: &Value) -> Result<ProvisionedBackend, BackendError> {
            let cfg = Self::parse(config)?;
            let store = Arc::new(Self::connect(&cfg).await?);
            Ok(ProvisionedBackend {
                retire: Arc::new(PostgresRetire {
                    store: Arc::clone(&store),
                }),
                backend: store,
            })
        }
    }
}

/// The concrete store-backend factories compiled into this build, for a binary
/// to inject into the lifecycle cache. Empty when no backend feature is set.
#[cfg(any(feature = "sqlite", feature = "_postgres"))]
#[must_use]
#[expect(clippy::vec_init_then_push, reason = "each push is feature-gated")]
pub fn store_backend_factories() -> Vec<std::sync::Arc<dyn praxis_ai_store::StoreBackendFactory>> {
    let mut factories: Vec<std::sync::Arc<dyn praxis_ai_store::StoreBackendFactory>> = Vec::new();
    #[cfg(feature = "sqlite")]
    factories.push(std::sync::Arc::new(sqlite::SqliteBackendFactory));
    #[cfg(feature = "_postgres")]
    factories.push(std::sync::Arc::new(postgres::PostgresBackendFactory));
    factories
}

#[cfg(test)]
#[cfg(feature = "sqlite")]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::sync::Arc;

    use praxis_ai_store::StoreBackendFactory;
    use praxis_ai_store_lifecycle::{BackendCache, ProvisionError, StoreRef};
    use serde_json::json;
    use tempfile::TempDir;

    #[cfg(feature = "_postgres")]
    use super::postgres::PostgresBackendFactory;
    use super::sqlite::SqliteBackendFactory;

    fn with_field(mut config: serde_json::Value, name: &str, value: serde_json::Value) -> serde_json::Value {
        config
            .as_object_mut()
            .expect("test store config must be an object")
            .insert(name.to_owned(), value);
        config
    }

    /// Build a cache holding only the real SQLite factory.
    fn sqlite_cache() -> BackendCache {
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(SqliteBackendFactory);
        BackendCache::new(vec![factory])
    }

    /// A store reference against a file-backed SQLite database at `path`.
    fn sqlite_ref(name: &str, dir: &TempDir, file: &str) -> StoreRef {
        let url = format!("sqlite://{}/{file}?mode=rwc", dir.path().display());
        StoreRef {
            name: Arc::from(name),
            backend_id: Arc::from("sqlite"),
            config: json!({
                "database_url": url,
                "responses_table": "responses",
                "conversations_table": "conversations",
                "items_table": "conversation_items",
            }),
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "compares every semantically equivalent default form"
    )]
    fn sqlite_effective_key_canonicalizes_pool_and_compression_defaults() {
        let base = json!({
            "database_url": "sqlite::memory:",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let factory = SqliteBackendFactory;
        let expected = factory.effective_key(&base).expect("base key");

        let empty_pool = factory
            .effective_key(&with_field(base.clone(), "pool", json!({})))
            .expect("empty pool key");
        let explicit_pool = factory
            .effective_key(&with_field(
                base.clone(),
                "pool",
                json!({
                    "max_connections": 10,
                    "min_connections": 0,
                    "idle_timeout_secs": 600,
                    "acquire_timeout_secs": 30,
                }),
            ))
            .expect("explicit pool defaults key");
        let compression_none = factory
            .effective_key(&with_field(base.clone(), "compression", json!({"algorithm": "none"})))
            .expect("explicit no-compression key");
        assert_eq!(empty_pool, expected);
        assert_eq!(explicit_pool, expected);
        assert_eq!(compression_none, expected);

        let implicit_zstd = factory
            .effective_key(&with_field(base.clone(), "compression", json!({"algorithm": "zstd"})))
            .expect("implicit zstd level key");
        let explicit_zstd = factory
            .effective_key(&with_field(
                base,
                "compression",
                json!({"algorithm": "zstd", "level": 3}),
            ))
            .expect("explicit zstd level key");
        assert_eq!(implicit_zstd, explicit_zstd);
    }

    #[test]
    fn sqlite_effective_key_ignores_pool_for_memory_databases() {
        let factory = SqliteBackendFactory;
        for database_url in ["sqlite::memory:", "sqlite:///tmp/praxis-shared?mode=memory"] {
            let config = |max_connections| {
                json!({
                    "database_url": database_url,
                    "responses_table": "responses",
                    "conversations_table": "conversations",
                    "pool": { "max_connections": max_connections },
                })
            };

            assert_eq!(
                factory.effective_key(&config(1)).expect("first pool key"),
                factory.effective_key(&config(8)).expect("second pool key"),
                "in-memory SQLite always uses one connection: {database_url}"
            );
        }

        let file_config = |max_connections| {
            json!({
                "database_url": "sqlite:///tmp/praxis-file.db?mode=rwc",
                "responses_table": "responses",
                "conversations_table": "conversations",
                "pool": { "max_connections": max_connections },
            })
        };
        assert_ne!(
            factory.effective_key(&file_config(1)).expect("first file pool key"),
            factory.effective_key(&file_config(8)).expect("second file pool key"),
            "file-backed pool settings remain part of backend identity"
        );
    }

    #[test]
    fn sqlite_namespace_key_ignores_pool_and_compression() {
        let base = json!({
            "database_url": "sqlite:///tmp/praxis-namespace.db?mode=rwc",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let factory = SqliteBackendFactory;
        let expected = factory.namespace_key(&base).expect("base namespace key");
        let different_runtime_settings = with_field(
            with_field(base, "pool", json!({"max_connections": 3})),
            "compression",
            json!({"algorithm": "zstd", "level": 5}),
        );

        assert_eq!(
            factory
                .namespace_key(&different_runtime_settings)
                .expect("namespace key with runtime overrides"),
            expected
        );
    }

    #[test]
    fn sqlite_private_memory_urls_have_no_namespace_key() {
        let factory = SqliteBackendFactory;
        for database_url in [
            "sqlite::memory:",
            "sqlite://:memory:",
            "sqlite://file::memory:",
            "sqlite://%3Amemory%3A",
            "sqlite://?mode=memory",
            "sqlite://",
            "sqlite://?mode=rwc",
            "sqlite://private?mode=memory&cache=private",
            "sqlite://private?vfs=memdb",
        ] {
            let config = json!({
                "database_url": database_url,
                "responses_table": "responses",
                "conversations_table": "conversations",
            });
            assert_eq!(
                factory.namespace_key(&config).expect("private memory namespace"),
                None,
                "{database_url}"
            );
        }
    }

    #[test]
    fn sqlite_named_shared_memory_has_stable_namespace_key() {
        let factory = SqliteBackendFactory;
        for database_url in ["sqlite://shared?mode=memory", "sqlite://file::memory:?cache=shared"] {
            let config = json!({
                "database_url": database_url,
                "responses_table": "responses",
                "conversations_table": "conversations",
            });

            let first = factory.namespace_key(&config).expect("first named memory namespace");
            let second = factory.namespace_key(&config).expect("second named memory namespace");
            assert!(first.is_some(), "{database_url}");
            assert_eq!(first, second, "{database_url}");
        }
    }

    #[test]
    fn sqlite_absolute_memdb_has_stable_namespace_key() {
        let factory = SqliteBackendFactory;
        let config = json!({
            "database_url": "sqlite:///tmp/praxis-absolute-memdb?vfs=memdb",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });

        let first = factory.namespace_key(&config).expect("first memdb namespace");
        let second = factory.namespace_key(&config).expect("second memdb namespace");
        assert!(first.is_some(), "absolute memdb names are shared across connections");
        assert_eq!(first, second);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the three namespace boundary assertions belong in one regression test"
    )]
    fn sqlite_named_memory_is_distinct_from_disk_at_same_path() {
        let factory = SqliteBackendFactory;
        let config = |database_url| {
            json!({
                "database_url": database_url,
                "responses_table": "responses",
                "conversations_table": "conversations",
            })
        };

        assert_ne!(
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?mode=memory"))
                .expect("memory namespace"),
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?mode=rwc"))
                .expect("file namespace"),
        );
        assert_ne!(
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?vfs=memdb"))
                .expect("memdb namespace"),
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?mode=rwc"))
                .expect("file namespace"),
        );
        assert_ne!(
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?vfs=memdb"))
                .expect("memdb namespace"),
            factory
                .namespace_key(&config("sqlite:///tmp/praxis-namespace.db?mode=memory"))
                .expect("shared-cache memory namespace"),
        );
    }

    #[test]
    fn sqlite_named_memory_preserves_exact_uri_name() {
        let factory = SqliteBackendFactory;
        let config = |database_url| {
            json!({
                "database_url": database_url,
                "responses_table": "responses",
                "conversations_table": "conversations",
            })
        };
        let absolute = format!(
            "sqlite://{}?mode=memory",
            std::env::current_dir()
                .expect("current directory")
                .join("praxis-named-memory")
                .display()
        );

        assert_ne!(
            factory
                .namespace_key(&config("sqlite://praxis-named-memory?mode=memory"))
                .expect("relative named memory namespace"),
            factory
                .namespace_key(&config(&absolute))
                .expect("absolute named memory namespace"),
        );
    }

    #[cfg(feature = "_postgres")]
    #[test]
    fn postgres_effective_key_canonicalizes_default_ssl_mode() {
        let base = json!({
            "database_url": "postgresql://user@8.8.8.8/store",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let factory = PostgresBackendFactory;
        let implicit = factory.effective_key(&base).expect("implicit TLS key");
        let explicit = factory
            .effective_key(&with_field(base, "ssl_mode", json!("verify-full")))
            .expect("explicit TLS key");

        assert_eq!(implicit, explicit);
    }

    #[cfg(feature = "_postgres")]
    #[test]
    fn postgres_namespace_key_ignores_credentials_tls_pool_and_compression() {
        let base = json!({
            "database_url": "postgresql://user:secret@db.example.com/store",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let factory = PostgresBackendFactory;
        let expected = factory.namespace_key(&base).expect("base namespace key");
        let different_runtime_settings = with_field(
            with_field(
                with_field(
                    with_field(
                        base,
                        "database_url",
                        json!("postgresql://user:other@db.example.com/store"),
                    ),
                    "ssl_mode",
                    json!("require"),
                ),
                "pool",
                json!({"max_connections": 3}),
            ),
            "compression",
            json!({"algorithm": "zstd", "level": 5}),
        );

        assert_eq!(
            factory
                .namespace_key(&different_runtime_settings)
                .expect("namespace key with runtime overrides"),
            expected
        );
    }

    #[cfg(feature = "_postgres")]
    #[test]
    fn postgres_namespace_key_conservatively_ignores_roles() {
        let base = json!({
            "database_url": "postgresql://first@db.example.com/store",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let factory = PostgresBackendFactory;
        let expected = factory.namespace_key(&base).expect("base namespace key");
        assert_eq!(
            factory
                .namespace_key(&with_field(
                    base,
                    "database_url",
                    json!("postgresql://second@db.example.com/store"),
                ))
                .expect("second-role namespace key"),
            expected,
            "credentials cannot hide collisions in the same explicit database"
        );
    }

    #[cfg(feature = "_postgres")]
    #[test]
    fn postgres_namespace_key_distinguishes_socket_ports() {
        let factory = PostgresBackendFactory;
        let socket_config = |port| {
            json!({
                "database_url": format!("postgresql:///store?host=/tmp&port={port}"),
                "responses_table": "responses",
                "conversations_table": "conversations",
            })
        };
        assert_ne!(
            factory
                .namespace_key(&socket_config(5432))
                .expect("first socket namespace key"),
            factory
                .namespace_key(&socket_config(5433))
                .expect("second socket namespace key"),
            "PostgreSQL socket filenames include the server port"
        );
    }

    #[tokio::test]
    async fn initial_load_builds_real_sqlite() {
        let dir = TempDir::new().expect("tempdir");
        let cache = sqlite_cache();

        let provisioned = cache
            .provision(&[sqlite_ref("default", &dir, "a.db")])
            .await
            .expect("real sqlite initial load");

        assert!(provisioned.registry.contains("default"));
    }

    #[tokio::test]
    async fn reload_reuses_same_sqlite_backend() {
        let dir = TempDir::new().expect("tempdir");
        let cache = sqlite_cache();
        let refs = [sqlite_ref("default", &dir, "a.db")];

        let gen1 = cache.provision(&refs).await.expect("gen1");
        let gen2 = cache.provision(&refs).await.expect("gen2");

        // Reused: releasing gen1 must not close the pool gen2 still uses, so a
        // gen2 provision-equivalent still resolves.
        gen1.lease.release().await;
        assert!(gen2.registry.contains("default"));
        gen2.lease.release().await;
    }

    #[tokio::test]
    async fn changed_url_builds_a_second_backend() {
        let dir = TempDir::new().expect("tempdir");
        let cache = sqlite_cache();

        let gen1 = cache
            .provision(&[sqlite_ref("default", &dir, "a.db")])
            .await
            .expect("gen1 on a.db");
        let gen2 = cache
            .provision(&[sqlite_ref("default", &dir, "b.db")])
            .await
            .expect("gen2 on b.db");

        assert!(gen1.registry.contains("default"));
        assert!(gen2.registry.contains("default"));
        gen1.lease.release().await;
        gen2.lease.release().await;
    }

    #[tokio::test]
    async fn unavailable_at_build_fails_closed_distinct_from_unknown() {
        let cache = sqlite_cache();
        // A read-only URL for a file that does not exist cannot initialize the
        // schema: a permanent SQLite failure.
        let bad = StoreRef {
            name: Arc::from("default"),
            backend_id: Arc::from("sqlite"),
            config: json!({
                "database_url": "sqlite:///nonexistent-dir/does-not-exist.db?mode=ro",
                "responses_table": "responses",
                "conversations_table": "conversations",
            }),
        };

        let result = cache.provision(&[bad]).await;
        let Err(err) = result else {
            panic!("expected a build failure")
        };
        match err {
            ProvisionError::Backend {
                source: praxis_ai_store::BackendError::Unavailable(_),
                ..
            } => {},
            other => panic!("expected backend-unavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dedup_shares_one_sqlite_pool() {
        let dir = TempDir::new().expect("tempdir");
        let cache = sqlite_cache();

        // Two names, one effective config (same url + tables) -> one backend.
        let provisioned = cache
            .provision(&[
                sqlite_ref("responses", &dir, "shared.db"),
                sqlite_ref("conversations", &dir, "shared.db"),
            ])
            .await
            .expect("both provision onto one pool");

        assert!(provisioned.registry.contains("responses"));
        assert!(provisioned.registry.contains("conversations"));
        assert!(
            provisioned.registry.shares_storage_with(&provisioned.registry),
            "same registry",
        );
        provisioned.lease.release().await;
    }
}

#[cfg(test)]
#[cfg(feature = "_postgres")]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "tests")]
mod postgres_tests {
    use praxis_ai_store::{BackendError, StoreBackendFactory as _, StoreError};
    use serde_json::json;

    use super::postgres::{PostgresBackendFactory, classify_initialization_error};

    #[test]
    fn schema_initialization_errors_are_terminal() {
        let schema_error = StoreError::Database(
            "schema validation failed: incompatible table; database recreation required".to_owned(),
        );
        assert!(matches!(
            classify_initialization_error("postgresql://user:secret@example.com/store", &schema_error),
            BackendError::Unavailable(message)
                if message.contains("database recreation required") && !message.contains("secret")
        ));

        let connection_error = StoreError::Database("connection refused".to_owned());
        assert!(matches!(
            classify_initialization_error("postgresql://user@example.com/store", &connection_error),
            BackendError::Transient(message) if message.contains("connection refused")
        ));
    }

    /// A client-cert mTLS config the response-store filter accepts must also
    /// pass factory validation: the factory runs the same TLS check.
    #[test]
    fn validate_accepts_client_cert_mtls_config() {
        let cfg = json!({
            "database_url": "postgres://svc@db.example.com:5432/app",
            "allow_private_database_url": true,
            "responses_table": "responses",
            "conversations_table": "conversations",
            "ssl_mode": "verify-full",
            "ssl_root_cert": "/etc/pki/ca.pem",
            "ssl_client_cert": "/etc/pki/client.pem",
            "ssl_client_key": "/etc/pki/client.key",
            "require_certificate_authentication": true,
        });
        PostgresBackendFactory
            .validate_config(&cfg)
            .expect("a client-cert config the filter accepts must validate");
    }

    /// A client cert under a non-verifying `ssl_mode` must be rejected, matching
    /// the filter's fail-closed check: `require` encrypts but does not verify.
    #[test]
    fn validate_rejects_client_cert_with_unverified_ssl_mode() {
        let cfg = json!({
            "database_url": "postgres://svc@db.example.com:5432/app",
            "allow_private_database_url": true,
            "responses_table": "responses",
            "conversations_table": "conversations",
            "ssl_mode": "require",
            "ssl_client_cert": "/etc/pki/client.pem",
            "ssl_client_key": "/etc/pki/client.key",
            "require_certificate_authentication": true,
        });
        let err = PostgresBackendFactory
            .validate_config(&cfg)
            .expect_err("a client cert under an unverified ssl_mode must be rejected")
            .to_string();
        // Assert the ssl_mode reason, so allowing the DNS host cannot let this
        // pass for the wrong reason.
        assert!(err.contains("verify-ca") && err.contains("verify-full"), "got: {err}");
    }

    /// A different client-certificate path is a different connection identity,
    /// so it must change the dedup key rather than collide onto one pool.
    #[test]
    fn client_cert_path_changes_the_effective_key() {
        let base = json!({
            "database_url": "postgres://svc@db.example.com:5432/app",
            "responses_table": "responses",
            "conversations_table": "conversations",
            "ssl_mode": "verify-full",
            "ssl_client_cert": "/etc/pki/client-a.pem",
            "ssl_client_key": "/etc/pki/client.key",
        });
        let mut other = base.clone();
        other
            .as_object_mut()
            .expect("object")
            .insert("ssl_client_cert".to_owned(), json!("/etc/pki/client-b.pem"));

        let k1 = PostgresBackendFactory.effective_key(&base).expect("base key");
        let k2 = PostgresBackendFactory.effective_key(&other).expect("other-cert key");
        assert_ne!(k1, k2, "a different client-certificate path must change the dedup key");
    }

    /// Certificate authentication changes the connection identity, so it must
    /// change the dedup key.
    #[test]
    fn cert_authentication_changes_the_effective_key() {
        let base = json!({
            "database_url": "postgres://svc@db.example.com:5432/app",
            "responses_table": "responses",
            "conversations_table": "conversations",
        });
        let mut with_cert = base.clone();
        with_cert
            .as_object_mut()
            .expect("object")
            .insert("require_certificate_authentication".to_owned(), json!(true));

        let k1 = PostgresBackendFactory.effective_key(&base).expect("base key");
        let k2 = PostgresBackendFactory.effective_key(&with_cert).expect("cert-auth key");
        assert_ne!(k1, k2, "certificate authentication must change the dedup key");
    }
}
