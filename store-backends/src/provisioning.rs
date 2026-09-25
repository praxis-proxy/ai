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

#[cfg(any(feature = "sqlite", feature = "postgres"))]
use praxis_ai_store::BackendError;

#[cfg(any(feature = "sqlite", feature = "postgres"))]
use super::redact_connection_error;

/// A permanent build failure, redacted, as a backend-unavailable error.
#[cfg(feature = "sqlite")]
fn permanent(url: &str, message: &str) -> BackendError {
    BackendError::Unavailable(redact_connection_error(url, message))
}

/// A transient build failure, redacted, so provisioning retries within budget.
#[cfg(feature = "postgres")]
fn transient(url: &str, message: &str) -> BackendError {
    BackendError::Transient(redact_connection_error(url, message))
}

/// A stable fingerprint of the pool overrides for the dedup key.
#[cfg(any(feature = "sqlite", feature = "postgres"))]
fn pool_fingerprint(pool: Option<&crate::PoolConfig>) -> String {
    pool.map_or_else(
        || "default".to_owned(),
        |p| {
            format!(
                "{:?}/{:?}/{:?}/{:?}",
                p.max_connections, p.min_connections, p.idle_timeout_secs, p.acquire_timeout_secs
            )
        },
    )
}

/// A stable fingerprint of the compression override for the dedup key.
#[cfg(any(feature = "sqlite", feature = "postgres"))]
fn compression_fingerprint(compression: Option<&praxis_ai_store::StoreCompressionConfig>) -> String {
    compression.map_or_else(|| "none".to_owned(), |c| format!("{:?}/{:?}", c.algorithm, c.level))
}

/// SQLite-backed store-backend factory.
#[cfg(feature = "sqlite")]
mod sqlite {
    use std::sync::Arc;

    use async_trait::async_trait;
    use praxis_ai_store::{
        EffectiveConfigKey, PoolConfig, ProvisionedBackend, RetireBackend, StoreBackendFactory, StoreCompressionConfig,
    };
    use secrecy::{ExposeSecret as _, SecretString};
    use serde::Deserialize;
    use serde_json::Value;

    use super::{BackendError, permanent};
    use crate::SqliteResponseStore;

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
            // The pool + url + table names identify one SQLite store instance.
            let key = format!(
                "sqlite\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
                cfg.database_url.expose_secret(),
                cfg.responses_table,
                cfg.conversations_table,
                cfg.items_table.as_deref().unwrap_or(""),
                super::pool_fingerprint(cfg.pool.as_ref()),
                super::compression_fingerprint(cfg.compression.as_ref()),
            );
            Ok(EffectiveConfigKey::new(key))
        }

        async fn build(&self, config: &Value) -> Result<ProvisionedBackend, BackendError> {
            let cfg = Self::parse(config)?;
            let url = cfg.database_url.expose_secret();
            // SQLite init failure is permanent: fail the build unavailable.
            let store = SqliteResponseStore::new(
                url,
                &cfg.responses_table,
                &cfg.conversations_table,
                cfg.items_table.as_deref(),
                cfg.pool.as_ref(),
                cfg.compression.as_ref(),
            )
            .await
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
#[cfg(feature = "postgres")]
mod postgres {
    use std::sync::Arc;

    use async_trait::async_trait;
    use praxis_ai_store::{
        EffectiveConfigKey, PoolConfig, ProvisionedBackend, RetireBackend, SslMode, StoreBackendFactory,
        StoreCompressionConfig,
    };
    use secrecy::{ExposeSecret as _, SecretString};
    use serde::Deserialize;
    use serde_json::Value;

    use super::{BackendError, transient};
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
            Box::pin(PostgresResponseStore::new(
                url,
                &cfg.responses_table,
                &cfg.conversations_table,
                cfg.items_table.as_deref(),
                &tls,
                cfg.pool.as_ref(),
                cfg.compression.as_ref(),
            ))
            .await
            .map_err(|e| transient(url, &e.to_string()))
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
                cfg.ssl_mode,
                cfg.ssl_root_cert.as_ref().map_or("", |s| s.expose_secret()),
                cfg.ssl_client_cert.as_ref().map_or("", |s| s.expose_secret()),
                cfg.ssl_client_key.as_ref().map_or("", |s| s.expose_secret()),
                cfg.require_certificate_authentication,
                super::pool_fingerprint(cfg.pool.as_ref()),
                super::compression_fingerprint(cfg.compression.as_ref()),
            );
            Ok(EffectiveConfigKey::new(key))
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
#[cfg(any(feature = "sqlite", feature = "postgres"))]
#[must_use]
#[expect(clippy::vec_init_then_push, reason = "each push is feature-gated")]
pub fn store_backend_factories() -> Vec<std::sync::Arc<dyn praxis_ai_store::StoreBackendFactory>> {
    let mut factories: Vec<std::sync::Arc<dyn praxis_ai_store::StoreBackendFactory>> = Vec::new();
    #[cfg(feature = "sqlite")]
    factories.push(std::sync::Arc::new(sqlite::SqliteBackendFactory));
    #[cfg(feature = "postgres")]
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

    use super::sqlite::SqliteBackendFactory;

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
#[cfg(feature = "postgres")]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "tests")]
mod postgres_tests {
    use praxis_ai_store::StoreBackendFactory as _;
    use serde_json::json;

    use super::postgres::PostgresBackendFactory;

    /// A client-cert mTLS config the response-store filter accepts must also
    /// pass factory validation: the factory runs the same TLS check.
    ///
    /// `require_certificate_authentication` is left off so the compliance
    /// profile's `PGPASSWORD`-env read does not make this env-dependent.
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
