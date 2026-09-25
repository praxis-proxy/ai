// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Backend factory interface.
//!
//! How a persisted-state backend is built from inline configuration, plus the
//! errors, capability descriptor, dedup key, and retirement hook that govern
//! provisioning. SQL-free: a concrete factory owns its config type and any SQL
//! or cryptography, so this interface and its consumers stay backend-agnostic.

use std::{fmt, sync::Arc};

use async_trait::async_trait;

use crate::traits::PersistedStateBackend;

/// Canonical dedup key for an effective backend configuration.
///
/// Two references that produce an equal key share one backend and one pool. The
/// factory computes it from the normalized connection-determining fields
/// (backend id, database url, TLS mode, certificate paths, pool parameters). It
/// must not embed secret material in a form that leaks through [`fmt::Debug`].
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct EffectiveConfigKey(Arc<str>);

impl EffectiveConfigKey {
    /// Wrap a factory-computed canonical key.
    #[must_use]
    pub fn new(key: impl Into<Arc<str>>) -> Self {
        Self(key.into())
    }
}

impl fmt::Debug for EffectiveConfigKey {
    /// Print only the wrapper so a key carrying a connection string never
    /// widens into a log line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("EffectiveConfigKey(..)")
    }
}

/// Why a backend could not be provisioned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    /// Configuration is invalid; a retry will not help. Fails the build.
    Config(String),
    /// The backend is unreachable in a way a retry will not resolve (for
    /// example a permanent initialization failure). Fails the build with an
    /// actionable backend-unavailable error rather than a first-traffic error.
    Unavailable(String),
    /// A transient failure such as a connect timeout. Provisioning may retry
    /// within a bounded budget before giving up as [`BackendError::Unavailable`].
    Transient(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "backend configuration error: {m}"),
            Self::Unavailable(m) => write!(f, "backend unavailable: {m}"),
            Self::Transient(m) => write!(f, "backend transiently unavailable: {m}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Explicit retirement of a backend's resources, such as closing a pool.
#[async_trait]
pub trait RetireBackend: Send + Sync {
    /// Release the backend's resources.
    ///
    /// Called once, after in-flight requests drain, when the backend is evicted
    /// from the process-wide cache. The default releases nothing, which suits an
    /// in-memory backend.
    async fn retire(&self) {}
}

/// A built, validated backend and the hook that retires its resources.
pub struct ProvisionedBackend {
    /// Combined backend handle registered for request-path use.
    pub backend: Arc<dyn PersistedStateBackend>,
    /// Retirement hook invoked on eviction.
    pub retire: Arc<dyn RetireBackend>,
}

/// Builds persisted-state backends of one kind from inline configuration.
///
/// Binaries inject concrete factories; the lifecycle layer routes a store
/// reference's config to a factory by [`StoreBackendFactory::backend_id`]. The
/// factory owns its config type and any SQL or cryptography, keeping this
/// interface and the lifecycle crate backend-agnostic and crypto-free.
#[async_trait]
pub trait StoreBackendFactory: Send + Sync {
    /// Stable identifier this factory provisions, for example `"sqlite"`.
    fn backend_id(&self) -> &str;

    /// Compute the dedup key from config without I/O.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Config`] when the configuration is malformed.
    fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError>;

    /// Validate configuration without I/O, so a malformed config fails at
    /// pipeline construction rather than at first traffic. The default checks
    /// the config parses well enough to compute the dedup key; a factory may
    /// override to validate more. Connectivity is not checked here (no runtime).
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Config`] when the configuration is malformed.
    fn validate_config(&self, config: &serde_json::Value) -> Result<(), BackendError> {
        self.effective_key(config).map(|_| ())
    }

    /// Build and eagerly validate a backend: open the pool and verify the
    /// schema, so an unusable backend fails here rather than on first traffic.
    ///
    /// # Errors
    ///
    /// [`BackendError::Config`] and [`BackendError::Unavailable`] fail the build;
    /// [`BackendError::Transient`] invites a bounded retry from the caller.
    async fn build(&self, config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError>;
}
