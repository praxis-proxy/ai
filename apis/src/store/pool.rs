// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Connection pool tuning configuration shared by all store backends.

use std::time::Duration;

use serde::Deserialize;
use sqlx::{Database, pool::PoolOptions};

/// Connection pool tuning options for store backends.
///
/// All fields are optional. When omitted, the sqlx defaults apply:
/// `max_connections = 10`, `min_connections = 0`,
/// `idle_timeout = 600s (10 min)`, `acquire_timeout = 30s`.
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
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.max_connections == Some(0) {
            return Err("pool.max_connections must be at least 1".into());
        }
        if self.acquire_timeout_secs == Some(0) {
            return Err(
                "pool.acquire_timeout_secs must be at least 1 (use idle_timeout_secs: 0 to disable idle timeout)"
                    .into(),
            );
        }
        if let (Some(min), Some(max)) = (self.min_connections, self.max_connections)
            && min > max
        {
            return Err(format!(
                "pool.min_connections ({min}) must not exceed pool.max_connections ({max})"
            ));
        }
        Ok(())
    }

    /// Convert `idle_timeout_secs` to a [`Duration`], treating `0`
    /// as "no timeout" (returns `None`).
    #[expect(clippy::option_option, reason = "three-valued: unset / disabled(0) / duration")]
    pub(crate) fn idle_timeout(&self) -> Option<Option<Duration>> {
        self.idle_timeout_secs.map(|secs| {
            if secs == 0 {
                None
            } else {
                Some(Duration::from_secs(secs))
            }
        })
    }

    /// Convert `acquire_timeout_secs` to a [`Duration`].
    pub(crate) fn acquire_timeout(&self) -> Option<Duration> {
        self.acquire_timeout_secs.map(Duration::from_secs)
    }
}

/// Apply user-supplied pool tuning to a [`PoolOptions`].
pub(crate) fn apply_pool_config<DB: Database>(
    mut opts: PoolOptions<DB>,
    pool_config: Option<&PoolConfig>,
) -> PoolOptions<DB> {
    let Some(cfg) = pool_config else {
        return opts;
    };
    if let Some(max) = cfg.max_connections {
        opts = opts.max_connections(max);
    }
    if let Some(min) = cfg.min_connections {
        opts = opts.min_connections(min);
    }
    if let Some(timeout) = cfg.idle_timeout() {
        opts = opts.idle_timeout(timeout);
    }
    if let Some(timeout) = cfg.acquire_timeout() {
        opts = opts.acquire_timeout(timeout);
    }
    opts
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn default_pool_config_has_no_overrides() {
        let cfg = PoolConfig::default();
        assert!(cfg.max_connections.is_none());
        assert!(cfg.min_connections.is_none());
        assert!(cfg.idle_timeout_secs.is_none());
        assert!(cfg.acquire_timeout_secs.is_none());
    }

    #[test]
    fn deserialize_all_fields() {
        let yaml = "
max_connections: 20
min_connections: 2
idle_timeout_secs: 600
acquire_timeout_secs: 30
";
        let cfg: PoolConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.max_connections, Some(20));
        assert_eq!(cfg.min_connections, Some(2));
        assert_eq!(cfg.idle_timeout_secs, Some(600));
        assert_eq!(cfg.acquire_timeout_secs, Some(30));
    }

    #[test]
    fn deserialize_partial_fields() {
        let yaml = "max_connections: 5\n";
        let cfg: PoolConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.max_connections, Some(5));
        assert!(cfg.min_connections.is_none());
        assert!(cfg.idle_timeout_secs.is_none());
        assert!(cfg.acquire_timeout_secs.is_none());
    }

    #[test]
    fn deserialize_empty_yields_defaults() {
        let cfg: PoolConfig = serde_yaml::from_str("{}").unwrap();
        assert!(cfg.max_connections.is_none());
    }

    #[test]
    fn idle_timeout_zero_means_no_timeout() {
        let cfg = PoolConfig {
            idle_timeout_secs: Some(0),
            ..PoolConfig::default()
        };
        assert_eq!(cfg.idle_timeout(), Some(None));
    }

    #[test]
    fn idle_timeout_nonzero_returns_duration() {
        let cfg = PoolConfig {
            idle_timeout_secs: Some(300),
            ..PoolConfig::default()
        };
        assert_eq!(cfg.idle_timeout(), Some(Some(Duration::from_secs(300))));
    }

    #[test]
    fn idle_timeout_none_means_use_default() {
        let cfg = PoolConfig::default();
        assert!(cfg.idle_timeout().is_none());
    }

    #[test]
    fn acquire_timeout_returns_duration() {
        let cfg = PoolConfig {
            acquire_timeout_secs: Some(60),
            ..PoolConfig::default()
        };
        assert_eq!(cfg.acquire_timeout(), Some(Duration::from_secs(60)));
    }

    #[test]
    fn validate_rejects_zero_max_connections() {
        let cfg = PoolConfig {
            max_connections: Some(0),
            ..PoolConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_connections"), "{err}");
    }

    #[test]
    fn validate_accepts_positive_max_connections() {
        let cfg = PoolConfig {
            max_connections: Some(1),
            ..PoolConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_omitted_max_connections() {
        let cfg = PoolConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_acquire_timeout() {
        let cfg = PoolConfig {
            acquire_timeout_secs: Some(0),
            ..PoolConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("acquire_timeout_secs"), "{err}");
    }

    #[test]
    fn validate_accepts_positive_acquire_timeout() {
        let cfg = PoolConfig {
            acquire_timeout_secs: Some(30),
            ..PoolConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_min_exceeds_max_connections() {
        let cfg = PoolConfig {
            min_connections: Some(20),
            max_connections: Some(10),
            ..PoolConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("min_connections"), "{err}");
        assert!(err.contains("max_connections"), "{err}");
    }

    #[test]
    fn validate_accepts_min_equals_max_connections() {
        let cfg = PoolConfig {
            min_connections: Some(5),
            max_connections: Some(5),
            ..PoolConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_min_less_than_max_connections() {
        let cfg = PoolConfig {
            min_connections: Some(2),
            max_connections: Some(10),
            ..PoolConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn denies_unknown_fields() {
        let yaml = "max_connections: 5\nbogus: true\n";
        let result: Result<PoolConfig, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "unknown fields should be rejected");
    }
}
