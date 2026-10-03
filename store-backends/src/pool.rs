// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Apply the SQL-free [`PoolConfig`] to a sqlx [`PoolOptions`].
//!
//! The config type and its validation live in the backend-free `praxis-ai-store`
//! crate. This module keeps only the sqlx-dependent conversion.

use praxis_ai_store::PoolConfig;
use sqlx::{Database, pool::PoolOptions};

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
    use std::time::Duration;

    use praxis_ai_store::{DEFAULT_MAX_CONNECTIONS, PoolConfig};

    // The min-vs-implicit-max check hinges on `DEFAULT_MAX_CONNECTIONS`
    // equaling sqlx's actual default. sqlx documents that value only as
    // "see the source", so a future bump could change it silently and
    // re-open the issue this validation closes. Pin it here, using
    // whichever driver is compiled (both share the generic default).
    #[cfg(any(feature = "postgres", feature = "sqlite"))]
    #[test]
    fn default_max_connections_matches_sqlx_default() {
        #[cfg(feature = "postgres")]
        let sqlx_default = sqlx::postgres::PgPoolOptions::new().get_max_connections();
        #[cfg(all(feature = "sqlite", not(feature = "postgres")))]
        let sqlx_default = sqlx::sqlite::SqlitePoolOptions::new().get_max_connections();
        assert_eq!(
            sqlx_default, DEFAULT_MAX_CONNECTIONS,
            "sqlx default max_connections drifted from DEFAULT_MAX_CONNECTIONS; re-verify the issue #1253 fix"
        );
    }

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
    fn validate_rejects_min_exceeds_implicit_default_max() {
        // Regression: with no explicit max, the effective maximum stays
        // at the sqlx default of 10, so a larger minimum is unreachable.
        let cfg = PoolConfig {
            min_connections: Some(20),
            ..PoolConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("min_connections"), "{err}");
        assert!(err.contains("default"), "{err}");
        assert!(err.contains(&DEFAULT_MAX_CONNECTIONS.to_string()), "{err}");
    }

    #[test]
    fn validate_accepts_min_equals_implicit_default_max() {
        let cfg = PoolConfig {
            min_connections: Some(DEFAULT_MAX_CONNECTIONS),
            ..PoolConfig::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "min_connections equal to the implicit default max ({DEFAULT_MAX_CONNECTIONS}) should be accepted"
        );
    }

    #[test]
    fn validate_accepts_min_below_implicit_default_max() {
        let cfg = PoolConfig {
            min_connections: Some(5),
            ..PoolConfig::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "min_connections below the implicit default max ({DEFAULT_MAX_CONNECTIONS}) should be accepted"
        );
    }

    #[test]
    fn validate_accepts_min_above_default_when_max_explicit() {
        // An explicit max above the default lifts the ceiling, so a
        // minimum that would be rejected against the default is fine.
        let cfg = PoolConfig {
            min_connections: Some(20),
            max_connections: Some(30),
            ..PoolConfig::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "min_connections above the default is valid when an explicit larger max is set"
        );
    }

    #[test]
    fn denies_unknown_fields() {
        let yaml = "max_connections: 5\nbogus: true\n";
        let result: Result<PoolConfig, _> = serde_yaml::from_str(yaml);
        assert!(result.is_err(), "unknown fields should be rejected");
    }
}
