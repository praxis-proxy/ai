// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! TTL and background reaper configuration for response stores.

use std::{num::NonZeroU32, time::Duration};

use serde::Deserialize;

/// Default sweep interval in seconds (1 hour).
pub const DEFAULT_SWEEP_INTERVAL_SECS: u64 = 3600;

/// Default batch size for reaping expired records.
pub const DEFAULT_SWEEP_BATCH_SIZE: NonZeroU32 = match NonZeroU32::new(1000) {
    Some(size) => size,
    // 1000 is non-zero, so this arm is unreachable.
    None => NonZeroU32::MIN,
};

/// TTL and reaper sweep configuration.
///
/// When `ttl` is set, expired records are deleted in periodic background sweeps.
/// When `ttl` is unset, the reaper is disabled and records persist indefinitely.
///
/// # YAML
///
/// The fields are flat, top-level filter keys (there is no `reaper:` block):
///
/// ```yaml
/// ttl: 7d
/// sweep_interval: 1h
/// sweep_batch_size: 1000
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReaperConfig {
    /// Time-to-live for stored records. Records older than this are eligible
    /// for deletion. When unset, the reaper is disabled.
    #[serde(default, with = "humantime_serde")]
    pub ttl: Option<Duration>,

    /// How often to run the reaper sweep. Defaults to 1 hour.
    #[serde(default, with = "humantime_serde")]
    pub sweep_interval: Option<Duration>,

    /// Maximum number of records to delete in a single sweep batch.
    #[serde(default)]
    pub sweep_batch_size: Option<NonZeroU32>,
}

impl ReaperConfig {
    /// Validate configuration constraints.
    ///
    /// # Errors
    ///
    /// Returns a message describing the invalid field.
    pub fn validate(&self) -> Result<(), String> {
        // The reaper fields are flat, top-level filter keys (`ttl`,
        // `sweep_interval`, `sweep_batch_size`) with no `reaper.` block, so the
        // messages name the keys the operator actually wrote; the caller wraps
        // them with the `{FILTER_NAME}:` prefix.
        if self.ttl == Some(Duration::ZERO) {
            return Err("ttl must be greater than zero".into());
        }
        if self.ttl.is_none() && self.sweep_interval.is_some() {
            return Err("sweep_interval requires ttl to be set".into());
        }
        if self.sweep_interval == Some(Duration::ZERO) {
            return Err("sweep_interval must be greater than zero".into());
        }
        Ok(())
    }

    /// Build a reaper policy if TTL is configured.
    ///
    /// Returns `None` when `ttl` is unset, indicating the reaper is disabled.
    /// Returns `Some(policy)` with defaults applied for unset sweep parameters.
    #[must_use]
    pub fn policy(&self) -> Option<ReaperPolicy> {
        self.ttl.map(|ttl| ReaperPolicy {
            ttl,
            sweep_interval: self
                .sweep_interval
                .unwrap_or_else(|| Duration::from_secs(DEFAULT_SWEEP_INTERVAL_SECS)),
            batch_size: self.sweep_batch_size.unwrap_or(DEFAULT_SWEEP_BATCH_SIZE),
        })
    }
}

/// Resolved reaper policy with all defaults applied.
///
/// This is always created from a [`ReaperConfig`] with `ttl` set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReaperPolicy {
    /// Time-to-live for stored records.
    pub ttl: Duration,

    /// How often to run the reaper sweep.
    pub sweep_interval: Duration,

    /// Maximum number of records to delete in a single sweep batch.
    pub batch_size: NonZeroU32,
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, clippy::expect_used, reason = "test fixtures")]

    use super::*;

    #[test]
    fn parses_human_durations_and_builds_policy() {
        let cfg: ReaperConfig = serde_yaml::from_str("ttl: 7d\nsweep_interval: 1h\n").unwrap();
        let p = cfg.policy().expect("ttl set => policy");
        assert_eq!(p.ttl, Duration::from_secs(7 * 24 * 3600));
        assert_eq!(p.sweep_interval, Duration::from_secs(3600));
        assert_eq!(p.batch_size.get(), 1000); // default
    }

    #[test]
    fn no_ttl_means_no_policy() {
        let cfg: ReaperConfig = serde_yaml::from_str("sweep_interval: 1h\n").unwrap();
        assert!(cfg.policy().is_none());
    }

    #[test]
    fn validate_rejects_zero_ttl_and_sweep_interval_without_ttl_and_zero_batch() {
        let zero_ttl = ReaperConfig {
            ttl: Some(Duration::ZERO),
            ..Default::default()
        }
        .validate()
        .expect_err("zero ttl must be rejected");
        assert_eq!(zero_ttl, "ttl must be greater than zero");

        let sweep_without_ttl = ReaperConfig {
            ttl: None,
            sweep_interval: Some(Duration::from_secs(60)),
            ..Default::default()
        }
        .validate()
        .expect_err("sweep_interval without ttl must be rejected");
        assert_eq!(sweep_without_ttl, "sweep_interval requires ttl to be set");

        // Messages name the flat top-level keys operators actually write, not a
        // nonexistent `reaper.` block.
        assert!(!zero_ttl.contains("reaper."));
        assert!(!sweep_without_ttl.contains("reaper."));

        // sweep_batch_size: 0 fails at deserialize (NonZeroU32); assert the serde error:
        assert!(serde_yaml::from_str::<ReaperConfig>("ttl: 1d\nsweep_batch_size: 0\n").is_err());
    }

    #[test]
    fn rejects_unparseable_duration() {
        assert!(serde_yaml::from_str::<ReaperConfig>("ttl: banana\n").is_err());
    }
}
