// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared config validation helpers for Responses API filters.

use praxis_core::callout::{CalloutConfig, CircuitBreakerConfig, FailureMode as CoreFailureMode};
use praxis_filter::FilterError;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// FailureMode
// -----------------------------------------------------------------------------

/// What happens when a callout to an external service fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureMode {
    /// Reject the request on failure (default).
    Closed,
    /// Continue without the callout result on failure.
    Open,
}

// -----------------------------------------------------------------------------
// CalloutSettings
// -----------------------------------------------------------------------------

/// Common callout fields shared by filters that make HTTP callouts.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CalloutSettings {
    /// Callout timeout in milliseconds.
    pub timeout_ms: u64,
    /// Failure mode for the callout.
    pub failure_mode: FailureMode,
    /// HTTP status code to return when rejecting on error.
    pub status_on_error: u16,
}

impl CalloutSettings {
    /// Build a [`CalloutConfig`] from these settings.
    pub(crate) fn build_callout_config(self) -> CalloutConfig {
        let failure_mode = match self.failure_mode {
            FailureMode::Closed => CoreFailureMode::Closed,
            FailureMode::Open => CoreFailureMode::Open,
        };
        CalloutConfig {
            circuit_breaker: Some(CircuitBreakerConfig {
                consecutive_failures: 5,
                recovery_window_ms: 30_000,
            }),
            failure_mode,
            status_on_error: self.status_on_error,
            timeout_ms: self.timeout_ms,
            ..CalloutConfig::default()
        }
    }
}

// -----------------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------------

/// Validate `timeout_ms`, applying a default and rejecting zero.
///
/// # Errors
///
/// Returns [`FilterError`] when the resolved value is zero.
pub(crate) fn validate_timeout_ms(
    filter: &str,
    raw: Option<u64>,
    default: u64,
) -> Result<u64, FilterError> {
    let value = raw.unwrap_or(default);
    if value == 0 {
        return Err(format!("{filter}: timeout_ms must be greater than 0").into());
    }
    Ok(value)
}

/// Validate `status_on_error`, applying a default and rejecting
/// values outside the HTTP status range.
///
/// # Errors
///
/// Returns [`FilterError`] when the resolved value is not in
/// `100..=599`.
pub(crate) fn validate_status_on_error(
    filter: &str,
    raw: Option<u16>,
    default: u16,
) -> Result<u16, FilterError> {
    let value = raw.unwrap_or(default);
    if !(100..=599).contains(&value) {
        return Err(
            format!("{filter}: status_on_error must be between 100 and 599, got {value}").into(),
        );
    }
    Ok(value)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn timeout_applies_default() {
        assert_eq!(validate_timeout_ms("test", None, 5000).unwrap(), 5000);
    }

    #[test]
    fn timeout_uses_provided_value() {
        assert_eq!(
            validate_timeout_ms("test", Some(10_000), 5000).unwrap(),
            10_000
        );
    }

    #[test]
    fn timeout_zero_rejected() {
        let err = validate_timeout_ms("test", Some(0), 5000).unwrap_err();
        assert!(
            err.to_string().contains("greater than 0"),
            "zero should be rejected, got: {err}"
        );
    }

    #[test]
    fn timeout_error_includes_filter_name() {
        let err = validate_timeout_ms("my_filter", Some(0), 5000).unwrap_err();
        assert!(
            err.to_string().contains("my_filter"),
            "error should include filter name, got: {err}"
        );
    }

    #[test]
    fn status_applies_default() {
        assert_eq!(
            validate_status_on_error("test", None, 502).unwrap(),
            502
        );
    }

    #[test]
    fn status_uses_provided_value() {
        assert_eq!(
            validate_status_on_error("test", Some(503), 502).unwrap(),
            503
        );
    }

    #[test]
    fn status_below_range_rejected() {
        let err = validate_status_on_error("test", Some(99), 502).unwrap_err();
        assert!(
            err.to_string().contains("between 100 and 599"),
            "below range should be rejected, got: {err}"
        );
    }

    #[test]
    fn status_above_range_rejected() {
        let err = validate_status_on_error("test", Some(600), 502).unwrap_err();
        assert!(
            err.to_string().contains("between 100 and 599"),
            "above range should be rejected, got: {err}"
        );
    }

    #[test]
    fn status_boundaries_accepted() {
        validate_status_on_error("test", Some(100), 502).expect("100 should be accepted");
        validate_status_on_error("test", Some(599), 502).expect("599 should be accepted");
    }

    #[test]
    fn status_error_includes_filter_name() {
        let err = validate_status_on_error("my_filter", Some(0), 502).unwrap_err();
        assert!(
            err.to_string().contains("my_filter"),
            "error should include filter name, got: {err}"
        );
    }
}
