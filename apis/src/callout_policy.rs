// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared failure-policy vocabulary for outbound AI callouts.
//!
//! # Classification is filter-specific
//!
//! These enums are a vocabulary: they fix the accepted values
//! and the default, not which conditions a filter routes through
//! which key. Each filter's `on_failure` / `on_missing` field docs
//! and behavior are authoritative.
//!
//! # Naming
//!
//! The external keys are `on_failure` and `on_missing`. A structural
//! `failure_mode` key is already owned by Core's pipeline entries.

use praxis_filter::FilterError;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// OnFailure
// -----------------------------------------------------------------------------

/// What happens when an outbound callout does not produce a usable
/// answer. Configured as `on_failure`.
///
/// For a callout that succeeds but reports an absent resource, use
/// [`OnMissing`].
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OnFailure {
    /// Reject the request on failure (default).
    #[default]
    Closed,

    /// Continue without the callout result on failure.
    Open,
}

// -----------------------------------------------------------------------------
// OnMissing
// -----------------------------------------------------------------------------

/// What happens when a requested resource cannot be fetched. Configured
/// as `on_missing`.
///
/// A filter may narrow the set of resources this governs, but must never
/// widen it to cover failures that carry a security signal (e.g. a file
/// URL that cannot be resolved - the target may be malicious or unreachable
/// for policy reasons).
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OnMissing {
    /// Continue without the resource.
    #[default]
    Continue,

    /// Return an error response to the client.
    Reject,
}

// -----------------------------------------------------------------------------
// CalloutSettings
// -----------------------------------------------------------------------------

/// Common callout fields shared by filters that make HTTP callouts.
#[derive(Debug, Clone, Copy)]
pub struct CalloutSettings {
    /// Callout timeout in milliseconds.
    pub timeout_ms: u64,

    /// Failure mode for the callout.
    pub on_failure: OnFailure,

    /// HTTP status code to return when rejecting on error.
    pub status_on_error: u16,
}

// -----------------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------------

/// Validate `timeout_ms`, applying a default and rejecting zero.
///
/// # Errors
///
/// Returns [`FilterError`] when the resolved value is zero.
pub fn validate_timeout_ms(filter: &str, raw: Option<u64>, default: u64) -> Result<u64, FilterError> {
    let value = raw.unwrap_or(default);
    if value == 0 {
        return Err(format!("{filter}: timeout_ms must be greater than 0").into());
    }
    Ok(value)
}

/// Validate `status_on_error`, applying a default and rejecting
/// values that are not HTTP error status codes (`400..=599`).
///
/// # Errors
///
/// Returns [`FilterError`] when the resolved value is not in
/// `400..=599`.
pub fn validate_status_on_error(filter: &str, raw: Option<u16>, default: u16) -> Result<u16, FilterError> {
    let value = raw.unwrap_or(default);
    if !(400..=599).contains(&value) {
        return Err(format!("{filter}: status_on_error must be between 400 and 599, got {value}").into());
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
        assert_eq!(validate_timeout_ms("test", Some(10_000), 5000).unwrap(), 10_000);
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
        assert_eq!(validate_status_on_error("test", None, 502).unwrap(), 502);
    }

    #[test]
    fn status_uses_provided_value() {
        assert_eq!(validate_status_on_error("test", Some(503), 502).unwrap(), 503);
    }

    #[test]
    fn status_below_range_rejected() {
        let err = validate_status_on_error("test", Some(99), 502).unwrap_err();
        assert!(
            err.to_string().contains("between 400 and 599"),
            "below range should be rejected, got: {err}"
        );
    }

    #[test]
    fn status_above_range_rejected() {
        let err = validate_status_on_error("test", Some(600), 502).unwrap_err();
        assert!(
            err.to_string().contains("between 400 and 599"),
            "above range should be rejected, got: {err}"
        );
    }

    #[test]
    fn status_non_error_classes_rejected() {
        for status in [100, 101, 200, 204, 301, 399] {
            let err = validate_status_on_error("test", Some(status), 502).unwrap_err();
            assert!(
                err.to_string().contains("between 400 and 599"),
                "non-error status {status} should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn status_boundaries_accepted() {
        validate_status_on_error("test", Some(400), 502).expect("400 should be accepted");
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

    // -------------------------------------------------------------------------
    // Canonical vocabulary
    // -------------------------------------------------------------------------

    #[test]
    fn on_failure_deserializes_canonical_values() {
        assert_eq!(serde_yaml::from_str::<OnFailure>("closed").unwrap(), OnFailure::Closed);
        assert_eq!(serde_yaml::from_str::<OnFailure>("open").unwrap(), OnFailure::Open);
    }

    #[test]
    fn on_failure_defaults_to_closed() {
        assert_eq!(OnFailure::default(), OnFailure::Closed);
    }

    #[test]
    fn on_missing_defaults_to_continue() {
        assert_eq!(OnMissing::default(), OnMissing::Continue);
    }

    #[test]
    fn on_missing_deserializes_canonical_values() {
        assert_eq!(
            serde_yaml::from_str::<OnMissing>("continue").unwrap(),
            OnMissing::Continue
        );
        assert_eq!(serde_yaml::from_str::<OnMissing>("reject").unwrap(), OnMissing::Reject);
    }

    #[test]
    fn on_failure_rejects_on_missing_vocabulary() {
        assert!(serde_yaml::from_str::<OnFailure>("continue").is_err());
        assert!(serde_yaml::from_str::<OnFailure>("reject").is_err());
        assert!(serde_yaml::from_str::<OnMissing>("open").is_err());
        assert!(serde_yaml::from_str::<OnMissing>("closed").is_err());
    }
}
