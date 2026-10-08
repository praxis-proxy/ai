// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Result types and the `NeMo` provider for external AI guardrails.

pub(super) mod nemo;

use std::{fmt, sync::Arc, time::Instant};

use praxis_filter::SubrequestRuntime;

// -----------------------------------------------------------------------------
// GuardPhase
// -----------------------------------------------------------------------------

/// Which phase of the proxy pipeline is being evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardPhase {
    /// Inspecting the client request before it reaches the upstream.
    Request,
    /// Inspecting the upstream response before it reaches the client.
    Response,
}

impl GuardPhase {
    /// Returns a static label for logging and diagnostics.
    pub fn label(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Response => "response",
        }
    }
}

impl fmt::Display for GuardPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// -----------------------------------------------------------------------------
// GuardResult
// -----------------------------------------------------------------------------

/// Normalized verdict from an external guardrail provider evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardResult {
    /// Content is safe — forward unchanged.
    Pass,
    /// Content violates policy — reject with reason.
    Block {
        /// Human-readable block reason from the provider.
        reason: String,
    },
    /// Content contains sensitive data — forward with masked text.
    ///
    /// `modified_text` is populated from `NeMo` but not applied to the request
    /// or response body until redaction support lands in `#49`.
    Redact {
        /// Provider-rewritten text with sensitive data masked.
        modified_text: String,
        /// Human-readable redaction reason from the provider.
        reason: String,
    },
}

impl GuardResult {
    /// Returns the status label written to [`FilterResultSet`].
    ///
    /// [`FilterResultSet`]: praxis_filter::FilterResultSet
    pub fn status_label(&self) -> &'static str {
        match self {
            Self::Pass => "passed",
            Self::Block { .. } => "blocked",
            Self::Redact { .. } => "redacted",
        }
    }
}

// -----------------------------------------------------------------------------
// GuardCalloutRuntime
// -----------------------------------------------------------------------------

/// Downstream attributes and sub-request depth for one guardrail callout.
#[derive(Clone)]
pub(crate) struct GuardCalloutRuntime<'a> {
    /// Originating client attributes forwarded into the outbound chain.
    pub downstream: SubrequestRuntime,
    /// Current filtered-subrequest nesting depth.
    pub depth: u8,
    /// Absolute deadline covering target preparation and the outbound exchange.
    pub deadline: Instant,
    /// Outbound filter chain bound to the guardrails filter.
    pub outbound: &'a Arc<praxis_filter::FilterPipeline>,
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_phase_display() {
        assert_eq!(GuardPhase::Request.to_string(), "request");
        assert_eq!(GuardPhase::Response.to_string(), "response");
    }

    #[test]
    fn guard_result_status_labels() {
        assert_eq!(GuardResult::Pass.status_label(), "passed");
        assert_eq!(GuardResult::Block { reason: "test".into() }.status_label(), "blocked");
        assert_eq!(
            GuardResult::Redact {
                modified_text: "***".into(),
                reason: "pii".into()
            }
            .status_label(),
            "redacted"
        );
    }
}
