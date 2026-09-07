// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Error types for OpenAI-compatible API client operations.

/// Errors from OpenAI-compatible API client HTTP operations.
///
/// Covers transport failures, bounded-read overflows, and URL
/// construction problems. Consumers map these to
/// domain-specific error types (e.g. `ResolveError` for the file
/// resolver).
#[derive(Debug)]
pub(crate) enum ApiClientError {
    /// The HTTP exchange failed before a valid response was received.
    Transport {
        /// Typed transport failure from the core sub-request boundary.
        source: crate::subrequest::SubRequestError,
    },

    /// A resource ID cannot be safely encoded as a URL path segment.
    InvalidResourceId {
        /// The resource ID that was rejected.
        resource_id: String,
        /// Human-readable error description.
        detail: String,
    },

    /// The response body exceeded the configured size limit during
    /// a bounded read.
    ResponseTooLarge {
        /// Maximum allowed response size in bytes.
        limit: usize,
    },
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport { source } => {
                let category = match source {
                    crate::subrequest::SubRequestError::AdmissionTimeout { .. } => "admission",
                    crate::subrequest::SubRequestError::CircuitOpen { .. } => "circuit",
                    crate::subrequest::SubRequestError::Connect(_) => "connect",
                    crate::subrequest::SubRequestError::DeadlineExceeded => "deadline",
                    crate::subrequest::SubRequestError::InvalidRequest(_) => "request",
                    crate::subrequest::SubRequestError::Io(_) => "I/O",
                    crate::subrequest::SubRequestError::ResponseTooLarge { .. } => "response size",
                    _ => "unknown",
                };
                write!(f, "API transport {category} failure")
            },
            Self::InvalidResourceId { resource_id, detail } => {
                write!(f, "invalid resource id '{resource_id}': {detail}")
            },
            Self::ResponseTooLarge { limit } => {
                write!(f, "response exceeds size limit ({limit} bytes)")
            },
        }
    }
}
