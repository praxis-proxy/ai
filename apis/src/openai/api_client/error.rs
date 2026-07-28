// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Error types for OpenAI-compatible API client operations.

/// Errors from OpenAI-compatible API client HTTP operations.
///
/// Covers transport failures, bounded-read overflows, JSON decode
/// errors, and URL construction problems. Consumers map these to
/// domain-specific error types (e.g. `ResolveError` for the file
/// resolver).
#[derive(Debug, Clone)]
pub(crate) enum ApiClientError {
    /// The HTTP callout failed (transport error, timeout, circuit
    /// open, or non-2xx status).
    CalloutFailed {
        /// Human-readable error description.
        detail: String,
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

    /// The response body could not be decoded as JSON.
    DecodeFailed {
        /// Human-readable error description.
        detail: String,
    },
}

impl std::fmt::Display for ApiClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CalloutFailed { detail } => {
                write!(f, "API callout failed: {detail}")
            },
            Self::InvalidResourceId { resource_id, detail } => {
                write!(f, "invalid resource id '{resource_id}': {detail}")
            },
            Self::ResponseTooLarge { limit } => {
                write!(f, "response exceeds size limit ({limit} bytes)")
            },
            Self::DecodeFailed { detail } => {
                write!(f, "response decode failed: {detail}")
            },
        }
    }
}
