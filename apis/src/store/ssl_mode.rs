// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `PostgreSQL` TLS mode configuration shared with backend-independent parsing.

use serde::Deserialize;

/// TLS mode for `PostgreSQL` connections.
///
/// Defaults to [`VerifyFull`](Self::VerifyFull), which requires TLS and
/// verifies both the server certificate chain and hostname. Use [`Disable`](Self::Disable)
/// or [`Prefer`](Self::Prefer) only for local development with an explicit opt-in.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SslMode {
    /// Do not use TLS.
    Disable,

    /// Attempt TLS, falling back to plaintext.
    Prefer,

    /// Require TLS without verifying the server certificate.
    Require,

    /// Require TLS and verify the server certificate chain.
    VerifyCa,

    /// Require TLS and verify both the certificate chain and hostname.
    #[default]
    VerifyFull,
}
