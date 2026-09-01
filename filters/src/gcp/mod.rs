// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! GCP Application Default Credentials (ADC) upstream authentication.
//!
//! **Experimental.** Requires the `gcp-adc-filter` cargo feature, which
//! is off by default and activates the `experimental` marker. The
//! configuration surface may change between releases.
//!
//! Token acquisition is not implemented yet — see [`GcpAdcFilter`] for
//! the current fail-closed behavior and the upstream tracking issues.
//!
//! Classic GKE Workload Identity is the metadata server (ADC), not
//! STS/WIF. Vertex AI needs an `OAuth2` access token with a **scope**,
//! not an identity-token audience.

mod config;
mod filter;
mod token;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp,
    reason = "tests"
)]
mod tests;

pub use self::filter::GcpAdcFilter;
