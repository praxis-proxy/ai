// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! In-flight per-model request tracking (issue #374).
//!
//! [`InFlightRegistry`] is the shared counter table and stays compiled
//! unconditionally, so a pipeline can always install it as an extension
//! regardless of whether the `inflight_tracker` filter itself is enabled. The
//! filter, its config, and its request-body scanner sit behind the
//! `inflight-tracker-filter` feature: nothing reads the registry's counters
//! yet (no scorer, no HTTP surface), and its eventual backing store is still
//! an open question, so the capability stays opt-in until both land.

#[cfg(feature = "inflight-tracker-filter")]
mod config;
#[cfg(feature = "inflight-tracker-filter")]
mod filter;
mod registry;
#[cfg(feature = "inflight-tracker-filter")]
mod scan;

#[cfg(feature = "inflight-tracker-filter")]
pub use filter::{FILTER_NAME, InFlightTrackerFilter};
pub use registry::{InFlightRegistry, ModelMetrics};
