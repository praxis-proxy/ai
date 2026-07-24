// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Grid gateway-to-gateway routing filters.
//!
//! Provides `grid_route` and `grid_credential_inject` filters for AI Grid
//! inference model routing and per-candidate credential injection.
//!
//! These filters belong in the AI proxy because they encode AI/Grid-specific
//! semantics — candidate freshness, local-site preference, and credential
//! reference resolution — that are not generic Praxis proxy mechanics.
//!
//! # Filter chain ordering
//!
//! ```text
//! grid_route            → selects candidate, writes grid.route.credential.* metadata
//! grid_credential_inject → reads credential metadata, injects Authorization header
//! load_balancer         → routes to selected cluster with injected headers
//! ```
//!
//! `grid_credential_inject` must appear after `grid_route` in the pipeline.
//! It is a no-op when the selected candidate has no credential.

mod credential_inject;
pub(crate) mod descriptor;
mod route;

pub use credential_inject::GridCredentialInjectFilter;
pub use route::GridRouteFilter;
