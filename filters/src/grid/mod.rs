// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Grid gateway-to-gateway routing filters.
//!
//! Provides the `grid_route` filter for static AI Grid inference model
//! and MCP tool routing.  This filter belongs in the AI proxy because
//! it encodes AI/Grid-specific semantics — candidate freshness, local-site
//! preference, and MCP tool-call routing — that are not generic Praxis
//! proxy mechanics.

pub(crate) mod descriptor;
mod route;

pub use route::GridRouteFilter;
