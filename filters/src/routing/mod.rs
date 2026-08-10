// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Intelligent routing filters.
//!
//! Provides the `intelligent_route` filter for static AI Grid inference model
//! and MCP tool routing.  This filter belongs in the AI proxy because
//! it encodes AI-specific semantics — candidate freshness, local-site
//! preference, and MCP tool-call routing — that are not generic Praxis
//! proxy mechanics.

pub(crate) mod descriptor;
mod intelligent_route;
pub(crate) mod metadata;
pub(crate) mod overlay;

pub use intelligent_route::IntelligentRouteFilter;
