// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Intelligent routing filters.
//!
//! Provides the edge `intelligent_route` filter and the provider-side
//! `provider_route` and `credential_inject` filters. These belong in
//! the AI proxy because they encode AI-specific routing contracts, not
//! generic Praxis proxy mechanics.

mod credential_inject;
pub(crate) mod descriptor;
pub(crate) mod group_index;
mod intelligent_route;
pub(crate) mod metadata;
pub(crate) mod overlay;
pub(crate) mod picker;
mod provider_route;

pub use credential_inject::CredentialInjectFilter;
pub use intelligent_route::IntelligentRouteFilter;
pub use provider_route::ProviderRouteFilter;
