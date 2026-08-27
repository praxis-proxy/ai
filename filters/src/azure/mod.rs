// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Azure-specific filters (Entra ID upstream authentication, and
//! future Azure integrations).

mod azure_ad;

pub use azure_ad::AzureAdFilter;
