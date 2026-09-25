// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` Responses API store utilities.
//!
//! Helpers that operate on the generic [`ResponseStore`] but are
//! specific to the `OpenAI` Responses API (e.g., input item
//! pagination for the `/v1/responses/{id}/input_items` endpoint).
//!
//! [`ResponseStore`]: crate::store::ResponseStore

mod config;
mod filter;

// Input-item listing moved to the service layer (`crate::service::responses`).
// Re-exported here at the historical path so the conversations module and the
// store tests keep resolving these names unchanged.
pub use self::filter::ResponseStoreFilter;
#[cfg(feature = "openai-conversations")]
pub(crate) use crate::service::responses::input_items::DEFAULT_PAGE_LIMIT;
pub(crate) use crate::service::responses::{InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, list_input_items};

#[cfg(test)]
#[cfg(all(feature = "store-postgres", feature = "store-sqlite"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    reason = "tests"
)]
mod tests;
