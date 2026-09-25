// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conversations filter: local `/v1/conversations` endpoints.
//!
//! Handles all 8 conversation and item CRUD operations locally
//! via `FilterAction::Reject`, backed by the `ConversationItemStore`
//! trait. Requests never reach upstream.

#[cfg(feature = "openai-conversations")]
mod config;
#[cfg(feature = "openai-conversations")]
pub(crate) mod contracts;
#[cfg(feature = "openai-conversations")]
mod filter;
#[cfg(feature = "openai-conversations")]
mod handlers;
#[cfg(feature = "openai-conversations")]
pub(crate) mod item_schema;
#[cfg(feature = "openai-conversations")]
pub mod openapi;
pub(crate) mod routes;
#[cfg(feature = "openai-conversations")]
mod validate;

#[cfg(feature = "openai-conversations")]
pub use config::store_ref_config;
#[cfg(feature = "openai-conversations")]
pub use filter::OpenaiConversationsFilter;
#[cfg(feature = "openai-conversations")]
pub use openapi::implementation_openapi_json;
pub use routes::{ConversationOperation, ConversationOperationSpec, operation_specs};

/// Registry name of the conversations store.
///
/// Distinct from the response store's `DEFAULT_STORE_NAME`: the two filters own
/// structurally different table sets, so they register under different names.
/// The lifecycle backend cache still shares one backend when their effective
/// configs match.
#[cfg(feature = "openai-conversations")]
pub const CONVERSATIONS_STORE_NAME: &str = "conversations";

#[cfg(test)]
#[cfg(all(
    feature = "openai-conversations",
    feature = "store-postgres",
    feature = "store-sqlite"
))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;
