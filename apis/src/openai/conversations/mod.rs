// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Conversations filter: local `/v1/conversations` endpoints.
//!
//! Handles all 8 conversation and item CRUD operations locally
//! via `FilterAction::Reject`, backed by the `ConversationItemStore`
//! trait. Requests never reach upstream.

mod config;
mod contracts;
mod filter;
mod handlers;
pub mod openapi;
mod routes;
mod validate;

pub use filter::OpenaiConversationsFilter;
pub use openapi::implementation_openapi_json;
pub use routes::{ConversationOperation, ConversationOperationSpec, operation_specs};

#[cfg(test)]
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
