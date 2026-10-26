// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Persistence helpers for the `OpenAI`-compatible APIs.
//!
//! Record assembly and input-item listing live here. CRUD and pending-approval
//! operations use the owner-scoped store handle resolved by each caller.
//!
//! Internal and unstable: a first-party workspace layer, not an external API.

#[cfg(feature = "openai-conversations")]
pub(crate) mod conversations;
pub(crate) mod responses;
