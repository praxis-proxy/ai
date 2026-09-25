// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Persistence service layer for the `OpenAI`-compatible APIs.
//!
//! The service owns the persistence business logic (record assembly, listing,
//! CRUD, and pending-approval coordination) over an owner-scoped store handle.
//! It receives the store through the persisted-state interface, constructs no
//! backend, and holds no connection pool, so its logic runs against the
//! in-memory backend with no request pipeline and no database.
//!
//! Internal and unstable: a first-party workspace layer, not an external API.

#[cfg(feature = "openai-conversations")]
pub(crate) mod conversations;
pub(crate) mod responses;
