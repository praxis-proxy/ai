// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(
    dead_code,
    reason = "the client and citation layers precede their stacked filter consumer"
)]

//! OGX-backed file search support for the OpenAI Responses API.

pub(crate) mod citations;
pub(crate) mod client;
