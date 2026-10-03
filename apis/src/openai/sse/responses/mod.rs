// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses API SSE event typing.

mod event;

#[cfg(feature = "openai-responses")]
pub(crate) use event::ResponsesEvent;
