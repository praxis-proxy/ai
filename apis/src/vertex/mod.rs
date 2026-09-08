// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI translation filters.

pub(crate) mod gemini;
pub(crate) mod wire;

pub use gemini::OpenaiChatCompletionsToVertexaiGeminiFilter;
