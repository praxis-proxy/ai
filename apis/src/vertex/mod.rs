// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI translation filters.

#[cfg(feature = "vertex-anthropic")]
pub(crate) mod anthropic;
pub(crate) mod gemini;
pub(crate) mod wire;

#[cfg(feature = "vertex-anthropic")]
pub use anthropic::AnthropicMessagesToVertexaiAnthropicFilter;
pub use gemini::OpenaiChatCompletionsToVertexaiGeminiFilter;
