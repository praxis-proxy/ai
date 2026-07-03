// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod agentic_routing;
#[cfg(feature = "ai-inference")]
mod anthropic_messages;
mod credential_injection;
#[cfg(feature = "ai-inference")]
mod full_flow;
#[cfg(feature = "ai-inference")]
mod model_to_header;
#[cfg(feature = "ai-inference")]
mod openai_conversations;
#[cfg(feature = "ai-inference")]
mod openai_response_store;
#[cfg(feature = "ai-inference")]
mod openai_response_store_postgres;
#[cfg(feature = "ai-inference")]
mod openai_responses_format;
#[cfg(feature = "ai-inference")]
mod openai_responses_model_rewrite;
#[cfg(feature = "ai-inference")]
mod openai_responses_validate;
#[cfg(feature = "ai-inference")]
mod prompt_enrichment;
#[cfg(feature = "ai-inference")]
mod rehydrate;
#[cfg(feature = "ai-inference")]
mod responses_proxy;
#[cfg(feature = "ai-inference")]
mod responses_routing;
mod token_usage_headers;
