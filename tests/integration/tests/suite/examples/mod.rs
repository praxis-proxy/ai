// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod agentic_routing;
mod anthropic_messages;
mod credential_injection;
mod full_flow;
mod mcp_broker;
mod model_to_header;
mod openai_conversations;
mod openai_response_store;
mod openai_response_store_postgres;
mod openai_responses_format;
mod openai_responses_model_rewrite;
mod openai_responses_validate;
mod openai_stream_events;
mod prompt_enrichment;
mod rehydrate;
mod responses_proxy;
mod responses_routing;
mod session_replay;
mod token_usage_headers;
