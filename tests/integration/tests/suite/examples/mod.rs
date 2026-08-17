// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod agentic_routing;
mod anthropic_messages;
mod anthropic_messages_web_search;
mod compact;
mod credential_injection;
mod file_search_callout;
mod full_flow;
mod full_flow_agentic;
mod guardrails;
mod inference_fallback;
mod mcp_broker;
mod model_to_header;
mod openai_agentic_loop;
mod openai_conversations;
mod openai_doc_extract;
mod openai_embeddings_routing;
mod openai_file_resolve;
mod openai_mcp_dispatch;
mod openai_mcp_tool_resolve;
mod openai_prompts_routing;
mod openai_response_store;
mod openai_response_store_postgres;
mod openai_responses_format;
mod openai_responses_model_rewrite;
mod openai_responses_proxy;
mod openai_responses_validate;
mod openai_stream_events;
mod openai_tool_parse;
mod prompt_enrichment;
mod provider_route;
mod rehydrate;
mod responses_routing;
mod responses_to_chat_completions;
mod session_replay;
mod time_to_first_token;
mod token_count;
mod token_counting;
mod token_usage_headers;
mod vector_stores_routing;
mod vllm_agentic_api;
mod web_search;
