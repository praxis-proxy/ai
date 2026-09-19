// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod operation_classifier;
mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod agentic_routing;
mod anthropic_full_flow_agentic;
mod anthropic_messages;
mod aws_sigv4;
#[cfg(feature = "azure-ad-filter")]
mod azure_ad;
mod azure_translation;
#[cfg(feature = "store-sqlite")]
mod compact;
mod credential_injection;
mod external_metering;
mod file_search_callout;
mod file_search_chat_completions;
mod file_search_streaming;
#[cfg(feature = "store-sqlite")]
mod full_flow_agentic;
#[cfg(feature = "gcp-adc-filter")]
mod gcp_adc;
mod guardrails;
mod guardrails_response;
mod identity_header_guard;
mod inference_fallback;
mod intelligent_route_hardening;
mod intelligent_route_management_skip;
mod irr_terminal_streaming;
#[cfg(feature = "http-callout-filter")]
mod lakera_guard;
#[cfg(feature = "llmd-ext-proc")]
mod llmd_ext_proc;
mod llmisvc_model_provider_resolver;
mod mcp_broker;
mod model_to_header;
#[cfg(feature = "store-sqlite")]
mod openai_agentic_loop;
#[cfg(feature = "store-sqlite")]
mod openai_client_tool_compat;
#[cfg(feature = "store-sqlite")]
mod openai_conversations;
mod openai_conversations_postgres_mtls;
mod openai_doc_extract;
mod openai_embeddings_routing;
mod openai_file_resolve;
mod openai_mcp_dispatch;
mod openai_mcp_outbound_chain;
mod openai_mcp_tool_resolve;
mod openai_prompts_routing;
#[cfg(feature = "store-sqlite")]
mod openai_response_store;
#[cfg(feature = "store-postgres")]
mod openai_response_store_postgres;
mod openai_response_store_postgres_mtls;
mod openai_responses_body_size_limits;
mod openai_responses_format;
mod openai_responses_model_rewrite;
mod openai_responses_proxy;
mod openai_responses_validate;
mod openai_state_ownership;
#[cfg(feature = "store-sqlite")]
mod openai_stream_events;
mod openai_tool_parse;
mod prompt_enrichment;
mod provider_route;
#[cfg(feature = "store-sqlite")]
mod rehydrate;
mod responses_routing;
#[cfg(feature = "store-sqlite")]
mod responses_to_chat_completions;
#[cfg(feature = "store-sqlite")]
mod responses_to_chat_completions_conformance;
#[cfg(feature = "store-sqlite")]
mod session_replay;
mod state_owner_headers;
mod time_to_first_token;
mod token_count;
mod token_counting;
#[cfg(feature = "token-rate-limit-filter")]
mod token_rate_limit;
mod token_usage_headers;
mod vector_stores_routing;
mod vllm_agentic_api;
mod web_search;
mod web_search_chat_completions;
