// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod operation_classifier;
mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod agentic_routing;
mod anthropic_full_flow_agentic;
mod anthropic_full_flow_agentic_chat;
#[cfg(feature = "openai-responses")]
mod anthropic_messages;
mod anthropic_messages_native_vllm;
mod anthropic_messages_to_openai_vllm;
mod anthropic_web_search_scoped_credentials;
mod anthropic_web_search_tavily_scoped_credentials;
mod aws_sigv4;
#[cfg(feature = "azure-ad-filter")]
mod azure_ad;
mod azure_translation;
mod bedrock;
#[cfg(feature = "openai-agentic-loop")]
mod budget_overflow_streaming;
#[cfg(feature = "openai-agentic-loop")]
#[cfg(feature = "store-sqlite")]
mod client_tool_compat_chat_completions;
#[cfg(all(feature = "openai-compact", feature = "openai-file-resolve-filter"))]
#[cfg(feature = "store-sqlite")]
mod compact;
mod credential_injection;
mod external_metering;
#[cfg(target_os = "linux")]
mod file_descriptor_limits;
#[cfg(feature = "openai-agentic-loop")]
mod file_search_chat_completions;
#[cfg(feature = "openai-agentic-loop")]
mod file_search_dispatch;
#[cfg(feature = "openai-agentic-loop")]
mod file_search_streaming;
#[cfg(feature = "openai-responses")]
mod format_routing;
#[cfg(all(
    feature = "openai-agentic-loop",
    feature = "openai-compact",
    feature = "openai-conversations",
    feature = "openai-file-resolve-filter",
    feature = "openai-mcp-tools"
))]
#[cfg(feature = "store-sqlite")]
mod full_flow_agentic;
#[cfg(feature = "gcp-adc-filter")]
mod gcp_adc;
mod guardrails;
mod guardrails_response;
mod identity_header_guard;
#[cfg(feature = "openai-responses")]
mod inference_fallback;
#[cfg(feature = "inflight-tracker-filter")]
mod inflight_tracking;
mod intelligent_route_hardening;
mod intelligent_route_management_skip;
mod intelligent_route_metrics;
mod intelligent_route_projected_credentials;
mod intelligent_route_selected_anthropic_legacy;
#[cfg(feature = "openai-responses")]
mod irr_terminal_streaming;
#[cfg(feature = "http-callout-filter")]
mod lakera_guard;
#[cfg(feature = "llmd-ext-proc")]
mod llmd_ext_proc;
mod llmisvc_model_provider_resolver;
mod mcp_broker;
mod model_to_header;
mod model_to_header_trust_existing;
mod model_to_provider;
#[cfg(all(feature = "openai-agentic-loop", feature = "openai-mcp-tools"))]
#[cfg(feature = "store-sqlite")]
mod openai_agentic_loop;
#[cfg(feature = "openai-agentic-loop")]
#[cfg(feature = "store-sqlite")]
mod openai_client_tool_compat;
#[cfg(feature = "openai-conversations")]
#[cfg(feature = "store-sqlite")]
mod openai_conversations;
#[cfg(all(feature = "store-postgres", feature = "openai-conversations"))]
mod openai_conversations_postgres_mtls;
#[cfg(feature = "openai-file-resolve-filter")]
mod openai_doc_extract;
mod openai_embeddings_routing;
#[cfg(feature = "openai-file-resolve-filter")]
mod openai_file_resolve;
#[cfg(all(
    feature = "openai-agentic-loop",
    feature = "openai-mcp-tools",
    feature = "store-sqlite"
))]
mod openai_mcp_approval_floor;
#[cfg(feature = "openai-mcp-tools")]
mod openai_mcp_dispatch;
#[cfg(feature = "openai-mcp-tools")]
mod openai_mcp_outbound_chain;
#[cfg(all(feature = "openai-agentic-loop", feature = "openai-mcp-tools"))]
#[cfg(feature = "store-sqlite")]
mod openai_mcp_streaming;
#[cfg(feature = "openai-mcp-tools")]
mod openai_mcp_tool_admission;
#[cfg(feature = "openai-mcp-tools")]
mod openai_mcp_tool_resolve;
mod openai_prompts_routing;
#[cfg(all(feature = "openai-file-resolve-filter", feature = "openai-mcp-tools"))]
mod openai_responses_body_size_limits;
#[cfg(feature = "openai-responses")]
mod openai_responses_model_rewrite;
#[cfg(feature = "openai-responses")]
mod openai_responses_proxy;
#[cfg(feature = "openai-responses")]
mod openai_responses_request;
#[cfg(feature = "store-sqlite")]
mod openai_responses_store;
#[cfg(feature = "store-postgres")]
mod openai_responses_store_postgres;
#[cfg(any(feature = "store-postgres", feature = "store-postgres-cert-auth"))]
mod openai_responses_store_postgres_mtls;
#[cfg(all(feature = "store-sqlite", feature = "openai-file-resolve-filter"))]
mod openai_retained_file_resolve;
#[cfg(feature = "store-sqlite")]
mod openai_retained_stream;
// The state-ownership example selects the SQLite store backend.
#[cfg(feature = "store-sqlite")]
mod openai_responses_to_chat_completions;
#[cfg(feature = "store-sqlite")]
mod openai_responses_to_chat_completions_conformance;
#[cfg(feature = "openai-agentic-loop")]
#[cfg(feature = "store-sqlite")]
mod openai_responses_to_chat_completions_reasoning;
#[cfg(feature = "openai-conversations")]
#[cfg(feature = "store-sqlite")]
mod openai_state_ownership;
#[cfg(all(
    feature = "openai-agentic-loop",
    feature = "openai-compact",
    feature = "openai-conversations",
    feature = "openai-file-resolve-filter",
    feature = "openai-mcp-tools"
))]
#[cfg(feature = "store-sqlite")]
mod openai_stream_events;
mod openai_tool_parse;
mod project_state_owner_headers;
mod prompt_enrichment;
mod provider_route;
#[cfg(feature = "store-sqlite")]
mod rehydrate;
#[cfg(feature = "openai-responses")]
mod responses_routing;
#[cfg(all(
    feature = "openai-agentic-loop",
    feature = "openai-compact",
    feature = "openai-conversations",
    feature = "openai-file-resolve-filter",
    feature = "openai-mcp-tools"
))]
#[cfg(feature = "store-sqlite")]
mod session_replay;
mod stream_usage_inject;
#[cfg(feature = "openai-responses")]
mod time_to_first_token;
#[cfg(feature = "token-ceiling-filter")]
mod token_ceiling;
mod token_count;
mod token_counting;
#[cfg(feature = "token-rate-limit-filter")]
mod token_rate_limit;
mod token_usage_headers;
mod vector_stores_routing;
#[cfg(feature = "vertex-anthropic-filter")]
mod vertex_anthropic;
mod vertex_gemini;
#[cfg(feature = "openai-responses")]
mod vllm_agentic_api;
#[cfg(feature = "openai-responses")]
mod web_search;
#[cfg(feature = "openai-agentic-loop")]
mod web_search_chat_completions;
#[cfg(feature = "openai-agentic-loop")]
mod web_search_scoped_credentials;
