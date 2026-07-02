// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Integration tests for example configurations.

mod test_utils;
#[expect(unreachable_pub)]
pub use test_utils::load_example_config;

mod access_logging;
mod admin_interface;
mod agentic_routing;
mod anthropic_messages;
mod api_key_filter;
mod basic_reverse_proxy;
mod canary_routing;
mod circuit_breaker;
mod conditional_filters;
mod credential_injection;
mod csrf;
mod default_config;
mod full_flow;
mod grpc_detection;
mod guardrails;
mod header_manipulation;
mod health_checks;
mod hostname_upstream;
mod least_connections;
mod logging;
mod max_body_guard;
mod max_connections;
mod model_to_header;
mod multi_listener;
mod openai_conversations;
mod openai_response_store;
mod openai_response_store_postgres;
mod openai_responses_format;
mod openai_responses_model_rewrite;
mod openai_responses_validate;
mod p2c;
mod path_based_routing;
mod path_rewriting;
mod payload_processing;
#[cfg(feature = "cpex-policy-engine")]
mod policy;
mod prompt_enrichment;
mod protocols;
mod redirect;
mod rehydrate;
mod responses_proxy;
mod responses_routing;
mod round_robin;
mod session_affinity;
mod static_response;
mod stream_buffer;
mod timeout;
mod token_usage_headers;
mod virtual_hosts;
mod websocket;
mod weighted_load_balancing;
