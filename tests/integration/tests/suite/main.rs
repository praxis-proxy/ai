// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Integration test suite for Praxis.

#![allow(
    clippy::allow_attributes_without_reason,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::clone_on_ref_ptr,
    clippy::cognitive_complexity,
    clippy::default_trait_access,
    clippy::disallowed_methods,
    clippy::doc_markdown,
    clippy::doc_nested_refdefs,
    clippy::expect_used,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::len_zero,
    clippy::manual_is_multiple_of,
    clippy::manual_let_else,
    clippy::map_unwrap_or,
    clippy::needless_raw_string_hashes,
    clippy::needless_raw_strings,
    clippy::panic,
    clippy::print_stderr,
    clippy::redundant_closure_for_method_calls,
    clippy::string_add,
    clippy::tests_outside_test_module,
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::used_underscore_binding,
    clippy::useless_format,
    reason = "test code"
)]

mod a2a;
mod agentic_mocks;
mod anthropic_messages;
mod claude_code;
mod claude_code_vllm;
#[cfg(all(feature = "openai-agentic-loop", feature = "store"))]
mod codex_http;
#[cfg(all(
    feature = "openai-agentic-loop",
    feature = "openai-compact",
    feature = "openai-conversations",
    feature = "openai-file-resolve-filter",
    feature = "openai-mcp-tools"
))]
mod codex_websocket;
#[cfg(feature = "openai-conversations")]
#[cfg(feature = "store-sqlite")]
mod conversations_rehydrate;
#[cfg(feature = "openai-responses")]
mod error_response_formatter;
mod examples;
mod failure_mode;
mod fips;
mod guardrails;
mod harness;
// Several replayed scenarios run examples backed by the SQLite response store.
#[cfg(all(
    feature = "store-sqlite",
    feature = "openai-agentic-loop",
    feature = "openai-mcp-tools",
    feature = "openai-compact",
    feature = "openai-conversations",
    feature = "openai-file-resolve-filter"
))]
mod inference_fixtures;
mod live_vllm;
mod mcp;
mod mcp_broker;
#[cfg(feature = "openai-mcp-tools")]
#[cfg(feature = "store-sqlite")]
mod openai_mcp_tool_resolve;
#[cfg(feature = "openai-responses")]
mod openai_responses_model_rewrite;
#[cfg(feature = "openai-responses")]
mod openai_responses_request;
#[cfg(feature = "openai-responses")]
mod openai_tool_parse;
mod opencode;
mod opencode_v2;
mod opencode_v2_vllm;
mod opencode_vllm;
mod prompt_enrich;
#[cfg(feature = "openai-responses")]
mod responses_routing;
