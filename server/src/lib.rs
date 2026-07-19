// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Server bootstrap for Praxis AI.

pub(crate) mod pipelines;
pub(crate) mod reload;
mod server;
pub(crate) mod watcher;
pub use pipelines::resolve_pipelines;
pub use praxis_core::{config::load_config, logging::init_tracing};
pub use server::{check_root_privilege, fatal, resolve_config_path, run_server, run_server_with_registry};

// -----------------------------------------------------------------------------
// External Filter Discovery
// -----------------------------------------------------------------------------

// Provides: fn register_external_filters(&mut FilterRegistry)
include!(concat!(env!("OUT_DIR"), "/external_filters.rs"));

/// Build a [`FilterRegistry`] with core builtins, AI filters, and
/// auto-discovered external filters.
///
/// [`FilterRegistry`]: praxis_filter::FilterRegistry
#[must_use]
pub fn build_full_registry() -> praxis_filter::FilterRegistry {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    register_ai_filters(&mut registry);
    register_external_filters(&mut registry);
    registry
}

/// Register all AI filters into the registry.
fn register_ai_filters(registry: &mut praxis_filter::FilterRegistry) {
    register_agentic_filters(registry);
    register_general_ai_filters(registry);
    register_anthropic_filters(registry);
    register_openai_filters(registry);
}

/// Register agentic protocol filters (A2A, MCP).
fn register_agentic_filters(registry: &mut praxis_filter::FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "a2a" => praxis_ai_filters::A2aFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "mcp" => praxis_ai_filters::McpFilter::from_config
    );
}

/// Register general-purpose AI filters.
fn register_general_ai_filters(registry: &mut praxis_filter::FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "ai_guardrails" => praxis_ai_filters::AiGuardrailsFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "model_to_header" => praxis_ai_filters::ModelToHeaderFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "prompt_enrich" => praxis_ai_filters::PromptEnrichFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "token_count" => praxis_ai_filters::TokenCountFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "token_usage_headers" => praxis_ai_filters::TokenUsageHeadersFilter::from_config
    );
}

/// Register Anthropic-specific filters.
fn register_anthropic_filters(registry: &mut praxis_filter::FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_messages_format" => praxis_ai_apis::anthropic::AnthropicMessagesFormatFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_messages_protocol" => praxis_ai_apis::anthropic::AnthropicMessagesProtocolFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_stream_events" => praxis_ai_apis::anthropic::AnthropicStreamEventsFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_to_openai" => praxis_ai_apis::anthropic::AnthropicToOpenaiFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "anthropic_validate" => praxis_ai_apis::anthropic::AnthropicValidateFilter::from_config
    );
}

/// Register OpenAI Responses API request-path filters.
fn register_openai_filters(registry: &mut praxis_filter::FilterRegistry) {
    register_openai_responses_filters(registry);
    praxis_filter::register_filters!(
        @register registry,
        http "openai_conversations" => praxis_ai_apis::openai::OpenaiConversationsFilter::from_config
    );
}

/// Register OpenAI Responses API filters.
fn register_openai_responses_filters(registry: &mut praxis_filter::FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_format" => praxis_ai_apis::openai::ResponsesFormatFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_model_rewrite" => praxis_ai_apis::openai::ModelRewriteFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_validate" => praxis_ai_apis::openai::OpenaiResponsesValidateFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_rehydrate" => praxis_ai_apis::openai::RehydrateFilter::from_config
    );
    register_openai_response_filters(registry);
}

/// Register OpenAI Responses API response-path and persistence filters.
fn register_openai_response_filters(registry: &mut praxis_filter::FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "openai_response_store" => praxis_ai_apis::openai::ResponseStoreFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_stream_events" => praxis_ai_apis::openai::OpenaiStreamEventsFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_responses_proxy" => praxis_ai_apis::openai::ResponsesProxyFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_mcp_tool_resolve" => praxis_ai_apis::openai::McpToolResolveFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_tool_parse" => praxis_ai_apis::openai::ToolParseFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_web_search" => praxis_ai_apis::openai::WebSearchFilter::from_config
    );
}
