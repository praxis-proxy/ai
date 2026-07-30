// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Public AI filter registration for consumers outside `praxis-ai-proxy`.

use praxis_filter::FilterRegistry;

use crate::{
    A2aFilter, AiGuardrailsFilter, McpFilter, ModelToHeaderFilter, PromptEnrichFilter, TimeToFirstTokenFilter,
    TokenCountFilter, TokenUsageHeadersFilter,
};

/// Register all in-tree AI HTTP filters into `registry`.
///
/// Does not call [`FilterRegistry::with_builtins`].
/// Does not register auto-discovered external filters.
///
/// Pipelines that use OpenAI store or rehydrate filters must also install:
///
/// ```rust,ignore
/// pipeline.add_pipeline_extension(
///     Box::new(praxis_ai_apis::store::ResponseStoreRegistry::new()),
/// );
/// ```
pub fn register_ai_filters(registry: &mut FilterRegistry) {
    register_agentic_filters(registry);
    register_general_ai_filters(registry);
    register_anthropic_filters(registry);
    register_openai_filters(registry);
}

/// Build a [`FilterRegistry`] with core builtins and in-tree AI filters.
///
/// Equivalent to [`FilterRegistry::with_builtins`] followed by
/// [`register_ai_filters`]. Does not register auto-discovered external
/// filters.
///
/// Pipelines that use OpenAI store or rehydrate filters must also install
/// [`praxis_ai_apis::store::ResponseStoreRegistry`] as a pipeline extension.
#[must_use]
pub fn build_ai_registry() -> FilterRegistry {
    let mut registry = FilterRegistry::with_builtins();
    register_ai_filters(&mut registry);
    registry
}

/// Register agentic protocol filters (A2A, MCP).
fn register_agentic_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "a2a" => A2aFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "mcp" => McpFilter::from_config
    );
}

/// Register general-purpose AI filters.
fn register_general_ai_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "ai_guardrails" => AiGuardrailsFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "model_to_header" => ModelToHeaderFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "prompt_enrich" => PromptEnrichFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "token_count" => TokenCountFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "token_usage_headers" => TokenUsageHeadersFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "time_to_first_token" => TimeToFirstTokenFilter::from_config
    );
}

/// Register Anthropic-specific filters.
fn register_anthropic_filters(registry: &mut FilterRegistry) {
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
fn register_openai_filters(registry: &mut FilterRegistry) {
    register_openai_responses_filters(registry);
    praxis_filter::register_filters!(
        @register registry,
        http "openai_conversations" => praxis_ai_apis::openai::OpenaiConversationsFilter::from_config
    );
}

/// Register OpenAI Responses API filters.
fn register_openai_responses_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "openai_doc_extract" => praxis_ai_apis::openai::DocExtractFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_file_resolve" => praxis_ai_apis::openai::FileResolveFilter::from_config
    );
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
fn register_openai_response_filters(registry: &mut FilterRegistry) {
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
    praxis_filter::register_filters!(
        @register registry,
        http "openai_mcp_dispatch" => praxis_ai_apis::openai::McpDispatchFilter::from_config
    );
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::build_ai_registry;

    #[test]
    fn build_ai_registry_includes_ai_and_builtin_filters() {
        let registry = build_ai_registry();
        let names = registry.available_filters();
        assert!(names.contains(&"ai_guardrails"), "expected ai_guardrails in registry");
        assert!(
            names.contains(&"openai_responses_validate"),
            "expected openai_responses_validate in registry"
        );
        assert!(names.contains(&"a2a"), "expected agentic filter a2a in registry");
        assert!(
            names.contains(&"anthropic_validate"),
            "expected anthropic filter in registry"
        );
        assert!(
            names.contains(&"request_id"),
            "expected core builtin request_id in registry"
        );
    }
}
