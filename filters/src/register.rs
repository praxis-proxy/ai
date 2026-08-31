// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Public AI filter registration for consumers outside `praxis-ai-proxy`.

use praxis_core::subrequest::SubRequestClient;
use praxis_filter::FilterRegistry;

#[cfg(feature = "azure-ad-filter")]
use crate::AzureAdFilter;
#[cfg(feature = "gcp-adc-filter")]
use crate::GcpAdcFilter;
#[cfg(feature = "http-callout-filter")]
use crate::HttpCalloutFilter;
#[cfg(feature = "token-rate-limit-filter")]
use crate::TokenRateLimitFilter;
use crate::{
    A2aFilter, AiGuardrailsFilter, CredentialInjectFilter, IntelligentRouteFilter, McpFilter, ModelToHeaderFilter,
    PromptEnrichFilter, ProviderRouteFilter, Sigv4SignFilter, TimeToFirstTokenFilter, TokenCountFilter,
    TokenUsageHeadersFilter,
};

/// Register all in-tree AI HTTP filters into `registry`.
///
/// When `subrequest_client` is provided, filters that make HTTP
/// callouts (`openai_file_resolve`, `openai_web_search`,
/// `anthropic_web_search`) capture the
/// shared client instead of creating isolated per-filter connectors.
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
pub fn register_ai_filters(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    register_agentic_filters(registry);
    register_aws_filters(registry);
    #[cfg(feature = "azure-ad-filter")]
    register_azure_filters(registry);
    #[cfg(feature = "gcp-adc-filter")]
    register_gcp_filters(registry);
    register_general_ai_filters(registry);
    register_anthropic_filters(registry, subrequest_client);
    register_openai_filters(registry, subrequest_client);
    register_routing_filters(registry);
}

/// Build a [`FilterRegistry`] with core builtins and in-tree AI filters.
///
/// Equivalent to [`FilterRegistry::with_builtins`] followed by
/// [`register_ai_filters`] with no shared sub-request client. Does
/// not register auto-discovered external filters.
///
/// Filters that make HTTP callouts create isolated per-filter
/// connectors. Use [`register_ai_filters`] with a shared client
/// when the server runtime is available.
///
/// Pipelines that use OpenAI store or rehydrate filters must also install
/// [`praxis_ai_apis::store::ResponseStoreRegistry`] as a pipeline extension.
#[must_use]
pub fn build_ai_registry() -> FilterRegistry {
    let mut registry = FilterRegistry::with_builtins();
    register_ai_filters(&mut registry, None);
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

/// Register AWS-specific filters.
fn register_aws_filters(registry: &mut FilterRegistry) {
    register_routing_security_filter(registry, "aws_sigv4_sign", Sigv4SignFilter::from_config);
}

/// Register Azure-specific filters.
#[cfg(feature = "azure-ad-filter")]
fn register_azure_filters(registry: &mut FilterRegistry) {
    register_routing_security_filter(registry, "azure_ad", AzureAdFilter::from_config);
}

/// Register GCP-specific filters.
#[cfg(feature = "gcp-adc-filter")]
fn register_gcp_filters(registry: &mut FilterRegistry) {
    register_routing_security_filter(registry, "gcp_adc", GcpAdcFilter::from_config);
}

/// Register general-purpose AI filters.
fn register_general_ai_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "ai_guardrails" => AiGuardrailsFilter::from_config
    );
    #[cfg(feature = "http-callout-filter")]
    praxis_filter::register_filters!(
        @register registry,
        http "http_callout" => HttpCalloutFilter::from_config
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
        http "time_to_first_token" => TimeToFirstTokenFilter::from_config
    );
    register_token_filters(registry);
}

/// Register token counting/usage/rate-limiting filters.
fn register_token_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "token_count" => TokenCountFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "token_usage_headers" => TokenUsageHeadersFilter::from_config
    );
    #[cfg(feature = "token-rate-limit-filter")]
    praxis_filter::register_filters!(
        @register registry,
        http "token_rate_limit" => TokenRateLimitFilter::from_config
    );
}

/// Register intelligent routing filters.
fn register_routing_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "intelligent_route" => IntelligentRouteFilter::from_config
    );
    register_routing_security_filter(registry, "provider_route", ProviderRouteFilter::from_config);
    register_routing_security_filter(registry, "credential_inject", CredentialInjectFilter::from_config);
}

/// Register a routing HTTP filter as security-critical.
#[expect(
    clippy::type_complexity,
    reason = "single-use registration helper; a type alias adds indirection"
)]
#[expect(clippy::panic, reason = "duplicate filter registration is a fatal configuration bug")]
fn register_routing_security_filter(
    registry: &mut FilterRegistry,
    name: &'static str,
    factory: fn(&serde_yaml::Value) -> Result<Box<dyn praxis_filter::HttpFilter>, praxis_filter::FilterError>,
) {
    registry
        .register_with_class(
            name,
            praxis_filter::FilterFactory::Http(std::sync::Arc::new(factory)),
            praxis_filter::SecurityClass::Security,
        )
        .unwrap_or_else(|_| panic!("duplicate filter name: '{name}'"));
}

/// Register Anthropic-specific filters.
fn register_anthropic_filters(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
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
    register_anthropic_web_search(registry, subrequest_client);
}

/// Register OpenAI Responses API request-path filters.
fn register_openai_filters(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    register_openai_responses_filters(registry, subrequest_client);
    praxis_filter::register_filters!(
        @register registry,
        http "openai_conversations" => praxis_ai_apis::openai::OpenaiConversationsFilter::from_config
    );
}

/// Register OpenAI Responses API filters.
fn register_openai_responses_filters(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    praxis_filter::register_filters!(
        @register registry,
        http "openai_doc_extract" => praxis_ai_apis::openai::DocExtractFilter::from_config
    );
    register_file_resolve(registry, subrequest_client);
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
    register_compact(registry, subrequest_client);
    register_file_search_callout(registry, subrequest_client);
    register_openai_response_filters(registry, subrequest_client);
}

/// Register OpenAI Responses API response-path and persistence filters.
fn register_openai_response_filters(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
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
        http "responses_to_chat_completions" => praxis_ai_apis::openai::ResponsesToChatCompletionsFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_mcp_tool_resolve" => praxis_ai_apis::openai::McpToolResolveFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_tool_parse" => praxis_ai_apis::openai::ToolParseFilter::from_config
    );
    register_web_search(registry, subrequest_client);
    register_openai_agentic_filters(registry);
}

/// Register OpenAI agentic loop and MCP dispatch filters.
fn register_openai_agentic_filters(registry: &mut FilterRegistry) {
    praxis_filter::register_filters!(
        @register registry,
        http "openai_mcp_dispatch" => praxis_ai_apis::openai::McpDispatchFilter::from_config
    );
    praxis_filter::register_filters!(
        @register registry,
        http "openai_agentic_loop" => praxis_ai_apis::openai::AgenticLoopFilter::from_config
    );
}

// -----------------------------------------------------------------------------
// Sub-request-aware registration
// -----------------------------------------------------------------------------

/// Register `anthropic_web_search` with the shared client when
/// available, otherwise fall back to an isolated per-filter connector.
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_anthropic_web_search(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry
            .register(
                "anthropic_web_search",
                praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                    praxis_ai_apis::anthropic::AnthropicWebSearchFilter::from_config_with_client(config, client.clone())
                })),
            )
            .unwrap_or_else(|_| panic!("duplicate filter name: 'anthropic_web_search'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "anthropic_web_search" => praxis_ai_apis::anthropic::AnthropicWebSearchFilter::from_config
        );
    }
}

/// Register `openai_file_resolve` with the shared client when
/// available, otherwise fall back to an isolated per-filter connector.
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_file_resolve(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry
            .register(
                "openai_file_resolve",
                praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                    praxis_ai_apis::openai::FileResolveFilter::from_config_with_client(config, client.clone())
                })),
            )
            .unwrap_or_else(|_| panic!("duplicate filter name: 'openai_file_resolve'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "openai_file_resolve" => praxis_ai_apis::openai::FileResolveFilter::from_config
        );
    }
}

/// Register `openai_responses_compact` with the shared client when
/// available, otherwise fall back to an isolated per-filter connector.
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_compact(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry
            .register(
                "openai_responses_compact",
                praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                    praxis_ai_apis::openai::CompactFilter::from_config_with_client(config, client.clone())
                })),
            )
            .unwrap_or_else(|_| panic!("duplicate filter name: 'openai_responses_compact'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "openai_responses_compact" => praxis_ai_apis::openai::CompactFilter::from_config
        );
    }
}

/// Register `openai_file_search_callout` with the shared client when
/// available, otherwise fall back to an isolated per-filter connector.
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_file_search_callout(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry
            .register(
                "openai_file_search_callout",
                praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                    praxis_ai_apis::openai::FileSearchCalloutFilter::from_config_with_client(config, client.clone())
                })),
            )
            .unwrap_or_else(|_| panic!("duplicate filter name: 'openai_file_search_callout'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "openai_file_search_callout" => praxis_ai_apis::openai::FileSearchCalloutFilter::from_config
        );
    }
}

/// Register `openai_web_search` with the shared client when
/// available, otherwise fall back to an isolated per-filter connector.
#[expect(clippy::panic, reason = "matches register_filters! macro convention")]
fn register_web_search(registry: &mut FilterRegistry, subrequest_client: Option<&SubRequestClient>) {
    if let Some(client) = subrequest_client {
        let client = client.clone();
        registry
            .register(
                "openai_web_search",
                praxis_filter::FilterFactory::Http(std::sync::Arc::new(move |config| {
                    praxis_ai_apis::openai::WebSearchFilter::from_config_with_client(config, client.clone())
                })),
            )
            .unwrap_or_else(|_| panic!("duplicate filter name: 'openai_web_search'"));
    } else {
        praxis_filter::register_filters!(
            @register registry,
            http "openai_web_search" => praxis_ai_apis::openai::WebSearchFilter::from_config
        );
    }
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
        let expected = [
            "ai_guardrails",
            "openai_responses_validate",
            "responses_to_chat_completions",
            "a2a",
            "intelligent_route",
            "provider_route",
            "credential_inject",
            "anthropic_validate",
            "anthropic_web_search",
            "request_id",
            "aws_sigv4_sign",
        ];
        for name in expected {
            assert!(names.contains(&name), "expected {name} in registry");
        }
    }

    /// Assert `name` is registered iff `enabled`.
    fn assert_experimental_registration(names: &[&str], name: &str, enabled: bool) {
        if enabled {
            assert!(
                names.contains(&name),
                "{name} must register when its feature is enabled"
            );
        } else {
            assert!(
                !names.contains(&name),
                "{name} must not register when its feature is disabled"
            );
        }
    }

    /// Experimental filters register only when their cargo feature is enabled.
    #[test]
    fn build_ai_registry_gates_experimental_filters() {
        let registry = build_ai_registry();
        let names = registry.available_filters();

        assert_experimental_registration(&names, "http_callout", cfg!(feature = "http-callout-filter"));
        assert_experimental_registration(&names, "azure_ad", cfg!(feature = "azure-ad-filter"));
        assert_experimental_registration(&names, "gcp_adc", cfg!(feature = "gcp-adc-filter"));
        assert_experimental_registration(&names, "token_rate_limit", cfg!(feature = "token-rate-limit-filter"));
    }

    #[test]
    fn build_ai_registry_marks_security_filters() {
        let registry = build_ai_registry();
        assert!(registry.is_security_filter("provider_route"));
        assert!(registry.is_security_filter("credential_inject"));
        assert!(registry.is_security_filter("aws_sigv4_sign"));
        #[cfg(feature = "azure-ad-filter")]
        assert!(registry.is_security_filter("azure_ad"));
        #[cfg(feature = "gcp-adc-filter")]
        assert!(registry.is_security_filter("gcp_adc"));
    }
}
