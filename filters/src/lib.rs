// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(unreachable_pub, reason = "migration: visibility will be tightened")]

//! AI filter implementations for Praxis.
//!
//! Contains agentic protocol filters (A2A, MCP), guardrails,
//! inference routing, prompt enrichment, and token usage handling.

pub mod agentic;
pub mod aws;
#[cfg(feature = "azure-ad-filter")]
pub mod azure;
#[cfg(feature = "http-callout-filter")]
pub mod callout;
#[cfg(feature = "gcp-adc-filter")]
pub mod gcp;
pub mod guardrails;
pub mod inference;
#[cfg(feature = "opentelemetry")]
mod opentelemetry;
pub mod prompt_enrich;
mod register;
pub mod routing;
mod time_to_first_token;
#[cfg(feature = "token-rate-limit-filter")]
mod token_rate_limit;
mod token_usage;

pub use agentic::{a2a::A2aFilter, mcp::McpFilter};
pub use aws::Sigv4SignFilter;
#[cfg(feature = "azure-ad-filter")]
pub use azure::AzureAdFilter;
#[cfg(feature = "http-callout-filter")]
pub use callout::HttpCalloutFilter;
#[cfg(feature = "gcp-adc-filter")]
pub use gcp::GcpAdcFilter;
pub use guardrails::AiGuardrailsFilter;
pub use inference::ModelToHeaderFilter;
pub use prompt_enrich::PromptEnrichFilter;
pub use register::{build_ai_registry, register_ai_filters};
pub use routing::{CredentialInjectFilter, IntelligentRouteFilter, ProviderRouteFilter};
pub use time_to_first_token::TimeToFirstTokenFilter;
#[cfg(feature = "token-rate-limit-filter")]
pub use token_rate_limit::TokenRateLimitFilter;
pub use token_usage::{TokenCountFilter, TokenUsageHeadersFilter};

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "test utilities")]
pub(crate) mod test_utils {
    use std::sync::LazyLock;

    use http::{HeaderMap, Method, Uri};
    use praxis_core::id::IdGenerator;
    use praxis_filter::{HttpFilterContext, Request, RequestExtensions, Response};

    /// Deterministic ID generator for tests (seed=0).
    static TEST_ID_GENERATOR: LazyLock<IdGenerator> = LazyLock::new(|| IdGenerator::with_seed(0));

    /// Build a minimal request for filter unit tests.
    pub(crate) fn make_request(method: Method, path: &str) -> Request {
        Request {
            method,
            uri: path.parse::<Uri>().expect("invalid URI in test"),
            headers: HeaderMap::new(),
        }
    }

    /// Build a minimal filter context for unit tests.
    #[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
    #[allow(
        clippy::too_many_lines,
        reason = "test context constructor mirrors all context fields"
    )]
    pub(crate) fn make_filter_context(req: &Request) -> HttpFilterContext<'_> {
        HttpFilterContext {
            buffered_request_body: None,
            body_done_indices: Vec::new(),
            branch_iterations: std::collections::HashMap::new(),
            client_addr: None,
            cluster: None,
            current_filter_id: None,
            downstream_tls: false,
            extensions: RequestExtensions::default(),
            executed_filter_indices: Vec::new(),
            extra_request_headers: Vec::new(),
            request_headers_to_remove: Vec::new(),
            request_headers_to_set: Vec::new(),
            filter_metadata: std::collections::HashMap::new(),
            pre_read_mutations: Vec::new(),
            structured_metadata: std::collections::HashMap::new(),
            filter_results: std::collections::HashMap::new(),
            filter_state: std::collections::HashMap::new(),
            health_registry: None,
            id_generator: &TEST_ID_GENERATOR,
            kv_stores: None,
            #[cfg(feature = "praxis-main")]
            session_stores: None,
            metrics_route: None,
            peer_identity: None,
            request: req,
            request_body_bytes: 0,
            request_body_mode: praxis_filter::BodyMode::Stream,
            request_start: std::time::Instant::now(),
            response_body_bytes: 0,
            response_body_mode: praxis_filter::BodyMode::Stream,
            response_header: None,
            response_headers_modified: false,
            subrequest_client: None,
            subrequest_response_mode: praxis_filter::SubRequestResponseMode::Buffered,
            #[cfg(feature = "praxis-main")]
            attempted_endpoints: Vec::new(),
            #[cfg(feature = "praxis-main")]
            retry_policy: None,
            #[cfg(feature = "praxis-main")]
            route_retry_policy: None,
            #[cfg(feature = "praxis-main")]
            cluster_retry_state: None,
            #[cfg(feature = "praxis-main")]
            cluster_retry_state_released: false,
            #[cfg(feature = "praxis-main")]
            endpoint_reselector: None,
            #[cfg(feature = "praxis-main")]
            pinned_endpoint_address: None,
            rewritten_path: None,
            selected_endpoint_index: None,
            time_source: &praxis_core::time::SystemTimeSource,
            upstream: None,
        }
    }

    /// Build a minimal OK response for filter unit tests.
    pub(crate) fn make_response() -> Response {
        Response {
            headers: HeaderMap::new(),
            status: http::StatusCode::OK,
        }
    }
}
