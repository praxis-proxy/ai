// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(unreachable_pub, reason = "migration: visibility will be tightened")]

//! AI filter implementations for Praxis.
//!
//! Contains agentic protocol filters (A2A, MCP), guardrails,
//! inference routing, prompt enrichment, and token usage
//! header injection.

pub mod agentic;
pub mod guardrails;
pub mod inference;
pub mod prompt_enrich;
mod token_count;
mod token_usage_headers;

pub use agentic::{a2a::A2aFilter, mcp::McpFilter};
pub use guardrails::AiGuardrailsFilter;
pub use inference::ModelToHeaderFilter;
pub use prompt_enrich::PromptEnrichFilter;
pub use token_count::TokenCountFilter;
pub use token_usage_headers::TokenUsageHeadersFilter;

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
            filter_results: std::collections::HashMap::new(),
            filter_state: std::collections::HashMap::new(),
            health_registry: None,
            id_generator: &TEST_ID_GENERATOR,
            kv_stores: None,
            request: req,
            request_body_bytes: 0,
            request_body_mode: praxis_filter::BodyMode::Stream,
            request_start: std::time::Instant::now(),
            response_body_bytes: 0,
            response_body_mode: praxis_filter::BodyMode::Stream,
            response_header: None,
            response_headers_modified: false,
            rewritten_path: None,
            selected_endpoint_index: None,
            time_source: &praxis_core::time::SystemTimeSource,
            upstream: None,
            #[cfg(feature = "praxis-main")]
            peer_identity: None,
            #[cfg(feature = "praxis-main")]
            pre_read_mutations: Vec::new(),
            #[cfg(feature = "praxis-main")]
            structured_metadata: std::collections::HashMap::new(),
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
