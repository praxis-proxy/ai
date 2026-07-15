// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(unreachable_pub, reason = "migration: visibility will be tightened")]

//! AI provider API types and persistence for Praxis.
//!
//! Contains provider-specific protocol types (OpenAI, Anthropic),
//! request classification, response storage backends, and token
//! usage extraction.

pub mod anthropic;
pub mod classifier;
pub(crate) mod mcp_client;
pub mod openai;
#[cfg(feature = "store")]
pub mod store;
pub mod token_usage;

/// Whether a `Content-Type` header value indicates `text/event-stream`,
/// ignoring parameters (e.g. `; charset=utf-8`) and ASCII case.
pub fn is_event_stream_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("text/event-stream")
}

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

    /// Build a [`FilterRegistry`] with core builtins plus AI API filters
    /// needed by pipeline integration tests.
    ///
    /// [`FilterRegistry`]: praxis_filter::FilterRegistry
    pub(crate) fn make_ai_registry() -> praxis_filter::FilterRegistry {
        let mut registry = praxis_filter::FilterRegistry::with_builtins();
        praxis_filter::register_filters!(
            @register registry,
            http "openai_responses_format" => crate::openai::ResponsesFormatFilter::from_config
        );
        praxis_filter::register_filters!(
            @register registry,
            http "openai_response_store" => crate::openai::ResponseStoreFilter::from_config
        );
        praxis_filter::register_filters!(
            @register registry,
            http "openai_responses_rehydrate" => crate::openai::RehydrateFilter::from_config
        );
        praxis_filter::register_filters!(
            @register registry,
            http "openai_stream_events" => crate::openai::OpenaiStreamEventsFilter::from_config
        );
        registry
    }
}
