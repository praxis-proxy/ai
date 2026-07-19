// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Web search filter for the Responses API.
//!
//! Validates configuration and constructs a search client at startup
//! but does not execute searches at runtime. When `tool_dispatch`
//! (#26) and branch re-entrance land in praxis-core, this filter
//! will handle model-driven `web_search_call` dispatch in the
//! agentic loop.

pub(crate) mod config;
pub(crate) mod provider;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::fmt::Write as _;

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use serde_json::Value;
use tracing::{debug, warn};

use self::{
    config::{SearchContextSize, WebSearchFilterConfig, build_config},
    provider::{SearchClient, SearchOutcome, SearchResult},
};
use crate::openai::responses::error::responses_error_rejection;

// -----------------------------------------------------------------------------
// WebSearchFilter
// -----------------------------------------------------------------------------

/// Web search filter for model-driven `web_search_call` dispatch.
///
/// Validates configuration and constructs a search client at startup.
/// At runtime this filter is a passthrough — it does not modify
/// requests or responses. When `tool_dispatch` (#26) and branch
/// re-entrance are available, this filter will execute searches
/// dispatched by the model during the agentic loop.
///
/// # YAML
///
/// ```yaml
/// filter: openai_web_search
/// provider: brave
/// api_key: ${WEB_SEARCH_API_KEY}
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_web_search
/// provider: brave
/// api_key: ${WEB_SEARCH_API_KEY}
/// default_context_size: medium
/// timeout_ms: 10000
/// failure_mode: closed
/// status_on_error: 502
/// max_body_bytes: 67108864
/// ```
#[expect(dead_code, reason = "fields used at startup and reserved for tool_dispatch (#26)")]
pub struct WebSearchFilter {
    /// The search client for executing queries.
    search_client: SearchClient,
    /// Default search context size.
    default_context_size: SearchContextSize,
    /// Maximum request body bytes to buffer.
    max_body_bytes: usize,
}

impl WebSearchFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or the
    /// search client cannot be constructed.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: WebSearchFilterConfig = parse_filter_config("openai_web_search", config)?;
        let validated = build_config(&cfg)?;
        let search_client = SearchClient::from_config(&validated)?;
        Ok(Box::new(Self {
            search_client,
            default_context_size: validated.default_context_size,
            max_body_bytes: validated.max_body_bytes,
        }))
    }
}

#[async_trait]
impl HttpFilter for WebSearchFilter {
    fn name(&self) -> &'static str {
        "openai_web_search"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }
}

// -----------------------------------------------------------------------------
// Helpers (pub(crate) for tool_dispatch)
// -----------------------------------------------------------------------------

/// Execute a search and resolve its outcome to a result list.
///
/// Returns `Err(FilterAction)` when the search is rejected under
/// closed failure mode.
#[expect(dead_code, reason = "scaffolded for tool_dispatch (#26)")]
pub(crate) async fn resolve_search_outcome(
    search_client: &SearchClient,
    query: &str,
    context_size: SearchContextSize,
    call_id: &str,
    streaming: bool,
) -> Result<Vec<SearchResult>, FilterAction> {
    match search_client.search(query, Some(context_size)).await {
        SearchOutcome::Results(r) => Ok(r),
        SearchOutcome::Skipped => {
            warn!(call_id, "search skipped (open failure mode)");
            Ok(Vec::new())
        },
        SearchOutcome::Rejected { status } => {
            warn!(call_id, status, "search rejected (closed failure mode)");
            Err(FilterAction::Reject(responses_error_rejection(
                status,
                "server_error",
                "web search provider unavailable",
                streaming,
            )))
        },
    }
}

/// Emit a `web_search_call` status update via filter results.
#[cfg_attr(not(test), expect(dead_code, reason = "scaffolded for tool_dispatch (#26)"))]
pub(crate) fn emit_status(ctx: &mut HttpFilterContext<'_>, call_id: &str, status: &str) {
    let key = format!("web_search_call_{call_id}");
    let results = ctx.filter_results.entry("openai_web_search").or_default();
    if results.set(key, status.to_owned()).is_ok() {
        debug!(call_id, status, "emitted web_search_call status");
    }
}

/// Build a `web_search_call` output item for the response.
#[cfg_attr(not(test), expect(dead_code, reason = "scaffolded for tool_dispatch (#26)"))]
pub(crate) fn build_output_item(call_id: &str, status: &str, query: &str, results: &[SearchResult]) -> Value {
    let mut item = serde_json::json!({
        "type": "web_search_call",
        "id": call_id,
        "status": status,
        "action": {
            "type": "search",
            "query": query,
        },
    });

    if !results.is_empty() {
        let sources: Vec<Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "title": r.title,
                    "url": r.url,
                })
            })
            .collect();
        if let Some(obj) = item.as_object_mut() {
            obj.insert("sources".to_owned(), Value::Array(sources));
        }
    }

    item
}

/// Build a tool result message to append to conversation history.
#[cfg_attr(not(test), expect(dead_code, reason = "scaffolded for tool_dispatch (#26)"))]
pub(crate) fn build_tool_result_message(call_id: &str, results: &[SearchResult]) -> Value {
    let content = if results.is_empty() {
        "No search results found.".to_owned()
    } else {
        format_search_results(results)
    };

    serde_json::json!({
        "type": "web_search_call",
        "id": call_id,
        "status": "completed",
        "output": content,
    })
}

/// Format search results as readable text for the model.
pub(crate) fn format_search_results(results: &[SearchResult]) -> String {
    let mut out = String::with_capacity(results.len() * 200);
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        let _infallible = write!(out, "[{}] {}\n{}\n{}", i + 1, r.title, r.url, r.snippet);
    }
    out
}
