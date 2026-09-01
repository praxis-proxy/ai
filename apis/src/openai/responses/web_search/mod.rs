// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Web search filter for the Responses API agentic loop.
//!
//! Operates in two phases within the `iterative_request_router`:
//!
//! 1. **Response path** (`on_response_body`): detects `web_search_call` items in [`ResponsesState::web_search_calls`]
//!    and writes `openai_web_search.action = "loop"` to [`filter_results`] for the IRR step transition.
//! 2. **Request path** (`on_request_body`, re-entry): executes pending web searches via [`SearchClient`] and appends
//!    results to `messages`, `persisted_messages`, and `accumulated_output`.
//!
//! # Pipeline dependencies
//!
//! - **`openai_agentic_loop`** must run before this filter in the response phase (after in YAML order) to extract
//!   `web_search_call` items from the model response into [`ResponsesState::web_search_calls`].
//! - The IRR transition must match `openai_web_search.action = "loop"` and target the same inference step.
//!
//! [`ResponsesState::web_search_calls`]: super::state::ResponsesState
//! [`filter_results`]: HttpFilterContext::filter_results
//! [`SearchClient`]: crate::web_search::SearchClient

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

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use super::state::ResponsesState;
use crate::{
    openai::responses::error::responses_error_rejection,
    web_search::{
        SearchClient, SearchContextSize, SearchOutcome, SearchResult, WebSearchFilterConfig, build_config,
        format_search_results,
    },
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter results key for the loop control action.
const FILTER_RESULT_KEY: &str = "openai_web_search";

/// Action value signalling a loop-back for web search dispatch.
const ACTION_LOOP: &str = "loop";

/// Action value signalling no web search dispatch needed.
const ACTION_DONE: &str = "done";

/// Include value that gates `action.sources` in the output item.
const INCLUDE_ACTION_SOURCES: &str = "web_search_call.action.sources";

// -----------------------------------------------------------------------------
// WebSearchFilter
// -----------------------------------------------------------------------------

/// Web search filter for model-driven `web_search_call` dispatch.
///
/// Detects pending web search calls in the response phase and
/// executes them on re-entry via the `iterative_request_router`
/// agentic loop.
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
/// provider_failure_mode: closed
/// status_on_error: 502
/// max_body_bytes: 67108864
/// ```
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
    /// Uses an isolated [`SubRequestClient`] with a default pool
    /// size of 4. Prefer [`from_config_with_client`] when a shared
    /// client is available.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or the
    /// search client cannot be constructed.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    /// [`from_config_with_client`]: Self::from_config_with_client
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client =
            crate::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None));
        Self::build(config, client)
    }

    /// Create a filter using the shared [`SubRequestClient`].
    ///
    /// The shared client inherits the server-level pool size and
    /// connection limits from the runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid or the
    /// search client cannot be constructed.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    /// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: crate::subrequest::SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::build(config, client)
    }

    /// Shared constructor body for [`from_config`](Self::from_config) and
    /// [`from_config_with_client`](Self::from_config_with_client).
    fn build(
        config: &serde_yaml::Value,
        subrequest_client: crate::subrequest::SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: WebSearchFilterConfig = parse_filter_config("openai_web_search", config)?;
        let validated = build_config("openai_web_search", &cfg)?;
        let search_client = SearchClient::from_config("openai_web_search", &validated, subrequest_client)?;
        Ok(Box::new(Self {
            search_client,
            default_context_size: validated.default_context_size,
            max_body_bytes: validated.max_body_bytes,
        }))
    }

    /// Execute a single web search call and append results to state.
    async fn execute_single_search(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        call: &Value,
        context_size: SearchContextSize,
    ) -> Result<(), FilterAction> {
        let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
        let query = call.get("action").and_then(|a| a.get("query")).and_then(Value::as_str);

        let Some(query) = query else {
            warn!(call_id, "web_search_call missing action.query, skipping");
            append_result(ctx, call_id, "incomplete", "", &[]);
            return Ok(());
        };

        let results = resolve_search_outcome(&self.search_client, query, context_size, call_id, false).await?;

        append_result(ctx, call_id, "completed", query, &results);
        Ok(())
    }
}

#[async_trait]
impl HttpFilter for WebSearchFilter {
    fn name(&self) -> &'static str {
        "openai_web_search"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if state.web_search_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let context_size = ctx
            .get_metadata("tool_parse.search_context_size")
            .map_or(self.default_context_size, SearchContextSize::from_str_or_default);

        let calls: Vec<Value> = state.web_search_calls.clone();
        debug!(count = calls.len(), "executing pending web search calls");

        for call in &calls {
            if let Err(rejection) = self.execute_single_search(ctx, call, context_size).await {
                return Ok(rejection);
            }
        }

        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.web_search_calls.clear();
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if state.web_search_calls.is_empty() {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }

        debug!(
            count = state.web_search_calls.len(),
            "web search calls pending, signaling loop"
        );
        set_action(ctx, ACTION_LOOP)?;
        Ok(FilterAction::Continue)
    }
}

/// Append search results to [`ResponsesState`].
fn append_result(ctx: &mut HttpFilterContext<'_>, call_id: &str, status: &str, query: &str, results: &[SearchResult]) {
    let include_sources = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(|s| s.include.iter().any(|v| v == INCLUDE_ACTION_SOURCES));

    let output_item = build_output_item(call_id, status, query, results, include_sources);
    let tool_result = build_tool_result_message(call_id, results);

    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.messages.push(tool_result.clone());
        state.persisted_messages.push(tool_result);
        state.accumulated_output.push(output_item);
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Execute a search and resolve its outcome to a result list.
///
/// Returns `Err(FilterAction)` when the search is rejected under
/// closed failure mode.
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
#[cfg_attr(not(test), expect(dead_code, reason = "reserved for per-call status tracking"))]
pub(crate) fn emit_status(ctx: &mut HttpFilterContext<'_>, call_id: &str, status: &str) {
    let key = format!("web_search_call_{call_id}");
    let results = ctx.filter_results.entry("openai_web_search").or_default();
    if results.set(key, status.to_owned()).is_ok() {
        debug!(call_id, status, "emitted web_search_call status");
    }
}

/// Build a `web_search_call` output item for the response.
///
/// `action.sources` is only included when `include_sources` is true,
/// matching the `web_search_call.action.sources` include gate.
pub(crate) fn build_output_item(
    call_id: &str,
    status: &str,
    query: &str,
    results: &[SearchResult],
    include_sources: bool,
) -> Value {
    let mut action = serde_json::json!({
        "type": "search",
        "query": query,
    });

    if include_sources {
        let sources: Vec<Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "type": "url",
                    "url": r.url,
                })
            })
            .collect();
        if let Some(obj) = action.as_object_mut() {
            obj.insert("sources".to_owned(), Value::Array(sources));
        }
    }

    serde_json::json!({
        "type": "web_search_call",
        "id": call_id,
        "status": status,
        "action": action,
    })
}

/// Build a tool result message to append to conversation history.
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

/// Write the loop control action to filter results.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_RESULT_KEY)
        .or_default()
        .set("action", action)?;
    Ok(())
}
