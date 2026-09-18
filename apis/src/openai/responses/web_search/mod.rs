// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Web search filter for the Responses API agentic loop.
//!
//! Executes calls prepared by `openai_agentic_loop` during the request-body
//! phase of the next `iterative_request_router` iteration. The owner performs
//! response classification, admission, and the sole loop transition; this
//! dispatcher only executes the admitted searches via [`SearchClient`] and
//! appends their results to request-scoped state.
//!
//! # Pipeline dependencies
//!
//! - **`openai_agentic_loop`** must run after this filter in request order, so response order is `openai_agentic_loop`
//!   then `openai_web_search`.
//! - The IRR transition must match `openai_agentic_loop.action = "loop"` and target the same inference step.
//!
//! [`ResponsesState::web_search_calls`]: super::state::ResponsesState
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

use std::mem;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES,
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use super::state::{
    ResponsesState, consumed_builtin_tool_calls_before_current_round, current_round_tool_call_admissions,
};
use crate::web_search::{
    OpenAiWebSearchConfig, SEARCH_UNAVAILABLE, SearchClient, SearchContextSize, SearchOutcome, SearchResult,
    build_config, config::MAX_CALLS_PER_ROUND, format_search_results, is_web_search_tool_type,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Step-local metadata carrying the configured response fan-out cap to the owner.
const MAX_CALLS_METADATA: &str = "responses.web_search_max_calls_per_round";

/// Include value that gates `action.sources` in the output item.
const INCLUDE_ACTION_SOURCES: &str = "web_search_call.action.sources";

/// Server-side hard cap on search queries dispatched in one continuation.
///
/// Bounds the number of (potentially paid) provider requests issued per
/// re-entry even when the client omits `max_tool_calls`. One
/// `web_search_call` may carry several queries, so this caps the
/// query fan-out rather than the number of tool calls. The tool-call allowance
/// is bounded separately by the client's `max_tool_calls` and by the configured
/// `max_calls_per_round`.
const MAX_WEB_SEARCH_QUERIES_PER_CONTINUATION: usize = 64;

/// Model-facing result for a malformed call without `action.query`.
const MISSING_QUERY_OUTPUT: &str = "Web search could not run because the query was missing.";

/// Model-facing result when the response-wide tool budget is exhausted.
const TOOL_LIMIT_OUTPUT: &str = "Web search was not executed because max_tool_calls was exhausted.";

/// Borrowed inputs for one request-side web-search dispatch batch.
struct PendingSearchBatch<'a> {
    /// Calls retained from the current model round.
    calls: &'a [Value],
    /// Ordered response-wide budget decisions, when the client set a limit.
    admissions: Option<&'a [bool]>,
    /// Remaining built-in tool calls allowed by the client's `max_tool_calls`.
    call_budget: usize,
    /// Remaining provider requests (queries) allowed by the server-side fan-out cap.
    query_budget: usize,
    /// Search result size requested for this response.
    context_size: SearchContextSize,
}

/// A normalized hosted search action ready for dispatch.
///
/// The owned action is moved into the public output item after the bridge has
/// serialized its arguments; no query data is cloned between those paths.
struct SearchRequest<'a> {
    /// Queries in provider-supplied order.
    queries: Vec<&'a str>,
    /// Canonical client-visible search action.
    action: Value,
}

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
/// max_calls_per_round: 32
/// ```
pub struct WebSearchFilter {
    /// The search client for executing queries.
    search_client: SearchClient,
    /// Default search context size.
    default_context_size: SearchContextSize,
    /// Maximum calls accepted from one model response.
    max_calls_per_round: usize,
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
        let cfg: OpenAiWebSearchConfig = parse_filter_config("openai_web_search", config)?;
        if cfg.max_calls_per_round == 0 || cfg.max_calls_per_round > MAX_CALLS_PER_ROUND {
            return Err(
                format!("openai_web_search: max_calls_per_round must be between 1 and {MAX_CALLS_PER_ROUND}").into(),
            );
        }
        let max_calls_per_round = cfg.max_calls_per_round;
        let validated = build_config("openai_web_search", &cfg.into_shared())?;
        let search_client = SearchClient::from_config("openai_web_search", &validated, subrequest_client)?;
        Ok(Box::new(Self {
            search_client,
            default_context_size: validated.default_context_size,
            max_calls_per_round,
        }))
    }

    /// Execute a single web search call and append its outcome to state.
    ///
    /// A provider failure never rejects the Response. The model instead
    /// receives a truthful `failed` `web_search_call` plus a bounded failure
    /// message — bridged as a backend-valid `function_call`/`function_call_output`
    /// pair — so the agentic loop can continue.
    ///
    /// `index` is the call's position within the pending queue. It keeps the
    /// synthetic bridge `call_id` unique even when the hosted source ids
    /// collide or are absent (issue #808).
    ///
    /// Returns how many queries were dispatched.
    async fn execute_single_search(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        call: &Value,
        index: usize,
        context_size: SearchContextSize,
        query_allowance: usize,
    ) -> usize {
        let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
        let Some(request) = parse_search_request(call, call_id) else {
            warn!(
                call_id,
                "web_search_call missing valid action.queries or action.query, skipping"
            );
            let bridge = bridge_call_id(call_id, &[], index);
            let ids = SearchCallIds::new(call_id, &bridge, index);
            append_incomplete(ctx, &ids);
            return 0;
        };

        let bridge = bridge_call_id(call_id, &request.queries, index);
        let ids = SearchCallIds::new(call_id, &bridge, index);
        let mut results = Vec::new();
        let mut dispatched = 0_usize;
        for query in request.queries.iter().take(query_allowance) {
            dispatched = dispatched.saturating_add(1);
            match self.search_client.search(query, Some(context_size)).await {
                SearchOutcome::Results(mut query_results) => results.append(&mut query_results),
                SearchOutcome::Failed => {
                    warn!(
                        call_id,
                        "web search provider failed; continuing with a failed tool result"
                    );
                    let status = if results.is_empty() { "failed" } else { "incomplete" };
                    append_search_turn(ctx, &ids, status, request, &results, SEARCH_UNAVAILABLE);
                    return dispatched;
                },
            }
        }
        let status = if dispatched == request.queries.len() {
            "completed"
        } else {
            "incomplete"
        };
        append_search_turn(ctx, &ids, status, request, &results, "Web search not performed.");
        dispatched
    }

    /// Execute admitted web search `calls` within the batch budgets, then
    /// update the cumulative execution count and clear the pending queue.
    ///
    /// Calls rejected by the response-wide ordered admission pass receive a
    /// failed result and force local completion without another model round.
    /// A call that can spend neither a tool-call unit nor a single provider
    /// request is surfaced as incomplete without reaching the provider; a call
    /// that gets only part of its queries is dispatched and reported incomplete
    /// with the results it obtained.
    async fn execute_pending_searches(&self, ctx: &mut HttpFilterContext<'_>, batch: PendingSearchBatch<'_>) -> bool {
        let mut calls_dispatched = 0_usize;
        let mut queries_dispatched = 0_usize;
        let mut tool_limit_exceeded = false;
        for (index, call) in batch.calls.iter().enumerate() {
            if batch
                .admissions
                .and_then(|values| values.get(index))
                .is_some_and(|admitted| !admitted)
            {
                append_tool_limit_exceeded(ctx, call, index);
                tool_limit_exceeded = true;
                continue;
            }
            let query_allowance = batch.query_budget.saturating_sub(queries_dispatched);
            if calls_dispatched >= batch.call_budget || query_allowance == 0 {
                append_excess_incomplete(ctx, call, index);
                continue;
            }
            let dispatched = self
                .execute_single_search(ctx, call, index, batch.context_size, query_allowance)
                .await;
            if dispatched > 0 {
                calls_dispatched = calls_dispatched.saturating_add(1);
                queries_dispatched = queries_dispatched.saturating_add(dispatched);
            }
        }

        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.web_search_calls_executed = state
                .web_search_calls_executed
                .saturating_add(u32::try_from(calls_dispatched).unwrap_or(u32::MAX));
            state.web_search_calls.clear();
        }
        tool_limit_exceeded
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
        // Buffer up to the absolute JSON ceiling; the pipeline's body_limits
        // governs the real raw-request cap (merged across sibling filters and
        // clamped to the transport ceiling by praxis core).
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "budgeted dispatch and local terminal response form one lifecycle"
    )]
    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        ctx.set_metadata(MAX_CALLS_METADATA, self.max_calls_per_round.to_string());

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if state.web_search_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let admissions = state
            .max_tool_calls
            .map(|_| current_round_tool_call_admissions(state, &state.web_search_calls));
        let context_size = ctx
            .get_metadata("tool_parse.search_context_size")
            .or_else(|| web_search_context_size_from_state(state))
            .map_or(self.default_context_size, SearchContextSize::from_str_or_default);

        // Compute the remaining budget while the shared immutable borrow is
        // still live, then move the web search calls out instead of cloning
        // and then removing them from state.
        let call_budget = remaining_web_search_budget(state);
        let calls: Vec<Value> = ctx
            .extensions
            .get_mut::<ResponsesState>()
            .map(|state| mem::take(&mut state.web_search_calls))
            .unwrap_or_default();

        let execute_count = admissions.as_ref().map_or(calls.len(), |values| {
            values.iter().filter(|admitted| **admitted).count()
        });
        let rejected_count = calls.len().saturating_sub(execute_count);
        debug!(
            count = execute_count,
            rejected = rejected_count,
            "executing pending web search calls"
        );

        let tool_limit_exceeded = self
            .execute_pending_searches(
                ctx,
                PendingSearchBatch {
                    calls: &calls,
                    admissions: admissions.as_deref(),
                    call_budget,
                    query_budget: MAX_WEB_SEARCH_QUERIES_PER_CONTINUATION,
                    context_size,
                },
            )
            .await;
        if tool_limit_exceeded && let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.deferred_tool_limit_completion = true;
        }
        Ok(FilterAction::Continue)
    }
}

/// Return the response fan-out cap this dispatcher published for the owner.
pub(crate) fn configured_max_calls_per_round(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    ctx.get_metadata(MAX_CALLS_METADATA)?.parse().ok()
}

/// Recover per-request search context after IRR resets step-local metadata.
fn web_search_context_size_from_state(state: &ResponsesState) -> Option<&str> {
    state.tools.iter().find_map(|tool| {
        let tool_type = tool.get("type").and_then(Value::as_str)?;
        if is_web_search_tool_type(tool_type) {
            tool.get("search_context_size").and_then(Value::as_str)
        } else {
            None
        }
    })
}

/// Parse a hosted search action while preserving legacy compatibility.
///
/// A valid, non-empty `queries` array is authoritative. If it is present but
/// invalid, do not fall back to `query`: doing so would execute an action other
/// than the one the provider supplied. When both forms are valid, `query` is
/// intentionally not appended to the current array, preventing duplicate
/// dispatch of the same search.
fn parse_search_request<'a>(call: &'a Value, call_id: &str) -> Option<SearchRequest<'a>> {
    let action = call.get("action")?;
    let legacy_query = action.get("query").and_then(Value::as_str);
    match action.get("queries") {
        Some(Value::Array(values)) if !values.is_empty() => {
            let queries = values.iter().map(Value::as_str).collect::<Option<Vec<_>>>()?;
            if legacy_query.is_some() {
                warn!(
                    call_id,
                    "web_search_call contains deprecated action.query and action.queries; using action.queries"
                );
            }
            Some(SearchRequest {
                action: serde_json::json!({"type": "search", "queries": queries}),
                queries,
            })
        },
        Some(_) => None,
        None => legacy_query.map(|query| SearchRequest {
            queries: vec![query],
            action: serde_json::json!({"type": "search", "query": query}),
        }),
    }
}

/// Public and bridge identifiers for one appended web-search result.
///
/// `public` is the hosted, client-facing `web_search_call.id` retained on the
/// public output item. `bridge` is the bounded, deterministic id used for the
/// backend-valid `function_call`/`function_call_output` pair, since the raw
/// hosted id can exceed the `OpenResponses` 64-character `call_id` limit (issue
/// #808).
struct SearchCallIds<'a> {
    /// Client-facing `web_search_call.id` for the public output item.
    public: &'a str,
    /// Bounded, backend-valid id for the synthetic bridge pair.
    bridge: &'a str,
    /// Position within the current model round.
    index: usize,
}

impl<'a> SearchCallIds<'a> {
    /// Pair one provider-facing ID with its bounded bridge ID and round position.
    fn new(public: &'a str, bridge: &'a str, index: usize) -> Self {
        Self { public, bridge, index }
    }
}

/// Remaining web searches this continuation may dispatch.
///
/// This is only the tool-call dimension: it counts logical built-in tool calls,
/// so a three-query `web_search_call` costs one unit. The (potentially paid)
/// provider requests such a call issues are bounded separately by
/// [`MAX_WEB_SEARCH_QUERIES_PER_CONTINUATION`]. When the client omits
/// `max_tool_calls`, only that server cap constrains the dispatch.
///
/// `max_tool_calls` is a single budget shared across *every* built-in tool
/// type, so the allowance is the declared maximum minus all built-in calls
/// already admitted in prior model rounds. Calls are reconstructed from retained
/// output occurrences rather than provider execution counts, so incomplete
/// calls remain charged and a current-round web execution cannot be counted
/// again when a sibling dispatcher evaluates ordered admission. Matching calls
/// copied between state owners are reconciled by occurrence multiplicity,
/// mirroring
/// the owner-side ordered admission applied to file-search assignments.
fn remaining_web_search_budget(state: &ResponsesState) -> usize {
    let used = consumed_builtin_tool_calls_before_current_round(state);
    state.max_tool_calls.map_or(usize::MAX, |max| {
        usize::try_from(max).unwrap_or(usize::MAX).saturating_sub(used)
    })
}

/// Surface an over-budget web search call as incomplete without dispatching.
///
/// Preserves the requested query in the output item so the model can see which
/// search was declined, matching the missing-query incomplete shape. No
/// provider request is issued, so no budget is charged. `index` keeps the
/// bridge `call_id` unique, mirroring [`WebSearchFilter::execute_single_search`].
fn append_excess_incomplete(ctx: &mut HttpFilterContext<'_>, call: &Value, index: usize) {
    let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
    let Some(request) = parse_search_request(call, call_id) else {
        let bridge = bridge_call_id(call_id, &[], index);
        let ids = SearchCallIds::new(call_id, &bridge, index);
        append_incomplete(ctx, &ids);
        return;
    };
    let bridge = bridge_call_id(call_id, &request.queries, index);
    let ids = SearchCallIds::new(call_id, &bridge, index);
    append_search_turn(ctx, &ids, "incomplete", request, &[], "Web search not performed.");
}

/// Append a completed search turn to [`ResponsesState`].
///
/// An empty `results` slice is a successful zero-result search: the model
/// receives `No search results found.` and the public item stays `completed`.
/// A non-`completed` status with no results (an over-budget call that was never
/// dispatched, or one whose every provider request failed) threads through a
/// truthful `failure_output` bridge, so the model-facing history never
/// contradicts the client-visible output item or pollutes durable rehydration
/// history. A partially dispatched call bridges the results it did gather. A
/// missing-query call uses the more specific [`append_incomplete`] instead.
fn append_search_turn(
    ctx: &mut HttpFilterContext<'_>,
    ids: &SearchCallIds<'_>,
    status: &str,
    request: SearchRequest<'_>,
    results: &[SearchResult],
    failure_output: &'static str,
) {
    let include_sources = include_action_sources(ctx);
    let bridge = build_tool_result_messages(ids.bridge, status, &request.action, results, failure_output);
    let output_item = build_output_item(ids.public, status, request.action, results, include_sources);
    push_search_turn(ctx, output_item, bridge, ids.index);
}

/// Append a malformed search turn to [`ResponsesState`].
///
/// The public item remains `incomplete`, while the backend-valid bridge carries
/// the missing arguments and an explicit failure message. This prevents the
/// next inference iteration and persisted replay from treating a missing query
/// as a successful search with zero results.
fn append_incomplete(ctx: &mut HttpFilterContext<'_>, ids: &SearchCallIds<'_>) {
    let include_sources = include_action_sources(ctx);
    let output_item = build_output_item(
        ids.public,
        "incomplete",
        serde_json::json!({"type": "search", "query": ""}),
        &[],
        include_sources,
    );
    let bridge = build_incomplete_tool_result_messages(ids.bridge);
    push_search_turn(ctx, output_item, bridge, ids.index);
}

/// Append a bounded failure for a call rejected by `max_tool_calls`.
fn append_tool_limit_exceeded(ctx: &mut HttpFilterContext<'_>, call: &Value, index: usize) {
    let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
    let Some(request) = parse_search_request(call, call_id) else {
        let bridge = bridge_call_id(call_id, &[], index);
        let ids = SearchCallIds::new(call_id, &bridge, index);
        append_incomplete(ctx, &ids);
        return;
    };
    let bridge = bridge_call_id(call_id, &request.queries, index);
    let ids = SearchCallIds::new(call_id, &bridge, index);
    append_search_turn(ctx, &ids, "failed", request, &[], TOOL_LIMIT_OUTPUT);
}

/// Whether `action.sources` should be included in output items, per the
/// `web_search_call.action.sources` include gate.
fn include_action_sources(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .is_some_and(|s| s.include.iter().any(|v| v == INCLUDE_ACTION_SOURCES))
}

/// Push a search turn — public output item plus the model-facing bridge pair —
/// into state.
///
/// `bridge` is the backend-valid `function_call`/`function_call_output` pair.
/// The per-element clone is required: `messages` and `persisted_messages` are
/// distinct owners of the bridge messages. The public `output_item` is upserted
/// so the placeholder accumulated during the response phase is replaced in
/// place, never duplicated.
fn push_search_turn(ctx: &mut HttpFilterContext<'_>, output_item: Value, bridge: [Value; 2], index: usize) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.messages.extend(bridge.iter().cloned());
        state.persisted_messages.extend(bridge);
        // Record execution provenance keyed on the item id `stream_events` reads:
        // this replaces the model's placeholder with an executed result, so only
        // now may the search's lifecycle be synthesized. A placeholder copied into
        // `accumulated_output` by a failed round never reaches here.
        if let Some(id) = output_item.get("id").and_then(Value::as_str) {
            state.locally_executed_output_items.insert(id.to_owned());
        }
        let round_start = state
            .current_round_output_start
            .unwrap_or(state.accumulated_output.len());
        upsert_output_item(&mut state.accumulated_output, round_start, index, output_item);
    }
}

/// Replace one current-round `web_search_call` by model position, or append it.
///
/// The response phase (`agentic_loop::collect_output_items`) already
/// accumulated each model placeholder in output order. Updating the indexed
/// placeholder keeps duplicate and absent provider IDs distinct. When no
/// current-round placeholder exists (isolated unit contexts), append instead.
fn upsert_output_item(accumulated: &mut Vec<Value>, round_start: usize, index: usize, output_item: Value) {
    if let Some(slot) = accumulated.get_mut(round_start..).and_then(|items| {
        items
            .iter_mut()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call"))
            .nth(index)
    }) {
        *slot = output_item;
        return;
    }
    accumulated.push(output_item);
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

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
    mut action: Value,
    results: &[SearchResult],
    include_sources: bool,
) -> Value {
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

/// Build the backend-valid continuation for a search.
///
/// A hosted `web_search_call` item is not a valid `OpenResponses` `input`
/// type (see issue #808), so the model-facing history bridges the result
/// through a synthetic `function_call` + `function_call_output` pair —
/// mirroring [`file_search_callout`](super::file_search_callout). The
/// public `web_search_call` output item is emitted separately by
/// [`build_output_item`] and only reaches `accumulated_output`.
///
/// `status` is threaded through so a call that was never dispatched
/// (over-budget or missing a query, `status != "completed"`) carries a
/// truthful `Web search not performed.` output instead of a fabricated
/// `No search results found.` result, keeping the model-facing bridge
/// consistent with the client-visible incomplete output item. A call that did
/// reach the provider keeps the results it gathered even when its remaining
/// queries were clipped or failed, so `failure_output` only applies to an empty
/// result set.
///
/// `call_id` must be a bounded, backend-valid identifier from
/// [`bridge_call_id`]: the raw hosted id can exceed the `OpenResponses`
/// 64-character `call_id` limit.
pub(crate) fn build_tool_result_messages(
    call_id: &str,
    status: &str,
    action: &Value,
    results: &[SearchResult],
    failure_output: &'static str,
) -> [Value; 2] {
    let content = if !results.is_empty() {
        format_search_results(results)
    } else if status == "completed" {
        "No search results found.".to_owned()
    } else {
        failure_output.to_owned()
    };
    let arguments = search_arguments(action).to_string();

    [
        serde_json::json!({
            "type": "function_call",
            "call_id": call_id,
            "name": "web_search",
            "arguments": arguments,
            "status": "completed",
        }),
        serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": content,
        }),
    ]
}

/// Derive the private function arguments from the client-visible action.
fn search_arguments(action: &Value) -> Value {
    match action.get("queries") {
        Some(queries) => serde_json::json!({"queries": queries}),
        None => serde_json::json!({"query": action.get("query").and_then(Value::as_str).unwrap_or_default()}),
    }
}

/// Build the backend-valid continuation for a call missing `action.query`.
///
/// The synthetic function call is fully generated, so its status is
/// `completed`; the empty arguments and explicit output truthfully describe
/// the incomplete hosted-tool execution.
pub(crate) fn build_incomplete_tool_result_messages(call_id: &str) -> [Value; 2] {
    [
        serde_json::json!({
            "type": "function_call",
            "call_id": call_id,
            "name": "web_search",
            "arguments": "{}",
            "status": "completed",
        }),
        serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": MISSING_QUERY_OUTPUT,
        }),
    ]
}

/// FNV-1a offset basis for deterministic bridge identities.
const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;

/// Derive a deterministic, bounded `call_id` for the synthetic bridge.
///
/// A hosted `web_search_call.id` is unbounded, but the synthetic
/// `function_call` `call_id` must stay within the `OpenResponses`
/// 64-character limit or a conforming backend rejects the continuation
/// (issue #808) — mirroring the bounded ids in
/// [`file_search_callout`](super::file_search_callout).
///
/// `index` is the call's position in the pending queue, guaranteeing distinct
/// ids even when `source_id` values collide or are absent — otherwise multiple
/// bridges would share one `call_id` and their `function_call_output` pairing
/// would be ambiguous. The `ws_{index}_{hash:016x}` form is at most 40 bytes.
pub(crate) fn bridge_call_id(source_id: &str, queries: &[&str], index: usize) -> String {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let mut hash = FNV_OFFSET_BASIS;
    for part in std::iter::once(source_id).chain(queries.iter().copied()) {
        for byte in part.as_bytes().iter().copied().chain(std::iter::once(0xFF)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    format!("ws_{index}_{hash:016x}")
}
