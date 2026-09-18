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

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES,
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use super::state::{
    DispatchFailure, ResponsesState, consumed_builtin_tool_calls_before_current_round,
    current_round_web_search_admissions, retained_json_bytes, retained_json_values_bytes,
};
use crate::web_search::{
    OpenAiWebSearchConfig, SEARCH_UNAVAILABLE, SearchClient, SearchContextSize, SearchOutcome, SearchResult,
    build_config, config::MAX_CALLS_PER_ROUND, format_search_results, is_web_search_tool_type,
    provider::MAX_SEARCH_RESPONSE_BYTES,
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Step-local metadata carrying the configured response fan-out cap to the owner.
const MAX_CALLS_METADATA: &str = "responses.web_search_max_calls_per_round";

/// Include value that gates `action.sources` in the output item.
const INCLUDE_ACTION_SOURCES: &str = "web_search_call.action.sources";

/// Server-side hard cap on web searches dispatched in one continuation.
///
/// Bounds the number of (potentially paid) provider requests issued per
/// re-entry even when the client omits `max_tool_calls`. Mirrors the
/// file-search `MAX_PENDING_CALLS` ceiling so the two built-in tools share
/// the same per-continuation fan-out limit. The enclosing
/// `iterative_request_router` deadlines and iteration cap bound the total
/// across continuations.
const MAX_WEB_SEARCH_CALLS_PER_CONTINUATION: usize = 64;

/// Model-facing result for a malformed call without `action.query`.
const MISSING_QUERY_OUTPUT: &str = "Web search could not run because the query was missing.";

/// Model-facing result when the response-wide tool budget is exhausted.
const TOOL_LIMIT_OUTPUT: &str = "Web search was not executed because max_tool_calls was exhausted.";

/// Borrowed inputs for one request-side web-search dispatch batch.
struct PendingSearchBatch<'a> {
    /// Calls retained from the current model round.
    calls: &'a [PendingSearchCall],
    /// Ordered response-wide budget decisions, when the client set a limit.
    admissions: Option<&'a [bool]>,
    /// Remaining provider requests allowed by the cumulative budget.
    provider_budget: usize,
    /// Search result size requested for this response.
    context_size: SearchContextSize,
}

/// Lightweight execution staging for one canonical web-search output item.
/// Only fields needed across the provider await are copied.
struct PendingSearchCall {
    /// Absolute canonical output index replaced by the result.
    output_index: usize,
    /// Position among web-search calls in this model round.
    ordinal: usize,
    /// Public call id.
    id: String,
    /// Search query, absent for malformed calls.
    query: Option<String>,
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
    /// Returns `true` when a provider request was dispatched — a `Results` or
    /// `Failed` outcome, both charged against the call budget — and `false`
    /// when the call was surfaced as incomplete without issuing a request
    /// (a missing query).
    #[expect(clippy::too_many_lines, reason = "budgeted provider dispatch and outcome handling")]
    async fn execute_single_search(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        call: &PendingSearchCall,
        context_size: SearchContextSize,
    ) -> bool {
        let call_id = call.id.as_str();
        let query = call.query.as_deref();

        let Some(query) = query else {
            warn!(call_id, "web_search_call missing action.query, skipping");
            let bridge = bridge_call_id(call_id, "", call.ordinal);
            let ids = SearchCallIds::new(call_id, &bridge, call.output_index);
            append_incomplete(ctx, &ids);
            return false;
        };

        let bridge = bridge_call_id(call_id, query, call.ordinal);
        let ids = SearchCallIds::new(call_id, &bridge, call.output_index);
        let Some(max_response_bytes) = web_search_response_limit(ctx, query) else {
            record_web_search_budget_failure(ctx);
            return false;
        };
        match self
            .search_client
            .search_with_response_limit(query, Some(context_size), max_response_bytes)
            .await
        {
            SearchOutcome::Results(results) => append_result(ctx, &ids, "completed", query, &results),
            SearchOutcome::RetainedLimitExceeded => {
                record_web_search_budget_failure(ctx);
            },
            SearchOutcome::Failed => {
                warn!(
                    call_id,
                    "web search provider failed; continuing with a failed tool result"
                );
                append_failed(ctx, &ids, query);
            },
        }
        true
    }

    /// Execute admitted web search `calls` up to the provider-request `budget`,
    /// then update the cumulative execution count and clear the pending queue.
    ///
    /// Calls rejected by the response-wide ordered admission pass receive a
    /// failed result and force local completion without another model round.
    /// Calls beyond the provider-request `budget` are surfaced as incomplete
    /// without issuing a potentially paid provider request. Only dispatched
    /// calls (a `Results` or `Failed` outcome) are added to the cumulative
    /// provider counter; a missing-query call consumes its model-call admission
    /// but does not issue a provider request.
    #[expect(clippy::too_many_lines, reason = "ordered batch dispatch with terminal budget stop")]
    async fn execute_pending_searches(&self, ctx: &mut HttpFilterContext<'_>, batch: PendingSearchBatch<'_>) -> bool {
        let mut dispatched = 0_usize;
        let mut tool_limit_exceeded = false;
        for (index, call) in batch.calls.iter().enumerate() {
            if ctx
                .extensions
                .get::<ResponsesState>()
                .is_some_and(|state| state.retained_payload_failed)
            {
                break;
            }
            if batch
                .admissions
                .and_then(|values| values.get(index))
                .is_some_and(|admitted| !admitted)
            {
                append_tool_limit_exceeded(ctx, call);
                tool_limit_exceeded = true;
                continue;
            }
            if dispatched >= batch.provider_budget {
                append_excess_incomplete(ctx, call);
                continue;
            }
            if self.execute_single_search(ctx, call, batch.context_size).await {
                dispatched = dispatched.saturating_add(1);
            }
        }

        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.web_search_calls_executed = state
                .web_search_calls_executed
                .saturating_add(u32::try_from(dispatched).unwrap_or(u32::MAX));
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
            .map(|_| current_round_web_search_admissions(state, &state.web_search_calls));
        let context_size = ctx
            .get_metadata("tool_parse.search_context_size")
            .or_else(|| web_search_context_size_from_state(state))
            .map_or(self.default_context_size, SearchContextSize::from_str_or_default);

        // Compute the remaining budget while the shared immutable borrow is
        // still live, then move the web search calls out instead of cloning
        // and then removing them from state.
        let budget = remaining_web_search_budget(state);
        let Some((calls, calls_bytes)) = ctx
            .extensions
            .get_mut::<ResponsesState>()
            .and_then(take_pending_search_calls)
        else {
            record_web_search_budget_failure(ctx);
            return Ok(FilterAction::Continue);
        };

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
                    provider_budget: budget,
                    context_size,
                },
            )
            .await;
        if tool_limit_exceeded && let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.deferred_tool_limit_completion = true;
        }
        drop(calls);
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.release_external_payload_bytes(calls_bytes);
        }
        Ok(FilterAction::Continue)
    }
}

/// Move the pending dispatcher queue to a filter-local owner without making
/// its payload disappear from aggregate accounting.
#[expect(
    clippy::too_many_lines,
    reason = "preflights and stages the minimal indexed search execution plan"
)]
fn take_pending_search_calls(state: &mut ResponsesState) -> Option<(Vec<PendingSearchCall>, usize)> {
    let bytes = state.web_search_calls.iter().try_fold(0_usize, |used, assignment| {
        let item = state.accumulated_output.get(assignment.output_index)?;
        used.checked_add(
            item.get("id")
                .and_then(Value::as_str)
                .map_or("ws_unknown".len(), str::len),
        )?
        .checked_add(
            item.get("action")
                .and_then(|action| action.get("query"))
                .and_then(Value::as_str)
                .map_or(0, str::len),
        )
    })?;
    if !state.can_retain_payload(bytes) || !state.retain_external_payload_bytes(bytes) {
        return None;
    }
    let calls = state
        .web_search_calls
        .iter()
        .map(|assignment| {
            let item = state.accumulated_output.get(assignment.output_index)?;
            Some(PendingSearchCall {
                output_index: assignment.output_index,
                ordinal: assignment.ordinal,
                id: item
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("ws_unknown")
                    .to_owned(),
                query: item
                    .get("action")
                    .and_then(|action| action.get("query"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(calls) = calls else {
        state.release_external_payload_bytes(bytes);
        return None;
    };
    state.web_search_calls.clear();
    Some((calls, bytes))
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
    /// Absolute canonical output index replaced by the result.
    output_index: usize,
}

impl<'a> SearchCallIds<'a> {
    /// Pair one provider-facing ID with its bounded bridge ID and round position.
    fn new(public: &'a str, bridge: &'a str, output_index: usize) -> Self {
        Self {
            public,
            bridge,
            output_index,
        }
    }
}

/// Remaining web searches this continuation may dispatch.
///
/// Intersects the client's remaining `max_tool_calls` allowance with the
/// server-side per-continuation hard cap. When the client omits
/// `max_tool_calls`, only the server cap applies.
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
    let client_remaining = state.max_tool_calls.map_or(usize::MAX, |max| {
        usize::try_from(max).unwrap_or(usize::MAX).saturating_sub(used)
    });
    client_remaining.min(MAX_WEB_SEARCH_CALLS_PER_CONTINUATION)
}

/// Surface an over-budget web search call as incomplete without dispatching.
///
/// Preserves the requested query in the output item so the model can see which
/// search was declined, matching the missing-query incomplete shape. No
/// provider request is issued, so no budget is charged. `index` keeps the
/// bridge `call_id` unique, mirroring [`WebSearchFilter::execute_single_search`].
fn append_excess_incomplete(ctx: &mut HttpFilterContext<'_>, call: &PendingSearchCall) {
    let query = call.query.as_deref().unwrap_or_default();
    let bridge = bridge_call_id(&call.id, query, call.ordinal);
    let ids = SearchCallIds::new(&call.id, &bridge, call.output_index);
    append_result(ctx, &ids, "incomplete", query, &[]);
}

/// Append a completed search turn to [`ResponsesState`].
///
/// An empty `results` slice is a successful zero-result search: the model
/// receives `No search results found.` and the public item stays `completed`.
/// A non-`completed` status (an over-budget call that was never dispatched)
/// threads through a truthful `Web search not performed.` bridge, so the
/// model-facing history never contradicts the client-visible incomplete output
/// item or pollutes durable rehydration history. A missing-query call uses the
/// more specific [`append_incomplete`] instead.
fn append_result(
    ctx: &mut HttpFilterContext<'_>,
    ids: &SearchCallIds<'_>,
    status: &str,
    query: &str,
    results: &[SearchResult],
) {
    let include_sources = include_action_sources(ctx);
    let Some(source_bytes) = web_search_results_bytes(results) else {
        record_web_search_budget_failure(ctx);
        return;
    };
    if !web_search_construction_fits(ctx, ids, status, query, results, include_sources, source_bytes) {
        record_web_search_budget_failure(ctx);
        return;
    }
    let output_item = build_output_item(ids.public, status, query, results, include_sources);
    let bridge = build_tool_result_messages(ids.bridge, status, query, results);
    push_search_turn(ctx, output_item, bridge, ids.output_index, source_bytes);
}

/// Append a malformed search turn to [`ResponsesState`].
///
/// The public item remains `incomplete`, while the backend-valid bridge carries
/// the missing arguments and an explicit failure message. This prevents the
/// next inference iteration and persisted replay from treating a missing query
/// as a successful search with zero results.
fn append_incomplete(ctx: &mut HttpFilterContext<'_>, ids: &SearchCallIds<'_>) {
    let include_sources = include_action_sources(ctx);
    if !web_search_construction_fits(ctx, ids, "incomplete", "", &[], include_sources, 0) {
        record_web_search_budget_failure(ctx);
        return;
    }
    let output_item = build_output_item(ids.public, "incomplete", "", &[], include_sources);
    let bridge = build_incomplete_tool_result_messages(ids.bridge);
    push_search_turn(ctx, output_item, bridge, ids.output_index, 0);
}

/// Append a failed search turn to [`ResponsesState`].
///
/// The public output item is marked `status: "failed"` and the model receives
/// the bounded [`SEARCH_UNAVAILABLE`] message through a backend-valid
/// `function_call`/`function_call_output` bridge — never a hosted
/// `web_search_call`, which is not a valid `OpenResponses` input (issue #808) —
/// so the agentic loop continues without exposing provider details to the client.
fn append_failed(ctx: &mut HttpFilterContext<'_>, ids: &SearchCallIds<'_>, query: &str) {
    let include_sources = include_action_sources(ctx);
    if !web_search_construction_fits(ctx, ids, "failed", query, &[], include_sources, 0) {
        record_web_search_budget_failure(ctx);
        return;
    }
    let output_item = build_output_item(ids.public, "failed", query, &[], include_sources);
    let bridge = build_failed_tool_result_messages(ids.bridge, query);
    push_search_turn(ctx, output_item, bridge, ids.output_index, 0);
}

/// Append a bounded failure for a call rejected by `max_tool_calls`.
fn append_tool_limit_exceeded(ctx: &mut HttpFilterContext<'_>, call: &PendingSearchCall) {
    let call_id = call.id.as_str();
    let query = call.query.as_deref().unwrap_or_default();
    let bridge_id = bridge_call_id(call_id, query, call.ordinal);
    let include_sources = include_action_sources(ctx);
    let ids = SearchCallIds::new(call_id, &bridge_id, call.output_index);
    if !web_search_construction_fits(ctx, &ids, "failed", query, &[], include_sources, 0) {
        record_web_search_budget_failure(ctx);
        return;
    }
    let output_item = build_output_item(call_id, "failed", query, &[], include_sources);
    let bridge = build_failed_tool_result_messages_with_output(&bridge_id, query, TOOL_LIMIT_OUTPUT);
    push_search_turn(ctx, output_item, bridge, call.output_index, 0);
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
#[expect(
    clippy::too_many_lines,
    reason = "transactional output, history, and provenance commit"
)]
fn push_search_turn(
    ctx: &mut HttpFilterContext<'_>,
    output_item: Value,
    bridge: [Value; 2],
    index: usize,
    source_bytes: usize,
) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        if state.retained_payload_failed {
            return;
        }
        let replaced = state.accumulated_output.get(index);
        let removed = replaced.and_then(retained_json_bytes).unwrap_or(0);
        let bridge_bytes = retained_json_values_bytes(&bridge).unwrap_or(usize::MAX);
        let output_bytes = retained_json_bytes(&output_item);
        let added = output_bytes
            .and_then(|output_bytes| output_bytes.checked_add(bridge_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(output_item.get("id").and_then(Value::as_str).map_or(0, str::len)));
        let staging = output_bytes
            .and_then(|output_bytes| output_bytes.checked_add(bridge_bytes))
            .and_then(|bytes| bytes.checked_add(source_bytes));
        if !added
            .zip(staging)
            .is_some_and(|(added, staging)| state.can_replace_retained_payload(removed, added, staging))
        {
            state.discard_payload_for_budget_error();
            state.dispatch_failure = Some(DispatchFailure {
                status: 502,
                code: "server_error",
                message: "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while appending web-search results"
                    .to_owned(),
            });
            return;
        }
        state.messages.extend(bridge.iter().cloned());
        state.persisted_messages.extend(bridge);
        // Record execution provenance keyed on the item id `stream_events` reads:
        // this replaces the model's placeholder with an executed result, so only
        // now may the search's lifecycle be synthesized. A placeholder copied into
        // `accumulated_output` by a failed round never reaches here.
        if let Some(id) = output_item.get("id").and_then(Value::as_str) {
            state.locally_executed_output_items.insert(id.to_owned());
        }
        upsert_output_item(&mut state.accumulated_output, index, output_item);
    }
}

/// Bound provider parsing, decoded results, and result construction before a
/// paid search is dispatched. Thirty-two response-body owners conservatively
/// cover the raw body, parsed provider JSON, decoded strings, worst-case JSON
/// escaping, public item, formatted bridge, and the two final history owners.
fn web_search_response_limit(ctx: &HttpFilterContext<'_>, query: &str) -> Option<usize> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Some(MAX_SEARCH_RESPONSE_BYTES);
    };
    let Some(limit) = state.retained_payload_limit() else {
        return Some(MAX_SEARCH_RESPONSE_BYTES);
    };
    let current = state.retained_payload_bytes_bounded(limit)?;
    let request_staging = query.len().checked_mul(8)?.checked_add(4_096)?;
    let available = limit.checked_sub(current)?.checked_sub(request_staging)?;
    let response_limit = available / 32;
    (response_limit > 0).then_some(response_limit.min(MAX_SEARCH_RESPONSE_BYTES))
}

/// Raw decoded payload retained by a provider result vector.
fn web_search_results_bytes(results: &[SearchResult]) -> Option<usize> {
    results.iter().try_fold(0_usize, |used, result| {
        used.checked_add(result.title.len())?
            .checked_add(result.url.len())?
            .checked_add(result.snippet.len())
    })
}

/// Admit every result-sized construction owner before formatting or cloning
/// provider strings into JSON values.
#[expect(
    clippy::too_many_arguments,
    reason = "wire identity, result projection, and source ownership are independent inputs"
)]
fn web_search_construction_fits(
    ctx: &HttpFilterContext<'_>,
    ids: &SearchCallIds<'_>,
    status: &str,
    query: &str,
    results: &[SearchResult],
    include_sources: bool,
    source_bytes: usize,
) -> bool {
    let formatting_overhead = results.len().saturating_mul(64);
    let source_projection = if include_sources {
        results.iter().try_fold(0_usize, |used, result| {
            used.checked_add(result.url.len().checked_mul(6)?)?.checked_add(128)
        })
    } else {
        Some(0)
    };
    let output_bound = source_projection.and_then(|sources| {
        sources
            .checked_add(query.len().checked_mul(6)?)?
            .checked_add(ids.public.len().checked_mul(6)?)?
            .checked_add(status.len().checked_mul(6)?)?
            .checked_add(512)
    });
    let bridge_bound = source_bytes
        .checked_add(formatting_overhead)
        .and_then(|bytes| bytes.checked_add(query.len()))
        .and_then(|bytes| bytes.checked_add(ids.bridge.len().saturating_mul(2)))
        .and_then(|bytes| bytes.checked_mul(6))
        .and_then(|bytes| bytes.checked_add(2_048));
    output_bound
        .zip(bridge_bound)
        .and_then(|(output, bridge)| source_bytes.checked_add(output)?.checked_add(bridge.checked_mul(3)?))
        .is_some_and(|additional| {
            ctx.extensions
                .get::<ResponsesState>()
                .is_none_or(|state| state.can_retain_payload(additional))
        })
}

/// Stop dispatch and hand one aggregate failure to the loop owner.
fn record_web_search_budget_failure(ctx: &mut HttpFilterContext<'_>) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
        state.dispatch_failure = Some(DispatchFailure {
            status: 502,
            code: "server_error",
            message:
                "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while executing web search"
                    .to_owned(),
        });
    }
}

/// Replace one current-round `web_search_call` by model position, or append it.
///
/// The response phase (`agentic_loop::collect_output_items`) already
/// accumulated each model placeholder in output order. Updating the indexed
/// placeholder keeps duplicate and absent provider IDs distinct. When no
/// current-round placeholder exists (isolated unit contexts), append instead.
fn upsert_output_item(accumulated: &mut Vec<Value>, index: usize, output_item: Value) {
    if let Some(slot) = accumulated.get_mut(index) {
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
/// consistent with the client-visible incomplete output item.
///
/// `call_id` must be a bounded, backend-valid identifier from
/// [`bridge_call_id`]: the raw hosted id can exceed the `OpenResponses`
/// 64-character `call_id` limit.
pub(crate) fn build_tool_result_messages(
    call_id: &str,
    status: &str,
    query: &str,
    results: &[SearchResult],
) -> [Value; 2] {
    let content = if status != "completed" {
        "Web search not performed.".to_owned()
    } else if results.is_empty() {
        "No search results found.".to_owned()
    } else {
        format_search_results(results)
    };
    let arguments = serde_json::json!({ "query": query }).to_string();

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
pub(crate) fn bridge_call_id(source_id: &str, query: &str, index: usize) -> String {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let mut hash = FNV_OFFSET_BASIS;
    for part in [source_id, query] {
        for byte in part.as_bytes().iter().copied().chain(std::iter::once(0xFF)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    format!("ws_{index}_{hash:016x}")
}

/// Build the backend-valid continuation for a failed search.
///
/// Mirrors [`build_tool_result_messages`] but carries the bounded
/// [`SEARCH_UNAVAILABLE`] notice as the `function_call_output`, so the agentic
/// loop continues with a truthful failure instead of a fabricated empty result.
/// A hosted `web_search_call` is not a valid `OpenResponses` input item (issue
/// #808), so a failure — like a success — must bridge through a synthetic
/// `function_call` + `function_call_output` pair.
///
/// `call_id` must be a bounded, backend-valid identifier from
/// [`bridge_call_id`]: the raw hosted id can exceed the `OpenResponses`
/// 64-character `call_id` limit.
pub(crate) fn build_failed_tool_result_messages(call_id: &str, query: &str) -> [Value; 2] {
    build_failed_tool_result_messages_with_output(call_id, query, SEARCH_UNAVAILABLE)
}

/// Build a failed web-search bridge with a caller-selected bounded message.
fn build_failed_tool_result_messages_with_output(call_id: &str, query: &str, output: &'static str) -> [Value; 2] {
    let arguments = serde_json::json!({ "query": query }).to_string();

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
            "output": output,
        }),
    ]
}
