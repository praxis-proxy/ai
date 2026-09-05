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
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES,
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use super::state::ResponsesState;
use crate::web_search::{
    OpenAiWebSearchConfig, SEARCH_UNAVAILABLE, SearchClient, SearchContextSize, SearchOutcome, SearchResult,
    build_config, format_search_results, is_web_search_tool_type,
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
/// ```
pub struct WebSearchFilter {
    /// The search client for executing queries.
    search_client: SearchClient,
    /// Default search context size.
    default_context_size: SearchContextSize,
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
        let validated = build_config("openai_web_search", &cfg.into_shared())?;
        let search_client = SearchClient::from_config("openai_web_search", &validated, subrequest_client)?;
        Ok(Box::new(Self {
            search_client,
            default_context_size: validated.default_context_size,
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
    async fn execute_single_search(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        call: &Value,
        index: usize,
        context_size: SearchContextSize,
    ) -> bool {
        let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
        let query = call.get("action").and_then(|a| a.get("query")).and_then(Value::as_str);

        let Some(query) = query else {
            warn!(call_id, "web_search_call missing action.query, skipping");
            let bridge = bridge_call_id(call_id, "", index);
            let ids = SearchCallIds {
                public: call_id,
                bridge: &bridge,
            };
            append_incomplete(ctx, &ids);
            return false;
        };

        let bridge = bridge_call_id(call_id, query, index);
        let ids = SearchCallIds {
            public: call_id,
            bridge: &bridge,
        };
        match self.search_client.search(query, Some(context_size)).await {
            SearchOutcome::Results(results) => append_result(ctx, &ids, "completed", query, &results),
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

    /// Execute pending web search `calls` up to `budget`, then update the
    /// cumulative execution count and clear the pending queue.
    ///
    /// Calls beyond `budget` are surfaced as incomplete without issuing a
    /// (potentially paid) provider request, honoring the client's
    /// `max_tool_calls` allowance and the per-continuation server cap. Only
    /// dispatched calls (a `Results` or `Failed` outcome) are charged against
    /// the budget; a missing-query call is surfaced as incomplete without a
    /// provider request and does not consume budget.
    async fn execute_pending_searches(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        calls: &[Value],
        budget: usize,
        context_size: SearchContextSize,
    ) {
        let mut dispatched = 0_usize;
        for (index, call) in calls.iter().enumerate() {
            if dispatched >= budget {
                append_excess_incomplete(ctx, call, index);
                continue;
            }
            if self.execute_single_search(ctx, call, index, context_size).await {
                dispatched = dispatched.saturating_add(1);
            }
        }

        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.web_search_calls_executed = state
                .web_search_calls_executed
                .saturating_add(u32::try_from(dispatched).unwrap_or(u32::MAX));
            state.web_search_calls.clear();
        }
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
        // Buffer up to the absolute JSON ceiling; the pipeline's body_limits
        // governs the real raw-response cap (merged across sibling filters and
        // clamped to the transport ceiling by praxis core).
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
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
            .or_else(|| web_search_context_size_from_state(state))
            .map_or(self.default_context_size, SearchContextSize::from_str_or_default);

        let budget = remaining_web_search_budget(state);
        let calls: Vec<Value> = state.web_search_calls.clone();
        debug!(
            count = calls.len(),
            budget, "executing pending web search calls within budget"
        );

        self.execute_pending_searches(ctx, &calls, budget, context_size).await;
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
}

/// Remaining web searches this continuation may dispatch.
///
/// Intersects the client's remaining `max_tool_calls` allowance with the
/// server-side per-continuation hard cap. When the client omits
/// `max_tool_calls`, only the server cap applies.
///
/// `max_tool_calls` is a single budget shared across *every* built-in tool
/// type, so the allowance is the declared maximum minus all built-in calls
/// already consumed — web searches dispatched across prior iterations
/// ([`ResponsesState::web_search_calls_executed`]) *plus* completed non-web
/// built-in calls (e.g. file search) accumulated this response. Without the
/// second term a mixed pipeline could dispatch a full web-search allowance on
/// top of already-executed file searches and overshoot the client's cap. Web
/// searches are counted through the dedicated counter rather than by scanning
/// [`ResponsesState::accumulated_output`] to avoid the echoed-call/result
/// double count documented on that field; non-web built-in calls have no such
/// counter and are counted directly, mirroring
/// [`remaining_file_search_call_budget`](super::file_search_callout).
fn remaining_web_search_budget(state: &ResponsesState) -> usize {
    let web_executed = usize::try_from(state.web_search_calls_executed).unwrap_or(usize::MAX);
    let other_builtin_calls = state
        .accumulated_output
        .iter()
        .filter(|item| {
            super::file_search_callout::is_builtin_tool_call(item)
                && item.get("type").and_then(Value::as_str) != Some("web_search_call")
        })
        .count();
    let used = web_executed.saturating_add(other_builtin_calls);
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
fn append_excess_incomplete(ctx: &mut HttpFilterContext<'_>, call: &Value, index: usize) {
    let call_id = call.get("id").and_then(Value::as_str).unwrap_or("ws_unknown");
    let query = call
        .get("action")
        .and_then(|a| a.get("query"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let bridge = bridge_call_id(call_id, query, index);
    let ids = SearchCallIds {
        public: call_id,
        bridge: &bridge,
    };
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
    let output_item = build_output_item(ids.public, status, query, results, include_sources);
    let bridge = build_tool_result_messages(ids.bridge, status, query, results);
    push_search_turn(ctx, output_item, bridge);
}

/// Append a malformed search turn to [`ResponsesState`].
///
/// The public item remains `incomplete`, while the backend-valid bridge carries
/// the missing arguments and an explicit failure message. This prevents the
/// next inference iteration and persisted replay from treating a missing query
/// as a successful search with zero results.
fn append_incomplete(ctx: &mut HttpFilterContext<'_>, ids: &SearchCallIds<'_>) {
    let include_sources = include_action_sources(ctx);
    let output_item = build_output_item(ids.public, "incomplete", "", &[], include_sources);
    let bridge = build_incomplete_tool_result_messages(ids.bridge);
    push_search_turn(ctx, output_item, bridge);
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
    let output_item = build_output_item(ids.public, "failed", query, &[], include_sources);
    let bridge = build_failed_tool_result_messages(ids.bridge, query);
    push_search_turn(ctx, output_item, bridge);
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
fn push_search_turn(ctx: &mut HttpFilterContext<'_>, output_item: Value, bridge: [Value; 2]) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.messages.extend(bridge.iter().cloned());
        state.persisted_messages.extend(bridge);
        upsert_output_item(&mut state.accumulated_output, output_item);
    }
}

/// Replace the accumulated `web_search_call` sharing this id, or append it.
///
/// The response phase (`agentic_loop::collect_output_items`) already
/// accumulated the model's placeholder `web_search_call` for this id. Updating
/// it in place keeps exactly one public item per call, rather than emitting a
/// contradictory `completed` + `failed` pair for the same id. When no
/// placeholder exists (isolated unit contexts), the item is appended.
fn upsert_output_item(accumulated: &mut Vec<Value>, output_item: Value) {
    if let Some(id) = output_item.get("id").and_then(Value::as_str)
        && let Some(slot) = accumulated.iter_mut().find(|item| {
            item.get("type").and_then(Value::as_str) == Some("web_search_call")
                && item.get("id").and_then(Value::as_str) == Some(id)
        })
    {
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
            "output": SEARCH_UNAVAILABLE,
        }),
    ]
}

/// Write the loop control action to filter results.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_RESULT_KEY)
        .or_default()
        .set("action", action)?;
    Ok(())
}
