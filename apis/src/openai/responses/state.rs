// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-scoped state for the Responses API filter set.
//!
//! [`ResponsesState`] is stored in [`RequestExtensions`] and shared
//! across filter phases. It holds the heavy data needed by the
//! validate → rehydrate → `openai_tool_parse` → `openai_responses_proxy` →
//! `stream_events` → `openai_agentic_loop` pipeline.
//!
//! [`RequestExtensions`]: praxis_filter::RequestExtensions

use std::collections::{BTreeSet, HashMap, HashSet};

use bytes::Bytes;

/// Maximum citation file mappings retained during one response execution.
pub(crate) const MAX_CITATION_FILES: usize = 1_024;

/// Return whether an output item consumes the response-wide built-in tool-call budget.
///
/// All local built-in dispatchers share this classifier so adding a provider
/// call type cannot make their `max_tool_calls` accounting disagree.
pub(crate) fn is_builtin_tool_call(item: &serde_json::Value) -> bool {
    matches!(
        item.get("type").and_then(serde_json::Value::as_str),
        Some(
            "apply_patch_call"
                | "code_interpreter_call"
                | "computer_call"
                | "custom_tool_call"
                | "file_search_call"
                | "image_generation_call"
                | "local_shell_call"
                | "mcp_call"
                | "multi_agent_call"
                | "shell_call"
                | "tool_search_call"
                | "web_search_call"
        )
    )
}

/// Return whether an output item requires a result from the API client.
///
/// These calls cannot share a completed model round with locally dispatched
/// MCP or web-search calls: continuing inference would treat the client-owned
/// call as resolved before its matching output exists.
pub(crate) fn is_client_executed_tool_call(item: &serde_json::Value) -> bool {
    match item.get("type").and_then(serde_json::Value::as_str) {
        Some("apply_patch_call" | "computer_call" | "custom_tool_call" | "local_shell_call") => true,
        Some("shell_call") => {
            item.get("environment")
                .and_then(|environment| environment.get("type"))
                .and_then(serde_json::Value::as_str)
                == Some("local")
        },
        Some("tool_search_call") => item.get("execution").and_then(serde_json::Value::as_str) == Some("client"),
        _ => false,
    }
}

/// Count admitted built-in tool-call occurrences across every retained owner.
///
/// A call is charged when the model makes it, even when local execution later
/// produces an `incomplete` result. Retained owners can echo the same call, so
/// matching `(type, id)` multiplicities are merged by their maximum rather than
/// summed. This preserves separate same-ID occurrences within one owner while
/// avoiding double-counting a call copied between lifecycle stores.
pub(crate) fn consumed_builtin_tool_calls(state: &ResponsesState) -> usize {
    let owners = [
        state.accumulated_output.as_slice(),
        state.file_search_output_items.as_slice(),
        state.output_items(),
    ];
    let web_calls = count_tool_call_occurrences(&owners, ToolCallClass::Web);
    web_calls.saturating_add(count_tool_call_occurrences(&owners, ToolCallClass::NonWeb))
}

/// Count calls consumed before the current model round began.
///
/// Current-round calls are admitted separately in model output order. This
/// helper therefore excludes the current suffix of `accumulated_output` while
/// still including moved file-search calls and prior web-search executions.
pub(crate) fn consumed_builtin_tool_calls_before_current_round(state: &ResponsesState) -> usize {
    let prior_end = if state.current_round_output_start == 0 && state.output_items().is_empty() {
        state.accumulated_output.len()
    } else {
        state.current_round_output_start
    };
    let prior_agentic_output = state.accumulated_output.get(..prior_end).unwrap_or_default();
    let owners = [prior_agentic_output, state.file_search_output_items.as_slice()];
    let web_calls = count_tool_call_occurrences(&owners, ToolCallClass::Web);
    web_calls.saturating_add(count_tool_call_occurrences(&owners, ToolCallClass::NonWeb))
}

/// Which locally supported model-call occurrences to count.
#[derive(Clone, Copy)]
enum ToolCallClass {
    /// Hosted web-search calls, regardless of execution outcome.
    Web,
    /// Every other built-in call after a local file-search placeholder advances.
    NonWeb,
}

/// Return whether an item belongs to one accounting class.
fn is_counted_tool_call(item: &serde_json::Value, item_type: &str, class: ToolCallClass) -> bool {
    match class {
        ToolCallClass::Web => item_type == "web_search_call",
        ToolCallClass::NonWeb => {
            item_type != "web_search_call" && is_builtin_tool_call(item) && !is_pending_file_search_call(item)
        },
    }
}

/// Count call occurrences without collapsing repeated IDs from separate rounds.
fn count_tool_call_occurrences(owners: &[&[serde_json::Value]], class: ToolCallClass) -> usize {
    let mut maximum_multiplicity: HashMap<(&str, &str), usize> = HashMap::new();
    let mut calls_without_ids = 0_usize;

    for items in owners {
        let mut owner_multiplicity: HashMap<(&str, &str), usize> = HashMap::new();
        for item in *items {
            let Some(item_type) = item.get("type").and_then(serde_json::Value::as_str) else {
                continue;
            };
            if !is_counted_tool_call(item, item_type, class) {
                continue;
            }
            let Some(id) = item.get("id").and_then(serde_json::Value::as_str) else {
                // Without an identity there is no safe way to prove that two
                // retained values are the same call. Conservatively count each.
                calls_without_ids = calls_without_ids.saturating_add(1);
                continue;
            };
            let count = owner_multiplicity.entry((item_type, id)).or_default();
            *count = count.saturating_add(1);
        }
        for (key, count) in owner_multiplicity {
            maximum_multiplicity
                .entry(key)
                .and_modify(|maximum| *maximum = (*maximum).max(count))
                .or_insert(count);
        }
    }

    maximum_multiplicity
        .into_values()
        .fold(calls_without_ids, usize::saturating_add)
}

/// Return whether a file-search call is waiting for local execution.
fn is_pending_file_search_call(item: &serde_json::Value) -> bool {
    item.get("type").and_then(serde_json::Value::as_str) == Some("file_search_call")
        && matches!(
            item.get("status").and_then(serde_json::Value::as_str),
            Some("searching" | "in_progress")
        )
}

/// Lifecycle state for MCP batches that must return after local execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum McpApprovalState {
    /// No deferred local response exists.
    #[default]
    None,
    /// Return approval requests after a sibling dispatcher finishes re-entry.
    ApprovalPendingThenReturn,
    /// Execute ungated siblings, then return the pending approval response.
    ExecuteUngatedThenReturn,
    /// Execute the allowed prefix, then return calls rejected by `max_tool_calls`.
    ToolLimitExceededThenReturn,
}

/// Request-scoped state shared across Responses API filters.
///
/// Created by `openai_responses_validate` for every Responses API
/// create request. When `previous_response_id` is present,
/// `openai_responses_rehydrate` replaces it with an enriched
/// version that includes conversation history. Uses
/// [`serde_json::Value`] for flexibility while the Responses API
/// types stabilize; can be refactored to typed structs later
/// without affecting external callers.
///
/// [`RequestExtensions`]: praxis_filter::RequestExtensions
#[expect(
    clippy::struct_excessive_bools,
    reason = "request, transport, and deferred lifecycle flags are independent"
)]
pub(crate) struct ResponsesState {
    /// Maps file IDs to filenames for citation annotation extraction.
    pub citation_files: HashMap<String, String>,

    /// Truncation strategy for managing context window limits.
    ///
    /// Preserves the full object from the request so filters can
    /// inspect both the strategy type and any parameters.
    pub context_management: Option<serde_json::Value>,

    /// Conversation scope for multi-turn state.
    ///
    /// Can be a string ID or an object with `id`. Controls which
    /// stored conversation this request belongs to.
    pub conversation: Option<serde_json::Value>,

    /// Public output retained across local file-search inference rounds.
    ///
    /// This is request-local continuation state. Private search context stays
    /// in [`Self::messages`] and is never exposed through Conversations.
    pub file_search_output_items: Vec<serde_json::Value>,

    /// Additional fields to include in the response.
    ///
    /// E.g. `["usage"]`, `["file_search_results"]`. Filters that
    /// construct the response object check this to decide which
    /// optional sections to populate.
    pub include: Vec<String>,

    /// Stable response ID used while several streamed inference rounds are
    /// exposed as one logical Responses stream.
    pub logical_stream_response_id: Option<String>,

    /// Next downstream sequence number for a logical Responses stream.
    pub logical_stream_sequence: u64,

    /// Index where the current model round begins in `accumulated_output`.
    ///
    /// Local dispatchers may append approval or result items before every
    /// sibling dispatcher has evaluated the round. Keeping the boundary
    /// explicitly prevents those local items from being mistaken for prior
    /// model output during response-wide tool-budget admission.
    pub current_round_output_start: usize,

    /// Whether stored history was successfully resolved into this state.
    ///
    /// The proxy uses this to distinguish locally consumed history identifiers
    /// from provider-owned identifiers that must pass through unchanged.
    pub history_rehydrated: bool,

    /// The current request's input items, immutable after construction.
    ///
    /// Preserved as-is so downstream filters can inspect what the
    /// client actually sent, independent of conversation history
    /// resolved by `rehydrate`.
    pub input: Vec<serde_json::Value>,

    /// Current agentic loop iteration (0-indexed). Incremented by
    /// `openai_agentic_loop` at the start of each new inference round.
    pub iteration: u32,

    /// Maximum number of built-in tool invocations.
    ///
    /// Enforced by built-in tool filters across retained output.
    /// `None` means no explicit limit was set by the client.
    pub max_tool_calls: Option<u32>,

    /// Resolved MCP tool definitions keyed by `(server_label,
    /// tool_name)`.
    ///
    /// Built by `openai_mcp_tool_resolve` from `tools/list` responses.
    /// Consumed by `mcp_tool` (#27) for dispatch routing.
    pub mcp_tool_map: HashMap<(String, String), serde_json::Value>,

    /// Lifecycle state for an MCP batch that must return after execution.
    pub mcp_approval_state: McpApprovalState,

    /// Whether a local dispatcher exhausted the response-wide tool budget.
    ///
    /// Request filters run in pipeline order. Deferring the terminal response
    /// lets later dispatchers retain rejected sibling calls before the agentic
    /// loop completes the current response locally.
    pub deferred_tool_limit_completion: bool,

    /// Whether an upstream logical stream supplied a `[DONE]` sentinel.
    ///
    /// This survives filter-state re-arming between IRR request steps so a
    /// request-side local completion can preserve the original stream shape.
    pub deferred_stream_done: bool,

    /// Resolved conversation history sent to the backend.
    ///
    /// Initialized from the current request's input. When
    /// `previous_response_id` is set, `rehydrate` prepends stored
    /// history. `openai_agentic_loop` appends tool results during agentic
    /// loops. `openai_responses_proxy` reads this as the authoritative
    /// conversation to send to the backend. Output-only metadata
    /// items must be omitted from this field.
    pub messages: Vec<serde_json::Value>,

    /// Whether tool calls may execute concurrently within an
    /// iteration. Defaults to `true` per the API spec.
    pub parallel_tool_calls: bool,

    /// Full message history to persist for future rehydration.
    ///
    /// This may include output-only metadata items omitted from
    /// [`Self::messages`] because it is not forwarded to backend
    /// inference.
    pub persisted_messages: Vec<serde_json::Value>,

    /// ID of a previous response to continue from.
    ///
    /// When set, `rehydrate` fetches the stored conversation
    /// history for this response and prepends it to `messages`.
    pub previous_response_id: Option<String>,

    /// MCP tool listings recovered from the previous response.
    pub previous_tools: Vec<serde_json::Value>,

    /// Token usage reported by the previous response.
    pub previous_usage: Option<serde_json::Value>,

    /// Client-visible tool choice retained when an internal continuation
    /// resets [`Self::tool_choice`] to `"auto"` for later model rounds.
    pub original_tool_choice: Option<serde_json::Value>,

    /// Stable creation timestamp for the public response across iterations.
    pub response_created_at: Option<u64>,

    /// Stable public response ID assigned by request validation.
    ///
    /// Iterative router steps preserve extensions while resetting per-step
    /// metadata, so translated responses must also be able to read the ID here.
    pub response_id: Option<String>,

    /// Parsed request body as received from the client.
    pub request_body: serde_json::Value,

    /// Whether provider-visible request fields require outbound serialization.
    pub request_body_rebuild: RequestBodyRebuild,

    /// The constructed response object for the current iteration.
    pub response_object: serde_json::Value,

    /// Tool calls from the current inference response only.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration
    /// before `stream_events` writes new ones. Without explicit
    /// clearing, stale tool calls from a previous iteration cause
    /// duplicate dispatch.
    pub tool_calls: Vec<serde_json::Value>,

    /// Web search calls from the current inference response only.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration.
    /// Stored separately from `tool_calls` because `web_search_call`
    /// items have a different shape (`action.query` instead of
    /// `name`/`arguments`) and are dispatched by a different filter.
    pub web_search_calls: Vec<serde_json::Value>,

    /// Cumulative web searches dispatched to the provider across all
    /// agentic-loop iterations.
    ///
    /// This is operational execution state, not `max_tool_calls` accounting:
    /// the response-wide API limit charges retained model-call admissions,
    /// including calls whose local execution is incomplete.
    pub web_search_calls_executed: u32,

    /// Tool choice setting. Reset to `"auto"` by `openai_agentic_loop`
    /// after the first iteration; the original value from the
    /// request only applies to the first inference call.
    pub tool_choice: serde_json::Value,

    /// Processed tool definitions from the request.
    pub tools: Vec<serde_json::Value>,

    /// Token usage accumulated across all iterations within the
    /// request. `stream_events` merges per-iteration usage into
    /// the running total.
    pub usage: serde_json::Value,

    /// Output items accumulated across all agentic loop iterations.
    ///
    /// Each round's model output items and MCP execution results are
    /// appended here so the final response contains the complete
    /// trace. `openai_agentic_loop` writes model items, `mcp_dispatch`
    /// writes `mcp_call` and `mcp_approval_request` items.
    pub accumulated_output: Vec<serde_json::Value>,

    /// Client-visible lifecycle progress of output items, keyed by item id.
    ///
    /// Tracked across IRR rounds so `stream_events` can synthesize incremental
    /// events for locally generated tool items (MCP calls and approvals, or web
    /// searches absent from the upstream stream): each milestone (`added`, the
    /// progress/outcome lifecycle, the last delivered content) exactly once,
    /// without duplicating an `output_item.added` the model already streamed,
    /// yet still re-emitting the outcome when the same item changes locally
    /// (e.g. a model `web_search_call` placeholder later completed under the
    /// same id, or gaining `action.sources` after local execution).
    pub emitted_output_items: HashMap<String, EmittedItem>,

    /// Item ids of local tool calls a dispatch filter actually executed this
    /// request (execution provenance), keyed by the output item's `id`.
    ///
    /// `stream_events` synthesizes the client-visible progress/outcome lifecycle
    /// only for items recorded here, never for every tool-typed item that reaches
    /// [`Self::accumulated_output`]. A non-dispatchable round that aborts on a
    /// parse error still copies the model's `web_search_call` placeholder into
    /// `accumulated_output` (via `agentic_loop::collect_streaming_output_items`),
    /// so keying synthesis on item type alone would fabricate an
    /// `in_progress`/`searching`/`done` lifecycle for a search that never ran.
    /// `mcp_dispatch` records both the approval-request and result item ids;
    /// `web_search` records the id it replaces with executed results.
    pub locally_executed_output_items: HashSet<String>,
}

/// Which client-visible lifecycle milestones a locally generated output item has
/// already reached the client, so `stream_events` synthesizes each exactly once
/// yet re-emits the outcome when the item's content later changes.
///
/// The model backend streams at most `output_item.added`/`output_item.done` for
/// these items and often never the tool-specific progress events
/// (`*.in_progress`, `*.searching`, `*.completed`, `*.failed`), so tracking each
/// milestone separately from content lets the proxy fill in the missing progress
/// lifecycle even when local execution leaves the item's content byte-identical
/// to the model's placeholder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct EmittedItem {
    /// `output_item.added` reached the client (model passthrough or synthesis).
    pub added: bool,
    /// The finalizing `output_item.done` envelope reached the client for this item
    /// (model passthrough that survived the premature-`done` check, or synthesis).
    ///
    /// Tracked separately from [`Self::added`] and the phase set: a backend may
    /// stream every tool-specific phase in-band yet be cut off before the `done`
    /// envelope, so a resumed round must still finalize the item with exactly one
    /// `done`. Conversely, once the envelope has been delivered it is re-emitted
    /// only when the item's content changes, never merely because a phase was
    /// synthesized.
    pub done_delivered: bool,
    /// The individual tool-specific progress and outcome events that have reached
    /// the client for this item (e.g. `response.web_search_call.in_progress`,
    /// `response.mcp_call.completed`), keyed by their event type.
    ///
    /// Each phase is tracked independently — they are distinct API lifecycle
    /// events, not one combined milestone. An entry is added only by observing an
    /// actual progress event (model passthrough) or by synthesizing one; never by
    /// `output_item.added`/`output_item.done` alone, which announce the item and
    /// carry its content but are no proof that any progress event was delivered.
    /// Tracking each phase separately lets the proxy fill in exactly the events a
    /// partial in-band lifecycle (e.g. `in_progress` then `done`) still owes,
    /// without duplicating the ones the model already streamed.
    pub streamed_phases: BTreeSet<String>,
    /// Fixed-size digest of the item content last delivered to the client,
    /// compared across rounds to re-emit the `output_item.done` envelope when the
    /// same item changes locally (e.g. gains `action.sources`).
    ///
    /// A digest rather than the full serialized item so retained state stays
    /// bounded: local tool payloads already live in `accumulated_output`, and an
    /// IRR response can reach tens of MiB across rounds, so keeping a second full
    /// copy per item here would be payload-scale memory amplification.
    pub content_digest: u64,
}

/// Whether the proxy can preserve the original request bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum RequestBodyRebuild {
    /// No provider-visible state has changed.
    #[default]
    PreserveOriginal,

    /// Provider-visible state must be serialized before inference.
    Required,
}

impl Default for ResponsesState {
    #[expect(clippy::too_many_lines, reason = "exhaustive struct field initialization")]
    fn default() -> Self {
        Self {
            citation_files: HashMap::new(),
            context_management: None,
            conversation: None,
            file_search_output_items: Vec::new(),
            include: Vec::new(),
            logical_stream_response_id: None,
            logical_stream_sequence: 0,
            current_round_output_start: 0,
            history_rehydrated: false,
            input: Vec::new(),
            iteration: 0,
            max_tool_calls: None,
            mcp_approval_state: McpApprovalState::None,
            deferred_tool_limit_completion: false,
            deferred_stream_done: false,
            mcp_tool_map: HashMap::new(),
            messages: Vec::new(),
            parallel_tool_calls: true,
            persisted_messages: Vec::new(),
            previous_response_id: None,
            previous_tools: Vec::new(),
            previous_usage: None,
            original_tool_choice: None,
            response_created_at: None,
            response_id: None,
            request_body: serde_json::Value::Null,
            request_body_rebuild: RequestBodyRebuild::PreserveOriginal,
            response_object: serde_json::Value::Null,
            tool_calls: Vec::new(),
            web_search_calls: Vec::new(),
            web_search_calls_executed: 0,
            tool_choice: serde_json::Value::String("auto".to_owned()),
            tools: Vec::new(),
            usage: serde_json::Value::Null,
            accumulated_output: Vec::new(),
            emitted_output_items: HashMap::new(),
            locally_executed_output_items: HashSet::new(),
        }
    }
}

impl ResponsesState {
    /// Create initial state from a parsed request body.
    pub(crate) fn from_request_body(body: serde_json::Value) -> Self {
        let messages = normalize_input(&body);
        let persisted_messages = messages.clone();
        let tool_choice = body
            .get("tool_choice")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::String("auto".to_owned()));

        let tools = extract_array_field(&body, "tools");
        Self {
            context_management: body.get("context_management").cloned(),
            conversation: body.get("conversation").cloned(),
            include: extract_string_array(&body, "include"),
            input: messages.clone(),
            max_tool_calls: extract_u32(&body, "max_tool_calls"),
            messages,
            parallel_tool_calls: extract_bool_or(&body, "parallel_tool_calls", true),
            persisted_messages,
            previous_response_id: extract_string(&body, "previous_response_id"),
            request_body: body,
            tool_choice,
            tools,
            accumulated_output: Vec::new(),
            ..Default::default()
        }
    }

    /// Require the proxy to serialize provider-visible request state.
    pub(crate) fn mark_request_body_for_rebuild(&mut self) {
        self.request_body_rebuild = RequestBodyRebuild::Required;
    }

    /// Borrow the public output owned by [`Self::response_object`].
    pub(crate) fn output_items(&self) -> &[serde_json::Value] {
        self.response_object
            .get("output")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Mutably borrow public output, creating a valid array when absent.
    pub(crate) fn output_items_mut(&mut self) -> &mut Vec<serde_json::Value> {
        if !self.response_object.is_object() {
            self.response_object = serde_json::Value::Object(serde_json::Map::new());
        }
        let serde_json::Value::Object(response) = &mut self.response_object else {
            unreachable!("response_object was normalized to an object")
        };
        let output = response
            .entry("output".to_owned())
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        if !output.is_array() {
            *output = serde_json::Value::Array(Vec::new());
        }
        let serde_json::Value::Array(items) = output else {
            unreachable!("output was normalized to an array")
        };
        items
    }

    /// Return whether provider-visible request state requires serialization.
    pub(crate) fn request_body_requires_rebuild(&self) -> bool {
        self.request_body_rebuild == RequestBodyRebuild::Required
    }

    /// Build the final response body from accumulated state.
    ///
    /// Replaces `response_object["output"]` with the full `accumulated_output`
    /// (all rounds), stamps accumulated usage, and serializes back to body bytes.
    pub(crate) fn finalize_response_body(&self, body: &mut Option<Bytes>) {
        if !self.response_object.is_object() {
            return;
        }
        let mut response = self.response_object.clone();
        if let Some(obj) = response.as_object_mut() {
            if !self.accumulated_output.is_empty() {
                obj.insert(
                    "output".to_owned(),
                    serde_json::Value::Array(self.accumulated_output.clone()),
                );
            }
            if !self.usage.is_null() {
                obj.insert("usage".to_owned(), self.usage.clone());
            }
        }
        if let Ok(serialized) = serde_json::to_vec(&response) {
            *body = Some(Bytes::from(serialized));
        }
    }
}

/// Return budget admission decisions for one kind of current-round local call.
///
/// Dispatch filters run independently on request re-entry, but the response-wide
/// limit applies in model output order. Replaying the immutable current response
/// here prevents pipeline order from letting a later web search displace an
/// earlier MCP call (or vice versa).
pub(crate) fn current_round_tool_call_admissions(
    state: &ResponsesState,
    target_calls: &[serde_json::Value],
    is_mcp_function: impl Fn(&serde_json::Value) -> bool,
) -> Vec<bool> {
    current_round_tool_call_admissions_by(
        state,
        target_calls.len(),
        |index| target_calls.get(index),
        is_mcp_function,
    )
}

/// Return current-round admissions for callers that already borrow tool calls.
pub(crate) fn current_round_borrowed_tool_call_admissions(
    state: &ResponsesState,
    target_calls: &[&serde_json::Value],
    is_mcp_function: impl Fn(&serde_json::Value) -> bool,
) -> Vec<bool> {
    current_round_tool_call_admissions_by(
        state,
        target_calls.len(),
        |index| target_calls.get(index).copied(),
        is_mcp_function,
    )
}

/// Replay model output order against an arbitrary borrowed target-call view.
fn current_round_tool_call_admissions_by<'a>(
    state: &ResponsesState,
    target_count: usize,
    target_call: impl Fn(usize) -> Option<&'a serde_json::Value>,
    is_mcp_function: impl Fn(&serde_json::Value) -> bool,
) -> Vec<bool> {
    let previous_calls = consumed_builtin_tool_calls_before_current_round(state);
    let mut remaining = state.max_tool_calls.map_or(usize::MAX, |limit| {
        usize::try_from(limit)
            .unwrap_or(usize::MAX)
            .saturating_sub(previous_calls)
    });
    let mut admissions = Vec::new();
    let mut target_index = 0;

    for item in state.output_items() {
        let mcp = is_mcp_function(item);
        let consumes_budget = mcp || is_builtin_tool_call(item);
        let admitted = !consumes_budget || remaining > 0;
        if consumes_budget {
            remaining = remaining.saturating_sub(1);
        }
        if target_index < target_count && target_call(target_index) == Some(item) {
            admissions.push(admitted);
            target_index = target_index.saturating_add(1);
        }
    }
    admissions
}

/// Normalize the `input` field into a message array.
///
/// The Responses API `input` can be a string (single user message),
/// a single item object, or an array of items. Normalizes all three
/// forms to a `Vec<Value>`.
fn normalize_input(body: &serde_json::Value) -> Vec<serde_json::Value> {
    match body.get("input") {
        Some(serde_json::Value::Array(arr)) => arr.clone(),
        Some(input @ serde_json::Value::Object(_)) => vec![input.clone()],
        Some(serde_json::Value::String(s)) => {
            vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": s,
            })]
        },
        _ => Vec::new(),
    }
}

/// Extract a JSON array field by name, defaulting to empty.
fn extract_array_field(body: &serde_json::Value, field: &str) -> Vec<serde_json::Value> {
    body.get(field)
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Extract a string field by name.
fn extract_string(body: &serde_json::Value, field: &str) -> Option<String> {
    body.get(field)
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

/// Extract an array of strings by name, defaulting to empty.
fn extract_string_array(body: &serde_json::Value, field: &str) -> Vec<String> {
    body.get(field)
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Extract a `u32` field by name, logging when a value is present
/// but not representable as `u32`.
fn extract_u32(body: &serde_json::Value, field: &str) -> Option<u32> {
    let raw = body.get(field)?;
    let result = raw.as_u64().and_then(|v| u32::try_from(v).ok());
    if result.is_none() {
        tracing::debug!(field, %raw, "ignoring non-u32 value");
    }
    result
}

/// Extract a bool field by name, returning a default if absent.
fn extract_bool_or(body: &serde_json::Value, field: &str, default: bool) -> bool {
    body.get(field).and_then(serde_json::Value::as_bool).unwrap_or(default)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn from_request_body_extracts_string_input() {
        let body = json!({
            "model": "gpt-4o",
            "input": "Hello, world!"
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.input.len(), 1, "string input should produce one item");
        assert_eq!(
            state.input[0]["role"], "user",
            "string input should default to user role"
        );
        assert_eq!(
            state.input[0]["type"], "message",
            "string input should produce a Responses message item"
        );
        assert_eq!(state.input[0]["content"], "Hello, world!");
    }

    #[test]
    fn from_request_body_extracts_array_input() {
        let body = json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"}
            ]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.input.len(), 2, "array input should preserve all items");
    }

    #[test]
    fn from_request_body_wraps_object_input_as_single_item() {
        let input = json!({
            "type": "message",
            "role": "developer",
            "content": "Be terse."
        });
        let state = ResponsesState::from_request_body(json!({
            "model": "gpt-4o",
            "input": input
        }));

        assert_eq!(state.input, vec![input]);
        assert_eq!(state.messages, state.input);
        assert_eq!(state.persisted_messages, state.input);
    }

    #[test]
    fn from_request_body_empty_input() {
        let body = json!({"model": "gpt-4o"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.input.is_empty(), "missing input should produce empty input");
    }

    #[test]
    fn input_and_messages_start_identical() {
        let body = json!({
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.input, state.messages,
            "input and messages should be identical at construction"
        );
        assert_eq!(
            state.input, state.persisted_messages,
            "input and persisted_messages should be identical at construction"
        );
    }

    #[test]
    fn from_request_body_extracts_tools() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "tools": [{"type": "function", "name": "get_weather"}]
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.tools.len(), 1, "should extract one tool");
    }

    #[test]
    fn from_request_body_default_tool_choice() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.tool_choice, json!("auto"), "default tool_choice should be auto");
    }

    #[test]
    fn from_request_body_explicit_tool_choice() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "tool_choice": "required"
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.tool_choice,
            json!("required"),
            "should preserve explicit tool_choice"
        );
    }

    #[test]
    fn initial_state_has_zero_iteration() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.iteration, 0, "initial iteration should be 0");
    }

    #[test]
    fn initial_state_has_empty_tool_calls() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.tool_calls.is_empty(), "initial tool_calls should be empty");
    }

    #[test]
    fn initial_state_has_null_usage() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.usage.is_null(), "initial usage should be null");
    }

    #[test]
    fn request_body_is_preserved() {
        let body = json!({"model": "gpt-4o", "input": "hello", "temperature": 0.7});
        let state = ResponsesState::from_request_body(body.clone());
        assert_eq!(state.request_body, body, "original request body should be preserved");
    }

    #[test]
    fn extracts_previous_response_id() {
        let body = json!({"model": "gpt-4o", "input": "test", "previous_response_id": "resp_abc123"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.previous_response_id.as_deref(), Some("resp_abc123"));
    }

    #[test]
    fn previous_response_id_defaults_to_none() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.previous_response_id.is_none());
    }

    #[test]
    fn extracts_conversation_string() {
        let body = json!({"model": "gpt-4o", "input": "test", "conversation": "conv_xyz"});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.conversation, Some(json!("conv_xyz")));
    }

    #[test]
    fn extracts_conversation_object() {
        let body = json!({"model": "gpt-4o", "input": "test", "conversation": {"id": "conv_xyz"}});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.conversation, Some(json!({"id": "conv_xyz"})));
    }

    #[test]
    fn extracts_context_management() {
        let body = json!({
            "model": "gpt-4o",
            "input": "test",
            "context_management": {"type": "truncation", "max_tokens": 4096}
        });
        let state = ResponsesState::from_request_body(body);
        assert_eq!(
            state.context_management,
            Some(json!({"type": "truncation", "max_tokens": 4096}))
        );
    }

    #[test]
    fn extracts_include() {
        let body = json!({"model": "gpt-4o", "input": "test", "include": ["usage", "file_search_results"]});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.include, vec!["usage", "file_search_results"]);
    }

    #[test]
    fn include_defaults_to_empty() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.include.is_empty());
    }

    #[test]
    fn extracts_max_tool_calls() {
        let body = json!({"model": "gpt-4o", "input": "test", "max_tool_calls": 5});
        let state = ResponsesState::from_request_body(body);
        assert_eq!(state.max_tool_calls, Some(5));
    }

    #[test]
    fn max_tool_calls_defaults_to_none() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.max_tool_calls.is_none());
    }

    #[test]
    fn current_round_budget_admission_follows_model_output_order() {
        let mcp = json!({
            "type":"function_call", "call_id":"mcp_1",
            "name":"utilities__weather", "status":"completed"
        });
        let web = json!({"type":"web_search_call", "id":"ws_1", "status":"completed"});
        let state = ResponsesState {
            max_tool_calls: Some(2),
            current_round_output_start: 1,
            accumulated_output: vec![json!({"type":"mcp_call", "id":"prior"}), mcp.clone(), web.clone()],
            response_object: json!({"output":[mcp, web]}),
            ..ResponsesState::default()
        };
        let is_mcp = |item: &serde_json::Value| {
            item.get("name").and_then(serde_json::Value::as_str) == Some("utilities__weather")
        };

        assert_eq!(current_round_tool_call_admissions(&state, &[mcp], is_mcp), vec![true]);
        assert_eq!(current_round_tool_call_admissions(&state, &[web], is_mcp), vec![false]);
    }

    #[test]
    fn incomplete_web_call_remains_charged_in_later_rounds() {
        let next = json!({"type":"web_search_call", "id":"ws_next", "status":"in_progress"});
        let state = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: 1,
            accumulated_output: vec![
                json!({"type":"web_search_call", "id":"ws_malformed", "status":"incomplete"}),
                next.clone(),
            ],
            response_object: json!({"output":[next.clone()]}),
            ..ResponsesState::default()
        };

        assert_eq!(consumed_builtin_tool_calls_before_current_round(&state), 1);
        assert_eq!(
            current_round_tool_call_admissions(&state, &[next], |_| false),
            vec![false],
            "an admitted call stays charged even when local execution was incomplete"
        );
    }

    #[test]
    fn current_provider_execution_does_not_double_charge_round_admission() {
        let web = json!({"type":"web_search_call", "id":"ws_current", "status":"completed"});
        let mcp = json!({
            "type":"function_call", "call_id":"mcp_current",
            "name":"utilities__weather", "status":"completed"
        });
        let state = ResponsesState {
            max_tool_calls: Some(2),
            web_search_calls_executed: 1,
            accumulated_output: vec![web.clone(), mcp.clone()],
            response_object: json!({"output":[web, mcp.clone()]}),
            ..ResponsesState::default()
        };

        assert_eq!(
            current_round_tool_call_admissions(&state, &[mcp], |item| {
                item.get("name").and_then(serde_json::Value::as_str) == Some("utilities__weather")
            }),
            vec![true],
            "the execution counter must not charge a current web call before ordered admission"
        );
    }

    #[test]
    fn repeated_file_search_ids_count_as_separate_occurrences() {
        let first = json!({"type":"file_search_call", "id":"fs_reused", "status":"completed"});
        let second = json!({"type":"file_search_call", "id":"fs_reused", "status":"incomplete"});
        let state = ResponsesState {
            accumulated_output: vec![first.clone(), second.clone()],
            // Echoes in another lifecycle owner must not add two more calls.
            file_search_output_items: vec![first, second],
            ..ResponsesState::default()
        };

        assert_eq!(
            consumed_builtin_tool_calls(&state),
            2,
            "same-ID calls from separate rounds remain separate budget occurrences"
        );
    }

    #[test]
    fn current_round_budget_ignores_locally_appended_approval_items() {
        let web = json!({"type":"web_search_call", "id":"ws_1", "status":"completed"});
        let gated = json!({
            "type":"function_call", "call_id":"mcp_gated",
            "name":"utilities__gated", "status":"completed"
        });
        let ungated = json!({
            "type":"function_call", "call_id":"mcp_ungated",
            "name":"utilities__ungated", "status":"completed"
        });
        let state = ResponsesState {
            max_tool_calls: Some(3),
            accumulated_output: vec![
                web.clone(),
                gated.clone(),
                ungated.clone(),
                json!({"type":"mcp_approval_request", "id":"mcp_gated"}),
            ],
            response_object: json!({"output":[web, gated, ungated.clone()]}),
            ..ResponsesState::default()
        };
        let is_mcp = |item: &serde_json::Value| {
            item.get("name")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| name.starts_with("utilities__"))
        };

        assert_eq!(
            current_round_tool_call_admissions(&state, &[ungated], is_mcp),
            vec![true]
        );
    }

    #[test]
    fn parallel_tool_calls_defaults_to_true() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.parallel_tool_calls);
    }

    #[test]
    #[expect(
        clippy::cognitive_complexity,
        reason = "exhaustive one-assert-per-field check of every default value"
    )]
    fn default_produces_expected_values() {
        let state = ResponsesState::default();
        assert!(state.context_management.is_none());
        assert!(state.conversation.is_none());
        assert!(state.include.is_empty());
        assert!(state.input.is_empty());
        assert_eq!(state.iteration, 0);
        assert!(state.max_tool_calls.is_none());
        assert!(state.mcp_tool_map.is_empty());
        assert!(state.messages.is_empty());
        assert!(state.output_items().is_empty());
        assert!(state.parallel_tool_calls);
        assert!(state.persisted_messages.is_empty());
        assert!(state.previous_response_id.is_none());
        assert!(state.previous_tools.is_empty());
        assert!(state.previous_usage.is_none());
        assert!(state.request_body.is_null());
        assert!(state.response_object.is_null());
        assert!(state.tool_calls.is_empty());
        assert!(state.web_search_calls.is_empty());
        assert_eq!(state.web_search_calls_executed, 0);
        assert_eq!(state.tool_choice, json!("auto"));
        assert!(state.tools.is_empty());
        assert!(state.usage.is_null());
        assert!(state.accumulated_output.is_empty());
        assert!(state.emitted_output_items.is_empty());
        assert!(state.locally_executed_output_items.is_empty());
    }

    #[test]
    fn parallel_tool_calls_explicit_false() {
        let body = json!({"model": "gpt-4o", "input": "test", "parallel_tool_calls": false});
        let state = ResponsesState::from_request_body(body);
        assert!(!state.parallel_tool_calls);
    }

    #[test]
    fn response_object_is_the_single_output_owner() {
        let first = json!({"type": "message", "id": "msg_1"});
        let second = json!({"type": "reasoning", "id": "rs_1"});
        let mut state = ResponsesState {
            response_object: json!({"id": "resp_1", "output": [first.clone()]}),
            ..Default::default()
        };

        state.output_items_mut().push(second.clone());
        assert_eq!(state.output_items(), &[first.clone(), second.clone()]);
        assert_eq!(state.response_object["output"], json!([first, second]));
        assert_eq!(state.response_object["id"], "resp_1");
    }

    #[test]
    fn mutable_output_normalizes_missing_or_malformed_response_output() {
        let mut state = ResponsesState {
            response_object: json!({"id": "resp_1", "output": "invalid"}),
            ..Default::default()
        };

        state.output_items_mut().push(json!({"type": "message"}));

        assert_eq!(state.output_items().len(), 1);
        assert!(state.response_object["output"].is_array());
        assert_eq!(state.response_object["id"], "resp_1");
    }

    #[test]
    fn default_has_empty_mcp_tool_map() {
        let state = ResponsesState::default();
        assert!(state.mcp_tool_map.is_empty(), "default mcp_tool_map should be empty");
    }

    #[test]
    fn from_request_body_has_empty_mcp_tool_map() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.mcp_tool_map.is_empty(), "initial mcp_tool_map should be empty");
    }

    #[test]
    fn builtin_tool_call_classifier_covers_shared_dispatch_budget_types() {
        for call_type in [
            "apply_patch_call",
            "code_interpreter_call",
            "computer_call",
            "custom_tool_call",
            "file_search_call",
            "image_generation_call",
            "local_shell_call",
            "mcp_call",
            "multi_agent_call",
            "shell_call",
            "tool_search_call",
            "web_search_call",
        ] {
            assert!(
                is_builtin_tool_call(&json!({"type":call_type})),
                "{call_type} must consume the shared built-in tool-call budget"
            );
        }
        assert!(!is_builtin_tool_call(&json!({"type":"function_call"})));
        assert!(!is_builtin_tool_call(&json!({"type":"message"})));
    }

    #[test]
    fn client_executed_tool_call_classifier_covers_result_owned_types() {
        for call_type in [
            "apply_patch_call",
            "computer_call",
            "custom_tool_call",
            "local_shell_call",
        ] {
            assert!(
                is_client_executed_tool_call(&json!({"type":call_type})),
                "{call_type} requires a client-supplied result"
            );
        }
        assert!(is_client_executed_tool_call(
            &json!({"type":"tool_search_call", "execution":"client"})
        ));
        assert!(!is_client_executed_tool_call(
            &json!({"type":"tool_search_call", "execution":"server"})
        ));
        assert!(is_client_executed_tool_call(
            &json!({"type":"shell_call", "environment":{"type":"local"}})
        ));
        assert!(!is_client_executed_tool_call(
            &json!({"type":"shell_call", "environment":{"type":"container_reference", "container_id":"cntr_1"}})
        ));
        assert!(!is_client_executed_tool_call(&json!({"type":"web_search_call"})));
    }
}
