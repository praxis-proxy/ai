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

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fmt,
    time::Duration,
};

use bytes::Bytes;
use praxis_filter::{FilterAction, body::MAX_JSON_BODY_BYTES};

use super::{
    bounded_json_size,
    error::responses_error_rejection,
    file_search_callout::citations::{annotate_response, annotation_staging_bytes},
};

/// Maximum citation file mappings retained during one response execution.
pub(crate) const MAX_CITATION_FILES: usize = 1_024;

/// Origin of a reconciled `file_search` item queued for EOS synthesis (#313 §4).
/// Captured at translate time (a `Private` item's opening was suppressed by
/// `stream_events` and must be reproduced; a `Native` item's opening already
/// streamed live, so only the tail is synthesized).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SynthesisKind {
    /// Translated from a private `function_call` this round; opening must be synthesized.
    Private,
    /// Native hybrid `file_search_call`; opening already streamed.
    Native,
}

/// One `file_search_call` the parse owner (`openai_agentic_loop`) accumulated
/// this round and handed to `openai_file_search_callout` for execution.
///
/// The owner is the sole parser: it appends the canonical `file_search_call`
/// output item to [`ResponsesState::accumulated_output`] and records its
/// absolute index here plus the synthesis origin (private-normalized vs native).
/// The dispatcher drains these at request-body EOS and mutates the indexed item
/// in place — setting `completed`/`incomplete`, adding public results, and
/// bridging model context — so the complete output item is never cloned into a
/// second owner (avoids the payload duplication the "keep `Vec<Value>`" interface
/// would incur; the recommended "store output indices" boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileSearchAssignment {
    /// Absolute index into [`ResponsesState::accumulated_output`] of the
    /// `file_search_call` item the dispatcher must execute and reconcile.
    pub output_index: usize,
    /// Whether the item was normalized from a private `function_call` this round
    /// (`Private`, opening suppressed) or streamed natively (`Native`).
    pub synthesis: SynthesisKind,
}

/// Absolute reference to an item in [`ResponsesState::accumulated_output`].
///
/// Dispatch queues retain these fixed-size references instead of cloning the
/// provider output item into a second JSON owner. The accumulator remains the
/// canonical public-response owner until terminal finalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputAssignment {
    /// Absolute index of the assigned output item.
    pub output_index: usize,
}

/// One hosted web-search call assigned to the local dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WebSearchAssignment {
    /// Absolute index of the canonical `web_search_call` output item.
    pub output_index: usize,
    /// Position among web-search calls in the current model round.
    pub ordinal: usize,
}

/// Compact invocation reconstructed from a server-owned MCP approval.
///
/// Approval resumptions have no current-round provider output item to point at,
/// so they retain only the fields execution needs. In particular the approval
/// id is owned once and serves as both the call id and approval correlation id.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ApprovedMcpInvocation {
    /// Encoded function name used to resolve the MCP target.
    pub encoded_name: String,
    /// Original approval id, also used as the MCP call id.
    pub approval_id: String,
    /// JSON-encoded invocation arguments.
    pub arguments: String,
}

/// Dispatch reference for one function call.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ToolCallAssignment {
    /// Provider-produced call owned by `accumulated_output`.
    Output(OutputAssignment),
    /// Approved call reconstructed from the durable approval record.
    Approved(ApprovedMcpInvocation),
}

/// A terminal failure a request-phase dispatcher recorded in shared state.
///
/// A dispatcher (e.g. `openai_file_search_callout`) never rejects or rewrites a
/// response itself — that would make it a second terminal-response owner. Instead
/// it records the failure here and returns `Continue`; the parse owner
/// (`openai_agentic_loop`), which runs next in the same request phase, converts it
/// into a buffered JSON rejection before commitment or a logical-stream SSE error
/// after commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatchFailure {
    /// HTTP status for the buffered (pre-commitment) rejection envelope.
    pub status: u16,
    /// Stable error `code` for both the buffered envelope and the SSE error frame.
    pub code: &'static str,
    /// Human-readable, bounded failure message.
    pub message: String,
}

/// How a lowered private `function` call is restored to its canonical
/// client-owned typed output item on a function-only Responses backend (#1131).
///
/// Recorded by `openai_client_tool_compat` when it lowers a rich client tool
/// declaration (`custom`, `namespace` member, local `shell`, or
/// client-executed `tool_search`) to a private `function` tool on the outbound
/// request. The buffered and streaming restoration paths look up a returned
/// `function_call` by name to rebuild the exact typed item the client expects.
/// Praxis never executes these tools; restoration only re-types the model's call
/// before it reaches the client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredClientTool {
    /// The canonical tool name the client declared, restored onto the output item.
    /// For a `namespace` member this is the bare MEMBER name (namespace stripped).
    pub original_name: String,
    /// The declared namespace to re-add for a `namespace` member; `None` otherwise.
    pub namespace: Option<String>,
    /// The typed output item the returned `function_call` must be restored to.
    pub restore: ClientToolRestore,
}

/// The canonical output-item type a lowered `function` call restores to (#1131).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClientToolRestore {
    /// A freeform `custom` tool: `function_call` → `custom_tool_call`, unwrapping
    /// the single string parameter into the plain-string `input` field.
    Custom,
    /// A `function` member of a `namespace`: keep `function_call`, restore the
    /// original member name and re-add the `namespace`.
    Namespace,
    /// A `custom` member of a `namespace`: `function_call` → `custom_tool_call`
    /// with the original member name, the re-added `namespace`, and the single
    /// string parameter unwrapped into the plain-string `input` field.
    NamespaceCustom,
    /// A local `shell` tool: `function_call` → `shell_call` with
    /// `environment.type == "local"`.
    Shell,
    /// A client-executed `tool_search` tool: `function_call` → `tool_search_call`
    /// with `execution == "client"`.
    ToolSearch,
}

/// Verbatim snapshot of the client-declared `tools`/`tool_choice` taken before
/// `openai_client_tool_compat` lowers rich client tools to private `function`
/// declarations (#1131).
///
/// The backend echoes the lowered request shape back in `response.tools` and
/// `response.tool_choice`. Restoring these two fields from the snapshot keeps
/// the canonical client-owned declarations (and private lowered names such as
/// `agentic_ns__{ns}__{member}`) out of client-visible output. Captured once on
/// the first lowered round and reused unchanged across IRR continuations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ClientToolEcho {
    /// The client's original `tools` array, verbatim.
    pub tools: Vec<serde_json::Value>,
    /// The client's original `tool_choice`, verbatim; `Null` when it was absent.
    pub tool_choice: serde_json::Value,
}

/// Return whether an output item consumes the response-wide built-in tool-call budget.
///
/// All local built-in dispatchers share this classifier so adding a provider
/// call type cannot make their `max_tool_calls` accounting disagree.
///
/// MCP calls (`mcp_call`, `mcp_approval_request`, `mcp_list_tools`) and the
/// pre-dispatch `function_call` items that encode them are deliberately absent:
/// the OpenAI Responses API scopes `max_tool_calls` to *built-in* tools and
/// keeps MCP tools on their own limits (see `openai_mcp_dispatch`'s
/// `max_calls_per_round`). Counting MCP here would let it silently starve the
/// built-in budget, which the provider contract forbids.
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

/// Count calls consumed before the current model round began.
///
/// Current-round calls are admitted separately in model output order. This
/// helper therefore excludes the current suffix of `accumulated_output` while
/// still including prior file-search and web-search executions retained earlier
/// in `accumulated_output`.
pub(crate) fn consumed_builtin_tool_calls_before_current_round(state: &ResponsesState) -> usize {
    let prior_end = state
        .current_round_output_start
        .unwrap_or(state.accumulated_output.len());
    let prior_agentic_output = state.accumulated_output.get(..prior_end).unwrap_or_default();
    let owners = [prior_agentic_output];
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
    reason = "request-scoped state bag; the request, transport, and deferred \
              lifecycle bool flags (history_rehydrated, parallel_tool_calls, \
              store_persist_armed, previous_response_id_stream_restore_armed) are \
              independent request facts, not a state machine or refactorable enum"
)]
pub(crate) struct ResponsesState {
    /// Request-wide aggregate retained-payload ceiling selected by
    /// `openai_agentic_loop`. `None` until the first loop instance arms it.
    /// Additional loop instances may only lower the limit.
    pub(crate) retained_payload_limit: Option<usize>,

    /// Payload retained by request-lifetime sibling filter state that is not
    /// represented directly in this bag (currently the response-store input
    /// snapshot). The loop owner refreshes this before initial admission.
    pub(crate) retained_external_payload_bytes: usize,

    /// Whether aggregate admission failed and only a terminal error may remain.
    pub(crate) retained_payload_failed: bool,

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
    pub current_round_output_start: Option<usize>,

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

    /// Configured MCP connectors waiting for `tool_search` discovery.
    ///
    /// Populated by `openai_mcp_tool_resolve` when a request uses
    /// `connector_id` with `defer_loading: true`. Consumed by
    /// `openai_mcp_dispatch` on a later `tool_search_call` to load
    /// definitions from the internally resolved endpoint. Holds the
    /// pipeline-local URL and credentials; never serialized to the
    /// inference backend, client responses, or persisted records.
    pub deferred_mcp: Vec<DeferredMcpConnector>,

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

    /// Reverse map from a lowered private `function` tool name to the canonical
    /// client-owned tool it restores to, for a function-only Responses backend.
    ///
    /// Populated by `openai_client_tool_compat` during request lowering (#1131)
    /// and read by its buffered restoration and by `openai_stream_events` on the
    /// streaming path. Empty for native passthrough. Bounded by the declared tool
    /// count. Private lowered names never leak to client output.
    pub client_tool_lowering: HashMap<String, LoweredClientTool>,

    /// Original client-declared `tools`/`tool_choice`, snapshotted before
    /// lowering so the echoed `response.tools`/`response.tool_choice` can be
    /// restored to their canonical shapes without leaking private `function`
    /// names into client-visible output (#1131). `None` for native passthrough
    /// (nothing lowered). Set once on the first lowered round and reused across
    /// IRR continuations.
    pub client_tool_echo: Option<ClientToolEcho>,

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

    /// Number of leading messages already persisted by a provider-owned
    /// conversation. Internal continuations send only the remaining delta.
    pub provider_history_len: usize,

    /// Whether tool calls may execute concurrently within an
    /// iteration. Defaults to `true` per the API spec.
    pub parallel_tool_calls: bool,

    /// Full message history to persist for future rehydration.
    ///
    /// This may include output-only metadata items omitted from
    /// [`Self::messages`] because it is not forwarded to backend
    /// inference.
    pub persisted_messages: Vec<serde_json::Value>,

    /// Server-owned pending MCP approvals emitted during this request.
    ///
    /// Populated by `mcp_dispatch` when it pauses on an
    /// `mcp_approval_request`, this is drained by the store filter and
    /// persisted as the authoritative record for correlating a later
    /// `mcp_approval_response`. Consent provenance lives here and in the
    /// store, never in the (client-influenced) conversation history.
    pub pending_approvals: Vec<crate::store::PendingApprovalRecord>,

    /// Whether the store filter armed persistence for this exchange.
    ///
    /// Set by `openai_response_store` during the request phase only after it
    /// initializes and registers a backend AND classifies this request as one
    /// whose response will be persisted. `mcp_dispatch` reads this
    /// exchange-scoped marker before emitting an `mcp_approval_request`: unlike
    /// pipeline-scoped registry membership, it proves the store filter actually
    /// ran and intends to persist THIS response, so it also catches a store
    /// filter that is absent, request-conditioned out, or ordered after
    /// dispatch. It cannot observe a response-phase persistence skip (a
    /// `response_conditions`-gated store filter, a non-2xx status, etc.); that
    /// narrower residual is unsupported for approval pipelines and still fails
    /// closed at resume.
    pub store_persist_armed: bool,

    /// Whether the streaming `previous_response_id` wire rewrite was armed.
    ///
    /// Set in the response header phase by `openai_responses_rehydrate`
    /// (`arm_streaming_restore`) to the result of `eligible_previous_response_id_stream`:
    /// `true` only for a `200 OK`, identity-coded, validator-free event stream from
    /// a rehydrated turn carrying a caller id. The persistence source
    /// (`canonicalize_logical_response`) runs in the body phase, where the response
    /// header is gone, so it reads this precomputed flag to restore the id into the
    /// stored `response_object` on exactly the streams whose client-visible frames
    /// the wire path rewrote — never on a validator-bearing or non-200 stream the
    /// wire path left untouched, which would make a later GET disagree with the
    /// terminal frame (issue #1150 review).
    pub previous_response_id_stream_restore_armed: bool,

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

    /// Prior streamed terminal response retained across request-side re-entry.
    ///
    /// `openai_stream_events` invalidates [`Self::response_object`] before the
    /// next inference round so an upstream error cannot persist stale success.
    /// A dispatcher can instead complete locally before that inference starts
    /// (for example, an MCP approval or exhausted tool budget), so the prior
    /// response metadata is moved here until the new upstream response begins.
    /// Its output has already been drained into [`Self::accumulated_output`].
    pub local_completion_response_template: serde_json::Value,

    /// Tool calls selected for dispatch from the current inference response.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration
    /// before `stream_events` writes new ones. Without explicit
    /// clearing, stale tool calls from a previous iteration cause
    /// duplicate dispatch.
    /// Provider-produced calls are fixed-size absolute assignments into
    /// [`Self::accumulated_output`]. Approval resumptions use a compact typed
    /// invocation because no provider output item owns their arguments.
    pub tool_calls: Vec<ToolCallAssignment>,

    /// `tool_search_call` items from the current inference response.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration.
    /// `openai_mcp_dispatch` consumes these to load deferred connector
    /// tools before the next inference round.
    pub tool_search_calls: Vec<OutputAssignment>,

    /// Web search calls from the current inference response only.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration.
    /// Stored separately from `tool_calls` because `web_search_call`
    /// items have a different shape (`action.query` instead of
    /// `name`/`arguments`) and are dispatched by a different filter.
    pub web_search_calls: Vec<WebSearchAssignment>,

    /// Cumulative web searches dispatched to the provider across all
    /// agentic-loop iterations.
    ///
    /// This is operational execution state, not `max_tool_calls` accounting:
    /// the response-wide API limit charges retained model-call admissions,
    /// including calls whose local execution is incomplete.
    pub web_search_calls_executed: u32,

    /// Absolute indices into [`Self::accumulated_output`] (+ synthesis origin)
    /// of the `file_search_call` items `openai_agentic_loop` accumulated this
    /// round for `openai_file_search_callout` to execute.
    ///
    /// The parse owner records one [`FileSearchAssignment`] per hosted file-search
    /// call it appended to `accumulated_output` (including private
    /// `function_call(name=file_search)` items it normalized into canonical
    /// `file_search_call` shape). The dispatcher drains these exactly once at
    /// request-body EOS and mutates the indexed item in place, so the call
    /// payload is never cloned into a second owner. Cleared by draining, mirroring
    /// [`Self::pending_local_tool_synthesis`].
    pub file_search_assignments: Vec<FileSearchAssignment>,

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

    /// Aggregate wire bytes `openai_stream_events` has charged against its
    /// accumulation budget across every IRR round of this request.
    ///
    /// Request-wide (not per-round) because the memory it bounds —
    /// [`Self::accumulated_output`], [`Self::emitted_output_items`], and the
    /// canonical terminal tree — persists across rounds: resetting the counter each
    /// round would let a multi-round stream accumulate unbounded state while no
    /// single round trips the cap. Monotonically non-decreasing, so once the
    /// budget is exceeded the stream stays failed closed. A fixed `usize`, so it
    /// contributes nothing meaningful to `max_state_bytes`.
    pub stream_accumulated_bytes: usize,

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

    /// Absolute `output_index` values into `accumulated_output` (+ origin) for the
    /// items `openai_file_search_callout` reconciled this round on the streaming
    /// path. Drained exactly once by `stream_events` at finalize (§4.2). Index + a
    /// 1-byte tag (no owned `Value`) so it needs no separate `continuation_state_fits` charge.
    pub pending_local_tool_synthesis: Vec<(usize, SynthesisKind)>,

    /// Provider `file_search_call` item ids whose terminal lifecycle
    /// `stream_events` already streamed live — a native hybrid whose terminal
    /// `output_item.done` passed through, cancelling EOS suppression (§6).
    /// The streaming reconcile reads this to skip re-queuing such a call for
    /// synthesis; a synthesized tail would emit a DUPLICATE terminal
    /// `output_item.done` (#313 P1). Recorded for ANY provider-terminal status
    /// (completed/failed/incomplete): the set membership — not the status — is
    /// authoritative. A callout-terminalized `incomplete` call is absent here
    /// (its live done was suppressed, never passed through) and still synthesizes.
    /// Keyed by item id (stable across the streamed done and the accumulated item).
    ///
    /// Lifecycle is one IRR round: `stream_events` records into it during the round's
    /// chunks, `file_search`'s EOS reconcile reads it, then `finalize_logical_stream`
    /// clears it (§6). The ids are stale after their round — they never re-match a later
    /// round's items — so clearing loses nothing and bounds the set. It is also charged
    /// against `max_state_bytes` in `continuation_state_fits` like every other
    /// request-scoped field (#313 P1 `DoS` bound).
    pub provider_streamed_terminal_ids: BTreeSet<String>,

    /// A terminal failure a request-phase dispatcher recorded for the parse owner
    /// to convert into the single client-facing rejection (buffered JSON before
    /// commitment, logical-stream SSE error after). `None` on the happy path.
    ///
    /// Keeping the terminal decision with `openai_agentic_loop` prevents a
    /// dispatcher from becoming a second terminal-response owner (see
    /// [`DispatchFailure`]).
    pub dispatch_failure: Option<DispatchFailure>,
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

/// Internally resolved MCP connector waiting for deferred discovery.
#[derive(Clone)]
pub(crate) struct DeferredMcpConnector {
    /// Allow loopback MCP endpoints for this listing.
    pub allow_loopback: bool,

    /// Request `authorization` forwarded to the MCP endpoint.
    pub authorization: Option<String>,

    /// Original `allowed_tools` filter from the request entry.
    pub allowed_tools: Option<serde_json::Value>,

    /// Pipeline-local connector identifier from the request.
    pub connector_id: String,

    /// Request `headers` forwarded to the MCP endpoint.
    pub headers: Option<serde_json::Value>,

    /// Maximum size of the provider-visible body after deferred expansion.
    pub max_rewritten_body_bytes: usize,

    /// Maximum tools accepted from a single `tools/list` response.
    pub max_tools: usize,

    /// Request `require_approval` policy preserved for later dispatch.
    pub require_approval: Option<serde_json::Value>,

    /// Public server label used in function-name encoding and dispatch.
    pub server_label: String,

    /// Configured MCP endpoint URL. Never written to backend requests,
    /// client-visible responses, logs, or persisted response state.
    pub server_url: String,

    /// Per-server timeout for the deferred `tools/list` call.
    pub timeout: Duration,
}

impl fmt::Debug for DeferredMcpConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferredMcpConnector")
            .field("allow_loopback", &self.allow_loopback)
            .field("authorization", &self.authorization.as_ref().map(|_| "<redacted>"))
            .field("allowed_tools", &self.allowed_tools)
            .field("connector_id", &self.connector_id)
            .field("headers", &self.headers.as_ref().map(|_| "<redacted>"))
            .field("max_rewritten_body_bytes", &self.max_rewritten_body_bytes)
            .field("max_tools", &self.max_tools)
            .field("require_approval", &self.require_approval)
            .field("server_label", &self.server_label)
            .field("server_url", &"<redacted>")
            .field("timeout", &self.timeout)
            .finish()
    }
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
            retained_payload_limit: None,
            retained_external_payload_bytes: 0,
            retained_payload_failed: false,
            citation_files: HashMap::new(),
            context_management: None,
            conversation: None,
            include: Vec::new(),
            logical_stream_response_id: None,
            logical_stream_sequence: 0,
            current_round_output_start: None,
            history_rehydrated: false,
            input: Vec::new(),
            iteration: 0,
            deferred_mcp: Vec::new(),
            max_tool_calls: None,
            mcp_approval_state: McpApprovalState::None,
            deferred_tool_limit_completion: false,
            deferred_stream_done: false,
            mcp_tool_map: HashMap::new(),
            client_tool_lowering: HashMap::new(),
            client_tool_echo: None,
            messages: Vec::new(),
            provider_history_len: 0,
            parallel_tool_calls: true,
            persisted_messages: Vec::new(),
            pending_approvals: Vec::new(),
            store_persist_armed: false,
            previous_response_id_stream_restore_armed: false,
            previous_response_id: None,
            previous_tools: Vec::new(),
            previous_usage: None,
            original_tool_choice: None,
            response_created_at: None,
            response_id: None,
            request_body: serde_json::Value::Null,
            request_body_rebuild: RequestBodyRebuild::PreserveOriginal,
            response_object: serde_json::Value::Null,
            local_completion_response_template: serde_json::Value::Null,
            tool_calls: Vec::new(),
            tool_search_calls: Vec::new(),
            web_search_calls: Vec::new(),
            web_search_calls_executed: 0,
            file_search_assignments: Vec::new(),
            tool_choice: serde_json::Value::String("auto".to_owned()),
            tools: Vec::new(),
            usage: serde_json::Value::Null,
            accumulated_output: Vec::new(),
            stream_accumulated_bytes: 0,
            emitted_output_items: HashMap::new(),
            locally_executed_output_items: HashSet::new(),
            pending_local_tool_synthesis: Vec::new(),
            provider_streamed_terminal_ids: BTreeSet::new(),
            dispatch_failure: None,
        }
    }
}

impl ResponsesState {
    /// Apply an aggregate retained-payload limit. The smallest limit seen by
    /// this request wins, so conditionally composed loop instances cannot
    /// weaken an earlier safety policy.
    pub(crate) fn apply_retained_payload_limit(&mut self, limit: usize) {
        self.retained_payload_limit = Some(self.retained_payload_limit.map_or(limit, |current| current.min(limit)));
    }

    /// Record payload retained for the request by sibling filter state.
    pub(crate) fn set_retained_external_payload_bytes(&mut self, bytes: usize) {
        self.retained_external_payload_bytes = bytes;
    }

    /// Transfer an owned payload out of this bag while keeping it charged to
    /// the request-wide meter for the lifetime of its filter-local owner.
    pub(crate) fn retain_external_payload_bytes(&mut self, bytes: usize) -> bool {
        let Some(total) = self.retained_external_payload_bytes.checked_add(bytes) else {
            return false;
        };
        self.retained_external_payload_bytes = total;
        true
    }

    /// Release payload previously transferred to a filter-local owner.
    pub(crate) fn release_external_payload_bytes(&mut self, bytes: usize) {
        self.retained_external_payload_bytes = self.retained_external_payload_bytes.saturating_sub(bytes);
    }

    /// Return the active request-wide retained-payload limit.
    pub(crate) const fn retained_payload_limit(&self) -> Option<usize> {
        self.retained_payload_limit
    }

    /// Count every payload independently owned by this state.
    ///
    /// JSON values are charged by compact serialized size. Values held in
    /// multiple collections are deliberately counted once per owner. Native
    /// strings and byte-like buffers are charged by their raw length. Fixed-size
    /// counters, flags, indices, and digests carry no payload charge.
    #[cfg(test)]
    pub(crate) fn retained_payload_bytes(&self) -> Option<usize> {
        self.retained_payload_bytes_bounded(usize::MAX)
    }

    /// Count retained payload, returning `None` immediately above `max_bytes`.
    pub(crate) fn retained_payload_bytes_bounded(&self, max_bytes: usize) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, true)
    }

    /// Count payload owned directly by this state, excluding sibling-filter
    /// owners that participate only in the aggregate agentic budget.
    ///
    /// File search's legacy `max_state_bytes` option predates the aggregate
    /// budget and covers iterative-router plus file-search continuation state;
    /// response-store snapshots and other sibling-filter owners must not change
    /// that independent compatibility limit.
    pub(crate) fn retained_payload_bytes_bounded_without_external(&self, max_bytes: usize) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, false)
    }

    /// Shared implementation for aggregate and state-only payload accounting.
    #[expect(
        clippy::too_many_lines,
        reason = "exhaustive accounting for the request-scoped state bag"
    )]
    fn retained_payload_bytes_bounded_inner(&self, max_bytes: usize, include_external: bool) -> Option<usize> {
        let mut meter = PayloadMeter::new(max_bytes);
        if include_external {
            meter.raw(self.retained_external_payload_bytes)?;
        }

        for value in [
            &self.request_body,
            &self.response_object,
            &self.local_completion_response_template,
            &self.tool_choice,
            &self.usage,
        ] {
            meter.json(value)?;
        }
        for values in [
            &self.accumulated_output,
            &self.input,
            &self.messages,
            &self.persisted_messages,
            &self.previous_tools,
            &self.tools,
        ] {
            meter.json_values(values)?;
        }
        for call in &self.tool_calls {
            if let ToolCallAssignment::Approved(invocation) = call {
                meter.raw(
                    invocation
                        .encoded_name
                        .len()
                        .saturating_add(invocation.approval_id.len())
                        .saturating_add(invocation.arguments.len()),
                )?;
            }
        }
        for value in [
            self.context_management.as_ref(),
            self.conversation.as_ref(),
            self.original_tool_choice.as_ref(),
            self.previous_usage.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            meter.json(value)?;
        }
        for ((server, tool), value) in &self.mcp_tool_map {
            meter.raw(server.len().saturating_add(tool.len()))?;
            meter.json(value)?;
        }
        meter.raw(
            self.citation_files
                .iter()
                .map(|(key, value)| key.len().saturating_add(value.len()))
                .chain(self.include.iter().map(String::len))
                .chain(self.logical_stream_response_id.iter().map(String::len))
                .chain(self.previous_response_id.iter().map(String::len))
                .chain(self.response_id.iter().map(String::len))
                .chain(self.provider_streamed_terminal_ids.iter().map(String::len))
                .chain(self.locally_executed_output_items.iter().map(String::len))
                .sum(),
        )?;
        for (id, emitted) in &self.emitted_output_items {
            meter.raw(id.len())?;
            meter.raw(emitted.streamed_phases.iter().map(String::len).sum())?;
        }
        for record in &self.pending_approvals {
            meter.raw(
                record
                    .approval_id
                    .len()
                    .saturating_add(record.server_label.len())
                    .saturating_add(record.tool_name.len())
                    .saturating_add(record.arguments.len())
                    .saturating_add(record.target_fingerprint.len()),
            )?;
        }
        if let Some(failure) = &self.dispatch_failure {
            meter.raw(failure.message.len())?;
        }
        Some(meter.used())
    }

    /// Return whether the current state plus independently owned payload bytes
    /// fits the active aggregate limit.
    pub(crate) fn can_retain_payload(&self, additional_bytes: usize) -> bool {
        self.can_replace_retained_payload(0, additional_bytes, 0)
    }

    /// Test a transactional payload replacement without mutating state.
    ///
    /// `removed_bytes` identifies owners that will be dropped by the same
    /// commit, `added_bytes` the new owners, and `external_bytes` filter-local
    /// parser or serialization staging retained alongside `ResponsesState`.
    pub(crate) fn can_replace_retained_payload(
        &self,
        removed_bytes: usize,
        added_bytes: usize,
        external_bytes: usize,
    ) -> bool {
        let Some(limit) = self.retained_payload_limit else {
            return true;
        };
        let Some(current) = self.retained_payload_bytes_bounded(limit.saturating_add(removed_bytes)) else {
            return false;
        };
        current
            .saturating_sub(removed_bytes)
            .saturating_add(added_bytes)
            .saturating_add(external_bytes)
            <= limit
    }

    /// Drop all dispatch selections and disable successful persistence after an
    /// aggregate budget failure.
    pub(crate) fn fail_retained_payload_budget(&mut self) {
        self.retained_payload_failed = true;
        self.tool_calls.clear();
        self.web_search_calls.clear();
        self.file_search_assignments.clear();
        self.pending_local_tool_synthesis.clear();
        self.deferred_tool_limit_completion = false;
        self.mcp_approval_state = McpApprovalState::None;
        self.store_persist_armed = false;
    }

    /// Release request payload after an already-committed stream exceeds the
    /// aggregate budget. Only the transport bit and budget remain live; no
    /// inference, dispatch, or successful persistence may follow this terminal
    /// error, so retaining conversation or response trees would serve no owner.
    pub(crate) fn discard_payload_for_budget_error(&mut self) {
        let streaming = self.request_body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
        self.fail_retained_payload_budget();
        self.citation_files.clear();
        self.context_management = None;
        self.conversation = None;
        self.include.clear();
        self.logical_stream_response_id = None;
        self.input.clear();
        self.mcp_tool_map.clear();
        self.messages.clear();
        self.persisted_messages.clear();
        self.pending_approvals.clear();
        self.previous_response_id = None;
        self.previous_tools.clear();
        self.previous_usage = None;
        self.original_tool_choice = None;
        self.response_id = None;
        self.request_body = serde_json::json!({ "stream": streaming });
        self.response_object = serde_json::Value::Null;
        self.local_completion_response_template = serde_json::Value::Null;
        self.tool_choice = serde_json::Value::Null;
        self.tools.clear();
        self.usage = serde_json::Value::Null;
        self.accumulated_output.clear();
        self.emitted_output_items.clear();
        self.locally_executed_output_items.clear();
        self.provider_streamed_terminal_ids.clear();
        self.dispatch_failure = None;
    }

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
            provider_history_len: 0,
            parallel_tool_calls: extract_bool_or(&body, "parallel_tool_calls", true),
            persisted_messages,
            previous_response_id: extract_string(&body, "previous_response_id"),
            request_body: body,
            tool_choice,
            tools,
            accumulated_output: Vec::new(),
            pending_local_tool_synthesis: Vec::new(),
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

    /// Finalize the response into [`Self::response_object`] and serialize it to
    /// `body`.
    ///
    /// Moves the full multi-round [`Self::accumulated_output`] into
    /// `response_object["output"]` via `mem::take` (avoiding a clone per the
    /// data-ownership rule), stamps accumulated usage, applies bounded citation
    /// annotation, enforces the JSON size bound, and writes the serialized bytes
    /// back to `body`. Reflecting the merged output in `response_object` keeps
    /// state-based consumers (streaming reader, persistence) consistent with the
    /// body bytes, instead of leaving `response_object` holding only the last
    /// round's output.
    ///
    /// Citation annotation is invoked **unconditionally**: the citations-present
    /// gate lives inside [`annotate_response`], which is a strict no-op when
    /// [`Self::citation_files`] is empty, so web- and MCP-only flows perform no
    /// annotation and need no caller-side conditional.
    ///
    /// This is a **terminal** finalizer: it consumes `accumulated_output`, so it
    /// must run only on a loop-exit transition, never on a loop-back round (a
    /// loop-back round's serialized body is discarded by the router, so it is not
    /// written at all).
    ///
    /// # Errors
    ///
    /// Returns a [`FilterAction::Reject`] carrying an HTTP 502 error envelope on
    /// citation-annotation failure, JSON size overflow, or serialization
    /// failure, closing the prior fail-open serialization gap.
    #[expect(
        clippy::too_many_lines,
        reason = "in-place canonical response finalization with budget preflight"
    )]
    pub(crate) fn finalize_response_body(&mut self, body: &mut Option<Bytes>) -> Result<(), FilterAction> {
        if !self.response_object.is_object() {
            return Ok(());
        }
        let final_output = if self.accumulated_output.is_empty() {
            self.response_object
                .get("output")
                .and_then(serde_json::Value::as_array)
                .map_or(&[][..], Vec::as_slice)
        } else {
            &self.accumulated_output
        };
        let annotation_staging = annotation_staging_bytes(final_output, &self.citation_files).map_err(|error| {
            tracing::warn!(%error, "failed to preflight final response annotations");
            finalize_rejection("failed to annotate final response")
        })?;
        let usage_staging = if self.usage.is_null() {
            0
        } else {
            retained_json_bytes(&self.usage).unwrap_or(usize::MAX)
        };
        let preflight_staging = annotation_staging.saturating_add(usage_staging);
        if !self.can_retain_payload(preflight_staging) {
            self.fail_retained_payload_budget();
            return Err(finalize_rejection(
                "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during final construction",
            ));
        }
        // Build the canonical terminal tree off-state using moves. This keeps
        // the retained-state mutation transactional through annotation and both
        // size admissions; on failure no oversized response or serialization
        // buffer becomes an additional state owner.
        let mut response = std::mem::take(&mut self.response_object);
        if let Some(obj) = response.as_object_mut() {
            if !self.accumulated_output.is_empty() {
                obj.insert(
                    "output".to_owned(),
                    serde_json::Value::Array(std::mem::take(&mut self.accumulated_output)),
                );
            }
            if !self.usage.is_null() {
                obj.insert("usage".to_owned(), std::mem::take(&mut self.usage));
            }
        }
        // The assignments point into `accumulated_output`, whose owner has just
        // moved into the terminal response. Finalization never dispatches them.
        self.tool_calls.clear();
        self.tool_search_calls.clear();
        self.web_search_calls.clear();
        if let Err(error) = annotate_response(&mut response, &self.citation_files) {
            tracing::warn!(%error, "failed to annotate final response");
            self.response_object = response;
            return Err(finalize_rejection("failed to annotate final response"));
        }
        let Some(serialized_bytes) = bounded_json_size(&response, MAX_JSON_BODY_BYTES).ok().flatten() else {
            self.response_object = response;
            return Err(finalize_rejection(
                "final response exceeds the JSON response byte limit",
            ));
        };
        // `response_object` and the final byte buffer coexist until the
        // framework accepts the rewritten body, so charge both owners before
        // allocating the buffer. `self.response_object` is temporarily `null`.
        let null_bytes = retained_json_bytes(&self.response_object).unwrap_or(0);
        if !self.can_replace_retained_payload(null_bytes, serialized_bytes, serialized_bytes) {
            // Do not commit the canonicalized/annotated tree after its final
            // serialization owner is denied. The local `response` is dropped
            // on return and all request payload is released; only the bounded
            // terminal-error state remains available to downstream filters.
            self.discard_payload_for_budget_error();
            return Err(finalize_rejection(
                "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during final serialization",
            ));
        }
        // Commit the canonical tree before serialization, release the obsolete
        // upstream buffered body, and serialize the sole tree by reference.
        self.response_object = response;
        *body = None;
        let serialized = match serde_json::to_vec(&self.response_object) {
            Ok(serialized) => serialized,
            Err(error) => {
                tracing::warn!(%error, "failed to encode final response");
                return Err(finalize_rejection("failed to encode final response"));
            },
        };
        *body = Some(Bytes::from(serialized));
        Ok(())
    }

    /// Move the pending local-tool synthesis queue out, leaving it empty.
    pub fn drain_pending_local_tool_synthesis(&mut self) -> Vec<(usize, SynthesisKind)> {
        std::mem::take(&mut self.pending_local_tool_synthesis)
    }

    /// Move the file-search assignment queue out, leaving it empty.
    ///
    /// `openai_file_search_callout` drains this exactly once at request-body EOS
    /// so each assigned `file_search_call` is executed and reconciled a single
    /// time (drain-once), mirroring [`Self::drain_pending_local_tool_synthesis`].
    pub fn drain_file_search_assignments(&mut self) -> Vec<FileSearchAssignment> {
        std::mem::take(&mut self.file_search_assignments)
    }

    /// Resolve one fixed-size output assignment against the canonical
    /// accumulator.
    pub(crate) fn assigned_output(&self, assignment: OutputAssignment) -> Option<&serde_json::Value> {
        self.accumulated_output.get(assignment.output_index)
    }
}

/// Allocation-free compact-JSON/raw-byte accumulator shared by all aggregate
/// retained-state checks.
pub(crate) struct PayloadMeter {
    /// Compact JSON and raw payload bytes admitted so far.
    used: usize,

    /// Inclusive request-wide byte ceiling.
    limit: usize,
}

impl PayloadMeter {
    /// Start an empty bounded measurement.
    pub(crate) const fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }

    /// Charge one independently owned JSON value.
    pub(crate) fn json<T: serde::Serialize + ?Sized>(&mut self, value: &T) -> Option<()> {
        let remaining = self.limit.saturating_sub(self.used);
        let bytes = bounded_json_size(value, remaining).ok().flatten()?;
        self.raw(bytes)
    }

    /// Charge each independently owned JSON value in a collection.
    pub(crate) fn json_values(&mut self, values: &[serde_json::Value]) -> Option<()> {
        for value in values {
            self.json(value)?;
        }
        Some(())
    }

    /// Charge raw string or buffer bytes.
    pub(crate) fn raw(&mut self, bytes: usize) -> Option<()> {
        self.used = self.used.checked_add(bytes)?;
        (self.used <= self.limit).then_some(())
    }

    /// Return charged bytes.
    pub(crate) const fn used(&self) -> usize {
        self.used
    }
}

/// Return the compact representation size of one independently owned JSON
/// value, or `None` on serialization/count overflow.
pub(crate) fn retained_json_bytes<T: serde::Serialize + ?Sized>(value: &T) -> Option<usize> {
    bounded_json_size(value, usize::MAX).ok().flatten()
}

/// Return the sum of compact sizes for independently owned JSON values.
pub(crate) fn retained_json_values_bytes(values: &[serde_json::Value]) -> Option<usize> {
    let mut meter = PayloadMeter::new(usize::MAX);
    meter.json_values(values)?;
    Some(meter.used())
}

/// Build the HTTP 502 rejection returned by [`ResponsesState::finalize_response_body`]
/// when the terminal response cannot be annotated, size-bounded, or serialized.
fn finalize_rejection(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(502, "server_error", message))
}

/// Return budget admission decisions for one kind of current-round built-in call.
///
/// Dispatch filters run independently on request re-entry, but the response-wide
/// `max_tool_calls` limit applies in model output order. Replaying the immutable
/// current response here prevents pipeline order from letting a later web search
/// displace an earlier built-in call (or vice versa). MCP calls are exempt from
/// this budget (see [`is_builtin_tool_call`]), so they never appear as budget
/// consumers even when interleaved with the target calls in model output order.
#[cfg(test)]
pub(crate) fn current_round_tool_call_admissions(
    state: &ResponsesState,
    target_calls: &[serde_json::Value],
) -> Vec<bool> {
    current_round_tool_call_admissions_by(state, target_calls.len(), |index| target_calls.get(index))
}

/// Return ordered built-in-budget admissions for web-search assignments.
pub(crate) fn current_round_web_search_admissions(
    state: &ResponsesState,
    assignments: &[WebSearchAssignment],
) -> Vec<bool> {
    current_round_output_assignment_admissions(state, assignments.iter().map(|assignment| assignment.output_index))
}

/// Return ordered built-in-budget admissions for generic output assignments.
pub(crate) fn current_round_output_assignment_admissions(
    state: &ResponsesState,
    assignments: impl IntoIterator<Item = usize>,
) -> Vec<bool> {
    let targets = assignments.into_iter().collect::<Vec<_>>();
    let previous_calls = consumed_builtin_tool_calls_before_current_round(state);
    let mut remaining = state.max_tool_calls.map_or(usize::MAX, |limit| {
        usize::try_from(limit)
            .unwrap_or(usize::MAX)
            .saturating_sub(previous_calls)
    });
    let mut admissions = Vec::with_capacity(targets.len());
    let mut target = 0;
    let round_start = state
        .current_round_output_start
        .unwrap_or(state.accumulated_output.len());

    for (offset, item) in state
        .accumulated_output
        .get(round_start..)
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let admitted = !is_builtin_tool_call(item) || remaining > 0;
        if is_builtin_tool_call(item) {
            remaining = remaining.saturating_sub(1);
        }
        let absolute = round_start.saturating_add(offset);
        if targets.get(target).copied() == Some(absolute) {
            admissions.push(admitted);
            target = target.saturating_add(1);
        }
    }
    admissions
}

/// Return budget admission decisions for file-search assignments in model order.
///
/// File-search dispatch uses absolute accumulator indices rather than cloned
/// output items. Replaying those indices against the current-round suffix keeps
/// its admission decisions identical to web search even when a streamed round
/// has already been drained out of `response_object.output`.
#[expect(
    clippy::too_many_lines,
    reason = "ordered cross-dispatch admission is clearer as one linear model-output scan"
)]
pub(crate) fn current_round_file_search_admissions(
    state: &ResponsesState,
    assignments: &[FileSearchAssignment],
) -> Vec<bool> {
    let previous_calls = consumed_builtin_tool_calls_before_current_round(state);
    let mut remaining = state.max_tool_calls.map_or(usize::MAX, |limit| {
        usize::try_from(limit)
            .unwrap_or(usize::MAX)
            .saturating_sub(previous_calls)
    });
    let mut admissions = Vec::with_capacity(assignments.len());
    let mut assignment_index = 0;
    let round_start = state
        .current_round_output_start
        .unwrap_or(state.accumulated_output.len());

    for (offset, item) in state
        .accumulated_output
        .get(round_start..)
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let consumes_budget = is_builtin_tool_call(item);
        let admitted = !consumes_budget || remaining > 0;
        if consumes_budget {
            remaining = remaining.saturating_sub(1);
        }
        let absolute_index = round_start.saturating_add(offset);
        if assignments
            .get(assignment_index)
            .is_some_and(|assignment| assignment.output_index == absolute_index)
        {
            admissions.push(admitted);
            assignment_index = assignment_index.saturating_add(1);
        }
    }
    admissions
}

/// Replay model output order against an arbitrary borrowed target-call view.
#[cfg(test)]
fn current_round_tool_call_admissions_by<'a>(
    state: &ResponsesState,
    target_count: usize,
    target_call: impl Fn(usize) -> Option<&'a serde_json::Value>,
) -> Vec<bool> {
    let previous_calls = consumed_builtin_tool_calls_before_current_round(state);
    let mut remaining = state.max_tool_calls.map_or(usize::MAX, |limit| {
        usize::try_from(limit)
            .unwrap_or(usize::MAX)
            .saturating_sub(previous_calls)
    });
    let mut admissions = Vec::new();
    let mut target_index = 0;

    let round_start = state
        .current_round_output_start
        .unwrap_or(state.accumulated_output.len());
    for item in state.accumulated_output.get(round_start..).unwrap_or_default() {
        let consumes_budget = is_builtin_tool_call(item);
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

/// Whether a queued hosted `tool_search_call` may still consume built-in budget.
///
/// Deferred connector discovery is the server-side execution of that search, so
/// it must not run `tools/list` or start another inference round after
/// `max_tool_calls` is exhausted. An omitted limit leaves discovery allowed.
/// When the current round's output is not yet replayable, remaining prior-round
/// budget is the admission signal.
pub(crate) fn tool_search_discovery_is_within_budget(state: &ResponsesState) -> bool {
    let Some(max) = state.max_tool_calls else {
        return true;
    };
    let remaining = usize::try_from(max)
        .unwrap_or(usize::MAX)
        .saturating_sub(consumed_builtin_tool_calls_before_current_round(state));
    if remaining == 0 {
        return false;
    }
    let admissions = current_round_output_assignment_admissions(
        state,
        state.tool_search_calls.iter().map(|assignment| assignment.output_index),
    );
    admissions.is_empty() || admissions.into_iter().any(|admitted| admitted)
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
    fn retained_payload_admits_below_and_at_limit_but_rejects_above() {
        let item = json!({"payload": "abc"});
        let item_bytes = retained_json_bytes(&item).unwrap();
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        state.apply_retained_payload_limit(baseline + item_bytes);

        assert!(state.can_retain_payload(item_bytes - 1), "below limit");
        assert!(state.can_retain_payload(item_bytes), "at limit");
        assert!(!state.can_retain_payload(item_bytes + 1), "above limit");
    }

    #[test]
    fn retained_payload_counts_duplicate_json_owners_separately() {
        let item = json!({"type": "message", "content": "owned four times"});
        let bytes = retained_json_bytes(&item).unwrap();
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        state.input.push(item.clone());
        state.messages.push(item.clone());
        state.persisted_messages.push(item.clone());
        state.accumulated_output.push(item);

        assert_eq!(state.retained_payload_bytes().unwrap() - baseline, bytes * 4);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "table-driven coverage of every retained owner class"
    )]
    fn retained_payload_counts_response_history_dispatch_and_raw_state() {
        let value = json!({"payload": "v"});
        let bytes = retained_json_bytes(&value).unwrap();
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        state.response_object = value.clone();
        state.context_management = Some(value.clone());
        state.conversation = Some(value.clone());
        state.previous_usage = Some(value.clone());
        state.original_tool_choice = Some(value.clone());
        state.previous_tools.push(value.clone());
        state
            .tool_calls
            .push(ToolCallAssignment::Output(OutputAssignment { output_index: 0 }));
        state.web_search_calls.push(WebSearchAssignment {
            output_index: 0,
            ordinal: 0,
        });
        state.tools.push(value.clone());
        state
            .mcp_tool_map
            .insert(("server".to_owned(), "tool".to_owned()), value);
        state.include.push("usage".to_owned());
        state.provider_streamed_terminal_ids.insert("terminal".to_owned());
        state.locally_executed_output_items.insert("executed".to_owned());
        state.pending_approvals.push(crate::store::PendingApprovalRecord {
            approval_id: "approval".to_owned(),
            server_label: "label".to_owned(),
            tool_name: "name".to_owned(),
            arguments: "args".to_owned(),
            target_fingerprint: "fingerprint".to_owned(),
        });
        state.dispatch_failure = Some(DispatchFailure {
            status: 502,
            code: "server_error",
            message: "failure".to_owned(),
        });

        let replaced_null = retained_json_bytes(&serde_json::Value::Null).unwrap();
        let json_delta = bytes * 8 - replaced_null;
        let raw_delta = "server".len()
            + "tool".len()
            + "usage".len()
            + "terminal".len()
            + "executed".len()
            + "approval".len()
            + "label".len()
            + "name".len()
            + "args".len()
            + "fingerprint".len()
            + "failure".len();
        assert_eq!(
            state.retained_payload_bytes().unwrap() - baseline,
            json_delta + raw_delta
        );
    }

    #[test]
    fn retained_payload_limit_can_only_decrease() {
        let mut state = ResponsesState::default();
        state.apply_retained_payload_limit(8_192);
        state.apply_retained_payload_limit(16_384);
        assert_eq!(state.retained_payload_limit(), Some(8_192));
        state.apply_retained_payload_limit(4_096);
        assert_eq!(state.retained_payload_limit(), Some(4_096));
    }

    #[test]
    fn retained_payload_counts_request_lifetime_external_owners() {
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        state.set_retained_external_payload_bytes(1_337);

        assert_eq!(state.retained_payload_bytes().unwrap(), baseline + 1_337);
    }

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
    fn mcp_calls_are_exempt_while_builtins_admit_in_output_order() {
        let mcp = json!({
            "type":"function_call", "call_id":"mcp_1",
            "name":"utilities__weather", "status":"completed"
        });
        let web_first = json!({"type":"web_search_call", "id":"ws_1", "status":"completed"});
        let web_second = json!({"type":"web_search_call", "id":"ws_2", "status":"completed"});
        let state = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: Some(0),
            accumulated_output: vec![mcp.clone(), web_first.clone(), web_second.clone()],
            response_object: json!({"output":[mcp.clone(), web_first.clone(), web_second.clone()]}),
            ..ResponsesState::default()
        };

        // The MCP call precedes both built-ins in model output order but is
        // exempt from `max_tool_calls`, so it never spends the single built-in
        // slot. The first web search claims that slot and the second is
        // rejected in order.
        assert_eq!(current_round_tool_call_admissions(&state, &[mcp]), vec![true]);
        assert_eq!(current_round_tool_call_admissions(&state, &[web_first]), vec![true]);
        assert_eq!(current_round_tool_call_admissions(&state, &[web_second]), vec![false]);
    }

    #[test]
    fn first_streamed_round_is_not_counted_as_prior_budget() {
        let web = json!({"type":"web_search_call", "id":"ws_first", "status":"in_progress"});
        let state = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: Some(0),
            accumulated_output: vec![web.clone()],
            response_object: json!({"output": []}),
            ..ResponsesState::default()
        };

        assert_eq!(
            consumed_builtin_tool_calls_before_current_round(&state),
            0,
            "the explicit zero boundary keeps the first streamed round current"
        );
        assert_eq!(
            current_round_tool_call_admissions(&state, &[web]),
            vec![true],
            "the first streamed web call must receive the available slot"
        );
    }

    #[test]
    fn file_and_web_calls_share_one_model_order_admission() {
        let file = json!({"type":"file_search_call", "id":"fs_first", "status":"searching"});
        let web = json!({"type":"web_search_call", "id":"ws_second", "status":"in_progress"});
        let assignment = FileSearchAssignment {
            output_index: 0,
            synthesis: SynthesisKind::Native,
        };
        let state = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: Some(0),
            accumulated_output: vec![file, web.clone()],
            response_object: json!({"output": []}),
            ..ResponsesState::default()
        };

        assert_eq!(
            current_round_file_search_admissions(&state, &[assignment]),
            vec![true],
            "the earlier file call must receive the only built-in slot"
        );
        assert_eq!(
            current_round_tool_call_admissions(&state, &[web]),
            vec![false],
            "the later web call must be rejected after the file call"
        );
    }

    #[test]
    fn incomplete_web_call_remains_charged_in_later_rounds() {
        let next = json!({"type":"web_search_call", "id":"ws_next", "status":"in_progress"});
        let state = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: Some(1),
            accumulated_output: vec![
                json!({"type":"web_search_call", "id":"ws_malformed", "status":"incomplete"}),
                next.clone(),
            ],
            response_object: json!({"output":[next.clone()]}),
            ..ResponsesState::default()
        };

        assert_eq!(consumed_builtin_tool_calls_before_current_round(&state), 1);
        assert_eq!(
            current_round_tool_call_admissions(&state, &[next]),
            vec![false],
            "an admitted call stays charged even when local execution was incomplete"
        );
    }

    #[test]
    fn tool_search_discovery_rejects_exhausted_budget() {
        let exhausted = ResponsesState {
            max_tool_calls: Some(0),
            current_round_output_start: Some(0),
            accumulated_output: vec![json!({"type": "tool_search_call", "id": "tsc_1", "status": "completed"})],
            tool_search_calls: vec![OutputAssignment { output_index: 0 }],
            ..ResponsesState::default()
        };
        assert!(
            !tool_search_discovery_is_within_budget(&exhausted),
            "a zero remaining budget must not admit deferred tools/list"
        );

        let admitted = ResponsesState {
            max_tool_calls: Some(1),
            current_round_output_start: Some(0),
            accumulated_output: vec![json!({"type": "tool_search_call", "id": "tsc_1", "status": "completed"})],
            tool_search_calls: vec![OutputAssignment { output_index: 0 }],
            ..ResponsesState::default()
        };
        assert!(
            tool_search_discovery_is_within_budget(&admitted),
            "the first admitted hosted search may still list deferred connectors"
        );
    }

    #[test]
    fn tool_search_discovery_follows_current_round_admission_order() {
        let search = json!({"type": "tool_search_call", "id": "tsc_1", "status": "completed"});
        let web = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
        let displaced = ResponsesState {
            max_tool_calls: Some(1),
            accumulated_output: vec![web.clone(), search.clone()],
            response_object: json!({"output": [web, search.clone()]}),
            tool_search_calls: vec![OutputAssignment { output_index: 1 }],
            ..ResponsesState::default()
        };
        assert!(
            !tool_search_discovery_is_within_budget(&displaced),
            "an earlier current-round built-in call consumes the shared cap first"
        );
    }

    #[test]
    fn current_provider_execution_does_not_double_charge_round_admission() {
        let web_first = json!({"type":"web_search_call", "id":"ws_current", "status":"completed"});
        let web_second = json!({"type":"web_search_call", "id":"ws_second", "status":"completed"});
        let state = ResponsesState {
            max_tool_calls: Some(2),
            current_round_output_start: Some(0),
            web_search_calls_executed: 1,
            accumulated_output: vec![web_first.clone(), web_second.clone()],
            response_object: json!({"output":[web_first, web_second.clone()]}),
            ..ResponsesState::default()
        };

        // The cumulative execution counter must not shrink the current-round
        // budget: both built-in web calls fit under `max_tool_calls` even
        // though one already ran this round.
        assert_eq!(
            current_round_tool_call_admissions(&state, &[web_second]),
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
            // Echoes in the current response owner must not add two more calls.
            response_object: json!({"output": [first, second]}),
            current_round_output_start: Some(2),
            ..ResponsesState::default()
        };

        assert_eq!(
            consumed_builtin_tool_calls_before_current_round(&state),
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
            max_tool_calls: Some(1),
            current_round_output_start: Some(0),
            accumulated_output: vec![
                web.clone(),
                gated.clone(),
                ungated.clone(),
                json!({"type":"mcp_approval_request", "id":"mcp_gated"}),
            ],
            response_object: json!({"output":[web.clone(), gated, ungated]}),
            ..ResponsesState::default()
        };

        // Admission replays model output, so the locally appended approval item
        // and the exempt MCP siblings never spend the single built-in slot the
        // web search claims.
        assert_eq!(current_round_tool_call_admissions(&state, &[web]), vec![true]);
    }

    #[test]
    fn parallel_tool_calls_defaults_to_true() {
        let body = json!({"model": "gpt-4o", "input": "test"});
        let state = ResponsesState::from_request_body(body);
        assert!(state.parallel_tool_calls);
    }

    #[test]
    fn default_produces_expected_values() {
        let state = ResponsesState::default();
        assert!(state.context_management.is_none());
        assert!(state.conversation.is_none());
        assert_eq!(state.iteration, 0);
        assert!(state.max_tool_calls.is_none());
        assert!(state.client_tool_lowering.is_empty());
        assert!(state.parallel_tool_calls);
        assert!(state.persisted_messages.is_empty());
        assert!(!state.store_persist_armed);
        assert!(state.previous_response_id.is_none());
        assert!(state.previous_usage.is_none());
        assert!(state.request_body.is_null());
        assert!(state.response_object.is_null());
        assert_eq!(state.web_search_calls_executed, 0);
        assert!(
            state.file_search_assignments.is_empty(),
            "file-search assignments must start empty"
        );
        assert_eq!(state.tool_choice, json!("auto"));
        assert!(state.usage.is_null());
    }

    #[test]
    fn default_produces_empty_collections() {
        let state = ResponsesState::default();
        assert!(state.include.is_empty());
        assert!(state.input.is_empty());
        assert!(state.mcp_tool_map.is_empty());
        assert!(state.messages.is_empty());
        assert!(state.output_items().is_empty());
        assert!(state.persisted_messages.is_empty());
        assert!(state.previous_tools.is_empty());
        assert!(state.tool_calls.is_empty());
        assert!(state.tool_search_calls.is_empty());
        assert!(state.deferred_mcp.is_empty());
        assert!(state.web_search_calls.is_empty());
        assert!(state.tools.is_empty());
        assert!(state.accumulated_output.is_empty());
        assert!(state.emitted_output_items.is_empty());
        assert!(state.locally_executed_output_items.is_empty());
        assert!(state.dispatch_failure.is_none(), "dispatch failure must start unset");
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
    fn drain_pending_local_tool_synthesis_moves_and_empties() {
        let mut state = ResponsesState::default();
        state.pending_local_tool_synthesis.push((3, SynthesisKind::Native));
        state.pending_local_tool_synthesis.push((7, SynthesisKind::Private));
        let drained = state.drain_pending_local_tool_synthesis();
        assert_eq!(drained, vec![(3, SynthesisKind::Native), (7, SynthesisKind::Private)]);
        assert!(
            state.pending_local_tool_synthesis.is_empty(),
            "drain must leave the queue empty (drain-once)"
        );
    }

    #[test]
    fn drain_file_search_assignments_moves_and_empties() {
        let mut state = ResponsesState::default();
        state.file_search_assignments.push(FileSearchAssignment {
            output_index: 2,
            synthesis: SynthesisKind::Private,
        });
        state.file_search_assignments.push(FileSearchAssignment {
            output_index: 5,
            synthesis: SynthesisKind::Native,
        });
        let drained = state.drain_file_search_assignments();
        assert_eq!(
            drained,
            vec![
                FileSearchAssignment {
                    output_index: 2,
                    synthesis: SynthesisKind::Private,
                },
                FileSearchAssignment {
                    output_index: 5,
                    synthesis: SynthesisKind::Native,
                },
            ]
        );
        assert!(
            state.file_search_assignments.is_empty(),
            "drain must leave the queue empty (drain-once)"
        );
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
        // MCP calls are exempt from the built-in `max_tool_calls` budget; the
        // OpenAI Responses API keeps MCP tools on their own per-round limit.
        for call_type in ["mcp_call", "mcp_approval_request", "mcp_list_tools"] {
            assert!(
                !is_builtin_tool_call(&json!({"type":call_type})),
                "{call_type} must not consume the built-in tool-call budget"
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

    #[test]
    fn finalize_response_body_moves_accumulated_output_into_response_object() {
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "id": "resp_1", "output": [{"type": "reasoning"}]}),
            accumulated_output: vec![
                json!({"type": "message", "role": "assistant", "content": "hi"}),
                json!({"type": "function_call", "name": "f"}),
            ],
            ..ResponsesState::default()
        };
        let mut body = None;
        state.finalize_response_body(&mut body).expect("finalize succeeds");

        assert!(
            state.accumulated_output.is_empty(),
            "accumulated_output must be moved out, not cloned"
        );
        let output = state.response_object["output"].as_array().unwrap();
        assert_eq!(output.len(), 2, "the full accumulated output replaces the last round");
        assert_eq!(output[0]["content"], "hi");
        let body_json: serde_json::Value = serde_json::from_slice(&body.expect("body written")).unwrap();
        assert_eq!(
            body_json, state.response_object,
            "body must serialize the finalized object"
        );
        assert_eq!(body_json["id"], "resp_1", "prior response metadata is preserved");
    }

    #[test]
    fn finalize_response_body_preserves_output_when_accumulated_empty() {
        // Empty-output guard: an empty accumulator must not overwrite the
        // existing `response_object["output"]`.
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": [{"type": "message", "content": "kept"}]}),
            accumulated_output: Vec::new(),
            ..ResponsesState::default()
        };
        let mut body = None;
        state.finalize_response_body(&mut body).expect("finalize succeeds");

        let output = state.response_object["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "empty accumulated_output must not overwrite output");
        assert_eq!(output[0]["content"], "kept");
        assert!(body.is_some(), "the response is still serialized to the body");
    }

    #[test]
    fn finalize_response_body_stamps_accumulated_usage() {
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![json!({"type": "message"})],
            usage: json!({"input_tokens": 3, "output_tokens": 5}),
            ..ResponsesState::default()
        };
        let mut body = None;
        state.finalize_response_body(&mut body).expect("finalize succeeds");
        assert_eq!(state.response_object["usage"]["output_tokens"], 5);
    }

    #[test]
    fn finalize_response_body_noop_when_response_object_absent() {
        let mut state = ResponsesState {
            response_object: serde_json::Value::Null,
            accumulated_output: vec![json!({"type": "message"})],
            ..ResponsesState::default()
        };
        let mut body = None;
        state.finalize_response_body(&mut body).expect("finalize succeeds");
        assert!(body.is_none(), "a null response_object leaves the body untouched");
        assert_eq!(
            state.accumulated_output.len(),
            1,
            "nothing is moved when there is nothing to finalize"
        );
    }

    #[test]
    fn finalize_response_body_skips_citation_annotation_without_citation_files() {
        // The web/MCP-only path registers no citation files, so unconditional
        // annotation is a strict no-op and any marker-like text survives verbatim.
        let text = "see [ref:file-1]";
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })],
            ..ResponsesState::default()
        };
        assert!(
            state.citation_files.is_empty(),
            "the no-citation regression requires an empty citation map"
        );
        let mut body = None;
        state.finalize_response_body(&mut body).expect("finalize succeeds");

        let part = &state.response_object["output"][0]["content"][0];
        assert_eq!(part["text"], text, "marker text is untouched without citation files");
        assert!(part.get("annotations").is_none(), "no annotations are added");
    }

    #[test]
    fn finalize_response_body_reserves_citation_staging_before_mutation() {
        let text = format!("{} <|file-known|>", "x".repeat(8_192));
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        });
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![item.clone()],
            citation_files: HashMap::from([("file-known".to_owned(), "known.txt".to_owned())]),
            ..ResponsesState::default()
        };
        let current = state.retained_payload_bytes().unwrap();
        let staging = annotation_staging_bytes(&state.accumulated_output, &state.citation_files).unwrap();
        state.apply_retained_payload_limit(current + staging - 1);

        let mut body = None;
        assert!(state.finalize_response_body(&mut body).is_err());
        assert!(state.retained_payload_failed);
        assert_eq!(state.accumulated_output, vec![item], "preflight commits no output move");
        assert!(body.is_none());
    }

    #[test]
    fn finalize_response_body_discards_canonical_tree_when_serialization_reservation_fails() {
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x".repeat(4_096)}],
            })],
            ..ResponsesState::default()
        };
        state.apply_retained_payload_limit(state.retained_payload_bytes().unwrap());

        let mut body = None;
        assert!(state.finalize_response_body(&mut body).is_err());
        assert!(state.retained_payload_failed);
        assert!(
            state.response_object.is_null(),
            "canonical response must not be committed"
        );
        assert!(
            state.accumulated_output.is_empty(),
            "failed request payload is released"
        );
        assert!(body.is_none());
    }

    #[test]
    fn finalize_response_body_fails_closed_when_annotation_exceeds_budget() {
        // More citation markers than the bounded rewriter admits (its private
        // `MAX_CITATION_MARKERS` is 4096) must surface as an HTTP 502 rather than
        // silently emitting an un-annotated body. The registered citation file
        // deliberately does not match the marker id, so the *marker* budget is the
        // binding constraint and each part only consumes one marker.
        const OVER_MARKER_BUDGET: usize = 5_000;
        let content: Vec<serde_json::Value> = (0..OVER_MARKER_BUDGET)
            .map(|_| json!({"type": "output_text", "text": "x <|file-a|>"}))
            .collect();
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": content,
            })],
            citation_files: HashMap::from([("file-known".to_owned(), "known.txt".to_owned())]),
            ..ResponsesState::default()
        };
        let mut body = None;
        let FilterAction::Reject(rejection) = state
            .finalize_response_body(&mut body)
            .expect_err("annotation-budget overflow must fail closed")
        else {
            panic!("finalize must reject, not continue");
        };
        assert_eq!(rejection.status, 502, "annotation failure returns a server error");
        assert!(body.is_none(), "no body is written on a finalize failure");
    }

    #[test]
    fn finalize_response_body_fails_closed_when_output_exceeds_byte_limit() {
        // A finalized response larger than `MAX_JSON_BODY_BYTES` must fail closed
        // with an HTTP 502 instead of serializing an over-limit body. The oversized
        // text carries no citation markers and no citation files are registered, so
        // annotation is a no-op and the byte-limit guard is what rejects.
        let oversized = "x".repeat(MAX_JSON_BODY_BYTES + 1);
        let mut state = ResponsesState {
            response_object: json!({"object": "response", "output": []}),
            accumulated_output: vec![json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": oversized}],
            })],
            ..ResponsesState::default()
        };
        let mut body = None;
        let FilterAction::Reject(rejection) = state
            .finalize_response_body(&mut body)
            .expect_err("over-limit response must fail closed")
        else {
            panic!("finalize must reject, not continue");
        };
        assert_eq!(rejection.status, 502, "size overflow returns a server error");
        assert!(body.is_none(), "no body is written on a finalize failure");
    }
}
