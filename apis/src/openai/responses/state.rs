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
    borrow::Borrow,
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

/// Internal persisted field identifying a Praxis-generated compaction item.
///
/// This field is stored only in the private rehydration history. The outbound
/// serializer removes it before any item reaches a provider or client.
pub(crate) const LOCAL_COMPACTION_MARKER: &str = "_praxis_local_compaction";

/// Mark a locally generated compaction item in private persisted history.
#[cfg(feature = "openai-compact")]
pub(crate) fn mark_local_compaction_item(item: &serde_json::Value) -> serde_json::Value {
    let mut marked = item.clone();
    if let Some(object) = marked.as_object_mut() {
        object.insert(LOCAL_COMPACTION_MARKER.to_owned(), serde_json::Value::Bool(true));
    }
    marked
}

/// Remove private provenance metadata before replaying a stored item.
#[cfg(feature = "store")]
pub(crate) fn strip_local_compaction_marker(mut item: serde_json::Value) -> serde_json::Value {
    if let Some(object) = item.as_object_mut() {
        object.remove(LOCAL_COMPACTION_MARKER);
    }
    item
}

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
/// absolute index and item ID here plus the synthesis origin
/// (private-normalized vs native).
/// The dispatcher drains these at request-body EOS and mutates the indexed item
/// in place — setting `completed`/`incomplete`, adding public results, and
/// bridging model context — so the complete output item is never cloned into a
/// second owner (avoids the payload duplication the "keep `Vec<Value>`" interface
/// would incur; the recommended "store output indices" boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileSearchAssignment {
    /// Absolute index into [`ResponsesState::accumulated_output`] of the
    /// `file_search_call` item the dispatcher must execute and reconcile.
    pub output_index: usize,
    /// Public item ID, assigned before the parse owner records selections.
    pub item_id: String,
    /// Whether the item was normalized from a private `function_call` this round
    /// (`Private`, opening suppressed) or streamed natively (`Native`).
    pub synthesis: SynthesisKind,
}

impl FileSearchAssignment {
    /// Resolve only the original current-round file-search placeholder.
    pub(crate) fn resolve<'a>(&self, state: &'a ResponsesState) -> Option<&'a serde_json::Value> {
        let round_start = state.current_round_output_start.unwrap_or(0);
        if self.output_index < round_start {
            return None;
        }
        let item = state.accumulated_output.get(self.output_index)?;
        (item.get("type").and_then(serde_json::Value::as_str) == Some("file_search_call")
            && item.get("id").and_then(serde_json::Value::as_str) == Some(self.item_id.as_str()))
        .then_some(item)
    }
}

/// A round-local dispatch selection into the canonical public output.
///
/// The item ID prevents an old selection from dispatching a different item if
/// output is replaced after an error. Callers also supply the expected type,
/// so a malformed or stale index fails closed rather than changing ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputAssignment {
    /// Absolute index in the canonical output array.
    pub output_index: usize,
    /// ID of the selected item, used to reject a replaced slot.
    pub item_id: String,
}

impl ResponsesState {
    /// Resolve current-round selections of one output type without copying items.
    fn selected_output<'a>(
        &'a self,
        assignments: &'a [OutputAssignment],
        expected_type: &'static str,
    ) -> impl Iterator<Item = &'a serde_json::Value> {
        let round_start = self.current_round_output_start.unwrap_or(0);
        assignments.iter().filter_map(move |assignment| {
            (assignment.output_index >= round_start)
                .then(|| assignment.resolve(&self.accumulated_output, expected_type))
                .flatten()
        })
    }

    /// Borrow selected function calls, including approved calls on resume.
    #[cfg(any(test, feature = "openai-mcp-tools"))]
    pub(crate) fn selected_tool_calls(&self) -> Vec<&serde_json::Value> {
        let calls: Vec<_> = self.selected_output(&self.tool_calls, "function_call").collect();
        #[cfg(feature = "openai-mcp-tools")]
        let calls = {
            let mut calls = calls;
            calls.extend(self.approved_tool_calls.iter());
            calls
        };
        calls
    }

    /// Borrow current-round hosted web-search calls.
    pub(crate) fn selected_web_search_calls(&self) -> Vec<&serde_json::Value> {
        self.selected_output(&self.web_search_calls, "web_search_call")
            .collect()
    }

    /// Preserve the absolute output index for each valid web-search selection.
    /// The dispatcher uses it to replace only the selected placeholder, even
    /// when an earlier selection in the same round has become stale.
    pub(crate) fn selected_web_search_outputs(&self) -> Vec<(usize, &serde_json::Value)> {
        let round_start = self.current_round_output_start.unwrap_or(0);
        self.web_search_calls
            .iter()
            .filter_map(|assignment| {
                (assignment.output_index >= round_start)
                    .then(|| {
                        assignment
                            .resolve(&self.accumulated_output, "web_search_call")
                            .map(|item| (assignment.output_index, item))
                    })
                    .flatten()
            })
            .collect()
    }

    /// Borrow current-round hosted tool-search calls.
    pub(crate) fn selected_tool_search_calls(&self) -> Vec<&serde_json::Value> {
        self.selected_output(&self.tool_search_calls, "tool_search_call")
            .collect()
    }
}

impl OutputAssignment {
    /// Select an item by absolute index and its current public ID.
    pub(crate) fn new(output_index: usize, item: &serde_json::Value) -> Option<Self> {
        Some(Self {
            output_index,
            item_id: item.get("id")?.as_str()?.to_owned(),
        })
    }

    /// Resolve only when the indexed item still has the expected ID and type.
    pub(crate) fn resolve<'a>(
        &self,
        output: &'a [serde_json::Value],
        expected_type: &str,
    ) -> Option<&'a serde_json::Value> {
        let item = output.get(self.output_index)?;
        (item.get("id")?.as_str()? == self.item_id && item.get("type")?.as_str()? == expected_type).then_some(item)
    }
}

#[cfg(test)]
impl ResponsesState {
    /// Install model items in their production owner for dispatcher unit tests.
    #[expect(
        clippy::too_many_lines,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        reason = "test fixture builder rejects malformed test items and unknown kinds"
    )]
    pub(crate) fn select_test_output(&mut self, kind: &str, items: Vec<serde_json::Value>) {
        for mut item in items {
            let first_selection =
                self.tool_calls.is_empty() && self.web_search_calls.is_empty() && self.tool_search_calls.is_empty();
            let existing_index = self.accumulated_output.iter().position(|existing| existing == &item);
            if item.get("type").and_then(serde_json::Value::as_str).is_none() {
                item.as_object_mut()
                    .expect("test output item")
                    .insert("type".to_owned(), serde_json::Value::String(kind.to_owned()));
                if let Some(index) = existing_index {
                    self.accumulated_output[index]
                        .as_object_mut()
                        .expect("test output item")
                        .insert("type".to_owned(), serde_json::Value::String(kind.to_owned()));
                }
            }
            if item.get("id").and_then(serde_json::Value::as_str).is_none() {
                let id = format!(
                    "test_output_{}",
                    existing_index.unwrap_or(self.accumulated_output.len())
                );
                item.as_object_mut()
                    .expect("test output item")
                    .insert("id".to_owned(), serde_json::Value::String(id.clone()));
                if let Some(index) = existing_index {
                    self.accumulated_output[index]
                        .as_object_mut()
                        .expect("test output item")
                        .insert("id".to_owned(), serde_json::Value::String(id));
                }
            }
            let output_index = existing_index
                .or_else(|| self.accumulated_output.iter().position(|existing| existing == &item))
                .unwrap_or_else(|| {
                    let index = self.accumulated_output.len();
                    self.accumulated_output.push(item.clone());
                    index
                });
            self.current_round_output_start = Some(if first_selection {
                output_index
            } else {
                self.current_round_output_start
                    .map_or(output_index, |start| start.min(output_index))
            });
            let assignment = OutputAssignment::new(output_index, &item).expect("test output id");
            match kind {
                "function_call" => self.tool_calls.push(assignment),
                "web_search_call" => self.web_search_calls.push(assignment),
                "tool_search_call" => self.tool_search_calls.push(assignment),
                _ => panic!("unknown test output selection {kind}"),
            }
        }
    }
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
    #[cfg_attr(
        all(not(test), not(feature = "openai-mcp-tools")),
        expect(dead_code, reason = "only MCP dispatch defers approval responses")
    )]
    ApprovalPendingThenReturn,
    /// Execute ungated siblings, then return the pending approval response.
    #[cfg_attr(
        not(feature = "openai-mcp-tools"),
        expect(dead_code, reason = "only MCP dispatch defers approval responses")
    )]
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
              store_persist_armed, previous_response_id_stream_restore_armed, \
              logical_stream_terminal_emitted) are independent request facts, \
              not a state machine or refactorable enum"
)]
pub(crate) struct ResponsesState {
    /// Request-wide aggregate retained-payload ceiling selected by
    /// `openai_agentic_loop`. `None` until the first loop instance arms it.
    /// Additional loop instances may only lower the limit.
    pub(crate) retained_payload_limit: Option<usize>,

    /// Payload retained by request-lifetime sibling filter state that is not
    /// represented directly in this bag (the response-store input snapshot,
    /// captured SSE replay rows, and the store decoder's unfinished record).
    /// The loop owner refreshes this before initial admission.
    pub(crate) retained_external_payload_bytes: usize,
    /// Parser payload published by `openai_stream_events` for other filters.
    pub(crate) retained_stream_parser_bytes: usize,
    /// Incomplete SSE frame held by the outer rehydration response rewrite.
    pub(crate) retained_rehydrate_stream_bytes: usize,
    /// Semantic and framing payload published by the Chat stream translator.
    pub(crate) retained_chat_converter_bytes: usize,
    /// Parked MCP session payload last published by the dispatcher.
    #[cfg(feature = "openai-mcp-tools")]
    pub(crate) retained_mcp_session_bytes: usize,
    /// Closing sessions release asynchronously, so read their live charge at
    /// each admission instead of keeping a stale dispatcher snapshot.
    #[cfg(feature = "openai-mcp-tools")]
    pub(crate) retained_mcp_closing_pool: Option<crate::mcp_client::McpSessionPool>,

    /// Revision of request, history, prior output, and resolved MCP definitions
    /// cached by streaming admission. In-place changes do not alter lengths.
    /// `None` means the counter overflowed and replay must fail closed.
    pub(crate) replay_stable_payload_revision: Option<u64>,
    /// Revision of the current response, local completion template, and
    /// completed tool calls. Kept separate so current-round item events do not
    /// invalidate the stream parser's once-measured prior output.
    pub(crate) current_output_revision: Option<u64>,

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

    /// Whether the client-visible terminal `response.completed` event has been
    /// emitted as a *deferred, non-end-of-stream* chunk for this logical stream.
    ///
    /// #937: `openai_stream_events` defers the terminal frame until the inner
    /// IRR stream ends, then surfaces it to the pre-IRR `openai_response_store`
    /// as an ordinary non-end-of-stream chunk — ahead of the empty
    /// end-of-stream callback where streaming persistence historically ran. Left
    /// uncoordinated, a client could observe `response.completed` for a record a
    /// later GET, DELETE, or `previous_response_id` continuation cannot find.
    ///
    /// [`emit_deferred_terminal`] sets this the moment it canonicalizes
    /// [`Self::response_object`] and appends that non-end-of-stream terminal
    /// frame, so the store persists synchronously BEFORE releasing the chunk
    /// (failing closed on error) and then skips the redundant end-of-stream
    /// persist. A request-phase local completion uses the separate
    /// [`Self::local_stream_terminal_emitted`] flag because IRR can deliver its
    /// terminal body as a non-end-of-stream chunk too.
    ///
    /// [`emit_deferred_terminal`]: crate::openai::responses::stream_events
    pub logical_stream_terminal_emitted: bool,

    /// A request-phase dispatcher encoded a local `response.completed` frame.
    /// IRR can deliver this as a non-EOS chunk, so outer persistence filters
    /// must commit the canonical response before releasing that chunk.
    pub local_stream_terminal_emitted: bool,

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

    /// Slot policy under which configured MCP connector state was resolved.
    ///
    /// `openai_mcp_tool_resolve` records this while building MCP state.
    /// `openai_mcp_dispatch` compares it with its own configuration before
    /// issuing a configured-connector callout, preventing discovery and
    /// execution from silently using different request-scoped credential or
    /// authorization slots. Direct `server_url` entries ignore this field.
    #[cfg(feature = "openai-mcp-tools")]
    pub mcp_connector_context_policy: McpConnectorContextPolicy,

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

    /// IDs of compaction items that came from a provider response.
    ///
    /// Locally generated Praxis summaries use the same public JSON shape as a
    /// provider compaction item, so the outbound proxy must not infer native
    /// support from the item type alone. Rehydration populates this set from
    /// persisted provenance metadata; items absent from the set are translated.
    pub provider_compaction_ids: HashSet<String>,

    /// Whether tool calls may execute concurrently within an
    /// iteration. Defaults to `true` per the API spec.
    #[cfg_attr(
        all(not(test), not(feature = "openai-mcp-tools")),
        expect(dead_code, reason = "only MCP dispatch schedules concurrent tool calls")
    )]
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
    #[cfg(feature = "store")]
    pub pending_approvals: Vec<praxis_ai_store::PendingApprovalRecord>,

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

    /// This round's buffered body was serialized and admitted by the agentic
    /// loop. The outer response hook cannot read IRR's inner filter results.
    pub(crate) buffered_canonical_finalized: bool,

    /// Prior streamed terminal response retained across request-side re-entry.
    ///
    /// `openai_stream_events` invalidates [`Self::response_object`] before the
    /// next inference round so an upstream error cannot persist stale success.
    /// A dispatcher can instead complete locally before that inference starts
    /// (for example, an MCP approval or exhausted tool budget), so the prior
    /// response metadata is moved here until the new upstream response begins.
    /// Its output has already been drained into [`Self::accumulated_output`].
    pub local_completion_response_template: serde_json::Value,

    /// Current-round function-call selections into [`Self::accumulated_output`].
    /// The loop records these after buffered parsing or a completed streamed
    /// round. Clearing them on re-entry prevents duplicate dispatch.
    pub tool_calls: Vec<OutputAssignment>,

    /// Store-derived approved calls have no item in the current provider
    /// output. Their independent owner is required until MCP execution.
    #[cfg(feature = "openai-mcp-tools")]
    pub approved_tool_calls: Vec<serde_json::Value>,

    /// Current-round hosted `tool_search_call` selections into public output.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration.
    /// `openai_mcp_dispatch` consumes these to load deferred connector
    /// tools before the next inference round.
    pub tool_search_calls: Vec<OutputAssignment>,

    /// Current-round web-search selections into public output.
    ///
    /// Cleared by `openai_agentic_loop` at the start of each iteration.
    /// Stored separately from `tool_calls` because `web_search_call`
    /// items have a different shape (`action.query` instead of
    /// `name`/`arguments`) and are dispatched by a different filter.
    pub web_search_calls: Vec<OutputAssignment>,

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
    /// terminal snapshot — persists across rounds: resetting the counter each
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

    /// A locally-detected security-context failure (e.g. a missing/invalid required per-user
    /// callout credential). Write-once via [`ResponsesState::record_security_failure`]; the
    /// agentic loop converts it into a terminal 401 BEFORE any generic [`Self::dispatch_failure`],
    /// so a security terminal always preempts a generic dispatch terminal.
    pub security_failure: Option<DispatchFailure>,
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
///
/// The deferred `tools/list` callout is issued later by `openai_mcp_dispatch`,
/// so its private/loopback posture is governed by that filter's bound outbound
/// pipeline (via the per-request `McpCallout`) rather than a field captured here.
#[derive(Clone)]
pub(crate) struct DeferredMcpConnector {
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

/// Request-scoped slot policy bound to configured MCP connector state.
///
/// This contains slot identifiers only, never credential or assertion values.
#[cfg(feature = "openai-mcp-tools")]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct McpConnectorContextPolicy {
    /// Per-user bearer slot configured on the resolver.
    credential_slot: Option<String>,

    /// Opaque authorization-assertion slot configured on the resolver.
    authorization_slot: Option<String>,
}

#[cfg(feature = "openai-mcp-tools")]
impl McpConnectorContextPolicy {
    /// Snapshot a filter's configured connector-context slots.
    pub(crate) fn new(credential_slot: Option<&str>, authorization_slot: Option<&str>) -> Self {
        Self {
            credential_slot: credential_slot.map(str::to_owned),
            authorization_slot: authorization_slot.map(str::to_owned),
        }
    }

    /// Bytes owned by the configured slot names in request-scoped state.
    pub(crate) fn retained_payload_bytes(&self) -> Option<usize> {
        self.credential_slot
            .as_ref()
            .map_or(0, String::len)
            .checked_add(self.authorization_slot.as_ref().map_or(0, String::len))
    }
}

impl fmt::Debug for DeferredMcpConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeferredMcpConnector")
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
            retained_stream_parser_bytes: 0,
            retained_rehydrate_stream_bytes: 0,
            retained_chat_converter_bytes: 0,
            #[cfg(feature = "openai-mcp-tools")]
            retained_mcp_session_bytes: 0,
            #[cfg(feature = "openai-mcp-tools")]
            retained_mcp_closing_pool: None,
            replay_stable_payload_revision: Some(0),
            current_output_revision: Some(0),
            retained_payload_failed: false,
            citation_files: HashMap::new(),
            context_management: None,
            conversation: None,
            include: Vec::new(),
            logical_stream_response_id: None,
            logical_stream_sequence: 0,
            logical_stream_terminal_emitted: false,
            local_stream_terminal_emitted: false,
            current_round_output_start: None,
            history_rehydrated: false,
            input: Vec::new(),
            iteration: 0,
            deferred_mcp: Vec::new(),
            #[cfg(feature = "openai-mcp-tools")]
            mcp_connector_context_policy: McpConnectorContextPolicy::default(),
            max_tool_calls: None,
            mcp_approval_state: McpApprovalState::None,
            deferred_tool_limit_completion: false,
            deferred_stream_done: false,
            mcp_tool_map: HashMap::new(),
            client_tool_lowering: HashMap::new(),
            client_tool_echo: None,
            messages: Vec::new(),
            provider_history_len: 0,
            provider_compaction_ids: HashSet::new(),
            parallel_tool_calls: true,
            persisted_messages: Vec::new(),
            #[cfg(feature = "store")]
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
            buffered_canonical_finalized: false,
            local_completion_response_template: serde_json::Value::Null,
            tool_calls: Vec::new(),
            #[cfg(feature = "openai-mcp-tools")]
            approved_tool_calls: Vec::new(),
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
            security_failure: None,
        }
    }
}

#[expect(
    clippy::multiple_inherent_impl,
    reason = "selection helpers are colocated with assignment types"
)]
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

    /// Update one sibling owner's charge without dropping charges held by
    /// other filters during the same request.
    #[cfg(feature = "store")]
    pub(crate) fn replace_retained_external_payload_bytes(&mut self, previous: usize, next: usize) -> bool {
        let Some(total) = self
            .retained_external_payload_bytes
            .checked_sub(previous)
            .and_then(|remaining| remaining.checked_add(next))
        else {
            self.retained_external_payload_bytes = usize::MAX;
            self.fail_retained_payload_budget();
            return false;
        };
        self.retained_external_payload_bytes = total;
        true
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
        if let Some(remaining) = self.retained_external_payload_bytes.checked_sub(bytes) {
            self.retained_external_payload_bytes = remaining;
        } else {
            self.retained_external_payload_bytes = usize::MAX;
            self.fail_retained_payload_budget();
        }
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
        self.retained_payload_bytes_bounded_inner(max_bytes, true, false, false, true, true, None)
    }

    /// Request, history, resolved MCP definitions, and the capture-once
    /// client-tool echo remain stable between upstream chunks. Measure them
    /// once per round; streaming filters refresh this baseline when discovery,
    /// binding, a request rewrite, or the next round changes it.
    #[expect(clippy::too_many_lines, reason = "all stable request owners share one revision")]
    pub(crate) fn stream_stable_payload_bytes_bounded(&self, max_bytes: usize) -> Option<usize> {
        let mut meter = PayloadMeter::new(max_bytes);
        meter.json(&self.request_body)?;
        for values in [
            &self.input,
            &self.messages,
            &self.persisted_messages,
            &self.previous_tools,
            &self.tools,
        ] {
            meter.json_values(values)?;
        }
        for id in &self.provider_compaction_ids {
            meter.raw(id.len())?;
        }
        if let Some(echo) = &self.client_tool_echo {
            meter.json_values(&echo.tools)?;
            meter.json(&echo.tool_choice)?;
        }
        meter.json(&self.tool_choice)?;
        if let Some(original_tool_choice) = &self.original_tool_choice {
            meter.json(original_tool_choice)?;
        }
        for ((server, tool), value) in &self.mcp_tool_map {
            meter.raw(server.len())?;
            meter.raw(tool.len())?;
            meter.json(value)?;
        }
        self.meter_deferred_mcp_payload(&mut meter)?;
        for value in [
            self.context_management.as_ref(),
            self.conversation.as_ref(),
            self.previous_usage.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            meter.json(value)?;
        }
        Some(meter.used())
    }

    /// Charge deferred connector definitions in the revision-invalidated stable owner.
    fn meter_deferred_mcp_payload(&self, meter: &mut PayloadMeter) -> Option<()> {
        for connector in &self.deferred_mcp {
            meter.raw(connector.connector_id.len())?;
            meter.raw(connector.server_label.len())?;
            meter.raw(connector.server_url.len())?;
            if let Some(authorization) = &connector.authorization {
                meter.raw(authorization.len())?;
            }
            for value in [
                connector.allowed_tools.as_ref(),
                connector.headers.as_ref(),
                connector.require_approval.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                meter.json(value)?;
            }
        }
        #[cfg(feature = "openai-mcp-tools")]
        for slot in [
            self.mcp_connector_context_policy.credential_slot.as_ref(),
            self.mcp_connector_context_policy.authorization_slot.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            meter.raw(slot.len())?;
        }
        Some(())
    }

    /// The Chat stream converter caches the current response/template/call
    /// charge by revision, but this meter still rechecks parser and sibling
    /// filter owners on every callback.
    pub(crate) fn chat_stream_changing_payload_bytes_bounded_with_current_output(
        &self,
        max_bytes: usize,
        current_output_bytes: usize,
    ) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, true, true, true, true, true, Some(current_output_bytes))
    }

    /// Stream events separately charges its parser state and cached prior
    /// output; the capture-once echo belongs to the common stable charge.
    #[cfg(test)]
    pub(crate) fn stream_changing_payload_bytes_bounded_for_parser(&self, max_bytes: usize) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, true, true, true, false, true, None)
    }

    /// Stream events separately caches the exact current response and fallback
    /// terminal template while those owners stay unchanged across SSE chunks.
    /// Canonical tool-call assignments are charged by their item IDs below.
    pub(crate) fn stream_changing_payload_bytes_bounded_with_current_output(
        &self,
        max_bytes: usize,
        current_output_bytes: usize,
    ) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, true, true, true, false, true, Some(current_output_bytes))
    }

    /// Store and rehydrate cache completed output while the stream parser's
    /// independently retained bytes must still be included on every chunk.
    #[cfg(feature = "store")]
    pub(crate) fn store_stream_changing_payload_bytes_bounded_with_current_output(
        &self,
        max_bytes: usize,
        current_output_bytes: usize,
    ) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, true, true, true, true, true, Some(current_output_bytes))
    }

    /// Count payload owned directly by this state, excluding sibling-filter
    /// owners that participate only in the aggregate agentic budget.
    ///
    /// File search's legacy `max_state_bytes` option predates the aggregate
    /// budget and covers iterative-router plus file-search continuation state;
    /// response-store snapshots and other sibling-filter owners must not change
    /// that independent compatibility limit.
    pub(crate) fn retained_payload_bytes_bounded_without_external(&self, max_bytes: usize) -> Option<usize> {
        self.retained_payload_bytes_bounded_inner(max_bytes, false, false, false, false, false, None)
    }

    /// Shared implementation for aggregate and state-only payload accounting.
    #[expect(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        clippy::fn_params_excessive_bools,
        reason = "exhaustive accounting for the request-scoped state bag"
    )]
    fn retained_payload_bytes_bounded_inner(
        &self,
        max_bytes: usize,
        include_external: bool,
        skip_stream_stable: bool,
        skip_accumulated_output: bool,
        include_stream_parser: bool,
        include_chat_converter: bool,
        cached_current_output_bytes: Option<usize>,
    ) -> Option<usize> {
        let mut meter = PayloadMeter::new(max_bytes);
        if include_external {
            meter.raw(self.retained_external_payload_bytes)?;
            meter.raw(self.retained_rehydrate_stream_bytes)?;
        }
        if include_stream_parser {
            meter.raw(self.retained_stream_parser_bytes)?;
        }
        if include_chat_converter {
            meter.raw(self.retained_chat_converter_bytes)?;
        }
        #[cfg(feature = "openai-mcp-tools")]
        if include_external {
            meter.raw(self.retained_mcp_session_bytes)?;
            if let Some(pool) = &self.retained_mcp_closing_pool {
                meter.raw(pool.retained_closing_payload_bytes()?)?;
            }
        }

        if !skip_stream_stable {
            meter.json(&self.request_body)?;
            for values in [
                &self.input,
                &self.messages,
                &self.persisted_messages,
                &self.previous_tools,
                &self.tools,
            ] {
                meter.json_values(values)?;
            }
            for id in &self.provider_compaction_ids {
                meter.raw(id.len())?;
            }
            if let Some(echo) = &self.client_tool_echo {
                meter.json_values(&echo.tools)?;
                meter.json(&echo.tool_choice)?;
            }
            meter.json(&self.tool_choice)?;
            if let Some(original_tool_choice) = &self.original_tool_choice {
                meter.json(original_tool_choice)?;
            }
            for ((server, tool), value) in &self.mcp_tool_map {
                meter.raw(server.len())?;
                meter.raw(tool.len())?;
                meter.json(value)?;
            }
            self.meter_deferred_mcp_payload(&mut meter)?;
            for value in [
                self.context_management.as_ref(),
                self.conversation.as_ref(),
                self.previous_usage.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                meter.json(value)?;
            }
        }
        if let Some(bytes) = cached_current_output_bytes {
            meter.raw(bytes)?;
        } else {
            meter.json(&self.response_object)?;
            meter.json(&self.local_completion_response_template)?;
        }
        meter.json(&self.usage)?;
        if !skip_accumulated_output {
            meter.json_values(&self.accumulated_output)?;
        }
        #[cfg(feature = "openai-mcp-tools")]
        meter.json_values(&self.approved_tool_calls)?;
        for assignment in self
            .tool_calls
            .iter()
            .chain(&self.tool_search_calls)
            .chain(&self.web_search_calls)
        {
            meter.raw(assignment.item_id.len())?;
        }
        for assignment in &self.file_search_assignments {
            meter.raw(assignment.item_id.len())?;
        }
        for (private_name, lowered) in &self.client_tool_lowering {
            meter.raw(private_name.len())?;
            meter.raw(lowered.original_name.len())?;
            if let Some(namespace) = &lowered.namespace {
                meter.raw(namespace.len())?;
            }
        }
        for (key, value) in &self.citation_files {
            meter.raw(key.len())?;
            meter.raw(value.len())?;
        }
        for value in &self.include {
            meter.raw(value.len())?;
        }
        for value in [
            self.logical_stream_response_id.as_ref(),
            self.previous_response_id.as_ref(),
            self.response_id.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            meter.raw(value.len())?;
        }
        for value in &self.provider_streamed_terminal_ids {
            meter.raw(value.len())?;
        }
        for value in &self.locally_executed_output_items {
            meter.raw(value.len())?;
        }
        for (id, emitted) in &self.emitted_output_items {
            meter.raw(id.len())?;
            for phase in &emitted.streamed_phases {
                meter.raw(phase.len())?;
            }
        }
        #[cfg(feature = "store")]
        for record in &self.pending_approvals {
            meter.raw(record.approval_id.len())?;
            meter.raw(record.server_label.len())?;
            meter.raw(record.tool_name.len())?;
            meter.raw(record.arguments.len())?;
            meter.raw(record.target_fingerprint.len())?;
        }
        if let Some(failure) = &self.dispatch_failure {
            meter.raw(failure.message.len())?;
        }
        if let Some(failure) = &self.security_failure {
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
        let Some(measurement_limit) = limit.checked_add(removed_bytes) else {
            return false;
        };
        let Some(current) = self.retained_payload_bytes_bounded(measurement_limit) else {
            return false;
        };
        current
            .checked_sub(removed_bytes)
            .and_then(|bytes| bytes.checked_add(added_bytes))
            .and_then(|bytes| bytes.checked_add(external_bytes))
            .is_some_and(|bytes| bytes <= limit)
    }

    /// Drop all dispatch selections and disable successful persistence after an
    /// aggregate budget failure.
    pub(crate) fn fail_retained_payload_budget(&mut self) {
        self.retained_payload_failed = true;
        self.tool_calls.clear();
        #[cfg(feature = "openai-mcp-tools")]
        self.approved_tool_calls.clear();
        self.mark_current_output_changed();
        self.tool_search_calls.clear();
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
    #[expect(clippy::too_many_lines, reason = "explicitly drops each independently owned payload")]
    pub(crate) fn discard_payload_for_budget_error(&mut self) {
        let streaming = self.request_body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
        self.fail_retained_payload_budget();
        self.citation_files.clear();
        self.context_management = None;
        self.conversation = None;
        self.include.clear();
        self.logical_stream_response_id = None;
        self.input.clear();
        self.deferred_mcp.clear();
        self.mcp_tool_map.clear();
        self.client_tool_lowering.clear();
        self.client_tool_echo = None;
        #[cfg(feature = "openai-mcp-tools")]
        {
            self.retained_mcp_session_bytes = 0;
            self.retained_mcp_closing_pool = None;
            self.mcp_connector_context_policy = McpConnectorContextPolicy::default();
        }
        self.messages.clear();
        self.persisted_messages.clear();
        #[cfg(feature = "store")]
        self.pending_approvals.clear();
        self.previous_response_id = None;
        self.previous_tools.clear();
        self.previous_usage = None;
        self.original_tool_choice = None;
        self.response_id = None;
        self.request_body = serde_json::json!({ "stream": streaming });
        self.response_object = serde_json::Value::Null;
        self.local_completion_response_template = serde_json::Value::Null;
        self.mark_current_output_changed();
        self.tool_choice = serde_json::Value::Null;
        self.tools.clear();
        self.usage = serde_json::Value::Null;
        self.accumulated_output.clear();
        self.emitted_output_items.clear();
        self.locally_executed_output_items.clear();
        self.provider_streamed_terminal_ids.clear();
        self.provider_compaction_ids.clear();
        self.retained_rehydrate_stream_bytes = 0;
        self.dispatch_failure = None;
        self.mark_replay_stable_payload_changed();
    }

    /// Create initial state from a parsed request body.
    pub(crate) fn from_request_body(body: serde_json::Value) -> Self {
        let messages = normalize_input(&body);
        let persisted_messages = messages.clone();
        let provider_compaction_ids = Self::provider_compaction_ids_from_messages(&messages);
        let tool_choice = body
            .get("tool_choice")
            .filter(|v| !v.is_null())
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
            provider_compaction_ids,
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

    /// Identify opaque provider compaction items already present in a stateless
    /// input array. Locally generated summaries use the `compact_` ID prefix or
    /// carry the private provenance marker and must still be translated.
    pub(crate) fn provider_compaction_ids_from_messages(messages: &[serde_json::Value]) -> HashSet<String> {
        messages
            .iter()
            .filter_map(Self::provider_compaction_id_from_message)
            .map(ToOwned::to_owned)
            .collect()
    }

    /// Borrow a provider compaction ID before a collector copies it into state.
    pub(crate) fn provider_compaction_id_from_message(item: &serde_json::Value) -> Option<&str> {
        (item.get("type").and_then(serde_json::Value::as_str) == Some("compaction"))
            .then_some(item)
            .filter(|item| item.get(LOCAL_COMPACTION_MARKER).and_then(serde_json::Value::as_bool) != Some(true))
            .and_then(|item| item.get("id").and_then(serde_json::Value::as_str))
            .filter(|id| !id.starts_with("compact_"))
    }

    /// Record the first security-context failure; later calls are ignored (first wins).
    pub(crate) fn record_security_failure(&mut self, failure: DispatchFailure) {
        if self.security_failure.is_none() {
            self.security_failure = Some(failure);
        }
    }

    /// Require the proxy to serialize provider-visible request state.
    pub(crate) fn mark_request_body_for_rebuild(&mut self) {
        self.request_body_rebuild = RequestBodyRebuild::Required;
        self.mark_replay_stable_payload_changed();
    }

    /// Invalidate streaming meters after an in-place mutation of a cached
    /// request, history, tool definition, or accumulated output owner.
    pub(crate) fn mark_replay_stable_payload_changed(&mut self) {
        self.replay_stable_payload_revision = self
            .replay_stable_payload_revision
            .and_then(|revision| revision.checked_add(1));
    }

    /// Invalidate store and rehydrate current-output measurements without
    /// invalidating the stream parser's completed prior-output cache.
    pub(crate) fn mark_current_output_changed(&mut self) {
        self.buffered_canonical_finalized = false;
        self.current_output_revision = self
            .current_output_revision
            .and_then(|revision| revision.checked_add(1));
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

    /// Borrow the **outbound** request tools that downstream translation emits.
    ///
    /// [`Self::request_body`] is the single lowered-tool view:
    /// `openai_client_tool_compat` lowers rich client tools into
    /// `request_body["tools"]` **only**, leaving canonical [`Self::tools`] holding
    /// the original rich types for response-side restore. Provider-body consumers
    /// (`openai_responses_proxy`, `responses_to_chat_completions`) must read the
    /// outbound tools through this accessor so they translate the lowered view, not
    /// the canonical rich tools that a function-only backend cannot accept.
    ///
    /// Returns `request_body["tools"]` when present as an array, else falls back to
    /// canonical [`Self::tools`]. A present-but-non-array override is not trusted.
    /// Compat must keep **not** writing canonical [`Self::tools`]; this accessor is
    /// the contract that keeps the lowered view single-sourced (no duplicate field,
    /// no clone).
    pub(crate) fn request_tools(&self) -> &[serde_json::Value] {
        self.request_body
            .get("tools")
            .and_then(serde_json::Value::as_array)
            .map_or(self.tools.as_slice(), Vec::as_slice)
    }

    /// Borrow the **outbound** request `tool_choice` downstream translation emits.
    ///
    /// Mirrors [`Self::request_tools`]: returns `request_body["tool_choice"]` when
    /// the key is present, else canonical [`Self::tool_choice`]. The check is on key
    /// presence, not value shape, so it preserves `auto` omission — it never
    /// synthesizes a choice the caller or `openai_agentic_loop` did not set.
    pub(crate) fn request_tool_choice(&self) -> &serde_json::Value {
        self.request_body.get("tool_choice").unwrap_or(&self.tool_choice)
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
        self.mark_current_output_changed();
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
            Some(0)
        } else {
            retained_json_bytes(&self.usage)
        };
        let preflight_staging = usage_staging.and_then(|usage| annotation_staging.checked_add(usage));
        if !preflight_staging.is_some_and(|staging| self.can_retain_payload(staging)) {
            self.discard_payload_for_budget_error();
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
                obj.insert("usage".to_owned(), self.usage.clone());
            }
        }
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
        let Some(null_bytes) = retained_json_bytes(&self.response_object) else {
            self.discard_payload_for_budget_error();
            return Err(finalize_rejection("failed to size final response state"));
        };
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
        // The size pass measured this exact serialization. Allocate its one
        // admitted wire owner up front so Vec growth cannot keep an old buffer
        // alive alongside its replacement near the request-wide limit.
        let mut serialized = Vec::with_capacity(serialized_bytes);
        if let Err(error) = serde_json::to_writer(&mut serialized, &response) {
            tracing::warn!(%error, "failed to encode final response");
            self.response_object = response;
            return Err(finalize_rejection("failed to encode final response"));
        }
        self.response_object = response;
        *body = Some(Bytes::from(serialized));
        self.buffered_canonical_finalized = true;
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
        let remaining = self.limit.checked_sub(self.used)?;
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
pub(crate) fn current_round_tool_call_admissions<T: Borrow<serde_json::Value>>(
    state: &ResponsesState,
    target_calls: &[T],
) -> Vec<bool> {
    current_round_tool_call_admissions_by(state, target_calls.len(), |index| {
        target_calls.get(index).map(Borrow::borrow)
    })
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
/// `max_tool_calls` is exhausted. A stale or invalid selection cannot authorize
/// an outbound listing, even when the limit is omitted.
#[cfg_attr(
    all(not(feature = "openai-mcp-tools"), not(test)),
    expect(dead_code, reason = "MCP resolver is the production caller")
)]
pub(crate) fn tool_search_discovery_is_within_budget(state: &ResponsesState) -> bool {
    let selected = state.selected_tool_search_calls();
    if selected.is_empty() {
        return false;
    }
    let Some(max) = state.max_tool_calls else {
        return true;
    };
    let remaining = usize::try_from(max)
        .unwrap_or(usize::MAX)
        .saturating_sub(consumed_builtin_tool_calls_before_current_round(state));
    if remaining == 0 {
        return false;
    }
    current_round_tool_call_admissions(state, &selected)
        .into_iter()
        .any(|admitted| admitted)
}

/// Normalize the `input` field into a message array.
///
/// The Responses API `input` can be a string (single user message),
/// a single item object, or an array of items. Normalizes all three
/// forms to a `Vec<Value>`.
fn normalize_input(body: &serde_json::Value) -> Vec<serde_json::Value> {
    normalize_input_value(body.get("input"))
}

/// Normalize one Responses API `input` value into a message array.
pub(crate) fn normalize_input_value(input: Option<&serde_json::Value>) -> Vec<serde_json::Value> {
    input
        .map(|input| normalize_input_owned(input.clone()))
        .unwrap_or_default()
}

/// Normalize an owned `input` value without cloning array or object items.
pub(crate) fn normalize_input_owned(input: serde_json::Value) -> Vec<serde_json::Value> {
    match input {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(item) => vec![serde_json::Value::Object(item)],
        serde_json::Value::String(s) => {
            vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": s
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
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use serde_json::json;

    use super::*;

    fn current_output_bytes(state: &ResponsesState) -> usize {
        retained_json_bytes(&state.response_object).unwrap()
            + retained_json_bytes(&state.local_completion_response_template).unwrap()
    }

    #[test]
    fn indexed_selection_rejects_stale_id_type_and_round() {
        let first = json!({"type":"function_call", "id":"fc_1", "arguments":"large"});
        let assignment = OutputAssignment::new(0, &first).unwrap();
        let mut state = ResponsesState {
            accumulated_output: vec![first],
            tool_calls: vec![assignment.clone()],
            current_round_output_start: Some(0),
            ..ResponsesState::default()
        };
        assert_eq!(state.selected_tool_calls().len(), 1);
        state.accumulated_output.push(json!({"type":"reasoning", "id":"r_2"}));
        assert_eq!(
            state.selected_tool_calls().len(),
            1,
            "growth preserves absolute indices"
        );
        state.current_round_output_start = Some(1);
        assert!(
            state.selected_tool_calls().is_empty(),
            "prior-round selection cannot redispatch"
        );
        state.current_round_output_start = Some(0);
        state.accumulated_output[0] = json!({"type":"function_call", "id":"fc_other"});
        assert!(
            state.selected_tool_calls().is_empty(),
            "replaced item cannot inherit a stale selection"
        );
        state.accumulated_output[0] = json!({"type":"web_search_call", "id":"fc_1"});
        assert!(
            state.selected_tool_calls().is_empty(),
            "wrong item type cannot dispatch"
        );
        assert!(
            OutputAssignment::new(99, &state.accumulated_output[0])
                .unwrap()
                .resolve(&state.accumulated_output, "web_search_call")
                .is_none()
        );
    }

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

        assert_eq!(
            state.retained_payload_bytes().unwrap() - baseline,
            bytes * 4,
            "each of the four owned JSON copies contributes its bytes"
        );
    }

    #[cfg(feature = "store")]
    #[test]
    fn cached_current_output_charges_assignment_id_without_another_call_copy() {
        let id = "f".repeat(8_192);
        let call = json!({"type": "function_call", "id": id, "arguments": "x".repeat(4_096)});
        let assignment = OutputAssignment::new(0, &call).unwrap();
        let mut state = ResponsesState {
            accumulated_output: vec![call.clone()],
            response_object: json!({"output": [call]}),
            ..ResponsesState::default()
        };
        let before = state.retained_payload_bytes().unwrap();
        state.tool_calls.push(assignment);
        let full = state.retained_payload_bytes().unwrap();
        assert_eq!(full - before, id.len(), "the assignment owns only its copied ID");

        let stable = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let prior_output = retained_json_values_bytes(&state.accumulated_output).unwrap();
        let changing = state
            .store_stream_changing_payload_bytes_bounded_with_current_output(usize::MAX, current_output_bytes(&state))
            .unwrap();
        assert_eq!(full, stable + prior_output + changing);
        assert_eq!(state.retained_payload_bytes_bounded(full - 1), None);
        assert_eq!(state.retained_payload_bytes_bounded(full), Some(full));
        assert_eq!(
            state.store_stream_changing_payload_bytes_bounded_with_current_output(
                changing - 1,
                current_output_bytes(&state),
            ),
            None,
            "the cached meter must still reserve the independently owned assignment ID"
        );
        assert_eq!(
            state.store_stream_changing_payload_bytes_bounded_with_current_output(
                changing,
                current_output_bytes(&state),
            ),
            Some(changing)
        );
    }

    #[test]
    fn retained_payload_counts_and_releases_provider_compaction_ids() {
        let id = "c".repeat(8_192);
        let mut state = ResponsesState::from_request_body(json!({
            "input": [{"type": "compaction", "id": &id, "encrypted_content": "opaque"}]
        }));
        let with_id = state.retained_payload_bytes().unwrap();
        let stable_with_id = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        state.provider_compaction_ids.clear();
        assert_eq!(with_id - state.retained_payload_bytes().unwrap(), id.len());
        assert_eq!(
            stable_with_id - state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap(),
            id.len()
        );

        state.provider_compaction_ids.insert(id);
        state.apply_retained_payload_limit(with_id - 1);
        assert!(!state.can_retain_payload(0));
        state.discard_payload_for_budget_error();
        assert!(state.provider_compaction_ids.is_empty());
    }

    #[test]
    fn tool_choice_copies_are_charged_once_in_stream_stable_baseline() {
        let mut state = ResponsesState::default();
        let stable_before = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let changing_before = state
            .stream_changing_payload_bytes_bounded_for_parser(usize::MAX)
            .unwrap();
        let large_choice = json!({"type":"allowed_tools","tools":["x".repeat(32_000)]});
        state.tool_choice = large_choice.clone();
        state.original_tool_choice = Some(large_choice);
        state.mark_replay_stable_payload_changed();

        let stable = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let changing = state
            .stream_changing_payload_bytes_bounded_for_parser(usize::MAX)
            .unwrap();
        assert!(
            stable > stable_before + 64_000,
            "both owned tool-choice values join the stable charge"
        );
        assert_eq!(
            changing, changing_before,
            "per-chunk accounting does not rescan stable tool choices"
        );
        assert_eq!(state.retained_payload_bytes().unwrap(), stable + changing);
        assert_eq!(state.replay_stable_payload_revision, Some(1));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks every streaming consumer of the stable context"
    )]
    fn unchanged_context_fields_use_the_stream_stable_charge() {
        let mut state = ResponsesState::default();
        let stable_before = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let changing_before = state
            .stream_changing_payload_bytes_bounded_for_parser(usize::MAX)
            .unwrap();
        state.context_management = Some(json!({"context": "c".repeat(1_048_576)}));
        state.conversation = Some(json!({"id": "v".repeat(4_096)}));
        state.previous_usage = Some(json!({"note": "p".repeat(4_096)}));
        state.mark_replay_stable_payload_changed();

        let stable = state.stream_stable_payload_bytes_bounded(1_200_000).unwrap();
        let current_output = current_output_bytes(&state);
        assert!(stable > stable_before + 1_048_576);
        assert_eq!(
            state.stream_changing_payload_bytes_bounded_for_parser(usize::MAX),
            Some(changing_before)
        );
        assert_eq!(state.retained_payload_bytes().unwrap(), stable + changing_before);

        let started = std::time::Instant::now();
        for _ in 0..100 {
            for changing in [
                state.stream_changing_payload_bytes_bounded_for_parser(1_024),
                state.chat_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
                #[cfg(feature = "store")]
                state.store_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
            ] {
                assert_eq!(changing, Some(changing_before));
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "stable context must not be serialized per chunk"
        );
    }

    #[test]
    fn client_tool_echo_moves_to_stream_stable_charge_for_every_consumer() {
        let mut state = ResponsesState::default();
        let before_capture = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        state.client_tool_echo = Some(ClientToolEcho {
            tools: vec![json!({"type": "function", "name": "large", "description": "x".repeat(1_048_576)})],
            tool_choice: json!("auto"),
        });
        // Client-tool compatibility calls mark_request_body_for_rebuild after
        // capture, invalidating store and rehydrate stable cache revisions.
        state.mark_request_body_for_rebuild();
        assert_eq!(state.replay_stable_payload_revision, Some(1));
        let stable = state.stream_stable_payload_bytes_bounded(1_200_000).unwrap();
        assert!(stable > before_capture + 1_048_576);
        let full = state.retained_payload_bytes().unwrap();
        let current_output = current_output_bytes(&state);

        let started = std::time::Instant::now();
        for _ in 0..200 {
            let changing = [
                state.stream_changing_payload_bytes_bounded_for_parser(1_024),
                state.chat_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
                #[cfg(feature = "store")]
                state.store_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
            ];
            for charge in changing {
                assert_eq!(stable + charge.unwrap(), full);
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the capture-once echo must not be serialized by changing meters: {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn resolved_mcp_definitions_are_measured_once_in_stream_stable_charge() {
        let mut state = ResponsesState::default();
        let baseline_stable = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let baseline_changing = state.stream_changing_payload_bytes_bounded_for_parser(1_024).unwrap();
        let definition = json!({"description": "x".repeat(1_048_576)});
        let map_bytes = "server".len() + "tool".len() + retained_json_bytes(&definition).unwrap();
        state
            .mcp_tool_map
            .insert(("server".to_owned(), "tool".to_owned()), definition);
        state.mark_replay_stable_payload_changed();

        let stable = state.stream_stable_payload_bytes_bounded(1_200_000).unwrap();
        assert_eq!(stable, baseline_stable + map_bytes);
        let current_output = current_output_bytes(&state);
        for changing in [
            state.stream_changing_payload_bytes_bounded_for_parser(1_024),
            state.chat_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
            #[cfg(feature = "store")]
            state.store_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
        ] {
            assert_eq!(changing, Some(baseline_changing));
        }
        assert_eq!(state.retained_payload_bytes().unwrap(), stable + baseline_changing);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks every deferred descriptor owner against all stream meters"
    )]
    fn deferred_mcp_payload_is_measured_once_in_stream_stable_charge() {
        let mut state = ResponsesState::default();
        let baseline_stable = state.stream_stable_payload_bytes_bounded(usize::MAX).unwrap();
        let baseline_changing = state.stream_changing_payload_bytes_bounded_for_parser(1_024).unwrap();
        let allowed_tools = json!({"names": ["x".repeat(1_048_576)]});
        let headers = json!({"x-tenant": "team"});
        let approval = json!({"never": ["lookup"]});
        let connector_bytes = "id".len()
            + "label".len()
            + "https://mcp.example.com".len()
            + "secret".len()
            + retained_json_bytes(&allowed_tools).unwrap()
            + retained_json_bytes(&headers).unwrap()
            + retained_json_bytes(&approval).unwrap();
        state.deferred_mcp.push(DeferredMcpConnector {
            authorization: Some("secret".to_owned()),
            allowed_tools: Some(allowed_tools),
            connector_id: "id".to_owned(),
            headers: Some(headers),
            max_rewritten_body_bytes: 1_200_000,
            max_tools: 128,
            require_approval: Some(approval),
            server_label: "label".to_owned(),
            server_url: "https://mcp.example.com".to_owned(),
            timeout: Duration::from_secs(5),
        });
        #[cfg(feature = "openai-mcp-tools")]
        let policy_bytes = {
            state.mcp_connector_context_policy =
                McpConnectorContextPolicy::new(Some("credential-slot"), Some("assertion-slot"));
            state.mcp_connector_context_policy.retained_payload_bytes().unwrap()
        };
        #[cfg(not(feature = "openai-mcp-tools"))]
        let policy_bytes = 0;
        state.mark_replay_stable_payload_changed();

        let stable = state.stream_stable_payload_bytes_bounded(1_200_000).unwrap();
        assert_eq!(stable, baseline_stable + connector_bytes + policy_bytes);
        let current_output = current_output_bytes(&state);
        let started = std::time::Instant::now();
        for _ in 0..100 {
            for changing in [
                state.stream_changing_payload_bytes_bounded_for_parser(1_024),
                state.chat_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
                #[cfg(feature = "store")]
                state.store_stream_changing_payload_bytes_bounded_with_current_output(1_024, current_output),
            ] {
                assert_eq!(changing, Some(baseline_changing));
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "deferred descriptors must not be rescanned per chunk"
        );
        assert_eq!(state.retained_payload_bytes().unwrap(), stable + baseline_changing);
    }

    #[cfg(feature = "store")]
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
        let function = json!({"type":"function_call", "id":"fc_1", "payload":"v"});
        let web = json!({"type":"web_search_call", "id":"ws_1", "payload":"v"});
        state.select_test_output("function_call", vec![function.clone()]);
        state.select_test_output("web_search_call", vec![web.clone()]);
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
        let json_delta =
            bytes * 8 - replaced_null + retained_json_bytes(&function).unwrap() + retained_json_bytes(&web).unwrap();
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
        let raw_delta = raw_delta + "fc_1".len() + "ws_1".len();
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
    fn from_request_body_records_provider_compaction_ids_for_stateless_chaining() {
        let provider_item = json!({
            "type": "compaction",
            "id": "cmp_provider",
            "encrypted_content": "opaque"
        });
        let state = ResponsesState::from_request_body(json!({
            "model": "gpt-4o",
            "input": [provider_item, {
                "type": "compaction",
                "id": "compact_local",
                "encrypted_content": "local"
            }]
        }));

        assert!(
            state.provider_compaction_ids.contains("cmp_provider"),
            "provider compaction IDs should include opaque provider item IDs"
        );
        assert!(
            !state.provider_compaction_ids.contains("compact_local"),
            "provider compaction IDs should exclude locally generated compaction IDs"
        );
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
            item_id: "fs_first".to_owned(),
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
        let search = json!({"type": "tool_search_call", "id": "tsc_1", "status": "completed"});
        let mut exhausted = ResponsesState {
            max_tool_calls: Some(0),
            ..ResponsesState::default()
        };
        exhausted.select_test_output("tool_search_call", vec![search.clone()]);
        assert!(
            !tool_search_discovery_is_within_budget(&exhausted),
            "a zero remaining budget must not admit deferred tools/list"
        );

        let mut admitted = ResponsesState {
            max_tool_calls: Some(1),
            ..ResponsesState::default()
        };
        admitted.select_test_output("tool_search_call", vec![search]);
        assert!(
            tool_search_discovery_is_within_budget(&admitted),
            "the first admitted hosted search may still list deferred connectors"
        );
    }

    #[test]
    fn tool_search_discovery_follows_current_round_admission_order() {
        let search = json!({"type": "tool_search_call", "id": "tsc_1", "status": "completed"});
        let web = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
        let mut displaced = ResponsesState {
            max_tool_calls: Some(1),
            accumulated_output: vec![web.clone(), search.clone()],
            response_object: json!({"output": [web, search.clone()]}),
            ..ResponsesState::default()
        };
        displaced.select_test_output("tool_search_call", vec![search]);
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
        assert!(
            !state.logical_stream_terminal_emitted,
            "logical stream terminal must start unemitted"
        );
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
            item_id: "fs_2".to_owned(),
            synthesis: SynthesisKind::Private,
        });
        state.file_search_assignments.push(FileSearchAssignment {
            output_index: 5,
            item_id: "fs_5".to_owned(),
            synthesis: SynthesisKind::Native,
        });
        let drained = state.drain_file_search_assignments();
        assert_eq!(
            drained,
            vec![
                FileSearchAssignment {
                    output_index: 2,
                    item_id: "fs_2".to_owned(),
                    synthesis: SynthesisKind::Private,
                },
                FileSearchAssignment {
                    output_index: 5,
                    item_id: "fs_5".to_owned(),
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
        assert!(
            state.finalize_response_body(&mut body).is_err(),
            "citation staging above the retained budget must fail finalization"
        );
        assert!(
            state.retained_payload_failed,
            "citation staging budget failure must mark retained payload as failed"
        );
        assert!(
            state.accumulated_output.is_empty(),
            "failed finalization releases retained output"
        );
        assert!(body.is_none(), "failed citation preflight must not produce a body");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "asserts each payload owner is released after reservation fails"
    )]
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
        state.buffered_canonical_finalized = true;

        let mut body = None;
        assert!(
            state.finalize_response_body(&mut body).is_err(),
            "serialization reservation above the retained budget must fail finalization"
        );
        assert!(
            state.retained_payload_failed,
            "serialization reservation failure must mark retained payload as failed"
        );
        assert!(
            state.response_object.is_null(),
            "canonical response must not be committed"
        );
        assert!(
            state.accumulated_output.is_empty(),
            "failed request payload is released"
        );
        assert!(
            body.is_none(),
            "failed serialization reservation must not produce a body"
        );
        assert!(
            !state.buffered_canonical_finalized,
            "a failed finalization cannot authorize buffered header persistence"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "wire allocation and canonical marker share one finalization witness"
    )]
    fn finalize_response_body_serializes_with_one_admitted_wire_allocation() {
        let mut state = ResponsesState {
            response_object: json!({
                "object": "response",
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "x".repeat(1_048_576)}]
                }]
            }),
            ..ResponsesState::default()
        };
        let serialized_bytes = bounded_json_size(&state.response_object, MAX_JSON_BODY_BYTES)
            .unwrap()
            .unwrap();
        let baseline = state.retained_payload_bytes().unwrap();
        state.apply_retained_payload_limit(baseline + serialized_bytes);

        let mut body = None;
        let allocations = allocation_counter::measure(|| {
            state.finalize_response_body(&mut body).unwrap();
        });
        assert_eq!(body.as_ref().map(Bytes::len), Some(serialized_bytes));
        assert!(state.buffered_canonical_finalized);
        assert!(!state.retained_payload_failed);
        assert!(
            allocations.bytes_max <= u64::try_from(serialized_bytes).unwrap() + 65_536,
            "one wire-sized buffer is admitted; serializer growth must not keep a second buffer: {allocations:?}"
        );
        state.mark_current_output_changed();
        assert!(
            !state.buffered_canonical_finalized,
            "a later output mutation invalidates the serialized body"
        );
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

    #[test]
    fn record_security_failure_is_write_once() {
        let mut state = ResponsesState::default();
        assert!(state.security_failure.is_none(), "security failure must start unset");

        state.record_security_failure(DispatchFailure {
            status: 401,
            code: "missing_callout_context",
            message: "first".to_owned(),
        });
        state.record_security_failure(DispatchFailure {
            status: 500,
            code: "other",
            message: "second".to_owned(),
        });

        let f = state.security_failure.as_ref().expect("recorded");
        assert_eq!(f.status, 401);
        assert_eq!(f.code, "missing_callout_context");
        assert_eq!(f.message, "first", "first failure wins (write-once)");
    }

    #[test]
    fn request_tools_returns_lowered_request_body_tools_over_canonical() {
        // `openai_client_tool_compat` lowers rich client tools into
        // `request_body["tools"]` only, leaving canonical `state.tools` rich. The
        // accessor must return the lowered outbound view so downstream translation
        // (`responses_to_chat_completions`) sees valid `function` tools.
        let mut state = ResponsesState::from_request_body(json!({
            "model": "m",
            "tools": [{"type": "custom", "name": "apply_patch"}],
        }));
        assert_eq!(
            state.tools,
            vec![json!({"type": "custom", "name": "apply_patch"})],
            "canonical tools start as the rich client types",
        );
        // Simulate compat lowering: rewrite only the outbound request body.
        state.request_body["tools"] = json!([{"type": "function", "name": "apply_patch"}]);

        assert_eq!(
            state.request_tools(),
            &[json!({"type": "function", "name": "apply_patch"})],
            "accessor returns the lowered outbound tools, not canonical rich tools",
        );
        assert_eq!(
            state.tools,
            vec![json!({"type": "custom", "name": "apply_patch"})],
            "accessor is read-only: canonical tools remain rich",
        );
    }

    #[test]
    fn request_tools_falls_back_to_canonical_when_key_absent() {
        // No outbound override present: the accessor must return canonical tools so
        // non-compat pipelines are byte-identical.
        let mut state = ResponsesState::from_request_body(json!({"model": "m"}));
        state.tools = vec![json!({"type": "function", "name": "lookup"})];
        state
            .request_body
            .as_object_mut()
            .expect("request body is an object")
            .remove("tools");

        assert_eq!(
            state.request_tools(),
            &[json!({"type": "function", "name": "lookup"})],
            "accessor falls back to canonical tools when request_body has no tools key",
        );
    }

    #[test]
    fn request_tools_falls_back_to_canonical_when_non_array() {
        // A present-but-malformed override must not be trusted; fall back to canonical.
        let mut state = ResponsesState::from_request_body(json!({"model": "m"}));
        state.tools = vec![json!({"type": "function", "name": "lookup"})];
        state.request_body["tools"] = json!("not-an-array");

        assert_eq!(
            state.request_tools(),
            &[json!({"type": "function", "name": "lookup"})],
            "accessor falls back to canonical tools when request_body tools is not an array",
        );
    }

    #[test]
    fn request_tool_choice_returns_lowered_request_body_value_over_canonical() {
        // Compat may lower a rich `tool_choice` (e.g. an `allowed_tools` object) into
        // the outbound body; the accessor must surface the lowered value.
        let mut state = ResponsesState::from_request_body(json!({
            "model": "m",
            "tool_choice": {"type": "allowed_tools", "tools": [{"type": "custom", "name": "apply_patch"}]},
        }));
        state.request_body["tool_choice"] = json!("required");

        assert_eq!(
            state.request_tool_choice(),
            &json!("required"),
            "accessor returns the lowered outbound tool_choice",
        );
        assert_eq!(
            state.tool_choice,
            json!({"type": "allowed_tools", "tools": [{"type": "custom", "name": "apply_patch"}]}),
            "accessor is read-only: canonical tool_choice remains rich",
        );
    }

    #[test]
    fn request_tool_choice_falls_back_to_canonical_when_key_absent() {
        // Preserve `auto` omission: when the outbound body carries no explicit
        // tool_choice, the accessor must return the canonical value unchanged rather
        // than synthesizing one.
        let mut state = ResponsesState::from_request_body(json!({"model": "m"}));
        state.tool_choice = json!("none");
        state
            .request_body
            .as_object_mut()
            .expect("request body is an object")
            .remove("tool_choice");

        assert_eq!(
            state.request_tool_choice(),
            &json!("none"),
            "accessor falls back to canonical tool_choice when request_body has no key",
        );
    }
}
