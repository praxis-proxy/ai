// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Compatibility adapter that lets a rich Codex-style Responses client talk to a
//! function-only Responses backend (#1131).
//!
//! Some Responses backends (for example vLLM at `POST /v1/responses`) accept only
//! `type: "function"` tool declarations and only emit `function_call` output
//! items. A Codex client, however, declares richer *client-owned* tools —
//! freeform `custom` tools, `namespace` groupings, a local `shell`, and a
//! client-executed `tool_search` — and expects the matching typed output items
//! back. This filter bridges the two **without** routing through
//! `/v1/chat/completions` and **without** ever executing a client-owned tool
//! inside Praxis.
//!
//! # Request phase — lowering
//!
//! On the outbound request the filter rewrites the client's rich tool
//! declarations into private `function` declarations the backend understands:
//!
//! - `custom` → a `function` whose single string `input` parameter carries the freeform payload; the model-visible
//!   description preserves the declared contract.
//! - `namespace` members (both `function` and `custom`) → flat `function` tools named
//!   `agentic_ns__{namespace}__{member}` (hashed when longer than 64 chars); a `custom` member keeps the single string
//!   `input` contract, and the namespace's required model-visible description is folded into every member so the group
//!   context survives flattening.
//! - local `shell` → a `shell` `function` tool whose description folds in any declared `environment.skills` (name,
//!   description, path) so the model still sees the available local skills.
//! - client-executed `tool_search` → a fixed `tool_search` `function` tool.
//!
//! Matching `tool_choice` selectors and prior typed client-owned items already in
//! the conversation history are lowered the same way so a continuation turn stays
//! valid for a function-only backend. A lowered client tool call's *output* item
//! (for example a `custom_tool_call_output` or `shell_call_output`) becomes a
//! `function_call_output` that carries every representation-compatible field the
//! backend can consume — the correlating `call_id`, the produced `output`, and any
//! schema-valid `id`, `status`, and `caller` — so continuation metadata is not
//! silently dropped.
//!
//! A client-executed `tool_search` that already ran in an earlier turn reports the
//! tools it discovered in a `tool_search_output.tools` array. On a **buffered**
//! continuation those discovered definitions are themselves lowered and hoisted
//! into the outbound `function` set (subject to the same name-collision and
//! tool-cap checks) so the function-only backend can actually call a tool the
//! search surfaced; a discovered tool that collides with an existing name, exceeds
//! the cap, or is itself an unsupported kind fails closed with HTTP 400. A
//! `tool_search_output` whose `status` is `completed`, absent, or null contributes
//! its tools (the field is optional and nullable in the input schema, and a
//! client-supplied continuation history commonly omits it); an explicitly
//! non-terminal or unrecognized status is treated as not-yet-authoritative and
//! contributes nothing. A discovered tool is being loaded now, so its
//! Responses-only `defer_loading` hint is consumed and it becomes callable: a
//! discovered `function` (top level or a `namespace` member) is sanitized to the
//! backend-callable fields (a restrictive `allowed_callers` or malformed
//! `output_schema` fails closed) and a discovered `custom` gains its single-string
//! `input` contract. Conversely, on the rich/discovered lowering path a tool the
//! client declared with `defer_loading: true` — a top-level `function` or `custom`,
//! or a `namespace` member of either kind — is withheld from the outbound set until
//! a search hoists it, so its deferred declaration never collides with its own
//! discovered copy (a request whose rich tools are *all* deferred forwards an empty
//! callable set, its originals echoed back on the response). A request with no rich
//! client tool and no discovered tools instead takes the native passthrough path,
//! which forwards its `function` declarations — including any `defer_loading` hint a
//! function-only backend can honor or ignore — verbatim. The
//! same `(type, name)` redefined across searches with definitions that lower to
//! different backend declarations is ambiguous and fails closed with HTTP 400;
//! listings that differ only in a decorator lowering drops (for example
//! `defer_loading` or a `null` `output_schema`) reduce to one hoisted tool.
//!
//! The lowered names never leak to the client:
//! a per-request reverse map (`LoweredClientTool`) and a verbatim
//! `ClientToolEcho` snapshot of the original `tools`/`tool_choice` are recorded
//! in `ResponsesState` for the response phase.
//!
//! The filter fails closed with HTTP 400 **before any upstream request** on a
//! name collision, an unsupported custom format (a non-`text` constrained
//! `format`), an invalid lowered function name, a
//! `namespace` missing its required `name`/`description`/`tools` or whose member
//! is neither `function` nor `custom`, a local `shell` whose `environment.skills`
//! is `null` or not an array, declares more than the schema maximum of 200
//! entries, or contains a skill missing any of its required string
//! `name`/`description`/`path` fields, a hoisted namespace member or discovered
//! `function` that declares a restrictive `allowed_callers` or a malformed
//! (non-object, non-null) `output_schema`, or a `tool_choice` selector that names
//! an undeclared client tool or whose declared kind disagrees with the member it
//! selects. A rewrite that grows the outbound body past the configured
//! `max_rewritten_body_bytes` — including a history-only continuation whose escaped
//! inputs expand — fails closed with HTTP 413 before any upstream call.
//!
//! ## `local_shell` is not yet supported
//!
//! The legacy `local_shell` tool restores to a distinct `local_shell_call` output
//! shape this adapter does not yet reconstruct. A `local_shell` *declaration* (top
//! level or as a namespace member) fails closed with HTTP 400, and its typed
//! history items (`local_shell_call`/`local_shell_call_output`) are rejected the
//! same way so an earlier turn's unsupported call can never reach a function-only
//! backend unchanged on a continuation. Callers should declare the modern local
//! shell tool (`type: "shell"` with `environment.type: "local"`) instead. Full
//! `local_shell` lowering/restoration is a follow-up.
//!
//! # Response phase — restoration
//!
//! Running immediately before `openai_agentic_loop` on the buffered response, the
//! filter restores each returned `function_call` whose name is in the reverse map
//! back to its canonical typed item (`custom_tool_call`, a namespaced
//! `function_call`, a namespaced `custom_tool_call`, `shell_call`, or
//! `tool_search_call`) and restores
//! `response.tools`/`response.tool_choice` from the echo snapshot. It fails closed
//! with HTTP 502 **before emitting a successful terminal response** when a
//! `shell_call` or `tool_search_call` cannot be reconstructed losslessly
//! (malformed arguments, or a non-completed client tool call).
//!
//! # Streaming
//!
//! On the streaming Responses path (`stream: true`) the lowered client-tool calls
//! are restored **live in the SSE lifecycle** by the `openai_stream_events` logical
//! owner, not by this filter: this filter records the reverse lowering map and the
//! `tools`/`tool_choice` echo snapshot on the request, and `openai_stream_events`
//! re-types each lowered `function_call` as it streams (#1159).
//!
//! That hand-off requires `openai_stream_events` to be placed **before**
//! `openai_client_tool_compat` in the inference step: on the request it publishes a
//! marker arming streaming restoration, which this filter reads before lowering. If
//! a streaming request declares rich client tools (or carries `tool_search`-discovered
//! tools that would be hoisted) but no `openai_stream_events` owner is armed ahead of
//! it, the filter fails closed with **HTTP 500** — an operator misconfiguration —
//! **before any upstream call**, so a private lowered `function` name can never reach
//! an un-restored SSE stream. A streaming request that neither declares rich client
//! tools nor carries discovered tools stays a transparent passthrough.
//!
//! When the request declares no rich client tools and carries no discovered tools
//! the filter is a transparent passthrough, so native traffic is unchanged.

mod config;

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
mod tests;

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde_json::{Map, Value, json};
use tracing::debug;

use self::config::{ClientToolCompatConfig, build_config};
use super::{
    body_limits::reject_rewritten_body_too_large,
    error::responses_error_rejection,
    state::{ClientToolEcho, ClientToolRestore, LoweredClientTool, ResponsesState, is_client_executed_tool_call},
};
use crate::json_body::{SerializedJson, serialize_json_body};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Fixed private function name a local `shell` tool lowers to.
const SHELL_FUNCTION_NAME: &str = "shell";

/// Fixed private function name a client-executed `tool_search` tool lowers to.
const TOOL_SEARCH_NAME: &str = "tool_search";

/// Prefix applied to a flattened Codex namespace member function name.
const NAMESPACE_MEMBER_PREFIX: &str = "agentic_ns__";

/// Separator [`namespace_member_name`] reserves to delimit the prefix, namespace,
/// and member components of a flattened wire name. A namespace or member name that
/// embeds it — or abuts it with a leading/trailing `_` — fails closed so the
/// flattening stays injective (see [`reject_reserved_namespace_delimiter`]).
const NAMESPACE_NAME_DELIMITER: &str = "__";

/// Maximum length of a lowered, model-visible function name (OpenAI schema).
const MAX_FUNCTION_NAME_LEN: usize = 64;

/// Maximum number of `environment.skills` entries a local `shell` tool may
/// declare (`LocalEnvironmentParam.skills` `maxItems: 200`).
const MAX_LOCAL_SKILLS: usize = 200;

/// Length of the `___{hash:016x}` suffix appended to a shortened namespace name.
/// The triple-underscore marker domain-separates the hashed shape from every
/// verbatim wire name (see [`namespace_member_name`]): `3 + 16 = 19`.
const HASHED_NAMESPACE_MEMBER_SUFFIX_LEN: usize = 19;

/// FNV-1a offset basis, matching the upstream stable name hash.
const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;

/// FNV-1a prime, matching the upstream stable name hash.
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// Default freeform tool-search description when the client omits one.
const TOOL_SEARCH_DEFAULT_DESCRIPTION: &str = "Search the client tool catalog";

/// Default tool-search `query` parameter description.
const TOOL_SEARCH_DEFAULT_QUERY_DESCRIPTION: &str = "A concise description of the needed capabilities.";

// -----------------------------------------------------------------------------
// ClientToolCompatFilter
// -----------------------------------------------------------------------------

/// Lowers rich client-owned tool declarations to private `function` tools for a
/// function-only Responses backend and restores the typed items on the way back.
///
/// # YAML
///
/// ```yaml
/// filter: openai_client_tool_compat
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_client_tool_compat
/// max_rewritten_body_bytes: 67108864
/// max_client_tools: 512
/// ```
///
/// Place the filter between `openai_agentic_loop` and `openai_responses_proxy` so
/// it lowers after history is prepared and before the outbound body is
/// serialized, and restores after the upstream body is captured and before the
/// agentic loop parses it.
pub struct ClientToolCompatFilter {
    /// Maximum size in bytes of a request or response body produced by lowering
    /// or restoration.
    max_rewritten_body_bytes: usize,

    /// Maximum number of client-owned tool declarations lowered per request.
    max_client_tools: usize,
}

impl ClientToolCompatFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ClientToolCompatConfig = parse_filter_config("openai_client_tool_compat", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self {
            max_rewritten_body_bytes: validated.max_rewritten_body_bytes,
            max_client_tools: validated.max_client_tools,
        }))
    }

    /// Lower the request's rich client tools, `tool_choice`, and typed history to
    /// the private function shape a function-only backend accepts.
    ///
    /// Returns `Err` with a terminal rejection when lowering cannot proceed
    /// losslessly; the request body is left byte-identical so no upstream call is
    /// made with a partial rewrite.
    fn lower_request(
        &self,
        state: &mut ResponsesState,
        streaming: bool,
        stream_restoration_armed: bool,
    ) -> Result<(), FilterAction> {
        // A present `tools` that is neither an array nor `null` is structurally
        // malformed: this filter's whole contract treats `tools` as an array. Fail
        // closed uniformly here — before any lowering, discovery, or native
        // passthrough — rather than rejecting it only on the discovery path while
        // forwarding the identical malformed value on the passthrough path. An absent
        // or `null` `tools` is treated as "no tools" and left for the normal paths.
        if let Some(tools) = state.request_body.get("tools")
            && !tools.is_null()
            && !tools.is_array()
        {
            return Err(reject_bad_request("tools must be a JSON array"));
        }
        let has_rich = request_has_rich_client_tool(state);
        // Discovered tools are always collected (even for a streaming request) so
        // an ambiguous discovery — a conflicting redefinition of the same tool — is
        // caught before any upstream call rather than being silently ignored on the
        // streaming path.
        let discovered = collect_discovered_tools(&state.messages)?;
        // #1159: streaming restoration requires the openai_stream_events logical
        // owner to be present (it published the marker in on_request). Without it
        // there is no SSE owner to restore lowered calls, so fail closed rather than
        // stream private lowered `function` shapes to the client.
        if streaming && (has_rich || !discovered.is_empty()) && !stream_restoration_armed {
            return Err(reject_streaming_missing_owner());
        }
        if !has_rich && discovered.is_empty() {
            // Native passthrough: still lower any prior typed client-owned items
            // so a continuation turn stays valid for a function-only backend.
            return lower_history_only(state, self.max_rewritten_body_bytes);
        }

        self.lower_declared_and_discovered(state, &discovered)
    }

    /// Lower the declared rich client tools and hoist any discovered tools into the
    /// outbound `function` set, then commit. Restores the tools verbatim on any
    /// failure so no partially lowered state leaks to the backend.
    fn lower_declared_and_discovered(
        &self,
        state: &mut ResponsesState,
        discovered: &[Value],
    ) -> Result<(), FilterAction> {
        // `lower_request` already failed closed on a present, non-null, non-array
        // `tools`, so here `tools` is an array (the declared/rich path) or absent /
        // `null` (a discovery-only continuation, normalized by `take_request_tools`
        // to an empty array the discovered set is hoisted onto).
        let original_tools = take_request_tools(state).unwrap_or_default();
        let mut lowering = Lowering::new(self.max_client_tools);
        let (mut lowered_tools, lowered_any) = match lowering.lower_tools(&original_tools) {
            Ok(result) => result,
            Err(action) => {
                restore_tools(state, original_tools);
                return Err(action);
            },
        };
        if !lowered_any && discovered.is_empty() {
            return self.finish_without_lowered_or_discovered(state, lowering, original_tools, lowered_tools);
        }

        if let Err(action) = lowering.lower_discovered_tools(discovered, &mut lowered_tools) {
            restore_tools(state, original_tools);
            return Err(action);
        }

        commit_lowering(
            state,
            lowering,
            original_tools,
            lowered_tools,
            self.max_rewritten_body_bytes,
        )
    }

    /// Finish a request that lowered nothing rich and has nothing to hoist.
    ///
    /// When rich tools were deferred and withheld, commit the lowered set (the rich
    /// declarations removed) rather than restoring the originals: a withheld
    /// `custom`/`namespace` cannot be forwarded verbatim to a function-only backend,
    /// and the client's originals are still echoed back on the buffered response via
    /// `client_tool_echo`. Otherwise the `has_rich` scan over-matched but nothing was
    /// actually lowerable, so restore the tools verbatim and lower history only.
    fn finish_without_lowered_or_discovered(
        &self,
        state: &mut ResponsesState,
        lowering: Lowering,
        original_tools: Vec<Value>,
        lowered_tools: Vec<Value>,
    ) -> Result<(), FilterAction> {
        if lowering.withheld_any {
            return commit_lowering(
                state,
                lowering,
                original_tools,
                lowered_tools,
                self.max_rewritten_body_bytes,
            );
        }
        restore_tools(state, original_tools);
        lower_history_only(state, self.max_rewritten_body_bytes)
    }

    /// Restore lowered `function_call` items and echoed `tools`/`tool_choice` in a
    /// buffered response body.
    ///
    /// Returns `Ok(None)` when nothing needs rewriting (the body is not a buffered
    /// Responses object, or the request lowered nothing).
    fn restore_response(&self, state: &ResponsesState, bytes: &[u8]) -> Result<Option<SerializedJson>, FilterAction> {
        if state.client_tool_echo.is_none() {
            return Ok(None);
        }
        let Ok(mut response) = serde_json::from_slice::<Value>(bytes) else {
            // Streaming SSE or a non-JSON body: leave it for `openai_stream_events`.
            return Ok(None);
        };
        if response.get("object").and_then(Value::as_str) != Some("response") {
            return Ok(None);
        }

        if let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) {
            for item in output.iter_mut() {
                restore_output_item(item, &state.client_tool_lowering).map_err(reject_lossy_restore)?;
            }
        }

        restore_snapshot_tools(&mut response, state.client_tool_echo.as_ref());

        let serialized = serialize_json_body(&response).map_err(|error| {
            FilterAction::Reject(responses_error_rejection(502, "server_error", &error.to_string()))
        })?;
        if serialized.len() > self.max_rewritten_body_bytes {
            return Err(reject_rewritten_body_too_large(
                serialized.len(),
                self.max_rewritten_body_bytes,
            ));
        }
        Ok(Some(serialized))
    }

    /// Ratchet the upstream response to a bounded `StreamBuffer` and strip the
    /// request's `Accept-Encoding` header so [`Self::restore_response`] sees a
    /// single, complete, uncompressed body.
    ///
    /// Both are armed only once lowering recorded a [`ClientToolEcho`]. Without
    /// buffering, a non-streaming response delivered in multiple chunks would let
    /// early chunks reach the client with lowered private function names un-restored
    /// (`restore_response` only rewrites the final end-of-stream chunk, and a
    /// partial chunk fails JSON parsing and passes through). Without the
    /// `Accept-Encoding` strip, a compressed body would likewise fail to parse and
    /// pass through with those names intact. The IRR clears `request_headers_to_remove`
    /// per inference iteration, so this re-strips the header on every continuation
    /// round that lowers. The buffer is bounded by the same rewrite cap
    /// `restore_response` enforces, failing closed rather than buffering an
    /// unbounded un-restored body.
    fn arm_restoration(&self, ctx: &mut HttpFilterContext<'_>) {
        ctx.set_response_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(self.max_rewritten_body_bytes),
        });
        ctx.request_headers_to_remove.push(http::header::ACCEPT_ENCODING);
    }
}

#[async_trait]
impl HttpFilter for ClientToolCompatFilter {
    fn name(&self) -> &'static str {
        "openai_client_tool_compat"
    }

    fn request_body_access(&self) -> BodyAccess {
        // Lowering mutates `ResponsesState`; the outbound body is serialized by
        // `openai_responses_proxy`. The raw request bytes are not needed here.
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        // The step already buffers for the proxy; only the end-of-stream signal
        // is required so lowering runs once on complete state.
        BodyMode::Stream
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        // Stream by default so a request that lowers nothing never pays to buffer
        // the response. When lowering records an echo, `on_request_body` ratchets
        // this up to a bounded `StreamBuffer` (see `arm_restoration`) so the whole
        // response is restored before any bytes reach the client. A streaming
        // response is never armed (an echo is only set on a buffered, non-streaming
        // request) and is handled by `openai_stream_events`.
        BodyMode::Stream
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
        let streaming = request_is_streaming(ctx);
        let stream_restoration_armed = ctx.get_metadata("responses.client_tool_stream_restoration").is_some();
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        if let Err(action) = self.lower_request(state, streaming, stream_restoration_armed) {
            return Ok(action);
        }
        // The `state` borrow above ends here. Re-read the echo flag before mutating
        // `ctx`: when lowering armed restoration, buffer the response and strip
        // `Accept-Encoding` so restoration sees a single, complete, uncompressed
        // body (see `arm_restoration`). On the streaming path, the stream owner
        // drives restoration, so the compat filter must not buffer.
        if !streaming && restoration_armed(ctx) {
            self.arm_restoration(ctx);
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Continue);
        };
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        match self.restore_response(state, bytes) {
            Ok(Some(serialized)) => {
                serialized.commit(body, self.name(), "body");
                Ok(FilterAction::Continue)
            },
            Ok(None) => Ok(FilterAction::Continue),
            Err(action) => Ok(action),
        }
    }
}

// -----------------------------------------------------------------------------
// Request lowering
// -----------------------------------------------------------------------------

/// Restore an owned `tools` array back into the request body verbatim.
fn restore_tools(state: &mut ResponsesState, tools: Vec<Value>) {
    if let Some(obj) = state.request_body.as_object_mut() {
        obj.insert("tools".to_owned(), Value::Array(tools));
    }
}

/// Insert `field` under `key` in a JSON object value, ignoring non-objects.
fn insert_field(value: &mut Value, key: &str, field: Value) {
    if let Some(object) = value.as_object_mut() {
        object.insert(key.to_owned(), field);
    }
}

/// Take the request's `tools` array out of `state`, leaving no lowerable slot.
///
/// Returns `None` when `tools` is absent or is not an array, which the caller
/// treats as nothing to lower.
fn take_request_tools(state: &mut ResponsesState) -> Option<Vec<Value>> {
    let slot = state.request_body.get_mut("tools")?;
    if !slot.is_array() {
        // Leave a non-array `tools` untouched so the discovered-tool-only path does
        // not destroy a value it did not consume.
        return None;
    }
    match std::mem::take(slot) {
        Value::Array(tools) => Some(tools),
        _ => None,
    }
}

/// Apply a completed [`Lowering`] to the request body, lower any `tool_choice`
/// and typed history, and record the restoration state for the response phase.
///
/// On a `tool_choice` failure the original `tools` are restored verbatim so no
/// upstream call is made with a partial rewrite.
fn commit_lowering(
    state: &mut ResponsesState,
    lowering: Lowering,
    original_tools: Vec<Value>,
    lowered_tools: Vec<Value>,
    max_rewritten_body_bytes: usize,
) -> Result<(), FilterAction> {
    let parts = lowering.into_parts();
    let original_tool_choice = state.request_body.get("tool_choice").cloned().unwrap_or(Value::Null);
    let mut lowered_tool_choice = original_tool_choice.clone();
    if let Err(action) = lower_tool_choice(&mut lowered_tool_choice, &parts) {
        restore_tools(state, original_tools);
        return Err(action);
    }

    if let Some(obj) = state.request_body.as_object_mut() {
        obj.insert("tools".to_owned(), Value::Array(lowered_tools));
        if !original_tool_choice.is_null() {
            obj.insert("tool_choice".to_owned(), lowered_tool_choice);
        }
    }

    // Fail closed on an unsupported prior history item before committing, rolling
    // the request tools/tool_choice back so no partial rewrite is sent upstream.
    if let Err(action) = lower_history_items(&mut state.messages) {
        restore_request_tools_and_choice(state, original_tools, &original_tool_choice);
        return Err(action);
    }

    let original_tools = enforce_rewrite_cap(state, original_tools, &original_tool_choice, max_rewritten_body_bytes)?;

    debug!(
        lowered = parts.reverse.len(),
        "openai_client_tool_compat lowered client tools to private functions"
    );
    state.client_tool_lowering = parts.reverse;
    state.client_tool_echo = Some(ClientToolEcho {
        tools: original_tools,
        tool_choice: original_tool_choice,
    });
    state.mark_request_body_for_rebuild();
    Ok(())
}

/// Enforce the operator-selected rewrite cap on the fully rebuilt outbound body.
///
/// The proxy's independent 64 MiB limit cannot honour a smaller configured cap,
/// and an expanded custom description can grow the request. On overflow or a
/// serialization failure, roll the request `tools`/`tool_choice` back to the
/// client's originals and fail closed before any upstream call so no partial
/// rewrite is sent. Returns `original_tools` on success so the caller can record
/// them in the echo snapshot.
fn enforce_rewrite_cap(
    state: &mut ResponsesState,
    original_tools: Vec<Value>,
    original_tool_choice: &Value,
    max_rewritten_body_bytes: usize,
) -> Result<Vec<Value>, FilterAction> {
    let outbound_len = match super::openai_responses_proxy::serialized_outbound_body_len(state) {
        Ok(len) => len,
        Err(error) => {
            restore_request_tools_and_choice(state, original_tools, original_tool_choice);
            return Err(FilterAction::Reject(responses_error_rejection(
                500,
                "server_error",
                &format!("failed to measure rewritten request body: {error}"),
            )));
        },
    };
    if outbound_len > max_rewritten_body_bytes {
        restore_request_tools_and_choice(state, original_tools, original_tool_choice);
        return Err(reject_rewritten_body_too_large(outbound_len, max_rewritten_body_bytes));
    }
    Ok(original_tools)
}

/// Restore the request's `tools` and `tool_choice` to the client's originals,
/// used to roll back a rewrite that exceeded the configured body cap.
fn restore_request_tools_and_choice(state: &mut ResponsesState, tools: Vec<Value>, tool_choice: &Value) {
    if let Some(obj) = state.request_body.as_object_mut() {
        obj.insert("tools".to_owned(), Value::Array(tools));
        if tool_choice.is_null() {
            obj.remove("tool_choice");
        } else {
            obj.insert("tool_choice".to_owned(), tool_choice.clone());
        }
    }
}

/// Return the client's stream preference: the classifier metadata first, falling
/// back to the request body's `stream` flag.
fn request_is_streaming(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream").map_or_else(
        || {
            ctx.extensions
                .get::<ResponsesState>()
                .and_then(|state| state.request_body.get("stream"))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        },
        |value| value == "true",
    )
}

/// Return whether the request declares at least one rich client-owned tool.
fn request_has_rich_client_tool(state: &ResponsesState) -> bool {
    state
        .request_body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_rich_client_tool))
}

/// Whether lowering recorded a response-restoration echo for this request, i.e.
/// [`ClientToolCompatFilter::restore_response`] will rewrite the buffered response.
fn restoration_armed(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .is_some_and(|state| state.client_tool_echo.is_some())
}

/// Collect the `Tool` definitions a client-executed `tool_search` returned in
/// prior `tool_search_output` history items, deduplicated by `(type, name)` so a
/// tool re-listed across searches is hoisted once.
///
/// Only outputs correlated to a client-executed `tool_search_call` this turn
/// lowers are considered (the same gate used to lower the output itself), so a
/// server/container-owned search never contributes discovered tools.
///
/// Only a terminal `tool_search_output` contributes tools: `status` is optional
/// in `ToolSearchOutputItemParam` (`anyOf[FunctionCallItemStatus, null]`; only
/// `type` and `tools` are required), so a **missing or `null`** status is treated
/// as `completed` — a client-supplied continuation that omits the status still
/// carries an authoritative loaded-tool list. A well-formed non-terminal status
/// (`in_progress` still searching, `incomplete` with no authoritative list) has no
/// authoritative tool list yet, so its tools stay as history context rather than
/// being hoisted into the callable set. A **malformed** status — a non-string value
/// or an unknown string — fails closed with a 400 rather than silently dropping the
/// tools the client believes it loaded. See [`tool_search_output_is_hoistable`].
///
/// Dedup and conflict detection compare the *lowered* projection of each listing
/// (see [`discovered_conflict_repr`]), not the raw `Value`: two listings of the
/// same `(type, name)` that reduce to the same backend declaration are hoisted
/// once even when they differ only in a decorator lowering drops (for example one
/// carries `defer_loading: true` and the other omits it). Fails closed only when
/// the same `(type, name)` is redefined across searches with definitions that lower
/// to genuinely different backend declarations: the correct declaration is then
/// ambiguous, so lowering it silently could hoist the wrong contract onto the
/// backend.
fn collect_discovered_tools(messages: &[Value]) -> Result<Vec<Value>, FilterAction> {
    let lowered_call_ids = client_executed_call_ids(messages);
    let mut seen: HashMap<(String, String), Value> = HashMap::new();
    let mut discovered = Vec::new();
    for item in messages {
        if item.get("type").and_then(Value::as_str) != Some("tool_search_output") {
            continue;
        }
        if !output_call_was_lowered(item, &lowered_call_ids) {
            continue;
        }
        if !tool_search_output_is_hoistable(item)? {
            continue;
        }
        if let Some(tools) = item.get("tools").and_then(Value::as_array) {
            merge_discovered_tools(tools, &mut seen, &mut discovered)?;
        }
    }
    Ok(discovered)
}

/// Whether a `tool_search_output` history item's status permits hoisting its
/// discovered tools, failing closed on a status the schema does not define.
///
/// `status` is optional in `ToolSearchOutputItemParam` (`anyOf[FunctionCallItemStatus,
/// null]`), so a missing or `null` status is treated as `completed`: the item is
/// authoritative and its tools are hoisted (`Ok(true)`). A well-formed non-terminal
/// status — `in_progress` (still searching) or `incomplete` (no authoritative list
/// yet) — is a deliberate client signal that the list is not yet loaded, so its tools
/// stay as history context rather than being hoisted (`Ok(false)`).
///
/// A status outside the `FunctionCallItemStatus` domain — a non-string value, or an
/// unknown string — is malformed input the proxy cannot interpret. Silently treating
/// it as non-terminal would strip a capability the client believes it loaded, so it
/// **fails closed** with a 400 instead, mirroring [`restore_call_status`], which
/// already rejects the same malformed shapes on the authoritative call-restoration
/// path.
fn tool_search_output_is_hoistable(item: &Value) -> Result<bool, FilterAction> {
    match item.get("status") {
        None | Some(Value::Null) => Ok(true),
        Some(Value::String(status)) => match status.as_str() {
            "completed" => Ok(true),
            "in_progress" | "incomplete" => Ok(false),
            other => Err(reject_bad_request(&format!(
                "tool_search_output has an unrecognized status '{other}'; expected in_progress, completed, or incomplete"
            ))),
        },
        Some(_) => Err(reject_bad_request(
            "tool_search_output status must be a string (in_progress, completed, or incomplete)",
        )),
    }
}

/// Merge one completed `tool_search_output`'s discovered `tools` into the running
/// dedup map and hoist list. Deduplicates a `(type, name)` re-listed with an
/// identical lowered projection and fails closed on a genuine conflict; see
/// [`collect_discovered_tools`] for the full semantics.
fn merge_discovered_tools(
    tools: &[Value],
    seen: &mut HashMap<(String, String), Value>,
    discovered: &mut Vec<Value>,
) -> Result<(), FilterAction> {
    for tool in tools {
        let key = (
            tool.get("type").and_then(Value::as_str).unwrap_or_default().to_owned(),
            tool.get("name").and_then(Value::as_str).unwrap_or_default().to_owned(),
        );
        let repr = discovered_conflict_repr(tool);
        match seen.get(&key) {
            // A re-listing that lowers to the same backend declaration is hoisted once.
            Some(existing) if *existing == repr => {},
            Some(_) => return Err(reject_conflicting_discovery(&key.0, &key.1)),
            None => {
                seen.insert(key, repr);
                discovered.push(tool.clone());
            },
        }
    }
    Ok(())
}

/// The canonical identity of a `tool_search`-discovered tool for dedup and
/// conflict detection.
///
/// Two discovered listings of the same `(type, name)` describe the same tool when
/// they resolve to the same lowering *outcome* — the same backend declaration, or
/// the same fail-closed rejection. For a discovered `function` the identity keeps
/// the fields [`sanitize_backend_function`] copies verbatim (`name`, `description`,
/// `parameters`, `strict`, including a `null` value, since `null` and absent lower
/// differently) and drops the pure Responses-only decorator `defer_loading`, so two
/// listings that differ only in a dropped decorator or a `null`/absent optional
/// reduce to an identical `Value` and are deduplicated rather than falsely rejected.
///
/// It also preserves the fields that *gate* `sanitize_backend_function`'s
/// fail-closed checks — any non-`null` `allowed_callers` (restrictive callers are
/// rejected) and any non-`null` `output_schema` (a non-object schema is rejected) —
/// so a clean listing and a would-reject twin of the same `(type, name)` never
/// reduce to the same identity. Without this, dedup keeps whichever listing came
/// first, and a stricter/malformed twin arriving second would be silently dropped
/// instead of failing closed (the outcome would depend on listing order). A
/// `null`/absent `allowed_callers` and a `null`/absent `output_schema` lower to
/// omission, so they are dropped and a pure decorator difference still dedups.
///
/// Every other discovered kind (`custom`, `namespace`) is lowered by a dedicated
/// path, so its identity is built by [`nonfunction_conflict_identity`] to mirror that
/// lowering outcome — dropping the fields lowering consumes or ignores among accepted
/// values while preserving every reject-gating value, so a would-reject twin can
/// never dedup behind a clean one.
fn discovered_conflict_repr(tool: &Value) -> Value {
    if tool.get("type").and_then(Value::as_str) == Some("function") {
        return function_conflict_identity(tool);
    }
    nonfunction_conflict_identity(tool)
}

/// Build the conflict identity for a discovered `function` (a top-level discovered
/// function or a `namespace` function member — both are lowered by the same
/// [`sanitize_backend_function`]), so two listings that lower identically dedup and a
/// would-reject twin fails closed regardless of listing order.
///
/// The identity is the whitelist `sanitize_backend_function` copies verbatim
/// (`type`, `name`, `description`, `parameters`, `strict`) plus the reject-gating
/// `allowed_callers`/`output_schema` *only when non-`null`*: a `null`/absent value
/// lowers to omission (so it is dropped and a pure-decorator difference still dedups),
/// while a non-`null` value — a restrictive `allowed_callers`, or a malformed
/// non-object `output_schema` — is kept so a clean listing and a would-reject twin
/// never reduce to the same identity. Every field outside the whitelist (a pure
/// Responses-only `defer_loading` hint, or any unknown extra) is dropped exactly as
/// lowering drops it, so it never spuriously distinguishes two listings.
fn function_conflict_identity(tool: &Value) -> Value {
    let mut repr = Map::new();
    repr.insert("type".to_owned(), json!("function"));
    for field in ["name", "description", "parameters", "strict"] {
        if let Some(value) = tool.get(field) {
            repr.insert(field.to_owned(), value.clone());
        }
    }
    for field in ["allowed_callers", "output_schema"] {
        if let Some(value) = tool.get(field).filter(|value| !value.is_null()) {
            repr.insert(field.to_owned(), value.clone());
        }
    }
    Value::Object(repr)
}

/// Build the conflict identity for a discovered non-`function` tool (a `custom`,
/// `shell`, `tool_search`, or `namespace`), reflecting its lowering *outcome* so two
/// listings that lower identically dedup instead of falsely conflicting, while a
/// would-reject twin still fails closed regardless of listing order.
///
/// The top-level tool is normalized by [`normalize_conflict_identity_fields`]. Each
/// `namespace` member is reduced by its own lowering path: a `function` member uses
/// [`function_conflict_identity`] (it lowers through [`sanitize_backend_function`],
/// exactly like a top-level function), and any other member (a `custom`) uses
/// [`normalize_conflict_identity_fields`] (it lowers through a prose-folding path that
/// keeps most fields).
fn nonfunction_conflict_identity(tool: &Value) -> Value {
    let Value::Object(map) = tool else {
        return tool.clone();
    };
    let mut map = map.clone();
    normalize_conflict_identity_fields(&mut map);
    if let Some(Value::Array(members)) = map.get_mut("tools") {
        for member in members.iter_mut() {
            *member = member_conflict_identity(member);
        }
    }
    Value::Object(map)
}

/// Reduce one `namespace` member to its conflict identity via the lowering path that
/// member kind takes: a `function` member through [`function_conflict_identity`]
/// (whitelist, matching [`sanitize_backend_function`]); any other member through
/// [`normalize_conflict_identity_fields`] (prose-folding, matching the `custom`
/// member path). A non-object member is returned verbatim.
///
/// A `function` member differs from a top-level discovered function in one respect:
/// [`Lowering::lower_namespace_member`] OVERWRITES the sanitized description with the
/// trimmed, namespace-folded value (`null`/absent coerced to `""`), so the member's
/// raw description only matters as its trimmed contribution. The whitelist identity is
/// therefore canonicalized with [`folded_description`] so two members whose
/// descriptions lower identically (`null` vs absent, or a trailing-space delta) dedup
/// instead of falsely conflicting, while a materially different description still
/// distinguishes them.
fn member_conflict_identity(member: &Value) -> Value {
    if member.get("type").and_then(Value::as_str) == Some("function") {
        let mut repr = function_conflict_identity(member);
        if let Value::Object(map) = &mut repr {
            map.insert(
                "description".to_owned(),
                json!(folded_description(member.get("description"))),
            );
        }
        return repr;
    }
    let Value::Object(map) = member else {
        return member.clone();
    };
    let mut map = map.clone();
    normalize_conflict_identity_fields(&mut map);
    Value::Object(map)
}

/// Drop from a non-`function` tool/member object the fields lowering consumes or
/// ignores among *accepted* values, so its conflict identity matches its lowering
/// outcome. Each drop-condition mirrors the corresponding accept-condition exactly, so
/// a reject-worthy value stays in the identity and a would-reject twin can never dedup
/// behind a clean one (the fail-closed guarantee stays independent of listing order):
///
/// - `defer_loading` is a pure Responses-only hint that lowering consumes uniformly on the discovery path, so it never
///   distinguishes two listings.
/// - a `null`/absent `allowed_callers` is accepted and lowers to omission (only a non-`null` restriction fails closed,
///   per [`reject_if_restricted_callers`]), so it is dropped while a non-`null` restriction is kept verbatim.
/// - an *accepted* custom `format` (absent or `{"type":"text"}`) is only validated, never read afterwards
///   ([`custom_extra_fields`] excludes it), so it lowers identically and is dropped; a reject-worthy `format` is kept.
///   Every other custom field (including `output_schema`) is folded into the lowered function's prose, so it stays.
/// - the `description` is canonicalized per kind by [`normalize_identity_description`] to mirror how that kind's
///   lowering folds its prose (`custom` trims and drops-when-empty, `namespace` trims, `shell` ignores it,
///   `tool_search` keeps it verbatim), so two listings whose descriptions lower identically dedup.
fn normalize_conflict_identity_fields(map: &mut Map<String, Value>) {
    map.remove("defer_loading");
    if map.get("allowed_callers").is_none_or(Value::is_null) {
        map.remove("allowed_callers");
    }
    if map.get("type").and_then(Value::as_str) == Some("custom") && custom_format_is_accepted(map) {
        map.remove("format");
    }
    normalize_identity_description(map);
}

/// Whether a custom tool object's `format` is one lowering accepts (absent, or an
/// object whose `type` is `"text"`), mirroring [`reject_unsupported_custom_declaration`].
fn custom_format_is_accepted(map: &Map<String, Value>) -> bool {
    match map.get("format") {
        None => true,
        Some(format) => format.get("type").and_then(Value::as_str) == Some("text"),
    }
}

/// The trimmed description string that a prose-folding lowering path keeps: `null`,
/// absent, a non-string, and a whitespace-only value all collapse to `""`, mirroring
/// the `.and_then(Value::as_str).unwrap_or_default().trim()` / `.map(str::trim)`
/// normalization those paths apply before folding a description into a member's or a
/// lowered custom's model-visible text.
fn folded_description(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().trim().to_owned()
}

/// Canonicalize a non-`function` tool/member's `description` in its conflict identity
/// to mirror how that kind's lowering folds it, so two listings that lower to the same
/// model-visible text dedup instead of falsely conflicting (and materially different
/// descriptions still conflict). Each kind normalizes differently, so this must not be
/// applied uniformly:
///
/// - a `custom` tool/member folds a *trimmed, dropped-when-empty* description into its prose
///   ([`custom_model_visible_description`]), so `null`/absent/empty/whitespace collapse to omission and a
///   trailing-space delta collapses to the trimmed value.
/// - a `namespace` tool *trims* its (required) description before folding it into every member ([`namespace_header`]),
///   so a trailing-space delta collapses.
/// - a `shell` tool *ignores* its `description` entirely (only `environment.skills` is folded, per
///   [`shell_model_visible_description`]), so it never distinguishes two listings.
/// - a `tool_search` tool uses its `description` *verbatim* (or a default when empty, per
///   [`Lowering::lower_tool_search`]), so it is left untouched — trimming it would collapse two listings that genuinely
///   lower to different text (a wrong dedup).
fn normalize_identity_description(map: &mut Map<String, Value>) {
    match map.get("type").and_then(Value::as_str) {
        Some("custom") => {
            let folded = folded_description(map.get("description"));
            if folded.is_empty() {
                map.remove("description");
            } else {
                map.insert("description".to_owned(), json!(folded));
            }
        },
        Some("namespace") => {
            map.insert(
                "description".to_owned(),
                json!(folded_description(map.get("description"))),
            );
        },
        Some("shell") => {
            map.remove("description");
        },
        _ => {},
    }
}

/// Lower any prior typed client-owned history items, marking the body for rebuild
/// when a continuation item changed. Used on the native-passthrough paths; fails
/// closed on an unsupported prior history item even when the current request
/// declares no rich client tools, and enforces the configured rewrite cap on the
/// rebuilt body when history lowering expanded it.
fn lower_history_only(state: &mut ResponsesState, max_rewritten_body_bytes: usize) -> Result<(), FilterAction> {
    if lower_history_items(&mut state.messages)? {
        enforce_history_rewrite_cap(state, max_rewritten_body_bytes)?;
        state.mark_request_body_for_rebuild();
    }
    Ok(())
}

/// Enforce the operator-selected rewrite cap on the fully rebuilt outbound body
/// after lowering only the conversation history (the native-passthrough paths).
///
/// History lowering escapes typed client-owned items into JSON-string arguments,
/// which can expand the outbound body past a configured cap smaller than the
/// proxy's own 64 MiB limit. Fail closed before any upstream call when the rebuilt
/// body exceeds the cap or cannot be measured, so a history-only rewrite honours
/// the same limit as the rich-tool path. Like `enforce_rewrite_cap`, the mutated
/// history is not rolled back: the request is rejected before any upstream call,
/// so the mutated state is discarded rather than sent.
fn enforce_history_rewrite_cap(state: &ResponsesState, max_rewritten_body_bytes: usize) -> Result<(), FilterAction> {
    let outbound_len = super::openai_responses_proxy::serialized_outbound_body_len(state).map_err(|error| {
        FilterAction::Reject(responses_error_rejection(
            500,
            "server_error",
            &format!("failed to measure rewritten request body: {error}"),
        ))
    })?;
    if outbound_len > max_rewritten_body_bytes {
        return Err(reject_rewritten_body_too_large(outbound_len, max_rewritten_body_bytes));
    }
    Ok(())
}

/// Return whether a declared tool is a rich client-owned tool this filter lowers.
fn is_rich_client_tool(tool: &Value) -> bool {
    match tool.get("type").and_then(Value::as_str) {
        // `local_shell` is a distinct client-owned declaration this filter cannot
        // yet restore; it is recognized here so it trips the streaming guard and
        // the fail-closed lowering path rather than being forwarded unchanged.
        Some("custom" | "namespace" | "local_shell") => true,
        Some("shell") => shell_is_local(tool),
        Some("tool_search") => tool_search_is_client(tool),
        _ => false,
    }
}

/// Return whether a `shell` tool targets the caller's local environment.
fn shell_is_local(tool: &Value) -> bool {
    tool.get("environment")
        .and_then(|environment| environment.get("type"))
        .and_then(Value::as_str)
        == Some("local")
}

/// Return whether a `tool_search` tool is client-executed (the default).
fn tool_search_is_client(tool: &Value) -> bool {
    matches!(tool.get("execution").and_then(Value::as_str), None | Some("client"))
}

/// Return whether a declared tool defers loading its full definition
/// (`defer_loading: true`), meaning it is not callable until a `tool_search`
/// discovers and hoists it into the callable set. The Responses schema permits
/// `defer_loading` on `function` and `custom` tools and on `namespace` members.
fn is_deferred_declaration(tool: &Value) -> bool {
    tool.get("defer_loading").and_then(Value::as_bool) == Some(true)
}

/// Reduce a `function` declaration — a hoisted discovered tool or a lowered
/// `namespace` member — to the fields a function-only backend accepts, stripping
/// the Responses-only decorators that never reach a single-shot backend.
///
/// Whitelists the callable essence (`type`/`name`/`description`/`parameters`/
/// `strict`/`output_schema`) and drops `defer_loading` (a lazy-load hint with no
/// meaning to a function-only backend). `output_schema` is preserved because the
/// backend delegates Responses tools to the OpenAI SDK `Tool` type, which carries
/// it; a `null` schema is dropped as equivalent to omission. Missing required
/// fields are left absent so the backend performs its own schema validation.
///
/// Fails closed rather than silently altering behavior when the declaration
/// carries a restrictive `allowed_callers` (which lowering would otherwise widen
/// to all callers) or a malformed non-object, non-null `output_schema`.
/// `descriptor` names the offending declaration for the rejection message.
fn sanitize_backend_function(tool: &Value, descriptor: &str) -> Result<Value, FilterAction> {
    reject_if_restricted_callers(tool, descriptor)?;
    let mut sanitized = Map::new();
    sanitized.insert("type".to_owned(), json!("function"));
    for field in ["name", "description", "parameters", "strict"] {
        if let Some(value) = tool.get(field) {
            sanitized.insert(field.to_owned(), value.clone());
        }
    }
    if let Some(output_schema) = tool.get("output_schema") {
        if !output_schema.is_null() && !output_schema.is_object() {
            return Err(reject_bad_request(&format!(
                "{descriptor} declares an output_schema that is not a JSON object or null"
            )));
        }
        if output_schema.is_object() {
            sanitized.insert("output_schema".to_owned(), output_schema.clone());
        }
    }
    Ok(Value::Object(sanitized))
}

/// The origin of a claimed function name, used for collision detection.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NameOrigin {
    /// A private function name synthesized by lowering a client tool.
    Lowered,
    /// A `function` tool the client declared directly.
    Passthrough,
}

/// Where a tool being lowered came from, which decides how `defer_loading` is
/// handled.
///
/// A `defer_loading: true` tool is, per the Responses spec, "deferred and
/// discovered via tool search": it is not loaded into the callable set up front.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoweringSource {
    /// The tool appears in the live request's `tools` declaration. A deferred tool
    /// is withheld from the outbound set until a `tool_search` discovers it.
    Declaration,
    /// The tool was hoisted from a prior `tool_search_output`. It is being loaded
    /// now, so its `defer_loading` hint is consumed and it becomes callable.
    Discovery,
}

/// The decomposed outcome of a completed [`Lowering`], consulted by the
/// `tool_choice` translation and the response-restoration phase.
struct LoweringParts {
    /// Reverse map from a lowered function name to its restoration recipe. Moved
    /// into `ResponsesState::client_tool_lowering` after `tool_choice` lowering.
    reverse: HashMap<String, LoweredClientTool>,
    /// Every function name callable on the outbound wire (lowered rich tools and
    /// passthrough/discovered functions), used to tell a declared-deferred tool
    /// that discovery re-loaded from one that remains dangling.
    callable: HashSet<String>,
    /// Wire names withheld from the outbound set (declared deferred, not yet
    /// discovered), so a forced selector for one fails closed as deferred.
    withheld: HashSet<String>,
}

/// Accumulates the lowered `tools` array and the reverse restoration map while
/// enforcing name uniqueness and the per-request tool cap.
struct Lowering {
    /// Maximum number of lowered client tools permitted this request.
    max_client_tools: usize,
    /// Every claimed function name and its origin, for collision detection.
    occupied: HashMap<String, NameOrigin>,
    /// Reverse map from a lowered function name to its restoration recipe.
    reverse: HashMap<String, LoweredClientTool>,
    /// Whether at least one rich client tool was actually lowered.
    lowered_any: bool,
    /// Whether at least one deferred declaration was withheld from the outbound
    /// set. When every rich tool is deferred, this distinguishes "nothing was
    /// lowerable" (safe to pass the originals through) from "rich tools were
    /// withheld" (the un-lowerable originals must not reach the backend).
    withheld_any: bool,
    /// Count of deferred declarations withheld from the outbound set (top-level
    /// and namespace members, named or not). Counted toward `max_client_tools`
    /// alongside the lowered reverse map: a withheld tool is removed from the
    /// outbound body yet still echoed back in the pre-lowering snapshot, so it
    /// escapes the outbound body-size cap and a request with an enormous all-deferred
    /// `tools` array must not grow request-scoped state past the cap. (A plain
    /// passthrough `function` stays in the outbound body, so it is bounded by the
    /// body-size cap instead and is not counted here; see [`Self::enforce_tool_cap`].)
    /// When a `tool_search` later discovers and lowers a withheld tool (same wire
    /// name), [`Self::reclaim_withheld`] releases its count so the one logical tool
    /// is not counted twice.
    withheld_count: usize,
    /// Wire names of the deferred declarations withheld from the outbound set, so a
    /// `tool_choice` forcing a still-deferred tool fails closed with a deferred-
    /// specific rejection instead of dangling a selector the backend cannot honor.
    withheld: HashSet<String>,
}

impl Lowering {
    /// Create an empty accumulator bounded by `max_client_tools`.
    fn new(max_client_tools: usize) -> Self {
        Self {
            max_client_tools,
            occupied: HashMap::new(),
            reverse: HashMap::new(),
            lowered_any: false,
            withheld_any: false,
            withheld_count: 0,
            withheld: HashSet::new(),
        }
    }

    /// Record that a deferred declaration was withheld from the outbound callable
    /// set. Tracks the tool's outbound wire name when present (a top-level tool's
    /// own name, or a namespace member's flat name) so a `tool_choice` that forces
    /// a still-deferred tool can fail closed with a deferred-specific rejection.
    ///
    /// Counts the withheld tool toward `max_client_tools`: it is echoed back in the
    /// pre-lowering snapshot, so an all-deferred `tools` array must not bypass the
    /// cap. Fails closed once the lowered-plus-withheld total exceeds the cap.
    fn withhold(&mut self, wire_name: Option<&str>) -> Result<(), FilterAction> {
        self.withheld_any = true;
        self.withheld_count += 1;
        if let Some(name) = wire_name.filter(|name| !name.is_empty()) {
            self.withheld.insert(name.to_owned());
        }
        self.enforce_tool_cap()
    }

    /// Fail closed once the combined count of *lowered* and *withheld* client tools
    /// exceeds `max_client_tools` (`reverse.len() + withheld_count`). The cap
    /// deliberately bounds only the request-scoped state that the outbound
    /// body-size cap ([`enforce_rewrite_cap`]) does not already bound:
    ///
    /// - a **lowered** tool carries a reverse restoration recipe — per-request state beyond the outbound body — so it
    ///   is counted;
    /// - a **withheld** deferred tool is removed from the outbound body yet still echoed back in the pre-lowering
    ///   snapshot, so it escapes the body-size cap and is counted here instead.
    ///
    /// A plain `function` forwarded through unchanged (passthrough) is **not** counted:
    /// it stays in the outbound body verbatim, so `max_rewritten_body_bytes` already
    /// bounds it, exactly as the native passthrough path (a `tools` array of plain
    /// functions with no rich tool) applies no tool-count cap at all. Counting
    /// passthrough here would reject a large plain-function set only when a rich tool
    /// happens to be co-declared, contradicting the documented "lowered" semantic of
    /// `max_client_tools`.
    fn enforce_tool_cap(&self) -> Result<(), FilterAction> {
        if self.reverse.len() + self.withheld_count > self.max_client_tools {
            return Err(reject_too_many_tools(self.max_client_tools));
        }
        Ok(())
    }

    /// Consume the accumulator into the maps the `tool_choice` and response phases
    /// consult: the reverse restoration map, every callable outbound function name,
    /// and the still-deferred withheld names.
    fn into_parts(self) -> LoweringParts {
        LoweringParts {
            callable: self.occupied.into_keys().collect(),
            reverse: self.reverse,
            withheld: self.withheld,
        }
    }

    /// Lower every declared tool, returning the rewritten `tools` array alongside
    /// whether at least one rich client tool was actually lowered.
    ///
    /// The array is returned even when nothing rich was lowered (every tool copied
    /// through verbatim) so the caller can hoist discovered tools onto it without a
    /// second clone of the originals.
    fn lower_tools(&mut self, tools: &[Value]) -> Result<(Vec<Value>, bool), FilterAction> {
        let mut lowered = Vec::with_capacity(tools.len());
        for tool in tools {
            // A top-level `function`/`custom` may not usurp the reserved namespace
            // wire-name prefix; fail closed before it is withheld or lowered so its
            // name cannot collide with a synthesized namespace member.
            reject_reserved_top_level_name(tool)?;
            // A deferred declaration (`function` or `custom`) is not callable until
            // a `tool_search` loads it; withhold it from the outbound set so it
            // neither forwards a Responses-only `defer_loading` semantic a
            // function-only backend cannot honor, nor collides with its own hoisted
            // discovered copy. It becomes callable via `lower_discovered_tools` once
            // discovered. (`namespace` withholds per-member; see `lower_namespace`.)
            if matches!(tool.get("type").and_then(Value::as_str), Some("function" | "custom"))
                && is_deferred_declaration(tool)
            {
                self.withhold(tool.get("name").and_then(Value::as_str))?;
                continue;
            }
            match tool.get("type").and_then(Value::as_str) {
                Some("custom") => self.lower_custom(tool, &mut lowered, LoweringSource::Declaration)?,
                Some("shell") if shell_is_local(tool) => self.lower_shell(tool, &mut lowered)?,
                Some("tool_search") if tool_search_is_client(tool) => self.lower_tool_search(tool, &mut lowered)?,
                Some("namespace") => self.lower_namespace(tool, &mut lowered, LoweringSource::Declaration)?,
                // `local_shell` restores to a distinct `local_shell_call` output
                // shape that is not yet implemented; fail closed before any
                // upstream call rather than forward an unsupported tool.
                Some("local_shell") => return Err(reject_local_shell_unsupported()),
                other => {
                    if other == Some("function")
                        && let Some(name) = tool.get("name").and_then(Value::as_str)
                    {
                        self.record_passthrough(name)?;
                    }
                    lowered.push(tool.clone());
                },
            }
        }
        Ok((lowered, self.lowered_any))
    }

    /// Lower the tool definitions a client-executed `tool_search` returned in prior
    /// history so a function-only backend can invoke them on the continuation turn.
    ///
    /// Each discovered `Tool` is lowered through the same accumulator as the
    /// declared tools — rich types (`custom`/`namespace`/local `shell`/client
    /// `tool_search`) gain restoration entries, plain `function` tools pass through
    /// — so the model can call a discovered tool and its returned `function_call`
    /// restores to the correct typed item. Fails closed on a name collision with an
    /// already-claimed tool, an unsupported discovered type, or more than
    /// `max_client_tools` discovered definitions.
    fn lower_discovered_tools(&mut self, discovered: &[Value], lowered: &mut Vec<Value>) -> Result<(), FilterAction> {
        if discovered.len() > self.max_client_tools {
            return Err(reject_too_many_tools(self.max_client_tools));
        }
        for tool in discovered {
            // A discovered top-level `function`/`custom` is subject to the same
            // reserved-prefix reservation as a declared one, so a hoisted tool cannot
            // impersonate a synthesized namespace member wire name either.
            reject_reserved_top_level_name(tool)?;
            match tool.get("type").and_then(Value::as_str) {
                Some("custom") => self.lower_custom(tool, lowered, LoweringSource::Discovery)?,
                Some("shell") if shell_is_local(tool) => self.lower_shell(tool, lowered)?,
                Some("tool_search") if tool_search_is_client(tool) => self.lower_tool_search(tool, lowered)?,
                Some("namespace") => self.lower_namespace(tool, lowered, LoweringSource::Discovery)?,
                Some("local_shell") => return Err(reject_local_shell_unsupported()),
                Some("function") => self.lower_discovered_function(tool, lowered)?,
                other => {
                    return Err(reject_bad_request(&format!(
                        "tool_search returned an unsupported '{}' tool that cannot be made callable on a \
                         function-only backend",
                        other.unwrap_or("<missing>"),
                    )));
                },
            }
        }
        Ok(())
    }

    /// Lower a plain `function` a `tool_search` discovered into a backend-callable
    /// declaration and claim its name for collision detection.
    ///
    /// Sanitizes the discovered declaration to the fields a function-only backend
    /// accepts (see [`sanitize_backend_function`]): drops the Responses-only
    /// `defer_loading` hint (the tool is being loaded now) and fails closed on a
    /// restrictive `allowed_callers` or malformed `output_schema` rather than
    /// forwarding them verbatim. A discovered `function` without a non-empty name
    /// also fails closed.
    fn lower_discovered_function(&mut self, tool: &Value, lowered: &mut Vec<Value>) -> Result<(), FilterAction> {
        let Some(name) = tool.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()) else {
            return Err(reject_bad_request(
                "tool_search returned a function tool without a non-empty name",
            ));
        };
        let sanitized = sanitize_backend_function(tool, &format!("tool_search discovered function '{name}'"))?;
        self.claim_discovered_passthrough(name)?;
        lowered.push(sanitized);
        Ok(())
    }

    /// Record a client-declared `function` name for collision detection.
    fn record_passthrough(&mut self, name: &str) -> Result<(), FilterAction> {
        match self.occupied.get(name) {
            Some(NameOrigin::Lowered) => Err(reject_collision(name)),
            Some(NameOrigin::Passthrough) => Ok(()),
            None => {
                self.occupied.insert(name.to_owned(), NameOrigin::Passthrough);
                Ok(())
            },
        }
    }

    /// Claim a synthesized lowered function name, validating charset and uniqueness.
    fn claim_lowered(&mut self, name: &str) -> Result<(), FilterAction> {
        if !is_valid_function_name(name) {
            return Err(reject_invalid_name(name));
        }
        if self.occupied.contains_key(name) {
            return Err(reject_collision(name));
        }
        self.occupied.insert(name.to_owned(), NameOrigin::Lowered);
        Ok(())
    }

    /// Claim a discovered `function` tool's name, failing closed on any collision
    /// so a hoisted discovery never duplicates or shadows an already-claimed tool
    /// (declared passthrough or lowered private name).
    fn claim_discovered_passthrough(&mut self, name: &str) -> Result<(), FilterAction> {
        if !is_valid_function_name(name) {
            return Err(reject_invalid_name(name));
        }
        if self.occupied.contains_key(name) {
            return Err(reject_collision(name));
        }
        self.occupied.insert(name.to_owned(), NameOrigin::Passthrough);
        Ok(())
    }

    /// Register one lowered tool in the reverse map and enforce the tool cap.
    fn register(&mut self, name: String, restore: LoweredClientTool) -> Result<(), FilterAction> {
        self.reverse.insert(name, restore);
        self.lowered_any = true;
        self.enforce_tool_cap()
    }

    /// Reclaim the withheld budget of a deferred declaration that a `tool_search` is
    /// now loading, so the one logical tool counts once toward `max_client_tools`
    /// rather than twice — once as withheld, once as lowered.
    ///
    /// Only the [`LoweringSource::Discovery`] path reloads a withheld declaration:
    /// the discovered copy is the same logical tool the client declared with
    /// `defer_loading: true`, matched by its shared outbound wire name. On the
    /// [`LoweringSource::Declaration`] path a registration is a *distinct* tool that
    /// merely happens to share a wire name with an earlier withheld one (for example
    /// a local `shell` lowered to the fixed `shell` name alongside a deferred `custom`
    /// the client named `shell`), so it must not credit the withheld budget — doing
    /// so would let two distinct tools slip past the cap as one. Callers therefore
    /// pass their lowering source, and reclaim is a no-op on the declaration path and
    /// for any name that was never withheld.
    ///
    /// Matching a reload by wire name alone is sound only because wire names are
    /// unique per logical tool. Two reservations guarantee that. Across the
    /// top-level/namespace boundary, a synthesized namespace member name always
    /// carries the reserved [`NAMESPACE_MEMBER_PREFIX`], and
    /// [`reject_reserved_top_level_name`] fails closed on any top-level
    /// `function`/`custom` that usurps it. Within the namespace flattening,
    /// [`reject_reserved_namespace_delimiter`] fails closed on a namespace or member
    /// name embedding the `__` delimiter or abutting it with a leading/trailing `_`,
    /// keeping the verbatim [`namespace_member_name`] form injective, and its
    /// hash-truncation branch is domain-separated with a `___` marker so a hashed
    /// name can never equal a verbatim one — together, two distinct members can never
    /// flatten to one wire name. Without these reservations a genuinely distinct
    /// deferred tool and a discovered tool could share a wire name, and this reclaim
    /// would credit one against the other.
    fn reclaim_withheld(&mut self, source: LoweringSource, name: &str) {
        if source == LoweringSource::Discovery && self.withheld.remove(name) {
            self.withheld_count = self.withheld_count.saturating_sub(1);
        }
    }

    /// Lower a `custom` tool to a `function` with a single string `input`.
    ///
    /// `source` distinguishes a declared custom from one a `tool_search` discovered:
    /// a discovered custom reloads a same-named deferred declaration, so its withheld
    /// budget is reclaimed (see [`Self::reclaim_withheld`]).
    fn lower_custom(
        &mut self,
        tool: &Value,
        lowered: &mut Vec<Value>,
        source: LoweringSource,
    ) -> Result<(), FilterAction> {
        let Some(name) = tool.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()) else {
            return Err(reject_bad_request("custom tool requires a non-empty name"));
        };
        reject_unsupported_custom_declaration(tool, &format!("custom tool '{name}'"))?;
        let name = name.to_owned();
        self.claim_lowered(&name)?;
        self.reclaim_withheld(source, &name);
        lowered.push(custom_lowered_function(&name, &custom_model_visible_description(tool)));
        self.register(
            name.clone(),
            LoweredClientTool {
                original_name: name,
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        )
    }

    /// Lower a local `shell` tool to the `shell` `function`, folding any declared
    /// local skills into the model-visible description.
    fn lower_shell(&mut self, tool: &Value, lowered: &mut Vec<Value>) -> Result<(), FilterAction> {
        reject_if_restricted_callers(tool, "shell tool")?;
        let description = shell_model_visible_description(tool)?;
        self.claim_lowered(SHELL_FUNCTION_NAME)?;
        lowered.push(shell_lowered_function(&description));
        self.register(
            SHELL_FUNCTION_NAME.to_owned(),
            LoweredClientTool {
                original_name: SHELL_FUNCTION_NAME.to_owned(),
                namespace: None,
                restore: ClientToolRestore::Shell,
            },
        )
    }

    /// Lower a client-executed `tool_search` tool to the fixed `tool_search`
    /// `function`.
    fn lower_tool_search(&mut self, tool: &Value, lowered: &mut Vec<Value>) -> Result<(), FilterAction> {
        // `ToolSearchToolParam.parameters` is `object | null`: an explicit null is
        // valid and treated identically to omission (the generated default schema).
        if tool
            .get("parameters")
            .is_some_and(|parameters| !parameters.is_null() && !parameters.is_object())
        {
            return Err(reject_bad_request(
                "tool_search parameters must be a JSON object or null for private function lowering",
            ));
        }
        self.claim_lowered(TOOL_SEARCH_NAME)?;
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .filter(|description| !description.is_empty())
            .unwrap_or(TOOL_SEARCH_DEFAULT_DESCRIPTION)
            .to_owned();
        let parameters = tool
            .get("parameters")
            .filter(|parameters| parameters.is_object())
            .cloned()
            .unwrap_or_else(default_tool_search_parameters);
        lowered.push(tool_search_lowered_function(&description, &parameters));
        self.register(
            TOOL_SEARCH_NAME.to_owned(),
            LoweredClientTool {
                original_name: TOOL_SEARCH_NAME.to_owned(),
                namespace: None,
                restore: ClientToolRestore::ToolSearch,
            },
        )
    }

    /// Lower each member of a `namespace` tool to a flat `function`.
    ///
    /// `NamespaceToolParam` itself carries no `defer_loading`, but each member may:
    /// on the [`LoweringSource::Declaration`] path a deferred member (function or
    /// custom) is withheld from the outbound set until a `tool_search` discovers it,
    /// while on the [`LoweringSource::Discovery`] path the member is being loaded now
    /// so its `defer_loading` hint is consumed and it becomes callable.
    fn lower_namespace(
        &mut self,
        tool: &Value,
        lowered: &mut Vec<Value>,
        source: LoweringSource,
    ) -> Result<(), FilterAction> {
        let (namespace, members) = namespace_header(tool)?;
        for member in members {
            // A member name embedding the reserved `__` delimiter would make its
            // flattened wire name ambiguous with a distinct namespace/member pair;
            // fail closed before it is withheld or lowered so the flattening stays
            // injective on the deferred path too (see the callable-collision guard in
            // `claim_lowered`). Members without a name are rejected in the helpers.
            if let Some(member_name) = member
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                reject_reserved_namespace_delimiter("member", member_name)?;
            }
            if source == LoweringSource::Declaration && is_deferred_declaration(member) {
                // A member's outbound wire name is its flattened `namespace/member`
                // name, so a forced namespaced selector for it fails closed as
                // deferred rather than dangling.
                let flat = member
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .map(|name| namespace_member_name(namespace.name, name));
                self.withhold(flat.as_deref())?;
                continue;
            }
            // `NamespaceToolParam.tools` permits function and custom members; both
            // are lowered. Fail closed on any other member type so it is never
            // silently dropped or forwarded intact to a function-only backend.
            match member.get("type").and_then(Value::as_str) {
                Some("function") => {
                    self.lower_namespace_member(namespace, member, lowered, source)?;
                },
                Some("custom") => {
                    self.lower_namespace_custom_member(namespace, member, lowered, source)?;
                },
                other => return Err(reject_unsupported_namespace_member(namespace.name, other)),
            }
        }
        Ok(())
    }

    /// Lower one `function` member of a `namespace` tool to a flat `function`,
    /// folding the namespace's model-visible description into the member.
    fn lower_namespace_member(
        &mut self,
        namespace: NamespaceContext<'_>,
        member: &Value,
        lowered: &mut Vec<Value>,
        source: LoweringSource,
    ) -> Result<(), FilterAction> {
        let Some(member_name) = member
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            return Err(reject_bad_request("namespace member requires a non-empty name"));
        };
        // Sanitize the member to the fields a function-only backend accepts before
        // folding in the flat name and namespace context: drop the Responses-only
        // `defer_loading` hint (a deferred declaration member is withheld earlier by
        // `lower_namespace`; a discovered member is being loaded now, so consuming
        // the hint makes it callable) and fail closed on a restrictive
        // `allowed_callers` or malformed `output_schema` rather than forwarding them
        // verbatim.
        let mut function = sanitize_backend_function(
            member,
            &format!("namespace '{}' member '{member_name}'", namespace.name),
        )?;
        let flat = namespace_member_name(namespace.name, member_name);
        self.claim_lowered(&flat)?;
        self.reclaim_withheld(source, &flat);
        let member_description = member.get("description").and_then(Value::as_str).unwrap_or_default();
        let description = prepend_namespace_context(namespace.name, namespace.description, member_description);
        insert_field(&mut function, "type", json!("function"));
        insert_field(&mut function, "name", json!(flat.clone()));
        insert_field(&mut function, "description", json!(description));
        lowered.push(function);
        self.register(
            flat,
            LoweredClientTool {
                original_name: member_name.to_owned(),
                namespace: Some(namespace.name.to_owned()),
                restore: ClientToolRestore::Namespace,
            },
        )
    }

    /// Lower one `custom` member of a `namespace` tool to a flat `function`
    /// carrying the freeform single-string `input` contract, restored to a
    /// namespaced `custom_tool_call`. The namespace's model-visible description is
    /// folded into the lowered member so the model keeps the group context.
    fn lower_namespace_custom_member(
        &mut self,
        namespace: NamespaceContext<'_>,
        member: &Value,
        lowered: &mut Vec<Value>,
        source: LoweringSource,
    ) -> Result<(), FilterAction> {
        let Some(member_name) = member
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            return Err(reject_bad_request("namespace member requires a non-empty name"));
        };
        reject_unsupported_custom_declaration(
            member,
            &format!("namespace '{}' member '{member_name}'", namespace.name),
        )?;
        let flat = namespace_member_name(namespace.name, member_name);
        self.claim_lowered(&flat)?;
        self.reclaim_withheld(source, &flat);
        let description = prepend_namespace_context(
            namespace.name,
            namespace.description,
            &custom_model_visible_description(member),
        );
        lowered.push(custom_lowered_function(&flat, &description));
        self.register(
            flat,
            LoweredClientTool {
                original_name: member_name.to_owned(),
                namespace: Some(namespace.name.to_owned()),
                restore: ClientToolRestore::NamespaceCustom,
            },
        )
    }
}

/// Model-visible context for a `namespace` group, folded into each flattened
/// member so the group's contract survives the flattening. Bundling the name and
/// description keeps the member-lowering helpers to a single context argument.
#[derive(Clone, Copy)]
struct NamespaceContext<'a> {
    /// The namespace's non-empty `name`, prefixed to each member's flat wire name.
    name: &'a str,
    /// The namespace's required model-visible `description`, folded into each member.
    description: &'a str,
}

/// Validate a `namespace` declaration's required fields, returning its
/// model-visible context and members.
///
/// `NamespaceToolParam` requires a non-empty `name`, a non-empty `description`
/// (`minLength: 1`, shown to the model), and a non-empty `tools` array
/// (`minItems: 1`). Each is schema-invalid when missing, so a request that omits
/// one fails closed rather than lowering members that drop the namespace's
/// model-visible contract or silently dropping the whole declaration.
fn namespace_header(tool: &Value) -> Result<(NamespaceContext<'_>, &[Value]), FilterAction> {
    let Some(name) = tool.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()) else {
        return Err(reject_bad_request("namespace tool requires a non-empty name"));
    };
    reject_reserved_namespace_delimiter("group", name)?;
    let Some(description) = tool
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
    else {
        return Err(reject_bad_request(&format!(
            "namespace '{name}' requires a non-empty description"
        )));
    };
    let Some(members) = tool
        .get("tools")
        .and_then(Value::as_array)
        .filter(|members| !members.is_empty())
    else {
        return Err(reject_bad_request(&format!(
            "namespace '{name}' requires a non-empty tools array"
        )));
    };
    Ok((NamespaceContext { name, description }, members.as_slice()))
}

/// Prepend a namespace's model-visible context to a flattened member description.
///
/// A flattened member loses its `namespace` grouping on the wire, so the
/// namespace's required description is folded into each member's description to
/// keep the model-visible tool contract intact. An empty member body yields just
/// the namespace context.
fn prepend_namespace_context(namespace: &str, namespace_description: &str, body: &str) -> String {
    let prefix = format!("Belongs to the `{namespace}` tool namespace: {namespace_description}");
    let body = body.trim();
    if body.is_empty() {
        prefix
    } else {
        format!("{prefix}\n\n{body}")
    }
}

/// Build the model-visible description for a lowered `custom` tool.
fn custom_model_visible_description(tool: &Value) -> String {
    let mut fragments: Vec<String> = Vec::new();
    if let Some(description) = tool
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        fragments.push(description.to_owned());
    }
    fragments.push("Provide the raw tool input in the `input` string field.".to_owned());
    let extra = custom_extra_fields(tool);
    if !extra.is_empty()
        && let Ok(serialized) = serde_json::to_string(&Value::Object(extra))
    {
        fragments.push(format!(
            "Additional custom tool declaration fields that must be respected:\n{serialized}"
        ));
    }
    fragments.join("\n\n")
}

/// Collect the custom tool's declaration fields the model must still respect.
fn custom_extra_fields(tool: &Value) -> Map<String, Value> {
    tool.as_object()
        .map(|object| {
            object
                .iter()
                .filter(|(key, _)| {
                    // `allowed_callers` is a caller restriction, not model-visible
                    // guidance; it is enforced (fail closed) in `lower_custom`, so
                    // it must never be reduced to unenforced prose here.
                    !matches!(
                        key.as_str(),
                        "type" | "name" | "description" | "format" | "defer_loading" | "allowed_callers"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Default `tool_search` parameters when the client omits them.
fn default_tool_search_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": TOOL_SEARCH_DEFAULT_QUERY_DESCRIPTION
            }
        },
        "required": ["query"],
        "additionalProperties": false
    })
}

/// Build the private `function` a `custom` tool lowers to.
fn custom_lowered_function(name: &str, description: &str) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Raw custom tool input. Follow the tool description and declared format exactly."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        },
        "strict": true,
    })
}

/// Base model-visible description for the private `function` a local `shell` tool
/// lowers to, before any declared skills are appended.
const SHELL_BASE_DESCRIPTION: &str = "Run one or more commands in the caller-provided local shell environment. The \
                                      caller executes the commands and returns their outputs.";

/// Build the model-visible description for a lowered local `shell` tool, folding
/// in the declared `environment.skills` so their names/descriptions/paths reach
/// the model.
///
/// Because the filter consumes the original `shell` declaration, the backend can
/// no longer validate `environment.skills`. This validates the exact
/// proxy-consumed schema in its place: when present, `skills` must be an array
/// (`LocalEnvironmentParam.skills` is non-nullable) of at most
/// [`MAX_LOCAL_SKILLS`] entries, and each entry must be a `LocalSkillParam` with
/// string `name`, `description`, and `path`. All three are required by the spec
/// but have no `minLength`, so empty strings are accepted — no stricter nonempty
/// constraint is imposed. A missing, `null`, non-array, non-string, or over-count
/// field fails closed rather than being silently erased.
fn shell_model_visible_description(tool: &Value) -> Result<String, FilterAction> {
    let mut description = SHELL_BASE_DESCRIPTION.to_owned();
    let Some(skills) = tool
        .get("environment")
        .and_then(|environment| environment.get("skills"))
    else {
        return Ok(description);
    };
    // `LocalEnvironmentParam.skills` is a non-nullable array in the spec, so an
    // explicit `null` is a schema violation and must fail closed rather than be
    // treated as "no skills" (which would silently drop a malformed declaration).
    let Some(skills) = skills.as_array() else {
        return Err(reject_bad_request(
            "shell tool environment.skills must be an array of local skills",
        ));
    };
    if skills.len() > MAX_LOCAL_SKILLS {
        return Err(reject_bad_request(&format!(
            "shell tool declares more than the maximum of {MAX_LOCAL_SKILLS} local skills",
        )));
    }
    if skills.is_empty() {
        return Ok(description);
    }
    let mut rendered = Vec::with_capacity(skills.len());
    for skill in skills {
        rendered.push(render_local_skill(skill)?);
    }
    description.push_str("\n\nThe caller's local shell environment provides these skills:\n");
    description.push_str(&rendered.join("\n"));
    Ok(description)
}

/// Validate and render one `LocalSkillParam` as a model-visible bullet.
///
/// The spec requires string `name`/`description`/`path` with no `minLength`, so a
/// missing or non-string field fails closed while a schema-valid empty string is
/// accepted (and simply not rendered). Caller-supplied text is rendered verbatim
/// — surrounding whitespace is not trimmed, so the model sees exactly what the
/// client declared.
fn render_local_skill(skill: &Value) -> Result<String, FilterAction> {
    let name = skill
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| reject_bad_request("local skill requires a string name"))?;
    let description = skill
        .get("description")
        .and_then(Value::as_str)
        .ok_or_else(|| reject_bad_request("local skill requires a string description"))?;
    let path = skill
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| reject_bad_request("local skill requires a string path"))?;
    let mut rendered = String::from("- ");
    rendered.push_str(name);
    if !description.is_empty() {
        rendered.push_str(": ");
        rendered.push_str(description);
    }
    if !path.is_empty() {
        rendered.push_str(" (path: ");
        rendered.push_str(path);
        rendered.push(')');
    }
    Ok(rendered)
}

/// Build the private `function` a local `shell` tool lowers to.
fn shell_lowered_function(description: &str) -> Value {
    json!({
        "type": "function",
        "name": SHELL_FUNCTION_NAME,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Commands to execute in order."
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional timeout in milliseconds."
                },
                "max_output_length": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional maximum captured output length."
                }
            },
            "required": ["commands"],
            "additionalProperties": false
        },
        "strict": false,
    })
}

/// Build the fixed private `function` a client-executed `tool_search` lowers to.
fn tool_search_lowered_function(description: &str, parameters: &Value) -> Value {
    json!({
        "type": "function",
        "name": TOOL_SEARCH_NAME,
        "description": description,
        "parameters": parameters,
        "strict": false,
    })
}

/// Lower a `tool_choice` value, translating client-owned selectors in place.
fn lower_tool_choice(choice: &mut Value, parts: &LoweringParts) -> Result<(), FilterAction> {
    let Some(object) = choice.as_object_mut() else {
        return Ok(());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("custom") => lower_custom_tool_choice(object, parts),
        Some("function") => lower_function_tool_choice(object, parts),
        Some("shell") => {
            lower_shell_tool_choice(object, &parts.reverse);
            Ok(())
        },
        Some("allowed_tools") => {
            if let Some(selectors) = object.get_mut("tools").and_then(Value::as_array_mut) {
                for selector in selectors.iter_mut() {
                    lower_tool_choice(selector, parts)?;
                }
            }
            Ok(())
        },
        _ => Ok(()),
    }
}

/// Lower a `{"type":"shell"}` (`SpecificFunctionShellParam`) `tool_choice`
/// selector to the private `shell` `function` selector when a local `shell` tool
/// was lowered this request. A native/container shell selector this filter did
/// not rewrite is left untouched.
///
/// The OpenAI schema defines no forced `tool_search` selector, so no analogous
/// translation is required for tool search.
fn lower_shell_tool_choice(object: &mut Map<String, Value>, reverse: &HashMap<String, LoweredClientTool>) {
    if matches!(
        reverse.get(SHELL_FUNCTION_NAME).map(|lowered| lowered.restore),
        Some(ClientToolRestore::Shell)
    ) {
        object.insert("type".to_owned(), json!("function"));
        object.insert("name".to_owned(), json!(SHELL_FUNCTION_NAME));
    }
}

/// Lower a `custom` `tool_choice` selector to a `function` selector in place.
///
/// A namespaced `custom` member (#1158) is selected with a `namespace` on the
/// selector; it lowers to the same flat private `function` selector as a namespaced
/// `function` member.
fn lower_custom_tool_choice(object: &mut Map<String, Value>, parts: &LoweringParts) -> Result<(), FilterAction> {
    if let Some(namespace) = namespaced_choice(object) {
        return lower_namespaced_tool_choice(object, parts, &namespace, ClientToolRestore::NamespaceCustom);
    }
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err(reject_bad_request("custom tool_choice requires a name"));
    };
    if !matches!(
        parts.reverse.get(name).map(|lowered| lowered.restore),
        Some(ClientToolRestore::Custom)
    ) {
        // Not callable as a lowered custom. A still-deferred withheld custom is
        // reported as deferred (loadable via tool_search); anything else as
        // undeclared.
        if parts.withheld.contains(name) {
            return Err(reject_deferred_choice(name));
        }
        return Err(reject_undeclared_choice(name));
    }
    object.insert("type".to_owned(), json!("function"));
    Ok(())
}

/// Lower a namespaced `function` `tool_choice` selector to its flat name in place.
///
/// A plain (non-namespaced) function selector is normally forwarded to the backend
/// verbatim: its name is not in the reverse map (a passthrough or discovered
/// function keeps its own name). The exception is a name that was declared with
/// `defer_loading: true` and withheld from the outbound set and never re-loaded by
/// a `tool_search`: forcing it would dangle a selector for an absent tool, so it
/// fails closed here with a deferred-specific rejection. A withheld name that
/// discovery later re-loaded is present in `callable`, so it still forwards.
fn lower_function_tool_choice(object: &mut Map<String, Value>, parts: &LoweringParts) -> Result<(), FilterAction> {
    if let Some(namespace) = namespaced_choice(object) {
        return lower_namespaced_tool_choice(object, parts, &namespace, ClientToolRestore::Namespace);
    }
    if let Some(name) = object.get("name").and_then(Value::as_str)
        && parts.withheld.contains(name)
        && !parts.callable.contains(name)
    {
        return Err(reject_deferred_choice(name));
    }
    Ok(())
}

/// Return the non-empty `namespace` of a `tool_choice` selector, if present.
fn namespaced_choice(object: &Map<String, Value>) -> Option<String> {
    object
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|namespace| !namespace.is_empty())
        .map(str::to_owned)
}

/// Lower a namespaced `tool_choice` selector (a `function` or `custom` member) to
/// its flat private `function` selector in place.
///
/// A namespaced `function` member and a namespaced `custom` member both lower to a
/// flat `function` on the wire, so a forced selector for either kind collapses the
/// same way: rewrite `type` to `function`, replace `name` with the flat member
/// name, and drop the `namespace`.
///
/// `expected` is the restore variant the selector's declared kind must match
/// (`Namespace` for a `function` selector, `NamespaceCustom` for a `custom`
/// selector). A selector whose kind disagrees with the declared member — a
/// `custom` selector naming a function member, or vice versa — fails closed rather
/// than silently forcing a member of the other kind, so an explicit `tool_choice`
/// is never widened or changed.
fn lower_namespaced_tool_choice(
    object: &mut Map<String, Value>,
    parts: &LoweringParts,
    namespace: &str,
    expected: ClientToolRestore,
) -> Result<(), FilterAction> {
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err(reject_bad_request("namespaced tool_choice requires a name"));
    };
    let flat = namespace_member_name(namespace, name);
    if parts.reverse.get(&flat).map(|lowered| lowered.restore) != Some(expected) {
        // Not callable as the expected member kind. A still-deferred withheld member
        // is reported as deferred; anything else as undeclared.
        if parts.withheld.contains(&flat) {
            return Err(reject_deferred_choice(name));
        }
        return Err(reject_undeclared_choice(name));
    }
    object.insert("type".to_owned(), json!("function"));
    object.insert("name".to_owned(), json!(flat));
    object.remove("namespace");
    Ok(())
}

/// Lower every typed *client-owned* history item to the function shape a
/// function-only backend accepts. Returns whether any item changed, or fails
/// closed on a prior history item this adapter cannot lower.
fn lower_history_items(messages: &mut [Value]) -> Result<bool, FilterAction> {
    reject_unsupported_history_items(messages)?;
    let lowered_call_ids = client_executed_call_ids(messages);
    let mut changed = false;
    for item in messages.iter_mut() {
        changed |= lower_history_item(item, &lowered_call_ids);
    }
    Ok(changed)
}

/// Fail closed on prior typed history items a function-only backend cannot accept
/// and this adapter does not yet lower.
///
/// A `local_shell` declaration is rejected on the request (see
/// [`reject_local_shell_unsupported`]); its typed history items
/// (`local_shell_call`/`local_shell_call_output`) are rejected here so an earlier
/// turn's unsupported call can never reach the backend unchanged on a
/// continuation. Full `local_shell` lowering/restoration is a follow-up.
fn reject_unsupported_history_items(messages: &[Value]) -> Result<(), FilterAction> {
    for item in messages {
        if matches!(
            item.get("type").and_then(Value::as_str),
            Some("local_shell_call" | "local_shell_call_output")
        ) {
            return Err(reject_local_shell_unsupported());
        }
    }
    Ok(())
}

/// Collect the `call_id`s of client-executed `shell_call`/`tool_search_call`
/// history items so their matching outputs are lowered in lockstep, while
/// server/container-owned calls (and their outputs) are left untouched.
fn client_executed_call_ids(messages: &[Value]) -> HashSet<String> {
    let mut ids = HashSet::new();
    for item in messages {
        let is_gated_call = matches!(
            item.get("type").and_then(Value::as_str),
            Some("shell_call" | "tool_search_call")
        );
        if is_gated_call
            && is_client_executed_tool_call(item)
            && let Some(call_id) = item.get("call_id").and_then(Value::as_str).filter(|id| !id.is_empty())
        {
            ids.insert(call_id.to_owned());
        }
    }
    ids
}

/// Return whether a typed `*_output` item's `call_id` matches a client-executed
/// call this turn lowered.
fn output_call_was_lowered(item: &Value, lowered_call_ids: &HashSet<String>) -> bool {
    item.get("call_id")
        .and_then(Value::as_str)
        .is_some_and(|call_id| lowered_call_ids.contains(call_id))
}

/// Lower a single conversation input item; returns whether it changed.
///
/// `custom_tool_call*` items are always client-owned and lowered unconditionally.
/// `shell_call`/`tool_search_call` items are lowered only when client-executed
/// (`environment.type == "local"` / `execution == "client"`), and their matching
/// `*_output` items only when their call was lowered — so a container/server call
/// keeps its wire semantics across a continuation turn.
fn lower_history_item(item: &mut Value, lowered_call_ids: &HashSet<String>) -> bool {
    match item.get("type").and_then(Value::as_str) {
        Some("custom_tool_call") => {
            *item = custom_call_to_function_call(item);
            true
        },
        Some("custom_tool_call_output") => {
            lower_custom_output(item);
            true
        },
        Some("shell_call") if is_client_executed_tool_call(item) => {
            *item = shell_call_to_function_call(item);
            true
        },
        Some("shell_call_output") if output_call_was_lowered(item, lowered_call_ids) => {
            *item = call_output_to_function_output(item, item.get("output"));
            true
        },
        Some("tool_search_call") if is_client_executed_tool_call(item) => {
            *item = tool_search_call_to_function_call(item);
            true
        },
        Some("tool_search_output") if output_call_was_lowered(item, lowered_call_ids) => {
            // The discovered `tools` are hoisted into the outbound tool set (see
            // `collect_discovered_tools`) so they become callable; the output itself
            // retains the raw definitions as the model-visible search result.
            *item = call_output_to_function_output(item, item.get("tools"));
            true
        },
        Some("function_call") => flatten_namespaced_history_call(item),
        _ => false,
    }
}

/// Re-type a `custom_tool_call_output` history item to a `function_call_output`.
fn lower_custom_output(item: &mut Value) {
    if let Some(object) = item.as_object_mut() {
        object.insert("type".to_owned(), json!("function_call_output"));
        object.remove("name");
    }
}

/// Convert a `custom_tool_call` history item to a `function_call`.
///
/// A namespaced custom call (a `custom_tool_call` carrying a `namespace`) flattens
/// to the same private member name it was lowered to, so a continuation turn
/// correlates with the backend's declared function.
fn custom_call_to_function_call(item: &Value) -> Value {
    let input = item.get("input").and_then(Value::as_str).unwrap_or_default();
    let mut out = json!({
        "type": "function_call",
        "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
        "name": namespaced_history_call_name(item),
        "arguments": json!({ "input": input }).to_string(),
    });
    apply_function_call_item_id(&mut out, item);
    apply_message_status(&mut out, item);
    carry_caller(&mut out, item);
    out
}

/// The lowered `function_call` name for a typed history call: a namespaced member
/// flattens to its collision-safe private name; a bare call keeps its original
/// name (or `null` when absent).
fn namespaced_history_call_name(item: &Value) -> Value {
    let Some(namespace) = item
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|namespace| !namespace.is_empty())
    else {
        return item.get("name").cloned().unwrap_or(Value::Null);
    };
    let member = item.get("name").and_then(Value::as_str).unwrap_or_default();
    json!(namespace_member_name(namespace, member))
}

/// Convert a `shell_call` history item to a `function_call` named `shell`.
fn shell_call_to_function_call(item: &Value) -> Value {
    let action = item.get("action").cloned().unwrap_or_else(|| json!({}));
    let arguments = serde_json::to_string(&action).unwrap_or_else(|_| "{}".to_owned());
    let mut out = json!({
        "type": "function_call",
        "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
        "name": SHELL_FUNCTION_NAME,
        "arguments": arguments,
    });
    apply_function_call_item_id(&mut out, item);
    apply_message_status(&mut out, item);
    carry_caller(&mut out, item);
    out
}

/// Convert a `tool_search_call` history item to a `function_call` named
/// `tool_search`.
fn tool_search_call_to_function_call(item: &Value) -> Value {
    let arguments = item.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let arguments = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_owned());
    let mut out = json!({
        "type": "function_call",
        "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
        "name": TOOL_SEARCH_NAME,
        "arguments": arguments,
    });
    if let Some(id) = item.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
        insert_field(&mut out, "id", json!(id));
    }
    apply_message_status(&mut out, item);
    out
}

/// Convert a typed client-owned `*_output` item to a `function_call_output`,
/// stringifying `payload` (the item's `output`/`tools` field) when structured.
///
/// `FunctionToolCallOutput` carries optional `id`, `status`, and `caller`, all
/// representation-compatible with the richer `FunctionShellCallOutput` /
/// `ToolSearchOutput` sources. Preserving them keeps a non-terminal
/// (`in_progress`/`incomplete`) continuation from reaching the backend as an
/// implicitly completed, identity-less output. An absent status is left absent
/// (not forced to `completed`) and a non-enum status is omitted, matching
/// [`apply_message_status`]; `caller` is `ToolCallCallerParam | null`, so a null
/// caller is omitted rather than forwarded.
fn call_output_to_function_output(item: &Value, payload: Option<&Value>) -> Value {
    let output = match payload {
        Some(Value::String(text)) => text.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    };
    let mut out = json!({
        "type": "function_call_output",
        "call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
        "output": output,
    });
    if let Some(id) = item.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
        insert_field(&mut out, "id", json!(id));
    }
    apply_message_status(&mut out, item);
    carry_caller(&mut out, item);
    out
}

/// Flatten a namespaced `function_call` history item to its lowered flat name.
fn flatten_namespaced_history_call(item: &mut Value) -> bool {
    let Some(namespace) = item
        .get("namespace")
        .and_then(Value::as_str)
        .filter(|namespace| !namespace.is_empty())
        .map(str::to_owned)
    else {
        return false;
    };
    let member = item.get("name").and_then(Value::as_str).unwrap_or_default();
    let flat = namespace_member_name(&namespace, member);
    if let Some(object) = item.as_object_mut() {
        object.insert("name".to_owned(), json!(flat));
        object.remove("namespace");
    }
    true
}

/// Copy a preserved item id onto a lowered history `function_call`, translating a
/// `ctc_`/`sh_` prefix to `fc_`.
fn apply_function_call_item_id(out: &mut Value, source: &Value) {
    let Some(item_id) = source.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) else {
        return;
    };
    let translated = item_id
        .strip_prefix("ctc_")
        .or_else(|| item_id.strip_prefix("sh_"))
        .filter(|suffix| !suffix.is_empty())
        .map_or_else(|| item_id.to_owned(), |suffix| format!("fc_{suffix}"));
    insert_field(out, "id", json!(translated));
}

/// Copy a schema-valid `FunctionCallStatus` (`in_progress`, `completed`, or
/// `incomplete`) onto a lowered `function_call`.
///
/// `FunctionToolCall.status` accepts all three values, so an `incomplete` prior
/// call keeps its terminal state across a continuation turn rather than reaching
/// the backend as a status-less call.
fn apply_message_status(out: &mut Value, source: &Value) {
    if let Some(status @ ("completed" | "in_progress" | "incomplete")) = source.get("status").and_then(Value::as_str) {
        insert_field(out, "status", json!(status));
    }
}

/// Carry a schema-valid non-null `caller` from a source tool-call item onto a
/// lowered or restored call.
///
/// `caller` is `ToolCallCaller | null` on `CustomToolCall`, `FunctionShellCall`,
/// and the `FunctionToolCall` they lower to (and restore from), so preserving it
/// keeps the execution context the model attached to the call across lowering and
/// restoration. A null caller is omitted rather than forwarded, matching the
/// `ToolCallCallerParam | null` shape.
fn carry_caller(out: &mut Value, source: &Value) {
    if let Some(caller) = source.get("caller").filter(|caller| !caller.is_null()) {
        insert_field(out, "caller", caller.clone());
    }
}

// -----------------------------------------------------------------------------
// Response restoration
// -----------------------------------------------------------------------------

/// Restore the client's original `tools`/`tool_choice` onto an echoed response
/// from the pre-lowering snapshot, keeping private lowered `function` names out
/// of client-visible output (#1159). Infallible: a `None` echo or a non-object
/// response is a no-op. A `Null` snapshot `tool_choice` means the client never
/// sent one, so the echoed field is REMOVED rather than set to null.
pub(crate) fn restore_snapshot_tools(response: &mut Value, echo: Option<&ClientToolEcho>) {
    let (Some(echo), Some(object)) = (echo, response.as_object_mut()) else {
        return;
    };
    object.insert("tools".to_owned(), Value::Array(echo.tools.clone()));
    if echo.tool_choice.is_null() {
        object.remove("tool_choice");
    } else {
        object.insert("tool_choice".to_owned(), echo.tool_choice.clone());
    }
}

/// Restore every lowered `function_call` in an echoed response's `output` array
/// to its canonical typed item (#1159). Fallible: `Err(item_type)` on the first
/// lossy item so the caller can fail the response closed. No-op when `output` is
/// absent or not an array.
pub(crate) fn restore_snapshot(
    response: &mut Value,
    reverse: &HashMap<String, LoweredClientTool>,
) -> Result<(), &'static str> {
    let Some(items) = response.get_mut("output").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for item in items {
        restore_output_item(item, reverse)?;
    }
    Ok(())
}

/// Restore one output item, re-typing a lowered `function_call` when its name is
/// in the reverse map.
pub(crate) fn restore_output_item(
    item: &mut Value,
    reverse: &HashMap<String, LoweredClientTool>,
) -> Result<(), &'static str> {
    if item.get("type").and_then(Value::as_str) != Some("function_call") {
        return Ok(());
    }
    let Some(name) = item.get("name").and_then(Value::as_str) else {
        return Ok(());
    };
    let Some(lowered) = reverse.get(name) else {
        return Ok(());
    };
    match lowered.restore {
        ClientToolRestore::Custom => {
            *item = restore_custom_call(item).map_err(|()| "custom_tool_call")?;
        },
        ClientToolRestore::Namespace => {
            restore_namespace_call(item, &lowered.original_name, lowered.namespace.as_deref());
        },
        ClientToolRestore::NamespaceCustom => {
            *item = restore_namespace_custom_call(item, &lowered.original_name, lowered.namespace.as_deref())
                .map_err(|()| "custom_tool_call")?;
        },
        ClientToolRestore::Shell => {
            *item = restore_shell_call(item).map_err(|()| "shell_call")?;
        },
        ClientToolRestore::ToolSearch => {
            *item = restore_tool_search_call(item).map_err(|()| "tool_search_call")?;
        },
    }
    Ok(())
}

/// Restore a lowered `function_call` to a `custom_tool_call`, failing closed when
/// the backend call lacks a usable `call_id`.
///
/// `CustomToolCall` requires a non-null `call_id`, so a missing, non-string, or
/// blank `call_id` fails closed (like [`restore_shell_call`] and
/// [`restore_tool_search_call`]) rather than emitting a schema-invalid
/// `call_id: null` the client could not map to an output. The freeform `input`
/// stays fail-open (best-effort recovery of the lowered arguments), and a
/// schema-valid `caller` is preserved.
///
/// Unlike [`restore_shell_call`] and [`restore_tool_search_call`], no `status`
/// is copied: `CustomToolCall` defines no `status` property in the OpenAI
/// schema, so carrying the backend `function_call`'s status would emit a
/// noncanonical field. "Preserve status" applies only to target item types that
/// define it.
pub(crate) fn restore_custom_call(item: &Value) -> Result<Value, ()> {
    let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
    let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or_default();
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|call_id| !call_id.trim().is_empty())
        .ok_or(())?;
    let input = input_from_arguments_strict(arguments)?;
    let mut out = json!({
        "type": "custom_tool_call",
        "id": custom_public_item_id(id),
        "call_id": call_id,
        "name": item.get("name").cloned().unwrap_or(Value::Null),
        "input": input,
    });
    carry_caller(&mut out, item);
    Ok(out)
}

/// Restore a lowered flat `function_call` to its namespaced form, in place.
pub(crate) fn restore_namespace_call(item: &mut Value, original_name: &str, namespace: Option<&str>) {
    if let Some(object) = item.as_object_mut() {
        object.insert("name".to_owned(), json!(original_name));
        if let Some(namespace) = namespace {
            object.insert("namespace".to_owned(), json!(namespace));
        }
    }
}

/// Restore a lowered flat `function_call` to a namespaced `custom_tool_call`,
/// recovering the member name and `namespace` and unwrapping the freeform input.
///
/// `CustomToolCall` carries an optional `namespace` field, so a namespaced custom
/// member round-trips to a `custom_tool_call` that names both its member and its
/// namespace.
pub(crate) fn restore_namespace_custom_call(
    item: &Value,
    member_name: &str,
    namespace: Option<&str>,
) -> Result<Value, ()> {
    let mut out = restore_custom_call(item)?;
    if let Some(object) = out.as_object_mut() {
        object.insert("name".to_owned(), json!(member_name));
        if let Some(namespace) = namespace {
            object.insert("namespace".to_owned(), json!(namespace));
        }
    }
    Ok(out)
}

/// Parse and validate a lowered shell call's `arguments` into a `shell_call`
/// `action`, failing closed on any shape the backend could not have produced.
///
/// `FunctionShellAction` requires the `timeout_ms` and `max_output_length` keys
/// (both nullable). The lowered function omits them, so a missing optional is
/// normalized to an explicit null to keep the restored action schema-complete.
pub(crate) fn parse_shell_action(arguments: &str) -> Result<Value, ()> {
    let mut action: Value = serde_json::from_str(arguments).ok().ok_or(())?;
    {
        let object = action.as_object().ok_or(())?;
        let commands = object.get("commands").and_then(Value::as_array).ok_or(())?;
        if !commands.iter().all(Value::is_string) {
            return Err(());
        }
        let is_bad_u64 = |field: &str| {
            object
                .get(field)
                .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
        };
        if is_bad_u64("timeout_ms") || is_bad_u64("max_output_length") {
            return Err(());
        }
    }
    if let Some(object) = action.as_object_mut() {
        object.entry("timeout_ms").or_insert(Value::Null);
        object.entry("max_output_length").or_insert(Value::Null);
    }
    Ok(action)
}

/// Restore a lowered `function_call` to a `shell_call` (fail-closed on bad args).
///
/// `FunctionShellCall` requires a non-null `call_id`, so a missing/empty
/// `call_id` fails closed. Every schema-valid `FunctionShellCallStatus`
/// (`in_progress`, `completed`, `incomplete`) is preserved verbatim so the client
/// never executes a call the backend reported as incomplete; an absent or null
/// status defaults to `completed`, while a wrong-typed or unknown status fails
/// closed so a malformed backend call never becomes an executable client call.
pub(crate) fn restore_shell_call(item: &Value) -> Result<Value, ()> {
    let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or_default();
    let action = parse_shell_action(arguments)?;
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|call_id| !call_id.trim().is_empty())
        .ok_or(())?;
    let status = restore_call_status(item)?;
    let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
    let mut out = json!({
        "type": "shell_call",
        "id": shell_public_item_id(id),
        "call_id": call_id,
        "action": action,
        // A local shell_call carries its environment so the client-executed
        // classifier recognizes it and mixed-ownership rounds fail closed.
        "environment": {"type": "local"},
        "status": status,
    });
    carry_caller(&mut out, item);
    Ok(out)
}

/// Restore a lowered `function_call` to a `tool_search_call` (fail-closed).
///
/// `ToolSearchCall.status` is a required `FunctionCallStatus`, so every valid
/// `in_progress`/`completed`/`incomplete` value is preserved verbatim (absent or
/// null defaults to `completed`); a wrong-typed or unknown status fails closed.
pub(crate) fn restore_tool_search_call(item: &Value) -> Result<Value, ()> {
    let id = item.get("id").and_then(Value::as_str).unwrap_or_default();
    let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or_default();
    if id.trim().is_empty() || call_id.trim().is_empty() {
        return Err(());
    }
    if item.get("namespace").is_some_and(|namespace| !namespace.is_null()) {
        return Err(());
    }
    let status = restore_call_status(item)?;
    let arguments = item.get("arguments").and_then(Value::as_str).unwrap_or_default();
    let arguments: Value = serde_json::from_str(arguments).ok().ok_or(())?;
    Ok(json!({
        "type": "tool_search_call",
        "id": tool_search_public_item_id(id),
        "call_id": call_id,
        "execution": "client",
        "arguments": arguments,
        "status": status,
    }))
}

/// Map a backend `function_call` `status` to a schema-valid `FunctionCallStatus`
/// for a restored client-owned call.
///
/// An absent or null status defaults to `completed`; a valid `in_progress`,
/// `completed`, or `incomplete` string is preserved so a non-terminal backend
/// call is never presented to the client as completed; a wrong-typed (non-string)
/// or unknown value fails closed so a malformed call cannot become executable.
pub(crate) fn restore_call_status(item: &Value) -> Result<&'static str, ()> {
    match item.get("status") {
        None | Some(Value::Null) => Ok("completed"),
        Some(Value::String(status)) => match status.as_str() {
            "in_progress" => Ok("in_progress"),
            "completed" => Ok("completed"),
            "incomplete" => Ok("incomplete"),
            _ => Err(()),
        },
        Some(_) => Err(()),
    }
}

/// The private wire envelope a lowered `custom` tool's arguments carry: exactly
/// one string field named `input`. `deny_unknown_fields` makes any extra key a
/// hard parse error so a malformed backend echo cannot leak a private shape.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CustomInputEnvelope {
    /// The single string argument the lowered `custom` tool wraps; unwrapped back
    /// to the plain-string `input` of a canonical `custom_tool_call`.
    input: String,
}

/// Unwrap a lowered `custom` call's `{"input": "<string>"}` arguments envelope
/// back to the plain-string `input` field of a canonical `custom_tool_call`.
///
/// Fail-closed (#1159): returns `Err(())` on invalid JSON, a missing or extra
/// key, or a non-string `input`, so restoration rejects the item rather than
/// leaking the private lowered arguments shape to the client.
pub(crate) fn input_from_arguments_strict(arguments: &str) -> Result<String, ()> {
    match serde_json::from_str::<CustomInputEnvelope>(arguments) {
        Ok(envelope) => Ok(envelope.input),
        Err(_) => Err(()),
    }
}

// -----------------------------------------------------------------------------
// Name + id helpers
// -----------------------------------------------------------------------------

/// Return whether a lowered function name matches the OpenAI schema
/// (`^[a-zA-Z0-9_-]{1,64}$`).
fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_FUNCTION_NAME_LEN
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Flatten a Codex namespace member to its model-visible function name.
///
/// This must be injective across *distinct* `(namespace, member)` pairs: the
/// reverse restoration map, the `tool_choice` rewriter, and the withheld-budget
/// reclaim ([`Lowering::reclaim_withheld`]) all key by the returned wire name, so
/// two distinct members flattening to one name would restore to the wrong item,
/// redirect a forced selector, or miscount the `max_client_tools` cap. The
/// verbatim form is kept injective by [`reject_reserved_namespace_delimiter`]. The
/// hash-truncation branch is domain-separated with a `___` marker: a verbatim
/// `agentic_ns__{namespace}__{member}` can never contain three consecutive
/// underscores (the prefix ends in `__`, and the delimiter guard forbids a
/// component from embedding `__` or abutting the separator with a leading/trailing
/// `_`, so every junction is `__` + non-`_`), so a hashed name — which always
/// contains `___` — is byte-disjoint from every verbatim name. Without the marker
/// a verbatim member name set to the forward FNV hex of a longer member's full
/// name collides with that member's hashed wire name (no hash search required).
fn namespace_member_name(namespace: &str, member: &str) -> String {
    let full_name = format!("{NAMESPACE_MEMBER_PREFIX}{namespace}__{member}");
    if full_name.chars().count() <= MAX_FUNCTION_NAME_LEN {
        return full_name;
    }
    let hash = stable_name_hash(&full_name);
    let readable_len = MAX_FUNCTION_NAME_LEN - HASHED_NAMESPACE_MEMBER_SUFFIX_LEN;
    let readable_prefix: String = full_name.chars().take(readable_len).collect();
    format!("{readable_prefix}___{hash:016x}")
}

/// Fail closed on a client-declared top-level `function` or `custom` tool whose name
/// uses the [`NAMESPACE_MEMBER_PREFIX`] this filter reserves for the flat wire names it
/// synthesizes for `namespace` members ([`namespace_member_name`]).
///
/// Every synthesized member name begins with that prefix (a hashed long name still
/// keeps the readable prefix), so a client tool sharing it would collide with a member
/// on the wire. That collision corrupts restoration (the reverse map keys by wire name,
/// so a backend `function_call` could resolve to the wrong typed item) and defeats the
/// `max_client_tools` accounting (a withheld top-level declaration and a discovered
/// namespace member sharing the reserved name would be miscounted as one logical tool,
/// letting distinct tools slip past the cap). Reserving the prefix keeps wire names
/// unique across the top-level/namespace boundary, which is what makes the withheld
/// reclaim ([`Lowering::reclaim_withheld`]) sound to match reloads by wire name alone.
fn reject_reserved_top_level_name(tool: &Value) -> Result<(), FilterAction> {
    if matches!(tool.get("type").and_then(Value::as_str), Some("function" | "custom"))
        && let Some(name) = tool.get("name").and_then(Value::as_str)
        && name.starts_with(NAMESPACE_MEMBER_PREFIX)
    {
        return Err(reject_bad_request(&format!(
            "client tool '{name}' uses the reserved '{NAMESPACE_MEMBER_PREFIX}' namespace prefix"
        )));
    }
    Ok(())
}

/// Fail closed on a `namespace` group name or member name that embeds — or abuts —
/// the `__` delimiter [`namespace_member_name`] reserves to separate the prefix,
/// namespace, and member components of a flattened wire name.
///
/// The flattening `agentic_ns__{namespace}__{member}` is injective only while a
/// component neither contains `__` nor places a single `_` against the `__`
/// separator. Both cases create an alternative split of the same wire name for two
/// *distinct* `(namespace, member)` pairs:
/// - embedded `__`: namespace `a` member `b__c` and namespace `a__b` member `c` both flatten to `agentic_ns__a__b__c`;
/// - boundary `_`: namespace `a` member `_b` and namespace `a_` member `b` both flatten to `agentic_ns__a___b` (the
///   boundary `_` merges with the separator into `___`, which re-splits).
///
/// [`Lowering::claim_lowered`] already fails closed on such a collision between two
/// callable members, but a *deferred* member is withheld (never claimed), so a later
/// distinct member colliding with it would slip past that guard and let
/// [`Lowering::reclaim_withheld`] credit one member's withheld budget to the other,
/// miscounting two logical tools as one under `max_client_tools` (and forcing a
/// namespaced `tool_choice` for one member onto the other via the shared wire name).
/// Rejecting both an embedded `__` and a leading or trailing `_` keeps the flattening
/// injective on the deferred path too, so wire-name matching stays sound. The leading-
/// `_`-on-a-group and trailing-`_`-on-a-member cases cannot merge with the separator,
/// but are rejected as well so the rule stays symmetric and robust to future format
/// changes. `kind` labels the offending component (`group` or `member`).
///
/// This guard only makes the *verbatim* flattening injective; the hash-truncation
/// branch of [`namespace_member_name`] (taken for names over
/// [`MAX_FUNCTION_NAME_LEN`]) is kept disjoint from every verbatim name by its own
/// `___` domain-separator marker, since these guarded names never yield a verbatim
/// wire containing three consecutive underscores.
fn reject_reserved_namespace_delimiter(kind: &str, name: &str) -> Result<(), FilterAction> {
    if name.contains(NAMESPACE_NAME_DELIMITER) || name.starts_with('_') || name.ends_with('_') {
        return Err(reject_bad_request(&format!(
            "namespace {kind} name '{name}' may not contain the reserved '{NAMESPACE_NAME_DELIMITER}' \
             delimiter or begin or end with '_'"
        )));
    }
    Ok(())
}

/// Derive the public `custom_tool_call` item id from a returned function id.
pub(crate) fn custom_public_item_id(item_id: &str) -> String {
    if item_id.starts_with("ctc_") {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ctc_{suffix}");
    }
    format!("ctc_{:016x}", stable_name_hash(item_id))
}

/// Derive the public `shell_call` item id from a returned function id.
fn shell_public_item_id(item_id: &str) -> String {
    if item_id.starts_with("sh_") {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("sh_{suffix}");
    }
    format!("sh_{:016x}", stable_name_hash(item_id))
}

/// Derive the public `tool_search_call` item id from a returned function id.
fn tool_search_public_item_id(item_id: &str) -> String {
    if item_id.strip_prefix("tsc_").is_some_and(|suffix| !suffix.is_empty()) {
        return item_id.to_owned();
    }
    if let Some(suffix) = item_id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("tsc_{suffix}");
    }
    format!("tsc_{:016x}", stable_name_hash(&format!("tool_search_item:{item_id}")))
}

/// FNV-1a hash matching the upstream stable name hash.
fn stable_name_hash(value: &str) -> u64 {
    value.bytes().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

// -----------------------------------------------------------------------------
// Rejections
// -----------------------------------------------------------------------------

/// Reject a request with an `invalid_request_error` before any upstream call.
fn reject_bad_request(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(400, "invalid_request_error", message))
}

/// Reject a lowered name that collides with another declared tool.
fn reject_collision(name: &str) -> FilterAction {
    reject_bad_request(&format!(
        "client tool '{name}' collides with another tool after lowering to a private function declaration"
    ))
}

/// Reject an ambiguous discovered tool: the same `(type, name)` was returned with
/// conflicting definitions across `tool_search` results, so the correct
/// declaration to hoist is ambiguous and lowering fails closed.
fn reject_conflicting_discovery(tool_type: &str, name: &str) -> FilterAction {
    reject_bad_request(&format!(
        "tool_search returned conflicting definitions for the '{tool_type}' tool '{name}'; the discovered tool \
         declaration is ambiguous and cannot be lowered"
    ))
}

/// Reject a lowered function name that is not schema-valid.
fn reject_invalid_name(name: &str) -> FilterAction {
    reject_bad_request(&format!(
        "client tool '{name}' cannot be lowered to a valid function name (must match ^[a-zA-Z0-9_-]{{1,64}}$)"
    ))
}

/// Reject a `tool_choice` selector that names an undeclared client tool.
fn reject_undeclared_choice(name: &str) -> FilterAction {
    reject_bad_request(&format!(
        "tool_choice selects client tool '{name}', but no matching client tool is declared"
    ))
}

/// Reject a `tool_choice` selector that forces a declared tool whose loading is
/// deferred (`defer_loading: true`): it is withheld from the callable set until a
/// `tool_search` discovers it, so it cannot be forced yet.
fn reject_deferred_choice(name: &str) -> FilterAction {
    reject_bad_request(&format!(
        "tool_choice selects client tool '{name}', but its loading is deferred until a tool_search discovers it"
    ))
}

/// Reject a request that lowers more client tools than the configured cap.
fn reject_too_many_tools(max: usize) -> FilterAction {
    reject_bad_request(&format!(
        "request lowers more than the configured maximum of {max} client tools"
    ))
}

/// Reject a `local_shell` tool before any upstream call.
///
/// `local_shell` restores to a distinct `local_shell_call` output shape that is
/// not yet implemented, so it fails closed rather than being forwarded to a
/// function-only backend as an unsupported tool.
fn reject_local_shell_unsupported() -> FilterAction {
    reject_bad_request(
        "local_shell tools are not yet supported on a function-only Responses backend; declare a local shell tool \
         (type \"shell\" with environment.type \"local\") instead",
    )
}

/// Reject a namespace member this filter cannot lower to a private function.
fn reject_unsupported_namespace_member(namespace: &str, member_type: Option<&str>) -> FilterAction {
    let member_type = member_type.unwrap_or("<missing>");
    reject_bad_request(&format!(
        "namespace '{namespace}' contains an unsupported '{member_type}' member; only function and custom members can \
         be lowered for a function-only Responses backend"
    ))
}

/// Fail closed when a `custom` declaration (a top-level `custom` tool or a
/// `custom` namespace member) uses features that cannot be preserved once lowered
/// to a private `function`: a non-`text` constrained `format` or a restricted
/// `allowed_callers`. `descriptor` names the offending tool for the rejection
/// message (e.g. `custom tool 'x'` or `namespace 'n' member 'm'`).
///
/// `defer_loading` is intentionally not rejected here: a deferred `custom`
/// declaration is withheld from the outbound set on the declaration path (see
/// `lower_tools`/`lower_namespace`) and consumed — loaded and made callable — on
/// the discovery path, exactly like a deferred `function`.
fn reject_unsupported_custom_declaration(tool: &Value, descriptor: &str) -> Result<(), FilterAction> {
    if let Some(format) = tool.get("format")
        && format.get("type").and_then(Value::as_str) != Some("text")
    {
        return Err(reject_bad_request(&format!(
            "{descriptor} uses an unsupported custom format; constrained decoding cannot be preserved for a \
             function-only backend"
        )));
    }
    reject_if_restricted_callers(tool, descriptor)
}

/// Fail closed when a rewritten tool declares a non-null `allowed_callers`
/// restriction: a function-only backend cannot enforce it, so lowering it silently
/// would let a restricted tool become freely callable by the model.
fn reject_if_restricted_callers(tool: &Value, descriptor: &str) -> Result<(), FilterAction> {
    if tool.get("allowed_callers").is_some_and(|callers| !callers.is_null()) {
        return Err(reject_bad_request(&format!(
            "{descriptor} declares allowed_callers, which cannot be enforced once lowered to a private function; \
             remove the restriction or omit the tool"
        )));
    }
    Ok(())
}

/// #1159: streaming client-tool restoration was requested but the
/// `openai_stream_events` logical SSE owner is not in the pipeline, so there is
/// nothing to restore the lowered calls in the stream. Misconfiguration, so a
/// 500 (not a client 4xx): the operator must place `openai_stream_events` before
/// `openai_client_tool_compat` in the inference step.
fn reject_streaming_missing_owner() -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        500,
        "server_error",
        "openai_stream_events must precede openai_client_tool_compat to restore \
         lowered client tools on the streaming Responses path",
    ))
}

/// Reject a response whose lowered call cannot be restored losslessly.
fn reject_lossy_restore(item_type: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        &format!("backend returned a {item_type} function call that cannot be restored losslessly"),
    ))
}
