// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming restoration of lowered Codex client tools (#1159).
//!
//! `openai_client_tool_compat` lowers rich client tools (`custom`, `namespace`
//! members, local `shell`, client-executed `tool_search`) to private `function`
//! tools on the outbound request. When the model returns a lowered
//! `function_call` in the Responses SSE stream, this module plans how to restore
//! the original typed item live, inside the single `openai_stream_events` logical
//! owner — no second accumulator, no full-body buffering.
//!
//! The plan pass ([`plan_client_tool_restore`]) is fallible and runs between the
//! commit's phase-2a accumulate and phase-2b append; the disposition applier is
//! infallible and runs inside phase 2b. Native passthrough (empty lowering map)
//! skips planning entirely for zero overhead.

use std::collections::HashMap;

use serde_json::Value;

use crate::openai::{
    responses::{
        openai_client_tool_compat::{
            custom_public_item_id, input_from_arguments_strict, restore_custom_call, restore_namespace_custom_call,
            restore_shell_call, restore_snapshot, restore_snapshot_tools, restore_tool_search_call,
        },
        state::{ClientToolEcho, ClientToolRestore, LoweredClientTool},
    },
    sse::{SseParseError, responses::ResponsesEvent},
};

/// Per-item lifecycle progress through a lowered `function_call`'s SSE events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClientToolPhase {
    /// `output_item.added` seen; arguments not yet complete.
    Opened,
    /// `function_call_arguments.done` seen; the arguments are final.
    ArgsComplete,
    /// `output_item.done` seen; the item is finalized.
    Done,
}

/// A lowered client-tool item tracked across its streaming lifecycle.
///
/// Created at `output_item.added` and advanced by later argument/finalizer
/// events so the plan pass can enforce lifecycle order (fail closed on a
/// premature `output_item.done`) and synthesize the typed restoration for the
/// `Custom`/`NamespaceCustom` and `Shell`/`ToolSearch` kinds.
#[derive(Clone, Debug)]
pub(super) struct ClientToolStreamItem {
    /// Stable key matching this item across its lifecycle events (`item:{id}` or
    /// `index:{output_index}`).
    pub key: String,
    /// The private lowered name the backend returned (`agentic_ns__{ns}__{member}`
    /// or a private `custom`/`shell`/`tool_search` name). Never leaked to clients.
    pub private_name: String,
    /// The typed item this lowered `function_call` restores to.
    pub restore: ClientToolRestore,
    /// Lifecycle progress observed so far.
    pub phase: ClientToolPhase,
    /// Absolute output index carried by the item's lifecycle events.
    pub output_index: u64,
    /// The item id (`item.id`), when the backend supplied one.
    pub item_id: Option<String>,
}

/// A completed private `function_call` item captured from accumulated storage at
/// `function_call_arguments.done`, keyed for typed synthesis of the restored
/// `custom_tool_call` lifecycle (#1159).
#[derive(Clone, Debug)]
pub(super) struct ClientToolCompletion {
    /// Stable key matching the completion to its tracked stream item.
    pub key: String,
    /// The completed private `function_call` item, cloned from storage.
    pub item: Value,
}

/// How one committed SSE event must be restored before it reaches the client.
///
/// Task 4 constructs `Passthrough`/`RetypeInPlace`; Task 5 adds
/// `Suppress`/`EmitCustomShell`/`EmitCustomInput`/`EmitCustomItemDone` for
/// `Custom`/`NamespaceCustom` synthesis. Task 6 adds `EmitTypedAdded`/`EmitTypedDone`
/// for `Shell`/`ToolSearch` synthesis; Task 7 adds `RestoreSnapshot` for non-terminal
/// snapshot restoration.
#[derive(Clone, Debug)]
pub(super) enum ClientToolDisposition {
    /// Forward the event unchanged.
    Passthrough,
    /// Drop the event entirely (the typed replacement is synthesized elsewhere).
    Suppress,
    /// Retype the event's `item` in place: set `type`/`name` and re-add or remove
    /// `namespace`. Used for `Namespace` members (retype-in-place, no synthesis).
    RetypeInPlace {
        /// The restored output-item type (`function_call`).
        item_type: &'static str,
        /// The original member name to restore.
        name: String,
        /// The namespace to re-add, or `None` to remove it.
        namespace: Option<String>,
    },
    /// Emit a synthesized `custom_tool_call` `output_item.added` for
    /// `Custom`/`NamespaceCustom` (Task 5).
    EmitCustomShell {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized custom-tool input event (Task 5).
    EmitCustomInput {
        /// Stable key of the tracked stream item.
        #[expect(dead_code, reason = "retained for debug formatting, not read by applier")]
        key: String,
        /// The item id to stamp on the synthesized event.
        item_id: String,
        /// The absolute output index.
        output_index: u64,
        /// The unwrapped plain-string input.
        input: String,
    },
    /// Emit a synthesized custom-tool `output_item.done` (Task 5).
    EmitCustomItemDone {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized typed `output_item.added` for `Shell`/`ToolSearch` (Task 6).
    EmitTypedAdded {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized typed `output_item.done` (Task 6).
    EmitTypedDone {
        /// The synthesized output item.
        item: Value,
    },
    /// Restore an intermediate (non-terminal) response snapshot's `tools`/`tool_choice`
    /// and output items so lowered names never appear in client-visible
    /// `response.created`/`response.queued`/`response.in_progress` frames (Task 7).
    /// Terminal snapshots (`response.completed`/`incomplete`/`failed`) are handled
    /// in `canonicalize_logical_response` (Task 9).
    RestoreSnapshot {
        /// The restored response object.
        response: Value,
    },
}

/// The planned restoration for one committed chunk: one disposition per event,
/// plus the advanced per-item lifecycle state to store back on the owner.
pub(super) struct PlannedRestore {
    /// One disposition per input event, in order.
    pub dispositions: Vec<ClientToolDisposition>,
    /// The tracked stream items after applying this chunk's lifecycle advances.
    pub next_items: Vec<ClientToolStreamItem>,
}

/// Map each committed-this-chunk event to a restoration disposition, staging the
/// per-item lifecycle advance in `next_items`. Fail closed (`Err`) on any
/// lifecycle-order violation, missing artifact, or lossy restore. Native
/// passthrough (empty `reverse`) returns an empty plan with zero work.
pub(super) fn plan_client_tool_restore(
    reverse: &HashMap<String, LoweredClientTool>,
    echo: Option<&ClientToolEcho>,
    committed: &[ClientToolStreamItem],
    events: &[ResponsesEvent],
    completions: &[ClientToolCompletion],
) -> Result<PlannedRestore, SseParseError> {
    let mut plan = PlannedRestore {
        dispositions: Vec::new(),
        next_items: committed.to_vec(),
    };
    if reverse.is_empty() {
        return Ok(plan);
    }
    for event in events {
        let disposition = plan_one_event(reverse, echo, &mut plan.next_items, completions, event)?;
        plan.dispositions.push(disposition);
    }
    Ok(plan)
}

/// Plan the restoration for a single committed event, advancing `next_items`.
fn plan_one_event(
    reverse: &HashMap<String, LoweredClientTool>,
    echo: Option<&ClientToolEcho>,
    next_items: &mut Vec<ClientToolStreamItem>,
    completions: &[ClientToolCompletion],
    event: &ResponsesEvent,
) -> Result<ClientToolDisposition, SseParseError> {
    // Non-terminal snapshot restoration: restore tools/tool_choice and output items
    // in intermediate response.* events (created/queued/in_progress). Terminal snapshots
    // (completed/incomplete/failed) are handled in canonicalize_logical_response (Task 9).
    if !event.is_terminal()
        && let Some(response_object) = event.payload().get("response")
    {
        let mut response = response_object.clone();
        restore_snapshot_tools(&mut response, echo);
        restore_snapshot(&mut response, reverse).map_err(|item_type| {
            client_tool_restore_error(
                "response-snapshot",
                &format!("response snapshot contains a lossy lowered {item_type}"),
            )
        })?;
        return Ok(ClientToolDisposition::RestoreSnapshot { response });
    }

    match event {
        ResponsesEvent::OutputItemAdded(payload) => plan_output_item_added(reverse, next_items, payload),
        ResponsesEvent::FunctionCallArgumentsDelta(payload) => Ok(plan_arguments_delta(next_items, payload)),
        ResponsesEvent::FunctionCallArgumentsDone(payload) => plan_arguments_done(next_items, completions, payload),
        ResponsesEvent::OutputItemDone(payload) => plan_output_item_done(reverse, next_items, payload),
        _ => Ok(ClientToolDisposition::Passthrough),
    }
}

/// Whether a lowered restore kind is tracked and restored across its streaming
/// lifecycle. Every current kind is, so this is a single explicit gate for the
/// plan pass rather than a behavioral filter; the kind-dispatch matches it guards
/// are exhaustive, so a new variant would force a compile error instead of being
/// silently mishandled (#1159).
fn is_restorable(restore: ClientToolRestore) -> bool {
    matches!(
        restore,
        ClientToolRestore::Namespace
            | ClientToolRestore::Custom
            | ClientToolRestore::NamespaceCustom
            | ClientToolRestore::Shell
            | ClientToolRestore::ToolSearch
    )
}

/// Plan an `output_item.added`: open tracking for a lowered `Namespace`,
/// `Custom`, `NamespaceCustom`, `Shell`, or `ToolSearch` call. `Namespace` retypes
/// the `function_call` in place; `Custom`/`NamespaceCustom` synthesize a
/// `custom_tool_call` added item carrying the public id; `Shell`/`ToolSearch`
/// suppress the raw `added` and synthesize the complete typed call at
/// `arguments.done` (#1159 Task 6). Non-lowered items pass through unchanged.
fn plan_output_item_added(
    reverse: &HashMap<String, LoweredClientTool>,
    next_items: &mut Vec<ClientToolStreamItem>,
    payload: &Value,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(item) = payload.get("item") else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    let Some(name) = item.get("name").and_then(Value::as_str) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    let Some(lowered) = reverse.get(name) else {
        return Ok(ClientToolDisposition::Passthrough);
    };

    // Every lowered kind is tracked so its lifecycle can be restored: Namespace
    // retypes in place, Custom/NamespaceCustom synthesize a custom_tool_call, and
    // Shell/ToolSearch suppress the raw lifecycle and synthesize a complete typed
    // call at arguments.done (#1159 Task 6). Any future kind not yet handled passes
    // through untracked.
    if !is_restorable(lowered.restore) {
        return Ok(ClientToolDisposition::Passthrough);
    }

    let Some(key) = client_tool_event_key(payload) else {
        // #1159 C1 defense-in-depth: `name` is already proven lowered (found in
        // `reverse`) and restorable, so this is not a genuine non-lowered item.
        // A lowered `added` with no resolvable event key cannot be tracked and its
        // later `.done` could not be matched, which would risk surfacing the raw
        // private name — fail the whole stream closed instead of passing through.
        return Err(client_tool_restore_error(
            "output-item-added",
            "lowered client-tool added without a resolvable event key",
        ));
    };
    next_items.push(ClientToolStreamItem {
        key,
        private_name: name.to_owned(),
        restore: lowered.restore,
        phase: ClientToolPhase::Opened,
        output_index: payload.get("output_index").and_then(Value::as_u64).unwrap_or_default(),
        item_id: item.get("id").and_then(Value::as_str).map(ToOwned::to_owned),
    });
    Ok(plan_added_disposition(lowered, payload))
}

/// Select the `output_item.added` disposition for a lowered call: retype a
/// `Namespace` member's `function_call` in place, synthesize a `custom_tool_call`
/// added item for `Custom`/`NamespaceCustom` (client-visible name + public id), or
/// suppress the raw `Shell`/`ToolSearch` added — its schema-complete typed
/// `output_item.added` is synthesized at `arguments.done` instead (#1159 Task 6).
fn plan_added_disposition(lowered: &LoweredClientTool, payload: &Value) -> ClientToolDisposition {
    match lowered.restore {
        ClientToolRestore::Namespace => ClientToolDisposition::RetypeInPlace {
            item_type: "function_call",
            name: lowered.original_name.clone(),
            namespace: lowered.namespace.clone(),
        },
        // Custom/NamespaceCustom: retype the announced item to `custom_tool_call`
        // with the client-visible name and public id; the private lowered name and
        // `fc_` id never reach the client.
        ClientToolRestore::Custom | ClientToolRestore::NamespaceCustom => ClientToolDisposition::EmitCustomShell {
            item: build_custom_added_payload(payload, lowered),
        },
        // Shell/ToolSearch: drop the raw added — a partial typed call at add time
        // cannot be schema-valid (it lacks `action`/`arguments`). The complete typed
        // `output_item.added` is synthesized once arguments are final.
        ClientToolRestore::Shell | ClientToolRestore::ToolSearch => ClientToolDisposition::Suppress,
    }
}

/// Build the retyped `custom_tool_call` `output_item.added` payload for a lowered
/// `Custom`/`NamespaceCustom` call.
///
/// Clones the committed `output_item.added` payload once (the plan pass only
/// borrows the event, so it needs an owned payload to hand the applier — AGENTS.md
/// ownership boundary) and re-types its nested item: `type` becomes
/// `custom_tool_call`, `name` becomes the client's original member name (never the
/// private lowered name), `namespace` is re-added for `NamespaceCustom`, and the
/// `fc_` id is rewritten to its public `ctc_` form. The freeform `input` is absent
/// at add time — it arrives via the synthesized `custom_tool_call_input` events.
fn build_custom_added_payload(payload: &Value, lowered: &LoweredClientTool) -> Value {
    let mut out = payload.clone();
    if let Some(item) = out.get_mut("item").and_then(Value::as_object_mut) {
        item.insert("type".to_owned(), Value::String("custom_tool_call".to_owned()));
        item.insert("name".to_owned(), Value::String(lowered.original_name.clone()));
        match &lowered.namespace {
            Some(namespace) => {
                item.insert("namespace".to_owned(), Value::String(namespace.clone()));
            },
            None => {
                item.remove("namespace");
            },
        }
        if let Some(id) = item.get("id").and_then(Value::as_str) {
            let public_id = custom_public_item_id(id);
            item.insert("id".to_owned(), Value::String(public_id));
        }
    }
    out
}

/// Plan a `function_call_arguments.delta`: suppress the backend's function-args
/// delta for a tracked `Custom`/`NamespaceCustom` call (the canonical lifecycle
/// uses synthesized `custom_tool_call_input.delta` instead) and for a tracked
/// `Shell`/`ToolSearch` call (no incremental client-visible input family exists;
/// the complete typed call is synthesized at `arguments.done`). The arguments frame
/// carries no typed name, so lowered-ness is resolved by matching tracked items by
/// key. Everything else (including `Namespace`, whose args pass through unchanged)
/// forwards untouched.
fn plan_arguments_delta(next_items: &[ClientToolStreamItem], payload: &Value) -> ClientToolDisposition {
    let Some(key) = client_tool_event_key(payload) else {
        return ClientToolDisposition::Passthrough;
    };
    match next_items
        .iter()
        .find(|tracked| tracked.key == key)
        .map(|tracked| tracked.restore)
    {
        Some(
            ClientToolRestore::Custom
            | ClientToolRestore::NamespaceCustom
            | ClientToolRestore::Shell
            | ClientToolRestore::ToolSearch,
        ) => ClientToolDisposition::Suppress,
        _ => ClientToolDisposition::Passthrough,
    }
}

/// Plan a `function_call_arguments.done`: advance the tracked item to
/// `ArgsComplete` and synthesize the restored lifecycle from the completion artifact
/// captured in phase 2a. `Custom`/`NamespaceCustom` synthesize the canonical custom
/// input event pair; `Shell`/`ToolSearch` synthesize the complete typed
/// `output_item.added` (the raw added was suppressed). The arguments frame carries
/// no typed name, so lowered-ness is resolved by matching tracked items by key. Fails
/// closed if the artifact is missing or the restore is lossy.
fn plan_arguments_done(
    next_items: &mut [ClientToolStreamItem],
    completions: &[ClientToolCompletion],
    payload: &Value,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(key) = client_tool_event_key(payload) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    let Some(tracked) = next_items.iter_mut().find(|tracked| tracked.key == key) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    if !is_restorable(tracked.restore) {
        return Ok(ClientToolDisposition::Passthrough);
    }
    tracked.phase = ClientToolPhase::ArgsComplete;
    match tracked.restore {
        // Namespace args pass through unchanged; only the phase advances.
        ClientToolRestore::Namespace => Ok(ClientToolDisposition::Passthrough),
        // Custom/NamespaceCustom: synthesize the canonical custom input event pair.
        ClientToolRestore::Custom | ClientToolRestore::NamespaceCustom => {
            plan_custom_arguments_done(tracked, completions, &key)
        },
        // Shell/ToolSearch: synthesize the complete typed `output_item.added` now
        // that arguments are final (the raw added was suppressed).
        ClientToolRestore::Shell | ClientToolRestore::ToolSearch => {
            plan_typed_arguments_done(tracked, completions, &key)
        },
    }
}

/// Synthesize a `Custom`/`NamespaceCustom` call's `EmitCustomInput` disposition at
/// `function_call_arguments.done`.
///
/// The canonical lifecycle replaces the backend's function-args frame with a
/// `custom_tool_call_input` delta+done pair. The completion artifact captured at
/// accumulation is authoritative for the arguments; fail closed if it is absent
/// (cannot synthesize truthfully) or its arguments are not an `{"input":...}`
/// envelope. The synthesized event references the PUBLIC `ctc_` id, never `fc_`.
fn plan_custom_arguments_done(
    tracked: &ClientToolStreamItem,
    completions: &[ClientToolCompletion],
    key: &str,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(completion) = completions.iter().find(|completion| completion.key == key) else {
        return Err(client_tool_restore_error(
            &tracked.key,
            "function_call_arguments.done without a captured completion artifact",
        ));
    };
    let arguments = completion
        .item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let input = input_from_arguments_strict(arguments).map_err(|()| {
        client_tool_restore_error(
            &tracked.key,
            "lowered custom-tool arguments are not a {\"input\":...} envelope",
        )
    })?;
    let raw_id = tracked
        .item_id
        .as_deref()
        .or_else(|| completion.item.get("id").and_then(Value::as_str))
        .unwrap_or_default();
    Ok(ClientToolDisposition::EmitCustomInput {
        key: tracked.key.clone(),
        item_id: custom_public_item_id(raw_id),
        output_index: tracked.output_index,
        input,
    })
}

/// Synthesize a `Shell`/`ToolSearch` call's `EmitTypedAdded` disposition at
/// `function_call_arguments.done`. The completion artifact captured at accumulation is
/// authoritative; fail closed if it is absent or the typed restore is lossy. The
/// synthesized `output_item.added` carries the restored typed call with its public id.
fn plan_typed_arguments_done(
    tracked: &ClientToolStreamItem,
    completions: &[ClientToolCompletion],
    key: &str,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(completion) = completions.iter().find(|c| c.key == key) else {
        return Err(client_tool_restore_error(
            &tracked.key,
            "function_call_arguments.done without a captured completion artifact",
        ));
    };
    let restored = restore_typed_item(tracked.restore, &completion.item, &tracked.key)?;
    Ok(ClientToolDisposition::EmitTypedAdded {
        item: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": tracked.output_index,
            "item": restored,
        }),
    })
}

/// Restore a lowered `Shell`/`ToolSearch` `function_call` item to its typed
/// `shell_call`/`tool_search_call`. Fails closed (with a client-tool restore error
/// shaped by `key`) if the kind is not a typed restore or the restore is lossy — a
/// lowered call is never surfaced under a schema-invalid typed item.
fn restore_typed_item(restore: ClientToolRestore, item: &Value, key: &str) -> Result<Value, SseParseError> {
    match restore {
        ClientToolRestore::Shell => restore_shell_call(item),
        ClientToolRestore::ToolSearch => restore_tool_search_call(item),
        _ => {
            return Err(client_tool_restore_error(
                key,
                "typed restore invoked for a non-typed restore kind",
            ));
        },
    }
    .map_err(|()| client_tool_restore_error(key, "lowered client tool could not be restored to its typed call"))
}

/// Build the fail-closed error for a client-tool restore violation (#1159).
///
/// `key` identifies the offending tool call (or `"response-snapshot"` /
/// `"terminal"` for whole-snapshot failures); `reason` is a specific,
/// client-safe description. Never carries a private lowered name.
fn client_tool_restore_error(key: &str, reason: &str) -> SseParseError {
    SseParseError::ClientToolRestore {
        key: key.to_owned(),
        reason: reason.to_owned(),
    }
}

/// Whether an untracked `output_item.done` carries a known lowered private name
/// (#1159 C1). `reverse` is keyed by the private lowered name, which is exactly what
/// the raw `function_call` item carries, so a hit means tracking was lost and the
/// caller must fail closed rather than pass the raw lowered shape through.
fn untracked_done_carries_lowered_name(payload: &Value, reverse: &HashMap<String, LoweredClientTool>) -> bool {
    payload
        .get("item")
        .and_then(|item| item.get("name"))
        .and_then(Value::as_str)
        .is_some_and(|name| reverse.contains_key(name))
}

/// Plan an `output_item.done`: finalize a tracked `Namespace`/`Custom`/
/// `NamespaceCustom` item. `Namespace` retypes the `function_call` in place;
/// `Custom`/`NamespaceCustom` rebuild the finalized `custom_tool_call`. Fail closed
/// if the finalizer arrives before `arguments.done` (C4) or a custom restore is
/// lossy.
fn plan_output_item_done(
    reverse: &HashMap<String, LoweredClientTool>,
    next_items: &mut [ClientToolStreamItem],
    payload: &Value,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(key) = client_tool_event_key(payload) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    let Some(tracked) = next_items.iter_mut().find(|tracked| tracked.key == key) else {
        // #1159 C1 defense-in-depth: an untracked item whose name is a known lowered
        // private name means tracking was lost (e.g. its `.added` was rolled back by
        // an earlier failed chunk). Never pass the raw lowered `function_call`
        // through — fail closed instead of surfacing its private name.
        if untracked_done_carries_lowered_name(payload, reverse) {
            return Err(client_tool_restore_error(
                "output-item-done",
                "untracked lowered client-tool item at output_item.done",
            ));
        }
        return Ok(ClientToolDisposition::Passthrough);
    };
    if !is_restorable(tracked.restore) {
        return Ok(ClientToolDisposition::Passthrough);
    }

    // C4 lifecycle-order check: an `output_item.done` before `arguments.done`
    // (item still `Opened`) is a malformed lifecycle; fail closed.
    if tracked.phase == ClientToolPhase::Opened {
        return Err(client_tool_restore_error(
            &tracked.key,
            "output_item.done before arguments.done",
        ));
    }
    tracked.phase = ClientToolPhase::Done;
    let restore = tracked.restore;

    // Restore the original member name and namespace from the lowering map keyed
    // by the private name the backend returned; never leak the lowered name.
    let Some(lowered) = reverse.get(tracked.private_name.as_str()) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    plan_done_disposition(payload, lowered, restore)
}

/// Select the `output_item.done` disposition for a finalized lowered call: retype a
/// `Namespace` member's `function_call` in place, rebuild the finalized
/// `custom_tool_call` for `Custom`/`NamespaceCustom`, or rebuild the finalized typed
/// `shell_call`/`tool_search_call` for `Shell`/`ToolSearch` — all from the
/// authoritative done item. Fails closed if a restore is lossy.
fn plan_done_disposition(
    payload: &Value,
    lowered: &LoweredClientTool,
    restore: ClientToolRestore,
) -> Result<ClientToolDisposition, SseParseError> {
    match restore {
        ClientToolRestore::Namespace => Ok(ClientToolDisposition::RetypeInPlace {
            item_type: "function_call",
            name: lowered.original_name.clone(),
            namespace: lowered.namespace.clone(),
        }),
        // Custom/NamespaceCustom: rebuild the finalized `custom_tool_call` from the
        // authoritative done item and splice it back onto the done envelope.
        ClientToolRestore::Custom | ClientToolRestore::NamespaceCustom => {
            Ok(ClientToolDisposition::EmitCustomItemDone {
                item: build_custom_done_payload(payload, lowered, restore)?,
            })
        },
        // Shell/ToolSearch: rebuild the finalized typed call from the authoritative
        // done item and splice it back onto the done envelope.
        ClientToolRestore::Shell | ClientToolRestore::ToolSearch => Ok(ClientToolDisposition::EmitTypedDone {
            item: build_typed_done_payload(payload, restore)?,
        }),
    }
}

/// Build the finalized `custom_tool_call` `output_item.done` payload for a lowered
/// `Custom`/`NamespaceCustom` call.
///
/// Restores the finalized typed item (`restore_custom_call` for `Custom`,
/// `restore_namespace_custom_call` for `NamespaceCustom` — the latter overwrites
/// the private lowered name with the member name and re-adds the namespace) and
/// splices it onto a single owned clone of the committed done envelope (the plan
/// pass only borrows the event — AGENTS.md ownership boundary). Fails closed if the
/// item cannot be restored (missing `call_id`, un-unwrappable input).
fn build_custom_done_payload(
    payload: &Value,
    lowered: &LoweredClientTool,
    restore: ClientToolRestore,
) -> Result<Value, SseParseError> {
    let Some(item) = payload.get("item") else {
        return Err(client_tool_restore_error(
            &lowered.original_name,
            "output_item.done carried no item to restore",
        ));
    };
    let restored = match restore {
        ClientToolRestore::NamespaceCustom => {
            restore_namespace_custom_call(item, &lowered.original_name, lowered.namespace.as_deref())
        },
        // Plain Custom keeps the client's own name (the wire name is unobfuscated).
        _ => restore_custom_call(item),
    }
    .map_err(|()| {
        client_tool_restore_error(
            &lowered.original_name,
            "lowered custom-tool call could not be restored to custom_tool_call",
        )
    })?;
    let mut out = payload.clone();
    if let Some(object) = out.as_object_mut() {
        object.insert("item".to_owned(), restored);
    }
    Ok(out)
}

/// Build the finalized typed `output_item.done` payload for a lowered `Shell`/`ToolSearch`
/// call: restore the typed item from the authoritative done item and splice it onto a single
/// owned clone of the committed done envelope. Fail closed if the item cannot be restored.
fn build_typed_done_payload(payload: &Value, restore: ClientToolRestore) -> Result<Value, SseParseError> {
    let Some(item) = payload.get("item") else {
        return Err(client_tool_restore_error(
            "shell/tool_search",
            "output_item.done carried no item to restore",
        ));
    };
    let restored = restore_typed_item(restore, item, "shell/tool_search")?;
    let mut out = payload.clone();
    if let Some(object) = out.as_object_mut() {
        object.insert("item".to_owned(), restored);
    }
    Ok(out)
}

/// Stable key matching a lowered client-tool item across its lifecycle events.
///
/// `output_item.added`/`.done` carry the id nested under `item`; the
/// `function_call_arguments.*` events carry a top-level `item_id`. Prefer either
/// id form, falling back to `output_index`, so one key matches the whole
/// lifecycle. Mirrors [`super::accumulator::tool_call_key`]'s
/// `item:{id}`/`index:{n}` shape.
fn client_tool_event_key(payload: &Value) -> Option<String> {
    let nested_id = payload
        .get("item")
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str);
    let top_id = payload.get("item_id").and_then(Value::as_str);
    if let Some(id) = nested_id.or(top_id) {
        return Some(format!("item:{id}"));
    }
    payload
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|output_index| format!("index:{output_index}"))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "test assertions favor direct unwrap/index/panic for clear failures"
)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::openai::responses::state::{ClientToolRestore, LoweredClientTool};

    fn reverse_namespace() -> HashMap<String, LoweredClientTool> {
        let mut m = HashMap::new();
        m.insert(
            "agentic_ns__fs__read".to_owned(),
            LoweredClientTool {
                original_name: "read".to_owned(),
                namespace: Some("fs".to_owned()),
                restore: ClientToolRestore::Namespace,
            },
        );
        m
    }

    fn reverse_custom() -> HashMap<String, LoweredClientTool> {
        let mut m = HashMap::new();
        m.insert(
            "run_python".to_owned(),
            LoweredClientTool {
                original_name: "run_python".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );
        m
    }

    fn reverse_shell() -> HashMap<String, LoweredClientTool> {
        let mut m = HashMap::new();
        m.insert(
            "shell".to_owned(),
            LoweredClientTool {
                original_name: "shell".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Shell,
            },
        );
        m
    }

    fn reverse_tool_search() -> HashMap<String, LoweredClientTool> {
        let mut m = HashMap::new();
        m.insert(
            "tool_search".to_owned(),
            LoweredClientTool {
                original_name: "tool_search".to_owned(),
                namespace: None,
                restore: ClientToolRestore::ToolSearch,
            },
        );
        m
    }

    #[test]
    fn custom_added_retypes_to_custom_tool_call_and_suppresses_backend_args() {
        let reverse = reverse_custom();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1"}
        }));
        let args_delta = ResponsesEvent::FunctionCallArgumentsDelta(serde_json::json!({
            "type": "response.function_call_arguments.delta", "item_id": "fc_1",
            "output_index": 0, "delta": "{\"input\":\"pri"
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"input\":\"print(1)\"}"
        }));
        // Completion artifact captured at args-done, cloned from accumulator storage.
        let completions = vec![ClientToolCompletion {
            key: "item:fc_1".to_owned(),
            item: serde_json::json!({
                "type": "function_call", "name": "run_python", "call_id": "c1",
                "id": "fc_1", "arguments": "{\"input\":\"print(1)\"}"
            }),
        }];
        let events = vec![added, args_delta, args_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &completions).unwrap();
        assert_eq!(plan.dispositions.len(), 3);
        assert!(matches!(
            plan.dispositions[0],
            ClientToolDisposition::EmitCustomShell { .. }
        ));
        assert!(matches!(plan.dispositions[1], ClientToolDisposition::Suppress));
        // args.done becomes the synthesized custom input delta+done pair, applied as EmitCustomInput.
        assert!(matches!(
            plan.dispositions[2],
            ClientToolDisposition::EmitCustomInput { .. }
        ));

        // The retyped added item is a `custom_tool_call` carrying the public id, and
        // the synthesized input carries the unwrapped plain-string value.
        match &plan.dispositions[0] {
            ClientToolDisposition::EmitCustomShell { item } => {
                let added_item = item.get("item").unwrap();
                assert_eq!(added_item.get("type").unwrap(), "custom_tool_call");
                assert_eq!(added_item.get("name").unwrap(), "run_python");
                assert_eq!(added_item.get("id").unwrap(), "ctc_1");
            },
            other => panic!("expected EmitCustomShell, got {other:?}"),
        }
        match &plan.dispositions[2] {
            ClientToolDisposition::EmitCustomInput { item_id, input, .. } => {
                assert_eq!(item_id, "ctc_1");
                assert_eq!(input, "print(1)");
            },
            other => panic!("expected EmitCustomInput, got {other:?}"),
        }
        // The private lowered call id is never surfaced on a client-visible id.
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::ArgsComplete));
    }

    #[test]
    fn custom_output_item_done_restores_custom_tool_call() {
        let reverse = reverse_custom();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"input\":\"print(1)\"}"
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"input\":\"print(1)\"}", "status": "completed"}
        }));
        let completions = vec![ClientToolCompletion {
            key: "item:fc_1".to_owned(),
            item: serde_json::json!({
                "type": "function_call", "name": "run_python", "call_id": "c1",
                "id": "fc_1", "arguments": "{\"input\":\"print(1)\"}"
            }),
        }];
        let events = vec![added, args_done, item_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &completions).unwrap();
        assert_eq!(plan.dispositions.len(), 3);
        assert!(matches!(
            plan.dispositions[0],
            ClientToolDisposition::EmitCustomShell { .. }
        ));
        assert!(matches!(
            plan.dispositions[1],
            ClientToolDisposition::EmitCustomInput { .. }
        ));
        match &plan.dispositions[2] {
            ClientToolDisposition::EmitCustomItemDone { item } => {
                let done_item = item.get("item").unwrap();
                assert_eq!(done_item.get("type").unwrap(), "custom_tool_call");
                assert_eq!(done_item.get("input").unwrap(), "print(1)");
                assert_eq!(done_item.get("id").unwrap(), "ctc_1");
                assert_eq!(done_item.get("call_id").unwrap(), "c1");
            },
            other => panic!("expected EmitCustomItemDone, got {other:?}"),
        }
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::Done));
    }

    #[test]
    fn custom_output_item_done_before_args_done_fails_closed() {
        let reverse = reverse_custom();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1"}
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"input\":\"print(1)\"}", "status": "completed"}
        }));
        // output_item.done while still `Opened` (no arguments.done) is a C4 violation.
        let events = [added, item_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(
            result.is_err(),
            "custom output_item.done before arguments.done must fail closed"
        );
    }

    #[test]
    fn custom_args_done_without_artifact_fails_closed() {
        let reverse = reverse_custom();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"input\":\"print(1)\"}"
        }));
        // No completion artifact captured (empty slice): the custom lifecycle cannot
        // be synthesized authoritatively, so fail closed rather than guess.
        let events = [added, args_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(
            result.is_err(),
            "custom arguments.done without a completion artifact must fail closed"
        );
    }

    #[test]
    fn shell_lifecycle_suppresses_raw_and_synthesizes_typed_shell_call() {
        let reverse = reverse_shell();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1"}
        }));
        let args_delta = ResponsesEvent::FunctionCallArgumentsDelta(serde_json::json!({
            "type": "response.function_call_arguments.delta", "item_id": "fc_1",
            "output_index": 0, "delta": "{\"commands\":"
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"commands\":[\"ls\"]}"
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"commands\":[\"ls\"]}", "status": "completed"}
        }));
        // Phase-2a completion artifact for the args.done key (item:fc_1).
        let completion = ClientToolCompletion {
            key: "item:fc_1".to_owned(),
            item: serde_json::json!({
                "type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1",
                "arguments": "{\"commands\":[\"ls\"]}", "status": "completed"
            }),
        };
        let events = vec![added, args_delta, args_done, item_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &[completion]).unwrap();

        assert!(matches!(plan.dispositions[0], ClientToolDisposition::Suppress)); // raw added
        assert!(matches!(plan.dispositions[1], ClientToolDisposition::Suppress)); // args delta
        match &plan.dispositions[2] {
            // args.done → typed added
            ClientToolDisposition::EmitTypedAdded { item } => {
                assert_eq!(item["type"], "response.output_item.added");
                assert_eq!(item["item"]["type"], "shell_call");
                assert_eq!(item["item"]["environment"]["type"], "local");
                assert_eq!(item["item"]["action"]["commands"][0], "ls");
                assert_eq!(item["item"]["id"], "sh_1"); // public id, not fc_1
            },
            other => panic!("expected EmitTypedAdded, got {other:?}"),
        }
        match &plan.dispositions[3] {
            // item.done → typed done
            ClientToolDisposition::EmitTypedDone { item } => {
                assert_eq!(item["type"], "response.output_item.done");
                assert_eq!(item["item"]["type"], "shell_call");
                assert_eq!(item["item"]["environment"]["type"], "local");
                assert_eq!(item["item"]["id"], "sh_1");
            },
            other => panic!("expected EmitTypedDone, got {other:?}"),
        }
    }

    #[test]
    fn tool_search_lifecycle_suppresses_raw_and_synthesizes_typed_tool_search_call() {
        let reverse = reverse_tool_search();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1"}
        }));
        let args_delta = ResponsesEvent::FunctionCallArgumentsDelta(serde_json::json!({
            "type": "response.function_call_arguments.delta", "item_id": "fc_1",
            "output_index": 0, "delta": "{\"query\":"
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"query\":\"rust\"}"
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "tool_search", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"query\":\"rust\"}", "status": "completed"}
        }));
        let completion = ClientToolCompletion {
            key: "item:fc_1".to_owned(),
            item: serde_json::json!({
                "type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1",
                "arguments": "{\"query\":\"rust\"}", "status": "completed"
            }),
        };
        let events = vec![added, args_delta, args_done, item_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &[completion]).unwrap();

        assert!(matches!(plan.dispositions[0], ClientToolDisposition::Suppress)); // raw added
        assert!(matches!(plan.dispositions[1], ClientToolDisposition::Suppress)); // args delta
        match &plan.dispositions[2] {
            // args.done → typed added
            ClientToolDisposition::EmitTypedAdded { item } => {
                assert_eq!(item["type"], "response.output_item.added");
                assert_eq!(item["item"]["type"], "tool_search_call");
                assert_eq!(item["item"]["execution"], "client");
                assert_eq!(item["item"]["arguments"]["query"], "rust");
                let id = item["item"]["id"].as_str().unwrap();
                assert!(id.starts_with("tsc_"), "expected public tsc_ id, got {id}");
            },
            other => panic!("expected EmitTypedAdded, got {other:?}"),
        }
        match &plan.dispositions[3] {
            // item.done → typed done
            ClientToolDisposition::EmitTypedDone { item } => {
                assert_eq!(item["type"], "response.output_item.done");
                assert_eq!(item["item"]["type"], "tool_search_call");
                assert_eq!(item["item"]["execution"], "client");
                let id = item["item"]["id"].as_str().unwrap();
                assert!(id.starts_with("tsc_"), "expected public tsc_ id, got {id}");
            },
            other => panic!("expected EmitTypedDone, got {other:?}"),
        }
    }

    #[test]
    fn shell_output_item_done_before_args_done_fails_closed() {
        let reverse = reverse_shell();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1"}
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"commands\":[\"ls\"]}", "status": "completed"}
        }));
        // output_item.done while still `Opened` (no arguments.done) is a C4 violation.
        let events = [added, item_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(
            result.is_err(),
            "shell output_item.done before arguments.done must fail closed"
        );
    }

    #[test]
    fn shell_args_done_without_artifact_fails_closed() {
        let reverse = reverse_shell();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"commands\":[\"ls\"]}"
        }));
        // No completion artifact captured (empty slice): the typed shell_call cannot
        // be synthesized authoritatively, so fail closed rather than guess.
        let events = [added, args_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(
            result.is_err(),
            "shell arguments.done without a completion artifact must fail closed"
        );
    }

    #[test]
    fn tool_search_lossy_typed_restore_fails_closed() {
        let reverse = reverse_tool_search();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done", "item_id": "fc_1",
            "output_index": 0, "arguments": "{\"query\":\"rust\"}"
        }));
        // Completion artifact is present, but it carries a `namespace`: a client-owned
        // tool_search_call must not be namespaced, so `restore_tool_search_call`
        // returns Err and the plan pass fails closed rather than emit a lossy item.
        let completion = ClientToolCompletion {
            key: "item:fc_1".to_owned(),
            item: serde_json::json!({
                "type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1",
                "namespace": "fs", "arguments": "{\"query\":\"rust\"}", "status": "completed"
            }),
        };
        let events = [added, args_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[completion]);
        assert!(
            result.is_err(),
            "a lossy typed restore (namespaced tool_search) must fail closed"
        );
    }

    #[test]
    fn native_passthrough_plans_nothing() {
        let reverse = HashMap::new();
        let plan = plan_client_tool_restore(&reverse, None, &[], &[], &[]).unwrap();
        assert!(plan.dispositions.is_empty());
        assert!(plan.next_items.is_empty());
    }

    #[test]
    fn namespace_output_item_added_retypes_in_place() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let plan = plan_client_tool_restore(&reverse, None, &[], std::slice::from_ref(&added), &[]).unwrap();
        assert_eq!(plan.dispositions.len(), 1);
        match &plan.dispositions[0] {
            ClientToolDisposition::RetypeInPlace {
                item_type,
                name,
                namespace,
            } => {
                assert_eq!(*item_type, "function_call");
                assert_eq!(name, "read");
                assert_eq!(namespace.as_deref(), Some("fs"));
            },
            other => panic!("expected RetypeInPlace, got {other:?}"),
        }
        assert_eq!(plan.next_items.len(), 1);
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::Opened));
    }

    #[test]
    fn namespace_full_lifecycle_advances_and_retypes_done() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"path\":\"/etc\"}"
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1", "status": "completed"}
        }));
        let events = [added, args_done, item_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &[]).unwrap();

        assert_eq!(plan.dispositions.len(), 3);
        assert!(matches!(
            plan.dispositions[0],
            ClientToolDisposition::RetypeInPlace { .. }
        ));
        assert!(matches!(plan.dispositions[1], ClientToolDisposition::Passthrough));
        match &plan.dispositions[2] {
            ClientToolDisposition::RetypeInPlace { name, namespace, .. } => {
                assert_eq!(name, "read");
                assert_eq!(namespace.as_deref(), Some("fs"));
            },
            other => panic!("expected RetypeInPlace on done, got {other:?}"),
        }
        assert_eq!(plan.next_items.len(), 1);
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::Done));
        // The private lowered name is tracked internally but never surfaces in a
        // client-visible disposition.
        assert_eq!(plan.next_items[0].private_name, "agentic_ns__fs__read");
    }

    #[test]
    fn namespace_output_item_done_before_args_done_fails_closed() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1", "status": "completed"}
        }));
        // output_item.done arriving while the item is still `Opened` (no
        // arguments.done seen) is a C4 lifecycle-order violation: fail closed.
        let events = [added, item_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(
            result.is_err(),
            "output_item.done before arguments.done must fail closed"
        );
    }

    #[test]
    fn in_progress_event_restores_snapshot_tools() {
        let reverse = reverse_custom();
        let echo = ClientToolEcho {
            tools: vec![serde_json::json!({"type": "custom", "name": "run_python"})],
            tool_choice: Value::Null,
        };
        let created = ResponsesEvent::ResponseCreated(serde_json::json!({
            "type": "response.created",
            "response": {
                "object": "response",
                "tools": [{"type": "function", "name": "run_python"}],
                "tool_choice": "auto"
            }
        }));
        let plan = plan_client_tool_restore(&reverse, Some(&echo), &[], std::slice::from_ref(&created), &[]).unwrap();
        match &plan.dispositions[0] {
            ClientToolDisposition::RestoreSnapshot { response } => {
                assert_eq!(
                    response["tools"],
                    serde_json::json!([{"type": "custom", "name": "run_python"}])
                );
                assert!(response.get("tool_choice").is_none());
            },
            other => panic!("expected RestoreSnapshot, got {other:?}"),
        }
    }

    #[test]
    fn in_progress_event_restores_snapshot_with_present_tool_choice() {
        let reverse = reverse_custom();
        let echo = ClientToolEcho {
            tools: vec![serde_json::json!({"type": "custom", "name": "run_python"})],
            tool_choice: serde_json::json!({"type": "custom", "name": "run_python"}),
        };
        let in_progress = ResponsesEvent::ResponseInProgress(serde_json::json!({
            "type": "response.in_progress",
            "response": {
                "object": "response",
                "tools": [{"type": "function", "name": "run_python"}],
                "tool_choice": {"type": "function", "name": "run_python"}
            }
        }));
        let plan =
            plan_client_tool_restore(&reverse, Some(&echo), &[], std::slice::from_ref(&in_progress), &[]).unwrap();
        match &plan.dispositions[0] {
            ClientToolDisposition::RestoreSnapshot { response } => {
                assert_eq!(
                    response["tools"],
                    serde_json::json!([{"type": "custom", "name": "run_python"}])
                );
                assert_eq!(
                    response["tool_choice"],
                    serde_json::json!({"type": "custom", "name": "run_python"})
                );
            },
            other => panic!("expected RestoreSnapshot, got {other:?}"),
        }
    }

    #[test]
    fn intermediate_snapshot_with_lossy_output_fails_closed() {
        let mut reverse = HashMap::new();
        reverse.insert(
            "run_python".to_owned(),
            LoweredClientTool {
                original_name: "run_python".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );
        let echo = ClientToolEcho {
            tools: vec![serde_json::json!({"type": "custom", "name": "run_python"})],
            tool_choice: Value::Null,
        };
        // A ResponseInProgress whose response.output contains a lossy lowered item
        // (custom function_call with no call_id, which restore_custom_call rejects).
        let in_progress = ResponsesEvent::ResponseInProgress(serde_json::json!({
            "type": "response.in_progress",
            "response": {
                "object": "response",
                "tools": [{"type": "function", "name": "run_python"}],
                "tool_choice": "auto",
                "output": [
                    {"type": "function_call", "name": "run_python", "id": "fc_1"}
                ]
            }
        }));
        let result = plan_client_tool_restore(&reverse, Some(&echo), &[], &[in_progress], &[]);
        assert!(
            result.is_err(),
            "non-terminal snapshot with lossy output item must fail closed"
        );
    }

    // #1159 C1 defense-in-depth: an `output_item.done` for a lowered item that is
    // NOT tracked in `next_items` (its `.added` was rolled back by an earlier failed
    // chunk) must fail closed rather than pass the raw lowered `function_call`
    // through, because `reverse` still knows the private name.
    #[test]
    fn untracked_output_item_done_with_lowered_name_fails_closed() {
        let reverse = reverse_custom();
        // No prior tracking and no matching `.added` in this batch: the item is
        // untracked, but its name is a known lowered private name.
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1",
                     "id": "fc_1", "arguments": "{\"input\":\"print(1)\"}", "status": "completed"}
        }));
        let result = plan_client_tool_restore(&reverse, None, &[], &[item_done], &[]);
        assert!(
            result.is_err(),
            "an untracked lowered item at output_item.done must fail closed, not leak its raw name"
        );
    }

    // #1159 C1 defense-in-depth companion: a genuine non-lowered item whose name is
    // NOT in `reverse` must still pass through untouched at `output_item.done`.
    #[test]
    fn untracked_output_item_done_native_name_passes_through() {
        let reverse = reverse_custom();
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done", "output_index": 0,
            "item": {"type": "function_call", "name": "native_tool", "call_id": "c2",
                     "id": "fc_2", "arguments": "{}", "status": "completed"}
        }));
        let plan = plan_client_tool_restore(&reverse, None, &[], &[item_done], &[]).unwrap();
        assert_eq!(plan.dispositions.len(), 1);
        assert!(
            matches!(plan.dispositions[0], ClientToolDisposition::Passthrough),
            "a genuine non-lowered item must still pass through unchanged"
        );
    }

    // #1159 C1 defense-in-depth: a lowered `output_item.added` with no resolvable
    // event key (no nested item id, no `output_index`) cannot be tracked, so its
    // later `.done` could not be matched — fail closed instead of passing the raw
    // lowered name through untracked.
    #[test]
    fn lowered_output_item_added_without_event_key_fails_closed() {
        let reverse = reverse_custom();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1"}
        }));
        let result = plan_client_tool_restore(&reverse, None, &[], &[added], &[]);
        assert!(
            result.is_err(),
            "a lowered added without a resolvable event key must fail closed"
        );
    }
}
