//! Reusable low-level injection primitive for `openai_stream_events` (#313 §4).
//!
//! `file_search` is the first and only consumer today; #276 will add MCP and
//! `web_search` as further consumers of the same pieces. Classification and
//! payload construction are per-consumer; suppression, dual-key registration,
//! and synthesis sequencing are shared here.

use std::collections::HashMap;

use praxis_filter::HttpFilterContext;
use serde::{Serialize, ser::SerializeMap as _};
use serde_json::Value;

use super::{encode_sse_event, normalize_logical_payload, take_logical_sequence};
use crate::openai::responses::{
    fs_end_stream_with_error_ctx,
    state::{ResponsesState, SynthesisKind},
};

/// Per-item suppression mode recorded at `output_item.added`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalToolMode {
    /// Drop every event for the item (a fully private `function_call`).
    Suppress,
    /// Pass the opening + progress; drop only a still-pending `output_item.done`
    /// (a native `file_search_call` resolved by the proxy at EOS).
    NativeHybridPending,
}

/// Record both dual keys (`item:{id}` and `index:{output_index}`) for an item
/// opened by a raw, pre-normalization `output_item.added` payload.
pub(crate) fn register_local_tool(
    map: &mut HashMap<String, LocalToolMode>,
    added_payload: &Value,
    mode: LocalToolMode,
) {
    if let Some(id) = added_payload
        .get("item")
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str)
    {
        map.insert(format!("item:{id}"), mode);
    }
    if let Some(index) = added_payload.get("output_index").and_then(Value::as_u64) {
        map.insert(format!("index:{index}"), mode);
    }
}

/// Resolve the suppression-map key(s) an incoming event carries. Events carry
/// either a nested `item.id` (`output_item.added/.done`) or a flat `item_id`
/// (argument events), plus an `output_index` when present.
pub(crate) fn event_local_tool_keys(payload: &Value) -> impl Iterator<Item = String> + '_ {
    let nested_id = payload
        .get("item")
        .and_then(|item| item.get("id"))
        .and_then(Value::as_str);
    let flat_id = payload.get("item_id").and_then(Value::as_str);
    let index = payload.get("output_index").and_then(Value::as_u64);
    nested_id
        .or(flat_id)
        .map(|id| format!("item:{id}"))
        .into_iter()
        .chain(index.map(|i| format!("index:{i}")))
}

/// Synthesize a complete private `file_search` lifecycle (5 events: opening → progress → completion → done).
/// Clone is necessary here — these are freshly-constructed synthesis payloads, not provider passthrough.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: opening + 3 progress events + done, each with its own payload construction"
)]
#[cfg(test)]
pub(crate) fn synthesize_private_complete(item: &Value, output_index: u64) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(output_index.into());

    // Opening item: clone with status → "searching", results removed
    let mut opening_item = item.clone();
    if let Some(obj) = opening_item.as_object_mut() {
        obj.insert("status".to_owned(), Value::String("searching".to_owned()));
        obj.remove("results");
    }

    let mut events = Vec::new();

    // 1. response.output_item.added
    let added = serde_json::json!({
        "type": "response.output_item.added",
        "output_index": idx(),
        "item": opening_item,
        "sequence_number": 0
    });
    events.push(("response.output_item.added", added));

    // 2-4. Progress events (in_progress, searching, completed)
    for event_type in [
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.file_search_call.completed",
    ] {
        let payload = if let Some(id) = item.get("id").and_then(Value::as_str) {
            serde_json::json!({
                "type": event_type,
                "output_index": idx(),
                "item_id": id,
                "sequence_number": 0
            })
        } else {
            serde_json::json!({
                "type": event_type,
                "output_index": idx(),
                "sequence_number": 0
            })
        };
        events.push((event_type, payload));
    }

    // 5. response.output_item.done (completed item with results)
    let done = serde_json::json!({
        "type": "response.output_item.done",
        "output_index": idx(),
        "item": item.clone(),
        "sequence_number": 0
    });
    events.push(("response.output_item.done", done));

    events
}

/// Synthesize the tail of a native complete `file_search` (2 events: completed → done).
/// Reuses the opening's `output_index`. Clone is necessary — synthesis payload.
#[cfg(test)]
pub(crate) fn synthesize_native_complete_tail(item: &Value, output_index: u64) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(output_index.into());

    let mut events = Vec::new();

    // 1. response.file_search_call.completed
    let completed = if let Some(id) = item.get("id").and_then(Value::as_str) {
        serde_json::json!({
            "type": "response.file_search_call.completed",
            "output_index": idx(),
            "item_id": id,
            "sequence_number": 0
        })
    } else {
        serde_json::json!({
            "type": "response.file_search_call.completed",
            "output_index": idx(),
            "sequence_number": 0
        })
    };
    events.push(("response.file_search_call.completed", completed));

    // 2. response.output_item.done
    let done = serde_json::json!({
        "type": "response.output_item.done",
        "output_index": idx(),
        "item": item.clone(),
        "sequence_number": 0
    });
    events.push(("response.output_item.done", done));

    events
}

/// Synthesize an incomplete tail (private: 4 events; native: 1 event).
/// Never emits `.completed`. Clone is necessary — synthesis payload.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: private branch (opening + 2 progress) + common terminal done; each with its own payload construction"
)]
#[cfg(test)]
pub(crate) fn synthesize_incomplete_tail(item: &Value, output_index: u64, private: bool) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(output_index.into());

    // Build the incomplete item: clone with status → "incomplete", results removed
    let mut incomplete_item = item.clone();
    if let Some(obj) = incomplete_item.as_object_mut() {
        obj.insert("status".to_owned(), Value::String("incomplete".to_owned()));
        obj.remove("results");
    }

    let mut events = Vec::new();

    if private {
        // Private incomplete: emit opening + progress events
        // 1. Opening item @ "searching", no results
        let mut opening_item = item.clone();
        if let Some(obj) = opening_item.as_object_mut() {
            obj.insert("status".to_owned(), Value::String("searching".to_owned()));
            obj.remove("results");
        }
        let added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": idx(),
            "item": opening_item,
            "sequence_number": 0
        });
        events.push(("response.output_item.added", added));

        // 2-3. Progress events (in_progress, searching)
        for event_type in [
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
        ] {
            let payload = if let Some(id) = item.get("id").and_then(Value::as_str) {
                serde_json::json!({
                    "type": event_type,
                    "output_index": idx(),
                    "item_id": id,
                    "sequence_number": 0
                })
            } else {
                serde_json::json!({
                    "type": event_type,
                    "output_index": idx(),
                    "sequence_number": 0
                })
            };
            events.push((event_type, payload));
        }
    }

    // Final event: response.output_item.done (incomplete, no results)
    let done = serde_json::json!({
        "type": "response.output_item.done",
        "output_index": idx(),
        "item": incomplete_item,
        "sequence_number": 0
    });
    events.push(("response.output_item.done", done));

    events
}

/// Drain the `file_search` EOS synthesis queue, deferring items the dispatcher
/// has not reconciled yet (#1046 §4.2 precedence).
///
/// The parse owner (`openai_agentic_loop`) queues each `file_search_call` by its
/// absolute `accumulated_output` index during the round it *parses* the call, but
/// the request-phase dispatcher (`openai_file_search_callout`) does not execute
/// and reconcile that call until the *next* IRR re-entry's request-body EOS. So at
/// the round-N finalize where the owner first queued it, the indexed item is still
/// pending and its lifecycle cannot be synthesized. This finalizer runs at every
/// round's response EOS, so it re-queues a still-pending item and synthesizes it at
/// the finalize of the round that reconciles it (once the dispatcher has stamped a
/// terminal `status` and attached results).
///
/// Absolute indices are stamped directly as `output_index` and normalized with a
/// zero offset — identical to the #276 local-item path — because
/// `accumulated_output` is the single logical response output array.
///
/// Takes only `&mut ctx`; `ResponsesState` is reacquired from ctx.extensions in
/// phases so the two borrows never overlap.
#[expect(
    clippy::too_many_lines,
    reason = "phased drain: partition ready vs deferred, then synthesize"
)]
pub(super) fn drain_local_tool_synthesis(ctx: &mut HttpFilterContext<'_>, out: &mut Vec<u8>) {
    type ResolvedItem = (usize, SynthesisKind);
    type DeferredItem = (usize, SynthesisKind);
    type SynthesisResolution = Result<(Vec<ResolvedItem>, Vec<DeferredItem>), &'static str>;

    // Phase A — take the queue out (drain-once), releasing the state borrow.
    let queue = match ctx.extensions.get_mut::<ResponsesState>() {
        Some(state) => state.drain_pending_local_tool_synthesis(),
        None => return,
    };
    if queue.is_empty() {
        return;
    }
    // Step 1: pre-existing error → discard without normalizing (preserves sequence).
    if ctx.get_metadata("responses.stream_error_code").is_some() {
        return;
    }
    // Phase B — partition the queue into items the dispatcher has already
    // reconciled (ready to synthesize now) and items still pending (re-queued for a
    // later finalize). Holds an immutable state borrow only for this block.
    let resolved: SynthesisResolution = (|| {
        let state = ctx
            .extensions
            .get::<ResponsesState>()
            .ok_or("openai_stream_events: file_search synthesis missing state")?;
        let mut ready: Vec<ResolvedItem> = Vec::with_capacity(queue.len());
        let mut deferred: Vec<DeferredItem> = Vec::new();
        for (absolute, kind) in &queue {
            let item = state
                .accumulated_output
                .get(*absolute)
                .ok_or("openai_stream_events: file_search synthesis index out of range")?;
            // The dispatcher stamps a terminal status (`completed`/`incomplete`) in
            // place once it executes the assigned call. Until then the owner's
            // canonical placeholder is still pending, so defer synthesis.
            let reconciled = matches!(
                item.get("status").and_then(Value::as_str),
                Some("completed" | "incomplete")
            );
            if reconciled {
                ready.push((*absolute, *kind));
            } else {
                deferred.push((*absolute, *kind));
            }
        }
        Ok((ready, deferred))
    })();
    let (ready, deferred) = match resolved {
        Ok(resolved) => resolved,
        // Invariant violation → five-write, discard, no frame.
        Err(message) => {
            fs_end_stream_with_error_ctx(ctx, "server_error", message);
            return;
        },
    };
    // Re-queue still-pending items so a later finalize (after the dispatcher
    // reconciles them next round) synthesizes their lifecycle.
    if !deferred.is_empty()
        && let Some(state) = ctx.extensions.get_mut::<ResponsesState>()
    {
        state.pending_local_tool_synthesis = deferred;
    }
    // Clean path — build, then normalize with a zero offset (indices are already
    // absolute) + encode. The state borrow is dropped; only &mut ctx is held here.
    for (output_index, kind) in ready {
        emit_borrowed_file_search_lifecycle(ctx, out, output_index, kind);
    }
}

/// Emit file-search lifecycle events from the canonical accumulator item.
/// Only fixed-size sequencing and a small item-id string are staged; item
/// envelopes borrow the complete output tree during serialization.
#[expect(
    clippy::too_many_lines,
    reason = "emits the ordered private and native file-search lifecycles"
)]
fn emit_borrowed_file_search_lifecycle(
    ctx: &mut HttpFilterContext<'_>,
    out: &mut Vec<u8>,
    item_index: usize,
    kind: SynthesisKind,
) {
    let (completed, item_id) = match ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|state| state.accumulated_output.get(item_index))
    {
        Some(item) => (
            item.get("status").and_then(Value::as_str) == Some("completed"),
            item.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
        ),
        None => return,
    };
    let output_index = u64::try_from(item_index).unwrap_or(u64::MAX);

    if kind == SynthesisKind::Private {
        encode_file_search_item_event(
            ctx,
            out,
            "response.output_item.added",
            output_index,
            item_index,
            Some("searching"),
            true,
        );
        for phase in [
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
        ] {
            encode_file_search_phase(ctx, out, phase, output_index, &item_id);
        }
    }

    if completed {
        encode_file_search_phase(ctx, out, "response.file_search_call.completed", output_index, &item_id);
    }

    encode_file_search_item_event(
        ctx,
        out,
        "response.output_item.done",
        output_index,
        item_index,
        (!completed).then_some("incomplete"),
        !completed,
    );
}

/// Encode one lightweight file-search progress phase.
fn encode_file_search_phase(
    ctx: &mut HttpFilterContext<'_>,
    out: &mut Vec<u8>,
    event_type: &str,
    output_index: u64,
    item_id: &str,
) {
    let mut payload = serde_json::json!({
        "type": event_type,
        "output_index": output_index,
        "item_id": item_id,
        "sequence_number": 0,
    });
    normalize_logical_payload(ctx, &mut payload, 0);
    encode_sse_event(event_type, &payload, out);
}

/// Encode one output-item envelope by borrowing the canonical item.
#[expect(
    clippy::too_many_arguments,
    reason = "wire envelope fields remain explicit at the two call sites"
)]
fn encode_file_search_item_event(
    ctx: &mut HttpFilterContext<'_>,
    out: &mut Vec<u8>,
    event_type: &str,
    output_index: u64,
    item_index: usize,
    status: Option<&str>,
    omit_results: bool,
) {
    let sequence_number = take_logical_sequence(ctx);
    let state = ctx.extensions.get::<ResponsesState>();
    let Some(item) = state.and_then(|state| state.accumulated_output.get(item_index)) else {
        return;
    };
    let payload = BorrowedFileSearchEnvelope {
        event_type,
        response_id: state.and_then(|state| state.logical_stream_response_id.as_deref()),
        output_index,
        item: FileSearchItemView {
            item,
            status,
            omit_results,
        },
        sequence_number,
    };
    encode_sse_event(event_type, &payload, out);
}

/// Borrowed output-item event envelope for synthesized file-search lifecycle.
#[derive(Serialize)]
struct BorrowedFileSearchEnvelope<'a> {
    #[serde(rename = "type")]
    /// SSE payload discriminator.
    event_type: &'a str,
    /// Logical response identifier, when known.
    response_id: Option<&'a str>,
    /// Absolute canonical output index.
    output_index: u64,
    /// Borrowed item view with lifecycle overrides.
    item: FileSearchItemView<'a>,
    /// Monotonic logical-stream sequence number.
    sequence_number: u64,
}

/// Serializer view that overrides status and optionally omits search results.
struct FileSearchItemView<'a> {
    /// Canonical file-search output item.
    item: &'a Value,
    /// Status substituted into the emitted view.
    status: Option<&'a str>,
    /// Whether the borrowed view suppresses the `results` field.
    omit_results: bool,
}

impl Serialize for FileSearchItemView<'_> {
    #[expect(
        clippy::too_many_lines,
        reason = "map serializer conditionally substitutes and omits fields"
    )]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let Some(object) = self.item.as_object() else {
            return self.item.serialize(serializer);
        };
        let mut len = object.len();
        if self.omit_results && object.contains_key("results") {
            len = len.saturating_sub(1);
        }
        if self.status.is_some() && !object.contains_key("status") {
            len = len.saturating_add(1);
        }
        let mut map = serializer.serialize_map(Some(len))?;
        for (key, value) in object {
            if self.omit_results && key == "results" {
                continue;
            }
            if key == "status" {
                if let Some(status) = self.status {
                    map.serialize_entry(key, status)?;
                } else {
                    map.serialize_entry(key, value)?;
                }
            } else {
                map.serialize_entry(key, value)?;
            }
        }
        if let Some(status) = self.status
            && !object.contains_key("status")
        {
            map.serialize_entry("status", status)?;
        }
        map.end()
    }
}

/// Dispatch to the Task 17 builders by origin `kind` + item `status`.
#[cfg(test)]
fn select_builder(item: &Value, output_index: u64, kind: SynthesisKind) -> Vec<(&'static str, Value)> {
    let completed = item.get("status").and_then(Value::as_str) == Some("completed");
    match (kind, completed) {
        (SynthesisKind::Private, true) => synthesize_private_complete(item, output_index),
        (SynthesisKind::Native, true) => synthesize_native_complete_tail(item, output_index),
        // Incomplete (terminalized / open-failure) tail; private replays its opening.
        (kind, false) => synthesize_incomplete_tail(item, output_index, kind == SynthesisKind::Private),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::indexing_slicing,
        clippy::str_to_string,
        clippy::unwrap_used,
        reason = "tests"
    )]

    use serde_json::json;

    use super::*;

    #[test]
    fn register_records_both_dual_keys() {
        let mut map = HashMap::new();
        let added = json!({
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {"id": "fc_123", "type": "function_call", "name": "file_search"}
        });
        register_local_tool(&mut map, &added, LocalToolMode::Suppress);
        assert_eq!(map.get("item:fc_123"), Some(&LocalToolMode::Suppress));
        assert_eq!(map.get("index:2"), Some(&LocalToolMode::Suppress));
    }

    #[test]
    fn event_keys_resolves_item_and_index() {
        // A function_call_arguments.delta carries only item_id.
        let delta = json!({"type": "response.function_call_arguments.delta", "item_id": "fc_123"});
        let keys: Vec<String> = event_local_tool_keys(&delta).collect();
        assert!(keys.contains(&"item:fc_123".to_string()));
        // An output_item.done carries a nested item.id and an output_index.
        let done = json!({"type": "response.output_item.done", "output_index": 2, "item": {"id": "fc_123"}});
        let keys: Vec<String> = event_local_tool_keys(&done).collect();
        assert!(keys.contains(&"item:fc_123".to_string()));
        assert!(keys.contains(&"index:2".to_string()));
    }

    #[test]
    fn private_complete_lifecycle_shape() {
        let item = json!({"type": "file_search_call", "status": "completed", "results": [{"file_id": "f1"}]});
        let events = synthesize_private_complete(&item, 0);
        let types: Vec<&str> = events.iter().map(|(t, _)| *t).collect();
        assert_eq!(
            types,
            vec![
                "response.output_item.added",
                "response.file_search_call.in_progress",
                "response.file_search_call.searching",
                "response.file_search_call.completed",
                "response.output_item.done",
            ]
        );
        assert_eq!(events[0].1["output_index"], json!(0));
        assert_eq!(events.last().unwrap().1["item"]["status"], json!("completed"));
    }

    #[test]
    fn native_tail_reuses_index_and_only_completes() {
        let item = json!({"type": "file_search_call", "status": "completed", "results": []});
        let events = synthesize_native_complete_tail(&item, 3);
        let types: Vec<&str> = events.iter().map(|(t, _)| *t).collect();
        assert_eq!(
            types,
            vec!["response.file_search_call.completed", "response.output_item.done"]
        );
        assert_eq!(
            events[0].1["output_index"],
            json!(3),
            "native tail reuses opening's index"
        );
    }

    #[test]
    fn incomplete_tail_has_no_results_no_completed() {
        let item = json!({"type": "file_search_call", "status": "incomplete"});
        let native = synthesize_incomplete_tail(&item, 2, false);
        assert_eq!(
            native.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec!["response.output_item.done"]
        );
        assert_eq!(native[0].1["item"]["status"], json!("incomplete"));
        assert!(
            native[0].1["item"].get("results").is_none(),
            "incomplete tail carries no results"
        );
        assert!(!native.iter().any(|(t, _)| *t == "response.file_search_call.completed"));
        let private = synthesize_incomplete_tail(&item, 2, true);
        assert!(
            private.iter().any(|(t, _)| *t == "response.output_item.added"),
            "private incomplete emits opening"
        );
        assert!(!private.iter().any(|(t, _)| *t == "response.file_search_call.completed"));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "allocation comparison constructs and measures both implementations"
    )]
    fn borrowed_synthesis_allocates_less_than_legacy_item_clones() {
        let make_item = || {
            json!({
                "type": "file_search_call",
                "id": "fs_large",
                "status": "completed",
                "results": [{"file_id": "f1", "text": "x".repeat(65_536)}]
            })
        };
        let legacy_item = make_item();
        let current_item = make_item();

        let legacy = allocation_counter::measure(|| {
            let events = select_builder(&legacy_item, 0, SynthesisKind::Private);
            let mut output = Vec::new();
            for (event_type, payload) in events {
                encode_sse_event(event_type, &payload, &mut output);
            }
            std::hint::black_box(output);
        });

        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context(Box::leak(Box::new(request)));
        ctx.extensions.insert(ResponsesState {
            accumulated_output: vec![current_item],
            ..ResponsesState::default()
        });
        let borrowed = allocation_counter::measure(|| {
            let mut output = Vec::new();
            emit_borrowed_file_search_lifecycle(&mut ctx, &mut output, 0, SynthesisKind::Private);
            std::hint::black_box(output);
        });

        assert!(
            borrowed.bytes_total + 65_536 < legacy.bytes_total,
            "borrowed synthesis must remove payload-sized item clones: legacy={} borrowed={}",
            legacy.bytes_total,
            borrowed.bytes_total
        );
    }
}
