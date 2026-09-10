//! Reusable low-level injection primitive for `openai_stream_events` (#313 §4).
//!
//! `file_search` is the first and only consumer today; #276 will add MCP and
//! `web_search` as further consumers of the same pieces. Classification and
//! payload construction are per-consumer; suppression, dual-key registration,
//! and synthesis sequencing are shared here.

use std::collections::HashMap;

use praxis_filter::HttpFilterContext;
use serde_json::Value;

use super::{encode_sse_event, normalize_logical_payload};
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
pub(crate) fn synthesize_private_complete(item: &Value, round_local_index: u64) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(round_local_index.into());

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
/// Reuses the opening's `round_local_index`. Clone is necessary — synthesis payload.
pub(crate) fn synthesize_native_complete_tail(item: &Value, round_local_index: u64) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(round_local_index.into());

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
pub(crate) fn synthesize_incomplete_tail(
    item: &Value,
    round_local_index: u64,
    private: bool,
) -> Vec<(&'static str, Value)> {
    let idx = || Value::Number(round_local_index.into());

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

/// §4.2 precedence: pre-existing error → validate all → error-contract → clean-emit.
/// Takes only `&mut ctx`; `ResponsesState` is reacquired from ctx.extensions in
/// phases so the two borrows never overlap.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: drain queue → pre-existing error check → validate all indices → fail-closed error → clean emit; each phase isolated"
)]
pub(super) fn drain_local_tool_synthesis(ctx: &mut HttpFilterContext<'_>, output_index_offset: u64, out: &mut Vec<u8>) {
    type SynthesisResolution = Result<Vec<(u64, SynthesisKind, Value)>, &'static str>;

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
    // Phase B — validate + resolve every index against the committed output,
    // holding an immutable state borrow only for the duration of this block.
    let resolved: SynthesisResolution = (|| {
        let state = ctx
            .extensions
            .get::<ResponsesState>()
            .ok_or("openai_stream_events: file_search synthesis missing state")?;
        let mut resolved = Vec::with_capacity(queue.len());
        for (absolute, kind) in &queue {
            // Step 2: validate via checked u64 conversion + checked_sub + range.
            let round_local = u64::try_from(*absolute)
                .ok()
                .and_then(|a| a.checked_sub(output_index_offset))
                .ok_or("openai_stream_events: file_search synthesis index invariant")?;
            let item = state
                .accumulated_output
                .get(*absolute)
                .cloned() // owned copy for the freshly-built synthesis payload; the
                          // original stays in accumulated_output for the terminal frame.
                .ok_or("openai_stream_events: file_search synthesis index out of range")?;
            resolved.push((round_local, *kind, item));
        }
        Ok(resolved)
    })();
    let resolved = match resolved {
        Ok(resolved) => resolved,
        // Step 3: invariant violation → five-write, discard, no frame.
        Err(message) => {
            fs_end_stream_with_error_ctx(ctx, "server_error", message);
            return;
        },
    };
    // Step 4: clean path — build (kind from the queue), then normalize + encode.
    // The state borrow is dropped; only &mut ctx is held here.
    for (round_local, kind, item) in resolved {
        for (event_type, mut payload) in select_builder(&item, round_local, kind) {
            normalize_logical_payload(ctx, &mut payload, output_index_offset);
            encode_sse_event(event_type, &payload, out);
        }
    }
}

/// Dispatch to the Task 17 builders by origin `kind` + item `status`.
fn select_builder(item: &Value, round_local: u64, kind: SynthesisKind) -> Vec<(&'static str, Value)> {
    let completed = item.get("status").and_then(Value::as_str) == Some("completed");
    match (kind, completed) {
        (SynthesisKind::Private, true) => synthesize_private_complete(item, round_local),
        (SynthesisKind::Native, true) => synthesize_native_complete_tail(item, round_local),
        // Incomplete (terminalized / open-failure) tail; private replays its opening.
        (kind, false) => synthesize_incomplete_tail(item, round_local, kind == SynthesisKind::Private),
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
}
