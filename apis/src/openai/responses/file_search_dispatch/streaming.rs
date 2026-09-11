// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming `file_search` EOS path (#313 §6). The buffered `capture_response`
//! linear path is untouched; this module handles the `stream: true` branch.

use praxis_filter::{FilterAction, FilterError, HttpFilterContext};
use serde_json::Value;

use super::{
    FNV_OFFSET_BASIS, citations::annotate_output_items, ensure_public_output_item_ids, has_client_function_call,
    has_file_search_tool, is_file_search_function_call, is_pending_file_search_call, remaining_file_search_call_budget,
    stable_call_hash, terminalize_all_pending_calls, translate_function_calls_to_file_search,
};
use crate::{
    callout_policy::OnFailure,
    openai::responses::{
        bounded_json_size,
        state::{ResponsesState, SynthesisKind},
        streamed_round_is_dispatchable,
    },
};

/// Process-wide ceiling on concurrently parked `block_in_place` `file_search`
/// bridges (§7.2). Fixed constant — the Pingora worker count is deployment-set,
/// so this is a safety cap, not a fraction of cores. TEMPORARY: removed with the
/// `block_in_place` bridge at praxis#1105.
pub(super) const FILE_SEARCH_EOS_ADMISSION: usize = 8;

/// Semaphore limiting concurrent EOS bridge operations. See [`FILE_SEARCH_EOS_ADMISSION`].
pub(super) static FILE_SEARCH_EOS_ADMISSION_SEM: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(FILE_SEARCH_EOS_ADMISSION);

/// Admission control result for the streaming EOS bridge.
pub(super) enum Admission<'a> {
    /// Permit acquired; bridge may proceed.
    Admitted(tokio::sync::SemaphorePermit<'a>),
    /// Overloaded, fail-closed: error already set.
    ShedClosed,
    /// Overloaded, fail-open: degrade gracefully.
    ShedOpen,
}

/// `sem` is a parameter (not a hardwired static) purely for test isolation; the
/// production call passes `&FILE_SEARCH_EOS_ADMISSION_SEM`. `on_failure` is the
/// FILTER's policy (`self.on_failure`), passed in because `ResponsesState` carries
/// no such field. `state` is threaded through only so the closed branch can write
/// the five-key fail-closed contract.
pub(super) fn admit_or_shed<'a>(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    sem: &'a tokio::sync::Semaphore,
    on_failure: OnFailure,
) -> Admission<'a> {
    match sem.try_acquire() {
        Ok(permit) => Admission::Admitted(permit),
        Err(_) => match on_failure {
            OnFailure::Closed => {
                fs_end_stream_with_error(
                    ctx,
                    state,
                    "server_busy",
                    "openai_file_search_dispatch: file_search admission overloaded",
                );
                Admission::ShedClosed
            },
            OnFailure::Open => Admission::ShedOpen,
        },
    }
}

/// file_search-side fail-closed: clear transient per-round state, then apply the
/// ctx-level five-write contract. (`web_search_calls` clearing is a later task's.)
pub(super) fn fs_end_stream_with_error(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    code: &str,
    message: &str,
) {
    state.tool_calls.clear();
    crate::openai::responses::fs_end_stream_with_error_ctx(ctx, code, message);
}

/// STEP 0 (§6): authoritative-success gate. Returns `true` if the gate-fail path
/// handled the round (caller returns Continue); `false` if the gate holds and the
/// caller proceeds to STEP 0.5/branch. Child-module `pub(super)` free fn.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: error-terminal branch + upstream-incomplete branch, both with shared annotate + size check"
)]
pub(super) fn step_0_gate(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    max_json_body_bytes: usize,
) -> bool {
    if streamed_round_is_dispatchable(ctx, state) {
        return false; // gate holds → run branches (side effects allowed)
    }
    // An error is authoritative for this round and must publish unmodified — either our
    // own parse/validation error (`stream_error_code` set), OR a flat upstream `error`
    // completion (`stream_completion == "error"`, which sets NO `stream_error_code`
    // because there is no `response` object to size, §6). Arm the two-layer stop first
    // so a stale `action="loop"` cannot suppress the error frame or fire another IRR
    // round (P1), then bypass STEP 0.5/annotate/terminal-bound so a local failure never
    // replaces the provider's error with our generic `server_error` (P2).
    if ctx.get_metadata("responses.stream_error_code").is_some()
        || ctx.get_metadata("responses.stream_completion") == Some("error")
    {
        crate::openai::responses::fs_arm_stream_stop(ctx);
        return true;
    }
    // Non-error terminal (incomplete/failed). Fall through to the shared annotate + size
    // bound so any prior-round accumulation is finalized correctly even when THIS round
    // carries an empty `output` (F2: an empty terminal round must still annotate and
    // bound the partial accumulated by earlier BRANCH A rounds). When this round does
    // carry output, sanitize + close pending calls below first (the loop is a no-op on
    // an empty round).
    let has_file_search = has_file_search_tool(state);
    let round = std::mem::take(state.output_items_mut()); // move, valid [] left
    for item in round {
        if has_file_search && is_file_search_function_call(&item) {
            continue; // drop suppressed private call (only when hosted file_search injected it) (F3)
        }
        let mut item = item;
        // Close a still-pending native hybrid item as incomplete (P1 round-8).
        if is_pending_file_search_call(&item) {
            if let Some(o) = item.as_object_mut() {
                o.insert("status".to_owned(), Value::String("incomplete".to_owned()));
                o.remove("results");
            }
            let idx = state.accumulated_output.len();
            state.accumulated_output.push(item);
            // Always Native: STEP 0 runs pre-translate, so no private-origin call can
            // reach here (raw file_search function_calls were dropped just above).
            state.pending_local_tool_synthesis.push((idx, SynthesisKind::Native));
            continue;
        }
        state.accumulated_output.push(item);
    }
    // Shared BRANCH B annotate + full-canonical bound; local failure → error frame.
    // Split so each check publishes its own diagnostic (matches `branch_b_terminal`).
    if annotate_output_items(&mut state.accumulated_output, &state.citation_files).is_err() {
        fs_end_stream_with_error(
            ctx,
            state,
            "server_error",
            "openai_file_search_dispatch: gate-fail citation annotation failed",
        );
    } else if !terminal_payload_fits(state, max_json_body_bytes) {
        fs_end_stream_with_error(
            ctx,
            state,
            "server_error",
            "openai_file_search_dispatch: gate-fail terminal response exceeds byte limit",
        );
    }
    true
}

/// STEP 0.5 (§6): translate private `file_search` `function_calls` (infallible),
/// then the mixed-tool gate — both before `build_search_plan`, matching the
/// buffered order (translate mod.rs:362 → mixed-tool mod.rs:373 → execute).
/// Returns the private-origin round-local indices for BRANCH A/B to tag Private.
pub(super) fn step_0_5_translate_and_mixed_tool(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
) -> Result<Vec<usize>, ()> {
    let translated = if has_file_search_tool(state) {
        translate_function_calls_to_file_search(&mut state.response_object)
    } else {
        Vec::new()
    };
    let response_identity_hash = state
        .response_object
        .get("id")
        .and_then(Value::as_str)
        .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
    if let Some(output) = state.response_object.get_mut("output").and_then(Value::as_array_mut) {
        ensure_public_output_item_ids(output, response_identity_hash);
    }
    // `reconcile_round_into_accumulated_output` uses `binary_search` on `translated`,
    // which requires ascending order. `translate_function_calls_to_file_search` produces
    // it via forward `iter_mut().enumerate()`; guard that invariant against a refactor.
    debug_assert!(
        translated.windows(2).all(|w| matches!(w, [a, b] if a < b)),
        "translated indices must be strictly ascending for binary_search"
    );
    let output = state.output_items();
    let mixed = output.iter().any(is_pending_file_search_call) && has_client_function_call(output);
    if mixed {
        fs_end_stream_with_error(
            ctx,
            state,
            "server_error",
            "openai_file_search_dispatch: response mixes file_search with a client tool call",
        );
        return Err(());
    }
    Ok(translated)
}

/// STEP 0.6 (§6): when the cross-round budget is exhausted, terminalize every
/// pending call to `incomplete` BEFORE planning (mirrors `capture_response`'s
/// buffered pending handling), so `build_search_plan` finds no executable pending
/// calls and BRANCH B is taken — no admission, no search, no iteration bump.
pub(super) fn step_0_6_apply_budget(state: &mut ResponsesState) {
    if remaining_file_search_call_budget(state) == 0 {
        terminalize_all_pending_calls(state);
    }
}

/// True for every item this round that `apply_batch` reconciled into a
/// `file_search_call` — executed→`completed` or over-cap→`incomplete`. Both need
/// EOS synthesis; non-`file_search_call` items (assistant message, reasoning) were
/// streamed live and are terminal content, so they are skipped.
fn is_file_search_call_item(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("file_search_call")
}

/// D-ACC: move this round out of `output_items` (leaving a valid `[]`), append each
/// item to `accumulated_output`, and queue each reconciled call's absolute index with
/// its synthesis origin. `translated` holds the round-local indices STEP 0.5 rewrote
/// from `function_call` → `file_search_call` (private origin), sorted ascending;
/// anything else needing synthesis is native. NB: BRANCH A does NOT annotate (§6 step 4).
///
/// A native `file_search_call` whose terminal lifecycle the provider already streamed
/// live is recorded (by id) in `state.provider_streamed_terminal_ids` by `stream_events`
/// — its terminal `output_item.done` passed through, cancelling suppression. Re-queuing
/// it would emit a DUPLICATE `output_item.done` tail at EOS (#313 P1), so it is skipped
/// for ANY provider status (completed/failed/incomplete): observed membership, not the
/// status string, is authoritative. Private calls always synthesize (their opening was
/// suppressed); a callout-terminalized `incomplete` call is absent from the set (its
/// live done was suppressed, never passed through) and still queues its tail.
pub(super) fn reconcile_round_into_accumulated_output(state: &mut ResponsesState, translated: &[usize]) {
    let response_identity_hash = state
        .response_object
        .get("id")
        .and_then(Value::as_str)
        .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
    ensure_public_output_item_ids(state.output_items_mut(), response_identity_hash);

    let base = state.accumulated_output.len();
    let round = std::mem::take(state.output_items_mut()); // owned Vec, leaves valid []
    for (i, item) in round.into_iter().enumerate() {
        if is_file_search_call_item(&item) {
            // `translated` is sorted ascending, so binary_search is O(log m), not the
            // O(m) linear `contains` scan (F6: quadratic under an adversarial round).
            let is_private = translated.binary_search(&i).is_ok();
            let observed_provider_terminal = item
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| state.provider_streamed_terminal_ids.contains(id));
            // Skip only a NATIVE call the provider already streamed to terminal live;
            // private calls (opening suppressed) always synthesize.
            let skip_synthesis = observed_provider_terminal && !is_private;
            if !skip_synthesis {
                let kind = if is_private {
                    SynthesisKind::Private
                } else {
                    SynthesisKind::Native
                };
                state.pending_local_tool_synthesis.push((base + i, kind));
            }
        }
        state.accumulated_output.push(item);
    }
}

/// BRANCH B (§6): terminal round — the model's final answer OR a zero-budget round
/// whose pending calls were all terminalized to `incomplete`. Reconcile → annotate
/// (single terminal pass) → full-canonical bound → publish pending=false. Child-module
/// `pub(super)` free fn: only `capture_streaming_inner` and this file's tests call it.
pub(super) fn branch_b_terminal(
    ctx: &mut HttpFilterContext<'_>,
    state: &mut ResponsesState,
    translated: &[usize],
    max_json_body_bytes: usize,
) -> Result<FilterAction, FilterError> {
    // 1. Reuse the Task 13 reconcile: move this round into accumulated_output and queue any file_search_call (only
    //    zero-budget-terminalized `incomplete` ones exist here, since no search ran) with its absolute index +
    //    SynthesisKind from `translated`. Provider-streamed natives are skipped by observation, not by status.
    reconcile_round_into_accumulated_output(state, translated);

    // 2. Terminal citations — the SOLE annotation pass.
    if annotate_output_items(&mut state.accumulated_output, &state.citation_files).is_err() {
        fs_end_stream_with_error(
            ctx,
            state,
            "server_error",
            "openai_file_search_dispatch: citation annotation failed",
        );
        return Ok(FilterAction::Continue);
    }

    // 3. Full-canonical-object size bound (no clone: swap fields in, size, swap out).
    if !terminal_payload_fits(state, max_json_body_bytes) {
        fs_end_stream_with_error(
            ctx,
            state,
            "server_error",
            "openai_file_search_dispatch: terminal response exceeds byte limit",
        );
        return Ok(FilterAction::Continue);
    }

    // 4. Publish pending=false only (no action=loop, no reset, no iteration bump).
    ctx.filter_results
        .entry("openai_file_search_dispatch")
        .or_default()
        .set("pending", "false")?;
    Ok(FilterAction::Continue)
}

/// Size the object `finalize_response_body` will serialize (`response_object` with
/// output = annotated `accumulated_output`, plus usage), move-not-clone: swap BOTH
/// fields in, size, swap BOTH back — state is restored intact on every path.
/// BRANCH B step 1 already normalized `response_object` to an object via
/// `output_items_mut()`, so `as_object_mut()` is `Some` on the happy path.
#[expect(
    clippy::too_many_lines,
    reason = "linear sequence: swap both fields in → size → swap both back, restoring any displaced key on every path"
)]
fn terminal_payload_fits(state: &mut ResponsesState, max_bytes: usize) -> bool {
    let output = std::mem::take(&mut state.accumulated_output); // leaves []
    let usage = std::mem::take(&mut state.usage); // leaves Null
    let usage_absent = usage.is_null();

    let Some(obj) = state.response_object.as_object_mut() else {
        state.accumulated_output = output; // defensive restore; treat as "does not fit"
        state.usage = usage;
        return false;
    };
    let prev_output = obj.insert("output".to_owned(), Value::Array(output));
    let prev_usage = if usage_absent {
        None
    } else {
        obj.insert("usage".to_owned(), usage)
    };

    let fits = bounded_json_size(&state.response_object, max_bytes)
        .ok()
        .flatten()
        .is_some();

    // Swap both fields back out, restoring any pre-existing key value we displaced.
    #[expect(
        clippy::expect_used,
        reason = "response_object normalized to object above via `let Some(obj) =` + early-return when None"
    )]
    let obj = state.response_object.as_object_mut().expect("normalized above");
    if let Some(Value::Array(v)) = match prev_output {
        Some(old) => obj.insert("output".to_owned(), old),
        None => obj.remove("output"),
    } {
        state.accumulated_output = v;
    }
    if !usage_absent {
        state.usage = match prev_usage {
            Some(old) => obj.insert("usage".to_owned(), old),
            None => obj.remove("usage"),
        }
        .unwrap_or(Value::Null);
    }
    fits
}
