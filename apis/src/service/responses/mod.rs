// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses persistence service.
//!
//! Owner-scoped business logic over the persisted-state store: record assembly,
//! input-item listing, CRUD, and pending-approval coordination. The service is
//! constructed from an already owner-scoped store handle resolved by the caller,
//! constructs no backend and holds no connection pool, and returns transport-
//! neutral [`StoreError`]s. The transport layer owns request decoding, HTTP input
//! validation, and status mapping.

use std::ops::RangeInclusive;

pub(crate) mod input_items;

pub(crate) use input_items::{InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, list_input_items};
use serde_json::Value;
use tracing::warn;

use crate::{
    openai::responses::{append_stored_input_items, state::CollectedRound},
    state_owner::StateOwner,
    store::{OwnerScopedResponseStore, PendingApprovalRecord, ResponseRecord, StoreError},
};

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

/// Owner-scoped Responses persistence service.
///
/// Bound to one validated owner at construction through [`OwnerScopedResponseStore`],
/// so no operation can widen or forge the owner scope. Persistence still passes the
/// facade's owner check on the record.
pub(crate) struct ResponsesService {
    /// Owner-bound store handle the service operates through.
    store: OwnerScopedResponseStore,
}

impl ResponsesService {
    /// Bind the service to an owner-scoped store the caller resolved from the
    /// registry (`get_scoped(name, &owner)`).
    #[must_use]
    pub(crate) fn new(store: OwnerScopedResponseStore) -> Self {
        Self { store }
    }

    /// Retrieve a stored response visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when the backend query fails.
    pub(crate) async fn get(&self, id: &str) -> Result<Option<ResponseRecord>, StoreError> {
        self.store.get_response(id).await
    }

    /// Delete a stored response visible to this owner.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when the backend mutation fails.
    pub(crate) async fn delete(&self, id: &str) -> Result<bool, StoreError> {
        self.store.delete_response(id).await
    }

    /// Persist a response and the pending approvals it issued in one transaction.
    ///
    /// The record's owner is re-checked against the bound owner by the facade.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] on owner mismatch or a backend failure.
    pub(crate) async fn persist(
        &self,
        record: &ResponseRecord,
        pending_approvals: &[PendingApprovalRecord],
    ) -> Result<(), StoreError> {
        self.store
            .persist_response_with_pending_approvals(record, pending_approvals)
            .await
    }

    /// Persist a response record, re-checking its owner against the bound owner.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] on owner mismatch or a backend failure.
    // Called only by the compact filter; absent when that feature is off.
    #[cfg(feature = "openai-compact")]
    pub(crate) async fn upsert(&self, record: &ResponseRecord) -> Result<(), StoreError> {
        self.store.upsert_response(record).await
    }

    /// Load pending approvals issued to this owner for a response.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when the backend query fails.
    // Called by the mcp-dispatch filter in production and by the service tests;
    // absent in a store build that enables neither.
    #[cfg(any(feature = "openai-mcp-tools", test))]
    pub(crate) async fn get_pending_approvals(
        &self,
        response_id: &str,
        approval_ids: &[&str],
    ) -> Result<Vec<PendingApprovalRecord>, StoreError> {
        self.store.get_pending_approvals(response_id, approval_ids).await
    }

    /// Atomically consume approvals issued to this owner.
    ///
    /// Returns `Ok(None)` when all ids were consumed, or `Ok(Some(index))` for
    /// the first already-consumed id (a replay), consuming none of the batch.
    ///
    /// # Errors
    ///
    /// Returns a [`StoreError`] when the backend mutation fails.
    // Called by the mcp-dispatch filter in production and by the service tests;
    // absent in a store build that enables neither.
    #[cfg(any(feature = "openai-mcp-tools", test))]
    pub(crate) async fn consume_approvals(
        &self,
        response_id: &str,
        approval_ids: &[&str],
        consumed_at: i64,
    ) -> Result<Option<usize>, StoreError> {
        self.store
            .consume_approvals(response_id, approval_ids, consumed_at)
            .await
    }

    /// Assemble a persisted [`ResponseRecord`] from a completed Responses API
    /// response object.
    ///
    /// Returns `None` when the object is null (an incomplete stream) or is
    /// missing a required field (`id`, `created_at`, `model`), i.e. it is not
    /// persistable. `request_input` is the original create-request `input`;
    /// `state_messages` is the accumulated persistence history when rehydrate
    /// populated it. `plan` carries the agentic collection metadata used to
    /// reconcile stored history so collected output is not duplicated by the
    /// final append, and to place late reasoning ahead of its turn.
    pub(crate) fn build_record(
        response_object: Value,
        owner: StateOwner,
        request_input: Option<Value>,
        state_messages: Option<Vec<Value>>,
        plan: StoredOutputPlan<'_>,
    ) -> Option<ResponseRecord> {
        if response_object.is_null() {
            warn!("response persistence: response_object is null (incomplete stream?)");
            return None;
        }

        let id = response_object.get("id").and_then(Value::as_str);
        let created_at = response_object.get("created_at").and_then(Value::as_i64);
        let model = response_object.get("model").and_then(Value::as_str);

        let (Some(id), Some(created_at), Some(model)) = (id, created_at, model) else {
            warn!("response persistence: missing required field (id, created_at, or model)");
            return None;
        };

        let capture = ResponseCapture::from_response_json(&response_object, request_input, state_messages, plan);

        Some(ResponseRecord {
            id: id.to_owned(),
            owner,
            created_at,
            model: model.to_owned(),
            response_object,
            input: capture.input,
            messages: capture.messages,
        })
    }
}

/// Stored input and message history extracted from a Responses API exchange.
struct ResponseCapture {
    /// Original request input used by rehydration.
    input: Value,

    /// Full message history used by rehydration.
    messages: Value,
}

impl ResponseCapture {
    /// Extract stored input and output from a Responses API exchange.
    fn from_response_json(
        json: &Value,
        request_input: Option<Value>,
        state_messages: Option<Vec<Value>>,
        plan: StoredOutputPlan<'_>,
    ) -> Self {
        let input = request_input
            .or_else(|| json.get("input").cloned())
            .unwrap_or(Value::Null);
        let history_input = state_messages.map_or_else(|| input.clone(), Value::Array);
        let messages = assemble_stored_messages(history_input, json.get("output"), plan);

        Self { input, messages }
    }
}

/// Agentic collection metadata storage assembly needs to reconcile the stored
/// history with the final response output without duplicating collected items.
///
/// `collected_rounds` empty means no output collector ran (the non-agentic path),
/// so assembly appends the response output verbatim and normalizes late reasoning
/// over the appended slice.
#[derive(Clone, Copy)]
pub(crate) struct StoredOutputPlan<'a> {
    /// Absolute `accumulated_output` ranges a translated Chat turn spans, ending
    /// at that turn's late reasoning item.
    pub reasoning_replay: &'a [RangeInclusive<usize>],
    /// One entry per agentic round an output collector processed, in order.
    pub collected_rounds: &'a [CollectedRound],
    /// `(accumulated_output index, persisted_messages index)` for each collected
    /// output item.
    pub collected_provenance: &'a [(usize, usize)],
}

impl StoredOutputPlan<'_> {
    /// An empty plan: no collector ran, so output is appended verbatim.
    pub(crate) const EMPTY: StoredOutputPlan<'static> = StoredOutputPlan {
        reasoning_replay: &[],
        collected_rounds: &[],
        collected_provenance: &[],
    };
}

/// Build the stored conversation history from response input and output.
///
/// When no agentic collector ran, the response output is appended to the input
/// history and late reasoning is rotated to the front of its turn. When a
/// collector ran, the history already contains the collected output interleaved
/// with tool results; assembly rebuilds it, dropping the now-duplicate collected
/// items, reinserting the uncollected ones (assistant messages, any other type)
/// into their original rounds, and moving late reasoning ahead of its turn.
fn assemble_stored_messages(input: Value, output: Option<&Value>, plan: StoredOutputPlan<'_>) -> Value {
    let mut history = Vec::new();
    append_stored_input_items(&mut history, input);

    let output_items: Vec<Value> = match output {
        Some(Value::Array(items)) => items.clone(),
        Some(item) if !item.is_null() => vec![item.clone()],
        Some(_) | None => Vec::new(),
    };

    if plan.collected_rounds.is_empty() {
        // Non-agentic path: history is input-only; append the output and rotate
        // late reasoning over just the appended slice.
        let output_start = history.len();
        history.extend(output_items);
        if let Some(appended) = history.get_mut(output_start..) {
            rotate_trailing_reasoning(appended, plan.reasoning_replay);
        }
        return Value::Array(history);
    }

    Value::Array(rebuild_collected_history(&history, &output_items, plan))
}

/// Rotate a turn's late reasoning to its front within the freshly appended output
/// slice. Each range ends at the turn's reasoning item regardless of any trailing
/// tool calls, so only that item moves and the wire output is unchanged.
fn rotate_trailing_reasoning(items: &mut [Value], reasoning_replay: &[RangeInclusive<usize>]) {
    for range in reasoning_replay {
        if let Some(turn) = items.get_mut(*range.start()..=*range.end())
            && turn.last().and_then(|item| item.get("type")).and_then(Value::as_str) == Some("reasoning")
        {
            turn.rotate_right(1);
        }
    }
}

/// Rebuild stored history from the collected rounds, reconciling it with the
/// final output so each collected item appears once.
///
/// `history` is the persisted history the collectors built (input items,
/// collected output items, and dispatch tool results, interleaved). `output_items`
/// is the final response output (`accumulated_output`). The provenance map and
/// round boundaries are coordinates in the ORIGINAL `history`/`output_items`, so
/// the rebuild emits into a fresh vector in one forward pass to keep those
/// coordinates valid.
#[expect(
    clippy::too_many_lines,
    reason = "one forward pass over rounds, inter-round artifacts, and the uncollected final round"
)]
fn rebuild_collected_history(history: &[Value], output_items: &[Value], plan: StoredOutputPlan<'_>) -> Vec<Value> {
    // accumulated_output index -> persisted_messages index for collected items.
    let provenance: std::collections::HashMap<usize, usize> = plan.collected_provenance.iter().copied().collect();
    let ctx = RebuildContext {
        output_items,
        provenance: &provenance,
        reasoning_replay: plan.reasoning_replay,
    };
    let mut rebuilt = Vec::with_capacity(history.len());

    let first_start = plan
        .collected_rounds
        .first()
        .map_or(history.len(), |round| round.persisted_start);
    rebuilt.extend(history.get(..first_start).unwrap_or_default().iter().cloned());

    for (index, round) in plan.collected_rounds.iter().enumerate() {
        let window = history
            .get(round.persisted_start..round.persisted_end)
            .unwrap_or_default();
        rebuilt.extend(reconstruct_round(
            window,
            round.output_start..round.output_end,
            round.persisted_start,
            &ctx,
        ));

        // Tool results and other dispatch artifacts up to the next round start
        // (or the end of history for the last round) stay where the loop left them.
        let next_start = plan
            .collected_rounds
            .get(index + 1)
            .map_or(history.len(), |next| next.persisted_start);
        rebuilt.extend(
            history
                .get(round.persisted_end..next_start)
                .unwrap_or_default()
                .iter()
                .cloned(),
        );
    }

    // An entirely uncollected final round: output items beyond the last recorded
    // span were never persisted, so append them (reordered) at the end of history.
    let last_output_end = plan.collected_rounds.last().map_or(0, |round| round.output_end);
    if last_output_end < output_items.len() {
        rebuilt.extend(reconstruct_round(
            &[],
            last_output_end..output_items.len(),
            history.len(),
            &ctx,
        ));
    }

    rebuilt
}

/// Shared inputs for reconstructing a round: the final output, the collected-item
/// provenance, and the translator replay ranges.
struct RebuildContext<'a> {
    /// The final response output (`accumulated_output`).
    output_items: &'a [Value],
    /// `accumulated_output` index -> `persisted_messages` index for collected items.
    provenance: &'a std::collections::HashMap<usize, usize>,
    /// Absolute `accumulated_output` ranges a translated turn spans, ending at its
    /// late reasoning item.
    reasoning_replay: &'a [RangeInclusive<usize>],
}

/// Reconstruct one round's assistant-turn items: every persisted window position
/// is preserved, each collected item is refreshed to its final output copy (so a
/// file-search item updated by dispatch after collection is stored as the client
/// saw it), every uncollected output item in the round's span is reinserted in its
/// original output order, and the turn's late reasoning is moved ahead of its turn
/// (named items only, no-op on mismatch).
fn reconstruct_round(
    window: &[Value],
    output_span: std::ops::Range<usize>,
    persisted_start: usize,
    ctx: &RebuildContext<'_>,
) -> Vec<Value> {
    // Start from the full window so no collected item (or any item without
    // provenance) is ever dropped; only uncollected output items are inserted.
    let mut items: Vec<Value> = window.to_vec();
    // output index -> position within `items`, used to move named reasoning.
    let mut positions: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut inserted = 0;
    let mut cursor = 0;

    for output_index in output_span.clone() {
        if let Some(&persisted_index) = ctx.provenance.get(&output_index) {
            // Collected: already in the window. Collected items are persisted in
            // output order, so window positions are monotonic; `inserted` accounts
            // for uncollected items placed ahead of this one.
            let position = persisted_index.saturating_sub(persisted_start) + inserted;
            // Replace the collector's pre-dispatch copy with the final output item
            // at the same position: file-search dispatch and terminalization update
            // an item's `accumulated_output` copy (status, results) after collection,
            // so the stored history must track the same final item the client saw.
            if let (Some(slot), Some(final_item)) = (items.get_mut(position), ctx.output_items.get(output_index)) {
                *slot = final_item.clone();
            }
            positions.insert(output_index, position);
            cursor = position + 1;
        } else if let Some(item) = ctx.output_items.get(output_index) {
            let position = cursor.min(items.len());
            items.insert(position, item.clone());
            positions.insert(output_index, position);
            inserted += 1;
            cursor = position + 1;
        }
    }

    debug_assert_eq!(
        items.len(),
        window.len() + inserted,
        "every window item must be preserved and every uncollected output item inserted exactly once"
    );

    move_named_reasoning_first(&mut items, &positions, output_span, ctx.reasoning_replay);
    items
}

/// Move the reasoning item each replay range names ahead of its turn, within the
/// reconstructed round. Leaves history untouched when the named item is no longer
/// a reasoning item (a mismatch), never reordering by a guessed match.
fn move_named_reasoning_first(
    items: &mut Vec<Value>,
    positions: &std::collections::HashMap<usize, usize>,
    output_span: std::ops::Range<usize>,
    reasoning_replay: &[RangeInclusive<usize>],
) {
    for range in reasoning_replay {
        if !(output_span.contains(range.start()) && output_span.contains(range.end())) {
            continue;
        }
        let (Some(&start_pos), Some(&end_pos)) = (positions.get(range.start()), positions.get(range.end())) else {
            continue;
        };
        let names_reasoning = items
            .get(end_pos)
            .and_then(|item| item.get("type"))
            .and_then(Value::as_str)
            == Some("reasoning");
        if end_pos > start_pos && end_pos < items.len() && names_reasoning {
            let reasoning = items.remove(end_pos);
            items.insert(start_pos, reasoning);
        }
    }
}
