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

pub(crate) mod input_items;

pub(crate) use input_items::{InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, list_input_items};
use serde_json::Value;
use tracing::warn;

use crate::{
    openai::responses::append_stored_input_items,
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
    /// populated it.
    pub(crate) fn build_record(
        response_object: Value,
        owner: StateOwner,
        request_input: Option<Value>,
        state_messages: Option<Vec<Value>>,
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

        let capture = ResponseCapture::from_response_json(&response_object, request_input, state_messages);

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
    fn from_response_json(json: &Value, request_input: Option<Value>, state_messages: Option<Vec<Value>>) -> Self {
        let input = request_input
            .or_else(|| json.get("input").cloned())
            .unwrap_or(Value::Null);
        let history_input = state_messages.map_or_else(|| input.clone(), Value::Array);
        let messages = assemble_stored_messages(history_input, json.get("output"));

        Self { input, messages }
    }
}

/// Build the stored conversation history from response input and output.
fn assemble_stored_messages(input: Value, output: Option<&Value>) -> Value {
    let mut messages = Vec::new();

    append_stored_input_items(&mut messages, input);

    match output {
        Some(Value::Array(items)) => messages.extend(items.iter().cloned()),
        Some(output) if !output.is_null() => messages.push(output.clone()),
        Some(_) | None => {},
    }

    Value::Array(messages)
}
