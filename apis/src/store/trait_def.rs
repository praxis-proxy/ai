// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! The [`ResponseStore`] and [`ConversationItemStore`] async traits
//! for response persistence and conversation lifecycle/item persistence.

use async_trait::async_trait;

use super::types::{ConversationItemRecord, ConversationRecord, PendingApprovalRecord, ResponseRecord, StoreError};

// -----------------------------------------------------------------------------
// ResponseStore Trait
// -----------------------------------------------------------------------------

/// Async persistence layer for Responses API records.
///
/// Every query is tenant-scoped. Single-tenant deployments pass a
/// default sentinel (e.g., `"default"`) as the `tenant_id`.
///
/// `get_response` returns `None` for both "not found" and "wrong
/// tenant" to avoid information leakage.
///
/// Conversation access is read-only here so response rehydration can
/// load cached conversations without taking ownership of conversation
/// lifecycle mutations.
#[async_trait]
pub trait ResponseStore: Send + Sync {
    /// Insert or update a response record.
    ///
    /// Uses the record's [`id`] as the primary key. If a record
    /// with the same ID already exists, it is replaced entirely.
    ///
    /// [`id`]: ResponseRecord::id
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn upsert_response(&self, record: &ResponseRecord) -> Result<(), StoreError>;

    /// Retrieve a response by ID, scoped to a tenant.
    ///
    /// Returns `None` if the response does not exist or belongs
    /// to a different tenant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn get_response(&self, tenant_id: &str, id: &str) -> Result<Option<ResponseRecord>, StoreError>;

    /// Delete a response by ID, scoped to a tenant.
    ///
    /// Returns `true` if a record was deleted, `false` if no
    /// matching record existed for this tenant.
    ///
    /// Any server-owned pending approvals issued by the deleted response are
    /// removed in the same transaction, so deleting a response leaves no
    /// consumable approval behind and retains no sensitive tool arguments.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn delete_response(&self, tenant_id: &str, id: &str) -> Result<bool, StoreError>;

    /// Retrieve conversation messages by conversation ID and tenant.
    ///
    /// Returns `None` if the conversation does not exist or belongs
    /// to a different tenant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn get_conversation(
        &self,
        tenant_id: &str,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError>;

    /// Record server-owned pending MCP approvals emitted by the proxy.
    ///
    /// Called from the proxy **output** path the moment one or more
    /// `mcp_approval_request` items are emitted, this writes the sole source
    /// of truth for correlating a later `mcp_approval_response` back to the
    /// call the proxy actually paused on. Each record captures the complete
    /// resolved target (server label, tool name, arguments, fingerprint) with
    /// `consumed_at` left `NULL`.
    ///
    /// `response_id` is the id of the response that issued these approvals; it
    /// scopes every row so a later `mcp_approval_response` must supply the same
    /// originating `previous_response_id` to load or consume it.
    ///
    /// Writes are idempotent: a row that already exists for
    /// `(tenant_id, response_id, approval_id)` is left untouched, so re-emitting
    /// the same pending approval never resets an already-consumed row back to
    /// outstanding (which would enable replay).
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn record_pending_approvals(
        &self,
        tenant_id: &str,
        response_id: &str,
        records: &[PendingApprovalRecord],
        created_at: i64,
    ) -> Result<(), StoreError>;

    /// Persist a response and the pending approvals it issued together.
    ///
    /// Called from the proxy output path when a streamed or buffered response
    /// carries one or more `mcp_approval_request` items. The pending rows are
    /// scoped to `record.id` (via [`record_pending_approvals`]) so a later
    /// `mcp_approval_response` correlates back through `previous_response_id`.
    ///
    /// Transactional backends **must** commit both writes in a single
    /// transaction, serialized against [`delete_response`]. A streaming client
    /// already knows the response id and can issue a concurrent
    /// `DELETE /v1/responses/{id}`; writing the two records separately leaves a
    /// window where the delete lands between them and the later approval insert
    /// orphans a row still holding the tool arguments. One transaction closes
    /// that window: a delete observes either both records or neither.
    ///
    /// The provided default persists the two records sequentially. It is correct
    /// for in-memory or non-transactional stores that are not subject to
    /// concurrent deletion, but it is **not** atomic; SQL backends override it.
    /// Like [`record_pending_approvals`], the approval writes are insert-if-absent,
    /// so re-persisting the same response never resets an already-consumed row.
    ///
    /// [`record_pending_approvals`]: ResponseStore::record_pending_approvals
    /// [`delete_response`]: ResponseStore::delete_response
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn persist_response_with_pending_approvals(
        &self,
        record: &ResponseRecord,
        pending_approvals: &[PendingApprovalRecord],
    ) -> Result<(), StoreError> {
        self.upsert_response(record).await?;
        if !pending_approvals.is_empty() {
            self.record_pending_approvals(&record.tenant_id, &record.id, pending_approvals, record.created_at)
                .await?;
        }
        Ok(())
    }

    /// Fetch the server-owned pending approvals matching `approval_ids` that
    /// were issued by `response_id`.
    ///
    /// Returns the records the proxy previously wrote via
    /// [`record_pending_approvals`] for the given issuing response, including
    /// rows whose `consumed_at` is already stamped (so callers can distinguish
    /// "never issued" from "already used"). Ids without a matching row under
    /// this `response_id` are simply absent from the result. Ordering is
    /// unspecified; callers key by `approval_id`.
    ///
    /// Scoping by `response_id` binds each approval to the response that issued
    /// it: an approval id is only visible to a follow-up that names the
    /// originating `previous_response_id`, so a fresh, unrelated request cannot
    /// see (and therefore cannot consume) a known outstanding approval, and the
    /// same model-generated call id reused across two responses resolves to the
    /// correct call. A client-forged `mcp_approval_request` persisted into the
    /// conversation history has no matching pending row and is likewise
    /// invisible, so the resume path fails closed.
    ///
    /// [`record_pending_approvals`]: ResponseStore::record_pending_approvals
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn get_pending_approvals(
        &self,
        tenant_id: &str,
        response_id: &str,
        approval_ids: &[&str],
    ) -> Result<Vec<PendingApprovalRecord>, StoreError>;

    /// Atomically claim single-use consumption of a batch of pending approvals
    /// issued by `response_id`.
    ///
    /// Within a single database transaction, stamps `consumed_at` (epoch
    /// milliseconds) on each server-owned pending row in `approval_ids` for
    /// `(tenant_id, response_id)`, transitioning it from outstanding
    /// (`consumed_at IS NULL`) to consumed. Callers first look the rows up via
    /// [`get_pending_approvals`] under the same `response_id`, so every id
    /// passed here is known to have a pending row; a row that fails to
    /// transition therefore means it was **already consumed**.
    ///
    /// The claim is **all-or-nothing** across the whole batch:
    /// - `Ok(None)` — *every* id transitioned outstanding→consumed; the transaction commits.
    /// - `Ok(Some(i))` — `approval_ids[i]` could not be claimed because it was already consumed (by a prior request or
    ///   as a duplicate appearing earlier in this same batch). The transaction rolls back, so **no** id in the batch is
    ///   consumed and every still-outstanding approval stays replayable by a corrected follow-up request.
    ///
    /// Callers must treat `Some(_)` as "at least one approval was already
    /// handled" and refuse to execute *any* tool call in the batch. Because
    /// the transition uses a conditional `UPDATE ... WHERE consumed_at IS NULL`,
    /// concurrent callers race for the row and exactly one observes `None`. An
    /// empty slice is a no-op that returns `Ok(None)`.
    ///
    /// [`get_pending_approvals`]: ResponseStore::get_pending_approvals
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails; on error the
    /// transaction rolls back and nothing is claimed. Callers enforcing
    /// strict single-use must fail closed on error rather than proceed with
    /// unclaimed approvals.
    async fn consume_approvals(
        &self,
        tenant_id: &str,
        response_id: &str,
        approval_ids: &[&str],
        consumed_at: i64,
    ) -> Result<Option<usize>, StoreError>;
}

// -----------------------------------------------------------------------------
// ConversationItemStore Trait
// -----------------------------------------------------------------------------

/// Async persistence layer for conversation lifecycle and item records.
///
/// Provides full conversation lifecycle management plus CRUD
/// operations for individual items within a conversation. Every query
/// is tenant- and conversation-scoped.
#[async_trait]
pub trait ConversationItemStore: Send + Sync {
    /// Insert or update a conversation message cache.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn upsert_conversation(&self, record: &ConversationRecord) -> Result<(), StoreError>;

    /// Update only the denormalized message cache for a conversation.
    ///
    /// Returns `true` if a record was updated, `false` if no matching
    /// conversation existed for this tenant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn update_conversation_messages(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError>;

    /// Replace the denormalized message cache only when it still equals
    /// `expected_messages`.
    ///
    /// Returns `true` when the compare-and-swap succeeds and `false` after a
    /// concurrent cache update or when the conversation does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if serialization or the database operation fails.
    async fn compare_and_swap_conversation_messages(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        expected_messages: &serde_json::Value,
        messages: &serde_json::Value,
    ) -> Result<bool, StoreError>;

    /// Retrieve conversation messages by conversation ID and tenant.
    ///
    /// Returns `None` if the conversation does not exist or belongs
    /// to a different tenant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    async fn get_conversation(
        &self,
        tenant_id: &str,
        conversation_id: &str,
    ) -> Result<Option<ConversationRecord>, StoreError>;

    /// Delete a conversation by ID, scoped to a tenant.
    ///
    /// Returns `true` if a record was deleted, `false` if no
    /// matching record existed for this tenant. To match the OpenAI
    /// Conversations API, this does not delete conversation item rows; items
    /// are deleted only through [`delete_conversation_item`] or an explicit
    /// retention cleanup path.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the database operation fails.
    ///
    /// [`delete_conversation_item`]: ConversationItemStore::delete_conversation_item
    async fn delete_conversation(&self, tenant_id: &str, conversation_id: &str) -> Result<bool, StoreError>;

    /// Insert one or more conversation items.
    ///
    /// Items are inserted individually. Duplicate `item_id` +
    /// `tenant_id` + `conversation_id` triples fail.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn create_conversation_items(&self, items: &[ConversationItemRecord]) -> Result<(), StoreError>;

    /// List items for a conversation with cursor-based pagination.
    ///
    /// Returns items ordered by `(position, item_id)`. When
    /// `after_item_id` is `Some`, only items whose ordering key
    /// compares past the cursor item are returned.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    #[expect(
        clippy::too_many_arguments,
        reason = "pagination query keeps scope and cursor fields explicit"
    )]
    async fn list_conversation_items(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        after_item_id: Option<&str>,
        limit: u32,
        ascending: bool,
    ) -> Result<Vec<ConversationItemRecord>, StoreError>;

    /// Return the subset of `item_ids` that already exist in a
    /// conversation, in a single round-trip.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn get_existing_conversation_item_ids(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        item_ids: &[&str],
    ) -> Result<Vec<String>, StoreError>;

    /// Retrieve a single item by ID, scoped to tenant and
    /// conversation.
    ///
    /// Returns `None` if the item does not exist or belongs to a
    /// different tenant or conversation.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn get_conversation_item(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<ConversationItemRecord>, StoreError>;

    /// Delete a single item by ID, scoped to tenant and
    /// conversation.
    ///
    /// Returns `true` if an item was deleted, `false` if no matching
    /// item existed.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn delete_conversation_item(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError>;

    /// Look up the position of a specific item.
    ///
    /// Returns `None` if the item does not exist in the given
    /// tenant and conversation scope.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn conversation_item_position(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<Option<i64>, StoreError>;

    /// Return the maximum item position for a conversation.
    ///
    /// Returns `0` if the conversation has no items.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn max_item_position(&self, tenant_id: &str, conversation_id: &str) -> Result<i64, StoreError>;

    /// Atomically insert items and rebuild the conversation message cache.
    ///
    /// Within a single database transaction this method:
    /// 1. Reads the current maximum item position.
    /// 2. Assigns sequential positions starting from `max + 1`.
    /// 3. Inserts the items.
    /// 4. Rebuilds the `messages` cache from **all** items.
    /// 5. Updates the conversation row.
    ///
    /// The `position` field in each input record is **ignored**; positions
    /// are assigned within the transaction to prevent collisions.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn create_items_and_sync_messages(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        items: &[ConversationItemRecord],
    ) -> Result<(), StoreError>;

    /// Atomically delete an item and rebuild the conversation message cache.
    ///
    /// Within a single database transaction this method:
    /// 1. Deletes the item row.
    /// 2. Rebuilds the `messages` cache from all remaining items.
    /// 3. Updates the conversation row.
    ///
    /// Returns `true` if the item was deleted, `false` if it did not
    /// exist.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] if the items table is not configured
    /// or a database operation fails.
    async fn delete_item_and_sync_messages(
        &self,
        tenant_id: &str,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<bool, StoreError>;
}
