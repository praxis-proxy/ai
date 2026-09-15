// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Data types for the response store persistence layer.

use std::fmt;

// -----------------------------------------------------------------------------
// ResponseRecord
// -----------------------------------------------------------------------------

/// A stored response record.
///
/// Holds the full response object, original input, and hidden
/// messages used for multi-turn conversation rehydration. JSON
/// columns use [`serde_json::Value`] — the store is intentionally
/// schema-agnostic about their contents.
#[derive(Clone, Debug)]
pub struct ResponseRecord {
    /// Unique response ID (e.g., `"resp_abc123"`).
    pub id: String,

    /// Tenant ID for multi-tenant isolation.
    pub tenant_id: String,

    /// Unix timestamp when the response was created.
    pub created_at: i64,

    /// Model name used for inference.
    pub model: String,

    /// Full `ResponseResource` as JSON (the public API object).
    pub response_object: serde_json::Value,

    /// Original input as JSON (preserved for the `input_items`
    /// endpoint).
    pub input: serde_json::Value,

    /// Hidden messages as JSON — source of truth for future
    /// turns. Includes system messages and internal state not
    /// exposed in the public response object.
    pub messages: serde_json::Value,
}

// -----------------------------------------------------------------------------
// ConversationRecord
// -----------------------------------------------------------------------------

/// A stored conversation record.
///
/// Holds the conversation object and accumulated messages for a
/// conversation ID. The `messages` field is used by the rehydrate
/// filter for multi-turn context; `metadata` and `created_at` are
/// exposed via the `/v1/conversations` API.
pub struct ConversationRecord {
    /// Conversation ID (e.g., `"conv_abc123"`).
    pub conversation_id: String,

    /// Tenant ID for multi-tenant isolation.
    pub tenant_id: String,

    /// Unix timestamp when the conversation was created.
    ///
    /// Conversation upserts preserve the original creation time;
    /// metadata and message refreshes must not rewrite this value.
    pub created_at: i64,

    /// User-defined metadata as JSON (up to 16 key-value pairs).
    pub metadata: serde_json::Value,

    /// Accumulated conversation messages as JSON.
    pub messages: serde_json::Value,
}

// -----------------------------------------------------------------------------
// ConversationItemRecord
// -----------------------------------------------------------------------------

/// A stored conversation item.
///
/// Items are the individual entries within a conversation (messages,
/// tool calls, tool outputs, etc.). Stored as opaque JSON blobs with
/// a monotonic position for ordering.
#[derive(Debug)]
pub struct ConversationItemRecord {
    /// Unique item ID (e.g., `"item_abc123"`).
    pub item_id: String,

    /// Tenant ID for multi-tenant isolation.
    pub tenant_id: String,

    /// Parent conversation ID.
    pub conversation_id: String,

    /// Verbatim item data as JSON.
    pub item_data: serde_json::Value,

    /// Unix timestamp when the item was created.
    pub created_at: i64,

    /// Monotonic position within the conversation for ordering.
    pub position: i64,
}

// -----------------------------------------------------------------------------
// PendingApprovalRecord
// -----------------------------------------------------------------------------

/// A server-owned pending MCP approval.
///
/// Written from proxy **output** the moment an `mcp_approval_request` is
/// emitted, this is the sole source of truth for correlating a later
/// `mcp_approval_response` back to the call the proxy actually paused on.
/// Consent provenance lives here, never in the conversation history: a client
/// can persist a forged `mcp_approval_request` into the trace, but it will
/// never have a matching pending record, so it fails closed.
///
/// The `target_fingerprint` binds the approval to the concrete resolved target
/// (URL, headers, authorization, connector) captured at approval time so a
/// resume turn cannot keep the approved `(server_label, tool_name)` while
/// redirecting execution elsewhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingApprovalRecord {
    /// Correlation id; equals the paused function-call `call_id` and the
    /// emitted `mcp_approval_request` `id`.
    pub approval_id: String,

    /// Server label of the resolved target.
    pub server_label: String,

    /// Original (un-encoded) tool name of the resolved target.
    pub tool_name: String,

    /// Tool arguments as a JSON string, captured from the paused call.
    pub arguments: String,

    /// Fingerprint of the resolved target captured at approval time.
    pub target_fingerprint: String,
}

// -----------------------------------------------------------------------------
// StoreError
// -----------------------------------------------------------------------------

/// Errors from response store operations.
///
/// Variants carry `String` payloads (not typed inner errors) to
/// avoid coupling the trait to any specific database driver.
#[derive(Debug)]
pub enum StoreError {
    /// Database connection or query failure.
    Database(String),

    /// Client-supplied input is invalid (e.g. malformed cursor).
    InvalidInput(String),

    /// JSON serialization or deserialization failure.
    Serialization(String),

    /// Store not initialized or unavailable.
    Unavailable(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Database(msg) => write!(f, "database error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::Unavailable(msg) => write!(f, "store unavailable: {msg}"),
        }
    }
}

impl std::error::Error for StoreError {}
