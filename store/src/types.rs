// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Data types for the response store persistence layer.

use std::fmt;

use crate::owner::StateOwner;

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

    /// Immutable tenant-qualified resource owner.
    pub owner: StateOwner,

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
#[derive(Clone, Debug)]
pub struct ConversationRecord {
    /// Conversation ID (e.g., `"conv_abc123"`).
    pub conversation_id: String,

    /// Immutable tenant-qualified resource owner.
    pub owner: StateOwner,

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
#[derive(Clone, Debug)]
pub struct ConversationItemRecord {
    /// Unique item ID (e.g., `"item_abc123"`).
    pub item_id: String,

    /// Immutable owner inherited from the parent conversation.
    pub owner: StateOwner,

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
// ResponseEventRecord
// -----------------------------------------------------------------------------

/// A single normalized SSE event captured from a stored streaming response.
///
/// The durable event log lets a completed, `stream: true` response be replayed
/// verbatim through `GET /v1/responses/{id}?stream=true`. Each record is one
/// outbound SSE event, stamped with the client-visible `sequence_number` from
/// `openai_stream_events`. The events are buffered in request scope and flushed
/// as one batch at the terminal seam, after the parent record is stored and
/// before the terminal frame is released. The `(response_id, sequence_number)`
/// pair is the primary key; the owner triple gates every access to the parent
/// response, exactly like [`PendingApprovalRecord`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseEventRecord {
    /// Parent response ID (e.g., `"resp_abc123"`).
    pub response_id: String,

    /// Immutable owner inherited from the parent response.
    pub owner: StateOwner,

    /// Client-visible logical stream sequence number (monotonic, contiguous).
    pub sequence_number: u64,

    /// Wire event name (e.g., `"response.output_text.delta"`), equal to the
    /// payload's `type`.
    pub event_type: String,

    /// The fully-normalized event's `data` payload as raw JSON bytes, exactly as
    /// delivered to the client. Held and replayed verbatim — never parsed into a
    /// [`serde_json::Value`] — so replay reproduces the original bytes without a
    /// parse/serialize round trip and without retaining a value tree per event.
    pub payload: Vec<u8>,

    /// True iff this is a terminal event (`completed`/`incomplete`/`failed`/
    /// `error`). A replayable log always ends with exactly one terminal event.
    pub terminal: bool,

    /// Unix timestamp when the event was persisted.
    pub created_at: i64,
}

// -----------------------------------------------------------------------------
// EventLogStatus
// -----------------------------------------------------------------------------

/// Cheap pre-stream gate describing a response's event log.
///
/// The three variants are mutually exclusive and each carries only the data
/// that state needs, so states no backend should produce (a terminal event
/// with no maximum sequence, or a terminal event with no rows) cannot be
/// represented. Only [`EventLogStatus::Replayable`] may be served as a
/// complete SSE replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EventLogStatus {
    /// No event rows exist (legacy or non-streamed record).
    #[default]
    Absent,

    /// Rows exist but no terminal event was recorded (a stream that aborted
    /// before its terminal event). Carries the highest stored
    /// `sequence_number`.
    Incomplete {
        /// Highest stored `sequence_number`.
        max_sequence: u64,
    },

    /// The log reached a terminal event and can be replayed in full. Carries
    /// the highest stored `sequence_number` (the terminal event's).
    Replayable {
        /// Highest stored `sequence_number`.
        max_sequence: u64,
    },
}

// -----------------------------------------------------------------------------
// StoreError
// -----------------------------------------------------------------------------

/// Errors from response store operations.
///
/// Variants carry `String` payloads (not typed inner errors) to
/// avoid coupling the trait to any specific database driver.
#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    /// No resource exists in the authorized owner scope.
    NotFound,

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
            Self::NotFound => write!(f, "resource not found"),
            Self::Database(msg) => write!(f, "database error: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::Serialization(msg) => write!(f, "serialization error: {msg}"),
            Self::Unavailable(msg) => write!(f, "store unavailable: {msg}"),
        }
    }
}

impl std::error::Error for StoreError {}
