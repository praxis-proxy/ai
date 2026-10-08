// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`ResponseStoreFilter`] persists Responses API responses to the
//! configured store backend and handles
//! `DELETE /v1/responses/{id}` locally.
//!
//! # Lifecycle design
//!
//! The filter spans three phases, each refining the "should we
//! persist?" decision as new information becomes available:
//!
//! - **`on_request`**: reads classifier metadata to decide whether the request needs the store (persistable POST or
//!   `previous_response_id`). Resolves the store from the per-request registry, which the serving runtime provisions.
//!   Rejects with a 500 response when a request that requires the store (persistence or rehydration) finds none
//!   provisioned. `GET` and `DELETE` endpoints owned by the store also reject rather than falling through to the
//!   upstream.
//!
//! - **`on_response`**: re-checks skip conditions, then inspects the response status and content-type. Non-2xx
//!   responses or responses with a content-type other than JSON or event-stream set `responses.skip_persist` and bail
//!   early.
//!
//! - **`on_response_body`**: at the terminal chunk for streams or at end-of-stream for buffered responses, extracts the
//!   record from the response JSON or accumulated streaming [`ResponsesState`] and persists it synchronously via
//!   [`block_in_place`] before returning to Pingora. This guarantees the record is durable before the client observes
//!   the completed response, preventing races with subsequent operations like `DELETE /v1/responses/{id}`.
//!   Non-persistable exchanges release chunks immediately via [`FilterAction::Release`] to avoid holding pass-through
//!   traffic in the `StreamBuffer`.
//!
//! [`block_in_place`]: tokio::task::block_in_place
//!
//! The repeated `should_skip_persist()` calls at each phase are
//! intentional. Each phase learns something new (request metadata,
//! response headers, body bytes), and early exit avoids wasted
//! work (store init, body buffering, JSON parsing). Cross-phase
//! control state is carried through string metadata in
//! [`filter_metadata`], following the same pattern as the A2A
//! filter. The original request `input` is carried through typed
//! per-filter state because it can be arbitrary JSON and is not part
//! of the Responses API response object. When rehydrate populated
//! [`ResponsesState`], its persistence history is used as the
//! stored message history so output-only metadata can survive
//! future rehydration without being replayed as backend input.
//!
//! [`filter_metadata`]: praxis_filter::HttpFilterContext::filter_metadata
//! [`ResponsesState`]: super::super::state::ResponsesState

use std::{
    borrow::Cow,
    num::{NonZeroU32, NonZeroU64},
};

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BoundUpstreamBodyOutcome, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    StreamingResponseBody, StreamingTerminalResponse,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
    sse::{SseDecoder, SseLimits, SseRecord},
};
use serde_json::Value;
use tracing::{debug, trace, warn};

use super::{
    super::{DEFAULT_STORE_NAME, bound_body_outcome, error::responses_error_rejection, state::ResponsesState},
    config::{ResponseStoreConfig, validate_config},
};
use crate::{
    classifier::is_responses_create,
    is_event_stream_content_type,
    openai::include::{IncludeFields, decode_query_component_strict, parse_include},
    service::responses::{InputItemPage, ListParams, MAX_PAGE_LIMIT, Order, build_record, list_input_items},
    state_owner::{StateOwner, require_state_owner},
    store::{
        EventLogStatus, OwnerScopedResponseStore, PendingApprovalRecord, ResponseEventRecord, ResponseRecord,
        ResponseStoreRegistry, StoreError,
    },
};

/// Number of replay-log events fetched from the store per streamed page. Bounds
/// the memory a replay holds at once; the whole log is never loaded.
pub(crate) const REPLAY_PAGE_LIMIT: u32 = 256;

/// Headroom added to the SSE decoder's line and record limits above the
/// configured `max_event_bytes` payload budget.
///
/// `max_event_bytes` bounds the JSON `data` payload of a single event, checked
/// authoritatively in [`ResponseStoreFilter::capture_one`] against `data.len()`
/// alone. The SSE decoder additionally counts framing the budget does not:
/// `max_record_bytes` sums field *values* (adding the `event:` type value), and
/// `max_line_bytes` measures a whole line including its `data: ` / `event: `
/// prefix. Sizing the decoder to exactly `max_event_bytes` therefore rejects an
/// event whose payload is within budget once framing is added (a 90-byte payload
/// under a 100-byte budget decodes to a 108-byte record), poisoning the decoder
/// and abandoning the log. This allowance dwarfs every real Responses event type
/// and field prefix, so any within-budget payload always reaches `capture_one`;
/// an over-budget payload is still rejected there.
const SSE_FRAMING_HEADROOM_BYTES: u64 = 1024;

/// 400 message returned for `GET /v1/responses/{id}?stream=true` against a
/// response that has no complete, replayable event log.
const NO_REPLAY_LOG_MESSAGE: &str = "This response has no replayable event stream. Only responses created with stream=true and stored by the proxy can be replayed.";

/// Persists Responses API responses to the configured response store backend.
///
/// For a stored response created with `stream: true`,
/// `GET /v1/responses/{id}?stream=true` replays its completed SSE event log.
/// `starting_after=N` skips events through sequence number `N`. A plain GET
/// returns the stored response as JSON.
///
/// Replay becomes available only after the original response and event log are
/// persisted. A GET before the response is stored returns 404; a stored response
/// without a complete replay log returns 400 for `stream=true`. This endpoint
/// does not follow generation in progress. If the original foreground connection
/// drops, generation and replay are not guaranteed to complete.
///
/// # YAML
///
/// ```yaml
/// filter: openai_response_store
/// backend: postgres
/// database_url: postgres://praxis:password@db.example.com/praxis
/// responses_table: openai_responses
/// conversations_table: openai_conversation_messages
/// allow_private_database_url: true
/// ```
pub struct ResponseStoreFilter {
    /// Maximum number of SSE events retained in a streamed response's replay log.
    max_event_count: NonZeroU32,

    /// Maximum total payload bytes retained in a streamed response's replay log.
    max_event_bytes: NonZeroU64,
}

/// The request-scoped persistence state was consumed by a streaming write
/// attempt. EOS must not retry it, including when `failure_mode: open` lets a
/// failed terminal-chunk write continue.
struct StreamingResponsePersistenceAttempted;

impl ResponseStoreFilter {
    /// Construct the filter with explicit replay-log bounds.
    ///
    /// Exposed to the crate's tests; production construction goes through
    /// [`Self::from_config`], which reads the bounds from validated YAML.
    #[must_use]
    pub(super) const fn with_bounds(max_event_count: NonZeroU32, max_event_bytes: NonZeroU64) -> Self {
        Self {
            max_event_count,
            max_event_bytes,
        }
    }

    /// Create a filter from parsed YAML config.
    ///
    /// The config is validated here so a malformed or unknown-backend config
    /// fails at pipeline construction. The serving-runtime provisioner opens the
    /// backend and registers it; the filter only resolves it at request time.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ResponseStoreConfig = parse_filter_config("openai_response_store", config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self::with_bounds(cfg.max_event_count, cfg.max_event_bytes)))
    }

    /// Handle `DELETE /v1/responses/{id}` by deleting from the store.
    async fn handle_delete(
        &self,
        ctx: &HttpFilterContext<'_>,
        owner: &StateOwner,
        id: &str,
    ) -> Result<FilterAction, FilterError> {
        let Some(store) = resolve_store(ctx, owner) else {
            return Ok(FilterAction::Reject(reject_store_error()));
        };

        let deleted = store
            .delete_response(id)
            .await
            .map_err(|e| FilterError::from(format!("openai_response_store: delete failed: {e}")))?;

        if deleted {
            debug!(id, "response deleted");
            Ok(FilterAction::Reject(delete_success_rejection(id)?))
        } else {
            debug!(id, "response not found for delete");
            Ok(FilterAction::Reject(delete_not_found_rejection(id)))
        }
    }

    /// Return whether this exchange should release response body
    /// chunks immediately instead of waiting for EOS.
    fn should_release_skipped_response_body(ctx: &HttpFilterContext<'_>) -> bool {
        should_skip_persist(ctx) || !store_available(ctx)
    }

    /// Persist a streaming response from accumulated `ResponsesState`.
    fn persist_from_streaming_state(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if should_skip_persist(ctx) {
            return Ok(FilterAction::Continue);
        }

        if stream_has_errors(ctx) {
            trace!("skipping streaming persistence: stream had errors or was incomplete");
            return Ok(FilterAction::Continue);
        }

        if !store_available(ctx) {
            trace!("skipping streaming persistence: store unavailable");
            return Ok(FilterAction::Continue);
        }

        // Capture the proxy-issued pending approvals before taking the context.
        let pending_approvals = pending_approvals_from_ctx(ctx);
        let persist = match take_persist_context(ctx) {
            Ok(parts) => parts,
            Err(action) => return Ok(action),
        };
        ctx.extensions.insert(StreamingResponsePersistenceAttempted);

        let Some(record) = build_streaming_record(ctx, persist.owner, persist.request_input) else {
            trace!("skipping streaming persistence: no persistable record");
            return Ok(FilterAction::Continue);
        };

        persist_response_blocking(&persist.store, &record, &pending_approvals)?;
        // Flush the replay event log only after the record is durable, so the
        // parent-exists gate is satisfied, and before releasing the terminal
        // frame, so a client never observes completion for a response whose
        // replay log did not persist.
        flush_event_log_blocking(
            &persist.store,
            &record,
            persist.captured_events,
            persist.events_over_budget,
        )?;
        Ok(FilterAction::Continue)
    }

    /// Persist a non-streaming response from the buffered body bytes.
    fn persist_from_buffered_body(
        ctx: &mut HttpFilterContext<'_>,
        body: &Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        if should_skip_persist(ctx) {
            return Ok(FilterAction::Continue);
        }
        let Some(bytes) = body.as_ref().filter(|b| !b.is_empty()) else {
            return Ok(FilterAction::Continue);
        };
        if !store_available(ctx) {
            return Ok(FilterAction::Continue);
        }

        // Capture the proxy-issued pending approvals before taking the context.
        let pending_approvals = pending_approvals_from_ctx(ctx);
        // A non-streaming response captures no SSE events, so the event log is
        // always empty here; only the buffered record is persisted.
        let PersistContext {
            store,
            owner,
            request_input,
            ..
        } = match take_persist_context(ctx) {
            Ok(parts) => parts,
            Err(action) => return Ok(action),
        };

        let Some(record) = build_buffered_record(ctx, bytes, owner, request_input) else {
            return Ok(FilterAction::Continue);
        };

        persist_response_blocking(&store, &record, &pending_approvals)?;
        Ok(FilterAction::Continue)
    }

    /// Parse the outbound SSE chunk into normalized events and accumulate them in
    /// request-scoped state for the terminal flush.
    ///
    /// Parse-only: it never blocks, mutates, or withholds the client bytes. The
    /// terminal seam ([`Self::persist_from_streaming_state`]) drains this state
    /// and writes the whole log in one `append_events` after the record is
    /// durable. A bound overrun or a poisoned decoder sets the sticky
    /// over-budget flag, which abandons the log (the response stays retrievable
    /// as plain JSON but becomes non-replayable).
    fn capture_stream_events(&self, ctx: &mut HttpFilterContext<'_>, body: &Option<Bytes>) {
        let Some(chunk) = body.as_ref().filter(|b| !b.is_empty()) else {
            return;
        };
        let mut state = ctx.extensions.remove::<ResponseStoreRequestState>().unwrap_or_default();
        state.error_event_detector.push(chunk);
        if !state.events_over_budget {
            self.accumulate_events(&mut state, chunk);
        }
        ctx.extensions.insert(state);
    }

    /// Decode `chunk` and append each contained SSE event, stopping and marking
    /// the log non-replayable on a bound overrun or decoder poison.
    fn accumulate_events(&self, state: &mut ResponseStoreRequestState, chunk: &Bytes) {
        let max_event_bytes = self.max_event_bytes.get();
        let decoder = state.event_decoder.get_or_insert_with(|| {
            // Allow a single event whose JSON `data` payload fills the whole byte
            // budget, plus framing headroom the decoder counts but the budget
            // does not (see `SSE_FRAMING_HEADROOM_BYTES`). Both the per-line and
            // per-record caps are raised together: a normalized event carries the
            // payload on one `data: <json>` line, and the record also sums the
            // `event:` type value. The authoritative payload bound stays in
            // `capture_one`, so this headroom never admits an over-budget payload.
            let budget =
                usize::try_from(max_event_bytes.saturating_add(SSE_FRAMING_HEADROOM_BYTES)).unwrap_or(usize::MAX);
            SseDecoder::with_limits(SseLimits {
                max_line_bytes: budget,
                max_record_bytes: budget,
                ..SseLimits::default()
            })
        });
        let batch = decoder.push(chunk);
        for record in &batch.records {
            if !self.capture_one(state, record) {
                // The replay log is abandoned, but the separate header scanner
                // still checks later chunks for a client-visible error.
                return;
            }
        }
        if batch.error.is_some() {
            // A poisoned decoder cannot reliably continue; abandon the log so a
            // truncated stream is never served as a complete replay.
            state.events_over_budget = true;
            state.events.clear();
            state.event_bytes = 0;
        }
    }

    /// Capture one decoded SSE record. Returns `false` once a bound is exceeded
    /// so the caller stops. Non-replay records (comments, `[DONE]`, non-JSON
    /// payloads, events without a numeric `sequence_number`) are skipped and
    /// return `true`.
    fn capture_one(&self, state: &mut ResponseStoreRequestState, record: &SseRecord) -> bool {
        let Some((byte_len, event)) = decode_replay_row(record) else {
            return true;
        };
        // Enforce both bounds before committing this event.
        let next_bytes = state.event_bytes.saturating_add(byte_len);
        if state.events.len() as u64 >= u64::from(self.max_event_count.get()) || next_bytes > self.max_event_bytes.get()
        {
            state.events_over_budget = true;
            state.events.clear();
            state.event_bytes = 0;
            return false;
        }
        state.event_bytes = next_bytes;
        state.events.push(event);
        true
    }
}

/// Decode one SSE record into a replay row plus its on-wire byte length.
/// `None` for records that are not replay rows: comments, `[DONE]`, non-JSON
/// payloads, and events without a numeric `sequence_number`.
fn decode_replay_row(record: &SseRecord) -> Option<(u64, CapturedEvent)> {
    if !record.is_event() {
        return None;
    }
    let data = record.data();
    if data.as_ref() == b"[DONE]" {
        return None;
    }
    // Peek only the header fields needed to index and classify the event; the
    // full `data` payload is retained as raw bytes, not a parsed value tree. A
    // struct deserialize still validates the whole JSON object and rejects a
    // missing or non-numeric `sequence_number`, matching the previous behavior.
    let Ok(head) = serde_json::from_slice::<ReplayEventHead<'_>>(&data) else {
        trace!("replay capture: outbound SSE event lacked a numeric sequence_number or was not valid JSON; skipping");
        return None;
    };
    let event_type = record
        .event()
        .and_then(|e| std::str::from_utf8(e).ok())
        .map(str::to_owned)
        .or_else(|| head.ty.map(Cow::into_owned))
        .unwrap_or_default();
    let terminal = is_terminal_event_type(&event_type);
    Some((
        data.len() as u64,
        CapturedEvent {
            sequence_number: head.sequence_number,
            event_type,
            payload: data.to_vec(),
            terminal,
        },
    ))
}

/// The only fields read from a captured SSE event's JSON `data` payload: the
/// required numeric `sequence_number` used to index the replay log, and the
/// optional `type` used as the event name when the SSE record omits an `event:`
/// line. The `type` is borrowed from the input where possible so the common
/// path (an `event:` line is present) allocates nothing. Every other field is
/// ignored — the payload itself is stored verbatim as raw bytes.
#[derive(serde::Deserialize)]
struct ReplayEventHead<'a> {
    /// Required client-visible logical-stream sequence number; a missing or
    /// non-numeric value fails the deserialize so the event is skipped.
    sequence_number: u64,
    /// Optional wire event name, used only when the SSE record has no `event:`
    /// line. Borrowed from the payload where no unescaping is needed.
    #[serde(borrow, default, rename = "type")]
    ty: Option<Cow<'a, str>>,
}

/// Resolve the owner-scoped Responses store from the per-request registry.
///
/// The store is provisioned into the registry on the serving runtime; the filter
/// takes an owner-bound handle at request time.
/// `None` when no registry is installed or the store is not provisioned.
fn resolve_store(ctx: &HttpFilterContext<'_>, owner: &StateOwner) -> Option<OwnerScopedResponseStore> {
    ctx.extensions
        .get::<ResponseStoreRegistry>()
        .and_then(|registry| registry.get_scoped(DEFAULT_STORE_NAME, owner))
}

/// Request-scoped persistence context captured before inference: the
/// owner-scoped store, the immutable owner, the original request input, the
/// captured replay events, and whether capture went over budget.
struct PersistContext {
    /// Owner-scoped Responses store used to persist the record and event log.
    store: OwnerScopedResponseStore,
    /// Immutable owner the response and its events belong to.
    owner: StateOwner,
    /// Original request input, persisted alongside the response.
    request_input: Option<Value>,
    /// Replay events captured from the outbound SSE stream, in sequence order.
    captured_events: Vec<CapturedEvent>,
    /// Whether capture exceeded a bound; the log is then abandoned (non-replayable).
    events_over_budget: bool,
}

/// Take the [`PersistContext`] from the request extensions. `Err` carries the
/// fail-closed rejection when the capture, owner, or store is missing.
fn take_persist_context(ctx: &mut HttpFilterContext<'_>) -> Result<PersistContext, FilterAction> {
    let capture = ctx
        .extensions
        .remove::<ResponseStoreRequestState>()
        .ok_or_else(|| FilterAction::Reject(reject_store_error()))?;
    let owner = capture
        .owner
        .ok_or_else(|| FilterAction::Reject(reject_store_error()))?;
    let store = resolve_store(ctx, &owner).ok_or_else(|| FilterAction::Reject(reject_store_error()))?;
    Ok(PersistContext {
        store,
        owner,
        request_input: capture.input,
        captured_events: capture.events,
        events_over_budget: capture.events_over_budget,
    })
}

/// Build the streaming record from accumulated [`ResponsesState`]. `None` when no
/// state was accumulated or the response is not persistable.
fn build_streaming_record(
    ctx: &HttpFilterContext<'_>,
    owner: StateOwner,
    request_input: Option<Value>,
) -> Option<ResponseRecord> {
    let state = ctx.extensions.get::<ResponsesState>()?;
    let response_object = state.response_object.clone();
    let state_messages = (!state.persisted_messages.is_empty()).then(|| state.persisted_messages.clone());
    build_record(response_object, owner, request_input, state_messages)
}

/// Build the buffered record from the decoded response body and the captured
/// request state. `None` when the body is invalid JSON or not persistable.
fn build_buffered_record(
    ctx: &HttpFilterContext<'_>,
    bytes: &[u8],
    owner: StateOwner,
    request_input: Option<Value>,
) -> Option<ResponseRecord> {
    let state_messages = ctx
        .extensions
        .get::<ResponsesState>()
        .map(|state| state.persisted_messages.clone());
    let json = decode_response_body(bytes)?;
    build_record(json, owner, request_input, state_messages)
}

/// Decode a buffered response body, logging and skipping on invalid JSON.
fn decode_response_body(bytes: &[u8]) -> Option<Value> {
    match serde_json::from_slice(bytes) {
        Ok(value) => Some(value),
        Err(e) => {
            warn!(error = %e, "response store: invalid response JSON");
            None
        },
    }
}

/// Whether the default store is provisioned into the per-request registry. The
/// owner-scoped read needs an owner, so a presence check that does not have one
/// (release decision, pre-persist gate) uses this instead.
fn store_available(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponseStoreRegistry>()
        .is_some_and(|registry| registry.contains(DEFAULT_STORE_NAME))
}

// -----------------------------------------------------------------------------
// Request / Response Capture
// -----------------------------------------------------------------------------

/// One normalized SSE event captured from the outbound stream, held in request
/// scope until the terminal seam flushes the whole log at once.
struct CapturedEvent {
    /// Client-visible logical-stream sequence number parsed from the payload.
    sequence_number: u64,
    /// Wire event name (equals the payload's `type`).
    event_type: String,
    /// The event's `data` payload as raw JSON bytes, captured verbatim so replay
    /// re-emits the original bytes without a parse/serialize round trip and no
    /// value tree is retained per event until the terminal seam.
    payload: Vec<u8>,
    /// Whether this is a terminal event.
    terminal: bool,
}

/// Detect an SSE `event: error` header independently of the replay decoder.
/// A replay byte limit can poison that decoder before a later error arrives.
/// Only a short line prefix is retained, so oversized `data:` lines cannot
/// prevent error detection or make this scanner buffer a response body.
#[derive(Default)]
struct SseErrorEventDetector {
    /// Prefix of the current SSE line, bounded independently of replay limits.
    line: [u8; 32],
    /// Number of bytes retained in `line`.
    line_len: usize,
    /// Whether the current line exceeded the fixed prefix capacity.
    line_overlong: bool,
    /// Sticky signal that the stream contained an `event: error` header.
    saw_error: bool,
}

impl SseErrorEventDetector {
    /// Inspect SSE line headers across arbitrary chunk boundaries.
    fn push(&mut self, chunk: &[u8]) {
        for &byte in chunk {
            if self.saw_error {
                return;
            }
            if byte == b'\n' || byte == b'\r' {
                if !self.line_overlong {
                    let line = self.line.get(..self.line_len).unwrap_or_default();
                    if let Some(value) = line.strip_prefix(b"event:") {
                        let value = value.strip_prefix(b" ").unwrap_or(value);
                        self.saw_error = value == b"error";
                    }
                }
                self.line_len = 0;
                self.line_overlong = false;
            } else if !self.line_overlong {
                if let Some(slot) = self.line.get_mut(self.line_len) {
                    *slot = byte;
                    self.line_len += 1;
                } else {
                    self.line_overlong = true;
                }
            }
        }
    }
}

/// Request-phase data needed when persisting the response.
#[derive(Default)]
struct ResponseStoreRequestState {
    /// Original `input` value from the Responses API create request.
    input: Option<Value>,
    /// Owner captured before inference begins.
    owner: Option<StateOwner>,
    /// Incremental SSE decoder over the outbound stream, lazily created on the
    /// first response chunk ([`SseDecoder`] has no `Default`).
    event_decoder: Option<SseDecoder>,
    /// Events captured so far, flushed together at the terminal seam.
    events: Vec<CapturedEvent>,
    /// Running total of captured payload bytes, for the byte bound.
    event_bytes: u64,
    /// Sticky flag: capture exceeded a bound or the decoder was poisoned, so the
    /// partial log is abandoned and the response becomes non-replayable.
    events_over_budget: bool,
    /// A client-visible SSE error superseded any completed upstream snapshot.
    error_event_detector: SseErrorEventDetector,
}

/// Capture the immutable owner once, before inference or a body-first consumer.
fn capture_persistence_owner(ctx: &mut HttpFilterContext<'_>) -> Result<(), FilterAction> {
    if !request_will_persist_response(ctx) {
        return Ok(());
    }
    let mut state = ctx.extensions.remove::<ResponseStoreRequestState>().unwrap_or_default();
    if state.owner.is_none() {
        state.owner = Some(require_state_owner(ctx)?.clone());
    }
    ctx.extensions.insert(state);
    Ok(())
}

/// Retain request input alongside the already captured owner.
fn capture_request_input(ctx: &mut HttpFilterContext<'_>, input: Value) {
    let mut state = ctx.extensions.remove::<ResponseStoreRequestState>().unwrap_or_default();
    state.input = Some(input);
    ctx.extensions.insert(state);
}

/// Extract the original Responses API request input from the buffered
/// create request body.
fn extract_request_input(body: &Option<Bytes>) -> Option<Value> {
    let bytes = body.as_ref().filter(|b| !b.is_empty())?;
    let mut json: Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => {
            trace!(error = %e, "response store: invalid request JSON");
            return None;
        },
    };
    json.as_object_mut()?.remove("input")
}

// -----------------------------------------------------------------------------
// Path Extraction
// -----------------------------------------------------------------------------

/// Extract the response ID from a `/v1/responses/{id}` path.
///
/// Returns `None` if the path does not match the expected pattern.
pub(super) fn extract_response_id(path: &str) -> Option<&str> {
    let path = path.strip_suffix('/').unwrap_or(path);
    let id = path.strip_prefix("/v1/responses/")?;
    (!id.is_empty() && !id.contains('/')).then_some(id)
}

// -----------------------------------------------------------------------------
// Persistence Arming
// -----------------------------------------------------------------------------

/// Mark this exchange as persistence-armed on the shared [`ResponsesState`].
///
/// This is the exchange-scoped signal `mcp_dispatch` requires before emitting an
/// `mcp_approval_request`. It is set only when the request classifies as one
/// whose response this filter will persist, so, unlike pipeline-scoped registry
/// membership, it proves persistence is armed for THIS exchange and catches a
/// store filter that is absent, request-conditioned out, or ordered after
/// dispatch.
///
/// It is written from `on_request_body` because `openai_responses_request`
/// creates `ResponsesState` in its own `on_request_body`, which runs earlier in
/// the same body phase, so `ResponsesState` is not yet present during
/// `on_request`.
fn arm_persistence_if_persisting(ctx: &mut HttpFilterContext<'_>) {
    if request_will_persist_response(ctx)
        && let Some(state) = ctx.extensions.get_mut::<ResponsesState>()
    {
        state.store_persist_armed = true;
    }
}

// -----------------------------------------------------------------------------
// Delete Response Helpers
// -----------------------------------------------------------------------------

/// Build the 200 rejection for a successful delete.
fn delete_success_rejection(id: &str) -> Result<Rejection, FilterError> {
    let body = serde_json::to_string(&serde_json::json!({
        "id": id,
        "object": "response.deleted",
        "deleted": true,
    }))
    .map_err(|e| FilterError::from(format!("openai_response_store: serialize failed: {e}")))?;

    Ok(Rejection::status(200)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(body)))
}

/// Build the 404 rejection for a missing response.
fn delete_not_found_rejection(id: &str) -> Rejection {
    responses_error_rejection(
        404,
        "invalid_request_error",
        &format!("No response found with id: '{id}'."),
    )
}

// -----------------------------------------------------------------------------
// Bypass Helpers
// -----------------------------------------------------------------------------

/// Check whether this request should skip persistence entirely.
fn should_skip(ctx: &HttpFilterContext<'_>) -> bool {
    is_non_post_request(ctx)
        || is_non_responses_format(ctx)
        || is_store_disabled(ctx)
        || !is_responses_create(&ctx.request.method, ctx.request.uri.path())
}

/// Check whether this request should initialize the store.
fn should_init_store_for_request(ctx: &HttpFilterContext<'_>) -> bool {
    request_will_persist_response(ctx) || request_needs_rehydrate_store(ctx)
}

/// Check whether this request can persist the eventual response.
fn request_will_persist_response(ctx: &HttpFilterContext<'_>) -> bool {
    is_responses_create(&ctx.request.method, ctx.request.uri.path())
        && is_responses_format(ctx)
        && !is_store_disabled(ctx)
}

/// Check whether rehydrate needs the store before the request phase.
fn request_needs_rehydrate_store(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.request.method == http::Method::POST
        && is_responses_format(ctx)
        && (has_previous_response_id(ctx) || has_conversation(ctx))
}

/// Return whether the request references a conversation.
fn has_conversation(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.has_conversation") == Some("true")
}

/// Return whether the request method is not persistable.
fn is_non_post_request(ctx: &HttpFilterContext<'_>) -> bool {
    let skip = ctx.request.method != http::Method::POST;
    if skip {
        trace!(method = %ctx.request.method, "skipping non-POST request");
    }
    skip
}

/// Return whether the request is not a Responses API request.
fn is_non_responses_format(ctx: &HttpFilterContext<'_>) -> bool {
    let format = ctx.get_metadata("openai_responses_format.format");
    let skip = !is_responses_format(ctx);
    if skip {
        trace!(format = ?format, "skipping non-responses format");
    }
    skip
}

/// Return whether the request is classified as a Responses API request.
fn is_responses_format(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.format") == Some("openai_responses")
}

/// Return whether the request explicitly disabled persistence.
fn is_store_disabled(ctx: &HttpFilterContext<'_>) -> bool {
    let skip = ctx.get_metadata("openai_responses_format.store") == Some("false");
    if skip {
        trace!("skipping persistence (store=false)");
    }
    skip
}

/// Return whether the request uses streaming responses.
fn is_streaming_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream") == Some("true")
}

/// Return whether this is a streaming replay retrieval
/// (`GET /v1/responses/{id}?stream=true`).
///
/// Only the bare `{id}` retrieval replays; `/input_items` and nested paths do
/// not. A malformed query returns `false` so the normal retrieval path reports
/// the parameter error as a non-streaming rejection.
fn is_replay_get(ctx: &HttpFilterContext<'_>) -> bool {
    if ctx.request.method != http::Method::GET {
        return false;
    }
    let path = ctx.request.uri.path();
    let path = path.strip_suffix('/').filter(|p| !p.is_empty()).unwrap_or(path);
    let Some(rest) = path.strip_prefix("/v1/responses/") else {
        return false;
    };
    if rest.is_empty() || rest.contains('/') {
        return false;
    }
    matches!(parse_get_response_query(ctx.request.uri.query()), Ok(parsed) if parsed.stream)
}

/// Return whether the canonical terminal frame is ready for release. Both a
/// deferred upstream terminal and a locally encoded completion may reach outer
/// filters as non-EOS chunks.
fn streaming_terminal_emitted(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .is_some_and(|state| state.logical_stream_terminal_emitted || state.local_stream_terminal_emitted)
}

/// Return whether the request references a previous response.
fn has_previous_response_id(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.has_previous_response_id") == Some("true")
}

/// Check whether persistence was skipped during the response phase.
fn should_skip_persist(ctx: &HttpFilterContext<'_>) -> bool {
    should_skip(ctx) || ctx.get_metadata("responses.skip_persist") == Some("true")
}

/// Return whether stream parsing or lifecycle completion failed.
fn stream_has_errors(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("responses.stream_parse_error") == Some("true")
        || ctx.get_metadata("responses.stream_incomplete") == Some("true")
        || ctx
            .extensions
            .get::<ResponseStoreRequestState>()
            .is_some_and(|state| state.error_event_detector.saw_error)
}

/// Return whether a `Content-Type` header is JSON.
fn is_json_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json")
}

/// Check response headers before enabling response body buffering.
fn response_is_persistable(ctx: &mut HttpFilterContext<'_>) -> bool {
    let Some(resp) = ctx.response_header.as_ref() else {
        return true;
    };

    if !resp.status.is_success() {
        trace!(status = %resp.status, "skipping persistence for non-2xx response");
        ctx.set_metadata("responses.skip_persist", "true");
        return false;
    }

    let content_type = resp
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    let is_persistable_content = is_json_content_type(content_type) || is_event_stream_content_type(content_type);
    if !is_persistable_content {
        trace!("skipping persistence for non-persistable content type");
        ctx.set_metadata("responses.skip_persist", "true");
        return false;
    }

    true
}

/// Persist a response record synchronously via [`block_in_place`].
///
/// Uses the current Tokio runtime handle to drive the async
/// `persist` call without yielding back to Pingora's
/// synchronous `response_body_filter`. This guarantees the record
/// is durable before the response reaches the client, preventing
/// races where a subsequent `DELETE /v1/responses/{id}` arrives
/// before the upsert completes.
///
/// Any server-owned pending approval requests the proxy emitted on this turn are
/// recorded in the **same transaction** as the response, so a follow-up
/// `mcp_approval_response` can correlate against a durable, server-written record
/// rather than trusting the (client-influenced) conversation history. Committing
/// both together, serialized against deletion, prevents a concurrent
/// `DELETE /v1/responses/{id}` from landing between the two writes and orphaning a
/// pending approval. `persist_response_with_pending_approvals` is insert-if-absent
/// for the approvals, so re-persisting the same response never resets an
/// already-consumed approval.
///
/// [`block_in_place`]: tokio::task::block_in_place
fn persist_response_blocking(
    store: &OwnerScopedResponseStore,
    record: &ResponseRecord,
    pending_approvals: &[PendingApprovalRecord],
) -> Result<(), FilterError> {
    debug!(
        id = %record.id,
        model = %record.model,
        pending_approvals = pending_approvals.len(),
        "persisting response"
    );

    let handle = tokio::runtime::Handle::current();
    tokio::task::block_in_place(|| {
        handle.block_on(async {
            store
                .persist_response_with_pending_approvals(record, pending_approvals)
                .await
        })
    })
    .map_err(|e| -> FilterError { Box::new(e) })
}

/// Flush the captured replay event log synchronously, right after the response
/// record is durable and before the terminal frame is released.
///
/// Mirrors [`persist_response_blocking`]'s durability guarantee (#937): an
/// append failure propagates so the client never observes a terminal frame for a
/// response whose replay log did not persist. When capture went over budget or
/// the decoder was poisoned, the partial log is dropped entirely — the response
/// stays retrievable as JSON but is not replayable, which the terminal-event
/// gate on the GET replay path enforces.
fn flush_event_log_blocking(
    store: &OwnerScopedResponseStore,
    record: &ResponseRecord,
    events: Vec<CapturedEvent>,
    over_budget: bool,
) -> Result<(), FilterError> {
    if over_budget {
        debug!(
            id = %record.id,
            "replay event log exceeded configured bounds; not persisting (response remains non-replayable)"
        );
        return Ok(());
    }
    if events.is_empty() {
        return Ok(());
    }

    // Ownership is required to build the durable rows: the id and owner are small
    // identifiers stamped from the record; the event payloads are moved, not
    // cloned. `created_at` reuses the record's deterministic timestamp.
    let records: Vec<ResponseEventRecord> = events
        .into_iter()
        .map(|event| ResponseEventRecord {
            response_id: record.id.clone(),
            owner: record.owner.clone(),
            sequence_number: event.sequence_number,
            event_type: event.event_type,
            payload: event.payload,
            terminal: event.terminal,
            created_at: record.created_at,
        })
        .collect();

    debug!(id = %record.id, events = records.len(), "persisting replay event log");
    let handle = tokio::runtime::Handle::current();
    tokio::task::block_in_place(|| handle.block_on(async { store.append_events(&record.id, &records).await }))
        .map_err(|e| -> FilterError { Box::new(e) })
}

/// Whether an event type is a logical-stream terminal. A replayable log always
/// ends with exactly one of these.
fn is_terminal_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.completed" | "response.incomplete" | "response.failed" | "error"
    )
}

/// Snapshot the proxy-issued pending approvals from request-scoped state.
///
/// Cloned because the record is written on the blocking store hook after `ctx`
/// is re-borrowed to build the response record; the buffer is small (one entry
/// per approval the proxy issued this turn).
fn pending_approvals_from_ctx(ctx: &HttpFilterContext<'_>) -> Vec<PendingApprovalRecord> {
    ctx.extensions
        .get::<ResponsesState>()
        .map(|state| state.pending_approvals.clone())
        .unwrap_or_default()
}

// -----------------------------------------------------------------------------
// HttpFilter Implementation
// -----------------------------------------------------------------------------

#[async_trait]
impl HttpFilter for ResponseStoreFilter {
    fn name(&self) -> &'static str {
        "openai_response_store"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn bound_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// Streaming by default. Non-streaming Responses requests select a
    /// bounded `StreamBuffer` dynamically in [`Self::on_request`].
    ///
    /// Non-streaming Responses API payloads are bounded by output
    /// token limits (typically under 2 MiB). The 64 MiB ceiling is
    /// 30x headroom; it will never fire in practice but guards
    /// against a misbehaving backend. The client is already waiting
    /// for the full model inference, so the hold-back latency from
    /// `StreamBuffer` is negligible.
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.request.method == http::Method::GET {
            return self.handle_get_request(ctx).await;
        }

        if is_responses_format(ctx) && !is_streaming_request(ctx) {
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(MAX_JSON_BODY_BYTES),
            });
        }

        if ctx.request.method == http::Method::DELETE {
            if let Some(id) = extract_response_id(ctx.request.uri.path()) {
                let owner = match require_state_owner(ctx) {
                    Ok(owner) => owner.clone(),
                    Err(action) => return Ok(action),
                };
                return self.handle_delete(ctx, &owner, id).await;
            }
            return Ok(FilterAction::Continue);
        }

        if let Err(action) = capture_persistence_owner(ctx) {
            return Ok(action);
        }

        // A persistable or rehydrating request needs the provisioned store; fail
        // fast (before inference) when it is absent rather than at response time.
        if should_init_store_for_request(ctx) && !store_available(ctx) {
            return Ok(FilterAction::Reject(reject_store_error()));
        }

        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream || ctx.request.method != http::Method::POST {
            return Ok(FilterAction::Continue);
        }
        if let Err(action) = capture_persistence_owner(ctx) {
            return Ok(action);
        }
        if !should_skip(ctx)
            && let Some(input) = extract_request_input(body)
        {
            capture_request_input(ctx, input);
        }
        if should_init_store_for_request(ctx) {
            if !store_available(ctx) {
                return Ok(FilterAction::Reject(reject_store_error()));
            }
            // Publish the exchange-scoped persistence-armed marker so a
            // downstream approval pause (mcp_dispatch) can tell that THIS
            // response will be persisted, not merely that a store is provisioned
            // somewhere in the pipeline.
            arm_persistence_if_persisting(ctx);
        }
        Ok(FilterAction::Continue)
    }

    async fn on_bound_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<BoundUpstreamBodyOutcome, FilterError> {
        let action = self.on_request_body(ctx, body, true).await?;
        bound_body_outcome(action)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if should_skip_persist(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !response_is_persistable(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !store_available(ctx) {
            return Ok(FilterAction::Reject(reject_store_error()));
        }

        trace!("response body persistence armed");

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if Self::should_release_skipped_response_body(ctx) {
            return Ok(FilterAction::Release);
        }

        if is_streaming_request(ctx) {
            // Capture this chunk's SSE events into request-scoped state before
            // any persist seam below drains that state. Parse-only; the
            // client-visible bytes are never withheld or altered.
            self.capture_stream_events(ctx, body);

            if !end_of_stream {
                // Deferred and locally encoded terminals can both reach this
                // pre-IRR filter before EOS. Once either marker is set,
                // `response_object` is canonical: persist synchronously BEFORE
                // releasing the chunk.
                if streaming_terminal_emitted(ctx)
                    && ctx.extensions.get::<StreamingResponsePersistenceAttempted>().is_none()
                {
                    // `persist_from_streaming_state` returns `Continue` once the
                    // record is durable (or persistence is legitimately skipped,
                    // e.g. no store configured); we still release the frame
                    // ourselves in that case. Anything else is a fail-closed
                    // decision — a `Reject` when the immutable owner/request
                    // state is missing (#1197), or an `Err` on a persistence
                    // failure — and must be propagated so the client never
                    // observes `response.completed` for an unpersisted record.
                    match Self::persist_from_streaming_state(ctx)? {
                        FilterAction::Continue => {},
                        action => return Ok(action),
                    }
                }
                return Ok(FilterAction::Release);
            }
            // Skip after an earlier chunk consumed persistence state. This
            // includes fail-open errors, which cannot be retried at EOS. The
            // fallback remains available to non-IRR streams without a terminal
            // marker; their captured SSE errors still suppress persistence.
            if ctx.extensions.get::<StreamingResponsePersistenceAttempted>().is_some() {
                return Ok(FilterAction::Continue);
            }
            return Self::persist_from_streaming_state(ctx);
        }

        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        Self::persist_from_buffered_body(ctx, body)
    }
}

// -----------------------------------------------------------------------------
// GET Retrieval
// -----------------------------------------------------------------------------

#[expect(clippy::multiple_inherent_impl, reason = "GET retrieval is a distinct concern")]
impl ResponseStoreFilter {
    /// Handle a GET request: replay GETs stream stored events, other Responses
    /// GETs fall through to normal retrieval, and unrelated paths continue.
    async fn handle_get_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // A replay GET returns a streaming SSE response. Force streaming
        // response mode first: a legacy `openai_responses_format` pipeline
        // classifies the GET as Responses-format with `stream=false` (the flag
        // is read from a POST body it never sees), which would otherwise select
        // the buffered override in `on_request` and make the runtime reject the
        // streaming replay body.
        if is_replay_get(ctx) {
            ctx.set_response_body_mode(BodyMode::Stream);
        }
        if let Some(action) = self.try_get_retrieval(ctx).await? {
            return Ok(action);
        }
        Ok(FilterAction::Continue)
    }

    /// Attempt to handle a GET request for a stored response or its
    /// input items. Returns `Some(action)` when the path matches a
    /// retrieval endpoint, or `None` for unrelated paths.
    async fn try_get_retrieval(&self, ctx: &HttpFilterContext<'_>) -> Result<Option<FilterAction>, FilterError> {
        let path = ctx.request.uri.path();
        let path = path.strip_suffix('/').filter(|p| !p.is_empty()).unwrap_or(path);
        let rest = match path.strip_prefix("/v1/responses/") {
            Some(r) if !r.is_empty() => r,
            _ => return Ok(None),
        };

        if let Some(id) = rest.strip_suffix("/input_items") {
            if !id.is_empty() && !id.contains('/') {
                return Ok(Some(self.handle_get_input_items(ctx, id).await));
            }
        } else if !rest.contains('/') {
            return Ok(Some(self.handle_get_response(ctx, rest).await));
        }

        Ok(None)
    }

    /// Serve `GET /v1/responses/{id}`.
    ///
    /// With `stream=true` this serves an incremental SSE replay of the stored
    /// event log ([`serve_replay`]); otherwise it returns the stored response
    /// object as JSON, unchanged.
    async fn handle_get_response(&self, ctx: &HttpFilterContext<'_>, id: &str) -> FilterAction {
        let parsed = match parse_get_response_query(ctx.request.uri.query()) {
            Ok(parsed) => parsed,
            Err(msg) => {
                debug!(response_id = id, error = %msg, "invalid get-response query parameter");
                return FilterAction::Reject(reject_invalid_input(&msg));
            },
        };

        let owner = match require_state_owner(ctx) {
            Ok(owner) => owner,
            Err(action) => return action,
        };

        let Some(store) = resolve_store(ctx, owner) else {
            return FilterAction::Reject(reject_store_error());
        };

        if parsed.stream {
            serve_replay(store, id, parsed.starting_after).await
        } else {
            Self::serve_stored_json(&store, id).await
        }
    }

    /// Serve the stored response object as JSON, unchanged.
    async fn serve_stored_json(store: &OwnerScopedResponseStore, id: &str) -> FilterAction {
        debug!(response_id = id, "retrieving stored response");
        match store.get_response(id).await {
            Ok(Some(record)) => {
                let body = serde_json::to_vec(&record.response_object).unwrap_or_default();
                FilterAction::Reject(
                    Rejection::status(200)
                        .with_header("content-type", "application/json")
                        .with_body(body),
                )
            },
            Ok(None) => {
                debug!(response_id = id, "response not found");
                FilterAction::Reject(reject_not_found(id))
            },
            Err(e) => {
                warn!(response_id = id, error = %e, "store lookup failed");
                FilterAction::Reject(reject_store_error())
            },
        }
    }

    /// Load a [`ResponseRecord`] from the store, returning a
    /// [`FilterAction`] rejection on store or not-found errors.
    async fn load_record(&self, ctx: &HttpFilterContext<'_>, id: &str) -> Result<ResponseRecord, FilterAction> {
        let owner = require_state_owner(ctx)?;
        let Some(store) = resolve_store(ctx, owner) else {
            return Err(FilterAction::Reject(reject_store_error()));
        };
        debug!(response_id = id, "retrieving input items");

        match store.get_response(id).await {
            Ok(Some(r)) => Ok(r),
            Ok(None) => {
                debug!(response_id = id, "response not found for input_items");
                Err(FilterAction::Reject(reject_not_found(id)))
            },
            Err(e) => {
                warn!(response_id = id, error = %e, "store lookup failed");
                Err(FilterAction::Reject(reject_store_error()))
            },
        }
    }

    /// Serve `GET /v1/responses/{id}/input_items`.
    async fn handle_get_input_items(&self, ctx: &HttpFilterContext<'_>, id: &str) -> FilterAction {
        let includes = match parse_include(ctx.request.uri.query()) {
            Ok(includes) => includes,
            Err(msg) => {
                debug!(response_id = id, error = %msg, "invalid input_items query parameter");
                return FilterAction::Reject(reject_invalid_input(&msg));
            },
        };
        let params = match parse_query_params(ctx.request.uri.query()) {
            Ok(p) => p,
            Err(msg) => {
                debug!(response_id = id, error = %msg, "invalid input_items query parameter");
                return FilterAction::Reject(reject_invalid_input(&msg));
            },
        };

        let record = match self.load_record(ctx, id).await {
            Ok(r) => r,
            Err(action) => return action,
        };
        build_input_items_response(id, &record, &params, includes)
    }
}

// -----------------------------------------------------------------------------
// GET Helpers
// -----------------------------------------------------------------------------

/// Build a paginated input items response from a stored record.
fn build_input_items_response(
    id: &str,
    record: &ResponseRecord,
    params: &ListParams,
    includes: IncludeFields,
) -> FilterAction {
    match list_input_items(record, params, includes) {
        Ok(page) => build_input_items_ok(id, &page),
        Err(StoreError::InvalidInput(msg)) => {
            debug!(response_id = id, error = %msg, "invalid input_items pagination parameter");
            FilterAction::Reject(reject_invalid_input(&msg))
        },
        Err(e) => {
            warn!(response_id = id, error = %e, "input_items pagination failed");
            FilterAction::Reject(reject_store_error())
        },
    }
}

/// Serialize a successful input items page into a 200 JSON response.
fn build_input_items_ok(id: &str, page: &InputItemPage) -> FilterAction {
    debug!(
        response_id = id,
        count = page.data.len(),
        has_more = page.has_more,
        "serving input items"
    );
    let bytes = match serde_json::to_vec(page) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!(response_id = id, error = %e, "input_items serialization failed");
            return FilterAction::Reject(reject_store_error());
        },
    };
    FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(bytes),
    )
}

/// Parse cursor-based pagination parameters from a query string.
///
/// Returns an error message suitable for a 400 response when the query
/// contains a malformed value, an out-of-range limit, an unknown order,
/// or an unsupported parameter.
///
/// Keys are percent-decoded before matching so both spellings of the
/// array-valued `include` parameter (`include[]` and its encoded
/// `include%5B%5D` form) resolve to the same name.
pub(super) fn parse_query_params(query: Option<&str>) -> Result<ListParams, String> {
    let Some(qs) = query else {
        return Ok(ListParams::default());
    };

    let mut params = ListParams::default();

    for pair in qs.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((raw_key, value)) = pair.split_once('=') else {
            let key = decode_query_component_strict(pair)?;
            reject_known_key_only_param(&key)?;
            continue;
        };
        let key = decode_query_component_strict(raw_key)?;
        apply_query_param(&mut params, &key, value)?;
    }

    Ok(params)
}

/// Apply a single query-string key/value pair to [`ListParams`].
fn apply_query_param(params: &mut ListParams, key: &str, value: &str) -> Result<(), String> {
    match key {
        "after" => {
            if value.is_empty() {
                return Err("Invalid value for 'after': cursor must not be empty.".to_owned());
            }
            params.cursor = Some(
                percent_encoding::percent_decode_str(value)
                    .decode_utf8_lossy()
                    .into_owned(),
            );
        },
        "limit" => params.limit = parse_limit(value)?,
        "order" => params.order = parse_order(value)?,
        // Include values are parsed and validated by `parse_include`.
        "include" | "include[]" => {},
        _ => return Err(format!("Unknown query parameter: '{key}'.")),
    }
    Ok(())
}

/// Parse and validate a `limit` query-string value.
fn parse_limit(value: &str) -> Result<u32, String> {
    let n: u32 = value
        .parse()
        .map_err(|_e| format!("Invalid value for 'limit': '{value}' is not a valid integer."))?;
    if n == 0 || n > MAX_PAGE_LIMIT {
        return Err(format!(
            "Invalid value for 'limit': must be between 1 and {MAX_PAGE_LIMIT}, got {n}."
        ));
    }
    Ok(n)
}

/// Parse and validate an `order` query-string value.
fn parse_order(value: &str) -> Result<Order, String> {
    match value {
        "asc" => Ok(Order::Ascending),
        "desc" => Ok(Order::Descending),
        _ => Err(format!(
            "Invalid value for 'order': must be 'asc' or 'desc', got '{value}'."
        )),
    }
}

/// Reject a key-only query component (no `=`) when it matches a known
/// parameter name. Unknown key-only components are ignored to match
/// OpenAI behavior.
fn reject_known_key_only_param(key: &str) -> Result<(), String> {
    match key {
        "limit" | "order" | "after" | "include" | "include[]" => {
            Err(format!("Missing value for query parameter '{key}'."))
        },
        _ => Ok(()),
    }
}

/// Known query parameters for `GET /v1/responses/{id}`, per the OpenAI spec.
pub(super) const GET_RESPONSE_KNOWN_PARAMS: &[&str] = &[
    "stream",
    "include",
    "include[]",
    "starting_after",
    "include_obfuscation",
];

/// Parsed query for `GET /v1/responses/{id}`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct GetResponseQuery {
    /// Whether the client requested an SSE replay (`stream=true`).
    pub(super) stream: bool,
    /// Replay cursor: only events with `sequence_number > starting_after` are
    /// returned. Requires `stream=true`.
    pub(super) starting_after: Option<u64>,
}

/// Parse and validate the query string for `GET /v1/responses/{id}`.
///
/// Recognizes `stream` (bool) and `starting_after` (u64 replay cursor, which
/// requires `stream=true`). `include`, `include[]`, and `include_obfuscation`
/// remain unsupported and are rejected, as are unknown parameters. Keys and
/// values are percent-decoded before validation. An absent or empty query yields
/// the default (plain JSON GET).
pub(super) fn parse_get_response_query(query: Option<&str>) -> Result<GetResponseQuery, String> {
    let mut parsed = GetResponseQuery::default();
    let Some(qs) = query else {
        return Ok(parsed);
    };

    for pair in qs.split('&') {
        if pair.is_empty() {
            continue;
        }

        let Some((raw_key, raw_value)) = pair.split_once('=') else {
            let key = percent_encoding::percent_decode_str(pair)
                .decode_utf8()
                .map_err(|_e| format!("Invalid percent-encoding in query parameter key '{pair}'."))?;
            if GET_RESPONSE_KNOWN_PARAMS.contains(&&*key) {
                return Err(format!("Missing value for query parameter '{key}'."));
            }
            return Err(format!("Unknown query parameter: '{key}'."));
        };

        let key = percent_encoding::percent_decode_str(raw_key)
            .decode_utf8()
            .map_err(|_e| format!("Invalid percent-encoding in query parameter key '{raw_key}'."))?;
        let value = percent_encoding::percent_decode_str(raw_value)
            .decode_utf8()
            .map_err(|_e| format!("Invalid percent-encoding in value for '{key}'."))?;

        apply_get_response_param(&mut parsed, &key, &value)?;
    }

    if parsed.starting_after.is_some() && !parsed.stream {
        return Err("The 'starting_after' parameter requires 'stream=true'.".to_owned());
    }

    Ok(parsed)
}

/// Apply a single decoded query parameter to [`GetResponseQuery`].
pub(super) fn apply_get_response_param(parsed: &mut GetResponseQuery, key: &str, value: &str) -> Result<(), String> {
    match key {
        "stream" => {
            parsed.stream = parse_stream_flag(value)?;
            Ok(())
        },
        "starting_after" => {
            parsed.starting_after = Some(parse_starting_after(value)?);
            Ok(())
        },
        "include" | "include[]" | "include_obfuscation" => {
            let name = if key == "include_obfuscation" {
                "include_obfuscation"
            } else {
                "include"
            };
            Err(format!(
                "The '{name}' parameter is not supported by the local response store."
            ))
        },
        _ => Err(format!("Unknown query parameter: '{key}'.")),
    }
}

/// Parse the `stream` query flag; only `true`/`false` are accepted.
fn parse_stream_flag(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!(
            "Invalid value for 'stream': must be 'true' or 'false', got '{value}'."
        )),
    }
}

/// Parse the `starting_after` cursor as a non-empty unsigned integer.
fn parse_starting_after(value: &str) -> Result<u64, String> {
    if value.is_empty() {
        return Err("Invalid value for 'starting_after': cursor must not be empty.".to_owned());
    }
    value
        .parse::<u64>()
        .map_err(|_e| format!("Invalid value for 'starting_after': '{value}' is not a valid integer."))
}

/// Build a 404 rejection with a Responses API error body.
fn reject_not_found(id: &str) -> Rejection {
    responses_error_rejection(
        404,
        "invalid_request_error",
        &format!("No response found with id '{id}'."),
    )
}

/// Build a 400 rejection for invalid client-supplied parameters.
fn reject_invalid_input(message: &str) -> Rejection {
    responses_error_rejection(400, "invalid_request_error", message)
}

/// Build a 500 rejection for internal store failures.
fn reject_store_error() -> Rejection {
    responses_error_rejection(500, "server_error", "Internal server error.")
}

// -----------------------------------------------------------------------------
// SSE Replay
// -----------------------------------------------------------------------------

/// Serve `GET /v1/responses/{id}?stream=true` as an incremental SSE replay.
///
/// Requires the response to exist for this owner (404 otherwise) and its event
/// log to have reached a terminal event (400 [`NO_REPLAY_LOG_MESSAGE`]
/// otherwise) — a failed or incomplete log never appears as a complete replay.
/// On success returns a [`FilterAction::StreamingTerminalResponse`] (200,
/// `text/event-stream`, `no-store`) whose body pages the log from the store; the
/// whole log is never held in memory.
async fn serve_replay(store: OwnerScopedResponseStore, id: &str, starting_after: Option<u64>) -> FilterAction {
    if let Err(action) = ensure_response_exists(&store, id).await {
        return action;
    }
    let terminal_sequence = match ensure_replayable_log(&store, id).await {
        Ok(sequence) => sequence,
        Err(action) => return action,
    };

    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    // `no-store`, not `no-cache`: the replay body is one owner's model output and
    // the request is authorized by the owner header, not the URL. `no-cache` still
    // lets a shared cache keyed on the URL store the body and revalidate, which
    // could hand one owner's replay to another; `no-store` forbids storing it.
    headers.insert(http::header::CACHE_CONTROL, http::HeaderValue::from_static("no-store"));

    debug!(response_id = id, "serving SSE replay");
    let body = ReplayStreamBody::new(store, id.to_owned(), starting_after, terminal_sequence);
    FilterAction::StreamingTerminalResponse(Box::new(
        StreamingTerminalResponse::new(200, Box::new(body)).with_headers(headers),
    ))
}

/// Require the response to exist for this owner; 404 when missing, 500 on error.
async fn ensure_response_exists(store: &OwnerScopedResponseStore, id: &str) -> Result<(), FilterAction> {
    match store.get_response(id).await {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            debug!(response_id = id, "replay: response not found");
            Err(FilterAction::Reject(reject_not_found(id)))
        },
        Err(e) => {
            warn!(response_id = id, error = %e, "replay: store lookup failed");
            Err(FilterAction::Reject(reject_store_error()))
        },
    }
}

/// Require a terminal-complete event log; 400 [`NO_REPLAY_LOG_MESSAGE`] when the
/// log is absent or incomplete, 500 on error. A failed or incomplete log never
/// appears as a complete replay.
///
/// On success returns the terminal event's sequence number. A replayable log ends
/// with exactly one terminal event, so this equals the log's maximum sequence and
/// bounds the replay: pages beyond it are a legitimate empty tail, while an empty
/// page short of it means rows were removed mid-replay.
async fn ensure_replayable_log(store: &OwnerScopedResponseStore, id: &str) -> Result<u64, FilterAction> {
    match store.event_log_status(id).await {
        Ok(EventLogStatus::Replayable { max_sequence }) => Ok(max_sequence),
        Ok(EventLogStatus::Absent | EventLogStatus::Incomplete { .. }) => {
            debug!(response_id = id, "replay: no replayable event log");
            Err(FilterAction::Reject(reject_invalid_input(NO_REPLAY_LOG_MESSAGE)))
        },
        Err(e) => {
            warn!(response_id = id, error = %e, "replay: event-log status lookup failed");
            Err(FilterAction::Reject(reject_store_error()))
        },
    }
}

/// Pull-based body that replays a stored response's SSE event log.
///
/// Each [`StreamingResponseBody::next_chunk`] fetches one page of rows with
/// [`OwnerScopedResponseStore::list_events_after`], re-encodes them to the canonical wire
/// SSE form, and returns them as one chunk. It stops after the terminal event, so
/// at most one page ([`REPLAY_PAGE_LIMIT`] rows) is held in memory at a time.
struct ReplayStreamBody {
    /// Owner-scoped store the body pages events through.
    store: OwnerScopedResponseStore,
    /// Response whose log is being replayed.
    response_id: String,
    /// Cursor: the next page starts after this sequence number.
    cursor: Option<u64>,
    /// Terminal event's sequence number, captured before streaming began. An empty
    /// page short of this boundary means the log was truncated mid-replay.
    terminal_sequence: u64,
    /// Whether the terminal event has been emitted; no further pages are read.
    done: bool,
}

impl ReplayStreamBody {
    /// Bind a replay body to an owner-scoped store, starting cursor, and the
    /// terminal boundary established when the log was confirmed replayable.
    fn new(
        store: OwnerScopedResponseStore,
        response_id: String,
        starting_after: Option<u64>,
        terminal_sequence: u64,
    ) -> Self {
        Self {
            store,
            response_id,
            cursor: starting_after,
            terminal_sequence,
            done: false,
        }
    }
}

#[async_trait]
impl StreamingResponseBody for ReplayStreamBody {
    async fn next_chunk(&mut self) -> Result<Option<Bytes>, FilterError> {
        if self.done {
            return Ok(None);
        }

        let events = self
            .store
            .list_events_after(&self.response_id, self.cursor, REPLAY_PAGE_LIMIT)
            .await
            .map_err(|e| -> FilterError { Box::new(e) })?;

        if events.is_empty() {
            self.done = true;
            // The terminal event sets `done` before we ever re-poll, so an empty page
            // here means either the caller's cursor already sits at/beyond the terminal
            // (a legitimate empty tail) or rows vanished mid-replay. Surface the latter
            // as an error rather than a clean EOF that hides the missing terminal frame.
            if self.cursor.is_some_and(|cursor| cursor >= self.terminal_sequence) {
                return Ok(None);
            }
            return Err(FilterError::from(format!(
                "openai_response_store: replay log for {} was truncated before its terminal event",
                self.response_id
            )));
        }

        let mut out = Vec::new();
        for event in &events {
            self.cursor = Some(event.sequence_number);
            encode_replay_event(event, &mut out);
            if event.terminal {
                self.done = true;
                break;
            }
        }
        Ok(Some(Bytes::from(out)))
    }

    async fn suppress(&mut self) -> Result<(), FilterError> {
        self.done = true;
        Ok(())
    }

    async fn cancel(&mut self) {
        self.done = true;
    }
}

/// Encode one stored event row back to the canonical wire SSE form
/// (`event: <type>\ndata: <payload>\n\n`). The stored payload is the original
/// event's `data` bytes, written verbatim with no parse/serialize round trip
/// that could reorder object keys or fail mid-write.
///
/// The payload is emitted as one `data:` field per line: an SSE decoder joins
/// multiple `data:` lines back with `\n`, so a payload carrying embedded
/// newlines still decodes to the original bytes rather than being truncated at
/// the first line. Normalized events are single-line compact JSON, so the common
/// case produces exactly one `data:` field.
fn encode_replay_event(event: &ResponseEventRecord, output: &mut Vec<u8>) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event.event_type.as_bytes());
    output.push(b'\n');
    for line in event.payload.split(|&b| b == b'\n') {
        output.extend_from_slice(b"data: ");
        output.extend_from_slice(line);
        output.push(b'\n');
    }
    output.push(b'\n');
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod encode_replay_event_tests {
    use bytes::Bytes;
    use praxis_filter::sse::SseDecoder;

    use super::{ResponseEventRecord, StateOwner, encode_replay_event};

    /// Build a minimal event record carrying `payload` for the encoder under test.
    fn record_with_payload(event_type: &str, payload: &[u8]) -> ResponseEventRecord {
        ResponseEventRecord {
            response_id: "resp_replay".to_owned(),
            owner: StateOwner::from_trusted_parts("t", "i", "s").expect("valid owner parts"),
            sequence_number: 1,
            event_type: event_type.to_owned(),
            payload: payload.to_vec(),
            terminal: false,
            created_at: 0,
        }
    }

    /// Decode a single-record `frame`, returning its event name and joined `data`.
    fn decode_single(frame: &[u8]) -> (Option<String>, Vec<u8>) {
        let mut decoder = SseDecoder::new();
        let batch = decoder.push(&Bytes::copy_from_slice(frame));
        let record = batch.records.first().expect("one SSE record decoded");
        let event = record
            .event()
            .map(|name| String::from_utf8(name.to_vec()).expect("utf8 event name"));
        (event, record.data().to_vec())
    }

    /// A single-line payload replays byte-identically through the SSE decoder.
    #[test]
    fn single_line_payload_round_trips() {
        let payload = br#"{"type":"response.completed","sequence_number":1}"#;
        let mut out = Vec::new();
        encode_replay_event(&record_with_payload("response.completed", payload), &mut out);
        let (event, data) = decode_single(&out);
        assert_eq!(event.as_deref(), Some("response.completed"));
        assert_eq!(data, payload.to_vec());
    }

    /// A payload with an embedded newline is emitted as multiple `data:` lines and
    /// an SSE decoder rejoins them into the original bytes rather than truncating
    /// at the first line (the regression a single `data:` field would cause).
    #[test]
    fn multi_line_payload_round_trips_without_truncation() {
        let payload = b"{\"a\":1}\n{\"b\":2}";
        let mut out = Vec::new();
        encode_replay_event(&record_with_payload("response.output_text.delta", payload), &mut out);
        let text = std::str::from_utf8(&out).expect("utf8 frame");
        assert_eq!(text.matches("data: ").count(), 2, "one data field per payload line");
        let (event, data) = decode_single(&out);
        assert_eq!(event.as_deref(), Some("response.output_text.delta"));
        assert_eq!(data, payload.to_vec());
    }
}
