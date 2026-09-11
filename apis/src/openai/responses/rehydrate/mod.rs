// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Rehydrate filter: validates `previous_response_id` by
//! fetching the stored response, confirming its status is
//! `"completed"`, and populating [`ResponsesState`] with the
//! full conversation history (stored turns + current input).
//!
//! The request body is **not** modified; downstream filters
//! read from `ResponsesState.messages` instead.
//!
//! ## Response-side `previous_response_id` restore
//!
//! When history is rehydrated, the proxy strips `previous_response_id`
//! from the upstream request (prior turns are replayed via the `input`
//! array), so the backend echoes `null`. The Responses API contract
//! always echoes the caller's `previous_response_id`, so the response
//! phase restores it on the way out:
//!
//! - **Non-streaming** (`application/json`): the finite body is buffered and the ID rewritten in one shot
//!   ([`restore_previous_response_id`]).
//! - **Streaming** (`text/event-stream`): frames pass through byte-for-byte; only a response-lifecycle frame's `data:`
//!   JSON payload is rewritten in place, and only that. No frame is reconstructed, so `id:`, `retry:`, comments, and
//!   unknown SSE fields survive untouched. At most one partial frame is ever buffered — never the whole stream
//!   ([`restore_previous_response_id_stream_chunk`]).
//!
//! Both paths fail open: any parse error passes the bytes through
//! untouched and never errors the request.
//!
//! [`ResponsesState`]: super::state::ResponsesState

use std::collections::HashSet;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use praxis_filter::{
    EmptyFilterConfig, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, trace, warn};

use super::{
    DEFAULT_STORE_NAME, DEFAULT_TENANT_ID, TENANT_METADATA_KEY, append_stored_input_items,
    canonical_openresponses_replay_item, error::responses_error_rejection, extract_conversation_id,
    state::ResponsesState,
};
use crate::{
    is_event_stream_content_type,
    store::{ConversationRecord, ResponseRecord, ResponseStoreRegistry},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key for previous response input token count.
const PREV_USAGE_INPUT_KEY: &str = "responses.previous_usage_input_tokens";

/// Metadata key for previous response output token count.
const PREV_USAGE_OUTPUT_KEY: &str = "responses.previous_usage_output_tokens";

/// Metadata key for previous response total token count.
const PREV_USAGE_TOTAL_KEY: &str = "responses.previous_usage_total_tokens";

// -----------------------------------------------------------------------------
// RehydrateFilter
// -----------------------------------------------------------------------------

/// Validates `previous_response_id` by fetching the stored
/// response, confirming its status is `"completed"`, and
/// populating `ResponsesState` with the full conversation
/// history (stored turns + current input).
///
/// The request body is **not** modified; downstream filters
/// read from `ResponsesState.messages` instead.
///
/// # YAML
///
/// ```yaml
/// filter: openai_responses_rehydrate
/// ```
pub struct RehydrateFilter;

impl RehydrateFilter {
    /// Create a filter from YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown
    /// fields.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        // The filter has no tunable options. Parsing still runs so that
        // `deny_unknown_fields` rejects any config keys, including the removed
        // `max_history_bytes` / `max_history_items` limits.
        let _: EmptyFilterConfig = parse_filter_config("openai_responses_rehydrate", config)?;
        Ok(Box::new(Self))
    }

    /// Parse body, resolve rehydration source (`previous_response_id` or
    /// `conversation`), and populate [`ResponsesState`] with the full
    /// conversation history.
    ///
    /// `previous_response_id` takes precedence when both fields are
    /// present.
    async fn rehydrate(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(bytes) = body.as_ref() else {
            return Ok(FilterAction::Release);
        };
        match parse_body_and_extract_id(bytes) {
            Ok((body, Some(id))) => self.rehydrate_from_response(ctx, body, id).await,
            Ok((body, None)) => self.rehydrate_from_conversation(ctx, body).await,
            Err(action) => Ok(action),
        }
    }

    /// Rehydrate from a stored response.
    async fn rehydrate_from_response(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        parsed_body: Value,
        prev_id: String,
    ) -> Result<FilterAction, FilterError> {
        let tenant_id = ctx
            .get_metadata(TENANT_METADATA_KEY)
            .unwrap_or(DEFAULT_TENANT_ID)
            .to_owned();
        let record = match fetch_and_validate_previous(ctx, &tenant_id, &prev_id).await {
            Ok(r) => r,
            Err(action) => return Ok(action),
        };
        let previous_tools = collect_mcp_tool_listings(&record);
        let previous_usage = record.response_object.get("usage").filter(|u| !u.is_null()).cloned();
        let stored = stored_messages_for_response(record);
        let state = build_state(parsed_body, stored, previous_tools, previous_usage);
        install_rehydrated_state(ctx, state);
        debug!(previous_response_id = %prev_id, "previous response validated, state populated");
        ctx.set_metadata("responses.previous_response_id", prev_id);
        Ok(FilterAction::Release)
    }

    /// Rehydrate from a stored conversation when no `previous_response_id`
    /// is present.
    async fn rehydrate_from_conversation(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        parsed_body: Value,
    ) -> Result<FilterAction, FilterError> {
        let conv_id = match resolve_conversation_id(&parsed_body) {
            Ok(id) => id,
            Err(action) => return Ok(action),
        };
        let tenant_id = ctx
            .get_metadata(TENANT_METADATA_KEY)
            .unwrap_or(DEFAULT_TENANT_ID)
            .to_owned();
        let record = match fetch_conversation(ctx, &tenant_id, &conv_id).await {
            Ok(r) => r,
            Err(action) => return Ok(action),
        };
        let stored = stored_messages_for_conversation(record);
        let state = build_state(parsed_body, stored, vec![], None);
        install_rehydrated_state(ctx, state);
        debug!(conversation_id = %conv_id, "conversation rehydrated, state populated");
        Ok(FilterAction::Release)
    }
}

#[async_trait]
impl HttpFilter for RehydrateFilter {
    fn name(&self) -> &'static str {
        "openai_responses_rehydrate"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    /// `StreamBuffer` so the protocol layer assembles the complete
    /// request body before delivering it at end-of-stream.
    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    /// `ReadWrite` so the response phase can restore the caller's
    /// `previous_response_id` into the response body.
    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    /// Streaming by default. A finite JSON response that requires the
    /// `previous_response_id` restore selects a bounded `StreamBuffer`
    /// dynamically in [`Self::on_response`].
    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if ctx.request.method != http::Method::POST {
            return Ok(FilterAction::Continue);
        }

        if is_responses_cancel_path(ctx.request.uri.path()) {
            return Ok(FilterAction::Release);
        }

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            return Ok(FilterAction::Release);
        }

        self.rehydrate(ctx, body).await
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // The response header is only available in this header phase, so every
        // eligibility decision (status + content-type + rehydration state) is
        // made here and carried into `on_response_body` via filter state. A finite
        // JSON response is buffered and rewritten in one shot; a streaming (SSE)
        // response is rewritten frame-by-frame without ever buffering the stream.
        // The two shapes are mutually exclusive (content-type), so only one path
        // ever arms.
        if !arm_json_restore(ctx) {
            arm_streaming_restore(ctx);
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // Streaming (SSE) restore runs on every chunk: rewrite the
        // `previous_response_id` inside response-lifecycle frames as they arrive,
        // never assembling the whole stream in memory.
        if let Some(mut armed) = ctx.remove_filter_state::<RestorePreviousResponseIdStream>() {
            let keep_armed = restore_previous_response_id_stream_chunk(&mut armed, body, end_of_stream);
            // Re-arm for the next chunk unless the stream ended or an overflow
            // disarmed the restore (fail-open: the buffered remainder is flushed raw
            // and the rest of the stream passes through untouched).
            if keep_armed && !end_of_stream {
                ctx.insert_filter_state(armed);
            }
            return Ok(FilterAction::Continue);
        }

        // Non-streaming JSON restore runs once, when the buffered body is complete.
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        // `response_header` is gone by the body phase; the armed decision from
        // `on_response` is the sole signal that a restore is required.
        let Some(armed) = ctx.remove_filter_state::<RestorePreviousResponseId>() else {
            return Ok(FilterAction::Continue);
        };

        restore_previous_response_id(armed.previous_response_id, body);
        Ok(FilterAction::Continue)
    }
}

/// Arm the non-streaming JSON `previous_response_id` restore when the response is
/// an eligible finite JSON body, returning whether it was armed.
///
/// Buffers the finite JSON response so `on_response_body` can restore the caller's
/// `previous_response_id`. Re-serializing the body changes its length, so
/// `Content-Length` must go — core recomputes it from the buffered body
/// (byte-identical when the body ends up unchanged). Nothing else is touched:
/// `eligible_previous_response_id` already declined any response carrying a
/// `Content-Encoding`, `Content-Range`, or a body validator / integrity digest
/// (`ETag`, `Last-Modified`, `Content-MD5`, `Digest`, `Content-Digest`,
/// `Repr-Digest`), so an eligible response has no header describing the exact
/// upstream bytes that a rewrite could invalidate.
///
/// Declining validator-bearing responses up front — rather than stripping the
/// validators here — is what keeps this sound. Praxis commits the response headers
/// before `on_response_body` runs, so the header phase cannot yet know whether the
/// body will actually be rewritten (that depends on it parsing as a Responses
/// resource). Stripping validators unconditionally would therefore also drop them
/// from a body we then leave unchanged (an unexpected non-Responses JSON shape),
/// handing the client an unchanged body with missing validators.
///
/// Caching-policy (`Cache-Control`, `Age`, ...), routing, tracing, and
/// `Content-Type` headers are unrelated to the byte content and are preserved so
/// the response reaches the client almost unchanged.
fn arm_json_restore(ctx: &mut HttpFilterContext<'_>) -> bool {
    let Some(prev_id) = eligible_previous_response_id(ctx) else {
        return false;
    };

    ctx.set_response_body_mode(BodyMode::StreamBuffer {
        max_bytes: Some(MAX_JSON_BODY_BYTES),
    });
    if let Some(response) = &mut ctx.response_header {
        response.headers.remove(http::header::CONTENT_LENGTH);
    }
    ctx.response_headers_modified = true;
    ctx.insert_filter_state(RestorePreviousResponseId {
        previous_response_id: prev_id,
    });

    true
}

/// Arm the streaming (SSE) `previous_response_id` restore when the response is an
/// eligible event stream, returning whether it was armed.
///
/// Unlike the JSON path, the stream is **not** buffered: `BodyMode::Stream` (the
/// default from [`RehydrateFilter::response_body_mode`]) is kept and each lifecycle
/// frame is rewritten as it arrives (see `restore_previous_response_id_stream_chunk`).
///
/// `eligible_previous_response_id_stream` already declined any response carrying a
/// body validator / integrity digest (via [`describes_exact_upstream_bytes`]), so an
/// eligible event stream has no such header to strip — the same fail-safe the JSON
/// path relies on. Only `Content-Length` is dropped, and only if present: rewriting
/// frames changes the byte length, and an event stream is chunked with no length to
/// recompute. (SSE normally carries none, so this is usually a no-op;
/// `response_headers_modified` is flipped only when a header actually changed.)
fn arm_streaming_restore(ctx: &mut HttpFilterContext<'_>) -> bool {
    let Some(prev_id) = eligible_previous_response_id_stream(ctx) else {
        return false;
    };

    let mut changed = false;
    if let Some(response) = &mut ctx.response_header
        && response.headers.remove(http::header::CONTENT_LENGTH).is_some()
    {
        changed = true;
    }
    if changed {
        ctx.response_headers_modified = true;
    }
    ctx.insert_filter_state(RestorePreviousResponseIdStream {
        pending: BytesMut::new(),
        scan_from: 0,
        scan_at_line_start: true,
        max_buffer_bytes: MAX_JSON_BODY_BYTES,
        previous_response_id: prev_id,
    });

    true
}

/// Marker carrying the caller's `previous_response_id` from the response header
/// phase to the response body phase.
///
/// The eligibility decision requires `ctx.response_header`, which is only
/// present during [`RehydrateFilter::on_response`]; this state hands the ID to
/// [`RehydrateFilter::on_response_body`], where the header is no longer available.
struct RestorePreviousResponseId {
    /// The `previous_response_id` the caller supplied, to echo back.
    previous_response_id: String,
}

/// Per-chunk state for restoring `previous_response_id` into a streaming (SSE)
/// Responses response, shuttled across `on_response_body` calls.
///
/// Unlike the one-shot [`RestorePreviousResponseId`] marker, this retains only the
/// raw bytes of a trailing partial frame between chunks, so a single incomplete
/// frame is ever held in memory — never the whole stream. When nothing is carried
/// over, a chunk's completed frames are forwarded directly as zero-copy slices of the
/// incoming buffer, never passing through `pending`. The raw retention is what lets
/// the fail-open path (oversized frame / unterminated end-of-stream) flush the
/// buffered remainder verbatim instead of losing it.
///
/// `pending` is a [`BytesMut`] so a carried-over partial can grow by appending in
/// place (amortized) and shed forwarded frames off its front with a zero-copy
/// `split_to`, and `scan_from` records how far it has already been scanned for a frame
/// boundary. Together these keep the work linear in the stream size: a single frame
/// split across many chunks is appended and scanned once, never re-copied or
/// re-scanned from byte zero on every chunk.
struct RestorePreviousResponseIdStream {
    /// Raw bytes of the frame still being assembled: everything received but not yet
    /// forwarded. Holds at most one incomplete SSE frame between chunks.
    pending: BytesMut,
    /// How many leading bytes of `pending` have already been scanned for a frame
    /// boundary without finding one, so the next chunk resumes here instead of
    /// rescanning from zero. May point at a lone trailing CR whose terminator is
    /// still ambiguous.
    scan_from: usize,
    /// Whether `scan_from` lands on an SSE line start (the only cross-chunk state the
    /// empty-line frame-boundary detection needs to resume correctly).
    scan_at_line_start: bool,
    /// Maximum bytes `pending` may hold before the restore fails open: a single
    /// frame larger than this can never complete, so the remainder is flushed raw
    /// and the restore disarms.
    max_buffer_bytes: usize,
    /// The `previous_response_id` the caller supplied, echoed into every
    /// response-lifecycle frame.
    previous_response_id: String,
}

/// Return the caller's `previous_response_id` when it must be restored into the
/// response body, or `None` when the response is ineligible.
///
/// When history was rehydrated, the proxy strips `previous_response_id` from the
/// upstream request (prior turns are replayed via the `input` array and the
/// backend never sees the ID), so the backend echoes `null`. The Responses API
/// contract always echoes the caller's `previous_response_id`, so it is restored
/// on the way out. Only a finite, identity-coded, complete `200 OK` JSON response
/// is eligible; streaming SSE, content-encoded, ranged/partial, non-`200`, and
/// responses carrying a body validator or integrity digest (`ETag`,
/// `Last-Modified`, `Content-MD5`, `Digest`, `Content-Digest`, `Repr-Digest`) are
/// left untouched, so re-serializing the body can never invalidate a header that
/// described the exact upstream bytes.
fn eligible_previous_response_id(ctx: &HttpFilterContext<'_>) -> Option<String> {
    // Clone the ID at the ownership boundary so it outlives the borrow on
    // `extensions` and can be carried into the body phase.
    let prev_id = ctx
        .extensions
        .get::<ResponsesState>()
        .filter(|state| state.history_rehydrated)
        .and_then(|state| state.previous_response_id.clone())?;

    let resp = ctx.response_header.as_ref()?;
    // Require an ordinary `200 OK`. Restoring the ID re-serializes the full body,
    // so any other success shape is not a complete, self-contained Responses
    // resource to rewrite: `206 Partial Content` is a byte fragment, `204 No
    // Content` has no body, and `202 Accepted` background handoffs are echoed
    // verbatim. All are passed through untouched.
    if resp.status != http::StatusCode::OK {
        return None;
    }

    // Decline any response whose headers describe or constrain the exact upstream
    // bytes; restoring the ID re-serializes the body and would invalidate them.
    // See [`describes_exact_upstream_bytes`] for the full set and rationale.
    if describes_exact_upstream_bytes(&resp.headers) {
        return None;
    }

    let content_type = resp
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    let is_json = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("application/json");

    is_json.then_some(prev_id)
}

/// Whether any response header describes or constrains the exact upstream bytes,
/// making the body unsafe to re-serialize when restoring `previous_response_id`:
///
/// - `Content-Encoding` — an opaque (e.g. compressed) body that would fail to parse as JSON; stripping the label would
///   ship encoded bytes mislabeled as identity JSON. (Mirrors the encoded-SSE decline in `openai_responses`
///   `stream_events`; see issue #668 for the same defense on the streaming path.)
/// - `Content-Range` — a fragment of a larger representation that cannot be soundly rewritten.
/// - `ETag` / `Last-Modified` / `Content-MD5` (RFC 1864) / `Digest` / `Content-Digest` / `Repr-Digest` (RFC 9530) —
///   body validators and integrity digests that name the exact upstream representation. Re-serializing the body
///   invalidates them, but Praxis commits response headers before `on_response_body` runs, so the proxy can neither
///   recompute them from the rewritten body nor tell in the header phase whether a rewrite will actually happen.
///   Stripping them there would drop them from a body we then leave unchanged (an unexpected non-Responses JSON shape).
///   Declining up front passes the response through byte-identical with its validators intact — a validator-bearing
///   `POST /v1/responses` body does not occur in practice (OpenAI and vLLM never send these here), so this only forgoes
///   the cosmetic ID echo in an anomalous case and never corrupts a response.
///
/// (`Content-MD5` and the RFC 9530 digests have no typed constant in `http`;
/// matched by lowercase name — `HeaderMap::contains_key` is case-insensitive.)
fn describes_exact_upstream_bytes(headers: &http::HeaderMap) -> bool {
    headers.contains_key(http::header::CONTENT_ENCODING)
        || headers.contains_key(http::header::CONTENT_RANGE)
        || headers.contains_key(http::header::ETAG)
        || headers.contains_key(http::header::LAST_MODIFIED)
        || headers.contains_key("content-md5")
        || headers.contains_key("digest")
        || headers.contains_key("content-digest")
        || headers.contains_key("repr-digest")
}

/// Return the caller's `previous_response_id` when a streaming (`text/event-stream`)
/// response must have it restored, or `None` when the response is ineligible.
///
/// Mirrors [`eligible_previous_response_id`] but selects an SSE response instead of
/// a finite JSON body. Only a `200 OK`, identity-coded, non-ranged event stream
/// from a rehydrated turn carrying a caller `previous_response_id` is eligible;
/// non-`200` and non-SSE responses, and any response whose headers describe the
/// exact upstream bytes (a `Content-Encoding`, `Content-Range`, or a body validator
/// / integrity digest — see [`describes_exact_upstream_bytes`]), are left
/// untouched.
fn eligible_previous_response_id_stream(ctx: &HttpFilterContext<'_>) -> Option<String> {
    // Clone the ID at the ownership boundary so it outlives the borrow on
    // `extensions` and can be carried into the body phase.
    let prev_id = ctx
        .extensions
        .get::<ResponsesState>()
        .filter(|state| state.history_rehydrated)
        .and_then(|state| state.previous_response_id.clone())?;

    let resp = ctx.response_header.as_ref()?;
    // Require an ordinary `200 OK`; any other status is not a live event stream to
    // rewrite (mirrors the non-streaming decline of `204`/`206`/`202`).
    if resp.status != http::StatusCode::OK {
        return None;
    }

    // Decline any response whose headers describe or constrain the exact upstream
    // bytes; rewriting lifecycle frames re-serializes their payloads and would
    // invalidate them. Same fail-safe as the non-streaming path — see
    // [`describes_exact_upstream_bytes`]. (`Content-Encoding` in particular can't
    // be parsed as SSE anyway, so an encoded stream is passed through verbatim,
    // mirroring the encoded-SSE decline in `openai_responses` stream_events; see
    // issue #668.)
    if describes_exact_upstream_bytes(&resp.headers) {
        return None;
    }

    let content_type = resp
        .headers
        .get(http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();

    is_event_stream_content_type(content_type).then_some(prev_id)
}

/// Restore the caller's `previous_response_id` into the buffered Responses
/// resource, replacing the `null` the backend echoed after the proxy stripped
/// the ID from the upstream request.
fn restore_previous_response_id(prev_id: String, body: &mut Option<Bytes>) {
    let Some(bytes) = body.as_ref() else {
        return;
    };
    let Ok(mut parsed) = serde_json::from_slice::<Value>(bytes) else {
        debug!("rehydrate: response body is not JSON, skipping previous_response_id restore");
        return;
    };
    let Some(object) = parsed.as_object_mut() else {
        return;
    };
    // Only rewrite a Responses resource, never an unexpected success shape.
    if object.get("object").and_then(Value::as_str) != Some("response") {
        return;
    }

    object.insert("previous_response_id".to_owned(), Value::String(prev_id));
    match serde_json::to_vec(&parsed) {
        Ok(serialized) => {
            *body = Some(Bytes::from(serialized));
            trace!("restored caller previous_response_id into response body");
        },
        Err(error) => warn!(error = %error, "rehydrate: failed to re-serialize response body"),
    }
}

/// Forward every complete frame in this chunk, restoring `previous_response_id`
/// into response-lifecycle frames along the way, and buffer only a trailing partial
/// frame for the next chunk — the whole stream is never assembled in memory.
///
/// Frames are spliced, not reconstructed: a non-lifecycle frame (deltas, comment
/// heartbeats, anything without the top-level `response` object) is forwarded
/// byte-for-byte, and a lifecycle frame has only its `data:` JSON payload rewritten
/// — every other line (`event:`, `id:`, `retry:`, comments, unknown fields) and
/// every original line terminator is preserved verbatim.
///
/// Ownership and cost: the incoming chunk is not copied to detect or forward frames.
/// With nothing carried over (the common case) it takes the zero-copy
/// [`restore_stream_fresh_chunk`] fast path; a partial that spans chunk boundaries
/// takes [`restore_stream_buffered_chunk`], which appends in place and resumes
/// scanning where it left off so a frame split across many chunks is copied and
/// scanned once — never re-copied or re-scanned from byte zero per chunk. See the
/// repository ownership guidance in `AGENTS.md`.
///
/// Returns `true` to keep the restore armed, or `false` when it disarms and the
/// remainder of the stream passes through untouched. Fail-open is non-negotiable and
/// bounded by `max_buffer_bytes`: a single frame that exceeds the limit —
/// whether still incomplete or already terminated — is never JSON-parsed; instead
/// every buffered byte from the current position onward is flushed **raw** (nothing
/// dropped, nothing errored) and the restore disarms. At `end_of_stream` any
/// unterminated trailing bytes are likewise flushed raw so no bytes are ever withheld
/// from the client.
fn restore_previous_response_id_stream_chunk(
    armed: &mut RestorePreviousResponseIdStream,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> bool {
    let incoming = body.take().unwrap_or_default();
    if armed.pending.is_empty() {
        restore_stream_fresh_chunk(armed, &incoming, body, end_of_stream)
    } else {
        restore_stream_buffered_chunk(armed, &incoming, body, end_of_stream)
    }
}

/// Fast path: nothing carried over, so scan the incoming chunk directly and forward
/// its complete frames as zero-copy slices of the original `Bytes`. Only a trailing
/// partial frame is copied into `pending` for the next chunk.
fn restore_stream_fresh_chunk(
    armed: &mut RestorePreviousResponseIdStream,
    incoming: &Bytes,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> bool {
    let scan = scan_complete_frames(
        incoming,
        &armed.previous_response_id,
        armed.max_buffer_bytes,
        ScanResume {
            from: 0,
            at_line_start: true,
        },
        end_of_stream,
    );
    let remainder = incoming.len() - scan.cursor;
    if flush_streaming_restore(&scan, remainder, armed.max_buffer_bytes, end_of_stream) {
        *body = finalize_view(scan.out, incoming, scan.run_start, incoming.len());
        return false;
    }
    *body = finalize_view(scan.out, incoming, scan.run_start, scan.cursor);
    armed
        .pending
        .extend_from_slice(incoming.get(scan.cursor..).unwrap_or_default());
    armed.scan_from = scan.resume_from - scan.cursor;
    armed.scan_at_line_start = scan.resume_at_line_start;
    true
}

/// Carry-over path: a partial frame spans chunk boundaries. Append the chunk to
/// `pending` in place (amortized), resume scanning from `scan_from` so buffered bytes
/// are never re-scanned, and shed forwarded frames off the front with a zero-copy
/// `split_to` — keeping total work linear in the stream size.
///
/// Cap semantics: `max_buffer_bytes` bounds a single SSE frame, not a transport chunk,
/// so whether the id is restored never depends on how the transport split the bytes.
/// The restore fails open only when a frame genuinely exceeds the budget — a complete
/// frame caught inside [`scan_complete_frames`] before it is parsed, or the retained
/// trailing partial (`remainder`) once complete frames are shed. A chunk that finishes
/// a valid carried frame and carries further frames is processed in full. The only
/// buffer persisted across callbacks is that trailing partial, always
/// `<= max_buffer_bytes`. The in-place append peaks transiently at `pending + incoming`
/// (the retained partial plus one transport chunk) before the shed — inherent to
/// detecting a frame boundary across the pending/incoming seam, and only near the cap
/// when a single frame is already near it.
fn restore_stream_buffered_chunk(
    armed: &mut RestorePreviousResponseIdStream,
    incoming: &Bytes,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> bool {
    armed.pending.extend_from_slice(incoming);
    let scan = scan_complete_frames(
        &armed.pending,
        &armed.previous_response_id,
        armed.max_buffer_bytes,
        ScanResume {
            from: armed.scan_from,
            at_line_start: armed.scan_at_line_start,
        },
        end_of_stream,
    );
    let remainder = armed.pending.len() - scan.cursor;
    if flush_streaming_restore(&scan, remainder, armed.max_buffer_bytes, end_of_stream) {
        *body = flush_pending(scan.out, &mut armed.pending, scan.run_start);
        return false;
    }
    *body = emit_pending_prefix(scan.out, &mut armed.pending, scan.cursor);
    armed.scan_from = scan.resume_from - scan.cursor;
    armed.scan_at_line_start = scan.resume_at_line_start;
    true
}

/// Decide whether the restore must fail open on this chunk — a complete frame exceeded
/// the parse budget, the trailing partial alone exceeded it, or the stream ended — and
/// log the budget breach once. When `true`, the caller flushes from the current run to
/// the buffer end raw and disarms.
fn flush_streaming_restore(scan: &FrameScan, remainder: usize, max_buffer_bytes: usize, end_of_stream: bool) -> bool {
    let budget_hit = scan.overflow || remainder > max_buffer_bytes;
    if budget_hit {
        debug!(
            limit = max_buffer_bytes,
            frame_overflow = scan.overflow,
            "rehydrate: SSE frame exceeds parse budget; flushing raw and disarming restore"
        );
    }
    budget_hit || end_of_stream
}

/// Emit the forwarded complete frames on the carry-over path and drop them off the
/// front of `pending`, leaving only the trailing partial. Zero-copy (`split_to` +
/// `freeze`) when nothing was rewritten; otherwise ships the rewritten buffer and
/// discards the consumed prefix. `None` when no complete frame was produced.
fn emit_pending_prefix(out: Option<Vec<u8>>, pending: &mut BytesMut, cursor: usize) -> Option<Bytes> {
    let consumed = pending.split_to(cursor);
    match out {
        Some(buf) => (!buf.is_empty()).then(|| Bytes::from(buf)),
        None => (!consumed.is_empty()).then(|| consumed.freeze()),
    }
}

/// Flush the buffered remainder raw when failing open on the carry-over path. Any
/// rewritten prefix in `out` covers `pending[..run_start]`; append the raw tail
/// `pending[run_start..]`. Zero-copy when nothing was rewritten (`out` is `None`, so
/// `run_start` is 0 and the whole buffer is handed out unchanged).
fn flush_pending(out: Option<Vec<u8>>, pending: &mut BytesMut, run_start: usize) -> Option<Bytes> {
    if let Some(mut buf) = out {
        buf.extend_from_slice(pending.get(run_start..).unwrap_or_default());
        (!buf.is_empty()).then(|| Bytes::from(buf))
    } else {
        let all = std::mem::take(pending);
        (!all.is_empty()).then(|| all.freeze())
    }
}

/// Result of walking the complete frames at the front of a chunk's buffer.
struct FrameScan {
    /// The rewritten output buffer, or `None` while every frame so far is a
    /// zero-copy pass-through (no rewrite has forced a fresh buffer yet).
    out: Option<Vec<u8>>,
    /// Byte offset just past the last complete frame; the rest of the buffer is a
    /// trailing partial frame.
    cursor: usize,
    /// Start of the run of forwarded-but-not-yet-copied bytes; where a fail-open flush
    /// must begin so it re-emits any deferred pass-through prefix. Stays 0 until the
    /// first rewrite, then tracks `cursor`.
    run_start: usize,
    /// A complete frame exceeded `max_buffer_bytes`, forcing a raw fail-open flush
    /// before it was ever parsed.
    overflow: bool,
    /// Where the next chunk must resume scanning the trailing partial (offset into the
    /// scanned buffer), so buffered bytes are never re-scanned.
    resume_from: usize,
    /// Whether `resume_from` lands on an SSE line start.
    resume_at_line_start: bool,
}

/// The resume point of a resumable frame-boundary scan.
#[derive(Clone, Copy)]
struct ScanResume {
    /// Offset into the buffer where scanning resumes; bytes before it are never
    /// re-examined.
    from: usize,
    /// Whether `from` lands on an SSE line start.
    at_line_start: bool,
}

/// Walk the complete SSE frames of `buf`, starting the boundary scan at `resume.from`
/// (treated as a line start iff `resume.at_line_start`) and restoring
/// `previous_response_id` into lifecycle frames. `out` stays `None` (pure pass-through
/// by zero-copy slice) until the first rewrite forces a buffer; from then on every
/// frame is copied into it so the output stays contiguous. A complete frame that
/// exceeds `max_buffer_bytes` stops the walk with `overflow`, before it is ever parsed.
/// When `at_stream_end` is set, a final frame terminated by a bare CR is still detected.
fn scan_complete_frames(
    buf: &[u8],
    prev_id: &str,
    max_buffer_bytes: usize,
    resume: ScanResume,
    at_stream_end: bool,
) -> FrameScan {
    let (mut out, mut overflow): (Option<Vec<u8>>, bool) = (None, false);
    let (mut cursor, mut run_start) = (0_usize, 0_usize);
    let (mut scan_pos, mut at_line_start) = (resume.from, resume.at_line_start);
    let (resume_from, resume_at_line_start) = loop {
        match scan_frame_end(buf, scan_pos, at_line_start, at_stream_end) {
            FrameScanState::Incomplete {
                resume_from,
                at_line_start,
            } => break (resume_from, at_line_start),
            FrameScanState::Complete(end) => {
                if end - cursor > max_buffer_bytes {
                    overflow = true;
                    break (end, true);
                }
                accumulate_frame(&mut out, &mut run_start, buf, cursor..end, prev_id);
                cursor = end;
                scan_pos = end;
                at_line_start = true;
            },
        }
    };
    FrameScan {
        out,
        cursor,
        run_start,
        overflow,
        resume_from,
        resume_at_line_start,
    }
}

/// Fold one complete frame `buf[cursor..end]` into the running output: rewrite a
/// lifecycle frame's payload (lazily materializing `out` and capturing the deferred
/// pass-through prefix on the first rewrite), or, once `out` exists, copy a
/// pass-through frame into it verbatim. While `out` is `None`, pass-through frames are
/// left in place for a later zero-copy slice and `run_start` stays put.
fn accumulate_frame(
    out: &mut Option<Vec<u8>>,
    run_start: &mut usize,
    buf: &[u8],
    frame_range: core::ops::Range<usize>,
    prev_id: &str,
) {
    let core::ops::Range { start, end } = frame_range;
    let frame = buf.get(start..end).unwrap_or_default();
    if let Some(rewritten) = rewritten_lifecycle_frame(frame, prev_id) {
        let dst = out.get_or_insert_with(|| buf.get(*run_start..start).unwrap_or_default().to_vec());
        dst.extend_from_slice(&rewritten);
        *run_start = end;
    } else if let Some(dst) = out.as_mut() {
        dst.extend_from_slice(frame);
        *run_start = end;
    }
}

/// Finalize a chunk's output from the frame scan. When a rewrite occurred, append the
/// raw tail `view[from..end]` to the rewritten buffer; otherwise ship `view[from..end]`
/// as a zero-copy slice. `None` when the result is empty (a withheld partial frame
/// emits no bytes).
fn finalize_view(out: Option<Vec<u8>>, view: &Bytes, from: usize, end: usize) -> Option<Bytes> {
    if let Some(mut buf) = out {
        buf.extend_from_slice(view.get(from..end).unwrap_or_default());
        (!buf.is_empty()).then(|| Bytes::from(buf))
    } else {
        let slice = view.slice(from..end);
        (!slice.is_empty()).then_some(slice)
    }
}

/// Result of a resumable scan for the next complete SSE frame boundary.
enum FrameScanState {
    /// A complete frame ends at this absolute offset in the scanned buffer (through
    /// the blank-line terminator that dispatches it).
    Complete(usize),
    /// No complete frame yet: resume the next scan at `resume_from`, treating that
    /// offset as an SSE line start iff `at_line_start`. `resume_from` may point at a
    /// lone trailing CR whose terminator is still ambiguous.
    Incomplete {
        /// Absolute offset in the scanned buffer where the next scan must resume.
        resume_from: usize,
        /// Whether `resume_from` lands on an SSE line start.
        at_line_start: bool,
    },
}

/// Classification of a possible line terminator at `buf[i]` (WHATWG event-stream line
/// endings). A lone trailing `\r` at the very end of `buf` is ambiguous mid-stream — it
/// could still become `\r\n` when the next chunk arrives — so it is reported separately
/// and the split deferred. At `end_of_stream` no more bytes can arrive, so that trailing
/// `\r` resolves to a definite one-byte terminator instead.
enum LineTerminator {
    /// `buf[i]` is not a terminator byte.
    None,
    /// A terminator of this byte length ends the line at `buf[i]`.
    End(usize),
    /// A lone trailing `\r` at the end of `buf` mid-stream: ambiguous until more bytes
    /// arrive. Only reachable when `at_stream_end` is false.
    Ambiguous,
}

/// Classify a potential line terminator at `buf[i]`. A lone trailing `\r` at the end of
/// `buf` is `Ambiguous` mid-stream but a definite one-byte terminator once
/// `at_stream_end` is set, since no continuation byte can follow.
fn terminator_at(buf: &[u8], i: usize, at_stream_end: bool) -> LineTerminator {
    match buf.get(i) {
        Some(b'\n') => LineTerminator::End(1),
        Some(b'\r') => {
            if buf.get(i + 1) == Some(&b'\n') {
                LineTerminator::End(2)
            } else if i + 1 == buf.len() && !at_stream_end {
                LineTerminator::Ambiguous
            } else {
                LineTerminator::End(1)
            }
        },
        _ => LineTerminator::None,
    }
}

/// Scan `buf[from..]` for the next blank-line frame terminator, resuming a prior scan.
///
/// The byte at `from` is treated as an SSE line start iff `at_line_start`, which is the
/// only cross-chunk state empty-line detection needs. Bytes before `from` are never
/// re-examined, so a frame split across N chunks is scanned once (linear), not
/// re-scanned from byte zero on every chunk. When `at_stream_end` is set, a lone
/// trailing `\r` resolves to a definite terminator instead of deferring as ambiguous,
/// so a final frame ending in a bare CR is still detected.
fn scan_frame_end(buf: &[u8], from: usize, at_line_start: bool, at_stream_end: bool) -> FrameScanState {
    let mut i = from;
    let mut line_has_content = !at_line_start;
    while i < buf.len() {
        match terminator_at(buf, i, at_stream_end) {
            LineTerminator::None => {
                line_has_content = true;
                i += 1;
            },
            LineTerminator::Ambiguous => {
                return FrameScanState::Incomplete {
                    resume_from: i,
                    at_line_start: !line_has_content,
                };
            },
            LineTerminator::End(len) => {
                if !line_has_content {
                    return FrameScanState::Complete(i + len);
                }
                line_has_content = false;
                i += len;
            },
        }
    }
    FrameScanState::Incomplete {
        resume_from: i,
        at_line_start: !line_has_content,
    }
}

/// Rewrite a lifecycle frame's `data:` payload in place, returning the new frame
/// bytes, or `None` when the frame is not a response-lifecycle frame (so the caller
/// forwards it untouched).
///
/// Only the `data:` line(s) change: the payload JSON gets `previous_response_id`
/// restored (see [`rewrite_lifecycle_frame_data`]) and is re-emitted as a single
/// `data:` line at the position of the frame's first `data:` line. Every other line
/// — `event:`, `id:`, `retry:`, comments (`:`), unknown fields, the blank-line
/// terminator — is copied verbatim with its original terminator, so nothing outside
/// the payload is lost or reordered. A frame's `data:` lines are joined with `\n`
/// (per the SSE spec) to reconstruct the payload before parsing.
fn rewritten_lifecycle_frame(frame: &[u8], prev_id: &str) -> Option<Vec<u8>> {
    let joined = join_sse_data_payload(frame)?;
    let new_json = rewrite_lifecycle_frame_data(&joined, prev_id)?;
    Some(rebuild_frame_with_data(frame, &new_json))
}

/// Join a frame's `data:` line values with `\n` (per the SSE spec) to reconstruct
/// the payload for parsing, or `None` when the frame carries no `data:` line.
fn join_sse_data_payload(frame: &[u8]) -> Option<Vec<u8>> {
    let mut joined: Vec<u8> = Vec::new();
    let mut has_data = false;
    for_each_sse_line(frame, |content, _term| {
        if let Some(value) = sse_field_value(content, b"data") {
            if has_data {
                joined.push(b'\n');
            }
            has_data = true;
            joined.extend_from_slice(value);
        }
    });
    has_data.then_some(joined)
}

/// Rebuild a frame, folding its `data:` run into a single `data: <new_json>` line at
/// the first data position and copying every other line (and its terminator)
/// verbatim.
fn rebuild_frame_with_data(frame: &[u8], new_json: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.len() + new_json.len());
    let mut data_emitted = false;
    for_each_sse_line(frame, |content, term| {
        if sse_field_value(content, b"data").is_some() {
            // Fold the frame's `data:` run into one rewritten line at the first
            // position; drop the rest (their content moved into `new_json`).
            if !data_emitted {
                out.extend_from_slice(b"data: ");
                out.extend_from_slice(new_json);
                out.extend_from_slice(term);
                data_emitted = true;
            }
        } else {
            out.extend_from_slice(content);
            out.extend_from_slice(term);
        }
    });
    out
}

/// Split raw SSE frame bytes into lines, invoking `f(content, terminator)` for each.
///
/// `terminator` is the exact `\n`, `\r`, or `\r\n` that ended the line, so callers
/// can re-emit lines byte-for-byte. A complete frame always ends with a terminator,
/// so no trailing content is left unterminated in practice.
fn for_each_sse_line(frame: &[u8], mut f: impl FnMut(&[u8], &[u8])) {
    let mut i = 0;
    let mut start = 0;
    while let Some(&byte) = frame.get(i) {
        match byte {
            b'\n' => {
                f(frame.get(start..i).unwrap_or_default(), b"\n");
                i += 1;
                start = i;
            },
            b'\r' => {
                let (term, term_len): (&[u8], usize) = if frame.get(i + 1) == Some(&b'\n') {
                    (b"\r\n", 2)
                } else {
                    (b"\r", 1)
                };
                f(frame.get(start..i).unwrap_or_default(), term);
                i += term_len;
                start = i;
            },
            _ => i += 1,
        }
    }
    if start < frame.len() {
        f(frame.get(start..).unwrap_or_default(), b"");
    }
}

/// Return an SSE line's field value when its field name equals `field`, mirroring
/// the field parsing in the shared `SseFrameParser`: a leading `:` marks a comment
/// (never a field), the name is the text before the first `:`, and one optional
/// space after the `:` is skipped. A line with no `:` is a field with an empty
/// value.
fn sse_field_value<'a>(line: &'a [u8], field: &[u8]) -> Option<&'a [u8]> {
    if line.first() == Some(&b':') {
        return None;
    }
    let (name, value) = match line.iter().position(|&b| b == b':') {
        Some(pos) => {
            let value_start = if line.get(pos + 1) == Some(&b' ') {
                pos + 2
            } else {
                pos + 1
            };
            (
                line.get(..pos).unwrap_or_default(),
                line.get(value_start..).unwrap_or_default(),
            )
        },
        None => (line, [].as_slice()),
    };
    (name == field).then_some(value)
}

/// Inject `prev_id` into a response-lifecycle SSE payload.
///
/// Returns `Some(rewritten bytes)` when `data` is a JSON object with a top-level
/// `response` member whose `object == "response"` — i.e. `response.created`,
/// `response.in_progress`, `response.completed`, `response.queued`,
/// `response.failed`, `response.incomplete`, and any future lifecycle event that
/// embeds the full response object. Returns `None` when the payload is a
/// delta/item event, non-JSON, or otherwise not a lifecycle frame, so it passes
/// through byte-for-byte. Detection is by shape, not by an event-type allowlist,
/// so new lifecycle event types are handled without changes here.
///
/// The insert is unconditional (overwrites whatever the backend echoed): under the
/// rehydration gate the backend always echoes `null`, so this restores exactly the
/// ID the proxy stripped from the upstream request. With `serde_json`'s
/// `preserve_order`, the existing `previous_response_id` slot is updated in place,
/// leaving every other field and its ordering untouched.
fn rewrite_lifecycle_frame_data(data: &[u8], prev_id: &str) -> Option<Vec<u8>> {
    let mut parsed: Value = serde_json::from_slice(data).ok()?;
    let response = parsed.as_object_mut()?.get_mut("response")?.as_object_mut()?;
    if response.get("object").and_then(Value::as_str) != Some("response") {
        return None;
    }
    response.insert("previous_response_id".to_owned(), Value::String(prev_id.to_owned()));
    serde_json::to_vec(&parsed).ok()
}

/// Return whether this request targets the body-less Responses cancel endpoint.
fn is_responses_cancel_path(path: &str) -> bool {
    let path = path.trim_end_matches('/');

    let Some(response_id) = path
        .strip_prefix("/v1/responses/")
        .and_then(|rest| rest.strip_suffix("/cancel"))
    else {
        return false;
    };

    !response_id.is_empty() && !response_id.contains('/')
}

// -----------------------------------------------------------------------------
// Stored message extraction
// -----------------------------------------------------------------------------

/// Stored messages from a response record.
fn stored_messages_for_response(mut record: ResponseRecord) -> Vec<Value> {
    match std::mem::take(&mut record.messages) {
        Value::Array(arr) if !arr.is_empty() => arr,
        _ => reconstruct_messages_from_public_response(record),
    }
}

/// Stored messages from a conversation record.
fn stored_messages_for_conversation(record: ConversationRecord) -> Vec<Value> {
    match record.messages {
        Value::Array(arr) => arr,
        _ => vec![],
    }
}

/// Fetch the previous response and validate its status in one step.
async fn fetch_and_validate_previous(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    prev_id: &str,
) -> Result<ResponseRecord, FilterAction> {
    let record = fetch_previous_response(ctx, tenant_id, prev_id).await?;
    validate_response_status(&record)?;
    Ok(record)
}

/// Resolve the conversation ID from the request body, returning a
/// `Release` when no conversation field is present or a `Reject`
/// when the field is malformed.
fn resolve_conversation_id(body: &Value) -> Result<String, FilterAction> {
    let has_field = body.get("conversation").is_some();
    extract_conversation_id(body).ok_or_else(|| {
        if has_field {
            FilterAction::Reject(responses_error_rejection(
                400,
                "invalid_request_error",
                "invalid conversation value: expected a string ID or {\"id\": \"...\"}",
            ))
        } else {
            FilterAction::Release
        }
    })
}

/// Fetch a conversation record from the store.
async fn fetch_conversation(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    conv_id: &str,
) -> Result<ConversationRecord, FilterAction> {
    let registry = ctx.extensions.get::<ResponseStoreRegistry>().ok_or_else(|| {
        warn!("rehydrate: response store registry not available");
        reject_server_error("response store is not available")
    })?;

    let store = registry.get(DEFAULT_STORE_NAME).ok_or_else(|| {
        warn!("rehydrate: default response store not registered");
        reject_server_error("response store is not available")
    })?;

    let record = store.get_conversation(tenant_id, conv_id).await.map_err(|e| {
        warn!(error = %e, "rehydrate: failed to fetch conversation");
        reject_server_error("failed to fetch conversation")
    })?;

    record.ok_or_else(|| {
        debug!(id = %conv_id, "rehydrate: conversation not found");
        reject_invalid(&format!("conversation '{conv_id}' not found"))
    })
}

/// Build [`ResponsesState`] by prepending stored messages before the current input.
fn build_state(
    parsed_body: Value,
    stored: Vec<Value>,
    previous_tools: Vec<Value>,
    previous_usage: Option<Value>,
) -> ResponsesState {
    let replay = replay_messages_from_stored(&stored);
    let mut state = ResponsesState::from_request_body(parsed_body);
    state.history_rehydrated = true;
    state.messages.splice(0..0, replay);
    state.persisted_messages.splice(0..0, stored);
    state.previous_tools = previous_tools;
    state.previous_usage = previous_usage;
    state
}

/// Return stored history, reconstructing from public fields for
/// records created before hidden messages were persisted.
fn reconstruct_messages_from_public_response(mut record: ResponseRecord) -> Vec<Value> {
    let mut messages = Vec::new();

    append_stored_input_items(&mut messages, record.input);

    let output = record
        .response_object
        .get_mut("output")
        .map(std::mem::take)
        .filter(|v| !v.is_null());
    if let Some(output) = output {
        append_stored_output_items(&mut messages, output);
    }

    messages
}

/// Append stored response output items to the persisted conversation history.
fn append_stored_output_items(messages: &mut Vec<Value>, output: Value) {
    match output {
        Value::Array(items) => messages.extend(items),
        other => messages.push(other),
    }
}

/// Return stored items that should be replayed as backend request input.
fn replay_messages_from_stored(stored: &[Value]) -> Vec<Value> {
    stored.iter().filter_map(canonical_openresponses_replay_item).collect()
}

/// Parse the request body and extract `previous_response_id`.
///
/// Returns the parsed body alongside the optional ID so callers
/// can reuse it for [`ResponsesState`] construction.
fn parse_body_and_extract_id(bytes: &[u8]) -> Result<(Value, Option<String>), FilterAction> {
    let parsed: Value = serde_json::from_slice(bytes).map_err(|e| {
        debug!(error = %e, "rehydrate: invalid request JSON");
        reject_invalid(&format!("invalid request body: {e}"))
    })?;

    let id = match parsed.get("previous_response_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(reject_invalid("previous_response_id must be a string")),
    };

    Ok((parsed, id))
}

// -----------------------------------------------------------------------------
// Fetch & Validate
// -----------------------------------------------------------------------------

/// Fetch the previous response record from the store.
async fn fetch_previous_response(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    prev_id: &str,
) -> Result<ResponseRecord, FilterAction> {
    let registry = ctx.extensions.get::<ResponseStoreRegistry>().ok_or_else(|| {
        warn!("rehydrate: response store registry not available");
        reject_server_error("response store is not available")
    })?;

    let store = registry.get(DEFAULT_STORE_NAME).ok_or_else(|| {
        warn!("rehydrate: default response store not registered");
        reject_server_error("response store is not available")
    })?;

    let record = store.get_response(tenant_id, prev_id).await.map_err(|e| {
        warn!(error = %e, "rehydrate: failed to fetch previous response");
        reject_server_error("failed to fetch previous response")
    })?;

    record.ok_or_else(|| {
        debug!(id = %prev_id, "rehydrate: previous response not found");
        reject_invalid(&format!("response '{prev_id}' not found"))
    })
}

/// Validate that the stored response has status `"completed"`.
fn validate_response_status(record: &ResponseRecord) -> Result<(), FilterAction> {
    let status = record
        .response_object
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    if status != "completed" {
        return Err(reject_invalid(&format!(
            "cannot continue from response with status '{status}'"
        )));
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// MCP Tool & Usage Extraction
// -----------------------------------------------------------------------------

/// Recover MCP tool listings from stored history and response output.
fn collect_mcp_tool_listings(record: &ResponseRecord) -> Vec<Value> {
    let mut listings = Vec::new();
    let mut seen = HashSet::new();

    if let Some(messages) = record.messages.as_array() {
        collect_mcp_tool_listings_from_items(messages, &mut seen, &mut listings);
    }

    if let Some(output) = record.response_object.get("output").and_then(Value::as_array) {
        collect_mcp_tool_listings_from_items(output, &mut seen, &mut listings);
    }

    listings
}

/// Append MCP tool listings from a sequence of response items.
fn collect_mcp_tool_listings_from_items(
    items: &[Value],
    seen: &mut HashSet<(String, Vec<String>)>,
    listings: &mut Vec<Value>,
) {
    listings.extend(items.iter().filter_map(|item| {
        if item.get("type").and_then(Value::as_str) != Some("mcp_list_tools") {
            return None;
        }

        let label = item.get("server_label").and_then(Value::as_str)?;
        let tools = item.get("tools").and_then(Value::as_array)?;
        let mut names = mcp_tool_names(tools);
        names.sort();
        names.dedup();

        if !seen.insert((label.to_owned(), names)) {
            return None;
        }

        let mut map = serde_json::Map::new();
        map.insert("server_label".to_owned(), Value::String(label.to_owned()));
        map.insert("tools".to_owned(), Value::Array(tools.clone()));
        if let Some(url) = item.get("server_url").and_then(Value::as_str) {
            map.insert("server_url".to_owned(), Value::String(url.to_owned()));
        }
        Some(Value::Object(map))
    }));
}

/// Extract tool names from MCP tool definitions.
fn mcp_tool_names(tools: &[Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(ToOwned::to_owned))
        .collect()
}

/// Install the rehydrated state, preserving request-phase markers that earlier
/// filters set on the pre-rehydrate state.
///
/// [`build_state`] reconstructs [`ResponsesState`] from the request body, so
/// markers not derivable from the body alone must be carried across the
/// replacement. Currently that is the store filter's
/// [`ResponsesState::store_persist_armed`] flag, which `openai_response_store`
/// sets before rehydrate runs. Dropping it here would make `mcp_dispatch`
/// falsely reject a continuation-turn `mcp_approval_request` as unresumable,
/// even though the store is configured and will persist the response.
///
/// It also seeds [`ResponsesState::response_id`] from the `responses.response_id`
/// metadata (assigned upstream), since the reconstructed state cannot derive it
/// from the request body.
fn install_rehydrated_state(ctx: &mut HttpFilterContext<'_>, mut state: ResponsesState) {
    let store_persist_armed = ctx
        .extensions
        .get::<ResponsesState>()
        .is_some_and(|prev| prev.store_persist_armed);
    state.store_persist_armed = store_persist_armed;
    state.response_id = ctx.get_metadata("responses.response_id").map(ToOwned::to_owned);
    write_previous_usage_metadata(ctx, state.previous_usage.as_ref());
    ctx.extensions.insert(state);
}

/// Extract token usage from the previous response and set
/// metadata keys for downstream auto-compaction.
///
/// Writes `input_tokens`, `output_tokens`, and `total_tokens` as
/// individual string metadata values when present.
fn write_previous_usage_metadata(ctx: &mut HttpFilterContext<'_>, usage: Option<&Value>) {
    let Some(usage) = usage else {
        return;
    };

    if let Some(input) = usage.get("input_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_INPUT_KEY, input.to_string());
    }

    if let Some(output) = usage.get("output_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_OUTPUT_KEY, output.to_string());
    }

    if let Some(total) = usage.get("total_tokens").and_then(Value::as_u64) {
        ctx.set_metadata(PREV_USAGE_TOTAL_KEY, total.to_string());
    }

    trace!("extracted previous response usage");
}

// -----------------------------------------------------------------------------
// Rejection Helpers
// -----------------------------------------------------------------------------

/// Build a 400 rejection with a Responses API error body.
fn reject_invalid(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(400, "invalid_request_error", message))
}

/// Build a 500 rejection with a Responses API error body.
fn reject_server_error(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(500, "server_error", message))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::needless_pass_by_value,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;
