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
//! - **`on_response`**: re-checks skip conditions, then inspects the response status and headers. Non-2xx responses,
//!   content-encoded bodies, or content types other than JSON or event-stream set `responses.skip_persist` and bail
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
    super::{
        DEFAULT_STORE_NAME,
        agentic_loop::{AgenticBudgetPolicy, buffered_parsed_json_bytes_upper_bound},
        bound_body_outcome,
        budget_error::reject_retained_payload_budget,
        error::responses_error_rejection,
        rehydrate::DirectFiniteRestoreFraming,
        state::{PayloadMeter, ResponsesState, retained_json_bytes, retained_json_values_bytes},
    },
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
/// Metadata key for the response-store filter's request-lifetime input snapshot.
const STORE_REQUEST_PAYLOAD_BYTES_METADATA: &str = "responses.store_request_payload_bytes";

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

/// A budgeted buffered conversation response was persisted before headers.
struct BufferedResponsePersistenceAttempted;

/// This exchange wrote its response before a later conversation append hook.
/// A budget failure in that hook may remove only this exchange's durable row.
#[cfg(feature = "openai-conversations")]
pub(crate) struct PersistedResponseForConversation(pub(crate) String);

/// Set only when the store's response hook participated in this exchange.
#[cfg(feature = "openai-conversations")]
pub(crate) struct StoreResponseHeaderRan(ResponseRound);

/// The store was armed on request, but response conditions excluded its hook.
#[cfg(feature = "openai-conversations")]
pub(crate) struct StoreResponseHeaderSkipped {
    /// The response attempt excluded by Store response conditions.
    round: ResponseRound,
    /// A committed parent SSE header keeps this decision through tool rounds.
    outer_stream: bool,
}

/// A response callback can run more than once inside IRR. Match handoff
/// markers to the current round so an intermediate response cannot satisfy
/// the final response's store participation check.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "openai-conversations")]
pub(crate) struct ResponseRound {
    /// Core's IRR round, when a router is active.
    router: Option<u32>,
    /// Responses loop round, including local continuations.
    agentic: Option<u32>,
}

#[cfg(feature = "openai-conversations")]
impl ResponseRound {
    /// The router strips its iteration extension before the committed parent
    /// streaming header runs. Step filters retain it until their response ends.
    pub(crate) fn is_outside_router(self) -> bool {
        self.router.is_none()
    }
}

/// Identify the current response attempt across both loop owners.
#[cfg(feature = "openai-conversations")]
pub(crate) fn response_round(ctx: &HttpFilterContext<'_>) -> ResponseRound {
    ResponseRound {
        router: ctx
            .extensions
            .get::<praxis_filter::IterationState>()
            .map(praxis_filter::IterationState::iteration),
        agentic: ctx.extensions.get::<ResponsesState>().map(|state| state.iteration),
    }
}

/// Return whether the Store response hook was skipped in this response round.
#[cfg(feature = "openai-conversations")]
pub(crate) fn store_response_header_skipped(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<StoreResponseHeaderSkipped>()
        .is_some_and(|marker| marker.outer_stream || marker.round == response_round(ctx))
}

/// Record that Store response conditions excluded this round's response hook.
#[cfg(feature = "openai-conversations")]
pub(crate) fn mark_store_response_header_skipped(ctx: &mut HttpFilterContext<'_>) -> bool {
    let round = response_round(ctx);
    if !request_persistence_armed(ctx)
        || ctx
            .extensions
            .get::<StoreResponseHeaderRan>()
            .is_some_and(|marker| marker.0 == round)
    {
        return false;
    }
    // Only the server-injected, top-level readiness gate calls this helper.
    // IRR may have swapped its iteration extension into the parent by now,
    // so the response-round extension cannot identify that outer placement.
    let outer_stream = ctx
        .response_header
        .as_ref()
        .and_then(|response| response.headers.get(http::header::CONTENT_TYPE))
        .and_then(|content_type| content_type.to_str().ok())
        .is_some_and(is_event_stream_content_type);
    ctx.extensions
        .insert(StoreResponseHeaderSkipped { round, outer_stream });
    true
}

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
    #[expect(
        clippy::too_many_lines,
        reason = "checks aggregate admission before constructing the store record"
    )]
    #[expect(
        clippy::cognitive_complexity,
        reason = "distinguishes absent state from an unmeasurable response before persistence"
    )]
    fn persist_from_streaming_state(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
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

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            trace!("skipping streaming persistence: no Responses state");
            return Ok(FilterAction::Continue);
        };
        let response_bytes = retained_json_bytes(&state.response_object);
        if !response_bytes.is_some_and(|bytes| persistence_construction_fits(ctx, bytes)) {
            return streaming_persistence_budget_failure(ctx, body);
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

        let budgeted_conversation = ctx.get_metadata("openai_responses_format.has_conversation") == Some("true")
            && ctx
                .extensions
                .get::<ResponsesState>()
                .and_then(ResponsesState::retained_payload_limit)
                .is_some();
        if budgeted_conversation {
            let inserted = match persist_response_if_absent_blocking(&persist.store, &record, &pending_approvals) {
                Ok(inserted) => inserted,
                Err(StoreError::PayloadTooLarge) => return streaming_persistence_budget_failure(ctx, body),
                Err(error) => return Err(Box::new(error)),
            };
            if !inserted {
                return streaming_persistence_budget_failure(ctx, body);
            }
        } else {
            persist_response_blocking(&persist.store, &record, &pending_approvals)?;
        }
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
        #[cfg(feature = "openai-conversations")]
        if ctx.get_metadata("openai_responses_format.has_conversation") == Some("true") {
            ctx.extensions.insert(PersistedResponseForConversation(record.id));
        }
        #[cfg(feature = "openai-conversations")]
        {
            crate::openai::conversations::append_after_store_body(ctx, body, false)
        }
        #[cfg(not(feature = "openai-conversations"))]
        {
            Ok(FilterAction::Continue)
        }
    }

    /// Persist a non-streaming response from the buffered body bytes.
    #[expect(
        clippy::too_many_lines,
        reason = "body record projection and conditional insert share one persistence seam"
    )]
    fn persist_from_buffered_body(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
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

        if !buffered_persistence_construction_fits(ctx, bytes) {
            return Ok(persistence_budget_failure(ctx, false, body));
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

        let budgeted_conversation = ctx.get_metadata("openai_responses_format.has_conversation") == Some("true")
            && ctx
                .extensions
                .get::<ResponsesState>()
                .and_then(ResponsesState::retained_payload_limit)
                .is_some();
        if budgeted_conversation {
            let inserted = match persist_response_if_absent_blocking(&store, &record, &pending_approvals) {
                Ok(inserted) => inserted,
                Err(StoreError::PayloadTooLarge) => return Ok(persistence_budget_failure(ctx, false, body)),
                Err(error) => return Err(Box::new(error)),
            };
            if !inserted {
                return Ok(persistence_budget_failure(ctx, false, body));
            }
        } else {
            persist_response_blocking(&store, &record, &pending_approvals)?;
        }
        #[cfg(feature = "openai-conversations")]
        if ctx.get_metadata("openai_responses_format.has_conversation") == Some("true") {
            ctx.extensions.insert(PersistedResponseForConversation(record.id));
        }
        #[cfg(feature = "openai-conversations")]
        {
            crate::openai::conversations::append_after_store_body(ctx, body, true)
        }
        #[cfg(not(feature = "openai-conversations"))]
        {
            Ok(FilterAction::Continue)
        }
    }

    /// Parse the outbound SSE chunk into normalized events and accumulate them in
    /// request-scoped state for the terminal flush.
    ///
    /// Parse-only: it never blocks, mutates, or withholds the client bytes. The
    /// terminal seam ([`Self::persist_from_streaming_state`]) drains this state
    /// and writes the whole log in one `append_events` after the record is
    /// durable. The store's own replay bound or a poisoned decoder abandons
    /// replay; exhausting an active agentic budget emits a terminal error and
    /// prevents successful persistence.
    #[expect(
        clippy::too_many_lines,
        reason = "preflights and accounts for replay capture alongside shared state"
    )]
    fn capture_stream_events(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &Option<Bytes>,
        end_of_stream: bool,
    ) -> bool {
        let Some(chunk) = body.as_ref().filter(|b| !b.is_empty()) else {
            return true;
        };
        let mut state = ctx.extensions.remove::<ResponseStoreRequestState>().unwrap_or_default();
        state.terminal_in_chunk = false;
        state.max_sequence_in_chunk = None;
        state.unclassified_in_chunk = false;
        let previously_retained = state.retained_payload_bytes();
        let terminal_ready = end_of_stream || streaming_terminal_emitted(ctx);
        if let Some(responses) = ctx.extensions.get::<ResponsesState>()
            && let Some(limit) = responses.retained_payload_limit()
            && (terminal_ready
                || state
                    .shared_stable_bytes
                    .as_ref()
                    .is_none_or(|cache| !cache.stable_matches(responses)))
        {
            let Some(stable) = responses.stream_stable_payload_bytes_bounded(limit) else {
                ctx.extensions.insert(state);
                return false;
            };
            let Some(remaining) = limit.checked_sub(stable) else {
                ctx.extensions.insert(state);
                return false;
            };
            let mut output_meter = PayloadMeter::new(remaining);
            if output_meter.json_values(&responses.accumulated_output).is_none() {
                ctx.extensions.insert(state);
                return false;
            }
            let Some(stable) = stable.checked_add(output_meter.used()) else {
                ctx.extensions.insert(state);
                return false;
            };
            // Completed output stays unchanged across ordinary text deltas. Measure
            // its response and template owners once per revision. Canonical
            // tool-call assignments retain only IDs, charged by ResponsesState.
            let Some(remaining) = limit.checked_sub(stable) else {
                ctx.extensions.insert(state);
                return false;
            };
            let mut current_output_meter = PayloadMeter::new(remaining);
            if current_output_meter.json(&responses.response_object).is_none()
                || current_output_meter
                    .json(&responses.local_completion_response_template)
                    .is_none()
            {
                ctx.extensions.insert(state);
                return false;
            }
            let Some(cache) = StoreStableCache::new(responses, stable, current_output_meter.used()) else {
                ctx.extensions.insert(state);
                return false;
            };
            state.shared_stable_bytes = Some(cache);
        } else if let Some(responses) = ctx.extensions.get::<ResponsesState>()
            && let Some(limit) = responses.retained_payload_limit()
            && let Some(cache) = state.shared_stable_bytes.as_mut()
            && !cache.current_matches(responses)
            && cache.refresh_current(responses, limit).is_none()
        {
            ctx.extensions.insert(state);
            return false;
        }
        // The decoder can simultaneously hold an unfinished record, completed
        // records, and a newly captured replay row. Reserve their peak before
        // feeding the chunk or making another owned payload copy.
        let admitted = state.events_over_budget
            || state
                .pending_wire_bytes
                .checked_add(chunk.len())
                .and_then(|bytes| bytes.checked_mul(4))
                .and_then(|peak| {
                    if state.charged_retained_bytes.is_some() {
                        Some(peak)
                    } else {
                        previously_retained?.checked_add(peak)
                    }
                })
                .is_some_and(|peak| {
                    ctx.extensions
                        .get::<ResponsesState>()
                        .is_none_or(|responses| store_stream_budget_fits(responses, state.shared_stable_bytes, 0, peak))
                });
        if !admitted {
            ctx.extensions.insert(state);
            return false;
        }
        let track_direct_metadata = ctx.extensions.get::<praxis_filter::IterationState>().is_none()
            && ctx
                .extensions
                .get::<ResponsesState>()
                .and_then(ResponsesState::retained_payload_limit)
                .is_some();
        state.error_event_detector.push(chunk);
        if state.events_over_budget {
            if track_direct_metadata {
                Self::observe_abandoned_replay(ctx, &mut state, chunk);
            }
        } else {
            self.accumulate_events(&mut state, chunk, track_direct_metadata);
        }
        state.track_pending_wire_bytes(chunk);
        let retained = state.retained_payload_bytes();
        let admitted = previously_retained.zip(retained).is_some_and(|(_, next)| {
            let budget_fits = ctx.extensions.get::<ResponsesState>().is_none_or(|responses| {
                store_stream_budget_fits(
                    responses,
                    state.shared_stable_bytes,
                    state.charged_retained_bytes.unwrap_or(0),
                    next,
                )
            });
            budget_fits && charge_store_stream_payload(ctx, &mut state, next)
        });
        ctx.extensions.insert(state);
        admitted
    }

    /// Decode `chunk` and append each contained SSE event, stopping and marking
    /// the log non-replayable on a bound overrun or decoder poison.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps replay capture and the decoder's failure cleanup together"
    )]
    fn accumulate_events(&self, state: &mut ResponseStoreRequestState, chunk: &Bytes, track_direct_metadata: bool) {
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
            if track_direct_metadata {
                observe_forwarded_record(state, record);
            } else {
                state.terminal_in_chunk |= is_forwarded_terminal_record(record, &record.data());
            }
        }
        for record in &batch.records {
            if !self.capture_one(state, record) {
                // The replay log is abandoned, but the separate header scanner
                // still checks later chunks for a client-visible error.
                if !track_direct_metadata {
                    state.event_decoder = None;
                }
                return;
            }
        }
        if batch.error.is_some() {
            // A poisoned decoder cannot reliably continue; abandon the log so a
            // truncated stream is never served as a complete replay.
            state.events_over_budget = true;
            state.events.clear();
            state.event_bytes = 0;
            state.event_name_bytes = 0;
            state.event_decoder = None;
            state.unclassified_in_chunk = true;
        }
    }

    /// Keep observing provider wire metadata after the replay log has reached
    /// its independent cap. The existing decoder is reused only when its
    /// transient scratch fits the aggregate budget; otherwise the released
    /// frame is marked unclassified and a later failure must abort the wire.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps the decoder peak check and metadata observation in one admission boundary"
    )]
    fn observe_abandoned_replay(ctx: &HttpFilterContext<'_>, state: &mut ResponseStoreRequestState, chunk: &Bytes) {
        let peak = state
            .pending_wire_bytes
            .checked_add(chunk.len())
            .and_then(|bytes| bytes.checked_mul(4))
            .and_then(|bytes| state.retained_payload_bytes()?.checked_add(bytes));
        let fits = state.event_decoder.is_some()
            && peak.is_some_and(|peak| {
                ctx.extensions.get::<ResponsesState>().is_none_or(|responses| {
                    store_stream_budget_fits(
                        responses,
                        state.shared_stable_bytes,
                        state.charged_retained_bytes.unwrap_or(0),
                        peak,
                    )
                })
            });
        if !fits {
            state.event_decoder = None;
            state.pending_wire_bytes = 0;
            state.unclassified_in_chunk = true;
            return;
        }
        let Some(decoder) = state.event_decoder.as_mut() else {
            return;
        };
        let batch = decoder.push(chunk);
        for record in &batch.records {
            observe_forwarded_record(state, record);
        }
        if batch.error.is_some() {
            state.event_decoder = None;
            state.pending_wire_bytes = 0;
            state.unclassified_in_chunk = true;
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
            state.event_name_bytes = 0;
            return false;
        }
        state.event_bytes = next_bytes;
        state.event_name_bytes = state.event_name_bytes.saturating_add(event.event_type.len());
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

/// Classify a completed SSE frame from the wire, even when it cannot be
/// indexed in the replay log (for example, a direct provider omitted a
/// sequence number). `SseDecoder` emits only frames with a `data:` field, so
/// a data-less `event:` line cannot falsely mark the stream complete.
fn is_forwarded_terminal_record(record: &SseRecord, data: &[u8]) -> bool {
    if !record.is_event() || data == b"[DONE]" {
        return false;
    }
    if let Some(event) = record.event().and_then(|value| std::str::from_utf8(value).ok()) {
        return is_terminal_event_type(event);
    }
    serde_json::from_slice::<EventType<'_>>(data)
        .ok()
        .and_then(|event| event.ty)
        .is_some_and(|event| is_terminal_event_type(&event))
}

/// Inspect provider wire metadata without retaining another event payload.
fn observe_forwarded_record(state: &mut ResponseStoreRequestState, record: &SseRecord) {
    if !record.is_event() {
        return;
    }
    let data = record.data();
    if data.as_ref() == b"[DONE]" {
        return;
    }
    state.terminal_in_chunk |= is_forwarded_terminal_record(record, &data);
    match serde_json::from_slice::<WireSequenceHead>(&data) {
        Ok(WireSequenceHead {
            sequence_number: Some(sequence),
        }) => {
            state.max_sequence_in_chunk = Some(
                state
                    .max_sequence_in_chunk
                    .map_or(sequence, |previous| previous.max(sequence)),
            );
        },
        Ok(_) => {},
        Err(_) => state.unclassified_in_chunk = true,
    }
}

/// Minimal borrowing parse of the public provider sequence.
#[derive(serde::Deserialize)]
struct WireSequenceHead {
    /// Sequence to follow if Store later emits a local budget error.
    sequence_number: Option<u64>,
}

/// Borrow the `type` field for a data-only SSE terminal without materializing
/// the provider's full response JSON.
#[derive(serde::Deserialize)]
struct EventType<'a> {
    /// Borrowed logical event name, when a data-only frame carries one.
    #[serde(borrow, default, rename = "type")]
    ty: Option<Cow<'a, str>>,
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
    ctx.set_metadata(
        STORE_REQUEST_PAYLOAD_BYTES_METADATA,
        capture.retained_payload_bytes().unwrap_or(usize::MAX).to_string(),
    );
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

/// Store request validation can complete before shared response state exists.
#[derive(Default)]
enum PersistenceArm {
    /// The store has not armed this request for persistence.
    #[default]
    Pending,
    /// The store has armed persistence for this request.
    Armed,
}

/// Cached request, history, prior output, and completed output charges.
#[derive(Clone, Copy)]
struct StoreStableCache {
    /// Logical loop round when the charge was measured.
    iteration: u32,
    /// In-place mutation revision of the cached owners.
    revision: u64,
    /// In-place mutation revision of the current response owners.
    current_output_revision: u64,
    /// Lengths of every cached collection. Dispatch may append after synthesis
    /// has already advanced the iteration.
    collection_lengths: [usize; 7],
    /// Serialized payload charge of the cached request, history, and output.
    bytes: usize,
    /// Serialized response, local completion template, and tool-call owners.
    current_output_bytes: usize,
}

impl StoreStableCache {
    /// Capture the measured charge and the current O(1) collection shape.
    fn new(state: &ResponsesState, bytes: usize, current_output_bytes: usize) -> Option<Self> {
        Some(Self {
            iteration: state.iteration,
            revision: state.replay_stable_payload_revision?,
            current_output_revision: state.current_output_revision?,
            collection_lengths: Self::collection_lengths(state),
            bytes,
            current_output_bytes,
        })
    }

    /// Whether the cached charge still describes the current round and shape.
    fn stable_matches(&self, state: &ResponsesState) -> bool {
        self.iteration == state.iteration
            && Some(self.revision) == state.replay_stable_payload_revision
            && self.collection_lengths == Self::collection_lengths(state)
    }

    /// Whether the independently changing current-output owners are fresh.
    fn current_matches(&self, state: &ResponsesState) -> bool {
        Some(self.current_output_revision) == state.current_output_revision
    }

    /// Refresh current-output charge without serializing the unchanged prompt and history.
    fn refresh_current(&mut self, state: &ResponsesState, limit: usize) -> Option<()> {
        let revision = state.current_output_revision?;
        let mut meter = PayloadMeter::new(limit.checked_sub(self.bytes)?);
        meter.json(&state.response_object)?;
        meter.json(&state.local_completion_response_template)?;
        // Canonical tool-call assignments are charged by ResponsesState's
        // inner meter; this cache owns only the two changing response trees.
        self.current_output_bytes = meter.used();
        self.current_output_revision = revision;
        Some(())
    }

    /// Whether both stable and current-output charges can be used for admission.
    fn matches(&self, state: &ResponsesState) -> bool {
        self.stable_matches(state) && self.current_matches(state)
    }

    /// Lengths of the collections covered by the stable charge.
    fn collection_lengths(state: &ResponsesState) -> [usize; 7] {
        [
            state.input.len(),
            state.messages.len(),
            state.persisted_messages.len(),
            state.previous_tools.len(),
            state.tools.len(),
            state.provider_compaction_ids.len(),
            state.accumulated_output.len(),
        ]
    }
}

/// Last logical line ending seen while tracking unfinished SSE wire bytes.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum WireLineEnding {
    /// The last byte belonged to a field line.
    #[default]
    None,
    /// The last byte ended a line with CR; a following LF is paired with it.
    Cr,
    /// The last byte ended a line with LF.
    Lf,
}

/// Request-phase data needed when persisting the response.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "stream replay status and per-chunk wire classification are independent"
)]
struct ResponseStoreRequestState {
    /// Original `input` value from the Responses API create request.
    input: Option<Value>,
    /// Compact size of `input`, computed once when the snapshot is captured.
    /// Re-serializing a large prompt for every SSE chunk would run on the
    /// synchronous downstream response path.
    input_retained_bytes: usize,
    /// Store bytes already charged to `ResponsesState`. The store can run
    /// before another filter creates that state.
    charged_retained_bytes: Option<usize>,
    /// Request, history, and prior output normally stay stable while a
    /// streamed round is delivered. Recheck their O(1) shape on each chunk:
    /// synthesis can advance the iteration before request-side dispatch adds
    /// history in that same iteration. Request-phase re-entry invalidates the
    /// cache for same-length rewrites of the request body and history.
    shared_stable_bytes: Option<StoreStableCache>,
    /// Store request-body validation completed before a later filter created
    /// `ResponsesState`; transfer its approval-persistence signal with the charge.
    persistence_arm: PersistenceArm,
    /// Owner captured before inference begins.
    owner: Option<StateOwner>,
    /// Incremental SSE decoder over the outbound stream, lazily created on the
    /// first response chunk ([`SseDecoder`] has no `Default`).
    event_decoder: Option<SseDecoder>,
    /// Events captured so far, flushed together at the terminal seam.
    events: Vec<CapturedEvent>,
    /// Running total of captured payload bytes, for the byte bound.
    event_bytes: u64,
    /// Raw bytes in independently owned captured event names.
    event_name_bytes: usize,
    /// Upper bound on raw bytes after the last complete SSE record.
    /// The retained meter charges this twice because the decoder may keep a
    /// field value and the current line independently.
    pending_wire_bytes: usize,
    /// Last wire line ending, including across chunk edges. A following LF
    /// after CR belongs to the same line ending.
    last_wire_line_ending: WireLineEnding,
    /// Sticky flag: capture exceeded a bound or the decoder was poisoned, so the
    /// partial log is abandoned and the response becomes non-replayable.
    events_over_budget: bool,
    /// A client-visible SSE error superseded any completed upstream snapshot.
    error_event_detector: SseErrorEventDetector,
    /// A logical terminal decoded from the current admitted wire chunk.
    terminal_in_chunk: bool,
    /// Largest provider sequence decoded in the current admitted chunk.
    max_sequence_in_chunk: Option<u64>,
    /// A released frame could not be classified safely after replay stopped.
    unclassified_in_chunk: bool,
}

/// A logical SSE terminal from a chunk that this filter already released in
/// this loop round. Kept outside capture state because persistence may consume
/// that state; a later IRR round must not inherit the prior round's marker.
struct StoreForwardedTerminal(u32);

/// A released frame could not be classified after replay capture stopped.
/// A later budget error must abort instead of guessing its terminal/sequence.
struct StoreForwardedUnclassified(u32);

impl ResponseStoreRequestState {
    /// Count the sibling-owned input, replay rows, and unfinished decoder data.
    fn retained_payload_bytes(&self) -> Option<usize> {
        self.input_retained_bytes
            .checked_add(usize::try_from(self.event_bytes).ok()?)?
            .checked_add(self.event_name_bytes)?
            .checked_add(self.pending_wire_bytes.checked_mul(2)?)
    }

    /// Track a safe upper bound for the decoder's unfinished record.
    fn track_pending_wire_bytes(&mut self, chunk: &[u8]) {
        if self.events_over_budget && self.event_decoder.is_none() {
            self.pending_wire_bytes = 0;
            return;
        }
        for &byte in chunk {
            match byte {
                b'\n' if self.last_wire_line_ending == WireLineEnding::Cr => {
                    // CRLF is one line ending. Its LF belongs to the still-open
                    // record only if the preceding CR did not finish that record.
                    if self.pending_wire_bytes > 0 {
                        self.pending_wire_bytes = self.pending_wire_bytes.saturating_add(1);
                    }
                },
                b'\r' | b'\n' if self.last_wire_line_ending == WireLineEnding::None => {
                    self.pending_wire_bytes = self.pending_wire_bytes.saturating_add(1);
                },
                b'\r' | b'\n' => self.pending_wire_bytes = 0,
                _ => {
                    self.pending_wire_bytes = self.pending_wire_bytes.saturating_add(1);
                },
            }
            self.last_wire_line_ending = match byte {
                b'\r' => WireLineEnding::Cr,
                b'\n' => WireLineEnding::Lf,
                _ => WireLineEnding::None,
            };
        }
    }
}

#[cfg(test)]
mod pending_wire_tests {
    use bytes::Bytes;
    use praxis_filter::sse::SseDecoder;

    use super::ResponseStoreRequestState;

    #[test]
    fn completed_crlf_sse_records_release_pending_wire_charge_across_chunks() {
        for frame in [
            b"event: response.output_text.delta\ndata: {}\n\n".as_slice(),
            b"event: response.output_text.delta\r\ndata: {}\r\n\r\n",
            b"event: response.output_text.delta\rdata: {}\r\r",
            b"event: response.output_text.delta\ndata: {}\r\n\n",
        ] {
            for split in 0..=frame.len() {
                let mut capture = ResponseStoreRequestState::default();
                let mut decoder = SseDecoder::new();
                let mut completed = 0;
                for chunk in [
                    frame.get(..split).unwrap_or_default(),
                    frame.get(split..).unwrap_or_default(),
                ] {
                    capture.track_pending_wire_bytes(chunk);
                    let batch = decoder.push(&Bytes::copy_from_slice(chunk));
                    assert!(batch.error.is_none());
                    completed += batch.records.len();
                }
                assert_eq!(completed, 1, "frame={frame:?}, split={split}");
                assert_eq!(capture.pending_wire_bytes, 0, "frame={frame:?}, split={split}");
            }
        }
    }

    #[test]
    fn single_crlf_line_ending_stays_charged_until_blank_line() {
        let mut capture = ResponseStoreRequestState::default();
        capture.track_pending_wire_bytes(b"data: {}\r");
        let before_lf = capture.pending_wire_bytes;
        capture.track_pending_wire_bytes(b"\n");
        assert_eq!(capture.pending_wire_bytes, before_lf + 1);
        capture.track_pending_wire_bytes(b"\r");
        assert_eq!(capture.pending_wire_bytes, 0);
        capture.track_pending_wire_bytes(b"\n");
        assert_eq!(capture.pending_wire_bytes, 0);
    }
}

/// Count the store-owned request snapshot, captured replay rows, and decoder
/// data charged to the request-wide agentic budget.
pub(crate) fn retained_request_payload_bytes(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    if let Some(state) = ctx.extensions.get::<ResponseStoreRequestState>() {
        return state.retained_payload_bytes();
    }
    ctx.get_metadata(STORE_REQUEST_PAYLOAD_BYTES_METADATA)
        .map_or(Some(0), |bytes| bytes.parse().ok())
}

/// Charge a store-owned snapshot when shared response state is present. A
/// resolver can create that state after the store captured its input, so the
/// first update charges the whole snapshot instead of replacing a missing one.
fn charge_store_payload(ctx: &mut HttpFilterContext<'_>, store: &mut ResponseStoreRequestState, next: usize) -> bool {
    let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() else {
        store.charged_retained_bytes = None;
        return true;
    };
    let admitted = if let Some(charged) = store.charged_retained_bytes {
        responses.can_replace_retained_payload(charged, next, 0)
            && responses.replace_retained_external_payload_bytes(charged, next)
    } else {
        responses.can_retain_payload(next) && responses.retain_external_payload_bytes(next)
    };
    if admitted {
        store.charged_retained_bytes = Some(next);
    }
    admitted
}

/// Check a replay transition using the per-round request/history charge. The
/// changing meter still includes the store's currently published charge, so a
/// replacement subtracts only that owner's prior bytes.
fn store_stream_budget_fits(
    responses: &ResponsesState,
    stable: Option<StoreStableCache>,
    removed: usize,
    added: usize,
) -> bool {
    let Some(limit) = responses.retained_payload_limit() else {
        return true;
    };
    let Some(cache) = stable.filter(|cache| cache.matches(responses)) else {
        return false;
    };
    let Some(remaining) = limit.checked_sub(cache.bytes) else {
        return false;
    };
    let Some(measurement_limit) = remaining.checked_add(removed) else {
        return false;
    };
    responses
        .store_stream_changing_payload_bytes_bounded_with_current_output(measurement_limit, cache.current_output_bytes)
        .and_then(|current| current.checked_sub(removed))
        .and_then(|current| current.checked_add(added))
        .is_some_and(|next| next <= remaining)
}

/// The caller has already admitted the replay transition with the cached
/// meter; only transfer its independently owned bytes into shared accounting.
fn charge_store_stream_payload(
    ctx: &mut HttpFilterContext<'_>,
    store: &mut ResponseStoreRequestState,
    next: usize,
) -> bool {
    let Some(responses) = ctx.extensions.get_mut::<ResponsesState>() else {
        store.charged_retained_bytes = None;
        return true;
    };
    let admitted = if let Some(charged) = store.charged_retained_bytes {
        responses.replace_retained_external_payload_bytes(charged, next)
    } else {
        responses.retain_external_payload_bytes(next)
    };
    if admitted {
        store.charged_retained_bytes = Some(next);
    }
    admitted
}

/// Record a store charge seeded while admitting the agentic budget or
/// installing rehydrated state.
pub(crate) fn mark_retained_request_payload_charged(ctx: &mut HttpFilterContext<'_>) {
    if let Some(store) = ctx.extensions.get_mut::<ResponseStoreRequestState>() {
        store.charged_retained_bytes = store.retained_payload_bytes();
    }
}

/// Whether an earlier store filter completed request-phase persistence arming.
pub(crate) fn request_persistence_armed(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponseStoreRequestState>()
        .is_some_and(|state| matches!(state.persistence_arm, PersistenceArm::Armed))
}

/// Release all store-owned request payload after a terminal budget failure.
pub(crate) fn discard_retained_request_payload(ctx: &mut HttpFilterContext<'_>) {
    let bytes = retained_request_payload_bytes(ctx).unwrap_or(usize::MAX);
    let charged = match ctx.extensions.remove::<ResponseStoreRequestState>() {
        Some(store) => store.charged_retained_bytes,
        None => Some(bytes),
    };
    ctx.set_metadata(STORE_REQUEST_PAYLOAD_BYTES_METADATA, "0");
    if let (Some(bytes), Some(state)) = (charged, ctx.extensions.get_mut::<ResponsesState>()) {
        state.release_external_payload_bytes(bytes);
    }
}

/// Count the independent response-ID strings built for replay rows at flush.
fn replay_row_id_copy_bytes(ctx: &HttpFilterContext<'_>, state: &ResponsesState) -> Option<usize> {
    let Some(capture) = ctx.extensions.get::<ResponseStoreRequestState>() else {
        return Some(0);
    };
    if capture.events_over_budget {
        return Some(0);
    }
    let id_bytes = state
        .response_object
        .get("id")
        .and_then(Value::as_str)
        .map_or(0, str::len);
    id_bytes.checked_mul(capture.events.len())
}

/// Reserve the zstd replay peak: one worker-owned copy of every payload plus
/// compressed frames alongside the still-live captured rows. Zstd's
/// `compressBound(n)` is at most `n + n / 256 + 64` per event. The store codec
/// is hidden behind the backend trait, so this also conservatively applies to
/// stores that borrow uncompressed replay payloads.
fn replay_payload_staging_bytes(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    let Some(capture) = ctx.extensions.get::<ResponseStoreRequestState>() else {
        return Some(0);
    };
    if capture.events_over_budget {
        return Some(0);
    }
    let payload_bytes = usize::try_from(capture.event_bytes).ok()?;
    payload_bytes
        .checked_mul(2)?
        .checked_add(payload_bytes >> 8)?
        .checked_add(capture.events.len().checked_mul(64)?)
}

/// Bound a buffered response's normalized JSON size before parsing its store record.
fn buffered_persistence_construction_fits(ctx: &HttpFilterContext<'_>, bytes: &[u8]) -> bool {
    let projected = if ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(ResponsesState::retained_payload_limit)
        .is_some()
    {
        // serde_json can expand exponent-form numbers while constructing the
        // record; use the same no-allocation upper bound as agentic admission.
        buffered_parsed_json_bytes_upper_bound(bytes)
    } else {
        Some(bytes.len())
    };
    projected.is_some_and(|response_bytes| persistence_construction_fits_with_wire(ctx, response_bytes, bytes.len()))
}

/// Admit a noncanonical JSON body before core commits the upstream headers.
/// Without the body we cannot inspect number tokens. Twelve times a trusted
/// wire length bounds the scanner's normalized JSON projection. A prior
/// Rehydrate filter can remove `Content-Length` after validating it; its typed
/// framing handoff also reserves the escaped replacement ID and any new key.
/// Missing or ambiguous framing cannot establish the persistence peak before
/// commitment.
fn buffered_header_persistence_length(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    let (wire_bytes, replacement_id_bytes) = trusted_persistence_framing(ctx)?;
    if wire_bytes > MAX_JSON_BODY_BYTES {
        return None;
    }
    let mut parsed_bytes = wire_bytes.checked_mul(12)?;
    if let Some(previous_id_bytes) = replacement_id_bytes {
        // JSON control-byte escapes can use six bytes per input byte. Allow
        // for adding the key when the upstream object omitted it entirely.
        parsed_bytes = parsed_bytes
            .checked_add(previous_id_bytes.checked_mul(6)?)?
            .checked_add(32)?;
    }
    // Core's multi-chunk freeze can hold two original wire copies. After
    // Rehydrate rewrites the body, Store instead holds the replacement wire,
    // which can be longer than both originals when the ID requires escaping.
    let transient_wire_bytes = wire_bytes.checked_mul(2)?;
    let transient_wire_bytes = if replacement_id_bytes.is_some() {
        transient_wire_bytes.max(parsed_bytes)
    } else {
        transient_wire_bytes
    };
    persistence_construction_fits_with_wire(ctx, parsed_bytes, transient_wire_bytes).then_some(wire_bytes)
}

/// Recover trusted upstream framing from the header or Rehydrate's bounded handoff.
fn trusted_persistence_framing(ctx: &HttpFilterContext<'_>) -> Option<(usize, Option<usize>)> {
    let response = ctx.response_header.as_ref()?;
    if response.headers.contains_key(http::header::TRANSFER_ENCODING)
        || response.headers.contains_key(http::header::CONTENT_ENCODING)
    {
        return None;
    }
    match &ctx.response_body_mode {
        BodyMode::Stream => {
            let mut lengths = response.headers.get_all(http::header::CONTENT_LENGTH).iter();
            let wire_bytes = lengths.next()?.to_str().ok()?.parse::<usize>().ok()?;
            if lengths.next().is_some() {
                return None;
            }
            Some((wire_bytes, None))
        },
        BodyMode::StreamBuffer {
            max_bytes: Some(max_bytes),
        } => {
            let framing = ctx.extensions.get::<DirectFiniteRestoreFraming>()?;
            if response.headers.contains_key(http::header::CONTENT_LENGTH) || *max_bytes != framing.wire_bytes {
                return None;
            }
            Some((framing.wire_bytes, Some(framing.previous_id_bytes)))
        },
        _ => None,
    }
}

/// The response, messages, and input columns may each be zstd-compressed and
/// then copied once more into `PostgreSQL`'s query arguments. Reserve twice the
/// maximum frame expansion while the original JSON columns remain live.
pub(super) fn encoded_column_headroom(
    response_bytes: usize,
    history_bytes: usize,
    input_bytes: usize,
) -> Option<usize> {
    response_bytes
        .checked_mul(2)?
        .checked_add(history_bytes)?
        .checked_add(input_bytes)?
        .checked_shr(8)?
        .checked_add(3 * 64)?
        .checked_mul(2)
}

/// Reserve record, history, replay-row IDs, backend serialization, and the
/// `PostgreSQL` input parameter buffer before building them.
pub(super) fn persistence_construction_fits(ctx: &HttpFilterContext<'_>, response_bytes: usize) -> bool {
    persistence_construction_fits_with_wire(ctx, response_bytes, 0)
}

/// The completed buffered wire body is a separate owner from the canonical
/// response tree while the record is written in the response-header hook.
#[expect(clippy::too_many_lines, reason = "persistence owners share one aggregate admission")]
fn persistence_construction_fits_with_wire(
    ctx: &HttpFilterContext<'_>,
    response_bytes: usize,
    wire_bytes: usize,
) -> bool {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return true;
    };
    if state.retained_payload_limit().is_none() {
        return true;
    }
    let Some(history_bytes) = retained_json_values_bytes(&state.persisted_messages) else {
        return false;
    };
    let Some(store_snapshot_bytes) = retained_request_payload_bytes(ctx) else {
        return false;
    };
    let input_bind_bytes = ctx
        .extensions
        .get::<ResponseStoreRequestState>()
        .map_or(0, |capture| capture.input_retained_bytes);
    let approval_bytes = state.pending_approvals.iter().try_fold(0_usize, |used, record| {
        used.checked_add(record.approval_id.len())?
            .checked_add(record.server_label.len())?
            .checked_add(record.tool_name.len())?
            .checked_add(record.arguments.len())?
            .checked_add(record.target_fingerprint.len())
    });
    let replay_id_bytes = replay_row_id_copy_bytes(ctx, state);
    let replay_payload_bytes = replay_payload_staging_bytes(ctx);
    let compression_headroom = encoded_column_headroom(response_bytes, history_bytes, input_bind_bytes);
    // Record construction owns the parsed response and a separate clone of
    // its output in messages. Serialization and PostgreSQL binds then own one
    // copy of each column, for six response-sized owners at the peak. Prior
    // history is charged separately and cannot cover newly appended output.
    let additional = response_bytes
        .checked_mul(6)
        .and_then(|bytes| history_bytes.checked_mul(3)?.checked_add(bytes))
        // The whole store snapshot is already charged in state. Serialization
        // owns another copy, and PostgreSQL's query bind copies only the input
        // from that snapshot, not its captured replay rows.
        .and_then(|bytes| bytes.checked_add(store_snapshot_bytes))
        .and_then(|bytes| bytes.checked_add(input_bind_bytes))
        .and_then(|bytes| approval_bytes?.checked_mul(2)?.checked_add(bytes))
        .and_then(|bytes| bytes.checked_add(replay_id_bytes?))
        .and_then(|bytes| bytes.checked_add(replay_payload_bytes?))
        .and_then(|bytes| bytes.checked_add(compression_headroom?))
        .and_then(|bytes| bytes.checked_add(wire_bytes));
    additional.is_some_and(|bytes| state.can_retain_payload(bytes))
}

/// Reject persistence and emit the appropriate buffered or committed-stream error.
fn persistence_budget_failure(
    ctx: &mut HttpFilterContext<'_>,
    streaming: bool,
    body: &mut Option<Bytes>,
) -> FilterAction {
    ctx.set_metadata("responses.skip_persist", "true");
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
    }
    discard_retained_request_payload(ctx);
    if streaming {
        ctx.set_metadata("responses.store_stream_budget_failed", "true");
        crate::openai::responses::fs_end_stream_with_error_ctx(
            ctx,
            "server_error",
            "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during persistence",
        );
        *body = super::super::stream_events::encode_local_error(
            ctx,
            "server_error",
            "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during persistence",
        );
        return FilterAction::Continue;
    }
    FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during persistence",
    ))
}

/// An error frame is valid only while no logical terminal has reached the
/// client. Once Store released one, a later budget failure must stop the
/// transport rather than append a contradictory second terminal.
fn streaming_persistence_budget_failure(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
) -> Result<FilterAction, FilterError> {
    let terminal_forwarded_this_round = ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
        ctx.extensions
            .get::<StoreForwardedTerminal>()
            .is_some_and(|marker| state.iteration == marker.0)
            || ctx
                .extensions
                .get::<StoreForwardedUnclassified>()
                .is_some_and(|marker| state.iteration == marker.0)
    });
    if !terminal_forwarded_this_round {
        return Ok(persistence_budget_failure(ctx, true, body));
    }
    ctx.set_metadata("responses.skip_persist", "true");
    ctx.set_metadata("responses.store_stream_budget_failed", "true");
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
    }
    discard_retained_request_payload(ctx);
    crate::openai::responses::fs_arm_stream_stop(ctx);
    *body = None;
    Err("response store retained budget exceeded after a forwarded SSE terminal".into())
}

/// Publish wire metadata only after Store has admitted and released the chunk.
/// A terminal or sequence in a rejected chunk must not affect local error
/// framing, and the round marker must not survive into a later IRR iteration.
fn mark_released_stream_chunk(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    terminal: bool,
    max_sequence: Option<u64>,
    unclassified: bool,
) {
    if body.is_none() {
        return;
    }
    let Some(iteration) = ctx.extensions.get::<ResponsesState>().map(|state| state.iteration) else {
        return;
    };
    if terminal {
        ctx.extensions.insert(StoreForwardedTerminal(iteration));
    }
    if unclassified {
        ctx.extensions.insert(StoreForwardedUnclassified(iteration));
    }
    if let Some(sequence) = max_sequence {
        if let Some(next) = sequence.checked_add(1) {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.logical_stream_sequence = state.logical_stream_sequence.max(next);
            }
        } else {
            ctx.extensions.insert(StoreForwardedUnclassified(iteration));
        }
    }
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
fn capture_request_input(ctx: &mut HttpFilterContext<'_>, input: Value) -> Result<(), FilterAction> {
    let Some(retained_bytes) = retained_json_bytes(&input) else {
        return Err(reject_retained_payload_budget(
            ctx,
            "request and rehydrated state exceed openai_agentic_loop.max_retained_bytes",
        ));
    };
    let mut state = ctx.extensions.remove::<ResponseStoreRequestState>().unwrap_or_default();
    state.input = Some(input);
    state.input_retained_bytes = retained_bytes;
    state.shared_stable_bytes = None;
    let admitted = state
        .retained_payload_bytes()
        .is_some_and(|next| charge_store_payload(ctx, &mut state, next));
    ctx.extensions.insert(state);
    if !admitted {
        return Err(reject_retained_payload_budget(
            ctx,
            "request and rehydrated state exceed openai_agentic_loop.max_retained_bytes",
        ));
    }
    ctx.set_metadata(STORE_REQUEST_PAYLOAD_BYTES_METADATA, retained_bytes.to_string());
    Ok(())
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
/// It is written from `on_request_body` because the state creator also runs
/// in that phase. If the store runs first, its local marker is transferred
/// when `ResponsesState` is created later in the filter chain.
fn arm_persistence_if_persisting(ctx: &mut HttpFilterContext<'_>) {
    if request_will_persist_response(ctx) {
        if let Some(store) = ctx.extensions.get_mut::<ResponseStoreRequestState>() {
            store.persistence_arm = PersistenceArm::Armed;
        }
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.store_persist_armed = true;
        }
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

    // An encoded representation is opaque here. Rehydrate also declines it,
    // and Store cannot safely parse or preflight its decoded JSON size.
    if resp.headers.contains_key(http::header::CONTENT_ENCODING) {
        trace!("skipping persistence for content-encoded response");
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

/// Bridge the atomic no-overwrite write to the synchronous body callback.
fn persist_response_if_absent_blocking(
    store: &OwnerScopedResponseStore,
    record: &ResponseRecord,
    pending_approvals: &[PendingApprovalRecord],
) -> Result<bool, StoreError> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::block_in_place(|| {
        handle.block_on(store.persist_response_with_pending_approvals_if_absent(record, pending_approvals))
    })
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

/// Reserve the original input snapshot before parsing another owned JSON tree.
#[expect(
    clippy::too_many_lines,
    reason = "both Store placements must reserve the live wire and parsed tree"
)]
fn admit_request_input_snapshot(ctx: &mut HttpFilterContext<'_>, body: &Option<Bytes>) -> Result<(), FilterAction> {
    if should_skip(ctx) {
        return Ok(());
    }
    let Some(policy_limit) = ctx
        .extensions
        .get::<AgenticBudgetPolicy>()
        .map(|policy| policy.max_retained_bytes())
    else {
        return Ok(());
    };
    let raw_bytes = body.as_ref().map_or(0, Bytes::len);
    // The buffered wire remains live while serde_json builds another owned
    // tree. Exponent-form numbers can expand in that tree before input is
    // moved into the store snapshot.
    let parsed_bound = body.as_deref().map_or(Some(0), buffered_parsed_json_bytes_upper_bound);
    let admitted = if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.apply_retained_payload_limit(policy_limit);
        parsed_bound
            .and_then(|parsed| parsed.checked_add(raw_bytes))
            .is_some_and(|peak| state.can_retain_payload(peak))
    } else {
        // A valid pipeline may place the store after format classification but
        // before the filter that creates ResponsesState. Apply the same
        // pre-parse headroom as the request filter until that state exists;
        // the creator then transfers this captured snapshot into its meter.
        let existing = retained_request_payload_bytes(ctx).unwrap_or(usize::MAX);
        parsed_bound
            .and_then(|parsed| parsed.checked_mul(super::super::INITIAL_RAW_REQUEST_BODY_BUDGET_MULTIPLIER))
            .and_then(|peak| existing.checked_add(peak))
            .is_some_and(|peak| peak <= policy_limit)
    };
    if admitted {
        return Ok(());
    }
    Err(reject_retained_payload_budget(
        ctx,
        "request and rehydrated state exceed openai_agentic_loop.max_retained_bytes",
    ))
}

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
    /// bounded `StreamBuffer` dynamically in [`Self::on_response`].
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
        // The preceding round may have cached its terminal after the agentic
        // loop advanced `iteration`; request-side filters can now rewrite
        // stable fields without changing that counter or vector lengths.
        if let Some(state) = ctx.extensions.get_mut::<ResponseStoreRequestState>() {
            state.shared_stable_bytes = None;
        }
        if ctx.request.method == http::Method::GET {
            return self.handle_get_request(ctx).await;
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
        if let Some(state) = ctx.extensions.get_mut::<ResponseStoreRequestState>() {
            state.shared_stable_bytes = None;
        }
        if let Err(action) = capture_persistence_owner(ctx) {
            return Ok(action);
        }
        if let Err(action) = admit_request_input_snapshot(ctx, body) {
            return Ok(action);
        }
        if !should_skip(ctx)
            && let Some(input) = extract_request_input(body)
            && let Err(action) = capture_request_input(ctx, input)
        {
            return Ok(action);
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

    #[expect(
        clippy::too_many_lines,
        reason = "canonical header persistence and fallback buffer selection share one phase"
    )]
    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.extensions.get::<BufferedResponsePersistenceAttempted>().is_some() {
            return Ok(FilterAction::Continue);
        }
        if should_skip_persist(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !response_is_persistable(ctx) {
            return Ok(FilterAction::Continue);
        }

        if !store_available(ctx) {
            return Ok(FilterAction::Reject(reject_store_error()));
        }

        let budgeted = ctx
            .extensions
            .get::<ResponsesState>()
            .and_then(ResponsesState::retained_payload_limit)
            .is_some();
        let body_finalized = budgeted
            && ctx
                .extensions
                .get::<ResponsesState>()
                .is_some_and(|state| state.buffered_canonical_finalized);
        let canonical_finalized = body_finalized && super::super::buffered_canonical_completed(ctx);
        let canonical_completed = has_conversation(ctx) && canonical_finalized;
        if !is_streaming_request(ctx) && !canonical_completed {
            // An IRR step has not committed its headers to the client; its
            // buffered body can be measured before the outer response starts.
            let direct_response = ctx.extensions.get::<praxis_filter::IterationState>().is_none();
            let admitted_wire_length = if budgeted && !body_finalized && direct_response {
                let Some(length) = buffered_header_persistence_length(ctx) else {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                };
                Some(length)
            } else {
                None
            };
            let max_bytes = if canonical_finalized {
                // The agentic loop already admitted and serialized this exact
                // body. A previous-response restore may have selected a 64 MiB
                // ceiling, but the ceiling itself owns no payload allocation.
                MAX_JSON_BODY_BYTES
            } else if let Some(state) = ctx.extensions.get::<ResponsesState>().filter(|_| budgeted) {
                let Some(limit) = state.retained_payload_limit() else {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                };
                let Some(current) = state.retained_payload_bytes_bounded(limit) else {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                };
                limit.saturating_sub(current).min(MAX_JSON_BODY_BYTES)
            } else {
                MAX_JSON_BODY_BYTES
            };
            if budgeted
                && !body_finalized
                && let BodyMode::StreamBuffer { max_bytes: existing } = &ctx.response_body_mode
                && existing.is_none_or(|bytes| bytes > max_bytes)
            {
                // Core keeps the larger cap. An earlier response filter's
                // buffer would otherwise allocate beyond this request's
                // remaining aggregate allowance before our body hook runs.
                return Ok(persistence_budget_failure(ctx, false, &mut None));
            }
            ctx.set_response_body_mode(BodyMode::StreamBuffer {
                max_bytes: Some(admitted_wire_length.map_or(max_bytes, |length| length.min(max_bytes))),
            });
        }

        if canonical_completed && !is_streaming_request(ctx) {
            let Some(state) = ctx.extensions.get::<ResponsesState>() else {
                return Ok(FilterAction::Reject(reject_store_error()));
            };
            if state.response_object.get("status").and_then(Value::as_str) == Some("completed") {
                let Some(response_bytes) = retained_json_bytes(&state.response_object) else {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                };
                // The serialized wire body remains live outside ResponsesState
                // while the record, SQL arguments, and conversation append are built.
                if !persistence_construction_fits_with_wire(ctx, response_bytes, response_bytes) {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                }
                let pending_approvals = pending_approvals_from_ctx(ctx);
                let persist = match take_persist_context(ctx) {
                    Ok(parts) => parts,
                    Err(action) => return Ok(action),
                };
                let Some(record) = build_streaming_record(ctx, persist.owner, persist.request_input) else {
                    return Ok(FilterAction::Reject(reject_store_error()));
                };
                let inserted = match persist
                    .store
                    .persist_response_with_pending_approvals_if_absent(&record, &pending_approvals)
                    .await
                {
                    Ok(inserted) => inserted,
                    Err(StoreError::PayloadTooLarge) => {
                        return Ok(persistence_budget_failure(ctx, false, &mut None));
                    },
                    Err(error) => return Err(Box::new(error)),
                };
                if !inserted {
                    return Ok(persistence_budget_failure(ctx, false, &mut None));
                }
                ctx.extensions.insert(BufferedResponsePersistenceAttempted);
                #[cfg(feature = "openai-conversations")]
                ctx.extensions.insert(StoreResponseHeaderRan(response_round(ctx)));
                #[cfg(feature = "openai-conversations")]
                ctx.extensions.insert(PersistedResponseForConversation(record.id));
                #[cfg(feature = "openai-conversations")]
                {
                    return crate::openai::conversations::append_after_store_response(ctx).await;
                }
                #[cfg(not(feature = "openai-conversations"))]
                {
                    return Ok(FilterAction::Continue);
                }
            }
        }

        #[cfg(feature = "openai-conversations")]
        ctx.extensions.insert(StoreResponseHeaderRan(response_round(ctx)));
        trace!("response body persistence armed");

        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "stream persistence and terminal handling share one callback"
    )]
    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if ctx.extensions.get::<BufferedResponsePersistenceAttempted>().is_some() {
            return Ok(FilterAction::Continue);
        }
        if ctx.get_metadata("responses.store_stream_budget_failed") == Some("true") {
            *body = None;
            return Ok(FilterAction::Continue);
        }
        if Self::should_release_skipped_response_body(ctx) {
            // A committed SSE error still needs the store filter's request
            // state while chunks are being released. Drop its input snapshot
            // only at EOS, after the wire terminal has been emitted.
            if end_of_stream
                && ctx
                    .extensions
                    .get::<ResponsesState>()
                    .is_some_and(|state| state.retained_payload_failed)
            {
                discard_retained_request_payload(ctx);
            }
            return Ok(FilterAction::Release);
        }

        if is_streaming_request(ctx) {
            // Capture this chunk's SSE events into request-scoped state before
            // any persist seam below drains that state. Parse-only; the
            // client-visible bytes are never withheld or altered.
            if !self.capture_stream_events(ctx, body, end_of_stream) {
                return streaming_persistence_budget_failure(ctx, body);
            }

            // This signal is only promoted after the whole chunk passes
            // admission. A terminal in a rejected chunk was never forwarded.
            let terminal_in_chunk = ctx
                .extensions
                .get_mut::<ResponseStoreRequestState>()
                .is_some_and(|state| std::mem::take(&mut state.terminal_in_chunk));
            let (max_sequence_in_chunk, unclassified_in_chunk) = ctx
                .extensions
                .get_mut::<ResponseStoreRequestState>()
                .map(|state| {
                    (
                        state.max_sequence_in_chunk.take(),
                        std::mem::take(&mut state.unclassified_in_chunk),
                    )
                })
                .unwrap_or_default();
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
                    match Self::persist_from_streaming_state(ctx, body)? {
                        FilterAction::Continue => {},
                        action => return Ok(action),
                    }
                }
                mark_released_stream_chunk(
                    ctx,
                    body,
                    terminal_in_chunk,
                    max_sequence_in_chunk,
                    unclassified_in_chunk,
                );
                return Ok(FilterAction::Release);
            }
            // Skip after an earlier chunk consumed persistence state. This
            // includes fail-open errors, which cannot be retried at EOS. The
            // fallback remains available to non-IRR streams without a terminal
            // marker; their captured SSE errors still suppress persistence.
            if ctx.extensions.get::<StreamingResponsePersistenceAttempted>().is_some() {
                return Ok(FilterAction::Continue);
            }
            let action = Self::persist_from_streaming_state(ctx, body)?;
            if matches!(action, FilterAction::Continue | FilterAction::Release) {
                mark_released_stream_chunk(
                    ctx,
                    body,
                    terminal_in_chunk,
                    max_sequence_in_chunk,
                    unclassified_in_chunk,
                );
            }
            return Ok(action);
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
    let first_id = page.data.first().and_then(|v| v.get("id")).and_then(|v| v.as_str());
    // Items normally carry a synthetic ID (see `normalize_input_items`),
    // but non-object array entries can't be tagged with one. Fall back
    // to the page's numeric cursor so `after`-based pagination stays
    // usable even for that edge case, instead of exposing a `null`
    // `last_id` clients have no way to resume from.
    let last_id = page
        .data
        .last()
        .and_then(|v| v.get("id"))
        .and_then(|v| v.as_str())
        .or(page.next_cursor.as_deref());

    let body = serde_json::json!({
        "object": "list",
        "data": page.data,
        "has_more": page.has_more,
        "first_id": first_id,
        "last_id": last_id,
    });
    debug!(
        response_id = id,
        count = page.data.len(),
        has_more = page.has_more,
        "serving input items"
    );
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
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
    use std::num::{NonZeroU32, NonZeroU64};

    use bytes::Bytes;
    use praxis_filter::{FilterAction, HttpFilter as _, sse::SseDecoder};
    use serde_json::json;

    use super::{
        AgenticBudgetPolicy, CapturedEvent, ResponseEventRecord, ResponseStoreFilter, StateOwner,
        admit_request_input_snapshot, buffered_parsed_json_bytes_upper_bound, buffered_persistence_construction_fits,
        capture_request_input, encode_replay_event, encoded_column_headroom, persistence_budget_failure,
        persistence_construction_fits,
    };
    use crate::openai::responses::{
        ObservedResponsesSse,
        state::{OutputAssignment, ResponsesState},
    };

    #[test]
    fn encoded_responses_are_not_persistable() {
        for (content_type, streaming) in [("application/json", "false"), ("text/event-stream", "true")] {
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            ctx.set_metadata("openai_responses_format.stream", streaming);
            let mut response = crate::test_utils::make_response();
            response
                .headers
                .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static(content_type));
            response
                .headers
                .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
            ctx.response_header = Some(&mut response);

            assert!(!super::response_is_persistable(&mut ctx), "{content_type}");
            assert_eq!(
                ctx.get_metadata("responses.skip_persist"),
                Some("true"),
                "{content_type}"
            );
        }
    }

    #[test]
    fn streaming_persistence_without_response_state_keeps_final_chunk() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from("default"),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "true");
        ctx.extensions.insert(super::ResponseStoreRequestState {
            owner: Some(StateOwner::from_trusted_parts("t", "i", "s").unwrap()),
            ..Default::default()
        });
        let frame = Bytes::from_static(b"event: response.completed\ndata: {}\n\n");
        let mut body = Some(frame.clone());

        let action = ResponseStoreFilter::persist_from_streaming_state(&mut ctx, &mut body).unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(body, Some(frame));
        assert_ne!(ctx.get_metadata("responses.store_stream_budget_failed"), Some("true"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[expect(
        clippy::too_many_lines,
        reason = "collision proof checks the response, replay, and approval rows"
    )]
    async fn budgeted_stream_collision_preserves_prior_response_events_and_approvals() {
        use crate::store::{PendingApprovalRecord, ResponseRecord};

        let owner = StateOwner::from_trusted_parts("t", "i", "s").unwrap();
        let store: std::sync::Arc<dyn crate::store::PersistedStateBackend> =
            std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new());
        let prior = ResponseRecord {
            id: "resp_collision".to_owned(),
            owner: owner.clone(),
            created_at: 1,
            model: "old".to_owned(),
            response_object: json!({"id":"resp_collision","status":"completed","output":[]}),
            input: json!([]),
            messages: json!([]),
        };
        let approval = PendingApprovalRecord {
            approval_id: "approval_old".to_owned(),
            server_label: "server".to_owned(),
            tool_name: "old_tool".to_owned(),
            arguments: "{\"old\":true}".to_owned(),
            target_fingerprint: "old_fp".to_owned(),
        };
        store
            .persist_response_with_pending_approvals(&prior, std::slice::from_ref(&approval))
            .await
            .unwrap();
        store
            .append_events(
                &owner,
                &prior.id,
                &[ResponseEventRecord {
                    response_id: prior.id.clone(),
                    owner: owner.clone(),
                    sequence_number: 1,
                    event_type: "response.completed".to_owned(),
                    payload: b"old event".to_vec(),
                    terminal: true,
                    created_at: 1,
                }],
            )
            .await
            .unwrap();
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(&std::sync::Arc::from("default"), std::sync::Arc::clone(&store))
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "true");
        ctx.set_metadata("openai_responses_format.has_conversation", "true");
        ctx.extensions.insert(super::ResponseStoreRequestState {
            owner: Some(owner.clone()),
            ..Default::default()
        });
        let mut state = ResponsesState {
            response_object: json!({
                "id":"resp_collision", "created_at":2, "model":"new", "status":"completed",
                "output":[{"type":"message","role":"assistant","content":[]}]
            }),
            ..Default::default()
        };
        state.apply_retained_payload_limit(1024 * 1024);
        ctx.extensions.insert(state);
        let mut terminal = Some(Bytes::from_static(b"event: response.completed\ndata: {}\n\n"));
        let action = ResponseStoreFilter::persist_from_streaming_state(&mut ctx, &mut terminal).unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert!(
            terminal
                .as_deref()
                .is_some_and(|body| body.starts_with(b"event: error\n"))
        );
        assert_eq!(
            store.get_response(&owner, &prior.id).await.unwrap().unwrap().model,
            "old"
        );
        assert_eq!(
            store
                .list_events_after(&owner, &prior.id, None, 10)
                .await
                .unwrap()
                .first()
                .unwrap()
                .payload,
            b"old event"
        );
        let approvals = store
            .get_pending_approvals(&owner, &prior.id, &["approval_old"])
            .await
            .unwrap();
        assert_eq!(approvals.first().unwrap().arguments, approval.arguments);
    }

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

    /// Many individually small replay rows must still exhaust the shared
    /// request budget before the store retains another row.
    #[test]
    fn replay_capture_charges_aggregate_event_payload() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState::default();
        responses.apply_retained_payload_limit(4_096);
        assert!(responses.retain_external_payload_bytes(128));
        ctx.extensions.insert(responses);

        let mut accepted = 0;
        for sequence in 0..100 {
            let frame = Bytes::from(format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{sequence},\"delta\":\"x\"}}\n\n"
            ));
            if !filter.capture_stream_events(&mut ctx, &Some(frame), false) {
                break;
            }
            accepted += 1;
        }
        assert!(accepted > 1 && accepted < 100, "aggregate budget must stop small rows");
        let store_bytes = super::retained_request_payload_bytes(&ctx).unwrap();
        assert!(store_bytes > 0, "the store must retain at least one replay row");
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            128 + store_bytes,
            "replay accounting must preserve other sibling-filter charges"
        );
    }

    /// The record builder accepts the provider's response ID, and the replay
    /// flush copies it into every row. Admit all those copies before persistence.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "sets the exact near-limit replay ownership boundary"
    )]
    fn replay_persistence_preflights_per_row_response_id_copies() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let response_id = format!("resp_{}", "x".repeat(8_192));
        let response_id_len = response_id.len();
        let mut store = super::ResponseStoreRequestState::default();
        for sequence_number in 0..64 {
            let payload = format!(
                "{{\"type\":\"response.output_text.delta\",\"sequence_number\":{sequence_number},\"delta\":\"x\"}}"
            )
            .into_bytes();
            store.event_bytes += payload.len() as u64;
            store.event_name_bytes += "response.output_text.delta".len();
            store.events.push(CapturedEvent {
                sequence_number,
                event_type: "response.output_text.delta".to_owned(),
                payload,
                terminal: false,
            });
        }
        let store_bytes = store.retained_payload_bytes().unwrap();
        let mut responses = ResponsesState {
            response_object: json!({"id": response_id, "created_at": 0, "model": "m", "output": []}),
            ..ResponsesState::default()
        };
        responses.set_retained_external_payload_bytes(store_bytes);
        let response_bytes = super::retained_json_bytes(&responses.response_object).unwrap();
        let baseline = responses.retained_payload_bytes().unwrap();
        let payload_bytes = usize::try_from(store.event_bytes).unwrap();
        let replay_staging = payload_bytes * 2 + (payload_bytes >> 8) + store.events.len() * 64;
        let record_staging =
            response_bytes * 6 + encoded_column_headroom(response_bytes, 0, 0).unwrap() + store_bytes + replay_staging;
        let replay_id_copies = response_id_len * store.events.len();
        responses.apply_retained_payload_limit(baseline + record_staging + replay_id_copies);
        ctx.extensions.insert(responses);
        ctx.extensions.insert(store);

        assert!(persistence_construction_fits(&ctx, response_bytes));
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(baseline + record_staging + replay_id_copies - 1);
        assert!(!persistence_construction_fits(&ctx, response_bytes));
    }

    /// Zstd copies the captured payloads into a worker, then owns the frames.
    /// Reserve that peak even when the codec is hidden behind the store trait.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "sets the exact replay compression staging boundary"
    )]
    fn replay_persistence_preflights_compression_payload_copies() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut capture = super::ResponseStoreRequestState::default();
        for sequence_number in 0..16 {
            let payload = serde_json::to_vec(&json!({
                "type": "response.output_text.delta",
                "sequence_number": sequence_number,
                "delta": "x".repeat(4_096),
            }))
            .unwrap();
            capture.event_bytes += payload.len() as u64;
            capture.event_name_bytes += "response.output_text.delta".len();
            capture.events.push(CapturedEvent {
                sequence_number,
                event_type: "response.output_text.delta".to_owned(),
                payload,
                terminal: false,
            });
        }
        let captured_bytes = capture.retained_payload_bytes().unwrap();
        let payload_bytes = usize::try_from(capture.event_bytes).unwrap();
        let replay_staging = payload_bytes * 2 + (payload_bytes >> 8) + capture.events.len() * 64;
        let mut responses = ResponsesState {
            response_object: json!({"id": "resp_zstd", "created_at": 0, "model": "m", "output": []}),
            ..ResponsesState::default()
        };
        responses.set_retained_external_payload_bytes(captured_bytes);
        let response_bytes = super::retained_json_bytes(&responses.response_object).unwrap();
        let baseline = responses.retained_payload_bytes().unwrap();
        let existing_staging = response_bytes * 6
            + encoded_column_headroom(response_bytes, 0, 0).unwrap()
            + captured_bytes
            + "resp_zstd".len() * capture.events.len();
        responses.apply_retained_payload_limit(baseline + existing_staging + replay_staging);
        ctx.extensions.insert(responses);
        ctx.extensions.insert(capture);

        assert!(persistence_construction_fits(&ctx, response_bytes));
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(baseline + existing_staging + replay_staging - 1);
        assert!(!persistence_construction_fits(&ctx, response_bytes));
    }

    /// `PostgreSQL` copies serialized input into its query argument buffer while
    /// both the request snapshot and serialized input still own their bytes.
    #[test]
    #[expect(clippy::too_many_lines, reason = "checks both input bind and replay row boundaries")]
    fn persistence_preflights_postgres_input_bind_copy() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let input = json!([{"role": "user", "content": "x".repeat(1_048_576)}]);
        let input_bytes = super::retained_json_bytes(&input).unwrap();
        let responses = ResponsesState {
            response_object: json!({"id": "resp_pg_bind", "created_at": 0, "model": "m", "output": []}),
            ..ResponsesState::default()
        };
        ctx.extensions.insert(responses);
        capture_request_input(&mut ctx, input).unwrap();

        let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        let response_bytes = super::retained_json_bytes(&state.response_object).unwrap();
        let baseline = state.retained_payload_bytes().unwrap();
        let construction =
            response_bytes * 6 + input_bytes * 2 + encoded_column_headroom(response_bytes, 0, input_bytes).unwrap();
        state.apply_retained_payload_limit(baseline + construction);
        assert!(persistence_construction_fits(&ctx, response_bytes));

        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(baseline + construction - 1);
        assert!(
            !persistence_construction_fits(&ctx, response_bytes),
            "one byte below the PostgreSQL bind-copy peak must reject"
        );

        // Replay rows share the store snapshot charge, but do not enter the
        // PostgreSQL input bind parameter. They must not be doubled here.
        let event_type = "response.output_text.delta";
        let replay_bytes = 65_536;
        let capture = ctx.extensions.get_mut::<super::ResponseStoreRequestState>().unwrap();
        capture.event_bytes += replay_bytes as u64;
        capture.event_name_bytes += event_type.len();
        capture.events.push(CapturedEvent {
            sequence_number: 1,
            event_type: event_type.to_owned(),
            payload: vec![b'x'; replay_bytes],
            terminal: false,
        });
        let snapshot_bytes = capture.retained_payload_bytes().unwrap();
        let replay_staging = replay_bytes * 2 + (replay_bytes >> 8) + 64;
        ctx.extensions.insert(ResponsesState {
            response_object: json!({"id": "resp_pg_bind", "created_at": 0, "model": "m", "output": []}),
            ..ResponsesState::default()
        });
        let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        state.set_retained_external_payload_bytes(snapshot_bytes);
        let baseline = state.retained_payload_bytes().unwrap();
        let construction = response_bytes * 6
            + encoded_column_headroom(response_bytes, 0, input_bytes).unwrap()
            + snapshot_bytes
            + input_bytes
            + replay_staging
            + "resp_pg_bind".len();
        state.apply_retained_payload_limit(baseline + construction);
        assert!(persistence_construction_fits(&ctx, response_bytes));
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(baseline + construction - 1);
        assert!(!persistence_construction_fits(&ctx, response_bytes));
    }

    /// A buffered assistant message enters the stored record's messages even
    /// though it is absent from the pre-persist history charge. The response
    /// and messages columns are both serialized and copied by `PostgreSQL`.
    #[test]
    fn persistence_preflights_new_assistant_output_message_bind_copy() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let response_object = json!({
            "id": "resp_output_bind", "created_at": 0, "model": "m",
            "output": [{"type": "message", "role": "assistant", "content": "x".repeat(1_048_576)}]
        });
        let response_bytes = super::retained_json_bytes(&response_object).unwrap();
        let mut state = ResponsesState {
            response_object,
            ..ResponsesState::default()
        };
        assert!(state.persisted_messages.is_empty());
        let baseline = state.retained_payload_bytes().unwrap();
        let headroom = encoded_column_headroom(response_bytes, 0, 0).unwrap();
        state.apply_retained_payload_limit(baseline + response_bytes * 6 + headroom);
        ctx.extensions.insert(state);
        assert!(persistence_construction_fits(&ctx, response_bytes));
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(baseline + response_bytes * 5 + response_bytes / 2);
        assert!(
            !persistence_construction_fits(&ctx, response_bytes),
            "newly appended output must count in both stored columns and binds"
        );
    }

    /// Exponent-form numbers in an opaque response field expand after JSON
    /// parsing, so the store must reserve parsed owners before decoding.
    #[test]
    fn buffered_persistence_preflights_numeric_normalization() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let numbers = vec!["1e15"; 128].join(",");
        let wire = format!(
            "{{\"id\":\"resp_numeric\",\"created_at\":0,\"model\":\"m\",\"output\":[],\"metadata\":[{numbers}]}}"
        );
        let projected = buffered_parsed_json_bytes_upper_bound(wire.as_bytes()).unwrap();
        assert!(projected > wire.len());
        let mut state = ResponsesState {
            response_object: serde_json::from_str(&wire).unwrap(),
            ..ResponsesState::default()
        };
        let baseline = state.retained_payload_bytes().unwrap();
        state.apply_retained_payload_limit(
            baseline
                + wire.len() * 6
                + (projected - wire.len()) * 3
                + encoded_column_headroom(projected, 0, 0).unwrap(),
        );
        ctx.extensions.insert(state);
        assert!(persistence_construction_fits(&ctx, wire.len()));
        assert!(!buffered_persistence_construction_fits(&ctx, wire.as_bytes()));
    }

    #[test]
    fn request_input_snapshot_preflights_numeric_normalization_with_live_history() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        let raw = Bytes::from(format!(
            r#"{{"model":"m","input":"hello","numbers":[{}]}}"#,
            vec!["1e15"; 256].join(",")
        ));
        let parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let parsed_bytes = super::retained_json_bytes(&parsed).unwrap();
        assert!(parsed_bytes > raw.len());
        let mut state = ResponsesState::from_request_body(parsed);
        state.iteration = 1;
        state
            .messages
            .insert(0, json!({"role":"assistant","content":"x".repeat(65_536)}));
        let baseline = state.retained_payload_bytes().unwrap();
        let limit = baseline + raw.len() + 1;
        state.apply_retained_payload_limit(limit);
        assert!(state.can_retain_payload(raw.len()));
        assert!(!state.can_retain_payload(parsed_bytes));
        ctx.extensions.insert(state);
        ctx.extensions.insert(
            AgenticBudgetPolicy::from_config(&serde_yaml::from_str(&format!("max_retained_bytes: {limit}")).unwrap())
                .unwrap(),
        );

        let result = admit_request_input_snapshot(&mut ctx, &Some(raw));

        assert!(matches!(result, Err(FilterAction::Reject(_))));
        assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    }

    #[test]
    fn request_input_snapshot_overflow_uses_continuation_wire() {
        for streaming in [false, true] {
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            ctx.set_metadata("openai_responses_format.format", "openai_responses");
            ctx.extensions.insert(
                AgenticBudgetPolicy::from_config(&serde_yaml::from_str("max_retained_bytes: 4096").unwrap()).unwrap(),
            );
            let mut state = ResponsesState {
                iteration: 1,
                request_body: json!({"stream": streaming}),
                messages: vec![json!("x".repeat(2800))],
                ..ResponsesState::default()
            };
            state.apply_retained_payload_limit(4096);
            assert!(state.can_retain_payload(0));
            ctx.extensions.insert(state);
            if streaming {
                ctx.extensions.insert(ObservedResponsesSse);
            }
            let body = Some(Bytes::from(
                serde_json::to_vec(&json!({"input":"y".repeat(1000)})).unwrap(),
            ));

            let action = admit_request_input_snapshot(&mut ctx, &body).unwrap_err();
            assert!(matches!(action, FilterAction::Reject(_)));
            if let FilterAction::Reject(rejection) = action {
                assert_eq!(rejection.status, if streaming { 200 } else { 502 });
            }
            assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
        }
    }

    #[test]
    fn captured_input_overflow_uses_continuation_wire() {
        for streaming in [false, true] {
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            let mut state = ResponsesState {
                iteration: 1,
                request_body: json!({"stream": streaming}),
                messages: vec![json!("x".repeat(2800))],
                ..ResponsesState::default()
            };
            let baseline = state.retained_payload_bytes().unwrap();
            state.apply_retained_payload_limit(baseline + 64);
            ctx.extensions.insert(state);
            if streaming {
                ctx.extensions.insert(ObservedResponsesSse);
            }

            let action = capture_request_input(&mut ctx, json!("y".repeat(1000))).unwrap_err();
            assert!(matches!(action, FilterAction::Reject(_)));
            if let FilterAction::Reject(rejection) = action {
                assert_eq!(rejection.status, if streaming { 200 } else { 502 });
            }
            assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
        }
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "checks input, replay, and replacement charges")]
    fn ordinary_stream_store_charges_input_before_replay_rows() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.extensions.insert(ResponsesState::default());
        let input = json!("plain store input");
        let input_bytes = super::retained_json_bytes(&input).unwrap();
        capture_request_input(&mut ctx, input).unwrap();
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            input_bytes,
        );
        let frame = Bytes::from_static(
            b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1}\n\n",
        );
        assert!(filter.capture_stream_events(&mut ctx, &Some(frame), false));
        let retained = super::retained_request_payload_bytes(&ctx).unwrap();
        assert!(retained > input_bytes);
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            retained,
        );
        let replacement = json!("replacement input");
        capture_request_input(&mut ctx, replacement).unwrap();
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            super::retained_request_payload_bytes(&ctx).unwrap(),
            "replacing the cached input charge must preserve replay row charges",
        );
    }

    #[test]
    fn stream_store_charges_input_when_response_state_is_created_later() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        capture_request_input(&mut ctx, json!("input captured before MCP resolution")).unwrap();
        ctx.extensions.insert(ResponsesState::default());

        let frame = Bytes::from_static(
            b"event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":1}\n\n",
        );
        assert!(filter.capture_stream_events(&mut ctx, &Some(frame), false));
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            super::retained_request_payload_bytes(&ctx).unwrap(),
        );
    }

    #[test]
    fn stream_store_replaces_charge_seeded_by_agentic_admission() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        capture_request_input(&mut ctx, json!("input captured before admission")).unwrap();
        let input_bytes = super::retained_request_payload_bytes(&ctx).unwrap();
        let mut responses = ResponsesState::default();
        responses.set_retained_external_payload_bytes(input_bytes);
        ctx.extensions.insert(responses);
        super::mark_retained_request_payload_charged(&mut ctx);

        let frame = Bytes::from_static(
            b"event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":1}\n\n",
        );
        assert!(filter.capture_stream_events(&mut ctx, &Some(frame), false));
        assert_eq!(
            ctx.extensions
                .get::<ResponsesState>()
                .unwrap()
                .retained_external_payload_bytes,
            super::retained_request_payload_bytes(&ctx).unwrap(),
        );
    }

    #[test]
    fn store_budget_error_suppresses_later_provider_terminal() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.extensions.insert(ResponsesState::default());
        let mut first = Some(Bytes::from_static(b"event: response.in_progress\ndata: {}\n\n"));
        drop(persistence_budget_failure(&mut ctx, true, &mut first));
        assert!(String::from_utf8_lossy(first.as_ref().unwrap()).contains("event: error"));
        assert_eq!(
            ctx.filter_results
                .get("openai_agentic_loop")
                .and_then(|result| result.get("action")),
            Some("done"),
        );
        let mut later = Some(Bytes::from_static(b"event: response.completed\ndata: {}\n\n"));
        drop(filter.on_response_body(&mut ctx, &mut later, true).unwrap());
        assert!(later.is_none(), "a provider terminal after the error must be withheld");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks admitted, rejected, and payload-only SSE frames"
    )]
    fn direct_store_overflow_after_forwarded_completion_aborts_transport() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        for terminal in [
            b"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":0}\n\n".as_slice(),
            b"data: {\"type\":\"response.completed\",\"sequence_number\":0}\n\n".as_slice(),
            b"event: response.completed\ndata: {}\n\n".as_slice(),
            b"data: {\"type\":\"response.completed\"}\n\n".as_slice(),
        ] {
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            ctx.set_metadata("openai_responses_format.format", "openai_responses");
            ctx.set_metadata("openai_responses_format.stream", "true");
            let registry = crate::store::ResponseStoreRegistry::new();
            registry
                .register(
                    &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                    std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
                )
                .unwrap();
            ctx.extensions.insert(registry);
            let mut responses = ResponsesState::from_request_body(json!({"model":"m","input":"x","stream":true}));
            responses.apply_retained_payload_limit(4_096);
            ctx.extensions.insert(responses);

            let mut completed = Some(Bytes::copy_from_slice(terminal));
            assert!(matches!(
                filter.on_response_body(&mut ctx, &mut completed, false).unwrap(),
                FilterAction::Release
            ));
            assert_eq!(completed.as_deref(), Some(terminal));

            let trailing = b": keepalive\n\n";
            let mut admitted_tail = Some(Bytes::from_static(trailing));
            assert!(matches!(
                filter.on_response_body(&mut ctx, &mut admitted_tail, false).unwrap(),
                FilterAction::Release
            ));
            assert_eq!(admitted_tail.as_deref(), Some(trailing.as_slice()));

            let mut overflow = Some(Bytes::from(vec![b'x'; 2_048]));
            assert!(
                filter.on_response_body(&mut ctx, &mut overflow, false).is_err(),
                "a completed stream cannot receive a second terminal error"
            );
            assert!(overflow.is_none(), "the rejected chunk must not reach the client");
            assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks that a rejected terminal does not change the wire state"
    )]
    fn terminal_in_rejected_chunk_is_not_marked_as_forwarded() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "true");
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        let mut responses = ResponsesState::from_request_body(json!({"model":"m","input":"x","stream":true}));
        responses.apply_retained_payload_limit(4_096);
        ctx.extensions.insert(responses);

        let mut chunk =
            b"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":0}\n\n".to_vec();
        chunk.extend_from_slice(&vec![b'x'; 2_048]);
        let mut rejected = Some(Bytes::from(chunk));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut rejected, false).unwrap(),
            FilterAction::Continue
        ));
        assert!(
            rejected
                .as_ref()
                .is_some_and(|bytes| bytes.starts_with(b"event: error")),
            "the only terminal should be the replacement error"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks data-less SSE framing across separate chunks"
    )]
    fn data_less_terminal_header_does_not_mark_a_forwarded_terminal() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "true");
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        let mut responses = ResponsesState::from_request_body(json!({"model":"m","input":"x","stream":true}));
        responses.apply_retained_payload_limit(4_096);
        ctx.extensions.insert(responses);

        let mut header = Some(Bytes::from_static(b"event: response.completed\n\n"));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut header, false).unwrap(),
            FilterAction::Release
        ));
        let mut overflow = Some(Bytes::from(vec![b'x'; 2_048]));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut overflow, false).unwrap(),
            FilterAction::Continue
        ));
        assert!(
            overflow
                .as_ref()
                .is_some_and(|bytes| bytes.starts_with(b"event: error"))
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "reproduces an EOS overflow after replay capture was abandoned"
    )]
    fn abandoned_replay_still_tracks_a_forwarded_terminal_before_eos() {
        let filter = ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "true");
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        let input = "x".repeat(400);
        let mut responses = ResponsesState::from_request_body(json!({"model":"m","input":input,"stream":true}));
        responses.apply_retained_payload_limit(4_096);
        ctx.extensions.insert(responses);
        capture_request_input(&mut ctx, json!(input)).unwrap();

        let mut first = Some(Bytes::from_static(
            b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0}\n\n",
        ));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut first, false).unwrap(),
            FilterAction::Release
        ));
        assert!(
            ctx.extensions
                .get::<super::ResponseStoreRequestState>()
                .unwrap()
                .events_over_budget
        );

        let terminal = b"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1}\n\n";
        let mut completed = Some(Bytes::copy_from_slice(terminal));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut completed, false).unwrap(),
            FilterAction::Release
        ));
        assert_eq!(completed.as_deref(), Some(terminal.as_slice()));

        let mut eos = None;
        assert!(filter.on_response_body(&mut ctx, &mut eos, true).is_err());
        assert!(eos.is_none(), "a second terminal must not follow the completed frame");
        assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks admitted provider sequences with and without replay capture"
    )]
    fn direct_store_budget_error_follows_admitted_provider_sequence() {
        for max_event_bytes in [1, 1_048_576] {
            let filter = ResponseStoreFilter::with_bounds(
                NonZeroU32::new(1_024).unwrap(),
                NonZeroU64::new(max_event_bytes).unwrap(),
            );
            let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            ctx.set_metadata("openai_responses_format.format", "openai_responses");
            ctx.set_metadata("openai_responses_format.stream", "true");
            let registry = crate::store::ResponseStoreRegistry::new();
            registry
                .register(
                    &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                    std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
                )
                .unwrap();
            ctx.extensions.insert(registry);
            let input = "x".repeat(400);
            let mut responses = ResponsesState::from_request_body(json!({"model":"m","input":input,"stream":true}));
            responses.apply_retained_payload_limit(4_096);
            ctx.extensions.insert(responses);
            capture_request_input(&mut ctx, json!(input)).unwrap();

            let first =
                b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":7}\n\n";
            let mut frame = Some(Bytes::copy_from_slice(first));
            assert!(matches!(
                filter.on_response_body(&mut ctx, &mut frame, false).unwrap(),
                FilterAction::Release
            ));
            assert_eq!(frame.as_deref(), Some(first.as_slice()));

            let mut eos = None;
            let action = filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
            assert!(matches!(action, FilterAction::Continue));
            let (_, payload) = decode_single(eos.as_deref().unwrap());
            let error: serde_json::Value = serde_json::from_slice(&payload).unwrap();
            assert_eq!(error.get("sequence_number"), Some(&json!(8)));
            assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
        }
    }

    /// Once replay is abandoned at its independent cap, later wire chunks do
    /// not create store-owned payload and must not consume the agentic budget.
    #[test]
    fn abandoned_replay_stops_charging_new_chunks() {
        let filter = ResponseStoreFilter::with_bounds(NonZeroU32::new(1).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState::default();
        responses.apply_retained_payload_limit(4_096);
        ctx.extensions.insert(responses);
        for sequence in 0..2 {
            let frame = Bytes::from(format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{sequence},\"delta\":\"x\"}}\n\n"
            ));
            assert!(filter.capture_stream_events(&mut ctx, &Some(frame), false));
        }
        assert!(filter.capture_stream_events(&mut ctx, &Some(Bytes::from(vec![b'x'; 3_000])), false));
        assert_eq!(super::retained_request_payload_bytes(&ctx), Some(0));
    }

    #[test]
    fn replay_meter_refreshes_large_request_charge_at_end_of_stream() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState::from_request_body(json!({"input": "x".repeat(1_048_576)}));
        responses.apply_retained_payload_limit(8_388_608);
        ctx.extensions.insert(responses);

        for sequence in 0..32 {
            let frame = Bytes::from(format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{sequence},\"delta\":\"x\"}}\n\n"
            ));
            assert!(filter.capture_stream_events(&mut ctx, &Some(frame), false));
        }
        assert!(
            ctx.extensions
                .get::<super::ResponseStoreRequestState>()
                .unwrap()
                .shared_stable_bytes
                .is_some()
        );

        // The agentic loop may append history before this filter sees EOS.
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .messages
            .push(json!("y".repeat(7 * 1_048_576)));
        let terminal = Some(Bytes::from_static(
            b"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":32}\n\n",
        ));
        assert!(!filter.capture_stream_events(&mut ctx, &terminal, true));
    }

    #[test]
    fn replay_meter_includes_published_stream_parser_charge() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let frame = Some(Bytes::from_static(
            b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1}\n\n",
        ));
        let mut responses = ResponsesState::default();
        let baseline = responses.retained_payload_bytes().unwrap();
        let peak = frame.as_ref().unwrap().len() * 4;
        responses.apply_retained_payload_limit(baseline + peak);
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.extensions.insert(responses);
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));

        let mut responses = ResponsesState {
            retained_stream_parser_bytes: 1,
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(baseline + peak);
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        ctx.extensions.insert(responses);
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[test]
    fn replay_meter_refreshes_accumulated_output_at_terminal_chunk() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let frame = Some(Bytes::from_static(
            b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1}\n\n",
        ));
        for terminal_is_eos in [false, true] {
            let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
            let mut responses = ResponsesState::default();
            responses
                .accumulated_output
                .push(json!({"type": "message", "content": "prior".repeat(100_000)}));
            responses.apply_retained_payload_limit(1_048_576);
            ctx.extensions.insert(responses);
            assert!(filter.capture_stream_events(&mut ctx, &frame, false));

            let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
            responses
                .accumulated_output
                .push(json!({"type": "message", "content": "x".repeat(700_000)}));
            responses.logical_stream_terminal_emitted = !terminal_is_eos;
            assert!(!filter.capture_stream_events(&mut ctx, &frame, terminal_is_eos));
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "reproduces a near-limit dispatch after a cached round completion"
    )]
    fn replay_meter_refreshes_history_added_after_next_iteration_was_cached() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState {
            iteration: 2,
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(32_768);
        ctx.extensions.insert(responses);

        // A continuing round can emit its synthesized completion after the
        // loop has already incremented iteration, before dispatch of the next
        // round adds history under that same iteration number.
        let frame = Some(Bytes::from_static(
            b"event: response.file_search_call.completed\ndata: {\"type\":\"response.file_search_call.completed\",\"sequence_number\":0}\n\n",
        ));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let published = ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .messages
            .push(json!({
                "type": "function_call_output",
                "output": "x".repeat(32_768 - published - 60),
            }));
        let responses = ctx.extensions.get::<ResponsesState>().unwrap();
        assert!(responses.can_retain_payload(0), "dispatch itself fits the budget");
        assert!(!responses.can_retain_payload(frame.as_ref().unwrap().len() * 4));
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "measures an in-place hosted output update at the replay boundary"
    )]
    fn replay_meter_refreshes_in_place_hosted_output_update() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState {
            accumulated_output: vec![json!({"type": "file_search_call", "id": "fs_1", "status": "searching"})],
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(32_768);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(
            b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"delta\":\"x\"}\n\n",
        ));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let published = ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        responses
            .accumulated_output
            .get_mut(0)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "results".to_owned(),
                json!([{"text": "x".repeat(32_768 - published - 80)}]),
            );
        responses.mark_replay_stable_payload_changed();
        assert_eq!(responses.accumulated_output.len(), 1);
        assert!(responses.messages.is_empty());
        assert!(responses.can_retain_payload(0));
        assert!(!responses.can_retain_payload(frame.as_ref().unwrap().len() * 4));
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "exercises repeated store admission and its cached charge"
    )]
    fn replay_meter_reuses_completed_output_across_small_chunks() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let call = json!({"type": "function_call", "id": "fc_stable", "arguments": "x".repeat(512 * 1_024)});
        let assignment = OutputAssignment::new(0, &call).unwrap();
        let mut responses = ResponsesState {
            response_object: json!({"output": [call.clone()]}),
            accumulated_output: vec![call],
            tool_calls: vec![assignment],
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(4 * 1_048_576);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(b": heartbeat\n\n"));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let cache = ctx
            .extensions
            .get::<super::ResponseStoreRequestState>()
            .unwrap()
            .shared_stable_bytes
            .unwrap();
        assert!(cache.current_output_bytes > 512 * 1_024);
        assert!(
            cache.current_output_bytes < 1_048_576,
            "the assignment does not own another full call"
        );

        for _ in 0..200 {
            assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        }
        assert!(
            ctx.extensions
                .get::<super::ResponseStoreRequestState>()
                .unwrap()
                .shared_stable_bytes
                .unwrap()
                .matches(ctx.extensions.get::<ResponsesState>().unwrap())
        );
        // A deliberately impossible cached charge proves the repeated-chunk
        // admission reads this measurement instead of rescanning the JSON.
        ctx.extensions
            .get_mut::<super::ResponseStoreRequestState>()
            .unwrap()
            .shared_stable_bytes
            .as_mut()
            .unwrap()
            .current_output_bytes = usize::MAX;
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[test]
    #[expect(clippy::indexing_slicing, reason = "the fixture contains one known output item")]
    fn replay_meter_refreshes_same_length_completed_output_rewrite() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState {
            response_object: json!({"output": [{"text": "a".repeat(4_096)}]}),
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(16_384);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(b": heartbeat\n\n"));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let old_cache = ctx
            .extensions
            .get::<super::ResponseStoreRequestState>()
            .unwrap()
            .shared_stable_bytes
            .unwrap();
        let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        responses.response_object["output"][0]["text"] = json!("\u{0001}".repeat(4_096));
        responses.mark_current_output_changed();
        assert_eq!(responses.output_items().len(), 1);
        assert!(!old_cache.matches(responses));
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[test]
    fn replay_meter_reuses_history_when_only_current_output_changes() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState::from_request_body(json!({"input": "x".repeat(1_048_576)}));
        responses.apply_retained_payload_limit(8 * 1_048_576);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(b": heartbeat\n\n"));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let cached = ctx
            .extensions
            .get_mut::<super::ResponseStoreRequestState>()
            .unwrap()
            .shared_stable_bytes
            .as_mut()
            .unwrap();
        // A distinct, still-admissible baseline reveals an unwanted rescan of
        // the megabyte prompt while the response output alone changes.
        cached.bytes += 32;
        let charged_baseline = cached.bytes;
        let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        responses.response_object = json!({"output": [{"text": "delta"}]});
        responses.mark_current_output_changed();
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        let cached = ctx
            .extensions
            .get::<super::ResponseStoreRequestState>()
            .unwrap()
            .shared_stable_bytes
            .unwrap();
        assert_eq!(cached.bytes, charged_baseline);
    }

    #[test]
    fn replay_meter_fails_closed_when_revision_overflows() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState {
            replay_stable_payload_revision: Some(u64::MAX),
            ..ResponsesState::default()
        };
        responses.apply_retained_payload_limit(8_192);
        responses.mark_replay_stable_payload_changed();
        assert_eq!(responses.replay_stable_payload_revision, None);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(
            b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1}\n\n",
        ));
        assert!(!filter.capture_stream_events(&mut ctx, &frame, false));
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "checks request re-entry invalidation and a same-shape rewrite"
    )]
    async fn replay_meter_invalidates_same_length_request_rewrites_on_reentry() {
        let filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(1_024).unwrap(), NonZeroU64::new(1_048_576).unwrap());
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut responses = ResponsesState::from_request_body(json!({"input": "x".repeat(1_024)}));
        responses.apply_retained_payload_limit(8_192);
        ctx.extensions.insert(responses);
        let frame = Some(Bytes::from_static(
            b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1}\n\n",
        ));
        assert!(filter.capture_stream_events(&mut ctx, &frame, false));
        assert!(
            ctx.extensions
                .get::<super::ResponseStoreRequestState>()
                .unwrap()
                .shared_stable_bytes
                .is_some()
        );

        assert!(matches!(
            filter.on_request(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(
            ctx.extensions
                .get::<super::ResponseStoreRequestState>()
                .unwrap()
                .shared_stable_bytes
                .is_none(),
            "the next request phase may replace payloads without changing their vector lengths"
        );
        *ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .request_body
            .get_mut("input")
            .unwrap() = json!("y".repeat(8_192));
        assert!(
            !filter.capture_stream_events(&mut ctx, &frame, false),
            "a same-shape request rewrite must be measured before the next replay chunk"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "rehydrate and store must share a realistic response context"
    )]
    async fn budgeted_previous_response_completion_keeps_rehydrate_buffer() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let store = std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new());
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                store,
            )
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "false");
        let mut state = ResponsesState {
            history_rehydrated: true,
            previous_response_id: Some("resp_previous".to_owned()),
            response_object: json!({"id":"resp_new", "status":"completed", "output":[]}),
            ..Default::default()
        };
        state.apply_retained_payload_limit(67_108_864);
        ctx.extensions.insert(state);
        ctx.filter_results
            .entry("openai_agentic_loop")
            .or_default()
            .set("action", "done")
            .unwrap();
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .buffered_canonical_finalized = true;
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        ctx.response_header = Some(&mut response);
        let rehydrate = crate::openai::RehydrateFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
        drop(rehydrate.on_response(&mut ctx).await.unwrap());
        assert_eq!(
            ctx.response_body_mode,
            praxis_filter::body::BodyMode::StreamBuffer {
                max_bytes: Some(praxis_filter::body::MAX_JSON_BODY_BYTES)
            }
        );
        let store_filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(100).unwrap(), NonZeroU64::new(1_000_000).unwrap());
        assert!(matches!(
            store_filter.on_response(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the direct response crosses both rehydrate and store header filters"
    )]
    async fn budgeted_direct_stored_restore_uses_validated_framing() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "false");
        let mut state = ResponsesState::from_request_body(json!({"model":"m","input":"next","store":true}));
        state.history_rehydrated = true;
        state.previous_response_id = Some("resp_previous".to_owned());
        state.apply_retained_payload_limit(65_536);
        ctx.extensions.insert(state);
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, "300".parse().unwrap());
        ctx.response_header = Some(&mut response);
        let rehydrate = crate::openai::RehydrateFilter::from_config(&serde_yaml::from_str("{}").unwrap()).unwrap();
        assert!(matches!(
            rehydrate.on_response(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(
            ctx.response_header
                .as_ref()
                .unwrap()
                .headers
                .get(http::header::CONTENT_LENGTH)
                .is_none()
        );
        let store_filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(100).unwrap(), NonZeroU64::new(1_000_000).unwrap());
        assert!(matches!(
            store_filter.on_response(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(
            ctx.response_body_mode,
            praxis_filter::body::BodyMode::StreamBuffer { max_bytes: Some(300) }
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the final response guard needs a complete store context"
    )]
    async fn incomplete_canonical_body_needs_no_direct_content_length() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "false");
        ctx.set_metadata("openai_responses_format.has_conversation", "true");
        let mut state = ResponsesState {
            response_object: json!({"id":"resp_incomplete", "status":"incomplete", "output":[]}),
            ..ResponsesState::default()
        };
        state.apply_retained_payload_limit(16_384);
        state.buffered_canonical_finalized = true;
        ctx.extensions.insert(state);
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        ctx.response_header = Some(&mut response);
        let store_filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(100).unwrap(), NonZeroU64::new(1_000_000).unwrap());

        assert!(matches!(
            store_filter.on_response(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(crate::openai::responses::final_conversation_buffer_budget_rejection(&ctx).is_none());
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the complete response filter context is needed to test header admission"
    )]
    async fn noncanonical_budgeted_store_rejects_oversized_known_body_before_headers() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let registry = crate::store::ResponseStoreRegistry::new();
        registry
            .register(
                &std::sync::Arc::from(crate::openai::responses::DEFAULT_STORE_NAME),
                std::sync::Arc::new(praxis_ai_store::memory::InMemoryStore::new()),
            )
            .unwrap();
        ctx.extensions.insert(registry);
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "false");
        let mut state = ResponsesState::from_request_body(json!({"model":"m","input":"hello","store":true}));
        state.apply_retained_payload_limit(16_384);
        ctx.extensions.insert(state);
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_TYPE, "application/json".parse().unwrap());
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, "3232".parse().unwrap());
        ctx.response_header = Some(&mut response);
        let store_filter =
            ResponseStoreFilter::with_bounds(NonZeroU32::new(100).unwrap(), NonZeroU64::new(1_000_000).unwrap());

        let rejection = match store_filter.on_response(&mut ctx).await.unwrap() {
            FilterAction::Reject(rejection) => Some(rejection),
            _ => None,
        }
        .expect("store staging must reject before core commits the upstream success header");
        assert_eq!(rejection.status, 502);
        let error: serde_json::Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(
            error.pointer("/error/type").and_then(serde_json::Value::as_str),
            Some("server_error")
        );
    }

    #[test]
    fn noncanonical_header_reserves_multichunk_freeze_owner() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let wire_bytes = 100;
        let parsed_bytes = wire_bytes * 8;
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        let headroom = encoded_column_headroom(parsed_bytes, 0, 0).unwrap();
        state.apply_retained_payload_limit(baseline + parsed_bytes * 6 + headroom + wire_bytes + wire_bytes / 2);
        ctx.extensions.insert(state);
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, wire_bytes.to_string().parse().unwrap());
        ctx.response_header = Some(&mut response);

        assert!(super::persistence_construction_fits_with_wire(
            &ctx,
            parsed_bytes,
            wire_bytes
        ));
        assert_eq!(super::buffered_header_persistence_length(&ctx), None);
    }

    #[test]
    fn shortest_exponent_tokens_stay_within_header_json_bound() {
        let wire = format!("[{}]", vec!["1e0"; 4096].join(","));
        let parsed_bound = buffered_parsed_json_bytes_upper_bound(wire.as_bytes()).unwrap();
        assert!(parsed_bound <= wire.len() * 12);
    }

    #[test]
    fn negative_zero_header_bound_covers_body_scanner_before_commit() {
        let wire = format!("[{}]", vec!["-0"; 4096].join(","));
        let bytes = wire.as_bytes();
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context_without_subrequest_client(&request);
        let mut state = ResponsesState::default();
        let baseline = state.retained_payload_bytes().unwrap();
        let eightfold = bytes.len() * 8;
        let headroom = encoded_column_headroom(eightfold, 0, 0).unwrap();
        state.apply_retained_payload_limit(baseline + eightfold * 6 + headroom + bytes.len() * 2);
        ctx.extensions.insert(state);
        let mut response = crate::test_utils::make_response();
        response
            .headers
            .insert(http::header::CONTENT_LENGTH, bytes.len().to_string().parse().unwrap());
        ctx.response_header = Some(&mut response);

        assert!(!buffered_persistence_construction_fits(&ctx, bytes));
        assert_eq!(super::buffered_header_persistence_length(&ctx), None);
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn router_step_stream_selection_does_not_outlive_its_round() {
        let inner_step = super::ResponseRound {
            router: Some(0),
            agentic: Some(0),
        };
        let parent_stream = super::ResponseRound {
            router: None,
            agentic: Some(0),
        };
        assert!(!inner_step.is_outside_router());
        assert!(parent_stream.is_outside_router());
    }
}
