// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Filter 8: execute MCP tool calls against upstream MCP servers.
//!
//! Operates in two phases within an
//! `iterative_request_router` inference step:
//!
//! 1. **Response path** (`on_response_body`): after `openai_agentic_loop` extracts model-produced function calls,
//!    identifies calls backed by [`ResponsesState::mcp_tool_map`], checks approval policies, and writes
//!    `openai_mcp_dispatch.action = "loop"` to filter results.
//! 2. **Request-body path** (`on_request_body`, next IRR iteration): executes pending MCP calls via
//!    [`mcp_client::call_tool`] and appends results to `messages`, `persisted_messages`, and `output_items` before
//!    `openai_proxy` serializes the next inference request.
//!
//! # Pipeline dependencies
//!
//! - **`mcp_tool_resolve`** must run before this filter so that [`ResponsesState::mcp_tool_map`] is populated.
//! - **`openai_stream_events`** (or equivalent accumulator) must populate [`ResponsesState::tool_calls`] from the
//!   upstream response. Currently only `function_call` events are accumulated; native `mcp_call` events require either
//!   `mcp_tool_resolve` rewriting MCP tools into function tools or the accumulator adding `mcp_call` support.
//! - **`openai_agentic_loop`** must run after this filter in request order, so response order is `openai_agentic_loop`
//!   then `openai_mcp_dispatch`.
//! - The IRR transition must match `openai_mcp_dispatch.action = "loop"` and target the same inference step.
//!
//! Ordinary client-side function calls do not match the MCP tool map,
//! so this filter reports `done` and returns them to the client.
//!
//! [`ResponsesState::tool_calls`]: super::state::ResponsesState
//! [`ResponsesState::mcp_tool_map`]: super::state::ResponsesState
//! [`filter_results`]: HttpFilterContext::filter_results

pub(crate) mod approval;
mod config;

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

use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{FutureExt as _, future::join_all};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, SubRequestResponseMode,
    body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use tracing::{debug, warn};

use self::{
    approval::{
        ApprovalError, ResolvedApproval, build_approved_tool_call, build_denial_message, extract_approval_responses,
        is_approval_response, parse_approval_policy, parse_approval_response, requires_approval, resolve_approval,
        target_fingerprint,
    },
    config::{MIN_RETAINED_RESULT_BYTES, McpDispatchConfig, build_config},
};
use super::{
    DEFAULT_STORE_NAME, DEFAULT_TENANT_ID, TENANT_METADATA_KEY,
    error::responses_error_rejection,
    mcp_tool_resolve::{McpToolIndex, McpToolMatch},
    state::{McpApprovalState, ResponsesState, current_round_borrowed_tool_call_admissions},
};
use crate::{
    json_body::serialized_len,
    mcp_client,
    store::{PendingApprovalRecord, ResponseStore, ResponseStoreRegistry},
};

/// Filter result key consumed by `iterative_request_router`.
pub(super) const FILTER_RESULT_KEY: &str = "openai_mcp_dispatch";

/// Continue with another inference iteration after MCP execution.
const ACTION_LOOP: &str = "loop";

/// Return the current model response to the client.
const ACTION_DONE: &str = "done";

/// Maximum `mcp_approval_response` items accepted in one resume request.
///
/// A round may emit several `mcp_approval_request` items (batched or parallel
/// tool calls), each persisted as its own server-owned pending record. Those are
/// resumed one per follow-up request: a resume turn legitimately carries a single
/// approval. Capping the batch bounds the fail-closed work per request and keeps
/// the server-owned consume query within `PostgreSQL`'s 16-bit Bind parameter
/// ceiling (it binds two scoping params plus one per approval id), which an
/// unbounded batch could otherwise overflow into an HTTP 500.
const MAX_APPROVAL_RESPONSES: usize = 1;

// -----------------------------------------------------------------------------
// McpDispatchFilter
// -----------------------------------------------------------------------------

/// Executes MCP tool calls against upstream MCP servers within
/// the Responses API agentic loop.
///
/// # YAML
///
/// ```yaml
/// filter: openai_mcp_dispatch
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_mcp_dispatch
/// timeout_ms: 30000
/// max_calls_per_round: 32
/// max_parallel_calls: 8
/// max_result_bytes: 1048576
/// max_total_result_bytes: 8388608
/// ```
pub struct McpDispatchFilter {
    /// Allow connections to loopback addresses.
    allow_loopback: bool,
    /// Timeout for MCP tool calls.
    timeout: Duration,
    /// Hard cap on calls accepted from one model round.
    max_calls_per_round: usize,
    /// Maximum number of concurrent calls.
    max_parallel_calls: usize,
    /// Maximum retained bytes for one call result.
    max_result_bytes: usize,
    /// Maximum retained bytes across one call batch.
    max_total_result_bytes: usize,
}

impl McpDispatchFilter {
    /// Build from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpDispatchConfig = parse_filter_config("openai_mcp_dispatch", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self {
            allow_loopback: validated.allow_loopback,
            timeout: Duration::from_millis(validated.timeout_ms),
            max_calls_per_round: validated.max_calls_per_round,
            max_parallel_calls: validated.max_parallel_calls,
            max_result_bytes: validated.max_result_bytes,
            max_total_result_bytes: validated.max_total_result_bytes,
        }))
    }

    /// Handle tool calls that require approval.
    ///
    /// Every approval round trip is server-owned: each pending call is persisted
    /// as a [`PendingApprovalRecord`] so the mandatory `mcp_approval_response`
    /// follow-up can correlate back to it via `previous_response_id`. Fails closed
    /// before any side effect — including sibling ungated execution — if the round
    /// could never be resumed: the client opted out of storage (`store=false`) or
    /// no store armed persistence for this exchange. Otherwise it records and emits
    /// the client-visible `mcp_approval_request` items, then either runs the ungated
    /// siblings in the next round or returns the pending response to the client.
    fn handle_approval_required(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        pending: Vec<PendingApproval>,
        ungated: Vec<serde_json::Value>,
    ) -> Result<FilterAction, FilterError> {
        debug!(count = pending.len(), "MCP tool calls require approval");

        if ctx.extensions.get::<ResponsesState>().is_none() {
            warn!("ResponsesState missing when handling approval");
            return Ok(FilterAction::Continue);
        }

        // An approval round trip is only resumable if its pending records are
        // persisted: the mandatory follow-up correlates each mcp_approval_response
        // back to the server-owned record via previous_response_id. Fail closed
        // before any side effect — including sibling ungated execution — if it
        // could never be resumed (the client opted out of storage or no store
        // backend armed persistence) instead of stranding the client with an
        // mcp_approval_request that can never be resumed.
        let tool_name = pending.first().map_or("", |p| p.tool_name.as_str());
        if let Some(rejection) = approval_persistence_rejection(ctx, tool_name) {
            return Self::reject_unresumable_approval(ctx, rejection);
        }

        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        record_and_emit_approvals(state, pending);

        // Approval requests are server-owned. Remove every MCP call before any
        // sibling dispatcher can request another inference round; otherwise a
        // web-search continuation could re-enter request dispatch and execute
        // an approval-gated call.
        let tool_index = McpToolIndex::new(&state.mcp_tool_map);
        state.tool_calls.retain(|call| !is_mcp_tool_call(call, &tool_index));

        if !ungated.is_empty() {
            state.tool_calls.extend(ungated);
            state.mcp_approval_state = McpApprovalState::ExecuteUngatedThenReturn;
            ctx.set_metadata("openai_mcp_dispatch.action", "execute_mcp");
            set_action(ctx, ACTION_LOOP)?;
            return Ok(FilterAction::Continue);
        }

        if !state.web_search_calls.is_empty() {
            state.mcp_approval_state = McpApprovalState::ApprovalPendingThenReturn;
        }

        state.finalize_response_body(body);
        ctx.set_metadata("openai_mcp_dispatch.action", "done");
        set_action(ctx, ACTION_DONE)?;

        Ok(FilterAction::Continue)
    }

    /// Fail closed on an approval-required round that could never be resumed.
    ///
    /// A buffered sub-request has not committed a response yet, so it rejects
    /// pre-commitment with the JSON `{"error":{...}}` envelope. A streaming
    /// sub-request has already put SSE headers and model events on the wire, so a
    /// `FilterAction::Reject` would be converted into a transport error *after* a
    /// truncated success — the explanation would be lost. In that case hand the
    /// terminal error to the `openai_stream_events` logical-stream finalizer via
    /// the committed-stream metadata and skip persisting the committed response,
    /// mirroring the agentic loop's committed-stream error path. Either way no
    /// unresumable `mcp_approval_request` is emitted.
    fn reject_unresumable_approval(
        ctx: &mut HttpFilterContext<'_>,
        rejection: ApprovalRejection,
    ) -> Result<FilterAction, FilterError> {
        if ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming {
            ctx.set_metadata("responses.stream_error_code".to_owned(), rejection.code.to_owned());
            ctx.set_metadata("responses.stream_error_message".to_owned(), rejection.message);
            ctx.set_metadata("responses.skip_persist".to_owned(), "true".to_owned());
            ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "done".to_owned());
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        Ok(FilterAction::Reject(responses_error_rejection(
            rejection.status,
            rejection.code,
            &rejection.message,
        )))
    }

    /// Terminate a round whose MCP batch exceeds its configured hard cap.
    fn oversized_response_action(
        ctx: &mut HttpFilterContext<'_>,
        streaming: bool,
    ) -> Result<FilterAction, FilterError> {
        const MESSAGE: &str = "model response exceeded the configured MCP call limit";
        if streaming {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.tool_calls.clear();
                state.web_search_calls.clear();
            }
            ctx.set_metadata("responses.stream_error_code", "server_error");
            ctx.set_metadata("responses.stream_error_message", MESSAGE);
            ctx.set_metadata("responses.skip_persist", "true");
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        Ok(FilterAction::Reject(responses_error_rejection(
            502,
            "server_error",
            MESSAGE,
        )))
    }

    /// Terminate a malformed batch before any MCP side effect can occur.
    fn invalid_call_ids_response_action(
        ctx: &mut HttpFilterContext<'_>,
        streaming: bool,
    ) -> Result<FilterAction, FilterError> {
        const MESSAGE: &str = "model response contained duplicate or missing MCP call_id values";
        if streaming {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.tool_calls.clear();
                state.web_search_calls.clear();
            }
            ctx.set_metadata("responses.stream_error_code", "server_error");
            ctx.set_metadata("responses.stream_error_message", MESSAGE);
            ctx.set_metadata("responses.skip_persist", "true");
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        Ok(FilterAction::Reject(responses_error_rejection(
            502,
            "server_error",
            MESSAGE,
        )))
    }

    /// Execute as many pending MCP calls as the remaining request budget allows.
    #[expect(
        clippy::too_many_lines,
        reason = "budget classification and bounded result assembly stay atomic"
    )]
    async fn execute_pending_calls(
        &self,
        state: &ResponsesState,
        mcp_calls: &[&serde_json::Value],
        tool_index: &McpToolIndex<'_>,
    ) -> Result<McpBatchResult, McpResultLimitExceeded> {
        let execute_count = state.max_tool_calls.map_or(mcp_calls.len(), |_| {
            let admissions = current_round_borrowed_tool_call_admissions(state, mcp_calls, |item| {
                is_mcp_tool_call(item, tool_index)
            });
            // Admissions are a prefix: the shared remaining budget only
            // decreases as model calls are visited in provider order.
            debug_assert!(
                !admissions
                    .iter()
                    .copied()
                    .skip_while(|admitted| *admitted)
                    .any(|admitted| admitted),
                "tool-call admissions must remain monotonic"
            );
            admissions.iter().take_while(|admitted| **admitted).count()
        });
        let (executable, rejected) = mcp_calls.split_at(execute_count);
        debug!(
            count = executable.len(),
            rejected = rejected.len(),
            parallel = state.parallel_tool_calls,
            "executing pending MCP tool calls"
        );
        let (per_result_limit, execution_batch_limit) = admitted_result_limits(
            executable.len(),
            rejected.len(),
            self.max_result_bytes,
            self.max_total_result_bytes,
        )?;
        let options = McpExecutionOptions {
            parallel: state.parallel_tool_calls,
            max_parallel_calls: self.max_parallel_calls,
            max_result_bytes: per_result_limit,
            max_total_result_bytes: execution_batch_limit,
            timeout: self.timeout,
            allow_loopback: self.allow_loopback,
        };
        let mut results = execute_mcp_calls(executable, tool_index, options).await?;
        let mut retained_bytes = retained_results_bytes(&results)?;
        for &call in rejected {
            push_result_within_budget(
                &mut results,
                &mut retained_bytes,
                fit_result_or_limit_error(
                    call,
                    error_result_for_dropped_call(
                        call,
                        "MCP call was not executed because max_tool_calls was exhausted",
                    ),
                    MIN_RETAINED_RESULT_BYTES,
                ),
                MIN_RETAINED_RESULT_BYTES,
                self.max_total_result_bytes,
            )?;
        }
        Ok(McpBatchResult {
            results,
            tool_limit_exceeded: !rejected.is_empty(),
        })
    }

    /// Append MCP execution results to private continuation and public output state.
    fn append_results(ctx: &mut HttpFilterContext<'_>, results: Vec<McpCallResult>) {
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            warn!("ResponsesState missing when appending results");
            return;
        };
        for result in results {
            state.messages.push(result.message.clone());
            state.persisted_messages.push(result.message);
            // Record execution provenance keyed on the item id `stream_events`
            // reads, so only this locally executed `mcp_call` gains a synthesized
            // lifecycle.
            if let Some(id) = result.output_item.get("id").and_then(serde_json::Value::as_str) {
                state.locally_executed_output_items.insert(id.to_owned());
            }
            state.accumulated_output.push(result.output_item);
        }
        let tool_index = McpToolIndex::new(&state.mcp_tool_map);
        state.tool_calls.retain(|call| !is_mcp_tool_call(call, &tool_index));
    }

    /// Record a response-wide tool-limit cutoff for terminalization by the loop.
    fn record_deferred_completion(ctx: &mut HttpFilterContext<'_>, tool_limit_exceeded: bool) {
        if tool_limit_exceeded && let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.mcp_approval_state = McpApprovalState::ToolLimitExceededThenReturn;
        }
    }

    /// Terminate without retaining a batch that exceeded its result-byte cap.
    #[expect(
        clippy::too_many_lines,
        reason = "streaming and buffered terminal responses share state cleanup"
    )]
    fn result_limit_action(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        const MESSAGE: &str = "MCP result batch exceeded the configured retained-byte limit";
        let mut sequence_number = 0;
        let streaming = ctx.extensions.get_mut::<ResponsesState>().is_some_and(|state| {
            state.tool_calls.clear();
            state.web_search_calls.clear();
            let streaming = state.request_body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
            if streaming {
                sequence_number = state.logical_stream_sequence;
                state.logical_stream_sequence = state.logical_stream_sequence.saturating_add(1);
            }
            streaming
        });
        ctx.set_metadata("responses.skip_persist", "true");
        set_action(ctx, ACTION_DONE)?;
        if streaming {
            return Ok(FilterAction::Reject(
                Rejection::status(200)
                    .with_header("content-type", "text/event-stream")
                    .with_body(super::error::responses_error_sse_body_at_sequence(
                        "server_error",
                        MESSAGE,
                        sequence_number,
                    ))
                    .preserving_keepalive(),
            ));
        }
        Ok(FilterAction::Reject(responses_error_rejection(
            502,
            "server_error",
            MESSAGE,
        )))
    }

    /// Resume any pending approvals carried by the current request input.
    ///
    /// Parses each `mcp_approval_response`, correlates it to the server-owned
    /// pending record the proxy wrote when it emitted the request, binds it to a
    /// unique current tool-map target (target identity, not just arguments),
    /// atomically claims single-use consumption, and then either injects an
    /// approved tool call or appends a denial `function_call_output`. Consent
    /// provenance comes only from the pending store, never from the
    /// (client-influenced) conversation history, so a forged or persisted
    /// `mcp_approval_request` has no matching record and fails closed. Also
    /// fails closed on any malformed, unknown, stale, replayed, or mismatched
    /// approval.
    #[expect(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "five borrow-scoped phases: parse, load, resolve, consume, apply"
    )]
    async fn resume_approvals(&self, ctx: &mut HttpFilterContext<'_>) -> Result<(), Rejection> {
        // Phase 0: parse the client-supplied approval responses and capture the
        // response that issued them. Only the correlation id, verdict, and
        // reason are trusted from the client; the pending call itself is looked
        // up server-side below, scoped to the issuing response.
        let (inputs, previous_response_id) = {
            let Some(state) = ctx.extensions.get::<ResponsesState>() else {
                return Ok(());
            };
            if state.iteration != 0 {
                return Ok(());
            }
            let responses = extract_approval_responses(&state.messages);
            if responses.is_empty() {
                return Ok(());
            }
            // Bound the batch before any store work. Each round's approvals are
            // resumed one per follow-up request, so a resume carries a single
            // approval; a larger batch is a client error and, left unbounded,
            // could overflow PostgreSQL's 16-bit Bind parameter ceiling in the
            // consume query.
            if responses.len() > MAX_APPROVAL_RESPONSES {
                let e = ApprovalError::Malformed(format!(
                    "a request may carry at most {MAX_APPROVAL_RESPONSES} mcp_approval_response item(s), but {} were supplied",
                    responses.len()
                ));
                warn!(error = %e.message(), "mcp_dispatch: rejecting oversized approval-response batch");
                return Err(approval_rejection(&e));
            }
            // A pending approval is bound to the response that issued it. Without
            // the originating previous_response_id the proxy cannot scope the
            // lookup to that response, so the approval fails closed rather than
            // letting a fresh, unrelated request claim a known outstanding
            // approval.
            let Some(previous_response_id) = state.previous_response_id.clone() else {
                let e = ApprovalError::Malformed(
                    "an mcp_approval_response requires previous_response_id to identify the originating request"
                        .to_owned(),
                );
                warn!(error = %e.message(), "mcp_dispatch: rejecting approval response without previous_response_id");
                return Err(approval_rejection(&e));
            };
            let mut inputs = Vec::with_capacity(responses.len());
            for response in responses {
                match parse_approval_response(response) {
                    Ok(input) => inputs.push(input),
                    Err(e) => {
                        warn!(error = %e.message(), "mcp_dispatch: rejecting malformed approval response");
                        return Err(approval_rejection(&e));
                    },
                }
            }
            (inputs, previous_response_id)
        };

        // Phase 1: load the server-owned pending records issued by
        // previous_response_id. History is never trusted for correlation: a
        // client-forged mcp_approval_request (whether inlined into the request or
        // persisted into the trace on a prior turn) has no matching pending row
        // and is therefore invisible here, and an approval issued by a different
        // response is out of scope under this previous_response_id.
        let tenant_id = ctx
            .get_metadata(TENANT_METADATA_KEY)
            .unwrap_or(DEFAULT_TENANT_ID)
            .to_owned();
        let store = ctx
            .extensions
            .get::<ResponseStoreRegistry>()
            .and_then(|registry| registry.get(DEFAULT_STORE_NAME))
            .ok_or_else(|| {
                warn!("mcp_dispatch: response store unavailable while resuming approvals");
                responses_error_rejection(500, "server_error", "response store is not available")
            })?;
        let approval_ids: Vec<&str> = inputs.iter().map(|i| i.approval_id.as_str()).collect();
        let pending_records = store
            .get_pending_approvals(&tenant_id, &previous_response_id, &approval_ids)
            .await
            .map_err(|e| {
                warn!(error = %e, "mcp_dispatch: failed to load pending approvals");
                responses_error_rejection(500, "server_error", "failed to load pending approvals")
            })?;

        // Phase 2: correlate each response to its pending record and bind it to a
        // unique current tool-map target. A response without a matching pending
        // record — unknown, stale, or forged — fails closed before any side effect.
        let resolved = {
            let Some(state) = ctx.extensions.get::<ResponsesState>() else {
                return Ok(());
            };
            let pending_by_id: HashMap<&str, &PendingApprovalRecord> =
                pending_records.iter().map(|r| (r.approval_id.as_str(), r)).collect();
            let mut resolved = Vec::with_capacity(inputs.len());
            for input in &inputs {
                let Some(&pending) = pending_by_id.get(input.approval_id.as_str()) else {
                    let e = ApprovalError::UnknownApprovalId(format!(
                        "no pending approval request matches id '{}'",
                        input.approval_id
                    ));
                    warn!(error = %e.message(), "mcp_dispatch: rejecting unknown approval response");
                    return Err(approval_rejection(&e));
                };
                match resolve_approval(input, pending, &state.mcp_tool_map) {
                    Ok(decision) => resolved.push(decision),
                    Err(e) => {
                        warn!(error = %e.message(), "mcp_dispatch: rejecting unresolvable approval response");
                        return Err(approval_rejection(&e));
                    },
                }
            }
            resolved
        };

        // Phase 3: atomically claim single-use consumption for the whole batch.
        // Every id here has a pending row, so a failed transition means the
        // approval was already consumed (replay) rather than never issued.
        let consumed_at = i64::try_from(ctx.time_source.now().as_millis()).unwrap_or(i64::MAX);
        let claim_ids: Vec<&str> = resolved.iter().map(|d| d.approval_id.as_str()).collect();
        consume_batch(
            store.as_ref(),
            &tenant_id,
            &previous_response_id,
            &claim_ids,
            consumed_at,
        )
        .await?;

        // Phase 4: apply the decisions to request-scoped state.
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(());
        };
        // Approval responses are proxy-level control items: keep them in the
        // persisted trace but strip them from backend-bound messages.
        state.messages.retain(|m| !is_approval_response(m));
        for decision in &resolved {
            apply_decision(state, decision);
        }
        Ok(())
    }
}

/// Atomically claim single-use consumption of every approval in the batch,
/// failing closed on replay, store error, or missing backend.
///
/// The claim is all-or-nothing: a single replayed or duplicated approval
/// rejects the whole batch without consuming any id, so a corrected retry can
/// still resume the legitimately approved calls.
async fn consume_batch(
    store: &dyn ResponseStore,
    tenant_id: &str,
    response_id: &str,
    approval_ids: &[&str],
    consumed_at: i64,
) -> Result<(), Rejection> {
    match store
        .consume_approvals(tenant_id, response_id, approval_ids, consumed_at)
        .await
    {
        Ok(None) => Ok(()),
        Ok(Some(index)) => {
            let approval_id = approval_ids.get(index).copied().unwrap_or_default();
            warn!(approval_id, "mcp_dispatch: approval already consumed; rejecting replay");
            Err(responses_error_rejection(
                400,
                "invalid_request_error",
                &format!("approval '{approval_id}' has already been used"),
            ))
        },
        Err(e) => {
            warn!(error = %e, "mcp_dispatch: failed to claim approval consumption");
            Err(responses_error_rejection(
                500,
                "server_error",
                "failed to record approval consumption",
            ))
        },
    }
}

/// Map an [`ApprovalError`] to a fail-closed `400 invalid_request_error`.
fn approval_rejection(error: &ApprovalError) -> Rejection {
    responses_error_rejection(400, "invalid_request_error", error.message())
}

/// Record each server-owned pending approval and emit its client-visible
/// `mcp_approval_request` into `accumulated_output`.
///
/// Each record captures the resolved target fingerprint — the sole source of
/// truth for a later `mcp_approval_response` — which is deliberately NOT echoed
/// on the client-visible event so a client cannot reproduce it. The records are
/// drained and persisted by the store filter so the resume turn can correlate a
/// response back to this proxy-issued request. Execution provenance is recorded
/// so `stream_events` may synthesize each approval item's lifecycle; a bare
/// `accumulated_output` push is not proof that this filter produced the item.
fn record_and_emit_approvals(state: &mut ResponsesState, pending: Vec<PendingApproval>) {
    for call in pending {
        let record = PendingApprovalRecord {
            approval_id: call.call_id,
            server_label: call.server_label,
            tool_name: call.tool_name,
            arguments: call.arguments,
            target_fingerprint: call.target_fingerprint,
        };
        state.locally_executed_output_items.insert(record.approval_id.clone());
        state.accumulated_output.push(serde_json::json!({
            "type": "mcp_approval_request",
            "id": record.approval_id,
            "name": record.tool_name,
            "server_label": record.server_label,
            "arguments": record.arguments,
        }));
        state.pending_approvals.push(record);
    }
}

/// Fail-closed rejection when an approval-required call could never be resumed.
///
/// The mandatory `mcp_approval_response` follow-up correlates back to a
/// server-owned pending record via `previous_response_id`. That record is
/// written only when this response is persisted, which requires BOTH the client
/// opting into storage (`store`, default `true`) AND the store filter having
/// armed persistence for THIS exchange. Either gap means the emitted
/// `mcp_approval_request` could never be resumed, so fail closed before emitting.
/// Returns `None` when the approval is resumable.
///
/// The armed marker (`ResponsesState::store_persist_armed`) is exchange-scoped,
/// so — unlike pipeline-scoped registry membership — it also rejects when the
/// store filter is absent, request-conditioned out, or ordered after this
/// dispatch filter. It is set during the request phase and therefore cannot
/// observe a response-phase persistence skip: a store filter gated by
/// `response_conditions` arms during the request but then skips persisting the
/// response. Composing a conditional store filter into an approval pipeline is
/// therefore unsupported. That narrower residual is not eliminated here, but it
/// still fails closed at resume (an unresumable approval yields a clean error,
/// never an executed unapproved tool) and requires a self-contradictory
/// configuration — conditioning the store to drop the very response it must
/// persist. Rejecting such a composition at pipeline-build time is a Praxis-level
/// follow-up.
///
/// The client-controlled `store=false` is a `400`; a store not armed to persist
/// this response is a server-configuration `500`, mirroring the resume path's
/// identical guard.
fn approval_persistence_rejection(ctx: &HttpFilterContext<'_>, tool_name: &str) -> Option<ApprovalRejection> {
    let state = ctx.extensions.get::<ResponsesState>();
    if state.is_some_and(|state| !response_will_be_stored(state)) {
        warn!(
            tool_name = %tool_name,
            "mcp_dispatch: rejecting approval-required call because store=false makes it unresumable"
        );
        return Some(approval_requires_store_rejection(tool_name));
    }
    if !state.is_some_and(|state| state.store_persist_armed) {
        warn!(
            tool_name = %tool_name,
            "mcp_dispatch: rejecting approval-required call because the response store did not arm \
             persistence for this exchange"
        );
        return Some(approval_store_unavailable_rejection(tool_name));
    }
    None
}

/// A fail-closed approval rejection as raw parts, so the caller can surface it as
/// a pre-commitment JSON envelope (buffered) or a committed-stream SSE error
/// (streaming) without re-deriving the status, code, and message.
struct ApprovalRejection {
    /// HTTP status for the pre-commitment JSON envelope. Ignored on a committed
    /// stream, where headers are already sent and the error is an SSE event.
    status: u16,
    /// Machine-readable error code, shared by both surfaces.
    code: &'static str,
    /// Human-readable explanation, shared by both surfaces.
    message: String,
}

/// Whether the client opted into persistence for this Responses request.
///
/// OpenAI's `store` field defaults to `true`; only an explicit `store: false`
/// disables persistence. This mirrors how the store filter reads the same flag
/// (fail-open toward persistence), so it catches the one persistence decision a
/// client controls directly. It does not re-check the store filter's other
/// preconditions (a success status, the expected content type, record
/// completeness, store availability); those guard against a non-conforming
/// backend, not client intent, and a failure there still fails closed at resume
/// time with a 400.
fn response_will_be_stored(state: &ResponsesState) -> bool {
    state
        .request_body
        .get("store")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true)
}

/// Fail-closed `400` for an approval-required call on a non-persisted response.
///
/// Combining `require_approval` with `store: false` is a client error: the
/// mandatory `mcp_approval_response` follow-up correlates to server-owned state
/// via `previous_response_id`, which only exists when the response is stored.
fn approval_requires_store_rejection(tool_name: &str) -> ApprovalRejection {
    ApprovalRejection {
        status: 400,
        code: "invalid_request_error",
        message: format!(
            "MCP tool '{tool_name}' requires approval, but this request set store=false; approvals require \
             store=true so the mcp_approval_response follow-up can resume via previous_response_id"
        ),
    }
}

/// Fail-closed `500` for an approval-required call when the response store did
/// not arm persistence for this exchange.
///
/// Unlike `store: false` (a client choice → `400`), an unarmed store is a
/// server-side configuration gap: the request is well-formed but no store filter
/// will persist this response's pending approval — because the store filter is
/// absent, request-conditioned out, or ordered after dispatch — so the mandatory
/// follow-up could never resume. This mirrors the resume path, which returns the
/// same server error when the store is unavailable.
fn approval_store_unavailable_rejection(tool_name: &str) -> ApprovalRejection {
    ApprovalRejection {
        status: 500,
        code: "server_error",
        message: format!(
            "MCP tool '{tool_name}' requires approval, but no response store is configured to persist the \
             pending approval; a store is required so the mcp_approval_response follow-up can resume"
        ),
    }
}

/// Apply one resolved approval decision to request-scoped state.
///
/// Approval injects a function-call-shaped tool call for the dispatch path to
/// execute; denial appends a truthful `function_call_output` so inference
/// resumes without a tool call.
fn apply_decision(state: &mut ResponsesState, decision: &ResolvedApproval) {
    if decision.approve {
        debug!(
            approval_id = %decision.approval_id,
            tool_name = %decision.tool_name,
            "resuming approved MCP tool call"
        );
        state.tool_calls.push(build_approved_tool_call(decision));
    } else {
        debug!(approval_id = %decision.approval_id, "recording denied MCP approval");
        let denial = build_denial_message(&decision.approval_id, decision.reason.as_deref());
        state.messages.push(denial.clone());
        state.persisted_messages.push(denial);
    }
}

#[async_trait]
impl HttpFilter for McpDispatchFilter {
    fn name(&self) -> &'static str {
        "openai_mcp_dispatch"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        // Accept up to the absolute ceiling; the pipeline's body_limits
        // decides the real raw cap. This filter only reads the body.
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "approval resume and bounded MCP dispatch are one request-phase lifecycle"
    )]
    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        // Resume approvals from the previous turn before executing any calls.
        // On approval this injects a function-call-shaped tool call that the
        // dispatch machinery below runs; on denial it appends a
        // `function_call_output` so inference resumes without a tool call.
        if let Err(rejection) = self.resume_approvals(ctx).await {
            return Ok(FilterAction::Reject(rejection));
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        if state.tool_calls.is_empty() || state.mcp_tool_map.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let tool_index = McpToolIndex::new(&state.mcp_tool_map);
        let mcp_call_count = count_mcp_tool_calls(&state.tool_calls, &tool_index);
        if mcp_call_count > self.max_calls_per_round {
            return Ok(FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                "model response exceeded the configured MCP call limit",
            )));
        }
        let mcp_calls = extract_mcp_tool_calls(&state.tool_calls, &tool_index);
        if mcp_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let result = match self.execute_pending_calls(state, &mcp_calls, &tool_index).await {
            Ok(result) => result,
            Err(_limit) => return Self::result_limit_action(ctx),
        };
        Self::append_results(ctx, result.results);
        Self::record_deferred_completion(ctx, result.tool_limit_exceeded);

        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "quota admission and approval partitioning must share model order"
    )]
    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        if state.tool_calls.is_empty() || state.mcp_tool_map.is_empty() {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }

        let tool_index = McpToolIndex::new(&state.mcp_tool_map);
        let mcp_call_count = count_mcp_tool_calls(&state.tool_calls, &tool_index);
        if mcp_call_count > self.max_calls_per_round {
            let streaming = state.request_body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
            return Self::oversized_response_action(ctx, streaming);
        }
        let mcp_calls = extract_mcp_tool_calls(&state.tool_calls, &tool_index);
        if mcp_calls.is_empty() {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        if !mcp_call_ids_are_unique_and_new(&mcp_calls, &state.accumulated_output) {
            let streaming = state.request_body.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
            return Self::invalid_call_ids_response_action(ctx, streaming);
        }

        let admissions =
            current_round_borrowed_tool_call_admissions(state, &mcp_calls, |item| is_mcp_tool_call(item, &tool_index));
        let mut pending = Vec::new();
        let mut ungated_or_rejected = Vec::new();
        for (index, call) in mcp_calls.into_iter().enumerate() {
            if admissions.get(index) == Some(&false) {
                ungated_or_rejected.push(call);
            } else if let Some(approval) = check_single_approval(call, &tool_index) {
                pending.push(approval);
            } else {
                ungated_or_rejected.push(call);
            }
        }
        if !pending.is_empty() {
            // These siblings must survive removal of the original MCP calls
            // while the server-owned approval request pauses the round.
            let retained_calls = ungated_or_rejected.into_iter().cloned().collect();
            return Self::handle_approval_required(ctx, body, pending, retained_calls);
        }

        ctx.set_metadata("openai_mcp_dispatch.action".to_owned(), "execute_mcp".to_owned());
        set_action(ctx, ACTION_LOOP)?;

        Ok(FilterAction::Continue)
    }
}

/// Publish the dispatch decision for IRR transition evaluation.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_RESULT_KEY)
        .or_default()
        .set("action", action)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Approval Handling
// -----------------------------------------------------------------------------

/// Info about a tool call that requires approval.
struct PendingApproval {
    /// Call ID.
    call_id: String,
    /// Server label.
    server_label: String,
    /// Tool name.
    tool_name: String,
    /// Tool arguments as JSON string.
    arguments: String,
    /// Fingerprint of the resolved target (URL, headers, authorization,
    /// connector) captured at approval time. Recorded on the
    /// `mcp_approval_request` so the resume turn can reject a redirected target.
    target_fingerprint: String,
}

// -----------------------------------------------------------------------------
// MCP Tool Call Identification
// -----------------------------------------------------------------------------

/// Borrow the MCP tool calls from the `tool_calls` list by checking
/// `mcp_tool_map`.
fn extract_mcp_tool_calls<'a>(
    tool_calls: &'a [serde_json::Value],
    tool_index: &McpToolIndex<'_>,
) -> Vec<&'a serde_json::Value> {
    tool_calls
        .iter()
        .filter(|tc| is_mcp_tool_call(tc, tool_index))
        .collect()
}

/// Count MCP-owned function calls without cloning provider payloads.
fn count_mcp_tool_calls(tool_calls: &[serde_json::Value], tool_index: &McpToolIndex<'_>) -> usize {
    tool_calls.iter().filter(|tc| is_mcp_tool_call(tc, tool_index)).count()
}

/// Require every MCP function call to carry a distinct non-empty correlation ID.
///
/// Results are matched and response-wide budgets are deduplicated by `call_id`,
/// so accepting a missing or repeated value would make separately executed
/// side effects indistinguishable.
fn mcp_call_ids_are_unique_and_new(
    tool_calls: &[&serde_json::Value],
    accumulated_output: &[serde_json::Value],
) -> bool {
    let mut seen = HashSet::with_capacity(tool_calls.len());
    tool_calls.iter().all(|call| {
        call.get("call_id")
            .and_then(serde_json::Value::as_str)
            .filter(|call_id| !call_id.is_empty())
            .is_some_and(|call_id| {
                seen.insert(call_id)
                    && !accumulated_output.iter().any(|item| {
                        item.get("type").and_then(serde_json::Value::as_str) == Some("mcp_call")
                            && item.get("id").and_then(serde_json::Value::as_str) == Some(call_id)
                    })
            })
    })
}

/// Check whether a tool call is an MCP tool call by matching the
/// encoded function name against the tool map.
fn is_mcp_tool_call(tool_call: &serde_json::Value, tool_index: &McpToolIndex<'_>) -> bool {
    tool_call
        .get("name")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|name| tool_index.contains(name))
}

/// Find an entry in the tool map by encoded function name.
///
/// Returns the unique match from a precomputed reverse index. Ambiguous lossy
/// encodings fail closed rather than routing nondeterministically.
#[cfg(test)]
fn find_by_encoded_name<'a>(
    tool_index: &McpToolIndex<'a>,
    encoded_name: &str,
) -> Option<(&'a (String, String), &'a serde_json::Value)> {
    match tool_index.get(encoded_name)? {
        McpToolMatch::Unique { key, entry } => Some((key, entry)),
        McpToolMatch::Ambiguous { count } => {
            warn!(
                encoded_name,
                server_count = count,
                "multiple entries produce the same encoded tool name"
            );
            None
        },
    }
}

// -----------------------------------------------------------------------------
// Approval Pre-check
// -----------------------------------------------------------------------------

/// Collect every MCP tool call that requires approval. The complete batch is
/// returned to the client before any member executes, so one approval-gated
/// call cannot race with a sibling side effect.
#[cfg(test)]
fn partition_calls_by_approval(
    mcp_calls: Vec<&serde_json::Value>,
    tool_map: &HashMap<(String, String), serde_json::Value>,
) -> (Vec<PendingApproval>, Vec<serde_json::Value>) {
    let tool_index = McpToolIndex::new(tool_map);
    let mut pending = Vec::new();
    let mut ungated = Vec::new();
    for call in mcp_calls {
        if let Some(approval) = check_single_approval(call, &tool_index) {
            pending.push(approval);
        } else {
            ungated.push((*call).clone());
        }
    }
    (pending, ungated)
}

/// Extract the call ID from a tool call value.
fn extract_call_id(tc: &serde_json::Value) -> String {
    tc.get("call_id")
        .or_else(|| tc.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned()
}

/// Normalize raw tool call arguments.
///
/// Function calls carry arguments either as a JSON string
/// (e.g., `"{\"city\":\"Paris\"}"`) or directly as a JSON value.
/// Returns `(parsed_value, canonical_string)` or an error for
/// malformed JSON strings.
fn normalize_arguments(raw: &serde_json::Value) -> Result<(serde_json::Value, String), String> {
    match raw {
        serde_json::Value::String(s) => {
            let parsed = serde_json::from_str(s).map_err(|e| format!("malformed tool arguments: {e}"))?;
            Ok((parsed, s.clone()))
        },
        other => Ok((other.clone(), other.to_string())),
    }
}

/// Extract serialised arguments from a tool call value.
///
/// Uses the same string-vs-non-string convention as
/// [`normalize_arguments`] but only produces the canonical string,
/// avoiding the deep clone that full normalization performs on
/// non-string values.
fn extract_arguments(tc: &serde_json::Value) -> String {
    match tc.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Check a single tool call for approval requirement.
///
/// Returns `Some` with approval details if approval is required,
/// or if the encoded tool name is ambiguous across servers.
#[expect(clippy::too_many_lines, reason = "linear validation with clear structure")]
fn check_single_approval(tc: &serde_json::Value, tool_index: &McpToolIndex<'_>) -> Option<PendingApproval> {
    let encoded_name = tc.get("name").and_then(serde_json::Value::as_str)?;

    let (key, entry) = match tool_index.get(encoded_name)? {
        McpToolMatch::Unique { key, entry } => (key, entry),
        McpToolMatch::Ambiguous { count } => {
            warn!(
                encoded_name,
                server_count = count,
                "ambiguous encoded tool name in approval check; requiring approval"
            );
            // Ambiguous targets cannot be uniquely bound on resume; leave the
            // fingerprint empty so resolution fails closed rather than matching.
            return Some(PendingApproval {
                call_id: extract_call_id(tc),
                server_label: "unknown".to_owned(),
                tool_name: encoded_name.to_owned(),
                arguments: extract_arguments(tc),
                target_fingerprint: String::new(),
            });
        },
    };
    let original_tool_name = &key.1;
    if !requires_approval(&parse_approval_policy(entry), original_tool_name) {
        return None;
    }

    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    Some(PendingApproval {
        call_id: extract_call_id(tc),
        server_label,
        tool_name: original_tool_name.to_owned(),
        arguments: extract_arguments(tc),
        target_fingerprint: target_fingerprint(entry),
    })
}

// -----------------------------------------------------------------------------
// Execution
// -----------------------------------------------------------------------------

/// Maximum simultaneous owners of a result payload while a task constructs
/// the temporary output plus private, public-output, error, and persisted
/// representations. Four is the worst case for an MCP tool-error result.
const RESULT_PAYLOAD_OWNER_COUNT: usize = 4;

/// Reserve fixed-size rejection records, then divide the remaining aggregate
/// allowance only among calls that can perform external work.
fn admitted_result_limits(
    executable_count: usize,
    rejected_count: usize,
    max_result_bytes: usize,
    max_total_result_bytes: usize,
) -> Result<(usize, usize), McpResultLimitExceeded> {
    let rejected_reservation = rejected_count
        .checked_mul(MIN_RETAINED_RESULT_BYTES)
        .ok_or(McpResultLimitExceeded)?;
    let execution_batch_limit = max_total_result_bytes
        .checked_sub(rejected_reservation)
        .ok_or(McpResultLimitExceeded)?;
    let per_result_limit = max_result_bytes.min(execution_batch_limit / executable_count.max(1));
    Ok((per_result_limit, execution_batch_limit))
}

/// Derive the payload ceiling represented by one retained-result reservation.
fn result_payload_limit(retained_result_limit: usize) -> usize {
    retained_result_limit / RESULT_PAYLOAD_OWNER_COUNT
}

/// Result of executing a single MCP tool call.
#[derive(Debug)]
struct McpCallResult {
    /// Tool result message for `messages` and `persisted_messages`.
    message: serde_json::Value,
    /// Output item for `output_items`.
    output_item: serde_json::Value,
}

impl McpCallResult {
    /// Serialized bytes retained across private replay, persistence, and public output.
    fn retained_bytes(&self) -> Option<usize> {
        serialized_len(&self.message)
            .ok()?
            .checked_mul(2)?
            .checked_add(serialized_len(&self.output_item).ok()?)
    }
}

/// A result batch exceeded its configured retained-byte ceiling.
#[derive(Debug)]
struct McpResultLimitExceeded;

/// Append one result only when all of its retained owners fit the batch cap.
fn push_result_within_budget(
    results: &mut Vec<McpCallResult>,
    retained_bytes: &mut usize,
    result: McpCallResult,
    max_result_bytes: usize,
    max_total_result_bytes: usize,
) -> Result<(), McpResultLimitExceeded> {
    let bytes = result.retained_bytes().ok_or(McpResultLimitExceeded)?;
    if bytes > max_result_bytes {
        return Err(McpResultLimitExceeded);
    }
    let next = retained_bytes.checked_add(bytes).ok_or(McpResultLimitExceeded)?;
    if next > max_total_result_bytes {
        return Err(McpResultLimitExceeded);
    }
    *retained_bytes = next;
    results.push(result);
    Ok(())
}

/// Count retained bytes for an already-bounded result prefix.
fn retained_results_bytes(results: &[McpCallResult]) -> Result<usize, McpResultLimitExceeded> {
    results.iter().try_fold(0_usize, |total, result| {
        total
            .checked_add(result.retained_bytes().ok_or(McpResultLimitExceeded)?)
            .ok_or(McpResultLimitExceeded)
    })
}

/// Replace an oversized completed result with a small truthful tool error.
///
/// The filter reserves at least [`MIN_RETAINED_RESULT_BYTES`] before dispatch,
/// so this fallback can always be retained after the external side effect has
/// happened. Keeping one result per executed call prevents retries from being
/// encouraged by a batch-wide proxy error.
fn fit_result_or_limit_error(
    tool_call: &serde_json::Value,
    result: McpCallResult,
    retained_limit: usize,
) -> McpCallResult {
    if result.retained_bytes().is_some_and(|bytes| bytes <= retained_limit) {
        return result;
    }
    let bounded_identity = |field: &str| {
        tool_call
            .get(field)
            .and_then(serde_json::Value::as_str)
            .filter(|value| value.len() <= 64)
            .unwrap_or("unknown")
    };
    let call_id = tool_call
        .get("call_id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| value.len() <= 64)
        .unwrap_or_else(|| bounded_identity("id"));
    let fallback = build_error_result(
        call_id,
        "unknown",
        bounded_identity("name"),
        "",
        "MCP tool result exceeded the configured retained-byte limit",
        approval_request_id(tool_call),
    );
    debug_assert!(
        fallback
            .retained_bytes()
            .is_some_and(|bytes| bytes <= MIN_RETAINED_RESULT_BYTES),
        "the fixed result-limit error must fit its pre-dispatch reservation"
    );
    fallback
}

/// Results and terminal decision for one response-wide-budgeted MCP batch.
struct McpBatchResult {
    /// Ordered results for executed and rejected calls.
    results: Vec<McpCallResult>,
    /// Whether at least one model call exceeded the remaining `max_tool_calls` allowance.
    tool_limit_exceeded: bool,
}

/// Controls execution of one homogeneous MCP call batch.
#[derive(Clone, Copy)]
struct McpExecutionOptions {
    /// Whether independent calls may execute concurrently.
    parallel: bool,
    /// Maximum number of calls concurrently in flight.
    max_parallel_calls: usize,
    /// Maximum raw bytes accepted for one MCP result event.
    max_result_bytes: usize,
    /// Maximum serialized bytes retained by one result batch.
    max_total_result_bytes: usize,
    /// Timeout applied independently to each MCP call.
    timeout: Duration,
    /// Whether MCP endpoints may resolve to loopback addresses.
    allow_loopback: bool,
}

/// Execute MCP tool calls — concurrently when `parallel` is true,
/// sequentially otherwise.
async fn execute_mcp_calls(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions,
) -> Result<Vec<McpCallResult>, McpResultLimitExceeded> {
    let minimum_reservation = mcp_calls
        .len()
        .checked_mul(MIN_RETAINED_RESULT_BYTES)
        .ok_or(McpResultLimitExceeded)?;
    if options.max_result_bytes < MIN_RETAINED_RESULT_BYTES
        || options.max_total_result_bytes < minimum_reservation
        || options.max_parallel_calls == 0
    {
        return Err(McpResultLimitExceeded);
    }
    let per_call_limit = options
        .max_result_bytes
        .min(options.max_total_result_bytes / mcp_calls.len().max(1));
    let bounded_options = McpExecutionOptions {
        max_result_bytes: per_call_limit,
        ..options
    };
    if options.parallel {
        Ok(execute_parallel(mcp_calls, tool_index, bounded_options).await)
    } else {
        Ok(execute_sequential(mcp_calls, tool_index, bounded_options).await)
    }
}

/// Execute MCP tool calls concurrently within the owning request future.
///
/// Avoid detached tasks: dropping the request must also cancel every pending
/// external side effect. Panics are converted to per-call errors so one faulty
/// future does not discard successful siblings.
#[expect(
    clippy::too_many_lines,
    reason = "task joining and ordered byte-budget commit are one lifecycle"
)]
async fn execute_parallel(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions,
) -> Vec<McpCallResult> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    let mut remaining_calls = mcp_calls;
    while !remaining_calls.is_empty() {
        let chunk_size = remaining_calls.len().min(options.max_parallel_calls);
        let (chunk, rest) = remaining_calls.split_at(chunk_size);
        remaining_calls = rest;
        let futures = chunk.iter().map(|tc| {
            std::panic::AssertUnwindSafe(execute_single_call(
                tc,
                tool_index,
                options.max_result_bytes,
                options.timeout,
                options.allow_loopback,
            ))
            .catch_unwind()
        });
        for (tc, outcome) in chunk.iter().zip(join_all(futures).await) {
            let result = match outcome {
                Ok(Some(result)) => result,
                Ok(None) => {
                    warn!(tool = ?tc.get("name"), "parallel MCP call returned None, emitting error");
                    error_result_for_dropped_call(tc, "internal error: call produced no result")
                },
                Err(_panic) => {
                    warn!(tool = ?tc.get("name"), "parallel MCP call future panicked, emitting error");
                    error_result_for_dropped_call(tc, "internal error: call future panicked")
                },
            };
            results.push(fit_result_or_limit_error(tc, result, options.max_result_bytes));
        }
    }
    results
}

/// Execute MCP tool calls sequentially, emitting error results
/// for any calls that produce no result.
async fn execute_sequential(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions,
) -> Vec<McpCallResult> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    for tc in mcp_calls {
        let result = if let Some(result) = execute_single_call(
            tc,
            tool_index,
            options.max_result_bytes,
            options.timeout,
            options.allow_loopback,
        )
        .await
        {
            result
        } else {
            warn!(tool = ?tc.get("name"), "sequential MCP call returned None, emitting error");
            error_result_for_dropped_call(tc, "internal error: call produced no result")
        };
        results.push(fit_result_or_limit_error(tc, result, options.max_result_bytes));
    }
    results
}

/// Resolve an encoded function name to its unique entry, rejecting
/// ambiguity.
#[expect(clippy::type_complexity, reason = "key+value pair needed by callers")]
fn resolve_tool_entry<'a>(
    tool_index: &McpToolIndex<'a>,
    encoded_name: &str,
    call_id: &str,
    approval_request_id: Option<&str>,
) -> Result<(&'a (String, String), &'a serde_json::Value), Option<Box<McpCallResult>>> {
    match tool_index.get(encoded_name) {
        Some(McpToolMatch::Unique { key, entry }) => Ok((key, entry)),
        Some(McpToolMatch::Ambiguous { count }) => {
            warn!(
                encoded_name,
                server_count = count,
                "ambiguous MCP tool name: multiple entries produce this encoded name"
            );
            Err(Some(Box::new(build_error_result(
                call_id,
                "unknown",
                encoded_name,
                "",
                &format!("ambiguous tool name: {count} servers expose '{encoded_name}'"),
                approval_request_id,
            ))))
        },
        None => {
            warn!(encoded_name, "tool not found in mcp_tool_map, skipping");
            Err(None)
        },
    }
}

/// Parse tool call arguments, handling JSON-string encoding.
fn parse_call_arguments(
    tool_call: &serde_json::Value,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    approval_request_id: Option<&str>,
) -> Result<(serde_json::Value, String), Box<McpCallResult>> {
    let empty = serde_json::Value::Object(serde_json::Map::new());
    let raw = tool_call.get("arguments").unwrap_or(&empty);
    normalize_arguments(raw).map_err(|e| {
        warn!(tool_name, error = %e, "malformed JSON in tool call arguments");
        Box::new(build_error_result(
            call_id,
            server_label,
            tool_name,
            raw.as_str().unwrap_or_default(),
            &e,
            approval_request_id,
        ))
    })
}

/// Build result from a completed (successful or failed) MCP call.
#[expect(clippy::too_many_lines, reason = "match branches with structured logging")]
#[expect(
    clippy::too_many_arguments,
    reason = "result identity and configured size policy are independent inputs"
)]
fn process_call_result(
    result: Result<rmcp::model::CallToolResult, mcp_client::McpClientError>,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments_string: &str,
    approval_request_id: Option<&str>,
    max_result_bytes: usize,
) -> McpCallResult {
    match result {
        Ok(r) => {
            let is_error = r.is_error.unwrap_or(false);
            let content_count = r.content.len();
            let output = content_blocks_to_output(&r.content, max_result_bytes);
            drop(r);
            match output {
                Ok(output_text) => {
                    debug!(tool_name, call_id, is_error, content_count, "MCP tool call completed");
                    build_success_result(
                        call_id,
                        server_label,
                        tool_name,
                        arguments_string,
                        &output_text,
                        is_error,
                        approval_request_id,
                    )
                },
                Err(e) => {
                    warn!(
                        tool_name, call_id, error = %e,
                        "failed to serialize MCP content blocks; returning tool error"
                    );
                    build_error_result(
                        call_id,
                        server_label,
                        tool_name,
                        arguments_string,
                        &e,
                        approval_request_id,
                    )
                },
            }
        },
        Err(e) => {
            warn!(tool_name, call_id, error = %e, "MCP tool call failed");
            build_error_result(
                call_id,
                server_label,
                tool_name,
                arguments_string,
                &e.to_string(),
                approval_request_id,
            )
        },
    }
}

/// Execute a single MCP tool call.
#[expect(clippy::too_many_lines, reason = "linear validation + async call")]
async fn execute_single_call(
    tool_call: &serde_json::Value,
    tool_index: &McpToolIndex<'_>,
    max_result_bytes: usize,
    timeout: Duration,
    allow_loopback: bool,
) -> Option<McpCallResult> {
    let encoded_name = tool_call.get("name").and_then(serde_json::Value::as_str)?;
    let call_id = tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let approval_request_id = approval_request_id(tool_call);

    let (key, entry) = match resolve_tool_entry(tool_index, encoded_name, call_id, approval_request_id) {
        Ok(r) => r,
        Err(opt) => return opt.map(|b| *b),
    };
    let original_tool_name = &key.1;
    let server_url = entry.get("server_url").and_then(serde_json::Value::as_str)?;
    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization").and_then(serde_json::Value::as_str);
    let (arguments, arguments_string) = match parse_call_arguments(
        tool_call,
        call_id,
        server_label,
        original_tool_name,
        approval_request_id,
    ) {
        Ok(r) => r,
        Err(r) => return Some(*r),
    };

    debug!(
        tool_name = original_tool_name,
        server_label, call_id, "executing MCP tool call"
    );

    // The temporary decoded output is copied into private and public result
    // fields, and an MCP tool-error also retains it in the public error field.
    // Limiting the wire/result payload to one quarter of the retained allowance
    // keeps every in-flight task within its aggregate reservation even at that
    // worst-case ownership point.
    let payload_limit = result_payload_limit(max_result_bytes);
    if payload_limit == 0 {
        return Some(build_error_result(
            call_id,
            server_label,
            original_tool_name,
            &arguments_string,
            "MCP result allowance is too small to retain a result",
            approval_request_id,
        ));
    }

    let result = mcp_client::call_tool(
        server_url,
        headers,
        authorization,
        original_tool_name,
        arguments,
        timeout,
        payload_limit,
        allow_loopback,
    )
    .await;
    Some(process_call_result(
        result,
        call_id,
        server_label,
        original_tool_name,
        &arguments_string,
        approval_request_id,
        payload_limit,
    ))
}

// -----------------------------------------------------------------------------
// Result Construction
// -----------------------------------------------------------------------------

/// Serialize rmcp `ContentBlock` values into a lossless string for the
/// Responses `output` field.
///
/// Text-only results keep the simple newline-joined form for readability and
/// backward compatibility. Any result carrying non-text content (image, audio,
/// embedded resource, or resource link) is serialized as the compact JSON MCP
/// content array so that no block is silently dropped.
///
/// Returns `Err` when the content cannot be represented, so the caller can
/// surface an explicit tool error instead of a lossy empty success.
#[expect(
    clippy::too_many_lines,
    reason = "text and structured MCP blocks share one pre-allocation size policy"
)]
fn content_blocks_to_output(blocks: &[rmcp::model::ContentBlock], max_result_bytes: usize) -> Result<String, String> {
    const TOO_LARGE: &str = "MCP tool result exceeded the configured per-result byte limit";
    if blocks
        .iter()
        .all(|block| matches!(block, rmcp::model::ContentBlock::Text(_)))
    {
        let mut output = String::new();
        for block in blocks {
            let rmcp::model::ContentBlock::Text(text) = block else {
                unreachable!("text-only content was checked above")
            };
            let separator_bytes = usize::from(!output.is_empty());
            let next_bytes = output
                .len()
                .checked_add(separator_bytes)
                .and_then(|bytes| bytes.checked_add(text.text.len()))
                .ok_or_else(|| TOO_LARGE.to_owned())?;
            if next_bytes > max_result_bytes {
                return Err(TOO_LARGE.to_owned());
            }
            if separator_bytes != 0 {
                output.push('\n');
            }
            output.push_str(&text.text);
        }
        return Ok(output);
    }

    let output = serde_json::to_string(blocks).map_err(|e| format!("failed to serialize MCP content blocks: {e}"))?;
    if output.len() > max_result_bytes {
        return Err(TOO_LARGE.to_owned());
    }
    Ok(output)
}

/// Build result structs for a successful MCP call.
#[expect(clippy::too_many_arguments, reason = "all args needed for result construction")]
#[expect(clippy::too_many_lines, reason = "success/error branches expand the json! blocks")]
fn build_success_result(
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
    output_text: &str,
    is_error: bool,
    approval_request_id: Option<&str>,
) -> McpCallResult {
    let message = serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": if is_error {
            format!("Error: {output_text}")
        } else {
            output_text.to_owned()
        },
    });

    let output_item = if is_error {
        serde_json::json!({
            "type": "mcp_call",
            "id": call_id,
            "approval_request_id": approval_request_id,
            "server_label": server_label,
            "name": tool_name,
            "arguments": arguments,
            "output": output_text,
            "error": output_text,
        })
    } else {
        serde_json::json!({
            "type": "mcp_call",
            "id": call_id,
            "approval_request_id": approval_request_id,
            "server_label": server_label,
            "name": tool_name,
            "arguments": arguments,
            "output": output_text,
        })
    };

    McpCallResult { message, output_item }
}

/// Build an error result for a tool call that was dropped
/// (task panic, cancellation, or missing fields).
fn error_result_for_dropped_call(tool_call: &serde_json::Value, reason: &str) -> McpCallResult {
    let call_id = tool_call
        .get("call_id")
        .or_else(|| tool_call.get("id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let tool_name = tool_call
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    build_error_result(
        call_id,
        "unknown",
        tool_name,
        "",
        reason,
        approval_request_id(tool_call),
    )
}

/// Extract the approval correlation id threaded through an approved tool call.
fn approval_request_id(tool_call: &serde_json::Value) -> Option<&str> {
    tool_call.get("approval_request_id").and_then(serde_json::Value::as_str)
}

/// Build result structs for a failed MCP call.
#[expect(clippy::too_many_arguments, reason = "all args needed for result construction")]
fn build_error_result(
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
    error_message: &str,
    approval_request_id: Option<&str>,
) -> McpCallResult {
    let message = serde_json::json!({
        "type": "function_call_output",
        "call_id": call_id,
        "output": format!("Error: {error_message}"),
    });

    let output_item = serde_json::json!({
        "type": "mcp_call",
        "id": call_id,
        "approval_request_id": approval_request_id,
        "server_label": server_label,
        "name": tool_name,
        "arguments": arguments,
        "output": "",
        "error": error_message,
    });

    McpCallResult { message, output_item }
}
