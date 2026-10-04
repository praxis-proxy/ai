// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Filter 8: execute MCP tool calls against upstream MCP servers.
//!
//! Executes work prepared by `openai_agentic_loop` during the request-body
//! phase of the next `iterative_request_router` iteration. The owner performs
//! response classification, approval partitioning, and terminal policy. This
//! dispatcher resumes client approval responses, lists deferred connectors
//! after a hosted `tool_search_call`, and executes prepared MCP calls via
//! [`mcp_client::call_tool_with_forwarded_headers_bounded_initialize`]. A streaming deferred `tools/list` failure is
//! stashed during the body pre-read and emitted from `on_request` as the
//! canonical `response.mcp_list_tools.failed` / `response.failed` lifecycle.
//!
//! # Pipeline dependencies
//!
//! - **`mcp_tool_resolve`** must run before this filter so that [`ResponsesState::mcp_tool_map`] is populated.
//! - **`openai_stream_events`** (or equivalent accumulator) must collect the upstream response, then
//!   `openai_agentic_loop` records [`ResponsesState::tool_calls`] as selections into canonical output. Currently only
//!   `function_call` events are selected; native `mcp_call` events require either `mcp_tool_resolve` rewriting MCP
//!   tools into function tools or the accumulator adding `mcp_call` support.
//! - **`openai_agentic_loop`** must run after this filter in request order, so response order is `openai_agentic_loop`
//!   then `openai_mcp_dispatch`.
//! - The IRR transition must match `openai_agentic_loop.action = "loop"` and target the same inference step.
//!
//! Ordinary client-side function calls do not match the MCP tool map and are
//! left untouched for the owner to return to the client.
//!
//! [`ResponsesState::tool_calls`]: super::state::ResponsesState
//! [`ResponsesState::mcp_tool_map`]: super::state::ResponsesState

mod approval;
mod config;

pub(crate) use approval::{OWNER_FINGERPRINT, owner_fingerprint};

#[cfg(test)]
#[cfg(feature = "store-sqlite")]
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
    borrow::{Borrow, Cow},
    collections::{HashMap, HashSet},
    future::Future,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{FutureExt as _, StreamExt as _, stream::FuturesUnordered};
use praxis_core::config::InsecureOptions;
use praxis_filter::{
    BodyAccess, BodyMode, ChainBindingContext, FilterAction, FilterError, FilterPipeline, HttpFilter,
    HttpFilterContext, IterationState, Rejection, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use tracing::{debug, warn};

use self::{
    approval::{
        ApprovalError, ResolvedApproval, bind_credential_context, bind_forwarded_header_context, bind_owner_context,
        build_approved_tool_call, build_denial_message, connector_binding_growth_bytes, extract_approval_responses,
        is_approval_response, parse_approval_response, resolve_approval, target_fingerprint,
    },
    config::{MIN_RETAINED_RESULT_BYTES, McpDispatchConfig, build_config, require_inline_outbound_chain},
};
use super::{
    DEFAULT_STORE_NAME,
    error::responses_error_rejection,
    mcp_classify::{McpDisposition, classify_mcp},
    openai_mcp_tool_resolve::{
        McpToolIndex, McpToolMatch, ResolveError, consume_pending_list_tools_failure,
        discover_deferred_connectors_with_forwarded_headers, has_pending_deferred_discovery, resolve_error_action,
        resolve_error_action_from_request_state,
    },
    state::{DispatchFailure, McpApprovalState, McpConnectorContextPolicy, ResponsesState, retained_json_bytes},
};
use crate::{
    callout_headers::effective_body_callout_headers,
    callout_identity::{McpCalloutIdentity, stage_mcp_callout_identity},
    json_body::serialized_len,
    mcp_client,
    state_owner::StateOwner,
    store::{OwnerScopedResponseStore, PendingApprovalRecord, ResponseStoreRegistry},
};

/// Step-local metadata carrying the configured response fan-out cap to the owner.
const MAX_CALLS_METADATA: &str = "responses.mcp_max_calls_per_round";

/// Whether this internally resolved tool entry names a configured connector.
///
/// Optional fields are serialized into the tool map as JSON `null`, so field
/// presence alone is not sufficient to distinguish a request-selected URL.
pub(super) fn is_connector_tool_entry(entry: &serde_json::Value) -> bool {
    entry
        .get("connector_id")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|connector_id| !connector_id.is_empty())
}

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
/// forward_headers:
///   - x-tenant-id
///   - x-user-id
/// timeout_ms: 30000
/// max_calls_per_round: 32
/// max_parallel_calls: 8
/// max_result_bytes: 1048576
/// max_total_result_bytes: 8388608
/// ```
///
/// `forward_headers` applies only to connector-backed tools resolved by
/// `openai_mcp_tool_resolve`. Direct client-selected `server_url` targets never
/// receive ambient request headers. Credential headers are rejected; use the
/// MCP tool entry's dedicated `authorization` field for per-target credentials.
#[derive(Clone)]
pub struct McpDispatchFilter {
    /// Per-filter namespace preventing sessions from crossing dispatcher
    /// configuration boundaries within one logical execution.
    pool_namespace: mcp_client::McpPoolNamespace,
    /// Optional per-user bearer slot required for configured connectors.
    user_credential_slot: Option<String>,
    /// Optional opaque assertion slot required for configured connectors.
    authorization_assertion_slot: Option<String>,
    /// Bound outbound pipeline the `tools/call` and deferred `tools/list`
    /// callouts dial through.
    ///
    /// Carries only operator-configured cross-cutting filters (if any); the dial
    /// target is staged by the transport, so no upstream-selecting filter is
    /// prepended. This filter runs inside an `iterative_request_router` step, which
    /// praxis core builds with a live [`ChainBindingContext`], so an inline
    /// `outbound_chain` (or none) is bound at step-build time via
    /// [`Self::from_config_with_binding`]; a named reference is rejected because IRR
    /// supplies each step an empty top-level named-chain map.
    outbound_pipeline: Arc<FilterPipeline>,
    /// Trusted request headers explicitly allowed across the MCP boundary.
    forward_headers: Vec<http::HeaderName>,
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
    /// Build from parsed YAML config as a plain builtin (no chain binding).
    ///
    /// This path cannot bind an operator `outbound_chain` (it has no
    /// [`ChainBindingContext`]), so it builds an empty outbound pipeline and
    /// rejects a configured `outbound_chain`. Production registers this filter as
    /// chain-binding via [`Self::from_config_with_binding`]; this method exists for
    /// the no-chain default and unit tests, and keeps the `name()` +
    /// `from_config()` pair the filter-docs generator anchors on.
    ///
    /// [`ChainBindingContext`]: praxis_filter::ChainBindingContext
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid, carries an
    /// `outbound_chain` (unsupported without chain binding), or the empty outbound
    /// pipeline cannot be built.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpDispatchConfig = parse_filter_config("openai_mcp_dispatch", config)?;
        let validated = build_config(cfg)?;
        if validated.outbound_chain.is_some() {
            return Err(FilterError::from(
                "openai_mcp_dispatch: outbound_chain requires chain-binding registration; register this \
                 filter with register_chain_binding, not as a plain builtin",
            ));
        }
        // Empty outbound pipeline; posture stays at its safe default until pipeline
        // finalization propagates the operator's global insecure options.
        let outbound_pipeline = mcp_client::build_bare_outbound_pipeline(false)
            .map_err(|error| FilterError::from(format!("openai_mcp_dispatch: {error}")))?;
        Ok(Self::assemble(&validated, outbound_pipeline))
    }

    /// Build from parsed YAML config, binding the operator `outbound_chain`
    /// against the active registry.
    ///
    /// Registered via [`FilterRegistry::register_chain_binding`]. Although this
    /// filter runs nested inside an `iterative_request_router` step, praxis core
    /// builds each IRR step with a live [`ChainBindingContext`], so an inline
    /// `outbound_chain` is bound at step-build time and its filters run as
    /// cross-cutting outbound filters on every `tools/call` and deferred
    /// `tools/list` callout. A named reference is rejected up front (via
    /// `require_inline_outbound_chain`): IRR supplies each step an empty
    /// top-level named-chain map, so a `Named` reference can never resolve inside a
    /// step. The dial target is staged by the transport, so the bound chain carries
    /// no upstream-selecting filter.
    ///
    /// [`FilterRegistry::register_chain_binding`]: praxis_filter::FilterRegistry::register_chain_binding
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the config is invalid, carries a named
    /// `outbound_chain` (unsupported inside an IRR step), or the inline outbound
    /// chain cannot be bound (a cycle, excessive nesting, a terminal filter, or an
    /// ordering violation).
    pub fn from_config_with_binding(
        config: &serde_yaml::Value,
        ctx: &ChainBindingContext<'_>,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: McpDispatchConfig = parse_filter_config("openai_mcp_dispatch", config)?;
        let mut validated = build_config(cfg)?;
        let outbound_chain = validated.outbound_chain.take();
        require_inline_outbound_chain(outbound_chain.as_ref())?;
        let outbound_pipeline =
            mcp_client::bind_mcp_outbound_chain(outbound_chain, ctx, "openai_mcp_dispatch_outbound")?;
        Ok(Self::assemble(&validated, outbound_pipeline))
    }

    /// Assemble the filter from a validated config and a bound outbound pipeline.
    fn assemble(validated: &McpDispatchConfig, outbound_pipeline: Arc<FilterPipeline>) -> Box<dyn HttpFilter> {
        Box::new(Self {
            pool_namespace: mcp_client::McpPoolNamespace::new(),
            user_credential_slot: validated.user_credential.clone(),
            authorization_assertion_slot: validated.authorization_assertion.clone(),
            outbound_pipeline,
            forward_headers: validated
                .forward_headers
                .iter()
                .filter_map(|name| http::HeaderName::from_bytes(name.as_bytes()).ok())
                .collect(),
            timeout: Duration::from_millis(validated.timeout_ms),
            max_calls_per_round: validated.max_calls_per_round,
            max_parallel_calls: validated.max_parallel_calls,
            max_result_bytes: validated.max_result_bytes,
            max_total_result_bytes: validated.max_total_result_bytes,
        })
    }

    /// Execute the pending MCP calls admitted by the per-round MCP limit.
    ///
    /// MCP tools are exempt from the client `max_tool_calls` budget (the OpenAI
    /// Responses API scopes that budget to built-in tools); the whole batch has
    /// already been bounded by `max_calls_per_round` before this runs, so every
    /// call here executes.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "callout threads the per-request MCP subrequest executor and aggregate policy through the batch"
    )]
    async fn execute_pending_calls(
        &self,
        state: &ResponsesState,
        mcp_calls: &[&serde_json::Value],
        tool_index: &McpToolIndex<'_>,
        forwarded_headers: &http::HeaderMap,
        callout: &mcp_client::McpCallout,
        connector_identity: Option<&McpCalloutIdentity>,
        session_pool: &mcp_client::McpSessionPool,
        max_total_result_bytes: usize,
        aggregate_constrained: bool,
    ) -> Result<Vec<McpCallResult>, McpResultLimitExceeded> {
        debug!(
            count = mcp_calls.len(),
            parallel = state.parallel_tool_calls,
            "executing pending MCP tool calls"
        );
        let (per_result_limit, execution_batch_limit) =
            admitted_result_limits(mcp_calls.len(), 0, self.max_result_bytes, max_total_result_bytes)?;
        let (configured_per_result_limit, _) =
            admitted_result_limits(mcp_calls.len(), 0, self.max_result_bytes, self.max_total_result_bytes)?;
        let aggregate_result_policy = if state.retained_payload_limit().is_none() {
            McpAggregateResultPolicy::Unbudgeted
        } else if aggregate_constrained && per_result_limit < configured_per_result_limit {
            McpAggregateResultPolicy::PerCallConstrained
        } else {
            McpAggregateResultPolicy::Budgeted
        };
        let options = McpExecutionOptions {
            parallel: state.parallel_tool_calls,
            max_parallel_calls: self.max_parallel_calls,
            max_result_bytes: per_result_limit,
            configured_max_result_bytes: configured_per_result_limit,
            max_total_result_bytes: execution_batch_limit,
            aggregate_result_policy,
            timeout: self.timeout,
            forwarded_header_names: &self.forward_headers,
            forwarded_headers: Some(forwarded_headers),
            connector_identity,
            session_pool,
            pool_namespace: self.pool_namespace,
        };
        execute_mcp_calls(mcp_calls, tool_index, options, callout).await
    }

    /// Select only configured headers from the effective body-phase request.
    fn forwarded_headers(&self, ctx: &HttpFilterContext<'_>) -> http::HeaderMap {
        let effective = effective_body_callout_headers(ctx, Cow::Borrowed(&ctx.request.headers));
        let mut forwarded = http::HeaderMap::with_capacity(self.forward_headers.len());
        for name in &self.forward_headers {
            if let Some(value) = effective.get(name) {
                forwarded.insert(name.clone(), value.clone());
            }
        }
        forwarded
    }

    /// Bind connector approvals to the ambient headers this request will send.
    fn bind_request_forwarded_header_context(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        headers: &http::HeaderMap,
        connector_identity: Option<&McpCalloutIdentity>,
    ) -> bool {
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return true;
        };
        if state.retained_payload_limit().is_some() {
            let growth = state.mcp_tool_map.values().try_fold(0_usize, |used, entry| {
                used.checked_add(connector_binding_growth_bytes(
                    entry,
                    connector_identity.is_some(),
                    connector_identity
                        .and_then(McpCalloutIdentity::user_credential)
                        .is_some(),
                )?)
            });
            if !growth.is_some_and(|bytes| state.can_retain_payload(bytes)) {
                return false;
            }
        }
        for entry in state.mcp_tool_map.values_mut() {
            bind_forwarded_header_context(entry, &self.forward_headers, headers);
            bind_owner_context(entry, connector_identity.map(McpCalloutIdentity::owner));
            bind_credential_context(entry, connector_identity.and_then(McpCalloutIdentity::user_credential));
        }
        if !state.mcp_tool_map.is_empty() {
            // The resolved map belongs to the cached streaming baseline. Header,
            // owner, or credential binding may rewrite entries without changing
            // the map's length, so invalidate store and rehydrate snapshots.
            state.mark_replay_stable_payload_changed();
        }
        true
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
        let output = &state.accumulated_output;
        state.tool_calls.retain(|assignment| {
            !assignment
                .resolve(output, "function_call")
                .is_some_and(|call| is_mcp_tool_call(call, &tool_index))
        });
        state.approved_tool_calls.clear();
        state.mark_current_output_changed();
    }

    /// Fail closed when no shared sub-request client is available to dial the
    /// MCP `tools/call` callout.
    ///
    /// The filtered-subrequest transport requires the parent transport captured
    /// from the request context; without it the callout cannot be issued, so the
    /// request is rejected with an HTTP 500 rather than silently skipping MCP
    /// execution (which would return the model's unresolved tool calls to the
    /// client).
    fn no_subrequest_client_action() -> FilterAction {
        FilterAction::Reject(responses_error_rejection(
            500,
            "server_error",
            "no sub-request client available for the MCP tools/call callout",
        ))
    }

    /// Record a terminal failure without retaining an oversized result batch.
    fn result_limit_action(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.tool_calls.clear();
            state.mark_current_output_changed();
            state.dispatch_failure = Some(DispatchFailure {
                status: 502,
                code: "server_error",
                message: "MCP result batch exceeded the configured retained-byte limit".to_owned(),
            });
        }
        FilterAction::Continue
    }

    /// Stop the request after an aggregate MCP result admission failure.
    fn aggregate_budget_action(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
        if let Some(pool) = ctx.extensions.get::<mcp_client::McpSessionPool>() {
            pool.drain_in_background();
        }
        let initial = ctx
            .extensions
            .get::<ResponsesState>()
            .is_none_or(|state| state.iteration == 0);
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
            state.discard_payload_for_budget_error();
            state.dispatch_failure = if initial {
                None
            } else {
                Some(DispatchFailure {
                    status: 502,
                    code: "server_error",
                    message: "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while dispatching MCP tools".to_owned(),
                })
            };
        }
        ctx.set_metadata("responses.skip_persist", "true");
        if initial {
            return FilterAction::Reject(responses_error_rejection(
                413,
                "invalid_request_error",
                "request and MCP dispatch exceed openai_agentic_loop.max_retained_bytes",
            ));
        }
        FilterAction::Continue
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
        clippy::large_stack_frames,
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
            // Bound the batch before any store work. A single model round can emit
            // several `mcp_approval_request` items (batched or parallel tool
            // calls), so a resume turn legitimately carries several matching
            // `mcp_approval_response` items. The per-round MCP cap already bounds
            // how many approvals one round could have emitted, so it is the
            // natural ceiling for the resume batch: it keeps the fail-closed work
            // per request finite and holds the server-owned pending-approval load
            // query (`get_pending_approvals`, one `IN (...)` placeholder per
            // approval id in a single statement) far under PostgreSQL's 16-bit Bind
            // parameter ceiling (max_calls_per_round is capped at 1024, versus the
            // ~65k parameter limit). A larger batch than one round could have
            // produced is a client error.
            let max_batch = self.max_calls_per_round;
            if responses.len() > max_batch {
                let e = ApprovalError::Malformed(format!(
                    "a request may carry at most {max_batch} mcp_approval_response item(s) (max_calls_per_round), but {} were supplied",
                    responses.len()
                ));
                warn!(error = %e.message(), "mcp_dispatch: rejecting oversized approval-response batch");
                return Err(approval_rejection(&e));
            }
            let parse_projection = responses
                .iter()
                .try_fold(0_usize, |used, value| used.checked_add(retained_json_bytes(*value)?))
                .and_then(|bytes| bytes.checked_mul(2));
            if !parse_projection.is_some_and(|bytes| state.can_retain_payload(bytes)) {
                record_approval_budget_failure(ctx);
                return Ok(());
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
            // Reject a repeated approval_request_id before any store access. Each
            // proxy-issued mcp_approval_request carries a unique id, so a batch that
            // names one twice is ambiguous (which verdict wins?) and a client
            // error. Failing closed here is clearer than letting the atomic
            // all-or-nothing consume roll back the whole batch and misreport the
            // repeat as an "already used" replay, and it keeps the store queries
            // duplicate-free.
            let mut seen_ids = HashSet::with_capacity(inputs.len());
            for input in &inputs {
                if !seen_ids.insert(input.approval_id.as_str()) {
                    let e = ApprovalError::Malformed(format!(
                        "an mcp_approval_response batch must not repeat approval_request_id '{}'",
                        input.approval_id
                    ));
                    warn!(error = %e.message(), "mcp_dispatch: rejecting duplicate approval-response id");
                    return Err(approval_rejection(&e));
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
        let owner = ctx.extensions.get::<StateOwner>().cloned().ok_or_else(|| {
            responses_error_rejection(401, "missing_state_owner", "trusted state owner assertion is required")
        })?;
        let store = ctx
            .extensions
            .get::<ResponseStoreRegistry>()
            .and_then(|registry| registry.get_scoped(DEFAULT_STORE_NAME, &owner))
            .ok_or_else(|| {
                warn!("mcp_dispatch: response store unavailable while resuming approvals");
                responses_error_rejection(500, "server_error", "response store is not available")
            })?;
        let approval_ids: Vec<&str> = inputs.iter().map(|i| i.approval_id.as_str()).collect();
        let input_local_bytes =
            approval_input_local_bytes(&inputs).and_then(|bytes| bytes.checked_add(previous_response_id.len()));
        let aggregate_budget_armed = ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| state.retained_payload_limit().is_some());
        if aggregate_budget_armed {
            let pending_bytes = store
                .pending_approval_payload_bytes(&previous_response_id, &approval_ids)
                .await
                .map_err(|e| {
                    warn!(error = %e, "mcp_dispatch: failed to size pending approvals");
                    responses_error_rejection(500, "server_error", "failed to load pending approvals")
                })?;
            let peak = input_local_bytes.and_then(|bytes| pending_bytes.checked_mul(2)?.checked_add(bytes));
            if !peak.is_some_and(|bytes| {
                ctx.extensions
                    .get::<ResponsesState>()
                    .is_some_and(|state| state.can_retain_payload(bytes))
            }) {
                record_approval_budget_failure(ctx);
                return Ok(());
            }
        }
        let pending_records = store
            .get_pending_approvals(&previous_response_id, &approval_ids)
            .await
            .map_err(|e| {
                warn!(error = %e, "mcp_dispatch: failed to load pending approvals");
                responses_error_rejection(500, "server_error", "failed to load pending approvals")
            })?;

        if aggregate_budget_armed
            && !approval_resume_peak_fits(ctx, &inputs, &pending_records, input_local_bytes.unwrap_or(usize::MAX))
        {
            record_approval_budget_failure(ctx);
            return Ok(());
        }

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

        if aggregate_budget_armed
            && !ctx
                .extensions
                .get::<ResponsesState>()
                .is_some_and(|state| approval_decisions_fit(state, &resolved, self.max_total_result_bytes))
        {
            record_approval_budget_failure(ctx);
            return Ok(());
        }

        // Phase 3: atomically claim single-use consumption for the whole batch.
        // Every id here has a pending row, so a failed transition means the
        // approval was already consumed (replay) rather than never issued.
        let consumed_at = i64::try_from(ctx.time_source.now().as_millis()).unwrap_or(i64::MAX);
        let claim_ids: Vec<&str> = resolved.iter().map(|d| d.approval_id.as_str()).collect();
        consume_batch(&store, &previous_response_id, &claim_ids, consumed_at).await?;

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

/// Preflight the independently owned values created by approval resumptions.
fn approval_decisions_fit(
    state: &ResponsesState,
    decisions: &[ResolvedApproval],
    max_total_result_bytes: usize,
) -> bool {
    let removed = state
        .messages
        .iter()
        .filter(|message| is_approval_response(message))
        .try_fold(0_usize, |used, message| used.checked_add(retained_json_bytes(message)?));
    let added = decisions.iter().try_fold(0_usize, |used, decision| {
        let bytes = if decision.approve {
            approved_tool_call_projection_bytes(&decision.approval_id, &decision.encoded_name, &decision.arguments)
        } else {
            denial_message_projection_bytes(&decision.approval_id, decision.reason.as_deref())
        };
        let owners = if decision.approve { 1 } else { 2 };
        used.checked_add(bytes?.checked_mul(owners)?)
    });
    let dispatch = if state.retained_payload_limit().is_some() {
        minimum_approval_dispatch_reserve(state, decisions, max_total_result_bytes)
    } else {
        Some(0)
    };
    removed
        .zip(added)
        .zip(dispatch)
        .is_some_and(|((removed, added), dispatch)| {
            added
                .checked_add(dispatch)
                .is_some_and(|added| state.can_replace_retained_payload(removed, added, 0))
        })
}

/// Reserve the minimum initialize and result capacity that dispatch requires
/// after approved calls are injected. This check runs before the durable
/// approval claim so an unexecuted call remains retryable after exhaustion.
#[expect(
    clippy::too_many_lines,
    reason = "counts existing and approved calls before the durable claim"
)]
fn minimum_approval_dispatch_reserve(
    state: &ResponsesState,
    decisions: &[ResolvedApproval],
    max_total_result_bytes: usize,
) -> Option<usize> {
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    let mut calls = 0_usize;
    let mut staging = 0_usize;
    let selected_calls = state.selected_tool_calls();
    for call in extract_mcp_tool_calls(&selected_calls, &tool_index) {
        calls = calls.checked_add(1)?;
        let arguments = mcp_argument_staging_bytes(call.get("arguments").unwrap_or(&serde_json::Value::Null))?;
        staging = staging.checked_add(mcp_call_dispatch_staging(
            call.get("call_id")
                .or_else(|| call.get("id"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            arguments,
            call.get("name").and_then(serde_json::Value::as_str).unwrap_or(""),
            &tool_index,
        )?)?;
    }
    for decision in decisions.iter().filter(|decision| decision.approve) {
        calls = calls.checked_add(1)?;
        staging = staging.checked_add(mcp_call_dispatch_staging(
            &decision.approval_id,
            mcp_string_argument_staging_bytes(&decision.arguments)?,
            &decision.encoded_name,
            &tool_index,
        )?)?;
    }
    let minimum = calls.checked_mul(MIN_RETAINED_RESULT_BYTES)?;
    if minimum > max_total_result_bytes {
        return None;
    }
    staging
        .checked_add(
            calls
                .checked_mul(mcp_client::MIN_TOOL_INITIALIZE_BYTES)?
                .checked_mul(4)?,
        )?
        .checked_add(mcp_callout_peak_bytes(minimum, calls)?)
}

/// Count independently owned callout arguments, correlation ID, and target
/// definition that can coexist with retained state before a result arrives.
fn mcp_call_dispatch_staging(
    id: &str,
    arguments_staging_bytes: usize,
    name: &str,
    tool_index: &McpToolIndex<'_>,
) -> Option<usize> {
    let tool_bytes = match tool_index.get(name) {
        Some(McpToolMatch::Unique { entry, .. }) => retained_json_bytes(entry)?,
        _ => 0,
    };
    id.len().checked_add(arguments_staging_bytes)?.checked_add(tool_bytes)
}

/// Bound each live argument owner without parsing the whole argument tree.
fn mcp_argument_staging_bytes(raw: &serde_json::Value) -> Option<usize> {
    match raw {
        serde_json::Value::String(text) => mcp_string_argument_staging_bytes(text),
        other => retained_json_bytes(other)?.checked_mul(4),
    }
}

/// The copied source text coexists with two rmcp message trees and a wire Vec.
fn mcp_string_argument_staging_bytes(text: &str) -> Option<usize> {
    let normalized = text
        .len()
        .checked_add(normalized_argument_number_growth(text.as_bytes())?)?;
    text.len().checked_add(normalized.checked_mul(3)?)
}

/// Count only numeric growth when `serde_json` normalizes a borrowed JSON string.
/// Non-numeric syntax and escaped string contents cannot grow on re-encoding.
fn normalized_argument_number_growth(data: &[u8]) -> Option<usize> {
    let mut in_string = false;
    let mut escaped = false;
    let mut number_start = None;
    let mut growth = 0_usize;
    for offset in 0..=data.len() {
        let byte = data.get(offset).copied();
        if in_string {
            match byte {
                Some(b'\\') if !escaped => escaped = true,
                Some(b'"') if !escaped => in_string = false,
                _ => escaped = false,
            }
            continue;
        }
        if let Some(start) = number_start {
            if matches!(byte, Some(b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')) {
                continue;
            }
            let token = data.get(start..offset)?;
            growth = growth.checked_add(normalized_number_token_growth(token))?;
            number_start = None;
        }
        match byte {
            Some(b'"') => in_string = true,
            Some(b'0'..=b'9' | b'-') => number_start = Some(offset),
            _ => {},
        }
    }
    Some(growth)
}

/// Measure one number's compact output without retaining the argument tree.
fn normalized_number_token_growth(token: &[u8]) -> usize {
    serde_json::from_slice::<serde_json::Number>(token)
        .map_or(32, |number| number.to_string().len().saturating_sub(token.len()))
}

/// Raw payload cloned while parsing client approval controls.
fn approval_input_local_bytes(inputs: &[approval::ApprovalResponseInput]) -> Option<usize> {
    inputs.iter().try_fold(0_usize, |used, input| {
        used.checked_add(input.approval_id.len())?
            .checked_add(input.reason.as_ref().map_or(0, String::len))
    })
}

/// Compact JSON size of an approved invocation without constructing the value.
fn approved_tool_call_projection_bytes(approval_id: &str, encoded_name: &str, arguments: &str) -> Option<usize> {
    // Object punctuation and static keys/values, plus the four dynamic strings.
    let fixed = br#"{"type":"function_call","name":,"call_id":,"arguments":,"approval_request_id":}"#.len();
    fixed
        .checked_add(retained_json_bytes(encoded_name)?)?
        .checked_add(retained_json_bytes(approval_id)?)?
        .checked_add(retained_json_bytes(arguments)?)?
        .checked_add(retained_json_bytes(approval_id)?)
}

/// Compact JSON upper bound for a denial bridge without formatting its output.
fn denial_message_projection_bytes(approval_id: &str, reason: Option<&str>) -> Option<usize> {
    let fixed = br#"{"type":"function_call_output","call_id":,"output":}"#.len();
    let prefix = "Tool call was denied by the user. Reason: ";
    let output_bytes = match reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        Some(reason) => retained_json_bytes(reason)?.checked_add(prefix.len()),
        None => Some(retained_json_bytes("Tool call was denied by the user.")?),
    }?;
    fixed
        .checked_add(retained_json_bytes(approval_id)?)?
        // `output_bytes` already includes string quotes for `reason`; treating
        // the ASCII prefix as additional content is a safe upper bound.
        .checked_add(output_bytes)
}

/// Admit all filter-local and final owners needed to resume stored approvals.
#[expect(
    clippy::too_many_lines,
    reason = "exhaustive projection of database, resolution, hashing, and final owners"
)]
fn approval_resume_peak_fits(
    ctx: &HttpFilterContext<'_>,
    inputs: &[approval::ApprovalResponseInput],
    records: &[PendingApprovalRecord],
    input_local_bytes: usize,
) -> bool {
    const MAX_ENCODED_NAME: &str = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return true;
    };
    let pending_bytes = records.iter().try_fold(0_usize, |used, record| {
        used.checked_add(record.approval_id.len())?
            .checked_add(record.server_label.len())?
            .checked_add(record.tool_name.len())?
            .checked_add(record.arguments.len())?
            .checked_add(record.target_fingerprint.len())
    });
    let target_entry_staging = records
        .iter()
        .filter_map(|record| {
            state
                .mcp_tool_map
                .iter()
                .find(|((server, tool), _)| server == &record.server_label && tool == &record.tool_name)
                .map(|(_, entry)| entry)
        })
        .try_fold(0_usize, |largest, entry| {
            retained_json_bytes(entry).map(|bytes| largest.max(bytes))
        })
        .and_then(|bytes| bytes.checked_add(64));
    // `encode_function_name` temporarily owns the unbounded raw and sanitized
    // names before truncating to 64 bytes. Resolution performs this for the
    // pending target and each map key, one candidate at a time.
    let target_name_staging = records
        .iter()
        .map(|record| record.server_label.len().checked_add(record.tool_name.len()))
        .chain(
            state
                .mcp_tool_map
                .keys()
                .map(|(server, tool)| server.len().checked_add(tool.len())),
        )
        .try_fold(0_usize, |largest, bytes| {
            bytes?.checked_mul(2)?.checked_add(66).map(|bytes| largest.max(bytes))
        });
    let target_resolution_staging = if records.is_empty() {
        Some(0)
    } else {
        target_entry_staging
            .zip(target_name_staging)
            // The retained 64-byte encoded name coexists with either candidate-name
            // construction or target-fingerprint construction.
            .and_then(|(entry, name)| entry.max(name).checked_add(64))
    };
    let mut decisions = Some(0_usize);
    for input in inputs {
        let Some(record) = records.iter().find(|record| record.approval_id == input.approval_id) else {
            // Preserve the existing unknown-id 400 path; absent rows retain no
            // stored payload and need no aggregate projection.
            continue;
        };
        let resolved_raw = record
            .approval_id
            .len()
            .checked_add(input.reason.as_ref().map_or(0, String::len))
            .and_then(|bytes| bytes.checked_add(record.server_label.len()))
            .and_then(|bytes| bytes.checked_add(record.tool_name.len()))
            .and_then(|bytes| bytes.checked_add(64))
            .and_then(|bytes| bytes.checked_add(record.arguments.len()));
        let final_bytes = if input.approve {
            approved_tool_call_projection_bytes(&record.approval_id, MAX_ENCODED_NAME, &record.arguments)
        } else {
            denial_message_projection_bytes(&record.approval_id, input.reason.as_deref())
                .and_then(|bytes| bytes.checked_mul(2))
        };
        decisions = decisions
            .and_then(|used| resolved_raw.and_then(|raw| used.checked_add(raw)))
            .and_then(|used| final_bytes.and_then(|final_bytes| used.checked_add(final_bytes)));
    }
    pending_bytes
        .and_then(|pending| input_local_bytes.checked_add(pending))
        .and_then(|bytes| target_resolution_staging.and_then(|staging| bytes.checked_add(staging)))
        .and_then(|bytes| decisions.and_then(|decisions| bytes.checked_add(decisions)))
        .is_some_and(|bytes| state.can_retain_payload(bytes))
}

/// Fail an approval resume without consuming its durable single-use record.
fn record_approval_budget_failure(ctx: &mut HttpFilterContext<'_>) {
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
        state.dispatch_failure = Some(DispatchFailure {
            status: 502,
            code: "server_error",
            message:
                "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while resuming MCP approval"
                    .to_owned(),
        });
    }
}

/// Atomically claim single-use consumption of every approval in the batch,
/// failing closed on replay, store error, or missing backend.
///
/// The claim is all-or-nothing: a single replayed or duplicated approval
/// rejects the whole batch without consuming any id, so a corrected retry can
/// still resume the legitimately approved calls.
async fn consume_batch(
    store: &OwnerScopedResponseStore,
    response_id: &str,
    approval_ids: &[&str],
    consumed_at: i64,
) -> Result<(), Rejection> {
    match store.consume_approvals(response_id, approval_ids, consumed_at).await {
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
fn approval_persistence_rejection(state: &ResponsesState, tool_name: &str) -> Option<ApprovalRejection> {
    if !response_will_be_stored(state) {
        warn!(
            tool_name = %tool_name,
            "mcp_dispatch: rejecting approval-required call because store=false makes it unresumable"
        );
        return Some(approval_requires_store_rejection(tool_name));
    }
    if !state.store_persist_armed {
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
        state.approved_tool_calls.push(build_approved_tool_call(decision));
        state.mark_current_output_changed();
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

    fn visit_nested_pipelines(&mut self, visitor: &mut dyn FnMut(&mut FilterPipeline)) {
        // Propagate runtime resources and the finalized SSRF posture into the
        // bound outbound pipeline. It is uniquely owned during configuration, so
        // `Arc::get_mut` succeeds; a shared handle would mean the pipeline was
        // already cloned before finalization, which must not happen.
        if let Some(pipeline) = Arc::get_mut(&mut self.outbound_pipeline) {
            visitor(pipeline);
        } else {
            debug_assert!(false, "outbound pipeline must be uniquely owned during configuration");
        }
    }

    fn referenced_files(&self) -> Vec<std::path::PathBuf> {
        self.outbound_pipeline.referenced_files()
    }

    fn apply_insecure_options(&self, options: &InsecureOptions) {
        self.outbound_pipeline.apply_insecure_options(options);
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
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Streaming deferred-discovery failures are stashed during the
        // request-body pre-read and emitted here, matching the initial
        // `openai_mcp_tool_resolve` header-phase lifecycle.
        Ok(consume_pending_list_tools_failure(ctx).unwrap_or(FilterAction::Continue))
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
        // StreamBuffer hooks run before the loop owner's first request hook.
        // Delay approvals and callouts until that owner has applied every
        // configured request-wide limit.
        if ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| state.retained_payload_limit().is_none())
            && ctx.extensions.get::<IterationState>().is_some()
        {
            ctx.extensions.insert(DeferredInitialMcpDispatch(self.clone()));
            return Ok(FilterAction::Continue);
        }
        self.dispatch(ctx, body, end_of_stream).await
    }
}

/// First-turn MCP execution waits for agentic budget admission.
struct DeferredInitialMcpDispatch(McpDispatchFilter);

/// Whether the first MCP dispatch is waiting for agentic budget admission.
pub(crate) fn initial_dispatch_is_deferred(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions.get::<DeferredInitialMcpDispatch>().is_some()
}

/// Run the pending first-turn MCP dispatch after every loop limit is applied.
pub(crate) async fn dispatch_after_budget_admission(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
) -> Result<FilterAction, FilterError> {
    let Some(deferred) = ctx.extensions.remove::<DeferredInitialMcpDispatch>() else {
        return Ok(FilterAction::Continue);
    };
    if ctx
        .extensions
        .get::<ResponsesState>()
        .is_none_or(|state| state.retained_payload_limit().is_none())
    {
        return Ok(FilterAction::Reject(responses_error_rejection(
            500,
            "server_error",
            "openai_mcp_dispatch requires openai_agentic_loop budget admission before first-turn dispatch",
        )));
    }
    deferred.0.dispatch(ctx, body, true).await
}

#[expect(
    clippy::multiple_inherent_impl,
    reason = "the dispatch implementation follows its deferred-entry helpers"
)]
impl McpDispatchFilter {
    /// Execute discovery, approval resume, and bounded MCP calls for this round.
    #[expect(
        clippy::too_many_lines,
        clippy::large_stack_frames,
        reason = "the existing MCP dispatch lifecycle now includes aggregate admission"
    )]
    async fn dispatch(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        ctx.set_metadata(MAX_CALLS_METADATA, self.max_calls_per_round.to_string());
        let forwarded_headers = self.forwarded_headers(ctx);

        // Approval resume runs before the approved call is injected into
        // `tool_calls`, so stage connector identity from the resolved tool map
        // itself. This binds pending approvals to the effective per-user bearer
        // without retaining the raw credential. Direct URL-only requests never
        // stage or bind ambient connector context.
        let has_connector_state = ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
            !state.deferred_mcp.is_empty() || state.mcp_tool_map.values().any(is_connector_tool_entry)
        });
        let dispatch_context_policy = McpConnectorContextPolicy::new(
            self.user_credential_slot.as_deref(),
            self.authorization_assertion_slot.as_deref(),
        );
        if has_connector_state
            && ctx
                .extensions
                .get::<ResponsesState>()
                .is_some_and(|state| state.mcp_connector_context_policy != dispatch_context_policy)
        {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.record_security_failure(DispatchFailure {
                    status: 401,
                    code: "missing_callout_context",
                    message: "MCP connector context policy does not match tool resolution".to_owned(),
                });
            }
            return Ok(FilterAction::Continue);
        }
        let connector_identity = if has_connector_state {
            match stage_mcp_callout_identity(
                ctx,
                self.user_credential_slot.as_deref(),
                self.authorization_assertion_slot.as_deref(),
            ) {
                Ok(identity) => identity,
                Err(_missing) => {
                    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                        state.record_security_failure(DispatchFailure {
                            status: 401,
                            code: "missing_callout_context",
                            message: "required MCP connector context is missing".to_owned(),
                        });
                    }
                    return Ok(FilterAction::Continue);
                },
            }
        } else {
            None
        };
        if !self.bind_request_forwarded_header_context(ctx, &forwarded_headers, connector_identity.as_ref()) {
            return Ok(Self::aggregate_budget_action(ctx));
        }

        // Fetch (or lazily create) the per-execution MCP session pool. It lives
        // in the request's threaded `RequestExtensions`, so this same pool is
        // shared across every agentic round and dropped — cancelling every idle
        // session — when the request completes, is cancelled, or the pipeline is
        // reloaded. The handle is a cheap `Arc` clone threaded into each call.
        let session_pool = ctx
            .extensions
            .get_or_insert_with(mcp_client::McpSessionPool::new)
            .clone();
        if !charge_pooled_mcp_sessions(ctx, &session_pool) {
            return Ok(Self::aggregate_budget_action(ctx));
        }

        // Resume approvals from the previous turn before executing any calls.
        // On approval this injects a function-call-shaped tool call that the
        // dispatch machinery below runs; on denial it appends a
        // `function_call_output` so inference resumes without a tool call.
        if let Err(rejection) = self.resume_approvals(ctx).await {
            return Ok(FilterAction::Reject(rejection));
        }
        if ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| state.retained_payload_failed)
        {
            return Ok(Self::aggregate_budget_action(ctx));
        }

        // Binding, parked sessions, and approval resume can each grow retained
        // state even when this round has no MCP call to execute.
        if ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| !state.can_retain_payload(0))
        {
            return Ok(Self::aggregate_budget_action(ctx));
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        let needs_discovery = has_pending_deferred_discovery(state);
        let selected_calls = state.selected_tool_calls();
        if !needs_discovery && (selected_calls.is_empty() || state.mcp_tool_map.is_empty()) {
            return Ok(FilterAction::Continue);
        }

        let has_connector_calls = selected_calls.iter().any(|call| {
            let Some(name) = call.get("name").and_then(serde_json::Value::as_str) else {
                return false;
            };
            let index = McpToolIndex::new(&state.mcp_tool_map);
            matches!(index.get(name), Some(McpToolMatch::Unique { entry, .. }) if is_connector_tool_entry(entry))
        });
        let needs_connector_context = needs_discovery || has_connector_calls;
        debug_assert!(
            !needs_connector_context || has_connector_state,
            "connector calls or discovery must originate from staged connector state"
        );

        // Capture the parent transport and downstream attributes once, pairing
        // them with the bound outbound pipeline. Both deferred `tools/list`
        // discovery and `tools/call` execution dial through it. Fail closed if the
        // pipeline exposes no shared sub-request client: an MCP callout cannot dial
        // without it, so reject rather than silently skip MCP work.
        let Some(callout) = mcp_client::McpCallout::from_context(ctx, Arc::clone(&self.outbound_pipeline)) else {
            return Ok(Self::no_subrequest_client_action());
        };

        if needs_discovery {
            let bytes = body.as_ref().filter(|bytes| !bytes.is_empty()).map(Bytes::as_ref);
            let action = discover_pending_connectors(
                ctx,
                bytes,
                &self.forward_headers,
                &forwarded_headers,
                &callout,
                connector_identity.as_ref(),
            )
            .await?;
            if !matches!(action, FilterAction::Continue) {
                return Ok(action);
            }
            // Deferred discovery inserts new map entries after the initial
            // binding check. Bind their approval context before constructing a
            // pending target fingerprint, then check the aggregate post-commit
            // state before any call can run.
            if !self.bind_request_forwarded_header_context(ctx, &forwarded_headers, connector_identity.as_ref()) {
                return Ok(Self::aggregate_budget_action(ctx));
            }
            if ctx
                .extensions
                .get::<ResponsesState>()
                .is_some_and(|state| !state.can_retain_payload(0))
            {
                return Ok(Self::aggregate_budget_action(ctx));
            }
        }

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        let selected_calls = state.selected_tool_calls();
        if selected_calls.is_empty() || state.mcp_tool_map.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let tool_index = McpToolIndex::new(&state.mcp_tool_map);
        let mcp_calls = extract_mcp_tool_calls(&selected_calls, &tool_index);
        if mcp_calls.is_empty() {
            return Ok(FilterAction::Continue);
        }

        let Some((result_limit, aggregate_constrained)) =
            aggregate_mcp_result_limit(state, &mcp_calls, self.max_total_result_bytes)
        else {
            return Ok(Self::aggregate_budget_action(ctx));
        };

        let results = match self
            .execute_pending_calls(
                state,
                &mcp_calls,
                &tool_index,
                &forwarded_headers,
                &callout,
                connector_identity.as_ref(),
                &session_pool,
                result_limit,
                aggregate_constrained,
            )
            .await
        {
            Ok(results) => results,
            Err(_limit) if aggregate_constrained => return Ok(Self::aggregate_budget_action(ctx)),
            Err(_limit) => return Ok(Self::result_limit_action(ctx)),
        };
        // rmcp retains initialization peer information (including instructions
        // and _meta) in every parked session. Publish its current charge before
        // admitting the result batch or the next agentic round.
        if !charge_pooled_mcp_sessions(ctx, &session_pool) {
            return Ok(Self::aggregate_budget_action(ctx));
        }
        if !ctx
            .extensions
            .get::<ResponsesState>()
            .is_some_and(|state| mcp_result_commit_fits(state, &results))
        {
            return Ok(Self::aggregate_budget_action(ctx));
        }
        Self::append_results(ctx, results);

        Ok(FilterAction::Continue)
    }
}

/// Publish the request pool's retained metadata only when aggregate admission
/// is active; ordinary MCP calls do not need to serialize parked peer info.
fn charge_pooled_mcp_sessions(ctx: &mut HttpFilterContext<'_>, pool: &mcp_client::McpSessionPool) -> bool {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return true;
    };
    if state.retained_payload_limit().is_none() {
        return true;
    }
    let Some((parked, _closing)) = pool.retained_payload_parts() else {
        return false;
    };
    state.retained_mcp_session_bytes = parked;
    state.retained_mcp_closing_pool = Some(pool.clone());
    true
}

/// Load deferred connector tools when a hosted `tool_search_call` is pending.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "deferred discovery needs request state, trusted headers, executor, and scoped identity"
)]
async fn discover_pending_connectors(
    ctx: &mut HttpFilterContext<'_>,
    body: Option<&[u8]>,
    forwarded_header_names: &[http::HeaderName],
    forwarded_headers: &http::HeaderMap,
    callout: &mcp_client::McpCallout,
    connector_identity: Option<&McpCalloutIdentity>,
) -> Result<FilterAction, FilterError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        warn!("ResponsesState missing when discovering deferred MCP connectors");
        return Ok(FilterAction::Continue);
    };
    match discover_deferred_connectors_with_forwarded_headers(
        state,
        forwarded_header_names,
        forwarded_headers,
        callout,
        connector_identity,
    )
    .await
    {
        Ok(()) => Ok(FilterAction::Continue),
        Err(err) => {
            if matches!(&err, ResolveError::RetainedBudget) {
                if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                    state.discard_payload_for_budget_error();
                    state.dispatch_failure = Some(DispatchFailure {
                        status: 502,
                        code: "server_error",
                        message: err.to_string(),
                    });
                }
                ctx.set_metadata("responses.skip_persist", "true");
                return Ok(FilterAction::Continue);
            }
            let streaming = ctx
                .get_metadata("openai_responses_format.stream")
                .is_some_and(|v| v == "true");
            Ok(match body {
                Some(body) => resolve_error_action(ctx, &err, streaming, body),
                None => resolve_error_action_from_request_state(ctx, &err, streaming),
            })
        },
    }
}

/// Return the response fan-out cap this dispatcher published for the owner.
pub(crate) fn configured_max_calls_per_round(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    ctx.get_metadata(MAX_CALLS_METADATA)?.parse().ok()
}

/// Validate and partition one model-produced MCP round for request-side execution.
///
/// Called by `openai_agentic_loop` during its response phase, before it publishes
/// the sole loop decision. This bounds and validates the batch, emits durable
/// approval requests, and leaves only executable MCP calls in `tool_calls` for
/// this dispatcher's next request phase.
#[expect(
    clippy::too_many_lines,
    reason = "validation and approval partitioning form one atomic pre-dispatch operation"
)]
pub(crate) fn prepare_response_round(
    state: &mut ResponsesState,
    max_calls_per_round: usize,
) -> Result<(), DispatchFailure> {
    let selected_calls = state.selected_tool_calls();
    if selected_calls.is_empty() || state.mcp_tool_map.is_empty() {
        return Ok(());
    }
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    let mcp_call_count = count_mcp_tool_calls(&selected_calls, &tool_index);
    if mcp_call_count > max_calls_per_round {
        return Err(DispatchFailure {
            status: 502,
            code: "server_error",
            message: "model response exceeded the configured MCP call limit".to_owned(),
        });
    }
    let mcp_calls = extract_mcp_tool_calls(&selected_calls, &tool_index);
    if mcp_calls.is_empty() {
        return Ok(());
    }
    let classification_staging = mcp_calls
        .iter()
        .try_fold(0_usize, |used, call| used.checked_add(retained_json_bytes(*call)?))
        .and_then(|bytes| bytes.checked_mul(8))
        .and_then(|bytes| mcp_calls.len().checked_mul(4_096)?.checked_add(bytes));
    if !classification_staging.is_some_and(|bytes| state.can_retain_payload(bytes)) {
        state.discard_payload_for_budget_error();
        return Err(mcp_budget_failure());
    }
    if !mcp_call_ids_are_unique_and_new(&mcp_calls, &state.accumulated_output) {
        return Err(DispatchFailure {
            status: 502,
            code: "server_error",
            message: "model response contained duplicate or missing MCP call_id values".to_owned(),
        });
    }

    let mut pending = Vec::new();
    let mut executable = Vec::new();
    for call in mcp_calls {
        if let Some(approval) = check_single_approval(call, &tool_index) {
            pending.push(approval);
        } else {
            if let Some(assignment) = state.tool_calls.iter().find(|assignment| {
                assignment
                    .resolve(&state.accumulated_output, "function_call")
                    .is_some_and(|item| std::ptr::eq(item, call))
            }) {
                executable.push(assignment.clone());
            }
        }
    }
    if pending.is_empty() {
        return Ok(());
    }

    debug!(count = pending.len(), "MCP tool calls require approval");
    let tool_name = pending.first().map_or("", |approval| approval.tool_name.as_str());
    if let Some(rejection) = approval_persistence_rejection(state, tool_name) {
        return Err(DispatchFailure {
            status: rejection.status,
            code: rejection.code,
            message: rejection.message,
        });
    }
    // Executable calls stay in canonical output. The temporary assignment
    // list still copies each item ID while the original selections live.
    let executable_ids = executable
        .iter()
        .try_fold(0_usize, |used, assignment| used.checked_add(assignment.item_id.len()));
    let admission =
        approval_record_and_output_staging_bytes(&pending).and_then(|bytes| bytes.checked_add(executable_ids?));
    if !admission.is_some_and(|bytes| state.can_retain_payload(bytes)) {
        state.discard_payload_for_budget_error();
        return Err(mcp_budget_failure());
    }
    record_and_emit_approvals(state, pending);
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    let output = &state.accumulated_output;
    state.tool_calls.retain(|assignment| {
        !assignment
            .resolve(output, "function_call")
            .is_some_and(|call| is_mcp_tool_call(call, &tool_index))
    });
    if executable.is_empty() {
        state.mcp_approval_state = McpApprovalState::ApprovalPendingThenReturn;
    } else {
        state.tool_calls.extend(executable);
        state.mcp_approval_state = McpApprovalState::ExecuteUngatedThenReturn;
    }
    state.mark_current_output_changed();
    Ok(())
}

/// Reserve every pending record, its execution-origin ID, and the escaped
/// public JSON item while the current tool calls and local pending values live.
/// Raw-string multipliers cannot cover a control-character label: JSON emits
/// each byte as `\u00xx` in the public approval item.
fn approval_record_and_output_staging_bytes(pending: &[PendingApproval]) -> Option<usize> {
    const OUTPUT_FIXED_BYTES: usize =
        br#"{"type":"mcp_approval_request","id":,"name":,"server_label":,"arguments":}"#.len();
    pending.iter().try_fold(0_usize, |used, call| {
        let record = call
            .call_id
            .len()
            .checked_add(call.server_label.len())?
            .checked_add(call.tool_name.len())?
            .checked_add(call.arguments.len())?
            .checked_add(call.target_fingerprint.len())?;
        let output = OUTPUT_FIXED_BYTES
            .checked_add(retained_json_bytes(&call.call_id)?)?
            .checked_add(retained_json_bytes(&call.tool_name)?)?
            .checked_add(retained_json_bytes(&call.server_label)?)?
            .checked_add(retained_json_bytes(&call.arguments)?)?;
        used.checked_add(record)?
            .checked_add(call.call_id.len())?
            .checked_add(output)
    })
}

/// Describe a failed approval classification caused by the aggregate budget.
fn mcp_budget_failure() -> DispatchFailure {
    DispatchFailure {
        status: 502,
        code: "server_error",
        message:
            "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while classifying MCP approvals"
                .to_owned(),
    }
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
fn extract_mcp_tool_calls<'a, T: Borrow<serde_json::Value>>(
    tool_calls: &'a [T],
    tool_index: &McpToolIndex<'_>,
) -> Vec<&'a serde_json::Value> {
    tool_calls
        .iter()
        .map(Borrow::borrow)
        .filter(|tc| is_mcp_tool_call(tc, tool_index))
        .collect()
}

/// Count MCP-owned function calls without cloning provider payloads.
fn count_mcp_tool_calls<T: Borrow<serde_json::Value>>(tool_calls: &[T], tool_index: &McpToolIndex<'_>) -> usize {
    tool_calls
        .iter()
        .filter(|tc| is_mcp_tool_call((*tc).borrow(), tool_index))
        .count()
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
///
/// This is the membership fast-path: its result is identical to
/// `classify_mcp(tool_call, tool_index) != McpDisposition::NotMcp`, but it is a
/// pure `contains` lookup that never parses the per-tool approval policy, so the
/// tool-extraction hot loops do not pay the policy-parse cost of a full
/// disposition on every call.
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
#[cfg(feature = "store-sqlite")]
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
#[cfg(feature = "store-sqlite")]
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

/// Build the client-visible approval request for a tool call, if it needs one.
///
/// Returns `Some` only when [`classify_mcp`] deems the call
/// [`McpDisposition::ApprovalRequired`] — an MCP call whose policy requires
/// approval, or one whose encoded name is ambiguous across servers
/// (fail-closed). The disposition decision is owned by [`classify_mcp`]; this
/// function only resolves the identity fields needed to render the
/// `mcp_approval_request`.
#[expect(clippy::too_many_lines, reason = "linear validation with clear structure")]
fn check_single_approval(tc: &serde_json::Value, tool_index: &McpToolIndex<'_>) -> Option<PendingApproval> {
    if classify_mcp(tc, tool_index) != McpDisposition::ApprovalRequired {
        return None;
    }

    let encoded_name = tc.get("name").and_then(serde_json::Value::as_str)?;
    // A unique match carries the true server label, original tool name, and a
    // bindable target fingerprint. An ambiguous encoded name cannot be
    // attributed to one server or bound on resume, so it fails closed with the
    // encoded name, an "unknown" label, and an empty fingerprint.
    let (key, entry) = match tool_index.get(encoded_name) {
        Some(McpToolMatch::Unique { key, entry }) => (key, entry),
        Some(McpToolMatch::Ambiguous { count }) => {
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
        // `classify_mcp` returned `ApprovalRequired`, so the name matched at
        // least one tool; a `None` here would contradict that. Fail closed.
        None => return None,
    };

    let server_label = entry
        .get("server_label")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    Some(PendingApproval {
        call_id: extract_call_id(tc),
        server_label,
        tool_name: key.1.clone(),
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

/// Bound the wire body alongside parsed and retained result owners. The
/// transport accepts a JSON-RPC envelope in addition to the decoded payload;
/// its streaming executor can temporarily hold twice that wire ceiling while
/// the adapter classifies an overflowing chunk. A session-ID server can also
/// leave a standalone GET parser and rmcp messages live after each call. Its
/// buffered DELETE response and a serialized automatic control reply remain
/// reserved while parked and closing.
/// Charge every call in the batch because their futures may run concurrently.
fn mcp_callout_peak_bytes(admitted_results: usize, call_count: usize) -> Option<usize> {
    if call_count == 0 {
        return (admitted_results == 0).then_some(0);
    }
    let per_call = admitted_results / call_count;
    let payload = result_payload_limit(per_call);
    let wire = mcp_client::tool_result_wire_cap(payload);
    let stream = mcp_client::tool_stream_retained_reserve(payload, payload)?;
    let delete = mcp_client::tool_delete_retained_reserve(payload)?;
    let control = mcp_client::tool_control_retained_reserve(payload)?;
    admitted_results
        .checked_mul(RESULT_PAYLOAD_OWNER_COUNT)?
        .checked_add(wire.checked_mul(2)?.checked_mul(call_count)?)
        .and_then(|peak| peak.checked_add(stream.checked_mul(call_count)?))
        .and_then(|peak| peak.checked_add(delete.checked_mul(call_count)?))
        .and_then(|peak| peak.checked_add(control.checked_mul(call_count)?))
}

/// Find the largest batch allowance whose complete wire and decoded peak
/// fits, including the transport's fixed JSON-RPC envelope.
fn admitted_result_bytes_for_peak(available: usize, call_count: usize, configured_limit: usize) -> Option<usize> {
    let minimum = call_count.checked_mul(MIN_RETAINED_RESULT_BYTES)?;
    let mut admitted = minimum;
    let mut upper = configured_limit.min(available / RESULT_PAYLOAD_OWNER_COUNT);
    if admitted > upper || mcp_callout_peak_bytes(admitted, call_count).is_none_or(|peak| peak > available) {
        return None;
    }
    while admitted < upper {
        let distance = upper - admitted;
        let candidate = admitted + distance / 2 + distance % 2;
        if mcp_callout_peak_bytes(candidate, call_count).is_some_and(|peak| peak <= available) {
            admitted = candidate;
        } else {
            upper = candidate - 1;
        }
    }
    Some(admitted)
}

/// Result of executing a single MCP tool call.
#[derive(Debug)]
struct McpCallResult {
    /// Tool result message for `messages` and `persisted_messages`.
    message: serde_json::Value,
    /// Output item for `output_items`.
    output_item: serde_json::Value,
    /// A transport or content size ceiling was hit before this error was built.
    size_limit_exceeded: Option<McpSizeLimitFailure>,
}

/// Which admitted size ceiling prevented a completed MCP response.
#[derive(Debug, Clone, Copy)]
enum McpSizeLimitFailure {
    /// Decoded content exceeded the admitted result allowance; retain its
    /// measured size so the configured cap can still own a recoverable error.
    Decoded(usize),
    /// An MCP exchange exceeded this wire ceiling and records its origin.
    Transport {
        /// The exceeded byte ceiling.
        limit: usize,
        /// Exchange that selected the ceiling.
        kind: mcp_client::McpResponseLimitKind,
    },
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

/// Reserve raw callout bodies and parsed result staging alongside the three
/// final result owners before making an external MCP call.
fn aggregate_mcp_result_limit(
    state: &ResponsesState,
    mcp_calls: &[&serde_json::Value],
    configured_limit: usize,
) -> Option<(usize, bool)> {
    if mcp_calls.is_empty() {
        return Some((configured_limit, false));
    }
    let Some(limit) = state.retained_payload_limit() else {
        return Some((configured_limit, false));
    };
    let current = state.retained_payload_bytes_bounded(limit)?;
    // Argument normalization and transport serialization can coexist with the
    // original calls. Parallel calls also retain their tool definitions and
    // result IDs, so reserve those owners before any external work.
    let tool_index = McpToolIndex::new(&state.mcp_tool_map);
    let staging = mcp_calls.iter().try_fold(0_usize, |used, call| {
        used.checked_add(mcp_call_dispatch_staging(
            call.get("call_id")
                .or_else(|| call.get("id"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            mcp_argument_staging_bytes(call.get("arguments").unwrap_or(&serde_json::Value::Null))?,
            call.get("name").and_then(serde_json::Value::as_str).unwrap_or(""),
            &tool_index,
        )?)
    })?;
    // Initialization can leave peer information in the pool after the result
    // is committed. The transport caps each initialize response to at most the
    // decoded tool payload allowance (one quarter of its retained-result
    // allowance), with a 1 KiB compatibility floor. Four simultaneous forms
    // cover buffered wire, decoded response, rmcp peer info, and conversion
    // staging; reserve that floor plus one extra result allowance per call.
    let initialize_floor = mcp_calls
        .len()
        .checked_mul(mcp_client::MIN_TOOL_INITIALIZE_BYTES)?
        .checked_mul(4)?;
    let available = limit
        .checked_sub(current)?
        .checked_sub(staging)?
        .checked_sub(initialize_floor)?;
    let admitted = admitted_result_bytes_for_peak(available, mcp_calls.len(), configured_limit)?;
    Some((admitted, admitted < configured_limit))
}

/// The result vector remains live while messages are cloned into two distinct
/// histories and public output is moved into the accumulator.
fn mcp_result_commit_fits(state: &ResponsesState, results: &[McpCallResult]) -> bool {
    let added = results
        .iter()
        .try_fold(0_usize, |used, result| used.checked_add(result.retained_bytes()?));
    added
        .and_then(|bytes| bytes.checked_mul(2))
        .is_some_and(|bytes| state.can_retain_payload(bytes))
}

/// A result batch exceeded its configured retained-byte ceiling.
#[derive(Debug)]
struct McpResultLimitExceeded;

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

/// Distinguish a request-wide size failure from a configured per-tool limit.
/// The latter stays a bounded tool error so an executed call is not retried.
fn lowered_exchange_ceiling(limit: usize, admitted: usize, configured: usize) -> bool {
    (limit == admitted && admitted < configured)
        || (limit == mcp_client::streaming_executor_backstop(admitted)
            && mcp_client::streaming_executor_backstop(admitted) < mcp_client::streaming_executor_backstop(configured))
}

/// Match the transport's POST, GET cumulative, and streaming executor ceilings.
fn aggregate_transport_ceiling_lowered(
    limit: usize,
    kind: mcp_client::McpResponseLimitKind,
    admitted_payload: usize,
    configured_payload: usize,
) -> bool {
    let (admitted, configured) = match kind {
        mcp_client::McpResponseLimitKind::Initialize | mcp_client::McpResponseLimitKind::Control => (
            admitted_payload.clamp(
                mcp_client::MIN_TOOL_INITIALIZE_BYTES,
                mcp_client::MAX_CONTROL_RESPONSE_BYTES,
            ),
            configured_payload.clamp(
                mcp_client::MIN_TOOL_INITIALIZE_BYTES,
                mcp_client::MAX_CONTROL_RESPONSE_BYTES,
            ),
        ),
        mcp_client::McpResponseLimitKind::Tool => (
            mcp_client::tool_result_wire_cap(admitted_payload),
            mcp_client::tool_result_wire_cap(configured_payload),
        ),
        mcp_client::McpResponseLimitKind::GetStream => (
            mcp_client::tool_stream_cumulative_cap(
                mcp_client::tool_result_wire_cap(admitted_payload),
                admitted_payload,
            ),
            mcp_client::tool_stream_cumulative_cap(
                mcp_client::tool_result_wire_cap(configured_payload),
                configured_payload,
            ),
        ),
    };
    lowered_exchange_ceiling(limit, admitted, configured)
}

/// Distinguish a request-wide size failure from a configured per-tool limit.
/// The latter stays a bounded tool error so an executed call is not retried.
fn aggregate_result_limit_exceeded(result: &McpCallResult, options: &McpExecutionOptions<'_>) -> bool {
    if !matches!(
        options.aggregate_result_policy,
        McpAggregateResultPolicy::PerCallConstrained
    ) {
        return false;
    }
    let admitted_payload = result_payload_limit(options.max_result_bytes);
    let configured_payload = result_payload_limit(options.configured_max_result_bytes);
    let exceeded_aggregate_cap = match result.size_limit_exceeded {
        Some(McpSizeLimitFailure::Decoded(actual)) => actual <= configured_payload,
        Some(McpSizeLimitFailure::Transport { limit, kind }) => {
            aggregate_transport_ceiling_lowered(limit, kind, admitted_payload, configured_payload)
        },
        None => false,
    };
    exceeded_aggregate_cap
        || result
            .retained_bytes()
            .is_none_or(|bytes| bytes > options.max_result_bytes)
}

/// How the request-wide retained budget affects one MCP call's wire ceiling.
#[derive(Clone, Copy)]
enum McpAggregateResultPolicy {
    /// No retained-payload policy is active.
    Unbudgeted,
    /// The policy applies but this call keeps its configured result ceiling.
    Budgeted,
    /// The policy lowered this call's configured result ceiling.
    PerCallConstrained,
}

/// Controls execution of one homogeneous MCP call batch.
#[derive(Clone, Copy)]
struct McpExecutionOptions<'a> {
    /// Whether independent calls may execute concurrently.
    parallel: bool,
    /// Maximum number of calls concurrently in flight.
    max_parallel_calls: usize,
    /// Maximum raw bytes accepted for one MCP result event.
    max_result_bytes: usize,
    /// Configured per-result cap before request-wide admission lowered it.
    configured_max_result_bytes: usize,
    /// Maximum serialized bytes retained by one result batch.
    max_total_result_bytes: usize,
    /// Whether the aggregate policy applies and lowered this call's ceiling.
    aggregate_result_policy: McpAggregateResultPolicy,
    /// Timeout applied independently to each MCP call.
    timeout: Duration,
    /// Names reserved for trusted forwarding, including when values are absent.
    forwarded_header_names: &'a [http::HeaderName],
    /// Trusted request headers selected by operator configuration.
    forwarded_headers: Option<&'a http::HeaderMap>,
    /// Request-scoped context injected only for configured connector entries.
    connector_identity: Option<&'a McpCalloutIdentity>,
    /// Per-execution pool of initialized MCP sessions reused across rounds.
    session_pool: &'a mcp_client::McpSessionPool,
    /// Namespace unique to the dispatcher whose transport configuration opened
    /// the session.
    pool_namespace: mcp_client::McpPoolNamespace,
}

/// Execute MCP tool calls — concurrently when `parallel` is true,
/// sequentially otherwise.
async fn execute_mcp_calls(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
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
        execute_parallel(mcp_calls, tool_index, bounded_options, callout).await
    } else {
        execute_sequential(mcp_calls, tool_index, bounded_options, callout).await
    }
}

/// Execute MCP tool calls concurrently within the owning request future.
///
/// Avoid detached tasks: dropping the request must also cancel every pending
/// external side effect. Panics are converted to per-call errors so one faulty
/// future does not discard successful siblings.
#[expect(
    clippy::too_many_lines,
    reason = "ordered cancellation and bounded per-call errors share one batch"
)]
async fn execute_parallel(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Result<Vec<McpCallResult>, McpResultLimitExceeded> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    let mut remaining_calls = mcp_calls;
    while !remaining_calls.is_empty() {
        let chunk_size = remaining_calls.len().min(options.max_parallel_calls);
        let (chunk, rest) = remaining_calls.split_at(chunk_size);
        remaining_calls = rest;
        let futures = chunk.iter().enumerate().map(|(index, tc)| async move {
            let outcome =
                std::panic::AssertUnwindSafe(Box::pin(execute_single_call(tc, tool_index, &options, callout)))
                    .catch_unwind()
                    .await;
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
            if aggregate_result_limit_exceeded(&result, &options) {
                return Err(McpResultLimitExceeded);
            }
            Ok((index, fit_result_or_limit_error(tc, result, options.max_result_bytes)))
        });
        // FuturesUnordered reports a later overflow immediately, even when an
        // earlier sibling is pending. Dropping it cancels every sibling.
        results.extend(collect_parallel_results(futures, chunk.len()).await?);
    }
    Ok(results)
}

/// Complete a parallel chunk by readiness while retaining the original call order.
async fn collect_parallel_results<F>(
    futures: impl IntoIterator<Item = F>,
    count: usize,
) -> Result<Vec<McpCallResult>, McpResultLimitExceeded>
where
    F: Future<Output = Result<(usize, McpCallResult), McpResultLimitExceeded>>,
{
    let mut pending: FuturesUnordered<F> = futures.into_iter().collect();
    let mut ordered: Vec<Option<McpCallResult>> = std::iter::repeat_with(|| None).take(count).collect();
    while let Some(outcome) = pending.next().await {
        let (index, result) = outcome?;
        *ordered.get_mut(index).ok_or(McpResultLimitExceeded)? = Some(result);
    }
    ordered
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(McpResultLimitExceeded)
}

/// Execute MCP tool calls sequentially, emitting error results
/// for any calls that produce no result.
async fn execute_sequential(
    mcp_calls: &[&serde_json::Value],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Result<Vec<McpCallResult>, McpResultLimitExceeded> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    for tc in mcp_calls {
        let result = if let Some(result) = execute_single_call(tc, tool_index, &options, callout).await {
            result
        } else {
            warn!(tool = ?tc.get("name"), "sequential MCP call returned None, emitting error");
            error_result_for_dropped_call(tc, "internal error: call produced no result")
        };
        if aggregate_result_limit_exceeded(&result, &options) {
            return Err(McpResultLimitExceeded);
        }
        results.push(fit_result_or_limit_error(tc, result, options.max_result_bytes));
    }
    Ok(results)
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
            let decoded_size_failure = output.as_ref().err().and_then(|error| {
                (error == MCP_CONTENT_TOO_LARGE)
                    .then(|| content_output_size(&r.content))
                    .flatten()
            });
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
                    let mut result = build_error_result(
                        call_id,
                        server_label,
                        tool_name,
                        arguments_string,
                        &e,
                        approval_request_id,
                    );
                    result.size_limit_exceeded = decoded_size_failure.map(McpSizeLimitFailure::Decoded);
                    result
                },
            }
        },
        Err(e) => {
            warn!(tool_name, call_id, error = %e, "MCP tool call failed");
            let size_limit_exceeded = match &e {
                mcp_client::McpClientError::ResponseTooLarge { limit, kind, .. } => {
                    Some(McpSizeLimitFailure::Transport {
                        limit: *limit,
                        kind: *kind,
                    })
                },
                _ => None,
            };
            let mut result = build_error_result(
                call_id,
                server_label,
                tool_name,
                arguments_string,
                &e.to_string(),
                approval_request_id,
            );
            result.size_limit_exceeded = size_limit_exceeded;
            result
        },
    }
}

/// Execute a single MCP tool call.
#[expect(clippy::too_many_lines, reason = "linear validation + async call")]
async fn execute_single_call(
    tool_call: &serde_json::Value,
    tool_index: &McpToolIndex<'_>,
    options: &McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
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
    let forwarded_headers = is_connector_tool_entry(entry)
        .then_some(options.forwarded_headers)
        .flatten();
    let connector_context = is_connector_tool_entry(entry)
        .then_some(options.connector_identity)
        .flatten()
        .map(|identity| mcp_client::McpConnectorContext {
            owner: identity.owner(),
            bearer: identity.user_credential(),
            assertion: identity.authorization(),
        });
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
    let payload_limit = result_payload_limit(options.max_result_bytes);
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

    // The opaque key binds target identity to this dispatcher's outbound
    // pipeline, timeout, and forwarding policy. Empty fingerprints prohibit
    // approval matching and session reuse, but the request pool still owns
    // their background close until rmcp releases the session.
    let session_key = mcp_client::McpPoolKey::new(options.pool_namespace, target_fingerprint(entry));
    let initialize_limit = if matches!(options.aggregate_result_policy, McpAggregateResultPolicy::Unbudgeted) {
        mcp_client::MAX_CONTROL_RESPONSE_BYTES
    } else {
        payload_limit
    };
    let result = mcp_client::call_tool_with_forwarded_headers_bounded_initialize(
        Some((options.session_pool, session_key.as_ref())),
        server_url,
        headers,
        authorization,
        options.forwarded_header_names,
        forwarded_headers,
        connector_context.as_ref(),
        original_tool_name,
        arguments,
        options.timeout,
        payload_limit,
        initialize_limit,
        !matches!(options.aggregate_result_policy, McpAggregateResultPolicy::Unbudgeted),
        callout,
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

/// Bounded tool error used when converted MCP content exceeds its admitted cap.
const MCP_CONTENT_TOO_LARGE: &str = "MCP tool result exceeded the configured per-result byte limit";

/// Measure the exact converted output on the rare admitted-cap failure without
/// constructing another provider payload copy.
fn content_output_size(blocks: &[rmcp::model::ContentBlock]) -> Option<usize> {
    let mut text_bytes = 0_usize;
    for block in blocks {
        let rmcp::model::ContentBlock::Text(text) = block else {
            return serialized_len(blocks).ok();
        };
        text_bytes = text_bytes
            .checked_add(usize::from(text_bytes != 0))?
            .checked_add(text.text.len())?;
    }
    Some(text_bytes)
}

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
fn content_blocks_to_output(blocks: &[rmcp::model::ContentBlock], max_result_bytes: usize) -> Result<String, String> {
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
                .ok_or_else(|| MCP_CONTENT_TOO_LARGE.to_owned())?;
            if next_bytes > max_result_bytes {
                return Err(MCP_CONTENT_TOO_LARGE.to_owned());
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
        return Err(MCP_CONTENT_TOO_LARGE.to_owned());
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

    McpCallResult {
        message,
        output_item,
        size_limit_exceeded: None,
    }
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

    McpCallResult {
        message,
        output_item,
        size_limit_exceeded: None,
    }
}
