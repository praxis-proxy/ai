// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vector store file search execution for the OpenAI Responses API.
//!
//! This filter is a pure request-phase dispatcher inside
//! `iterative_request_router` (#1046). `openai_agentic_loop` is the sole loop
//! owner: it parses each model response, normalizes private
//! `function_call(name="file_search")` into a canonical `file_search_call`,
//! appends it to `ResponsesState.accumulated_output`, and records a
//! [`FileSearchAssignment`](crate::openai::responses::state::FileSearchAssignment).
//! On the next IRR re-entry this dispatcher drains
//! those assignments at request-body EOS, searches the vector store, and
//! reconciles each call in place (status, results, citations, and a private
//! model-context bridge for the next round). It never parses the response body
//! and never decides whether another round runs. Search context remains private
//! to the model round trip; completed call items and citations are assembled by
//! the owner into the final public Responses API object.

pub(crate) mod citations;
pub(crate) mod client;
mod config;
mod model_context;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderMap;
use praxis_filter::{
    BodyAccess, BodyMode, ChainBindingContext, FilterAction, FilterError, FilterPipeline, HttpFilter,
    HttpFilterContext, IterationState, SubrequestRuntime, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde_json::Value;
use tracing::warn;

use self::{
    client::{
        CalloutTransport, FileSearchClient, FileSearchClientConfig, FileSearchError, MAX_QUERY_BYTES,
        MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES, SearchBatch, SearchFailure, SearchOptions, SearchSpec,
        request_error,
    },
    config::{FileSearchFilterConfig, ValidatedConfig, build_config_with_client, require_inline_outbound_chain},
    model_context::{FormatLimits, FormatTemplates, MODEL_CONTEXT_TEMPLATES, format_search_results},
};
use crate::{
    callout_headers::effective_body_callout_headers,
    callout_policy::OnFailure,
    http_hop::connection_nominates_header,
    openai::responses::{
        bounded_json_size,
        state::{
            DispatchFailure, FileSearchAssignment, MAX_CITATION_FILES, ResponsesState, retained_json_bytes,
            retained_json_values_bytes,
        },
    },
    subrequest::SubRequestClient,
};

/// Hard cap on vector-store/query fan-out per filter execution.
const MAX_SEARCH_SPECS: usize = 64;

/// Maximum pending file-search calls processed in one continuation.
const MAX_PENDING_CALLS: usize = 64;

/// Maximum queries retained from one pending file-search call.
const MAX_QUERIES_PER_CALL: usize = 64;

/// Maximum formatted context retained across one continuation execution.
///
/// This remains well below the 64 MiB proxy request ceiling after bridge
/// metadata is added. Synthetic bridge messages are used only for the next
/// inference round and are not persisted into rehydration history.
const MAX_TOTAL_MODEL_CONTEXT_BYTES: usize = 2_097_152;

/// Dispatches the loop owner's pending file-search assignments against a vector
/// store API compatible backend.
///
/// The enclosing iterative router owns model re-entry and `openai_agentic_loop`
/// owns the loop decision; this filter only executes the assignments the owner
/// recorded, at request-body EOS on re-entry. Streaming composes through the
/// step-local `openai_stream_events` filter, whose finalizer synthesizes
/// citation-annotated `file_search` lifecycle frames at EOS. Search queries are
/// forwarded unchanged; model context and citation marker formatting are
/// internal.
pub struct FileSearchCalloutFilter {
    /// Callout client for the vector store API.
    client: FileSearchClient,

    /// Prebuilt outbound filter chain every vector-store sub-request runs
    /// through. Held directly (not inside [`FileSearchClient`]) so
    /// [`Arc::get_mut`] succeeds during configuration when the framework visits
    /// nested pipelines.
    outbound: Arc<FilterPipeline>,

    /// Combined router and filter continuation-state ceiling.
    max_state_bytes: usize,

    /// Whether a failed callout rejects or produces an incomplete result.
    on_failure: OnFailure,
}

impl FileSearchCalloutFilter {
    /// Create a filter, binding its configured `outbound_chain` into a prebuilt
    /// pipeline through the chain-binding context.
    ///
    /// This is the only construction path: the filter is registered via
    /// [`FilterRegistry::register_chain_binding`], which supplies the
    /// [`ChainBindingContext`] used to resolve `outbound_chain`. `client` is the
    /// shared (or dedicated) sub-request transport that drives the bound chain.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration is invalid or the outbound
    /// chain cannot be built.
    ///
    /// [`FilterRegistry::register_chain_binding`]: praxis_filter::FilterRegistry::register_chain_binding
    pub fn from_config_with_binding(
        config: &serde_yaml::Value,
        client: SubRequestClient,
        ctx: &ChainBindingContext<'_>,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: FileSearchFilterConfig = parse_filter_config("openai_file_search_callout", config)?;
        require_inline_outbound_chain(&cfg.outbound_chain)?;
        let outbound = ctx.bind_chain(&cfg.outbound_chain)?;
        let validated = build_config_with_client(&cfg, client)?;
        Ok(Self::build(validated, Arc::new(outbound)))
    }

    /// Assemble a filter from validated config and a bound outbound pipeline.
    fn build(validated: ValidatedConfig, outbound: Arc<FilterPipeline>) -> Box<dyn HttpFilter> {
        let client = FileSearchClient::new(FileSearchClientConfig {
            base_url: validated.base_url,
            subrequest_client: validated.subrequest_client,
            forward_header_names: validated.forward_header_names,
            on_failure: validated.on_failure,
            max_response_bytes: validated.max_response_bytes,
            max_total_response_bytes: validated.max_total_response_bytes,
            timeout: validated.timeout,
        });

        Box::new(Self {
            client,
            outbound,
            max_state_bytes: validated.max_state_bytes,
            on_failure: validated.on_failure,
        })
    }

    /// Apply one completed search batch to request-scoped response state.
    ///
    /// Mutates the assigned `file_search_call` items in place inside
    /// [`ResponsesState::accumulated_output`] — the single authoritative public
    /// output — rather than owning a private response body: it sets
    /// `completed`/`incomplete`, adds public results when requested, appends the
    /// private model-context bridges to `messages`, and extends `citation_files`.
    /// On a size overflow it records a shared [`DispatchFailure`] the loop owner
    /// converts into the terminal wire form; the dispatcher never rejects itself.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "sequential result formatting and one transactional state commit"
    )]
    fn apply_batch(
        state: &mut ResponsesState,
        assignments: &[FileSearchAssignment],
        plan: &SearchPlan,
        batch: &SearchBatch,
        framework_bytes: usize,
        max_state_bytes: usize,
    ) -> Result<(), DispatchFailure> {
        let failed_calls: HashSet<usize> = batch.failures.iter().map(|failure| failure.call_index).collect();
        let expose_results = state.include.iter().any(|value| value == "file_search_call.results");
        // Formatting creates a second owner for model context and, when
        // requested, a public projection of every decoded result. Reserve the
        // transient owners before allocating either projection. Checking only
        // after `format_search_results` would let a large search response
        // temporarily exceed the request-wide aggregate ceiling.
        let mut remaining_model_bytes = match reserve_file_search_formatting(state, plan, batch, expose_results) {
            Ok(bytes) => bytes,
            Err(failure) => {
                state.discard_payload_for_budget_error();
                return Err(failure);
            },
        };
        let mut bridges = Vec::with_capacity(plan.calls.len());
        let mut updates = Vec::with_capacity(assignments.len());
        let response_identity_hash = state
            .response_object
            .get("id")
            .and_then(Value::as_str)
            .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
        let mut new_citation_files = HashMap::new();

        for (call_index, call) in plan.calls.iter().enumerate() {
            let results = batch.results_by_call.get(call_index).map_or(&[][..], Vec::as_slice);
            let (query, query_truncated) = join_queries_bounded(&call.queries);
            let Some(source_object) = state
                .accumulated_output
                .get(call.output_index)
                .and_then(Value::as_object)
            else {
                continue;
            };
            let public_id = source_object
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(
                    || Some(format!("fs_{response_identity_hash:016x}_{}", call.output_index)),
                    |_| None,
                );
            let source_id = public_id
                .as_deref()
                .or_else(|| source_object.get("id").and_then(Value::as_str))
                .unwrap_or_default();
            let BudgetedSearchResults {
                citation_files,
                model_messages: call_model_messages,
                public_results,
                serialized_bytes,
                truncated,
            } = BridgeBudget {
                known_citation_files: &state.citation_files,
                staged_citation_files: &new_citation_files,
                max_new_citation_files: MAX_CITATION_FILES
                    .saturating_sub(state.citation_files.len().saturating_add(new_citation_files.len())),
                remaining_model_bytes,
                source_id,
                output_index: call.output_index,
                query: &query,
                response_identity_hash,
                templates: &MODEL_CONTEXT_TEMPLATES,
            }
            .format(results, expose_results);
            remaining_model_bytes = remaining_model_bytes.saturating_sub(serialized_bytes);

            let complete = !call.queries.is_empty()
                && call.planning_error.is_none()
                && !plan.vector_store_ids.is_empty()
                && call.expected_specs == call.scheduled_specs
                && !failed_calls.contains(&call_index)
                && !query_truncated
                && !truncated
                && call_model_messages.is_some();
            let status = if complete { "completed" } else { "incomplete" };
            updates.push(OutputUpdate::new(
                call.output_index,
                source_object,
                public_id,
                status,
                expose_results.then_some(public_results),
            )?);
            if let Some(messages) = call_model_messages {
                bridges.push(messages);
            }
            for (file_id, filename) in citation_files {
                new_citation_files.insert(file_id, filename);
            }
        }

        // Calls dropped by the per-continuation cap still receive a valid public
        // identity and terminal status, but participate in the same transaction.
        for assignment in assignments {
            let output_index = assignment.output_index;
            if plan.calls.iter().any(|call| call.output_index == output_index) {
                continue;
            }
            let Some(object) = state.accumulated_output.get(output_index).and_then(Value::as_object) else {
                continue;
            };
            let public_id = object
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .is_none()
                .then(|| format!("fs_{response_identity_hash:016x}_{output_index}"));
            updates.push(OutputUpdate::new(output_index, object, public_id, "incomplete", None)?);
        }

        let removed_output_bytes = updates
            .iter()
            .try_fold(0_usize, |total, update| total.checked_add(update.previous_bytes))
            .ok_or_else(file_search_state_accounting_failure)?;
        let added_output_bytes = updates
            .iter()
            .try_fold(0_usize, |total, update| total.checked_add(update.updated_bytes))
            .ok_or_else(file_search_state_accounting_failure)?;
        let current_output_bytes =
            retained_json_bytes(&state.accumulated_output).ok_or_else(file_search_state_accounting_failure)?;
        let projected_output_bytes = current_output_bytes
            .checked_sub(removed_output_bytes)
            .and_then(|bytes| bytes.checked_add(added_output_bytes))
            .ok_or_else(file_search_state_accounting_failure)?;
        if projected_output_bytes > MAX_JSON_BODY_BYTES {
            return Err(DispatchFailure {
                status: 502,
                code: "server_error",
                message: "openai_file_search_callout: continuation output exceeds the JSON response byte limit"
                    .to_owned(),
            });
        }

        let bridge_bytes = bridges.iter().try_fold(0_usize, |total, bridge| {
            retained_json_values_bytes(bridge).and_then(|bytes| total.checked_add(bytes))
        });
        let citation_bytes = new_citation_files
            .iter()
            .try_fold(0_usize, |total, (file_id, filename)| {
                total.checked_add(file_id.len().saturating_add(filename.len()))
            });
        let added_bytes = bridge_bytes
            .and_then(|bytes| citation_bytes.and_then(|citations| bytes.checked_add(citations)))
            .and_then(|bytes| bytes.checked_add(added_output_bytes))
            .ok_or_else(file_search_state_accounting_failure)?;

        // The decoded batch and every staged replacement remain live until the
        // transactional commit below. Charge those owners separately from the
        // final state owners represented by `added_bytes`.
        let staged_update_bytes = updates.iter().try_fold(0_usize, |total, update| {
            let id_bytes = update.public_id.as_ref().map_or(0, String::len);
            let result_bytes = update.public_results.as_ref().map_or(Some(0), retained_json_bytes);
            total
                .checked_add(id_bytes)
                .and_then(|total| result_bytes.and_then(|bytes| total.checked_add(bytes)))
        });
        let staged_citation_bytes = new_citation_files
            .iter()
            .try_fold(0_usize, |total, (file_id, filename)| {
                total.checked_add(file_id.len().saturating_add(filename.len()))
            });
        let staging_bytes = batch
            .staging_bytes()
            .and_then(|bytes| {
                bridges.iter().try_fold(bytes, |total, bridge| {
                    retained_json_values_bytes(bridge).and_then(|bytes| total.checked_add(bytes))
                })
            })
            .and_then(|bytes| staged_update_bytes.and_then(|updates| bytes.checked_add(updates)))
            .and_then(|bytes| staged_citation_bytes.and_then(|citations| bytes.checked_add(citations)))
            .ok_or_else(file_search_state_accounting_failure)?;

        if !state.can_replace_retained_payload(removed_output_bytes, added_bytes, staging_bytes) {
            state.discard_payload_for_budget_error();
            return Err(DispatchFailure {
                status: 502,
                code: "server_error",
                message: "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while appending file-search results"
                    .to_owned(),
            });
        }
        if !continuation_state_replacement_fits(
            framework_bytes,
            state,
            max_state_bytes,
            removed_output_bytes,
            added_bytes,
            staging_bytes,
        ) {
            return Err(continuation_state_dispatch_failure());
        }

        for update in updates {
            update.commit(&mut state.accumulated_output);
        }
        state.citation_files.extend(new_citation_files);
        // Append only the private model-context bridges (function_call +
        // function_call_output pairs) to `messages`. The loop owner already routed
        // this round's reasoning items into `messages`; a hosted file_search_call
        // is not valid OpenResponses input (issue #808), so no output item is
        // replayed here. Bridges are next-round-only and never persisted (see
        // `MAX_TOTAL_MODEL_CONTEXT_BYTES`).
        for bridge in bridges {
            state.messages.extend(bridge);
        }
        Ok(())
    }

    /// Execute the bounded fan-out for a completed plan.
    #[expect(
        clippy::too_many_lines,
        reason = "separates global, per-call, and transport planning failures"
    )]
    async fn execute_plan(
        &self,
        plan: &SearchPlan,
        request_headers: &HeaderMap,
        max_decoded_bytes: usize,
        transport: &CalloutTransport<'_>,
    ) -> SearchBatch {
        if let Some(message) = plan.planning_error {
            return SearchBatch::with_failures(
                plan.calls.len(),
                plan.calls
                    .iter()
                    .enumerate()
                    .map(|(call_index, _call)| SearchFailure {
                        call_index,
                        error: request_error("planning", message),
                    })
                    .collect(),
            );
        }
        let planning_failures = plan
            .calls
            .iter()
            .enumerate()
            .filter_map(|(call_index, call)| {
                call.planning_error.map(|message| SearchFailure {
                    call_index,
                    error: request_error("planning", message),
                })
            })
            .collect::<Vec<_>>();
        let specs = build_search_specs(plan);
        let mut batch = if specs.is_empty() {
            SearchBatch::new(plan.calls.len())
        } else if max_decoded_bytes == usize::MAX {
            self.client
                .search(&specs, plan.calls.len(), request_headers, transport)
                .await
        } else {
            self.client
                .search_with_retained_limit(
                    &specs,
                    SearchOptions {
                        call_count: plan.calls.len(),
                        request_headers,
                        max_decoded_bytes,
                        transport,
                    },
                )
                .await
        };
        batch.failures.extend(planning_failures);
        batch
    }

    /// Log every failed search and, when configured to fail closed, build the
    /// shared terminal outcome for the loop owner to convert.
    ///
    /// The dispatcher records this in [`ResponsesState::dispatch_failure`] and
    /// returns `Continue`; it never rejects, because `openai_agentic_loop` is the
    /// sole terminal-response owner (issue #1046).
    fn dispatch_failure(&self, batch: &SearchBatch) -> Option<DispatchFailure> {
        for failure in &batch.failures {
            warn!(
                call_index = failure.call_index,
                error = %failure.error,
                "vector store search failed"
            );
        }
        if self.on_failure != OnFailure::Closed {
            return None;
        }
        // Prefer an oversized-response failure so it maps to the actionable 413
        // instead of being masked by an earlier generic callout failure recorded
        // in the same batch.
        let failure = batch
            .failures
            .iter()
            .find(|failure| matches!(failure.error, FileSearchError::ResponseTooLarge { .. }))
            .or_else(|| batch.failures.first())?;
        let (status, code) = failure.error.dispatch_status();
        Some(DispatchFailure {
            status,
            code,
            message: format!("openai_file_search_callout: {}", failure.error),
        })
    }

    /// Execute the file-search calls the loop owner assigned this round.
    ///
    /// Runs once at request-body EOS on IRR re-entry, before
    /// `openai_agentic_loop` prepares the next inference request. It drains the
    /// [`FileSearchAssignment`]s, runs the bounded vector-store fan-out, and
    /// reconciles each assigned item in place inside `accumulated_output`
    /// (setting `completed`/`incomplete`, adding results, bridging model context,
    /// extending citations). It records a [`DispatchFailure`] on failure but never
    /// decides whether another inference round occurs and never commits a terminal
    /// response.
    #[expect(clippy::too_many_lines, reason = "sequential drain, plan, execute, and reconcile")]
    async fn dispatch(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let assignments = match ctx.extensions.get_mut::<ResponsesState>() {
            Some(state) => state.drain_file_search_assignments(),
            None => return Ok(FilterAction::Continue),
        };
        if assignments.is_empty() {
            return Ok(FilterAction::Continue);
        }
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        if !file_search_plan_projection_fits(state, &assignments) {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.discard_payload_for_budget_error();
                state.dispatch_failure = Some(file_search_budget_failure());
            }
            return Ok(FilterAction::Continue);
        }
        let plan = build_search_plan(state, &assignments);
        let Some(plan_bytes) = plan.retained_payload_bytes() else {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.discard_payload_for_budget_error();
                state.dispatch_failure = Some(file_search_budget_failure());
            }
            return Ok(FilterAction::Continue);
        };
        let hdrs = callout_request_headers(ctx);
        let Some(max_decoded_bytes) = retained_payload_available(state, plan_bytes) else {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.discard_payload_for_budget_error();
                state.dispatch_failure = Some(file_search_budget_failure());
            }
            return Ok(FilterAction::Continue);
        };
        // Capture the downstream client attributes forwarded into every filtered
        // sub-request, then bind the request-scoped transport to the prebuilt
        // outbound chain. `run` takes `&self`, so one transport drives the whole
        // bounded fan-out.
        let downstream = SubrequestRuntime::new(
            ctx.client_addr,
            ctx.downstream_tls,
            ctx.peer_identity.clone(),
            ctx.request_start,
        );
        let transport = CalloutTransport {
            outbound: &self.outbound,
            downstream,
        };
        let batch = self.execute_plan(&plan, &hdrs, max_decoded_bytes, &transport).await;
        if batch.retained_payload_overflow() {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.discard_payload_for_budget_error();
                state.dispatch_failure = Some(file_search_budget_failure());
            }
            return Ok(FilterAction::Continue);
        }
        if let Some(failure) = self.dispatch_failure(&batch) {
            if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
                state.dispatch_failure = Some(failure);
            }
            return Ok(FilterAction::Continue);
        }
        let framework_bytes = retained_iteration_bytes(ctx);
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        let apply_result = Self::apply_batch(
            state,
            &assignments,
            &plan,
            &batch,
            framework_bytes,
            self.max_state_bytes,
        );
        drop(batch);
        if let Err(failure) = apply_result {
            state.dispatch_failure = Some(failure);
            return Ok(FilterAction::Continue);
        }
        Ok(FilterAction::Continue)
    }
}

#[async_trait]
impl HttpFilter for FileSearchCalloutFilter {
    fn name(&self) -> &'static str {
        "openai_file_search_callout"
    }

    /// Propagate server-provided runtime resources into the bound outbound
    /// pipeline. The pipeline is uniquely owned during configuration, so
    /// [`Arc::get_mut`] succeeds; a live clone would mean this ran after request
    /// dispatch, which the framework never does.
    fn visit_nested_pipelines(&mut self, visitor: &mut dyn FnMut(&mut FilterPipeline)) {
        if let Some(pipeline) = Arc::get_mut(&mut self.outbound) {
            visitor(pipeline);
        } else {
            debug_assert!(false, "outbound pipeline must be uniquely owned during configuration");
        }
    }

    /// Surface the bound chain's referenced files so hot-reload discovers them.
    fn referenced_files(&self) -> Vec<std::path::PathBuf> {
        self.outbound.referenced_files()
    }

    /// Apply the operator's insecure posture to the bound chain, so central
    /// `allow_private_upstreams` gating reaches every filtered sub-request.
    fn apply_insecure_options(&self, options: &praxis_core::config::InsecureOptions) {
        self.outbound.apply_insecure_options(options);
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        // The dispatcher no longer parses or rewrites response bytes — the loop
        // owner (`openai_agentic_loop`) is the sole response parser and terminal
        // owner. Reading nothing keeps this filter a pure request-phase executor.
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        // Declaring `Stream` keeps this filter composable in an iterative-router
        // step with `openai_responses_proxy`, which always advertises the
        // streaming capability and requires every response-body filter in its
        // step to use `BodyMode::Stream` (or reject streaming) rather than
        // silently buffer. The filter implements no `on_response_body`, so it is a
        // transparent pass-through on the response path.
        BodyMode::Stream
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        self.dispatch(ctx).await
    }
}

/// On IRR continuation iterations the synthetic request lacks client
/// credentials; return the original request headers so `forward_headers`
/// can reach the vector store. Borrows on the first iteration; on
/// continuations filters out headers nominated by `Connection`.
fn callout_request_headers<'a>(ctx: &'a HttpFilterContext<'_>) -> Cow<'a, HeaderMap> {
    let headers = if let Some(state) = ctx.extensions.get::<IterationState>().filter(|s| s.iteration() > 0) {
        let original = &state.original_request.headers;
        let mut filtered = HeaderMap::with_capacity(original.len());
        for (name, value) in original {
            if !connection_nominates_header(original, name) {
                filtered.append(name.clone(), value.clone());
            }
        }
        Cow::Owned(filtered)
    } else {
        Cow::Borrowed(&ctx.request.headers)
    };
    effective_body_callout_headers(ctx, headers)
}

/// Framework-owned bytes already charged by the iterative router.
fn retained_iteration_bytes(ctx: &HttpFilterContext<'_>) -> usize {
    ctx.extensions
        .get::<IterationState>()
        .map_or(0, IterationState::retained_bytes)
}

/// Check framework-retained and filter-owned continuation payloads together.
#[cfg(test)]
fn continuation_state_fits(
    framework_bytes: usize,
    state: &ResponsesState,
    max_bytes: usize,
    incoming_bytes: usize,
) -> bool {
    let framework_and_incoming = framework_bytes.saturating_add(incoming_bytes);
    let remaining = max_bytes.saturating_sub(framework_and_incoming);
    framework_and_incoming <= max_bytes
        && state
            .retained_payload_bytes_bounded_without_external(remaining)
            .is_some()
}

/// Check a prospective payload replacement against file-search's independent
/// compatibility ceiling without committing it to shared state.
#[expect(
    clippy::too_many_arguments,
    reason = "exact replacement accounting across two independent budgets"
)]
fn continuation_state_replacement_fits(
    framework_bytes: usize,
    state: &ResponsesState,
    max_bytes: usize,
    removed_bytes: usize,
    added_bytes: usize,
    incoming_bytes: usize,
) -> bool {
    let framework_and_incoming = framework_bytes.saturating_add(incoming_bytes);
    let remaining = max_bytes.saturating_sub(framework_and_incoming);
    let Some(current) = state.retained_payload_bytes_bounded_without_external(remaining.saturating_add(removed_bytes))
    else {
        return false;
    };
    framework_and_incoming <= max_bytes
        && current.saturating_sub(removed_bytes).saturating_add(added_bytes) <= remaining
}

/// Build the shared terminal outcome for continuation state exceeding its ceiling.
///
/// The dispatcher records this in [`ResponsesState::dispatch_failure`]; the loop
/// owner converts it into a buffered JSON rejection before commitment or a
/// logical-stream SSE error after commitment.
fn continuation_state_dispatch_failure() -> DispatchFailure {
    DispatchFailure {
        status: 413,
        code: "invalid_request_error",
        message: "openai_file_search_callout: continuation state exceeds max_state_bytes".to_owned(),
    }
}

/// Owned execution plan, independent from request state during callouts.
struct SearchPlan {
    /// Pending calls in response output order.
    calls: Vec<PendingCall>,

    /// Metadata filters shared by every search spec.
    filters: Option<Value>,

    /// Maximum number of aggregate results per call.
    max_num_results: Option<u64>,

    /// Structural or resource error found before owning request parameters.
    planning_error: Option<&'static str>,

    /// Ranking configuration shared by every search spec.
    ranking_options: Option<Value>,

    /// Bounded scheduled fan-out coordinates.
    spec_coordinates: Vec<SpecCoordinate>,

    /// Vector store identifiers shared by every call.
    vector_store_ids: Vec<String>,
}

impl SearchPlan {
    /// Payload retained by the owned callout plan while responses are decoded
    /// and formatted. Fixed-size coordinates and counters carry no charge.
    fn retained_payload_bytes(&self) -> Option<usize> {
        let mut used = self
            .calls
            .iter()
            .flat_map(|call| &call.queries)
            .chain(&self.vector_store_ids)
            .try_fold(0_usize, |used, value| used.checked_add(value.len()))?;
        for value in [self.filters.as_ref(), self.ranking_options.as_ref()]
            .into_iter()
            .flatten()
        {
            used = used.checked_add(retained_json_bytes(value)?)?;
        }
        Some(used)
    }
}

/// One pending output item and its fan-out accounting.
struct PendingCall {
    /// Number of searches implied before the global cap.
    expected_specs: usize,

    /// Position in `ResponsesState::output_items()`.
    output_index: usize,

    /// Structural error isolated to this call.
    planning_error: Option<&'static str>,

    /// Original search queries.
    queries: Vec<String>,

    /// Number of searches actually scheduled under the global cap.
    scheduled_specs: usize,
}

/// Exact execution-wide budget inputs for one synthetic model bridge.
struct BridgeBudget<'a> {
    /// Citation mappings already retained by earlier calls.
    known_citation_files: &'a HashMap<String, String>,

    /// Citation mappings staged by earlier calls in this batch.
    staged_citation_files: &'a HashMap<String, String>,

    /// Maximum new mappings this call may retain.
    max_new_citation_files: usize,

    /// Remaining compact JSON bytes for immediate model messages.
    remaining_model_bytes: usize,

    /// Final public call identity used to derive a deterministic bridge identity.
    source_id: &'a str,

    /// Per-call response output index.
    output_index: usize,

    /// Query included in synthetic function arguments.
    query: &'a str,

    /// Stable response identity seed.
    response_identity_hash: u64,

    /// Model context templates.
    templates: &'a FormatTemplates<'a>,
}

/// Final budgeted forms committed for one call.
struct BudgetedSearchResults {
    /// Newly retained citation mappings.
    citation_files: HashMap<String, String>,

    /// Model-visible context bridge when at least one context form fits.
    model_messages: Option<[Value; 2]>,

    /// Canonical results optionally exposed in public output.
    public_results: Vec<Value>,

    /// Exact execution-wide byte charge.
    serialized_bytes: usize,

    /// Whether formatting or budget bounds omitted context.
    truncated: bool,
}

/// One output-item mutation staged until every applicable byte budget admits
/// the full file-search batch.
struct OutputUpdate {
    /// Generated identity, absent when the provider identity remains valid.
    public_id: Option<String>,

    /// Replacement public results, absent when the field must be removed.
    public_results: Option<Vec<Value>>,

    /// Absolute canonical output position.
    output_index: usize,

    /// Compact JSON bytes of the current independently owned item.
    previous_bytes: usize,

    /// Terminal status committed with this update.
    status: &'static str,

    /// Compact JSON bytes of the prospective independently owned item.
    updated_bytes: usize,
}

impl OutputUpdate {
    /// Stage one exact-size object mutation without copying the source payload.
    #[expect(clippy::too_many_lines, reason = "exact JSON object member replacement accounting")]
    fn new(
        output_index: usize,
        object: &serde_json::Map<String, Value>,
        public_id: Option<String>,
        status: &'static str,
        public_results: Option<Vec<Value>>,
    ) -> Result<Self, DispatchFailure> {
        let previous_bytes = retained_json_bytes(object).ok_or_else(file_search_state_accounting_failure)?;
        let previous_commas = object.len().saturating_sub(1);
        let mut content_bytes = previous_bytes
            .checked_sub(2_usize.saturating_add(previous_commas))
            .ok_or_else(file_search_state_accounting_failure)?;
        let mut field_count = object.len();

        for key in [public_id.as_ref().map(|_| "id"), Some("status"), Some("results")]
            .into_iter()
            .flatten()
        {
            if let Some(value) = object.get(key) {
                content_bytes = content_bytes
                    .checked_sub(json_member_bytes(key, value).ok_or_else(file_search_state_accounting_failure)?)
                    .ok_or_else(file_search_state_accounting_failure)?;
                field_count = field_count.saturating_sub(1);
            }
        }

        let mut added_content = json_member_bytes("status", status).ok_or_else(file_search_state_accounting_failure)?;
        let mut added_fields = 1_usize;
        if let Some(public_id) = &public_id {
            added_content = added_content
                .checked_add(json_member_bytes("id", public_id).ok_or_else(file_search_state_accounting_failure)?)
                .ok_or_else(file_search_state_accounting_failure)?;
            added_fields = added_fields.saturating_add(1);
        }
        if let Some(public_results) = &public_results {
            added_content = added_content
                .checked_add(
                    json_member_bytes("results", public_results).ok_or_else(file_search_state_accounting_failure)?,
                )
                .ok_or_else(file_search_state_accounting_failure)?;
            added_fields = added_fields.saturating_add(1);
        }
        field_count = field_count.saturating_add(added_fields);
        let updated_bytes = content_bytes
            .checked_add(added_content)
            .and_then(|bytes| bytes.checked_add(2_usize.saturating_add(field_count.saturating_sub(1))))
            .ok_or_else(file_search_state_accounting_failure)?;

        Ok(Self {
            public_id,
            public_results,
            output_index,
            previous_bytes,
            status,
            updated_bytes,
        })
    }

    /// Commit the already-admitted update to the canonical response item.
    fn commit(self, output: &mut [Value]) {
        let Some(object) = output.get_mut(self.output_index).and_then(Value::as_object_mut) else {
            return;
        };
        if let Some(public_id) = self.public_id {
            object.insert("id".to_owned(), Value::String(public_id));
        }
        object.insert("status".to_owned(), Value::String(self.status.to_owned()));
        if let Some(public_results) = self.public_results {
            object.insert("results".to_owned(), Value::Array(public_results));
        } else {
            object.remove("results");
        }
    }
}

/// Return the compact bytes occupied by one JSON object member, excluding its
/// separator comma.
fn json_member_bytes<T: serde::Serialize + ?Sized>(key: &str, value: &T) -> Option<usize> {
    retained_json_bytes(key)?
        .checked_add(1)
        .and_then(|bytes| retained_json_bytes(value).and_then(|value_bytes| bytes.checked_add(value_bytes)))
}

/// Build a bounded server error for impossible serialization/accounting state.
fn file_search_state_accounting_failure() -> DispatchFailure {
    DispatchFailure {
        status: 502,
        code: "server_error",
        message: "openai_file_search_callout: failed to account continuation state".to_owned(),
    }
}

/// Admit the decoded search batch and transient formatting projections before
/// constructing those projections.
///
/// The public projection is conservatively charged at four times the decoded
/// result bytes: one bound for its serialized representation and one for the
/// staged/final owners, with room for the canonical result wrapper. Model
/// bridges have three simultaneous owners while formatting (the rendered
/// string, the bridge value, and its staged/final state owner), so the model
/// budget is capped to one third of the remaining aggregate allowance.
#[expect(
    clippy::too_many_lines,
    reason = "preflight accounts each transient owner before formatting"
)]
fn reserve_file_search_formatting(
    state: &ResponsesState,
    plan: &SearchPlan,
    batch: &SearchBatch,
    expose_results: bool,
) -> Result<usize, DispatchFailure> {
    let batch_bytes = batch.staging_bytes().ok_or_else(file_search_state_accounting_failure)?;
    let result_bytes = batch
        .results_by_call
        .iter()
        .try_fold(0_usize, |total, results| {
            results.iter().try_fold(total, |total, result| {
                retained_json_bytes(result).and_then(|bytes| total.checked_add(bytes))
            })
        })
        .ok_or_else(file_search_state_accounting_failure)?;
    let plan_bytes = plan
        .retained_payload_bytes()
        .ok_or_else(file_search_state_accounting_failure)?;
    let public_bytes = expose_results
        .then_some(result_bytes.checked_mul(4))
        .flatten()
        .unwrap_or(0);
    let non_model_bytes = batch_bytes
        .checked_add(public_bytes)
        .and_then(|bytes| bytes.checked_add(plan_bytes))
        .ok_or_else(file_search_state_accounting_failure)?;

    let model_bytes = match state.retained_payload_limit() {
        Some(limit) => {
            let current = state
                .retained_payload_bytes_bounded(limit)
                .ok_or_else(file_search_budget_failure)?;
            let available = limit
                .checked_sub(current)
                .and_then(|bytes| bytes.checked_sub(non_model_bytes))
                .ok_or_else(file_search_budget_failure)?;
            let model_bytes = MAX_TOTAL_MODEL_CONTEXT_BYTES.min(available / 3);
            let staging = non_model_bytes
                .checked_add(
                    model_bytes
                        .checked_mul(3)
                        .ok_or_else(file_search_state_accounting_failure)?,
                )
                .ok_or_else(file_search_state_accounting_failure)?;
            if !state.can_replace_retained_payload(0, 0, staging) {
                return Err(file_search_budget_failure());
            }
            model_bytes
        },
        None => MAX_TOTAL_MODEL_CONTEXT_BYTES,
    };
    Ok(model_bytes)
}

/// Build the shared aggregate-budget failure before any large file-search
/// formatting allocation is made.
fn file_search_budget_failure() -> DispatchFailure {
    DispatchFailure {
        status: 502,
        code: "server_error",
        message: "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes while formatting file-search results"
            .to_owned(),
    }
}

/// Return payload capacity available for decoded dispatcher results before the
/// callout begins. An unarmed aggregate budget leaves the independent
/// file-search limits authoritative.
fn retained_payload_available(state: &ResponsesState, external_bytes: usize) -> Option<usize> {
    let Some(limit) = state.retained_payload_limit() else {
        return Some(usize::MAX);
    };
    let used = state.retained_payload_bytes_bounded(limit)?;
    limit.checked_sub(used)?.checked_sub(external_bytes)
}

/// Admit the bounded owned execution plan before cloning its query/tool fields
/// out of request state.
fn file_search_plan_projection_fits(state: &ResponsesState, assignments: &[FileSearchAssignment]) -> bool {
    let tool_bytes = state
        .tools
        .iter()
        .find(|tool| tool.get("type").and_then(Value::as_str) == Some("file_search"))
        .map_or(Some(0), retained_json_bytes);
    let call_bytes = assignments
        .iter()
        .take(MAX_PENDING_CALLS)
        .try_fold(0_usize, |used, assignment| {
            state
                .accumulated_output
                .get(assignment.output_index)
                .map_or(Some(used), |item| {
                    retained_json_bytes(item).and_then(|bytes| used.checked_add(bytes))
                })
        });
    tool_bytes
        .zip(call_bytes)
        .and_then(|(tool, calls)| tool.checked_add(calls))
        .is_some_and(|bytes| state.can_retain_payload(bytes))
}

impl BridgeBudget<'_> {
    /// Reserve exact structural and metadata bytes, then render chunks once.
    #[expect(clippy::too_many_lines, reason = "one ordered format and exact-budget transaction")]
    fn format(self, results: &[client::SearchResult], include_public_results: bool) -> BudgetedSearchResults {
        let empty_model_messages = model_context_messages(
            self.source_id,
            self.output_index,
            self.response_identity_hash,
            self.query,
            "",
        );
        let structural_bytes = bounded_json_size(&empty_model_messages, self.remaining_model_bytes)
            .ok()
            .flatten();
        let max_context_bytes = structural_bytes
            .and_then(|bytes| self.remaining_model_bytes.checked_sub(bytes))
            .unwrap_or_default();
        let formatted = format_search_results(
            results,
            self.query,
            self.templates,
            &FormatLimits {
                max_model_context_bytes: max_context_bytes,
                max_new_citation_files: self.max_new_citation_files,
                known_citation_files: self.known_citation_files,
                staged_citation_files: self.staged_citation_files,
                include_public_results,
            },
        );
        let model_messages = model_context_messages(
            self.source_id,
            self.output_index,
            self.response_identity_hash,
            self.query,
            &formatted.model_context,
        );
        let serialized_bytes = bounded_json_size(&model_messages, self.remaining_model_bytes)
            .ok()
            .flatten();
        let context_available = !formatted.model_context.is_empty() || !formatted.truncated;
        if let Some(serialized_bytes) = serialized_bytes
            && context_available
        {
            return BudgetedSearchResults {
                citation_files: formatted.citation_files,
                model_messages: Some(model_messages),
                public_results: formatted.public_results,
                serialized_bytes,
                truncated: formatted.truncated,
            };
        }

        BudgetedSearchResults {
            citation_files: HashMap::new(),
            model_messages: None,
            public_results: formatted.public_results,
            serialized_bytes: 0,
            truncated: true,
        }
    }
}

/// Index-only coordinate used to borrow from one stable owned plan.
struct SpecCoordinate {
    /// Index into `SearchPlan.calls`.
    call_index: usize,

    /// Index into the pending call's queries.
    query_index: usize,

    /// Index into `SearchPlan.vector_store_ids`.
    store_index: usize,
}

/// Tool configuration extracted from the original request.
struct FileSearchToolDef {
    /// Metadata filter passed through to the vector store backend.
    filters: Option<Value>,

    /// Maximum number of aggregate results.
    max_num_results: Option<u64>,

    /// Ranking options passed through to the vector store backend.
    ranking_options: Option<Value>,

    /// Structural or resource error in a required execution field.
    planning_error: Option<&'static str>,

    /// Vector stores to search.
    vector_store_ids: Vec<String>,

    /// Number of string vector-store IDs before retention limits.
    vector_store_count: usize,
}

/// Build an owned plan for the assigned pending calls before the fan-out cap.
fn build_search_plan(state: &ResponsesState, assignments: &[FileSearchAssignment]) -> SearchPlan {
    let tool = extract_file_search_tool_def(&state.tools);
    let mut calls = pending_calls_from_assignments(state, assignments, tool.vector_store_count);
    let spec_coordinates = schedule_searches(&mut calls, tool.vector_store_ids.len());

    SearchPlan {
        calls,
        filters: tool.filters,
        max_num_results: tool.max_num_results,
        planning_error: tool.planning_error,
        ranking_options: tool.ranking_options,
        spec_coordinates,
        vector_store_ids: tool.vector_store_ids,
    }
}

/// Build one [`PendingCall`] per assignment before applying scheduling limits.
///
/// Each [`FileSearchAssignment::output_index`] is an absolute index into
/// [`ResponsesState::accumulated_output`], the single authoritative output. The
/// per-continuation server cap bounds how many assignments enter the plan. The
/// loop owner already applied the client budget in model output order, and the
/// dispatcher terminalizes any assignment the server cap drops.
#[expect(
    clippy::too_many_lines,
    reason = "bounded structural validation and ownership happen together"
)]
fn pending_calls_from_assignments(
    state: &ResponsesState,
    assignments: &[FileSearchAssignment],
    store_count: usize,
) -> Vec<PendingCall> {
    let mut calls = Vec::new();
    for assignment in assignments.iter().take(MAX_PENDING_CALLS) {
        let output_index = assignment.output_index;
        let Some(item) = state.accumulated_output.get(output_index) else {
            continue;
        };
        let Some(query_values) = item.get("queries").and_then(Value::as_array) else {
            calls.push(PendingCall {
                expected_specs: store_count,
                output_index,
                planning_error: Some("file_search_call.queries must be an array"),
                queries: Vec::new(),
                scheduled_specs: 0,
            });
            continue;
        };
        let planning_error = query_values
            .iter()
            .any(|query| !query.is_string())
            .then_some("file_search_call.queries entries must be strings");
        let queries: Vec<String> = query_values
            .iter()
            .filter_map(Value::as_str)
            .take(MAX_QUERIES_PER_CALL)
            .map(|query| bounded_string_copy(query, MAX_QUERY_BYTES))
            .collect();
        calls.push(PendingCall {
            expected_specs: query_values.len().saturating_mul(store_count),
            output_index,
            planning_error,
            queries,
            scheduled_specs: 0,
        });
    }
    calls
}

/// Schedule bounded search coordinates while retaining every pending call.
fn schedule_searches(calls: &mut [PendingCall], store_count: usize) -> Vec<SpecCoordinate> {
    let mut coordinates = Vec::new();
    for (call_index, call) in calls.iter_mut().enumerate() {
        if call.planning_error.is_some() {
            continue;
        }
        for store_index in 0..store_count {
            for query_index in 0..call.queries.len() {
                if coordinates.len() == MAX_SEARCH_SPECS {
                    return coordinates;
                }
                coordinates.push(SpecCoordinate {
                    call_index,
                    query_index,
                    store_index,
                });
                call.scheduled_specs = call.scheduled_specs.saturating_add(1);
            }
        }
    }
    coordinates
}

/// Borrow all request data from one stable plan without deep per-spec clones.
fn build_search_specs(plan: &SearchPlan) -> Vec<SearchSpec<'_>> {
    plan.spec_coordinates
        .iter()
        .filter_map(|coordinate| {
            let call = plan.calls.get(coordinate.call_index)?;
            let query = call.queries.get(coordinate.query_index)?;
            let store_id = plan.vector_store_ids.get(coordinate.store_index)?;
            Some(SearchSpec {
                call_index: coordinate.call_index,
                filters: plan.filters.as_ref(),
                max_num_results: plan.max_num_results,
                query,
                ranking_options: plan.ranking_options.as_ref(),
                store_id,
            })
        })
        .collect()
}

/// Extract the first file search tool definition without validating backend
/// parameter semantics.
#[expect(
    clippy::too_many_lines,
    reason = "validates related file-search execution fields before cloning"
)]
fn extract_file_search_tool_def(tools: &[Value]) -> FileSearchToolDef {
    let tool = tools
        .iter()
        .find(|tool| tool.get("type").and_then(Value::as_str) == Some("file_search"));
    let vector_store_field = tool.and_then(|tool| tool.get("vector_store_ids"));
    let vector_store_values = vector_store_field
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let mut planning_error = match (tool, vector_store_field) {
        (None, _) => Some("a file_search tool definition is required for file_search_call execution"),
        (Some(_), None) => Some("file_search.vector_store_ids is required"),
        (Some(_), Some(value)) if !value.is_array() => Some("file_search.vector_store_ids must be an array"),
        (Some(_), Some(_)) => None,
    };
    if vector_store_values.iter().any(|store_id| !store_id.is_string()) {
        planning_error.get_or_insert("file_search.vector_store_ids entries must be strings");
    }
    let vector_store_count = vector_store_values.len();
    let vector_store_ids = vector_store_values
        .iter()
        .filter_map(Value::as_str)
        .take(MAX_SEARCH_SPECS)
        .map(|store_id| bounded_string_copy(store_id, MAX_VECTOR_STORE_ID_BYTES))
        .collect();

    let filters = tool.and_then(|tool| tool.get("filters"));
    let ranking_options = tool.and_then(|tool| tool.get("ranking_options"));
    if bounded_json_size(&(filters, ranking_options), MAX_SEARCH_REQUEST_BYTES)
        .ok()
        .flatten()
        .is_none()
    {
        planning_error.get_or_insert("file_search filters and ranking_options exceed the outbound request byte limit");
    }
    let max_num_results_field = tool.and_then(|tool| tool.get("max_num_results"));
    let max_num_results = max_num_results_field.and_then(Value::as_u64);
    if max_num_results_field.is_some_and(|value| value.as_u64().is_none()) {
        planning_error.get_or_insert("file_search.max_num_results must be a non-negative integer");
    }

    FileSearchToolDef {
        filters: planning_error.is_none().then_some(filters).flatten().cloned(),
        max_num_results,
        ranking_options: planning_error.is_none().then_some(ranking_options).flatten().cloned(),
        planning_error,
        vector_store_ids,
        vector_store_count,
    }
}

/// Copy a valid bounded value, or only enough of an oversized value for the
/// client to reject it without duplicating the entire request field.
fn bounded_string_copy(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }

    let mut end = max_bytes.saturating_add(1).min(value.len());
    while !value.is_char_boundary(end) {
        end = end.saturating_add(1).min(value.len());
    }
    value.get(..end).unwrap_or_default().to_owned()
}

/// Join model-facing query metadata without duplicating more than one query's
/// maximum outbound byte allowance.
fn join_queries_bounded(queries: &[String]) -> (String, bool) {
    let mut joined = String::new();
    for query in queries {
        let separator_bytes = usize::from(!joined.is_empty());
        let Some(next_len) = joined
            .len()
            .checked_add(separator_bytes)
            .and_then(|length| length.checked_add(query.len()))
        else {
            return (joined, true);
        };
        if next_len > MAX_QUERY_BYTES {
            return (joined, true);
        }
        if separator_bytes != 0 {
            joined.push('\n');
        }
        joined.push_str(query);
    }
    (joined, false)
}

/// Return whether one output item still requires local file-search execution.
///
/// Shared with `openai_web_search`, which must exclude these pending
/// placeholders when counting non-web built-in calls against the shared
/// `max_tool_calls` budget during the owner's ordered admission pass.
pub(crate) fn is_pending_file_search_call(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("file_search_call")
        && matches!(
            item.get("status").and_then(Value::as_str),
            Some("searching" | "in_progress")
        )
}

/// Whether an output item is a vLLM-emitted `function_call` for file search.
pub(crate) fn is_file_search_function_call(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("function_call")
        && item.get("name").and_then(Value::as_str) == Some("file_search")
}

/// Whether the response state contains a file search tool.
pub(crate) fn has_file_search_tool(state: &ResponsesState) -> bool {
    state
        .tools
        .iter()
        .any(|tool| tool.get("type").and_then(Value::as_str) == Some("file_search"))
}

/// Parse file-search queries from a `function_call` arguments string.
///
/// Handles three conventions:
/// 1. `{"query": "..."}` — single query string (most common from vLLM)
/// 2. `{"queries": ["...", "..."]}` — explicit query array
/// 3. Raw string fallback — entire arguments string used as one query
fn extract_file_search_queries(arguments: &str) -> Vec<String> {
    if let Ok(parsed) = serde_json::from_str::<Value>(arguments) {
        if let Some(query) = parsed.get("query").and_then(Value::as_str)
            && !query.is_empty()
        {
            return vec![query.to_owned()];
        }
        if let Some(queries) = parsed.get("queries").and_then(Value::as_array) {
            let result: Vec<String> = queries
                .iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if !result.is_empty() {
                return result;
            }
        }
    }
    if arguments.is_empty() {
        Vec::new()
    } else {
        vec![arguments.to_owned()]
    }
}

/// Translate vLLM `function_call` items with `name == "file_search"` into
/// `file_search_call` items so the pending-call scan recognizes them.
///
/// Returns the round-local output indices it rewrote.
pub(crate) fn translate_function_calls_to_file_search(response: &mut Value) -> Vec<usize> {
    let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) else {
        return Vec::new();
    };

    let mut translated = Vec::new();
    for (index, item) in output.iter_mut().enumerate() {
        if !is_file_search_function_call(item) {
            continue;
        }
        if let Some(object) = item.as_object_mut() {
            rewrite_function_call_as_file_search(object);
            translated.push(index);
        }
    }
    translated
}

/// Rewrite one `function_call(name="file_search")` object into a canonical
/// `file_search_call` in place, preserving a stable public id derived from the
/// model's `call_id` when the item arrived without one (issue #955).
fn rewrite_function_call_as_file_search(object: &mut serde_json::Map<String, Value>) {
    let queries = object
        .get("arguments")
        .and_then(Value::as_str)
        .map(extract_file_search_queries)
        .unwrap_or_default();

    let call_id = object.get("call_id").and_then(Value::as_str).map(ToOwned::to_owned);

    object.insert("type".to_owned(), Value::String("file_search_call".to_owned()));
    object.insert("status".to_owned(), Value::String("searching".to_owned()));
    object.insert(
        "queries".to_owned(),
        Value::Array(queries.into_iter().map(Value::String).collect()),
    );

    if object.get("id").and_then(Value::as_str).is_none_or(str::is_empty)
        && let Some(call_id) = &call_id
    {
        object.insert("id".to_owned(), Value::String(format!("fs_{call_id}")));
    }

    object.remove("name");
    object.remove("arguments");
    object.remove("call_id");
}

/// Build the standard Responses bridge carrying private model context.
fn model_context_messages(
    source_id: &str,
    output_index: usize,
    response_identity_hash: u64,
    query: &str,
    output: &str,
) -> [Value; 2] {
    let fallback_id = output_index.to_string();
    let source_id = if source_id.is_empty() { &fallback_id } else { source_id };
    let call_hash = stable_call_hash_with_seed(response_identity_hash, &[source_id, query]);
    let call_id = format!("file_search_{output_index}_{call_hash:016x}");
    let arguments = serde_json::json!({ "query": query }).to_string();
    [
        serde_json::json!({
            "type": "function_call",
            "call_id": &call_id,
            "name": "file_search",
            "arguments": arguments,
            "status": "completed",
        }),
        serde_json::json!({
            "type": "function_call_output",
            "call_id": &call_id,
            "output": output,
        }),
    ]
}

/// Build a deterministic bounded identity for one synthetic bridge.
const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;

/// Build a deterministic bounded identity for one synthetic bridge.
fn stable_call_hash(parts: &[&str]) -> u64 {
    stable_call_hash_with_seed(FNV_OFFSET_BASIS, parts)
}

/// Extend a pre-hashed response identity with bounded per-call fields.
fn stable_call_hash_with_seed(mut hash: u64, parts: &[&str]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    for part in parts {
        for byte in part.as_bytes().iter().copied().chain(std::iter::once(0xFF)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Assign stable synthetic IDs to every id-less public output item in
/// `response["output"]`.
///
/// vLLM and some providers emit output items without an `id`, but every public
/// item must carry a stable identifier. Derive a per-response seed from
/// `response["id"]` and stamp any missing/empty id in place (issue #955). The
/// loop owner runs this after normalization and before accumulation so both the
/// buffered and streaming paths share one implementation.
pub(crate) fn ensure_public_output_item_ids_in_response(response: &mut Value) {
    let response_identity_hash = response
        .get("id")
        .and_then(Value::as_str)
        .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
    if let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) {
        ensure_public_output_item_ids(output, response_identity_hash);
    }
}

/// Normalize missing or malformed provider IDs for every public output item.
fn ensure_public_output_item_ids(items: &mut [Value], response_identity_hash: u64) {
    for (output_index, item) in items.iter_mut().enumerate() {
        let Some(object) = item.as_object_mut() else {
            continue;
        };
        let valid_id = object
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty());
        if !valid_id {
            let prefix = match object.get("type").and_then(Value::as_str) {
                Some("file_search_call") => "fs",
                Some("web_search_call") => "ws",
                Some("function_call") => "fc",
                Some("message") => "msg",
                Some("reasoning") => "rs",
                _ => "item",
            };
            object.insert(
                "id".to_owned(),
                Value::String(format!("{prefix}_{response_identity_hash:016x}_{output_index}")),
            );
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::bool_assert_comparison,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing,
    clippy::needless_raw_string_hashes,
    clippy::needless_raw_strings,
    clippy::panic,
    clippy::redundant_closure,
    clippy::significant_drop_tightening,
    clippy::str_to_string,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests;
