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
        MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES, SearchBatch, SearchFailure, SearchSpec, request_error,
    },
    config::{FileSearchFilterConfig, ValidatedConfig, build_config_with_client, require_inline_outbound_chain},
    model_context::{FormatLimits, FormatTemplates, MODEL_CONTEXT_TEMPLATES, format_search_results},
};
use crate::{
    callout_headers::effective_body_callout_headers,
    callout_policy::OnFailure,
    http_hop::connection_nominates_header,
    openai::{
        responses::{
            bounded_json_size,
            error::responses_error_rejection,
            state::{DispatchFailure, FileSearchAssignment, MAX_CITATION_FILES, ResponsesState},
        },
        translation::chat_completions::{
            TranslationError, responses_file_search_tool_choice_lowering, synthesized_file_search_tool_responses,
            validate_file_search_tools,
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
    #[expect(clippy::too_many_lines, reason = "sequential result formatting and state commit")]
    fn apply_batch(state: &mut ResponsesState, plan: &SearchPlan, batch: &SearchBatch) -> Result<(), DispatchFailure> {
        let failed_calls: HashSet<usize> = batch.failures.iter().map(|failure| failure.call_index).collect();
        let expose_results = state.include.iter().any(|value| value == "file_search_call.results");
        let mut bridges = Vec::with_capacity(plan.calls.len());
        let mut remaining_model_bytes = MAX_TOTAL_MODEL_CONTEXT_BYTES;
        let response_identity_hash = state
            .response_object
            .get("id")
            .and_then(Value::as_str)
            .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
        ensure_pending_file_search_call_ids(state, plan, response_identity_hash);

        for (call_index, call) in plan.calls.iter().enumerate() {
            let results = batch.results_by_call.get(call_index).map_or(&[][..], Vec::as_slice);
            let (query, query_truncated) = join_queries_bounded(&call.queries);
            let Some(source_item) = state.accumulated_output.get(call.output_index) else {
                continue;
            };
            let BudgetedSearchResults {
                citation_files,
                model_messages: call_model_messages,
                public_results,
                serialized_bytes,
                truncated,
            } = BridgeBudget {
                known_citation_files: &state.citation_files,
                max_new_citation_files: MAX_CITATION_FILES.saturating_sub(state.citation_files.len()),
                remaining_model_bytes,
                source_item,
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

            let mut applied = false;
            if let Some(item) = state.accumulated_output.get_mut(call.output_index)
                && let Some(object) = item.as_object_mut()
            {
                object.insert("status".to_owned(), Value::String(status.to_owned()));
                if expose_results {
                    object.insert("results".to_owned(), Value::Array(public_results));
                } else {
                    object.remove("results");
                }

                if let Some(messages) = call_model_messages {
                    bridges.push(messages);
                }
                applied = true;
            }
            if applied {
                state.citation_files.extend(citation_files);
            }
        }

        if !accumulated_output_fits(state, MAX_JSON_BODY_BYTES) {
            return Err(DispatchFailure {
                status: 502,
                code: "server_error",
                message: "openai_file_search_callout: continuation output exceeds the JSON response byte limit"
                    .to_owned(),
            });
        }
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
        } else {
            self.client
                .search(&specs, plan.calls.len(), request_headers, transport)
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
        let plan = build_search_plan(state, &assignments);
        let hdrs = callout_request_headers(ctx);
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
        let batch = self.execute_plan(&plan, &hdrs, &transport).await;
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
        // Terminalize any assignments the per-continuation server cap dropped
        // from the plan, then reconcile the executed calls in place.
        terminalize_unplanned_pending_calls(state, &assignments, &plan);
        if let Err(failure) = Self::apply_batch(state, &plan, &batch) {
            state.dispatch_failure = Some(failure);
            return Ok(FilterAction::Continue);
        }
        if !continuation_state_fits(framework_bytes, state, self.max_state_bytes, 0) {
            state.dispatch_failure = Some(continuation_state_dispatch_failure());
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
        // Lower a hosted file_search tool into a private function before dispatch
        // so a native `/v1/responses` backend that cannot consume hosted tools
        // still runs the search. This mutates only the outbound body; the hosted
        // configuration the dispatcher reads stays in `state.tools`.
        if let Some(state) = ctx.extensions.get_mut::<ResponsesState>()
            && let Err(rejection) = lower_native_file_search(state)
        {
            return Ok(rejection);
        }
        self.dispatch(ctx).await
    }
}

/// Lower a hosted Responses `file_search` tool into a private function for a
/// native `/v1/responses` backend that cannot consume hosted tools.
///
/// Runs once at request-body EOS before dispatch. Self-gating and idempotent: it
/// fires only while `request_body["tools"]` still carries a hosted
/// `{"type":"file_search"}` entry, so continuation rounds (already lowered) and
/// requests without hosted file search are no-ops. It mutates only
/// `state.request_body` — the outbound body `openai_responses_proxy` serializes —
/// and leaves `state.tools`/`state.tool_choice` holding the hosted configuration
/// the dispatcher and response normalizer read. Rejections reuse the Chat
/// Completions translation's validation so both backends reject the same
/// malformed requests.
fn lower_native_file_search(state: &mut ResponsesState) -> Result<(), FilterAction> {
    if !request_body_has_hosted_file_search(state) {
        return Ok(());
    }
    validate_file_search_tools(&state.tools).map_err(|error| reject_file_search(&error))?;
    // Resolve the lowered choice before mutating, so a rejected choice leaves the
    // outbound body untouched (no half-lowered request).
    let lowered_choice = match state.request_body.get("tool_choice") {
        Some(choice) => {
            responses_file_search_tool_choice_lowering(choice).map_err(|error| reject_file_search(&error))?
        },
        None => None,
    };
    if let Some(choice) = lowered_choice {
        set_request_body_tool_choice(state, choice);
    }
    lower_request_body_file_search_tools(state);
    state.mark_request_body_for_rebuild();
    Ok(())
}

/// Map a Chat-translation file-search error to a `400` rejection so the native
/// and Chat paths reject identical malformed declarations.
fn reject_file_search(error: &TranslationError) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        400,
        "invalid_request_error",
        &error.to_string(),
    ))
}

/// True while the outbound request body still carries a hosted `file_search` tool.
fn request_body_has_hosted_file_search(state: &ResponsesState) -> bool {
    state
        .request_body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_hosted_file_search_tool))
}

/// True for a hosted `{"type":"file_search"}` tool declaration.
fn is_hosted_file_search_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("file_search")
}

/// Replace every hosted `file_search` entry in the outbound `tools` with the
/// private Responses function, preserving order and any client tools.
fn lower_request_body_file_search_tools(state: &mut ResponsesState) {
    let Some(tools) = state.request_body.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };
    for tool in tools.iter_mut() {
        if is_hosted_file_search_tool(tool) {
            *tool = synthesized_file_search_tool_responses();
        }
    }
}

/// Overwrite the outbound `tool_choice` with the lowered function choice.
fn set_request_body_tool_choice(state: &mut ResponsesState, choice: Value) {
    if let Some(object) = state.request_body.as_object_mut() {
        object.insert("tool_choice".to_owned(), choice);
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
#[expect(
    clippy::too_many_lines,
    reason = "accounts each retained state field without allocation"
)]
fn continuation_state_fits(
    framework_bytes: usize,
    state: &ResponsesState,
    max_bytes: usize,
    incoming_bytes: usize,
) -> bool {
    let mut used = framework_bytes.saturating_add(incoming_bytes);
    for value in [
        &state.request_body,
        &state.response_object,
        &state.local_completion_response_template,
        &state.tool_choice,
        &state.usage,
    ] {
        let Some(size) = bounded_json_size(value, max_bytes.saturating_sub(used)).ok().flatten() else {
            return false;
        };
        used = used.saturating_add(size);
    }
    for values in [
        &state.accumulated_output,
        &state.input,
        &state.messages,
        &state.persisted_messages,
        &state.previous_tools,
        &state.tool_calls,
        &state.tools,
        &state.web_search_calls,
    ] {
        let Some(size) = bounded_json_size(values, max_bytes.saturating_sub(used)).ok().flatten() else {
            return false;
        };
        used = used.saturating_add(size);
    }
    for value in [
        state.context_management.as_ref(),
        state.conversation.as_ref(),
        state.original_tool_choice.as_ref(),
        state.previous_usage.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let Some(size) = bounded_json_size(value, max_bytes.saturating_sub(used)).ok().flatten() else {
            return false;
        };
        used = used.saturating_add(size);
    }
    let string_bytes = state
        .citation_files
        .iter()
        .map(|(key, value)| key.len().saturating_add(value.len()))
        .chain(state.include.iter().map(String::len))
        .chain(state.previous_response_id.iter().map(String::len))
        .chain(state.response_id.iter().map(String::len))
        .chain(
            state
                .mcp_tool_map
                .iter()
                .map(|((server, tool), _)| server.len().saturating_add(tool.len())),
        )
        // #313 P1: charge the per-round provider-streamed observation set against the same
        // ceiling as every other request-scoped field, so one round streaming many distinct
        // native ids cannot bypass max_state_bytes (finalize clears it between rounds).
        .chain(state.provider_streamed_terminal_ids.iter().map(String::len))
        // #1131: charge the client-tool lowering reverse map (private name -> original name,
        // plus the optional restored namespace) against the same ceiling; `ClientToolRestore`
        // is a 1-byte `Copy` tag with no owned payload, so only the strings need accounting.
        .chain(state.client_tool_lowering.iter().map(|(name, lowered)| {
            name.len()
                .saturating_add(lowered.original_name.len())
                .saturating_add(lowered.namespace.as_ref().map_or(0, String::len))
        }))
        .fold(0_usize, usize::saturating_add);
    used = used.saturating_add(string_bytes);
    for value in state.mcp_tool_map.values() {
        let Some(size) = bounded_json_size(value, max_bytes.saturating_sub(used)).ok().flatten() else {
            return false;
        };
        used = used.saturating_add(size);
    }
    // #1131: charge the pre-lowering echo snapshot (the client's original `tools`
    // array and `tool_choice`, restored onto the echoed response) against the same
    // ceiling as every other request-scoped field.
    if let Some(echo) = state.client_tool_echo.as_ref() {
        let Some(size) = bounded_json_size(&echo.tools, max_bytes.saturating_sub(used))
            .ok()
            .flatten()
        else {
            return false;
        };
        used = used.saturating_add(size);
        let Some(size) = bounded_json_size(&echo.tool_choice, max_bytes.saturating_sub(used))
            .ok()
            .flatten()
        else {
            return false;
        };
        used = used.saturating_add(size);
    }
    used <= max_bytes
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

    /// Maximum new mappings this call may retain.
    max_new_citation_files: usize,

    /// Remaining compact JSON bytes for immediate model messages.
    remaining_model_bytes: usize,

    /// Response item used to derive a deterministic bridge identity.
    source_item: &'a Value,

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

impl BridgeBudget<'_> {
    /// Reserve exact structural and metadata bytes, then render chunks once.
    #[expect(clippy::too_many_lines, reason = "one ordered format and exact-budget transaction")]
    fn format(self, results: &[client::SearchResult], include_public_results: bool) -> BudgetedSearchResults {
        let empty_model_messages = model_context_messages(
            self.source_item,
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
                include_public_results,
            },
        );
        let model_messages = model_context_messages(
            self.source_item,
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

/// Mark assigned pending calls the per-continuation server cap dropped as incomplete.
///
/// The loop owner recorded one [`FileSearchAssignment`] per pending call; the
/// dispatcher only planned the first `MAX_PENDING_CALLS` of them.
/// Any assignment whose absolute [`FileSearchAssignment::output_index`] did not
/// enter `plan.calls` is terminalized in place inside `accumulated_output`.
fn terminalize_unplanned_pending_calls(
    state: &mut ResponsesState,
    assignments: &[FileSearchAssignment],
    plan: &SearchPlan,
) {
    let response_identity_hash = state
        .response_object
        .get("id")
        .and_then(Value::as_str)
        .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
    for assignment in assignments {
        let output_index = assignment.output_index;
        if plan.calls.iter().any(|call| call.output_index == output_index) {
            continue;
        }
        if let Some(object) = state
            .accumulated_output
            .get_mut(output_index)
            .and_then(Value::as_object_mut)
        {
            // A terminalized call still reaches the public output, so it needs a
            // valid id exactly like the planned calls (`ensure_pending_file_search_call_ids`).
            ensure_public_file_search_call_id(object, output_index, response_identity_hash);
            object.insert("status".to_owned(), Value::String("incomplete".to_owned()));
            object.remove("results");
        }
    }
}

/// Give every planned pending call its final public identity before budgeting.
fn ensure_pending_file_search_call_ids(state: &mut ResponsesState, plan: &SearchPlan, response_identity_hash: u64) {
    for call in &plan.calls {
        let output_index = call.output_index;
        if let Some(object) = state
            .accumulated_output
            .get_mut(output_index)
            .and_then(Value::as_object_mut)
        {
            ensure_public_file_search_call_id(object, output_index, response_identity_hash);
        }
    }
}

/// Build the standard Responses bridge carrying private model context.
fn model_context_messages(
    item: &Value,
    output_index: usize,
    response_identity_hash: u64,
    query: &str,
    output: &str,
) -> [Value; 2] {
    let fallback_id = output_index.to_string();
    let source_id = item.get("id").and_then(Value::as_str).unwrap_or(&fallback_id);
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

/// Return whether the reconciled public output remains within its hard body ceiling.
///
/// The loop owner serializes [`ResponsesState::accumulated_output`] as the public
/// `output`, so the dispatcher charges its post-reconciliation size against the
/// same JSON response ceiling before returning control.
fn accumulated_output_fits(state: &ResponsesState, max_bytes: usize) -> bool {
    bounded_json_size(&state.accumulated_output, max_bytes)
        .ok()
        .flatten()
        .is_some()
}

/// Normalize a malformed provider call ID without changing valid opaque IDs.
fn ensure_public_file_search_call_id(
    item: &mut serde_json::Map<String, Value>,
    output_index: usize,
    response_identity_hash: u64,
) {
    let valid_id = item.get("id").and_then(Value::as_str).is_some_and(|id| !id.is_empty());
    if !valid_id {
        item.insert(
            "id".to_owned(),
            Value::String(format!("fs_{response_identity_hash:016x}_{output_index}")),
        );
    }
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
