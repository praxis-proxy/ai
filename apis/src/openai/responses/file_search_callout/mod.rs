// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! OGX-backed file search execution for the OpenAI Responses API.
//!
//! The filter is registered and continuation-ready. On an ordinary first pass
//! it is a no-op because response output items do not exist until inference has
//! completed. A continuation or agentic-loop owner can re-enter the request
//! phase with those items populated, at which point this filter executes a
//! bounded set of pending `file_search_call` items and terminalizes any excess.

pub(crate) mod citations;
pub(crate) mod client;
mod config;

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use praxis_callout_core::callout::FailureMode;
use praxis_filter::{
    FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use self::{
    citations::{FormatLimits, FormatTemplates, format_search_results},
    client::{
        FileSearchClient, FileSearchClientConfig, MAX_QUERY_BYTES, MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES,
        SearchBatch, SearchFailure, SearchSpec, request_error,
    },
    config::{FileSearchFilterConfig, build_config},
};
use crate::openai::responses::{
    bounded_json_size,
    error::responses_error_rejection,
    state::{MAX_CITATION_FILES, ResponsesState},
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

/// Executes pending file search calls against an OGX vector store API.
///
/// First-pass requests remain unchanged until core continuation support can
/// re-enter the request phase with pending file-search calls. Re-entered
/// streaming requests are rejected because citation markers require an
/// incremental SSE transformer.
pub struct FileSearchCalloutFilter {
    /// Per-result formatting template.
    annotation_template: String,

    /// Callout client for the vector store API.
    client: FileSearchClient,

    /// Model-facing context template.
    context_template: String,

    /// Whether a failed callout rejects or produces an incomplete result.
    failure_mode: FailureMode,
}

impl FileSearchCalloutFilter {
    /// Create a filter from parsed YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration or callout client
    /// construction fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: FileSearchFilterConfig = parse_filter_config("openai_file_search_callout", config)?;
        let validated = build_config(cfg)?;
        let client = FileSearchClient::new(FileSearchClientConfig {
            api_client: validated.api_client,
            authorization: validated.authorization,
            failure_mode: validated.failure_mode,
            max_response_bytes: validated.max_response_bytes,
            max_total_response_bytes: validated.max_total_response_bytes,
            search_template: validated.search_template,
            timeout: validated.timeout,
        });

        Ok(Box::new(Self {
            annotation_template: validated.annotation_template,
            client,
            context_template: validated.context_template,
            failure_mode: validated.failure_mode,
        }))
    }

    /// Apply one completed search batch to request-scoped response state.
    #[expect(clippy::too_many_lines, reason = "sequential result formatting and state commit")]
    fn apply_batch(
        &self,
        state: &mut ResponsesState,
        plan: &SearchPlan,
        batch: &SearchBatch,
    ) -> Result<(), FilterAction> {
        let failed_calls: HashSet<usize> = batch.failures.iter().map(|failure| failure.call_index).collect();
        let expose_results = state.include.iter().any(|value| value == "file_search_call.results");
        let templates = FormatTemplates {
            annotation: &self.annotation_template,
            context: &self.context_template,
        };
        let mut model_messages = Vec::with_capacity(plan.calls.len().saturating_mul(2));
        let mut remaining_model_bytes = MAX_TOTAL_MODEL_CONTEXT_BYTES;
        let response_identity_hash = state
            .response_object
            .get("id")
            .and_then(Value::as_str)
            .map_or(FNV_OFFSET_BASIS, |response_id| stable_call_hash(&[response_id]));
        ensure_pending_file_search_call_ids(state, response_identity_hash);

        for (call_index, call) in plan.calls.iter().enumerate() {
            let results = batch.results_by_call.get(call_index).map_or(&[][..], Vec::as_slice);
            let (query, query_truncated) = join_queries_bounded(&call.queries);
            let Some(source_item) = state.output_items().get(call.output_index) else {
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
                templates: &templates,
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
            if let Some(item) = state.output_items_mut().get_mut(call.output_index)
                && let Some(object) = item.as_object_mut()
            {
                object.insert("status".to_owned(), Value::String(status.to_owned()));
                if expose_results {
                    object.insert("results".to_owned(), Value::Array(public_results));
                } else {
                    object.remove("results");
                }

                if let Some(messages) = call_model_messages {
                    model_messages.extend(messages);
                }
                applied = true;
            }
            if applied {
                state.citation_files.extend(citation_files);
            }
        }

        terminalize_unplanned_pending_calls(state, plan);
        if !response_fits(state, MAX_JSON_BODY_BYTES) {
            return Err(FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                "openai_file_search_callout: continuation output exceeds the JSON response byte limit",
                false,
            )));
        }
        state.messages.extend(model_messages);
        Ok(())
    }

    /// Execute the bounded fan-out for a completed plan.
    #[expect(
        clippy::too_many_lines,
        reason = "separates global, per-call, and transport planning failures"
    )]
    async fn execute_plan(&self, plan: &SearchPlan) -> SearchBatch {
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
            self.client.search(&specs, plan.calls.len()).await
        };
        batch.failures.extend(planning_failures);
        batch
    }

    /// Build a fail-closed rejection after logging every failed search.
    fn failure_rejection(&self, batch: &SearchBatch) -> Option<FilterAction> {
        for failure in &batch.failures {
            warn!(
                call_index = failure.call_index,
                error = %failure.error,
                "vector store search failed"
            );
        }
        let failure = (self.failure_mode == FailureMode::Closed)
            .then(|| batch.failures.first())
            .flatten()?;
        Some(FilterAction::Reject(responses_error_rejection(
            502,
            "server_error",
            &format!("openai_file_search_callout: {}", failure.error),
            false,
        )))
    }
}

#[async_trait]
impl HttpFilter for FileSearchCalloutFilter {
    fn name(&self) -> &'static str {
        "openai_file_search_callout"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        if let Some(rejection) = unsupported_streaming_rejection(ctx, state) {
            return Ok(rejection);
        }

        let plan = build_search_plan(state);
        if !plan.has_pending_calls {
            return Ok(FilterAction::Continue);
        }

        debug!(
            pending_calls = plan.calls.len(),
            scheduled_searches = plan.spec_coordinates.len(),
            "executing file search callouts"
        );

        let batch = self.execute_plan(&plan).await;
        if let Some(rejection) = self.failure_rejection(&batch) {
            return Ok(rejection);
        }

        let state = ctx
            .extensions
            .get_mut::<ResponsesState>()
            .ok_or_else(|| -> FilterError { "openai_file_search_callout: ResponsesState disappeared".into() })?;
        if let Err(rejection) = self.apply_batch(state, &plan, &batch) {
            return Ok(rejection);
        }

        Ok(FilterAction::Continue)
    }
}

/// Owned execution plan, independent from request state during callouts.
struct SearchPlan {
    /// Pending calls in response output order.
    calls: Vec<PendingCall>,

    /// Metadata filters shared by every search spec.
    filters: Option<Value>,

    /// Whether the response contained any pending call before local caps.
    has_pending_calls: bool,

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
    /// Metadata filter passed through to OGX.
    filters: Option<Value>,

    /// Maximum number of aggregate results.
    max_num_results: Option<u64>,

    /// Ranking options passed through to OGX.
    ranking_options: Option<Value>,

    /// Structural or resource error in a required execution field.
    planning_error: Option<&'static str>,

    /// Vector stores to search.
    vector_store_ids: Vec<String>,

    /// Number of string vector-store IDs before retention limits.
    vector_store_count: usize,
}

/// Build an owned plan for every pending call before applying the fan-out cap.
fn build_search_plan(state: &ResponsesState) -> SearchPlan {
    let tool = extract_file_search_tool_def(&state.tools);
    let has_pending_calls = state.output_items().iter().any(is_pending_file_search_call);
    let call_budget = remaining_file_search_call_budget(state);
    let mut calls = pending_calls(state, tool.vector_store_count, call_budget);
    let spec_coordinates = schedule_searches(&mut calls, tool.vector_store_ids.len());

    SearchPlan {
        calls,
        filters: tool.filters,
        has_pending_calls,
        max_num_results: tool.max_num_results,
        planning_error: tool.planning_error,
        ranking_options: tool.ranking_options,
        spec_coordinates,
        vector_store_ids: tool.vector_store_ids,
    }
}

/// Extract every pending output call before applying scheduling limits.
#[expect(
    clippy::too_many_lines,
    reason = "bounded structural validation and ownership happen together"
)]
fn pending_calls(state: &ResponsesState, store_count: usize, call_budget: usize) -> Vec<PendingCall> {
    let mut calls = Vec::new();
    for (output_index, item) in state
        .output_items()
        .iter()
        .enumerate()
        .filter(|(_, item)| is_pending_file_search_call(item))
        .take(MAX_PENDING_CALLS.min(call_budget))
    {
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

/// Resolve the remaining client-declared built-in tool call allowance.
fn remaining_file_search_call_budget(state: &ResponsesState) -> usize {
    let Some(max_tool_calls) = state.max_tool_calls else {
        return MAX_PENDING_CALLS;
    };
    let used_calls = state
        .output_items()
        .iter()
        .filter(|item| is_builtin_tool_call(item) && !is_pending_file_search_call(item))
        .count();
    usize::try_from(max_tool_calls)
        .unwrap_or(usize::MAX)
        .saturating_sub(used_calls)
}

/// Return whether an output item is a provider-hosted built-in tool call.
fn is_builtin_tool_call(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some(
            "apply_patch_call"
                | "code_interpreter_call"
                | "computer_call"
                | "file_search_call"
                | "image_generation_call"
                | "local_shell_call"
                | "multi_agent_call"
                | "shell_call"
                | "tool_search_call"
                | "web_search_call"
        )
    )
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

/// Reject only a streaming continuation that is ready to execute file search.
fn unsupported_streaming_rejection(ctx: &HttpFilterContext<'_>, state: &ResponsesState) -> Option<FilterAction> {
    let streaming = ctx
        .get_metadata("openai_responses_format.stream")
        .is_some_and(|value| value == "true");
    let pending = state.output_items().iter().any(is_pending_file_search_call);
    (streaming && (pending || !state.citation_files.is_empty())).then(|| {
        FilterAction::Reject(responses_error_rejection(
            400,
            "invalid_request_error",
            "openai_file_search_callout: stream=true is not supported because file citation markers require SSE transformation",
            true,
        ))
    })
}

/// Return whether one output item still requires local file-search execution.
fn is_pending_file_search_call(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("file_search_call")
        && matches!(
            item.get("status").and_then(Value::as_str),
            Some("searching" | "in_progress")
        )
}

/// Mark pending calls that could not be scheduled as incomplete.
fn terminalize_unplanned_pending_calls(state: &mut ResponsesState, plan: &SearchPlan) {
    for (output_index, item) in state.output_items_mut().iter_mut().enumerate() {
        if !is_pending_file_search_call(item)
            || plan
                .calls
                .binary_search_by_key(&output_index, |call| call.output_index)
                .is_ok()
        {
            continue;
        }
        if let Some(object) = item.as_object_mut() {
            object.insert("status".to_owned(), Value::String("incomplete".to_owned()));
            object.remove("results");
        }
    }
}

/// Give every pending call its final public identity before budgeting or capping.
fn ensure_pending_file_search_call_ids(state: &mut ResponsesState, response_identity_hash: u64) {
    for (output_index, item) in state.output_items_mut().iter_mut().enumerate() {
        if is_pending_file_search_call(item)
            && let Some(object) = item.as_object_mut()
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

/// Return whether replacing the response output remains within its hard body ceiling.
fn response_fits(state: &ResponsesState, max_bytes: usize) -> bool {
    bounded_json_size(&state.response_object, max_bytes)
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

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::needless_raw_string_hashes,
    clippy::needless_raw_strings,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests;
