// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vector store file search execution for the OpenAI Responses API.
//!
//! The filter runs inside `iterative_request_router` so a model response that
//! contains `file_search_call` output can trigger a vector store search and another model
//! inference within the same client request. Search context remains private to
//! the model round trip; completed call items and citations are assembled into
//! the final public Responses API object.

pub(crate) mod citations;
pub(crate) mod client;
mod config;
mod model_context;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
};

use async_trait::async_trait;
use bytes::Bytes;
use http::HeaderMap;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState,
    body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use serde::{Deserialize, de::SeqAccess};
use serde_json::Value;
use tracing::{debug, warn};

use self::{
    citations::annotate_response,
    client::{
        FileSearchClient, FileSearchClientConfig, MAX_QUERY_BYTES, MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES,
        SearchBatch, SearchFailure, SearchSpec, request_error,
    },
    config::{FileSearchFilterConfig, ValidatedConfig, build_config, build_config_with_client},
    model_context::{FormatLimits, FormatTemplates, MODEL_CONTEXT_TEMPLATES, format_search_results},
};
use crate::{
    openai::responses::{
        bounded_json_size,
        config_validation::FailureMode,
        error::responses_error_rejection,
        state::{MAX_CITATION_FILES, ResponsesState},
        usage::merge_usage,
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

/// Executes pending file search calls against a vector store API compatible backend.
///
/// The enclosing iterative router owns model re-entry. Streaming requests are
/// rejected because citation markers require an incremental SSE transformer.
/// Search queries are forwarded unchanged; model context and citation
/// marker formatting are internal.
pub struct FileSearchCalloutFilter {
    /// Callout client for the vector store API.
    client: FileSearchClient,

    /// Combined router and filter continuation-state ceiling.
    max_state_bytes: usize,

    /// Whether a failed callout rejects or produces an incomplete result.
    failure_mode: FailureMode,
}

/// Request-local marker used to reject streaming before the first subrequest.
struct StreamingRequest;

impl FileSearchCalloutFilter {
    /// Create a filter from parsed YAML configuration.
    ///
    /// Falls back to a dedicated per-filter sub-request connector with a
    /// pool size of 4. Prefer [`from_config_with_client`] when a shared
    /// server-level sub-request client is available.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration or callout client
    /// construction fails.
    ///
    /// [`from_config_with_client`]: Self::from_config_with_client
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: FileSearchFilterConfig = parse_filter_config("openai_file_search_callout", config)?;
        let validated = build_config(&cfg)?;
        Ok(Self::build(validated))
    }

    /// Create a filter using a shared sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when configuration or callout client
    /// construction fails.
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: FileSearchFilterConfig = parse_filter_config("openai_file_search_callout", config)?;
        let validated = build_config_with_client(&cfg, client)?;
        Ok(Self::build(validated))
    }

    /// Assemble a filter from validated config.
    fn build(validated: ValidatedConfig) -> Box<dyn HttpFilter> {
        let client = FileSearchClient::new(FileSearchClientConfig {
            api_client: validated.api_client,
            failure_mode: validated.failure_mode,
            max_response_bytes: validated.max_response_bytes,
            max_total_response_bytes: validated.max_total_response_bytes,
            timeout: validated.timeout,
        });

        Box::new(Self {
            client,
            max_state_bytes: validated.max_state_bytes,
            failure_mode: validated.failure_mode,
        })
    }

    /// Apply one completed search batch to request-scoped response state.
    #[expect(clippy::too_many_lines, reason = "sequential result formatting and state commit")]
    fn apply_batch(state: &mut ResponsesState, plan: &SearchPlan, batch: &SearchBatch) -> Result<(), FilterAction> {
        let failed_calls: HashSet<usize> = batch.failures.iter().map(|failure| failure.call_index).collect();
        let expose_results = state.include.iter().any(|value| value == "file_search_call.results");
        let mut bridges = Vec::with_capacity(plan.calls.len());
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
                    bridges.push((call.output_index, messages));
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
        state
            .messages
            .extend(continuation_replay_items(state.output_items(), bridges));
        Ok(())
    }

    /// Execute the bounded fan-out for a completed plan.
    #[expect(
        clippy::too_many_lines,
        reason = "separates global, per-call, and transport planning failures"
    )]
    async fn execute_plan(&self, plan: &SearchPlan, request_headers: &HeaderMap) -> SearchBatch {
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
            self.client.search(&specs, plan.calls.len(), request_headers).await
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

    /// Execute pending calls before the next inference body is serialized.
    #[expect(
        clippy::too_many_lines,
        reason = "linear sequence: plan → callout → apply → size check"
    )]
    async fn execute_pending(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if let Some(rejection) = unsupported_streaming_rejection(ctx) {
            return Ok(rejection);
        }
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        let plan = build_search_plan(state);
        if !plan.has_pending_calls {
            return Ok(FilterAction::Continue);
        }
        let hdrs = callout_request_headers(ctx);
        let batch = self.execute_plan(&plan, &hdrs).await;
        if let Some(rejection) = self.failure_rejection(&batch) {
            return Ok(rejection);
        }
        let framework_bytes = retained_iteration_bytes(ctx);
        let state = ctx
            .extensions
            .get_mut::<ResponsesState>()
            .ok_or_else(|| -> FilterError { "openai_file_search_callout: ResponsesState disappeared".into() })?;
        if let Err(rejection) = Self::apply_batch(state, &plan, &batch) {
            return Ok(rejection);
        }
        if !continuation_state_fits(framework_bytes, state, self.max_state_bytes, 0) {
            return Ok(continuation_state_rejection());
        }
        reset_tool_choice(state);
        state.iteration = state.iteration.saturating_add(1);
        Ok(FilterAction::Continue)
    }

    /// Capture a model response and expose whether another inference is needed.
    #[expect(
        clippy::too_many_lines,
        reason = "response state commit and final assembly are sequential"
    )]
    fn capture_response(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        max_state_bytes: usize,
    ) -> Result<FilterAction, FilterError> {
        let is_success = ctx
            .response_header
            .as_deref()
            .is_some_and(|response| response.status.is_success());
        if !is_success {
            return Ok(FilterAction::Continue);
        }

        let continued = ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
            state.iteration != 0 || !state.file_search_output_items.is_empty() || !state.citation_files.is_empty()
        });
        let Some(bytes) = body.as_ref() else {
            return Ok(invalid_success_response_action(continued));
        };
        let framework_bytes = retained_iteration_bytes(ctx);
        let response_wire_bytes = bytes.len();
        if let Some(state) = ctx.extensions.get::<ResponsesState>()
            && !continuation_state_fits(
                framework_bytes,
                state,
                max_state_bytes,
                response_wire_bytes.saturating_mul(2),
            )
        {
            return Ok(continuation_state_rejection());
        }
        let Ok(mut response) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(invalid_success_response_action(continued));
        };
        if !response.is_object() {
            return Ok(invalid_success_response_action(continued));
        }
        let has_file_search_tool = ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
            state
                .tools
                .iter()
                .any(|tool| tool.get("type").and_then(Value::as_str) == Some("file_search"))
        });
        if has_file_search_tool {
            let translated = translate_function_calls_to_file_search(&mut response);
            if translated > 0 {
                debug!(
                    count = translated,
                    "translated function_call(name=file_search) to file_search_call"
                );
            }
        }
        let Some(output) = response.get("output").and_then(Value::as_array) else {
            return Ok(invalid_success_response_action(continued));
        };
        if output.iter().any(is_pending_file_search_call) && has_client_function_call(output) {
            return Ok(mixed_tool_response_rejection());
        }

        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };
        if let Some(usage) = response.get("usage").filter(|usage| !usage.is_null()) {
            merge_usage(&mut state.usage, usage);
        }
        if !state.usage.is_null()
            && let Some(object) = response.as_object_mut()
        {
            object.insert("usage".to_owned(), state.usage.clone());
        }
        if !combined_output_fits(state, &response, MAX_JSON_BODY_BYTES) {
            return Ok(FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                "openai_file_search_callout: accumulated output exceeds the JSON response byte limit",
                false,
            )));
        }
        let completed_output = std::mem::take(state.output_items_mut());
        state.file_search_output_items.extend(completed_output);
        state.response_object = response;
        if !continuation_state_fits(framework_bytes, state, max_state_bytes, response_wire_bytes) {
            return Ok(continuation_state_rejection());
        }

        if remaining_file_search_call_budget(state) == 0 {
            terminalize_all_pending_calls(state);
        }
        if state.output_items().iter().any(is_pending_file_search_call) {
            ctx.filter_results
                .entry("openai_file_search_callout")
                .or_default()
                .set("pending", "true")?;
            return Ok(FilterAction::Continue);
        }

        let encoded = match finalize_public_response(state) {
            Ok(encoded) => encoded,
            Err(rejection) => return Ok(rejection),
        };
        clear_rewritten_response_headers(ctx);
        *body = Some(encoded);
        ctx.filter_results
            .entry("openai_file_search_callout")
            .or_default()
            .set("pending", "false")?;

        Ok(FilterAction::Continue)
    }
}

#[async_trait]
impl HttpFilter for FileSearchCalloutFilter {
    fn name(&self) -> &'static str {
        "openai_file_search_callout"
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
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let action = self.execute_pending(ctx).await?;
        if matches!(action, FilterAction::Continue) {
            preserve_original_request_headers(ctx);
            ctx.request_headers_to_set.push((
                http::header::ACCEPT_ENCODING,
                http::HeaderValue::from_static("identity"),
            ));
        }
        Ok(action)
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
        if let Some(rejection) = initialize_file_search_state(ctx, body, self.max_state_bytes) {
            return Ok(rejection);
        }
        self.execute_pending(ctx).await
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        Self::capture_response(ctx, body, self.max_state_bytes)
    }
}

/// On IRR continuation iterations the synthetic request lacks client
/// credentials; return the original request headers so `forward_headers`
/// can reach the vector store. Borrows on the first iteration; on
/// continuations filters out headers nominated by `Connection`.
fn callout_request_headers<'a>(ctx: &'a HttpFilterContext<'_>) -> Cow<'a, HeaderMap> {
    let Some(state) = ctx.extensions.get::<IterationState>().filter(|s| s.iteration() > 0) else {
        return Cow::Borrowed(&ctx.request.headers);
    };
    let original = &state.original_request.headers;
    let mut filtered = HeaderMap::with_capacity(original.len());
    for (name, value) in original {
        if !connection_nominates_header(original, name) {
            filtered.append(name.clone(), value.clone());
        }
    }
    Cow::Owned(filtered)
}

/// Restore end-to-end client headers after the iterative router isolates a
/// transitioned step. Headers tied to the original wire representation or
/// request identity cannot be replayed after the JSON body changes.
fn preserve_original_request_headers(ctx: &mut HttpFilterContext<'_>) {
    let Some(state) = ctx.extensions.get::<IterationState>() else {
        return;
    };
    if state.iteration() == 0 {
        return;
    }
    let headers = state
        .original_request
        .headers
        .iter()
        .filter(|(name, _)| {
            should_replay_original_header(name) && !connection_nominates_header(&state.original_request.headers, name)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Vec<_>>();
    ctx.request_headers_to_set.extend(headers);
}

/// Whether `Connection` marks a request header as specific to one hop.
fn connection_nominates_header(headers: &HeaderMap, name: &http::header::HeaderName) -> bool {
    headers
        .get_all(http::header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .any(|token| token.eq_ignore_ascii_case(name.as_str()))
}

/// Whether a header remains valid after the continuation body is rewritten.
fn should_replay_original_header(name: &http::header::HeaderName) -> bool {
    !praxis_core::reserved_headers::is_reserved(name.as_str())
        && !matches!(
            name.as_str(),
            "accept-encoding"
                | "connection"
                | "content-encoding"
                | "content-length"
                | "content-md5"
                | "digest"
                | "expect"
                | "host"
                | "idempotency-key"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "signature"
                | "signature-input"
                | "te"
                | "trailer"
                | "transfer-encoding"
                | "upgrade"
        )
}

/// Create file-search state inside the router after a bounded preflight.
fn initialize_file_search_state(
    ctx: &mut HttpFilterContext<'_>,
    body: &Option<Bytes>,
    max_state_bytes: usize,
) -> Option<FilterAction> {
    let bytes = body.as_ref()?;
    let probe = probe_request(bytes)?;
    if probe.stream {
        ctx.extensions.insert(StreamingRequest);
    }
    if ctx.extensions.get::<ResponsesState>().is_some() {
        return None;
    }
    if !probe.tools.0 {
        return None;
    }
    let framework_bytes = retained_iteration_bytes(ctx);
    if framework_bytes.saturating_add(bytes.len().saturating_mul(4)) > max_state_bytes {
        return Some(continuation_state_rejection());
    }
    let parsed = serde_json::from_slice::<Value>(bytes).ok()?;
    let state = ResponsesState::from_request_body(parsed);
    if !continuation_state_fits(framework_bytes, &state, max_state_bytes, 0) {
        return Some(continuation_state_rejection());
    }
    ctx.extensions.insert(state);
    None
}

/// Minimal root object used to detect hosted file search without retaining the
/// full request for unrelated Responses calls.
#[derive(Deserialize)]
struct FileSearchRequestProbe {
    /// Whether the client requested an SSE response.
    #[serde(default)]
    stream: bool,

    /// Whether the request's tools array contains a file-search declaration.
    #[serde(default)]
    tools: FileSearchToolsProbe,
}

/// Allocation-free result of scanning the request's tools array.
#[derive(Default)]
struct FileSearchToolsProbe(bool);

impl<'de> Deserialize<'de> for FileSearchToolsProbe {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Sequence visitor that remembers whether any tool declares file search.
        struct ToolsVisitor;

        impl<'de> serde::de::Visitor<'de> for ToolsVisitor {
            type Value = FileSearchToolsProbe;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array of Responses API tools")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut declared = false;
                while let Some(tool) = seq.next_element::<FileSearchToolProbe<'de>>()? {
                    declared |= tool.kind.as_deref() == Some("file_search");
                }
                Ok(FileSearchToolsProbe(declared))
            }
        }

        deserializer.deserialize_seq(ToolsVisitor)
    }
}

/// Minimal borrowed view of one Responses API tool declaration.
#[derive(Deserialize)]
struct FileSearchToolProbe<'a> {
    /// Hosted tool discriminator.
    #[serde(borrow, default, rename = "type")]
    kind: Option<Cow<'a, str>>,
}

/// Extract only the routing facts needed before retaining full request state.
fn probe_request(bytes: &[u8]) -> Option<FileSearchRequestProbe> {
    serde_json::from_slice(bytes).ok()
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
        &state.file_search_output_items,
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
        .chain(
            state
                .mcp_tool_map
                .iter()
                .map(|((server, tool), _)| server.len().saturating_add(tool.len())),
        )
        .fold(0_usize, usize::saturating_add);
    used = used.saturating_add(string_bytes);
    for value in state.mcp_tool_map.values() {
        let Some(size) = bounded_json_size(value, max_bytes.saturating_sub(used)).ok().flatten() else {
            return false;
        };
        used = used.saturating_add(size);
    }
    used <= max_bytes
}

/// Reject work whose request-local continuation state exceeds its ceiling.
fn continuation_state_rejection() -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        413,
        "invalid_request_error",
        "openai_file_search_callout: continuation state exceeds max_state_bytes",
        false,
    ))
}

/// Reject an invalid successful model response only after local continuation began.
fn invalid_success_response_action(continued: bool) -> FilterAction {
    if continued {
        FilterAction::Reject(responses_error_rejection(
            502,
            "server_error",
            "openai_file_search_callout: inference continuation returned an invalid response",
            false,
        ))
    } else {
        FilterAction::Continue
    }
}

/// Assemble accumulated output and citations into a bounded public response.
fn finalize_public_response(state: &mut ResponsesState) -> Result<Bytes, FilterAction> {
    let final_output = std::mem::take(state.output_items_mut());
    let mut combined_output = std::mem::take(&mut state.file_search_output_items);
    combined_output.extend(final_output);
    *state.output_items_mut() = combined_output;

    annotate_response(&mut state.response_object, &state.citation_files).map_err(|error| {
        warn!(%error, "failed to annotate final file-search response");
        final_response_rejection("openai_file_search_callout: failed to annotate final response")
    })?;
    bounded_json_size(&state.response_object, MAX_JSON_BODY_BYTES)
        .ok()
        .flatten()
        .ok_or_else(|| {
            final_response_rejection("openai_file_search_callout: final response exceeds the JSON response byte limit")
        })?;
    serde_json::to_vec(&state.response_object)
        .map(Bytes::from)
        .map_err(|error| {
            warn!(%error, "failed to encode final file-search response");
            final_response_rejection("openai_file_search_callout: failed to encode final response")
        })
}

/// Build a consistent failure while assembling a model's final response.
fn final_response_rejection(message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(502, "server_error", message, false))
}

/// Whether a model output requires a client-supplied function result.
fn has_client_function_call(output: &[Value]) -> bool {
    output
        .iter()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
}

/// Reject mixed tools until private search context can resume across requests.
fn mixed_tool_response_rejection() -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        "openai_file_search_callout: a model response cannot combine file_search_call with client-executed function_call",
        false,
    ))
}

/// Allow the model to answer after satisfying the first forced search call.
fn reset_tool_choice(state: &mut ResponsesState) {
    state.tool_choice = Value::String("auto".to_owned());
    if let Some(request) = state.request_body.as_object_mut() {
        request.remove("tool_choice");
    }
}

/// Retain model-facing output and replace hosted calls with private bridges.
fn continuation_replay_items(output: &[Value], bridges: Vec<(usize, [Value; 2])>) -> Vec<Value> {
    let mut replay = Vec::with_capacity(output.len().saturating_add(bridges.len()));
    let mut bridges = bridges.into_iter().peekable();
    for (output_index, item) in output.iter().enumerate() {
        let replay_item = matches!(item.get("type").and_then(Value::as_str), Some("reasoning"))
            || (item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("assistant"));
        if replay_item {
            // The item remains public and must also enter the next request.
            replay.push(item.clone());
        }
        while bridges.peek().is_some_and(|(index, _)| *index == output_index) {
            let Some((_index, messages)) = bridges.next() else {
                break;
            };
            replay.extend(messages);
        }
    }
    replay
}

/// Drop upstream representation metadata after replacing the response bytes.
fn clear_rewritten_response_headers(ctx: &mut HttpFilterContext<'_>) {
    if let Some(response) = &mut ctx.response_header {
        response.headers.remove(http::header::CONTENT_ENCODING);
        response.headers.remove(http::header::CONTENT_LENGTH);
        response.headers.remove(http::header::CONTENT_RANGE);
        response.headers.remove(http::header::ETAG);
        response.headers.remove(http::header::LAST_MODIFIED);
        ctx.response_headers_modified = true;
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
        .file_search_output_items
        .iter()
        .chain(state.output_items())
        .filter(|item| is_builtin_tool_call(item) && !is_pending_file_search_call(item))
        .count();
    usize::try_from(max_tool_calls)
        .unwrap_or(usize::MAX)
        .saturating_sub(used_calls)
}

/// Check all retained, current, and incoming output against one request budget.
fn combined_output_fits(state: &ResponsesState, incoming_response: &Value, max_bytes: usize) -> bool {
    let incoming_output = incoming_response
        .get("output")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    bounded_json_size(
        &(
            state.file_search_output_items.as_slice(),
            state.output_items(),
            incoming_output,
        ),
        max_bytes,
    )
    .ok()
    .flatten()
    .is_some()
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

/// Reject streaming before the iterative router can silently buffer SSE.
fn unsupported_streaming_rejection(ctx: &HttpFilterContext<'_>) -> Option<FilterAction> {
    (ctx.extensions.get::<StreamingRequest>().is_some()
        || ctx
            .get_metadata("openai_responses_format.stream")
            .is_some_and(|value| value == "true"))
    .then(|| {
        FilterAction::Reject(responses_error_rejection(
            400,
            "invalid_request_error",
            "openai_file_search_callout: stream=true is not supported by an iterative file-search pipeline",
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

/// Whether an output item is a vLLM-emitted `function_call` for file search.
fn is_file_search_function_call(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("function_call")
        && item.get("name").and_then(Value::as_str) == Some("file_search")
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
/// Returns the number of translated items.
fn translate_function_calls_to_file_search(response: &mut Value) -> usize {
    let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) else {
        return 0;
    };

    let mut translated = 0;
    for item in output.iter_mut() {
        if !is_file_search_function_call(item) {
            continue;
        }
        let Some(object) = item.as_object_mut() else {
            continue;
        };

        let queries = object
            .get("arguments")
            .and_then(Value::as_str)
            .map(extract_file_search_queries)
            .unwrap_or_default();

        object.insert("type".to_owned(), Value::String("file_search_call".to_owned()));
        object.insert("status".to_owned(), Value::String("searching".to_owned()));
        object.insert(
            "queries".to_owned(),
            Value::Array(queries.into_iter().map(Value::String).collect()),
        );

        object.remove("name");
        object.remove("arguments");
        object.remove("call_id");

        translated += 1;
    }
    translated
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

/// Mark every pending call incomplete when no built-in tool budget remains.
fn terminalize_all_pending_calls(state: &mut ResponsesState) {
    for item in state.output_items_mut() {
        if is_pending_file_search_call(item)
            && let Some(object) = item.as_object_mut()
        {
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
