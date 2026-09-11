// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Compact filter: token counting and context window management.
//!
//! When a request's `context_management` contains a compaction
//! configuration and the token count exceeds the specified threshold,
//! this filter summarizes the conversation history via a sub-request
//! to an inference backend, replacing it with a single compaction
//! item. Runs after `rehydrate` (which populates messages and
//! previous usage) and after `openai_tool_parse`. Place
//! `openai_file_resolve` and `openai_doc_extract` before compact so
//! rewritten current-turn content is what compaction preserves.
//!
//! # Scope
//!
//! Compaction applies in two scenarios:
//!
//! - **Reactive** — multi-turn requests where `openai_rehydrate` has loaded stored conversation history, i.e. requests
//!   that include `previous_response_id` or `conversation`. Single-turn requests (no stored history, even with
//!   `context_management` set) are released without compaction because there is no prior history to summarize.
//! - **Explicit** — `POST /v1/responses/compact`, which summarizes any previously stored response (plus optional inline
//!   `input`) regardless of rehydration, returning a `response.compaction` object.
//!
//! See [`CompactFilter`] for the full description of both scenarios.
//!
//! Praxis runs `StreamBuffer` body hooks before header-phase request
//! filters. Configuration therefore requires an explicit
//! `allow_pre_security_callout: true` acknowledgement and should only
//! be used behind an outer authentication and authorization boundary.

pub(super) mod config;

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

use std::{borrow::Cow, time::Duration};

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, body::MAX_JSON_BODY_BYTES,
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, warn};

use self::config::{CompactFilterConfig, ValidatedConfig, build_config};
use super::{error::responses_error_rejection, state::ResponsesState};
use crate::{
    callout_policy::OnFailure,
    store::{ResponseRecord, ResponseStoreRegistry},
    subrequest::{self, SubRequest, SubRequestClient},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum response body size for summarization callouts (1 MiB).
const MAX_SUMMARIZATION_RESPONSE_BYTES: usize = 1_048_576;

/// Minimum allowed `compact_threshold` for compaction (1,000 tokens).
const MIN_COMPACT_THRESHOLD: u64 = 1_000; // 1,000 tokens

/// System prompt for the summarization call.
const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
Summarize the following conversation concisely. \
Preserve all key facts, decisions, code snippets, \
user preferences, and important context. The summary \
will replace the full conversation history, so it must \
capture everything needed to continue coherently.";

/// Default prefix prepended to the summary when translating
/// compaction items to backend-compatible messages.
pub const DEFAULT_SUMMARY_PREFIX: &str = "[Previous conversation summary]\n\n";

// -----------------------------------------------------------------------------
// CompactionParams
// -----------------------------------------------------------------------------

/// Parsed compaction parameters from the request's `context_management`.
#[derive(Debug, Eq, PartialEq)]
struct CompactionParams {
    /// Token threshold above which compaction triggers.
    compact_threshold: u64,
    /// Optional model override for the summarization call.
    compaction_model: Option<String>,
}

/// Result of a successful summarization callout.
struct Summarization {
    /// The generated summary text.
    content: String,
    /// The callout's Chat Completions `usage` object, when reported.
    usage: Option<Value>,
}

// -----------------------------------------------------------------------------
// CompactFilter
// -----------------------------------------------------------------------------

/// Summarizes conversation history when the token count exceeds a
/// configured threshold.
///
/// `compact_threshold` in `context_management` must be an integer
/// of at least 1000. Invalid or missing `compact_threshold` values
/// produce an `invalid_request_error`.
///
/// Compaction applies in two scenarios:
///
/// - **Rehydrated history** - stored history loaded via `previous_response_id` or `conversation`. Only the stored
///   history is summarized; the current turn is preserved.
///
/// - **Explicit compact** - `POST /v1/responses/compact` with a required `model` and an inline `input` conversation
///   and/or a `previous_response_id`. Loads any stored history, appends the inline input, summarizes the combined
///   conversation, and returns a `response.compaction` object (with `output` and `usage`) per the OpenAI contract.
///
/// Direct input requests (full conversation in `input` with no stored history) skip reactive compaction because
/// `state.input == state.messages` - there is no separable "current turn" to preserve after summarization.
/// Requests without rehydrated history are released without compaction.
///
/// Praxis runs `StreamBuffer` body hooks before header-phase request
/// filters. This filter therefore requires
/// `allow_pre_security_callout: true` and should only be used behind
/// an outer authentication and authorization boundary.
///
/// # YAML
///
/// ```yaml
/// filter: openai_compact
/// allow_pre_security_callout: true
/// inference_url: "http://localhost:11434/v1/chat/completions"
/// allow_private_inference_url: true
/// default_model: llama3.2:1b
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_compact
/// allow_pre_security_callout: true
/// inference_url: "http://localhost:11434/v1/chat/completions"
/// allow_private_inference_url: true
/// default_model: gpt-4o-mini
/// tiktoken_encoding: cl100k_base
/// summary_prefix: "[Previous conversation summary]\n\n"
/// timeout_ms: 30000
/// on_failure: closed
/// status_on_error: 502
/// ```
pub struct CompactFilter {
    /// HTTP client for the summarization inference call.
    client: SubRequestClient,
    /// Validated filter configuration.
    config: ValidatedConfig,
}

impl CompactFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config validation fails.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client = SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None));
        Self::build(config, client)
    }

    /// Create a filter from parsed YAML config using a shared sub-request client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config validation fails.
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::build(config, client)
    }

    /// Shared constructor: parse config, validate, eager-init tiktoken, and box.
    fn build(config: &serde_yaml::Value, client: SubRequestClient) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: CompactFilterConfig = parse_filter_config("openai_compact", config)?;
        let validated = build_config(&cfg)?;
        eager_init_tiktoken(&validated.tiktoken_encoding);
        Ok(Box::new(Self {
            client,
            config: validated,
        }))
    }

    /// Run the summarization callout and return the summary text.
    async fn execute_compaction(
        &self,
        state: &ResponsesState,
        params: &CompactionParams,
        conversation_text: &str,
    ) -> Result<Option<String>, FilterAction> {
        let model = params.compaction_model.as_deref().unwrap_or(&self.config.default_model);
        let instructions = state.request_body.get("instructions").and_then(Value::as_str);
        let request = build_summarization_request(conversation_text, instructions, model);
        let timeout = Duration::from_millis(self.config.callout.timeout_ms);
        let result = subrequest::execute_url(
            &self.client,
            &self.config.inference_url,
            request,
            MAX_SUMMARIZATION_RESPONSE_BYTES,
            timeout,
            self.config.address_policy,
        )
        .await;
        Ok(self.handle_subrequest_result(result)?.map(|s| s.content))
    }

    /// Map a subrequest result to a parsed summarization or a filter action.
    fn handle_subrequest_result(
        &self,
        result: Result<subrequest::SubResponse, subrequest::SubRequestError>,
    ) -> Result<Option<Summarization>, FilterAction> {
        match result {
            Ok(resp) if (200..300).contains(&(resp.status as usize)) => {
                parse_summarization_response(&resp.body).map(Some).or_else(|e| {
                    warn!(error = %e, "failed to parse summarization response");
                    self.on_callout_error("failed to parse summarization response")
                })
            },
            Ok(resp) => {
                warn!(status = resp.status, "summarization callout returned non-2xx");
                self.on_callout_error("summarization callout rejected")
            },
            Err(e) => {
                warn!(error = %e, "summarization callout failed");
                self.on_callout_error("summarization callout failed")
            },
        }
    }

    /// Run a summarization callout for an explicit compact request.
    async fn summarize_messages(
        &self,
        req: &ExplicitCompactRequest,
        messages: &[Value],
    ) -> Result<Option<Summarization>, FilterAction> {
        let conversation_text = build_conversation_text(messages);
        let request = build_summarization_request(&conversation_text, req.instructions.as_deref(), &req.model);
        let timeout = Duration::from_millis(self.config.callout.timeout_ms);
        let result = subrequest::execute_url(
            &self.client,
            &self.config.inference_url,
            request,
            MAX_SUMMARIZATION_RESPONSE_BYTES,
            timeout,
            self.config.address_policy,
        )
        .await;
        self.handle_subrequest_result(result)
    }

    /// Apply the configured open/closed policy on a callout error.
    fn on_callout_error(&self, message: &str) -> Result<Option<Summarization>, FilterAction> {
        match self.config.callout.on_failure {
            OnFailure::Open => Ok(None),
            OnFailure::Closed => Err(FilterAction::Reject(responses_error_rejection(
                self.config.callout.status_on_error,
                "server_error",
                message,
            ))),
        }
    }

    /// Apply compaction results: replace the conversation history with
    /// the compaction item. The compacted messages are persisted by the
    /// normal response store filter as part of the eventual response.
    fn apply_compaction(&self, ctx: &mut HttpFilterContext<'_>, summary: &str) {
        let compaction_id = format!("compact_{}", ctx.id_generator.generate(ctx.time_source));
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            warn!("ResponsesState missing in apply_compaction");
            return;
        };
        replace_messages(
            state,
            build_compaction_item(&compaction_id, summary, &self.config.summary_prefix),
        );
    }

    /// Check the threshold and run summarization if it is exceeded.
    async fn check_and_summarize(&self, state: &ResponsesState) -> Result<Option<String>, FilterAction> {
        let (params, conversation_text) = match should_compact(state, &self.config.tiktoken_encoding) {
            Ok(Some(pair)) => pair,
            Ok(None) => return Ok(None),
            Err(msg) => {
                let rej = responses_error_rejection(400, "invalid_request_error", &msg);
                return Err(FilterAction::Reject(rej));
            },
        };
        self.execute_compaction(state, &params, &conversation_text).await
    }

    /// Handle an explicit `POST /v1/responses/compact` request.
    ///
    /// Loads a stored conversation by `response_id`, compacts it via
    /// a summarization callout, stores the compacted result, and
    /// returns the new response.
    async fn handle_explicit_compact(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: &Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        match self.do_explicit_compact(ctx, body).await {
            Ok(action) | Err(action) => Ok(action),
        }
    }

    /// Inner logic for explicit compact, using `FilterAction` as the error type.
    async fn do_explicit_compact(
        &self,
        ctx: &HttpFilterContext<'_>,
        body: &Option<Bytes>,
    ) -> Result<FilterAction, FilterAction> {
        let req = parse_compact_request_body(body)?;
        let (store, tenant_id) = resolve_store_and_tenant(ctx)?;
        let messages = collect_compact_messages(&*store, &tenant_id, &req).await?;
        let writer = CompactionWriter {
            filter: self,
            ctx,
            store: &*store,
            tenant_id: &tenant_id,
            req: &req,
            messages: &messages,
        };
        let response_object = if let Some(summary) = self.summarize_messages(&req, &messages).await? {
            writer.persist_compacted(&summary).await?
        } else {
            warn!("fail-open compaction: summarization callout failed; persisting uncompacted no-op");
            writer.persist_uncompacted().await?
        };
        let body_bytes = serde_json::to_vec(&response_object).unwrap_or_default();
        Ok(FilterAction::Reject(
            praxis_filter::Rejection::status(200)
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(body_bytes)),
        ))
    }
}

#[async_trait]
impl HttpFilter for CompactFilter {
    fn name(&self) -> &'static str {
        "openai_compact"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
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
        if is_explicit_compact_request(ctx) {
            return self.handle_explicit_compact(ctx, body).await;
        }
        if !is_responses_request(ctx) {
            return Ok(FilterAction::Release);
        }
        if let Some(action) = reject_invalid_compaction_config(ctx) {
            return Ok(action);
        }
        if !ensure_compactable_state(ctx) {
            return Ok(FilterAction::Release);
        }
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            warn!("ResponsesState missing after ensure_compactable_state");
            return Ok(FilterAction::Release);
        };
        let summary = match self.check_and_summarize(state).await {
            Ok(Some(summary)) => summary,
            Ok(None) | Err(FilterAction::Release) => return Ok(FilterAction::Release),
            Err(action) => return Ok(action),
        };
        self.apply_compaction(ctx, &summary);
        ctx.set_metadata("responses.compacted", "true");
        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Compaction Logic
// -----------------------------------------------------------------------------

/// Returns `true` when compaction should proceed.
fn ensure_compactable_state(ctx: &HttpFilterContext<'_>) -> bool {
    is_compactable(ctx.extensions.get::<ResponsesState>())
}

/// Reject the request when a compaction config is present but invalid.
///
/// Runs for every Responses request so an invalid `compact_threshold` is
/// rejected even when reactive compaction is ultimately skipped (e.g. direct
/// input without a rehydrated history to separate from the current turn).
/// Returns `None` when there is no config, or the config is valid.
fn reject_invalid_compaction_config(ctx: &HttpFilterContext<'_>) -> Option<FilterAction> {
    let state = ctx.extensions.get::<ResponsesState>()?;
    match extract_compaction_config(&state.context_management) {
        Ok(_) => None,
        Err(msg) => Some(FilterAction::Reject(responses_error_rejection(
            400,
            "invalid_request_error",
            &msg,
        ))),
    }
}

/// Check whether the given state qualifies for reactive compaction.
///
/// Returns `true` only when rehydrated history is present. Direct
/// input requests (no `previous_response_id`) are skipped because
/// `state.input == state.messages` - there is no separable "current
/// turn" to preserve after summarization. Use the explicit
/// `POST /v1/responses/compact` endpoint for non-rehydrated history.
///
/// This gate is the single decision point for direct-input handling.
/// `should_compact` still evaluates the threshold for direct input (and
/// unit tests exercise that lower-level contract directly), but the
/// summarization it would trigger is unreachable through the filter while
/// this returns `false` for non-rehydrated state. Threshold *validation*
/// is separate: `reject_invalid_compaction_config` runs before this gate,
/// so an invalid `compact_threshold` is rejected for direct input too.
fn is_compactable(state: Option<&ResponsesState>) -> bool {
    let Some(state) = state else {
        return false;
    };
    state.history_rehydrated
}

/// Check whether compaction should run and return the params + text.
///
/// Returns `None` if there is no compaction config, the encoding is
/// unknown, or the token count is below the threshold.
///
/// When `previous_usage` is available from the rehydrated response,
/// its `total_tokens` is used directly - avoiding the cost of BPE
/// tokenization. Falls back to tiktoken estimation otherwise. The
/// fallback estimate includes instructions and tool definitions in
/// addition to conversation messages, since all three contribute to
/// the rendered context sent to the model.
///
/// The check is reactive: the token count reflects the *previous*
/// turn's usage, not the current one. If the previous turn exceeded
/// the threshold, we compact before sending this turn.
fn should_compact(
    state: &ResponsesState,
    tiktoken_encoding: &str,
) -> Result<Option<(CompactionParams, String)>, String> {
    let Some(params) = extract_compaction_config(&state.context_management)? else {
        return Ok(None);
    };
    // Summarize the full conversation, including the current turn, so the
    // summarizer has complete context. The current turn is not lost from the
    // rehydrated history: `replace_messages` preserves it verbatim after the
    // compaction item via a tail split, so it appears exactly once.
    let history = &state.messages;

    if let Some(token_count) = previous_usage_total(state) {
        if !exceeds_threshold(token_count, &params) {
            return Ok(None);
        }
        return Ok(Some((params, build_conversation_text(history))));
    }

    debug!("previous_usage unavailable, falling back to tiktoken estimation");
    let conversation_text = build_conversation_text(history);
    let overhead = build_context_overhead_text(&state.request_body);
    let full_text = format!("{conversation_text}\n\n{overhead}");
    let Some(token_count) = get_token_count(&full_text, tiktoken_encoding) else {
        return Ok(None);
    };
    if !exceeds_threshold(token_count, &params) {
        return Ok(None);
    }
    Ok(Some((params, conversation_text)))
}

/// Log and return whether `token_count` exceeds the compaction threshold.
fn exceeds_threshold(token_count: u64, params: &CompactionParams) -> bool {
    if token_count <= params.compact_threshold {
        debug!(
            token_count,
            threshold = params.compact_threshold,
            "under threshold, skipping"
        );
        return false;
    }
    debug!(
        token_count,
        threshold = params.compact_threshold,
        "threshold exceeded, compacting"
    );
    true
}

/// Extract `total_tokens` from the previous response's usage object.
fn previous_usage_total(state: &ResponsesState) -> Option<u64> {
    let total = state.previous_usage.as_ref()?.get("total_tokens")?.as_u64()?;
    debug!(
        count = total,
        source = "previous_usage",
        "token count from prior response"
    );
    Some(total)
}

/// Check whether this is an explicit `POST /v1/responses/compact` request.
pub(super) fn is_explicit_compact_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.request.method == http::Method::POST && ctx.request.uri.path().trim_end_matches('/') == "/v1/responses/compact"
}

/// Check whether this is an OpenAI Responses API request.
fn is_responses_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_format.format") == Some("openai_responses")
}

// -----------------------------------------------------------------------------
// Explicit Compact Endpoint Helpers
// -----------------------------------------------------------------------------

/// Parsed body for `POST /v1/responses/compact`.
struct ExplicitCompactRequest {
    /// Required model for the compaction pass.
    model: String,
    /// Inline conversation items to compact (may be empty).
    input: Vec<Value>,
    /// Optional stored response whose history is loaded and compacted.
    previous_response_id: Option<String>,
    /// Optional instructions to prepend to the summarization prompt.
    instructions: Option<String>,
}

/// Parse and validate the `POST /v1/responses/compact` body.
#[expect(clippy::too_many_lines, reason = "linear field parsing and validation")]
fn parse_compact_request_body(body: &Option<Bytes>) -> Result<ExplicitCompactRequest, FilterAction> {
    let bytes = body
        .as_ref()
        .filter(|b| !b.is_empty())
        .ok_or_else(|| reject_compact(400, "invalid_request_error", "request body is empty"))?;
    let parsed: Value = serde_json::from_slice(bytes).map_err(|e| {
        debug!(error = %e, "compact request body parse failed");
        reject_compact(400, "invalid_request_error", "invalid JSON body")
    })?;
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| reject_compact(400, "invalid_request_error", "missing required field: model"))?
        .to_owned();
    let input = parse_compact_input(parsed.get("input"));
    let previous_response_id = parsed
        .get("previous_response_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned);
    if input.is_empty() && previous_response_id.is_none() {
        return Err(reject_compact(
            400,
            "invalid_request_error",
            "request must include input or previous_response_id",
        ));
    }
    Ok(ExplicitCompactRequest {
        model,
        input,
        previous_response_id,
        instructions: parsed
            .get("instructions")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

/// Normalize the `input` field into a list of conversation items.
///
/// A bare string is coerced into a single `user` message, matching the
/// contract where a string is equivalent to a text user input.
fn parse_compact_input(input: Option<&Value>) -> Vec<Value> {
    match input {
        Some(Value::String(s)) if !s.is_empty() => {
            vec![serde_json::json!({"role": "user", "content": s})]
        },
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    }
}

/// Look up the store and tenant from the request context.
fn resolve_store_and_tenant(
    ctx: &HttpFilterContext<'_>,
) -> Result<(std::sync::Arc<dyn crate::store::ResponseStore>, String), FilterAction> {
    let store = ctx
        .extensions
        .get::<ResponseStoreRegistry>()
        .and_then(|r| r.get("default"))
        .ok_or_else(|| reject_compact(500, "server_error", "response store not available"))?;
    let tenant_id = ctx.get_metadata("responses.tenant_id").unwrap_or("default").to_owned();
    Ok((store, tenant_id))
}

/// Assemble the conversation to compact from stored history and inline input.
///
/// When `previous_response_id` is set, its stored messages are loaded
/// first and the inline `input` items are appended after.
async fn collect_compact_messages(
    store: &dyn crate::store::ResponseStore,
    tenant_id: &str,
    req: &ExplicitCompactRequest,
) -> Result<Vec<Value>, FilterAction> {
    let mut messages = Vec::new();
    if let Some(prev) = req.previous_response_id.as_deref() {
        let record = fetch_response(store, tenant_id, prev).await?;
        messages.extend(stored_message_array(record.messages));
    }
    messages.extend(req.input.iter().cloned());
    if messages.is_empty() {
        return Err(reject_compact(400, "invalid_request_error", "no messages to compact"));
    }
    Ok(messages)
}

/// Fetch a stored response by id.
async fn fetch_response(
    store: &dyn crate::store::ResponseStore,
    tenant_id: &str,
    response_id: &str,
) -> Result<ResponseRecord, FilterAction> {
    match store.get_response(tenant_id, response_id).await {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(reject_compact(404, "not_found_error", "response not found")),
        Err(e) => {
            warn!(error = %e, "failed to fetch response for compact");
            Err(reject_compact(500, "server_error", "failed to fetch response"))
        },
    }
}

/// Extract a stored record's messages as an array, or an empty list.
fn stored_message_array(messages: Value) -> Vec<Value> {
    match messages {
        Value::Array(arr) => arr,
        _ => Vec::new(),
    }
}

/// Borrows everything needed to build and persist an explicit compaction
/// result, so the persist paths read as methods rather than long argument lists.
struct CompactionWriter<'a> {
    /// Filter config (summary prefix, tiktoken encoding).
    filter: &'a CompactFilter,
    /// Request context, used for id and timestamp generation.
    ctx: &'a HttpFilterContext<'a>,
    /// Response store the compaction record is persisted to.
    store: &'a dyn crate::store::ResponseStore,
    /// Tenant the record is scoped to.
    tenant_id: &'a str,
    /// The parsed explicit compact request (supplies the response model).
    req: &'a ExplicitCompactRequest,
    /// The conversation being compacted.
    messages: &'a [Value],
}

impl CompactionWriter<'_> {
    /// Persist the compaction result for a successful summarization. The
    /// response `output` and the persisted history are the same single
    /// compaction item.
    async fn persist_compacted(&self, summary: &Summarization) -> Result<Value, FilterAction> {
        let compaction_id = format!("compact_{}", self.ctx.id_generator.generate(self.ctx.time_source));
        let item = Value::Array(vec![build_compaction_item(
            &compaction_id,
            &summary.content,
            &self.filter.config.summary_prefix,
        )]);
        let usage = build_compaction_usage(self.messages, Some(summary), &self.filter.config.tiktoken_encoding);
        self.persist_response(item.clone(), item, usage).await
    }

    /// Persist an uncompacted no-op when the summarization callout fails under
    /// `on_failure: open`.
    ///
    /// The response `output` is an empty array: there is no summary, so no
    /// `compaction` item is emitted, and its absence is the caller's no-op
    /// signal. Returning the raw `{role, content}` messages instead would
    /// violate the Responses output schema (items require `type`/`id`/`status`).
    /// The intact conversation is still persisted as the record's history so a
    /// follow-up request referencing this response id rehydrates it in full.
    async fn persist_uncompacted(&self) -> Result<Value, FilterAction> {
        let usage = build_compaction_usage(self.messages, None, &self.filter.config.tiktoken_encoding);
        self.persist_response(Value::Array(Vec::new()), Value::Array(self.messages.to_vec()), usage)
            .await
    }

    /// Assemble the `response.compaction` object, persist the record, and return
    /// the object. `output` is the API-facing output array; `stored_messages` is
    /// the history persisted for rehydration continuity (the two differ on
    /// fail-open).
    async fn persist_response(
        &self,
        output: Value,
        stored_messages: Value,
        usage: Value,
    ) -> Result<Value, FilterAction> {
        let resp_id = format!("resp_{}", self.ctx.id_generator.generate(self.ctx.time_source));
        let created_at = i64::try_from(self.ctx.time_source.now().as_secs()).unwrap_or(i64::MAX);
        let response_object = serde_json::json!({
            "id": resp_id,
            "object": "response.compaction",
            "created_at": created_at,
            "status": "completed",
            "output": output,
            "usage": usage,
        });

        let record = ResponseRecord {
            id: resp_id,
            tenant_id: self.tenant_id.to_owned(),
            created_at,
            model: self.req.model.clone(),
            response_object: response_object.clone(),
            input: stored_messages.clone(),
            messages: stored_messages,
        };
        self.store.upsert_response(&record).await.map_err(|e| {
            warn!(error = %e, "failed to persist explicit compaction response");
            reject_compact(500, "server_error", "failed to persist compaction response")
        })?;
        Ok(response_object)
    }
}

/// Build a `ResponseUsage` object for the compaction pass.
///
/// Prefers the token counts the summarization callout itself reported,
/// mapping the Chat Completions `usage` to the Responses shape. When the
/// backend omits usage (or on a fail-open pass-through), falls back to a
/// tiktoken estimate of the source conversation and produced summary.
fn build_compaction_usage(messages: &[Value], summary: Option<&Summarization>, tiktoken_encoding: &str) -> Value {
    if let Some(usage) = summary.and_then(|s| s.usage.as_ref()) {
        return map_chat_usage(usage);
    }
    let conversation_text = build_conversation_text(messages);
    let input_tokens = get_token_count(&conversation_text, tiktoken_encoding).unwrap_or(0);
    let output_tokens = summary
        .and_then(|s| get_token_count(&s.content, tiktoken_encoding))
        .unwrap_or(0);
    build_usage(input_tokens, 0, output_tokens, 0)
}

/// Map a Chat Completions `usage` object to the Responses `ResponseUsage` shape.
fn map_chat_usage(usage: &Value) -> Value {
    let field = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input_tokens = field("prompt_tokens");
    let output_tokens = field("completion_tokens");
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| input_tokens.saturating_add(output_tokens));
    serde_json::json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total,
    })
}

/// Assemble a `ResponseUsage` object from raw token counts.
fn build_usage(input_tokens: u64, cached_tokens: u64, output_tokens: u64, reasoning_tokens: u64) -> Value {
    serde_json::json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens.saturating_add(output_tokens),
    })
}

/// Build a `FilterAction::Reject` for an explicit compact error.
fn reject_compact(status: u16, code: &str, message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(status, code, message))
}

/// Parse the `context_management` JSON to find a compaction config.
///
/// The `context_management` field is an array like:
/// `[{"type": "compaction", "compact_threshold": 50000}]`
///
/// Returns:
/// - `Ok(None)` if `context_management` is absent, `null`, or an array with no compaction entry.
/// - `Ok(Some(params))` if a valid compaction entry is present.
/// - `Err(msg)` if `context_management` is present but not an array, or a compaction entry has an invalid
///   `compact_threshold` or `compaction_model`.
fn extract_compaction_config(context_management: &Option<Value>) -> Result<Option<CompactionParams>, String> {
    // Absent or explicit null: OpenAI treats both as "no context management".
    let Some(value) = context_management.as_ref().filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let Some(array) = value.as_array() else {
        return Err("context_management must be an array".to_owned());
    };
    for entry in array {
        if entry.get("type").and_then(Value::as_str) == Some("compaction") {
            return parse_compaction_entry(entry).map(Some);
        }
    }
    Ok(None)
}

/// Parse a single `{"type": "compaction", ...}` entry into [`CompactionParams`].
fn parse_compaction_entry(entry: &Value) -> Result<CompactionParams, String> {
    let err_msg = "compact_threshold must be an integer of at least 1000";
    let raw_threshold = entry.get("compact_threshold").ok_or_else(|| err_msg.to_owned())?;
    let compact_threshold = raw_threshold.as_u64().ok_or_else(|| err_msg.to_owned())?;
    if compact_threshold < MIN_COMPACT_THRESHOLD {
        return Err(err_msg.to_owned());
    }
    let compaction_model = match entry.get("compaction_model") {
        Some(m) => Some(
            m.as_str()
                .ok_or_else(|| "compaction_model must be a string".to_owned())?
                .to_owned(),
        ),
        None => None,
    };
    Ok(CompactionParams {
        compact_threshold,
        compaction_model,
    })
}

/// Resolve the tiktoken singleton for the given encoding name.
fn resolve_tiktoken(encoding: &str) -> Option<&'static tiktoken_rs::CoreBPE> {
    match encoding {
        "cl100k_base" => Some(tiktoken_rs::cl100k_base_singleton()),
        "o200k_base" => Some(tiktoken_rs::o200k_base_singleton()),
        other => {
            warn!(encoding = other, "unknown tiktoken encoding, cannot estimate tokens");
            None
        },
    }
}

/// Pre-load the tiktoken BPE singleton at pipeline build time so the
/// first request does not pay the ~100ms merge-rule loading cost.
fn eager_init_tiktoken(encoding: &str) {
    resolve_tiktoken(encoding);
}

/// Estimate the token count for the given messages using tiktoken.
///
/// Uses the configured encoding (e.g. `cl100k_base`, `o200k_base`)
/// to tokenize the serialized conversation text. Runs inside
/// `block_in_place` because BPE tokenization is CPU-bound.
///
/// Returns `None` if the encoding name is not recognized.
fn get_token_count(conversation_text: &str, tiktoken_encoding: &str) -> Option<u64> {
    let bpe = resolve_tiktoken(tiktoken_encoding)?;
    let count = tokio::task::block_in_place(|| bpe.count_ordinary(conversation_text)) as u64;
    debug!(
        count,
        source = "tiktoken",
        encoding = tiktoken_encoding,
        "token count estimated"
    );
    Some(count)
}

/// Build a Chat Completions request for summarization.
///
/// The request body has this shape:
/// ```json
/// {
///   "model": "<model>",
///   "messages": [
///     {"role": "system", "content": "<system prompt + instructions>"},
///     {"role": "user", "content": "<conversation text>"}
///   ]
/// }
/// ```
fn build_summarization_request(conversation_text: &str, instructions: Option<&str>, model: &str) -> SubRequest {
    let system_content = match instructions {
        Some(inst) => format!("{inst}\n\n{SUMMARIZATION_SYSTEM_PROMPT}"),
        None => SUMMARIZATION_SYSTEM_PROMPT.to_owned(),
    };

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": conversation_text}
        ]
    });

    let body_bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());

    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));

    SubRequest {
        method: http::Method::POST,
        uri: http::Uri::default(),
        headers,
        body: body_bytes,
    }
}

/// Parse the Chat Completions response, extracting the summary text and
/// the callout's own token usage.
///
/// Expected shape:
/// `{"choices": [{"message": {"content": "..."}}], "usage": {...}}`
fn parse_summarization_response(body: &[u8]) -> Result<Summarization, String> {
    let body: Value =
        serde_json::from_slice(body).map_err(|err| format!("failed to parse Chat Completions response JSON: {err}"))?;
    let content = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|msg| msg.get("content"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| "Chat Completions response missing choices[0].message.content".to_owned())?;
    Ok(Summarization {
        content,
        usage: body.get("usage").cloned(),
    })
}

/// Build the compaction output item.
///
/// Returns: `{"type": "compaction", "id": "<id>", "encrypted_content": "<base64>"}`
/// with an optional `"summary_prefix"` when it differs from the default.
///
/// The summary is base64-encoded into `encrypted_content` to match the
/// OpenAI Responses API compaction item shape and make the content opaque
/// to clients.
fn build_compaction_item(id: &str, summary: &str, summary_prefix: &str) -> Value {
    let encrypted_content = base64::engine::general_purpose::STANDARD.encode(summary);
    let mut item = serde_json::json!({
        "type": "compaction",
        "id": id,
        "encrypted_content": encrypted_content
    });
    if summary_prefix != DEFAULT_SUMMARY_PREFIX
        && let Some(obj) = item.as_object_mut()
    {
        obj.insert("summary_prefix".to_owned(), Value::String(summary_prefix.to_owned()));
    }
    item
}

/// Replace conversation history with the compaction item.
///
/// After replacement:
/// - `state.messages` = `[compaction_item, ...current_turn]`
/// - `state.persisted_messages` = `[compaction_item, ...current_turn]`
///
/// The compaction item is `{"type": "compaction", "encrypted_content": "<base64>"}`.
/// The current turn is the tail of each message list whose length
/// matches `state.input`. File resolution and document extraction
/// rewrite that tail in place and leave `state.input` as the original
/// client payload, so compaction must not rebuild from `state.input`.
fn replace_messages(state: &mut ResponsesState, compaction_item: Value) {
    let input_len = state.input.len();
    let message_tail = split_current_turn(&mut state.messages, input_len);
    let persisted_tail = split_current_turn(&mut state.persisted_messages, input_len);

    state.messages.clear();
    state.messages.push(compaction_item.clone());
    state.messages.extend(message_tail);

    state.persisted_messages.clear();
    state.persisted_messages.push(compaction_item);
    state.persisted_messages.extend(persisted_tail);
}

/// Move the current-turn tail off `items`, leaving history behind to drop.
fn split_current_turn(items: &mut Vec<Value>, input_len: usize) -> Vec<Value> {
    let start = items.len().saturating_sub(input_len);
    items.split_off(start)
}

/// Build a text representation of instructions and tool definitions for token counting.
///
/// Returns an empty string when neither field is present. The result is
/// concatenated with conversation text so tiktoken counts the full context
/// window overhead, matching the behavior of `previous_usage.total_tokens`.
fn build_context_overhead_text(request_body: &Value) -> String {
    let mut buf = String::new();
    if let Some(instructions) = request_body.get("instructions").and_then(Value::as_str)
        && !instructions.is_empty()
    {
        append_line(&mut buf, "instructions", instructions);
    }
    if let Some(tools) = request_body.get("tools").and_then(Value::as_array)
        && !tools.is_empty()
    {
        let serialized = serde_json::to_string(tools).unwrap_or_default();
        append_line(&mut buf, "tools", &serialized);
    }
    buf
}

/// Format a message array as readable text for the summarization prompt.
///
/// Each message becomes `<label>: <text>`, separated by blank lines.
/// Handles regular messages, tool calls, tool outputs, and prior compaction items.
fn build_conversation_text(messages: &[Value]) -> String {
    let mut buf = String::with_capacity(messages.len() * 100);
    for msg in messages {
        append_item(&mut buf, msg);
    }
    buf
}

/// Append a single conversation item to the text buffer.
fn append_item(buf: &mut String, msg: &Value) {
    match msg.get("type").and_then(Value::as_str) {
        Some("compaction") => append_compaction_summary(buf, msg),
        Some("function_call") => {
            let name = msg.get("name").and_then(Value::as_str).unwrap_or("unknown");
            let args = msg.get("arguments").and_then(Value::as_str).unwrap_or("");
            append_function_call(buf, name, args);
        },
        Some("function_call_output") => {
            let output = msg.get("output").and_then(Value::as_str).unwrap_or("");
            if !output.is_empty() {
                append_line(buf, "function_call_output", output);
            }
        },
        _ => {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("unknown");
            let content = extract_content(msg);
            if !content.is_empty() {
                append_line(buf, role, &content);
            }
        },
    }
}

/// Append `"<label>: <text>"` to `buf`, preceded by a blank line if not empty.
fn append_line(buf: &mut String, label: &str, text: &str) {
    if !buf.is_empty() {
        buf.push_str("\n\n");
    }
    buf.push_str(label);
    buf.push_str(": ");
    buf.push_str(text);
}

/// Append a compaction item's summary to the buffer.
///
/// Decodes `encrypted_content` and borrows the result as `&str` directly,
/// avoiding the `String::from_utf8` conversion that would copy the buffer.
fn append_compaction_summary(buf: &mut String, msg: &Value) {
    let Some(encoded) = msg.get("encrypted_content").and_then(Value::as_str) else {
        return;
    };
    let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        return;
    };
    if let Ok(summary) = std::str::from_utf8(&decoded)
        && !summary.is_empty()
    {
        append_line(buf, "[previous context summary]", summary);
    }
}

/// Append a function call entry without a temporary allocation.
fn append_function_call(buf: &mut String, name: &str, args: &str) {
    if !buf.is_empty() {
        buf.push_str("\n\n");
    }
    buf.push_str("function_call: ");
    buf.push_str(name);
    buf.push('(');
    buf.push_str(args);
    buf.push(')');
}

/// Extract text content from a message's `content` field.
///
/// Content can be a plain string, an array of content parts
/// (each with a `"text"` field), or absent/null.
///
/// Returns `Cow::Borrowed` for plain strings (zero-copy) and
/// `Cow::Owned` for array content that must be joined.
fn extract_content(msg: &Value) -> Cow<'_, str> {
    let Some(content) = msg.get("content") else {
        return Cow::Borrowed("");
    };
    if let Some(s) = content.as_str() {
        return Cow::Borrowed(s);
    }
    if let Some(arr) = content.as_array() {
        let mut joined = String::new();
        for part in arr {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !joined.is_empty() {
                    joined.push(' ');
                }
                joined.push_str(text);
            }
        }
        if !joined.is_empty() {
            return Cow::Owned(joined);
        }
        return Cow::Borrowed("");
    }
    Cow::Borrowed("")
}
