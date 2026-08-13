// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Compact filter: token counting and context window management.
//!
//! When a request's `context_management` contains a compaction
//! configuration and the token count exceeds the specified threshold,
//! this filter summarizes the conversation history via a sub-request
//! to an inference backend, replacing it with a single compaction
//! item. Runs after `rehydrate` (which populates messages and
//! previous usage) and after `openai_tool_parse`.
//!
//! # Scope
//!
//! Compaction only applies to **multi-turn requests** where
//! `openai_responses_rehydrate` has loaded stored conversation history
//! — i.e. requests that include `previous_response_id` or
//! `conversation`. Single-turn requests (no stored history, even with
//! `context_management` set) are released without compaction because
//! there is no prior history to summarize.

pub(super) mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
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
    openai::responses::config_validation::FailureMode,
    store::{ResponseRecord, ResponseStoreRegistry},
    subrequest::{self, SubRequest, SubRequestClient},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum response body size for summarization callouts (1 MiB).
const MAX_SUMMARIZATION_RESPONSE_BYTES: usize = 1_048_576;

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
struct CompactionParams {
    /// Token threshold above which compaction triggers.
    compact_threshold: u64,
    /// Optional model override for the summarization call.
    compaction_model: Option<String>,
}

// -----------------------------------------------------------------------------
// CompactFilter
// -----------------------------------------------------------------------------

/// Summarizes conversation history when the token count exceeds a
/// configured threshold.
///
/// `compact_threshold` in `context_management` must be an integer.
/// Floating-point values (e.g. `0.9`) are ignored and compaction
/// is skipped.
///
/// Compaction only applies to multi-turn requests where
/// `openai_responses_rehydrate` has loaded stored conversation
/// history. Single-turn requests are released without compaction.
///
/// # YAML
///
/// ```yaml
/// filter: openai_responses_compact
/// inference_url: "http://localhost:11434/v1/chat/completions"
/// default_model: llama3.2:1b
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_responses_compact
/// inference_url: "http://localhost:11434/v1/chat/completions"
/// default_model: gpt-4o-mini
/// tiktoken_encoding: cl100k_base
/// summary_prefix: "[Previous conversation summary]\n\n"
/// timeout_ms: 30000
/// callout_failure_mode: closed
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
        let cfg: CompactFilterConfig = parse_filter_config("openai_responses_compact", config)?;
        let validated = build_config(&cfg)?;
        eager_init_tiktoken(&validated.tiktoken_encoding);
        Ok(Box::new(Self {
            client,
            config: validated,
        }))
    }

    /// Run the summarization callout and return the summary text.
    ///
    /// Returns `Ok(Some(summary))` on success, `Ok(None)` when
    /// compaction should be skipped, or `Err(FilterAction)` to
    /// short-circuit the request.
    async fn execute_compaction(
        &self,
        state: &ResponsesState,
        params: &CompactionParams,
        streaming: bool,
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
        )
        .await;
        self.handle_subrequest_result(result, streaming)
    }

    /// Map a subrequest result to a summary string or a filter action.
    fn handle_subrequest_result(
        &self,
        result: Result<subrequest::SubResponse, subrequest::SubRequestError>,
        streaming: bool,
    ) -> Result<Option<String>, FilterAction> {
        match result {
            Ok(resp) if (200..300).contains(&(resp.status as usize)) => {
                parse_summarization_response(&resp.body).map(Some).or_else(|e| {
                    warn!(error = %e, "failed to parse summarization response");
                    self.on_callout_error("failed to parse summarization response", streaming)
                })
            },
            Ok(resp) => {
                warn!(status = resp.status, "summarization callout returned non-2xx");
                self.on_callout_error("summarization callout rejected", streaming)
            },
            Err(e) => {
                warn!(error = %e, "summarization callout failed");
                self.on_callout_error("summarization callout failed", streaming)
            },
        }
    }

    /// Run a summarization callout for an explicit compact request.
    async fn summarize_messages(
        &self,
        req: &ExplicitCompactRequest,
        messages: &[Value],
    ) -> Result<String, FilterAction> {
        let conversation_text = build_conversation_text(messages);
        let model = req.model.as_deref().unwrap_or(&self.config.default_model);
        let instructions = req.instructions.as_deref();
        let request = build_summarization_request(&conversation_text, instructions, model);
        let timeout = Duration::from_millis(self.config.callout.timeout_ms);
        let result = subrequest::execute_url(
            &self.client,
            &self.config.inference_url,
            request,
            MAX_SUMMARIZATION_RESPONSE_BYTES,
            timeout,
        )
        .await;
        match self.handle_subrequest_result(result, false) {
            Ok(Some(s)) => Ok(s),
            Ok(None) => Err(FilterAction::Reject(responses_error_rejection(
                502,
                "server_error",
                "compaction callout failed",
                false,
            ))),
            Err(action) => Err(action),
        }
    }

    /// Apply the configured open/closed policy on a callout error.
    fn on_callout_error(&self, message: &str, streaming: bool) -> Result<Option<String>, FilterAction> {
        match self.config.callout.failure_mode {
            FailureMode::Open => Ok(None),
            FailureMode::Closed => Err(FilterAction::Reject(responses_error_rejection(
                self.config.callout.status_on_error,
                "server_error",
                message,
                streaming,
            ))),
        }
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
        let record = fetch_response_blocking(&*store, &tenant_id, &req)?;
        let messages = extract_stored_messages(&record)?;
        let summary = self.summarize_messages(&req, &messages).await?;
        let response_object = build_and_persist_compaction(self, ctx, &*store, &tenant_id, &req, &summary)?;
        let body_bytes = serde_json::to_vec(&response_object).unwrap_or_default();
        Ok(FilterAction::Reject(
            praxis_filter::Rejection::status(200)
                .with_header("content-type", "application/json")
                .with_body(Bytes::from(body_bytes)),
        ))
    }
}

#[async_trait]
#[expect(clippy::too_many_lines, reason = "filter pipeline method with sequential stages")]
impl HttpFilter for CompactFilter {
    fn name(&self) -> &'static str {
        "openai_responses_compact"
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
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        if is_explicit_compact_request(ctx) {
            return self.handle_explicit_compact(ctx, _body).await;
        }
        if !is_responses_request(ctx) {
            return Ok(FilterAction::Release);
        }
        let streaming = is_streaming(ctx);
        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            return Ok(FilterAction::Release);
        };
        if !state.history_rehydrated {
            return Ok(FilterAction::Release);
        }
        let Some((params, conversation_text)) = should_compact(state, &self.config.tiktoken_encoding) else {
            return Ok(FilterAction::Release);
        };
        let compaction = self.execute_compaction(state, &params, streaming, &conversation_text);
        let summary = match compaction.await {
            Ok(Some(s)) => s,
            Ok(None) | Err(FilterAction::Release) => return Ok(FilterAction::Release),
            Err(action) => return Ok(action),
        };
        let model = params
            .compaction_model
            .as_deref()
            .unwrap_or(&self.config.default_model)
            .to_owned();
        let compaction_id = format!("compact_{}", ctx.id_generator.generate(ctx.time_source));
        let compaction_resp_id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
        let created_at = i64::try_from(ctx.time_source.now().as_secs()).unwrap_or(i64::MAX);
        let store = ctx
            .extensions
            .get::<ResponseStoreRegistry>()
            .and_then(|r| r.get("default"));
        let tenant_id = ctx.get_metadata("responses.tenant_id").unwrap_or("default").to_owned();
        let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
            return Ok(FilterAction::Release);
        };
        replace_messages(
            state,
            build_compaction_item(&compaction_id, &summary, &self.config.summary_prefix),
        );
        if let Some(store) = store.as_deref() {
            persist_compaction_response(store, &compaction_resp_id, &model, &tenant_id, created_at, state);
        }
        ctx.set_metadata("responses.compacted", "true");
        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Compaction Logic
// -----------------------------------------------------------------------------

/// Check whether compaction should run and return the params + text.
///
/// Returns `None` if there is no compaction config, the encoding is
/// unknown, or the token count is below the threshold.
///
/// When `previous_usage` is available from the rehydrated response,
/// its `total_tokens` is used directly — avoiding the cost of BPE
/// tokenization. Falls back to tiktoken estimation otherwise.
///
/// The token estimate includes instructions and tool definitions in
/// addition to conversation messages, since all three contribute to
/// the rendered context sent to the model.
fn should_compact(state: &ResponsesState, tiktoken_encoding: &str) -> Option<(CompactionParams, String)> {
    let params = extract_compaction_config(&state.context_management)?;

    let conversation_text = build_conversation_text(&state.messages);

    let token_count =
        previous_usage_total(state).or_else(|| get_token_count(&conversation_text, tiktoken_encoding))?;

    if token_count <= params.compact_threshold {
        debug!(
            token_count,
            threshold = params.compact_threshold,
            "under threshold, skipping"
        );
        return None;
    }
    debug!(
        token_count,
        threshold = params.compact_threshold,
        "threshold exceeded, compacting"
    );
    Some((params, conversation_text))
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
fn is_explicit_compact_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.request.method == http::Method::POST && ctx.request.uri.path().trim_end_matches('/') == "/v1/responses/compact"
}

/// Check whether this is an OpenAI Responses API request.
fn is_responses_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.format") == Some("openai_responses")
}

/// Check whether the client requested streaming.
fn is_streaming(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream")
        .is_some_and(|v| v == "true")
}

// -----------------------------------------------------------------------------
// Explicit Compact Endpoint Helpers
// -----------------------------------------------------------------------------

/// Parsed body for `POST /v1/responses/compact`.
struct ExplicitCompactRequest {
    /// The stored response to compact.
    response_id: String,
    /// Optional model override for the summarization call.
    model: Option<String>,
    /// Optional instructions to prepend to the summarization prompt.
    instructions: Option<String>,
}

/// Parse and validate the `POST /v1/responses/compact` body.
fn parse_compact_request_body(body: &Option<Bytes>) -> Result<ExplicitCompactRequest, FilterAction> {
    let bytes = body
        .as_ref()
        .filter(|b| !b.is_empty())
        .ok_or_else(|| reject_compact(400, "invalid_request_error", "request body is empty"))?;
    let parsed: Value = serde_json::from_slice(bytes).map_err(|e| {
        debug!(error = %e, "compact request body parse failed");
        reject_compact(400, "invalid_request_error", "invalid JSON body")
    })?;
    let response_id = parsed
        .get("response_id")
        .and_then(Value::as_str)
        .ok_or_else(|| reject_compact(400, "invalid_request_error", "missing required field: response_id"))?
        .to_owned();
    Ok(ExplicitCompactRequest {
        response_id,
        model: parsed.get("model").and_then(Value::as_str).map(ToOwned::to_owned),
        instructions: parsed
            .get("instructions")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
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

/// Fetch a stored response, blocking the current thread.
fn fetch_response_blocking(
    store: &dyn crate::store::ResponseStore,
    tenant_id: &str,
    req: &ExplicitCompactRequest,
) -> Result<ResponseRecord, FilterAction> {
    let handle = tokio::runtime::Handle::current();
    match tokio::task::block_in_place(|| handle.block_on(store.get_response(tenant_id, &req.response_id))) {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(reject_compact(404, "not_found_error", "response not found")),
        Err(e) => {
            warn!(error = %e, "failed to fetch response for compact");
            Err(reject_compact(500, "server_error", "failed to fetch response"))
        },
    }
}

/// Extract messages from a stored response.
fn extract_stored_messages(record: &ResponseRecord) -> Result<Vec<Value>, FilterAction> {
    let messages: Vec<Value> = match &record.messages {
        Value::Array(arr) => arr.clone(),
        _ => Vec::new(),
    };
    if messages.is_empty() {
        return Err(reject_compact(
            400,
            "invalid_request_error",
            "response has no messages to compact",
        ));
    }
    Ok(messages)
}

/// Build and persist the compaction result for an explicit compact request.
#[expect(clippy::too_many_arguments, reason = "all parameters are needed")]
fn build_and_persist_compaction(
    filter: &CompactFilter,
    ctx: &HttpFilterContext<'_>,
    store: &dyn crate::store::ResponseStore,
    tenant_id: &str,
    req: &ExplicitCompactRequest,
    summary: &str,
) -> Result<Value, FilterAction> {
    let compaction_id = format!("compact_{}", ctx.id_generator.generate(ctx.time_source));
    let resp_id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
    let created_at = i64::try_from(ctx.time_source.now().as_secs()).unwrap_or(i64::MAX);
    let model = req.model.as_deref().unwrap_or(&filter.config.default_model);
    let compaction_item = build_compaction_item(&compaction_id, summary, &filter.config.summary_prefix);
    let compacted_messages = Value::Array(vec![compaction_item]);
    let response_object = serde_json::json!({
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "previous_response_id": req.response_id,
    });

    let record = ResponseRecord {
        id: resp_id,
        tenant_id: tenant_id.to_owned(),
        created_at,
        model: model.to_owned(),
        response_object: response_object.clone(),
        input: compacted_messages.clone(),
        messages: compacted_messages,
    };
    let handle = tokio::runtime::Handle::current();
    tokio::task::block_in_place(|| handle.block_on(store.upsert_response(&record))).map_err(|e| {
        warn!(error = %e, "failed to persist explicit compaction response");
        reject_compact(500, "server_error", "failed to persist compaction response")
    })?;
    Ok(response_object)
}

/// Build a `FilterAction::Reject` for an explicit compact error.
fn reject_compact(status: u16, code: &str, message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(status, code, message, false))
}

/// Parse the `context_management` JSON to find a compaction config.
///
/// The `context_management` field is an array like:
/// `[{"type": "compaction", "compact_threshold": 50000}]`
///
/// Returns `None` if no compaction entry is found.
fn extract_compaction_config(context_management: &Option<Value>) -> Option<CompactionParams> {
    let array = context_management.as_ref()?.as_array()?;

    for entry in array {
        let Some(entry_type) = entry.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        if entry_type != "compaction" {
            continue;
        }
        let Some(raw_threshold) = entry.get("compact_threshold") else {
            continue;
        };
        let Some(compact_threshold) = raw_threshold.as_u64() else {
            warn!(value = %raw_threshold, "compact_threshold is not a valid integer, skipping compaction");
            continue;
        };
        let compaction_model = entry
            .get("compaction_model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return Some(CompactionParams {
            compact_threshold,
            compaction_model,
        });
    }
    None
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

/// Parse the Chat Completions response and extract the summary text.
///
/// Expected shape: `{"choices": [{"message": {"content": "..."}}]}`
fn parse_summarization_response(body: &[u8]) -> Result<String, String> {
    match serde_json::from_slice::<Value>(body) {
        Ok(body) => body
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| "Chat Completions response missing choices[0].message.content".to_owned()),
        Err(err) => Err(format!("failed to parse Chat Completions response JSON: {err}")),
    }
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
/// - `state.messages` = `[compaction_item, ...state.input]`
/// - `state.persisted_messages` = `[compaction_item, ...state.input]`
///
/// The compaction item is `{"type": "compaction", "encrypted_content": "<base64>"}`.
/// `state.input` holds the current request's input items (unchanged
/// by rehydrate), so the current turn's messages are preserved.
fn replace_messages(state: &mut ResponsesState, compaction_item: Value) {
    let mut new_messages = Vec::with_capacity(state.input.len() + 1);
    new_messages.push(compaction_item);
    new_messages.extend(state.input.iter().cloned());
    state.persisted_messages = new_messages.clone();
    state.messages = new_messages;
}

/// Persist a hidden compaction response to the store so that it can
/// be referenced via `previous_response_id` in future requests.
///
/// Best-effort: a store failure is logged but does not block the
/// request — the main response's store filter will still persist
/// the compacted messages as part of the regular response record.
#[expect(clippy::too_many_arguments, reason = "all fields are needed for the record")]
fn persist_compaction_response(
    store: &dyn crate::store::ResponseStore,
    response_id: &str,
    model: &str,
    tenant_id: &str,
    created_at: i64,
    state: &ResponsesState,
) {
    let record = ResponseRecord {
        id: response_id.to_owned(),
        tenant_id: tenant_id.to_owned(),
        created_at,
        model: model.to_owned(),
        response_object: serde_json::json!({
            "id": response_id,
            "object": "response",
            "status": "completed",
        }),
        input: Value::Array(state.persisted_messages.clone()),
        messages: Value::Array(state.persisted_messages.clone()),
    };
    let handle = tokio::runtime::Handle::current();
    if let Err(e) = tokio::task::block_in_place(|| handle.block_on(store.upsert_response(&record))) {
        warn!(error = %e, id = %response_id, "failed to persist compaction response");
    }
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

/// Decode a compaction item's `encrypted_content` field to a UTF-8 string.
/// Append a compaction item's summary to the buffer without a String allocation.
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
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        return Cow::Owned(texts.join(" "));
    }
    Cow::Borrowed("")
}
