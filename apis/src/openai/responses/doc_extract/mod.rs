// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Extracts document content from `input_file` parts and converts
//! them to `input_text` for inference backends that do not natively
//! support `input_file` (e.g. vLLM).
//!
//! Walks `message` content arrays and `function_call_output` output
//! arrays, finds `input_file` parts with inline `file_data`, and
//! replaces text-safe documents with `input_text` containing the
//! decoded UTF-8 text.
//!
//! This filter is an explicitly configured backend adapter. It
//! should only be enabled for routes to backends that cannot consume
//! `input_file` parts directly. For `OpenAI`-compatible backends
//! with native document support, leave this filter out of the
//! pipeline.
//!
//! Runs after `openai_file_resolve` (which resolves `file_id` to
//! inline `file_data`) and before `openai_responses_proxy` (which
//! rebuilds the body from state). Parts without inline `file_data`
//! (unresolved `file_id` or `file_url`) are skipped — this filter
//! does not perform network I/O.
//!
//! Text-safe MIME types (`text/*`, `application/json`,
//! `application/xml`) are decoded from base64 and validated as
//! UTF-8. Unsupported MIME types are either left unchanged
//! (`on_unsupported: continue`) or rejected (`on_unsupported:
//! reject`).
//!
//! When [`ResponsesState`] is present (e.g. after `rehydrate`),
//! converted content is synced back into `state.request_body`,
//! `state.messages`, and `state.persisted_messages` so that
//! `responses_proxy` does not overwrite the rewritten body.
//!
//! [`ResponsesState`]: super::state::ResponsesState

pub(crate) mod config;
mod extract;

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

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, BoundUpstreamBodyOutcome, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    Rejection, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    config::{DocExtractConfig, validate_config},
    extract::{ExtractError, ExtractionBudget, extract_input_file, parse_data_uri},
};
use super::{
    agentic_loop::{
        AgenticBudgetPolicy,
        budget::{SimpleBudget, input_charge},
    },
    body_limits::reject_rewritten_body_too_large,
    bound_body_outcome,
    content_parts::{content_parts, content_parts_mut, infer_mime_from_filename},
    error::responses_error_rejection,
    openai_responses_proxy::serialized_outbound_body_len,
    state::ResponsesState,
};
use crate::{classifier::is_responses_create, json_body::serialize_json_body};

/// Converts `input_file` content parts to `input_text` for backends
/// that do not support `input_file` natively (e.g. vLLM, llm-d).
///
/// # YAML
///
/// ```yaml
/// filter: openai_doc_extract
/// allow_pre_security_callout: true
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_doc_extract
/// allow_pre_security_callout: true
/// on_unsupported: continue
/// max_rewritten_body_bytes: 67108864
/// max_content_bytes: 10485760
/// max_file_references: 32
/// max_total_text_bytes: 67108864
/// ```
pub struct DocExtractFilter {
    /// Validated filter configuration.
    config: DocExtractConfig,
}

impl DocExtractFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: DocExtractConfig = parse_filter_config("openai_doc_extract", config)?;
        let validated = validate_config(cfg)?;

        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for DocExtractFilter {
    fn name(&self) -> &'static str {
        "openai_doc_extract"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn bound_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        // Accept up to the absolute ceiling; the pipeline's body_limits
        // decides the real raw cap. max_rewritten_body_bytes bounds only
        // the body produced after input_file → input_text conversion.
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

        if !is_responses_create(&ctx.request.method, ctx.request.uri.path()) {
            trace!("skipping non-create request");
            return Ok(FilterAction::Release);
        }

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            trace!("skipping non-responses request");
            return Ok(FilterAction::Release);
        }

        let Some(raw) = body.as_ref() else {
            trace!("no body, releasing");
            return Ok(FilterAction::Release);
        };

        // The ingress charge covers the canonical request owners. This filter
        // parses a second tree while those owners remain live, so check that
        // temporary projection before serde allocates it.
        let doc_budget = match preflight_doc_parse(ctx, raw) {
            Ok(budget) => budget,
            Err(action) => return Ok(action),
        };

        let parsed: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, "body is not valid JSON, releasing");
                return Ok(FilterAction::Release);
            },
        };

        extract_and_rewrite(self, ctx, body, parsed, doc_budget)
    }

    async fn on_bound_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<BoundUpstreamBodyOutcome, FilterError> {
        let action = self.on_request_body(ctx, body, true).await?;
        bound_body_outcome(action)
    }
}

/// Run extraction on the current input and history, then rewrite
/// the body and sync state.
///
/// Takes ownership of the parsed body so the extracted value can be
/// moved into [`ResponsesState`] instead of deep-cloned.
fn extract_and_rewrite(
    filter: &DocExtractFilter,
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    mut parsed: serde_json::Value,
    doc_budget: Option<SimpleBudget>,
) -> Result<FilterAction, FilterError> {
    let max_bytes = filter.config.max_rewritten_body_bytes;
    let mut budget = ExtractionBudget::new(&filter.config);

    let next_budget = match reserve_document_growth(ctx, &parsed, body.as_deref(), doc_budget) {
        Ok(budget) => budget,
        Err(action) => return Ok(action),
    };

    let count = match extract_current_input(&mut parsed, &mut budget) {
        Ok(count) => count,
        Err(e) => return Ok(reject_extract_error(&e)),
    };

    if count == 0 {
        return finish_history_only(ctx, &mut budget, max_bytes, next_budget);
    }

    debug!(count, "extracted input_file parts");
    if let Some(rejection) = rewrite_body(body, &parsed, max_bytes, filter.name())? {
        return Ok(rejection);
    }
    if let Err(e) = sync_state_after_rewrite(ctx, parsed, &mut budget) {
        return Ok(reject_extract_error(&e));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, max_bytes)? {
        return Ok(rejection);
    }

    commit_document_budget(ctx, next_budget);

    Ok(FilterAction::Continue)
}

/// Process restored documents when the current input has no inline file.
fn finish_history_only(
    ctx: &mut HttpFilterContext<'_>,
    budget: &mut ExtractionBudget,
    max_bytes: usize,
    next_budget: Option<SimpleBudget>,
) -> Result<FilterAction, FilterError> {
    trace!("no input_file parts to extract");
    if let Err(e) = extract_state_history(ctx, budget) {
        return Ok(reject_extract_error(&e));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, max_bytes)? {
        return Ok(rejection);
    }
    commit_document_budget(ctx, next_budget);
    Ok(FilterAction::Continue)
}

/// Reject a budgeted pipeline that has no admitted request owner, and bound
/// the extra JSON tree before parsing the body again.
fn preflight_doc_parse(ctx: &HttpFilterContext<'_>, raw: &[u8]) -> Result<Option<SimpleBudget>, FilterAction> {
    if ctx.extensions.get::<AgenticBudgetPolicy>().is_none() {
        return Ok(None);
    }
    let Some(shared) = ctx
        .extensions
        .get::<ResponsesState>()
        .and_then(|state| state.simple_budget)
    else {
        return Err(FilterAction::Reject(responses_error_rejection(
            400,
            "invalid_request_error",
            "document extraction requires admitted Responses state under openai_agentic_loop.max_retained_bytes",
        )));
    };
    let parsed = input_charge(raw).unwrap_or(usize::MAX);
    if shared.remaining_bytes().is_none_or(|remaining| parsed > remaining) {
        return Err(reject_retained_document_budget());
    }
    Ok(Some(shared))
}

/// Keep the temporary parse and the replacement owners within the same
/// request budget before decoding or cloning any document content.
fn reserve_document_growth(
    ctx: &HttpFilterContext<'_>,
    parsed: &serde_json::Value,
    raw: Option<&[u8]>,
    budget: Option<SimpleBudget>,
) -> Result<Option<SimpleBudget>, FilterAction> {
    let Some(mut shared) = budget else { return Ok(None) };
    let growth = projected_document_growth(ctx, parsed).ok_or_else(reject_retained_document_budget)?;
    let parse_charge = raw.and_then(input_charge).unwrap_or(usize::MAX);
    let peak = parse_charge.saturating_add(growth);
    if shared.remaining_bytes().is_none_or(|remaining| peak > remaining) || !shared.reserve_additional_input(growth) {
        return Err(reject_retained_document_budget());
    }
    Ok(Some(shared))
}

/// Reserve each independently rewritten document before base64 decode. The
/// source tree, decoded buffer, new text, serialized body, request state, and
/// two message projections overlap. Sixfold JSON escaping can enlarge decoded
/// control characters, so 64 bytes per possible text byte covers those copies
/// and the temporary serializer capacity. A fixed part charge covers maps and
/// Vec spare capacity even for a one-byte document.
fn projected_document_growth(ctx: &HttpFilterContext<'_>, parsed: &serde_json::Value) -> Option<usize> {
    let mut growth = parsed
        .get("input")
        .and_then(serde_json::Value::as_array)
        .map_or(Some(0), |items| parts_growth(items))?;
    if let Some(state) = ctx.extensions.get::<ResponsesState>() {
        let history_end = state.messages.len().checked_sub(state.input.len())?;
        let persisted_end = state.persisted_messages.len().checked_sub(state.input.len())?;
        growth = growth
            .checked_add(parts_growth(state.messages.get(..history_end)?)?)?
            .checked_add(parts_growth(state.persisted_messages.get(..persisted_end)?)?)?;
    }
    Some(growth)
}

/// Sum charges for every independently rewritten part in one message vector.
fn parts_growth(items: &[serde_json::Value]) -> Option<usize> {
    let mut charge = 0_usize;
    for item in items {
        let Some(parts) = content_parts(item) else { continue };
        for part in parts {
            charge = charge.checked_add(part_growth(part)?)?;
        }
    }
    Some(charge)
}

/// Conservatively charge decoded text and serialized JSON projections.
fn part_growth(part: &serde_json::Value) -> Option<usize> {
    const TEXT_OWNER_MULTIPLIER: usize = 64;
    const PART_NODE_RESERVE: usize = 8_192;
    if part.get("type").and_then(serde_json::Value::as_str) != Some("input_file") {
        return Some(0);
    }
    let Some(data) = part.get("file_data").and_then(serde_json::Value::as_str) else {
        return Some(0);
    };
    let filename = part.get("filename").and_then(serde_json::Value::as_str);
    let mime = parse_data_uri(data)
        .map(|uri| uri.mime)
        .or_else(|| infer_mime_from_filename(filename))
        .unwrap_or("application/octet-stream");
    if !config::is_text_safe_mime(mime) {
        return Some(0);
    }
    let prefix = filename
        .filter(|name| !name.is_empty())
        .map_or(0, str::len)
        .checked_add(11)?;
    data.len()
        .checked_add(prefix)?
        .checked_mul(TEXT_OWNER_MULTIPLIER)?
        .checked_add(PART_NODE_RESERVE)
}

/// Publish the growth reservation only after every rewrite and size check succeeds.
fn commit_document_budget(ctx: &mut HttpFilterContext<'_>, budget: Option<SimpleBudget>) {
    if let Some(budget) = budget
        && let Some(state) = ctx.extensions.get_mut::<ResponsesState>()
    {
        state.simple_budget = Some(budget);
    }
}

/// Return one consistent request-wide overflow response for this filter.
fn reject_retained_document_budget() -> FilterAction {
    FilterAction::Reject(responses_error_rejection(
        413,
        "invalid_request_error",
        "document extraction exceeds openai_agentic_loop.max_retained_bytes",
    ))
}

/// Walk the current request input and extract text-safe `input_file` parts.
fn extract_current_input(parsed: &mut serde_json::Value, budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let Some(items) = parsed.get_mut("input").and_then(serde_json::Value::as_array_mut) else {
        return Ok(0);
    };
    extract_items(items, budget)
}

/// Walk items and extract text-safe `input_file` content parts.
fn extract_items(items: &mut [serde_json::Value], budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let mut count = 0;
    for item in items.iter_mut() {
        count += extract_item_parts(item, budget)?;
    }
    Ok(count)
}

/// Extract text-safe `input_file` parts from a single item.
fn extract_item_parts(item: &mut serde_json::Value, budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let Some(parts) = content_parts_mut(item) else {
        return Ok(0);
    };
    let mut count = 0;
    for part in parts.iter_mut() {
        if part.get("type").and_then(serde_json::Value::as_str) != Some("input_file") {
            continue;
        }
        if let Some(text) = extract_input_file(part, budget)? {
            *part = serde_json::json!({"type": "input_text", "text": text});
            count += 1;
        }
    }
    Ok(count)
}

/// Serialize the extracted JSON and replace the buffered request body.
fn rewrite_body(
    body: &mut Option<Bytes>,
    parsed: &serde_json::Value,
    max_rewritten_body_bytes: usize,
    filter_name: &'static str,
) -> Result<Option<FilterAction>, FilterError> {
    let rewritten = serialize_json_body(parsed)
        .map_err(|e| -> FilterError { format!("{filter_name}: failed to serialize body: {e}").into() })?;
    if rewritten.len() > max_rewritten_body_bytes {
        warn!(
            actual = rewritten.len(),
            limit = max_rewritten_body_bytes,
            "rewritten request body exceeds configured limit"
        );
        return Ok(Some(reject_rewritten_body_too_large(
            rewritten.len(),
            max_rewritten_body_bytes,
        )));
    }
    rewritten.commit(body, filter_name, "input");
    Ok(None)
}

/// Sync converted content back into [`ResponsesState`] after a body
/// rewrite.
///
/// Takes `resolved_body` by value and moves it into `request_body`
/// rather than deep-cloning a tree that may carry inlined file data.
fn sync_state_after_rewrite(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: serde_json::Value,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    state.request_body = resolved_body;

    let input_len = state.input.len();
    let ResponsesState {
        request_body,
        messages,
        persisted_messages,
        ..
    } = state;

    let Some(resolved_input) = request_body.get("input").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };

    sync_message_history(messages, input_len, Some(resolved_input), budget)?;
    sync_persisted_history(persisted_messages, input_len, Some(resolved_input), budget)
}

/// Test helper that creates an isolated request extraction budget.
#[cfg(test)]
fn sync_state(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: serde_json::Value,
    config: &DocExtractConfig,
) -> Result<(), ExtractError> {
    let mut budget = ExtractionBudget::new(config);
    sync_state_after_rewrite(ctx, resolved_body, &mut budget)
}

/// Extract `input_file` parts in rehydrated history when the
/// current input had no `input_file` parts to extract.
fn extract_state_history(ctx: &mut HttpFilterContext<'_>, budget: &mut ExtractionBudget) -> Result<(), ExtractError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    let input_len = state.input.len();

    sync_message_history(&mut state.messages, input_len, None, budget)?;
    sync_persisted_history(&mut state.persisted_messages, input_len, None, budget)
}

/// Sync the persisted-messages mirror with independent count and
/// byte accounting.
fn sync_persisted_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let saved = budget.begin_independent_accounting();
    let result = sync_message_history(messages, input_len, resolved_input, budget);
    budget.restore_accounting(&saved);
    result
}

/// Replace the current-input tail, then extract text-safe
/// `input_file` parts from the history prefix.
fn sync_message_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let Some(history_end) = messages.len().checked_sub(input_len) else {
        return Ok(());
    };
    if let Some(resolved_input) = resolved_input {
        replace_tail(messages, history_end, resolved_input);
    }
    extract_history(messages, history_end, budget)
}

/// Copy resolved input items into the current-input tail of a
/// message vector, starting at `history_end`.
fn replace_tail(messages: &mut [serde_json::Value], history_end: usize, resolved_input: &[serde_json::Value]) {
    for (i, item) in resolved_input.iter().enumerate() {
        if let Some(slot) = messages.get_mut(history_end + i) {
            *slot = item.clone();
        }
    }
}

/// Extract text-safe `input_file` parts from history messages (the
/// prefix before the current input).
fn extract_history(
    messages: &mut [serde_json::Value],
    history_end: usize,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    if history_end == 0 {
        return Ok(());
    }
    let Some(history) = messages.get_mut(..history_end) else {
        return Ok(());
    };
    extract_items(history, budget).map(|_count| ())
}

/// Enforce the body limit against the exact request shape that
/// `openai_responses_proxy` will later serialize from state.
fn reject_oversized_state_body(
    ctx: &HttpFilterContext<'_>,
    max_rewritten_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(None);
    };
    let len = serialized_outbound_body_len(state).map_err(|e| -> FilterError {
        format!("openai_doc_extract: failed to measure rebuilt request body: {e}").into()
    })?;
    Ok((len > max_rewritten_body_bytes).then(|| {
        warn!(
            actual = len,
            limit = max_rewritten_body_bytes,
            "rebuilt state body exceeds configured limit"
        );
        reject_rewritten_body_too_large(len, max_rewritten_body_bytes)
    }))
}

// -- Error responses ------------------------------------------------

/// Map one extraction error to an HTTP rejection.
fn reject_extract_error(err: &ExtractError) -> FilterAction {
    let (status, message) = extract_error_response(err);

    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "doc_extract_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}

/// Map an extraction error to an HTTP status code and message.
fn extract_error_response(err: &ExtractError) -> (u16, String) {
    let (status, message) = match err {
        ExtractError::DecodeFailed { detail } => (400, format!("file_data decode failed: {detail}")),
        ExtractError::TooManyReferences { limit } => (413, format!("request exceeds {limit} input_file references")),
        ExtractError::TooLarge { detail, limit } => (413, format!("extracted content exceeds {limit} bytes: {detail}")),
        ExtractError::Unsupported { mime } => (400, format!("unsupported file type: {mime}")),
    };
    warn!(%err, "extraction error");
    (status, message)
}
