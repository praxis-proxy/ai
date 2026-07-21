// SPDX-License-Identifier: MIT
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

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    config::{DocExtractConfig, validate_config},
    extract::{ExtractError, ExtractionBudget, extract_input_file},
};
use super::{
    file_resolve::resolve::content_parts_mut, openai_responses_proxy::serialized_outbound_body_len,
    state::ResponsesState,
};
use crate::classifier::is_responses_create;

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
/// max_body_bytes: 67108864
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

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
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
        if raw.len() > self.config.max_body_bytes {
            return Ok(reject_raw_body_too_large(raw.len(), self.config.max_body_bytes));
        }

        let mut parsed: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, "body is not valid JSON, releasing");
                return Ok(FilterAction::Release);
            },
        };

        extract_and_rewrite(self, ctx, body, &mut parsed)
    }
}

/// Run extraction on the current input and history, then rewrite
/// the body and sync state.
fn extract_and_rewrite(
    filter: &DocExtractFilter,
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    parsed: &mut serde_json::Value,
) -> Result<FilterAction, FilterError> {
    let mut budget = ExtractionBudget::new(&filter.config);

    let count = match extract_current_input(parsed, &mut budget) {
        Ok(count) => count,
        Err(e) => return Ok(reject_extract_error(&e)),
    };

    if count == 0 {
        trace!("no input_file parts to extract");
        if let Err(e) = extract_state_history(ctx, &mut budget) {
            return Ok(reject_extract_error(&e));
        }
        if let Some(rejection) = reject_oversized_state_body(ctx, filter.config.max_body_bytes)? {
            return Ok(rejection);
        }
        return Ok(FilterAction::Continue);
    }

    debug!(count, "extracted input_file parts");
    if let Some(rejection) = rewrite_body(body, parsed, ctx, filter.config.max_body_bytes)? {
        return Ok(rejection);
    }
    if let Err(e) = sync_state_after_rewrite(ctx, parsed, &mut budget) {
        return Ok(reject_extract_error(&e));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, filter.config.max_body_bytes)? {
        return Ok(rejection);
    }

    Ok(FilterAction::Continue)
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
    ctx: &mut HttpFilterContext<'_>,
    max_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let rewritten = serde_json::to_vec(parsed)
        .map_err(|e| -> FilterError { format!("openai_doc_extract: failed to serialize body: {e}").into() })?;
    let len = rewritten.len();
    if len > max_body_bytes {
        return Ok(Some(reject_rewritten_body_too_large(len, max_body_bytes)));
    }
    *body = Some(Bytes::from(rewritten));
    ctx.extra_request_headers
        .push((Cow::Borrowed("content-length"), len.to_string()));
    Ok(None)
}

/// Sync converted content back into [`ResponsesState`] after a body
/// rewrite.
fn sync_state_after_rewrite(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: &serde_json::Value,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    state.request_body = resolved_body.clone();

    let Some(resolved_input) = resolved_body.get("input").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };

    let input_len = state.input.len();

    sync_message_history(&mut state.messages, input_len, Some(resolved_input), budget)?;
    sync_persisted_history(&mut state.persisted_messages, input_len, Some(resolved_input), budget)
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
    max_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(None);
    };
    let len = serialized_outbound_body_len(state).map_err(|e| -> FilterError {
        format!("openai_doc_extract: failed to measure rebuilt request body: {e}").into()
    })?;
    Ok((len > max_body_bytes).then(|| reject_rewritten_body_too_large(len, max_body_bytes)))
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

/// Build a 413 rejection for an oversized raw request body.
fn reject_raw_body_too_large(actual: usize, limit: usize) -> FilterAction {
    warn!(actual, limit, "buffered request body exceeds configured limit");
    let body = serde_json::json!({
        "error": {
            "message": format!("request body exceeds {limit} bytes"),
            "type": "doc_extract_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(413)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}

/// Build a 413 rejection for an oversized rewritten body.
fn reject_rewritten_body_too_large(actual: usize, limit: usize) -> FilterAction {
    warn!(actual, limit, "rewritten request body exceeds configured limit");
    let body = serde_json::json!({
        "error": {
            "message": format!("rewritten request body exceeds {limit} bytes"),
            "type": "doc_extract_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(413)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}
