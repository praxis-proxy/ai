// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Compact filter: token counting and context window management.
//!
//! When a request's `context_management` contains a compaction
//! configuration and the token count exceeds the specified threshold,
//! this filter summarizes the conversation history via a sub-request
//! to an inference backend, replacing it with a single compaction
//! item. Runs after `rehydrate` (which populates messages and
//! previous usage) and before `openai_tool_parse`.

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

#[expect(
    unused_imports,
    reason = "scaffolding — all imports needed once todo!()s are implemented"
)]
use {
    async_trait::async_trait,
    bytes::Bytes,
    praxis_core::callout::{
        CalloutClient, CalloutConfig, CalloutRequest, CalloutResult, CircuitBreakerConfig,
        FailureMode as CoreFailureMode,
    },
    praxis_filter::{
        BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
        body::MAX_JSON_BODY_BYTES, parse_filter_config,
    },
    serde_json::Value,
    tracing::{debug, warn},
    self::config::{CompactFilterConfig, FailureMode, ValidatedConfig, build_config},
    super::{error::responses_error_rejection, state::ResponsesState},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Metadata key for previous response total token count.
/// Written by the rehydrate filter.
const PREV_USAGE_TOTAL_KEY: &str = "responses.previous_usage_total_tokens";

#[expect(dead_code, reason = "scaffolding — used once build_summarization_request is implemented")]
/// System prompt for the summarization call.
const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
Summarize the following conversation concisely. \
Preserve all key facts, decisions, code snippets, \
user preferences, and important context. The summary \
will replace the full conversation history, so it must \
capture everything needed to continue coherently.";

// -----------------------------------------------------------------------------
// CompactionParams
// -----------------------------------------------------------------------------

#[expect(dead_code, reason = "scaffolding — used once extract_compaction_config is implemented")]
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
/// timeout_ms: 30000
/// failure_mode: closed
/// status_on_error: 502
/// ```
#[expect(
    dead_code,
    reason = "scaffolding — fields used once from_config and on_request_body are implemented"
)]
pub struct CompactFilter {
    /// HTTP client for the summarization inference call.
    callout_client: CalloutClient,
    /// Validated filter configuration.
    config: ValidatedConfig,
}

impl CompactFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// Constructs the [`CalloutClient`] at startup so it can be
    /// reused across requests.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if config validation fails or the
    /// callout client cannot be constructed.
    #[expect(
        clippy::todo,
        unused_variables,
        reason = "scaffolding — implement CalloutClient construction"
    )]
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: CompactFilterConfig = parse_filter_config("openai_responses_compact", config)?;
        let validated = build_config(&cfg)?;

        // TODO: build CalloutConfig from validated config
        //       (follow the pattern in web_search/provider.rs:build_callout_config)
        //
        // Map FailureMode -> CoreFailureMode:
        //   FailureMode::Closed -> CoreFailureMode::Closed
        //   FailureMode::Open   -> CoreFailureMode::Open
        //
        // Include CircuitBreakerConfig with:
        //   consecutive_failures: 5
        //   recovery_window_ms: 30_000
        //
        // Then construct Self { callout_client, config: validated }
        todo!("construct CalloutClient and return CompactFilter")
    }
}

#[async_trait]
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

    async fn on_request(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
    ) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::todo,
        unused_variables,
        reason = "scaffolding — implement the compaction flow"
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

        // Skip non-Responses requests
        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            return Ok(FilterAction::Release);
        }

        let streaming = ctx
            .get_metadata("openai_responses_format.stream")
            .is_some_and(|v| v == "true");

        // TODO: implement the compaction flow:
        //
        // 1. Get ResponsesState from ctx.extensions
        //    - If absent, return Release (no state to compact)
        //
        // 2. Call extract_compaction_config() on state.context_management
        //    - If None, return Release (no compaction requested)
        //
        // 3. Call get_token_count() to get the current token count
        //    - Try previous_usage metadata first, fall back to tiktoken
        //    - If None (no count available), return Release
        //
        // 4. Compare token_count against params.compact_threshold
        //    - If token_count <= threshold, return Release
        //
        // 5. Build summarization request via build_summarization_request()
        //    - Use params.compaction_model or self.config.default_model
        //    - Include state.request_body["instructions"] if present
        //
        // 6. Execute via self.callout_client.execute(request).await
        //    - On Success: parse response, build compaction item,
        //      replace messages
        //    - On Failed (open mode): log warning, return Release
        //    - On Rejected: return Reject with responses_error_rejection()
        //
        // 7. Set metadata: "responses.compacted" = "true"
        //
        // 8. Return Release
        todo!("implement compaction flow")
    }
}

// -----------------------------------------------------------------------------
// Compaction Logic
// -----------------------------------------------------------------------------

/// Parse the `context_management` JSON to find a compaction config.
///
/// The `context_management` field is an array like:
/// `[{"type": "compaction", "compact_threshold": 50000}]`
///
/// Returns `None` if no compaction entry is found.
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement context_management parsing"
)]
fn extract_compaction_config(context_management: &Option<Value>) -> Option<CompactionParams> {
    // TODO: iterate the array, find the first entry where
    //       type == "compaction", extract compact_threshold (u64)
    //       and optional compaction_model (String)
    todo!("parse context_management array")
}

/// Get the current token count.
///
/// Tries `previous_usage` metadata first (exact, written by
/// rehydrate). Falls back to tiktoken estimation on
/// `state.messages` using the configured encoding.
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement tiktoken fallback"
)]
fn get_token_count(
    ctx: &HttpFilterContext<'_>,
    messages: &[Value],
    tiktoken_encoding: &str,
) -> Option<u64> {
    // Try metadata from rehydrate first (exact count from the
    // previous response's usage stats)
    if let Some(total) = ctx.get_metadata(PREV_USAGE_TOTAL_KEY) {
        if let Ok(count) = total.parse::<u64>() {
            debug!(count, source = "previous_usage", "token count from metadata");
            return Some(count);
        }
    }

    // TODO: fall back to tiktoken estimation
    //
    // 1. Call build_conversation_text(messages) to serialize
    // 2. Use tiktoken_rs to get a BPE encoder for tiktoken_encoding
    //    (e.g., tiktoken_rs::cl100k_base() for "cl100k_base")
    // 3. Encode the text and return the token count as u64
    // 4. Log with source = "tiktoken_estimate"
    // 5. Return None if encoding fails
    todo!("tiktoken fallback estimation")
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
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement request construction"
)]
fn build_summarization_request(
    messages: &[Value],
    instructions: Option<&str>,
    model: &str,
    inference_url: &str,
) -> CalloutRequest {
    // TODO:
    // 1. Build the system prompt:
    //    - If instructions is Some, prepend them before
    //      SUMMARIZATION_SYSTEM_PROMPT
    //    - Otherwise just use SUMMARIZATION_SYSTEM_PROMPT
    //
    // 2. Build user content via build_conversation_text(messages)
    //
    // 3. Construct the Chat Completions JSON body
    //
    // 4. Return a CalloutRequest with:
    //    - method: POST
    //    - url: inference_url.to_owned()
    //    - headers: Content-Type + Accept application/json
    //    - body: Some(serialized JSON bytes)
    //    - depth: 0
    todo!("build summarization CalloutRequest")
}

/// Parse the Chat Completions response and extract the summary text.
///
/// Expected shape: `{"choices": [{"message": {"content": "..."}}]}`
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement response parsing"
)]
fn parse_summarization_response(body: &[u8]) -> Result<String, String> {
    // TODO: parse JSON, navigate to choices[0].message.content,
    //       return the string. Return Err with a descriptive
    //       message if parsing fails or the path doesn't exist.
    todo!("parse Chat Completions response")
}

/// Build the compaction output item.
///
/// Returns: `{"type": "compaction", "encrypted_content": "<summary>"}`
///
/// Note: `encrypted_content` is a misnomer from the OpenAI spec —
/// in the proxy context this is plain text.
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement compaction item construction"
)]
fn build_compaction_item(summary: &str) -> Value {
    // TODO: construct the JSON value
    todo!("build compaction item JSON")
}

/// Replace conversation history with the compaction item.
///
/// After replacement:
/// - `state.messages` = `[compaction_item, ...state.input]`
/// - `state.persisted_messages` = `[compaction_item, ...state.input]`
///
/// `state.input` holds the current request's input items (unchanged
/// by rehydrate), so the current turn's messages are preserved.
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement message replacement"
)]
fn replace_messages(state: &mut ResponsesState, compaction_item: Value) {
    // TODO: replace state.messages and state.persisted_messages
    //       with [compaction_item] followed by state.input items
    todo!("replace messages with compacted history")
}

/// Format a message array as readable text for the summarization prompt.
///
/// Each message becomes: `<role>: <content>`
/// Messages are separated by blank lines.
#[expect(
    clippy::todo,
    dead_code,
    unused_variables,
    reason = "scaffolding — implement message serialization"
)]
fn build_conversation_text(messages: &[Value]) -> String {
    // TODO: iterate messages, extract role and content from each,
    //       format as "role: content" separated by "\n\n"
    //
    // Content can be:
    //   - A string: use directly
    //   - An array of content parts: extract "text" from each
    //     input_text/output_text part and join them
    //   - Missing/null: skip the message
    todo!("serialize messages to readable text")
}
