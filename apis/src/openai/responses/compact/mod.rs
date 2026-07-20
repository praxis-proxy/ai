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
    praxis_core::callout::{CalloutClient, CalloutRequest, CalloutResult},
    praxis_filter::{
        BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
        body::MAX_JSON_BODY_BYTES, parse_filter_config,
    },
    serde_json::Value,
    tracing::{debug, warn},
    self::config::{CompactFilterConfig, ValidatedConfig, build_config},
    super::{error::responses_error_rejection, state::ResponsesState},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

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
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: CompactFilterConfig = parse_filter_config("openai_responses_compact", config)?;
        let validated = build_config(&cfg)?;
        let callout_config = validated.callout.build_callout_config();
        let callout_client =
            CalloutClient::new(callout_config).map_err(|e| FilterError::from(e.to_string()))?;
        Ok(Box::new(Self {
            callout_client,
            config: validated,
        }))
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

        let Some(state) = ctx.extensions.get::<ResponsesState>() else {
            debug!("no ResponsesState in extensions, passthrough");
            return Ok(FilterAction::Release);
        };


        let Some(compaction_config) = extract_compaction_config(&state.context_management) else {
            debug!("no compaction config in context management");
            return Ok(FilterAction::Release);
        };

        let Some(token_count) = get_token_count(&state.messages, &self.config.tiktoken_encoding) else {
            return Ok(FilterAction::Release);
        };

        if token_count <= compaction_config.compact_threshold {
            debug!(token_count, threshold = compaction_config.compact_threshold, "under threshold, skipping compaction");
            return Ok(FilterAction::Release);
        }

        debug!(token_count, threshold = compaction_config.compact_threshold, "threshold exceeded, compaction needed");

        // TODO: implement remaining compaction flow:
        //
        // 5. Build summarization request via build_summarization_request()
        //    - Use compaction_config.compaction_model or self.config.default_model
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
        warn!("compaction not yet implemented, passing through");
        Ok(FilterAction::Release)
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
fn extract_compaction_config(context_management: &Option<Value>) -> Option<CompactionParams> {
   let array = context_management.as_ref()?.as_array()?;

    for entry in array {
        let Some(entry_type) = entry.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        if entry_type != "compaction" {
            continue;
        }
        let compact_threshold = entry
            .get("compact_threshold")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let compaction_model = entry
            .get("compaction_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        return Some(CompactionParams {
            compact_threshold,
            compaction_model,
        });
    }
    None
}

/// Estimate the token count for the given messages using tiktoken.
///
/// Uses the configured encoding (e.g. `cl100k_base`, `o200k_base`)
/// to tokenize the serialized conversation text.
///
/// Returns `None` if the encoding name is not recognized.
fn get_token_count(messages: &[Value], tiktoken_encoding: &str) -> Option<u64> {
    let bpe = match tiktoken_encoding {
        "cl100k_base" => tiktoken_rs::cl100k_base_singleton(),
        "o200k_base" => tiktoken_rs::o200k_base_singleton(),
        other => {
            warn!(encoding = other, "unknown tiktoken encoding, cannot estimate tokens");
            return None;
        }
    };
    let text = build_conversation_text(messages);
    let count = bpe.encode_ordinary(&text).len() as u64;
    debug!(count, source = "tiktoken", encoding = tiktoken_encoding, "token count estimated");
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
fn build_conversation_text(messages: &[Value]) -> String {
    let mut parts = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("unknown");
        let content = extract_content(msg);
        if content.is_empty() {
            continue;
        }
        parts.push(format!("{role}: {content}"));
    }
    parts.join("\n\n")
}

/// Extract text content from a message's `content` field.
///
/// Content can be a plain string, an array of content parts
/// (each with a `"text"` field), or absent/null.
fn extract_content(msg: &Value) -> String {
    let Some(content) = msg.get("content") else {
        return String::new();
    };
    if let Some(s) = content.as_str() {
        return s.to_owned();
    }
    if let Some(arr) = content.as_array() {
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        return texts.join(" ");
    }
    String::new()
}
