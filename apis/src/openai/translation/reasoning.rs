// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Reasoning translation between Responses and Chat Completions.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::chat_completions::{TranslationError, json_type_name};

/// Maximum size limit for raw reasoning.
const DEFAULT_MAX_REASONING_BYTES: usize = 65_536;

/// Default opening marker wrapping replayed prior-turn reasoning inlined into an
/// assistant message's content.
const DEFAULT_THINK_OPEN: &str = "<think>";

/// Default closing marker for inlined replayed reasoning.
const DEFAULT_THINK_CLOSE: &str = "</think>";

/// The reasoning item content part type carrying raw chain-of-thought.
const REASONING_TEXT_PART_TYPE: &str = "reasoning_text";

/// Backend specific named reasoning dialect.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ReasoningDialect {
    /// Portable Chat Completions fields only, reasoning.summary unsupported.
    #[default]
    None,
    /// Supports `message.reasoning` and falls back to the deprecated `message.reasoning_content`.
    Vllm,
}

impl ReasoningDialect {
    /// Whether the dialect exposes safe summaries.
    const fn supports_safe_summary(self) -> bool {
        // Generating a summary through an additional inference call is out of scope.
        match self {
            Self::None | Self::Vllm => false,
        }
    }

    /// Whether a backend reasoning contract is active.
    pub(crate) const fn is_enabled(self) -> bool {
        match self {
            Self::None => false,
            Self::Vllm => true,
        }
    }
}

/// Resolved backend-specific reasoning configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct ReasoningOptions {
    /// Selected backend reasoning contract.
    pub(crate) dialect: ReasoningDialect,
    /// Maximum raw reasoning size preserved per response.
    pub(crate) max_reasoning_bytes: usize,
    /// Opening marker for inlined replayed reasoning. Defaults to `<think>`.
    pub(crate) think_open: String,
    /// Closing marker for inlined replayed reasoning. Defaults to `</think>`.
    pub(crate) think_close: String,
}

impl Default for ReasoningOptions {
    fn default() -> Self {
        Self {
            dialect: ReasoningDialect::None,
            max_reasoning_bytes: DEFAULT_MAX_REASONING_BYTES,
            think_open: DEFAULT_THINK_OPEN.to_owned(),
            think_close: DEFAULT_THINK_CLOSE.to_owned(),
        }
    }
}

/// Prior-turn reasoning buffered for inline replay onto the assistant turn it
/// precedes, paired with the dialect options that govern extraction and framing.
pub(crate) struct ReplayBuffer<'a> {
    /// Raw chain-of-thought awaiting an assistant turn, if any.
    pending: Option<String>,
    /// Dialect options governing extraction, the byte ceiling, and think markers.
    options: &'a ReasoningOptions,
}

impl<'a> ReplayBuffer<'a> {
    /// Create an empty buffer bound to the active reasoning options.
    pub(crate) fn new(options: &'a ReasoningOptions) -> Self {
        Self { pending: None, options }
    }

    /// Preserve reasoning-only output as an assistant turn at input boundaries.
    pub(crate) fn flush_standalone(&mut self, messages: &mut Vec<Value>) -> Result<(), TranslationError> {
        if self.pending.is_some() {
            let content = self.take_inline(Value::Null)?;
            messages.push(json!({"role": "assistant", "content": content}));
        }
        Ok(())
    }

    /// Buffer a rehydrated reasoning item's raw text for replay onto the assistant
    /// turn it precedes, concatenating consecutive reasoning items.
    pub(crate) fn buffer(&mut self, obj: &Map<String, Value>) -> Result<(), TranslationError> {
        let text = reasoning_input_text(obj, self.options)?;
        match &mut self.pending {
            Some(existing) => {
                // Each item is bounded individually, but consecutive items
                // concatenate (with a newline separator), so the running total
                // must still respect the ceiling.
                let total = existing.len().saturating_add(1).saturating_add(text.len());
                enforce_reasoning_limit(total, self.options.max_reasoning_bytes)?;
                existing.push('\n');
                existing.push_str(&text);
            },
            None => self.pending = Some(text),
        }
        Ok(())
    }

    /// Fold buffered prior-turn reasoning into assistant content using the
    /// dialect's think markers, returning the content unchanged when nothing is
    /// buffered.
    pub(crate) fn take_inline(&mut self, content: Value) -> Result<Value, TranslationError> {
        let Some(text) = self.pending.take() else {
            return Ok(content);
        };
        let block = format!("{}{}{}", self.options.think_open, text, self.options.think_close);
        match content {
            Value::Null => Ok(Value::String(block)),
            Value::String(existing) => Ok(Value::String(format!("{block}{existing}"))),
            Value::Array(mut parts) => {
                parts.insert(0, json!({"type": "text", "text": block}));
                Ok(Value::Array(parts))
            },
            _ => Err(TranslationError::InvalidMessageContent),
        }
    }
}

/// Resolve the requested reasoning summary from a `reasoning` block,
/// treating the deprecated `generate_summary` as an alias of `summary`.
/// Conflicting non-null string values are rejected.
pub(crate) fn requested_summary(reasoning: &Map<String, Value>) -> Result<Option<&str>, TranslationError> {
    let summary = summary_control(reasoning, "summary")?;
    let generate = summary_control(reasoning, "generate_summary")?;
    match (summary, generate) {
        (Some(summary), Some(generate)) if summary != generate => Err(TranslationError::ConflictingReasoningSummary),
        (Some(summary), _) | (_, Some(summary)) => Ok(Some(summary)),
        (None, None) => Ok(None),
    }
}

/// Read a summary control field, failing closed on non-string, non-null values.
fn summary_control<'a>(
    reasoning: &'a Map<String, Value>,
    field: &'static str,
) -> Result<Option<&'a str>, TranslationError> {
    match reasoning.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(summary)) => Ok(Some(summary)),
        Some(other) => Err(TranslationError::MalformedReasoningSummary {
            field,
            actual: json_type_name(other).to_owned(),
        }),
    }
}

/// Validate that a requested reasoning summary is compatible with the dialect.
/// A summary request against a dialect without a safe-summary contract is rejected.
pub(crate) fn validate_requested_reasoning(
    request: &Map<String, Value>,
    options: &ReasoningOptions,
) -> Result<(), TranslationError> {
    let reasoning = match request.get("reasoning") {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Object(reasoning)) => reasoning,
        Some(other) => {
            return Err(TranslationError::MalformedReasoningBlock(
                json_type_name(other).to_owned(),
            ));
        },
    };

    if requested_summary(reasoning)?.is_some() && !options.dialect.supports_safe_summary() {
        return Err(TranslationError::UnsupportedReasoningSummary);
    }

    Ok(())
}

/// Build a stable reasoning output item id.
pub(crate) fn reasoning_item_id(response_id: &str) -> String {
    format!("rs_{response_id}")
}

/// Build a `Responses` reasoning output item from a Chat Completions message.
pub(crate) fn extract_reasoning_item(
    message: Option<&Value>,
    reasoning_item_id: String,
    status: &str,
    options: &ReasoningOptions,
) -> Result<Option<Value>, TranslationError> {
    let Some(message) = message.and_then(Value::as_object) else {
        return Ok(None);
    };
    let Some(text) = resolve_raw_reasoning(message, options.dialect)? else {
        return Ok(None);
    };
    enforce_reasoning_limit(text.len(), options.max_reasoning_bytes)?;
    Ok(Some(reasoning_item(reasoning_item_id, status, text)))
}

/// Report whether a Chat Completions message carries extractable raw reasoning
/// for the configured dialect.
pub(crate) fn message_has_reasoning(
    message: Option<&Value>,
    options: &ReasoningOptions,
) -> Result<bool, TranslationError> {
    let Some(message) = message.and_then(Value::as_object) else {
        return Ok(false);
    };
    Ok(resolve_raw_reasoning(message, options.dialect)?.is_some())
}

/// Extract replayable raw reasoning text from a `Responses` reasoning item.
fn reasoning_input_text(item: &Map<String, Value>, options: &ReasoningOptions) -> Result<String, TranslationError> {
    if !options.dialect.is_enabled() {
        return Err(TranslationError::UnsupportedReasoningInput(
            "a reasoning dialect must be configured",
        ));
    }
    if item.get("encrypted_content").is_some_and(|value| !value.is_null()) {
        return Err(TranslationError::UnsupportedReasoningInput(
            "encrypted_content cannot be replayed",
        ));
    }
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or(TranslationError::UnsupportedReasoningInput(
            "content must contain raw reasoning_text parts",
        ))?;
    let text = concat_reasoning_text_parts(parts, options.max_reasoning_bytes)?;
    if text.is_empty() {
        return Err(TranslationError::UnsupportedReasoningInput(
            "content must contain non-empty raw reasoning text",
        ));
    }
    Ok(text)
}

/// Validate and concatenate raw reasoning parts within the configured byte limit.
fn concat_reasoning_text_parts(parts: &[Value], max_bytes: usize) -> Result<String, TranslationError> {
    let mut text = String::new();
    for part in parts {
        if part.get("type").and_then(Value::as_str) != Some(REASONING_TEXT_PART_TYPE) {
            return Err(TranslationError::UnsupportedReasoningInput(
                "content must contain only reasoning_text parts",
            ));
        }
        let chunk = part.get("text").and_then(Value::as_str).ok_or_else(|| {
            TranslationError::MalformedReasoningInput(
                json_type_name(part.get("text").unwrap_or(&Value::Null)).to_owned(),
            )
        })?;
        enforce_reasoning_limit(text.len().saturating_add(chunk.len()), max_bytes)?;
        text.push_str(chunk);
    }
    Ok(text)
}

/// Fail closed when a raw reasoning byte count exceeds the configured ceiling.
fn enforce_reasoning_limit(bytes: usize, max_reasoning_bytes: usize) -> Result<(), TranslationError> {
    if bytes > max_reasoning_bytes {
        return Err(TranslationError::ReasoningTooLarge {
            bytes,
            max_bytes: max_reasoning_bytes,
        });
    }
    Ok(())
}

/// Resolve raw reasoning text from a Chat Completions message for a dialect.
///
/// Each dialect owns where and how its backend encodes raw reasoning, so the
/// per-dialect resolvers keep that contract in one place. Dialects without a
/// raw-reasoning contract yield `None` and no extraction occurs. The match is
/// exhaustive: a new dialect must declare its extraction here to compile.
fn resolve_raw_reasoning(
    message: &Map<String, Value>,
    dialect: ReasoningDialect,
) -> Result<Option<&str>, TranslationError> {
    match dialect {
        ReasoningDialect::None => Ok(None),
        ReasoningDialect::Vllm => resolve_vllm_reasoning(message),
    }
}

/// Resolve raw reasoning for vLLM. Prefers the current `reasoning` field
/// and falls back to the deprecated `reasoning_content` alias.
fn resolve_vllm_reasoning(message: &Map<String, Value>) -> Result<Option<&str>, TranslationError> {
    for field in ["reasoning", "reasoning_content"] {
        match message.get(field) {
            // A null, empty, or absent field is not a payload; try the next name.
            None | Some(Value::Null) => {},
            Some(Value::String(text)) if text.is_empty() => {},
            Some(Value::String(text)) => return Ok(Some(text)),
            Some(other) => return Err(TranslationError::MalformedReasoning(json_type_name(other).to_owned())),
        }
    }
    Ok(None)
}

/// Build a schema-complete `Responses` reasoning item carrying raw reasoning.
fn reasoning_item(id: String, status: &str, text: &str) -> Value {
    json!({
        "id": Value::String(id),
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{
            "type": REASONING_TEXT_PART_TYPE,
            "text": text
        }]
    })
}
