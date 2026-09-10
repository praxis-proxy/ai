// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Reasoning translation between Responses and Chat Completions.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::chat_completions::{TranslationError, json_type_name};

/// Maximum size limit for raw reasoning.
pub(crate) const DEFAULT_MAX_REASONING_BYTES: usize = 65_536;

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
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(crate) struct ReasoningOptions {
    /// Selected backend reasoning contract.
    pub(crate) dialect: ReasoningDialect,
    /// Maximum raw reasoning size preserved per response.
    pub(crate) max_reasoning_bytes: usize,
}

impl Default for ReasoningOptions {
    fn default() -> Self {
        Self {
            dialect: ReasoningDialect::None,
            max_reasoning_bytes: DEFAULT_MAX_REASONING_BYTES,
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
    let Some(reasoning) = request.get("reasoning").and_then(Value::as_object) else {
        return Ok(());
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
    if text.len() > options.max_reasoning_bytes {
        return Err(TranslationError::ReasoningTooLarge {
            bytes: text.len(),
            max_bytes: options.max_reasoning_bytes,
        });
    }
    Ok(Some(reasoning_item(reasoning_item_id, status, text)))
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
            "type": "reasoning_text",
            "text": text
        }]
    })
}
