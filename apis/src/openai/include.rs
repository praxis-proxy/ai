// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Shared `include` query-parameter functionality for the `OpenAI` APIs.
//!
//! The Responses and Conversations APIs gate the same optional item
//! fields behind the same `include` values, and both return the same
//! `ItemResource` shapes:
//!
//! - `GET /v1/responses/{id}`
//! - `GET /v1/responses/{id}/input_items`
//! - `POST /v1/conversations/{id}/items`
//! - `GET /v1/conversations/{id}/items`
//! - `GET /v1/conversations/{id}/items/{item_id}`
//!
//! Parsing and projection therefore live here once rather than being
//! duplicated per endpoint module.

use std::borrow::Cow;

use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;

/// The spellings match OpenAI's `IncludeEnum` exactly. Runtime query parsing
/// and generated `OpenAPI` both consume this enum.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToSchema)]
pub(crate) enum IncludeField {
    /// Include file search result payloads.
    #[serde(rename = "file_search_call.results")]
    #[schema(rename = "file_search_call.results")]
    FileSearchCallResults,
    /// Include web search result payloads.
    #[serde(rename = "web_search_call.results")]
    #[schema(rename = "web_search_call.results")]
    WebSearchCallResults,
    /// Include the sources used by web search actions.
    #[serde(rename = "web_search_call.action.sources")]
    #[schema(rename = "web_search_call.action.sources")]
    WebSearchCallActionSources,
    /// Include image URLs in message input-image parts.
    #[serde(rename = "message.input_image.image_url")]
    #[schema(rename = "message.input_image.image_url")]
    MessageInputImageImageUrl,
    /// Include image URLs in computer-call outputs.
    #[serde(rename = "computer_call_output.output.image_url")]
    #[schema(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    /// Include code-interpreter output payloads.
    #[serde(rename = "code_interpreter_call.outputs")]
    #[schema(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    /// Include encrypted reasoning content.
    #[serde(rename = "reasoning.encrypted_content")]
    #[schema(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
    /// Include token log probabilities in message output-text parts.
    #[serde(rename = "message.output_text.logprobs")]
    #[schema(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
}

impl IncludeField {
    /// Parse one decoded query value using the official enum spelling.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "file_search_call.results" => Some(Self::FileSearchCallResults),
            "web_search_call.results" => Some(Self::WebSearchCallResults),
            "web_search_call.action.sources" => Some(Self::WebSearchCallActionSources),
            "message.input_image.image_url" => Some(Self::MessageInputImageImageUrl),
            "computer_call_output.output.image_url" => Some(Self::ComputerCallOutputImageUrl),
            "code_interpreter_call.outputs" => Some(Self::CodeInterpreterCallOutputs),
            "reasoning.encrypted_content" => Some(Self::ReasoningEncryptedContent),
            "message.output_text.logprobs" => Some(Self::MessageOutputTextLogprobs),
            _ => None,
        }
    }

    /// Return this field's bit in the compact runtime include set.
    const fn bit(self) -> u8 {
        match self {
            Self::FileSearchCallResults => 1 << 0,
            Self::WebSearchCallResults => 1 << 1,
            Self::WebSearchCallActionSources => 1 << 2,
            Self::MessageInputImageImageUrl => 1 << 3,
            Self::ComputerCallOutputImageUrl => 1 << 4,
            Self::CodeInterpreterCallOutputs => 1 << 5,
            Self::ReasoningEncryptedContent => 1 << 6,
            Self::MessageOutputTextLogprobs => 1 << 7,
        }
    }
}

/// Set of requested optional item fields.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct IncludeFields(u8);

impl IncludeFields {
    /// Add one requested field.
    pub(crate) fn insert(&mut self, field: IncludeField) {
        self.0 |= field.bit();
    }

    /// Return whether a field was requested.
    pub(crate) const fn contains(self, field: IncludeField) -> bool {
        self.0 & field.bit() != 0
    }
}

/// Parse both official SDK encodings for the array-valued `include` query:
/// repeated `include=value` pairs and bracketed `include[]=value` pairs.
pub(crate) fn parse_include(query: Option<&str>) -> Result<IncludeFields, String> {
    let Some(query) = query else {
        return Ok(IncludeFields::default());
    };

    let mut includes = IncludeFields::default();
    for pair in query.split('&') {
        let Some((raw_key, raw_value)) = pair.split_once('=') else {
            let key = decode_query_component_strict(pair)?;
            if matches!(key.as_ref(), "include" | "include[]") {
                return Err("'include' query parameter requires a value".to_owned());
            }
            continue;
        };
        let key = decode_query_component_strict(raw_key)?;
        if !matches!(key.as_ref(), "include" | "include[]") {
            continue;
        }
        let value = decode_query_component_strict(raw_value)?;
        let field = IncludeField::parse(&value).ok_or_else(|| format!("unsupported include value: '{value}'"))?;
        includes.insert(field);
    }
    Ok(includes)
}

/// Strictly decode one query component, including form-style `+` spaces.
pub(crate) fn decode_query_component_strict(value: &str) -> Result<Cow<'_, str>, String> {
    if value.contains('+') {
        let normalized = value.replace('+', " ");
        return percent_decode_str(&normalized)
            .decode_utf8()
            .map(|decoded| Cow::Owned(decoded.into_owned()))
            .map_err(|e| format!("query parameter must be valid UTF-8: {e}"));
    }
    percent_decode_str(value)
        .decode_utf8()
        .map_err(|e| format!("query parameter must be valid UTF-8: {e}"))
}

/// Remove optional fields that were not requested through `include`.
pub(crate) fn project_item(item: &mut Value, includes: IncludeFields) {
    let Some(object) = item.as_object_mut() else {
        return;
    };
    match projection_kind(object) {
        ProjectionKind::Reasoning => remove_unless_included(
            object,
            "encrypted_content",
            includes.contains(IncludeField::ReasoningEncryptedContent),
        ),
        ProjectionKind::FileSearch => remove_unless_included(
            object,
            "results",
            includes.contains(IncludeField::FileSearchCallResults),
        ),
        ProjectionKind::WebSearch => project_web_search_fields(object, includes),
        ProjectionKind::CodeInterpreter => remove_unless_included(
            object,
            "outputs",
            includes.contains(IncludeField::CodeInterpreterCallOutputs),
        ),
        ProjectionKind::ComputerOutput => project_computer_output_fields(object, includes),
        ProjectionKind::Message => project_message_fields(object, includes),
        ProjectionKind::Other => {},
    }
}

/// Item variants with fields controlled by `include`.
#[derive(Clone, Copy)]
enum ProjectionKind {
    /// Reasoning item with optional encrypted content.
    Reasoning,
    /// File-search call with optional results.
    FileSearch,
    /// Web-search call with optional results and sources.
    WebSearch,
    /// Code-interpreter call with optional outputs.
    CodeInterpreter,
    /// Computer-call output with an optional image URL.
    ComputerOutput,
    /// Message with optional fields in typed content parts.
    Message,
    /// Item without any fields controlled by `include`.
    Other,
}

/// Classify an item without retaining a borrow into the mutable object.
fn projection_kind(object: &Map<String, Value>) -> ProjectionKind {
    match object.get("type").and_then(Value::as_str) {
        Some("reasoning") => ProjectionKind::Reasoning,
        Some("file_search_call") => ProjectionKind::FileSearch,
        Some("web_search_call") => ProjectionKind::WebSearch,
        Some("code_interpreter_call") => ProjectionKind::CodeInterpreter,
        Some("computer_call_output") => ProjectionKind::ComputerOutput,
        Some("message") => ProjectionKind::Message,
        _ => ProjectionKind::Other,
    }
}

/// Remove one top-level field unless it was explicitly requested.
fn remove_unless_included(object: &mut Map<String, Value>, field: &str, included: bool) {
    if !included {
        object.remove(field);
    }
}

/// Project web-search fields controlled by independent include values.
fn project_web_search_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    remove_unless_included(object, "results", includes.contains(IncludeField::WebSearchCallResults));
    if !includes.contains(IncludeField::WebSearchCallActionSources)
        && let Some(action) = object.get_mut("action").and_then(Value::as_object_mut)
    {
        action.remove("sources");
    }
}

/// Project the nested image URL from a computer-call output.
fn project_computer_output_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    if !includes.contains(IncludeField::ComputerCallOutputImageUrl)
        && let Some(output) = object.get_mut("output").and_then(Value::as_object_mut)
    {
        output.remove("image_url");
    }
}

/// Project optional fields from typed message content parts.
fn project_message_fields(object: &mut Map<String, Value>, includes: IncludeFields) {
    let Some(content) = object.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for part in content {
        let Some(part) = part.as_object_mut() else {
            continue;
        };
        if part.get("type").and_then(Value::as_str) == Some("input_image")
            && !includes.contains(IncludeField::MessageInputImageImageUrl)
        {
            part.remove("image_url");
        } else if part.get("type").and_then(Value::as_str) == Some("output_text")
            && !includes.contains(IncludeField::MessageOutputTextLogprobs)
        {
            part.remove("logprobs");
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // include parsing and projection
    // -------------------------------------------------------------------------

    #[test]
    fn parse_include_fields_supports_python_and_node_sdk_encodings() {
        let includes = parse_include(Some(
            "include=reasoning.encrypted_content&include%5B%5D=message.output_text.logprobs",
        ))
        .unwrap();

        assert!(
            includes.contains(IncludeField::ReasoningEncryptedContent),
            "repeated-key encoding should parse reasoning encrypted content"
        );
        assert!(
            includes.contains(IncludeField::MessageOutputTextLogprobs),
            "bracket encoding should parse output-text log probabilities"
        );
        assert!(
            !includes.contains(IncludeField::FileSearchCallResults),
            "unrequested include values must remain absent"
        );
    }

    #[test]
    fn parse_include_fields_rejects_unknown_or_malformed_values() {
        let unknown = parse_include(Some("include=future.secret_field")).unwrap_err();
        assert!(
            unknown.contains("unsupported include value"),
            "unknown values should produce an unsupported-value diagnostic: {unknown}"
        );

        let missing = parse_include(Some("include")).unwrap_err();
        assert!(
            missing.contains("requires a value"),
            "missing include values should identify the required value: {missing}"
        );

        let invalid_utf8 = parse_include(Some("include=%FF")).unwrap_err();
        assert!(
            invalid_utf8.contains("valid UTF-8"),
            "invalid encoding should identify the UTF-8 requirement: {invalid_utf8}"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "one fixture covers every include projection path")]
    fn projection_removes_every_unrequested_include_gated_field() {
        let mut items = vec![
            serde_json::json!({
                "type": "reasoning",
                "encrypted_content": "secret",
                "summary": []
            }),
            serde_json::json!({
                "type": "file_search_call",
                "results": [{"file_id": "file_1"}],
                "status": "completed"
            }),
            serde_json::json!({
                "type": "web_search_call",
                "results": [{"url": "https://example.com"}],
                "action": {
                    "type": "search",
                    "sources": [{"type": "url", "url": "https://example.com"}]
                }
            }),
            serde_json::json!({
                "type": "code_interpreter_call",
                "outputs": [{"type": "logs", "logs": "done"}],
                "status": "completed"
            }),
            serde_json::json!({
                "type": "computer_call_output",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AA=="}
            }),
            serde_json::json!({
                "type": "message",
                "content": [
                    {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"},
                    {"type": "output_text", "text": "answer", "annotations": [], "logprobs": []},
                    {"type": "input_text", "text": "keep me"}
                ]
            }),
        ];

        for item in &mut items {
            project_item(item, IncludeFields::default());
        }

        assert!(
            items[0].get("encrypted_content").is_none(),
            "reasoning encrypted content should be omitted"
        );
        assert!(
            items[1].get("results").is_none(),
            "file-search results should be omitted"
        );
        assert!(
            items[2].get("results").is_none(),
            "web-search results should be omitted"
        );
        assert!(
            items[2]["action"].get("sources").is_none(),
            "web-search action sources should be omitted"
        );
        assert!(
            items[3].get("outputs").is_none(),
            "code-interpreter outputs should be omitted"
        );
        assert!(
            items[4]["output"].get("image_url").is_none(),
            "computer-output image URLs should be omitted"
        );
        assert!(
            items[5]["content"][0].get("image_url").is_none(),
            "message input-image URLs should be omitted"
        );
        assert!(
            items[5]["content"][1].get("logprobs").is_none(),
            "message output-text log probabilities should be omitted"
        );
        assert_eq!(items[5]["content"][2]["text"], "keep me");
    }

    #[test]
    fn projection_preserves_every_requested_include_gated_field() {
        let mut includes = IncludeFields::default();
        for field in [
            IncludeField::FileSearchCallResults,
            IncludeField::WebSearchCallResults,
            IncludeField::WebSearchCallActionSources,
            IncludeField::MessageInputImageImageUrl,
            IncludeField::ComputerCallOutputImageUrl,
            IncludeField::CodeInterpreterCallOutputs,
            IncludeField::ReasoningEncryptedContent,
            IncludeField::MessageOutputTextLogprobs,
        ] {
            includes.insert(field);
        }
        let original = serde_json::json!({
            "type": "message",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/image.png"},
                {"type": "output_text", "logprobs": [{"token": "x"}]}
            ]
        });
        let mut projected = original.clone();

        project_item(&mut projected, includes);

        assert_eq!(projected, original);
    }
}
