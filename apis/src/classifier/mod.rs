// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Pure request body classifier for AI API format detection.
//!
//! Disambiguates Responses API, Anthropic Messages, and Chat
//! Completions from a single JSON body parse.

// -----------------------------------------------------------------------------
// AiRequestFormat
// -----------------------------------------------------------------------------

/// Classified request body format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AiRequestFormat {
    /// `OpenAI` Responses API (has `input` field).
    Responses,
    /// Anthropic Messages API (`messages` + required `max_tokens`).
    AnthropicMessages,
    /// Chat Completions API (has `messages` without required `max_tokens`).
    ChatCompletions,
    /// Valid JSON but neither recognized format.
    UnknownJson,
    /// Body is not valid JSON.
    InvalidJson,
    /// Body is empty or absent.
    NonJson,
}

impl AiRequestFormat {
    /// Stable string representation for headers, metadata, and filter results.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
            Self::ChatCompletions => "openai_chat_completions",
            Self::UnknownJson => "unknown",
            Self::InvalidJson => "invalid_json",
            Self::NonJson => "non_json",
        }
    }
}

// -----------------------------------------------------------------------------
// ClassifiedRequest
// -----------------------------------------------------------------------------

/// Extracted facts from a classified request body.
#[derive(Debug)]
#[expect(clippy::struct_excessive_bools, reason = "independent presence flags from JSON body")]
pub(crate) struct ClassifiedRequest {
    /// Extracted `background` field value, if present.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub background: Option<bool>,
    /// Detected body format.
    pub format: AiRequestFormat,
    /// Whether `conversation` is present and non-null.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub has_conversation: bool,
    /// Whether `previous_response_id` is present and non-null.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub has_previous_response_id: bool,
    /// Whether `prompt.id` is present and non-null.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub has_prompt_id: bool,
    /// Whether `tools` is a non-empty array (coarse presence check).
    ///
    /// This does NOT validate individual entries: an array like
    /// `[{"unexpected": true}]` will set this to `true` even though
    /// no entry carries a recognised `type` discriminator.
    /// [`openai_tool_parse`] applies stricter per-entry classification and
    /// may disagree on malformed arrays.
    ///
    /// [`openai_tool_parse`]: crate::openai::responses::tool_parse
    pub has_tools: bool,
    /// Extracted `max_output_tokens` field value (Responses API), if present.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub max_output_tokens: Option<u64>,
    /// Extracted `max_tokens` field value, if present.
    pub max_tokens: Option<u64>,
    /// Extracted `model` field value, if present.
    pub model: Option<String>,
    /// Extracted `store` field value, if present.
    #[cfg_attr(
        all(not(feature = "openai-responses"), not(test)),
        expect(dead_code, reason = "read only by the Responses request processor")
    )]
    pub store: Option<bool>,
    /// Extracted `stream` field value, if present.
    pub stream: Option<bool>,
}

// -----------------------------------------------------------------------------
// Path Classification
// -----------------------------------------------------------------------------

/// Check whether a method + path pair is the Responses API create endpoint.
///
/// Returns `true` only for `POST /v1/responses` (with optional trailing slash).
/// Sub-resource POSTs like `/v1/responses/{id}/cancel` return `false`.
pub(crate) fn is_responses_create(method: &http::Method, path: &str) -> bool {
    method == http::Method::POST && normalize_trailing_slash(path) == "/v1/responses"
}

/// Check whether a method + path pair is the Chat Completions create endpoint.
///
/// Returns `true` only for `POST /v1/chat/completions`. Kept separate from
/// [`is_responses_create`] so Responses API filters keep their current endpoint
/// semantics; only filters that are genuinely endpoint-agnostic, such as
/// `openai_responses_model_rewrite`, accept both.
pub(crate) fn is_chat_completions_create(method: &http::Method, path: &str) -> bool {
    method == http::Method::POST && path == "/v1/chat/completions"
}

// -----------------------------------------------------------------------------
// Body Classification
// -----------------------------------------------------------------------------

/// Classify a request body and extract routing facts.
///
/// This function is pure: no I/O, no side effects, no mutation of
/// the input bytes.
pub(crate) fn classify_request_body(body: &[u8]) -> ClassifiedRequest {
    if body.is_empty() {
        return empty_result(AiRequestFormat::NonJson);
    }

    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return empty_result(AiRequestFormat::InvalidJson);
    };

    let Some(obj) = value.as_object_mut() else {
        return empty_result(AiRequestFormat::InvalidJson);
    };

    // This parse is a throwaway owned by this function, so the model string is
    // moved out rather than copied.
    let model = take_string(obj, "model");
    classify_fields(obj, model)
}

/// Extract routing facts from an already-parsed request object.
///
/// Callers that parsed the body for their own reasons use this instead of
/// [`classify_request_body`], so one request is deserialized once rather than
/// once per consumer.
///
/// Borrows the object rather than taking ownership of fields out of it: a
/// caller that reuses the same parsed value afterwards — to build request
/// state, say — must still see the body the client actually sent. That costs
/// one copy of the model string, which the owned path above avoids.
pub(crate) fn classify_object(obj: &serde_json::Map<String, serde_json::Value>) -> ClassifiedRequest {
    let model = copy_string(obj, "model");
    classify_fields(obj, model)
}

/// Extract the remaining facts once the model has been obtained.
///
/// Split out so the owned and borrowed entry points differ only in how they
/// get the model string, and cannot otherwise drift.
fn classify_fields(obj: &serde_json::Map<String, serde_json::Value>, model: Option<String>) -> ClassifiedRequest {
    let format = classify_format(obj);

    ClassifiedRequest {
        background: obj.get("background").and_then(serde_json::Value::as_bool),
        format,
        has_conversation: obj.get("conversation").is_some_and(|v| !v.is_null()),
        has_previous_response_id: obj.get("previous_response_id").is_some_and(|v| !v.is_null()),
        has_prompt_id: obj
            .get("prompt")
            .and_then(serde_json::Value::as_object)
            .and_then(|prompt| prompt.get("id"))
            .is_some_and(|v| !v.is_null()),
        has_tools: obj
            .get("tools")
            .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty())),
        max_output_tokens: obj.get("max_output_tokens").and_then(serde_json::Value::as_u64),
        max_tokens: obj.get("max_tokens").and_then(serde_json::Value::as_u64),
        model,
        store: obj.get("store").and_then(serde_json::Value::as_bool),
        stream: obj.get("stream").and_then(serde_json::Value::as_bool),
    }
}

/// Determine format from top-level keys.
///
/// Precedence: `input`, `prompt` object, `previous_response_id`,
/// or `conversation` → Responses, then `messages` with Anthropic
/// signals → Anthropic Messages, then `messages` alone → Chat
/// Completions.
///
/// Anthropic signals: `max_tokens` is required AND at least one of
/// top-level `system` field or typed content blocks (arrays of
/// objects with a `type` key in `messages`). This prevents false
/// positives when `OpenAI` Chat Completions requests include the
/// optional `max_tokens` field.
fn classify_format(obj: &serde_json::Map<String, serde_json::Value>) -> AiRequestFormat {
    if obj.contains_key("input")
        || obj.get("prompt").is_some_and(serde_json::Value::is_object)
        || obj.contains_key("previous_response_id")
        || obj.contains_key("conversation")
    {
        return AiRequestFormat::Responses;
    }

    if obj.contains_key("messages") {
        if obj.contains_key("max_tokens") && has_anthropic_signals(obj) {
            return AiRequestFormat::AnthropicMessages;
        }
        return AiRequestFormat::ChatCompletions;
    }

    AiRequestFormat::UnknownJson
}

/// Check for Anthropic-specific structural signals beyond `max_tokens`.
///
/// Returns true if any of:
/// - Top-level `system` field is present as a string or array (Anthropic separates system from messages; `OpenAI` puts
///   it in the messages array)
/// - Any message in `messages` has typed content blocks (array of objects with a `type` key, e.g. `[{"type": "text",
///   ...}]`)
fn has_anthropic_signals(obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    if matches!(
        obj.get("system"),
        Some(serde_json::Value::String(_) | serde_json::Value::Array(_))
    ) {
        return true;
    }

    if let Some(serde_json::Value::Array(messages)) = obj.get("messages") {
        for msg in messages {
            if let Some(serde_json::Value::Array(blocks)) = msg.get("content")
                && blocks.iter().any(|b| b.get("type").is_some())
            {
                return true;
            }
        }
    }

    false
}

// -----------------------------------------------------------------------------
// Private Utilities
// -----------------------------------------------------------------------------

/// Strip a single trailing slash unless the path is the root `/`.
fn normalize_trailing_slash(path: &str) -> &str {
    path.strip_suffix('/').filter(|p| !p.is_empty()).unwrap_or(path)
}

/// Build a result with no extracted facts.
pub(crate) fn empty_result(format: AiRequestFormat) -> ClassifiedRequest {
    ClassifiedRequest {
        background: None,
        format,
        has_conversation: false,
        has_previous_response_id: false,
        has_prompt_id: false,
        has_tools: false,
        max_output_tokens: None,
        max_tokens: None,
        model: None,
        store: None,
        stream: None,
    }
}

/// Take a string field out of a JSON object, converting numbers/booleans
/// to their string representation.
///
/// Moves the owned `String` out of the parsed body instead of copying it,
/// leaving an empty string in its place. Only for a parse the caller owns and
/// discards; `copy_string` is the variant for a shared parse, and only
/// exists when the Responses filters that share a parse are compiled in.
fn take_string(obj: &mut serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    obj.get_mut(key).and_then(|v| match v {
        serde_json::Value::String(s) => Some(std::mem::take(s)),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    })
}

/// Copy a string field out of a JSON object, converting numbers/booleans to
/// their string representation.
///
/// Copies rather than moves, because the caller reuses the same parsed value
/// afterwards. Moving would leave an empty string behind and forward a request
/// the client never sent.
fn copy_string(obj: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(|v| match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    })
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    reason = "tests"
)]
mod tests {

    #[test]
    fn the_owned_path_moves_the_model_out_of_its_throwaway_parse() {
        let body = br#"{"model":"gpt-4.1","input":"hi"}"#;
        assert_eq!(classify_request_body(body).model.as_deref(), Some("gpt-4.1"));
    }

    #[cfg(feature = "openai-responses")]
    #[test]
    fn the_borrowed_path_leaves_the_caller_s_value_intact() {
        // A shared parse is reused afterwards to build request state, so the
        // model must survive classification rather than be moved out.
        let mut value: serde_json::Value = serde_json::from_slice(br#"{"model":"gpt-4.1","input":"hi"}"#).unwrap();
        let obj = value.as_object_mut().unwrap();

        let classified = classify_object(obj);

        assert_eq!(classified.model.as_deref(), Some("gpt-4.1"));
        assert_eq!(
            obj.get("model").and_then(serde_json::Value::as_str),
            Some("gpt-4.1"),
            "classification must not empty the caller's model field"
        );
    }
    use super::*;

    #[test]
    fn responses_string_input() {
        let body = br#"{"model":"gpt-4.1-mini","input":"Hello, world!"}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "string input should classify as responses"
        );
        assert_eq!(
            result.model.as_deref(),
            Some("gpt-4.1-mini"),
            "model should be extracted"
        );
    }

    #[test]
    fn responses_array_input() {
        let body = br#"{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "array input should classify as responses"
        );
        assert_eq!(result.model.as_deref(), Some("gpt-4.1"), "model should be extracted");
    }

    #[test]
    fn responses_null_input_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","input":null}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "input key should classify as responses even when input is null"
        );
        assert_eq!(result.model.as_deref(), Some("gpt-4.1"), "model should be extracted");
    }

    #[test]
    fn responses_with_stream_store_previous_response_id() {
        let body =
            br#"{"model":"gpt-4.1","input":"test","stream":true,"store":false,"background":true,"previous_response_id":"resp_abc"}"#;
        let result = classify_request_body(body);

        assert_eq!(result.format, AiRequestFormat::Responses, "should be responses");
        assert_eq!(result.stream, Some(true), "stream should be extracted");
        assert_eq!(result.store, Some(false), "store should be extracted");
        assert_eq!(result.background, Some(true), "background should be extracted");
        assert!(
            result.has_previous_response_id,
            "previous_response_id should be detected"
        );
    }

    #[test]
    fn responses_max_output_tokens_extracted() {
        let body = br#"{"model":"gpt-4.1","input":"test","max_output_tokens":2048}"#;
        let result = classify_request_body(body);

        assert_eq!(result.format, AiRequestFormat::Responses, "should be responses");
        assert_eq!(
            result.max_output_tokens,
            Some(2048),
            "max_output_tokens should be extracted"
        );
        assert!(result.max_tokens.is_none(), "max_tokens should be None");
    }

    #[test]
    fn responses_absent_max_output_tokens_is_none() {
        let body = br#"{"model":"gpt-4.1","input":"test"}"#;
        let result = classify_request_body(body);

        assert!(
            result.max_output_tokens.is_none(),
            "absent max_output_tokens should be None"
        );
    }

    #[test]
    fn responses_with_conversation() {
        let body = br#"{"model":"gpt-4.1","input":"test","conversation":{"id":"conv_123"}}"#;
        let result = classify_request_body(body);

        assert_eq!(result.format, AiRequestFormat::Responses, "should be responses");
        assert!(result.has_conversation, "conversation should be detected");
        assert!(!result.has_previous_response_id, "no previous_response_id");
    }

    #[test]
    fn chat_completions_messages_without_max_tokens() {
        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::ChatCompletions,
            "messages without max_tokens should classify as chat_completions"
        );
        assert_eq!(result.model.as_deref(), Some("gpt-4"), "model should be extracted");
    }

    #[test]
    fn chat_completions_with_stream() {
        let body = br#"{"model":"gpt-4","messages":[],"stream":true}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::ChatCompletions,
            "should be chat_completions"
        );
        assert_eq!(result.stream, Some(true), "stream should be extracted");
    }

    #[test]
    fn anthropic_messages_with_system() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"system":"Be helpful.","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::AnthropicMessages,
            "messages + max_tokens + system should classify as anthropic_messages"
        );
        assert_eq!(
            result.model.as_deref(),
            Some("claude-opus-4-8"),
            "model should be extracted"
        );
        assert_eq!(result.max_tokens, Some(1024), "max_tokens should be extracted");
    }

    #[test]
    fn anthropic_messages_with_typed_content_blocks() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":512,"messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}],"stream":true}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::AnthropicMessages,
            "typed content blocks should classify as anthropic_messages"
        );
        assert_eq!(result.stream, Some(true), "stream should be extracted");
    }

    #[test]
    fn chat_completions_with_max_tokens_not_misclassified() {
        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::ChatCompletions,
            "messages + max_tokens without Anthropic signals should be chat_completions"
        );
    }

    #[test]
    fn anthropic_messages_max_tokens_without_signals_is_chat() {
        let body = br#"{"model":"claude-opus-4-8","max_tokens":1024,"messages":[{"role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::ChatCompletions,
            "max_tokens + string content + no system should be chat_completions — header override disambiguates in the filter layer"
        );
    }

    #[test]
    fn chat_completions_with_null_system_not_misclassified() {
        let body = br#"{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}],"max_tokens":100,"system":null}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::ChatCompletions,
            "system: null should not be treated as an Anthropic signal"
        );
    }

    #[test]
    fn unknown_json_no_input_no_messages() {
        let body = br#"{"model":"gpt-4","prompt":"hello"}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::UnknownJson,
            "JSON without input or messages should be unknown"
        );
        assert_eq!(
            result.model.as_deref(),
            Some("gpt-4"),
            "model should still be extracted"
        );
    }

    #[test]
    fn invalid_json() {
        let body = b"not json at all {{{";
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::InvalidJson,
            "garbage should be invalid_json"
        );
        assert!(result.model.is_none(), "no model from invalid JSON");
    }

    #[test]
    fn empty_body() {
        let result = classify_request_body(b"");

        assert_eq!(result.format, AiRequestFormat::NonJson, "empty body should be non_json");
    }

    #[test]
    fn json_array_is_invalid() {
        let body = b"[1, 2, 3]";
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::InvalidJson,
            "JSON array should be invalid (not an object)"
        );
    }

    #[test]
    fn null_previous_response_id_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","previous_response_id":null}"#;
        let result = classify_request_body(body);

        assert!(
            !result.has_previous_response_id,
            "null previous_response_id should not be detected as present"
        );
    }

    #[test]
    fn null_conversation_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","conversation":null}"#;
        let result = classify_request_body(body);

        assert!(
            !result.has_conversation,
            "null conversation should not be detected as present"
        );
    }

    #[test]
    fn missing_model_returns_none() {
        let body = br#"{"input":"test"}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "should still classify as responses"
        );
        assert!(result.model.is_none(), "missing model should return None");
    }

    #[test]
    fn stream_and_store_absent_returns_none() {
        let body = br#"{"model":"gpt-4.1","input":"test"}"#;
        let result = classify_request_body(body);

        assert!(result.stream.is_none(), "absent stream should be None");
        assert!(result.store.is_none(), "absent store should be None");
        assert!(result.background.is_none(), "absent background should be None");
    }

    #[test]
    fn background_false_extracted() {
        let body = br#"{"model":"gpt-4.1","input":"test","background":false}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.background,
            Some(false),
            "top-level boolean background:false should be extracted"
        );
    }

    #[test]
    fn null_background_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","background":null}"#;
        let result = classify_request_body(body);

        assert!(
            result.background.is_none(),
            "null background should not be detected as present"
        );
    }

    #[test]
    fn tools_non_empty_array_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function"}]}"#;
        let result = classify_request_body(body);

        assert!(result.has_tools, "non-empty tools array should be detected");
    }

    #[test]
    fn tools_empty_array_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","tools":[]}"#;
        let result = classify_request_body(body);

        assert!(!result.has_tools, "empty tools array should not be detected");
    }

    #[test]
    fn tools_absent_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test"}"#;
        let result = classify_request_body(body);

        assert!(!result.has_tools, "absent tools should not be detected");
    }

    #[test]
    fn tools_null_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","tools":null}"#;
        let result = classify_request_body(body);

        assert!(!result.has_tools, "null tools should not be detected");
    }

    #[test]
    fn prompt_id_nested_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt":{"id":"pmpt_123"}}"#;
        let result = classify_request_body(body);

        assert!(result.has_prompt_id, "nested prompt.id should be detected");
    }

    #[test]
    fn prompt_id_absent_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test"}"#;
        let result = classify_request_body(body);

        assert!(!result.has_prompt_id, "absent prompt should not be detected");
    }

    #[test]
    fn prompt_id_null_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt":{"id":null}}"#;
        let result = classify_request_body(body);

        assert!(!result.has_prompt_id, "null prompt.id should not be detected");
    }

    #[test]
    fn prompt_object_without_prompt_id_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt":{"variables":{"city":"SF"}}}"#;
        let result = classify_request_body(body);

        assert!(
            !result.has_prompt_id,
            "prompt object without id should not set has_prompt_id"
        );
    }

    #[test]
    fn prompt_object_prompt_id_field_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt":{"id":"pmpt_123"}}"#;
        let result = classify_request_body(body);

        assert!(
            result.has_prompt_id,
            "prompt.id should be detected as the prompt identifier"
        );
    }

    #[test]
    fn prompt_string_not_detected_as_prompt_id() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt":"some string"}"#;
        let result = classify_request_body(body);

        assert!(
            !result.has_prompt_id,
            "string prompt should not be treated as prompt object"
        );
    }

    #[test]
    fn prompt_object_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","prompt":{"id":"pmpt_123","variables":{"city":"SF"}}}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "prompt object should classify as responses even without input"
        );
        assert!(result.has_prompt_id, "prompt.id should be detected");
    }

    #[test]
    fn top_level_prompt_id_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","prompt_id":"pmpt_123"}"#;
        let result = classify_request_body(body);

        assert!(
            !result.has_prompt_id,
            "top-level prompt_id should not be detected (must be nested in prompt object)"
        );
    }

    #[test]
    fn non_boolean_background_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":"test","background":"true"}"#;
        let result = classify_request_body(body);

        assert!(
            result.background.is_none(),
            "non-boolean background should not be detected as present"
        );
    }

    #[test]
    fn nested_background_not_detected() {
        let body = br#"{"model":"gpt-4.1","input":[{"type":"input_image","background":true}]}"#;
        let result = classify_request_body(body);

        assert!(
            result.background.is_none(),
            "nested background fields should not be detected as top-level background"
        );
    }

    #[test]
    fn oversized_model_extracted() {
        let long_model = "x".repeat(1024);
        let body = format!(r#"{{"model":"{long_model}","input":"test"}}"#);
        let result = classify_request_body(body.as_bytes());

        assert_eq!(
            result.model.as_deref(),
            Some(long_model.as_str()),
            "oversized model should still be extracted by classifier"
        );
    }

    #[test]
    fn both_input_and_messages_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","input":"test","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "input takes precedence when both input and messages are present"
        );
    }

    #[test]
    fn previous_response_id_with_messages_classifies_as_responses() {
        let body =
            br#"{"model":"gpt-4.1","previous_response_id":"resp_abc","messages":[{"role":"user","content":"Hi"}]}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "previous_response_id should take precedence over messages"
        );
    }

    #[test]
    fn conversation_with_messages_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","conversation":{"id":"conv_123"},"messages":[{"role":"user","content":"Hi"}],"max_tokens":1024,"system":"Be helpful."}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "conversation should take precedence over Anthropic signals"
        );
    }

    // -------------------------------------------------------------------------
    // Create-Endpoint Classification
    // -------------------------------------------------------------------------

    #[test]
    fn create_matches_post_v1_responses() {
        assert!(
            is_responses_create(&http::Method::POST, "/v1/responses"),
            "POST /v1/responses should match create"
        );
    }

    #[test]
    fn create_matches_post_v1_responses_trailing_slash() {
        assert!(
            is_responses_create(&http::Method::POST, "/v1/responses/"),
            "POST /v1/responses/ should match create"
        );
    }

    #[test]
    fn create_rejects_get() {
        assert!(
            !is_responses_create(&http::Method::GET, "/v1/responses"),
            "GET /v1/responses should not match create"
        );
    }

    #[test]
    fn create_rejects_cancel_subresource() {
        assert!(
            !is_responses_create(&http::Method::POST, "/v1/responses/resp_abc/cancel"),
            "POST /v1/responses/{{id}}/cancel should not match create"
        );
    }

    #[test]
    fn create_rejects_input_tokens() {
        assert!(
            !is_responses_create(&http::Method::POST, "/v1/responses/input_tokens"),
            "POST /v1/responses/input_tokens should not match create"
        );
    }

    #[test]
    fn create_rejects_compact() {
        assert!(
            !is_responses_create(&http::Method::POST, "/v1/responses/compact"),
            "POST /v1/responses/compact should not match create"
        );
    }

    #[test]
    fn create_rejects_chat_completions() {
        assert!(
            !is_responses_create(&http::Method::POST, "/v1/chat/completions"),
            "POST /v1/chat/completions should not match create"
        );
    }

    #[test]
    fn chat_create_matches_post_v1_chat_completions() {
        assert!(
            is_chat_completions_create(&http::Method::POST, "/v1/chat/completions"),
            "POST /v1/chat/completions should match chat create"
        );
    }

    #[test]
    fn chat_create_rejects_post_v1_chat_completions_trailing_slash() {
        assert!(
            !is_chat_completions_create(&http::Method::POST, "/v1/chat/completions/"),
            "POST /v1/chat/completions/ should not match chat create"
        );
    }

    #[test]
    fn chat_create_rejects_get() {
        assert!(
            !is_chat_completions_create(&http::Method::GET, "/v1/chat/completions"),
            "GET /v1/chat/completions should not match chat create"
        );
    }

    #[test]
    fn chat_create_rejects_responses() {
        assert!(
            !is_chat_completions_create(&http::Method::POST, "/v1/responses"),
            "POST /v1/responses should not match chat create"
        );
    }

    #[test]
    fn chat_create_rejects_legacy_completions() {
        assert!(
            !is_chat_completions_create(&http::Method::POST, "/v1/completions"),
            "POST /v1/completions should not match chat create"
        );
    }

    #[test]
    fn chat_create_rejects_subresource() {
        assert!(
            !is_chat_completions_create(&http::Method::POST, "/v1/chat/completions/chatcmpl_abc"),
            "POST /v1/chat/completions/{{id}} should not match chat create"
        );
    }

    #[test]
    fn previous_response_id_only_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","previous_response_id":"resp_abc"}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "previous_response_id without input should classify as responses"
        );
        assert!(
            result.has_previous_response_id,
            "previous_response_id should be detected"
        );
    }

    #[test]
    fn previous_response_id_with_input_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","previous_response_id":"resp_abc","input":"hello"}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "previous_response_id with input should still classify as responses"
        );
    }

    #[test]
    fn conversation_only_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","conversation":{"id":"conv_123"}}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "conversation without input should classify as responses"
        );
        assert!(result.has_conversation, "conversation should be detected");
    }

    #[test]
    fn null_previous_response_id_still_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","previous_response_id":null}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "previous_response_id key present (even null) should classify as responses"
        );
        assert!(
            !result.has_previous_response_id,
            "null value should not set the has_previous_response_id flag"
        );
    }

    #[test]
    fn null_conversation_still_classifies_as_responses() {
        let body = br#"{"model":"gpt-4.1","conversation":null}"#;
        let result = classify_request_body(body);

        assert_eq!(
            result.format,
            AiRequestFormat::Responses,
            "conversation key present (even null) should classify as responses"
        );
        assert!(
            !result.has_conversation,
            "null value should not set the has_conversation flag"
        );
    }

    #[test]
    fn control_char_model_extracted() {
        let body = b"{\"model\":\"bad\\nmodel\",\"input\":\"test\"}";
        let result = classify_request_body(body);

        assert_eq!(
            result.model.as_deref(),
            Some("bad\nmodel"),
            "model with control chars should still be extracted by classifier"
        );
    }
}
