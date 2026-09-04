// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Borrowed deserialization of Chat Completions streaming chunks.
//!
//! Fields that may contain JSON escapes use [`Cow`] so serde borrows from the
//! frame buffer when possible and only allocates when unescaping is required.
//! Unknown provider fields are ignored; only fields the translation needs are
//! modeled.

use std::borrow::Cow;

use serde::{Deserialize, de::IgnoredAny};
use serde_json::Value;

/// One Chat Completions streaming chunk (`chat.completion.chunk`).
#[derive(Debug, Deserialize)]
pub(super) struct ChatChunk<'a> {
    /// Stable completion id, when present.
    #[serde(default, borrow)]
    pub id: Option<Cow<'a, str>>,
    /// Object discriminator, expected to be `chat.completion.chunk`.
    #[serde(default, borrow)]
    pub object: Option<Cow<'a, str>>,
    /// Provider model name, when present.
    #[serde(default, borrow)]
    pub model: Option<Cow<'a, str>>,
    /// Provider service tier, when present.
    #[serde(default, borrow)]
    pub service_tier: Option<Cow<'a, str>>,
    /// Response choices; only index `0` is supported by the translation.
    #[serde(default)]
    pub choices: Vec<ChatChoice<'a>>,
    /// Token usage, present only on the final usage-bearing chunk.
    #[serde(default)]
    pub usage: Option<Value>,
}

/// One Chat Completions choice fragment.
#[derive(Debug, Deserialize)]
pub(super) struct ChatChoice<'a> {
    /// Choice index; the translation only accepts `0`.
    #[serde(default)]
    pub index: u64,
    /// Incremental delta for this choice.
    #[serde(default, borrow)]
    pub delta: Option<ChatDelta<'a>>,
    /// Terminal finish reason, present on the final content chunk.
    #[serde(default, borrow)]
    pub finish_reason: Option<Cow<'a, str>>,
    /// Token logprobs for this choice, when requested.
    #[serde(default)]
    pub logprobs: Option<Value>,
}

/// The incremental delta payload of a Chat Completions choice.
#[derive(Debug, Deserialize)]
pub(super) struct ChatDelta<'a> {
    /// Incremental assistant text.
    #[serde(default, borrow)]
    pub content: Option<Cow<'a, str>>,
    /// Incremental refusal text.
    #[serde(default, borrow)]
    pub refusal: Option<Cow<'a, str>>,
    /// Incremental tool-call fragments keyed by their own index.
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCallFragment<'a>>,
    /// Presence marker for the deprecated singular `function_call` field. The
    /// converter does not translate legacy function calling; capturing only its
    /// presence (never its payload, so nothing is allocated) lets the translation
    /// fail closed instead of silently dropping the call.
    #[serde(default)]
    pub function_call: Option<IgnoredAny>,
}

/// One incremental tool-call fragment.
#[derive(Debug, Deserialize)]
pub(super) struct ChatToolCallFragment<'a> {
    /// Tool-call index, stable across fragments of the same call. Absent when
    /// the provider omits it; the translation fails closed rather than
    /// defaulting to `0` and merging distinct calls.
    #[serde(default)]
    pub index: Option<u64>,
    /// Tool-call id fragment, typically present on the first fragment.
    #[serde(default, borrow)]
    pub id: Option<Cow<'a, str>>,
    /// Function name and argument fragments.
    #[serde(default, borrow)]
    pub function: Option<ChatFunctionFragment<'a>>,
}

/// A function name and/or argument fragment.
#[derive(Debug, Deserialize)]
pub(super) struct ChatFunctionFragment<'a> {
    /// Function name fragment.
    #[serde(default, borrow)]
    pub name: Option<Cow<'a, str>>,
    /// Function argument fragment.
    #[serde(default, borrow)]
    pub arguments: Option<Cow<'a, str>>,
}

/// Extract the `logprobs.content` array from a choice logprobs value.
pub(super) fn logprobs_content(logprobs: Option<&Value>) -> Value {
    logprobs
        .and_then(|value| value.get("content"))
        .filter(|content| content.is_array())
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

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
mod tests {
    use super::*;

    #[test]
    fn parses_text_delta_chunk() {
        let raw = br#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"Hel"}}]}"#;
        let chunk: ChatChunk<'_> = serde_json::from_slice(raw).unwrap();
        assert_eq!(chunk.id.as_deref(), Some("chatcmpl_1"));
        assert_eq!(chunk.object.as_deref(), Some("chat.completion.chunk"));
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.as_ref().unwrap().content.as_deref(), Some("Hel"));
    }

    #[test]
    fn borrows_escaped_content_by_allocating() {
        let raw = br#"{"choices":[{"index":0,"delta":{"content":"line\n\"quoted\""}}]}"#;
        let chunk: ChatChunk<'_> = serde_json::from_slice(raw).unwrap();
        assert_eq!(
            chunk.choices[0].delta.as_ref().unwrap().content.as_deref(),
            Some("line\n\"quoted\""),
            "escaped content should be unescaped correctly"
        );
    }

    #[test]
    fn parses_tool_call_fragment() {
        let raw = br#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather"}}]}}]}"#;
        let chunk: ChatChunk<'_> = serde_json::from_slice(raw).unwrap();
        let call = &chunk.choices[0].delta.as_ref().unwrap().tool_calls[0];
        assert_eq!(call.index, Some(0));
        assert_eq!(call.id.as_deref(), Some("call_1"));
        assert_eq!(call.function.as_ref().unwrap().name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn ignores_unknown_fields() {
        let raw = br#"{"id":"x","system_fingerprint":"fp","choices":[{"index":0,"delta":{},"provider_extra":true}]}"#;
        let chunk: ChatChunk<'_> = serde_json::from_slice(raw).unwrap();
        assert_eq!(chunk.id.as_deref(), Some("x"));
    }

    #[test]
    fn logprobs_content_defaults_to_empty_array() {
        assert_eq!(logprobs_content(None), Value::Array(Vec::new()));
        assert_eq!(
            logprobs_content(Some(&serde_json::json!({"content": [{"token": "a"}]}))),
            serde_json::json!([{"token": "a"}])
        );
    }
}
