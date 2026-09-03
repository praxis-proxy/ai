// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Azure OpenAI response transformation.
//!
//! Azure OpenAI responses are Chat Completions-compatible. The only
//! normalization needed is stripping Azure-specific content-filter
//! fields (`prompt_filter_results`, per-choice `content_filter_results`)
//! so downstream clients receive a clean Chat Completions response.

use serde_json::Value;

/// Azure-specific top-level fields to strip from responses.
const TOP_LEVEL_STRIP: &[&str] = &["prompt_filter_results"];

/// Azure-specific per-choice fields to strip.
const CHOICE_STRIP: &[&str] = &["content_filter_results", "content_filter_offsets"];

/// Strip Azure-specific fields from a Chat Completions response.
///
/// Returns `Some(cleaned_bytes)` when fields were removed, or `None`
/// when the response was already clean (callers should keep the original).
pub(crate) fn strip_azure_fields(body: &[u8]) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object_mut()?;

    let mut modified = false;

    for key in TOP_LEVEL_STRIP {
        if obj.remove(*key).is_some() {
            modified = true;
        }
    }

    if let Some(Value::Array(choices)) = obj.get_mut("choices") {
        for choice in choices {
            if let Some(choice_obj) = choice.as_object_mut() {
                for key in CHOICE_STRIP {
                    if choice_obj.remove(*key).is_some() {
                        modified = true;
                    }
                }
            }
        }
    }

    if !modified {
        return None;
    }

    serde_json::to_vec(&value).ok()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn strips_prompt_filter_results() {
        let input = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}]
        });
        let result = strip_azure_fields(input.to_string().as_bytes()).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(parsed.get("prompt_filter_results").is_none());
        assert_eq!(parsed["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn strips_per_choice_content_filter() {
        let input = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "content_filter_results": {"hate": {"filtered": false}},
                "content_filter_offsets": {"check_offset": 0}
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        });
        let result = strip_azure_fields(input.to_string().as_bytes()).unwrap();
        let parsed: Value = serde_json::from_slice(&result).unwrap();

        assert!(parsed["choices"][0].get("content_filter_results").is_none());
        assert!(parsed["choices"][0].get("content_filter_offsets").is_none());
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn returns_none_for_clean_response() {
        let input = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        });
        assert!(strip_azure_fields(input.to_string().as_bytes()).is_none());
    }

    #[test]
    fn returns_none_for_invalid_json() {
        assert!(strip_azure_fields(b"not json").is_none());
    }
}
