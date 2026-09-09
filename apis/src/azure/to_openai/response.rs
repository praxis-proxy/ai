// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Azure OpenAI response transformation.
//!
//! Azure OpenAI responses are Chat Completions-compatible. The only
//! normalization needed is stripping Azure-specific content-filter
//! fields (`prompt_filter_results`, per-choice `content_filter_results`,
//! `content_filter_offsets`, `content_filter_raw`) so downstream clients
//! receive a clean Chat Completions response.

use serde_json::Value;

/// Azure-specific top-level fields to strip from responses.
const TOP_LEVEL_STRIP: &[&str] = &["prompt_filter_results"];

/// Azure-specific per-choice fields to strip.
const CHOICE_STRIP: &[&str] = &["content_filter_results", "content_filter_offsets", "content_filter_raw"];

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

/// Returns `true` when the SSE payload is an Azure async-filter annotation
/// and carries no Chat Completions stream data.
///
/// Azure asynchronous filtering emits chunks with `content_filter_offsets`
/// and no `delta`. Those crash standard OpenAI SDKs. Do not treat a
/// `finish_reason` chunk as filter-only: Azure often attaches filter
/// metadata to the terminal choice, and dropping it would hide the stop.
pub(crate) fn is_filter_only_chunk(data: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<Value>(data) else {
        return false;
    };
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return false;
    };
    !choices.is_empty() && choices.iter().all(|c| !choice_has_stream_payload(c))
}

/// `true` when a choice still has Chat Completions stream content after
/// Azure fields are stripped: a `delta` and/or a non-null `finish_reason`.
fn choice_has_stream_payload(choice: &Value) -> bool {
    if choice.get("delta").is_some() {
        return true;
    }
    matches!(choice.get("finish_reason"), Some(Value::String(s)) if !s.is_empty())
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
    fn finish_reason_without_delta_is_not_filter_only() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "content_filter_results": {"hate": {"filtered": false}}
            }]
        });
        let stripped = strip_azure_fields(chunk.to_string().as_bytes()).unwrap();
        assert!(
            !is_filter_only_chunk(&stripped),
            "terminal finish_reason must not be dropped with Azure filter metadata"
        );
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

    // -- is_filter_only_chunk tests ------------------------------------------

    #[test]
    fn filter_only_chunk_detected() {
        let chunk = serde_json::json!({
            "id": "",
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "content_filter_results": {"hate": {"filtered": false}},
                "content_filter_offsets": {"check_offset": 44, "start_offset": 44, "end_offset": 198}
            }]
        });
        assert!(is_filter_only_chunk(chunk.to_string().as_bytes()));
    }

    #[test]
    fn normal_chunk_with_delta_not_filter_only() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]
        });
        assert!(!is_filter_only_chunk(chunk.to_string().as_bytes()));
    }

    #[test]
    fn stripped_chunk_without_delta_is_filter_only() {
        // After strip_azure_fields removes content_filter_*, the choice
        // has no delta — this is a filter-only annotation.
        let raw = serde_json::json!({
            "id": "",
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "content_filter_results": {"hate": {"filtered": false}},
                "content_filter_offsets": {"check_offset": 44}
            }]
        });
        let stripped = strip_azure_fields(raw.to_string().as_bytes()).unwrap();
        assert!(is_filter_only_chunk(&stripped));
    }

    #[test]
    fn empty_choices_not_filter_only() {
        let chunk = serde_json::json!({"id": "chatcmpl-abc", "choices": []});
        assert!(!is_filter_only_chunk(chunk.to_string().as_bytes()));
    }
}
