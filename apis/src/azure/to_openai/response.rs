// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Azure OpenAI response transformation.
//!
//! Azure OpenAI responses are Chat Completions-compatible. The only
//! normalization needed is stripping Azure-specific content-filter
//! fields (`prompt_filter_results`, per-choice `content_filter_results`,
//! `content_filter_offsets`, `content_filter_raw`) so downstream clients
//! receive a clean Chat Completions response.
//!
//! Streaming `data:` payloads are parsed once via [`normalize_sse_payload`],
//! which both strips those fields and classifies async-filter annotation
//! chunks.

#[cfg(test)]
use std::cell::Cell;

use serde_json::Value;

/// Azure-specific top-level fields to strip from responses.
const TOP_LEVEL_STRIP: &[&str] = &["prompt_filter_results"];

/// Azure-specific per-choice fields to strip.
const CHOICE_STRIP: &[&str] = &["content_filter_results", "content_filter_offsets", "content_filter_raw"];

/// Outcome of a single parse-and-normalize pass over an Azure streaming `data:` payload.
#[derive(Debug, Eq, PartialEq)]
#[must_use]
pub(crate) enum SsePayloadAction<'a> {
    /// Forward the original payload; no Azure-only field was removed.
    ForwardOriginal(&'a [u8]),
    /// Forward rewritten JSON with Azure-only fields removed.
    ForwardRewritten(Vec<u8>),
    /// Drop an Azure async-filter annotation that has no stream content.
    Drop,
}

/// Parse a streaming payload once, strip Azure-only fields, and classify filter-only chunks.
///
/// Malformed JSON is forwarded unchanged, matching the historical double-parse
/// path (`strip_azure_fields` returning `None`, then filter-only classification
/// returning `false`).
pub(crate) fn normalize_sse_payload(data: &[u8]) -> SsePayloadAction<'_> {
    let Some(mut value) = parse_sse_json(data) else {
        return SsePayloadAction::ForwardOriginal(data);
    };

    let modified = strip_azure_fields_in_place(&mut value);
    if is_filter_only_value(&value) {
        return SsePayloadAction::Drop;
    }

    if modified {
        rewritten_or_original(data, &value)
    } else {
        SsePayloadAction::ForwardOriginal(data)
    }
}

/// Strip Azure-specific fields from a Chat Completions response.
///
/// Returns `Some(cleaned_bytes)` when fields were removed, or `None`
/// when the response was already clean (callers should keep the original).
pub(crate) fn strip_azure_fields(body: &[u8]) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    if strip_azure_fields_in_place(&mut value) {
        serde_json::to_vec(&value).ok()
    } else {
        None
    }
}

// -----------------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// JSON `from_slice` count for [`normalize_sse_payload`] on this thread.
    static SSE_PAYLOAD_PARSE_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Parse `data` as JSON, recording the attempt when tests measure parse count.
fn parse_sse_json(data: &[u8]) -> Option<Value> {
    #[cfg(test)]
    SSE_PAYLOAD_PARSE_COUNT.with(|count| count.set(count.get() + 1));
    serde_json::from_slice(data).ok()
}

/// Remove Azure-only fields from a parsed Chat Completions value.
///
/// Returns `true` when any field was removed.
fn strip_azure_fields_in_place(value: &mut Value) -> bool {
    let Some(obj) = value.as_object_mut() else {
        return false;
    };

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

    modified
}

/// Serialize a modified payload, or fall back to the original bytes.
fn rewritten_or_original<'a>(original: &'a [u8], value: &Value) -> SsePayloadAction<'a> {
    match serde_json::to_vec(value) {
        Ok(rewritten) => SsePayloadAction::ForwardRewritten(rewritten),
        Err(_) => SsePayloadAction::ForwardOriginal(original),
    }
}

/// Returns `true` when every choice is an Azure async-filter annotation.
fn is_filter_only_value(value: &Value) -> bool {
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return false;
    };
    !choices.is_empty() && choices.iter().all(|choice| !choice_has_stream_payload(choice))
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

    const CLEAN_DELTA: &[u8] = br#"{"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"}}]}"#;
    const WITH_AZURE_FIELDS: &[u8] =
        br#"{"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"},"content_filter_results":{}}]}"#;
    const FILTER_ONLY: &[u8] = br#"{"choices":[{"finish_reason":null,"content_filter_results":{}}]}"#;
    const TERMINAL_WITH_FILTER: &[u8] = br#"{"choices":[{"finish_reason":"stop","content_filter_results":{}}]}"#;
    const EMPTY_CHOICES: &[u8] = br#"{"id":"chatcmpl-abc","choices":[]}"#;

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
        let binding = chunk.to_string();
        let action = normalize_sse_payload(binding.as_bytes());
        assert!(
            !matches!(action, SsePayloadAction::Drop),
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

    // -------------------------------------------------------------------------
    // filter-only classification
    // -------------------------------------------------------------------------

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
        assert_eq!(
            normalize_sse_payload(chunk.to_string().as_bytes()),
            SsePayloadAction::Drop,
            "async-filter annotation with no delta must be dropped"
        );
    }

    #[test]
    fn normal_chunk_with_delta_not_filter_only() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]
        });
        let bytes = chunk.to_string();
        assert_forward_original(bytes.as_bytes());
    }

    #[test]
    fn stripped_chunk_without_delta_is_filter_only() {
        let raw = serde_json::json!({
            "id": "",
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "content_filter_results": {"hate": {"filtered": false}},
                "content_filter_offsets": {"check_offset": 44}
            }]
        });
        assert_eq!(
            normalize_sse_payload(raw.to_string().as_bytes()),
            SsePayloadAction::Drop,
            "annotation with only Azure filter fields must be dropped"
        );
    }

    #[test]
    fn empty_choices_not_filter_only() {
        let chunk = serde_json::json!({"id": "chatcmpl-abc", "choices": []});
        let bytes = chunk.to_string();
        assert_forward_original(bytes.as_bytes());
    }

    // -------------------------------------------------------------------------
    // normalize_sse_payload tests
    // -------------------------------------------------------------------------

    #[test]
    fn normalize_sse_payload_forwards_clean_payload_as_original_slice() {
        assert_forward_original(CLEAN_DELTA);
    }

    #[test]
    fn normalize_sse_payload_forwards_malformed_json_unchanged() {
        assert_forward_original(b"not json {");
    }

    #[test]
    fn normalize_sse_payload_forwards_non_object_json_unchanged() {
        assert_forward_original(b"[1,2,3]");
    }

    #[test]
    fn normalize_sse_payload_rewrites_azure_only_fields() {
        let action = normalize_sse_payload(WITH_AZURE_FIELDS);
        let rewritten = rewritten_slice(&action);
        assert!(
            !rewritten.is_empty(),
            "payload with Azure fields must be rewritten, got {action:?}"
        );
        let parsed: Value = serde_json::from_slice(rewritten).unwrap();
        assert!(
            parsed["choices"][0].get("content_filter_results").is_none(),
            "content_filter_results must be stripped"
        );
        assert_eq!(parsed["choices"][0]["delta"]["content"], "Hi");
    }

    #[test]
    fn normalize_sse_payload_drops_filter_only_chunk() {
        assert_eq!(
            normalize_sse_payload(FILTER_ONLY),
            SsePayloadAction::Drop,
            "annotation chunks with no delta and no finish_reason must be dropped"
        );
    }

    #[test]
    fn normalize_sse_payload_preserves_terminal_finish_reason() {
        let action = normalize_sse_payload(TERMINAL_WITH_FILTER);
        let rewritten = rewritten_slice(&action);
        assert!(
            !rewritten.is_empty(),
            "terminal finish_reason must be forwarded as rewritten JSON, got {action:?}"
        );
        let parsed: Value = serde_json::from_slice(rewritten).unwrap();
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert!(
            parsed["choices"][0].get("content_filter_results").is_none(),
            "terminal chunks must keep finish_reason and drop Azure filter fields"
        );
    }

    #[test]
    fn normalize_sse_payload_does_not_drop_empty_choices() {
        assert_forward_original(EMPTY_CHOICES);
    }

    #[test]
    fn normalize_sse_payload_matches_legacy_double_parse() {
        for data in [
            CLEAN_DELTA,
            WITH_AZURE_FIELDS,
            FILTER_ONLY,
            TERMINAL_WITH_FILTER,
            EMPTY_CHOICES,
            b"not json",
            b"[]",
        ] {
            assert_eq!(
                normalize_sse_payload(data),
                legacy_normalize_sse_payload(data),
                "single-parse must match legacy strip-then-classify"
            );
        }
    }

    #[test]
    fn normalize_sse_payload_parses_each_payload_once() {
        for data in [
            CLEAN_DELTA,
            WITH_AZURE_FIELDS,
            FILTER_ONLY,
            TERMINAL_WITH_FILTER,
            b"not json",
        ] {
            reset_sse_payload_parse_count();
            drop(normalize_sse_payload(data));
            assert_eq!(
                take_sse_payload_parse_count(),
                1,
                "each SSE payload must be parsed at most once"
            );
        }
    }

    #[test]
    fn normalize_sse_payload_allocates_less_than_legacy_for_clean_chunk() {
        let data = large_clean_delta_payload();
        assert_allocates_less_than_legacy(&data);
    }

    #[test]
    fn normalize_sse_payload_allocates_less_than_legacy_for_rewritten_chunk() {
        let data = large_rewritten_delta_payload();
        assert_allocates_less_than_legacy(&data);
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Reset the test-only JSON parse counter used by [`normalize_sse_payload`].
    fn reset_sse_payload_parse_count() {
        SSE_PAYLOAD_PARSE_COUNT.with(|count| count.set(0));
    }

    /// Read and clear the test-only JSON parse counter.
    fn take_sse_payload_parse_count() -> usize {
        SSE_PAYLOAD_PARSE_COUNT.with(|count| count.replace(0))
    }

    /// Assert that `data` is forwarded as the original slice, not rewritten.
    fn assert_forward_original(data: &[u8]) {
        let action = normalize_sse_payload(data);
        assert!(
            matches!(
                &action,
                SsePayloadAction::ForwardOriginal(payload) if std::ptr::eq(*payload, data)
            ),
            "expected original-slice ForwardOriginal, got {action:?}"
        );
    }

    /// Borrow rewritten JSON bytes, or an empty slice when the payload was not rewritten.
    fn rewritten_slice<'a>(action: &'a SsePayloadAction<'a>) -> &'a [u8] {
        match action {
            SsePayloadAction::ForwardRewritten(bytes) => bytes,
            SsePayloadAction::Drop | SsePayloadAction::ForwardOriginal(_) => b"",
        }
    }

    /// Faithful reconstruction of the previous strip-then-reparse SSE path.
    fn legacy_normalize_sse_payload(data: &[u8]) -> SsePayloadAction<'_> {
        let stripped = strip_azure_fields(data);
        let payload = stripped.as_deref().unwrap_or(data);
        let filter_only = serde_json::from_slice::<Value>(payload).is_ok_and(|value| is_filter_only_value(&value));
        if filter_only {
            return SsePayloadAction::Drop;
        }
        match stripped {
            Some(rewritten) => SsePayloadAction::ForwardRewritten(rewritten),
            None => SsePayloadAction::ForwardOriginal(data),
        }
    }

    /// Compare allocations of the single-parse path against the legacy double parse.
    fn assert_allocates_less_than_legacy(data: &[u8]) {
        let optimized = allocation_counter::measure(|| {
            drop(std::hint::black_box(normalize_sse_payload(data)));
        });
        let legacy = allocation_counter::measure(|| {
            drop(std::hint::black_box(legacy_normalize_sse_payload(data)));
        });
        assert_eq!(
            normalize_sse_payload(data),
            legacy_normalize_sse_payload(data),
            "single-parse outcome must match the legacy double-parse path"
        );
        let optimized_bytes = optimized.bytes_total;
        let legacy_bytes = legacy.bytes_total;
        assert!(
            optimized_bytes < legacy_bytes,
            "fewer bytes than strip+reparse (optimized={optimized_bytes} legacy={legacy_bytes})"
        );
    }

    /// Large clean delta payload so the extra legacy parse is measurable.
    fn large_clean_delta_payload() -> Vec<u8> {
        serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{"delta": {"content": "x".repeat(8192)}}]
        })
        .to_string()
        .into_bytes()
    }

    /// Large payload that still requires Azure-field stripping.
    fn large_rewritten_delta_payload() -> Vec<u8> {
        serde_json::json!({
            "id": "chatcmpl-abc",
            "choices": [{
                "delta": {"content": "x".repeat(8192)},
                "content_filter_results": {"hate": {"filtered": false}}
            }],
            "prompt_filter_results": [{"prompt_index": 0}]
        })
        .to_string()
        .into_bytes()
    }
}
