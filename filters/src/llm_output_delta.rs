// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Detect SSE payloads that carry generated output tokens.
//!
//! Shared by TTFT, TPOT, and ITL so their event definitions stay aligned.
//! OpenAI rules mirror `vllm bench serve`: Chat Completions counts content,
//! reasoning, tool calls, and function calls; Responses counts only
//! output-text and reasoning-text deltas.

/// Largest event inspected for output-delta structure.
///
/// Output deltas are small. Large terminal events (for example a Responses
/// `response.completed` payload) can embed the whole response and are never
/// output deltas; skipping them keeps the hot path bounded.
const MAX_OUTPUT_DELTA_PAYLOAD_BYTES: usize = 64 * 1024;

/// Returns whether an SSE data payload carries a generated output delta.
#[expect(clippy::too_many_lines, reason = "provider rules are clearer inline")]
pub(crate) fn payload_has_output_delta(payload: &[u8]) -> bool {
    if payload.is_empty() || payload.len() > MAX_OUTPUT_DELTA_PAYLOAD_BYTES {
        return false;
    }

    let Ok(value) = serde_json::from_slice::<serde_json::Value>(payload) else {
        return false;
    };

    if let Some(event_type) = value.get("type").and_then(serde_json::Value::as_str) {
        if event_type.starts_with("response.") {
            return matches!(
                event_type,
                "response.output_text.delta" | "response.reasoning_text.delta"
            );
        }
        if event_type == "content_block_delta" {
            return ["text", "thinking", "partial_json"].iter().any(|key| {
                value
                    .get("delta")
                    .and_then(|delta| delta.get(*key))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|text| !text.is_empty())
            });
        }
        if event_type == "contentBlockDelta" {
            return ["text", "input"].iter().any(|key| {
                value
                    .get("delta")
                    .and_then(|delta| delta.get(*key))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|text| !text.is_empty())
            });
        }
    }

    if let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array) {
        let Some(choice) = choices.first() else {
            return false;
        };
        if choice.get("text").is_some() {
            return true;
        }
        if let Some(delta) = choice.get("delta") {
            if delta
                .get("content")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|content| !content.is_empty() || delta.get("role").is_none())
            {
                return true;
            }
            if ["reasoning", "reasoning_content"].iter().any(|key| {
                delta
                    .get(*key)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|text| !text.is_empty())
            }) {
                return true;
            }
            if delta.get("tool_calls").is_some() || delta.get("function_call").is_some() {
                return true;
            }
        }
        return false;
    }

    if let Some(candidates) = value.get("candidates").and_then(serde_json::Value::as_array) {
        return candidates.iter().any(|candidate| {
            candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(serde_json::Value::as_array)
                .is_some_and(|parts| {
                    parts.iter().any(|part| {
                        part.get("text")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|text| !text.is_empty())
                    })
                })
        });
    }

    false
}

#[cfg(test)]
mod tests {
    use super::payload_has_output_delta;

    #[test]
    #[expect(clippy::too_many_lines, reason = "provider matrix is clearer inline")]
    fn matches_vllm_output_chunk_rules() {
        let cases: &[(&[u8], bool)] = &[
            (br#"{"choices":[{"delta":{"role":"assistant"}}]}"#.as_slice(), false),
            (br#"{"choices":[{"delta":{"content":"Hi"}}]}"#.as_slice(), true),
            (br#"{"choices":[{"delta":{"reasoning":"thinking"}}]}"#.as_slice(), true),
            (
                br#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#.as_slice(),
                true,
            ),
            (br#"{"choices":[{"text":"Hi"}]}"#.as_slice(), true),
            (br#"{"choices":[{"text":""}]}"#.as_slice(), true),
            (br#"{"type":"response.created"}"#.as_slice(), false),
            (
                br#"{"type":"response.output_text.delta","delta":"Hi"}"#.as_slice(),
                true,
            ),
            (
                br#"{"type":"response.reasoning_text.delta","delta":"Hi"}"#.as_slice(),
                true,
            ),
            (
                br#"{"type":"response.function_call_arguments.delta","delta":"{"}"#.as_slice(),
                false,
            ),
            (
                br#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}"#.as_slice(),
                true,
            ),
            (
                br#"{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"Hi"}}"#.as_slice(),
                true,
            ),
            (
                br#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"}}"#.as_slice(),
                true,
            ),
            (br#"{"type":"message_start"}"#.as_slice(), false),
            (
                br#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}"#.as_slice(),
                true,
            ),
            (
                br#"{"type":"contentBlockDelta","delta":{"text":"Hi"}}"#.as_slice(),
                true,
            ),
            (br#"{"type":"messageStop"}"#.as_slice(), false),
            (&b""[..], false),
        ];

        for (payload, want) in cases {
            assert_eq!(payload_has_output_delta(payload), *want, "{payload:?}");
        }
    }
}
