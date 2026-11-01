// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Detect SSE payloads that carry generated output tokens.
//!
//! Shared by TTFT, TPOT, and ITL so their event definitions stay aligned.
//! OpenAI rules mirror `vllm bench serve`: Chat Completions counts content,
//! reasoning, tool calls, and function calls; Responses counts only
//! output-text and reasoning-text deltas.

/// Returns whether an SSE data payload carries a generated output delta.
pub(crate) fn payload_has_output_delta(payload: &[u8]) -> bool {
    if payload.is_empty() {
        return false;
    }

    if json_string_starts_with(payload, b"\"type\"", b"response.") {
        return json_string_equals(payload, b"\"type\"", b"response.output_text.delta")
            || json_string_equals(payload, b"\"type\"", b"response.reasoning_text.delta");
    }
    if json_string_equals(payload, b"\"type\"", b"content_block_delta") {
        return json_string_nonempty(payload, b"\"text\"")
            || json_string_nonempty(payload, b"\"thinking\"")
            || json_string_nonempty(payload, b"\"partial_json\"");
    }
    if json_string_equals(payload, b"\"type\"", b"contentBlockDelta") {
        return json_string_nonempty(payload, b"\"text\"") || json_string_nonempty(payload, b"\"input\"");
    }

    if find_subslice(payload, b"\"choices\"").is_some() {
        if find_subslice(payload, b"\"text\"").is_some() {
            return true;
        }
        if json_string_nonempty(payload, b"\"content\"")
            || (json_string_empty(payload, b"\"content\"") && find_subslice(payload, b"\"role\"").is_none())
        {
            return true;
        }
        return json_string_nonempty(payload, b"\"reasoning\"")
            || json_string_nonempty(payload, b"\"reasoning_content\"")
            || find_subslice(payload, b"\"tool_calls\"").is_some()
            || find_subslice(payload, b"\"function_call\"").is_some();
    }

    find_subslice(payload, b"\"candidates\"").is_some() && json_string_nonempty(payload, b"\"text\"")
}

/// Return whether the JSON string after `needle` starts with `prefix`.
fn json_string_starts_with(payload: &[u8], needle: &[u8], prefix: &[u8]) -> bool {
    json_string_after(payload, needle).is_some_and(|value| value.starts_with(prefix))
}

/// Return whether the JSON string after `needle` is non-empty.
fn json_string_nonempty(payload: &[u8], needle: &[u8]) -> bool {
    json_string_after(payload, needle).is_some_and(|value| !value.is_empty())
}

/// Return whether the JSON string after `needle` is empty.
fn json_string_empty(payload: &[u8], needle: &[u8]) -> bool {
    json_string_after(payload, needle).is_some_and(<[u8]>::is_empty)
}

/// Return whether the JSON string after `needle` equals `expected`.
fn json_string_equals(payload: &[u8], needle: &[u8], expected: &[u8]) -> bool {
    json_string_after(payload, needle).is_some_and(|value| value == expected)
}

/// Extract a quoted JSON string value immediately following `needle`.
fn json_string_after<'a>(payload: &'a [u8], needle: &[u8]) -> Option<&'a [u8]> {
    let start = find_subslice(payload, needle)?;
    let mut rest = payload.get(start + needle.len()..)?;
    rest = skip_json_separator(rest)?;
    if rest.first() != Some(&b'"') {
        return None;
    }
    rest = rest.get(1..)?;
    let end = rest.iter().position(|&byte| byte == b'"')?;
    rest.get(..end)
}

/// Skip ASCII whitespace and a single `:` before a JSON value.
fn skip_json_separator(mut rest: &[u8]) -> Option<&[u8]> {
    while let Some((&first, tail)) = rest.split_first() {
        if !first.is_ascii_whitespace() {
            break;
        }
        rest = tail;
    }
    let (&separator, tail) = rest.split_first()?;
    if separator != b':' {
        return None;
    }
    rest = tail;
    while let Some((&first, tail)) = rest.split_first() {
        if !first.is_ascii_whitespace() {
            break;
        }
        rest = tail;
    }
    Some(rest)
}

/// Locate `needle` in `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|window| window == needle)
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

    #[test]
    fn large_output_delta_is_not_dropped() {
        let mut payload = Vec::new();
        payload.extend_from_slice(br#"{"choices":[{"delta":{"content":""#);
        payload.extend(std::iter::repeat_n(b'x', 70_000));
        payload.extend_from_slice(br#""}}]}"#);
        assert!(payload_has_output_delta(&payload));
    }

    #[test]
    fn large_terminal_event_is_not_an_output_delta() {
        let mut payload = Vec::new();
        payload.extend_from_slice(br#"{"type":"response.completed","response":{"output":""#);
        payload.extend(std::iter::repeat_n(b'x', 70_000));
        payload.extend_from_slice(br#""}}"#);
        assert!(!payload_has_output_delta(&payload));
    }
}
