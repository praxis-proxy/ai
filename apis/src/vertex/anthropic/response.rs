// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Vertex AI response transformation.
//!
//! Vertex answers with the Anthropic Messages wire shape but with three
//! dialect leaks this module closes:
//!
//! - the response `model` is the Vertex snapshot id (`claude-sonnet-4-5-20250929`), not the user-facing id clients
//!   sent; clients that compare request vs response `model` would break;
//! - platform-layer failures (quota, auth, org policy, missing model) arrive in Google's error envelope, while client
//!   retry logic keys on Anthropic error types;
//! - in streaming responses the `model` lives inside the `message_start` SSE event (`message.model`), so the rewrite is
//!   a frame-scoped patch, never a whole-stream re-serialization.
//!
//! Error translation covers **pre-stream** failures only: a mid-stream
//! kill arrives as a dropped stream with no body to rewrite, and is
//! forwarded as-is.

use serde_json::{Map, Value};

use crate::{
    anthropic::{ErrorType, error_body},
    openai::sse::SseFrameParser,
};

/// Translate a Google error envelope into the Anthropic error shape.
///
/// Returns `None` (forward unchanged) when the body is not JSON, is not
/// a Google `{"error":{…}}` envelope, or already carries the Anthropic
/// `{"type":"error",…}` shape, since Vertex model-layer errors are
/// Anthropic-shaped at the source and must not be double-wrapped.
/// The HTTP status is preserved so client retry policies see the
/// familiar code, and the Anthropic `error.type` comes from
/// [`ErrorType::from_status`], the same mapping the rest of the Anthropic
/// surface uses (Google-envelope statuses such as `RESOURCE_EXHAUSTED`
/// 429 and `UNAUTHENTICATED` 401 share the numeric code semantics).
pub(crate) fn translate_google_error(body: &[u8], status: u16) -> Option<Vec<u8>> {
    let value: Value = serde_json::from_slice(body).ok()?;
    if value.get("type").and_then(Value::as_str) == Some("error") {
        return None;
    }
    let error = value.get("error")?.as_object()?;

    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("upstream returned an unparseable error");
    let google_status = error.get("status").and_then(Value::as_str);
    let detailed = match google_status {
        Some(google_status) => format!("{message} [{google_status}]"),
        None => message.to_owned(),
    };

    Some(error_body(ErrorType::from_status(status), &detailed, None))
}

/// Restore the user-facing model id in a buffered JSON response body.
/// Returns `None` (forward unchanged) when the body is not a JSON
/// object, carries no `model` (e.g. `count_tokens` responses, where injecting
/// one would be wrong), or already carries the user-facing id.
pub(crate) fn restore_model(body: &[u8], user_model: &str) -> Option<Vec<u8>> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object_mut()?;
    patch_model_field(obj, user_model)?;
    serde_json::to_vec(&value).ok()
}

/// Replace a string `model` field with `user_model`, at the top level
/// or nested under `message` (the `message_start` SSE event shape).
/// Returns whether a replacement was made.
fn patch_model_field(obj: &mut Map<String, Value>, user_model: &str) -> Option<()> {
    if obj
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|current| current != user_model)
    {
        obj.insert("model".to_owned(), Value::String(user_model.to_owned()));
        return Some(());
    }
    if let Some(Value::Object(message)) = obj.get_mut("message") {
        return patch_model_field(message, user_model);
    }
    None
}

/// Event name whose payload carries the model to restore.
const MESSAGE_START: &[u8] = b"message_start";

/// Restore message-start models while forwarding every other wire frame unchanged.
pub(crate) fn rebuild_sse_frames(frames: &[Vec<u8>], user_model: Option<&str>, initial_frame: bool) -> Vec<u8> {
    let mut output = Vec::new();
    for (index, raw) in frames.iter().enumerate() {
        let patched = user_model
            .filter(|_| raw.windows(MESSAGE_START.len()).any(|bytes| bytes == MESSAGE_START))
            .and_then(|model| patch_wire_frame(raw, model, initial_frame && index == 0));
        output.extend_from_slice(patched.as_deref().unwrap_or(raw));
    }
    output
}

/// Preserve non-data fields while replacing a message-start JSON payload.
fn patch_wire_frame(raw: &[u8], user_model: &str, initial_frame: bool) -> Option<Vec<u8>> {
    let raw = if initial_frame {
        raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(raw)
    } else {
        raw
    };
    let mut parser = SseFrameParser::new(raw.len());
    // BOM recognition belongs to the logical stream, not to each parser instance.
    parser.parse_chunk(b"\n").ok()?;
    let frames = parser.parse_chunk(raw).ok()?;
    let frame = frames
        .first()
        .filter(|frame| frame.event_type.as_deref() == Some("message_start"))?;
    let patched = restore_model(&frame.data, user_model)?;
    Some(replace_wire_data(raw, &patched))
}

/// Keep the ordered non-data fields and emit exactly one replacement payload.
fn replace_wire_data(raw: &[u8], patched: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    let mut wrote_data = false;
    for line in raw
        .split(|byte| matches!(byte, b'\r' | b'\n'))
        .filter(|line| !line.is_empty())
    {
        if line == b"data" || line.starts_with(b"data:") {
            if wrote_data {
                continue;
            }
            output.extend_from_slice(b"data: ");
            output.extend_from_slice(patched);
            wrote_data = true;
        } else {
            output.extend_from_slice(line);
        }
        output.push(b'\n');
    }
    output.push(b'\n');
    output
}

/// Signal a failed logical stream in the client's Anthropic wire format.
pub(super) fn sse_error(message: &str) -> Vec<u8> {
    let mut output = b"event: error\ndata: ".to_vec();
    output.extend_from_slice(&error_body(ErrorType::Api, message, None));
    output.extend_from_slice(b"\n\n");
    output
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn google_envelope_translates_with_status_mapping() {
        let body = json!({
            "error": {"code": 429, "message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}
        })
        .to_string();
        let translated: Value = serde_json::from_slice(&translate_google_error(body.as_bytes(), 429).unwrap()).unwrap();
        assert_eq!(translated["type"], "error");
        assert_eq!(translated["error"]["type"], "rate_limit_error");
        assert_eq!(
            translated["error"]["message"], "Resource has been exhausted [RESOURCE_EXHAUSTED]",
            "google status must ride along for support cases"
        );

        for (status, expected) in [
            (400_u16, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "invalid_request_error"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (502, "api_error"),
            (503, "api_error"),
            (504, "timeout_error"),
            (520, "api_error"),
            (529, "overloaded_error"),
        ] {
            let body = json!({"error": {"code": status, "message": "x"}}).to_string();
            let out: Value = serde_json::from_slice(&translate_google_error(body.as_bytes(), status).unwrap()).unwrap();
            assert_eq!(out["error"]["type"], expected, "status {status}");
        }
    }

    #[test]
    fn already_anthropic_shaped_bodies_pass_through() {
        // Vertex model-layer errors are Anthropic-shaped at the source.
        let anthropic = json!({"type": "error", "error": {"type": "invalid_request_error", "message": "model: Extra inputs are not permitted"}})
            .to_string();
        assert!(translate_google_error(anthropic.as_bytes(), 400).is_none());

        // Non-JSON (HTML proxy errors, empty bodies) passes through too.
        assert!(translate_google_error(b"<html>bad gateway</html>", 502).is_none());
        assert!(translate_google_error(b"", 500).is_none());

        // Google envelope without a usable message still translates.
        let odd: Value =
            serde_json::from_slice(&translate_google_error(br#"{"error": {"code": 400}}"#, 400).unwrap()).unwrap();
        assert_eq!(odd["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn restore_model_patches_top_level_and_skips_correctly() {
        let body =
            json!({"model": "claude-sonnet-4-5-20250929", "type": "message", "usage": {"input_tokens": 3}}).to_string();
        let restored: Value =
            serde_json::from_slice(&restore_model(body.as_bytes(), "vertex/claude-sonnet-4-5").unwrap()).unwrap();
        assert_eq!(restored["model"], "vertex/claude-sonnet-4-5");
        assert_eq!(restored["usage"]["input_tokens"], 3, "metering fields untouched");

        // Already user-facing, model absent (count_tokens), or non-JSON:
        // all forward unchanged.
        assert!(restore_model(br#"{"model":"vertex/m"}"#, "vertex/m").is_none());
        assert!(
            restore_model(br#"{"input_tokens": 5}"#, "vertex/x").is_none(),
            "absent model must not be injected"
        );
        assert!(restore_model(b"not json", "m").is_none());
    }

    #[test]
    fn restore_model_reaches_nested_message_model() {
        let body = br#"{"type":"message_start","message":{"model":"claude-sonnet-4-5-20250929","role":"assistant"}}"#;
        let patched: Value = serde_json::from_slice(&restore_model(body, "vertex/claude-sonnet-4-5").unwrap()).unwrap();
        assert_eq!(patched["message"]["model"], "vertex/claude-sonnet-4-5");
        assert_eq!(patched["type"], "message_start", "event payload otherwise untouched");
    }

    #[test]
    fn sse_rebuild_preserves_comments_ids_retry_and_multiline_data() {
        let start =
            b": start\nid: first\nretry: 1000\nevent: message_start\ndata: {\"message\":{\"model\":\"snapshot\"}}\n\n"
                .to_vec();
        let unchanged =
            b": ping\r\nid: second\r\nretry: 2000\r\nevent: content_block_delta\r\ndata: first\r\ndata: second\r\n\r\n"
                .to_vec();
        let multiline_json = b"event: content_block_delta\ndata: {\ndata: \"type\":\"content_block_delta\",\ndata: \"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}\ndata: }\n\n".to_vec();
        let frames = vec![start, unchanged.clone(), multiline_json.clone()];
        let out = rebuild_sse_frames(&frames, Some("vertex/model"), true);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"model\":\"vertex/model\""));
        assert!(text.starts_with(": start\nid: first\nretry: 1000\n"));
        assert!(
            text.as_bytes().ends_with(&[unchanged, multiline_json].concat()),
            "unmodified frames must preserve multiline JSON and all wire bytes"
        );
    }

    #[test]
    fn sse_rebuild_preserves_delta_payloads_containing_the_candidate_marker() {
        let raw = b"event: content_block_delta\r\ndata: {\"delta\":{\"text\":\"message_start\"},\"model\":\"snapshot\"}\r\n\r\n";
        assert_eq!(rebuild_sse_frames(&[raw.to_vec()], Some("public-model"), true), raw);
    }

    #[test]
    fn sse_rebuild_keeps_malformed_or_unmapped_message_start_unchanged() {
        let frames = vec![b"event: message_start\ndata: {not json\n\n".to_vec()];
        assert_eq!(rebuild_sse_frames(&frames, Some("vertex/x"), true), frames.concat());
        assert_eq!(rebuild_sse_frames(&frames, None, true), frames.concat());
    }

    #[test]
    fn sse_rebuild_handles_bom_and_data_before_message_start_event() {
        let raw = b"\xEF\xBB\xBFdata: {\"message\":{\"model\":\"snapshot\"}}\nid: first\nevent: message_start\n\n";
        let output = rebuild_sse_frames(&[raw.to_vec()], Some("vertex/client-model"), true);
        let frames = SseFrameParser::new(1024).parse_chunk(&output).unwrap();
        assert_eq!(frames.len(), 1, "rewritten SSE must contain a single event");
        let value: Value = serde_json::from_slice(&frames[0].data).unwrap();
        assert_eq!(value["message"]["model"], "vertex/client-model");
        assert!(!output.windows(8).any(|bytes| bytes == b"snapshot"));
        assert!(output.windows(9).any(|bytes| bytes == b"id: first"));
    }
    #[test]
    fn sse_rebuild_does_not_strip_a_bom_after_stream_start() {
        let raw = b"\xEF\xBB\xBFdata: {\"message\":{\"model\":\"snapshot\"}}\nevent: message_start\n\n";
        assert_eq!(
            rebuild_sse_frames(&[raw.to_vec()], Some("vertex/client-model"), false),
            raw
        );
    }
}
