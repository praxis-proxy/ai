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
//!   a frame-scoped patch — never a whole-stream re-serialization.
//!
//! Error translation covers **pre-stream** failures only: a mid-stream
//! kill arrives as a dropped stream with no body to rewrite, and is
//! forwarded as-is.

use serde_json::{Map, Value};

use crate::{anthropic::error_body, openai::sse::SseFrame};

/// Map an HTTP status to the Anthropic error type clients retry on.
/// Vertex Google-envelope statuses (`RESOURCE_EXHAUSTED` → 429,
/// `UNAUTHENTICATED` → 401, …) share the numeric code semantics.
fn anthropic_error_type(status: u16) -> &'static str {
    match status {
        400 | 409 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        500..=502 => "api_error",
        503 | 504 => "overloaded_error",
        _ if status >= 500 => "api_error",
        _ => "invalid_request_error",
    }
}

/// Translate a Google error envelope into the Anthropic error shape.
///
/// Returns `None` (forward unchanged) when the body is not JSON, is not
/// a Google `{"error":{…}}` envelope, or already carries the Anthropic
/// `{"type":"error",…}` shape — Vertex model-layer errors are
/// Anthropic-shaped at the source and must not be double-wrapped.
/// The HTTP status is preserved so client retry policies see the
/// familiar code with an Anthropic `error.type`.
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

    Some(error_body(anthropic_error_type(status), &detailed, None))
}

/// Restore the user-facing model id in a buffered JSON response body.
/// Returns `None` (forward unchanged) when the body is not a JSON
/// object, carries no `model` (e.g. `count_tokens` responses — injecting
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

/// Rebuild SSE frames for the client, restoring the user-facing model
/// inside `message_start` events only. Every other frame — deltas, tool
/// use, `message_stop` — is re-emitted with its original data bytes;
/// partial frames stay inside the caller's [`SseFrameParser`] across
/// chunk boundaries, so at most one in-flight event is ever buffered.
pub(crate) fn rebuild_sse_frames(frames: &[SseFrame], user_model: Option<&str>) -> Vec<u8> {
    let mut output = Vec::new();
    for frame in frames {
        if let Some(event) = &frame.event_type {
            output.extend_from_slice(b"event: ");
            output.extend_from_slice(event.as_bytes());
            output.extend_from_slice(b"\n");
        }

        // Only message_start carries a model worth patching; a failed
        // patch (malformed data) forwards the frame as received rather
        // than corrupting the stream.
        let patched = match (frame.event_type.as_deref(), user_model) {
            (Some("message_start"), Some(user_model)) => restore_model(&frame.data, user_model),
            _ => None,
        };
        let data = patched.as_deref().unwrap_or(&frame.data);

        output.extend_from_slice(b"data: ");
        output.extend_from_slice(data);
        output.extend_from_slice(b"\n\n");
    }
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
            (401_u16, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (500, "api_error"),
            (503, "overloaded_error"),
            (520, "api_error"),
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
    fn sse_rebuild_patches_only_message_start() {
        let frames = vec![
            SseFrame {
                event_type: Some("message_start".to_owned()),
                data:
                    br#"{"type":"message_start","message":{"model":"claude-sonnet-4-5-20250929","role":"assistant"}}"#
                        .to_vec(),
            },
            SseFrame {
                event_type: Some("content_block_delta".to_owned()),
                data: br#"{"type":"content_block_delta","delta":{"text":"hi"}}"#.to_vec(),
            },
        ];
        let out = rebuild_sse_frames(&frames, Some("vertex/claude-sonnet-4-5"));
        let text = String::from_utf8(out).unwrap();
        let mut parts = text.split("\n\n");
        let first = parts.next().unwrap();
        assert!(
            first.starts_with("event: message_start\ndata: {"),
            "frame shape preserved"
        );
        assert!(
            first.contains("\"model\":\"vertex/claude-sonnet-4-5\""),
            "message_start model restored: {first}"
        );
        assert!(!first.contains("20250929"), "snapshot id fully replaced: {first}");
        let second = parts.next().unwrap();
        assert_eq!(
            second, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}",
            "non-message_start frames are byte-identical"
        );
    }

    #[test]
    fn sse_rebuild_without_user_model_passes_frames_through() {
        let frames = vec![SseFrame {
            event_type: Some("message_start".to_owned()),
            data: br#"{"type":"message_start"}"#.to_vec(),
        }];
        let out = rebuild_sse_frames(&frames, None);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n"
        );
    }

    #[test]
    fn sse_rebuild_tolerates_unparseable_message_start() {
        // A malformed message_start data line must not corrupt the
        // stream: it is forwarded as received.
        let frames = vec![SseFrame {
            event_type: Some("message_start".to_owned()),
            data: b"{not json".to_vec(),
        }];
        let out = rebuild_sse_frames(&frames, Some("vertex/x"));
        assert!(String::from_utf8(out).unwrap().contains("data: {not json"));
    }
}
