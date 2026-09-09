// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses API streaming event constructors and SSE encoding.
//!
//! Each constructor returns a [`StreamEvent`] whose payload omits the
//! `type` and `sequence_number` fields; [`encode`] injects both at emission
//! time so the state machine owns sequencing centrally.

use serde_json::{Value, json};

/// A Responses streaming event ready for sequencing and encoding.
pub(super) struct StreamEvent {
    /// The Responses SSE `event:` name and payload `type`.
    event_type: &'static str,
    /// Event payload object without `type` or `sequence_number`.
    payload: Value,
}

impl StreamEvent {
    /// Construct an event from its type and partial payload.
    fn new(event_type: &'static str, payload: Value) -> Self {
        Self { event_type, payload }
    }
}

/// Encode one event as a Responses SSE frame, injecting `type` and
/// `sequence_number`, and append it to `out`.
///
/// # Errors
///
/// Returns [`serde_json::Error`] when the payload cannot be serialized.
pub(super) fn encode(mut event: StreamEvent, sequence_number: u64, out: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    if let Value::Object(map) = &mut event.payload {
        map.insert("type".to_owned(), Value::String(event.event_type.to_owned()));
        map.insert("sequence_number".to_owned(), Value::Number(sequence_number.into()));
    }
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.event_type.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    serde_json::to_writer(&mut *out, &event.payload)?;
    out.extend_from_slice(b"\n\n");
    Ok(())
}

/// `response.created` carrying an in-progress resource snapshot.
pub(super) fn response_created(resource: &Value) -> StreamEvent {
    StreamEvent::new("response.created", json!({ "response": resource }))
}

/// `response.in_progress` carrying an in-progress resource snapshot.
pub(super) fn response_in_progress(resource: &Value) -> StreamEvent {
    StreamEvent::new("response.in_progress", json!({ "response": resource }))
}

/// `response.output_item.added` announcing a new output item.
pub(super) fn output_item_added(output_index: usize, item: &Value) -> StreamEvent {
    StreamEvent::new(
        "response.output_item.added",
        json!({ "output_index": output_index, "item": item }),
    )
}

/// `response.output_item.done` completing an output item.
pub(super) fn output_item_done(output_index: usize, item: &Value) -> StreamEvent {
    StreamEvent::new(
        "response.output_item.done",
        json!({ "output_index": output_index, "item": item }),
    )
}

/// `response.content_part.added` announcing a new content part.
pub(super) fn content_part_added(
    item_id: &str,
    output_index: usize,
    content_index: usize,
    part: &Value,
) -> StreamEvent {
    StreamEvent::new(
        "response.content_part.added",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        }),
    )
}

/// `response.content_part.done` completing a content part.
pub(super) fn content_part_done(item_id: &str, output_index: usize, content_index: usize, part: &Value) -> StreamEvent {
    StreamEvent::new(
        "response.content_part.done",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": part,
        }),
    )
}

/// `response.output_text.delta` carrying one incremental text fragment.
pub(super) fn output_text_delta(
    item_id: &str,
    output_index: usize,
    content_index: usize,
    delta: &str,
    logprobs: &Value,
) -> StreamEvent {
    StreamEvent::new(
        "response.output_text.delta",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
            "logprobs": logprobs,
        }),
    )
}

/// `response.output_text.done` carrying the accumulated text.
pub(super) fn output_text_done(
    item_id: &str,
    output_index: usize,
    content_index: usize,
    text: &str,
    logprobs: &Value,
) -> StreamEvent {
    StreamEvent::new(
        "response.output_text.done",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": text,
            "logprobs": logprobs,
        }),
    )
}

/// `response.refusal.delta` carrying one incremental refusal fragment.
pub(super) fn refusal_delta(item_id: &str, output_index: usize, content_index: usize, delta: &str) -> StreamEvent {
    StreamEvent::new(
        "response.refusal.delta",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "delta": delta,
        }),
    )
}

/// `response.refusal.done` carrying the accumulated refusal text.
pub(super) fn refusal_done(item_id: &str, output_index: usize, content_index: usize, refusal: &str) -> StreamEvent {
    StreamEvent::new(
        "response.refusal.done",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "refusal": refusal,
        }),
    )
}

/// `response.function_call_arguments.delta` carrying one argument fragment.
pub(super) fn function_call_arguments_delta(item_id: &str, output_index: usize, delta: &str) -> StreamEvent {
    StreamEvent::new(
        "response.function_call_arguments.delta",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "delta": delta,
        }),
    )
}

/// `response.function_call_arguments.done` carrying the accumulated arguments.
///
/// The Responses schema requires `name` on this event, unlike the `.delta`
/// counterpart.
pub(super) fn function_call_arguments_done(
    item_id: &str,
    output_index: usize,
    name: &str,
    arguments: &str,
) -> StreamEvent {
    StreamEvent::new(
        "response.function_call_arguments.done",
        json!({
            "item_id": item_id,
            "output_index": output_index,
            "name": name,
            "arguments": arguments,
        }),
    )
}

/// `response.completed` carrying the terminal resource snapshot.
pub(super) fn response_completed(resource: &Value) -> StreamEvent {
    StreamEvent::new("response.completed", json!({ "response": resource }))
}

/// `response.incomplete` carrying the terminal resource snapshot.
pub(super) fn response_incomplete(resource: &Value) -> StreamEvent {
    StreamEvent::new("response.incomplete", json!({ "response": resource }))
}

/// `response.failed` carrying the partial terminal resource snapshot.
pub(super) fn response_failed(resource: &Value) -> StreamEvent {
    StreamEvent::new("response.failed", json!({ "response": resource }))
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
    fn encode_injects_type_and_sequence_number() {
        let mut out = Vec::new();
        encode(output_text_delta("msg_1", 0, 0, "hi", &json!([])), 4, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("event: response.output_text.delta\ndata: "),
            "frame should start with the event line"
        );
        assert!(text.ends_with("\n\n"), "frame should end with a blank line");
        let data = text
            .strip_prefix("event: response.output_text.delta\ndata: ")
            .unwrap()
            .trim_end();
        let parsed: Value = serde_json::from_str(data).unwrap();
        assert_eq!(parsed["type"], "response.output_text.delta");
        assert_eq!(parsed["sequence_number"], 4);
        assert_eq!(parsed["delta"], "hi");
        assert_eq!(parsed["item_id"], "msg_1");
    }

    #[test]
    fn response_created_wraps_resource() {
        let mut out = Vec::new();
        encode(
            response_created(&json!({"id": "resp_1", "status": "in_progress"})),
            0,
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        let data = text.strip_prefix("event: response.created\ndata: ").unwrap().trim_end();
        let parsed: Value = serde_json::from_str(data).unwrap();
        assert_eq!(parsed["response"]["id"], "resp_1");
        assert_eq!(parsed["response"]["status"], "in_progress");
        assert_eq!(parsed["sequence_number"], 0);
    }
}
