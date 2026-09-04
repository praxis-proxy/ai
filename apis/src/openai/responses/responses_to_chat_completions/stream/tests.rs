// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Behavioral tests for the streaming Chat Completions to Responses converter.

use serde_json::{Value, json};

use super::*;
use crate::openai::translation::chat_completions::{ResponseContext, chat_response_to_response_resource};

/// Stable response id used across tests.
const RESPONSE_ID: &str = "resp_test";
/// Resource creation timestamp.
const CREATED_AT: u64 = 1_700_000_000;
/// Wall-clock time supplied to each callback.
const NOW: u64 = 1_700_000_005;

/// Generous limits that never trip in the happy-path tests.
fn wide_limits() -> StreamLimits {
    StreamLimits {
        max_sse_buffer_bytes: 1 << 20,
        max_stream_events: 100_000,
        max_tool_call_argument_bytes: 1 << 20,
        max_tool_calls: 128,
        stream_timeout_secs: 0,
        max_body_bytes: 1 << 24,
        max_stream_frames: 1_000_000,
        max_emitted_sse_frame_bytes: 1 << 24,
    }
}

/// Canonical Responses request body borrowed by snapshots.
fn request_body() -> Value {
    json!({"model": "gpt-4.1-mini", "input": "hi", "stream": true})
}

/// Build a converter with the given limits.
fn converter(limits: StreamLimits) -> StreamConverter {
    StreamConverter::new(RESPONSE_ID.to_owned(), CREATED_AT, limits)
}

/// Feed one chunk and append any emitted bytes to `raw`.
fn push(conv: &mut StreamConverter, body: &Value, chunk: &[u8], raw: &mut Vec<u8>) {
    let inputs = SnapshotInputs {
        request_body: body,
        now: NOW,
    };
    if let Some(bytes) = conv.push(chunk, &inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }
}

/// Finalize the stream and append any emitted bytes to `raw`.
fn finish(conv: &mut StreamConverter, body: &Value, raw: &mut Vec<u8>) {
    let inputs = SnapshotInputs {
        request_body: body,
        now: NOW,
    };
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }
}

/// Wrap Chat chunk JSON bodies as an SSE provider stream ending with `[DONE]`.
fn provider_stream(chunks: &[&str]) -> Vec<u8> {
    let mut raw = String::new();
    for chunk in chunks {
        raw.push_str("data: ");
        raw.push_str(chunk);
        raw.push_str("\n\n");
    }
    raw.push_str("data: [DONE]\n\n");
    raw.into_bytes()
}

/// Parse emitted Responses SSE bytes into `(event_type, payload)` pairs.
fn parse_events(raw: &[u8]) -> Vec<(String, Value)> {
    let text = std::str::from_utf8(raw).unwrap();
    text.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let mut event_type = String::new();
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event_type = rest.to_owned();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_owned();
                }
            }
            (event_type, serde_json::from_str::<Value>(&data).unwrap())
        })
        .collect()
}

/// Run a full provider stream at once and return parsed events.
fn run_stream(chunks: &[&str], limits: StreamLimits) -> Vec<(String, Value)> {
    let body = request_body();
    run_stream_with_body(chunks, &body, limits)
}

/// Run a complete provider stream against an explicit Responses request body.
fn run_stream_with_body(chunks: &[&str], body: &Value, limits: StreamLimits) -> Vec<(String, Value)> {
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    push(&mut conv, body, &provider_stream(chunks), &mut raw);
    finish(&mut conv, body, &mut raw);
    parse_events(&raw)
}

/// Return the event-type sequence for compact ordering assertions.
fn types(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(name, _)| name.as_str()).collect()
}

/// Build the reference finite Responses resource for a full Chat completion.
fn finite_resource(full_completion: &Value) -> Value {
    let body = request_body();
    let context =
        ResponseContext::from_responses_request(&body, RESPONSE_ID.to_owned(), CREATED_AT).with_completed_at(NOW);
    chat_response_to_response_resource(full_completion, &context).unwrap()
}

#[test]
fn text_path_emits_canonical_sequence() {
    let events = run_stream(
        &[
            r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#,
            r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":" world"}}]}"#,
            r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
        ],
        wide_limits(),
    );

    assert_eq!(
        types(&events),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
    );

    // Sequence numbers are monotonic from zero.
    for (index, (_, payload)) in events.iter().enumerate() {
        assert_eq!(payload["sequence_number"], index as u64, "sequence number at {index}");
    }

    // Stable message item id follows the finite convention.
    let added = &events[2].1;
    assert_eq!(added["item"]["id"], "msg_resp_test");
    assert_eq!(added["output_index"], 0);

    // Accumulated text and logprobs.
    let text_done = &events[6].1;
    assert_eq!(text_done["text"], "Hello world");
    assert_eq!(text_done["logprobs"], json!([]));

    // Terminal resource matches the finite translation exactly.
    let full = json!({
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    });
    assert_eq!(events[9].1["response"], finite_resource(&full));
}

#[test]
fn arbitrary_byte_splits_produce_identical_events() {
    let chunks = [
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"Hi"}}]}"#,
        r#"{"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    ];
    let stream = provider_stream(&chunks);

    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    for byte in &stream {
        push(&mut conv, &body, std::slice::from_ref(byte), &mut raw);
    }
    finish(&mut conv, &body, &mut raw);

    let split = parse_events(&raw);
    let whole = run_stream(&chunks, wide_limits());
    assert_eq!(split, whole);
}

#[test]
fn refusal_path_emits_refusal_events() {
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"refusal":"I cannot"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"refusal":" help"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        wide_limits(),
    );

    assert_eq!(
        types(&events),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.refusal.delta",
            "response.refusal.delta",
            "response.refusal.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
    );
    let refusal_done = events.iter().find(|(name, _)| name == "response.refusal.done").unwrap();
    assert_eq!(refusal_done.1["refusal"], "I cannot help");
}

#[test]
fn terminal_content_order_matches_streamed_content_index() {
    // Content parts claim their content_index in arrival order, so a refusal that
    // streams before text is announced at content_index 0 and text at 1. The
    // terminal snapshot must honor that streamed order instead of the finite
    // builder's fixed text-then-refusal layout, or the client sees the parts in a
    // different order than the incremental events reported.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"refusal":"no"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"yes"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        wide_limits(),
    );

    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.completed");
    let content = &payload["response"]["output"][0]["content"];
    assert_eq!(
        content[0]["type"], "refusal",
        "refusal streamed first must remain first in the terminal snapshot: {content}",
    );
    assert_eq!(
        content[1]["type"], "output_text",
        "text streamed second must remain second: {content}"
    );
}

#[test]
fn length_finish_emits_incomplete_terminal() {
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"partial"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#,
        ],
        wide_limits(),
    );

    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.incomplete");
    assert_eq!(payload["response"]["status"], "incomplete");
    assert_eq!(payload["response"]["incomplete_details"]["reason"], "max_output_tokens");
    // An incomplete response never completed, so it carries no completion time.
    assert_eq!(
        payload["response"]["completed_at"],
        Value::Null,
        "an incomplete response has no completion moment, so completed_at must be null: {payload}",
    );
    // The closed message item carries the incomplete status.
    let item_done = events
        .iter()
        .find(|(name, _)| name == "response.output_item.done")
        .unwrap();
    assert_eq!(item_done.1["item"]["status"], "incomplete");
}

#[test]
fn single_tool_call_emits_function_events() {
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    assert_eq!(
        types(&events),
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ],
    );
    let added = &events[2].1;
    assert_eq!(added["item"]["type"], "function_call");
    assert_eq!(added["item"]["id"], "fc_call_1");
    assert_eq!(added["item"]["name"], "get_weather");
    let args_done = &events[5].1;
    assert_eq!(args_done["arguments"], "{\"city\":\"Paris\"}");
    // The Responses schema requires `name` on function_call_arguments.done.
    assert_eq!(args_done["name"], "get_weather");

    let full = json!({
        "id": "c1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]
            },
            "finish_reason": "tool_calls"
        }]
    });
    assert_eq!(events.last().unwrap().1["response"], finite_resource(&full));
}

#[test]
fn multiple_tool_calls_preserve_output_order() {
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"first","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"second","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    let output = &events.last().unwrap().1["response"]["output"];
    let names: Vec<&str> = output
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["first", "second"]);
    let added: Vec<&Value> = events
        .iter()
        .filter(|(name, _)| name == "response.output_item.added")
        .map(|(_, p)| p)
        .collect();
    assert_eq!(added[0]["output_index"], 0);
    assert_eq!(added[1]["output_index"], 1);
}

#[test]
fn staggered_tool_calls_use_emit_order() {
    // Two tool calls whose identity and argument fragments interleave: call 0's
    // id and name arrive first, but call 1's arguments begin before call 0's.
    // Under dense allocation the output index is claimed when each item's
    // `output_item.added` is emitted (at the first argument fragment), so call 1
    // — whose arguments begin first — takes index 0, and call 0 takes index 1.
    // Canonical OpenResponses defines `output_index` as the index of the item
    // actually added, so allocating at emit keeps the streamed indices dense.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"first"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"second","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    let output = &events.last().unwrap().1["response"]["output"];
    let names: Vec<&str> = output
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["second", "first"],
        "tool order follows the order items are emitted (arguments begin), not first appearance",
    );
    // Each item's streamed output_index follows emit order: call 1's arguments
    // begin first, so its item is announced first and claims index 0.
    let mut announced: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (name, payload) in &events {
        if name == "response.output_item.added" {
            let id = payload["item"]["id"].as_str().unwrap().to_owned();
            announced.insert(id, payload["output_index"].as_u64().unwrap());
        }
    }
    assert_eq!(
        announced.get("fc_call_b"),
        Some(&0),
        "the call whose arguments begin first takes output index 0"
    );
    assert_eq!(
        announced.get("fc_call_a"),
        Some(&1),
        "the call whose arguments begin later takes output index 1"
    );
}

#[test]
fn split_identity_tool_calls_use_emit_order() {
    // Call 0 appears first carrying only its id; call 1 then arrives fully
    // identified (id and name) and emits its item before call 0's name arrives.
    // Under dense allocation the output index is claimed when each item's
    // `output_item.added` is emitted, so call 1 takes index 0 and call 0 takes
    // index 1. An id-only fragment that is only named later is ordered by when its
    // item is emitted, not by when it first appeared — the trade-off dense
    // allocation makes to keep the streamed indices hole-free. Realistic providers
    // send id and name together, so emit order equals appearance order for them.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a"}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"second","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"first","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.completed", "both calls are fully identified: {payload}");
    let names: Vec<&str> = payload["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["second", "first"],
        "tool order follows the order items are emitted; the fully-identified call emits first: {payload}",
    );

    let mut announced: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (name, payload) in &events {
        if name == "response.output_item.added" {
            let id = payload["item"]["id"].as_str().unwrap().to_owned();
            announced.insert(id, payload["output_index"].as_u64().unwrap());
        }
    }
    assert_eq!(
        announced.get("fc_call_b"),
        Some(&0),
        "the call whose item is emitted first takes output index 0",
    );
    assert_eq!(
        announced.get("fc_call_a"),
        Some(&1),
        "the id-only-first call takes output index 1 because its item is emitted later",
    );
}

#[test]
fn malformed_tool_after_partial_output_fails_with_empty_snapshot() {
    // An id-only tool call never gains a name, so it never emits an item and
    // never claims an output index: the later text message claims index 0 with no
    // hole (dense allocation at emit). Closing then fails closed with
    // ToolCallMissingIdentity during prevalidation, before the message item
    // streams any done event. The failed terminal must not manufacture completed
    // output: its snapshot carries an empty output array and a null completed_at,
    // since a failed response has no completion moment.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x"}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"partial answer"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    let (name, payload) = events.last().unwrap();
    assert_eq!(
        name, "response.failed",
        "an id-only tool call must fail closed at close"
    );

    // The unidentified tool call never emits an item, so it claims no index; the
    // message claims index 0 with no hole.
    let mut announced: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (name, payload) in &events {
        if name == "response.output_item.added" {
            let id = payload["item"]["id"].as_str().unwrap().to_owned();
            announced.insert(id, payload["output_index"].as_u64().unwrap());
        }
    }
    assert_eq!(
        announced.get("msg_resp_test"),
        Some(&0),
        "the message claims output index 0 because the unidentified tool call never emits an item",
    );
    assert!(
        !announced.contains_key("fc_call_x"),
        "an unidentified tool call is never announced",
    );

    // The failed snapshot manufactures no output and carries no completion time.
    assert_eq!(
        payload["response"]["status"], "failed",
        "the terminal resource status must be failed: {payload}",
    );
    assert_eq!(
        payload["response"]["output"],
        json!([]),
        "a failed terminal must not manufacture completed output: {payload}",
    );
    assert_eq!(
        payload["response"]["completed_at"],
        Value::Null,
        "a failed response has no completion moment, so completed_at must be null: {payload}",
    );
}

#[test]
fn partial_frame_after_terminal_fails_at_eof() {
    // A partial `data:` frame buffered after the successful terminal is
    // post-terminal data. It never completes, so `push` tolerates it, but EOF
    // must detect the buffered frame and fail closed rather than silently accept.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        &provider_stream(&[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
        ]),
        &mut raw,
    );
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    // A partial trailing frame (no blank line) buffers but yields no complete
    // frame, so the terminal-phase push tolerates it.
    conv.push(b"data: {\"id\":\"c1\"", &inputs)
        .expect("a partial post-terminal chunk yields no complete frame yet");
    let result = conv.finish(&inputs);
    assert!(
        result.is_err(),
        "a partial frame buffered after the terminal must fail closed at EOF",
    );
}

#[test]
fn large_retained_logprobs_trip_the_body_byte_limit() {
    // Each frame carries a single-byte text delta but a large logprobs payload
    // that is cloned into retained message state. Charging only the text delta
    // would let retained logprobs grow without bound past `max_body_bytes`.
    let limits = StreamLimits {
        max_body_bytes: 64,
        ..wide_limits()
    };
    let big_token = "a".repeat(120);
    let chunk = format!(
        r#"{{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{{"index":0,"delta":{{"content":"a"}},"logprobs":{{"content":[{{"token":"a","logprob":-0.1,"top_logprobs":[{{"token":"{big_token}","logprob":-1.0}}]}}]}}}}]}}"#,
    );
    // A trailing finish makes the only difference the logprobs accounting: the
    // one-byte text delta alone (charged) never trips the ceiling, so without
    // charging the retained logprobs this stream would complete successfully.
    let events = run_stream(
        &[
            chunk.as_str(),
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        limits,
    );

    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "retained logprobs exceeding max_body_bytes must fail closed"
    );
}

#[test]
fn tool_first_then_text_keeps_output_index_consistent() {
    // A tool call arrives before any assistant text, so it claims output index
    // 0 and the later message claims index 1. The terminal `output` array must
    // present each item at the position its streamed output_index announced.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{}"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"Here you go"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        ],
        wide_limits(),
    );

    let mut announced: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (name, payload) in &events {
        if name == "response.output_item.added" {
            let id = payload["item"]["id"].as_str().unwrap().to_owned();
            announced.insert(id, payload["output_index"].as_u64().unwrap());
        }
    }
    assert_eq!(announced.get("fc_call_1"), Some(&0), "tool call streamed at index 0");
    assert_eq!(announced.get("msg_resp_test"), Some(&1), "message streamed at index 1");

    let output = events.last().unwrap().1["response"]["output"]
        .as_array()
        .unwrap()
        .clone();
    for (position, item) in output.iter().enumerate() {
        let id = item["id"].as_str().unwrap();
        assert_eq!(
            announced.get(id).copied(),
            Some(position as u64),
            "terminal output item {id} at position {position} contradicts its streamed output_index",
        );
    }
}

#[test]
fn clean_eof_without_done_emits_terminal() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.completed");
}

#[test]
fn incomplete_frame_at_eof_fails() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"partial\"",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn malformed_json_emits_failed_without_leaking_bytes() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(&mut conv, &body, b"data: {not valid json}\n\n", &mut raw);
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
    // The raw provider bytes never appear in the emitted stream.
    assert!(
        !std::str::from_utf8(&raw).unwrap().contains("not valid json"),
        "the malformed provider bytes must never appear in the emitted stream",
    );
}

#[test]
fn parse_failure_after_partial_output_emits_empty_failed_snapshot() {
    // A message item streams (announced but never completed), then a malformed
    // chunk fails the stream mid-flight. The failed snapshot must not manufacture
    // the partially accumulated message as completed output — no output_item.done
    // was ever emitted for it — so its output array is empty and completed_at null.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial answer\"}}]}\n\n",
        &mut raw,
    );
    push(&mut conv, &body, b"data: {not valid json}\n\n", &mut raw);
    let events = parse_events(&raw);

    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "a mid-stream parse failure must fail closed");
    // The partial message was announced but never completed.
    assert!(
        events
            .iter()
            .any(|(name, p)| name == "response.output_item.added" && p["item"]["type"] == "message"),
        "the partial message item was announced mid-stream: {events:?}",
    );
    assert_eq!(
        payload["response"]["output"],
        json!([]),
        "a failed terminal must not manufacture the partial message as completed output: {payload}",
    );
    assert_eq!(
        payload["response"]["completed_at"],
        Value::Null,
        "a failed response has no completion moment, so completed_at must be null: {payload}",
    );
}

#[test]
fn multiple_choices_emit_failed() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{}},{\"index\":1,\"delta\":{}}]}\n\n",
        &mut raw,
    );
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn missing_finish_reason_emits_failed() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn name_after_arguments_emits_failed() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"foo\",\"arguments\":\"{}\"}}]}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"bar\"}}]}}]}\n\n",
        &mut raw,
    );
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn late_call_id_fragment_after_arguments_emits_failed() {
    // The item id is frozen as `fc_{call_id}` when arguments begin and the item
    // is announced. A later id fragment would change the terminal id after the
    // stream already announced a different one, so it must fail closed rather
    // than emit inconsistent ids across the added/done/terminal events.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_\",\"function\":{\"name\":\"foo\",\"arguments\":\"{}\"}}]}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"123\"}]}}]}\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "an id fragment after the item was announced must fail closed",
    );
}

#[test]
fn tool_call_without_identity_fails_closed() {
    // A tool call that streamed a fragment but never received both an id and a
    // name is incomplete. Dropping it would silently turn an intended tool call
    // into an empty success, so it must fail closed instead.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\"}]}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a tool call missing id or name must fail closed, not be silently dropped",
    );
}

#[test]
fn tool_call_fragment_without_index_fails_closed() {
    // The Chat Completions streaming format correlates tool-call fragments by
    // their `index`. Two distinct calls that both omit `index` would otherwise
    // default to index 0 and be silently merged into one corrupted call, so a
    // fragment missing `index` must fail closed instead.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_a\",\"function\":{\"name\":\"first\"}}]}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_b\",\"function\":{\"name\":\"second\"}}]}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a tool-call fragment without an index must fail closed, not merge distinct calls",
    );
}

#[test]
fn legacy_function_call_delta_fails_closed() {
    // The deprecated Chat Completions `delta.function_call` carries a real tool
    // call that this converter does not translate. Silently ignoring it would
    // complete the stream with empty output and drop the call, so a delta that
    // carries `function_call` must fail closed instead.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"function_call\"}]}\n\n",
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a legacy function_call delta must fail closed, not complete with empty output",
    );
}

#[test]
fn finish_only_function_call_fails_closed() {
    // A `function_call` finish reason signals legacy function calling even when no
    // `delta.function_call` payload was streamed. Accepting it completes the
    // stream with empty output and drops the call, so it must fail closed.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}"#,
        ],
        wide_limits(),
    );
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a finish-only function_call stream must fail closed, not complete with empty output",
    );
}

#[test]
fn fragmented_first_frame_emits_lifecycle_immediately() {
    // The opening lifecycle events must be emitted on the first non-empty
    // callback, even when that callback carries only part of the first SSE frame.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    // A partial frame: no terminating blank line yet, so no frame parses.
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,",
        &mut raw,
    );
    assert_eq!(
        types(&parse_events(&raw)),
        vec!["response.created", "response.in_progress"],
        "lifecycle must be emitted on the first non-empty callback",
    );
}

#[test]
fn trailing_frame_after_terminal_in_later_callback_fails() {
    // A provider frame delivered in a callback after the completed terminal is a
    // protocol violation. Silently dropping it could mask a misbehaving upstream
    // or response splitting, so the converter must fail the transport instead.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        &provider_stream(&[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
        ]),
        &mut raw,
    );
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    let result = conv.push(
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"more\"}}]}\n\n",
        &inputs,
    );
    assert!(
        result.is_err(),
        "a provider frame after the completed terminal must fail the stream"
    );
}

#[test]
fn trailing_frame_after_terminal_in_same_callback_fails() {
    // A trailing content frame packed into the same callback as `[DONE]` must not
    // be silently dropped after the terminal is emitted; it must fail the stream.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    let mut chunk = provider_stream(&[
        r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
    ]);
    chunk.extend_from_slice(
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"more\"}}]}\n\n",
    );
    let result = conv.push(&chunk, &inputs);
    assert!(
        result.is_err(),
        "a trailing frame after [DONE] in the same callback must fail the stream"
    );
}

#[test]
fn empty_callback_after_terminal_is_tolerated() {
    // A benign empty trailing callback (e.g. a connection drain read) carries no
    // provider frame, so it must be tolerated rather than failing the stream.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        &provider_stream(&[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
        ]),
        &mut raw,
    );
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    let result = conv
        .push(b"", &inputs)
        .expect("an empty trailing callback must be tolerated");
    assert!(result.is_none(), "an empty trailing callback must emit nothing");
}

#[test]
fn frame_limit_emits_failed() {
    // Frames with an empty `choices` array decode cleanly but emit no events and
    // charge no semantic bytes, so they slip past the event and byte ceilings.
    // Only a decoded-frame cap bounds an upstream that streams unbounded no-op
    // frames; exceeding it must fail closed.
    let limits = StreamLimits {
        max_stream_frames: 3,
        ..wide_limits()
    };
    let body = request_body();
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let mut stream = String::new();
    for _ in 0..6 {
        stream.push_str("data: ");
        stream.push_str(r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[]}"#);
        stream.push_str("\n\n");
    }
    push(&mut conv, &body, stream.as_bytes(), &mut raw);
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "an unbounded run of no-op frames must trip the frame limit and fail closed",
    );
}

#[test]
fn event_limit_emits_failed() {
    let limits = StreamLimits {
        max_stream_events: 1,
        ..wide_limits()
    };
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        limits,
    );
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn tool_call_limit_emits_failed() {
    let limits = StreamLimits {
        max_tool_calls: 1,
        ..wide_limits()
    };
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"x"}}]}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"y"}}]}}]}"#,
        ],
        limits,
    );
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn tool_argument_limit_emits_failed() {
    let limits = StreamLimits {
        max_tool_call_argument_bytes: 4,
        ..wide_limits()
    };
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"x","arguments":"{\"big\":\"value\"}"}}]}}]}"#,
        ],
        limits,
    );
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn body_byte_limit_emits_failed() {
    let limits = StreamLimits {
        max_body_bytes: 3,
        ..wide_limits()
    };
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"way too long"}}]}"#,
        ],
        limits,
    );
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn inconsistent_metadata_emits_failed() {
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n",
        &mut raw,
    );
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn timeout_emits_failed() {
    let limits = StreamLimits {
        stream_timeout_secs: 10,
        ..wide_limits()
    };
    let body = request_body();
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    // First push establishes the start time.
    let first = SnapshotInputs {
        request_body: &body,
        now: 100,
    };
    if let Some(bytes) = conv
        .push(
            b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n",
            &first,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    // A later push beyond the timeout window fails.
    let late = SnapshotInputs {
        request_body: &body,
        now: 200,
    };
    if let Some(bytes) = conv
        .push(
            b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n",
            &late,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    let events = parse_events(&raw);
    assert_eq!(events.last().unwrap().0, "response.failed");
}

#[test]
fn unknown_finish_reason_emits_failed() {
    // The finite translation only recognizes the documented finish reasons. A
    // reason the converter does not understand must fail closed rather than be
    // silently mapped to a completed terminal.
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"banana"}]}"#,
        ],
        wide_limits(),
    );
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "an unrecognized finish reason must fail closed",
    );
}

#[test]
fn oversized_terminal_resource_fails_closed() {
    // Each text delta is tiny and passes the incremental byte charge, but the
    // fully serialized terminal resource (envelope plus request echo) exceeds
    // the small body ceiling. The finite path enforces this same limit on the
    // serialized response, so the streaming terminal must too.
    let limits = StreamLimits {
        max_body_bytes: 150,
        ..wide_limits()
    };
    let events = run_stream(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        limits,
    );
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a serialized terminal exceeding max_body_bytes must fail closed",
    );
}

#[test]
fn oversized_terminal_failed_snapshot_is_bounded() {
    // The accumulated text passes the incremental charge but makes the serialized
    // terminal exceed the ceiling, so `emit_terminal` trips `ByteLimit` and the
    // failure path runs. The emitted `response.failed` must NOT re-serialize the
    // oversized accumulated snapshot: it has to stay within `max_body_bytes` by
    // dropping the unbounded output rather than merely swapping the event type.
    let max_body_bytes = 2048;
    let limits = StreamLimits {
        max_body_bytes,
        ..wide_limits()
    };
    let big_text = "a".repeat(2000);
    let content_chunk = format!(
        r#"{{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{{"index":0,"delta":{{"content":"{big_text}"}}}}]}}"#,
    );
    let events = run_stream(
        &[
            content_chunk.as_str(),
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ],
        limits,
    );

    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "an oversized terminal must fail closed");
    let serialized = serde_json::to_vec(&payload["response"]).unwrap();
    assert!(
        serialized.len() <= max_body_bytes,
        "the response.failed terminal must respect max_body_bytes (was {} bytes)",
        serialized.len(),
    );
    let output = payload["response"]["output"].as_array();
    assert!(
        output.is_none_or(Vec::is_empty),
        "the failed terminal must not embed the oversized accumulated output",
    );
}

#[test]
fn failed_terminal_does_not_echo_unbounded_request_fields() {
    // The failure fallback must be constant-bounded: a large request-controlled
    // field (here `metadata`) must not be copied into the `response.failed`
    // terminal, or the terminal would exceed `max_body_bytes` and leak the very
    // request data the ceiling exists to bound.
    let max_body_bytes = 1000;
    let limits = StreamLimits {
        max_body_bytes,
        ..wide_limits()
    };
    let big_metadata = "m".repeat(4000);
    let body = json!({
        "model": "gpt-4.1-mini",
        "input": "hi",
        "stream": true,
        "metadata": {"note": big_metadata},
    });
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    // Tiny output passes the incremental charge, but echoing the large request
    // metadata into the terminal resource trips the byte limit.
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "an oversized terminal must fail closed");
    let serialized = serde_json::to_vec(&payload["response"]).unwrap();
    assert!(
        serialized.len() <= max_body_bytes,
        "the failed terminal must not echo unbounded request fields (was {} bytes)",
        serialized.len(),
    );
    assert!(
        !std::str::from_utf8(&serialized).unwrap().contains(&big_metadata),
        "the failed terminal leaked the large request metadata",
    );
}

#[test]
fn failed_terminal_fallback_is_schema_complete() {
    // The constant-bounded failure fallback must still be a valid Response
    // object: it has to carry every schema-required field, not just id/status/
    // error/output. A large request-controlled `metadata` forces the fallback.
    let max_body_bytes = 1000;
    let limits = StreamLimits {
        max_body_bytes,
        ..wide_limits()
    };
    let body = json!({
        "model": "gpt-4.1-mini",
        "input": "hi",
        "stream": true,
        "metadata": {"note": "m".repeat(4000)},
    });
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed");
    let resource = &payload["response"];
    assert_eq!(resource["status"], "failed");
    // Every field the Responses `Response` schema marks required must be present.
    for field in [
        "id",
        "object",
        "created_at",
        "error",
        "incomplete_details",
        "instructions",
        "model",
        "tools",
        "output",
        "parallel_tool_calls",
        "metadata",
        "tool_choice",
        "temperature",
        "top_p",
    ] {
        assert!(
            resource.get(field).is_some(),
            "the failed fallback must include required field `{field}`: {resource}",
        );
    }
}

#[test]
fn failed_terminal_fallback_is_bounded_by_request_fields() {
    // The constant-bounded fallback exists to cap the failed terminal's size when
    // the full snapshot echoes unbounded request-controlled fields. It must not
    // itself echo any of them: a huge `model` and `previous_response_id` force the
    // fallback, and the emitted terminal must stay within the response ceiling
    // rather than re-serializing those request values.
    let max_body_bytes = 1000;
    let limits = StreamLimits {
        max_body_bytes,
        ..wide_limits()
    };
    let body = json!({
        "model": "m".repeat(8000),
        "input": "hi",
        "stream": true,
        "previous_response_id": "r".repeat(8000),
    });
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed");
    let resource = &payload["response"];
    let serialized = serde_json::to_vec(resource).expect("resource should serialize");
    assert!(
        serialized.len() <= max_body_bytes,
        "the failed fallback must stay within the response ceiling; got {} bytes",
        serialized.len(),
    );
    // The unbounded request-controlled values must not be echoed at all.
    assert_ne!(resource["model"].as_str(), Some("m".repeat(8000).as_str()));
    assert!(
        resource["previous_response_id"].is_null(),
        "previous_response_id must be neutralized: {resource}",
    );
}

#[test]
fn bounded_failure_has_null_completed_at() {
    // A failed response never completed, so the bounded fallback must carry a null
    // completed_at, matching the schema where only a completed response is stamped
    // with a completion time.
    let max_body_bytes = 1000;
    let limits = StreamLimits {
        max_body_bytes,
        ..wide_limits()
    };
    let body = json!({
        "model": "m".repeat(8000),
        "input": "hi",
        "stream": true,
    });
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed");
    let resource = &payload["response"];
    assert_eq!(resource["created_at"], CREATED_AT);
    assert_eq!(
        resource["completed_at"],
        Value::Null,
        "a failed response has no completion moment, so completed_at must be null: {resource}",
    );
}

#[test]
fn finish_after_timeout_emits_failed() {
    // A provider-done stream whose terminal is only emitted at EOF must still
    // honor the wall-clock limit: an EOF callback arriving past the timeout has
    // to fail closed rather than bypass the check on the empty-finish path.
    let limits = StreamLimits {
        stream_timeout_secs: 2,
        ..wide_limits()
    };
    let body = request_body();
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    // Establish the start time and reach provider-done without emitting a
    // terminal (no `[DONE]`), so the terminal is deferred to `finish`.
    let started = SnapshotInputs {
        request_body: &body,
        now: 100,
    };
    if let Some(bytes) = conv
        .push(
            b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
            &started,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    // The EOF callback arrives well past the timeout window.
    let elapsed = SnapshotInputs {
        request_body: &body,
        now: 200,
    };
    if let Some(bytes) = conv.finish(&elapsed).unwrap() {
        raw.extend_from_slice(&bytes);
    }
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "an EOF past the timeout must fail closed",
    );
    // The message item completed (finish_reason arrived) before the timeout, but
    // its `response.output_item.done` must not be on the wire: closeout is
    // deferred to the terminal, so a post-finish timeout fails closed with a
    // coherent empty snapshot rather than a `response.failed` contradicting an
    // already-completed item.
    assert!(
        !types(&events).iter().any(|name| name.ends_with(".done")),
        "no `*.done` event may precede a post-finish failure: {:?}",
        types(&events),
    );
    let failed = &events.last().unwrap().1;
    assert!(
        failed["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "a failed terminal must report empty output: {failed}",
    );
}

#[test]
fn changed_service_tier_emits_failed() {
    // `service_tier` is part of the consistency invariant: like a changed id or
    // model, a conflicting value across chunks means the upstream stream is not
    // describing a single completion and must fail closed.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"service_tier\":\"default\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"service_tier\":\"scale\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        &mut raw,
    );
    let events = parse_events(&raw);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a changed service_tier must fail closed like a changed id or model",
    );
}

#[test]
fn closeout_budget_is_atomic_across_items() {
    // A message and a tool call are both open when the finish reason arrives, so
    // the closeout emits several `*.done` events plus the terminal. Without a
    // preflight, the message's `output_item.done` could reach the client and then
    // the tool close could trip the event cap, leaving a completed item behind a
    // failed terminal that reports empty output. `close_open_items` preflights the
    // whole closeout against the budget, so a budget too small to hold it fails
    // *before* emitting any closeout event.
    let chunks = [
        r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{}"}}]}}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    ];

    // Seven streaming events precede closeout: created, in_progress, the message's
    // item.added/content_part.added/output_text.delta, and the tool call's
    // item.added/arguments.delta. Closeout needs five more `*.done` events, and
    // the uncapped terminal reserves one slot, so twelve is one short of the
    // thirteen a full closeout-plus-terminal requires.
    let too_small = StreamLimits {
        max_stream_events: 12,
        ..wide_limits()
    };
    let events = run_stream(&chunks, too_small);
    assert_eq!(
        events.last().unwrap().0,
        "response.failed",
        "a budget too small for the full closeout must fail closed",
    );
    assert!(
        !types(&events).iter().any(|name| name.ends_with(".done")),
        "no closeout `*.done` event may be emitted when the closeout cannot complete atomically: {:?}",
        types(&events),
    );

    // One more slot lets the entire closeout and the terminal fit.
    let exact = StreamLimits {
        max_stream_events: 13,
        ..wide_limits()
    };
    let events = run_stream(&chunks, exact);
    assert_eq!(
        events.last().unwrap().0,
        "response.completed",
        "a budget that holds the whole closeout plus terminal must complete",
    );
    let completed_items = types(&events)
        .iter()
        .filter(|name| **name == "response.output_item.done")
        .count();
    assert_eq!(
        completed_items,
        2,
        "both the message and tool-call items must be completed: {:?}",
        types(&events),
    );
}

#[test]
fn first_chunk_missing_metadata_fails_closed() {
    // The Chat Completions schema requires `id`, `model`, and `object` on every
    // `chat.completion.chunk`. A first chunk missing any of them signals a
    // malformed or non-Chat stream; translating it would produce a successful
    // Responses stream with a null model, so the converter fails closed with an
    // empty-output terminal instead.
    for (label, chunk) in [
        (
            "missing id",
            r#"{"object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ),
        (
            "missing model",
            r#"{"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ),
        (
            "missing object",
            r#"{"id":"c1","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ),
    ] {
        let events = run_stream(&[chunk], wide_limits());
        let (name, payload) = events.last().unwrap();
        assert_eq!(name, "response.failed", "a first chunk {label} must fail closed");
        let output = payload["response"]["output"].as_array();
        assert!(
            output.is_none_or(Vec::is_empty),
            "the failed terminal for a first chunk {label} must carry empty output: {payload}",
        );
    }
}

#[test]
fn large_lifecycle_frame_needs_accumulator_headroom() {
    use crate::openai::sse::{SseFrameParser, SseParseError, SseParserConfig};

    // A lifecycle event echoes request-controlled fields (here `instructions`), so
    // the single `response.created` frame the converter emits can dwarf the
    // downstream accumulator's *default* reassembly buffer. At the converter's own
    // *default* per-frame ceiling this frame fails closed and never reaches the
    // client; this test raises `max_emitted_sse_frame_bytes` to permit it, matching
    // the example config's wide ceiling. The client then receives that frame, but
    // `openai_stream_events` would reject it and silently skip persistence unless
    // its `max_buffer_bytes` is raised in lockstep — exactly what the example does.
    let big_instructions = "a".repeat(12 * 1024 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": big_instructions,
        "input": "hi",
        "stream": true,
    });
    let limits = StreamLimits {
        max_body_bytes: 64 * 1024 * 1024,
        max_emitted_sse_frame_bytes: 64 * 1024 * 1024,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    // Isolate the largest emitted SSE frame, restoring its blank-line terminator.
    // The lifecycle data line is a single JSON object, so it never contains an
    // internal blank line to split on.
    let text = std::str::from_utf8(&raw).unwrap();
    let created_frame = text
        .split("\n\n")
        .find(|frame| frame.starts_with("event: response.created"))
        .map(|frame| format!("{frame}\n\n"))
        .expect("stream should contain a response.created frame");

    let default_buffer = SseParserConfig::default().max_buffer_bytes;
    assert!(
        created_frame.len() > default_buffer,
        "the lifecycle frame ({} bytes) should exceed the accumulator's default buffer ({} bytes)",
        created_frame.len(),
        default_buffer,
    );

    // At the default buffer the accumulator rejects the frame — the persistence
    // gap the finding describes.
    let mut tight = SseFrameParser::new(default_buffer);
    assert!(
        matches!(
            tight.parse_chunk(created_frame.as_bytes()),
            Err(SseParseError::BufferOverflow { .. })
        ),
        "the default accumulator buffer must reject a frame larger than itself",
    );

    // With headroom matching the example's max_buffer_bytes the frame parses.
    let mut roomy = SseFrameParser::new(64 * 1024 * 1024);
    let frames = roomy
        .parse_chunk(created_frame.as_bytes())
        .expect("a buffer sized above the frame must accept it");
    assert_eq!(
        frames.len(),
        1,
        "the headroom buffer must reassemble exactly the one lifecycle frame",
    );
}

#[test]
fn lifecycle_resource_over_body_limit_fails_before_lifecycle_emission() {
    let marker = "request-controlled-metadata".repeat(100);
    let body = json!({
        "model": "gpt-4.1-mini",
        "input": "hi",
        "metadata": {"marker": marker},
        "stream": true,
    });
    let limits = StreamLimits {
        max_body_bytes: 1000,
        max_emitted_sse_frame_bytes: 16 * 1024,
        ..wide_limits()
    };

    let events = run_stream_with_body(
        &[
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ],
        &body,
        limits,
    );

    assert_eq!(
        events.len(),
        1,
        "an oversized lifecycle must emit only the failure terminal"
    );
    assert_eq!(events[0].0, "response.failed");
    assert_eq!(events[0].1["sequence_number"], 0);
    assert!(
        !events[0].1.to_string().contains("request-controlled-metadata"),
        "the bounded failure must not echo request-controlled lifecycle fields",
    );
}

#[test]
fn lifecycle_frame_overflow_rolls_back_the_pair() {
    let body = json!({
        "model": "gpt-4.1-mini",
        "input": "hi",
        "instructions": "x".repeat(4096),
        "stream": true,
    });
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    let mut probe = converter(wide_limits());
    let probe_raw = probe
        .push(b"data: partial", &inputs)
        .unwrap()
        .expect("the first non-empty callback must emit the lifecycle pair");
    let frame_lengths: Vec<usize> = std::str::from_utf8(&probe_raw)
        .unwrap()
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| frame.len() + 2)
        .collect();
    assert_eq!(
        frame_lengths.len(),
        2,
        "the probe must contain exactly the lifecycle pair"
    );
    assert!(
        frame_lengths[1] > frame_lengths[0],
        "response.in_progress framing must be larger than response.created for this boundary test",
    );

    let limits = StreamLimits {
        max_emitted_sse_frame_bytes: frame_lengths[0],
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let raw = conv
        .push(b"data: partial", &inputs)
        .unwrap()
        .expect("the oversized lifecycle must be replaced by response.failed");
    let events = parse_events(&raw);

    assert_eq!(types(&events), ["response.failed"]);
    assert_eq!(events[0].1["sequence_number"], 0);
}

#[test]
fn data_after_finish_before_terminal_fails_with_empty_snapshot() {
    // The finish reason arrives (opening and completing nothing on the wire),
    // then the provider sends another content chunk before `[DONE]`. That is data
    // after finish and must fail closed. Because closeout is deferred to the
    // terminal, the message item the first chunk opened was never committed, so
    // the failure carries a coherent empty snapshot with no `*.done` preceding it.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"more\"}}]}\n\n",
        &mut raw,
    );
    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "data after the finish reason must fail closed");
    assert!(
        !types(&events).iter().any(|event| event.ends_with(".done")),
        "no `*.done` event may precede a post-finish failure: {:?}",
        types(&events),
    );
    assert!(
        payload["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "the failed terminal must report empty output: {payload}",
    );
    assert_eq!(
        payload["response"]["completed_at"],
        Value::Null,
        "a failed response has no completion moment: {payload}",
    );
}

#[test]
fn malformed_chunk_after_finish_fails_with_empty_snapshot() {
    // The finish reason arrives, then a malformed (non-JSON) chunk follows before
    // `[DONE]`. The converter fails closed with an empty snapshot, and the raw
    // provider bytes of the malformed chunk never leak into the emitted stream.
    let body = request_body();
    let mut conv = converter(wide_limits());
    let mut raw = Vec::new();
    push(
        &mut conv,
        &body,
        b"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        &mut raw,
    );
    push(
        &mut conv,
        &body,
        b"data: {\"leaked_marker\": not-valid-json}\n\n",
        &mut raw,
    );
    assert!(
        !raw.windows(b"leaked_marker".len())
            .any(|window| window == b"leaked_marker"),
        "raw provider bytes of a malformed chunk must never leak into the emitted stream",
    );
    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(
        name, "response.failed",
        "a malformed chunk after finish must fail closed"
    );
    assert!(
        !types(&events).iter().any(|event| event.ends_with(".done")),
        "no `*.done` event may precede a post-finish failure: {:?}",
        types(&events),
    );
    assert!(
        payload["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "the failed terminal must report empty output: {payload}",
    );
}

#[test]
fn terminal_over_byte_limit_fails_without_committed_items() {
    // The terminal resource echoes request-controlled `instructions`, so a large
    // request can push the terminal past `max_body_bytes` even though the streamed
    // content is tiny. The byte-limit check runs *before* closeout, so the trip
    // fails closed with no `response.output_item.done` already on the wire — this
    // guards the ordering: were closeout emitted first, a completed item would
    // precede the `response.failed`.
    let big_instructions = "a".repeat(2 * 1024 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": big_instructions,
        "input": "hi",
        "stream": true,
    });
    // The 1 MiB ceiling holds the tiny streamed content ("hi") but not the ~2 MiB
    // terminal resource that re-echoes the instructions.
    let limits = StreamLimits {
        max_body_bytes: 1024 * 1024,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }
    let events = parse_events(&raw);
    let (name, payload) = events.last().unwrap();
    assert_eq!(
        name, "response.failed",
        "a terminal exceeding max_body_bytes must fail closed",
    );
    assert!(
        !types(&events).iter().any(|event| event.ends_with(".done")),
        "a byte-limit trip at the terminal must emit no `*.done` before the failure: {:?}",
        types(&events),
    );
    assert!(
        payload["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "the failed terminal must report empty output: {payload}",
    );
}

#[test]
fn oversized_lifecycle_frame_fails_closed_with_minimal_snapshot() {
    // The in-progress lifecycle snapshot echoes request-controlled `instructions`,
    // so a 12 MiB request makes the `response.created` frame exceed the default
    // 8 MiB per-frame ceiling. Nothing partial is emitted; the converter fails
    // closed. The `response.failed` snapshot would *also* re-echo the 12 MiB
    // instructions (its own frame exceeding the ceiling), so the converter falls
    // back to the constant-bounded minimal resource that echoes no request data.
    let big_instructions = "a".repeat(12 * 1024 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": big_instructions,
        "input": "hi",
        "stream": true,
    });
    // Default 8 MiB per-frame ceiling; a generous body ceiling (from wide_limits)
    // so the *frame-size* path (not the byte-size fallback) minimizes the failure.
    let limits = StreamLimits {
        max_emitted_sse_frame_bytes: 8 * 1024 * 1024,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let events = parse_events(&raw);
    // The oversized lifecycle frames are rolled back: neither reaches the client.
    assert!(
        !types(&events)
            .iter()
            .any(|event| *event == "response.created" || *event == "response.in_progress"),
        "an oversized lifecycle frame must not reach the client: {:?}",
        types(&events),
    );
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "an oversized lifecycle must fail closed");
    assert!(
        !types(&events).iter().any(|event| event.ends_with(".done")),
        "no `*.done` event may precede the failure: {:?}",
        types(&events),
    );
    // The minimal fallback neutralizes every request-controlled field, so the
    // 12 MiB instructions are never re-echoed in the failure snapshot.
    assert_eq!(
        payload["response"]["instructions"],
        Value::Null,
        "the minimal failed snapshot must not echo the request instructions",
    );
    assert_eq!(
        payload["response"]["model"], "",
        "the minimal failed snapshot must neutralize the model",
    );
    assert!(
        payload["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "the failed terminal must report empty output",
    );
    assert_eq!(
        payload["response"]["error"]["code"], "server_error",
        "the failed snapshot must carry a schema-complete error object",
    );
    // Both the rolled-back lifecycle frame and the rolled-back full-size failure
    // frame consumed no sequence number, so the surviving minimal terminal is the
    // stream's first and only event: sequence_number 0. This locks in the
    // rollback contract that a per-frame size trip never advances the counter.
    assert_eq!(
        payload["sequence_number"], 0,
        "the surviving minimal response.failed must carry sequence_number 0",
    );
}

#[test]
fn minimal_failed_frame_persists_under_default_accumulator() {
    use crate::openai::sse::{SseFrameParser, SseParseError, SseParserConfig};

    // The failure frame the converter emits when a lifecycle frame overflows must
    // itself fit the downstream accumulator's *default* reassembly buffer, so the
    // stream the client received is also persisted rather than dropped. Reproduce
    // the oversized-lifecycle failure at the default per-frame ceiling, then feed
    // the emitted bytes through a frame parser sized at the accumulator default.
    let big_instructions = "a".repeat(12 * 1024 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": big_instructions,
        "input": "hi",
        "stream": true,
    });
    let limits = StreamLimits {
        max_emitted_sse_frame_bytes: 8 * 1024 * 1024,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    let default_buffer = SseParserConfig::default().max_buffer_bytes;
    assert!(
        raw.len() < default_buffer,
        "the whole emitted failure stream ({} bytes) must fit the accumulator default buffer ({} bytes)",
        raw.len(),
        default_buffer,
    );

    // The default accumulator buffer reassembles the failure frame without
    // overflowing — the persistence guarantee the per-frame ceiling exists for.
    let mut parser = SseFrameParser::new(default_buffer);
    let frames = match parser.parse_chunk(&raw) {
        Ok(frames) => frames,
        Err(SseParseError::BufferOverflow { .. }) => {
            panic!("the minimal failure frame must not overflow the accumulator default buffer")
        },
        Err(other) => panic!("unexpected parse error: {other:?}"),
    };
    assert!(
        frames.iter().any(|frame| {
            std::str::from_utf8(frame.data.as_slice())
                .ok()
                .and_then(|data| serde_json::from_str::<Value>(data).ok())
                .is_some_and(|value| value["type"] == "response.failed")
        }),
        "the accumulator must reassemble the response.failed frame",
    );

    // The surviving terminal is the stream's only event, so it must carry
    // sequence_number 0 — every rolled-back frame consumed no number.
    let events = parse_events(&raw);
    let (name, payload) = events.last().expect("the stream must emit a terminal event");
    assert_eq!(name, "response.failed", "the terminal event must be response.failed");
    assert_eq!(
        payload["sequence_number"], 0,
        "the surviving minimal response.failed must carry sequence_number 0",
    );
}

#[test]
fn mid_closeout_frame_overflow_rolls_back_committed_closeout() {
    // A small message closes cleanly — its `output_text.done`, `content_part.done`,
    // and `output_item.done` frames each fit the per-frame ceiling — but the
    // terminal `response.completed` frame, which additionally echoes the request
    // `instructions` alongside the accumulated output, exceeds it. The overflow
    // trips *after* `close_open_items` has already written those `*.done` frames,
    // exercising the emit_terminal rollback that truncates the committed closeout
    // bytes and restores the sequence counter. Without it, the `*.done` frames and
    // an advanced sequence number would leak ahead of the `response.failed`
    // `handle_failure` then emits, violating F1.
    //
    // Sizing (ceiling 250 KiB): the 200 KiB `instructions` inflate the lifecycle
    // and terminal frames but not the message-item closeout frames; the 150 KiB
    // message text inflates the closeout frames yet each stays under the ceiling.
    // Only the terminal (instructions + output text ≈ 350 KiB) overflows, and only
    // once the closeout frames are already on the buffer. The generous body
    // ceiling from `wide_limits` keeps the *frame*-size path — not the byte-size
    // fallback — the trigger.
    let instructions = "a".repeat(200 * 1024);
    let text = "b".repeat(150 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": instructions,
        "input": "hi",
        "stream": true,
    });
    let ceiling = 250 * 1024;
    let limits = StreamLimits {
        max_emitted_sse_frame_bytes: ceiling,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let content_chunk = format!(
        r#"{{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{{"index":0,"delta":{{"content":"{text}"}}}}]}}"#,
    );
    push(
        &mut conv,
        &body,
        &provider_stream(&[
            &content_chunk,
            r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ]),
        &mut raw,
    );
    finish(&mut conv, &body, &mut raw);

    let events = parse_events(&raw);

    // The lifecycle succeeded and was not rolled back: the client saw the stream
    // open normally before the closeout overflow.
    assert!(
        types(&events).contains(&"response.created") && types(&events).contains(&"response.in_progress"),
        "the lifecycle must precede the mid-closeout failure: {:?}",
        types(&events),
    );
    // Content actually streamed, so `close_open_items` had committed `*.done`
    // frames to roll back — this is the mid-closeout case, not an early failure.
    assert!(
        types(&events).contains(&"response.output_text.delta"),
        "the message text must stream before the closeout: {:?}",
        types(&events),
    );
    // The committed closeout frames were truncated off the buffer: no `*.done`
    // event reaches the client ahead of the terminal.
    assert!(
        !types(&events).iter().any(|event| event.ends_with(".done")),
        "a mid-closeout size trip must roll every `*.done` frame back: {:?}",
        types(&events),
    );
    // The stream still terminates with a fail-closed `response.failed`.
    let (name, payload) = events.last().unwrap();
    assert_eq!(name, "response.failed", "a mid-closeout overflow must fail closed");
    assert!(
        payload["response"]["output"].as_array().is_none_or(Vec::is_empty),
        "the failed terminal must report empty output",
    );
    // Sequence numbers stay contiguous from zero: the rolled-back closeout frames
    // consumed no numbers, so the surviving `response.failed` follows the last
    // delta without a gap. A dropped `self.emit` restore would leave one here.
    for (index, (_, event)) in events.iter().enumerate() {
        assert_eq!(
            event["sequence_number"], index as u64,
            "sequence number at {index} must be contiguous across the rollback",
        );
    }
}

#[test]
fn minimal_terminal_fits_at_floor_ceiling() {
    // build_config floors `max_emitted_sse_frame_bytes` at 4096 so the fail-closed
    // minimal `response.failed` frame always fits. Prove that value is sufficient:
    // at a 4096-byte ceiling, an oversized-lifecycle failure whose full snapshot
    // (echoing 12 MiB of instructions) overflows both the lifecycle *and* the
    // full failure frame still terminates — the constant-bounded minimal resource
    // is emitted and the whole emitted stream fits within the floor. The generous
    // body ceiling from `wide_limits` keeps the frame-size retry — not the
    // byte-size fallback — the path under test.
    const FLOOR: usize = 4096;
    let big_instructions = "a".repeat(12 * 1024 * 1024);
    let body = json!({
        "model": "gpt-4.1-mini",
        "instructions": big_instructions,
        "input": "hi",
        "stream": true,
    });
    let limits = StreamLimits {
        max_emitted_sse_frame_bytes: FLOOR,
        ..wide_limits()
    };
    let mut conv = converter(limits);
    let mut raw = Vec::new();
    let inputs = SnapshotInputs {
        request_body: &body,
        now: NOW,
    };
    if let Some(bytes) = conv
        .push(
            &provider_stream(&[
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
                r#"{"id":"c1","object":"chat.completion.chunk","model":"gpt-4.1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ]),
            &inputs,
        )
        .unwrap()
    {
        raw.extend_from_slice(&bytes);
    }
    if let Some(bytes) = conv.finish(&inputs).unwrap() {
        raw.extend_from_slice(&bytes);
    }

    // The entire emitted stream is just the minimal terminal, and it fits the
    // configured floor — the guarantee the build_config floor exists to hold.
    assert!(
        !raw.is_empty() && raw.len() <= FLOOR,
        "the fail-closed minimal terminal ({} bytes) must fit the floor ceiling ({FLOOR} bytes)",
        raw.len(),
    );
    let events = parse_events(&raw);
    let (name, payload) = events.last().expect("the stream must emit a terminal event");
    assert_eq!(
        name, "response.failed",
        "the floor ceiling must still terminate the stream"
    );
    assert_eq!(
        payload["response"]["instructions"],
        Value::Null,
        "the minimal terminal must not echo the request instructions",
    );
    assert_eq!(
        payload["sequence_number"], 0,
        "the minimal terminal is the stream's only event",
    );
}
