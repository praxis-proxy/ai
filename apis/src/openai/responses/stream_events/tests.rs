// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

#![allow(
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::indexing_slicing,
    clippy::string_slice,
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::unused_async,
    unused_must_use,
    reason = "tests"
)]

use bytes::Bytes;
use praxis_filter::{FilterAction, HttpFilter, SubRequestResponseMode};
use serde_json::json;

use super::{
    ArmDecision, CompletionState, OpenaiStreamEventsFilter, StreamEventsState, accumulate_response_object,
    arm_decision, canonicalize_logical_response, encode_local_completion, encode_local_error,
};
use crate::{
    openai::{
        responses::state::{
            ClientToolEcho, ClientToolRestore, LoweredClientTool, OutputAssignment, ResponsesState, SynthesisKind,
        },
        sse::{SseFrameParser, SseParseError},
    },
    test_utils::{make_filter_context, make_request},
};

#[test]
fn lowered_snapshot_echo_fails_before_planning_many_owned_copies() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter());
    let mut state = ResponsesState::default();
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: None,
            restore: ClientToolRestore::Custom,
        },
    );
    state.client_tool_echo = Some(ClientToolEcho {
        tools: vec![json!({"type": "custom", "name": "public", "description": "x".repeat(16_000)})],
        tool_choice: json!("auto"),
    });
    state.apply_retained_payload_limit(65_536);
    assert!(state.can_retain_payload(0));
    ctx.extensions.insert(state);
    let frame = make_sse_chunk(
        "response.in_progress",
        &json!({"response": {"id": "resp_probe", "output": []}}),
    );
    let mut body = Some(Bytes::from(frame.repeat(20)));
    let allocations = allocation_counter::measure(|| {
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    });
    assert!(body.is_none(), "the whole co-batched chunk must be suppressed");
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some(super::RETAINED_PAYLOAD_OVERFLOW_MESSAGE)
    );
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert!(
        allocations.bytes_max < 65_536 * 2,
        "restoration must reject before retaining many echo copies: {}",
        allocations.bytes_max
    );
}

#[test]
fn lowered_namespace_batch_fails_before_restoration_fanout() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter());
    let mut state = ResponsesState::default();
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "read".to_owned(),
            namespace: Some("n".repeat(4_096)),
            restore: ClientToolRestore::Namespace,
        },
    );
    state.apply_retained_payload_limit(512 * 1_024);
    assert!(state.can_retain_payload(0), "the namespace definition itself must fit");
    ctx.extensions.insert(state);
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({"response": {"id": "resp_batch", "status": "in_progress", "output": []}}),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    assert!(created.is_some(), "the lifecycle event must be admitted");

    let mut wire = Vec::new();
    for index in 0..150 {
        wire.extend_from_slice(&make_sse_chunk(
            "response.output_item.added",
            &json!({
                "output_index": index,
                "item": {
                    "type": "function_call",
                    "name": "private",
                    "id": format!("fc_{index}"),
                    "call_id": format!("c{index}"),
                    "arguments": "",
                    "status": "in_progress"
                }
            }),
        ));
    }
    let mut body = Some(Bytes::from(wire));
    let allocations = allocation_counter::measure(|| {
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    });
    assert!(body.is_none(), "the over-budget batch must be withheld");
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some(super::RETAINED_PAYLOAD_OVERFLOW_MESSAGE),
        "the committed stream must report aggregate exhaustion"
    );
    assert!(
        allocations.bytes_max < 512 * 1_024 * 2,
        "restoration must reject before retaining many namespace copies: {}",
        allocations.bytes_max
    );
}

#[test]
fn native_item_done_does_not_reserve_an_unused_lowered_namespace() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter());
    let mut state = ResponsesState::default();
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "read".to_owned(),
            namespace: Some("n".repeat(4_096)),
            restore: ClientToolRestore::Namespace,
        },
    );
    state.apply_retained_payload_limit(65_536);
    ctx.extensions.insert(state);
    let item = json!({
        "type": "function_call", "name": "native_read", "id": "fc_native",
        "call_id": "call_native", "arguments": "{}", "status": "completed"
    });
    for (event_type, payload) in [
        (
            "response.created",
            json!({"response":{"id":"resp_native", "status":"in_progress", "output":[]}}),
        ),
        ("response.output_item.added", json!({"output_index":0, "item":item})),
        (
            "response.function_call_arguments.done",
            json!({"output_index":0, "item_id":"fc_native", "arguments":"{}"}),
        ),
        ("response.output_item.done", json!({"output_index":0, "item":item})),
    ] {
        let mut body = Some(make_sse_chunk(event_type, &payload));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(
            body.is_some(),
            "native event {event_type} must not reserve an unused namespace"
        );
    }
}

#[test]
fn native_terminal_avoids_duplicate_restoration_staging() {
    assert_terminal_avoids_duplicate_staging(false);
}

#[test]
fn lowered_terminal_avoids_duplicate_restoration_staging() {
    assert_terminal_avoids_duplicate_staging(true);
}

#[test]
fn lowered_completion_uses_post_accumulation_staging_near_cap() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::default();
    responses.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some("ns".to_owned()),
            restore: ClientToolRestore::Namespace,
        },
    );
    responses.apply_retained_payload_limit(65_536);
    ctx.extensions.insert(responses);
    let mut added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {
                "type": "function_call",
                "name": "private",
                "id": "fc_1",
                "call_id": "c1",
                "arguments": "",
                "status": "in_progress",
                "padding": "x".repeat(8_192)
            }
        }),
    ));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();
    assert!(added.is_some());

    let arguments = format!("{{\"text\":\"{}\"}}", "y".repeat(4_096));
    let mut done = Some(make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({"output_index": 0, "item_id": "fc_1", "arguments": arguments}),
    ));
    filter.on_response_body(&mut ctx, &mut done, false).unwrap();
    assert!(done.is_some(), "the live owners fit within the 64 KiB request budget");
    assert!(!ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn co_batched_completions_do_not_reserialize_unrelated_large_output() {
    assert_completion_batch_does_not_reserialize_unrelated_output(false);
}

#[test]
fn interleaved_item_done_does_not_reserialize_unrelated_large_output() {
    assert_completion_batch_does_not_reserialize_unrelated_output(true);
}

fn assert_completion_batch_does_not_reserialize_unrelated_output(interleave_done: bool) {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::default();
    responses.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some("ns".to_owned()),
            restore: ClientToolRestore::Namespace,
        },
    );
    responses.apply_retained_payload_limit(64 * 1024 * 1024);
    ctx.extensions.insert(responses);
    let mut added = make_sse_chunk(
        "response.output_item.added",
        &json!({"output_index": 0, "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "x".repeat(1_048_576)}]}}),
    )
    .to_vec();
    let mut done = Vec::new();
    for index in 0..300 {
        added.extend_from_slice(&make_sse_chunk(
            "response.output_item.added",
            &json!({"output_index": index + 1, "item": {"type": "function_call", "name": "private", "id": format!("fc_{index}"), "call_id": format!("c{index}"), "arguments": "", "status": "in_progress"}}),
        ));
        done.extend_from_slice(&make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": index + 1, "item_id": format!("fc_{index}"), "arguments": "{}"}),
        ));
        if interleave_done {
            done.extend_from_slice(&make_sse_chunk(
                "response.output_item.done",
                &json!({"output_index": index + 1, "item": {"type": "function_call", "name": "private", "id": format!("fc_{index}"), "call_id": format!("c{index}"), "arguments": "{}", "status": "completed"}}),
            ));
        }
    }
    let mut body = Some(Bytes::from(added));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(body.is_some());
    body = Some(Bytes::from(done));
    let started = std::time::Instant::now();
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(body.is_some(), "the completion batch must fit its 64 MiB budget");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(3),
        "300 small completions (interleaved={interleave_done}) must not rescan a 1 MiB output per event: {:?}",
        started.elapsed()
    );
}

#[test]
fn co_batched_completion_snapshots_reject_before_next_large_clone() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::default();
    responses.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some("ns".to_owned()),
            restore: ClientToolRestore::Namespace,
        },
    );
    ctx.extensions.insert(responses);
    let mut added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({"output_index": 0, "item": {"type": "function_call", "name": "private", "id": "fc_1", "call_id": "c1", "arguments": "", "status": "in_progress", "padding": "x".repeat(16_384)}}),
    ));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();
    let baseline = ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(65_536);
    let frame = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({"output_index": 0, "item_id": "fc_1", "arguments": "{}"}),
    );
    // Canonical output no longer clones a call into the dispatch queue. Four
    // restoration snapshots still exceed the remaining headroom.
    let mut body = Some(Bytes::from(frame.repeat(4)));
    let allocations = allocation_counter::measure(|| {
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    });
    assert!(body.is_none());
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some(super::RETAINED_PAYLOAD_OVERFLOW_MESSAGE)
    );
    assert!(
        allocations.bytes_max < u64::try_from(65_536 - baseline).unwrap(),
        "reject before the next clone exceeds remaining headroom: {allocations:?}"
    );
}

#[test]
fn done_delta_transfer_rechecks_shared_charge_before_next_snapshot() {
    let (filter, mut ctx) = make_armed_context();
    let mut stream = filter.new_round_state(0, 0);
    let mut responses = ResponsesState::default();
    responses.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some("ns".to_owned()),
            restore: ClientToolRestore::Namespace,
        },
    );
    let mut events = Vec::new();
    for index in 0..8 {
        let id = format!("fc_{index}");
        responses.output_items_mut().push(json!({
            "type": "function_call", "id": id, "call_id": format!("c_{index}"),
            "name": "private", "arguments": "", "status": "in_progress"
        }));
        stream
            .tool_call_args
            .insert(format!("item:{id}"), format!("{{\"text\":\"{}\"}}", "x".repeat(10_000)));
        events.push(
            crate::openai::sse::responses::ResponsesEvent::FunctionCallArgumentsDone(
                json!({"output_index": index, "item_id": id}),
            ),
        );
    }
    let parsed_owner_bytes: usize = events
        .iter()
        .map(|event| super::retained_event_payload_bytes(event).unwrap())
        .sum();
    responses.apply_retained_payload_limit(120_000);
    ctx.extensions.insert(responses);
    assert!(super::stream_payload_fits(&ctx, &stream, parsed_owner_bytes));

    let snapshots = super::accumulate_chunk(&mut stream, &mut ctx, &mut events, parsed_owner_bytes).unwrap();

    assert!(
        snapshots.is_none(),
        "transferred arguments and live snapshots exceed the cap"
    );
    assert_eq!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_external_payload_bytes,
        0,
        "rejected snapshots release their shared charge"
    );
}

#[test]
fn terminal_usage_merge_peak_is_reserved_before_accumulation() {
    let (filter, mut ctx) = make_armed_context();
    ctx.extensions.insert(ResponsesState::default());
    let mut stream = filter.new_round_state(0, 0);
    let mut event = crate::openai::sse::responses::ResponsesEvent::ResponseCompleted(json!({
        "type": "response.completed",
        "response": {
            "id": "resp_1", "object": "response", "status": "completed", "output": [],
            "usage": {"detail": "x".repeat(1_000_000)}
        }
    }));
    let projected = super::projected_responses_state_clone_bytes(&ctx, std::slice::from_ref(&event)).unwrap();
    let allocations = allocation_counter::measure(|| {
        super::accumulate_event(&mut ctx, &mut stream, &mut event);
    });

    assert!(
        allocations.bytes_max <= u64::try_from(projected + 64_000).unwrap(),
        "projected {projected} additional bytes, but accumulation allocated {} bytes at peak",
        allocations.bytes_max
    );
}

/// A terminal response is retained in shared state before restoration planning.
fn assert_terminal_avoids_duplicate_staging(lowered: bool) {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::from_request_body(json!({"model": "m", "input": "hi", "stream": true}));
    if lowered {
        responses.client_tool_lowering.insert(
            "private".to_owned(),
            LoweredClientTool {
                original_name: "public".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );
        responses.client_tool_echo = Some(ClientToolEcho {
            tools: vec![json!({"type": "custom", "name": "public"})],
            tool_choice: json!("auto"),
        });
    }
    responses.apply_retained_payload_limit(30_000);
    ctx.extensions.insert(responses);
    let metadata: serde_json::Map<String, serde_json::Value> =
        (0..8).map(|i| (format!("key{i}"), json!("x".repeat(512)))).collect();
    let terminal = make_sse_chunk(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": {
                "id": "resp_budget",
                "object": "response",
                "status": "completed",
                "output": [],
                "metadata": metadata,
            }
        }),
    );
    for slice in terminal.chunks(256) {
        let mut body = Some(Bytes::copy_from_slice(slice));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(
            ctx.get_metadata("responses.stream_error_code").is_none(),
            "{:?}",
            ctx.get_metadata("responses.stream_error_message")
        );
    }
    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let body = std::str::from_utf8(eos.as_ref().unwrap()).unwrap();
    assert!(body.contains("event: response.completed"), "{body}");
    assert!(ctx.get_metadata("responses.stream_error_code").is_none());
}

#[test]
fn native_terminal_does_not_pay_restoration_staging_after_move() {
    assert_terminal_does_not_pay_moved_staging(false);
}

#[test]
fn lowered_terminal_does_not_pay_restoration_staging_after_move() {
    assert_terminal_does_not_pay_moved_staging(true);
}

/// A terminal payload moves into shared state before the restoration plan.
fn assert_terminal_does_not_pay_moved_staging(lowered: bool) {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::from_request_body(json!({"model": "m", "input": "hi", "stream": true}));
    if lowered {
        responses.client_tool_lowering.insert(
            "private".to_owned(),
            LoweredClientTool {
                original_name: "public".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );
        responses.client_tool_echo = Some(ClientToolEcho {
            tools: vec![json!({"type": "custom", "name": "public"})],
            tool_choice: json!("auto"),
        });
    }
    responses.apply_retained_payload_limit(12_000);
    ctx.extensions.insert(responses);
    let metadata: serde_json::Map<String, serde_json::Value> =
        (0..8).map(|i| (format!("key{i}"), json!("x".repeat(512)))).collect();
    let terminal = make_sse_chunk(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": {
                "id": "resp_budget",
                "object": "response",
                "status": "completed",
                "output": [],
                "metadata": metadata,
            }
        }),
    );
    for slice in terminal.chunks(256) {
        let mut body = Some(Bytes::copy_from_slice(slice));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(
            ctx.get_metadata("responses.stream_error_code").is_none(),
            "{:?}",
            ctx.get_metadata("responses.stream_error_message")
        );
    }
    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let body = std::str::from_utf8(eos.as_ref().unwrap()).unwrap();
    assert!(body.contains("event: response.completed"), "{body}");
    assert!(ctx.get_metadata("responses.stream_error_code").is_none());
}

#[test]
fn done_after_terminal_at_max_events_is_allowed() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter_from("max_events: 1"));
    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});

    let mut terminal = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert_eq!(ctx.get_filter_state::<StreamEventsState>().unwrap().event_count, 1);

    let mut done = Some(make_done_chunk());
    filter.on_response_body(&mut ctx, &mut done, false).unwrap();
    let state = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert_eq!(state.event_count, 1, "[DONE] must not consume an event slot");
    assert_eq!(state.completion_state, CompletionState::TerminalLifecycle);
    assert!(state.deferred_done, "the downstream sentinel must be retained");
    assert!(ctx.get_metadata("responses.stream_parse_error").is_none());

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    assert!(ctx.get_metadata("responses.stream_error_code").is_none());
    assert!(ctx.get_metadata("responses.stream_incomplete").is_none());
    assert!(
        eos.as_deref().is_some_and(|bytes| bytes.ends_with(b"data: [DONE]\n\n")),
        "a valid terminal stream must forward its [DONE] sentinel"
    );
}

#[test]
fn counted_event_beyond_max_events_is_rejected() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter_from("max_events: 1"));
    let mut first = Some(make_sse_chunk("response.output_text.delta", &json!({"delta": "first"})));
    filter.on_response_body(&mut ctx, &mut first, false).unwrap();
    assert!(ctx.get_metadata("responses.stream_parse_error").is_none());

    let mut second = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({"delta": "second"}),
    ));
    filter.on_response_body(&mut ctx, &mut second, false).unwrap();
    assert_eq!(ctx.get_metadata("responses.stream_parse_error"), Some("true"));
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
}

fn make_filter() -> OpenaiStreamEventsFilter {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    OpenaiStreamEventsFilter::build(&yaml).unwrap()
}

fn make_filter_from(yaml: &str) -> OpenaiStreamEventsFilter {
    let yaml: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
    OpenaiStreamEventsFilter::build(&yaml).unwrap()
}

/// Build a context and arm the filter as the IRR runner plus `on_request`
/// would inside a step.
///
/// `on_request` fails closed unless an `IterationState` is present, and unit
/// tests cannot construct one (its fields are private to praxis-filter). So
/// tests arm directly through `arm`, which mirrors what `on_request` does once
/// the IRR-placement guard has admitted the request. The guard's decision table
/// (arm inside IRR, reject outside, ignore otherwise) is unit tested directly
/// through `arm_decision`; the end-to-end arming effect with a real IRR-inserted
/// `IterationState` is covered by the functional integration tests.
fn make_armed_context() -> (OpenaiStreamEventsFilter, praxis_filter::HttpFilterContext<'static>) {
    make_armed_context_with_filter(make_filter())
}

#[tokio::test]
async fn parser_payload_is_visible_to_a_different_agentic_filter_id() {
    let (stream_filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState {
        request_body: json!({"model": "m", "input": "hello", "stream": true}),
        ..ResponsesState::default()
    };
    responses.apply_retained_payload_limit(64 * 1024);
    ctx.extensions.insert(responses);
    let mut partial = Some(Bytes::from(format!("data: {}", "x".repeat(5_000))));
    stream_filter.on_response_body(&mut ctx, &mut partial, false).unwrap();
    let parser_bytes = ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .retained_stream_parser_bytes;
    assert!(parser_bytes >= 5_000, "unfinished frame must be published");

    // Praxis's filter state is keyed by current_filter_id. The loop callback
    // cannot read the parser's slot, but its shared budget must still reject it.
    ctx.current_filter_id = Some(1);
    assert!(ctx.get_filter_state::<StreamEventsState>().is_none());
    let agentic = crate::openai::responses::agentic_loop::AgenticLoopFilter::from_config(
        &serde_yaml::from_str("max_retained_bytes: 4096").unwrap(),
    )
    .unwrap();
    let action = agentic.on_request(&mut ctx).await.unwrap();
    assert!(matches!(action, FilterAction::Reject(_)));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn arm_decision_arms_streaming_responses_inside_irr() {
    assert_eq!(arm_decision(true, true), ArmDecision::Arm);
}

#[test]
fn arm_decision_rejects_streaming_responses_outside_irr() {
    assert_eq!(arm_decision(true, false), ArmDecision::RejectOutsideIrr);
}

#[test]
fn arm_decision_ignores_non_streaming_or_non_responses_requests() {
    assert_eq!(arm_decision(false, true), ArmDecision::Ignore);
    assert_eq!(arm_decision(false, false), ArmDecision::Ignore);
}

#[test]
fn default_config_parses() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    let filter = OpenaiStreamEventsFilter::from_config(&yaml).unwrap();
    assert_eq!(filter.name(), "openai_stream_events");
}

#[test]
fn custom_config_overrides_apply() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("max_buffer_bytes: 1048576\nmax_events: 500\ntimeout_secs: 60").unwrap();
    let filter = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(filter.is_ok(), "custom config should parse");
}

#[test]
fn stream_retained_payload_counts_parser_arguments_and_deferred_terminal() {
    let (_filter, mut ctx) = make_armed_context();
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let baseline = state.retained_payload_bytes().unwrap();
    let scratch = b"event: partial";
    state.frame_parser.parse_chunk(scratch).unwrap();
    state
        .tool_call_args
        .insert("item:call_1".to_owned(), "{\"x\":".to_owned());
    state.rejected_tool_call_args.insert("item:rejected".to_owned());
    state
        .local_tool_items
        .insert("item:local".to_owned(), super::local_tools::LocalToolMode::Suppress);
    let payload = json!({"type":"response.completed","response":{"output":[]}});
    state.deferred_terminal = Some(super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: payload.clone(),
    });

    let expected = scratch.len()
        + "item:call_1".len()
        + "{\"x\":".len()
        + "item:rejected".len()
        + "item:local".len()
        + "response.completed".len()
        + crate::openai::responses::state::retained_json_bytes(&payload).unwrap();
    assert_eq!(state.retained_payload_bytes().unwrap() - baseline, expected);
}

#[test]
fn local_completion_encodes_canonical_logical_sse_terminal() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        logical_stream_response_id: Some("resp_logical".to_owned()),
        logical_stream_sequence: 4,
        accumulated_output: vec![json!({"type":"mcp_approval_request", "id":"call_1"})],
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "output": []
        }),
        usage: json!({"total_tokens": 3}),
        ..ResponsesState::default()
    });

    let encoded = encode_local_completion(&mut ctx).expect("response object should encode");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    assert!(encoded.starts_with("event: response.completed\ndata: "));
    assert!(encoded.ends_with("\n\n"));
    let payload: serde_json::Value = serde_json::from_str(
        encoded
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE data line should exist"),
    )
    .unwrap();
    assert_eq!(payload["type"], "response.completed");
    assert_eq!(payload["sequence_number"], 4);
    assert_eq!(payload["response"]["id"], "resp_logical");
    assert_eq!(
        payload["response"]["output"].as_array().map(Vec::as_slice),
        state.response_object["output"].as_array().map(Vec::as_slice)
    );
    assert!(
        state.accumulated_output.is_empty(),
        "terminal output moved into response_object"
    );
    assert_eq!(payload["response"]["usage"], state.usage);
    assert_eq!(state.logical_stream_sequence, 5);
}

#[test]
fn local_completion_preflights_canonical_output_owners() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState {
        accumulated_output: vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "x".repeat(4_096)}]
        })],
        response_object: json!({"id": "resp_budget", "object": "response", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let baseline = state.retained_payload_bytes().unwrap();
    let staging = super::canonicalization_staging_bytes(&state, 0).unwrap();
    state.apply_retained_payload_limit(baseline + staging - 1);
    ctx.extensions.insert(state);

    let encoded = encode_local_completion(&mut ctx).expect("budget failure should still emit an error");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    assert!(encoded.contains("event: error"), "{encoded}");
    assert!(!encoded.contains("response.completed"), "{encoded}");
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(state.retained_payload_failed);
    assert!(state.accumulated_output.is_empty());
    assert!(state.response_object.is_null());
}

#[test]
fn local_completion_preflights_the_wire_copy_of_the_response() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState {
        response_object: json!({
            "id": "resp_budget",
            "object": "response",
            "status": "completed",
            "output": [],
            "metadata": {"large": "x".repeat(4_096)}
        }),
        ..ResponsesState::default()
    };
    let baseline = state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(baseline + 256);
    ctx.extensions.insert(state);

    let encoded = encode_local_completion(&mut ctx).expect("budget failure should emit an error");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    assert!(encoded.contains("event: error"), "{encoded}");
    assert!(!encoded.contains("response.completed"), "{encoded}");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
}

#[test]
fn local_completion_preflights_synthesized_sse_output() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let item = json!({
        "type": "mcp_call",
        "id": "call_budget",
        "status": "completed",
        "name": "tool",
        "arguments": "x".repeat(4_096)
    });
    let mut state = ResponsesState {
        logical_stream_response_id: Some("resp_budget".to_owned()),
        accumulated_output: vec![item],
        locally_executed_output_items: ["call_budget".to_owned()].into_iter().collect(),
        response_object: json!({"id": "resp_budget", "object": "response", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let baseline = state.retained_payload_bytes().unwrap();
    let local_output = super::local_terminal_output_upper_bound(&state).unwrap();
    let canonical_staging = super::canonicalization_staging_bytes(&state, local_output).unwrap();
    state.apply_retained_payload_limit(baseline + canonical_staging - 1);
    ctx.extensions.insert(state);

    let encoded = encode_local_completion(&mut ctx).expect("budget failure should still emit an error");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    assert!(encoded.contains("event: error"), "{encoded}");
    assert!(!encoded.contains("response.output_item.added"), "{encoded}");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_replaces_published_parser_charge_once() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    parser_state
        .tool_call_args
        .insert("item:large".to_owned(), "x".repeat(5_000));
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({
            "type": "response.completed",
            "response": {"id": "resp_budget", "object": "response", "status": "completed", "output": []}
        }),
    };
    let mut state = ResponsesState {
        response_object: json!({"id": "resp_budget", "object": "response", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let local_bytes = parser_state.retained_payload_bytes().unwrap();
    state.retained_stream_parser_bytes = local_bytes + terminal.retained_payload_bytes().unwrap();
    let baseline = state.retained_payload_bytes().unwrap();
    let canonical_staging = super::canonicalization_staging_bytes(&state, 0).unwrap();
    state.apply_retained_payload_limit(
        baseline + canonical_staging + terminal.retained_payload_bytes().unwrap() + 2_000,
    );
    ctx.extensions.insert(state);

    let mut output = Vec::new();
    super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output)
        .expect("the published parser owner must only be charged once");
    assert!(String::from_utf8_lossy(&output).contains("event: response.completed"));
    assert!(!ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_reserves_final_wire_capacity_before_serializing() {
    let fixture = || {
        let (_filter, ctx) = make_armed_context();
        let state = ResponsesState {
            accumulated_output: vec![json!({
                "type": "message",
                "content": [{"type": "output_text", "text": "x".repeat(256 * 1024)}]
            })],
            response_object: json!({"id": "resp_wire", "object": "response", "status": "completed", "output": []}),
            ..ResponsesState::default()
        };
        let terminal = super::DeferredTerminalEvent {
            event_type: "response.completed".to_owned(),
            payload: json!({"type": "response.completed", "response": null}),
        };
        (ctx, state, terminal, vec![b'x'; 64 * 1024])
    };
    let (mut ctx, mut state, mut terminal, mut output) = fixture();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let initial_capacity = output.capacity();
    state.apply_retained_payload_limit(900 * 1024);
    ctx.extensions.insert(state);

    let allocations = allocation_counter::measure(|| {
        super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output)
            .expect("an admitted terminal should serialize within its reserved wire owner");
    });
    assert!(output.ends_with(b"\n\n"));
    let measured_peak_bound = output.len() + initial_capacity + 64 * 1024;
    assert!(
        allocations.bytes_max <= u64::try_from(measured_peak_bound).unwrap(),
        "terminal serialization must stay within the admitted wire and old-buffer peak: {allocations:?}"
    );
    assert!(
        output.capacity() <= output.len() + initial_capacity,
        "terminal writer should reserve its final capacity before serializing: len={} capacity={}",
        output.len(),
        output.capacity()
    );

    // At this cap the canonical response and final wire fit, but replacing
    // the already allocated output buffer would keep its old capacity alive
    // during the new allocation. Reject before that uncharged peak occurs.
    let post_state = ctx.extensions.get::<ResponsesState>().unwrap();
    let near_cap = post_state.retained_payload_bytes().unwrap()
        + parser_state.retained_payload_bytes().unwrap()
        + output.len()
        + terminal.retained_payload_bytes().unwrap()
        + initial_capacity / 2;
    let (mut tight_ctx, mut tight_state, mut tight_terminal, mut tight_output) = fixture();
    let mut tight_parser = tight_ctx.remove_filter_state::<StreamEventsState>().unwrap();
    assert_eq!(tight_output.capacity(), initial_capacity);
    let initial_preflight = tight_state.retained_payload_bytes().unwrap()
        + tight_parser.retained_payload_bytes().unwrap()
        + super::canonicalization_staging_bytes(&tight_state, tight_output.len()).unwrap()
        + tight_terminal.retained_payload_bytes().unwrap();
    assert!(
        initial_preflight <= near_cap,
        "the canonicalization preflight should fit"
    );
    tight_state.apply_retained_payload_limit(near_cap);
    tight_ctx.extensions.insert(tight_state);
    assert!(
        super::emit_deferred_terminal(
            &mut tight_ctx,
            &mut tight_terminal,
            &mut tight_parser,
            &mut tight_output
        )
        .is_err(),
        "a cap below the old-buffer relocation peak must reject the terminal"
    );
    assert!(tight_output.is_empty());
    assert!(
        tight_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_failed
    );
}

#[test]
fn deferred_terminal_preflights_canonical_output_owners() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut state = ResponsesState {
        accumulated_output: vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "x".repeat(4_096)}]
        })],
        response_object: json!({"id": "resp_budget", "object": "response", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({
            "type": "response.completed",
            "response": {"id": "resp_budget", "status": "completed", "output": []}
        }),
    };
    let baseline = state.retained_payload_bytes().unwrap();
    let staging =
        super::canonicalization_staging_bytes(&state, 0).unwrap() + terminal.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(baseline + staging - 1);
    ctx.extensions.insert(state);
    let mut output = Vec::new();
    let mut terminal = terminal;

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output,).is_err());
    assert!(output.is_empty());
    assert!(terminal.payload["response"]["output"].as_array().unwrap().is_empty());
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_wire_overflow_keeps_sequence_for_error() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut state = ResponsesState {
        logical_stream_sequence: 4,
        accumulated_output: vec![json!({"type": "message", "content": [{"text": "x".repeat(4_096)}]})],
        response_object: json!({"id": "resp_budget", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type": "response.completed", "sequence_number": 4, "response": null}),
    };
    let preflight = super::canonicalization_staging_bytes(&state, 0).unwrap()
        + terminal.retained_payload_bytes().unwrap()
        + parser_state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(state.retained_payload_bytes().unwrap() + preflight + 128);
    ctx.extensions.insert(state);
    let mut output = Vec::new();

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(output.is_empty());
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().logical_stream_sequence,
        4
    );
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
}

#[test]
fn deferred_terminal_reserves_metadata_while_serializing_wire() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut state = ResponsesState {
        logical_stream_response_id: Some("resp_budget".to_owned()),
        accumulated_output: vec![json!({"type": "message", "content": [{"text": "x".repeat(256 * 1024)}]})],
        response_object: json!({"id": "resp_budget", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": null,
            "provider_metadata": "m".repeat(64 * 1024)
        }),
    };
    state.apply_retained_payload_limit(600 * 1024);
    let preflight = super::canonicalization_staging_bytes(&state, 0).unwrap()
        + terminal.retained_payload_bytes().unwrap()
        + parser_state.retained_payload_bytes().unwrap();
    assert!(
        state.can_replace_retained_payload(0, 0, preflight),
        "canonicalization must fit so this exercises the later wire admission"
    );
    ctx.extensions.insert(state);
    let mut output = Vec::new();

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(
        output.is_empty(),
        "over-budget terminal and metadata must stay off the wire"
    );
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
}

#[test]
fn deferred_terminal_preflights_one_echo_owner_before_canonicalization() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut state = ResponsesState {
        response_object: json!({"id": "resp_budget", "status": "completed", "output": [], "tools": []}),
        client_tool_echo: Some(ClientToolEcho {
            tools: vec![json!({"type": "custom", "name": "public", "description": "x".repeat(65_536)})],
            tool_choice: json!("auto"),
        }),
        ..ResponsesState::default()
    };
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: None,
            restore: ClientToolRestore::Custom,
        },
    );
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type": "response.completed", "response": null}),
    };
    let echo = state.client_tool_echo.as_ref().unwrap();
    let echo_bytes = crate::openai::responses::state::retained_json_values_bytes(&echo.tools).unwrap()
        + crate::openai::responses::state::retained_json_bytes(&echo.tool_choice).unwrap()
        + 64;
    let staging = super::canonicalization_staging_bytes(&state, 0).unwrap();
    let terminal_bytes = terminal.retained_payload_bytes().unwrap();
    let parser_bytes = parser_state.retained_payload_bytes().unwrap();
    let baseline = state.retained_payload_bytes().unwrap();
    let no_echo_staging = staging - echo_bytes;
    state.apply_retained_payload_limit(baseline + parser_bytes + terminal_bytes + no_echo_staging + 1_024);
    assert!(
        state.can_replace_retained_payload(0, parser_bytes, terminal_bytes + no_echo_staging),
        "the other terminal owners fit this budget"
    );
    assert!(
        !state.can_replace_retained_payload(0, parser_bytes, terminal_bytes + staging),
        "the restored echo owner must be charged before mutation"
    );
    ctx.extensions.insert(state);
    let mut output = Vec::new();

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(output.is_empty());
    assert!(
        terminal.payload.get("sequence_number").is_none(),
        "preflight must reject before it mutates the deferred envelope"
    );
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_preflights_namespaced_call_fanout_before_canonicalization() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let namespace = "n".repeat(8 * 1024);
    let mut state = ResponsesState {
        accumulated_output: (0..128)
            .map(|index| {
                json!({
                    "type": "function_call", "id": format!("fc_{index}"),
                    "call_id": format!("call_{index}"), "name": "private", "arguments": "{}"
                })
            })
            .collect(),
        response_object: json!({"id": "resp_budget", "status": "completed", "output": []}),
        ..ResponsesState::default()
    };
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some(namespace.clone()),
            restore: ClientToolRestore::Namespace,
        },
    );
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type": "response.completed", "response": null}),
    };
    let restored_name_bytes = (crate::openai::responses::state::retained_json_bytes("public").unwrap()
        + crate::openai::responses::state::retained_json_bytes(&namespace).unwrap()
        + 64)
        * 128;
    let staging = super::canonicalization_staging_bytes(&state, 0).unwrap();
    let without_fanout = staging - restored_name_bytes;
    let parser_bytes = parser_state.retained_payload_bytes().unwrap();
    let terminal_bytes = terminal.retained_payload_bytes().unwrap();
    let baseline = state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(baseline + parser_bytes + terminal_bytes + without_fanout + 1_024);
    assert!(
        state.can_replace_retained_payload(0, parser_bytes, terminal_bytes + without_fanout),
        "the other terminal owners fit this budget"
    );
    assert!(
        !state.can_replace_retained_payload(0, parser_bytes, terminal_bytes + staging),
        "the public namespace is copied into every canonical call"
    );
    ctx.extensions.insert(state);
    let mut output = Vec::new();

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(output.is_empty());
    assert!(terminal.payload.get("sequence_number").is_none());
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_reserves_namespaced_call_fanout_before_canonicalization() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let namespace = "n".repeat(8_192);
    let mut state = ResponsesState {
        accumulated_output: (0..128)
            .map(|index| {
                json!({
                    "type": "function_call", "id": format!("fc_{index}"),
                    "call_id": format!("call_{index}"), "name": "private",
                    "arguments": "{}", "status": "completed"
                })
            })
            .collect(),
        response_object: json!({"id":"resp_budget", "object":"response", "status":"completed", "output":[]}),
        client_tool_echo: Some(ClientToolEcho {
            tools: vec![
                json!({"type":"namespace", "name":namespace.as_str(), "tools":[{"type":"function", "name":"public", "parameters":{"type":"object"}}]}),
            ],
            tool_choice: json!("auto"),
        }),
        ..ResponsesState::default()
    };
    state.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: Some(namespace),
            restore: ClientToolRestore::Namespace,
        },
    );
    state.apply_retained_payload_limit(400_000);
    assert!(state.can_retain_payload(0));
    assert!(!state.can_retain_payload(super::canonicalization_staging_bytes(&state, 0).unwrap()));
    ctx.extensions.insert(state);
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type":"response.completed", "response":{"id":"resp_budget", "output":[]}}),
    };
    let mut output = Vec::new();

    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(output.is_empty());
    assert!(terminal.payload["response"]["output"].as_array().unwrap().is_empty());
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn deferred_terminal_reserves_citation_expansion_before_rewriting() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let item = json!({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "<|file-known|>".repeat(1_000)}],
    });
    let mut state = ResponsesState {
        accumulated_output: vec![item],
        response_object: json!({"id": "resp_citations", "object": "response", "status": "completed", "output": []}),
        citation_files: [("file-known".to_owned(), "x".repeat(1_024))].into(),
        ..ResponsesState::default()
    };
    let output_bytes = crate::openai::responses::state::retained_json_values_bytes(&state.accumulated_output).unwrap();
    let expanded_staging = super::canonicalization_staging_bytes(&state, 0).unwrap();
    assert!(
        expanded_staging > output_bytes * 10,
        "citation objects must be included in the preflight"
    );
    let terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type": "response.completed", "response": {"id": "resp_citations", "output": []}}),
    };
    let baseline = state.retained_payload_bytes().unwrap();
    state
        .apply_retained_payload_limit(baseline + output_bytes * 2 + terminal.retained_payload_bytes().unwrap() + 1_000);
    ctx.extensions.insert(state);
    let mut terminal = terminal;
    let mut output = Vec::new();
    assert!(super::emit_deferred_terminal(&mut ctx, &mut terminal, &mut parser_state, &mut output).is_err());
    assert!(output.is_empty());
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn streaming_budget_cache_counts_stable_owners_once_per_round() {
    let (filter, mut ctx) = make_armed_context();
    let stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut responses = ResponsesState {
        request_body: json!({"input": "p".repeat(32_768)}),
        input: vec![json!({"text": "history"})],
        response_object: json!({"output": []}),
        ..ResponsesState::default()
    };
    responses.apply_retained_payload_limit(1_000_000);
    ctx.extensions.insert(responses);
    let initial = super::shared_retained_budget(&ctx, &stream).unwrap().1.unwrap();
    assert!(stream.shared_stable_bytes.get().copied().unwrap() > 32_768);
    assert_eq!(
        initial,
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap()
    );
    ctx.insert_filter_state(stream);
    let mut body = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({"output_index": 0, "item": {"text": "new"}}),
    ));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    let stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let next = super::shared_retained_budget(&ctx, &stream).unwrap().1.unwrap();
    assert_eq!(
        next,
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap()
    );
    assert!(stream.shared_stable_bytes.get().copied().unwrap() > 32_768);
}

#[test]
fn streaming_budget_caches_client_tool_echo_captured_after_arm() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::default();
    // Client-tool lowering captures this snapshot after stream-events arms at
    // request time, before the first upstream response chunk is checked.
    responses.client_tool_echo = Some(ClientToolEcho {
        tools: vec![json!({"type": "function", "name": "large", "description": "x".repeat(1_048_576)})],
        tool_choice: json!("auto"),
    });
    responses.apply_retained_payload_limit(1_200_000);
    let expected = responses.retained_payload_bytes().unwrap();
    ctx.extensions.insert(responses);

    let started = std::time::Instant::now();
    for _ in 0..200 {
        let mut body = Some(Bytes::from_static(b": ping\n\n"));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "the stable echo must not be serialized on every tiny SSE chunk: {:?}",
        started.elapsed()
    );

    let stream = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert!(stream.shared_stable_bytes.get().copied().unwrap() > 1_048_576);
    assert_eq!(super::shared_retained_budget(&ctx, stream).unwrap().1, Some(expected));
    let responses = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        responses
            .stream_changing_payload_bytes_bounded_for_parser(1_024)
            .is_some()
    );
    assert!(!responses.retained_payload_failed);
}

#[test]
fn streaming_budget_reuses_unchanged_current_output_charge() {
    let (_filter, mut ctx) = make_armed_context();
    let stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let call = json!({
        "type": "function_call",
        "id": "fc_completed",
        "name": "lookup",
        "arguments": "x".repeat(512 * 1024),
        "status": "completed",
    });
    let mut responses = ResponsesState {
        response_object: json!({"output": [call.clone()]}),
        tool_calls: vec![OutputAssignment::new(0, &call).unwrap()],
        accumulated_output: vec![call],
        ..ResponsesState::default()
    };
    let expected = responses.retained_payload_bytes().unwrap();
    responses.apply_retained_payload_limit(expected + 1);
    ctx.extensions.insert(responses);

    let started = std::time::Instant::now();
    for _ in 0..200 {
        assert_eq!(super::shared_retained_budget(&ctx, &stream).unwrap().1, Some(expected));
    }
    assert!(super::stream_payload_fits(&ctx, &stream, 1));
    assert!(!super::stream_payload_fits(&ctx, &stream, 2));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "unchanged completed output must not be reserialized on every SSE admission: {:?}",
        started.elapsed()
    );
}

#[test]
fn flushed_local_item_does_not_reproject_on_heartbeat_chunks() {
    let (filter, mut ctx) = make_armed_context();
    let item = json!({
        "type": "mcp_call",
        "id": "call_flushed",
        "status": "completed",
        "name": "tool",
        "result": "x".repeat(1_048_576),
    });
    let mut responses = ResponsesState {
        accumulated_output: vec![item],
        locally_executed_output_items: ["call_flushed".to_owned()].into_iter().collect(),
        ..ResponsesState::default()
    };
    let baseline = responses.retained_payload_bytes().unwrap();
    responses.apply_retained_payload_limit(64 * 1_048_576);
    ctx.extensions.insert(responses);
    let mut parser = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    parser.local_items_flushed = true;
    assert!(
        super::local_terminal_output_upper_bound(ctx.extensions.get::<ResponsesState>().unwrap()).unwrap() > 1_048_576
    );
    assert_eq!(super::logical_output_upper_bound(&ctx, true, &[]), Some(0));
    ctx.insert_filter_state(parser);

    let started = std::time::Instant::now();
    for _ in 0..200 {
        let mut body = Some(Bytes::from_static(b": heartbeat\n\n"));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(4),
        "heartbeat admission must not serialize the completed local item: {:?}",
        started.elapsed()
    );

    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(baseline + 512);
    let mut heartbeat = Some(Bytes::from_static(b": heartbeat\n\n"));
    filter.on_response_body(&mut ctx, &mut heartbeat, false).unwrap();
    assert!(!ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert!(ctx.get_metadata("responses.stream_error_message").is_none());
}

#[test]
fn unbounded_stream_skips_echo_projection_without_snapshots() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::default();
    responses.client_tool_echo = Some(ClientToolEcho {
        tools: vec![json!({"type": "function", "name": "large", "description": "x".repeat(1_048_576)})],
        tool_choice: json!("auto"),
    });
    responses.client_tool_lowering.insert(
        "private".to_owned(),
        LoweredClientTool {
            original_name: "public".to_owned(),
            namespace: None,
            restore: ClientToolRestore::Custom,
        },
    );
    ctx.extensions.insert(responses);

    let started = std::time::Instant::now();
    for _ in 0..200 {
        let mut body = Some(Bytes::from_static(b": ping\n\n"));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    }
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "heartbeat chunks must not serialize a 1 MiB echo without snapshots: {:?}",
        started.elapsed()
    );
    assert!(ctx.get_metadata("responses.stream_error_message").is_none());
}

#[test]
fn streaming_budget_cache_refreshes_prior_output_after_revision_change() {
    let (_filter, mut ctx) = make_armed_context();
    let stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut responses = ResponsesState {
        accumulated_output: vec![json!({"text": "a".repeat(128 * 1024)})],
        ..ResponsesState::default()
    };
    responses.apply_retained_payload_limit(1_000_000);
    ctx.extensions.insert(responses);

    let initial = super::shared_retained_budget(&ctx, &stream).unwrap().1.unwrap();
    let cache = stream.shared_prior_output_bytes.get().unwrap().unwrap();
    assert!(cache.bytes > 128 * 1024);
    assert_eq!(
        initial,
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap()
    );
    assert_eq!(super::shared_retained_budget(&ctx, &stream).unwrap().1, Some(initial));

    let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    responses.accumulated_output[0]["text"] = json!("b".repeat(256 * 1024));
    responses.mark_replay_stable_payload_changed();
    let next = super::shared_retained_budget(&ctx, &stream).unwrap().1.unwrap();
    assert_eq!(
        next,
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap()
    );
    assert_eq!(next - initial, 128 * 1024);
}

#[test]
fn eos_remeasures_history_appended_after_the_first_stream_chunk() {
    let (filter, mut ctx) = make_armed_context();
    let stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut responses = ResponsesState::from_request_body(json!({"model": "test", "input": "hi", "stream": true}));
    let baseline = responses.retained_payload_bytes().unwrap();
    responses.apply_retained_payload_limit(baseline + 2_000);
    ctx.extensions.insert(responses);
    assert!(super::shared_retained_budget(&ctx, &stream).unwrap().1.is_some());
    ctx.insert_filter_state(stream);

    // Agentic collection runs before the stream-events EOS callback and copies
    // the round's reasoning/tool item into both history owners.
    let item = json!({"type": "reasoning", "id": "rs_eos", "summary": "x".repeat(3_000)});
    let responses = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    responses.messages.push(item.clone());
    responses.persisted_messages.push(item);

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some(super::RETAINED_PAYLOAD_OVERFLOW_MESSAGE)
    );
}

#[test]
fn finalized_parser_charge_does_not_follow_the_next_irr_step() {
    use super::client_tools::{ClientToolPhase, ClientToolStreamItem};
    use crate::openai::responses::state::ClientToolRestore;

    let (_filter, mut ctx) = make_armed_context();
    let mut stream = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    stream.client_tool_items.push(ClientToolStreamItem {
        key: "item:call_done".to_owned(),
        private_name: "private_tool".repeat(100),
        restore: ClientToolRestore::Custom,
        phase: ClientToolPhase::Done,
        output_index: 0,
        item_id: Some("call_done".to_owned()),
    });
    let parser_bytes = stream.retained_payload_bytes().unwrap();
    let mut responses = ResponsesState::from_request_body(json!({"model": "test", "input": "hi", "stream": true}));
    let baseline = responses.retained_payload_bytes().unwrap();
    responses.apply_retained_payload_limit(baseline + parser_bytes + 128);
    ctx.extensions.insert(responses);
    assert!(super::publish_stream_payload(&mut ctx, &stream));
    ctx.insert_filter_state(stream);
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "loop");

    let mut output = None;
    super::finalize_logical_stream(&mut ctx, &mut output);
    assert_eq!(
        ctx.get_filter_state::<StreamEventsState>()
            .unwrap()
            .retained_payload_bytes(),
        Some(0)
    );
    let responses = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(responses.retained_stream_parser_bytes, 0);
    assert!(
        responses.can_retain_payload(parser_bytes + 128),
        "the next IRR step must have room for payload that the old parser released"
    );
}

#[test]
fn shared_eos_wire_rejects_before_copying_a_second_large_buffer() {
    let (_filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::from_request_body(json!({"model": "m", "input": "q", "stream": true}));
    let baseline = responses.retained_payload_bytes().unwrap();
    let wire = Bytes::from(vec![b'x'; 32 * 1024]);
    // A downstream handle keeps the normalized EOS chunk shared while this
    // filter finalizes it. The old and new buffers coexist during a copy.
    let _other_owner = wire.clone();
    responses.apply_retained_payload_limit(baseline + wire.len() * 2 - 1);
    ctx.extensions.insert(responses);
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "loop");

    let mut body = Some(wire);
    let allocations = allocation_counter::measure(|| {
        super::finalize_logical_stream(&mut ctx, &mut body);
    });
    let output = String::from_utf8(body.expect("bounded error frame").to_vec()).unwrap();
    assert_eq!(output.matches("event: error").count(), 1);
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert!(
        allocations.bytes_max < 32 * 1024,
        "the oversized second wire buffer must never be allocated: {allocations:?}"
    );
}

#[test]
fn numeric_sse_payload_rejects_before_parsed_value_expands() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::from_request_body(json!({"model": "m", "input": "q", "stream": true}));
    responses.apply_retained_payload_limit(45 * 1024);
    ctx.extensions.insert(responses);

    // serde_json normalizes each four-byte `1e15` into an eighteen-byte
    // decimal. The raw frame fits the parser preflight, but its parsed JSON
    // and the retained frame cannot coexist under this limit.
    let data = format!(
        "{{\"type\":\"vendor.extension\",\"numbers\":[{}]}}",
        vec!["1e15"; 2_000].join(",")
    );
    let mut body = Some(Bytes::from(format!("event: vendor.extension\ndata: {data}\n\n")));
    let allocations = allocation_counter::measure(|| {
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    });
    assert!(body.is_none(), "the overflowing frame must not be forwarded");
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some(super::RETAINED_PAYLOAD_OVERFLOW_MESSAGE)
    );
    assert!(
        allocations.bytes_max < 40 * 1024,
        "reject before allocating the expanded JSON tree: {allocations:?}"
    );
}

#[test]
fn store_budget_error_drops_later_stream_events() {
    let (filter, mut ctx) = make_armed_context();
    ctx.set_metadata("responses.store_stream_budget_failed", "true");
    let mut body = Some(Bytes::from_static(
        b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n",
    ));
    drop(filter.on_response_body(&mut ctx, &mut body, true).unwrap());
    assert!(body.is_none());
    assert_eq!(ctx.get_metadata("responses.stream_completion"), Some("open"));
}

#[test]
fn reentry_arm_preserves_response_template_for_local_completion() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        logical_stream_response_id: Some("resp_logical".to_owned()),
        accumulated_output: vec![json!({"type":"mcp_approval_request", "id":"approval_1"})],
        response_object: json!({
            "id":"resp_upstream", "object":"response", "status":"completed", "output":[]
        }),
        ..ResponsesState::default()
    });

    filter.arm(&mut ctx);

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .response_object
            .is_null(),
        "re-entry must invalidate the prior upstream terminal"
    );
    let encoded = encode_local_completion(&mut ctx).expect("the preserved response template should encode");
    let encoded = std::str::from_utf8(&encoded).unwrap();
    assert!(
        encoded.contains("event: response.completed"),
        "local completion must restore a terminal response after re-entry: {encoded}"
    );
    assert!(
        encoded.contains("\"id\":\"approval_1\""),
        "the restored terminal must contain accumulated output: {encoded}"
    );
}

#[test]
fn local_completion_preserves_deferred_done_sentinel() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        response_object: json!({"id":"resp_1", "object":"response", "status":"completed", "output":[]}),
        deferred_stream_done: true,
        ..ResponsesState::default()
    });

    let encoded = encode_local_completion(&mut ctx).expect("response object should encode");
    assert!(
        encoded.ends_with(b"data: [DONE]\n\n"),
        "request-side completion must preserve the upstream sentinel"
    );
}

#[test]
fn local_completion_does_not_arm_deferred_store_persistence() {
    // #937 review regression: a request-phase local completion is returned to the
    // store as a buffered `TerminalResponse` at end-of-stream, where the store
    // already persists before the body is written. It must NOT set
    // `logical_stream_terminal_emitted` — that flag means the terminal is a
    // deferred non-end-of-stream chunk, and setting it here would make the store
    // skip its end-of-stream persist and lose the record.
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        response_object: json!({"id":"resp_1", "object":"response", "status":"completed", "output":[]}),
        ..ResponsesState::default()
    });

    let encoded = encode_local_completion(&mut ctx).expect("response object should encode");
    assert!(
        std::str::from_utf8(&encoded)
            .unwrap()
            .contains("event: response.completed"),
        "local completion must emit a terminal response.completed frame"
    );
    assert!(
        !ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .logical_stream_terminal_emitted,
        "a buffered local completion must not arm the deferred non-EOS persist path"
    );
}

#[test]
fn local_completion_flushes_file_search_lifecycle_before_terminal() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        accumulated_output: vec![json!({
            "type":"file_search_call",
            "id":"fs_local",
            "status":"completed",
            "results":[]
        })],
        pending_local_tool_synthesis: vec![(0, SynthesisKind::Private)],
        response_object: json!({
            "id":"resp_local",
            "object":"response",
            "status":"completed",
            "output":[]
        }),
        ..ResponsesState::default()
    });

    let encoded = encode_local_completion(&mut ctx).expect("local completion should encode");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();

    let done = encoded
        .find("event: response.output_item.done")
        .expect("the local file-search lifecycle should be flushed");
    let terminal = encoded
        .find("event: response.completed")
        .expect("the response terminal should be emitted");
    assert!(
        done < terminal,
        "the local tool lifecycle must precede response.completed"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .pending_local_tool_synthesis
            .is_empty(),
        "local completion must drain the synthesis queue exactly once"
    );
}

#[test]
fn local_error_flushes_file_search_lifecycle_before_terminal() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        accumulated_output: vec![json!({
            "type":"file_search_call",
            "id":"fs_local",
            "status":"completed",
            "results":[]
        })],
        pending_local_tool_synthesis: vec![(0, SynthesisKind::Private)],
        ..ResponsesState::default()
    });

    let encoded = encode_local_error(&mut ctx, "server_error", "dispatch failed").expect("local error should encode");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();

    let done = encoded
        .find("event: response.output_item.done")
        .expect("the local file-search lifecycle should be flushed");
    let terminal = encoded
        .find("event: error")
        .expect("the error terminal should be emitted");
    assert!(done < terminal, "the local tool lifecycle must precede the SSE error");
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .pending_local_tool_synthesis
            .is_empty(),
        "local error must drain the synthesis queue exactly once"
    );
}

#[test]
fn local_error_preflights_its_message_and_wire_buffer() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    let mut state = ResponsesState::default();
    let baseline = state.retained_payload_bytes().unwrap();
    state.apply_retained_payload_limit(baseline + 512);
    ctx.extensions.insert(state);

    let encoded =
        encode_local_error(&mut ctx, "server_error", &"x".repeat(4_096)).expect("budget failure should emit an error");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    assert!(encoded.contains("event: error"), "{encoded}");
    assert!(encoded.contains("max_retained_bytes"), "{encoded}");
    assert!(!encoded.contains(&"x".repeat(128)), "{encoded}");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
}

#[test]
fn canonicalize_restores_previous_response_id_into_store_source() {
    // #1150: a rehydrated streaming turn strips `previous_response_id` from the
    // upstream request, so the backend echoes `null` in its terminal lifecycle
    // response. The rehydrate filter repairs the client-visible SSE bytes, but
    // the persistence source is this independent `response_object`. The
    // canonicalization boundary must restore the caller's id here too, or a
    // later GET returns different continuation metadata than the terminal frame.
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        history_rehydrated: true,
        previous_response_id: Some("resp_prev".to_owned()),
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "previous_response_id": serde_json::Value::Null,
            "output": []
        }),
        ..ResponsesState::default()
    });

    let encoded = encode_local_completion(&mut ctx).expect("response object should encode");
    let encoded = String::from_utf8(encoded.to_vec()).unwrap();
    let payload: serde_json::Value = serde_json::from_str(
        encoded
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE data line should exist"),
    )
    .unwrap();

    // The store source (`response_object`) is the record a later GET serves.
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.response_object["previous_response_id"], "resp_prev",
        "the persisted store source must carry the caller's previous_response_id, \
         not the backend's null"
    );
    assert_eq!(
        payload["response"]["previous_response_id"], "resp_prev",
        "the canonical logical terminal must agree with the wire terminal"
    );
}

#[test]
fn canonicalize_preserves_backend_previous_response_id_without_rehydration() {
    // Without rehydration the proxy leaves `previous_response_id` on the upstream
    // request, so the backend echoes the real value. The canonicalization
    // boundary must not overwrite it with request state (which is `None` here),
    // and it must never fabricate one when history was not rehydrated.
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(&req);
    ctx.extensions.insert(ResponsesState {
        history_rehydrated: false,
        previous_response_id: None,
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "previous_response_id": "resp_backend",
            "output": []
        }),
        ..ResponsesState::default()
    });

    encode_local_completion(&mut ctx).expect("response object should encode");

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.response_object["previous_response_id"], "resp_backend",
        "a non-rehydrated turn must keep the backend-echoed previous_response_id"
    );
}

#[test]
fn canonicalize_skips_previous_response_id_when_wire_rewrite_declined() {
    // #1150 review: for a validator-bearing or non-200 event stream,
    // `openai_responses_rehydrate` declines the wire rewrite and leaves the
    // streamed terminal's `previous_response_id` as the backend-echoed `null`.
    // Canonicalization is the persistence source and MUST make the same decision,
    // or a later GET returns a `previous_response_id` the streamed response never
    // carried. The upstream-streaming caller passes the filter's declined
    // eligibility (`restore = false`) even on a rehydrated turn.
    let mut state = ResponsesState {
        history_rehydrated: true,
        previous_response_id: Some("resp_prev".to_owned()),
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "previous_response_id": serde_json::Value::Null,
            "output": []
        }),
        ..ResponsesState::default()
    };

    canonicalize_logical_response(&mut state, false).expect("canonicalize");

    assert_eq!(
        state.response_object["previous_response_id"],
        serde_json::Value::Null,
        "a wire-ineligible stream must leave the stored previous_response_id untouched \
         so the persisted record matches the un-rewritten terminal frame"
    );
}

#[test]
fn canonicalize_restores_previous_response_id_when_wire_rewrite_armed() {
    // The counterpart to the declined case: when the wire rewrite is armed
    // (`restore = true`), the persistence source restores the caller's id so a
    // later GET agrees with the rewritten terminal frame.
    let mut state = ResponsesState {
        history_rehydrated: true,
        previous_response_id: Some("resp_prev".to_owned()),
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "previous_response_id": serde_json::Value::Null,
            "output": []
        }),
        ..ResponsesState::default()
    };

    canonicalize_logical_response(&mut state, true).expect("canonicalize");

    assert_eq!(
        state.response_object["previous_response_id"], "resp_prev",
        "an armed wire rewrite must restore the caller's previous_response_id into \
         the persisted store source"
    );
}

#[test]
fn canonicalize_restores_lowered_client_tool_terminal_snapshot() {
    // #1159: the terminal response.completed carries the lowered private
    // function_call plus the backend-echoed LOWERED tools/tool_choice.
    // canonicalize is the last-chance restore: re-type the call to a
    // custom_tool_call and restore the client's own tools declaration, leaking
    // no private lowered name.
    let mut state = ResponsesState {
        client_tool_lowering: std::collections::HashMap::from([(
            "run_python".to_owned(),
            LoweredClientTool {
                original_name: "run_python".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        )]),
        client_tool_echo: Some(ClientToolEcho {
            tools: vec![json!({"type": "custom", "name": "run_python"})],
            tool_choice: serde_json::Value::Null,
        }),
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "tools": [{"type": "function", "name": "run_python"}],
            "tool_choice": "auto",
            "output": [{
                "type": "function_call",
                "name": "run_python",
                "call_id": "call_1",
                "arguments": r#"{"input":"print(1)"}"#
            }]
        }),
        ..ResponsesState::default()
    };

    canonicalize_logical_response(&mut state, false).expect("terminal restore");
    let output = state.response_object["output"].as_array().unwrap();

    assert_eq!(output[0]["type"], "custom_tool_call", "lowered function_call retyped");
    assert_eq!(
        output[0]["input"], "print(1)",
        "single string parameter unwrapped to plain input"
    );
    assert_eq!(
        state.response_object["tools"],
        json!([{"type": "custom", "name": "run_python"}]),
        "tools restored to the client's custom declaration"
    );
    assert_eq!(
        state.response_object["tool_choice"], "auto",
        "a Null tool_choice echo normalizes to auto"
    );
    assert!(
        !state.response_object.to_string().contains("agentic_ns__"),
        "no private lowered name leaks into the terminal response object"
    );
}

#[test]
fn canonicalize_fails_closed_on_lossy_terminal_client_tool_restore() {
    // #1159: a lossy terminal restore (here a lowered custom call missing its
    // call_id) fails the whole logical stream closed rather than emitting a
    // private lowered shape. The error must not echo the offending tool name.
    let mut state = ResponsesState {
        client_tool_lowering: std::collections::HashMap::from([(
            "run_python".to_owned(),
            LoweredClientTool {
                original_name: "run_python".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        )]),
        response_object: json!({
            "id": "resp_upstream",
            "object": "response",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "name": "run_python",
                "arguments": r#"{"input":"print(1)"}"#
            }]
        }),
        ..ResponsesState::default()
    };

    let err =
        canonicalize_logical_response(&mut state, false).expect_err("a lossy client-tool restore must fail closed");
    assert!(
        matches!(err, SseParseError::ClientToolRestore { .. }),
        "fail-closed uses the dedicated client-tool restore variant: {err:?}"
    );
    assert!(
        !err.to_string().contains("run_python"),
        "the error must not echo the lowered tool name: {err}"
    );
}

#[test]
fn response_body_access_is_always_read_write() {
    let filter = make_filter();
    assert_eq!(
        filter.response_body_access(),
        praxis_filter::BodyAccess::ReadWrite,
        "logical lifecycle normalization always rewrites emitted SSE frames"
    );
}

#[test]
fn unknown_config_field_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("bogus_field: true").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

#[test]
fn zero_max_buffer_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_buffer_bytes: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_buffer_bytes should be rejected");
}

#[test]
fn zero_max_events_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_events: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_events should be rejected");
}

#[test]
fn zero_timeout_secs_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_secs: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero timeout_secs should be rejected");
}

#[test]
fn zero_max_tool_call_argument_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_tool_call_argument_bytes should be rejected");
}

#[test]
fn oversized_max_buffer_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_buffer_bytes: 100000000").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "max_buffer_bytes above 64 MiB should be rejected");
}

#[test]
fn oversized_max_tool_call_argument_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 100000000").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(
        result.is_err(),
        "max_tool_call_argument_bytes above 64 MiB should be rejected"
    );
}

#[test]
fn oversized_max_events_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_events: 100000000").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "max_events above the ceiling should be rejected");
}

#[test]
fn zero_max_accumulated_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_accumulated_bytes: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_accumulated_bytes should be rejected");
}

#[test]
fn oversized_max_accumulated_bytes_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_accumulated_bytes: 100000000").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "max_accumulated_bytes above 64 MiB should be rejected");
}

#[test]
fn zero_max_output_items_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_output_items: 0").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "zero max_output_items should be rejected");
}

#[test]
fn oversized_max_output_items_rejected() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_output_items: 100000000").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_err(), "max_output_items above the ceiling should be rejected");
}

#[test]
fn accumulation_budget_fields_accepted() {
    let yaml: serde_yaml::Value =
        serde_yaml::from_str("max_accumulated_bytes: 1048576\nmax_output_items: 500").unwrap();
    let result = OpenaiStreamEventsFilter::from_config(&yaml);
    assert!(result.is_ok(), "in-range accumulation budget fields should be accepted");
}

#[tokio::test]
async fn on_request_rejects_streaming_responses_outside_irr() {
    // Without an `IterationState` in extensions the filter is placed outside an
    // `iterative_request_router` step. It cannot compose a logical stream there,
    // so it must fail closed rather than arm.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 500),
        "out-of-IRR placement must fail closed with a 500"
    );
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "a rejected request must not arm the SSE parser"
    );
}

#[tokio::test]
async fn on_request_rejects_typed_streaming_outside_irr() {
    // The typed terminal-streaming selection path must also fail closed when the
    // filter is not inside an `iterative_request_router` step.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(&action, FilterAction::Reject(rejection) if rejection.status == 500),
        "out-of-IRR typed streaming must fail closed with a 500"
    );
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "a rejected typed-streaming request must not arm the SSE parser"
    );
}

#[test]
fn arm_publishes_logical_stream_marker() {
    // openai_agentic_loop reads and consumes this marker to fail closed on the
    // unsafe terminal_streaming + agentic_loop combination when stream_events is
    // absent. Arming always publishes it because the filter is always logical.
    let (_filter, ctx) = make_armed_context();

    assert_eq!(
        ctx.get_metadata("responses.logical_stream"),
        Some("true"),
        "arming must publish the per-round marker openai_agentic_loop consumes"
    );
    assert_eq!(
        ctx.get_metadata(super::super::STREAM_ERROR_FINALIZER_ARMED_KEY),
        Some("true"),
        "the translator can leave overflow emission to this armed parser"
    );
}

#[tokio::test]
async fn does_not_arm_for_non_streaming() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "false".to_owned());
    ctx.current_filter_id = Some(0);

    let _action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "filter should not arm for non-streaming"
    );
}

#[tokio::test]
async fn does_not_arm_for_non_responses_format() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/chat/completions");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_chat_completions".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    let _action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "filter should not arm for non-responses format"
    );
}

#[tokio::test]
async fn does_not_arm_for_other_responses_routes() {
    for (method, path) in [
        (http::Method::GET, "/v1/responses"),
        (http::Method::POST, "/v1/responses/input_tokens"),
    ] {
        let filter = make_filter();
        let req = make_request(method, path);
        let mut ctx = make_filter_context(Box::leak(Box::new(req)));
        ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
        ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
        ctx.current_filter_id = Some(0);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(
            matches!(action, FilterAction::Continue),
            "non-create Responses route should continue"
        );
        assert!(
            ctx.get_filter_state::<StreamEventsState>().is_none(),
            "filter should not arm for {path}"
        );
    }
}

#[test]
fn unarmed_filter_passes_through_body() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);

    let mut body = Some(Bytes::from("data: {}\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "unarmed response body should continue"
    );
    assert!(body.is_some(), "body should not be consumed");
}

fn make_sse_chunk(event_type: &str, data: &serde_json::Value) -> Bytes {
    let mut obj = data.clone();
    obj.as_object_mut()
        .unwrap()
        .entry("type")
        .or_insert_with(|| serde_json::Value::String(event_type.to_owned()));
    let data_str = serde_json::to_string(&obj).unwrap();
    Bytes::from(format!("event: {event_type}\ndata: {data_str}\n\n"))
}

#[tokio::test]
async fn logical_stream_suppresses_intermediate_terminal_and_normalizes_resumed_turn() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_first", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    assert!(
        String::from_utf8_lossy(created.as_ref().unwrap()).contains("resp_first"),
        "the first response lifecycle should be emitted"
    );

    let function_call = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather__get",
        "arguments": "{}",
        "status": "completed"
    });
    let mut terminal = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {"id": "resp_first", "status": "completed", "output": [function_call.clone()]},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(terminal.is_none(), "the per-turn terminal must be withheld");
    // After the #1046 unification the loop owner (`openai_agentic_loop`) is the
    // single continuation authority: it records `action="loop"` for the
    // MCP-classified call before the intermediate terminal is suppressed.
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();
    assert!(
        first_eos.is_none(),
        "an agentic transition must suppress the intermediate terminal"
    );

    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![function_call, json!({"type": "mcp_call", "id": "mcp_1"})];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_agentic_loop");
    filter.arm(&mut ctx);

    let mut resumed_created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_second", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut resumed_created, false).unwrap();
    assert!(
        resumed_created.is_none(),
        "resumed lifecycle creation must be suppressed"
    );

    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_second",
            "output_index": 0,
            "content_index": 0,
            "delta": "done",
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();
    let delta = String::from_utf8(delta.unwrap().to_vec()).unwrap();
    assert!(
        delta.contains(r#""response_id":"resp_first""#),
        "logical response ID should remain stable: {delta}"
    );
    assert!(
        delta.contains(r#""output_index":2"#),
        "resumed output index should include prior tool items: {delta}"
    );
    // #276: the locally executed MCP call must be synthesized as an incremental
    // output item, at its reserved index, before the resumed model output.
    assert!(
        delta.contains("event: response.output_item.added") && delta.contains("event: response.output_item.done"),
        "the locally executed MCP call should surface as incremental output-item events: {delta}"
    );
    assert!(
        delta.contains(r#""output_index":1"#),
        "the synthesized MCP call should occupy its reserved output index: {delta}"
    );
    // #276: the synthesized MCP call must carry tool-specific progress and
    // outcome events between the generic output-item events. A successful call
    // (no `error` key) emits `in_progress` then `completed`.
    assert!(
        delta.contains("event: response.mcp_call.in_progress"),
        "the synthesized MCP call must emit an in_progress progress event: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_call.completed"),
        "a successful MCP call must emit a completed outcome event: {delta}"
    );
    assert!(
        !delta.contains("event: response.mcp_call.failed"),
        "a successful MCP call must not emit a failed outcome event: {delta}"
    );
    let added = delta.find("event: response.output_item.added").unwrap();
    let in_progress = delta.find("event: response.mcp_call.in_progress").unwrap();
    let completed = delta.find("event: response.mcp_call.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        added < in_progress && in_progress < completed && completed < done,
        "progress and outcome events must be ordered added -> in_progress -> completed -> done: {delta}"
    );
    let synthesized_mcp = delta.find("mcp_1").unwrap();
    let resumed_text = delta.find("response.output_text.delta").unwrap();
    assert!(
        synthesized_mcp < resumed_text,
        "synthesized tool activity must precede resumed model output: {delta}"
    );

    let final_message = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done"}]
    });
    let mut final_terminal = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {
                "id": "resp_second",
                "status": "completed",
                "output": [final_message.clone()],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            },
            "sequence_number": 2
        }),
    ));
    filter.on_response_body(&mut ctx, &mut final_terminal, false).unwrap();
    assert!(final_terminal.is_none(), "final terminal should be held until EOS");
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .accumulated_output
        .push(final_message);
    let mut final_eos = None;
    filter.on_response_body(&mut ctx, &mut final_eos, true).unwrap();
    let final_eos = String::from_utf8(final_eos.unwrap().to_vec()).unwrap();
    assert!(
        final_eos.contains("event: response.completed"),
        "final terminal should be emitted: {final_eos}"
    );
    assert!(
        final_eos.contains(r#""id":"resp_first""#),
        "terminal response ID should remain stable: {final_eos}"
    );
    assert!(
        final_eos.contains("mcp_1"),
        "terminal output should contain the accumulated tool result: {final_eos}"
    );
    assert!(
        final_eos.contains("msg_1"),
        "terminal output should contain the final message: {final_eos}"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.response_object["id"], "resp_first",
        "the persisted response object must use the client-visible logical ID"
    );
    assert_eq!(
        state.response_object["output"].as_array().map(Vec::len),
        Some(3),
        "the persisted response object must contain every logical-stream output item"
    );
}

#[tokio::test]
async fn logical_stream_synthesizes_missing_progress_for_local_and_model_declared_items() {
    // A resumed round carries three prior output items: a web_search_call the
    // model announced with `output_item.added` (but never a progress event), plus
    // an mcp_call and an isolated web_search_call the dispatch filters executed
    // locally. #276 must synthesize the missing tool-specific progress lifecycle
    // for every one of them, while never duplicating the `output_item.added` the
    // model already streamed for the model-declared search.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    filter.arm(&mut ctx);

    // Round 0: the model streams a web_search_call as an incremental item.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let web_search_call = json!({
        "type": "web_search_call",
        "id": "ws_model_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust"}
    });
    let mut model_item = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": web_search_call.clone(),
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_item, false).unwrap();
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the model's web_search_call was upserted in place; a locally
    // executed mcp_call and a web_search_call absent from any upstream stream
    // were appended after it.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![
        web_search_call,
        json!({"type": "mcp_call", "id": "mcp_local_1"}),
        json!({"type": "web_search_call", "id": "ws_local_2", "status": "completed"}),
    ];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let mut resumed_created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_b", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut resumed_created, false).unwrap();
    assert!(
        resumed_created.is_none(),
        "resumed lifecycle creation must be suppressed"
    );

    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_b",
            "output_index": 0,
            "content_index": 0,
            "delta": "done",
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();
    let delta = String::from_utf8(delta.unwrap().to_vec()).unwrap();
    assert!(
        delta.contains("mcp_local_1"),
        "the locally executed MCP call must be synthesized: {delta}"
    );
    assert!(
        delta.contains("ws_local_2"),
        "a web_search_call absent from the upstream stream must be synthesized: {delta}"
    );
    // Only the two fully-local items get a fresh `output_item.added`; the
    // model-declared search reuses the announcement the model already streamed.
    assert_eq!(
        delta.matches("event: response.output_item.added").count(),
        2,
        "only the two fully-local items should get a fresh output_item.added: {delta}"
    );
    // #276 (Finding 1): the model streamed `output_item.added` for ws_model_1 but
    // never its progress events, so the proxy must now fill in that missing
    // progress lifecycle (and a fresh done) rather than dropping it entirely.
    assert!(
        delta.contains("ws_model_1"),
        "a model-declared web_search_call must receive its missing progress lifecycle: {delta}"
    );
    // Both web searches (model-declared + isolated) emit the full leading
    // progress events and a completed outcome; every local item gets a done.
    assert_eq!(
        delta.matches("event: response.web_search_call.in_progress").count(),
        2,
        "both web searches must emit a leading in_progress event: {delta}"
    );
    assert_eq!(
        delta.matches("event: response.web_search_call.searching").count(),
        2,
        "both web searches must emit a searching event: {delta}"
    );
    assert_eq!(
        delta.matches("event: response.web_search_call.completed").count(),
        2,
        "both completed web searches must emit a completed outcome: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_call.in_progress") && delta.contains("event: response.mcp_call.completed"),
        "the synthesized MCP call must emit its progress and outcome events: {delta}"
    );
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        3,
        "every local tool item must receive a fresh output_item.done: {delta}"
    );

    // A second output-bearing event in the same round must not re-flush.
    let mut delta2 = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_b",
            "output_index": 0,
            "content_index": 0,
            "delta": "!",
            "sequence_number": 2
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta2, false).unwrap();
    let delta2 = String::from_utf8(delta2.unwrap().to_vec()).unwrap();
    assert!(
        !delta2.contains("mcp_local_1") && !delta2.contains("ws_local_2") && !delta2.contains("ws_model_1"),
        "local items must be synthesized only once per resumed round: {delta2}"
    );
}

#[tokio::test]
async fn logical_stream_suppresses_malformed_chunk_and_emits_terminal_error() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_first", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let mut malformed = Some(Bytes::from(
        "event: response.output_text.delta\ndata: {\"response_id\":\"resp_second\",bad}\n\n",
    ));
    filter.on_response_body(&mut ctx, &mut malformed, false).unwrap();
    assert!(
        malformed.is_none(),
        "a malformed logical-stream chunk must never bypass normalization"
    );

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        eos.contains("event: error"),
        "the logical stream should terminate with an SSE error: {eos}"
    );
    assert!(
        !eos.contains("resp_second"),
        "the malformed resumed response identity must not leak downstream: {eos}"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "parse-error streams must not be persisted"
    );
}

/// Arm a plain streaming logical-stream context (no hosted tools) for the given
/// filter, then feed a `response.created` opener. Returns the armed context.
fn arm_plain_stream(filter: &OpenaiStreamEventsFilter) -> praxis_filter::HttpFilterContext<'static> {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_budget", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    ctx
}

fn output_item_added_chunk(index: usize) -> Bytes {
    make_sse_chunk(
        "response.output_item.added",
        &json!({
            "item": {"type": "message", "id": format!("item_{index}"), "content": []},
            "output_index": index,
        }),
    )
}

#[tokio::test]
async fn accumulation_byte_budget_fails_closed_across_chunks() {
    // Each output_item.added frame is individually valid (well under
    // max_buffer_bytes), but their aggregate crosses the byte ceiling: the
    // stream must fail closed once the running total exceeds it (#556).
    let filter = make_filter_from("max_accumulated_bytes: 400");
    let mut ctx = arm_plain_stream(&filter);

    let mut failed = false;
    for i in 0..50 {
        let mut chunk = Some(output_item_added_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        if chunk.is_none() {
            failed = true;
            break;
        }
    }

    assert!(
        failed,
        "aggregate output-item bytes must eventually fail the stream closed"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_parse_error"),
        Some("true"),
        "budget overflow must be recorded as a stream parse error"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "budget-overflow streams must not be persisted"
    );
}

#[tokio::test]
async fn accumulation_item_count_budget_fails_closed() {
    // The item-count dimension trips independently of the byte ceiling: with a
    // generous byte cap, the third output item exceeds max_output_items: 2.
    let filter = make_filter_from("max_output_items: 2\nmax_accumulated_bytes: 67108864");
    let mut ctx = arm_plain_stream(&filter);

    for i in 0..2 {
        let mut chunk = Some(output_item_added_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(chunk.is_some(), "items within the count cap must pass through");
    }

    let mut overflow = Some(output_item_added_chunk(2));
    filter.on_response_body(&mut ctx, &mut overflow, false).unwrap();
    assert!(overflow.is_none(), "the item past the count cap must fail closed");
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "count-overflow streams must not be persisted"
    );
}

#[tokio::test]
async fn accumulation_budget_poisons_subsequent_terminal() {
    // Once tripped, the budget stays tripped: a later terminal event (which is
    // itself not charged) must not resurrect the stream and commit a success,
    // so the store never persists a poisoned response.
    let filter = make_filter_from("max_output_items: 1\nmax_accumulated_bytes: 67108864");
    let mut ctx = arm_plain_stream(&filter);

    let mut first = Some(output_item_added_chunk(0));
    filter.on_response_body(&mut ctx, &mut first, false).unwrap();
    assert!(first.is_some(), "the first item is within the count cap");

    let mut second = Some(output_item_added_chunk(1));
    filter.on_response_body(&mut ctx, &mut second, false).unwrap();
    assert!(second.is_none(), "the second item overflows the count cap");

    let mut completed = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {"id": "resp_budget", "status": "completed", "output": []},
            "sequence_number": 5
        }),
    ));
    filter.on_response_body(&mut ctx, &mut completed, false).unwrap();
    assert!(
        completed.is_none(),
        "a terminal event after budget overflow must stay suppressed"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "a poisoned stream must not be persisted even after a terminal event"
    );
}

/// A `response.web_search_call.in_progress` local-tool progress frame carrying a
/// distinct `item_id`. Each grows `emitted_output_items` via
/// `record_model_output_item`, so each must be charged against the byte budget.
fn web_search_progress_chunk(index: usize) -> Bytes {
    make_sse_chunk(
        "response.web_search_call.in_progress",
        &json!({"item_id": format!("ws_{index:06}"), "output_index": index}),
    )
}

/// A `response.function_call_arguments.delta` frame for a distinct tool call. The
/// per-call argument buffer is capped individually, but the number of distinct
/// keys is bounded only by the aggregate byte budget.
fn function_call_delta_chunk(index: usize) -> Bytes {
    make_sse_chunk(
        "response.function_call_arguments.delta",
        &json!({"item_id": format!("fc_{index:06}"), "output_index": index, "delta": "{\"q\":\"x\"}"}),
    )
}

/// A `response.output_item.done` frame that finalizes a fresh item (no prior
/// `output_item.added`), exercising the append branch of `handle_output_item_done`.
fn output_item_done_chunk(index: usize) -> Bytes {
    make_sse_chunk(
        "response.output_item.done",
        &json!({
            "item": {"type": "message", "id": format!("done_{index}"), "content": []},
            "output_index": index,
        }),
    )
}

#[tokio::test]
async fn accumulation_charges_local_tool_progress_events() {
    // Local-tool progress events (`response.web_search_call.*` etc.) grow the
    // `emitted_output_items` map without pushing an output item, so before the
    // charge they escaped the aggregate ceiling entirely. Each must now count
    // against the byte budget so distinct `item_id`s cannot exhaust memory (#556).
    let filter = make_filter_from("max_accumulated_bytes: 400");
    let mut ctx = arm_plain_stream(&filter);

    let mut failed = false;
    for i in 0..50 {
        let mut chunk = Some(web_search_progress_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        if chunk.is_none() {
            failed = true;
            break;
        }
    }

    assert!(
        failed,
        "aggregate local-tool progress bytes must eventually fail the stream closed"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_parse_error"),
        Some("true"),
        "budget overflow must be recorded as a stream parse error"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "budget-overflow streams must not be persisted"
    );
}

#[tokio::test]
async fn accumulation_charges_tool_call_argument_bytes() {
    // The tool-call argument accumulator — the dimension #556 named as the core
    // vuln, whose key count was unbounded — is charged through the same byte
    // budget: many distinct function-call deltas must fail the stream closed.
    let filter = make_filter_from("max_accumulated_bytes: 400");
    let mut ctx = arm_plain_stream(&filter);

    let mut failed = false;
    for i in 0..50 {
        let mut chunk = Some(function_call_delta_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        if chunk.is_none() {
            failed = true;
            break;
        }
    }

    assert!(failed, "aggregate tool-call argument bytes must fail the stream closed");
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "tool-call budget overflow streams must not be persisted"
    );
}

#[tokio::test]
async fn accumulation_count_charges_output_item_done() {
    // `output_item.done` can append a brand-new item (no prior `added`), so it
    // must bump the item counter too; a done-only stream cannot bypass the count
    // cap. With a generous byte cap, the third done item trips max_output_items: 2.
    let filter = make_filter_from("max_output_items: 2\nmax_accumulated_bytes: 67108864");
    let mut ctx = arm_plain_stream(&filter);

    for i in 0..2 {
        let mut chunk = Some(output_item_done_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(chunk.is_some(), "done items within the count cap must pass through");
    }

    let mut overflow = Some(output_item_done_chunk(2));
    filter.on_response_body(&mut ctx, &mut overflow, false).unwrap();
    assert!(overflow.is_none(), "the done item past the count cap must fail closed");
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "count-overflow streams must not be persisted"
    );
}

#[test]
fn default_budget_is_mandatory() {
    // #556 requires the aggregate budget to apply even when unconfigured. A filter
    // built from empty config must carry the default 64 MiB / 100,000 caps, not an
    // unbounded (opt-in) ceiling.
    let filter = make_filter();
    assert_eq!(
        filter.max_accumulated_bytes,
        64 * 1024 * 1024,
        "omitted max_accumulated_bytes must default to the mandatory 64 MiB ceiling"
    );
    assert_eq!(
        filter.max_output_items, 100_000,
        "omitted max_output_items must default to the mandatory 100,000 cap"
    );
}

#[tokio::test]
async fn accumulation_count_dedups_added_done_pair() {
    // A canonical `output_item.added` -> `output_item.done` pair for the SAME item
    // is one retained output item, not two. The count derives from the retained
    // output (which `done` replaces in place), so a matched pair must not trip
    // `max_output_items: 1`. Regression for the per-envelope double count, where
    // `added` and `done` each bumped a separate counter (#556 review finding 3).
    let filter = make_filter_from("max_output_items: 1\nmax_accumulated_bytes: 67108864");
    let mut ctx = arm_plain_stream(&filter);

    let mut added = Some(output_item_added_chunk(0));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();
    assert!(added.is_some(), "the added envelope is the single retained item");

    // Same id and output_index: `done` replaces the item in place, so the retained
    // count stays at one.
    let mut done = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "item": {"type": "message", "id": "item_0", "content": []},
            "output_index": 0,
        }),
    ));
    filter.on_response_body(&mut ctx, &mut done, false).unwrap();
    assert!(
        done.is_some(),
        "a matched added/done pair is one item and must not trip the count cap"
    );
    assert_ne!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "a canonical added/done pair must not fail the stream closed"
    );
}

#[tokio::test]
async fn repeated_function_call_done_reuses_one_output_item() {
    let filter = make_filter_from("max_accumulated_bytes: 65536");
    let mut ctx = arm_plain_stream(&filter);
    let mut added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "item": {"type": "function_call", "name": "n".repeat(5000), "arguments": "", "status": "in_progress"},
            "output_index": 0,
        }),
    ));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();
    assert!(added.is_some());

    for _ in 0..50 {
        let mut done = Some(make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": 0, "arguments": "{}"}),
        ));
        filter.on_response_body(&mut ctx, &mut done, false).unwrap();
        assert!(
            done.is_some(),
            "repeated completion must not clone the full output item"
        );
    }
    assert_ne!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
    assert!(
        state.tool_calls.is_empty(),
        "dispatch is selected after the round moves to accumulated output"
    );
}

#[tokio::test]
async fn coalesced_function_call_done_reuses_one_output_item() {
    let filter = make_filter_from("max_accumulated_bytes: 65536");
    let mut ctx = arm_plain_stream(&filter);
    let mut coalesced = Vec::new();
    coalesced.extend_from_slice(&make_sse_chunk(
        "response.output_item.added",
        &json!({
            "item": {"type": "function_call", "name": "n".repeat(5000), "arguments": "", "status": "in_progress"},
            "output_index": 0,
        }),
    ));
    for _ in 0..50 {
        coalesced.extend_from_slice(&make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": 0, "arguments": "{}"}),
        ));
    }
    let mut chunk = Some(Bytes::from(coalesced));
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(chunk.is_some(), "a coalesced batch must retain one completed item");
    assert_ne!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
}

#[test]
fn co_batched_native_completions_reuse_canonical_output_without_amplification() {
    let (filter, mut ctx) = make_armed_context_with_filter(make_filter_from("max_accumulated_bytes: 67108864"));
    let mut responses = ResponsesState::default();
    responses.apply_retained_payload_limit(1_048_576);
    ctx.extensions.insert(responses);

    let mut coalesced = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "n".repeat(65_536), "arguments": "", "status": "in_progress"}
        }),
    )
    .to_vec();
    for _ in 0..40 {
        coalesced.extend_from_slice(&make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": 0, "arguments": "{}"}),
        ));
    }

    let mut body = Some(Bytes::from(coalesced));
    let allocations = allocation_counter::measure(|| {
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    });
    let responses = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(body.is_some(), "repeated done events reuse one canonical output item");
    assert!(
        !responses.retained_payload_failed,
        "reusing the item stays within the retained payload budget"
    );
    assert!(
        responses.retained_payload_bytes_bounded(1_048_576).is_some(),
        "no completion clone may put retained state beyond the admitted budget"
    );
    assert_eq!(responses.output_items().len(), 1);
    assert_ne!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert!(
        allocations.bytes_max < 2 * 1_048_576,
        "repeated native done events must not amplify owned output: {}",
        allocations.bytes_max
    );
}

#[tokio::test]
async fn mismatched_announced_index_still_updates_the_real_output_item() {
    let filter = make_filter_from("max_accumulated_bytes: 65536");
    let mut ctx = arm_plain_stream(&filter);
    let mut coalesced = Vec::new();
    coalesced.extend_from_slice(&make_sse_chunk(
        "response.output_item.added",
        &json!({
            "item": {"type": "function_call", "name": "n".repeat(5000), "arguments": "", "status": "in_progress"},
            "output_index": 7,
        }),
    ));
    for _ in 0..50 {
        coalesced.extend_from_slice(&make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": 0, "arguments": "{}"}),
        ));
    }
    let mut chunk = Some(Bytes::from(coalesced));
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(chunk.is_some());
    assert_ne!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(state.output_items()[0]["arguments"], "{}");
}

#[tokio::test]
async fn accumulation_argument_done_without_item_is_incremental() {
    // Finding 2 (#556 re-review): the clone charge must not rescan the parsed-event
    // history for every completion. A coalesced chunk of many
    // `function_call_arguments.done` events with NO matching output item was the
    // O(N^2) worst case the old rescan hit (~1.9s for 60k events). Charging the clone
    // at commit makes each done O(1): it finds no item, clones nothing, and charges
    // nothing. A large such chunk must be processed without spurious overflow and
    // without retaining any tool call.
    let filter = make_filter_from("max_accumulated_bytes: 67108864\nmax_events: 100000");
    let mut ctx = arm_plain_stream(&filter);

    let mut coalesced = Vec::new();
    for _ in 0..20_000 {
        coalesced.extend_from_slice(&make_sse_chunk(
            "response.function_call_arguments.done",
            &json!({"output_index": 0, "arguments": "{}"}),
        ));
    }

    let mut chunk = Some(Bytes::from(coalesced));
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    // No matching output item ever existed, so nothing was cloned or charged as a
    // clone and the stream is not failed closed.
    assert_ne!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "dones without a matching output item must not spuriously overflow the byte budget"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "a done with no matching output item must retain no tool call"
    );
}

#[tokio::test]
async fn count_overflow_rejects_before_recording_local_tool_milestone() {
    // A chunk that overflows the item cap must fail closed *before* recording any
    // delivery milestone: otherwise a local tool whose progress streamed earlier in
    // the same chunk leaves a committed milestone that EOS recovery trusts, dropping
    // the executed tool from the client-visible stream. The count guard now runs
    // between accumulation and milestone recording (#556 review finding).
    let filter = make_filter_from("max_output_items: 1\nmax_accumulated_bytes: 67108864");
    let mut ctx = arm_plain_stream(&filter);

    // Round-0 item fills the single-item cap.
    let mut first = Some(output_item_added_chunk(0));
    filter.on_response_body(&mut ctx, &mut first, false).unwrap();
    assert!(first.is_some(), "the first item is within the count cap");

    // One chunk: a local-tool progress frame (which would record a milestone for
    // its `item_id`) followed by an `output_item.added` that overflows the count
    // cap. The whole chunk must be rejected atomically.
    let mut combined = Vec::new();
    combined.extend_from_slice(&web_search_progress_chunk(1));
    combined.extend_from_slice(&output_item_added_chunk(1));
    let mut chunk = Some(Bytes::from(combined));
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(chunk.is_none(), "the overflowing chunk must fail closed");
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "a count-overflow chunk must not be persisted"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        !state.emitted_output_items.contains_key("ws_000001"),
        "no local-tool milestone may be recorded for a chunk rejected on count overflow"
    );
}

#[tokio::test]
async fn accumulation_byte_budget_spans_irr_rounds() {
    // The aggregate byte budget is request-wide: bytes charged in an earlier IRR
    // round persist across the per-round re-arm, so a later round cannot reset the
    // counter and accumulate unbounded state while no single round trips the cap.
    // Regression for the per-round reset (#556 review finding 1).

    // Measure what one output item charges so the cap can admit two items but
    // reject the third, independent of the exact wire size.
    let probe = make_filter_from("max_accumulated_bytes: 67108864");
    let mut probe_ctx = arm_plain_stream(&probe);
    let mut probe_chunk = Some(output_item_added_chunk(0));
    probe.on_response_body(&mut probe_ctx, &mut probe_chunk, false).unwrap();
    let per_item = probe_ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .stream_accumulated_bytes;
    assert!(per_item > 1, "streaming an output item must charge the byte budget");

    // Two items fit; the third does not (2*per_item <= cap < 3*per_item).
    let cap = per_item * 2 + 1;
    let filter = make_filter_from(&format!("max_accumulated_bytes: {cap}"));
    let mut ctx = arm_plain_stream(&filter);

    for i in 0..2 {
        let mut chunk = Some(output_item_added_chunk(i));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(chunk.is_some(), "round-0 items stay under the aggregate cap");
    }

    // Re-arm as the IRR runner would before the next round's request phase. A
    // per-round counter would reset to zero here; the request-wide one must not.
    filter.arm(&mut ctx);
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().stream_accumulated_bytes,
        per_item * 2,
        "the request-wide byte budget must survive the per-round re-arm"
    );

    // Round 1: one more item crosses the cap because round-0 bytes still count.
    let mut overflow = Some(output_item_added_chunk(2));
    filter.on_response_body(&mut ctx, &mut overflow, false).unwrap();
    assert!(
        overflow.is_none(),
        "bytes charged in round 0 must carry into round 1 and fail the stream closed"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "a stream that overflows the request-wide byte budget must not be persisted"
    );
}

#[test]
fn lowered_completion_snapshots_count_toward_both_budgets() {
    for aggregate_limit in [None, Some(65_536)] {
        let stream_limit = if aggregate_limit.is_some() { 1_048_576 } else { 65_536 };
        let (filter, mut ctx) =
            make_armed_context_with_filter(make_filter_from(&format!("max_accumulated_bytes: {stream_limit}")));
        let mut responses = ResponsesState::default();
        responses.client_tool_lowering.insert(
            "agentic_ns__fs__read".to_owned(),
            LoweredClientTool {
                original_name: "read".to_owned(),
                namespace: Some("fs".to_owned()),
                restore: ClientToolRestore::Namespace,
            },
        );
        ctx.extensions.insert(responses);

        let mut added = Some(make_sse_chunk(
            "response.output_item.added",
            &json!({
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "name": "agentic_ns__fs__read",
                    "call_id": "c1",
                    "id": "fc_1",
                    "arguments": "",
                    "status": "in_progress",
                    "padding": "x".repeat(8_192)
                }
            }),
        ));
        filter.on_response_body(&mut ctx, &mut added, false).unwrap();
        assert!(added.is_some(), "the large item itself must fit");
        if let Some(limit) = aggregate_limit {
            ctx.extensions
                .get_mut::<ResponsesState>()
                .unwrap()
                .apply_retained_payload_limit(limit);
        }

        // Each small frame would clone the large completed item into the
        // restoration plan. All snapshots coexist until the chunk is planned.
        let mut frames = Vec::new();
        for _ in 0..50 {
            frames.extend_from_slice(&make_sse_chunk(
                "response.function_call_arguments.done",
                &json!({"output_index": 0, "item_id": "fc_1", "arguments": "{}"}),
            ));
        }
        let mut body = Some(Bytes::from(frames));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(body.is_none(), "the snapshot copies must trip the byte ceiling");
        if aggregate_limit.is_some() {
            let responses = ctx.extensions.get::<ResponsesState>().unwrap();
            assert!(responses.retained_payload_failed);
            assert_eq!(
                responses.retained_external_payload_bytes, 0,
                "dropped snapshots must release their charge"
            );
        } else {
            assert_eq!(ctx.get_metadata("responses.stream_parse_error"), Some("true"));
        }
        assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    }
}

#[tokio::test]
async fn accumulation_terminal_snapshot_charged_against_byte_budget() {
    // A terminal `response.completed` snapshots the full accumulated output plus
    // usage into `response_object` and is retained again as the deferred terminal,
    // so a terminal frame that alone exceeds the byte ceiling must fail closed
    // rather than slip through just because it fits `max_buffer_bytes`. Regression
    // for the uncharged terminal snapshot (#556 review finding 2).
    let filter = make_filter_from("max_accumulated_bytes: 500\nmax_output_items: 100000");
    let mut ctx = arm_plain_stream(&filter);

    let big_text = "x".repeat(2000);
    let mut completed = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {
                "id": "resp_big_terminal",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": "m1",
                    "content": [{"type": "output_text", "text": big_text}]
                }]
            },
            "sequence_number": 9
        }),
    ));
    filter.on_response_body(&mut ctx, &mut completed, false).unwrap();
    assert!(
        completed.is_none(),
        "a terminal snapshot larger than the byte ceiling must fail the stream closed"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_parse_error"),
        Some("true"),
        "an over-cap terminal snapshot must be recorded as a stream parse error"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "an over-cap terminal snapshot must not be persisted"
    );
}

#[tokio::test]
async fn committed_stream_budget_overflow_emits_one_error_without_completion_or_done() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    let mut response_state = ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    }));
    response_state.apply_retained_payload_limit(4_096);
    response_state.store_persist_armed = true;
    ctx.extensions.insert(response_state);
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_budget", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    assert!(created.is_some(), "the logical stream is committed before overflow");

    let mut offending = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {
                "id": "call_too_large",
                "type": "function_call",
                "name": "tool",
                "arguments": "x".repeat(5_000),
                "status": "completed"
            }
        }),
    ));
    filter.on_response_body(&mut ctx, &mut offending, false).unwrap();
    assert!(offending.is_none(), "the offending chunk must be suppressed");

    let mut after_failure = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"id": "call_after_failure", "type": "function_call", "name": "tool", "arguments": "{}"}
        }),
    ));
    filter.on_response_body(&mut ctx, &mut after_failure, false).unwrap();
    assert!(
        after_failure.is_none(),
        "later input must not revive the poisoned stream"
    );

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let terminal = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert_eq!(terminal.matches("event: error").count(), 1, "{terminal}");
    assert!(!terminal.contains("response.completed"), "{terminal}");
    assert!(!terminal.contains("[DONE]"), "{terminal}");
    assert_eq!(ctx.filter_results["openai_agentic_loop"].get("action"), Some("done"));
    let response_state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(response_state.retained_payload_failed);
    assert!(response_state.tool_calls.is_empty());
    assert!(response_state.accumulated_output.is_empty());
    assert!(!response_state.store_persist_armed);
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert_eq!(
        ctx.get_filter_state::<StreamEventsState>()
            .unwrap()
            .frame_parser
            .retained_bytes(),
        0,
        "the offending chunk must not remain in parser scratch"
    );
}

#[tokio::test]
async fn transient_frame_and_event_ownership_is_admitted_before_parsing() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    let response_state = ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    }));
    let response_bytes = response_state.retained_payload_bytes().unwrap();
    ctx.extensions.insert(response_state);
    filter.arm(&mut ctx);

    let chunk = make_sse_chunk(
        "response.output_text.delta",
        &json!({"response_id": "resp_budget", "delta": "hello"}),
    );
    let split_at = chunk.len() - 1;
    let partial_chunk = chunk.slice(..split_at);
    let terminating_chunk = chunk.slice(split_at..);

    let mut partial = Some(partial_chunk);
    filter.on_response_body(&mut ctx, &mut partial, false).unwrap();
    assert!(partial.is_none(), "the incomplete frame must remain buffered");

    let parser_state = ctx.get_filter_state::<StreamEventsState>().unwrap();
    let local_bytes = parser_state.retained_payload_bytes().unwrap();
    let parser_scratch_bytes = parser_state.frame_parser.retained_bytes();
    let persistent_local_bytes = local_bytes - parser_scratch_bytes;
    assert!(local_bytes > 0, "the partial frame must be retained");

    let mut parser = SseFrameParser::new(65_536);
    let frames = parser.parse_chunk(&chunk).unwrap();
    let frame_bytes = super::retained_frame_payload_bytes(&frames).unwrap();
    let event_construction_bytes = frames
        .iter()
        .filter(|frame| frame.data != b"[DONE]")
        .map(|frame| frame.data.len() + frame.event_type.as_ref().map_or(0, String::len))
        .sum::<usize>();
    let parser_projection_peak = response_bytes + local_bytes + terminating_chunk.len() * 4;
    let frame_and_event_ownership_peak =
        response_bytes + persistent_local_bytes + frame_bytes + event_construction_bytes;
    let limit = parser_projection_peak + 1;
    assert!(
        limit < frame_and_event_ownership_peak,
        "the cap must admit parser projection but reject simultaneous frame/event ownership"
    );
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(limit);

    let mut body = Some(terminating_chunk);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert!(body.is_none(), "the frame/event ownership peak must be rejected");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[tokio::test]
async fn parser_projection_is_reserved_without_retaining_framework_chunk() {
    let (filter, mut ctx) = make_armed_context();
    let mut response_state = ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    }));
    let retained_before = response_state.retained_payload_bytes().unwrap();
    let chunk = Bytes::from_static(b": framework-owned chunk\n");
    let exact_limit = retained_before + chunk.len() * 4;
    response_state.apply_retained_payload_limit(exact_limit);
    ctx.extensions.insert(response_state);

    // The framework chunk itself is not retained, while the parser's transient
    // line/data projection is admitted before parsing and released afterward.
    let mut body = Some(chunk);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.retained_payload_failed);
    assert_eq!(state.retained_payload_bytes().unwrap(), retained_before);
}

#[test]
fn split_data_line_rejects_before_copying_prior_chunk_into_frame() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState::from_request_body(json!({
        "model": "test-model", "input": "hello", "stream": true
    }));
    responses.apply_retained_payload_limit(65_536);
    ctx.extensions.insert(responses);

    let line = format!("data: {}", "x".repeat(57_000));
    for segment in line.as_bytes().chunks(256) {
        let mut body = Some(Bytes::copy_from_slice(segment));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(
            !ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed,
            "the unterminated line fits while it has a single parser owner"
        );
    }

    let mut newline = Some(Bytes::from_static(b"\n"));
    filter.on_response_body(&mut ctx, &mut newline, false).unwrap();
    assert!(newline.is_none(), "the duplicated line must be suppressed");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert_eq!(
        ctx.get_filter_state::<StreamEventsState>()
            .unwrap()
            .frame_parser
            .retained_bytes(),
        0,
        "overflow cleanup must drop the pending line"
    );
}

#[test]
fn multiline_data_rejects_before_reallocating_prior_frame_buffer() {
    let (filter, mut ctx) = make_armed_context();
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model", "input": "hello", "stream": true
    })));

    let mut first_line = Some(Bytes::from(format!("data: {}\n", "x".repeat(30_000))));
    filter.on_response_body(&mut ctx, &mut first_line, false).unwrap();
    let response = ctx.extensions.get::<ResponsesState>().unwrap();
    let shared = response.retained_payload_bytes().unwrap() - response.retained_stream_parser_bytes;
    let local = ctx
        .get_filter_state::<StreamEventsState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    let second_line = Bytes::from_static(b"data: x\n");
    let limit = shared + local + second_line.len() * 4 + 30_000 - 1;
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(limit);

    let mut body = Some(second_line);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        body.is_none(),
        "reallocating the existing data buffer must be preflighted"
    );
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn crlf_does_not_hide_carried_data_buffer_copy() {
    for split_crlf in [false, true] {
        let (filter, mut ctx) = make_armed_context();
        ctx.extensions.insert(ResponsesState::from_request_body(json!({
            "model": "test-model", "input": "hello", "stream": true
        })));

        let mut first_line = Some(Bytes::from(format!("data: {}\n", "x".repeat(30_000))));
        filter.on_response_body(&mut ctx, &mut first_line, false).unwrap();
        if split_crlf {
            let mut event_line = Some(Bytes::from_static(b"event: ping\r"));
            filter.on_response_body(&mut ctx, &mut event_line, false).unwrap();
        }

        let suffix = if split_crlf {
            Bytes::from_static(b"\ndata: x\n")
        } else {
            Bytes::from_static(b"event: ping\r\ndata: x\n")
        };
        let response = ctx.extensions.get::<ResponsesState>().unwrap();
        let shared = response.retained_payload_bytes().unwrap() - response.retained_stream_parser_bytes;
        let local = ctx
            .get_filter_state::<StreamEventsState>()
            .unwrap()
            .retained_payload_bytes()
            .unwrap();
        let limit = shared + local + suffix.len() * 4 + 30_000 - 1;
        ctx.extensions
            .get_mut::<ResponsesState>()
            .unwrap()
            .apply_retained_payload_limit(limit);

        let mut body = Some(suffix);
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
        assert!(
            ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed,
            "carried data must stay live across CRLF, split={split_crlf}"
        );
        assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    }
}

#[test]
fn completed_event_field_needs_no_second_copy_on_blank_line() {
    let (filter, mut ctx) = make_armed_context();
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model", "input": "hello", "stream": true
    })));
    let mut field = Some(Bytes::from_static(b"event: example\n"));
    filter.on_response_body(&mut ctx, &mut field, false).unwrap();

    let response = ctx.extensions.get::<ResponsesState>().unwrap();
    let shared = response.retained_payload_bytes().unwrap() - response.retained_stream_parser_bytes;
    let local = ctx
        .get_filter_state::<StreamEventsState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(shared + local + 4);

    let mut blank = Some(Bytes::from_static(b"\n"));
    filter.on_response_body(&mut ctx, &mut blank, false).unwrap();
    assert!(!ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[tokio::test]
async fn logical_output_is_admitted_before_allocation() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    let item = json!({
        "type": "mcp_call",
        "id": "call_budget",
        "status": "completed",
        "name": "tool",
        "arguments": "x".repeat(4_096)
    });
    let mut response_state = ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    }));
    response_state.logical_stream_response_id = Some("resp_budget".to_owned());
    response_state.accumulated_output = vec![item];
    response_state
        .locally_executed_output_items
        .insert("call_budget".to_owned());
    let current = response_state.retained_payload_bytes().unwrap();
    ctx.extensions.insert(response_state);
    filter.arm(&mut ctx);

    let chunk = make_sse_chunk(
        "response.output_text.delta",
        &json!({"response_id": "resp_budget", "output_index": 1, "delta": "ok"}),
    );
    let mut parser = SseFrameParser::new(65_536);
    let frames = parser.parse_chunk(&chunk).unwrap();
    let event = crate::openai::sse::responses::ResponsesEvent::from_frame(&frames[0]).unwrap();
    let output_upper_bound = super::logical_output_upper_bound(&ctx, false, &[event]).unwrap();
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(current + output_upper_bound - 1);

    let mut body = Some(chunk);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert!(body.is_none(), "the logical output must be rejected before allocation");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[tokio::test]
async fn commit_preflight_accounts_event_and_response_state_owners() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let chunk = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {
                "id": "call_budget",
                "type": "function_call",
                "name": "tool",
                "arguments": "x".repeat(4_096),
                "status": "completed"
            }
        }),
    );
    let mut parser = SseFrameParser::new(65_536);
    let frames = parser.parse_chunk(&chunk).unwrap();
    let event = crate::openai::sse::responses::ResponsesEvent::from_frame(&frames[0]).unwrap();
    let events = [event];
    let frame_bytes = super::retained_frame_payload_bytes(&frames).unwrap();
    let event_bytes = super::retained_event_payload_bytes(&events[0]).unwrap();
    let projected_state_bytes = super::projected_responses_state_clone_bytes(&ctx, &events);
    let output_upper_bound = super::logical_output_upper_bound(&ctx, false, &events).unwrap();
    let current = ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    let limit = current + frame_bytes + event_bytes + projected_state_bytes.unwrap() + output_upper_bound - 1;
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .apply_retained_payload_limit(limit);

    let mut body = Some(chunk);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert!(body.is_none(), "the commit must reject all simultaneous owners");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn commit_projection_charges_argument_delta_and_terminal_usage_owners() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    let mut response_state = ResponsesState::default();
    response_state.usage = json!({"input_tokens": 10});
    ctx.extensions.insert(response_state);

    let delta = crate::openai::sse::responses::ResponsesEvent::FunctionCallArgumentsDelta(json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "call_1",
        "delta": "x".repeat(1_024)
    }));
    let delta_projection = super::projected_responses_state_clone_bytes(&ctx, &[delta]).unwrap();
    assert!(delta_projection >= 1_024 + "item:call_1".len());
    let unkeyed_delta = crate::openai::sse::responses::ResponsesEvent::FunctionCallArgumentsDelta(json!({
        "type": "response.function_call_arguments.delta",
        "delta": "ignored"
    }));
    assert!(super::projected_responses_state_clone_bytes(&ctx, &[unkeyed_delta]).is_some());

    let response = json!({
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "call_1",
            "name": "tool",
            "arguments": "x".repeat(1_024),
            "status": "completed"
        }],
        "usage": {"input_tokens": 1}
    });
    let response_bytes = crate::openai::responses::state::retained_json_bytes(&response).unwrap();
    let old_usage_bytes =
        crate::openai::responses::state::retained_json_bytes(&ctx.extensions.get::<ResponsesState>().unwrap().usage)
            .unwrap();
    let new_usage_bytes = crate::openai::responses::state::retained_json_bytes(&response["usage"]).unwrap();
    let terminal = crate::openai::sse::responses::ResponsesEvent::ResponseCompleted(json!({
        "type": "response.completed",
        "response": response.clone()
    }));
    let terminal_projection = super::projected_responses_state_clone_bytes(&ctx, &[terminal]).unwrap();
    assert_eq!(terminal_projection, (old_usage_bytes + new_usage_bytes) * 2);
    assert!(
        terminal_projection < response_bytes,
        "the response tree moves from the parsed event"
    );
}

#[test]
fn commit_projection_charges_bare_terminal_usage_clones() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.extensions.insert(ResponsesState::default());
    let response = json!({
        "type": "response.completed",
        "id": "resp_bare_usage",
        "output": [],
        "usage": {"large_detail": "x".repeat(64 * 1024)}
    });
    let response_bytes = crate::openai::responses::state::retained_json_bytes(&response).unwrap();
    let usage_bytes = crate::openai::responses::state::retained_json_bytes(&response["usage"]).unwrap();
    let event = crate::openai::sse::responses::ResponsesEvent::ResponseCompleted(response);

    let projection = super::projected_responses_state_clone_bytes(&ctx, &[event]).unwrap();
    assert!(
        projection >= response_bytes + usage_bytes * 2,
        "the bare fallback clones the response and usage merging retains two more owners"
    );
}

/// Record execution provenance for every item currently in `accumulated_output`,
/// mirroring what a dispatch filter (`openai_mcp_dispatch`, `openai_web_search`)
/// records when it actually executes a tool. Tests that seed `accumulated_output`
/// with genuinely executed items call this so synthesis is not suppressed by the
/// provenance gate; the fabricated-lifecycle regression test deliberately does
/// not, leaving its ghost placeholder unrecorded.
fn mark_accumulated_output_executed(state: &mut ResponsesState) {
    let executed: Vec<String> = state
        .accumulated_output
        .iter()
        .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
        .collect();
    state.locally_executed_output_items.extend(executed);
}

/// Arm a logical-stream filter, run a bare round 0 that hands off to a dispatch
/// loop, then stage `accumulated` for a resumed round 1. Returns the armed
/// filter and context positioned to process round 1 response events.
async fn arm_resumed_round_with_accumulated(
    loop_filter: &'static str,
    accumulated: Vec<serde_json::Value>,
) -> (OpenaiStreamEventsFilter, praxis_filter::HttpFilterContext<'static>) {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    ctx.filter_results
        .entry(loop_filter)
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = accumulated;
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove(loop_filter);
    filter.arm(&mut ctx);

    (filter, ctx)
}

/// Drive one resumed model text delta and return the normalized SSE it produced.
fn resumed_text_delta(filter: &dyn HttpFilter, ctx: &mut praxis_filter::HttpFilterContext<'_>) -> String {
    let mut resumed_created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_b", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(ctx, &mut resumed_created, false).unwrap();
    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_b",
            "output_index": 0,
            "content_index": 0,
            "delta": "x",
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(ctx, &mut delta, false).unwrap();
    String::from_utf8(delta.unwrap().to_vec()).unwrap()
}

#[tokio::test]
async fn logical_stream_failed_mcp_call_emits_failed_outcome_event() {
    // #276: a locally executed MCP call that carries an `error` must surface a
    // `response.mcp_call.failed` outcome event, never a `completed` one.
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_mcp_dispatch",
        vec![json!({
            "type": "mcp_call",
            "id": "mcp_fail_1",
            "error": "connection refused",
        })],
    )
    .await;

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.mcp_call.in_progress"),
        "a failed MCP call still emits an in_progress event first: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_call.failed"),
        "an MCP call carrying an error must emit a failed outcome event: {delta}"
    );
    assert!(
        !delta.contains("event: response.mcp_call.completed"),
        "a failed MCP call must not emit a completed outcome event: {delta}"
    );
    let added = delta.find("event: response.output_item.added").unwrap();
    let failed = delta.find("event: response.mcp_call.failed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        added < failed && failed < done,
        "outcome must be ordered added -> failed -> done: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_successful_mcp_list_tools_emits_lifecycle_events() {
    // Locally generated deferred listings must surface as incremental
    // added / in_progress / completed / done events, not only the final
    // response snapshot.
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_mcp_dispatch",
        vec![json!({
            "type": "mcp_list_tools",
            "id": "mcpl_1",
            "server_label": "weather",
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        })],
    )
    .await;

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.output_item.added") && delta.contains("event: response.output_item.done"),
        "a locally generated listing must surface as incremental output-item events: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_list_tools.in_progress"),
        "a successful listing must emit an in_progress progress event: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_list_tools.completed"),
        "a successful listing must emit a completed outcome event: {delta}"
    );
    assert!(
        !delta.contains("event: response.mcp_list_tools.failed"),
        "a successful listing must not emit a failed outcome event: {delta}"
    );
    let added = delta.find("event: response.output_item.added").unwrap();
    let in_progress = delta.find("event: response.mcp_list_tools.in_progress").unwrap();
    let completed = delta.find("event: response.mcp_list_tools.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        added < in_progress && in_progress < completed && completed < done,
        "listing lifecycle must be ordered added -> in_progress -> completed -> done: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_flushes_index_zero_local_item_on_iteration_zero_resume() {
    // Regression (PR #1029, Finding #2): an MCP approval resume executes the
    // approved tool during `on_request_body`, before any inference round, leaving
    // the local `mcp_call` at `accumulated_output[0]` with `iteration` still 0. The
    // resumed model stream must announce that index-0 item ahead of the model's own
    // output (shifted to index 1); the earlier flush gate only fired at
    // `iteration > 0`, so index 0 was never announced and a client stream
    // accumulator saw index 1 with no index 0 and panicked.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    // The approved MCP call the dispatch filter executed at request time, before
    // the first inference round: index 0 in accumulated_output, iteration still 0.
    {
        let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        state.accumulated_output = vec![json!({"type": "mcp_call", "id": "mcp_resumed_0"})];
        mark_accumulated_output_executed(state);
        assert_eq!(
            state.iteration, 0,
            "an approval resume runs its first model round at iteration 0"
        );
    }

    // arm() captures output_index_offset = 1 from the pre-seeded accumulated_output.
    // Unit tests cannot construct the IRR-owned `IterationState`; arm directly
    // after setting up the state that would enter the step.
    filter.arm(&mut ctx);

    // The first (and only) logical response.created must be forwarded — unlike a
    // resumed round at iteration > 0, iteration 0 has no earlier lifecycle to dedup.
    // The local item must NOT be flushed ahead of it.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_resume", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    let created = String::from_utf8(created.unwrap().to_vec()).unwrap();
    assert!(
        created.contains("event: response.created"),
        "the first lifecycle creation must reach the client at iteration 0: {created}"
    );
    assert!(
        !created.contains("mcp_resumed_0"),
        "the local item must be announced after response.created, not before it: {created}"
    );

    // The model's first content event carries output_index 0 in its own stream; it
    // must be shifted to index 1, with the index-0 mcp_call announced ahead of it.
    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_resume",
            "output_index": 0,
            "content_index": 0,
            "delta": "x",
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();
    let delta = String::from_utf8(delta.unwrap().to_vec()).unwrap();

    assert!(
        delta.contains("event: response.output_item.added") && delta.contains("mcp_resumed_0"),
        "the index-0 local mcp_call must be synthesized on the resumed stream: {delta}"
    );
    assert!(
        delta.contains("event: response.mcp_call.in_progress") && delta.contains("event: response.mcp_call.completed"),
        "the synthesized MCP call must carry its progress lifecycle: {delta}"
    );
    let local_item = delta.find(r#""output_index":0"#).unwrap();
    let model_output = delta.find(r#""output_index":1"#).unwrap();
    assert!(
        local_item < model_output,
        "the index-0 local item must precede the model output shifted to index 1: {delta}"
    );
}

#[test]
fn is_local_tool_item_recognizes_mcp_list_tools() {
    // #1022: a successful discovery listing seeded by `openai_mcp_tool_resolve` is a
    // locally generated item the model backend never streams, so the logical stream
    // must own its lifecycle synthesis.
    assert!(
        super::is_local_tool_item(&json!({"type": "mcp_list_tools", "id": "mcpl_1"})),
        "mcp_list_tools must be treated as a locally generated tool item"
    );
    assert!(
        !super::is_local_tool_item(&json!({"type": "message", "id": "msg_1"})),
        "model message output is not a local tool item"
    );
}

#[test]
fn expected_phase_events_covers_mcp_list_tools() {
    // #1022: a successful discovery item owes exactly in_progress then completed;
    // failure takes the separate `mcp_list_tools.failed` path in
    // `openai_mcp_tool_resolve` (#320) and never reaches synthesis here.
    let item = json!({"type": "mcp_list_tools", "id": "mcpl_1", "error": null});
    assert_eq!(
        super::expected_phase_events(&item),
        vec![
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
        ],
        "mcp_list_tools progresses in_progress -> completed"
    );
}

#[tokio::test]
async fn logical_stream_synthesizes_mcp_list_tools_discovery_lifecycle_at_iteration_zero() {
    // #1022: `openai_mcp_tool_resolve` resolves the MCP `tools/list` during
    // `on_request_body`, before any inference round, seeding one `mcp_list_tools`
    // item at `accumulated_output[0]` with `iteration` still 0. The logical stream
    // must synthesize its full lifecycle — output_item.added ->
    // mcp_list_tools.in_progress -> mcp_list_tools.completed -> output_item.done —
    // ahead of the model output (shifted to index 1).
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    // The discovery listing the resolve filter seeded at request time: index 0 in
    // accumulated_output, iteration still 0, recorded as locally executed.
    {
        let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
        state.accumulated_output = vec![json!({
            "type": "mcp_list_tools",
            "id": "mcpl_0",
            "server_label": "weather",
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "error": null,
        })];
        mark_accumulated_output_executed(state);
        assert_eq!(state.iteration, 0, "discovery resolves before the first model round");
    }

    // arm() captures output_index_offset = 1 from the pre-seeded accumulated_output.
    filter.arm(&mut ctx);

    // The discovery item must be announced after response.created, not before it.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_discovery", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    let created = String::from_utf8(created.unwrap().to_vec()).unwrap();
    assert!(
        created.contains("event: response.created") && !created.contains("mcpl_0"),
        "the discovery item must be announced after response.created, not before it: {created}"
    );

    // The model's first content event carries output_index 0 in its own stream and
    // must be shifted to index 1, with the index-0 discovery item ahead of it.
    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_discovery",
            "output_index": 0,
            "content_index": 0,
            "delta": "x",
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();
    let delta = String::from_utf8(delta.unwrap().to_vec()).unwrap();

    // The full canonical discovery lifecycle must be synthesized, in order.
    let added = delta
        .find("event: response.output_item.added")
        .expect("output_item.added synthesized");
    let in_progress = delta
        .find("event: response.mcp_list_tools.in_progress")
        .expect("mcp_list_tools.in_progress synthesized");
    let completed = delta
        .find("event: response.mcp_list_tools.completed")
        .expect("mcp_list_tools.completed synthesized");
    let done = delta
        .find("event: response.output_item.done")
        .expect("output_item.done synthesized");
    assert!(
        added < in_progress && in_progress < completed && completed < done,
        "events must be ordered added -> in_progress -> completed -> done: {delta}"
    );
    assert!(
        delta.contains("mcpl_0") && delta.contains(r#""server_label":"weather""#),
        "the synthesized item must carry the discovery listing: {delta}"
    );
    assert!(
        !delta.contains("event: response.mcp_list_tools.failed"),
        "a successful discovery must not emit a failed event: {delta}"
    );

    // The discovery item occupies index 0; the model output is shifted to index 1.
    let local_item = delta.find(r#""output_index":0"#).unwrap();
    let model_output = delta.find(r#""output_index":1"#).unwrap();
    assert!(
        local_item < model_output,
        "the index-0 discovery item must precede the model output at index 1: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_forwards_backend_native_mcp_list_tools_lifecycle() {
    // #1022 regression: a *deferred* MCP entry (`defer_loading: true`, or one lacking a
    // `server_url`) is passed through unresolved by `openai_mcp_tool_resolve`, so the
    // model backend performs `tools/list` itself and natively streams the discovery
    // lifecycle for an `mcp_list_tools` item that was NEVER seeded locally (its id is
    // absent from `locally_executed_output_items`). The logical stream must forward the
    // backend's real lifecycle untouched — its terminal `response.output_item.done` in
    // particular must not be suppressed as premature — and must not synthesize a
    // duplicate lifecycle. Before recognizing `response.mcp_list_tools.*` as an in-band
    // progress event, the completed phase went unrecorded, so `is_premature_local_tool_done`
    // dropped the native `done` and left the client with an unterminated output item.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    // Deferred passthrough: the backend resolves the listing, so nothing is seeded
    // locally — accumulated_output is empty and no id is recorded as locally executed.
    {
        let state = ctx.extensions.get::<ResponsesState>().unwrap();
        assert!(state.accumulated_output.is_empty(), "no locally seeded listing");
        assert!(
            state.locally_executed_output_items.is_empty(),
            "the native listing id is not recorded as locally executed"
        );
    }

    // arm() captures output_index_offset = 0 (nothing pre-seeded).
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_native", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let listing = json!({
        "id": "mcpl_native",
        "type": "mcp_list_tools",
        "server_label": "weather",
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "error": null,
    });

    // The backend natively streams the full discovery lifecycle in canonical order.
    let native_events = [
        (
            "response.output_item.added",
            json!({"output_index": 0, "item": listing.clone(), "sequence_number": 1}),
        ),
        (
            "response.mcp_list_tools.in_progress",
            json!({"output_index": 0, "item_id": "mcpl_native", "sequence_number": 2}),
        ),
        (
            "response.mcp_list_tools.completed",
            json!({"output_index": 0, "item_id": "mcpl_native", "sequence_number": 3}),
        ),
        (
            "response.output_item.done",
            json!({"output_index": 0, "item": listing, "sequence_number": 4}),
        ),
    ];
    let mut forwarded = String::new();
    for (event_type, payload) in native_events {
        let mut chunk = Some(make_sse_chunk(event_type, &payload));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        if let Some(bytes) = chunk {
            forwarded.push_str(core::str::from_utf8(&bytes).unwrap());
        }
    }

    // The backend's real terminal done must be forwarded, not suppressed as premature.
    assert_eq!(
        forwarded.matches("event: response.output_item.done").count(),
        1,
        "the backend-native output_item.done must be forwarded exactly once, not suppressed: {forwarded}"
    );
    // No synthesized duplicate: the item is not locally executed, so the flush skips it.
    assert_eq!(
        forwarded.matches("event: response.output_item.added").count(),
        1,
        "no duplicate output_item.added synthesized for a backend-native listing: {forwarded}"
    );
    assert!(
        forwarded.contains("event: response.mcp_list_tools.in_progress")
            && forwarded.contains("event: response.mcp_list_tools.completed"),
        "the backend's native progress events must pass through: {forwarded}"
    );
}

#[tokio::test]
async fn logical_stream_forwards_backend_native_mcp_list_tools_failed_lifecycle() {
    // #1022 regression (failure counterpart): a deferred MCP entry the backend
    // resolves natively can also *fail* its `tools/list`, streaming
    // `response.mcp_list_tools.failed` on an item carrying a non-null `error`. The
    // listing lacks local provenance (its id is absent from
    // `locally_executed_output_items`), so no synthesized finalizer can replace a
    // dropped `done`. `expected_phase_events` must select the `failed` terminal
    // phase from the item's error — mirroring `mcp_call` — so the backend's real
    // `output_item.done` is forwarded, not suppressed as premature. Before this,
    // the expected terminal was always `completed`, which a failed stream never
    // reaches, so the real `done` was dropped and the item left unterminated.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    // Deferred passthrough: the backend resolves the listing, so nothing is seeded
    // locally — accumulated_output is empty and no id is recorded as locally executed.
    {
        let state = ctx.extensions.get::<ResponsesState>().unwrap();
        assert!(state.accumulated_output.is_empty(), "no locally seeded listing");
        assert!(
            state.locally_executed_output_items.is_empty(),
            "the native listing id is not recorded as locally executed"
        );
    }

    // arm() captures output_index_offset = 0 (nothing pre-seeded).
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_native_fail", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // The failed listing surfaces a non-null `error` on the item, as the backend does.
    let listing = json!({
        "id": "mcpl_native_fail",
        "type": "mcp_list_tools",
        "server_label": "weather",
        "tools": [],
        "error": {"type": "mcp_error", "message": "tools/list failed"},
    });

    // The backend natively streams the full discovery lifecycle, terminating in failed.
    let native_events = [
        (
            "response.output_item.added",
            json!({"output_index": 0, "item": listing.clone(), "sequence_number": 1}),
        ),
        (
            "response.mcp_list_tools.in_progress",
            json!({"output_index": 0, "item_id": "mcpl_native_fail", "sequence_number": 2}),
        ),
        (
            "response.mcp_list_tools.failed",
            json!({"output_index": 0, "item_id": "mcpl_native_fail", "sequence_number": 3}),
        ),
        (
            "response.output_item.done",
            json!({"output_index": 0, "item": listing, "sequence_number": 4}),
        ),
    ];
    let mut forwarded = String::new();
    for (event_type, payload) in native_events {
        let mut chunk = Some(make_sse_chunk(event_type, &payload));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        if let Some(bytes) = chunk {
            forwarded.push_str(core::str::from_utf8(&bytes).unwrap());
        }
    }

    // The backend's real terminal done must be forwarded, not suppressed as premature.
    assert_eq!(
        forwarded.matches("event: response.output_item.done").count(),
        1,
        "the backend-native output_item.done must be forwarded exactly once for a failed listing: {forwarded}"
    );
    // No synthesized duplicate: the item is not locally executed, so the flush skips it.
    assert_eq!(
        forwarded.matches("event: response.output_item.added").count(),
        1,
        "no duplicate output_item.added synthesized for a backend-native failed listing: {forwarded}"
    );
    // Exactly one of each native progress event — no synthesized duplicate.
    assert_eq!(
        forwarded.matches("event: response.mcp_list_tools.in_progress").count(),
        1,
        "the backend's native in_progress event must pass through exactly once: {forwarded}"
    );
    assert_eq!(
        forwarded.matches("event: response.mcp_list_tools.failed").count(),
        1,
        "the backend's native failed event must pass through exactly once: {forwarded}"
    );
    assert!(
        !forwarded.contains("event: response.mcp_list_tools.completed"),
        "a failed listing must not surface a synthesized completed event: {forwarded}"
    );
}

#[tokio::test]
async fn logical_stream_synthesizes_progress_for_model_declared_item_without_repeating_added() {
    // #276 (Finding 1): the model announces a `web_search_call` placeholder with
    // `output_item.added` (status `in_progress`) but never streams the
    // tool-specific progress events; the proxy completes the search locally under
    // the same id. The proxy must synthesize the full progress lifecycle
    // (in_progress -> searching -> completed) and a fresh `output_item.done`,
    // without repeating the `output_item.added` the model already streamed.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // Round 0: the model streams the placeholder search as `in_progress`.
    let mut model_item = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_item, false).unwrap();
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the same id is now completed locally under the same output index.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![json!({"type": "web_search_call", "id": "ws_1", "status": "completed"})];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.web_search_call.completed"),
        "a locally completed web search must emit a completed outcome event: {delta}"
    );
    assert!(
        delta.contains("event: response.output_item.done"),
        "the synthesized lifecycle must emit a fresh output_item.done: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the model already streamed output_item.added; synthesis must not repeat it: {delta}"
    );
    // The model streamed only `output_item.added` (item status), never the
    // `web_search_call.in_progress`/`searching` progress events, so the first
    // lifecycle synthesis must emit them.
    assert!(
        delta.contains("event: response.web_search_call.in_progress")
            && delta.contains("event: response.web_search_call.searching"),
        "the missing leading progress events must be synthesized once: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_synthesizes_progress_when_model_streams_added_then_done_without_progress() {
    // #276 (Finding 1): the model announces a web_search_call with
    // `output_item.added` AND finalizes it with `output_item.done` in round 0, but
    // never streams the tool-specific progress events; the proxy completes the
    // search locally under the same id with byte-identical content. `output_item.done`
    // means only that the item completed, not that its progress lifecycle reached
    // the client, and the tool has not even executed yet in round 0 — so the
    // premature round-0 `done` must be suppressed, and the resumed round must fill
    // in the missing in_progress/searching/completed events and emit exactly one
    // ordered `output_item.done` (without repeating the announcement the model sent).
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // Round 0: the model streams both `output_item.added` and `output_item.done`
    // for the placeholder search, but no `response.web_search_call.*` events.
    let completed_item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();
    let mut model_done = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": completed_item.clone(),
            "sequence_number": 2
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_done, false).unwrap();
    assert!(
        model_done.is_none(),
        "the premature round-0 output_item.done for a locally executed tool must be suppressed"
    );
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the same id is completed locally with content identical to the item
    // the model already finalized via `output_item.done`.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![completed_item];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.web_search_call.in_progress")
            && delta.contains("event: response.web_search_call.searching")
            && delta.contains("event: response.web_search_call.completed"),
        "output_item.done must not suppress the still-missing progress lifecycle: {delta}"
    );
    // The round-0 `done` was suppressed, so the only `output_item.done` the client
    // sees is the single ordered one the resumed synthesis emits after progress.
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        1,
        "the resumed round must emit exactly one output_item.done: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the model already streamed output_item.added; synthesis must not repeat it: {delta}"
    );
    let in_progress = delta.find("event: response.web_search_call.in_progress").unwrap();
    let searching = delta.find("event: response.web_search_call.searching").unwrap();
    let completed = delta.find("event: response.web_search_call.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        in_progress < searching && searching < completed && completed < done,
        "the single done must be ordered after the progress lifecycle: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_does_not_resynthesize_progress_streamed_in_band_by_model() {
    // #276 (3rd review, Finding 1, converse): when the model DOES stream the
    // tool-specific progress events in-band (a fully hosted web_search_call:
    // added -> in_progress -> searching -> completed -> done), those events reach
    // the client and pass through untouched. The item still lands in
    // `accumulated_output`, so a resumed round must recognize that its progress
    // lifecycle was already delivered and neither re-announce nor re-synthesize
    // it. Observing the `response.web_search_call.*` events (never `done`) is what
    // records that the lifecycle streamed.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let hosted_item = json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust"}
    });
    // Round 0: the model streams the full hosted lifecycle in-band.
    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();
    for (seq, event_type) in [
        (2, "response.web_search_call.in_progress"),
        (3, "response.web_search_call.searching"),
        (4, "response.web_search_call.completed"),
    ] {
        let mut progress = Some(make_sse_chunk(
            event_type,
            &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": seq}),
        ));
        filter.on_response_body(&mut ctx, &mut progress, false).unwrap();
        assert!(
            progress.is_some(),
            "the model's in-band progress event must pass through: {event_type}"
        );
    }
    let mut model_done = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": hosted_item.clone(),
            "sequence_number": 5
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_done, false).unwrap();
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the hosted item is carried in accumulated_output unchanged.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![hosted_item];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        !delta.contains("ws_1"),
        "an item whose lifecycle streamed in-band must not be re-synthesized: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.in_progress")
            && !delta.contains("event: response.web_search_call.searching")
            && !delta.contains("event: response.web_search_call.completed"),
        "the in-band progress lifecycle must not be duplicated: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added") && !delta.contains("event: response.output_item.done"),
        "a fully delivered hosted item must not be re-announced or re-finalized: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_synthesizes_missing_phases_after_partial_in_band_lifecycle() {
    // #276 (partial lifecycle): the model announces a web_search_call, streams ONLY
    // the `in_progress` progress event in-band, then finalizes with
    // `output_item.done` — never `searching` or `completed`. Each phase is a
    // distinct API lifecycle event, so recording one must not mark the whole
    // lifecycle delivered. The premature round-0 `done` is suppressed, and the
    // resumed round must synthesize exactly the still-missing `searching` and
    // `completed` events plus one ordered `done`, without repeating the
    // `output_item.added` or the `in_progress` the client already saw in-band.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // Round 0: added, then ONLY the `in_progress` progress event, then a done.
    let completed_item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();
    let mut in_progress = Some(make_sse_chunk(
        "response.web_search_call.in_progress",
        &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": 2}),
    ));
    filter.on_response_body(&mut ctx, &mut in_progress, false).unwrap();
    assert!(
        in_progress.is_some(),
        "the model's in-band in_progress event must pass through to the client"
    );
    let mut model_done = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": completed_item.clone(),
            "sequence_number": 3
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_done, false).unwrap();
    assert!(
        model_done.is_none(),
        "a done that finalizes a still-partial local-tool lifecycle must be suppressed"
    );
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the same id is completed locally with content identical to the item
    // the model already finalized via the suppressed `output_item.done`.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![completed_item];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    // The missing middle/terminal phases must be filled even though `in_progress`
    // already streamed and the item's content is byte-identical to round 0.
    assert!(
        delta.contains("event: response.web_search_call.searching"),
        "a phase missing from the in-band lifecycle must be synthesized: {delta}"
    );
    assert!(
        delta.contains("event: response.web_search_call.completed"),
        "the terminal outcome missing from the in-band lifecycle must be synthesized: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.in_progress"),
        "the in_progress phase already streamed in-band must not be repeated: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the model already streamed output_item.added; synthesis must not repeat it: {delta}"
    );
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        1,
        "the resumed round must emit exactly one output_item.done: {delta}"
    );
    let searching = delta.find("event: response.web_search_call.searching").unwrap();
    let completed = delta.find("event: response.web_search_call.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        searching < completed && completed < done,
        "the synthesized phases must be ordered searching -> completed -> done: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_fills_middle_phase_after_leading_in_band_progress() {
    // #276 (partial lifecycle, no round-0 done): the model announces a
    // web_search_call and streams ONLY the `in_progress` progress event, then the
    // round ends (IRR loop) with the tool still unfinished — no `searching`,
    // `completed`, or `done`. The proxy completes the search locally, changing the
    // item's content (status in_progress -> completed). The resumed round must
    // supply the missing `searching` AND `completed` phases, not just the terminal
    // outcome: a single-bool "lifecycle streamed" flag would drop `searching`.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();
    let mut in_progress = Some(make_sse_chunk(
        "response.web_search_call.in_progress",
        &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": 2}),
    ));
    filter.on_response_body(&mut ctx, &mut in_progress, false).unwrap();
    assert!(
        in_progress.is_some(),
        "the model's in-band in_progress event must pass through to the client"
    );
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: local execution completes the search (content changes to completed).
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![json!({"type": "web_search_call", "id": "ws_1", "status": "completed"})];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.web_search_call.searching"),
        "the middle `searching` phase missing in-band must be synthesized, not dropped: {delta}"
    );
    assert!(
        delta.contains("event: response.web_search_call.completed"),
        "the terminal outcome must be synthesized: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.in_progress"),
        "the in_progress phase already streamed in-band must not be repeated: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the model already streamed output_item.added; synthesis must not repeat it: {delta}"
    );
    let searching = delta.find("event: response.web_search_call.searching").unwrap();
    let completed = delta.find("event: response.web_search_call.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        searching < completed && completed < done,
        "the synthesized phases must be ordered searching -> completed -> done: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_keeps_in_band_done_when_outcome_streams_without_searching() {
    // #276 (non-prefix partial lifecycle): the model announces a web_search_call
    // and streams `in_progress` then `completed` in-band, deliberately SKIPPING the
    // optional `searching` phase, then finalizes with `output_item.done`. Because
    // the terminal phase reached the client, that `done` is authoritative and must
    // pass through unchanged. The resumed round must NOT resurrect the skipped
    // `searching` (it would land after `completed`, out of canonical order) nor
    // replace the backend's real `done` with a synthesized duplicate.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let completed_item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();

    let mut in_progress = Some(make_sse_chunk(
        "response.web_search_call.in_progress",
        &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": 2}),
    ));
    filter.on_response_body(&mut ctx, &mut in_progress, false).unwrap();
    assert!(in_progress.is_some(), "the in-band in_progress event must pass through");

    // The backend jumps straight to `completed`, skipping the optional `searching`.
    let mut completed = Some(make_sse_chunk(
        "response.web_search_call.completed",
        &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": 3}),
    ));
    filter.on_response_body(&mut ctx, &mut completed, false).unwrap();
    assert!(completed.is_some(), "the in-band completed event must pass through");

    let mut model_done = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": completed_item.clone(),
            "sequence_number": 4
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_done, false).unwrap();
    assert!(
        model_done.is_some(),
        "a done whose terminal phase already streamed in-band is authoritative and must pass through"
    );

    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: the same id is completed locally with content identical to round 0.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![completed_item];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        !delta.contains("event: response.web_search_call.searching"),
        "a phase the backend skipped must not be back-filled after the outcome: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.completed"),
        "the terminal outcome already streamed in-band must not be duplicated: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.done"),
        "the backend's authoritative done was already delivered; none must be synthesized: {delta}"
    );
    assert!(
        !delta.contains("ws_1"),
        "a fully delivered hosted item must not be re-synthesized in the resumed round: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_honors_skipped_leading_phase_when_only_searching_streamed() {
    // #276 (non-prefix partial lifecycle): the model streams ONLY `searching`
    // in-band — skipping the earlier `in_progress` — then the round ends (IRR loop)
    // before any outcome. Local execution completes the search. The resumed round
    // must synthesize only the still-owed `completed` and a single `done`; it must
    // NOT back-fill the skipped `in_progress`, which would land after the already
    // streamed `searching`, out of canonical order.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();

    // The backend streams `searching` without first streaming `in_progress`.
    let mut searching = Some(make_sse_chunk(
        "response.web_search_call.searching",
        &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": 2}),
    ));
    filter.on_response_body(&mut ctx, &mut searching, false).unwrap();
    assert!(searching.is_some(), "the in-band searching event must pass through");

    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: local execution completes the search (content changes to completed).
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![json!({"type": "web_search_call", "id": "ws_1", "status": "completed"})];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert!(
        delta.contains("event: response.web_search_call.completed"),
        "the still-owed terminal outcome must be synthesized: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.in_progress"),
        "the skipped leading in_progress must not be back-filled after searching: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.searching"),
        "the searching phase already streamed in-band must not be repeated: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the model already streamed output_item.added; synthesis must not repeat it: {delta}"
    );
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        1,
        "the resumed round must emit exactly one output_item.done: {delta}"
    );
    let completed = delta.find("event: response.web_search_call.completed").unwrap();
    let done = delta.find("event: response.output_item.done").unwrap();
    assert!(
        completed < done,
        "the synthesized outcome must precede the fresh done: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_reemits_outcome_when_local_item_gains_sources() {
    // #276 (Finding 2): a web_search_call whose progress lifecycle already
    // reached the client gains `action.sources` after further local execution.
    // A content change (not just type/status) must re-emit a fresh
    // `output_item.done` carrying the sources, without repeating the
    // `output_item.added`, the leading progress events, or the terminal phase
    // event the client already saw. Only the `done` envelope carries the item
    // payload — the terminal phase event is payloadless, so re-emitting it would
    // be a pure duplicate that conveys nothing about the change.
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_web_search",
        vec![json!({
            "type": "web_search_call",
            "id": "ws_sources_1",
            "status": "completed",
            "action": {"type": "search", "query": "rust"}
        })],
    )
    .await;

    // Round 1: the isolated web search is synthesized with its full lifecycle.
    let first = resumed_text_delta(&filter, &mut ctx);
    assert!(
        first.contains("event: response.output_item.added")
            && first.contains("event: response.web_search_call.in_progress")
            && first.contains("event: response.web_search_call.searching")
            && first.contains("event: response.web_search_call.completed")
            && first.contains("event: response.output_item.done"),
        "the first synthesis must emit the full search lifecycle: {first}"
    );

    // Round 2: local execution adds `action.sources` to the same completed item.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 2;
    state.accumulated_output = vec![json!({
        "type": "web_search_call",
        "id": "ws_sources_1",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "rust",
            "sources": [{"type": "url", "url": "https://blog.rust-lang.org"}]
        }
    })];
    mark_accumulated_output_executed(state);
    filter.arm(&mut ctx);

    let second = resumed_text_delta(&filter, &mut ctx);
    assert!(
        second.contains("blog.rust-lang.org"),
        "the re-emitted output_item.done must carry the newly added sources: {second}"
    );
    assert_eq!(
        second.matches("event: response.output_item.done").count(),
        1,
        "a content change must re-emit exactly one fresh output_item.done: {second}"
    );
    assert!(
        !second.contains("event: response.web_search_call.completed"),
        "the payloadless terminal phase must not be re-emitted on a content change; \
         only the output_item.done envelope carries the changed item: {second}"
    );
    assert!(
        !second.contains("event: response.output_item.added"),
        "content re-emission must not repeat output_item.added: {second}"
    );
    assert!(
        !second.contains("event: response.web_search_call.in_progress")
            && !second.contains("event: response.web_search_call.searching"),
        "content re-emission must not repeat the leading progress events: {second}"
    );
}

#[test]
fn item_digest_ignores_key_order_but_tracks_value_changes() {
    // #276 (Finding 3): the content digest must depend only on content, not object
    // key order. The two sides of a cross-round change comparison come from
    // different backend serializations whose key order is not stable
    // (`preserve_order` makes serde_json retain insertion order), so a pure reorder
    // — top-level and nested — must produce an identical digest, while a genuine
    // nested value change must produce a different one.
    let base = json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust", "sources": [{"url": "https://a", "type": "url"}]}
    });
    let reordered = json!({
        "action": {"sources": [{"type": "url", "url": "https://a"}], "query": "rust", "type": "search"},
        "status": "completed",
        "id": "ws_1",
        "type": "web_search_call"
    });
    assert_eq!(
        super::item_digest(&base),
        super::item_digest(&reordered),
        "a pure key reorder (top-level and nested) must not change the digest"
    );

    let changed = json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust", "sources": [{"url": "https://b", "type": "url"}]}
    });
    assert_ne!(
        super::item_digest(&base),
        super::item_digest(&changed),
        "a nested value change must change the digest"
    );
}

#[test]
fn item_digest_hashes_numbers_without_conflating_shapes() {
    // #276 (Finding 2): numbers are hashed from their primitive representation
    // without allocating a string, yet must retain the previous string-form
    // semantics — an integer and a float of equal magnitude are distinct JSON
    // values and must not collide, and a genuine numeric change must be detected.
    let integer = json!({"type": "mcp_call", "id": "c", "n": 5});
    let float = json!({"type": "mcp_call", "id": "c", "n": 5.0});
    assert_ne!(
        super::item_digest(&integer),
        super::item_digest(&float),
        "integer 5 and float 5.0 are distinct values and must not share a digest"
    );

    let same_integer = json!({"type": "mcp_call", "id": "c", "n": 5});
    assert_eq!(
        super::item_digest(&integer),
        super::item_digest(&same_integer),
        "the same integer must hash identically across serializations"
    );

    let changed = json!({"type": "mcp_call", "id": "c", "n": 6});
    assert_ne!(
        super::item_digest(&integer),
        super::item_digest(&changed),
        "a numeric value change must change the digest"
    );

    let negative = json!({"type": "mcp_call", "id": "c", "n": -5});
    assert_ne!(
        super::item_digest(&integer),
        super::item_digest(&negative),
        "sign is part of a number's value and must change the digest"
    );
}

#[tokio::test]
async fn logical_stream_finalizes_local_item_when_done_envelope_missing_and_content_unchanged() {
    // #276 (Finding 1): a backend streams a web_search_call's full phase lifecycle
    // in-band but the round is cut off (IRR loop) before the finalizing
    // `output_item.done` envelope, and local execution yields content identical to
    // what already streamed. The `done` envelope is tracked apart from the phase set
    // and the content digest, so the resumed round must still finalize the item with
    // exactly one `output_item.done` — otherwise the client is left with an item
    // that never received its terminal envelope. No phase may be re-emitted (all
    // already streamed) and no second `output_item.added`.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // The backend announces the item already in its terminal (completed) form, then
    // streams every phase, so the content the client last saw matches the eventual
    // accumulated item and only the `done` envelope is ever missing.
    let completed_item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed"});
    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": completed_item.clone(),
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();

    for (seq, phase) in [
        (2, "response.web_search_call.in_progress"),
        (3, "response.web_search_call.searching"),
        (4, "response.web_search_call.completed"),
    ] {
        let mut chunk = Some(make_sse_chunk(
            phase,
            &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": seq}),
        ));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(chunk.is_some(), "the in-band {phase} event must pass through");
    }

    // The round ends (IRR loop) before the backend sends `output_item.done`.
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: local execution yields the identical completed item.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![completed_item];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        1,
        "the missing done envelope must be finalized with exactly one output_item.done: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.completed"),
        "no phase may be re-emitted; every phase already streamed in-band: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.in_progress")
            && !delta.contains("event: response.web_search_call.searching"),
        "no leading phase may be re-emitted; all already streamed in-band: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the item was already announced; output_item.added must not repeat: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_finalizes_without_duplicating_terminal_phase_when_content_changed() {
    // #276 (Finding 1): a backend streams the full phase lifecycle in-band (added
    // with the in_progress placeholder, then in_progress/searching/completed) but
    // the round is cut off before `output_item.done`; local execution then completes
    // the item, changing its content. The resumed round must finalize it with
    // exactly one `output_item.done` carrying the updated item, and must NOT re-emit
    // the payloadless terminal phase already streamed in-band — that phase event
    // carries no item data, so re-emitting it would be a pure duplicate.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let mut model_added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut model_added, false).unwrap();

    for (seq, phase) in [
        (2, "response.web_search_call.in_progress"),
        (3, "response.web_search_call.searching"),
        (4, "response.web_search_call.completed"),
    ] {
        let mut chunk = Some(make_sse_chunk(
            phase,
            &json!({"item_id": "ws_1", "output_index": 0, "sequence_number": seq}),
        ));
        filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
        assert!(chunk.is_some(), "the in-band {phase} event must pass through");
    }

    // The round ends (IRR loop) before the backend sends `output_item.done`.
    ctx.filter_results
        .entry("openai_web_search")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();

    // Resume: local execution completes the search, changing the item's content.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![json!({
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "rust", "sources": [{"type": "url", "url": "https://a"}]}
    })];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_web_search");
    filter.arm(&mut ctx);

    let delta = resumed_text_delta(&filter, &mut ctx);
    assert_eq!(
        delta.matches("event: response.output_item.done").count(),
        1,
        "a content change with a missing done envelope must yield exactly one output_item.done: {delta}"
    );
    assert!(
        delta.contains("https://a"),
        "the finalizing output_item.done must carry the updated item content: {delta}"
    );
    assert!(
        !delta.contains("event: response.web_search_call.completed"),
        "the payloadless terminal phase already streamed in-band must not be re-emitted: {delta}"
    );
    assert!(
        !delta.contains("event: response.output_item.added"),
        "the item was already announced; output_item.added must not repeat: {delta}"
    );
}

#[tokio::test]
async fn logical_stream_error_still_flushes_pending_local_items() {
    // #276 (Finding 3): a resumed-round parse error must still stream the tool
    // items the dispatch filter already executed before terminating with an
    // error, so already-committed tool activity is never silently dropped.
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_mcp_dispatch",
        vec![json!({"type": "mcp_call", "id": "mcp_pending_1"})],
    )
    .await;

    let mut malformed = Some(Bytes::from(
        "event: response.output_text.delta\ndata: {\"response_id\":\"resp_b\",bad}\n\n",
    ));
    filter.on_response_body(&mut ctx, &mut malformed, false).unwrap();
    assert!(malformed.is_none(), "a malformed resumed chunk must be suppressed");

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        eos.contains("mcp_pending_1"),
        "an already-executed local tool item must be streamed before the terminal error: {eos}"
    );
    assert!(
        eos.contains("event: response.output_item.added") && eos.contains("event: response.mcp_call.completed"),
        "the pending local item must carry its synthesized lifecycle: {eos}"
    );
    assert!(
        eos.contains("event: error"),
        "the stream must still terminate with an error: {eos}"
    );
    let flushed = eos.find("mcp_pending_1").unwrap();
    let error = eos.find("event: error").unwrap();
    assert!(
        flushed < error,
        "pending local items must precede the terminal error: {eos}"
    );
}

#[tokio::test]
async fn retained_overflow_while_flushing_local_items_emits_only_the_error() {
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_mcp_dispatch",
        vec![json!({
            "type": "mcp_call",
            "id": "mcp_must_not_leak",
            "name": "tool",
            "arguments": "x".repeat(4_096),
            "status": "completed"
        })],
    )
    .await;

    let parser_bytes = ctx
        .get_filter_state::<StreamEventsState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    let retained = state.retained_payload_bytes().unwrap();
    let local_output = super::local_terminal_output_upper_bound(state).unwrap();
    state.apply_retained_payload_limit(retained + parser_bytes + local_output - 1);
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.stream_error_message", "upstream failed");
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "done")
        .unwrap();

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert_eq!(eos.matches("event: error").count(), 1, "{eos}");
    assert!(!eos.contains("mcp_must_not_leak"), "{eos}");
    assert!(!eos.contains("response.output_item"), "{eos}");
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[tokio::test]
async fn logical_stream_multi_frame_parse_error_still_flushes_pending_local_items() {
    // #276 (Finding 3): a resumed-round chunk that carries a VALID event followed
    // by a malformed frame must be handled atomically. The valid frame alone would
    // trigger the local-item flush and record the pending mcp_call as delivered —
    // but the malformed frame then aborts the chunk, so `handle_parse_error`
    // discards ALL of its bytes. If the milestone had been committed, EOS recovery
    // would treat the executed tool as already delivered and drop it, leaving the
    // client only the terminal error. The chunk must not commit any milestone
    // until it fully parses, so the executed local item still reaches the client.
    let (filter, mut ctx) = arm_resumed_round_with_accumulated(
        "openai_mcp_dispatch",
        vec![json!({"type": "mcp_call", "id": "mcp_pending_multi"})],
    )
    .await;

    // One chunk, two frames: a valid resumed text delta, then a malformed frame.
    let mut chunk = Some(Bytes::from(concat!(
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_b\",",
        "\"output_index\":0,\"content_index\":0,\"delta\":\"hi\",\"sequence_number\":1}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"response_id\":\"resp_b\",bad}\n\n",
    )));
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();
    assert!(
        chunk.is_none(),
        "a chunk whose later frame is malformed must be discarded wholesale"
    );

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        eos.contains("mcp_pending_multi"),
        "the executed local tool item must survive the mid-chunk parse error: {eos}"
    );
    assert!(
        eos.contains("event: response.output_item.added") && eos.contains("event: response.mcp_call.completed"),
        "the recovered local item must carry its synthesized lifecycle: {eos}"
    );
    assert!(
        eos.contains("event: error"),
        "the stream must still terminate with an error: {eos}"
    );
    let flushed = eos.find("mcp_pending_multi").unwrap();
    let error = eos.find("event: error").unwrap();
    assert!(
        flushed < error,
        "the recovered local item must precede the terminal error: {eos}"
    );
}

#[tokio::test]
async fn logical_stream_error_does_not_fabricate_lifecycle_for_unexecuted_placeholder() {
    // #276 (provenance): a model-streamed `web_search_call` placeholder that a
    // non-dispatchable round copies into `accumulated_output` (via
    // `agentic_loop::collect_streaming_output_items`, which runs even when a parse
    // error blocks dispatch) was never executed by the web-search dispatch. The
    // terminal error flush keys synthesis on execution provenance, not item type,
    // so it must NOT fabricate an in_progress/searching lifecycle and a `done`
    // envelope for a search that never ran; only executed local tool items may be
    // synthesized. Without the provenance gate the flush invents a full lifecycle
    // for `ws_ghost`.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hi",
        "stream": true
    })));
    filter.arm(&mut ctx);

    // The round announces a `web_search_call` placeholder, then hits a parse error
    // before the tool lifecycle streams or the dispatch filter executes it.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_a", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    let mut added = Some(make_sse_chunk(
        "response.output_item.added",
        &json!({
            "response_id": "resp_a",
            "output_index": 0,
            "item": {"type": "web_search_call", "id": "ws_ghost", "status": "in_progress"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();

    // Simulate `collect_streaming_output_items` copying the model placeholder into
    // `accumulated_output` on the failed round, with NO dispatch execution having
    // recorded provenance for it.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.accumulated_output = vec![json!({
        "type": "web_search_call",
        "id": "ws_ghost",
        "status": "in_progress"
    })];

    let mut malformed = Some(Bytes::from(
        "event: response.output_text.delta\ndata: {\"response_id\":\"resp_a\",bad}\n\n",
    ));
    filter.on_response_body(&mut ctx, &mut malformed, false).unwrap();

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        eos.contains("event: error"),
        "the stream must terminate with an error: {eos}"
    );
    assert!(
        !eos.contains("event: response.web_search_call.in_progress")
            && !eos.contains("event: response.web_search_call.searching"),
        "an unexecuted web_search_call placeholder must not gain a fabricated lifecycle: {eos}"
    );
    assert!(
        !eos.contains("event: response.output_item.done"),
        "an unexecuted placeholder must not be finalized with a synthesized done: {eos}"
    );
}

fn make_done_chunk() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

#[tokio::test]
async fn terminal_event_writes_response_object() {
    let (filter, mut ctx) = make_armed_context();

    let response_payload = json!({
        "id": "resp_123",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    });

    let mut body = Some(make_sse_chunk("response.completed", &response_payload));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["id"], "resp_123");
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(state.usage["total_tokens"], 15);
    assert_eq!(ctx.get_metadata("responses.status"), Some("completed"),);
}

#[tokio::test]
async fn logical_terminal_only_output_survives_canonicalization() {
    // A plain one-round logical stream: the model's output arrives solely in the
    // terminal `response.completed` event, with no incremental `output_item.*`
    // events and no dispatch/loop filter to fill `accumulated_output`. The
    // finalized logical terminal — and the response-store source it mirrors —
    // must preserve that output rather than replace it with an empty accumulator.
    let (filter, mut ctx) = make_armed_context();

    let message = json!({
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello from stream"}]
    });
    let completed = json!({
        "response": {
            "id": "resp_terminal_only",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "created_at": 1_700_000_000,
            "output": [message.clone()]
        },
        "sequence_number": 0
    });

    let mut terminal = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(terminal.is_none(), "the terminal event must be deferred until finalize");

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    // `unwrap` (not `expect`) matches this module's test conventions; a `None`
    // here means finalize failed to emit the deferred terminal.
    let emitted = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        emitted.contains("Hello from stream"),
        "the finalized terminal must carry the terminal event's own output: {emitted}"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.accumulated_output.is_empty(),
        "no dispatch/loop filter ran, so the cross-round accumulator stays empty"
    );
    assert_eq!(
        state.output_items().len(),
        1,
        "the store source must retain the terminal output"
    );
    assert_eq!(
        state.output_items()[0],
        message,
        "the store source output must equal the terminal event output"
    );
}

#[tokio::test]
async fn deferred_terminal_arms_store_persistence_at_terminal_frame() {
    // #937: the deferred terminal frame reaches the pre-IRR store as a
    // non-end-of-stream chunk. `emit_deferred_terminal` must mark
    // `logical_stream_terminal_emitted` exactly when it appends that frame so the
    // store persists before releasing it. The flag must stay unset while the
    // terminal is still deferred (held, not yet emitted).
    let (filter, mut ctx) = make_armed_context();

    let completed = json!({
        "response": {
            "id": "resp_937_defer",
            "object": "response",
            "status": "completed",
            "model": "gpt-4o",
            "created_at": 1_700_000_000,
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi"}]}]
        },
        "sequence_number": 0
    });

    let mut terminal = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(terminal.is_none(), "the terminal event must be deferred until finalize");
    assert!(
        !ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .logical_stream_terminal_emitted,
        "persistence must not be armed while the terminal is still deferred"
    );

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    assert!(eos.is_some(), "finalize must emit the deferred terminal frame");
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .logical_stream_terminal_emitted,
        "emit_deferred_terminal must arm store persistence when it appends the terminal frame"
    );
}

#[tokio::test]
async fn terminal_event_authoritatively_populates_completed_function_calls() {
    let (filter, mut ctx) = make_armed_context();
    ctx.extensions.insert(ResponsesState::default());
    ctx.extensions.get_mut::<ResponsesState>().unwrap().select_test_output(
        "function_call",
        vec![json!({
            "type": "function_call",
            "id": "fc_stale",
            "call_id": "call_stale",
            "name": "stale",
            "arguments": "{}",
            "status": "completed"
        })],
    );

    let completed = json!({
        "id": "resp_123",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_final",
            "call_id": "call_final",
            "name": "lookup",
            "arguments": r#"{"query":"Praxis"}"#,
            "status": "completed"
        }]
    });
    let mut body = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.output_items().len(),
        1,
        "the authoritative terminal response owns the completed call"
    );
    assert_eq!(
        state.output_items()[0]["call_id"],
        "call_final",
        "a terminal-only completed function call must be dispatchable"
    );
}

#[test]
fn response_accumulation_sums_usage_across_iterations() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.extensions.insert(ResponsesState::default());
    let first = json!({
        "status":"completed",
        "output":[],
        "usage":{
            "input_tokens":10,
            "output_tokens":4,
            "total_tokens":14,
            "input_tokens_details":{"cached_tokens":3}
        }
    });
    let second = json!({
        "status":"completed",
        "output":[],
        "usage":{
            "input_tokens":7,
            "output_tokens":2,
            "total_tokens":9,
            "input_tokens_details":{"cached_tokens":1}
        }
    });

    assert!(
        !accumulate_response_object(&mut ctx, first, None),
        "in-progress response must not report terminal completion"
    );
    assert!(
        accumulate_response_object(&mut ctx, second, None),
        "completed response must report terminal completion"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.usage["input_tokens"], 17);
    assert_eq!(state.usage["output_tokens"], 6);
    assert_eq!(state.usage["total_tokens"], 23);
    assert_eq!(state.usage["input_tokens_details"]["cached_tokens"], 4);
    assert_eq!(state.response_object["usage"], state.usage);

    let final_without_usage = json!({"status":"completed","output":[]});
    assert!(
        accumulate_response_object(&mut ctx, final_without_usage, None),
        "completed response without usage must remain terminal"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["usage"], state.usage);
    assert_eq!(state.usage["total_tokens"], 23);
}

#[tokio::test]
async fn output_item_added_accumulates_incrementally() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({"type": "message", "role": "assistant", "id": "item_1"});
    let payload = json!({"item": item});

    let mut body = Some(make_sse_chunk("response.output_item.added", &payload));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(state.output_items()[0]["id"], "item_1");
}

#[tokio::test]
async fn terminal_event_overwrites_incremental_output() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({"item": {"type": "message", "id": "item_1"}});
    let mut body1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut body1, false).unwrap();
    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().output_items().len(), 1);

    let completed = json!({
        "id": "resp_123",
        "status": "completed",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [
            {"type": "message", "id": "item_final_1"},
            {"type": "message", "id": "item_final_2"}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    });
    let mut body2 = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut body2, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.output_items().len(),
        2,
        "terminal event should overwrite incremental output"
    );
    assert_eq!(state.output_items()[0]["id"], "item_final_1");
}

#[tokio::test]
async fn function_call_accumulation() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_item_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut item_body = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut item_body, false).unwrap();

    let delta1 = json!({"item_id": "fc_item_1", "output_index": 0, "delta": "{\"city\":"});
    let mut b1 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta1));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let delta2 = json!({"item_id": "fc_item_1", "output_index": 0, "delta": "\"NYC\"}"});
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta2));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let done = json!({
        "item_id": "fc_item_1",
        "output_index": 0,
        "arguments": "{\"city\":\"NYC\"}"
    });
    let mut b3 = Some(make_sse_chunk("response.function_call_arguments.done", &done));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(state.output_items()[0]["id"], "fc_item_1");
    assert_eq!(state.output_items()[0]["call_id"], "call_1");
    assert_eq!(state.output_items()[0]["name"], "get_weather");
    assert_eq!(state.output_items()[0]["arguments"], "{\"city\":\"NYC\"}");
    assert_eq!(state.output_items()[0]["status"], "completed");
    assert_eq!(state.output_items()[0]["arguments"], "{\"city\":\"NYC\"}");
}

#[tokio::test]
async fn missing_state_does_not_panic() {
    let (filter, mut ctx) = make_armed_context();

    let completed = json!({
        "id": "resp_123",
        "status": "completed",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    });
    let mut body = Some(make_sse_chunk("response.completed", &completed));
    let result = filter.on_response_body(&mut ctx, &mut body, false);
    assert!(result.is_ok(), "should not panic with missing ResponsesState");
    assert!(
        ctx.extensions.get::<ResponsesState>().is_some(),
        "should have created ResponsesState"
    );
}

#[tokio::test]
async fn eos_validates_stream_completeness() {
    let (filter, mut ctx) = make_armed_context();

    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});
    let mut b1 = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let mut b2 = Some(make_done_chunk());
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let mut empty = None;
    filter.on_response_body(&mut ctx, &mut empty, true).unwrap();
    assert!(
        ctx.get_metadata("responses.stream_parse_error").is_none(),
        "DONE sentinel should not set parse-error metadata"
    );
    assert!(
        ctx.get_metadata("responses.stream_incomplete").is_none(),
        "complete stream should not set incomplete flag"
    );
}

#[tokio::test]
async fn eos_without_terminal_sets_incomplete() {
    let (filter, mut ctx) = make_armed_context();

    let delta = json!({"text": "hi"});
    let mut b1 = Some(make_sse_chunk("response.output_text.delta", &delta));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let mut empty = None;
    filter.on_response_body(&mut ctx, &mut empty, true).unwrap();
    assert_eq!(
        ctx.get_metadata("responses.stream_incomplete"),
        Some("true"),
        "missing terminal should set incomplete flag"
    );
}

#[tokio::test]
async fn logical_eos_without_terminal_emits_error() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.arm(&mut ctx);

    let mut delta = Some(make_sse_chunk(
        "response.output_text.delta",
        &json!({
            "response_id": "resp_partial",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    assert!(
        eos.contains("event: error"),
        "a logical stream must explicitly terminate when upstream omits its terminal event: {eos}"
    );

    let data = eos
        .split("event: error\n")
        .nth(1)
        .and_then(|rest| rest.strip_prefix("data: "))
        .and_then(|rest| rest.split('\n').next())
        .unwrap();
    let payload: serde_json::Value = serde_json::from_str(data).unwrap();
    assert_eq!(payload["type"], "error", "event type is always \"error\"");
    assert_eq!(
        payload["code"], "server_error",
        "the caller-selected machine-readable code is preserved at the top level"
    );
    assert_eq!(
        payload["message"], "upstream Responses stream did not terminate cleanly",
        "message is a top-level field"
    );
    assert!(payload["param"].is_null(), "param is a top-level field");
    assert!(
        payload["sequence_number"].is_number(),
        "sequence_number is a top-level field normalized to the stream position"
    );
    assert!(
        payload.get("error").is_none(),
        "a committed-stream SSE error event must not nest fields under an \"error\" object: {payload}"
    );

    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "a stream missing its terminal event must not be persisted"
    );
}

#[test]
fn armed_first_round_emits_created_lifecycle() {
    // A single inference round is a one-round logical stream: the first round's
    // created lifecycle is emitted (normalized) rather than suppressed.
    let (filter, mut ctx) = make_armed_context();

    let mut body = Some(make_sse_chunk(
        "response.created",
        &json!({"response": {"id": "r1", "status": "in_progress", "output": []}}),
    ));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let emitted = String::from_utf8(body.unwrap().to_vec()).unwrap();
    assert!(
        emitted.contains("event: response.created"),
        "the first round's created lifecycle should be emitted: {emitted}"
    );
    assert!(
        emitted.contains(r#""id":"r1""#),
        "the first round keeps its own response identity: {emitted}"
    );
}

#[test]
fn encode_sse_event_writes_compact_json_directly_into_output() {
    let payload = json!({
        "type": "response.output_text.delta",
        "delta": "α".repeat(2048),
        "sequence_number": 12,
        "response_id": "resp_logical",
        "output_index": 3
    });
    let json_bytes = serde_json::to_vec(&payload).unwrap();
    let mut output = Vec::new();
    super::encode_sse_event("response.output_text.delta", &payload, &mut output);

    let prefix = b"event: response.output_text.delta\ndata: ";
    let suffix = b"\n\n";
    assert_eq!(
        output.len(),
        prefix.len() + json_bytes.len() + suffix.len(),
        "logical-stream SSE must be framing plus compact JSON with no intermediate String"
    );
    assert_eq!(
        &output[..prefix.len()],
        prefix.as_slice(),
        "SSE event name and data delimiter must be unchanged"
    );
    assert_eq!(
        &output[prefix.len()..prefix.len() + json_bytes.len()],
        json_bytes.as_slice(),
        "payload JSON must be written with to_writer compact encoding"
    );
    assert_eq!(
        &output[output.len() - suffix.len()..],
        suffix.as_slice(),
        "SSE event delimiter must remain a trailing blank line"
    );
}

#[test]
fn write_json_or_rollback_discards_partial_bytes_on_failure() {
    let mut output = b"event: response.failed\ndata: ".to_vec();
    let prefix = output.clone();
    let result = super::write_json_or_rollback(&mut output, |out| {
        out.extend_from_slice(br#"{"type":"response.failed""#);
        Err(std::io::Error::other("injected write failure"))
    });
    assert!(
        result.is_err(),
        "injected failure must surface so encode_sse_event can skip the SSE delimiter"
    );
    let err = result.unwrap_err();

    assert_eq!(
        err.kind(),
        std::io::ErrorKind::Other,
        "rollback must preserve the original write error"
    );
    assert_eq!(
        output, prefix,
        "a failed payload write must not leave partial JSON in the SSE buffer"
    );
    assert!(
        !output.windows(2).any(|window| window == b"\n\n"),
        "encode_sse_event must return before appending the SSE delimiter"
    );
}

/// Previous logical-stream encoder: compact JSON through an intermediate `String`.
fn encode_sse_event_with_intermediate_string(event_type: &str, payload: &serde_json::Value, output: &mut Vec<u8>) {
    output.extend_from_slice(b"event: ");
    output.extend_from_slice(event_type.as_bytes());
    output.extend_from_slice(b"\ndata: ");
    let json = serde_json::to_string(payload).unwrap();
    output.extend_from_slice(json.as_bytes());
    output.extend_from_slice(b"\n\n");
}

fn large_delta_payload() -> serde_json::Value {
    json!({
        "type": "response.output_text.delta",
        "delta": "α".repeat(4096),
        "sequence_number": 12,
        "response_id": "resp_logical",
        "output_index": 3
    })
}

fn large_completed_payload() -> serde_json::Value {
    json!({
        "type": "response.completed",
        "sequence_number": 99,
        "response": {
            "id": "resp_logical",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "α".repeat(4096)}]
            }]
        }
    })
}

fn assert_writer_allocates_less_than_string(event_type: &str, payload: &serde_json::Value) {
    let capacity = serde_json::to_vec(payload).unwrap().len() + 64;
    let mut via_writer = Vec::with_capacity(capacity);
    let mut via_string = Vec::with_capacity(capacity);
    let writer = allocation_counter::measure(|| {
        super::encode_sse_event(event_type, payload, &mut via_writer);
    });
    let string = allocation_counter::measure(|| {
        encode_sse_event_with_intermediate_string(event_type, payload, &mut via_string);
    });
    assert_eq!(
        via_writer, via_string,
        "{event_type} SSE bytes must stay equivalent while measuring allocations"
    );
    assert!(
        writer.bytes_total < string.bytes_total,
        "{event_type} to_writer must allocate fewer bytes than to_string: writer={} string={}",
        writer.bytes_total,
        string.bytes_total
    );
    assert!(
        writer.count_total < string.count_total,
        "{event_type} to_writer must allocate fewer times than to_string: writer={} string={}",
        writer.count_total,
        string.count_total
    );
}

#[test]
fn encode_sse_event_allocates_less_than_intermediate_string_for_ordinary_and_deferred_terminal() {
    assert_writer_allocates_less_than_string("response.output_text.delta", &large_delta_payload());
    assert_writer_allocates_less_than_string("response.completed", &large_completed_payload());
}

#[test]
fn deferred_terminal_stores_moved_payload_without_deep_clone() {
    let cloned = large_completed_payload();
    let moved = large_completed_payload();
    let clone_info = allocation_counter::measure(|| {
        std::hint::black_box(super::DeferredTerminalEvent {
            event_type: "response.completed".to_owned(),
            payload: cloned.clone(),
        });
    });
    let move_info = allocation_counter::measure(|| {
        std::hint::black_box(super::DeferredTerminalEvent {
            event_type: "response.completed".to_owned(),
            payload: moved,
        });
    });
    assert!(
        clone_info.bytes_total >= 4096,
        "deferred-terminal clone must copy the completed output text, allocated {}",
        clone_info.bytes_total
    );
    assert!(
        move_info.bytes_total < clone_info.bytes_total,
        "deferred-terminal store must move the payload: clone={} move={}",
        clone_info.bytes_total,
        move_info.bytes_total
    );
}

#[test]
fn terminal_canonicalization_moves_large_output_without_a_payload_clone() {
    let text = "x".repeat(256 * 1024);
    let mut state = ResponsesState {
        response_object: json!({"id": "resp_move", "output": []}),
        accumulated_output: vec![json!({
            "type": "message",
            "id": "msg_move",
            "content": [{"type": "output_text", "text": text}]
        })],
        ..ResponsesState::default()
    };
    let old_ptr = state.accumulated_output[0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .as_ptr();
    let clone_baseline = allocation_counter::measure(|| {
        std::hint::black_box(state.accumulated_output.clone());
    });
    let moved = allocation_counter::measure(|| {
        canonicalize_logical_response(&mut state, false).unwrap();
    });
    let new_ptr = state.response_object["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .as_ptr();
    assert_eq!(new_ptr, old_ptr, "canonical output keeps the original text allocation");
    assert!(state.accumulated_output.is_empty());
    assert!(clone_baseline.bytes_total >= 256 * 1024);
    assert!(
        moved.bytes_total < clone_baseline.bytes_total / 4,
        "canonicalization allocated {} bytes versus {} for the old output clone",
        moved.bytes_total,
        clone_baseline.bytes_total
    );
}

#[test]
fn terminal_accumulation_moves_parsed_response_into_shared_state() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser_state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut event = crate::openai::sse::responses::ResponsesEvent::ResponseCompleted(json!({
        "type": "response.completed",
        "sequence_number": 9,
        "response": {
            "id": "resp_move",
            "output": [{"type": "message", "text": "x".repeat(256 * 1024)}]
        }
    }));
    let original_text = event.payload()["response"]["output"][0]["text"]
        .as_str()
        .unwrap()
        .as_ptr();

    super::accumulator::accumulate_event(&mut ctx, &mut parser_state, &mut event);

    let retained = &ctx.extensions.get::<ResponsesState>().unwrap().response_object;
    assert_eq!(retained["output"][0]["text"].as_str().unwrap().as_ptr(), original_text);
    assert!(event.payload()["response"].is_null());
    assert_eq!(event.payload()["sequence_number"], 9);
}

#[test]
fn borrowed_terminal_serializes_canonical_response_without_cloning_it() {
    let response = json!({"id": "resp_borrow", "output": [{"text": "x".repeat(256 * 1024)}]});
    let metadata = json!({"type": "response.completed", "sequence_number": 7, "response": null});
    let expected = json!({
        "type": "response.completed",
        "sequence_number": 7,
        "response": response
    });
    let baseline = allocation_counter::measure(|| {
        std::hint::black_box(response.clone());
    });
    let terminal = super::BorrowedTerminalPayload {
        metadata: &metadata,
        response: &response,
    };
    let mut wire = Vec::with_capacity(serde_json::to_vec(&expected).unwrap().len());
    let borrowed = allocation_counter::measure(|| {
        serde_json::to_writer(&mut wire, &terminal).unwrap();
    });
    assert_eq!(
        wire,
        serde_json::to_vec(&expected).unwrap(),
        "terminal wire key order stays stable"
    );
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&wire).unwrap(), expected);
    assert!(baseline.bytes_total >= 256 * 1024);
    assert!(
        borrowed.bytes_total < baseline.bytes_total / 4,
        "borrowed serialization allocated {} bytes versus {} for a response clone",
        borrowed.bytes_total,
        baseline.bytes_total
    );
}

#[test]
fn bare_terminal_payload_keeps_its_original_wire_shape() {
    let response = json!({"id": "resp_bare", "output": []});
    let metadata = json!({"type": "response.completed", "id": "resp_bare", "output": []});
    let borrowed = super::BorrowedTerminalPayload {
        metadata: &metadata,
        response: &response,
    };
    assert_eq!(
        serde_json::to_vec(&borrowed).unwrap(),
        serde_json::to_vec(&metadata).unwrap(),
        "bare upstream terminal has no response wrapper to replace"
    );
}

#[test]
fn bare_upstream_terminal_remains_bare_after_deferred_emission() {
    let (filter, mut ctx) = make_armed_context();
    let mut terminal = Some(make_sse_chunk(
        "response.completed",
        &json!({"id": "resp_bare", "status": "completed", "output": []}),
    ));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    assert!(
        terminal.is_none(),
        "terminal remains deferred until stream finalization"
    );

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let wire = String::from_utf8(eos.unwrap().to_vec()).unwrap();
    let data = wire.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
    let payload: serde_json::Value = serde_json::from_str(data).unwrap();
    assert!(
        payload.get("response").is_none(),
        "bare upstream payload must stay bare: {wire}"
    );
    assert_eq!(payload["id"], "resp_bare");
}

#[test]
fn parse_error_sets_metadata() {
    let (filter, mut ctx) = make_armed_context();
    ctx.insert_filter_state(StreamEventsState {
        frame_parser: SseFrameParser::new(10),
        event_count: 0,
        max_events: 100_000,
        timeout: std::time::Duration::from_secs(300),
        started_at: None,
        completed_at: None,
        completion_state: CompletionState::Open,
        tool_call_args: std::collections::HashMap::new(),
        rejected_tool_call_args: std::collections::HashSet::new(),
        max_tool_call_argument_bytes: 1024 * 1024,
        max_accumulated_bytes: 64 * 1024 * 1024,
        max_output_items: 100_000,
        iteration: 0,
        output_index_offset: 0,
        deferred_terminal: None,
        deferred_done: false,
        local_items_flushed: false,
        local_tool_items: std::collections::HashMap::new(),
        client_tool_items: Vec::new(),
        stream_failed: false,
        shared_stable_bytes: std::sync::OnceLock::new(),
        shared_prior_output_bytes: std::sync::OnceLock::new(),
        shared_current_output_bytes: std::sync::atomic::AtomicUsize::new(0),
    });

    let large_chunk =
        Bytes::from("event: response.created\ndata: {\"id\": \"resp_overflow_test_with_a_very_long_payload\"}\n\n");
    let mut body = Some(large_chunk);
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert_eq!(
        ctx.get_metadata("responses.stream_parse_error"),
        Some("true"),
        "parse error should set metadata flag"
    );
}

#[test]
fn incomplete_client_tool_lifecycle_fails_closed() {
    use super::{
        CompletionState, StreamEventsState,
        client_tools::{ClientToolPhase, ClientToolStreamItem},
        validate_stream_end,
    };
    use crate::openai::responses::state::ClientToolRestore;

    let (_filter, mut ctx) = make_armed_context();
    ctx.insert_filter_state(StreamEventsState {
        frame_parser: SseFrameParser::new(10),
        event_count: 0,
        max_events: 100_000,
        timeout: std::time::Duration::from_secs(300),
        started_at: None,
        completed_at: None,
        // Terminal so the ONLY incomplete trigger is the stuck client-tool item.
        completion_state: CompletionState::TerminalLifecycle,
        tool_call_args: std::collections::HashMap::new(),
        rejected_tool_call_args: std::collections::HashSet::new(),
        max_tool_call_argument_bytes: 1024 * 1024,
        max_accumulated_bytes: 64 * 1024 * 1024,
        max_output_items: 100_000,
        iteration: 0,
        output_index_offset: 0,
        deferred_terminal: None,
        deferred_done: false,
        local_items_flushed: false,
        local_tool_items: std::collections::HashMap::new(),
        client_tool_items: vec![ClientToolStreamItem {
            key: "item:call_1".to_owned(),
            private_name: "custom_run_python".to_owned(),
            restore: ClientToolRestore::Custom,
            phase: ClientToolPhase::Opened, // never reached Done
            output_index: 0,
            item_id: Some("call_1".to_owned()),
        }],
        stream_failed: false,
        shared_stable_bytes: std::sync::OnceLock::new(),
        shared_prior_output_bytes: std::sync::OnceLock::new(),
        shared_current_output_bytes: std::sync::atomic::AtomicUsize::new(0),
    });

    validate_stream_end(&mut ctx);

    assert_eq!(
        ctx.get_metadata("responses.stream_incomplete"),
        Some("true"),
        "incomplete client-tool lifecycle must flag the stream incomplete"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_error_code"),
        Some("server_error"),
        "incomplete client-tool lifecycle must fail closed"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_error_message"),
        Some("upstream Responses stream did not terminate cleanly"),
        "fail-closed metadata must carry the unclean-termination message"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "must skip persisting a truncated client-tool restore"
    );
}

#[tokio::test]
async fn output_item_done_replaces_by_index() {
    let (filter, mut ctx) = make_armed_context();

    let added = json!({"item": {"type": "message", "id": "item_1", "content": []}});
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &added));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let done = json!({
        "output_index": 0,
        "item": {"type": "message", "id": "item_1", "content": [{"type": "output_text", "text": "final"}]}
    });
    let mut b2 = Some(make_sse_chunk("response.output_item.done", &done));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1, "should replace, not append");
    assert!(
        state.output_items()[0]["content"][0]["text"] == "final",
        "should have updated content"
    );
}

#[tokio::test]
async fn terminal_incomplete_sets_status() {
    let (filter, mut ctx) = make_armed_context();

    let payload = json!({
        "id": "resp_inc",
        "status": "incomplete",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [{"type": "message", "id": "item_1"}],
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        "incomplete_details": {"reason": "max_output_tokens"}
    });
    let mut body = Some(make_sse_chunk("response.incomplete", &payload));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["id"], "resp_inc");
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(ctx.get_metadata("responses.status"), Some("incomplete"));
}

#[tokio::test]
async fn terminal_failed_sets_status() {
    let (filter, mut ctx) = make_armed_context();

    let payload = json!({
        "id": "resp_fail",
        "status": "failed",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [],
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
        "error": {"code": "server_error", "message": "internal failure"}
    });
    let mut body = Some(make_sse_chunk("response.failed", &payload));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["id"], "resp_fail");
    assert_eq!(state.output_items().len(), 0);
    assert_eq!(ctx.get_metadata("responses.status"), Some("failed"));
}

#[tokio::test]
async fn output_item_done_replaces_by_id() {
    let (filter, mut ctx) = make_armed_context();

    let added = json!({"item": {"type": "message", "id": "item_A", "content": []}});
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &added));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let done = json!({
        "item": {"type": "message", "id": "item_A", "content": [{"type": "output_text", "text": "replaced"}]}
    });
    let mut b2 = Some(make_sse_chunk("response.output_item.done", &done));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1, "should replace by id, not append");
    assert_eq!(state.output_items()[0]["content"][0]["text"], "replaced");
}

#[test]
fn completed_output_revision_changes_on_same_length_done_but_not_text_delta() {
    let (filter, mut ctx) = make_armed_context();
    let mut responses = ResponsesState {
        accumulated_output: vec![json!({"text": "prior".repeat(200_000)})],
        ..ResponsesState::default()
    };
    responses.apply_retained_payload_limit(16 * 1_048_576);
    ctx.extensions.insert(responses);
    let stream = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert!(super::shared_retained_budget(&ctx, stream).unwrap().1.is_some());
    let added = json!({
        "output_index": 0,
        "item": {"type": "message", "id": "item_A", "content": [{"type": "output_text", "text": "aaaa"}]}
    });
    let mut body = Some(make_sse_chunk("response.output_item.added", &added));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    let responses = ctx.extensions.get::<ResponsesState>().unwrap();
    let stable_revision = responses.replay_stable_payload_revision;
    let before = responses.current_output_revision;
    let before_bytes = crate::openai::responses::state::retained_json_bytes(&responses.response_object).unwrap();
    let stream = ctx.get_filter_state::<StreamEventsState>().unwrap();
    let prior_cache = stream.shared_prior_output_bytes.get().unwrap().unwrap();
    assert!(prior_cache.matches(responses));

    for _ in 0..200 {
        let mut body = Some(make_sse_chunk(
            "response.output_text.delta",
            &json!({"output_index": 0, "content_index": 0, "delta": "x"}),
        ));
        filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    }
    assert_eq!(
        ctx.extensions.get::<ResponsesState>().unwrap().current_output_revision,
        before
    );

    let done = json!({
        "output_index": 0,
        "item": {"type": "message", "id": "item_A", "content": [{"type": "output_text", "text": "\u{0001}\u{0001}\u{0001}\u{0001}"}]}
    });
    let mut body = Some(make_sse_chunk("response.output_item.done", &done));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    let responses = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(responses.output_items().len(), 1);
    assert_eq!(
        responses.current_output_revision,
        before.and_then(|revision| revision.checked_add(1))
    );
    assert!(crate::openai::responses::state::retained_json_bytes(&responses.response_object).unwrap() > before_bytes);
    assert_eq!(responses.replay_stable_payload_revision, stable_revision);
    let stream = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert!(
        stream
            .shared_prior_output_bytes
            .get()
            .unwrap()
            .unwrap()
            .matches(responses)
    );
    for _ in 0..200 {
        assert!(super::shared_retained_budget(&ctx, stream).unwrap().1.is_some());
    }
}

#[tokio::test]
async fn upsert_tool_call_dedup() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_dup",
            "call_id": "call_dup",
            "name": "search",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let done1 = json!({"item_id": "fc_dup", "output_index": 0, "arguments": "{\"q\":\"v1\"}"});
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.done", &done1));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().output_items().len(), 1);

    let done2 = json!({"item_id": "fc_dup", "output_index": 0, "arguments": "{\"q\":\"v2\"}"});
    let mut b3 = Some(make_sse_chunk("response.function_call_arguments.done", &done2));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1, "should replace, not append duplicate");
    assert_eq!(state.output_items()[0]["arguments"], "{\"q\":\"v2\"}");
}

#[tokio::test]
async fn function_call_done_without_prior_deltas() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_no_delta",
            "call_id": "call_nd",
            "name": "get_time",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let done = json!({
        "item_id": "fc_no_delta",
        "output_index": 0,
        "arguments": "{\"tz\":\"UTC\"}"
    });
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.done", &done));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items().len(), 1);
    assert_eq!(
        state.output_items()[0]["arguments"],
        "{\"tz\":\"UTC\"}",
        "should use payload arguments when no deltas were accumulated"
    );
}

#[tokio::test]
async fn done_payload_wins_over_accumulated_deltas() {
    let (filter, mut ctx) = make_armed_context();

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_diff",
            "call_id": "call_diff",
            "name": "lookup",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let delta = json!({"item_id": "fc_diff", "output_index": 0, "delta": "{\"from\":\"delta\"}"});
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let done = json!({
        "item_id": "fc_diff",
        "output_index": 0,
        "arguments": "{\"from\":\"done_payload\"}"
    });
    let mut b3 = Some(make_sse_chunk("response.function_call_arguments.done", &done));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.output_items()[0]["arguments"],
        "{\"from\":\"done_payload\"}",
        "done-event arguments should take precedence over accumulated deltas"
    );
}

#[tokio::test]
async fn unknown_event_type_ignored() {
    let (filter, mut ctx) = make_armed_context();

    let payload = json!({"some_field": "some_value"});
    let mut body = Some(make_sse_chunk("response.future_event_type", &payload));
    let result = filter.on_response_body(&mut ctx, &mut body, false);

    assert!(result.is_ok(), "unknown event type should not error");
    assert!(body.is_some(), "body should pass through unchanged");
}

#[tokio::test]
async fn error_event_does_not_mutate_state() {
    let (filter, mut ctx) = make_armed_context();

    let payload = json!({"code": "server_error", "message": "something broke"});
    let mut body = Some(make_sse_chunk("error", &payload));
    let result = filter.on_response_body(&mut ctx, &mut body, false);

    assert!(result.is_ok(), "error event should not fail the filter");
    assert!(
        ctx.extensions.get::<ResponsesState>().is_none(),
        "error event should not create ResponsesState"
    );
}

#[tokio::test]
async fn error_after_terminal_lifecycle_is_accepted() {
    let (filter, mut ctx) = make_armed_context();

    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});
    let mut b1 = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let error = json!({"code": "server_error", "message": "late error"});
    let mut b2 = Some(make_sse_chunk("error", &error));
    let result = filter.on_response_body(&mut ctx, &mut b2, false);

    assert!(
        result.is_ok(),
        "first error after terminal lifecycle should be accepted"
    );
    assert!(
        ctx.get_metadata("responses.stream_parse_error").is_none(),
        "accepted error should not set parse error"
    );
}

#[tokio::test]
async fn second_error_after_terminal_is_rejected() {
    let (filter, mut ctx) = make_armed_context();

    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});
    let mut b1 = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let error1 = json!({"code": "server_error", "message": "first error"});
    let mut b2 = Some(make_sse_chunk("error", &error1));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let error2 = json!({"code": "server_error", "message": "second error"});
    let mut b3 = Some(make_sse_chunk("error", &error2));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    assert_eq!(
        ctx.get_metadata("responses.stream_parse_error"),
        Some("true"),
        "second error event should be rejected as EventAfterTerminal"
    );
}

#[tokio::test]
async fn resumed_round_error_does_not_persist_prior_round_success() {
    // A successful round followed by a resumed round that only emits a provider
    // `error` must not persist the previous round's completed response as the
    // logical result. Re-arming invalidates the prior `response_object`, and the
    // error round never repopulates it, so `build_record` skips persistence.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    // Round 1: a completed response carrying a tool call, transitioning the loop.
    filter.arm(&mut ctx);
    let function_call = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "weather__get",
        "arguments": "{}",
        "status": "completed"
    });
    let mut terminal = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {"id": "resp_first", "status": "completed", "output": [function_call.clone()]},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();
    ctx.filter_results
        .entry("openai_mcp_dispatch")
        .or_default()
        .set("action", "loop")
        .unwrap();
    let mut first_eos = None;
    filter.on_response_body(&mut ctx, &mut first_eos, true).unwrap();
    assert_eq!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .response_object
            .get("id")
            .and_then(serde_json::Value::as_str),
        Some("resp_first"),
        "round 1 completion should populate the response object"
    );

    // Resume round 2.
    let state = ctx.extensions.get_mut::<ResponsesState>().unwrap();
    state.iteration = 1;
    state.accumulated_output = vec![function_call, json!({"type": "mcp_call", "id": "mcp_1"})];
    mark_accumulated_output_executed(state);
    ctx.filter_results.remove("openai_mcp_dispatch");
    filter.arm(&mut ctx);

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .response_object
            .is_null(),
        "re-arming must invalidate the prior round's response object"
    );

    // Round 2 emits only a provider error, with no terminal lifecycle event.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_second", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    let mut error = Some(make_sse_chunk(
        "error",
        &json!({"code": "server_error", "message": "backend exploded"}),
    ));
    filter.on_response_body(&mut ctx, &mut error, false).unwrap();

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .response_object
            .is_null(),
        "a resumed provider error must not resurrect the prior round's success for persistence"
    );
}

#[tokio::test]
async fn tool_call_argument_bytes_cap_enforced() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 20").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    filter.arm(&mut ctx);

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_big",
            "call_id": "call_big",
            "name": "big_fn",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let delta1 = json!({"item_id": "fc_big", "output_index": 0, "delta": "0123456789"});
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta1));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let delta2 = json!({"item_id": "fc_big", "output_index": 0, "delta": "0123456789X"});
    let mut b3 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta2));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    let state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    assert!(
        !state.tool_call_args.contains_key("item:fc_big"),
        "exceeding max_tool_call_argument_bytes should drop the accumulator entry"
    );
    assert!(
        state.rejected_tool_call_args.contains("item:fc_big"),
        "an overflowing tool call should remain rejected"
    );
}

#[tokio::test]
async fn tool_call_argument_bytes_cap_rejects_restart_after_overflow() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 20").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    filter.arm(&mut ctx);

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_restart",
            "call_id": "call_restart",
            "name": "restart_fn",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut added = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();

    let oversized = json!({"item_id": "fc_restart", "output_index": 0, "delta": "012345678901234567890"});
    let mut delta = Some(make_sse_chunk("response.function_call_arguments.delta", &oversized));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();

    let restart = json!({"item_id": "fc_restart", "output_index": 0, "delta": "{\"x\":1}"});
    let mut delta = Some(make_sse_chunk("response.function_call_arguments.delta", &restart));
    filter.on_response_body(&mut ctx, &mut delta, false).unwrap();

    let done = json!({"item_id": "fc_restart", "output_index": 0});
    let mut done_body = Some(make_sse_chunk("response.function_call_arguments.done", &done));
    filter.on_response_body(&mut ctx, &mut done_body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "a rejected tool call should not be finalized"
    );
    assert_eq!(state.output_items()[0]["arguments"], "");
    assert_eq!(state.output_items()[0]["status"], "in_progress");
}

#[tokio::test]
async fn tool_call_argument_bytes_cap_rejects_oversized_done_payload() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 20").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    filter.arm(&mut ctx);

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_done_big",
            "call_id": "call_done_big",
            "name": "done_fn",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut added = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut added, false).unwrap();

    let done = json!({
        "item_id": "fc_done_big",
        "output_index": 0,
        "arguments": "012345678901234567890"
    });
    let mut done_body = Some(make_sse_chunk("response.function_call_arguments.done", &done));
    filter.on_response_body(&mut ctx, &mut done_body, false).unwrap();

    let retry = json!({
        "item_id": "fc_done_big",
        "output_index": 0,
        "arguments": "{\"x\":1}"
    });
    let mut retry_body = Some(make_sse_chunk("response.function_call_arguments.done", &retry));
    filter.on_response_body(&mut ctx, &mut retry_body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state.tool_calls.is_empty(),
        "an oversized done payload should keep the tool call rejected"
    );
    assert_eq!(state.output_items()[0]["arguments"], "");
    assert_eq!(state.output_items()[0]["status"], "in_progress");
}

#[tokio::test]
async fn tool_call_argument_bytes_within_limit() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("max_tool_call_argument_bytes: 50").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    filter.arm(&mut ctx);

    let item = json!({
        "item": {
            "type": "function_call",
            "id": "fc_ok",
            "call_id": "call_ok",
            "name": "small_fn",
            "arguments": "",
            "status": "in_progress"
        },
        "output_index": 0
    });
    let mut b1 = Some(make_sse_chunk("response.output_item.added", &item));
    filter.on_response_body(&mut ctx, &mut b1, false).unwrap();

    let delta = json!({"item_id": "fc_ok", "output_index": 0, "delta": "{\"k\":\"v\"}"});
    let mut b2 = Some(make_sse_chunk("response.function_call_arguments.delta", &delta));
    filter.on_response_body(&mut ctx, &mut b2, false).unwrap();

    let state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    assert_eq!(
        state.tool_call_args.get("item:fc_ok").unwrap(),
        "{\"k\":\"v\"}",
        "within-limit deltas should accumulate normally"
    );
}

#[tokio::test]
async fn on_response_disarms_for_non_2xx_status() {
    let (filter, mut ctx) = make_armed_context();
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "test setup should arm the SSE parser"
    );

    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.status = http::StatusCode::BAD_REQUEST;
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    ctx.response_header = Some(resp);

    filter.on_response(&mut ctx).await.unwrap();

    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "filter should be disarmed for non-2xx response"
    );
}

#[tokio::test]
async fn on_response_disarms_for_non_sse_content_type() {
    let (filter, mut ctx) = make_armed_context();

    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    ctx.response_header = Some(resp);

    filter.on_response(&mut ctx).await.unwrap();

    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "filter should be disarmed for non-SSE content type"
    );
}

#[tokio::test]
async fn on_response_stays_armed_for_sse_with_charset() {
    let (filter, mut ctx) = make_armed_context();
    ctx.extensions.insert(ResponsesState {
        local_completion_response_template: json!({
            "id":"resp_prior", "object":"response", "status":"completed", "output":[]
        }),
        ..ResponsesState::default()
    });

    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    ctx.response_header = Some(resp);

    filter.on_response(&mut ctx).await.unwrap();

    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "filter should stay armed for text/event-stream with charset parameter"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .local_completion_response_template
            .is_null(),
        "upstream response headers make the request-side fallback unreachable"
    );
}

#[tokio::test]
async fn on_request_keeps_accept_encoding_when_not_arming() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "false".to_owned());
    ctx.current_filter_id = Some(0);

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "non-streaming request should continue without arming"
    );
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "non-streaming request must not arm"
    );
    assert!(
        !ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
        "Accept-Encoding must only be stripped when logical parsing is armed"
    );
}

#[tokio::test]
async fn on_response_disarms_for_content_encoded_sse() {
    let (filter, mut ctx) = make_armed_context();

    // A non-compliant backend returns a gzip-encoded event stream despite the
    // stripped Accept-Encoding. The raw SSE parser cannot decode it, so the
    // filter must decline rather than parse opaque bytes into a spurious error.
    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    resp.headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
    ctx.response_header = Some(resp);

    filter.on_response(&mut ctx).await.unwrap();

    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_none(),
        "filter should disarm for a Content-Encoding SSE response it cannot parse"
    );
}

#[tokio::test]
async fn disarmed_filter_passes_error_body_through() {
    let (filter, mut ctx) = make_armed_context();

    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.status = http::StatusCode::BAD_REQUEST;
    resp.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    ctx.response_header = Some(resp);
    filter.on_response(&mut ctx).await.unwrap();

    let error_json = r#"{"error":{"message":"bad request","type":"invalid_request_error"}}"#;
    let mut body = Some(Bytes::from(error_json));
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        error_json.as_bytes(),
        "error body should pass through unchanged after disarming"
    );
}

#[tokio::test]
async fn on_response_preserves_content_length_when_not_armed() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);

    let resp = Box::leak(Box::new(crate::test_utils::make_response()));
    resp.headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("1234"));
    ctx.response_header = Some(resp);

    filter.on_response(&mut ctx).await.unwrap();

    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::CONTENT_LENGTH)
            .is_some(),
        "Content-Length should be preserved when filter is not armed"
    );
}

#[tokio::test]
async fn apply_arm_does_not_cap_timeout_before_first_chunk() {
    use std::sync::Arc;

    use praxis_core::connectivity::{ConnectionOptions, Upstream};

    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_secs: 1").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    ctx.upstream = Some(Upstream {
        address: Arc::from("127.0.0.1:9"),
        authority: None,
        tls: None,
        connection: Arc::new(ConnectionOptions::default()),
    });

    filter.apply_arm_decision(&mut ctx, ArmDecision::Arm);

    assert!(
        ctx.upstream.as_ref().and_then(|u| u.connection.read_timeout).is_none(),
        "timeout_secs must not cap upstream reads before the first SSE chunk"
    );
}

#[tokio::test]
async fn apply_arm_does_not_cap_before_load_balancing() {
    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_secs: 1").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);
    ctx.upstream = None;

    filter.apply_arm_decision(&mut ctx, ArmDecision::Arm);

    assert!(
        OpenaiStreamEventsFilter::is_armed(&ctx),
        "arming must still install parser state when ctx.upstream is unset"
    );
    assert!(
        ctx.upstream.is_none(),
        "timeout_secs cannot invent a peer before load balancing"
    );
}

#[tokio::test]
async fn apply_arm_keeps_a_tighter_cluster_read_timeout() {
    use std::{sync::Arc, time::Duration};

    use praxis_core::connectivity::{ConnectionOptions, Upstream};

    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_secs: 300").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);
    ctx.upstream = Some(Upstream {
        address: Arc::from("127.0.0.1:9"),
        authority: None,
        tls: None,
        connection: Arc::new(ConnectionOptions {
            read_timeout: Some(Duration::from_millis(250)),
            ..ConnectionOptions::default()
        }),
    });

    filter.apply_arm_decision(&mut ctx, ArmDecision::Arm);

    assert_eq!(
        ctx.upstream.as_ref().and_then(|u| u.connection.read_timeout),
        Some(Duration::from_millis(250)),
        "arming must not relax a tighter cluster read timeout before the first chunk"
    );
}

#[test]
fn stream_deadline_is_none_before_first_chunk() {
    let (_filter, ctx) = make_armed_context();
    let state = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert!(
        super::stream_deadline_at(state).is_none(),
        "timeout_secs must not start before the first SSE chunk"
    );
}

#[test]
fn stream_deadline_is_absolute_from_first_chunk() {
    use std::time::Duration;

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.get_filter_state::<StreamEventsState>().unwrap();
    let started = state.started_at.expect("first chunk must start the deadline");
    assert_eq!(
        super::stream_deadline_at(state),
        Some(started + state.timeout),
        "the cutoff must stay anchored at first-chunk + timeout_secs, not restart on each poll"
    );
    assert_eq!(state.timeout, Duration::from_secs(300));
}

#[test]
fn first_chunk_recaps_absolute_deadline_onto_live_body() {
    use std::time::Duration;

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx
        .get_filter_state::<StreamEventsState>()
        .expect("parser state must remain installed");
    let started = state.started_at.expect("first chunk must start the deadline");
    let expected = started + state.timeout;
    let applied = ctx
        .stream_deadline_cap()
        .expect("the first chunk must publish an absolute deadline");
    assert_eq!(applied, expected);
    assert!(
        state.timeout <= Duration::from_secs(300),
        "unexpected test timeout budget: {:?}",
        state.timeout
    );
}

#[test]
fn stream_deadline_cap_shrinks_after_first_chunk() {
    use std::time::Duration;

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    state.started_at.expect("first chunk must start the deadline");
    state.timeout = Duration::from_secs(1);
    let adjusted_started = std::time::Instant::now() - Duration::from_millis(750);
    state.started_at = Some(adjusted_started);
    let expected = adjusted_started + state.timeout;
    ctx.insert_filter_state(state);

    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "again"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert_eq!(
        ctx.stream_deadline_cap(),
        Some(expected),
        "each chunk must republish the same absolute cutoff, not a fresh relative timer"
    );
}

#[test]
fn stream_deadline_caps_live_body_after_first_chunk() {
    use std::time::Duration;

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let started = state.started_at.expect("first chunk must start the deadline");
    state.timeout = Duration::from_secs(1);
    let expected = started + state.timeout;
    ctx.insert_filter_state(state);
    super::recap_stream_deadline(&mut ctx, expected);

    assert_eq!(
        ctx.stream_deadline_cap(),
        Some(expected),
        "absolute cutoff must be published on the live body"
    );
}

#[tokio::test]
async fn stream_deadline_recaps_live_body_on_irr_body_context() {
    use std::{sync::Arc, time::Duration};

    use praxis_core::connectivity::{ConnectionOptions, Upstream};

    let yaml: serde_yaml::Value = serde_yaml::from_str("timeout_secs: 1").unwrap();
    let filter = OpenaiStreamEventsFilter::build(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    ctx.upstream = Some(Upstream {
        address: Arc::from("127.0.0.1:9"),
        authority: None,
        tls: None,
        connection: Arc::new(ConnectionOptions {
            read_timeout: Some(Duration::from_secs(30)),
            ..ConnectionOptions::default()
        }),
    });

    filter.apply_arm_decision(&mut ctx, ArmDecision::Arm);

    // IRR reconstructs response-body contexts with upstream: None.
    ctx.upstream = None;
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let started = state.started_at.expect("first chunk must start the deadline");
    state.timeout = Duration::from_secs(1);
    state.started_at = Some(started.checked_sub(Duration::from_millis(750)).unwrap_or(started));
    let expected = state.started_at.unwrap() + state.timeout;
    ctx.insert_filter_state(state);

    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert_eq!(
        ctx.stream_deadline_cap(),
        Some(expected),
        "absolute cutoff must be published for the live body even when ctx.upstream is None"
    );
}

#[test]
fn io_before_first_chunk_is_not_a_stream_timeout() {
    use std::time::Instant;

    let (_filter, ctx) = make_armed_context();
    let state = ctx.get_filter_state::<StreamEventsState>().unwrap();
    assert!(
        !super::io_exceeded_stream_deadline(state, Instant::now()),
        "a transport reset before any SSE chunk is not the stream deadline"
    );
}

#[test]
fn io_before_deadline_is_not_a_stream_timeout() {
    use std::time::{Duration, Instant};

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    state.timeout = Duration::from_secs(300);
    assert!(
        !super::io_exceeded_stream_deadline(&state, Instant::now()),
        "a truncated chunk must not be labelled a timeout"
    );
}

#[test]
fn io_after_deadline_is_a_stream_timeout() {
    use std::time::Duration;

    let (filter, mut ctx) = make_armed_context();
    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let started = state.started_at.expect("first chunk must start the deadline");
    state.timeout = Duration::from_secs(1);
    assert!(
        super::io_exceeded_stream_deadline(&state, started + Duration::from_secs(1)),
        "an Io abort after timeout_secs from the first chunk is the stream deadline"
    );
}

#[tokio::test]
async fn idle_timeout_does_not_fail_a_completed_sse_stream() {
    let (filter, mut ctx) = make_armed_context();

    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});
    let mut body = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    // Transport-level IdleTimeout cannot be injected here (`StreamTermination`
    // is crate-private). The completeness guard is the same function that
    // `record_idle_transport_timeout` calls before mark_stream_termination_handled.
    super::publish_idle_timeout_if_incomplete(&mut ctx);
    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();

    assert_eq!(
        ctx.get_filter_state::<StreamEventsState>()
            .map(|state| state.completion_state),
        Some(CompletionState::TerminalLifecycle),
        "response.completed must leave the parser in a terminal lifecycle"
    );
    assert!(
        ctx.get_metadata("responses.stream_error_code").is_none(),
        "a terminal lifecycle event must not be rewritten as a transport timeout"
    );
    assert!(
        ctx.get_metadata("responses.skip_persist").is_none(),
        "a completed SSE stream must remain persistable when the HTTP body closes slowly"
    );
}

#[tokio::test]
async fn idle_timeout_fails_an_open_sse_stream() {
    let (filter, mut ctx) = make_armed_context();

    let mut body = Some(make_sse_chunk("response.output_text.delta", &json!({"text": "hi"})));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    super::publish_idle_timeout_if_incomplete(&mut ctx);

    assert_eq!(
        ctx.get_metadata("responses.stream_error_code"),
        Some("server_error"),
        "an idle abort before a terminal event is a stream timeout"
    );
    assert_eq!(
        ctx.get_metadata("responses.skip_persist"),
        Some("true"),
        "an incomplete idle abort must not be persisted"
    );
}

// Test helpers for Task 4 file_search classification/suppression tests
fn test_ctx_with_hosted_file_search_tool() -> praxis_filter::HttpFilterContext<'static> {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);
    let mut state = ResponsesState::default();
    state.tools = vec![json!({"type": "file_search", "vector_store_ids": ["vs_1"]})];
    ctx.extensions.insert(state);
    ctx
}

fn test_ctx_without_file_search_tool() -> praxis_filter::HttpFilterContext<'static> {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);
    let state = ResponsesState::default();
    ctx.extensions.insert(state);
    ctx
}

fn responses_event(event_type: &str, mut payload: serde_json::Value) -> crate::openai::sse::responses::ResponsesEvent {
    use crate::openai::sse::{SseFrame, responses::ResponsesEvent};
    // Ensure the payload has the required "type" field matching the event_type
    if let serde_json::Value::Object(ref mut obj) = payload {
        obj.insert("type".to_owned(), serde_json::Value::String(event_type.to_owned()));
    }
    let data = serde_json::to_vec(&payload).unwrap();
    let frame = SseFrame {
        event_type: Some(event_type.to_owned()),
        data,
    };
    ResponsesEvent::from_frame(&frame).unwrap()
}

impl OpenaiStreamEventsFilter {
    fn test_filter() -> Self {
        Self {
            parser_config: crate::openai::sse::SseParserConfig {
                max_buffer_bytes: 10_485_760,
                max_events: 100_000,
                timeout: std::time::Duration::from_secs(300),
            },
            max_tool_call_argument_bytes: 1024 * 1024,
            max_accumulated_bytes: 64 * 1024 * 1024,
            max_output_items: 100_000,
        }
    }
}

#[test]
fn suppress_mode_drops_private_function_call_lifecycle() {
    use super::{StreamEventsState, append_logical_event, local_tools::LocalToolMode};

    // A private function_call(name=file_search) opened while a hosted file_search
    // tool is declared: its added/delta/done all suppressed (nothing emitted).
    let mut ctx = test_ctx_with_hosted_file_search_tool();
    let filter = OpenaiStreamEventsFilter::test_filter();
    filter.arm(&mut ctx);
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();

    let added = responses_event(
        "response.output_item.added",
        json!({"output_index": 0, "item": {"id": "fc_1", "type": "function_call", "name": "file_search", "arguments": "{\"query\":\"x\"}"}}),
    );
    let delta = responses_event(
        "response.function_call_arguments.delta",
        json!({"item_id": "fc_1", "delta": "..."}),
    );
    let args_done = responses_event(
        "response.function_call_arguments.done",
        json!({"item_id": "fc_1", "arguments": "{\"query\":\"x\"}"}),
    );
    let item_done = responses_event(
        "response.output_item.done",
        json!({"output_index": 0, "item": {"id": "fc_1", "type": "function_call", "name": "file_search"}}),
    );
    let mut out = Vec::new();
    for event in [added, delta, args_done, item_done] {
        append_logical_event(&mut state, &mut ctx, event, &mut out);
    }
    assert!(
        out.is_empty(),
        "every event for a Suppress item (added/delta/arguments.done/output_item.done) must be dropped"
    );
    assert_eq!(state.local_tool_items.get("item:fc_1"), Some(&LocalToolMode::Suppress));
}

#[test]
fn client_function_call_without_hosted_tool_passes_through() {
    use super::{StreamEventsState, append_logical_event};

    // has_file_search_tool == false: not suppressed, reaches the wire (P1 round-11).
    let mut ctx = test_ctx_without_file_search_tool();
    let filter = OpenaiStreamEventsFilter::test_filter();
    filter.arm(&mut ctx);
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let added = responses_event(
        "response.output_item.added",
        json!({"output_index": 0, "item": {"id": "fc_1", "type": "function_call", "name": "file_search"}}),
    );
    let mut out = Vec::new();
    append_logical_event(&mut state, &mut ctx, added, &mut out);
    assert!(
        !out.is_empty(),
        "client function_call must pass through when no hosted tool is declared"
    );
    assert!(
        state.local_tool_items.is_empty(),
        "nothing classified without a hosted tool"
    );
}

#[test]
fn native_hybrid_drops_pending_done_and_passes_opening() {
    use super::{StreamEventsState, append_logical_event};

    let mut ctx = test_ctx_with_hosted_file_search_tool();
    let filter = OpenaiStreamEventsFilter::test_filter();
    filter.arm(&mut ctx);
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let added = responses_event(
        "response.output_item.added",
        json!({"output_index": 0, "item": {"id": "fs_1", "type": "file_search_call", "status": "searching"}}),
    );
    let pending_done = responses_event(
        "response.output_item.done",
        json!({"output_index": 0, "item": {"id": "fs_1", "type": "file_search_call", "status": "searching"}}),
    );
    let mut out = Vec::new();
    append_logical_event(&mut state, &mut ctx, added, &mut out);
    assert!(!out.is_empty(), "native opening passes through");
    out.clear();
    append_logical_event(&mut state, &mut ctx, pending_done, &mut out);
    assert!(
        out.is_empty(),
        "a still-pending output_item.done is dropped (no double-done)"
    );
    assert!(
        state.local_tool_items.contains_key("item:fs_1"),
        "item stays registered for EOS synthesis"
    );
}

#[test]
fn native_terminal_done_passes_and_cancels_synthesis() {
    use super::{StreamEventsState, append_logical_event};

    let mut ctx = test_ctx_with_hosted_file_search_tool();
    let filter = OpenaiStreamEventsFilter::test_filter();
    filter.arm(&mut ctx);
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let added = responses_event(
        "response.output_item.added",
        json!({"output_index": 0, "item": {"id": "fs_1", "type": "file_search_call", "status": "searching"}}),
    );
    let terminal_done = responses_event(
        "response.output_item.done",
        json!({"output_index": 0, "item": {"id": "fs_1", "type": "file_search_call", "status": "completed", "results": []}}),
    );
    let mut out = Vec::new();
    append_logical_event(&mut state, &mut ctx, added, &mut out);
    out.clear();
    append_logical_event(&mut state, &mut ctx, terminal_done, &mut out);
    assert!(!out.is_empty(), "a terminal (completed) done passes through");
    assert!(
        !state.local_tool_items.contains_key("item:fs_1"),
        "keys removed → EOS synthesis cancelled"
    );
    assert!(!state.local_tool_items.contains_key("index:0"));
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .provider_streamed_terminal_ids
            .contains("fs_1"),
        "provider-streamed terminal id recorded so the EOS reconcile skips re-queuing it (P1)"
    );
}

#[test]
fn native_failed_done_records_observation_for_reconcile_skip() {
    use super::{StreamEventsState, append_logical_event};

    // #313 P1: a provider-streamed FAILED (or incomplete) native done also passes through
    // and must be recorded — the reconcile skips by observed membership, not status, so a
    // synthesized tail cannot duplicate this live terminal done.
    let mut ctx = test_ctx_with_hosted_file_search_tool();
    let filter = OpenaiStreamEventsFilter::test_filter();
    filter.arm(&mut ctx);
    let mut state = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let added = responses_event(
        "response.output_item.added",
        json!({"output_index": 0, "item": {"id": "fs_9", "type": "file_search_call", "status": "searching"}}),
    );
    let failed_done = responses_event(
        "response.output_item.done",
        json!({"output_index": 0, "item": {"id": "fs_9", "type": "file_search_call", "status": "failed"}}),
    );
    let mut out = Vec::new();
    append_logical_event(&mut state, &mut ctx, added, &mut out);
    out.clear();
    append_logical_event(&mut state, &mut ctx, failed_done, &mut out);
    assert!(!out.is_empty(), "a terminal (failed) done passes through");
    assert!(
        !state.local_tool_items.contains_key("item:fs_9"),
        "keys removed → EOS synthesis cancelled"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .provider_streamed_terminal_ids
            .contains("fs_9"),
        "a provider-streamed FAILED native is recorded so reconcile skips it (P1)"
    );
}

#[tokio::test]
async fn logical_stream_finalize_clears_provider_streamed_terminal_ids() {
    // #313 P1 (DoS bound): file_search's EOS reconcile (a prior response-phase filter) consumes
    // this round's observation set; finalize must clear it so it cannot accumulate across IRR
    // continuation rounds and bypass max_state_bytes. Seeding it stands in for a round in which
    // stream_events observed a provider-streamed native terminal done.
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    // Unit tests cannot construct the IRR-owned `IterationState`; arm directly.
    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_first", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let mut terminal = Some(make_sse_chunk(
        "response.completed",
        &json!({
            "response": {"id": "resp_first", "status": "completed", "output": []},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut terminal, false).unwrap();

    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .provider_streamed_terminal_ids
        .insert("fs_round_n".to_owned());

    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();

    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .provider_streamed_terminal_ids
            .is_empty(),
        "finalize must clear the per-round observation set so it cannot grow across IRR rounds (P1 DoS bound)"
    );
}

#[test]
fn logical_stream_continues_recognizes_owner_loop() {
    use super::logical_stream_continues;

    // After the #1046 unification the loop owner (`openai_agentic_loop`) is the
    // single continuation authority: its `action="loop"` — covering pending
    // file_search assignments, web_search calls, and MCP-classified tool calls
    // alike — is what keeps the logical stream open.
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "loop")
        .unwrap();
    assert!(
        logical_stream_continues(&ctx),
        "owner action=loop must continue the logical stream"
    );
}

#[test]
fn drain_emits_absolute_output_index() {
    // accumulated len 6; the reconciled call sits at absolute index 5. The queue
    // stores absolute indices and drain normalizes with a zero offset, so the wire
    // `output_index` is the absolute 5 (no per-round offset arithmetic).
    let mut ctx = test_ctx_without_file_search_tool();
    let mut state = ResponsesState::default();
    state.accumulated_output = (0..6)
        .map(|_| json!({"type":"file_search_call","status":"completed","results":[]}))
        .collect();
    state.pending_local_tool_synthesis = vec![(5, SynthesisKind::Native)];
    ctx.extensions.insert(state);
    let mut out = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut ctx, &mut out);
    let emitted = String::from_utf8(out).unwrap();
    assert!(
        emitted.contains("\"output_index\":5"),
        "absolute index 5 is emitted verbatim as the wire output_index; got: {emitted}"
    );
}

#[test]
fn drain_defers_still_pending_item() {
    // A queued item whose owner placeholder is not yet reconciled (the dispatcher
    // runs next round) is re-queued for a later finalize, not synthesized now.
    let mut ctx = test_ctx_without_file_search_tool();
    let mut state = ResponsesState::default();
    state.accumulated_output = vec![json!({"type":"file_search_call","status":"in_progress"})];
    state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Private)];
    ctx.extensions.insert(state);
    let mut out = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut ctx, &mut out);
    assert!(out.is_empty(), "a still-pending item emits no lifecycle yet");
    assert_eq!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .pending_local_tool_synthesis,
        vec![(0, SynthesisKind::Private)],
        "the pending item is re-queued for the finalize that follows its reconciliation"
    );
}

#[test]
fn drain_pre_existing_error_suppresses_synthesis() {
    let mut ctx = test_ctx_without_file_search_tool();
    ctx.set_metadata("responses.stream_error_code", "server_error"); // set before drain
    let mut state = ResponsesState::default();
    state.accumulated_output = vec![json!({"type":"file_search_call","status":"completed","results":[]})];
    state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Native)]; // a VALID queued item
    ctx.extensions.insert(state);
    let mut out = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut ctx, &mut out);
    assert!(
        out.is_empty(),
        "a valid queued item is discarded when an error is already committed"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .pending_local_tool_synthesis
            .is_empty(),
        "queue drained once regardless of the error short-circuit"
    );
}

#[test]
fn drain_invalid_index_sets_error_and_no_gap() {
    let mut ctx = test_ctx_without_file_search_tool();
    let mut state = ResponsesState::default();
    state.accumulated_output = vec![json!({"type":"file_search_call","status":"completed","results":[]})];
    // Absolute index 5 is out of range for a one-item output → invariant failure.
    state.pending_local_tool_synthesis = vec![(5, SynthesisKind::Native)];
    ctx.extensions.insert(state);
    let mut out = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut ctx, &mut out);
    assert!(out.is_empty(), "no guessed frame on invariant failure");
    assert!(
        ctx.get_metadata("responses.stream_error_code").is_some(),
        "five-write set"
    );
}

// §10 P0: native-progress-precedes-closed-error ordering (no gap, no rewind).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_progress_precedes_closed_error_ordering() {
    let filter = OpenaiStreamEventsFilter::test_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    let mut state = ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    }));
    state.tools = vec![json!({"type": "file_search", "vector_store_ids": ["vs_1"]})];
    ctx.extensions.insert(state);

    // Unit tests cannot construct the IRR-owned `IterationState`; arm directly.
    filter.arm(&mut ctx);

    // Drive one native file_search progress frame through to get a sequence_number
    let mut progress = Some(make_sse_chunk(
        "response.file_search_call.in_progress",
        &json!({
            "response_id": "resp_fail",
            "output_index": 0,
            "item": {"type": "file_search_call", "id": "fs_1", "status": "searching"},
            "sequence_number": 1
        }),
    ));
    filter.on_response_body(&mut ctx, &mut progress, false).unwrap();
    let progress_out = String::from_utf8(progress.unwrap().to_vec()).unwrap();
    // Extract the normalized sequence_number from the emitted frame
    let last_live_sequence: u64 = progress_out
        .lines()
        .find(|line| line.starts_with("data:"))
        .and_then(|line| serde_json::from_str::<serde_json::Value>(&line[5..]).ok())
        .and_then(|v| v.get("sequence_number").and_then(serde_json::Value::as_u64))
        .expect("native progress frame must have sequence_number");

    // Set up closed-failure state: file_search publishes action=done + logical_stream_error
    ctx.filter_results
        .entry("openai_file_search_callout")
        .or_default()
        .set("action", "done")
        .unwrap();
    ctx.set_metadata("responses.stream_error_code", "server_error");
    ctx.set_metadata("responses.stream_error_message", "file_search failed");

    // Call finalizer → should emit error frame
    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos = String::from_utf8(eos.unwrap().to_vec()).unwrap();

    // Assert (a) error frame's sequence_number == last_live_sequence + 1
    assert!(eos.contains("event: error"), "EOS must contain error frame");
    let error_sequence: u64 = eos
        .lines()
        .find(|line| line.starts_with("data:"))
        .and_then(|line| serde_json::from_str::<serde_json::Value>(&line[5..]).ok())
        .and_then(|v| v.get("sequence_number").and_then(serde_json::Value::as_u64))
        .expect("error frame must have sequence_number");
    assert_eq!(
        error_sequence,
        last_live_sequence + 1,
        "error frame sequence_number == last_live + 1 (no gap, no rewind)"
    );

    // Assert (b) error frame is the last frame (no frames after it)
    let event_count = eos.matches("event:").count();
    assert_eq!(
        event_count, 1,
        "error frame is the last frame (no retraction of earlier frames)"
    );

    // Assert (c) earlier native progress frame was emitted (verified by progress_out above)
    assert!(
        progress_out.contains("file_search_call.in_progress"),
        "native progress frame was emitted and not retracted"
    );
}

// §10 P1: synthesis atomicity validate-all-before-emit (valid item before invalid → both suppressed).
#[test]
fn drain_atomicity_valid_item_before_invalid_both_suppressed() {
    let mut ctx = test_ctx_without_file_search_tool();
    // accumulated len 6; queue [(5,Native)=valid, (99,Native)=out of range]. The
    // resolution pass validates every index before emitting any frame, so the
    // out-of-range item suppresses the valid one too.
    let mut state = ResponsesState::default();
    state.accumulated_output = (0..6)
        .map(|_| json!({"type":"file_search_call","status":"completed","results":[]}))
        .collect();
    state.pending_local_tool_synthesis = vec![(5, SynthesisKind::Native), (99, SynthesisKind::Native)];
    ctx.extensions.insert(state);
    let mut out = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut ctx, &mut out);
    assert!(
        out.is_empty(),
        "atomicity: the VALID item is not emitted when a later queued item is invalid"
    );
    assert!(
        ctx.get_metadata("responses.stream_error_code").is_some(),
        "validation failure sets the five-write"
    );
}

#[test]
fn file_search_synthesis_peak_is_rejected_before_draining() {
    let item = json!({
        "type": "file_search_call",
        "id": "fs_peak",
        "status": "completed",
        "queries": ["q"],
        "results": [{"file_id": "file-a", "text": "x".repeat(32 * 1024)}]
    });
    // The old post-drain check saw only the serialized output. Measure that
    // wire size separately to show it fits while the simultaneous ready clone,
    // lifecycle payload, and output buffer do not.
    let mut preview_ctx = test_ctx_without_file_search_tool();
    let mut preview_state = ResponsesState::default();
    preview_state.accumulated_output = vec![item.clone()];
    preview_state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Private)];
    preview_ctx.extensions.insert(preview_state);
    let mut preview = Vec::new();
    super::local_tools::drain_local_tool_synthesis(&mut preview_ctx, &mut preview);

    let (_filter, mut ctx) = make_armed_context();
    let mut state = ResponsesState::from_request_body(json!({"model": "m", "input": "q", "stream": true}));
    state.accumulated_output = vec![item.clone()];
    state.response_object = json!({
        "id": "resp_peak", "object": "response", "status": "completed", "output": [item]
    });
    state.logical_stream_response_id = Some("resp_peak".to_owned());
    state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Private)];
    let baseline = state.retained_payload_bytes().unwrap();
    let limit = 128 * 1024;
    let staging = super::ready_file_search_synthesis_staging_bytes(&state).unwrap();
    assert!(
        baseline + preview.len() < limit,
        "the old post-drain check would admit the wire copy"
    );
    assert!(
        baseline + staging > limit,
        "the simultaneous synthesis owners exceed the ceiling"
    );
    state.apply_retained_payload_limit(limit);
    ctx.extensions.insert(state);
    ctx.filter_results
        .entry("openai_agentic_loop")
        .or_default()
        .set("action", "loop")
        .unwrap();

    let mut eos = None;
    super::finalize_logical_stream(&mut ctx, &mut eos);
    let wire = String::from_utf8(eos.expect("bounded error frame").to_vec()).unwrap();
    assert_eq!(wire.matches("event: error").count(), 1, "one terminal error: {wire}");
    assert!(
        !wire.contains("response.output_item.done"),
        "no synthesis escaped: {wire}"
    );
    assert!(
        !wire.contains("event: response.completed"),
        "no success terminal escaped: {wire}"
    );
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert_eq!(
        ctx.filter_results
            .get("openai_agentic_loop")
            .and_then(|result| result.get("action")),
        Some("done")
    );
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

#[test]
fn local_completion_charges_ready_file_search_clone_before_drain() {
    let mut ctx = test_ctx_without_file_search_tool();
    let item = json!({
        "type": "file_search_call", "id": "fs_local_peak", "status": "completed",
        "results": [{"file_id": "file-a", "text": "x".repeat(8 * 1024)}]
    });
    let mut state = ResponsesState::from_request_body(json!({"model": "m", "input": "q", "stream": true}));
    state.accumulated_output = vec![item.clone()];
    state.response_object = json!({"id": "resp_local_peak", "object": "response", "output": [item]});
    state.pending_local_tool_synthesis = vec![(0, SynthesisKind::Private)];
    let baseline = state.retained_payload_bytes().unwrap();
    let wire_bound = super::local_terminal_output_upper_bound(&state).unwrap();
    let old_projection = super::canonicalization_staging_bytes(&state, wire_bound).unwrap();
    let ready_clone = super::ready_file_search_clone_bytes(&state).unwrap();
    let limit = baseline + old_projection + ready_clone / 2;
    state.apply_retained_payload_limit(limit);
    assert!(
        state.can_retain_payload(old_projection),
        "the old local-completion preflight admitted the peak"
    );
    ctx.extensions.insert(state);

    let wire = String::from_utf8(encode_local_completion(&mut ctx).unwrap().to_vec()).unwrap();
    assert_eq!(wire.matches("event: error").count(), 1, "one bounded error: {wire}");
    assert!(
        !wire.contains("response.output_item.done"),
        "no synthesis escaped: {wire}"
    );
    assert_eq!(ctx.get_metadata("responses.skip_persist"), Some("true"));
    assert!(ctx.extensions.get::<ResponsesState>().unwrap().retained_payload_failed);
}

// #1159 Task 4: end-to-end proof that a lowered `Namespace` member is retyped in
// place through the real commit path (`commit_chunk_events` ->
// `restore_and_append_chunk` -> `apply_client_tool_disposition`), asserting the
// EMITTED SSE bytes carry the restored member name + namespace and never leak the
// private lowered name. This is the one client-visible, security-relevant behavior
// the task ships, and it exercises the payload-mutating apply code the unit tests
// in `client_tools.rs` cannot reach.
#[tokio::test]
async fn namespace_lowered_call_retyped_in_place_through_commit_never_leaks_private_name() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    // Arm a Namespace lowering on the shared `ResponsesState`, inserted the same
    // way `restore_and_append_chunk` reads it (`ctx.extensions.get::<ResponsesState>()`).
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "agentic_ns__fs__read".to_owned(),
            LoweredClientTool {
                original_name: "read".to_owned(),
                namespace: Some("fs".to_owned()),
                restore: ClientToolRestore::Namespace,
            },
        );

    filter.arm(&mut ctx);

    // Open the logical response so the incremental item events are delivered.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // One chunk carrying the lowered `function_call`'s whole lifecycle: added ->
    // arguments.done -> item.done for the same item.
    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read", "call_id": "c1", "id": "fc_1"},
            "sequence_number": 1
        }),
    );
    let args_done = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"path\":\"/etc\"}",
            "sequence_number": 2
        }),
    );
    let item_done = make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read", "call_id": "c1", "id": "fc_1",
                     "arguments": "{\"path\":\"/etc\"}", "status": "completed"},
            "sequence_number": 3
        }),
    );
    let mut chunk = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&args_done);
        buf.extend_from_slice(&item_done);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    let emitted = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
    assert!(
        emitted.contains(r#""name":"read""#),
        "the restored member name must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""namespace":"fs""#),
        "the namespace must be re-added on the restored item: {emitted}"
    );
    assert!(
        !emitted.contains("agentic_ns__fs__read"),
        "the private lowered name must never leak to the client: {emitted}"
    );
}

// #1159 Task 5: end-to-end proof that a lowered `custom` tool is restored to the
// canonical `custom_tool_call` lifecycle through the real commit path
// (`commit_chunk_events` phase-2a artifact capture -> `restore_and_append_chunk` ->
// `apply_client_tool_disposition`). This exercises the whole synthesis the unit
// tests in `client_tools.rs` cannot reach: the phase-2a `find_output_item` capture
// of the completed private `function_call` item, the retyped `output_item.added`,
// the suppressed backend args frame replaced by synthesized `custom_tool_call_input`
// delta+done, and the restored `output_item.done`. Asserts the EMITTED SSE bytes
// carry the public `custom_tool_call` shape and never leak the private `fc_` id.
#[tokio::test]
async fn custom_lowered_call_restored_to_custom_tool_call_lifecycle_through_commit() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    // Arm a Custom lowering on the shared `ResponsesState`, read the same way
    // `restore_and_append_chunk` / phase-2a capture read it.
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "run_python".to_owned(),
            LoweredClientTool {
                original_name: "run_python".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );

    filter.arm(&mut ctx);

    // Open the logical response so the incremental item events are delivered.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // One chunk carrying the lowered `function_call`'s whole lifecycle: added ->
    // arguments.done -> item.done for the same item. The lowered arguments are the
    // `{"input":...}` envelope `openai_client_tool_compat` produced on the way out.
    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1"},
            "sequence_number": 1
        }),
    );
    let args_done = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"input\":\"print(1)\"}",
            "sequence_number": 2
        }),
    );
    let item_done = make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "run_python", "call_id": "c1", "id": "fc_1",
                     "arguments": "{\"input\":\"print(1)\"}", "status": "completed"},
            "sequence_number": 3
        }),
    );
    let mut chunk = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&args_done);
        buf.extend_from_slice(&item_done);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    let emitted = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
    // The retyped added and restored done both carry the public custom_tool_call type.
    assert!(
        emitted.contains(r#""type":"custom_tool_call""#),
        "the restored item type must reach the client: {emitted}"
    );
    // The backend function-args frame is replaced by the synthesized input pair.
    assert!(
        emitted.contains("response.custom_tool_call_input.delta"),
        "the synthesized custom input delta must be emitted: {emitted}"
    );
    assert!(
        emitted.contains("response.custom_tool_call_input.done"),
        "the synthesized custom input done must be emitted: {emitted}"
    );
    assert!(
        emitted.contains(r#""input":"print(1)""#),
        "the unwrapped plain-string input must reach the client: {emitted}"
    );
    // Every client-visible id is the public `ctc_` form, never the private `fc_` id.
    assert!(
        emitted.contains(r#""id":"ctc_1""#),
        "the public custom item id must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""call_id":"c1""#),
        "the client's own call id must be preserved on the restored call: {emitted}"
    );
    assert!(
        !emitted.contains("fc_1"),
        "the private lowered item id must never leak to the client: {emitted}"
    );
    assert!(
        !emitted.contains("function_call_arguments"),
        "the backend function-args frames must be replaced, not forwarded: {emitted}"
    );
    assert!(
        !emitted.contains(r#""type":"function_call""#),
        "the private lowered item type must never surface to the client: {emitted}"
    );
}

// #1159 Task 5: end-to-end proof for the *obfuscated* `NamespaceCustom` path — the
// case #1159 exists to protect. A namespaced custom member is lowered to the private
// `agentic_ns__{ns}__{member}` wire name, so its restoration must recover the member
// name AND re-add its namespace while never leaking the private lowered name or its
// `fc_` id. Mirrors `custom_lowered_call_restored_to_custom_tool_call_lifecycle_through_commit`
// (the plain-Custom commit-path e2e) so the whole synthesis runs through the real
// commit path (`commit_chunk_events` phase-2a artifact capture ->
// `restore_and_append_chunk` -> `apply_client_tool_disposition`).
#[tokio::test]
async fn namespace_custom_lowered_call_restored_through_commit_never_leaks_private_name() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    // Arm a NamespaceCustom lowering on the shared `ResponsesState`, read the same
    // way `restore_and_append_chunk` / phase-2a capture read it. The wire name is the
    // obfuscated `agentic_ns__{ns}__{member}` form; the restore must recover the bare
    // member name and re-add the namespace.
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "agentic_ns__code__run".to_owned(),
            LoweredClientTool {
                original_name: "run".to_owned(),
                namespace: Some("code".to_owned()),
                restore: ClientToolRestore::NamespaceCustom,
            },
        );

    filter.arm(&mut ctx);

    // Open the logical response so the incremental item events are delivered.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // One chunk carrying the lowered `function_call`'s whole lifecycle: added ->
    // arguments.done -> item.done for the same item, keyed by the private `fc_ns1`
    // id. The phase-2a accumulate must store the completed `function_call` item
    // before the done event is planned, exactly as the plain-Custom e2e relies on.
    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__code__run", "call_id": "c_ns1", "id": "fc_ns1"},
            "sequence_number": 1
        }),
    );
    let args_done = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({
            "output_index": 0,
            "item_id": "fc_ns1",
            "arguments": "{\"input\":\"print(1)\"}",
            "sequence_number": 2
        }),
    );
    let item_done = make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__code__run", "call_id": "c_ns1", "id": "fc_ns1",
                     "arguments": "{\"input\":\"print(1)\"}", "status": "completed"},
            "sequence_number": 3
        }),
    );
    let mut chunk = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&args_done);
        buf.extend_from_slice(&item_done);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    let emitted = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
    // The restored item is a namespaced custom_tool_call carrying the bare member
    // name, its namespace, the unwrapped input, and the public `ctc_` id.
    assert!(
        emitted.contains(r#""type":"custom_tool_call""#),
        "the restored item type must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""name":"run""#),
        "the restored member name must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""namespace":"code""#),
        "the namespace must be re-added on the restored item: {emitted}"
    );
    assert!(
        emitted.contains(r#""input":"print(1)""#),
        "the unwrapped plain-string input must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""id":"ctc_ns1""#),
        "the public custom item id must reach the client: {emitted}"
    );
    // The load-bearing no-leak assertions: the obfuscated wire name, its private
    // `fc_` id, and the raw lowered `function_call` type must never surface.
    assert!(
        !emitted.contains("agentic_ns__"),
        "the private lowered namespace name must never leak to the client: {emitted}"
    );
    assert!(
        !emitted.contains("fc_ns1"),
        "the private lowered item id must never leak to the client: {emitted}"
    );
    assert!(
        !emitted.contains(r#""type":"function_call""#),
        "the private lowered item type must never surface to the client: {emitted}"
    );
}

// #1159 Task 6: end-to-end proof for the local `shell` path through the real commit
// path (`commit_chunk_events` phase-2a artifact capture -> `restore_and_append_chunk`
// -> `apply_client_tool_disposition`). A lowered `shell` `function_call` has its raw
// added/args-delta/args-done lifecycle SUPPRESSED and a schema-complete `shell_call`
// synthesized at args.done, followed by the restored `output_item.done`. Asserts the
// EMITTED SSE bytes carry the public `shell_call` shape (`environment.type=="local"`,
// public `sh_` id, parsed `commands`) and never leak the private `fc_` id, the raw
// `function_call` type, or a `function_call_arguments` frame.
#[tokio::test]
async fn shell_lowered_call_restored_to_shell_call_lifecycle_through_commit() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "shell".to_owned(),
            LoweredClientTool {
                original_name: "shell".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Shell,
            },
        );

    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    // The lowered arguments are the shell action envelope openai_client_tool_compat
    // produced on the way out: a `{"commands":[...]}` object.
    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1"},
            "sequence_number": 1
        }),
    );
    let args_delta = make_sse_chunk(
        "response.function_call_arguments.delta",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "delta": "{\"commands\":",
            "sequence_number": 2
        }),
    );
    let args_done = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"commands\":[\"ls\"]}",
            "sequence_number": 3
        }),
    );
    let item_done = make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "shell", "call_id": "c1", "id": "fc_1",
                     "arguments": "{\"commands\":[\"ls\"]}", "status": "completed"},
            "sequence_number": 4
        }),
    );
    let mut chunk = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&args_delta);
        buf.extend_from_slice(&args_done);
        buf.extend_from_slice(&item_done);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    let emitted = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
    assert!(
        emitted.contains(r#""type":"shell_call""#),
        "the restored shell_call type must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""environment":{"type":"local"}"#),
        "the restored shell_call must carry a local environment: {emitted}"
    );
    assert!(
        emitted.contains(r#""id":"sh_1""#),
        "the public shell item id must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""commands":["ls"]"#),
        "the parsed shell commands must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""call_id":"c1""#),
        "the client's own call id must be preserved on the restored call: {emitted}"
    );
    assert!(
        !emitted.contains("fc_1"),
        "the private lowered item id must never leak to the client: {emitted}"
    );
    assert!(
        !emitted.contains(r#""type":"function_call""#),
        "the private lowered item type must never surface to the client: {emitted}"
    );
    assert!(
        !emitted.contains("function_call_arguments"),
        "the backend function-args frames must be suppressed, not forwarded: {emitted}"
    );
}

// #1159 Task 6: end-to-end proof for the client-executed `tool_search` path through
// the real commit path. Mirrors the `shell` e2e: the raw lowered lifecycle is
// suppressed and a schema-complete `tool_search_call` (`execution=="client"`, public
// `tsc_` id, parsed `arguments`) is synthesized at args.done, followed by the restored
// `output_item.done`. The private `fc_` id, raw `function_call` type, and
// `function_call_arguments` frame must never surface.
#[tokio::test]
async fn tool_search_lowered_call_restored_to_tool_search_call_lifecycle_through_commit() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "tool_search".to_owned(),
            LoweredClientTool {
                original_name: "tool_search".to_owned(),
                namespace: None,
                restore: ClientToolRestore::ToolSearch,
            },
        );

    filter.arm(&mut ctx);

    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();

    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1"},
            "sequence_number": 1
        }),
    );
    let args_delta = make_sse_chunk(
        "response.function_call_arguments.delta",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "delta": "{\"query\":",
            "sequence_number": 2
        }),
    );
    let args_done = make_sse_chunk(
        "response.function_call_arguments.done",
        &json!({
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"query\":\"rust\"}",
            "sequence_number": 3
        }),
    );
    let item_done = make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "tool_search", "call_id": "c1", "id": "fc_1",
                     "arguments": "{\"query\":\"rust\"}", "status": "completed"},
            "sequence_number": 4
        }),
    );
    let mut chunk = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&args_delta);
        buf.extend_from_slice(&args_done);
        buf.extend_from_slice(&item_done);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk, false).unwrap();

    let emitted = String::from_utf8(chunk.unwrap().to_vec()).unwrap();
    assert!(
        emitted.contains(r#""type":"tool_search_call""#),
        "the restored tool_search_call type must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""execution":"client""#),
        "the restored tool_search_call must be client-executed: {emitted}"
    );
    assert!(
        emitted.contains(r#""id":"tsc_1""#),
        "the public tool_search item id must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""query":"rust""#),
        "the parsed tool_search arguments must reach the client: {emitted}"
    );
    assert!(
        emitted.contains(r#""call_id":"c1""#),
        "the client's own call id must be preserved on the restored call: {emitted}"
    );
    assert!(
        !emitted.contains("fc_1"),
        "the private lowered item id must never leak to the client: {emitted}"
    );
    assert!(
        !emitted.contains(r#""type":"function_call""#),
        "the private lowered item type must never surface to the client: {emitted}"
    );
    assert!(
        !emitted.contains("function_call_arguments"),
        "the backend function-args frames must be suppressed, not forwarded: {emitted}"
    );
}

// #1159 C1 (fail-closed security regression): a fatal mid-stream error must poison
// the whole logical stream so a co-batched lowered item whose `output_item.added`
// was rolled back by the failed chunk cannot later leak its raw private name on a
// subsequent `output_item.done`. Backends batch several SSE frames per network
// chunk, so this drives the real commit path through `on_response_body`:
//   * Chunk N: the lowered item's `output_item.added` FOLLOWED by a lossy snapshot (a `response.in_progress` whose
//     output holds a lowered custom `function_call` with no `call_id`). The plan pass fails closed AFTER tracking the
//     added item, so `state.client_tool_items` is rolled back and the chunk emits nothing.
//   * Chunk N+1: that same item's `output_item.done` carrying the private name.
// Without the sticky poison flag + defense-in-depth, chunk N+1 would find the item
// untracked and pass the RAW `function_call` (private `agentic_ns__...` name)
// straight to the client. Asserts no frame ever carries the private name and the
// stream terminates as an error, not a successful completion.
#[tokio::test]
async fn poisoned_stream_never_leaks_rolled_back_lowered_name_on_later_done() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    // Arm a Custom lowering keyed by a private lowered name, read the same way
    // `restore_and_append_chunk` reads it.
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .client_tool_lowering
        .insert(
            "agentic_ns__demo__apply_patch".to_owned(),
            LoweredClientTool {
                original_name: "apply_patch".to_owned(),
                namespace: None,
                restore: ClientToolRestore::Custom,
            },
        );

    filter.arm(&mut ctx);

    // Open the logical response.
    let mut created = Some(make_sse_chunk(
        "response.created",
        &json!({
            "response": {"id": "resp_1", "status": "in_progress", "output": []},
            "sequence_number": 0
        }),
    ));
    filter.on_response_body(&mut ctx, &mut created, false).unwrap();
    let created_out = created.map_or_else(String::new, |b| String::from_utf8(b.to_vec()).unwrap());

    // Chunk N: the lowered item's `output_item.added` (tracked) followed by a lossy
    // snapshot that fails the plan pass closed — rolling back the just-added item.
    let added = make_sse_chunk(
        "response.output_item.added",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__demo__apply_patch", "call_id": "c1", "id": "fc_1"},
            "sequence_number": 1
        }),
    );
    let lossy_in_progress = make_sse_chunk(
        "response.in_progress",
        &json!({
            "response": {
                "object": "response",
                "output": [
                    {"type": "function_call", "name": "agentic_ns__demo__apply_patch", "id": "fc_2"}
                ]
            },
            "sequence_number": 2
        }),
    );
    let mut chunk_n = Some({
        let mut buf = added.to_vec();
        buf.extend_from_slice(&lossy_in_progress);
        Bytes::from(buf)
    });
    filter.on_response_body(&mut ctx, &mut chunk_n, false).unwrap();
    let chunk_n_out = chunk_n.map_or_else(String::new, |b| String::from_utf8(b.to_vec()).unwrap());
    assert!(
        chunk_n_out.is_empty(),
        "the failed chunk must emit nothing (plan pass fails before any append): {chunk_n_out}"
    );

    // Chunk N+1: the rolled-back item's `output_item.done` carrying the private name.
    let mut chunk_n1 = Some(make_sse_chunk(
        "response.output_item.done",
        &json!({
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__demo__apply_patch", "call_id": "c1", "id": "fc_1",
                     "arguments": "{\"input\":\"x\"}", "status": "completed"},
            "sequence_number": 3
        }),
    ));
    filter.on_response_body(&mut ctx, &mut chunk_n1, false).unwrap();
    let chunk_n1_out = chunk_n1.map_or_else(String::new, |b| String::from_utf8(b.to_vec()).unwrap());
    assert!(
        chunk_n1_out.is_empty(),
        "a post-failure chunk must be dropped closed on the poisoned stream: {chunk_n1_out}"
    );

    // End of stream: the logical terminal must be an error, not a completion.
    let mut eos = None;
    filter.on_response_body(&mut ctx, &mut eos, true).unwrap();
    let eos_out = eos.map_or_else(String::new, |b| String::from_utf8(b.to_vec()).unwrap());

    let combined = format!("{created_out}{chunk_n_out}{chunk_n1_out}{eos_out}");
    assert!(
        !combined.contains("agentic_ns__demo__apply_patch"),
        "the private lowered name must never reach the client on a poisoned stream: {combined}"
    );
    assert!(
        !combined.contains("fc_1") && !combined.contains("fc_2"),
        "the private lowered `fc_` ids must never reach the client: {combined}"
    );
    assert!(
        !combined.contains(r#""type":"function_call""#),
        "the raw lowered function_call item must never surface to the client: {combined}"
    );
    assert!(
        !combined.contains("response.completed"),
        "a poisoned stream must not emit a successful completion terminal: {combined}"
    );
    assert_eq!(
        ctx.get_metadata("responses.stream_error_code"),
        Some("server_error"),
        "the poisoned stream must fail closed"
    );
    assert!(
        eos_out.contains("event: error"),
        "the logical stream must terminate with an error event: {eos_out}"
    );
}

// Test Utilities

fn make_armed_context_with_filter(
    filter: OpenaiStreamEventsFilter,
) -> (OpenaiStreamEventsFilter, praxis_filter::HttpFilterContext<'static>) {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    filter.arm(&mut ctx);
    (filter, ctx)
}

#[tokio::test]
async fn chat_budget_failure_replaces_deferred_terminal_with_error() {
    use crate::openai::responses::responses_to_chat_completions::ResponsesToChatCompletionsFilter;

    let (parser, mut ctx) = make_armed_context();
    let mut state = ResponsesState::from_request_body(json!({
        "model": "m", "input": "hi", "stream": true, "store": false
    }));
    state.apply_retained_payload_limit(65_536);
    ctx.extensions.insert(state);
    ctx.current_filter_id = Some(1);
    ctx.set_metadata("responses_to_chat_completions.armed", "true");
    ctx.set_metadata("responses.response_id", "resp_deferred_budget");
    ctx.set_metadata("responses_to_chat_completions.created_at", "1700000000");
    let translator = ResponsesToChatCompletionsFilter::from_config(&serde_yaml::Value::Null).unwrap();
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/event-stream"),
    );
    ctx.response_header = Some(response);
    assert!(matches!(
        translator.on_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    ctx.current_filter_id = Some(0);
    parser.on_response(&mut ctx).await.unwrap();
    ctx.response_header = None;

    let first_chunk = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({
            "id": "chatcmpl_deferred", "object": "chat.completion.chunk", "model": "m", "created": 1,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        })
    );
    let mut body = Some(Bytes::from(first_chunk));
    ctx.current_filter_id = Some(1);
    translator.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(
        std::str::from_utf8(body.as_ref().unwrap())
            .unwrap()
            .contains("event: response.completed")
    );
    ctx.current_filter_id = Some(0);
    parser.on_response_body(&mut ctx, &mut body, false).unwrap();
    let wire = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(wire.contains("event: response.created"));
    assert!(!wire.contains("event: response.completed"));
    assert!(
        ctx.get_filter_state::<StreamEventsState>()
            .unwrap()
            .deferred_terminal
            .is_some()
    );

    ctx.current_filter_id = Some(1);
    body = Some(Bytes::from(format!(": {}\n\n", "x".repeat(16_384))));
    assert!(
        translator.on_response_body(&mut ctx, &mut body, false).is_ok(),
        "the parser still owns the terminal, so a budget error must remain in band"
    );
    ctx.current_filter_id = Some(0);
    parser.on_response_body(&mut ctx, &mut body, true).unwrap();
    let final_wire = std::str::from_utf8(body.as_ref().unwrap()).unwrap();
    assert!(final_wire.contains("event: error"), "{final_wire}");
    assert!(!final_wire.contains("event: response.completed"), "{final_wire}");
}

#[test]
fn deferred_terminal_normalization_admits_id_copy_before_allocation() {
    let (_filter, mut ctx) = make_armed_context();
    let mut parser = ctx.remove_filter_state::<StreamEventsState>().unwrap();
    let mut state = ResponsesState {
        logical_stream_response_id: Some("r".repeat(100 * 1024)),
        response_object: json!({"id":"r", "output":[], "status":"completed"}),
        ..ResponsesState::default()
    };
    let mut terminal = super::DeferredTerminalEvent {
        event_type: "response.completed".to_owned(),
        payload: json!({"type":"response.completed", "response":null, "response_id":"r"}),
    };
    let baseline = state.retained_payload_bytes().unwrap();
    let cap = 225 * 1024;
    state.apply_retained_payload_limit(cap);
    ctx.extensions.insert(state);
    let mut output = Vec::new();
    let allocations = allocation_counter::measure(|| {
        drop(super::emit_deferred_terminal(
            &mut ctx,
            &mut terminal,
            &mut parser,
            &mut output,
        ));
    });
    assert!(
        allocations.bytes_max <= u64::try_from(cap - baseline).unwrap(),
        "normalization allocates an unreserved second logical ID before admission"
    );
}

#[test]
fn canonical_local_completion_admits_wire_capacity_near_limit() {
    let (_filter, mut ctx) = make_armed_context();
    let mut state = ResponsesState {
        accumulated_output: vec![
            json!({"type":"message", "id":"msg_large", "content":[{"type":"output_text", "text":"x".repeat(300 * 1024)}]}),
        ],
        local_completion_response_template: json!({"id":"resp_local", "status":"completed", "output":[]}),
        ..ResponsesState::default()
    };
    let cap = 700 * 1024;
    state.apply_retained_payload_limit(cap);
    ctx.extensions.insert(state);

    let result = encode_local_completion(&mut ctx).unwrap();
    assert!(
        result
            .as_ref()
            .windows(b"event: response.completed".len())
            .any(|window| window == b"event: response.completed"),
    );
    let retained = ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    let wire_capacity = result.try_into_mut().unwrap().capacity();
    assert!(retained + wire_capacity <= cap);
}

#[test]
fn local_completion_admits_retained_wire_capacity() {
    let (_filter, mut ctx) = make_armed_context();
    let mut state = ResponsesState {
        accumulated_output: vec![
            json!({"type":"message", "id":"msg_large", "content":[{"type":"output_text", "text":"x".repeat(300 * 1024)}]}),
        ],
        local_completion_response_template: json!({"id":"resp_local", "status":"completed", "output":[]}),
        ..ResponsesState::default()
    };
    let cap = 1024 * 1024;
    state.apply_retained_payload_limit(cap);
    let baseline = state.retained_payload_bytes().unwrap();
    assert!(baseline < cap);
    ctx.extensions.insert(state);

    let result = encode_local_completion(&mut ctx).unwrap();
    let retained = ctx
        .extensions
        .get::<ResponsesState>()
        .unwrap()
        .retained_payload_bytes()
        .unwrap();
    assert!(
        result
            .as_ref()
            .windows(b"event: response.completed".len())
            .any(|window| window == b"event: response.completed"),
        "this boundary must admit the completed terminal"
    );
    let wire_capacity = result.try_into_mut().unwrap().capacity();
    assert!(
        retained + wire_capacity <= cap,
        "retained response and terminal wire capacity exceed the request limit"
    );
}
