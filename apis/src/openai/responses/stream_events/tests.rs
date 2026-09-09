// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    unused_must_use,
    reason = "tests"
)]

use bytes::Bytes;
use praxis_filter::{FilterAction, HttpFilter, SubRequestResponseMode};
use serde_json::json;

use super::{
    CompletionState, OpenaiStreamEventsFilter, StreamEventsState, accumulate_response_object, encode_local_completion,
};
use crate::{
    openai::{responses::state::ResponsesState, sse::SseFrameParser},
    test_utils::{make_filter_context, make_request},
};

fn make_filter() -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
    OpenaiStreamEventsFilter::from_config(&yaml).unwrap()
}

fn make_armed_context() -> (Box<dyn HttpFilter>, praxis_filter::HttpFilterContext<'static>) {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    (filter, ctx)
}

fn make_logical_filter() -> Box<dyn HttpFilter> {
    let yaml: serde_yaml::Value = serde_yaml::from_str("logical_stream: true").unwrap();
    OpenaiStreamEventsFilter::from_config(&yaml).unwrap()
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
        Some(state.accumulated_output.as_slice())
    );
    assert_eq!(payload["response"]["usage"], state.usage);
    assert_eq!(state.logical_stream_sequence, 5);
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
fn logical_stream_requires_response_write_access() {
    let filter = make_logical_filter();
    assert_eq!(
        filter.response_body_access(),
        praxis_filter::BodyAccess::ReadWrite,
        "logical lifecycle normalization rewrites emitted SSE frames"
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

#[tokio::test]
async fn arms_for_streaming_responses_request() {
    let (filter, mut ctx) = make_armed_context();
    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "metadata-selected streaming request should continue"
    );
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "filter should be armed"
    );
}

#[tokio::test]
async fn arms_for_typed_streaming_selection_without_classifier_metadata() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);

    let action = filter.on_request(&mut ctx).await.unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "typed terminal streaming selection should continue"
    );
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "typed terminal streaming selection should arm the SSE parser"
    );
}

#[tokio::test]
async fn arm_publishes_logical_stream_marker_when_enabled() {
    let (_default, mut ctx) = make_armed_context();
    let filter = make_logical_filter();

    filter.on_request(&mut ctx).await.unwrap();

    // openai_agentic_loop reads and consumes this marker to fail closed on the
    // unsafe automatic-terminal-streaming + agentic_loop without-logical_stream
    // combo.
    assert_eq!(
        ctx.get_metadata("responses.logical_stream"),
        Some("true"),
        "logical_stream must publish the per-round marker openai_agentic_loop consumes"
    );
}

#[tokio::test]
async fn arm_omits_logical_stream_marker_when_disabled() {
    let (filter, mut ctx) = make_armed_context();

    filter.on_request(&mut ctx).await.unwrap();

    assert!(
        ctx.get_metadata("responses.logical_stream").is_none(),
        "a non-logical stream_events filter must not publish the logical_stream marker"
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    filter.on_request(&mut ctx).await.unwrap();

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
    ctx.filter_results
        .entry("openai_mcp_dispatch")
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
    ctx.filter_results.remove("openai_mcp_dispatch");
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));

    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
) -> (Box<dyn HttpFilter>, praxis_filter::HttpFilterContext<'static>) {
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
async fn logical_stream_synthesizes_progress_for_model_declared_item_without_repeating_added() {
    // #276 (Finding 1): the model announces a `web_search_call` placeholder with
    // `output_item.added` (status `in_progress`) but never streams the
    // tool-specific progress events; the proxy completes the search locally under
    // the same id. The proxy must synthesize the full progress lifecycle
    // (in_progress -> searching -> completed) and a fresh `output_item.done`,
    // without repeating the `output_item.added` the model already streamed.
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let first = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    filter.on_request(&mut ctx).await.unwrap();

    let second = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

    let delta = resumed_text_delta(filter.as_ref(), &mut ctx);
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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hi",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
async fn terminal_event_authoritatively_populates_completed_function_calls() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();
    ctx.extensions.insert(ResponsesState::default());
    ctx.extensions
        .get_mut::<ResponsesState>()
        .unwrap()
        .tool_calls
        .push(json!({
            "type": "function_call",
            "id": "fc_stale",
            "call_id": "call_stale",
            "name": "stale",
            "arguments": "{}",
            "status": "completed"
        }));

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
        state.tool_calls.len(),
        1,
        "the authoritative terminal response must replace incremental tool calls"
    );
    assert_eq!(
        state.tool_calls[0]["call_id"], "call_final",
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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    assert_eq!(state.tool_calls.len(), 1);
    assert_eq!(state.tool_calls[0]["id"], "fc_item_1");
    assert_eq!(state.tool_calls[0]["call_id"], "call_1");
    assert_eq!(state.tool_calls[0]["name"], "get_weather");
    assert_eq!(state.tool_calls[0]["arguments"], "{\"city\":\"NYC\"}");
    assert_eq!(state.tool_calls[0]["status"], "completed");
    assert_eq!(state.output_items()[0]["arguments"], "{\"city\":\"NYC\"}");
}

#[tokio::test]
async fn missing_state_does_not_panic() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = make_logical_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_subrequest_response_mode(SubRequestResponseMode::Streaming);
    ctx.current_filter_id = Some(0);
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "test-model",
        "input": "hello",
        "stream": true
    })));
    filter.on_request(&mut ctx).await.unwrap();

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
fn body_passes_through_unchanged() {
    let (filter, mut ctx) = make_armed_context();
    ctx.insert_filter_state(StreamEventsState {
        frame_parser: SseFrameParser::new(10_485_760),
        event_count: 0,
        max_events: 100_000,
        timeout: std::time::Duration::from_secs(300),
        started_at: None,
        completed_at: None,
        completion_state: CompletionState::Open,
        tool_call_args: std::collections::HashMap::new(),
        rejected_tool_call_args: std::collections::HashSet::new(),
        max_tool_call_argument_bytes: 1024 * 1024,
        logical_stream: false,
        iteration: 0,
        output_index_offset: 0,
        deferred_terminal: None,
        deferred_done: false,
        local_items_flushed: false,
    });

    let original = Bytes::from("event: response.created\ndata: {\"type\":\"response.created\",\"id\":\"r1\"}\n\n");
    let mut body = Some(original.clone());
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    assert_eq!(
        body.as_ref().unwrap().as_ref(),
        original.as_ref(),
        "body should pass through unchanged in ReadOnly mode"
    );
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
        logical_stream: false,
        iteration: 0,
        output_index_offset: 0,
        deferred_terminal: None,
        deferred_done: false,
        local_items_flushed: false,
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

#[tokio::test]
async fn output_item_done_replaces_by_index() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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

#[tokio::test]
async fn upsert_tool_call_dedup() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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

    assert_eq!(ctx.extensions.get::<ResponsesState>().unwrap().tool_calls.len(), 1);

    let done2 = json!({"item_id": "fc_dup", "output_index": 0, "arguments": "{\"q\":\"v2\"}"});
    let mut b3 = Some(make_sse_chunk("response.function_call_arguments.done", &done2));
    filter.on_response_body(&mut ctx, &mut b3, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_calls.len(), 1, "should replace, not append duplicate");
    assert_eq!(state.tool_calls[0]["arguments"], "{\"q\":\"v2\"}");
}

#[tokio::test]
async fn function_call_done_without_prior_deltas() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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
    assert_eq!(state.tool_calls.len(), 1);
    assert_eq!(
        state.tool_calls[0]["arguments"], "{\"tz\":\"UTC\"}",
        "should use payload arguments when no deltas were accumulated"
    );
}

#[tokio::test]
async fn done_payload_wins_over_accumulated_deltas() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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
        state.tool_calls[0]["arguments"], "{\"from\":\"done_payload\"}",
        "done-event arguments should take precedence over accumulated deltas"
    );
}

#[tokio::test]
async fn unknown_event_type_ignored() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

    let payload = json!({"some_field": "some_value"});
    let mut body = Some(make_sse_chunk("response.future_event_type", &payload));
    let result = filter.on_response_body(&mut ctx, &mut body, false);

    assert!(result.is_ok(), "unknown event type should not error");
    assert!(body.is_some(), "body should pass through unchanged");
}

#[tokio::test]
async fn error_event_does_not_mutate_state() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = make_logical_filter();
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
    filter.on_request(&mut ctx).await.unwrap();
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
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = OpenaiStreamEventsFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = OpenaiStreamEventsFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = OpenaiStreamEventsFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);
    filter.on_request(&mut ctx).await.unwrap();

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
    let filter = OpenaiStreamEventsFilter::from_config(&yaml).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "true".to_owned());
    ctx.current_filter_id = Some(0);

    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();
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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
}

#[tokio::test]
async fn on_request_strips_accept_encoding_when_arming() {
    let (filter, mut ctx) = make_armed_context();

    filter.on_request(&mut ctx).await.unwrap();

    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "test setup should arm the SSE parser"
    );
    assert!(
        ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING),
        "arming logical parsing must strip Accept-Encoding so the backend returns plaintext SSE"
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

    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
    filter.on_request(&mut ctx).await.unwrap();

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
