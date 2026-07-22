// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    unused_must_use,
    reason = "tests"
)]

use bytes::Bytes;
use praxis_filter::{BodyMode, FilterAction, FilterEntry, FilterPipeline, HttpFilter};
use serde_json::json;

use super::{
    CompletionState, OpenaiStreamEventsFilter, StreamEventsState, accumulate_response_object,
    has_identity_content_encoding, process_non_streaming_response,
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
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "preserve this input",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "stream": true,
    })));
    ctx.current_filter_id = Some(0);
    (filter, ctx)
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
    assert!(matches!(action, FilterAction::Continue));
    assert!(
        ctx.get_filter_state::<StreamEventsState>().is_some(),
        "filter should be armed"
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

#[test]
fn unarmed_filter_passes_through_body() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.current_filter_id = Some(0);

    let mut body = Some(Bytes::from("data: {}\n\n"));
    let action = filter.on_response_body(&mut ctx, &mut body, false).unwrap();
    assert!(matches!(action, FilterAction::Continue));
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
    assert_eq!(state.request_body["input"], "preserve this input");
    assert_eq!(state.tools[0]["type"], "file_search");
    assert_eq!(ctx.get_metadata("responses.status"), Some("completed"),);
}

#[tokio::test]
async fn non_streaming_json_uses_bounded_buffer_and_accumulates_state() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "false".to_owned());
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "keep me",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
    })));
    ctx.current_filter_id = Some(0);

    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json; charset=utf-8"),
    );
    ctx.response_header = Some(response);

    filter.on_request(&mut ctx).await.unwrap();
    assert!(ctx.request_headers_to_remove.contains(&http::header::ACCEPT_ENCODING));
    assert_eq!(
        ctx.request_headers_to_set
            .iter()
            .find(|(name, _value)| name == http::header::ACCEPT_ENCODING)
            .map(|(_name, value)| value),
        Some(&http::HeaderValue::from_static("identity"))
    );
    filter.on_response(&mut ctx).await.unwrap();
    assert_eq!(
        ctx.response_body_mode,
        BodyMode::StreamBuffer {
            max_bytes: Some(praxis_filter::body::MAX_JSON_BODY_BYTES),
        },
        "non-streaming JSON should be buffered with the shared JSON limit"
    );

    let response_body = json!({
        "id": "resp_json",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o",
        "created_at": 1_700_000_000,
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "file_search_call", "id": "fs_1", "status": "in_progress", "queries": ["query"]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object, response_body);
    assert_eq!(state.output_items().len(), 2);
    assert_eq!(state.usage["total_tokens"], 14);
    assert_eq!(state.request_body["input"], "keep me");
    assert_eq!(state.tools[0]["vector_store_ids"][0], "vs_1");
    assert_eq!(ctx.get_metadata("responses.status"), Some("completed"));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(body.as_ref().unwrap()).unwrap(),
        response_body,
        "ordinary first-pass accumulation must not rewrite the response body"
    );
}

#[tokio::test]
async fn non_streaming_continuation_prepends_prior_output_exactly_once() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");
    let prior = json!({
        "type": "file_search_call",
        "id": "fs_1",
        "status": "completed"
    });
    let mut state = ResponsesState::from_request_body(json!({"model":"gpt-4o","input":"search"}));
    state.output_items_mut().push(prior.clone());
    state.continuation_output_count = 1;
    ctx.extensions.insert(state);
    ctx.current_filter_id = Some(0);

    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("999"));
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("stale"));
    ctx.response_header = Some(response);

    filter.on_request(&mut ctx).await.unwrap();
    filter.on_response(&mut ctx).await.unwrap();
    let backend_response = json!({
        "id":"resp-final",
        "status":"completed",
        "output":[{
            "type":"message",
            "id":"msg-final",
            "role":"assistant",
            "content":[{"type":"output_text","text":"Final answer","annotations":[]}]
        }]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&backend_response).unwrap()));

    filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let merged: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let output = merged["output"].as_array().unwrap();
    assert_eq!(output.len(), 2, "prior and final output should both be visible");
    assert_eq!(output[0], prior, "prior continuation output should remain first");
    assert_eq!(output[1], backend_response["output"][0]);
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object, merged);
    assert_eq!(state.output_items(), output.as_slice());
    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::CONTENT_LENGTH)
            .is_none(),
        "continuity rewriting invalidates the original body length"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::ETAG)
            .is_none(),
        "continuity rewriting invalidates representation validators"
    );

    filter.on_response(&mut ctx).await.unwrap();
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    let repeated: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(
        repeated["output"].as_array().unwrap().len(),
        2,
        "repeated accumulation must not duplicate prior output"
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

    assert!(!accumulate_response_object(&mut ctx, first, None));
    assert!(accumulate_response_object(&mut ctx, second, None));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.usage["input_tokens"], 17);
    assert_eq!(state.usage["output_tokens"], 6);
    assert_eq!(state.usage["total_tokens"], 23);
    assert_eq!(state.usage["input_tokens_details"]["cached_tokens"], 4);
    assert_eq!(state.response_object["usage"], state.usage);

    let final_without_usage = json!({"status":"completed","output":[]});
    assert!(accumulate_response_object(&mut ctx, final_without_usage, None));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["usage"], state.usage);
    assert_eq!(state.usage["total_tokens"], 23);
}

#[test]
fn continuation_restores_original_public_tool_policy() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    let mut state = ResponsesState::from_request_body(json!({
        "model":"gpt-4o",
        "input":"search",
        "tool_choice":{"type":"file_search"},
        "max_tool_calls":3
    }));
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!("auto"));
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(
        br#"{"id":"resp","output":[],"tool_choice":"auto","max_tool_calls":1}"#,
    ));

    process_non_streaming_response(&mut ctx, &mut body).unwrap();

    let public: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(public["tool_choice"], json!({"type":"file_search"}));
    assert_eq!(public["max_tool_calls"], 3);
    assert_eq!(state.response_object["tool_choice"], public["tool_choice"]);
    assert_eq!(state.response_object["max_tool_calls"], 3);
}

#[test]
fn continuation_restores_effective_auto_when_client_omitted_tool_choice() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    let mut state = ResponsesState::from_request_body(json!({
        "model":"gpt-4o",
        "input":"search",
        "max_tool_calls":1
    }));
    state.iteration = 1;
    state.continuation_tool_choice = Some(json!("none"));
    ctx.extensions.insert(state);
    let mut body = Some(Bytes::from_static(br#"{"id":"resp","output":[],"tool_choice":"none"}"#));

    process_non_streaming_response(&mut ctx, &mut body).unwrap();

    let public: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(public["tool_choice"], "auto");
    assert_eq!(public["max_tool_calls"], 1);
    assert!(!state.tool_choice_present);
    assert_eq!(state.tool_choice, "auto");
}

#[test]
fn continuation_restores_canonical_response_tool_choice_from_request_shorthand() {
    let cases = [
        (json!(null), json!("auto")),
        (
            json!({
                "type":"allowed_tools",
                "tools":[{"type":"function","name":"lookup"}]
            }),
            json!({
                "type":"allowed_tools",
                "mode":"auto",
                "tools":[{"type":"function","name":"lookup"}]
            }),
        ),
    ];
    for (request_choice, expected) in cases {
        let req = make_request(http::Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(Box::leak(Box::new(req)));
        let mut state = ResponsesState::from_request_body(json!({
            "model":"gpt-4o",
            "input":"search",
            "tool_choice":request_choice
        }));
        state.iteration = 1;
        state.continuation_tool_choice = Some(json!("none"));
        ctx.extensions.insert(state);
        let mut body = Some(Bytes::from_static(br#"{"id":"resp","output":[],"tool_choice":"none"}"#));

        process_non_streaming_response(&mut ctx, &mut body).unwrap();

        let public: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(public["tool_choice"], expected);
        assert_eq!(
            ctx.extensions.get::<ResponsesState>().unwrap().response_object["tool_choice"],
            expected
        );
    }
}

#[test]
fn response_accumulation_materializes_deferred_first_pass_history() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    let initial_input = json!([
        {"type": "message", "role": "user", "content": "search the files"},
        {"type": "message", "role": "assistant", "content": "prior context"}
    ]);
    ctx.extensions
        .insert(ResponsesState::from_file_search_request_body(json!({
            "model": "gpt-4.1",
            "input": initial_input,
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]
        })));

    let response = json!({
        "id": "resp_1",
        "status": "completed",
        "output": [{"type": "file_search_call", "id": "fs_1", "status": "in_progress"}]
    });
    assert!(!accumulate_response_object(&mut ctx, response, None));

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(!state.has_deferred_history());
    assert_eq!(state.input, initial_input.as_array().unwrap().clone());
    assert_eq!(state.messages, initial_input.as_array().unwrap().clone());
    assert_eq!(state.persisted_messages, state.messages);
    assert_eq!(state.request_body["model"], "gpt-4.1");
    assert!(state.request_body.get("input").is_none());
}

#[tokio::test]
async fn non_streaming_json_rewrites_file_citations_consistently() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "false".to_owned());
    let mut state = ResponsesState::from_request_body(json!({"model":"gpt-4o","input":"search"}));
    state
        .citation_files
        .insert("file-a".to_owned(), "report.pdf".to_owned());
    ctx.extensions.insert(state);
    ctx.current_filter_id = Some(0);

    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_LENGTH, http::HeaderValue::from_static("999"));
    response
        .headers
        .insert(http::header::ETAG, http::HeaderValue::from_static("stale"));
    response
        .headers
        .insert("content-digest", http::HeaderValue::from_static("sha-256=:stale:"));
    ctx.response_header = Some(response);

    filter.on_response(&mut ctx).await.unwrap();
    assert!(
        ctx.response_headers_modified,
        "rewritten response headers should be marked modified"
    );
    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::ETAG)
            .is_none()
    );
    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get("content-digest")
            .is_none()
    );

    let response_body = json!({
        "id":"resp-citation",
        "status":"completed",
        "output":[{
            "type":"message",
            "id":"msg-1",
            "role":"assistant",
            "content":[{
                "type":"output_text",
                "text":"See source <|file-a|>.",
                "annotations":[{"type":"url_citation","start_index":0,"end_index":3,"url":"https://example.com","title":"existing"}]
            }]
        }]
    });
    let mut body = Some(Bytes::from(serde_json::to_vec(&response_body).unwrap()));
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let rewritten: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    let text = &rewritten["output"][0]["content"][0];
    assert_eq!(text["text"], "See source.");
    assert_eq!(text["annotations"].as_array().unwrap().len(), 2);
    assert_eq!(text["annotations"][1]["type"], "file_citation");
    assert_eq!(text["annotations"][1]["file_id"], "file-a");
    assert_eq!(text["annotations"][1]["filename"], "report.pdf");
    assert_eq!(text["annotations"][1]["index"], 10);

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object, rewritten);
    assert_eq!(state.output_items(), rewritten["output"].as_array().unwrap());
    assert!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::CONTENT_LENGTH)
            .is_none(),
        "the protocol layer must frame the rewritten buffered response"
    );
}

#[tokio::test]
async fn pipeline_open_failure_mode_cannot_bypass_citation_rewrite_failure() {
    let mut entries: Vec<FilterEntry> = serde_yaml::from_str(
        "
        - filter: openai_stream_events
          failure_mode: open
        ",
    )
    .unwrap();
    let registry = crate::test_utils::make_ai_registry();
    let pipeline = FilterPipeline::build(&mut entries, &registry).unwrap();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");
    let mut state = ResponsesState::from_request_body(json!({
        "model":"gpt-4o",
        "input":"search",
        "tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]
    }));
    state
        .citation_files
        .insert("file-a".to_owned(), "report.pdf".to_owned());
    ctx.extensions.insert(state);

    assert!(matches!(
        pipeline.execute_http_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    ctx.response_header = Some(response);
    assert!(matches!(
        pipeline.execute_http_response(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let backend = json!({
        "id":"resp-citation-overflow",
        "status":"completed",
        "output":[{
            "type":"message",
            "role":"assistant",
            "content":[{
                "type":"output_text",
                "text":"<|file-a|>".repeat(4_097),
                "annotations":[]
            }]
        }]
    });
    let original = Bytes::from(serde_json::to_vec(&backend).unwrap());
    let mut body = Some(original.clone());

    let action = pipeline.execute_http_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(&action, FilterAction::Reject(_)),
        "rewrite failure must reject even when the pipeline failure mode is open"
    );
    let FilterAction::Reject(rejection) = action else {
        return;
    };
    assert_eq!(rejection.status, 502);
    assert_eq!(
        body,
        Some(original),
        "the backend body must never be released after rejection"
    );
    let error: serde_json::Value = serde_json::from_slice(rejection.body.as_ref().unwrap()).unwrap();
    assert_eq!(error["error"]["code"], "server_error");
}

#[tokio::test]
async fn encoded_file_search_response_is_rejected_without_header_corruption() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model":"gpt-4o",
        "input":"search",
        "tools":[{"type":"file_search","vector_store_ids":["vs_1"]}]
    })));
    ctx.current_filter_id = Some(0);

    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    response
        .headers
        .insert(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
    ctx.response_header = Some(response);

    assert!(matches!(
        filter.on_response(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert_eq!(
        ctx.response_header
            .as_ref()
            .unwrap()
            .headers
            .get(http::header::CONTENT_ENCODING),
        Some(&http::HeaderValue::from_static("gzip"))
    );
}

#[test]
fn content_encoding_requires_only_identity_across_all_values() {
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.append(
        http::header::CONTENT_ENCODING,
        http::HeaderValue::from_static("identity"),
    );
    response
        .headers
        .append(http::header::CONTENT_ENCODING, http::HeaderValue::from_static("gzip"));
    ctx.response_header = Some(response);

    assert!(
        !has_identity_content_encoding(&ctx),
        "a later non-identity header value must not bypass validation"
    );

    let headers = &mut ctx.response_header.as_mut().unwrap().headers;
    headers.remove(http::header::CONTENT_ENCODING);
    headers.insert(
        http::header::CONTENT_ENCODING,
        http::HeaderValue::from_static("identity, identity"),
    );
    assert!(
        has_identity_content_encoding(&ctx),
        "a list containing only identity tokens must be accepted"
    );

    let headers = &mut ctx.response_header.as_mut().unwrap().headers;
    headers.insert(
        http::header::CONTENT_ENCODING,
        http::HeaderValue::from_static("identity, gzip"),
    );
    assert!(
        !has_identity_content_encoding(&ctx),
        "a non-identity token in one header value must be rejected"
    );
}

#[tokio::test]
async fn non_streaming_json_parse_failure_preserves_initialized_state() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses".to_owned());
    ctx.set_metadata("openai_responses_format.stream", "false".to_owned());
    ctx.extensions.insert(ResponsesState::from_request_body(json!({
        "model": "gpt-4o",
        "input": "keep me",
        "tools": [{"type":"file_search","vector_store_ids":["vs_1"]}]
    })));
    ctx.current_filter_id = Some(0);

    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response.headers.insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("application/json"),
    );
    ctx.response_header = Some(response);
    filter.on_response(&mut ctx).await.unwrap();

    let mut body = Some(Bytes::from_static(b"{not json"));
    filter.on_response_body(&mut ctx, &mut body, true).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.request_body["input"], "keep me");
    assert!(state.response_object.is_null());
    assert_eq!(ctx.get_metadata("responses.response_parse_error"), Some("true"));
}

#[tokio::test]
async fn continuation_rejects_empty_or_malformed_json_instead_of_losing_prior_output() {
    for mut body in [None, Some(Bytes::new()), Some(Bytes::from_static(b"{not json"))] {
        let filter = make_filter();
        let req = make_request(http::Method::POST, "/v1/responses");
        let mut ctx = make_filter_context(Box::leak(Box::new(req)));
        ctx.set_metadata("openai_responses_format.format", "openai_responses");
        ctx.set_metadata("openai_responses_format.stream", "false");
        let prior = json!({"type":"file_search_call","id":"fs-1","status":"completed"});
        let mut state = ResponsesState::from_request_body(json!({"model":"gpt-4o","input":"search"}));
        state.response_object = json!({"id":"resp-prior","output":[prior]});
        state.continuation_output_count = 1;
        ctx.extensions.insert(state);
        ctx.current_filter_id = Some(0);
        let response = Box::leak(Box::new(crate::test_utils::make_response()));
        response.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        ctx.response_header = Some(response);

        assert!(matches!(
            filter.on_response(&mut ctx).await.unwrap(),
            FilterAction::Continue
        ));
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut body, true).unwrap(),
            FilterAction::Reject(_)
        ));
        assert_eq!(ctx.get_metadata("responses.response_parse_error"), Some("true"));
    }
}

#[tokio::test]
async fn continuation_rejects_success_response_with_non_json_content_type() {
    let filter = make_filter();
    let req = make_request(http::Method::POST, "/v1/responses");
    let mut ctx = make_filter_context(Box::leak(Box::new(req)));
    ctx.set_metadata("openai_responses_format.format", "openai_responses");
    ctx.set_metadata("openai_responses_format.stream", "false");
    let mut state = ResponsesState::from_request_body(json!({"model":"gpt-4o","input":"search"}));
    state.response_object = json!({
        "id":"resp-prior",
        "output":[{"type":"file_search_call","id":"fs-1","status":"completed"}]
    });
    state.continuation_output_count = 1;
    ctx.extensions.insert(state);
    ctx.current_filter_id = Some(0);
    let response = Box::leak(Box::new(crate::test_utils::make_response()));
    response
        .headers
        .insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static("text/plain"));
    ctx.response_header = Some(response);

    let action = filter.on_response(&mut ctx).await.unwrap();

    assert!(matches!(action, FilterAction::Reject(_)));
}

#[tokio::test]
async fn complete_response_without_initialized_state_preserves_standalone_behavior() {
    let (filter, mut ctx) = make_armed_context();
    filter.on_request(&mut ctx).await.unwrap();
    ctx.extensions.remove::<ResponsesState>();

    let completed =
        json!({"id": "resp_1", "status": "completed", "model": "m", "created_at": 0, "output": [], "usage": {}});
    let mut body = Some(make_sse_chunk("response.completed", &completed));
    filter.on_response_body(&mut ctx, &mut body, false).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.response_object["id"], "resp_1");
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
        max_tool_call_argument_bytes: 1024 * 1024,
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
        max_tool_call_argument_bytes: 1024 * 1024,
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
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.request_body["input"], "preserve this input");
    assert!(
        state.response_object.is_null(),
        "error event should not write a response"
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
    assert!(ctx.get_filter_state::<StreamEventsState>().is_some());

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
