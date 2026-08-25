// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

use std::{
    collections::HashMap,
    io::{Read as _, Write as _},
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use praxis_filter::{FilterAction, HttpFilter};
use serde_json::{Value, json};

use super::{
    client::{
        ContentChunk, ContentChunkType, FileSearchClient, FileSearchClientConfig, MAX_CONCURRENT_SEARCHES,
        MAX_QUERY_BYTES, MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES, SearchResult, VectorStoreSearchRequest,
        VectorStoreSearchResponse, request_error,
    },
    config::{FileSearchFilterConfig, ValidatedConfig, build_config},
    *,
};
// -----------------------------------------------------------------------------
// Configuration and transport validation
// -----------------------------------------------------------------------------

#[test]
fn minimal_config_uses_safe_defaults() {
    let raw: FileSearchFilterConfig = serde_yaml::from_str(
        r#"
        vector_store_url: "https://8.8.8.8"
        "#,
    )
    .unwrap();
    let config = build_config(&raw).unwrap();

    assert_eq!(config.api_client.api_base_url(), "https://8.8.8.8");
    assert_eq!(config.max_response_bytes, 10_485_760);
    assert_eq!(config.max_total_response_bytes, 67_108_864);
    assert_eq!(config.max_state_bytes, 52_428_800);
    assert_eq!(config.failure_mode, FailureMode::Closed);
}

#[test]
fn config_rejects_unknown_and_removed_circuit_breaker_fields() {
    for yaml in [
        "vector_store_url: https://8.8.8.8\nunknown: true\n",
        "vector_store_url: https://8.8.8.8\ncircuit_breaker: {}\n",
        "vector_store_url: https://8.8.8.8\nsearch_template: '{query}'\n",
        "vector_store_url: https://8.8.8.8\nannotation_template: '{content}'\n",
        "vector_store_url: https://8.8.8.8\ncontext_template: '{results}'\n",
    ] {
        assert!(
            serde_yaml::from_str::<FileSearchFilterConfig>(yaml).is_err(),
            "unknown fields must be rejected: {yaml}"
        );
    }
}

#[test]
fn config_rejects_ambiguous_or_invalid_urls() {
    for url in [
        "",
        "ftp://8.8.8.8",
        "https://user:pass@8.8.8.8",
        "https://8.8.8.8?token=x",
        "https://8.8.8.8#fragment",
        "http://[2606:4700:4700::1111]garbage",
        "http://[::1",
    ] {
        let result = parse_config(&format!("vector_store_url: '{url}'\n"));
        assert!(result.is_err(), "URL must be rejected: {url}");
    }
}

#[test]
fn config_rejects_dns_and_sensitive_ip_targets_by_default() {
    for url in [
        "http://localhost:8001",
        "http://vector-store.internal:8001",
        "http://127.0.0.1:8001",
        "http://10.0.0.1:8001",
        "http://169.254.169.254:8001",
        "http://100.64.0.1:8001",
        "http://0.7.8.9:8001",
        "http://[::1]:8001",
        "http://[::ffff:10.0.0.1]:8001",
    ] {
        assert!(parse_config(&format!("vector_store_url: '{url}'\n")).is_err(), "{url}");
    }
}

#[test]
fn config_private_override_is_explicit() {
    let config = parse_config("vector_store_url: 'http://localhost:8001'\nallow_private_url: true\n").unwrap();
    assert_eq!(config.api_client.api_base_url(), "http://localhost:8001");
}

#[test]
fn config_validates_timeouts_and_budgets() {
    for yaml in [
        "vector_store_url: https://8.8.8.8\ntimeout_ms: 0\n",
        "vector_store_url: https://8.8.8.8\nmax_response_bytes: 0\n",
        "vector_store_url: https://8.8.8.8\nmax_total_response_bytes: 0\n",
        "vector_store_url: https://8.8.8.8\nmax_response_bytes: 100\nmax_total_response_bytes: 99\n",
        "vector_store_url: https://8.8.8.8\nmax_state_bytes: 0\n",
        "vector_store_url: https://8.8.8.8\nmax_state_bytes: 268435457\n",
    ] {
        assert!(parse_config(yaml).is_err(), "invalid bound must fail: {yaml}");
    }
}

#[test]
fn search_request_translates_responses_hybrid_ranking_to_vector_store() {
    let filters = json!({"type": "eq", "key": "team", "value": "infra"});
    let ranking = json!({
        "ranker": "auto",
        "score_threshold": 0.2,
        "hybrid_search": {"embedding_weight": 0.4, "text_weight": 0.2}
    });
    let request = VectorStoreSearchRequest::new(Some(&filters), Some(7), "rendered query", Some(&ranking)).unwrap();

    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({
            "filters": filters,
            "max_num_results": 7,
            "query": "rendered query",
            "ranking_options": {
                "ranker": "weighted",
                "alpha": 2.0_f64 / 3.0_f64,
                "score_threshold": 0.2,
            },
            "rewrite_query": false,
            "search_mode": "hybrid",
        })
    );
}

#[test]
fn hybrid_ranking_translation_rejects_untranslatable_weights() {
    for ranking in [
        json!({"hybrid_search": []}),
        json!({"hybrid_search": {"embedding_weight": 0.5}}),
        json!({"hybrid_search": {"embedding_weight": "high", "text_weight": 0.5}}),
        json!({"hybrid_search": {"embedding_weight": 0.0, "text_weight": 0.0}}),
        json!({"hybrid_search": {"embedding_weight": -1.0, "text_weight": -1.0}}),
        json!({"hybrid_search": {"embedding_weight": -1.0, "text_weight": 2.0}}),
        json!({"hybrid_search": {"embedding_weight": 1e308, "text_weight": 1e308}}),
    ] {
        assert!(
            VectorStoreSearchRequest::new(None, None, "query", Some(&ranking)).is_err(),
            "hybrid weights must be structurally translatable: {ranking}"
        );
    }
}

#[test]
fn search_response_requires_page_data_and_result_content() {
    let response: VectorStoreSearchResponse = serde_json::from_value(json!({
        "data": [{
            "file_id": "file-a",
            "filename": "a.txt",
            "score": 0.5,
            "content": []
        }]
    }))
    .unwrap();
    assert!(response.data[0].content.is_empty());
    assert!(response.data[0].attributes.is_none());

    assert!(
        serde_json::from_value::<VectorStoreSearchResponse>(json!({})).is_err(),
        "page data is required"
    );
    assert!(
        serde_json::from_value::<VectorStoreSearchResponse>(json!({
            "data": [{"file_id":"file-a","filename":"a.txt","score":0.5}]
        }))
        .is_err(),
        "result content is required"
    );
}

#[test]
fn file_search_errors_escape_store_id_controls() {
    let rendered = request_error("store\n\u{1b}[31m", "failed").to_string();

    assert!(!rendered.contains('\n'));
    assert!(!rendered.contains('\u{1b}'));
    assert!(rendered.contains("store\\n\\u{1b}[31m"));
}

// -----------------------------------------------------------------------------
// Planning and state replay
// -----------------------------------------------------------------------------

#[test]
fn plan_accounts_for_every_call_after_global_cap() {
    let first_queries: Vec<String> = (0..MAX_SEARCH_SPECS).map(|index| format!("q{index}")).collect();
    let state = state_with(
        &["vs-a"],
        vec![
            json!({"type":"file_search_call","id":"fs-a","status":"searching","queries":first_queries}),
            json!({"type":"file_search_call","id":"fs-b","status":"in_progress","queries":["later"]}),
            json!({"type":"file_search_call","id":"fs-c","status":"searching","queries":[]}),
        ],
    );

    let plan = build_search_plan(&state);
    assert_eq!(plan.spec_coordinates.len(), MAX_SEARCH_SPECS);
    assert_eq!(plan.calls.len(), 3);
    assert_eq!(plan.calls[0].scheduled_specs, MAX_SEARCH_SPECS);
    assert_eq!(plan.calls[1].expected_specs, 1);
    assert_eq!(plan.calls[1].scheduled_specs, 0);
    assert_eq!(plan.calls[2].expected_specs, 0);
    assert_eq!(plan.calls[2].scheduled_specs, 0);
}

#[test]
fn plan_bounds_owned_inputs_but_accounts_for_every_store() {
    let oversized_store = "é".repeat(MAX_VECTOR_STORE_ID_BYTES);
    let mut store_ids = vec![oversized_store];
    store_ids.extend((1..=MAX_SEARCH_SPECS).map(|index| format!("vs-{index}")));
    let store_refs: Vec<&str> = store_ids.iter().map(String::as_str).collect();
    let oversized_query = "é".repeat(MAX_QUERY_BYTES);
    let state = state_with(
        &store_refs,
        vec![json!({
            "type":"file_search_call",
            "id":"fs-a",
            "status":"searching",
            "queries":[oversized_query],
        })],
    );

    let plan = build_search_plan(&state);

    assert_eq!(plan.vector_store_ids.len(), MAX_SEARCH_SPECS);
    assert!(plan.vector_store_ids[0].len() > MAX_VECTOR_STORE_ID_BYTES);
    assert!(plan.vector_store_ids[0].len() <= MAX_VECTOR_STORE_ID_BYTES + 4);
    assert!(plan.calls[0].queries[0].len() > MAX_QUERY_BYTES);
    assert!(plan.calls[0].queries[0].len() <= MAX_QUERY_BYTES + 4);
    assert_eq!(plan.calls[0].expected_specs, MAX_SEARCH_SPECS + 1);
    assert_eq!(plan.calls[0].scheduled_specs, MAX_SEARCH_SPECS);
}

#[test]
fn synthetic_bridge_call_ids_are_bounded_and_response_specific() {
    let item = json!({"id":"x".repeat(100_000)});
    let first = model_context_messages(&item, usize::MAX, stable_call_hash(&["resp-a"]), "query", "output");
    let second = model_context_messages(&item, usize::MAX, stable_call_hash(&["resp-b"]), "query", "output");
    let first_id = first[0]["call_id"].as_str().unwrap();
    let second_id = second[0]["call_id"].as_str().unwrap();

    assert!(!first_id.is_empty());
    assert!(first_id.len() <= 64);
    assert_eq!(first[1]["call_id"], first_id);
    assert_ne!(first_id, second_id);
}

#[test]
fn bridge_budget_retains_ascii_context_near_the_execution_cap() {
    let result = SearchResult {
        attributes: None,
        content: (0..model_context::MAX_FORMATTED_CHUNKS)
            .map(|_| ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "a".repeat(512),
            })
            .collect(),
        file_id: "file-a".to_owned(),
        filename: "a.txt".to_owned(),
        score: 0.9,
    };

    let budgeted = format_test_bridge(
        &[result],
        "query",
        "{content}",
        "BEGIN{results}END",
        MAX_TOTAL_MODEL_CONTEXT_BYTES,
    );

    let messages = budgeted.model_messages.expect("a bounded prefix should fit");
    assert!(budgeted.truncated);
    assert!(budgeted.serialized_bytes <= MAX_TOTAL_MODEL_CONTEXT_BYTES);
    assert!(
        budgeted.serialized_bytes > MAX_TOTAL_MODEL_CONTEXT_BYTES - 2_048,
        "the largest complete chunk prefix should use the available budget"
    );
    assert!(messages[1]["output"].as_str().unwrap().starts_with("BEGIN"));
}

#[test]
fn bridge_budget_charges_escape_heavy_query_and_context_json() {
    let result = SearchResult {
        attributes: None,
        content: (0..400)
            .map(|_| ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "\0".repeat(1_000),
            })
            .collect(),
        file_id: "file-a".to_owned(),
        filename: "a.txt".to_owned(),
        score: 0.9,
    };
    let query = "\0\"\\\n".repeat(1_000);

    let budgeted = format_test_bridge(
        &[result],
        &query,
        "{content}",
        "{query}|{results}",
        MAX_TOTAL_MODEL_CONTEXT_BYTES,
    );

    let messages = budgeted.model_messages.as_ref().expect("an escaped prefix should fit");
    let exact = bounded_json_size(messages, MAX_TOTAL_MODEL_CONTEXT_BYTES)
        .unwrap()
        .expect("accepted bridge must serialize within the cap");
    assert_eq!(budgeted.serialized_bytes, exact);
    assert!(budgeted.truncated);
    assert!(budgeted.serialized_bytes <= MAX_TOTAL_MODEL_CONTEXT_BYTES);
}

#[test]
fn bridge_budget_reserves_nonmonotonic_outer_wrapper_before_chunks() {
    let result = SearchResult {
        attributes: None,
        content: (0..32)
            .map(|_| ContentChunk {
                _chunk_type: ContentChunkType::Text,
                text: "chunk".repeat(200),
            })
            .collect(),
        file_id: "file-a".to_owned(),
        filename: "a.txt".to_owned(),
        score: 0.9,
    };
    let prefix = "P".repeat(32_000);
    let suffix = "S".repeat(32_000);
    let context_template = format!("{prefix}{{results}}{suffix}");

    let budgeted = format_test_bridge(&[result], "query", "{content}", &context_template, 80_000);

    let output = budgeted
        .model_messages
        .as_ref()
        .and_then(|messages| messages[1]["output"].as_str())
        .expect("wrapper reservation should leave a bounded chunk prefix");
    assert!(output.starts_with(&prefix));
    assert!(output.ends_with(&suffix));
    assert!(output.contains("chunk"));
    assert!(budgeted.serialized_bytes <= 80_000);
}

#[test]
fn model_facing_query_join_is_bounded() {
    let queries = vec!["a".repeat(40_000), "b".repeat(40_000)];
    let (joined, truncated) = join_queries_bounded(&queries);

    assert!(truncated);
    assert_eq!(joined.len(), 40_000);
    assert!(joined.len() <= MAX_QUERY_BYTES);
}

#[test]
fn response_output_is_bounded_after_search_formatting() {
    let state = ResponsesState {
        response_object: json!({"id":"resp","output":[{"type":"message","content":"x".repeat(200)}]}),
        ..Default::default()
    };

    assert!(!response_fits(&state, 128));
    assert!(response_fits(&state, 1_024));
}

#[test]
fn continuation_header_replay_excludes_stale_request_metadata() {
    for name in [
        http::header::AUTHORIZATION,
        http::header::ACCEPT,
        http::header::CONTENT_TYPE,
        http::header::HeaderName::from_static("x-tenant-id"),
    ] {
        assert!(should_replay_original_header(&name), "{name} should be replayed");
    }

    for name in [
        http::header::HOST,
        http::header::CONTENT_LENGTH,
        http::header::CONTENT_ENCODING,
        http::header::ACCEPT_ENCODING,
        http::header::HeaderName::from_static("idempotency-key"),
        http::header::HeaderName::from_static("x-praxis-internal-test"),
    ] {
        assert!(!should_replay_original_header(&name), "{name} should not be replayed");
    }
}

#[test]
fn continuation_header_replay_excludes_connection_nominated_headers() {
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::CONNECTION,
        http::HeaderValue::from_static("keep-alive, X-Request-Hop"),
    );
    let nominated = http::header::HeaderName::from_static("x-request-hop");

    assert!(connection_nominates_header(&headers, &nominated));
    assert!(!connection_nominates_header(&headers, &http::header::AUTHORIZATION));
}

#[test]
fn output_budget_is_shared_across_inference_rounds() {
    let state = ResponsesState {
        file_search_output_items: vec![json!({"type":"reasoning","content":"a".repeat(40)})],
        response_object: json!({"output":[{"type":"file_search_call","content":"b".repeat(40)}]}),
        ..Default::default()
    };
    let incoming = json!({"output":[{"type":"message","content":"c".repeat(40)}]});

    assert!(combined_output_fits(&state, &incoming, 512));
    assert!(!combined_output_fits(&state, &incoming, 128));
}

#[tokio::test]
async fn no_state_or_pending_calls_is_a_noop() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(None);
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(
        ctx.request_headers_to_set
            .iter()
            .find(|(name, _)| *name == http::header::ACCEPT_ENCODING)
            .map(|(_, value)| value),
        Some(&http::HeaderValue::from_static("identity"))
    );

    ctx.extensions.insert(state_with(&["vs-a"], vec![]));
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn request_body_initializes_file_search_state_inside_router() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(None);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"search","tools":[{"type":"file_search","vector_store_ids":["vs-a"]}]}"#,
    ));

    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx
        .extensions
        .get::<ResponsesState>()
        .expect("state should be initialized");
    assert_eq!(state.messages[0]["content"], "search");
    assert_eq!(state.tools[0]["type"], "file_search");
}

#[tokio::test]
async fn non_file_search_body_does_not_consume_continuation_budget() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "max_state_bytes: 128\n");
    let mut ctx = make_context(None);
    let mut body = Some(Bytes::from(json!({"input":"x".repeat(256)}).to_string()));

    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Continue
    ));
    assert!(ctx.extensions.get::<ResponsesState>().is_none());
}

#[tokio::test]
async fn initial_state_is_rejected_before_oversized_json_duplication() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "max_state_bytes: 128\n");
    let mut ctx = make_context(None);
    let mut body = Some(Bytes::from(
        json!({
            "model":"gpt-4.1",
            "input":"x".repeat(256),
            "tools":[{"type":"file_search","vector_store_ids":["vs-a"]}]
        })
        .to_string(),
    ));

    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(ctx.extensions.get::<ResponsesState>().is_none());
}

#[tokio::test]
async fn streaming_file_search_is_rejected_before_callout() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));
    ctx.set_metadata("openai_responses_format.stream", "true");

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn streaming_without_file_search_is_rejected_before_buffering() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(None);
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"hello","stream":true}"#,
    ));

    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn streaming_rehydrated_citations_are_rejected_without_a_pending_call() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut state = state_with(&["vs-a"], vec![]);
    state.citation_files.insert("file-a".to_owned(), "a.txt".to_owned());
    let mut ctx = make_context(Some(state));
    ctx.set_metadata("openai_responses_format.stream", "true");

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn streaming_rehydrated_state_is_rejected_from_the_request_body() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut state = state_with(&["vs-a"], vec![]);
    state.history_rehydrated = true;
    let mut ctx = make_context(Some(state));
    let mut body = Some(Bytes::from_static(
        br#"{"model":"gpt-4.1","input":"search","stream":true,"tools":[{"type":"file_search","vector_store_ids":["vs-a"]}]}"#,
    ));

    assert!(matches!(
        filter.on_request_body(&mut ctx, &mut body, true).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn first_pass_streaming_file_search_is_rejected() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(Some(state_with(&["vs-a"], vec![])));
    ctx.set_metadata("openai_responses_format.stream", "true");
    ctx.set_metadata("openai_tool_parse.has_file_search", "true");

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn successful_callout_preserves_full_output_order_and_is_idempotent() {
    let server = MockServer::json(200, &one_result("file-a", "report.pdf", 0.95, "Revenue grew."));
    let filter = make_filter(server.port, "");
    let output = vec![
        json!({"type":"reasoning","id":"rs-1","summary":[],"encrypted_content":"opaque-reasoning"}),
        json!({"type":"file_search_call","id":"fs-1","status":"searching","queries":["revenue"]}),
        json!({"type":"mcp_list_tools","id":"mcp-list","tools":[]}),
        json!({"type":"message","id":"msg-1","role":"assistant","content":[{"type":"output_text","text":"I will check."}]}),
    ];
    let mut state = state_with(&["vs-a"], output.clone());
    state.include.push("file_search_call.results".to_owned());
    state
        .messages
        .push(json!({"type":"message","id":"input-1","role":"user","content":"search"}));
    state.response_object = json!({"id":"resp-1","output":output});
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[1]["status"], "completed");
    assert_eq!(state.messages.iter().filter(|item| item["id"] == "mcp-list").count(), 0);
    assert_eq!(item_ids(&state.messages), vec!["input-1", "rs-1", "msg-1"]);
    assert_eq!(
        state.messages.iter().find(|item| item["id"] == "rs-1").unwrap()["encrypted_content"],
        "opaque-reasoning"
    );
    assert_eq!(
        state
            .messages
            .iter()
            .skip(1)
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["reasoning", "function_call", "function_call_output", "message"]
    );
    assert_eq!(
        state.messages.iter().find(|item| item["id"] == "msg-1").unwrap()["content"][0]["text"],
        "I will check."
    );
    assert_eq!(
        state.response_object["output"],
        Value::Array(state.output_items().to_vec())
    );
    assert_eq!(
        state.citation_files.get("file-a").map(String::as_str),
        Some("report.pdf")
    );
    assert_eq!(state.output_items()[1]["results"][0]["text"], "Revenue grew.");
    let model_output = state
        .messages
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .and_then(|item| item["output"].as_str())
        .unwrap();
    assert!(model_output.starts_with("file_search found 1 chunks for \"revenue\":"));
    assert!(model_output.contains("<|file-a|>"));
    assert!(model_output.ends_with("Revenue grew.\n"));
    let message_len = state.messages.len();

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages.len(), message_len);
}

#[test]
fn mixed_client_and_file_search_calls_are_rejected_before_search() {
    let state = state_with(&["vs-a"], vec![]);
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id":"resp-1",
            "output":[
                {"type":"file_search_call","id":"fs-1","status":"searching","queries":["revenue"]},
                {"type":"function_call","id":"fc-1","call_id":"call-1","name":"lookup_tax","arguments":"{}","status":"completed"}
            ]
        })
        .to_string(),
    ));

    assert!(matches!(
        FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap(),
        FilterAction::Reject(_)
    ));
    let state = ctx.extensions.remove::<ResponsesState>().unwrap();
    assert!(state.output_items().is_empty());
}

#[tokio::test]
async fn forced_tool_choice_resets_after_search_execution() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut state = one_pending_state(&["vs-a"]);
    state.request_body = json!({
        "input":"search",
        "tool_choice":{"type":"file_search"},
        "tools":[{"type":"file_search","vector_store_ids":["vs-a"]}]
    });
    state.tool_choice = json!({"type":"file_search"});
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.tool_choice, "auto");
    assert!(state.request_body.get("tool_choice").is_none());
}

#[tokio::test]
async fn local_calls_keep_public_ids() {
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.8, "A"));
    let filter = make_filter(server.port, "");
    let long_id = format!("fs-{}", "x".repeat(512));
    let output = vec![
        json!({"type":"file_search_call","status":"searching","queries":["missing id"]}),
        json!({"type":"file_search_call","id":long_id,"status":"searching","queries":["long id"]}),
    ];
    let mut state = state_with(&["vs-a"], output);
    state.response_object["id"] = json!("resp-id-normalization");
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let generated_id = state.output_items()[0]["id"].as_str().unwrap();
    assert!(generated_id.starts_with("fs_"));
    assert_eq!(state.output_items()[1]["id"], long_id);
}

#[tokio::test]
async fn mixed_valid_and_empty_calls_are_both_terminalized() {
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.8, "A"));
    let filter = make_filter(server.port, "");
    let state = state_with(
        &["vs-a"],
        vec![
            json!({"type":"file_search_call","id":"same","status":"searching","queries":["valid"]}),
            json!({"type":"file_search_call","id":"same","status":"in_progress","queries":[]}),
        ],
    );
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[0]["status"], "completed");
    assert_eq!(state.output_items()[1]["status"], "incomplete");
    assert!(state.output_items()[0].get("results").is_none());
    assert!(state.output_items()[1].get("results").is_none());
    assert_eq!(
        state
            .messages
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .count(),
        2
    );
    let call_ids: HashSet<&str> = state
        .messages
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .filter_map(|item| item["call_id"].as_str())
        .collect();
    assert_eq!(call_ids.len(), 2, "synthetic bridge IDs must be unique");
}

#[tokio::test]
async fn model_context_budget_is_shared_across_calls() {
    let chunks: Vec<Value> = (0..128)
        .map(|_| json!({"type":"text","text":"a".repeat(12_000)}))
        .collect();
    let server = MockServer::json(
        200,
        &json!({
            "data": [{
                "file_id":"file-a",
                "filename":"a.txt",
                "score":0.9,
                "content":chunks,
                "attributes":null,
            }]
        }),
    );
    let filter = make_filter(server.port, "");
    let state = state_with(
        &["vs-a"],
        vec![
            json!({"type":"file_search_call","id":"fs-a","status":"searching","queries":["a"]}),
            json!({"type":"file_search_call","id":"fs-b","status":"searching","queries":["b"]}),
        ],
    );
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[0]["status"], "completed");
    assert_eq!(state.output_items()[1]["status"], "incomplete");
    let retained_context_bytes: usize = state
        .messages
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .filter_map(|item| item["output"].as_str())
        .map(str::len)
        .sum();
    assert!(retained_context_bytes <= MAX_TOTAL_MODEL_CONTEXT_BYTES);
}

#[tokio::test]
async fn pending_call_cap_terminalizes_excess_and_preserves_duplicate_siblings() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut output = vec![
        json!({"type":"reasoning","id":"duplicate","summary":[]}),
        json!({"type":"reasoning","id":"duplicate","summary":[]}),
    ];
    output.extend((0..MAX_PENDING_CALLS.saturating_add(2)).map(|_| {
        json!({
            "type":"file_search_call",
            "status":"searching",
            "queries":["query"],
        })
    }));
    let mut ctx = make_context(Some(state_with(&[], output)));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state
            .output_items()
            .iter()
            .filter(|item| item["type"] == "file_search_call")
            .all(|item| item["status"] == "incomplete")
    );
    assert!(
        state
            .output_items()
            .iter()
            .filter(|item| item["type"] == "file_search_call")
            .all(|item| item["id"].as_str().is_some_and(|id| !id.is_empty()))
    );
    assert_eq!(
        state
            .messages
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .count(),
        MAX_PENDING_CALLS
    );
    assert_eq!(
        state
            .messages
            .iter()
            .filter(|item| item["type"] == "file_search_call")
            .count(),
        0,
        "local hosted calls are not valid model-facing replay items"
    );
    assert_eq!(
        state
            .output_items()
            .iter()
            .filter(|item| item["id"] == "duplicate")
            .count(),
        2
    );
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn max_tool_calls_counts_completed_calls_before_pending_execution() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let prior = json!({"type":"apply_patch_call","id":"ap-prior","status":"completed"});
    let pending = json!({"type":"file_search_call","id":"fs-new","status":"searching","queries":["q"]});
    let mut state = state_with(&["vs-a"], vec![prior, pending]);
    state.max_tool_calls = Some(1);
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[1]["status"], "incomplete");
    assert!(server.requests().is_empty());
    assert_eq!(
        state
            .messages
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .count(),
        0
    );
}

#[tokio::test]
async fn max_tool_calls_counts_completed_calls_in_the_current_output() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let completed = json!({"type":"web_search_call","id":"ws-current","status":"completed"});
    let pending = json!({"type":"file_search_call","id":"fs-new","status":"searching","queries":["q"]});
    let mut state = state_with(&["vs-a"], vec![completed, pending]);
    state.max_tool_calls = Some(1);
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[1]["status"], "incomplete");
    assert!(server.requests().is_empty());
}

#[test]
fn exhausted_tool_budget_finishes_without_another_inference() {
    let prior = json!({"type":"file_search_call","id":"fs-prior","status":"completed"});
    let mut state = state_with(&["vs-a"], vec![]);
    state.file_search_output_items.push(prior);
    state.max_tool_calls = Some(1);
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id":"resp-budget",
            "output":[{
                "type":"file_search_call",
                "id":"fs-exhausted",
                "status":"searching",
                "queries":["again"]
            }]
        })
        .to_string(),
    ));

    assert!(matches!(
        FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap(),
        FilterAction::Continue
    ));
    let encoded: Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
    assert_eq!(encoded["output"][0]["status"], "completed");
    assert_eq!(encoded["output"][1]["status"], "incomplete");
    assert_eq!(
        ctx.filter_results["openai_file_search_callout"].get("pending"),
        Some("false")
    );
}

#[test]
fn malformed_success_after_search_is_rejected() {
    let mut state = state_with(&["vs-a"], vec![]);
    state.iteration = 1;
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from_static(b"not-json"));

    assert!(matches!(
        FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap(),
        FilterAction::Reject(_)
    ));
}

#[test]
fn malformed_output_shape_after_search_is_rejected() {
    for response in [json!({}), json!({"output":"invalid"})] {
        let mut state = state_with(&["vs-a"], vec![]);
        state.iteration = 1;
        let mut ctx = make_context(Some(state));
        let mut response_header = crate::test_utils::make_response();
        ctx.response_header = Some(&mut response_header);
        let mut body = Some(Bytes::from(response.to_string()));

        assert!(matches!(
            FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap(),
            FilterAction::Reject(_)
        ));
    }
}

#[test]
fn final_response_rewrite_clears_representation_headers() {
    let mut state = state_with(&["vs-a"], vec![]);
    state.iteration = 1;
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    for name in [
        http::header::CONTENT_ENCODING,
        http::header::CONTENT_LENGTH,
        http::header::CONTENT_RANGE,
        http::header::ETAG,
        http::header::LAST_MODIFIED,
    ] {
        response_header
            .headers
            .insert(name, http::HeaderValue::from_static("stale"));
    }
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(json!({"id":"resp-final","output":[]}).to_string()));

    assert!(matches!(
        FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap(),
        FilterAction::Continue
    ));
    let response = ctx.response_header.as_deref().unwrap();
    assert!(response.headers.is_empty());
    assert!(ctx.response_headers_modified);
}

#[tokio::test]
async fn mcp_calls_do_not_consume_the_builtin_tool_budget() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mcp = json!({"type":"mcp_call","id":"mcp-prior","status":"completed"});
    let pending = json!({"type":"file_search_call","id":"fs-new","status":"searching","queries":["q"]});
    let mut state = state_with(&["vs-a"], vec![mcp, pending]);
    state.max_tool_calls = Some(1);
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[1]["status"], "completed");
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn no_store_ids_terminalizes_calls_and_replays_siblings() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let output = vec![
        json!({"type":"reasoning","id":"rs-1","summary":[],"encrypted_content":"opaque-reasoning"}),
        json!({"type":"file_search_call","id":"fs-1","status":"searching","queries":["q"]}),
    ];
    let mut ctx = make_context(Some(state_with(&[], output)));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[1]["status"], "incomplete");
    assert!(state.output_items()[1].get("results").is_none());
    assert_eq!(
        state
            .messages
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["reasoning", "function_call", "function_call_output"]
    );
    assert_eq!(state.messages[0]["encrypted_content"], "opaque-reasoning");
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn zero_match_search_completes_with_bounded_context() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    assert_eq!(state.output_items()[0]["status"], "completed");
    let output = state
        .messages
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .and_then(|item| item["output"].as_str());
    assert_eq!(output, Some("file_search found 0 chunks for \"query\":\n"));
}

#[tokio::test]
async fn query_cap_marks_the_call_incomplete() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let queries: Vec<String> = (0..MAX_QUERIES_PER_CALL.saturating_add(1))
        .map(|index| format!("query-{index}"))
        .collect();
    let state = state_with(
        &["vs-a"],
        vec![json!({
            "type":"file_search_call",
            "id":"fs-1",
            "status":"searching",
            "queries":queries,
        })],
    );
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[0]["status"], "incomplete");
    assert_eq!(server.requests().len(), MAX_QUERIES_PER_CALL);
}

#[tokio::test]
async fn ranking_filters_rewrite_policy_and_safe_path_are_sent_to_vector_store() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut state = state_with(
        &["vs_../../etc/passwd"],
        vec![json!({"type":"file_search_call","id":"fs-1","status":"searching","queries":["raw"]})],
    );
    state.tools[0]["max_num_results"] = json!(7);
    state.tools[0]["filters"] = json!({"type":"eq","key":"team","value":"infra"});
    state.tools[0]["ranking_options"] = json!({
        "ranker":"default-2024-11-15",
        "score_threshold":0.25,
        "hybrid_search":{"embedding_weight":0.7,"text_weight":0.3}
    });
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    let request_line = requests[0].lines().next().unwrap();
    assert!(!request_line.contains("/../"), "{request_line}");
    assert!(
        request_line.contains("vs_%2E%2E%2F%2E%2E%2Fetc%2Fpasswd"),
        "{request_line}"
    );
    let body = request_json(&requests[0]);
    assert_eq!(body["query"], "raw");
    assert_eq!(body["rewrite_query"], false);
    assert_eq!(body["max_num_results"], 7);
    assert_eq!(body["filters"]["value"], "infra");
    assert_eq!(body["search_mode"], "hybrid");
    assert_eq!(body["ranking_options"]["ranker"], "weighted");
    assert_eq!(body["ranking_options"]["alpha"], 0.7);
    assert_eq!(body["ranking_options"]["score_threshold"], 0.25);
    assert!(body["ranking_options"].get("hybrid_search").is_none());
    assert!(body["ranking_options"].get("weights").is_none());
}

#[tokio::test]
async fn open_and_closed_failure_modes_are_distinct() {
    for (status, body) in [
        (401, json!({"error":"unauthorized"}).to_string()),
        (403, "not-json".to_owned()),
        (404, String::new()),
        (429, "plain text".to_owned()),
        (500, json!({"error":"failed"}).to_string()),
    ] {
        let closed_body = body.clone();
        let closed_server = MockServer::start_with(move |_path| MockResponse {
            body: closed_body.clone(),
            body_delay: Duration::ZERO,
            status,
        });
        let closed = make_filter(closed_server.port, "callout_failure_mode: closed\n");
        let mut closed_ctx = make_context(Some(one_pending_state(&["vs-a"])));
        assert!(matches!(
            closed.on_request(&mut closed_ctx).await.unwrap(),
            FilterAction::Reject(_)
        ));
        assert_eq!(
            closed_ctx.extensions.get::<ResponsesState>().unwrap().output_items()[0]["status"],
            "searching",
            "status {status}"
        );

        let open_server = MockServer::start_with(move |_path| MockResponse {
            body: body.clone(),
            body_delay: Duration::ZERO,
            status,
        });
        let open = make_filter(open_server.port, "callout_failure_mode: open\n");
        let mut open_ctx = make_context(Some(one_pending_state(&["vs-a"])));
        assert!(matches!(
            open.on_request(&mut open_ctx).await.unwrap(),
            FilterAction::Continue
        ));
        let state = open_ctx.extensions.get::<ResponsesState>().unwrap();
        assert_eq!(state.output_items()[0]["status"], "incomplete", "status {status}");
        assert!(state.output_items()[0].get("results").is_none());
        assert!(
            state.messages.iter().any(|item| item["type"] == "function_call_output"),
            "status {status}"
        );
    }
}

#[tokio::test]
async fn aggregate_budget_stops_later_searches_and_marks_call_incomplete() {
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.9, "small"));
    let filter = make_filter(
        server.port,
        "callout_failure_mode: open\nmax_response_bytes: 512\nmax_total_response_bytes: 512\n",
    );
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b"])));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        server.requests().len(),
        1,
        "second request should not consume an unreserved body"
    );
    assert_eq!(state.output_items()[0]["status"], "incomplete");
    assert!(state.output_items()[0].get("results").is_none());
    assert_eq!(state.citation_files.get("file-a").map(String::as_str), Some("a.txt"));
}


#[tokio::test]
async fn malformed_success_bodies_are_charged_to_the_aggregate_budget() {
    let server = MockServer::start_with(|_path| MockResponse {
        body: "x".to_owned(),
        body_delay: Duration::ZERO,
        status: 200,
    });
    let filter = make_filter(
        server.port,
        "callout_failure_mode: open\nmax_response_bytes: 1\nmax_total_response_bytes: 4\n",
    );
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b", "vs-c", "vs-d", "vs-e"])));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(
        server.requests().len(),
        4,
        "malformed HTTP successes must consume the shared body budget"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[0]["status"], "incomplete");
}

#[tokio::test]
async fn core_limit_rejects_oversized_response_before_full_collection() {
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.9, &"x".repeat(4_096)));
    let filter = make_filter(server.port, "max_response_bytes: 128\nmax_total_response_bytes: 128\n");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));
    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
}

#[tokio::test]
async fn whole_call_timeout_covers_slow_response_body() {
    let server = MockServer::slow_body(&json!({"data": []}), Duration::from_millis(500));
    let filter = make_filter(server.port, "timeout_ms: 50\n");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));
    let started = Instant::now();

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(started.elapsed() < Duration::from_millis(400));
}

#[tokio::test]
async fn one_execution_deadline_covers_later_concurrency_chunks() {
    let server = MockServer::start_with(|path| MockResponse {
        body: json!({"data": []}).to_string(),
        body_delay: if path.contains("vs-8") {
            Duration::from_secs(8)
        } else {
            Duration::from_secs(1)
        },
        status: 200,
    });
    let filter = make_filter(
        server.port,
        "timeout_ms: 4000\nmax_response_bytes: 1024\nmax_total_response_bytes: 9216\n",
    );
    let store_ids: Vec<String> = (0..=MAX_CONCURRENT_SEARCHES)
        .map(|index| format!("vs-{index}"))
        .collect();
    let store_refs: Vec<&str> = store_ids.iter().map(String::as_str).collect();
    let mut ctx = make_context(Some(one_pending_state(&store_refs)));
    let started = Instant::now();

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "later chunks must use the original execution deadline"
    );
    assert_eq!(
        server.requests().len(),
        MAX_CONCURRENT_SEARCHES + 1,
        "the later chunk should start but not receive a fresh timeout"
    );
}

#[tokio::test]
async fn fail_closed_stops_scheduling_after_the_current_chunk() {
    let server = MockServer::json(500, &json!({"error": "failed"}));
    let filter = make_filter(server.port, "callout_failure_mode: closed\n");
    let store_ids: Vec<String> = (0..=MAX_CONCURRENT_SEARCHES)
        .map(|index| format!("vs-{index}"))
        .collect();
    let store_refs: Vec<&str> = store_ids.iter().map(String::as_str).collect();
    let mut ctx = make_context(Some(one_pending_state(&store_refs)));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));
    let requests = server.requests();
    assert!(!requests.is_empty());
    assert!(requests.len() <= MAX_CONCURRENT_SEARCHES);
    assert!(
        requests.iter().all(|request| !request.contains("/vs-8/search")),
        "the first failed chunk must prevent a later chunk from being scheduled"
    );
}

#[tokio::test]
async fn searches_multiple_stores_concurrently() {
    let server = MockServer::slow_body(&json!({"data": []}), Duration::from_millis(100));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b"])));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    assert_eq!(server.requests().len(), 2);
    assert!(
        server.max_active() >= 2,
        "multiple vector-store requests must overlap in flight"
    );
}

#[tokio::test]
async fn aggregate_results_are_score_sorted_and_limited_to_top_k() {
    let server = MockServer::routes([
        (
            "vs-a",
            200,
            search_results(&[("file-a", 0.4), ("file-b", 0.9)]),
            Duration::ZERO,
        ),
        (
            "vs-b",
            200,
            search_results(&[("file-c", 0.8), ("file-d", 0.95)]),
            Duration::ZERO,
        ),
    ]);
    let filter = make_filter(server.port, "callout_failure_mode: open\n");
    let mut state = one_pending_state(&["vs-a", "vs-b"]);
    state.tools[0]["max_num_results"] = json!(3);
    state.include.push("file_search_call.results".to_owned());
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let results = state.output_items()[0]["results"].as_array().unwrap();
    let scores: Vec<f64> = results.iter().filter_map(|result| result["score"].as_f64()).collect();
    let file_ids: Vec<&str> = results.iter().filter_map(|result| result["file_id"].as_str()).collect();

    assert_eq!(scores, vec![0.95, 0.9, 0.8]);
    assert_eq!(file_ids, vec!["file-d", "file-b", "file-c"]);
}

#[tokio::test]
async fn fail_open_retains_successful_results_from_a_partial_fan_out() {
    let server = MockServer::routes([
        (
            "vs-a",
            200,
            one_result("file-a", "file-a.txt", 0.9, "result"),
            Duration::ZERO,
        ),
        ("vs-b", 500, json!({"error": "failed"}), Duration::ZERO),
    ]);
    let filter = make_filter(server.port, "callout_failure_mode: open\n");
    let mut state = one_pending_state(&["vs-a", "vs-b"]);
    state.include.push("file_search_call.results".to_owned());
    let mut ctx = make_context(Some(state));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(server.requests().len(), 2);
    assert_eq!(state.output_items()[0]["status"], "incomplete");
    assert_eq!(state.output_items()[0]["results"].as_array().unwrap().len(), 1);
    assert_eq!(state.output_items()[0]["results"][0]["file_id"], "file-a");
}

#[tokio::test]
async fn outbound_query_store_id_and_request_body_are_bounded() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "callout_failure_mode: open\n");

    let oversized_store = "s".repeat(MAX_VECTOR_STORE_ID_BYTES + 1);
    let mut store_ctx = make_context(Some(one_pending_state(&[&oversized_store])));
    assert!(matches!(
        filter.on_request(&mut store_ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut query_state = one_pending_state(&["vs-query"]);
    query_state.output_items_mut()[0]["queries"] = json!(["q".repeat(MAX_QUERY_BYTES + 1)]);
    let mut query_ctx = make_context(Some(query_state));
    assert!(matches!(
        filter.on_request(&mut query_ctx).await.unwrap(),
        FilterAction::Continue
    ));

    let mut request_state = one_pending_state(&["vs-request"]);
    request_state.tools[0]["filters"] = json!({"type":"eq","key":"blob","value":"x".repeat(MAX_SEARCH_REQUEST_BYTES)});
    let mut request_ctx = make_context(Some(request_state));
    assert!(matches!(
        filter.on_request(&mut request_ctx).await.unwrap(),
        FilterAction::Continue
    ));

    assert!(
        server.requests().is_empty(),
        "invalid outbound inputs must fail before opening a connection"
    );
}

#[tokio::test]
async fn malformed_execution_fields_fail_without_silent_normalization() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "callout_failure_mode: closed\n");

    let mut invalid_stores = one_pending_state(&["vs-a"]);
    invalid_stores.tools[0]["vector_store_ids"] = json!(["vs-a", 7]);
    let mut invalid_stores_ctx = make_context(Some(invalid_stores));
    assert!(matches!(
        filter.on_request(&mut invalid_stores_ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));

    let mut invalid_max = one_pending_state(&["vs-a"]);
    invalid_max.tools[0]["max_num_results"] = json!("10");
    let mut invalid_max_ctx = make_context(Some(invalid_max));
    assert!(matches!(
        filter.on_request(&mut invalid_max_ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));

    let mut invalid_queries = one_pending_state(&["vs-a"]);
    invalid_queries.output_items_mut()[0]["queries"] = json!(["valid", 7]);
    let mut invalid_queries_ctx = make_context(Some(invalid_queries));
    assert!(matches!(
        filter.on_request(&mut invalid_queries_ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));

    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn missing_file_search_tool_fields_fail_closed() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "callout_failure_mode: closed\n");

    let mut missing_ids = one_pending_state(&["vs-a"]);
    missing_ids.tools[0].as_object_mut().unwrap().remove("vector_store_ids");
    let mut missing_ids_ctx = make_context(Some(missing_ids));
    assert!(matches!(
        filter.on_request(&mut missing_ids_ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));

    let mut missing_tool = one_pending_state(&["vs-a"]);
    missing_tool.tools.clear();
    let mut missing_tool_ctx = make_context(Some(missing_tool));
    assert!(matches!(
        filter.on_request(&mut missing_tool_ctx).await.unwrap(),
        FilterAction::Reject(_)
    ));

    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn fail_open_isolates_a_malformed_pending_call() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "callout_failure_mode: open\n");
    let malformed = json!({
        "type":"file_search_call","id":"fs-bad","status":"searching","queries":["valid", 7]
    });
    let valid = json!({
        "type":"file_search_call","id":"fs-good","status":"searching","queries":["query"]
    });
    let mut ctx = make_context(Some(state_with(&["vs-a"], vec![malformed, valid])));

    assert!(matches!(
        filter.on_request(&mut ctx).await.unwrap(),
        FilterAction::Continue
    ));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.output_items()[0]["status"], "incomplete");
    assert_eq!(state.output_items()[1]["status"], "completed");
    assert_eq!(server.requests().len(), 1);
}

// -----------------------------------------------------------------------------
// function_call → file_search_call translation
// -----------------------------------------------------------------------------

#[test]
fn extract_file_search_queries_handles_all_conventions() {
    assert_eq!(
        extract_file_search_queries(r#"{"query": "revenue"}"#),
        vec!["revenue"],
        "single query field should be extracted"
    );
    assert_eq!(
        extract_file_search_queries(r#"{"queries": ["a", "b"]}"#),
        vec!["a", "b"],
        "queries array should be extracted"
    );
    assert_eq!(
        extract_file_search_queries(r#"{"query": "", "queries": ["fallback"]}"#),
        vec!["fallback"],
        "empty query should fall through to queries array"
    );
    assert_eq!(
        extract_file_search_queries("not json"),
        vec!["not json"],
        "malformed JSON should use raw string as single query"
    );
    assert!(
        extract_file_search_queries("").is_empty(),
        "empty string should produce no queries"
    );
    assert_eq!(
        extract_file_search_queries(r#"{"other": "field"}"#),
        vec![r#"{"other": "field"}"#],
        "valid JSON without query fields should use raw string"
    );
}

#[test]
fn translate_single_query() {
    let mut response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "file_search",
            "arguments": "{\"query\": \"revenue growth\"}",
            "status": "completed"
        }]
    });

    let count = translate_function_calls_to_file_search(&mut response);

    assert_eq!(count, 1, "one item should be translated");
    let item = &response["output"][0];
    assert_eq!(item["type"], "file_search_call", "type should be rewritten");
    assert_eq!(item["status"], "searching", "status should be set to searching");
    assert_eq!(item["queries"], json!(["revenue growth"]), "query should be extracted");
    assert_eq!(item["id"], "fc_1", "id should be preserved");
    assert!(item.get("name").is_none(), "name should be removed");
    assert!(item.get("arguments").is_none(), "arguments should be removed");
    assert!(item.get("call_id").is_none(), "call_id should be removed");
}

#[test]
fn translate_queries_array() {
    let mut response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "file_search",
            "arguments": "{\"queries\": [\"revenue\", \"profit\"]}",
            "status": "completed"
        }]
    });

    translate_function_calls_to_file_search(&mut response);

    assert_eq!(
        response["output"][0]["queries"],
        json!(["revenue", "profit"]),
        "queries array should be extracted from arguments"
    );
}

#[test]
fn translate_malformed_arguments() {
    let mut response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "file_search",
            "arguments": "not valid json",
            "status": "completed"
        }]
    });

    translate_function_calls_to_file_search(&mut response);

    assert_eq!(
        response["output"][0]["queries"],
        json!(["not valid json"]),
        "malformed arguments should fall back to raw string"
    );
}

#[test]
fn translate_skips_non_file_search() {
    let mut response = json!({
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\": \"Paris\"}",
            "status": "completed"
        }]
    });

    let count = translate_function_calls_to_file_search(&mut response);

    assert_eq!(count, 0, "non-file_search function_calls should not be translated");
    assert_eq!(
        response["output"][0]["type"], "function_call",
        "type should remain function_call"
    );
}

#[test]
fn translate_skips_without_tool_declaration() {
    let state = ResponsesState {
        response_object: json!({"output": []}),
        tools: vec![json!({"type": "function", "name": "get_weather"})],
        ..Default::default()
    };
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id": "resp-no-tool",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "file_search",
                "arguments": "{\"query\": \"revenue\"}",
                "status": "completed"
            }]
        })
        .to_string(),
    ));

    let _action = FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap();

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state.output_items()[0]["type"],
        "function_call",
        "should not translate without file_search tool in request"
    );
}

#[test]
fn translate_triggers_pending_flag() {
    let state = state_with(&["vs-a"], vec![]);
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id": "resp-vllm-1",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "file_search",
                "arguments": "{\"query\": \"revenue growth\"}",
                "status": "completed"
            }]
        })
        .to_string(),
    ));

    let action = FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap();

    assert!(
        matches!(action, FilterAction::Continue),
        "translated file_search_call should trigger pending continuation"
    );
    assert_eq!(
        ctx.filter_results["openai_file_search_callout"].get("pending"),
        Some("true"),
        "pending flag should be set after translation"
    );
}

#[test]
fn translate_mixed_triggers_rejection() {
    let state = state_with(&["vs-a"], vec![]);
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id": "resp-vllm-mixed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "file_search",
                    "arguments": "{\"query\": \"revenue\"}",
                    "status": "completed"
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call_2",
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Paris\"}",
                    "status": "completed"
                }
            ]
        })
        .to_string(),
    ));

    let action = FileSearchCalloutFilter::capture_response(&mut ctx, &mut body, 268_435_456).unwrap();

    assert!(
        matches!(action, FilterAction::Reject(_)),
        "translated file_search_call + client function_call should trigger mixed-tool rejection"
    );
}

#[tokio::test]
async fn translate_end_to_end_executes_search() {
    let server = MockServer::json(200, &one_result("file-a", "report.pdf", 0.95, "Revenue grew 37%."));
    let filter = make_filter(server.port, "");

    let mut state = state_with(&["vs-a"], vec![]);
    state.include.push("file_search_call.results".to_owned());
    let mut ctx = make_context(Some(state));
    let mut response_header = crate::test_utils::make_response();
    ctx.response_header = Some(&mut response_header);
    let mut body = Some(Bytes::from(
        json!({
            "id": "resp-vllm-e2e",
            "output": [{
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "file_search",
                "arguments": "{\"query\": \"revenue growth\"}",
                "status": "completed"
            }]
        })
        .to_string(),
    ));

    let action = filter.on_response_body(&mut ctx, &mut body, true).unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "capture_response should continue with pending translation"
    );
    assert_eq!(
        ctx.filter_results["openai_file_search_callout"].get("pending"),
        Some("true"),
        "pending flag should be set"
    );

    let action = filter.on_request(&mut ctx).await.unwrap();
    assert!(
        matches!(action, FilterAction::Continue),
        "execute_pending should complete search"
    );

    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let item = &state.output_items()[0];
    assert_eq!(item["type"], "file_search_call", "type should be file_search_call");
    assert_eq!(
        item["status"], "completed",
        "status should be completed after execution"
    );
    assert_eq!(server.requests().len(), 1, "search callout should have fired");
}

// -----------------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------------

fn parse_config(yaml: &str) -> Result<ValidatedConfig, FilterError> {
    let raw: FileSearchFilterConfig =
        serde_yaml::from_str(yaml).map_err(|error| -> FilterError { error.to_string().into() })?;
    build_config(&raw)
}

fn make_filter(port: u16, extra: &str) -> Box<dyn HttpFilter> {
    Box::new(make_concrete_filter(port, extra))
}

fn make_concrete_filter(port: u16, extra: &str) -> FileSearchCalloutFilter {
    let yaml = format!("vector_store_url: 'http://127.0.0.1:{port}'\nallow_private_url: true\n{extra}");
    let raw: FileSearchFilterConfig = serde_yaml::from_str(&yaml).unwrap();
    let validated = build_config(&raw).unwrap();
    let client = FileSearchClient::new(FileSearchClientConfig {
        api_client: validated.api_client,
        failure_mode: validated.failure_mode,
        max_response_bytes: validated.max_response_bytes,
        max_total_response_bytes: validated.max_total_response_bytes,
        timeout: validated.timeout,
    });
    FileSearchCalloutFilter {
        client,
        max_state_bytes: validated.max_state_bytes,
        failure_mode: validated.failure_mode,
    }
}

fn make_context(state: Option<ResponsesState>) -> HttpFilterContext<'static> {
    let request = Box::leak(Box::new(crate::test_utils::make_request(
        http::Method::POST,
        "/v1/responses",
    )));
    let mut ctx = crate::test_utils::make_filter_context(request);
    ctx.set_metadata("openai_responses_format.stream", "false");
    if let Some(state) = state {
        ctx.extensions.insert(state);
    }
    ctx
}

fn format_test_bridge(
    results: &[SearchResult],
    query: &str,
    annotation_template: &str,
    context_template: &str,
    remaining_model_bytes: usize,
) -> BudgetedSearchResults {
    let source_item = json!({
        "type": "file_search_call",
        "id": "fs-test",
        "status": "searching",
        "queries": [query],
    });
    let known_citation_files = HashMap::new();
    let templates = FormatTemplates {
        annotation: annotation_template,
        context: context_template,
    };
    BridgeBudget {
        known_citation_files: &known_citation_files,
        max_new_citation_files: MAX_CITATION_FILES,
        remaining_model_bytes,
        source_item: &source_item,
        output_index: 0,
        query,
        response_identity_hash: FNV_OFFSET_BASIS,
        templates: &templates,
    }
    .format(results, false)
}

fn state_with(store_ids: &[&str], output_items: Vec<Value>) -> ResponsesState {
    let mut response = serde_json::Map::new();
    response.insert("output".to_owned(), Value::Array(output_items));
    ResponsesState {
        response_object: Value::Object(response),
        tools: vec![json!({
            "type": "file_search",
            "vector_store_ids": store_ids,
        })],
        ..Default::default()
    }
}

fn one_pending_state(store_ids: &[&str]) -> ResponsesState {
    state_with(
        store_ids,
        vec![json!({
            "type": "file_search_call",
            "id": "fs-1",
            "status": "searching",
            "queries": ["query"],
        })],
    )
}

fn one_result(file_id: &str, filename: &str, score: f64, text: &str) -> Value {
    json!({
        "data": [{
            "file_id": file_id,
            "filename": filename,
            "score": score,
            "content": [{"type":"text","text":text}],
            "attributes": null,
        }]
    })
}

fn search_results(results: &[(&str, f64)]) -> Value {
    json!({
        "data": results
            .iter()
            .map(|(file_id, score)| json!({
                "file_id": file_id,
                "filename": format!("{file_id}.txt"),
                "score": score,
                "content": [{"type":"text","text":file_id}],
                "attributes": null,
            }))
            .collect::<Vec<_>>()
    })
}

fn item_ids(items: &[Value]) -> Vec<&str> {
    items
        .iter()
        .filter_map(|item| item.get("id").and_then(Value::as_str))
        .collect()
}

fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    let mut buffer = Vec::with_capacity(16_384);
    let mut chunk = [0_u8; 4_096];
    loop {
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        let text = String::from_utf8_lossy(&buffer);
        if let Some(headers_end) = text.find("\r\n\r\n") {
            let content_length = text
                .get(..headers_end)
                .unwrap_or_default()
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            if buffer.len() >= headers_end.saturating_add(4).saturating_add(content_length) {
                break;
            }
        }
    }
    String::from_utf8(buffer).unwrap()
}

fn request_json(request: &str) -> Value {
    let body = request.split_once("\r\n\r\n").map(|(_, body)| body).unwrap();
    serde_json::from_str(body).unwrap()
}

struct MockServer {
    max_active: Arc<AtomicUsize>,
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct MockResponse {
    body: String,
    body_delay: Duration,
    status: u16,
}

impl MockServer {
    fn json(status: u16, body: &Value) -> Self {
        Self::start(status, body, Duration::ZERO)
    }

    fn slow_body(body: &Value, delay: Duration) -> Self {
        Self::start(200, body, delay)
    }

    fn start(status: u16, body: &Value, body_delay: Duration) -> Self {
        let response = MockResponse {
            body: body.to_string(),
            body_delay,
            status,
        };
        Self::start_with(move |_path| response.clone())
    }

    fn routes<const N: usize>(routes: [(&str, u16, Value, Duration); N]) -> Self {
        let routes: HashMap<String, MockResponse> = routes
            .into_iter()
            .map(|(store_id, status, body, body_delay)| {
                (
                    format!("/v1/vector_stores/{store_id}/search"),
                    MockResponse {
                        body: body.to_string(),
                        body_delay,
                        status,
                    },
                )
            })
            .collect();
        Self::start_with(move |path| {
            routes.get(path).cloned().unwrap_or_else(|| MockResponse {
                body: json!({"error":"route not found"}).to_string(),
                body_delay: Duration::ZERO,
                status: 404,
            })
        })
    }

    fn start_with(responder: impl Fn(&str) -> MockResponse + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let server_active = Arc::clone(&active);
        let server_max_active = Arc::clone(&max_active);
        let responder = Arc::new(responder);

        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut stream) = connection else { break };
                let captured = Arc::clone(&captured);
                let active = Arc::clone(&server_active);
                let max_active = Arc::clone(&server_max_active);
                let responder = Arc::clone(&responder);
                std::thread::spawn(move || {
                    let request = read_http_request(&mut stream);
                    let path = request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or("/")
                        .to_owned();
                    captured.lock().unwrap().push(request);

                    let current = active.fetch_add(1, Ordering::SeqCst).saturating_add(1);
                    max_active.fetch_max(current, Ordering::SeqCst);
                    let response = responder(&path);
                    let reason = if response.status == 200 { "OK" } else { "Error" };
                    let headers = format!(
                        "HTTP/1.1 {} {reason}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n",
                        response.status,
                        response.body.len()
                    );
                    if stream.write_all(headers.as_bytes()).is_ok() {
                        if !response.body_delay.is_zero() {
                            std::thread::park_timeout(response.body_delay);
                        }
                        let _result = stream.write_all(response.body.as_bytes());
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        Self {
            max_active,
            port,
            requests,
        }
    }

    fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}
