// SPDX-License-Identifier: Apache-2.0
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

use praxis_filter::{
    BodyMode, FilterAction, FilterEntry, FilterPipeline, FilterRegistry, HttpFilter, TrustedHeaderMutation,
};
use serde_json::{Value, json};

use super::{
    client::{
        ContentChunk, ContentChunkType, FileSearchClient, FileSearchClientConfig, MAX_CONCURRENT_SEARCHES,
        MAX_QUERY_BYTES, MAX_SEARCH_REQUEST_BYTES, MAX_VECTOR_STORE_ID_BYTES, SearchResult, VectorStoreSearchRequest,
        request_error,
    },
    config::{FileSearchFilterConfig, ValidatedConfig, build_config_with_client},
    *,
};
use crate::{
    callout_policy::OnFailure,
    openai::responses::state::{FileSearchAssignment, SynthesisKind},
    subrequest::{SubRequestClient, SubRequestError, SubResponse},
};
// -----------------------------------------------------------------------------
// Configuration and transport validation
// -----------------------------------------------------------------------------

#[test]
fn minimal_config_uses_safe_defaults() {
    let raw: FileSearchFilterConfig = serde_yaml::from_str(
        r#"
        vector_store_url: "https://8.8.8.8"
        outbound_chain: vector_store_chain
        "#,
    )
    .unwrap();
    let config = build_config_with_client(&raw, test_subrequest_client()).unwrap();

    assert_eq!(config.base_url, "https://8.8.8.8");
    assert_eq!(config.max_response_bytes, 10_485_760);
    assert_eq!(config.max_total_response_bytes, 67_108_864);
    assert_eq!(config.max_state_bytes, 52_428_800);
    assert_eq!(config.on_failure, OnFailure::Closed);
}

#[test]
fn config_rejects_unknown_and_removed_circuit_breaker_fields() {
    for yaml in [
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\nunknown: true\n",
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\ncircuit_breaker: {}\n",
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\nsearch_template: '{query}'\n",
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\nannotation_template: '{content}'\n",
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\ncontext_template: '{results}'\n",
        // The removed per-filter SSRF opt-in is now an unknown field.
        "outbound_chain: vector_store_chain\nvector_store_url: https://8.8.8.8\nallow_private_url: true\n",
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
fn config_defers_private_targets_to_the_connect_time_ssrf_gate() {
    // Private/loopback literals and DNS names all pass config-time structural
    // validation. SSRF is decided at connect time by `prepare_url_target`, which
    // honours `insecure_options.allow_private_upstreams`; startup cannot see that
    // flag, so it must not reject a literal private target here (that would make
    // the central opt-in unable to ever permit one).
    for url in [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://10.0.0.1:8001",
        "http://169.254.169.254:8001",
        "http://100.64.0.1:8001",
        "http://0.7.8.9:8001",
        "http://[::1]:8001",
        "http://[::ffff:10.0.0.1]:8001",
        "http://vector-store.example:8001",
    ] {
        assert!(
            parse_config(&format!("vector_store_url: '{url}'\n")).is_ok(),
            "private-address gating is deferred to the runtime hook: {url}"
        );
    }
}

#[tokio::test]
async fn build_config_with_client_shares_connector_circuit_state() {
    use praxis_core::{
        circuit::CircuitBreakerConfig,
        subrequest::{SubRequestConnector, SubRequestConnectorOptions, SubRequestError},
    };

    let shared = SubRequestClient::with_max_response_bytes(
        SubRequestConnector::with_options(SubRequestConnectorOptions {
            keepalive_pool_size: 4,
            max_connections: None,
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: 1,
                recovery_window: Duration::from_secs(30),
                half_open_timeout: Duration::from_secs(30),
            }),
        }),
        usize::MAX,
    );
    // Any structurally-valid target passes config-time validation; the callout
    // still inherits the shared connector's circuit.
    let raw: FileSearchFilterConfig = serde_yaml::from_str(
        "
vector_store_url: http://vector-store.test
outbound_chain: vector_store_chain
",
    )
    .unwrap();
    let inherited = build_config_with_client(&raw, shared.clone()).unwrap();
    let peer_addr = closed_loopback_addr();

    let first = execute_against(&shared, &peer_addr).await;
    assert!(
        matches!(first, Err(SubRequestError::Connect(_))),
        "first refusal should record a failure on the shared connector; got {first:?}"
    );
    let second = execute_against(&shared, &peer_addr).await;
    assert!(
        matches!(second, Err(SubRequestError::CircuitOpen { .. })),
        "shared connector should open after threshold 1; got {second:?}"
    );

    let inherited_result = execute_against(&inherited.subrequest_client, &peer_addr).await;
    assert!(
        matches!(inherited_result, Err(SubRequestError::CircuitOpen { .. })),
        "build_config_with_client must keep the callout on the same connector Arc; got {inherited_result:?}"
    );
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
fn search_result_requires_content() {
    let response: SearchResult = serde_json::from_value(json!({
        "file_id": "file-a",
        "filename": "a.txt",
        "score": 0.5,
        "content": []
    }))
    .unwrap();
    assert!(response.content.is_empty());
    assert!(response.attributes.is_none());
    assert!(
        serde_json::from_value::<SearchResult>(json!({"file_id":"file-a","filename":"a.txt","score":0.5})).is_err(),
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

    let plan = build_search_plan(&state, &state.file_search_assignments);
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

    let plan = build_search_plan(&state, &state.file_search_assignments);

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
fn callout_headers_apply_trusted_body_phase_overlay() {
    let mut request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    request
        .headers
        .insert("x-auth-user", http::HeaderValue::from_static("raw-assertion"));
    request
        .headers
        .insert("x-user-id", http::HeaderValue::from_static("spoofed-user"));
    let request = Box::leak(Box::new(request));
    let mut ctx = crate::test_utils::make_filter_context(request);
    ctx.pre_read_mutations
        .push(TrustedHeaderMutation::Remove(http::HeaderName::from_static(
            "x-auth-user",
        )));
    ctx.pre_read_mutations.push(TrustedHeaderMutation::Set(
        http::HeaderName::from_static("x-user-id"),
        http::HeaderValue::from_static("alice"),
    ));

    let headers = callout_request_headers(&ctx);

    assert!(headers.get("x-auth-user").is_none());
    assert_eq!(headers.get("x-user-id").unwrap(), "alice");
}

#[tokio::test]
async fn no_state_or_pending_calls_is_a_noop() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    // No request state at all: the dispatcher drains nothing and continues.
    let mut ctx = make_context(None);
    assert!(
        matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue),
        "a dispatcher without Responses state should be a no-op"
    );

    // State present but with no pending file-search assignments: still a no-op.
    ctx.extensions.insert(state_with(&["vs-a"], vec![]));
    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
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
    let mut state = state_with(&["vs-a"], output);
    state.include.push("file_search_call.results".to_owned());
    state
        .messages
        .push(json!({"type":"message","id":"input-1","role":"user","content":"search"}));
    let mut ctx = make_context(Some(state));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    // The dispatcher reconciles the assigned call in place inside `accumulated_output`
    // and leaves every sibling output item untouched and in original order.
    assert_eq!(
        state
            .accumulated_output
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["reasoning", "file_search_call", "mcp_list_tools", "message"]
    );
    assert_eq!(state.accumulated_output[1]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["results"][0]["text"], "Revenue grew.");
    assert_eq!(
        state.citation_files.get("file-a").map(String::as_str),
        Some("report.pdf")
    );
    // Only the private model-context bridge is appended to `messages`; routing of
    // reasoning/message items is the loop owner's responsibility, not the dispatcher's.
    assert_eq!(
        state
            .messages
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["message", "function_call", "function_call_output"]
    );
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

    // Re-entry is a no-op: the assignments were drained by the first dispatch.
    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.messages.len(), message_len);
}

#[tokio::test]
async fn outbound_chain_filters_run_on_every_sub_request() {
    // Every vector-store sub-request is dispatched through the configured
    // `outbound_chain`. Wire a chain whose `headers` filter stamps a marker and
    // assert the store observed it on each of two fan-out sub-requests: the chain
    // is not a passive destination selector, its filters execute per callout.
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.9, "hit"));
    let filter = make_concrete_filter_with_outbound(
        server.port,
        "",
        outbound_pipeline_stamping_header("X-Vector-Store-Client", "praxis-ai-gateway"),
    );
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b"])));

    assert!(matches!(dispatch(&filter, &mut ctx).await, FilterAction::Continue));
    let requests = server.requests();
    assert_eq!(requests.len(), 2, "both stores should be searched");
    for request in &requests {
        assert!(
            request
                .lines()
                .any(|line| line.eq_ignore_ascii_case("X-Vector-Store-Client: praxis-ai-gateway")),
            "outbound chain filter must stamp the marker header on each sub-request; got:\n{request}"
        );
    }
    // The callout still reconciled the assignment, so the chain runs on the real
    // dispatch path rather than a discarded probe.
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "completed");
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
    let mut ctx = make_context(Some(state_with(&["vs-a"], output)));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let generated_id = state.accumulated_output[0]["id"].as_str().unwrap();
    assert!(generated_id.starts_with("fs_"));
    assert_eq!(state.accumulated_output[1]["id"], long_id);
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
    assert!(state.accumulated_output[0].get("results").is_none());
    assert!(state.accumulated_output[1].get("results").is_none());
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "completed");
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert!(
        state
            .accumulated_output
            .iter()
            .filter(|item| item["type"] == "file_search_call")
            .all(|item| item["status"] == "incomplete")
    );
    // Both the planned calls and the cap-dropped excess receive a valid public id.
    assert!(
        state
            .accumulated_output
            .iter()
            .filter(|item| item["type"] == "file_search_call")
            .all(|item| item["id"].as_str().is_some_and(|id| !id.is_empty()))
    );
    // Only the planned calls (capped at `MAX_PENDING_CALLS`) emit a model-context bridge.
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
            .accumulated_output
            .iter()
            .filter(|item| item["id"] == "duplicate")
            .count(),
        2
    );
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn dispatcher_executes_owner_admitted_assignment_with_mcp_history() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mcp = json!({"type":"mcp_call","id":"mcp-prior","status":"completed"});
    let pending = json!({"type":"file_search_call","id":"fs-new","status":"searching","queries":["q"]});
    let state = state_with(&["vs-a"], vec![mcp, pending]);
    let mut ctx = make_context(Some(state));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    // The owner already applied the response-wide budget before recording this
    // assignment. The execution-only dispatcher must not re-count MCP history.
    assert_eq!(state.accumulated_output[1]["status"], "completed");
    assert!(
        !server.requests().is_empty(),
        "the budget-admitted file search must reach the upstream"
    );
}

#[tokio::test]
async fn no_store_ids_terminalizes_the_assigned_call() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let output = vec![
        json!({"type":"reasoning","id":"rs-1","summary":[],"encrypted_content":"opaque-reasoning"}),
        json!({"type":"file_search_call","id":"fs-1","status":"searching","queries":["q"]}),
    ];
    let mut ctx = make_context(Some(state_with(&[], output)));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    // With no vector stores the assigned call is terminalized incomplete in place.
    assert_eq!(state.accumulated_output[1]["status"], "incomplete");
    assert!(state.accumulated_output[1].get("results").is_none());
    // The sibling reasoning item stays untouched in `accumulated_output`; the
    // dispatcher only appends its own private bridge to `messages` (sibling
    // replay is the loop owner's responsibility, not the dispatcher's).
    assert_eq!(state.accumulated_output[0]["encrypted_content"], "opaque-reasoning");
    assert_eq!(
        state
            .messages
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["function_call", "function_call_output"]
    );
    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn zero_match_search_completes_with_bounded_context() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();

    assert_eq!(state.accumulated_output[0]["status"], "completed");
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
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
async fn forwarded_credentials_stay_pinned_to_config_upstream_despite_hostile_store_id() {
    // The vector-store destination is derived solely from operator config
    // (`vector_store_url`); a client/model-controlled `vector_store_id` only fills
    // a single, percent-encoded path segment. A store ID crafted to look like an
    // alternate authority must not retarget the callout: the forwarded credential
    // has to reach the config-pinned upstream and nowhere else. The 0.5.6 executor
    // pins the `StagedUpstream` derived from the prepared target and re-pins after
    // the request phase, so no chain filter can move the credential either.
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_concrete_filter_with_outbound(
        server.port,
        "on_failure: closed\nforward_headers:\n  - authorization\n",
        test_outbound_pipeline(),
    );
    // A store ID that tries to smuggle in an alternate host authority.
    let hostile_store_id = "vs-a@evil.example.com/steal";
    let mut ctx = make_context_with_request_headers(
        Some(one_pending_state(&[hostile_store_id])),
        &[("authorization", "Bearer super-secret-token")],
    );

    assert!(matches!(dispatch(&filter, &mut ctx).await, FilterAction::Continue));
    let requests = server.requests();
    // Exactly one callout, and it reached the config-pinned MockServer.
    assert_eq!(
        requests.len(),
        1,
        "credential must reach exactly the configured upstream"
    );
    let request = &requests[0];

    // The forwarded credential is present on the callout to the trusted host.
    assert!(
        request
            .lines()
            .any(|line| line.eq_ignore_ascii_case("authorization: Bearer super-secret-token")),
        "forwarded credential must be sent to the configured upstream; got:\n{request}"
    );
    // The authority stays the config-derived loopback host: the hostile store ID
    // could not move the credential to `evil.example.com`.
    assert!(
        request
            .lines()
            .any(|line| line.to_ascii_lowercase().starts_with("host: 127.0.0.1:")),
        "callout Host must be the config authority, not the client store ID; got:\n{request}"
    );
    assert!(
        !request.contains("evil.example.com"),
        "hostile store ID must stay percent-encoded in the path, never an authority; got:\n{request}"
    );
    // The store ID is confined to a single, percent-encoded path segment.
    let request_line = request.lines().next().unwrap();
    assert!(
        request_line.contains("evil%2Eexample%2Ecom%2Fsteal"),
        "store ID must be percent-encoded into one path segment; got:\n{request_line}"
    );
}

#[tokio::test]
async fn open_and_closed_on_failures_are_distinct() {
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
        let closed = make_filter(closed_server.port, "on_failure: closed\n");
        let mut closed_ctx = make_context(Some(one_pending_state(&["vs-a"])));
        // The dispatcher never rejects: a closed-mode failure records a shared
        // 502 dispatch failure for the loop owner and leaves the call untouched.
        assert!(matches!(
            dispatch(&*closed, &mut closed_ctx).await,
            FilterAction::Continue
        ));
        let closed_state = closed_ctx.extensions.get::<ResponsesState>().unwrap();
        assert_eq!(
            closed_state
                .dispatch_failure
                .as_ref()
                .unwrap_or_else(|| panic!("status {status} must record a dispatch failure"))
                .status,
            502,
            "status {status}"
        );
        assert_eq!(
            closed_state.accumulated_output[0]["status"], "searching",
            "status {status}"
        );

        let open_server = MockServer::start_with(move |_path| MockResponse {
            body: body.clone(),
            body_delay: Duration::ZERO,
            status,
        });
        let open = make_filter(open_server.port, "on_failure: open\n");
        let mut open_ctx = make_context(Some(one_pending_state(&["vs-a"])));
        assert!(matches!(dispatch(&*open, &mut open_ctx).await, FilterAction::Continue));
        let state = open_ctx.extensions.get::<ResponsesState>().unwrap();
        assert!(state.dispatch_failure.is_none(), "status {status}");
        assert_eq!(state.accumulated_output[0]["status"], "incomplete", "status {status}");
        assert!(state.accumulated_output[0].get("results").is_none());
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
        "on_failure: open\nmax_response_bytes: 512\nmax_total_response_bytes: 512\n",
    );
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b"])));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        server.requests().len(),
        1,
        "second request should not consume an unreserved body"
    );
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
    assert!(state.accumulated_output[0].get("results").is_none());
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
        "on_failure: open\nmax_response_bytes: 1\nmax_total_response_bytes: 4\n",
    );
    let mut ctx = make_context(Some(one_pending_state(&["vs-a", "vs-b", "vs-c", "vs-d", "vs-e"])));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    assert_eq!(
        server.requests().len(),
        4,
        "malformed HTTP successes must consume the shared body budget"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
}

#[tokio::test]
async fn core_limit_fails_closed_on_oversized_response_maps_to_413() {
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.9, &"x".repeat(4_096)));
    let filter = make_filter(server.port, "max_response_bytes: 128\nmax_total_response_bytes: 128\n");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));
    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    // `run_classified` surfaces the size overflow as `CalloutOutcome::ResponseTooLarge`
    // instead of collapsing it into an opaque 502. The default closed policy records
    // an actionable 413 dispatch failure and leaves the call unreconciled rather than
    // rejecting inline, mirroring the continuation-state ceiling.
    let failure = state
        .dispatch_failure
        .as_ref()
        .expect("oversized response must record a dispatch failure");
    assert_eq!(failure.status, 413, "oversized response must map to HTTP 413");
    assert_eq!(failure.code, "invalid_request_error");
    assert!(
        failure.message.contains("exceeded the 128-byte limit"),
        "message should name the exceeded ceiling: {}",
        failure.message
    );
    assert_eq!(state.accumulated_output[0]["status"], "searching");
}

#[tokio::test]
async fn whole_call_timeout_covers_slow_response_body() {
    let server = MockServer::slow_body(&json!({"data": []}), Duration::from_millis(500));
    let filter = make_filter(server.port, "timeout_ms: 50\n");
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));
    let started = Instant::now();

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    assert!(started.elapsed() < Duration::from_millis(400));
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );
}

#[tokio::test]
async fn runtime_ssrf_gate_rejects_private_target_without_opt_in() {
    // The store resolves to a loopback address. Without
    // `allow_private_upstreams` on the outbound pipeline, the connect-time
    // `prepare_url_target` hook must reject it before any dial. The closed policy
    // records a 502 dispatch failure and leaves the call unreconciled, and the
    // store never receives a request.
    let server = MockServer::json(200, &one_result("file-a", "a.txt", 0.9, "unreached"));
    let filter =
        make_concrete_filter_with_outbound(server.port, "on_failure: closed\n", deny_private_outbound_pipeline());
    let mut ctx = make_context(Some(one_pending_state(&["vs-a"])));

    assert!(matches!(dispatch(&filter, &mut ctx).await, FilterAction::Continue));
    assert!(
        server.requests().is_empty(),
        "SSRF rejection must occur before any sub-request reaches the store"
    );
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(
        state
            .dispatch_failure
            .as_ref()
            .expect("a rejected private dial must record a dispatch failure")
            .status,
        502
    );
    assert_eq!(state.accumulated_output[0]["status"], "searching");
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );
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
    let filter = make_filter(server.port, "on_failure: closed\n");
    let store_ids: Vec<String> = (0..=MAX_CONCURRENT_SEARCHES)
        .map(|index| format!("vs-{index}"))
        .collect();
    let store_refs: Vec<&str> = store_ids.iter().map(String::as_str).collect();
    let mut ctx = make_context(Some(one_pending_state(&store_refs)));

    assert!(
        matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue),
        "fail-closed dispatch should record its failure without committing a response"
    );
    assert!(
        ctx.extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some(),
        "the first failed search chunk must publish a dispatch failure"
    );
    let requests = server.requests();
    assert!(
        !requests.is_empty(),
        "the first concurrent search chunk should reach the provider"
    );
    assert!(
        requests.len() <= MAX_CONCURRENT_SEARCHES,
        "fail-closed scheduling must not start work beyond the current bounded chunk"
    );
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

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
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
    let filter = make_filter(server.port, "on_failure: open\n");
    let mut state = one_pending_state(&["vs-a", "vs-b"]);
    state.tools[0]["max_num_results"] = json!(3);
    state.include.push("file_search_call.results".to_owned());
    let mut ctx = make_context(Some(state));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    let results = state.accumulated_output[0]["results"].as_array().unwrap();
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
    let filter = make_filter(server.port, "on_failure: open\n");
    let mut state = one_pending_state(&["vs-a", "vs-b"]);
    state.include.push("file_search_call.results".to_owned());
    let mut ctx = make_context(Some(state));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(server.requests().len(), 2);
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
    assert_eq!(state.accumulated_output[0]["results"].as_array().unwrap().len(), 1);
    assert_eq!(state.accumulated_output[0]["results"][0]["file_id"], "file-a");
}

#[tokio::test]
async fn outbound_query_store_id_and_request_body_are_bounded() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "on_failure: open\n");

    let oversized_store = "s".repeat(MAX_VECTOR_STORE_ID_BYTES + 1);
    let mut store_ctx = make_context(Some(one_pending_state(&[&oversized_store])));
    assert!(matches!(
        dispatch(&*filter, &mut store_ctx).await,
        FilterAction::Continue
    ));

    let mut query_state = one_pending_state(&["vs-query"]);
    query_state.accumulated_output[0]["queries"] = json!(["q".repeat(MAX_QUERY_BYTES + 1)]);
    let mut query_ctx = make_context(Some(query_state));
    assert!(matches!(
        dispatch(&*filter, &mut query_ctx).await,
        FilterAction::Continue
    ));

    let mut request_state = one_pending_state(&["vs-request"]);
    request_state.tools[0]["filters"] = json!({"type":"eq","key":"blob","value":"x".repeat(MAX_SEARCH_REQUEST_BYTES)});
    let mut request_ctx = make_context(Some(request_state));
    assert!(matches!(
        dispatch(&*filter, &mut request_ctx).await,
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
    let filter = make_filter(server.port, "on_failure: closed\n");

    let mut invalid_stores = one_pending_state(&["vs-a"]);
    invalid_stores.tools[0]["vector_store_ids"] = json!(["vs-a", 7]);
    let mut invalid_stores_ctx = make_context(Some(invalid_stores));
    assert!(matches!(
        dispatch(&*filter, &mut invalid_stores_ctx).await,
        FilterAction::Continue
    ));
    assert!(
        invalid_stores_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );

    let mut invalid_max = one_pending_state(&["vs-a"]);
    invalid_max.tools[0]["max_num_results"] = json!("10");
    let mut invalid_max_ctx = make_context(Some(invalid_max));
    assert!(matches!(
        dispatch(&*filter, &mut invalid_max_ctx).await,
        FilterAction::Continue
    ));
    assert!(
        invalid_max_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );

    let mut invalid_queries = one_pending_state(&["vs-a"]);
    invalid_queries.accumulated_output[0]["queries"] = json!(["valid", 7]);
    let mut invalid_queries_ctx = make_context(Some(invalid_queries));
    assert!(matches!(
        dispatch(&*filter, &mut invalid_queries_ctx).await,
        FilterAction::Continue
    ));
    assert!(
        invalid_queries_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );

    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn missing_file_search_tool_fields_fail_closed() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "on_failure: closed\n");

    let mut missing_ids = one_pending_state(&["vs-a"]);
    missing_ids.tools[0].as_object_mut().unwrap().remove("vector_store_ids");
    let mut missing_ids_ctx = make_context(Some(missing_ids));
    assert!(matches!(
        dispatch(&*filter, &mut missing_ids_ctx).await,
        FilterAction::Continue
    ));
    assert!(
        missing_ids_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );

    let mut missing_tool = one_pending_state(&["vs-a"]);
    missing_tool.tools.clear();
    let mut missing_tool_ctx = make_context(Some(missing_tool));
    assert!(matches!(
        dispatch(&*filter, &mut missing_tool_ctx).await,
        FilterAction::Continue
    ));
    assert!(
        missing_tool_ctx
            .extensions
            .get::<ResponsesState>()
            .unwrap()
            .dispatch_failure
            .is_some()
    );

    assert!(server.requests().is_empty());
}

#[tokio::test]
async fn fail_open_isolates_a_malformed_pending_call() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "on_failure: open\n");
    let malformed = json!({
        "type":"file_search_call","id":"fs-bad","status":"searching","queries":["valid", 7]
    });
    let valid = json!({
        "type":"file_search_call","id":"fs-good","status":"searching","queries":["query"]
    });
    let mut ctx = make_context(Some(state_with(&["vs-a"], vec![malformed, valid])));

    assert!(matches!(dispatch(&*filter, &mut ctx).await, FilterAction::Continue));
    let state = ctx.extensions.get::<ResponsesState>().unwrap();
    assert_eq!(state.accumulated_output[0]["status"], "incomplete");
    assert_eq!(state.accumulated_output[1]["status"], "completed");
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

    assert_eq!(count, vec![0], "one item translated at round-local index 0");
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

    assert!(
        count.is_empty(),
        "non-file_search function_calls should not be translated"
    );
    assert_eq!(
        response["output"][0]["type"], "function_call",
        "type should remain function_call"
    );
}

// -----------------------------------------------------------------------------
// Test helpers
// -----------------------------------------------------------------------------

fn parse_config(yaml: &str) -> Result<ValidatedConfig, FilterError> {
    // `outbound_chain` is a required field; inject a named reference so URL and
    // budget assertions exercise `build_config_with_client` without every caller
    // repeating the chain line.
    let yaml = format!("outbound_chain: vector_store_chain\n{yaml}");
    let raw: FileSearchFilterConfig =
        serde_yaml::from_str(&yaml).map_err(|error| -> FilterError { error.to_string().into() })?;
    build_config_with_client(&raw, test_subrequest_client())
}

/// A default sub-request client for config-parsing tests that never dials.
fn test_subrequest_client() -> SubRequestClient {
    use praxis_core::subrequest::SubRequestConnector;

    SubRequestClient::new(SubRequestConnector::new(4, None))
}

/// An outbound pipeline that permits loopback/private upstreams so tests can
/// dial the in-process `MockServer` on `127.0.0.1`. Production wires this policy
/// from `insecure_options.allow_private_upstreams`; here we opt in explicitly.
fn test_outbound_pipeline() -> Arc<FilterPipeline> {
    let registry = FilterRegistry::with_builtins();
    let mut pipeline = FilterPipeline::build(&mut [], &registry).expect("empty pipeline builds");
    pipeline.set_allow_private_upstreams(true);
    Arc::new(pipeline)
}

/// An outbound pipeline that does not permit private upstreams, mirroring the
/// default `insecure_options` posture. The connect-time SSRF hook rejects any
/// resolved private/loopback address dialed through it.
fn deny_private_outbound_pipeline() -> Arc<FilterPipeline> {
    let registry = FilterRegistry::with_builtins();
    let pipeline = FilterPipeline::build(&mut [], &registry).expect("empty pipeline builds");
    Arc::new(pipeline)
}

/// An outbound pipeline whose single `headers` filter stamps a request header on
/// every sub-request, mirroring the credential/observability chain shipped in the
/// example configs. Loopback dials are permitted so the sub-request reaches the
/// in-process `MockServer`; the stamped header proves the chain's filters ran.
fn outbound_pipeline_stamping_header(name: &str, value: &str) -> Arc<FilterPipeline> {
    let registry = FilterRegistry::with_builtins();
    let mut entries: Vec<FilterEntry> = vec![
        serde_yaml::from_str(&format!(
            "filter: headers\nrequest_set:\n  - name: {name}\n    value: {value}\n"
        ))
        .expect("headers filter entry parses"),
    ];
    let mut pipeline = FilterPipeline::build(&mut entries, &registry).expect("headers pipeline builds");
    pipeline.set_allow_private_upstreams(true);
    Arc::new(pipeline)
}

fn closed_loopback_addr() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr.to_string()
}

async fn execute_against(client: &SubRequestClient, peer_addr: &str) -> Result<SubResponse, SubRequestError> {
    use bytes::Bytes;
    use http::HeaderMap;
    use pingora_core::upstreams::peer::HttpPeer;
    use praxis_core::subrequest::SubRequest;

    let peer = HttpPeer::new(peer_addr.to_owned(), false, String::new());
    let request = SubRequest {
        method: http::Method::GET,
        uri: "/".parse().unwrap(),
        headers: HeaderMap::new(),
        body: Bytes::new(),
    };
    Box::pin(client.execute(&peer, &request, 1024, Duration::from_secs(2), None)).await
}

fn make_filter(port: u16, extra: &str) -> Box<dyn HttpFilter> {
    Box::new(make_concrete_filter(port, extra))
}

fn make_concrete_filter(port: u16, extra: &str) -> FileSearchCalloutFilter {
    make_concrete_filter_with_outbound(port, extra, test_outbound_pipeline())
}

fn make_concrete_filter_with_outbound(
    port: u16,
    extra: &str,
    outbound: Arc<FilterPipeline>,
) -> FileSearchCalloutFilter {
    // Parse budgets/policy through the real config path with a DNS placeholder URL,
    // then repoint the client at the in-process `MockServer`. The supplied outbound
    // pipeline decides whether the runtime SSRF hook allows the loopback dial.
    let yaml = format!("vector_store_url: 'http://vector-store.test'\noutbound_chain: vector_store_chain\n{extra}");
    let raw: FileSearchFilterConfig = serde_yaml::from_str(&yaml).unwrap();
    let validated = build_config_with_client(&raw, test_subrequest_client()).unwrap();
    let client = FileSearchClient::new(FileSearchClientConfig {
        base_url: format!("http://127.0.0.1:{port}"),
        subrequest_client: validated.subrequest_client,
        forward_header_names: validated.forward_header_names,
        on_failure: validated.on_failure,
        max_response_bytes: validated.max_response_bytes,
        max_total_response_bytes: validated.max_total_response_bytes,
        timeout: validated.timeout,
    });
    FileSearchCalloutFilter {
        client,
        outbound,
        max_state_bytes: validated.max_state_bytes,
        on_failure: validated.on_failure,
    }
}

fn make_context(state: Option<ResponsesState>) -> HttpFilterContext<'static> {
    make_context_with_request_headers(state, &[])
}

/// Like [`make_context`] but seeds the downstream request with the given
/// headers so `forward_headers` has something to copy onto the callout.
fn make_context_with_request_headers(
    state: Option<ResponsesState>,
    headers: &[(&'static str, &str)],
) -> HttpFilterContext<'static> {
    let mut request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    for (name, value) in headers {
        request.headers.insert(
            http::HeaderName::from_static(name),
            http::HeaderValue::from_str(value).unwrap(),
        );
    }
    let request = Box::leak(Box::new(request));
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

/// Build a state the way the parse owner (`openai_agentic_loop`) leaves it for the
/// demoted dispatcher (issue #1046): every output item is appended to
/// `accumulated_output`, and each *pending* `file_search_call` is recorded as a
/// native `FileSearchAssignment`. The dispatcher reads `accumulated_output` and
/// mutates the assigned items in place; it no longer parses `response_object`.
fn state_with(store_ids: &[&str], output_items: Vec<Value>) -> ResponsesState {
    let mut state = ResponsesState {
        tools: vec![json!({
            "type": "file_search",
            "vector_store_ids": store_ids,
        })],
        ..Default::default()
    };
    register_assignments(&mut state, output_items);
    state
}

/// Append `output_items` to `accumulated_output` and record a native
/// `FileSearchAssignment` for every pending `file_search_call`, mirroring the
/// owner's `collect_output_items`.
fn register_assignments(state: &mut ResponsesState, output_items: Vec<Value>) {
    for item in output_items {
        let output_index = state.accumulated_output.len();
        let pending = is_pending_file_search_call(&item);
        state.accumulated_output.push(item);
        if pending {
            state.file_search_assignments.push(FileSearchAssignment {
                output_index,
                synthesis: SynthesisKind::Native,
            });
        }
    }
}

/// Drive the demoted dispatcher the way the IRR request phase does: run
/// `on_request_body` at end-of-stream, which drains `file_search_assignments`,
/// executes the vector-store callouts, and reconciles `accumulated_output`.
async fn dispatch(filter: &dyn HttpFilter, ctx: &mut HttpFilterContext<'_>) -> FilterAction {
    filter.on_request_body(ctx, &mut None, true).await.unwrap()
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

#[test]
fn file_search_predicates_are_crate_visible() {
    use super::{has_file_search_tool, is_file_search_function_call, is_pending_file_search_call};
    let pending = serde_json::json!({"type": "file_search_call", "status": "searching"});
    let private = serde_json::json!({"type": "function_call", "name": "file_search"});
    assert!(
        is_pending_file_search_call(&pending),
        "searching file call should be pending"
    );
    assert!(
        is_file_search_function_call(&private),
        "private file-search function call should be recognized"
    );
    let mut state = ResponsesState::default();
    state.tools = vec![serde_json::json!({"type": "file_search"})];
    assert!(
        has_file_search_tool(&state),
        "configured file-search tool should be detected"
    );
}

#[test]
fn response_body_mode_is_stream() {
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    assert!(matches!(filter.response_body_mode(), BodyMode::Stream));
}

// #313 P1 (DoS bound): the per-round provider-streamed observation set is charged against
// max_state_bytes like every other request-scoped field, so a backend streaming many distinct
// native ids in one round cannot bypass the ceiling within that round.
#[test]
fn continuation_state_charges_provider_streamed_terminal_ids() {
    let mut state = ResponsesState::default();
    assert!(
        continuation_state_fits(0, &state, 64, 0),
        "an empty observation set fits a tiny ceiling"
    );
    // ~48 bytes each × 100 ids ≫ 64-byte ceiling: only charged bytes can trip this.
    state.provider_streamed_terminal_ids = (0..100).map(|i| format!("fs_call_{i:040}")).collect();
    assert!(
        !continuation_state_fits(0, &state, 64, 0),
        "a large observation set is charged and overflows the ceiling (P1 DoS bound)"
    );
}

#[test]
fn continuation_state_charges_local_completion_response_template() {
    let mut state = ResponsesState {
        local_completion_response_template: json!({
            "id": "resp_prior",
            "object": "response",
            "status": "completed",
            "metadata": "x".repeat(128),
            "output": []
        }),
        ..ResponsesState::default()
    };
    assert!(
        !continuation_state_fits(0, &state, 64, 0),
        "the re-entry response template must remain inside the continuation ceiling"
    );

    state.local_completion_response_template = Value::Null;
    assert!(
        continuation_state_fits(0, &state, 64, 0),
        "clearing the response template must release its continuation-state charge"
    );
}

// -----------------------------------------------------------------------------
// Native request-side file_search lowering
//
// A native `/v1/responses` backend (e.g. vLLM) cannot consume the hosted
// `{"type":"file_search"}` tool, so this dispatcher lowers it into a private
// Responses function before dispatch. Lowering mutates only the outbound body
// (`ResponsesState.request_body`) and leaves `state.tools`/`state.tool_choice`
// holding the hosted configuration the dispatcher and response normalizer read.
// -----------------------------------------------------------------------------

/// Assert the action is a rejection and return its `(status, message)`.
fn reject_parts(action: &FilterAction) -> (u16, String) {
    let FilterAction::Reject(rejection) = action else {
        panic!("expected FilterAction::Reject");
    };
    let body = rejection.body.clone().expect("rejection has a body");
    let parsed: Value = serde_json::from_slice(&body).expect("rejection body is JSON");
    let message = parsed["error"]["message"]
        .as_str()
        .expect("rejection carries an error message")
        .to_owned();
    (rejection.status, message)
}

#[test]
fn native_lowering_replaces_hosted_file_search_tool_with_private_function() {
    let mut state = ResponsesState::from_request_body(json!({
        "model": "qwen",
        "input": "find the launch checklist",
        "tools": [{
            "type": "file_search",
            "vector_store_ids": ["vs_123"],
            "max_num_results": 5
        }]
    }));

    lower_native_file_search(&mut state).expect("valid hosted file_search lowers");

    let lowered = &state.request_body["tools"][0];
    assert_eq!(lowered["type"], json!("function"));
    assert_eq!(lowered["name"], json!("file_search"));
    assert_eq!(lowered["strict"], json!(true));
    assert_eq!(
        lowered["parameters"]["properties"]["query"]["maxLength"],
        json!(65_536),
        "the flat Responses schema must share the Chat query bound, not the shim's 4096"
    );
    assert!(
        lowered.get("function").is_none(),
        "Responses lowering must stay flat, not nest under `function`"
    );

    // The dispatcher and response normalizer keep reading the hosted config.
    assert_eq!(state.tools[0]["type"], json!("file_search"));
    assert_eq!(state.tools[0]["vector_store_ids"], json!(["vs_123"]));
    assert!(
        state.request_body_requires_rebuild(),
        "the outbound body must be re-serialized after lowering"
    );
}

#[test]
fn native_lowering_preserves_hosted_configuration_in_state() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{
            "type": "file_search",
            "vector_store_ids": ["vs_a", "vs_b"],
            "max_num_results": 7,
            "ranking_options": {"score_threshold": 0.5},
            "filters": {"type": "eq", "key": "k", "value": "v"}
        }]
    }));

    lower_native_file_search(&mut state).expect("valid hosted file_search lowers");

    let hosted = &state.tools[0];
    assert_eq!(hosted["vector_store_ids"], json!(["vs_a", "vs_b"]));
    assert_eq!(hosted["max_num_results"], json!(7));
    assert_eq!(hosted["ranking_options"], json!({"score_threshold": 0.5}));
    assert_eq!(hosted["filters"], json!({"type": "eq", "key": "k", "value": "v"}));
    // The private outbound tool exposes none of the hosted configuration.
    assert_eq!(state.request_body["tools"][0]["type"], json!("function"));
    assert!(
        state.request_body["tools"][0].get("vector_store_ids").is_none(),
        "the lowered private function must not leak the hosted vector_store_ids to the backend"
    );
}

#[test]
fn native_lowering_converts_forced_file_search_tool_choice() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "tool_choice": {"type": "file_search"}
    }));

    lower_native_file_search(&mut state).expect("forced file_search choice lowers");

    assert_eq!(
        state.request_body["tool_choice"],
        json!({"type": "function", "name": "file_search"})
    );
    // The hosted choice is retained for the client-visible view.
    assert_eq!(state.tool_choice, json!({"type": "file_search"}));
}

#[test]
fn native_lowering_preserves_string_tool_choice() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "tool_choice": "auto"
    }));

    lower_native_file_search(&mut state).expect("auto choice lowers tools only");

    assert_eq!(state.request_body["tool_choice"], json!("auto"));
    assert_eq!(state.request_body["tools"][0]["type"], json!("function"));
}

#[test]
fn native_lowering_preserves_unrelated_forced_function_choice() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "function", "name": "lookup"}
        ],
        "tool_choice": {"type": "function", "name": "lookup"}
    }));

    lower_native_file_search(&mut state).expect("unrelated forced function is preserved");

    assert_eq!(
        state.request_body["tool_choice"],
        json!({"type": "function", "name": "lookup"})
    );
    assert_eq!(state.request_body["tools"][0]["name"], json!("file_search"));
    assert_eq!(
        state.request_body["tools"][1],
        json!({"type": "function", "name": "lookup"}),
        "a client function tool must survive lowering unchanged"
    );
}

#[test]
fn native_lowering_rejects_client_function_named_file_search() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "function", "name": "file_search", "parameters": {"type": "object"}}
        ]
    }));

    let err = lower_native_file_search(&mut state).expect_err("colliding client function rejects");
    let (status, message) = reject_parts(&err);
    assert_eq!(status, 400);
    assert!(
        message.contains("conflicts with the synthesized file_search function"),
        "unexpected message: {message}"
    );
}

#[test]
fn native_lowering_rejects_forced_private_function_tool_choice() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "tool_choice": {"type": "function", "name": "file_search"}
    }));

    let err = lower_native_file_search(&mut state).expect_err("forcing the private function rejects");
    let (status, message) = reject_parts(&err);
    assert_eq!(status, 400);
    assert!(
        message.contains("tool_choice for hosted file_search must use type file_search"),
        "unexpected message: {message}"
    );
    // A rejected request must not leave a half-lowered outbound body.
    assert_eq!(state.request_body["tools"][0]["type"], json!("file_search"));
    assert!(
        !state.request_body_requires_rebuild(),
        "a rejected lowering must not request an outbound rebuild"
    );
}

#[test]
fn native_lowering_rejects_multiple_file_search_tools() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "file_search", "vector_store_ids": ["vs_2"]}
        ]
    }));

    let err = lower_native_file_search(&mut state).expect_err("two file_search tools reject");
    let (status, message) = reject_parts(&err);
    assert_eq!(status, 400);
    assert!(
        message.contains("only one file_search tool may be declared"),
        "unexpected message: {message}"
    );
}

#[test]
fn native_lowering_rejects_invalid_vector_store_ids() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": []}]
    }));

    let err = lower_native_file_search(&mut state).expect_err("empty vector_store_ids reject");
    let (status, message) = reject_parts(&err);
    assert_eq!(status, 400);
    assert!(
        message.contains("vector_store_ids must be a non-empty array of non-empty strings"),
        "unexpected message: {message}"
    );
}

#[test]
fn native_lowering_is_idempotent_across_continuations() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "tool_choice": {"type": "file_search"}
    }));

    lower_native_file_search(&mut state).expect("round 0 lowers");
    let after_first = state.request_body.clone();

    lower_native_file_search(&mut state).expect("continuation is a no-op");
    assert_eq!(
        state.request_body, after_first,
        "re-lowering an already-lowered body must not change it"
    );
    // The hosted config still drives the dispatcher after continuation.
    assert_eq!(state.tools[0]["type"], json!("file_search"));
}

#[test]
fn native_lowering_noop_without_hosted_file_search() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "function", "name": "lookup"}],
        "tool_choice": {"type": "file_search"}
    }));
    let before = state.request_body.clone();

    lower_native_file_search(&mut state).expect("no hosted file_search is a no-op");

    assert_eq!(state.request_body, before, "no hosted tool means no rewrite");
    assert!(
        !state.request_body_requires_rebuild(),
        "no rewrite means no rebuild request"
    );
}

#[test]
fn native_lowering_noop_when_request_body_absent() {
    // Existing dispatch tests build state via `state_with`, leaving `request_body`
    // null. Lowering must skip those so the demoted dispatcher path is unchanged.
    let mut state = state_with(&["vs_1"], Vec::new());
    assert!(state.request_body.is_null(), "state_with must leave request_body null");

    lower_native_file_search(&mut state).expect("absent request body is a no-op");

    assert!(
        state.request_body.is_null(),
        "lowering a null request_body must leave it null"
    );
    assert!(
        !state.request_body_requires_rebuild(),
        "a no-op lowering must not request an outbound rebuild"
    );
}

#[test]
fn native_lowering_preserves_tool_order_around_file_search() {
    let mut state = ResponsesState::from_request_body(json!({
        "tools": [
            {"type": "function", "name": "alpha"},
            {"type": "file_search", "vector_store_ids": ["vs_1"]},
            {"type": "function", "name": "beta"}
        ]
    }));

    lower_native_file_search(&mut state).expect("mixed tools lower");

    let tools = state.request_body["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 3);
    assert_eq!(tools[0], json!({"type": "function", "name": "alpha"}));
    assert_eq!(tools[1]["name"], json!("file_search"));
    assert_eq!(tools[1]["type"], json!("function"));
    assert!(
        tools[1].get("vector_store_ids").is_none(),
        "the lowered middle tool must not leak the hosted vector_store_ids"
    );
    assert_eq!(tools[2], json!({"type": "function", "name": "beta"}));
}

#[tokio::test]
async fn on_request_body_lowers_native_file_search_before_dispatch() {
    // No assignments are recorded, so `dispatch` is a no-op `Continue`; the only
    // observable effect is the request-side lowering wired ahead of dispatch.
    let server = MockServer::json(200, &json!({"data": []}));
    let filter = make_filter(server.port, "");
    let state = ResponsesState::from_request_body(json!({
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
        "tool_choice": {"type": "file_search"}
    }));
    let mut ctx = make_context(Some(state));

    let action = dispatch(&*filter, &mut ctx).await;
    assert!(
        matches!(action, FilterAction::Continue),
        "no assignments recorded means dispatch continues without a callout"
    );

    let state = ctx.extensions.get::<ResponsesState>().expect("state present");
    assert_eq!(state.request_body["tools"][0]["type"], json!("function"));
    assert_eq!(
        state.request_body["tool_choice"],
        json!({"type": "function", "name": "file_search"})
    );
    assert_eq!(state.tools[0]["type"], json!("file_search"));
    assert!(
        server.requests().is_empty(),
        "no assignments means no vector-store callout"
    );
}
