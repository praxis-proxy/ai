// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Require an explicit body-contract decision for every registered AI filter.
//!
//! Active witnesses link mutators to raw capture tests or recorded translations.
//! Bypass-only pipelines never satisfy the active mutation requirement. Other filters must have
//! an explicit non-request-mutating classification. This guards omissions; it
//! does not infer a new filter's permitted edits or prove every runtime branch.

use std::collections::BTreeSet;
#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
use std::path::Path;

use praxis_filter::{FilterFactory, FilterRegistry};
#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
use praxis_test_utils::{
    example_config_path,
    inference_fixture::{InferenceScenario, WireFixture, discover_recordings, discover_scenarios},
};
use praxis_test_utils::{filters::BodyMutatingStreamBufferFilter, test_subrequest_client};
use serde_yaml::Value;

/// Explicit classifications for filters outside request-body contract pipelines.
/// A request mutator belongs in the executed cases or replay corpus, not here.
const NON_REQUEST_MUTATORS: &[(&str, &str)] = &[
    (
        "aws_sigv4_sign",
        "signs authentication headers over the existing payload",
    ),
    ("credential_inject", "injects provider authentication headers"),
    ("ai_operation", "classifies method and path"),
    ("ai_format", "classifies body format without rewriting it"),
    ("state_owner", "establishes trusted state ownership"),
    ("model_to_header", "reads the model into a routing header"),
    ("intelligent_route", "chooses a route from policy and metadata"),
    ("credential_injection", "injects upstream authentication headers"),
    ("anthropic_messages_request", "classifies Messages requests"),
    ("anthropic_messages_protocol", "enforces protocol headers"),
    (
        "anthropic_messages_to_chat_completions_stream",
        "transforms response SSE only",
    ),
    (
        "openai_chat_completions_to_anthropic_stream",
        "transforms response SSE only",
    ),
    (
        "openai_tool_parse",
        "parses tools into canonical state without changing outbound tools",
    ),
    (
        "openai_responses_request",
        "classifies requests and initializes canonical state",
    ),
    (
        "openai_responses_store",
        "persists responses without rewriting requests",
    ),
    ("openai_stream_events", "parses response SSE only"),
    ("a2a", "reads protocol facts into metadata and headers"),
    ("callout_credentials", "captures credentials in request-scoped context"),
    (
        "external_metering",
        "reports usage through callouts without rewriting the request",
    ),
    ("identity_header_guard", "removes untrusted identity headers"),
    (
        "openai_chat_completions_request",
        "classifies the request without rewriting it",
    ),
    (
        "openai_conversations",
        "serves a local resource API rather than forwarding a mutated request",
    ),
    (
        "openai_mcp_streaming_selector",
        "selects response transport from canonical state",
    ),
    (
        "openai_responses_compact",
        "serves a local compaction API; does not rewrite the forwarded request body",
    ),
    (
        "praxis_store_readiness_gate",
        "rejects unavailable-store requests before dispatch",
    ),
    (
        "project_state_owner_headers",
        "derives ownership context from trusted headers",
    ),
    (
        "provider_route",
        "selects an upstream without rewriting the request body",
    ),
    ("time_to_first_token", "observes response timing"),
    ("token_usage_metrics", "records token usage without rewriting requests"),
    (
        "openai_chat_completions_streaming_selector",
        "selects response transport without rewriting requests",
    ),
    ("token_count", "reads input to count tokens"),
    ("token_usage_headers", "projects usage into response headers"),
    ("azure_ad", "injects upstream authentication headers"),
    ("gcp_adc", "injects upstream authentication headers"),
    (
        "http_callout",
        "reads input for policy callouts and publishes results to metadata",
    ),
    ("token_rate_limit", "reads usage and enforces admission limits"),
    ("token_ceiling", "reads input and rejects over-budget requests"),
    ("inflight_tracker", "tracks concurrent inference requests"),
];

/// External processors own their output bytes and have separate protocol tests.
/// They cannot promise the native AI filters' JSON-preservation contract.
const EXTERNAL_PROCESSORS: &[(&str, &str)] = &[(
    "ext_proc",
    "processor-directed body replacement; covered by the llmd ext-proc suite",
)];

#[test]
fn every_registered_ai_filter_has_a_body_contract_decision() {
    let registry = praxis_ai::build_full_registry(&test_subrequest_client());
    let covered = covered_filters();
    let missing = unclassified_filters(&registry, &covered);
    assert!(
        missing.is_empty(),
        "registered AI filters lack body-contract coverage or classification: {missing:?}; \
         classify request mutators and link an active mutation witness; \
         only non-request-mutators may be explicitly exempted"
    );
}

#[test]
fn a_new_body_writer_cannot_silently_skip_the_contract_suite() {
    let mut registry = praxis_ai::build_full_registry(&test_subrequest_client());
    registry
        .register(
            "uncovered_test_body_writer",
            FilterFactory::Http(std::sync::Arc::new(|_| {
                Ok(Box::new(BodyMutatingStreamBufferFilter::default_test()))
            })),
        )
        .unwrap();
    let missing = unclassified_filters(&registry, &covered_filters());
    assert!(missing.contains("uncovered_test_body_writer"));
}

#[test]
fn result_conditions_do_not_count_as_instantiated_filter_coverage() {
    let config: Value = serde_yaml::from_str(
        "filters:\n  - filter: actual\n    on_result:\n      - filter: condition_only\n    outbound_chain:\n      filters:\n        - filter: nested\n",
    ).unwrap();
    let mut names = BTreeSet::new();
    collect_filter_names(&config, false, &mut names);
    assert_eq!(names, BTreeSet::from(["actual".to_owned(), "nested".to_owned()]));
}

/// Classifications are explicit even for observers and response-only writers.
fn unclassified_filters(registry: &FilterRegistry, covered: &BTreeSet<String>) -> BTreeSet<String> {
    let core = FilterRegistry::with_builtins();
    let core_names = core.available_filters();
    let mut exemptions = BTreeSet::new();
    for &(name, reason) in NON_REQUEST_MUTATORS.iter().chain(EXTERNAL_PROCESSORS) {
        assert!(!reason.trim().is_empty(), "{name} needs a classification reason");
        assert!(exemptions.insert(name), "duplicate classification for {name}");
    }
    registry
        .available_filters()
        .into_iter()
        .filter(|name| !core_names.contains(name) && !covered.contains(*name) && !exemptions.contains(name))
        .map(str::to_owned)
        .collect()
}

/// Audited request/body and canonical-outbound-state mutators.
const REQUEST_MUTATORS: &[&str] = &[
    "stream_usage_inject",
    "model_to_provider",
    "llmisvc_model_provider_resolver",
    "openai_responses_model_rewrite",
    "openai_chat_completions_to_azureai_chat_completions",
    "prompt_enrich",
    "ai_guardrails",
    "mcp",
    "openai_doc_extract",
    "openai_file_resolve",
    "openai_mcp_tool_resolve",
    "openai_client_tool_compat",
    "openai_file_search_dispatch",
    "openai_web_search_dispatch",
    "openai_mcp_dispatch",
    "openai_agentic_loop",
    "openai_responses_proxy",
    "openai_responses_rehydrate",
    "anthropic_web_search",
    "anthropic_messages_to_chat_completions",
    "openai_chat_completions_to_anthropic",
    "openai_chat_completions_to_bedrock_converse",
    "openai_chat_completions_to_vertexai_gemini",
    "anthropic_messages_to_vertexai_anthropic",
    "openai_responses_to_chat_completions",
];

fn covered_filters() -> BTreeSet<String> {
    REQUEST_MUTATORS.iter().map(|name| (*name).to_owned()).collect()
}

/// The CI body-contract profile must exercise every audited mutator actively.
#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
#[test]
fn every_request_mutator_has_an_active_witness() {
    let covered = active_witnesses();
    let missing = missing_active_witnesses(&covered_filters(), &covered);
    assert!(
        missing.is_empty(),
        "mutators need active witnesses, not bypass-only pipelines: {missing:?}"
    );
}

#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
fn active_witnesses() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let _ = super::same_format_mutations_change_only_owned_fields;
    for case in super::mutation_cases() {
        let before: serde_json::Value = serde_json::from_str(&case.body).unwrap();
        assert_ne!(
            before.pointer(case.changed_pointer),
            case.expected_value.as_ref(),
            "a mutation witness must require an actual change"
        );
        collect_filter_names(&serde_yaml::from_str(case.filters).unwrap(), true, &mut names);
    }
    let _ = super::enrichment_preserves_original_messages_and_tool_order;
    let _ = super::guardrails_preserve_everything_except_redacted_message_content;
    let _ = super::mcp_broker_changes_only_the_routed_tool_name;
    let _ = super::anthropic_web_search_appends_only_messages_and_retains_tool_schemas;
    names.extend(["prompt_enrich", "ai_guardrails", "mcp", "anthropic_web_search"].map(str::to_owned));
    names.extend(super::active::mutation_witnesses().into_iter().map(str::to_owned));
    let _ = super::super::inference_fixtures::all_inference_fixtures_preserve_retained_json_order;
    let _ = super::translations::every_translation_preserves_multiple_tools_and_nested_schema_order;
    for (filter, scenario_id) in [
        ("anthropic_messages_to_chat_completions", "messages/basic-nonstream"),
        (
            "openai_chat_completions_to_anthropic",
            "openai_legacy/anthropic-json-sse-error",
        ),
        ("openai_chat_completions_to_bedrock_converse", "bedrock/basic-nonstream"),
        (
            "openai_chat_completions_to_vertexai_gemini",
            "vertex/fallback-response-id",
        ),
        (
            "anthropic_messages_to_vertexai_anthropic",
            "vertex/anthropic-wire-translation",
        ),
        ("openai_responses_to_chat_completions", "responses/chat-basic-nonstream"),
    ] {
        assert_recorded_mutation(filter, scenario_id);
        names.insert(filter.to_owned());
    }
    names
}

/// A fixture pipeline alone is insufficient: a recorded request must change.
#[cfg(all(
    feature = "openai-all",
    feature = "store-sqlite",
    feature = "vertex-anthropic-filter"
))]
fn assert_recorded_mutation(filter: &str, id: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/inference");
    let scenarios = discover_scenarios(&root).unwrap();
    let scenario = InferenceScenario::load(&scenarios[id]).unwrap();
    let source = std::fs::read_to_string(example_config_path(&scenario.example_config)).unwrap();
    let mut configured = BTreeSet::new();
    collect_filter_names(&serde_yaml::from_str(&source).unwrap(), false, &mut configured);
    assert!(configured.contains(filter), "{id} must instantiate {filter}");
    assert!(
        discover_recordings(&root)
            .unwrap()
            .into_iter()
            .filter(|r| r.scenario_id == id)
            .any(|r| {
                WireFixture::load(&r.path)
                    .unwrap()
                    .turns
                    .iter()
                    .any(|turn| turn.client.request.body != turn.upstream.request.body)
            }),
        "{filter} needs a recorded active request mutation in {id}"
    );
}

/// Count entries in real filter lists, including nested steps and outbound chains.
fn collect_filter_names(value: &Value, in_filter_list: bool, names: &mut BTreeSet<String>) {
    match value {
        Value::Sequence(items) => {
            for item in items {
                collect_filter_names(item, in_filter_list, names);
            }
        },
        Value::Mapping(mapping) => {
            if in_filter_list && let Some(name) = mapping.get("filter").and_then(Value::as_str) {
                names.insert(name.to_owned());
            }
            for (key, value) in mapping {
                collect_filter_names(value, key.as_str() == Some("filters"), names);
            }
        },
        _ => {},
    }
}

/// Missing active evidence is evaluated independently from configuration presence.
fn missing_active_witnesses(required: &BTreeSet<String>, active: &BTreeSet<String>) -> BTreeSet<String> {
    required.difference(active).cloned().collect()
}

#[test]
fn bypass_pipeline_does_not_provide_an_active_witness() {
    let config: Value = serde_yaml::from_str("filters:\n  - filter: uncovered_test_body_writer\n").unwrap();
    let mut bypass_names = BTreeSet::new();
    collect_filter_names(&config, false, &mut bypass_names);
    let missing = missing_active_witnesses(&bypass_names, &BTreeSet::new());
    assert!(missing.contains("uncovered_test_body_writer"));
}
