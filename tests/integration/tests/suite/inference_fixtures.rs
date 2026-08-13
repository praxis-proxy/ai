// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Data-driven replay for committed inference transformation fixtures.

use std::{collections::BTreeSet, path::PathBuf};

use praxis_test_utils::inference_fixture::{
    RecordedBody, ScenarioRunner, WireFixture, check_coverage, discover_recordings, discover_scenarios,
};
use serde_json::json;

const STREAM_SCENARIO: &str = "messages/basic-stream";
const STREAM_PROVIDER: &str = "openai";
const ERROR_SCENARIO: &str = "messages/upstream-error";
const ERROR_PROVIDER: &str = "synthetic";
const NATIVE_ANTHROPIC_PROVIDER: &str = "anthropic";
const NATIVE_ANTHROPIC_SCENARIOS: [&str; 3] = [
    "messages/native-basic-nonstream",
    "messages/native-basic-stream",
    "messages/native-tool-use",
];
const NATIVE_TOOL_USE_SCENARIO: &str = "messages/native-tool-use";
const NATIVE_RESPONSES_PROVIDERS: [&str; 2] = ["openai", "vllm"];
const NATIVE_RESPONSES_SCENARIOS: [&str; 3] = [
    "responses/native-basic-nonstream",
    "responses/native-basic-stream",
    "responses/native-tool-call",
];
const NATIVE_RESPONSES_TEXT_SCENARIO: &str = "responses/native-basic-nonstream";
const NATIVE_RESPONSES_STREAM_SCENARIO: &str = "responses/native-basic-stream";
const NATIVE_RESPONSES_TOOL_SCENARIO: &str = "responses/native-tool-call";
const AGENTIC_PARALLEL_TOOL_CALLS_SCENARIO: &str = "responses/agentic-parallel-tool-calls";
const AGENTIC_PARALLEL_TOOL_CALLS_PROVIDER: &str = "synthetic";

#[tokio::test]
async fn all_inference_fixtures_replay() {
    let root = fixture_root();
    check_coverage(&root).unwrap_or_else(|error| {
        panic!("inference fixture coverage must be complete: {error}");
    });
    let scenarios = discover_scenarios(&root).unwrap_or_else(|error| {
        panic!("inference scenario discovery must succeed: {error}");
    });
    let recordings = discover_recordings(&root).unwrap_or_else(|error| {
        panic!("inference recording discovery must succeed: {error}");
    });
    let mut recorded_scenarios = BTreeSet::new();
    let mut native_anthropic_scenarios = BTreeSet::new();
    let mut native_responses_recordings = BTreeSet::new();
    let mut saw_stream_representative = false;
    let mut saw_error_representative = false;
    let mut saw_agentic_parallel_tool_calls = false;

    for recording in recordings {
        let scenario_id = recording.scenario_id.as_str();
        let provider = recording.provider.as_str();
        let scenario_path = scenarios.get(scenario_id).unwrap_or_else(|| {
            panic!("recording for scenario `{scenario_id}` and provider `{provider}` has no declared scenario");
        });
        let scenario =
            praxis_test_utils::inference_fixture::InferenceScenario::load(scenario_path).unwrap_or_else(|error| {
                panic!("scenario `{scenario_id}` for provider `{provider}` must load: {error}");
            });
        let expected = WireFixture::load(&recording.path).unwrap_or_else(|error| {
            panic!("recording for scenario `{scenario_id}` and provider `{provider}` must load: {error}");
        });
        assert_eq!(
            expected.scenario_id, scenario.id,
            "recording scenario identity mismatch for scenario `{scenario_id}` and provider `{provider}`"
        );
        assert_eq!(
            expected.provenance.provider, provider,
            "recording provider identity mismatch for scenario `{scenario_id}` and provider `{provider}`"
        );

        let report = ScenarioRunner::replay(&scenario, &expected)
            .await
            .unwrap_or_else(|error| {
                panic!("replay failed for scenario `{scenario_id}` and provider `{provider}`: {error}");
            });

        if scenario_id == STREAM_SCENARIO && provider == STREAM_PROVIDER {
            assert_stream_event_order(&report.actual, scenario_id, provider);
            saw_stream_representative = true;
        }
        if scenario_id == ERROR_SCENARIO && provider == ERROR_PROVIDER {
            assert_synthetic_rate_limit(&report.actual, scenario_id, provider);
            saw_error_representative = true;
        }
        if provider == NATIVE_ANTHROPIC_PROVIDER && NATIVE_ANTHROPIC_SCENARIOS.contains(&scenario_id) {
            native_anthropic_scenarios.insert(scenario_id.to_owned());
            if scenario_id == NATIVE_TOOL_USE_SCENARIO {
                assert_native_tool_use(&report.actual, scenario_id, provider);
            }
        }
        if scenario_id == AGENTIC_PARALLEL_TOOL_CALLS_SCENARIO && provider == AGENTIC_PARALLEL_TOOL_CALLS_PROVIDER {
            assert_agentic_parallel_tool_calls(&report.actual, scenario_id, provider);
            saw_agentic_parallel_tool_calls = true;
        }
        if NATIVE_RESPONSES_PROVIDERS.contains(&provider) && NATIVE_RESPONSES_SCENARIOS.contains(&scenario_id) {
            native_responses_recordings.insert((scenario_id.to_owned(), provider.to_owned()));
            assert_native_responses_passthrough(&report.actual, scenario_id, provider);
            match scenario_id {
                NATIVE_RESPONSES_TEXT_SCENARIO => assert_native_responses_text(&report.actual, scenario_id, provider),
                NATIVE_RESPONSES_STREAM_SCENARIO => {
                    assert_native_responses_stream(&report.actual, scenario_id, provider);
                },
                NATIVE_RESPONSES_TOOL_SCENARIO => {
                    assert_native_responses_tool_call(&report.actual, scenario_id, provider);
                },
                _ => unreachable!("the scenario was checked against NATIVE_RESPONSES_SCENARIOS"),
            }
        }
        recorded_scenarios.insert(recording.scenario_id);
    }

    for scenario_id in scenarios.keys() {
        assert!(
            recorded_scenarios.contains(scenario_id),
            "declared scenario `{scenario_id}` has no discovered recording"
        );
    }
    assert!(
        saw_stream_representative,
        "missing representative recording for scenario `{STREAM_SCENARIO}` and provider `{STREAM_PROVIDER}`"
    );
    assert!(
        saw_error_representative,
        "missing representative recording for scenario `{ERROR_SCENARIO}` and provider `{ERROR_PROVIDER}`"
    );
    assert!(
        saw_agentic_parallel_tool_calls,
        "missing representative recording for scenario `{AGENTIC_PARALLEL_TOOL_CALLS_SCENARIO}` and provider `{AGENTIC_PARALLEL_TOOL_CALLS_PROVIDER}`"
    );
    for scenario_id in NATIVE_ANTHROPIC_SCENARIOS {
        assert!(
            native_anthropic_scenarios.contains(scenario_id),
            "missing native recording for scenario `{scenario_id}` and provider `{NATIVE_ANTHROPIC_PROVIDER}`"
        );
    }
    for scenario_id in NATIVE_RESPONSES_SCENARIOS {
        for provider in NATIVE_RESPONSES_PROVIDERS {
            assert!(
                native_responses_recordings
                    .iter()
                    .any(|(recorded_scenario, recorded_provider)| {
                        recorded_scenario == scenario_id && recorded_provider == provider
                    }),
                "missing native recording for scenario `{scenario_id}` and provider `{provider}`"
            );
        }
    }
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/inference")
}

fn assert_stream_event_order(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Sse { frames, done } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a client SSE body");
    };
    assert!(
        !done,
        "scenario `{scenario_id}` and provider `{provider}` must not add an Anthropic DONE marker"
    );
    assert_eq!(
        frames.iter().map(|frame| frame.event.as_deref()).collect::<Vec<_>>(),
        [
            Some("message_start"),
            Some("content_block_start"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_delta"),
            Some("content_block_stop"),
            Some("message_delta"),
            Some("message_stop"),
        ],
        "client SSE event order changed for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_synthetic_rate_limit(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let upstream_expected = json!({"error": {"message": "slow down", "type": "rate_limit_error"}});
    let client_expected = json!({
        "type": "error",
        "error": {"message": "slow down", "type": "rate_limit_error"},
        "request_id": null
    });
    assert_eq!(
        turn.upstream.response.status, 429,
        "upstream status changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    assert_eq!(
        turn.client.response.status, 429,
        "client status changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    let RecordedBody::Json { value: upstream } = &turn.upstream.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay an upstream JSON error");
    };
    let RecordedBody::Json { value: client } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a client JSON error");
    };
    assert_eq!(
        upstream, &upstream_expected,
        "upstream 429 body changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    assert_eq!(
        client, &client_expected,
        "client 429 body changed for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_native_tool_use(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Json { value } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a client JSON body");
    };
    let content = value["content"].as_array().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay Anthropic content blocks");
    });
    let tool_use = content
        .iter()
        .find(|block| block["type"] == "tool_use")
        .unwrap_or_else(|| {
            panic!("scenario `{scenario_id}` and provider `{provider}` must replay a tool_use content block");
        });
    assert_eq!(
        tool_use["name"], "get_weather",
        "tool name changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    assert_eq!(
        value["stop_reason"], "tool_use",
        "stop reason changed for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_native_responses_passthrough(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    assert_eq!(
        turn.client.request.path, "/v1/responses",
        "client path changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    assert_eq!(
        turn.upstream.request.path, "/v1/responses",
        "upstream path changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    assert_eq!(
        turn.client.request.body, turn.upstream.request.body,
        "native Responses request changed for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_native_responses_text(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Json { value } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a JSON Responses body");
    };
    assert_eq!(
        value["object"], "response",
        "response object changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    let output = value["output"].as_array().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` must contain Responses output items");
    });
    assert!(
        output.iter().any(|item| {
            item["type"] == "message"
                && item["content"].as_array().is_some_and(|content| {
                    content
                        .iter()
                        .any(|part| part["type"] == "output_text" && part["text"].is_string())
                })
        }),
        "scenario `{scenario_id}` and provider `{provider}` must contain output text"
    );
}

fn assert_native_responses_stream(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Sse { frames, done } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a Responses SSE body");
    };
    assert!(!done, "Responses streams must not add a `[DONE]` marker");
    let mut compact = Vec::new();
    for event in frames.iter().filter_map(|frame| frame.event.as_deref()) {
        if compact.last().copied() != Some(event) {
            compact.push(event);
        }
    }
    assert_eq!(
        compact,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ],
        "Responses SSE event order changed for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_native_responses_tool_call(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Json { value } = &turn.client.response.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a JSON Responses body");
    };
    let function_call = value["output"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["type"] == "function_call"))
        .unwrap_or_else(|| {
            panic!("scenario `{scenario_id}` and provider `{provider}` must contain a function_call output item");
        });
    assert_eq!(
        function_call["name"], "get_weather",
        "function name changed for scenario `{scenario_id}` and provider `{provider}`"
    );
    let arguments = function_call["arguments"].as_str().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` must contain string function arguments");
    });
    let decoded: serde_json::Value = serde_json::from_str(arguments).unwrap_or_else(|error| {
        panic!("scenario `{scenario_id}` and provider `{provider}` returned invalid function arguments: {error}");
    });
    assert!(
        decoded.is_object(),
        "function arguments changed shape for scenario `{scenario_id}` and provider `{provider}`"
    );
}

fn assert_agentic_parallel_tool_calls(actual: &WireFixture, scenario_id: &str, provider: &str) {
    let turn = actual.turns.first().unwrap_or_else(|| {
        panic!("scenario `{scenario_id}` and provider `{provider}` replayed without a turn");
    });
    let RecordedBody::Json { value: upstream_body } = &turn.upstream.request.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a JSON upstream request body");
    };
    assert_eq!(
        upstream_body["parallel_tool_calls"], false,
        "agentic loop must inject parallel_tool_calls=false for scenario `{scenario_id}` and provider `{provider}`"
    );
    let RecordedBody::Json { value: client_body } = &turn.client.request.body else {
        panic!("scenario `{scenario_id}` and provider `{provider}` must replay a JSON client request body");
    };
    assert!(
        client_body.get("parallel_tool_calls").is_none(),
        "client request must not contain parallel_tool_calls for scenario `{scenario_id}` and provider `{provider}`"
    );
}
