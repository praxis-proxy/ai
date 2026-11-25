// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Add adversarial tool data even when a provider recording has no tools.

use std::path::Path;

use praxis_test_utils::inference_fixture::{
    InferenceScenario, RecordedBody, RedactionRules, ScenarioRunner, WireFixture, discover_recordings,
    discover_scenarios, sanitize_fixture,
};
use serde_json::{Value, json};

/// Independent wire expectations for each provider's function-tool envelope.
fn tools(shape: &str) -> Value {
    let schemas: Value = serde_json::from_str(
        r#"[
        {"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"}},"required":["z","a"]},
        {"type":"object","properties":{"second":{"type":"boolean"},"first":{"type":"string"}}}
    ]"#,
    )
    .unwrap();
    let declarations: Vec<_> = ["zebra", "alpha"]
        .into_iter()
        .zip(schemas.as_array().unwrap())
        .map(|(name, schema)| match shape {
            "messages" => json!({"name":name,"description":"ordered","input_schema":schema}),
            "responses" => json!({"type":"function","name":name,"description":"ordered","parameters":schema}),
            "chat" => json!({"type":"function","function":{"name":name,"description":"ordered","parameters":schema}}),
            "bedrock" => json!({"toolSpec":{"name":name,"description":"ordered","inputSchema":{"json":schema}}}),
            "gemini" => json!({"name":name,"description":"ordered","parameters":schema}),
            _ => panic!("unknown test tool envelope"),
        })
        .collect();
    if shape == "gemini" {
        json!([{"functionDeclarations":declarations}])
    } else {
        Value::Array(declarations)
    }
}

/// Six actual translator pipelines, each retaining two tools and nested schemas.
#[tokio::test]
pub(super) async fn every_translation_preserves_multiple_tools_and_nested_schema_order() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/inference");
    let scenarios = discover_scenarios(&root).unwrap();
    let recordings = discover_recordings(&root).unwrap();
    for (id, from, to) in [
        ("messages/provider-parameter-passthrough", "messages", "chat"),
        ("openai_legacy/anthropic-json-sse-error", "chat", "messages"),
        ("bedrock/basic-nonstream", "chat", "bedrock"),
        ("vertex/fallback-response-id", "chat", "gemini"),
        ("vertex/anthropic-wire-translation", "messages", "messages"),
        ("responses/chat-basic-nonstream", "responses", "chat"),
    ] {
        let mut scenario = InferenceScenario::load(&scenarios[id]).unwrap();
        let recording = recordings.iter().find(|recording| recording.scenario_id == id).unwrap();
        let mut expected = WireFixture::load(&recording.path).unwrap();
        // Use recorded responses only. The injected requests are controlled
        // local test variants, not newly claimed live-provider observations.
        scenario.turns.truncate(1);
        expected.turns.truncate(1);
        for body in [
            &mut scenario.turns[0].request.body,
            &mut expected.turns[0].client.request.body,
        ] {
            let RecordedBody::Json { value } = body else {
                panic!("translator witness needs JSON input")
            };
            value["tools"] = tools(from);
        }
        let RecordedBody::Json { value } = &mut expected.turns[0].upstream.request.body else {
            panic!("translator witness needs JSON output")
        };
        if to == "bedrock" {
            value["toolConfig"] = json!({"tools":tools(to),"toolChoice":{"auto":{}}});
        } else {
            value["tools"] = tools(to);
        }
        if from == "responses" {
            value["tool_choice"] = json!("auto");
            let RecordedBody::Json { value } = &mut expected.turns[0].client.response.body else {
                panic!("Responses witness needs a JSON response")
            };
            value["tools"] = tools(from);
            for tool in value["tools"].as_array_mut().unwrap() {
                tool["strict"] = json!(false);
            }
        }
        sanitize_fixture(&mut expected, &RedactionRules::default()).unwrap();
        ScenarioRunner::replay(&scenario, &expected)
            .await
            .unwrap_or_else(|error| panic!("tool values, order, or schema order changed for {id}: {error}"));
    }
}
