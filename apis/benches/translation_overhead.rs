// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Focused request-translation benchmarks with allocation evidence.
//!
//! Run with:
//!
//! ```console
//! cargo bench -p praxis-ai-apis --bench translation_overhead
//! ```
//!
//! Each benchmark first prints one `allocation_report,<label>,count=…,bytes=…`
//! line to stderr (via `allocation_counter::measure`) so allocation counts
//! can be compared across translator changes, then reports wall time.

#![expect(
    clippy::print_stderr,
    reason = "allocation reports are consumed from benchmark stderr"
)]
#![expect(clippy::unwrap_used, reason = "benchmark fixtures are valid by construction")]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

#[cfg_attr(
    test,
    expect(
        dead_code,
        clippy::missing_docs_in_private_items,
        reason = "bench target includes the cfg(test) module whose fixtures are dead in a benchmark harness"
    )
)]
#[path = "../src/bedrock/converse/request.rs"]
mod bedrock_request;

#[cfg_attr(
    test,
    expect(
        dead_code,
        clippy::missing_docs_in_private_items,
        reason = "bench target includes the cfg(test) module whose fixtures are dead in a benchmark harness"
    )
)]
// `request.rs` carries a module-wide `#[expect(clippy::indexing_slicing)]`
// for its tests. That expectation is fulfilled when the module is compiled as
// a lib test but not in a benchmark harness, and no single `#[expect]` can
// express "fulfilled in one target, not the other". Suppress the unfulfilled
// case only for this path import; do not weaken the module's own attribute.
#[expect(
    clippy::allow_attributes,
    reason = "target-dependent expectation cannot be expressed with #[expect] alone"
)]
#[allow(
    unfulfilled_lint_expectations,
    reason = "request.rs test expectations are unfulfilled only in the benchmark harness"
)]
#[path = "../src/vertex/gemini/request.rs"]
mod gemini_request;

/// Number of tools in the large-schema fixture.
const LARGE_TOOL_COUNT: usize = 4;

/// Size of each nested schema description string, in bytes.
const SCHEMA_DESCRIPTION_BYTES: usize = 16 * 1024;

/// Number of stop sequences in the large fixture.
const STOP_SEQUENCE_COUNT: usize = 16;

/// Run Bedrock request translation on an opaque benchmark input.
fn translate_bedrock(body: &[u8]) {
    let result = bedrock_request::transform_request(std::hint::black_box(body)).unwrap();
    std::hint::black_box((&result.body, &result.model, result.stream, result.include_usage));
}

/// Run Gemini request translation on an opaque benchmark input.
fn translate_gemini(body: &[u8]) {
    let result = gemini_request::transform_request(std::hint::black_box(body)).unwrap();
    std::hint::black_box((
        &result.body,
        &result.model,
        result.stream,
        result.include_usage,
        result.candidate_count,
    ));
}

/// Build a Bedrock request with several large nested tool schemas and a stop
/// array, the shapes whose ownership the translator moves.
fn bedrock_large_tool_request() -> Vec<u8> {
    let description = "s".repeat(SCHEMA_DESCRIPTION_BYTES);
    let tools: Vec<Value> = (0..LARGE_TOOL_COUNT)
        .map(|index| {
            json!({
                "type": "function",
                "function": {
                    "name": format!("tool_{index}"),
                    "description": "large schema fixture",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "string", "description": description}
                        }
                    }
                }
            })
        })
        .collect();
    let stops: Vec<String> = (0..STOP_SEQUENCE_COUNT)
        .map(|index| format!("stop_{index}_{}", "x".repeat(256)))
        .collect();

    serde_json::to_vec(&json!({
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "messages": [{"role": "user", "content": "translate"}],
        "stop": stops,
        "tools": tools
    }))
    .unwrap()
}

/// Build a Gemini request with matched tool-call and tool-result messages.
fn gemini_call_result_request(call_result_pairs: usize) -> Vec<u8> {
    let mut calls = Vec::with_capacity(call_result_pairs);
    let mut results = Vec::with_capacity(call_result_pairs);
    for index in 0..call_result_pairs {
        let id = format!("call_{index}");
        let name = format!("tool_{index}");
        calls.push(json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"}
        }));
        results.push(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": "{}"
        }));
    }

    let mut messages = Vec::with_capacity(call_result_pairs + 2);
    messages.push(json!({"role": "user", "content": "run all tools"}));
    messages.push(json!({"role": "assistant", "content": null, "tool_calls": calls}));
    messages.extend(results);
    serde_json::to_vec(&json!({"model": "gemini-1.5-pro", "messages": messages})).unwrap()
}

/// Build a Gemini request with large image, schema, stop, and result fields.
fn gemini_large_request() -> Vec<u8> {
    let image_data = format!("data:image/png;base64,{}", "A".repeat(512 * 1024));
    let schema_description = "d".repeat(16 * 1024);
    let stop_sequences: Vec<String> = (0..16)
        .map(|index| format!("stop_{index}_{}", "x".repeat(256)))
        .collect();

    serde_json::to_vec(&json!({
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data}}]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_lookup", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": "signature"}}}
            ]},
            {"role": "tool", "tool_call_id": "call_lookup", "content": "x".repeat(32 * 1024)}
        ],
        "stop": stop_sequences,
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "type": "object",
            "properties": {"result": {"type": "string", "description": schema_description}}
        }}},
        "tools": [{"type": "function", "function": {
            "name": "lookup",
            "description": "large tool schema",
            "parameters": {"type": "object", "properties": {
                "payload": {"type": "string", "description": schema_description}
            }}
        }}]
    }))
    .unwrap()
}

/// Isolate the systemInstruction wrapping boundary: one 512 KiB system text
/// whose single copy decides whether the wrapper serialized it again.
fn gemini_large_system_request() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "system", "content": "s".repeat(512 * 1024)},
            {"role": "user", "content": "go"}
        ]
    }))
    .unwrap()
}

/// Isolate the parsed-argument boundary: `arguments` arrives as a JSON string
/// that must be parsed, then moved into `functionCall.args` rather than copied.
fn gemini_large_arguments_request() -> Vec<u8> {
    let arguments = json!({"payload": "a".repeat(512 * 1024)}).to_string();
    serde_json::to_vec(&json!({
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "user", "content": "call it"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_big", "type": "function",
                 "function": {"name": "lookup", "arguments": arguments}}
            ]},
            {"role": "tool", "tool_call_id": "call_big", "content": "{}"}
        ]
    }))
    .unwrap()
}

/// Isolate the nested-signature boundary: a 512 KiB object under
/// `extra_content.google.thought_signature` that must move onto the Part.
fn gemini_large_signature_request() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "gemini-1.5-pro",
        "messages": [
            {"role": "user", "content": "call it"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_sig", "type": "function",
                 "function": {"name": "lookup", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": {"nested": "g".repeat(512 * 1024)}}}}
            ]},
            {"role": "tool", "name": "lookup", "content": "{}"}
        ]
    }))
    .unwrap()
}

/// Report allocator activity for one translation fixture.
fn report_allocation(label: &str, translation: impl FnOnce()) {
    let allocations = allocation_counter::measure(translation);
    eprintln!(
        "allocation_report,{label},count={},bytes={}",
        allocations.count_total, allocations.bytes_total
    );
}

/// Benchmark Bedrock translation for several large tool schemas.
fn bedrock_large_tools(c: &mut Criterion) {
    let body = bedrock_large_tool_request();
    // Warm up once outside the measurement so one-time lazy state (the
    // compiled model-ID regex) is excluded from the allocation evidence.
    translate_bedrock(&body);
    report_allocation("bedrock_large_tools", || translate_bedrock(&body));

    let mut group = c.benchmark_group("bedrock_large_tools");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("four_16k_schemas_and_stop_array", |b| {
        b.iter(|| translate_bedrock(&body));
    });
    group.finish();
}

/// Benchmark Gemini translation as paired tool calls and results scale.
fn gemini_lookup_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemini_lookup_scaling");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for call_result_pairs in [8, 16, 32, 64, 128, 256] {
        let body = gemini_call_result_request(call_result_pairs);
        // Warm up lazy regex state.
        translate_gemini(&body);
        report_allocation(&format!("gemini_lookup_pairs_{call_result_pairs}"), || {
            translate_gemini(&body);
        });
        group.bench_with_input(BenchmarkId::from_parameter(call_result_pairs), &body, |b, body| {
            b.iter(|| translate_gemini(body));
        });
    }

    group.finish();
}

/// Benchmark Gemini translation with large multimodal and schema fields.
fn gemini_large_fields(c: &mut Criterion) {
    let body = gemini_large_request();
    translate_gemini(&body);
    report_allocation("gemini_large_fields", || translate_gemini(&body));

    let mut group = c.benchmark_group("gemini_large_fields");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("512k_image_schema_stops_and_tool_result", |b| {
        b.iter(|| translate_gemini(&body));
    });
    group.finish();
}

/// Benchmark the ownership boundaries the issue names: system text, parsed
/// tool-call arguments, and nested thought signatures.
fn gemini_owned_boundaries(c: &mut Criterion) {
    let fixtures = [
        ("gemini_large_system", gemini_large_system_request()),
        ("gemini_large_arguments", gemini_large_arguments_request()),
        ("gemini_large_signature", gemini_large_signature_request()),
    ];
    let mut group = c.benchmark_group("gemini_owned_boundaries");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for (label, body) in &fixtures {
        translate_gemini(body);
        report_allocation(label, || translate_gemini(body));
        group.bench_with_input(BenchmarkId::from_parameter(label), body, |b, body| {
            b.iter(|| translate_gemini(body));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bedrock_large_tools, gemini_lookup_scaling, gemini_large_fields,
        gemini_owned_boundaries
}
criterion_main!(benches);
