// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Gemini request-translation scaling over parallel tool-call/result pairs.
//!
//! Each scale translates one request containing a single assistant message
//! with K `tool_calls` followed by K `tool` results — the parallel-batch
//! worst case from praxis-proxy/ai#1517, where per-result name lookup
//! rescanned history and the tool-call array (O(K^2)).
//!
//! `by_tool_call_id` resolves result names through the request-local lookup
//! index; `explicit_names` carries the name on every result and bypasses
//! lookup, matching the issue's probe baseline. The two variants must stay
//! within noise and scale near-linearly in K.
//!
//! Run with:
//!
//! ```console
//! cargo bench -p praxis-ai-benchmarks --bench gemini_translation
//! ```

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde_json::json;

// The translator is `pub(crate)` in `praxis-ai-apis`, so the bench compiles
// the self-contained module directly. The attribute stack mirrors the one
// validated for path-imported translator modules: bench targets compile with
// `cfg(test)`, so the module's `#[cfg(test)]` test code is included and its
// fixtures surface as dead code, unused imports, and undocumented items in a
// harness that never calls them.
#[expect(
    clippy::allow_attributes,
    reason = "an `#[allow]` is required for compilation modes where the module-level expectations do not fire"
)]
#[allow(
    unfulfilled_lint_expectations,
    reason = "path-imported expectations only fire in compilations that include the cfg(test) module"
)]
#[expect(
    dead_code,
    unused_imports,
    clippy::missing_docs_in_private_items,
    reason = "bench target includes the cfg(test) module whose fixtures are dead in a benchmark harness"
)]
#[path = "../../apis/src/vertex/gemini/request.rs"]
mod gemini_request;

/// Parallel tool-call/result pair counts exercised by the scaling group.
const PAIRS: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1000];

/// Run Gemini request translation on an opaque benchmark input.
#[expect(clippy::expect_used, reason = "benchmark fixtures are built from valid JSON")]
fn translate_gemini(body: &[u8]) {
    let result = gemini_request::transform_request(black_box(body)).expect("fixture translates");
    black_box((
        &result.body,
        &result.model,
        result.stream,
        result.include_usage,
        result.candidate_count,
    ));
}

/// Build a Gemini request with one assistant message carrying `pairs` tool
/// calls followed by `pairs` tool results.
///
/// When `explicit_names` is set, every result carries its function `name`
/// directly (the issue probe's bypass baseline); otherwise each result
/// references its call by `tool_call_id` and exercises name resolution.
#[expect(clippy::expect_used, reason = "benchmark fixture serializes valid JSON")]
fn gemini_call_result_request(pairs: usize, explicit_names: bool) -> Vec<u8> {
    let mut calls = Vec::with_capacity(pairs);
    let mut results = Vec::with_capacity(pairs);
    for index in 0..pairs {
        let id = format!("call_{index}");
        let name = format!("tool_{index}");
        calls.push(json!({
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"}
        }));
        if explicit_names {
            results.push(json!({
                "role": "tool",
                "name": format!("tool_{index}"),
                "content": "{}"
            }));
        } else {
            results.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": "{}"
            }));
        }
    }

    let mut messages = Vec::with_capacity(pairs + 2);
    messages.push(json!({"role": "user", "content": "run all tools"}));
    messages.push(json!({"role": "assistant", "content": null, "tool_calls": calls}));
    messages.extend(results);
    serde_json::to_vec(&json!({"model": "gemini-1.5-pro", "messages": messages})).expect("fixture serializes")
}

/// Benchmark Gemini translation as paired tool calls and results scale.
fn gemini_tool_call_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemini_tool_call_lookup");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for pairs in PAIRS {
        group.throughput(Throughput::Elements(pairs as u64));
        let by_id = gemini_call_result_request(pairs, false);
        group.bench_with_input(BenchmarkId::new("by_tool_call_id", pairs), &by_id, |b, body| {
            b.iter(|| translate_gemini(body));
        });
        let by_name = gemini_call_result_request(pairs, true);
        group.bench_with_input(BenchmarkId::new("explicit_names", pairs), &by_name, |b, body| {
            b.iter(|| translate_gemini(body));
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = gemini_tool_call_lookup
}
criterion_main!(benches);
