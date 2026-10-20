// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bedrock Converse request-translation benchmark with allocation evidence.
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

use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

/// Bedrock Converse request translator, path-imported from the crate source
/// because `transform_request` is `pub(crate)` and benchmark targets are
/// separate crates. The workspace's `praxis-ai-benchmarks` crate is not used
/// because it only reaches the public API; the path import keeps the private
/// function benchmarkable without widening visibility, at the cost of
/// compiling `request.rs` a second time.
///
/// Cargo compiles `harness = false` bench targets with `--cfg test` (verified
/// via `cargo bench --no-run -v`), so the module's `#[cfg(test)]` test code is
/// included and its fixtures surface as dead code in a harness that never
/// calls them. Apply the expectations only in that mode so compilations
/// without the test module do not produce unfulfilled expectations.
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

/// Report allocator activity for one translation fixture.
fn report_allocation(label: &str, translation: impl FnOnce()) {
    let allocations = allocation_counter::measure(translation);
    eprintln!(
        "allocation_report,{label},count={},bytes={}",
        allocations.count_total, allocations.bytes_total
    );
}

/// Benchmark Bedrock translation for several large tool schemas.
///
/// Wall time is measured with `allocation-counter`'s counting global
/// allocator installed for the whole bench process, which inflates it
/// relative to a production build. Compare times only within this bench;
/// the allocation report is the trustworthy cross-build metric.
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

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bedrock_large_tools
}
criterion_main!(benches);
