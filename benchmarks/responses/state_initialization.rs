// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Compare Responses state initialization and store-input capture before and after #1618.
//!
//! Run with:
//!
//! ```console
//! cargo bench -p praxis-ai-benchmarks --features responses-state --bench responses_state
//! ```
#![expect(
    clippy::expect_used,
    reason = "the benchmark builds deterministic fixed fixtures and aborts on invalid setup"
)]
#![expect(
    clippy::print_stdout,
    reason = "allocation totals are printed next to Criterion timing results"
)]

use std::{hint::black_box, time::Duration};

use allocation_counter::measure;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use praxis_ai_apis::openai::responses_state_benchmark;
use serde_json::{Value, json};

/// Parsed request and stored history for one benchmark scenario.
struct Fixture {
    /// Criterion case label.
    name: &'static str,
    /// Already-parsed request body passed to state construction.
    body: Value,
    /// Prior response history prepended during rehydration.
    stored: Vec<Value>,
    /// Serialized copy used to reproduce the legacy store reparse.
    raw_body: Vec<u8>,
}

/// Build the large-text and many-small-items scenarios.
fn fixtures() -> [Fixture; 2] {
    [
        fixture(
            "large_text",
            json!({"model": "gpt-4o", "previous_response_id": "resp_previous", "input": "x".repeat(200_000)}),
        ),
        fixture(
            "many_small_messages",
            json!({
                "model": "gpt-4o",
                "previous_response_id": "resp_previous",
                "input": (0..2_000)
                    .map(|index| json!({"type": "message", "role": "user", "content": format!("message-{index}")}))
                    .collect::<Vec<_>>(),
            }),
        ),
    ]
}

/// Prepare request bytes and the same prior history for either implementation.
fn fixture(name: &'static str, body: Value) -> Fixture {
    let raw_body = serde_json::to_vec(&body).expect("serialize request fixture");
    Fixture {
        name,
        body,
        stored: vec![json!({"type": "message", "role": "assistant", "content": "previous"})],
        raw_body,
    }
}

/// Print a single-run allocation comparison outside Criterion's timed loop.
fn print_allocation_comparison(fixture: &Fixture) {
    let old_body = fixture.body.clone();
    let old_stored = fixture.stored.clone();
    let legacy = measure(|| {
        responses_state_benchmark::legacy(old_body, old_stored, &fixture.raw_body)
            .expect("benchmark request JSON is valid");
    });

    let current_body = fixture.body.clone();
    let current_stored = fixture.stored.clone();
    let current = measure(|| responses_state_benchmark::current(current_body, current_stored));

    println!(
        "{} allocations: legacy={} bytes/{} allocs, current={} bytes/{} allocs",
        fixture.name, legacy.bytes_total, legacy.count_total, current.bytes_total, current.count_total
    );
}

/// Benchmark the modeled previous path and current rehydration path.
fn benchmark_responses_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("responses_state_initialization_and_store_capture");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for fixture in fixtures() {
        print_allocation_comparison(&fixture);

        group.bench_function(BenchmarkId::new("legacy", fixture.name), |bencher| {
            bencher.iter_batched(
                || (fixture.body.clone(), fixture.stored.clone()),
                |(body, stored)| {
                    responses_state_benchmark::legacy(black_box(body), black_box(stored), black_box(&fixture.raw_body))
                        .expect("benchmark request JSON is valid");
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("current", fixture.name), |bencher| {
            bencher.iter_batched(
                || (fixture.body.clone(), fixture.stored.clone()),
                |(body, stored)| {
                    responses_state_benchmark::current(black_box(body), black_box(stored));
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_responses_state);
criterion_main!(benches);
