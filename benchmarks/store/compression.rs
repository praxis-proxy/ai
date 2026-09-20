// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Criterion benchmark for the responses-store data compression.
//!
//! Run with:
//!
//! ```console
//! cargo bench -p praxis-ai-benchmarks
//! ```
#![expect(
    missing_docs,
    reason = "criterion_group!/criterion_main! expand to undocumented items"
)]
#![expect(
    clippy::expect_used,
    reason = "benches build fixed inputs up front; a panic just aborts the run"
)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use praxis_ai_apis::store::{CompressionAlgorithm, StoreCompressionConfig, decode};
use serde_json::{Value, json};

/// A synthetic `response_object` column: model output with some repetition
/// (assistant prose compresses well) plus less-compressible ids/usage.
fn sample_response_object(paragraphs: usize) -> Value {
    let body = "The quick brown fox jumps over the lazy dog. ".repeat(paragraphs * 12);
    json!({
        "id": "resp_0123456789abcdef0123456789abcdef",
        "object": "response",
        "status": "completed",
        "model": "google/gemma-4-12B-it",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": body}],
        }],
        "usage": {"input_tokens": 128, "output_tokens": 512, "total_tokens": 640},
    })
}

/// A synthetic `input` column: a multi-turn user/assistant transcript.
fn sample_input(turns: usize) -> Value {
    let messages: Vec<Value> = (0..turns)
        .map(|i| {
            json!({
                "type": "message",
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": [{
                    "type": "input_text",
                    "text": format!(
                        "Turn {i}: please summarize the preceding context and \
                         continue the plan in detail. {}",
                        "context ".repeat(24)
                    ),
                }],
            })
        })
        .collect();
    Value::Array(messages)
}

/// Build the three columns for one stored response at a given scale.
fn response_columns(scale: usize) -> [Value; 3] {
    [
        sample_response_object(scale),
        sample_input(scale * 2),
        sample_input(scale * 2),
    ]
}

/// zstd config at the default level, the compressed path under test.
fn zstd_config() -> StoreCompressionConfig {
    StoreCompressionConfig {
        algorithm: CompressionAlgorithm::Zstd,
        level: None,
    }
}

/// The uncompressed baseline: raw JSON bytes, no zstd.
fn none_config() -> StoreCompressionConfig {
    StoreCompressionConfig::default()
}

/// Serialized JSON byte length, used as Criterion throughput so results read as
/// MB/s over the uncompressed payload.
fn json_len(value: &Value) -> u64 {
    serde_json::to_vec(value).expect("sample serializes").len() as u64
}

/// Encode throughput for a single `response_object` column, zstd vs none.
fn bench_encode(c: &mut Criterion) {
    let zstd = zstd_config();
    let none = none_config();
    let mut group = c.benchmark_group("encode_response_object");
    for scale in [1_usize, 8, 64] {
        let value = sample_response_object(scale);
        group.throughput(Throughput::Bytes(json_len(&value)));
        group.bench_with_input(BenchmarkId::new("zstd", scale), &value, |b, v| {
            b.iter(|| zstd.encode(black_box(v)).expect("encode"));
        });
        group.bench_with_input(BenchmarkId::new("none", scale), &value, |b, v| {
            b.iter(|| none.encode(black_box(v)).expect("encode"));
        });
    }
    group.finish();
}

/// Decode throughput for a single `response_object` column, zstd vs none.
fn bench_decode(c: &mut Criterion) {
    let zstd = zstd_config();
    let mut group = c.benchmark_group("decode_response_object");
    for scale in [1_usize, 8, 64] {
        let value = sample_response_object(scale);
        let compressed = zstd.encode(&value).expect("encode");
        let plain = serde_json::to_vec(&value).expect("serialize");
        group.throughput(Throughput::Bytes(json_len(&value)));
        group.bench_with_input(BenchmarkId::new("zstd", scale), &compressed, |b, stored| {
            b.iter(|| decode(black_box(stored)).expect("decode"));
        });
        group.bench_with_input(BenchmarkId::new("none", scale), &plain, |b, stored| {
            b.iter(|| decode(black_box(stored)).expect("decode"));
        });
    }
    group.finish();
}

/// The accumulated cost of one stored response: all three columns are
/// compressed independently on the write path, so this is the figure that
/// maps to per-write CPU.
fn bench_response_write(c: &mut Criterion) {
    let zstd = zstd_config();
    let mut group = c.benchmark_group("encode_response_write");
    for scale in [1_usize, 8, 64] {
        let columns = response_columns(scale);
        let total: u64 = columns.iter().map(json_len).sum();
        group.throughput(Throughput::Bytes(total));
        group.bench_with_input(BenchmarkId::new("zstd_3_columns", scale), &columns, |b, cols| {
            b.iter(|| {
                for col in cols {
                    black_box(zstd.encode(black_box(col)).expect("encode"));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_response_write);
criterion_main!(benches);
