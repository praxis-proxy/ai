// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Store upsert/get latency with and without payload compression.
//!
//! Run with:
//!
//! ```console
//! cargo bench -p praxis-ai-benchmarks
//! # add the PostgreSQL backend (plaintext local database):
//! DATABASE_URL=postgres://user:pass@localhost/bench \
//!   PRAXIS_BENCH_PG_SSLMODE=disable cargo bench -p praxis-ai-benchmarks
//! ```
#![expect(
    missing_docs,
    reason = "criterion_group!/criterion_main! expand to undocumented items"
)]
#![expect(
    clippy::expect_used,
    reason = "benches build fixed inputs up front; a panic just aborts the run"
)]
#![expect(
    clippy::print_stdout,
    reason = "benchmarks print size/ratio and skip notices to stdout"
)]

use std::{
    hint::black_box,
    time::{SystemTime, UNIX_EPOCH},
};

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main, measurement::WallTime,
};
use praxis_ai_apis::{
    StateOwner,
    store::{
        CompressionAlgorithm, PgTlsConfig, PostgresResponseStore, ResponseRecord, ResponseStore, SqliteResponseStore,
        SslMode, StoreCompressionConfig,
    },
};
use serde_json::{Value, json};
use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgSslMode};

/// Payload scales exercised by every backend and compression setting.
const SCALES: [usize; 3] = [1, 8, 64];

/// Distinct payload variants cycled through the timed upsert loop. Writing the
/// same bytes every iteration lets a backend skip dirtying unchanged pages, so
/// alternating contents keeps the compression-versus-disk-write path honest.
const PAYLOAD_VARIANTS: usize = 4;

/// A tiny deterministic PRNG so payloads are varied (realistic entropy) yet
/// reproducible from run to run. Real LLM output is not one sentence repeated,
/// so a fixed-string payload would compress far better than production data.
struct Lcg {
    /// Current generator state, advanced on every draw.
    state: u64,
}

impl Lcg {
    /// Seed the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the next pseudo-random value.
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants; the high bits carry the entropy.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Pick a uniformly random element from `items` using the high bits.
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        let index = ((self.next_u64() >> 33) as usize) % items.len();
        items.get(index).copied().expect("modulo keeps the index in bounds")
    }
}

/// A modest, varied vocabulary. Enough distinct tokens that zstd achieves a
/// realistic ratio rather than the ~40x of a single repeated sentence.
const WORD_BANK: &[&str] = &[
    "model",
    "context",
    "request",
    "response",
    "token",
    "latency",
    "cluster",
    "route",
    "policy",
    "filter",
    "stream",
    "buffer",
    "cache",
    "schema",
    "record",
    "payload",
    "header",
    "cursor",
    "session",
    "prompt",
    "vector",
    "gradient",
    "tensor",
    "sample",
    "budget",
    "quota",
    "region",
    "shard",
    "replica",
    "commit",
    "branch",
    "merge",
    "queue",
    "worker",
    "signal",
    "handle",
    "future",
    "async",
    "await",
    "encode",
    "decode",
    "verify",
    "resolve",
    "propose",
    "observe",
    "measure",
    "analyze",
    "summarize",
    "continue",
    "compress",
];

/// Build a paragraph of `words` varied tokens with occasional sentence breaks,
/// so the text has realistic (not degenerate) compressibility.
fn prose(rng: &mut Lcg, words: usize) -> String {
    let mut out = String::with_capacity(words * 8);
    for i in 0..words {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(rng.pick(WORD_BANK));
        // Roughly one sentence break every ~12 words.
        if is_sentence_break(rng) {
            out.push('.');
        }
    }
    out
}

/// Draw whether the current position ends a sentence (~1 in 12).
fn is_sentence_break(rng: &mut Lcg) -> bool {
    (rng.next_u64() >> 40).is_multiple_of(12)
}

/// A random 32-hex-character id; kept unique so ids do not compress away.
fn random_id(rng: &mut Lcg, prefix: &str) -> String {
    format!("{prefix}_{:016x}{:016x}", rng.next_u64(), rng.next_u64())
}

/// A synthetic `response_object` column: assistant prose plus less-compressible
/// ids and usage counters.
fn sample_response_object(rng: &mut Lcg, scale: usize) -> Value {
    json!({
        "id": random_id(rng, "resp"),
        "object": "response",
        "status": "completed",
        "model": "google/gemma-4-12B-it",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": prose(rng, scale * 64)}],
        }],
        "usage": {
            "input_tokens": rng.next_u64() % 4096,
            "output_tokens": rng.next_u64() % 4096,
            "total_tokens": rng.next_u64() % 8192,
        },
    })
}

/// A synthetic `input` column: a multi-turn user/assistant transcript.
fn sample_input(rng: &mut Lcg, turns: usize) -> Value {
    let messages: Vec<Value> = (0..turns)
        .map(|i| {
            json!({
                "type": "message",
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": [{"type": "input_text", "text": prose(rng, 32)}],
            })
        })
        .collect();
    Value::Array(messages)
}

/// Build the three columns for one stored response at a given scale and variant.
/// The seed is derived from both, so a given (scale, variant) is identical
/// across backends and compression settings, while different variants differ in
/// content (so writing them in turn actually changes stored bytes).
fn response_columns(scale: usize, variant: usize) -> [Value; 3] {
    let mut rng = Lcg::new(0x5150_1234_ABCD_0001 ^ ((scale as u64) << 8) ^ variant as u64);
    [
        sample_response_object(&mut rng, scale),
        sample_input(&mut rng, scale * 2),
        sample_input(&mut rng, scale * 2),
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

/// Build a multi-thread runtime so the offloaded codec (`spawn_blocking`) and
/// the store's blocking write path both run as they do in production.
fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

/// File-backed SQLite store: one run per compression setting against its own DB
/// file so writes measure the real disk path, not an in-memory shortcut.
fn bench_sqlite_store(c: &mut Criterion) {
    let runtime = build_runtime();
    let dir = tempfile::tempdir().expect("temp dir");
    for (algorithm, config) in [("none", none_config()), ("zstd", zstd_config())] {
        let path = dir.path().join(format!("{algorithm}.db"));
        let url = format!("sqlite://{}?mode=rwc", path.display());
        let store = runtime.block_on(sqlite_store(&url, &config));
        let stored_size = |id: &str| runtime.block_on(sqlite_stored_bytes(&url, id));
        bench_backend(
            c,
            &runtime,
            &format!("sqlite_response_{algorithm}"),
            &store,
            &stored_size,
        );
    }
}

/// `PostgreSQL` store, only when `DATABASE_URL` is set. Each compression setting
/// gets its own tables under a per-run token, so settings never share TOAST or
/// row churn and concurrent runs cannot overwrite each other. Tables are dropped
/// after each setting, outside the timed loop.
fn bench_postgres_store(c: &mut Criterion) {
    let Ok(url) = std::env::var("DATABASE_URL") else {
        println!("skipping postgres benchmarks: DATABASE_URL not set");
        return;
    };
    let runtime = build_runtime();
    let tls = pg_tls_config();
    let token = unique_token();
    for (algorithm, config) in [("none", none_config()), ("zstd", zstd_config())] {
        let responses = format!("bench_resp_{token}_{algorithm}");
        let conversations = format!("bench_conv_{token}_{algorithm}");
        let store = runtime.block_on(postgres_store(&url, &tls, &responses, &conversations, &config));
        let stored_size = |id: &str| runtime.block_on(postgres_stored_bytes(&url, &tls, &responses, id));
        bench_backend(
            c,
            &runtime,
            &format!("postgres_response_{algorithm}"),
            &store,
            &stored_size,
        );
        runtime.block_on(drop_pg_tables(&url, &tls, &[&responses, &conversations]));
    }
}

/// TLS settings for the benchmark's `PostgreSQL` connection.
///
/// Defaults to the store's secure default (`verify-full`); the store always
/// imposes its own `ssl_mode`, so plaintext must be opted into explicitly via
/// `PRAXIS_BENCH_PG_SSLMODE` (e.g. `disable` for a local throwaway database).
fn pg_tls_config() -> PgTlsConfig<'static> {
    let ssl_mode = std::env::var("PRAXIS_BENCH_PG_SSLMODE")
        .ok()
        .map(|raw| match raw.as_str() {
            "disable" => SslMode::Disable,
            "prefer" => SslMode::Prefer,
            "require" => SslMode::Require,
            "verify-ca" => SslMode::VerifyCa,
            _ => SslMode::VerifyFull,
        });
    PgTlsConfig {
        ssl_mode,
        ..PgTlsConfig::default()
    }
}

/// A short per-run identifier for table isolation: process id mixed with a
/// nanosecond timestamp, hex-encoded, so parallel or repeated runs never collide
/// while staying within `PostgreSQL`'s 63-byte identifier limit (the store
/// derives longer suffixed names from these).
#[expect(
    clippy::cast_possible_truncation,
    reason = "only the low 64 bits are needed for a collision-resistant token"
)]
fn unique_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{:x}", (nanos as u64) ^ u64::from(std::process::id()))
}

/// Drop the benchmark's `PostgreSQL` tables. For each base name this also drops
/// the tables the store derives from it (`_pending_approvals`,
/// `_schema_version`, ...), found by prefix so cleanup stays complete without
/// hard-coding the store's internal suffixes. Runs outside timing; names come
/// from `pg_tables`, so interpolating them into the `DROP` is safe here.
async fn drop_pg_tables(url: &str, tls: &PgTlsConfig<'_>, tables: &[&str]) {
    let options = url
        .parse::<PgConnectOptions>()
        .expect("parse database url")
        .ssl_mode(PgSslMode::from(tls.ssl_mode.unwrap_or_default()));
    let pool = PgPoolOptions::new()
        .connect_with(options)
        .await
        .expect("cleanup connect");
    for base in tables {
        // Escape LIKE metacharacters so the `_` in the token stays literal.
        let pattern = format!("{}%", base.replace('_', "\\_"));
        let names: Vec<String> =
            sqlx::query_scalar("SELECT tablename FROM pg_tables WHERE tablename LIKE $1 ESCAPE '\\'")
                .bind(pattern)
                .fetch_all(&pool)
                .await
                .expect("list bench tables");
        for name in names {
            sqlx::query(sqlx::AssertSqlSafe(format!("DROP TABLE IF EXISTS {name} CASCADE")))
                .execute(&pool)
                .await
                .expect("drop table");
        }
    }
    pool.close().await;
}

/// Seed and time upsert/get for every scale against one initialized store.
///
/// `stored_size` reports the stored size of the three JSON columns for a given
/// record id, read straight from the backend, so the ratio reflects the codec's
/// effect on what the backend actually persists for those columns. It is a
/// per-column value size, not the total database footprint (indexes, other
/// columns, and page/row overhead are excluded).
fn bench_backend(
    c: &mut Criterion,
    runtime: &tokio::runtime::Runtime,
    group_name: &str,
    store: &dyn ResponseStore,
    stored_size: &dyn Fn(&str) -> u64,
) {
    let mut group = c.benchmark_group(group_name);
    for scale in SCALES {
        // Distinct payloads sharing one id, prepared outside timing. The upsert
        // loop cycles them so each write updates an existing row with new bytes.
        let records = sample_records(scale, PAYLOAD_VARIANTS);
        let representative = records.first().expect("at least one payload variant");
        let uncompressed: u64 = [
            &representative.response_object,
            &representative.input,
            &representative.messages,
        ]
        .into_iter()
        .map(json_len)
        .sum();
        group.throughput(Throughput::Bytes(uncompressed));
        // Seed first, then read the actual persisted size for the ratio report.
        runtime.block_on(store.upsert_response(representative)).expect("seed");
        report_ratio(group_name, scale, uncompressed, stored_size(&representative.id));
        bench_store_record(&mut group, runtime, store, &records, scale);
    }
    group.finish();
}

/// Print uncompressed vs actually-stored bytes and the resulting ratio.
#[expect(
    clippy::cast_precision_loss,
    reason = "byte counts stay far below f64's exact-integer range"
)]
fn report_ratio(label: &str, scale: usize, uncompressed: u64, stored: u64) {
    let ratio = uncompressed as f64 / stored.max(1) as f64;
    println!("{label} scale={scale}: {uncompressed} B -> {stored} B ({ratio:.2}x)");
}

/// Stored value size of the three JSON columns for `id` in a SQLite store.
///
/// `length()` on a blob returns its byte count. SQLite does not compress column
/// values, so this is the bytes the codec actually wrote to those columns.
async fn sqlite_stored_bytes(url: &str, id: &str) -> u64 {
    let pool = sqlx::sqlite::SqlitePool::connect(url).await.expect("size connect");
    let bytes: i64 = sqlx::query_scalar(
        "SELECT length(response_object) + length(input) + length(messages) FROM responses WHERE id = ?",
    )
    .bind(id)
    .fetch_one(&pool)
    .await
    .expect("sqlite stored size");
    pool.close().await;
    u64::try_from(bytes).unwrap_or(0)
}

/// Stored value size of the three JSON columns for `id` in a `PostgreSQL` store,
/// read back with the benchmark's resolved TLS settings.
///
/// Uses `pg_column_size`, which reports the on-disk size of each value including
/// `PostgreSQL`'s own column (TOAST) compression, so the ratio also reflects any
/// compression the backend applies to the `none` codec's plaintext JSON.
async fn postgres_stored_bytes(url: &str, tls: &PgTlsConfig<'_>, table: &str, id: &str) -> u64 {
    let options = url
        .parse::<PgConnectOptions>()
        .expect("parse database url")
        .ssl_mode(PgSslMode::from(tls.ssl_mode.unwrap_or_default()));
    let pool = PgPoolOptions::new().connect_with(options).await.expect("size connect");
    let sql = format!(
        "SELECT (pg_column_size(response_object) + pg_column_size(input) + pg_column_size(messages))::bigint \
         FROM {table} WHERE id = $1"
    );
    let bytes: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(sql))
        .bind(id)
        .fetch_one(&pool)
        .await
        .expect("postgres stored size");
    pool.close().await;
    u64::try_from(bytes).unwrap_or(0)
}

/// Time only the public store operations against an already initialized store.
/// The upsert loop rotates through `records` (same id, differing content) so it
/// measures real page writes rather than no-op updates of unchanged bytes.
fn bench_store_record(
    group: &mut BenchmarkGroup<'_, WallTime>,
    runtime: &tokio::runtime::Runtime,
    store: &dyn ResponseStore,
    records: &[ResponseRecord],
    scale: usize,
) {
    let seed = records.first().expect("at least one payload variant");
    let owner = &seed.owner;
    let id = &seed.id;
    group.bench_function(BenchmarkId::new("upsert", scale), |b| {
        let mut turn = 0_usize;
        b.iter(|| {
            let record = records
                .get(turn % records.len())
                .expect("modulo keeps the index in bounds");
            turn += 1;
            runtime
                .block_on(store.upsert_response(black_box(record)))
                .expect("upsert");
        });
    });
    // The upsert loop leaves whichever variant ran last, and criterion picks
    // iteration counts per benchmark. Reseed the representative record (outside
    // timing) so every get reads identical content matching the reported size.
    runtime.block_on(store.upsert_response(seed)).expect("reseed");
    group.bench_function(BenchmarkId::new("get", scale), |b| {
        b.iter(|| {
            runtime
                .block_on(store.get_response(owner, black_box(id)))
                .expect("get")
                .expect("record")
        });
    });
}

/// Initialize a file-backed SQLite schema and pool once, outside timing loops.
async fn sqlite_store(url: &str, config: &StoreCompressionConfig) -> SqliteResponseStore {
    SqliteResponseStore::new(url, "responses", "conversations", None, None, Some(config))
        .await
        .expect("sqlite store")
}

/// Initialize the `PostgreSQL` schema and pool once, outside timing loops.
async fn postgres_store(
    url: &str,
    tls: &PgTlsConfig<'_>,
    responses_table: &str,
    conversations_table: &str,
    config: &StoreCompressionConfig,
) -> PostgresResponseStore {
    PostgresResponseStore::new(url, responses_table, conversations_table, None, tls, None, Some(config))
        .await
        .expect("postgres store")
}

/// Build `count` records that share one id and owner but carry distinct payload
/// contents, so upserting them in turn is an update-in-place with changed bytes.
fn sample_records(scale: usize, count: usize) -> Vec<ResponseRecord> {
    let owner = StateOwner::from_trusted_parts("benchmark", "issuer", "subject").expect("owner");
    (0..count)
        .map(|variant| {
            let [response_object, input, messages] = response_columns(scale, variant);
            ResponseRecord {
                id: "resp_benchmark".to_owned(),
                owner: owner.clone(),
                created_at: 1000,
                model: "benchmark".to_owned(),
                response_object,
                input,
                messages,
            }
        })
        .collect()
}

criterion_group!(benches, bench_sqlite_store, bench_postgres_store);
criterion_main!(benches);
