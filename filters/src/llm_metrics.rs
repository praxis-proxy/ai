// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! LLM gateway metrics emitted by in-tree filters.
//!
//! These complement the proxy-owned `praxis_http_*` metrics (which live in
//! the external `praxis-proxy` crates) with model/provider-labeled signals
//! that the proxy cannot produce on its own: token accounting from backend
//! usage metadata, per-request TPOT, per-interval ITL, finish reasons, and
//! empty completions.
//!
//! Every metric carries a `model` label for drill-down and a `cluster` label
//! for provider-level rollup, so both per-model and per-provider views are
//! one `sum by (...)` away.
//!
//! See `docs/llm-metrics.md` for the metric catalog, vLLM parity notes, query
//! examples, and the overhead budget.
//!
//! The `model` label is bounded: at most [`MAX_MODEL_LABELS`] distinct values
//! are admitted per process lifetime, and later unseen values are labeled
//! `other`, so the exporter registry cannot grow without bound on chains
//! without a model allowlist (the exporter has no idle eviction).
//!
//! # Metric rationale
//!
//! - `praxis_ai_prompt_tokens_total` / `praxis_ai_output_tokens_total`: token volumes per model/provider (capacity,
//!   cost, output-length blowups).
//! - `praxis_ai_prompt_tokens` / `praxis_ai_output_tokens` (histograms): per-request token distributions (average
//!   prompt/output size per model).
//! - `praxis_ai_reasoning_tokens_total`: thinking-model behavior.
//! - `praxis_ai_cache_read_tokens_total`: prompt-cache hit volume (cost/latency).
//! - `praxis_ai_finish_reasons_total{reason}`: *why* generations end — `length` share = max-token/capacity friction,
//!   `content_filter` = policy.
//! - `praxis_ai_empty_completions_total`: successful zero-output responses, excluding tool-call, refusal, and
//!   content-filter terminations.
//! - `praxis_ai_tpot_seconds` (histogram): vLLM-style per-request time per output token, `(last output delta - first
//!   output delta) / (output_tokens - 1)`.
//! - `praxis_ai_itl_seconds` (histogram): time between consecutive output deltas, matching vLLM's chunk-level ITL
//!   distribution.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex, Once, OnceLock},
};

use arc_swap::ArcSwap;
use metrics::{Histogram, Unit, counter, describe_histogram, histogram};
use praxis_ai_apis::promotion::is_promotable_value;
use praxis_filter::HttpFilterContext;

/// Prompt tokens (backend-reported) per request, cumulative.
pub const METRIC_PROMPT_TOKENS_TOTAL: &str = "praxis_ai_prompt_tokens_total";
/// Output tokens (backend-reported) per request, cumulative.
pub const METRIC_OUTPUT_TOKENS_TOTAL: &str = "praxis_ai_output_tokens_total";
/// Prompt tokens per request, histogram.
pub const METRIC_PROMPT_TOKENS: &str = "praxis_ai_prompt_tokens";
/// Output tokens per request, histogram.
pub const METRIC_OUTPUT_TOKENS: &str = "praxis_ai_output_tokens";
/// Reasoning / thinking tokens (backend-reported), cumulative.
pub const METRIC_REASONING_TOKENS_TOTAL: &str = "praxis_ai_reasoning_tokens_total";
/// Input tokens served from the provider's prompt cache, cumulative.
pub const METRIC_CACHE_READ_TOKENS_TOTAL: &str = "praxis_ai_cache_read_tokens_total";
/// Successful responses reporting zero output tokens, excluding tool-call,
/// refusal, and content-filter completions.
pub const METRIC_EMPTY_COMPLETIONS_TOTAL: &str = "praxis_ai_empty_completions_total";
/// Why a generation ended (`stop`, `length`, `content_filter`, `tool_calls`, ...).
pub const METRIC_FINISH_REASONS_TOTAL: &str = "praxis_ai_finish_reasons_total";
/// vLLM-style per-request time per output token, excluding the first token.
pub const METRIC_TPOT_SECONDS: &str = "praxis_ai_tpot_seconds";
/// Time between consecutive output deltas in a streamed response.
pub const METRIC_ITL_SECONDS: &str = "praxis_ai_itl_seconds";

/// Metadata key where the `model_to_header` filter publishes the model it
/// promoted from the request body. Server-published (not a raw client header),
/// and validated by `model_to_header` for header-safety; distinct admitted
/// values are capped at [`MAX_MODEL_LABELS`] per process lifetime.
pub(crate) const META_MODEL_TO_HEADER_MODEL: &str = "model_to_header.model";

/// Upper bound on distinct `model` label values admitted per process.
///
/// The production Prometheus recorder has no idle eviction, so every distinct
/// label value lives for the recorder lifetime. Routing bounds the model set
/// in chains with a model allowlist, but chains without one (for example the
/// token-counting example's catch-all route) would let clients expand the
/// label set — and with it the registry and `/metrics` output — without
/// bound. Values arriving after this cap is full are labeled
/// [`OTHER_MODEL_LABEL`]. The current `EnMaaS` catalog serves ~29 models, so
/// 1024 leaves an order of magnitude of headroom for catalog churn.
pub(crate) const MAX_MODEL_LABELS: usize = 1024;

/// Label for model values that arrive once [`MAX_MODEL_LABELS`] distinct
/// values have been admitted. Distinct from `"unknown"` (no model metadata)
/// so capped cardinality stays observable.
const OTHER_MODEL_LABEL: &str = "other";

/// Guards the one-time histogram descriptions.
static DESCRIBED: Once = Once::new();

/// Describe histogram units once so the Prometheus exporter renders sensible units.
fn ensure_described() {
    DESCRIBED.call_once(|| {
        describe_histogram!(METRIC_PROMPT_TOKENS, Unit::Count, "prompt tokens per request");
        describe_histogram!(METRIC_OUTPUT_TOKENS, Unit::Count, "output tokens per request");
        describe_histogram!(
            METRIC_TPOT_SECONDS,
            Unit::Seconds,
            "per-request time per output token: (last-first)/(output_tokens-1)"
        );
        describe_histogram!(
            METRIC_ITL_SECONDS,
            Unit::Seconds,
            "time between consecutive output deltas"
        );
    });
}

/// Resolve the `model` label from request/filter metadata, falling back to
/// `unknown` when no upstream format filter (or `model_to_header`) ran.
///
/// Distinct values are capped at [`MAX_MODEL_LABELS`] for the process
/// lifetime; values arriving after the cap are labeled `"other"`.
pub fn resolve_model(ctx: &HttpFilterContext<'_>) -> String {
    // Validate each candidate independently so an invalid value from one
    // format filter cannot suppress a later valid model_to_header value.
    let model = [
        "openai_responses_request.model",
        "openai_chat_completions_request.model",
        "anthropic_messages_request.model",
        "anthropic_messages_to_chat_completions.model",
        // The model_to_header filter publishes the client-requested model as
        // `model_to_header.model` metadata before token_count runs; fall back to
        // it so the label is populated even when no format filter ran. Only that
        // server-published, header-safety-validated value is used — a raw
        // client-supplied `X-Model` request header is never read, so clients
        // cannot spoof the label.
        META_MODEL_TO_HEADER_MODEL,
    ]
    .into_iter()
    .find_map(|key| {
        ctx.get_metadata(key)
            .filter(|value| !value.is_empty() && is_promotable_value(value))
    })
    .unwrap_or("unknown");
    if model == OTHER_MODEL_LABEL || model == "unknown" {
        return model.to_owned();
    }
    admit_model(model).to_owned()
}

/// Immutable snapshot of the admitted model labels. Reads are lock-free (an
/// atomic snapshot load), so membership checks never block — not on each
/// other and not on an in-flight admission swap. First-time admissions
/// replace the whole snapshot under [`ADMISSION_LOCK`], which keeps the
/// [`MAX_MODEL_LABELS`] cap exact under simultaneous first-time admissions.
static ADMITTED_MODELS: OnceLock<ArcSwap<HashSet<String>>> = OnceLock::new();

/// Serializes snapshot replacement during first-time admission only.
static ADMISSION_LOCK: Mutex<()> = Mutex::new(());

/// Admit `value` against `admitted` while below `cap`; values arriving after
/// the cap map to [`OTHER_MODEL_LABEL`]. Already-admitted values keep their
/// label, so the cap bounds the set without relabeling earlier traffic.
fn admit_model_label<'v>(admitted: &mut HashSet<String>, cap: usize, value: &'v str) -> &'v str {
    if admitted.contains(value) {
        return value;
    }
    if admitted.len() < cap {
        admitted.insert(value.to_owned());
        value
    } else {
        OTHER_MODEL_LABEL
    }
}

/// Apply the process-lifetime [`MAX_MODEL_LABELS`] bound to `value`.
///
/// The fast path is a lock-free snapshot load for already-admitted labels.
/// First-time admissions serialize on [`ADMISSION_LOCK`] and swap in a fresh
/// snapshot; the over-cap check runs before the clone so a stream of novel
/// values (once the cap is full) never copies or swaps.
fn admit_model(value: &str) -> &str {
    let cell = ADMITTED_MODELS.get_or_init(|| ArcSwap::from_pointee(HashSet::new()));
    // Fast path: lock-free membership check. The Guard is scoped so the
    // borrow checker guarantees it is dropped before any `store`.
    {
        let snapshot = cell.load();
        if snapshot.contains(value) {
            return value;
        }
    }

    // Slow path: first-time admission, rare (at most MAX_MODEL_LABELS times
    // per process). `load_full` takes the snapshot Arc outright, so the
    // clone below is the deep HashSet clone.
    let _admission = ADMISSION_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut updated = {
        // Re-check under the lock: a concurrent admission may have inserted
        // this value or filled the cap meanwhile.
        let snapshot = cell.load_full();
        let current: &HashSet<String> = &snapshot;
        if current.contains(value) {
            return value;
        }
        if current.len() >= MAX_MODEL_LABELS {
            return OTHER_MODEL_LABEL;
        }
        current.clone()
    };
    let label = admit_model_label(&mut updated, MAX_MODEL_LABELS, value);
    cell.store(Arc::new(updated));
    label
}

/// Resolve the `cluster` (provider) label.
pub fn cluster(ctx: &HttpFilterContext<'_>) -> String {
    ctx.cluster_name().map_or_else(|| "unknown".to_owned(), str::to_owned)
}

/// Emit per-request token, cache, reasoning and empty-completion metrics from
/// backend-reported usage.
#[expect(
    clippy::cast_precision_loss,
    reason = "token counts are recorded as f64 histogram values"
)]
#[expect(clippy::too_many_arguments, reason = "explicit metric inputs")]
pub fn record_token_usage(
    ctx: &HttpFilterContext<'_>,
    input: u64,
    output: u64,
    cache_read: Option<u64>,
    reasoning: Option<u64>,
    count_empty: bool,
) {
    ensure_described();
    let model = resolve_model(ctx);
    let cluster = cluster(ctx);

    counter!(METRIC_PROMPT_TOKENS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone()).increment(input);
    histogram!(METRIC_PROMPT_TOKENS, "model" => model.clone(), "cluster" => cluster.clone()).record(input as f64);

    counter!(METRIC_OUTPUT_TOKENS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone()).increment(output);
    histogram!(METRIC_OUTPUT_TOKENS, "model" => model.clone(), "cluster" => cluster.clone()).record(output as f64);

    if let Some(cache_read) = cache_read {
        counter!(METRIC_CACHE_READ_TOKENS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone())
            .increment(cache_read);
    }
    if let Some(reasoning) = reasoning {
        counter!(METRIC_REASONING_TOKENS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone())
            .increment(reasoning);
    }
    if count_empty && output == 0 {
        counter!(METRIC_EMPTY_COMPLETIONS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone()).increment(1);
    }
}

/// Build a per-stream TPOT histogram handle with fixed labels.
pub(crate) fn tpot_histogram(model: &str, cluster: &str) -> Histogram {
    ensure_described();
    histogram!(
        METRIC_TPOT_SECONDS,
        "model" => model.to_owned(),
        "cluster" => cluster.to_owned()
    )
}

/// Build a per-stream ITL histogram handle with fixed labels.
pub(crate) fn itl_histogram(model: &str, cluster: &str) -> Histogram {
    ensure_described();
    histogram!(
        METRIC_ITL_SECONDS,
        "model" => model.to_owned(),
        "cluster" => cluster.to_owned()
    )
}

/// Emit one finish-reason observation.
pub fn record_finish_reason(ctx: &HttpFilterContext<'_>, reason: &str) {
    let model = resolve_model(ctx);
    let cluster = cluster(ctx);
    counter!(
        METRIC_FINISH_REASONS_TOTAL,
        "model" => model,
        "cluster" => cluster,
        "reason" => reason.to_owned()
    )
    .increment(1);
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use http::{HeaderValue, Method};
    use metrics_util::debugging::DebuggingRecorder;

    use super::*;
    use crate::test_utils::make_filter_context;

    #[test]
    fn resolve_model_prefers_format_metadata() {
        let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("openai_chat_completions_request.model", "gpt-format");
        ctx.set_metadata("model_to_header.model", "gpt-header");
        assert_eq!(resolve_model(&ctx), "gpt-format");
    }

    #[test]
    fn resolve_model_uses_model_to_header_metadata() {
        let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("model_to_header.model", "gpt-header");
        assert_eq!(resolve_model(&ctx), "gpt-header");
    }

    #[test]
    fn resolve_model_falls_through_invalid_format_metadata() {
        let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("openai_chat_completions_request.model", "bad\nmodel");
        ctx.set_metadata("model_to_header.model", "gpt-header");
        assert_eq!(resolve_model(&ctx), "gpt-header");
    }

    #[test]
    fn resolve_model_ignores_raw_client_x_model_header() {
        let mut req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        req.headers
            .insert("x-model", HeaderValue::from_static("client-spoofed"));
        let ctx = make_filter_context(&req);
        assert_eq!(
            resolve_model(&ctx),
            "unknown",
            "a raw client X-Model header must never become a label"
        );
    }

    #[test]
    fn record_token_usage_emits_labeled_metrics() {
        let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.cluster = Some("cb-glm".to_owned().into());
        ctx.set_metadata("openai_chat_completions_request.model", "gpt-x");

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || record_token_usage(&ctx, 12, 34, Some(5), Some(2), true));
        let snap = snapshotter.snapshot().into_vec();

        let total = |name: &str| -> u64 {
            snap.iter()
                .filter(|(k, ..)| k.key().name() == name)
                .filter_map(|(_, _, _, v)| match v {
                    metrics_util::debugging::DebugValue::Counter(c) => Some(*c),
                    _ => None,
                })
                .sum()
        };
        let present = |name: &str| snap.iter().any(|(k, ..)| k.key().name() == name);
        let label = |name: &str, want: &str| -> Option<String> {
            snap.iter()
                .find(|(k, ..)| k.key().name() == name)
                .and_then(|(k, ..)| k.key().labels().find(|l| l.key() == want).map(|l| l.value().to_owned()))
        };

        assert_eq!(total(METRIC_PROMPT_TOKENS_TOTAL), 12);
        assert_eq!(total(METRIC_OUTPUT_TOKENS_TOTAL), 34);
        assert_eq!(total(METRIC_CACHE_READ_TOKENS_TOTAL), 5);
        assert_eq!(total(METRIC_REASONING_TOKENS_TOTAL), 2);
        assert!(
            !present(METRIC_EMPTY_COMPLETIONS_TOTAL),
            "empty_completions must not be recorded on a non-empty response"
        );
        for metric in [METRIC_PROMPT_TOKENS, METRIC_OUTPUT_TOKENS] {
            let recorded: Vec<_> = snap
                .iter()
                .filter(|(k, ..)| k.key().name() == metric)
                .filter_map(|(_, _, _, v)| match v {
                    metrics_util::debugging::DebugValue::Histogram(h) => Some(h.len()),
                    _ => None,
                })
                .collect();
            assert_eq!(recorded, vec![1], "{metric} should record once");
        }
        assert_eq!(label(METRIC_PROMPT_TOKENS_TOTAL, "model").as_deref(), Some("gpt-x"));
        assert_eq!(label(METRIC_PROMPT_TOKENS_TOTAL, "cluster").as_deref(), Some("cb-glm"));
        assert_eq!(
            label(METRIC_REASONING_TOKENS_TOTAL, "cluster").as_deref(),
            Some("cb-glm")
        );
    }

    #[test]
    fn record_token_usage_bumps_empty_completions_on_zero_output() {
        let req = crate::test_utils::make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("model_to_header.model", "gpt-x");

        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || record_token_usage(&ctx, 12, 0, None, None, true));
        let snap = snapshotter.snapshot().into_vec();

        let total = |name: &str| -> u64 {
            snap.iter()
                .filter(|(k, ..)| k.key().name() == name)
                .filter_map(|(_, _, _, v)| match v {
                    metrics_util::debugging::DebugValue::Counter(c) => Some(*c),
                    _ => None,
                })
                .sum()
        };
        let label = |name: &str, want: &str| -> Option<String> {
            snap.iter()
                .find(|(k, ..)| k.key().name() == name)
                .and_then(|(k, ..)| k.key().labels().find(|l| l.key() == want).map(|l| l.value().to_owned()))
        };
        let present = |name: &str| snap.iter().any(|(k, ..)| k.key().name() == name);

        assert_eq!(total(METRIC_EMPTY_COMPLETIONS_TOTAL), 1);
        assert_eq!(label(METRIC_EMPTY_COMPLETIONS_TOTAL, "model").as_deref(), Some("gpt-x"));
        assert!(
            !present(METRIC_CACHE_READ_TOKENS_TOTAL),
            "cache-read tokens must not be recorded when unreported"
        );
        assert!(
            !present(METRIC_REASONING_TOKENS_TOTAL),
            "reasoning tokens must not be recorded when unreported"
        );
    }

    #[test]
    fn admit_model_label_caps_distinct_values() {
        let mut admitted = HashSet::new();
        assert_eq!(admit_model_label(&mut admitted, 3, "m1"), "m1");
        assert_eq!(admit_model_label(&mut admitted, 3, "m2"), "m2");
        assert_eq!(admit_model_label(&mut admitted, 3, "m3"), "m3");
        assert_eq!(
            admit_model_label(&mut admitted, 3, "m4"),
            "other",
            "values arriving after the cap map to other"
        );
        assert_eq!(
            admit_model_label(&mut admitted, 3, "m1"),
            "m1",
            "already-admitted values keep their label"
        );
        assert_eq!(admitted.len(), 3, "the cap bounds the admitted set");
    }

    #[test]
    fn admit_model_admits_values_under_the_process_cap() {
        let value = format!("cap-probe-{}", line!());
        assert_eq!(
            admit_model(&value),
            value,
            "values under the process cap are admitted unchanged"
        );
    }

    /// Number of distinct labels currently in the process-wide admitted set.
    fn admitted_len() -> usize {
        ADMITTED_MODELS
            .get_or_init(|| ArcSwap::from_pointee(HashSet::new()))
            .load()
            .len()
    }

    /// True when `value` is in the process-wide admitted set. Tests run in
    /// parallel and share the set, so assertions use membership (a `HashSet`
    /// admits each value exactly once) instead of set-length deltas.
    fn admitted_contains(value: &str) -> bool {
        ADMITTED_MODELS
            .get_or_init(|| ArcSwap::from_pointee(HashSet::new()))
            .load()
            .contains(value)
    }

    #[test]
    fn admit_model_admits_concurrent_distinct_values() {
        const THREADS: usize = 8;
        const PER_THREAD: usize = 32;
        let start = admitted_len();
        // One extra waiter: the main thread releases all admission threads
        // simultaneously after spawning them.
        let barrier = Arc::new(std::sync::Barrier::new(THREADS + 1));
        let mismatches = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut handles = Vec::new();
        for t in 0..THREADS {
            let barrier = Arc::clone(&barrier);
            let mismatches = Arc::clone(&mismatches);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..PER_THREAD {
                    let value = format!("concurrent-t{t}-m{i}");
                    let label = admit_model(&value).to_owned();
                    if label != value {
                        mismatches
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(format!("{value} -> {label}"));
                    }
                }
            }));
        }
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        let mismatches = std::mem::take(&mut *mismatches.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        assert!(
            mismatches.is_empty(),
            "simultaneous first-time admissions were mislabeled: {mismatches:?}"
        );
        for t in 0..THREADS {
            for i in 0..PER_THREAD {
                assert!(
                    admitted_contains(&format!("concurrent-t{t}-m{i}")),
                    "every distinct value must be admitted"
                );
            }
        }
        assert!(
            admitted_len() - start >= THREADS * PER_THREAD,
            "the admitted set must have grown by at least the test's values"
        );
    }

    #[test]
    fn admit_model_dedupes_concurrent_same_value() {
        let value = format!("dedupe-probe-{}", line!());
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let value = value.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                admit_model(&value).to_owned()
            }));
        }
        barrier.wait();
        let labels: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(labels.len(), 2);
        assert!(
            labels.iter().all(|label| label == &value),
            "both racers must observe the admitted value: {labels:?}"
        );
        assert!(
            admitted_contains(&value),
            "the same value must be admitted exactly once (set membership)"
        );
    }

    #[test]
    fn admit_model_reads_stay_consistent_during_admissions() {
        let probe = format!("consistent-probe-{}", line!());
        assert_eq!(admit_model(&probe), probe);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let stop = Arc::clone(&stop);
            let probe = probe.clone();
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    assert_eq!(admit_model(&probe), probe);
                }
            })
        };
        for i in 0..64 {
            let value = format!("consistent-churn-{i}");
            assert_eq!(admit_model(&value), value);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();
    }
}
