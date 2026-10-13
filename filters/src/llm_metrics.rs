// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! LLM gateway metrics emitted by in-tree filters.
//!
//! These complement the proxy-owned `praxis_http_*` metrics (which live in
//! the external `praxis-proxy` crates) with model/provider-labeled signals
//! that the proxy cannot produce on its own: token accounting from backend
//! usage metadata, per-token decode timing, finish reasons, empty completions,
//! and precise body-size distributions.
//!
//! Every metric carries a `model` label for drill-down and a `cluster` label
//! for provider-level rollup, so both per-model and per-provider views are
//! one `sum by (...)` away.
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
//! - `praxis_ai_empty_completions_total`: zero-output responses = silent generation failures.
//! - `praxis_ai_tpot_seconds` (histogram): inter-token decode time (tokens/s ≈ 1/TPOT) — pure generation throughput.

use std::sync::Once;

use metrics::{Unit, counter, describe_histogram, histogram};
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
/// Responses that produced zero output tokens.
pub const METRIC_EMPTY_COMPLETIONS_TOTAL: &str = "praxis_ai_empty_completions_total";
/// Why a generation ended (`stop`, `length`, `content_filter`, `tool_calls`, ...).
pub const METRIC_FINISH_REASONS_TOTAL: &str = "praxis_ai_finish_reasons_total";
/// Time between consecutive streamed deltas (decode speed).
pub const METRIC_TPOT_SECONDS: &str = "praxis_ai_tpot_seconds";

/// Guards the one-time histogram descriptions.
static DESCRIBED: Once = Once::new();

/// Describe histogram units once so the Prometheus exporter renders sensible units.
fn ensure_described() {
    DESCRIBED.call_once(|| {
        describe_histogram!(METRIC_PROMPT_TOKENS, Unit::Count, "prompt tokens per request");
        describe_histogram!(METRIC_OUTPUT_TOKENS, Unit::Count, "output tokens per request");
        describe_histogram!(METRIC_TPOT_SECONDS, Unit::Seconds, "time per output token");
    });
}

/// Resolve the `model` label from request/filter metadata, falling back to
/// `unknown` when no upstream format filter ran before this one.
pub fn resolve_model(ctx: &HttpFilterContext<'_>) -> String {
    ctx.get_metadata("openai_responses_request.model")
        .or_else(|| ctx.get_metadata("openai_chat_completions_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_to_chat_completions.model"))
        // The model_to_header filter stamps the client-requested model as an
        // `X-Model` request header before token_count runs; fall back to it so
        // the label is populated even when no format filter ran.
        .or_else(|| ctx.request.headers.get("x-model").and_then(|v| v.to_str().ok()))
        .filter(|v| !v.is_empty() && is_promotable_value(v))
        .unwrap_or("unknown")
        .to_owned()
}

/// Resolve the `cluster` (provider) label.
pub fn cluster(ctx: &HttpFilterContext<'_>) -> String {
    ctx.cluster_name().map_or_else(|| "unknown".to_owned(), str::to_owned)
}

/// Emit per-request token, cache, reasoning, empty-completion and body-size
/// metrics from backend-reported usage.
#[expect(
    clippy::cast_precision_loss,
    reason = "token and byte counts are recorded as f64 histogram values"
)]
pub fn record_token_usage(
    ctx: &HttpFilterContext<'_>,
    input: u64,
    output: u64,
    cache_read: Option<u64>,
    reasoning: Option<u64>,
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
    if output == 0 {
        counter!(METRIC_EMPTY_COMPLETIONS_TOTAL, "model" => model.clone(), "cluster" => cluster.clone()).increment(1);
    }
}

/// Emit an inter-delta (decode) timing sample.
#[expect(dead_code, reason = "wired by consumers that measure inter-delta timing")]
pub fn record_tpot(ctx: &HttpFilterContext<'_>, delta_seconds: f64) {
    ensure_described();
    let model = resolve_model(ctx);
    let cluster = cluster(ctx);
    histogram!(METRIC_TPOT_SECONDS, "model" => model.clone(), "cluster" => cluster.clone()).record(delta_seconds);
}

/// Emit one finish-reason observation.
#[expect(dead_code, reason = "wired by consumers that parse finish reasons")]
pub fn record_finish_reason(ctx: &HttpFilterContext<'_>, reason: &str) {
    let model = resolve_model(ctx);
    let cluster = cluster(ctx);
    counter!(
        METRIC_FINISH_REASONS_TOTAL,
        "model" => model.clone(),
        "cluster" => cluster.clone(),
        "reason" => reason.to_owned()
    )
    .increment(1);
}
