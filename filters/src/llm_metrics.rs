// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! LLM gateway metrics emitted by in-tree filters.
//!
//! These complement the proxy-owned `praxis_http_*` metrics (which live in
//! the external `praxis-proxy` crates) with model/provider-labeled signals
//! that the proxy cannot produce on its own: token accounting from backend
//! usage metadata, per-token decode timing, finish reasons, and empty
//! completions.
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

/// Metadata key where the `model_to_header` filter publishes the model it
/// promoted from the request body. Server-published (not a raw client header),
/// and validated by `model_to_header` for header-safety; keep it bounded to the
/// models routing actually serves.
pub(crate) const META_MODEL_TO_HEADER_MODEL: &str = "model_to_header.model";

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
/// `unknown` when no upstream format filter (or `model_to_header`) ran.
pub fn resolve_model(ctx: &HttpFilterContext<'_>) -> String {
    ctx.get_metadata("openai_responses_request.model")
        .or_else(|| ctx.get_metadata("openai_chat_completions_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_to_chat_completions.model"))
        // The model_to_header filter publishes the client-requested model as
        // `model_to_header.model` metadata before token_count runs; fall back to
        // it so the label is populated even when no format filter ran. Only that
        // server-published, header-safety-validated value is used — a raw
        // client-supplied `X-Model` request header is never read, so clients
        // cannot spoof the label. Distinct values stay bounded to the model set
        // the gateway routes (a catch-all route could still admit new values).
        .or_else(|| ctx.get_metadata(META_MODEL_TO_HEADER_MODEL))
        .filter(|v| !v.is_empty() && is_promotable_value(v))
        .unwrap_or("unknown")
        .to_owned()
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
        metrics::with_local_recorder(&recorder, || record_token_usage(&ctx, 12, 34, Some(5), Some(2)));
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
        metrics::with_local_recorder(&recorder, || record_token_usage(&ctx, 12, 0, None, None));
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
}
