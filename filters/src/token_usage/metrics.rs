// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded, per-tenant Prometheus token usage metrics.
//!
//! Reads token counts written by [`super::TokenCountFilter`] and emits
//! `praxis_ai_tokens_total{tenant, model, kind}`. Tenant identity is accepted
//! only from one exact, unnamespaced metadata key reserved for an
//! authentication filter. Raw headers, `identity_header_guard` metadata, and
//! `external_metering` fallbacks are deliberately ignored.
//!
//! Tenant and model labels are selected from static allowlists. Authenticated
//! values outside an allowlist use a fixed overflow label, so request order
//! can never decide which values consume exporter cardinality.

#![cfg_attr(
    not(test),
    expect(
        clippy::missing_docs_in_private_items,
        reason = "implementation details are documented at the filter API boundary"
    )
)]

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use bytes::Bytes;
use metrics::{SharedString, counter};
use praxis_ai_apis::promotion::is_promotable_value;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::Deserialize;
use tracing::{debug, trace};

use super::{META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_STATUS, META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW};
use crate::metering::META_METERING_MODEL;

const METRIC_TOKENS_TOTAL: &str = "praxis_ai_tokens_total";
const MAX_CONFIGURED_LABELS: usize = 1024;
/// Hard ceiling across tenant, model, and kind label combinations.
const MAX_METRIC_SERIES: usize = 10_000;
const DEFAULT_TENANT_METADATA_KEY: &str = "x-tenant-username";
const MODEL_UNKNOWN: &str = "unknown";
const DEFAULT_OVERFLOW_TENANT: &str = "other";
const DEFAULT_OVERFLOW_MODEL: &str = "other";

/// Deserialized YAML config for `token_usage_metrics`.
///
/// ```yaml
/// filter: token_usage_metrics
/// tenant_metadata_key: "x-tenant-username"
/// allowed_tenants: ["tenant-a", "tenant-b"]
/// allowed_models: ["gpt-4o", "claude-sonnet-4"]
/// overflow_tenant: other
/// overflow_model: other
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenUsageMetricsConfig {
    /// Exact unnamespaced metadata key populated from verified credentials by
    /// an authentication filter. Namespaced keys are rejected because
    /// `identity_header_guard` stores unverified client headers there.
    #[serde(default = "default_tenant_metadata_key")]
    tenant_metadata_key: String,

    /// Tenant labels admitted as their own Prometheus series.
    allowed_tenants: Vec<String>,

    /// Model labels admitted as their own Prometheus series.
    allowed_models: Vec<String>,

    /// Label for an authenticated tenant not in `allowed_tenants`.
    #[serde(default = "default_overflow_tenant")]
    overflow_tenant: String,

    /// Label for a model not in `allowed_models`.
    #[serde(default = "default_overflow_model")]
    overflow_model: String,
}

fn default_tenant_metadata_key() -> String {
    DEFAULT_TENANT_METADATA_KEY.to_owned()
}

fn default_overflow_tenant() -> String {
    DEFAULT_OVERFLOW_TENANT.to_owned()
}

fn default_overflow_model() -> String {
    DEFAULT_OVERFLOW_MODEL.to_owned()
}

#[expect(
    clippy::too_many_lines,
    reason = "configuration validation keeps all cardinality invariants together"
)]
fn validate_config(cfg: &TokenUsageMetricsConfig) -> Result<(), FilterError> {
    if cfg.tenant_metadata_key.is_empty()
        || cfg.tenant_metadata_key.contains('.')
        || !cfg
            .tenant_metadata_key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(
            "token_usage_metrics: tenant_metadata_key must be a non-empty unnamespaced metadata key containing only ASCII letters, digits, '-' or '_'"
                .into(),
        );
    }
    validate_overflow_label("overflow_tenant", &cfg.overflow_tenant)?;
    validate_overflow_label("overflow_model", &cfg.overflow_model)?;
    validate_allowlist("allowed_tenants", &cfg.allowed_tenants, &cfg.overflow_tenant)?;
    validate_allowlist("allowed_models", &cfg.allowed_models, &cfg.overflow_model)?;
    if cfg.overflow_model == MODEL_UNKNOWN {
        return Err("token_usage_metrics: overflow_model must differ from the reserved 'unknown' label".into());
    }
    if cfg.allowed_models.iter().any(|model| model == MODEL_UNKNOWN) {
        return Err("token_usage_metrics: allowed_models must not contain the reserved 'unknown' label".into());
    }
    let possible_series = (cfg.allowed_tenants.len() + 1)
        .saturating_mul(cfg.allowed_models.len() + 2)
        .saturating_mul(3);
    if possible_series > MAX_METRIC_SERIES {
        return Err(format!(
            "token_usage_metrics: allowlists could create {possible_series} series, exceeding the {MAX_METRIC_SERIES} series limit"
        )
        .into());
    }
    Ok(())
}

fn validate_overflow_label(field: &str, value: &str) -> Result<(), FilterError> {
    if value.is_empty() || !is_promotable_value(value) {
        return Err(format!("token_usage_metrics: {field} must be a non-empty promotable label").into());
    }
    Ok(())
}

fn validate_allowlist(field: &str, values: &[String], overflow: &str) -> Result<(), FilterError> {
    if values.is_empty() {
        return Err(format!("token_usage_metrics: {field} must contain at least one label").into());
    }
    if values.len() > MAX_CONFIGURED_LABELS {
        return Err(
            format!("token_usage_metrics: {field} cannot contain more than {MAX_CONFIGURED_LABELS} labels").into(),
        );
    }
    let mut unique = HashSet::with_capacity(values.len());
    for value in values {
        if value.is_empty() || !is_promotable_value(value) {
            return Err(format!("token_usage_metrics: {field} contains a non-promotable label").into());
        }
        if value == overflow {
            return Err(format!("token_usage_metrics: {field} must not contain its overflow label").into());
        }
        if !unique.insert(value) {
            return Err(format!("token_usage_metrics: {field} contains duplicate label '{value}'").into());
        }
    }
    Ok(())
}

/// Emits bounded per-tenant Prometheus token counters from `token.*` metadata.
///
/// The filter fails configuration unless tenant and model allowlists are
/// explicit. Requests without authenticated tenant metadata are skipped.
/// Values outside either allowlist map to fixed overflow labels.
///
/// The admin `/metrics` endpoint remains an operator boundary: protect it with
/// network policy and authentication. Prometheus labels do not provide tenant
/// authorization or billing-grade accounting.
///
/// Declare this filter **before** `token_count` so response hooks (which run in
/// reverse order) extract counts first.
pub struct TokenUsageMetricsFilter {
    allowed_models: HashMap<String, Arc<str>>,
    allowed_tenants: HashMap<String, Arc<str>>,
    overflow_model: Arc<str>,
    overflow_tenant: Arc<str>,
    tenant_metadata_key: String,
}

impl TokenUsageMetricsFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if required allowlists are absent or labels are
    /// unsafe.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config)?))
    }

    fn build(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: TokenUsageMetricsConfig = parse_filter_config("token_usage_metrics", config)?;
        validate_config(&cfg)?;
        Ok(Self {
            allowed_models: shared_labels(cfg.allowed_models),
            allowed_tenants: shared_labels(cfg.allowed_tenants),
            overflow_model: Arc::from(cfg.overflow_model.as_str()),
            overflow_tenant: Arc::from(cfg.overflow_tenant.as_str()),
            tenant_metadata_key: cfg.tenant_metadata_key,
        })
    }

    fn tenant_label(&self, raw: &str) -> SharedString {
        self.allowed_tenants
            .get(raw)
            .map_or_else(|| shared_string(&self.overflow_tenant), shared_string)
    }

    fn model_label(&self, raw: Option<&str>) -> SharedString {
        let Some(raw) = raw else {
            return SharedString::const_str(MODEL_UNKNOWN);
        };
        self.allowed_models
            .get(raw)
            .map_or_else(|| shared_string(&self.overflow_model), shared_string)
    }
}

fn shared_labels(values: Vec<String>) -> HashMap<String, Arc<str>> {
    values
        .into_iter()
        .map(|value| {
            let label: Arc<str> = Arc::from(value.as_str());
            (value, label)
        })
        .collect()
}

fn shared_string(label: &Arc<str>) -> SharedString {
    SharedString::from_shared(Arc::clone(label))
}

struct MetricsRequestState {
    model: SharedString,
    tenant: SharedString,
}

#[async_trait]
impl HttpFilter for TokenUsageMetricsFilter {
    fn name(&self) -> &'static str {
        "token_usage_metrics"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(tenant) = ctx.get_metadata(&self.tenant_metadata_key) else {
            trace!(metadata_key = %self.tenant_metadata_key, "no authenticated tenant metadata; skipping token usage metrics");
            return Ok(FilterAction::Continue);
        };
        if tenant.is_empty() || !is_promotable_value(tenant) {
            debug!(metadata_key = %self.tenant_metadata_key, "invalid authenticated tenant metadata; skipping token usage metrics");
            return Ok(FilterAction::Continue);
        }

        let tenant = self.tenant_label(tenant);
        let model = self.model_label(resolve_model(ctx));
        ctx.insert_filter_state(MetricsRequestState { model, tenant });
        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        let Some(state) = ctx.remove_filter_state::<MetricsRequestState>() else {
            return Ok(FilterAction::Continue);
        };
        emit_token_counters(ctx, &state);
        Ok(FilterAction::Continue)
    }
}

fn resolve_model<'a>(ctx: &'a HttpFilterContext<'_>) -> Option<&'a str> {
    ctx.get_metadata("openai_responses_request.model")
        .or_else(|| ctx.get_metadata("openai_chat_completions_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_request.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_to_chat_completions.model"))
        .or_else(|| ctx.get_metadata("model_to_header.model"))
        .or_else(|| ctx.get_metadata(META_METERING_MODEL))
        .filter(|value| !value.is_empty() && is_promotable_value(value))
}

fn emit_token_counters(ctx: &HttpFilterContext<'_>, state: &MetricsRequestState) {
    if ctx.get_metadata(META_TOKEN_STATUS) == Some(TOKEN_STATUS_OVERFLOW) {
        debug!("token capture overflow, skipping token usage metrics");
        return;
    }
    let (Some(input), Some(output), Some(total)) = (
        read_token_meta(ctx, META_TOKEN_INPUT),
        read_token_meta(ctx, META_TOKEN_OUTPUT),
        read_token_meta(ctx, META_TOKEN_TOTAL),
    ) else {
        return;
    };
    increment_tokens(state, "input", input);
    increment_tokens(state, "output", output);
    increment_tokens(state, "total", total);
    debug!(tenant = %state.tenant, model = %state.model, input, output, total, "recorded per-tenant token usage metrics");
}

fn increment_tokens(state: &MetricsRequestState, usage_type: &'static str, value: u64) {
    counter!(
        METRIC_TOKENS_TOTAL,
        "tenant" => state.tenant.clone(),
        "model" => state.model.clone(),
        "kind" => usage_type,
    )
    .increment(value);
}

fn read_token_meta(ctx: &HttpFilterContext<'_>, key: &str) -> Option<u64> {
    ctx.get_metadata(key).and_then(|value| value.parse().ok())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use bytes::Bytes;
    use praxis_filter::{FilterAction, HttpFilter as _};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    const BASE_CONFIG: &str = "allowed_tenants: [alice, bob]\nallowed_models: [gpt-4o, claude-sonnet-4]\n";

    fn filter() -> TokenUsageMetricsFilter {
        from_yaml(BASE_CONFIG)
    }

    fn from_yaml(yaml: &str) -> TokenUsageMetricsFilter {
        let value = serde_yaml::from_str(yaml).expect("test config should parse");
        TokenUsageMetricsFilter::build(&value).expect("test config should build")
    }

    fn set_counts(ctx: &mut HttpFilterContext<'_>, input: u64, output: u64, total: u64) {
        ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
        ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
        ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
    }

    async fn request(filter: &TokenUsageMetricsFilter, ctx: &mut HttpFilterContext<'_>) {
        ctx.current_filter_id = Some(0);
        let action = filter.on_request(ctx).await.expect("request hook should succeed");
        assert!(matches!(action, FilterAction::Continue), "filter should continue");
    }

    type Snapshot = Vec<(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<SharedString>,
        metrics_util::debugging::DebugValue,
    )>;

    fn emit(filter: &TokenUsageMetricsFilter, ctx: &mut HttpFilterContext<'_>) -> Snapshot {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let mut body = None;
            drop(
                filter
                    .on_response_body(ctx, &mut body, true)
                    .expect("response hook should succeed"),
            );
        });
        snapshotter.snapshot().into_vec()
    }

    fn value(snapshot: &Snapshot, tenant: &str, model: &str, kind: &str) -> Option<u64> {
        snapshot.iter().find_map(|(key, _, _, value)| {
            let labels: Vec<_> = key.key().labels().collect();
            let matches = key.key().name() == METRIC_TOKENS_TOTAL
                && labels
                    .iter()
                    .any(|label| label.key() == "tenant" && label.value() == tenant)
                && labels
                    .iter()
                    .any(|label| label.key() == "model" && label.value() == model)
                && labels
                    .iter()
                    .any(|label| label.key() == "kind" && label.value() == kind);
            match (matches, value) {
                (true, metrics_util::debugging::DebugValue::Counter(value)) => Some(*value),
                (true, other) => panic!("expected counter, got {other:?}"),
                (false, _) => None,
            }
        })
    }

    #[test]
    fn rejects_missing_allowlists_and_namespaced_key() {
        assert!(
            TokenUsageMetricsFilter::from_config(&serde_yaml::Value::Null).is_err(),
            "allowlists are required"
        );
        let yaml = format!("{BASE_CONFIG}tenant_metadata_key: identity.x-tenant-username\n");
        let value = serde_yaml::from_str(&yaml).expect("test config should parse");
        assert!(
            TokenUsageMetricsFilter::from_config(&value).is_err(),
            "namespaced key must fail"
        );
    }

    #[test]
    fn rejects_duplicate_and_reserved_labels() {
        for yaml in [
            "allowed_tenants: [alice, alice]\nallowed_models: [gpt-4o]\n",
            "allowed_tenants: [other]\nallowed_models: [gpt-4o]\n",
            "allowed_tenants: [alice]\nallowed_models: [unknown]\noverflow_model: unknown\n",
        ] {
            let value = serde_yaml::from_str(yaml).expect("test config should parse");
            assert!(
                TokenUsageMetricsFilter::from_config(&value).is_err(),
                "invalid config should fail: {yaml}"
            );
        }
    }

    #[test]
    fn rejects_excessive_cross_product() {
        let tenants = (0..100).map(|i| format!("tenant-{i}")).collect();
        let models = (0..100).map(|i| format!("model-{i}")).collect();
        let cfg = TokenUsageMetricsConfig {
            tenant_metadata_key: default_tenant_metadata_key(),
            allowed_tenants: tenants,
            allowed_models: models,
            overflow_tenant: default_overflow_tenant(),
            overflow_model: default_overflow_model(),
        };
        let err = validate_config(&cfg).expect_err("excessive series cross-product should fail");
        assert!(err.to_string().contains("series limit"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn emits_verified_allowlisted_counters() {
        let filter = filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(DEFAULT_TENANT_METADATA_KEY, "alice");
        ctx.set_metadata("openai_chat_completions_request.model", "gpt-4o");
        request(&filter, &mut ctx).await;
        set_counts(&mut ctx, 10, 20, 30);
        let snapshot = emit(&filter, &mut ctx);
        assert_eq!(value(&snapshot, "alice", "gpt-4o", "input"), Some(10));
        assert_eq!(value(&snapshot, "alice", "gpt-4o", "output"), Some(20));
        assert_eq!(value(&snapshot, "alice", "gpt-4o", "total"), Some(30));
    }

    #[tokio::test]
    async fn maps_verified_non_allowlisted_values_to_overflow() {
        let filter = filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(DEFAULT_TENANT_METADATA_KEY, "carol");
        ctx.set_metadata("openai_chat_completions_request.model", "novel-model");
        request(&filter, &mut ctx).await;
        set_counts(&mut ctx, 1, 2, 3);
        let snapshot = emit(&filter, &mut ctx);
        assert_eq!(value(&snapshot, "other", "other", "total"), Some(3));
        assert_eq!(value(&snapshot, "carol", "novel-model", "total"), None);
    }

    #[tokio::test]
    async fn ignores_guard_metadata_and_raw_headers() {
        let filter = filter();
        let mut req = make_request(http::Method::POST, "/v1/chat/completions");
        req.headers
            .insert("x-tenant-username", "mallory".parse().expect("valid header"));
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("identity.x-tenant-username", "mallory");
        request(&filter, &mut ctx).await;
        set_counts(&mut ctx, 1, 2, 3);
        assert!(
            emit(&filter, &mut ctx).is_empty(),
            "unverified identity must not emit metrics"
        );
        assert!(
            ctx.request_headers_to_remove.is_empty(),
            "metrics must not consume raw identity headers"
        );
    }

    #[tokio::test]
    async fn skips_invalid_identity_and_incomplete_or_overflowed_counts() {
        for mode in ["invalid", "incomplete", "overflow"] {
            let filter = filter();
            let req = make_request(http::Method::POST, "/v1/chat/completions");
            let mut ctx = make_filter_context(&req);
            ctx.set_metadata(
                DEFAULT_TENANT_METADATA_KEY,
                if mode == "invalid" { "bad\ntenant" } else { "alice" },
            );
            request(&filter, &mut ctx).await;
            if mode == "overflow" {
                set_counts(&mut ctx, 1, 2, 3);
                ctx.set_metadata(META_TOKEN_STATUS, TOKEN_STATUS_OVERFLOW);
            }
            assert!(
                emit(&filter, &mut ctx).is_empty(),
                "{mode} request must not emit metrics"
            );
        }
    }

    #[tokio::test]
    async fn waits_for_end_of_stream() {
        let filter = filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(DEFAULT_TENANT_METADATA_KEY, "alice");
        request(&filter, &mut ctx).await;
        set_counts(&mut ctx, 10, 20, 30);
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let mut body = Some(Bytes::from_static(b"chunk"));
            drop(
                filter
                    .on_response_body(&mut ctx, &mut body, false)
                    .expect("response hook should succeed"),
            );
        });
        assert!(
            snapshotter.snapshot().into_vec().is_empty(),
            "must wait for end of stream"
        );
    }
}
