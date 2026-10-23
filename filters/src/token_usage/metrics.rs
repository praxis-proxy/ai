// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-tenant Prometheus token usage metrics (POC / opt-in).
//!
//! Reads `token.*` metadata written by [`super::TokenCountFilter`] and
//! resolved tenant identity from filter metadata, then emits
//! `praxis_ai_tokens_total{tenant, model, kind}` counters.
//!
//! This filter is intentionally opt-in: it is off unless declared in
//! YAML. It is not a production usage dashboard. Cardinality is bounded
//! by `max_tenants`; additional tenants share the `overflow_tenant` label.

use std::{collections::HashSet, sync::Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use metrics::counter;
use praxis_ai_apis::promotion::is_promotable_value;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::Deserialize;
use tracing::{debug, trace};

use super::{META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_STATUS, META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW};
use crate::identity::{
    DEFAULT_IDENTITY_HEADER_PREFIX, DEFAULT_IDENTITY_METADATA_NAMESPACE, resolve_tenant_identity_from_metadata,
};
use crate::metering::{META_METERING_MODEL, META_METERING_USERNAME};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Prometheus counter name for billed token counts.
const METRIC_TOKENS_TOTAL: &str = "praxis_ai_tokens_total";

/// Default maximum distinct tenant labels before overflow aggregation.
const DEFAULT_MAX_TENANTS: usize = 32;

/// Label value for tenants beyond [`TokenUsageMetricsConfig::max_tenants`].
const DEFAULT_OVERFLOW_TENANT: &str = "other";

/// Label value when identity is missing or unsafe.
const TENANT_UNKNOWN: &str = "unknown";

/// Model label when no format-filter metadata is present or the value is unsafe.
const MODEL_UNKNOWN: &str = "unknown";

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for `token_usage_metrics`.
///
/// ```yaml
/// filter: token_usage_metrics
/// identity_header_prefix: "x-tenant-"
/// identity_metadata_namespace: "identity"
/// default_tenant: "unknown"
/// max_tenants: 32
/// overflow_tenant: other
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenUsageMetricsConfig {
    /// Prefix of tenant identity metadata keys. Must match
    /// `identity_header_guard` / `external_metering` when those filters run
    /// in the same pipeline.
    #[serde(default = "default_identity_header_prefix")]
    identity_header_prefix: String,

    /// Metadata namespace `identity_header_guard` writes captured identity
    /// under (tier 2 of identity resolution).
    #[serde(default = "default_identity_metadata_namespace")]
    identity_metadata_namespace: String,

    /// Fallback tenant label when no identity is resolved. When unset,
    /// requests without identity are not counted.
    #[serde(default)]
    default_tenant: Option<String>,

    /// Maximum distinct tenant labels before overflow aggregation.
    #[serde(default = "default_max_tenants")]
    max_tenants: usize,

    /// Label used for tenants beyond `max_tenants`.
    #[serde(default = "default_overflow_tenant")]
    overflow_tenant: String,
}

/// Serde default for `identity_header_prefix`.
fn default_identity_header_prefix() -> String {
    DEFAULT_IDENTITY_HEADER_PREFIX.to_owned()
}

/// Serde default for `identity_metadata_namespace`.
fn default_identity_metadata_namespace() -> String {
    DEFAULT_IDENTITY_METADATA_NAMESPACE.to_owned()
}

/// Serde default for `max_tenants`.
fn default_max_tenants() -> usize {
    DEFAULT_MAX_TENANTS
}

/// Serde default for `overflow_tenant`.
fn default_overflow_tenant() -> String {
    DEFAULT_OVERFLOW_TENANT.to_owned()
}

/// Validate config at construction time.
fn validate_config(cfg: &TokenUsageMetricsConfig) -> Result<(), FilterError> {
    if cfg.identity_header_prefix.is_empty() {
        return Err("token_usage_metrics: identity_header_prefix must not be empty".into());
    }
    if http::header::HeaderName::from_bytes(cfg.identity_header_prefix.as_bytes()).is_err() {
        return Err(
            "token_usage_metrics: identity_header_prefix must contain only valid HTTP header name characters".into(),
        );
    }
    if cfg.identity_metadata_namespace.is_empty() {
        return Err("token_usage_metrics: identity_metadata_namespace must not be empty".into());
    }
    if cfg.max_tenants == 0 {
        return Err("token_usage_metrics: max_tenants must be greater than 0".into());
    }
    if cfg.overflow_tenant.is_empty() || !is_promotable_value(&cfg.overflow_tenant) {
        return Err("token_usage_metrics: overflow_tenant must be a non-empty promotable label".into());
    }
    if let Some(tenant) = &cfg.default_tenant
        && (tenant.is_empty() || !is_promotable_value(tenant))
    {
        return Err("token_usage_metrics: default_tenant must be a non-empty promotable label".into());
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

/// Emits per-tenant Prometheus token counters from `token.*` metadata.
///
/// Opt-in POC for small known tenant sets. Off unless declared in YAML.
/// Distinct tenant labels are bounded by `max_tenants`; additional
/// tenants share `overflow_tenant`. Tenant identity must already be in
/// filter metadata, normally from `external_metering` or
/// `identity_header_guard`; raw request headers are never consumed directly.
/// Place either identity-producing filter before this one in the request
/// pipeline.
///
/// # YAML configuration
///
/// ```yaml
/// filter: token_usage_metrics
/// identity_header_prefix: "x-tenant-"
/// identity_metadata_namespace: "identity"
/// default_tenant: "unknown"
/// max_tenants: 32
/// overflow_tenant: other
/// ```
///
/// Declare this filter **before** `token_count` so response hooks (which
/// run in reverse order) extract counts first.
///
/// # Example
///
/// ```ignore
/// use praxis_ai_filters::TokenUsageMetricsFilter;
/// use praxis_filter::HttpFilter;
///
/// let filter = TokenUsageMetricsFilter::from_config(&serde_yaml::Value::Null).unwrap();
/// assert_eq!(filter.name(), "token_usage_metrics");
/// ```
pub struct TokenUsageMetricsFilter {
    /// Fallback tenant when identity is missing.
    default_tenant: Option<String>,

    /// Prefix of tenant identity headers, lowercased at construction.
    identity_header_prefix: String,

    /// Metadata namespace for `identity_header_guard` captured headers.
    identity_metadata_namespace: String,

    /// Maximum distinct tenant labels before overflow aggregation.
    max_tenants: usize,

    /// Label used for tenants beyond `max_tenants`.
    overflow_tenant: String,

    /// Distinct tenant ids that already occupy a Prometheus label slot.
    seen_tenants: Mutex<HashSet<String>>,
}

impl TokenUsageMetricsFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::build(config)?))
    }

    /// Build the concrete filter from parsed YAML config.
    fn build(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: TokenUsageMetricsConfig = parse_filter_config("token_usage_metrics", config)?;
        validate_config(&cfg)?;

        Ok(Self {
            default_tenant: cfg.default_tenant,
            identity_header_prefix: cfg.identity_header_prefix.to_ascii_lowercase(),
            identity_metadata_namespace: cfg.identity_metadata_namespace,
            max_tenants: cfg.max_tenants,
            overflow_tenant: cfg.overflow_tenant,
            seen_tenants: Mutex::new(HashSet::new()),
        })
    }
}

/// Per-request tenant captured during the request phase.
struct MetricsRequestState {
    /// Tenant username to label metrics with. Empty means skip.
    tenant: String,
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
        let tenant = ctx
            .get_metadata(META_METERING_USERNAME)
            .map(str::to_owned)
            .or_else(|| {
                let identity = resolve_tenant_identity_from_metadata(
                    ctx,
                    &self.identity_header_prefix,
                    &self.identity_metadata_namespace,
                );
                (!identity.username.is_empty()).then_some(identity.username)
            })
            .or_else(|| self.default_tenant.clone())
            .unwrap_or_default();

        if tenant.is_empty() {
            trace!("no tenant identity, skipping token metrics");
            return Ok(FilterAction::Continue);
        }

        ctx.insert_filter_state(MetricsRequestState { tenant });
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

        emit_token_counters(self, ctx, &state);
        Ok(FilterAction::Continue)
    }
}

/// Emit Prometheus counters when token metadata is complete and capture succeeded.
fn emit_token_counters(filter: &TokenUsageMetricsFilter, ctx: &HttpFilterContext<'_>, state: &MetricsRequestState) {
    if ctx.get_metadata(META_TOKEN_STATUS) == Some(TOKEN_STATUS_OVERFLOW) {
        debug!("token capture overflow, skipping token metrics");
        return;
    }

    let Some(input) = read_token_meta(ctx, META_TOKEN_INPUT) else {
        return;
    };
    let Some(output) = read_token_meta(ctx, META_TOKEN_OUTPUT) else {
        return;
    };
    let Some(total) = read_token_meta(ctx, META_TOKEN_TOTAL) else {
        return;
    };

    let tenant = bound_tenant(filter, &state.tenant);
    let model = resolve_model(ctx);

    increment_tokens(&tenant, &model, "input", input);
    increment_tokens(&tenant, &model, "output", output);
    increment_tokens(&tenant, &model, "total", total);

    debug!(
        tenant = %tenant,
        model = %model,
        input,
        output,
        total,
        "recorded per-tenant token usage metrics"
    );
}

/// Increment `praxis_ai_tokens_total` for one usage type.
fn increment_tokens(tenant: &str, model: &str, usage_type: &'static str, value: u64) {
    counter!(
        METRIC_TOKENS_TOTAL,
        "tenant" => tenant.to_owned(),
        "model" => model.to_owned(),
        "kind" => usage_type,
    )
    .increment(value);
}

/// Map a raw tenant id onto a bounded Prometheus label.
fn bound_tenant(filter: &TokenUsageMetricsFilter, raw: &str) -> String {
    if raw.is_empty() || !is_promotable_value(raw) {
        return TENANT_UNKNOWN.to_owned();
    }
    if raw == TENANT_UNKNOWN || raw == filter.overflow_tenant {
        return raw.to_owned();
    }
    let Ok(mut seen_tenants) = filter.seen_tenants.lock() else {
        // A poisoned bound must fail closed into the overflow bucket rather
        // than admitting unbounded new label values.
        return filter.overflow_tenant.clone();
    };
    if seen_tenants.contains(raw) {
        return raw.to_owned();
    }
    if seen_tenants.len() < filter.max_tenants {
        seen_tenants.insert(raw.to_owned());
        return raw.to_owned();
    }
    filter.overflow_tenant.clone()
}

/// Resolve the model label from format-filter metadata with fallback.
fn resolve_model(ctx: &HttpFilterContext<'_>) -> String {
    ctx.get_metadata("openai_responses_format.model")
        .or_else(|| ctx.get_metadata("anthropic_messages_format.model"))
        .or_else(|| ctx.get_metadata("anthropic_messages_to_chat_completions.model"))
        .or_else(|| ctx.get_metadata(META_METERING_MODEL))
        .filter(|v| is_promotable_value(v))
        .unwrap_or(MODEL_UNKNOWN)
        .to_owned()
}

/// Parse a token count from filter metadata.
fn read_token_meta(ctx: &HttpFilterContext<'_>, key: &str) -> Option<u64> {
    ctx.get_metadata(key).and_then(|v| v.parse().ok())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use bytes::Bytes;
    use http::header::HeaderName;
    use praxis_filter::{FilterAction, HttpFilter as _};

    use super::*;
    use crate::test_utils::{make_filter_context, make_request};

    fn default_filter() -> TokenUsageMetricsFilter {
        from_yaml("{}")
    }

    fn from_yaml(yaml: &str) -> TokenUsageMetricsFilter {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        TokenUsageMetricsFilter::build(&value).unwrap()
    }

    fn set_token_counts(ctx: &mut HttpFilterContext<'_>, input: u64, output: u64, total: u64) {
        ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
        ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
        ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
    }

    async fn run_request(filter: &TokenUsageMetricsFilter, ctx: &mut HttpFilterContext<'_>) {
        ctx.current_filter_id = Some(0);
        let action = filter.on_request(ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
    }

    type MetricSnapshot = Vec<(
        metrics_util::CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        metrics_util::debugging::DebugValue,
    )>;

    /// Snapshot once: `DebuggingRecorder::snapshot` resets counters via `swap(0)`.
    fn emit_with_metrics(filter: &TokenUsageMetricsFilter, ctx: &mut HttpFilterContext<'_>) -> MetricSnapshot {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let mut body = None;
            drop(filter.on_response_body(ctx, &mut body, true).unwrap());
        });
        snapshotter.snapshot().into_vec()
    }

    fn counter_value(snapshot: &MetricSnapshot, tenant: &str, model: &str, usage_kind: &str) -> Option<u64> {
        snapshot.iter().find_map(|(key, _, _, value)| {
            if key.key().name() != METRIC_TOKENS_TOTAL {
                return None;
            }
            let labels: Vec<_> = key.key().labels().collect();
            let tenant_ok = labels.iter().any(|l| l.key() == "tenant" && l.value() == tenant);
            let model_ok = labels.iter().any(|l| l.key() == "model" && l.value() == model);
            let kind_ok = labels.iter().any(|l| l.key() == "kind" && l.value() == usage_kind);
            if tenant_ok && model_ok && kind_ok {
                match value {
                    metrics_util::debugging::DebugValue::Counter(v) => Some(*v),
                    other => panic!("expected counter, got {other:?}"),
                }
            } else {
                None
            }
        })
    }

    #[test]
    fn from_config_defaults() {
        let filter = TokenUsageMetricsFilter::from_config(&serde_yaml::Value::Null).unwrap();
        assert_eq!(filter.name(), "token_usage_metrics");
    }

    #[test]
    fn from_config_rejects_unknown_fields() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("bogus: true").unwrap();
        assert!(TokenUsageMetricsFilter::from_config(&yaml).is_err());
    }

    #[test]
    fn from_config_rejects_zero_max_tenants() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("max_tenants: 0").unwrap();
        let err = TokenUsageMetricsFilter::from_config(&yaml).err().unwrap();
        assert!(err.to_string().contains("max_tenants"));
    }

    #[test]
    fn from_config_rejects_empty_prefix() {
        let yaml: serde_yaml::Value = serde_yaml::from_str("identity_header_prefix: \"\"").unwrap();
        assert!(TokenUsageMetricsFilter::from_config(&yaml).is_err());
    }

    #[tokio::test]
    async fn emits_counters_from_guard_metadata_identity() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("identity.x-tenant-username", "alice");
        ctx.set_metadata("openai_responses_format.model", "gpt-4o".to_owned());
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 10, 20, 30);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(counter_value(&snapshot, "alice", "gpt-4o", "input"), Some(10));
        assert_eq!(counter_value(&snapshot, "alice", "gpt-4o", "output"), Some(20));
        assert_eq!(counter_value(&snapshot, "alice", "gpt-4o", "total"), Some(30));
    }

    #[tokio::test]
    async fn skips_without_token_metadata() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("identity.x-tenant-username", "alice");
        run_request(&filter, &mut ctx).await;

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert!(snapshot.is_empty(), "no counters without token metadata");
    }

    #[tokio::test]
    async fn skips_without_identity_or_default() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 10, 20, 30);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert!(snapshot.is_empty(), "no counters without tenant identity");
    }

    #[tokio::test]
    async fn uses_default_tenant_when_identity_missing() {
        let filter = from_yaml("default_tenant: unknown");
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 1, 2, 3);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(counter_value(&snapshot, "unknown", MODEL_UNKNOWN, "total"), Some(3));
    }

    #[tokio::test]
    async fn verified_identity_wins_over_headers() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.filter_metadata
            .insert("x-tenant-username".to_owned(), "alice".to_owned());
        ctx.filter_metadata
            .insert("identity.x-tenant-username".to_owned(), "mallory".to_owned());
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 4, 5, 9);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(counter_value(&snapshot, "alice", MODEL_UNKNOWN, "total"), Some(9));
        assert_eq!(counter_value(&snapshot, "mallory", MODEL_UNKNOWN, "total"), None);
    }

    #[tokio::test]
    async fn guard_metadata_identity_emits_counters() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.filter_metadata
            .insert("identity.x-tenant-username".to_owned(), "bob".to_owned());
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 1, 1, 2);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(counter_value(&snapshot, "bob", MODEL_UNKNOWN, "total"), Some(2));
    }

    #[tokio::test]
    async fn external_metering_identity_wins_over_other_metadata() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata(META_METERING_USERNAME, "alice");
        ctx.set_metadata("identity.x-tenant-username", "mallory");
        ctx.set_metadata(META_METERING_MODEL, "gpt-4o");
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 2, 3, 5);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(counter_value(&snapshot, "alice", "gpt-4o", "total"), Some(5));
        assert_eq!(counter_value(&snapshot, "mallory", "gpt-4o", "total"), None);
    }

    #[tokio::test]
    async fn ignores_raw_identity_headers() {
        let filter = default_filter();
        let mut req = make_request(http::Method::POST, "/v1/chat/completions");
        req.headers.insert("x-tenant-username", "alice".parse().unwrap());
        req.headers.insert("authorization", "Bearer sk-test".parse().unwrap());
        let mut ctx = make_filter_context(&req);
        run_request(&filter, &mut ctx).await;

        let removed: Vec<&str> = ctx.request_headers_to_remove.iter().map(HeaderName::as_str).collect();
        assert!(
            removed.is_empty(),
            "metrics filter must not consume raw identity headers"
        );
        assert!(ctx.filter_state.is_empty(), "raw identity must not create metric state");
    }

    #[tokio::test]
    async fn overflow_tenants_share_other_label() {
        let filter = from_yaml("max_tenants: 1");
        for (user, total) in [("alice", 10_u64), ("bob", 20)] {
            let req = make_request(http::Method::POST, "/v1/chat/completions");
            let mut ctx = make_filter_context(&req);
            ctx.set_metadata("identity.x-tenant-username", user);
            run_request(&filter, &mut ctx).await;
            set_token_counts(&mut ctx, 0, 0, total);
            let snapshot = emit_with_metrics(&filter, &mut ctx);
            if user == "alice" {
                assert_eq!(counter_value(&snapshot, "alice", MODEL_UNKNOWN, "total"), Some(10));
            } else {
                assert_eq!(counter_value(&snapshot, "other", MODEL_UNKNOWN, "total"), Some(20));
                assert_eq!(counter_value(&snapshot, "bob", MODEL_UNKNOWN, "total"), None);
            }
        }
    }

    #[tokio::test]
    async fn unsafe_tenant_becomes_unknown() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.filter_metadata
            .insert("x-tenant-username".to_owned(), "bad\ntenant".to_owned());
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 1, 1, 2);

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert_eq!(
            counter_value(&snapshot, TENANT_UNKNOWN, MODEL_UNKNOWN, "total"),
            Some(2)
        );
    }

    #[tokio::test]
    async fn skips_token_status_overflow() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("identity.x-tenant-username", "alice");
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 10, 20, 30);
        ctx.set_metadata(META_TOKEN_STATUS, TOKEN_STATUS_OVERFLOW.to_owned());

        let snapshot = emit_with_metrics(&filter, &mut ctx);
        assert!(snapshot.is_empty(), "overflow capture must not emit counters");
    }

    #[tokio::test]
    async fn ignores_body_chunks_until_end_of_stream() {
        let filter = default_filter();
        let req = make_request(http::Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&req);
        ctx.set_metadata("identity.x-tenant-username", "alice");
        run_request(&filter, &mut ctx).await;
        set_token_counts(&mut ctx, 10, 20, 30);

        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let mut body = Some(Bytes::from_static(b"chunk"));
            drop(filter.on_response_body(&mut ctx, &mut body, false).unwrap());
        });
        assert!(
            snapshotter.snapshot().into_vec().is_empty(),
            "metrics must wait for end of stream"
        );
    }
}
