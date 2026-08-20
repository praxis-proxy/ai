// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token quota admission with local or shared sliding-window accounting.
//!
//! The minimum POC uses trusted principal metadata plus a bounded, configured
//! model allowlist from the `X-Model` request header as the local quota key.
//!
//! ```yaml
//! key:
//!   principal:
//!     source: metadata
//!     name: identity.user_id
//!     onMissing: reject
//!   model:
//!     source: header
//!     name: x-model
//!     onMissing: reject
//!     allowedModels: [model-a, model-b]
//! reservationTimeout: 2m
//! limits:
//!   maxKeys: 10000
//!   maxKeyLength: 256
//!   maxActiveReservations: 50000
//! rules:
//!   - name: default
//!     estimation: { strategy: fixed, tokens: 500 }
//!     token_budgets:
//!       - { window: 1m, capacity: 3000 }
//! backend:
//!   kind: valkey
//!   url: ${TOKEN_RATE_LIMIT_VALKEY_URL}
//!   namespace: praxis:token-rate-limit
//! ```
//!
//! Put this filter after authentication and before `intelligent_route`, with
//! `token_count` before it in the response-body lifecycle. Admission reserves
//! the fixed estimate atomically across all configured windows. Successful
//! responses remain reserved until the terminal body callback publishes
//! `token.total`; failures and missing usage are charged conservatively at the
//! estimate. Reconciliation is idempotent.
//!
//! The memory backend is process-local. The Valkey backend shares the same
//! rule-scoped quota across gateway processes, uses a reconnecting connection
//! manager, cached Lua scripts, one-second accounting buckets, and fails
//! closed when Valkey is unavailable. Valkey URLs may use `rediss://`; deploy
//! the matching Rustls trust roots and verify the server certificate. A
//! private network or password alone does not encrypt credentials in transit.
//!
//! `maxKeys`, `maxKeyLength`, and `maxActiveReservations` are per configured
//! rule for both backends. The final principal-plus-model key is validated
//! before state allocation. Window sizes are bounded to 4,096 one-second
//! buckets so the shared script has explicit, bounded work and cardinality.

#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    clippy::multiple_inherent_impl,
    reason = "private configuration schema is covered by the public filter contract and focused tests"
)]

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, reason = "tests")]
mod tests;

mod backend;
mod ledger;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use async_trait::async_trait;
use http::header::HeaderName;
use metrics::{counter, gauge};
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config,
};
use serde::Deserialize;

use self::{
    backend::{
        BackendReserve, BackendSettlement, InMemoryTokenRateLimitBackend, ReconcileRequest, ReserveRequest,
        TokenRateLimitStateBackend, ValkeyBackendConfig, ValkeyTokenRateLimitBackend,
    },
    ledger::{Budget, Ledger, LedgerConfig},
};

const META_RESERVATION_ID: &str = "token_rate_limit.reservation_id";
const META_ACTIVE: &str = "token_rate_limit.active";
const META_KEY: &str = "token_rate_limit.key";
const META_TOKEN_TOTAL: &str = "token.total";
const MAX_RULES: usize = 32;
const MAX_BUDGETS: usize = 8;
const MAX_DURATION_MS: u64 = 7 * 24 * 60 * 60 * 1000;
const BUCKET_MS: u64 = 1_000;
const MAX_BUCKETS_PER_WINDOW: u64 = 4_096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenRateLimitConfig {
    key: KeyConfig,
    #[serde(rename = "reservationTimeout", alias = "reservation_timeout")]
    reservation_timeout: String,
    limits: LimitsConfig,
    rules: Vec<RuleConfig>,
    #[serde(default)]
    backend: BackendConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackendConfig {
    #[serde(default)]
    kind: BackendKind,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum BackendKind {
    #[default]
    Memory,
    Valkey,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyConfig {
    principal: PrincipalKeyConfig,
    model: ModelKeyConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrincipalKeyConfig {
    source: KeySource,
    name: String,
    #[serde(rename = "onMissing", alias = "on_missing")]
    on_missing: MissingKeyAction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelKeyConfig {
    source: ModelKeySource,
    name: String,
    #[serde(rename = "onMissing", alias = "on_missing")]
    on_missing: MissingKeyAction,
    #[serde(rename = "allowedModels", alias = "allowed_models")]
    allowed_models: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum KeySource {
    Metadata,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ModelKeySource {
    Header,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MissingKeyAction {
    Reject,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimationConfig {
    strategy: EstimationStrategy,
    tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum EstimationStrategy {
    Fixed,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LimitsConfig {
    #[serde(rename = "maxKeys", alias = "max_keys")]
    max_keys: usize,
    #[serde(rename = "maxKeyLength", alias = "max_key_length")]
    max_key_length: usize,
    #[serde(rename = "maxActiveReservations", alias = "max_active_reservations")]
    max_active_reservations: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleConfig {
    name: String,
    #[serde(rename = "match")]
    rule_match: Option<RuleMatchConfig>,
    estimation: EstimationConfig,
    #[serde(rename = "token_budgets", alias = "budgets")]
    token_budgets: Vec<BudgetConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleMatchConfig {
    metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetConfig {
    window: String,
    capacity: u64,
}

struct RuleRuntime {
    match_value: Option<String>,
    backend: Arc<dyn TokenRateLimitStateBackend>,
    estimate: u64,
}

/// Token quota admission with local or shared sliding-window accounting.
///
/// Place this filter after authentication and before `intelligent_route`.
/// `token_count` must publish `token.total` before the terminal response-body
/// callback so successful reservations reconcile actual usage. Failures and
/// missing usage are charged conservatively at the fixed estimate.
///
/// The memory backend is process-local. Valkey shares one rule-scoped quota
/// across gateway processes, uses a reconnecting connection manager and
/// cached Lua scripts, and fails closed when unavailable. Use `rediss://` with
/// Rustls trust roots when credentials must be protected in transit.
///
/// The configured key and active-reservation limits are per rule for both
/// backends. The final principal/model key is validated before allocation.
/// Windows are represented with bounded one-second buckets; a window may use
/// at most 4,096 buckets.
pub struct TokenRateLimitFilter {
    principal_key_name: String,
    model_header: HeaderName,
    allowed_models: HashSet<String>,
    max_key_length: usize,
    rules: Vec<RuleRuntime>,
    epoch: Instant,
}

impl TokenRateLimitFilter {
    fn from_config_inner(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg: TokenRateLimitConfig = parse_filter_config("token_rate_limit", config)?;
        if cfg.key.principal.name.is_empty() || cfg.key.principal.name.len() > 64 {
            return Err("token_rate_limit: key.principal.name must be 1-64 bytes".into());
        }
        if !matches!(cfg.key.principal.source, KeySource::Metadata)
            || !matches!(cfg.key.principal.on_missing, MissingKeyAction::Reject)
            || !matches!(cfg.key.model.source, ModelKeySource::Header)
            || !matches!(cfg.key.model.on_missing, MissingKeyAction::Reject)
        {
            return Err("token_rate_limit: only metadata/header with reject keying is supported".into());
        }
        let model_header = HeaderName::try_from(cfg.key.model.name.as_str())
            .map_err(|error| format!("token_rate_limit: invalid key.model.name: {error}"))?;
        let allowed_model_count = cfg.key.model.allowed_models.len();
        if allowed_model_count == 0 || allowed_model_count > MAX_RULES {
            return Err("token_rate_limit: key.model.allowedModels must contain 1-32 models".into());
        }
        let allowed_models = cfg.key.model.allowed_models.into_iter().collect::<HashSet<_>>();
        if allowed_models.len() != allowed_model_count
            || allowed_models.iter().any(|model| model.is_empty() || model.len() > 256)
        {
            return Err("token_rate_limit: key.model.allowedModels must contain unique 1-256 byte models".into());
        }
        if cfg.rules.is_empty() || cfg.rules.len() > MAX_RULES {
            return Err(format!("token_rate_limit: rules must contain 1-{MAX_RULES} entries").into());
        }
        if cfg.limits.max_key_length == 0 || cfg.limits.max_key_length > 256 {
            return Err("token_rate_limit: limits.max_key_length must be 1-256".into());
        }
        let mut names = HashSet::new();
        let mut match_values = HashSet::new();
        let timeout_ms = parse_duration_ms(&cfg.reservation_timeout)?;
        let mut rules = Vec::with_capacity(cfg.rules.len());
        let rule_count = cfg.rules.len();
        for (rule_index, rule) in cfg.rules.into_iter().enumerate() {
            if rule.name.is_empty() || !names.insert(rule.name.clone()) {
                return Err("token_rate_limit: rule names must be non-empty and unique".into());
            }
            if !matches!(rule.estimation.strategy, EstimationStrategy::Fixed) || rule.estimation.tokens == 0 {
                return Err(format!(
                    "token_rate_limit: rule '{}' must use fixed positive estimation",
                    rule.name
                )
                .into());
            }
            if rule.token_budgets.is_empty() || rule.token_budgets.len() > MAX_BUDGETS {
                return Err(format!("token_rate_limit: each rule must contain 1-{MAX_BUDGETS} budgets").into());
            }
            let match_value = match rule.rule_match {
                None => {
                    if rule_index + 1 != rule_count {
                        return Err("token_rate_limit: the default rule must be last".into());
                    }
                    None
                },
                Some(rule_match) => {
                    if rule_match.metadata.len() != 1 {
                        return Err("token_rate_limit: each rule match must contain exactly one metadata key".into());
                    }
                    let Some(value) = rule_match.metadata.get(&cfg.key.principal.name) else {
                        return Err("token_rate_limit: rule match must use the configured principal metadata".into());
                    };
                    if value.is_empty()
                        || value.len() > cfg.limits.max_key_length
                        || !match_values.insert(value.clone())
                    {
                        return Err("token_rate_limit: rule match values must be non-empty and unique".into());
                    }
                    Some(value.clone())
                },
            };
            let budgets = rule
                .token_budgets
                .iter()
                .map(parse_budget)
                .collect::<Result<Vec<_>, _>>()?;
            let backend_budgets = budgets.clone();
            let ledger = Ledger::new(LedgerConfig {
                budgets,
                reservation_timeout_ms: timeout_ms,
                max_keys: cfg.limits.max_keys,
                max_key_length: cfg.limits.max_key_length,
                max_active_reservations: cfg.limits.max_active_reservations,
            })
            .map_err(|e| format!("token_rate_limit: {e}"))?;
            let backend: Arc<dyn TokenRateLimitStateBackend> = match cfg.backend.kind {
                BackendKind::Memory => Arc::new(InMemoryTokenRateLimitBackend::new(ledger)),
                BackendKind::Valkey => {
                    let url = cfg
                        .backend
                        .url
                        .as_deref()
                        .ok_or("token_rate_limit: backend.url is required for Valkey")?;
                    let url = expand_backend_url(url)?;
                    let namespace = cfg.backend.namespace.as_deref().unwrap_or("praxis:trl");
                    Arc::new(ValkeyTokenRateLimitBackend::new(ValkeyBackendConfig {
                        url,
                        namespace: namespace.to_owned(),
                        rule: rule.name.clone(),
                        budgets: backend_budgets,
                        reservation_timeout_ms: timeout_ms,
                        max_keys: cfg.limits.max_keys,
                        max_active_reservations: cfg.limits.max_active_reservations,
                    })?)
                },
            };
            rules.push(RuleRuntime {
                match_value,
                backend,
                estimate: rule.estimation.tokens,
            });
        }
        Ok(Self {
            principal_key_name: cfg.key.principal.name,
            model_header,
            allowed_models,
            max_key_length: cfg.limits.max_key_length,
            rules,
            epoch: Instant::now(),
        })
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis().min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
    }

    fn matching_rule(&self, key: &str) -> Option<usize> {
        self.rules
            .iter()
            .position(|rule| rule.match_value.as_deref().is_some_and(|expected| expected == key))
            .or_else(|| self.rules.iter().position(|rule| rule.match_value.is_none()))
    }

    async fn reconcile(&self, ctx: &mut HttpFilterContext<'_>, actual: Option<u64>) {
        let Some((rule_index, id)) = ctx
            .get_metadata(META_RESERVATION_ID)
            .and_then(|value| value.split_once(':'))
            .and_then(|(rule, reservation)| Some((rule.parse::<usize>().ok()?, reservation.parse::<u64>().ok()?)))
        else {
            return;
        };
        let Some(key) = ctx.get_metadata(META_KEY).map(str::to_owned) else {
            return;
        };
        if let Some(rule) = self.rules.get(rule_index) {
            match rule
                .backend
                .reconcile(ReconcileRequest {
                    key,
                    reservation_id: id,
                    actual,
                    estimate: rule.estimate,
                    now_ms: self.now_ms(),
                })
                .await
            {
                Ok(BackendSettlement::Applied {
                    actual,
                    refund,
                    overage,
                }) => {
                    counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "reconciled").increment(1);
                    counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "actual").increment(actual);
                    counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "refunded").increment(refund);
                    counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "overage").increment(overage);
                },
                Ok(BackendSettlement::Noop) => {},
                Err(error) => {
                    counter!("praxis_ai_token_rate_limit_backend_errors_total", "backend" => "valkey", "operation" => "reconcile").increment(1);
                    tracing::error!(%error, "token-rate-limit response reconciliation failed");
                },
            }
        }
        ctx.filter_metadata.remove(META_RESERVATION_ID);
        ctx.filter_metadata.remove(META_ACTIVE);
        ctx.filter_metadata.remove(META_KEY);
        self.record_state_metrics();
    }

    fn record_state_metrics(&self) {
        let local = self
            .rules
            .iter()
            .filter_map(|rule| rule.backend.local_state().map(|(ledger, _)| ledger))
            .collect::<Vec<_>>();
        let active = local.iter().map(|ledger| ledger.active_count()).sum::<usize>();
        let keys = local.iter().map(|ledger| ledger.key_count()).sum::<usize>();
        #[expect(
            clippy::cast_precision_loss,
            reason = "metrics gauges use f64 and configured bounds keep values practical"
        )]
        gauge!("praxis_ai_token_rate_limit_active_reservations").set(active as f64);
        #[expect(
            clippy::cast_precision_loss,
            reason = "metrics gauges use f64 and configured bounds keep values practical"
        )]
        gauge!("praxis_ai_token_rate_limit_active_keys").set(keys as f64);
    }

    /// Construct the filter from YAML configuration.
    pub(crate) fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self::from_config_inner(config)?))
    }
}

fn expand_backend_url(url: &str) -> Result<String, FilterError> {
    let Some(start) = url.find("${") else {
        return Ok(url.to_owned());
    };
    let Some(name) = url.strip_prefix("${").and_then(|value| value.strip_suffix('}')) else {
        return Err("token_rate_limit: backend.url supports one complete ${ENV_VAR} reference".into());
    };
    if start != 0 || name.contains("${") {
        return Err("token_rate_limit: backend.url supports one complete ${ENV_VAR} reference".into());
    }
    if name.is_empty()
        || !name
            .bytes()
            .enumerate()
            .all(|(index, byte)| byte == b'_' || byte.is_ascii_uppercase() || (index > 0 && byte.is_ascii_digit()))
    {
        return Err("token_rate_limit: backend.url contains an invalid environment variable reference".into());
    }
    std::env::var(name).map_err(|_error| "token_rate_limit: backend.url environment variable is not set".into())
}

impl std::fmt::Debug for TokenRateLimitFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenRateLimitFilter")
            .field("principal_key_name", &self.principal_key_name)
            .field("model_header", &self.model_header)
            .field("rules", &self.rules.len())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl HttpFilter for TokenRateLimitFilter {
    fn name(&self) -> &'static str {
        "token_rate_limit"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let now_ms = self.now_ms();
        for rule in &self.rules {
            let Some((ledger, _)) = rule.backend.local_state() else {
                continue;
            };
            let orphaned = ledger.cleanup(now_ms, 1);
            counter!("praxis_ai_token_rate_limit_cleanup_total").increment(1);
            if orphaned > 0 {
                counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "orphaned")
                    .increment(orphaned as u64);
            }
        }
        self.record_state_metrics();
        let Some(principal) = ctx.get_metadata(&self.principal_key_name).map(str::to_owned) else {
            counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "missing_identity").increment(1);
            return Ok(FilterAction::Reject(Rejection::status(401)));
        };
        let Some(model) = ctx
            .request
            .headers
            .get(&self.model_header)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty() && value.len() <= 256)
        else {
            counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "missing_model")
                .increment(1);
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };
        if !self.allowed_models.contains(model) {
            counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "unknown_model")
                .increment(1);
            return Ok(FilterAction::Reject(Rejection::status(404)));
        }
        let key = format!("{}:{}:{}", principal.len(), principal, model);
        if key.len() > self.max_key_length {
            counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "key_too_long")
                .increment(1);
            return Ok(FilterAction::Reject(Rejection::status(400)));
        }
        let Some(rule_index) = self.matching_rule(&principal) else {
            counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "no_rule")
                .increment(1);
            return Ok(FilterAction::Reject(Rejection::status(403)));
        };
        let rule = &self.rules[rule_index];
        match rule
            .backend
            .reserve(ReserveRequest {
                key: key.clone(),
                estimate: rule.estimate,
                now_ms,
            })
            .await
        {
            Ok(BackendReserve::Admitted {
                reservation_id,
                estimate,
            }) => {
                counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "admitted", "reason" => "reserved")
                    .increment(1);
                counter!("praxis_ai_token_rate_limit_reservations_total", "result" => "created").increment(1);
                counter!("praxis_ai_token_rate_limit_tokens_total", "kind" => "estimated").increment(estimate);
                ctx.set_metadata(META_RESERVATION_ID, format!("{rule_index}:{reservation_id}"));
                ctx.set_metadata(META_ACTIVE, "true");
                ctx.set_metadata(META_KEY, &key);
                if ctx.get_metadata(META_RESERVATION_ID).is_none() {
                    drop(rule.backend.enqueue_reconcile(ReconcileRequest {
                        key: key.clone(),
                        reservation_id,
                        actual: None,
                        estimate: rule.estimate,
                        now_ms,
                    }));
                    counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "metadata_capacity")
                        .increment(1);
                    return Ok(FilterAction::Reject(Rejection::status(500)));
                }
                self.record_state_metrics();
                Ok(FilterAction::Continue)
            },
            Ok(BackendReserve::Denied { retry_after_ms }) => {
                counter!("praxis_ai_token_rate_limit_requests_total", "decision" => "denied", "reason" => "capacity")
                    .increment(1);
                let seconds = retry_after_ms.saturating_add(999) / 1000;
                Ok(FilterAction::Reject(
                    Rejection::status(429)
                        .with_header("Retry-After", seconds.max(1).to_string())
                        .with_header("X-RateLimit-Limit", rule.backend.limit().to_string())
                        .with_header("X-RateLimit-Remaining", "0")
                        .with_header("X-RateLimit-Reset", seconds.max(1).to_string()),
                ))
            },
            Err(error) => {
                counter!("praxis_ai_token_rate_limit_backend_errors_total", "backend" => "valkey", "operation" => "reserve").increment(1);
                tracing::error!(%error, "token-rate-limit admission backend failed");
                Ok(FilterAction::Reject(Rejection::status(503)))
            },
        }
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let success = ctx
            .response_header
            .as_ref()
            .is_some_and(|response| response.status.is_success());
        if !success {
            // Successful responses remain reserved until the terminal body
            // hook, where token_count has published actual usage. Failures
            // have no reliable body usage and are charged conservatively now.
            self.reconcile(ctx, None).await;
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<bytes::Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            let actual = ctx
                .get_metadata(META_TOKEN_TOTAL)
                .and_then(|value| value.parse::<u64>().ok());
            let Some((rule_index, id)) = ctx
                .get_metadata(META_RESERVATION_ID)
                .and_then(|value| value.split_once(':'))
                .and_then(|(rule, reservation)| Some((rule.parse::<usize>().ok()?, reservation.parse::<u64>().ok()?)))
            else {
                return Ok(FilterAction::Continue);
            };
            let Some(rule) = self.rules.get(rule_index) else {
                return Ok(FilterAction::Continue);
            };
            if let Some((ledger, _)) = rule.backend.local_state() {
                let _ = ledger.reconcile(id, actual, self.now_ms());
                ctx.filter_metadata.remove(META_RESERVATION_ID);
                ctx.filter_metadata.remove(META_ACTIVE);
                ctx.filter_metadata.remove(META_KEY);
            } else if let Some(key) = ctx.get_metadata(META_KEY).map(str::to_owned) {
                rule.backend
                    .enqueue_reconcile(ReconcileRequest {
                        key,
                        reservation_id: id,
                        actual,
                        estimate: rule.estimate,
                        now_ms: self.now_ms(),
                    })
                    .map_err(|error| -> FilterError { error.into() })?;
                ctx.filter_metadata.remove(META_RESERVATION_ID);
                ctx.filter_metadata.remove(META_ACTIVE);
                ctx.filter_metadata.remove(META_KEY);
            }
        }
        Ok(FilterAction::Continue)
    }
}

fn parse_budget(config: &BudgetConfig) -> Result<Budget, FilterError> {
    if config.capacity == 0 {
        return Err("token_rate_limit: budget capacity must be positive".into());
    }
    let window_ms = parse_duration_ms(&config.window)?;
    if window_ms.div_ceil(BUCKET_MS) > MAX_BUCKETS_PER_WINDOW {
        return Err(format!(
            "token_rate_limit: budget window cannot exceed {MAX_BUCKETS_PER_WINDOW} one-second buckets"
        )
        .into());
    }
    Ok(Budget {
        window_ms,
        capacity: config.capacity,
    })
}

fn parse_duration_ms(value: &str) -> Result<u64, FilterError> {
    let value = value.trim();
    let (number, multiplier) = if let Some(value) = value.strip_suffix("ms") {
        (value, 1_u64)
    } else if let Some(value) = value.strip_suffix('s') {
        (value, 1_000_u64)
    } else if let Some(value) = value.strip_suffix('m') {
        (value, 60_000_u64)
    } else if let Some(value) = value.strip_suffix('h') {
        (value, 3_600_000_u64)
    } else {
        return Err(format!("invalid duration '{value}'").into());
    };
    let amount = number
        .parse::<u64>()
        .map_err(|error| format!("invalid duration '{value}': {error}"))?;
    let millis = amount.checked_mul(multiplier).ok_or("duration overflow")?;
    if millis == 0 || millis > MAX_DURATION_MS {
        return Err("duration must be positive and bounded".into());
    }
    Ok(millis)
}
