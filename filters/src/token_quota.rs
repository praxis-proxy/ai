// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! PPE host plugin for policy-selected weighted token quota.

use std::{collections::HashMap, sync::Arc};

use ppe::prelude::{
    AnyHookHandler, CmfHook, Extensions, HookHandler, MessagePayload, Plugin, PluginConfig, PluginContext, PluginError,
    PluginFactory, PluginInstance, PluginResult, TypedHandlerAdapter,
};
use praxis_filter::{HttpFilterContext, PolicyResponseContext, PolicyResponseOutput};
use serde::{Deserialize, Serialize};

use crate::token_usage::normalized::{
    NormalizedUsage, UsageCounts, UsageInput, weighted_cost as normalized_weighted_cost,
};

/// PPE plugin kind configured by operators.
pub(crate) const KIND: &str = "ai/token-quota";
/// Trusted custom namespace shared with PPE.
const CUSTOM_KEY: &str = "token_quota";
/// Maximum stable formula identifier length.
const MAX_FORMULA_ID_BYTES: usize = 64;
/// Maximum non-currency unit name length.
const MAX_UNIT_BYTES: usize = 32;

/// Operator configuration for one policy-selected formula plugin instance.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenQuotaConfig {
    /// Stable operator-owned formula identifier.
    formula_id: String,
    /// Non-currency result unit.
    #[serde(default = "default_unit")]
    unit: String,
    /// Per-partition weights.
    #[serde(default)]
    weights: TokenWeights,
}

/// Weights for normalized, non-overlapping token partitions.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TokenWeights {
    /// Uncached and unclassified input.
    input: f64,
    /// Cache-read input.
    cached_input: f64,
    /// Cache-write input.
    cache_write: f64,
    /// Visible output.
    output: f64,
    /// Reasoning output.
    reasoning: f64,
}

impl Default for TokenWeights {
    fn default() -> Self {
        Self {
            input: 1.0,
            cached_input: 1.0,
            cache_write: 1.0,
            output: 1.0,
            reasoning: 1.0,
        }
    }
}

/// Trusted input produced by `token_count` for the response policy.
#[derive(Debug, Serialize, Deserialize)]
struct TokenQuotaInput {
    /// Input availability status.
    status: TokenQuotaInputStatus,
    /// Normalized usage, when available.
    usage: Option<NormalizedUsage>,
    /// Trusted routing facts.
    context: RouteContext,
}

/// Availability of trusted usage supplied to the PPE plugin.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TokenQuotaInputStatus {
    /// Normalized usage is present.
    Ready,
    /// The provider did not report usable usage.
    MissingUsage,
    /// Usage capture exceeded its configured bound.
    UsageOverflow,
}

/// Evaluation outcome returned by PPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TokenQuotaResultStatus {
    /// The formula produced a quota value.
    Evaluated,
    /// The provider did not report usable usage.
    MissingUsage,
    /// Usage capture exceeded its configured bound.
    UsageOverflow,
}

impl TokenQuotaResultStatus {
    /// Stable status name used at generic metering boundaries.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Evaluated => "evaluated",
            Self::MissingUsage => "missing_usage",
            Self::UsageOverflow => "usage_overflow",
        }
    }
}

impl From<TokenQuotaInputStatus> for TokenQuotaResultStatus {
    fn from(status: TokenQuotaInputStatus) -> Self {
        match status {
            TokenQuotaInputStatus::Ready | TokenQuotaInputStatus::MissingUsage => Self::MissingUsage,
            TokenQuotaInputStatus::UsageOverflow => Self::UsageOverflow,
        }
    }
}

/// Stable PPE output consumed by rate limiting and metering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TokenQuotaResult {
    /// Evaluation outcome.
    pub(crate) status: TokenQuotaResultStatus,
    /// Rounded quota units on success.
    pub(crate) value: Option<u64>,
    /// Formula that produced the result.
    pub(crate) formula_id: String,
    /// Non-currency result unit.
    pub(crate) unit: String,
    /// Exact normalized usage supplied to the plugin.
    pub(crate) usage: Option<serde_json::Value>,
    /// Exact routing context supplied to the plugin.
    pub(crate) context: serde_json::Value,
}

/// Minimal PPE decision needed by rate-limit reconciliation.
#[cfg(feature = "token-rate-limit-filter")]
#[derive(Debug, Clone, Copy, Deserialize)]
pub(crate) struct TokenQuotaDecision {
    /// Evaluation outcome.
    pub(crate) status: TokenQuotaResultStatus,
    /// Rounded quota units on success.
    pub(crate) value: Option<u64>,
}

/// Bounded routing facts supplied to policy for audit and route selection.
#[derive(Debug, Serialize, Deserialize)]
struct RouteContext {
    /// Logical route name.
    route: String,
    /// Selected backend cluster.
    backend: String,
    /// Trusted provider or API format.
    provider: String,
    /// Effective routed model.
    model: String,
}

impl RouteContext {
    /// Capture bounded, trusted routing values from the filter context.
    fn from_context(ctx: &HttpFilterContext<'_>) -> Self {
        Self {
            route: ctx
                .get_metadata("intelligent_route.name")
                .or(ctx.metrics_route.as_deref())
                .unwrap_or("")
                .to_owned(),
            backend: ctx.cluster_name().unwrap_or("").to_owned(),
            provider: ctx
                .get_metadata("provider_route.provider_id")
                .or_else(|| ctx.get_metadata("intelligent_route.api_format"))
                .unwrap_or("")
                .to_owned(),
            model: ctx
                .get_metadata("provider_route.model")
                .or_else(|| ctx.get_metadata("intelligent_route.target_model"))
                .or_else(|| ctx.get_metadata("openai_responses_model_rewrite.effective_model"))
                .or_else(|| ctx.get_metadata("openai_responses_request.model"))
                .or_else(|| ctx.get_metadata("openai_chat_completions_request.model"))
                .or_else(|| ctx.get_metadata("anthropic_messages_request.model"))
                .or_else(|| ctx.get_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL))
                .unwrap_or("")
                .to_owned(),
        }
    }
}

/// PPE plugin that computes one configured formula over trusted usage input.
struct TokenQuotaPlugin {
    /// PPE instance configuration.
    plugin: PluginConfig,
    /// Validated formula configuration.
    config: TokenQuotaConfig,
}

impl Plugin for TokenQuotaPlugin {
    fn config(&self) -> &PluginConfig {
        &self.plugin
    }
}

impl HookHandler<CmfHook> for TokenQuotaPlugin {
    #[expect(clippy::too_many_lines, reason = "linear read, calculate, and publish transform")]
    async fn handle(
        &self,
        _payload: &MessagePayload,
        extensions: &Extensions,
        _ctx: &mut PluginContext,
    ) -> PluginResult<MessagePayload> {
        let Some(input) = extensions
            .custom
            .as_ref()
            .and_then(|custom| custom.get(CUSTOM_KEY))
            .and_then(|value| TokenQuotaInput::deserialize(value).ok())
        else {
            return PluginResult::allow();
        };

        let usage = input.usage.as_ref().and_then(|usage| serde_json::to_value(usage).ok());
        let context = serde_json::to_value(input.context).unwrap_or(serde_json::Value::Null);
        let (status, value) = input.usage.as_ref().map_or_else(
            || (TokenQuotaResultStatus::from(input.status), None),
            |usage| {
                (
                    TokenQuotaResultStatus::Evaluated,
                    Some(normalized_weighted_cost(
                        usage,
                        [
                            self.config.weights.input,
                            self.config.weights.cached_input,
                            self.config.weights.cache_write,
                            self.config.weights.output,
                            self.config.weights.reasoning,
                        ],
                    )),
                )
            },
        );
        let result = TokenQuotaResult {
            status,
            value,
            formula_id: self.config.formula_id.clone(),
            unit: self.config.unit.clone(),
            usage,
            context,
        };
        let Ok(value) = serde_json::to_value(result) else {
            return PluginResult::allow();
        };
        let mut modified = extensions.cow_copy();
        modified
            .custom
            .get_or_insert_default()
            .insert(CUSTOM_KEY.to_owned(), value);
        PluginResult::modify_extensions(modified)
    }
}

/// Factory registered into the Praxis policy filter before pipelines load.
pub(crate) struct TokenQuotaFactory;

impl PluginFactory for TokenQuotaFactory {
    fn create(&self, config: &PluginConfig) -> Result<PluginInstance, Box<PluginError>> {
        let raw = config
            .config
            .clone()
            .ok_or_else(|| config_error(config, "a `config:` block is required"))?;
        let typed: TokenQuotaConfig =
            serde_json::from_value(raw).map_err(|error| config_error(config, &format!("invalid config: {error}")))?;
        validate_config(config, &typed)?;
        let quota = Arc::new(TokenQuotaPlugin {
            plugin: config.clone(),
            config: typed,
        });
        let handler: Arc<dyn AnyHookHandler> = Arc::new(TypedHandlerAdapter::<CmfHook, _>::new(Arc::clone(&quota)));
        Ok(PluginInstance {
            plugin: quota,
            handlers: vec![(ppe::praxis_policy_core::cmf::constants::HOOK_CMF_LLM_OUTPUT, handler)],
        })
    }
}

/// Construct a consistently attributed PPE configuration error.
fn config_error(plugin: &PluginConfig, message: &str) -> Box<PluginError> {
    Box::new(PluginError::Config {
        message: format!("plugin '{}' (token quota): {message}", plugin.name),
    })
}

/// Validate bounded labels and non-negative finite weights.
fn validate_config(plugin: &PluginConfig, config: &TokenQuotaConfig) -> Result<(), Box<PluginError>> {
    validate_label(plugin, "formula_id", &config.formula_id, MAX_FORMULA_ID_BYTES)?;
    validate_label(plugin, "unit", &config.unit, MAX_UNIT_BYTES)?;
    for (name, value) in [
        ("input", config.weights.input),
        ("cached_input", config.weights.cached_input),
        ("cache_write", config.weights.cache_write),
        ("output", config.weights.output),
        ("reasoning", config.weights.reasoning),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(config_error(
                plugin,
                &format!("weights.{name} must be a finite number >= 0"),
            ));
        }
    }
    Ok(())
}

/// Validate a bounded printable operator label.
fn validate_label(plugin: &PluginConfig, field: &str, value: &str, max_bytes: usize) -> Result<(), Box<PluginError>> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(config_error(
            plugin,
            &format!("{field} must be 1..={max_bytes} bytes without control characters"),
        ));
    }
    Ok(())
}

/// Default non-currency quota unit.
fn default_unit() -> String {
    "quota_units".to_owned()
}

/// Register the AI quota plugin factory with the policy filter.
pub(crate) fn register_factory() {
    praxis_filter::register_policy_plugin_factory(KIND, Arc::new(|| Box::new(TokenQuotaFactory)));
}

/// Publish the latest normalized usage and trusted route context for PPE.
pub(crate) fn publish_policy_context(ctx: &mut HttpFilterContext<'_>) {
    let (status, usage) = match UsageCounts::from_context(ctx).normalize() {
        UsageInput::Available(usage) => (TokenQuotaInputStatus::Ready, Some(usage)),
        UsageInput::Missing => (TokenQuotaInputStatus::MissingUsage, None),
        UsageInput::Overflow => (TokenQuotaInputStatus::UsageOverflow, None),
    };
    let input = TokenQuotaInput {
        status,
        usage,
        context: RouteContext::from_context(ctx),
    };
    if let Ok(value) = serde_json::to_value(input) {
        ctx.extensions.insert(PolicyResponseContext::new(HashMap::from([(
            CUSTOM_KEY.to_owned(),
            value,
        )])));
    }
}

/// Read the quota result returned by PPE.
pub(crate) fn result(ctx: &HttpFilterContext<'_>) -> Option<TokenQuotaResult> {
    ctx.extensions
        .get::<PolicyResponseOutput>()
        .and_then(|output| output.get(CUSTOM_KEY))
        .and_then(|value| TokenQuotaResult::deserialize(value).ok())
}

/// Read only the quota decision needed by rate-limit reconciliation.
#[cfg(feature = "token-rate-limit-filter")]
pub(crate) fn decision(ctx: &HttpFilterContext<'_>) -> Option<TokenQuotaDecision> {
    ctx.extensions
        .get::<PolicyResponseOutput>()
        .and_then(|output| output.get(CUSTOM_KEY))
        .and_then(|value| TokenQuotaDecision::deserialize(value).ok())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests fail fast on fixtures"
)]
mod tests {
    use super::*;

    fn weighted_cost(usage: &NormalizedUsage, weights: TokenWeights) -> u64 {
        normalized_weighted_cost(
            usage,
            [
                weights.input,
                weights.cached_input,
                weights.cache_write,
                weights.output,
                weights.reasoning,
            ],
        )
    }

    fn usage() -> NormalizedUsage {
        match (UsageCounts {
            input: Some(120),
            output: Some(40),
            total: Some(160),
            cache_read: Some(90),
            cache_write: Some(20),
            reasoning: None,
            overflow: false,
        })
        .normalize()
        {
            UsageInput::Available(usage) => usage,
            UsageInput::Missing | UsageInput::Overflow => panic!("fixture must normalize"),
        }
    }

    #[test]
    fn weighted_formula_partitions_cache_and_rounds_up() {
        let weights = TokenWeights {
            input: 1.0 / 3.0,
            cached_input: 1.0 / 30.0,
            cache_write: 1.0 / 2.4,
            output: 1.0,
            reasoning: 1.0,
        };
        assert_eq!(weighted_cost(&usage(), weights), 55);
    }

    #[test]
    fn total_only_usage_keeps_legacy_unweighted_cost() {
        let usage = match (UsageCounts {
            total: Some(42),
            ..UsageCounts::default()
        })
        .normalize()
        {
            UsageInput::Available(usage) => usage,
            UsageInput::Missing | UsageInput::Overflow => panic!("fixture must normalize"),
        };
        let weights = TokenWeights {
            input: 1.0 / 3.0,
            ..TokenWeights::default()
        };
        assert_eq!(weighted_cost(&usage, weights), 42);
    }

    #[test]
    fn factory_rejects_invalid_weights() {
        let plugin = PluginConfig {
            name: "quota".to_owned(),
            kind: KIND.to_owned(),
            config: Some(serde_json::json!({
                "formula_id": "v1",
                "weights": {"input": -1.0}
            })),
            ..Default::default()
        };
        assert!(TokenQuotaFactory.create(&plugin).is_err());
    }

    #[cfg(feature = "token-rate-limit-filter")]
    #[test]
    fn ppe_formula_matches_the_legacy_weighted_cost() {
        let usage = match (UsageCounts {
            input: Some(100),
            output: Some(50),
            total: Some(150),
            cache_read: Some(10),
            cache_write: Some(5),
            reasoning: Some(10),
            overflow: false,
        })
        .normalize()
        {
            UsageInput::Available(usage) => usage,
            UsageInput::Missing | UsageInput::Overflow => panic!("fixture must normalize"),
        };
        let weights = TokenWeights {
            input: 1.0,
            cached_input: 0.1,
            cache_write: 2.4,
            output: 1.0,
            reasoning: 1.0,
        };
        assert_eq!(
            Some(weighted_cost(&usage, weights)),
            crate::token_rate_limit::tests::legacy_weighted_cost_parity_fixture(),
        );
    }
}
