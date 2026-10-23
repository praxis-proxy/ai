// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! CEL-derived quota units over normalized per-request token usage.
//!
//! This is the response-side bridge between `token_count` and policy/limiter
//! consumers. It evaluates one bounded expression over the final usage
//! snapshot and writes the result back to durable filter metadata. Raw
//! `token.*` metadata remains unchanged so audit and billing systems can
//! always recompute a value later.
//!
//! The filter is an experimental bridge and UX prototype for eventually
//! exposing this normalized contract through the Praxis Policy Engine (PPE).
//! It is not a second general-purpose policy engine: the CEL environment is
//! deliberately limited to the fixed `usage` and `context` objects below.

use async_trait::async_trait;
use bytes::Bytes;
use cel::{Context, Program, Value};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::{Deserialize, Serialize};

use crate::token_usage::{
    META_TOKEN_QUOTA_FORMULA, META_TOKEN_QUOTA_STATUS, META_TOKEN_QUOTA_UNIT, META_TOKEN_QUOTA_VALUE,
    TOKEN_QUOTA_STATUS_EVALUATED, TokenQuotaAudit,
    normalized::{UsageCounts, UsageInput},
};

/// Longest accepted CEL expression. The expression has no unbounded input
/// collections, and this cap also bounds its parse tree and evaluation work.
const MAX_EXPRESSION_BYTES: usize = 4096;
/// Longest formula identifier accepted into bounded filter metadata.
const MAX_FORMULA_ID_BYTES: usize = 64;
/// Longest unit name accepted into bounded filter metadata.
const MAX_UNIT_BYTES: usize = 32;
/// Largest exact integer representable by the ledgers' `f64` accounting.
const QUOTA_U64_CAP: u64 = 9_007_199_254_740_992; // 2^53
/// Floating-point form of [`QUOTA_U64_CAP`].
const QUOTA_F64_CAP: f64 = 9_007_199_254_740_992.0;

/// No complete or total-only usage snapshot was available.
const STATUS_MISSING_USAGE: &str = "missing_usage";
/// `token_count` abandoned capture after its configured bound was exceeded.
const STATUS_USAGE_OVERFLOW: &str = "usage_overflow";
/// CEL execution failed for this request's inputs.
const STATUS_EVALUATION_ERROR: &str = "evaluation_error";
/// CEL returned a non-numeric, negative, or non-finite value.
const STATUS_INVALID_RESULT: &str = "invalid_result";

/// Configuration for the `token_quota` response filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenQuotaConfig {
    /// Stable, operator-owned formula identifier recorded with the result.
    formula_id: String,
    /// CEL expression evaluated over `usage` and `context`.
    expression: String,
    /// Unit attached to the result; intentionally not a currency.
    #[serde(default = "default_unit")]
    unit: String,
}

/// Default result unit.
fn default_unit() -> String {
    "quota_units".to_owned()
}

/// Evaluates a configured CEL formula over final normalized token usage.
///
/// This experimental filter models the intended PPE input/output contract
/// while PPE token-input exposure and metadata export are still being settled.
/// Its restricted evaluator is intentionally not a general policy surface.
///
/// Place this filter before `token_count` in the request chain. Response
/// filters run in reverse order, so `token_count` publishes the complete
/// `token.*` snapshot before this filter evaluates it.
///
/// # CEL inputs
///
/// `usage` is an object with this stable shape:
///
/// - `input.total`, `input.uncached`, `input.cache_read`, `input.cache_write`
/// - `output.total`, `output.visible`, `output.reasoning`
/// - `total`, `unclassified`, `typed`
/// - `reported.{input,output,total,cache_read,cache_write,reasoning}`
///
/// Optional provider counts are represented by a numeric zero plus a
/// corresponding `reported` boolean. Cache counts are clamped and partitioned
/// from input; reasoning is partitioned from output when provider totals show
/// it is nested. `unclassified` conservatively carries any remainder of the
/// provider total.
///
/// `context` exposes bounded routing facts: `route`, `backend`, `provider`,
/// and `model`. Missing facts are empty strings.
///
/// The expression must return a non-negative number. Fractional results are
/// rounded up so enforcement never undercharges. Invalid runtime results leave
/// `token.quota.value` unset and publish an explicit status.
///
/// # YAML
///
/// ```yaml
/// - filter: token_quota
///   formula_id: standard-v1
///   unit: quota_units
///   expression: >-
///     double(usage.input.uncached + usage.unclassified) / 3.0
///     + double(usage.output.visible)
///     + double(usage.input.cache_write) / 2.4
///     + double(usage.input.cache_read) / 30.0
///     + double(usage.output.reasoning)
/// ```
pub struct TokenQuotaFilter {
    /// Compiled once at configuration load, never reparsed on the hot path.
    program: Program,
    /// Stable formula identity written alongside each outcome.
    formula_id: String,
    /// Non-currency result unit written alongside each outcome.
    unit: String,
}

impl TokenQuotaFilter {
    /// Construct a token quota filter from YAML.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identifiers, oversized or invalid CEL, or
    /// expressions that reference variables other than `usage` and `context`.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: TokenQuotaConfig = parse_filter_config("token_quota", config)?;
        validate_label("formula_id", &cfg.formula_id, MAX_FORMULA_ID_BYTES)?;
        validate_label("unit", &cfg.unit, MAX_UNIT_BYTES)?;
        if cfg.expression.is_empty() || cfg.expression.len() > MAX_EXPRESSION_BYTES {
            return Err(format!("token_quota: expression must be 1..={MAX_EXPRESSION_BYTES} bytes").into());
        }
        let program = Program::compile(&cfg.expression)
            .map_err(|error| FilterError::from(format!("token_quota: invalid CEL expression: {error}")))?;
        let references = program.references();
        let mut invalid_variables = references
            .variables()
            .into_iter()
            .filter(|name| !matches!(*name, "usage" | "context"))
            .collect::<Vec<_>>();
        invalid_variables.sort_unstable();
        if !invalid_variables.is_empty() {
            return Err(format!(
                "token_quota: expression may reference only 'usage' and 'context'; found {}",
                invalid_variables.join(", ")
            )
            .into());
        }
        validate_program_result(&program)?;
        Ok(Box::new(Self {
            program,
            formula_id: cfg.formula_id,
            unit: cfg.unit,
        }))
    }

    /// Evaluate and publish a quota result for the final response snapshot.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps one quota outcome atomic: clear stale value, evaluate, then publish status and result"
    )]
    fn evaluate(&self, ctx: &mut HttpFilterContext<'_>) {
        ctx.filter_metadata.remove(META_TOKEN_QUOTA_VALUE);
        ctx.extensions.remove::<TokenQuotaAudit>();
        ctx.set_metadata(META_TOKEN_QUOTA_FORMULA, &self.formula_id);
        ctx.set_metadata(META_TOKEN_QUOTA_UNIT, &self.unit);

        let usage = match UsageCounts::from_context(ctx).normalize() {
            UsageInput::Available(usage) => usage,
            UsageInput::Missing => {
                ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_MISSING_USAGE);
                return;
            },
            UsageInput::Overflow => {
                ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_USAGE_OVERFLOW);
                return;
            },
        };
        let route_context = RouteContext::from_context(ctx);
        let (Ok(usage_audit), Ok(context_audit)) = (serde_json::to_value(usage), serde_json::to_value(route_context))
        else {
            ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_EVALUATION_ERROR);
            return;
        };
        let mut cel_context = Context::default();
        if cel_context.add_variable("usage", usage).is_err()
            || cel_context.add_variable("context", route_context).is_err()
        {
            // These structures contain only CEL-native scalar/object fields;
            // keep the request outcome explicit if that invariant changes.
            ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_EVALUATION_ERROR);
            return;
        }
        ctx.extensions.insert(TokenQuotaAudit {
            usage: usage_audit,
            context: context_audit,
        });
        let value = match self.program.execute(&cel_context) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(%error, formula_id = %self.formula_id, "token_quota: CEL evaluation failed");
                ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_EVALUATION_ERROR);
                return;
            },
        };
        let Some(value) = quota_value(&value) else {
            ctx.set_metadata(META_TOKEN_QUOTA_STATUS, STATUS_INVALID_RESULT);
            return;
        };
        ctx.set_metadata(META_TOKEN_QUOTA_VALUE, value.to_string());
        ctx.set_metadata(META_TOKEN_QUOTA_STATUS, TOKEN_QUOTA_STATUS_EVALUATED);
    }
}

/// Execute a representative complete snapshot at configuration load so
/// formulas with an unconditional type/domain error fail before traffic.
#[expect(
    clippy::too_many_lines,
    reason = "the validation snapshot and both CEL bindings must stay visibly identical to runtime evaluation"
)]
fn validate_program_result(program: &Program) -> Result<(), FilterError> {
    let usage = match (UsageCounts {
        input: Some(1),
        output: Some(1),
        total: Some(2),
        cache_read: Some(0),
        cache_write: Some(0),
        reasoning: Some(0),
        overflow: false,
    })
    .normalize()
    {
        UsageInput::Available(usage) => usage,
        UsageInput::Missing | UsageInput::Overflow => {
            return Err("token_quota: internal validation snapshot is unavailable".into());
        },
    };
    let route_context = RouteContext {
        route: "",
        backend: "",
        provider: "",
        model: "",
    };
    let mut context = Context::default();
    context
        .add_variable("usage", usage)
        .map_err(|error| FilterError::from(format!("token_quota: cannot build validation usage: {error}")))?;
    context
        .add_variable("context", route_context)
        .map_err(|error| FilterError::from(format!("token_quota: cannot build validation context: {error}")))?;
    let result = program
        .execute(&context)
        .map_err(|error| FilterError::from(format!("token_quota: CEL validation failed: {error}")))?;
    if quota_value(&result).is_none() {
        return Err("token_quota: CEL expression must return a finite, non-negative number".into());
    }
    Ok(())
}

#[async_trait]
impl HttpFilter for TokenQuotaFilter {
    fn name(&self) -> &'static str {
        "token_quota"
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if end_of_stream {
            self.evaluate(ctx);
        }
        Ok(FilterAction::Continue)
    }
}

/// Validate a bounded, printable metadata label.
fn validate_label(field: &str, value: &str, max_bytes: usize) -> Result<(), FilterError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(format!("token_quota: {field} must be 1..={max_bytes} bytes without control characters").into());
    }
    Ok(())
}

/// Convert a CEL numeric result to conservative integer quota units.
fn quota_value(value: &Value) -> Option<u64> {
    match value {
        Value::UInt(value) => Some((*value).min(QUOTA_U64_CAP)),
        Value::Int(value) => u64::try_from(*value).ok().map(|value| value.min(QUOTA_U64_CAP)),
        Value::Float(value) if value.is_finite() && *value >= 0.0 => {
            let rounded = value.min(QUOTA_F64_CAP).ceil();
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "rounded is finite, non-negative, and capped at the exactly representable 2^53"
            )]
            Some(rounded as u64)
        },
        Value::List(_)
        | Value::Map(_)
        | Value::Function(..)
        | Value::String(_)
        | Value::Bytes(_)
        | Value::Bool(_)
        | Value::Duration(_)
        | Value::Timestamp(_)
        | Value::Opaque(_)
        | Value::Null
        | Value::Float(_) => None,
    }
}

/// Trusted routing facts available to formula selection/branching.
#[derive(Debug, Clone, Copy, Serialize)]
struct RouteContext<'a> {
    /// Logical route name, when routing published one.
    route: &'a str,
    /// Selected backend cluster.
    backend: &'a str,
    /// Trusted provider identity or API format.
    provider: &'a str,
    /// Effective routed model when available, otherwise the parsed request model.
    model: &'a str,
}

impl<'a> RouteContext<'a> {
    /// Build the bounded context without exposing arbitrary request headers or bodies.
    fn from_context(ctx: &'a HttpFilterContext<'_>) -> Self {
        Self {
            route: ctx
                .get_metadata("intelligent_route.name")
                .or(ctx.metrics_route.as_deref())
                .unwrap_or(""),
            backend: ctx.cluster_name().unwrap_or(""),
            provider: ctx
                .get_metadata("provider_route.provider_id")
                .or_else(|| ctx.get_metadata("intelligent_route.api_format"))
                .unwrap_or(""),
            model: ctx
                .get_metadata("provider_route.model")
                .or_else(|| ctx.get_metadata("intelligent_route.target_model"))
                .or_else(|| ctx.get_metadata("openai_responses_model_rewrite.effective_model"))
                .or_else(|| ctx.get_metadata("openai_responses_request.model"))
                .or_else(|| ctx.get_metadata("openai_chat_completions_request.model"))
                .or_else(|| ctx.get_metadata("anthropic_messages_request.model"))
                .or_else(|| ctx.get_metadata(crate::llm_metrics::META_MODEL_TO_HEADER_MODEL))
                .unwrap_or(""),
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used,
    reason = "tests fail fast on fixture mistakes"
)]
mod tests {
    use http::Method;

    use super::*;
    use crate::{
        test_utils::{make_filter_context, make_request},
        token_usage::{
            META_TOKEN_CACHE_READ, META_TOKEN_CACHE_WRITE, META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_STATUS,
            META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW,
        },
    };

    /// Build the documented weighted formula.
    fn filter(expression: &str) -> Box<dyn HttpFilter> {
        TokenQuotaFilter::from_config(
            &serde_yaml::from_str(&format!(
                "formula_id: standard-v1\nunit: quota_units\nexpression: {expression:?}\n"
            ))
            .unwrap(),
        )
        .unwrap()
    }

    /// Populate the three parent usage counts.
    fn set_usage(ctx: &mut HttpFilterContext<'_>, input: u64, output: u64, total: u64) {
        ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
        ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
        ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
    }

    #[test]
    fn documented_formula_partitions_cache_and_rounds_up() {
        let filter = filter(
            "double(usage.input.uncached + usage.unclassified) / 3.0 + double(usage.output.visible) + double(usage.input.cache_write) / 2.4 + double(usage.input.cache_read) / 30.0 + double(usage.output.reasoning)",
        );
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        set_usage(&mut ctx, 120, 40, 160);
        ctx.set_metadata(META_TOKEN_CACHE_READ, "90");
        ctx.set_metadata(META_TOKEN_CACHE_WRITE, "20");
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
            FilterAction::Continue
        ));

        // 10/3 + 40 + 20/2.4 + 90/30 = 54.666... -> 55.
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE), Some("55"));
        assert_eq!(
            ctx.get_metadata(META_TOKEN_QUOTA_STATUS),
            Some(TOKEN_QUOTA_STATUS_EVALUATED)
        );
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_FORMULA), Some("standard-v1"));
    }

    #[test]
    fn cache_overreport_is_clamped_before_cel() {
        let filter = filter("usage.input.cache_read + usage.input.cache_write + usage.input.uncached");
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        set_usage(&mut ctx, 10, 0, 10);
        ctx.set_metadata(META_TOKEN_CACHE_READ, "8");
        ctx.set_metadata(META_TOKEN_CACHE_WRITE, "8");
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE), Some("10"));
    }

    #[test]
    fn missing_optional_fields_are_zero_with_presence_flags() {
        let filter = filter("usage.reported.cache_read ? 999u : usage.input.uncached");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        set_usage(&mut ctx, 12, 3, 15);
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE), Some("12"));
    }

    #[test]
    fn audit_snapshot_preserves_exact_cel_inputs() {
        let filter = filter("usage.total");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        set_usage(&mut ctx, 12, 3, 15);
        ctx.set_metadata("provider_route.provider_id", "openai");
        ctx.set_metadata("provider_route.model", "gpt-4");
        drop(filter.on_response_body(&mut ctx, &mut None, true).unwrap());

        let audit = ctx.extensions.get::<TokenQuotaAudit>().expect("audit snapshot");
        assert_json_pointer(&audit.usage, "/input/uncached", &serde_json::json!(12));
        assert_json_pointer(&audit.usage, "/reported/cache_read", &serde_json::json!(false));
        assert_json_pointer(&audit.context, "/provider", &serde_json::json!("openai"));
        assert_json_pointer(&audit.context, "/model", &serde_json::json!("gpt-4"));
    }

    /// Assert one path in a retained CEL audit object.
    fn assert_json_pointer(root: &serde_json::Value, pointer: &str, expected: &serde_json::Value) {
        assert_eq!(root.pointer(pointer), Some(expected));
    }

    #[test]
    fn total_only_usage_has_conservative_unclassified_path() {
        let filter = filter("usage.typed ? 0u : usage.unclassified");
        let request = make_request(Method::POST, "/v1/chat/completions");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata(META_TOKEN_TOTAL, "42");
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE), Some("42"));
    }

    #[test]
    fn overflow_and_missing_usage_are_explicit_and_publish_no_value() {
        for (status, expected) in [
            (None, STATUS_MISSING_USAGE),
            (Some(TOKEN_STATUS_OVERFLOW), STATUS_USAGE_OVERFLOW),
        ] {
            let filter = filter("usage.total");
            let request = make_request(Method::POST, "/v1/chat/completions");
            let mut ctx = make_filter_context(&request);
            if let Some(status) = status {
                ctx.set_metadata(META_TOKEN_STATUS, status);
            }
            assert!(matches!(
                filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
                FilterAction::Continue
            ));
            assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_STATUS), Some(expected));
            assert!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE).is_none());
        }
    }

    #[test]
    fn context_can_select_provider_or_model_policy() {
        let filter = filter(
            "context.provider == 'anthropic' && context.model == 'claude-sonnet' ? usage.total / 2u : usage.total",
        );
        let request = make_request(Method::POST, "/v1/messages");
        let mut ctx = make_filter_context(&request);
        ctx.set_metadata("provider_route.provider_id", "anthropic");
        ctx.set_metadata("provider_route.model", "claude-sonnet");
        set_usage(&mut ctx, 80, 20, 100);
        assert!(matches!(
            filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
            FilterAction::Continue
        ));
        assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE), Some("50"));
    }

    #[test]
    fn invalid_runtime_results_never_publish_quota() {
        for expression in [
            "context.provider == 'bad' ? -1 : usage.total",
            "context.provider == 'bad' ? 'not a number' : usage.total",
            "context.provider == 'bad' ? double('NaN') : usage.total",
        ] {
            let filter = filter(expression);
            let request = make_request(Method::POST, "/v1/chat/completions");
            let mut ctx = make_filter_context(&request);
            ctx.set_metadata("provider_route.provider_id", "bad");
            set_usage(&mut ctx, 1, 1, 2);
            assert!(matches!(
                filter.on_response_body(&mut ctx, &mut None, true).unwrap(),
                FilterAction::Continue
            ));
            assert_eq!(ctx.get_metadata(META_TOKEN_QUOTA_STATUS), Some(STATUS_INVALID_RESULT));
            assert!(ctx.get_metadata(META_TOKEN_QUOTA_VALUE).is_none());
        }
    }

    #[test]
    fn configuration_rejects_unbounded_input_names_and_bad_labels() {
        for yaml in [
            "formula_id: f\nexpression: request.headers.cost\n",
            "formula_id: f\nexpression: \"'not a number'\"\n",
            "formula_id: f\nexpression: -1\n",
            "formula_id: \"bad\\nlabel\"\nexpression: usage.total\n",
            "formula_id: f\nunit: \"\"\nexpression: usage.total\n",
        ] {
            let error = TokenQuotaFilter::from_config(&serde_yaml::from_str(yaml).unwrap())
                .err()
                .expect("configuration must fail");
            assert!(error.to_string().contains("token_quota"), "{error}");
        }
    }
}
