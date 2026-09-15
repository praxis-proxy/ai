// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Token-type-aware weighted cost for M4 reconciliation.
//!
//! Admission still reserves a single unweighted estimate
//! ([`super::config::RuleConfig::reserved_tokens`]). After
//! `token_count` publishes typed usage, reconciliation charges
//!
//! ```text
//! cost = Σ (partitioned_tokens_of_type × weight_of_type)
//! ```
//!
//! Cache read/write counts are a **breakdown** of `token.input`, not an
//! addition. Reasoning is a breakdown of `token.output` on OpenAI /
//! Anthropic, and **additive** on Google (thoughts are not inside
//! `candidatesTokenCount`). The partition uses those facts so operators
//! never double-count. See [`weighted_cost`].

use praxis_filter::{FilterError, HttpFilterContext};
use serde::Deserialize;

use crate::token_usage::{
    META_TOKEN_CACHE_READ, META_TOKEN_CACHE_WRITE, META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_REASONING,
    META_TOKEN_STATUS, META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW,
};

/// Cap for the float→integer conversion. Matches the Lua `f64` safe-integer
/// bound the ledgers already enforce on `capacity`, so a weighted cost can
/// never be a value the backends cannot represent exactly.
const COST_U64_CAP: u64 = 9_007_199_254_740_992; // 2^53

/// Operator-facing per-type weights. Omitted keys stay `None` so a rule
/// overlay can replace only the types it cares about.
///
/// Unknown type names are rejected (`deny_unknown_fields`) rather than
/// silently ignored — a typo like `cached: 0.1` must not look like a
/// working cache discount.
#[derive(Debug, Default, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TokenTypeWeightsConfig {
    /// Weight for uncached input tokens (the residual of `token.input`
    /// after subtracting cache read/write). Defaults to `1.0` when omitted.
    #[serde(default)]
    pub input: Option<f64>,

    /// Weight for visible output tokens (the residual of `token.output`
    /// after subtracting nested reasoning). Defaults to `1.0` when omitted.
    #[serde(default)]
    pub output: Option<f64>,

    /// Weight for prompt-cache hits (`token.cache_read`). A value below
    /// `1.0` cheapens cached input; the proposal's example is `0.1`.
    #[serde(default)]
    pub cached_input: Option<f64>,

    /// Weight for prompt-cache writes (`token.cache_write`). Anthropic
    /// cache creation is typically priced *above* uncached input; omit
    /// to keep `1.0`.
    #[serde(default)]
    pub cache_write: Option<f64>,

    /// Weight for reasoning / thinking tokens (`token.reasoning`).
    #[serde(default)]
    pub reasoning: Option<f64>,
}

/// Resolved weights applied on the reconcile hot path. Every field is
/// finite and `>= 0` — validated at config load, never on each request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TokenWeights {
    /// Uncached input.
    pub input: f64,
    /// Visible (non-reasoning, when nested) output.
    pub output: f64,
    /// Prompt-cache hits.
    pub cached_input: f64,
    /// Prompt-cache writes.
    pub cache_write: f64,
    /// Reasoning / thinking tokens.
    pub reasoning: f64,
}

impl TokenWeights {
    /// Filter-wide identity: every type costs `1.0`, matching pre-M4
    /// `token.total` accounting when no `default_weights` are set.
    pub(super) const UNITY: Self = Self {
        input: 1.0,
        output: 1.0,
        cached_input: 1.0,
        cache_write: 1.0,
        reasoning: 1.0,
    };

    /// Overlay `cfg`'s present keys onto `self`, validating each value.
    ///
    /// `loc` is a human-readable path used in error messages
    /// (`default_weights` or `rule 'team-alpha'`).
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if any set weight is negative, `NaN`, or infinite.
    pub(super) fn overlay(mut self, cfg: &TokenTypeWeightsConfig, loc: &str) -> Result<Self, FilterError> {
        if let Some(value) = cfg.input {
            self.input = checked_weight(value, loc, "input")?;
        }
        if let Some(value) = cfg.output {
            self.output = checked_weight(value, loc, "output")?;
        }
        if let Some(value) = cfg.cached_input {
            self.cached_input = checked_weight(value, loc, "cached_input")?;
        }
        if let Some(value) = cfg.cache_write {
            self.cache_write = checked_weight(value, loc, "cache_write")?;
        }
        if let Some(value) = cfg.reasoning {
            self.reasoning = checked_weight(value, loc, "reasoning")?;
        }
        Ok(self)
    }
}

/// Reject a non-finite or negative weight at config load.
fn checked_weight(value: f64, loc: &str, field: &str) -> Result<f64, FilterError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(format!("token_rate_limit: {loc}: {field} must be a finite number >= 0, got {value}").into())
    }
}

/// Typed usage recovered from `token_count` metadata. Every count is
/// optional so absence stays distinct from a reported zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct UsageCounts {
    /// `token.input`, cache included.
    pub input: Option<u64>,
    /// `token.output` (visible candidates on Google).
    pub output: Option<u64>,
    /// `token.total`.
    pub total: Option<u64>,
    /// `token.cache_read`, a subset of `input`.
    pub cache_read: Option<u64>,
    /// `token.cache_write`, a subset of `input`.
    pub cache_write: Option<u64>,
    /// `token.reasoning`.
    pub reasoning: Option<u64>,
    /// `token.status == overflow`: capture was abandoned; do not trust
    /// any partial counts for quota.
    pub overflow: bool,
}

impl UsageCounts {
    /// Read the `token.*` keys `token_count` publishes onto `ctx`.
    ///
    /// Unparseable values are treated as absent, matching the pre-M4
    /// `token.total` parse.
    pub(super) fn from_context(ctx: &HttpFilterContext<'_>) -> Self {
        Self {
            input: parse_u64_meta(ctx, META_TOKEN_INPUT),
            output: parse_u64_meta(ctx, META_TOKEN_OUTPUT),
            total: parse_u64_meta(ctx, META_TOKEN_TOTAL),
            cache_read: parse_u64_meta(ctx, META_TOKEN_CACHE_READ),
            cache_write: parse_u64_meta(ctx, META_TOKEN_CACHE_WRITE),
            reasoning: parse_u64_meta(ctx, META_TOKEN_REASONING),
            overflow: ctx.get_metadata(META_TOKEN_STATUS) == Some(TOKEN_STATUS_OVERFLOW),
        }
    }
}

/// Parse a `token.*` metadata value as `u64`. Missing or unparseable
/// strings are `None`, matching the pre-M4 `token.total` path.
fn parse_u64_meta(ctx: &HttpFilterContext<'_>, key: &str) -> Option<u64> {
    ctx.get_metadata(key).and_then(|value| value.parse::<u64>().ok())
}

/// Compute the integer cost to settle against the reservation.
///
/// Returns:
/// - `None` on overflow or when no usable counts exist, so the ledger keeps the admission estimate (today's
///   missing-`token.total` path).
/// - `Some(total)` when typed parents are missing but `token.total` is present (Bedrock Converse, streaming without
///   typed breakdowns).
/// - `Some(weighted)` when `token.input` and `token.output` are both present: cache is partitioned out of input,
///   reasoning is nested or additive based on how `token.total` relates to the parts, then `ceil(Σ type × weight)`.
#[must_use]
pub(super) fn weighted_cost(usage: UsageCounts, weights: TokenWeights) -> Option<u64> {
    if usage.overflow {
        return None;
    }
    match (usage.input, usage.output) {
        (Some(input), Some(output)) => Some(partitioned_cost(input, output, usage, weights)),
        _ => usage.total,
    }
}

/// Partition nested types, then apply weights. Malformed breakdowns that
/// exceed their parent are clamped so the residual never underflows.
fn partitioned_cost(input: u64, output: u64, usage: UsageCounts, weights: TokenWeights) -> u64 {
    let cache_read = usage.cache_read.unwrap_or(0).min(input);
    let after_read = input - cache_read;
    let cache_write = usage.cache_write.unwrap_or(0).min(after_read);
    let uncached = after_read - cache_write;

    let reasoning = usage.reasoning.unwrap_or(0);
    let (visible_output, reasoning_tokens) = if reasoning_is_nested(input, output, usage.total, reasoning) {
        let nested = reasoning.min(output);
        (output - nested, nested)
    } else {
        (output, reasoning)
    };

    let cost = to_cost_f64(uncached) * weights.input
        + to_cost_f64(cache_read) * weights.cached_input
        + to_cost_f64(cache_write) * weights.cache_write
        + to_cost_f64(visible_output) * weights.output
        + to_cost_f64(reasoning_tokens) * weights.reasoning;
    ceil_cost(cost)
}

/// Decide whether reasoning is already inside `output`.
///
/// `token_count` publishes `token.total = input + output` for OpenAI /
/// Anthropic (reasoning ⊆ output) and `token.total = input + output +
/// reasoning` for Google (thoughts billed separately). When `total` is
/// absent, nested is the conservative default: it never charges reasoning
/// twice on the common providers.
fn reasoning_is_nested(input: u64, output: u64, total: Option<u64>, reasoning: u64) -> bool {
    if reasoning == 0 {
        return true;
    }
    let Some(total) = total else {
        return true;
    };
    let nested_sum = input.saturating_add(output);
    let additive_sum = nested_sum.saturating_add(reasoning);
    // Closer to `input + output` → nested (OpenAI/Anthropic). Closer to
    // `input + output + reasoning` → additive (Google). Equal distance
    // prefers nested so we never double-count the common case.
    total.abs_diff(nested_sum) <= total.abs_diff(additive_sum)
}

/// `f64` form of [`COST_U64_CAP`]. `2^53` is an exact `f64` value.
const COST_F64_CAP: f64 = 9_007_199_254_740_992.0;

/// Convert a token count to `f64` for weighting. Clamped to [`COST_U64_CAP`]
/// so the mantissa stays exact (the ledgers already reject capacities
/// above this bound).
fn to_cost_f64(tokens: u64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "clamped to 2^53, which is exactly representable in f64"
    )]
    {
        tokens.min(COST_U64_CAP) as f64
    }
}

/// Ceiling conversion so a quota never under-charges a fractional cost
/// (9 cached tokens × 0.1 = 0.9 → 1). Non-finite or negative results
/// (should be unreachable after config validation) become 0.
fn ceil_cost(cost: f64) -> u64 {
    if !cost.is_finite() || cost <= 0.0 {
        return 0;
    }
    let capped = cost.min(COST_F64_CAP).ceil();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "capped is finite, non-negative, and <= 2^53, so ceil is an exact u64"
    )]
    {
        capped as u64
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    reason = "tests fail fast on fixture mistakes"
)]
mod tests {
    use super::*;

    fn unity() -> TokenWeights {
        TokenWeights::UNITY
    }

    fn weights(cached_input: f64, reasoning: f64) -> TokenWeights {
        TokenWeights {
            cached_input,
            reasoning,
            ..TokenWeights::UNITY
        }
    }

    fn usage(input: u64, output: u64, total: u64) -> UsageCounts {
        UsageCounts {
            input: Some(input),
            output: Some(output),
            total: Some(total),
            ..UsageCounts::default()
        }
    }

    #[test]
    fn missing_counts_yield_none_so_the_estimate_stands() {
        assert_eq!(weighted_cost(UsageCounts::default(), unity()), None);
    }

    #[test]
    fn overflow_ignores_partial_counts_and_yields_none() {
        let usage = UsageCounts {
            input: Some(10),
            output: Some(10),
            total: Some(20),
            overflow: true,
            ..UsageCounts::default()
        };
        assert_eq!(weighted_cost(usage, unity()), None);
    }

    #[test]
    fn total_only_falls_back_at_weight_one() {
        let usage = UsageCounts {
            total: Some(42),
            ..UsageCounts::default()
        };
        assert_eq!(weighted_cost(usage, weights(0.1, 0.9)), Some(42));
    }

    #[test]
    fn input_without_output_falls_back_to_total() {
        let usage = UsageCounts {
            input: Some(100),
            total: Some(150),
            ..UsageCounts::default()
        };
        assert_eq!(weighted_cost(usage, unity()), Some(150));
    }

    #[test]
    fn unity_weights_on_typed_counts_match_input_plus_output() {
        // OpenAI: reasoning nested, so 120+800 not 120+800+640.
        let mut counts = usage(120, 800, 920);
        counts.reasoning = Some(640);
        assert_eq!(weighted_cost(counts, unity()), Some(920));
    }

    #[test]
    fn naive_sum_of_parent_plus_cache_is_not_used() {
        // 1000 input (includes 900 cached) + 50 output. Naive
        // input*1 + cache*0.1 + output*1 would be 1140. Partition is 100+90+50.
        let mut counts = usage(1000, 50, 1050);
        counts.cache_read = Some(900);
        assert_eq!(weighted_cost(counts, weights(0.1, 1.0)), Some(240));
    }

    #[test]
    fn anthropic_cache_write_is_partitioned_out_of_input() {
        // Praxis token.input is already uncached+read+write = 6050.
        let mut counts = usage(6050, 100, 6150);
        counts.cache_read = Some(5000);
        counts.cache_write = Some(1000);
        let w = TokenWeights {
            cached_input: 0.1,
            cache_write: 1.25,
            ..TokenWeights::UNITY
        };
        // uncached 50*1 + read 5000*0.1 + write 1000*1.25 + output 100 = 50+500+1250+100
        assert_eq!(weighted_cost(counts, w), Some(1900));
    }

    #[test]
    fn openai_nested_reasoning_is_not_added_on_top_of_output() {
        let mut counts = usage(120, 800, 920);
        counts.reasoning = Some(640);
        // visible 160*1 + reasoning 640*0.9 + input 120 = 160+576+120 = 856
        assert_eq!(weighted_cost(counts, weights(1.0, 0.9)), Some(856));
    }

    #[test]
    fn google_additive_reasoning_is_not_subtracted_from_output() {
        // prompt 50 + candidates 80 + thoughts 200 = 330. Nested subtract
        // would drop the 80 visible tokens.
        let mut counts = usage(50, 80, 330);
        counts.reasoning = Some(200);
        // 50*1 + 80*1 + 200*0.9 = 310
        assert_eq!(weighted_cost(counts, weights(1.0, 0.9)), Some(310));
    }

    #[test]
    fn google_thoughts_larger_than_output_does_not_underflow() {
        let mut counts = usage(50, 80, 330);
        counts.reasoning = Some(200);
        assert_eq!(weighted_cost(counts, unity()), Some(330));
    }

    #[test]
    fn total_identity_classifies_nested_vs_additive_reasoning() {
        assert!(
            reasoning_is_nested(120, 800, Some(920), 640),
            "OpenAI: total == input+output"
        );
        assert!(
            !reasoning_is_nested(50, 80, Some(330), 200),
            "Google: total == input+output+reasoning"
        );
        assert!(
            reasoning_is_nested(50, 80, None, 200),
            "missing total defaults to nested"
        );
    }

    #[test]
    fn missing_total_defaults_to_nested_reasoning() {
        let mut counts = UsageCounts {
            input: Some(10),
            output: Some(20),
            reasoning: Some(15),
            ..UsageCounts::default()
        };
        // nested: visible 5 + reasoning 15 + input 10 = 30, not 10+20+15=45
        assert_eq!(weighted_cost(counts, unity()), Some(30));
        counts.total = None;
        assert_eq!(weighted_cost(counts, unity()), Some(30));
    }

    #[test]
    fn absent_cache_keys_charge_all_input_at_input_weight() {
        let counts = usage(100, 10, 110);
        assert_eq!(weighted_cost(counts, weights(0.1, 1.0)), Some(110));
    }

    #[test]
    fn zero_cache_read_is_a_reported_miss_not_a_discount() {
        let mut counts = usage(100, 10, 110);
        counts.cache_read = Some(0);
        assert_eq!(weighted_cost(counts, weights(0.1, 1.0)), Some(110));
    }

    #[test]
    fn ceil_never_undercharges_a_fractional_cache_cost() {
        let mut counts = usage(9, 0, 9);
        counts.cache_read = Some(9);
        // 9 * 0.1 = 0.9 → 1
        assert_eq!(weighted_cost(counts, weights(0.1, 1.0)), Some(1));
    }

    #[test]
    fn cache_breakdown_exceeding_input_is_clamped() {
        let mut counts = usage(10, 0, 10);
        counts.cache_read = Some(8);
        counts.cache_write = Some(8); // 8+8 > 10; write clamped to leftover 2
        let w = TokenWeights {
            cached_input: 0.1,
            cache_write: 1.0,
            ..TokenWeights::UNITY
        };
        // uncached 0 + read 8*0.1 + write 2*1 = 0.8+2 → ceil 3
        assert_eq!(weighted_cost(counts, w), Some(3));
    }

    #[test]
    fn overlay_replaces_only_the_named_types() {
        let defaults = TokenWeights::UNITY
            .overlay(
                &TokenTypeWeightsConfig {
                    cached_input: Some(0.1),
                    reasoning: Some(0.9),
                    ..TokenTypeWeightsConfig::default()
                },
                "default_weights",
            )
            .unwrap();
        assert!((defaults.cached_input - 0.1).abs() < f64::EPSILON);
        assert!((defaults.input - 1.0).abs() < f64::EPSILON);

        let rule = defaults
            .overlay(
                &TokenTypeWeightsConfig {
                    cached_input: Some(0.05),
                    ..TokenTypeWeightsConfig::default()
                },
                "rule 'alpha'",
            )
            .unwrap();
        assert!((rule.cached_input - 0.05).abs() < f64::EPSILON);
        assert!((rule.reasoning - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn overlay_rejects_negative_and_non_finite_weights() {
        for value in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = TokenWeights::UNITY
                .overlay(
                    &TokenTypeWeightsConfig {
                        cached_input: Some(value),
                        ..TokenTypeWeightsConfig::default()
                    },
                    "default_weights",
                )
                .expect_err("must reject");
            assert!(err.to_string().contains("cached_input"), "got: {err}");
        }
    }

    #[test]
    fn unknown_weight_key_is_rejected_by_serde() {
        let err = serde_yaml::from_str::<TokenTypeWeightsConfig>("cached: 0.1\n").expect_err("typo must fail");
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }
}
