// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared normalized usage snapshot for quota and rate-limit consumers.

use praxis_filter::HttpFilterContext;
use serde::Serialize;

use super::{
    META_TOKEN_CACHE_READ, META_TOKEN_CACHE_WRITE, META_TOKEN_INPUT, META_TOKEN_OUTPUT, META_TOKEN_REASONING,
    META_TOKEN_STATUS, META_TOKEN_TOTAL, TOKEN_STATUS_OVERFLOW,
};

/// Typed usage recovered from `token_count` metadata. Every count is optional
/// so absence stays distinct from a reported zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UsageCounts {
    /// `token.input`, cache included.
    pub(crate) input: Option<u64>,
    /// `token.output` (visible candidates on Google).
    pub(crate) output: Option<u64>,
    /// `token.total`.
    pub(crate) total: Option<u64>,
    /// `token.cache_read`, a subset of input.
    pub(crate) cache_read: Option<u64>,
    /// `token.cache_write`, a subset of input.
    pub(crate) cache_write: Option<u64>,
    /// `token.reasoning`.
    pub(crate) reasoning: Option<u64>,
    /// Capture overflow makes partial counts unusable.
    pub(crate) overflow: bool,
}

impl UsageCounts {
    /// Read the `token.*` keys `token_count` publishes onto `ctx`.
    pub(crate) fn from_context(ctx: &HttpFilterContext<'_>) -> Self {
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

    /// Sanitize and partition one final request-local usage snapshot.
    #[expect(
        clippy::too_many_lines,
        reason = "normalization keeps overlap, presence, and conservative fallback invariants in one audit path"
    )]
    pub(crate) fn normalize(self) -> UsageInput {
        if self.overflow {
            return UsageInput::Overflow;
        }
        let reported = ReportedUsage {
            input: self.input.is_some(),
            output: self.output.is_some(),
            total: self.total.is_some(),
            cache_read: self.cache_read.is_some(),
            cache_write: self.cache_write.is_some(),
            reasoning: self.reasoning.is_some(),
        };
        let (Some(input), Some(output)) = (self.input, self.output) else {
            return self.total.map_or(UsageInput::Missing, |total| {
                UsageInput::Available(NormalizedUsage {
                    input: InputUsage::default(),
                    output: OutputUsage::default(),
                    total,
                    unclassified: total,
                    typed: false,
                    reported,
                })
            });
        };

        let cache_read = self.cache_read.unwrap_or(0).min(input);
        let after_read = input - cache_read;
        let cache_write = self.cache_write.unwrap_or(0).min(after_read);
        let uncached = after_read - cache_write;
        let reasoning = self.reasoning.unwrap_or(0);
        let nested = reasoning_is_nested(input, output, self.total, reasoning);
        let (visible, reasoning) = if nested {
            let reasoning = reasoning.min(output);
            (output - reasoning, reasoning)
        } else {
            (output, reasoning)
        };
        let accounted = input
            .saturating_add(output)
            .saturating_add(if nested { 0 } else { reasoning });
        let total = self.total.unwrap_or(accounted);
        UsageInput::Available(NormalizedUsage {
            input: InputUsage {
                total: input,
                uncached,
                cache_read,
                cache_write,
            },
            output: OutputUsage {
                total: output,
                visible,
                reasoning,
            },
            total,
            unclassified: total.saturating_sub(accounted),
            typed: true,
            reported,
        })
    }
}

/// Parse metadata as `u64`, preserving missing or unusable values as `None`.
pub(crate) fn parse_u64_meta(ctx: &HttpFilterContext<'_>, key: &str) -> Option<u64> {
    ctx.get_metadata(key).and_then(|value| value.parse::<u64>().ok())
}

/// Whether a usable usage input exists for this request.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UsageInput {
    /// Complete typed usage or a conservative total-only snapshot.
    Available(NormalizedUsage),
    /// No usable counts were published.
    Missing,
    /// Capture overflow made any partial counts untrustworthy.
    Overflow,
}

/// Stable CEL usage contract.
#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct NormalizedUsage {
    /// Sanitized input-token partitions.
    pub(crate) input: InputUsage,
    /// Sanitized output-token partitions.
    pub(crate) output: OutputUsage,
    /// Provider-reported or normalized overall total.
    pub(crate) total: u64,
    /// Total remainder not classified by typed parent counts.
    pub(crate) unclassified: u64,
    /// Whether both input and output parent counts were available.
    pub(crate) typed: bool,
    /// Presence flags for every provider-facing count.
    pub(crate) reported: ReportedUsage,
}

/// Input token partitions exposed to consumers.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct InputUsage {
    /// Total normalized input, including cache partitions.
    pub(crate) total: u64,
    /// Input remaining after cache read/write partitions.
    pub(crate) uncached: u64,
    /// Input served from prompt cache, clamped to `total`.
    pub(crate) cache_read: u64,
    /// Input written to prompt cache, clamped to remaining input.
    pub(crate) cache_write: u64,
}

/// Output token partitions exposed to consumers.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub(crate) struct OutputUsage {
    /// Total normalized output parent count.
    pub(crate) total: u64,
    /// Non-reasoning output when reasoning is nested.
    pub(crate) visible: u64,
    /// Reasoning tokens, nested or additive according to provider totals.
    pub(crate) reasoning: u64,
}

/// Provider field-presence flags exposed to consumers.
#[derive(Debug, Clone, Copy, Serialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "CEL needs an explicit presence bit for each independently optional provider field"
)]
pub(crate) struct ReportedUsage {
    /// Input parent was present.
    pub(crate) input: bool,
    /// Output parent was present.
    pub(crate) output: bool,
    /// Total was present.
    pub(crate) total: bool,
    /// Cache-read breakdown was present.
    pub(crate) cache_read: bool,
    /// Cache-write breakdown was present.
    pub(crate) cache_write: bool,
    /// Reasoning breakdown was present.
    pub(crate) reasoning: bool,
}

/// Determine whether reasoning is included in output or additive.
fn reasoning_is_nested(input: u64, output: u64, total: Option<u64>, reasoning: u64) -> bool {
    if reasoning == 0 {
        return true;
    }
    let Some(total) = total else {
        return true;
    };
    let nested_sum = input.saturating_add(output);
    let additive_sum = nested_sum.saturating_add(reasoning);
    total.abs_diff(nested_sum) <= total.abs_diff(additive_sum)
}
