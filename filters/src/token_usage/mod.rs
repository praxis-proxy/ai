// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Token usage extraction and exposure filters.
//!
//! This module owns the complete in-process token usage flow: parsing
//! provider responses, storing normalized counts in filter metadata, and
//! optionally exposing those counts as downstream response headers.

mod count;
mod headers;
mod providers;
mod streaming;

pub use count::TokenCountFilter;
pub use headers::TokenUsageHeadersFilter;
use praxis_filter::HttpFilterContext;

/// Metadata key for the input token count.
const META_TOKEN_INPUT: &str = "token.input";

/// Metadata key for the output token count.
const META_TOKEN_OUTPUT: &str = "token.output";

/// Metadata key for the total token count.
const META_TOKEN_TOTAL: &str = "token.total";

/// Unified token usage extracted from an AI provider response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenUsage {
    /// Tokens in the input/prompt.
    input: u64,

    /// Tokens in the output/completion.
    output: u64,

    /// Total tokens.
    total: u64,
}

impl TokenUsage {
    /// Creates normalized usage, computing a saturating total when omitted.
    fn new(input: u64, output: u64, total: Option<u64>) -> Self {
        Self {
            input,
            output,
            total: total.unwrap_or_else(|| input.saturating_add(output)),
        }
    }

    /// Returns the normalized input token count.
    fn input_tokens(self) -> u64 {
        self.input
    }

    /// Returns the normalized output token count.
    fn output_tokens(self) -> u64 {
        self.output
    }

    /// Returns the provider-supplied or computed total token count.
    fn total_tokens(self) -> u64 {
        self.total
    }
}

/// Stores normalized token usage for downstream filters, logging, and metrics.
fn set_token_usage(ctx: &mut HttpFilterContext<'_>, input: u64, output: u64, total: Option<u64>) {
    let total = total.unwrap_or_else(|| input.saturating_add(output));

    ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
    ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
    ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
}
