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

/// Metadata key for input tokens served from the provider's prompt cache.
const META_TOKEN_CACHE_READ: &str = "token.cache_read";

/// Metadata key for input tokens written to the provider's prompt cache.
const META_TOKEN_CACHE_WRITE: &str = "token.cache_write";

/// Unified token usage extracted from an AI provider response.
///
/// Providers that support prompt caching also report how much of the input was
/// served from, or written to, their cache. Those tokens are priced differently
/// from fresh input — typically a fraction of the fresh rate for a cache read
/// and a premium for a cache write — so they are carried alongside the totals
/// rather than folded away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TokenUsage {
    /// Tokens in the input/prompt, cached tokens included.
    input: u64,

    /// Tokens in the output/completion.
    output: u64,

    /// Total tokens.
    total: u64,

    /// Input tokens served from the provider's prompt cache.
    cache_read: u64,

    /// Input tokens written to the provider's prompt cache.
    cache_write: u64,
}

impl TokenUsage {
    /// Creates normalized usage, computing a saturating total when omitted.
    ///
    /// Records no cache activity; use [`Self::with_cache`] for providers that
    /// report a prompt cache breakdown.
    fn new(input: u64, output: u64, total: Option<u64>) -> Self {
        Self {
            input,
            output,
            total: total.unwrap_or_else(|| input.saturating_add(output)),
            ..Self::default()
        }
    }

    /// Attaches prompt cache counts, which break down [`Self::input_tokens`]
    /// rather than adding to it.
    fn with_cache(mut self, cache_read: u64, cache_write: u64) -> Self {
        self.cache_read = cache_read;
        self.cache_write = cache_write;
        self
    }

    /// Returns the normalized input token count, cached tokens included.
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

    /// Returns input tokens served from the provider's prompt cache.
    fn cache_read_tokens(self) -> u64 {
        self.cache_read
    }

    /// Returns input tokens written to the provider's prompt cache.
    fn cache_write_tokens(self) -> u64 {
        self.cache_write
    }
}

/// Token counts recovered from a single streaming event.
///
/// Providers spread usage across the event stream and omit fields that did not
/// change, so every count is optional and merged as it arrives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StreamingTokens {
    /// Input tokens reported by the event, cached tokens included.
    input: Option<u64>,

    /// Output tokens reported by the event.
    output: Option<u64>,

    /// Input tokens the event reports as served from the prompt cache.
    cache_read: Option<u64>,

    /// Input tokens the event reports as written to the prompt cache.
    cache_write: Option<u64>,
}

/// Stores normalized token usage for downstream filters, logging, and metrics.
fn set_token_usage(ctx: &mut HttpFilterContext<'_>, input: u64, output: u64, total: Option<u64>) {
    let total = total.unwrap_or_else(|| input.saturating_add(output));

    ctx.set_metadata(META_TOKEN_INPUT, input.to_string());
    ctx.set_metadata(META_TOKEN_OUTPUT, output.to_string());
    ctx.set_metadata(META_TOKEN_TOTAL, total.to_string());
}

/// Stores the prompt cache breakdown of the input tokens.
///
/// Recorded separately from [`set_token_usage`] because cached tokens are a
/// subset of the input count, not an addition to it.
fn set_cache_token_usage(ctx: &mut HttpFilterContext<'_>, cache_read: u64, cache_write: u64) {
    ctx.set_metadata(META_TOKEN_CACHE_READ, cache_read.to_string());
    ctx.set_metadata(META_TOKEN_CACHE_WRITE, cache_write.to_string());
}
