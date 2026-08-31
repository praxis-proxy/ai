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
///
/// `pub(crate)` only under `token-rate-limit-filter`, so that experimental
/// filter's reconciliation path can reference this constant directly
/// instead of duplicating the string literal — see the duplication risk
/// this avoids: [`ai#351`](https://github.com/praxis-proxy/ai/issues/351)
/// (cached-token double-counting caused by a second, independent parsing
/// path drifting from this one). Private otherwise, so this stable filter's
/// public surface is unaffected when the experimental feature is disabled.
#[cfg(feature = "token-rate-limit-filter")]
pub(crate) const META_TOKEN_TOTAL: &str = "token.total";
/// Metadata key for the total token count.
#[cfg(not(feature = "token-rate-limit-filter"))]
const META_TOKEN_TOTAL: &str = "token.total";

/// Metadata key signaling that usage could not be captured because the
/// response exceeded the configured capture limit. Absent on success,
/// including when the provider genuinely reported no usage — consumers
/// must not treat "no counts" the same as "counts unavailable."
const META_TOKEN_STATUS: &str = "token.status";

/// Value of [`META_TOKEN_STATUS`] when capture was abandoned due to
/// exceeding the configured size limit.
const TOKEN_STATUS_OVERFLOW: &str = "overflow";

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
    ///
    /// `None` when the provider did not report the count at all, which is
    /// distinct from a reported zero: an absent count means the response
    /// carried no cache information, a zero means the cache was not hit.
    cache_read: Option<u64>,

    /// Input tokens written to the provider's prompt cache.
    ///
    /// `None` when the provider did not report the count at all, as for
    /// providers whose API has no cache-write concept.
    cache_write: Option<u64>,
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
    ///
    /// Pass `None` for a count the provider did not report, so that it stays
    /// distinguishable from a count the provider reported as zero.
    fn with_cache(mut self, cache_read: Option<u64>, cache_write: Option<u64>) -> Self {
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

    /// Returns input tokens served from the provider's prompt cache, or `None`
    /// when the provider did not report the count.
    fn cache_read_tokens(self) -> Option<u64> {
        self.cache_read
    }

    /// Returns input tokens written to the provider's prompt cache, or `None`
    /// when the provider did not report the count.
    fn cache_write_tokens(self) -> Option<u64> {
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

/// Marks token usage as unavailable due to exceeding the capture limit,
/// distinguishable from a genuine zero-usage response.
fn set_token_status_overflow(ctx: &mut HttpFilterContext<'_>) {
    ctx.set_metadata(META_TOKEN_STATUS, TOKEN_STATUS_OVERFLOW.to_owned());
}

/// Stores the prompt cache breakdown of the input tokens.
///
/// Recorded separately from [`set_token_usage`] because cached tokens are a
/// subset of the input count, not an addition to it.
///
/// Each count is written only when the provider reported it, so a consumer can
/// tell "no cache information in this response" from "cache reported zero".
fn set_cache_token_usage(ctx: &mut HttpFilterContext<'_>, cache_read: Option<u64>, cache_write: Option<u64>) {
    if let Some(cache_read) = cache_read {
        ctx.set_metadata(META_TOKEN_CACHE_READ, cache_read.to_string());
    }

    if let Some(cache_write) = cache_write {
        ctx.set_metadata(META_TOKEN_CACHE_WRITE, cache_write.to_string());
    }
}
