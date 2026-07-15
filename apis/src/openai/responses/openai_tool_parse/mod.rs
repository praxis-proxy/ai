// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Tool parse filter for the Responses API.
//!
//! Parses the `tools` array and `tool_choice` from a Responses API
//! request body. Classifies each tool by type (function, `web_search`,
//! `file_search`, `code_interpreter`, `computer_use`,
//! `image_generation`, `tool_search`, MCP) and promotes summary facts
//! to metadata and filter results for branch conditions.
//!
//! Does not mutate the request body.
//! `StreamBuffer` pre-read makes these body-derived facts available
//! before branch evaluation.

mod config;
pub(crate) mod parser;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    builtins::http::value_safety::is_safe_promoted_value, parse_filter_config,
};
use tracing::{debug, trace};

use self::{
    config::{ToolParseConfig, build_config},
    parser::{ParsedTools, parse_tools},
};
use crate::classifier::is_responses_create;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum length of a body-derived value promoted to filter results.
const MAX_PROMOTED_VALUE_LEN: usize = 256;

// -----------------------------------------------------------------------------
// ToolParseFilter
// -----------------------------------------------------------------------------

/// Parses tool definitions and `tool_choice` from Responses API
/// request bodies and promotes routing facts to metadata and filter
/// results without mutating the body.
///
/// # YAML
///
/// ```yaml
/// filter: openai_tool_parse
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_tool_parse
/// max_body_bytes: 67108864
/// ```
pub struct ToolParseFilter {
    /// Parsed and validated configuration.
    config: ToolParseConfig,
}

impl ToolParseFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    ///
    /// [`FilterError`]: praxis_filter::FilterError
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ToolParseConfig = parse_filter_config("openai_tool_parse", config)?;
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for ToolParseFilter {
    fn name(&self) -> &'static str {
        "openai_tool_parse"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Re-promote filter_results from metadata. When a preceding
        // filter's branch evaluation runs first, the pipeline clears
        // filter_results before this filter's branches are checked.
        // Metadata survives across phases, so we rebuild from it.
        restore_filter_results(ctx)?;
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if !is_responses_create(&ctx.request.method, ctx.request.uri.path()) {
            return Ok(FilterAction::Continue);
        }

        let bytes = match body.as_ref() {
            Some(b) => b.as_ref(),
            None => &[],
        };

        let parsed = parse_tools(bytes);

        debug!(
            function_count = parsed.function_count,
            builtin_count = parsed.builtin_count,
            mcp_count = parsed.mcp_count,
            has_tools = parsed.has_tools(),
            tool_choice = ?parsed.tool_choice.as_ref().map(parser::ToolChoice::as_str),
            "parsed tool definitions"
        );

        if !parsed.has_tools() {
            return Ok(FilterAction::Continue);
        }

        write_metadata(ctx, &parsed);
        promote_filter_results(ctx, &parsed)?;

        Ok(FilterAction::Release)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Write durable metadata that persists across all Pingora lifecycle phases.
fn write_metadata(ctx: &mut HttpFilterContext<'_>, parsed: &ParsedTools) {
    if parsed.has_tools() {
        ctx.set_metadata("openai_tool_parse.has_tools", "true");
    }

    if parsed.function_count > 0 {
        ctx.set_metadata("openai_tool_parse.function_count", parsed.function_count.to_string());
    }

    write_tool_presence_metadata(ctx, parsed);

    if let Some(tc) = &parsed.tool_choice {
        if let Some(val) = promotable_tool_choice_value(tc) {
            ctx.set_metadata("openai_tool_parse.tool_choice", val);
        }
        if let Some(type_str) = promotable_tool_choice_type(tc) {
            ctx.set_metadata("openai_tool_parse.tool_choice_type", type_str);
        }
    }
}

/// Write per-tool-type presence flags to metadata.
fn write_tool_presence_metadata(ctx: &mut HttpFilterContext<'_>, parsed: &ParsedTools) {
    let flags: &[(&str, bool)] = &[
        ("openai_tool_parse.has_web_search", parsed.has_web_search()),
        ("openai_tool_parse.has_file_search", parsed.has_file_search()),
        ("openai_tool_parse.has_mcp", parsed.has_mcp()),
        ("openai_tool_parse.has_code_interpreter", parsed.has_code_interpreter()),
        ("openai_tool_parse.has_computer_use", parsed.has_computer_use()),
        ("openai_tool_parse.has_image_generation", parsed.has_image_generation()),
        ("openai_tool_parse.has_tool_search", parsed.has_tool_search()),
    ];
    for &(key, present) in flags {
        if present {
            ctx.set_metadata(key, "true");
        }
    }
}

/// Re-promote filter results from metadata written during the body phase.
///
/// The pipeline clears `filter_results` after evaluating each filter's
/// branch conditions. When `openai_tool_parse` is not the first filter in the
/// chain, a preceding filter's (possibly empty) branch evaluation
/// clears results before `openai_tool_parse`'s branches are checked. Metadata
/// persists across phases, so we rebuild `filter_results` from it.
fn restore_filter_results(ctx: &mut HttpFilterContext<'_>) -> Result<(), FilterError> {
    let any = restore_presence_flags(ctx)? | restore_function_count(ctx)? | restore_tool_choice(ctx)?;

    if any {
        trace!("restored filter_results from metadata");
    }

    Ok(())
}

/// Restore boolean presence flags from metadata to filter results.
fn restore_presence_flags(ctx: &mut HttpFilterContext<'_>) -> Result<bool, FilterError> {
    const FLAGS: &[(&str, &str)] = &[
        ("openai_tool_parse.has_tools", "has_tools"),
        ("openai_tool_parse.has_web_search", "has_web_search"),
        ("openai_tool_parse.has_file_search", "has_file_search"),
        ("openai_tool_parse.has_mcp", "has_mcp"),
        ("openai_tool_parse.has_code_interpreter", "has_code_interpreter"),
        ("openai_tool_parse.has_computer_use", "has_computer_use"),
        ("openai_tool_parse.has_image_generation", "has_image_generation"),
        ("openai_tool_parse.has_tool_search", "has_tool_search"),
    ];

    let mut any = false;
    for &(meta_key, result_key) in FLAGS {
        if ctx.get_metadata(meta_key).is_some_and(|v| v == "true") {
            let results = ctx.filter_results.entry("openai_tool_parse").or_default();
            results.set(result_key, "true")?;
            any = true;
        }
    }
    Ok(any)
}

/// Restore `function_count` from metadata.
fn restore_function_count(ctx: &mut HttpFilterContext<'_>) -> Result<bool, FilterError> {
    if let Some(fc) = ctx.get_metadata("openai_tool_parse.function_count") {
        let fc = fc.to_owned();
        let results = ctx.filter_results.entry("openai_tool_parse").or_default();
        results.set("function_count", fc)?;
        return Ok(true);
    }
    Ok(false)
}

/// Restore `tool_choice` and `tool_choice_type` from metadata.
fn restore_tool_choice(ctx: &mut HttpFilterContext<'_>) -> Result<bool, FilterError> {
    let mut any = false;

    if let Some(tc) = ctx.get_metadata("openai_tool_parse.tool_choice") {
        let tc = tc.to_owned();
        let results = ctx.filter_results.entry("openai_tool_parse").or_default();
        results.set("tool_choice", tc)?;
        any = true;
    }

    if let Some(tct) = ctx.get_metadata("openai_tool_parse.tool_choice_type") {
        let tct = tct.to_owned();
        let results = ctx.filter_results.entry("openai_tool_parse").or_default();
        results.set("tool_choice_type", tct)?;
        any = true;
    }

    Ok(any)
}

/// Promote tool facts to filter results for branch conditions.
fn promote_filter_results(ctx: &mut HttpFilterContext<'_>, parsed: &ParsedTools) -> Result<(), FilterError> {
    let results = ctx.filter_results.entry("openai_tool_parse").or_default();

    if parsed.has_tools() {
        results.set("has_tools", "true")?;
    }

    if parsed.function_count > 0 {
        results.set("function_count", parsed.function_count.to_string())?;
    }

    promote_tool_presence_results(results, parsed)?;

    if let Some(tc) = &parsed.tool_choice {
        if let Some(val) = promotable_tool_choice_value(tc) {
            results.set("tool_choice", val.to_owned())?;
        }
        if let Some(type_str) = promotable_tool_choice_type(tc) {
            results.set("tool_choice_type", type_str.to_owned())?;
        }
    }

    Ok(())
}

/// Promote per-tool-type presence flags to filter results.
fn promote_tool_presence_results(
    results: &mut praxis_filter::FilterResultSet,
    parsed: &ParsedTools,
) -> Result<(), FilterError> {
    let flags: &[(&str, bool)] = &[
        ("has_web_search", parsed.has_web_search()),
        ("has_file_search", parsed.has_file_search()),
        ("has_mcp", parsed.has_mcp()),
        ("has_code_interpreter", parsed.has_code_interpreter()),
        ("has_computer_use", parsed.has_computer_use()),
        ("has_image_generation", parsed.has_image_generation()),
        ("has_tool_search", parsed.has_tool_search()),
    ];
    for &(key, present) in flags {
        if present {
            results.set(key, "true")?;
        }
    }
    Ok(())
}

/// Return a safe, bounded `tool_choice` value for metadata and
/// filter results.
fn promotable_tool_choice_value(choice: &parser::ToolChoice) -> Option<&str> {
    is_safe_promoted(choice.as_str())
}

/// Return a safe, bounded `tool_choice_type` value for metadata
/// and filter results.
fn promotable_tool_choice_type(choice: &parser::ToolChoice) -> Option<&str> {
    choice.type_str().and_then(is_safe_promoted)
}

/// Check that a value is safe and bounded for promotion.
fn is_safe_promoted(val: &str) -> Option<&str> {
    (val.len() <= MAX_PROMOTED_VALUE_LEN && is_safe_promoted_value(val)).then_some(val)
}
