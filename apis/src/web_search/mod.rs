// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral web-search provider support.

pub(crate) mod config;
pub(crate) mod provider;

use std::fmt::Write as _;

#[cfg(feature = "openai-responses")]
pub(crate) use config::OpenAiWebSearchConfig;
pub(crate) use config::{SearchContextSize, ValidatedConfig, WebSearchFilterConfig, build_config};
#[cfg(test)]
use praxis_filter::FilterError;
pub(crate) use provider::{CalloutContext, SearchClient, SearchOutcome, SearchResult};

/// Test-only: build a minimal bound outbound pipeline for a callout unit test.
///
/// Holds a single observable `request_id` builtin, standing in for the
/// cross-cutting filters a real deployment binds. No seed filter is installed:
/// the [`FilteredSubrequestExecutor`] seeds `filter_ctx.upstream` from the
/// `StagedUpstream` the [`SearchClient`] stages (and re-pins it after the request
/// phase), so the chain needs no upstream-selecting filter. Private upstreams are
/// permitted so tests can dial loopback mocks; the executor still enforces
/// SSRF/TLS/`Host` centrally.
///
/// # Errors
///
/// Returns [`FilterError`] if the pipeline cannot be built.
///
/// [`FilteredSubrequestExecutor`]: praxis_filter::FilteredSubrequestExecutor
#[cfg(test)]
pub(crate) fn test_outbound_pipeline() -> Result<praxis_filter::FilterPipeline, FilterError> {
    let registry = praxis_filter::FilterRegistry::with_builtins();
    let mut entries: Vec<praxis_filter::FilterEntry> =
        serde_yaml::from_str("- filter: request_id\n").map_err(|e| FilterError::from(e.to_string()))?;
    let mut pipeline = praxis_filter::FilterPipeline::build(&mut entries, &registry)?;
    pipeline.set_allow_private_upstreams(true);
    Ok(pipeline)
}

/// Bounded tool-result message fed to the model when a search provider fails,
/// so both provider loops continue with a truthful failure instead of rejecting.
pub(crate) const SEARCH_UNAVAILABLE: &str = "Web search unavailable.";

/// Hosted web-search tool `type` discriminators recognized by the local web
/// executor, including every dated preview variant.
///
/// Single source of truth for identifying a Responses web-search tool so the
/// translation, response-synthesis, and tool-parsing paths cannot silently
/// drift when a future variant is added.
pub(crate) const WEB_SEARCH_TOOL_TYPES: [&str; 4] = [
    "web_search",
    "web_search_preview",
    "web_search_preview_2025_03_11",
    "web_search_2025_08_26",
];

/// Return whether a tool `type` discriminator is a hosted web-search tool.
pub(crate) fn is_web_search_tool_type(tool_type: &str) -> bool {
    WEB_SEARCH_TOOL_TYPES.contains(&tool_type)
}

/// Format search results as readable text for a model prompt.
pub(crate) fn format_search_results(results: &[SearchResult]) -> String {
    let mut output = String::with_capacity(results.len() * 200);
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push_str("\n\n");
        }
        let _infallible = write!(
            output,
            "[{}] {}\n{}\n{}",
            index + 1,
            result.title,
            result.url,
            result.snippet
        );
    }
    output
}
