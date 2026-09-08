// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral web-search provider support.

pub(crate) mod config;
pub(crate) mod provider;

use std::fmt::Write as _;

pub(crate) use config::{
    OpenAiWebSearchConfig, SearchContextSize, ValidatedConfig, WebSearchFilterConfig, build_config,
};
pub(crate) use provider::{SearchClient, SearchOutcome, SearchResult};

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
