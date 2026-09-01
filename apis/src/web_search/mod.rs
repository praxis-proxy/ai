// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral web-search provider support.

pub(crate) mod config;
pub(crate) mod provider;

use std::fmt::Write as _;

pub(crate) use config::{SearchContextSize, ValidatedConfig, WebSearchFilterConfig, build_config};
pub(crate) use provider::{SearchClient, SearchOutcome, SearchResult};

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
