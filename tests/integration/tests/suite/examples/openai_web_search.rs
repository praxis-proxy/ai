// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Integration coverage for the direct-vLLM web-search example configuration.

use praxis_test_utils::example_config_path;

#[test]
fn web_search_example_config_parses_and_registers_filter() {
    let yaml = std::fs::read_to_string(example_config_path("openai/responses/web-search-vllm.yaml"))
        .expect("web-search example config should exist");
    let config = praxis_core::config::Config::from_yaml(&yaml).expect("web-search example config should parse");
    let filters = config
        .filter_chains
        .iter()
        .flat_map(|chain| chain.filters.iter())
        .collect::<Vec<_>>();
    assert!(
        filters.iter().any(|filter| filter.filter_type == "openai_web_search"),
        "example config should register openai_web_search"
    );
}
