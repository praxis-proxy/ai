// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Store-filter discovery shared by startup provisioning and reload validation.

#![cfg(feature = "store")]

use std::collections::{HashMap, HashSet};

use praxis_core::config::{ChainRef, FilterEntry, Listener};

/// Parse the nested filter entries owned by an iterative request router.
fn iterative_step_filters(entry: &FilterEntry) -> Vec<FilterEntry> {
    if entry.filter_type != "iterative_request_router" {
        return Vec::new();
    }
    entry
        .config
        .get("steps")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(|step| step.get("filters").and_then(serde_yaml::Value::as_sequence))
        .flatten()
        .filter_map(|filter| serde_yaml::from_value(filter.clone()).ok())
        .collect()
}

/// Collect store configs reachable from `entries`, following inline and named
/// branch chains plus filter-owned iterative-router steps. `visited` guards
/// against named-chain cycles and prevents duplicate named-chain traversal.
fn collect_store_configs(
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
    filter_type: &str,
    visited: &mut HashSet<String>,
    found: &mut Vec<serde_yaml::Value>,
) {
    for entry in entries {
        if entry.filter_type == filter_type {
            found.push(entry.config.clone());
        }
        let step_filters = iterative_step_filters(entry);
        if !step_filters.is_empty() {
            collect_store_configs(&step_filters, chains, filter_type, visited, found);
        }
        let Some(branches) = entry.branch_chains.as_ref() else {
            continue;
        };
        for chain in branches.iter().flat_map(|branch| branch.chains.iter()) {
            let nested = match chain {
                ChainRef::Inline { filters, .. } => filters.as_slice(),
                ChainRef::Named(name) => {
                    if visited.insert(name.clone()) {
                        chains.get(name.as_str()).copied().unwrap_or_default()
                    } else {
                        &[]
                    }
                },
            };
            collect_store_configs(nested, chains, filter_type, visited, found);
        }
    }
}

/// Find every `filter_type` across a listener's chains, following branch chains.
///
/// Startup uses the complete set to reject ambiguous configurations before a
/// listener can route requests to a backend selected from only one branch.
/// Reload uses the same traversal so it compares every reachable store.
pub(crate) fn find_listener_store_configs(
    listener: &Listener,
    chains: &HashMap<&str, &[FilterEntry]>,
    filter_type: &str,
) -> Vec<serde_yaml::Value> {
    let mut visited = HashSet::new();
    let mut found = Vec::new();
    for name in &listener.filter_chains {
        if !visited.insert(name.clone()) {
            continue;
        }
        let Some(filters) = chains.get(name.as_str()).copied() else {
            continue;
        };
        collect_store_configs(filters, chains, filter_type, &mut visited, &mut found);
    }
    found
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests")]
mod tests {
    use praxis_core::config::Config;

    use super::*;

    #[test]
    #[expect(clippy::too_many_lines, reason = "complete iterative-router YAML fixture")]
    fn finds_store_inside_iterative_router_step() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: iterative_request_router
        initial_step: local
        steps:
          - name: local
            filters:
              - filter: openai_conversations
                backend: sqlite
                database_url: "sqlite::memory:"
            on_result:
              - default: true
                done: true
"#,
        )
        .expect("iterative store config");
        let chains = config
            .filter_chains
            .iter()
            .map(|chain| (chain.name.as_str(), chain.filters.as_slice()))
            .collect();
        let listener = config.listeners.first().expect("listener");

        let found = find_listener_store_configs(listener, &chains, "openai_conversations");

        assert_eq!(found.len(), 1);
        assert_eq!(
            found
                .first()
                .and_then(|value| value.get("backend"))
                .and_then(serde_yaml::Value::as_str),
            Some("sqlite")
        );
    }
}
