// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Store-filter discovery for serving transports.

use std::collections::{HashMap, HashSet};

use praxis_ai_store::BackendError;
use praxis_core::config::{ChainRef, FilterEntry};

/// Parse the nested filter entries owned by an iterative request router.
fn iterative_step_filters(entry: &FilterEntry) -> Vec<Result<FilterEntry, BackendError>> {
    if entry.filter_type != "iterative_request_router" {
        return Vec::new();
    }
    entry
        .config
        .get("steps")
        .and_then(serde_yaml::Value::as_sequence)
        .into_iter()
        .flatten()
        .filter_map(|step| step.get("filters"))
        .flat_map(|value| {
            value.as_sequence().map_or_else(
                || vec![Err(invalid_step_filters())],
                |filters| {
                    filters
                        .iter()
                        .map(|filter| serde_yaml::from_value(filter.clone()).map_err(|_error| invalid_step_filters()))
                        .collect()
                },
            )
        })
        .collect()
}

/// Nested filter diagnostics must not disclose configuration values.
fn invalid_step_filters() -> BackendError {
    BackendError::Config("invalid iterative-router store filters".to_owned())
}

/// Retain valid nested entries and expose each parse error for caller policy.
fn collect_step_configs(
    entry: &FilterEntry,
    chains: &HashMap<&str, &[FilterEntry]>,
    filter_type: &str,
    visited: &mut HashSet<String>,
    found: &mut Vec<Result<serde_yaml::Value, BackendError>>,
) {
    for nested in iterative_step_filters(entry) {
        match nested {
            Ok(nested) => collect_store_configs(std::slice::from_ref(&nested), chains, filter_type, visited, found),
            Err(error) => found.push(Err(error)),
        }
    }
}

/// Collect store configs reachable from `entries`, following inline and named
/// branch chains plus filter-owned iterative-router steps. `visited` guards
/// against named-chain cycles and prevents duplicate named-chain traversal.
fn collect_store_configs(
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
    filter_type: &str,
    visited: &mut HashSet<String>,
    found: &mut Vec<Result<serde_yaml::Value, BackendError>>,
) {
    for entry in entries {
        if entry.filter_type == filter_type {
            found.push(Ok(entry.config.clone()));
        }
        collect_step_configs(entry, chains, filter_type, visited, found);
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

/// Find every `filter_type` reachable from the selected root chains.
///
/// Startup uses the complete set to reject ambiguous configurations before a
/// listener can route requests to a backend selected from only one branch.
/// Reload uses the same traversal so it compares every reachable store.
///
/// Parse errors remain alongside valid configs so each transport can retain its
/// existing strict or permissive discovery behavior.
pub(super) fn find_store_configs(
    roots: &[String],
    chains: &HashMap<&str, &[FilterEntry]>,
    filter_type: &str,
) -> Vec<Result<serde_yaml::Value, BackendError>> {
    let mut visited = HashSet::new();
    let mut found = Vec::new();
    for name in roots {
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
    fn malformed_iterative_filters_fail_without_disclosing_config() {
        let entry =
            serde_yaml::from_str("filter: iterative_request_router\nsteps: [{filters: secret}]").expect("filter entry");
        let error = iterative_step_filters(&entry)
            .into_iter()
            .next()
            .expect("parse result")
            .expect_err("malformed filter list");
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn inline_and_named_branches_are_discovered_once_despite_cycles() {
        let main: Vec<FilterEntry> = serde_yaml::from_str(
            "- filter: headers\n  branch_chains:\n    - name: branch\n      rejoin: terminal\n      chains:\n        - named\n        - name: inline\n          filters: [{filter: openai_responses_store, backend: sqlite}]",
        ).expect("main chain");
        let named: Vec<FilterEntry> = serde_yaml::from_str(
            "- filter: openai_responses_store\n  backend: sqlite\n  branch_chains:\n    - name: cycle\n      rejoin: terminal\n      chains: [main]",
        ).expect("named chain");
        let chains = HashMap::from([("main", main.as_slice()), ("named", named.as_slice())]);
        let roots = vec!["main".to_owned(), "named".to_owned()];
        let found = find_store_configs(&roots, &chains, "openai_responses_store")
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("discovery");
        assert_eq!(found.len(), 2);
    }

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

        let found = find_store_configs(&listener.filter_chains, &chains, "openai_conversations")
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("store discovery");

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
