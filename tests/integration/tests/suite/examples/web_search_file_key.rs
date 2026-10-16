// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the file-sourced web-search provider key example
//! configs.
//!
//! Both examples source the shared provider `api_key` from a projected `file:`
//! rather than `env_var:`. These tests prove the shipped configs parse and the
//! pipeline builds (reading the key file at startup), and that an unreadable key
//! file fails the build loud rather than dialing a provider unauthenticated.

use std::{collections::HashMap, io::Write as _, sync::Arc};

use praxis_core::config::Config;
use praxis_test_utils::example_config_path;

/// The projected-file path both example configs reference for the provider key.
const KEY_FILE_PLACEHOLDER: &str = "/var/run/secrets/web-search/api-key";

/// Resolve the full AI pipeline from raw YAML, surfacing any build error.
///
/// Exercises the real server pipeline-resolution path, so a config whose
/// file-sourced `api_key` cannot be read fails here rather than at request time.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client = praxis_test_utils::test_subrequest_client();
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(
        &config,
        &registry,
        &health,
        &kv_stores,
        &praxis_ai::InFlightRegistry::new(),
        &subrequest_client,
    )
    .map(|_pipelines| ())
    .map_err(|error| error.to_string())
}

/// Read an example config and point its `api_key.file` at `path`.
fn example_yaml_with_key_file(example: &str, path: &str) -> String {
    let yaml = std::fs::read_to_string(example_config_path(example)).unwrap_or_else(|e| panic!("read {example}: {e}"));
    let patched = yaml.replace(KEY_FILE_PLACEHOLDER, path);
    assert_ne!(
        patched, yaml,
        "example drift: {KEY_FILE_PLACEHOLDER} not found in {example}"
    );
    patched
}

/// Each example, keyed by the path passed to `example_config_path`.
const EXAMPLES: &[&str] = &[
    "openai/responses/web-search-file-key.yaml",
    "anthropic/web-search-file-key.yaml",
];

#[test]
fn web_search_file_key_examples_build() {
    for example in EXAMPLES {
        let mut key = tempfile::NamedTempFile::new().unwrap();
        write!(key, "test-web-search-key").unwrap();
        let patched = example_yaml_with_key_file(example, key.path().to_str().unwrap());
        resolve(&patched).unwrap_or_else(|e| panic!("file-key example {example} must build: {e}"));
    }
}

#[test]
fn web_search_file_key_examples_fail_build_without_key_file() {
    // The examples reference a file that does not exist; building must fail loud
    // at startup rather than silently dialing the provider unauthenticated.
    for example in EXAMPLES {
        let patched = example_yaml_with_key_file(example, "/nonexistent/praxis-web-search-key-xyz");
        assert!(
            resolve(&patched).is_err(),
            "an unreadable key file must fail the pipeline build for {example}"
        );
    }
}
