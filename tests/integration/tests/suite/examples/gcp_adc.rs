// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the GCP ADC example configuration.
//!
//! The full "token injected, reaches upstream" path is covered with a
//! test-only loopback metadata server in `filters/src/gcp/`. Production config
//! must keep the cleartext credential request pinned to Google's metadata
//! hostname.

use std::{collections::HashMap, sync::Arc};

use praxis_core::config::Config;

/// Resolve the complete AI pipeline so filter construction errors are visible.
fn resolve(yaml: &str) -> Result<(), String> {
    let config = Config::from_yaml(yaml).map_err(|error| error.to_string())?;
    let health = Arc::new(HashMap::new());
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    let subrequest_client = praxis_test_utils::test_subrequest_client();
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    praxis_ai::resolve_pipelines(&config, &registry, &health, &kv_stores, &subrequest_client)
        .map(|_pipelines| ())
        .map_err(|error| error.to_string())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn gcp_adc_config_parses() {
    let config = super::load_example_config("gcp-adc.yaml", 29912, HashMap::from([("127.0.0.1:3000", 29913_u16)]));

    assert_eq!(config.listeners.len(), 1, "should have 1 listener");
    assert_eq!(&*config.listeners[0].name, "gateway", "listener name should be gateway");
}

#[test]
fn gcp_adc_rejects_loopback_metadata_host() {
    let path = praxis_test_utils::example_config_path("gcp-adc.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = yaml.replace(
        "      - filter: gcp_adc",
        "      - filter: gcp_adc\n        metadata_host: 127.0.0.1:1",
    );
    let error = resolve(&patched).expect_err("loopback metadata host must be rejected");
    assert!(
        error.contains("metadata_host"),
        "validation error should identify metadata_host: {error}"
    );
}
