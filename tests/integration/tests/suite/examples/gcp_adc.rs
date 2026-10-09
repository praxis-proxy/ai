// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Tests for the GCP ADC example configuration.
//!
//! The full "token injected, reaches upstream" path is covered with
//! test-only loopback metadata and token endpoints in `filters/src/gcp/`.
//! Production config must keep the cleartext credential request pinned to
//! Google's metadata hostname and the signed key-file assertion pinned to
//! Google's token endpoint, so these tests exercise what the shipped build
//! accepts and rejects at construct time. A request that mints a token from
//! a key file would reach `oauth2.googleapis.com`, which these hermetic
//! tests never contact.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use praxis_core::config::Config;
use praxis_test_utils::{free_port, http_send, parse_status, start_header_echo_backend};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// The token endpoint Google writes into every service-account key file.
const GOOGLE_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

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
    let patched = example_with_filter_fields("        metadata_host: 127.0.0.1:1");
    let error = resolve(&patched).expect_err("loopback metadata host must be rejected");
    assert!(
        error.contains("metadata_host"),
        "validation error should identify metadata_host: {error}"
    );
}

#[test]
fn gcp_adc_rejects_incomplete_key_file_at_config_time() {
    let credentials = tempfile::tempdir().expect("create credential directory");
    let credentials_file = credentials.path().join("service-account.json");
    std::fs::write(
        &credentials_file,
        r#"{"type":"service_account","client_email":"sa@example.com"}"#,
    )
    .expect("write service-account credentials");

    let error = resolve(&key_file_example(&credentials_file))
        .expect_err("a key file without private_key and token_uri must fail when the pipeline is built");
    assert!(
        error.contains("private_key"),
        "validation error should name the missing private_key: {error}"
    );
}

#[test]
fn gcp_adc_complete_key_file_resolves() {
    let credentials = tempfile::tempdir().expect("create credential directory");
    let credentials_file = write_service_account_key_file(credentials.path(), GOOGLE_TOKEN_URI);

    resolve(&key_file_example(&credentials_file))
        .unwrap_or_else(|error| panic!("a complete service-account key file must build the pipeline: {error}"));
}

#[test]
fn gcp_adc_rejects_loopback_token_uri() {
    let credentials = tempfile::tempdir().expect("create credential directory");
    let credentials_file = write_service_account_key_file(credentials.path(), "http://127.0.0.1:1/token");

    let error = resolve(&key_file_example(&credentials_file))
        .expect_err("the shipped build must not send the signed assertion to a loopback token endpoint");
    assert!(
        error.contains("token_uri"),
        "validation error should identify token_uri: {error}"
    );
}

#[test]
fn gcp_adc_cluster_scope_leaves_other_clusters_untouched() {
    let backend = start_header_echo_backend();
    let proxy_port = free_port();
    let path = praxis_test_utils::example_config_path("gcp-adc.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let patched =
        praxis_test_utils::patch_yaml(&yaml, proxy_port, &HashMap::from([("127.0.0.1:3000", backend.port())]));
    // The router selects `vertex`, which is outside this scope, so the filter
    // must neither fetch a token nor reject the request.
    let patched = patched.replace(
        "      - filter: gcp_adc",
        "      - filter: gcp_adc\n        source: metadata\n        clusters: [vertex-other]",
    );
    let config = Config::from_yaml(&patched).unwrap_or_else(|error| panic!("parse gcp-adc.yaml: {error}"));

    let proxy = praxis_test_utils::start_proxy(&config);
    let raw = http_send(
        proxy.addr(),
        "POST /v1/models HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Length: 15\r\n\
         Connection: close\r\n\r\n\
         {\"prompt\":\"hi\"}",
    );

    assert_eq!(
        parse_status(&raw),
        200,
        "a request routed outside the cluster scope must be forwarded without token acquisition: {raw}"
    );
    assert!(
        !raw.to_ascii_lowercase().contains("authorization:"),
        "no Authorization header may be injected for an out-of-scope cluster: {raw}"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

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

/// The example config with `fields` (already indented) added to its
/// `gcp_adc` entry.
fn example_with_filter_fields(fields: &str) -> String {
    let path = praxis_test_utils::example_config_path("gcp-adc.yaml");
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    yaml.replace("      - filter: gcp_adc", &format!("      - filter: gcp_adc\n{fields}"))
}

/// The example config switched to `source: key_file` at `credentials_file`.
fn key_file_example(credentials_file: &Path) -> String {
    example_with_filter_fields(&format!(
        "        source: key_file\n        credentials_file: {}",
        credentials_file.display()
    ))
}

/// Write a complete `type: service_account` key file into `dir`, around an
/// RSA key generated for this test alone, so no key material is committed.
fn write_service_account_key_file(dir: &Path, token_uri: &str) -> PathBuf {
    let private_key_pem = openssl::rsa::Rsa::generate(2048)
        .and_then(openssl::pkey::PKey::from_rsa)
        .and_then(|key| key.private_key_to_pem_pkcs8())
        .expect("generate ephemeral service-account key");
    let key_file = serde_json::json!({
        "type": "service_account",
        "client_email": "praxis-test@example-project.iam.gserviceaccount.com",
        "private_key": String::from_utf8(private_key_pem).expect("PEM is UTF-8"),
        "token_uri": token_uri,
    });
    let path = dir.join("service-account.json");
    std::fs::write(&path, key_file.to_string()).expect("write service-account key file");
    path
}
