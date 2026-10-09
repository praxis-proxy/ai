// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Check-mode command helpers shared by `--validate` and `--dump`.

use praxis_core::config::Config;

use crate::dump;

// -----------------------------------------------------------------------------
// Shared Validation
// -----------------------------------------------------------------------------

/// Load and fully validate configuration without starting the server.
///
/// Shared by `--validate` and `--dump`. Runs the same validation
/// checks used during server startup: log override validation,
/// filter factory instantiation, chain expansion, ordering checks,
/// and body-limit application.
///
/// # Errors
///
/// Returns an error if loading or validation fails.
pub(crate) fn load_and_validate_for_cli(
    explicit: Option<&str>,
) -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
    let config = praxis_ai::load_config(explicit)?;
    validate_config_for_startup(&config)?;
    Ok(config)
}

/// Validate a parsed configuration by building filter pipelines.
pub(crate) fn validate_config_for_startup(config: &Config) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    praxis_core::logging::validate_log_overrides(config)?;
    let subrequest_client = praxis_ai::create_subrequest_client(config);
    let registry = praxis_ai::build_full_registry(&subrequest_client);
    // The same refusal the server makes at boot, so `--validate` under
    // PRAXIS_REQUIRE_FIPS answers for the binary it runs as.
    if praxis_tls::provider::required()
        && let Some(reason) = praxis_ai::fips_blocker(&registry)
    {
        return Err(reason.into());
    }
    let health_registry = praxis_core::health::build_health_registry(&config.clusters);
    let kv_stores = praxis_core::kv::KvStoreRegistry::new();
    #[cfg(feature = "_store-backend")]
    praxis_ai::validate_pipelines_with_store_wiring(
        config,
        &registry,
        &health_registry,
        &kv_stores,
        &subrequest_client,
    )?;
    #[cfg(not(feature = "_store-backend"))]
    praxis_ai::resolve_pipelines(config, &registry, &health_registry, &kv_stores, &subrequest_client)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// Dump
// -----------------------------------------------------------------------------

/// Load, validate, and dump effective configuration to stdout.
///
/// # Errors
///
/// Returns an error if loading, validation, or serialization fails.
pub(crate) fn run_dump(explicit: Option<&str>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = load_and_validate_for_cli(explicit)?;
    let source = match explicit {
        Some(path) => path.to_owned(),
        None => default_config_source(),
    };
    let dump_model = dump::build_dump(&config, &source)?;
    dump::write_dump(&dump_model)
}

/// Determine the human-readable label for implicit config sources.
fn default_config_source() -> String {
    if std::path::Path::new("praxis.yaml").exists() {
        return "praxis.yaml".to_owned();
    }
    "<built-in default>".to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn validate_catches_invalid_log_overrides() {
        let err = Config::from_yaml(
            r#"
runtime:
  log_overrides:
    "invalid module": "info"
    "praxis_core": "invalid_level"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters: []
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("invalid module") || err.contains("log_overrides"),
            "error should mention the invalid log override: {err}"
        );
    }

    #[test]
    fn validate_rejects_unknown_filter_type() {
        // Validation builds a sub-request client, which needs the provider first.
        praxis_ai::install_crypto_provider();
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: nonexistent_filter_type
"#,
        )
        .unwrap();
        let result = validate_config_for_startup(&config);
        assert!(result.is_err(), "unknown filter type should fail validation");
    }

    #[cfg(any(feature = "store-postgres", feature = "store-sqlite"))]
    #[test]
    #[expect(clippy::too_many_lines, reason = "complete conflicting listener fixture")]
    fn validate_rejects_conflicting_listener_store_configs() {
        praxis_ai::install_crypto_provider();
        #[cfg(feature = "store-sqlite")]
        let (backend, first_url, second_url) = ("sqlite", "sqlite:///first.db", "sqlite:///second.db");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend, first_url, second_url) = (
            "postgres",
            "postgresql://user:password@8.8.8.8/store",
            "postgresql://user:password@1.1.1.1/store",
        );
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_response_store
        backend: {backend}
        database_url: "{first_url}"
        responses_table: responses
        conversations_table: conversations
      - filter: openai_response_store
        backend: {backend}
        database_url: "{second_url}"
        responses_table: responses
        conversations_table: conversations
"#
        ))
        .expect("conflicting store config should parse");

        let error =
            validate_config_for_startup(&config).expect_err("CLI validation must match startup store validation");
        assert!(error.to_string().contains("conflicting stores"), "got: {error}");
    }

    #[cfg(any(feature = "store-postgres", feature = "store-sqlite"))]
    #[test]
    #[expect(clippy::too_many_lines, reason = "complete provider trust and store fixture")]
    fn validate_preserves_first_position_provider_trust_with_store_gate() {
        praxis_ai::install_crypto_provider();
        #[cfg(feature = "store-sqlite")]
        let (backend, database_url, backend_options) = ("sqlite", "sqlite::memory:", "");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend, database_url, backend_options) = (
            "postgres",
            "postgresql://user:password@8.8.8.8/store",
            "        ssl_mode: disable\n",
        );
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: provider
    address: "127.0.0.1:8080"
    filter_chains: [main]
    tls:
      certificates:
        - cert_path: /tmp/provider-cert.pem
          key_path: /tmp/provider-key.pem
      client_ca:
        ca_path: /tmp/provider-ca.pem
      client_cert_mode: require
filter_chains:
  - name: main
    filters:
      - filter: peer_identity_trust
        trusted_peers:
          - organization: ai-grid
      - filter: provider_route
        provider_id: test-provider
        routes:
          - candidate_id: test-candidate
            cluster: backend
            model: test-model
            paths: [/v1/responses]
      - filter: state_owner
        mode: single_tenant
        tenant_id: test
      - filter: openai_response_store
        backend: {backend}
        database_url: "{database_url}"
{backend_options}        responses_table: responses
        conversations_table: conversations
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["127.0.0.1:12345"]
insecure_options:
  allow_private_endpoints: true
  skip_pipeline_validation: true
"#
        ))
        .expect("provider store config should parse");

        validate_config_for_startup(&config).expect("the readiness gate must not displace first-position peer trust");
    }

    #[test]
    fn invalid_yaml_syntax_returns_error() {
        let result = Config::from_yaml("{{{{ not: valid: yaml: [");
        assert!(result.is_err(), "malformed YAML syntax should fail parsing");
    }

    #[test]
    fn empty_config_string_returns_error() {
        let result = Config::from_yaml("");
        assert!(result.is_err(), "empty config string should fail parsing");
    }

    #[test]
    fn config_with_invalid_chain_reference_returns_error() {
        let result = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [nonexistent_chain]
filter_chains:
  - name: main
    filters: []
"#,
        );
        assert!(
            result.is_err(),
            "listener referencing undefined chain should fail validation"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("nonexistent_chain"),
            "error should mention the missing chain name: {err}"
        );
    }

    #[test]
    fn validate_accepts_subrequest_circuit_breaker_config() {
        // Validation builds a sub-request client, which needs the provider first.
        praxis_ai::install_crypto_provider();
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
runtime:
  subrequest_circuit_breaker:
    consecutive_failures: 3
    recovery_window_secs: 30
"#,
        )
        .unwrap();
        assert!(
            config.runtime.subrequest_circuit_breaker.is_some(),
            "CLI validation path should parse runtime.subrequest_circuit_breaker"
        );
        validate_config_for_startup(&config).expect("configured circuit breaker should not fail startup validation");
    }

    static CWD_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    struct CwdGuard(std::path::PathBuf);

    impl CwdGuard {
        fn new(path: &std::path::Path) -> Self {
            let original = std::env::current_dir().unwrap();
            std::env::set_current_dir(path).unwrap();
            Self(original)
        }
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            std::env::set_current_dir(&self.0).expect("failed to restore working directory");
        }
    }

    #[test]
    fn default_config_source_returns_builtin_when_no_yaml() {
        let _lock = CWD_MUTEX.get_or_init(Default::default).lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let _guard = CwdGuard::new(dir.path());

        let source = default_config_source();

        assert_eq!(
            source, "<built-in default>",
            "should return built-in default when praxis.yaml is absent"
        );
    }
}
