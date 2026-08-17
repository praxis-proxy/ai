// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Server bootstrap for Praxis AI.

pub(crate) mod pipelines;
pub(crate) mod reload;
mod server;
pub(crate) mod watcher;
pub use pipelines::resolve_pipelines;
pub use praxis_core::logging::init_tracing;
pub use server::{check_root_privilege, fatal, resolve_config_path, run_server, run_server_with_registry};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Built-in fallback configuration branded for praxis-ai.
const DEFAULT_CONFIG: &str = include_str!("default.yaml");

// -----------------------------------------------------------------------------
// Configuration Loading
// -----------------------------------------------------------------------------

/// Load configuration from an explicit path, falling back to
/// `praxis.yaml` in the working directory, then the praxis-ai
/// built-in default.
///
/// # Errors
///
/// Returns [`ProxyError::Config`] if the resolved config source
/// cannot be loaded or is invalid.
///
/// [`ProxyError::Config`]: praxis_core::errors::ProxyError::Config
pub fn load_config(
    explicit_path: Option<&str>,
) -> Result<praxis_core::config::Config, praxis_core::errors::ProxyError> {
    praxis_core::config::Config::load(explicit_path, DEFAULT_CONFIG)
}

// -----------------------------------------------------------------------------
// External Filter Discovery
// -----------------------------------------------------------------------------

// Provides: fn register_external_filters(&mut FilterRegistry)
include!(concat!(env!("OUT_DIR"), "/external_filters.rs"));

/// Build a [`FilterRegistry`] with core builtins, AI filters, and
/// auto-discovered external filters.
///
/// The shared [`SubRequestClient`] is captured by filters that make
/// HTTP callouts so they share the server-level connection pool
/// instead of creating isolated per-filter connectors.
///
/// [`FilterRegistry`]: praxis_filter::FilterRegistry
/// [`SubRequestClient`]: praxis_core::subrequest::SubRequestClient
#[must_use]
pub fn build_full_registry(
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
) -> praxis_filter::FilterRegistry {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_ai_filters::register_ai_filters(&mut registry, Some(subrequest_client));
    register_external_filters(&mut registry);
    registry
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use praxis_core::config::Config;

    use super::*;

    fn test_subrequest_client() -> praxis_core::subrequest::SubRequestClient {
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None))
    }

    #[test]
    fn default_config_parses_successfully() {
        let config = Config::from_yaml(DEFAULT_CONFIG).expect("DEFAULT_CONFIG should parse");
        assert!(
            !config.listeners.is_empty(),
            "default config should define at least one listener"
        );
    }

    #[test]
    fn default_config_brands_praxis_ai() {
        let config = Config::from_yaml(DEFAULT_CONFIG).expect("DEFAULT_CONFIG should parse");
        let body = config.filter_chains[0].filters[0].config["body"]
            .as_str()
            .expect("AI default static response should define a string body");
        assert_eq!(
            body, r#"{"status": "ok", "server": "praxis-ai"}"#,
            "default config should use the AI-branded response body"
        );
    }

    #[test]
    fn load_config_none_succeeds() {
        let config = load_config(None).expect("load_config(None) should succeed");
        assert!(
            !config.listeners.is_empty(),
            "loaded config should define at least one listener"
        );
    }

    #[test]
    fn routing_filters_are_registered_exactly_once() {
        let registry = build_full_registry(&test_subrequest_client());
        let names = registry.available_filters();
        for expected in ["intelligent_route", "provider_route", "credential_inject"] {
            assert_eq!(
                names.iter().filter(|name| **name == expected).count(),
                1,
                "{expected} must be registered exactly once"
            );
        }
        assert!(!registry.is_security_filter("intelligent_route"));
        assert!(registry.is_security_filter("provider_route"));
        assert!(registry.is_security_filter("credential_inject"));
    }

    #[test]
    fn provider_pipeline_filters_construct_from_registry() {
        let registry = build_full_registry(&test_subrequest_client());
        let peer_config = serde_yaml::from_str(
            "trusted_peers:\n\
             \x20 - cert_digest: 0000000000000000000000000000000000000000000000000000000000000000\n\
             \x20   organization: ai-grid\n",
        )
        .expect("peer config");
        let provider_route_config = serde_yaml::from_str(
            "provider_id: site-a\n\
             routes:\n\
             \x20 - candidate_id: candidate-a\n\
             \x20   cluster: backend\n\
             \x20   model: model-a\n\
             \x20   paths: [/v1/chat/completions]\n",
        )
        .expect("provider route config");
        let credential_config = serde_yaml::from_str(
            "credentials:\n\
             \x20 - name: provider-token\n\
             \x20   namespace: grid-system\n\
             \x20   key: token\n\
             \x20   value: test-only-token\n",
        )
        .expect("credential config");

        for (name, config) in [
            ("peer_identity_trust", &peer_config),
            ("provider_route", &provider_route_config),
            ("credential_inject", &credential_config),
        ] {
            registry
                .create(name, config)
                .unwrap_or_else(|error| panic!("{name} must construct from the full registry: {error}"));
        }
    }
}
