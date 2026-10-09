// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Build-time guarantees for policy subrequests.

// The policy registration contract exists only with the policy engine; the
// FIPS build leaves it out and must still compile every other test.
#![cfg(feature = "policy-engine")]

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, reason = "integration test")]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use praxis_core::config::Config;
    use praxis_filter::FilterRegistry;

    const CONFIG: &str = r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints:
              - "127.0.0.1:3000"
insecure_options:
  allow_private_endpoints: true
"#;

    /// The proxy's shared client, built on the crypto provider the binary
    /// installs at startup (the connector needs it before it can exist).
    fn client(config: &Config) -> praxis_core::subrequest::SubRequestClient {
        praxis_ai::install_crypto_provider();
        praxis_ai::create_subrequest_client(config)
    }

    #[test]
    fn registry_clone_preserves_connector_and_pipelines_build_successfully() {
        let config = Config::from_yaml(CONFIG).expect("the test config must parse");
        let client = client(&config);
        let expected_connector = client.connector().clone();
        let registry = FilterRegistry::with_builtins();

        let mut runtime = registry.clone();
        runtime.set_policy_connector(client.connector());

        let held = runtime
            .policy_connector()
            .expect("the runtime registry must carry a policy connector after set_policy_connector");
        assert!(
            std::ptr::eq(held.connector(), expected_connector.connector()),
            "policy filters must receive the proxy's shared connector"
        );
        assert!(
            registry.policy_connector().is_none(),
            "the caller's registry must not pick up the runtime's connector"
        );

        let result = praxis_ai::resolve_pipelines(
            &config,
            &registry,
            &Arc::new(HashMap::new()),
            &praxis_core::kv::KvStoreRegistry::new(),
            &client,
        );
        assert!(
            result.is_ok(),
            "resolve_pipelines must succeed with the connector set: {}",
            result.err().map_or_else(String::new, |e| e.to_string())
        );
    }
}
