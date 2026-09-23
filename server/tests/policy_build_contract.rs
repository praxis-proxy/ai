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
    use praxis_filter::{FilterFactory, FilterRegistry, registered_policy_subrequest_connector};

    const CONFIG: &str = r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: policy_registration_probe
"#;

    #[test]
    fn resolving_pipelines_registers_policy_connector_before_filter_construction() {
        let config = Config::from_yaml(CONFIG).expect("the test config must parse");
        let client = praxis_ai::create_subrequest_client(&config);
        let expected_connector = client.connector().clone();
        let mut registry = FilterRegistry::with_builtins();
        registry
            .register(
                "policy_registration_probe",
                FilterFactory::Http(Arc::new(move |_| {
                    let registered = registered_policy_subrequest_connector()
                        .expect("pipeline construction must have a policy connector registered");
                    assert!(
                        std::ptr::eq(registered.connector(), expected_connector.connector()),
                        "policy filters must receive the proxy's shared connector"
                    );
                    Err("registration probe completed".into())
                })),
            )
            .expect("the probe filter name must be unique");

        let result = praxis_ai::resolve_pipelines(
            &config,
            &registry,
            &Arc::new(HashMap::new()),
            &praxis_core::kv::KvStoreRegistry::new(),
            &client,
        );

        let error = result.err().expect("the probe factory must stop pipeline construction");
        assert!(
            error.to_string().contains("registration probe completed"),
            "the probe factory must be reached: {error}"
        );
    }
}
