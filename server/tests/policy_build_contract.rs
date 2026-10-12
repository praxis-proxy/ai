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
    use praxis_core::config::Config;

    const CONFIG: &str = r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters: []
"#;

    /// The proxy's shared client, built on the crypto provider the binary
    /// installs at startup (the connector needs it before it can exist).
    fn client(config: &Config) -> praxis_core::subrequest::SubRequestClient {
        praxis_ai::install_crypto_provider();
        praxis_ai::create_subrequest_client(config)
    }

    /// Praxis 0.7.4 moved the policy sub-request connector off a process-global
    /// onto the registry (`FilterRegistry::set_policy_connector`). This asserts
    /// `build_full_registry` hands the registry's `policy` filters the proxy's
    /// shared connector, so their outbound calls (JWKS fetches, token exchanges)
    /// reuse one keepalive pool, admission limit, and circuit breaker instead of
    /// each opening a pool of its own.
    #[test]
    fn build_full_registry_hands_policy_filters_the_shared_connector() {
        let config = Config::from_yaml(CONFIG).expect("the test config must parse");
        let client = client(&config);
        let expected_connector = client.connector().clone();

        let registry = praxis_ai::build_full_registry(&client);

        let registered = registry
            .policy_connector()
            .expect("build_full_registry must hand policy filters the shared connector");
        assert!(
            std::ptr::eq(registered.connector(), expected_connector.connector()),
            "policy filters must receive the proxy's shared connector"
        );
    }
}
