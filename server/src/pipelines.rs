// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Filter pipeline resolution for server listeners.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use praxis_core::config::{ChainRef, Config, FailureMode, FilterEntry, InsecureOptions, Listener};
use praxis_filter::{FilterPipeline, FilterRegistry};
use praxis_protocol::ListenerPipelines;
use praxis_tls::ClientCertMode;

// -----------------------------------------------------------------------------
// Pipeline Resolution
// -----------------------------------------------------------------------------

/// Build a [`FilterPipeline`] for each listener by resolving named chains.
///
/// # Errors
///
/// Returns an error when pipeline construction fails (unknown filter chain
/// referenced by listener, filter instantiation failure, branch chain
/// resolution error, body limit conflict, or pipeline ordering violation).
///
/// [`FilterPipeline`]: praxis_filter::FilterPipeline
pub fn resolve_pipelines(
    config: &Config,
    registry: &FilterRegistry,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
) -> Result<ListenerPipelines, Box<dyn std::error::Error + Send + Sync>> {
    let chains: HashMap<&str, &[_]> = config
        .filter_chains
        .iter()
        .map(|c| (c.name.as_str(), c.filters.as_slice()))
        .collect();

    let mut pipelines = HashMap::with_capacity(config.listeners.len());

    for listener in &config.listeners {
        let mut entries = Vec::new();
        for chain_name in &listener.filter_chains {
            let chain_filters = chains.get(chain_name.as_str()).ok_or_else(|| {
                let lname = &listener.name;
                format!("unknown chain '{chain_name}' for listener '{lname}'")
            })?;
            entries.extend_from_slice(chain_filters);
        }

        let mut pipeline = FilterPipeline::build_with_chains(&mut entries, registry, &chains)?;
        configure_pipeline(&mut pipeline, config, health_registry, kv_stores, subrequest_client)?;

        validate_provider_boundary(listener, &entries, &chains)?;
        validate_pipeline(&pipeline, &entries, &listener.name, &config.insecure_options)?;

        pipelines.insert(listener.name.clone(), Arc::new(pipeline));
    }

    Ok(ListenerPipelines::new(pipelines))
}

/// Apply body limits, health registry, KV stores, and insecure options to a
/// pipeline.
fn configure_pipeline(
    pipeline: &mut FilterPipeline,
    config: &Config,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    pipeline.apply_body_limits(
        config.body_limits.max_request_bytes,
        config.body_limits.max_response_bytes,
        config.insecure_options.allow_unbounded_body,
    )?;
    if !health_registry.is_empty() {
        pipeline.set_health_registry(Arc::clone(health_registry));
    }
    if !kv_stores.is_empty() {
        pipeline.set_kv_stores(kv_stores.clone());
    }
    pipeline.add_pipeline_extension(Box::new(praxis_ai_apis::store::ResponseStoreRegistry::new()));
    pipeline.set_subrequest_client(subrequest_client.clone());
    pipeline.apply_insecure_options(&config.insecure_options);
    Ok(())
}

// -----------------------------------------------------------------------------
// Pipeline Validation
// -----------------------------------------------------------------------------

/// Enforce the non-bypassable trust contract for AI-owned provider-hop context.
///
/// `x-ai-routing-*` fields are intentionally outside Praxis's reserved namespace
/// so they can cross the upstream boundary. Every provider consumer therefore
/// requires mandatory client certificates and an unconditional, fail-closed
/// `peer_identity_trust` as the first filter in the provider chain.
fn validate_provider_boundary(
    listener: &Listener,
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if provider_consumer_exists_in_branch(entries, chains, &mut HashSet::new()) {
        return Err(format!(
            "listener '{}': provider_route must be top-level, not branch-conditional",
            listener.name
        )
        .into());
    }

    let provider_indices = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| (entry.filter_type == "provider_route").then_some(index))
        .collect::<Vec<_>>();
    if provider_indices.is_empty() {
        return Ok(());
    }

    validate_provider_listener_tls(listener)?;
    for provider_index in provider_indices {
        validate_provider_entry(listener, entries, provider_index)?;
    }

    Ok(())
}

/// Require mutual TLS on a listener that consumes provider context.
fn validate_provider_listener_tls(listener: &Listener) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if listener
        .tls
        .as_ref()
        .is_none_or(|tls| tls.client_cert_mode != ClientCertMode::Require)
    {
        return Err(format!(
            "listener '{}': provider_route requires tls.client_cert_mode: require",
            listener.name
        )
        .into());
    }
    Ok(())
}

/// Validate one provider consumer and its non-bypassable peer trust filter.
fn validate_provider_entry(
    listener: &Listener,
    entries: &[FilterEntry],
    provider_index: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let provider = entries
        .get(provider_index)
        .ok_or_else(|| format!("listener '{}': invalid provider_route index", listener.name))?;
    validate_unconditional_closed(&listener.name, provider)?;
    let peer = required_peer_identity_trust(&listener.name, entries, provider_index)?;
    validate_unconditional_closed(&listener.name, peer)?;
    Ok(())
}

/// Find the mandatory first-position peer trust filter for a provider consumer.
fn required_peer_identity_trust<'a>(
    listener_name: &str,
    entries: &'a [FilterEntry],
    provider_index: usize,
) -> Result<&'a FilterEntry, Box<dyn std::error::Error + Send + Sync>> {
    let preceding = entries
        .get(..provider_index)
        .ok_or_else(|| format!("listener '{listener_name}': invalid provider_route prefix"))?;
    let (peer_index, peer) = preceding
        .iter()
        .enumerate()
        .rev()
        .find(|(_, entry)| entry.filter_type == "peer_identity_trust")
        .ok_or_else(|| {
            format!("listener '{listener_name}': provider_route requires a preceding peer_identity_trust")
        })?;
    if peer_index != 0 {
        return Err(format!(
            "listener '{listener_name}': peer_identity_trust must be the first filter in a provider chain"
        )
        .into());
    }
    Ok(peer)
}

/// Detect provider consumers nested in any reachable inline or named branch.
fn provider_consumer_exists_in_branch(
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
    visited: &mut HashSet<String>,
) -> bool {
    entries.iter().any(|entry| {
        entry.branch_chains.as_ref().is_some_and(|branches| {
            branches.iter().any(|branch| {
                branch.chains.iter().any(|chain| {
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
                    nested.iter().any(|filter| filter.filter_type == "provider_route")
                        || provider_consumer_exists_in_branch(nested, chains, visited)
                })
            })
        })
    })
}

/// Require a security-boundary filter to execute unconditionally and fail closed.
fn validate_unconditional_closed(
    listener_name: &str,
    entry: &FilterEntry,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if entry.failure_mode != FailureMode::Closed || !entry.conditions.is_empty() {
        return Err(format!(
            "listener '{listener_name}': {} must be unconditional and fail-closed for the provider boundary",
            entry.filter_type
        )
        .into());
    }
    Ok(())
}

/// Run pipeline ordering validation; either fail or warn depending
/// on insecure option flags.
#[expect(clippy::cognitive_complexity, reason = "pre-existing complexity above threshold")]
fn validate_pipeline(
    pipeline: &FilterPipeline,
    entries: &[FilterEntry],
    listener_name: &str,
    insecure_options: &InsecureOptions,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let errors = pipeline.ordering_errors(
        entries,
        insecure_options.allow_open_security_filters,
        &insecure_options.effective_pipeline_checks(),
    );

    if insecure_options.skip_pipeline_validation {
        for msg in &errors {
            tracing::warn!(listener = %listener_name, "{msg}");
        }
    } else if !errors.is_empty() {
        for msg in &errors {
            tracing::error!(listener = %listener_name, "{msg}");
        }
        return Err(format!(
            "pipeline validation failed for listener '{listener_name}': {}",
            errors.join("; ")
        )
        .into());
    }

    for warning in pipeline.ordering_warnings() {
        tracing::warn!(listener = %listener_name, "{warning}");
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use praxis_core::{config::Config, health::HealthRegistry};
    use praxis_filter::FilterRegistry;

    use super::*;

    #[test]
    fn resolve_pipelines_builds_for_each_listener() {
        let config = valid_config();
        let registry = FilterRegistry::with_builtins();
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        )
        .unwrap();
        assert!(
            pipelines.get("web").is_some(),
            "pipeline should exist for 'web' listener"
        );
    }

    #[test]
    fn config_rejects_unknown_filter_chain() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [nonexistent]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        );
        assert!(
            config.is_err(),
            "config referencing nonexistent chain should fail to parse"
        );
    }

    #[test]
    fn resolve_pipelines_empty_chains_produces_empty_pipeline() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters: []
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        )
        .unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        assert!(
            pipeline.is_empty(),
            "pipeline with empty filter chain should have no filters"
        );
    }

    #[test]
    fn resolve_pipelines_multiple_chains_concatenated() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [observability, routing]
filter_chains:
  - name: observability
    filters:
      - filter: request_id
  - name: routing
    filters:
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        )
        .unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        assert_eq!(pipeline.len(), 3, "two chains should produce 3 filters total");
    }

    #[test]
    fn resolve_pipelines_applies_body_limits() {
        let config = Config::from_yaml(
            r#"
body_limits:
  max_request_bytes: 1024
  max_response_bytes: 2048
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
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        )
        .unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        let caps = pipeline.body_capabilities();
        assert!(caps.needs_request_body, "body limits should enable request body access");
        assert!(
            caps.needs_response_body,
            "body limits should enable response body access"
        );
        assert_eq!(
            caps.request_body_mode,
            praxis_filter::BodyMode::SizeLimit { max_bytes: 1024 },
            "default Stream should become SizeLimit for enforcement"
        );
        assert_eq!(
            caps.response_body_mode,
            praxis_filter::BodyMode::SizeLimit { max_bytes: 2048 },
            "default Stream should become SizeLimit for enforcement"
        );
    }

    #[test]
    fn resolve_pipelines_allows_router_without_lb() {
        let config = Config::from_yaml(
            r#"
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
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let result = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        );
        assert!(result.is_ok(), "router without LB should be a warning, not an error");
    }

    #[test]
    fn resolve_pipelines_skip_validation_downgrades_to_warnings() {
        let config = Config::from_yaml(
            r#"
insecure_options:
  skip_pipeline_validation: true
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
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let result = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        );
        assert!(result.is_ok(), "skip_pipeline_validation should allow startup");
    }

    #[test]
    fn provider_route_rejects_plaintext_listener() {
        let (listener, entries) = provider_boundary_parts("", peer_then_provider_filters());

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("plaintext provider listener must fail");

        assert!(err.to_string().contains("client_cert_mode: require"), "{err}");
    }

    #[test]
    fn provider_route_rejects_optional_client_certificate() {
        let (listener, entries) = provider_boundary_parts(&provider_tls("request"), peer_then_provider_filters());

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("optional client certificate must fail");

        assert!(err.to_string().contains("client_cert_mode: require"), "{err}");
    }

    #[test]
    fn provider_route_rejects_missing_peer_trust() {
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), provider_filter());

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("provider route without peer trust must fail");

        assert!(err.to_string().contains("preceding peer_identity_trust"), "{err}");
    }

    #[test]
    fn provider_route_rejects_peer_trust_after_consumer() {
        let filters = format!("{}{}", provider_filter(), peer_filter(""));
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("peer trust after provider route must fail");

        assert!(err.to_string().contains("preceding peer_identity_trust"), "{err}");
    }

    #[test]
    fn provider_route_rejects_open_peer_trust() {
        let filters = format!("{}{}", peer_filter("        failure_mode: open\n"), provider_filter());
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("fail-open peer trust must fail");

        assert!(err.to_string().contains("unconditional and fail-closed"), "{err}");
    }

    #[test]
    fn provider_route_rejects_filter_before_peer_trust() {
        let filters = format!("      - filter: request_id\n{}{}", peer_filter(""), provider_filter());
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("an earlier filter could branch around peer trust");

        assert!(err.to_string().contains("must be the first filter"), "{err}");
    }

    #[test]
    fn provider_route_rejects_open_provider_consumer() {
        let provider = provider_filter().replacen(
            "        provider_id:",
            "        failure_mode: open\n        provider_id:",
            1,
        );
        let filters = format!("{}{}", peer_filter(""), provider);
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("fail-open provider consumer must fail");

        assert!(err.to_string().contains("unconditional and fail-closed"), "{err}");
    }

    #[test]
    fn provider_route_rejects_conditional_peer_trust() {
        let filters = format!(
            "{}{}",
            peer_filter("        conditions:\n          - when:\n              path_prefix: /v1\n"),
            provider_filter()
        );
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("conditional peer trust must fail");

        assert!(err.to_string().contains("unconditional and fail-closed"), "{err}");
    }

    #[test]
    fn provider_route_rejects_conditional_provider_consumer() {
        let provider = provider_filter().replacen(
            "        provider_id:",
            "        conditions:\n          - when:\n              path_prefix: /v1\n        provider_id:",
            1,
        );
        let filters = format!("{}{}", peer_filter(""), provider);
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), &filters);

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("conditional provider consumer must fail");

        assert!(err.to_string().contains("unconditional and fail-closed"), "{err}");
    }

    #[test]
    fn provider_route_rejects_branch_conditional_consumer() {
        let (listener, entries) = provider_boundary_parts(
            &provider_tls("require"),
            "      - filter: peer_identity_trust
        trusted_peers:
          - organization: ai-grid
        branch_chains:
          - name: provider-branch
            chains:
              - name: inline-provider
                filters:
                  - filter: provider_route
                    provider_id: test-provider
                    routes:
                      - candidate_id: candidate-a
                        cluster: backend
                        model: model-a
                        paths: [/v1/chat/completions]
",
        );

        let err = validate_provider_boundary(&listener, &entries, &HashMap::new())
            .expect_err("branch-conditional provider consumer must fail");

        assert!(err.to_string().contains("must be top-level"), "{err}");
    }

    #[test]
    fn provider_route_accepts_required_mtls_and_preceding_peer_trust() {
        let (listener, entries) = provider_boundary_parts(&provider_tls("require"), peer_then_provider_filters());

        validate_provider_boundary(&listener, &entries, &HashMap::new()).expect("valid provider boundary");
    }

    #[test]
    fn resolve_pipelines_rejects_misaligned_clusters() {
        let config = Config::from_yaml(
            r#"
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
            cluster: missing
      - filter: load_balancer
        clusters:
          - name: other
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let result = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        );
        assert!(result.is_err(), "misaligned clusters should fail validation");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("missing") && err.contains("not defined"),
            "error should name the missing cluster: {err}"
        );
    }

    #[test]
    fn resolve_pipelines_rejects_open_security_filter() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ip_acl
        allow: ["10.0.0.0/8"]
        failure_mode: open
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let result = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        );
        assert!(result.is_err(), "open security filter should fail validation");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("failure_mode: open") && err.contains("ip_acl"),
            "error should mention open ip_acl: {err}"
        );
    }

    #[test]
    fn resolve_pipelines_allows_open_security_with_insecure_flag() {
        let config = Config::from_yaml(
            r#"
insecure_options:
  allow_open_security_filters: true
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: ip_acl
        allow: ["10.0.0.0/8"]
        failure_mode: open
      - filter: router
        routes:
          - path_prefix: "/"
            cluster: backend
      - filter: load_balancer
        clusters:
          - name: backend
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap();
        let registry = FilterRegistry::with_builtins();
        let result = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &test_client(),
        );
        assert!(result.is_ok(), "allow_open_security_filters should permit open ip_acl");
    }

    #[test]
    fn resolve_pipelines_threads_kv_stores() {
        let config = valid_config();
        let registry = FilterRegistry::with_builtins();
        let kv = make_kv_registry();
        let pipelines = resolve_pipelines(&config, &registry, &empty_health_registry(), &kv, &test_client()).unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        assert!(pipeline.kv_stores().is_some(), "pipeline should have kv_stores set");
    }

    #[test]
    fn resolve_pipelines_empty_kv_not_set() {
        let config = valid_config();
        let registry = FilterRegistry::with_builtins();
        let kv = empty_kv_stores();
        let pipelines = resolve_pipelines(&config, &registry, &empty_health_registry(), &kv, &test_client()).unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        assert!(
            pipeline.kv_stores().is_none(),
            "empty kv_stores should not be set on pipeline"
        );
    }

    #[test]
    fn resolve_pipelines_threads_subrequest_client() {
        let config = valid_config();
        let registry = FilterRegistry::with_builtins();
        let client = test_client();
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &client,
        )
        .unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        assert!(
            pipeline.subrequest_client().is_some(),
            "pipeline should have subrequest_client set after resolve_pipelines",
        );
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Empty health registry for tests without health checks.
    fn empty_health_registry() -> HealthRegistry {
        Arc::new(HashMap::new())
    }

    /// Empty KV store registry for tests without KV stores.
    fn empty_kv_stores() -> praxis_core::kv::KvStoreRegistry {
        praxis_core::kv::KvStoreRegistry::new()
    }

    /// Minimal sub-request client for tests.
    fn test_client() -> praxis_core::subrequest::SubRequestClient {
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None))
    }

    /// KV store registry with one test store.
    fn make_kv_registry() -> praxis_core::kv::KvStoreRegistry {
        let registry = praxis_core::kv::KvStoreRegistry::new();
        registry.get_or_create("test");
        registry
    }

    /// Minimal valid config with one listener for pipeline tests.
    fn valid_config() -> Config {
        Config::from_yaml(
            r#"
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
            endpoints: ["10.0.0.1:80"]
"#,
        )
        .unwrap()
    }

    /// Parse one listener and return its resolved top-level filter entries.
    fn provider_boundary_parts(tls: &str, filters: &str) -> (Listener, Vec<FilterEntry>) {
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: provider
    address: "127.0.0.1:8443"
    filter_chains: [provider]{tls}
filter_chains:
  - name: provider
    filters:
{filters}
"#
        ))
        .expect("provider boundary config should parse");
        (config.listeners[0].clone(), config.filter_chains[0].filters.clone())
    }

    /// Listener TLS block with the requested downstream client-cert mode.
    fn provider_tls(mode: &str) -> String {
        format!(
            r#"
    tls:
      certificates:
        - cert_path: "/tmp/provider.crt"
          key_path: "/tmp/provider.key"
      client_ca:
        ca_path: "/tmp/grid-ca.crt"
      client_cert_mode: {mode}"#
        )
    }

    /// Fail-closed peer filter followed by the provider consumer.
    fn peer_then_provider_filters() -> &'static str {
        concat!(
            "      - filter: peer_identity_trust\n",
            "        trusted_peers:\n",
            "          - organization: ai-grid\n",
            "      - filter: provider_route\n",
            "        provider_id: test-provider\n",
            "        routes:\n",
            "          - candidate_id: candidate-a\n",
            "            cluster: backend\n",
            "            model: model-a\n",
            "            paths: [/v1/chat/completions]\n",
        )
    }

    /// Peer filter YAML with optional entry-level fields.
    fn peer_filter(entry_fields: &str) -> String {
        format!(
            "      - filter: peer_identity_trust\n\
             {entry_fields}\
             \x20       trusted_peers:\n\
             \x20         - organization: ai-grid\n"
        )
    }

    /// Provider consumer filter YAML.
    fn provider_filter() -> &'static str {
        concat!(
            "      - filter: provider_route\n",
            "        provider_id: test-provider\n",
            "        routes:\n",
            "          - candidate_id: candidate-a\n",
            "            cluster: backend\n",
            "            model: model-a\n",
            "            paths: [/v1/chat/completions]\n",
        )
    }
}
