// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Filter pipeline resolution for server listeners.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[cfg(feature = "openai-responses")]
use praxis_ai_apis::openai::AgenticBudgetPolicy;
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
    #[cfg(feature = "store")]
    {
        resolve_pipelines_with_stores(
            config,
            registry,
            health_registry,
            kv_stores,
            subrequest_client,
            &HashMap::new(),
        )
    }
    #[cfg(not(feature = "store"))]
    {
        build_listener_pipelines(
            config,
            registry,
            health_registry,
            kv_stores,
            subrequest_client,
            |_| false,
            |_, _| {},
        )
    }
}

/// Validate pipelines with the same listener store registries and readiness
/// gates that server startup constructs.
///
/// This is the CLI validation path for builds with a concrete store backend.
/// It validates store references without opening pools, then builds the exact
/// effective filter ordering used while serving.
///
/// # Errors
///
/// Returns an error from store wiring validation or pipeline construction.
#[cfg(any(feature = "store-postgres", feature = "store-sqlite"))]
pub fn validate_pipelines_with_store_wiring(
    config: &Config,
    registry: &FilterRegistry,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (store_registries, _service, _reload, _readiness) = crate::store_provision::build_store_wiring(config)?;
    resolve_pipelines_with_stores(
        config,
        registry,
        health_registry,
        kv_stores,
        subrequest_client,
        &store_registries,
    )?;
    Ok(())
}

/// Like [`resolve_pipelines`], but attaches a caller-provided
/// response-store registry per listener. The async serve and reload paths
/// provision backends first and pass the populated registries here; a listener
/// without a provided registry gets an empty one, which is the behaviour the
/// validation, CLI, and test paths rely on.
///
/// # Errors
///
/// Same as [`resolve_pipelines`].
#[cfg(feature = "store")]
#[expect(
    clippy::too_many_arguments,
    reason = "threads config, registries, shared services, and per-listener stores"
)]
pub(crate) fn resolve_pipelines_with_stores(
    config: &Config,
    registry: &FilterRegistry,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
    store_registries: &crate::StoreRegistries,
) -> Result<ListenerPipelines, Box<dyn std::error::Error + Send + Sync>> {
    build_listener_pipelines(
        config,
        registry,
        health_registry,
        kv_stores,
        subrequest_client,
        |listener| store_registries.contains_key(&listener.name),
        |listener, pipeline| {
            pipeline.add_pipeline_extension(Box::new(
                store_registries.get(&listener.name).cloned().unwrap_or_default(),
            ));
        },
    )
}

/// Build a pipeline per listener, applying `attach` to each after the shared
/// configuration and before validation. `attach` is where the store path adds
/// its per-listener registry; the store-free build passes a no-op.
///
/// # Errors
///
/// Returns an error when pipeline construction fails.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "threads config, registries, shared services, the policy-connector setup, and a per-listener hook"
)]
fn build_listener_pipelines(
    config: &Config,
    registry: &FilterRegistry,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
    gate_store_traffic: impl Fn(&Listener) -> bool,
    attach: impl Fn(&Listener, &mut FilterPipeline),
) -> Result<ListenerPipelines, Box<dyn std::error::Error + Send + Sync>> {
    praxis_filter::set_policy_subrequest_connector(subrequest_client.connector());
    let configured_chains: HashMap<&str, &[_]> = config
        .filter_chains
        .iter()
        .map(|c| (c.name.as_str(), c.filters.as_slice()))
        .collect();

    let mut pipelines = HashMap::with_capacity(config.listeners.len());

    for listener in &config.listeners {
        let mut entries = Vec::new();
        for chain_name in &listener.filter_chains {
            let chain_filters = configured_chains.get(chain_name.as_str()).ok_or_else(|| {
                let lname = &listener.name;
                format!("unknown chain '{chain_name}' for listener '{lname}'")
            })?;
            entries.extend_from_slice(chain_filters);
        }

        #[cfg(feature = "store")]
        if gate_store_traffic(listener) {
            // Provider chains require peer_identity_trust to remain the first
            // filter. The readiness gate is otherwise the first operator so it
            // rejects cold-start traffic before any store consumer runs.
            let gate_index = usize::from(
                entries
                    .first()
                    .is_some_and(|entry| entry.filter_type == "peer_identity_trust"),
            );
            entries.insert(gate_index, store_readiness_gate_entry());
        }
        #[cfg(not(feature = "store"))]
        let _ = &gate_store_traffic;

        #[cfg(feature = "openai-responses")]
        let mut owned_chains: HashMap<String, Vec<FilterEntry>> = config
            .filter_chains
            .iter()
            .map(|chain| (chain.name.clone(), chain.filters.clone()))
            .collect();
        #[cfg(feature = "openai-responses")]
        let budget_policy = prepare_agentic_budget_entries(&mut entries, &mut owned_chains)?;
        #[cfg(feature = "openai-responses")]
        let chains: HashMap<&str, &[_]> = owned_chains
            .iter()
            .map(|(name, filters)| (name.as_str(), filters.as_slice()))
            .collect();
        #[cfg(not(feature = "openai-responses"))]
        let chains = configured_chains.clone();
        #[cfg(feature = "openai-responses")]
        let request_body_limit = agentic_request_body_cap(config.body_limits.max_request_bytes, budget_policy);
        #[cfg(not(feature = "openai-responses"))]
        let request_body_limit = config.body_limits.max_request_bytes;
        let mut pipeline =
            FilterPipeline::build_with_chains(&mut entries, registry, &chains, &config.insecure_options)?;
        configure_pipeline(
            &mut pipeline,
            config,
            health_registry,
            kv_stores,
            subrequest_client,
            request_body_limit,
        )?;
        #[cfg(feature = "openai-responses")]
        if let Some(policy) = budget_policy {
            pipeline.add_pipeline_extension(Box::new(policy));
        }
        attach(listener, &mut pipeline);

        validate_provider_boundary(listener, &entries, &chains)?;
        validate_pipeline(&pipeline, &entries, &listener.name, &config.insecure_options)?;

        pipelines.insert(listener.name.clone(), Arc::new(pipeline));
    }

    Ok(ListenerPipelines::new(pipelines))
}

/// Prepare the loop budget and IRR response caps before building a listener
/// pipeline. The integration harness uses this same path as the server.
///
/// # Errors
///
/// Returns an error for invalid loop configuration or an unbounded loop route.
#[cfg(feature = "openai-responses")]
pub fn prepare_agentic_budget_entries(
    entries: &mut [FilterEntry],
    chains: &mut HashMap<String, Vec<FilterEntry>>,
) -> Result<Option<AgenticBudgetPolicy>, Box<dyn std::error::Error + Send + Sync>> {
    // Scan against a stable snapshot while rewriting both the listener entries
    // and its named branch chains. The response cap is local to this listener.
    let original = chains.clone();
    let original_refs: HashMap<&str, &[_]> = original
        .iter()
        .map(|(name, filters)| (name.as_str(), filters.as_slice()))
        .collect();
    let policy = agentic_budget_policy(entries, &original_refs)?;
    if let Some(policy) = policy {
        cap_agentic_irr_responses(entries, &original_refs, policy)?;
        for filters in chains.values_mut() {
            cap_agentic_irr_responses(filters, &original_refs, policy)?;
        }
        let rewritten_refs: HashMap<&str, &[_]> = chains
            .iter()
            .map(|(name, filters)| (name.as_str(), filters.as_slice()))
            .collect();
        validate_agentic_irr_caps(entries, &rewritten_refs, policy)?;
    }
    Ok(policy)
}

/// Clamp the whole listener's raw request-body buffer before any filter sees
/// a create body. This also constrains other routes on that listener until
/// core exposes a route-specific transport limit.
#[cfg(feature = "openai-responses")]
#[must_use]
pub fn agentic_request_body_cap(configured: Option<usize>, policy: Option<AgenticBudgetPolicy>) -> Option<usize> {
    policy.map_or(configured, |policy| {
        let maximum = policy.max_request_body_bytes();
        Some(configured.map_or(maximum, |configured| configured.min(maximum)))
    })
}

/// Discover nested loop limits at pipeline construction, before the first
/// request filter can parse or copy a body. Branches are combined by the
/// smallest reachable limit because their runtime selection is not yet known.
#[cfg(feature = "openai-responses")]
#[expect(
    clippy::too_many_lines,
    reason = "recursive traversal of nested IRR steps and branch chains"
)]
fn agentic_budget_policy(
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
) -> Result<Option<AgenticBudgetPolicy>, Box<dyn std::error::Error + Send + Sync>> {
    #[expect(
        clippy::too_many_lines,
        reason = "recursive traversal of nested IRR steps and branch chains"
    )]
    fn visit(
        entries: &[FilterEntry],
        chains: &HashMap<&str, &[FilterEntry]>,
        visited: &mut HashSet<String>,
        policy: &mut Option<AgenticBudgetPolicy>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for entry in entries {
            if entry.filter_type == "openai_agentic_loop" {
                let next = AgenticBudgetPolicy::from_config(&entry.config)?;
                *policy = Some(policy.map_or(next, |current| current.min(next)));
            }
            if entry.filter_type == "iterative_request_router"
                && let Some(steps) = entry.config.get("steps").and_then(serde_yaml::Value::as_sequence)
            {
                for step in steps {
                    if let Some(filters) = step.get("filters").and_then(serde_yaml::Value::as_sequence) {
                        let nested: Vec<FilterEntry> = filters
                            .iter()
                            .cloned()
                            .map(serde_yaml::from_value)
                            .collect::<Result<_, _>>()?;
                        visit(&nested, chains, visited, policy)?;
                    }
                }
            }
            if let Some(branches) = &entry.branch_chains {
                for chain in branches.iter().flat_map(|branch| branch.chains.iter()) {
                    match chain {
                        ChainRef::Inline { filters, .. } => visit(filters, chains, visited, policy)?,
                        ChainRef::Named(name) if visited.insert(name.clone()) => {
                            if let Some(filters) = chains.get(name.as_str()) {
                                visit(filters, chains, visited, policy)?;
                            }
                        },
                        ChainRef::Named(_) => {},
                    }
                }
            }
        }
        Ok(())
    }

    let mut policy = None;
    visit(entries, chains, &mut HashSet::new(), &mut policy)?;
    Ok(policy)
}

/// Lower loop-bearing IRR transport caps before core builds their subpipelines.
/// Core buffers that response before any AI body hook can reject it.
#[cfg(feature = "openai-responses")]
#[expect(
    clippy::too_many_lines,
    reason = "rewrites reachable inline and nested IRR configs before core builds them"
)]
fn cap_agentic_irr_responses(
    entries: &mut [FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
    policy: AgenticBudgetPolicy,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for entry in entries {
        if entry.filter_type == "iterative_request_router"
            && let Some(steps) = entry.config.get("steps").and_then(serde_yaml::Value::as_sequence)
        {
            let mut step_entries = Vec::new();
            for step in steps {
                if let Some(filters) = step.get("filters").and_then(serde_yaml::Value::as_sequence) {
                    let parsed: Vec<FilterEntry> = filters
                        .iter()
                        .cloned()
                        .map(serde_yaml::from_value)
                        .collect::<Result<_, _>>()?;
                    step_entries.extend(parsed);
                }
            }
            if agentic_budget_policy(&step_entries, chains)?.is_some() {
                let maximum = policy.max_irr_response_bytes();
                let configured = configured_irr_response_cap(&entry.config)?;
                if let Some(mapping) = entry.config.as_mapping_mut() {
                    mapping.insert(
                        serde_yaml::Value::String("max_response_bytes".to_owned()),
                        serde_yaml::to_value(configured.min(maximum))?,
                    );
                }
            }
            // A step may itself contain an IRR. Rewrite that nested config
            // before core builds the step's subpipeline.
            if let Some(steps) = entry
                .config
                .get_mut("steps")
                .and_then(serde_yaml::Value::as_sequence_mut)
            {
                for step in steps {
                    if let Some(filters) = step.get_mut("filters").and_then(serde_yaml::Value::as_sequence_mut) {
                        for filter in filters {
                            let mut nested: FilterEntry = serde_yaml::from_value(filter.clone())?;
                            cap_agentic_irr_responses(std::slice::from_mut(&mut nested), chains, policy)?;
                            *filter = serde_yaml::to_value(nested)?;
                        }
                    }
                }
            }
        }
        if let Some(branches) = &mut entry.branch_chains {
            for chain in branches.iter_mut().flat_map(|branch| branch.chains.iter_mut()) {
                if let ChainRef::Inline { filters, .. } = chain {
                    cap_agentic_irr_responses(filters, chains, policy)?;
                }
            }
        }
    }
    Ok(())
}

/// Read the effective core IRR response cap while preserving malformed values.
#[cfg(feature = "openai-responses")]
fn configured_irr_response_cap(config: &serde_yaml::Value) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // This is the default in the pinned core IRR. A present invalid value must
    // retain core's validation error instead of being silently overwritten.
    const DEFAULT_IRR_RESPONSE_BYTES: usize = 10_485_760;
    match config.get("max_response_bytes") {
        None => Ok(DEFAULT_IRR_RESPONSE_BYTES),
        Some(value) => {
            let value = value
                .as_u64()
                .ok_or("iterative_request_router.max_response_bytes must be a positive integer")?;
            Ok(usize::try_from(value)?)
        },
    }
}

/// Core 0.7.2 buffers an IRR subresponse before any AI response filter runs.
/// Require its existing transport cap to fit inside the budget's raw-response
/// reserve, even when an unrelated IRR on this listener has a larger cap.
#[cfg(feature = "openai-responses")]
#[expect(
    clippy::too_many_lines,
    reason = "validates the same nested IRR and branch graph as cap preparation"
)]
fn validate_agentic_irr_caps(
    entries: &[FilterEntry],
    chains: &HashMap<&str, &[FilterEntry]>,
    policy: AgenticBudgetPolicy,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[expect(
        clippy::too_many_lines,
        reason = "walks nested IRR steps and branch chains with cycle protection"
    )]
    fn visit(
        entries: &[FilterEntry],
        chains: &HashMap<&str, &[FilterEntry]>,
        visited: &mut HashSet<(String, bool)>,
        maximum: usize,
        inside_irr: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for entry in entries {
            if entry.filter_type == "openai_agentic_loop" && !inside_irr {
                return Err("openai_agentic_loop requires a bounded iterative_request_router step".into());
            }
            if entry.filter_type == "iterative_request_router"
                && let Some(steps) = entry.config.get("steps").and_then(serde_yaml::Value::as_sequence)
            {
                let mut step_entries = Vec::new();
                for step in steps {
                    if let Some(filters) = step.get("filters").and_then(serde_yaml::Value::as_sequence) {
                        let parsed: Vec<FilterEntry> = filters
                            .iter()
                            .cloned()
                            .map(serde_yaml::from_value)
                            .collect::<Result<_, _>>()?;
                        step_entries.extend(parsed);
                    }
                }
                if agentic_budget_policy(&step_entries, chains)?.is_some() {
                    let cap = configured_irr_response_cap(&entry.config)?;
                    if cap > maximum {
                        return Err(format!(
                            "iterative_request_router with openai_agentic_loop requires max_response_bytes <= {maximum} for the retained-payload budget"
                        )
                        .into());
                    }
                }
                visit(&step_entries, chains, visited, maximum, true)?;
            }
            if let Some(branches) = &entry.branch_chains {
                for chain in branches.iter().flat_map(|branch| branch.chains.iter()) {
                    match chain {
                        ChainRef::Inline { filters, .. } => visit(filters, chains, visited, maximum, inside_irr)?,
                        ChainRef::Named(name) if visited.insert((name.clone(), inside_irr)) => {
                            if let Some(filters) = chains.get(name.as_str()) {
                                visit(filters, chains, visited, maximum, inside_irr)?;
                            }
                        },
                        ChainRef::Named(_) => {},
                    }
                }
            }
        }
        Ok(())
    }

    visit(
        entries,
        chains,
        &mut HashSet::new(),
        policy.max_irr_response_bytes(),
        false,
    )
}

/// Build the server-owned gate placed before store consumers on a listener
/// whose store registry is provisioned asynchronously.
#[cfg(feature = "store")]
fn store_readiness_gate_entry() -> FilterEntry {
    FilterEntry {
        filter_type: praxis_ai_filters::STORE_READINESS_GATE_FILTER_NAME.to_owned(),
        branch_chains: None,
        conditions: Vec::new(),
        name: None,
        response_conditions: Vec::new(),
        failure_mode: FailureMode::Closed,
        config: serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
    }
}

/// Apply body limits, health registry, KV stores, and insecure options to a
/// pipeline. Store registries are attached by the caller's `attach` hook.
#[expect(
    clippy::too_many_arguments,
    reason = "threads distinct pipeline services and the effective transport cap"
)]
fn configure_pipeline(
    pipeline: &mut FilterPipeline,
    config: &Config,
    health_registry: &praxis_core::health::HealthRegistry,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
    request_body_limit: Option<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    pipeline.apply_body_limits(
        request_body_limit,
        config.body_limits.max_response_bytes,
        config.insecure_options.allow_unbounded_body,
    )?;
    if !health_registry.is_empty() {
        pipeline.set_health_registry(Arc::clone(health_registry));
    }
    if !kv_stores.is_empty() {
        pipeline.set_kv_stores(kv_stores.clone());
    }
    crate::install_pipeline_extensions(pipeline);
    pipeline.set_subrequest_client(subrequest_client.clone());
    // Propagate the private-upstream override into the pipeline and its nested
    // callout chains (e.g. `openai_file_resolve`'s and the MCP callouts'
    // outbound chains) so their runtime SSRF checks read the configured value on
    // every (re)build.
    pipeline.set_allow_private_upstreams(config.insecure_options.allow_private_upstreams);
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
#[expect(
    clippy::allow_attributes,
    reason = "the lint only fires with some tracing feature sets"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "pre-existing complexity above threshold; tracing macro expansion varies with enabled features"
)]
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
    #[cfg(feature = "openai-responses")]
    fn nested_agentic_budget_is_available_before_request_filters() {
        let entries: Vec<FilterEntry> = serde_yaml::from_str(
            "
- filter: openai_responses_format
- filter: openai_responses_validate
- filter: iterative_request_router
  initial_step: inference
  steps:
    - name: inference
      filters:
        - filter: openai_agentic_loop
          max_retained_bytes: 8192
        - filter: openai_agentic_loop
          max_retained_bytes: 4096
",
        )
        .unwrap();
        let policy = agentic_budget_policy(&entries, &HashMap::new()).unwrap().unwrap();
        assert_eq!(policy.max_retained_bytes(), 4096);
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn loop_transport_cap_is_lowered_before_core_builds_it() {
        let mut entries: Vec<FilterEntry> = serde_yaml::from_str(
            "\n- filter: iterative_request_router\n  initial_step: inference\n  max_response_bytes: 10485760\n  steps:\n    - name: inference\n      filters:\n        - filter: openai_agentic_loop\n          max_retained_bytes: 4096\n",
        )
        .unwrap();
        let policy = agentic_budget_policy(&entries, &HashMap::new()).unwrap().unwrap();
        cap_agentic_irr_responses(&mut entries, &HashMap::new(), policy).unwrap();
        validate_agentic_irr_caps(&entries, &HashMap::new(), policy).unwrap();
        assert_eq!(entries[0].config["max_response_bytes"].as_u64(), Some(512));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn named_branch_irr_receives_the_same_transport_cap() {
        let mut chains = HashMap::from([(
            "agentic".to_owned(),
            serde_yaml::from_str(
                "- filter: iterative_request_router\n  initial_step: inference\n  steps:\n    - name: inference\n      filters:\n        - filter: openai_agentic_loop\n",
            )
            .unwrap(),
        )]);
        let mut entries: Vec<FilterEntry> = serde_yaml::from_str(
            "- filter: request_id\n  branch_chains:\n    - name: go\n      rejoin: terminal\n      chains: [agentic]\n",
        )
        .unwrap();

        let policy = prepare_agentic_budget_entries(&mut entries, &mut chains)
            .unwrap()
            .unwrap();
        assert_eq!(policy.max_retained_bytes(), 67_108_864);
        assert_eq!(
            chains["agentic"][0].config["max_response_bytes"].as_u64(),
            Some(8_388_608)
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn nested_irr_receives_the_same_transport_cap() {
        let mut entries: Vec<FilterEntry> = serde_yaml::from_str(
            "- filter: iterative_request_router\n  initial_step: outer\n  steps:\n    - name: outer\n      filters:\n        - filter: iterative_request_router\n          initial_step: inference\n          steps:\n            - name: inference\n              filters:\n                - filter: openai_agentic_loop\n",
        )
        .unwrap();

        prepare_agentic_budget_entries(&mut entries, &mut HashMap::new()).unwrap();
        assert_eq!(entries[0].config["max_response_bytes"].as_u64(), Some(8_388_608));
        assert_eq!(
            entries[0].config["steps"][0]["filters"][0]["max_response_bytes"].as_u64(),
            Some(8_388_608),
        );
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn loop_listener_raw_body_cap_precedes_parsing() {
        let policy =
            AgenticBudgetPolicy::from_config(&serde_yaml::from_str("max_retained_bytes: 4096").unwrap()).unwrap();
        assert_eq!(agentic_request_body_cap(Some(10_485_760), Some(policy)), Some(128));
        assert_eq!(agentic_request_body_cap(None, Some(policy)), Some(128));
        assert_eq!(agentic_request_body_cap(Some(64), Some(policy)), Some(64));
        assert_eq!(agentic_request_body_cap(Some(10_485_760), None), Some(10_485_760));
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn malformed_loop_transport_cap_is_not_repaired() {
        let mut entries: Vec<FilterEntry> = serde_yaml::from_str(
            "\n- filter: iterative_request_router\n  initial_step: inference\n  max_response_bytes: invalid\n  steps:\n    - name: inference\n      filters:\n        - filter: openai_agentic_loop\n",
        )
        .unwrap();
        let policy = agentic_budget_policy(&entries, &HashMap::new()).unwrap().unwrap();
        assert!(cap_agentic_irr_responses(&mut entries, &HashMap::new(), policy).is_err());
    }

    #[test]
    #[cfg(feature = "openai-responses")]
    fn branch_only_agentic_budget_is_discovered_before_branch_consumption() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: request_id
        branch_chains:
          - name: bounded
            rejoin: terminal
            chains:
              - name: bounded-inline
                filters:
                  - filter: openai_agentic_loop
                    max_retained_bytes: 4096
                  - filter: static_response
                    status: 200
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();
        let chains: HashMap<&str, &[FilterEntry]> = config
            .filter_chains
            .iter()
            .map(|chain| (chain.name.as_str(), chain.filters.as_slice()))
            .collect();
        let entries = &config.filter_chains[0].filters;
        assert_eq!(
            agentic_budget_policy(entries, &chains)
                .unwrap()
                .unwrap()
                .max_retained_bytes(),
            4096,
        );
    }

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
insecure_options:
  allow_private_endpoints: true
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
insecure_options:
  allow_private_endpoints: true
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

    #[cfg(all(feature = "openai-file-resolve-filter", feature = "openai-mcp-tools"))]
    #[test]
    fn resolve_pipelines_transport_limit_governs_openai_responses_raw_body() {
        // Every OpenAI Responses body filter declares a 64 MiB StreamBuffer
        // ceiling. The pipeline's body_limits.max_request_bytes is the only
        // raw transport cap: it must clamp the merged StreamBuffer down to the
        // configured limit, proving a filter's large declaration cannot widen
        // the raw buffer past what the transport allows.
        let config = Config::from_yaml(
            r#"
insecure_options:
  skip_pipeline_validation: true
body_limits:
  max_request_bytes: 1048576
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_file_resolve
        files_api_url: "http://files-api:8321"
        allow_pre_security_callout: true
        outbound_chain:
          name: files-api-outbound
          filters:
            - filter: headers
              request_set:
                - name: x-file-callout
                  value: file-resolve
      - filter: openai_doc_extract
        allow_pre_security_callout: true
      - filter: openai_tool_parse
      - filter: openai_mcp_tool_resolve
      - filter: openai_responses_proxy
"#,
        )
        .unwrap();
        let mut registry = FilterRegistry::with_builtins();
        let client = test_client();
        praxis_ai_filters::register_ai_filters(&mut registry, Some(&client));
        let pipelines = resolve_pipelines(
            &config,
            &registry,
            &empty_health_registry(),
            &empty_kv_stores(),
            &client,
        )
        .unwrap();
        let pipeline = pipelines.get("web").unwrap().load();
        let caps = pipeline.body_capabilities();
        assert_eq!(
            caps.request_body_mode,
            praxis_filter::BodyMode::StreamBuffer {
                max_bytes: Some(1_048_576)
            },
            "the pipeline body_limits ceiling must clamp the filters' 64 MiB StreamBuffer to the transport cap"
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
insecure_options:
  allow_private_endpoints: true
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
insecure_options:
  allow_private_endpoints: true
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
  allow_private_endpoints: true
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

    /// Minimal sub-request client for tests. The connector builds a rustls
    /// config, which needs the process-wide crypto provider first (the binary
    /// installs it at startup; a no-op after the first call).
    fn test_client() -> praxis_core::subrequest::SubRequestClient {
        praxis_tls::provider::install();
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
insecure_options:
  allow_private_endpoints: true
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
