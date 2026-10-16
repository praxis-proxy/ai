// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Offline store planning shared by serving transports.
//!
//! Transports supply chain roots and compiled backend factories. This module
//! discovers filter-owned stores, normalizes their configs, shares compatible
//! backends and rejects ambiguous bindings before any database is opened.

#[cfg(feature = "openai-conversations")]
use std::collections::HashSet;
use std::{collections::HashMap, sync::Arc};

#[cfg(feature = "openai-conversations")]
use praxis_ai_store::EffectiveConfigKey;
use praxis_ai_store::{BackendError, StoreBackendFactory};
use praxis_ai_store_lifecycle::{ProvisionError, StoreRef};
use praxis_core::config::FilterEntry;
use serde_json::Value;

#[cfg(feature = "openai-conversations")]
use super::{CONVERSATIONS_STORE_FILTER_NAME, CONVERSATIONS_STORE_NAME, conversations_store_ref_config};
use super::{DEFAULT_STORE_NAME, RESPONSE_STORE_FILTER_NAME, discovery::find_store_configs};

/// Location of a store reference within a transport's registry plans.
pub type StoreRefLocation = (usize, usize);

/// Discover stores reachable from the selected root chains, without connecting.
///
/// # Errors
///
/// Rejects malformed nested filters or store configurations.
pub fn discover_store_refs(
    roots: &[String],
    chains: &HashMap<&str, &[FilterEntry]>,
) -> Result<Vec<StoreRef>, ProvisionError> {
    let configs = find_store_configs(roots, chains, RESPONSE_STORE_FILTER_NAME)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| backend_error(DEFAULT_STORE_NAME, source))?;
    let refs: Vec<StoreRef> = configs.iter().map(response_store_ref).collect::<Result<_, _>>()?;
    #[cfg(feature = "openai-conversations")]
    {
        let mut refs = refs;
        let configs = find_store_configs(roots, chains, CONVERSATIONS_STORE_FILTER_NAME)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| backend_error(CONVERSATIONS_STORE_NAME, source))?;
        refs.extend(
            configs
                .iter()
                .map(conversations_store_ref)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| backend_error(CONVERSATIONS_STORE_NAME, BackendError::Config(error.to_string())))?,
        );
        Ok(refs)
    }
    #[cfg(not(feature = "openai-conversations"))]
    Ok(refs)
}

/// Discover stores using the Praxis server's existing permissive behavior.
///
/// Malformed nested entries and response-store references are skipped while
/// valid siblings remain discoverable. Invalid Conversations configs are logged
/// and skipped, matching the server's pre-extraction planning behavior.
#[must_use]
pub fn discover_store_refs_permissive(roots: &[String], chains: &HashMap<&str, &[FilterEntry]>) -> Vec<StoreRef> {
    let refs: Vec<StoreRef> = find_store_configs(roots, chains, RESPONSE_STORE_FILTER_NAME)
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|config| response_store_ref(&config).ok())
        .collect();
    #[cfg(feature = "openai-conversations")]
    {
        let mut refs = refs;
        refs.extend(
            find_store_configs(roots, chains, CONVERSATIONS_STORE_FILTER_NAME)
                .into_iter()
                .filter_map(Result::ok)
                .filter_map(|config| permissive_conversations_ref(&config)),
        );
        refs
    }
    #[cfg(not(feature = "openai-conversations"))]
    refs
}

/// Preserve the server's Conversations diagnostics without rejecting discovery.
#[cfg(feature = "openai-conversations")]
fn permissive_conversations_ref(config: &serde_yaml::Value) -> Option<StoreRef> {
    match conversations_store_ref(config) {
        Ok(store_ref) => Some(store_ref),
        Err(error) => {
            tracing::error!(error = %error, "conversations store config could not be prepared for provisioning");
            None
        },
    }
}

/// Translate backend errors without adding connection material to diagnostics.
fn backend_error(name: &str, source: BackendError) -> ProvisionError {
    ProvisionError::Backend {
        name: Arc::from(name),
        source,
    }
}

/// Remove response-filter options before passing configuration to a factory.
fn response_store_ref(filter_config: &serde_yaml::Value) -> Result<StoreRef, ProvisionError> {
    let backend = filter_config
        .get("backend")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| {
            backend_error(
                DEFAULT_STORE_NAME,
                BackendError::Config("response store requires a backend".to_owned()),
            )
        })?;
    let mut config = serde_json::to_value(filter_config).map_err(|_error| {
        backend_error(
            DEFAULT_STORE_NAME,
            BackendError::Config("response store must be a mapping".to_owned()),
        )
    })?;
    let object = config.as_object_mut().ok_or_else(|| {
        backend_error(
            DEFAULT_STORE_NAME,
            BackendError::Config("response store must be a mapping".to_owned()),
        )
    })?;
    for field in ["backend", "max_event_count", "max_event_bytes"] {
        object.remove(field);
    }
    normalize_sqlite_factory_config(backend, &mut config);
    Ok(StoreRef {
        name: Arc::from(DEFAULT_STORE_NAME),
        backend_id: Arc::from(backend),
        config,
    })
}

/// Let the Conversations API own its table defaults and placeholder table.
#[cfg(feature = "openai-conversations")]
fn conversations_store_ref(filter_config: &serde_yaml::Value) -> Result<StoreRef, praxis_filter::FilterError> {
    let (backend, mut config) = conversations_store_ref_config(filter_config)?;
    normalize_sqlite_factory_config(&backend, &mut config);
    Ok(StoreRef {
        name: Arc::from(CONVERSATIONS_STORE_NAME),
        backend_id: Arc::from(backend),
        config,
    })
}

/// Remove accepted no-op `PostgreSQL` fields before dispatching to SQLite.
fn normalize_sqlite_factory_config(backend: &str, config: &mut Value) {
    if backend != "sqlite" {
        return;
    }
    let Some(object) = config.as_object_mut() else {
        return;
    };
    for field in ["ssl_mode", "ssl_root_cert", "ssl_client_cert", "ssl_client_key"] {
        if object.get(field).is_some_and(Value::is_null) {
            object.remove(field);
        }
    }
    for field in ["require_certificate_authentication", "allow_private_database_url"] {
        if object.get(field).and_then(Value::as_bool) == Some(false) {
            object.remove(field);
        }
    }
}

/// Resolve an injected backend factory for one reference.
fn factory_for<'a>(
    store_ref: &StoreRef,
    factories: &HashMap<&str, &'a dyn StoreBackendFactory>,
) -> Result<&'a dyn StoreBackendFactory, ProvisionError> {
    factories
        .get(store_ref.backend_id.as_ref())
        .copied()
        .ok_or_else(|| ProvisionError::UnknownBackend {
            name: Arc::clone(&store_ref.name),
            backend_id: Arc::clone(&store_ref.backend_id),
        })
}

/// Compare connection identity while letting the factory normalize defaults.
#[cfg(feature = "openai-conversations")]
fn connection_key(
    store_ref: &StoreRef,
    factory: &dyn StoreBackendFactory,
) -> Result<EffectiveConfigKey, ProvisionError> {
    let mut config = store_ref.config.clone();
    let object = config.as_object_mut().ok_or_else(|| {
        backend_error(
            &store_ref.name,
            BackendError::Config("store config must be an object".to_owned()),
        )
    })?;
    for (field, table) in [
        ("responses_table", "shared_responses"),
        ("conversations_table", "shared_conversations"),
        ("items_table", "shared_items"),
    ] {
        object.insert(field.to_owned(), Value::String(table.to_owned()));
    }
    factory
        .effective_key(&config)
        .map_err(|source| backend_error(&store_ref.name, source))
}

/// Inventory table and generated index names for collision checks.
fn schema_objects(config: &Value) -> Vec<(&'static str, String)> {
    let mut objects = Vec::new();
    for field in ["responses_table", "conversations_table", "items_table"] {
        if let Some(table) = config.get(field).and_then(Value::as_str) {
            objects.push((field, table.to_owned()));
            match field {
                "responses_table" => objects.extend([
                    (
                        "responses pending-approvals table",
                        format!("{table}_pending_approvals"),
                    ),
                    ("responses schema-version table", format!("{table}_schema_version")),
                    ("responses event-log table", format!("{table}_events")),
                ]),
                "conversations_table" => objects.push(("conversations tenant index", format!("idx_{table}_tenant_id"))),
                _ => objects.extend([
                    ("items conversation index", format!("idx_{table}_conversation")),
                    ("items position index", format!("idx_{table}_position")),
                ]),
            }
        }
    }
    objects
}

/// Allow repeated DDL of the same kind, never a different schema shape.
fn validate_objects(first: &StoreRef, second: &StoreRef) -> Result<(), ProvisionError> {
    validate_config_objects(&first.name, &first.config, &second.config)
}

/// Check two schema inventories without changing either store reference.
fn validate_config_objects(name: &str, first: &Value, second: &Value) -> Result<(), ProvisionError> {
    let first_objects = schema_objects(first);
    let second_objects = schema_objects(second);
    for (first_kind, first_name) in &first_objects {
        for (second_kind, second_name) in &second_objects {
            if first_kind != second_kind && first_name.eq_ignore_ascii_case(second_name) {
                return Err(backend_error(
                    name,
                    BackendError::Config(format!(
                        "Store configurations target the same SQL namespace, but {first_kind} collides with {second_kind}; configure distinct responses_table, conversations_table, and items_table names"
                    )),
                ));
            }
        }
    }
    Ok(())
}

/// Form a combined table set only when non-table identities also match.
#[cfg(feature = "openai-conversations")]
fn combined_config(
    response: &StoreRef,
    conversation: &StoreRef,
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<Option<Value>, ProvisionError> {
    if response.backend_id != conversation.backend_id {
        return Ok(None);
    }
    let Some(factory) = factories.get(response.backend_id.as_ref()).copied() else {
        return Ok(None);
    };
    if connection_key(response, factory)? != connection_key(conversation, factory)?
        || response.config.get("conversations_table") != conversation.config.get("conversations_table")
    {
        return Ok(None);
    }
    let Some(items_table) = conversation.config.get("items_table").cloned() else {
        return Ok(None);
    };
    let mut combined = response.config.clone();
    let Some(object) = combined.as_object_mut() else {
        return Ok(None);
    };
    object.insert("items_table".to_owned(), items_table);
    if complete_sql_inventory(&combined) {
        validate_config_objects(&response.name, &combined, &combined)?;
    }
    Ok(Some(combined))
}

/// The server inventories only configs with both required SQL table names.
fn complete_sql_inventory(config: &Value) -> bool {
    ["responses_table", "conversations_table"]
        .iter()
        .all(|field| config.get(field).and_then(Value::as_str).is_some())
}

/// One compatible pairing and its complete factory-normalized identity.
#[cfg(feature = "openai-conversations")]
struct CombinedCandidate {
    /// Responses location in the caller's plans.
    response: StoreRefLocation,
    /// Conversations location in the caller's plans.
    conversation: StoreRefLocation,
    /// Shared backend factory identifier.
    backend_id: Arc<str>,
    /// Effective key for the combined table set.
    effective: EffectiveConfigKey,
    /// Combined factory configuration.
    config: Value,
}

/// Find references with one registry name across every plan.
#[cfg(feature = "openai-conversations")]
fn named_refs<'a, P: AsRef<[StoreRef]>>(plans: &'a [P], name: &'a str) -> Vec<(StoreRefLocation, &'a StoreRef)> {
    plans
        .iter()
        .enumerate()
        .flat_map(|(plan_index, plan)| {
            plan.as_ref()
                .iter()
                .enumerate()
                .filter_map(move |(ref_index, store_ref)| {
                    (store_ref.name.as_ref() == name).then_some(((plan_index, ref_index), store_ref))
                })
        })
        .collect()
}

/// Build all compatible pairings; ambiguous pairings remain unpromoted.
#[cfg(feature = "openai-conversations")]
fn combined_candidates<P: AsRef<[StoreRef]>>(
    plans: &[P],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<Vec<CombinedCandidate>, ProvisionError> {
    let response_refs = named_refs(plans, DEFAULT_STORE_NAME);
    let conversation_refs = named_refs(plans, CONVERSATIONS_STORE_NAME);
    let mut candidates = Vec::new();
    for &(response, response_ref) in &response_refs {
        for &(conversation, conversation_ref) in &conversation_refs {
            if let Some(config) = combined_config(response_ref, conversation_ref, factories)? {
                let effective = factory_for(response_ref, factories)?
                    .effective_key(&config)
                    .map_err(|source| backend_error(&response_ref.name, source))?;
                candidates.push(CombinedCandidate {
                    response,
                    conversation,
                    backend_id: Arc::clone(&response_ref.backend_id),
                    effective,
                    config,
                });
            }
        }
    }
    Ok(candidates)
}

/// Select a pairing only when all candidates imply the same combined key.
#[cfg(feature = "openai-conversations")]
fn selected_candidate<'a>(
    candidates: &'a [CombinedCandidate],
    location: StoreRefLocation,
    name: &str,
) -> Option<&'a CombinedCandidate> {
    let mut eligible = candidates.iter().filter(|candidate| match name {
        DEFAULT_STORE_NAME => candidate.response == location,
        CONVERSATIONS_STORE_NAME => candidate.conversation == location,
        _ => false,
    });
    let first = eligible.next()?;
    eligible
        .all(|candidate| candidate.backend_id == first.backend_id && candidate.effective == first.effective)
        .then_some(first)
}

/// Require both registry references to select the same combined backend.
#[cfg(feature = "openai-conversations")]
fn mutually_selected_candidate<'a>(
    candidates: &'a [CombinedCandidate],
    location: StoreRefLocation,
    name: &str,
) -> Option<&'a CombinedCandidate> {
    let candidate = selected_candidate(candidates, location, name)?;
    let response = selected_candidate(candidates, candidate.response, DEFAULT_STORE_NAME)?;
    let conversation = selected_candidate(candidates, candidate.conversation, CONVERSATIONS_STORE_NAME)?;
    (response.backend_id == candidate.backend_id
        && response.effective == candidate.effective
        && conversation.backend_id == candidate.backend_id
        && conversation.effective == candidate.effective)
        .then_some(candidate)
}

/// Promote compatible Responses/Conversations stores across registry plans.
///
/// Returns promoted locations so a reload owner can preserve active cache keys
/// only for the remaining references. No backend is opened here.
///
/// # Errors
///
/// Returns an error when a backend rejects a proposed combined configuration.
#[cfg(feature = "openai-conversations")]
pub fn coalesce_compatible_filter_refs<P: AsRef<[StoreRef]> + AsMut<[StoreRef]>>(
    plans: &mut [P],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<HashSet<StoreRefLocation>, ProvisionError> {
    let candidates = combined_candidates(plans, factories)?;
    let mut promoted = HashSet::new();
    for (plan_index, plan) in plans.iter_mut().enumerate() {
        for (ref_index, store_ref) in plan.as_mut().iter_mut().enumerate() {
            if let Some(candidate) = mutually_selected_candidate(&candidates, (plan_index, ref_index), &store_ref.name)
            {
                store_ref.config.clone_from(&candidate.config);
                promoted.insert((plan_index, ref_index));
            }
        }
    }
    Ok(promoted)
}

/// Reject incompatible table or index names in the final SQL namespace plan.
///
/// # Errors
///
/// Returns a configuration error for generated-object collisions or invalid backends.
pub fn validate_sql_namespace_collisions<P: AsRef<[StoreRef]>>(
    plans: &[P],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<(), ProvisionError> {
    let refs: Vec<&StoreRef> = plans.iter().flat_map(AsRef::as_ref).collect();
    for (index, first) in refs.iter().enumerate() {
        let factory = factory_for(first, factories)?;
        let namespace = factory
            .namespace_key(&first.config)
            .map_err(|source| backend_error(&first.name, source))?;
        validate_objects(first, first)?;
        for second in refs.iter().skip(index + 1) {
            if first.backend_id == second.backend_id
                && namespace.is_some()
                && namespace
                    == factory
                        .namespace_key(&second.config)
                        .map_err(|source| backend_error(&second.name, source))?
            {
                validate_objects(first, second)?;
            }
        }
    }
    Ok(())
}

/// Check cross-store SQL namespace collisions using the server's existing scope.
///
/// Unknown factories and incomplete SQL inventories are left to backend/pipeline
/// validation. Single-store collision checks remain part of combined promotion,
/// not an additional validation pass over every raw reference.
///
/// # Errors
///
/// Returns an error for overlapping incompatible schema objects or backend keys.
pub fn validate_cross_store_collisions<P: AsRef<[StoreRef]>>(
    plans: &[P],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<(), ProvisionError> {
    let refs: Vec<&StoreRef> = plans.iter().flat_map(AsRef::as_ref).collect();
    for (index, first) in refs.iter().enumerate() {
        let Some(factory) = factories.get(first.backend_id.as_ref()).copied() else {
            continue;
        };
        let Some(namespace) = factory
            .namespace_key(&first.config)
            .map_err(|source| backend_error(&first.name, source))?
        else {
            continue;
        };
        for second in refs
            .iter()
            .skip(index + 1)
            .filter(|second| second.backend_id == first.backend_id)
        {
            let second_namespace = factory
                .namespace_key(&second.config)
                .map_err(|source| backend_error(&second.name, source))?;
            if second_namespace.as_ref() == Some(&namespace)
                && complete_sql_inventory(&first.config)
                && complete_sql_inventory(&second.config)
            {
                validate_objects(first, second)?;
            }
        }
    }
    Ok(())
}

/// Collapse repeated bindings within one registry, rejecting different identities.
///
/// # Errors
///
/// Returns an error for unknown backends, invalid configs or conflicting bindings.
pub fn deduplicate_refs(
    context: &str,
    refs: &mut Vec<StoreRef>,
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<(), ProvisionError> {
    let mut bound = HashMap::new();
    let mut unique = Vec::with_capacity(refs.len());
    for store_ref in refs.drain(..) {
        let factory = factory_for(&store_ref, factories)?;
        let key = (
            Arc::clone(&store_ref.backend_id),
            factory
                .effective_key(&store_ref.config)
                .map_err(|source| backend_error(&store_ref.name, source))?,
        );
        if let Some(existing) = bound.get(&store_ref.name) {
            if existing != &key {
                return Err(backend_error(
                    &store_ref.name,
                    BackendError::Config(format!(
                        "{context} configures conflicting stores for registry name '{}'",
                        store_ref.name
                    )),
                ));
            }
        } else {
            bound.insert(Arc::clone(&store_ref.name), key);
            unique.push(store_ref);
        }
    }
    *refs = unique;
    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use async_trait::async_trait;
    use praxis_ai_store::{BackendNamespaceKey, EffectiveConfigKey, ProvisionedBackend};
    use serde_json::json;

    use super::*;

    #[test]
    fn permissive_discovery_keeps_valid_nested_siblings() {
        let filters: Vec<FilterEntry> = serde_yaml::from_str(
            "- filter: openai_responses_store\n  backend: 7\n- filter: iterative_request_router\n  steps:\n    - filters: malformed\n    - filters: [7, {filter: openai_responses_store, backend: sqlite, database_url: 'sqlite::memory:'}]",
        ).expect("mixed filter chain");
        let chains = HashMap::from([("main", filters.as_slice())]);
        let roots = ["main".to_owned()];
        let refs = discover_store_refs_permissive(&roots, &chains);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].backend_id.as_ref(), "sqlite");
        assert!(
            discover_store_refs(&roots, &chains).is_err(),
            "strict discovery still rejects malformed filters"
        );
    }

    #[test]
    fn permissive_discovery_skips_invalid_response_references() {
        let filters: Vec<FilterEntry> =
            serde_yaml::from_str("[{filter: openai_responses_store}]").expect("filter chain");
        let chains = HashMap::from([("main", filters.as_slice())]);
        let roots = ["main".to_owned()];
        assert!(discover_store_refs_permissive(&roots, &chains).is_empty());
        assert!(discover_store_refs(&roots, &chains).is_err());
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn permissive_discovery_skips_invalid_conversation_references() {
        let filters: Vec<FilterEntry> =
            serde_yaml::from_str("[{filter: openai_conversations, backend: unsupported}]").expect("filter chain");
        let chains = HashMap::from([("main", filters.as_slice())]);
        let roots = ["main".to_owned()];
        assert!(discover_store_refs_permissive(&roots, &chains).is_empty());
        assert!(discover_store_refs(&roots, &chains).is_err());
    }

    #[test]
    fn cross_store_validation_retains_the_server_scope() {
        let mut store_ref = reference("default", "a");
        store_ref.config["responses_table"] = json!("responses");
        store_ref.config["conversations_table"] = json!("responses");
        let plans = vec![vec![store_ref]];
        validate_cross_store_collisions(&plans, &factories()).expect("no cross-store pairs");
        assert!(validate_sql_namespace_collisions(&plans, &factories()).is_err());
    }

    #[test]
    fn cross_store_validation_leaves_incomplete_inventories_to_backend_validation() {
        let mut response = reference("default", "a");
        response.config["responses_table"] = json!("responses");
        let mut conversation = reference("conversations", "a");
        conversation.config["responses_table"] = json!("unused");
        conversation.config["conversations_table"] = json!("conversations");
        conversation.config["items_table"] = json!("responses");
        let plans = vec![vec![response, conversation]];
        validate_cross_store_collisions(&plans, &factories()).expect("incomplete inventory is not newly rejected");
        assert!(validate_sql_namespace_collisions(&plans, &factories()).is_err());
    }

    #[test]
    fn stateless_chains_require_no_backend() {
        let filters: Vec<FilterEntry> = serde_yaml::from_str("[{filter: request_id}]").expect("stateless chain");
        let chains = HashMap::from([("main", filters.as_slice())]);
        let mut refs = discover_store_refs(&["main".to_owned()], &chains).expect("stateless discovery");
        assert!(refs.is_empty());
        validate_sql_namespace_collisions(std::slice::from_ref(&refs), &HashMap::new()).expect("no namespace");
        deduplicate_refs("test", &mut refs, &HashMap::new()).expect("no bindings");
    }

    #[test]
    fn generated_objects_cannot_collide_inside_one_store() {
        for table in [
            "responses_events",
            "RESPONSES_PENDING_APPROVALS",
            "idx_conversations_tenant_id",
        ] {
            let mut store_ref = reference("default", "a");
            store_ref.config["responses_table"] = json!("responses");
            store_ref.config["conversations_table"] = json!("conversations");
            store_ref.config["items_table"] = json!(table);
            assert!(
                validate_objects(&store_ref, &store_ref).is_err(),
                "generated collision: {table}"
            );
        }
    }

    #[test]
    fn sqlite_normalization_removes_only_accepted_noop_fields() {
        let mut config = json!({"ssl_mode": null, "ssl_root_cert": null,
            "ssl_client_cert": null, "ssl_client_key": null,
            "require_certificate_authentication": false, "allow_private_database_url": false,
            "database_url": "sqlite::memory:"});
        normalize_sqlite_factory_config("sqlite", &mut config);
        assert_eq!(config, json!({"database_url": "sqlite::memory:"}));
    }

    #[test]
    fn sqlite_normalization_preserves_rejected_nondefault_fields() {
        let mut config = json!({"ssl_mode": "verify_full",
            "require_certificate_authentication": true, "allow_private_database_url": true});
        let before = config.clone();
        normalize_sqlite_factory_config("sqlite", &mut config);
        assert_eq!(config, before);
    }

    #[test]
    fn response_options_stay_out_of_backend_config() {
        let filter = serde_yaml::from_str(
            "backend: sqlite\ndatabase_url: 'sqlite::memory:'\nmax_event_count: 10\nmax_event_bytes: 1024",
        )
        .expect("filter config");
        let store_ref = response_store_ref(&filter).expect("store reference");
        for field in ["backend", "max_event_count", "max_event_bytes"] {
            assert!(store_ref.config.get(field).is_none(), "filter-only option: {field}");
        }
        assert!(response_store_ref(&serde_yaml::from_str("database_url: secret").expect("config")).is_err());
    }

    #[test]
    fn deduplication_rejects_unknown_or_conflicting_bindings() {
        let mut refs = vec![reference("default", "a"), reference("default", "a")];
        deduplicate_refs("test", &mut refs, &factories()).expect("identical bindings");
        assert_eq!(refs.len(), 1);
        refs.push(reference("default", "b"));
        let error = deduplicate_refs("test", &mut refs, &factories()).expect_err("conflicting bindings");
        assert!(error.to_string().contains("conflicting stores"));
        assert!(deduplicate_refs("test", &mut vec![reference("default", "a")], &HashMap::new()).is_err());
    }

    #[test]
    fn generated_schema_collisions_require_one_namespace() {
        let mut response = reference("default", "a");
        response.config["responses_table"] = json!("responses");
        let mut conversation = reference("conversations", "a");
        conversation.config["items_table"] = json!("RESPONSES_EVENTS");
        let mut refs = vec![response, conversation];
        assert!(validate_sql_namespace_collisions(std::slice::from_ref(&refs), &factories()).is_err());
        refs[1].config["database_url"] = json!("b");
        validate_sql_namespace_collisions(std::slice::from_ref(&refs), &factories()).expect("different namespaces");
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn compatible_stores_share_across_registry_plans() {
        let mut response = reference(DEFAULT_STORE_NAME, "a");
        response.config["responses_table"] = json!("responses");
        response.config["conversations_table"] = json!("conversations");
        let mut conversation = reference(CONVERSATIONS_STORE_NAME, "a");
        conversation.config["responses_table"] = json!("unused_responses");
        conversation.config["conversations_table"] = json!("conversations");
        conversation.config["items_table"] = json!("items");
        let mut plans = vec![vec![response], vec![conversation]];
        let promoted = coalesce_compatible_filter_refs(&mut plans, &factories()).expect("compatible stores");
        assert_eq!(promoted.len(), 2);
        assert_eq!(plans[0][0].config, plans[1][0].config);
        assert_eq!(plans[0][0].config["items_table"], "items");
        validate_sql_namespace_collisions(&plans, &factories()).expect("combined schema");
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn different_connection_options_do_not_share_backends() {
        let mut response = reference(DEFAULT_STORE_NAME, "a");
        response.config["conversations_table"] = json!("conversations");
        let mut conversation = reference(CONVERSATIONS_STORE_NAME, "a");
        conversation.config["conversations_table"] = json!("conversations");
        conversation.config["items_table"] = json!("items");
        conversation.config["pool"] = json!({"max_connections": 2});
        let mut plans = vec![vec![response, conversation]];
        assert!(
            coalesce_compatible_filter_refs(&mut plans, &factories())
                .expect("valid refs")
                .is_empty()
        );
    }

    /// Offline factory comparing normalized configuration without opening a pool.
    struct TestFactory;

    #[async_trait]
    impl StoreBackendFactory for TestFactory {
        fn backend_id(&self) -> &str {
            "sqlite"
        }

        fn effective_key(&self, config: &Value) -> Result<EffectiveConfigKey, BackendError> {
            if config.get("database_url").and_then(Value::as_str).is_none() {
                return Err(BackendError::Config("missing database_url".to_owned()));
            }
            let fields: std::collections::BTreeMap<_, _> = config.as_object().expect("config object").iter().collect();
            Ok(EffectiveConfigKey::new(
                serde_json::to_string(&fields).expect("normalized config"),
            ))
        }

        fn namespace_key(&self, config: &Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            Ok(config
                .get("database_url")
                .and_then(Value::as_str)
                .map(BackendNamespaceKey::new))
        }

        async fn build(&self, _config: &Value) -> Result<ProvisionedBackend, BackendError> {
            Err(BackendError::Unavailable("offline test factory".to_owned()))
        }
    }

    /// Borrow the offline factory by its backend id.
    fn factories() -> HashMap<&'static str, &'static dyn StoreBackendFactory> {
        let factory: &dyn StoreBackendFactory = &TestFactory;
        HashMap::from([("sqlite", factory)])
    }

    /// Minimal lifecycle reference for offline tests.
    fn reference(name: &str, url: &str) -> StoreRef {
        StoreRef {
            name: Arc::from(name),
            backend_id: Arc::from("sqlite"),
            config: json!({"database_url": url}),
        }
    }
}
