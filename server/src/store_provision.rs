// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Serving-runtime provisioning of response-store backends.
//!
//! The store filter reads an owner-scoped handle from a per-listener registry it
//! never populates. A Pingora background service provisions the configured
//! backends and registers them into that registry on the serving runtime. sqlx
//! pools bind to the runtime that opens them, so provisioning must run there, not
//! on the config-watcher runtime a reload uses.

#![cfg(any(feature = "store-postgres", feature = "store-sqlite"))]

#[cfg(feature = "openai-conversations")]
use std::collections::HashSet;
use std::{
    collections::HashMap,
    sync::{Arc, Weak, mpsc as std_mpsc},
    time::Duration,
};

use async_trait::async_trait;
use futures::{StreamExt as _, stream::FuturesUnordered};
#[cfg(feature = "openai-conversations")]
use percent_encoding::percent_decode_str;
use pingora_core::{
    server::ShutdownWatch,
    services::{ServiceReadyNotifier, background::BackgroundService},
};
#[cfg(feature = "openai-conversations")]
use praxis_ai_apis::store::{
    CONVERSATIONS_STORE_FILTER_NAME, CONVERSATIONS_STORE_NAME, conversations_store_ref_config,
};
use praxis_ai_apis::store::{
    DEFAULT_STORE_NAME, RESPONSE_STORE_FILTER_NAME, ResponseStoreRegistry, store_backend_factories,
};
use praxis_ai_store::{BackendError, EffectiveConfigKey, StoreBackendFactory, StoreRegistry};
use praxis_ai_store_lifecycle::{BackendCache, BackendLease, ProvisionError, StoreRef};
use praxis_core::config::{Config, FilterEntry};
use praxis_filter::FilterPipeline;
use tokio::sync::{Mutex as AsyncMutex, mpsc, watch};
use tracing::{error, info};

use crate::store_config::find_listener_store_configs;

/// Poll interval while an old pipeline generation drains in-flight requests.
const PIPELINE_DRAIN_POLL: Duration = Duration::from_millis(10);

/// Readiness of the configured response-store backends.
///
/// One aggregate state across every configured listener: `Ready` only once all
/// hold a provisioned backend. Observers read it through a [`StoreReadinessHandle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreReadiness {
    /// Provisioning has not yet completed for every listener.
    Pending,
    /// The initial generation failed terminally. The production startup path
    /// exits after publishing this state so it never serves persistent 503s.
    Failed,
    /// Every configured listener holds a provisioned backend.
    Ready,
}

/// A cloneable, read-only handle to observe store-provisioning readiness.
///
/// The provisioning background service owns the sender. Consumers (the readiness
/// endpoint, the integration harness, sibling workstreams) clone this to observe
/// or await readiness. One owned state, many observers.
#[derive(Clone)]
pub struct StoreReadinessHandle {
    /// Receives readiness transitions from the provisioning service.
    rx: watch::Receiver<StoreReadiness>,
}

impl StoreReadinessHandle {
    /// A handle already at `Ready`, for when no store is configured. The sender
    /// is dropped, so the value never changes and `wait_ready` returns at once.
    #[must_use]
    pub fn ready() -> Self {
        let (_tx, rx) = watch::channel(StoreReadiness::Ready);
        Self { rx }
    }

    /// The current readiness snapshot.
    #[must_use]
    pub fn current(&self) -> StoreReadiness {
        *self.rx.borrow()
    }

    /// Whether every configured store is provisioned.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.current() == StoreReadiness::Ready
    }

    /// Await until provisioning either succeeds or fails terminally. Returns at
    /// once when already settled, and also returns if the provisioner stops so
    /// a caller cannot block forever. Inspect [`Self::current`] for the result.
    pub async fn wait_ready(&mut self) {
        // Err only if the sender drops while still pending. Either way, stop
        // waiting. The borrowed guard is released at once.
        let _settled = self.rx.wait_for(|s| *s != StoreReadiness::Pending).await;
    }
}

/// A listener's shared store registry and the references provisioned into it.
///
/// The registry handle is installed into the listener's pipeline (as a
/// [`ResponseStoreRegistry`]) before serving starts, and the background service
/// registers the provisioned backends into the same backing map.
#[derive(Clone)]
struct ListenerStorePlan {
    /// Listener the plan belongs to.
    listener: String,
    /// Shared registry backing both the pipeline handle and provisioning.
    registry: StoreRegistry,
    /// Store references to provision (one default store, so at most one).
    refs: Vec<StoreRef>,
}

/// Return whether `candidate` is the same filter-owned store configuration as
/// `active`, ignoring only the other filter's table added during promotion.
#[cfg(feature = "openai-conversations")]
fn same_filter_identity(
    candidate: &StoreRef,
    active: &StoreRef,
    factory: &dyn StoreBackendFactory,
) -> Result<bool, BackendError> {
    if candidate.name != active.name || candidate.backend_id != active.backend_id {
        return Ok(false);
    }
    let mut candidate_config = candidate.config.clone();
    let mut active_config = active.config.clone();
    let (Some(candidate_object), Some(active_object)) =
        (candidate_config.as_object_mut(), active_config.as_object_mut())
    else {
        return Ok(false);
    };
    match candidate.name.as_ref() {
        DEFAULT_STORE_NAME => {
            candidate_object.remove("items_table");
            active_object.remove("items_table");
        },
        CONVERSATIONS_STORE_NAME => {
            let sentinel = serde_json::Value::String("shared_responses".to_owned());
            candidate_object.insert("responses_table".to_owned(), sentinel.clone());
            active_object.insert("responses_table".to_owned(), sentinel);
        },
        _ => return Ok(false),
    }
    Ok(factory.effective_key(&active_config)? == factory.effective_key(&candidate_config)?)
}

/// Whether a store reference targets a process-local SQLite database.
#[cfg(feature = "openai-conversations")]
fn is_in_memory_sqlite(store_ref: &StoreRef) -> bool {
    if store_ref.backend_id.as_ref() != "sqlite" {
        return false;
    }
    let Some(url) = store_ref.config.get("database_url").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let url = url
        .trim()
        .strip_prefix("sqlite://")
        .or_else(|| url.trim().strip_prefix("sqlite:"))
        .unwrap_or_else(|| url.trim());
    let (database, query) = url.split_once('?').unwrap_or((url, ""));
    let database = percent_decode_str(database).decode_utf8_lossy();
    if matches!(database.as_ref(), ":memory:" | "file::memory:") {
        return true;
    }
    query.split('&').any(|parameter| {
        percent_decode_str(parameter)
            .decode_utf8_lossy()
            .eq_ignore_ascii_case("mode=memory")
    })
}

/// Preserve exact active cache keys for logically unchanged filter stores.
#[cfg(feature = "openai-conversations")]
#[expect(
    clippy::too_many_lines,
    reason = "active key preservation and restart-required in-memory promotion are one reload decision"
)]
fn preserve_active_store_configs(
    plans: &mut [ListenerStorePlan],
    active_plans: &[ListenerStorePlan],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
    cache: &BackendCache,
    promoted: &HashSet<StoreRefLocation>,
) -> Result<(), ProvisionError> {
    for (plan_index, plan) in plans.iter_mut().enumerate() {
        for (ref_index, store_ref) in plan.refs.iter_mut().enumerate() {
            let Some(factory) = factories.get(store_ref.backend_id.as_ref()) else {
                continue;
            };
            if promoted.contains(&(plan_index, ref_index)) {
                if is_in_memory_sqlite(store_ref) {
                    let candidate_key =
                        factory
                            .effective_key(&store_ref.config)
                            .map_err(|source| ProvisionError::Backend {
                                name: Arc::clone(&store_ref.name),
                                source,
                            })?;
                    for active_ref in active_plans.iter().flat_map(|active| &active.refs) {
                        if active_ref.backend_id == store_ref.backend_id {
                            let active_key = factory.effective_key(&active_ref.config).map_err(|source| {
                                ProvisionError::Backend {
                                    name: Arc::clone(&store_ref.name),
                                    source,
                                }
                            })?;
                            if active_key == candidate_key {
                                continue;
                            }
                        }
                        let is_source = same_filter_identity(store_ref, active_ref, *factory).map_err(|source| {
                            ProvisionError::Backend {
                                name: Arc::clone(&store_ref.name),
                                source,
                            }
                        })?;
                        if is_source && cache.contains(active_ref)? {
                            return Err(ProvisionError::Backend {
                                name: Arc::clone(&store_ref.name),
                                source: BackendError::Config(
                                    "adding a compatible Conversations store to an active in-memory SQLite Responses store requires a restart because replacing its pool would lose persisted state"
                                        .to_owned(),
                                ),
                            });
                        }
                    }
                }
                continue;
            }
            for active_ref in active_plans.iter().flat_map(|active| &active.refs) {
                let same = same_filter_identity(store_ref, active_ref, *factory).map_err(|source| {
                    ProvisionError::Backend {
                        name: Arc::clone(&store_ref.name),
                        source,
                    }
                })?;
                if same && cache.contains(active_ref)? {
                    store_ref.config.clone_from(&active_ref.config);
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Build the response-store reference from a response-store filter's config.
///
/// The `backend` field selects the factory. The rest is the inline config the
/// factory parses. Returns `None` for a config missing a string `backend` or one
/// that cannot map to JSON.
fn response_store_ref(filter_config: &serde_yaml::Value) -> Option<StoreRef> {
    let backend = filter_config.get("backend").and_then(serde_yaml::Value::as_str)?;
    let mut config = serde_json::to_value(filter_config).ok()?;
    let object = config.as_object_mut()?;
    object.remove("backend");
    // The replay event-log bounds are consumed by the response-store filter, not
    // the backend factory, whose config rejects unknown fields. Drop them for
    // every backend before dispatching so they never reach the factory parse.
    object.remove("max_event_count");
    object.remove("max_event_bytes");
    normalize_sqlite_factory_config(backend, &mut config);
    Some(StoreRef {
        name: Arc::from(DEFAULT_STORE_NAME),
        backend_id: Arc::from(backend),
        config,
    })
}

/// Build the conversations-store reference from a conversations filter's config.
///
/// The apis layer owns the table defaults and the generated (unused)
/// responses-table name the combined backend requires. The store is registered
/// under its own name; the lifecycle cache still shares one backend with the
/// response store when their effective configs match.
#[cfg(feature = "openai-conversations")]
fn conversations_store_ref(filter_config: &serde_yaml::Value) -> Option<StoreRef> {
    match conversations_store_ref_config(filter_config) {
        Ok((backend_id, mut config)) => {
            normalize_sqlite_factory_config(&backend_id, &mut config);
            Some(StoreRef {
                name: Arc::from(CONVERSATIONS_STORE_NAME),
                backend_id: Arc::from(backend_id.as_str()),
                config,
            })
        },
        Err(e) => {
            error!(error = %e, "conversations store config could not be prepared for provisioning");
            None
        },
    }
}

/// Remove accepted no-op `PostgreSQL` fields before dispatching to `SQLite`.
///
/// The public filter configs accept explicit false/null defaults for backward
/// compatibility, while the backend factory intentionally rejects fields that
/// are not part of its SQLite schema. Preserve that strict factory boundary by
/// removing only values that filter validation has already established as
/// no-ops. Non-default values remain and still fail closed.
fn normalize_sqlite_factory_config(backend_id: &str, config: &mut serde_json::Value) {
    if backend_id != "sqlite" {
        return;
    }
    let Some(config) = config.as_object_mut() else {
        return;
    };
    for field in ["ssl_mode", "ssl_root_cert", "ssl_client_cert", "ssl_client_key"] {
        if config.get(field).is_some_and(serde_json::Value::is_null) {
            config.remove(field);
        }
    }
    for field in ["require_certificate_authentication", "allow_private_database_url"] {
        if config.get(field).and_then(serde_json::Value::as_bool) == Some(false) {
            config.remove(field);
        }
    }
}

/// Return an effective key that compares connection, TLS, compression, and
/// pool identity while deliberately ignoring the tables one filter uses.
///
/// Responses and Conversations expose different table-shaped configs even
/// when they target the same combined backend. Replacing every table with the
/// same valid sentinel lets the backend factory retain ownership of all other
/// default normalization used by its real cache key.
#[cfg(feature = "openai-conversations")]
fn table_agnostic_effective_key(
    factory: &dyn StoreBackendFactory,
    store_ref: &StoreRef,
) -> Result<EffectiveConfigKey, BackendError> {
    let mut config = store_ref.config.clone();
    let object = config
        .as_object_mut()
        .ok_or_else(|| BackendError::Config("store config must be an object".to_owned()))?;
    object.insert(
        "responses_table".to_owned(),
        serde_json::Value::String("shared_responses".to_owned()),
    );
    object.insert(
        "conversations_table".to_owned(),
        serde_json::Value::String("shared_conversations".to_owned()),
    );
    object.insert(
        "items_table".to_owned(),
        serde_json::Value::String("shared_items".to_owned()),
    );
    factory.effective_key(&config)
}

/// Kind of SQL schema object created by a store configuration.
#[cfg(feature = "openai-conversations")]
#[derive(Clone, Copy, Eq, PartialEq)]
enum SqlObjectKind {
    /// Main response-object table.
    ResponsesTable,
    /// Pending tool-approval table associated with a response table.
    PendingApprovalsTable,
    /// Schema-version table associated with a response table.
    SchemaVersionTable,
    /// Durable SSE event-log table associated with a response table.
    EventsTable,
    /// Conversation metadata table.
    ConversationsTable,
    /// Tenant lookup index associated with a conversations table.
    ConversationsTenantIndex,
    /// Conversation-item table.
    ItemsTable,
    /// Conversation lookup index associated with an items table.
    ItemsConversationIndex,
    /// Position uniqueness index associated with an items table.
    ItemsPositionIndex,
}

/// One table or generated index created by a store configuration.
#[cfg(feature = "openai-conversations")]
struct SqlObject {
    /// Schema shape represented by this object.
    kind: SqlObjectKind,
    /// Human-readable config or generated-object owner for diagnostics.
    owner: &'static str,
    /// SQL namespace name compared case-insensitively.
    name: String,
}

/// Inventory the schema objects created by one store configuration.
#[cfg(feature = "openai-conversations")]
#[expect(
    clippy::too_many_lines,
    reason = "the complete SQL object-name inventory must remain visible together"
)]
fn sql_objects(config: &serde_json::Value) -> Option<Vec<SqlObject>> {
    let responses_table = config.get("responses_table")?.as_str()?;
    let conversations_table = config.get("conversations_table")?.as_str()?;
    let mut objects = vec![
        SqlObject {
            kind: SqlObjectKind::ResponsesTable,
            owner: "responses_table",
            name: responses_table.to_owned(),
        },
        SqlObject {
            kind: SqlObjectKind::PendingApprovalsTable,
            owner: "responses pending-approvals table",
            name: format!("{responses_table}_pending_approvals"),
        },
        SqlObject {
            kind: SqlObjectKind::SchemaVersionTable,
            owner: "responses schema-version table",
            name: format!("{responses_table}_schema_version"),
        },
        SqlObject {
            kind: SqlObjectKind::EventsTable,
            owner: "responses event-log table",
            name: format!("{responses_table}_events"),
        },
        SqlObject {
            kind: SqlObjectKind::ConversationsTable,
            owner: "conversations_table",
            name: conversations_table.to_owned(),
        },
        SqlObject {
            kind: SqlObjectKind::ConversationsTenantIndex,
            owner: "conversations tenant index",
            name: format!("idx_{conversations_table}_tenant_id"),
        },
    ];
    if let Some(items_table) = config.get("items_table").and_then(serde_json::Value::as_str) {
        objects.extend([
            SqlObject {
                kind: SqlObjectKind::ItemsTable,
                owner: "items_table",
                name: items_table.to_owned(),
            },
            SqlObject {
                kind: SqlObjectKind::ItemsConversationIndex,
                owner: "items conversation index",
                name: format!("idx_{items_table}_conversation"),
            },
            SqlObject {
                kind: SqlObjectKind::ItemsPositionIndex,
                owner: "items position index",
                name: format!("idx_{items_table}_position"),
            },
        ]);
    }
    Some(objects)
}

/// Return the first incompatible object-name collision between two stores.
///
/// Equal kinds are compatible duplicate DDL, such as both filters creating the
/// same conversations table. Reusing the name for a different kind is unsafe.
#[cfg(feature = "openai-conversations")]
fn cross_store_schema_name_collision(
    response_ref: &StoreRef,
    conversation_ref: &StoreRef,
) -> Option<(&'static str, &'static str)> {
    let response_objects = sql_objects(&response_ref.config)?;
    let conversation_objects = sql_objects(&conversation_ref.config)?;
    response_objects.iter().find_map(|response| {
        conversation_objects.iter().find_map(|conversation| {
            (response.kind != conversation.kind && response.name.eq_ignore_ascii_case(&conversation.name))
                .then_some((response.owner, conversation.owner))
        })
    })
}

/// Return the first incompatible collision within one final schema inventory.
#[cfg(feature = "openai-conversations")]
fn schema_name_collision(objects: &[SqlObject]) -> Option<(&'static str, &'static str)> {
    objects.iter().enumerate().find_map(|(index, first)| {
        objects.iter().skip(index + 1).find_map(|second| {
            (first.kind != second.kind && first.name.eq_ignore_ascii_case(&second.name))
                .then_some((first.owner, second.owner))
        })
    })
}

/// Build an actionable SQL namespace-collision error.
#[cfg(feature = "openai-conversations")]
fn namespace_collision_error(store_ref: &StoreRef, first: &str, second: &str) -> ProvisionError {
    ProvisionError::Backend {
        name: Arc::clone(&store_ref.name),
        source: BackendError::Config(format!(
            "Store configurations target the same SQL namespace, but {first} collides with {second}; configure distinct responses_table, conversations_table, and items_table names"
        )),
    }
}

/// Build the combined table configuration when two filter references share all
/// non-table backend identity and select the same conversations table.
#[cfg(feature = "openai-conversations")]
#[expect(
    clippy::too_many_lines,
    reason = "coalescing validation is clearer as one transactional decision"
)]
fn combined_filter_config(
    response_ref: &StoreRef,
    conversation_ref: &StoreRef,
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<Option<serde_json::Value>, ProvisionError> {
    if response_ref.backend_id != conversation_ref.backend_id {
        return Ok(None);
    }
    let Some(factory) = factories.get(response_ref.backend_id.as_ref()) else {
        return Ok(None);
    };
    let sharing_key = |store_ref: &StoreRef| {
        table_agnostic_effective_key(*factory, store_ref).map_err(|source| ProvisionError::Backend {
            name: Arc::clone(&store_ref.name),
            source,
        })
    };
    let sharing_matches = sharing_key(response_ref)? == sharing_key(conversation_ref)?;
    let response_table = response_ref.config.get("conversations_table");
    let conversation_table = conversation_ref.config.get("conversations_table");
    if sharing_matches && response_table == conversation_table {
        let Some(items_table) = conversation_ref.config.get("items_table").cloned() else {
            return Ok(None);
        };
        let mut combined = response_ref.config.clone();
        let Some(object) = combined.as_object_mut() else {
            return Ok(None);
        };
        object.insert("items_table".to_owned(), items_table);
        if let Some(objects) = sql_objects(&combined)
            && let Some((first, second)) = schema_name_collision(&objects)
        {
            return Err(namespace_collision_error(response_ref, first, second));
        }
        return Ok(Some(combined));
    }
    Ok(None)
}

/// Validate object-name compatibility across the final promoted store plan.
#[cfg(feature = "openai-conversations")]
#[expect(
    clippy::too_many_lines,
    reason = "namespace resolution and pairwise object validation form one pass"
)]
fn validate_sql_namespace_collisions(
    plans: &[ListenerStorePlan],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<(), ProvisionError> {
    let refs: Vec<&StoreRef> = plans.iter().flat_map(|plan| &plan.refs).collect();
    for (index, first) in refs.iter().enumerate() {
        let Some(factory) = factories.get(first.backend_id.as_ref()) else {
            continue;
        };
        let first_namespace = factory
            .namespace_key(&first.config)
            .map_err(|source| ProvisionError::Backend {
                name: Arc::clone(&first.name),
                source,
            })?;
        let Some(first_namespace) = first_namespace else {
            continue;
        };
        for second in refs.iter().skip(index + 1) {
            if first.backend_id != second.backend_id {
                continue;
            }
            let second_namespace = factory
                .namespace_key(&second.config)
                .map_err(|source| ProvisionError::Backend {
                    name: Arc::clone(&second.name),
                    source,
                })?;
            if second_namespace.as_ref() == Some(&first_namespace)
                && let Some((first_owner, second_owner)) = cross_store_schema_name_collision(first, second)
            {
                return Err(namespace_collision_error(first, first_owner, second_owner));
            }
        }
    }
    Ok(())
}

/// Location of one store reference within the server's listener plans.
#[cfg(feature = "openai-conversations")]
type StoreRefLocation = (usize, usize);

/// One compatible Responses/Conversations pairing and its full backend key.
#[cfg(feature = "openai-conversations")]
struct CombinedCandidate {
    /// Responses reference participating in the pairing.
    response: StoreRefLocation,
    /// Conversations reference participating in the pairing.
    conversation: StoreRefLocation,
    /// Factory id shared by both references.
    backend_id: Arc<str>,
    /// Factory-normalized key for the combined config.
    effective: EffectiveConfigKey,
    /// Complete table configuration accepted by the factory.
    config: serde_json::Value,
}

/// Find every store reference with one registry name.
#[cfg(feature = "openai-conversations")]
fn named_store_refs<'a>(plans: &'a [ListenerStorePlan], name: &'a str) -> Vec<(StoreRefLocation, &'a StoreRef)> {
    plans
        .iter()
        .enumerate()
        .flat_map(|(plan_index, plan)| {
            plan.refs.iter().enumerate().filter_map(move |(ref_index, store_ref)| {
                (store_ref.name.as_ref() == name).then_some(((plan_index, ref_index), store_ref))
            })
        })
        .collect()
}

/// Build all compatible combined configurations across the server plan.
#[cfg(feature = "openai-conversations")]
fn combined_candidates(
    plans: &[ListenerStorePlan],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<Vec<CombinedCandidate>, ProvisionError> {
    let response_refs = named_store_refs(plans, DEFAULT_STORE_NAME);
    let conversation_refs = named_store_refs(plans, CONVERSATIONS_STORE_NAME);
    let mut candidates = Vec::new();
    for &(response, response_ref) in &response_refs {
        for &(conversation, conversation_ref) in &conversation_refs {
            let Some(config) = combined_filter_config(response_ref, conversation_ref, factories)? else {
                continue;
            };
            let Some(factory) = factories.get(response_ref.backend_id.as_ref()) else {
                continue;
            };
            let effective = factory
                .effective_key(&config)
                .map_err(|source| ProvisionError::Backend {
                    name: Arc::clone(&response_ref.name),
                    source,
                })?;
            candidates.push(CombinedCandidate {
                response,
                conversation,
                backend_id: Arc::clone(&response_ref.backend_id),
                effective,
                config,
            });
        }
    }
    Ok(candidates)
}

/// Select the one combined config consistently implied for a reference.
#[cfg(feature = "openai-conversations")]
fn selected_candidate<'a>(
    candidates: &'a [CombinedCandidate],
    location: StoreRefLocation,
    name: &str,
) -> Option<&'a CombinedCandidate> {
    let matches_ref = |candidate: &CombinedCandidate| match name {
        DEFAULT_STORE_NAME => candidate.response == location,
        CONVERSATIONS_STORE_NAME => candidate.conversation == location,
        _ => false,
    };
    let mut eligible = candidates.iter().filter(|candidate| matches_ref(candidate));
    let first = eligible.next()?;
    eligible
        .all(|candidate| candidate.backend_id == first.backend_id && candidate.effective == first.effective)
        .then_some(first)
}

/// Return a candidate only when both participating references select its key.
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

/// Promote compatible Responses and Conversations references to one combined
/// backend configuration across every listener, so all matching registry
/// names share one backend and pool.
#[cfg(feature = "openai-conversations")]
fn coalesce_compatible_filter_refs(
    plans: &mut [ListenerStorePlan],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<HashSet<StoreRefLocation>, ProvisionError> {
    let candidates = combined_candidates(plans, factories)?;
    let mut promoted = HashSet::new();
    for (plan_index, plan) in plans.iter_mut().enumerate() {
        for (ref_index, store_ref) in plan.refs.iter_mut().enumerate() {
            if let Some(candidate) = mutually_selected_candidate(&candidates, (plan_index, ref_index), &store_ref.name)
            {
                store_ref.config.clone_from(&candidate.config);
                promoted.insert((plan_index, ref_index));
            }
        }
    }
    Ok(promoted)
}

/// Build a per-listener store plan for every listener whose chains configure a
/// response or conversations store. All reachable filters are retained here so
/// their effective configurations can be checked before duplicate registry
/// names are collapsed.
fn build_listener_store_plans(config: &Config) -> Vec<ListenerStorePlan> {
    let chains: HashMap<&str, &[FilterEntry]> = config
        .filter_chains
        .iter()
        .map(|c| (c.name.as_str(), c.filters.as_slice()))
        .collect();

    let mut plans = Vec::new();
    for listener in &config.listeners {
        let mut refs = Vec::new();
        refs.extend(
            find_listener_store_configs(listener, &chains, RESPONSE_STORE_FILTER_NAME)
                .into_iter()
                .filter_map(|filter_config| response_store_ref(&filter_config)),
        );
        #[cfg(feature = "openai-conversations")]
        refs.extend(
            find_listener_store_configs(listener, &chains, CONVERSATIONS_STORE_FILTER_NAME)
                .into_iter()
                .filter_map(|filter_config| conversations_store_ref(&filter_config)),
        );
        if !refs.is_empty() {
            plans.push(ListenerStorePlan {
                listener: listener.name.clone(),
                registry: StoreRegistry::new(),
                refs,
            });
        }
    }
    plans
}

/// Whether any listener reaches a persisted-state store filter.
pub(crate) fn config_uses_store(config: &Config) -> bool {
    !build_listener_store_plans(config).is_empty()
}

/// Listener names whose live pipelines can hold references to a store
/// generation built from `config`.
pub(crate) fn store_listener_names(config: &Config) -> Vec<String> {
    build_listener_store_plans(config)
        .into_iter()
        .map(|plan| plan.listener)
        .collect()
}

/// Reject one listener binding a registry name to different effective backend
/// configurations, then collapse repeated references to the same backend.
#[expect(
    clippy::too_many_lines,
    reason = "factory resolution, effective-key comparison, and deduplication are one validation pass"
)]
fn validate_and_deduplicate_refs(
    plans: &mut [ListenerStorePlan],
    factories: &[Arc<dyn StoreBackendFactory>],
    #[cfg_attr(
        not(feature = "openai-conversations"),
        expect(unused_variables, reason = "cache pinning is only needed for cross-filter promotion")
    )]
    cache: Option<&BackendCache>,
    #[cfg_attr(
        not(feature = "openai-conversations"),
        expect(
            unused_variables,
            reason = "active identity is only needed for cross-filter promotion"
        )
    )]
    active_plans: Option<&[ListenerStorePlan]>,
) -> Result<(), ProvisionError> {
    let factories: HashMap<&str, &dyn StoreBackendFactory> = factories
        .iter()
        .map(|factory| (factory.backend_id(), factory.as_ref()))
        .collect();

    #[cfg(feature = "openai-conversations")]
    {
        // Current filter compatibility takes precedence over reusing an older,
        // narrower backend key. Otherwise adding Conversations to an active
        // Responses-only configuration pins the raw Responses pool and leaves
        // the replacement generation split across two pools.
        let promoted = coalesce_compatible_filter_refs(plans, &factories)?;
        if let (Some(cache), Some(active_plans)) = (cache, active_plans) {
            preserve_active_store_configs(plans, active_plans, &factories, cache, &promoted)?;
        }
        validate_sql_namespace_collisions(plans, &factories)?;
    }

    for plan in plans {
        let mut bound: HashMap<Arc<str>, (Arc<str>, EffectiveConfigKey)> = HashMap::new();
        let mut unique = Vec::with_capacity(plan.refs.len());
        for store_ref in plan.refs.drain(..) {
            let factory =
                factories
                    .get(store_ref.backend_id.as_ref())
                    .ok_or_else(|| ProvisionError::UnknownBackend {
                        name: Arc::clone(&store_ref.name),
                        backend_id: Arc::clone(&store_ref.backend_id),
                    })?;
            let effective = factory
                .effective_key(&store_ref.config)
                .map_err(|source| ProvisionError::Backend {
                    name: Arc::clone(&store_ref.name),
                    source,
                })?;
            if let Some((backend_id, existing)) = bound.get(&store_ref.name) {
                if backend_id != &store_ref.backend_id || existing != &effective {
                    return Err(ProvisionError::Backend {
                        name: Arc::clone(&store_ref.name),
                        source: BackendError::Config(format!(
                            "listener '{}' configures conflicting stores for registry name '{}'",
                            plan.listener, store_ref.name
                        )),
                    });
                }
                continue;
            }
            bound.insert(
                Arc::clone(&store_ref.name),
                (Arc::clone(&store_ref.backend_id), effective),
            );
            unique.push(store_ref);
        }
        plan.refs = unique;
    }
    Ok(())
}

/// Map each plan to the [`ResponseStoreRegistry`] installed into its pipeline,
/// sharing the plan's backing storage so provisioning is observed there.
fn registries_map(plans: &[ListenerStorePlan]) -> HashMap<String, ResponseStoreRegistry> {
    plans
        .iter()
        .map(|p| (p.listener.clone(), ResponseStoreRegistry::from(p.registry.clone())))
        .collect()
}

/// The per-listener registries to install into pipelines, the serving-runtime
/// provisioner, its reload handle, and a readiness handle observers await.
pub type StoreWiring = (
    HashMap<String, ResponseStoreRegistry>,
    StoreProvisionService,
    StoreReloadHandle,
    StoreReadinessHandle,
);

/// Build the per-listener store registries to install into pipelines and, when
/// any store is configured, the serving-runtime provisioner and its readiness
/// handle.
///
/// Store configuration is validated eagerly so a malformed or unknown-backend
/// config fails startup rather than at first traffic. Connectivity surfaces
/// later, when the background service opens the pools on the serving runtime.
/// The readiness handle reaches `Ready` once every pool is open.
///
/// # Errors
///
/// Returns [`ProvisionError`] when a configured store names an unknown backend
/// or its configuration is rejected.
pub fn build_store_wiring(config: &Config) -> Result<StoreWiring, ProvisionError> {
    let mut plans = build_listener_store_plans(config);
    let factories = store_backend_factories();
    validate_and_deduplicate_refs(&mut plans, &factories, None, None)?;
    let registries = registries_map(&plans);
    let cache = Arc::new(BackendCache::new(factories.clone()));
    let refs: Vec<StoreRef> = plans.iter().flat_map(|p| p.refs.iter().cloned()).collect();
    cache.validate(&refs)?;
    let initial_readiness = if plans.is_empty() {
        StoreReadiness::Ready
    } else {
        StoreReadiness::Pending
    };
    let (readiness, rx) = watch::channel(initial_readiness);
    let (commands, command_rx) = mpsc::unbounded_channel();
    Ok((
        registries,
        StoreProvisionService {
            cache,
            factories,
            plans,
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        },
        StoreReloadHandle { commands },
        StoreReadinessHandle { rx },
    ))
}

/// A store generation provisioned on the serving runtime but not yet installed
/// into the live pipelines.
pub(crate) struct PreparedStoreReload {
    /// Monotonic generation identifier owned by the provisioner.
    generation: u64,
    /// Ready registries to attach while building the replacement pipelines.
    pub(crate) registries: crate::StoreRegistries,
}

/// Cloneable command handle used by the config-watcher thread.
///
/// Commands cross onto the serving runtime before touching SQL pools. The
/// watcher waits synchronously because it must not swap a pipeline until its
/// replacement generation is fully provisioned.
#[derive(Clone)]
pub struct StoreReloadHandle {
    /// Commands consumed by [`StoreProvisionService`].
    commands: mpsc::UnboundedSender<StoreCommand>,
}

impl Default for StoreReloadHandle {
    fn default() -> Self {
        let (commands, _receiver) = mpsc::unbounded_channel();
        Self { commands }
    }
}

impl StoreReloadHandle {
    /// Provision and validate a replacement store generation.
    pub(crate) fn prepare(&self, config: &Config) -> Result<PreparedStoreReload, String> {
        let (reply, response) = std_mpsc::sync_channel(1);
        self.commands
            .send(StoreCommand::Prepare {
                config: Box::new(config.clone()),
                reply,
            })
            .map_err(|_send_error| "store provisioner stopped before reload preparation".to_owned())?;
        response
            .recv()
            .map_err(|_recv_error| "store provisioner dropped the reload preparation response".to_owned())?
    }

    /// Promote a prepared generation immediately before the replacement
    /// pipelines are published.
    pub(crate) fn commit(
        &self,
        prepared: PreparedStoreReload,
        old_pipelines: Vec<Weak<FilterPipeline>>,
    ) -> Result<(), String> {
        let PreparedStoreReload { generation, registries } = prepared;
        drop(registries);
        let (reply, response) = std_mpsc::sync_channel(1);
        self.commands
            .send(StoreCommand::Commit {
                generation,
                old_pipelines,
                reply,
            })
            .map_err(|_send_error| "store provisioner stopped before reload commit".to_owned())?;
        response
            .recv()
            .map_err(|_recv_error| "store provisioner dropped the reload commit response".to_owned())?
    }

    /// Release a prepared generation whose pipeline build failed.
    pub(crate) fn abort(&self, prepared: PreparedStoreReload) -> Result<(), String> {
        let PreparedStoreReload { generation, registries } = prepared;
        drop(registries);
        let (reply, response) = std_mpsc::sync_channel(1);
        self.commands
            .send(StoreCommand::Abort { generation, reply })
            .map_err(|_send_error| "store provisioner stopped before reload abort".to_owned())?;
        response
            .recv()
            .map_err(|_recv_error| "store provisioner dropped the reload abort response".to_owned())?
    }
}

/// Commands sent from the watcher runtime to the serving runtime.
enum StoreCommand {
    /// Build a candidate generation without disturbing the active one.
    Prepare {
        /// Reloaded proxy configuration.
        config: Box<Config>,
        /// Completion sent after provisioning succeeds or fails.
        reply: std_mpsc::SyncSender<Result<PreparedStoreReload, String>>,
    },
    /// Make a prepared generation active after pipeline swap.
    Commit {
        /// Prepared generation identifier.
        generation: u64,
        /// Weak observers of previous pipelines whose request-held `Arc`s must
        /// drain. Weak references cannot keep each other alive across reloads.
        old_pipelines: Vec<Weak<FilterPipeline>>,
        /// Completion acknowledgement.
        reply: std_mpsc::SyncSender<Result<(), String>>,
    },
    /// Discard a prepared generation after pipeline construction failed.
    Abort {
        /// Prepared generation identifier.
        generation: u64,
        /// Completion acknowledgement.
        reply: std_mpsc::SyncSender<Result<(), String>>,
    },
}

/// Provisions response-store backends on the serving runtime and owns every
/// active or pending generation lease.
pub struct StoreProvisionService {
    /// Process-wide backend cache built from the compiled-in factories.
    cache: Arc<BackendCache>,
    /// Factories used to validate and de-duplicate reload plans.
    factories: Vec<Arc<dyn StoreBackendFactory>>,
    /// Per-listener registries and references to provision.
    plans: Vec<ListenerStorePlan>,
    /// Publishes readiness transitions to observers.
    readiness: watch::Sender<StoreReadiness>,
    /// Single receiver taken when the background service starts.
    commands: AsyncMutex<Option<mpsc::UnboundedReceiver<StoreCommand>>>,
}

impl StoreProvisionService {
    /// Validate one config and construct its fresh per-listener registries.
    fn prepare_plans(
        &self,
        config: &Config,
        active_plans: &[ListenerStorePlan],
    ) -> Result<Vec<ListenerStorePlan>, ProvisionError> {
        let mut plans = build_listener_store_plans(config);
        validate_and_deduplicate_refs(&mut plans, &self.factories, Some(&self.cache), Some(active_plans))?;
        let refs: Vec<StoreRef> = plans.iter().flat_map(|p| p.refs.iter().cloned()).collect();
        self.cache.validate(&refs)?;
        Ok(plans)
    }

    /// Provision all listeners concurrently as one atomic generation.
    ///
    /// [`BackendCache`] already applies the bounded transient retry budget. Any
    /// error returned here, including [`BackendError::Unavailable`], is terminal
    /// for this generation and must not be retried by the server.
    #[expect(
        clippy::too_many_lines,
        reason = "concurrent provisioning, rollback, and atomic publication form one operation"
    )]
    async fn provision_all(&self, plans: &[ListenerStorePlan]) -> Result<Vec<BackendLease>, ProvisionError> {
        let mut workers = plans
            .iter()
            .map(|plan| async move {
                let result = self.cache.provision_into(&plan.refs, &plan.registry).await;
                (plan, result)
            })
            .collect::<FuturesUnordered<_>>();
        let mut leases = Vec::with_capacity(plans.len());
        let mut failure = None;

        while let Some((plan, result)) = workers.next().await {
            match result {
                Ok(lease) => {
                    info!(listener = %plan.listener, "persisted-state stores provisioned");
                    leases.push(lease);
                },
                Err(e) => {
                    error!(
                        listener = %plan.listener,
                        error = %e,
                        "response store provisioning failed permanently; not retrying",
                    );
                    if failure.is_none() {
                        failure = Some(e);
                    }
                },
            }
        }

        if let Some(error) = failure {
            release_leases(leases).await;
            return Err(error);
        }

        for plan in plans {
            plan.registry.mark_ready();
        }
        Ok(leases)
    }

    /// Own the initial startup gate and every subsequent reload generation.
    ///
    /// Pingora supplies `ready_notifier` in production. Holding it until all
    /// pools are open prevents the provisioner itself from becoming ready, and
    /// an initial terminal failure rejects process startup. Tests call
    /// [`BackgroundService::start`] without a notifier so they can observe the
    /// terminal state without exiting their test process.
    #[expect(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        reason = "startup and reload commands share one serving-runtime lease owner"
    )]
    async fn run(&self, mut shutdown: ShutdownWatch, ready_notifier: Option<ServiceReadyNotifier>) {
        let Some(mut commands) = self.commands.lock().await.take() else {
            error!("store provisioner command receiver was already taken");
            return;
        };
        let initial_provisioning = self.provision_all(&self.plans);
        tokio::pin!(initial_provisioning);
        let initial_result = tokio::select! {
            result = &mut initial_provisioning => result,
            _ = shutdown.changed() => return,
        };
        let mut active_leases = match initial_result {
            Ok(leases) => {
                let _sent = self.readiness.send(StoreReadiness::Ready);
                if !self.plans.is_empty() {
                    info!("all persisted-state stores provisioned");
                }
                if let Some(notifier) = ready_notifier {
                    notifier.notify_ready();
                }
                leases
            },
            Err(initial_error) => {
                let _sent = self.readiness.send(StoreReadiness::Failed);
                if ready_notifier.is_some() {
                    crate::server::fatal(&format_args!(
                        "initial response-store provisioning failed: {initial_error}"
                    ));
                }
                return;
            },
        };
        let mut active_plans = self.plans.clone();
        let mut pending: HashMap<u64, (Vec<BackendLease>, Vec<ListenerStorePlan>)> = HashMap::new();
        let mut next_generation = 1_u64;

        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                command = commands.recv() => {
                    let Some(command) = command else { break };
                    match command {
                        StoreCommand::Prepare { config, reply } => {
                            let result = match self.prepare_plans(&config, &active_plans) {
                                Ok(plans) => {
                                    let provisioning = self.provision_all(&plans);
                                    tokio::pin!(provisioning);
                                    let provisioned = tokio::select! {
                                        result = &mut provisioning => result,
                                        _ = shutdown.changed() => {
                                            let _sent = reply.send(Err(
                                                "store provisioner stopped during reload preparation".to_owned(),
                                            ));
                                            break;
                                        },
                                    };
                                    match provisioned {
                                        Ok(leases) => {
                                            let generation = next_generation;
                                            next_generation = next_generation.saturating_add(1);
                                            let registries = registries_map(&plans);
                                            pending.insert(generation, (leases, plans.clone()));
                                            Ok(PreparedStoreReload { generation, registries })
                                        },
                                        Err(e) => Err(e.to_string()),
                                    }
                                },
                                Err(e) => Err(e.to_string()),
                            };
                            let _sent = reply.send(result);
                        },
                        StoreCommand::Commit { generation, old_pipelines, reply } => {
                            let result = if let Some((new_leases, new_plans)) = pending.remove(&generation) {
                                let old_leases = std::mem::replace(&mut active_leases, new_leases);
                                active_plans = new_plans;
                                tokio::spawn(release_after_drain(old_pipelines, old_leases));
                                let _sent = self.readiness.send(StoreReadiness::Ready);
                                Ok(())
                            } else {
                                Err(format!("unknown prepared store generation {generation}"))
                            };
                            let _sent = reply.send(result);
                        },
                        StoreCommand::Abort { generation, reply } => {
                            let result = if let Some((leases, _plans)) = pending.remove(&generation) {
                                release_leases(leases).await;
                                Ok(())
                            } else {
                                Err(format!("unknown prepared store generation {generation}"))
                            };
                            let _sent = reply.send(result);
                        },
                    }
                },
            }
        }

        // Pending generations were never attached to request paths and can be
        // retired immediately. Active leases intentionally remain held through
        // Pingora's shutdown drain and disappear with the serving runtime.
        for (_, (leases, _plans)) in pending {
            release_leases(leases).await;
        }
        drop(active_leases);
    }
}

#[async_trait]
impl BackgroundService for StoreProvisionService {
    async fn start_with_ready_notifier(&self, shutdown: ShutdownWatch, ready_notifier: ServiceReadyNotifier) {
        self.run(shutdown, Some(ready_notifier)).await;
    }

    async fn start(&self, shutdown: ShutdownWatch) {
        self.run(shutdown, None).await;
    }
}

/// Release every lease in a generation on the serving runtime.
async fn release_leases(leases: Vec<BackendLease>) {
    for lease in leases {
        lease.release().await;
    }
}

/// Retain the old backend generation until every pipeline owner and request-held
/// `Arc` has drained, then retire backends no newer generation references.
async fn release_after_drain(old_pipelines: Vec<Weak<FilterPipeline>>, leases: Vec<BackendLease>) {
    while old_pipelines.iter().any(|pipeline| pipeline.strong_count() > 0) {
        tokio::time::sleep(PIPELINE_DRAIN_POLL).await;
    }
    release_leases(leases).await;
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::too_many_lines, reason = "tests")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use praxis_ai_store::{
        BackendError, EffectiveConfigKey, ProvisionedBackend, RetireBackend, StoreBackendFactory, memory::InMemoryStore,
    };
    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    use praxis_ai_store::{ConversationRecord, StateOwner};
    use praxis_filter::FilterRegistry;
    use serde_json::json;

    use super::*;

    struct CountingRetire(Arc<AtomicUsize>);

    #[async_trait]
    impl RetireBackend for CountingRetire {
        async fn retire(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct FakeFactory {
        retires: Arc<AtomicUsize>,
    }

    #[cfg(feature = "openai-conversations")]
    struct TableAwareFactory {
        backend_id: &'static str,
        builds: Arc<AtomicUsize>,
        retires: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StoreBackendFactory for FakeFactory {
        fn backend_id(&self) -> &str {
            "fake"
        }

        fn effective_key(&self, _config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            Ok(EffectiveConfigKey::new("fake"))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire(Arc::clone(&self.retires))),
            })
        }
    }

    #[cfg(feature = "openai-conversations")]
    #[async_trait]
    impl StoreBackendFactory for TableAwareFactory {
        fn backend_id(&self) -> &str {
            self.backend_id
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let field = |name| config.get(name).map_or_else(String::new, serde_json::Value::to_string);
            Ok(EffectiveConfigKey::new(format!(
                "{}|{}|{}|{}|{}",
                field("database_url"),
                field("responses_table"),
                field("conversations_table"),
                field("items_table"),
                field("pool"),
            )))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire(Arc::clone(&self.retires))),
            })
        }
    }

    struct UnavailableFactory {
        builds: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StoreBackendFactory for UnavailableFactory {
        fn backend_id(&self) -> &str {
            "unavailable"
        }

        fn effective_key(&self, _config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            Ok(EffectiveConfigKey::new("unavailable"))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Err(BackendError::Unavailable("test backend is down".to_owned()))
        }
    }

    struct RecoveringFactory {
        builds: Arc<AtomicUsize>,
        retires: Arc<AtomicUsize>,
    }

    struct HangingFactory {
        entered: Arc<tokio::sync::Notify>,
    }

    struct ReloadFactory {
        builds: Arc<AtomicUsize>,
        retires: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StoreBackendFactory for HangingFactory {
        fn backend_id(&self) -> &str {
            "hanging"
        }

        fn effective_key(&self, _config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            Ok(EffectiveConfigKey::new("hanging"))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.entered.notify_one();
            std::future::pending().await
        }
    }

    #[async_trait]
    impl StoreBackendFactory for ReloadFactory {
        fn backend_id(&self) -> &str {
            "reload"
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let url = config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| BackendError::Config("missing database_url".to_owned()))?;
            Ok(EffectiveConfigKey::new(url))
        }

        async fn build(&self, config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            if config.get("database_url").and_then(serde_json::Value::as_str) == Some("failed") {
                return Err(BackendError::Unavailable("replacement backend is down".to_owned()));
            }
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire(Arc::clone(&self.retires))),
            })
        }
    }

    #[async_trait]
    impl StoreBackendFactory for RecoveringFactory {
        fn backend_id(&self) -> &str {
            "recovering"
        }

        fn effective_key(&self, _config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            Ok(EffectiveConfigKey::new("recovering"))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            let attempt = self.builds.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 {
                return Err(BackendError::Transient("test backend has not recovered yet".to_owned()));
            }
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire(Arc::clone(&self.retires))),
            })
        }
    }

    fn conditional_store_config(backend: &str, first_url: &str, second_url: &str) -> Config {
        Config::from_yaml(&format!(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: router
        branch_chains:
          - name: first
            chains:
              - name: first-store
                filters:
                  - filter: openai_response_store
                    backend: {backend}
                    database_url: "{first_url}"
                    responses_table: responses
                    conversations_table: conversations
          - name: second
            chains:
              - name: second-store
                filters:
                  - filter: openai_response_store
                    backend: {backend}
                    database_url: "{second_url}"
                    responses_table: responses
                    conversations_table: conversations
"#
        ))
        .expect("conditional store config")
    }

    fn single_store_config(backend: &str, database_url: &str) -> Config {
        Config::from_yaml(&format!(
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
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
"#
        ))
        .expect("single store config")
    }

    #[cfg(feature = "store-sqlite")]
    fn sqlite_store_config_with_event_bounds() -> Config {
        Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_response_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
        max_event_count: 5000
        max_event_bytes: 8388608
"#,
        )
        .expect("sqlite store config with explicit replay bounds")
    }

    #[cfg(feature = "store-postgres")]
    fn postgres_store_config_with_event_bounds() -> Config {
        Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_response_store
        backend: postgres
        database_url: "postgresql://user:password@8.8.8.8/store"
        responses_table: responses
        conversations_table: conversations
        max_event_count: 5000
        max_event_bytes: 8388608
"#,
        )
        .expect("postgres store config with explicit replay bounds")
    }

    #[cfg(feature = "openai-conversations")]
    fn responses_and_conversations_config_for(backend: &str, database_url: &str) -> Config {
        Config::from_yaml(&format!(
            r#"
listeners:
  - name: combined
    address: "127.0.0.1:8080"
    filter_chains: [responses, conversations]
  - name: responses-only
    address: "127.0.0.1:8081"
    filter_chains: [responses]
filter_chains:
  - name: responses
    filters:
      - filter: openai_response_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        pool:
          max_connections: 4
  - name: conversations
    filters:
      - filter: openai_conversations
        backend: {backend}
        database_url: "{database_url}"
        conversations_table: conversations
        items_table: items
        pool:
          max_connections: 4
"#,
        ))
        .expect("combined Responses and Conversations config")
    }

    #[cfg(feature = "openai-conversations")]
    fn responses_and_conversations_config() -> Config {
        #[cfg(feature = "store-sqlite")]
        let (backend, database_url) = ("sqlite", "sqlite::memory:");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend, database_url) = ("postgres", "postgresql://user:password@8.8.8.8/store");
        responses_and_conversations_config_for(backend, database_url)
    }

    #[cfg(feature = "openai-conversations")]
    fn ambiguous_combined_config() -> Config {
        #[cfg(feature = "store-sqlite")]
        let (backend, database_url) = ("sqlite", "sqlite::memory:");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend, database_url) = ("postgres", "postgresql://user:password@8.8.8.8/store");
        Config::from_yaml(&format!(
            r#"
listeners:
  - name: first
    address: "127.0.0.1:8080"
    filter_chains: [responses-a, conversations]
  - name: second
    address: "127.0.0.1:8081"
    filter_chains: [responses-b, conversations]
filter_chains:
  - name: responses-a
    filters:
      - filter: openai_response_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations
  - name: responses-b
    filters:
      - filter: openai_response_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses_b
        conversations_table: conversations
  - name: conversations
    filters:
      - filter: openai_conversations
        backend: {backend}
        database_url: "{database_url}"
        conversations_table: conversations
        items_table: items
"#,
        ))
        .expect("ambiguous combined store config")
    }

    #[test]
    fn store_listener_names_exclude_stateless_listeners() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: stateful
    address: "127.0.0.1:8080"
    filter_chains: [store]
  - name: stateless
    address: "127.0.0.1:8081"
    filter_chains: [plain]
filter_chains:
  - name: store
    filters:
      - filter: openai_response_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
  - name: plain
    filters: []
"#,
        )
        .expect("mixed listener config");

        assert_eq!(store_listener_names(&config), ["stateful"]);
    }

    #[tokio::test]
    async fn wait_ready_returns_after_terminal_failure() {
        let (readiness, rx) = watch::channel(StoreReadiness::Pending);
        let mut handle = StoreReadinessHandle { rx };

        readiness.send(StoreReadiness::Failed).expect("readiness observer");
        tokio::time::timeout(Duration::from_millis(100), handle.wait_ready())
            .await
            .expect("terminal provisioning failure must unblock readiness waiters");
        assert_eq!(handle.current(), StoreReadiness::Failed);
    }

    #[test]
    fn conflicting_conditional_store_configs_are_rejected() {
        #[cfg(feature = "store-sqlite")]
        let config = conditional_store_config("sqlite", "sqlite:///first.db", "sqlite:///second.db");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let config = conditional_store_config(
            "postgres",
            "postgresql://user:password@8.8.8.8/store",
            "postgresql://user:password@1.1.1.1/store",
        );
        let result = build_store_wiring(&config);

        assert!(
            matches!(
                result,
                Err(ProvisionError::Backend {
                    source: BackendError::Config(message),
                    ..
                }) if message.contains("conflicting stores")
            ),
            "one registry name must not silently select the first branch's backend"
        );
    }

    #[test]
    fn identical_conditional_store_configs_share_one_reference() {
        #[cfg(feature = "store-sqlite")]
        let config = conditional_store_config("sqlite", "sqlite:///shared.db", "sqlite:///shared.db");
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let config = conditional_store_config(
            "postgres",
            "postgresql://user:password@8.8.8.8/store",
            "postgresql://user:password@8.8.8.8/store",
        );
        let (registries, provisioner, _reload, _readiness) = build_store_wiring(&config).expect("identical stores");

        assert_eq!(registries.len(), 1);
        assert_eq!(provisioner.plans.len(), 1);
        assert_eq!(provisioner.plans.first().expect("listener plan").refs.len(), 1);
    }

    #[cfg(feature = "store-sqlite")]
    #[test]
    fn explicit_replay_event_bounds_do_not_block_sqlite_provisioning() {
        // The replay-log bounds are filter-only knobs; the SQLite factory config
        // rejects unknown fields. Provisioning must strip them so an operator that
        // sets them explicitly in YAML still boots, and the backend never sees them.
        let config = sqlite_store_config_with_event_bounds();
        let (_registries, provisioner, _reload, _readiness) =
            build_store_wiring(&config).expect("explicit replay bounds must not reach the SQLite factory");
        let store_ref = provisioner
            .plans
            .first()
            .and_then(|plan| plan.refs.first())
            .expect("one provisioned response store");
        assert!(
            store_ref.config.get("max_event_count").is_none(),
            "filter-only max_event_count must be stripped before dispatch to the backend factory"
        );
        assert!(
            store_ref.config.get("max_event_bytes").is_none(),
            "filter-only max_event_bytes must be stripped before dispatch to the backend factory"
        );
    }

    #[cfg(feature = "store-postgres")]
    #[test]
    fn explicit_replay_event_bounds_do_not_block_postgres_provisioning() {
        // Same filter-only strip must apply to the Postgres factory, whose config
        // also rejects unknown fields; `response_store_ref` drops the bounds for
        // every backend before dispatch.
        let config = postgres_store_config_with_event_bounds();
        let result = build_store_wiring(&config);
        assert!(
            result.is_ok(),
            "explicit replay bounds must be stripped for the Postgres factory too: {:?}",
            result.err()
        );
    }

    #[cfg(feature = "openai-conversations")]
    #[tokio::test]
    async fn responses_and_conversations_share_one_backend_and_pool() {
        #[cfg(feature = "store-sqlite")]
        let backend_id = "sqlite";
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let backend_id = "postgres";
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareFactory {
            backend_id,
            builds: Arc::clone(&builds),
            retires: Arc::clone(&retires),
        });
        let config = responses_and_conversations_config();
        let (_registries, production_provisioner, _reload, _readiness) =
            build_store_wiring(&config).expect("production factories should accept the shared config");
        let production_plan = production_provisioner.plans.first().expect("production listener plan");
        assert_eq!(production_plan.refs.len(), 2);
        assert_eq!(
            production_plan.refs.first().map(|r| &r.config),
            production_plan.refs.get(1).map(|r| &r.config),
            "real backend factories must receive one combined config"
        );

        let mut plans = build_listener_store_plans(&config);
        validate_and_deduplicate_refs(&mut plans, &[Arc::clone(&factory)], None, None)
            .expect("compatible filter stores");
        let plan = plans.first().expect("listener plan");
        let responses_only = plans.get(1).expect("Responses-only listener plan");
        assert_eq!(plan.refs.len(), 2, "both registry names remain addressable");
        assert_eq!(responses_only.refs.len(), 1);
        assert_eq!(
            plan.refs.first().map(|r| &r.config),
            plan.refs.get(1).map(|r| &r.config)
        );
        assert_eq!(
            plan.refs.first().map(|r| &r.config),
            responses_only.refs.first().map(|r| &r.config),
            "matching listeners must retain one process-wide backend key"
        );
        assert_eq!(
            plan.refs.first().and_then(|r| r.config.get("responses_table")),
            Some(&json!("responses"))
        );
        assert_eq!(
            plan.refs.first().and_then(|r| r.config.get("items_table")),
            Some(&json!("items"))
        );

        let cache = BackendCache::new(vec![factory]);
        let combined_registry = StoreRegistry::new();
        let combined_lease = cache
            .provision_into(&plan.refs, &combined_registry)
            .await
            .expect("combined backend should provision");
        let responses_registry = StoreRegistry::new();
        let responses_lease = cache
            .provision_into(&responses_only.refs, &responses_registry)
            .await
            .expect("Responses-only listener should reuse the backend");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "one combined config must open one pool"
        );
        assert!(combined_registry.contains(DEFAULT_STORE_NAME));
        assert!(combined_registry.contains(CONVERSATIONS_STORE_NAME));
        assert!(responses_registry.contains(DEFAULT_STORE_NAME));

        combined_lease.release().await;
        assert_eq!(retires.load(Ordering::SeqCst), 0);
        responses_lease.release().await;
        assert_eq!(retires.load(Ordering::SeqCst), 1);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[tokio::test]
    async fn in_memory_sqlite_ignores_pool_overrides_when_sharing_state() {
        let mut config = responses_and_conversations_config_for("sqlite", "sqlite::memory:");
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        conversations
            .config
            .as_mapping_mut()
            .expect("Conversations config")
            .insert(
                serde_yaml::Value::String("pool".to_owned()),
                serde_yaml::to_value(json!({"max_connections": 5})).expect("pool config"),
            );

        let (_registries, provisioner, _reload, _readiness) =
            build_store_wiring(&config).expect("ignored in-memory pool settings must coalesce");
        let plan = provisioner.plans.first().expect("combined listener plan");
        assert_eq!(plan.refs.len(), 2);
        assert_eq!(
            plan.refs.first().map(|store_ref| &store_ref.config),
            plan.refs.get(1).map(|store_ref| &store_ref.config),
            "both filter names must resolve to one combined backend"
        );

        let lease = provisioner
            .cache
            .provision_into(&plan.refs, &plan.registry)
            .await
            .expect("combined in-memory backend");
        let owner = StateOwner::from_trusted_parts("tenant", "issuer", "subject").expect("owner");
        let conversations_store = plan
            .registry
            .get_scoped(CONVERSATIONS_STORE_NAME, &owner)
            .expect("Conversations store");
        conversations_store
            .upsert_conversation(&ConversationRecord {
                conversation_id: "conv_shared".to_owned(),
                owner: owner.clone(),
                created_at: 1,
                metadata: json!({}),
                messages: json!([]),
            })
            .await
            .expect("create conversation");
        let responses_store = plan
            .registry
            .get_scoped(DEFAULT_STORE_NAME, &owner)
            .expect("Responses store");
        assert!(
            responses_store
                .get_conversation("conv_shared")
                .await
                .expect("read conversation through Responses")
                .is_some(),
            "Responses must observe Conversations state"
        );
        lease.release().await;
    }

    #[cfg(feature = "openai-conversations")]
    #[test]
    fn ambiguous_table_sets_preserve_shared_conversations_identity() {
        #[cfg(feature = "store-sqlite")]
        let backend_id = "sqlite";
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let backend_id = "postgres";
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareFactory {
            backend_id,
            builds: Arc::new(AtomicUsize::new(0)),
            retires: Arc::new(AtomicUsize::new(0)),
        });
        let mut plans = build_listener_store_plans(&ambiguous_combined_config());

        validate_and_deduplicate_refs(&mut plans, &[factory], None, None).expect("valid ambiguous stores");

        let first = plans.first().expect("first listener");
        let second = plans.get(1).expect("second listener");
        assert_eq!(
            first.refs.get(1).map(|r| &r.config),
            second.refs.get(1).map(|r| &r.config)
        );
        assert_ne!(
            first.refs.first().map(|r| &r.config),
            first.refs.get(1).map(|r| &r.config)
        );
        assert_ne!(
            second.refs.first().map(|r| &r.config),
            second.refs.get(1).map(|r| &r.config)
        );
    }

    #[cfg(feature = "openai-conversations")]
    fn assert_sql_namespace_collisions_are_rejected(mut config: Config) {
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        let items_table = conversations
            .config
            .as_mapping_mut()
            .and_then(|mapping| mapping.get_mut(serde_yaml::Value::String("items_table".to_owned())))
            .expect("items table config");
        *items_table = serde_yaml::Value::String("responses".to_owned());

        let error = build_store_wiring(&config)
            .err()
            .expect("cross-filter table collision must fail startup");
        let message = error.to_string();
        assert!(
            message.contains("same SQL namespace"),
            "actionable namespace error: {message}"
        );
        assert!(
            message.contains("responses_table collides with items_table"),
            "collision owners: {message}"
        );

        let response_filter = config
            .filter_chains
            .first()
            .and_then(|chain| chain.filters.first())
            .expect("Responses filter");
        let backend = response_filter
            .config
            .get("backend")
            .and_then(serde_yaml::Value::as_str)
            .expect("backend")
            .to_owned();
        let database_url = response_filter
            .config
            .get("database_url")
            .and_then(serde_yaml::Value::as_str)
            .expect("database URL")
            .to_owned();
        let mut config = responses_and_conversations_config_for(&backend, &database_url);
        let responses = config
            .filter_chains
            .first_mut()
            .and_then(|chain| chain.filters.first_mut())
            .expect("Responses filter");
        let responses_table = responses
            .config
            .as_mapping_mut()
            .and_then(|mapping| mapping.get_mut(serde_yaml::Value::String("responses_table".to_owned())))
            .expect("responses table config");
        *responses_table = serde_yaml::Value::String("idx_items_position".to_owned());

        let error = build_store_wiring(&config)
            .err()
            .expect("generated-index collision must fail startup");
        let message = error.to_string();
        assert!(
            message.contains("same SQL namespace"),
            "actionable namespace error: {message}"
        );
        assert!(
            message.contains("responses_table collides with items position index"),
            "generated-index owners: {message}"
        );
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn coalesced_in_memory_sqlite_namespace_collisions_are_rejected() {
        let config = responses_and_conversations_config_for("sqlite", "sqlite::memory:");

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn event_log_table_participates_in_namespace_collision_checks() {
        // The response store derives an event-log table `<responses>_events`. A
        // conversations items_table that reuses that derived name must be caught
        // by the pre-provisioning collision check, not left to fail later when
        // the second CREATE TABLE runs against an incompatible existing table.
        let mut config = responses_and_conversations_config_for("sqlite", "sqlite::memory:");
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        let items_table = conversations
            .config
            .as_mapping_mut()
            .and_then(|mapping| mapping.get_mut(serde_yaml::Value::String("items_table".to_owned())))
            .expect("items table config");
        *items_table = serde_yaml::Value::String("responses_events".to_owned());

        let error = build_store_wiring(&config)
            .err()
            .expect("event-log table collision must fail startup");
        let message = error.to_string();
        assert!(
            message.contains("same SQL namespace"),
            "actionable namespace error: {message}"
        );
        assert!(
            message.contains("responses event-log table"),
            "event-log owner in collision: {message}"
        );
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn coalesced_schema_ignores_discarded_conversations_placeholder_objects() {
        let mut config = responses_and_conversations_config_for("sqlite", "sqlite::memory:");
        let responses = config
            .filter_chains
            .first_mut()
            .and_then(|chain| chain.filters.first_mut())
            .expect("Responses filter");
        responses.config.as_mapping_mut().expect("Responses config").insert(
            serde_yaml::Value::String("responses_table".to_owned()),
            serde_yaml::Value::String("conversations_unused_responses_pending_approvals".to_owned()),
        );

        build_store_wiring(&config).expect("discarded placeholder objects cannot collide with the combined schema");
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn final_plan_validation_ignores_promoted_placeholder_objects() {
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: combined
    address: "127.0.0.1:8080"
    filter_chains: [responses, conversations]
  - name: other
    address: "127.0.0.1:8081"
    filter_chains: [other-responses]
filter_chains:
  - name: responses
    filters:
      - filter: openai_response_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
  - name: conversations
    filters:
      - filter: openai_conversations
        backend: sqlite
        database_url: "{database_url}"
        conversations_table: conversations
        items_table: items
  - name: other-responses
    filters:
      - filter: openai_response_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: conversations_unused_responses_pending_approvals
        conversations_table: other_conversations
"#,
        ))
        .expect("combined and independent Responses config");

        build_store_wiring(&config).expect("only objects in the final promoted plan participate in collision checks");
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn private_memory_sqlite_pool_overrides_do_not_hide_internal_collisions() {
        for database_url in [
            "sqlite://file::memory:",
            "sqlite://%3Amemory%3A",
            "sqlite://private?mode=memory&cache=private",
        ] {
            let mut config = responses_and_conversations_config_for("sqlite", database_url);
            let conversations = config
                .filter_chains
                .get_mut(1)
                .and_then(|chain| chain.filters.first_mut())
                .expect("Conversations filter");
            conversations
                .config
                .as_mapping_mut()
                .expect("Conversations config")
                .insert(
                    serde_yaml::Value::String("pool".to_owned()),
                    serde_yaml::to_value(json!({"max_connections": 5})).expect("pool config"),
                );

            assert_sql_namespace_collisions_are_rejected(config);
        }
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn file_backed_sqlite_namespace_collisions_with_mismatched_pool_are_rejected() {
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let mut config = responses_and_conversations_config_for("sqlite", &database_url);
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        conversations
            .config
            .as_mapping_mut()
            .expect("Conversations config")
            .extend([
                (
                    serde_yaml::Value::String("conversations_table".to_owned()),
                    serde_yaml::Value::String("other_conversations".to_owned()),
                ),
                (
                    serde_yaml::Value::String("pool".to_owned()),
                    serde_yaml::to_value(json!({"max_connections": 5})).expect("pool config"),
                ),
            ]);

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn absolute_memdb_namespace_collisions_with_mismatched_pool_are_rejected() {
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?vfs=memdb", directory.path().join("shared").display());
        let mut config = responses_and_conversations_config_for("sqlite", &database_url);
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        conversations
            .config
            .as_mapping_mut()
            .expect("Conversations config")
            .insert(
                serde_yaml::Value::String("pool".to_owned()),
                serde_yaml::to_value(json!({"max_connections": 5})).expect("pool config"),
            );

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[test]
    fn file_backed_sqlite_namespace_collisions_with_mismatched_compression_are_rejected() {
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let mut config = responses_and_conversations_config_for("sqlite", &database_url);
        let responses = config
            .filter_chains
            .first_mut()
            .and_then(|chain| chain.filters.first_mut())
            .expect("Responses filter");
        responses.config.as_mapping_mut().expect("Responses config").insert(
            serde_yaml::Value::String("compression".to_owned()),
            serde_yaml::to_value(json!({"algorithm": "zstd"})).expect("compression config"),
        );

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-postgres"))]
    #[test]
    fn postgres_namespace_collisions_are_rejected() {
        let config = responses_and_conversations_config_for("postgres", "postgresql://user:password@8.8.8.8/store");

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-postgres"))]
    #[test]
    fn postgres_namespace_collisions_across_roles_are_rejected() {
        let mut config =
            responses_and_conversations_config_for("postgres", "postgresql://first:password@8.8.8.8/store");
        let conversations = config
            .filter_chains
            .get_mut(1)
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        conversations
            .config
            .as_mapping_mut()
            .expect("Conversations config")
            .insert(
                serde_yaml::Value::String("database_url".to_owned()),
                serde_yaml::Value::String("postgresql://second:password@8.8.8.8/store".to_owned()),
            );

        assert_sql_namespace_collisions_are_rejected(config);
    }

    #[cfg(feature = "openai-conversations")]
    #[tokio::test]
    async fn reload_promotes_responses_only_to_one_combined_backend() {
        #[cfg(feature = "store-sqlite")]
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        #[cfg(feature = "store-sqlite")]
        let (backend_id, database_url) = (
            "sqlite",
            format!("sqlite://{}?mode=rwc", directory.path().join("reload.db").display()),
        );
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend_id, database_url) = ("postgres", "postgresql://user:password@8.8.8.8/store".to_owned());
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareFactory {
            backend_id,
            builds: Arc::clone(&builds),
            retires: Arc::clone(&retires),
        });
        let factories = vec![Arc::clone(&factory)];
        let mut initial = responses_and_conversations_config_for(backend_id, &database_url);
        initial.listeners.first_mut().expect("combined listener").filter_chains = vec!["responses".to_owned()];
        let mut initial_plans = build_listener_store_plans(&initial);
        validate_and_deduplicate_refs(&mut initial_plans, &factories, None, None).expect("initial stores");
        let cache = BackendCache::new(factories.clone());
        let mut initial_leases = Vec::new();
        for plan in &initial_plans {
            initial_leases.push(
                cache
                    .provision_into(&plan.refs, &plan.registry)
                    .await
                    .expect("initial backend"),
            );
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        let replacement = responses_and_conversations_config_for(backend_id, &database_url);
        let mut replacement_plans = build_listener_store_plans(&replacement);
        validate_and_deduplicate_refs(&mut replacement_plans, &factories, Some(&cache), Some(&initial_plans))
            .expect("replacement stores");
        let combined = replacement_plans.first().expect("combined listener");
        assert_eq!(
            combined.refs.first().map(|r| &r.config),
            combined.refs.get(1).map(|r| &r.config),
            "adding Conversations must promote both names to one combined key"
        );
        let mut replacement_leases = Vec::new();
        for plan in &replacement_plans {
            replacement_leases.push(
                cache
                    .provision_into(&plan.refs, &plan.registry)
                    .await
                    .expect("replacement backend"),
            );
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "reload should build one combined replacement, not a second filter-specific pool"
        );

        for lease in initial_leases {
            lease.release().await;
        }
        assert_eq!(
            retires.load(Ordering::SeqCst),
            1,
            "the drained raw Responses pool retires"
        );
        for lease in replacement_leases {
            lease.release().await;
        }
        assert_eq!(
            retires.load(Ordering::SeqCst),
            2,
            "the combined pool retires once after its generation"
        );
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[tokio::test]
    async fn reload_rejects_in_memory_sqlite_topology_expansion() {
        for database_url in [
            ":memory:",
            "sqlite://file::memory:",
            "sqlite://%3Amemory%3A",
            "sqlite::memory:",
            "sqlite::memory:?cache=shared",
        ] {
            let store_ref = StoreRef {
                name: Arc::from(DEFAULT_STORE_NAME),
                backend_id: Arc::from("sqlite"),
                config: json!({ "database_url": database_url }),
            };
            assert!(is_in_memory_sqlite(&store_ref), "memory URL: {database_url}");
        }
        let database_url = "sqlite::memory:?cache=shared";
        let builds = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareFactory {
            backend_id: "sqlite",
            builds: Arc::clone(&builds),
            retires: Arc::new(AtomicUsize::new(0)),
        });
        let factories = vec![Arc::clone(&factory)];
        let mut initial = responses_and_conversations_config_for("sqlite", database_url);
        initial.listeners.first_mut().expect("combined listener").filter_chains = vec!["responses".to_owned()];
        let mut initial_plans = build_listener_store_plans(&initial);
        validate_and_deduplicate_refs(&mut initial_plans, &factories, None, None).expect("initial stores");
        for plan in &mut initial_plans {
            plan.listener = format!("old-{}", plan.listener);
        }
        let cache = BackendCache::new(factories.clone());
        let mut initial_leases = Vec::new();
        for plan in &initial_plans {
            initial_leases.push(
                cache
                    .provision_into(&plan.refs, &plan.registry)
                    .await
                    .expect("initial backend"),
            );
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        let replacement = responses_and_conversations_config_for("sqlite", database_url);
        let mut replacement_plans = build_listener_store_plans(&replacement);
        let error =
            validate_and_deduplicate_refs(&mut replacement_plans, &factories, Some(&cache), Some(&initial_plans))
                .expect_err("an in-memory pool cannot be replaced without losing state");
        let message = error.to_string();
        assert!(message.contains("in-memory SQLite"), "actionable backend: {message}");
        assert!(
            message.contains("requires a restart"),
            "actionable remediation: {message}"
        );
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "rejected reload must leave the active pool unchanged"
        );

        for lease in initial_leases {
            lease.release().await;
        }
    }

    #[cfg(all(feature = "openai-conversations", feature = "store-sqlite"))]
    #[tokio::test]
    async fn unchanged_reload_matches_the_corresponding_in_memory_store() {
        let builds = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareFactory {
            backend_id: "sqlite",
            builds: Arc::clone(&builds),
            retires: Arc::new(AtomicUsize::new(0)),
        });
        let factories = vec![Arc::clone(&factory)];
        let mut config = responses_and_conversations_config();
        let mut other = config.filter_chains.first().expect("Responses chain").clone();
        other.name = "other-responses".to_owned();
        let filter = other.filters.first_mut().expect("Responses filter");
        let mapping = filter.config.as_mapping_mut().expect("Responses config");
        *mapping
            .get_mut(serde_yaml::Value::String("responses_table".to_owned()))
            .expect("responses table") = serde_yaml::Value::String("other_responses".to_owned());
        *mapping
            .get_mut(serde_yaml::Value::String("conversations_table".to_owned()))
            .expect("conversations table") = serde_yaml::Value::String("other_conversations".to_owned());
        config.filter_chains.push(other);
        config.listeners.get_mut(1).expect("second listener").filter_chains = vec!["other-responses".to_owned()];

        let mut active_plans = build_listener_store_plans(&config);
        validate_and_deduplicate_refs(&mut active_plans, &factories, None, None).expect("active stores");
        let cache = BackendCache::new(factories.clone());
        let mut leases = Vec::new();
        for plan in &active_plans {
            leases.push(
                cache
                    .provision_into(&plan.refs, &plan.registry)
                    .await
                    .expect("active backend"),
            );
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "distinct table sets need distinct pools"
        );

        let mut replacement_plans = build_listener_store_plans(&config);
        validate_and_deduplicate_refs(&mut replacement_plans, &factories, Some(&cache), Some(&active_plans))
            .expect("an unchanged reload must reuse each listener's exact pool");

        let mut changed = config.clone();
        let conversation = changed
            .filter_chains
            .iter_mut()
            .find(|chain| chain.name == "conversations")
            .and_then(|chain| chain.filters.first_mut())
            .expect("Conversations filter");
        let mapping = conversation.config.as_mapping_mut().expect("Conversations config");
        *mapping
            .get_mut(serde_yaml::Value::String("items_table".to_owned()))
            .expect("items table") = serde_yaml::Value::String("items_v2".to_owned());
        let mut changed_plans = build_listener_store_plans(&changed);
        let error = validate_and_deduplicate_refs(&mut changed_plans, &factories, Some(&cache), Some(&active_plans))
            .expect_err("changing an in-memory combined table set requires a restart");
        assert!(error.to_string().contains("requires a restart"));

        for lease in leases {
            lease.release().await;
        }
    }

    #[test]
    fn sqlite_normalization_removes_only_accepted_noop_fields() {
        let mut config = json!({
            "ssl_mode": null,
            "ssl_root_cert": null,
            "ssl_client_cert": null,
            "ssl_client_key": null,
            "require_certificate_authentication": false,
            "allow_private_database_url": false,
            "database_url": "sqlite::memory:",
        });

        normalize_sqlite_factory_config("sqlite", &mut config);

        let config = config.as_object().expect("object");
        assert_eq!(config.len(), 1);
        assert_eq!(config.get("database_url"), Some(&json!("sqlite::memory:")));
    }

    #[test]
    fn sqlite_normalization_preserves_rejected_nondefault_fields() {
        let mut config = json!({
            "ssl_mode": "verify_full",
            "require_certificate_authentication": true,
            "allow_private_database_url": true,
        });

        normalize_sqlite_factory_config("sqlite", &mut config);

        let config = config.as_object().expect("object");
        assert_eq!(config.get("ssl_mode"), Some(&json!("verify_full")));
        assert_eq!(config.get("require_certificate_authentication"), Some(&json!(true)));
        assert_eq!(config.get("allow_private_database_url"), Some(&json!(true)));
    }

    #[tokio::test]
    async fn ready_backend_is_not_retired_on_initial_shutdown_signal() {
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(FakeFactory {
            retires: Arc::clone(&retires),
        });
        let cache = Arc::new(BackendCache::new(vec![factory]));
        let registry = StoreRegistry::new();
        let (readiness, mut readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let service = Arc::new(StoreProvisionService {
            cache,
            factories: Vec::new(),
            plans: vec![ListenerStorePlan {
                listener: "web".to_owned(),
                registry,
                refs: vec![StoreRef {
                    name: Arc::from("default"),
                    backend_id: Arc::from("fake"),
                    config: json!({}),
                }],
            }],
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });

        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("service should become ready");
        shutdown_tx.send(true).expect("service should still receive shutdown");
        running.await.expect("provisioner task should stop");

        assert_eq!(
            retires.load(Ordering::SeqCst),
            0,
            "the initial shutdown signal precedes request draining"
        );
    }

    #[tokio::test]
    async fn permanent_failure_is_not_retried_or_partially_published() {
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factories: Vec<Arc<dyn StoreBackendFactory>> = vec![
            Arc::new(UnavailableFactory {
                builds: Arc::clone(&builds),
            }),
            Arc::new(FakeFactory {
                retires: Arc::clone(&retires),
            }),
        ];
        let failed_registry = StoreRegistry::new();
        let healthy_registry = StoreRegistry::new();
        let (readiness, mut readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let service = Arc::new(StoreProvisionService {
            cache: Arc::new(BackendCache::new(factories)),
            factories: Vec::new(),
            plans: vec![
                ListenerStorePlan {
                    listener: "failed".to_owned(),
                    registry: failed_registry.clone(),
                    refs: vec![StoreRef {
                        name: Arc::from("default"),
                        backend_id: Arc::from("unavailable"),
                        config: json!({}),
                    }],
                },
                ListenerStorePlan {
                    listener: "healthy".to_owned(),
                    registry: healthy_registry.clone(),
                    refs: vec![StoreRef {
                        name: Arc::from("default"),
                        backend_id: Arc::from("fake"),
                        config: json!({}),
                    }],
                },
            ],
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });

        tokio::time::timeout(
            Duration::from_secs(1),
            readiness_rx.wait_for(|state| *state == StoreReadiness::Failed),
        )
        .await
        .expect("permanent failure should be published")
        .expect("terminal failure state should remain observable");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "terminal unavailable failures must not be retried by the server"
        );
        assert!(!failed_registry.is_ready());
        assert!(
            !healthy_registry.is_ready(),
            "a failed generation must not publish partially"
        );

        running.await.expect("initially failed provisioner should stop");
        assert_eq!(
            retires.load(Ordering::SeqCst),
            1,
            "a successfully provisioned sibling must be retired when the generation fails"
        );
    }

    #[test]
    #[expect(
        clippy::exit,
        reason = "the child exit code distinguishes a missed fatal startup path"
    )]
    fn terminal_initial_failure_exits_startup_process() {
        const CHILD_MARKER: &str = "PRAXIS_TEST_INITIAL_STORE_FAILURE_CHILD";
        if std::env::var_os(CHILD_MARKER).is_some() {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("child Tokio runtime");
            runtime.block_on(async {
                let factory: Arc<dyn StoreBackendFactory> = Arc::new(UnavailableFactory {
                    builds: Arc::new(AtomicUsize::new(0)),
                });
                let (readiness, _readiness_rx) = watch::channel(StoreReadiness::Pending);
                let (_commands, command_rx) = mpsc::unbounded_channel();
                let service = StoreProvisionService {
                    cache: Arc::new(BackendCache::new(vec![factory])),
                    factories: Vec::new(),
                    plans: vec![ListenerStorePlan {
                        listener: "failed".to_owned(),
                        registry: StoreRegistry::new(),
                        refs: vec![StoreRef {
                            name: Arc::from("default"),
                            backend_id: Arc::from("unavailable"),
                            config: json!({}),
                        }],
                    }],
                    readiness,
                    commands: AsyncMutex::new(Some(command_rx)),
                };
                let (_shutdown_tx, shutdown_rx) = watch::channel(false);
                let (ready_tx, _ready_rx) = watch::channel(false);
                service
                    .start_with_ready_notifier(shutdown_rx, ServiceReadyNotifier::new(ready_tx))
                    .await;
            });
            std::process::exit(2);
        }

        let output = std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "--exact",
                "store_provision::tests::terminal_initial_failure_exits_startup_process",
                "--nocapture",
            ])
            .env(CHILD_MARKER, "1")
            .output()
            .expect("run startup-failure child process");

        assert_eq!(output.status.code(), Some(1), "child must reject startup");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("fatal: initial response-store provisioning failed"),
            "actionable startup diagnostic: {stderr}"
        );
        assert!(
            stderr.contains("test backend is down"),
            "backend cause must be preserved: {stderr}"
        );
    }

    #[tokio::test]
    async fn transient_failure_recovers_within_cache_retry_budget() {
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factories: Vec<Arc<dyn StoreBackendFactory>> = vec![Arc::new(RecoveringFactory {
            builds: Arc::clone(&builds),
            retires: Arc::clone(&retires),
        })];
        let recovered_registry = StoreRegistry::new();
        let (readiness, _readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let service = Arc::new(StoreProvisionService {
            cache: Arc::new(BackendCache::new(factories)),
            factories: Vec::new(),
            plans: vec![ListenerStorePlan {
                listener: "recovering".to_owned(),
                registry: recovered_registry.clone(),
                refs: vec![StoreRef {
                    name: Arc::from("default"),
                    backend_id: Arc::from("recovering"),
                    config: json!({}),
                }],
            }],
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });

        tokio::time::timeout(Duration::from_millis(500), async {
            while !recovered_registry.is_ready() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("transient failure should recover within the cache retry budget");
        assert_eq!(builds.load(Ordering::SeqCst), 3);

        shutdown_tx.send(true).expect("service should still receive shutdown");
        running.await.expect("provisioner task should stop");
        assert_eq!(retires.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_cancels_reload_preparation_and_unblocks_watcher() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(HangingFactory {
            entered: Arc::clone(&entered),
        });
        let (readiness, _readiness_rx) = watch::channel(StoreReadiness::Ready);
        let (commands, command_rx) = mpsc::unbounded_channel();
        let handle = StoreReloadHandle { commands };
        let service = Arc::new(StoreProvisionService {
            cache: Arc::new(BackendCache::new(vec![Arc::clone(&factory)])),
            factories: vec![factory],
            plans: Vec::new(),
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        let config = single_store_config("hanging", "unused");
        let preparing = tokio::task::spawn_blocking(move || handle.prepare(&config));

        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .expect("replacement provisioning should start");
        shutdown_tx.send(true).expect("service should receive shutdown");
        let result = tokio::time::timeout(Duration::from_secs(1), preparing)
            .await
            .expect("watcher must not remain blocked by backend initialization")
            .expect("prepare task");
        assert!(result.is_err(), "shutdown must reject the pending reload");
        let error = result.err().expect("error checked above");
        assert!(error.contains("stopped during reload preparation"));
        tokio::time::timeout(Duration::from_secs(1), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_reload_keeps_active_generation_and_accepts_later_reload() {
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReloadFactory {
            builds: Arc::clone(&builds),
            retires: Arc::clone(&retires),
        });
        let factories = vec![Arc::clone(&factory)];
        let mut plans = build_listener_store_plans(&single_store_config("reload", "initial"));
        validate_and_deduplicate_refs(&mut plans, &factories, None, None).expect("initial plans");
        let (readiness, mut readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (commands, command_rx) = mpsc::unbounded_channel();
        let handle = StoreReloadHandle { commands };
        let service = Arc::new(StoreProvisionService {
            cache: Arc::new(BackendCache::new(factories.clone())),
            factories,
            plans,
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });

        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");

        let failed = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&single_store_config("reload", "failed"))
        })
        .await
        .expect("failed prepare task");
        let error = failed.err().expect("terminal candidate failure must reject the reload");
        assert!(error.contains("replacement backend is down"), "backend cause: {error}");
        assert_eq!(*readiness_rx.borrow(), StoreReadiness::Ready);

        let recovered = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&single_store_config("reload", "replacement"))
        })
        .await
        .expect("recovery prepare task")
        .expect("later replacement should provision");
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.abort(recovered)
        })
        .await
        .expect("abort task")
        .expect("prepared recovery generation should abort cleanly");

        assert_eq!(builds.load(Ordering::SeqCst), 3);
        assert_eq!(
            retires.load(Ordering::SeqCst),
            1,
            "only the aborted candidate should retire"
        );
        shutdown_tx.send(true).expect("service should receive shutdown");
        running.await.expect("provisioner task should stop");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reload_builds_swaps_drains_and_retires_store_generation() {
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReloadFactory {
            builds: Arc::clone(&builds),
            retires: Arc::clone(&retires),
        });
        let factories = vec![Arc::clone(&factory)];
        let initial_config = single_store_config("reload", "initial");
        let mut plans = build_listener_store_plans(&initial_config);
        validate_and_deduplicate_refs(&mut plans, &factories, None, None).expect("initial plans");
        let cache = Arc::new(BackendCache::new(factories.clone()));
        let (readiness, mut readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (commands, command_rx) = mpsc::unbounded_channel();
        let handle = StoreReloadHandle { commands };
        let service = Arc::new(StoreProvisionService {
            cache,
            factories,
            plans,
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });

        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");
        assert_eq!(builds.load(Ordering::SeqCst), 1);

        let next_config = single_store_config("reload", "replacement");
        let prepared = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&next_config)
        })
        .await
        .expect("prepare task")
        .expect("replacement generation should provision");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert!(
            prepared
                .registries
                .get("web")
                .is_some_and(ResponseStoreRegistry::is_ready),
            "replacement registry must be ready before pipeline swap"
        );

        let registry = FilterRegistry::with_builtins();
        let mut entries = [];
        let old_pipeline = Arc::new(FilterPipeline::build(&mut entries, &registry).expect("old pipeline"));
        let in_flight = Arc::clone(&old_pipeline);
        let old_pipeline_observer = Arc::downgrade(&old_pipeline);
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.commit(prepared, vec![old_pipeline_observer])
        })
        .await
        .expect("commit task")
        .expect("replacement generation should commit");
        drop(old_pipeline);

        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            retires.load(Ordering::SeqCst),
            0,
            "old backend must remain live while an in-flight request holds its pipeline"
        );
        drop(in_flight);
        tokio::time::timeout(Duration::from_secs(1), async {
            while retires.load(Ordering::SeqCst) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("old backend should retire after request drain");

        shutdown_tx.send(true).expect("service should still receive shutdown");
        running.await.expect("provisioner task should stop");
        assert_eq!(
            retires.load(Ordering::SeqCst),
            1,
            "active generation retires after runtime drain"
        );
    }

    #[tokio::test]
    async fn concurrent_drain_observers_do_not_keep_a_pipeline_alive() {
        let builds = Arc::new(AtomicUsize::new(0));
        let retires = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReloadFactory {
            builds,
            retires: Arc::clone(&retires),
        });
        let cache = BackendCache::new(vec![factory]);
        let first = cache
            .provision(&[StoreRef {
                name: Arc::from("default"),
                backend_id: Arc::from("reload"),
                config: json!({"database_url": "first"}),
            }])
            .await
            .expect("first generation");
        let second = cache
            .provision(&[StoreRef {
                name: Arc::from("default"),
                backend_id: Arc::from("reload"),
                config: json!({"database_url": "second"}),
            }])
            .await
            .expect("second generation");

        let registry = FilterRegistry::with_builtins();
        let pipeline = Arc::new(FilterPipeline::build(&mut [], &registry).expect("pipeline"));
        let observer = Arc::downgrade(&pipeline);
        let first_release = tokio::spawn(release_after_drain(vec![Weak::clone(&observer)], vec![first.lease]));
        let second_release = tokio::spawn(release_after_drain(vec![observer], vec![second.lease]));

        drop(pipeline);
        tokio::time::timeout(Duration::from_secs(1), async {
            first_release.await.expect("first release task");
            second_release.await.expect("second release task");
        })
        .await
        .expect("weak observers must not block one another");
        assert_eq!(retires.load(Ordering::SeqCst), 2);
    }
}
