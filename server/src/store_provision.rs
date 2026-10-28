// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Serving-runtime provisioning of response-store backends.
//!
//! The store filter reads an owner-scoped handle from a per-listener registry it
//! never populates. A Pingora background service provisions the configured
//! backends and registers them into that registry on the serving runtime. sqlx
//! pools bind to the runtime that opens them, so provisioning must run there, not
//! on the config-watcher runtime a reload uses.

#![cfg(feature = "_store-backend")]

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
use praxis_ai_apis::store::plan::{StoreRefLocation, coalesce_compatible_filter_refs, validate_cross_store_collisions};
#[cfg(feature = "openai-conversations")]
use praxis_ai_apis::store::{CONVERSATIONS_STORE_NAME, DEFAULT_STORE_NAME};
use praxis_ai_apis::store::{ResponseStoreRegistry, plan, store_backend_factories};
use praxis_ai_store::{
    BackendError, BackendNamespaceKey, EffectiveConfigKey, PersistedStateBackend, ReaperPolicy, StoreBackendFactory,
    StoreRegistry,
};
use praxis_ai_store_lifecycle::{BackendCache, BackendLease, ProvisionError, StoreRef};
use praxis_core::config::{Config, FilterEntry};
use praxis_filter::FilterPipeline;
use tokio::sync::{Mutex as AsyncMutex, mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::reaper::{ReaperHandle, run_reaper};

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
    /// TTL reaper policy per store name, parsed from the filter config before its
    /// ttl fields are stripped from `refs`. Empty when no store declares a `ttl`.
    reaper_bindings: Vec<(Arc<str>, ReaperPolicy)>,
}

impl AsRef<[StoreRef]> for ListenerStorePlan {
    fn as_ref(&self) -> &[StoreRef] {
        &self.refs
    }
}

impl AsMut<[StoreRef]> for ListenerStorePlan {
    fn as_mut(&mut self) -> &mut [StoreRef] {
        &mut self.refs
    }
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
        let refs = plan::discover_store_refs_permissive(&listener.filter_chains, &chains);
        if !refs.is_empty() {
            let reaper_bindings = plan::discover_reaper_bindings_permissive(&listener.filter_chains, &chains);
            plans.push(ListenerStorePlan {
                listener: listener.name.clone(),
                registry: StoreRegistry::new(),
                refs,
                reaper_bindings,
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
        validate_cross_store_collisions(plans, &factories)?;
    }

    for plan in plans.iter_mut() {
        plan::deduplicate_refs(&format!("listener '{}'", plan.listener), &mut plan.refs, &factories)?;
    }
    validate_reaper_policy_conflicts(plans, &factories)?;
    Ok(())
}

/// Suffix the responses schema appends to derive its server-owned
/// pending-approvals table. Mirrors `schemas::pending_approvals_table` in the
/// store-backends crate; kept in sync so the reaper-overlap check sweeps the exact
/// physical table the store creates.
const PENDING_APPROVALS_SUFFIX: &str = "_pending_approvals";

/// Suffix the responses schema appends to derive its durable SSE event-log table.
/// Mirrors `schemas::events_table` in the store-backends crate.
const EVENTS_SUFFIX: &str = "_events";

/// Reject a configuration whose store filters assign two different reaper policies
/// to the same physical table.
///
/// Reapers are keyed on backend identity (the factory's [`EffectiveConfigKey`]) so
/// every distinct cached backend is swept exactly once (see
/// [`binding_effective_key`]). Distinct effective keys do not imply disjoint
/// tables, though: two stores on one physical database with different
/// `responses_table` but the same `conversations_table` resolve to different keys
/// yet both sweep that shared conversations table, so a shorter TTL would delete
/// rows the longer retention still needs. The real rule is therefore physical:
/// within one database, no two bindings may give different policies to the same
/// physical table.
///
/// Persistent backends expose a [`BackendNamespaceKey`], so they are checked by
/// `(database, table)` overlap. In-memory backends report no namespace key and
/// never share a physical table across distinct backends; they keep the
/// effective-key check, so identical in-memory config with a different ttl (one
/// shared backend) is still rejected. Both checks must hold. The same policy on a
/// shared table is idempotent and allowed.
///
/// The loop walks **every** provisioned store (`plan.refs`), not just the
/// TTL-enabled ones in `plan.reaper_bindings`. A store without a `ttl` is recorded
/// with a `None` policy so a TTL-enabled store cannot quietly reap a physical table
/// a TTL-disabled store shares: omitting `ttl` is documented to retain rows
/// indefinitely, so an enabled-vs-disabled overlap is a conflict, rejected before
/// provisioning. Reaper creation itself stays limited to TTL-enabled stores (see
/// [`StoreProvisionService::desired_reapers`]).
fn validate_reaper_policy_conflicts(
    plans: &[ListenerStorePlan],
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<(), ProvisionError> {
    let mut by_table = ReaperTableOwners::new();
    let mut by_effective_key: HashMap<EffectiveConfigKey, (Option<ReaperPolicy>, Arc<str>)> = HashMap::new();
    for plan in plans {
        for store_ref in &plan.refs {
            let store_name = &store_ref.name;
            let policy = plan
                .reaper_bindings
                .iter()
                .find(|(name, _)| name == store_name)
                .map(|(_, policy)| *policy);
            let factory = binding_factory(store_ref, factories)?;
            let namespace = factory
                .namespace_key(&store_ref.config)
                .map_err(|source| ProvisionError::Backend {
                    name: Arc::clone(store_name),
                    source,
                })?;
            if let Some(namespace) = namespace {
                check_shared_table_policies(&mut by_table, &namespace, store_ref, store_name, policy)?;
            } else {
                check_in_memory_backend_policy(&mut by_effective_key, factory, store_ref, store_name, policy)?;
            }
        }
    }
    Ok(())
}

/// Maps each `(physical database, folded physical table)` to the reaper policy and
/// the store that first claimed it, so a later binding assigning a different policy
/// to a shared table can be rejected by name. `None` marks a TTL-disabled store, so
/// an enabled-vs-disabled overlap is caught like any other policy mismatch.
type ReaperTableOwners = HashMap<(BackendNamespaceKey, String), (Option<ReaperPolicy>, Arc<str>)>;

/// Record a store's swept tables, rejecting a different policy on any physical table
/// another store in the same database already claimed. A `None` policy (no `ttl`)
/// conflicts with any TTL-enabled policy on the same table.
fn check_shared_table_policies(
    by_table: &mut ReaperTableOwners,
    namespace: &BackendNamespaceKey,
    store_ref: &StoreRef,
    store_name: &Arc<str>,
    policy: Option<ReaperPolicy>,
) -> Result<(), ProvisionError> {
    for table in swept_physical_tables(&store_ref.config) {
        let key = (namespace.clone(), table.to_ascii_lowercase());
        match by_table.get(&key) {
            Some((existing_policy, existing_store)) if *existing_policy != policy => {
                return Err(ProvisionError::Backend {
                    name: Arc::clone(store_name),
                    source: BackendError::Config(reaper_policy_conflict_message(
                        *existing_policy,
                        existing_store,
                        policy,
                        store_name,
                        Some(&table),
                    )),
                });
            },
            Some(_) => {},
            None => {
                by_table.insert(key, (policy, Arc::clone(store_name)));
            },
        }
    }
    Ok(())
}

/// Record an in-memory store's backend identity, rejecting a different policy on the
/// same shared in-memory backend (identical config, different or absent ttl).
fn check_in_memory_backend_policy(
    by_effective_key: &mut HashMap<EffectiveConfigKey, (Option<ReaperPolicy>, Arc<str>)>,
    factory: &dyn StoreBackendFactory,
    store_ref: &StoreRef,
    store_name: &Arc<str>,
    policy: Option<ReaperPolicy>,
) -> Result<(), ProvisionError> {
    let key = factory
        .effective_key(&store_ref.config)
        .map_err(|source| ProvisionError::Backend {
            name: Arc::clone(store_name),
            source,
        })?;
    match by_effective_key.get(&key) {
        Some((existing_policy, existing_store)) if *existing_policy != policy => Err(ProvisionError::Backend {
            name: Arc::clone(store_name),
            source: BackendError::Config(reaper_policy_conflict_message(
                *existing_policy,
                existing_store,
                policy,
                store_name,
                None,
            )),
        }),
        Some(_) => Ok(()),
        None => {
            by_effective_key.insert(key, (policy, Arc::clone(store_name)));
            Ok(())
        },
    }
}

/// Build the actionable error for two stores assigning different reaper policies to
/// a shared target. `table` names the shared physical table for persistent backends;
/// `None` describes a shared in-memory backend. A `Some`/`None` policy pair (one
/// store sets a `ttl`, the other does not) is reported as the TTL-disabled store's
/// rows being at risk; two differing `Some` policies are reported as a ttl mismatch.
fn reaper_policy_conflict_message(
    existing_policy: Option<ReaperPolicy>,
    existing_store: &Arc<str>,
    policy: Option<ReaperPolicy>,
    store_name: &Arc<str>,
    table: Option<&str>,
) -> String {
    let target = table.map_or_else(
        || "the same in-memory store backend".to_owned(),
        |table| format!("the shared physical table '{table}' in one database"),
    );
    if let (Some(_), Some(_)) = (existing_policy, policy) {
        format!(
            "conflicting reaper policies target {target} (stores '{existing_store}' and '{store_name}'); \
             configure one ttl/sweep_interval/sweep_batch_size for every store that shares it"
        )
    } else {
        let (enabled, disabled) = if existing_policy.is_some() {
            (existing_store, store_name)
        } else {
            (store_name, existing_store)
        };
        format!(
            "store '{enabled}' enables a ttl reaper on {target} shared with store '{disabled}', which sets \
             no ttl and must retain its rows; give both stores the same ttl or stop sharing it"
        )
    }
}

/// Physical tables a reaper for this store sweeps, derived from its config.
///
/// Mirrors the store's own schema: the configured `responses_table`,
/// `conversations_table`, and optional `items_table`, plus the two tables the
/// responses schema derives from the responses name
/// (`{responses}_pending_approvals` and `{responses}_events`). Names are returned
/// as configured; the caller folds case so the overlap check matches both
/// backends' identifier folding.
fn swept_physical_tables(config: &serde_json::Value) -> Vec<String> {
    let field = |name| config.get(name).and_then(serde_json::Value::as_str);
    let mut tables = Vec::new();
    if let Some(responses) = field("responses_table") {
        tables.push(responses.to_owned());
        tables.push(format!("{responses}{PENDING_APPROVALS_SUFFIX}"));
        tables.push(format!("{responses}{EVENTS_SUFFIX}"));
    }
    if let Some(conversations) = field("conversations_table") {
        tables.push(conversations.to_owned());
    }
    if let Some(items) = field("items_table") {
        tables.push(items.to_owned());
    }
    tables
}

/// Resolve the store reference backing one reaper binding within its plan.
fn binding_store_ref<'plan>(
    plan: &'plan ListenerStorePlan,
    store_name: &Arc<str>,
) -> Result<&'plan StoreRef, ProvisionError> {
    plan.refs
        .iter()
        .find(|store_ref| store_ref.name == *store_name)
        .ok_or_else(|| ProvisionError::Backend {
            name: Arc::clone(store_name),
            source: BackendError::Config(format!("store reference {store_name} not found in plan")),
        })
}

/// Resolve the injected factory for one store reference.
fn binding_factory<'factory>(
    store_ref: &StoreRef,
    factories: &HashMap<&str, &'factory dyn StoreBackendFactory>,
) -> Result<&'factory dyn StoreBackendFactory, ProvisionError> {
    factories
        .get(store_ref.backend_id.as_ref())
        .copied()
        .ok_or_else(|| ProvisionError::Backend {
            name: Arc::clone(&store_ref.name),
            source: BackendError::Config(format!("no factory injected for backend_id {}", store_ref.backend_id)),
        })
}

/// Resolve the backend-identity key for one reaper binding.
///
/// Keys on the factory's [`EffectiveConfigKey`], which [`BackendCache`] also uses
/// to deduplicate cached backends 1:1. Keying reapers on it therefore yields
/// exactly one reaper per distinct cached backend -- every backend under the
/// coordinator is swept with no drops, and two stores on one database with
/// different table sets (distinct effective keys) each get their own reaper. The
/// stripped `ttl`/`sweep_interval`/`sweep_batch_size` fields are not part of the
/// key, so a ttl-only reload keeps the same key (pool reused, policy handed off in
/// place) while a pool/compression reload changes it (old reaper retired, new one
/// spawned on the rebuilt backend).
fn binding_effective_key(
    plan: &ListenerStorePlan,
    store_name: &Arc<str>,
    factories: &HashMap<&str, &dyn StoreBackendFactory>,
) -> Result<EffectiveConfigKey, ProvisionError> {
    let store_ref = binding_store_ref(plan, store_name)?;
    let factory = binding_factory(store_ref, factories)?;
    factory
        .effective_key(&store_ref.config)
        .map_err(|source| ProvisionError::Backend {
            name: Arc::clone(&store_ref.name),
            source,
        })
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
    /// error returned here, including [`praxis_ai_store::BackendError::Unavailable`], is terminal
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

    /// Build the desired reaper set for a committed generation.
    ///
    /// One entry per distinct backend identity (the factory's
    /// [`EffectiveConfigKey`]) that declares a `ttl`, carrying the committed policy
    /// and the backend handle resolved from the generation's registry.
    /// [`BackendCache`] deduplicates cached backends 1:1 by `effective_key`, so
    /// keying here on the same value yields exactly one reaper per cached backend:
    /// every distinct backend is swept, and two stores on one physical database
    /// with different (non-coalescing) table sets each get their own reaper rather
    /// than the second being silently dropped.
    ///
    /// A store with no registered backend in this generation is skipped.
    /// Conflicting policies on one `effective_key` are rejected earlier by
    /// [`validate_reaper_policy_conflicts`], so the first binding per key wins and
    /// later equal bindings reuse it.
    fn desired_reapers(
        &self,
        plans: &[ListenerStorePlan],
    ) -> HashMap<EffectiveConfigKey, (ReaperPolicy, Arc<dyn PersistedStateBackend>)> {
        let factories: HashMap<&str, &dyn StoreBackendFactory> = self
            .factories
            .iter()
            .map(|factory| (factory.backend_id(), factory.as_ref()))
            .collect();
        let mut desired: HashMap<EffectiveConfigKey, (ReaperPolicy, Arc<dyn PersistedStateBackend>)> = HashMap::new();
        for plan in plans {
            for (store_name, policy) in &plan.reaper_bindings {
                let Some(backend) = plan.registry.backend(store_name) else {
                    debug!(
                        listener = %plan.listener,
                        store = %store_name,
                        "ttl reaper: store declares a ttl but has no registered backend in this generation; skipping"
                    );
                    continue;
                };
                let key = match binding_effective_key(plan, store_name, &factories) {
                    Ok(key) => key,
                    Err(error) => {
                        warn!(%error, listener = %plan.listener, "ttl reaper: cannot compute effective key; skipping");
                        continue;
                    },
                };
                // Accepted narrow edge: two backends that share the same tables but
                // differ only in pool/compression have distinct effective keys, so
                // each gets its own reaper sweeping those tables — harmless
                // (idempotent, bounded) and a bizarre config.
                desired.entry(key).or_insert((*policy, backend));
            }
        }
        desired
    }

    /// Reconcile running TTL reapers to a committed generation's policies.
    ///
    /// Stops (cancel + await) any reaper whose backend identity (`effective_key`)
    /// left the generation or whose policy changed, then spawns a reaper for every
    /// desired backend not already running. Stopping before spawning gives the
    /// design's handoff: the outgoing reaper finishes its current batch and is
    /// acknowledged stopped before the replacement sweeps, so two policies never
    /// share one backend.
    ///
    /// A reload that changes only `pool`/`compression` produces a new
    /// `effective_key`, so [`BackendCache`] builds a fresh backend `Arc` under a
    /// new reaper key: the old key is absent from the desired set and its reaper is
    /// retired, while a new reaper is spawned on the rebuilt backend. A ttl-only
    /// reload keeps the same `effective_key`, so the same reaper key sees a changed
    /// policy and hands off in place while the pool is reused. The retained
    /// `Arc::ptr_eq` backend check stays correct because the same `effective_key`
    /// resolves to the same cached backend `Arc`.
    ///
    /// Called only for an accepted generation (initial provisioning and
    /// `Commit`), never for a prepared-but-aborted one, so a rejected reload
    /// activates no policy and deletes nothing.
    #[expect(
        clippy::too_many_lines,
        reason = "staleness detection (policy or backend swap), handoff await, and respawn are one reconciliation step"
    )]
    async fn reconcile_reapers(
        &self,
        reapers: &mut HashMap<EffectiveConfigKey, ReaperHandle>,
        plans: &[ListenerStorePlan],
        shutdown: &ShutdownWatch,
    ) {
        let desired = self.desired_reapers(plans);

        // Stop reapers whose backend identity left the committed generation or
        // whose policy changed. Either hands off: the outgoing task is cancelled
        // and awaited here before its replacement is spawned below.
        let stale: Vec<EffectiveConfigKey> = reapers
            .iter()
            .filter(|&(key, handle)| match desired.get(key) {
                Some((policy, backend)) => *policy != handle.policy || !Arc::ptr_eq(backend, &handle.backend),
                None => true,
            })
            .map(|(key, _handle)| key.clone())
            .collect();
        for key in stale {
            if let Some(handle) = reapers.remove(&key) {
                handle.cancel.cancel();
                let _joined = handle.join.await;
            }
        }

        // Spawn a reaper for every desired backend not already running.
        for (key, (policy, backend)) in desired {
            if reapers.contains_key(&key) {
                continue;
            }
            let cancel = CancellationToken::new();
            let join = tokio::spawn(run_reaper(
                Arc::clone(&backend),
                policy,
                cancel.clone(),
                shutdown.clone(),
            ));
            reapers.insert(
                key,
                ReaperHandle {
                    cancel,
                    join,
                    policy,
                    backend,
                },
            );
        }
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
        clippy::large_stack_frames,
        reason = "startup and reload commands share one serving-runtime lease owner; its one cold long-lived task holds the concurrent provisioning futures"
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
        let mut reapers: HashMap<EffectiveConfigKey, ReaperHandle> = HashMap::new();
        // Box the reconcile future so the cold reload path does not inflate this
        // long-lived task's stack frame.
        Box::pin(self.reconcile_reapers(&mut reapers, &active_plans, &shutdown)).await;
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
                                // The handoff cancels and awaits each stale reaper inline (finishing
                                // its current bounded batch), so this commit can block the command
                                // loop up to DB_WAIT_BOUND (30s) per stale reaper before the swap
                                // completes. Acceptable: it runs only on a policy/backend change.
                                Box::pin(self.reconcile_reapers(&mut reapers, &active_plans, &shutdown)).await;
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
        // Reapers subscribe to shutdown directly, but cancel and await them here
        // too so a stuck sweep cannot outlive the service; each finishes its
        // current bounded batch before stopping.
        for (_key, handle) in reapers.drain() {
            handle.cancel.cancel();
            let _joined = handle.join.await;
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
#[cfg(any(feature = "store-postgres", feature = "store-sqlite"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::too_many_lines, reason = "tests")]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use praxis_ai_store::{
        BackendError, BackendNamespaceKey, ConversationItemRecord, ConversationItemStore, ConversationRecord,
        EffectiveConfigKey, EventLogStatus, PendingApprovalRecord, ProvisionedBackend, ResponseEventRecord,
        ResponseRecord, ResponseStore, RetireBackend, StateOwner, StoreBackendFactory, StoreError, SweepStats,
        memory::InMemoryStore,
    };
    use praxis_filter::FilterRegistry;
    use serde_json::json;
    use tokio::sync::Notify;

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
        entered: Arc<Notify>,
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
                  - filter: openai_responses_store
                    backend: {backend}
                    database_url: "{first_url}"
                    responses_table: responses
                    conversations_table: conversations
          - name: second
            chains:
              - name: second-store
                filters:
                  - filter: openai_responses_store
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations
  - name: responses-b
    filters:
      - filter: openai_responses_store
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
    fn store_discovery_keeps_permissive_server_behavior() {
        let mut config = single_store_config("sqlite", "sqlite::memory:");
        config.filter_chains.first_mut().expect("main chain").filters = serde_yaml::from_str(
            "- filter: openai_responses_store\n  backend: 7\n- filter: iterative_request_router\n  steps:\n    - filters: malformed\n    - filters: [7, {filter: openai_responses_store, backend: sqlite, database_url: 'sqlite::memory:'}]",
        ).expect("mixed filter entries");
        let plans = build_listener_store_plans(&config);
        assert_eq!(plans.len(), 1);
        let refs = &plans.first().expect("listener plan").refs;
        assert_eq!(refs.len(), 1, "valid nested sibling must survive malformed entries");
        assert_eq!(refs.first().expect("store reference").backend_id.as_ref(), "sqlite");
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
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
      - filter: openai_responses_store
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
                reaper_bindings: Vec::new(),
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
                    reaper_bindings: Vec::new(),
                },
                ListenerStorePlan {
                    listener: "healthy".to_owned(),
                    registry: healthy_registry.clone(),
                    refs: vec![StoreRef {
                        name: Arc::from("default"),
                        backend_id: Arc::from("fake"),
                        config: json!({}),
                    }],
                    reaper_bindings: Vec::new(),
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
                        reaper_bindings: Vec::new(),
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
                reaper_bindings: Vec::new(),
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
        let entered = Arc::new(Notify::new());
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

    // -------------------------------------------------------------------------
    // TTL reaper coordinator lifecycle
    // -------------------------------------------------------------------------

    /// Shared record of every committed sweep, inspected by the reaper tests.
    #[derive(Default)]
    struct SweepSpy {
        /// Cutoff second recorded by each `sweep_expired` call that returned Ok.
        cutoffs: std::sync::Mutex<Vec<i64>>,
        /// Count of committed `sweep_expired` calls.
        committed: AtomicUsize,
    }

    impl SweepSpy {
        /// Snapshot the recorded cutoffs.
        fn cutoffs(&self) -> Vec<i64> {
            match self.cutoffs.lock() {
                Ok(guard) => guard.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            }
        }

        /// Number of sweeps that committed.
        fn committed(&self) -> usize {
            self.committed.load(Ordering::SeqCst)
        }
    }

    /// Lets a test hold one sweep batch open, then release it, to drive the
    /// shutdown-during-active-batch case.
    struct SweepGate {
        /// Fired once the gated batch has entered `sweep_expired`.
        entered: Notify,
        /// Awaited inside the gated batch until the test lets it proceed.
        proceed: Notify,
        /// Gates only the first batch; later batches run through.
        armed: AtomicBool,
    }

    /// A `PersistedStateBackend` that records sweeps and never touches real
    /// storage. Non-sweep methods are never called by the reaper and return
    /// benign values.
    struct SpyBackend {
        spy: Arc<SweepSpy>,
        gate: Option<Arc<SweepGate>>,
    }

    #[async_trait]
    impl ResponseStore for SpyBackend {
        async fn sweep_expired(&self, cutoff_secs: i64, _batch_size: u32) -> Result<SweepStats, StoreError> {
            if let Some(gate) = &self.gate
                && gate.armed.swap(false, Ordering::SeqCst)
            {
                gate.entered.notify_one();
                gate.proceed.notified().await;
            }
            match self.spy.cutoffs.lock() {
                Ok(mut guard) => guard.push(cutoff_secs),
                Err(poisoned) => poisoned.into_inner().push(cutoff_secs),
            }
            self.spy.committed.fetch_add(1, Ordering::SeqCst);
            // Zero stats end the per-tick sweep loop after this single batch.
            Ok(SweepStats::default())
        }

        async fn upsert_response(&self, _record: &ResponseRecord) -> Result<(), StoreError> {
            Ok(())
        }

        async fn get_response(&self, _owner: &StateOwner, _id: &str) -> Result<Option<ResponseRecord>, StoreError> {
            Ok(None)
        }

        async fn delete_response(&self, _owner: &StateOwner, _id: &str) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn get_conversation(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
        ) -> Result<Option<ConversationRecord>, StoreError> {
            Ok(None)
        }

        async fn record_pending_approvals(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
            _records: &[PendingApprovalRecord],
            _created_at: i64,
        ) -> Result<(), StoreError> {
            Ok(())
        }

        async fn get_pending_approvals(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
            _approval_ids: &[&str],
        ) -> Result<Vec<PendingApprovalRecord>, StoreError> {
            Ok(Vec::new())
        }

        async fn consume_approvals(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
            _approval_ids: &[&str],
            _consumed_at: i64,
        ) -> Result<Option<usize>, StoreError> {
            Ok(None)
        }

        async fn append_events(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
            _events: &[ResponseEventRecord],
        ) -> Result<(), StoreError> {
            Ok(())
        }

        async fn list_events_after(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
            _after: Option<u64>,
            _limit: u32,
        ) -> Result<Vec<ResponseEventRecord>, StoreError> {
            Ok(Vec::new())
        }

        async fn event_log_status(
            &self,
            _owner: &StateOwner,
            _response_id: &str,
        ) -> Result<EventLogStatus, StoreError> {
            Ok(EventLogStatus::Absent)
        }
    }

    #[async_trait]
    impl ConversationItemStore for SpyBackend {
        async fn upsert_conversation(&self, _record: &ConversationRecord) -> Result<(), StoreError> {
            Ok(())
        }

        async fn update_conversation_messages(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _messages: &serde_json::Value,
        ) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn update_conversation_metadata(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _metadata: &serde_json::Value,
        ) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn compare_and_swap_conversation_messages(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _expected_messages: &serde_json::Value,
            _messages: &serde_json::Value,
        ) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn get_conversation(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
        ) -> Result<Option<ConversationRecord>, StoreError> {
            Ok(None)
        }

        async fn delete_conversation(&self, _owner: &StateOwner, _conversation_id: &str) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn create_conversation_items(&self, _items: &[ConversationItemRecord]) -> Result<(), StoreError> {
            Ok(())
        }

        async fn list_conversation_items(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _after_item_id: Option<&str>,
            _limit: u32,
            _ascending: bool,
        ) -> Result<Vec<ConversationItemRecord>, StoreError> {
            Ok(Vec::new())
        }

        async fn get_existing_conversation_item_ids(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _item_ids: &[&str],
        ) -> Result<Vec<String>, StoreError> {
            Ok(Vec::new())
        }

        async fn get_conversation_item(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _item_id: &str,
        ) -> Result<Option<ConversationItemRecord>, StoreError> {
            Ok(None)
        }

        async fn delete_conversation_item(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _item_id: &str,
        ) -> Result<bool, StoreError> {
            Ok(false)
        }

        async fn conversation_item_position(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _item_id: &str,
        ) -> Result<Option<i64>, StoreError> {
            Ok(None)
        }

        async fn max_item_position(&self, _owner: &StateOwner, _conversation_id: &str) -> Result<i64, StoreError> {
            Ok(0)
        }

        async fn conversation_history(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
        ) -> Result<Option<Vec<serde_json::Value>>, StoreError> {
            Ok(None)
        }

        async fn create_items_and_sync_messages(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _items: &[ConversationItemRecord],
        ) -> Result<(), StoreError> {
            Ok(())
        }

        async fn delete_item_and_sync_messages(
            &self,
            _owner: &StateOwner,
            _conversation_id: &str,
            _item_id: &str,
        ) -> Result<bool, StoreError> {
            Ok(false)
        }
    }

    /// Offline factory that deduplicates pool and namespace by `database_url`,
    /// counts builds, and provisions a [`SpyBackend`]. Identifies as `sqlite` so
    /// `backend: sqlite` filter configs route to it.
    struct ReaperFactory {
        builds: Arc<AtomicUsize>,
        spy: Arc<SweepSpy>,
        gate: Option<Arc<SweepGate>>,
    }

    #[async_trait]
    impl StoreBackendFactory for ReaperFactory {
        fn backend_id(&self) -> &str {
            "sqlite"
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let url = config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| BackendError::Config("missing database_url".to_owned()))?;
            Ok(EffectiveConfigKey::new(url))
        }

        fn namespace_key(&self, config: &serde_json::Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            Ok(config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .map(BackendNamespaceKey::new))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(ProvisionedBackend {
                backend: Arc::new(SpyBackend {
                    spy: Arc::clone(&self.spy),
                    gate: self.gate.as_ref().map(Arc::clone),
                }),
                retire: Arc::new(CountingRetire(Arc::new(AtomicUsize::new(0)))),
            })
        }
    }

    /// A reaper factory that provisions a spy-backed store but reports NO
    /// physical namespace key, mirroring a private in-memory SQLite store at
    /// `sqlite::memory:`. The coordinator must still run a reaper for its live
    /// backend instead of silently dropping the configured `ttl`.
    struct NamespacelessReaperFactory {
        builds: Arc<AtomicUsize>,
        spy: Arc<SweepSpy>,
    }

    #[async_trait]
    impl StoreBackendFactory for NamespacelessReaperFactory {
        fn backend_id(&self) -> &str {
            "sqlite"
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let url = config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| BackendError::Config("missing database_url".to_owned()))?;
            Ok(EffectiveConfigKey::new(url))
        }

        fn namespace_key(&self, _config: &serde_json::Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            // A private in-memory store has no physical namespace identity.
            Ok(None)
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(ProvisionedBackend {
                backend: Arc::new(SpyBackend {
                    spy: Arc::clone(&self.spy),
                    gate: None,
                }),
                retire: Arc::new(CountingRetire(Arc::new(AtomicUsize::new(0)))),
            })
        }
    }

    /// A reaper factory that mints a FRESH spy-backed store on every build and
    /// records each spy in order, so a test can tell which backend generation a
    /// reaper is sweeping. Its `effective_key` folds in `pool` (so a pool-only
    /// reload rebuilds the backend), while its `namespace_key` keys only on
    /// `database_url` (so the reaper namespace is unchanged across that reload).
    struct SwappingReaperFactory {
        builds: Arc<AtomicUsize>,
        spies: Arc<std::sync::Mutex<Vec<Arc<SweepSpy>>>>,
    }

    #[async_trait]
    impl StoreBackendFactory for SwappingReaperFactory {
        fn backend_id(&self) -> &str {
            "sqlite"
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let field = |name| config.get(name).map_or_else(String::new, serde_json::Value::to_string);
            Ok(EffectiveConfigKey::new(format!(
                "{}|{}",
                field("database_url"),
                field("pool")
            )))
        }

        fn namespace_key(&self, config: &serde_json::Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            Ok(config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .map(BackendNamespaceKey::new))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            let spy = Arc::new(SweepSpy::default());
            match self.spies.lock() {
                Ok(mut guard) => guard.push(Arc::clone(&spy)),
                Err(poisoned) => poisoned.into_inner().push(Arc::clone(&spy)),
            }
            Ok(ProvisionedBackend {
                backend: Arc::new(SpyBackend { spy, gate: None }),
                retire: Arc::new(CountingRetire(Arc::new(AtomicUsize::new(0)))),
            })
        }
    }

    /// A reaper factory whose `effective_key` folds in the table set (so distinct
    /// table sets are distinct backends) while its `namespace_key` keys only on the
    /// database path (so distinct table sets share one physical namespace). This is
    /// the DEFECT 1 shape: keying reapers on the namespace dropped the second
    /// backend's tables; keying on the effective key covers every backend.
    struct TableAwareReaperFactory {
        builds: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl StoreBackendFactory for TableAwareReaperFactory {
        fn backend_id(&self) -> &str {
            "sqlite"
        }

        fn effective_key(&self, config: &serde_json::Value) -> Result<EffectiveConfigKey, BackendError> {
            let field = |name| config.get(name).map_or_else(String::new, serde_json::Value::to_string);
            Ok(EffectiveConfigKey::new(format!(
                "{}|{}|{}",
                field("database_url"),
                field("responses_table"),
                field("conversations_table"),
            )))
        }

        fn namespace_key(&self, config: &serde_json::Value) -> Result<Option<BackendNamespaceKey>, BackendError> {
            Ok(config
                .get("database_url")
                .and_then(serde_json::Value::as_str)
                .map(BackendNamespaceKey::new))
        }

        async fn build(&self, _config: &serde_json::Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire(Arc::new(AtomicUsize::new(0)))),
            })
        }
    }

    /// A single-listener response-store config that opts into the TTL reaper.
    fn reaper_store_config(database_url: &str, ttl: &str, sweep_interval: &str) -> Config {
        Config::from_yaml(&format!(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        ttl: {ttl}
        sweep_interval: {sweep_interval}
"#
        ))
        .expect("reaper store config")
    }

    /// A single-listener response-store config that opts into the TTL reaper and
    /// sets an explicit pool size, so a reload can change only the pool to force a
    /// backend rebuild while keeping the reaper policy and namespace.
    fn reaper_store_config_with_pool(
        database_url: &str,
        ttl: &str,
        sweep_interval: &str,
        max_connections: u32,
    ) -> Config {
        Config::from_yaml(&format!(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        ttl: {ttl}
        sweep_interval: {sweep_interval}
        pool:
          max_connections: {max_connections}
"#
        ))
        .expect("reaper store config with pool")
    }

    /// Current Unix time in whole seconds, for cutoff-window assertions.
    fn unix_now_secs() -> i64 {
        i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_secs(),
        )
        .expect("seconds fit i64")
    }

    /// Build a manually wired provisioner around the spying reaper factory.
    fn reaper_service(
        config: &Config,
        factory: Arc<dyn StoreBackendFactory>,
    ) -> (
        Arc<StoreProvisionService>,
        StoreReloadHandle,
        watch::Receiver<StoreReadiness>,
    ) {
        let factories = vec![factory];
        let mut plans = build_listener_store_plans(config);
        validate_and_deduplicate_refs(&mut plans, &factories, None, None).expect("initial plans");
        let (readiness, readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (commands, command_rx) = mpsc::unbounded_channel();
        let service = Arc::new(StoreProvisionService {
            cache: Arc::new(BackendCache::new(factories.clone())),
            factories,
            plans,
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        });
        (service, StoreReloadHandle { commands }, readiness_rx)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ttl_only_reload_reuses_pool_and_swaps_policy() {
        let builds = Arc::new(AtomicUsize::new(0));
        let spy = Arc::new(SweepSpy::default());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReaperFactory {
            builds: Arc::clone(&builds),
            spy: Arc::clone(&spy),
            gate: None,
        });
        // Long ttl, then a reload that changes only the ttl to a short one.
        let initial = reaper_store_config("sqlite:///ttl_swap.db", "100000s", "50ms");
        let (service, handle, mut readiness_rx) = reaper_service(&initial, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");
        // The long initial policy must sweep before the reload.
        tokio::time::timeout(Duration::from_secs(2), async {
            while spy.cutoffs().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the initial policy should sweep");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "initial generation opens one pool");

        let next = reaper_store_config("sqlite:///ttl_swap.db", "10s", "50ms");
        let prepared = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&next)
        })
        .await
        .expect("prepare task")
        .expect("ttl-only replacement should provision");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "a ttl-only reload must reuse the existing pool, not rebuild it"
        );
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.commit(prepared, Vec::new())
        })
        .await
        .expect("commit task")
        .expect("replacement generation should commit");

        // Commit awaits the old reaper's handoff, so every cutoff recorded past
        // this marker belongs to the new short policy alone.
        let marker = spy.cutoffs().len();

        // After the handoff only the new short policy may sweep.
        tokio::time::timeout(Duration::from_secs(2), async {
            while spy.cutoffs().len() <= marker + 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the swapped-in policy should sweep");
        let now = unix_now_secs();
        let cutoffs = spy.cutoffs();
        let (before, after) = cutoffs.split_at(marker);
        assert!(
            before.iter().any(|&cutoff| cutoff <= now - 1000),
            "the long initial policy must have swept with an old cutoff"
        );
        assert!(
            after.iter().all(|&cutoff| cutoff > now - 1000),
            "only the new short policy sweeps after commit; the old reaper must be gone"
        );

        shutdown_tx.send(true).expect("service should receive shutdown");
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pool_reload_swaps_reaper_backend() {
        // A reload that keeps the same ttl but changes only the pool rebuilds the
        // backend (new effective_key → new Arc, old pool retired/closed) while the
        // reaper namespace is unchanged. The coordinator must re-spawn the reaper
        // on the new backend; otherwise it keeps sweeping the closed pool forever.
        let builds = Arc::new(AtomicUsize::new(0));
        let spies = Arc::new(std::sync::Mutex::new(Vec::<Arc<SweepSpy>>::new()));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(SwappingReaperFactory {
            builds: Arc::clone(&builds),
            spies: Arc::clone(&spies),
        });
        let initial = reaper_store_config_with_pool("sqlite:///pool_swap.db", "100000s", "50ms", 4);
        let (service, handle, mut readiness_rx) = reaper_service(&initial, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");
        assert_eq!(builds.load(Ordering::SeqCst), 1, "gen A opens one pool");

        // Gen A's backend (the first spy) must sweep before the reload.
        let gen_a = {
            let guard = spies.lock().expect("spies");
            Arc::clone(guard.first().expect("gen A spy recorded"))
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            while gen_a.committed() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("gen A reaper should sweep");

        // Reload with the SAME ttl but a different pool: the backend is rebuilt.
        let next = reaper_store_config_with_pool("sqlite:///pool_swap.db", "100000s", "50ms", 8);
        let prepared = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&next)
        })
        .await
        .expect("prepare task")
        .expect("pool-only reload should provision");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "a pool change must rebuild the backend under a new effective key"
        );
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.commit(prepared, Vec::new())
        })
        .await
        .expect("commit task")
        .expect("pool-only reload should commit");

        // Commit awaited the old reaper's handoff, so gen A is already stopped.
        let frozen_gen_a = gen_a.committed();

        // The reaper must now sweep the freshly built backend (the second spy).
        let gen_b = {
            let guard = spies.lock().expect("spies");
            Arc::clone(guard.get(1).expect("gen B spy recorded"))
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            while gen_b.committed() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the reaper must sweep the new backend after a pool reload");

        // Gen A's (closed) backend must no longer be swept.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            gen_a.committed(),
            frozen_gen_a,
            "the reaper must stop sweeping the retired backend after the swap"
        );
        assert!(gen_b.committed() > 0, "the reaper must sweep the freshly built backend");

        shutdown_tx.send(true).expect("service should receive shutdown");
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn namespaceless_store_still_gets_reaper() {
        // A store whose factory reports no physical namespace key (a private
        // in-memory SQLite store at `sqlite::memory:`) still has a live backend
        // with rows to sweep. The coordinator must synthesize a per-backend key
        // and run a reaper for it instead of silently disabling the configured
        // ttl.
        let builds = Arc::new(AtomicUsize::new(0));
        let spy = Arc::new(SweepSpy::default());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(NamespacelessReaperFactory {
            builds: Arc::clone(&builds),
            spy: Arc::clone(&spy),
        });
        let config = reaper_store_config("sqlite::memory:", "100000s", "50ms");
        let (service, _handle, mut readiness_rx) = reaper_service(&config, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "the namespaceless store opens one backend"
        );

        // The reaper must run and sweep even though the backend reports no
        // physical namespace key.
        tokio::time::timeout(Duration::from_secs(2), async {
            while spy.committed() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("a namespaceless store with a ttl must still be swept");

        shutdown_tx.send(true).expect("service should receive shutdown");
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[tokio::test]
    async fn independent_table_sets_same_db_each_get_a_reaper() {
        // Two store filters target the SAME file-backed SQLite database with
        // DIFFERENT, non-coalescing table sets, both with a ttl. They share one
        // physical namespace — the DEFECT 1 shape that dropped the second backend
        // when reapers keyed on the namespace — but resolve to DISTINCT effective
        // keys. Keying reapers on backend identity must yield TWO reapers, one per
        // cached backend, so neither store's tables are dropped from the sweep.
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let builds = Arc::new(AtomicUsize::new(0));
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareReaperFactory {
            builds: Arc::clone(&builds),
        });
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations_a
        ttl: 2s
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_b
        conversations_table: conversations_b
        ttl: 2s
"#
        ))
        .expect("two stores sharing one database with distinct table sets");

        let factories = vec![Arc::clone(&factory)];
        let mut plans = build_listener_store_plans(&config);
        // Different table sets → different effective keys → not a policy conflict.
        validate_and_deduplicate_refs(&mut plans, &factories, None, None)
            .expect("distinct table sets are independent and must not conflict");
        assert_eq!(
            plans.iter().map(|plan| plan.reaper_bindings.len()).sum::<usize>(),
            2,
            "both stores declare a ttl, so both contribute a reaper binding"
        );

        let cache = Arc::new(BackendCache::new(factories.clone()));
        let mut leases = Vec::new();
        for plan in &plans {
            leases.push(
                cache
                    .provision_into(&plan.refs, &plan.registry)
                    .await
                    .expect("each distinct table set provisions its own backend"),
            );
        }
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "distinct table sets open two backends"
        );

        let (readiness, _readiness_rx) = watch::channel(StoreReadiness::Pending);
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let service = StoreProvisionService {
            cache,
            factories,
            plans: plans.clone(),
            readiness,
            commands: AsyncMutex::new(Some(command_rx)),
        };

        let desired = service.desired_reapers(&plans);
        assert_eq!(
            desired.len(),
            2,
            "every distinct backend under the coordinator must get its own reaper; \
             neither store's tables may be dropped from the sweep"
        );

        for lease in leases {
            lease.release().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aborted_reload_never_applies_candidate_policy() {
        let builds = Arc::new(AtomicUsize::new(0));
        let spy = Arc::new(SweepSpy::default());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReaperFactory {
            builds: Arc::clone(&builds),
            spy: Arc::clone(&spy),
            gate: None,
        });
        // No active TTL, so no reaper runs unless a committed policy activates one.
        let initial = single_store_config("sqlite", "sqlite:///abort_policy.db");
        let (service, handle, mut readiness_rx) = reaper_service(&initial, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("initial generation should become ready");

        // Prepare a short-ttl candidate, then abort it without committing.
        let candidate = reaper_store_config("sqlite:///abort_policy.db", "10s", "50ms");
        let prepared = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&candidate)
        })
        .await
        .expect("prepare task")
        .expect("candidate should provision");
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.abort(prepared)
        })
        .await
        .expect("abort task")
        .expect("candidate generation should abort");

        // Give any erroneously-spawned reaper time to sweep, then prove none did.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let now = unix_now_secs();
        assert!(
            spy.cutoffs().iter().all(|&cutoff| cutoff <= now - 1000),
            "an aborted reload must never sweep with the candidate's short cutoff"
        );

        shutdown_tx.send(true).expect("service should receive shutdown");
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_during_active_batch_stops_promptly() {
        let builds = Arc::new(AtomicUsize::new(0));
        let spy = Arc::new(SweepSpy::default());
        let gate = Arc::new(SweepGate {
            entered: Notify::new(),
            proceed: Notify::new(),
            armed: AtomicBool::new(true),
        });
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReaperFactory {
            builds: Arc::clone(&builds),
            spy: Arc::clone(&spy),
            gate: Some(Arc::clone(&gate)),
        });
        // One immediate sweep (long interval) that blocks inside the batch.
        let config = reaper_store_config("sqlite:///shutdown_batch.db", "1d", "1h");
        let (service, _handle, mut readiness_rx) = reaper_service(&config, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("generation should become ready");

        tokio::time::timeout(Duration::from_secs(2), gate.entered.notified())
            .await
            .expect("the reaper should enter its first batch");

        shutdown_tx.send(true).expect("service should receive shutdown");
        // Release the in-flight batch so it commits and the reaper can stop.
        gate.proceed.notify_one();

        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("run() must return promptly after the in-flight batch finishes")
            .expect("provisioner task");
        assert_eq!(
            spy.committed(),
            1,
            "the batch that was in flight at shutdown must still commit"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unchanged_policy_reload_does_not_restart_reaper() {
        let builds = Arc::new(AtomicUsize::new(0));
        let spy = Arc::new(SweepSpy::default());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(ReaperFactory {
            builds: Arc::clone(&builds),
            spy: Arc::clone(&spy),
            gate: None,
        });
        // Long interval, so a running reaper sweeps exactly once (the immediate
        // first tick). A restart would sweep again immediately.
        let config = reaper_store_config("sqlite:///unchanged_policy.db", "1d", "1h");
        let (service, handle, mut readiness_rx) = reaper_service(&config, factory);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let running = tokio::spawn({
            let service = Arc::clone(&service);
            async move { service.start(shutdown_rx).await }
        });
        readiness_rx
            .wait_for(|state| *state == StoreReadiness::Ready)
            .await
            .expect("generation should become ready");
        tokio::time::timeout(Duration::from_secs(2), async {
            while spy.committed() < 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the reaper should sweep once on startup");

        // Reload an identical config: same namespace, same policy.
        let unchanged = reaper_store_config("sqlite:///unchanged_policy.db", "1d", "1h");
        let prepared = tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.prepare(&unchanged)
        })
        .await
        .expect("prepare task")
        .expect("identical reload should provision");
        tokio::task::spawn_blocking({
            let handle = handle.clone();
            move || handle.commit(prepared, Vec::new())
        })
        .await
        .expect("commit task")
        .expect("identical reload should commit");

        // A restart would have swept again immediately; an unchanged reaper stays idle.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            spy.committed(),
            1,
            "an unchanged-policy reload must not cancel and respawn the reaper"
        );
        assert_eq!(builds.load(Ordering::SeqCst), 1, "an unchanged reload reuses the pool");

        shutdown_tx.send(true).expect("service should receive shutdown");
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .expect("provisioner should stop promptly")
            .expect("provisioner task");
    }

    #[test]
    fn conflicting_policies_same_namespace_fail_validation() {
        #[cfg(feature = "store-sqlite")]
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        #[cfg(feature = "store-sqlite")]
        let (backend, database_url) = (
            "sqlite",
            format!("sqlite://{}?mode=rwc", directory.path().join("conflict.db").display()),
        );
        #[cfg(all(not(feature = "store-sqlite"), feature = "store-postgres"))]
        let (backend, database_url) = ("postgres", "postgresql://user:password@8.8.8.8/store".to_owned());

        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        ttl: 1h
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: {backend}
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
        ttl: 2h
"#
        ))
        .expect("conflicting ttl config");

        let error = build_store_wiring(&config)
            .err()
            .expect("two policies on one physical store must fail startup");
        assert!(
            error.to_string().contains("conflicting reaper policies"),
            "actionable conflict error: {error}"
        );
    }

    #[test]
    #[cfg(feature = "store-sqlite")]
    fn conflicting_policies_shared_memory_backend_fail_validation() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
        ttl: 1h
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
        ttl: 2h
"#,
        )
        .expect("conflicting ttl in-memory config");

        let error = build_store_wiring(&config)
            .err()
            .expect("two policies on one shared in-memory backend must fail startup");
        assert!(
            error.to_string().contains("conflicting reaper policies"),
            "actionable conflict error for in-memory backend: {error}"
        );
    }

    #[test]
    #[cfg(feature = "store-sqlite")]
    fn matching_policies_shared_memory_backend_succeed() {
        let config = Config::from_yaml(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
        ttl: 1h
        sweep_interval: 5m
        sweep_batch_size: 100
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "sqlite::memory:"
        responses_table: responses
        conversations_table: conversations
        ttl: 1h
        sweep_interval: 5m
        sweep_batch_size: 100
"#,
        )
        .expect("matching ttl in-memory config");

        let result = build_store_wiring(&config);
        assert!(
            result.is_ok(),
            "two bindings with the same policy on one shared in-memory backend should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn conflicting_policies_on_shared_table_across_table_sets_fail_validation() {
        // Two response stores on the SAME physical database with DIFFERENT
        // responses_table but the SAME conversations_table, each with a different
        // ttl. They resolve to DISTINCT effective keys (two reapers, full coverage),
        // yet both sweep the shared conversations table, so the 1h policy would
        // delete conversations before the 7d retention expires. The overlap must be
        // rejected even though the effective keys differ.
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareReaperFactory {
            builds: Arc::new(AtomicUsize::new(0)),
        });
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations
        ttl: 1h
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_b
        conversations_table: conversations
        ttl: 7d
"#
        ))
        .expect("two stores sharing one conversations table");

        let mut plans = build_listener_store_plans(&config);
        let error = validate_and_deduplicate_refs(&mut plans, &[factory], None, None)
            .expect_err("a shared physical table with conflicting ttls must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("conflicting reaper policies"),
            "actionable shared-table conflict error: {message}"
        );
        assert!(
            message.contains("conversations"),
            "the error must name the shared physical table: {message}"
        );
    }

    #[test]
    fn independent_policies_on_disjoint_table_sets_same_db_succeed() {
        // Two response stores on the SAME physical database with fully DISJOINT
        // table sets and different ttls. No physical table is shared, so the two
        // retention policies are independent and must not be rejected.
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareReaperFactory {
            builds: Arc::new(AtomicUsize::new(0)),
        });
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations_a
        ttl: 1h
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_b
        conversations_table: conversations_b
        ttl: 7d
"#
        ))
        .expect("two stores with disjoint table sets");

        let mut plans = build_listener_store_plans(&config);
        validate_and_deduplicate_refs(&mut plans, &[factory], None, None)
            .expect("disjoint table sets on one database may set independent retention policies");
    }

    #[test]
    fn ttl_enabled_store_sharing_table_with_ttl_disabled_store_fails_validation() {
        // Two response stores on the SAME physical database with DIFFERENT
        // responses_table but the SAME conversations_table. One sets a ttl; the other
        // sets none. The enabled store's reaper would sweep the shared conversations
        // table, deleting the disabled store's rows even though omitting ttl is
        // documented to retain them indefinitely. The TTL-disabled store is absent
        // from reaper_bindings, so the overlap must be caught by walking every ref.
        let directory = tempfile::tempdir().expect("temporary SQLite directory");
        let database_url = format!("sqlite://{}?mode=rwc", directory.path().join("shared.db").display());
        let factory: Arc<dyn StoreBackendFactory> = Arc::new(TableAwareReaperFactory {
            builds: Arc::new(AtomicUsize::new(0)),
        });
        let config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: a
    address: "127.0.0.1:8080"
    filter_chains: [chain-a]
  - name: b
    address: "127.0.0.1:8081"
    filter_chains: [chain-b]
filter_chains:
  - name: chain-a
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_a
        conversations_table: conversations
        ttl: 1h
  - name: chain-b
    filters:
      - filter: openai_responses_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses_b
        conversations_table: conversations
"#
        ))
        .expect("one ttl store sharing a conversations table with a ttl-less store");

        let mut plans = build_listener_store_plans(&config);
        let error = validate_and_deduplicate_refs(&mut plans, &[factory], None, None)
            .expect_err("a ttl reaper sharing a physical table with a ttl-less store must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("must retain its rows"),
            "the error must explain the ttl-less store's rows are at risk: {message}"
        );
        assert!(
            message.contains("conversations"),
            "the error must name the shared physical table: {message}"
        );
    }
}
