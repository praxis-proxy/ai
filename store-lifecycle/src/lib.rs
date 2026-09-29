// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Backend provisioning and lifecycle for praxis-ai persisted state.
//!
//! Backend-agnostic and crypto-free: it depends only on the praxis-ai-store
//! interface and routes inline configuration to injected [`StoreBackendFactory`]
//! implementations that binaries supply. It builds and validates backends
//! eagerly at pipeline construction, deduplicates identical effective
//! configurations onto one pooled backend, and reuses or retires backends
//! across a configuration reload.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use dashmap::{DashMap, mapref::entry::Entry as MapEntry};
use praxis_ai_store::{
    BackendError, EffectiveConfigKey, PersistedStateBackend, ProvisionedBackend, RetireBackend, StoreBackendFactory,
    StoreRegistry,
};
use serde_json::Value;
use tokio::sync::Mutex;

/// A store a pipeline requires: a registry name, the backend kind, and the
/// inline configuration the factory interprets.
#[derive(Clone, Debug)]
pub struct StoreRef {
    /// Registry key the transport layer resolves at request time.
    pub name: Arc<str>,
    /// Backend kind, routed to the factory with the matching id.
    pub backend_id: Arc<str>,
    /// Inline configuration passed verbatim to the factory.
    pub config: Value,
}

/// Why provisioning failed at pipeline construction.
///
/// Distinct from an unknown-filter error: a named-but-unprovisionable backend
/// fails the build here with an actionable cause rather than a first-traffic
/// error.
#[derive(Debug)]
pub enum ProvisionError {
    /// No factory was injected for the reference's backend id.
    UnknownBackend {
        /// The store reference name.
        name: Arc<str>,
        /// The unrecognized backend id.
        backend_id: Arc<str>,
    },
    /// The factory could not build the backend.
    Backend {
        /// The store reference name.
        name: Arc<str>,
        /// The underlying backend error: unavailable after retries, or config.
        source: BackendError,
    },
}

impl std::fmt::Display for ProvisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownBackend { name, backend_id } => {
                write!(f, "store '{name}' names unknown backend '{backend_id}'")
            },
            Self::Backend { name, source } => write!(f, "store '{name}' backend unavailable: {source}"),
        }
    }
}

impl std::error::Error for ProvisionError {}

/// Bounded retry budget for a transient build failure.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Maximum build attempts before a transient failure becomes unavailable.
    pub max_attempts: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_attempts: 3 }
    }
}

/// A cached backend and the count of live generations referencing it.
struct CachedBackend {
    /// Shared combined backend handle.
    backend: Arc<dyn PersistedStateBackend>,
    /// Hook invoked once the last generation releases this backend.
    retire: Arc<dyn RetireBackend>,
    /// Number of live pipeline generations holding this backend.
    refcount: usize,
}

/// Effective identity of a backend: its kind plus the factory's dedup key.
#[derive(Clone, Eq, Hash, PartialEq)]
struct CacheKey {
    /// Backend kind (the factory id).
    backend_id: Arc<str>,
    /// Factory-computed key over the connection-determining config.
    config: EffectiveConfigKey,
}

/// One cold-build singleflight and the terminal result shared by its waiters.
struct InitializerState {
    /// Winner error retained until every participant in this wave observes it.
    outcome: Mutex<Option<BackendError>>,
    /// Futures that joined this wave, including ones waiting for the mutex.
    participants: AtomicUsize,
}

/// Shared handle to one cold-build wave.
type Initializer = Arc<InitializerState>;

/// One participant in a cold-build wave.
///
/// Dropping this guard also handles task cancellation. The last participant
/// removes the wave only when no new participant joined before cleanup.
struct InitializerParticipant {
    /// Effective backend key whose wave this future joined.
    key: CacheKey,
    /// Initialization state shared by every participant in the wave.
    initializer: Initializer,
    /// Map from which the last participant removes the completed wave.
    initializers: Arc<DashMap<CacheKey, Initializer>>,
}

impl Drop for InitializerParticipant {
    fn drop(&mut self) {
        if self.initializer.participants.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if let MapEntry::Occupied(entry) = self.initializers.entry(self.key.clone())
            && Arc::ptr_eq(entry.get(), &self.initializer)
            && entry.get().participants.load(Ordering::Acquire) == 0
        {
            entry.remove();
        }
    }
}

/// Process-wide, refcounted cache of provisioned backends.
///
/// Identical effective configurations share one backend and one pool. A backend
/// is retired once no live pipeline generation holds it.
pub struct BackendCache {
    /// Shared, process-wide cache of provisioned backends.
    entries: Arc<DashMap<CacheKey, CachedBackend>>,
    /// Per-key initialization state that coalesces concurrent cold misses.
    /// A terminal winner error is retained until its queued waiters observe it.
    initializers: Arc<DashMap<CacheKey, Initializer>>,
    /// Injected factories, keyed by backend id.
    factories: HashMap<Arc<str>, Arc<dyn StoreBackendFactory>>,
    /// Bounded retry budget for transient build failures.
    retry: RetryPolicy,
}

/// The set of backends one pipeline generation holds.
///
/// Retirement is async, so a dropped lease is not enough: the owner calls
/// [`BackendLease::release`] once the old pipeline has drained.
pub struct BackendLease {
    /// Shared handle to the process-wide cache.
    entries: Arc<DashMap<CacheKey, CachedBackend>>,
    /// The distinct cache keys this generation holds one refcount on.
    keys: Vec<CacheKey>,
}

/// A provisioned generation: the registry the transport layer resolves against,
/// and the lease that keeps its backends alive until release.
pub struct Provisioned {
    /// Registry mapping each store name to its backend.
    pub registry: StoreRegistry,
    /// Lease to release when this generation retires.
    pub lease: BackendLease,
}

impl BackendCache {
    /// Return whether the exact effective backend for `store_ref` is currently
    /// held by an active or prepared generation.
    ///
    /// # Errors
    ///
    /// Returns [`ProvisionError::UnknownBackend`] when no injected factory owns
    /// the reference, or [`ProvisionError::Backend`] when its config is invalid.
    pub fn contains(&self, store_ref: &StoreRef) -> Result<bool, ProvisionError> {
        let factory = self
            .factories
            .get(&store_ref.backend_id)
            .ok_or_else(|| ProvisionError::UnknownBackend {
                name: Arc::clone(&store_ref.name),
                backend_id: Arc::clone(&store_ref.backend_id),
            })?;
        let config = factory
            .effective_key(&store_ref.config)
            .map_err(|source| ProvisionError::Backend {
                name: Arc::clone(&store_ref.name),
                source,
            })?;
        Ok(self.entries.contains_key(&CacheKey {
            backend_id: Arc::clone(&store_ref.backend_id),
            config,
        }))
    }

    /// Build a cache from injected factories, keyed by backend id.
    #[must_use]
    pub fn new(factories: Vec<Arc<dyn StoreBackendFactory>>) -> Self {
        Self::with_retry(factories, RetryPolicy::default())
    }

    /// Build a cache with a specific transient-retry budget.
    #[must_use]
    pub fn with_retry(factories: Vec<Arc<dyn StoreBackendFactory>>, retry: RetryPolicy) -> Self {
        let factories = factories.into_iter().map(|f| (Arc::from(f.backend_id()), f)).collect();
        Self {
            entries: Arc::new(DashMap::new()),
            initializers: Arc::new(DashMap::new()),
            factories,
            retry,
        }
    }

    /// Provision every store reference, reusing cached backends and building the
    /// rest eagerly, and return a registry plus the generation's lease.
    ///
    /// # Errors
    ///
    /// Returns [`ProvisionError`] when a reference names an unknown backend or a
    /// backend cannot be built (config invalid, or unavailable after the retry
    /// budget). References already provisioned in this call are released on
    /// error so no refcount leaks.
    pub async fn provision(&self, refs: &[StoreRef]) -> Result<Provisioned, ProvisionError> {
        let registry = StoreRegistry::new();
        let lease = self.provision_into(refs, &registry).await?;
        Ok(Provisioned { registry, lease })
    }

    /// Provision every store reference into `registry`, reusing cached backends
    /// and building the rest eagerly, and return the generation's lease.
    ///
    /// The serving runtime installs an empty registry into each pipeline before
    /// it exists, then registers backends into that same map through a clone, so
    /// this populates the shared handle rather than a fresh one. [`provision`]
    /// wraps this over a private registry.
    ///
    /// [`provision`]: Self::provision
    ///
    /// # Errors
    ///
    /// Same as [`provision`]. References already provisioned in this call are
    /// released on error so no refcount leaks.
    #[expect(
        clippy::too_many_lines,
        reason = "resolve-and-register loop plus rollback on either failure path"
    )]
    pub async fn provision_into(
        &self,
        refs: &[StoreRef],
        registry: &StoreRegistry,
    ) -> Result<BackendLease, ProvisionError> {
        let mut held: Vec<CacheKey> = Vec::new();
        let mut this_gen: HashSet<CacheKey> = HashSet::new();
        // Names registered in this call, deregistered on error so a later
        // reference's failure leaves the registry as it was, not with earlier
        // entries that make a retry fail as a duplicate.
        let mut registered: Vec<Arc<str>> = Vec::new();

        for r in refs {
            let backend = match self.resolve(r, &mut this_gen, &mut held).await {
                Ok(backend) => backend,
                Err(e) => {
                    for name in &registered {
                        registry.deregister(name);
                    }
                    release_into(&self.entries, &held).await;
                    return Err(e);
                },
            };
            if registry.register(&r.name, backend).is_err() {
                for name in &registered {
                    registry.deregister(name);
                }
                release_into(&self.entries, &held).await;
                return Err(ProvisionError::Backend {
                    name: Arc::clone(&r.name),
                    source: BackendError::Config(format!("duplicate store name '{}'", r.name)),
                });
            }
            registered.push(Arc::clone(&r.name));
        }

        Ok(BackendLease {
            entries: Arc::clone(&self.entries),
            keys: held,
        })
    }

    /// Validate every store reference's configuration without building a pool or
    /// touching a runtime, so a malformed or unknown-backend config fails at
    /// pipeline construction rather than at first traffic.
    ///
    /// Pool creation stays on the serving runtime: sqlx pools bind to the
    /// runtime that opens them, and the server owns that runtime, which is not
    /// reachable at synchronous pipeline construction. This method performs
    /// eager config validation; the server provisioner then opens every pool
    /// before store readiness admits traffic.
    ///
    /// # Errors
    ///
    /// [`ProvisionError::UnknownBackend`] for a reference whose backend id has no
    /// injected factory, or [`ProvisionError::Backend`] when a factory rejects
    /// the configuration.
    pub fn validate(&self, refs: &[StoreRef]) -> Result<(), ProvisionError> {
        for r in refs {
            let factory = self
                .factories
                .get(&r.backend_id)
                .ok_or_else(|| ProvisionError::UnknownBackend {
                    name: Arc::clone(&r.name),
                    backend_id: Arc::clone(&r.backend_id),
                })?;
            factory
                .validate_config(&r.config)
                .map_err(|source| ProvisionError::Backend {
                    name: Arc::clone(&r.name),
                    source,
                })?;
        }
        Ok(())
    }

    /// Resolve one reference to a backend, reusing a cached entry or building.
    #[expect(
        clippy::too_many_lines,
        reason = "cache lookup, per-key singleflight, build, and registration form one resolution path"
    )]
    async fn resolve(
        &self,
        r: &StoreRef,
        this_gen: &mut HashSet<CacheKey>,
        held: &mut Vec<CacheKey>,
    ) -> Result<Arc<dyn PersistedStateBackend>, ProvisionError> {
        let factory = self
            .factories
            .get(&r.backend_id)
            .ok_or_else(|| ProvisionError::UnknownBackend {
                name: Arc::clone(&r.name),
                backend_id: Arc::clone(&r.backend_id),
            })?;
        let config_key = factory
            .effective_key(&r.config)
            .map_err(|source| ProvisionError::Backend {
                name: Arc::clone(&r.name),
                source,
            })?;
        let ckey = CacheKey {
            backend_id: Arc::clone(&r.backend_id),
            config: config_key,
        };

        if let Some(backend) = self.reuse(&ckey, this_gen, held) {
            return Ok(backend);
        }

        // Serialize cold initialization per effective key. A listener that
        // arrived while the winner was building rechecks the cache after the
        // lock, so it reuses that pool instead of opening a duplicate or
        // propagating a redundant build failure.
        let participant = self.initializer(&ckey);
        let mut guard = participant.initializer.outcome.lock().await;
        if let Some(backend) = self.reuse(&ckey, this_gen, held) {
            drop(guard);
            return Ok(backend);
        }
        if let Some(source) = guard.as_ref().cloned() {
            drop(guard);
            return Err(ProvisionError::Backend {
                name: Arc::clone(&r.name),
                source,
            });
        }

        let built = match self.build_with_retry(factory.as_ref(), &r.config).await {
            Ok(built) => built,
            Err(source) => {
                *guard = Some(source.clone());
                drop(guard);
                return Err(ProvisionError::Backend {
                    name: Arc::clone(&r.name),
                    source,
                });
            },
        };
        let backend = self.insert_or_reuse(ckey.clone(), built, this_gen, held).await;
        drop(guard);
        Ok(backend)
    }

    /// Return the shared initialization lock for one effective backend key.
    fn initializer(&self, ckey: &CacheKey) -> InitializerParticipant {
        let entry = self.initializers.entry(ckey.clone()).or_insert_with(|| {
            Arc::new(InitializerState {
                outcome: Mutex::new(None),
                participants: AtomicUsize::new(0),
            })
        });
        let initializer = Arc::clone(entry.value());
        initializer.participants.fetch_add(1, Ordering::Relaxed);
        drop(entry);
        InitializerParticipant {
            key: ckey.clone(),
            initializer,
            initializers: Arc::clone(&self.initializers),
        }
    }

    /// Reuse a cached backend if present, taking one refcount per generation.
    fn reuse(
        &self,
        ckey: &CacheKey,
        this_gen: &mut HashSet<CacheKey>,
        held: &mut Vec<CacheKey>,
    ) -> Option<Arc<dyn PersistedStateBackend>> {
        let mut entry = self.entries.get_mut(ckey)?;
        let backend = Arc::clone(&entry.backend);
        if this_gen.insert(ckey.clone()) {
            entry.refcount += 1;
            held.push(ckey.clone());
        }
        drop(entry);
        Some(backend)
    }

    /// Insert a freshly built backend, or reuse one a concurrent provision won.
    async fn insert_or_reuse(
        &self,
        ckey: CacheKey,
        built: ProvisionedBackend,
        this_gen: &mut HashSet<CacheKey>,
        held: &mut Vec<CacheKey>,
    ) -> Arc<dyn PersistedStateBackend> {
        match self.entries.entry(ckey.clone()) {
            MapEntry::Occupied(mut occ) => {
                let backend = Arc::clone(&occ.get().backend);
                if this_gen.insert(ckey) {
                    occ.get_mut().refcount += 1;
                    held.push(occ.key().clone());
                }
                drop(occ);
                // Retire the duplicate we built while another provision won.
                built.retire.retire().await;
                backend
            },
            MapEntry::Vacant(vac) => {
                let backend = Arc::clone(&built.backend);
                this_gen.insert(ckey.clone());
                held.push(ckey);
                vac.insert(CachedBackend {
                    backend: built.backend,
                    retire: built.retire,
                    refcount: 1,
                });
                backend
            },
        }
    }

    /// Attempt a build, retrying a transient failure within the budget.
    async fn build_with_retry(
        &self,
        factory: &dyn StoreBackendFactory,
        config: &Value,
    ) -> Result<ProvisionedBackend, BackendError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match factory.build(config).await {
                Ok(built) => return Ok(built),
                Err(BackendError::Transient(_)) if attempt < self.retry.max_attempts => {
                    // Retry immediately: an immediate-failure mode (connection
                    // refused) burns the budget fast, but max_attempts bounds it
                    // and this runs once at provision, off any request path.
                },
                Err(BackendError::Transient(m)) => {
                    return Err(BackendError::Unavailable(format!(
                        "transient failure persisted after {attempt} attempts: {m}"
                    )));
                },
                Err(other) => return Err(other),
            }
        }
    }
}

impl BackendLease {
    /// Release this generation's hold, retiring any backend no live generation
    /// references. Call after the old pipeline has drained.
    pub async fn release(self) {
        release_into(&self.entries, &self.keys).await;
    }
}

/// Decrement each key's refcount and retire the backends that reach zero.
///
/// The decrement and the zero-check removal run under one entry lock so a
/// concurrent reuse cannot resurrect a backend between the two steps; the async
/// retire runs after the lock is released.
async fn release_into(entries: &DashMap<CacheKey, CachedBackend>, keys: &[CacheKey]) {
    let mut to_retire: Vec<Arc<dyn RetireBackend>> = Vec::new();
    for ckey in keys {
        if let MapEntry::Occupied(mut occ) = entries.entry(ckey.clone()) {
            occ.get_mut().refcount = occ.get().refcount.saturating_sub(1);
            if occ.get().refcount == 0 {
                to_retire.push(occ.remove().retire);
            }
        }
    }
    for retire in to_retire {
        retire.retire().await;
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::panic, reason = "tests")]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use async_trait::async_trait;
    use praxis_ai_store::memory::InMemoryStore;
    use serde_json::json;
    use tokio::sync::{Barrier, Semaphore};

    use super::*;

    /// Retirement hook that records how many times it fired.
    struct CountingRetire {
        /// Shared count of retire calls.
        retires: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl RetireBackend for CountingRetire {
        async fn retire(&self) {
            self.retires.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// What a fake factory does on each build.
    enum Behavior {
        /// Always build successfully.
        Ok,
        /// Always fail permanently (fails the build immediately).
        Unavailable,
        /// Fail transiently the given number of times, then succeed.
        TransientThenOk(AtomicUsize),
    }

    /// Test factory: counts builds and retirements and keys on the config url.
    struct FakeFactory {
        /// Backend id this fake answers to.
        id: &'static str,
        /// Shared count of build calls.
        builds: Arc<AtomicUsize>,
        /// Shared count of retire calls (handed to each built backend).
        retires: Arc<AtomicUsize>,
        /// What each build does.
        behavior: Behavior,
    }

    /// Factory whose first cold build succeeds and any duplicate fails.
    struct SingleflightFactory {
        /// Number of build calls that reached the factory.
        builds: Arc<AtomicUsize>,
        /// Gate keeping the first build cold while a second caller arrives.
        release: Semaphore,
        /// Whether build one succeeds and later builds fail, or vice versa.
        first_build_succeeds: bool,
    }

    impl FakeFactory {
        fn new(id: &'static str, behavior: Behavior) -> Arc<Self> {
            Arc::new(Self {
                id,
                builds: Arc::new(AtomicUsize::new(0)),
                retires: Arc::new(AtomicUsize::new(0)),
                behavior,
            })
        }

        fn ok_backend(&self) -> ProvisionedBackend {
            ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire {
                    retires: Arc::clone(&self.retires),
                }),
            }
        }
    }

    #[async_trait]
    impl StoreBackendFactory for FakeFactory {
        fn backend_id(&self) -> &str {
            self.id
        }

        fn effective_key(&self, config: &Value) -> Result<EffectiveConfigKey, BackendError> {
            let url = config
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| BackendError::Config("missing url".to_owned()))?;
            Ok(EffectiveConfigKey::new(format!("{}:{url}", self.id)))
        }

        async fn build(&self, _config: &Value) -> Result<ProvisionedBackend, BackendError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            match &self.behavior {
                Behavior::Ok => Ok(self.ok_backend()),
                Behavior::Unavailable => Err(BackendError::Unavailable("permanent init failure".to_owned())),
                Behavior::TransientThenOk(remaining) => {
                    if remaining.load(Ordering::SeqCst) > 0 {
                        remaining.fetch_sub(1, Ordering::SeqCst);
                        Err(BackendError::Transient("connect blip".to_owned()))
                    } else {
                        Ok(self.ok_backend())
                    }
                },
            }
        }
    }

    #[async_trait]
    impl StoreBackendFactory for SingleflightFactory {
        fn backend_id(&self) -> &str {
            "singleflight"
        }

        fn effective_key(&self, _config: &Value) -> Result<EffectiveConfigKey, BackendError> {
            Ok(EffectiveConfigKey::new("shared"))
        }

        async fn build(&self, _config: &Value) -> Result<ProvisionedBackend, BackendError> {
            let attempt = self.builds.fetch_add(1, Ordering::SeqCst) + 1;
            self.release
                .acquire()
                .await
                .map_err(|_closed| BackendError::Unavailable("test release gate closed".to_owned()))?
                .forget();
            if (attempt == 1) != self.first_build_succeeds {
                return Err(BackendError::Unavailable("singleflight winner failed".to_owned()));
            }
            Ok(ProvisionedBackend {
                backend: Arc::new(InMemoryStore::new()),
                retire: Arc::new(CountingRetire {
                    retires: Arc::new(AtomicUsize::new(0)),
                }),
            })
        }
    }

    /// Erase a fake factory to a trait object (implicit unsizing, no cast).
    fn as_dyn(factory: Arc<FakeFactory>) -> Arc<dyn StoreBackendFactory> {
        factory
    }

    /// Erase a singleflight factory while retaining its concrete test handle.
    fn singleflight_as_dyn(factory: Arc<SingleflightFactory>) -> Arc<dyn StoreBackendFactory> {
        factory
    }

    fn store_ref(name: &str, backend_id: &str, url: &str) -> StoreRef {
        StoreRef {
            name: Arc::from(name),
            backend_id: Arc::from(backend_id),
            config: json!({ "url": url }),
        }
    }

    #[tokio::test]
    async fn initial_load_builds_and_registers() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);

        let provisioned = cache
            .provision(&[store_ref("default", "fake", "a")])
            .await
            .expect("initial load succeeds");

        assert_eq!(factory.builds.load(Ordering::SeqCst), 1);
        assert!(provisioned.registry.contains("default"));
    }

    #[tokio::test]
    async fn reload_reuses_unchanged_config_without_rebuild() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);
        let refs = [store_ref("default", "fake", "a")];

        let gen1 = cache.provision(&refs).await.expect("gen1");
        let gen2 = cache.provision(&refs).await.expect("gen2 reuses");

        // Built once, reused for the second generation; nothing retired yet.
        assert_eq!(factory.builds.load(Ordering::SeqCst), 1);
        assert_eq!(factory.retires.load(Ordering::SeqCst), 0);

        // Releasing the first generation leaves the backend live for gen2.
        gen1.lease.release().await;
        assert_eq!(factory.retires.load(Ordering::SeqCst), 0);
        assert!(gen2.registry.contains("default"));
    }

    #[tokio::test]
    async fn last_release_retires_the_backend() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);
        let refs = [store_ref("default", "fake", "a")];

        let gen1 = cache.provision(&refs).await.expect("gen1");
        let gen2 = cache.provision(&refs).await.expect("gen2");

        gen1.lease.release().await;
        assert_eq!(factory.retires.load(Ordering::SeqCst), 0, "still held by gen2");
        gen2.lease.release().await;
        assert_eq!(factory.retires.load(Ordering::SeqCst), 1, "retired after last holder");
    }

    #[tokio::test]
    async fn changed_config_retires_the_old_backend_on_release() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);

        let gen1 = cache
            .provision(&[store_ref("default", "fake", "a")])
            .await
            .expect("gen1 on url a");
        let gen2 = cache
            .provision(&[store_ref("default", "fake", "b")])
            .await
            .expect("gen2 on url b");

        // Two distinct effective configs, two backends built.
        assert_eq!(factory.builds.load(Ordering::SeqCst), 2);
        // Retiring gen1 releases the "a" backend that gen2 no longer references.
        gen1.lease.release().await;
        assert_eq!(factory.retires.load(Ordering::SeqCst), 1);
        drop(gen2);
    }

    #[tokio::test]
    async fn unavailable_backend_fails_the_build() {
        let factory = FakeFactory::new("fake", Behavior::Unavailable);
        let cache = BackendCache::new(vec![factory]);

        let result = cache.provision(&[store_ref("default", "fake", "a")]).await;
        let Err(err) = result else {
            panic!("expected a permanent-failure error")
        };

        match err {
            ProvisionError::Backend {
                source: BackendError::Unavailable(_),
                ..
            } => {},
            other => panic!("expected backend-unavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_backend_id_is_distinct_from_a_build_failure() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![factory]);

        let result = cache.provision(&[store_ref("default", "missing", "a")]).await;
        let Err(err) = result else {
            panic!("expected an unknown-backend error")
        };

        assert!(matches!(err, ProvisionError::UnknownBackend { .. }));
    }

    #[tokio::test]
    async fn transient_failure_retries_then_builds_within_budget() {
        let factory = FakeFactory::new("fake", Behavior::TransientThenOk(AtomicUsize::new(2)));
        let cache = BackendCache::with_retry(vec![as_dyn(Arc::clone(&factory))], RetryPolicy { max_attempts: 3 });

        let provisioned = cache
            .provision(&[store_ref("default", "fake", "a")])
            .await
            .expect("succeeds on the third attempt");

        assert_eq!(factory.builds.load(Ordering::SeqCst), 3, "two transient, then ok");
        assert!(provisioned.registry.contains("default"));
    }

    #[tokio::test]
    async fn transient_failure_past_budget_becomes_unavailable() {
        let factory = FakeFactory::new("fake", Behavior::TransientThenOk(AtomicUsize::new(5)));
        let cache = BackendCache::with_retry(vec![factory], RetryPolicy { max_attempts: 3 });

        let result = cache.provision(&[store_ref("default", "fake", "a")]).await;
        let Err(err) = result else {
            panic!("expected budget exhaustion")
        };

        assert!(matches!(
            err,
            ProvisionError::Backend {
                source: BackendError::Unavailable(_),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn dedup_shares_one_pool_for_matching_config() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);

        let provisioned = cache
            .provision(&[
                store_ref("responses", "fake", "a"),
                store_ref("conversations", "fake", "a"),
            ])
            .await
            .expect("both references provision");

        // One effective config -> one pool -> one build, shared by both names.
        assert_eq!(factory.builds.load(Ordering::SeqCst), 1);
        assert!(provisioned.registry.contains("responses"));
        assert!(provisioned.registry.contains("conversations"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[expect(
        clippy::too_many_lines,
        reason = "the concurrency regression coordinates two cold provisions and verifies both leases"
    )]
    async fn concurrent_cold_misses_share_the_winning_build() {
        let builds = Arc::new(AtomicUsize::new(0));
        let factory = Arc::new(SingleflightFactory {
            builds: Arc::clone(&builds),
            release: Semaphore::new(0),
            first_build_succeeds: true,
        });
        let factory_dyn = singleflight_as_dyn(Arc::clone(&factory));
        let cache = Arc::new(BackendCache::new(vec![factory_dyn]));
        let first = tokio::spawn({
            let cache = Arc::clone(&cache);
            async move { cache.provision(&[store_ref("first", "singleflight", "unused")]).await }
        });
        while builds.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }

        let second_started = Arc::new(Barrier::new(2));
        let second = tokio::spawn({
            let cache = Arc::clone(&cache);
            let second_started = Arc::clone(&second_started);
            async move {
                second_started.wait().await;
                cache.provision(&[store_ref("second", "singleflight", "unused")]).await
            }
        });
        second_started.wait().await;
        let mut waiter_observed = false;
        for _ in 0..1_000 {
            waiter_observed = cache
                .initializers
                .iter()
                .any(|entry| entry.value().participants.load(Ordering::Acquire) >= 2);
            if waiter_observed {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(waiter_observed, "second provision must wait on the cold build");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "only the singleflight winner may build"
        );

        // One permit releases the only factory call; the waiting provision then
        // rechecks the cache and shares its backend.
        factory.release.add_permits(1);
        let first = first.await.expect("first provision task").expect("winning provision");
        let second = second.await.expect("second provision task").expect("waiting provision");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(first.registry.contains("first"));
        assert!(second.registry.contains("second"));
        first.lease.release().await;
        second.lease.release().await;
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the concurrent failure wave and later retry are one lifecycle proof"
    )]
    async fn concurrent_waiters_share_winner_failure_then_later_call_retries() {
        let builds = Arc::new(AtomicUsize::new(0));
        let factory = Arc::new(SingleflightFactory {
            builds: Arc::clone(&builds),
            release: Semaphore::new(0),
            first_build_succeeds: false,
        });
        let cache = Arc::new(BackendCache::new(vec![singleflight_as_dyn(Arc::clone(&factory))]));
        let first = tokio::spawn({
            let cache = Arc::clone(&cache);
            async move { cache.provision(&[store_ref("first", "singleflight", "unused")]).await }
        });
        while builds.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        let second = tokio::spawn({
            let cache = Arc::clone(&cache);
            async move { cache.provision(&[store_ref("second", "singleflight", "unused")]).await }
        });
        let mut waiter_observed = false;
        for _ in 0..1_000 {
            waiter_observed = cache
                .initializers
                .iter()
                .any(|entry| entry.value().participants.load(Ordering::Acquire) >= 2);
            if waiter_observed {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(waiter_observed, "second provision must queue behind the failing winner");

        factory.release.add_permits(1);
        let first_error = first
            .await
            .expect("first provision task")
            .err()
            .expect("winner must fail");
        let second_error = second
            .await
            .expect("second provision task")
            .err()
            .expect("waiter must share failure");
        assert!(first_error.to_string().contains("singleflight winner failed"));
        assert!(second_error.to_string().contains("singleflight winner failed"));
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "a waiter must not start a second build"
        );

        factory.release.add_permits(1);
        let recovered = cache
            .provision(&[store_ref("later", "singleflight", "unused")])
            .await
            .expect("a later generation may retry after the failed wave");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        recovered.lease.release().await;
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "the cancellation and immediate recovery sequence form one lifecycle proof"
    )]
    async fn cancelled_waiter_does_not_pin_a_failed_singleflight_wave() {
        let builds = Arc::new(AtomicUsize::new(0));
        let factory = Arc::new(SingleflightFactory {
            builds: Arc::clone(&builds),
            release: Semaphore::new(0),
            first_build_succeeds: false,
        });
        let cache = Arc::new(BackendCache::new(vec![singleflight_as_dyn(Arc::clone(&factory))]));
        let winner = tokio::spawn({
            let cache = Arc::clone(&cache);
            async move { cache.provision(&[store_ref("winner", "singleflight", "unused")]).await }
        });
        while builds.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        let waiter = tokio::spawn({
            let cache = Arc::clone(&cache);
            async move { cache.provision(&[store_ref("waiter", "singleflight", "unused")]).await }
        });
        for _ in 0..1_000 {
            if cache
                .initializers
                .iter()
                .any(|entry| entry.value().participants.load(Ordering::Acquire) >= 2)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            cache
                .initializers
                .iter()
                .map(|entry| entry.value().participants.load(Ordering::Acquire))
                .max(),
            Some(2),
            "waiter must join the winner's wave"
        );

        waiter.abort();
        assert!(matches!(waiter.await, Err(error) if error.is_cancelled()));
        factory.release.add_permits(1);
        assert!(winner.await.expect("winner task").is_err());

        factory.release.add_permits(1);
        let recovered = cache
            .provision(&[store_ref("later", "singleflight", "unused")])
            .await
            .expect("a later generation retries immediately after cancellation");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        recovered.lease.release().await;
    }

    #[tokio::test]
    async fn provision_into_populates_a_shared_registry() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(factory)]);
        let shared = StoreRegistry::new();

        // A clone shares the backing map, as the serving-runtime install does.
        let installed = shared.clone();
        let lease = cache
            .provision_into(&[store_ref("default", "fake", "a")], &shared)
            .await
            .expect("provision into shared registry");

        assert!(
            installed.contains("default"),
            "backend visible through the shared clone"
        );
        lease.release().await;
    }

    #[test]
    fn validate_accepts_a_well_formed_config() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(factory)]);
        cache
            .validate(&[store_ref("default", "fake", "a")])
            .expect("valid config passes construction-time validation");
    }

    #[test]
    fn validate_rejects_an_unknown_backend() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(factory)]);
        let err = cache
            .validate(&[store_ref("default", "missing", "a")])
            .expect_err("unknown backend id fails validation");
        assert!(matches!(err, ProvisionError::UnknownBackend { .. }));
    }

    #[test]
    fn validate_rejects_a_malformed_config_without_building() {
        let factory = FakeFactory::new("fake", Behavior::Ok);
        let cache = BackendCache::new(vec![as_dyn(Arc::clone(&factory))]);
        // Missing the "url" field the fake factory's key requires.
        let bad = StoreRef {
            name: Arc::from("default"),
            backend_id: Arc::from("fake"),
            config: json!({}),
        };
        let err = cache.validate(&[bad]).expect_err("malformed config fails validation");
        assert!(matches!(
            err,
            ProvisionError::Backend {
                source: BackendError::Config(_),
                ..
            }
        ));
        // Validation performs no I/O: nothing is built.
        assert_eq!(factory.builds.load(Ordering::SeqCst), 0);
    }
}
