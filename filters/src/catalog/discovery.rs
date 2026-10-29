// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Upstream model discovery with stale-while-revalidate caching.
//!
//! There is no long-lived background task. The first catalog request fetches
//! every upstream inline. After that, a request that finds the snapshot due
//! for refresh is still answered from it and starts one background refresh,
//! so catalog requests never wait on an upstream once warm. An upstream that
//! fails keeps its last known list, and a failed refresh is retried sooner
//! than a successful one.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use arc_swap::ArcSwapOption;
use bytes::Bytes;
use http::{HeaderMap, HeaderName, HeaderValue, Method, header};
use praxis_ai_apis::{
    callout_target::AddressPolicy,
    subrequest::{SubRequest, SubRequestClient, UrlResolutionPolicy, address_policy_hook, map_url_error},
};
use praxis_filter::FilterError;
use serde::Deserialize;
use tokio::{sync::Mutex, time::Instant};
use tracing::{debug, warn};

use super::{
    config::{ModelCatalogConfig, UnconfiguredModels, UpstreamConfig, validate_id},
    listing::{Catalog, Overlay, UpstreamListing, UpstreamModel},
};

/// Budget for one upstream model list round trip.
const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest upstream model list accepted.
const MAX_LIST_BYTES: usize = 4 * 1024 * 1024;

/// Upper bound on the delay before retrying after a failed refresh.
const FAILURE_RETRY: Duration = Duration::from_secs(5);

// -----------------------------------------------------------------------------
// Discovery
// -----------------------------------------------------------------------------

/// Upstream model lists and the catalog built from them.
pub(super) struct Discovery {
    /// Entries that decorate discovered models.
    overlay: Arc<Overlay>,
    /// Sub-request client for model list fetches.
    client: SubRequestClient,
    /// Upstreams in config order; earlier upstreams win duplicate IDs.
    upstreams: Vec<Upstream>,
    /// Delay between successful refreshes.
    refresh_interval: Duration,
    /// Whether upstream models without an entry are listed.
    show_unconfigured: bool,
    /// Latest snapshot; `None` until the first refresh completes.
    snapshot: ArcSwapOption<Snapshot>,
    /// Serializes refreshes so concurrent requests trigger one fetch.
    refresh_lock: Mutex<()>,
    /// Whether a background refresh is already scheduled.
    refreshing: AtomicBool,
}

/// One resolved upstream model list endpoint.
struct Upstream {
    /// Name used in logs.
    name: String,
    /// Model list URL.
    url: String,
    /// Provider reported for models without an entry.
    provider: Option<String>,
    /// Request headers, credentials marked sensitive.
    headers: HeaderMap,
    /// Connect-time address policy.
    address_policy: AddressPolicy,
}

/// Upstream lists as of one refresh, and the catalog built from them.
pub(super) struct Snapshot {
    /// Catalog to serve; `None` while no upstream has ever answered.
    pub catalog: Option<Arc<Catalog>>,
    /// Last known list per upstream, indexed like [`Discovery::upstreams`].
    lists: Vec<Option<Arc<[UpstreamModel]>>>,
    /// When the snapshot is due for refresh.
    refresh_at: Instant,
}

/// Body of an OpenAI-compatible `GET /v1/models` response.
#[derive(Deserialize)]
struct ModelList {
    /// Listed models.
    data: Vec<UpstreamModel>,
}

impl Discovery {
    /// Resolve upstream credentials and validate their targets.
    pub(super) fn new(
        cfg: &mut ModelCatalogConfig,
        overlay: Arc<Overlay>,
        client: SubRequestClient,
    ) -> Result<Self, FilterError> {
        let upstreams = std::mem::take(&mut cfg.upstreams)
            .into_iter()
            .map(Upstream::new)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            overlay,
            client,
            upstreams,
            refresh_interval: Duration::from_secs(cfg.refresh_interval_secs),
            show_unconfigured: cfg.unconfigured_models == UnconfiguredModels::Show,
            snapshot: ArcSwapOption::empty(),
            refresh_lock: Mutex::new(()),
            refreshing: AtomicBool::new(false),
        })
    }

    /// Override the refresh interval (tests only).
    #[cfg(test)]
    pub(super) fn set_refresh_interval(&mut self, interval: Duration) {
        self.refresh_interval = interval;
    }

    /// Current snapshot without triggering a refresh (tests only).
    #[cfg(test)]
    pub(super) fn current(&self) -> Option<Arc<Snapshot>> {
        self.snapshot.load_full()
    }

    /// Current snapshot, fetching inline only when none exists yet.
    pub(super) async fn snapshot(self: &Arc<Self>) -> Option<Arc<Snapshot>> {
        if let Some(snapshot) = self.snapshot.load_full() {
            if Instant::now() >= snapshot.refresh_at {
                self.spawn_refresh();
            }
            return Some(snapshot);
        }
        self.refresh_if_due().await;
        self.snapshot.load_full()
    }

    /// Start one background refresh unless one is already scheduled.
    fn spawn_refresh(self: &Arc<Self>) {
        if self.refreshing.swap(true, Ordering::AcqRel) {
            return;
        }
        let discovery = Arc::clone(self);
        drop(tokio::spawn(async move {
            discovery.refresh_if_due().await;
            discovery.refreshing.store(false, Ordering::Release);
        }));
    }

    /// Refresh unless another caller did while this one waited for the lock.
    pub(super) async fn refresh_if_due(&self) {
        let _guard = self.refresh_lock.lock().await;
        let previous = self.snapshot.load_full();
        if previous
            .as_ref()
            .is_some_and(|snapshot| Instant::now() < snapshot.refresh_at)
        {
            return;
        }
        let snapshot = self.refresh(previous.as_deref()).await;
        self.snapshot.store(Some(Arc::new(snapshot)));
    }

    /// Fetch every upstream and rebuild the catalog.
    async fn refresh(&self, previous: Option<&Snapshot>) -> Snapshot {
        let mut lists = Vec::with_capacity(self.upstreams.len());
        let mut failed = false;
        for (index, upstream) in self.upstreams.iter().enumerate() {
            match upstream.fetch(&self.client).await {
                Ok(models) => lists.push(Some(Arc::from(models))),
                Err(error) => {
                    warn!(upstream = %upstream.name, %error, "model_catalog: model list refresh failed; keeping last known list");
                    failed = true;
                    lists.push(previous.and_then(|snapshot| snapshot.lists.get(index).cloned().flatten()));
                },
            }
        }

        let catalog = self
            .build_catalog(&lists)
            .or_else(|| previous.and_then(|p| p.catalog.clone()));
        let delay = if failed {
            FAILURE_RETRY.min(self.refresh_interval)
        } else {
            self.refresh_interval
        };
        Snapshot {
            catalog,
            lists,
            refresh_at: Instant::now() + delay,
        }
    }

    /// Build a catalog from the known lists, or `None` when there are none.
    fn build_catalog(&self, lists: &[Option<Arc<[UpstreamModel]>>]) -> Option<Arc<Catalog>> {
        let listings: Vec<UpstreamListing<'_>> = self
            .upstreams
            .iter()
            .zip(lists)
            .filter_map(|(upstream, list)| {
                list.as_deref().map(|models| UpstreamListing {
                    provider: upstream.provider.as_deref(),
                    models,
                })
            })
            .collect();
        if listings.is_empty() {
            return None;
        }
        match Catalog::from_upstreams(&self.overlay, &listings, self.show_unconfigured) {
            Ok(catalog) => Some(Arc::new(catalog)),
            Err(error) => {
                warn!(%error, "model_catalog: failed to build catalog from upstream lists");
                None
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Upstream
// -----------------------------------------------------------------------------

impl Upstream {
    /// Resolve headers from config and validate the target URL.
    fn new(cfg: UpstreamConfig) -> Result<Self, FilterError> {
        let address_policy = AddressPolicy::from_allow_private(cfg.allow_private);
        praxis_ai_apis::callout_target::validate_configured_http_target("model_catalog", &cfg.url, address_policy)?;

        let mut headers = HeaderMap::with_capacity(cfg.headers.len() + 1);
        if let Some(var) = &cfg.bearer_token_env_var {
            let token = read_env(var, &cfg.name)?;
            headers.insert(
                header::AUTHORIZATION,
                sensitive_value(&format!("Bearer {token}"), &cfg.name)?,
            );
        }
        for entry in cfg.headers {
            let name = HeaderName::try_from(entry.name.as_str()).map_err(|error| -> FilterError {
                format!("model_catalog: invalid header name '{}': {error}", entry.name).into()
            })?;
            let value = match (entry.value, entry.value_env_var) {
                (Some(value), _) => HeaderValue::try_from(value).map_err(|error| -> FilterError {
                    format!("model_catalog: invalid value for header '{name}': {error}").into()
                })?,
                (None, Some(var)) => sensitive_value(&read_env(&var, &cfg.name)?, &cfg.name)?,
                (None, None) => continue,
            };
            headers.insert(name, value);
        }

        Ok(Self {
            name: cfg.name,
            url: cfg.url,
            provider: cfg.provider,
            headers,
            address_policy,
        })
    }

    /// Fetch and parse the upstream model list.
    async fn fetch(&self, client: &SubRequestClient) -> Result<Vec<UpstreamModel>, String> {
        // `SubRequest` owns its headers; this copy is per refresh, not per
        // catalog request.
        let request = SubRequest {
            method: Method::GET,
            uri: http::Uri::default(),
            headers: self.headers.clone(),
            body: Bytes::new(),
        };
        let response = client
            .execute_url(
                &self.url,
                request,
                address_policy_hook(self.address_policy),
                UrlResolutionPolicy::OperatorCached,
                FETCH_TIMEOUT,
                MAX_LIST_BYTES,
                None,
            )
            .await
            .map_err(|error| format!("request failed: {}", map_url_error(error)))?;
        if !(200..300).contains(&response.status) {
            return Err(format!("upstream returned HTTP status {}", response.status));
        }
        self.parse(&response.body)
    }

    /// Parse a model list body, dropping IDs the catalog cannot serve.
    fn parse(&self, body: &[u8]) -> Result<Vec<UpstreamModel>, String> {
        let list: ModelList = serde_json::from_slice(body).map_err(|error| format!("invalid model list: {error}"))?;
        let mut models = list.data;
        models.retain(|model| {
            let valid = validate_id(&model.id).is_ok();
            if !valid {
                debug!(upstream = %self.name, "model_catalog: skipping upstream model with an unroutable id");
            }
            valid
        });
        Ok(models)
    }
}

/// Read a credential from the environment at build time.
fn read_env(var: &str, upstream: &str) -> Result<String, FilterError> {
    std::env::var(var).map_err(|error| -> FilterError {
        format!("model_catalog: environment variable '{var}' for upstream '{upstream}' is not set: {error}").into()
    })
}

/// Build a header value that is redacted from debug output.
fn sensitive_value(value: &str, upstream: &str) -> Result<HeaderValue, FilterError> {
    let mut value = HeaderValue::from_str(value).map_err(|error| -> FilterError {
        format!("model_catalog: credential for upstream '{upstream}' is not a valid header value: {error}").into()
    })?;
    value.set_sensitive(true);
    Ok(value)
}
