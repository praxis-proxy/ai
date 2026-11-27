// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-caller tool discovery for the `mcp` broker.
//!
//! A server with `discover: true` contributes the tools its own `tools/list`
//! returns, fetched with the caller's forwarded credential and cached per
//! `(caller, server)`. Servers can show different tools to different callers,
//! so every caller has its own catalog.
//!
//! Each listed tool is vetted before any client sees it: it needs a name and
//! an object input schema, and it must pass operator admission
//! ([`evaluate_operator_tool`]) under the server's `tool_policy`. A static
//! `tools:` entry with the same name then overrides the description or hides
//! the tool. Exposed names keep the server's `tool_prefix`; a later server
//! exposing a name an earlier one already exposes is left out of that
//! caller's catalog.
//!
//! A server whose discovery fails keeps serving the caller's last good tools
//! for it, and the failure is logged and counted. Cached listings are
//! replaced whole, so `tools/list` and `tools/call` routing always read one
//! consistent listing per server.

use std::{
    collections::{HashMap, HashSet},
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::StreamExt as _;
use http::HeaderMap;
use metrics::counter;
use praxis_ai_apis::{
    mcp_admission::{McpToolVerdict, evaluate_operator_tool},
    mcp_session::{MAX_LISTING_RESPONSE_BYTES, McpListingError, McpListingLimits, McpToolPage, paginate_tools},
};
use praxis_core::subrequest::SubRequestClient;
use tracing::{debug, warn};

use super::{
    BackendSessions, BrokerServer,
    backend::{BackendCallout, BackendInit, BackendTarget, McpBackendError},
    config::{
        CatalogTool, InvalidToolPolicy, ServerDiscovery, StatelessDiscovery, ValidatedDiscovery, validate_schema_object,
    },
    sessions::{BackendSessionKey, CredentialFingerprint, credential_fingerprint},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Counter of failed discoveries, labeled by server name and reason.
pub(crate) const METRIC_DISCOVERY_FAILURES_TOTAL: &str = "praxis_ai_mcp_broker_discovery_failures_total";

/// Longest wait before a caller's failed discovery of a server is retried,
/// so requests do not each wait out the budget on a server that is down.
/// Capped by `discovery.max_age_ms`.
const FAILURE_RETRY_INTERVAL: Duration = Duration::from_secs(30);

// -----------------------------------------------------------------------------
// CatalogCaller
// -----------------------------------------------------------------------------

/// Whose catalog a discovered listing belongs to.
///
/// Today this is the fingerprint of the forwarded credential headers, the
/// same value backend sessions are bound to. Once the policy filter publishes
/// a verified identity (#2191 Phase 4), build it from that identity instead,
/// so a refreshed or re-minted token keeps the caller's catalog.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CatalogCaller(CredentialFingerprint);

impl CatalogCaller {
    /// The caller presenting `forwarded` credential headers.
    pub(crate) fn from_forwarded(forwarded: &HeaderMap) -> Self {
        Self(credential_fingerprint(forwarded))
    }
}

// -----------------------------------------------------------------------------
// Catalog Snapshots
// -----------------------------------------------------------------------------

/// One server's tools as one caller sees them, in listing order.
pub(crate) type ServerTools = Arc<[CatalogTool]>;

/// A catalog tool that keeps its server's listing alive.
pub(crate) struct ToolRef {
    /// The listing holding the tool.
    tools: ServerTools,
    /// Index of the tool in `tools`.
    index: usize,
}

impl Deref for ToolRef {
    type Target = CatalogTool;

    #[expect(
        clippy::indexing_slicing,
        reason = "the index comes from a search of the same listing"
    )]
    fn deref(&self) -> &CatalogTool {
        &self.tools[self.index]
    }
}

/// The tools one caller sees, grouped by server in configuration order.
#[derive(Clone, Debug, Default)]
pub(crate) struct CallerCatalog {
    /// Each contributing server's listing.
    servers: Vec<ServerTools>,
}

impl CallerCatalog {
    /// Assemble a catalog from per-server listings in configuration order.
    ///
    /// A server exposing a name an earlier server already exposes is left
    /// out, and the collision is logged with both server names and counted.
    pub(crate) fn assemble(listings: &[ServerTools]) -> Self {
        let mut exposed: HashMap<&str, &str> = HashMap::new();
        let mut kept = Vec::with_capacity(listings.len());
        for listing in listings {
            let collision = listing
                .iter()
                .find_map(|tool| exposed.get(tool.exposed_name.as_str()).map(|owner| (*owner, tool)));
            if let Some((earlier, tool)) = collision {
                warn!(
                    server = %tool.server_name,
                    conflicting_server = %earlier,
                    "mcp discovered tool name already exposed by another server; server left out of the catalog"
                );
                count_failure(&tool.server_name, "name_collision");
                continue;
            }
            for tool in listing.iter() {
                exposed.insert(tool.exposed_name.as_str(), tool.server_name.as_str());
            }
            kept.push(Arc::clone(listing));
        }
        drop(exposed);
        Self { servers: kept }
    }

    /// Every tool, server by server.
    pub(crate) fn tools(&self) -> impl Iterator<Item = &CatalogTool> {
        self.servers.iter().flat_map(|listing| listing.iter())
    }

    /// Number of tools.
    pub(crate) fn len(&self) -> usize {
        self.servers.iter().map(|listing| listing.len()).sum()
    }

    /// The tool exposed as `name`.
    pub(crate) fn find(&self, name: &str) -> Option<ToolRef> {
        self.servers.iter().find_map(|listing| {
            listing
                .iter()
                .position(|tool| tool.exposed_name == name)
                .map(|index| ToolRef {
                    tools: Arc::clone(listing),
                    index,
                })
        })
    }
}

// -----------------------------------------------------------------------------
// DiscoveryCache
// -----------------------------------------------------------------------------

/// One cached listing.
#[derive(Debug)]
struct CacheEntry {
    /// When the listing was discovered.
    discovered_at: Instant,
    /// Last time a request read it.
    last_used: Instant,
    /// No rediscovery before this instant, set after a failure.
    retry_at: Option<Instant>,
    /// The last good listing; `None` when no discovery has succeeded.
    tools: Option<ServerTools>,
}

/// State behind the cache lock.
#[derive(Debug)]
struct CacheInner {
    /// Listings by caller and server index.
    entries: HashMap<(CatalogCaller, usize), CacheEntry>,
    /// Last idle sweep.
    last_sweep: Instant,
}

/// A cached listing and whether it needs rediscovery.
pub(crate) struct Cached {
    /// The last good listing, if any.
    pub tools: Option<ServerTools>,
    /// Whether it was discovered less than `max_age` ago, or a recent
    /// failure defers the next attempt.
    pub fresh: bool,
}

/// Bounded per-caller listing cache.
///
/// A listing older than `max_age` is rediscovered on the caller's next
/// request and served until a rediscovery succeeds. After a failure the next
/// attempt waits [`FAILURE_RETRY_INTERVAL`], at most `max_age`. A listing no request has
/// read for `idle_timeout` is dropped. Inserting into a full cache drops the
/// least recently read listing. The cache is per filter instance, so a
/// reload starts empty.
#[derive(Debug)]
pub(crate) struct DiscoveryCache {
    /// Entries and sweep bookkeeping.
    inner: Mutex<CacheInner>,
    /// Lifetime of a listing no request reads.
    idle_timeout: Duration,
    /// Rediscovery age.
    max_age: Duration,
    /// Maximum number of entries.
    max_entries: usize,
}

impl DiscoveryCache {
    /// Create an empty cache.
    pub(crate) fn new(max_entries: usize, max_age: Duration, idle_timeout: Duration) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                last_sweep: Instant::now(),
            }),
            idle_timeout,
            max_age,
            max_entries,
        }
    }

    /// The caller's listing for `server`, marking it used.
    pub(crate) fn get(&self, caller: CatalogCaller, server: usize, now: Instant) -> Option<Cached> {
        let mut inner = self.lock();
        self.sweep(&mut inner, now);
        let entry = inner.entries.get_mut(&(caller, server))?;
        if self.idle(entry, now) {
            inner.entries.remove(&(caller, server));
            return None;
        }
        entry.last_used = entry.last_used.max(now);
        let cached = Cached {
            tools: entry.tools.as_ref().map(Arc::clone),
            fresh: entry.retry_at.is_some_and(|retry_at| now < retry_at)
                || (entry.tools.is_some() && now.saturating_duration_since(entry.discovered_at) < self.max_age),
        };
        drop(inner);
        Some(cached)
    }

    /// Replace the caller's listing for `server`.
    pub(crate) fn store(&self, caller: CatalogCaller, server: usize, tools: ServerTools, now: Instant) {
        let mut inner = self.lock();
        self.make_room(&mut inner, (caller, server), now);
        inner.entries.insert(
            (caller, server),
            CacheEntry {
                discovered_at: now,
                last_used: now,
                retry_at: None,
                tools: Some(tools),
            },
        );
    }

    /// Record a failed discovery: keep the last good listing, if any, and
    /// defer the caller's next attempt for `server`.
    pub(crate) fn defer(&self, caller: CatalogCaller, server: usize, now: Instant) {
        let retry_at = now.checked_add(FAILURE_RETRY_INTERVAL.min(self.max_age));
        let mut inner = self.lock();
        self.make_room(&mut inner, (caller, server), now);
        let entry = inner.entries.entry((caller, server)).or_insert(CacheEntry {
            discovered_at: now,
            last_used: now,
            retry_at: None,
            tools: None,
        });
        entry.retry_at = retry_at;
        drop(inner);
    }

    /// Sweep idle listings, then free one entry when `key` is new and the
    /// cache is full, dropping the least recently read listing.
    fn make_room(&self, inner: &mut CacheInner, key: (CatalogCaller, usize), now: Instant) {
        self.sweep(inner, now);
        if inner.entries.contains_key(&key) || inner.entries.len() < self.max_entries {
            return;
        }
        let oldest = inner
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| *key);
        if let Some(oldest) = oldest {
            inner.entries.remove(&oldest);
        }
    }

    /// Number of cached listings.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Drop idle listings, at most once per `idle_timeout`.
    fn sweep(&self, inner: &mut CacheInner, now: Instant) {
        if now.saturating_duration_since(inner.last_sweep) < self.idle_timeout {
            return;
        }
        inner.last_sweep = now;
        inner.entries.retain(|_, entry| !self.idle(entry, now));
    }

    /// Whether no request has read `entry` for `idle_timeout`.
    fn idle(&self, entry: &CacheEntry, now: Instant) -> bool {
        now.saturating_duration_since(entry.last_used) >= self.idle_timeout
    }

    /// Lock the entries, recovering from a poisoned lock: every mutation
    /// completes before any fallible step.
    fn lock(&self) -> MutexGuard<'_, CacheInner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------

/// Why one server's discovery failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DiscoveryError {
    /// The session or a `tools/list` exchange failed.
    Backend(McpBackendError),
    /// The server listed more tools than `discovery.max_tools`, or did not
    /// finish within the page cap.
    TooManyTools,
    /// The listing exceeded the cumulative byte cap.
    ListingTooLarge,
    /// The server listed one tool name twice.
    DuplicateTool,
    /// A listed tool is invalid under `invalid_tool_policy: reject_server`.
    InvalidTool,
}

impl DiscoveryError {
    /// Stable, credential-free reason label for logs and metrics.
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            Self::Backend(error) => error.reason(),
            Self::TooManyTools => "too_many_tools",
            Self::ListingTooLarge => "listing_too_large",
            Self::DuplicateTool => "duplicate_tool",
            Self::InvalidTool => "invalid_tool",
        }
    }
}

impl From<McpListingError<McpBackendError>> for DiscoveryError {
    fn from(error: McpListingError<McpBackendError>) -> Self {
        match error {
            McpListingError::Page(error) => Self::Backend(error),
            McpListingError::TooManyTools { .. } => Self::TooManyTools,
            McpListingError::TooLarge { .. } => Self::ListingTooLarge,
        }
    }
}

/// Log and count one failed discovery.
pub(crate) fn record_failure(server: &str, error: &DiscoveryError) {
    warn!(
        server = %server,
        reason = error.reason(),
        "mcp tool discovery failed; the caller keeps its last good tools for the server, if any"
    );
    count_failure(server, error.reason());
}

/// Count one failed discovery.
fn count_failure(server: &str, reason: &'static str) {
    counter!(METRIC_DISCOVERY_FAILURES_TOTAL, "server" => server.to_owned(), "reason" => reason).increment(1);
}

// -----------------------------------------------------------------------------
// Listing
// -----------------------------------------------------------------------------

/// How a `tools/list` page is sent.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), expect(dead_code, reason = "read once McpSessionClient sends tools/list"))]
pub(crate) enum ListingScope<'a> {
    /// Inside the caller's backend session (current profile).
    Session(&'a BackendInit),
    /// Without a session, at the given protocol version (stateless profile).
    Stateless {
        /// Protocol version the request declares.
        protocol_version: &'a str,
    },
}

/// Fetches one `tools/list` page from a backend server.
///
/// Implementations must bound every response they read. Each tool is
/// returned as its JSON object.
#[async_trait]
pub(crate) trait McpToolPages: Send + Sync {
    /// Request the page at `cursor` (`None` for the first page).
    async fn list_tools_page(
        &self,
        target: BackendTarget<'_>,
        scope: ListingScope<'_>,
        cursor: Option<String>,
        callout: &BackendCallout,
    ) -> Result<McpToolPage<serde_json::Value>, McpBackendError>;
}

/// Fetch a server's complete listing within `limits`.
pub(crate) async fn list_server_tools(
    pages: &dyn McpToolPages,
    target: BackendTarget<'_>,
    scope: ListingScope<'_>,
    callout: &BackendCallout,
    limits: McpListingLimits,
) -> Result<Vec<serde_json::Value>, DiscoveryError> {
    let fetch_page = |cursor| pages.list_tools_page(target, scope, cursor, callout);
    Box::pin(paginate_tools(limits, fetch_page))
        .await
        .map_err(DiscoveryError::from)
}

/// Run `discover` for each server index, at most `max_concurrency` at a time.
pub(crate) async fn discover_concurrently<F, Fut, T>(
    servers: Vec<usize>,
    max_concurrency: usize,
    discover: F,
) -> Vec<(usize, T)>
where
    F: Fn(usize) -> Fut + Send + Sync,
    Fut: Future<Output = T> + Send,
    T: Send,
{
    futures::stream::iter(servers)
        .map(|server| {
            let run = discover(server);
            async move { (server, run.await) }
        })
        .buffer_unordered(max_concurrency.max(1))
        .collect()
        .await
}

// -----------------------------------------------------------------------------
// Vetting
// -----------------------------------------------------------------------------

/// Identity of the server a listing came from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ListedServer<'a> {
    /// Index in configuration order.
    pub index: usize,
    /// Configured server name.
    pub name: &'a str,
    /// Backend cluster.
    pub cluster: &'a str,
    /// Backend MCP path.
    pub path: &'a str,
    /// Prefix added to every exposed name.
    pub tool_prefix: Option<&'a str>,
}

/// Build the caller-visible tools of one server from its listing.
///
/// Vets each listed tool, applies the static overrides, and appends static
/// entries the server did not list. Invalid and denied tools are left out;
/// an invalid tool fails the whole listing under
/// `invalid_tool_policy: reject_server`, and a name listed twice always does.
///
/// # Errors
///
/// Returns [`DiscoveryError::DuplicateTool`] or
/// [`DiscoveryError::InvalidTool`].
#[expect(
    clippy::too_many_lines,
    reason = "one pass keeps duplicate detection over every listed name"
)]
pub(crate) fn build_server_tools(
    server: ListedServer<'_>,
    settings: &ServerDiscovery,
    static_tools: &[CatalogTool],
    listing: Vec<serde_json::Value>,
) -> Result<ServerTools, DiscoveryError> {
    let mut seen = HashSet::with_capacity(listing.len());
    let mut tools = Vec::with_capacity(listing.len().saturating_add(static_tools.len()));
    for definition in listing {
        let tool = match vet_tool(server, settings, definition) {
            Ok(tool) => tool,
            Err(rejection) => {
                debug!(server = %server.name, reason = rejection.reason(), "mcp discovered tool left out");
                if matches!(rejection, ToolRejection::Invalid(_))
                    && settings.invalid_tool_policy == InvalidToolPolicy::RejectServer
                {
                    return Err(DiscoveryError::InvalidTool);
                }
                if let ToolRejection::Denied(Some(name)) | ToolRejection::Hidden(name) = rejection {
                    // Still a listed name: a later duplicate must fail.
                    if !seen.insert(name) {
                        return Err(DiscoveryError::DuplicateTool);
                    }
                }
                continue;
            },
        };
        if !seen.insert(tool.original_name.clone()) {
            return Err(DiscoveryError::DuplicateTool);
        }
        tools.push(tool);
    }
    tools.extend(
        static_tools
            .iter()
            .filter(|tool| !seen.contains(&tool.original_name))
            .cloned(),
    );
    Ok(Arc::from(tools))
}

/// Why a listed tool is left out.
enum ToolRejection {
    /// The definition is not a valid MCP tool.
    Invalid(&'static str),
    /// Operator admission denied it; carries the name when it had one.
    Denied(Option<String>),
    /// A static entry hides it.
    Hidden(String),
}

impl ToolRejection {
    /// Stable reason label for logs.
    fn reason(&self) -> &'static str {
        match self {
            Self::Invalid(reason) => reason,
            Self::Denied(_) => "admission_denied",
            Self::Hidden(_) => "hidden",
        }
    }
}

/// Vet one listed tool and turn it into a catalog entry.
#[expect(clippy::too_many_lines, reason = "linear field checks with early returns")]
fn vet_tool(
    server: ListedServer<'_>,
    settings: &ServerDiscovery,
    definition: serde_json::Value,
) -> Result<CatalogTool, ToolRejection> {
    if let McpToolVerdict::Denied(_) = evaluate_operator_tool(settings.tool_policy, &definition) {
        let name = definition
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        return Err(ToolRejection::Denied(name));
    }
    let serde_json::Value::Object(mut fields) = definition else {
        return Err(ToolRejection::Invalid("tool_not_object"));
    };
    let Some(serde_json::Value::String(name)) = fields.remove("name") else {
        return Err(ToolRejection::Invalid("missing_name"));
    };
    if name.is_empty() {
        return Err(ToolRejection::Invalid("missing_name"));
    }
    let override_entry = settings.overrides.get(&name);
    if override_entry.is_some_and(|entry| entry.hidden) {
        return Err(ToolRejection::Hidden(name));
    }
    let Some(input_schema) = fields.remove("inputSchema") else {
        return Err(ToolRejection::Invalid("missing_input_schema"));
    };
    if validate_schema_object("inputSchema", &input_schema).is_err() {
        return Err(ToolRejection::Invalid("invalid_input_schema"));
    }
    let listed_description = match fields.remove("description") {
        None => None,
        Some(serde_json::Value::String(description)) => Some(description),
        Some(_) => return Err(ToolRejection::Invalid("invalid_description")),
    };
    let annotations = fields.remove("annotations");
    let exposed_name = server
        .tool_prefix
        .map_or_else(|| name.clone(), |prefix| format!("{prefix}{name}"));
    Ok(CatalogTool {
        annotations,
        backend_path: server.path.to_owned(),
        cluster: server.cluster.to_owned(),
        description: override_entry
            .and_then(|entry| entry.description.clone())
            .or(listed_description),
        exposed_name,
        extra: fields,
        input_schema,
        original_name: name,
        server_index: server.index,
        server_name: server.name.to_owned(),
    })
}

// -----------------------------------------------------------------------------
// Discovery
// -----------------------------------------------------------------------------

/// Per-caller discovery runtime of one broker instance.
pub(crate) struct Discovery {
    /// Cached listings.
    cache: DiscoveryCache,
    /// Bounds on one server's listing.
    limits: McpListingLimits,
    /// Servers discovered concurrently for one request.
    max_concurrency: usize,
    /// Fetches `tools/list` pages.
    pages: Arc<dyn McpToolPages>,
    /// Per-server settings in configuration order; `None` without discovery.
    servers: Vec<Option<ServerDiscovery>>,
    /// Stateless-profile transport settings.
    stateless: Option<StatelessTransport>,
}

/// Stateless-profile discovery transport.
struct StatelessTransport {
    /// Transport used when the request context exposes no shared client.
    fallback_client: Option<SubRequestClient>,
    /// Client headers copied onto discovery exchanges.
    forward_headers: Vec<http::HeaderName>,
    /// Budget for discovering every server one request needs.
    timeout: Duration,
}


/// How a request's discovery callout is built.
#[derive(Clone, Copy)]
pub(crate) struct CalloutSettings<'a> {
    /// Transport used when the request context exposes no shared client.
    pub fallback_client: Option<&'a SubRequestClient>,
    /// Client headers copied onto discovery exchanges.
    pub forward_headers: &'a [http::HeaderName],
    /// Budget shared by every discovery of the request.
    pub budget: Duration,
}

/// The parts of the broker a discovery run reads.
#[derive(Clone, Copy)]
pub(crate) struct BrokerView<'a> {
    /// Configured servers in configuration order.
    pub servers: &'a [BrokerServer],
    /// Each server's static tools, in configuration order.
    pub static_tools: &'a [ServerTools],
}

/// Request-scoped facts discovery runs with.
pub(crate) struct DiscoveryRun<'a> {
    /// Callout carrying the caller's forwarded credential and the budget.
    pub callout: BackendCallout,
    /// Profile-specific scope.
    pub scope: RunScope<'a>,
}

/// Where a request's discoveries run.
#[derive(Clone, Copy)]
pub(crate) enum RunScope<'a> {
    /// In the caller's backend sessions of `gateway`.
    Session {
        /// Backend session runtime.
        backend: &'a BackendSessions,
        /// Client gateway session id.
        gateway: &'a str,
        /// Protocol version offered to backends.
        version: &'a str,
    },
    /// Without a session, at the request's protocol version.
    Stateless {
        /// Protocol version the request declares.
        version: &'a str,
    },
}

/// The caller's current listings and the servers that need discovery.
pub(crate) struct CallerListings {
    /// Whose listings these are.
    caller: CatalogCaller,
    /// Each server's cached listing, in configuration order.
    listings: Vec<Option<ServerTools>>,
    /// Discovering servers with no listing younger than `max_age`.
    pub needed: Vec<usize>,
}

impl Discovery {
    /// Build the runtime from validated settings.
    pub(crate) fn new(
        settings: &ValidatedDiscovery,
        pages: Arc<dyn McpToolPages>,
        fallback_client: Option<SubRequestClient>,
    ) -> Self {
        Self {
            cache: DiscoveryCache::new(settings.max_entries, settings.max_age, settings.idle_timeout),
            limits: McpListingLimits {
                max_tools: settings.max_tools,
                max_bytes: MAX_LISTING_RESPONSE_BYTES,
            },
            max_concurrency: settings.max_concurrency,
            pages,
            servers: settings.servers.clone(),
            stateless: settings
                .stateless
                .as_ref()
                .map(|stateless: &StatelessDiscovery| StatelessTransport {
                    fallback_client,
                    forward_headers: stateless.forward_headers.clone(),
                    timeout: stateless.timeout,
                }),
        }
    }

    /// Client headers discovery exchanges forward and their shared budget,
    /// with the transport used when the request exposes none. `None` for the
    /// current profile, whose discovery runs with `backend_sessions`.
    pub(crate) fn stateless_transport(&self) -> Option<CalloutSettings<'_>> {
        self.stateless.as_ref().map(|stateless| CalloutSettings {
            fallback_client: stateless.fallback_client.as_ref(),
            forward_headers: &stateless.forward_headers,
            budget: stateless.timeout,
        })
    }

    /// The caller's cached listings, marking them used.
    pub(crate) fn cached(&self, forwarded: &HeaderMap) -> CallerListings {
        let caller = CatalogCaller::from_forwarded(forwarded);
        let now = Instant::now();
        let mut listings = Vec::with_capacity(self.servers.len());
        let mut needed = Vec::new();
        for (index, settings) in self.servers.iter().enumerate() {
            let cached = settings.as_ref().and_then(|_| self.cache.get(caller, index, now));
            if settings.is_some() && !cached.as_ref().is_some_and(|cached| cached.fresh) {
                needed.push(index);
            }
            listings.push(cached.and_then(|cached| cached.tools));
        }
        CallerListings {
            caller,
            listings,
            needed,
        }
    }

    /// Discover every needed server, at most `max_concurrency` at a time,
    /// and cache each listing that passes vetting. A failure is logged and
    /// counted; the server keeps its last good listing.
    pub(crate) async fn discover(&self, view: BrokerView<'_>, run: &DiscoveryRun<'_>, listings: &mut CallerListings) {
        let needed = std::mem::take(&mut listings.needed);
        let results = discover_concurrently(needed, self.max_concurrency, |index| {
            self.discover_server(view, run, index)
        })
        .await;
        let now = Instant::now();
        for (index, result) in results {
            match result.and_then(|listing| self.vet_listing(view, index, listing)) {
                Ok(tools) => {
                    self.cache.store(listings.caller, index, Arc::clone(&tools), now);
                    if let Some(slot) = listings.listings.get_mut(index) {
                        *slot = Some(tools);
                    }
                },
                Err(error) => {
                    self.cache.defer(listings.caller, index, now);
                    record_server_failure(view, index, &error);
                },
            }
        }
    }

    /// Log, count, and defer every needed server as failed with `error`.
    pub(crate) fn fail_needed(&self, view: BrokerView<'_>, listings: &CallerListings, error: &DiscoveryError) {
        let now = Instant::now();
        for index in &listings.needed {
            self.cache.defer(listings.caller, *index, now);
            record_server_failure(view, *index, error);
        }
    }

    /// Discover one server's listing.
    #[expect(clippy::too_many_lines, reason = "one match over the two profiles")]
    async fn discover_server(
        &self,
        view: BrokerView<'_>,
        run: &DiscoveryRun<'_>,
        index: usize,
    ) -> Result<Vec<serde_json::Value>, DiscoveryError> {
        let server = view
            .servers
            .get(index)
            .ok_or(DiscoveryError::Backend(McpBackendError::Transport))?;
        let target = BackendTarget {
            cluster: &server.cluster,
            path: &server.path,
        };
        match run.scope {
            RunScope::Session {
                backend,
                gateway,
                version,
            } => {
                let key = BackendSessionKey {
                    gateway_session: Box::from(gateway),
                    server: index,
                };
                let session = SessionListing {
                    backend,
                    key: &key,
                    version,
                };
                Box::pin(session.list(&*self.pages, target, &run.callout, self.limits)).await
            },
            RunScope::Stateless { version } => {
                let scope = ListingScope::Stateless {
                    protocol_version: version,
                };
                list_server_tools(&*self.pages, target, scope, &run.callout, self.limits).await
            },
        }
    }

    /// Vet a server's listing into its caller-visible tools.
    fn vet_listing(
        &self,
        view: BrokerView<'_>,
        index: usize,
        listing: Vec<serde_json::Value>,
    ) -> Result<ServerTools, DiscoveryError> {
        let (Some(server), Some(Some(settings)), Some(static_tools)) = (
            view.servers.get(index),
            self.servers.get(index),
            view.static_tools.get(index),
        ) else {
            return Err(DiscoveryError::Backend(McpBackendError::Transport));
        };
        let listed = ListedServer {
            index,
            name: &server.name,
            cluster: &server.cluster,
            path: &server.path,
            tool_prefix: server.tool_prefix.as_deref(),
        };
        build_server_tools(listed, settings, static_tools, listing)
    }
}

impl CallerListings {
    /// The catalog from these listings, using static entries for a server
    /// without one.
    pub(crate) fn catalog(&self, view: BrokerView<'_>) -> CallerCatalog {
        let listings: Vec<ServerTools> = view
            .static_tools
            .iter()
            .zip(&self.listings)
            .map(|(static_tools, listing)| Arc::clone(listing.as_ref().unwrap_or(static_tools)))
            .collect();
        CallerCatalog::assemble(&listings)
    }
}

/// Log and count a failed discovery of server `index`.
fn record_server_failure(view: BrokerView<'_>, index: usize, error: &DiscoveryError) {
    let name = view.servers.get(index).map_or("", |server| server.name.as_str());
    record_failure(name, error);
}

/// A listing run inside the caller's backend session for one server.
struct SessionListing<'a> {
    /// Backend session runtime.
    backend: &'a BackendSessions,
    /// Session map key.
    key: &'a BackendSessionKey,
    /// Protocol version offered when a session is opened.
    version: &'a str,
}

impl SessionListing<'_> {
    /// List the server's tools in the session, opening it when there is
    /// none.
    ///
    /// A backend answering `404` has ended the session: the mapping is
    /// dropped and the listing retried once on a new session.
    async fn list(
        &self,
        pages: &dyn McpToolPages,
        target: BackendTarget<'_>,
        callout: &BackendCallout,
        limits: McpListingLimits,
    ) -> Result<Vec<serde_json::Value>, DiscoveryError> {
        let mut retried = false;
        loop {
            let (slot, init) = self
                .backend
                .ensure_session(self.key, target, self.version, callout)
                .await
                .map_err(DiscoveryError::Backend)?;
            let result = list_server_tools(pages, target, ListingScope::Session(&init), callout, limits).await;
            self.backend.store.release(self.key, slot);
            match (&result, init.session_id.as_ref()) {
                (Err(DiscoveryError::Backend(McpBackendError::Rejected { status: 404 })), Some(session_id))
                    if !retried =>
                {
                    self.backend.invalidate_forwarded(
                        self.key,
                        session_id,
                        "mcp backend session expired during discovery; mapping invalidated",
                    );
                    retried = true;
                },
                _ => return result,
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use praxis_ai_apis::mcp_admission::ConnectorToolPolicy;
    use serde_json::json;

    use super::*;
    use crate::agentic::mcp::broker::config::ToolOverride;

    const WEATHER: ListedServer<'static> = ListedServer {
        index: 0,
        name: "weather",
        cluster: "weather-mcp",
        path: "/mcp",
        tool_prefix: Some("weather_"),
    };

    fn settings(tool_policy: ConnectorToolPolicy) -> ServerDiscovery {
        ServerDiscovery {
            invalid_tool_policy: InvalidToolPolicy::FilterOut,
            overrides: HashMap::new(),
            tool_policy,
        }
    }

    fn tool(name: &str) -> serde_json::Value {
        json!({"name": name, "description": "listed", "inputSchema": {"type": "object"}})
    }

    fn names(tools: &ServerTools) -> Vec<&str> {
        tools.iter().map(|tool| tool.exposed_name.as_str()).collect()
    }

    #[test]
    fn listed_tools_keep_the_server_prefix_and_extra_fields() {
        let mut listed = tool("get");
        listed["outputSchema"] = json!({"type": "object"});
        listed["title"] = json!("Get");
        let tools = build_server_tools(WEATHER, &settings(ConnectorToolPolicy::AllowAll), &[], vec![listed]).unwrap();
        assert_eq!(names(&tools), ["weather_get"]);
        assert_eq!(tools[0].original_name, "get");
        assert_eq!(tools[0].description.as_deref(), Some("listed"));
        assert_eq!(tools[0].extra["title"], "Get");
        assert!(tools[0].extra.contains_key("outputSchema"));
        assert!(!tools[0].extra.contains_key("name"), "the backend name is not served");
    }

    #[test]
    fn read_only_policy_leaves_out_tools_without_an_affirmative_hint() {
        let mut read_only = tool("read");
        read_only["annotations"] = json!({"readOnlyHint": true});
        let mut writable = tool("write");
        writable["annotations"] = json!({"readOnlyHint": false});
        let tools = build_server_tools(
            WEATHER,
            &settings(ConnectorToolPolicy::ReadOnly),
            &[],
            vec![read_only, writable, tool("unknown")],
        )
        .unwrap();
        assert_eq!(names(&tools), ["weather_read"]);
    }

    #[test]
    fn invalid_tools_are_left_out_or_fail_the_listing() {
        let listing = || {
            vec![
                tool("ok"),
                json!({"name": "no_schema"}),
                json!({"name": "array", "inputSchema": {"type": "array"}}),
                json!({"inputSchema": {"type": "object"}}),
                json!("not a tool"),
            ]
        };
        let tools = build_server_tools(WEATHER, &settings(ConnectorToolPolicy::AllowAll), &[], listing()).unwrap();
        assert_eq!(names(&tools), ["weather_ok"]);

        let strict = ServerDiscovery {
            invalid_tool_policy: InvalidToolPolicy::RejectServer,
            ..settings(ConnectorToolPolicy::AllowAll)
        };
        assert_eq!(
            build_server_tools(WEATHER, &strict, &[], listing()).unwrap_err(),
            DiscoveryError::InvalidTool
        );
    }

    #[test]
    fn a_name_listed_twice_fails_the_listing() {
        let error = build_server_tools(
            WEATHER,
            &settings(ConnectorToolPolicy::AllowAll),
            &[],
            vec![tool("get"), tool("get")],
        )
        .unwrap_err();
        assert_eq!(error, DiscoveryError::DuplicateTool);
    }

    #[test]
    fn static_entries_override_description_and_visibility() {
        let mut overrides = settings(ConnectorToolPolicy::AllowAll);
        overrides.overrides.insert(
            "get".to_owned(),
            ToolOverride {
                description: Some("operator".to_owned()),
                hidden: false,
            },
        );
        overrides.overrides.insert(
            "secret".to_owned(),
            ToolOverride {
                description: None,
                hidden: true,
            },
        );
        overrides.overrides.insert(
            "plain".to_owned(),
            ToolOverride {
                description: None,
                hidden: false,
            },
        );
        let static_only = CatalogTool {
            annotations: None,
            backend_path: "/mcp".to_owned(),
            cluster: "weather-mcp".to_owned(),
            description: Some("static".to_owned()),
            exposed_name: "weather_offline".to_owned(),
            extra: serde_json::Map::new(),
            input_schema: json!({"type": "object"}),
            original_name: "offline".to_owned(),
            server_index: 0,
            server_name: "weather".to_owned(),
        };
        let tools = build_server_tools(
            WEATHER,
            &overrides,
            &[static_only],
            vec![tool("get"), tool("secret"), tool("plain")],
        )
        .unwrap();
        assert_eq!(names(&tools), ["weather_get", "weather_plain", "weather_offline"]);
        assert_eq!(
            tools[0].description.as_deref(),
            Some("operator"),
            "static description wins"
        );
        assert_eq!(
            tools[1].description.as_deref(),
            Some("listed"),
            "no override keeps the listed one"
        );
        assert_eq!(
            tools[2].description.as_deref(),
            Some("static"),
            "unlisted static entries stay"
        );
    }

    #[test]
    fn a_hidden_name_listed_twice_still_fails() {
        let mut hidden = settings(ConnectorToolPolicy::AllowAll);
        hidden.overrides.insert(
            "get".to_owned(),
            ToolOverride {
                description: None,
                hidden: true,
            },
        );
        assert_eq!(
            build_server_tools(WEATHER, &hidden, &[], vec![tool("get"), tool("get")]).unwrap_err(),
            DiscoveryError::DuplicateTool
        );
    }

    fn listing(server: usize, server_name: &str, names: &[&str]) -> ServerTools {
        let listed = ListedServer {
            index: server,
            name: server_name,
            cluster: "c",
            path: "/mcp",
            tool_prefix: None,
        };
        let tools = names.iter().map(|name| tool(name)).collect();
        build_server_tools(listed, &settings(ConnectorToolPolicy::AllowAll), &[], tools).unwrap()
    }

    #[test]
    fn a_later_server_colliding_with_an_earlier_one_is_left_out() {
        let recorder = metrics_util::debugging::DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let catalog = metrics::with_local_recorder(&recorder, || {
            CallerCatalog::assemble(&[
                listing(0, "first", &["search"]),
                listing(1, "second", &["fetch", "search"]),
                listing(2, "third", &["fetch"]),
            ])
        });
        let exposed: Vec<(&str, &str)> = catalog
            .tools()
            .map(|tool| (tool.server_name.as_str(), tool.exposed_name.as_str()))
            .collect();
        assert_eq!(exposed, [("first", "search"), ("third", "fetch")]);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.find("fetch").unwrap().server_name, "third");
        assert!(catalog.find("missing").is_none());
        let collisions: Vec<_> = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .filter(|(key, ..)| key.key().name() == METRIC_DISCOVERY_FAILURES_TOTAL)
            .map(|(key, ..)| {
                key.key()
                    .labels()
                    .map(|label| format!("{}={}", label.key(), label.value()))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(collisions, [["server=second", "reason=name_collision"]]);
    }

    fn caller(token: &str) -> CatalogCaller {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", token.parse().unwrap());
        CatalogCaller::from_forwarded(&headers)
    }

    #[test]
    fn cache_is_per_caller_and_reports_freshness() {
        let cache = DiscoveryCache::new(8, Duration::from_secs(60), Duration::from_secs(600));
        let start = Instant::now();
        cache.store(caller("Bearer a"), 0, listing(0, "s", &["a"]), start);
        assert!(
            cache.get(caller("Bearer b"), 0, start).is_none(),
            "another caller sees nothing"
        );
        assert!(
            cache.get(caller("Bearer a"), 1, start).is_none(),
            "another server sees nothing"
        );
        let cached = cache.get(caller("Bearer a"), 0, start).unwrap();
        assert!(cached.fresh);
        assert_eq!(names(cached.tools.as_ref().unwrap()), ["a"]);

        // Read just before max_age: stale, kept as last good.
        let later = start + Duration::from_secs(59);
        assert!(cache.get(caller("Bearer a"), 0, later).unwrap().fresh);
        let stale = cache
            .get(caller("Bearer a"), 0, start + Duration::from_secs(61))
            .unwrap();
        assert!(!stale.fresh, "a listing past max_age is rediscovered");
        let unread = cache
            .get(caller("Bearer a"), 0, start + Duration::from_secs(500))
            .unwrap();
        assert!(!unread.fresh, "a stale listing stays as last good until idle_timeout");
    }

    #[test]
    fn a_failure_defers_the_next_attempt_and_keeps_the_last_good_listing() {
        let cache = DiscoveryCache::new(8, Duration::from_secs(60), Duration::from_secs(600));
        let start = Instant::now();
        cache.defer(caller("Bearer a"), 0, start);
        let first = cache
            .get(caller("Bearer a"), 0, start + Duration::from_secs(1))
            .unwrap();
        assert!(first.fresh, "a failure defers the next attempt");
        assert!(first.tools.is_none(), "no listing was ever good");
        let retry = cache
            .get(caller("Bearer a"), 0, start + Duration::from_secs(31))
            .unwrap();
        assert!(!retry.fresh, "the attempt is retried after the interval");

        cache.store(caller("Bearer b"), 0, listing(0, "s", &["b"]), start);
        let later = start + Duration::from_secs(61);
        cache.defer(caller("Bearer b"), 0, later);
        let kept = cache.get(caller("Bearer b"), 0, later).unwrap();
        assert!(kept.fresh);
        assert_eq!(
            names(kept.tools.as_ref().unwrap()),
            ["b"],
            "the last good listing stays"
        );

        let short = DiscoveryCache::new(8, Duration::from_secs(5), Duration::from_secs(600));
        short.defer(caller("Bearer a"), 0, start);
        assert!(
            !short
                .get(caller("Bearer a"), 0, start + Duration::from_secs(6))
                .unwrap()
                .fresh,
            "the retry interval never exceeds max_age"
        );
    }

    #[test]
    fn idle_listings_expire_and_the_least_recently_read_is_evicted() {
        let cache = DiscoveryCache::new(2, Duration::from_secs(1), Duration::from_secs(60));
        let start = Instant::now();
        cache.store(caller("Bearer a"), 0, listing(0, "s", &["a"]), start);
        cache.store(
            caller("Bearer b"),
            0,
            listing(0, "s", &["b"]),
            start + Duration::from_secs(1),
        );
        cache
            .get(caller("Bearer a"), 0, start + Duration::from_secs(2))
            .unwrap();
        cache.store(
            caller("Bearer c"),
            0,
            listing(0, "s", &["c"]),
            start + Duration::from_secs(3),
        );
        assert_eq!(cache.len(), 2);
        let at = start + Duration::from_secs(4);
        assert!(
            cache.get(caller("Bearer b"), 0, at).is_none(),
            "least recently read is evicted"
        );
        assert!(cache.get(caller("Bearer a"), 0, at).is_some());

        assert!(
            cache.get(caller("Bearer c"), 0, at + Duration::from_secs(60)).is_none(),
            "a listing nobody read for idle_timeout is dropped"
        );
    }

    /// Pages served in order; records each requested cursor.
    struct ScriptedPages {
        pages: Mutex<Vec<Result<McpToolPage<serde_json::Value>, McpBackendError>>>,
        cursors: Mutex<Vec<Option<String>>>,
    }

    #[async_trait]
    impl McpToolPages for ScriptedPages {
        async fn list_tools_page(
            &self,
            _target: BackendTarget<'_>,
            scope: ListingScope<'_>,
            cursor: Option<String>,
            _callout: &BackendCallout,
        ) -> Result<McpToolPage<serde_json::Value>, McpBackendError> {
            assert!(matches!(scope, ListingScope::Stateless { protocol_version: "v" }));
            self.cursors.lock().unwrap().push(cursor);
            self.pages.lock().unwrap().remove(0)
        }
    }

    fn callout() -> BackendCallout {
        BackendCallout {
            client: crate::isolated_subrequest_client(1),
            deadline: Instant::now() + Duration::from_secs(5),
            depth: 0,
            downstream: praxis_filter::SubrequestRuntime::new(None, false, None, Instant::now()),
            forwarded_headers: HeaderMap::new(),
            trace_context: None,
        }
    }

    #[tokio::test]
    async fn listing_follows_cursors_and_maps_bounds() {
        let cluster: Arc<str> = Arc::from("c");
        let target = BackendTarget {
            cluster: &cluster,
            path: "/mcp",
        };
        let scope = ListingScope::Stateless { protocol_version: "v" };
        let limits = McpListingLimits {
            max_tools: 2,
            max_bytes: 1 << 20,
        };
        let pages = ScriptedPages {
            pages: Mutex::new(vec![
                Ok(McpToolPage {
                    tools: vec![tool("a")],
                    next_cursor: Some("next".to_owned()),
                }),
                Ok(McpToolPage {
                    tools: vec![tool("b")],
                    next_cursor: None,
                }),
            ]),
            cursors: Mutex::new(Vec::new()),
        };
        let tools = list_server_tools(&pages, target, scope, &callout(), limits)
            .await
            .unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(*pages.cursors.lock().unwrap(), [None, Some("next".to_owned())]);

        let too_many = ScriptedPages {
            pages: Mutex::new(vec![Ok(McpToolPage {
                tools: vec![tool("a"), tool("b"), tool("c")],
                next_cursor: None,
            })]),
            cursors: Mutex::new(Vec::new()),
        };
        assert_eq!(
            list_server_tools(&too_many, target, scope, &callout(), limits)
                .await
                .unwrap_err(),
            DiscoveryError::TooManyTools
        );

        let failing = ScriptedPages {
            pages: Mutex::new(vec![Err(McpBackendError::Timeout)]),
            cursors: Mutex::new(Vec::new()),
        };
        let error = list_server_tools(&failing, target, scope, &callout(), limits)
            .await
            .unwrap_err();
        assert_eq!(error.reason(), "timeout");
    }

    #[tokio::test]
    async fn discovery_runs_concurrently_within_the_bound() {
        let running = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let results = discover_concurrently((0..6).collect(), 2, |server| {
            let running = Arc::clone(&running);
            let peak = Arc::clone(&peak);
            async move {
                let now = running.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                peak.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                tokio::task::yield_now().await;
                running.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                server * 10
            }
        })
        .await;
        let mut results = results;
        results.sort_unstable();
        assert_eq!(results, [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
