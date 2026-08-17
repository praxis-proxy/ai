// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Intelligent route filter: selects an upstream cluster for the request
//! based on the inference model name or MCP tool name.
//!
//! **Lookup precedence:** if `mcp.method` filter metadata exists, the
//! filter attempts MCP tool routing first.  `tools/call` with a valid
//! `mcp.name` matches `mcp_tool` candidates.  Any other MCP method
//! returns `Continue` without routing.  When no `mcp.method` metadata
//! is present, the filter reads the configured model header and matches
//! `inference_model` candidates.
//!
//! MCP metadata takes precedence over the model header to prevent a
//! client-supplied model name from hijacking MCP routing.
//!
//! Candidate selection is deterministic: the first matching candidate in
//! the configured or overlay-rendered order wins after admission filtering.
//! The filter does not recompute source geography, load, or scoring.
//!
//! No request-time metrics or control-plane lookups are performed.

use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use dashmap::DashMap;
use http::{HeaderName, HeaderValue};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;

use super::{
    descriptor::{self, AdmissionState, CandidateConfig, CapabilityKind, RouteCandidate},
    metadata::{
        OVERLAY_REVISION_HEADER, PROVIDER_HOP_REQUEST_ID_HEADER, ROUTE_ADMISSION_STATE, ROUTE_CLUSTER, ROUTE_KIND,
        ROUTE_LOCAL_SITE, ROUTE_NAME, ROUTE_PROVIDER_HOP_REQUEST_ID, ROUTE_RANK, ROUTE_SELECTION_TIER, ROUTE_SITE,
        ROUTE_STABLE_ID, SELECTED_CANDIDATE_HEADER, set_credential_metadata,
    },
    overlay::{self, ExpectedOverlayScope, OverlayReloadHandle, RouteSnapshot},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum length for header values read from the request.
const MAX_HEADER_VALUE_LEN: usize = 256;

/// Maximum session bindings before eviction (single-process scope).
const MAX_BINDINGS: usize = 10_000;

/// Maximum session affinity TTL in seconds (24 hours).
const MAX_TTL_SECS: u64 = 86_400;

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------

/// Deserialized YAML config for the intelligent route filter.
///
/// Supports two modes:
///
/// **Static mode** — candidates and `local_site` are specified inline:
///
/// ```yaml
/// filter: intelligent_route
/// local_site: site-a
/// model_header: x-model
/// candidates:
///   - kind: inference_model
///     name: local-model
///     site: site-a
///     cluster: local-inference
/// ```
///
/// **Overlay mode** — candidates are loaded from an overlay envelope:
///
/// ```yaml
/// filter: intelligent_route
/// overlay_file: /etc/praxis/routing/routing-overlay.json
/// model_header: x-model
/// expected_overlay_scope:
///   network: production-grid
///   gateway: public-edge
///   namespace: grid-system
///   local_site: east-edge
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntelligentRouteConfig {
    /// Static list of route candidates (mutually exclusive with `overlay_file`).
    candidates: Option<Vec<CandidateConfig>>,

    /// Name of the local site (required in static mode, provided by overlay
    /// in overlay mode).
    local_site: Option<String>,

    /// Header name that carries the model name (default: `X-Model`).
    #[serde(default = "default_model_header")]
    model_header: String,

    /// Clusters that terminate the authenticated provider-hop protocol.
    ///
    /// A selected candidate emits the fixed routing context only when
    /// its cluster is present in this allowlist. Each named cluster must use an
    /// mTLS-authenticated Praxis provider gateway. Direct API/backend clusters
    /// remain absent.
    #[serde(default)]
    provider_hop_clusters: Vec<String>,

    /// Expected scope of the overlay envelope.
    ///
    /// When set, each specified field is validated against the
    /// envelope scope on load and every reload. Rejected on mismatch.
    /// Only relevant in overlay mode with envelope-format files.
    expected_overlay_scope: Option<ExpectedOverlayScope>,

    /// Path to a routing overlay JSON file (`routing-overlay.json` envelope or
    /// legacy `routing-config.json`).
    ///
    /// When set, candidates and `local_site` are read from the overlay
    /// instead of the YAML config.
    overlay_file: Option<PathBuf>,

    /// Hot reload configuration for overlay mode.
    ///
    /// Only valid when `overlay_file` is set.  Providing a `reload:`
    /// block with static `candidates` is rejected — static candidates
    /// are immutable for the lifetime of the filter.
    reload: Option<ReloadConfig>,

    /// Session affinity configuration (disabled by default).
    session_affinity: Option<SessionAffinityConfig>,
}

/// Hot reload settings for overlay file watching.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReloadConfig {
    /// Whether file watching is enabled (default: `true`).
    #[serde(default = "default_reload_enabled")]
    enabled: bool,

    /// Debounce window in milliseconds (default: 500).
    #[serde(default = "default_debounce_ms")]
    debounce_ms: u64,
}

impl Default for ReloadConfig {
    fn default() -> Self {
        Self {
            enabled: default_reload_enabled(),
            debounce_ms: default_debounce_ms(),
        }
    }
}

/// Session affinity configuration for binding sessions to stable candidates.
///
/// When enabled, the filter extracts a session key from the configured
/// header or cookie and binds it to a candidate's `stable_id`.
/// Subsequent requests with the same key reuse the bound candidate
/// as long as it remains eligible and the binding has not expired.
///
/// **Scope:** bindings are stored in-memory (single-process).
/// They are not shared across gateway instances and are lost on
/// restart.  This is sufficient for the POC/demo scope.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionAffinityConfig {
    /// Name of a cookie to extract the session key from.
    cookie: Option<String>,

    /// Whether session affinity is enabled (default: `false`).
    #[serde(default)]
    enabled: bool,

    /// Header name to extract the session key from.
    header: Option<String>,

    /// Binding TTL in seconds (default: 3600, max: 86400).
    #[serde(default = "default_ttl_secs")]
    ttl_secs: u64,
}

/// Default session affinity TTL (1 hour).
fn default_ttl_secs() -> u64 {
    3600
}

/// Default model header name.
fn default_model_header() -> String {
    "X-Model".to_owned()
}

/// Default reload enabled state.
fn default_reload_enabled() -> bool {
    true
}

/// Default debounce window in milliseconds.
fn default_debounce_ms() -> u64 {
    overlay::DEFAULT_DEBOUNCE_MS
}

// -----------------------------------------------------------------------------
// Session Affinity (runtime)
// -----------------------------------------------------------------------------

/// In-memory session affinity state (single-process scope).
struct SessionAffinity {
    /// Session key → bound candidate mapping.
    bindings: DashMap<String, Binding>,

    /// Cookie name to extract the session key from.
    cookie: Option<Arc<str>>,

    /// Header name to extract the session key from.
    header: Option<HeaderName>,

    /// Binding time-to-live.
    ttl: Duration,
}

impl std::fmt::Debug for SessionAffinity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAffinity")
            .field("bindings_len", &self.bindings.len())
            .field("cookie", &self.cookie)
            .field("header", &self.header)
            .field("ttl", &self.ttl)
            .finish()
    }
}

/// A single session binding.
struct Binding {
    /// When this binding expires.
    expires: Instant,

    /// Stable ID of the bound candidate.
    stable_id: Arc<str>,
}

/// Result of a session affinity lookup.
enum AffinityOutcome<'a> {
    /// No affinity configured.
    Inactive,

    /// Affinity enabled but no session key in request.
    NoKey,

    /// Session key found but no binding exists.
    New,

    /// Binding existed but candidate is gone, expired, or excluded.
    Failover,

    /// Bound candidate is still eligible.
    Reused(&'a RouteCandidate),
}

// -----------------------------------------------------------------------------
// IntelligentRouteFilter
// -----------------------------------------------------------------------------

/// Selects an upstream cluster from a site/capability descriptor
/// by matching either an inference model name or MCP tool name.
///
/// This filter is registered by the AI proxy (not Praxis core) because it
/// encodes AI-specific routing semantics: ordered candidate consumption,
/// admission-state filtering, session affinity, and MCP tool-call routing.
/// Praxis core provides the generic filter runtime; this filter adds the intelligent routing
/// candidate model on top.
///
/// **Modes:**
/// - **Static:** candidates are declared inline in the YAML config.
/// - **Overlay:** candidates are loaded from an overlay file (`routing-overlay.json` envelope or legacy
///   `routing-config.json`) and hot-reloaded via [`ArcSwap`] when the file changes.
///
/// **Behavior:**
/// - If `ctx.cluster` is already set by an earlier filter, the selection is preserved and no metadata is written.
/// - If no routing source is present, the filter returns `Continue` without routing.
/// - If the model header or MCP tool name is blank, oversized, or invalid, the filter rejects with 400.
/// - If a matching candidate is found, `ctx.cluster` is set and bounded route-decision metadata is written.
/// - If no matching candidate is found, the filter rejects with 404.
///
/// **Selection:** the first matching candidate in the configured or
/// overlay-rendered order wins after admission filtering.  Praxis AI does not
/// recompute source geography, load, or score.  `admission_state=none` is never
/// eligible.  `admission_state=existing_only` is only eligible through an
/// already-bound session affinity entry.
///
/// **Metadata:** on successful selection, bounded in-process filter
/// metadata is written under the `intelligent_route.` namespace (`kind`, `name`,
/// `site`, `cluster`, `local_site`, `stable_id`, `admission_state`, and
/// optionally `rank`, `selection_tier`).  When session affinity is enabled,
/// `session.bound`, `session.reused`, and `session.failover` keys are also
/// written. When the selected cluster is present in `provider_hop_clusters`,
/// client-supplied `x-ai-routing-candidate`,
/// `x-ai-routing-request-id`, and `x-ai-routing-revision` values
/// are removed and replaced with the selected stable ID, a generated
/// provider-hop request ID, and the serving overlay revision (envelope mode
/// only). These AI-owned, non-reserved headers are sent only to an
/// mTLS-authenticated provider gateway; the provider must run
/// `peer_identity_trust` before consuming them.
/// No credential reference or value is forwarded. No request-time database,
/// control-plane, or metrics lookups are performed.
///
/// **MCP lookup:** if `mcp.method` filter metadata is set to `tools/call`
/// and `mcp.name` is present, `mcp_tool` candidates are matched.
/// Other MCP methods (`initialize`, `notifications/*`, etc.) skip routing.
///
/// **Hot reload:** when `reload.enabled` is `true` (the default in overlay
/// mode), the filter watches the overlay file's parent directory for
/// filesystem events.  On change, the file is re-read, SHA-256 hashed
/// (skipped if identical), parsed, validated, and atomically swapped in
/// via `ArcSwap`.  In-flight requests continue using their previously
/// loaded snapshot.  Unreadable or invalid files retain the previous
/// snapshot.  Kubernetes `ConfigMap` projected volumes use atomic symlink
/// replacement (`..data`), which the watcher detects as a Create/Modify
/// event on the parent directory.  The overlay `ConfigMap` **must not**
/// use `subPath` volume mounts — `subPath` bypasses the `..data` symlink
/// mechanism and the watcher will not detect updates.
///
/// **Envelope contract:** The configuration producer publishes
/// `routing-overlay.json` as a versioned, content-addressed envelope alongside
/// the legacy `routing-config.json` payload. The AI-owned v1 shape is:
///
/// ```text
/// {
///   "schema_version": "1.0.0",
///   "revision": {
///     "kind": "content_addressed",
///     "algorithm": "sha256",
///     "value": "<64 lowercase hex characters>"
///   },
///   "content_digest": {
///     "algorithm": "sha256",
///     "value": "<same value as revision>"
///   },
///   "scope": {
///     "network": "<routing network>",
///     "gateway": "<destination gateway>",
///     "namespace": "<destination namespace>",
///     "local_site": "<gateway site>"
///   },
///   "provenance": {
///     "producer": "<producer name>",
///     "producer_version": "<producer version>",
///     "source_name": "<same value as scope.network>",
///     "source_uid": "<opaque source-resource identity>",
///     "source_generation": 1,
///     "rendered_at": "<RFC 3339 timestamp>"
///   },
///   "overlay": {
///     "network": "<same value as scope.network>",
///     "local_site": "<same value as scope.local_site>",
///     "candidates": []
///   }
/// }
/// ```
///
/// `source_generation` is a positive monotonic generation of the source
/// resource. Unknown additive envelope, provenance, and candidate fields are
/// accepted for forward compatibility; credential objects reject unknown
/// fields to prevent secret material from entering the overlay.
///
/// Any reserved envelope field selects strict envelope parsing; malformed
/// envelopes never fall back to legacy. When `expected_overlay_scope` is
/// configured, legacy payloads are rejected so scope validation cannot be
/// bypassed. Praxis AI recomputes the RFC 8785 canonical SHA-256 digest over
/// `network`, `local_site`, and the ordered candidate list before accepting a
/// snapshot.
///
/// The revision lifecycle is observable as:
/// - **rendered:** The producer constructed the envelope.
/// - **distributed:** The producer applied it to the destination `ConfigMap`.
/// - **accepted:** Praxis AI parsed, scope-checked, and digest-verified it.
/// - **serving:** a request selected a route from that exact snapshot.
///
/// Invalid cold-start envelopes fail filter construction. Invalid reloads
/// retain the same-process last-known-good snapshot. Envelope-mode provider
/// hops carry the serving revision from the same immutable snapshot used for
/// candidate selection.
///
/// **Scope:** overlay hot reload swaps the candidate list and `local_site`
/// only.  It cannot add or remove `load_balancer` clusters, change
/// cluster endpoints or TLS configuration, or inject credential values.
/// Those changes require a full pipeline reload or pod restart.
/// Every cluster name that may appear in any overlay version must
/// already be configured in the downstream `load_balancer` filter.
/// An overlay that references an unknown cluster will cause
/// request-time failures, not a reload rejection.
///
/// [`ArcSwap`]: arc_swap::ArcSwap
pub struct IntelligentRouteFilter {
    /// Clusters allowed to receive authenticated provider-hop context.
    provider_hop_clusters: BTreeSet<String>,
    /// Header that carries the model name.
    model_header: HeaderName,
    /// Watcher handle for overlay hot reload (None in static mode).
    _reload_handle: Option<OverlayReloadHandle>,
    /// In-memory session affinity (None when disabled).
    session_affinity: Option<SessionAffinity>,
    /// Atomic snapshot of routing state (candidates + `local_site`).
    snapshot: Arc<ArcSwap<RouteSnapshot>>,
}

impl IntelligentRouteFilter {
    /// Create an intelligent route filter from parsed YAML config.
    ///
    /// In **overlay mode** (`overlay_file` set), reads the overlay file,
    /// builds an initial snapshot, and optionally spawns a background
    /// watcher for hot reload.
    ///
    /// In **static mode** (`candidates` set), validates the inline
    /// candidates and builds a static snapshot with no watcher.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if:
    /// - both `overlay_file` and `candidates` are set
    /// - neither `overlay_file` nor `candidates` is set
    /// - the overlay file cannot be read or parsed
    /// - the candidate list is empty or invalid
    /// - the model header is invalid
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: IntelligentRouteConfig = parse_filter_config("intelligent_route", config)?;
        let model_header = descriptor::validate_model_header(&cfg.model_header)?;

        if cfg.overlay_file.is_some() && cfg.candidates.is_some() {
            return Err("intelligent_route: cannot set both overlay_file and candidates".into());
        }
        if cfg.overlay_file.is_none() && cfg.expected_overlay_scope.is_some() {
            return Err("intelligent_route: expected_overlay_scope requires envelope overlay_file mode".into());
        }

        let (snapshot, reload_handle) = if let Some(path) = cfg.overlay_file {
            let reload = cfg.reload.unwrap_or_default();
            build_overlay_snapshot(path, &reload, cfg.expected_overlay_scope)?
        } else if let Some(candidates_raw) = cfg.candidates {
            if cfg.reload.is_some() {
                return Err("intelligent_route: reload block is not valid with static candidates".into());
            }
            build_static_snapshot(candidates_raw, cfg.local_site)?
        } else {
            return Err("intelligent_route: either overlay_file or candidates must be set".into());
        };

        let session_affinity = build_session_affinity(cfg.session_affinity)?;
        let provider_hop_clusters = validate_provider_hop_clusters(cfg.provider_hop_clusters)?;

        Ok(Box::new(Self {
            model_header,
            _reload_handle: reload_handle,
            provider_hop_clusters,
            session_affinity,
            snapshot,
        }))
    }

    /// Core routing path: session affinity lookup, admission filtering,
    /// candidate selection, metadata output.
    #[expect(clippy::too_many_lines, reason = "sequential affinity/selection/metadata pipeline")]
    fn select_and_route(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        snap: &RouteSnapshot,
        kind: CapabilityKind,
        name: &str,
    ) -> Result<FilterAction, FilterError> {
        let session_key = self.session_affinity.as_ref().and_then(|a| extract_session_key(a, ctx));
        let outcome = resolve_affinity(
            self.session_affinity.as_ref(),
            session_key.as_deref(),
            &snap.candidates,
            kind,
            name,
        );
        if let AffinityOutcome::Reused(c) = outcome {
            return apply_reused(
                ctx,
                &snap.local_site,
                c,
                &self.provider_hop_clusters,
                snap.semantic_revision.as_ref(),
            );
        }
        let failover = matches!(outcome, AffinityOutcome::Failover);
        let Some(c) = select_admitted(&snap.candidates, kind, name) else {
            tracing::debug!(kind = kind.as_str(), name = %name, "intelligent_route: no candidate");
            return Ok(FilterAction::Reject(Rejection::status(404)));
        };
        apply_route(
            ctx,
            &snap.local_site,
            c,
            &self.provider_hop_clusters,
            snap.semantic_revision.as_ref(),
        )?;
        if let Some(aff) = &self.session_affinity {
            record_session(aff, ctx, &c.stable_id, session_key.as_deref(), failover);
        }
        Ok(FilterAction::Continue)
    }
}

/// Return type for snapshot builders: shared snapshot + optional watcher.
type SnapshotResult = Result<(Arc<ArcSwap<RouteSnapshot>>, Option<OverlayReloadHandle>), FilterError>;

/// Build an overlay-backed snapshot with optional watcher.
fn build_overlay_snapshot(
    path: PathBuf,
    reload: &ReloadConfig,
    expected_scope: Option<ExpectedOverlayScope>,
) -> SnapshotResult {
    let content = overlay::read_overlay_bounded(&path).map_err(|e| {
        FilterError::from(format!(
            "intelligent_route: failed to read overlay file {}: {e}",
            path.display()
        ))
    })?;
    let snap = RouteSnapshot::from_overlay_with_scope(&content, expected_scope.as_ref())?;
    tracing::info!(
        path = %path.display(),
        contract_format = ?snap.contract_format,
        schema_version = snap.schema_version.as_deref().unwrap_or("none"),
        accepted_revision = snap.semantic_revision.as_deref().unwrap_or("none"),
        serving_revision = snap.semantic_revision.as_deref().unwrap_or("none"),
        candidate_count = snap.candidates.len(),
        "intelligent_route: overlay snapshot initialized"
    );
    let shared = Arc::new(ArcSwap::from_pointee(snap));
    let handle = if reload.enabled {
        Some(overlay::spawn_overlay_watcher_with_scope(
            path,
            Arc::clone(&shared),
            reload.debounce_ms,
            expected_scope,
        )?)
    } else {
        None
    };
    Ok((shared, handle))
}

/// Build a static snapshot from inline candidates.
fn build_static_snapshot(candidates_raw: Vec<CandidateConfig>, local_site: Option<String>) -> SnapshotResult {
    let local_site_str = local_site
        .ok_or_else(|| FilterError::from("intelligent_route: local_site is required when candidates is set"))?;
    descriptor::validate_local_site(&local_site_str)?;
    let candidates = descriptor::validate_candidates(candidates_raw)?;
    let snap = RouteSnapshot::from_static(candidates, Arc::from(local_site_str.as_str()));
    Ok((Arc::new(ArcSwap::from_pointee(snap)), None))
}

/// Build the runtime [`SessionAffinity`] from config, if enabled.
fn build_session_affinity(config: Option<SessionAffinityConfig>) -> Result<Option<SessionAffinity>, FilterError> {
    let Some(cfg) = config else {
        return Ok(None);
    };
    if !cfg.enabled {
        return Ok(None);
    }
    validate_session_affinity_config(&cfg)?;
    let header = cfg
        .header
        .as_deref()
        .filter(|h| !h.trim().is_empty())
        .map(str::parse::<HeaderName>)
        .transpose()
        .map_err(|e| -> FilterError { format!("intelligent_route: invalid session_affinity.header: {e}").into() })?;
    let cookie = cfg.cookie.as_deref().filter(|c| !c.trim().is_empty()).map(Arc::from);
    Ok(Some(SessionAffinity {
        bindings: DashMap::new(),
        cookie,
        header,
        ttl: Duration::from_secs(cfg.ttl_secs),
    }))
}

/// Validate and deduplicate the bounded provider-hop cluster allowlist.
fn validate_provider_hop_clusters(clusters: Vec<String>) -> Result<BTreeSet<String>, FilterError> {
    let mut validated = BTreeSet::new();
    for cluster in clusters {
        descriptor::validate_cluster_name("provider_hop_clusters", &cluster)?;
        if !validated.insert(cluster) {
            return Err("intelligent_route: duplicate provider_hop_clusters entry".into());
        }
    }
    Ok(validated)
}

/// Validate session affinity config constraints.
fn validate_session_affinity_config(cfg: &SessionAffinityConfig) -> Result<(), FilterError> {
    let has_header = cfg.header.as_deref().is_some_and(|h| !h.trim().is_empty());
    let has_cookie = cfg.cookie.as_deref().is_some_and(|c| !c.trim().is_empty());
    if !has_header && !has_cookie {
        return Err("intelligent_route: session_affinity requires at least one of header or cookie".into());
    }
    if cfg.ttl_secs == 0 || cfg.ttl_secs > MAX_TTL_SECS {
        return Err(format!("intelligent_route: session_affinity.ttl_secs must be 1-{MAX_TTL_SECS}").into());
    }
    Ok(())
}

#[async_trait]
impl HttpFilter for IntelligentRouteFilter {
    fn name(&self) -> &'static str {
        "intelligent_route"
    }

    /// `intelligent_route` selects `ctx.cluster` from configured candidates.
    ///
    /// Returning `true` here tells the Praxis pipeline validator that this
    /// filter satisfies the "cluster-selecting filter before `load_balancer`"
    /// requirement.  Without this, the validator would reject pipelines that
    /// use `intelligent_route → load_balancer` without an intervening `router`.
    fn selects_cluster(&self) -> bool {
        true
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Client-supplied routing protocol context is never allowed to survive an
        // edge routing decision.
        ctx.request_headers_to_remove
            .push(HeaderName::from_static(SELECTED_CANDIDATE_HEADER));
        ctx.request_headers_to_remove
            .push(HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER));
        ctx.request_headers_to_remove
            .push(HeaderName::from_static(OVERLAY_REVISION_HEADER));

        if ctx.cluster.is_some() {
            tracing::debug!("intelligent_route: cluster already set; preserving");
            return Ok(FilterAction::Continue);
        }

        let snap = self.snapshot.load();
        let lookup = extract_lookup(ctx, &self.model_header);

        let (kind, name) = match lookup {
            Lookup::Route { kind, name } => (kind, name),
            Lookup::Skip => return Ok(FilterAction::Continue),
            Lookup::Invalid => return Ok(FilterAction::Reject(Rejection::status(400))),
        };

        self.select_and_route(ctx, &snap, kind, &name)
    }
}

/// Apply a reused (session-affinity-bound) candidate.
fn apply_reused(
    ctx: &mut HttpFilterContext<'_>,
    local_site: &Arc<str>,
    candidate: &RouteCandidate,
    provider_hop_clusters: &BTreeSet<String>,
    semantic_revision: Option<&Arc<str>>,
) -> Result<FilterAction, FilterError> {
    ctx.cluster = Some(Arc::clone(&candidate.cluster));
    record_route_decision(ctx, local_site, candidate);
    write_provider_context(ctx, candidate, provider_hop_clusters, semantic_revision)?;
    #[cfg(feature = "opentelemetry")]
    crate::opentelemetry::record_routing_selection(candidate, local_site, semantic_revision);
    ctx.set_metadata("intelligent_route.session.bound", "true");
    ctx.set_metadata("intelligent_route.session.reused", "true");
    ctx.set_metadata("intelligent_route.session.failover", "false");
    Ok(FilterAction::Continue)
}

/// Set cluster and record route decision metadata.
fn apply_route(
    ctx: &mut HttpFilterContext<'_>,
    local_site: &Arc<str>,
    candidate: &RouteCandidate,
    provider_hop_clusters: &BTreeSet<String>,
    semantic_revision: Option<&Arc<str>>,
) -> Result<(), FilterError> {
    ctx.cluster = Some(Arc::clone(&candidate.cluster));
    record_route_decision(ctx, local_site, candidate);
    write_provider_context(ctx, candidate, provider_hop_clusters, semantic_revision)?;
    #[cfg(feature = "opentelemetry")]
    crate::opentelemetry::record_routing_selection(candidate, local_site, semantic_revision);
    Ok(())
}

/// Record session-affinity metadata and store a binding.
fn record_session(
    affinity: &SessionAffinity,
    ctx: &mut HttpFilterContext<'_>,
    stable_id: &Arc<str>,
    session_key: Option<&str>,
    failover: bool,
) {
    let bound = if let Some(key) = session_key {
        store_binding(affinity, key, stable_id);
        true
    } else {
        false
    };
    ctx.set_metadata("intelligent_route.session.bound", if bound { "true" } else { "false" });
    ctx.set_metadata("intelligent_route.session.reused", "false");
    ctx.set_metadata(
        "intelligent_route.session.failover",
        if failover { "true" } else { "false" },
    );
}

// -----------------------------------------------------------------------------
// Lookup Extraction
// -----------------------------------------------------------------------------

/// Result of extracting a routable capability from the request.
enum Lookup {
    /// A routable capability was found.
    Route {
        /// Capability kind.
        kind: CapabilityKind,
        /// Capability name.
        name: String,
    },
    /// No routable capability; continue without routing.
    Skip,
    /// Input is present but invalid; fail closed.
    Invalid,
}

/// Extract the routable capability from request context.
///
/// MCP metadata takes precedence over the model header: if `mcp.method`
/// metadata is present (set by an upstream MCP classifier filter), the
/// filter dispatches to MCP tool lookup.  Otherwise it falls back to the
/// configured model header.
fn extract_lookup(ctx: &HttpFilterContext<'_>, model_header: &HeaderName) -> Lookup {
    if let Some(mcp_method) = ctx.get_metadata("mcp.method") {
        return extract_mcp_lookup(ctx, mcp_method);
    }
    extract_model_lookup(ctx, model_header)
}

/// Extract an MCP tool lookup from filter metadata.
///
/// Only `tools/call` is routable.  Any other MCP method continues without
/// routing even if a model header is present — the request is an MCP
/// protocol message, not an inference request.
fn extract_mcp_lookup(ctx: &HttpFilterContext<'_>, method: &str) -> Lookup {
    if method != "tools/call" {
        tracing::debug!(
            method = method,
            "intelligent_route: non-tools/call MCP method; skipping"
        );
        return Lookup::Skip;
    }
    let Some(name) = ctx.get_metadata("mcp.name") else {
        tracing::debug!("intelligent_route: tools/call without mcp.name; rejecting");
        return Lookup::Invalid;
    };
    if name.trim().is_empty() || name.len() > MAX_HEADER_VALUE_LEN {
        tracing::debug!("intelligent_route: mcp.name blank or oversized; rejecting");
        return Lookup::Invalid;
    }
    Lookup::Route {
        kind: CapabilityKind::McpTool,
        name: name.to_owned(),
    }
}

/// Extract an inference model lookup from the promoted model header.
fn extract_model_lookup(ctx: &HttpFilterContext<'_>, model_header: &HeaderName) -> Lookup {
    let Some(value) = ctx.request.headers.get(model_header) else {
        tracing::debug!("intelligent_route: no model header; skipping");
        return Lookup::Skip;
    };
    let Ok(model) = value.to_str() else {
        tracing::debug!("intelligent_route: model header is not valid UTF-8; rejecting");
        return Lookup::Invalid;
    };
    if model.trim().is_empty() || model.len() > MAX_HEADER_VALUE_LEN {
        tracing::debug!("intelligent_route: model header blank or oversized; rejecting");
        return Lookup::Invalid;
    }
    Lookup::Route {
        kind: CapabilityKind::InferenceModel,
        name: model.to_owned(),
    }
}

// -----------------------------------------------------------------------------
// Session Affinity Helpers
// -----------------------------------------------------------------------------

/// Extract a session key from the request using the configured sources.
///
/// Checks the header first, then falls back to the cookie.
fn extract_session_key(affinity: &SessionAffinity, ctx: &HttpFilterContext<'_>) -> Option<String> {
    if let Some(header_name) = &affinity.header
        && let Some(v) = ctx.request.headers.get(header_name)
        && let Ok(s) = v.to_str()
    {
        let s = s.trim();
        if !s.is_empty() && s.len() <= MAX_HEADER_VALUE_LEN {
            return Some(s.to_owned());
        }
    }
    affinity
        .cookie
        .as_deref()
        .and_then(|name| extract_cookie_value(ctx, name))
}

/// Extract a named cookie value from the `Cookie` header.
fn extract_cookie_value(ctx: &HttpFilterContext<'_>, name: &str) -> Option<String> {
    let cookie_hdr = ctx.request.headers.get(http::header::COOKIE)?;
    let cookie_str = cookie_hdr.to_str().ok()?;
    let prefix = format!("{name}=");
    for part in cookie_str.split(';') {
        let trimmed = part.trim();
        if let Some(value) = trimmed.strip_prefix(&prefix)
            && !value.is_empty()
            && value.len() <= MAX_HEADER_VALUE_LEN
        {
            return Some(value.to_owned());
        }
    }
    None
}

/// Resolve session affinity state for the current request.
fn resolve_affinity<'a>(
    affinity: Option<&SessionAffinity>,
    session_key: Option<&str>,
    candidates: &'a [RouteCandidate],
    kind: CapabilityKind,
    name: &str,
) -> AffinityOutcome<'a> {
    let Some(aff) = affinity else {
        return AffinityOutcome::Inactive;
    };
    let Some(key) = session_key else {
        return AffinityOutcome::NoKey;
    };
    lookup_binding(aff, key, candidates, kind, name)
}

/// Look up an existing binding and find its candidate.
fn lookup_binding<'a>(
    affinity: &SessionAffinity,
    key: &str,
    candidates: &'a [RouteCandidate],
    kind: CapabilityKind,
    name: &str,
) -> AffinityOutcome<'a> {
    let Some(binding) = affinity.bindings.get(key) else {
        return AffinityOutcome::New;
    };
    if binding.expires < Instant::now() {
        drop(binding);
        affinity.bindings.remove(key);
        return AffinityOutcome::New;
    }
    let stable = Arc::clone(&binding.stable_id);
    drop(binding);
    for c in candidates {
        if c.kind != kind || &*c.name != name || *c.stable_id != *stable {
            continue;
        }
        if c.admission_state == AdmissionState::Excluded {
            return AffinityOutcome::Failover;
        }
        return AffinityOutcome::Reused(c);
    }
    AffinityOutcome::Failover
}

/// Store or update a session binding.
fn store_binding(affinity: &SessionAffinity, key: &str, stable_id: &Arc<str>) {
    if affinity.bindings.len() >= MAX_BINDINGS {
        evict_expired(affinity);
    }
    if affinity.bindings.len() >= MAX_BINDINGS {
        tracing::warn!("intelligent_route: session binding table full; routing without binding");
        return;
    }
    affinity.bindings.insert(
        key.to_owned(),
        Binding {
            expires: Instant::now() + affinity.ttl,
            stable_id: Arc::clone(stable_id),
        },
    );
}

/// Remove all expired bindings.
fn evict_expired(affinity: &SessionAffinity) {
    let now = Instant::now();
    affinity.bindings.retain(|_, b| b.expires > now);
}

// -----------------------------------------------------------------------------
// Candidate Selection
// -----------------------------------------------------------------------------

/// Select the first candidate admitted for a new request.
///
/// Always excludes [`Excluded`] candidates.  When `is_new_session`
/// is `true`, also excludes [`ExistingOnly`] candidates.
///
/// Praxis AI intentionally preserves the configured or overlay-rendered
/// candidate order. The source owns geography, load-aware ordering, and rank.
///
/// [`Excluded`]: AdmissionState::Excluded
/// [`ExistingOnly`]: AdmissionState::ExistingOnly
fn select_admitted<'a>(
    candidates: &'a [RouteCandidate],
    kind: CapabilityKind,
    name: &str,
) -> Option<&'a RouteCandidate> {
    for c in candidates {
        if c.kind != kind || &*c.name != name {
            continue;
        }
        if !is_admitted_for_new_request(c.admission_state) {
            continue;
        }
        return Some(c);
    }
    None
}

/// Whether a candidate passes admission filtering.
fn is_admitted_for_new_request(state: AdmissionState) -> bool {
    state == AdmissionState::NewAndExisting
}

// -----------------------------------------------------------------------------
// Route Decision Metadata
// -----------------------------------------------------------------------------

/// Write bounded route-decision metadata on successful selection.
///
/// Keys use `intelligent_route.` namespace. All values are bounded by the
/// existing `set_metadata` limits.  No HTTP forwarding headers are
/// written by this function.
fn record_route_decision(ctx: &mut HttpFilterContext<'_>, local_site: &Arc<str>, candidate: &RouteCandidate) {
    ctx.set_metadata(ROUTE_ADMISSION_STATE, candidate.admission_state.as_str());
    ctx.set_metadata(ROUTE_CLUSTER, &*candidate.cluster);
    ctx.set_metadata(ROUTE_KIND, candidate.kind.as_str());
    ctx.set_metadata(ROUTE_LOCAL_SITE, &**local_site);
    ctx.set_metadata(ROUTE_NAME, &*candidate.name);
    ctx.set_metadata(ROUTE_SITE, &*candidate.site);
    ctx.set_metadata(ROUTE_STABLE_ID, &*candidate.stable_id);
    if let Some(rank) = candidate.rank {
        ctx.set_metadata(ROUTE_RANK, rank.to_string());
    }
    if let Some(tier) = &candidate.selection_tier {
        ctx.set_metadata(ROUTE_SELECTION_TIER, &**tier);
    }
    set_credential_metadata(ctx, candidate.credential.as_ref());
}

/// Emit the minimal AI-owned edge-to-provider context.
///
/// These non-reserved headers survive the Praxis upstream boundary. This
/// function uses remove-and-overwrite semantics so a client value cannot become
/// peer context. The fields are routing context, not an authorization grant;
/// the provider pipeline must authenticate and authorize the mTLS peer before
/// consuming them.
#[expect(clippy::too_many_lines, reason = "writes all provider-hop headers in one sequence")]
fn write_provider_context(
    ctx: &mut HttpFilterContext<'_>,
    candidate: &RouteCandidate,
    provider_hop_clusters: &BTreeSet<String>,
    semantic_revision: Option<&Arc<str>>,
) -> Result<(), FilterError> {
    if !provider_hop_clusters.contains(candidate.cluster.as_ref()) {
        return Ok(());
    }
    if candidate.stable_id.trim().is_empty() || candidate.stable_id.len() > MAX_HEADER_VALUE_LEN {
        return Err(format!(
            "intelligent_route: selected candidate ID must be 1-{MAX_HEADER_VALUE_LEN} non-blank bytes"
        )
        .into());
    }

    let candidate_id = HeaderValue::from_str(candidate.stable_id.as_ref()).map_err(|e| {
        FilterError::from(format!(
            "intelligent_route: selected candidate ID is not a valid header value: {e}"
        ))
    })?;
    let hop_request_id = ctx.id_generator.generate(ctx.time_source);
    let request_id_value = HeaderValue::from_str(&hop_request_id).map_err(|e| {
        FilterError::from(format!(
            "intelligent_route: generated provider-hop request ID is invalid: {e}"
        ))
    })?;

    ctx.request_headers_to_set
        .push((HeaderName::from_static(SELECTED_CANDIDATE_HEADER), candidate_id));
    ctx.request_headers_to_set.push((
        HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
        request_id_value,
    ));
    if let Some(rev) = semantic_revision {
        let rev_value = HeaderValue::from_str(rev).map_err(|error| {
            FilterError::from(format!("intelligent_route: invalid serving overlay revision: {error}"))
        })?;
        ctx.request_headers_to_set
            .push((HeaderName::from_static(OVERLAY_REVISION_HEADER), rev_value));
    }
    ctx.set_metadata(ROUTE_PROVIDER_HOP_REQUEST_ID, hop_request_id);
    Ok(())
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use http::Method;

    use super::*;

    // ---- Config validation ----

    #[test]
    fn valid_minimal_config() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: llama\n    site: site-a\n    cluster: inf\n    fresh: true\n";
        assert!(parse(yaml).is_ok(), "minimal valid config should parse");
    }

    #[tokio::test]
    async fn default_model_header_is_x_model() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "default model header X-Model should route"
        );
        assert_eq!(ctx.cluster.as_deref(), Some("inf"), "cluster should be set");
    }

    #[test]
    fn blank_local_site_rejected() {
        let err = parse_err(
            "local_site: \"\"\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n",
        );
        assert!(
            err.to_string().contains("blank") || err.to_string().contains("non-blank"),
            "blank local_site should be rejected: {err}"
        );
    }

    #[test]
    fn missing_candidates_rejected() {
        let err = parse_err("local_site: site-a\ncandidates: []\n");
        assert!(
            err.to_string().contains("empty"),
            "empty candidates should be rejected: {err}"
        );
    }

    #[test]
    fn blank_model_header_rejected() {
        let err = parse_err(
            "local_site: site-a\nmodel_header: \"\"\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n",
        );
        assert!(
            err.to_string().contains("blank") || err.to_string().contains("empty"),
            "blank model_header should be rejected: {err}"
        );
    }

    #[test]
    fn reserved_model_header_rejected() {
        let err = parse_err(
            "local_site: site-a\nmodel_header: x-praxis-foo\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n",
        );
        assert!(
            err.to_string().contains("reserved"),
            "reserved model_header should be rejected: {err}"
        );
    }

    #[test]
    fn invalid_candidate_rejected() {
        let err = parse_err(
            "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: \"\"\n    site: s\n    cluster: c\n    fresh: true\n",
        );
        assert!(
            err.to_string().contains("blank") || err.to_string().contains("non-blank"),
            "blank candidate name should be rejected: {err}"
        );
    }

    // ---- Model header extraction ----

    #[tokio::test]
    async fn absent_model_header_continues() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "absent model header should continue without routing"
        );
        assert!(ctx.cluster.is_none(), "no cluster should be set");
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn blank_model_header_rejects() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static(""));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "blank model header should reject 400"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn oversized_model_header_rejects_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        let big = "a".repeat(MAX_HEADER_VALUE_LEN + 1);
        req.headers.insert("X-Model", HeaderValue::from_str(&big).unwrap());
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "oversized model header should reject 400"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn invalid_utf8_model_header_rejects_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers
            .insert("X-Model", HeaderValue::from_bytes(b"\xff\xfe").unwrap());
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "non-UTF-8 model header should reject 400"
        );
        assert_no_route_metadata(&ctx);
    }

    // ---- Candidate selection ----

    #[tokio::test]
    async fn unknown_model_rejects_404() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("unknown-model"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "unknown model should reject 404"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn local_inference_sets_cluster() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "local-inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.cluster.as_deref(), Some("local-inf"), "cluster should be set");
    }

    #[tokio::test]
    async fn remote_inference_sets_gateway_cluster() {
        let f = make_filter(&[("inference_model", "llama", "site-b", "remote-gw")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.cluster.as_deref(), Some("remote-gw"));
    }

    // ---- MCP tool routing ----

    #[tokio::test]
    async fn mcp_tools_call_routes_to_matching_tool() {
        let f = make_filter(&[
            ("mcp_tool", "weather", "site-c", "grid-site-c"),
            ("inference_model", "llama", "site-a", "local-inf"),
        ]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "weather");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "valid MCP tool should route");
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("grid-site-c"),
            "cluster should be mcp_tool cluster"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.kind"), Some("mcp_tool"));
        assert_eq!(ctx.get_metadata("intelligent_route.name"), Some("weather"));
    }

    #[tokio::test]
    async fn mcp_tools_call_beats_model_header() {
        let f = make_filter(&[
            ("mcp_tool", "weather", "site-c", "mcp-cluster"),
            ("inference_model", "llama", "site-a", "inf-cluster"),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/mcp");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "weather");

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("mcp-cluster"),
            "MCP metadata must win over model header"
        );
    }

    #[tokio::test]
    async fn mcp_non_tools_call_skips_even_with_model_header() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf-cluster")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/mcp");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "initialize");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Continue),
            "non-tools/call MCP method must skip without routing"
        );
        assert!(ctx.cluster.is_none(), "no cluster should be set for non-tools/call");
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn mcp_tools_call_missing_name_rejects_400() {
        let f = make_filter(&[("mcp_tool", "weather", "site-c", "c")]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        // mcp.name not set

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "missing mcp.name must reject 400"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn mcp_tools_call_blank_name_rejects_400() {
        let f = make_filter(&[("mcp_tool", "weather", "site-c", "c")]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "blank mcp.name must reject 400"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn unknown_mcp_tool_rejects_404() {
        let f = make_filter(&[("mcp_tool", "weather", "site-c", "c")]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "unknown-tool");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "unknown mcp_tool must reject 404"
        );
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn inference_candidate_not_matched_by_mcp_lookup() {
        // Only inference_model candidates configured; MCP tools/call should not match them.
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "llama");

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "inference_model candidates must not match MCP lookup"
        );
    }

    #[tokio::test]
    async fn mcp_tool_uses_configured_order() {
        let f = make_filter_with_fresh(&[
            ("mcp_tool", "weather", "site-b", "remote-mcp", true),
            ("mcp_tool", "weather", "site-a", "local-mcp", true),
        ]);
        let req = crate::test_utils::make_request(Method::POST, "/mcp");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.set_metadata("mcp.method", "tools/call");
        ctx.set_metadata("mcp.name", "weather");

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("remote-mcp"),
            "first matching MCP tool candidate should win"
        );
    }

    // ---- Cluster preservation ----

    #[tokio::test]
    async fn preserves_existing_cluster() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.cluster = Some(Arc::from("pre-set-cluster"));

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("pre-set-cluster"),
            "pre-set cluster should be preserved"
        );
        assert_no_route_metadata(&ctx);
    }

    // ---- Ordered selection ----

    #[tokio::test]
    async fn configured_order_beats_locality() {
        let f = make_filter(&[
            ("inference_model", "llama", "site-b", "remote-inf"),
            ("inference_model", "llama", "site-a", "local-inf"),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("remote-inf"),
            "first configured candidate must win; intelligent_route does not recompute locality"
        );
    }

    #[tokio::test]
    async fn configured_order_selects_first_matching_candidate() {
        let f = make_filter(&[
            ("inference_model", "llama", "site-b", "first-remote"),
            ("inference_model", "llama", "site-c", "second-remote"),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("first-remote"),
            "first configured candidate wins"
        );
    }

    #[tokio::test]
    async fn configured_order_preserves_stale_candidate_position() {
        let f = make_filter_with_fresh(&[
            ("inference_model", "llama", "site-b", "stale-remote", false),
            ("inference_model", "llama", "site-c", "fresh-remote", true),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("stale-remote"),
            "freshness must not reorder candidates; the source owns overlay ordering"
        );
    }

    #[tokio::test]
    async fn configured_order_beats_freshness_and_locality() {
        let f = make_filter_with_fresh(&[
            ("inference_model", "llama", "site-a", "stale-local", false),
            ("inference_model", "llama", "site-b", "fresh-remote", true),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("stale-local"),
            "intelligent_route must preserve overlay/config order after admission filtering"
        );
    }

    #[tokio::test]
    async fn configured_order_beats_stale_locality() {
        let f = make_filter_with_fresh(&[
            ("inference_model", "llama", "site-b", "stale-remote", false),
            ("inference_model", "llama", "site-a", "stale-local", false),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("stale-remote"),
            "locality must not reorder candidates"
        );
    }

    // ---- Route metadata ----

    #[tokio::test]
    async fn route_metadata_reflects_ordered_winner() {
        let f = make_filter(&[
            ("inference_model", "llama", "site-b", "remote-inf"),
            ("inference_model", "llama", "site-a", "local-inf"),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata("intelligent_route.cluster"), Some("remote-inf"));
        assert_eq!(ctx.get_metadata("intelligent_route.kind"), Some("inference_model"));
        assert_eq!(ctx.get_metadata("intelligent_route.name"), Some("llama"));
        assert_eq!(ctx.get_metadata("intelligent_route.site"), Some("site-b"));
        assert_eq!(ctx.get_metadata("intelligent_route.local_site"), Some("site-a"));
    }

    #[tokio::test]
    async fn local_route_writes_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "local-inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata("intelligent_route.cluster"), Some("local-inf"));
        assert_eq!(ctx.get_metadata("intelligent_route.local_site"), Some("site-a"));
    }

    #[tokio::test]
    async fn remote_route_writes_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-b", "remote-gw")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata("intelligent_route.cluster"), Some("remote-gw"));
        assert_eq!(ctx.get_metadata("intelligent_route.site"), Some("site-b"));
    }

    #[tokio::test]
    async fn unknown_model_writes_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("unknown"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn blank_model_writes_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static(""));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn missing_header_writes_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_no_route_metadata(&ctx);
    }

    #[tokio::test]
    async fn preserved_cluster_writes_no_metadata() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let req = crate::test_utils::make_request(Method::POST, "/chat");
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.cluster = Some(Arc::from("pre-set"));

        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert!(
            ctx.get_metadata("intelligent_route.kind").is_none(),
            "preserved cluster path should not write route metadata"
        );
    }

    // ---- Overlay config validation ----

    #[test]
    fn from_config_overlay_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"llama","site":"site-a","cluster":"local","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!("overlay_file: {}\nreload:\n  enabled: false\n", path.display());
        assert!(parse(&yaml).is_ok(), "overlay_file config should parse");
    }

    #[test]
    fn from_config_overlay_file_missing() {
        let yaml = "overlay_file: /nonexistent/routing-config.json\nreload:\n  enabled: false\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string().contains("failed to read"),
            "missing overlay file should error: {err}"
        );
    }

    #[test]
    fn from_config_both_candidates_and_overlay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!(
            "overlay_file: {}\nlocal_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n",
            path.display()
        );
        let err = parse_err(&yaml);
        assert!(
            err.to_string().contains("cannot set both"),
            "both overlay_file and candidates should be rejected: {err}"
        );
    }

    #[test]
    fn from_config_neither_candidates_nor_overlay() {
        let yaml = "model_header: X-Model\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string().contains("either overlay_file or candidates"),
            "neither source should be rejected: {err}"
        );
    }

    #[test]
    fn from_config_static_backwards_compat() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: llama\n    site: site-a\n    cluster: inf\n    fresh: true\n";
        assert!(parse(yaml).is_ok(), "existing static config should still work");
    }

    #[tokio::test]
    async fn on_request_after_snapshot_swap() {
        let shared = Arc::new(ArcSwap::from_pointee(make_snapshot("cluster-v1")));
        let filter = IntelligentRouteFilter {
            model_header: HeaderName::from_static("x-model"),
            _reload_handle: None,
            provider_hop_clusters: BTreeSet::new(),
            session_affinity: None,
            snapshot: Arc::clone(&shared),
        };

        assert_eq!(route_model(&filter, "llama").await.as_deref(), Some("cluster-v1"));

        shared.store(Arc::new(make_snapshot("cluster-v2")));

        assert_eq!(route_model(&filter, "llama").await.as_deref(), Some("cluster-v2"));
    }

    #[test]
    fn reload_config_defaults() {
        let cfg: ReloadConfig = serde_yaml::from_str("{}").unwrap();
        assert!(cfg.enabled, "default enabled should be true");
        assert_eq!(cfg.debounce_ms, overlay::DEFAULT_DEBOUNCE_MS, "default debounce_ms");
    }

    #[test]
    fn reload_config_custom_debounce() {
        let cfg: ReloadConfig = serde_yaml::from_str("debounce_ms: 1000\n").unwrap();
        assert_eq!(cfg.debounce_ms, 1000);
    }

    #[test]
    fn reload_disabled_no_watcher() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!("overlay_file: {}\nreload:\n  enabled: false\n", path.display());
        let _filter = parse(&yaml).unwrap();
    }

    // ---- Static/dynamic config matrix (item 6) ----

    #[test]
    fn reload_block_with_static_candidates_rejected() {
        let yaml = "local_site: site-a\nreload:\n  enabled: true\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string()
                .contains("reload block is not valid with static candidates"),
            "reload block with static candidates should be rejected: {err}"
        );
    }

    #[test]
    fn reload_block_disabled_with_static_candidates_rejected() {
        let yaml = "local_site: site-a\nreload:\n  enabled: false\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string()
                .contains("reload block is not valid with static candidates"),
            "reload block with static candidates should be rejected even if enabled=false: {err}"
        );
    }

    #[test]
    fn overlay_without_reload_block_uses_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!("overlay_file: {}\n", path.display());
        assert!(parse(&yaml).is_ok(), "overlay without reload block should use defaults");
    }

    #[test]
    fn overlay_with_reload_enabled_false() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!("overlay_file: {}\nreload:\n  enabled: false\n", path.display());
        assert!(parse(&yaml).is_ok(), "overlay with reload disabled should work");
    }

    #[test]
    fn overlay_with_reload_enabled_true() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!(
            "overlay_file: {}\nreload:\n  enabled: true\n  debounce_ms: 100\n",
            path.display()
        );
        assert!(parse(&yaml).is_ok(), "overlay with reload enabled should work");
    }

    #[test]
    fn static_candidates_no_reload_block() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\n";
        assert!(
            parse(yaml).is_ok(),
            "static candidates without reload block should work"
        );
    }

    #[test]
    fn overlay_with_custom_debounce() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(
            &path,
            r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#,
        )
        .unwrap();
        let yaml = format!(
            "overlay_file: {}\nreload:\n  enabled: false\n  debounce_ms: 2000\n",
            path.display()
        );
        assert!(parse(&yaml).is_ok(), "overlay with custom debounce should work");
    }

    #[test]
    fn neither_source_no_reload_rejected() {
        let yaml = "model_header: X-Model\nreload:\n  enabled: true\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string().contains("either overlay_file or candidates"),
            "no source + reload should still be rejected: {err}"
        );
    }

    // ---- Selection contract (item 7) ----

    #[tokio::test]
    async fn reorder_after_reload_changes_selection() {
        let shared = Arc::new(ArcSwap::from_pointee(make_two_candidate_snapshot(
            "cluster-a",
            "site-b",
            "cluster-b",
            "site-a",
        )));
        let filter = IntelligentRouteFilter {
            model_header: HeaderName::from_static("x-model"),
            _reload_handle: None,
            provider_hop_clusters: BTreeSet::new(),
            session_affinity: None,
            snapshot: Arc::clone(&shared),
        };
        assert_eq!(route_model(&filter, "llama").await.as_deref(), Some("cluster-a"));

        shared.store(Arc::new(make_two_candidate_snapshot(
            "cluster-c",
            "site-c",
            "cluster-a",
            "site-a",
        )));
        assert_eq!(route_model(&filter, "llama").await.as_deref(), Some("cluster-c"));
    }

    #[tokio::test]
    async fn stale_first_preserved_when_source_orders_it_first() {
        let f = make_filter_with_fresh(&[
            ("inference_model", "llama", "site-a", "stale-local", false),
            ("inference_model", "llama", "site-b", "fresh-remote", true),
        ]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("stale-local"),
            "intelligent_route must preserve input order; the source is responsible for freshness ordering"
        );
    }

    // ---- Overlay bounded read at startup (item 3) ----

    #[test]
    fn from_config_overlay_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let content = vec![b'x'; (overlay::MAX_OVERLAY_SIZE + 1) as usize];
        std::fs::write(&path, &content).unwrap();
        let yaml = format!("overlay_file: {}\nreload:\n  enabled: false\n", path.display());
        let err = parse_err(&yaml);
        assert!(
            err.to_string().contains("exceeds"),
            "oversized overlay at startup should be rejected: {err}"
        );
    }

    // ---- Admission filtering ----

    #[tokio::test]
    async fn excluded_always_skipped() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a"), ("none", "c-b")]);
        let filter = make_affinity_filter(Arc::new(ArcSwap::from_pointee(snap)), None);
        assert_eq!(route_model(&filter, "llama").await.as_deref(), Some("c-a"));
    }

    #[tokio::test]
    async fn new_session_skips_existing_only() {
        let snap = make_overlay_snapshot(&[("existing_only", "c-a"), ("new_and_existing", "c-b")]);
        let filter = make_affinity_filter(Arc::new(ArcSwap::from_pointee(snap)), Some(make_test_affinity()));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("new-key"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-b"),
            "new session must skip existing_only"
        );
    }

    #[tokio::test]
    async fn affinity_disabled_skips_existing_only() {
        let snap = make_overlay_snapshot(&[("existing_only", "c-a"), ("new_and_existing", "c-b")]);
        let filter = make_affinity_filter(Arc::new(ArcSwap::from_pointee(snap)), None);
        assert_eq!(
            route_model(&filter, "llama").await.as_deref(),
            Some("c-b"),
            "without affinity every request is a new request and must skip existing_only"
        );
    }

    #[tokio::test]
    async fn no_session_key_skips_existing_only() {
        let snap = make_overlay_snapshot(&[("existing_only", "c-a"), ("new_and_existing", "c-b")]);
        let filter = make_affinity_filter(Arc::new(ArcSwap::from_pointee(snap)), Some(make_test_affinity()));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-b"),
            "affinity-enabled request without a key must skip existing_only"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("false"));
    }

    #[tokio::test]
    async fn existing_session_can_use_existing_only() {
        let snap = make_overlay_snapshot(&[("existing_only", "c-drain"), ("new_and_existing", "c-new")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        affinity.bindings.insert(
            "returning-user".to_owned(),
            Binding {
                expires: Instant::now() + Duration::from_secs(300),
                stable_id: Arc::from("inference_model/llama/s/c-drain"),
            },
        );
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers
            .insert("x-session-id", HeaderValue::from_static("returning-user"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-drain"),
            "existing session reuses bound candidate"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("true"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("true"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.failover"), Some("false"));
    }

    #[tokio::test]
    async fn failover_skips_existing_only_candidate() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-old")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        affinity.bindings.insert(
            "sess-1".to_owned(),
            Binding {
                expires: Instant::now() + Duration::from_secs(300),
                stable_id: Arc::from("inference_model/llama/s/c-old"),
            },
        );
        let filter = make_affinity_filter(Arc::clone(&shared), Some(affinity));

        let snap_v2 = make_overlay_snapshot(&[("existing_only", "c-drain"), ("new_and_existing", "c-new")]);
        shared.store(Arc::new(snap_v2));

        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("sess-1"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-new"),
            "failover must choose a new_and_existing candidate, not another existing_only candidate"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("true"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("false"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.failover"), Some("true"));
    }

    #[tokio::test]
    async fn no_new_and_existing_returns_404() {
        let snap = make_overlay_snapshot(&[("existing_only", "c-a"), ("none", "c-b")]);
        let filter = make_affinity_filter(Arc::new(ArcSwap::from_pointee(snap)), Some(make_test_affinity()));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers
            .insert("x-session-id", HeaderValue::from_static("brand-new"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "new session with only existing_only/excluded must get 404"
        );
    }

    // ---- Session affinity ----

    #[tokio::test]
    async fn first_request_creates_binding() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        let filter = make_affinity_filter(Arc::clone(&shared), Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("sess-1"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("true"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("false"));
        assert!(
            filter
                .session_affinity
                .as_ref()
                .unwrap()
                .bindings
                .contains_key("sess-1"),
            "binding should be created"
        );
    }

    #[tokio::test]
    async fn second_request_reuses_stable_id() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a"), ("new_and_existing", "c-b")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        affinity.bindings.insert(
            "sess-1".to_owned(),
            Binding {
                expires: Instant::now() + Duration::from_secs(300),
                stable_id: Arc::from("inference_model/llama/s/c-a"),
            },
        );
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("sess-1"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("true"));
    }

    #[tokio::test]
    async fn different_session_key_independent() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        let filter = make_affinity_filter(shared, Some(affinity));

        for key in &["sess-a", "sess-b"] {
            let mut req = crate::test_utils::make_request(Method::POST, "/chat");
            req.headers.insert("X-Model", HeaderValue::from_static("llama"));
            req.headers.insert("x-session-id", HeaderValue::from_str(key).unwrap());
            let mut ctx = crate::test_utils::make_filter_context(&req);
            let _unused = filter.on_request(&mut ctx).await.unwrap();
        }
        let aff = filter.session_affinity.as_ref().unwrap();
        assert!(aff.bindings.contains_key("sess-a"), "sess-a should be bound");
        assert!(aff.bindings.contains_key("sess-b"), "sess-b should be bound");
    }

    #[tokio::test]
    async fn expired_binding_treated_as_new() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        affinity.bindings.insert(
            "expired-sess".to_owned(),
            Binding {
                expires: Instant::now() - Duration::from_secs(1),
                stable_id: Arc::from("old-id"),
            },
        );
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers
            .insert("x-session-id", HeaderValue::from_static("expired-sess"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("true"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("false"));
    }

    #[tokio::test]
    async fn binding_survives_reload_with_same_stable_id() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        let filter = make_affinity_filter(Arc::clone(&shared), Some(affinity));

        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("sticky"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));

        let snap_v2 = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        shared.store(Arc::new(snap_v2));

        let mut req2 = crate::test_utils::make_request(Method::POST, "/chat");
        req2.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req2.headers.insert("x-session-id", HeaderValue::from_static("sticky"));
        let mut ctx2 = crate::test_utils::make_filter_context(&req2);
        let _unused = filter.on_request(&mut ctx2).await.unwrap();
        assert_eq!(ctx2.cluster.as_deref(), Some("c-a"));
        assert_eq!(ctx2.get_metadata("intelligent_route.session.reused"), Some("true"));
    }

    #[tokio::test]
    async fn missing_candidate_after_reload_fails_over() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a"), ("new_and_existing", "c-b")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        affinity.bindings.insert(
            "sess-1".to_owned(),
            Binding {
                expires: Instant::now() + Duration::from_secs(300),
                stable_id: Arc::from("inference_model/llama/s/c-a"),
            },
        );
        let filter = make_affinity_filter(Arc::clone(&shared), Some(affinity));

        let snap_v2 = make_overlay_snapshot(&[("new_and_existing", "c-b")]);
        shared.store(Arc::new(snap_v2));

        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("sess-1"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-b"),
            "should fail over to remaining candidate"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.reused"), Some("false"));
        assert_eq!(ctx.get_metadata("intelligent_route.session.failover"), Some("true"));
    }

    #[tokio::test]
    async fn no_session_key_no_binding() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert!(
            filter.session_affinity.as_ref().unwrap().bindings.is_empty(),
            "no session key means no binding stored"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("false"));
    }

    #[tokio::test]
    async fn capacity_limit_does_not_grow_unbounded() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = make_test_affinity();
        for i in 0..MAX_BINDINGS {
            affinity.bindings.insert(
                format!("fill-{i}"),
                Binding {
                    expires: Instant::now() + Duration::from_secs(3600),
                    stable_id: Arc::from("x"),
                },
            );
        }
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert("x-session-id", HeaderValue::from_static("overflow"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.cluster.as_deref(),
            Some("c-a"),
            "routing should succeed even at capacity"
        );
        let aff = filter.session_affinity.as_ref().unwrap();
        assert!(aff.bindings.len() <= MAX_BINDINGS, "bindings must not exceed capacity");
    }

    // ---- Session affinity config validation ----

    #[test]
    fn session_affinity_default_disabled() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: false\n";
        assert!(parse(yaml).is_ok(), "disabled session_affinity should parse");
    }

    #[test]
    fn session_affinity_enabled_requires_source() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string().contains("requires at least one"),
            "enabled without source should be rejected: {err}"
        );
    }

    #[test]
    fn session_affinity_blank_header_rejected() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n  header: \"   \"\n";
        let err = parse_err(yaml);
        assert!(
            err.to_string().contains("requires at least one"),
            "blank header should be rejected: {err}"
        );
    }

    #[test]
    fn session_affinity_ttl_bounds() {
        let yaml_zero = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n  header: x-session-id\n  ttl_secs: 0\n";
        let err = parse_err(yaml_zero);
        assert!(
            err.to_string().contains("ttl_secs must be"),
            "zero ttl should be rejected: {err}"
        );

        let yaml_high = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n  header: x-session-id\n  ttl_secs: 100000\n";
        let err2 = parse_err(yaml_high);
        assert!(
            err2.to_string().contains("ttl_secs must be"),
            "over-max ttl should be rejected: {err2}"
        );
    }

    #[test]
    fn session_affinity_valid_with_header() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n  header: x-session-id\n";
        assert!(parse(yaml).is_ok(), "valid session_affinity with header should parse");
    }

    #[test]
    fn session_affinity_valid_with_cookie() {
        let yaml = "local_site: site-a\ncandidates:\n  - kind: inference_model\n    name: m\n    site: s\n    cluster: c\n    fresh: true\nsession_affinity:\n  enabled: true\n  cookie: session_id\n";
        assert!(parse(yaml).is_ok(), "valid session_affinity with cookie should parse");
    }

    // ---- Route metadata with new fields ----

    #[tokio::test]
    async fn route_metadata_includes_stable_id_and_admission() {
        let f = make_filter(&[("inference_model", "llama", "site-a", "local-inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = f.on_request(&mut ctx).await.unwrap();
        assert!(ctx.get_metadata("intelligent_route.stable_id").is_some());
        assert_eq!(
            ctx.get_metadata("intelligent_route.admission_state"),
            Some("new_and_existing")
        );
    }

    #[tokio::test]
    async fn cookie_session_key_extraction() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = SessionAffinity {
            bindings: DashMap::new(),
            cookie: Some(Arc::from("sid")),
            header: None,
            ttl: Duration::from_secs(3600),
        };
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert(
            http::header::COOKIE,
            HeaderValue::from_static("other=x; sid=my-session; trail=y"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert!(
            filter
                .session_affinity
                .as_ref()
                .unwrap()
                .bindings
                .contains_key("my-session"),
            "cookie-extracted session key should create binding"
        );
    }

    #[tokio::test]
    async fn oversized_cookie_session_key_is_ignored() {
        let snap = make_overlay_snapshot(&[("new_and_existing", "c-a")]);
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let affinity = SessionAffinity {
            bindings: DashMap::new(),
            cookie: Some(Arc::from("sid")),
            header: None,
            ttl: Duration::from_secs(3600),
        };
        let filter = make_affinity_filter(shared, Some(affinity));
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let cookie = format!("sid={}", "x".repeat(MAX_HEADER_VALUE_LEN + 1));
        req.headers
            .insert(http::header::COOKIE, HeaderValue::from_str(&cookie).unwrap());
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.cluster.as_deref(), Some("c-a"));
        assert!(
            filter.session_affinity.as_ref().unwrap().bindings.is_empty(),
            "oversized cookie value must not be stored as a session key"
        );
        assert_eq!(ctx.get_metadata("intelligent_route.session.bound"), Some("false"));
    }

    // ---- Provider-hop context ----

    #[tokio::test]
    async fn provider_context_disabled_strips_spoofed_headers_without_setting_replacements() {
        let filter = make_provider_context_filter(false);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_static("spoofed-candidate"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_static("spoofed-request"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_provider_headers_removed(&ctx);
        assert!(ctx.request_headers_to_set.is_empty());
    }

    #[tokio::test]
    async fn provider_context_enabled_overwrites_spoofed_headers() {
        let filter = make_provider_context_filter(true);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_static("spoofed-candidate"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_static("spoofed-request"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_provider_headers_removed(&ctx);
        assert_eq!(
            pending_header(&ctx, SELECTED_CANDIDATE_HEADER),
            Some("inference_model/llama/site-b/provider-gateway")
        );
        let request_id = pending_header(&ctx, PROVIDER_HOP_REQUEST_ID_HEADER);
        assert!(request_id.is_some_and(|value| value != "spoofed-request"));
        assert_eq!(
            ctx.request_headers_to_set.len(),
            2,
            "only fixed peer-context headers may be set"
        );
    }

    #[tokio::test]
    async fn provider_context_contains_no_credential_or_provider_output_fields() {
        let filter = make_provider_context_filter(true);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        let emitted = ctx
            .request_headers_to_set
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            emitted,
            [SELECTED_CANDIDATE_HEADER, PROVIDER_HOP_REQUEST_ID_HEADER],
            "edge may emit only the two fixed peer-context fields"
        );
        for (_, value) in &ctx.request_headers_to_set {
            let value = value.to_str().unwrap();
            assert!(!value.contains("credential"));
            assert!(!value.contains("secret"));
        }
    }

    #[tokio::test]
    async fn provider_context_allowlist_excludes_direct_cluster() {
        let filter = parse(
            "local_site: site-a\n\
             provider_hop_clusters: [provider-gateway]\n\
             candidates:\n\
             \x20 - kind: inference_model\n\
             \x20   name: llama\n\
             \x20   site: site-a\n\
             \x20   cluster: direct-backend\n",
        )
        .unwrap();
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(ctx.cluster.as_deref(), Some("direct-backend"));
        assert_provider_headers_removed(&ctx);
        assert!(
            ctx.request_headers_to_set.is_empty(),
            "direct candidates must not receive provider-hop context"
        );
    }

    #[test]
    fn provider_hop_cluster_allowlist_rejects_blank_and_duplicates() {
        let candidate = "candidates:\n  - kind: inference_model\n    name: llama\n    site: site-a\n    cluster: provider-gateway\n";
        for list in ["['']", "[provider-gateway, provider-gateway]"] {
            let yaml = format!("local_site: site-a\nprovider_hop_clusters: {list}\n{candidate}");
            assert!(parse(&yaml).is_err(), "invalid allowlist must fail: {list}");
        }
    }

    #[tokio::test]
    async fn preselected_cluster_still_strips_spoofed_provider_context() {
        let filter = make_provider_context_filter(true);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_static("spoofed-candidate"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_static("spoofed-request"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);
        ctx.cluster = Some(Arc::from("already-selected"));

        let action = filter.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert_provider_headers_removed(&ctx);
        assert!(ctx.request_headers_to_set.is_empty());
    }

    #[test]
    fn provider_context_rejects_empty_and_oversized_stable_ids() {
        for stable_id in [String::new(), "x".repeat(MAX_HEADER_VALUE_LEN + 1)] {
            let mut snapshot = make_snapshot("provider-gateway");
            snapshot.candidates[0].stable_id = Arc::from(stable_id.as_str());
            let req = crate::test_utils::make_request(Method::POST, "/chat");
            let mut ctx = crate::test_utils::make_filter_context(&req);
            let allowlist = BTreeSet::from(["provider-gateway".to_owned()]);
            let error = write_provider_context(&mut ctx, &snapshot.candidates[0], &allowlist, None)
                .expect_err("invalid candidate ID must fail closed");
            assert!(error.to_string().contains("candidate ID"));
        }
    }

    #[tokio::test]
    async fn affinity_reuse_preserves_candidate_id_and_rotates_hop_request_id() {
        let snapshot = Arc::new(ArcSwap::from_pointee(make_overlay_snapshot(&[(
            "new_and_existing",
            "provider-gateway",
        )])));
        let filter = IntelligentRouteFilter {
            model_header: HeaderName::from_static("x-model"),
            _reload_handle: None,
            provider_hop_clusters: BTreeSet::from(["provider-gateway".to_owned()]),
            session_affinity: Some(make_test_affinity()),
            snapshot,
        };

        let mut first = crate::test_utils::make_request(Method::POST, "/chat");
        first.headers.insert("X-Model", HeaderValue::from_static("llama"));
        first
            .headers
            .insert("x-session-id", HeaderValue::from_static("provider-session"));
        let mut first_ctx = crate::test_utils::make_filter_context(&first);
        let _first_action = filter.on_request(&mut first_ctx).await.unwrap();

        let mut second = crate::test_utils::make_request(Method::POST, "/chat");
        second.headers.insert("X-Model", HeaderValue::from_static("llama"));
        second
            .headers
            .insert("x-session-id", HeaderValue::from_static("provider-session"));
        let mut second_ctx = crate::test_utils::make_filter_context(&second);
        let _second_action = filter.on_request(&mut second_ctx).await.unwrap();

        assert_eq!(
            pending_header(&first_ctx, SELECTED_CANDIDATE_HEADER),
            pending_header(&second_ctx, SELECTED_CANDIDATE_HEADER)
        );
        assert_ne!(
            pending_header(&first_ctx, PROVIDER_HOP_REQUEST_ID_HEADER),
            pending_header(&second_ctx, PROVIDER_HOP_REQUEST_ID_HEADER)
        );
        assert_eq!(
            second_ctx.get_metadata("intelligent_route.session.reused"),
            Some("true")
        );
    }

    // ---- Overlay revision header ----

    #[tokio::test]
    async fn envelope_snapshot_emits_revision_header() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/valid-minimal.json");
        let snap = RouteSnapshot::from_overlay(fixture).unwrap();
        let expected_rev = snap.semantic_revision.clone().unwrap();
        let shared = Arc::new(ArcSwap::from_pointee(snap));
        let filter = IntelligentRouteFilter {
            model_header: HeaderName::from_static("x-model"),
            _reload_handle: None,
            provider_hop_clusters: BTreeSet::from(["cluster-a".to_owned()]),
            session_affinity: None,
            snapshot: shared,
        };
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("model-a"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let action = filter.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        assert_eq!(
            pending_header(&ctx, OVERLAY_REVISION_HEADER),
            Some(expected_rev.as_ref()),
            "envelope revision must be emitted as a request header"
        );
    }

    #[tokio::test]
    async fn legacy_snapshot_does_not_emit_revision_header() {
        let filter = make_provider_context_filter(true);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            pending_header(&ctx, OVERLAY_REVISION_HEADER).is_none(),
            "legacy/static snapshots must not emit overlay revision header"
        );
    }

    #[tokio::test]
    async fn client_supplied_revision_header_stripped() {
        let filter = make_filter(&[("inference_model", "llama", "site-a", "inf")]);
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_static("llama"));
        req.headers.insert(
            HeaderName::from_static(OVERLAY_REVISION_HEADER),
            HeaderValue::from_static("spoofed-revision"),
        );
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        assert!(
            ctx.request_headers_to_remove
                .contains(&HeaderName::from_static(OVERLAY_REVISION_HEADER)),
            "client-supplied revision header must be stripped"
        );
    }

    // ---- Overlay envelope config ----

    #[test]
    fn expected_overlay_scope_accepted_in_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-overlay.json");
        std::fs::write(
            &path,
            include_str!("../../../tests/fixtures/overlay-contract/v1/valid-minimal.json"),
        )
        .unwrap();
        let yaml = format!(
            "overlay_file: {}\nreload:\n  enabled: false\nexpected_overlay_scope:\n  network: test-net\n  gateway: gw\n  namespace: ns\n  local_site: site-a\n",
            path.display()
        );
        assert!(parse(&yaml).is_ok(), "valid expected_overlay_scope should parse");
    }

    #[test]
    fn expected_overlay_scope_mismatch_fails_at_startup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-overlay.json");
        std::fs::write(
            &path,
            include_str!("../../../tests/fixtures/overlay-contract/v1/valid-minimal.json"),
        )
        .unwrap();
        let yaml = format!(
            "overlay_file: {}\nreload:\n  enabled: false\nexpected_overlay_scope:\n  network: wrong-network\n",
            path.display()
        );
        let err = parse_err(&yaml);
        assert!(
            err.to_string().contains("expected scope.network"),
            "scope mismatch at startup should be rejected: {err}"
        );
    }

    #[test]
    fn expected_overlay_scope_rejected_with_static_candidates() {
        let yaml = "local_site: site-a\n\
                    expected_overlay_scope:\n\
                    \x20 network: test-net\n\
                    candidates:\n\
                    \x20 - kind: inference_model\n\
                    \x20   name: model-a\n\
                    \x20   site: site-a\n\
                    \x20   cluster: cluster-a\n";
        let error = parse_err(yaml);
        assert!(
            error
                .to_string()
                .contains("expected_overlay_scope requires envelope overlay_file mode"),
            "unexpected static scope error: {error}"
        );
    }

    // ---- Test utilities ----

    fn assert_no_route_metadata(ctx: &HttpFilterContext<'_>) {
        for key in &[
            "intelligent_route.admission_state",
            "intelligent_route.cluster",
            "intelligent_route.kind",
            "intelligent_route.local_site",
            "intelligent_route.name",
            "intelligent_route.site",
            "intelligent_route.stable_id",
        ] {
            assert!(ctx.get_metadata(key).is_none(), "{key} should be absent");
        }
    }

    fn parse(yaml: &str) -> Result<Box<dyn HttpFilter>, FilterError> {
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        IntelligentRouteFilter::from_config(&val)
    }

    fn parse_err(yaml: &str) -> FilterError {
        parse(yaml).err().expect("config should have been rejected")
    }

    fn make_filter(candidates: &[(&str, &str, &str, &str)]) -> Box<dyn HttpFilter> {
        let with_fresh: Vec<(&str, &str, &str, &str, bool)> =
            candidates.iter().map(|(k, n, s, c)| (*k, *n, *s, *c, true)).collect();
        make_filter_with_fresh(&with_fresh)
    }

    fn make_filter_with_fresh(candidates: &[(&str, &str, &str, &str, bool)]) -> Box<dyn HttpFilter> {
        use std::fmt::Write as _;

        let mut yaml = String::from("local_site: site-a\ncandidates:\n");
        for (kind, name, site, cluster, fresh) in candidates {
            writeln!(
                yaml,
                "  - kind: {kind}\n    name: {name}\n    site: {site}\n    cluster: {cluster}\n    fresh: {fresh}"
            )
            .expect("String write is infallible");
        }
        parse(&yaml).unwrap()
    }

    fn make_provider_context_filter(enabled: bool) -> Box<dyn HttpFilter> {
        let provider_hop_clusters = if enabled {
            "provider_hop_clusters: [provider-gateway]\n"
        } else {
            ""
        };
        parse(&format!(
            "local_site: site-a\n\
             {provider_hop_clusters}\
             candidates:\n\
             \x20 - kind: inference_model\n\
             \x20   name: llama\n\
             \x20   site: site-b\n\
             \x20   cluster: provider-gateway\n"
        ))
        .unwrap()
    }

    fn assert_provider_headers_removed(ctx: &HttpFilterContext<'_>) {
        for name in [
            SELECTED_CANDIDATE_HEADER,
            PROVIDER_HOP_REQUEST_ID_HEADER,
            OVERLAY_REVISION_HEADER,
        ] {
            assert!(
                ctx.request_headers_to_remove.contains(&HeaderName::from_static(name)),
                "{name} must be removed"
            );
        }
    }

    fn pending_header<'a>(ctx: &'a HttpFilterContext<'_>, name: &'static str) -> Option<&'a str> {
        ctx.request_headers_to_set
            .iter()
            .find(|(header, _)| header == name)
            .and_then(|(_, value)| value.to_str().ok())
    }

    fn make_two_candidate_snapshot(cluster1: &str, site1: &str, cluster2: &str, site2: &str) -> RouteSnapshot {
        RouteSnapshot::from_static(
            descriptor::validate_candidates(vec![
                CandidateConfig {
                    cluster: cluster1.to_owned(),
                    credential: None,
                    fresh: true,
                    kind: CapabilityKind::InferenceModel,
                    name: "llama".to_owned(),
                    site: site1.to_owned(),
                },
                CandidateConfig {
                    cluster: cluster2.to_owned(),
                    credential: None,
                    fresh: true,
                    kind: CapabilityKind::InferenceModel,
                    name: "llama".to_owned(),
                    site: site2.to_owned(),
                },
            ])
            .unwrap(),
            Arc::from("site-a"),
        )
    }

    fn make_snapshot(cluster: &str) -> RouteSnapshot {
        RouteSnapshot::from_static(
            descriptor::validate_candidates(vec![CandidateConfig {
                cluster: cluster.to_owned(),
                credential: None,
                fresh: true,
                kind: CapabilityKind::InferenceModel,
                name: "llama".to_owned(),
                site: "site-a".to_owned(),
            }])
            .unwrap(),
            Arc::from("site-a"),
        )
    }

    async fn route_model(filter: &IntelligentRouteFilter, model: &str) -> Option<String> {
        let mut req = crate::test_utils::make_request(Method::POST, "/chat");
        req.headers.insert("X-Model", HeaderValue::from_str(model).unwrap());
        let mut ctx = crate::test_utils::make_filter_context(&req);
        let _unused = filter.on_request(&mut ctx).await.unwrap();
        ctx.cluster.as_deref().map(str::to_owned)
    }

    fn make_test_affinity() -> SessionAffinity {
        SessionAffinity {
            bindings: DashMap::new(),
            cookie: None,
            header: Some(HeaderName::from_static("x-session-id")),
            ttl: Duration::from_secs(3600),
        }
    }

    fn make_affinity_filter(
        snapshot: Arc<ArcSwap<RouteSnapshot>>,
        session_affinity: Option<SessionAffinity>,
    ) -> IntelligentRouteFilter {
        IntelligentRouteFilter {
            model_header: HeaderName::from_static("x-model"),
            _reload_handle: None,
            provider_hop_clusters: BTreeSet::new(),
            session_affinity,
            snapshot,
        }
    }

    fn make_overlay_snapshot(candidates: &[(&str, &str)]) -> RouteSnapshot {
        let mut route_candidates = Vec::new();
        for (admission_str, cluster) in candidates {
            route_candidates.push(RouteCandidate {
                admission_state: AdmissionState::from_overlay_str(admission_str).unwrap(),
                cluster: Arc::from(*cluster),
                credential: None,
                fresh: true,
                kind: CapabilityKind::InferenceModel,
                name: Arc::from("llama"),
                rank: None,
                selection_tier: None,
                site: Arc::from("s"),
                stable_id: descriptor::default_stable_id(CapabilityKind::InferenceModel, "llama", "s", cluster),
            });
        }
        RouteSnapshot::from_static(route_candidates, Arc::from("site-a"))
    }
}
