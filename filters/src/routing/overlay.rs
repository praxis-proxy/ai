// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Routing overlay types, snapshot, and file watcher.
//!
//! Provides [`RouteSnapshot`] (the atomic unit of routing state swapped
//! by [`ArcSwap`]) and the file watcher that detects routing config
//! changes and performs filter-local hot reload.
//!
//! The overlay wire types ([`OverlayDocument`], [`OverlayCandidate`])
//! mirror the JSON structure rendered by the operator into a
//! Kubernetes `ConfigMap`.  Only the fields needed for routing are
//! consumed, including credential references without secret values.
//!
//! [`ArcSwap`]: arc_swap::ArcSwap

use std::{
    fmt::Write as _,
    io::Read as _,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use arc_swap::ArcSwap;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher as _};
use praxis_filter::FilterError;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    descriptor::{self, AdmissionState, CandidateConfig, CapabilityKind, RouteCandidate},
    metadata::{CandidateCredential, CredentialRef},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default debounce window for overlay file events (milliseconds).
pub(crate) const DEFAULT_DEBOUNCE_MS: u64 = 500;

/// Maximum overlay file size (2 MiB).
///
/// Kubernetes `ConfigMaps` are limited to 1 MiB, so this provides
/// headroom while preventing unbounded memory allocation from a
/// misconfigured or malicious mount.
pub(crate) const MAX_OVERLAY_SIZE: u64 = 2 * 1024 * 1024;

/// Timeout for joining the watcher thread during [`Drop`].
///
/// The watcher is cancellation-aware, so it should exit within one
/// debounce window after shutdown is signalled.  If it has not exited
/// after this timeout, a warning is logged and the thread is detached.
const JOIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Maximum time to wait for the overlay watcher to register.
const WATCHER_START_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum length of envelope scope and provenance string fields.
const MAX_ENVELOPE_FIELD_LEN: usize = 256;

// -----------------------------------------------------------------------------
// Contract format
// -----------------------------------------------------------------------------

/// Whether the snapshot was loaded from the versioned envelope or
/// the legacy flat payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContractFormat {
    /// Versioned content-addressed envelope (`routing-overlay.json`).
    Envelope,
    /// Legacy flat overlay (`routing-config.json`).
    Legacy,
}

// -----------------------------------------------------------------------------
// Envelope wire types (JSON)
// -----------------------------------------------------------------------------

/// Versioned content-addressed envelope wrapping a [`OverlayDocument`].
///
/// Does **not** use `deny_unknown_fields` — the configuration producer may add
/// new envelope metadata before AI is updated.
#[derive(Debug, Deserialize)]
struct EnvelopeDocument {
    /// Envelope schema version (must be `"1.0.0"`).
    schema_version: String,

    /// Content-addressed revision.
    revision: RevisionField,

    /// Content digest (must equal revision in v1).
    content_digest: DigestField,

    /// Scope identifying the producing gateway and network.
    scope: ScopeField,

    /// Provenance metadata — validated but not consumed for routing decisions.
    provenance: ProvenanceField,

    /// The embedded routing overlay.
    overlay: OverlayDocument,
}

/// Content-addressed revision field.
#[derive(Debug, Deserialize)]
struct RevisionField {
    /// Revision kind (must be `"content_addressed"`).
    kind: String,

    /// Hash algorithm (must be `"sha256"`).
    algorithm: String,

    /// 64 lowercase hex character hash value.
    value: String,
}

/// Content digest field.
///
/// In v1, the digest has `algorithm` and `value` but no `kind`
/// (the kind is implicit from the revision).
#[derive(Debug, Deserialize)]
struct DigestField {
    /// Hash algorithm (must be `"sha256"`).
    algorithm: String,

    /// 64 lowercase hex character hash value.
    value: String,
}

/// Scope field identifying the producing gateway.
#[derive(Debug, Deserialize)]
struct ScopeField {
    /// Source network name.
    network: String,

    /// Gateway name.
    gateway: String,

    /// Namespace.
    namespace: String,

    /// Local site identifier.
    local_site: String,
}

/// Provenance metadata for audit and debugging.
///
/// Validated for structural presence and scope consistency, but not
/// consumed for routing decisions.  Does **not** use `deny_unknown_fields`
/// — the configuration producer may add new provenance fields.
#[derive(Debug, Deserialize)]
struct ProvenanceField {
    /// Producer identifier (e.g. `"grid-operator"`).
    producer: String,

    /// Producer version.
    producer_version: String,

    /// Routing network this overlay was produced for — must match `scope.network`.
    source_name: String,

    /// Opaque identifier of the source resource that rendered this overlay.
    source_uid: String,

    /// Monotonic generation counter of the source resource.
    source_generation: u64,

    /// ISO-8601 timestamp when the overlay was rendered.
    rendered_at: String,
}

/// Expected overlay scope for validation at load/reload time.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExpectedOverlayScope {
    /// Expected network name.
    #[serde(default)]
    pub(crate) network: Option<String>,

    /// Expected gateway name.
    #[serde(default)]
    pub(crate) gateway: Option<String>,

    /// Expected namespace.
    #[serde(default)]
    pub(crate) namespace: Option<String>,

    /// Expected local site.
    #[serde(default)]
    pub(crate) local_site: Option<String>,
}

/// Supported schema version.
const SUPPORTED_SCHEMA_VERSION: &str = "1.0.0";

// -----------------------------------------------------------------------------
// Overlay wire types (JSON)
// -----------------------------------------------------------------------------

/// Top-level routing overlay document as rendered by the operator.
///
/// Serialised as JSON in the overlay `ConfigMap`.
#[derive(Debug, Deserialize)]
pub(crate) struct OverlayDocument {
    /// Local site identifier for scoring and metadata.
    pub(crate) local_site: String,

    /// Routing candidates, ordered by the scoring engine.
    pub(crate) candidates: Vec<OverlayCandidate>,

    /// ISO-8601 timestamp when the overlay was generated.
    #[serde(default)]
    pub(crate) generated_at: Option<String>,

    /// Network name.  Used for scope validation in envelope mode.
    #[serde(default)]
    pub(super) network: Option<String>,
}

/// A single routing candidate from the overlay.
///
/// Does **not** use `deny_unknown_fields` — the configuration producer may
/// add new metadata fields before AI is updated.
#[derive(Debug, Deserialize)]
pub(crate) struct OverlayCandidate {
    /// Admission state string from the configuration producer.
    #[serde(default)]
    pub(crate) admission_state: Option<String>,

    /// Upstream cluster identifier.
    pub(crate) cluster: String,

    /// Credential reference projected by the operator.
    ///
    /// Carried as bounded in-process metadata for `credential_inject`.
    #[serde(default)]
    credential: Option<OverlayCredential>,

    /// Whether this candidate is considered fresh by the configuration producer.
    #[serde(default = "default_fresh")]
    pub(crate) fresh: bool,

    /// Capability kind string (e.g. `"inference_model"`, `"mcp_tool"`).
    pub(crate) kind: String,

    /// Capability name (model name, tool name).
    pub(crate) name: String,

    /// Producer-assigned rank within the overlay (lower is better).
    #[serde(default)]
    pub(crate) rank: Option<u32>,

    /// Producer-assigned locality tier (e.g. `"same_region"`).
    #[serde(default)]
    pub(crate) selection_tier: Option<String>,

    /// Site that owns this capability.
    pub(crate) site: String,

    /// Deterministic identifier assigned by the configuration producer.
    #[serde(default)]
    pub(crate) stable_id: Option<String>,
}

/// Default freshness for overlay candidates.
fn default_fresh() -> bool {
    true
}

/// Projected credential reference from the routing overlay.
///
/// Contains only the Secret reference — never the token value.
/// Uses `deny_unknown_fields` to reject token-like fields that
/// should never appear in the overlay.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlayCredential {
    /// Authentication strategy (e.g. `"bearer_token"`).
    strategy: String,

    /// Reference to the Secret holding the credential.
    #[serde(rename = "secretRef", alias = "secret_ref")]
    secret_ref: OverlaySecretRef,
}

/// Secret reference within a projected credential.
///
/// Uses `deny_unknown_fields` to reject fields like `value` or
/// `token` that might contain actual secret material.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlaySecretRef {
    /// Secret name.
    name: String,

    /// Secret namespace.
    namespace: String,

    /// Key within the Secret data.
    key: String,
}

// -----------------------------------------------------------------------------
// RouteSnapshot
// -----------------------------------------------------------------------------

/// Atomic snapshot of routing state loaded by `intelligent_route` on each request.
///
/// Stored behind [`ArcSwap`] so the watcher can swap in new state while
/// in-flight requests continue using their loaded snapshot.
///
/// [`ArcSwap`]: arc_swap::ArcSwap
#[derive(Debug)]
pub(crate) struct RouteSnapshot {
    /// Validated route candidates.
    pub(crate) candidates: Vec<RouteCandidate>,

    /// Whether this snapshot was loaded from an envelope or legacy payload.
    pub(crate) contract_format: ContractFormat,

    /// SHA-256 digest of the raw overlay file content that produced
    /// this snapshot.  Used for change detection; `[0; 32]` for
    /// statically configured snapshots.
    pub(crate) content_hash: [u8; 32],

    /// ISO-8601 timestamp when the overlay was generated by the producer.
    #[expect(dead_code, reason = "stored for future freshness/staleness policy")]
    pub(crate) generated_at: Option<Arc<str>>,

    /// Local site identifier.
    pub(crate) local_site: Arc<str>,

    /// Envelope schema version (`"1.0.0"` for envelope, [`None`] for legacy).
    pub(crate) schema_version: Option<Arc<str>>,

    /// Semantic content-addressed revision (hex string for envelope,
    /// [`None`] for legacy/static).
    pub(crate) semantic_revision: Option<Arc<str>>,
}

impl RouteSnapshot {
    /// Build a snapshot from raw overlay file content.
    ///
    /// Detects the format (versioned envelope vs legacy flat payload)
    /// by checking for a `schema_version` field.  Envelope payloads
    /// are fully validated: schema version, revision shape, digest
    /// recomputation, scope consistency.  A malformed envelope never
    /// falls back to legacy parsing.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the JSON is invalid, a candidate kind
    /// is unrecognised, validation fails, or envelope integrity checks
    /// do not pass.
    #[cfg(test)]
    pub(crate) fn from_overlay(content: &[u8]) -> Result<Self, FilterError> {
        Self::from_overlay_with_scope(content, None)
    }

    /// Build a snapshot from raw overlay file content with optional
    /// scope validation.
    ///
    /// When `expected_scope` is provided, each specified field is
    /// compared against the envelope scope.  Mismatches are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] on any parse, validation, digest, or
    /// scope mismatch error.
    pub(crate) fn from_overlay_with_scope(
        content: &[u8],
        expected_scope: Option<&ExpectedOverlayScope>,
    ) -> Result<Self, FilterError> {
        let content_hash: [u8; 32] = Sha256::digest(content).into();

        let value: serde_json::Value = serde_json::from_slice(content)
            .map_err(|e| FilterError::from(format!("routing: overlay parse error: {e}")))?;

        if has_envelope_signal(&value) {
            Self::from_envelope(content, content_hash, &value, expected_scope)
        } else if expected_scope.is_some() {
            Err(
                "routing: expected envelope format (expected_overlay_scope is configured) but received legacy payload"
                    .into(),
            )
        } else {
            Self::from_legacy(content, content_hash)
        }
    }

    /// Parse and validate a versioned envelope payload.
    #[expect(
        clippy::too_many_lines,
        reason = "sequential envelope validation with scope/digest checks"
    )]
    fn from_envelope(
        content: &[u8],
        content_hash: [u8; 32],
        value: &serde_json::Value,
        expected_scope: Option<&ExpectedOverlayScope>,
    ) -> Result<Self, FilterError> {
        let envelope: EnvelopeDocument = serde_json::from_slice(content)
            .map_err(|e| FilterError::from(format!("routing: envelope parse error: {e}")))?;

        if envelope.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(format!(
                "routing: unsupported envelope schema version: {}",
                envelope.schema_version
            )
            .into());
        }

        validate_envelope_metadata(&envelope)?;
        validate_revision_shape(&envelope.revision, "revision")?;
        validate_revision_shape_digest(&envelope.content_digest, "content_digest")?;

        if envelope.revision.value != envelope.content_digest.value {
            return Err("routing: envelope revision and content_digest disagree".into());
        }

        let overlay_value = value
            .get("overlay")
            .ok_or_else(|| FilterError::from("routing: envelope missing overlay field"))?;
        let recomputed = compute_semantic_digest(overlay_value)?;
        if recomputed != envelope.content_digest.value {
            return Err(format!(
                "routing: envelope digest mismatch: expected {}, computed {recomputed}",
                envelope.content_digest.value
            )
            .into());
        }

        let overlay_network = envelope
            .overlay
            .network
            .as_deref()
            .ok_or_else(|| FilterError::from("routing: envelope overlay.network is required"))?;
        if overlay_network != envelope.scope.network {
            return Err(format!(
                "routing: envelope scope.network ({}) != overlay.network ({overlay_network})",
                envelope.scope.network
            )
            .into());
        }
        if envelope.scope.local_site != envelope.overlay.local_site {
            return Err(format!(
                "routing: envelope scope.local_site ({}) != overlay.local_site ({})",
                envelope.scope.local_site, envelope.overlay.local_site
            )
            .into());
        }

        if envelope.provenance.source_name != envelope.scope.network {
            return Err(format!(
                "routing: envelope provenance.source_name ({}) != scope.network ({})",
                envelope.provenance.source_name, envelope.scope.network
            )
            .into());
        }

        if let Some(expected) = expected_scope {
            validate_expected_scope(expected, &envelope.scope)?;
        }

        descriptor::validate_local_site(&envelope.overlay.local_site)?;
        let candidates = overlay_to_candidates(&envelope.overlay)?;
        let generated_at = envelope.overlay.generated_at.map(|s| Arc::from(s.as_str()));

        Ok(Self {
            candidates,
            contract_format: ContractFormat::Envelope,
            content_hash,
            generated_at,
            local_site: Arc::from(envelope.overlay.local_site.as_str()),
            schema_version: Some(Arc::from(envelope.schema_version.as_str())),
            semantic_revision: Some(Arc::from(envelope.revision.value.as_str())),
        })
    }

    /// Parse a legacy flat overlay payload.
    fn from_legacy(content: &[u8], content_hash: [u8; 32]) -> Result<Self, FilterError> {
        let doc: OverlayDocument = serde_json::from_slice(content)
            .map_err(|e| FilterError::from(format!("routing: overlay parse error: {e}")))?;

        descriptor::validate_local_site(&doc.local_site)?;
        let candidates = overlay_to_candidates(&doc)?;
        let generated_at = doc.generated_at.map(|s| Arc::from(s.as_str()));

        Ok(Self {
            candidates,
            contract_format: ContractFormat::Legacy,
            content_hash,
            generated_at,
            local_site: Arc::from(doc.local_site.as_str()),
            schema_version: None,
            semantic_revision: None,
        })
    }

    /// Build a snapshot from statically configured candidates.
    ///
    /// The content hash is `[0; 32]` (never compared against file content).
    pub(crate) fn from_static(candidates: Vec<RouteCandidate>, local_site: Arc<str>) -> Self {
        Self {
            candidates,
            contract_format: ContractFormat::Legacy,
            content_hash: [0; 32],
            generated_at: None,
            local_site,
            schema_version: None,
            semantic_revision: None,
        }
    }
}

// -----------------------------------------------------------------------------
// Envelope validation helpers
// -----------------------------------------------------------------------------

/// Reserved top-level fields that signal an envelope document.
///
/// If any of these is present, the document is treated as an envelope
/// and must pass full envelope validation — no fallback to legacy parsing.
const ENVELOPE_SIGNAL_FIELDS: &[&str] = &["schema_version", "revision", "content_digest", "scope", "provenance"];

/// Returns `true` if the JSON value contains any reserved envelope field.
fn has_envelope_signal(value: &serde_json::Value) -> bool {
    ENVELOPE_SIGNAL_FIELDS.iter().any(|f| value.get(*f).is_some())
}

/// Validate shape of a revision field.
fn validate_revision_shape(field: &RevisionField, name: &str) -> Result<(), FilterError> {
    if field.kind != "content_addressed" {
        return Err(format!("routing: envelope {name}.kind must be \"content_addressed\"").into());
    }
    if field.algorithm != "sha256" {
        return Err(format!("routing: envelope {name}.algorithm must be \"sha256\"").into());
    }
    validate_hex_value(&field.value, name)
}

/// Validate shape of a digest field.
fn validate_revision_shape_digest(field: &DigestField, name: &str) -> Result<(), FilterError> {
    if field.algorithm != "sha256" {
        return Err(format!("routing: envelope {name}.algorithm must be \"sha256\"").into());
    }
    validate_hex_value(&field.value, name)
}

/// Validate that a value is exactly 64 lowercase hex characters.
fn validate_hex_value(value: &str, name: &str) -> Result<(), FilterError> {
    if value.len() != 64 {
        return Err(format!("routing: envelope {name}.value must be 64 hex characters").into());
    }
    if !value.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return Err(format!("routing: envelope {name}.value must be lowercase hex").into());
    }
    Ok(())
}

/// Validate bounded v1 scope and provenance fields.
fn validate_envelope_metadata(envelope: &EnvelopeDocument) -> Result<(), FilterError> {
    for (name, value) in [
        ("scope.network", envelope.scope.network.as_str()),
        ("scope.gateway", envelope.scope.gateway.as_str()),
        ("scope.namespace", envelope.scope.namespace.as_str()),
        ("scope.local_site", envelope.scope.local_site.as_str()),
        ("provenance.producer", envelope.provenance.producer.as_str()),
        (
            "provenance.producer_version",
            envelope.provenance.producer_version.as_str(),
        ),
        ("provenance.source_name", envelope.provenance.source_name.as_str()),
        ("provenance.source_uid", envelope.provenance.source_uid.as_str()),
        ("provenance.rendered_at", envelope.provenance.rendered_at.as_str()),
    ] {
        validate_bounded_nonblank(name, value, MAX_ENVELOPE_FIELD_LEN)?;
    }
    if envelope.provenance.source_generation == 0 {
        return Err("routing: envelope provenance.source_generation must be positive".into());
    }
    chrono::DateTime::parse_from_rfc3339(&envelope.provenance.rendered_at).map_err(|error| {
        FilterError::from(format!(
            "routing: envelope provenance.rendered_at is not RFC 3339: {error}"
        ))
    })?;
    Ok(())
}

/// Validate a bounded, non-whitespace string.
fn validate_bounded_nonblank(field: &str, value: &str, max_len: usize) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > max_len {
        return Err(format!("routing: envelope {field} must be 1-{max_len} non-blank bytes").into());
    }
    Ok(())
}

/// Recompute the semantic digest of an overlay [`serde_json::Value`].
///
/// Extracts `candidates`, `local_site`, and `network` from the value,
/// canonicalizes via RFC 8785, and computes SHA-256.
fn compute_semantic_digest(overlay_value: &serde_json::Value) -> Result<String, FilterError> {
    let mut semantic = serde_json::Map::new();
    if let Some(candidates) = overlay_value.get("candidates") {
        semantic.insert("candidates".to_owned(), candidates.clone());
    }
    if let Some(local_site) = overlay_value.get("local_site") {
        semantic.insert("local_site".to_owned(), local_site.clone());
    }
    if let Some(network) = overlay_value.get("network") {
        semantic.insert("network".to_owned(), network.clone());
    }
    let semantic_value = serde_json::Value::Object(semantic);
    let canonical = serde_json_canonicalizer::to_vec(&semantic_value)
        .map_err(|e| FilterError::from(format!("routing: canonicalization error: {e}")))?;
    let digest: [u8; 32] = Sha256::digest(&canonical).into();
    let mut hex = String::with_capacity(64);
    for b in &digest {
        let _unused = write!(hex, "{b:02x}");
    }
    Ok(hex)
}

/// Validate the envelope scope against expected values.
fn validate_expected_scope(expected: &ExpectedOverlayScope, scope: &ScopeField) -> Result<(), FilterError> {
    if let Some(network) = &expected.network
        && *network != scope.network
    {
        return Err(format!("routing: expected scope.network={network}, got {}", scope.network).into());
    }
    if let Some(gateway) = &expected.gateway
        && *gateway != scope.gateway
    {
        return Err(format!("routing: expected scope.gateway={gateway}, got {}", scope.gateway).into());
    }
    if let Some(namespace) = &expected.namespace
        && *namespace != scope.namespace
    {
        return Err(format!("routing: expected scope.namespace={namespace}, got {}", scope.namespace).into());
    }
    if let Some(local_site) = &expected.local_site
        && *local_site != scope.local_site
    {
        return Err(format!(
            "routing: expected scope.local_site={local_site}, got {}",
            scope.local_site
        )
        .into());
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Overlay → RouteCandidate conversion
// -----------------------------------------------------------------------------

/// Convert overlay candidates to validated [`RouteCandidate`]s.
fn overlay_to_candidates(doc: &OverlayDocument) -> Result<Vec<RouteCandidate>, FilterError> {
    let raw: Vec<CandidateConfig> = doc
        .candidates
        .iter()
        .map(|oc| {
            let kind = CapabilityKind::from_overlay_str(&oc.kind)?;
            Ok(CandidateConfig {
                cluster: oc.cluster.clone(),
                credential: oc.credential.as_ref().map(|credential| CandidateCredential {
                    strategy: credential.strategy.clone(),
                    secret_ref: CredentialRef {
                        name: credential.secret_ref.name.clone(),
                        namespace: credential.secret_ref.namespace.clone(),
                        key: credential.secret_ref.key.clone(),
                    },
                }),
                fresh: oc.fresh,
                kind,
                name: oc.name.clone(),
                site: oc.site.clone(),
            })
        })
        .collect::<Result<Vec<_>, FilterError>>()?;

    let mut candidates = descriptor::validate_candidates(raw)?;
    enrich_from_overlay(&mut candidates, &doc.candidates)?;
    Ok(candidates)
}

/// Apply producer-supplied metadata to validated candidates.
///
/// Zips the validated candidate list with the original overlay entries
/// and sets `admission_state`, `rank`, `selection_tier`, and `stable_id`.
/// Called after [`validate_candidates`] so `deny_unknown_fields` on
/// [`CandidateConfig`] is never bypassed.
///
/// [`validate_candidates`]: descriptor::validate_candidates
pub(super) fn enrich_from_overlay(
    candidates: &mut [RouteCandidate],
    overlay: &[OverlayCandidate],
) -> Result<(), FilterError> {
    for (i, (c, oc)) in candidates.iter_mut().zip(overlay.iter()).enumerate() {
        if let Some(s) = &oc.admission_state {
            c.admission_state = AdmissionState::from_overlay_str(s)
                .map_err(|e| FilterError::from(format!("routing: candidate {i}: {e}")))?;
        }
        c.rank = oc.rank;
        if let Some(t) = &oc.selection_tier {
            if t.trim().is_empty() || t.len() > 128 {
                return Err(format!("routing: candidate {i}: selection_tier must be 1-128 non-blank bytes").into());
            }
            c.selection_tier = Some(Arc::from(t.as_str()));
        }
        if let Some(id) = &oc.stable_id {
            if id.trim().is_empty() || id.len() > 256 || id.parse::<http::HeaderValue>().is_err() {
                return Err(
                    format!("routing: candidate {i}: stable_id must be a valid 1-256 byte header value").into(),
                );
            }
            c.stable_id = Arc::from(id.as_str());
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// OverlayReloadHandle
// -----------------------------------------------------------------------------

/// Handle to the overlay file watcher thread.
///
/// On [`Drop`], signals shutdown via [`CancellationToken`] and performs a
/// bounded join on the watcher thread (up to [`JOIN_TIMEOUT`]).  If the
/// thread has not exited by the timeout, a warning is logged and the
/// thread is detached — the `CancellationToken` remains signalled so the
/// thread will exit when it next checks.
pub(crate) struct OverlayReloadHandle {
    /// Shutdown signal for the watcher thread.
    shutdown: CancellationToken,

    /// Watcher thread join handle.
    thread: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for OverlayReloadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayReloadHandle")
            .field("shutdown_requested", &self.shutdown.is_cancelled())
            .finish()
    }
}

impl Drop for OverlayReloadHandle {
    #[expect(
        clippy::disallowed_methods,
        reason = "Drop is sync; tokio::time::sleep cannot be used here"
    )]
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(handle) = self.thread.take() {
            let start = std::time::Instant::now();
            while !handle.is_finished() {
                if start.elapsed() >= JOIN_TIMEOUT {
                    tracing::warn!(
                        timeout_secs = JOIN_TIMEOUT.as_secs(),
                        "intelligent_route: overlay watcher thread did not exit within timeout"
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(handle.join());
        }
    }
}

// -----------------------------------------------------------------------------
// Watcher
// -----------------------------------------------------------------------------

/// Spawn a background file watcher for the overlay file.
///
/// Returns a handle that cancels and joins the watcher on drop.
///
/// # Errors
///
/// Returns [`FilterError`] if the watcher thread, runtime, or filesystem
/// watcher cannot be initialized within [`WATCHER_START_TIMEOUT`].
#[cfg(test)]
pub(crate) fn spawn_overlay_watcher(
    path: PathBuf,
    snapshot: Arc<ArcSwap<RouteSnapshot>>,
    debounce_ms: u64,
) -> Result<OverlayReloadHandle, FilterError> {
    spawn_overlay_watcher_with_scope(path, snapshot, debounce_ms, None)
}

/// Spawn a file watcher with optional expected scope validation.
///
/// Blocks until the watcher thread confirms readiness (watcher
/// registered and authoritative re-read complete) or reports failure.
#[expect(
    clippy::too_many_lines,
    reason = "keeps watcher startup, readiness rendezvous, and cleanup together"
)]
pub(crate) fn spawn_overlay_watcher_with_scope(
    path: PathBuf,
    snapshot: Arc<ArcSwap<RouteSnapshot>>,
    debounce_ms: u64,
    expected_scope: Option<ExpectedOverlayScope>,
) -> Result<OverlayReloadHandle, FilterError> {
    let shutdown = CancellationToken::new();
    let token = shutdown.clone();

    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(0);

    let thread = std::thread::Builder::new()
        .name("routing-overlay-watcher".to_owned())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime,
                Err(error) => {
                    drop(ready_tx.send(Err(format!("failed to create watcher runtime: {error}"))));
                    return;
                },
            };
            rt.block_on(watch_loop(path, snapshot, debounce_ms, token, expected_scope, ready_tx));
        })
        .map_err(|error| FilterError::from(format!("routing: failed to spawn overlay watcher: {error}")))?;

    let handle = OverlayReloadHandle {
        shutdown,
        thread: Some(thread),
    };
    match ready_rx.recv_timeout(WATCHER_START_TIMEOUT) {
        Ok(Ok(())) => Ok(handle),
        Ok(Err(message)) => {
            drop(handle);
            Err(FilterError::from(format!(
                "routing: overlay watcher failed to initialize: {message}"
            )))
        },
        Err(error) => {
            drop(handle);
            Err(FilterError::from(format!(
                "routing: overlay watcher did not initialize within {} seconds: {error}",
                WATCHER_START_TIMEOUT.as_secs()
            )))
        },
    }
}

/// Core watch loop: set up the notify watcher, debounce events,
/// and trigger overlay reloads.
#[expect(clippy::too_many_arguments, reason = "readiness channel added for startup barrier")]
async fn watch_loop(
    path: PathBuf,
    snapshot: Arc<ArcSwap<RouteSnapshot>>,
    debounce_ms: u64,
    shutdown: CancellationToken,
    expected_scope: Option<ExpectedOverlayScope>,
    ready_tx: std::sync::mpsc::SyncSender<Result<(), String>>,
) {
    let (tx, mut rx) = mpsc::channel::<()>(16);

    let watch_dir = watch_dir_for_path(&path);

    let _watcher = match setup_watcher(tx, &watch_dir) {
        Ok(w) => w,
        Err(e) => {
            let msg = format!("{e}");
            tracing::error!(error = %e, "intelligent_route: failed to start overlay file watcher");
            drop(ready_tx.send(Err(msg)));
            return;
        },
    };

    // Authoritative re-read after watcher registration to close the
    // startup race window: any overlay change between the initial read
    // in build_overlay_snapshot and watcher registration is caught here.
    handle_overlay_reload(&path, &snapshot, expected_scope.as_ref());

    drop(ready_tx.send(Ok(())));

    tracing::info!(
        path = %path.display(),
        debounce_ms = debounce_ms,
        "intelligent_route: overlay file watcher started"
    );

    run_event_loop(
        &mut rx,
        &path,
        &snapshot,
        debounce_ms,
        &shutdown,
        expected_scope.as_ref(),
    )
    .await;
}

/// Process filesystem events until shutdown is requested.
#[expect(
    clippy::cognitive_complexity,
    reason = "complexity is from tokio::select! macro expansion"
)]
#[expect(clippy::too_many_arguments, reason = "watcher loop needs all context")]
async fn run_event_loop(
    rx: &mut mpsc::Receiver<()>,
    path: &Path,
    snapshot: &ArcSwap<RouteSnapshot>,
    debounce_ms: u64,
    shutdown: &CancellationToken,
    expected_scope: Option<&ExpectedOverlayScope>,
) {
    loop {
        tokio::select! {
            Some(()) = rx.recv() => {
                tracing::debug!(debounce_ms = debounce_ms, "intelligent_route: overlay change detected, debouncing");
                if !drain_and_debounce(rx, debounce_ms, shutdown).await {
                    tracing::info!("intelligent_route: overlay file watcher shutting down");
                    return;
                }
                handle_overlay_reload(path, snapshot, expected_scope);
            }
            () = shutdown.cancelled() => {
                tracing::info!("intelligent_route: overlay file watcher shutting down");
                return;
            }
        }
    }
}

/// Read, validate, and swap the overlay snapshot.
fn handle_overlay_reload(
    path: &Path,
    snapshot: &ArcSwap<RouteSnapshot>,
    expected_scope: Option<&ExpectedOverlayScope>,
) {
    let Some(content) = read_overlay(path) else {
        return;
    };

    if is_unchanged(&content, snapshot) {
        return;
    }

    apply_overlay(path, &content, snapshot, expected_scope);
}

/// Read the overlay file with a bounded read.
///
/// Uses [`read_overlay_bounded`] and logs errors.
fn read_overlay(path: &Path) -> Option<Vec<u8>> {
    match read_overlay_bounded(path) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::error!(
                path = %path.display(),
                error = %e,
                "intelligent_route: failed to read overlay file for reload"
            );
            None
        },
    }
}

/// Read at most [`MAX_OVERLAY_SIZE`] bytes from the overlay file.
///
/// Opens the file and reads at most `MAX_OVERLAY_SIZE + 1` bytes.
/// If the extra byte is present, the file exceeds the limit and an
/// error is returned — without ever allocating the full file size.
///
/// # Errors
///
/// Returns [`std::io::Error`] if the file cannot be opened, read,
/// or exceeds [`MAX_OVERLAY_SIZE`].
pub(crate) fn read_overlay_bounded(path: &Path) -> Result<Vec<u8>, std::io::Error> {
    let file = std::fs::File::open(path)?;
    let limit = MAX_OVERLAY_SIZE + 1;
    let mut buf = Vec::new();
    file.take(limit).read_to_end(&mut buf)?;
    if buf.len() as u64 > MAX_OVERLAY_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("overlay file exceeds {MAX_OVERLAY_SIZE} byte limit"),
        ));
    }
    Ok(buf)
}

/// Check whether the content hash matches the current snapshot.
fn is_unchanged(content: &[u8], snapshot: &ArcSwap<RouteSnapshot>) -> bool {
    let new_hash: [u8; 32] = Sha256::digest(content).into();
    let unchanged = new_hash == snapshot.load().content_hash;
    if unchanged {
        tracing::debug!("intelligent_route: overlay content unchanged (hash match)");
    }
    unchanged
}

/// Parse the overlay and swap the snapshot on success.
#[expect(clippy::too_many_lines, reason = "error and success paths with structured logging")]
fn apply_overlay(
    path: &Path,
    content: &[u8],
    snapshot: &ArcSwap<RouteSnapshot>,
    expected_scope: Option<&ExpectedOverlayScope>,
) {
    match RouteSnapshot::from_overlay_with_scope(content, expected_scope) {
        Ok(new_snap) => {
            let previous_serving_revision = snapshot
                .load()
                .semantic_revision
                .as_deref()
                .unwrap_or("none")
                .to_owned();
            let accepted_revision = new_snap.semantic_revision.as_deref().unwrap_or("none").to_owned();
            let schema_version = new_snap.schema_version.as_deref().unwrap_or("none").to_owned();
            let candidate_count = new_snap.candidates.len();
            let local_site = Arc::clone(&new_snap.local_site);
            let contract_format = new_snap.contract_format;
            snapshot.store(Arc::new(new_snap));
            tracing::info!(
                candidate_count,
                local_site = &*local_site,
                contract_format = ?contract_format,
                schema_version = %schema_version,
                accepted_revision = %accepted_revision,
                serving_revision = %accepted_revision,
                previous_serving_revision = %previous_serving_revision,
                "intelligent_route: overlay reloaded"
            );
        },
        Err(e) => {
            let serving_rev = snapshot
                .load()
                .semantic_revision
                .as_deref()
                .unwrap_or("none")
                .to_owned();
            tracing::error!(
                path = %path.display(),
                error = %e,
                retained_serving_revision = %serving_rev,
                "intelligent_route: overlay reload failed, retaining previous snapshot"
            );
        },
    }
}

/// Set up a [`RecommendedWatcher`] that sends to the given channel
/// on relevant filesystem events.
fn setup_watcher(tx: mpsc::Sender<()>, watch_dir: &Path) -> Result<RecommendedWatcher, notify::Error> {
    let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| match res {
        Ok(event) if is_relevant_event(event.kind) && tx.try_send(()).is_err() => {
            tracing::trace!("intelligent_route: overlay watcher channel full, event coalesced by debounce");
        },
        Err(e) => {
            tracing::warn!(error = %e, "intelligent_route: overlay file watcher error");
        },
        _ => {},
    })?;

    watcher.watch(watch_dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

/// Cancellation-aware debounce: sleep for the debounce window, then
/// drain any queued events.
///
/// Returns `true` to proceed with reload, `false` if shutdown was
/// requested during the debounce.
async fn drain_and_debounce(rx: &mut mpsc::Receiver<()>, debounce_ms: u64, shutdown: &CancellationToken) -> bool {
    tokio::select! {
        () = tokio::time::sleep(Duration::from_millis(debounce_ms)) => {
            while rx.try_recv().is_ok() {}
            true
        }
        () = shutdown.cancelled() => false
    }
}

/// Whether a notify event kind is relevant for overlay reload.
///
/// Accepts Create, Modify (including rename/Name events), and Remove.
/// No path filtering is applied — any relevant event on the watched
/// parent directory triggers a re-read.  Hash comparison handles
/// false positives.
fn is_relevant_event(kind: EventKind) -> bool {
    matches!(kind, EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_))
}

/// Resolve the directory to watch for a given overlay path.
///
/// Falls back to `.` when the path has no non-empty parent.
fn watch_dir_for_path(path: &Path) -> PathBuf {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::disallowed_methods,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Overlay parsing
    // -------------------------------------------------------------------------

    #[test]
    fn parse_minimal_overlay() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3",
                "site": "site-a",
                "cluster": "local-inference",
                "fresh": true
            }]
        }"#;
        let doc: OverlayDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.local_site, "site-a");
        assert_eq!(doc.candidates.len(), 1);
        assert_eq!(doc.candidates[0].kind, "inference_model");
        assert_eq!(doc.candidates[0].name, "llama-3");
        assert!(doc.candidates[0].fresh);
        assert!(doc.candidates[0].credential.is_none());
    }

    #[test]
    fn parse_overlay_with_credential_camel_case() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "gpt-4",
                "site": "site-b",
                "cluster": "api-provider",
                "fresh": true,
                "credential": {
                    "strategy": "bearer_token",
                    "secretRef": {
                        "name": "provider-token",
                        "namespace": "grid-system",
                        "key": "token"
                    }
                }
            }]
        }"#;
        let doc: OverlayDocument = serde_json::from_str(json).unwrap();
        assert!(doc.candidates[0].credential.is_some());
    }

    #[test]
    fn parse_overlay_with_credential_snake_case_alias() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "gpt-4",
                "site": "site-b",
                "cluster": "api-provider",
                "fresh": true,
                "credential": {
                    "strategy": "bearer_token",
                    "secret_ref": {
                        "name": "openai-key",
                        "namespace": "grid-system",
                        "key": "token"
                    }
                }
            }]
        }"#;
        let doc: OverlayDocument = serde_json::from_str(json).unwrap();
        assert!(doc.candidates[0].credential.is_some());
    }

    #[test]
    fn parse_overlay_with_network_field() {
        let json = r#"{
            "network": "production",
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3",
                "site": "site-a",
                "cluster": "local",
                "fresh": true
            }]
        }"#;
        let doc: OverlayDocument = serde_json::from_str(json).unwrap();
        assert_eq!(doc.local_site, "site-a");
    }

    #[test]
    fn parse_overlay_unknown_kind_rejected() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "unknown_type",
                "name": "foo",
                "site": "site-a",
                "cluster": "local",
                "fresh": true
            }]
        }"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "unknown kind should be rejected");
    }

    #[test]
    fn parse_overlay_empty_candidates() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": []
        }"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "empty candidates should be rejected");
    }

    #[test]
    fn parse_overlay_missing_required_field() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3",
                "fresh": true
            }]
        }"#;
        let result: Result<OverlayDocument, _> = serde_json::from_str(json);
        assert!(result.is_err(), "missing site/cluster should fail");
    }

    // -------------------------------------------------------------------------
    // Credential safety (deny_unknown_fields)
    // -------------------------------------------------------------------------

    #[test]
    fn credential_rejects_token_field() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "gpt-4",
                "site": "site-b",
                "cluster": "api",
                "fresh": true,
                "credential": {
                    "strategy": "bearer_token",
                    "token": "sk-1234567890",
                    "secretRef": { "name": "k", "namespace": "ns", "key": "t" }
                }
            }]
        }"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "token field in credential must be rejected");
    }

    #[test]
    fn secret_ref_rejects_value_field() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "gpt-4",
                "site": "site-b",
                "cluster": "api",
                "fresh": true,
                "credential": {
                    "strategy": "bearer_token",
                    "secretRef": {
                        "name": "k",
                        "namespace": "ns",
                        "key": "t",
                        "value": "sk-secret-value"
                    }
                }
            }]
        }"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "value field in secret_ref must be rejected");
    }

    #[test]
    fn snapshot_from_produced_overlay() {
        let json = r#"{
            "local_site": "us-east-1",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3-70b",
                "site": "us-east-1",
                "cluster": "gpu-pool-a",
                "fresh": true,
                "stable_id": "cand-abc123",
                "admission_state": "new_and_existing",
                "selection_tier": "same_site",
                "rank": 0,
                "generated_at": "2026-07-24T12:00:00Z",
                "credential": {
                    "strategy": "bearer_token",
                    "secretRef": {
                        "name": "provider-token",
                        "namespace": "grid-system",
                        "key": "token"
                    }
                }
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.candidates.len(), 1);
        assert_eq!(&*snap.candidates[0].name, "llama-3-70b");
        assert_eq!(&*snap.local_site, "us-east-1");
    }

    // -------------------------------------------------------------------------
    // Overlay metadata enrichment
    // -------------------------------------------------------------------------

    #[test]
    fn parse_overlay_with_all_metadata_fields() {
        let json = r#"{
            "local_site": "site-a",
            "generated_at": "2026-07-24T10:00:00Z",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3",
                "site": "site-a",
                "cluster": "gpu-pool",
                "fresh": true,
                "stable_id": "inf/llama-3/site-a/gpu-pool",
                "admission_state": "new_and_existing",
                "selection_tier": "same_site",
                "rank": 0
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        let c = &snap.candidates[0];
        assert_eq!(&*c.stable_id, "inf/llama-3/site-a/gpu-pool");
        assert_eq!(c.admission_state, AdmissionState::NewAndExisting);
        assert_eq!(c.selection_tier.as_deref(), Some("same_site"));
        assert_eq!(c.rank, Some(0));
    }

    #[test]
    fn parse_overlay_defaults_missing_admission_state() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "m",
                "site": "s",
                "cluster": "c",
                "fresh": true
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.candidates[0].admission_state, AdmissionState::NewAndExisting);
    }

    #[test]
    fn parse_overlay_admission_state_none_is_excluded() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "m",
                "site": "s",
                "cluster": "c",
                "fresh": true,
                "admission_state": "none"
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.candidates[0].admission_state, AdmissionState::Excluded);
    }

    #[test]
    fn parse_overlay_unknown_admission_state_rejected() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "m",
                "site": "s",
                "cluster": "c",
                "fresh": true,
                "admission_state": "future_state_v2"
            }]
        }"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "unknown admission state must be rejected");
    }

    #[test]
    fn parse_overlay_stable_id_fallback() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama",
                "site": "site-a",
                "cluster": "gpu",
                "fresh": true
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(
            &*snap.candidates[0].stable_id, "inference_model/llama/site-a/gpu",
            "absent stable_id should use deterministic default"
        );
    }

    #[test]
    fn parse_overlay_existing_only_admission() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "m",
                "site": "s",
                "cluster": "c",
                "fresh": true,
                "admission_state": "existing_only"
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.candidates[0].admission_state, AdmissionState::ExistingOnly);
    }

    #[test]
    fn parse_overlay_unknown_fields_still_accepted() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "m",
                "site": "s",
                "cluster": "c",
                "fresh": true,
                "future_field": "anything",
                "another_future": 42
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(snap.is_ok(), "unknown fields on candidates must still be accepted");
    }

    #[test]
    fn parse_overlay_oversized_stable_id_rejected() {
        let long_id = "x".repeat(257);
        let json = format!(
            r#"{{"local_site":"site-a","candidates":[{{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"stable_id":"{long_id}"}}]}}"#
        );
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "stable_id exceeding 256 bytes must be rejected");
    }

    #[test]
    fn parse_overlay_empty_stable_id_rejected() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"stable_id":""}]}"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "empty stable_id must be rejected");
    }

    #[test]
    fn parse_overlay_whitespace_stable_id_rejected() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"stable_id":"   "}]}"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "whitespace stable_id must be rejected");
    }

    #[test]
    fn parse_overlay_oversized_selection_tier_rejected() {
        let long_tier = "t".repeat(129);
        let json = format!(
            r#"{{"local_site":"site-a","candidates":[{{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"selection_tier":"{long_tier}"}}]}}"#
        );
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "selection_tier exceeding 128 bytes must be rejected");
    }

    #[test]
    fn parse_overlay_empty_selection_tier_rejected() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"selection_tier":""}]}"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "empty selection_tier must be rejected");
    }

    // -------------------------------------------------------------------------
    // RouteSnapshot
    // -------------------------------------------------------------------------

    #[test]
    fn snapshot_from_overlay_valid() {
        let json = r#"{
            "local_site": "site-a",
            "candidates": [{
                "kind": "inference_model",
                "name": "llama-3",
                "site": "site-a",
                "cluster": "local-inference",
                "fresh": true
            }]
        }"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.candidates.len(), 1);
        assert_eq!(&*snap.local_site, "site-a");
        assert_ne!(snap.content_hash, [0; 32]);
    }

    #[test]
    fn snapshot_from_overlay_invalid_json() {
        let result = RouteSnapshot::from_overlay(b"not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_content_hash_deterministic() {
        let json = br#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        let s1 = RouteSnapshot::from_overlay(json).unwrap();
        let s2 = RouteSnapshot::from_overlay(json).unwrap();
        assert_eq!(s1.content_hash, s2.content_hash);
    }

    #[test]
    fn snapshot_content_hash_differs() {
        let json_a = br#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        let json_b = br#"{"local_site":"b","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        let s1 = RouteSnapshot::from_overlay(json_a).unwrap();
        let s2 = RouteSnapshot::from_overlay(json_b).unwrap();
        assert_ne!(s1.content_hash, s2.content_hash);
    }

    #[test]
    fn snapshot_from_static_zero_hash() {
        let snap = RouteSnapshot::from_static(vec![], Arc::from("site-a"));
        assert_eq!(snap.content_hash, [0; 32]);
    }

    // -------------------------------------------------------------------------
    // Bounded read
    // -------------------------------------------------------------------------

    #[test]
    fn read_bounded_rejects_oversized_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.json");
        let content = vec![b'x'; (MAX_OVERLAY_SIZE + 1) as usize];
        std::fs::write(&path, &content).unwrap();
        let result = read_overlay_bounded(&path);
        assert!(result.is_err(), "oversized file must be rejected");
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn read_bounded_accepts_within_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.json");
        let json = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json).unwrap();
        let result = read_overlay_bounded(&path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), json.as_bytes());
    }

    #[test]
    fn read_bounded_missing_file() {
        let result = read_overlay_bounded(Path::new("/nonexistent/overlay.json"));
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Last-known-good retention (handle_overlay_reload)
    // -------------------------------------------------------------------------

    fn make_valid_snapshot() -> (Arc<ArcSwap<RouteSnapshot>>, [u8; 32]) {
        let json = make_overlay_json("site-a", "llama-3", "local");
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        let hash = snap.content_hash;
        (Arc::new(ArcSwap::from_pointee(snap)), hash)
    }

    #[test]
    fn retain_on_read_failure() {
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(Path::new("/nonexistent/overlay.json"), &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(&path, b"").unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(&path, "{{not json}}").unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_blank_local_site() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = r#"{"local_site":"","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        std::fs::write(&path, json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_empty_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        std::fs::write(&path, r#"{"local_site":"a","candidates":[]}"#).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_unknown_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json =
            r#"{"local_site":"a","candidates":[{"kind":"bad_kind","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        std::fs::write(&path, json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_oversized_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let content = vec![b'x'; (MAX_OVERLAY_SIZE + 1) as usize];
        std::fs::write(&path, &content).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_invalid_credential_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"credential":{"strategy":"bearer_token","token":"leaked","secretRef":{"name":"k","namespace":"n","key":"k"}}}]}"#;
        std::fs::write(&path, json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash);
    }

    #[test]
    fn retain_on_unknown_admission_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"admission_state":"future_state"}]}"#;
        std::fs::write(&path, json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(
            snap.load().content_hash,
            hash,
            "unknown admission_state must retain LKG"
        );
    }

    #[test]
    fn retain_on_invalid_stable_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let long_id = "x".repeat(257);
        let json = format!(
            r#"{{"local_site":"a","candidates":[{{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true,"stable_id":"{long_id}"}}]}}"#
        );
        std::fs::write(&path, &json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash, "oversized stable_id must retain LKG");
    }

    #[test]
    fn retain_on_legacy_downgrade_with_scope() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = r#"{"local_site":"a","candidates":[{"kind":"inference_model","name":"m","site":"s","cluster":"c","fresh":true}]}"#;
        std::fs::write(&path, json).unwrap();
        let (snap, hash) = make_valid_snapshot();
        let scope = ExpectedOverlayScope {
            network: Some("net".to_owned()),
            gateway: None,
            namespace: None,
            local_site: None,
        };
        handle_overlay_reload(&path, &snap, Some(&scope));
        assert_eq!(
            snap.load().content_hash,
            hash,
            "legacy payload with expected_scope must retain LKG"
        );
    }

    // -------------------------------------------------------------------------
    // Envelope parsing
    // -------------------------------------------------------------------------

    #[expect(
        clippy::too_many_lines,
        reason = "test helper constructs full envelope with computed digest"
    )]
    fn make_envelope_json(local_site: &str, model: &str, cluster: &str, network: &str) -> String {
        let overlay_obj = serde_json::json!({
            "local_site": local_site,
            "network": network,
            "candidates": [{
                "kind": "inference_model",
                "name": model,
                "site": local_site,
                "cluster": cluster,
                "fresh": true
            }]
        });
        let semantic = serde_json::json!({
            "candidates": overlay_obj["candidates"],
            "local_site": overlay_obj["local_site"],
            "network": overlay_obj["network"],
        });
        let canonical = serde_json_canonicalizer::to_vec(&semantic).unwrap();
        let digest: [u8; 32] = Sha256::digest(&canonical).into();
        let mut hex = String::with_capacity(64);
        for b in &digest {
            let _unused = std::fmt::Write::write_fmt(&mut hex, format_args!("{b:02x}"));
        }
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": "1.0.0",
            "revision": {
                "kind": "content_addressed",
                "algorithm": "sha256",
                "value": hex
            },
            "content_digest": {
                "algorithm": "sha256",
                "value": hex
            },
            "scope": {
                "network": network,
                "gateway": "gw",
                "namespace": "ns",
                "local_site": local_site
            },
            "provenance": {
                "producer": "test",
                "producer_version": "0.1.0",
                "source_name": network,
                "source_uid": "test-uid",
                "source_generation": 1,
                "rendered_at": "2026-07-29T00:00:00Z"
            },
            "overlay": overlay_obj
        }))
        .unwrap()
    }

    #[test]
    fn envelope_accepted_with_valid_digest() {
        let json = make_envelope_json("site-a", "llama-3", "local", "test-net");
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
        assert_eq!(snap.schema_version.as_deref(), Some("1.0.0"));
        assert!(snap.semantic_revision.is_some());
        assert_eq!(snap.candidates.len(), 1);
        assert_eq!(&*snap.local_site, "site-a");
    }

    #[test]
    fn legacy_accepted_without_schema_version() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"site-a","cluster":"c","fresh":true}]}"#;
        let snap = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Legacy);
        assert!(snap.semantic_revision.is_none());
        assert!(snap.schema_version.is_none());
    }

    #[test]
    fn unsupported_schema_version_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["schema_version"] = serde_json::json!("99.0.0");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn envelope_no_legacy_fallback_on_malformed() {
        let json = r#"{"schema_version":"1.0.0","overlay":{"local_site":"a","candidates":[]}}"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "malformed envelope must not fall back to legacy");
    }

    #[test]
    fn any_reserved_field_triggers_envelope_path() {
        for field in &["revision", "content_digest", "scope", "provenance"] {
            let json = format!(r#"{{"local_site":"a","candidates":[],"{field}":"present"}}"#,);
            let result = RouteSnapshot::from_overlay(json.as_bytes());
            assert!(
                result.is_err(),
                "reserved field {field} must trigger envelope path and fail, not parse as legacy"
            );
        }
    }

    #[test]
    fn hybrid_missing_schema_no_downgrade() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("schema_version");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(
            result.is_err(),
            "envelope with schema_version removed but other envelope fields present must not downgrade to legacy"
        );
    }

    #[test]
    fn digest_mismatch_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        let bad_digest = "0".repeat(64);
        json["content_digest"]["value"] = serde_json::json!(bad_digest);
        json["revision"]["value"] = serde_json::json!(bad_digest);
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("digest mismatch"));
    }

    #[test]
    fn revision_digest_disagreement_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["revision"]["value"] = serde_json::json!("1".repeat(64));
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disagree"));
    }

    #[test]
    fn scope_network_mismatch_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["scope"]["network"] = serde_json::json!("wrong-net");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scope.network"));
    }

    #[test]
    fn scope_local_site_mismatch_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["scope"]["local_site"] = serde_json::json!("wrong-site");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("scope.local_site"));
    }

    #[test]
    fn expected_scope_validation_passes() {
        let json = make_envelope_json("site-a", "m", "c", "test-net");
        let scope = ExpectedOverlayScope {
            network: Some("test-net".to_owned()),
            gateway: Some("gw".to_owned()),
            namespace: Some("ns".to_owned()),
            local_site: Some("site-a".to_owned()),
        };
        let snap = RouteSnapshot::from_overlay_with_scope(json.as_bytes(), Some(&scope)).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
    }

    #[test]
    fn expected_scope_network_mismatch_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "test-net");
        let scope = ExpectedOverlayScope {
            network: Some("wrong-net".to_owned()),
            gateway: None,
            namespace: None,
            local_site: None,
        };
        let result = RouteSnapshot::from_overlay_with_scope(json.as_bytes(), Some(&scope));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected scope.network"));
    }

    #[test]
    fn legacy_rejected_when_expected_scope_configured() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"site-a","cluster":"c","fresh":true}]}"#;
        let scope = ExpectedOverlayScope {
            network: Some("strict-net".to_owned()),
            gateway: None,
            namespace: None,
            local_site: None,
        };
        let result = RouteSnapshot::from_overlay_with_scope(json.as_bytes(), Some(&scope));
        assert!(
            result.is_err(),
            "legacy payload must be rejected when expected_overlay_scope is configured"
        );
    }

    #[test]
    fn legacy_accepted_without_expected_scope() {
        let json = r#"{"local_site":"site-a","candidates":[{"kind":"inference_model","name":"m","site":"site-a","cluster":"c","fresh":true}]}"#;
        let snap = RouteSnapshot::from_overlay_with_scope(json.as_bytes(), None).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Legacy);
    }

    #[test]
    fn missing_provenance_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value.as_object_mut().unwrap().remove("provenance");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "envelope without provenance must be rejected");
    }

    #[test]
    fn incomplete_provenance_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["provenance"].as_object_mut().unwrap().remove("producer_version");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(
            result.is_err(),
            "envelope with incomplete v1 provenance must be rejected"
        );
    }

    #[test]
    fn blank_provenance_field_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["provenance"]["source_uid"] = serde_json::json!("   ");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "blank v1 provenance values must be rejected");
    }

    #[test]
    fn zero_source_generation_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["provenance"]["source_generation"] = serde_json::json!(0);
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "zero source generation must be rejected");
    }

    #[test]
    fn invalid_rendered_at_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["provenance"]["rendered_at"] = serde_json::json!("not-a-timestamp");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "rendered_at must be valid RFC 3339");
    }

    #[test]
    fn blank_scope_field_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["scope"]["gateway"] = serde_json::json!(" ");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "blank scope values must be rejected");
    }

    #[test]
    fn missing_overlay_network_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["overlay"].as_object_mut().unwrap().remove("network");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(
            result.is_err(),
            "envelope with missing overlay.network must be rejected (digest or required check)"
        );
    }

    #[test]
    fn provenance_scope_network_mismatch_rejected() {
        let json = make_envelope_json("site-a", "m", "c", "n");
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["provenance"]["source_name"] = serde_json::json!("wrong-net");
        let result = RouteSnapshot::from_overlay(serde_json::to_string(&value).unwrap().as_bytes());
        assert!(result.is_err(), "provenance/scope network mismatch must be rejected");
        assert!(
            result.unwrap_err().to_string().contains("provenance.source_name"),
            "error should mention provenance mismatch"
        );
    }

    #[test]
    fn envelope_unknown_fields_accepted() {
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["future_field"] = serde_json::json!("forward_compat");
        let snap = RouteSnapshot::from_overlay(serde_json::to_string(&json).unwrap().as_bytes()).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
    }

    #[test]
    fn timestamp_change_same_revision() {
        let json_a = make_envelope_json("site-a", "m", "c", "n");
        let snap_a = RouteSnapshot::from_overlay(json_a.as_bytes()).unwrap();

        let mut json_b: serde_json::Value = serde_json::from_str(&json_a).unwrap();
        json_b["overlay"]["generated_at"] = serde_json::json!("2099-12-31T23:59:59Z");
        json_b["provenance"]["rendered_at"] = serde_json::json!("2099-12-31T23:59:59Z");
        let snap_b = RouteSnapshot::from_overlay(serde_json::to_string(&json_b).unwrap().as_bytes()).unwrap();

        assert_eq!(
            snap_a.semantic_revision.as_deref(),
            snap_b.semantic_revision.as_deref(),
            "timestamp changes must not affect semantic revision"
        );
    }

    // -------------------------------------------------------------------------
    // Envelope LKG retention
    // -------------------------------------------------------------------------

    #[test]
    fn retain_on_envelope_version_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.json");
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        json["schema_version"] = serde_json::json!("99.0.0");
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash, "LKG retained on schema version error");
    }

    #[test]
    fn retain_on_envelope_digest_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.json");
        let mut json: serde_json::Value = serde_json::from_str(&make_envelope_json("site-a", "m", "c", "n")).unwrap();
        let bad = "0".repeat(64);
        json["content_digest"]["value"] = serde_json::json!(bad);
        json["revision"]["value"] = serde_json::json!(bad);
        std::fs::write(&path, serde_json::to_string(&json).unwrap()).unwrap();
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, None);
        assert_eq!(snap.load().content_hash, hash, "LKG retained on digest error");
    }

    #[test]
    fn retain_on_envelope_scope_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.json");
        std::fs::write(&path, make_envelope_json("site-a", "m", "c", "n")).unwrap();
        let scope = ExpectedOverlayScope {
            network: Some("wrong".to_owned()),
            gateway: None,
            namespace: None,
            local_site: None,
        };
        let (snap, hash) = make_valid_snapshot();
        handle_overlay_reload(&path, &snap, Some(&scope));
        assert_eq!(snap.load().content_hash, hash, "LKG retained on scope mismatch");
    }

    // -------------------------------------------------------------------------
    // Fixture-based tests (overlay contract fixtures)
    // -------------------------------------------------------------------------

    #[test]
    fn overlay_fixture_valid_minimal() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/valid-minimal.json");
        let snap = RouteSnapshot::from_overlay(fixture).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
        assert_eq!(
            snap.semantic_revision.as_deref(),
            Some("abd5f4855454390febb53ad2085d182c465a78c5d14f0132317b4c9335aa8494"),
            "revision must match manifest"
        );
    }

    #[test]
    fn overlay_fixture_valid_multi_candidate() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/valid-multi-candidate.json");
        let snap = RouteSnapshot::from_overlay(fixture).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
        assert!(
            snap.candidates.len() >= 2,
            "multi-candidate fixture must have 2+ candidates"
        );
        assert_eq!(
            snap.semantic_revision.as_deref(),
            Some("75b057d750d9db77030ecd5a073c235c56b2b0460d3d517340b3e44020e83056"),
            "revision must match manifest"
        );
    }

    #[test]
    fn overlay_fixture_legacy_payload() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/legacy-payload.json");
        let snap = RouteSnapshot::from_overlay(fixture).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Legacy);
        assert!(snap.semantic_revision.is_none());
    }

    #[test]
    fn overlay_fixture_unsupported_schema_version() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/unsupported-schema-version.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "unsupported schema version must be rejected");
    }

    #[test]
    fn overlay_fixture_invalid_digest() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/invalid-digest.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "invalid digest must be rejected");
    }

    #[test]
    fn overlay_fixture_revision_digest_disagreement() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/revision-digest-disagreement.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "revision/digest disagreement must be rejected");
    }

    #[test]
    fn overlay_fixture_timestamp_change_same_revision() {
        let valid_fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/valid-multi-candidate.json");
        let ts_fixture =
            include_bytes!("../../../tests/fixtures/overlay-contract/v1/timestamp-change-same-revision.json");
        let snap_valid = RouteSnapshot::from_overlay(valid_fixture).unwrap();
        let snap_ts = RouteSnapshot::from_overlay(ts_fixture).unwrap();
        assert_eq!(
            snap_valid.semantic_revision.as_deref(),
            snap_ts.semantic_revision.as_deref(),
            "timestamp change must not affect semantic revision"
        );
        assert_eq!(
            snap_ts.semantic_revision.as_deref(),
            Some("75b057d750d9db77030ecd5a073c235c56b2b0460d3d517340b3e44020e83056"),
            "revision must match manifest"
        );
    }

    #[test]
    fn overlay_fixture_unknown_additive_field() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/unknown-additive-field.json");
        let snap = RouteSnapshot::from_overlay(fixture).unwrap();
        assert_eq!(snap.contract_format, ContractFormat::Envelope);
        assert_eq!(
            snap.semantic_revision.as_deref(),
            Some("75b057d750d9db77030ecd5a073c235c56b2b0460d3d517340b3e44020e83056"),
            "additive fields must not change revision"
        );
    }

    #[test]
    fn overlay_fixture_network_scope_mismatch() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/network-scope-mismatch.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "network scope mismatch must be rejected");
    }

    #[test]
    fn overlay_fixture_local_site_scope_mismatch() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/local-site-scope-mismatch.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "local-site scope mismatch must be rejected");
    }

    #[test]
    fn overlay_fixture_malformed_envelope() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/malformed-envelope.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(result.is_err(), "malformed envelope must be rejected");
    }

    #[test]
    fn overlay_fixture_hybrid_missing_schema() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/hybrid-missing-schema.json");
        let result = RouteSnapshot::from_overlay(fixture);
        assert!(
            result.is_err(),
            "hybrid document with envelope fields but missing schema_version must be rejected"
        );
    }

    #[test]
    fn overlay_fixture_credential_ref_no_secret_value() {
        let fixture = include_bytes!("../../../tests/fixtures/overlay-contract/v1/valid-multi-candidate.json");
        let content = std::str::from_utf8(fixture).unwrap().to_lowercase();
        assert!(!content.contains("bearer "), "fixture must not contain bearer token");
        assert!(!content.contains("token_value"), "fixture must not contain token value");
        assert!(!content.contains("-----begin"), "fixture must not contain PEM key");
        assert!(content.contains("secretref"), "fixture must use credential references");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one manifest-driven test checks every cross-repository fixture"
    )]
    fn overlay_fixture_manifest_complete() {
        let manifest_bytes = include_bytes!("../../../tests/fixtures/overlay-contract/v1/manifest.json");
        let manifest: serde_json::Value = serde_json::from_slice(manifest_bytes).unwrap();
        assert_eq!(manifest["contract_version"], "1.0.0");
        assert_eq!(manifest["source"], "praxis-proxy/ai");
        assert_eq!(manifest["source_path"], "tests/fixtures/overlay-contract/v1");
        let fixtures = manifest["fixtures"].as_object().unwrap();
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests/fixtures/overlay-contract/v1");
        for name in fixtures.keys() {
            assert!(
                fixture_dir.join(name).exists(),
                "manifest references {name} but file is missing"
            );
        }
        for (name, expectation) in fixtures {
            let content = std::fs::read(fixture_dir.join(name)).unwrap();
            let result = RouteSnapshot::from_overlay(&content);
            match expectation["expected"].as_str() {
                Some("accept") => {
                    let snapshot = result.unwrap_or_else(|error| {
                        panic!("fixture {name} should be accepted: {error}");
                    });
                    assert_eq!(snapshot.contract_format, ContractFormat::Envelope);
                    assert_eq!(
                        snapshot.semantic_revision.as_deref(),
                        expectation["revision"].as_str(),
                        "fixture {name} revision differs from the manifest"
                    );
                },
                Some("accept_legacy") => {
                    let snapshot = result.unwrap_or_else(|error| {
                        panic!("legacy fixture {name} should be accepted: {error}");
                    });
                    assert_eq!(snapshot.contract_format, ContractFormat::Legacy);
                },
                Some("reject") => {
                    assert!(result.is_err(), "fixture {name} should be rejected");
                    assert!(
                        expectation["reason"].as_str().is_some_and(|reason| !reason.is_empty()),
                        "rejected fixture {name} must declare a reason"
                    );
                },
                other => panic!("fixture {name} has unsupported expected outcome: {other:?}"),
            }
        }
        for entry in std::fs::read_dir(&fixture_dir).unwrap() {
            let path = entry.unwrap().path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.ends_with(".json")
                && name != "manifest.json"
            {
                assert!(
                    fixtures.contains_key(name),
                    "fixture file {name} exists but is not listed in manifest"
                );
            }
        }
    }

    #[test]
    fn unknown_candidate_kind_rejected() {
        let json = r#"{"local_site":"a","candidates":[{"kind":"unknown_kind","name":"m","site":"a","cluster":"c","fresh":true}]}"#;
        let result = RouteSnapshot::from_overlay(json.as_bytes());
        assert!(result.is_err(), "unknown candidate kind must be rejected");
    }

    #[test]
    fn max_overlay_size_is_bounded() {
        const { assert!(MAX_OVERLAY_SIZE <= 2 * 1024 * 1024) };
        const { assert!(MAX_OVERLAY_SIZE > 0) };
    }

    // -------------------------------------------------------------------------
    // Watcher lifecycle
    // -------------------------------------------------------------------------

    /// Test-only startup wait for the background notify watcher.
    const WATCHER_STARTUP_MS: u64 = 750;

    fn make_overlay_json(local_site: &str, model: &str, cluster: &str) -> String {
        format!(
            r#"{{"local_site":"{local_site}","candidates":[{{"kind":"inference_model","name":"{model}","site":"{local_site}","cluster":"{cluster}","fresh":true}}]}}"#,
        )
    }

    #[test]
    fn watcher_starts_and_stops() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json).unwrap();

        let snap = Arc::new(ArcSwap::from_pointee(
            RouteSnapshot::from_overlay(json.as_bytes()).unwrap(),
        ));
        let handle = spawn_overlay_watcher(path, snap, DEFAULT_DEBOUNCE_MS).unwrap();

        std::thread::sleep(Duration::from_millis(100));
        assert!(!handle.shutdown.is_cancelled(), "shutdown should not be cancelled yet");
        let start = std::time::Instant::now();
        drop(handle);
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "Drop should complete within bounded join timeout"
        );
    }

    #[test]
    fn watcher_initialization_failure_is_returned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing").join("routing-config.json");
        let json = make_overlay_json("site-a", "llama-3", "local");
        let snap = Arc::new(ArcSwap::from_pointee(
            RouteSnapshot::from_overlay(json.as_bytes()).unwrap(),
        ));

        let error = spawn_overlay_watcher(path, snap, DEFAULT_DEBOUNCE_MS)
            .expect_err("missing watcher directory must fail filter initialization");

        assert!(
            error.to_string().contains("failed to initialize"),
            "unexpected watcher initialization error: {error}"
        );
    }

    #[test]
    fn watcher_no_thread_accumulation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json).unwrap();

        for i in 0..10 {
            let snap = Arc::new(ArcSwap::from_pointee(
                RouteSnapshot::from_overlay(json.as_bytes()).unwrap(),
            ));
            let handle = spawn_overlay_watcher(path.clone(), Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            let start = std::time::Instant::now();
            drop(handle);
            assert!(
                start.elapsed() < Duration::from_secs(3),
                "watcher {i} Drop should complete within bounded join timeout"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Startup race
    // -------------------------------------------------------------------------

    #[test]
    fn watcher_catches_change_before_registration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");

        let json_v1 = make_overlay_json("site-a", "llama-3", "cluster-v1");
        std::fs::write(&path, &json_v1).unwrap();

        let initial = RouteSnapshot::from_overlay(json_v1.as_bytes()).unwrap();
        let snap = Arc::new(ArcSwap::from_pointee(initial));

        let json_v2 = make_overlay_json("site-a", "gpt-4", "cluster-v2");
        std::fs::write(&path, &json_v2).unwrap();

        let _handle = spawn_overlay_watcher(path, Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();
        let v2_hash = RouteSnapshot::from_overlay(json_v2.as_bytes()).unwrap().content_hash;

        poll_until(Duration::from_secs(5), || snap.load().content_hash == v2_hash);

        assert_eq!(
            &*snap.load().candidates[0].name,
            "gpt-4",
            "watcher should catch the change that happened before registration"
        );
    }

    // -------------------------------------------------------------------------
    // Watcher reload
    // -------------------------------------------------------------------------

    #[test]
    fn watcher_detects_file_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json_v1 = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json_v1).unwrap();

        let initial = RouteSnapshot::from_overlay(json_v1.as_bytes()).unwrap();
        let old_hash = initial.content_hash;
        let snap = Arc::new(ArcSwap::from_pointee(initial));
        let _handle = spawn_overlay_watcher(path.clone(), Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();

        std::thread::sleep(Duration::from_millis(WATCHER_STARTUP_MS));

        let json_v2 = make_overlay_json("site-a", "gpt-4", "api-provider");
        std::fs::write(&path, json_v2).unwrap();

        poll_until(Duration::from_secs(5), || snap.load().content_hash != old_hash);

        let loaded = snap.load();
        assert_ne!(
            loaded.content_hash, old_hash,
            "snapshot should be swapped after file change"
        );
        assert_eq!(loaded.candidates.len(), 1);
        assert_eq!(&*loaded.candidates[0].name, "gpt-4");
    }

    #[test]
    fn watcher_skips_unchanged_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json).unwrap();

        let initial = RouteSnapshot::from_overlay(json.as_bytes()).unwrap();
        let snap = Arc::new(ArcSwap::from_pointee(initial));
        let _handle = spawn_overlay_watcher(path.clone(), Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();

        std::thread::sleep(Duration::from_millis(WATCHER_STARTUP_MS));

        let ptr_before = Arc::as_ptr(&snap.load());
        std::fs::write(&path, &json).unwrap();
        std::thread::sleep(Duration::from_millis(DEFAULT_DEBOUNCE_MS + 300));

        let ptr_after = Arc::as_ptr(&snap.load());
        assert_eq!(
            ptr_before, ptr_after,
            "snapshot pointer should be unchanged when content is identical"
        );
    }

    #[test]
    fn watcher_survives_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing-config.json");
        let json_v1 = make_overlay_json("site-a", "llama-3", "local");
        std::fs::write(&path, &json_v1).unwrap();

        let initial = RouteSnapshot::from_overlay(json_v1.as_bytes()).unwrap();
        let old_hash = initial.content_hash;
        let snap = Arc::new(ArcSwap::from_pointee(initial));
        let _handle = spawn_overlay_watcher(path.clone(), Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();

        std::thread::sleep(Duration::from_millis(WATCHER_STARTUP_MS));

        std::fs::write(&path, "invalid json {{{{").unwrap();
        std::thread::sleep(Duration::from_millis(DEFAULT_DEBOUNCE_MS + 300));

        assert_eq!(
            snap.load().content_hash,
            old_hash,
            "snapshot should be retained after invalid JSON"
        );

        let json_v2 = make_overlay_json("site-a", "gpt-4", "api-provider");
        std::fs::write(&path, json_v2).unwrap();

        poll_until(Duration::from_secs(5), || snap.load().content_hash != old_hash);

        let loaded = snap.load();
        assert_ne!(
            loaded.content_hash, old_hash,
            "snapshot should recover after valid JSON"
        );
        assert_eq!(&*loaded.candidates[0].name, "gpt-4");
    }

    // -------------------------------------------------------------------------
    // Symlink swap (Kubernetes AtomicWriter pattern)
    // -------------------------------------------------------------------------

    #[cfg(unix)]
    #[test]
    fn watcher_detects_symlink_swap() {
        let dir = tempfile::tempdir().unwrap();
        let data_v1 = dir.path().join("data_v1");
        let data_v2 = dir.path().join("data_v2");
        std::fs::create_dir_all(&data_v1).unwrap();
        std::fs::create_dir_all(&data_v2).unwrap();

        let json_v1 = make_overlay_json("site-a", "llama-3", "local");
        let json_v2 = make_overlay_json("site-a", "gpt-4", "api-provider");
        std::fs::write(data_v1.join("config.json"), &json_v1).unwrap();
        std::fs::write(data_v2.join("config.json"), &json_v2).unwrap();

        let data_link = dir.path().join("..data");
        std::os::unix::fs::symlink(&data_v1, &data_link).unwrap();
        let overlay_path = dir.path().join("routing-config.json");
        std::os::unix::fs::symlink("..data/config.json", &overlay_path).unwrap();

        let initial = RouteSnapshot::from_overlay(json_v1.as_bytes()).unwrap();
        let old_hash = initial.content_hash;
        let snap = Arc::new(ArcSwap::from_pointee(initial));
        let _handle = spawn_overlay_watcher(overlay_path, Arc::clone(&snap), DEFAULT_DEBOUNCE_MS).unwrap();

        std::thread::sleep(Duration::from_millis(WATCHER_STARTUP_MS));

        let tmp_link = dir.path().join("..data_tmp");
        std::os::unix::fs::symlink(&data_v2, &tmp_link).unwrap();
        std::fs::rename(&tmp_link, &data_link).unwrap();

        poll_until(Duration::from_secs(5), || snap.load().content_hash != old_hash);
        assert_eq!(&*snap.load().candidates[0].name, "gpt-4");
    }

    // -------------------------------------------------------------------------
    // Event kind helpers
    // -------------------------------------------------------------------------

    #[test]
    fn is_relevant_event_create() {
        assert!(
            is_relevant_event(EventKind::Create(notify::event::CreateKind::File)),
            "Create events should be relevant"
        );
    }

    #[test]
    fn is_relevant_event_modify_name() {
        assert!(
            is_relevant_event(EventKind::Modify(notify::event::ModifyKind::Name(
                notify::event::RenameMode::Any
            ))),
            "Modify(Name) events should be relevant (covers renames)"
        );
    }

    #[test]
    fn is_relevant_event_remove() {
        assert!(
            is_relevant_event(EventKind::Remove(notify::event::RemoveKind::File)),
            "Remove events should be relevant"
        );
    }

    #[test]
    fn is_relevant_event_access_not_relevant() {
        assert!(
            !is_relevant_event(EventKind::Access(notify::event::AccessKind::Read)),
            "Access events should not be relevant"
        );
    }

    // -------------------------------------------------------------------------
    // Watch directory resolution
    // -------------------------------------------------------------------------

    #[test]
    fn watch_dir_for_path_bare_filename() {
        assert_eq!(
            watch_dir_for_path(Path::new("routing-config.json")),
            PathBuf::from("."),
            "bare filename should resolve to current directory"
        );
    }

    #[test]
    fn watch_dir_for_path_with_directory() {
        assert_eq!(
            watch_dir_for_path(Path::new("/etc/routing/routing-config.json")),
            PathBuf::from("/etc/routing"),
            "absolute path should use its parent directory"
        );
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Poll `predicate` every 20ms until it returns `true` or `timeout` elapses.
    fn poll_until(timeout: Duration, predicate: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("poll_until timed out after {timeout:?}");
    }
}
