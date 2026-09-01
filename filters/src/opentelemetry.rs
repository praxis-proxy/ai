// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Feature-gated routing semantics for Praxis core request traces.
//!
//! Praxis core owns subscriber installation, OTLP export, propagation,
//! sampling, request lifecycle spans, and provider-hop client spans. AI owns
//! only the semantic decisions made by `intelligent_route` (edge routing
//! selection) and `provider_route` (provider-local backend resolution).

use std::sync::Arc;

use tracing::field::Empty;

use crate::routing::descriptor::RouteCandidate;

/// Borrowed, validated attributes for a routing decision span.
struct RoutingSelection<'a> {
    /// Producer-assigned admission state label.
    admission_state: &'static str,
    /// Selected upstream cluster name.
    cluster: &'a str,
    /// Routed capability kind.
    kind: &'static str,
    /// Site handling the inbound request.
    local_site: &'a str,
    /// Selected provider capability name.
    provider: &'a str,
    /// Producer-assigned candidate rank, when available.
    rank: Option<u32>,
    /// Serving overlay semantic revision, when available.
    revision: Option<&'a str>,
    /// Site that owns the selected provider.
    site: &'a str,
    /// Stable identifier assigned to the selected provider.
    stable_id: &'a str,
    /// Producer-assigned selection tier, when available.
    tier: Option<&'a str>,
}

impl<'a> RoutingSelection<'a> {
    /// Project only bounded routing state; request and credential state is not
    /// accepted by this constructor.
    fn from_candidate(
        candidate: &'a RouteCandidate,
        local_site: &'a Arc<str>,
        semantic_revision: Option<&'a Arc<str>>,
    ) -> Self {
        Self {
            admission_state: candidate.admission_state.as_str(),
            cluster: candidate.cluster.as_ref(),
            kind: candidate.kind.as_str(),
            local_site: local_site.as_ref(),
            provider: candidate.name.as_ref(),
            rank: candidate.rank,
            revision: semantic_revision.map(AsRef::as_ref),
            site: candidate.site.as_ref(),
            stable_id: candidate.stable_id.as_ref(),
            tier: candidate.selection_tier.as_deref(),
        }
    }
}

/// Borrowed, validated attributes for a provider-local routing decision span.
struct ProviderRouteSelection<'a> {
    /// Configured provider-boundary identifier.
    provider_id: &'a str,
    /// Provider-local backend cluster resolved for the candidate.
    cluster: &'a str,
    /// Configured model accepted by the resolved route.
    model: &'a str,
    /// Validated candidate identifier supplied by the trusted edge hop.
    candidate_id: &'a str,
    /// Edge serving-overlay revision, when supplied and validated.
    revision: Option<&'a str>,
}

impl<'a> ProviderRouteSelection<'a> {
    /// Project only bounded provider-route state; request and credential state
    /// is not accepted by this constructor.
    fn new(
        provider_id: &'a Arc<str>,
        cluster: &'a Arc<str>,
        model: &'a Arc<str>,
        candidate_id: &'a str,
        revision: Option<&'a str>,
    ) -> Self {
        Self {
            provider_id: provider_id.as_ref(),
            cluster: cluster.as_ref(),
            model: model.as_ref(),
            candidate_id,
            revision,
        }
    }
}

/// Emit a bounded child span for a completed routing decision.
///
/// No prompt, body, credential, authorization header, cookie, session key, or
/// raw request identifier is recorded. When the feature is disabled this call
/// site is compiled out entirely.
pub(crate) fn record_routing_selection(
    candidate: &RouteCandidate,
    local_site: &Arc<str>,
    semantic_revision: Option<&Arc<str>>,
) {
    let selection = RoutingSelection::from_candidate(candidate, local_site, semantic_revision);
    let span = tracing::info_span!(
        "routing.select",
        "selected.provider" = selection.provider,
        "selected.cluster" = selection.cluster,
        "selected.site" = selection.site,
        "selected.stable_id" = selection.stable_id,
        "routing.admission_state" = selection.admission_state,
        "routing.kind" = selection.kind,
        "routing.local_site" = selection.local_site,
        "routing.rank" = Empty,
        "routing.selection_tier" = Empty,
        "overlay.revision" = Empty,
    );
    if let Some(rank) = selection.rank {
        span.record("routing.rank", rank);
    }
    if let Some(tier) = selection.tier {
        span.record("routing.selection_tier", tier);
    }
    if let Some(revision) = selection.revision {
        span.record("overlay.revision", revision);
    }
    let _entered = span.enter();
}

/// Emit a bounded child span for a completed `provider_route` resolution.
///
/// Records the resolved backend cluster the request was routed to; it is not
/// proof that a downstream endpoint, pod, or model server successfully served
/// the request. `provider_id` is the configured provider-boundary identifier,
/// not necessarily the mTLS peer identity. `overlay_revision` is the
/// edge-supplied serving overlay revision after syntax and trust-boundary
/// validation; it is correlation evidence, not a provider-local config
/// revision or an authorization decision.
///
/// No prompt, body, credential, authorization header, cookie, session key, or
/// raw request identifier is recorded. When the feature is disabled this call
/// site is compiled out entirely.
pub(crate) fn record_provider_route_selection(
    provider_id: &Arc<str>,
    cluster: &Arc<str>,
    model: &Arc<str>,
    candidate_id: &str,
    overlay_revision: Option<&str>,
) {
    let selection = ProviderRouteSelection::new(provider_id, cluster, model, candidate_id, overlay_revision);
    let span = tracing::info_span!(
        "provider.route",
        "provider.id" = selection.provider_id,
        "provider.backend.cluster" = selection.cluster,
        "provider.route.model" = selection.model,
        "provider.route.candidate_id" = selection.candidate_id,
        "overlay.revision" = Empty,
    );
    if let Some(revision) = selection.revision {
        span.record("overlay.revision", revision);
    }
    let _entered = span.enter();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ProviderRouteSelection, RoutingSelection, record_provider_route_selection};
    use crate::routing::descriptor::{AdmissionState, CapabilityKind, RouteCandidate};

    #[test]
    fn projects_only_validated_routing_attributes() {
        let candidate = candidate();
        let local_site = Arc::from("site-local");
        let revision = Arc::from("revision-a");
        let fields = RoutingSelection::from_candidate(&candidate, &local_site, Some(&revision));

        assert_eq!(fields.provider, "model-a", "provider must match the candidate name");
        assert_eq!(fields.cluster, "provider-a", "cluster must match the selected upstream");
        assert_eq!(fields.site, "site-a", "site must match the provider owner");
        assert_eq!(fields.stable_id, "stable-a", "stable ID must match the candidate");
        assert_eq!(
            fields.local_site, "site-local",
            "local site must match the routing context"
        );
        assert_eq!(fields.rank, Some(2), "rank must preserve producer metadata");
        assert_eq!(
            fields.tier,
            Some("same_region"),
            "selection tier must preserve producer metadata"
        );
        assert_eq!(
            fields.revision,
            Some("revision-a"),
            "revision must identify the serving overlay"
        );
    }

    #[test]
    fn optional_attributes_remain_absent() {
        let mut candidate = candidate();
        candidate.rank = None;
        candidate.selection_tier = None;
        let local_site = Arc::from("site-local");
        let fields = RoutingSelection::from_candidate(&candidate, &local_site, None);

        assert_eq!(fields.rank, None, "missing rank must remain absent");
        assert_eq!(fields.tier, None, "missing selection tier must remain absent");
        assert_eq!(fields.revision, None, "missing overlay revision must remain absent");
    }

    #[test]
    fn projects_only_validated_provider_route_attributes() {
        let provider_id: Arc<str> = Arc::from("provider-a");
        let cluster: Arc<str> = Arc::from("cluster-a");
        let model: Arc<str> = Arc::from("model-a");
        let fields = ProviderRouteSelection::new(&provider_id, &cluster, &model, "candidate-a", Some("revision-a"));

        assert_eq!(fields.provider_id, "provider-a");
        assert_eq!(fields.cluster, "cluster-a");
        assert_eq!(fields.model, "model-a");
        assert_eq!(fields.candidate_id, "candidate-a");
        assert_eq!(fields.revision, Some("revision-a"));

        // Keep a smoke assertion around the tracing macro in addition to the
        // projection contract above.
        record_provider_route_selection(&provider_id, &cluster, &model, "candidate-a", Some("revision-a"));
    }

    #[test]
    fn provider_route_optional_revision_remains_absent() {
        let provider_id: Arc<str> = Arc::from("provider-a");
        let cluster: Arc<str> = Arc::from("cluster-a");
        let model: Arc<str> = Arc::from("model-a");
        let fields = ProviderRouteSelection::new(&provider_id, &cluster, &model, "candidate-a", None);

        assert_eq!(fields.revision, None);
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Build a fully populated candidate for attribute projection tests.
    fn candidate() -> RouteCandidate {
        RouteCandidate {
            admission_state: AdmissionState::NewAndExisting,
            cluster: Arc::from("provider-a"),
            credential: None,
            fresh: true,
            kind: CapabilityKind::InferenceModel,
            name: Arc::from("model-a"),
            rank: Some(2),
            selection_tier: Some(Arc::from("same_region")),
            site: Arc::from("site-a"),
            stable_id: Arc::from("stable-a"),
        }
    }
}
