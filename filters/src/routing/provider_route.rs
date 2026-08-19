// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Provider-local validation and routing for an edge-selected provider hop.
//!
//! This is intentionally not a second `intelligent_route`: it does not discover,
//! score, rank, fail over, or maintain sessions. It validates the exact
//! edge-selected candidate against provider-owned model/path policy, selects a
//! preconfigured local backend cluster, and emits an optional local credential
//! reference for `credential_inject`. Edge peer headers are AI-owned,
//! non-reserved fields that reach this filter only after provider mTLS and
//! `peer_identity_trust`; this filter removes them before the backend hop.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use http::{HeaderName, HeaderValue, header::AUTHORIZATION};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, parse_filter_config};
use serde::Deserialize;

use super::{
    descriptor,
    metadata::{
        CandidateCredential, OVERLAY_REVISION_HEADER, PROVIDER_ATTRIBUTION_HEADER,
        PROVIDER_ATTRIBUTION_RESPONSE_HEADER, PROVIDER_HOP_REQUEST_ID_HEADER, PROVIDER_OVERLAY_REVISION_HEADER,
        PROVIDER_REQUEST_ID_HEADER, PROVIDER_ROUTE_CANDIDATE_ID, PROVIDER_ROUTE_CLUSTER, PROVIDER_ROUTE_MODEL,
        PROVIDER_ROUTE_OVERLAY_REVISION, PROVIDER_ROUTE_PROVIDER_ID, PROVIDER_ROUTE_REQUEST_ID,
        SELECTED_CANDIDATE_HEADER, set_credential_metadata,
    },
};

/// Upper bound on provider route entries.
const MAX_PROVIDER_ROUTES: usize = 1024;
/// Upper bound on exact request paths per provider route.
const MAX_PATHS_PER_ROUTE: usize = 64;
/// Upper bound on header/field value length.
const MAX_VALUE_LEN: usize = 256;

/// Invalid edge-supplied overlay revision.
#[derive(Debug)]
struct InvalidPeerOverlayRevision;

/// Deserialized configuration for the `provider_route` filter.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderRouteFilterConfig {
    /// Provider-owned identifier used for observability and demo attribution.
    provider_id: String,

    /// Header populated by an inference parser with the requested model.
    #[serde(default = "default_model_header")]
    model_header: String,

    /// Exact provider-local candidate mappings.
    routes: Vec<ProviderRouteConfig>,

    /// Add a provider attribution response header for demo evidence.
    #[serde(default)]
    emit_demo_attribution: bool,
}

/// A single candidate-to-cluster mapping in the provider config.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderRouteConfig {
    /// Stable candidate ID selected by the edge `intelligent_route`.
    candidate_id: String,

    /// Provider-local backend cluster.
    cluster: String,

    /// Optional provider-local credential reference for the final API hop.
    credential: Option<CandidateCredential>,

    /// Exact model accepted for this candidate.
    model: String,

    /// Exact inference paths accepted for this candidate.
    paths: Vec<String>,
}

/// Returns the default model header name (`X-Model`).
fn default_model_header() -> String {
    "X-Model".to_owned()
}

/// Resolved route entry for a single candidate.
#[derive(Debug)]
struct ProviderRoute {
    /// Backend cluster name.
    cluster: Arc<str>,
    /// Optional credential reference for `credential_inject`.
    credential: Option<CandidateCredential>,
    /// Accepted model name.
    model: Arc<str>,
    /// Accepted request paths.
    paths: Vec<Arc<str>>,
}

/// Exact provider-local mapping from an authenticated intelligent routing
/// selection to a private backend cluster.
///
/// The provider listener requires downstream mTLS and must run
/// `peer_identity_trust` before this filter. The filter consumes the exact
/// `x-ai-routing-candidate`, `x-ai-routing-request-id`, and optional
/// `x-ai-routing-revision` fields, validates candidate/model/path
/// against provider-local configuration, and removes all `x-ai-routing-*`
/// peer fields before the backend hop. It also removes client-supplied
/// provider attribution fields before writing provider-owned replacements.
/// A valid peer overlay revision is rewritten into the provider-owned
/// namespace for backend telemetry; it is correlation evidence, not an
/// authorization grant.
///
/// These names are AI-owned rather than Praxis-reserved because Praxis
/// intentionally strips `x-praxis-*` headers before upstream requests.
/// Praxis AI startup validation rejects optional/plaintext client certificate
/// modes, a provider chain that does not begin with `peer_identity_trust`,
/// conditional/fail-open boundary filters, and branch-conditional provider
/// consumers.
pub struct ProviderRouteFilter {
    /// Emit a demo attribution header on responses.
    emit_demo_attribution: bool,
    /// Header name containing the requested model.
    model_header: HeaderName,
    /// Provider-owned identifier for observability.
    provider_id: Arc<str>,
    /// Prevalidated provider attribution header.
    provider_id_header: HeaderValue,
    /// Candidate-to-route map.
    routes: HashMap<Arc<str>, ProviderRoute>,
}

impl ProviderRouteFilter {
    /// Construct a provider-local route map.
    ///
    /// Pipeline configuration must place downstream mTLS and
    /// `peer_identity_trust` enforcement before this filter.
    ///
    /// # Errors
    ///
    /// Returns an error if routes are empty, duplicated, or contain
    /// invalid values.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ProviderRouteFilterConfig = parse_filter_config("provider_route", config)?;
        validate_value("provider_id", &cfg.provider_id)?;
        let provider_id_header = HeaderValue::from_str(&cfg.provider_id)
            .map_err(|error| FilterError::from(format!("provider_route: invalid provider_id header: {error}")))?;
        if cfg.routes.is_empty() || cfg.routes.len() > MAX_PROVIDER_ROUTES {
            return Err(format!("provider_route: routes must contain 1-{MAX_PROVIDER_ROUTES} entries").into());
        }

        let model_header = descriptor::validate_model_header(&cfg.model_header)?;
        let mut routes = HashMap::with_capacity(cfg.routes.len());
        for route in cfg.routes {
            validate_route(&route)?;
            let candidate_id: Arc<str> = Arc::from(route.candidate_id.as_str());
            let provider_route = ProviderRoute {
                cluster: Arc::from(route.cluster.as_str()),
                credential: route.credential,
                model: Arc::from(route.model.as_str()),
                paths: route.paths.into_iter().map(Arc::from).collect(),
            };
            if routes.insert(candidate_id, provider_route).is_some() {
                return Err("provider_route: duplicate candidate_id".into());
            }
        }

        Ok(Box::new(Self {
            emit_demo_attribution: cfg.emit_demo_attribution,
            model_header,
            provider_id: Arc::from(cfg.provider_id.as_str()),
            provider_id_header,
            routes,
        }))
    }
}

#[async_trait]
impl HttpFilter for ProviderRouteFilter {
    fn name(&self) -> &'static str {
        "provider_route"
    }

    fn selects_cluster(&self) -> bool {
        true
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keeps the fail-closed provider-boundary decision atomic"
    )]
    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        strip_edge_headers(ctx);

        let Some(candidate_id) = request_header(ctx, SELECTED_CANDIDATE_HEADER).map(str::to_owned) else {
            return Ok(FilterAction::Reject(Rejection::status(403)));
        };
        let Some(request_id) = request_header(ctx, PROVIDER_HOP_REQUEST_ID_HEADER).map(str::to_owned) else {
            return Ok(FilterAction::Reject(Rejection::status(403)));
        };
        let overlay_revision = match peer_overlay_revision(ctx) {
            Ok(revision) => revision.map(str::to_owned),
            Err(InvalidPeerOverlayRevision) => return Ok(FilterAction::Reject(Rejection::status(403))),
        };
        let Some(model) = request_header_by_name(ctx, &self.model_header).map(str::to_owned) else {
            return Ok(FilterAction::Reject(Rejection::status(400)));
        };
        let Some(route) = self.routes.get(candidate_id.as_str()) else {
            return Ok(FilterAction::Reject(Rejection::status(403)));
        };
        if route.model.as_ref() != model || !route.paths.iter().any(|path| path.as_ref() == ctx.request.uri.path()) {
            return Ok(FilterAction::Reject(Rejection::status(404)));
        }

        ctx.cluster = Some(Arc::clone(&route.cluster));
        ctx.set_metadata(PROVIDER_ROUTE_CANDIDATE_ID, &candidate_id);
        ctx.set_metadata(PROVIDER_ROUTE_CLUSTER, &*route.cluster);
        ctx.set_metadata(PROVIDER_ROUTE_MODEL, &model);
        ctx.set_metadata(PROVIDER_ROUTE_PROVIDER_ID, &*self.provider_id);
        ctx.set_metadata(PROVIDER_ROUTE_REQUEST_ID, &request_id);
        if let Some(revision) = &overlay_revision {
            ctx.set_metadata(PROVIDER_ROUTE_OVERLAY_REVISION, revision);
        }
        set_credential_metadata(ctx, route.credential.as_ref());
        set_provider_headers(ctx, &self.provider_id_header, &request_id, overlay_revision.as_deref())?;

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !self.emit_demo_attribution {
            return Ok(FilterAction::Continue);
        }
        let Some(response) = ctx.response_header.as_mut() else {
            return Ok(FilterAction::Continue);
        };
        response.headers.insert(
            HeaderName::from_static(PROVIDER_ATTRIBUTION_RESPONSE_HEADER),
            self.provider_id_header.clone(),
        );
        ctx.response_headers_modified = true;
        Ok(FilterAction::Continue)
    }
}

/// Remove peer fields, spoofable backend attribution, and customer credentials.
///
/// Provider attribution is re-created below from provider-owned configuration.
/// This exact remove-and-overwrite contract is required because these AI-owned
/// fields are intentionally outside Praxis's globally stripped namespaces.
fn strip_edge_headers(ctx: &mut HttpFilterContext<'_>) {
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(SELECTED_CANDIDATE_HEADER));
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER));
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(OVERLAY_REVISION_HEADER));
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(PROVIDER_REQUEST_ID_HEADER));
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(PROVIDER_OVERLAY_REVISION_HEADER));
    ctx.request_headers_to_remove
        .push(HeaderName::from_static(PROVIDER_ATTRIBUTION_HEADER));
    ctx.request_headers_to_remove.push(AUTHORIZATION);
}

/// Set provider request ID and attribution headers on the backend request.
fn set_provider_headers(
    ctx: &mut HttpFilterContext<'_>,
    provider_id: &HeaderValue,
    request_id: &str,
    overlay_revision: Option<&str>,
) -> Result<(), FilterError> {
    ctx.request_headers_to_set.push((
        HeaderName::from_static(PROVIDER_REQUEST_ID_HEADER),
        HeaderValue::from_str(request_id)
            .map_err(|e| FilterError::from(format!("provider_route: invalid provider request ID: {e}")))?,
    ));
    ctx.request_headers_to_set.push((
        HeaderName::from_static(PROVIDER_ATTRIBUTION_HEADER),
        provider_id.clone(),
    ));
    if let Some(revision) = overlay_revision {
        ctx.request_headers_to_set.push((
            HeaderName::from_static(PROVIDER_OVERLAY_REVISION_HEADER),
            HeaderValue::from_str(revision).map_err(|error| {
                FilterError::from(format!("provider_route: invalid provider overlay revision: {error}"))
            })?,
        ));
    }
    Ok(())
}

/// Read and validate the optional serving revision supplied by the edge.
fn peer_overlay_revision<'a>(ctx: &'a HttpFilterContext<'_>) -> Result<Option<&'a str>, InvalidPeerOverlayRevision> {
    let header = HeaderName::from_static(OVERLAY_REVISION_HEADER);
    let Some(value) = ctx.request.headers.get(header) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_error| InvalidPeerOverlayRevision)?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(InvalidPeerOverlayRevision);
    }
    Ok(Some(value))
}

/// Extract a bounded, non-empty header value by static name.
fn request_header<'a>(ctx: &'a HttpFilterContext<'_>, name: &'static str) -> Option<&'a str> {
    let header = HeaderName::from_static(name);
    request_header_by_name(ctx, &header)
}

/// Extract a bounded, non-empty header value by [`HeaderName`].
fn request_header_by_name<'a>(ctx: &'a HttpFilterContext<'_>, name: &HeaderName) -> Option<&'a str> {
    let value = ctx.request.headers.get(name)?.to_str().ok()?;
    (!value.trim().is_empty() && value.len() <= MAX_VALUE_LEN).then_some(value)
}

/// Validate all fields of a single route entry.
fn validate_route(route: &ProviderRouteConfig) -> Result<(), FilterError> {
    validate_value("candidate_id", &route.candidate_id)?;
    validate_value("cluster", &route.cluster)?;
    validate_value("model", &route.model)?;
    if route.paths.is_empty() || route.paths.len() > MAX_PATHS_PER_ROUTE {
        return Err(format!("provider_route: paths must contain 1-{MAX_PATHS_PER_ROUTE} entries").into());
    }
    let mut seen_paths = HashSet::with_capacity(route.paths.len());
    for path in &route.paths {
        if !path.starts_with('/') || path.len() > MAX_VALUE_LEN {
            return Err("provider_route: paths must be absolute and bounded".into());
        }
        if !seen_paths.insert(path) {
            return Err("provider_route: duplicate path".into());
        }
    }
    if let Some(credential) = &route.credential {
        validate_value("credential.secretRef.name", &credential.secret_ref.name)?;
        validate_value("credential.secretRef.namespace", &credential.secret_ref.namespace)?;
        validate_value("credential.secretRef.key", &credential.secret_ref.key)?;
        if credential.strategy != super::metadata::STRATEGY_BEARER_TOKEN {
            return Err("provider_route: unsupported credential strategy".into());
        }
    }
    Ok(())
}

/// Reject blank or oversized string values.
fn validate_value(field: &str, value: &str) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > MAX_VALUE_LEN {
        return Err(format!("provider_route: {field} must be non-blank and at most {MAX_VALUE_LEN} bytes").into());
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
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use http::Method;

    use super::*;
    use crate::{
        routing::{
            credential_inject::CredentialInjectFilter,
            metadata::{CREDENTIAL_KEY, CREDENTIAL_NAME, CREDENTIAL_NAMESPACE, CREDENTIAL_STRATEGY},
        },
        test_utils,
    };

    // -------------------------------------------------------------------------
    // Header Stripping
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn strips_routing_protocol_headers() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            ctx.request_headers_to_remove
                .iter()
                .any(|h| h.as_str() == SELECTED_CANDIDATE_HEADER),
            "must remove selected-candidate header"
        );
        assert!(
            ctx.request_headers_to_remove
                .iter()
                .any(|h| h.as_str() == PROVIDER_HOP_REQUEST_ID_HEADER),
            "must remove hop-request-id header"
        );
        assert!(
            ctx.request_headers_to_remove
                .iter()
                .any(|h| h.as_str() == OVERLAY_REVISION_HEADER),
            "must remove overlay-revision header"
        );
        assert!(
            ctx.request_headers_to_remove.contains(&AUTHORIZATION),
            "must remove customer Authorization"
        );
    }

    #[tokio::test]
    async fn strips_spoofed_attribution_header() {
        let f = make_filter("abc123");
        let mut req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        req.headers.insert(
            HeaderName::from_static(PROVIDER_ATTRIBUTION_HEADER),
            HeaderValue::from_static("spoofed-value"),
        );
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            ctx.request_headers_to_remove
                .iter()
                .any(|h| h.as_str() == PROVIDER_ATTRIBUTION_HEADER),
            "must remove spoofed attribution header"
        );
    }

    #[tokio::test]
    async fn overwrites_all_spoofable_backend_attribution() {
        let f = make_filter("abc123");
        let mut req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        req.headers.insert(
            HeaderName::from_static(PROVIDER_ATTRIBUTION_HEADER),
            HeaderValue::from_static("spoofed-provider"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_REQUEST_ID_HEADER),
            HeaderValue::from_static("spoofed-request"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_OVERLAY_REVISION_HEADER),
            HeaderValue::from_static("spoofed-revision"),
        );
        let mut ctx = test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        for name in [
            PROVIDER_ATTRIBUTION_HEADER,
            PROVIDER_REQUEST_ID_HEADER,
            PROVIDER_OVERLAY_REVISION_HEADER,
        ] {
            assert!(
                ctx.request_headers_to_remove.contains(&HeaderName::from_static(name)),
                "{name} must be removed before provider-owned output"
            );
        }
        assert_eq!(pending_header(&ctx, PROVIDER_ATTRIBUTION_HEADER), Some("test-provider"));
        assert_eq!(pending_header(&ctx, PROVIDER_REQUEST_ID_HEADER), Some("req-1"));
        assert_eq!(
            pending_header(&ctx, PROVIDER_OVERLAY_REVISION_HEADER),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    #[tokio::test]
    async fn peer_context_is_consumed_without_backend_forwarding() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        for name in [
            SELECTED_CANDIDATE_HEADER,
            PROVIDER_HOP_REQUEST_ID_HEADER,
            OVERLAY_REVISION_HEADER,
        ] {
            assert!(
                ctx.request_headers_to_remove.contains(&HeaderName::from_static(name)),
                "{name} must be removed"
            );
            assert!(
                pending_header(&ctx, name).is_none(),
                "{name} must not be re-emitted to the backend"
            );
        }
        assert_eq!(
            pending_header(&ctx, PROVIDER_OVERLAY_REVISION_HEADER),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            "provider must rewrite the peer revision under provider ownership"
        );
    }

    #[tokio::test]
    async fn malformed_peer_overlay_revision_is_denied() {
        let f = make_filter("abc123");
        let mut req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        req.headers.insert(
            HeaderName::from_static(OVERLAY_REVISION_HEADER),
            HeaderValue::from_static("not-a-sha256-revision"),
        );
        let mut ctx = test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Reject(r) if r.status == 403));
        assert!(
            ctx.request_headers_to_remove
                .contains(&HeaderName::from_static(OVERLAY_REVISION_HEADER))
        );
        assert!(ctx.request_headers_to_set.is_empty());
    }

    #[tokio::test]
    async fn legacy_peer_without_overlay_revision_remains_supported() {
        let f = make_filter("abc123");
        let mut req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        req.headers.remove(OVERLAY_REVISION_HEADER);
        let mut ctx = test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(pending_header(&ctx, PROVIDER_OVERLAY_REVISION_HEADER).is_none());
    }

    #[tokio::test]
    async fn denied_request_still_removes_peer_context_and_customer_auth() {
        let f = make_filter("abc123");
        let mut req = request_with_peer_context("/v1/chat/completions", "unknown", "req-1", "sim-model-v1");
        req.headers
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer customer-secret"));
        let mut ctx = test_utils::make_filter_context(&req);

        let action = f.on_request(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Reject(r) if r.status == 403));
        for name in [
            SELECTED_CANDIDATE_HEADER,
            PROVIDER_HOP_REQUEST_ID_HEADER,
            OVERLAY_REVISION_HEADER,
        ] {
            assert!(ctx.request_headers_to_remove.contains(&HeaderName::from_static(name)));
        }
        assert!(ctx.request_headers_to_remove.contains(&AUTHORIZATION));
        assert!(ctx.request_headers_to_set.is_empty());
    }

    // -------------------------------------------------------------------------
    // Exact Candidate Matching
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn known_candidate_selects_cluster() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue), "should route");
        assert_eq!(ctx.cluster.as_deref(), Some("mock-backend"));
    }

    #[tokio::test]
    async fn unknown_candidate_denied_403() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "wrong-id", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "unknown candidate must be denied 403"
        );
    }

    #[tokio::test]
    async fn wrong_model_denied_404() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "wrong-model");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "wrong model must be denied 404"
        );
    }

    #[tokio::test]
    async fn wrong_path_denied_404() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/wrong/path", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 404),
            "wrong path must be denied 404"
        );
    }

    // -------------------------------------------------------------------------
    // Missing Required Headers
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn missing_candidate_header_denied_403() {
        let f = make_filter("abc123");
        let mut req = test_utils::make_request(Method::POST, "/v1/chat/completions");
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_static("req-1"),
        );
        req.headers.insert("X-Model", HeaderValue::from_static("sim-model-v1"));
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "missing candidate header must be denied 403"
        );
    }

    #[tokio::test]
    async fn missing_hop_request_id_denied_403() {
        let f = make_filter("abc123");
        let mut req = test_utils::make_request(Method::POST, "/v1/chat/completions");
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_static("abc123"),
        );
        req.headers.insert("X-Model", HeaderValue::from_static("sim-model-v1"));
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 403),
            "missing hop request ID must be denied 403"
        );
    }

    #[tokio::test]
    async fn missing_model_header_denied_400() {
        let f = make_filter("abc123");
        let mut req = test_utils::make_request(Method::POST, "/v1/chat/completions");
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_static("abc123"),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_static("req-1"),
        );
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(r) if r.status == 400),
            "missing model header must be denied 400"
        );
    }

    // -------------------------------------------------------------------------
    // Oversized Or Empty Values Fail Closed
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn empty_candidate_id_denied() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "empty candidate ID must be denied"
        );
    }

    #[tokio::test]
    async fn oversized_candidate_id_denied() {
        let f = make_filter("abc123");
        let big = "a".repeat(MAX_VALUE_LEN + 1);
        let req = request_with_peer_context("/v1/chat/completions", &big, "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let action = f.on_request(&mut ctx).await.unwrap();
        assert!(
            matches!(action, FilterAction::Reject(_)),
            "oversized candidate ID must be denied"
        );
    }

    // -------------------------------------------------------------------------
    // Metadata Output
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn sets_provider_route_metadata() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        assert_eq!(ctx.get_metadata(PROVIDER_ROUTE_CANDIDATE_ID), Some("abc123"));
        assert_eq!(ctx.get_metadata(PROVIDER_ROUTE_CLUSTER), Some("mock-backend"));
        assert_eq!(ctx.get_metadata(PROVIDER_ROUTE_MODEL), Some("sim-model-v1"));
        assert_eq!(ctx.get_metadata(PROVIDER_ROUTE_PROVIDER_ID), Some("test-provider"));
        assert!(ctx.get_metadata(PROVIDER_ROUTE_REQUEST_ID).is_some());
    }

    // -------------------------------------------------------------------------
    // Provider Attribution
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn sets_provider_attribution_request_header() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        let attr = ctx
            .request_headers_to_set
            .iter()
            .find(|(h, _)| h.as_str() == PROVIDER_ATTRIBUTION_HEADER);
        assert!(attr.is_some(), "must set provider attribution header");
        assert_eq!(attr.unwrap().1.to_str().unwrap(), "test-provider");
    }

    #[tokio::test]
    async fn emits_demo_attribution_response_header() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        let mut resp = test_utils::make_response();
        ctx.response_header = Some(&mut resp);
        let action = f.on_response(&mut ctx).await.unwrap();
        assert!(matches!(action, FilterAction::Continue));
        let value = resp.headers.get(PROVIDER_ATTRIBUTION_RESPONSE_HEADER);
        assert_eq!(
            value.map(|v| v.to_str().unwrap()),
            Some("test-provider"),
            "demo attribution response header"
        );
    }

    #[tokio::test]
    async fn omits_demo_attribution_response_header_by_default() {
        let config = provider_config_with_demo_attribution(
            "abc123",
            "sim-model-v1",
            &["/v1/chat/completions"],
            "mock-backend",
            None,
        );
        let f = ProviderRouteFilter::from_config(&config).unwrap();
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        let _action = f.on_request(&mut ctx).await.unwrap();
        let mut resp = test_utils::make_response();
        ctx.response_header = Some(&mut resp);

        let action = f.on_response(&mut ctx).await.unwrap();

        assert!(matches!(action, FilterAction::Continue));
        assert!(!ctx.response_headers_modified);
        assert!(resp.headers.get(PROVIDER_ATTRIBUTION_RESPONSE_HEADER).is_none());
    }

    // -------------------------------------------------------------------------
    // Credential Metadata
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn no_credential_clears_metadata() {
        let f = make_filter("abc123");
        let req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        let mut ctx = test_utils::make_filter_context(&req);
        ctx.set_metadata(CREDENTIAL_STRATEGY, "stale");
        ctx.set_metadata(CREDENTIAL_NAME, "stale");
        ctx.set_metadata(CREDENTIAL_NAMESPACE, "stale");
        ctx.set_metadata(CREDENTIAL_KEY, "stale");
        let _action = f.on_request(&mut ctx).await.unwrap();
        for key in [
            CREDENTIAL_STRATEGY,
            CREDENTIAL_NAME,
            CREDENTIAL_NAMESPACE,
            CREDENTIAL_KEY,
        ] {
            assert!(
                ctx.get_metadata(key).is_none(),
                "{key} must be cleared when no credential is configured"
            );
        }
    }

    #[tokio::test]
    async fn credential_route_drives_final_hop_injection() {
        let provider = ProviderRouteFilter::from_config(&credential_provider_config()).unwrap();
        let inject_config: serde_yaml::Value = serde_yaml::from_str(
            "credentials:\n\
             \x20 - name: provider-token\n\
             \x20   namespace: grid-system\n\
             \x20   key: token\n\
             \x20   value: provider-secret\n",
        )
        .unwrap();
        let inject = CredentialInjectFilter::from_config(&inject_config).unwrap();
        let mut req = request_with_peer_context("/v1/chat/completions", "abc123", "req-1", "sim-model-v1");
        req.headers
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer customer-secret"));
        let mut ctx = test_utils::make_filter_context(&req);

        let provider_action = provider.on_request(&mut ctx).await.unwrap();
        assert!(matches!(provider_action, FilterAction::Continue));
        assert_eq!(ctx.get_metadata(CREDENTIAL_STRATEGY), Some("bearer_token"));
        assert_eq!(ctx.get_metadata(CREDENTIAL_NAME), Some("provider-token"));
        assert_eq!(ctx.get_metadata(CREDENTIAL_NAMESPACE), Some("grid-system"));
        assert_eq!(ctx.get_metadata(CREDENTIAL_KEY), Some("token"));

        let inject_action = inject.on_request(&mut ctx).await.unwrap();
        assert!(matches!(inject_action, FilterAction::Continue));
        let authorization_values = ctx
            .request_headers_to_set
            .iter()
            .filter(|(name, _)| name == AUTHORIZATION)
            .collect::<Vec<_>>();
        assert_eq!(authorization_values.len(), 1, "provider Authorization must be set once");
        assert_eq!(authorization_values[0].1, "Bearer provider-secret");
        assert!(
            ctx.request_headers_to_remove.contains(&AUTHORIZATION),
            "customer Authorization must be removed"
        );
    }

    // -------------------------------------------------------------------------
    // Config Validation
    // -------------------------------------------------------------------------

    #[test]
    fn empty_routes_rejected() {
        let yaml = "provider_id: p\nroutes: []\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(ProviderRouteFilter::from_config(&val).is_err());
    }

    #[test]
    fn too_many_routes_rejected() {
        let routes = (0..=MAX_PROVIDER_ROUTES)
            .map(|index| {
                format!(
                    "  - candidate_id: candidate-{index}\n    model: model-{index}\n    paths: [/v1/chat/completions]\n    cluster: cluster-{index}\n"
                )
            })
            .collect::<String>();
        let yaml = format!("provider_id: p\nroutes:\n{routes}");
        let val: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();

        assert!(
            ProviderRouteFilter::from_config(&val).is_err(),
            "route count must be bounded"
        );
    }

    #[test]
    fn blank_provider_id_rejected() {
        let yaml = "provider_id: ''\nroutes:\n  - candidate_id: a\n    model: m\n    paths: [/x]\n    cluster: c\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(ProviderRouteFilter::from_config(&val).is_err());
    }

    #[test]
    fn invalid_provider_id_header_rejected_at_construction() {
        let yaml = "provider_id: \"bad\\0value\"\nroutes:\n  - candidate_id: a\n    model: m\n    paths: [/x]\n    cluster: c\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let error = ProviderRouteFilter::from_config(&val)
            .err()
            .expect("invalid header value must fail construction");
        assert!(error.to_string().contains("invalid provider_id header"), "{error}");
    }

    #[test]
    fn duplicate_candidate_id_rejected() {
        let yaml = "provider_id: p\nroutes:\n  - candidate_id: a\n    model: m\n    paths: [/x]\n    cluster: c\n  - candidate_id: a\n    model: m2\n    paths: [/y]\n    cluster: c2\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = ProviderRouteFilter::from_config(&val);
        assert!(result.is_err(), "duplicate candidate_id must be rejected");
    }

    #[test]
    fn empty_paths_rejected() {
        let yaml = "provider_id: p\nroutes:\n  - candidate_id: a\n    model: m\n    paths: []\n    cluster: c\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(ProviderRouteFilter::from_config(&val).is_err());
    }

    #[test]
    fn relative_path_rejected() {
        let yaml =
            "provider_id: p\nroutes:\n  - candidate_id: a\n    model: m\n    paths: [relative]\n    cluster: c\n";
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(ProviderRouteFilter::from_config(&val).is_err());
    }

    #[test]
    fn duplicate_path_rejected() {
        let config = provider_config("a", "m", &["/x", "/x"], "c");
        assert!(ProviderRouteFilter::from_config(&config).is_err());
    }

    #[test]
    fn too_many_paths_rejected() {
        let paths = (0..=MAX_PATHS_PER_ROUTE)
            .map(|i| format!("/path-{i}"))
            .collect::<Vec<_>>();
        let path_refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
        let config = provider_config("a", "m", &path_refs, "c");
        assert!(ProviderRouteFilter::from_config(&config).is_err());
    }

    #[test]
    fn selects_cluster_returns_true() {
        let f = make_filter("abc123");
        assert!(f.selects_cluster(), "provider route must declare cluster selection");
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    fn provider_config(candidate_id: &str, model: &str, paths: &[&str], cluster: &str) -> serde_yaml::Value {
        provider_config_with_demo_attribution(candidate_id, model, paths, cluster, Some(true))
    }

    fn provider_config_with_demo_attribution(
        candidate_id: &str,
        model: &str,
        paths: &[&str],
        cluster: &str,
        emit_demo_attribution: Option<bool>,
    ) -> serde_yaml::Value {
        let paths_yaml: Vec<serde_yaml::Value> = paths.iter().map(|p| serde_yaml::Value::from(*p)).collect();
        let mut config = serde_yaml::Mapping::from_iter([
            (
                serde_yaml::Value::from("provider_id"),
                serde_yaml::Value::from("test-provider"),
            ),
            (
                serde_yaml::Value::from("routes"),
                serde_yaml::Value::Sequence(vec![
                    serde_yaml::to_value(serde_yaml::Mapping::from_iter([
                        (
                            serde_yaml::Value::from("candidate_id"),
                            serde_yaml::Value::from(candidate_id),
                        ),
                        (serde_yaml::Value::from("model"), serde_yaml::Value::from(model)),
                        (
                            serde_yaml::Value::from("paths"),
                            serde_yaml::Value::Sequence(paths_yaml),
                        ),
                        (serde_yaml::Value::from("cluster"), serde_yaml::Value::from(cluster)),
                    ]))
                    .unwrap(),
                ]),
            ),
        ]);
        if let Some(enabled) = emit_demo_attribution {
            config.insert(
                serde_yaml::Value::from("emit_demo_attribution"),
                serde_yaml::Value::from(enabled),
            );
        }
        serde_yaml::to_value(config).unwrap()
    }

    fn make_filter(candidate_id: &str) -> Box<dyn HttpFilter> {
        let config = provider_config(candidate_id, "sim-model-v1", &["/v1/chat/completions"], "mock-backend");
        ProviderRouteFilter::from_config(&config).unwrap()
    }

    fn credential_provider_config() -> serde_yaml::Value {
        serde_yaml::from_str(
            "provider_id: test-provider\n\
             routes:\n\
             \x20 - candidate_id: abc123\n\
             \x20   cluster: api-backend\n\
             \x20   model: sim-model-v1\n\
             \x20   paths: [/v1/chat/completions]\n\
             \x20   credential:\n\
             \x20     strategy: bearer_token\n\
             \x20     secretRef:\n\
             \x20       name: provider-token\n\
             \x20       namespace: grid-system\n\
             \x20       key: token\n",
        )
        .unwrap()
    }

    fn request_with_peer_context(
        path: &str,
        candidate_id: &str,
        hop_request_id: &str,
        model: &str,
    ) -> praxis_filter::Request {
        let mut req = test_utils::make_request(Method::POST, path);
        req.headers.insert(
            HeaderName::from_static(SELECTED_CANDIDATE_HEADER),
            HeaderValue::from_str(candidate_id).unwrap(),
        );
        req.headers.insert(
            HeaderName::from_static(PROVIDER_HOP_REQUEST_ID_HEADER),
            HeaderValue::from_str(hop_request_id).unwrap(),
        );
        req.headers.insert(
            HeaderName::from_static(OVERLAY_REVISION_HEADER),
            HeaderValue::from_static("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        );
        req.headers.insert("X-Model", HeaderValue::from_str(model).unwrap());
        req
    }

    /// Read the final queued value for an overwritten request header.
    fn pending_header<'a>(ctx: &'a HttpFilterContext<'_>, name: &'static str) -> Option<&'a str> {
        ctx.request_headers_to_set
            .iter()
            .rev()
            .find(|(header, _)| header == name)
            .and_then(|(_, value)| value.to_str().ok())
    }
}
