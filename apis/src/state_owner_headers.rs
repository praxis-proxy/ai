// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Destination-bound header projection for trusted state ownership.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use http::{HeaderName, HeaderValue};
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, TrustedHeaderMutation, parse_filter_config,
};
use serde::Deserialize;

use crate::state_owner::{StateOwner, StateOwnerIngressHeaders, reject_owner};

/// Configuration for [`StateOwnerHeadersFilter`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateOwnerHeadersConfig {
    /// Header receiving the stable tenant namespace.
    tenant_header: String,
    /// Header receiving the stable subject identifier.
    subject_header: String,
    /// Optional header receiving the identity-provider or trust-domain identifier.
    #[serde(default)]
    issuer_header: Option<String>,
}

/// Projects a normalized [`StateOwner`] into destination-specific HTTP headers.
///
/// Place this filter only in a chain whose destination is authorized to receive
/// identity. The ingress `state_owner` filter removes its assertion headers;
/// this filter recreates configured headers from the validated, immutable
/// context, so client-supplied values cannot shadow the trusted projection. In
/// an IRR step it also strips the raw ingress names carried as bounded transport
/// metadata before the child request is dispatched.
///
/// # OGX YAML configuration
///
/// ```yaml
/// filter: state_owner_headers
/// tenant_header: x-tenant-id
/// subject_header: x-user-id
/// ```
///
/// A service that also consumes the trust domain can opt into an issuer header:
///
/// # Issuer YAML configuration
///
/// ```yaml
/// filter: state_owner_headers
/// tenant_header: x-service-tenant
/// subject_header: x-service-user
/// issuer_header: x-service-issuer
/// ```
pub struct StateOwnerHeadersFilter {
    /// Header receiving the stable tenant namespace.
    tenant_header: HeaderName,
    /// Header receiving the stable subject identifier.
    subject_header: HeaderName,
    /// Optional header receiving the trust domain.
    issuer_header: Option<HeaderName>,
}

impl StateOwnerHeadersFilter {
    /// Parse and validate destination header mappings.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when a name is empty, invalid, unsafe for an
    /// end-to-end identity assertion, or reused for multiple components.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: StateOwnerHeadersConfig = parse_filter_config("state_owner_headers", value)?;
        let tenant_header = parse_projection_header("tenant_header", &config.tenant_header)?;
        let subject_header = parse_projection_header("subject_header", &config.subject_header)?;
        let issuer_header = config
            .issuer_header
            .as_deref()
            .map(|header| parse_projection_header("issuer_header", header))
            .transpose()?;
        ensure_distinct_projection_headers(&tenant_header, &subject_header, issuer_header.as_ref())?;
        Ok(Box::new(Self {
            tenant_header,
            subject_header,
            issuer_header,
        }))
    }

    /// Queue destination-bound headers from the normalized owner context.
    fn project(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) -> FilterAction {
        let Some(owner) = ctx.extensions.get::<StateOwner>() else {
            return reject_owner(
                401,
                "missing_state_owner",
                "trusted state owner context is required before header projection",
            );
        };

        let tenant = match projection_value("tenant", owner.tenant_id()) {
            Ok(value) => value,
            Err(action) => return action,
        };
        let subject = match projection_value("subject", owner.subject()) {
            Ok(value) => value,
            Err(action) => return action,
        };
        let issuer = match &self.issuer_header {
            Some(_) => match projection_value("issuer", owner.issuer()) {
                Ok(value) => Some(value),
                Err(action) => return action,
            },
            None => None,
        };

        // IRR carries extensions into each step, but its initial request can
        // still contain the uncommitted ingress identity headers. Strip those
        // raw inputs in the destination chain before emitting only its
        // explicitly configured identity contract.
        // Always use the ordered body-phase log. If a later filter starts using
        // it, Core gives that log precedence over all grouped queues.
        let ordered = body_phase;
        strip_ingress_headers(ctx, ordered);
        queue_projection(ctx, &self.tenant_header, tenant, ordered);
        queue_projection(ctx, &self.subject_header, subject, ordered);
        if let (Some(header), Some(value)) = (&self.issuer_header, issuer) {
            queue_projection(ctx, header, value, ordered);
        }
        FilterAction::Continue
    }
}

/// Remove raw transport assertions before projecting destination identity.
fn strip_ingress_headers(ctx: &mut HttpFilterContext<'_>, ordered: bool) {
    let ingress_headers = ctx
        .extensions
        .get::<StateOwnerIngressHeaders>()
        .map(|headers| Arc::clone(&headers.0));
    if let Some(headers) = ingress_headers {
        ctx.request_headers_to_remove.extend(headers.iter().cloned());
        if ordered {
            ctx.pre_read_mutations
                .extend(headers.iter().cloned().map(TrustedHeaderMutation::Remove));
        }
    }
}

/// Parse an end-to-end header name used for destination identity projection.
fn parse_projection_header(field: &str, header: &str) -> Result<HeaderName, FilterError> {
    if header.is_empty() {
        return Err(format!("state_owner_headers: '{field}' must not be empty").into());
    }
    let header = HeaderName::from_bytes(header.as_bytes())
        .map_err(|error| -> FilterError { format!("state_owner_headers: invalid '{field}': {error}").into() })?;
    let name = header.as_str();
    if name == http::header::HOST.as_str()
        || name == http::header::CONTENT_LENGTH.as_str()
        || praxis_core::reserved_headers::HOP_BY_HOP_HEADERS.contains(&name)
        || praxis_core::reserved_headers::is_reserved(name)
    {
        return Err(format!("state_owner_headers: '{field}' must name a non-routing, end-to-end header").into());
    }
    Ok(header)
}

/// Require one distinct output header per projected owner component.
fn ensure_distinct_projection_headers(
    tenant: &HeaderName,
    subject: &HeaderName,
    issuer: Option<&HeaderName>,
) -> Result<(), FilterError> {
    let mut seen = std::collections::HashSet::with_capacity(3);
    for header in [Some(tenant), Some(subject), issuer].into_iter().flatten() {
        if !seen.insert(header) {
            return Err("state_owner_headers: output header names must be distinct".into());
        }
    }
    Ok(())
}

/// Convert one normalized owner component to an HTTP field value.
fn projection_value(component: &str, value: &str) -> Result<HeaderValue, FilterAction> {
    HeaderValue::from_str(value).map_err(|_error| {
        reject_owner(
            400,
            "invalid_state_owner",
            &format!("trusted state owner {component} cannot be represented as an HTTP header value"),
        )
    })
}

/// Queue an overwrite in the mutation channel for the active lifecycle phase.
fn queue_projection(ctx: &mut HttpFilterContext<'_>, header: &HeaderName, value: HeaderValue, ordered: bool) {
    ctx.request_headers_to_set.push((header.clone(), value.clone()));
    if ordered {
        ctx.pre_read_mutations
            .push(TrustedHeaderMutation::Set(header.clone(), value));
    }
}

#[async_trait]
impl HttpFilter for StateOwnerHeadersFilter {
    fn name(&self) -> &'static str {
        "state_owner_headers"
    }

    fn request_body_access(&self) -> BodyAccess {
        // Projection must be visible to destination-bound callouts that run
        // during StreamBuffer pre-read, not only to the primary upstream.
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(self.project(ctx, false))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(match self.project(ctx, true) {
            FilterAction::Continue => FilterAction::BodyDone,
            action => action,
        })
    }
}
