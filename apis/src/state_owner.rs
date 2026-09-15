// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Trusted ownership context for persisted private state.
//!
//! The filter in this module normalizes identity supplied by a trusted upstream
//! authentication boundary. Deployments must prevent untrusted clients from
//! supplying configured identity headers or bypassing that boundary.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used, reason = "tests")]
mod tests;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::header::HeaderName;
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection, TrustedHeaderMutation,
    parse_filter_config,
};
use serde::Deserialize;

/// Maximum encoded owner assertion accepted from the trusted boundary.
const MAX_ASSERTION_BYTES: usize = 4_096;

/// Maximum UTF-8 bytes accepted in one owner component.
const MAX_COMPONENT_BYTES: usize = 1_024;

/// Supported assertion envelope version.
const ASSERTION_VERSION_PREFIX: &str = "v1.";

/// Stable issuer used by the explicit shared-owner compatibility mode.
const SINGLE_TENANT_ISSUER: &str = "urn:praxis:single-tenant";

/// Stable subject used by the explicit shared-owner compatibility mode.
const SINGLE_TENANT_SUBJECT: &str = "shared";

/// Immutable owner of persisted private state.
///
/// Tenant, issuer, and subject together form the ownership identity. A subject
/// alone is not globally unique and must never be used as the storage scope.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct StateOwner {
    /// Stable tenant namespace.
    tenant_id: String,
    /// Stable identity-provider or trust-domain identifier.
    issuer: String,
    /// Stable subject identifier within `issuer`.
    subject: String,
}

impl StateOwner {
    /// Stable tenant namespace.
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Stable identity-provider or trust-domain identifier.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// Stable subject identifier within [`Self::issuer`].
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// Construct an owner from trusted, normalized identity parts.
    ///
    /// This is crate-visible so a future native Core identity adapter can
    /// populate the same context without serializing identity through a header.
    pub(crate) fn from_trusted_parts(
        tenant_id: String,
        issuer: String,
        subject: String,
    ) -> Result<Self, OwnerAssertionError> {
        validate_component("tenant", &tenant_id)?;
        validate_component("issuer", &issuer)?;
        validate_component("subject", &subject)?;
        Ok(Self {
            tenant_id,
            issuer,
            subject,
        })
    }
}

/// Configuration for [`StateOwnerFilter`].
#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum StateOwnerConfig {
    /// Explicit compatibility mode sharing one owner within a tenant namespace.
    SingleTenant {
        /// Stable namespace assigned to every request in this pipeline.
        tenant_id: String,
    },
    /// Strict owner isolation sourced from a trusted assertion header.
    TrustedOwner {
        /// Exact header name carrying the versioned assertion.
        header: String,
    },
    /// Owner identity assembled from independently configured trusted sources.
    TrustedHeaders {
        /// Source of the stable tenant namespace.
        tenant: OwnerComponentConfig,
        /// Source of the stable identity-provider or trust-domain identifier.
        issuer: OwnerComponentConfig,
        /// Source of the stable subject identifier within the issuer.
        subject: OwnerComponentConfig,
    },
    /// Reserved for the later PPE-backed authorization integration.
    Policy {
        /// Exact header name that will carry the current caller assertion.
        header: String,
    },
}

/// One configured source for an owner component.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OwnerComponentConfig {
    /// Read the component from an exact trusted header name.
    Header(OwnerHeaderComponentConfig),
    /// Use the same configured value for every request.
    Static(OwnerStaticComponentConfig),
}

/// Header-backed owner component configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnerHeaderComponentConfig {
    /// Exact header name containing the component value.
    header: String,
}

/// Static owner component configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnerStaticComponentConfig {
    /// Stable component value assigned to every request.
    #[serde(rename = "static")]
    value: String,
}

/// Configured source of the normalized owner context.
enum OwnerSource {
    /// One shared owner for an explicitly configured tenant namespace.
    Static(StateOwner),
    /// Versioned assertion from a trusted upstream boundary.
    TrustedHeader(HeaderName),
    /// Owner assembled from independently configured trusted component sources.
    TrustedComponents {
        /// Stable tenant namespace source.
        tenant: OwnerComponentSource,
        /// Stable issuer source.
        issuer: OwnerComponentSource,
        /// Stable subject source.
        subject: OwnerComponentSource,
    },
}

/// Validated runtime source for one owner component.
enum OwnerComponentSource {
    /// Read the value from a trusted request header.
    Header(HeaderName),
    /// Reuse a configured stable value.
    Static(String),
}

impl OwnerComponentSource {
    /// Validate one configured component source.
    fn from_config(component: &'static str, config: OwnerComponentConfig) -> Result<Self, FilterError> {
        match config {
            OwnerComponentConfig::Header(config) => {
                let field = format!("{component}.header");
                Ok(Self::Header(parse_header_name(&field, &config.header)?))
            },
            OwnerComponentConfig::Static(config) => {
                validate_component(component, &config.value)
                    .map_err(|error| format!("state_owner: {}", error.client_message()))?;
                Ok(Self::Static(config.value))
            },
        }
    }

    /// Return the configured header, when this component is header-backed.
    fn header(&self) -> Option<&HeaderName> {
        match self {
            Self::Header(header) => Some(header),
            Self::Static(_) => None,
        }
    }
}

/// Reject configurations that assign one header to multiple owner components.
fn ensure_distinct_component_headers(sources: [&OwnerComponentSource; 3]) -> Result<(), FilterError> {
    let mut seen = std::collections::HashSet::with_capacity(sources.len());
    for header in sources.into_iter().filter_map(OwnerComponentSource::header) {
        if !seen.insert(header) {
            return Err("state_owner: trusted_headers component headers must be distinct".into());
        }
    }
    Ok(())
}

/// Establishes a normalized [`StateOwner`] from trusted identity sources.
///
/// Every configured header must be governed by a trusted upstream boundary.
/// This filter validates and strips consumed headers; it does not authenticate
/// their producer. In `trusted_headers` mode each component must select exactly
/// one `header` or `static` source, and component header names must be distinct.
///
/// # Versioned-assertion YAML configuration
///
/// ```yaml
/// filter: state_owner
/// mode: trusted_owner
/// header: x-authenticated-state-owner
/// ```
///
/// Existing gateways can map separate headers and static values into the same
/// normalized owner:
///
/// # Mapped-header YAML configuration
///
/// ```yaml
/// filter: state_owner
/// mode: trusted_headers
/// tenant:
///   header: x-maas-tenant
/// issuer:
///   static: https://authorino.example
/// subject:
///   header: x-maas-user
/// ```
///
/// Explicit single-tenant compatibility mode does not consume a header:
///
/// # Single-tenant YAML configuration
///
/// ```yaml
/// filter: state_owner
/// mode: single_tenant
/// tenant_id: local
/// ```
pub struct StateOwnerFilter {
    /// Validated source used to populate the request context.
    source: OwnerSource,
}

impl StateOwnerFilter {
    /// Parse and validate filter configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the selected mode is incomplete or invalid.
    /// `policy` is intentionally rejected until the PPE integration is present,
    /// so selecting it can never silently degrade to owner-only authorization.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: StateOwnerConfig = parse_filter_config("state_owner", value)?;
        let source = match config {
            StateOwnerConfig::SingleTenant { tenant_id } => single_tenant_source(tenant_id)?,
            StateOwnerConfig::TrustedOwner { header } => {
                OwnerSource::TrustedHeader(parse_header_name("header", &header)?)
            },
            StateOwnerConfig::TrustedHeaders {
                tenant,
                issuer,
                subject,
            } => trusted_components_source(tenant, issuer, subject)?,
            StateOwnerConfig::Policy { header } => {
                // Validate the complete adapter config before reporting the
                // reserved mode, so a future implementation inherits the same
                // strict header contract.
                drop(parse_header_name("header", &header)?);
                return Err("state_owner: 'policy' mode requires the PPE integration".into());
            },
        };
        Ok(Box::new(Self { source }))
    }

    /// Resolve and install the owner, rejecting invalid or absent assertions.
    fn resolve(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) -> FilterAction {
        self.queue_header_removal(ctx, body_phase);

        if ctx.extensions.get::<StateOwner>().is_some() {
            return FilterAction::Continue;
        }

        let owner = match &self.source {
            OwnerSource::Static(owner) => owner.clone(),
            OwnerSource::TrustedHeader(header) => match resolve_trusted_header(ctx, header) {
                Ok(owner) => owner,
                Err(action) => return action,
            },
            OwnerSource::TrustedComponents {
                tenant,
                issuer,
                subject,
            } => match resolve_trusted_components(ctx, tenant, issuer, subject) {
                Ok(owner) => owner,
                Err(action) => return action,
            },
        };
        ctx.extensions.insert(owner);
        FilterAction::Continue
    }

    /// Queue removal using the mutation channel appropriate for the lifecycle.
    fn queue_header_removal(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) {
        match &self.source {
            OwnerSource::Static(_) => {},
            OwnerSource::TrustedHeader(header) => queue_one_header_removal(ctx, header, body_phase),
            OwnerSource::TrustedComponents {
                tenant,
                issuer,
                subject,
            } => {
                for header in [tenant, issuer, subject]
                    .into_iter()
                    .filter_map(OwnerComponentSource::header)
                {
                    queue_one_header_removal(ctx, header, body_phase);
                }
            },
        }
    }
}

/// Build the explicit shared owner used by single-tenant deployments.
fn single_tenant_source(tenant_id: String) -> Result<OwnerSource, FilterError> {
    let owner = StateOwner::from_trusted_parts(
        tenant_id,
        SINGLE_TENANT_ISSUER.to_owned(),
        SINGLE_TENANT_SUBJECT.to_owned(),
    )
    .map_err(|error| format!("state_owner: {}", error.client_message()))?;
    Ok(OwnerSource::Static(owner))
}

/// Validate and build independently mapped owner component sources.
fn trusted_components_source(
    tenant: OwnerComponentConfig,
    issuer: OwnerComponentConfig,
    subject: OwnerComponentConfig,
) -> Result<OwnerSource, FilterError> {
    let tenant = OwnerComponentSource::from_config("tenant", tenant)?;
    let issuer = OwnerComponentSource::from_config("issuer", issuer)?;
    let subject = OwnerComponentSource::from_config("subject", subject)?;
    ensure_distinct_component_headers([&tenant, &issuer, &subject])?;
    Ok(OwnerSource::TrustedComponents {
        tenant,
        issuer,
        subject,
    })
}

/// Queue removal of one consumed identity header.
fn queue_one_header_removal(ctx: &mut HttpFilterContext<'_>, header: &HeaderName, body_phase: bool) {
    if body_phase {
        ctx.pre_read_mutations
            .push(TrustedHeaderMutation::Remove(header.clone()));
    } else {
        ctx.request_headers_to_remove.push(header.clone());
    }
}

/// Parse the trusted assertion header into a complete owner.
fn resolve_trusted_header(ctx: &HttpFilterContext<'_>, header: &HeaderName) -> Result<StateOwner, FilterAction> {
    let value = exactly_one_header_value(ctx, header)?;
    if value.as_bytes().len() > MAX_ASSERTION_BYTES {
        return Err(reject_owner(
            400,
            "invalid_state_owner",
            "trusted state owner assertion is too large",
        ));
    }
    let Ok(value) = value.to_str() else {
        return Err(reject_owner(
            400,
            "invalid_state_owner",
            "trusted state owner assertion must be text",
        ));
    };
    match decode_assertion(value) {
        Ok(owner) => Ok(owner),
        Err(error) => {
            tracing::debug!(reason = error.category(), "rejected trusted state owner assertion");
            Err(reject_owner(400, "invalid_state_owner", error.client_message()))
        },
    }
}

/// Assemble one normalized owner from independently configured sources.
fn resolve_trusted_components(
    ctx: &HttpFilterContext<'_>,
    tenant: &OwnerComponentSource,
    issuer: &OwnerComponentSource,
    subject: &OwnerComponentSource,
) -> Result<StateOwner, FilterAction> {
    // The request extension owns its identity for the complete filter
    // lifecycle, so the three bounded component values must cross this
    // boundary as owned strings.
    let tenant_id = resolve_owner_component(ctx, "tenant", tenant)?;
    let issuer = resolve_owner_component(ctx, "issuer", issuer)?;
    let subject = resolve_owner_component(ctx, "subject", subject)?;
    StateOwner::from_trusted_parts(tenant_id, issuer, subject)
        .map_err(|error| reject_owner(400, "invalid_state_owner", error.client_message()))
}

/// Resolve and validate one mapped owner component.
fn resolve_owner_component(
    ctx: &HttpFilterContext<'_>,
    component: &'static str,
    source: &OwnerComponentSource,
) -> Result<String, FilterAction> {
    let header = match source {
        OwnerComponentSource::Header(header) => header,
        OwnerComponentSource::Static(value) => return Ok(value.clone()),
    };

    let value = exactly_one_component_header_value(ctx, component, header)?;
    let Ok(value) = value.to_str() else {
        return Err(reject_owner(
            400,
            "invalid_state_owner",
            &format!("trusted state owner {component} header must be text"),
        ));
    };
    validate_component(component, value)
        .map_err(|error| reject_owner(400, "invalid_state_owner", error.client_message()))?;
    Ok(value.to_owned())
}

/// Require exactly one value for a mapped component header.
fn exactly_one_component_header_value<'a>(
    ctx: &'a HttpFilterContext<'_>,
    component: &'static str,
    header: &HeaderName,
) -> Result<&'a http::HeaderValue, FilterAction> {
    let mut values = ctx.request.headers.get_all(header).iter();
    let Some(value) = values.next() else {
        return Err(reject_owner(
            401,
            "missing_state_owner",
            &format!("trusted state owner {component} header is required"),
        ));
    };
    if values.next().is_some() {
        return Err(reject_owner(
            400,
            "invalid_state_owner",
            &format!("trusted state owner {component} header must appear exactly once"),
        ));
    }
    Ok(value)
}

/// Require exactly one value for the configured assertion header.
fn exactly_one_header_value<'a>(
    ctx: &'a HttpFilterContext<'_>,
    header: &HeaderName,
) -> Result<&'a http::HeaderValue, FilterAction> {
    let mut values = ctx.request.headers.get_all(header).iter();
    let Some(value) = values.next() else {
        return Err(reject_owner(
            401,
            "missing_state_owner",
            "trusted state owner assertion is required",
        ));
    };
    if values.next().is_some() {
        return Err(reject_owner(
            400,
            "invalid_state_owner",
            "trusted state owner assertion must appear exactly once",
        ));
    }
    Ok(value)
}

/// Parse a nonempty trusted header name.
fn parse_header_name(field: &str, header: &str) -> Result<HeaderName, FilterError> {
    if header.is_empty() {
        return Err(format!("state_owner: '{field}' must not be empty").into());
    }
    HeaderName::from_bytes(header.as_bytes()).map_err(|e| format!("state_owner: invalid '{field}': {e}").into())
}

#[async_trait]
impl HttpFilter for StateOwnerFilter {
    fn name(&self) -> &'static str {
        "state_owner"
    }

    fn request_body_access(&self) -> BodyAccess {
        // This filter does not inspect body bytes. ReadOnly participation makes
        // it run before downstream StreamBuffer consumers during pre-read.
        BodyAccess::ReadOnly
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(self.resolve(ctx, false))
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        _body: &mut Option<Bytes>,
        _end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        Ok(match self.resolve(ctx, true) {
            FilterAction::Continue => FilterAction::BodyDone,
            action => action,
        })
    }
}

/// Validation failure for a versioned owner assertion.
#[derive(Debug)]
pub(crate) enum OwnerAssertionError {
    /// Version prefix is missing or unsupported.
    UnsupportedVersion,
    /// Payload is not canonical unpadded base64url.
    InvalidEncoding,
    /// Decoded payload is not exactly three JSON strings.
    InvalidPayload,
    /// One identity component is absent, too large, or contains controls.
    InvalidComponent(&'static str),
}

impl OwnerAssertionError {
    /// Stable low-cardinality category for internal diagnostics.
    fn category(&self) -> &'static str {
        match self {
            Self::UnsupportedVersion => "unsupported_version",
            Self::InvalidEncoding => "invalid_encoding",
            Self::InvalidPayload => "invalid_payload",
            Self::InvalidComponent(_) => "invalid_component",
        }
    }

    /// Bounded client-facing explanation without assertion contents.
    fn client_message(&self) -> &'static str {
        match self {
            Self::UnsupportedVersion => "trusted state owner assertion version is unsupported",
            Self::InvalidEncoding => "trusted state owner assertion encoding is invalid",
            Self::InvalidPayload => "trusted state owner assertion payload is invalid",
            Self::InvalidComponent("tenant") => "trusted state owner tenant is invalid",
            Self::InvalidComponent("issuer") => "trusted state owner issuer is invalid",
            Self::InvalidComponent("subject") => "trusted state owner subject is invalid",
            Self::InvalidComponent(_) => "trusted state owner component is invalid",
        }
    }
}

/// Decode the provider-neutral `v1` assertion envelope.
fn decode_assertion(value: &str) -> Result<StateOwner, OwnerAssertionError> {
    let payload = value
        .strip_prefix(ASSERTION_VERSION_PREFIX)
        .ok_or(OwnerAssertionError::UnsupportedVersion)?;
    if payload.is_empty() {
        return Err(OwnerAssertionError::InvalidEncoding);
    }
    let decoded = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_error| OwnerAssertionError::InvalidEncoding)?;
    let [tenant_id, issuer, subject]: [String; 3] =
        serde_json::from_slice(&decoded).map_err(|_error| OwnerAssertionError::InvalidPayload)?;
    StateOwner::from_trusted_parts(tenant_id, issuer, subject)
}

/// Validate one stable owner component.
fn validate_component(name: &'static str, value: &str) -> Result<(), OwnerAssertionError> {
    if value.is_empty() || value.len() > MAX_COMPONENT_BYTES || value.chars().any(char::is_control) {
        return Err(OwnerAssertionError::InvalidComponent(name));
    }
    Ok(())
}

/// Build a bounded rejection compatible with OpenAI and Anthropic clients.
pub(crate) fn reject_owner(status: u16, code: &str, message: &str) -> FilterAction {
    let error_type = if status == 401 {
        "authentication_error"
    } else {
        "invalid_request_error"
    };
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "code": code,
            "message": message,
        }
    });
    FilterAction::Reject(
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(serde_json::to_vec(&body).unwrap_or_default()),
    )
}
