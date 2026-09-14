// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Trusted ownership context for OpenAI persisted state.
//!
//! The filter in this module decodes a provider-neutral, versioned owner
//! assertion installed by a trusted upstream authentication boundary. The
//! assertion is transport only: deployments must prevent untrusted clients
//! from supplying it or bypassing that boundary.

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used, reason = "tests")]
mod tests;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use bytes::Bytes;
use http::header::HeaderName;
use praxis_filter::{
    BodyAccess, FilterAction, FilterError, HttpFilter, HttpFilterContext, TrustedHeaderMutation, parse_filter_config,
};
use serde::Deserialize;

use super::responses::error::responses_error_rejection;

/// Maximum encoded owner assertion accepted from the trusted boundary.
const MAX_ASSERTION_BYTES: usize = 4_096;

/// Maximum UTF-8 bytes accepted in one owner component.
const MAX_COMPONENT_BYTES: usize = 1_024;

/// Supported assertion envelope version.
const ASSERTION_VERSION_PREFIX: &str = "v1.";

/// Immutable owner of OpenAI persisted state.
///
/// Tenant, issuer, and subject together form the ownership identity. A subject
/// alone is not globally unique and must never be used as the storage scope.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct OpenAiStateOwner {
    /// Stable tenant namespace.
    tenant_id: String,
    /// Stable identity-provider or trust-domain identifier.
    issuer: String,
    /// Stable subject identifier within `issuer`.
    subject: String,
}

impl OpenAiStateOwner {
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

/// Configuration for [`OpenAiStateOwnerFilter`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiStateOwnerConfig {
    /// Exact header name carrying the versioned assertion.
    header: String,
}

/// Decodes a trusted owner assertion into [`OpenAiStateOwner`].
///
/// The configured header must be governed by a trusted upstream boundary. This
/// filter validates and strips it; it does not authenticate its producer.
///
/// # YAML configuration
///
/// ```yaml
/// filter: openai_state_owner
/// header: x-praxis-state-owner
/// ```
pub struct OpenAiStateOwnerFilter {
    /// Validated assertion header name.
    header: HeaderName,
}

impl OpenAiStateOwnerFilter {
    /// Parse and validate filter configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when `header` is empty or is not a valid HTTP
    /// header name.
    pub fn from_config(value: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: OpenAiStateOwnerConfig = parse_filter_config("openai_state_owner", value)?;
        if config.header.is_empty() {
            return Err("openai_state_owner: 'header' must not be empty".into());
        }
        let header = HeaderName::from_bytes(config.header.as_bytes())
            .map_err(|e| format!("openai_state_owner: invalid 'header': {e}"))?;
        Ok(Box::new(Self { header }))
    }

    /// Resolve and install the owner, rejecting invalid or absent assertions.
    fn resolve(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) -> FilterAction {
        if ctx.extensions.get::<OpenAiStateOwner>().is_some() {
            return FilterAction::Continue;
        }

        self.queue_header_removal(ctx, body_phase);

        let mut values = ctx.request.headers.get_all(&self.header).iter();
        let Some(value) = values.next() else {
            return reject_owner(401, "missing_state_owner", "trusted state owner assertion is required");
        };
        if values.next().is_some() {
            return reject_owner(
                400,
                "invalid_state_owner",
                "trusted state owner assertion must appear exactly once",
            );
        }
        if value.as_bytes().len() > MAX_ASSERTION_BYTES {
            return reject_owner(400, "invalid_state_owner", "trusted state owner assertion is too large");
        }
        let Ok(value) = value.to_str() else {
            return reject_owner(400, "invalid_state_owner", "trusted state owner assertion must be text");
        };
        let owner = match decode_assertion(value) {
            Ok(owner) => owner,
            Err(error) => {
                tracing::debug!(reason = error.category(), "rejected trusted state owner assertion");
                return reject_owner(400, "invalid_state_owner", error.client_message());
            },
        };
        ctx.extensions.insert(owner);
        FilterAction::Continue
    }

    /// Queue removal using the mutation channel appropriate for the lifecycle.
    fn queue_header_removal(&self, ctx: &mut HttpFilterContext<'_>, body_phase: bool) {
        if body_phase {
            ctx.pre_read_mutations
                .push(TrustedHeaderMutation::Remove(self.header.clone()));
        } else {
            ctx.request_headers_to_remove.push(self.header.clone());
        }
    }
}

#[async_trait]
impl HttpFilter for OpenAiStateOwnerFilter {
    fn name(&self) -> &'static str {
        "openai_state_owner"
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
fn decode_assertion(value: &str) -> Result<OpenAiStateOwner, OwnerAssertionError> {
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
    OpenAiStateOwner::from_trusted_parts(tenant_id, issuer, subject)
}

/// Validate one stable owner component.
fn validate_component(name: &'static str, value: &str) -> Result<(), OwnerAssertionError> {
    if value.is_empty() || value.len() > MAX_COMPONENT_BYTES || value.chars().any(char::is_control) {
        return Err(OwnerAssertionError::InvalidComponent(name));
    }
    Ok(())
}

/// Build a bounded OpenAI-compatible ownership rejection.
fn reject_owner(status: u16, code: &str, message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(status, code, message))
}
