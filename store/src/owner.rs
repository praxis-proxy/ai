// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Trusted ownership identity for persisted private state.
//!
//! [`StateOwner`] is the tenant-qualified owner of every persisted record.
//! [`StateOwner::from_trusted_parts`] is the only constructor, and it takes
//! already-normalized parts. The request-to-owner projection stays in the
//! transport layer (apis), so a contracts consumer cannot mint an owner from
//! request-controlled input.

use std::{fmt, sync::Arc};

/// Maximum UTF-8 bytes accepted in one owner component.
const MAX_COMPONENT_BYTES: usize = 1_024;

/// Immutable tenant-qualified owner of persisted private state.
///
/// Tenant, issuer, and subject together form the ownership identity. The
/// reference-counted components make passing the owner through records and
/// request-scoped store handles cheap without copying identity strings.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct StateOwner {
    /// Stable tenant namespace.
    tenant_id: Arc<str>,
    /// Stable identity-provider or trust-domain identifier.
    issuer: Arc<str>,
    /// Stable principal identifier within the issuer.
    subject: Arc<str>,
}

impl StateOwner {
    /// Construct an owner from trusted, normalized identity parts.
    ///
    /// This is the only constructor. The request-to-owner projection lives in
    /// the transport layer, so the contracts crate cannot be handed a
    /// request-forged owner.
    ///
    /// # Errors
    ///
    /// Returns [`StateOwnerError`] when a component is empty, oversized, or
    /// contains a control character.
    pub fn from_trusted_parts(
        tenant_id: impl Into<Arc<str>>,
        issuer: impl Into<Arc<str>>,
        subject: impl Into<Arc<str>>,
    ) -> Result<Self, StateOwnerError> {
        let owner = Self {
            tenant_id: tenant_id.into(),
            issuer: issuer.into(),
            subject: subject.into(),
        };
        validate_component("tenant", &owner.tenant_id)?;
        validate_component("issuer", &owner.issuer)?;
        validate_component("subject", &owner.subject)?;
        Ok(owner)
    }

    /// Stable tenant namespace.
    #[must_use]
    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    /// Stable identity-provider or trust-domain identifier.
    #[must_use]
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// Stable subject identifier within [`Self::issuer`].
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }
}

/// Invalid stable owner component.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateOwnerError {
    /// Stable name of the component that failed validation.
    component: &'static str,
}

impl StateOwnerError {
    /// Stable name of the invalid component.
    #[must_use]
    pub fn component(self) -> &'static str {
        self.component
    }
}

impl fmt::Display for StateOwnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid state owner {}", self.component)
    }
}

impl std::error::Error for StateOwnerError {}

/// Validate one bounded, nonempty owner component.
///
/// Exposed so the transport layer can validate an individual owner component
/// (tenant, issuer, or subject) as it parses trusted headers, before it has all
/// three to call [`StateOwner::from_trusted_parts`].
///
/// # Errors
///
/// Returns [`StateOwnerError`] when the value is empty, oversized, or contains a
/// control character.
pub fn validate_component(component: &'static str, value: &str) -> Result<(), StateOwnerError> {
    if value.is_empty() || value.len() > MAX_COMPONENT_BYTES || value.chars().any(char::is_control) {
        return Err(StateOwnerError { component });
    }
    Ok(())
}
