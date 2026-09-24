// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared request-body lifecycle configuration for filters that can run either
//! before request headers or after logical upstream binding.

use serde::Deserialize;

/// Request-body lifecycle used by provider-gateable filters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestBodyPhase {
    /// Preserve the original pre-read lifecycle for standalone and legacy
    /// pipelines that do not bind a logical upstream first.
    #[default]
    PreRead,

    /// Run once after an earlier router has bound the logical upstream.
    BoundUpstream,
}

impl RequestBodyPhase {
    /// Return body access for the ordinary pre-read lifecycle.
    pub(crate) const fn pre_read_access(self, access: praxis_filter::BodyAccess) -> praxis_filter::BodyAccess {
        match self {
            Self::PreRead => access,
            Self::BoundUpstream => praxis_filter::BodyAccess::None,
        }
    }

    /// Return body access for the post-binding lifecycle.
    pub(crate) const fn bound_upstream_access(self, access: praxis_filter::BodyAccess) -> praxis_filter::BodyAccess {
        match self {
            Self::PreRead => praxis_filter::BodyAccess::None,
            Self::BoundUpstream => access,
        }
    }
}
