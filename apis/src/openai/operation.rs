// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared operation metadata and generated `OpenAPI` support.
//!
//! The runtime half ([`OpenAiOperationSpec`]) is always compiled so the
//! `ai_operation` classifier can identify requests. The `OpenAPI` contract
//! half lives in the `openapi` submodule and depends on `utoipa`.

#[cfg(any(
    feature = "openai-conversations",
    feature = "openai-files",
    feature = "openai-responses-openapi"
))]
mod openapi;

/// Shared conformance test helpers.
#[cfg(test)]
#[cfg(any(
    all(
        feature = "openai-conversations",
        feature = "_store-postgres",
        feature = "store-sqlite"
    ),
    feature = "openai-files"
))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
pub(crate) mod conformance;

#[cfg(feature = "openai-conversations")]
pub(crate) use openapi::ResponseHeaderSpec;
#[cfg(all(test, any(feature = "openai-conversations", feature = "openai-files")))]
pub(crate) use openapi::collect_component_references;
#[cfg(any(
    feature = "openai-conversations",
    feature = "openai-files",
    feature = "openai-responses-openapi"
))]
pub(crate) use openapi::{
    MediaTypeSpec, OwnedOperationContract, RequestBodySpec, ResponseSpec, SchemaBinding, implementation_openapi,
    schema_binding,
};
#[cfg(any(feature = "openai-conversations", feature = "openai-files"))]
pub(crate) use openapi::{ParameterLocation, ParameterSpec, path_parameter, query_parameter};

use crate::operation::{
    ApplicationProtocol, HandlingMode, HttpMethod, OperationEntry, OperationSpec, RequestBody, Transport,
};

/// One OpenAI operation: shared runtime identity plus OpenAI-owned contract.
///
/// The runtime half is the crate's shared `OperationSpec`, which the matcher understands
/// and every protocol registry uses. The remaining fields are OpenAI's own:
/// where the operation appears in the pinned OpenAI specification, and the
/// contract Praxis generates into its implementation `OpenAPI` document. Keeping
/// them here is what lets the matcher stay provider-neutral while `OpenAPI`
/// generation stays provider-owned.
#[derive(Clone, Copy)]
pub struct OpenAiOperationSpec {
    /// Runtime identity shared with every other protocol registry.
    pub(crate) runtime: OperationSpec,
    /// Path as it appears in the OpenAI spec, without `/v1`.
    pub spec_path: &'static str,
    /// Contract generated into the implementation `OpenAPI` document.
    #[cfg(any(
        feature = "openai-conversations",
        feature = "openai-files",
        feature = "openai-responses-openapi"
    ))]
    pub(crate) owned_contract: Option<OwnedOperationContract>,
}

impl OpenAiOperationSpec {
    /// Stable `OpenAPI` operation ID.
    #[must_use]
    pub const fn operation_id(&self) -> &'static str {
        self.runtime.operation_id
    }

    /// Typed HTTP method.
    #[must_use]
    pub const fn method(&self) -> HttpMethod {
        self.runtime.method
    }

    /// Application protocol that owns the operation.
    #[must_use]
    pub const fn application_protocol(&self) -> ApplicationProtocol {
        self.runtime.application_protocol
    }

    /// Transport the operation is reached over.
    #[must_use]
    pub const fn transport(&self) -> Transport {
        self.runtime.transport
    }

    /// Runtime path template handled by Praxis.
    #[must_use]
    pub const fn runtime_path(&self) -> &'static str {
        self.runtime.runtime_path
    }

    /// Runtime request-body shape.
    #[must_use]
    pub const fn request_body(&self) -> RequestBody {
        self.runtime.request_body
    }

    /// Proxy handling mode.
    #[must_use]
    pub const fn mode(&self) -> HandlingMode {
        self.runtime.mode
    }

    /// Whether Praxis owns this operation's externally visible contract.
    ///
    /// True for `Transform` and `Local` operations. It describes ownership
    /// only: `OpenAPI` generation is gated on a declared
    /// `owned_contract`, not on this flag. It lives on
    /// the OpenAI wrapper rather than on the shared handling mode.
    #[must_use]
    pub const fn owns_contract(&self) -> bool {
        matches!(self.runtime.mode, HandlingMode::Transform | HandlingMode::Local)
    }

    /// Whether the operation declares an `OpenAPI` contract.
    ///
    /// Generation and conformance comparison are gated on this, not on
    /// [`Self::owns_contract`]: a passthrough operation may still declare the
    /// contract it will own once local handlers land. Always `false` when no
    /// `OpenAPI` feature is compiled.
    #[must_use]
    pub const fn declares_contract(&self) -> bool {
        #[cfg(any(
            feature = "openai-conversations",
            feature = "openai-files",
            feature = "openai-responses-openapi"
        ))]
        {
            self.owned_contract.is_some()
        }
        #[cfg(not(any(
            feature = "openai-conversations",
            feature = "openai-files",
            feature = "openai-responses-openapi"
        )))]
        {
            let _ = self;
            false
        }
    }

    /// Return the locally owned `OpenAPI` contract, when applicable.
    #[cfg(any(
        feature = "openai-conversations",
        feature = "openai-files",
        feature = "openai-responses-openapi"
    ))]
    pub(crate) const fn owned_contract(&self) -> Option<OwnedOperationContract> {
        self.owned_contract
    }
}

impl OperationEntry for OpenAiOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.runtime
    }
}
