// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared operation metadata and generated `OpenAPI` support.
//!
//! The runtime half ([`OpenAiOperationSpec`]) is always compiled so the
//! `openai_operation` classifier can identify requests. The `OpenAPI` contract
//! half lives in the `openapi` submodule and depends on `utoipa`.

mod openapi;

pub(crate) use openapi::{
    MediaTypeSpec, OwnedOperationContract, ParameterLocation, ParameterSpec, RequestBodySpec, ResponseSpec,
    SchemaBinding, implementation_openapi, schema_binding,
};

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
    /// Contract ownership is `OpenAPI` generation policy rather than runtime
    /// identity, so it lives on the OpenAI wrapper beside the contract it
    /// governs rather than on the shared handling mode.
    #[must_use]
    pub const fn owns_contract(&self) -> bool {
        matches!(self.runtime.mode, HandlingMode::Transform | HandlingMode::Local)
    }

    /// Whether this operation consumes a request body.
    ///
    /// Answers the runtime question directly rather than inferring it from
    /// contract ownership, so proxied operations report their real body shape.
    pub(crate) const fn has_request_body(&self) -> bool {
        self.runtime.has_request_body()
    }

    /// Return the locally owned `OpenAPI` contract, when applicable.
    pub(crate) const fn owned_contract(&self) -> Option<OwnedOperationContract> {
        self.owned_contract
    }
}

impl OperationEntry for OpenAiOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.runtime
    }
}
