// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! AWS Bedrock runtime operation registry.
//!
//! Operation identity comes from the request head — method, path, and transport
//! — rather than from body heuristics, so a request is recognized before any
//! payload is read.
//!
//! # Source of truth
//!
//! The inventory below follows the AWS Bedrock runtime API
//! (`bedrock-runtime.{region}.amazonaws.com`), which serves model operations
//! under `/model/{model_id}/{action}`. AWS publishes this API as an RPC
//! service model rather than an `OpenAPI` document, so there is no pinned spec
//! to drift-check against; the operation IDs are the AWS API names, reproduced
//! verbatim.
//!
//! # Handling modes
//!
//! Every entry is [`HandlingMode::Passthrough`]: no Bedrock filter deliberately
//! targets these heads today. The registry exists so classifier consumers,
//! such as `token_count` in auto mode, can identify the dialect from the
//! published match instead of matching paths themselves. That includes the
//! `EventStream` actions: they classify, and it is the consumer's decision
//! that an `EventStream` body has no usage to extract.

use crate::operation::{
    ApplicationProtocol, HandlingMode, HttpMethod, OperationEntry, OperationSpec, RequestBody, RouteParams, Transport,
    match_operation,
};

/// Application protocol these operations belong to.
///
/// Declared beside the registry that owns it, so registering a protocol
/// never edits a shared list.
pub(crate) const APPLICATION_PROTOCOL: ApplicationProtocol = ApplicationProtocol::new("amazon_bedrock");

/// Static metadata for one Bedrock operation.
///
/// Holds the crate's shared `OperationSpec` directly: Praxis generates no
/// `OpenAPI` contract for Bedrock, so there is no provider-owned metadata to
/// carry beside the runtime identity.
#[derive(Clone, Copy)]
pub struct BedrockOperationSpec {
    /// Runtime operation.
    pub operation: BedrockOperation,
    /// Shared operation metadata.
    ///
    /// Crate-private because the shared `OperationSpec` is: external callers read this
    /// entry through the accessors below rather than the shared struct.
    pub(crate) definition: OperationSpec,
}

impl OperationEntry for BedrockOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.definition
    }
}

impl BedrockOperationSpec {
    /// Stable operation ID, as spelled in the AWS API.
    #[must_use]
    pub const fn operation_id(&self) -> &'static str {
        self.definition.operation_id
    }

    /// Typed HTTP method.
    #[must_use]
    pub const fn method(&self) -> HttpMethod {
        self.definition.method
    }

    /// Application protocol that owns the operation.
    #[must_use]
    pub const fn application_protocol(&self) -> ApplicationProtocol {
        self.definition.application_protocol
    }

    /// Transport the operation is reached over.
    #[must_use]
    pub const fn transport(&self) -> Transport {
        self.definition.transport
    }

    /// Runtime path template handled by Praxis.
    #[must_use]
    pub const fn runtime_path(&self) -> &'static str {
        self.definition.runtime_path
    }

    /// Runtime request-body shape.
    #[must_use]
    pub const fn request_body(&self) -> RequestBody {
        self.definition.request_body
    }

    /// Proxy handling mode.
    #[must_use]
    pub const fn mode(&self) -> HandlingMode {
        self.definition.mode
    }
}

/// Convert a registry body declaration into a runtime request-body shape.
macro_rules! request_body_shape {
    ([required json]) => {
        RequestBody::Json { required: true }
    };
}

/// Declare each Bedrock operation once and derive its metadata.
macro_rules! bedrock_operations {
    (
        $(
            $operation:ident {
                operation_id: $operation_id:literal,
                method: $method:ident,
                transport: $transport:ident,
                path: $path:literal,
                mode: $mode:ident,
                body: $body:tt $(,)?
            }
        ),+ $(,)?
    ) => {
        /// One Bedrock operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum BedrockOperation {
            $(
                #[doc = concat!(stringify!($method), " ", $path)]
                $operation,
            )+
        }

        /// All Bedrock operations recognized by Praxis.
        pub const OPERATION_SPECS: &[BedrockOperationSpec] = &[
            $(
                BedrockOperationSpec {
                    operation: BedrockOperation::$operation,
                    definition: OperationSpec {
                        application_protocol: APPLICATION_PROTOCOL,
                        operation_id: $operation_id,
                        method: HttpMethod::$method,
                        transport: Transport::$transport,
                        runtime_path: $path,
                        mode: HandlingMode::$mode,
                        request_body: request_body_shape!($body),
                    },
                },
            )+
        ];
    };
}

bedrock_operations! {
    InvokeModel {
        operation_id: "InvokeModel",
        method: Post,
        transport: Http,
        path: "/model/{model_id}/invoke",
        mode: Passthrough,
        body: [required json],
    },
    InvokeModelWithResponseStream {
        operation_id: "InvokeModelWithResponseStream",
        method: Post,
        transport: Http,
        path: "/model/{model_id}/invoke-with-response-stream",
        mode: Passthrough,
        body: [required json],
    },
    Converse {
        operation_id: "Converse",
        method: Post,
        transport: Http,
        path: "/model/{model_id}/converse",
        mode: Passthrough,
        body: [required json],
    },
    ConverseStream {
        operation_id: "ConverseStream",
        method: Post,
        transport: Http,
        path: "/model/{model_id}/converse-stream",
        mode: Passthrough,
        body: [required json],
    },
}

/// One matched Bedrock route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedBedrockRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static BedrockOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    pub(crate) params: RouteParams<'a>,
}

/// Return all Bedrock operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [BedrockOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Bedrock operation.
///
/// Bedrock is reached over plain HTTP only. `SigV4` authentication is contract
/// validation left to request processing and the backend: method and path
/// already identify the operation, so credentials must not change which
/// operation is selected.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedBedrockRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, Transport::Http).map(|matched| MatchedBedrockRoute {
        spec: matched.spec,
        params: matched.params,
    })
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Stand-in model identifier used when filling path templates.
    const MODEL_ID: &str = "us.anthropic.claude-3-7-sonnet";

    #[test]
    fn registry_keys_and_operation_ids_are_unique() {
        let keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method(), spec.transport().as_str(), spec.runtime_path()))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), OPERATION_SPECS.len(), "duplicate method/transport/path key");

        let ids = OPERATION_SPECS
            .iter()
            .map(BedrockOperationSpec::operation_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), OPERATION_SPECS.len(), "duplicate operation ID");
    }

    #[test]
    fn every_registered_operation_resolves_from_its_own_template() {
        for spec in OPERATION_SPECS {
            let path = spec.runtime_path().replace("{model_id}", MODEL_ID);
            let matched = match_route(spec.method().as_str(), &path).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method().as_str()
            );
        }
    }

    #[test]
    fn every_operation_belongs_to_the_amazon_bedrock_protocol() {
        assert!(
            OPERATION_SPECS
                .iter()
                .all(|spec| spec.application_protocol() == APPLICATION_PROTOCOL),
            "every entry must declare the registry's protocol"
        );
    }

    #[test]
    fn the_model_parameter_is_its_own_segment() {
        let matched = match_route("POST", "/model/us.anthropic.claude-3-7-sonnet/converse").unwrap();
        assert_eq!(matched.params.get("model_id"), Some("us.anthropic.claude-3-7-sonnet"));
    }

    #[test]
    fn an_unknown_action_does_not_match() {
        assert!(match_route("POST", "/model/us.meta.llama-4/invokeModelWithBidirectionalStream").is_none());
        assert!(match_route("POST", "/model/us.meta.llama-4").is_none());
    }
}
