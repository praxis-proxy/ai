// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Google Gemini operation registry.
//!
//! Operation identity comes from the request head — method, path, and transport
//! — rather than from body heuristics, so a request is recognized before any
//! payload is read.
//!
//! # Source of truth
//!
//! The inventory below follows the Gemini Developer API reference
//! (`generativelanguage.googleapis.com`), whose paths put the action verb in the
//! model segment after a colon: `/v1/models/{model}:generateContent`. Google
//! publishes no `OpenAPI` document for this surface, so there is no pinned
//! spec to drift-check against; the operation IDs are Google's own RPC names,
//! reproduced verbatim. Both `v1` and `v1beta` prefixes are registered because
//! clients deploy on either.
//!
//! The Vertex AI form —
//! `/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
//! — is deliberately absent: Praxis reaches Vertex through the
//! `openai_chat_completions_to_vertexai_gemini` translation filter, which owns
//! its own path generation, so native Vertex passthrough heads publish no
//! operation match.
//!
//! # Handling modes
//!
//! Every entry is [`HandlingMode::Passthrough`]: no Gemini filter deliberately
//! targets these heads today. The registry exists so classifier consumers,
//! such as `token_count` in auto mode, can identify the dialect from the
//! published match instead of matching paths themselves.

use crate::operation::{
    ApplicationProtocol, HandlingMode, HttpMethod, OperationEntry, OperationSpec, RequestBody, RouteParams, Transport,
    match_operation,
};

/// Application protocol these operations belong to.
///
/// Declared beside the registry that owns it, so registering a protocol
/// never edits a shared list.
pub(crate) const APPLICATION_PROTOCOL: ApplicationProtocol = ApplicationProtocol::new("google_gemini");

/// Static metadata for one Gemini operation.
///
/// Holds the crate's shared `OperationSpec` directly: Praxis generates no
/// `OpenAPI` contract for Gemini, so there is no provider-owned metadata to
/// carry beside the runtime identity.
#[derive(Clone, Copy)]
pub struct GeminiOperationSpec {
    /// Runtime operation.
    pub operation: GeminiOperation,
    /// Shared operation metadata.
    ///
    /// Crate-private because the shared `OperationSpec` is: external callers read this
    /// entry through the accessors below rather than the shared struct.
    pub(crate) definition: OperationSpec,
}

impl OperationEntry for GeminiOperationSpec {
    fn spec(&self) -> &OperationSpec {
        &self.definition
    }
}

impl GeminiOperationSpec {
    /// Stable operation ID, as spelled in Google's API reference.
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

/// Declare each Gemini operation once and derive its metadata.
macro_rules! gemini_operations {
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
        /// One Gemini operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum GeminiOperation {
            $(
                #[doc = concat!(stringify!($method), " ", $path)]
                $operation,
            )+
        }

        /// All Gemini operations recognized by Praxis.
        pub const OPERATION_SPECS: &[GeminiOperationSpec] = &[
            $(
                GeminiOperationSpec {
                    operation: GeminiOperation::$operation,
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

gemini_operations! {
    GenerateContent {
        operation_id: "generateContent",
        method: Post,
        transport: Http,
        path: "/v1/models/{model}:generateContent",
        mode: Passthrough,
        body: [required json],
    },
    StreamGenerateContent {
        operation_id: "streamGenerateContent",
        method: Post,
        transport: Http,
        path: "/v1/models/{model}:streamGenerateContent",
        mode: Passthrough,
        body: [required json],
    },
    CountTokens {
        operation_id: "countTokens",
        method: Post,
        transport: Http,
        path: "/v1/models/{model}:countTokens",
        mode: Passthrough,
        body: [required json],
    },
    GenerateContentV1Beta {
        operation_id: "generateContent",
        method: Post,
        transport: Http,
        path: "/v1beta/models/{model}:generateContent",
        mode: Passthrough,
        body: [required json],
    },
    StreamGenerateContentV1Beta {
        operation_id: "streamGenerateContent",
        method: Post,
        transport: Http,
        path: "/v1beta/models/{model}:streamGenerateContent",
        mode: Passthrough,
        body: [required json],
    },
    CountTokensV1Beta {
        operation_id: "countTokens",
        method: Post,
        transport: Http,
        path: "/v1beta/models/{model}:countTokens",
        mode: Passthrough,
        body: [required json],
    },
}

/// One matched Gemini route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedGeminiRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static GeminiOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    pub(crate) params: RouteParams<'a>,
}

/// Return all Gemini operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [GeminiOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Gemini operation.
///
/// Gemini is reached over plain HTTP only. Streaming selects `streamGenerateContent`
/// by path rather than by query parameter, so the `?alt=sse` query clients add
/// never changes which operation is selected: query strings do not participate
/// in matching.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedGeminiRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, Transport::Http).map(|matched| MatchedGeminiRoute {
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

    #[test]
    fn registry_keys_are_unique() {
        let keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method(), spec.transport().as_str(), spec.runtime_path()))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), OPERATION_SPECS.len(), "duplicate method/transport/path key");
    }

    /// Operation IDs repeat across `v1` and `v1beta` by design: they are the
    /// same Google RPC served on two API versions, and operation identity is
    /// protocol plus ID, not path. The ID set is therefore asserted as a set
    /// of the distinct RPC names rather than for uniqueness.
    #[test]
    fn operation_ids_are_the_distinct_rpc_names() {
        let ids = OPERATION_SPECS
            .iter()
            .map(GeminiOperationSpec::operation_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            ids,
            BTreeSet::from(["countTokens", "generateContent", "streamGenerateContent"])
        );
    }

    #[test]
    fn every_registered_operation_resolves_from_its_own_template() {
        for spec in OPERATION_SPECS {
            let path = spec.runtime_path().replace("{model}", "gemini-2.0-flash");
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
    fn every_operation_belongs_to_the_google_gemini_protocol() {
        assert!(
            OPERATION_SPECS
                .iter()
                .all(|spec| spec.application_protocol() == APPLICATION_PROTOCOL),
            "every entry must declare the registry's protocol"
        );
    }

    #[test]
    fn the_model_parameter_excludes_the_action_suffix() {
        let matched = match_route("POST", "/v1/models/gemini-2.0-flash:generateContent").unwrap();
        assert_eq!(matched.params.get("model"), Some("gemini-2.0-flash"));
    }

    #[test]
    fn an_unknown_action_does_not_match() {
        assert!(match_route("POST", "/v1/models/gemini-2.0-flash:embedContent").is_none());
        assert!(match_route("POST", "/v1/models/gemini-2.0-flash").is_none());
    }
}
