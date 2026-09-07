// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Responses operation registry.
//!
//! Operation identity comes from the request head — method, path, and protocol
//! headers — rather than from body heuristics, so a request is recognized before
//! any payload is read.
//!
//! Praxis proxies the Responses contract rather than owning it, so these
//! operations declare their runtime request-body shape without an
//! `OwnedOperationContract`. Operation IDs are the official ones from the pinned
//! OpenAI specification, reproduced verbatim including upstream's casing.

use std::ops::Deref;

use crate::openai::operation::{
    OpenAiApiFamily, OpenAiHandlingMode, OpenAiHttpMethod, OpenAiOperationSpec, OpenAiRequestBody, OpenAiTransport,
    OperationEntry, RouteParams, match_operation,
};

/// Static metadata for one Responses operation.
#[derive(Clone, Copy)]
pub struct ResponsesOperationSpec {
    /// Runtime operation.
    pub operation: ResponsesOperation,
    /// Shared operation metadata.
    pub definition: OpenAiOperationSpec,
}

impl Deref for ResponsesOperationSpec {
    type Target = OpenAiOperationSpec;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

impl OperationEntry for ResponsesOperationSpec {
    fn spec(&self) -> &OpenAiOperationSpec {
        &self.definition
    }
}

/// Convert a registry body declaration into a runtime request-body shape.
macro_rules! request_body_shape {
    ([none]) => {
        OpenAiRequestBody::None
    };
    ([required json]) => {
        OpenAiRequestBody::Json { required: true }
    };
    ([optional json]) => {
        OpenAiRequestBody::Json { required: false }
    };
}

/// Declare each Responses operation once and derive its runtime metadata.
macro_rules! responses_operations {
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
        /// One Responses operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum ResponsesOperation {
            $(
                #[doc = concat!(stringify!($method), " /v1", $path)]
                $operation,
            )+
        }

        /// All Responses operations recognized by Praxis.
        pub const OPERATION_SPECS: &[ResponsesOperationSpec] = &[
            $(
                ResponsesOperationSpec {
                    operation: ResponsesOperation::$operation,
                    definition: OpenAiOperationSpec {
                        family: OpenAiApiFamily::Responses,
                        operation_id: $operation_id,
                        method: OpenAiHttpMethod::$method,
                        transport: OpenAiTransport::$transport,
                        spec_path: $path,
                        runtime_path: concat!("/v1", $path),
                        mode: OpenAiHandlingMode::$mode,
                        request_body: request_body_shape!($body),
                        owned_contract: None,
                    },
                },
            )+
        ];
    };
}

responses_operations! {
    CreateResponse {
        operation_id: "createResponse",
        method: Post,
        transport: Http,
        path: "/responses",
        mode: Inspect,
        body: [required json],
    },
    CreateResponseWebSocket {
        operation_id: "praxis_createResponseWebSocket",
        method: Get,
        transport: WebSocket,
        path: "/responses",
        mode: Passthrough,
        body: [none],
    },
    GetResponse {
        operation_id: "getResponse",
        method: Get,
        transport: Http,
        path: "/responses/{response_id}",
        mode: Passthrough,
        body: [none],
    },
    DeleteResponse {
        operation_id: "deleteResponse",
        method: Delete,
        transport: Http,
        path: "/responses/{response_id}",
        mode: Passthrough,
        body: [none],
    },
    CancelResponse {
        operation_id: "cancelResponse",
        method: Post,
        transport: Http,
        path: "/responses/{response_id}/cancel",
        mode: Passthrough,
        body: [none],
    },
    ListInputItems {
        operation_id: "listInputItems",
        method: Get,
        transport: Http,
        path: "/responses/{response_id}/input_items",
        mode: Passthrough,
        body: [none],
    },
    CountInputTokens {
        operation_id: "Getinputtokencounts",
        method: Post,
        transport: Http,
        path: "/responses/input_tokens",
        mode: Passthrough,
        // The pinned specification omits `required` on this requestBody,
        // which defaults to false under OpenAPI.
        body: [optional json],
    },
    CompactConversation {
        operation_id: "Compactconversation",
        method: Post,
        transport: Http,
        path: "/responses/compact",
        mode: Passthrough,
        // The pinned specification omits `required` on this requestBody,
        // which defaults to false under OpenAPI.
        body: [optional json],
    },
}

/// Operation IDs Praxis defines itself because the pinned specification does
/// not represent them as HTTP operations.
///
/// The Responses `WebSocket` handshake shares its method and path with an HTTP
/// operation and is separated by transport, so it carries a Praxis-owned ID
/// rather than an official one. Drift checks skip these.
pub const PROTOCOL_EXTENSION_OPERATION_IDS: &[&str] = &["praxis_createResponseWebSocket"];

/// One matched Responses route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedResponsesRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static ResponsesOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    params: RouteParams<'a>,
}

impl<'a> MatchedResponsesRoute<'a> {
    /// Return the borrowed response ID path segment.
    pub(crate) fn response_id(&self) -> Option<&'a str> {
        self.params.get("response_id")
    }
}

/// Return all Responses operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [ResponsesOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Responses operation.
///
/// Matching rules, precedence, and path normalization live in the shared
/// operation module. Transport separates `POST /v1/responses` from the
/// `WebSocket` handshake at the same path.
pub(crate) fn match_route<'a>(
    method: &str,
    path: &'a str,
    transport: OpenAiTransport,
) -> Option<MatchedResponsesRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, transport).map(|matched| MatchedResponsesRoute {
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
    fn registry_keys_and_operation_ids_are_unique() {
        let keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method, spec.transport.as_str(), spec.spec_path))
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), OPERATION_SPECS.len(), "duplicate method/transport/path key");

        let ids = OPERATION_SPECS
            .iter()
            .map(|spec| spec.operation_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), OPERATION_SPECS.len(), "duplicate operation ID");
    }

    #[test]
    fn every_registered_operation_resolves_from_its_own_template() {
        for spec in OPERATION_SPECS {
            let path = spec.runtime_path.replace("{response_id}", "resp_test");
            let matched = match_route(spec.method.as_str(), &path, spec.transport).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method.as_str()
            );
        }
    }

    #[test]
    fn static_endpoints_are_not_consumed_as_response_ids() {
        for (path, expected) in [
            ("/v1/responses/input_tokens", ResponsesOperation::CountInputTokens),
            ("/v1/responses/compact", ResponsesOperation::CompactConversation),
        ] {
            let matched = match_route("POST", path, OpenAiTransport::Http).unwrap();
            assert_eq!(matched.spec.operation, expected, "{path}");
            assert_eq!(matched.response_id(), None, "{path} must not capture a response ID");
        }
    }

    #[test]
    fn identifier_paths_still_capture_the_response_id() {
        let matched = match_route("GET", "/v1/responses/resp_abc123", OpenAiTransport::Http).unwrap();
        assert_eq!(matched.spec.operation, ResponsesOperation::GetResponse);
        assert_eq!(matched.response_id(), Some("resp_abc123"));

        let matched = match_route("POST", "/v1/responses/resp_abc123/cancel", OpenAiTransport::Http).unwrap();
        assert_eq!(matched.spec.operation, ResponsesOperation::CancelResponse);
        assert_eq!(matched.response_id(), Some("resp_abc123"));

        let matched = match_route("GET", "/v1/responses/resp_abc123/input_items", OpenAiTransport::Http).unwrap();
        assert_eq!(matched.spec.operation, ResponsesOperation::ListInputItems);
        assert_eq!(matched.response_id(), Some("resp_abc123"));
    }

    #[test]
    fn create_and_websocket_are_separated_without_reading_a_body() {
        let create = match_route("POST", "/v1/responses", OpenAiTransport::Http).unwrap();
        assert_eq!(create.spec.operation, ResponsesOperation::CreateResponse);

        let socket = match_route("GET", "/v1/responses", OpenAiTransport::WebSocket).unwrap();
        assert_eq!(socket.spec.operation, ResponsesOperation::CreateResponseWebSocket);

        assert!(
            match_route("GET", "/v1/responses", OpenAiTransport::Http).is_none(),
            "a plain GET on the collection is not a registered operation"
        );
        assert!(
            match_route("POST", "/v1/responses", OpenAiTransport::WebSocket).is_none(),
            "create is not reachable over a websocket handshake"
        );
    }

    #[test]
    fn unsupported_methods_and_paths_do_not_match() {
        for (method, path) in [
            ("PUT", "/v1/responses"),
            ("PATCH", "/v1/responses/resp_abc"),
            ("DELETE", "/v1/responses"),
            ("GET", "/v1/responses/resp_abc/cancel"),
            ("GET", "/v1/responses/resp_abc/other"),
            ("POST", "/v1/responses/resp_abc/input_items"),
        ] {
            assert!(
                match_route(method, path, OpenAiTransport::Http).is_none(),
                "{method} {path} must not match a Responses operation"
            );
        }
    }

    #[test]
    fn body_shapes_match_the_pinned_specification() {
        for spec in OPERATION_SPECS {
            let expected = match spec.operation {
                // The only Responses operation the specification marks required.
                ResponsesOperation::CreateResponse => OpenAiRequestBody::Json { required: true },
                // `requestBody` present but `required` omitted, so false.
                ResponsesOperation::CountInputTokens | ResponsesOperation::CompactConversation => {
                    OpenAiRequestBody::Json { required: false }
                },
                _ => OpenAiRequestBody::None,
            };
            assert_eq!(
                spec.request_body, expected,
                "{:?} reported the wrong body shape",
                spec.operation
            );
        }
    }

    #[test]
    fn only_the_websocket_operation_is_a_praxis_extension() {
        let extensions = OPERATION_SPECS
            .iter()
            .filter(|spec| PROTOCOL_EXTENSION_OPERATION_IDS.contains(&spec.operation_id))
            .map(|spec| spec.operation)
            .collect::<Vec<_>>();
        assert_eq!(extensions, vec![ResponsesOperation::CreateResponseWebSocket]);

        assert!(
            OPERATION_SPECS
                .iter()
                .filter(|spec| spec.transport == OpenAiTransport::WebSocket)
                .all(|spec| PROTOCOL_EXTENSION_OPERATION_IDS.contains(&spec.operation_id)),
            "every websocket operation must be declared as a Praxis protocol extension"
        );
    }

    #[test]
    fn registry_declares_no_owned_contract() {
        assert!(
            OPERATION_SPECS.iter().all(|spec| spec.owned_contract().is_none()),
            "Praxis proxies the Responses contract rather than owning it"
        );
    }
}
