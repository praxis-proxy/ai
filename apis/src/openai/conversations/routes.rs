// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Conversations operation registry and zero-allocation runtime matcher.

use std::ops::Deref;

use utoipa::PartialSchema;

use super::contracts::{
    ConversationItem, ConversationItemList, ConversationResource, CreateConversationItemsRequest,
    CreateConversationRequest, DeletedConversationResource, ItemOrder, UpdateConversationRequest,
};
use crate::openai::{
    include::IncludeField,
    operation::{
        MediaTypeSpec, OpenAiApiFamily, OpenAiHandlingMode, OpenAiHttpMethod, OpenAiOperationSpec, OpenAiRequestBody,
        OpenAiTransport, OperationEntry, OwnedOperationContract, ParameterLocation, ParameterSpec, RequestBodySpec,
        ResponseSpec, RouteParams, match_operation, schema_binding,
    },
};

/// JSON media type used by all Conversations bodies.
const JSON_CONTENT_TYPE: &str = "application/json";

/// Static metadata for one Conversations operation.
#[derive(Clone, Copy)]
pub struct ConversationOperationSpec {
    /// Runtime operation.
    pub operation: ConversationOperation,
    /// Shared operation and owned-contract metadata.
    pub definition: OpenAiOperationSpec,
}

impl Deref for ConversationOperationSpec {
    type Target = OpenAiOperationSpec;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

impl OperationEntry for ConversationOperationSpec {
    fn spec(&self) -> &OpenAiOperationSpec {
        &self.definition
    }
}

/// Convert a registry request declaration into an optional schema binding.
macro_rules! request_binding {
    ([none]) => {
        None
    };
    ([required $schema:ty]) => {
        Some(RequestBodySpec {
            required: true,
            content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!($schema))],
        })
    };
    ([optional $schema:ty]) => {
        Some(RequestBodySpec {
            required: false,
            content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!($schema))],
        })
    };
}

/// Convert a registry contract declaration into optional owned metadata.
#[expect(
    unused_macro_rules,
    reason = "non-owning form is part of the registry API but current Conversations operations are all local"
)]
macro_rules! operation_contract {
    (none {}) => {
        None
    };
    (
        owned {
            parameters: [$($parameter:expr),* $(,)?],
            request: $request:tt,
            response: $response:ty $(,)?
        }
    ) => {
        Some(OwnedOperationContract {
            parameters: &[$($parameter),*],
            request: request_binding!($request),
            responses: &[ResponseSpec {
                status: "200",
                description: "OK",
                content: &[MediaTypeSpec::new(JSON_CONTENT_TYPE, schema_binding!($response))],
            }],
        })
    };
}

/// Derive the runtime request-body shape from a registry request declaration.
///
/// All Conversations bodies are JSON; other families supply multipart or binary
/// shapes through the same shared [`OpenAiRequestBody`] type.
macro_rules! request_body_shape {
    ([none]) => {
        OpenAiRequestBody::None
    };
    ([required $schema:ty]) => {
        OpenAiRequestBody::Json { required: true }
    };
    ([optional $schema:ty]) => {
        OpenAiRequestBody::Json { required: false }
    };
}

/// Derive the runtime request-body shape from an operation contract declaration.
///
/// Reads the same `request:` token as [`operation_contract`], so the body shape
/// and the generated contract cannot drift apart.
#[expect(
    unused_macro_rules,
    reason = "non-owning form is part of the registry API but current Conversations operations are all local"
)]
macro_rules! contract_request_body {
    (none {}) => {
        OpenAiRequestBody::None
    };
    (owned { parameters: [$($parameter:expr),* $(,)?],request: $request:tt,response: $response:ty $(,)? }) => {
        request_body_shape!($request)
    };
}

/// Declare a required string path parameter.
macro_rules! path_parameter {
    ($name:literal, $description:literal) => {
        ParameterSpec::new(
            $name,
            ParameterLocation::Path,
            true,
            $description,
            <String as PartialSchema>::schema,
        )
    };
}

/// Declare an optional typed query parameter.
macro_rules! query_parameter {
    ($name:literal, $schema:ty, $description:literal) => {
        ParameterSpec::new(
            $name,
            ParameterLocation::Query,
            false,
            $description,
            <$schema as PartialSchema>::schema,
        )
    };
}

/// Declare each operation once and derive both runtime and `OpenAPI` metadata.
macro_rules! conversation_operations {
    (
        $(
            $operation:ident {
                operation_id: $operation_id:literal,
                method: $method:ident,
                path: $path:literal,
                mode: $mode:ident,
                contract: $contract_kind:ident $contract:tt $(,)?
            }
        ),+ $(,)?
    ) => {
        /// One Conversations operation recognized by the local filter.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum ConversationOperation {
            $(
                #[doc = concat!(stringify!($method), " /v1", $path)]
                $operation,
            )+
        }

        /// All Conversations operations recognized by the local filter.
        pub const OPERATION_SPECS: &[ConversationOperationSpec] = &[
            $(
                ConversationOperationSpec {
                    operation: ConversationOperation::$operation,
                    definition: OpenAiOperationSpec {
                        family: OpenAiApiFamily::Conversations,
                        operation_id: $operation_id,
                        method: OpenAiHttpMethod::$method,
                        transport: OpenAiTransport::Http,
                        spec_path: $path,
                        runtime_path: concat!("/v1", $path),
                        mode: OpenAiHandlingMode::$mode,
                        request_body: contract_request_body!($contract_kind $contract),
                        owned_contract: operation_contract!($contract_kind $contract),
                    },
                },
            )+
        ];
    };
}

conversation_operations! {
    CreateConversation {
        operation_id: "createConversation",
        method: Post,
        path: "/conversations",
        mode: Local,
        contract: owned {
            parameters: [],
            request: [optional CreateConversationRequest],
            response: ConversationResource,
        },
    },
    GetConversation {
        operation_id: "getConversation",
        method: Get,
        path: "/conversations/{conversation_id}",
        mode: Local,
        contract: owned {
            parameters: [path_parameter!(
                "conversation_id",
                "The ID of the conversation to retrieve."
            )],
            request: [none],
            response: ConversationResource,
        },
    },
    UpdateConversation {
        operation_id: "updateConversation",
        method: Post,
        path: "/conversations/{conversation_id}",
        mode: Local,
        contract: owned {
            parameters: [path_parameter!(
                "conversation_id",
                "The ID of the conversation to update."
            )],
            request: [required UpdateConversationRequest],
            response: ConversationResource,
        },
    },
    DeleteConversation {
        operation_id: "deleteConversation",
        method: Delete,
        path: "/conversations/{conversation_id}",
        mode: Local,
        contract: owned {
            parameters: [path_parameter!(
                "conversation_id",
                "The ID of the conversation to delete."
            )],
            request: [none],
            response: DeletedConversationResource,
        },
    },
    CreateConversationItems {
        operation_id: "createConversationItems",
        method: Post,
        path: "/conversations/{conversation_id}/items",
        mode: Local,
        contract: owned {
            parameters: [
                path_parameter!(
                    "conversation_id",
                    "The ID of the conversation to add the items to."
                ),
                query_parameter!(
                    "include",
                    Vec<IncludeField>,
                    "Additional fields to include in the response."
                ),
            ],
            request: [required CreateConversationItemsRequest],
            response: ConversationItemList,
        },
    },
    ListConversationItems {
        operation_id: "listConversationItems",
        method: Get,
        path: "/conversations/{conversation_id}/items",
        mode: Local,
        contract: owned {
            parameters: [
                path_parameter!(
                    "conversation_id",
                    "The ID of the conversation to list items for."
                ),
                query_parameter!("limit", u32, "Maximum number of items to return."),
                query_parameter!("order", ItemOrder, "Sort order for returned items."),
                query_parameter!("after", String, "Item ID to list after."),
                query_parameter!(
                    "include",
                    Vec<IncludeField>,
                    "Additional fields to include in the response."
                ),
            ],
            request: [none],
            response: ConversationItemList,
        },
    },
    GetConversationItem {
        operation_id: "getConversationItem",
        method: Get,
        path: "/conversations/{conversation_id}/items/{item_id}",
        mode: Local,
        contract: owned {
            parameters: [
                path_parameter!(
                    "conversation_id",
                    "The ID of the conversation that contains the item."
                ),
                path_parameter!("item_id", "The ID of the item to retrieve."),
                query_parameter!(
                    "include",
                    Vec<IncludeField>,
                    "Additional fields to include in the response."
                ),
            ],
            request: [none],
            response: ConversationItem,
        },
    },
    DeleteConversationItem {
        operation_id: "deleteConversationItem",
        method: Delete,
        path: "/conversations/{conversation_id}/items/{item_id}",
        mode: Local,
        contract: owned {
            parameters: [
                path_parameter!(
                    "conversation_id",
                    "The ID of the conversation that contains the item."
                ),
                path_parameter!("item_id", "The ID of the item to delete."),
            ],
            request: [none],
            response: ConversationResource,
        },
    },
}

/// One matched runtime route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedConversationRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static ConversationOperationSpec,
    /// Borrowed path parameters, captured by the shared matcher.
    params: RouteParams<'a>,
}

impl<'a> MatchedConversationRoute<'a> {
    /// Return the borrowed conversation ID path segment.
    pub(crate) fn conversation_id(&self) -> Option<&'a str> {
        self.params.get("conversation_id")
    }

    /// Return the borrowed item ID path segment.
    pub(crate) fn item_id(&self) -> Option<&'a str> {
        self.params.get("item_id")
    }
}

/// Return all Conversations operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [ConversationOperationSpec] {
    OPERATION_SPECS
}

/// Match an HTTP method and runtime path to a Conversations operation.
///
/// Conversations is reached over plain HTTP only; matching rules, precedence,
/// and path normalization live in the shared operation module.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedConversationRoute<'a>> {
    match_operation(OPERATION_SPECS, method, path, OpenAiTransport::Http).map(|matched| MatchedConversationRoute {
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
    use crate::openai::operation::MAX_PATH_PARAMS;

    #[test]
    fn registry_has_unique_local_conversations_operations() {
        assert_eq!(OPERATION_SPECS.len(), 8);

        let operation_keys = OPERATION_SPECS
            .iter()
            .map(|spec| (spec.method, spec.spec_path))
            .collect::<BTreeSet<_>>();
        assert_eq!(operation_keys.len(), OPERATION_SPECS.len());
        let operation_ids = OPERATION_SPECS
            .iter()
            .map(|spec| spec.operation_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(operation_ids.len(), OPERATION_SPECS.len());
        assert!(OPERATION_SPECS.iter().all(|spec| spec.mode == OpenAiHandlingMode::Local
            && spec.mode.owns_contract()
            && spec.owned_contract().is_some()));
    }

    #[test]
    fn declared_request_body_shape_matches_generated_contract() {
        for spec in OPERATION_SPECS {
            let contract_has_body = spec.owned_contract().is_some_and(|contract| contract.request.is_some());
            assert_eq!(
                spec.request_body.is_present(),
                contract_has_body,
                "request_body shape drifted from the generated contract for {:?}",
                spec.operation
            );
            if let Some(request) = spec.owned_contract().and_then(|contract| contract.request) {
                assert_eq!(
                    spec.request_body.is_required(),
                    request.required,
                    "required flag drifted from the generated contract for {:?}",
                    spec.operation
                );
            }
        }
    }

    #[test]
    fn conversations_bodies_are_json_and_bodyless_reads_have_none() {
        for spec in OPERATION_SPECS {
            match spec.operation {
                ConversationOperation::CreateConversation => {
                    assert_eq!(spec.request_body, OpenAiRequestBody::Json { required: false });
                },
                ConversationOperation::UpdateConversation | ConversationOperation::CreateConversationItems => {
                    assert_eq!(spec.request_body, OpenAiRequestBody::Json { required: true });
                },
                _ => assert_eq!(spec.request_body, OpenAiRequestBody::None),
            }
        }
    }

    #[test]
    fn handling_modes_classify_contract_ownership() {
        assert!(!OpenAiHandlingMode::Passthrough.owns_contract());
        assert!(!OpenAiHandlingMode::Inspect.owns_contract());
        assert!(OpenAiHandlingMode::Transform.owns_contract());
        assert!(OpenAiHandlingMode::Local.owns_contract());
    }

    #[test]
    fn matches_static_runtime_path() {
        let route = match_route("POST", "/v1/conversations").unwrap();
        assert_eq!(route.spec.operation, ConversationOperation::CreateConversation);
        assert!(route.conversation_id().is_none());
    }

    #[test]
    fn matches_parameterized_runtime_path_without_allocating_params() {
        let route = match_route("GET", "/v1/conversations/conv_123/items/item_456").unwrap();
        assert_eq!(route.spec.operation, ConversationOperation::GetConversationItem);
        assert_eq!(route.conversation_id(), Some("conv_123"));
        assert_eq!(route.item_id(), Some("item_456"));
    }

    #[test]
    fn every_registry_runtime_template_matches_its_operation() {
        for spec in OPERATION_SPECS {
            let path = spec
                .runtime_path
                .replace("{conversation_id}", "conv_test")
                .replace("{item_id}", "item_test");
            let route = match_route(spec.method.as_str(), &path).unwrap();
            assert_eq!(route.spec.operation, spec.operation);
        }
    }

    #[test]
    fn rejects_empty_parameter() {
        assert!(match_route("GET", "/v1/conversations/").is_none());
    }

    #[test]
    fn template_parameters_fit_capacity() {
        for spec in OPERATION_SPECS {
            let declared = spec.runtime_path.split('/').filter(|s| s.starts_with('{')).count();
            assert!(
                declared <= MAX_PATH_PARAMS,
                "{} declares {declared} path parameters, above the {MAX_PATH_PARAMS} capacity",
                spec.runtime_path
            );
        }
    }


    #[test]
    fn unmatched_parameter_name_returns_none() {
        let route = match_route("GET", "/v1/conversations/conv_123/items/item_456").unwrap();
        assert_eq!(route.conversation_id(), Some("conv_123"));
        assert_eq!(route.item_id(), Some("item_456"));
        assert_eq!(route.params.get("file_id"), None);
    }
}
