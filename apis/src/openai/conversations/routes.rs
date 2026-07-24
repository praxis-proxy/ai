// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Conversations operation registry and zero-allocation runtime matcher.

use std::ops::Deref;

use utoipa::PartialSchema;

use super::contracts::{
    ConversationItem, ConversationItemList, ConversationResource, CreateConversationItemsRequest,
    CreateConversationRequest, DeletedConversationResource, ItemOrder, UpdateConversationRequest,
};
use crate::openai::operation::{
    MediaTypeSpec, OpenAiHandlingMode, OpenAiOperationSpec, OwnedOperationContract, ParameterLocation, ParameterSpec,
    RequestBodySpec, ResponseSpec, schema_binding,
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

/// Convert a registry request declaration into an optional schema binding.
macro_rules! request_binding {
    (none) => {
        None
    };
    ($schema:ty) => {
        Some(RequestBodySpec {
            required: true,
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

/// Convert a typed registry method token into its stable HTTP spelling.
macro_rules! method_name {
    (Get) => {
        "GET"
    };
    (Post) => {
        "POST"
    };
    (Delete) => {
        "DELETE"
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
                        operation_id: $operation_id,
                        method: method_name!($method),
                        spec_path: $path,
                        runtime_path: concat!("/v1", $path),
                        mode: OpenAiHandlingMode::$mode,
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
            request: CreateConversationRequest,
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
            request: none,
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
            request: UpdateConversationRequest,
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
            request: none,
            response: DeletedConversationResource,
        },
    },
    CreateConversationItems {
        operation_id: "createConversationItems",
        method: Post,
        path: "/conversations/{conversation_id}/items",
        mode: Local,
        contract: owned {
            parameters: [path_parameter!(
                "conversation_id",
                "The ID of the conversation to add the items to."
            )],
            request: CreateConversationItemsRequest,
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
            ],
            request: none,
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
            ],
            request: none,
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
            request: none,
            response: ConversationResource,
        },
    },
}

/// Path parameters borrowed directly from the request URI.
#[derive(Clone, Copy, Debug, Default)]
struct RouteParams<'a> {
    /// Conversation identifier path segment.
    conversation_id: Option<&'a str>,
    /// Conversation item identifier path segment.
    item_id: Option<&'a str>,
}

impl<'a> RouteParams<'a> {
    /// Record a parameter recognized by the Conversations registry.
    fn insert(&mut self, name: &str, value: &'a str) -> Option<()> {
        let slot = match name {
            "conversation_id" => &mut self.conversation_id,
            "item_id" => &mut self.item_id,
            _ => return None,
        };
        if slot.replace(value).is_some() {
            return None;
        }
        Some(())
    }
}

/// One matched runtime route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedConversationRoute<'a> {
    /// Matched operation metadata.
    pub spec: &'static ConversationOperationSpec,
    /// Borrowed path parameters.
    params: RouteParams<'a>,
}

impl<'a> MatchedConversationRoute<'a> {
    /// Return the borrowed conversation ID path segment.
    pub(crate) const fn conversation_id(&self) -> Option<&'a str> {
        self.params.conversation_id
    }

    /// Return the borrowed item ID path segment.
    pub(crate) const fn item_id(&self) -> Option<&'a str> {
        self.params.item_id
    }
}

/// Return all Conversations operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [ConversationOperationSpec] {
    OPERATION_SPECS
}

/// Match an HTTP method and runtime path to a Conversations operation.
pub(crate) fn match_route<'a>(method: &str, path: &'a str) -> Option<MatchedConversationRoute<'a>> {
    let path = path.strip_suffix('/').filter(|path| !path.is_empty()).unwrap_or(path);
    OPERATION_SPECS
        .iter()
        .filter(|spec| spec.method == method)
        .find_map(|spec| {
            match_path_template(spec.runtime_path, path).map(|params| MatchedConversationRoute { spec, params })
        })
}

/// Match one path against a template with `{param}` placeholders.
fn match_path_template<'a>(template: &'static str, path: &'a str) -> Option<RouteParams<'a>> {
    let mut template_segments = template.split('/');
    let mut path_segments = path.split('/');
    let mut params = RouteParams::default();

    loop {
        match (template_segments.next(), path_segments.next()) {
            (None, None) => return Some(params),
            (Some(template_segment), Some(path_segment)) => {
                if let Some(name) = template_segment
                    .strip_prefix('{')
                    .and_then(|segment| segment.strip_suffix('}'))
                {
                    if path_segment.is_empty() {
                        return None;
                    }
                    params.insert(name, path_segment)?;
                } else if template_segment != path_segment {
                    return None;
                }
            },
            _ => return None,
        }
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

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
            let route = match_route(spec.method, &path).unwrap();
            assert_eq!(route.spec.operation, spec.operation);
        }
    }

    #[test]
    fn rejects_empty_parameter() {
        assert!(match_route("GET", "/v1/conversations/").is_none());
    }
}
