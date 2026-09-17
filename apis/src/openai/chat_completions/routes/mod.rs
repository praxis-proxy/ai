// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Chat Completions operation registry.
//!
//! Operation identity comes from the request head — method, path, and protocol
//! headers — rather than from body heuristics, so a request is recognized before
//! any payload is read.
//!
//! Praxis proxies the Chat Completions contract rather than owning it, so these
//! operations declare their runtime request-body shape without an
//! `OwnedOperationContract`. Operation IDs are the official ones from the pinned
//! OpenAI specification, reproduced verbatim including upstream's casing.

use std::ops::Deref;

use crate::openai::operation::{
    OpenAiApiFamily, OpenAiHandlingMode, OpenAiHttpMethod, OpenAiOperationSpec, OpenAiRequestBody, OpenAiTransport,
    OperationEntry, match_operation,
};

/// Static metadata for one Chat Completions operation.
#[derive(Clone, Copy)]
pub struct ChatCompletionsOperationSpec {
    /// Runtime operation.
    pub operation: ChatCompletionsOperation,
    /// Shared operation metadata.
    pub definition: OpenAiOperationSpec,
}

impl Deref for ChatCompletionsOperationSpec {
    type Target = OpenAiOperationSpec;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

impl OperationEntry for ChatCompletionsOperationSpec {
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
}

/// Declare each Chat Completions operation once and derive its runtime metadata.
macro_rules! chat_completions_operations {
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
        /// One Chat Completions operation recognized from the request head.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum ChatCompletionsOperation {
            $(
                #[doc = concat!(stringify!($method), " /v1", $path)]
                $operation,
            )+
        }

        /// All Chat Completions operations recognized by Praxis.
        pub const OPERATION_SPECS: &[ChatCompletionsOperationSpec] = &[
            $(
                ChatCompletionsOperationSpec {
                    operation: ChatCompletionsOperation::$operation,
                    definition: OpenAiOperationSpec {
                        family: OpenAiApiFamily::ChatCompletions,
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

chat_completions_operations! {
    ListChatCompletions {
        operation_id: "listChatCompletions",
        method: Get,
        transport: Http,
        path: "/chat/completions",
        mode: Passthrough,
        body: [none],
    },
    CreateChatCompletion {
        operation_id: "createChatCompletion",
        method: Post,
        transport: Http,
        path: "/chat/completions",
        mode: Passthrough,
        body: [required json],
    },
    GetChatCompletion {
        operation_id: "getChatCompletion",
        method: Get,
        transport: Http,
        path: "/chat/completions/{completion_id}",
        mode: Passthrough,
        body: [none],
    },
    UpdateChatCompletion {
        operation_id: "updateChatCompletion",
        method: Post,
        transport: Http,
        path: "/chat/completions/{completion_id}",
        mode: Passthrough,
        body: [required json],
    },
    DeleteChatCompletion {
        operation_id: "deleteChatCompletion",
        method: Delete,
        transport: Http,
        path: "/chat/completions/{completion_id}",
        mode: Passthrough,
        body: [none],
    },
    GetChatCompletionMessages {
        operation_id: "getChatCompletionMessages",
        method: Get,
        transport: Http,
        path: "/chat/completions/{completion_id}/messages",
        mode: Passthrough,
        body: [none],
    },
}

/// One matched Chat Completions route.
#[derive(Clone, Copy)]
pub(crate) struct MatchedChatCompletionsRoute {
    /// Matched operation metadata.
    pub spec: &'static ChatCompletionsOperationSpec,
}

/// Return all Chat Completions operation specs.
#[must_use]
pub const fn operation_specs() -> &'static [ChatCompletionsOperationSpec] {
    OPERATION_SPECS
}

/// Match a request head to a Chat Completions operation.
///
/// Chat Completions is reached over plain HTTP only.
pub(crate) fn match_route(method: &str, path: &str) -> Option<MatchedChatCompletionsRoute> {
    match_operation(OPERATION_SPECS, method, path, OpenAiTransport::Http)
        .map(|matched| MatchedChatCompletionsRoute { spec: matched.spec })
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
            let path = spec.runtime_path.replace("{completion_id}", "chatcmpl_test");
            let matched = match_route(spec.method.as_str(), &path).unwrap();
            assert_eq!(
                matched.spec.operation,
                spec.operation,
                "{} {path} resolved to the wrong operation",
                spec.method.as_str()
            );
        }
    }

    #[test]
    fn identifier_paths_capture_the_completion_id() {
        let matched = match_operation(
            OPERATION_SPECS,
            "GET",
            "/v1/chat/completions/chatcmpl_abc123",
            OpenAiTransport::Http,
        )
        .unwrap();
        assert_eq!(matched.spec.operation, ChatCompletionsOperation::GetChatCompletion);
        assert_eq!(matched.params.get("completion_id"), Some("chatcmpl_abc123"));

        let matched = match_operation(
            OPERATION_SPECS,
            "GET",
            "/v1/chat/completions/chatcmpl_abc123/messages",
            OpenAiTransport::Http,
        )
        .unwrap();
        assert_eq!(
            matched.spec.operation,
            ChatCompletionsOperation::GetChatCompletionMessages
        );
        assert_eq!(matched.params.get("completion_id"), Some("chatcmpl_abc123"));
    }

    #[test]
    fn create_and_list_are_separated_without_reading_a_body() {
        let create = match_route("POST", "/v1/chat/completions").unwrap();
        assert_eq!(create.spec.operation, ChatCompletionsOperation::CreateChatCompletion);

        let list = match_route("GET", "/v1/chat/completions").unwrap();
        assert_eq!(list.spec.operation, ChatCompletionsOperation::ListChatCompletions);
    }

    #[test]
    fn trailing_slash_and_query_string_are_normalized() {
        for path in [
            "/v1/chat/completions",
            "/v1/chat/completions/",
            "/v1/chat/completions?limit=5",
            "/v1/chat/completions/?limit=5",
        ] {
            let matched = match_route("POST", path).unwrap();
            assert_eq!(
                matched.spec.operation,
                ChatCompletionsOperation::CreateChatCompletion,
                "{path}"
            );
        }
    }

    #[test]
    fn unsupported_methods_and_paths_do_not_match() {
        for (method, path) in [
            ("PUT", "/v1/chat/completions"),
            ("PATCH", "/v1/chat/completions/chatcmpl_abc"),
            ("DELETE", "/v1/chat/completions"),
            ("GET", "/v1/chat/completions/chatcmpl_abc/other"),
            ("POST", "/v1/chat/completions/chatcmpl_abc/messages"),
        ] {
            assert!(
                match_route(method, path).is_none(),
                "{method} {path} must not match a Chat Completions operation"
            );
        }
    }

    #[test]
    fn body_shapes_match_the_pinned_specification() {
        for spec in OPERATION_SPECS {
            let expected = match spec.operation {
                ChatCompletionsOperation::CreateChatCompletion | ChatCompletionsOperation::UpdateChatCompletion => {
                    OpenAiRequestBody::Json { required: true }
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
    fn registry_proxies_without_inspecting_or_owning_the_contract() {
        assert!(
            OPERATION_SPECS
                .iter()
                .all(|spec| spec.mode == OpenAiHandlingMode::Passthrough && spec.owned_contract().is_none()),
            "Praxis proxies Chat Completions without inspecting or owning the contract"
        );
    }
}
