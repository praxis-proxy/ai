// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` API filters: Responses API pipeline.

#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "the shared API client intentionally exposes operations used by different OpenAI filters"
)]
pub(crate) mod api_client;
pub(crate) mod chat_completions;
pub(crate) mod conversations;
pub(crate) mod error_response_formatter;
pub(crate) mod files;
#[cfg(feature = "store")]
pub(crate) mod include;
mod operation;
pub(crate) mod responses;
pub(crate) mod sse;
#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "Responses translation helpers are wired into the HTTP filter in a later stack entry"
)]
pub(crate) mod translation;
pub(crate) mod vector_stores;

pub use chat_completions::{
    request::OpenaiChatCompletionsRequestFilter,
    routes::{
        ChatCompletionsOperation, ChatCompletionsOperationSpec, operation_specs as chat_completions_operation_specs,
    },
};
pub use conversations::{
    ConversationOperation, ConversationOperationSpec, operation_specs as conversations_operation_specs,
};
#[cfg(feature = "openai-conversations")]
pub use conversations::{OpenaiConversationsFilter, implementation_openapi_json as conversations_openapi_json};
pub use files::routes::{FilesOperation, FilesOperationSpec, operation_specs as files_operation_specs};
pub use operation::OpenAiOperationSpec;
#[cfg(feature = "openai-compact")]
pub use responses::CompactFilter;
#[cfg(feature = "openai-file-resolve-filter")]
pub use responses::FileResolveFilter;
#[cfg(feature = "openai-responses-openapi")]
pub use responses::implementation_openapi_json as responses_openapi_json;
#[cfg(feature = "openai-responses")]
pub use responses::{
    AgenticLoopFilter, ClientToolCompatFilter, DocExtractFilter, FileSearchCalloutFilter, OpenaiResponsesRequestFilter,
    WebSearchFilter, local_tool_guardrail_messages, openai_responses_proxy::ResponsesProxyFilter,
    record_local_tool_guardrail_failure, responses_to_chat_completions::ResponsesToChatCompletionsFilter,
    stream_events::OpenaiStreamEventsFilter,
};
#[cfg(feature = "openai-mcp-tools")]
pub use responses::{McpDispatchFilter, McpToolResolveFilter};
pub use responses::{
    ModelRewriteFilter, ToolParseFilter,
    routes::{
        PROTOCOL_EXTENSION_OPERATION_IDS as RESPONSES_PROTOCOL_EXTENSION_OPERATION_IDS, ResponsesOperation,
        ResponsesOperationSpec, operation_specs as responses_operation_specs,
    },
};
#[cfg(feature = "store")]
pub use responses::{RehydrateFilter, ResponseStoreFilter};
pub use vector_stores::routes::{
    VectorStoresOperation, VectorStoresOperationSpec, operation_specs as vector_stores_operation_specs,
};

#[cfg(feature = "openai-mcp-tools")]
pub use crate::mcp_client::McpStreamingSelectorFilter;
