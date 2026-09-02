// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` API filters: Responses API pipeline.

#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "the shared API client intentionally exposes operations used by different OpenAI filters"
)]
pub(crate) mod api_client;
pub(crate) mod conversations;
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
pub(crate) mod url_security;

pub use conversations::{
    ConversationOperation, ConversationOperationSpec, OpenaiConversationsFilter,
    implementation_openapi_json as conversations_openapi_json, operation_specs as conversations_operation_specs,
};
pub use operation::{OpenAiHandlingMode, OpenAiOperationSpec, OpenAiRequestBody};
pub use responses::{
    AgenticLoopFilter, CompactFilter, DocExtractFilter, FileResolveFilter, FileSearchCalloutFilter, McpDispatchFilter,
    McpToolResolveFilter, ModelRewriteFilter, OpenaiResponsesValidateFilter, RehydrateFilter, ResponseStoreFilter,
    ResponsesFormatFilter, ToolParseFilter, WebSearchFilter, openai_responses_proxy::ResponsesProxyFilter,
    responses_to_chat_completions::ResponsesToChatCompletionsFilter, stream_events::OpenaiStreamEventsFilter,
};
