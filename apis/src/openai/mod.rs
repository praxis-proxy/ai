// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` API filters: Responses API pipeline.

#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "api_base_url and timeout are wired in by the vector-store search client (#312)"
)]
pub(crate) mod api_client;
pub(crate) mod conversations;
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
pub use operation::{OpenAiHandlingMode, OpenAiOperationSpec};
pub use responses::{
    DocExtractFilter, FileResolveFilter, McpDispatchFilter, McpToolResolveFilter, ModelRewriteFilter,
    OpenaiResponsesValidateFilter, RehydrateFilter, ResponseStoreFilter, ResponsesFormatFilter, ToolParseFilter,
    WebSearchFilter, openai_responses_proxy::ResponsesProxyFilter, stream_events::OpenaiStreamEventsFilter,
};
