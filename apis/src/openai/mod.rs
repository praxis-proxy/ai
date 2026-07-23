// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` API filters: Responses API pipeline.

#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "the shared API client intentionally exposes operations used by different OpenAI filters"
)]
pub(crate) mod api_client;
pub(crate) mod conversations;
pub(crate) mod responses;
pub(crate) mod sse;
#[expect(clippy::allow_attributes, reason = "dead_code expect unfulfilled on module")]
#[allow(
    dead_code,
    reason = "Responses translation helpers are wired into the HTTP filter in a later stack entry"
)]
pub(crate) mod translation;

pub use conversations::OpenaiConversationsFilter;
pub use responses::{
    DocExtractFilter, FileResolveFilter, FileSearchCalloutFilter, McpDispatchFilter, McpToolResolveFilter,
    ModelRewriteFilter, OpenaiResponsesValidateFilter, RehydrateFilter, ResponseStoreFilter, ResponsesFormatFilter,
    ToolParseFilter, WebSearchFilter, openai_responses_proxy::ResponsesProxyFilter,
    stream_events::OpenaiStreamEventsFilter,
};
