// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` API filters: Responses API pipeline.

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
    McpToolResolveFilter, ModelRewriteFilter, OpenaiResponsesValidateFilter, RehydrateFilter, ResponseStoreFilter,
    ResponsesFormatFilter, ToolParseFilter, proxy::ResponsesProxyFilter, stream_events::OpenaiStreamEventsFilter,
};
