// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic protocol filters and the Messages operation registry.

/// Default Anthropic Messages API version.
pub(crate) const API_VERSION_2023_06_01: &str = "2023-06-01";
/// Anthropic Messages API version 2026-10-21.
pub(crate) const API_VERSION_2026_10_21: &str = "2026-10-21";

pub(crate) mod error_response_formatter;
#[cfg(feature = "anthropic-mcp-connector")]
mod mcp_connector;
mod messages_request;
pub(crate) mod messages_to_chat_completions;
mod messages_to_chat_completions_stream;
mod openai_chat_completions_to_anthropic;
mod openai_chat_completions_to_anthropic_stream;
mod protocol;
pub mod routes;
mod tool_stream;
mod web_search;
mod wire;

#[cfg(feature = "anthropic-mcp-connector")]
pub use mcp_connector::AnthropicMcpConnectorFilter;
pub use messages_request::{AnthropicMessagesRequestFilter, AnthropicMessagesState};
pub use messages_to_chat_completions::AnthropicMessagesToChatCompletionsFilter;
pub use messages_to_chat_completions_stream::AnthropicMessagesToChatCompletionsStreamFilter;
pub use openai_chat_completions_to_anthropic::OpenaiChatCompletionsToAnthropicFilter;
pub use openai_chat_completions_to_anthropic_stream::OpenaiChatCompletionsToAnthropicStreamFilter;
pub use protocol::AnthropicMessagesProtocolFilter;
pub use web_search::AnthropicWebSearchFilter;
#[cfg(feature = "vertex-anthropic")]
pub(crate) use wire::{ErrorType, error_body, error_rejection, invalid_request_rejection};

/// Whether a response carries unencoded bytes that Messages filters can parse.
fn has_identity_encoding(headers: &http::HeaderMap) -> bool {
    headers.get_all(http::header::CONTENT_ENCODING).iter().all(|value| {
        value.to_str().is_ok_and(|list| {
            list.split(',')
                .all(|coding| coding.trim().eq_ignore_ascii_case("identity"))
        })
    })
}
