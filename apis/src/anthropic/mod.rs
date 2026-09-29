// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic protocol filters.

use praxis_filter::{BoundUpstreamBodyOutcome, FilterAction, FilterError};

pub(crate) mod error_response_formatter;
mod messages_format;
pub(crate) mod messages_to_chat_completions;
mod messages_to_chat_completions_stream;
mod protocol;
mod validate;
mod web_search;
mod wire;

pub use messages_format::AnthropicMessagesFormatFilter;
pub use messages_to_chat_completions::AnthropicMessagesToChatCompletionsFilter;
pub use messages_to_chat_completions_stream::AnthropicMessagesToChatCompletionsStreamFilter;
pub use protocol::AnthropicMessagesProtocolFilter;
pub use validate::AnthropicValidateFilter;
pub use web_search::AnthropicWebSearchFilter;

/// Reduce a request-body action to the bound-upstream body's narrower
/// continue-or-reject contract.
pub(crate) fn bound_body_outcome(action: FilterAction) -> Result<BoundUpstreamBodyOutcome, FilterError> {
    match action {
        FilterAction::Continue | FilterAction::Release | FilterAction::BodyDone => {
            Ok(BoundUpstreamBodyOutcome::Continue)
        },
        FilterAction::Reject(rejection) => Ok(BoundUpstreamBodyOutcome::Reject(rejection)),
        FilterAction::TerminalResponse(_) | FilterAction::StreamingTerminalResponse(_) => {
            Err("terminal response is invalid during bound-upstream body processing".into())
        },
    }
}
