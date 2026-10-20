// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Agentic loop controller for the Responses API pipeline.
//!
//! Manages the inference loop lifecycle: iteration counting,
//! tool-choice reset, exit conditions, and the loop/done signal
//! available to `iterative_request_router` step transitions.
//!
//! Classifies each round's output by dispatch target to make the
//! loop decision, but does **not** execute any call — MCP execution
//! is handled by `openai_mcp_dispatch`, web search by
//! `openai_web_search_dispatch`, and file search by
//! `openai_file_search_dispatch`. As the sole owner it runs before
//! every dispatcher in the response phase, so it is the central
//! authority that decides loop-vs-done; dispatchers only consume the
//! calls the owner has already routed to them.
//!
//! # Loop control
//!
//! Writes `filter_results` during `on_response_body` where
//! `iterative_request_router` evaluates step transitions:
//!   - `openai_agentic_loop.action = "loop"` — at least one *dispatchable* call present (web search, file search, or an
//!     MCP `function_call`), loop back
//!   - `openai_agentic_loop.action = "done"` — no dispatchable calls (empty, model-owned finish, or client-only
//!     `function_call`s), exit to client
//!
//! A `function_call` that resolves to no configured MCP tool is
//! client-owned: no local dispatcher can execute it, so a round
//! carrying only client calls exits as `done` and returns those calls
//! to the client, rather than looping uselessly to the iteration cap.
//!
//! # Tool call extraction
//!
//! For non-streaming responses,
//! this filter parses the response body JSON and routes each `output`
//! item to its dispatch target: `function_call` items into
//! `state.tool_calls`, `web_search_call` items into
//! `state.web_search_calls`, and each pending `file_search_call` item —
//! appended to the canonical `state.accumulated_output` — into a
//! `state.file_search_assignments` entry recording its absolute output
//! index and synthesis origin. Client `function_call` and `reasoning`
//! items are appended to `state.messages` so the model sees its own
//! calls on re-entry. Hosted `web_search_call` and `file_search_call`
//! items are **not** valid `OpenResponses` input (issue #808), so they
//! never enter `state.messages`; `web_search_call`s remain in
//! `state.web_search_calls` and `file_search_call`s are reached by index
//! through `state.file_search_assignments`, for `openai_web_search_dispatch` /
//! `openai_file_search_dispatch` to dispatch and bridge into backend
//! history, and reach the client only through `state.accumulated_output`.
//!
//! For streaming responses, `stream_events` populates
//! `state.tool_calls` via SSE event parsing. When the body is
//! `None` at end-of-stream (consumed by streaming filters), this
//! filter skips body parsing and checks `state.tool_calls` as-is.
//!
//! `on_request_body` handles iteration bookkeeping: clearing stale
//! tool calls, web search calls, and file search assignments from the
//! previous round and resetting `tool_choice` to `"auto"` on re-entry.
//! The client's `parallel_tool_calls` value is preserved across every
//! round.
//!
//! # Filter order
//!
//! For tool execution, it must appear after `openai_web_search_dispatch`,
//! `openai_mcp_dispatch`, and `openai_file_search_dispatch` and before
//! `openai_responses_proxy`. Response filters execute in reverse
//! order, so the owner parses and classifies each round's output
//! *before* the dispatchers run, routing every call to the vector the
//! matching dispatcher consumes. Because the owner is the sole filter
//! that publishes the IRR transition, `on_result` keys **only** on
//! `openai_agentic_loop.action`; the dispatchers publish no action of
//! their own.
//!
//! ```yaml
//! filter: iterative_request_router
//! initial_step: inference
//! max_iterations: 11
//! steps:
//!   - name: inference
//!     filters:
//!       - filter: openai_web_search_dispatch
//!         provider: brave
//!         api_key: ${WEB_SEARCH_API_KEY}
//!       - filter: openai_mcp_dispatch
//!       - filter: openai_file_search_dispatch
//!       - filter: openai_agentic_loop
//!         max_infer_iters: 10
//!       - filter: openai_responses_proxy
//!       - filter: router
//!         routes:
//!           - cluster: model-backend
//!       - filter: load_balancer
//!         clusters:
//!           - name: model-backend
//!             endpoints: ["127.0.0.1:3001"]
//!     on_result:
//!       - filter: openai_agentic_loop
//!         key: action
//!         value: loop
//!         next: inference
//!       - default: true
//!         done: true
//! ```
//!
//! # State dependency
//!
//! Requires [`ResponsesState`] in request extensions. Without it
//! the filter passes through silently. State is created by
//! `openai_responses_request` for every Responses API create
//! request.

pub(crate) mod budget;
mod config;
#[cfg(feature = "openai-agentic-loop")]
mod controller;

use praxis_filter::{FilterError, HttpFilterContext, PipelineExtension, RequestExtensions, parse_filter_config};

use self::config::{AgenticLoopConfig, build_config};
use super::state::ResponsesState;

/// The smallest loop budget reachable by a listener, installed before any
/// request filter runs. The serving pipeline derives it from its nested IRR
/// steps, so validation and rehydration use the same limit as the loop.
#[derive(Clone, Copy)]
pub struct AgenticBudgetPolicy {
    /// Effective byte ceiling for every reachable loop on this listener.
    max_retained_bytes: usize,
}

impl AgenticBudgetPolicy {
    /// Read the validated limit from one `openai_agentic_loop` entry.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] for an invalid loop configuration.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Self, FilterError> {
        let cfg = if config.is_null() {
            AgenticLoopConfig::default()
        } else {
            parse_filter_config("openai_agentic_loop", config)?
        };
        let cfg = build_config(cfg)?;
        Ok(Self {
            max_retained_bytes: cfg.max_retained_bytes.get(),
        })
    }

    /// The effective request-wide byte limit.
    #[must_use]
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }

    /// Largest raw create body admitted before the request JSON is parsed.
    #[must_use]
    pub const fn max_request_body_bytes(self) -> usize {
        self.max_retained_bytes / budget::INPUT_WIRE_MULTIPLIER
    }

    /// Conservatively charge a JSON body before parsing it in another filter.
    #[must_use]
    pub fn input_charge(bytes: &[u8]) -> Option<usize> {
        budget::input_charge(bytes)
    }

    /// Check a filter's temporary parse and added values against live headroom.
    #[must_use]
    pub fn has_body_headroom(self, ctx: &HttpFilterContext<'_>, bytes: &[u8], extra_charge: usize) -> bool {
        let Some(charge) = Self::input_charge(bytes).and_then(|charge| charge.checked_add(extra_charge)) else {
            return false;
        };
        match ctx.extensions.get::<ResponsesState>() {
            Some(state) => state
                .simple_budget
                .is_some_and(|budget| budget.remaining_bytes().is_some_and(|remaining| charge <= remaining)),
            None => charge <= self.max_retained_bytes,
        }
    }

    /// Reserve a filter's rewritten body when Responses state already exists.
    /// Before state creation the request filter rechecks the resulting body.
    pub fn reserve_body_projection(self, ctx: &mut HttpFilterContext<'_>, bytes: &[u8], extra_charge: usize) -> bool {
        let Some(charge) = Self::input_charge(bytes).and_then(|charge| charge.checked_add(extra_charge)) else {
            return false;
        };
        match ctx.extensions.get_mut::<ResponsesState>() {
            Some(state) => state
                .simple_budget
                .as_mut()
                .is_some_and(|budget| budget.reserve_additional_input(charge)),
            None => charge <= self.max_retained_bytes,
        }
    }

    /// Largest buffered inference response admitted by the core router.
    #[must_use]
    pub const fn max_irr_response_bytes(self) -> usize {
        self.max_retained_bytes / budget::IRR_RESPONSE_DIVISOR
    }

    /// Lower this policy to another reachable loop's configured limit.
    #[must_use]
    pub const fn min(self, other: Self) -> Self {
        Self {
            max_retained_bytes: if self.max_retained_bytes < other.max_retained_bytes {
                self.max_retained_bytes
            } else {
                other.max_retained_bytes
            },
        }
    }
}

impl PipelineExtension for AgenticBudgetPolicy {
    fn prepare(&self, extensions: &mut RequestExtensions) {
        extensions.insert(*self);
    }
}

#[cfg(feature = "openai-agentic-loop")]
pub use controller::AgenticLoopFilter;
