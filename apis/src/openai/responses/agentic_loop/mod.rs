// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Agentic loop controller for the Responses API pipeline.
//!
//! Manages the inference loop lifecycle: iteration counting,
//! tool-choice reset, exit conditions, and the loop/done signal
//! that branch chains read to decide whether to re-enter inference.
//!
//! Does **not** classify tool calls by type or execute them —
//! classification is handled by `tool_parse`; execution by
//! sub-filters routed via branch chains.
//!
//! # Loop control
//!
//! Writes `filter_results` during `on_request` — the only phase
//! where Praxis evaluates branch chain conditions:
//!   - `agentic_loop.action = "loop"` — tool calls present, loop back
//!   - `agentic_loop.action = "done"` — exit to client
//!
//! On the first pass through the pipeline, `tool_calls` is empty and
//! the filter sets `action = "done"`. After the response body is
//! processed by upstream filters (e.g. `stream_events`, `tool_parse`)
//! and tool calls are written to `state.tool_calls`, the branch chain
//! re-enters the pipeline. On re-entry `on_request` runs again, sees
//! the populated `tool_calls`, and sets `action = "loop"`.
//!
//! A branch chain with `on_result` matching `agentic_loop.action =
//! loop` and `rejoin` targeting the `responses_proxy` filter drives
//! the re-entry loop. The branch chain's `max_iterations` provides
//! an infrastructure-level safety cap, while this filter's
//! `max_infer_iters` provides application-level semantics (marking
//! the response as `incomplete` when the limit is reached).
//!
//! ```yaml
//! - filter: responses_proxy
//!   name: inference
//! - filter: agentic_loop
//!   max_infer_iters: 10
//!   branch_chains:
//!     - name: tool-loop
//!       on_result:
//!         filter: agentic_loop
//!         key: action
//!         result: loop
//!       rejoin: inference
//!       max_iterations: 15
//!       chains:
//!         - name: tool-execution
//!           filters:
//!             - filter: headers
//!               request_add:
//!                 - name: X-Agentic-Loop
//!                   value: "true"
//! ```
//!
//! # State dependency
//!
//! Requires [`ResponsesState`] in request extensions. Without it
//! the filter passes through silently. State is created by
//! `openai_responses_validate` for every Responses API create
//! request.

mod config;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use serde_json::json;
use tracing::{debug, trace};

use self::config::{AgenticLoopConfig, build_config};
use super::state::ResponsesState;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter results key for the loop control action.
const FILTER_RESULT_KEY: &str = "agentic_loop";

/// Action value signalling a loop-back to `responses_proxy`.
const ACTION_LOOP: &str = "loop";

/// Action value signalling loop exit.
const ACTION_DONE: &str = "done";

/// Metadata key for the response status set on incomplete exits.
const META_STATUS: &str = "responses.status";

// -----------------------------------------------------------------------------
// AgenticLoopFilter
// -----------------------------------------------------------------------------

/// Agentic loop controller for the Responses API pipeline.
///
/// Reads tool calls from [`ResponsesState`] during `on_request`,
/// checks exit conditions, manages iteration state, and writes
/// `filter_results` to control branch chain looping. Branch chains
/// only evaluate conditions after `on_request`, so all loop control
/// must happen in this phase.
///
/// # YAML
///
/// ```yaml
/// filter: agentic_loop
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: agentic_loop
/// max_infer_iters: 10
/// ```
///
/// # Example
///
/// ```rust
/// use praxis_ai_apis::openai::AgenticLoopFilter;
///
/// let yaml = serde_yaml::Value::Null;
/// let filter = AgenticLoopFilter::from_config(&yaml).unwrap();
/// assert_eq!(filter.name(), "agentic_loop");
/// ```
pub struct AgenticLoopFilter {
    /// Parsed and validated configuration.
    config: AgenticLoopConfig,
}

impl AgenticLoopFilter {
    /// Create from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config contains unknown
    /// fields or invalid values.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: AgenticLoopConfig = if config.is_null() {
            AgenticLoopConfig::default()
        } else {
            parse_filter_config("agentic_loop", config)?
        };
        let validated = build_config(cfg)?;
        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for AgenticLoopFilter {
    fn name(&self) -> &'static str {
        "agentic_loop"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let Some(mut state) = ctx.extensions.remove::<ResponsesState>() else {
            return Ok(FilterAction::Continue);
        };

        let tool_calls = std::mem::take(&mut state.tool_calls);

        let result = if tool_calls.is_empty() {
            trace!("no tool calls, signaling done");
            set_done(ctx)
        } else if let Some(action) = check_exit_conditions(&state, &self.config) {
            ctx.set_metadata(META_STATUS, "incomplete");
            set_action(ctx, action)?;
            Ok(FilterAction::Continue)
        } else {
            state.iteration += 1;
            if state.iteration > 1 {
                state.tool_choice = json!("auto");
            }
            debug!(
                iteration = state.iteration,
                tool_calls = tool_calls.len(),
                "tool calls present, signaling loop"
            );
            set_action(ctx, ACTION_LOOP)?;
            Ok(FilterAction::Continue)
        };

        ctx.extensions.insert(state);
        result
    }
}

// -----------------------------------------------------------------------------
// Exit Condition Checks
// -----------------------------------------------------------------------------

/// Check whether the loop should exit early due to length,
/// iteration limits, or the request-level `max_tool_calls` cap.
/// Returns the action to set if exiting.
fn check_exit_conditions(state: &ResponsesState, config: &AgenticLoopConfig) -> Option<&'static str> {
    if is_finish_reason_length(state) {
        debug!("finish_reason is length, exiting loop as incomplete");
        return Some(ACTION_DONE);
    }
    if let Some(max) = state.max_tool_calls
        && state.iteration >= max
    {
        debug!(
            iteration = state.iteration,
            max_tool_calls = max,
            "request-level max_tool_calls reached, exiting loop as incomplete"
        );
        return Some(ACTION_DONE);
    }
    if state.iteration >= config.max_infer_iters {
        debug!(
            iteration = state.iteration,
            max = config.max_infer_iters,
            "config iteration limit reached, exiting loop as incomplete"
        );
        return Some(ACTION_DONE);
    }
    None
}

/// Check whether the response finished due to length limit.
fn is_finish_reason_length(state: &ResponsesState) -> bool {
    state
        .response_object
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s == "incomplete")
        || state
            .response_object
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|r| r == "max_output_tokens")
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Shorthand: set `action = "done"` and return `Continue`.
fn set_done(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
    set_action(ctx, ACTION_DONE)?;
    Ok(FilterAction::Continue)
}

/// Write the loop control action to filter results.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    let results = ctx.filter_results.entry(FILTER_RESULT_KEY).or_default();
    results.set("action", action)?;
    Ok(())
}
