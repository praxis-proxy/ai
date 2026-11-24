// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Protocol-neutral execution of model-emitted MCP tool calls.
//!
//! A protocol adapter extracts each model-emitted call into a borrowed
//! [`McpToolCall`], then runs a batch through [`execute_mcp_calls`]. Execution
//! re-enforces operator admission, resolves the encoded name to exactly one
//! tool-map entry, parses and schema-validates the arguments, performs the
//! bounded `tools/call`, and normalizes the MCP content into one output string.
//!
//! Wire shapes stay with the adapter: every result, including admission
//! denials, malformed arguments, and size-limit fallbacks, is built through
//! the adapter's [`McpCallOutcome`] implementation, so the Responses adapter
//! can emit `function_call_output` plus `mcp_call` while the Anthropic adapter
//! emits `tool_result` plus `mcp_tool_result`.

use std::time::Duration;

use futures::{FutureExt as _, future::join_all};
use serde_json::Value;
use tracing::{debug, warn};

use super::{
    admission::McpAdmissionDecisions,
    fingerprint::{is_connector_tool_entry, target_fingerprint},
    naming::{McpToolIndex, McpToolMatch},
    validate,
};
use crate::{callout_identity::McpCalloutIdentity, json_body::serialized_len, mcp_client};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Minimum retained reservation needed for one bounded tool-error result.
pub(crate) const MIN_RETAINED_RESULT_BYTES: usize = 1_024;

/// Stable internal reason for a result that exceeded its admitted content cap.
pub(crate) const MCP_RESULT_TOO_LARGE: &str = "MCP tool result exceeded the configured per-result byte limit";

/// Longest model-supplied identity echoed by the fixed size-limit fallback.
const MAX_FALLBACK_IDENTITY_BYTES: usize = 64;

// -----------------------------------------------------------------------------
// Call and Result Contracts
// -----------------------------------------------------------------------------

/// One model-emitted tool call, borrowed from the adapter's wire shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct McpToolCall<'a> {
    /// Model-facing encoded tool name, or `None` when the call carried none.
    pub encoded_name: Option<&'a str>,
    /// Correlation id echoed in every result for this call.
    pub call_id: &'a str,
    /// Correlation id echoed by the fixed size-limit fallback, which must fit
    /// the minimum retained reservation.
    pub fallback_call_id: &'a str,
    /// Raw arguments: a JSON-encoded string or an inline JSON value.
    pub arguments: Option<&'a Value>,
    /// Approval that authorized this call, threaded into its result.
    pub approval_request_id: Option<&'a str>,
}

/// Identity of one executed or rejected call, passed to result builders.
#[derive(Clone, Copy, Debug)]
pub(crate) struct McpCallIdentity<'a> {
    /// Correlation id of the model-emitted call.
    pub call_id: &'a str,
    /// Resolved server label, or `"unknown"` before resolution.
    pub server_label: &'a str,
    /// Original MCP tool name, or the encoded name before resolution.
    pub tool_name: &'a str,
    /// Canonical argument string, or empty when arguments were not parsed.
    pub arguments: &'a str,
    /// Approval that authorized this call.
    pub approval_request_id: Option<&'a str>,
}

/// Adapter-owned result of one MCP call.
pub(crate) trait McpCallOutcome: Sized + Send {
    /// Maximum simultaneous copies of one result payload while the adapter
    /// builds its representations. The per-call wire limit is the retained
    /// reservation divided by this count.
    const PAYLOAD_OWNER_COUNT: usize;

    /// Build the result of a completed `tools/call`. `is_error` mirrors the
    /// MCP `isError` flag.
    fn success(call: &McpCallIdentity<'_>, output: &str, is_error: bool) -> Self;

    /// Build a proxy-generated tool error.
    fn error(call: &McpCallIdentity<'_>, message: &str) -> Self;

    /// Serialized bytes the adapter retains for this result.
    fn retained_bytes(&self) -> Option<usize>;

    /// Whether a transport or result ceiling failed before a complete result
    /// was kept.
    fn limit_exceeded(&self) -> bool;

    /// Flag the result as a size-limit replacement.
    fn mark_limit_exceeded(&mut self);
}

/// A result batch exceeded its configured retained-byte ceiling.
#[derive(Debug)]
pub(crate) struct McpResultLimitExceeded;

/// Controls execution of one homogeneous MCP call batch.
#[derive(Clone, Copy)]
pub(crate) struct McpExecutionOptions<'a> {
    /// Whether independent calls may execute concurrently.
    pub parallel: bool,
    /// Maximum number of calls concurrently in flight.
    pub max_parallel_calls: usize,
    /// Maximum raw bytes accepted for one MCP result event.
    pub max_result_bytes: usize,
    /// Maximum serialized bytes retained by one result batch.
    pub max_total_result_bytes: usize,
    /// Timeout applied independently to each MCP call.
    pub timeout: Duration,
    /// Names reserved for trusted forwarding, including when values are absent.
    pub forwarded_header_names: &'a [http::HeaderName],
    /// Trusted request headers selected by operator configuration.
    pub forwarded_headers: Option<&'a http::HeaderMap>,
    /// Request-scoped context injected only for configured connector entries.
    pub connector_identity: Option<&'a McpCalloutIdentity>,
    /// Per-execution pool of initialized MCP sessions reused across rounds.
    pub session_pool: &'a mcp_client::McpSessionPool,
    /// Namespace unique to the dispatcher whose transport configuration opened
    /// the session.
    pub pool_namespace: mcp_client::McpPoolNamespace,
    /// Operator admission decisions, re-enforced at dispatch so a policy-denied
    /// or ineligible call cannot execute even if it reaches this point (#1724).
    pub admission: &'a McpAdmissionDecisions,
    /// Optional request-wide transport and parse admission for this call.
    pub transport_budget: Option<mcp_client::McpBudgetedCallLimits>,
}

// -----------------------------------------------------------------------------
// Batch Execution
// -----------------------------------------------------------------------------

/// Derive the payload ceiling represented by one retained-result reservation.
pub(crate) fn result_payload_limit<R: McpCallOutcome>(retained_result_limit: usize) -> usize {
    retained_result_limit / R::PAYLOAD_OWNER_COUNT
}

/// Execute MCP tool calls — concurrently when `parallel` is true,
/// sequentially otherwise.
///
/// # Errors
///
/// Returns [`McpResultLimitExceeded`] when the batch cannot reserve a bounded
/// result for every call, or when a budgeted call exceeded its limit.
pub(crate) async fn execute_mcp_calls<R: McpCallOutcome>(
    mcp_calls: &[McpToolCall<'_>],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Result<Vec<R>, McpResultLimitExceeded> {
    let minimum_reservation = mcp_calls
        .len()
        .checked_mul(MIN_RETAINED_RESULT_BYTES)
        .ok_or(McpResultLimitExceeded)?;
    if options.max_result_bytes < MIN_RETAINED_RESULT_BYTES
        || options.max_total_result_bytes < minimum_reservation
        || options.max_parallel_calls == 0
    {
        return Err(McpResultLimitExceeded);
    }
    let per_call_limit = options
        .max_result_bytes
        .min(options.max_total_result_bytes / mcp_calls.len().max(1));
    let bounded_options = McpExecutionOptions {
        max_result_bytes: per_call_limit,
        ..options
    };
    if options.parallel {
        execute_parallel(mcp_calls, tool_index, bounded_options, callout).await
    } else {
        execute_sequential(mcp_calls, tool_index, bounded_options, callout).await
    }
}

/// Execute MCP tool calls concurrently within the owning request future.
///
/// Avoid detached tasks: dropping the request must also cancel every pending
/// external side effect. Panics are converted to per-call errors so one faulty
/// future does not discard successful siblings.
#[expect(clippy::too_many_lines, reason = "batch execution and ordered outcome handling")]
async fn execute_parallel<R: McpCallOutcome>(
    mcp_calls: &[McpToolCall<'_>],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Result<Vec<R>, McpResultLimitExceeded> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    let mut remaining_calls = mcp_calls;
    while !remaining_calls.is_empty() {
        let chunk_size = remaining_calls.len().min(options.max_parallel_calls);
        let (chunk, rest) = remaining_calls.split_at(chunk_size);
        remaining_calls = rest;
        let futures = chunk.iter().map(|call| {
            std::panic::AssertUnwindSafe(execute_single_call::<R>(call, tool_index, &options, callout)).catch_unwind()
        });
        for (call, outcome) in chunk.iter().zip(join_all(futures).await) {
            let result = match outcome {
                Ok(Some(result)) => result,
                Ok(None) => {
                    warn!(tool = ?call.encoded_name, "parallel MCP call returned None, emitting error");
                    error_result_for_dropped_call(call, "internal error: call produced no result")
                },
                Err(_panic) => {
                    warn!(tool = ?call.encoded_name, "parallel MCP call future panicked, emitting error");
                    error_result_for_dropped_call(call, "internal error: call future panicked")
                },
            };
            let result = fit_result_or_limit_error(call, result, options.max_result_bytes)?;
            if options.transport_budget.is_some() && result.limit_exceeded() {
                return Err(McpResultLimitExceeded);
            }
            results.push(result);
        }
    }
    Ok(results)
}

/// Execute MCP tool calls sequentially, emitting error results
/// for any calls that produce no result.
async fn execute_sequential<R: McpCallOutcome>(
    mcp_calls: &[McpToolCall<'_>],
    tool_index: &McpToolIndex<'_>,
    options: McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Result<Vec<R>, McpResultLimitExceeded> {
    let mut results = Vec::with_capacity(mcp_calls.len());
    for call in mcp_calls {
        let result = if let Some(result) = execute_single_call::<R>(call, tool_index, &options, callout).await {
            result
        } else {
            warn!(tool = ?call.encoded_name, "sequential MCP call returned None, emitting error");
            error_result_for_dropped_call(call, "internal error: call produced no result")
        };
        let result = fit_result_or_limit_error(call, result, options.max_result_bytes)?;
        if options.transport_budget.is_some() && result.limit_exceeded() {
            return Err(McpResultLimitExceeded);
        }
        results.push(result);
    }
    Ok(results)
}

// -----------------------------------------------------------------------------
// Result Size Fitting
// -----------------------------------------------------------------------------

/// Replace an oversized completed result with a small truthful tool error, or
/// fail the batch terminally.
///
/// The adapter reserves at least [`MIN_RETAINED_RESULT_BYTES`] before
/// dispatch, so this fallback can always be retained after the external side
/// effect has happened. Keeping one result per executed call prevents retries
/// from being encouraged by a batch-wide proxy error. The full
/// `approval_request_id` is preserved — never truncated — because it
/// correlates the result to the approval that authorized the side effect. If
/// it cannot fit the reservation, the batch fails terminally instead of
/// retaining an oversized fallback.
pub(crate) fn fit_result_or_limit_error<R: McpCallOutcome>(
    call: &McpToolCall<'_>,
    result: R,
    retained_limit: usize,
) -> Result<R, McpResultLimitExceeded> {
    if result.retained_bytes().is_some_and(|bytes| bytes <= retained_limit) {
        return Ok(result);
    }
    bounded_result_limit_error(call, retained_limit)
}

/// Build the fixed limit-error fallback, or fail when it cannot fit its
/// call's retained reservation.
pub(crate) fn bounded_result_limit_error<R: McpCallOutcome>(
    call: &McpToolCall<'_>,
    retained_limit: usize,
) -> Result<R, McpResultLimitExceeded> {
    let mut fallback = size_limit_fallback::<R>(call);
    if fallback.retained_bytes().is_none_or(|bytes| bytes > retained_limit) {
        warn!(
            approval_request_bytes = call.approval_request_id.map_or(0, str::len),
            "MCP result-limit fallback exceeded its retained reservation; failing the batch"
        );
        return Err(McpResultLimitExceeded);
    }
    // A budgeted batch fails rather than retaining a truncated result; flag the
    // replacement so the caller can enforce that when a transport budget is set.
    fallback.mark_limit_exceeded();
    Ok(fallback)
}

/// Build the fixed size-limit fallback for `call` without checking that it
/// fits a reservation. Only bounded identity fields are echoed, so the
/// fallback size depends on the correlation ids alone.
pub(crate) fn size_limit_fallback<R: McpCallOutcome>(call: &McpToolCall<'_>) -> R {
    let tool_name = call
        .encoded_name
        .filter(|name| name.len() <= MAX_FALLBACK_IDENTITY_BYTES)
        .unwrap_or("unknown");
    R::error(
        &McpCallIdentity {
            call_id: call.fallback_call_id,
            server_label: "unknown",
            tool_name,
            arguments: "",
            approval_request_id: call.approval_request_id,
        },
        "MCP result exceeds the platform administrator's size limit.",
    )
}

/// Return the bounded fallback call id for a model-supplied id: the primary id
/// when present, otherwise a secondary id no longer than the fallback bound.
pub(crate) fn bounded_fallback_call_id<'a>(primary: Option<&'a str>, secondary: Option<&'a str>) -> &'a str {
    primary.filter(|value| !value.is_empty()).unwrap_or_else(|| {
        secondary
            .filter(|value| value.len() <= MAX_FALLBACK_IDENTITY_BYTES)
            .unwrap_or("unknown")
    })
}

/// Build an error result for a tool call that was dropped
/// (task panic, cancellation, or missing fields).
fn error_result_for_dropped_call<R: McpCallOutcome>(call: &McpToolCall<'_>, reason: &str) -> R {
    R::error(
        &McpCallIdentity {
            call_id: call.call_id,
            server_label: "unknown",
            tool_name: call.encoded_name.unwrap_or("unknown"),
            arguments: "",
            approval_request_id: call.approval_request_id,
        },
        reason,
    )
}

/// Build a schema-rejection result that fits the per-call retained-byte
/// reservation.
///
/// The validation message is trimmed on a character boundary until the
/// adapter's result fits [`MIN_RETAINED_RESULT_BYTES`], so an adversarially
/// deep schema path cannot overrun the reservation and replace the actionable
/// error with a generic size-limit fallback.
pub(crate) fn schema_rejection_result<R: McpCallOutcome>(call: &McpCallIdentity<'_>, message: &str) -> R {
    let mut fitted = message.to_owned();
    loop {
        let result = R::error(call, &fitted);
        if fitted.is_empty()
            || result
                .retained_bytes()
                .is_none_or(|bytes| bytes <= MIN_RETAINED_RESULT_BYTES)
        {
            return result;
        }
        // Trim roughly a quarter (at least one byte) on a char boundary and retry.
        let target = (fitted.len() * 3 / 4).min(fitted.len().saturating_sub(1));
        let mut end = target;
        while end > 0 && !fitted.is_char_boundary(end) {
            end -= 1;
        }
        fitted.truncate(end);
    }
}

// -----------------------------------------------------------------------------
// Single Call
// -----------------------------------------------------------------------------

/// Resolve an encoded function name to its unique entry, rejecting
/// ambiguity.
///
/// # Errors
///
/// Returns `Err(Some(result))` with a tool error for an ambiguous name, and
/// `Err(None)` when no resolved tool owns the name.
#[expect(clippy::type_complexity, reason = "key+value pair needed by callers")]
pub(crate) fn resolve_tool_entry<'a, R: McpCallOutcome>(
    tool_index: &McpToolIndex<'a>,
    encoded_name: &str,
    call_id: &str,
    approval_request_id: Option<&str>,
) -> Result<(&'a (String, String), &'a Value), Option<Box<R>>> {
    match tool_index.get(encoded_name) {
        Some(McpToolMatch::Unique { key, entry }) => Ok((key, entry)),
        Some(McpToolMatch::Ambiguous { count }) => {
            warn!(
                encoded_name,
                server_count = count,
                "ambiguous MCP tool name: multiple entries produce this encoded name"
            );
            Err(Some(Box::new(R::error(
                &McpCallIdentity {
                    call_id,
                    server_label: "unknown",
                    tool_name: encoded_name,
                    arguments: "",
                    approval_request_id,
                },
                &format!("ambiguous tool name: {count} servers expose '{encoded_name}'"),
            ))))
        },
        None => {
            warn!(encoded_name, "tool not found in mcp_tool_map, skipping");
            Err(None)
        },
    }
}

/// Normalize raw tool call arguments.
///
/// Tool calls carry arguments either as a JSON string
/// (e.g., `"{\"city\":\"Paris\"}"`) or directly as a JSON value.
/// Returns `(parsed_value, canonical_string)` or an error for
/// malformed JSON strings.
///
/// # Errors
///
/// Returns the parse error message for a malformed JSON string.
pub(crate) fn normalize_arguments(raw: &Value) -> Result<(Value, String), String> {
    match raw {
        Value::String(s) => {
            let parsed = serde_json::from_str(s).map_err(|e| format!("malformed tool arguments: {e}"))?;
            Ok((parsed, s.clone()))
        },
        other => Ok((other.clone(), other.to_string())),
    }
}

/// Parse tool call arguments, handling JSON-string encoding. Absent arguments
/// are an empty object.
///
/// # Errors
///
/// Returns a tool error result for malformed JSON-string arguments.
pub(crate) fn parse_call_arguments<R: McpCallOutcome>(
    arguments: Option<&Value>,
    call_id: &str,
    server_label: &str,
    tool_name: &str,
    approval_request_id: Option<&str>,
) -> Result<(Value, String), Box<R>> {
    let empty = Value::Object(serde_json::Map::new());
    let raw = arguments.unwrap_or(&empty);
    normalize_arguments(raw).map_err(|e| {
        warn!(tool_name, error = %e, "malformed JSON in tool call arguments");
        Box::new(R::error(
            &McpCallIdentity {
                call_id,
                server_label,
                tool_name,
                arguments: raw.as_str().unwrap_or_default(),
                approval_request_id,
            },
            &e,
        ))
    })
}

/// Build result from a completed (successful or failed) MCP call.
pub(crate) fn process_call_result<R: McpCallOutcome>(
    result: Result<rmcp::model::CallToolResult, mcp_client::McpClientError>,
    call: &McpCallIdentity<'_>,
    max_result_bytes: usize,
) -> R {
    let tool_name = call.tool_name;
    let call_id = call.call_id;
    match result {
        Ok(r) => {
            let is_error = r.is_error.unwrap_or(false);
            let content_count = r.content.len();
            let output = call_result_to_output(&r, max_result_bytes);
            drop(r);
            match output {
                Ok(output_text) => {
                    debug!(tool_name, call_id, is_error, content_count, "MCP tool call completed");
                    R::success(call, &output_text, is_error)
                },
                Err(e) => {
                    warn!(
                        tool_name, call_id, error = %e,
                        "failed to serialize MCP content blocks; returning tool error"
                    );
                    limited_error(call, &e, e == MCP_RESULT_TOO_LARGE)
                },
            }
        },
        Err(e) => {
            warn!(tool_name, call_id, error = %e, "MCP tool call failed");
            let exceeded = matches!(e, mcp_client::McpClientError::ResponseTooLarge { .. });
            limited_error(call, &e.to_string(), exceeded)
        },
    }
}

/// Build a tool error, flagging it as a size-limit replacement when the
/// failure came from a transport or result ceiling.
fn limited_error<R: McpCallOutcome>(call: &McpCallIdentity<'_>, message: &str, limit_exceeded: bool) -> R {
    let mut result = R::error(call, message);
    if limit_exceeded {
        result.mark_limit_exceeded();
    }
    result
}

/// Execute a single MCP tool call.
#[expect(
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::large_stack_frames,
    reason = "linear validation, dispatch-time admission re-enforcement, and async call"
)]
pub(crate) async fn execute_single_call<R: McpCallOutcome>(
    call: &McpToolCall<'_>,
    tool_index: &McpToolIndex<'_>,
    options: &McpExecutionOptions<'_>,
    callout: &mcp_client::McpCallout,
) -> Option<R> {
    let encoded_name = call.encoded_name?;
    let call_id = call.call_id;
    let approval_request_id = call.approval_request_id;

    // Dispatch-time admission re-enforcement (#1724): a tool denied at discovery
    // is not in the map, so reject it explicitly here rather than letting it fall
    // through as an unresolved client function.
    if let Some(reason) = options.admission.denied_reason(encoded_name) {
        warn!(encoded_name, reason = ?reason, "rejecting model-generated call to a policy-denied MCP tool");
        return Some(R::error(
            &McpCallIdentity {
                call_id,
                server_label: "unknown",
                tool_name: encoded_name,
                arguments: "",
                approval_request_id,
            },
            reason.message(),
        ));
    }

    let (key, entry) = match resolve_tool_entry::<R>(tool_index, encoded_name, call_id, approval_request_id) {
        Ok(r) => r,
        Err(opt) => return opt.map(|b| *b),
    };
    let original_tool_name = &key.1;
    // Re-verify the resolved entry against its operator-owned verdict. Defense in
    // depth: an admitted entry with no verdict, or a writable tool under operator
    // read-only mode, fails closed before any external side effect.
    if let Some(reason) = options.admission.admitted_denial(&key.0, original_tool_name) {
        warn!(encoded_name, reason = ?reason, "rejecting MCP call ineligible under operator admission policy");
        return Some(R::error(
            &McpCallIdentity {
                call_id,
                server_label: &key.0,
                tool_name: original_tool_name,
                arguments: "",
                approval_request_id,
            },
            reason.message(),
        ));
    }
    let server_url = entry.get("server_url").and_then(Value::as_str)?;
    let server_label = entry.get("server_label").and_then(Value::as_str).unwrap_or("unknown");
    let headers = entry.get("headers");
    let authorization = entry.get("authorization").and_then(Value::as_str);
    let forwarded_headers = is_connector_tool_entry(entry)
        .then_some(options.forwarded_headers)
        .flatten();
    let connector_context = is_connector_tool_entry(entry)
        .then_some(options.connector_identity)
        .flatten()
        .map(|identity| mcp_client::McpConnectorContext {
            owner: identity.owner(),
            bearer: identity.user_credential(),
            assertion: identity.authorization(),
        });
    let (arguments, arguments_string) = match parse_call_arguments::<R>(
        call.arguments,
        call_id,
        server_label,
        original_tool_name,
        approval_request_id,
    ) {
        Ok(r) => r,
        Err(r) => return Some(*r),
    };

    // Enforce the tool's advertised `inputSchema` before any outbound side
    // effect. A schema violation (wrong type, missing required field, or a field
    // forbidden by `additionalProperties: false`) is returned as a tool error so
    // the model can correct itself, and no `tools/call` is dialed (issue #1125).
    // The message is value-free, and the rejected arguments are not echoed back:
    // a malformed value may carry private content, and echoing it would also risk
    // overrunning the result-byte budget and discarding the actionable error.
    if let Err(message) = validate::validate_arguments(entry.get("tool_definition"), &arguments) {
        warn!(
            tool_name = original_tool_name,
            server_label, call_id, %message,
            "rejecting MCP tool call: arguments failed input-schema validation"
        );
        return Some(schema_rejection_result(
            &McpCallIdentity {
                call_id,
                server_label,
                tool_name: original_tool_name,
                arguments: "",
                approval_request_id,
            },
            &message,
        ));
    }

    debug!(
        tool_name = original_tool_name,
        server_label, call_id, "executing MCP tool call"
    );

    let identity = McpCallIdentity {
        call_id,
        server_label,
        tool_name: original_tool_name,
        arguments: &arguments_string,
        approval_request_id,
    };

    // The temporary decoded output is copied into every adapter representation.
    // Limiting the wire/result payload to the retained allowance divided by the
    // adapter's owner count keeps every in-flight task within its aggregate
    // reservation even at that worst-case ownership point.
    let payload_limit = result_payload_limit::<R>(options.max_result_bytes);
    if payload_limit == 0 {
        return Some(R::error(
            &identity,
            "MCP result allowance is too small to retain a result",
        ));
    }

    // The opaque key binds target identity to this dispatcher's outbound
    // pipeline, timeout, and forwarding policy. Empty fingerprints construct no
    // key and therefore remain fail-closed and unpooled.
    let result = if let Some(limits) = options.transport_budget {
        mcp_client::call_tool_with_forwarded_headers_budgeted(
            server_url,
            headers,
            authorization,
            options.forwarded_header_names,
            forwarded_headers,
            connector_context.as_ref(),
            original_tool_name,
            arguments,
            options.timeout,
            payload_limit,
            callout,
            limits,
        )
        .await
    } else {
        let session_key = mcp_client::McpPoolKey::new(options.pool_namespace, target_fingerprint(entry));
        mcp_client::call_tool_with_forwarded_headers(
            session_key.as_ref().map(|key| (options.session_pool, key)),
            server_url,
            headers,
            authorization,
            options.forwarded_header_names,
            forwarded_headers,
            connector_context.as_ref(),
            original_tool_name,
            arguments,
            options.timeout,
            payload_limit,
            callout,
        )
        .await
    };
    Some(process_call_result(result, &identity, payload_limit))
}

// -----------------------------------------------------------------------------
// Result Normalization
// -----------------------------------------------------------------------------

/// Preserve both MCP result representations when structured content is present.
/// Text duplication is only a SHOULD in MCP 2025-06-18, so `content` may be empty
/// or carry information distinct from `structuredContent`.
fn call_result_to_output(result: &rmcp::model::CallToolResult, max_result_bytes: usize) -> Result<String, String> {
    /// MCP result fields represented together in one output object.
    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    struct StructuredResult<'content> {
        /// Ordered unstructured MCP blocks.
        content: &'content [rmcp::model::ContentBlock],
        /// Structured MCP result data.
        structured_content: &'content Value,
    }

    let Some(structured_content) = &result.structured_content else {
        return content_blocks_to_output(&result.content, max_result_bytes);
    };

    let output = StructuredResult {
        content: &result.content,
        structured_content,
    };
    let output_bytes = serialized_len(&output).map_err(|e| format!("failed to size MCP result content: {e}"))?;
    if output_bytes > max_result_bytes {
        return Err(MCP_RESULT_TOO_LARGE.to_owned());
    }
    serde_json::to_string(&output).map_err(|e| format!("failed to serialize MCP result content: {e}"))
}

/// Serialize rmcp `ContentBlock` values into one lossless output string.
///
/// Text-only results keep the simple newline-joined form for readability and
/// backward compatibility. Any result carrying non-text content (image, audio,
/// embedded resource, or resource link) is serialized as the compact JSON MCP
/// content array so that no block is silently dropped.
///
/// # Errors
///
/// Returns `Err` when the content cannot be represented within
/// `max_result_bytes`, so the caller can surface an explicit tool error
/// instead of a lossy empty success.
pub(crate) fn content_blocks_to_output(
    blocks: &[rmcp::model::ContentBlock],
    max_result_bytes: usize,
) -> Result<String, String> {
    if blocks
        .iter()
        .all(|block| matches!(block, rmcp::model::ContentBlock::Text(_)))
    {
        let mut output = String::new();
        for block in blocks {
            let rmcp::model::ContentBlock::Text(text) = block else {
                unreachable!("text-only content was checked above")
            };
            let separator_bytes = usize::from(!output.is_empty());
            let next_bytes = output
                .len()
                .checked_add(separator_bytes)
                .and_then(|bytes| bytes.checked_add(text.text.len()))
                .ok_or_else(|| MCP_RESULT_TOO_LARGE.to_owned())?;
            if next_bytes > max_result_bytes {
                return Err(MCP_RESULT_TOO_LARGE.to_owned());
            }
            if separator_bytes != 0 {
                output.push('\n');
            }
            output.push_str(&text.text);
        }
        return Ok(output);
    }

    let output_bytes = serialized_len(blocks).map_err(|e| format!("failed to size MCP content blocks: {e}"))?;
    if output_bytes > max_result_bytes {
        return Err(MCP_RESULT_TOO_LARGE.to_owned());
    }
    serde_json::to_string(blocks).map_err(|e| format!("failed to serialize MCP content blocks: {e}"))
}

#[cfg(test)]
mod tests;
