// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Structural-complexity guard for managed Responses initialization (#1618).
//!
//! Initialization materializes memory proportional to the request's structural
//! size — normalized input, the Arc-shared working/persistence histories, and
//! persistence capture — not to its serialized byte length. A body of many tiny
//! objects (e.g. thousands of empty input items) amplifies far beyond its wire
//! size, so a byte cap alone is insufficient. This guard counts structural nodes
//! (object members, array entries, and scalars) over the already-parsed request
//! and rejects a managed create before that expansion runs.
//!
//! It deliberately protects *subsequent expansion*, not the initial parser
//! allocation: the body is already parsed (and bounded by `max_body_bytes` /
//! `MAX_JSON_BODY_BYTES`) by the time this runs. It complements, rather than
//! replaces, the optional agentic-loop retained-payload budget (#1475/#1483),
//! which is only present when an agentic-loop filter is configured; this guard
//! fires for every managed create, including chains without that filter.
//!
//! Traversal is depth-first over borrowed `serde_json::Value` with an explicit
//! iterator stack, so it bounds auxiliary memory by nesting depth (never by
//! breadth) and stops promptly once a limit is exceeded — it never clones.

use std::num::NonZeroUsize;

#[cfg(any(feature = "store", feature = "openai-mcp-tools"))]
use praxis_filter::{FilterAction, HttpFilterContext};
use serde_json::Value;

/// Metadata key the managed request filter publishes the resolved node limit
/// under, so the alternate initialization paths (rehydration, MCP state creation)
/// enforce the same bound and cannot bypass the guard.
pub(crate) const MAX_INPUT_NODES_METADATA: &str = "openai_responses.max_input_nodes";

/// Default structural-node ceiling for a managed create request.
///
/// A 64 MiB body (`MAX_JSON_BODY_BYTES`) of two-byte array entries parses to
/// ~32M nodes; initialization retains each as a normalized item across the
/// working and persistence histories plus provenance, so the per-node cost is on
/// the order of hundreds of bytes. One million nodes bounds that initial
/// expansion to a few hundred MiB while leaving ample headroom for legitimate
/// multi-thousand-message conversations (tens to low-hundreds of thousands of
/// nodes). Operators tune it with `max_input_nodes`.
pub(crate) const DEFAULT_MAX_INPUT_NODES: NonZeroUsize = NonZeroUsize::new(1_000_000).unwrap();

/// Maximum nesting depth traversed before rejection.
///
/// Bounds the auxiliary iterator stack and rejects pathologically deep input.
/// `serde_json` already rejects beyond its own recursion limit at parse time, so
/// this is a defensive secondary bound well above any legitimate request shape.
const MAX_DEPTH: usize = 256;

/// Why a request was rejected by the structural-complexity guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ComplexityReason {
    /// Structural node count exceeded the configured `max_input_nodes`.
    Nodes,
    /// Nesting depth exceeded [`MAX_DEPTH`].
    Depth,
}

impl ComplexityReason {
    /// A bounded, request-content-free rejection message.
    #[must_use]
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::Nodes => "request structural complexity exceeds the configured limit",
            Self::Depth => "request nesting depth exceeds the supported limit",
        }
    }
}

/// Borrowed container iterator; the traversal stack holds at most `MAX_DEPTH` of
/// these, bounding auxiliary memory by nesting depth rather than breadth.
enum Frame<'a> {
    /// Remaining members of an object (values only; keys are not charged).
    Object(serde_json::map::Iter<'a>),
    /// Remaining entries of an array.
    Array(std::slice::Iter<'a, Value>),
}

/// A borrowed container iterator for `value`, or `None` for a scalar.
fn container_frame(value: &Value) -> Option<Frame<'_>> {
    match value {
        Value::Object(map) => Some(Frame::Object(map.iter())),
        Value::Array(items) => Some(Frame::Array(items.iter())),
        _ => None,
    }
}

/// Reject when `root` exceeds `max_nodes` structural nodes or [`MAX_DEPTH`].
///
/// Counts object members and array entries (plus their scalar leaves); object
/// keys are not charged. Returns `None` when within budget. Depth-first with an
/// explicit stack: O(nodes) time with prompt early-exit, O(depth) auxiliary
/// memory, no cloning. A scalar root materializes nothing proportional to
/// structure, so it is always accepted.
pub(crate) fn rejects(root: &Value, max_nodes: usize) -> Option<ComplexityReason> {
    let mut stack: Vec<Frame<'_>> = vec![container_frame(root)?];
    let mut nodes: usize = 0;
    while let Some(frame) = stack.last_mut() {
        let next = match frame {
            Frame::Object(iter) => iter.next().map(|(_, value)| value),
            Frame::Array(iter) => iter.next(),
        };
        let Some(child) = next else {
            stack.pop();
            continue;
        };
        nodes = nodes.saturating_add(1);
        if nodes > max_nodes {
            return Some(ComplexityReason::Nodes);
        }
        if let Some(frame) = container_frame(child) {
            if stack.len() >= MAX_DEPTH {
                return Some(ComplexityReason::Depth);
            }
            stack.push(frame);
        }
    }
    None
}

/// Enforce the structural-complexity bound on an alternate initialization path
/// (rehydration, MCP state creation) using the limit the managed request filter
/// published, defaulting to [`DEFAULT_MAX_INPUT_NODES`] if absent. Returns the
/// bounded 413 rejection when exceeded.
#[cfg(any(feature = "store", feature = "openai-mcp-tools"))]
pub(crate) fn enforce(ctx: &HttpFilterContext<'_>, parsed: &Value) -> Result<(), FilterAction> {
    // A `NonZeroUsize` parse rejects a `0` (or malformed) published value without a
    // separate runtime check, falling back to the default.
    let limit = ctx
        .get_metadata(MAX_INPUT_NODES_METADATA)
        .and_then(|value| value.parse::<NonZeroUsize>().ok())
        .unwrap_or(DEFAULT_MAX_INPUT_NODES);
    if let Some(reason) = rejects(parsed, limit.get()) {
        return Err(FilterAction::Reject(super::error::responses_error_rejection(
            413,
            "invalid_request_error",
            reason.message(),
        )));
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, reason = "tests")]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn within_budget_passes() {
        let value = json!({"model": "m", "input": [{"role": "user", "content": "hi"}]});
        assert_eq!(rejects(&value, DEFAULT_MAX_INPUT_NODES.get()), None);
    }

    #[test]
    fn scalar_root_is_never_rejected() {
        assert_eq!(rejects(&json!(42), 0), None);
        assert_eq!(rejects(&json!("x"), 0), None);
    }

    #[test]
    fn many_small_objects_exceed_node_budget_even_when_bytes_are_small() {
        // 50 tiny entries; a small node budget rejects on structure, not bytes.
        let value = json!({"input": (0..50).map(|_| json!({})).collect::<Vec<_>>()});
        assert_eq!(rejects(&value, 10), Some(ComplexityReason::Nodes));
        // The same body passes under a generous budget.
        assert_eq!(rejects(&value, DEFAULT_MAX_INPUT_NODES.get()), None);
    }

    #[test]
    fn node_budget_boundary_is_inclusive() {
        // One object member = exactly one node.
        let value = json!({"a": 1});
        assert_eq!(rejects(&value, 1), None, "exactly at the limit passes");
        assert_eq!(
            rejects(&value, 0),
            Some(ComplexityReason::Nodes),
            "one over the limit rejects"
        );
    }

    #[test]
    fn excessive_depth_is_rejected_with_bounded_memory() {
        // Build nesting deeper than MAX_DEPTH without recursion.
        let mut value = json!("leaf");
        for _ in 0..(MAX_DEPTH + 10) {
            value = Value::Array(vec![value]);
        }
        assert_eq!(rejects(&value, usize::MAX), Some(ComplexityReason::Depth));
    }
}
