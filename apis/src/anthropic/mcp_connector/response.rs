// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Response-side handling of the Anthropic MCP connector.
//!
//! Classifies each buffered model round, turns executed MCP calls into the
//! next request's `tool_use` / `tool_result` turns, and rebuilds the final
//! client response with Anthropic's `mcp_tool_use` / `mcp_tool_result` blocks.

use serde_json::{Map, Value, json};

use super::request::{MCP_TOOL_RESULT_TYPE, MCP_TOOL_USE_TYPE};
use crate::{
    json_body::serialized_len,
    mcp_core::{
        admission::McpAdmissionDecisions,
        execute::{McpCallIdentity, McpCallOutcome},
        naming::{McpToolIndex, McpToolMatch},
    },
};

// -----------------------------------------------------------------------------
// Call Result
// -----------------------------------------------------------------------------

/// Fixed serialization overhead of one result in both representations.
const RESULT_ENVELOPE_BYTES: usize = 192;

/// Result of one executed MCP call.
#[derive(Debug)]
pub(super) struct McpResult {
    /// Model-emitted `tool_use` id this result answers.
    pub tool_use_id: String,
    /// Normalized MCP output, or the proxy error message.
    pub output: String,
    /// Whether the call failed.
    pub is_error: bool,
    /// Whether a size ceiling replaced the result.
    limit_exceeded: bool,
}

impl McpCallOutcome for McpResult {
    /// The output is held here, then copied into the model `tool_result` and
    /// the client `mcp_tool_result`.
    const PAYLOAD_OWNER_COUNT: usize = 3;

    fn success(call: &McpCallIdentity<'_>, output: &str, is_error: bool) -> Self {
        Self {
            tool_use_id: call.call_id.to_owned(),
            output: output.to_owned(),
            is_error,
            limit_exceeded: false,
        }
    }

    fn error(call: &McpCallIdentity<'_>, message: &str) -> Self {
        Self {
            tool_use_id: call.call_id.to_owned(),
            output: message.to_owned(),
            is_error: true,
            limit_exceeded: false,
        }
    }

    fn retained_bytes(&self) -> Option<usize> {
        let output = serialized_len(self.output.as_str()).ok()?;
        let id = serialized_len(self.tool_use_id.as_str()).ok()?;
        output
            .checked_add(id)?
            .checked_mul(2)?
            .checked_add(RESULT_ENVELOPE_BYTES)
    }

    fn limit_exceeded(&self) -> bool {
        self.limit_exceeded
    }

    fn mark_limit_exceeded(&mut self) {
        self.limit_exceeded = true;
    }
}

impl McpResult {
    /// The `tool_result` block sent back to the model.
    pub(super) fn model_block(&self) -> Value {
        let mut block = Map::with_capacity(4);
        block.insert("type".to_owned(), Value::String("tool_result".to_owned()));
        block.insert("tool_use_id".to_owned(), Value::String(self.tool_use_id.clone()));
        block.insert("content".to_owned(), Value::String(self.output.clone()));
        if self.is_error {
            block.insert("is_error".to_owned(), Value::Bool(true));
        }
        Value::Object(block)
    }

    /// The `mcp_tool_result` block returned to the client.
    pub(super) fn client_block(self) -> Value {
        json!({
            "type": MCP_TOOL_RESULT_TYPE,
            "tool_use_id": self.tool_use_id,
            "is_error": self.is_error,
            "content": [{"type": "text", "text": self.output}],
        })
    }
}

// -----------------------------------------------------------------------------
// Round Classification
// -----------------------------------------------------------------------------

/// Whether a `tool_use` name belongs to the resolved MCP tools.
///
/// A call to a policy-denied MCP tool is owned too, so dispatch rejects it
/// explicitly instead of returning it to the client as a client tool.
pub(super) fn is_mcp_owned(name: &str, index: &McpToolIndex<'_>, admission: &McpAdmissionDecisions) -> bool {
    index.contains(name) || admission.is_denied_encoded(name)
}

/// Whether a buffered round must re-enter inference to run MCP calls.
///
/// A round loops when it is an assistant message that stopped for tool use
/// and calls at least one MCP tool. A `max_tokens` stop may carry a truncated
/// call, so it never loops.
pub(super) fn round_calls_mcp(response: &Value, index: &McpToolIndex<'_>, admission: &McpAdmissionDecisions) -> bool {
    if response.get("type").and_then(Value::as_str) != Some("message")
        || response.get("role").and_then(Value::as_str) != Some("assistant")
        // vLLM's Messages endpoint labels some tool-use responses `end_turn`.
        || !matches!(
            response.get("stop_reason").and_then(Value::as_str),
            Some("tool_use" | "end_turn")
        )
    {
        return false;
    }
    content_blocks(response).iter().any(|block| {
        block.get("type").and_then(Value::as_str) == Some("tool_use")
            && block
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| is_mcp_owned(name, index, admission))
    })
}

/// Borrow a response's `content` array, or an empty slice.
pub(super) fn content_blocks(response: &Value) -> &[Value] {
    response
        .get("content")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Client tool calls left in a round after MCP calls are removed.
pub(super) fn has_client_tool_use(
    content: &[Value],
    index: &McpToolIndex<'_>,
    admission: &McpAdmissionDecisions,
) -> bool {
    content.iter().any(|block| {
        block.get("type").and_then(Value::as_str) == Some("tool_use")
            && !block
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| is_mcp_owned(name, index, admission))
    })
}

// -----------------------------------------------------------------------------
// Client Blocks
// -----------------------------------------------------------------------------

/// Rewrite one round's assistant content for the client.
///
/// Every MCP `tool_use` becomes an `mcp_tool_use` with its original tool and
/// server name, immediately followed by its `mcp_tool_result`. Other blocks,
/// including client tool calls, are kept in order.
#[expect(clippy::too_many_lines, reason = "one ordered pass over the round content")]
pub(super) fn client_content(
    content: &[Value],
    index: &McpToolIndex<'_>,
    admission: &McpAdmissionDecisions,
    results: Vec<McpResult>,
) -> Vec<Value> {
    let mut results: std::collections::HashMap<String, McpResult> = results
        .into_iter()
        .map(|result| (result.tool_use_id.clone(), result))
        .collect();
    let mut blocks = Vec::with_capacity(content.len().saturating_add(results.len()));
    for block in content {
        let owned_name = (block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .then(|| block.get("name").and_then(Value::as_str))
            .flatten()
            .filter(|name| is_mcp_owned(name, index, admission));
        let Some(encoded_name) = owned_name else {
            blocks.push(block.clone());
            continue;
        };
        let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
        let (server_name, tool_name) = match index.get(encoded_name) {
            Some(McpToolMatch::Unique { key, .. }) => (key.0.as_str(), key.1.as_str()),
            _ => ("unknown", encoded_name),
        };
        blocks.push(json!({
            "type": MCP_TOOL_USE_TYPE,
            "id": id,
            "name": tool_name,
            "server_name": server_name,
            "input": block.get("input").cloned().unwrap_or_else(|| Value::Object(Map::new())),
        }));
        if let Some(result) = results.remove(id) {
            blocks.push(result.client_block());
        }
    }
    blocks
}

/// Token usage carried across MCP rounds.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct RoundUsage {
    /// Input tokens of completed rounds.
    pub input_tokens: u64,
    /// Output tokens of completed rounds.
    pub output_tokens: u64,
}

impl RoundUsage {
    /// Add one round's `usage` object.
    pub(super) fn add(&mut self, response: &Map<String, Value>) {
        let usage = response.get("usage");
        let tokens = |field: &str| usage.and_then(|usage| usage.get(field)).and_then(Value::as_u64);
        self.input_tokens = self.input_tokens.saturating_add(tokens("input_tokens").unwrap_or(0));
        self.output_tokens = self.output_tokens.saturating_add(tokens("output_tokens").unwrap_or(0));
    }

    /// Fold earlier rounds into a response's `usage`.
    pub(super) fn apply(self, response: &mut Map<String, Value>) {
        let Some(Value::Object(usage)) = response.get_mut("usage") else {
            return;
        };
        for (field, earlier) in [
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ] {
            let current = usage.get(field).and_then(Value::as_u64).unwrap_or(0);
            usage.insert(field.to_owned(), Value::from(current.saturating_add(earlier)));
        }
    }
}

/// Build the client response: earlier rounds' blocks, then this round's
/// content, with usage summed over every round.
pub(super) fn final_response(mut response: Map<String, Value>, earlier: Vec<Value>, usage: RoundUsage) -> Value {
    let mut content = earlier;
    if let Some(Value::Array(current)) = response.remove("content") {
        content.extend(current);
    }
    response.insert("content".to_owned(), Value::Array(content));
    usage.apply(&mut response);
    Value::Object(response)
}
