// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `OpenAPI` contract schema types for Praxis-transformed Responses operations.

#![expect(
    clippy::allow_attributes,
    clippy::large_stack_frames,
    reason = "utoipa macro-generated schema builders allocate large temporary values"
)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

/// Request body accepted by `POST /responses`.
#[derive(Debug, Default, Deserialize, ToSchema)]
#[allow(dead_code, reason = "schema-only contract types for OpenAPI description generation")]
pub(super) struct CreateResponseRequest {
    /// Target model identifier.
    #[schema(example = "gpt-4o")]
    pub(super) model: Option<String>,

    /// Response input items or string prompt.
    pub(super) input: Option<Value>,

    /// System instructions.
    pub(super) instructions: Option<String>,

    /// Tools available for model execution.
    pub(super) tools: Option<Vec<Value>>,

    /// Tool selection policy.
    pub(super) tool_choice: Option<Value>,

    /// Sampling temperature.
    pub(super) temperature: Option<f64>,

    /// Nucleus sampling `top_p`.
    pub(super) top_p: Option<f64>,

    /// Whether to stream SSE response events.
    pub(super) stream: Option<bool>,

    /// Whether to store the response resource.
    pub(super) store: Option<bool>,
}

/// Response body emitted by `POST /responses`.
#[derive(Debug, Serialize, ToSchema)]
#[allow(dead_code, reason = "schema-only contract types for OpenAPI description generation")]
pub(super) struct ResponseResource {
    /// Unique response identifier.
    #[schema(example = "resp_123")]
    pub(super) id: String,

    /// Object type label. Always `response`.
    #[schema(example = "response")]
    pub(super) object: String,

    /// Generation status.
    #[schema(example = "completed")]
    pub(super) status: String,

    /// Creation timestamp in epoch seconds.
    pub(super) created_at: u64,

    /// Completion timestamp in epoch seconds.
    pub(super) completed_at: Option<u64>,

    /// Model used for response generation.
    pub(super) model: String,

    /// Generated output items.
    pub(super) output: Vec<Value>,

    /// Token usage details.
    pub(super) usage: Option<Value>,

    /// Error details when status is `failed`.
    pub(super) error: Option<Value>,

    /// Incomplete details when status is `incomplete`.
    pub(super) incomplete_details: Option<Value>,

    /// Instructions provided or inherited.
    pub(super) instructions: Option<Value>,

    /// Maximum output tokens allowed.
    pub(super) max_output_tokens: Option<u64>,

    /// Original response input items.
    pub(super) input: Option<Value>,

    /// Parallel tool calls enabled flag.
    pub(super) parallel_tool_calls: Option<bool>,

    /// Previous response ID in continuation chain.
    pub(super) previous_response_id: Option<Value>,

    /// Reasoning configuration or null.
    pub(super) reasoning: Option<Value>,

    /// Whether response is stored in response store.
    pub(super) store: Option<bool>,

    /// Sampling temperature.
    pub(super) temperature: Option<f64>,

    /// Text response format configuration.
    pub(super) text: Option<Value>,

    /// Tool selection policy.
    pub(super) tool_choice: Option<Value>,

    /// Tools available for execution.
    pub(super) tools: Option<Value>,

    /// Nucleus sampling `top_p`.
    pub(super) top_p: Option<f64>,

    /// Truncation strategy.
    pub(super) truncation: Option<String>,

    /// Metadata map.
    pub(super) metadata: Option<Value>,

    /// Whether the response executes in background mode.
    pub(super) background: Option<bool>,

    /// Service tier used for request.
    pub(super) service_tier: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::too_many_lines, reason = "straight-line struct instantiation test")]
    fn contract_types_can_be_instantiated() {
        let req = CreateResponseRequest {
            model: Some("gpt-4o".to_owned()),
            input: Some(serde_json::json!("Hello")),
            instructions: Some("You are a helpful assistant.".to_owned()),
            tools: Some(vec![]),
            tool_choice: Some(serde_json::json!("auto")),
            temperature: Some(1.0),
            top_p: Some(1.0),
            stream: Some(false),
            store: Some(false),
        };
        assert_eq!(req.model.as_deref(), Some("gpt-4o"));
        assert_eq!(req.instructions.as_deref(), Some("You are a helpful assistant."));
        assert_eq!(req.temperature, Some(1.0));
        assert_eq!(req.top_p, Some(1.0));
        assert_eq!(req.stream, Some(false));
        assert_eq!(req.store, Some(false));

        let res = ResponseResource {
            id: "resp_1".to_owned(),
            object: "response".to_owned(),
            status: "completed".to_owned(),
            created_at: 1_700_000_000,
            completed_at: Some(1_700_000_005),
            model: "gpt-4o".to_owned(),
            output: Vec::new(),
            usage: Some(serde_json::json!({"total_tokens": 10})),
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            input: None,
            parallel_tool_calls: Some(true),
            previous_response_id: None,
            reasoning: None,
            store: Some(false),
            temperature: Some(1.0),
            text: None,
            tool_choice: Some(serde_json::json!("auto")),
            tools: Some(serde_json::json!([])),
            top_p: Some(1.0),
            truncation: Some("disabled".to_owned()),
            metadata: Some(serde_json::json!({})),
            background: Some(false),
            service_tier: Some(serde_json::json!("default")),
        };
        assert_eq!(res.id, "resp_1");
        assert_eq!(res.object, "response");
        assert_eq!(res.status, "completed");
        assert_eq!(res.created_at, 1_700_000_000);
        assert_eq!(res.completed_at, Some(1_700_000_005));
        assert_eq!(res.model, "gpt-4o");
        assert_eq!(res.parallel_tool_calls, Some(true));
        assert_eq!(res.temperature, Some(1.0));
        assert_eq!(res.top_p, Some(1.0));
        assert_eq!(res.background, Some(false));
    }
}
