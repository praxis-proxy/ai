// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages web-search loop support.

use std::borrow::Cow;

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{CONTENT_TYPE, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, NextIterationBody,
    Rejection, parse_filter_config,
};
use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Value, json};

use crate::web_search::{
    SEARCH_UNAVAILABLE, SearchClient, SearchContextSize, SearchOutcome, WebSearchFilterConfig, build_config,
    format_search_results,
};

/// Registry name and filter-results namespace.
const FILTER_NAME: &str = "anthropic_web_search";
/// IRR action that re-enters the inference step.
const ACTION_LOOP: &str = "loop";
/// IRR action that returns the current response to the client.
const ACTION_DONE: &str = "done";
/// IRR accumulator entry holding the latest serialized Messages request.
const REQUEST_ACCUMULATOR_KEY: &str = "anthropic_web_search.request";
/// Maximum UTF-8 size accepted for a server-managed search query.
const MAX_SEARCH_QUERY_BYTES: usize = 8 * 1024;

/// Server-owned search call classified from the accounted previous response.
#[derive(Debug)]
struct PendingSearch {
    /// Anthropic tool-use identifier matched by the result block.
    id: String,
    /// Search query supplied by the model.
    query: String,
}

/// Classification of a buffered Messages response.
enum ResponseDecision {
    /// Return the response to the client unchanged.
    Done,
    /// Execute this server-owned search and re-enter inference.
    Managed(PendingSearch),
    /// Reject a malformed server-owned search call.
    InvalidManagedCall,
    /// Reject a server-owned search call whose query is too large.
    QueryTooLong,
}

/// Initial request fields inspected without materializing the full payload.
#[derive(Deserialize)]
struct RequestEnvelope {
    /// Whether the client requested streaming.
    stream: Option<bool>,
}

/// A borrowed JSON string or an ignored value of another type.
#[derive(Deserialize)]
#[serde(untagged)]
enum TextField<'a> {
    /// Borrowed string field, allocating only when JSON escaping requires it.
    Text(#[serde(borrow)] Cow<'a, str>),
    /// Field with a non-string value.
    Other(IgnoredAny),
}

impl TextField<'_> {
    /// Return the string value when this field is a JSON string.
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value.as_ref()),
            Self::Other(_) => None,
        }
    }
}

/// Borrowed input object for a candidate managed search.
#[derive(Deserialize)]
struct SearchInput<'a> {
    /// Candidate search query.
    #[serde(borrow)]
    query: Option<TextField<'a>>,
}

/// A search input object or an ignored value of another type.
#[derive(Deserialize)]
#[serde(untagged)]
enum InputField<'a> {
    /// Parsed input object.
    Input(#[serde(borrow)] SearchInput<'a>),
    /// Input with a non-object value.
    Other(IgnoredAny),
}

/// Borrowed fields from one response content block.
#[derive(Deserialize)]
struct ResponseBlock<'a> {
    /// Content block type.
    #[serde(rename = "type", borrow)]
    kind: Option<TextField<'a>>,
    /// Tool name.
    #[serde(borrow)]
    name: Option<TextField<'a>>,
    /// Tool-use identifier.
    #[serde(borrow)]
    id: Option<TextField<'a>>,
    /// Tool input.
    #[serde(borrow)]
    input: Option<InputField<'a>>,
}

/// A response content object or an ignored value of another type.
#[derive(Deserialize)]
#[serde(untagged)]
enum ContentField<'a> {
    /// Parsed content block.
    Block(#[serde(borrow)] ResponseBlock<'a>),
    /// Non-object content value.
    Other(IgnoredAny),
}

/// Response fields inspected before deciding whether IRR should loop.
#[derive(Deserialize)]
struct ResponseEnvelope<'a> {
    /// Anthropic object type.
    #[serde(rename = "type", borrow)]
    kind: Option<TextField<'a>>,
    /// Message role.
    #[serde(borrow)]
    role: Option<TextField<'a>>,
    /// Stop reason.
    #[serde(borrow)]
    stop_reason: Option<TextField<'a>>,
    /// Message content blocks.
    #[serde(borrow)]
    content: Option<Vec<ContentField<'a>>>,
}

/// Executes server-owned `WebSearch` tool calls in an Anthropic Messages loop.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_web_search
/// provider: you
/// api_key: ${WEB_SEARCH_API_KEY}
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_web_search
/// provider: you
/// api_key: ${WEB_SEARCH_API_KEY}
/// default_context_size: medium
/// timeout_ms: 10000
/// max_body_bytes: 67108864
/// ```
///
/// # Live demo YAML
///
/// ```yaml
/// # cargo run -p praxis-test-utils --example anthropic_messages_web_search_mock
/// # WEB_SEARCH_API_KEY="$WEB_SEARCH_API_KEY" cargo run -p praxis-ai-proxy -- \
/// #   -c examples/configs/anthropic/messages-web-search.yaml
/// # curl http://127.0.0.1:8080/v1/messages \
/// #   -H 'content-type: application/json' \
/// #   -d '{"model":"openai/gpt-oss-20b","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":"Use web search to look up potato, then summarize in one sentence."}],"tools":[{"name":"WebSearch","description":"Search the web","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}'
/// ```
pub struct AnthropicWebSearchFilter {
    /// Result-count hint passed to the provider.
    default_context_size: SearchContextSize,
    /// Maximum request and response body size buffered by the loop.
    max_body_bytes: usize,
    /// Shared provider client used for You.com callouts.
    search_client: SearchClient,
}

impl AnthropicWebSearchFilter {
    /// Create a filter with an isolated subrequest client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the filter configuration is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let client =
            crate::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(4, None));
        Self::build(config, client)
    }

    /// Create a filter with the server's shared subrequest client.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] when the filter configuration is invalid.
    pub fn from_config_with_client(
        config: &serde_yaml::Value,
        client: crate::subrequest::SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Self::build(config, client)
    }

    /// Build the filter around the supplied subrequest client.
    fn build(
        config: &serde_yaml::Value,
        client: crate::subrequest::SubRequestClient,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let config: WebSearchFilterConfig = parse_filter_config(FILTER_NAME, config)?;
        let validated = build_config(FILTER_NAME, &config)?;
        let search_client = SearchClient::from_config(FILTER_NAME, &validated, client)?;
        Ok(Box::new(Self {
            default_context_size: validated.default_context_size,
            max_body_bytes: validated.max_body_bytes,
            search_client,
        }))
    }

    /// Execute one pending call, returning the provider outcome.
    ///
    /// A provider failure never rejects the Messages response: the caller
    /// appends a truthful `is_error` tool result so the loop can continue.
    async fn execute_pending_search(&self, pending: &PendingSearch) -> SearchOutcome {
        self.search_client
            .search(&pending.query, Some(self.default_context_size))
            .await
    }

    /// Execute a retained search and replace the IRR request body.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps accounted state access and bounded body replacement adjacent"
    )]
    async fn handle_reentry(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(iteration_state) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable during re-entry"
            )));
        };
        let request_bytes = iteration_state
            .accumulator
            .get(REQUEST_ACCUMULATOR_KEY)
            .unwrap_or(&iteration_state.original_request.body);
        let Some(previous_response) = iteration_state.previous_response.as_ref() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: previous IRR response unavailable during re-entry"
            )));
        };

        let (pending, assistant_content) = managed_search_from_response(&previous_response.body)?;

        let mut request: Value = match serde_json::from_slice(request_bytes) {
            Ok(value) => value,
            Err(error) => {
                return Err(FilterError::from(format!(
                    "{FILTER_NAME}: retained request parsing failed: {error}"
                )));
            },
        };
        if request.get("messages").and_then(Value::as_array).is_none() {
            return Ok(FilterAction::Reject(anthropic_rejection(
                400,
                "invalid_request_error",
                "messages must be an array for web search re-entry",
            )));
        }
        let outcome = self.execute_pending_search(&pending).await;
        if let Err(rejection) = append_search_turns(&mut request, assistant_content, pending, &outcome) {
            return Ok(FilterAction::Reject(rejection));
        }
        let rebuilt = serde_json::to_vec(&request)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))?;
        if rebuilt.len() > self.max_body_bytes {
            return Ok(FilterAction::Reject(anthropic_rejection(
                413,
                "invalid_request_error",
                "web search request exceeds configured max_body_bytes",
            )));
        }
        let rebuilt = Bytes::from(rebuilt);
        let iteration_state = ctx.extensions.get_mut::<IterationState>().ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable while retaining request"
            ))
        })?;
        // These `Bytes` clones share one allocation across the accounted state,
        // the next iteration, and the active step body.
        iteration_state
            .accumulator
            .insert(REQUEST_ACCUMULATOR_KEY.to_owned(), rebuilt.clone());
        ctx.extensions.insert(NextIterationBody(rebuilt.clone()));
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
        *body = Some(rebuilt);
        Ok(FilterAction::Continue)
    }
}

#[async_trait]
impl HttpFilter for AnthropicWebSearchFilter {
    fn name(&self) -> &'static str {
        "anthropic_web_search"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if ctx
            .extensions
            .get::<IterationState>()
            .and_then(|state| state.previous_response.as_ref())
            .is_some()
        {
            return self.handle_reentry(ctx, body).await;
        }

        let Some(bytes) = body.as_deref() else {
            return Ok(FilterAction::Continue);
        };
        let request: RequestEnvelope = match serde_json::from_slice(bytes) {
            Ok(value) => value,
            Err(_) => return Ok(FilterAction::Continue),
        };
        if request.stream == Some(true) {
            return Ok(FilterAction::Reject(anthropic_rejection(
                400,
                "invalid_request_error",
                "streaming is not supported with anthropic_web_search",
            )));
        }

        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if !is_success_response(ctx) {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }

        let decision = body.as_deref().map_or(ResponseDecision::Done, classify_response);

        match decision {
            ResponseDecision::Done => set_action(ctx, ACTION_DONE)?,
            ResponseDecision::Managed(_) => set_action(ctx, ACTION_LOOP)?,
            ResponseDecision::InvalidManagedCall => {
                return Ok(FilterAction::Reject(anthropic_rejection(
                    400,
                    "invalid_request_error",
                    "WebSearch tool use requires a non-empty id and input.query",
                )));
            },
            ResponseDecision::QueryTooLong => {
                return Ok(FilterAction::Reject(anthropic_rejection(
                    400,
                    "invalid_request_error",
                    "WebSearch input.query must not exceed 8192 bytes",
                )));
            },
        }
        Ok(FilterAction::Continue)
    }
}

/// Whether the current upstream response may contain a managed call.
fn is_success_response(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_none_or(|response| response.status.is_success())
}

/// Select a sole, well-formed server-owned search call.
#[expect(
    clippy::too_many_lines,
    reason = "validates one small external JSON envelope linearly"
)]
fn classify_response(response_bytes: &[u8]) -> ResponseDecision {
    let Ok(response) = serde_json::from_slice::<ResponseEnvelope<'_>>(response_bytes) else {
        return ResponseDecision::Done;
    };
    let stop_reason = response.stop_reason.as_ref().and_then(TextField::as_str);
    if response.kind.as_ref().and_then(TextField::as_str) != Some("message")
        || response.role.as_ref().and_then(TextField::as_str) != Some("assistant")
        // vLLM's Messages-compatible endpoint currently labels otherwise
        // valid tool-use responses as `end_turn`.
        || !matches!(stop_reason, Some("tool_use" | "end_turn"))
    {
        return ResponseDecision::Done;
    }
    let Some(content) = response.content.as_deref() else {
        return ResponseDecision::Done;
    };
    let mut tools = content.iter().filter_map(|field| match field {
        ContentField::Block(block) if block.kind.as_ref().and_then(TextField::as_str) == Some("tool_use") => {
            Some(block)
        },
        ContentField::Block(_) | ContentField::Other(_) => None,
    });
    let Some(tool) = tools.next() else {
        return ResponseDecision::Done;
    };
    if tools.next().is_some() {
        return ResponseDecision::Done;
    }
    if tool.name.as_ref().and_then(TextField::as_str) != Some("WebSearch") {
        return ResponseDecision::Done;
    }
    let Some(id) = tool
        .id
        .as_ref()
        .and_then(TextField::as_str)
        .filter(|value| !value.is_empty())
    else {
        return ResponseDecision::InvalidManagedCall;
    };
    let Some(query) = tool
        .input
        .as_ref()
        .and_then(|input| match input {
            InputField::Input(input) => input.query.as_ref().and_then(TextField::as_str),
            InputField::Other(_) => None,
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ResponseDecision::InvalidManagedCall;
    };
    if query.len() > MAX_SEARCH_QUERY_BYTES {
        return ResponseDecision::QueryTooLong;
    }
    let id = id.to_owned();
    let query = query.to_owned();
    ResponseDecision::Managed(PendingSearch { id, query })
}

/// Recover the managed call and complete content from the accounted response.
fn managed_search_from_response(response_bytes: &[u8]) -> Result<(PendingSearch, Vec<Value>), FilterError> {
    let ResponseDecision::Managed(pending) = classify_response(response_bytes) else {
        return Err(FilterError::from(format!(
            "{FILTER_NAME}: previous response no longer contains a managed WebSearch call"
        )));
    };
    let mut response: Value = serde_json::from_slice(response_bytes).map_err(|error| {
        FilterError::from(format!(
            "{FILTER_NAME}: previous response parsing failed during re-entry: {error}"
        ))
    })?;
    let assistant_content = response
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: previous response content unavailable during re-entry"
            ))
        })?;
    Ok((pending, assistant_content))
}

/// Append the assistant tool call and matching user result block.
fn append_search_turns(
    request: &mut Value,
    assistant_content: Vec<Value>,
    pending: PendingSearch,
    outcome: &SearchOutcome,
) -> Result<(), Rejection> {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return Err(anthropic_rejection(
            400,
            "invalid_request_error",
            "messages must be an array for web search re-entry",
        ));
    };
    let PendingSearch { id, query: _ } = pending;
    let mut assistant_turn = serde_json::Map::new();
    assistant_turn.insert("role".to_owned(), Value::String("assistant".to_owned()));
    assistant_turn.insert("content".to_owned(), Value::Array(assistant_content));
    messages.push(Value::Object(assistant_turn));
    messages.push(build_tool_result_turn(&id, outcome));
    if request.get("tool_choice").is_some()
        && let Some(object) = request.as_object_mut()
    {
        object.insert("tool_choice".to_owned(), json!({"type":"auto"}));
    }
    Ok(())
}

/// Build the user turn carrying the search tool result.
///
/// A provider failure yields a truthful `is_error` result carrying the bounded
/// [`SEARCH_UNAVAILABLE`] message so the loop continues; a successful empty
/// search reports `No search results found.` without `is_error`.
fn build_tool_result_turn(tool_use_id: &str, outcome: &SearchOutcome) -> Value {
    match outcome {
        SearchOutcome::Results(results) => {
            let content = if results.is_empty() {
                "No search results found.".to_owned()
            } else {
                format_search_results(results)
            };
            json!({"role":"user","content":[{
                "type":"tool_result","tool_use_id":tool_use_id,"content":content
            }]})
        },
        SearchOutcome::Failed => json!({"role":"user","content":[{
            "type":"tool_result","tool_use_id":tool_use_id,"content":SEARCH_UNAVAILABLE,"is_error":true
        }]}),
    }
}

/// Publish the loop decision for the IRR transition table.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_NAME)
        .or_default()
        .set("action", action)?;
    Ok(())
}

/// Build an Anthropic JSON error response.
fn anthropic_rejection(status: u16, error_type: &str, message: &str) -> Rejection {
    Rejection::status(status)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(super::wire::error_body(error_type, message, None)))
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::needless_raw_strings,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;
