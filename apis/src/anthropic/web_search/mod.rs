// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Anthropic Messages web-search loop support.

mod streaming;

use std::{borrow::Cow, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, NextIterationBody,
    Rejection, StreamTerminationCause, SubRequestResponseMode, parse_filter_config,
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
/// The unified example serves both buffered (`stream: false`) and streaming
/// (`stream: true`) clients from one pipeline via `terminal_streaming: true`.
///
/// ```yaml
/// # cargo run -p praxis-test-utils --example anthropic_messages_web_search_mock
/// # WEB_SEARCH_API_KEY="$WEB_SEARCH_API_KEY" cargo run -p praxis-ai-proxy -- \
/// #   -c examples/configs/anthropic/full-flow-agentic.yaml
/// # curl http://127.0.0.1:8080/v1/messages \
/// #   -H 'content-type: application/json' \
/// #   -d '{"model":"openai/gpt-oss-20b","max_tokens":1024,"stream":false,"messages":[{"role":"user","content":"Use web search to look up potato, then summarize in one sentence."}],"tools":[{"name":"WebSearch","description":"Search the web","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}'
/// ```
pub struct AnthropicWebSearchFilter {
    /// Result-count hint passed to the provider.
    default_context_size: SearchContextSize,
    /// Maximum request and response body size buffered by the loop.
    max_body_bytes: usize,
    /// Whether an effective `stream: true` Messages request may use Praxis's
    /// streaming subrequest transport to deliver the terminal response
    /// incrementally across IRR rounds.
    terminal_streaming: bool,
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
            terminal_streaming: validated.terminal_streaming,
            search_client,
        }))
    }

    /// Align the Praxis subrequest response transport with the outbound body.
    ///
    /// Only meaningful under `terminal_streaming`: an effective `stream: true`
    /// request selects the streaming transport so the terminal Messages
    /// response reaches the client incrementally, while a non-streaming request
    /// keeps the buffered transport. The buffered loop leaves the default mode
    /// untouched.
    fn apply_streaming_transport(&self, ctx: &mut HttpFilterContext<'_>, streaming: bool) {
        if !self.terminal_streaming {
            return;
        }
        let mode = if streaming {
            SubRequestResponseMode::Streaming
        } else {
            SubRequestResponseMode::Buffered
        };
        ctx.set_subrequest_response_mode(mode);
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
        // A streamed round leaves the IRR buffered response empty, so recover the
        // managed call from the logical stream's reconstruction and reset it for
        // the upcoming round. A buffered round has no logical stream and falls
        // back to the accounted previous response below.
        let reconstructed = ctx
            .extensions
            .get_mut::<streaming::LogicalStream>()
            .and_then(|logical| {
                let reconstructed = logical.take_reconstructed();
                logical.begin_round();
                reconstructed
            });

        let Some(iteration_state) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable during re-entry"
            )));
        };
        let request_bytes = iteration_state
            .accumulator
            .get(REQUEST_ACCUMULATOR_KEY)
            .unwrap_or(&iteration_state.original_request.body);

        let (pending, assistant_content) = if let Some(reconstructed) = reconstructed.as_deref() {
            managed_search_from_response(reconstructed)?
        } else {
            let Some(previous_response) = iteration_state.previous_response.as_ref() else {
                return Err(FilterError::from(format!(
                    "{FILTER_NAME}: previous IRR response unavailable during re-entry"
                )));
            };
            managed_search_from_response(&previous_response.body)?
        };

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
        // Every round talks to the backend with the caller's original transport
        // intent so the terminal round can be streamed the moment it arrives.
        let streaming = request.get("stream").and_then(Value::as_bool).unwrap_or(false);
        self.apply_streaming_transport(ctx, streaming);
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

    /// Forward one terminal-streaming response chunk and drive the loop decision.
    ///
    /// The cross-round [`streaming::LogicalStream`] lives in `ctx.extensions` so
    /// it survives IRR re-entry: text frames are forwarded incrementally, the
    /// managed `WebSearch` block is suppressed, and a single terminal
    /// `message_delta` / `message_stop` lifecycle is emitted only when the loop
    /// finishes. Every framing failure is fail-closed to one terminal `error`
    /// event so no raw upstream bytes leak into the transformed stream.
    fn on_streaming_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // A transport-level failure (connect/IO fault, circuit trip, admission or
        // idle timeout, deadline, or byte-ceiling breach) surfaces as a Praxis
        // stream termination during the completion hook (empty body, end of
        // stream). Convert it into one terminal error event and mark it handled;
        // otherwise the router discards the completion output and the client
        // stream ends in an abrupt EOF.
        if let Some(cause) = ctx.stream_termination().map(praxis_filter::StreamTermination::cause) {
            return handle_stream_termination(ctx, body, cause);
        }

        // Only a 2xx, identity-encoded upstream stream is a Messages SSE
        // lifecycle we can transform. A non-2xx status or a content-encoded body
        // cannot be parsed as UTF-8 Messages events.
        //
        // The IRR streaming body phase leaves `ctx.response_header` unset, so the
        // status and encoding cannot be re-read here. `on_response` inspects the
        // still-populated header once per round and records `UntransformableRound`
        // when the round is not a transformable lifecycle; declining on that
        // marker keeps a non-2xx or compressed round — including the initial
        // round, whose raw status and body must pass through unchanged — from
        // being parsed as events.
        if ctx.extensions.get::<UntransformableRound>().is_some() {
            return decline_untransformable_round(ctx, body, end_of_stream);
        }

        let reentry = router_reentry(ctx);
        let chunk = body.take();
        let logical = ctx
            .extensions
            .get_or_insert_with(|| streaming::LogicalStream::new(self.max_body_bytes, self.max_body_bytes));
        if logical.has_failed() {
            // The stream already failed closed: drop remaining upstream bytes so
            // exactly one terminal error reaches the client.
            *body = None;
            return Ok(FilterAction::Continue);
        }

        let (output, action) =
            drive_streaming_chunk(logical, chunk.as_deref().unwrap_or_default(), end_of_stream, reentry);
        *body = (!output.is_empty()).then(|| Bytes::from(output));
        if let Some(action) = action {
            set_action(ctx, action)?;
        }
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
        // Terminal streaming rewrites the SSE body incrementally; the buffered
        // loop only inspects the accumulated response.
        if self.terminal_streaming {
            BodyAccess::ReadWrite
        } else {
            BodyAccess::ReadOnly
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        // A streaming-capable response pipeline may not use `StreamBuffer`; the
        // terminal serializer delivers chunks as they arrive.
        if self.terminal_streaming {
            BodyMode::Stream
        } else {
            BodyMode::StreamBuffer {
                max_bytes: Some(self.max_body_bytes),
            }
        }
    }

    fn may_select_streaming_subrequest_response(&self) -> bool {
        self.terminal_streaming
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Strip client-negotiated content coding so every backend round returns
        // identity-encoded bytes. A compressed SSE stream cannot be parsed into
        // Messages events, and a compressed buffered body cannot be classified;
        // stripping here keeps both the streaming and buffered loops working.
        ctx.request_headers_to_remove.push(ACCEPT_ENCODING);
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
        let streaming = request.stream == Some(true);
        if streaming && !self.terminal_streaming {
            return Ok(FilterAction::Reject(anthropic_rejection(
                400,
                "invalid_request_error",
                "streaming is not supported with anthropic_web_search",
            )));
        }
        self.apply_streaming_transport(ctx, streaming);

        Ok(FilterAction::Continue)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // The IRR runs the response-header phase once per streaming round with
        // `ctx.response_header` populated, then clears it before the streaming
        // body phase. Record here whether this round can be transformed as a
        // Messages SSE lifecycle so the body phase — which no longer sees the
        // status or encoding — declines an untransformable round instead of
        // parsing non-SSE bytes as events. The marker is re-evaluated every
        // round, so a success round clears any marker left by a prior one.
        if self.terminal_streaming {
            if !is_success_response(ctx) || response_is_encoded(ctx) {
                ctx.extensions.insert(UntransformableRound);
            } else {
                ctx.extensions.remove::<UntransformableRound>();
            }
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        // A streamed round (effective `stream: true` under `terminal_streaming`)
        // is transformed incrementally; a buffered round keeps the accumulate-
        // then-classify path below.
        if self.terminal_streaming && ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming {
            return self.on_streaming_response_body(ctx, body, end_of_stream);
        }

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

/// Marks the current streaming round as untransformable (non-2xx status or
/// content-encoded body), recorded in the response-header phase.
///
/// The IRR streaming body phase leaves `ctx.response_header` unset, so the
/// status and encoding cannot be re-checked there. [`on_response`] inspects the
/// still-populated header once per round and records this marker when the round
/// cannot be parsed as a Messages SSE lifecycle; the streaming body phase reads
/// it to decline the round instead of parsing non-SSE bytes as events.
///
/// [`on_response`]: AnthropicWebSearchFilter::on_response
struct UntransformableRound;

/// Whether the current upstream response may contain a managed call.
///
/// Reads `ctx.response_header`, which the buffered response path populates but
/// the IRR streaming body phase leaves unset; an absent header is treated as a
/// success so a streamed round is transformed rather than declined.
fn is_success_response(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_none_or(|response| response.status.is_success())
}

/// Whether the upstream response carries a `Content-Encoding` header.
///
/// Like [`is_success_response`], this reads `ctx.response_header`; it reports
/// `false` when the header is absent (including the IRR streaming body phase).
fn response_is_encoded(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_some_and(|response| response.headers.contains_key(CONTENT_ENCODING))
}

/// Handle a non-2xx or content-encoded upstream response on the streaming path.
///
/// Round 0 (no started [`streaming::LogicalStream`]): the client has not yet
/// seen any transformed SSE bytes, so the raw upstream response is passed
/// through untouched and the loop ends — the upstream status and body are the
/// whole client-visible response.
///
/// A later round (a stream that has already forwarded `message_start`): the
/// client is mid-stream on a committed 200 SSE lifecycle, so raw non-2xx or
/// compressed bytes cannot be forwarded without corrupting it. The stream fails
/// closed to one terminal `error` event, drops the untransformable upstream
/// body, and ends the loop.
fn decline_untransformable_round(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> Result<FilterAction, FilterError> {
    if let Some(logical) = ctx.extensions.get_mut::<streaming::LogicalStream>()
        && logical.has_started()
    {
        let terminal = (!logical.has_failed()).then(|| {
            logical.fail();
            streaming::error_event_bytes(&streaming::StreamError::UpstreamUnprocessable)
        });
        *body = terminal.map(Bytes::from);
        set_action(ctx, ACTION_DONE)?;
        return Ok(FilterAction::Continue);
    }
    if end_of_stream {
        set_action(ctx, ACTION_DONE)?;
    }
    Ok(FilterAction::Continue)
}

/// Map a Praxis stream-termination cause to the fail-closed [`streaming::StreamError`]
/// forwarded as the terminal `error` event.
///
/// A transport-level deadline reuses the deadline message so the client sees a
/// consistent reason whether the deadline lapses before re-entry or mid-stream;
/// every other abnormal cause folds into [`streaming::StreamError::UpstreamTerminated`].
/// The arms are explicit so a new Praxis cause forces a compile error here rather
/// than silently mapping to a generic message.
fn termination_stream_error(cause: StreamTerminationCause) -> streaming::StreamError {
    match cause {
        StreamTerminationCause::DeadlineExceeded => streaming::StreamError::DeadlineExceeded,
        StreamTerminationCause::AdmissionTimeout
        | StreamTerminationCause::CircuitOpen
        | StreamTerminationCause::Connect
        | StreamTerminationCause::IdleTimeout
        | StreamTerminationCause::Io
        | StreamTerminationCause::Filter
        | StreamTerminationCause::ResponseTooLarge => streaming::StreamError::UpstreamTerminated,
    }
}

/// Convert an abnormal Praxis stream termination into one terminal `error` event
/// and mark it handled so the router forwards the completion bytes.
///
/// The IRR runs the completion body hook with an empty body at `end_of_stream`
/// after inserting a [`StreamTermination`]; unless the filter both emits a
/// terminal sequence and calls [`mark_stream_termination_handled`], the router
/// discards the completion output and the client sees an abrupt EOF.
///
/// A logical stream that already failed closed keeps its single terminal (no
/// second `error` event); a termination before any bytes still emits one
/// terminal so the client learns why the stream ended.
///
/// [`StreamTermination`]: praxis_filter::StreamTermination
/// [`mark_stream_termination_handled`]: HttpFilterContext::mark_stream_termination_handled
fn handle_stream_termination(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    cause: StreamTerminationCause,
) -> Result<FilterAction, FilterError> {
    let error = termination_stream_error(cause);
    let already_terminal = ctx
        .extensions
        .get_mut::<streaming::LogicalStream>()
        .is_some_and(|logical| {
            let failed = logical.has_failed();
            logical.fail();
            failed
        });
    *body = (!already_terminal).then(|| Bytes::from(streaming::error_event_bytes(&error)));
    ctx.mark_stream_termination_handled();
    set_action(ctx, ACTION_DONE)?;
    Ok(FilterAction::Continue)
}

/// Whether the router can open another inference round, and if not, why.
///
/// The IRR owns the open-next step and exposes no terminal-failure hook, so a
/// re-entry it would refuse surfaces as an abrupt EOF. A managed round detects
/// the deterministic refusal conditions here and terminates with a coherent
/// error instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reentry {
    /// The router has iteration and deadline headroom to open the next round.
    Available,
    /// Re-entering would meet the router's iteration ceiling.
    IterationCeiling,
    /// The router's deadline has elapsed; opening the next round would fail.
    DeadlineExceeded,
}

/// Classify whether the router can open another inference round after this one.
///
/// Re-entering inference produces the next round at `iteration + 1`; the IRR
/// checks the iteration ceiling before the deadline when opening a step, so this
/// mirrors that precedence: an exhausted ceiling reports [`Reentry::IterationCeiling`]
/// even when the deadline has also elapsed. The deadline check mirrors the IRR's
/// `open_step`, which refuses to open once no time remains.
fn reentry_from_state(iteration: u32, max_iterations: u32, deadline: Instant, now: Instant) -> Reentry {
    if iteration.saturating_add(1) >= max_iterations {
        Reentry::IterationCeiling
    } else if deadline.checked_duration_since(now).unwrap_or_default().is_zero() {
        Reentry::DeadlineExceeded
    } else {
        Reentry::Available
    }
}

/// Resolve the router's re-entry capacity from the current IRR state.
///
/// Absent IRR state (a non-routed unit context) the loop keeps its existing
/// behavior and re-enters.
fn router_reentry(ctx: &HttpFilterContext<'_>) -> Reentry {
    ctx.extensions
        .get::<IterationState>()
        .map_or(Reentry::Available, |state| {
            reentry_from_state(
                state.iteration(),
                state.max_iterations(),
                state.deadline(),
                Instant::now(),
            )
        })
}

/// Drive one terminal-streaming chunk through the cross-round logical stream.
///
/// Returns the client-visible bytes to forward and, once the round completes at
/// `end_of_stream`, the IRR action to publish. Every framing failure is folded
/// into a single terminal `error` event so no raw upstream bytes leak downstream.
///
/// `reentry` reports whether the router can open the next round. A managed round
/// that wants to re-enter when the router would refuse is terminated with a
/// coherent error rather than an [`ACTION_LOOP`] the IRR would fail to open —
/// which would otherwise abort the client stream with an abrupt EOF. The two
/// deterministic refusals are detected proactively: the iteration ceiling maps
/// to [`streaming::StreamError::IterationLimit`] and an elapsed deadline to
/// [`streaming::StreamError::DeadlineExceeded`]. A transport-level failure that
/// aborts an already-committed round's body — a connect/IO fault, circuit trip,
/// admission or idle timeout, deadline, or byte-ceiling breach — instead surfaces
/// as a Praxis stream termination and is converted into a terminal error by
/// [`handle_stream_termination`]. The residual gap in praxis 0.5.4 is the failure
/// to *open* the next round after an [`ACTION_LOOP`]: that transition commits no
/// response body, so it runs no completion hook and still ends the client stream
/// in an abrupt EOF — the IRR driver owns the open-next step and exposes no
/// terminal-failure hook, the same upstream limitation documented for the OpenAI
/// Responses loop.
fn drive_streaming_chunk(
    logical: &mut streaming::LogicalStream,
    chunk: &[u8],
    end_of_stream: bool,
    reentry: Reentry,
) -> (Vec<u8>, Option<&'static str>) {
    let mut output = Vec::new();
    let forwarded = match logical.on_chunk(chunk, end_of_stream) {
        Ok(forwarded) => forwarded,
        Err(error) => {
            output.extend_from_slice(&streaming::error_event_bytes(&error));
            return (output, Some(ACTION_DONE));
        },
    };
    output.extend_from_slice(&forwarded);
    if !end_of_stream {
        return (output, None);
    }
    let (terminal, action) = finish_streaming_round(logical, reentry);
    output.extend_from_slice(&terminal);
    (output, Some(action))
}

/// Resolve a completed round into terminal bytes and the IRR action to publish.
///
/// A round that wants to re-enter when the router would refuse the next step is
/// turned into a coherent terminal error rather than an [`ACTION_LOOP`] the IRR
/// would fail to open: an iteration ceiling maps to
/// [`streaming::StreamError::IterationLimit`] and an elapsed deadline to
/// [`streaming::StreamError::DeadlineExceeded`]. Every framing failure is
/// likewise folded into a single terminal `error` event.
fn finish_streaming_round(logical: &mut streaming::LogicalStream, reentry: Reentry) -> (Vec<u8>, &'static str) {
    match logical.finish_round() {
        Ok(streaming::FinishOutcome {
            action: streaming::RoundAction::Loop,
            ..
        }) => match reentry {
            Reentry::Available => (Vec::new(), ACTION_LOOP),
            Reentry::IterationCeiling => (
                streaming::error_event_bytes(&streaming::StreamError::IterationLimit),
                ACTION_DONE,
            ),
            Reentry::DeadlineExceeded => (
                streaming::error_event_bytes(&streaming::StreamError::DeadlineExceeded),
                ACTION_DONE,
            ),
        },
        Ok(streaming::FinishOutcome {
            action: streaming::RoundAction::Done,
            terminal,
        }) => (terminal, ACTION_DONE),
        Err(error) => (streaming::error_event_bytes(&error), ACTION_DONE),
    }
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
