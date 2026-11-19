// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Executes the Anthropic Messages MCP connector inside Praxis.
//!
//! Anthropic's MCP connector (`mcp_servers` plus `mcp_toolset` tools) is run
//! by Anthropic's servers, so a vLLM or Chat Completions backend cannot honor
//! it. This filter resolves the connector into plain tools, runs the model's
//! MCP calls between inference rounds of an `iterative_request_router`, and
//! returns Anthropic's `mcp_tool_use` / `mcp_tool_result` blocks to the
//! client. The protocol is shared with the OpenAI Responses MCP filters
//! through [`crate::mcp_core`].

mod config;
mod request;
mod response;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    reason = "tests"
)]
mod tests;

use std::{borrow::Cow, collections::HashMap, sync::Arc, time::Instant};

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{ACCEPT_ENCODING, CONTENT_TYPE, HeaderName, HeaderValue};
use praxis_core::config::InsecureOptions;
use praxis_filter::{
    BodyAccess, BodyMode, ChainBindingContext, FilterAction, FilterError, FilterPipeline, HttpFilter,
    HttpFilterContext, IterationState, NextIterationBody, Rejection, SubRequestResponseMode, parse_filter_config,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tracing::debug;

use self::{
    config::{AnthropicMcpConnectorConfig, FILTER_NAME, ValidatedConfig, build_config},
    request::{ConnectorRequest, ServerDecl},
    response::{McpResult, RoundUsage},
};
use super::{
    messages_request::{MCP_CONNECTOR_METADATA_KEY, MCP_CONNECTOR_RESOLVED},
    messages_to_chat_completions::RESPONSE_RAW_BYTES_KEY,
    protocol::strip_mcp_client_betas,
    wire,
};
use crate::{
    callout_headers::effective_body_callout_headers,
    callout_identity::{McpCalloutIdentity, stage_mcp_callout_identity},
    mcp_client,
    mcp_core::{
        admission::{McpAdmissionDecisions, McpDenyReason},
        execute::{McpExecutionOptions, McpToolCall, bounded_fallback_call_id, execute_mcp_calls},
        fingerprint::{
            bind_credential_context, bind_forwarded_header_context, bind_owner_context, is_connector_tool_entry,
        },
        naming::McpToolIndex,
        tool_map::{admit_tools, insert_tools, protected_exclusion, select_forward_headers, tool_policy_for_entry},
    },
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// IRR action that re-enters the inference step.
const ACTION_LOOP: &str = "loop";

/// IRR action that returns the current response to the client.
const ACTION_DONE: &str = "done";

/// IRR accumulator entry holding the latest serialized Messages request.
const REQUEST_ACCUMULATOR_KEY: &str = "anthropic_mcp_connector.request";

/// Header carrying the caller's Anthropic beta features.
const ANTHROPIC_BETA_HEADER: HeaderName = HeaderName::from_static("anthropic-beta");

/// Client-facing reason for refusing a streamed connector request.
const STREAMING_UNSUPPORTED: &str =
    "streaming (`stream: true`) is not yet supported with the MCP connector on this gateway";

// -----------------------------------------------------------------------------
// Request State
// -----------------------------------------------------------------------------

/// Resolved connector state carried across inference rounds.
///
/// Lives in the request extensions, which the `iterative_request_router`
/// threads through every round of one client request.
struct ConnectorState {
    /// Resolved MCP tools keyed by `(server_name, tool_name)`.
    tool_map: HashMap<(String, String), Value>,
    /// Operator admission verdicts, re-enforced at dispatch.
    admission: McpAdmissionDecisions,
    /// Whether any server resolved to an operator connector.
    has_connector: bool,
    /// Client blocks from completed MCP rounds, in order.
    earlier: Vec<Value>,
    /// Token usage of completed MCP rounds.
    usage: RoundUsage,
    /// Whether the last classified round asked to run MCP calls.
    pending: bool,
}

/// Request fields read before deciding whether the body needs parsing.
#[derive(Deserialize)]
struct StreamEnvelope {
    /// Whether the client requested streaming.
    stream: Option<bool>,
}

// -----------------------------------------------------------------------------
// AnthropicMcpConnectorFilter
// -----------------------------------------------------------------------------

/// Executes the Anthropic Messages MCP connector for non-Anthropic backends.
///
/// Place it first in the inference step of an `iterative_request_router`, so
/// it rewrites the request before any translation and sees the response after
/// translation back to Anthropic Messages.
///
/// On the first round it validates `mcp_servers` and `mcp_toolset` tools,
/// lists each server's tools, keeps the ones the toolset enables, and replaces
/// the connector with plain custom tools named `{server}__{tool}`. Replayed
/// `mcp_tool_use` / `mcp_tool_result` history becomes ordinary tool turns.
/// When the model calls MCP tools, the filter runs them and loops; the final
/// response carries `mcp_tool_use` / `mcp_tool_result` blocks for every call.
///
/// A client server is matched to an operator connector by URL. Unmatched URLs
/// are rejected unless `allow_client_servers` is set. Streaming requests that
/// use the connector are rejected for now.
///
/// # YAML
///
/// ```yaml
/// filter: anthropic_mcp_connector
/// connectors:
///   - id: weather
///     server_url: https://weather-mcp.example.com/mcp
///     tool_policy: allow_all
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: anthropic_mcp_connector
/// connectors:
///   - id: weather
///     server_url: https://weather-mcp.example.com/mcp
///     tool_policy: read_only
/// allow_client_servers: false
/// tool_admission:
///   mode: protected
/// forward_headers:
///   - x-tenant-id
/// timeout_ms: 30000
/// max_servers: 10
/// max_tools: 128
/// max_body_bytes: 16777216
/// max_parallel_calls: 8
/// max_result_bytes: 1048576
/// max_total_result_bytes: 8388608
/// ```
pub struct AnthropicMcpConnectorFilter {
    /// Validated settings.
    config: ValidatedConfig,
    /// Outbound filter chain every MCP callout runs through.
    outbound: Arc<FilterPipeline>,
    /// Session-pool namespace unique to this filter instance.
    pool_namespace: mcp_client::McpPoolNamespace,
}

impl AnthropicMcpConnectorFilter {
    /// Build from YAML as a plain builtin with an empty outbound chain.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] for an invalid configuration or a configured
    /// `outbound_chain`, which needs chain-binding registration.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let raw: AnthropicMcpConnectorConfig = parse_filter_config(FILTER_NAME, config)?;
        if raw.outbound_chain.is_some() {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: outbound_chain requires chain-binding registration"
            )));
        }
        let outbound = mcp_client::build_bare_outbound_pipeline(false)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: {error}")))?;
        Self::assemble(raw, outbound)
    }

    /// Build from YAML, binding the configured `outbound_chain`.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] for an invalid configuration or an outbound
    /// chain that cannot be bound.
    pub fn from_config_with_binding(
        config: &serde_yaml::Value,
        ctx: &ChainBindingContext<'_>,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        let mut raw: AnthropicMcpConnectorConfig = parse_filter_config(FILTER_NAME, config)?;
        let outbound_chain = raw.outbound_chain.take();
        let outbound = mcp_client::bind_mcp_outbound_chain(outbound_chain, ctx, "anthropic_mcp_connector_outbound")?;
        Self::assemble(raw, outbound)
    }

    /// Build with an allow-private outbound pipeline for loopback unit tests.
    #[cfg(test)]
    fn from_config_allow_private(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let raw: AnthropicMcpConnectorConfig = parse_filter_config(FILTER_NAME, config)?;
        let outbound = mcp_client::build_bare_outbound_pipeline(true)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: {error}")))?;
        Self::assemble(raw, outbound)
    }

    /// Validate the configuration and assemble the filter.
    fn assemble(
        raw: AnthropicMcpConnectorConfig,
        outbound: Arc<FilterPipeline>,
    ) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self {
            config: build_config(raw)?,
            outbound,
            pool_namespace: mcp_client::McpPoolNamespace::new(),
        }))
    }

    /// Stage connector identity when any resolved server is a connector.
    fn stage_identity(
        &self,
        ctx: &HttpFilterContext<'_>,
        has_connector: bool,
    ) -> Result<Option<McpCalloutIdentity>, Rejection> {
        if !has_connector {
            return Ok(None);
        }
        stage_mcp_callout_identity(
            ctx,
            self.config.user_credential.as_deref(),
            self.config.authorization_assertion.as_deref(),
        )
        .map_err(|_missing| {
            wire::error_rejection(
                401,
                wire::ErrorType::Authentication,
                "the MCP connector requires caller context that was not provided",
            )
        })
    }

    /// Acquire the MCP callout or fail closed.
    fn callout(&self, ctx: &HttpFilterContext<'_>) -> Result<mcp_client::McpCallout, Rejection> {
        mcp_client::McpCallout::from_context(ctx, Arc::clone(&self.outbound)).ok_or_else(|| {
            wire::error_rejection(
                500,
                wire::ErrorType::Api,
                "no sub-request client is available for MCP callouts",
            )
        })
    }

    /// Select the operator-allowed headers forwarded to connectors.
    fn forwarded_headers(&self, ctx: &HttpFilterContext<'_>) -> http::HeaderMap {
        let effective = effective_body_callout_headers(ctx, Cow::Borrowed(&ctx.request.headers));
        select_forward_headers(&self.config.forward_headers, &effective)
    }

    // -------------------------------------------------------------------------
    // First Round
    // -------------------------------------------------------------------------

    /// Resolve the connector and rewrite the first-round request.
    #[expect(
        clippy::too_many_lines,
        reason = "parse, validate, resolve, and commit form one first-round transaction"
    )]
    async fn handle_initial(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(bytes) = body.as_deref() else {
            return Ok(FilterAction::Continue);
        };
        let streaming = serde_json::from_slice::<StreamEnvelope>(bytes)
            .ok()
            .and_then(|envelope| envelope.stream)
            .unwrap_or(false);
        apply_transport(ctx, streaming);
        if !contains_mcp_marker(bytes) {
            return Ok(FilterAction::Continue);
        }
        let Ok(Value::Object(mut object)) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(FilterAction::Continue);
        };
        let history = request::has_mcp_history(&object);
        let connector = match request::parse_connector(&object, self.config.max_servers) {
            Ok(connector) => connector,
            Err(message) => return Ok(FilterAction::Reject(wire::invalid_request_rejection(&message))),
        };
        if connector.is_none() && !history {
            return Ok(FilterAction::Continue);
        }
        if connector.is_some() && streaming {
            return Ok(FilterAction::Reject(wire::invalid_request_rejection(
                STREAMING_UNSUPPORTED,
            )));
        }
        if history && let Err(message) = request::convert_mcp_history(&mut object) {
            return Ok(FilterAction::Reject(wire::invalid_request_rejection(&message)));
        }
        if let Some(connector) = connector {
            let state = match Box::pin(self.resolve(ctx, &mut object, connector)).await {
                Ok(state) => state,
                Err(rejection) => return Ok(FilterAction::Reject(rejection)),
            };
            debug!(
                tool_count = state.tool_map.len(),
                connector = state.has_connector,
                "resolved Anthropic MCP connector"
            );
            ctx.extensions.insert(state);
            ctx.set_metadata(MCP_CONNECTOR_METADATA_KEY, MCP_CONNECTOR_RESOLVED);
            strip_beta_header(ctx);
        }
        let rebuilt = serde_json::to_vec(&object)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))?;
        if rebuilt.len() > self.config.max_body_bytes {
            return Ok(FilterAction::Reject(oversized_request()));
        }
        commit_request(ctx, body, Bytes::from(rebuilt), false);
        Ok(FilterAction::Continue)
    }

    /// List every declared server and rewrite the connector into tools.
    #[expect(
        clippy::too_many_lines,
        reason = "listing, selection, admission, and rewrite form one transaction"
    )]
    async fn resolve(
        &self,
        ctx: &HttpFilterContext<'_>,
        object: &mut Map<String, Value>,
        connector: ConnectorRequest,
    ) -> Result<ConnectorState, Rejection> {
        let entries = connector
            .servers
            .iter()
            .enumerate()
            .map(|(index, server)| request::server_entry(index, server, &self.config))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|message| wire::invalid_request_rejection(&message))?;
        let has_connector = entries.iter().any(is_connector_tool_entry);
        let identity = self.stage_identity(ctx, has_connector)?;
        let callout = self.callout(ctx)?;
        let forwarded = self.forwarded_headers(ctx);

        let listing_context = ListingContext {
            forwarded: &forwarded,
            identity: identity.as_ref(),
            callout: &callout,
        };
        let listings = futures::future::join_all(
            connector
                .servers
                .iter()
                .zip(&entries)
                .map(|(server, entry)| self.list_server_tools(server, entry, &listing_context)),
        )
        .await;

        let mut admission = McpAdmissionDecisions::new(self.config.admission_policy);
        let mut tool_map = HashMap::new();
        let mut generated = Vec::with_capacity(connector.toolsets.len());
        let mut listings: Vec<Option<Listing>> = listings.into_iter().map(Some).collect();
        for toolset in &connector.toolsets {
            let (Some(server), Some(entry), Some(listing)) = (
                connector.servers.get(toolset.server_index),
                entries.get(toolset.server_index),
                listings.get_mut(toolset.server_index).and_then(Option::take),
            ) else {
                generated.push(Vec::new());
                continue;
            };
            let tools = match listing {
                Listing::Tools(tools) => tools,
                Listing::Excluded(reason) => {
                    admission.record_excluded_namespace(&server.name, reason);
                    generated.push(Vec::new());
                    continue;
                },
                Listing::Failed => {
                    return Err(wire::error_rejection(
                        502,
                        wire::ErrorType::Api,
                        &format!(
                            "mcp_servers[{}]: the MCP server could not list its tools",
                            toolset.server_index
                        ),
                    ));
                },
            };
            let enabled: Vec<Value> = tools
                .into_iter()
                .filter(|tool| {
                    tool.get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| toolset.is_enabled(name))
                })
                .collect();
            let tool_policy = tool_policy_for_entry(entry, &self.config.connector_tool_policies);
            let admitted = admit_tools(
                self.config.admission_policy,
                tool_policy,
                &server.name,
                enabled,
                &mut admission,
            );
            generated.push(
                admitted
                    .iter()
                    .map(|tool| request::mcp_tool_to_custom_tool(&server.name, tool))
                    .collect(),
            );
            insert_tools(admitted, entry, &mut tool_map);
        }

        request::rewrite_tools(object, &connector.toolsets, generated)
            .map_err(|message| wire::invalid_request_rejection(&message))?;
        for name in request::tool_names(object) {
            admission.mark_available_name(name);
        }
        for entry in tool_map.values_mut() {
            bind_forwarded_header_context(entry, &self.config.forward_headers, &forwarded);
            bind_owner_context(entry, identity.as_ref().map(McpCalloutIdentity::owner));
            bind_credential_context(entry, identity.as_ref().and_then(McpCalloutIdentity::user_credential));
        }
        Ok(ConnectorState {
            tool_map,
            admission,
            has_connector,
            earlier: Vec::new(),
            usage: RoundUsage::default(),
            pending: false,
        })
    }

    /// Run `tools/list` for one server, or skip a protected-mode exclusion.
    #[expect(
        clippy::too_many_lines,
        reason = "exclusion, identity projection, and listing share one call"
    )]
    async fn list_server_tools(&self, server: &ServerDecl, entry: &Value, listing: &ListingContext<'_>) -> Listing {
        let ListingContext {
            forwarded,
            identity,
            callout,
        } = *listing;
        if let Some(reason) = protected_exclusion(
            entry,
            self.config.admission_policy,
            &self.config.connector_tool_policies,
        ) {
            debug!(?reason, "protected mode: excluding MCP server without dialing");
            return Listing::Excluded(reason);
        }
        let Some(server_url) = entry.get("server_url").and_then(Value::as_str) else {
            return Listing::Failed;
        };
        let is_connector = is_connector_tool_entry(entry);
        let context = is_connector
            .then_some(identity)
            .flatten()
            .map(|identity| mcp_client::McpConnectorContext {
                owner: identity.owner(),
                bearer: identity.user_credential(),
                assertion: identity.authorization(),
            });
        let result = mcp_client::list_tools_with_forwarded_headers(
            server_url,
            None,
            server.authorization_token.as_deref(),
            &self.config.forward_headers,
            is_connector.then_some(forwarded),
            context.as_ref(),
            self.config.timeout,
            self.config.max_tools,
            callout,
        )
        .await;
        match result {
            Ok(tools) => Listing::Tools(tools),
            Err(error) => {
                debug!(connector = is_connector, error = %error, "MCP tools/list failed");
                Listing::Failed
            },
        }
    }

    // -------------------------------------------------------------------------
    // Later Rounds
    // -------------------------------------------------------------------------

    /// Run the previous round's MCP calls and rebuild the next request.
    #[expect(
        clippy::too_many_lines,
        reason = "execution and request rebuild share one round transaction"
    )]
    async fn handle_reentry(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        if ctx.extensions.get::<ConnectorState>().is_none() {
            return Ok(FilterAction::Continue);
        }
        ctx.set_metadata(MCP_CONNECTOR_METADATA_KEY, MCP_CONNECTOR_RESOLVED);
        let Some(state) = ctx.extensions.get::<ConnectorState>().filter(|state| state.pending) else {
            return Ok(FilterAction::Continue);
        };
        let has_connector = state.has_connector;
        let Some(iteration) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable"
            )));
        };
        let previous: Value = iteration
            .previous_response
            .as_ref()
            .and_then(|previous| serde_json::from_slice(&previous.body).ok())
            .ok_or_else(|| FilterError::from(format!("{FILTER_NAME}: previous MCP round is unavailable")))?;
        let request_bytes = iteration
            .accumulator
            .get(REQUEST_ACCUMULATOR_KEY)
            .unwrap_or(&iteration.original_request.body);
        let Ok(Value::Object(mut request)) = serde_json::from_slice::<Value>(request_bytes) else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: retained request is not a JSON object"
            )));
        };

        let identity = match self.stage_identity(ctx, has_connector) {
            Ok(identity) => identity,
            Err(rejection) => return Ok(FilterAction::Reject(rejection)),
        };
        let callout = match self.callout(ctx) {
            Ok(callout) => callout,
            Err(rejection) => return Ok(FilterAction::Reject(rejection)),
        };
        let forwarded = self.forwarded_headers(ctx);
        let session_pool = ctx
            .extensions
            .get_or_insert_with(mcp_client::McpSessionPool::new)
            .clone();

        let Some(state) = ctx.extensions.get::<ConnectorState>() else {
            return Ok(FilterAction::Continue);
        };
        let index = McpToolIndex::new(&state.tool_map);
        let content = response::content_blocks(&previous);
        let calls: Vec<McpToolCall<'_>> = content
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .filter_map(|block| {
                let name = block.get("name").and_then(Value::as_str)?;
                response::is_mcp_owned(name, &index, &state.admission).then(|| {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("unknown");
                    McpToolCall {
                        encoded_name: Some(name),
                        call_id: id,
                        fallback_call_id: bounded_fallback_call_id(Some(id), None),
                        arguments: block.get("input"),
                        approval_request_id: None,
                    }
                })
            })
            .collect();
        debug!(count = calls.len(), "executing Anthropic MCP tool calls");
        let options = McpExecutionOptions {
            parallel: true,
            max_parallel_calls: self.config.max_parallel_calls,
            max_result_bytes: self.config.max_result_bytes,
            max_total_result_bytes: self.config.max_total_result_bytes,
            timeout: self.config.timeout,
            forwarded_header_names: &self.config.forward_headers,
            forwarded_headers: Some(&forwarded),
            connector_identity: identity.as_ref(),
            session_pool: &session_pool,
            pool_namespace: self.pool_namespace,
            admission: &state.admission,
            transport_budget: None,
        };
        let Ok(results) = Box::pin(execute_mcp_calls::<McpResult>(&calls, &index, options, &callout)).await else {
            return Ok(FilterAction::Reject(wire::error_rejection(
                502,
                wire::ErrorType::Api,
                "MCP tool results exceed a size limit set by the platform administrator.",
            )));
        };
        let model_results: Vec<Value> = results.iter().map(McpResult::model_block).collect();
        let client_blocks = response::client_content(content, &index, &state.admission, results);
        let mixed = response::has_client_tool_use(content, &index, &state.admission);
        drop(index);

        let Some(state) = ctx.extensions.get_mut::<ConnectorState>() else {
            return Ok(FilterAction::Continue);
        };
        state.pending = false;
        let Value::Object(mut previous) = previous else {
            return Ok(FilterAction::Continue);
        };
        if mixed {
            // Client tools in the same turn end the loop: return the MCP
            // results inline, as Anthropic does, and let the client answer
            // its own calls.
            let mut earlier = std::mem::take(&mut state.earlier);
            earlier.extend(client_blocks);
            previous.remove("content");
            let synthesized = response::final_response(previous, earlier, state.usage);
            return Ok(FilterAction::Reject(synthesized_response(&synthesized)));
        }
        state.usage.add(&previous);
        state.earlier.extend(client_blocks);

        let assistant_content = previous.remove("content").unwrap_or_else(|| Value::Array(Vec::new()));
        let Some(Value::Array(messages)) = request.get_mut("messages") else {
            return Ok(FilterAction::Reject(wire::invalid_request_rejection(
                "messages must be an array for MCP re-entry",
            )));
        };
        messages.push(json!({"role": "assistant", "content": assistant_content}));
        messages.push(json!({"role": "user", "content": model_results}));
        if request.contains_key("tool_choice") {
            request.insert("tool_choice".to_owned(), json!({"type": "auto"}));
        }
        let rebuilt = serde_json::to_vec(&request)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))?;
        if rebuilt.len() > self.config.max_body_bytes {
            return Ok(FilterAction::Reject(oversized_request()));
        }
        commit_request(ctx, body, Bytes::from(rebuilt), true);
        Ok(FilterAction::Continue)
    }

    // -------------------------------------------------------------------------
    // Response
    // -------------------------------------------------------------------------

    /// Classify a buffered round and finish or loop.
    #[expect(
        clippy::too_many_lines,
        reason = "size checks, classification, and final assembly share the round"
    )]
    fn on_buffered_round(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let observed = body.as_ref().map_or(0, Bytes::len);
        let raw = ctx
            .get_metadata(RESPONSE_RAW_BYTES_KEY)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(observed);
        if raw > self.config.max_body_bytes || observed > self.config.max_body_bytes {
            return Ok(FilterAction::Reject(wire::error_rejection(
                502,
                wire::ErrorType::Api,
                "MCP connector upstream response exceeds a size limit set by the platform administrator.",
            )));
        }
        let success = ctx
            .response_header
            .as_ref()
            .is_none_or(|response| response.status.is_success());
        let parsed = body
            .as_deref()
            .filter(|_| success)
            .and_then(|bytes| serde_json::from_slice::<Value>(bytes).ok());
        let Some(state) = ctx.extensions.get::<ConnectorState>() else {
            return Ok(FilterAction::Continue);
        };
        let calls_mcp = parsed.as_ref().is_some_and(|response| {
            response::round_calls_mcp(response, &McpToolIndex::new(&state.tool_map), &state.admission)
        });
        if calls_mcp {
            if !router_can_reenter(ctx) {
                return Ok(FilterAction::Reject(wire::error_rejection(
                    502,
                    wire::ErrorType::Api,
                    "the MCP tool loop reached a limit set by the platform administrator",
                )));
            }
            if let Some(state) = ctx.extensions.get_mut::<ConnectorState>() {
                state.pending = true;
            }
            set_action(ctx, ACTION_LOOP)?;
            return Ok(FilterAction::Continue);
        }
        set_action(ctx, ACTION_DONE)?;
        let Some(state) = ctx
            .extensions
            .get_mut::<ConnectorState>()
            .filter(|state| !state.earlier.is_empty())
        else {
            return Ok(FilterAction::Continue);
        };
        if let Some(Value::Object(response)) = parsed {
            let earlier = std::mem::take(&mut state.earlier);
            let rebuilt = response::final_response(response, earlier, state.usage);
            let bytes = serde_json::to_vec(&rebuilt)
                .map_err(|error| FilterError::from(format!("{FILTER_NAME}: response serialization failed: {error}")))?;
            *body = Some(Bytes::from(bytes));
        }
        Ok(FilterAction::Continue)
    }
}

/// Request-scoped inputs shared by every `tools/list` call.
#[derive(Clone, Copy)]
struct ListingContext<'a> {
    /// Operator-allowed headers forwarded to connectors.
    forwarded: &'a http::HeaderMap,
    /// Staged connector identity, when any server is a connector.
    identity: Option<&'a McpCalloutIdentity>,
    /// Outbound callout for the request.
    callout: &'a mcp_client::McpCallout,
}

/// Outcome of listing one server's tools.
enum Listing {
    /// The server's tools.
    Tools(Vec<Value>),
    /// Protected mode excluded the server without dialing it.
    Excluded(McpDenyReason),
    /// `tools/list` failed.
    Failed,
}

#[async_trait]
impl HttpFilter for AnthropicMcpConnectorFilter {
    fn name(&self) -> &'static str {
        "anthropic_mcp_connector"
    }

    fn visit_nested_pipelines(&mut self, visitor: &mut dyn FnMut(&mut FilterPipeline)) {
        if let Some(pipeline) = Arc::get_mut(&mut self.outbound) {
            visitor(pipeline);
        } else {
            debug_assert!(
                false,
                "{FILTER_NAME} outbound pipeline must be uniquely owned during configuration"
            );
        }
    }

    fn referenced_files(&self) -> Vec<std::path::PathBuf> {
        self.outbound.referenced_files()
    }

    fn apply_insecure_options(&self, options: &InsecureOptions) {
        self.outbound.apply_insecure_options(options);
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.config.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        // Advertise incremental delivery so the filter composes in a
        // streaming-capable step; connector requests select the buffered
        // subrequest transport per request.
        BodyMode::Stream
    }

    fn may_select_streaming_subrequest_response(&self) -> bool {
        // Requests without the connector keep the client's transport.
        true
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if ctx.extensions.get::<ConnectorState>().is_some() {
            // Every round must return identity-encoded JSON to be classified.
            ctx.request_headers_to_remove.push(ACCEPT_ENCODING);
        }
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
        let reentry = ctx
            .extensions
            .get::<IterationState>()
            .is_some_and(|state| state.previous_response.is_some());
        if reentry {
            return Box::pin(self.handle_reentry(ctx, body)).await;
        }
        Box::pin(self.handle_initial(ctx, body)).await
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        let rewrites_body = ctx
            .extensions
            .get::<ConnectorState>()
            .is_some_and(|state| !state.earlier.is_empty());
        if rewrites_body && let Some(response) = &mut ctx.response_header {
            response.headers.remove(http::header::CONTENT_LENGTH);
            response.headers.remove(http::header::CONTENT_ENCODING);
            response.headers.remove(http::header::ETAG);
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream
            || ctx.extensions.get::<ConnectorState>().is_none()
            || ctx.subrequest_response_mode() == SubRequestResponseMode::Streaming
        {
            return Ok(FilterAction::Continue);
        }
        self.on_buffered_round(ctx, body)
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Whether the body may carry the connector or replayed MCP blocks.
fn contains_mcp_marker(bytes: &[u8]) -> bool {
    const MARKER: &[u8] = b"mcp_";
    bytes.windows(MARKER.len()).any(|window| window == MARKER)
}

/// Select the subrequest transport from the client's `stream` flag.
fn apply_transport(ctx: &mut HttpFilterContext<'_>, streaming: bool) {
    ctx.set_subrequest_response_mode(if streaming {
        SubRequestResponseMode::Streaming
    } else {
        SubRequestResponseMode::Buffered
    });
}

/// Drop MCP connector tokens from the caller's `anthropic-beta` header.
fn strip_beta_header(ctx: &mut HttpFilterContext<'_>) {
    let Some(stripped) = ctx
        .request
        .headers
        .get(&ANTHROPIC_BETA_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(strip_mcp_client_betas)
    else {
        return;
    };
    if stripped.is_empty() {
        ctx.request_headers_to_remove.push(ANTHROPIC_BETA_HEADER);
    } else if let Ok(value) = HeaderValue::from_str(&stripped) {
        ctx.request_headers_to_set.push((ANTHROPIC_BETA_HEADER, value));
    }
}

/// Replace the request body and retain it for the next round.
fn commit_request(ctx: &mut HttpFilterContext<'_>, body: &mut Option<Bytes>, rebuilt: Bytes, next_iteration: bool) {
    // These `Bytes` clones share one allocation across the retained request,
    // the next iteration, and the active step body.
    if let Some(iteration) = ctx.extensions.get_mut::<IterationState>() {
        iteration
            .accumulator
            .insert(REQUEST_ACCUMULATOR_KEY.to_owned(), rebuilt.clone());
    }
    if next_iteration {
        ctx.extensions.insert(NextIterationBody(rebuilt.clone()));
    }
    ctx.request_headers_to_set
        .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
    *body = Some(rebuilt);
}

/// Reject a rebuilt request over `max_body_bytes`.
fn oversized_request() -> Rejection {
    wire::error_rejection(
        413,
        wire::ErrorType::InvalidRequest,
        "MCP connector request body exceeds a size limit set by the platform administrator.",
    )
}

/// Return a synthesized Messages response to the client.
fn synthesized_response(response: &Value) -> Rejection {
    let bytes = serde_json::to_vec(response).unwrap_or_default();
    Rejection::status(200)
        .with_header("content-type", "application/json")
        .with_body(Bytes::from(bytes))
}

/// Whether the router has iteration and deadline headroom for another round.
///
/// Without IRR state (a unit context) the loop proceeds.
fn router_can_reenter(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions.get::<IterationState>().is_none_or(|state| {
        state.iteration().saturating_add(1) < state.max_iterations()
            && !state
                .deadline()
                .checked_duration_since(Instant::now())
                .unwrap_or_default()
                .is_zero()
    })
}

/// Publish the loop decision for the IRR transition table.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_NAME)
        .or_default()
        .set("action", action)?;
    Ok(())
}
