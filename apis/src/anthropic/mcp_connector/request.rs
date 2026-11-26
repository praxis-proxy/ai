// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Request-side handling of the Anthropic MCP connector.
//!
//! Parses and validates `mcp_servers` plus `mcp_toolset` tools, maps each
//! server to an operator connector (or, when allowed, a direct client URL),
//! selects enabled tools, and rewrites the body into plain custom tools that
//! any Messages or Chat Completions backend accepts. Replayed `mcp_tool_use`
//! and `mcp_tool_result` history is converted into ordinary `tool_use` and
//! `tool_result` turns.
//!
//! Error messages name fields and positions but never echo client values.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};

use super::config::{ValidatedConfig, normalize_server_url};
use crate::mcp_core::naming::encode_function_name;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Tool type declaring an MCP toolset.
const MCP_TOOLSET_TYPE: &str = "mcp_toolset";

/// Assistant content block type for an MCP call.
pub(super) const MCP_TOOL_USE_TYPE: &str = "mcp_tool_use";

/// Assistant content block type for an MCP result.
pub(super) const MCP_TOOL_RESULT_TYPE: &str = "mcp_tool_result";

/// Longest accepted MCP server name.
const MAX_SERVER_NAME_LEN: usize = 128;

// -----------------------------------------------------------------------------
// Parsed Connector
// -----------------------------------------------------------------------------

/// One validated `mcp_servers` entry.
#[derive(Debug)]
pub(super) struct ServerDecl {
    /// Client-chosen server name, used as the MCP server label.
    pub name: String,
    /// Client-declared server URL.
    pub url: String,
    /// Optional bearer token for the server.
    pub authorization_token: Option<String>,
}

/// One validated `mcp_toolset` tool.
#[derive(Debug)]
pub(super) struct ToolsetDecl {
    /// Position of the toolset in the request `tools` array.
    pub tools_index: usize,
    /// Position of the referenced server in `servers`.
    pub server_index: usize,
    /// Whether tools without a per-tool entry are enabled.
    pub default_enabled: bool,
    /// Per-tool `enabled` overrides keyed by MCP tool name.
    pub configs: HashMap<String, bool>,
    /// Prompt-caching breakpoint carried onto the last generated tool.
    pub cache_control: Option<Value>,
}

impl ToolsetDecl {
    /// Whether the named MCP tool is enabled by this toolset.
    pub(super) fn is_enabled(&self, tool_name: &str) -> bool {
        self.configs.get(tool_name).copied().unwrap_or(self.default_enabled)
    }
}

/// A validated MCP connector declaration.
#[derive(Debug)]
pub(super) struct ConnectorRequest {
    /// Declared servers in request order.
    pub servers: Vec<ServerDecl>,
    /// Declared toolsets; exactly one per server.
    pub toolsets: Vec<ToolsetDecl>,
}

// -----------------------------------------------------------------------------
// Parsing and Validation
// -----------------------------------------------------------------------------

/// Parse and validate the connector declaration.
///
/// Returns `Ok(None)` when the body declares no server and no toolset.
///
/// # Errors
///
/// Returns a client-safe message for a malformed server or toolset, a toolset
/// naming an undeclared server, a server without exactly one toolset,
/// `defer_loading`, a pinned tool list, or too many servers.
pub(super) fn parse_connector(
    body: &Map<String, Value>,
    max_servers: usize,
) -> Result<Option<ConnectorRequest>, String> {
    let servers = parse_servers(body.get("mcp_servers"))?;
    if servers.len() > max_servers {
        return Err(format!(
            "mcp_servers declares {} servers; this gateway accepts at most {max_servers}",
            servers.len()
        ));
    }
    let toolsets = parse_toolsets(body.get("tools"), &servers)?;
    if servers.is_empty() && toolsets.is_empty() {
        return Ok(None);
    }
    let mut referenced = vec![false; servers.len()];
    for toolset in &toolsets {
        if let Some(seen) = referenced.get_mut(toolset.server_index) {
            *seen = true;
        }
    }
    if let Some(unused) = referenced.iter().position(|seen| !seen) {
        return Err(format!(
            "mcp_servers[{unused}] must be referenced by exactly one mcp_toolset tool"
        ));
    }
    Ok(Some(ConnectorRequest { servers, toolsets }))
}

/// Parse the `mcp_servers` array.
fn parse_servers(value: Option<&Value>) -> Result<Vec<ServerDecl>, String> {
    let entries = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(entries)) => entries,
        Some(_) => return Err("mcp_servers must be an array".to_owned()),
    };
    let mut names = HashSet::with_capacity(entries.len());
    let mut servers = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let server = parse_server(index, entry)?;
        if !names.insert(server.name.clone()) {
            return Err(format!("mcp_servers[{index}].name duplicates an earlier server name"));
        }
        servers.push(server);
    }
    Ok(servers)
}

/// Parse one `mcp_servers` entry.
#[expect(clippy::too_many_lines, reason = "one match arm per server field")]
fn parse_server(index: usize, entry: &Value) -> Result<ServerDecl, String> {
    let object = entry
        .as_object()
        .ok_or_else(|| format!("mcp_servers[{index}] must be an object"))?;
    let mut name = None;
    let mut url = None;
    let mut authorization_token = None;
    let mut server_type = None;
    for (key, value) in object {
        match key.as_str() {
            "type" => server_type = value.as_str(),
            "name" => name = Some(non_empty_string(value, || format!("mcp_servers[{index}].name"))?),
            "url" => url = Some(non_empty_string(value, || format!("mcp_servers[{index}].url"))?),
            "authorization_token" => {
                if !value.is_null() {
                    authorization_token = Some(non_empty_string(value, || {
                        format!("mcp_servers[{index}].authorization_token")
                    })?);
                }
            },
            "tool_configuration" => {
                return Err(format!(
                    "mcp_servers[{index}].tool_configuration belongs to the deprecated mcp-client-2025-04-04 beta; \
                     configure tools with an mcp_toolset tool"
                ));
            },
            _ => return Err(format!("mcp_servers[{index}] has an unsupported field")),
        }
    }
    if server_type != Some("url") {
        return Err(format!("mcp_servers[{index}].type must be \"url\""));
    }
    let name = name.ok_or_else(|| format!("mcp_servers[{index}].name is required"))?;
    if name.len() > MAX_SERVER_NAME_LEN {
        return Err(format!("mcp_servers[{index}].name exceeds {MAX_SERVER_NAME_LEN} bytes"));
    }
    let url = url.ok_or_else(|| format!("mcp_servers[{index}].url is required"))?;
    Ok(ServerDecl {
        name: name.to_owned(),
        url: url.to_owned(),
        authorization_token: authorization_token.map(str::to_owned),
    })
}

/// Parse every `mcp_toolset` entry of `tools`.
fn parse_toolsets(tools: Option<&Value>, servers: &[ServerDecl]) -> Result<Vec<ToolsetDecl>, String> {
    let Some(Value::Array(tools)) = tools else {
        return Ok(Vec::new());
    };
    let mut toolsets = Vec::new();
    let mut claimed = vec![false; servers.len()];
    for (tools_index, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some(MCP_TOOLSET_TYPE) {
            continue;
        }
        let toolset = parse_toolset(tools_index, tool, servers)?;
        let claim = claimed
            .get_mut(toolset.server_index)
            .ok_or_else(|| format!("tools[{tools_index}].mcp_server_name is invalid"))?;
        if *claim {
            return Err(format!(
                "tools[{tools_index}] references an MCP server that another mcp_toolset already uses"
            ));
        }
        *claim = true;
        toolsets.push(toolset);
    }
    Ok(toolsets)
}

/// Parse one `mcp_toolset` tool.
#[expect(clippy::too_many_lines, reason = "one match arm per toolset field")]
fn parse_toolset(tools_index: usize, tool: &Value, servers: &[ServerDecl]) -> Result<ToolsetDecl, String> {
    let object = tool
        .as_object()
        .ok_or_else(|| format!("tools[{tools_index}] must be an object"))?;
    let mut server_name = None;
    let mut default_enabled = true;
    let mut configs = HashMap::new();
    let mut cache_control = None;
    for (key, value) in object {
        match key.as_str() {
            "type" => {},
            "mcp_server_name" => {
                server_name = Some(non_empty_string(value, || {
                    format!("tools[{tools_index}].mcp_server_name")
                })?);
            },
            "default_config" => {
                if let Some(enabled) = parse_tool_config(value, || format!("tools[{tools_index}].default_config"))? {
                    default_enabled = enabled;
                }
            },
            "configs" => configs = parse_tool_configs(tools_index, value)?,
            "cache_control" => {
                if !value.is_null() {
                    cache_control = Some(value.clone());
                }
            },
            "tools" => {
                return Err(format!(
                    "tools[{tools_index}].tools (a pinned MCP tool list) is not supported by this gateway"
                ));
            },
            _ => return Err(format!("tools[{tools_index}] has an unsupported mcp_toolset field")),
        }
    }
    let server_name = server_name.ok_or_else(|| format!("tools[{tools_index}].mcp_server_name is required"))?;
    let server_index = servers
        .iter()
        .position(|server| server.name == server_name)
        .ok_or_else(|| format!("tools[{tools_index}].mcp_server_name does not match any mcp_servers entry"))?;
    Ok(ToolsetDecl {
        tools_index,
        server_index,
        default_enabled,
        configs,
        cache_control,
    })
}

/// Parse the `configs` map of one toolset.
fn parse_tool_configs(tools_index: usize, value: &Value) -> Result<HashMap<String, bool>, String> {
    let entries = match value {
        Value::Null => return Ok(HashMap::new()),
        Value::Object(entries) => entries,
        _ => return Err(format!("tools[{tools_index}].configs must be an object")),
    };
    let mut configs = HashMap::with_capacity(entries.len());
    for (tool_name, config) in entries {
        if let Some(enabled) = parse_tool_config(config, || format!("tools[{tools_index}].configs entry"))? {
            configs.insert(tool_name.clone(), enabled);
        }
    }
    Ok(configs)
}

/// Parse one tool configuration object, returning its `enabled` value.
///
/// `defer_loading: true` is refused: the gateway has no tool search to load
/// deferred tools later, so honoring it would silently hide them.
fn parse_tool_config(value: &Value, path: impl Fn() -> String) -> Result<Option<bool>, String> {
    let config = match value {
        Value::Null => return Ok(None),
        Value::Object(config) => config,
        _ => return Err(format!("{} must be an object", path())),
    };
    let mut enabled = None;
    for (key, field) in config {
        match key.as_str() {
            "enabled" => match field {
                Value::Bool(value) => enabled = Some(*value),
                Value::Null => {},
                _ => return Err(format!("{}.enabled must be a boolean", path())),
            },
            "defer_loading" => match field {
                Value::Bool(false) | Value::Null => {},
                Value::Bool(true) => {
                    return Err(format!("{}.defer_loading is not supported by this gateway", path()));
                },
                _ => return Err(format!("{}.defer_loading must be a boolean", path())),
            },
            _ => return Err(format!("{} has an unsupported field", path())),
        }
    }
    Ok(enabled)
}

/// Borrow a non-empty JSON string or describe the field error.
fn non_empty_string(value: &Value, path: impl Fn() -> String) -> Result<&str, String> {
    match value.as_str() {
        Some(text) if !text.is_empty() => Ok(text),
        _ => Err(format!("{} must be a non-empty string", path())),
    }
}

// -----------------------------------------------------------------------------
// Server Gating
// -----------------------------------------------------------------------------

/// Build the tool-map entry for one declared server.
///
/// A server whose URL matches an operator connector resolves to that
/// connector; its tools run with operator identity and tool policy. Any other
/// URL is rejected unless `allow_client_servers` is set, in which case it is
/// dialed directly as an untrusted, client-selected target.
///
/// # Errors
///
/// Returns a client-safe message when the server is not allowed.
pub(super) fn server_entry(index: usize, server: &ServerDecl, config: &ValidatedConfig) -> Result<Value, String> {
    let mut entry = Map::new();
    entry.insert("server_label".to_owned(), Value::String(server.name.clone()));
    let connector = normalize_server_url(&server.url).and_then(|key| config.connectors.get(&key));
    match connector {
        Some(connector) => {
            entry.insert("server_url".to_owned(), Value::String(connector.server_url.clone()));
            entry.insert("connector_id".to_owned(), Value::String(connector.id.clone()));
        },
        None if config.allow_client_servers => {
            entry.insert("server_url".to_owned(), Value::String(server.url.clone()));
        },
        None => {
            return Err(format!(
                "mcp_servers[{index}].url is not an MCP server configured on this gateway"
            ));
        },
    }
    if let Some(token) = &server.authorization_token {
        entry.insert("authorization".to_owned(), Value::String(token.clone()));
    }
    Ok(Value::Object(entry))
}

// -----------------------------------------------------------------------------
// Tool Rewriting
// -----------------------------------------------------------------------------

/// Convert one MCP tool definition into an Anthropic custom tool.
pub(super) fn mcp_tool_to_custom_tool(server_name: &str, definition: &Value) -> Value {
    let tool_name = definition.get("name").and_then(Value::as_str).unwrap_or("unknown");
    let mut tool = Map::new();
    tool.insert(
        "name".to_owned(),
        Value::String(encode_function_name(server_name, tool_name)),
    );
    if let Some(description) = definition.get("description").filter(|value| value.is_string()) {
        tool.insert("description".to_owned(), description.clone());
    }
    let input_schema = definition
        .get("inputSchema")
        .or_else(|| definition.get("input_schema"))
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object"}));
    tool.insert("input_schema".to_owned(), input_schema);
    Value::Object(tool)
}

/// Replace `mcp_toolset` entries with generated tools and drop `mcp_servers`.
///
/// `generated` holds, per toolset in `toolsets` order, the custom tools built
/// from its enabled MCP tools. A toolset's `cache_control` moves onto its last
/// generated tool so the client's caching breakpoint keeps its position. An
/// empty resulting `tools` array is removed.
///
/// # Errors
///
/// Returns a client-safe message when a generated tool name collides with
/// another generated tool or a client tool.
#[expect(clippy::too_many_lines, reason = "order-preserving rewrite with collision checks")]
pub(super) fn rewrite_tools(
    body: &mut Map<String, Value>,
    toolsets: &[ToolsetDecl],
    generated: Vec<Vec<Value>>,
) -> Result<Vec<String>, String> {
    // `shift_remove` and in-place replacement keep every other member's
    // position; `Map::remove` would swap the last member into the gap.
    body.shift_remove("mcp_servers");
    let original = match body.get_mut("tools") {
        Some(Value::Array(tools)) => std::mem::take(tools),
        _ => Vec::new(),
    };
    let mut by_index: HashMap<usize, (Vec<Value>, Option<&Value>)> = toolsets
        .iter()
        .zip(generated)
        .map(|(toolset, tools)| (toolset.tools_index, (tools, toolset.cache_control.as_ref())))
        .collect();
    let mut rewritten = Vec::with_capacity(original.len());
    let mut names = HashSet::new();
    let mut generated_names = Vec::new();
    for (index, tool) in original.into_iter().enumerate() {
        let Some((mut tools, cache_control)) = by_index.remove(&index) else {
            if let Some(name) = tool.get("name").and_then(Value::as_str)
                && !names.insert(name.to_owned())
            {
                return Err(format!("tools[{index}].name collides with another tool name"));
            }
            rewritten.push(tool);
            continue;
        };
        if let (Some(cache_control), Some(Value::Object(last))) = (cache_control, tools.last_mut()) {
            last.insert("cache_control".to_owned(), cache_control.clone());
        }
        for generated in tools {
            if let Some(name) = generated.get("name").and_then(Value::as_str) {
                if !names.insert(name.to_owned()) {
                    return Err(format!(
                        "tools[{index}] exposes an MCP tool whose name collides with another tool name"
                    ));
                }
                generated_names.push(name.to_owned());
            }
            rewritten.push(generated);
        }
    }
    if rewritten.is_empty() {
        body.shift_remove("tools");
    } else if let Some(tools) = body.get_mut("tools") {
        *tools = Value::Array(rewritten);
    }
    Ok(generated_names)
}

/// Every model-visible tool name in a rewritten body.
pub(super) fn tool_names(body: &Map<String, Value>) -> impl Iterator<Item = &str> {
    body.get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
}

// -----------------------------------------------------------------------------
// History Replay
// -----------------------------------------------------------------------------

/// Whether any message carries a replayed MCP block.
pub(super) fn has_mcp_history(body: &Map<String, Value>) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content").and_then(Value::as_array))
        .flatten()
        .any(|block| {
            matches!(
                block.get("type").and_then(Value::as_str),
                Some(MCP_TOOL_USE_TYPE | MCP_TOOL_RESULT_TYPE)
            )
        })
}

/// Convert replayed `mcp_tool_use` / `mcp_tool_result` blocks into ordinary
/// tool turns.
///
/// Anthropic returns both blocks inside one assistant message. Backends only
/// understand a `tool_use` in an assistant turn answered by a `tool_result` in
/// the next user turn, so each assistant message is split at its MCP results:
///
/// ```text
/// assistant [text, mcp_tool_use A, mcp_tool_result A, text2]
///   -> assistant [text, tool_use A]
///      user      [tool_result A]
///      assistant [text2]
/// ```
///
/// Results that end an assistant message are prepended to the following user
/// message, so roles keep alternating.
///
/// # Errors
///
/// Returns a client-safe message for a malformed MCP block or a result that
/// answers no earlier call in the same message.
pub(super) fn convert_mcp_history(body: &mut Map<String, Value>) -> Result<(), String> {
    let Some(Value::Array(messages)) = body.get_mut("messages") else {
        return Ok(());
    };
    let original = std::mem::take(messages);
    let mut converted = Vec::with_capacity(original.len());
    let mut carried_results: Vec<Value> = Vec::new();
    for (index, message) in original.into_iter().enumerate() {
        let message = prepend_carried_results(message, &mut carried_results, &mut converted);
        if !is_assistant_with_mcp(&message) {
            converted.push(message);
            continue;
        }
        let trailing = split_assistant_message(index, message, &mut converted)?;
        carried_results = trailing;
    }
    if !carried_results.is_empty() {
        converted.push(role_message("user", carried_results));
    }
    *messages = converted;
    Ok(())
}

/// Attach results carried from the previous assistant message to `message`.
///
/// A following user message receives them in front of its own content. Any
/// other message gets a synthetic user turn inserted before it.
fn prepend_carried_results(message: Value, carried: &mut Vec<Value>, converted: &mut Vec<Value>) -> Value {
    if carried.is_empty() {
        return message;
    }
    let results = std::mem::take(carried);
    match message {
        Value::Object(mut object) if object.get("role").and_then(Value::as_str) == Some("user") => {
            let mut content = results;
            let slot = object.entry("content").or_insert(Value::Null);
            match std::mem::take(slot) {
                Value::Array(blocks) => content.extend(blocks),
                Value::String(text) => content.push(serde_json::json!({"type": "text", "text": text})),
                Value::Null => {},
                other => content.push(other),
            }
            *slot = Value::Array(content);
            Value::Object(object)
        },
        other => {
            converted.push(role_message("user", results));
            other
        },
    }
}

/// Whether `message` is an assistant turn carrying MCP blocks.
fn is_assistant_with_mcp(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("assistant")
        && message.get("content").and_then(Value::as_array).is_some_and(|blocks| {
            blocks.iter().any(|block| {
                matches!(
                    block.get("type").and_then(Value::as_str),
                    Some(MCP_TOOL_USE_TYPE | MCP_TOOL_RESULT_TYPE)
                )
            })
        })
}

/// Split one assistant message at its MCP results, pushing completed turns
/// and returning results that end the message.
#[expect(clippy::too_many_lines, reason = "one pass over the assistant blocks")]
fn split_assistant_message(index: usize, message: Value, converted: &mut Vec<Value>) -> Result<Vec<Value>, String> {
    let Value::Object(mut object) = message else {
        return Ok(Vec::new());
    };
    // Take the blocks but keep the `content` slot, so every rebuilt turn
    // keeps the message's original member order.
    let Some(Value::Array(blocks)) = object.get_mut("content").map(Value::take) else {
        return Ok(Vec::new());
    };
    let mut assistant: Vec<Value> = Vec::new();
    let mut results: Vec<Value> = Vec::new();
    let mut open_calls: HashSet<String> = HashSet::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some(MCP_TOOL_USE_TYPE) => {
                flush_turns(&object, &mut assistant, &mut results, converted);
                let (id, tool_use) = mcp_tool_use_to_tool_use(index, &block)?;
                open_calls.insert(id);
                assistant.push(tool_use);
            },
            Some(MCP_TOOL_RESULT_TYPE) => {
                let (id, tool_result) = mcp_tool_result_to_tool_result(index, block)?;
                if !open_calls.remove(&id) {
                    return Err(format!(
                        "messages[{index}] has an mcp_tool_result that answers no earlier mcp_tool_use"
                    ));
                }
                results.push(tool_result);
            },
            _ => {
                flush_turns(&object, &mut assistant, &mut results, converted);
                assistant.push(block);
            },
        }
    }
    if !assistant.is_empty() {
        converted.push(assistant_message(&object, assistant));
    }
    Ok(results)
}

/// Emit the pending assistant turn and its results once results exist.
fn flush_turns(
    template: &Map<String, Value>,
    assistant: &mut Vec<Value>,
    results: &mut Vec<Value>,
    converted: &mut Vec<Value>,
) {
    if results.is_empty() {
        return;
    }
    converted.push(assistant_message(template, std::mem::take(assistant)));
    converted.push(role_message("user", std::mem::take(results)));
}

/// Rebuild an assistant message with `content`, keeping its other fields.
fn assistant_message(template: &Map<String, Value>, content: Vec<Value>) -> Value {
    let mut object = template.clone();
    object.insert("content".to_owned(), Value::Array(content));
    Value::Object(object)
}

/// Move the assistant call and corresponding results into the next model request.
pub(super) fn rebuild_after_calls(
    mut request: Map<String, Value>,
    previous: &mut Map<String, Value>,
    results: Vec<Value>,
    limit: usize,
) -> Result<Vec<u8>, praxis_filter::Rejection> {
    let Some(Value::Array(messages)) = request.get_mut("messages") else {
        return Err(super::wire::invalid_request_rejection(
            "messages must be an array for MCP re-entry",
        ));
    };
    let Some(Value::Array(content)) = previous.remove("content") else {
        return Err(super::wire::error_rejection(
            502,
            super::wire::ErrorType::Api,
            "MCP inference round lacks assistant content",
        ));
    };
    messages.push(role_message("assistant", content));
    messages.push(role_message("user", results));
    if let Some(choice) = request.get_mut("tool_choice") {
        *choice = serde_json::json!({"type": "auto"});
    }
    praxis_filter::json_size::to_vec_capped(&request, limit).map_err(|_error| super::oversized_request())
}

/// Build a `{role, content}` message.
fn role_message(role: &str, content: Vec<Value>) -> Value {
    let mut object = Map::with_capacity(2);
    object.insert("role".to_owned(), Value::String(role.to_owned()));
    object.insert("content".to_owned(), Value::Array(content));
    Value::Object(object)
}

/// Convert a replayed `mcp_tool_use` into a `tool_use` with the encoded name.
fn mcp_tool_use_to_tool_use(index: usize, block: &Value) -> Result<(String, Value), String> {
    let field = |name: &str| {
        block
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("messages[{index}] has an mcp_tool_use without a valid {name}"))
    };
    let id = field("id")?;
    let name = field("name")?;
    let server_name = field("server_name")?;
    let input = block
        .get("input")
        .filter(|input| input.is_object())
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut tool_use = Map::with_capacity(4);
    tool_use.insert("type".to_owned(), Value::String("tool_use".to_owned()));
    tool_use.insert("id".to_owned(), Value::String(id.to_owned()));
    tool_use.insert(
        "name".to_owned(),
        Value::String(encode_function_name(server_name, name)),
    );
    tool_use.insert("input".to_owned(), input);
    Ok((id.to_owned(), Value::Object(tool_use)))
}

/// Convert a replayed `mcp_tool_result` into a `tool_result`.
fn mcp_tool_result_to_tool_result(index: usize, block: Value) -> Result<(String, Value), String> {
    let Value::Object(mut object) = block else {
        return Err(format!("messages[{index}] has a malformed mcp_tool_result"));
    };
    let id = match object.remove("tool_use_id") {
        Some(Value::String(id)) if !id.is_empty() => id,
        _ => {
            return Err(format!(
                "messages[{index}] has an mcp_tool_result without a valid tool_use_id"
            ));
        },
    };
    let mut tool_result = Map::with_capacity(4);
    tool_result.insert("type".to_owned(), Value::String("tool_result".to_owned()));
    tool_result.insert("tool_use_id".to_owned(), Value::String(id.clone()));
    if let Some(content) = object.remove("content").filter(|content| !content.is_null()) {
        tool_result.insert("content".to_owned(), content);
    }
    if object.get("is_error") == Some(&Value::Bool(true)) {
        tool_result.insert("is_error".to_owned(), Value::Bool(true));
    }
    Ok((id, Value::Object(tool_result)))
}
